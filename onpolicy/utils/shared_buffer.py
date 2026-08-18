import torch
import numpy as np
import torch.nn.functional as F
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

def _graph_cast(x):
    return x.reshape(-1, *x.shape[2:])

class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.data_chunk_length = max(1, int(getattr(args, 'data_chunk_length', args.episode_length)))
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.num_agents = num_agents

        # obs_shape = get_shape_from_obs_space(obs_space)
        # share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        # if type(obs_shape[-1]) == list:
        #     obs_shape = obs_shape[:1]

        # if type(share_obs_shape[-1]) == list:
        #     share_obs_shape = share_obs_shape[:1]

        # self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1, *share_obs_shape),
        #                           dtype=np.float32)
        # self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        self.graph_obs =  [[None for _ in range(self.n_rollout_threads)] for _ in range(self.episode_length + 1)]

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        # if act_space.__class__.__name__ == 'Discrete':
        #     self.available_actions = np.zeros((self.episode_length + 1, self.n_rollout_threads, act_space.n),
        #                                      dtype=np.float32)
        # else:
        #     self.available_actions = None

        # act_shape = get_shape_from_act_space(act_space)

        self.actions = -np.ones(
            (self.episode_length, self.n_rollout_threads, num_agents, 3), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.action_dists = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 2), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        # Populated after rollout completion. In case-balanced mode every
        # environment/case contributes total sample mass 1.
        self.policy_sample_weights = np.zeros_like(self.rewards)
        self.value_sample_weights = np.zeros_like(self.rewards)
        self.team_returns = np.zeros((self.n_rollout_threads,), dtype=np.float32)
        self.team_cmax_values = np.zeros(
            (self.n_rollout_threads,), dtype=np.float32
        )
        self.decision_times = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads), dtype=np.float32
        )
        self.potential_values = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads), dtype=np.float32
        )
        self.last_team_time_potential_diagnostics = {}

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.zeros_like(self.masks)
        # active_masks describe environment activity; policy_masks exclude
        # forced one-choice actions such as post-contention device no-op.
        self.policy_masks = np.zeros_like(self.masks)
        self.agent_types = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents),
            dtype=np.int64,
        )

        self.step = 0
        self.filled_steps = 0

    def reset_rollout(self):
        self.step = 0
        self.filled_steps = 0
        self.graph_obs = [[None for _ in range(self.n_rollout_threads)] for _ in range(self.episode_length + 1)]
        self.rnn_states.fill(0.0)
        self.rnn_states_critic.fill(0.0)
        self.value_preds.fill(0.0)
        self.returns.fill(0.0)
        self.advantages.fill(0.0)
        self.actions.fill(-1.0)
        self.action_log_probs.fill(0.0)
        self.action_dists.fill(0.0)
        self.rewards.fill(0.0)
        self.decision_times.fill(0.0)
        self.potential_values.fill(0.0)
        self.last_team_time_potential_diagnostics = {}
        self.policy_sample_weights.fill(0.0)
        self.value_sample_weights.fill(0.0)
        self.team_returns.fill(0.0)
        self.team_cmax_values.fill(0.0)
        self.masks.fill(1.0)
        self.bad_masks.fill(1.0)
        self.active_masks.fill(0.0)
        self.policy_masks.fill(0.0)
        self.agent_types.fill(0)

    def graph_insert(self, obs, rnn_states, actions, action_log_probs,
                   value_preds, rewards, masks, active_masks, policy_masks=None,
                   decision_times=None, potential_values=None):
        """
        Insert data into the buffer. This insert function is used specifically for PyG graph data observations.
        :param obs: (list of HeteroData) local agent observations, length equals n_rollout_threads.
        :param rnn_states: (np.ndarray) RNN states for actor network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        """
        
        if self.step >= self.episode_length:
            raise RuntimeError(
                f"Replay buffer overflow: collected more than {self.episode_length} rollout steps. "
                "Increase --rollout_max_steps."
            )

        # ========================================================
        # 1. 异构图数据存入逻辑 (遍历列表存入，绝对对齐线程索引)
        # ========================================================
        for thread_idx in range(self.n_rollout_threads):
            self.graph_obs[self.step + 1][thread_idx] = obs[thread_idx].clone()
            graph_agent_types = getattr(obs[thread_idx], 'agent_types', None)
            if graph_agent_types is not None:
                if torch.is_tensor(graph_agent_types):
                    graph_agent_types = graph_agent_types.detach().cpu().numpy()
                self.agent_types[self.step + 1, thread_idx] = np.asarray(
                    graph_agent_types,
                    dtype=np.int64,
                ).reshape(self.num_agents)

        # ========================================================
        # 2. 常规 Numpy 张量存入逻辑
        # ========================================================
        self.rnn_states[self.step + 1] = rnn_states.copy()
        actions = np.asarray(actions)
        if actions.shape[-1] not in (2, 3):
            raise ValueError(
                "HKBZ actions must contain [operation, site] and optional "
                f"plane-order rank; got shape {actions.shape}."
            )
        stored_actions = -np.ones(
            (self.n_rollout_threads, self.num_agents, 3), dtype=np.float32
        )
        stored_actions[..., :actions.shape[-1]] = actions
        self.actions[self.step] = stored_actions
        
        self.action_log_probs[self.step] = action_log_probs.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        self.value_preds[self.step] = value_preds.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        # self.rewards[self.step] = rewards.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        
        self.masks[self.step + 1] = masks.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        self.active_masks[self.step + 1] = active_masks.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        if policy_masks is not None:
            self.policy_masks[self.step] = policy_masks.reshape(
                self.n_rollout_threads,
                self.num_agents,
                1,
            ).copy()
        if decision_times is not None:
            decision_times = np.asarray(
                decision_times, dtype=np.float32
            ).reshape(self.n_rollout_threads)
            if not np.isfinite(decision_times).all():
                raise ValueError('decision_times contains NaN or Inf.')
            previous_times = self.decision_times[self.step]
            if np.any(decision_times + 1e-6 < previous_times):
                raise ValueError('Environment time moved backwards in rollout.')
            self.decision_times[self.step + 1] = decision_times
        if potential_values is not None:
            potential_values = np.asarray(
                potential_values, dtype=np.float32
            ).reshape(self.n_rollout_threads)
            if not np.isfinite(potential_values).all():
                raise ValueError('potential_values contains NaN or Inf.')
            self.potential_values[self.step + 1] = potential_values

        self.step += 1
        self.filled_steps = max(self.filled_steps, self.step)

    def graph_after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        向量化版本 SMDP GAE (Vectorized Sequential GAE without Done Masks).
        支持 PopArt / ValueNorm。
        """
        # T: collected rollout length, N: Threads, M: Agents
        T = int(getattr(self, 'filled_steps', self.episode_length))
        if T <= 0:
            raise RuntimeError("Cannot compute returns for an empty rollout.")
        _, N, M, _ = self.rewards.shape
        self.rewards[:T] = np.nan_to_num(self.rewards[:T], nan=0.0, posinf=1e4, neginf=-1e4)
        self.value_preds[:T] = np.nan_to_num(self.value_preds[:T], nan=0.0, posinf=1e4, neginf=-1e4)
        next_value = np.nan_to_num(next_value, nan=0.0, posinf=1e4, neginf=-1e4)
        
        use_v_norm = (getattr(self, '_use_popart', False) or getattr(self, '_use_valuenorm', False)) and (value_normalizer is not None)
        
        if self._use_gae:
            # --- 1. 一次性反归一化 (保留原维度 T, N, M, 1) ---
            if use_v_norm:
                denorm_values = value_normalizer.denormalize(self.value_preds[:T])
                denorm_next_value = value_normalizer.denormalize(next_value)
            else:
                denorm_values = self.value_preds[:T]
                denorm_next_value = next_value
            denorm_values = np.nan_to_num(denorm_values, nan=0.0, posinf=1e4, neginf=-1e4)
            denorm_next_value = np.nan_to_num(denorm_next_value, nan=0.0, posinf=1e4, neginf=-1e4)
                
            # 【修复点 1】：绝对不能用 self.returns = xxx 覆盖原数组！
            # 必须把真实尺度的 next_value 存在第 T+1 步 ([-1] 哨兵位)
            self.returns[T] = denorm_next_value
            
            # gae 现在的形状是 (N, M, 1)，每个环境的每架飞机都有独立计算的优势
            gae = np.zeros((N, M, 1), dtype=np.float32)
            
            # next_active_value 的形状也是 (N, M, 1)
            next_active_value = denorm_next_value * self.masks[T]

            # --- 2. 仅在时间维度 T 上反向迭代 ---
            for step in reversed(range(T)):
                is_decision_step = self.active_masks[step]
                next_nonterminal = self.masks[step + 1]
                masked_next_active_value = next_active_value * next_nonterminal
                
                # --- A. 计算 Delta ---
                delta = self.rewards[step] + self.gamma * masked_next_active_value - denorm_values[step]
                delta = np.nan_to_num(delta, nan=0.0, posinf=1e4, neginf=-1e4)
                
                # --- B. 更新 GAE ---
                gae_update = delta + self.gamma * self.gae_lambda * gae * next_nonterminal
                gae_update = np.nan_to_num(gae_update, nan=0.0, posinf=1e4, neginf=-1e4)
                gae = is_decision_step * gae_update + (1.0 - is_decision_step) * gae
                gae = np.nan_to_num(gae, nan=0.0, posinf=1e4, neginf=-1e4)
                
                # --- C. 计算 Returns ---
                # 【修复点 2】：安全地原地赋值给已有的 returns 数组
                self.returns[step] = np.nan_to_num(gae + denorm_values[step], nan=0.0, posinf=1e4, neginf=-1e4)
                
                # --- D. 更新 Bootstrap 指针 ---
                next_active_value = is_decision_step * denorm_values[step] + (1.0 - is_decision_step) * masked_next_active_value

            # 哨兵位更新保持归一化原值
            self.value_preds[T] = next_value

        else:
            # 不使用 GAE 的情况 (也做了向量化对齐)
            if use_v_norm:
                self.returns[T] = value_normalizer.denormalize(next_value)
            else:
                self.returns[T] = next_value
            self.returns[T] = np.nan_to_num(self.returns[T], nan=0.0, posinf=1e4, neginf=-1e4)
                
            for step in reversed(range(T)):
                self.returns[step] = np.nan_to_num(
                    self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step],
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )

    def compute_team_returns(self, team_returns, next_value):
        """Assign one undiscounted Monte Carlo team return to every decision.

        HKBZ scheduling optimizes an episode-level C_max objective with
        ``gamma=1``. Broadcasting a single case return to active decisions
        avoids per-agent terminal duplication and decision-count-dependent
        GAE attenuation of early scheduling choices.
        """
        T = int(getattr(self, 'filled_steps', self.episode_length))
        if T <= 0:
            raise RuntimeError("Cannot compute team returns for an empty rollout.")

        team_returns = np.asarray(team_returns, dtype=np.float32).reshape(-1)
        expected_shape = (self.n_rollout_threads,)
        if team_returns.shape != expected_shape:
            raise ValueError(
                "team_returns must contain one scalar per rollout thread: "
                f"expected {expected_shape}, got {team_returns.shape}."
            )
        if not np.isfinite(team_returns).all():
            raise ValueError("team_returns contains NaN or Inf.")

        self.team_returns[:] = team_returns
        self.returns.fill(0.0)
        self.rewards.fill(0.0)
        for env_idx, team_return in enumerate(team_returns):
            active = self.active_masks[:T, env_idx, :, 0] > 0.0
            self.returns[:T, env_idx, :, 0][active] = team_return

            # Logging-only decomposition: trainable decision rewards sum to
            # exactly one case return. PPO targets use ``returns`` above.
            trainable = (
                (self.policy_masks[:T, env_idx, :, 0] > 0.0)
                & active
            )
            decision_count = int(trainable.sum())
            if decision_count > 0:
                self.rewards[:T, env_idx, :, 0][trainable] = (
                    team_return / float(decision_count)
                )

        self.value_preds[T] = np.nan_to_num(
            next_value,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

    def set_team_cmax_values(self, cmax_values):
        """Store one finite terminal Cmax per case for tail-credit weighting."""
        values = np.asarray(cmax_values, dtype=np.float32).reshape(-1)
        expected = (self.n_rollout_threads,)
        if values.shape != expected:
            raise ValueError(
                f'cmax_values must have shape {expected}, got {values.shape}.'
            )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError('cmax_values must be finite and positive.')
        self.team_cmax_values[:] = values

    def compute_team_time_returns(
        self,
        team_returns,
        cmax_values,
        cmax_coef,
        next_value,
    ):
        """Assign global negative remaining makespan at each decision.

        ``team_returns`` contains the terminal objective (including any cycle
        penalty). Adding ``cmax_coef * time(s)`` yields
        ``-cmax_coef * (Cmax-time(s))`` while preserving the terminal penalty.
        All simultaneous agents receive the same target; joint PPO consumes it
        once rather than duplicating it per plane.
        """
        T = int(getattr(self, 'filled_steps', self.episode_length))
        if T <= 0:
            raise RuntimeError('Cannot compute team-time returns for an empty rollout.')
        team_returns = np.asarray(team_returns, dtype=np.float32).reshape(-1)
        cmax_values = np.asarray(cmax_values, dtype=np.float32).reshape(-1)
        expected = (self.n_rollout_threads,)
        if team_returns.shape != expected or cmax_values.shape != expected:
            raise ValueError(
                f'team_returns and cmax_values must have shape {expected}.'
            )
        if not np.isfinite(team_returns).all() or not np.isfinite(cmax_values).all():
            raise ValueError('Team-time objective contains NaN or Inf.')
        if np.any(self.decision_times[:T] > cmax_values[None, :] + 1e-4):
            raise ValueError('A recorded decision time exceeds final Cmax.')

        self.team_returns[:] = team_returns
        self.returns.fill(0.0)
        self.rewards.fill(0.0)
        coef = float(cmax_coef)
        for env_idx, terminal_return in enumerate(team_returns):
            state_returns = terminal_return + (
                coef * self.decision_times[:T, env_idx]
            )
            active = self.active_masks[:T, env_idx, :, 0] > 0.0
            self.returns[:T, env_idx, :, 0] = np.where(
                active, state_returns[:, None], 0.0
            )
            trainable = active & (
                self.policy_masks[:T, env_idx, :, 0] > 0.0
            )
            for step in range(T):
                count = int(trainable[step].sum())
                if count > 0:
                    elapsed = (
                        self.decision_times[step + 1, env_idx]
                        - self.decision_times[step, env_idx]
                    )
                    self.rewards[step, env_idx, :, 0][trainable[step]] = (
                        -coef * elapsed / float(count)
                    )
        self.value_preds[T] = np.nan_to_num(
            next_value, nan=0.0, posinf=1e4, neginf=-1e4
        )

    def compute_team_time_potential_returns(
        self,
        team_returns,
        cmax_values,
        cmax_coef,
        potential_coef,
        next_value,
    ):
        """Combine exact team-time targets with a telescoping state potential.

        The environment stores ``Phi(s) <= 0`` as the negative calibrated
        time-to-go cost.  With gamma=1 the shaped state target is
        ``terminal + cmax_coef*time - potential_coef*Phi(s)`` and each logged
        transition receives ``potential_coef*(Phi(next)-Phi(current))``.
        The absorbing terminal potential is defined as zero, so the sum is a
        policy-invariant constant determined by the initial case state.
        """
        T = int(getattr(self, 'filled_steps', self.episode_length))
        if T <= 0:
            raise RuntimeError(
                'Cannot compute team-time potential returns for an empty rollout.'
            )
        team_returns = np.asarray(team_returns, dtype=np.float32).reshape(-1)
        cmax_values = np.asarray(cmax_values, dtype=np.float32).reshape(-1)
        expected = (self.n_rollout_threads,)
        if team_returns.shape != expected or cmax_values.shape != expected:
            raise ValueError(
                f'team_returns and cmax_values must have shape {expected}.'
            )
        if (
            not np.isfinite(team_returns).all()
            or not np.isfinite(cmax_values).all()
            or np.any(cmax_values <= 0.0)
        ):
            raise ValueError('Team-time potential objective contains NaN or Inf.')
        if np.any(self.decision_times[:T] > cmax_values[None, :] + 1e-4):
            raise ValueError('A recorded decision time exceeds final Cmax.')
        potentials = np.asarray(
            self.potential_values[:T + 1], dtype=np.float64
        ).copy()
        if not np.isfinite(potentials).all():
            raise ValueError('Team-time potential contains NaN or Inf.')
        if np.any(potentials > 1e-4):
            raise ValueError(
                'IGA potential must be non-positive because it is the '
                'negative of a non-negative calibrated state cost.'
            )
        raw_terminal_potential = np.zeros(
            self.n_rollout_threads, dtype=np.float64
        )
        terminal_steps = np.full(
            self.n_rollout_threads, T, dtype=np.int64
        )
        for env_idx in range(self.n_rollout_threads):
            terminal_candidates = np.flatnonzero(np.all(
                self.masks[1:T + 1, env_idx, :, 0] <= 0.0,
                axis=1,
            ))
            terminal_step = (
                int(terminal_candidates[0]) + 1
                if terminal_candidates.size else T
            )
            terminal_steps[env_idx] = terminal_step
            raw_terminal_potential[env_idx] = potentials[
                terminal_step, env_idx
            ]
            potentials[terminal_step:, env_idx] = 0.0

        self.team_returns[:] = team_returns
        self.returns.fill(0.0)
        self.rewards.fill(0.0)
        cmax_coef = float(cmax_coef)
        potential_coef = float(potential_coef)
        if not np.isfinite(cmax_coef) or not np.isfinite(potential_coef):
            raise ValueError('Team-time coefficients must be finite.')
        if cmax_coef <= 0.0 or potential_coef < 0.0:
            raise ValueError(
                'cmax_coef must be positive and potential_coef non-negative.'
            )

        for env_idx, terminal_return in enumerate(team_returns):
            state_returns = (
                terminal_return
                + cmax_coef * self.decision_times[:T, env_idx]
                - potential_coef * potentials[:T, env_idx]
            )
            active = self.active_masks[:T, env_idx, :, 0] > 0.0
            self.returns[:T, env_idx, :, 0] = np.where(
                active, state_returns[:, None], 0.0
            )
            trainable = active & (
                self.policy_masks[:T, env_idx, :, 0] > 0.0
            )
            for step in range(T):
                count = int(trainable[step].sum())
                if count <= 0:
                    continue
                elapsed = (
                    self.decision_times[step + 1, env_idx]
                    - self.decision_times[step, env_idx]
                )
                shaping = potential_coef * (
                    potentials[step + 1, env_idx]
                    - potentials[step, env_idx]
                )
                self.rewards[step, env_idx, :, 0][trainable[step]] = (
                    (-cmax_coef * elapsed + shaping) / float(count)
                )

        expected_telescoping = -potential_coef * potentials[0]
        observed_telescoping = potential_coef * np.sum(
            potentials[1:] - potentials[:-1], axis=0
        )
        telescoping_error = observed_telescoping - expected_telescoping
        self.last_team_time_potential_diagnostics = {
            'potential_coef': potential_coef,
            'initial_potential_mean': float(np.mean(potentials[0])),
            'raw_terminal_potential_abs_max': float(
                np.max(np.abs(raw_terminal_potential))
            ),
            'terminal_step_min': int(np.min(terminal_steps)),
            'terminal_step_max': int(np.max(terminal_steps)),
            'telescoping_max_abs_error': float(
                np.max(np.abs(telescoping_error))
            ),
        }
        if self.last_team_time_potential_diagnostics[
            'telescoping_max_abs_error'
        ] > 1e-5:
            raise RuntimeError(
                'Team-time potential failed its telescoping invariant: '
                f'{self.last_team_time_potential_diagnostics}'
            )
        self.value_preds[T] = np.nan_to_num(
            next_value, nan=0.0, posinf=1e4, neginf=-1e4
        )
        return dict(self.last_team_time_potential_diagnostics)


    def build_case_balanced_weights(self, role_loss_coef):
        """Build per-decision weights whose total mass is one per case."""
        T = int(getattr(self, 'filled_steps', self.episode_length))
        self.policy_sample_weights.fill(0.0)
        self.value_sample_weights.fill(0.0)
        if T <= 0:
            return {
                'policy_case_weight_max_error': 0.0,
                'value_case_weight_max_error': 0.0,
            }

        role_loss_coef = {
            int(role): max(0.0, float(coef))
            for role, coef in role_loss_coef.items()
        }
        policy_mask = (
            (self.policy_masks[:T, ..., 0] > 0.0)
            & (self.active_masks[:T, ..., 0] > 0.0)
        )
        value_mask = self.active_masks[:T, ..., 0] > 0.0
        agent_types = self.agent_types[:T]

        def populate(mask, destination):
            case_masses = []
            for env_idx in range(self.n_rollout_threads):
                role_counts = {}
                for role, coef in role_loss_coef.items():
                    if coef <= 0.0:
                        continue
                    count = int(
                        (mask[:, env_idx] & (agent_types[:, env_idx] == role)).sum()
                    )
                    if count > 0:
                        role_counts[role] = count
                coefficient_mass = sum(role_loss_coef[role] for role in role_counts)
                if coefficient_mass <= 0.0:
                    case_masses.append(0.0)
                    continue
                for role, count in role_counts.items():
                    role_mask = mask[:, env_idx] & (agent_types[:, env_idx] == role)
                    weight = role_loss_coef[role] / coefficient_mass / float(count)
                    destination[:T, env_idx, :, 0][role_mask] = weight
                case_masses.append(float(destination[:T, env_idx].sum()))
            errors = [abs(mass - 1.0) for mass in case_masses if mass > 0.0]
            return max(errors, default=0.0)

        policy_error = populate(policy_mask, self.policy_sample_weights)
        value_error = populate(value_mask, self.value_sample_weights)
        return {
            'policy_case_weight_max_error': float(policy_error),
            'value_case_weight_max_error': float(value_error),
        }

    def apply_time_tail_policy_weights(self, start_fraction, final_weight):
        """Redistribute each case's policy mass toward its late decisions.

        The ramp uses environment time divided by terminal Cmax.  It changes
        neither the value targets nor the total weight of a case, preventing
        long trajectories from regaining the decision-count bias removed by
        case-balanced PPO.
        """
        start_fraction = float(start_fraction)
        final_weight = float(final_weight)
        if not 0.0 <= start_fraction <= 1.0:
            raise ValueError('tail policy start fraction must be in [0, 1].')
        if final_weight < 1.0 or not np.isfinite(final_weight):
            raise ValueError('tail policy weight must be finite and at least 1.')
        T = int(getattr(self, 'filled_steps', self.episode_length))
        if T <= 0 or start_fraction >= 1.0 or final_weight <= 1.0:
            return {
                'tail_policy_enabled': 0.0,
                'tail_policy_weighted_fraction': 0.0,
                'tail_policy_case_weight_max_error': 0.0,
            }
        if np.any(self.team_cmax_values <= 0.0):
            raise RuntimeError(
                'Tail policy weighting requires terminal Cmax for every case.'
            )

        weighted_count = 0
        positive_count = 0
        errors = []
        denominator = max(1.0 - start_fraction, 1e-12)
        for env_idx, cmax in enumerate(self.team_cmax_values):
            weights = self.policy_sample_weights[:T, env_idx, :, 0]
            positive = weights > 0.0
            old_mass = float(weights.sum())
            if old_mass <= 0.0:
                continue
            progress = np.clip(
                self.decision_times[:T, env_idx] / float(cmax), 0.0, 1.0
            )
            ramp = np.clip(
                (progress - start_fraction) / denominator, 0.0, 1.0
            )
            multipliers = 1.0 + (final_weight - 1.0) * ramp
            weights *= multipliers[:, None]
            new_mass = float(weights.sum())
            if new_mass <= 0.0 or not np.isfinite(new_mass):
                raise RuntimeError('Tail policy weighting produced invalid mass.')
            weights *= old_mass / new_mass
            errors.append(abs(float(weights.sum()) - old_mass))
            positive_count += int(positive.sum())
            weighted_count += int(
                (positive & (ramp[:, None] > 0.0)).sum()
            )
        return {
            'tail_policy_enabled': 1.0,
            'tail_policy_weighted_fraction': (
                float(weighted_count) / max(float(positive_count), 1.0)
            ),
            'tail_policy_case_weight_max_error': float(max(errors, default=0.0)),
        }

    def graph_recurrent_generator(self, advantages, mini_batch_size):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param mini_batch_size: (int) number of environments (threads) to include in each mini-batch.
        """
        episode_length = int(getattr(self, 'filled_steps', self.episode_length))
        if episode_length <= 0:
            raise RuntimeError("Cannot generate PPO batches for an empty rollout.")
        n_rollout_threads, num_agents = self.rewards.shape[1:3]

        assert n_rollout_threads >= mini_batch_size, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of environments per batch ({})."
            "".format(n_rollout_threads, mini_batch_size))

        rand = torch.randperm(n_rollout_threads).numpy()

        for start_id in range(0, n_rollout_threads, mini_batch_size):
            end_id = min(start_id + mini_batch_size, n_rollout_threads)
            ind = rand[start_id:end_id]

            for chunk_start in range(0, episode_length, self.data_chunk_length):
                chunk_end = min(chunk_start + self.data_chunk_length, episode_length)
                active_chunk = self.active_masks[chunk_start:chunk_end, ind]
                if active_chunk.sum() <= 0:
                    continue

                graph_obs_batch = []
                for step in range(chunk_start, chunk_end):
                    for thread_idx in ind:
                        graph_obs_batch.append(self.graph_obs[step][thread_idx])

                rnn_states_batch = _graph_cast(self.rnn_states[chunk_start:chunk_end, ind])
                actions_batch = _graph_cast(self.actions[chunk_start:chunk_end, ind])

                current_actions = self.actions[chunk_start:chunk_end, ind]
                last_actions = np.full_like(current_actions, -1)
                if chunk_start > 0:
                    last_actions[0] = self.actions[chunk_start - 1, ind]
                last_actions[1:] = current_actions[:-1]
                last_op_batch = _graph_cast(last_actions[..., 0])
                last_site_batch = _graph_cast(last_actions[..., 1])

                value_preds_batch = _graph_cast(self.value_preds[chunk_start:chunk_end, ind])
                return_batch = _graph_cast(self.returns[chunk_start:chunk_end, ind])
                rewards_batch = _graph_cast(self.rewards[chunk_start:chunk_end, ind])
                active_masks_batch = _graph_cast(active_chunk)
                policy_masks_batch = _graph_cast(self.policy_masks[chunk_start:chunk_end, ind])
                agent_types_batch = _graph_cast(self.agent_types[chunk_start:chunk_end, ind])
                policy_sample_weights_batch = _graph_cast(
                    self.policy_sample_weights[chunk_start:chunk_end, ind]
                )
                value_sample_weights_batch = _graph_cast(
                    self.value_sample_weights[chunk_start:chunk_end, ind]
                )
                old_action_log_probs_batch = _graph_cast(self.action_log_probs[chunk_start:chunk_end, ind])
                adv_targ = _graph_cast(advantages[chunk_start:chunk_end, ind])

                yield graph_obs_batch, rnn_states_batch, actions_batch,\
                      value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch,\
                      adv_targ, last_op_batch, last_site_batch, rewards_batch,\
                      policy_masks_batch, agent_types_batch,\
                      policy_sample_weights_batch, value_sample_weights_batch
