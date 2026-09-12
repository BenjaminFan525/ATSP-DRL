import torch
import numpy as np
import torch.nn.functional as F
from collections.abc import Mapping
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
        self.team_case_deltas = np.full(
            (self.n_rollout_threads,), np.nan, dtype=np.float32
        )
        # Optional action-independent per-case control variate used only by
        # the Actor.  Critic targets remain on the native environment-return
        # scale, which avoids teaching the value network an IGA residual.
        self.actor_case_baseline_offsets = np.zeros(
            (self.n_rollout_threads,), dtype=np.float32
        )
        self.decision_times = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads), dtype=np.float32
        )
        self.potential_values = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads), dtype=np.float32
        )
        self.last_team_time_potential_diagnostics = {}
        self.last_role_event_return_diagnostics = {}

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
        self.last_role_event_return_diagnostics = {}
        self.policy_sample_weights.fill(0.0)
        self.value_sample_weights.fill(0.0)
        self.team_returns.fill(0.0)
        self.team_cmax_values.fill(0.0)
        self.team_case_deltas.fill(np.nan)
        self.actor_case_baseline_offsets.fill(0.0)
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

    def set_team_case_deltas(self, case_deltas):
        """Store paired Cmax excess for variance/tail-aware PPO."""
        values = np.asarray(case_deltas, dtype=np.float32).reshape(-1)
        expected = (self.n_rollout_threads,)
        if values.shape != expected:
            raise ValueError(
                f'case_deltas must have shape {expected}, got {values.shape}.'
            )
        if not np.isfinite(values).all():
            raise ValueError('case_deltas must be finite.')
        self.team_case_deltas[:] = values

    def set_actor_case_baseline_offsets(self, offsets):
        """Store one finite, action-independent Actor offset per case."""
        values = np.asarray(offsets, dtype=np.float32).reshape(-1)
        expected = (self.n_rollout_threads,)
        if values.shape != expected:
            raise ValueError(
                f'Actor case-baseline offsets must have shape {expected}, '
                f'got {values.shape}.'
            )
        if not np.isfinite(values).all():
            raise ValueError('Actor case-baseline offsets must be finite.')
        self.actor_case_baseline_offsets[:] = values

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

    def compute_role_event_time_returns(
        self,
        team_returns,
        cmax_values,
        cmax_coef,
        next_value,
        *,
        gae_lambda=1.0,
        role_gae_lambdas=None,
        value_baselines=None,
        event_credit_weights=None,
        event_credit_mode='elapsed',
        event_credit_uniform_mix=0.15,
        potential_coef=0.0,
    ):
        """Build team-time targets on three independent physical role clocks.

        A mixed environment step can contain plane, ordinary-device, and R014
        decisions.  Treating it as one joint PPO event entangles decision
        frequencies.  This routine walks each role's physical event sequence,
        accumulates elapsed-time cost exactly once per role event, and applies
        TD(lambda) between consecutive events of *the same role*.

        Gamma is deliberately one for the finite-horizon scheduling objective.
        Lambda=1 is an audited identity with the old Monte-Carlo target;
        lambda<1 is the genuinely different Stage3 credit assignment.  The
        terminal bootstrap is always zero and the complete terminal team-cost
        residual is attached to the last event on every role clock.

        ``critical_path`` and ``critical_path_v2`` credit change only the
        decomposition of the fixed
        role return: post-episode non-negative event scores receive
        ``1-uniform_mix`` of the elapsed Cmax cost, while ``uniform_mix`` keeps
        the physical-time decomposition as a variance floor.  The total is
        checked independently for every case/role.  A fitted state potential
        may be composed with either credit mode; its role-event differences
        telescope to ``-beta*Phi(first_role_event)``.
        """

        T = int(getattr(self, 'filled_steps', self.episode_length))
        if T <= 0:
            raise RuntimeError(
                'Cannot compute role-event returns for an empty rollout.'
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
        ):
            raise ValueError('Role-event objective contains NaN or Inf.')
        if np.any(self.decision_times[:T] > cmax_values[None, :] + 1e-4):
            raise ValueError('A role decision time exceeds final Cmax.')

        coef = float(cmax_coef)
        if not np.isfinite(coef) or coef <= 0.0:
            raise ValueError('Role-event cmax_coef must be finite and positive.')
        event_credit_mode = str(event_credit_mode)
        if event_credit_mode not in {
            'elapsed', 'critical_path', 'critical_path_v2'
        }:
            raise ValueError(
                'event_credit_mode must be elapsed, critical_path, or '
                'critical_path_v2.'
            )
        event_credit_uniform_mix = float(event_credit_uniform_mix)
        if (
            not np.isfinite(event_credit_uniform_mix)
            or not 0.0 <= event_credit_uniform_mix <= 1.0
        ):
            raise ValueError('event_credit_uniform_mix must be in [0, 1].')
        potential_coef = float(potential_coef)
        if not np.isfinite(potential_coef) or potential_coef < 0.0:
            raise ValueError('Role-event potential_coef must be non-negative.')
        if event_credit_weights is None:
            event_credit_weights = np.zeros(
                (T, self.n_rollout_threads, self.num_agents),
                dtype=np.float32,
            )
        else:
            event_credit_weights = np.asarray(
                event_credit_weights, dtype=np.float64
            )
            expected_credit_shape = (
                T, self.n_rollout_threads, self.num_agents
            )
            if event_credit_weights.shape != expected_credit_shape:
                raise ValueError(
                    'event_credit_weights must have shape '
                    f'{expected_credit_shape}, got '
                    f'{event_credit_weights.shape}.'
                )
            if (
                not np.isfinite(event_credit_weights).all()
                or np.any(event_credit_weights < 0.0)
            ):
                raise ValueError(
                    'event_credit_weights must be finite and non-negative.'
                )
        potentials = np.asarray(
            self.potential_values[:T], dtype=np.float64
        )
        if not np.isfinite(potentials).all():
            raise ValueError('Role-event potential contains NaN or Inf.')
        if potential_coef > 0.0 and np.any(potentials > 1e-4):
            raise ValueError(
                'Role-event IGA potential must be non-positive.'
            )
        gae_lambda = float(gae_lambda)
        if not np.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError('Role-event GAE lambda must be in [0, 1].')
        role_names = {0: 'plane', 1: 'device', 2: 'transporter'}
        role_lambdas = {role: gae_lambda for role in role_names}
        if role_gae_lambdas is not None:
            if isinstance(role_gae_lambdas, Mapping):
                for role, role_name in role_names.items():
                    if role in role_gae_lambdas:
                        role_lambdas[role] = float(role_gae_lambdas[role])
                    elif role_name in role_gae_lambdas:
                        role_lambdas[role] = float(
                            role_gae_lambdas[role_name]
                        )
            else:
                values = tuple(role_gae_lambdas)
                if len(values) != len(role_names):
                    raise ValueError(
                        'role_gae_lambdas must contain plane, device, and '
                        'transporter values.'
                    )
                role_lambdas = {
                    role: float(values[role]) for role in role_names
                }
        for role, value in role_lambdas.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f'{role_names[role]} role-event GAE lambda must be in '
                    '[0, 1].'
                )
        if value_baselines is None:
            if not all(np.isclose(value, 1.0) for value in role_lambdas.values()):
                raise ValueError(
                    'Role-event TD(lambda<1) requires raw-scale critic '
                    'value_baselines.'
                )
            value_baselines = np.zeros_like(
                self.value_preds[:T], dtype=np.float32
            )
        else:
            value_baselines = np.asarray(value_baselines, dtype=np.float32)
            expected_baseline_shape = self.value_preds[:T].shape
            if value_baselines.shape != expected_baseline_shape:
                raise ValueError(
                    'Role-event value_baselines must have shape '
                    f'{expected_baseline_shape}, got {value_baselines.shape}.'
                )
            if not np.isfinite(value_baselines).all():
                raise ValueError(
                    'Role-event value_baselines contains NaN or Inf.'
                )
        self.team_returns[:] = team_returns
        self.returns.fill(0.0)
        self.rewards.fill(0.0)
        role_event_counts = {0: 0, 1: 0, 2: 0}
        role_td_mc_abs_sum = {0: 0.0, 1: 0.0, 2: 0.0}
        role_td_mc_abs_max = {0: 0.0, 1: 0.0, 2: 0.0}
        max_return_error = 0.0
        max_conservation_error = 0.0
        max_base_conservation_error = 0.0
        max_potential_telescoping_error = 0.0
        credit_sequence_count = 0
        credit_fallback_sequence_count = 0
        credit_positive_event_count = 0
        credit_max_event_share = 0.0
        credit_effective_event_fraction_sum = 0.0

        for env_idx, terminal_return in enumerate(team_returns):
            terminal_residual = float(
                terminal_return + coef * cmax_values[env_idx]
            )
            for role in (0, 1, 2):
                role_active = (
                    (self.active_masks[:T, env_idx, :, 0] > 0.0)
                    & (self.agent_types[:T, env_idx] == role)
                )
                event_steps = np.flatnonzero(role_active.any(axis=1))
                if event_steps.size == 0:
                    continue
                role_event_counts[role] += int(event_steps.size)
                event_times = self.decision_times[event_steps, env_idx].astype(
                    np.float64
                )
                next_times = np.concatenate((
                    event_times[1:],
                    np.asarray([cmax_values[env_idx]], dtype=np.float64),
                ))
                elapsed_costs = coef * (next_times - event_times)
                if np.any(elapsed_costs < -1e-6):
                    raise RuntimeError(
                        'Role-event physical time moved backwards.'
                    )
                elapsed_costs = np.maximum(elapsed_costs, 0.0)
                base_cost = float(elapsed_costs.sum())
                elapsed_distribution = (
                    elapsed_costs / base_cost
                    if base_cost > 1e-12
                    else np.full(
                        event_steps.size,
                        1.0 / float(event_steps.size),
                        dtype=np.float64,
                    )
                )
                distribution = elapsed_distribution
                if event_credit_mode in {
                    'critical_path', 'critical_path_v2'
                }:
                    credit_sequence_count += 1
                    raw_credit = np.asarray([
                        float(event_credit_weights[
                            step, env_idx, role_active[step]
                        ].sum())
                        for step in event_steps
                    ], dtype=np.float64)
                    positive_count = int(np.count_nonzero(raw_credit > 0.0))
                    credit_positive_event_count += positive_count
                    if float(raw_credit.sum()) > 1e-12:
                        causal_distribution = raw_credit / raw_credit.sum()
                        distribution = (
                            (1.0 - event_credit_uniform_mix)
                            * causal_distribution
                            + event_credit_uniform_mix
                            * elapsed_distribution
                        )
                    else:
                        credit_fallback_sequence_count += 1
                    credit_max_event_share = max(
                        credit_max_event_share,
                        float(distribution.max(initial=0.0)),
                    )
                    positive_distribution = distribution[distribution > 0.0]
                    entropy = -float(np.sum(
                        positive_distribution
                        * np.log(positive_distribution)
                    ))
                    credit_effective_event_fraction_sum += (
                        float(np.exp(entropy)) / float(event_steps.size)
                    )

                base_event_rewards = -base_cost * distribution
                event_rewards = base_event_rewards.copy()
                event_rewards[-1] += terminal_residual

                event_potentials = potentials[event_steps, env_idx]
                next_event_potentials = np.concatenate((
                    event_potentials[1:],
                    np.zeros((1,), dtype=np.float64),
                ))
                potential_rewards = potential_coef * (
                    next_event_potentials - event_potentials
                )
                event_rewards += potential_rewards
                mc_returns = np.cumsum(event_rewards[::-1])[::-1]
                if event_credit_mode == 'elapsed':
                    expected_returns = (
                        terminal_return
                        + coef * event_times
                        - potential_coef * event_potentials
                    )
                    max_return_error = max(
                        max_return_error,
                        float(np.max(np.abs(
                            mc_returns - expected_returns
                        ))),
                    )
                expected_initial_return = (
                    terminal_return
                    + coef * event_times[0]
                    - potential_coef * event_potentials[0]
                )
                max_conservation_error = max(
                    max_conservation_error,
                    abs(float(
                        event_rewards.sum() - expected_initial_return
                    )),
                )
                max_base_conservation_error = max(
                    max_base_conservation_error,
                    abs(float(
                        base_event_rewards.sum() + base_cost
                    )),
                )
                max_potential_telescoping_error = max(
                    max_potential_telescoping_error,
                    abs(float(
                        potential_rewards.sum()
                        + potential_coef * event_potentials[0]
                    )),
                )

                event_values = np.asarray([
                    float(value_baselines[
                        step, env_idx, role_active[step], 0
                    ].mean())
                    for step in event_steps
                ], dtype=np.float64)
                next_event_values = np.concatenate((
                    event_values[1:], np.zeros((1,), dtype=np.float64)
                ))
                deltas = event_rewards + next_event_values - event_values
                gae = 0.0
                event_returns = np.zeros_like(event_values)
                for event_idx in reversed(range(event_steps.size)):
                    gae = (
                        float(deltas[event_idx])
                        + role_lambdas[role] * gae
                    )
                    event_returns[event_idx] = event_values[event_idx] + gae

                td_mc_abs = np.abs(event_returns - mc_returns)
                role_td_mc_abs_sum[role] += float(td_mc_abs.sum())
                role_td_mc_abs_max[role] = max(
                    role_td_mc_abs_max[role],
                    float(td_mc_abs.max(initial=0.0)),
                )
                for event_idx, step in enumerate(event_steps):
                    active_agents = role_active[step]
                    self.returns[step, env_idx, :, 0][active_agents] = (
                        event_returns[event_idx]
                    )
                    count = int(active_agents.sum())
                    self.rewards[step, env_idx, :, 0][active_agents] = (
                        event_rewards[event_idx] / float(count)
                    )

        tolerance = 2e-3
        td_mc_count = max(sum(role_event_counts.values()), 1)
        all_mc = all(
            np.isclose(value, 1.0) for value in role_lambdas.values()
        )
        self.last_role_event_return_diagnostics = {
            'gamma': 1.0,
            'gae_lambda': float(gae_lambda),
            'event_credit_mode_critical_path': float(
                event_credit_mode == 'critical_path'
            ),
            'event_credit_mode_critical_path_v2': float(
                event_credit_mode == 'critical_path_v2'
            ),
            'event_credit_uniform_mix': float(event_credit_uniform_mix),
            'event_credit_sequence_count': int(credit_sequence_count),
            'event_credit_fallback_sequence_count': int(
                credit_fallback_sequence_count
            ),
            'event_credit_positive_event_count': int(
                credit_positive_event_count
            ),
            'event_credit_max_event_share': float(
                credit_max_event_share
            ),
            'event_credit_effective_event_fraction_mean': float(
                credit_effective_event_fraction_sum
                / max(credit_sequence_count, 1)
            ),
            'potential_coef': float(potential_coef),
            'heterogeneous_gae_lambda': float(
                len({round(value, 12) for value in role_lambdas.values()}) > 1
            ),
            'plane_event_count': int(role_event_counts[0]),
            'device_event_count': int(role_event_counts[1]),
            'transporter_event_count': int(role_event_counts[2]),
            'return_identity_max_abs_error': float(max_return_error),
            'cost_conservation_max_abs_error': float(max_conservation_error),
            'base_cost_conservation_max_abs_error': float(
                max_base_conservation_error
            ),
            'potential_telescoping_max_abs_error': float(
                max_potential_telescoping_error
            ),
            'td_mc_mean_abs_difference': float(
                sum(role_td_mc_abs_sum.values()) / td_mc_count
            ),
            'td_mc_max_abs_difference': float(
                max(role_td_mc_abs_max.values(), default=0.0)
            ),
            'mc_identity_enforced': float(all_mc),
            'terminal_bootstrap_abs_max': 0.0,
        }
        for role, role_name in role_names.items():
            count = max(role_event_counts[role], 1)
            self.last_role_event_return_diagnostics.update({
                f'{role_name}_gae_lambda': float(role_lambdas[role]),
                f'{role_name}_td_mc_mean_abs_difference': float(
                    role_td_mc_abs_sum[role] / count
                ),
                f'{role_name}_td_mc_max_abs_difference': float(
                    role_td_mc_abs_max[role]
                ),
            })
        if max(
            max_return_error,
            max_conservation_error,
            max_base_conservation_error,
            max_potential_telescoping_error,
        ) > tolerance:
            raise RuntimeError(
                'Role-event return conservation failed: '
                f'{self.last_role_event_return_diagnostics}'
            )
        if (
            all_mc
            and self.last_role_event_return_diagnostics[
                'td_mc_max_abs_difference'
            ] > tolerance
        ):
            raise RuntimeError(
                'Role-event lambda=1 failed to reproduce Monte-Carlo returns: '
                f'{self.last_role_event_return_diagnostics}'
            )
        self.value_preds[T] = np.nan_to_num(
            next_value, nan=0.0, posinf=1e4, neginf=-1e4
        )
        return dict(self.last_role_event_return_diagnostics)

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


    @staticmethod
    def _bounded_sqrt_role_shares(
        role_event_counts,
        min_share,
        max_share,
    ):
        """Project sqrt(event count) shares onto a bounded unit simplex."""

        active = {
            int(role): int(count)
            for role, count in role_event_counts.items()
            if int(count) > 0
        }
        if not active:
            return {}
        if len(active) == 1:
            return {next(iter(active)): 1.0}
        min_share = float(min_share)
        max_share = float(max_share)
        if (
            not np.isfinite(min_share)
            or not np.isfinite(max_share)
            or min_share < 0.0
            or max_share <= 0.0
            or min_share > max_share
        ):
            raise ValueError('Invalid sqrt-event role-share bounds.')
        role_count = len(active)
        lower = min(min_share, 1.0 / role_count)
        upper = max(max_share, 1.0 / role_count)
        raw = {
            role: float(np.sqrt(count)) for role, count in active.items()
        }

        # sum(clip(scale * raw)) is monotone, so bisection gives a stable
        # bounded projection without privileging a role when a bound is hit.
        low_scale = 0.0
        high_scale = 1.0
        while sum(
            min(upper, max(lower, high_scale * value))
            for value in raw.values()
        ) < 1.0:
            high_scale *= 2.0
        for _ in range(80):
            scale = 0.5 * (low_scale + high_scale)
            mass = sum(
                min(upper, max(lower, scale * value))
                for value in raw.values()
            )
            if mass < 1.0:
                low_scale = scale
            else:
                high_scale = scale
        shares = {
            role: min(upper, max(lower, high_scale * value))
            for role, value in raw.items()
        }
        normalizer = sum(shares.values())
        return {role: share / normalizer for role, share in shares.items()}

    def build_case_balanced_weights(
        self,
        role_loss_coef,
        *,
        role_loss_weighting='fixed',
        role_loss_min_share=0.15,
        role_loss_max_share=0.60,
    ):
        """Build case-balanced fixed or event-adaptive role weights.

        ``fixed`` retains the exact historical per-decision weighting used by
        A1. ``sqrt_event`` gives each case total mass one, allocates role mass
        proportional to sqrt(role event count), and divides each role's mass
        uniformly over its joint events (then over simultaneous agents).
        """
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
        role_loss_weighting = str(role_loss_weighting)
        if role_loss_weighting not in {'fixed', 'sqrt_event'}:
            raise ValueError(
                'role_loss_weighting must be fixed or sqrt_event.'
            )
        policy_mask = (
            (self.policy_masks[:T, ..., 0] > 0.0)
            & (self.active_masks[:T, ..., 0] > 0.0)
        )
        value_mask = self.active_masks[:T, ..., 0] > 0.0
        agent_types = self.agent_types[:T]

        def populate(mask, destination, prefix):
            case_masses = []
            role_case_shares = {role: [] for role in role_loss_coef}
            role_event_totals = {role: 0 for role in role_loss_coef}
            for env_idx in range(self.n_rollout_threads):
                role_decision_counts = {}
                role_event_counts = {}
                for role, coef in role_loss_coef.items():
                    if coef <= 0.0:
                        continue
                    role_mask = (
                        mask[:, env_idx]
                        & (agent_types[:, env_idx] == role)
                    )
                    decision_count = int(role_mask.sum())
                    event_count = int(role_mask.any(axis=1).sum())
                    if decision_count > 0 and event_count > 0:
                        role_decision_counts[role] = decision_count
                        role_event_counts[role] = event_count
                        role_event_totals[role] += event_count
                if role_loss_weighting == 'sqrt_event':
                    role_shares = self._bounded_sqrt_role_shares(
                        role_event_counts,
                        role_loss_min_share,
                        role_loss_max_share,
                    )
                else:
                    coefficient_mass = sum(
                        role_loss_coef[role]
                        for role in role_decision_counts
                    )
                    role_shares = {
                        role: role_loss_coef[role] / coefficient_mass
                        for role in role_decision_counts
                    } if coefficient_mass > 0.0 else {}
                coefficient_mass = sum(role_shares.values())
                if coefficient_mass <= 0.0:
                    case_masses.append(0.0)
                    continue
                for role, share in role_shares.items():
                    role_mask = (
                        mask[:, env_idx]
                        & (agent_types[:, env_idx] == role)
                    )
                    if role_loss_weighting == 'fixed':
                        destination[:T, env_idx, :, 0][role_mask] = (
                            share / float(role_decision_counts[role])
                        )
                    else:
                        per_event_mass = share / float(role_event_counts[role])
                        for step in np.flatnonzero(role_mask.any(axis=1)):
                            active_agents = role_mask[step]
                            destination[step, env_idx, active_agents, 0] = (
                                per_event_mass / float(active_agents.sum())
                            )
                    role_case_shares[role].append(float(share))
                case_masses.append(float(destination[:T, env_idx].sum()))
            errors = [abs(mass - 1.0) for mass in case_masses if mass > 0.0]
            diagnostics = {
                f'{prefix}_case_weight_max_error': float(
                    max(errors, default=0.0)
                ),
            }
            role_names = {0: 'plane', 1: 'device', 2: 'transporter'}
            for role, values in role_case_shares.items():
                role_name = role_names.get(role, f'role_{role}')
                diagnostics.update({
                    f'{prefix}_{role_name}_event_count': float(
                        role_event_totals[role]
                    ),
                    f'{prefix}_{role_name}_share_mean': float(
                        np.mean(values) if values else 0.0
                    ),
                    f'{prefix}_{role_name}_share_min': float(
                        np.min(values) if values else 0.0
                    ),
                    f'{prefix}_{role_name}_share_max': float(
                        np.max(values) if values else 0.0
                    ),
                })
            return diagnostics

        diagnostics = {
            'role_loss_weighting_sqrt_event': float(
                role_loss_weighting == 'sqrt_event'
            ),
            'role_loss_min_share': float(role_loss_min_share),
            'role_loss_max_share': float(role_loss_max_share),
        }
        diagnostics.update(populate(
            policy_mask, self.policy_sample_weights, 'policy'
        ))
        diagnostics.update(populate(
            value_mask, self.value_sample_weights, 'value'
        ))
        return diagnostics

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

    def apply_cvar_case_weights(
        self, tail_fraction, tail_weight, case_metric='cmax'
    ):
        """Reweight the worst terminal-Cmax cases without changing total mass.

        Case-balanced PPO normally assigns every rollout thread unit mass.
        CVaR deliberately changes the *relative* mass across cases, while a
        global normalization keeps the optimizer scale identical to the
        control arm.  Stable sorting makes ties and repeated canaries exactly
        reproducible.  Both actor and critic weights use the same case factor
        so the counterfactual baseline is fitted to the policy objective.
        """
        tail_fraction = float(tail_fraction)
        tail_weight = float(tail_weight)
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError('CVaR tail fraction must be in (0, 1].')
        if tail_weight < 1.0 or not np.isfinite(tail_weight):
            raise ValueError('CVaR tail weight must be finite and at least 1.')
        T = int(getattr(self, 'filled_steps', self.episode_length))
        if T <= 0 or tail_fraction >= 1.0 or tail_weight <= 1.0:
            return {
                'cvar_policy_enabled': 0.0,
                'cvar_tail_fraction': float(tail_fraction),
                'cvar_tail_weight': float(tail_weight),
                'cvar_tail_case_count': 0.0,
                'cvar_mass_conservation_error': 0.0,
                'cvar_threshold_cmax': 0.0,
            }
        case_metric = str(case_metric)
        if case_metric not in {'cmax', 'paired_delta'}:
            raise ValueError('CVaR case metric must be cmax or paired_delta.')
        cmax = np.asarray(self.team_cmax_values, dtype=np.float64)
        if cmax.shape != (self.n_rollout_threads,):
            raise RuntimeError('CVaR terminal-Cmax shape is invalid.')
        if not np.isfinite(cmax).all() or np.any(cmax <= 0.0):
            raise RuntimeError('CVaR requires one positive terminal Cmax per case.')

        ranking_values = cmax
        if case_metric == 'paired_delta':
            ranking_values = np.asarray(
                self.team_case_deltas, dtype=np.float64
            )
            if (
                ranking_values.shape != (self.n_rollout_threads,)
                or not np.isfinite(ranking_values).all()
            ):
                raise RuntimeError(
                    'Paired-delta CVaR requires one finite reference delta '
                    'per rollout case.'
                )

        case_count = int(cmax.size)
        tail_count = max(1, int(np.ceil(tail_fraction * case_count)))
        ordered = np.argsort(ranking_values, kind='stable')
        tail_indices = ordered[-tail_count:]
        multipliers = np.ones(case_count, dtype=np.float64)
        multipliers[tail_indices] = tail_weight

        policy_old = float(self.policy_sample_weights[:T].sum())
        value_old = float(self.value_sample_weights[:T].sum())
        for env_idx, multiplier in enumerate(multipliers):
            self.policy_sample_weights[:T, env_idx] *= multiplier
            self.value_sample_weights[:T, env_idx] *= multiplier
        policy_new = float(self.policy_sample_weights[:T].sum())
        value_new = float(self.value_sample_weights[:T].sum())
        if policy_old > 0.0:
            self.policy_sample_weights[:T] *= policy_old / policy_new
        if value_old > 0.0:
            self.value_sample_weights[:T] *= value_old / value_new
        errors = []
        if policy_old > 0.0:
            errors.append(abs(float(self.policy_sample_weights[:T].sum()) - policy_old))
        if value_old > 0.0:
            errors.append(abs(float(self.value_sample_weights[:T].sum()) - value_old))
        return {
            'cvar_policy_enabled': 1.0,
            'cvar_tail_fraction': float(tail_fraction),
            'cvar_tail_weight': float(tail_weight),
            'cvar_tail_case_count': float(tail_count),
            'cvar_mass_conservation_error': float(max(errors, default=0.0)),
            'cvar_threshold_cmax': float(cmax[tail_indices].min()),
            'cvar_case_metric_paired_delta': float(
                case_metric == 'paired_delta'
            ),
            'cvar_threshold_case_metric': float(
                ranking_values[tail_indices].min()
            ),
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

        # Keep all environment slices from the same temporal chunk adjacent.
        # Gradient accumulation can then reconstruct one complete
        # ``n_rollout_threads * chunk_length`` effective batch even when the
        # final environment slice is smaller than ``mini_batch_size``.  The
        # previous environment-major order grouped several time chunks from
        # the large slice first and produced highly uneven optimizer steps.
        environment_slices = [
            (start_id, min(start_id + mini_batch_size, n_rollout_threads))
            for start_id in range(0, n_rollout_threads, mini_batch_size)
        ]
        for chunk_start in range(0, episode_length, self.data_chunk_length):
            for start_id, end_id in environment_slices:
                ind = rand[start_id:end_id]
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
