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
            (self.episode_length, self.n_rollout_threads, num_agents, 2), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.action_dists = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 2), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.zeros_like(self.masks)

        self.step = 0

    def graph_insert(self, obs, rnn_states, actions, action_log_probs,
                   value_preds, rewards, masks, active_masks):
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
        
        # ========================================================
        # 1. 异构图数据存入逻辑 (遍历列表存入，绝对对齐线程索引)
        # ========================================================
        for thread_idx in range(self.n_rollout_threads):
            self.graph_obs[self.step + 1][thread_idx] = obs[thread_idx].clone()

        # ========================================================
        # 2. 常规 Numpy 张量存入逻辑
        # ========================================================
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.actions[self.step] = actions.copy()
        
        self.action_log_probs[self.step] = action_log_probs.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        self.value_preds[self.step] = value_preds.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        # self.rewards[self.step] = rewards.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        
        self.masks[self.step + 1] = masks.reshape(self.n_rollout_threads, self.num_agents, 1).copy()
        self.active_masks[self.step + 1] = active_masks.reshape(self.n_rollout_threads, self.num_agents, 1).copy()

        invalid_indices = (actions[:, :, 1] == 0)
        if invalid_indices.any():
            self.active_masks[self.step][invalid_indices] = 0.0
            
        self.step = (self.step + 1) % self.episode_length

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
        # T: Episode Length, N: Threads, M: Agents
        T, N, M, _ = self.rewards.shape
        
        use_v_norm = (getattr(self, '_use_popart', False) or getattr(self, '_use_valuenorm', False)) and (value_normalizer is not None)
        
        if self._use_gae:
            # --- 1. 一次性反归一化 (保留原维度 T, N, M, 1) ---
            if use_v_norm:
                denorm_values = value_normalizer.denormalize(self.value_preds[:-1])
                denorm_next_value = value_normalizer.denormalize(next_value)
            else:
                denorm_values = self.value_preds[:-1]
                denorm_next_value = next_value
                
            # 【修复点 1】：绝对不能用 self.returns = xxx 覆盖原数组！
            # 必须把真实尺度的 next_value 存在第 T+1 步 ([-1] 哨兵位)
            self.returns[-1] = denorm_next_value
            
            # gae 现在的形状是 (N, M, 1)，每个环境的每架飞机都有独立计算的优势
            gae = np.zeros((N, M, 1), dtype=np.float32)
            
            # next_active_value 的形状也是 (N, M, 1)
            next_active_value = denorm_next_value

            # --- 2. 仅在时间维度 T 上反向迭代 ---
            for step in reversed(range(T)):
                is_decision_step = self.active_masks[step]
                
                # --- A. 计算 Delta ---
                delta = self.rewards[step] + self.gamma * next_active_value - denorm_values[step]
                
                # --- B. 更新 GAE ---
                gae_update = delta + self.gamma * self.gae_lambda * gae
                gae = is_decision_step * gae_update + (1.0 - is_decision_step) * gae
                
                # --- C. 计算 Returns ---
                # 【修复点 2】：安全地原地赋值给已有的 returns 数组
                self.returns[step] = gae + denorm_values[step]
                
                # --- D. 更新 Bootstrap 指针 ---
                next_active_value = is_decision_step * denorm_values[step] + (1.0 - is_decision_step) * next_active_value

            # 哨兵位更新保持归一化原值
            self.value_preds[-1] = next_value

        else:
            # 不使用 GAE 的情况 (也做了向量化对齐)
            if use_v_norm:
                self.returns[-1] = value_normalizer.denormalize(next_value)
            else:
                self.returns[-1] = next_value
                
            for step in reversed(range(T)):
                self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def graph_recurrent_generator(self, advantages, mini_batch_size):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param mini_batch_size: (int) number of environments (threads) to include in each mini-batch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]

        assert n_rollout_threads >= mini_batch_size, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of environments per batch ({})."
            "".format(n_rollout_threads, mini_batch_size))

        rand = torch.randperm(n_rollout_threads).numpy()

        for start_id in range(0, n_rollout_threads, mini_batch_size):
            end_id = min(start_id + mini_batch_size, n_rollout_threads)
            ind = rand[start_id:end_id]

            graph_obs_batch = []
            for step in range(episode_length):
                for thread_idx in ind:
                    graph_obs_batch.append(self.graph_obs[step][thread_idx])

            rnn_states_batch = _graph_cast(self.rnn_states[:-1, ind])
            actions_batch = _graph_cast(self.actions[:, ind])

            current_actions = self.actions[:, ind]
            last_actions = np.full_like(current_actions, -1)
            last_actions[1:] = current_actions[:-1]
            last_op_batch = _graph_cast(last_actions[..., 0])
            last_site_batch = _graph_cast(last_actions[..., 1])

            value_preds_batch = _graph_cast(self.value_preds[:-1, ind])
            return_batch = _graph_cast(self.returns[:-1, ind])
            rewards_batch = _graph_cast(self.rewards[:, ind])
            active_masks_batch = _graph_cast(self.active_masks[:-1, ind])
            old_action_log_probs_batch = _graph_cast(self.action_log_probs[:, ind])
            adv_targ = _graph_cast(advantages[:, ind])

            yield graph_obs_batch, rnn_states_batch, actions_batch,\
                  value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, last_op_batch, last_site_batch, rewards_batch