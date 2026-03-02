import torch
import numpy as np
from torch_geometric.data import Batch, HeteroData
from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import GNN_Actor_Critic
from onpolicy.utils.util import update_linear_schedule, update_linear_anneal

class GNN_MAPPOPolicy:
    """
    MAPPO 策略封装类。
    包装了 Actor 和 Critic 网络，用于在 PPO 训练循环中计算动作、价值和对数概率。
    """

    def __init__(self, args, ac_cfg, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.anneal_final = args.anneal_final
        self.anneal_original = args.anneal_original

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # 初始化重构后的 GNN_Actor_Critic
        self.ac = GNN_Actor_Critic(**ac_cfg, device=device)
        
        # 配置优化器
        self.actor_optimizer = torch.optim.Adam(self.ac.actor_param.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.ac.critic_param.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def _to_tensor(self, x, dtype=torch.float32):
        """安全的类型转换器"""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def _build_inputs(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices):
        """将 Numpy/List 数据组装成网络所需的 Tensor/Batch"""
        
        # 1. 图数据自动 Batching (兼容单环境测试与多线程环境收集)
        if isinstance(graph_obs, np.ndarray):
            graph_obs = Batch.from_data_list(graph_obs.tolist())
        elif isinstance(graph_obs, list):
            graph_obs = Batch.from_data_list(graph_obs)
        elif isinstance(graph_obs, HeteroData):
            graph_obs = Batch.from_data_list([graph_obs])
            
        graph_obs = graph_obs.to(self.device)
        
        # 2. 构建输入字典
        data = {
            'graph': graph_obs, 
            'hidden_states': self._to_tensor(rnn_states, torch.float32)
        }
        
        info = {
            'active_agents': self._to_tensor(active_agents, torch.bool),
            'last_op_indices': self._to_tensor(last_op_indices, torch.long),
            'last_site_indices': self._to_tensor(last_site_indices, torch.long),
        }
        
        return data, info

    def lr_decay(self, episode, episodes):
        """衰减学习率"""
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def hyperparams_anneal(self, episode, episodes):
        """退火温度系数 tau"""
        update_linear_anneal(self.ac, self.anneal_original, self.anneal_final, episode, episodes)

    def get_actions(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, deterministic=False):
        """
        环境交互/收集数据 (Rollout) 时调用。
        输出动作、价值、对数概率和新的 GRU 隐藏状态。
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)
        
        values, actions, action_log_probs, new_rnn_states = self.ac(data, info, deterministic=deterministic)

        return values, actions, action_log_probs, new_rnn_states

    def get_values(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices):
        """
        计算广义优势估计 (GAE) 时，获取状态的基线价值 V(s)。
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)
        values = self.ac(data, info, criticize_only=True)

        return values

    def evaluate_actions(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, actions):
        """
        PPO 更新网络阶段 (Update) 调用。
        强制给定历史动作 (actions)，评估在当前最新策略下的对数概率 (用于计算 Ratio) 和信息熵。
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)
        actions = self._to_tensor(actions, torch.long)
        chosen_op, chosen_site = actions[..., 0], actions[..., 1]
        action_log_probs, dist_entropy = self.ac(data, info, chosen_op=chosen_op, chosen_site=chosen_site, eval_action=True)

        return action_log_probs, dist_entropy

    def evaluate_values(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, actions):
        """
        PPO 更新网络阶段 (Update) 调用。
        强制给定历史动作 (actions)，评估在当前最新策略下的对数概率 (用于计算 Ratio) 和信息熵。
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)
        actions = self._to_tensor(actions, torch.long)
        chosen_op, chosen_site = actions[..., 0], actions[..., 1]
        values = self.ac(data, info, chosen_op=chosen_op, chosen_site=chosen_site, actor_grad=False, criticize_only=True)

        return values

    def act(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, deterministic=False):
        """
        纯评估部署 (Evaluation/Testing) 时调用。
        不需要计算 Critic，直接吐出动作和新的状态。
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)
        actions, new_rnn_states = self.ac(data, info, deterministic=deterministic, criticize=False)
        return actions, new_rnn_states