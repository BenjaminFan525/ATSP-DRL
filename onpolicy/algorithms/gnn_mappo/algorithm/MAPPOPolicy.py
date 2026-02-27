import torch
import numpy as np
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

    def _build_inputs(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices):
        """
        极其精简的输入构建器。
        将 Runner 传来的 numpy 数组或 PyG Batch 转换为大网络 forward 需要的字典格式。
        """
        # 1. 构建 data 字典
        data = {
            # graph_obs 应该是 PyG 的 Batch(HeteroData) 对象
            'graph': graph_obs.to(self.device), 
            'hidden_states': torch.from_numpy(rnn_states).to(self.device, dtype=torch.float32) if rnn_states is not None else None
        }
        
        # 2. 构建 info 字典
        info = {
            # 布尔类型掩码
            'active_agents': torch.from_numpy(active_agents).to(self.device, dtype=torch.bool),
            # 动作索引 (Long类型)
            'last_op_indices': torch.from_numpy(last_op_indices).to(self.device, dtype=torch.long),
            'last_site_indices': torch.from_numpy(last_site_indices).to(self.device, dtype=torch.long),
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
        
        # 调用底层 GNN_Actor_Critic 的 forward
        values, actions, action_log_probs, new_rnn_states = self.ac(data, info, deterministic=deterministic)

        return values, actions, action_log_probs, new_rnn_states

    def get_values(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices):
        """
        计算广义优势估计 (GAE) 时，获取状态的基线价值 V(s)。
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)

        # criticize_only=True，底层网络只计算并返回 Critic 结果
        values = self.ac(data, info, criticize_only=True)

        return values

    def evaluate_actions(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, actions):
        """
        PPO 更新网络阶段 (Update) 调用。
        强制给定历史动作 (actions)，评估在当前最新策略下的对数概率 (用于计算 Ratio) 和信息熵。
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)

        # 拆解动作张量：actions 形状通常为 [Batch, M, 2]，其中 [..., 0] 是工序，[..., 1] 是机位
        chosen_op = torch.from_numpy(actions[:, :, 0]).long().to(self.device)
        chosen_site = torch.from_numpy(actions[:, :, 1]).long().to(self.device)

        # eval_action=True, 强制给定动作计算概率
        action_log_probs, dist_entropy = self.ac(
            data, info, 
            chosen_idx=chosen_op, 
            chosen_entry=chosen_site, 
            eval_action=True
        )

        return action_log_probs, dist_entropy
    
    def evaluate_values(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, actions):
        """
        (备用) 评估阶段单纯获取 V(s)
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)
        
        values = self.ac(
            data, info, 
            actor_grad=False, 
            criticize_only=True
        )

        return values

    def act(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, deterministic=False):
        """
        纯评估部署 (Evaluation/Testing) 时调用。
        不需要计算 Critic，直接吐出动作和新的状态。
        """
        data, info = self._build_inputs(graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices)

        # criticize=False，直接关闭价值评估分支以加速推理
        actions, new_rnn_states = self.ac(data, info, deterministic=deterministic, criticize=False)
        
        return actions, new_rnn_states