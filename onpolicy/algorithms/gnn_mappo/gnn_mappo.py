import numpy as np
import math
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss, expand_slice
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from torch_geometric.loader.dataloader import Batch
from collections import defaultdict

class MAPPO_Trainer():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.mini_batch_size = args.mini_batch_size
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.n_rollout_threads = args.n_rollout_threads

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.grad_accumulation_steps = args.grad_accumulation_steps
        
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
            self.value_normalizer = ValueNorm(input_shape=1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss (with Value Normalization, without Clip/Huber).
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value predictions from data batch.
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        # 1. 价值归一化 (Value Normalization / PopArt)
        if self._use_popart or self._use_valuenorm:
            # 更新归一化器的均值和方差
            self.value_normalizer.update(return_batch)
            # 将真实回报归一化，使其与 Critic 网络的输出尺度对齐
            norm_return_batch = self.value_normalizer.normalize(return_batch)
            error_original = norm_return_batch - values
        else:
            error_original = return_batch - values

        # 2. 计算纯粹的均方误差 (MSE Loss)
        value_loss = mse_loss(error_original)

        # 3. 应用有效动作掩膜 (Active Masks)
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def update_policy_net(self, sample, update_actor=True, perform_step=True):
        graph_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, last_op_batch, last_site_batch, rewards_batch = sample
        
        # Reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy = self.policy.evaluate_actions(graph_batch,
                                                                      rnn_states_batch,
                                                                      active_masks_batch,
                                                                      last_op_batch,
                                                                      last_site_batch,
                                                                      actions_batch)
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        rewards = (rewards_batch*active_masks_batch).sum() / active_masks_batch.sum()
        
        # actor update
        imp_weights = torch.exp(action_log_probs.unsqueeze(-1) - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # 🚨 移除此处的无脑 zero_grad
        # self.policy.actor_optimizer.zero_grad()

        if update_actor:
            # 引入梯度累加：对 Loss 进行缩放
            loss = (policy_loss - dist_entropy * self.entropy_coef) / self.grad_accumulation_steps
            loss.backward()

        actor_grad_norm = torch.tensor(0.0)
        
        # 🚨 当满足累加步数条件时，才真正更新网络和清空梯度
        if perform_step and update_actor:
            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.ac.actor_param.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_gard_norm(self.policy.ac.actor_param.parameters())

            self.policy.actor_optimizer.step()
            self.policy.actor_optimizer.zero_grad()

        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm,
            "dist_entropy": dist_entropy.item(),
            "advantages": adv_targ.mean().item(),
            "rewards": rewards.item()
        }

    def update_value_net(self, sample, perform_step=True):
        graph_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, last_op_batch, last_site_batch, rewards_batch = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        values = self.policy.evaluate_values(graph_batch,
                                             rnn_states_batch,
                                             active_masks_batch,
                                             last_op_batch,
                                             last_site_batch,
                                             actions_batch)

        value_loss = self.cal_value_loss(
            values.view(-1, 1), 
            value_preds_batch.view(-1, 1), 
            return_batch.view(-1, 1), 
            active_masks_batch.view(-1, 1)
        )

        # 🚨 移除此处的无脑 zero_grad
        # self.policy.critic_optimizer.zero_grad()

        # 引入梯度累加：对 Loss 进行缩放
        loss = (value_loss * self.value_loss_coef) / self.grad_accumulation_steps
        loss.backward()

        critic_grad_norm = torch.tensor(0.0)
        
        # 🚨 当满足累加步数条件时，才真正更新网络和清空梯度
        if perform_step:
            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.ac.critic_param.parameters(), self.max_grad_norm)
            else:
                critic_grad_norm = get_gard_norm(self.policy.ac.critic_param.parameters())

            self.policy.critic_optimizer.step()
            self.policy.critic_optimizer.zero_grad()

        return {
            "value_loss": value_loss.item(),
            "critic_grad_norm": critic_grad_norm.item() if isinstance(critic_grad_norm, torch.Tensor) else critic_grad_norm,
            "value_mean": values.mean().item()
        }

    def train(self, buffer, update_actor=True):
        # 🚨 在所有更新开始前，计算出固定的 Advantages。
        # 注意：整个 ppo_epoch 期间，Advantages 必须保持绝对固定！
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        
        train_info = defaultdict(float)
        
        num_mini_batch = math.ceil(self.n_rollout_threads / self.mini_batch_size)
        total_actor_updates = self.ppo_epoch * num_mini_batch if update_actor else 0
        total_critic_updates = self.ppo_epoch * num_mini_batch

        # ==========================================
        # Phase 1: 集中更新 Actor (策略网络)
        # ==========================================
        if update_actor:
            # 开启 Actor 特定的网络模式
            self.policy.ac.eval()
            self.policy.ac.sel_enc.train()
            self.policy.actor_optimizer.zero_grad() # 确保起跑前梯度干净

            for epoch in range(self.ppo_epoch):
                # 每次 epoch 重新打乱数据
                data_generator = buffer.graph_recurrent_generator(advantages, self.mini_batch_size)
                data_samples = list(data_generator)
                total_steps = len(data_samples)

                for step, sample in enumerate(data_samples):
                    # 梯度累加逻辑
                    perform_step = ((step + 1) % self.grad_accumulation_steps == 0) or (step + 1 == total_steps)
                    
                    policy_results = self.update_policy_net(sample, update_actor=True, perform_step=perform_step)
                    
                    # 累加 Actor 的各项指标
                    for k, v in policy_results.items():
                        train_info[k] += v

        # ==========================================
        # Phase 2: 集中更新 Critic (价值网络)
        # ==========================================
        # 恢复正常的训练模式
        self.policy.ac.train()
        self.policy.critic_optimizer.zero_grad() # 确保起跑前梯度干净

        for epoch in range(self.ppo_epoch):
            # 同样每次 epoch 重新打乱数据（用相同的 generator 保证切分维度合法）
            data_generator = buffer.graph_recurrent_generator(advantages, self.mini_batch_size)
            data_samples = list(data_generator)
            total_steps = len(data_samples)

            for step, sample in enumerate(data_samples):
                # 梯度累加逻辑
                perform_step = ((step + 1) % self.grad_accumulation_steps == 0) or (step + 1 == total_steps)

                value_results = self.update_value_net(sample, perform_step=perform_step)
                
                # 累加 Critic 的各项指标
                for k, v in value_results.items():
                    train_info[k] += v

        # ==========================================
        # 均摊统计指标
        # ==========================================
        # 定义哪些指标属于谁，分别除以对应的总更新次数
        actor_keys = ["policy_loss", "actor_grad_norm", "dist_entropy", "advantages", "rewards"]
        critic_keys = ["value_loss", "critic_grad_norm", "value_mean"]

        if total_actor_updates > 0:
            for k in actor_keys:
                if k in train_info:
                    train_info[k] /= total_actor_updates

        if total_critic_updates > 0:
            for k in critic_keys:
                if k in train_info:
                    train_info[k] /= total_critic_updates

        return train_info
    
    def prep_training(self):
        self.policy.ac.train()

    def prep_rollout(self):
        self.policy.ac.eval()
