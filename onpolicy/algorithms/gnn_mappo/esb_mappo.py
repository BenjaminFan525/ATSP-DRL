import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss, expand_slice
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from torch_geometric.loader.dataloader import Batch
from collections import defaultdict
from copy import deepcopy

class MAPPO_Trainer():
    """
    Trainer class for MAPPO to update policies with ESB-Lagrangian Safe RL bounds.
    """
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.mini_batch_size = args.mini_batch_size
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.obj = args.obj
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma

        # --- [ESB Added] Safe RL Lagrangian Parameters ---
        self.constraint = getattr(args, 'constraint', ['cost']) 
        self.cost_limits = getattr(args, 'cost_limits', {key: 10.0 for key in self.constraint})
        self.alpha_limit = getattr(args, 'alpha_limit', 0.001)
        
        # 1. Inner loop alpha (State-level safety bound)
        self.alphas = {key: torch.tensor(1 - self.alpha_limit, device=self.device) for key in self.constraint}
        self.SCALE_alpha_MIN_MAX = (0, 1)
        init_value_alpha = max(getattr(args, 'lambda_init', 0.001), 1e-5)
        self.log_lams = {
            key: torch.nn.Parameter(torch.log(torch.tensor(init_value_alpha, device=self.device)), requires_grad=True)
            for key in self.constraint
        }
        alpha_lr = getattr(args, 'alpha_lr', 0.01)
        self.alpha_optimizer = optim.Adam(list(self.log_lams.values()), lr=alpha_lr)

        # 2. Outer loop global lambda (Episodic cost limit)
        init_value_lam = max(getattr(args, 'lagrangian_multiplier_init', 0.001), 1e-5)
        self.lagrangian_multipliers = {
            key: torch.nn.Parameter(torch.tensor(init_value_lam, device=self.device), requires_grad=True) 
            for key in self.constraint
        }
        lambda_lr = getattr(args, 'lambda_lr', 0.008)
        self.lambda_optimizer = optim.Adam(list(self.lagrangian_multipliers.values()), lr=lambda_lr)
        self.lambda_range_projection = torch.nn.ReLU()
        # --------------------------------------------------

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_value_active_masks = args.use_value_active_masks
        
        self.value_normalizer = None # Simplified for brevity, add back popart if needed

        # ESB tracking buffers
        self.lae = {}
        self.B1 = {}
        self.B2 = {}
        self.G1 = {key: [] for key in self.constraint}
        self.G2 = {key: [] for key in self.constraint}

    # --- [ESB Added] Core ESB Computation Methods ---
    def compute_loss_cost_performance(self, safety_states, cost_vals, next_cost_vals, costs):
        lae, B1, B2 = {}, {}, {}
        for key in self.constraint:
            # 评估状态的危险程度 beta
            beta = 1 + torch.clip(torch.tanh(safety_states[key]), -1, 0)
            
            # 计算 Local Advantage Estimate (LAE)
            lae[key] = (next_cost_vals[key] - cost_vals[key]) + \
                       self.alphas[key] * (cost_vals[key] - beta * next_cost_vals[key])
            
            B1[key] = (1 - self.gamma) * next_cost_vals[key] - costs[key]
            B2[key] = self.alphas[key] * (1 - beta) / (1 - self.alphas[key]) * next_cost_vals[key]
        return lae, B1, B2

    def compute_alpha_loss(self, loss_costs: dict):
        loss = 0
        for key, val in loss_costs.items():
            loss -= self.log_lams[key] * val
        return loss

    def lambda_l_op(self, key):
        return torch.clamp(torch.exp(self.log_lams[key]), *self.SCALE_alpha_MIN_MAX)

    def update_alphas(self, loss_costs):
        self.alpha_optimizer.zero_grad()
        alpha_loss = self.compute_alpha_loss(loss_costs)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        for key in self.constraint:
            self.alphas[key] = torch.clip(torch.tanh(0.05 / self.lambda_l_op(key).detach()), 0.01, 1 - self.alpha_limit)

    def update_lagrange_multiplier(self, ep_costs):
        self.lambda_optimizer.zero_grad()
        lambda_loss = 0
        for key in self.constraint:
            lambda_loss -= self.lagrangian_multipliers[key] * (ep_costs[key] - self.cost_limits[key]).mean()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        for param in self.lagrangian_multipliers.values():
            param.data.clamp_(0)
    # --------------------------------------------------

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        error_original = return_batch - values
        value_loss = mse_loss(error_original)
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        return value_loss

    def update_policy_net(self, graph_batch, sample, update_actor=True):
        # [ESB Added] Unpack extended sample with cost arrays
        obs_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, veh_nums_batch, rewards_batch, \
        safety_states_batch, cost_vals_batch, next_cost_vals_batch, costs_batch, cum_costs_batch = sample

        global_nodes_batch, nodes_batch, mask_batch = self.policy.ac.encoder.nodes_encoder(graph_batch)
        total_batch_size = obs_batch.shape[0]
        num_envs = global_nodes_batch.shape[0]
        episode_length = total_batch_size // num_envs

        global_nodes = expand_slice(global_nodes_batch, episode_length, total_batch_size)
        nodes = expand_slice(nodes_batch, episode_length, total_batch_size)
        node_key_padding_mask = expand_slice(mask_batch, episode_length, total_batch_size)
        
        action_log_probs, dist_entropy = self.policy.evaluate_actions(
            global_nodes, nodes, node_key_padding_mask, obs_batch, rnn_states_batch,
            masks_batch, active_masks_batch, available_actions_batch, veh_nums_batch, actions_batch
        )
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        loss_masks_batch = check(1 - active_masks_batch).to(**self.tpdv)

        imp_weights = torch.exp(action_log_probs.unsqueeze(-1) - old_action_log_probs_batch)
        ratio_clip = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)

        # 1. Standard Reward Loss
        surr1 = imp_weights * adv_targ
        surr2 = ratio_clip * adv_targ
        if self._use_policy_active_masks:
            policy_action_loss = (torch.sum(torch.max(surr1, surr2), dim=-1, keepdim=True) * loss_masks_batch).sum() / loss_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # 2. [ESB Added] Integrate Cost Loss into Policy Loss
        surrogate = deepcopy(cum_costs_batch)
        for key in self.constraint:
            surrogate[key] = surrogate[key] + ((ratio_clip.detach() - 1) * self.lae[key] * loss_masks_batch).sum() / (1 - self.alphas[key]) / (1 - self.gamma)
            self.G1[key].append((((ratio_clip - 1) * self.B1[key] * loss_masks_batch).sum() / (1 - self.gamma)).item())
            self.G2[key].append((((ratio_clip - 1) * self.B2[key] * loss_masks_batch).sum() / (1 - self.gamma)).item())
        
        # 外层全局乘子更新
        self.update_lagrange_multiplier(surrogate)
        
        penaltys = {key: self.lambda_range_projection(lam).item() for key, lam in self.lagrangian_multipliers.items()}
        lam_total = (1 + sum(penaltys.values()))

        # 归一化 Reward Loss
        policy_loss /= lam_total
        
        # 合并 Cost Loss (Convex Combination)
        for key, lam in penaltys.items():
            l = lam / lam_total
            cost_surr1 = imp_weights * self.lae[key]
            cost_surr2 = ratio_clip * self.lae[key]
            if self._use_policy_active_masks:
                cost_loss = (torch.sum(torch.max(cost_surr1, cost_surr2), dim=-1, keepdim=True) * loss_masks_batch).sum() / loss_masks_batch.sum()
            else:
                cost_loss = torch.sum(torch.max(cost_surr1, cost_surr2), dim=-1, keepdim=True).mean()
            
            policy_loss += l * cost_loss

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.ac.actor_param.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.ac.actor_param.parameters())

        self.policy.actor_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "dist_entropy": dist_entropy.item(),
            "lam_total": lam_total
        }

    def update_value_net(self, graph_batch, sample):
        obs_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, veh_nums_batch, rewards_batch = sample
        
        # critic update
        global_nodes_batch, nodes_batch, mask_batch = self.policy.ac.encoder.nodes_encoder(graph_batch)

        total_batch_size = obs_batch.shape[0]
        num_envs = global_nodes_batch.shape[0]
        episode_length = total_batch_size // num_envs

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        loss_masks_batch = check(1-active_masks_batch).to(**self.tpdv)

        global_nodes = expand_slice(global_nodes_batch, episode_length, total_batch_size)
        nodes = expand_slice(nodes_batch, episode_length, total_batch_size)
        node_key_padding_mask = expand_slice(mask_batch, episode_length, total_batch_size)

        # Reshape to do in a single forward pass for all steps
        values = self.policy.evaluate_values(global_nodes, nodes, node_key_padding_mask,
                                                                              obs_batch, 
                                                                              rnn_states_batch,
                                                                              masks_batch, 
                                                                              active_masks_batch,
                                                                              available_actions_batch,
                                                                              veh_nums_batch,
                                                                              actions_batch)

        value_loss = self.cal_value_loss(values[self.obj].unsqueeze(-1), value_preds_batch, return_batch, loss_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.ac.critic_param.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.ac.critic_param.parameters())

        self.policy.critic_optimizer.step()

        return {
            "value_loss": value_loss.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "value_mean": values[self.obj].mean().item()
        }

    def update_constraint_coeff(self, graph_batch, sample):
        """ [ESB Added] 先于策略更新执行，用于更新局部的 alpha 乘子 """
        obs_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, veh_nums_batch, rewards_batch, \
        safety_states_batch, cost_vals_batch, next_cost_vals_batch, costs_batch, cum_costs_batch = sample

        # 计算局部的代价优势 (Local Advantage Estimate)
        self.lae, self.B1, self.B2 = self.compute_loss_cost_performance(safety_states_batch, cost_vals_batch, next_cost_vals_batch, costs_batch)
        
        with torch.no_grad():
            global_nodes_batch, nodes_batch, mask_batch = self.policy.ac.encoder.nodes_encoder(graph_batch)
            total_batch_size = obs_batch.shape[0]
            num_envs = global_nodes_batch.shape[0]
            episode_length = total_batch_size // num_envs
            global_nodes = expand_slice(global_nodes_batch, episode_length, total_batch_size)
            nodes = expand_slice(nodes_batch, episode_length, total_batch_size)
            node_key_padding_mask = expand_slice(mask_batch, episode_length, total_batch_size)
            
            action_log_probs, _ = self.policy.evaluate_actions(
                global_nodes, nodes, node_key_padding_mask, obs_batch, rnn_states_batch,
                masks_batch, active_masks_batch, available_actions_batch, veh_nums_batch, actions_batch
            )
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        loss_masks_batch = check(1 - active_masks_batch).to(**self.tpdv)
        ratio = torch.exp(action_log_probs.unsqueeze(-1) - old_action_log_probs_batch)

        loss_cost = {}
        for key in self.constraint:
            if self._use_policy_active_masks:
                loss_cost[key] = ( (ratio * self.lae[key]) * loss_masks_batch).sum() / loss_masks_batch.sum()
            else:
                loss_cost[key] = (ratio * self.lae[key]).mean()
        
        # 更新 alpha
        self.update_alphas(loss_cost)
        
        # 重新计算更新 alpha 后的 LAE，供后续 policy_net 更新使用
        self.lae, self.B1, self.B2 = self.compute_loss_cost_performance(safety_states_batch, cost_vals_batch, next_cost_vals_batch, costs_batch)
        self.G1 = {key: [] for key in self.constraint}
        self.G2 = {key: [] for key in self.constraint}

    def ppo_update(self, graph_batch, sample, update_actor=True):
        train_info = defaultdict(list)
        
        # [ESB Added] 在进行 Actor 更新前，先更新约束系数 alpha 和预计算 LAE
        if update_actor:
            self.policy.ac.eval()
            self.update_constraint_coeff(graph_batch, sample)

        if update_actor:
            self.policy.ac.sel_enc.train()
            for _ in range(self.ppo_epoch):
                policy_results = self.update_policy_net(graph_batch, sample, update_actor)
                for k, v in policy_results.items():
                    train_info[k].append(v)
            self.policy.ac.train()

        for _ in range(self.ppo_epoch):
            value_results = self.update_value_net(graph_batch, sample)
            for k, v in value_results.items():
                train_info[k].append(v)
        
        return {k: np.mean(v) if v else 0.0 for k, v in train_info.items()}

    def train(self, graph_obs, buffer, update_actor=True):
        advantages = buffer.returns - buffer.value_preds[:-1]
        train_info = defaultdict(float)
    
        data_generator = buffer.graph_recurrent_generator(advantages, self.mini_batch_size)
        for sample in data_generator:
            update_results = self.ppo_update(
                Batch.from_data_list([graph_obs[i] for i in sample[-1]]).to(self.device), 
                sample[:-1], 
                update_actor
            )
            for k, v in update_results.items():
                train_info[k] += v

        num_mini_batch = math.ceil(self.n_rollout_threads / self.mini_batch_size)
        for k in train_info.keys():
            train_info[k] /= num_mini_batch
 
        return train_info