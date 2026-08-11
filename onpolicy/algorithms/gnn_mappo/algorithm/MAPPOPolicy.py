import copy

import torch
import numpy as np
from torch_geometric.data import Batch, HeteroData
from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import GNN_Actor_Critic
from onpolicy.utils.util import update_linear_anneal

class GNN_MAPPOPolicy:
    """
    MAPPO 策略封装类。
    包装了 Actor 和 Critic 网络，用于在 PPO 训练循环中计算动作、价值和对数概率。
    """

    def __init__(self, args, ac_cfg, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.anneal_final = args.anneal_final
        self.anneal_original = args.anneal_original
        self.tau_anneal_epochs = int(
            getattr(args, 'tau_anneal_epochs', 0)
        )
        if self.tau_anneal_epochs < 0:
            raise ValueError('--tau_anneal_epochs must be non-negative.')
        self.shared_actor_lr_scale = float(getattr(args, 'shared_actor_lr_scale', 1.0))
        self.plane_actor_lr_scale = float(getattr(args, 'plane_actor_lr_scale', 1.0))
        self.device_actor_lr_scale = float(getattr(args, 'device_actor_lr_scale', 1.0))
        self.transporter_actor_lr_scale = float(getattr(args, 'transporter_actor_lr_scale', 1.0))
        self.actor_lr_multiplier = 1.0
        self.actor_lr_decay_factor = 1.0

        # 初始化重构后的 GNN_Actor_Critic
        self.max_plane_agents = args.max_agent_num
        self.max_device_agents = getattr(args, 'max_device_num', 0) if getattr(args, 'resource_policy', 'heuristic') == 'drl' else 0
        self.ac = GNN_Actor_Critic(**ac_cfg,
                                   max_plane_agents=self.max_plane_agents,
                                   max_device_agents=self.max_device_agents,
                                   plane_order_mode=getattr(
                                       args, 'plane_order_mode', 'fixed'
                                   ),
                                   plane_pair_decoder=getattr(
                                       args, 'plane_pair_decoder', 'joint_pair'
                                   ),
                                   central_team_critic=bool(getattr(
                                       args, 'central_team_critic', False
                                   )),
                                   device=device)
        self.bc_reference_ac = None
        
        self.reset_optimizers()

    def reset_optimizers(self):
        self.actor_lr_multiplier = 1.0
        self.actor_lr_decay_factor = 1.0
        actor_groups = [
            {
                'params': list(self.ac.shared_actor_param.parameters()),
                'lr': self.lr * self.shared_actor_lr_scale,
                'lr_scale': self.shared_actor_lr_scale,
                'name': 'shared_encoder',
            },
            {
                'params': list(self.ac.plane_actor_param.parameters()),
                'lr': self.lr * self.plane_actor_lr_scale,
                'lr_scale': self.plane_actor_lr_scale,
                'name': 'plane_actor',
            },
            {
                'params': list(self.ac.device_actor_param.parameters()),
                'lr': self.lr * self.device_actor_lr_scale,
                'lr_scale': self.device_actor_lr_scale,
                'name': 'device_actor',
            },
            {
                'params': list(self.ac.transporter_actor_param.parameters()),
                'lr': self.lr * self.transporter_actor_lr_scale,
                'lr_scale': self.transporter_actor_lr_scale,
                'name': 'transporter_actor',
            },
        ]
        self.actor_optimizer = torch.optim.Adam(
            actor_groups,
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(self.ac.critic_param.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def capture_bc_reference(self, state_dict=None):
        """Freeze an immutable post-BC policy used only for PPO diagnostics/regularization."""
        reference = copy.deepcopy(self.ac)
        if state_dict is not None:
            reference.load_state_dict(state_dict)
        reference.to(self.device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self.bc_reference_ac = reference

    def has_bc_reference(self):
        return self.bc_reference_ac is not None

    @staticmethod
    def _set_module_group_trainable(module_group, trainable):
        for param in module_group.parameters():
            param.requires_grad_(trainable)

    def set_resource_joint_training_stage(
        self,
        freeze_plane=True,
        freeze_shared=True,
    ):
        """Configure the canonical resource-joint PPO trainability contract.

        Stage 2 adapts only the resource actors and all critics.  The
        protected scope is deliberately explicit: the shared GNN encoder,
        the plane selection GRU, the plane pair actor and (when present) the
        learned plane-order actor.  ``plane_actor_param`` owns the latter
        three modules, so one switch protects the complete plane backend.
        """
        self._set_module_group_trainable(self.ac.shared_actor_param, not freeze_shared)
        self._set_module_group_trainable(self.ac.plane_actor_param, not freeze_plane)
        self._set_module_group_trainable(self.ac.device_actor_param, True)
        self._set_module_group_trainable(self.ac.transporter_actor_param, True)
        self._set_module_group_trainable(self.ac.critic_param, True)

    def set_joint_training_stage(self, freeze_plane=False, freeze_shared=False):
        """Compatibility wrapper for the historical joint-stage entry point.

        New callers should use :meth:`set_resource_joint_training_stage` so
        the intended Stage-2 semantics are visible at the call site.  The
        wrapper is retained because old runners/tests still invoke it.
        """
        return self.set_resource_joint_training_stage(
            freeze_plane=freeze_plane,
            freeze_shared=freeze_shared,
        )

    def set_plane_pretraining_stage(self, freeze_shared=False, freeze_order=False):
        """Train the Stage-1 plane policy while optionally protecting BC modules."""
        self._set_module_group_trainable(
            self.ac.shared_actor_param, not freeze_shared
        )
        self._set_module_group_trainable(self.ac.plane_actor_param, True)
        if freeze_order and self.ac.plane_order_actor is not None:
            self._set_module_group_trainable(self.ac.plane_order_actor, False)
        self._set_module_group_trainable(self.ac.device_actor_param, False)
        self._set_module_group_trainable(self.ac.transporter_actor_param, False)
        self._set_module_group_trainable(self.ac.plane_critic_param, True)
        self._set_module_group_trainable(self.ac.device_critic_param, False)
        self._set_module_group_trainable(self.ac.transporter_critic_param, False)

    def _to_tensor(self, x, dtype=torch.float32):
        """安全的类型转换器"""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def _build_inputs(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
                      agent_types=None):
        """将 Numpy/List 数据组装成网络所需的 Tensor/Batch"""
        
        # 1. 图数据自动 Batching (兼容单环境测试与多线程环境收集)
        # A PyG Batch is also a HeteroData instance.  Handle it first or an
        # already batched rollout is silently wrapped as a single graph.
        if isinstance(graph_obs, Batch):
            pass
        elif isinstance(graph_obs, np.ndarray):
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
        
        active_agents = self._to_tensor(active_agents, torch.bool)
        if active_agents.dim() == 3 and active_agents.shape[-1] == 1:
            active_agents = active_agents.squeeze(-1)

        info = {
            'active_agents': active_agents,
            'last_op_indices': self._to_tensor(last_op_indices, torch.long),
            'last_site_indices': self._to_tensor(last_site_indices, torch.long),
        }
        if agent_types is not None:
            agent_types = self._to_tensor(agent_types, torch.long)
            if agent_types.dim() == 3 and agent_types.shape[-1] == 1:
                agent_types = agent_types.squeeze(-1)
            info['agent_types'] = agent_types
        
        return data, info

    def lr_decay(self, episode, episodes):
        """衰减学习率"""
        factor = max(0.0, 1.0 - float(episode) / max(1, episodes))
        self.actor_lr_decay_factor = factor
        for group in self.actor_optimizer.param_groups:
            group['lr'] = (
                self.lr
                * group.get('lr_scale', 1.0)
                * factor
                * self.actor_lr_multiplier
            )
        for group in self.critic_optimizer.param_groups:
            group['lr'] = self.critic_lr * factor

    def adapt_actor_lr(
        self,
        measured_kl,
        *,
        low,
        high,
        min_scale,
        max_scale,
        up,
        down,
    ):
        """Adapt all actor groups while preserving their relative LR scales."""
        measured_kl = float(measured_kl)
        previous = float(self.actor_lr_multiplier)
        updated = previous
        if np.isfinite(measured_kl):
            if measured_kl < float(low):
                updated *= float(up)
            elif measured_kl > float(high):
                updated *= float(down)
        updated = float(np.clip(updated, float(min_scale), float(max_scale)))
        return self._set_actor_lr_multiplier(updated, previous=previous)

    def _set_actor_lr_multiplier(self, multiplier, *, previous=None):
        """Apply one Actor-LR multiplier to every role-specific parameter group."""
        if previous is None:
            previous = float(self.actor_lr_multiplier)
        updated = float(multiplier)
        self.actor_lr_multiplier = updated
        for group in self.actor_optimizer.param_groups:
            group['lr'] = (
                self.lr
                * group.get('lr_scale', 1.0)
                * self.actor_lr_decay_factor
                * updated
            )
        return {
            'actor_lr_multiplier': updated,
            'actor_lr_multiplier_previous': previous,
            'actor_lr_adapted': float(not np.isclose(previous, updated)),
            'actor_lr': float(self.actor_optimizer.param_groups[0]['lr']),
        }

    def downshift_actor_lr(self, *, min_scale, down):
        """Conservatively reduce Actor LR after a blocked/incomplete PPO update."""
        previous = float(self.actor_lr_multiplier)
        updated = max(float(min_scale), previous * float(down))
        metrics = self._set_actor_lr_multiplier(updated, previous=previous)
        metrics['actor_lr_incomplete_downshift'] = float(
            not np.isclose(previous, updated)
        )
        return metrics

    def load_model_state(self, state_dict):
        """Load compatible checkpoint tensors and skip newly introduced/mismatched modules."""
        current_state = self.ac.state_dict()
        compatible_state = {
            k: v for k, v in state_dict.items()
            if k in current_state and current_state[k].shape == v.shape
        }
        current_state.update(compatible_state)

        def copy_missing_prefix(dst_prefix, src_prefix):
            if any(k.startswith(dst_prefix) for k in compatible_state):
                return 0
            copied = 0
            for key in list(current_state.keys()):
                if not key.startswith(dst_prefix):
                    continue
                src_key = key.replace(dst_prefix, src_prefix, 1)
                if src_key in state_dict and state_dict[src_key].shape == current_state[key].shape:
                    current_state[key] = state_dict[src_key].clone()
                    copied += 1
                elif src_key in current_state and current_state[src_key].shape == current_state[key].shape:
                    current_state[key] = current_state[src_key].clone()
                    copied += 1
            return copied

        copied_role_state = 0
        for dst_prefix, src_prefix in (
            ('plane_sel_enc.', 'sel_enc.'),
            ('device_sel_enc.', 'sel_enc.'),
            ('transporter_sel_enc.', 'sel_enc.'),
            ('plane_critic.', 'critic.'),
            ('device_critic.', 'critic.'),
            ('transporter_critic.', 'critic.'),
            ('transporter_actor.', 'device_actor.'),
        ):
            copied_role_state += copy_missing_prefix(dst_prefix, src_prefix)

        self.ac.load_state_dict(current_state)
        skipped = len(state_dict) - len(compatible_state)
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} incompatible checkpoint tensors for the role-separated policy.")
        if copied_role_state > 0:
            print(f"[Info] Initialized {copied_role_state} role-separated tensors from compatible legacy modules.")

    def hyperparams_anneal(self, episode, episodes):
        """退火温度系数 tau；reference 使用同温度以仅度量参数漂移。"""
        update_linear_anneal(
            self.ac,
            self.anneal_original,
            self.anneal_final,
            episode,
            episodes,
            self.tau_anneal_epochs,
        )
        if self.bc_reference_ac is not None:
            self.bc_reference_ac.tau = self.ac.tau

    def get_actions(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
                    deterministic=False, agent_types=None, return_decision_mask=False):
        """
        环境交互/收集数据 (Rollout) 时调用。
        输出动作、价值、对数概率和新的 GRU 隐藏状态。
        """
        data, info = self._build_inputs(
            graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
            agent_types=agent_types,
        )
        
        outputs = self.ac(
            data,
            info,
            deterministic=deterministic,
            return_decision_mask=return_decision_mask,
        )
        if return_decision_mask:
            values, actions, action_log_probs, new_rnn_states, decision_mask = outputs
            return values, actions, action_log_probs, new_rnn_states, decision_mask
        values, actions, action_log_probs, new_rnn_states = outputs

        return values, actions, action_log_probs, new_rnn_states

    def get_values(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
                   agent_types=None):
        """
        计算广义优势估计 (GAE) 时，获取状态的基线价值 V(s)。
        """
        data, info = self._build_inputs(
            graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
            agent_types=agent_types,
        )
        values = self.ac(data, info, criticize_only=True)

        return values

    def evaluate_actions(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, actions,
                         agent_types=None, return_decision_mask=False,
                         return_log_prob_components=False):
        """
        PPO 更新网络阶段 (Update) 调用。
        强制给定历史动作 (actions)，评估在当前最新策略下的对数概率 (用于计算 Ratio) 和信息熵。
        """
        data, info = self._build_inputs(
            graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
            agent_types=agent_types,
        )
        actions = self._to_tensor(actions, torch.long)
        chosen_op, chosen_site = actions[..., 0], actions[..., 1]
        chosen_order = (
            actions[..., 2] if actions.shape[-1] > 2 else None
        )
        outputs = self.ac(
            data,
            info,
            chosen_op=chosen_op,
            chosen_site=chosen_site,
            chosen_order=chosen_order,
            eval_action=True,
            return_decision_mask=return_decision_mask,
            return_log_prob_components=return_log_prob_components,
        )
        if return_decision_mask or return_log_prob_components:
            return outputs

        action_log_probs, dist_entropy = outputs
        return action_log_probs, dist_entropy

    @torch.no_grad()
    def evaluate_bc_reference_actions(
        self,
        graph_obs,
        rnn_states,
        active_agents,
        last_op_indices,
        last_site_indices,
        actions,
        agent_types=None,
        return_decision_mask=False,
    ):
        if self.bc_reference_ac is None:
            raise RuntimeError(
                "BC-reference evaluation requested before capture_bc_reference()."
            )
        data, info = self._build_inputs(
            graph_obs,
            rnn_states,
            active_agents,
            last_op_indices,
            last_site_indices,
            agent_types=agent_types,
        )
        actions = self._to_tensor(actions, torch.long)
        chosen_order = actions[..., 2] if actions.shape[-1] > 2 else None
        self.bc_reference_ac.eval()
        return self.bc_reference_ac(
            data,
            info,
            chosen_op=actions[..., 0],
            chosen_site=actions[..., 1],
            chosen_order=chosen_order,
            eval_action=True,
            return_decision_mask=return_decision_mask,
        )

    def evaluate_values(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices, actions,
                        agent_types=None):
        """
        PPO 更新网络阶段 (Update) 调用。
        强制给定历史动作 (actions)，评估在当前最新策略下的对数概率 (用于计算 Ratio) 和信息熵。
        """
        data, info = self._build_inputs(
            graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
            agent_types=agent_types,
        )
        actions = self._to_tensor(actions, torch.long)
        chosen_op, chosen_site = actions[..., 0], actions[..., 1]
        chosen_order = (
            actions[..., 2] if actions.shape[-1] > 2 else None
        )
        values = self.ac(
            data, info, chosen_op=chosen_op, chosen_site=chosen_site,
            chosen_order=chosen_order, actor_grad=False,
            criticize_only=True,
        )

        return values

    def act(self, graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
            deterministic=False, agent_types=None):
        """
        纯评估部署 (Evaluation/Testing) 时调用。
        不需要计算 Critic，直接吐出动作和新的状态。
        """
        data, info = self._build_inputs(
            graph_obs, rnn_states, active_agents, last_op_indices, last_site_indices,
            agent_types=agent_types,
        )
        actions, new_rnn_states = self.ac(data, info, deterministic=deterministic, criticize=False)
        return actions, new_rnn_states
