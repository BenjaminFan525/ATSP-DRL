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
        self.shared_actor_lr_max_multiplier = float(getattr(
            args, 'shared_actor_lr_max_multiplier', 0.0
        ))
        if self.shared_actor_lr_scale < 0.0:
            raise ValueError('--shared_actor_lr_scale must be non-negative.')
        if self.shared_actor_lr_max_multiplier < 0.0:
            raise ValueError(
                '--shared_actor_lr_max_multiplier must be non-negative.'
            )
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
                                   stage1_baseline=getattr(
                                       args, 'stage1_baseline', 'proposed'
                                   ),
                                   central_team_critic=bool(getattr(
                                       args, 'central_team_critic', False
                                   )),
                                   counterfactual_q_baseline=bool(getattr(
                                       args, 'counterfactual_q_baseline', False
                                   )),
                                   counterfactual_q_topk=int(getattr(
                                       args, 'counterfactual_q_topk', 8
                                   )),
                                   counterfactual_q_min_mass=float(getattr(
                                       args, 'counterfactual_q_min_mass', 0.90
                                   )),
                                   counterfactual_baseline_mix=float(getattr(
                                       args, 'counterfactual_baseline_mix', 1.0
                                   )),
                                   device_policy_head_mode=str(getattr(
                                       args, 'device_policy_head_mode', 'shared'
                                   )),
                                   ordinary_device_type_count=int(getattr(
                                       args, 'ordinary_device_type_count', 10
                                   )),
                                   device_timing_head=bool(getattr(
                                       args, 'device_timing_head', False
                                   )),
                                   device_global_matching=bool(getattr(
                                       args, 'device_global_matching', False
                                   )),
                                   request_ready_prediction=bool(getattr(
                                       args, 'request_ready_prediction', False
                                   )),
                                   request_ready_time_scale=float(getattr(
                                       args, 'request_ready_time_scale', 3600.0
                                   )),
                                   request_ready_policy_injection=str(getattr(
                                       args,
                                       'request_ready_policy_injection',
                                       'learned',
                                   )),
                                   request_ready_hard_blocking=bool(getattr(
                                       args,
                                       'request_ready_hard_blocking',
                                       False,
                                   )),
                                   request_ready_context_features=bool(getattr(
                                       args,
                                       'request_ready_context_features',
                                       False,
                                   )),
                                   request_ready_head_mode=str(getattr(
                                       args,
                                       'request_ready_head_mode',
                                       'shared',
                                   )),
                                   request_ready_quantile_head=bool(getattr(
                                       args,
                                       'request_ready_quantile_head',
                                       False,
                                   )),
                                   device_resource_adapter=bool(getattr(
                                       args, 'device_resource_adapter', False
                                   )),
                                   device=device)
        self.shared_encoder_activation_checkpoint = bool(getattr(
            args, 'shared_encoder_activation_checkpoint', False
        ))
        self.ac.shared_encoder_activation_checkpoint = (
            self.shared_encoder_activation_checkpoint
        )
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
        train_device=True,
        train_transporter=True,
    ):
        """Configure selective joint-RL trainability for Stage3 schedules.

        This method is intentionally separate from supervised Stage2, which
        calls :meth:`set_resource_supervised_training_stage` and never enables
        a critic.  The switches here let Stage3 protect the shared GNN and/or
        aircraft backend during a stabilization window before full joint RL.
        """
        self._set_module_group_trainable(self.ac.shared_actor_param, not freeze_shared)
        self._set_module_group_trainable(self.ac.plane_actor_param, not freeze_plane)
        self._set_module_group_trainable(
            self.ac.device_actor_param, bool(train_device)
        )
        self._set_module_group_trainable(
            self.ac.transporter_actor_param, bool(train_transporter)
        )
        self._set_module_group_trainable(self.ac.critic_param, True)

    def set_resource_supervised_training_stage(self):
        """Freeze S1 and critics; train only prediction/resource policy heads."""

        self._set_module_group_trainable(self.ac.shared_actor_param, False)
        self._set_module_group_trainable(self.ac.plane_actor_param, False)
        self._set_module_group_trainable(self.ac.device_actor_param, True)
        self._set_module_group_trainable(self.ac.transporter_actor_param, True)
        self._set_module_group_trainable(self.ac.critic_param, False)

    def set_joint_training_stage(self, freeze_plane=False, freeze_shared=False):
        """Configure canonical Stage3 full-joint trainability.

        Plane, ordinary-device, and transporter actors share the existing
        encoder; no encoder split or extra policy network is introduced.
        Freeze switches provide a short stabilization schedule before every
        actor group and the shared encoder train jointly.
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
        runtime = getattr(self, 'stage3_execution_cache', None)
        stage3_prepared = runtime is not None and runtime.active
        if stage3_prepared:
            graph_obs = runtime.prepare_graph(graph_obs)
        
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
            
        if not stage3_prepared:
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
        self._apply_actor_group_lrs()
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
        self._apply_actor_group_lrs()
        shared_lr = self._actor_group_lr('shared_encoder')
        return {
            'actor_lr_multiplier': updated,
            'actor_lr_multiplier_previous': previous,
            'actor_lr_adapted': float(not np.isclose(previous, updated)),
            'actor_lr': float(shared_lr),
            'shared_actor_lr': float(shared_lr),
            'shared_actor_lr_multiplier_effective': float(
                self._shared_actor_lr_multiplier()
            ),
        }

    def _shared_actor_lr_multiplier(self):
        multiplier = float(self.actor_lr_multiplier)
        if self.shared_actor_lr_max_multiplier > 0.0:
            multiplier = min(
                multiplier, float(self.shared_actor_lr_max_multiplier)
            )
        return multiplier

    def _apply_actor_group_lrs(self):
        """Refresh group LRs while keeping shared adaptation independently capped."""

        for group in self.actor_optimizer.param_groups:
            multiplier = (
                self._shared_actor_lr_multiplier()
                if str(group.get('name', '')) == 'shared_encoder'
                else float(self.actor_lr_multiplier)
            )
            group['lr'] = (
                self.lr
                * float(group.get('lr_scale', 1.0))
                * self.actor_lr_decay_factor
                * multiplier
            )

    def _actor_group_lr(self, name):
        for group in self.actor_optimizer.param_groups:
            if str(group.get('name', '')) == str(name):
                return float(group['lr'])
        raise KeyError(f'Unknown Actor optimizer group: {name!r}')

    def set_shared_actor_lr_scale(self, scale):
        """Apply one epoch's shared-encoder LR scale without touching heads."""

        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError('Shared Actor LR scale must be finite and non-negative.')
        self.shared_actor_lr_scale = scale
        found = False
        for group in self.actor_optimizer.param_groups:
            if str(group.get('name', '')) != 'shared_encoder':
                continue
            group['lr_scale'] = scale
            found = True
            break
        if not found:
            raise RuntimeError('Actor optimizer has no shared_encoder group.')
        self._apply_actor_group_lrs()
        return {
            'shared_actor_lr_scale': scale,
            'shared_actor_lr': self._actor_group_lr('shared_encoder'),
            'shared_actor_lr_multiplier_effective': (
                self._shared_actor_lr_multiplier()
            ),
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
        if self.ac.device_policy_head_mode == 'per_type':
            for type_id in range(self.ac.ordinary_device_type_count):
                copied_role_state += copy_missing_prefix(
                    f'device_type_sel_encs.{type_id}.',
                    'device_sel_enc.',
                )
                copied_role_state += copy_missing_prefix(
                    f'device_type_actors.{type_id}.',
                    'device_actor.',
                )

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

    def set_counterfactual_baseline_mix(self, mix):
        return self.ac.set_counterfactual_baseline_mix(mix)

    def reset_counterfactual_diagnostics(self):
        self.ac.reset_counterfactual_diagnostics()

    def consume_counterfactual_diagnostics(self):
        return self.ac.consume_counterfactual_diagnostics()

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

    def get_actor_actions(
        self,
        graph_obs,
        rnn_states,
        active_agents,
        last_op_indices,
        last_site_indices,
        deterministic=False,
        agent_types=None,
        return_encoder_cache=False,
    ):
        """Collect actor actions without evaluating any critic heads.

        Resource behavior cloning discards rollout values and log-probability
        tensors.  Calling :meth:`get_actions` there nevertheless evaluated a
        critic for every active plane and device.  The actor path and updated
        recurrent state are independent of those critic outputs, so this
        narrower entry point is numerically identical for the values consumed
        by DeviceBC while avoiding unused work.
        """
        data, info = self._build_inputs(
            graph_obs,
            rnn_states,
            active_agents,
            last_op_indices,
            last_site_indices,
            agent_types=agent_types,
        )
        encoded_graph = (
            self.ac.encoder(data['graph'])
            if return_encoder_cache else None
        )
        actions, new_rnn_states = self.ac(
            data,
            info,
            deterministic=deterministic,
            criticize=False,
            encoded_graph=encoded_graph,
        )
        if return_encoder_cache:
            return actions, new_rnn_states, {
                'graph': data['graph'],
                'encoded_graph': encoded_graph,
            }
        return actions, new_rnn_states

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
                         return_log_prob_components=False,
                         encoded_graph=None,
                         return_rnn_states=False):
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
            encoded_graph=encoded_graph,
            return_rnn_states=return_rnn_states,
        )
        if (
            return_decision_mask
            or return_log_prob_components
            or return_rnn_states
        ):
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
