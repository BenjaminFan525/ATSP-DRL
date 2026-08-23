import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from onpolicy.algorithms.utils.gnn import HeteroGraphEncoder
from onpolicy.algorithms.utils.gru import SelectionEncoder
from onpolicy.algorithms.utils.ptr_actor import (
    CascadePtrActor,
    DeviceRequestPtrActor,
    JointPairPtrActor,
    PlaneOrderPointer,
)
from onpolicy.algorithms.utils.step_critic import GlobalStepCritic, StepCritic

class _ModuleGroup:
    """Lightweight parameter group that does not register duplicate state_dict keys."""
    def __init__(self, modules):
        self.modules = list(modules)

    def parameters(self):
        for module in self.modules:
            yield from module.parameters()

    def train(self, mode=True):
        for module in self.modules:
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

class GNN_Actor_Critic(nn.Module):
    AGENT_TYPE_PLANE = 0
    AGENT_TYPE_DEVICE = 1
    AGENT_TYPE_TRANSPORTER = 2

    def __init__(self, common_cfg=None, encoder_cfg=None, sel_encoder_cfg=None, actor_cfg=None, critic_cfg=None,
                 max_plane_agents=24, max_device_agents=0,
                 plane_order_mode='fixed', plane_pair_decoder='joint_pair',
                 central_team_critic=False,
                 device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.common = common_cfg
        self.encoder_cfg = encoder_cfg
        self.selection_enc = sel_encoder_cfg
        self.actor_cfg = actor_cfg
        self.critic_cfg = critic_cfg
        self.max_plane_agents = max_plane_agents
        self.max_device_agents = max_device_agents
        self.plane_order_mode = str(plane_order_mode)
        self.plane_pair_decoder = str(plane_pair_decoder)
        self.central_team_critic = bool(central_team_critic)
        if self.plane_order_mode not in {'fixed', 'learned'}:
            raise ValueError(
                f"Unsupported plane_order_mode={self.plane_order_mode!r}."
            )
        if self.plane_pair_decoder not in {'cascade', 'joint_pair'}:
            raise ValueError(
                f"Unsupported plane_pair_decoder={self.plane_pair_decoder!r}."
            )


        # 1. 异构图编码器
        self.encoder = HeteroGraphEncoder(self.common, self.encoder_cfg['gnn_cfg'], self.encoder_cfg['gff_cfg'], **self.factory_kwargs)        
        
        # 2. 角色独立 GRU 时序记忆编码器；角色之间除 GNN 外不共享后端参数。
        self.plane_sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)
        self.device_sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)
        self.transporter_sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)
        
        # 3. 角色独立 Actor 动作网络
        plane_actor_cls = (
            CascadePtrActor
            if self.plane_pair_decoder == 'cascade'
            else JointPairPtrActor
        )
        plane_actor_cfg = dict(self.actor_cfg)
        if self.plane_pair_decoder == 'cascade':
            plane_actor_cfg.pop('pair_feature_dim', None)
        self.actor = plane_actor_cls(
            **self.common, **plane_actor_cfg, **self.factory_kwargs
        )
        self.plane_order_actor = (
            PlaneOrderPointer(**self.common, **self.factory_kwargs)
            if self.plane_order_mode == 'learned' else None
        )
        role_actor_cfg = dict(self.actor_cfg)
        role_actor_cfg.pop('pair_feature_dim', None)
        self.device_actor = DeviceRequestPtrActor(
            **self.common, **role_actor_cfg, **self.factory_kwargs
        )
        self.transporter_actor = DeviceRequestPtrActor(
            **self.common, **role_actor_cfg, **self.factory_kwargs
        )
        
        # 4. 角色独立 Critic 价值网络
        self.plane_critic = StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)
        self.device_critic = StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)
        self.team_critic = GlobalStepCritic(
            **self.common, **self.critic_cfg, **self.factory_kwargs
        )
        self.transporter_critic = StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)
        
        self.tau = 1.0

        self.cfg = {
            'common_cfg': self.common,
            'encoder_cfg': self.encoder_cfg,
            'sel_encoder_cfg': self.selection_enc,
            'actor_cfg': self.actor_cfg,
            'critic_cfg': self.critic_cfg,
            'plane_order_mode': self.plane_order_mode,
            'plane_pair_decoder': self.plane_pair_decoder,
        }

        # Optimizer ownership stays disjoint; learned ordering belongs only to
        # the plane actor backend.
        plane_actor_modules = [self.plane_sel_enc, self.actor]
        if self.plane_order_actor is not None:
            plane_actor_modules.append(self.plane_order_actor)
        optional_order = (
            [] if self.plane_order_actor is None
            else [self.plane_order_actor]
        )
        self.actor_param = _ModuleGroup([
            self.encoder,
            self.plane_sel_enc,
            self.device_sel_enc,
            self.transporter_sel_enc,
            self.actor,
            self.device_actor,
            self.transporter_actor,
            *optional_order,
        ])
        self.actor_param_without_gnn = _ModuleGroup([
            self.plane_sel_enc,
            self.device_sel_enc,
            self.transporter_sel_enc,
            self.actor,
            self.device_actor,
            self.transporter_actor,
            *optional_order,
        ])
        self.shared_actor_param = _ModuleGroup([self.encoder])
        self.plane_actor_param = _ModuleGroup(plane_actor_modules)
        self.device_actor_param = _ModuleGroup([
            self.device_sel_enc, self.device_actor
        ])
        self.transporter_actor_param = _ModuleGroup([
            self.transporter_sel_enc,
            self.transporter_actor,
        ])
        # Keep optimizer ownership disjoint. Previously the encoder and all
        # role GRUs were stepped by two independent Adam instances.
        self.critic_param = _ModuleGroup([
            self.plane_critic,
            self.device_critic,
            self.transporter_critic,
            self.team_critic,
        ])
        self.plane_critic_param = _ModuleGroup([self.plane_critic])
        self.device_critic_param = _ModuleGroup([self.device_critic])
        self.transporter_critic_param = _ModuleGroup([self.transporter_critic])

        self.to(device)

    @staticmethod
    def _local_plane_history_index(last_op_idx, op_start, n_jobs):
        """Convert a replayed global operation id to this plane's local id."""
        local_idx = last_op_idx - int(op_start)
        valid = (local_idx >= 0) & (local_idx < int(n_jobs))
        safe = torch.clamp(local_idx, min=0, max=max(0, int(n_jobs) - 1))
        return safe, valid

    def _resolve_agent_types(self, graph, info, bsz, n_agents, device):
        agent_types = info.get('agent_types', None)
        if agent_types is None and hasattr(graph, 'agent_types'):
            agent_types = graph.agent_types

        fallback = torch.zeros((bsz, n_agents), dtype=torch.long, device=device)
        if n_agents > self.max_plane_agents:
            fallback[:, self.max_plane_agents:] = self.AGENT_TYPE_DEVICE
        if agent_types is None:
            return fallback

        if not torch.is_tensor(agent_types):
            agent_types = torch.as_tensor(agent_types)
        agent_types = agent_types.to(device=device, dtype=torch.long)

        if agent_types.dim() == 1:
            if agent_types.numel() == bsz * n_agents:
                return agent_types.view(bsz, n_agents)
            if agent_types.numel() == n_agents:
                return agent_types.view(1, n_agents).expand(bsz, -1)
        elif agent_types.dim() == 2:
            if agent_types.shape == (bsz, n_agents):
                return agent_types
            if agent_types.shape[0] == 1 and agent_types.shape[1] == n_agents:
                return agent_types.expand(bsz, -1)
            if agent_types.numel() == bsz * n_agents:
                return agent_types.reshape(bsz, n_agents)

        flat_types = agent_types.reshape(-1)
        width = min(flat_types.numel(), bsz * n_agents)
        fallback.reshape(-1)[:width] = flat_types[:width]
        return fallback

    def _dispatch_device_actor(self, query, request_nodes, request_valid_mask, deterministic,
                               chosen_request, actor_grad, is_transporter):
        active_count = query.shape[0]
        n_requests = request_nodes.shape[1]
        cur_req = torch.zeros(active_count, dtype=torch.long, device=query.device)
        cur_log_prob = torch.zeros(active_count, dtype=query.dtype, device=query.device)
        cur_dist = torch.full(
            (active_count, n_requests),
            float('-inf'),
            dtype=query.dtype,
            device=query.device,
        )
        routed = torch.zeros(active_count, dtype=torch.bool, device=query.device)

        role_routes = (
            ((~is_transporter).view(-1).bool(), self.device_actor),
            (is_transporter.view(-1).bool(), self.transporter_actor),
        )
        for role_mask, actor_head in role_routes:
            if not role_mask.any():
                continue

            role_chosen = chosen_request[role_mask] if chosen_request is not None else None
            if actor_grad:
                role_req, role_log_prob, role_dist = actor_head(
                    query=query.unsqueeze(1)[role_mask],
                    request_nodes=request_nodes[role_mask],
                    request_valid_mask=request_valid_mask[role_mask],
                    deterministic=deterministic,
                    tau=self.tau,
                    chosen_request=role_chosen,
                )
            else:
                with torch.no_grad():
                    role_req, role_log_prob, role_dist = actor_head(
                        query=query.unsqueeze(1)[role_mask],
                        request_nodes=request_nodes[role_mask],
                        request_valid_mask=request_valid_mask[role_mask],
                        deterministic=deterministic,
                        tau=self.tau,
                        chosen_request=role_chosen,
                    )

            cur_req[role_mask] = role_req
            cur_log_prob[role_mask] = role_log_prob
            cur_dist[role_mask] = role_dist
            routed[role_mask] = True

        if not routed.all():
            bad_rows = torch.nonzero(~routed, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Device agents were not routed to an actor head for rows {bad_rows}.")
        return cur_req, cur_log_prob, cur_dist

    def forward(self, data, info, deterministic: bool = False, chosen_op=None,
                chosen_site=None, chosen_order=None,
                actor_grad: bool = True, criticize: bool = True, eval_action: bool = False, 
                dist_only: bool = False, criticize_only: bool = False,
                return_decision_mask: bool = False,
                return_log_prob_components: bool = False,
                encoded_graph=None,
                return_rnn_states: bool = False):
        '''
        FJSP 异构图 Actor-Critic 前向传播核心逻辑
        
        参数映射说明：
        chosen_idx -> 代表选中的工序 (chosen_op)
        chosen_entry -> 代表选中的机位 (chosen_site)
        '''
        assert not (dist_only and criticize_only)
        if dist_only: criticize = False
        if criticize_only: criticize = True; actor_grad = False
        if eval_action: criticize = False; actor_grad = True

        # ========================================================
        # 1. 编码异构图 (Hetero Encoder)
        # ========================================================
        # Stage2 freezes the shared encoder.  DeviceBC first runs the actor to
        # advance recurrent state and then replays the teacher action for its
        # supervised loss on exactly the same observation.  Accepting the
        # first pass' detached encoding avoids doing the frozen GNN work twice
        # without changing any trainable activation or gradient.
        enc_out = (
            self.encoder(data['graph'])
            if encoded_graph is None
            else encoded_graph
        )
        global_emb = torch.nan_to_num(enc_out['global_emb'], nan=0.0, posinf=1e4, neginf=-1e4)      # [B, Embed_Dim]
        op_nodes = torch.nan_to_num(enc_out['op_nodes'], nan=0.0, posinf=1e4, neginf=-1e4)          # [B, N_ops, Embed_Dim]
        site_nodes = torch.nan_to_num(enc_out['site_nodes'], nan=0.0, posinf=1e4, neginf=-1e4)      # [B, N_sites, Embed_Dim]
        device_nodes = torch.nan_to_num(enc_out['device_nodes'], nan=0.0, posinf=1e4, neginf=-1e4)  # [B, N_devices, Embed_Dim]
        request_nodes = torch.nan_to_num(enc_out['request_nodes'], nan=0.0, posinf=1e4, neginf=-1e4) # [B, N_requests, Embed_Dim]
        
        # 获取环境生成的合法动作掩码 (True 代表合法可用)
        op_valid_mask = data['graph'].op_mask.clone()             # [B, N_agents, N_ops]
        site_mask_matrix = data['graph'].site_mask_matrix.clone() # [B, N_agents, N_sites]
        job_site_mask_matrix = data['graph'].job_site_mask_matrix.clone()
        agent_job_site_mask_matrix = getattr(
            data['graph'],
            'agent_job_site_mask_matrix',
            None,
        )
        request_mask_matrix = data['graph'].request_mask_matrix.clone()

        # ========================================================
        # 2. 初始化张量与隐状态
        # ========================================================
        bsz = global_emb.shape[0]
        # PPO 传进来的 info 中的 tensor 均需要挪到相应 device
        active_agents = info['active_agents']
        last_op_indices = info['last_op_indices']
        last_site_indices = info['last_site_indices']
        
        M = active_agents.shape[1] # 最大飞机数 n_agents
        agent_types = self._resolve_agent_types(data['graph'], info, bsz, M, global_emb.device)
        n_sites = data['graph'].site_mask_matrix.shape[-1]
        n_requests = data['graph'].request_mask_matrix.shape[-1]
        if op_nodes.shape[1] % self.max_plane_agents != 0:
            raise RuntimeError(
                f"Operation node count {op_nodes.shape[1]} is not divisible by "
                f"max_plane_agents={self.max_plane_agents}."
            )
        n_jobs = op_nodes.shape[1] // self.max_plane_agents
        op_valid_mask = data['graph'].op_mask.clone().view(bsz, M, -1)
        site_mask_matrix = data['graph'].site_mask_matrix.clone().view(bsz, M, n_sites)
        job_site_mask_matrix = data['graph'].job_site_mask_matrix.clone().view(
            bsz,
            n_jobs,
            n_sites,
        )
        if agent_job_site_mask_matrix is None:
            agent_job_site_mask_matrix = job_site_mask_matrix.unsqueeze(1).expand(
                -1,
                M,
                -1,
                -1,
            )
        else:
            agent_job_site_mask_matrix = agent_job_site_mask_matrix.clone().view(
                bsz,
                M,
                n_jobs,
                n_sites,
            )
        request_mask_matrix = data['graph'].request_mask_matrix.clone().view(bsz, M, n_requests)
        raw_request_is_lookahead = getattr(
            data['graph'], 'request_is_lookahead', None
        )
        if raw_request_is_lookahead is None:
            request_is_lookahead = torch.zeros(
                (bsz, n_requests),
                dtype=torch.bool,
                device=global_emb.device,
            )
        else:
            request_is_lookahead = raw_request_is_lookahead.to(
                device=global_emb.device, dtype=torch.bool
            ).view(bsz, n_requests)
        raw_pair_features = getattr(data['graph'], 'pair_features', None)
        raw_sparse_pair_values = getattr(
            data['graph'], 'pair_feature_values', None
        )
        raw_sparse_pair_flat_ids = getattr(
            data['graph'], 'pair_feature_flat_ids', None
        )
        raw_sparse_pair_counts = getattr(
            data['graph'], 'pair_feature_counts', None
        )
        if self.plane_pair_decoder == 'joint_pair':
            has_dense_pair_features = raw_pair_features is not None
            sparse_parts = (
                raw_sparse_pair_values,
                raw_sparse_pair_flat_ids,
                raw_sparse_pair_counts,
            )
            has_sparse_pair_features = any(
                value is not None for value in sparse_parts
            )
            if has_dense_pair_features and has_sparse_pair_features:
                raise RuntimeError(
                    'Graph cannot contain both dense and sparse pair features.'
                )
            if has_dense_pair_features:
                if raw_pair_features.shape[0] % bsz != 0:
                    raise RuntimeError(
                        'Dense pair feature rows are not divisible by the '
                        f'graph batch size: rows={raw_pair_features.shape[0]}, '
                        f'batch={bsz}.'
                    )
                pair_feature_agent_rows = raw_pair_features.shape[0] // bsz
                if pair_feature_agent_rows < min(self.max_plane_agents, M):
                    raise RuntimeError(
                        'Dense pair features do not contain every plane row: '
                        f'rows_per_graph={pair_feature_agent_rows}.'
                    )
                pair_features = raw_pair_features.view(
                    bsz,
                    pair_feature_agent_rows,
                    n_jobs,
                    n_sites,
                    -1,
                )
                sparse_pair_values = None
                sparse_pair_graph_ids = None
                sparse_pair_agent_ids = None
                sparse_pair_local_flat_ids = None
            else:
                if not all(value is not None for value in sparse_parts):
                    raise RuntimeError(
                        'joint_pair decoder requires either graph.pair_features '
                        'or the complete sparse pair feature representation.'
                    )
                sparse_pair_counts = raw_sparse_pair_counts.long().view(-1)
                if sparse_pair_counts.numel() != bsz:
                    raise RuntimeError(
                        'Sparse pair feature counts do not match graph batch: '
                        f'counts={sparse_pair_counts.numel()}, batch={bsz}.'
                    )
                sparse_pair_values = raw_sparse_pair_values.view(
                    -1, self.actor.pair_feature_dim
                )
                sparse_pair_flat_ids = raw_sparse_pair_flat_ids.long().view(-1)
                pair_span = n_jobs * n_sites
                sparse_pair_graph_ids = torch.repeat_interleave(
                    torch.arange(bsz, device=global_emb.device),
                    sparse_pair_counts,
                )
                expected_sparse_rows = sparse_pair_graph_ids.numel()
                if (
                    sparse_pair_values.shape[0] != expected_sparse_rows
                    or sparse_pair_flat_ids.numel() != expected_sparse_rows
                ):
                    raise RuntimeError(
                        'Sparse pair feature payload does not match counts: '
                        f'values={sparse_pair_values.shape[0]}, '
                        f'ids={sparse_pair_flat_ids.numel()}, '
                        f'expected={expected_sparse_rows}.'
                    )
                sparse_pair_agent_ids = sparse_pair_flat_ids // pair_span
                sparse_pair_local_flat_ids = sparse_pair_flat_ids % pair_span
                valid_sparse_ids = (
                    (sparse_pair_agent_ids >= 0)
                    & (sparse_pair_agent_ids < min(self.max_plane_agents, M))
                )
                if not valid_sparse_ids.all():
                    raise RuntimeError(
                        'Sparse pair features contain a non-plane agent id.'
                    )
                pair_features = None
        else:
            pair_features = None
            sparse_pair_values = None
            sparse_pair_graph_ids = None
            sparse_pair_agent_ids = None
            sparse_pair_local_flat_ids = None

        # In learned mode the exact sampled plane rank is stored as the third
        # action component, keeping structured PPO replay exactly on-policy.
        decision_order = torch.arange(
            M, dtype=torch.long, device=global_emb.device
        ).view(1, M).expand(bsz, -1).clone()
        plane_order_rank = torch.full(
            (bsz, M), -1, dtype=torch.long, device=global_emb.device
        )
        plane_order_log_prob = torch.zeros(
            (bsz, M), dtype=global_emb.dtype, device=global_emb.device
        )
        plane_order_entropy = torch.zeros_like(plane_order_log_prob)
        plane_order_trainable = torch.zeros(
            (bsz, M), dtype=torch.bool, device=global_emb.device
        )
        if self.plane_order_actor is not None:
            plane_count = min(self.max_plane_agents, M)
            plane_contexts = []
            for plane_idx in range(plane_count):
                start = plane_idx * n_jobs
                end = start + n_jobs
                local_nodes = op_nodes[:, start:end, :]
                local_mask = op_valid_mask[:, plane_idx, start:end]
                local_count = local_mask.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1)
                plane_contexts.append(
                    (
                        local_nodes
                        * local_mask.unsqueeze(-1).to(local_nodes.dtype)
                    ).sum(dim=1) / local_count.to(local_nodes.dtype)
                )
            plane_contexts = torch.stack(plane_contexts, dim=1)
            active_planes = active_agents[:, :plane_count].bool()
            replay_ranks = (
                chosen_order[:, :plane_count]
                if chosen_order is not None else None
            )
            (
                order_by_rank,
                rank_by_plane,
                order_log_prob,
                order_entropy,
                order_trainable,
            ) = self.plane_order_actor(
                global_emb=global_emb,
                plane_nodes=plane_contexts,
                active_mask=active_planes,
                deterministic=(deterministic or criticize_only),
                chosen_ranks=replay_ranks,
                tau=self.tau,
            )
            decision_order[:, :plane_count] = order_by_rank
            plane_order_rank[:, :plane_count] = rank_by_plane
            plane_order_log_prob[:, :plane_count] = order_log_prob
            plane_order_entropy[:, :plane_count] = order_entropy
            plane_order_trainable[:, :plane_count] = order_trainable
        
        op_choice = -torch.ones((bsz, M), device=global_emb.device, dtype=torch.long)
        site_choice = -torch.ones((bsz, M), device=global_emb.device, dtype=torch.long)
        log_prob = torch.zeros((bsz, M), device=global_emb.device)
        pair_log_prob = torch.zeros_like(log_prob)
        pair_decision_trainable = torch.zeros(
            (bsz, M), dtype=torch.bool, device=global_emb.device
        )
        value_mask = torch.zeros((bsz, M), dtype=torch.bool, device=global_emb.device)
        decision_validated = torch.zeros((bsz, M), dtype=torch.bool, device=global_emb.device)
        decision_trainable = torch.zeros((bsz, M), dtype=torch.bool, device=global_emb.device)
        value = torch.zeros((bsz, M), device=global_emb.device)
        
        # 继承并克隆 GRU 的历史记忆
        team_value = None
        if criticize and self.central_team_critic:
            critic_global = global_emb.detach() if criticize_only else global_emb
            team_value = self.team_critic(critic_global).view(-1)

        new_hidden_state = data['hidden_states'].clone()
        
        # 用于累加信息熵
        total_entropy = torch.tensor(0.0, device=global_emb.device)
        entropy_counts = torch.zeros(
            (), dtype=global_emb.dtype, device=global_emb.device
        )

        # ========================================================
        # 3. 遍历智能体 (飞机) 进行串行自回归决策
        if (eval_action or dist_only) and self.plane_order_actor is not None:
            total_entropy = total_entropy + plane_order_entropy[
                plane_order_trainable
            ].sum()
            entropy_counts += plane_order_trainable.sum().to(
                dtype=global_emb.dtype
            )

        # ========================================================
        # Fixed-order policies used to execute the generic per-batch
        # permutation loop below.  With 24 planes plus 80 resource slots that
        # meant M*M=10,816 tiny CUDA mask kernels and host synchronizations per
        # forward even though only the M diagonal entries can ever be active.
        # Resolve the globally active fixed-order agents once and preserve the
        # exact ascending autoregressive order.  Learned ordering retains the
        # general path because different batch rows may select different
        # agents at the same rank.
        if self.plane_order_actor is None:
            fixed_active_mask = active_agents.bool()
            type_min = torch.where(
                fixed_active_mask,
                agent_types,
                torch.full_like(agent_types, 3),
            ).amin(dim=0)
            type_max = torch.where(
                fixed_active_mask,
                agent_types,
                torch.full_like(agent_types, -1),
            ).amax(dim=0)
            fixed_metadata = torch.stack(
                [
                    fixed_active_mask.any(dim=0).long(),
                    type_min,
                    type_max,
                ],
                dim=-1,
            ).detach().cpu().tolist()
            active_fixed_agents = [
                agent_idx
                for agent_idx, metadata in enumerate(fixed_metadata)
                if metadata[0]
            ]
            fixed_agent_roles = {
                agent_idx: metadata[1]
                for agent_idx, metadata in enumerate(fixed_metadata)
                if metadata[0] and metadata[1] == metadata[2]
            }
            decision_iterations = (
                (agent_idx, agent_idx) for agent_idx in active_fixed_agents
            )
        else:
            fixed_agent_roles = {}
            decision_iterations = (
                (iteration_idx // M, iteration_idx % M)
                for iteration_idx in range(M * M)
            )

        for decision_rank, agent_idx in decision_iterations:
            active_mask_i = active_agents[:, agent_idx].view(-1).bool()
            if self.plane_order_actor is not None:
                active_mask_i = (
                    active_mask_i
                    & (decision_order[:, decision_rank] == agent_idx)
                )
                if not active_mask_i.any():
                    continue
                
            value_mask[:, agent_idx] = active_mask_i
            is_device_agent = agent_idx >= self.max_plane_agents

            if is_device_agent:
                device_local_idx = agent_idx - self.max_plane_agents
                if device_local_idx >= device_nodes.shape[1]:
                    raise RuntimeError(
                        f"Active device agent {agent_idx} has no matching device node "
                        f"(available device nodes: {device_nodes.shape[1]})."
                    )

                active_batch_indices = torch.nonzero(active_mask_i).squeeze(1)
                active_count = active_batch_indices.shape[0]
                last_req_idx = last_op_indices[:, agent_idx].view(-1)[active_mask_i]
                valid_last_req = (last_req_idx >= 0) & (last_req_idx < request_nodes.shape[1])
                safe_last_req_idx = torch.clamp(last_req_idx, min=0, max=max(0, request_nodes.shape[1] - 1))

                last_req_emb = request_nodes[active_mask_i, safe_last_req_idx, :]
                last_req_emb = last_req_emb * valid_last_req.unsqueeze(-1).float()
                last_selection_emb = last_req_emb.unsqueeze(1)

                dev_emb = device_nodes[active_mask_i, device_local_idx, :].unsqueeze(1)

                cur_chosen_req = chosen_op[:, agent_idx].view(-1)[active_mask_i] if chosen_op is not None else None
                cur_req_mask = request_mask_matrix[:, agent_idx, :].clone()
                current_real_mask = cur_req_mask[:, 1:]
                if agent_idx + 1 < M:
                    later_real_mask = request_mask_matrix[:, agent_idx + 1:, 1:].any(dim=1)
                else:
                    later_real_mask = torch.zeros_like(current_real_mask)
                # Lookahead requests are explicitly deferrable even for the
                # last compatible device.  Only already-blocking requests may
                # force no-op out of the autoregressive action mask.  This is
                # the exact network counterpart of env._sequential_device_options.
                blocking_real_mask = current_real_mask & ~request_is_lookahead[:, 1:]
                last_chance_mask = blocking_real_mask & ~later_real_mask
                has_last_chance = last_chance_mask.any(dim=-1)
                cur_req_mask[:, 0] = ~has_last_chance
                cur_req_mask[:, 1:] = torch.where(
                    has_last_chance.unsqueeze(-1),
                    last_chance_mask,
                    current_real_mask,
                )
                request_mask_matrix[:, agent_idx, :] = cur_req_mask
                cur_req = torch.zeros(active_count, dtype=torch.long, device=global_emb.device)
                cur_log_prob = torch.zeros(active_count, dtype=global_emb.dtype, device=global_emb.device)
                cur_dist = torch.full(
                    (active_count, n_requests),
                    float('-inf'),
                    dtype=global_emb.dtype,
                    device=global_emb.device,
                )
                cur_value = torch.zeros(active_count, dtype=global_emb.dtype, device=global_emb.device)
                cur_selected_valid = torch.zeros(
                    active_count,
                    dtype=torch.bool,
                    device=global_emb.device,
                )

                fixed_role = fixed_agent_roles.get(agent_idx)
                if fixed_role == self.AGENT_TYPE_DEVICE:
                    role_routes = ((
                        None,
                        self.device_sel_enc,
                        self.device_actor,
                        self.device_critic,
                        'ordinary',
                    ),)
                elif fixed_role == self.AGENT_TYPE_TRANSPORTER:
                    role_routes = ((
                        None,
                        self.transporter_sel_enc,
                        self.transporter_actor,
                        self.transporter_critic,
                        'transporter',
                    ),)
                else:
                    is_transporter = (
                        agent_types[:, agent_idx].view(-1)[active_mask_i]
                        == self.AGENT_TYPE_TRANSPORTER
                    )
                    role_routes = (
                        (
                            (~is_transporter).view(-1).bool(),
                            self.device_sel_enc,
                            self.device_actor,
                            self.device_critic,
                            'ordinary',
                        ),
                        (
                            is_transporter.view(-1).bool(),
                            self.transporter_sel_enc,
                            self.transporter_actor,
                            self.transporter_critic,
                            'transporter',
                        ),
                    )
                for role_mask, sel_enc, actor_head, critic_head, role_name in role_routes:
                    if role_mask is None:
                        role_index = slice(None)
                    else:
                        if not role_mask.any():
                            continue
                        role_index = role_mask

                    role_batch_indices = active_batch_indices[role_index]
                    hidden_state_i = (
                        data['hidden_states'][role_batch_indices, agent_idx:agent_idx+1, :]
                        .squeeze(1)
                        .transpose(0, 1)
                    )
                    seq_embed, slice_embed, new_hidden_state_i = sel_enc(
                        selection=last_selection_emb[role_index],
                        veh=dev_emb[role_index],
                        hidden_state=hidden_state_i,
                    )
                    new_hidden_state[role_batch_indices, agent_idx:agent_idx+1, :] = (
                        new_hidden_state_i.transpose(0, 1).unsqueeze(1)
                    )
                    role_query = torch.cat([
                        global_emb[role_batch_indices],
                        seq_embed.squeeze(1),
                        slice_embed.squeeze(1),
                    ], dim=-1)
                    role_req_mask = cur_req_mask[role_batch_indices]
                    role_chosen = (
                        cur_chosen_req[role_index]
                        if cur_chosen_req is not None else None
                    )
                    role_decision_trainable = role_req_mask.sum(dim=-1) > 1
                    decision_trainable[role_batch_indices, agent_idx] = role_decision_trainable

                    try:
                        if actor_grad:
                            role_req, role_log_prob, role_dist = actor_head(
                                query=role_query.unsqueeze(1),
                                request_nodes=request_nodes[role_batch_indices],
                                request_valid_mask=role_req_mask,
                                deterministic=deterministic,
                                tau=self.tau,
                                chosen_request=role_chosen,
                            )
                        else:
                            with torch.no_grad():
                                role_req, role_log_prob, role_dist = actor_head(
                                    query=role_query.unsqueeze(1),
                                    request_nodes=request_nodes[role_batch_indices],
                                    request_valid_mask=role_req_mask,
                                    deterministic=deterministic,
                                    tau=self.tau,
                                    chosen_request=role_chosen,
                                )
                    except RuntimeError as error:
                        if role_chosen is None:
                            raise
                        bad = ~role_req_mask.gather(
                            1, role_chosen.long().unsqueeze(-1)
                        ).squeeze(-1)
                        details = [
                            {
                                'batch': int(role_batch_indices[index]),
                                'chosen_request': int(role_chosen[index]),
                                'legal_requests': torch.nonzero(
                                    role_req_mask[index], as_tuple=False
                                ).flatten().detach().cpu().tolist(),
                            }
                            for index in torch.nonzero(
                                bad, as_tuple=False
                            ).flatten().detach().cpu().tolist()
                        ]
                        raise RuntimeError(
                            'Resource replay mask mismatch at '
                            f'agent={agent_idx}, role='
                            f'{role_name}, '
                            f'details={details[:8]}'
                        ) from error

                    selected_req_valid = role_req_mask.gather(
                        1,
                        role_req.unsqueeze(-1),
                    ).squeeze(-1)

                    cur_req[role_index] = role_req
                    cur_log_prob[role_index] = role_log_prob
                    cur_dist[role_index] = role_dist
                    cur_selected_valid[role_index] = selected_req_valid

                    if eval_action or dist_only:
                        dist_req = torch.distributions.Categorical(logits=role_dist)
                        role_entropy = dist_req.entropy()
                        total_entropy += role_entropy[role_decision_trainable].sum()
                        entropy_counts += role_decision_trainable.sum().to(
                            dtype=global_emb.dtype
                        )

                    if criticize:
                        cur_req_pad_mask = ~role_req_mask
                        critic_query = role_query.detach() if criticize_only else role_query
                        critic_requests = (
                            request_nodes[role_batch_indices].detach()
                            if criticize_only else request_nodes[role_batch_indices]
                        )
                        critic_sites = (
                            site_nodes[role_batch_indices].detach()
                            if criticize_only else site_nodes[role_batch_indices]
                        )
                        cur_value[role_index] = critic_head(
                            query=critic_query.unsqueeze(1),
                            op_nodes=critic_requests,
                            site_nodes=critic_sites,
                            op_pad_mask=cur_req_pad_mask,
                            site_pad_mask=None,
                        ).view(-1)

                op_choice[active_mask_i, agent_idx] = cur_req
                site_choice[active_mask_i, agent_idx] = 0
                log_prob[active_mask_i, agent_idx] = cur_log_prob
                decision_validated[active_mask_i, agent_idx] = (
                    cur_selected_valid
                )

                if criticize:
                    value[active_mask_i, agent_idx] = cur_value

                batch_indices = active_batch_indices
                real_req_mask = cur_req > 0
                affected_batches = batch_indices[real_req_mask]
                request_mask_matrix[
                    affected_batches,
                    :,
                    cur_req[real_req_mask],
                ] = False

                continue
            
            # ---------------------------------------------------------
            # A. 提取 GRU 的输入 1：上一次的选择 (刚做完的工序)
            # ---------------------------------------------------------
            op_start = agent_idx * n_jobs
            op_end = op_start + n_jobs
            cur_op_nodes = op_nodes[active_mask_i, op_start:op_end, :]
            cur_agent_mask = op_valid_mask[:, agent_idx, op_start:op_end]

            last_op_idx = last_op_indices[:, agent_idx].view(-1)[active_mask_i]
            safe_last_op_idx, valid_last_op = self._local_plane_history_index(
                last_op_idx,
                op_start,
                n_jobs,
            )
            
            # 抽取高阶特征并处理冷启动
            last_op_emb = cur_op_nodes[
                torch.arange(cur_op_nodes.shape[0], device=cur_op_nodes.device),
                safe_last_op_idx,
                :,
            ]
            last_op_emb = last_op_emb * valid_last_op.unsqueeze(-1).float()

            last_site_idx = last_site_indices[:, agent_idx].view(-1)[active_mask_i]
            valid_last_site = (last_site_idx >= 0) & (last_site_idx < n_sites)
            safe_last_site_idx = torch.clamp(
                last_site_idx,
                min=0,
                max=max(0, n_sites - 1),
            )
            last_site_emb = site_nodes[
                active_mask_i,
                safe_last_site_idx,
                :,
            ]
            last_site_emb = last_site_emb * valid_last_site.unsqueeze(-1).float()
            history_count = (
                valid_last_op.float() + valid_last_site.float()
            ).clamp_min(1.0).unsqueeze(-1)
            last_selection_emb = (
                (last_op_emb + last_site_emb) / history_count
            ).unsqueeze(1)

            # ---------------------------------------------------------
            # B. 提取 GRU 的输入 2：当前飞机的诉求 (当前可用工序上下文)
            # ---------------------------------------------------------
            cur_site_mask = (
                agent_job_site_mask_matrix[:, agent_idx, :, :]
                & site_mask_matrix[:, agent_idx, :].unsqueeze(1)
            )
            joint_choice_count = (
                cur_agent_mask[active_mask_i].unsqueeze(-1)
                & cur_site_mask[active_mask_i]
            ).reshape(cur_op_nodes.shape[0], -1).sum(dim=-1)
            pair_decision_trainable[active_mask_i, agent_idx] = (
                joint_choice_count > 1
            )
            decision_trainable[active_mask_i, agent_idx] = (
                pair_decision_trainable[active_mask_i, agent_idx]
                | plane_order_trainable[active_mask_i, agent_idx]
            )
            valid_ops_emb = cur_op_nodes * cur_agent_mask[active_mask_i].unsqueeze(-1).float()
            
            counts = cur_agent_mask[active_mask_i].sum(dim=1, keepdim=True).clamp(min=1e-5)
            agent_context = valid_ops_emb.sum(dim=1) / counts
            veh_emb = agent_context.unsqueeze(1) # [B_active, 1, Embed_Dim]

            # ---------------------------------------------------------
            # C. 过 GRU 记忆更新并拼接 Query
            # ---------------------------------------------------------
            if new_hidden_state is not None:
                hidden_state_i = data['hidden_states'][active_mask_i, agent_idx:agent_idx+1, :].squeeze(1).transpose(0, 1)
                
                seq_embed, slice_embed, new_hidden_state_i = self.plane_sel_enc(
                    selection=last_selection_emb, 
                    veh=veh_emb, 
                    hidden_state=hidden_state_i
                )
                
                new_hidden_state[active_mask_i, agent_idx:agent_idx+1, :] = new_hidden_state_i.transpose(0, 1).unsqueeze(1)
                
                query = torch.cat([global_emb[active_mask_i], seq_embed.squeeze(1), slice_embed.squeeze(1)], dim=-1)
            else:
                query = torch.cat([global_emb[active_mask_i], veh_emb.squeeze(1)], dim=-1) # fallback

            # ---------------------------------------------------------
            # D. PPO 强制动作注入 (算 Loss 用)
            # ---------------------------------------------------------
            cur_chosen_op = None
            if chosen_op is not None:
                cur_chosen_op = (
                    chosen_op[:, agent_idx].view(-1)[active_mask_i] - op_start
                )
            cur_chosen_site = chosen_site[:, agent_idx].view(-1)[active_mask_i] if chosen_site is not None else None
            pair_actor_kwargs = {}
            if self.plane_pair_decoder == 'joint_pair':
                if pair_features is not None:
                    pair_actor_kwargs['pair_features'] = pair_features[
                        active_mask_i, agent_idx
                    ]
                else:
                    active_batch_indices = torch.nonzero(
                        active_mask_i, as_tuple=False
                    ).flatten()
                    batch_to_local = torch.full(
                        (bsz,),
                        -1,
                        dtype=torch.long,
                        device=global_emb.device,
                    )
                    batch_to_local[active_batch_indices] = torch.arange(
                        active_batch_indices.numel(),
                        dtype=torch.long,
                        device=global_emb.device,
                    )
                    sparse_selection = (
                        (sparse_pair_agent_ids == agent_idx)
                        & active_mask_i[sparse_pair_graph_ids]
                    )
                    selected_graph_ids = sparse_pair_graph_ids[
                        sparse_selection
                    ]
                    pair_actor_kwargs.update({
                        'pair_feature_values': sparse_pair_values[
                            sparse_selection
                        ],
                        'pair_feature_batch_indices': batch_to_local[
                            selected_graph_ids
                        ],
                        'pair_feature_flat_ids': (
                            sparse_pair_local_flat_ids[sparse_selection]
                        ),
                    })
            # ---------------------------------------------------------
            # E. 级联 Actor 决策
            # ---------------------------------------------------------
            if actor_grad:
                cur_op, cur_site, cur_log_prob, cur_dist = self.actor(
                    query=query.unsqueeze(1), 
                    op_nodes=cur_op_nodes,
                    site_nodes=site_nodes[active_mask_i], 
                    op_valid_mask=cur_agent_mask[active_mask_i], 
                    site_valid_mask=cur_site_mask[active_mask_i], 
                    deterministic=deterministic, 
                    tau=self.tau,
                    chosen_op=cur_chosen_op, 
                    chosen_site=cur_chosen_site,
                    **pair_actor_kwargs,
                )
            else:
                with torch.no_grad():
                    cur_op, cur_site, cur_log_prob, cur_dist = self.actor(
                        query=query.unsqueeze(1),
                        op_nodes=cur_op_nodes,
                        site_nodes=site_nodes[active_mask_i],
                        op_valid_mask=cur_agent_mask[active_mask_i],
                        site_valid_mask=cur_site_mask[active_mask_i],
                        deterministic=deterministic,
                        chosen_op=cur_chosen_op,
                        chosen_site=cur_chosen_site,
                        tau=self.tau,
                        **pair_actor_kwargs,
                    )
            selected_pair_valid = cur_agent_mask[active_mask_i].gather(
                1,
                cur_op.unsqueeze(-1),
            ).squeeze(-1)
            selected_pair_valid &= cur_site_mask[active_mask_i][
                torch.arange(cur_op.shape[0], device=cur_op.device),
                cur_op,
                cur_site,
            ]
            # 记录决策动作与概率
            op_choice[active_mask_i, agent_idx] = cur_op + op_start
            site_choice[active_mask_i, agent_idx] = cur_site
            pair_log_prob[active_mask_i, agent_idx] = cur_log_prob
            log_prob[active_mask_i, agent_idx] = (
                pair_log_prob[active_mask_i, agent_idx]
                + plane_order_log_prob[active_mask_i, agent_idx]
            )
            decision_validated[active_mask_i, agent_idx] = selected_pair_valid

            # 累加信息熵 (评估阶段专用)
            if eval_action or dist_only:
                joint_logits = cur_dist # 这是刚从 Actor 返回的 [B, N_ops * N_sites]
                # 直接使用 logits 构建分布，计算联合熵 H(X,Y)
                dist_joint = torch.distributions.Categorical(logits=joint_logits)
                plane_decision_trainable = joint_choice_count > 1
                total_entropy += dist_joint.entropy()[plane_decision_trainable].sum()
                entropy_counts += plane_decision_trainable.sum().to(
                    dtype=global_emb.dtype
                )

            # ---------------------------------------------------------
            # F. Critic 价值评估
            # ---------------------------------------------------------
            if criticize and team_value is None:
                cur_op_pad_mask = ~cur_agent_mask[active_mask_i] 
                cur_site_pad_mask = None # 全局资源视角
                critic_query = query.detach() if criticize_only else query
                critic_ops = cur_op_nodes.detach() if criticize_only else cur_op_nodes
                critic_sites = (
                    site_nodes[active_mask_i].detach()
                    if criticize_only else site_nodes[active_mask_i]
                )
                
                # 直接使用 .view(-1) 强制展平为 1D 向量
                value[active_mask_i, agent_idx] = self.plane_critic(
                    query=critic_query.unsqueeze(1),
                    op_nodes=critic_ops,
                    site_nodes=critic_sites,
                    op_pad_mask=cur_op_pad_mask,
                    site_pad_mask=cur_site_pad_mask
                ).view(-1)

            # ---------------------------------------------------------
            # G. 自回归动态掩码刷新 (避免同一个 Batch 互相抢资源)
            # ---------------------------------------------------------
            batch_indices = torch.nonzero(active_mask_i).squeeze(1)
            # 刚才被选走的工序不能再选
            # op_valid_mask[batch_indices, agent_idx, cur_op] = False
            # 刚才被分配的机位不能再选
            site_mask_matrix[batch_indices, :, cur_site] = False


        if criticize and team_value is not None:
            value = torch.where(
                value_mask,
                team_value.unsqueeze(1).expand(-1, M),
                value,
            )

        # ========================================================
        # 4. 返回值格式化
        # ========================================================
        missing_decisions = active_agents.bool() & ~decision_validated
        if missing_decisions.any():
            bad_entries = torch.nonzero(missing_decisions, as_tuple=False).tolist()
            raise RuntimeError(
                f"Active agents reached policy output without a validated legal action: "
                f"{bad_entries[:16]}."
            )

        # Actor 直接返回与实际 Categorical 采样一致的 log-prob。
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=-20.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 兼容旧代码，将两个动作打在一个 Tensor 的最后一个维度里
        if self.plane_order_actor is None:
            action = torch.stack([op_choice, site_choice], dim=2)
        else:
            action = torch.stack(
                [op_choice, site_choice, plane_order_rank],
                dim=2,
            )
        
        # 计算平均熵
        ent = torch.nan_to_num(
            total_entropy / entropy_counts.clamp_min(1.0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if criticize_only:
            return value
        if dist_only:
            # 级联网络没有单一的 Categorical 分布，通常返回 Entropy 即可供 PPO 更新
            return None # 或者自定义联合分布返回
        if criticize:
            if return_decision_mask:
                return value, action, log_prob, new_hidden_state, decision_trainable
            return value, action, log_prob, new_hidden_state
        if eval_action:
            if return_log_prob_components:
                components = {
                    'pair_log_probs': pair_log_prob,
                    'order_log_probs': plane_order_log_prob,
                    'pair_decision_mask': pair_decision_trainable,
                    'order_decision_mask': plane_order_trainable,
                }
                if return_decision_mask:
                    outputs = (
                        log_prob,
                        ent,
                        decision_trainable,
                        components,
                    )
                else:
                    outputs = (log_prob, ent, components)
                return (
                    (*outputs, new_hidden_state)
                    if return_rnn_states else outputs
                )
            if return_decision_mask:
                outputs = (log_prob, ent, decision_trainable)
            else:
                outputs = (log_prob, ent)
            return (
                (*outputs, new_hidden_state)
                if return_rnn_states else outputs
            )
            
        return action, new_hidden_state
