import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
import math
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
from onpolicy.utils.stage2_matching import solve_resource_matching
from onpolicy.algorithms.utils.stage1_baselines import (
    build_stage1_baseline_actor,
    build_stage1_baseline_encoder,
    normalize_stage1_baseline,
)

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


class RequestReadyTimeHead(nn.Module):
    """Predict the physical lead time of every real resource request.

    The dependency graph already supplies a conservative lower bound.  The
    network therefore predicts only a non-negative residual, which prevents a
    learned estimate from claiming that an operation can become ready before
    its known predecessors finish.
    """

    KIND_COUNT = 5
    BLOCKING_KIND = 3
    EXPLICIT_CONTEXT_DIM = KIND_COUNT + 6

    def __init__(
        self,
        embed_dim,
        time_scale,
        *,
        context_features=False,
        head_mode='shared',
        quantile_head=False,
        device,
        dtype,
    ):
        super().__init__()
        self.time_scale = float(time_scale)
        self.context_features = bool(context_features)
        self.head_mode = str(head_mode)
        self.quantile_head = bool(quantile_head)
        if self.time_scale <= 0.0:
            raise ValueError("request ready time_scale must be positive")
        if self.head_mode not in {'shared', 'horizon_split'}:
            raise ValueError(
                'request ready head_mode must be shared or horizon_split'
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        input_dim = 4 * int(embed_dim) + 1
        if self.context_features:
            input_dim += self.EXPLICIT_CONTEXT_DIM
        output_dim = 3 if self.quantile_head else 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(embed_dim), **factory_kwargs),
            nn.SiLU(),
            nn.LayerNorm(int(embed_dim), **factory_kwargs),
            nn.Linear(int(embed_dim), output_dim, **factory_kwargs),
        )
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Start almost exactly at the dependency lower bound.  This is a safe
        # Stage-1 -> Stage-2 boundary and still leaves a finite gradient for
        # supervised residual learning.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -10.0)
        self.kind_outputs = None
        if self.head_mode == 'horizon_split':
            # The common trunk learns timing context while each semantic
            # horizon owns its final calibration.  Keeping ``network`` as the
            # trunk also preserves the default head's historical state names.
            self.network = self.network[:-1]
            self.kind_outputs = nn.ModuleList([
                nn.Linear(
                    int(embed_dim), output_dim, **factory_kwargs
                )
                for _ in range(self.KIND_COUNT)
            ])
            for projection in self.kind_outputs:
                nn.init.zeros_(projection.weight)
                nn.init.constant_(projection.bias, -10.0)

    def forward(
        self,
        request_nodes,
        operation_nodes,
        site_nodes,
        global_embedding,
        dag_lead_seconds,
        *,
        explicit_context=None,
        kind_ids=None,
        hard_blocking=False,
        return_quantiles=False,
    ):
        scaled_dag = torch.log1p(
            dag_lead_seconds.clamp_min(0.0) / self.time_scale
        ).unsqueeze(-1)
        parts = [
            request_nodes,
            operation_nodes,
            site_nodes,
            global_embedding,
            scaled_dag,
        ]
        if self.context_features:
            if explicit_context is None:
                raise RuntimeError(
                    'explicit request-ready context is enabled but missing'
                )
            if explicit_context.shape[:-1] != request_nodes.shape[:-1]:
                raise RuntimeError(
                    'explicit request-ready context has an incompatible shape: '
                    f'{tuple(explicit_context.shape)} vs '
                    f'{tuple(request_nodes.shape)}'
                )
            if explicit_context.shape[-1] != self.EXPLICIT_CONTEXT_DIM:
                raise RuntimeError(
                    'explicit request-ready context width differs from the '
                    f'contract: {explicit_context.shape[-1]} != '
                    f'{self.EXPLICIT_CONTEXT_DIM}'
                )
            parts.append(explicit_context.to(dtype=request_nodes.dtype))
        context = torch.cat(parts, dim=-1)
        hidden_or_raw = self.network(context)
        if self.kind_outputs is None:
            raw = hidden_or_raw
        else:
            if kind_ids is None:
                raise RuntimeError('horizon_split ready head requires kind ids')
            safe_kind_ids = kind_ids.long().clamp(0, self.KIND_COUNT - 1)
            all_outputs = torch.stack(
                [projection(hidden_or_raw) for projection in self.kind_outputs],
                dim=-2,
            )
            raw = torch.gather(
                all_outputs,
                -2,
                safe_kind_ids.unsqueeze(-1).unsqueeze(-1).expand(
                    *safe_kind_ids.shape, 1, all_outputs.shape[-1]
                ),
            ).squeeze(-2)

        quantile_predictions = None
        if self.quantile_head:
            # Positive increments make the quantile ordering structural rather
            # than a soft loss that can be violated at inference time.
            increments = self.time_scale * F.softplus(raw)
            q20 = increments[..., 0]
            q50 = q20 + increments[..., 1]
            q80 = q50 + increments[..., 2]
            residual_seconds = q50
            quantile_predictions = torch.stack((q20, q50, q80), dim=-1)
        else:
            residual_seconds = self.time_scale * F.softplus(
                raw.squeeze(-1)
            )
        prediction_seconds = (
            dag_lead_seconds.clamp_min(0.0) + residual_seconds
        )
        if hard_blocking:
            if kind_ids is None:
                raise RuntimeError('hard blocking routing requires kind ids')
            blocking = kind_ids.long() == self.BLOCKING_KIND
            prediction_seconds = torch.where(
                blocking,
                torch.zeros_like(prediction_seconds),
                prediction_seconds,
            )
            residual_seconds = torch.where(
                blocking,
                torch.zeros_like(residual_seconds),
                residual_seconds,
            )
            if quantile_predictions is not None:
                quantile_predictions = torch.where(
                    blocking.unsqueeze(-1),
                    torch.zeros_like(quantile_predictions),
                    quantile_predictions,
                )
        if return_quantiles:
            return prediction_seconds, residual_seconds, quantile_predictions
        return prediction_seconds, residual_seconds


class ResourceResidualAdapter(nn.Module):
    """A checkpoint-safe resource-only view of shared graph embeddings.

    The last projection starts at exactly zero, so enabling the adapter does
    not perturb the frozen Stage1 policy boundary.  Only resource queries and
    request/device tokens consume this residual; aircraft decisions continue
    to use the original shared-encoder tensors.
    """

    def __init__(self, embed_dim, *, device, dtype):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.network = nn.Sequential(
            nn.LayerNorm(int(embed_dim), **factory_kwargs),
            nn.Linear(int(embed_dim), int(embed_dim), **factory_kwargs),
            nn.SiLU(),
            nn.Linear(int(embed_dim), int(embed_dim), **factory_kwargs),
        )
        nn.init.xavier_uniform_(self.network[1].weight)
        nn.init.zeros_(self.network[1].bias)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values):
        return values + self.network(values)

class GNN_Actor_Critic(nn.Module):
    AGENT_TYPE_PLANE = 0
    AGENT_TYPE_DEVICE = 1
    AGENT_TYPE_TRANSPORTER = 2

    def __init__(self, common_cfg=None, encoder_cfg=None, sel_encoder_cfg=None, actor_cfg=None, critic_cfg=None,
                 max_plane_agents=24, max_device_agents=0,
                 plane_order_mode='fixed', plane_pair_decoder='joint_pair',
                 stage1_baseline='proposed',
                 central_team_critic=False, counterfactual_q_baseline=False,
                 counterfactual_q_topk=8, counterfactual_q_min_mass=0.90,
                 counterfactual_baseline_mix=1.0,
                 device_policy_head_mode='shared',
                 ordinary_device_type_count=10,
                 device_timing_head=False,
                 device_global_matching=False,
                 request_ready_prediction=False,
                 request_ready_time_scale=3600.0,
                 request_ready_policy_injection='learned',
                 request_ready_hard_blocking=False,
                 request_ready_context_features=False,
                 request_ready_head_mode='shared',
                 request_ready_quantile_head=False,
                 device_resource_adapter=False,
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
        self.stage1_baseline = normalize_stage1_baseline(stage1_baseline)
        self.central_team_critic = bool(central_team_critic)
        self.counterfactual_q_baseline = bool(counterfactual_q_baseline)
        self.counterfactual_q_topk = int(counterfactual_q_topk)
        self.counterfactual_q_min_mass = float(counterfactual_q_min_mass)
        self.counterfactual_baseline_mix = float(counterfactual_baseline_mix)
        self._counterfactual_represented_masses = []
        if self.counterfactual_q_topk < 1:
            raise ValueError('counterfactual_q_topk must be positive.')
        if not 0.0 <= self.counterfactual_q_min_mass <= 1.0:
            raise ValueError('counterfactual_q_min_mass must be in [0, 1].')
        if not 0.0 <= self.counterfactual_baseline_mix <= 1.0:
            raise ValueError('counterfactual_baseline_mix must be in [0, 1].')
        self.device_policy_head_mode = str(device_policy_head_mode)
        self.ordinary_device_type_count = int(ordinary_device_type_count)
        self.device_timing_head = bool(device_timing_head)
        self.device_global_matching = bool(device_global_matching)
        self.request_ready_prediction = bool(request_ready_prediction)
        self.request_ready_time_scale = float(request_ready_time_scale)
        self.request_ready_policy_injection = str(
            request_ready_policy_injection
        )
        self.request_ready_hard_blocking = bool(
            request_ready_hard_blocking
        )
        self.request_ready_context_features = bool(
            request_ready_context_features
        )
        self.request_ready_head_mode = str(request_ready_head_mode)
        self.request_ready_quantile_head = bool(
            request_ready_quantile_head
        )
        self.device_resource_adapter_enabled = bool(device_resource_adapter)
        if self.request_ready_time_scale <= 0.0:
            raise ValueError('request_ready_time_scale must be positive.')
        if self.request_ready_policy_injection not in {
            'none', 'dag', 'learned'
        }:
            raise ValueError(
                'request_ready_policy_injection must be none, dag or learned, '
                f'got {self.request_ready_policy_injection!r}.'
            )
        if self.device_policy_head_mode not in {
            'shared', 'type_adapter', 'per_type'
        }:
            raise ValueError(
                'device_policy_head_mode must be shared, type_adapter or '
                f'per_type, got {self.device_policy_head_mode!r}.'
            )
        if self.ordinary_device_type_count < 1:
            raise ValueError('ordinary_device_type_count must be positive.')
        if self.counterfactual_q_topk < 1:
            raise ValueError('counterfactual_q_topk must be positive.')
        if not 0.0 < self.counterfactual_q_min_mass <= 1.0:
            raise ValueError('counterfactual_q_min_mass must be in (0, 1].')
        if self.plane_order_mode not in {'fixed', 'learned'}:
            raise ValueError(
                f"Unsupported plane_order_mode={self.plane_order_mode!r}."
            )
        if self.plane_pair_decoder not in {'cascade', 'joint_pair'}:
            raise ValueError(
                f"Unsupported plane_pair_decoder={self.plane_pair_decoder!r}."
            )
        if (
            self.stage1_baseline != 'proposed'
            and self.plane_order_mode != 'fixed'
        ):
            raise ValueError(
                'Stage-1 learning baselines require plane_order_mode=fixed.'
            )
        if (
            self.stage1_baseline != 'proposed'
            and self.plane_pair_decoder != 'joint_pair'
        ):
            raise ValueError(
                'Stage-1 learning baselines require '
                'plane_pair_decoder=joint_pair for the common action contract.'
            )


        # 1. 异构图编码器
        if self.stage1_baseline == 'proposed':
            self.encoder = HeteroGraphEncoder(
                self.common,
                self.encoder_cfg['gnn_cfg'],
                self.encoder_cfg['gff_cfg'],
                **self.factory_kwargs,
            )
        else:
            self.encoder = build_stage1_baseline_encoder(
                self.stage1_baseline,
                self.common,
                self.encoder_cfg,
                **self.factory_kwargs,
            )
        
        # 2. 角色独立 GRU 时序记忆编码器；角色之间除 GNN 外不共享后端参数。
        self.plane_sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)
        self.device_sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)
        self.transporter_sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)
        
        # 3. 角色独立 Actor 动作网络
        if self.stage1_baseline == 'proposed':
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
        else:
            self.actor = build_stage1_baseline_actor(
                self.stage1_baseline,
                self.common,
                self.actor_cfg,
                **self.factory_kwargs,
            )
        self.actor_requires_pair_features = bool(getattr(
            self.actor, 'requires_pair_features',
            self.stage1_baseline == 'proposed'
            and self.plane_pair_decoder == 'joint_pair',
        ))
        self.plane_order_actor = (
            PlaneOrderPointer(**self.common, **self.factory_kwargs)
            if self.plane_order_mode == 'learned' else None
        )
        role_actor_cfg = dict(self.actor_cfg)
        role_actor_cfg.pop('pair_feature_dim', None)
        role_actor_cfg['timing_head'] = self.device_timing_head
        self.device_actor = DeviceRequestPtrActor(
            **self.common, **role_actor_cfg, **self.factory_kwargs
        )
        self.transporter_actor = DeviceRequestPtrActor(
            **self.common, **role_actor_cfg, **self.factory_kwargs
        )

        # One request-level prediction module is shared by ordinary devices
        # and R014.  It reuses the already computed graph embeddings; there is
        # no second GNN forward and no second encoder copy.
        self.request_ready_head = None
        self.request_ready_feature = None
        if self.request_ready_prediction:
            embed_dim = int(self.common['embed_dim'])
            self.request_ready_head = RequestReadyTimeHead(
                embed_dim,
                self.request_ready_time_scale,
                context_features=self.request_ready_context_features,
                head_mode=self.request_ready_head_mode,
                quantile_head=self.request_ready_quantile_head,
                device=device,
                dtype=dtype,
            )
            self.request_ready_feature = nn.Linear(
                2,
                embed_dim,
                bias=False,
                **self.factory_kwargs,
            )
            # The initial Stage2 policy is action-equivalent to the loaded S1
            # model.  Supervised action gradients can then learn how strongly
            # the predicted timing should affect request ranking.
            nn.init.zeros_(self.request_ready_feature.weight)

        self.resource_residual_adapter = None
        if self.device_resource_adapter_enabled:
            self.resource_residual_adapter = ResourceResidualAdapter(
                int(self.common['embed_dim']),
                device=device,
                dtype=dtype,
            )
        
        # 4. 角色独立 Critic 价值网络
        self.plane_critic = StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)
        self.device_critic = StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)
        self.team_critic = GlobalStepCritic(
            **self.common, **self.critic_cfg, **self.factory_kwargs
        )
        self.transporter_critic = StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)

        # The old shared modules remain registered under their historical
        # names so every Stage1/Stage2 checkpoint stays loadable.  New modes
        # are zero/copy initialized and therefore reproduce the shared policy
        # bit-for-bit at the architecture boundary.
        self.device_type_adapter = None
        self.device_type_sel_encs = None
        self.device_type_actors = None
        if self.device_policy_head_mode == 'type_adapter':
            query_dim = int(self.actor_cfg['query_dim'])
            self.device_type_adapter = nn.Embedding(
                self.ordinary_device_type_count,
                query_dim,
                **self.factory_kwargs,
            )
            nn.init.zeros_(self.device_type_adapter.weight)
        elif self.device_policy_head_mode == 'per_type':
            self.device_type_sel_encs = nn.ModuleList([
                deepcopy(self.device_sel_enc)
                for _ in range(self.ordinary_device_type_count)
            ])
            self.device_type_actors = nn.ModuleList([
                deepcopy(self.device_actor)
                for _ in range(self.ordinary_device_type_count)
            ])
        
        self.tau = 1.0
        # Runtime-only memory policy.  It deliberately stays out of ``cfg``
        # and the state dict, so enabling it cannot change checkpoint hashes or
        # the strict Stage2 -> Stage3 architecture contract.
        self.shared_encoder_activation_checkpoint = False

        self.cfg = {
            'common_cfg': self.common,
            'encoder_cfg': self.encoder_cfg,
            'sel_encoder_cfg': self.selection_enc,
            'actor_cfg': self.actor_cfg,
            'critic_cfg': self.critic_cfg,
            'plane_order_mode': self.plane_order_mode,
            'plane_pair_decoder': self.plane_pair_decoder,
            'stage1_baseline': self.stage1_baseline,
            'device_policy_head_mode': self.device_policy_head_mode,
            'ordinary_device_type_count': self.ordinary_device_type_count,
            'device_timing_head': self.device_timing_head,
            'device_global_matching': self.device_global_matching,
            'request_ready_prediction': self.request_ready_prediction,
            'request_ready_time_scale': self.request_ready_time_scale,
            'request_ready_policy_injection': (
                self.request_ready_policy_injection
            ),
            'request_ready_hard_blocking': self.request_ready_hard_blocking,
            'request_ready_context_features': (
                self.request_ready_context_features
            ),
            'request_ready_head_mode': self.request_ready_head_mode,
            'request_ready_quantile_head': self.request_ready_quantile_head,
            'device_resource_adapter': self.device_resource_adapter_enabled,
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
        ordinary_device_modules = (
            [self.device_sel_enc, self.device_actor]
            if self.device_policy_head_mode == 'shared'
            else [
                self.device_sel_enc,
                self.device_actor,
                self.device_type_adapter,
            ]
            if self.device_policy_head_mode == 'type_adapter'
            else [self.device_type_sel_encs, self.device_type_actors]
        )
        if self.request_ready_prediction:
            ordinary_device_modules.extend((
                self.request_ready_head,
                self.request_ready_feature,
            ))
        if self.resource_residual_adapter is not None:
            ordinary_device_modules.append(self.resource_residual_adapter)
        self.actor_param = _ModuleGroup([
            self.encoder,
            self.plane_sel_enc,
            self.transporter_sel_enc,
            self.actor,
            self.transporter_actor,
            *ordinary_device_modules,
            *optional_order,
        ])
        self.actor_param_without_gnn = _ModuleGroup([
            self.plane_sel_enc,
            self.transporter_sel_enc,
            self.actor,
            self.transporter_actor,
            *ordinary_device_modules,
            *optional_order,
        ])
        self.shared_actor_param = _ModuleGroup([self.encoder])
        self.plane_actor_param = _ModuleGroup(plane_actor_modules)
        self.device_actor_param = _ModuleGroup(ordinary_device_modules)
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
    def _reshape_request_metadata(graph, name, bsz, n_requests, *, device, dtype):
        raw = getattr(graph, name, None)
        if raw is None:
            raise RuntimeError(
                f"request ready prediction requires graph.{name}."
            )
        value = raw.to(device=device, dtype=dtype)
        expected = int(bsz) * int(n_requests)
        if value.numel() != expected:
            raise RuntimeError(
                f"graph.{name} has {value.numel()} values; expected "
                f"{expected} ({bsz} graphs x {n_requests} requests)."
            )
        return value.reshape(bsz, n_requests)

    @staticmethod
    def _reshape_request_matrix_metadata(
        graph, name, bsz, n_requests, width, *, device, dtype
    ):
        raw = getattr(graph, name, None)
        if raw is None:
            raise RuntimeError(
                f"request ready prediction requires graph.{name}."
            )
        value = raw.to(device=device, dtype=dtype)
        expected = int(bsz) * int(n_requests) * int(width)
        if value.numel() != expected:
            raise RuntimeError(
                f"graph.{name} has {value.numel()} values; expected "
                f"{expected} ({bsz} graphs x {n_requests} requests x "
                f"{width} features)."
            )
        return value.reshape(bsz, n_requests, width)

    def _augment_requests_with_ready_prediction(
        self,
        graph,
        *,
        global_emb,
        op_nodes,
        site_nodes,
        request_nodes,
    ):
        """Return request embeddings plus auditable timing predictions."""

        if not self.request_ready_prediction:
            return request_nodes, {
                'request_ready_prediction_seconds': None,
                'request_ready_residual_seconds': None,
                'request_ready_dag_seconds': None,
                'request_ready_prediction_valid': None,
                'request_ready_kind_ids': None,
                'request_ready_quantile_seconds': None,
            }
        bsz, n_requests, embed_dim = request_nodes.shape
        device = request_nodes.device
        operation_indices = self._reshape_request_metadata(
            graph,
            'request_operation_indices',
            bsz,
            n_requests,
            device=device,
            dtype=torch.long,
        )
        site_indices = self._reshape_request_metadata(
            graph,
            'request_site_indices',
            bsz,
            n_requests,
            device=device,
            dtype=torch.long,
        )
        dag_seconds = self._reshape_request_metadata(
            graph,
            'request_dag_lead_times',
            bsz,
            n_requests,
            device=device,
            dtype=request_nodes.dtype,
        )
        valid = self._reshape_request_metadata(
            graph,
            'request_prediction_valid',
            bsz,
            n_requests,
            device=device,
            dtype=torch.bool,
        )
        valid &= operation_indices.ge(0) & operation_indices.lt(op_nodes.shape[1])
        valid &= site_indices.ge(0) & site_indices.lt(site_nodes.shape[1])
        raw_kind_ids = getattr(graph, 'request_kind_ids', None)
        if raw_kind_ids is None:
            if (
                self.request_ready_hard_blocking
                or self.request_ready_context_features
                or self.request_ready_head_mode == 'horizon_split'
            ):
                raise RuntimeError(
                    'configured reliable ready head requires '
                    'graph.request_kind_ids'
                )
            kind_ids = torch.zeros(
                (bsz, n_requests), dtype=torch.long, device=device
            )
        else:
            kind_ids = self._reshape_request_metadata(
                graph,
                'request_kind_ids',
                bsz,
                n_requests,
                device=device,
                dtype=torch.long,
            )

        explicit_context = None
        if self.request_ready_context_features:
            raw_context = self._reshape_request_matrix_metadata(
                graph,
                'request_ready_context_values',
                bsz,
                n_requests,
                6,
                device=device,
                dtype=request_nodes.dtype,
            )
            scale = float(self.request_ready_time_scale)
            numeric_context = torch.stack((
                raw_context[..., 0].clamp(0.0, 4.0) / 4.0,
                torch.log1p(raw_context[..., 1].clamp_min(0.0))
                / math.log(16.0),
                torch.log1p(raw_context[..., 2].clamp_min(0.0) / scale),
                torch.log1p(raw_context[..., 3].clamp_min(0.0) / scale),
                torch.log1p(raw_context[..., 4].clamp_min(0.0) / scale),
                torch.log1p(raw_context[..., 5].clamp_min(0.0) / scale),
            ), dim=-1)
            kind_one_hot = F.one_hot(
                kind_ids.clamp(0, RequestReadyTimeHead.KIND_COUNT - 1),
                num_classes=RequestReadyTimeHead.KIND_COUNT,
            ).to(dtype=request_nodes.dtype)
            explicit_context = torch.cat(
                (kind_one_hot, numeric_context), dim=-1
            )

        safe_operation_indices = operation_indices.clamp(
            min=0, max=max(0, op_nodes.shape[1] - 1)
        )
        safe_site_indices = site_indices.clamp(
            min=0, max=max(0, site_nodes.shape[1] - 1)
        )
        operation_context = torch.gather(
            op_nodes,
            1,
            safe_operation_indices.unsqueeze(-1).expand(-1, -1, embed_dim),
        )
        site_context = torch.gather(
            site_nodes,
            1,
            safe_site_indices.unsqueeze(-1).expand(-1, -1, embed_dim),
        )
        global_context = global_emb.unsqueeze(1).expand(-1, n_requests, -1)
        (
            prediction_seconds,
            residual_seconds,
            quantile_residual_seconds,
        ) = self.request_ready_head(
            request_nodes,
            operation_context,
            site_context,
            global_context,
            dag_seconds,
            explicit_context=explicit_context,
            kind_ids=kind_ids,
            hard_blocking=self.request_ready_hard_blocking,
            return_quantiles=True,
        )
        valid_float = valid.to(dtype=request_nodes.dtype)
        prediction_seconds = torch.where(
            valid,
            prediction_seconds,
            torch.zeros_like(prediction_seconds),
        )
        residual_seconds = torch.where(
            valid,
            residual_seconds,
            torch.zeros_like(residual_seconds),
        )
        quantile_seconds = None
        if quantile_residual_seconds is not None:
            quantile_seconds = (
                dag_seconds.clamp_min(0.0).unsqueeze(-1)
                + quantile_residual_seconds
            )
            quantile_seconds = torch.where(
                valid.unsqueeze(-1),
                quantile_seconds,
                torch.zeros_like(quantile_seconds),
            )
            if self.request_ready_hard_blocking:
                quantile_seconds = torch.where(
                    (kind_ids == RequestReadyTimeHead.BLOCKING_KIND).unsqueeze(-1),
                    torch.zeros_like(quantile_seconds),
                    quantile_seconds,
                )
        timing_features = torch.stack(
            (
                torch.log1p(dag_seconds.clamp_min(0.0) / self.request_ready_time_scale),
                torch.log1p(prediction_seconds / self.request_ready_time_scale),
            ),
            dim=-1,
        )
        if self.request_ready_policy_injection == 'none':
            augmented = request_nodes
        else:
            if self.request_ready_policy_injection == 'dag':
                timing_features = timing_features.clone()
                timing_features[..., 1] = 0.0
            augmented = request_nodes + valid_float.unsqueeze(-1) * (
                self.request_ready_feature(timing_features)
            )
        return augmented, {
            'request_ready_prediction_seconds': prediction_seconds,
            'request_ready_residual_seconds': residual_seconds,
            'request_ready_dag_seconds': dag_seconds.clamp_min(0.0),
            'request_ready_prediction_valid': valid,
            'request_ready_kind_ids': kind_ids,
            'request_ready_quantile_seconds': quantile_seconds,
        }

    def _counterfactual_step_value(
        self,
        critic,
        *,
        query,
        op_nodes,
        site_nodes,
        legal_log_probs,
        chosen_action=None,
        n_sites=None,
    ):
        """Evaluate replayed Q(s,a) or its policy expectation without new tensors.

        The existing role-specific ``StepCritic`` is reused as an action-value
        head by exposing only the selected operation/request (and, for plane
        pairs, the selected site) to its attention masks.  This deliberately
        adds no checkpoint parameters: Stage2 -> Stage3 remains bit-exact.

        During rollout ``chosen_action`` is absent and the returned baseline is
        the normalized top-k policy expectation.  During critic replay the
        actual action is supplied and the returned scalar is Q(s,a), which is
        trained against the exact role-clock Monte-Carlo return.
        """

        batch_size, action_count = legal_log_probs.shape
        if n_sites is not None:
            n_sites = int(n_sites)
            if n_sites < 1 or action_count % n_sites != 0:
                raise RuntimeError(
                    'Plane counterfactual action space is not divisible by '
                    f'n_sites: actions={action_count}, sites={n_sites}.'
                )

        def evaluate(action_ids):
            if action_ids.dim() == 1:
                action_ids = action_ids.unsqueeze(1)
            width = action_ids.shape[1]
            flat_actions = action_ids.reshape(-1)
            repeated_query = query.repeat_interleave(width, dim=0)
            repeated_ops = op_nodes.repeat_interleave(width, dim=0)
            repeated_sites = site_nodes.repeat_interleave(width, dim=0)
            rows = torch.arange(
                flat_actions.numel(), device=flat_actions.device
            )

            if n_sites is None:
                op_ids = flat_actions
                site_pad_mask = None
            else:
                op_ids = torch.div(
                    flat_actions, n_sites, rounding_mode='floor'
                )
                site_ids = torch.remainder(flat_actions, n_sites)
                site_pad_mask = torch.ones(
                    (flat_actions.numel(), site_nodes.shape[1]),
                    dtype=torch.bool,
                    device=flat_actions.device,
                )
                site_pad_mask[rows, site_ids] = False

            op_pad_mask = torch.ones(
                (flat_actions.numel(), op_nodes.shape[1]),
                dtype=torch.bool,
                device=flat_actions.device,
            )
            op_pad_mask[rows, op_ids] = False
            values = critic(
                query=repeated_query.unsqueeze(1),
                op_nodes=repeated_ops,
                site_nodes=repeated_sites,
                op_pad_mask=op_pad_mask,
                site_pad_mask=site_pad_mask,
            ).reshape(batch_size, width)
            return values

        if chosen_action is not None:
            return evaluate(chosen_action.long().reshape(batch_size)).reshape(-1)

        finite = torch.isfinite(legal_log_probs)
        probabilities = torch.where(
            finite, torch.exp(legal_log_probs), torch.zeros_like(legal_log_probs)
        )
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        topk = min(self.counterfactual_q_topk, action_count)
        top_probabilities, top_actions = torch.topk(
            probabilities, k=topk, dim=-1
        )
        represented_mass = top_probabilities.sum(dim=-1, keepdim=True)
        # Keep only tiny detached tensors until the enclosing rollout shard is
        # complete.  This avoids a GPU synchronization on every environment
        # decision while still exposing top-k coverage as an auditable metric.
        if not hasattr(self, '_counterfactual_represented_masses'):
            self._counterfactual_represented_masses = []
        self._counterfactual_represented_masses.append(
            represented_mass.detach().reshape(-1)
        )
        weights = top_probabilities / represented_mass.clamp_min(1e-12)
        q_values = evaluate(top_actions)
        return (
            float(getattr(self, 'counterfactual_baseline_mix', 1.0))
            * (q_values * weights).sum(dim=-1)
        )

    def set_counterfactual_baseline_mix(self, mix):
        """Set rollout control-variate strength without changing Q targets."""
        mix = float(mix)
        if not np.isfinite(mix) or not 0.0 <= mix <= 1.0:
            raise ValueError('counterfactual_baseline_mix must be in [0, 1].')
        self.counterfactual_baseline_mix = mix
        return self.counterfactual_baseline_mix

    def reset_counterfactual_diagnostics(self):
        self._counterfactual_represented_masses = []

    def consume_counterfactual_diagnostics(self):
        """Return and reset rollout top-k probability-mass diagnostics."""
        metrics = {
            'counterfactual_baseline_mix': float(
                self.counterfactual_baseline_mix
            ),
            'counterfactual_topk_mass_count': 0.0,
            'counterfactual_topk_mass_mean': 0.0,
            'counterfactual_topk_mass_min': 0.0,
            'counterfactual_topk_mass_p10': 0.0,
            'counterfactual_topk_mass_target_fraction': 0.0,
        }
        masses = self._counterfactual_represented_masses
        self._counterfactual_represented_masses = []
        if not masses:
            return metrics
        values = torch.cat(masses).float()
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return metrics
        metrics.update({
            'counterfactual_topk_mass_count': float(values.numel()),
            'counterfactual_topk_mass_mean': float(values.mean().item()),
            'counterfactual_topk_mass_min': float(values.min().item()),
            'counterfactual_topk_mass_p10': float(
                torch.quantile(values, 0.10).item()
            ),
            'counterfactual_topk_mass_target_fraction': float(
                (values >= self.counterfactual_q_min_mass).float().mean().item()
            ),
        })
        return metrics

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

    def _resolve_device_type_ids(self, graph, bsz, n_devices, device):
        """Return categorical ordinary-device ids, with -1 for R014/padding."""
        raw = getattr(graph, 'device_type_ids', None)
        fallback = torch.full(
            (bsz, n_devices), -1, dtype=torch.long, device=device
        )
        if raw is None:
            return fallback
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw)
        raw = raw.to(device=device, dtype=torch.long)
        if raw.numel() != bsz * n_devices:
            raise RuntimeError(
                'device_type_ids shape does not match the batched device '
                f'nodes: values={raw.numel()}, expected={bsz*n_devices}.'
            )
        resolved = raw.reshape(bsz, n_devices)
        invalid = (resolved < -1) | (
            resolved >= self.ordinary_device_type_count
        )
        if invalid.any():
            values = torch.unique(resolved[invalid]).detach().cpu().tolist()
            raise RuntimeError(
                'device_type_ids contain values outside the configured '
                f'vocabulary: {values!r}.'
            )
        return resolved

    def _apply_device_global_matching(
        self,
        resource_action_logits,
        request_is_lookahead,
        active_agents,
        op_choice,
        log_prob,
        decision_validated,
    ):
        """Decode deterministic resource actions as one bipartite matching.

        Training still exposes the normalized per-device score distributions,
        while validation resolves request collisions globally instead of by
        device list order.  Each device owns a private no-op column and every
        real request owns one shared column, so the assignment is one-to-one
        without changing the environment's legal-action contract.
        """
        if resource_action_logits is None:
            raise RuntimeError(
                'Global resource matching requires per-device action logits.'
            )
        for batch_idx in range(int(resource_action_logits.shape[0])):
            rows = torch.nonzero(
                active_agents[batch_idx, self.max_plane_agents:].bool(),
                as_tuple=False,
            ).flatten() + self.max_plane_agents
            if rows.numel() == 0:
                continue
            scores = resource_action_logits[
                batch_idx, rows
            ].detach().cpu().double().numpy()
            assignments = solve_resource_matching(
                scores, request_is_lookahead[batch_idx].detach().cpu().numpy()
            )
            for local_row, request_id in enumerate(assignments.tolist()):
                selected_log_prob = resource_action_logits[
                    batch_idx, rows[local_row], request_id
                ]
                if not bool(torch.isfinite(selected_log_prob).item()):
                    raise RuntimeError(
                        'Global resource matching has no feasible complete '
                        f'assignment in graph row {batch_idx}; device_agent='
                        f'{int(rows[local_row])}, request={request_id}.'
                    )
                op_choice[batch_idx, rows[local_row]] = request_id
                log_prob[batch_idx, rows[local_row]] = selected_log_prob
                decision_validated[batch_idx, rows[local_row]] = True

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

    def _encode_graph(self, graph, *, actor_grad):
        """Encode one graph batch with only the gradients the optimizer owns.

        The shared encoder belongs exclusively to the actor optimizer.  A
        critic-only pass previously built encoder activations whenever Stage3
        unfroze the encoder, even though the critic optimizer could never step
        those parameters.  Detaching that path is both mathematically exact
        for the configured optimizer ownership and removes wasted VRAM.

        A graph=1000 trainable actor forward has large TransformerConv edge
        activations.  Non-reentrant activation checkpointing retains the exact
        forward values and gradients while recomputing the encoder during
        backward, keeping four GPU-local trainers viable.  PPO runs the model
        in eval mode, so this recomputation contains no stochastic dropout.
        """

        runtime = getattr(self, 'stage3_execution_cache', None)
        if runtime is not None and runtime.active:
            return runtime.encode(graph, actor_grad=actor_grad)
        encoder_trainable = bool(
            actor_grad
            and torch.is_grad_enabled()
            and any(parameter.requires_grad for parameter in self.encoder.parameters())
        )
        if not encoder_trainable:
            with torch.no_grad():
                return self.encoder(graph)
        if self.shared_encoder_activation_checkpoint:
            return checkpoint(
                self.encoder,
                graph,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return self.encoder(graph)

    def forward(self, data, info, deterministic: bool = False, chosen_op=None,
                chosen_site=None, chosen_order=None,
                actor_grad: bool = True, criticize: bool = True, eval_action: bool = False, 
                dist_only: bool = False, criticize_only: bool = False,
                return_decision_mask: bool = False,
                return_log_prob_components: bool = False,
                encoded_graph=None,
                return_rnn_states: bool = False,
                resource_rl: bool = False,
                return_actor_details: bool = False,
                plane_deterministic=None):
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
        plane_deterministic = (
            deterministic if plane_deterministic is None else bool(plane_deterministic)
        )

        # ========================================================
        # 1. 编码异构图 (Hetero Encoder)
        # ========================================================
        # Stage2 freezes the shared encoder.  DeviceBC first runs the actor to
        # advance recurrent state and then replays the teacher action for its
        # supervised loss on exactly the same observation.  Accepting the
        # first pass' detached encoding avoids doing the frozen GNN work twice
        # without changing any trainable activation or gradient.
        enc_out = (
            self._encode_graph(data['graph'], actor_grad=actor_grad)
            if encoded_graph is None
            else encoded_graph
        )
        role_encodings = enc_out.get('role_encodings')
        if resource_rl and not bool(torch.stack([
            torch.isfinite(value).all() for value in enc_out.values()
            if torch.is_tensor(value)
        ]).all().item()):
            raise FloatingPointError('Non-finite frozen encoding in resource RL.')
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
        device_type_ids = self._resolve_device_type_ids(
            data['graph'], bsz, device_nodes.shape[1], global_emb.device
        )
        n_sites = data['graph'].site_mask_matrix.shape[-1]
        n_requests = data['graph'].request_mask_matrix.shape[-1]
        request_nodes, request_ready_outputs = (
            self._augment_requests_with_ready_prediction(
                data['graph'],
                global_emb=global_emb,
                op_nodes=op_nodes,
                site_nodes=site_nodes,
                request_nodes=request_nodes,
            )
        )
        resource_global_emb = global_emb
        resource_v6 = getattr(self, 'resource_v6', None)
        v6_context = None
        if resource_v6 is not None:
            from onpolicy.algorithms.utils.resource_actor_v6 import resource_context
            v6_context = resource_context(data['graph'])
            if criticize:
                raise ValueError('V6 uses its independent resource critic, not legacy agent values.')
        # V5 opt-in role routing. Legacy models have no resource_encoder and
        # follow the original path bit for bit. Ready/plane always consume the
        # original encoding above; a learned resource view is never injected
        # into either of those protected heads or the frozen B0 reference.
        resource_encoder = getattr(self, 'resource_encoder', None)
        if resource_encoder is not None:
            if self.request_ready_policy_injection != 'none' or self.resource_residual_adapter is not None:
                raise ValueError('Split graph encoding requires unmodified B0 ready/adapter semantics.')
            if torch.is_grad_enabled() and actor_grad:
                if getattr(self, 'resource_encoder_activation_checkpoint', False):
                    resource_view = checkpoint(resource_encoder, data['graph'],
                        use_reentrant=False, preserve_rng_state=True)
                else:
                    resource_view = resource_encoder(data['graph'])
            else:
                with torch.no_grad():
                    resource_view = resource_encoder(data['graph'])
            if not bool(torch.stack([torch.isfinite(v).all()
                    for v in resource_view.values() if torch.is_tensor(v)]).all()):
                raise FloatingPointError('Non-finite split resource encoding.')
            resource_global_emb = resource_view['global_emb']
            request_nodes = resource_view['request_nodes']
            device_nodes = resource_view['device_nodes']
        if self.resource_residual_adapter is not None:
            resource_global_emb = self.resource_residual_adapter(global_emb)
            request_nodes = self.resource_residual_adapter(request_nodes)
            device_nodes = self.resource_residual_adapter(device_nodes)
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
        if self.actor_requires_pair_features:
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
                deterministic=(plane_deterministic or criticize_only),
                chosen_ranks=replay_ranks,
                tau=self.tau,
            )
            decision_order[:, :plane_count] = order_by_rank
            plane_order_rank[:, :plane_count] = rank_by_plane
            plane_order_log_prob[:, :plane_count] = order_log_prob
            plane_order_entropy[:, :plane_count] = order_entropy
            plane_order_trainable[:, :plane_count] = order_trainable
        
        # ``action`` is also the recurrent selection history consumed by the
        # next global event.  An inactive agent did not make a new decision,
        # so returning the historical selection is essential: resetting it
        # to -1 at every unrelated event makes a previously active plane look
        # cold-started when it next becomes ready.  Besides corrupting PPO
        # replay, that used to make Stage2's frozen-plane rollout diverge from
        # the IGA label replay, whose history comes directly from the
        # environment.  Active rows below are still overwritten by the newly
        # sampled/replayed action; only inactive rows carry their history.
        op_choice = last_op_indices.to(
            device=global_emb.device, dtype=torch.long
        ).view(bsz, M).clone()
        site_choice = last_site_indices.to(
            device=global_emb.device, dtype=torch.long
        ).view(bsz, M).clone()
        log_prob = torch.zeros((bsz, M), device=global_emb.device)
        pair_log_prob = torch.zeros_like(log_prob)
        pair_decision_trainable = torch.zeros(
            (bsz, M), dtype=torch.bool, device=global_emb.device
        )
        value_mask = torch.zeros((bsz, M), dtype=torch.bool, device=global_emb.device)
        decision_validated = torch.zeros((bsz, M), dtype=torch.bool, device=global_emb.device)
        decision_trainable = torch.zeros((bsz, M), dtype=torch.bool, device=global_emb.device)
        value = torch.zeros((bsz, M), device=global_emb.device)
        use_device_global_matching = bool(
            self.device_global_matching
            and not resource_rl
            and deterministic
            and chosen_op is None
            and not criticize_only
        )
        # During supervised replay, expose every device's logits against the
        # same pre-claim request pool.  Otherwise a teacher assignment made by
        # an earlier equivalent device erases the candidate before a
        # permutation-invariant team loss can see it.  Stochastic PPO rollout
        # deliberately retains the historical autoregressive decoder.
        use_device_global_scoring = bool(
            self.device_global_matching
            and not resource_rl
            and (return_log_prob_components or use_device_global_matching)
            and not criticize_only
        )
        resource_action_logits = (
            torch.full(
                (bsz, M, n_requests),
                float('-inf'),
                dtype=global_emb.dtype,
                device=global_emb.device,
            )
            if return_log_prob_components or use_device_global_matching or return_actor_details
            else None
        )
        
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

                if resource_v6 is not None:
                    if resource_v6.arm == 'S1':
                        last_req_emb = resource_v6.history_embedding(
                            v6_context, active_batch_indices, device_local_idx)
                    else:
                        last_req_emb = torch.zeros_like(device_nodes[active_mask_i, device_local_idx])
                else:
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
                if use_device_global_scoring:
                    # A private no-op is legal for every row because the team
                    # Hungarian decoder, not device-list order, now enforces
                    # blocking request coverage and uniqueness.
                    cur_req_mask[:, 0] = True
                    cur_req_mask[:, 1:] = current_real_mask
                else:
                    blocking_real_mask = (
                        current_real_mask & ~request_is_lookahead[:, 1:]
                    )
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
                active_device_type_ids = device_type_ids[
                    active_mask_i, device_local_idx
                ]
                if fixed_role == self.AGENT_TYPE_DEVICE:
                    ordinary_mask = torch.ones(
                        active_count, dtype=torch.bool,
                        device=global_emb.device,
                    )
                    transporter_mask = torch.zeros_like(ordinary_mask)
                elif fixed_role == self.AGENT_TYPE_TRANSPORTER:
                    transporter_mask = torch.ones(
                        active_count, dtype=torch.bool,
                        device=global_emb.device,
                    )
                    ordinary_mask = torch.zeros_like(transporter_mask)
                else:
                    transporter_mask = (
                        agent_types[:, agent_idx].view(-1)[active_mask_i]
                        == self.AGENT_TYPE_TRANSPORTER
                    ).view(-1).bool()
                    ordinary_mask = ~transporter_mask

                role_routes = []
                if ordinary_mask.any():
                    ordinary_ids = active_device_type_ids[ordinary_mask]
                    if (
                        self.device_policy_head_mode != 'shared'
                        and (ordinary_ids < 0).any()
                    ):
                        bad_rows = active_batch_indices[
                            ordinary_mask
                        ][ordinary_ids < 0].detach().cpu().tolist()
                        raise RuntimeError(
                            'Ordinary device agents are missing categorical '
                            f'device_type_ids for graph rows {bad_rows}.'
                        )
                    if self.device_policy_head_mode == 'per_type':
                        for type_id in torch.unique(
                            ordinary_ids, sorted=True
                        ).detach().cpu().tolist():
                            type_mask = ordinary_mask & (
                                active_device_type_ids == int(type_id)
                            )
                            role_routes.append((
                                type_mask,
                                self.device_type_sel_encs[int(type_id)],
                                self.device_type_actors[int(type_id)],
                                self.device_critic,
                                f'ordinary_type_{int(type_id)}',
                                int(type_id),
                            ))
                    else:
                        role_routes.append((
                            ordinary_mask,
                            self.device_sel_enc,
                            self.device_actor,
                            self.device_critic,
                            'ordinary',
                            None,
                        ))
                if transporter_mask.any():
                    role_routes.append((
                        transporter_mask,
                        self.transporter_sel_enc,
                        self.transporter_actor,
                        self.transporter_critic,
                        'transporter',
                        None,
                    ))
                for (
                    role_mask, sel_enc, actor_head, critic_head, role_name,
                    routed_type_id,
                ) in role_routes:
                    if role_mask is None:
                        role_index = slice(None)
                    else:
                        if not role_mask.any():
                            continue
                        role_index = role_mask

                    role_batch_indices = active_batch_indices[role_index]
                    # Optional Stage3 private graph tails. Keep the same joint
                    # masks, request claims and autoregressive decision order.
                    role_features = (role_encodings['2' if role_name == 'transporter' else '1']
                                     if role_encodings is not None else None)
                    role_requests = request_nodes if role_features is None else role_features['request_nodes']
                    role_sites = site_nodes if role_features is None else role_features['site_nodes']
                    role_global = resource_global_emb if role_features is None else role_features['global_emb']
                    role_selection = last_selection_emb[role_index]
                    role_vehicle = dev_emb[role_index]
                    if role_features is not None:
                        role_selection = (role_requests[role_batch_indices, safe_last_req_idx[role_index], :]
                                          * valid_last_req[role_index].unsqueeze(-1).float()).unsqueeze(1)
                        role_vehicle = role_features['device_nodes'][role_batch_indices, device_local_idx, :].unsqueeze(1)
                    hidden_state_i = (
                        data['hidden_states'][role_batch_indices, agent_idx:agent_idx+1, :]
                        .squeeze(1)
                        .transpose(0, 1)
                    )
                    if resource_v6 is not None and resource_v6.arm == 'S2':
                        role_query = None  # Opt-in stateless Stage2 resource head.
                    else:
                        seq_embed, slice_embed, new_hidden_state_i = sel_enc(
                            selection=role_selection,
                            veh=role_vehicle,
                            hidden_state=hidden_state_i,
                        )
                        new_hidden_state[role_batch_indices, agent_idx:agent_idx+1, :] = (
                            new_hidden_state_i.transpose(0, 1).unsqueeze(1)
                        )
                        role_query = torch.cat([
                            role_global[role_batch_indices],
                            seq_embed.squeeze(1),
                            slice_embed.squeeze(1),
                        ], dim=-1)
                    if (
                        self.device_policy_head_mode == 'type_adapter'
                        and role_name == 'ordinary'
                    ):
                        role_type_ids = active_device_type_ids[role_index]
                        role_query = role_query + self.device_type_adapter(
                            role_type_ids
                        )
                    role_req_mask = cur_req_mask[role_batch_indices]
                    role_chosen = (
                        cur_chosen_req[role_index]
                        if cur_chosen_req is not None else None
                    )
                    role_decision_trainable = role_req_mask.sum(dim=-1) > 1
                    decision_trainable[role_batch_indices, agent_idx] = role_decision_trainable

                    try:
                        if resource_v6 is not None and resource_v6.arm == 'S2':
                            with torch.set_grad_enabled(torch.is_grad_enabled() and actor_grad):
                                role_req, role_log_prob, role_dist = resource_v6.act(
                                    v6_context, role_batch_indices, device_local_idx,
                                    role_name, role_req_mask, deterministic, self.tau, role_chosen)
                        elif actor_grad:
                            role_req, role_log_prob, role_dist = actor_head(
                                query=role_query.unsqueeze(1),
                                request_nodes=role_requests[role_batch_indices],
                                request_valid_mask=role_req_mask,
                                deterministic=deterministic,
                                tau=self.tau,
                                chosen_request=role_chosen,
                            )
                        else:
                            with torch.no_grad():
                                role_req, role_log_prob, role_dist = actor_head(
                                    query=role_query.unsqueeze(1),
                                    request_nodes=role_requests[role_batch_indices],
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
                        critic_query = role_query.detach() if criticize_only else role_query
                        critic_requests = (
                            role_requests[role_batch_indices].detach()
                            if criticize_only else role_requests[role_batch_indices]
                        )
                        critic_sites = (
                            role_sites[role_batch_indices].detach()
                            if criticize_only else role_sites[role_batch_indices]
                        )
                        if self.counterfactual_q_baseline:
                            cur_value[role_index] = (
                                self._counterfactual_step_value(
                                    critic_head,
                                    query=critic_query,
                                    op_nodes=critic_requests,
                                    site_nodes=critic_sites,
                                    legal_log_probs=role_dist,
                                    chosen_action=role_chosen,
                                )
                            )
                        else:
                            cur_req_pad_mask = ~role_req_mask
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
                if resource_action_logits is not None:
                    resource_action_logits[
                        active_batch_indices, agent_idx, :
                    ] = cur_dist

                if criticize:
                    value[active_mask_i, agent_idx] = cur_value

                batch_indices = active_batch_indices
                real_req_mask = cur_req > 0
                affected_batches = batch_indices[real_req_mask]
                if not use_device_global_scoring:
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
            if self.stage1_baseline != 'proposed':
                # The published baselines are Markov graph policies and do
                # not inherit the proposed decoder's recurrent selection
                # memory.  Keep the common critic input width while leaving
                # its two recurrent slots exactly zero.
                baseline_global = global_emb[active_mask_i]
                query = torch.cat((
                    baseline_global,
                    torch.zeros_like(baseline_global),
                    torch.zeros_like(baseline_global),
                ), dim=-1)
            elif new_hidden_state is not None:
                hidden_state_i = data['hidden_states'][active_mask_i, agent_idx:agent_idx+1, :].squeeze(1).transpose(0, 1)
                
                # Resource BPTT stores all roles in one hidden-state tensor.
                # Even a frozen plane slice can consequently carry a grad_fn;
                # cut that graph explicitly instead of asking eval-mode cuDNN
                # plane GRUs to backpropagate through an irrelevant zero path.
                with torch.set_grad_enabled(torch.is_grad_enabled() and not resource_rl):
                    seq_embed, slice_embed, new_hidden_state_i = self.plane_sel_enc(
                        selection=last_selection_emb,
                        veh=veh_emb,
                        hidden_state=hidden_state_i.detach() if resource_rl else hidden_state_i,
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
            if self.actor_requires_pair_features:
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
            if actor_grad and not resource_rl:
                cur_op, cur_site, cur_log_prob, cur_dist = self.actor(
                    query=query.unsqueeze(1), 
                    op_nodes=cur_op_nodes,
                    site_nodes=site_nodes[active_mask_i], 
                    op_valid_mask=cur_agent_mask[active_mask_i], 
                    site_valid_mask=cur_site_mask[active_mask_i], 
                    deterministic=plane_deterministic,
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
                        deterministic=plane_deterministic,
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
                critic_query = query.detach() if criticize_only else query
                critic_ops = cur_op_nodes.detach() if criticize_only else cur_op_nodes
                critic_sites = (
                    site_nodes[active_mask_i].detach()
                    if criticize_only else site_nodes[active_mask_i]
                )
                if self.counterfactual_q_baseline:
                    replay_pair = None
                    if cur_chosen_op is not None:
                        replay_pair = cur_chosen_op * n_sites + cur_chosen_site
                    value[active_mask_i, agent_idx] = (
                        self._counterfactual_step_value(
                            self.plane_critic,
                            query=critic_query,
                            op_nodes=critic_ops,
                            site_nodes=critic_sites,
                            legal_log_probs=cur_dist,
                            chosen_action=replay_pair,
                            n_sites=n_sites,
                        )
                    )
                else:
                    cur_op_pad_mask = ~cur_agent_mask[active_mask_i]
                    # Legacy V(s) sees all globally valid site embeddings.
                    value[active_mask_i, agent_idx] = self.plane_critic(
                        query=critic_query.unsqueeze(1),
                        op_nodes=critic_ops,
                        site_nodes=critic_sites,
                        op_pad_mask=cur_op_pad_mask,
                        site_pad_mask=None,
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

        if use_device_global_matching:
            self._apply_device_global_matching(
                resource_action_logits,
                request_is_lookahead,
                active_agents,
                op_choice,
                log_prob,
                decision_validated,
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
        if resource_rl and not bool(torch.isfinite(log_prob).all().item()):
            raise FloatingPointError('Non-finite action log probability in resource RL.')
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

        if return_actor_details:
            return {
                'actions': action,
                'log_probs': log_prob,
                'decision_mask': decision_trainable,
                'resource_logits': resource_action_logits,
                'rnn_states': new_hidden_state,
                'global_emb': global_emb,
            }
        
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
                    'resource_action_logits': resource_action_logits,
                    **request_ready_outputs,
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
