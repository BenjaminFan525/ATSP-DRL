import numpy as np
import math
import copy
import itertools
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss, expand_slice
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from torch_geometric.loader.dataloader import Batch
from collections import defaultdict

def _finite_item(value, fallback=0.0):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.float().mean().item()
    if not np.isfinite(value):
        return fallback
    return float(value)

def _sanitize_tensor(value, fallback=0.0):
    return torch.nan_to_num(value, nan=fallback, posinf=fallback, neginf=fallback)

def _weighted_quantile(values, weights, quantile):
    """Return a finite weighted quantile over positive-weight PPO decisions."""
    values = values.detach().reshape(-1)
    weights = weights.detach().reshape(-1)
    valid = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return torch.zeros((), dtype=values.dtype, device=values.device)
    values = values[valid]
    weights = weights[valid]
    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = torch.cumsum(weights, dim=0)
    threshold = float(quantile) * cumulative[-1]
    index = torch.searchsorted(cumulative, threshold, right=False)
    index = torch.clamp(index, max=values.numel() - 1)
    return values[index]

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
        self.grad_accumulation_steps = max(1, int(args.grad_accumulation_steps))
        actor_accumulation = int(
            getattr(args, 'actor_grad_accumulation_steps', 0)
        )
        self.actor_grad_accumulation_steps = (
            self.grad_accumulation_steps
            if actor_accumulation <= 0
            else actor_accumulation
        )
        self.grad_accumulation_target_graphs = max(
            0, int(getattr(args, 'grad_accumulation_target_graphs', 0))
        )
        self.actor_grad_accumulation_target_graphs = max(
            0,
            int(getattr(
                args, 'actor_grad_accumulation_target_graphs', 0
            )),
        )
        self.target_kl = max(0.0, float(getattr(args, 'target_kl', 0.0)))
        self.actor_kl_backtrack = bool(
            getattr(args, 'actor_kl_backtrack', False)
        )
        backtrack_raw = str(getattr(
            args, 'actor_kl_backtrack_scales', '0.5,0.25,0.125'
        ))
        self.actor_kl_backtrack_scales = tuple(
            float(item.strip())
            for item in backtrack_raw.split(',') if item.strip()
        )
        if self.actor_kl_backtrack and not self.actor_kl_backtrack_scales:
            raise ValueError('--actor_kl_backtrack requires retry scales.')
        if any(
            not np.isfinite(value) or not 0.0 < value < 1.0
            for value in self.actor_kl_backtrack_scales
        ) or any(
            later >= earlier
            for earlier, later in zip(
                self.actor_kl_backtrack_scales,
                self.actor_kl_backtrack_scales[1:],
            )
        ):
            raise ValueError(
                '--actor_kl_backtrack_scales must be strictly decreasing '
                'finite values in (0, 1).'
            )
        self.bc_reference_kl_coef = max(
            0.0, float(getattr(args, 'bc_reference_kl_coef', 0.0))
        )
        self.bc_reference_target_kl = max(
            0.0, float(getattr(args, 'bc_reference_target_kl', 0.0))
        )
        self.bc_reference_hard_gate = bool(
            getattr(args, 'bc_reference_hard_gate', False)
        )
        self.adaptive_bc_reference_kl = bool(
            getattr(args, 'adaptive_bc_reference_kl', False)
        )
        self.adaptive_bc_reference_target_kl = float(getattr(
            args, 'adaptive_bc_reference_target_kl', 0.03
        ))
        self.adaptive_bc_reference_coef_min = float(getattr(
            args, 'adaptive_bc_reference_coef_min', 0.02
        ))
        self.adaptive_bc_reference_coef_max = float(getattr(
            args, 'adaptive_bc_reference_coef_max', 1.0
        ))
        self.adaptive_bc_reference_coef_up = float(getattr(
            args, 'adaptive_bc_reference_coef_up', 1.5
        ))
        self.adaptive_bc_reference_coef_down = float(getattr(
            args, 'adaptive_bc_reference_coef_down', 0.8
        ))
        if self.adaptive_bc_reference_kl:
            if self.bc_reference_hard_gate:
                raise ValueError(
                    'Adaptive BC-reference KL is a soft controller and is '
                    'incompatible with --bc_reference_hard_gate.'
                )
            if not (
                self.adaptive_bc_reference_target_kl > 0.0
                and 0.0 <= self.adaptive_bc_reference_coef_min
                <= self.adaptive_bc_reference_coef_max
                and self.adaptive_bc_reference_coef_up > 1.0
                and 0.0 < self.adaptive_bc_reference_coef_down < 1.0
            ):
                raise ValueError('Adaptive BC-reference settings are invalid.')
            self.bc_reference_kl_coef = float(np.clip(
                self.bc_reference_kl_coef,
                self.adaptive_bc_reference_coef_min,
                self.adaptive_bc_reference_coef_max,
            ))
        self.tail_policy_start_fraction = float(
            getattr(args, 'tail_policy_start_fraction', 1.0)
        )
        self.tail_policy_weight = float(
            getattr(args, 'tail_policy_weight', 1.0)
        )
        self.cvar_policy_fraction = float(
            getattr(args, 'cvar_policy_fraction', 1.0)
        )
        self.cvar_policy_weight = float(
            getattr(args, 'cvar_policy_weight', 1.0)
        )
        self.cvar_case_metric = str(
            getattr(args, 'cvar_case_metric', 'cmax')
        )
        if not 0.0 <= self.tail_policy_start_fraction <= 1.0:
            raise ValueError('--tail_policy_start_fraction must be in [0, 1].')
        if self.tail_policy_weight < 1.0:
            raise ValueError('--tail_policy_weight must be at least 1.')
        if not 0.0 < self.cvar_policy_fraction <= 1.0:
            raise ValueError('--cvar_policy_fraction must be in (0, 1].')
        if self.cvar_policy_weight < 1.0:
            raise ValueError('--cvar_policy_weight must be at least 1.')
        if self.cvar_case_metric not in {'cmax', 'paired_delta'}:
            raise ValueError('--cvar_case_metric must be cmax or paired_delta.')
        self.normalize_advantages = bool(getattr(args, 'normalize_advantages', True))
        self.role_balanced_loss = bool(getattr(args, 'role_balanced_loss', True))
        self.case_balanced_loss = bool(getattr(args, 'case_balanced_loss', True))
        self.role_loss_coef = {
            0: float(getattr(args, 'plane_loss_coef', 1.0)),
            1: float(getattr(args, 'device_loss_coef', 0.5)),
            2: float(getattr(args, 'transporter_loss_coef', 0.5)),
        }
        self.joint_team_ppo = bool(
            getattr(args, 'joint_team_ppo', False)
        )
        self.joint_team_ppo_scope = str(getattr(
            args, 'joint_team_ppo_scope', 'plane'
        ))
        if self.joint_team_ppo_scope not in {'plane', 'all'}:
            raise ValueError(
                '--joint_team_ppo_scope must be plane or all.'
            )
        self.role_atomic_ppo = bool(getattr(args, 'role_atomic_ppo', False))
        self.role_event_returns = bool(getattr(args, 'role_event_returns', False))
        self.role_event_gae_lambda = float(getattr(
            args, 'role_event_gae_lambda', 1.0
        ))
        self.role_event_gae_lambdas = {
            role: (
                self.role_event_gae_lambda
                if float(getattr(
                    args, f'{role_name}_role_gae_lambda', -1.0
                )) < 0.0
                else float(getattr(
                    args, f'{role_name}_role_gae_lambda', -1.0
                ))
            )
            for role, role_name in {
                0: 'plane', 1: 'device', 2: 'transporter'
            }.items()
        }
        self.role_loss_weighting = str(getattr(
            args, 'role_loss_weighting', 'fixed'
        ))
        self.role_loss_min_share = float(getattr(
            args, 'role_loss_min_share', 0.15
        ))
        self.role_loss_max_share = float(getattr(
            args, 'role_loss_max_share', 0.60
        ))
        self.role_valuenorm = bool(getattr(args, 'role_valuenorm', False))
        self.shared_gradient_diagnostics = bool(getattr(
            args, 'shared_gradient_diagnostics', False
        ))
        self.shared_encoder_pcgrad = bool(getattr(
            args, 'shared_encoder_pcgrad', False
        ))
        self.shared_gradient_method = str(getattr(
            args, 'shared_gradient_method', 'sum'
        ))
        self.shared_grad_ema_beta = float(getattr(
            args, 'shared_grad_ema_beta', 0.97
        ))
        self.shared_grad_norm_power = float(getattr(
            args, 'shared_grad_norm_power', 0.5
        ))
        self.shared_grad_min_scale = float(getattr(
            args, 'shared_grad_min_scale', 0.5
        ))
        self.shared_grad_max_scale = float(getattr(
            args, 'shared_grad_max_scale', 2.0
        ))
        self.shared_grad_conflict_threshold = float(getattr(
            args, 'shared_grad_conflict_threshold', -0.05
        ))
        self.shared_cagrad_c = float(getattr(
            args, 'shared_cagrad_c', 0.2
        ))
        self.shared_grad_norm_ema = {}
        self.shared_grad_norm_ema_updates = 0
        self.role_sequential_ppo = bool(getattr(
            args, 'role_sequential_ppo', False
        ))
        self.role_sequential_factor_clip = float(getattr(
            args, 'role_sequential_factor_clip', 2.0
        ))
        self.role_sequential_min_ess = float(getattr(
            args, 'role_sequential_min_ess', 0.50
        ))
        if self.actor_kl_backtrack and self.role_sequential_ppo:
            raise ValueError(
                '--actor_kl_backtrack currently requires atomic joint Actor '
                'updates, not --role_sequential_ppo.'
            )
        self.role_names = {0: 'plane', 1: 'device', 2: 'transporter'}
        self.role_target_kl = {
            0: max(0.0, float(getattr(args, 'plane_target_kl', 0.0025))),
            1: max(0.0, float(getattr(args, 'device_target_kl', 0.005))),
            2: max(0.0, float(getattr(args, 'transporter_target_kl', 0.005))),
        }
        if self.role_atomic_ppo and (
            not self.joint_team_ppo or self.joint_team_ppo_scope != 'all'
        ):
            raise ValueError(
                '--role_atomic_ppo requires --joint_team_ppo '
                '--joint_team_ppo_scope all.'
            )
        if self.role_event_returns and not self.role_atomic_ppo:
            raise ValueError('--role_event_returns requires --role_atomic_ppo.')
        if not 0.0 <= self.role_event_gae_lambda <= 1.0:
            raise ValueError('--role_event_gae_lambda must be in [0, 1].')
        for role_name, value in zip(
            ('plane', 'device', 'transporter'),
            self.role_event_gae_lambdas.values(),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f'--{role_name}_role_gae_lambda must inherit or resolve '
                    'to a value in [0, 1].'
                )
        if (
            not self.role_event_returns
            and (
                not np.isclose(self.role_event_gae_lambda, 1.0)
                or any(
                    not np.isclose(value, 1.0)
                    for value in self.role_event_gae_lambdas.values()
                )
            )
        ):
            raise ValueError(
                'non-Monte-Carlo role GAE lambda requires '
                '--role_event_returns.'
            )
        if self.role_loss_weighting not in {'fixed', 'sqrt_event'}:
            raise ValueError(
                '--role_loss_weighting must be fixed or sqrt_event.'
            )
        if self.role_loss_weighting == 'sqrt_event' and not self.role_atomic_ppo:
            raise ValueError(
                '--role_loss_weighting sqrt_event requires --role_atomic_ppo.'
            )
        if (
            not np.isfinite(self.role_loss_min_share)
            or not np.isfinite(self.role_loss_max_share)
            or self.role_loss_min_share < 0.0
            or self.role_loss_max_share <= 0.0
            or self.role_loss_min_share > self.role_loss_max_share
        ):
            raise ValueError('Invalid role-loss share bounds.')
        if self.role_valuenorm and not self.role_event_returns:
            raise ValueError('--role_valuenorm requires --role_event_returns.')
        if self.shared_encoder_pcgrad and not self.role_atomic_ppo:
            raise ValueError(
                '--shared_encoder_pcgrad requires --role_atomic_ppo.'
            )
        if self.shared_gradient_method not in {
            'sum', 'norm_balance', 'norm_pcgrad', 'cagrad'
        }:
            raise ValueError(
                '--shared_gradient_method must be sum, norm_balance, '
                'norm_pcgrad, or cagrad.'
            )
        if (
            self.shared_gradient_method != 'sum'
            and not self.role_atomic_ppo
        ):
            raise ValueError(
                'non-sum --shared_gradient_method requires '
                '--role_atomic_ppo.'
            )
        if self.shared_encoder_pcgrad and self.shared_gradient_method != 'sum':
            raise ValueError(
                '--shared_encoder_pcgrad cannot be combined with the new '
                '--shared_gradient_method modes.'
            )
        if not 0.0 <= self.shared_grad_ema_beta < 1.0:
            raise ValueError('--shared_grad_ema_beta must be in [0, 1).')
        if not 0.0 <= self.shared_grad_norm_power <= 1.0:
            raise ValueError('--shared_grad_norm_power must be in [0, 1].')
        if (
            self.shared_grad_min_scale <= 0.0
            or self.shared_grad_max_scale < self.shared_grad_min_scale
        ):
            raise ValueError('Invalid shared role-gradient scale bounds.')
        if not -1.0 <= self.shared_grad_conflict_threshold <= 0.0:
            raise ValueError(
                '--shared_grad_conflict_threshold must be in [-1, 0].'
            )
        if not 0.0 <= self.shared_cagrad_c < 1.0:
            raise ValueError('--shared_cagrad_c must be in [0, 1).')
        if self.role_sequential_ppo and not self.role_atomic_ppo:
            raise ValueError(
                '--role_sequential_ppo requires --role_atomic_ppo.'
            )
        if self.role_sequential_factor_clip < 1.0:
            raise ValueError(
                '--role_sequential_factor_clip must be at least 1.'
            )
        if not 0.0 < self.role_sequential_min_ess <= 1.0:
            raise ValueError(
                '--role_sequential_min_ess must be in (0, 1].'
            )
        self.safe_graph_batch_pipeline = bool(
            getattr(args, 'safe_graph_batch_pipeline', False)
        )
        
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(input_shape=1, device=self.device)
        else:
            self.value_normalizer = None
        if getattr(self, 'role_valuenorm', False):
            if self._use_popart or not self._use_valuenorm:
                raise ValueError(
                    '--role_valuenorm requires --use_valuenorm and is '
                    'incompatible with PopArt.'
                )
            self.role_value_normalizers = {
                role: ValueNorm(input_shape=1, device=self.device)
                for role in self.role_names
            }
        else:
            self.role_value_normalizers = {}
        self._last_actor_step_state = None

    @staticmethod
    def _prepare_graph_sample(sample):
        """Build the deterministic CPU PyG batch once for one PPO sample."""
        graph_batch = sample[0]
        if isinstance(graph_batch, Batch):
            return sample
        if isinstance(graph_batch, np.ndarray):
            graph_batch = Batch.from_data_list(graph_batch.tolist())
        elif isinstance(graph_batch, list):
            graph_batch = Batch.from_data_list(graph_batch)
        else:
            graph_batch = Batch.from_data_list([graph_batch])
        return (graph_batch, *sample[1:])

    def _iter_prepared_graph_samples(self, samples, executor):
        """Yield samples in the original order while preparing the next one."""
        if executor is None or not samples:
            yield from samples
            return
        future = executor.submit(self._prepare_graph_sample, samples[0])
        for index in range(len(samples)):
            prepared = future.result()
            future = (
                executor.submit(
                    self._prepare_graph_sample, samples[index + 1]
                )
                if index + 1 < len(samples) else None
            )
            yield prepared

    def _role_sample_weights(self, masks, agent_types, base_weights=None):
        masks = masks.float()
        agent_types = agent_types.long()
        if base_weights is not None:
            weights = masks * base_weights.float()
            if weights.sum() > 0:
                return weights
        weights = torch.zeros_like(masks)
        for role, coef in self.role_loss_coef.items():
            role_mask = masks * (agent_types.unsqueeze(-1) == role).float()
            if role_mask.sum() <= 0 or coef <= 0:
                continue
            if self.role_balanced_loss:
                weights += role_mask * (coef / role_mask.sum().clamp_min(1.0))
            else:
                weights += role_mask * coef
        if weights.sum() <= 0:
            return masks
        return weights

    def _trainable_actor_roles(self):
        """Return agent roles connected to at least one trainable actor group.

        A trainable shared encoder makes every role differentiable. Otherwise
        each backend only makes its own role differentiable. This distinction
        matters in Stage2, where plane/shared parameters are frozen and some
        temporal PPO chunks contain only plane decisions.
        """

        roles = set()
        group_roles = {
            'shared_encoder': {0, 1, 2},
            'plane_actor': {0},
            'device_actor': {1},
            'transporter_actor': {2},
        }
        for group in self.policy.actor_optimizer.param_groups:
            if not any(
                parameter.requires_grad for parameter in group['params']
            ):
                continue
            roles.update(group_roles.get(str(group.get('name', '')), {0, 1, 2}))
        if self.joint_team_ppo:
            roles.intersection_update(
                {0} if self.joint_team_ppo_scope == 'plane' else {0, 1, 2}
            )
        return frozenset(roles)

    def _normalize_rollout_advantages(self, advantages, buffer, rollout_steps):
        if not self.normalize_advantages:
            return advantages
        masks = buffer.policy_masks[:rollout_steps, ..., 0] > 0.0
        if not masks.any():
            masks = buffer.active_masks[:rollout_steps, ..., 0] > 0.0
        agent_types = buffer.agent_types[:rollout_steps]
        normalized = advantages.copy()
        values = normalized[..., 0]
        balance_weights = buffer.policy_sample_weights[:rollout_steps, ..., 0]
        for role in self.role_loss_coef:
            role_mask = masks & (agent_types == role)
            if not role_mask.any():
                continue
            role_values = values[role_mask]
            if self.case_balanced_loss:
                role_weights = balance_weights[role_mask].astype(np.float64)
                weight_sum = float(role_weights.sum())
            else:
                role_weights = None
                weight_sum = 0.0
            if role_weights is not None and weight_sum > 0.0:
                mean = float(np.sum(role_values * role_weights) / weight_sum)
                variance = float(
                    np.sum(((role_values - mean) ** 2) * role_weights) / weight_sum
                )
                std = math.sqrt(max(variance, 0.0))
            else:
                mean = float(role_values.mean())
                std = float(role_values.std())
            values[role_mask] = (role_values - mean) / max(std, 1e-5)
        normalized[..., 0] = values
        return normalized

    def load_value_normalizer_state(self, shared_state, role_states=None):
        """Restore ValueNorm without weakening the Stage2 hand-off contract.

        Stage2 checkpoints contain one shared normalizer.  The first A3 run
        clones that exact state into every role; subsequent checkpoints can
        restore the three independently evolved states.
        """

        if self.value_normalizer is None or shared_state is None:
            return
        if len(shared_state) > 0:
            self.value_normalizer.load_state_dict(shared_state)
        if not self.role_valuenorm:
            return
        role_states = role_states if isinstance(role_states, dict) else {}
        for role, normalizer in self.role_value_normalizers.items():
            state = role_states.get(str(role), role_states.get(role))
            normalizer.load_state_dict(state or shared_state)

    def role_value_normalizer_state_dict(self):
        if not self.role_valuenorm:
            return None
        return {
            str(role): normalizer.state_dict()
            for role, normalizer in self.role_value_normalizers.items()
        }

    def denormalize_value_predictions(self, value_preds, agent_types=None):
        """Express stored critic predictions on the raw objective scale."""

        value_preds = np.nan_to_num(
            np.asarray(value_preds), nan=0.0, posinf=1e4, neginf=-1e4
        )
        if getattr(self, 'role_valuenorm', False):
            if agent_types is None:
                raise ValueError('Role ValueNorm requires rollout agent types.')
            agent_types = np.asarray(agent_types)
            value_baseline = np.zeros_like(value_preds, dtype=np.float32)
            for role, normalizer in self.role_value_normalizers.items():
                role_mask = agent_types == role
                if not role_mask.any():
                    continue
                role_values = value_preds[..., 0][role_mask].reshape(-1, 1)
                value_baseline[..., 0][role_mask] = normalizer.denormalize(
                    role_values
                ).reshape(-1)
        elif self._use_popart or self._use_valuenorm:
            value_baseline = self.value_normalizer.denormalize(value_preds)
        else:
            value_baseline = value_preds
        value_baseline = np.nan_to_num(
            value_baseline, nan=0.0, posinf=1e4, neginf=-1e4
        )
        return value_baseline

    def _compute_rollout_advantages(
        self,
        returns,
        value_preds,
        agent_types=None,
    ):
        """Subtract a critic baseline expressed on the same scale as returns."""
        returns = np.nan_to_num(
            np.asarray(returns), nan=0.0, posinf=1e4, neginf=-1e4
        )
        value_baseline = self.denormalize_value_predictions(
            value_preds, agent_types=agent_types
        )
        return np.nan_to_num(returns - value_baseline, nan=0.0, posinf=1e4, neginf=-1e4), value_baseline

    def cal_value_loss(
        self,
        values,
        value_preds_batch,
        return_batch,
        active_masks_batch,
        sample_weights=None,
        value_normalizer=None,
    ):
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
            return_batch = _sanitize_tensor(return_batch)
            values = _sanitize_tensor(values)
            # The normalizer is updated once per rollout in ``train``
            # with case-balanced weights. Updating it once per RNN chunk would
            # reintroduce trajectory-length bias.
            # 将真实回报归一化，使其与 Critic 网络的输出尺度对齐
            normalizer = value_normalizer or self.value_normalizer
            if normalizer is None:
                raise RuntimeError('Value normalization is enabled without a normalizer.')
            norm_return_batch = normalizer.normalize(return_batch)
            error_original = norm_return_batch - values
        else:
            return_batch = _sanitize_tensor(return_batch)
            values = _sanitize_tensor(values)
            error_original = return_batch - values

        # 2. 计算纯粹的均方误差 (MSE Loss)
        value_loss = mse_loss(error_original)

        # 3. 应用有效动作掩膜 (Active Masks)
        if sample_weights is not None:
            value_loss = (value_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        elif self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            value_loss = value_loss.mean()

        return _sanitize_tensor(value_loss)

    @staticmethod
    def _sample_mass(sample, mask_index):
        mask = np.asarray(sample[mask_index], dtype=np.float64)
        return max(float(np.maximum(mask, 0.0).sum()), 0.0)

    @staticmethod
    def _role_filtered_sample_mass(
        sample,
        mask_index,
        eligible_roles,
        agent_type_index=12,
    ):
        """Return sample mass belonging to differentiable actor roles only."""

        eligible_roles = tuple(sorted(int(role) for role in eligible_roles))
        if not eligible_roles:
            return 0.0
        mask = np.asarray(sample[mask_index], dtype=np.float64)
        agent_types = np.asarray(sample[agent_type_index], dtype=np.int64)
        if mask.ndim == agent_types.ndim + 1 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.shape != agent_types.shape:
            raise ValueError(
                'Actor sample mass shape mismatch: '
                f'mask={mask.shape}, agent_types={agent_types.shape}.'
            )
        role_mask = np.isin(agent_types, eligible_roles)
        return max(
            float((np.maximum(mask, 0.0) * role_mask).sum()),
            0.0,
        )

    @staticmethod
    def _backward_actor_loss(loss):
        """Backpropagate a finite differentiable loss, otherwise skip safely."""

        if not bool(torch.isfinite(loss).item()) or not loss.requires_grad:
            return False
        loss.backward()
        return True

    def _trainable_shared_actor_parameters(self):
        for group in self.policy.actor_optimizer.param_groups:
            if str(group.get('name', '')) != 'shared_encoder':
                continue
            return [
                parameter for parameter in group['params']
                if parameter.requires_grad
            ]
        return []

    @staticmethod
    def _gradient_dot(left, right):
        value = None
        for left_grad, right_grad in zip(left, right):
            if left_grad is None or right_grad is None:
                continue
            term = torch.sum(left_grad * right_grad)
            value = term if value is None else value + term
        return value

    @classmethod
    def _gradient_norm_sq(cls, gradients):
        value = cls._gradient_dot(gradients, gradients)
        if value is None:
            return 0.0
        return max(_finite_item(value), 0.0)

    @staticmethod
    def _scale_gradient_set(gradients, scale):
        return [
            None if gradient is None else gradient * float(scale)
            for gradient in gradients
        ]

    @staticmethod
    def _mean_gradient_sets(gradient_sets):
        if not gradient_sets:
            return []
        averaged = []
        for parameter_gradients in zip(*gradient_sets):
            present = [
                gradient for gradient in parameter_gradients
                if gradient is not None
            ]
            averaged.append(
                None if not present
                else sum(present[1:], present[0].clone()) / len(present)
            )
        return averaged

    @staticmethod
    def _sum_gradient_sets(gradient_sets):
        if not gradient_sets:
            return []
        combined = []
        for parameter_gradients in zip(*gradient_sets):
            present = [
                gradient for gradient in parameter_gradients
                if gradient is not None
            ]
            combined.append(
                None if not present
                else sum(present[1:], present[0].clone())
            )
        return combined

    @classmethod
    def _gradient_cosine(cls, left, right):
        denominator = math.sqrt(
            cls._gradient_norm_sq(left) * cls._gradient_norm_sq(right)
        )
        dot = cls._gradient_dot(left, right)
        if denominator <= 1e-24 or dot is None:
            return 0.0
        return float(np.clip(_finite_item(dot) / denominator, -1.0, 1.0))

    @staticmethod
    def _simplex_grid(task_count, resolution=100):
        """Deterministic low-dimensional simplex grid used by CAGrad.

        Stage3 has exactly three roles.  A 0.01 simplex grid has only 5,151
        points, so this avoids an optional SciPy dependency while making the
        CAGrad subproblem reproducible across worker processes.
        """

        if task_count == 1:
            return torch.ones((1, 1), dtype=torch.float64)
        if task_count == 2:
            return torch.tensor([
                (index / resolution, 1.0 - index / resolution)
                for index in range(resolution + 1)
            ], dtype=torch.float64)
        if task_count == 3:
            return torch.tensor([
                (
                    left / resolution,
                    middle / resolution,
                    (resolution - left - middle) / resolution,
                )
                for left in range(resolution + 1)
                for middle in range(resolution - left + 1)
            ], dtype=torch.float64)
        raise ValueError('CAGrad simplex solver supports at most three roles.')

    def _cagrad_combination(self, gradients, ordered_roles):
        """Return the CAGrad direction with sum-loss-compatible scaling."""

        task_count = len(ordered_roles)
        gram = torch.zeros((task_count, task_count), dtype=torch.float64)
        for left_index, left_role in enumerate(ordered_roles):
            for right_index, right_role in enumerate(ordered_roles):
                dot = self._gradient_dot(
                    gradients[left_role], gradients[right_role]
                )
                gram[left_index, right_index] = (
                    0.0 if dot is None else _finite_item(dot)
                )
        # Round-off can make a Gram matrix microscopically asymmetric.
        gram = 0.5 * (gram + gram.T)
        uniform = torch.full(
            (task_count,), 1.0 / task_count, dtype=torch.float64
        )
        g0_norm = math.sqrt(max(float(gram.mean().item()), 0.0))
        conflict_radius = float(self.shared_cagrad_c) * g0_norm
        candidates = self._simplex_grid(task_count)
        linear = candidates @ gram @ uniform
        quadratic = torch.einsum(
            'bi,ij,bj->b', candidates, gram, candidates
        ).clamp_min(0.0)
        objective = linear + conflict_radius * torch.sqrt(
            quadratic + 1e-24
        )
        weights = candidates[int(torch.argmin(objective).item())]
        weighted = self._sum_gradient_sets([
            self._scale_gradient_set(gradients[role], weights[index].item())
            for index, role in enumerate(ordered_roles)
        ])
        mean_gradient = self._scale_gradient_set(
            self._sum_gradient_sets([
                gradients[role] for role in ordered_roles
            ]),
            1.0 / task_count,
        )
        weighted_norm = math.sqrt(self._gradient_norm_sq(weighted))
        multiplier = conflict_radius / max(weighted_norm, 1e-12)
        direction = self._sum_gradient_sets([
            mean_gradient,
            self._scale_gradient_set(weighted, multiplier),
        ])
        # The joint Actor loss is a sum of role contributions.  Multiplying by
        # K retains that first-order scale and prevents the CAGrad arm from
        # winning merely because it silently reduced the effective LR.
        direction = self._scale_gradient_set(direction, task_count)
        return direction, {
            role: float(weights[index].item())
            for index, role in enumerate(ordered_roles)
        }

    def shared_gradient_state_dict(self):
        """Small checkpoint payload required for exact EMA continuation."""

        return {
            'version': 1,
            'method': self.shared_gradient_method,
            'norm_ema': {
                str(role): float(value)
                for role, value in self.shared_grad_norm_ema.items()
            },
            'updates': int(self.shared_grad_norm_ema_updates),
        }

    def load_shared_gradient_state_dict(self, state):
        if not state:
            return
        if int(state.get('version', 0)) != 1:
            raise ValueError('Unsupported shared-gradient checkpoint state.')
        if str(state.get('method', self.shared_gradient_method)) != (
            self.shared_gradient_method
        ):
            raise ValueError(
                'Shared-gradient checkpoint method does not match this run.'
            )
        self.shared_grad_norm_ema = {
            int(role): float(value)
            for role, value in dict(state.get('norm_ema', {})).items()
            if np.isfinite(float(value)) and float(value) >= 0.0
        }
        self.shared_grad_norm_ema_updates = max(
            0, int(state.get('updates', 0))
        )

    def _balanced_role_gradients(self, gradients, ordered_roles, metrics):
        current_norms = {
            role: math.sqrt(self._gradient_norm_sq(gradients[role]))
            for role in ordered_roles
        }
        for role in ordered_roles:
            current = current_norms[role]
            previous = self.shared_grad_norm_ema.get(role)
            self.shared_grad_norm_ema[role] = (
                current if previous is None
                else self.shared_grad_ema_beta * previous
                + (1.0 - self.shared_grad_ema_beta) * current
            )
        self.shared_grad_norm_ema_updates += 1
        positive_emas = [
            value for role, value in self.shared_grad_norm_ema.items()
            if role in ordered_roles and value > 1e-12
        ]
        target = float(np.median(positive_emas)) if positive_emas else 0.0
        balanced = {}
        for role in ordered_roles:
            ema = self.shared_grad_norm_ema[role]
            scale = 1.0
            if target > 0.0 and ema > 1e-12:
                scale = float(np.clip(
                    (target / ema) ** self.shared_grad_norm_power,
                    self.shared_grad_min_scale,
                    self.shared_grad_max_scale,
                ))
            balanced[role] = self._scale_gradient_set(
                gradients[role], scale
            )
            role_name = self.role_names[role]
            metrics[f'shared_{role_name}_grad_ema'] = float(ema)
            metrics[f'shared_{role_name}_grad_scale'] = float(scale)
            metrics[f'shared_{role_name}_balanced_grad_norm'] = (
                current_norms[role] * scale
            )
        metrics['shared_grad_balance_target_norm'] = target
        return balanced

    def _project_role_gradients(
        self, gradients, ordered_roles, *, threshold, symmetric
    ):
        """Project conflict components, optionally averaging all role orders."""

        projection_count = 0
        comparison_count = 0
        projected = {}
        for role in ordered_roles:
            other_roles = [item for item in ordered_roles if item != role]
            orders = (
                list(itertools.permutations(other_roles))
                if symmetric and len(other_roles) > 1
                else [tuple(other_roles)]
            )
            variants = []
            for order in orders:
                current = [
                    None if gradient is None else gradient.clone()
                    for gradient in gradients[role]
                ]
                for other_role in order:
                    other_norm_sq = self._gradient_norm_sq(
                        gradients[other_role]
                    )
                    if other_norm_sq <= 1e-24:
                        continue
                    comparison_count += 1
                    cosine = self._gradient_cosine(
                        current, gradients[other_role]
                    )
                    if cosine >= threshold:
                        continue
                    dot = self._gradient_dot(
                        current, gradients[other_role]
                    )
                    if dot is None:
                        continue
                    coefficient = dot / dot.new_tensor(
                        other_norm_sq
                    ).clamp_min(1e-24)
                    current = [
                        (
                            own - coefficient * other
                            if own is not None and other is not None
                            else own
                        )
                        for own, other in zip(
                            current, gradients[other_role]
                        )
                    ]
                    projection_count += 1
                variants.append(current)
            projected[role] = self._mean_gradient_sets(variants)
        return projected, projection_count, comparison_count

    def _shared_role_gradient_payload(
        self, role_losses, *, apply_pcgrad=None, gradient_method=None
    ):
        """Measure and combine role gradients on the shared encoder.

        Role losses already include the causal sample mass.  Ordinary backward
        still computes every head, entropy and critic contribution; this
        payload replaces only the shared policy-gradient component before the
        single atomic Actor optimizer step.
        """

        if gradient_method is None:
            gradient_method = (
                'legacy_pcgrad' if bool(apply_pcgrad)
                else self.shared_gradient_method
            )

        parameters = self._trainable_shared_actor_parameters()
        ordered_roles = [
            role for role in sorted(role_losses)
            if role_losses[role].requires_grad
        ]
        if not parameters or not ordered_roles:
            return None, {}

        gradients = {}
        for role in ordered_roles:
            gradients[role] = list(torch.autograd.grad(
                role_losses[role],
                parameters,
                retain_graph=True,
                allow_unused=True,
            ))

        norm_sq = {
            role: self._gradient_norm_sq(gradients[role])
            for role in ordered_roles
        }
        metrics = {'shared_gradient_diagnostic_samples': 1.0}
        for role in self.role_names:
            metrics[
                f"shared_{self.role_names[role]}_grad_norm"
            ] = math.sqrt(norm_sq.get(role, 0.0))

        conflict_count = 0
        comparable_pairs = 0
        pair_names = (
            (0, 1, 'plane_device'),
            (0, 2, 'plane_transporter'),
            (1, 2, 'device_transporter'),
        )
        for left_role, right_role, pair_name in pair_names:
            cosine = 0.0
            if left_role in gradients and right_role in gradients:
                denominator = math.sqrt(
                    norm_sq[left_role] * norm_sq[right_role]
                )
                dot = self._gradient_dot(
                    gradients[left_role], gradients[right_role]
                )
                if denominator > 1e-24 and dot is not None:
                    cosine = _finite_item(dot) / denominator
                    comparable_pairs += 1
                    conflict_count += int(cosine < 0.0)
            metrics[f'shared_{pair_name}_grad_cosine'] = float(cosine)
        metrics['shared_gradient_conflict_rate'] = (
            float(conflict_count) / comparable_pairs
            if comparable_pairs else 0.0
        )
        severe_conflicts = 0
        severe_comparable = 0
        for left_role, right_role, _ in pair_names:
            if left_role not in gradients or right_role not in gradients:
                continue
            if norm_sq[left_role] <= 1e-24 or norm_sq[right_role] <= 1e-24:
                continue
            severe_comparable += 1
            severe_conflicts += int(
                self._gradient_cosine(
                    gradients[left_role], gradients[right_role]
                ) < self.shared_grad_conflict_threshold
            )
        metrics['shared_gradient_severe_conflict_rate'] = (
            severe_conflicts / severe_comparable
            if severe_comparable else 0.0
        )
        metrics['shared_gradient_method_code'] = float({
            'sum': 0, 'legacy_pcgrad': 1, 'norm_balance': 2,
            'norm_pcgrad': 3, 'cagrad': 4,
        }[gradient_method])
        for role in self.role_names:
            role_name = self.role_names[role]
            metrics[f'shared_{role_name}_grad_ema'] = float(
                self.shared_grad_norm_ema.get(role, norm_sq.get(role, 0.0) ** 0.5)
            )
            metrics[f'shared_{role_name}_grad_scale'] = 1.0
            metrics[f'shared_{role_name}_balanced_grad_norm'] = math.sqrt(
                norm_sq.get(role, 0.0)
            )
            metrics[f'shared_cagrad_{role_name}_weight'] = 0.0
        metrics['shared_grad_balance_target_norm'] = 0.0
        metrics['shared_pcgrad_applied'] = 0.0
        metrics['shared_pcgrad_projection_fraction'] = 0.0
        metrics['shared_pcgrad_projection_rate'] = 0.0

        raw_sum = self._sum_gradient_sets([
            gradients[role] for role in ordered_roles
        ])
        target = raw_sum
        projection_count = 0
        comparison_count = 0
        if gradient_method == 'legacy_pcgrad' and len(ordered_roles) >= 2:
            projected, projection_count, comparison_count = (
                self._project_role_gradients(
                    gradients, ordered_roles, threshold=0.0, symmetric=False
                )
            )
            target = self._sum_gradient_sets([
                projected[role] for role in ordered_roles
            ])
        elif gradient_method in {'norm_balance', 'norm_pcgrad'}:
            balanced = self._balanced_role_gradients(
                gradients, ordered_roles, metrics
            )
            if gradient_method == 'norm_pcgrad' and len(ordered_roles) >= 2:
                balanced, projection_count, comparison_count = (
                    self._project_role_gradients(
                        balanced,
                        ordered_roles,
                        threshold=self.shared_grad_conflict_threshold,
                        symmetric=True,
                    )
                )
            target = self._sum_gradient_sets([
                balanced[role] for role in ordered_roles
            ])
        elif gradient_method == 'cagrad' and len(ordered_roles) >= 2:
            target, cagrad_weights = self._cagrad_combination(
                gradients, ordered_roles
            )
            for role, weight in cagrad_weights.items():
                metrics[
                    f'shared_cagrad_{self.role_names[role]}_weight'
                ] = weight

        if gradient_method == 'sum':
            metrics['shared_combined_grad_norm'] = math.sqrt(
                self._gradient_norm_sq(raw_sum)
            )
            metrics['shared_combined_to_raw_norm_ratio'] = 1.0
            for role in self.role_names:
                metrics[f'shared_combined_{self.role_names[role]}_cosine'] = (
                    self._gradient_cosine(raw_sum, gradients[role])
                    if role in gradients else 0.0
                )
            return None, metrics

        corrections = []
        raw_sum_sq = 0.0
        correction_sq = 0.0
        target_sum_sq = 0.0
        for raw_gradient, target_gradient in zip(raw_sum, target):
            if raw_gradient is None and target_gradient is None:
                corrections.append(None)
                continue
            if raw_gradient is None:
                correction = target_gradient.clone()
            elif target_gradient is None:
                correction = -raw_gradient
            else:
                correction = target_gradient - raw_gradient
            corrections.append(correction)
            if raw_gradient is not None:
                raw_sum_sq += _finite_item(torch.sum(
                    raw_gradient * raw_gradient
                ))
            if target_gradient is not None:
                target_sum_sq += _finite_item(torch.sum(
                    target_gradient * target_gradient
                ))
            correction_sq += _finite_item(torch.sum(correction * correction))

        metrics['shared_pcgrad_applied'] = float(projection_count > 0)
        metrics['shared_pcgrad_projection_fraction'] = math.sqrt(
            max(correction_sq, 0.0)
        ) / max(math.sqrt(max(raw_sum_sq, 0.0)), 1e-12)
        metrics['shared_pcgrad_projection_rate'] = (
            projection_count / comparison_count if comparison_count else 0.0
        )
        metrics['shared_combined_grad_norm'] = math.sqrt(
            max(target_sum_sq, 0.0)
        )
        metrics['shared_combined_to_raw_norm_ratio'] = math.sqrt(
            max(target_sum_sq, 0.0)
        ) / max(math.sqrt(max(raw_sum_sq, 0.0)), 1e-12)
        for role in self.role_names:
            metrics[f'shared_combined_{self.role_names[role]}_cosine'] = (
                self._gradient_cosine(target, gradients[role])
                if role in gradients else 0.0
            )
        return (parameters, corrections), metrics

    @staticmethod
    def _apply_shared_gradient_correction(payload):
        if payload is None:
            return
        parameters, corrections = payload
        for parameter, correction in zip(parameters, corrections):
            if correction is None:
                continue
            if parameter.grad is None:
                parameter.grad = correction.detach().clone()
            else:
                parameter.grad.add_(correction.detach())

    def _perform_actor_optimizer_step(self, *, grad_divisor=1.0):
        """Finish one accumulated atomic Actor update and expose true metrics."""

        if not self._actor_has_accumulated_grad():
            self.policy.actor_optimizer.zero_grad()
            return {
                'actor_grad_norm': 0.0,
                'actor_grad_norm_clipped': 0.0,
                'actor_grad_clip_applied': 0.0,
                'actor_optimizer_steps': 0.0,
                'actor_update_skipped': 1.0,
            }
        grad_divisor = float(grad_divisor)
        if not np.isfinite(grad_divisor) or grad_divisor <= 0.0:
            raise ValueError('Actor gradient divisor must be finite and positive.')
        if not np.isclose(grad_divisor, 1.0):
            for group in self.policy.actor_optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None:
                        parameter.grad.div_(grad_divisor)

        actor_snapshot = self._snapshot_actor_parameters()
        keep_atomic_state = bool(
            self.role_atomic_ppo
            or self.bc_reference_hard_gate
            or self.actor_kl_backtrack
        )
        self._last_actor_step_state = (
            self._snapshot_actor_step_state(actor_snapshot)
            if keep_atomic_state else None
        )
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.ac.actor_param.parameters(), self.max_grad_norm
            )
            actor_grad_norm_clipped = torch.clamp(
                actor_grad_norm.detach(), max=self.max_grad_norm
            )
            actor_grad_clip_applied = float(
                actor_grad_norm.detach().item() > self.max_grad_norm
            )
        else:
            actor_grad_norm = get_gard_norm(
                self.policy.ac.actor_param.parameters()
            )
            actor_grad_norm_clipped = actor_grad_norm
            actor_grad_clip_applied = 0.0

        if self._last_actor_step_state is not None:
            # Capture the normalized/clipped gradient once.  Every retry is
            # therefore the exact same Adam update with only the LR scaled,
            # rather than a newly sampled or newly accumulated gradient.
            self._last_actor_step_state['gradients'] = [
                (
                    parameter,
                    None if parameter.grad is None
                    else parameter.grad.detach().clone(),
                )
                for group in self.policy.actor_optimizer.param_groups
                for parameter in group['params']
                if parameter.requires_grad
            ]

        metrics = {
            'actor_grad_norm': _finite_item(actor_grad_norm),
            'actor_grad_norm_clipped': _finite_item(actor_grad_norm_clipped),
            'actor_grad_clip_applied': actor_grad_clip_applied,
            'actor_optimizer_steps': 0.0,
            'actor_update_skipped': 0.0,
        }
        if np.isfinite(_finite_item(actor_grad_norm)):
            self.policy.actor_optimizer.step()
            metrics['actor_optimizer_steps'] = 1.0
            metrics.update(self._actor_parameter_update_metrics(actor_snapshot))
        else:
            metrics['actor_update_skipped'] = 1.0
            self._last_actor_step_state = None
        self.policy.actor_optimizer.zero_grad()
        return metrics

    def _actor_has_accumulated_grad(self):
        return any(
            parameter.requires_grad and parameter.grad is not None
            for group in self.policy.actor_optimizer.param_groups
            for parameter in group['params']
        )

    def _snapshot_actor_parameters(self):
        """Clone trainable Actor parameters immediately before an optimizer step."""
        snapshot = []
        for group_idx, group in enumerate(self.policy.actor_optimizer.param_groups):
            group_name = str(group.get('name', f'group_{group_idx}'))
            params = [param for param in group['params'] if param.requires_grad]
            snapshot.append(
                (group_name, [(param, param.detach().clone()) for param in params])
            )
        return snapshot

    def _snapshot_actor_step_state(self, parameter_snapshot):
        """Capture the complete atomic boundary before ``Adam.step``."""

        return {
            'parameters': parameter_snapshot,
            'optimizer': copy.deepcopy(
                self.policy.actor_optimizer.state_dict()
            ),
            'base_lrs': [
                float(group['lr'])
                for group in self.policy.actor_optimizer.param_groups
            ],
        }

    def rollback_last_actor_step(self, *, preserve_state=False):
        """Restore parameters and Adam moments after a post-step KL rejection."""

        state = self._last_actor_step_state
        if state is None:
            raise RuntimeError('No actor optimizer step is available to roll back.')
        with torch.no_grad():
            for _, parameter_pairs in state['parameters']:
                for parameter, before in parameter_pairs:
                    parameter.copy_(before)
        self.policy.actor_optimizer.load_state_dict(state['optimizer'])
        self.policy.actor_optimizer.zero_grad()
        if not preserve_state:
            self._last_actor_step_state = None

    def retry_last_actor_step(self, scale):
        """Retry the last atomic Actor group from its exact pre-step state."""
        state = self._last_actor_step_state
        if state is None:
            raise RuntimeError('No actor optimizer step is available to retry.')
        scale = float(scale)
        if not np.isfinite(scale) or not 0.0 < scale < 1.0:
            raise ValueError('Actor backtrack scale must be finite and in (0, 1).')
        self.rollback_last_actor_step(preserve_state=True)
        gradients = state.get('gradients', ())
        for parameter, gradient in gradients:
            if gradient is not None:
                parameter.grad = gradient.detach().clone()
        base_lrs = state['base_lrs']
        if len(base_lrs) != len(self.policy.actor_optimizer.param_groups):
            raise RuntimeError('Actor optimizer group count changed during retry.')
        for group, base_lr in zip(
            self.policy.actor_optimizer.param_groups, base_lrs
        ):
            group['lr'] = float(base_lr) * scale
        self.policy.actor_optimizer.step()
        for group, base_lr in zip(
            self.policy.actor_optimizer.param_groups, base_lrs
        ):
            group['lr'] = float(base_lr)
        self.policy.actor_optimizer.zero_grad()
        metrics = {
            'actor_optimizer_steps': 1.0,
            'actor_backtrack_step_scale': scale,
        }
        metrics.update(self._actor_parameter_update_metrics(state['parameters']))
        return metrics

    def accept_last_actor_step(self):
        self._last_actor_step_state = None

    def _adapt_bc_reference_coefficient(self, observed_kl, sample_count):
        """Advance the soft BC dual controller once per rollout shard."""
        previous = float(self.bc_reference_kl_coef)
        observed_kl = float(observed_kl)
        sample_count = float(sample_count)
        if (
            not self.adaptive_bc_reference_kl
            or sample_count <= 0.0
            or not np.isfinite(observed_kl)
        ):
            return previous
        if observed_kl > self.adaptive_bc_reference_target_kl:
            updated = previous * self.adaptive_bc_reference_coef_up
        elif observed_kl < 0.5 * self.adaptive_bc_reference_target_kl:
            updated = previous * self.adaptive_bc_reference_coef_down
        else:
            updated = previous
        self.bc_reference_kl_coef = float(np.clip(
            updated,
            self.adaptive_bc_reference_coef_min,
            self.adaptive_bc_reference_coef_max,
        ))
        return self.bc_reference_kl_coef

    @staticmethod
    def _actor_parameter_update_metrics(snapshot):
        """Measure the actual optimizer displacement, globally and by LR group."""
        total_delta_sq = 0.0
        total_reference_sq = 0.0
        metrics = {}
        group_delta_squares = {}
        for group_name, params in snapshot:
            group_delta_sq = 0.0
            group_reference_sq = 0.0
            for param, before in params:
                after = param.detach()
                group_delta_sq += float(
                    torch.sum((after.double() - before.double()) ** 2).item()
                )
                group_reference_sq += float(
                    torch.sum(before.double() ** 2).item()
                )
            group_delta = math.sqrt(max(group_delta_sq, 0.0))
            group_relative = group_delta / max(
                math.sqrt(max(group_reference_sq, 0.0)),
                1e-12,
            )
            safe_name = ''.join(
                character if character.isalnum() else '_'
                for character in group_name
            ).strip('_') or 'unnamed'
            group_delta_squares[safe_name] = group_delta_sq
            metrics[f'actor_{safe_name}_update_l2'] = group_delta
            metrics[f'actor_{safe_name}_update_relative'] = group_relative
            total_delta_sq += group_delta_sq
            total_reference_sq += group_reference_sq

        total_delta = math.sqrt(max(total_delta_sq, 0.0))
        metrics['actor_param_update_l2'] = total_delta
        metrics['actor_param_update_relative'] = total_delta / max(
            math.sqrt(max(total_reference_sq, 0.0)),
            1e-12,
        )
        for safe_name, group_delta_sq in group_delta_squares.items():
            metrics[f'actor_{safe_name}_update_fraction'] = (
                group_delta_sq / max(total_delta_sq, 1e-24)
            )
        return metrics

    def _kl_gate_decision(self, old_policy_kl, bc_reference_kl):
        """Separate target monitoring from the two independently configured gates."""
        old_policy_kl = _finite_item(old_policy_kl)
        bc_reference_kl = _finite_item(bc_reference_kl)
        old_target_exceeded = bool(
            self.target_kl > 0.0 and old_policy_kl > self.target_kl
        )
        reference_target_exceeded = bool(
            self.bc_reference_target_kl > 0.0
            and bc_reference_kl > self.bc_reference_target_kl
        )
        reference_early_stop = bool(
            self.bc_reference_hard_gate and reference_target_exceeded
        )
        early_stop = old_target_exceeded or reference_early_stop
        return {
            'old_policy_target_exceeded': old_target_exceeded,
            'bc_reference_target_exceeded': reference_target_exceeded,
            'old_policy_kl_early_stop': old_target_exceeded,
            'bc_reference_kl_early_stop': reference_early_stop,
            'kl_early_stop': early_stop,
            'kl_stop_reason_code': (
                int(old_target_exceeded) + 2 * int(reference_early_stop)
            ),
        }

    def _bc_reference_kl_metrics(
        self,
        graph_batch,
        rnn_states_batch,
        active_masks_batch,
        last_op_batch,
        last_site_batch,
        actions_batch,
        agent_types_batch,
        current_action_log_probs,
        sample_weights,
        reference_outputs=None,
    ):
        enabled = (
            self.bc_reference_kl_coef > 0.0
            or self.bc_reference_target_kl > 0.0
        )
        zero = torch.zeros(
            (), dtype=current_action_log_probs.dtype,
            device=current_action_log_probs.device,
        )
        if not enabled:
            return zero, {
                'bc_reference_approx_kl': 0.0,
                'bc_reference_approx_kl_p95': 0.0,
                'bc_reference_abs_log_ratio_p95': 0.0,
            }
        if not self.policy.has_bc_reference():
            raise RuntimeError(
                'BC-reference PPO regularization is enabled, but no frozen '
                'post-BC policy has been captured.'
            )
        if reference_outputs is None:
            reference_outputs = self.policy.evaluate_bc_reference_actions(
                graph_batch,
                rnn_states_batch,
                active_masks_batch,
                last_op_batch,
                last_site_batch,
                actions_batch,
                agent_types=agent_types_batch,
                return_decision_mask=True,
            )
        reference_log_probs, _, reference_decision_mask = reference_outputs
        reference_log_probs = _sanitize_tensor(reference_log_probs)
        reference_weights = sample_weights * reference_decision_mask.unsqueeze(
            -1
        ).to(sample_weights.dtype)
        denominator = reference_weights.sum().clamp_min(1e-8)
        reference_log_ratio = torch.clamp(
            reference_log_probs.unsqueeze(-1)
            - current_action_log_probs.unsqueeze(-1),
            min=-20.0,
            max=20.0,
        )
        reference_ratio = torch.exp(reference_log_ratio)
        sampled_reference_kl = (
            reference_ratio - 1.0 - reference_log_ratio
        )
        reference_kl = (
            sampled_reference_kl * reference_weights
        ).sum() / denominator
        absolute_log_ratio = torch.abs(reference_log_ratio)
        return reference_kl, {
            'bc_reference_approx_kl': _finite_item(reference_kl),
            'bc_reference_approx_kl_p95': _finite_item(
                _weighted_quantile(
                    sampled_reference_kl, reference_weights, 0.95
                )
            ),
            'bc_reference_abs_log_ratio_p95': _finite_item(
                _weighted_quantile(
                    absolute_log_ratio, reference_weights, 0.95
                )
            ),
        }

    def _role_joint_views(
        self,
        action_log_probs,
        old_action_log_probs,
        advantages,
        policy_masks,
        agent_types,
        base_weights,
    ):
        """Return independent plane/device/R014 event views of one batch."""

        views = []
        for role, role_name in self.role_names.items():
            role_mask = policy_masks * (
                agent_types.unsqueeze(-1) == role
            ).to(policy_masks.dtype)
            role_count = role_mask.sum(dim=1).clamp_min(1.0)
            valid = (role_mask.sum(dim=1) > 0.0).to(policy_masks.dtype)
            if not bool(valid.any().item()):
                continue
            if self.case_balanced_loss:
                weights = (base_weights * role_mask).sum(dim=1) * valid
            else:
                weights = valid
            if not bool((weights.sum() > 0.0).item()):
                continue
            views.append({
                'role': role,
                'name': role_name,
                'coefficient': max(0.0, self.role_loss_coef[role]),
                'new_log_prob': (
                    action_log_probs * role_mask.squeeze(-1)
                ).sum(dim=1, keepdim=True),
                'old_log_prob': (
                    old_action_log_probs.squeeze(-1)
                    * role_mask.squeeze(-1)
                ).sum(dim=1, keepdim=True),
                'advantage': (
                    advantages * role_mask
                ).sum(dim=1) / role_count,
                'weights': weights,
                'valid': valid,
                'decision_count': role_mask.sum(),
            })
        return [view for view in views if view['coefficient'] > 0.0]

    def _role_kl_gate(self, metrics, prefix=''):
        exceeded = {}
        for role, role_name in self.role_names.items():
            target = self.role_target_kl[role]
            observed = float(metrics.get(
                f'{prefix}{role_name}_approx_kl', 0.0
            ))
            exceeded[role_name] = bool(target > 0.0 and observed > target)
        return exceeded

    def _role_view_loss_weights(self, views):
        """Return event weights for one role-atomic replay sample.

        Fixed weighting preserves A1's independently normalized role losses.
        The sqrt-event arm must instead preserve the per-case masses already
        encoded in ``policy_sample_weights``; renormalizing each role here
        would silently turn it back into fixed weighting.
        """

        if self.role_loss_weighting == 'sqrt_event':
            return [view['weights'] for view in views]
        coefficient_sum = sum(view['coefficient'] for view in views)
        return [
            view['weights']
            / view['weights'].sum().clamp_min(1e-8)
            * (view['coefficient'] / coefficient_sum)
            for view in views
        ]

    @torch.no_grad()
    def measure_post_update_policy_shift(self, sample):
        """Replay one probe minibatch after ``optimizer.step``.

        Ordinary PPO diagnostics are evaluated before a gradient-accumulation
        group steps. With one PPO epoch they are therefore expected to be nearly
        zero. This probe measures the policy that will collect the next rollout
        against the stored rollout log-probabilities.
        """
        graph_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, last_op_batch, last_site_batch, rewards_batch, \
        policy_masks_batch, agent_types_batch, \
        policy_sample_weights_batch, value_sample_weights_batch = sample

        action_log_probs, dist_entropy, replay_decision_mask = self.policy.evaluate_actions(
            graph_batch,
            rnn_states_batch,
            active_masks_batch,
            last_op_batch,
            last_site_batch,
            actions_batch,
            agent_types=agent_types_batch,
            return_decision_mask=True,
        )
        action_log_probs = _sanitize_tensor(action_log_probs)
        old_action_log_probs_batch = _sanitize_tensor(
            check(old_action_log_probs_batch).to(**self.tpdv)
        )
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        policy_masks_batch = check(policy_masks_batch).to(**self.tpdv)
        replay_decision_mask = replay_decision_mask.unsqueeze(-1).to(**self.tpdv)
        policy_masks_batch = active_masks_batch * policy_masks_batch * replay_decision_mask
        agent_types_batch = check(agent_types_batch).to(self.device, dtype=torch.long)
        policy_sample_weights_batch = check(policy_sample_weights_batch).to(**self.tpdv)
        reference_outputs = None
        if (
            self.bc_reference_kl_coef > 0.0
            or self.bc_reference_target_kl > 0.0
        ):
            reference_outputs = self.policy.evaluate_bc_reference_actions(
                graph_batch,
                rnn_states_batch,
                active_masks_batch,
                last_op_batch,
                last_site_batch,
                actions_batch,
                agent_types=agent_types_batch,
                return_decision_mask=True,
            )
        role_post_metrics = {}
        if self.role_atomic_ppo:
            role_views = self._role_joint_views(
                action_log_probs,
                old_action_log_probs_batch,
                torch.zeros_like(old_action_log_probs_batch),
                policy_masks_batch,
                agent_types_batch,
                policy_sample_weights_batch,
            )
            if not role_views:
                return {
                    'post_update_probe_approx_kl': 0.0,
                    'post_update_probe_approx_kl_p95': 0.0,
                    'post_update_probe_approx_kl_max': 0.0,
                    'post_update_probe_abs_log_ratio_p95': 0.0,
                    'post_update_probe_abs_log_ratio_max': 0.0,
                    'post_update_probe_clip_fraction': 0.0,
                    'post_update_probe_entropy': _finite_item(dist_entropy),
                    'post_update_probe_decisions': 0.0,
                    'post_update_bc_reference_approx_kl': 0.0,
                    'post_update_bc_reference_approx_kl_p95': 0.0,
                    'post_update_bc_reference_abs_log_ratio_p95': 0.0,
                    'post_update_empty_replay_samples': 1.0,
                }
            reference_views_by_role = {}
            if reference_outputs is not None:
                reference_log_probs, _, _ = reference_outputs
                reference_views_by_role = {
                    int(view['role']): view
                    for view in self._role_joint_views(
                        _sanitize_tensor(reference_log_probs),
                        old_action_log_probs_batch,
                        torch.zeros_like(old_action_log_probs_batch),
                        policy_masks_batch,
                        agent_types_batch,
                        policy_sample_weights_batch,
                    )
                }
            combined_view_weights = self._role_view_loss_weights(role_views)
            new_log_probs = []
            old_log_probs = []
            reference_log_probs_by_view = []
            normalized_weights = []
            for view, combined_weights in zip(
                role_views, combined_view_weights
            ):
                denominator = view['weights'].sum().clamp_min(1e-8)
                view_log_ratio = torch.clamp(
                    view['new_log_prob'] - view['old_log_prob'],
                    min=-20.0,
                    max=20.0,
                )
                view_ratio = torch.exp(view_log_ratio)
                view_kl = (view_ratio - 1.0) - view_log_ratio
                role_post_metrics.update({
                    f"post_update_{view['name']}_approx_kl": _finite_item(
                        (view_kl * view['weights']).sum() / denominator
                    ),
                    f"post_update_{view['name']}_effective_events": _finite_item(
                        view['valid'].sum()
                    ),
                })
                new_log_probs.append(view['new_log_prob'].reshape(-1))
                old_log_probs.append(view['old_log_prob'].reshape(-1, 1))
                if reference_outputs is not None:
                    reference_view = reference_views_by_role.get(
                        int(view['role'])
                    )
                    if reference_view is None:
                        raise RuntimeError(
                            'BC-reference post-update replay omitted an active '
                            f"role-atomic view: {view['name']}."
                        )
                    reference_log_probs_by_view.append(
                        reference_view['new_log_prob'].reshape(-1)
                    )
                normalized_weights.append(
                    combined_weights.reshape(-1, 1)
                )
            action_log_probs = torch.cat(new_log_probs, dim=0)
            old_action_log_probs_batch = torch.cat(old_log_probs, dim=0)
            sample_weights = torch.cat(normalized_weights, dim=0)
            policy_masks_batch = torch.ones_like(sample_weights)
            reference_metrics = {
                'bc_reference_approx_kl': 0.0,
                'bc_reference_approx_kl_p95': 0.0,
                'bc_reference_abs_log_ratio_p95': 0.0,
            }
            if reference_outputs is not None:
                reference_log_probs_flat = torch.cat(
                    reference_log_probs_by_view, dim=0
                )
                reference_log_ratio = torch.clamp(
                    reference_log_probs_flat.unsqueeze(-1)
                    - action_log_probs.unsqueeze(-1),
                    min=-20.0,
                    max=20.0,
                )
                reference_ratio = torch.exp(reference_log_ratio)
                sampled_reference_kl = (
                    reference_ratio - 1.0 - reference_log_ratio
                )
                reference_denominator = sample_weights.sum().clamp_min(1e-8)
                reference_kl = (
                    sampled_reference_kl * sample_weights
                ).sum() / reference_denominator
                reference_metrics = {
                    'bc_reference_approx_kl': _finite_item(reference_kl),
                    'bc_reference_approx_kl_p95': _finite_item(
                        _weighted_quantile(
                            sampled_reference_kl, sample_weights, 0.95
                        )
                    ),
                    'bc_reference_abs_log_ratio_p95': _finite_item(
                        _weighted_quantile(
                            torch.abs(reference_log_ratio),
                            sample_weights,
                            0.95,
                        )
                    ),
                }
        else:
            sample_weights = self._role_sample_weights(
                policy_masks_batch,
                agent_types_batch,
                base_weights=(
                    policy_sample_weights_batch if self.case_balanced_loss else None
                ),
            )
            _, reference_metrics = self._bc_reference_kl_metrics(
                graph_batch,
                rnn_states_batch,
                active_masks_batch,
                last_op_batch,
                last_site_batch,
                actions_batch,
                agent_types_batch,
                action_log_probs,
                sample_weights,
                reference_outputs=reference_outputs,
            )
        weight_denominator = sample_weights.sum().clamp_min(1e-8)
        log_ratio = torch.clamp(
            action_log_probs.unsqueeze(-1) - old_action_log_probs_batch,
            min=-20.0,
            max=20.0,
        )
        imp_weights = torch.exp(log_ratio)
        sampled_approx_kl = (imp_weights - 1.0) - log_ratio
        sampled_abs_log_ratio = torch.abs(log_ratio)
        positive_weight_mask = sample_weights > 0.0
        if positive_weight_mask.any():
            approx_kl_max = sampled_approx_kl[positive_weight_mask].max()
            abs_log_ratio_max = sampled_abs_log_ratio[positive_weight_mask].max()
        else:
            approx_kl_max = torch.zeros(
                (), dtype=sampled_approx_kl.dtype, device=sampled_approx_kl.device
            )
            abs_log_ratio_max = torch.zeros(
                (), dtype=sampled_abs_log_ratio.dtype, device=sampled_abs_log_ratio.device
            )
        results = {
            'post_update_probe_approx_kl': _finite_item(
                (sampled_approx_kl * sample_weights).sum() / weight_denominator
            ),
            'post_update_probe_approx_kl_p95': _finite_item(
                _weighted_quantile(sampled_approx_kl, sample_weights, 0.95)
            ),
            'post_update_probe_approx_kl_max': _finite_item(approx_kl_max),
            'post_update_probe_abs_log_ratio_p95': _finite_item(
                _weighted_quantile(sampled_abs_log_ratio, sample_weights, 0.95)
            ),
            'post_update_probe_abs_log_ratio_max': _finite_item(abs_log_ratio_max),
            'post_update_probe_clip_fraction': _finite_item(
                (
                    (torch.abs(imp_weights - 1.0) > self.clip_param).float()
                    * sample_weights
                ).sum() / weight_denominator
            ),
            'post_update_probe_entropy': _finite_item(dist_entropy),
            'post_update_probe_decisions': _finite_item(policy_masks_batch.sum()),
        }
        results.update(
            {
                f'post_update_{key}': value
                for key, value in reference_metrics.items()
            }
        )
        results.update(role_post_metrics)
        return results

    @torch.no_grad()
    def _preceding_role_importance(self, sample, preceding_roles, target_role):
        """Return the same-event HAPPO correction and its normalized ESS."""

        graph_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, last_op_batch, last_site_batch, rewards_batch, \
        policy_masks_batch, agent_types_batch, \
        policy_sample_weights_batch, value_sample_weights_batch = sample
        action_log_probs, _, replay_decision_mask = self.policy.evaluate_actions(
            graph_batch,
            rnn_states_batch,
            active_masks_batch,
            last_op_batch,
            last_site_batch,
            actions_batch,
            agent_types=agent_types_batch,
            return_decision_mask=True,
        )
        action_log_probs = _sanitize_tensor(action_log_probs)
        old_log_probs = _sanitize_tensor(
            check(old_action_log_probs_batch).to(**self.tpdv)
        ).squeeze(-1)
        active_masks = check(active_masks_batch).to(**self.tpdv)
        policy_masks = check(policy_masks_batch).to(**self.tpdv)
        replay_mask = replay_decision_mask.unsqueeze(-1).to(**self.tpdv)
        policy_masks = active_masks * policy_masks * replay_mask
        agent_types = check(agent_types_batch).to(
            self.device, dtype=torch.long
        )
        batch_size = action_log_probs.shape[0]
        factor = torch.ones(
            (batch_size, 1), dtype=action_log_probs.dtype,
            device=action_log_probs.device,
        )
        for role in preceding_roles:
            role_mask = policy_masks.squeeze(-1) * (agent_types == int(role))
            valid = role_mask.sum(dim=1, keepdim=True) > 0.0
            log_ratio = (
                (action_log_probs - old_log_probs) * role_mask
            ).sum(dim=1, keepdim=True)
            ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
            factor = factor * torch.where(valid, ratio, torch.ones_like(ratio))
        factor = torch.clamp(
            torch.nan_to_num(factor, nan=1.0, posinf=1.0, neginf=1.0),
            min=1.0 / self.role_sequential_factor_clip,
            max=self.role_sequential_factor_clip,
        )
        target_valid = (
            policy_masks.squeeze(-1)
            * (agent_types == int(target_role))
        ).sum(dim=1) > 0.0
        valid_factors = factor.reshape(-1)[target_valid]
        if valid_factors.numel() == 0:
            return factor, 1.0, 0
        ess = (
            valid_factors.sum().square()
            / (
                float(valid_factors.numel())
                * valid_factors.square().sum().clamp_min(1e-12)
            )
        )
        return factor, _finite_item(ess), int(valid_factors.numel())

    def _actor_trainability_snapshot(self):
        return {
            parameter: bool(parameter.requires_grad)
            for group in self.policy.actor_optimizer.param_groups
            for parameter in group['params']
        }

    @staticmethod
    def _restore_actor_trainability(snapshot):
        for parameter, trainable in snapshot.items():
            parameter.requires_grad_(trainable)

    def _select_actor_optimizer_groups(self, enabled_names, original):
        enabled_names = set(enabled_names)
        for group in self.policy.actor_optimizer.param_groups:
            enabled = str(group.get('name', '')) in enabled_names
            for parameter in group['params']:
                parameter.requires_grad_(
                    bool(original.get(parameter, False) and enabled)
                )

    def _sequential_actor_group_update(
        self,
        group,
        masses,
        *,
        order,
        diagnose_shared_gradients=False,
    ):
        """Apply three role-head steps plus one shared step as one macro step.

        The complete parameter/Adam state is snapshotted once.  Any invalid
        gradient, low-ESS preceding-role factor, or later KL rejection restores
        that macro boundary, so partial role updates can never escape.
        """

        original = self._actor_trainability_snapshot()
        macro_parameters = self._snapshot_actor_parameters()
        macro_state = self._snapshot_actor_step_state(macro_parameters)
        metrics = defaultdict(float)
        valid_any = False
        role_group_names = {
            0: 'plane_actor', 1: 'device_actor', 2: 'transporter_actor'
        }
        preceding_roles = []
        minimum_ess = 1.0
        try:
            for role in order:
                self._select_actor_optimizer_groups(
                    {role_group_names[int(role)]}, original
                )
                self.policy.actor_optimizer.zero_grad()
                valid_mass = 0.0
                for sample, sample_mass in zip(group, masses):
                    factor, ess, event_count = self._preceding_role_importance(
                        sample, preceding_roles, role
                    )
                    minimum_ess = min(minimum_ess, ess)
                    metrics['role_sequential_factor_samples'] += float(
                        event_count
                    )
                    metrics['role_sequential_factor_ess_sum'] += float(ess)
                    results = self.update_policy_net(
                        sample,
                        update_actor=True,
                        perform_step=False,
                        loss_scale=sample_mass,
                        active_role=role,
                        external_importance_factor=factor,
                    )
                    for key, value in results.items():
                        metrics[key] += value
                    if results.get('actor_replay_valid', 0.0) > 0.0:
                        valid_mass += sample_mass
                if minimum_ess < self.role_sequential_min_ess:
                    raise RuntimeError(
                        'role-sequential importance ESS fell below the '
                        f'configured gate: {minimum_ess:.6f} < '
                        f'{self.role_sequential_min_ess:.6f}'
                    )
                step = self._perform_actor_optimizer_step(
                    grad_divisor=max(valid_mass, 1e-8)
                )
                if valid_mass > 0.0 and step.get(
                    'actor_optimizer_steps', 0.0
                ) <= 0.0:
                    metrics[
                        f"{self.role_names[int(role)]}_zero_step_chunks"
                    ] += 1.0
                metrics['actor_optimizer_substeps'] += step.get(
                    'actor_optimizer_steps', 0.0
                )
                valid_any |= step.get('actor_optimizer_steps', 0.0) > 0.0
                preceding_roles.append(int(role))

            # Shared representation receives a single equal-role/PCGrad pass,
            # after all role heads have been updated.  A zero-LR/frozen shared
            # group is skipped without invalidating the macro update.
            shared_trainable = any(
                original.get(parameter, False)
                for group_spec in self.policy.actor_optimizer.param_groups
                if str(group_spec.get('name', '')) == 'shared_encoder'
                for parameter in group_spec['params']
            )
            if shared_trainable:
                self._select_actor_optimizer_groups({'shared_encoder'}, original)
                self.policy.actor_optimizer.zero_grad()
                valid_mass = 0.0
                for sample, sample_mass in zip(group, masses):
                    results = self.update_policy_net(
                        sample,
                        update_actor=True,
                        perform_step=False,
                        loss_scale=sample_mass,
                        diagnose_shared_gradients=diagnose_shared_gradients,
                    )
                    for key, value in results.items():
                        metrics[key] += value
                    if results.get('actor_replay_valid', 0.0) > 0.0:
                        valid_mass += sample_mass
                step = self._perform_actor_optimizer_step(
                    grad_divisor=max(valid_mass, 1e-8)
                )
                if valid_mass > 0.0 and step.get(
                    'actor_optimizer_steps', 0.0
                ) <= 0.0:
                    metrics['shared_encoder_zero_step_chunks'] += 1.0
                metrics['actor_optimizer_substeps'] += step.get(
                    'actor_optimizer_steps', 0.0
                )
                valid_any |= step.get('actor_optimizer_steps', 0.0) > 0.0

            self._last_actor_step_state = macro_state
            metrics['actor_optimizer_steps'] = float(valid_any)
            metrics['actor_update_skipped'] = float(not valid_any)
            # A role can legitimately be absent from one recurrent chunk.
            # That is not an empty *macro* replay sample as long as another
            # role produced the atomic update; do not poison the global
            # zero-update/empty-replay health gate with role sparsity.
            metrics['actor_empty_replay_samples'] = float(not valid_any)
            metrics['actor_replay_valid'] = float(valid_any)
            metrics['role_sequential_min_ess'] = float(minimum_ess)
            metrics['role_sequential_macro_steps'] = float(valid_any)
            return dict(metrics)
        except Exception:
            self._last_actor_step_state = macro_state
            self.rollback_last_actor_step()
            raise
        finally:
            self._restore_actor_trainability(original)
            self.policy.actor_optimizer.zero_grad()

    def update_policy_net(
        self,
        sample,
        update_actor=True,
        perform_step=True,
        loss_scale=1.0,
        diagnose_shared_gradients=False,
        active_role=None,
        external_importance_factor=None,
    ):
        graph_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, last_op_batch, last_site_batch, rewards_batch, \
        policy_masks_batch, agent_types_batch, \
        policy_sample_weights_batch, value_sample_weights_batch = sample
        
        # Evaluate the frozen reference first. Its no-grad activations are
        # released before the trainable policy builds a backward graph, which
        # keeps the 350-graph formal configuration inside the tested VRAM band.
        reference_outputs = None
        if (
            self.bc_reference_kl_coef > 0.0
            or self.bc_reference_target_kl > 0.0
        ):
            reference_outputs = self.policy.evaluate_bc_reference_actions(
                graph_batch,
                rnn_states_batch,
                active_masks_batch,
                last_op_batch,
                last_site_batch,
                actions_batch,
                agent_types=agent_types_batch,
                return_decision_mask=True,
            )

        # Reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, replay_decision_mask = self.policy.evaluate_actions(
            graph_batch,
            rnn_states_batch,
            active_masks_batch,
            last_op_batch,
            last_site_batch,
            actions_batch,
            agent_types=agent_types_batch,
            return_decision_mask=True,
        )
        action_log_probs = _sanitize_tensor(action_log_probs)
        dist_entropy = _sanitize_tensor(dist_entropy)
        
        old_action_log_probs_batch = _sanitize_tensor(check(old_action_log_probs_batch).to(**self.tpdv))
        adv_targ = _sanitize_tensor(check(adv_targ).to(**self.tpdv))
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        policy_masks_batch = check(policy_masks_batch).to(**self.tpdv)
        replay_decision_mask = replay_decision_mask.unsqueeze(-1).to(**self.tpdv)
        policy_masks_batch = active_masks_batch * policy_masks_batch * replay_decision_mask
        agent_types_batch = check(agent_types_batch).to(self.device, dtype=torch.long)
        policy_sample_weights_batch = check(policy_sample_weights_batch).to(**self.tpdv)
        role_atomic_metrics = {}
        role_gradient_slices = []
        if self.role_atomic_ppo:
            role_views = self._role_joint_views(
                action_log_probs,
                old_action_log_probs_batch,
                adv_targ,
                policy_masks_batch,
                agent_types_batch,
                policy_sample_weights_batch,
            )
            if active_role is not None:
                active_role = int(active_role)
                role_views = [
                    view for view in role_views
                    if int(view['role']) == active_role
                ]
            if not role_views:
                return {
                    'policy_loss': 0.0,
                    'actor_grad_norm': 0.0,
                    'dist_entropy': _finite_item(dist_entropy),
                    'actor_grad_norm_clipped': 0.0,
                    'actor_grad_clip_applied': 0.0,
                    'advantages': 0.0,
                    'rewards': 0.0,
                    'actor_update_skipped': 1.0,
                    'actor_no_grad_samples': 1.0,
                    'actor_optimizer_steps': 0.0,
                    'kl_early_stop': 0.0,
                    'old_policy_kl_early_stop': 0.0,
                    'bc_reference_kl_early_stop': 0.0,
                    'old_policy_target_exceeded': 0.0,
                    'bc_reference_target_exceeded': 0.0,
                    'old_policy_kl_stop_events': 0.0,
                    'bc_reference_kl_stop_events': 0.0,
                    'actor_kl_stop_events': 0.0,
                    'bc_reference_kl_penalty': 0.0,
                    'joint_team_ppo': float(self.joint_team_ppo),
                    'joint_team_ppo_all_roles': 1.0,
                    'role_atomic_ppo': 1.0,
                    'approx_kl': 0.0,
                    'sampled_approx_kl_p95': 0.0,
                    'sampled_approx_kl_max': 0.0,
                    'sampled_abs_log_ratio_p95': 0.0,
                    'sampled_abs_log_ratio_max': 0.0,
                    'clip_fraction': 0.0,
                    'actor_effective_decisions': 0.0,
                    'actor_effective_weight_mass': 0.0,
                    'trainable_decision_fraction': 0.0,
                    'actor_empty_replay_samples': 1.0,
                    'actor_replay_valid': 0.0,
                }
            reference_views_by_role = {}
            if reference_outputs is not None:
                reference_log_probs, _, _ = reference_outputs
                reference_views_by_role = {
                    int(view['role']): view
                    for view in self._role_joint_views(
                        _sanitize_tensor(reference_log_probs),
                        old_action_log_probs_batch,
                        adv_targ,
                        policy_masks_batch,
                        agent_types_batch,
                        policy_sample_weights_batch,
                    )
                }
            combined_view_weights = self._role_view_loss_weights(role_views)
            new_log_probs = []
            old_log_probs = []
            reference_log_probs_by_view = []
            role_advantages = []
            normalized_weights = []
            for view, combined_weights in zip(
                role_views, combined_view_weights
            ):
                denominator = view['weights'].sum().clamp_min(1e-8)
                view_log_ratio = torch.clamp(
                    view['new_log_prob'] - view['old_log_prob'],
                    min=-20.0,
                    max=20.0,
                )
                view_ratio = torch.exp(view_log_ratio)
                view_kl = (view_ratio - 1.0) - view_log_ratio
                role_atomic_metrics.update({
                    f"{view['name']}_approx_kl": _finite_item(
                        (view_kl * view['weights']).sum() / denominator
                    ),
                    f"{view['name']}_effective_events": _finite_item(
                        view['valid'].sum()
                    ),
                    f"{view['name']}_effective_decisions": _finite_item(
                        view['decision_count']
                    ),
                    f"{view['name']}_effective_weight_mass": _finite_item(
                        view['weights'].sum()
                    ),
                })
                new_log_probs.append(view['new_log_prob'].reshape(-1))
                old_log_probs.append(view['old_log_prob'].reshape(-1, 1))
                if reference_outputs is not None:
                    reference_view = reference_views_by_role.get(
                        int(view['role'])
                    )
                    if reference_view is None:
                        raise RuntimeError(
                            'BC-reference replay omitted an active role-atomic '
                            f"view: {view['name']}."
                        )
                    reference_log_probs_by_view.append(
                        reference_view['new_log_prob'].reshape(-1)
                    )
                role_advantages.append(view['advantage'].reshape(-1, 1))
                normalized_weights.append(combined_weights.reshape(-1, 1))
                slice_start = sum(
                    item[3] - item[2] for item in role_gradient_slices
                )
                slice_end = slice_start + int(view['new_log_prob'].numel())
                role_gradient_slices.append((
                    int(view['role']), view['name'], slice_start, slice_end
                ))
            action_log_probs = torch.cat(new_log_probs, dim=0)
            old_action_log_probs_batch = torch.cat(old_log_probs, dim=0)
            adv_targ = torch.cat(role_advantages, dim=0)
            sample_weights = torch.cat(normalized_weights, dim=0)
            policy_masks_batch = torch.ones_like(sample_weights)
            zero = torch.zeros(
                (), dtype=action_log_probs.dtype,
                device=action_log_probs.device,
            )
            bc_reference_kl = zero
            reference_metrics = {
                'bc_reference_approx_kl': 0.0,
                'bc_reference_approx_kl_p95': 0.0,
                'bc_reference_abs_log_ratio_p95': 0.0,
            }
            if reference_outputs is not None:
                reference_log_probs_flat = torch.cat(
                    reference_log_probs_by_view, dim=0
                )
                reference_log_ratio = torch.clamp(
                    reference_log_probs_flat.unsqueeze(-1)
                    - action_log_probs.unsqueeze(-1),
                    min=-20.0,
                    max=20.0,
                )
                reference_ratio = torch.exp(reference_log_ratio)
                sampled_reference_kl = (
                    reference_ratio - 1.0 - reference_log_ratio
                )
                reference_denominator = sample_weights.sum().clamp_min(1e-8)
                bc_reference_kl = (
                    sampled_reference_kl * sample_weights
                ).sum() / reference_denominator
                reference_metrics = {
                    'bc_reference_approx_kl': _finite_item(
                        bc_reference_kl
                    ),
                    'bc_reference_approx_kl_p95': _finite_item(
                        _weighted_quantile(
                            sampled_reference_kl, sample_weights, 0.95
                        )
                    ),
                    'bc_reference_abs_log_ratio_p95': _finite_item(
                        _weighted_quantile(
                            torch.abs(reference_log_ratio),
                            sample_weights,
                            0.95,
                        )
                    ),
                }
                cursor = 0
                for view, weights in zip(role_views, normalized_weights):
                    width = int(view['new_log_prob'].numel())
                    local_kl = sampled_reference_kl[cursor:cursor + width]
                    local_denominator = weights.sum().clamp_min(1e-8)
                    role_atomic_metrics[
                        f"{view['name']}_bc_reference_approx_kl"
                    ] = _finite_item(
                        (local_kl * weights).sum() / local_denominator
                    )
                    cursor += width
        elif self.joint_team_ppo:
            if self.joint_team_ppo_scope == 'plane':
                joint_role_mask = policy_masks_batch * (
                    agent_types_batch.unsqueeze(-1) == 0
                ).to(policy_masks_batch.dtype)
            else:
                joint_role_mask = policy_masks_batch
            joint_count = joint_role_mask.sum(dim=1).clamp_min(1.0)
            joint_valid = (joint_role_mask.sum(dim=1) > 0.0).to(
                policy_masks_batch.dtype
            )
            action_log_probs = (
                action_log_probs * joint_role_mask.squeeze(-1)
            ).sum(dim=1)
            old_action_log_probs_batch = (
                old_action_log_probs_batch.squeeze(-1)
                * joint_role_mask.squeeze(-1)
            ).sum(dim=1, keepdim=True)
            adv_targ = (
                adv_targ * joint_role_mask
            ).sum(dim=1) / joint_count
            if self.case_balanced_loss:
                sample_weights = (
                    policy_sample_weights_batch * joint_role_mask
                ).sum(dim=1)
            else:
                sample_weights = joint_valid
            sample_weights = sample_weights * joint_valid
            policy_masks_batch = joint_valid

            zero = torch.zeros(
                (), dtype=action_log_probs.dtype,
                device=action_log_probs.device,
            )
            bc_reference_kl = zero
            reference_metrics = {
                'bc_reference_approx_kl': 0.0,
                'bc_reference_approx_kl_p95': 0.0,
                'bc_reference_abs_log_ratio_p95': 0.0,
            }
            if reference_outputs is not None:
                reference_log_probs, _, _ = reference_outputs
                reference_joint_log_probs = (
                    _sanitize_tensor(reference_log_probs)
                    * joint_role_mask.squeeze(-1)
                ).sum(dim=1)
                reference_log_ratio = torch.clamp(
                    reference_joint_log_probs - action_log_probs,
                    min=-20.0,
                    max=20.0,
                ).unsqueeze(-1)
                reference_ratio = torch.exp(reference_log_ratio)
                sampled_reference_kl = (
                    reference_ratio - 1.0 - reference_log_ratio
                )
                reference_denominator = sample_weights.sum().clamp_min(1e-8)
                bc_reference_kl = (
                    sampled_reference_kl * sample_weights
                ).sum() / reference_denominator
                reference_metrics = {
                    'bc_reference_approx_kl': _finite_item(bc_reference_kl),
                    'bc_reference_approx_kl_p95': _finite_item(
                        _weighted_quantile(
                            sampled_reference_kl, sample_weights, 0.95
                        )
                    ),
                    'bc_reference_abs_log_ratio_p95': _finite_item(
                        _weighted_quantile(
                            torch.abs(reference_log_ratio), sample_weights, 0.95
                        )
                    ),
                }
        else:
            sample_weights = self._role_sample_weights(
                policy_masks_batch,
                agent_types_batch,
                base_weights=(
                    policy_sample_weights_batch
                    if self.case_balanced_loss else None
                ),
            )
            bc_reference_kl, reference_metrics = self._bc_reference_kl_metrics(
                graph_batch,
                rnn_states_batch,
                active_masks_batch,
                last_op_batch,
                last_site_batch,
                actions_batch,
                agent_types_batch,
                action_log_probs,
                sample_weights,
                reference_outputs=reference_outputs,
            )
        weight_denominator = sample_weights.sum().clamp_min(1e-8)
        rewards_batch = _sanitize_tensor(check(rewards_batch).to(**self.tpdv))
        active_denominator = active_masks_batch.sum().clamp_min(1.0)
        rewards = (rewards_batch * active_masks_batch).sum() / active_denominator
        
        # actor update
        log_ratio = torch.clamp(action_log_probs.unsqueeze(-1) - old_action_log_probs_batch, min=-20.0, max=20.0)
        imp_weights = torch.exp(log_ratio)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        sequential_factor = None
        if external_importance_factor is not None:
            sequential_factor = check(external_importance_factor).to(**self.tpdv)
            sequential_factor = sequential_factor.reshape(-1, 1)
            if sequential_factor.shape != surr1.shape:
                raise RuntimeError(
                    'Sequential importance factor shape mismatch: '
                    f'factor={tuple(sequential_factor.shape)}, '
                    f'surrogate={tuple(surr1.shape)}.'
                )
            sequential_factor = torch.clamp(
                torch.nan_to_num(
                    sequential_factor, nan=1.0, posinf=1.0, neginf=1.0
                ),
                min=1.0 / self.role_sequential_factor_clip,
                max=self.role_sequential_factor_clip,
            )
            surr1 = surr1 * sequential_factor
            surr2 = surr2 * sequential_factor

        role_shared_losses = {}
        if role_gradient_slices and (
            self.shared_encoder_pcgrad
            or self.shared_gradient_method != 'sum'
            or diagnose_shared_gradients
        ):
            minimum_surrogate = torch.min(surr1, surr2)
            for role, _, slice_start, slice_end in role_gradient_slices:
                role_shared_losses[role] = (
                    -torch.sum(
                        minimum_surrogate[slice_start:slice_end],
                        dim=-1,
                        keepdim=True,
                    )
                    * sample_weights[slice_start:slice_end]
                ).sum() / weight_denominator * float(loss_scale)
        
        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * sample_weights
            ).sum() / weight_denominator
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = (
            policy_action_loss
            + self.bc_reference_kl_coef * bc_reference_kl
        )
        sampled_approx_kl = (imp_weights - 1.0) - log_ratio
        approx_kl = (
            (sampled_approx_kl * sample_weights).sum()
            / weight_denominator
        )
        approx_kl_p95 = _weighted_quantile(sampled_approx_kl, sample_weights, 0.95)
        sampled_abs_log_ratio = torch.abs(log_ratio)
        abs_log_ratio_p95 = _weighted_quantile(sampled_abs_log_ratio, sample_weights, 0.95)
        positive_weight_mask = sample_weights > 0.0
        if positive_weight_mask.any():
            approx_kl_max = sampled_approx_kl[positive_weight_mask].max()
            abs_log_ratio_max = sampled_abs_log_ratio[positive_weight_mask].max()
        else:
            approx_kl_max = torch.zeros(
                (), dtype=sampled_approx_kl.dtype, device=sampled_approx_kl.device
            )
            abs_log_ratio_max = torch.zeros(
                (), dtype=sampled_abs_log_ratio.dtype, device=sampled_abs_log_ratio.device
            )
        clip_fraction = (
            ((torch.abs(imp_weights - 1.0) > self.clip_param).float() * sample_weights).sum()
            / weight_denominator
        )
        gate = self._kl_gate_decision(approx_kl, bc_reference_kl)
        role_gate = self._role_kl_gate(role_atomic_metrics)
        if self.role_atomic_ppo and any(role_gate.values()):
            gate['old_policy_target_exceeded'] = True
            gate['old_policy_kl_early_stop'] = True
            gate['kl_early_stop'] = True
            gate['kl_stop_reason_code'] = int(gate['kl_stop_reason_code']) | 1
        if self.actor_kl_backtrack:
            # A pre-step KL observation is still reported, but it must not
            # discard the accumulated gradient: the atomic post-step probe
            # below is the authoritative accept/retry decision.  A separately
            # requested BC hard gate retains its legacy semantics.
            gate['old_policy_kl_early_stop'] = False
            gate['kl_early_stop'] = bool(
                gate['bc_reference_kl_early_stop']
            )
            gate['kl_stop_reason_code'] = (
                2 if gate['bc_reference_kl_early_stop'] else 0
            )
        old_policy_kl_early_stop = gate['old_policy_kl_early_stop']
        reference_kl_early_stop = gate['bc_reference_kl_early_stop']
        kl_early_stop = gate['kl_early_stop']

        actor_update_skipped = 0.0
        actor_no_grad_sample = 0.0
        actor_optimizer_step = 0.0
        actor_update_metrics = {}
        shared_gradient_payload = None
        shared_gradient_metrics = {}
        if update_actor and not kl_early_stop:
            # Accumulation is normalized by the actual decision mass of the
            # current group. A short final group therefore keeps full weight.
            loss = (
                policy_loss - dist_entropy * self.entropy_coef
            ) * float(loss_scale)
            if role_shared_losses:
                (
                    shared_gradient_payload,
                    shared_gradient_metrics,
                ) = self._shared_role_gradient_payload(
                    role_shared_losses,
                    apply_pcgrad=self.shared_encoder_pcgrad,
                    gradient_method=(
                        None if self.shared_encoder_pcgrad
                        else self.shared_gradient_method
                    ),
                )
            if not self._backward_actor_loss(loss):
                actor_update_skipped = 1.0
                actor_no_grad_sample = float(
                    bool(torch.isfinite(loss).item())
                    and not loss.requires_grad
                )
            else:
                self._apply_shared_gradient_correction(
                    shared_gradient_payload
                )
        elif kl_early_stop:
            actor_update_skipped = 1.0
            # Discard any partial accumulation from this group.
            self.policy.actor_optimizer.zero_grad()

        actor_grad_norm = 0.0
        actor_grad_norm_clipped = 0.0
        actor_grad_clip_applied = 0.0
        if perform_step and update_actor and not kl_early_stop:
            step_metrics = self._perform_actor_optimizer_step()
            actor_grad_norm = step_metrics.pop('actor_grad_norm')
            actor_grad_norm_clipped = step_metrics.pop(
                'actor_grad_norm_clipped'
            )
            actor_grad_clip_applied = step_metrics.pop(
                'actor_grad_clip_applied'
            )
            actor_optimizer_step = step_metrics.pop('actor_optimizer_steps')
            actor_update_skipped = max(
                actor_update_skipped,
                step_metrics.pop('actor_update_skipped'),
            )
            actor_update_metrics.update(step_metrics)

        results = {
            "policy_loss": _finite_item(policy_loss),
            "actor_grad_norm": _finite_item(actor_grad_norm),
            "dist_entropy": _finite_item(dist_entropy),
            "actor_grad_norm_clipped": _finite_item(actor_grad_norm_clipped),
            "actor_grad_clip_applied": actor_grad_clip_applied,
            "advantages": _finite_item((adv_targ * sample_weights).sum() / weight_denominator),
            "rewards": _finite_item(rewards),
            "actor_update_skipped": actor_update_skipped,
            "actor_no_grad_samples": actor_no_grad_sample,
            "actor_optimizer_steps": actor_optimizer_step,
            "kl_early_stop": float(kl_early_stop),
            "old_policy_kl_early_stop": float(old_policy_kl_early_stop),
            "bc_reference_kl_early_stop": float(reference_kl_early_stop),
            "old_policy_target_exceeded": float(
                gate['old_policy_target_exceeded']
            ),
            "bc_reference_target_exceeded": float(
                gate['bc_reference_target_exceeded']
            ),
            "old_policy_kl_stop_events": float(old_policy_kl_early_stop),
            "bc_reference_kl_stop_events": float(
                reference_kl_early_stop
            ),
            "actor_kl_stop_events": float(kl_early_stop),
            "bc_reference_kl_penalty": _finite_item(
                self.bc_reference_kl_coef * bc_reference_kl
            ),
            "joint_team_ppo": float(self.joint_team_ppo),
            "joint_team_ppo_all_roles": float(
                self.joint_team_ppo
                and self.joint_team_ppo_scope == 'all'
            ),
            "role_atomic_ppo": float(self.role_atomic_ppo),
            "role_sequential_ppo": float(self.role_sequential_ppo),
            "role_sequential_factor_mean": _finite_item(
                sequential_factor.mean()
                if sequential_factor is not None
                else torch.ones((), device=self.device)
            ),
            "approx_kl": _finite_item(approx_kl),
            "sampled_approx_kl_p95": _finite_item(approx_kl_p95),
            "sampled_approx_kl_max": _finite_item(approx_kl_max),
            "sampled_abs_log_ratio_p95": _finite_item(abs_log_ratio_p95),
            "sampled_abs_log_ratio_max": _finite_item(abs_log_ratio_max),
            "clip_fraction": _finite_item(clip_fraction),
            "actor_effective_decisions": _finite_item(policy_masks_batch.sum()),
            "actor_effective_weight_mass": _finite_item(
                sample_weights.sum()
            ),
            "trainable_decision_fraction": _finite_item(
                policy_masks_batch.sum() / active_denominator
            ),
            "actor_empty_replay_samples": 0.0,
            "actor_replay_valid": 1.0,
        }
        results.update(role_atomic_metrics)
        for role_name, exceeded in role_gate.items():
            results[f'{role_name}_target_kl_exceeded'] = float(exceeded)
        if self.role_atomic_ppo:
            results['role_atomic_effective_events'] = _finite_item(
                policy_masks_batch.sum()
            )
            results['role_atomic_effective_weight_mass'] = _finite_item(
                sample_weights.sum()
            )
        elif self.joint_team_ppo:
            scope_name = (
                'plane' if self.joint_team_ppo_scope == 'plane' else 'joint'
            )
            results[f'{scope_name}_trainable_decision_fraction'] = _finite_item(
                policy_masks_batch.mean()
            )
            results[f'{scope_name}_effective_decisions'] = _finite_item(
                policy_masks_batch.sum()
            )
            results[f'{scope_name}_effective_weight_mass'] = _finite_item(
                sample_weights.sum()
            )
        else:
            role_names = {0: 'plane', 1: 'device', 2: 'transporter'}
            for role, role_name in role_names.items():
                role_active = active_masks_batch * (agent_types_batch.unsqueeze(-1) == role).float()
                role_denominator = role_active.sum().clamp_min(1.0)
                role_policy_mask = (
                    policy_masks_batch
                    * (agent_types_batch.unsqueeze(-1) == role).float()
                )
                results[f'{role_name}_trainable_decision_fraction'] = _finite_item(
                    (role_policy_mask * role_active).sum() / role_denominator
                )
                results[f'{role_name}_effective_decisions'] = _finite_item(
                    role_policy_mask.sum()
                )
                results[f'{role_name}_effective_weight_mass'] = _finite_item(
                    (
                        sample_weights
                        * (agent_types_batch.unsqueeze(-1) == role).float()
                    ).sum()
                )
        results.update(reference_metrics)
        results.update(actor_update_metrics)
        results.update(shared_gradient_metrics)
        return results

    def update_value_net(self, sample, perform_step=True, loss_scale=1.0):
        graph_batch, rnn_states_batch, actions_batch, \
        value_preds_batch, return_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, last_op_batch, last_site_batch, rewards_batch, \
        policy_masks_batch, agent_types_batch, \
        policy_sample_weights_batch, value_sample_weights_batch = sample

        value_preds_batch = _sanitize_tensor(check(value_preds_batch).to(**self.tpdv))
        return_batch = _sanitize_tensor(check(return_batch).to(**self.tpdv))
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        agent_types_batch = check(agent_types_batch).to(self.device, dtype=torch.long)
        value_sample_weights_batch = check(value_sample_weights_batch).to(**self.tpdv)
        value_weights = self._role_sample_weights(
            active_masks_batch,
            agent_types_batch,
            base_weights=(
                value_sample_weights_batch if self.case_balanced_loss else None
            ),
        )

        values = self.policy.evaluate_values(graph_batch,
                                             rnn_states_batch,
                                             active_masks_batch,
                                             last_op_batch,
                                             last_site_batch,
                                             actions_batch,
                                             agent_types=agent_types_batch)
        values = _sanitize_tensor(values)

        role_value_losses = {}
        if self.role_valuenorm:
            weighted_losses = []
            active_coefficients = []
            for role, normalizer in self.role_value_normalizers.items():
                role_mask = (
                    agent_types_batch.unsqueeze(-1) == role
                ).to(value_weights.dtype)
                role_weights = value_weights * role_mask
                if not bool((role_weights.sum() > 0.0).item()):
                    continue
                role_loss = self.cal_value_loss(
                    values.view(-1, 1),
                    value_preds_batch.view(-1, 1),
                    return_batch.view(-1, 1),
                    active_masks_batch.view(-1, 1),
                    sample_weights=role_weights.view(-1, 1),
                    value_normalizer=normalizer,
                )
                coefficient = max(0.0, self.role_loss_coef[role])
                if coefficient <= 0.0:
                    continue
                weighted_losses.append(role_loss * coefficient)
                active_coefficients.append(coefficient)
                role_value_losses[
                    f'{self.role_names[role]}_value_loss'
                ] = _finite_item(role_loss)
            if not weighted_losses:
                value_loss = values.sum() * 0.0
            else:
                value_loss = sum(weighted_losses) / max(
                    sum(active_coefficients), 1e-8
                )
        else:
            value_loss = self.cal_value_loss(
                values.view(-1, 1),
                value_preds_batch.view(-1, 1),
                return_batch.view(-1, 1),
                active_masks_batch.view(-1, 1),
                sample_weights=value_weights.view(-1, 1),
            )

        # 🚨 移除此处的无脑 zero_grad
        # self.policy.critic_optimizer.zero_grad()

        # Normalize by actual active decision mass in the accumulation group.
        loss = value_loss * self.value_loss_coef * float(loss_scale)
        critic_update_skipped = 0.0
        critic_optimizer_step = 0.0
        if torch.isfinite(loss):
            loss.backward()
        else:
            critic_update_skipped = 1.0

        critic_grad_norm = torch.tensor(0.0)
        
        # 🚨 当满足累加步数条件时，才真正更新网络和清空梯度
        if perform_step:
            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.ac.critic_param.parameters(), self.max_grad_norm)
            else:
                critic_grad_norm = get_gard_norm(self.policy.ac.critic_param.parameters())

            if torch.isfinite(critic_grad_norm):
                self.policy.critic_optimizer.step()
                critic_optimizer_step = 1.0
            else:
                critic_update_skipped = 1.0
                critic_grad_norm = torch.tensor(0.0, device=self.device)
            self.policy.critic_optimizer.zero_grad()

        results = {
            "value_loss": _finite_item(value_loss),
            "critic_grad_norm": _finite_item(critic_grad_norm),
            "value_mean": _finite_item(values.mean()),
            "critic_update_skipped": critic_update_skipped,
            "critic_optimizer_steps": critic_optimizer_step,
            "critic_effective_decisions": _finite_item(active_masks_batch.sum()),
        }
        results.update(role_value_losses)
        return results

    @staticmethod
    def _sample_graph_count(sample):
        """Return the physical graph count of one replay microbatch."""

        graphs = sample[0]
        try:
            graph_count = len(graphs)
        except TypeError as error:
            raise TypeError(
                'Graph replay samples must expose a sized graph collection.'
            ) from error
        if graph_count <= 0:
            raise RuntimeError('PPO received an empty graph replay sample.')
        return int(graph_count)

    @classmethod
    def _accumulation_groups(cls, samples, fixed_steps, target_graphs=0):
        """Partition whole forwards into optimizer groups.

        A fixed microbatch count makes the effective optimizer batch grow when
        ``max_graphs_per_forward`` grows.  A positive graph target instead
        keeps approximately the same number of optimizer steps and average
        graph mass while retaining each physical forward intact.  Groups are
        balanced globally because a 1500-graph forward cannot be split safely
        across parameter updates.
        """

        samples = list(samples)
        if not samples:
            return []
        fixed_steps = max(1, int(fixed_steps))
        target_graphs = max(0, int(target_graphs))
        if target_graphs <= 0:
            return [
                samples[start:start + fixed_steps]
                for start in range(0, len(samples), fixed_steps)
            ]

        graph_counts = [cls._sample_graph_count(sample) for sample in samples]
        total_graphs = int(sum(graph_counts))
        group_count = int(math.floor(
            float(total_graphs) / float(target_graphs) + 0.5
        ))
        group_count = min(len(samples), max(1, group_count))

        groups = []
        start = 0
        remaining_graphs = total_graphs
        for group_index in range(group_count):
            groups_left = group_count - group_index
            if groups_left == 1:
                end = len(samples)
            else:
                max_end = len(samples) - (groups_left - 1)
                desired_graphs = float(remaining_graphs) / float(groups_left)
                candidate_graphs = 0
                best_end = start + 1
                best_error = math.inf
                for candidate_end in range(start + 1, max_end + 1):
                    candidate_graphs += graph_counts[candidate_end - 1]
                    error = abs(float(candidate_graphs) - desired_graphs)
                    if error < best_error:
                        best_error = error
                        best_end = candidate_end
                    elif candidate_graphs >= desired_graphs:
                        break
                end = best_end
            group = samples[start:end]
            groups.append(group)
            group_graphs = sum(
                graph_counts[index] for index in range(start, end)
            )
            remaining_graphs -= int(group_graphs)
            start = end

        if start != len(samples) or not all(groups):
            raise RuntimeError('Graph-target accumulation lost replay samples.')
        return groups

    def train(self, buffer, update_actor=True):
        # 🚨 在所有更新开始前，计算出固定的 Advantages。
        # 注意：整个 ppo_epoch 期间，Advantages 必须保持绝对固定！
        rollout_steps = int(getattr(buffer, 'filled_steps', buffer.episode_length))
        case_weight_info = {
            'policy_case_weight_max_error': 0.0,
            'value_case_weight_max_error': 0.0,
        }
        if self.case_balanced_loss:
            case_weight_info = buffer.build_case_balanced_weights(
                self.role_loss_coef,
                role_loss_weighting=self.role_loss_weighting,
                role_loss_min_share=self.role_loss_min_share,
                role_loss_max_share=self.role_loss_max_share,
            )
            case_weight_info.update(buffer.apply_time_tail_policy_weights(
                self.tail_policy_start_fraction,
                self.tail_policy_weight,
            ))
            case_weight_info.update(buffer.apply_cvar_case_weights(
                self.cvar_policy_fraction,
                self.cvar_policy_weight,
                self.cvar_case_metric,
            ))
        returns = np.nan_to_num(buffer.returns[:rollout_steps], nan=0.0, posinf=1e4, neginf=-1e4)
        value_preds = np.nan_to_num(buffer.value_preds[:rollout_steps], nan=0.0, posinf=1e4, neginf=-1e4)
        raw_advantages, value_baseline = self._compute_rollout_advantages(
            returns,
            value_preds,
            agent_types=buffer.agent_types[:rollout_steps],
        )
        critic_raw_advantages = raw_advantages.copy()
        actor_case_offsets = np.asarray(
            getattr(
                buffer,
                'actor_case_baseline_offsets',
                np.zeros((buffer.n_rollout_threads,), dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        if actor_case_offsets.shape != (buffer.n_rollout_threads,):
            raise RuntimeError('Actor case-baseline offset shape is invalid.')
        if not np.isfinite(actor_case_offsets).all():
            raise RuntimeError('Actor case-baseline offsets must be finite.')
        if np.any(actor_case_offsets != 0.0):
            raw_advantages = raw_advantages + actor_case_offsets[
                None, :, None, None
            ]
        advantages = raw_advantages.copy()
        policy_valid = buffer.policy_masks[:rollout_steps, ..., 0] > 0.0
        if not policy_valid.any():
            policy_valid = buffer.active_masks[:rollout_steps, ..., 0] > 0.0
        advantages = self._normalize_rollout_advantages(
            advantages,
            buffer,
            rollout_steps,
        )

        if self._use_popart or self._use_valuenorm:
            normalizer_weights = (
                buffer.value_sample_weights[:rollout_steps]
                if self.case_balanced_loss
                else buffer.active_masks[:rollout_steps]
            )
            if self.role_valuenorm:
                rollout_agent_types = buffer.agent_types[:rollout_steps]
                for role, normalizer in self.role_value_normalizers.items():
                    valid = (
                        (normalizer_weights[..., 0] > 0.0)
                        & (rollout_agent_types == role)
                    )
                    if valid.any():
                        normalizer.update(
                            returns[..., 0][valid].reshape(-1, 1),
                            weights=normalizer_weights[..., 0][valid].reshape(-1, 1),
                        )
            else:
                valid = normalizer_weights[..., 0] > 0.0
                if valid.any():
                    self.value_normalizer.update(
                        returns[..., 0][valid].reshape(-1, 1),
                        weights=normalizer_weights[..., 0][valid].reshape(-1, 1),
                    )
        
        diagnostic_info = {
            'return_mean_raw_scale': 0.0,
            'value_pred_mean_normalized': 0.0,
            'value_pred_mean_raw_scale': 0.0,
            'advantage_mean_raw_scale': 0.0,
            'advantage_std_raw_scale': 0.0,
            'critic_calibration_ratio': 0.0,
            'actor_case_baseline_offset_mean': float(
                actor_case_offsets.mean()
            ),
            'actor_case_baseline_offset_abs_mean': float(
                np.abs(actor_case_offsets).mean()
            ),
            'value_normalizer_mean': 0.0,
            'value_normalizer_std': 0.0,
        }
        if policy_valid.any():
            advantage_mean = float(
                critic_raw_advantages[..., 0][policy_valid].mean()
            )
            advantage_std = float(
                critic_raw_advantages[..., 0][policy_valid].std()
            )
            diagnostic_info.update({
                'return_mean_raw_scale': float(returns[..., 0][policy_valid].mean()),
                'value_pred_mean_normalized': float(value_preds[..., 0][policy_valid].mean()),
                'value_pred_mean_raw_scale': float(value_baseline[..., 0][policy_valid].mean()),
                'advantage_mean_raw_scale': advantage_mean,
                'advantage_std_raw_scale': advantage_std,
                'critic_calibration_ratio': (
                    abs(advantage_mean) / max(advantage_std, 1e-8)
                ),
            })
        if self.value_normalizer is not None and not self.role_valuenorm:
            normalizer_mean, normalizer_var = self.value_normalizer.running_mean_var()
            diagnostic_info.update({
                'value_normalizer_mean': _finite_item(normalizer_mean),
                'value_normalizer_std': _finite_item(torch.sqrt(normalizer_var)),
            })
        elif self.role_valuenorm:
            for role, normalizer in self.role_value_normalizers.items():
                normalizer_mean, normalizer_var = normalizer.running_mean_var()
                role_name = self.role_names[role]
                diagnostic_info.update({
                    f'{role_name}_value_normalizer_mean': _finite_item(
                        normalizer_mean
                    ),
                    f'{role_name}_value_normalizer_std': _finite_item(
                        torch.sqrt(normalizer_var)
                    ),
                })

        train_info = defaultdict(float)
        train_info.update(case_weight_info)
        train_info.update(diagnostic_info)
        train_info['actor_grad_accumulation_steps'] = float(
            self.actor_grad_accumulation_steps
        )
        train_info['critic_grad_accumulation_steps'] = float(self.grad_accumulation_steps)
        train_info['actor_grad_accumulation_target_graphs'] = float(
            self.actor_grad_accumulation_target_graphs
        )
        train_info['critic_grad_accumulation_target_graphs'] = float(
            self.grad_accumulation_target_graphs
        )
        
        actor_sample_count = 0
        actor_data_sample_count = 0
        actor_accumulation_groups = 0
        actor_planned_optimizer_steps = 0
        actor_group_graph_counts = []
        post_update_probe_count = 0
        shared_gradient_probe_count = 0
        critic_sample_count = 0
        critic_group_graph_counts = []
        graph_batch_executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix='hkbz-pyg-prefetch',
            )
            if self.safe_graph_batch_pipeline else None
        )

        # ==========================================
        # Phase 1: 集中更新 Actor (策略网络)
        # ==========================================
        if update_actor:
            # Keep the stochastic GNN encoder in eval mode so PPO replay
            # matches rollout collection. The GRU/backend must remain in train
            # mode because cuDNN only supports RNN backward in training mode;
            # these modules do not contain stochastic dropout.
            self.policy.ac.eval()
            self.policy.ac.actor_param_without_gnn.train()
            self.policy.actor_optimizer.zero_grad() # 确保起跑前梯度干净
            trainable_actor_roles = self._trainable_actor_roles()

            for epoch in range(self.ppo_epoch):
                # 每次 epoch 重新打乱数据
                data_generator = buffer.graph_recurrent_generator(advantages, self.mini_batch_size)
                data_samples = list(data_generator)
                actor_mass_index = 13 if self.case_balanced_loss else 11
                actor_groups = self._accumulation_groups(
                    data_samples,
                    self.actor_grad_accumulation_steps,
                    self.actor_grad_accumulation_target_graphs,
                )
                if not actor_groups:
                    continue
                group_trainable_masses = [
                    [
                        self._role_filtered_sample_mass(
                            sample,
                            actor_mass_index,
                            trainable_actor_roles,
                        )
                        for sample in group
                    ]
                    for group in actor_groups
                ]

                stop_actor_epoch = False
                planned_groups_this_epoch = sum(
                    any(mass > 0.0 for mass in trainable_masses)
                    for trainable_masses in group_trainable_masses
                )
                actor_planned_optimizer_steps += planned_groups_this_epoch
                for accumulation_group_idx, (group, trainable_masses) in enumerate(
                    zip(actor_groups, group_trainable_masses)
                ):
                    actor_group_graph_counts.append(sum(
                        self._sample_graph_count(sample) for sample in group
                    ))
                    if not any(mass > 0.0 for mass in trainable_masses):
                        train_info['actor_zero_trainable_mass_groups'] += 1.0
                        train_info['actor_zero_trainable_mass_samples'] += float(
                            len(group)
                        )
                        continue
                    actor_accumulation_groups += 1
                    actor_data_sample_count += len(group)
                    masses = [
                        self._sample_mass(sample, actor_mass_index)
                        for sample in group
                    ]
                    prepared_probe_sample = None
                    prepared_probe_mass = -1.0
                    valid_group_mass = 0.0
                    group_optimizer_steps = 0.0
                    prepared_group = list(self._iter_prepared_graph_samples(
                        group, graph_batch_executor
                    ))
                    if self.role_sequential_ppo:
                        role_orders = tuple(itertools.permutations((0, 1, 2)))
                        role_order = role_orders[
                            (epoch * len(actor_groups)
                             + accumulation_group_idx) % len(role_orders)
                        ]
                        sequential_results = self._sequential_actor_group_update(
                            prepared_group,
                            masses,
                            order=role_order,
                            diagnose_shared_gradients=bool(
                                self.shared_gradient_diagnostics
                                and shared_gradient_probe_count == 0
                            ),
                        )
                        for key, value in sequential_results.items():
                            train_info[key] += value
                        train_info['role_sequential_order_code_sum'] += float(
                            100 * role_order[0]
                            + 10 * role_order[1]
                            + role_order[2]
                        )
                        group_optimizer_steps = sequential_results.get(
                            'actor_optimizer_steps', 0.0
                        )
                        diagnostic_samples = int(sequential_results.get(
                            'shared_gradient_diagnostic_samples', 0.0
                        ))
                        shared_gradient_probe_count += diagnostic_samples
                        actor_sample_count += len(prepared_group) * (
                            4 if self._trainable_shared_actor_parameters() else 3
                        )
                        valid_group_mass = sum(
                            sample_mass
                            for sample_mass, trainable_mass in zip(
                                masses, trainable_masses
                            )
                            if trainable_mass > 0.0
                        )
                        if prepared_group:
                            probe_index = max(
                                range(len(prepared_group)),
                                key=lambda index: masses[index],
                            )
                            prepared_probe_sample = prepared_group[probe_index]
                            prepared_probe_mass = masses[probe_index]
                        # The sequential helper already consumed and stepped
                        # every prepared sample.  Leave the ordinary replay
                        # loop empty while retaining the common post-step KL
                        # probe and macro rollback below.
                        prepared_group = []
                    for group_idx, (sample, sample_mass) in enumerate(
                        zip(prepared_group, masses)
                    ):
                        diagnose_shared = bool(
                            self.shared_gradient_diagnostics
                            and shared_gradient_probe_count == 0
                        )
                        policy_results = self.update_policy_net(
                            sample,
                            update_actor=True,
                            perform_step=False,
                            # Normalize once at the optimizer boundary using
                            # only replay-valid samples.  A mask-empty sample
                            # can therefore be skipped without diluting the
                            # accumulated gradient or losing the final step.
                            loss_scale=sample_mass,
                            diagnose_shared_gradients=diagnose_shared,
                        )

                        for k, v in policy_results.items():
                            train_info[k] += v
                        for role_name in self.role_names.values():
                            metric = f'{role_name}_approx_kl'
                            maximum = f'{metric}_max_seen'
                            train_info[maximum] = max(
                                train_info[maximum],
                                float(policy_results.get(metric, 0.0)),
                            )
                        diagnostic_samples = int(policy_results.get(
                            'shared_gradient_diagnostic_samples', 0.0
                        ))
                        shared_gradient_probe_count += diagnostic_samples
                        if policy_results.get('actor_replay_valid', 0.0) > 0.0:
                            valid_group_mass += sample_mass
                            if sample_mass > prepared_probe_mass:
                                prepared_probe_sample = sample
                                prepared_probe_mass = sample_mass
                        actor_sample_count += 1
                        if policy_results.get('kl_early_stop', 0.0) > 0.0:
                            train_info['actor_kl_stop_ppo_epoch'] = float(epoch)
                            train_info['actor_kl_stop_group_index'] = float(
                                accumulation_group_idx
                            )
                            train_info['actor_kl_stop_sample_index'] = float(
                                group_idx
                            )
                            train_info['actor_kl_stop_role_code'] = float(sum(
                                (role + 1) * int(policy_results.get(
                                    f'{role_name}_target_kl_exceeded', 0.0
                                ) > 0.0)
                                for role, role_name in self.role_names.items()
                            ))
                            stop_actor_epoch = True
                            break
                    if not stop_actor_epoch and not self.role_sequential_ppo:
                        step_results = self._perform_actor_optimizer_step(
                            grad_divisor=max(valid_group_mass, 1e-8)
                        )
                        for key, value in step_results.items():
                            train_info[key] += value
                        group_optimizer_steps += step_results.get(
                            'actor_optimizer_steps', 0.0
                        )
                    if group_optimizer_steps > 0.0:
                        if prepared_probe_sample is None:
                            raise RuntimeError(
                                'Actor stepped without a replay-valid KL probe sample.'
                            )
                        retry_scales = (
                            self.actor_kl_backtrack_scales
                            if self.actor_kl_backtrack else ()
                        )
                        attempted_scale = 1.0
                        retry_index = 0
                        accepted_step = False
                        last_post_update_results = None
                        last_role_post_gate = None
                        while True:
                            post_update_results = (
                                self.measure_post_update_policy_shift(
                                    prepared_probe_sample
                                )
                            )
                            last_post_update_results = post_update_results
                            for k, v in post_update_results.items():
                                train_info[k] += v
                            for role_name in self.role_names.values():
                                metric = f'post_update_{role_name}_approx_kl'
                                maximum = f'{metric}_max_seen'
                                train_info[maximum] = max(
                                    train_info[maximum],
                                    float(post_update_results.get(metric, 0.0)),
                                )
                            post_update_probe_count += 1
                            post_gate = self._kl_gate_decision(
                                post_update_results[
                                    'post_update_probe_approx_kl'
                                ],
                                post_update_results[
                                    'post_update_bc_reference_approx_kl'
                                ],
                            )
                            role_post_gate = self._role_kl_gate(
                                post_update_results,
                                prefix='post_update_',
                            )
                            last_role_post_gate = role_post_gate
                            role_kl_exceeded = bool(
                                self.role_atomic_ppo
                                and any(role_post_gate.values())
                            )
                            for role_name, exceeded in role_post_gate.items():
                                train_info[
                                    f'post_update_{role_name}_target_kl_exceeded'
                                ] += float(exceeded)
                            if role_kl_exceeded:
                                post_gate['old_policy_target_exceeded'] = True
                                post_gate['old_policy_kl_early_stop'] = True
                                post_gate['kl_early_stop'] = True
                            train_info[
                                'post_update_target_kl_exceeded'
                            ] += float(post_gate['old_policy_target_exceeded'])
                            train_info[
                                'post_update_bc_reference_target_kl_exceeded'
                            ] += float(post_gate[
                                'bc_reference_target_exceeded'
                            ])
                            if not post_gate['kl_early_stop']:
                                accepted_step = True
                                break

                            if not self.actor_kl_backtrack:
                                break
                            train_info['actor_backtrack_rejected_probes'] += 1.0
                            if retry_index >= len(retry_scales):
                                break
                            attempted_scale = retry_scales[retry_index]
                            retry_index += 1
                            train_info['actor_backtrack_retry_attempts'] += 1.0
                            retry_metrics = self.retry_last_actor_step(
                                attempted_scale
                            )
                            # The logical optimizer group is counted once.
                            # Replace its displacement diagnostic with that of
                            # the accepted/rejected scaled retry.
                            for key, value in retry_metrics.items():
                                if key in {
                                    'actor_optimizer_steps',
                                    'actor_backtrack_step_scale',
                                }:
                                    continue
                                train_info[key] += (
                                    float(value)
                                    - float(step_results.get(key, 0.0))
                                )
                                step_results[key] = float(value)

                        if accepted_step:
                            train_info[
                                'accepted_post_update_bc_reference_kl_sum'
                            ] += float(last_post_update_results[
                                'post_update_bc_reference_approx_kl'
                            ])
                            train_info[
                                'accepted_post_update_bc_reference_kl_count'
                            ] += 1.0
                            if attempted_scale < 1.0:
                                train_info['actor_backtrack_accepted_retries'] += 1.0
                                train_info['actor_backtrack_accepted_scale_sum'] += float(
                                    attempted_scale
                                )
                                current_min = train_info.get(
                                    'actor_backtrack_accepted_scale_min', 0.0
                                )
                                train_info['actor_backtrack_accepted_scale_min'] = (
                                    float(attempted_scale)
                                    if current_min <= 0.0 else
                                    min(float(current_min), float(attempted_scale))
                                )
                            if (
                                self.role_atomic_ppo
                                or self.bc_reference_hard_gate
                                or self.actor_kl_backtrack
                            ):
                                self.accept_last_actor_step()
                        else:
                            train_info['actor_kl_stop_ppo_epoch'] = float(epoch)
                            train_info['actor_kl_stop_group_index'] = float(
                                accumulation_group_idx
                            )
                            train_info['actor_kl_stop_sample_index'] = -1.0
                            train_info['actor_kl_stop_role_code'] = float(sum(
                                (role + 1) * int(last_role_post_gate[role_name])
                                for role, role_name in self.role_names.items()
                            ))
                            if last_post_update_results is not None:
                                old_exceeded = bool(
                                    self.target_kl > 0.0
                                    and float(last_post_update_results[
                                        'post_update_probe_approx_kl'
                                    ]) > self.target_kl
                                ) or bool(any(last_role_post_gate.values()))
                                reference_exceeded = bool(
                                    self.bc_reference_hard_gate
                                    and self.bc_reference_target_kl > 0.0
                                    and float(last_post_update_results[
                                        'post_update_bc_reference_approx_kl'
                                    ]) > self.bc_reference_target_kl
                                )
                                train_info['old_policy_kl_stop_events'] += float(
                                    old_exceeded
                                )
                                train_info['bc_reference_kl_stop_events'] += float(
                                    reference_exceeded
                                )
                            if (
                                self.role_atomic_ppo
                                or self.bc_reference_hard_gate
                                or self.actor_kl_backtrack
                            ):
                                self.rollback_last_actor_step()
                                train_info['actor_optimizer_steps'] = max(
                                    0.0,
                                    train_info.get('actor_optimizer_steps', 0.0)
                                    - 1.0,
                                )
                                for key, value in step_results.items():
                                    if key in {
                                        'actor_optimizer_steps',
                                        'actor_update_skipped',
                                    }:
                                        continue
                                    if key.startswith('actor_'):
                                        train_info[key] -= float(value)
                                group_optimizer_steps = 0.0
                                train_info['actor_optimizer_rollbacks'] += 1.0
                            train_info['actor_kl_stop_events'] += 1.0
                            if self.actor_kl_backtrack:
                                train_info['actor_backtrack_failed_groups'] += 1.0
                                print(
                                    '[PPO] All KL backtracking scales rejected '
                                    f'for group={accumulation_group_idx}; '
                                    'restored the group and continuing.',
                                    flush=True,
                                )
                            else:
                                stop_actor_epoch = True
                                print(
                                    '[PPO] Post-update KL gate exceeded: '
                                    f"old={last_post_update_results['post_update_probe_approx_kl']:.6g}/"
                                    f'{self.target_kl:.6g}, '
                                    f"bc_ref={last_post_update_results['post_update_bc_reference_approx_kl']:.6g}/"
                                    f'{self.bc_reference_target_kl:.6g} '
                                    f'(hard_gate={self.bc_reference_hard_gate}); '
                                    f'role_gate={last_role_post_gate}; '
                                    'stopping Actor updates for this shard.'
                                )
                    if stop_actor_epoch:
                        remaining_epochs = self.ppo_epoch - epoch - 1
                        actor_planned_optimizer_steps += (
                            remaining_epochs * planned_groups_this_epoch
                        )
                        break
                if stop_actor_epoch:
                    break

        # ==========================================
        # Phase 2: 集中更新 Critic (价值网络)
        # ==========================================
        # Keep the stochastic GNN deterministic. Critic gradients traverse the
        # role GRUs, so they must also stay in train mode for cuDNN backward.
        self.policy.ac.eval()
        self.policy.ac.actor_param_without_gnn.train()
        self.policy.ac.critic_param.train()
        self.policy.critic_optimizer.zero_grad() # 确保起跑前梯度干净

        for epoch in range(self.ppo_epoch):
            # 同样每次 epoch 重新打乱数据（用相同的 generator 保证切分维度合法）
            data_generator = buffer.graph_recurrent_generator(advantages, self.mini_batch_size)
            data_samples = list(data_generator)
            critic_groups = self._accumulation_groups(
                data_samples,
                self.grad_accumulation_steps,
                self.grad_accumulation_target_graphs,
            )
            if not critic_groups:
                continue

            for group in critic_groups:
                critic_group_graph_counts.append(sum(
                    self._sample_graph_count(sample) for sample in group
                ))
                critic_mass_index = 14 if self.case_balanced_loss else 5
                masses = [self._sample_mass(sample, critic_mass_index) for sample in group]
                group_mass = max(sum(masses), 1e-8)
                prepared_group = self._iter_prepared_graph_samples(
                    group, graph_batch_executor
                )
                for group_idx, (sample, sample_mass) in enumerate(
                    zip(prepared_group, masses)
                ):
                    perform_step = group_idx == len(group) - 1
                    value_results = self.update_value_net(
                        sample,
                        perform_step=perform_step,
                        loss_scale=sample_mass / group_mass,
                    )

                    for k, v in value_results.items():
                        train_info[k] += v
                    critic_sample_count += 1

        # ==========================================
        # 均摊统计指标
        # ==========================================
        # Sample metrics, optimizer-step metrics, and post-update probes have
        # different denominators. Mixing them made the old gradient norms look
        # artificially smaller by the number of replay minibatches.
        actor_sample_keys = [
            "policy_loss",
            "dist_entropy",
            "advantages",
            "rewards",
            "actor_update_skipped",
            "old_policy_kl_early_stop",
            "bc_reference_kl_early_stop",
            "old_policy_target_exceeded",
            "bc_reference_target_exceeded",
            "bc_reference_kl_penalty",
            "bc_reference_approx_kl",
            "bc_reference_approx_kl_p95",
            "bc_reference_abs_log_ratio_p95",
            "approx_kl",
            "sampled_approx_kl_p95",
            "sampled_approx_kl_max",
            "sampled_abs_log_ratio_p95",
            "sampled_abs_log_ratio_max",
            "clip_fraction",
            "trainable_decision_fraction",
            "plane_trainable_decision_fraction",
            "device_trainable_decision_fraction",
            "transporter_trainable_decision_fraction",
            "plane_approx_kl",
            "device_approx_kl",
            "transporter_approx_kl",
            "plane_target_kl_exceeded",
            "device_target_kl_exceeded",
            "transporter_target_kl_exceeded",
        ]
        actor_step_keys = [
            "actor_grad_norm",
            "actor_shared_encoder_update_l2",
            "actor_grad_norm_clipped",
            "actor_grad_clip_applied",
            "actor_shared_encoder_update_relative",
            "actor_plane_actor_update_l2",
            "actor_plane_actor_update_relative",
            "actor_device_actor_update_l2",
            "actor_device_actor_update_relative",
            "actor_transporter_actor_update_l2",
            "actor_transporter_actor_update_relative",
            "actor_param_update_l2",
            "actor_param_update_relative",
            "actor_shared_encoder_update_fraction",
            "actor_plane_actor_update_fraction",
            "actor_device_actor_update_fraction",
            "actor_transporter_actor_update_fraction",
        ]
        post_update_keys = [
            "post_update_probe_approx_kl",
            "post_update_probe_approx_kl_p95",
            "post_update_probe_approx_kl_max",
            "post_update_probe_abs_log_ratio_p95",
            "post_update_probe_abs_log_ratio_max",
            "post_update_probe_clip_fraction",
            "post_update_probe_entropy",
            "post_update_probe_decisions",
            "post_update_bc_reference_approx_kl",
            "post_update_bc_reference_approx_kl_p95",
            "post_update_bc_reference_abs_log_ratio_p95",
            "post_update_plane_approx_kl",
            "post_update_device_approx_kl",
            "post_update_transporter_approx_kl",
            "post_update_plane_target_kl_exceeded",
            "post_update_device_target_kl_exceeded",
            "post_update_transporter_target_kl_exceeded",
        ]
        shared_gradient_keys = [
            'shared_plane_grad_norm',
            'shared_device_grad_norm',
            'shared_transporter_grad_norm',
            'shared_plane_grad_ema',
            'shared_device_grad_ema',
            'shared_transporter_grad_ema',
            'shared_plane_grad_scale',
            'shared_device_grad_scale',
            'shared_transporter_grad_scale',
            'shared_plane_balanced_grad_norm',
            'shared_device_balanced_grad_norm',
            'shared_transporter_balanced_grad_norm',
            'shared_plane_device_grad_cosine',
            'shared_plane_transporter_grad_cosine',
            'shared_device_transporter_grad_cosine',
            'shared_gradient_conflict_rate',
            'shared_gradient_severe_conflict_rate',
            'shared_gradient_method_code',
            'shared_grad_balance_target_norm',
            'shared_pcgrad_applied',
            'shared_pcgrad_projection_fraction',
            'shared_pcgrad_projection_rate',
            'shared_combined_grad_norm',
            'shared_combined_to_raw_norm_ratio',
            'shared_combined_plane_cosine',
            'shared_combined_device_cosine',
            'shared_combined_transporter_cosine',
            'shared_cagrad_plane_weight',
            'shared_cagrad_device_weight',
            'shared_cagrad_transporter_weight',
        ]
        critic_sample_keys = [
            "value_loss",
            "value_mean",
            "critic_update_skipped",
        ]
        critic_step_keys = ["critic_grad_norm"]

        if actor_sample_count > 0:
            for key in actor_sample_keys:
                if key in train_info:
                    train_info[key] /= actor_sample_count

        actor_optimizer_steps = train_info.get('actor_optimizer_steps', 0.0)
        if actor_optimizer_steps > 0.0:
            for key in actor_step_keys:
                if key in train_info:
                    train_info[key] /= actor_optimizer_steps

        if post_update_probe_count > 0:
            for key in post_update_keys:
                if key in train_info:
                    train_info[key] /= post_update_probe_count

        shared_gradient_samples = float(train_info.get(
            'shared_gradient_diagnostic_samples', 0.0
        ))
        if shared_gradient_samples > 0.0:
            for key in shared_gradient_keys:
                if key in train_info:
                    train_info[key] /= shared_gradient_samples

        if critic_sample_count > 0:
            for key in critic_sample_keys:
                if key in train_info:
                    train_info[key] /= critic_sample_count

        critic_optimizer_steps = train_info.get('critic_optimizer_steps', 0.0)
        if critic_optimizer_steps > 0.0:
            for key in critic_step_keys:
                if key in train_info:
                    train_info[key] /= critic_optimizer_steps

        train_info['actor_data_sample_count'] = float(actor_data_sample_count)
        train_info['actor_empty_replay_fraction'] = float(
            train_info.get('actor_empty_replay_samples', 0.0)
        ) / max(float(actor_sample_count), 1.0)
        train_info['actor_accumulation_groups'] = float(actor_accumulation_groups)
        actual_actor_steps = float(
            train_info.get('actor_optimizer_steps', 0.0)
        )
        train_info['actor_planned_optimizer_steps'] = float(
            actor_planned_optimizer_steps
        )
        train_info['actor_step_completion_rate'] = (
            actual_actor_steps / actor_planned_optimizer_steps
            if actor_planned_optimizer_steps > 0 else 1.0
        )
        train_info['actor_zero_update'] = float(
            actor_planned_optimizer_steps > 0 and actual_actor_steps <= 0.0
        )
        train_info['actor_incomplete_update'] = float(
            actor_planned_optimizer_steps > 0
            and actual_actor_steps < actor_planned_optimizer_steps
        )
        old_stop_events = float(
            train_info.get('old_policy_kl_stop_events', 0.0)
        )
        reference_stop_events = float(
            train_info.get('bc_reference_kl_stop_events', 0.0)
        )
        train_info['actor_kl_stop_reason_code'] = float(
            int(old_stop_events > 0.0)
            + 2 * int(reference_stop_events > 0.0)
        )
        train_info['kl_early_stop'] = float(
            train_info.get('actor_kl_stop_events', 0.0) > 0.0
        )
        train_info['bc_reference_hard_gate_enabled'] = float(
            self.bc_reference_hard_gate
        )
        train_info['actor_kl_backtrack_enabled'] = float(
            self.actor_kl_backtrack
        )
        failed_backtrack_groups = float(
            train_info.get('actor_backtrack_failed_groups', 0.0)
        )
        train_info['actor_backtrack_failed_group_fraction'] = (
            failed_backtrack_groups / max(float(actor_accumulation_groups), 1.0)
        )
        accepted_retry_count = float(
            train_info.get('actor_backtrack_accepted_retries', 0.0)
        )
        train_info['actor_backtrack_accepted_scale_mean'] = (
            float(train_info.get('actor_backtrack_accepted_scale_sum', 0.0))
            / max(accepted_retry_count, 1.0)
        )
        previous_bc_coef = float(self.bc_reference_kl_coef)
        accepted_bc_count = float(train_info.get(
            'accepted_post_update_bc_reference_kl_count', 0.0
        ))
        accepted_bc_kl = (
            float(train_info.get(
                'accepted_post_update_bc_reference_kl_sum', 0.0
            )) / accepted_bc_count
            if accepted_bc_count > 0.0 else 0.0
        )
        self._adapt_bc_reference_coefficient(
            accepted_bc_kl, accepted_bc_count
        )
        train_info['adaptive_bc_reference_enabled'] = float(
            self.adaptive_bc_reference_kl
        )
        train_info['accepted_post_update_bc_reference_kl'] = float(
            accepted_bc_kl
        )
        train_info['bc_reference_kl_coef_before'] = previous_bc_coef
        train_info['bc_reference_kl_coef_after'] = float(
            self.bc_reference_kl_coef
        )
        train_info['post_update_probe_count'] = float(post_update_probe_count)
        train_info['safe_graph_batch_pipeline'] = float(
            self.safe_graph_batch_pipeline
        )
        for prefix, graph_counts in (
            ('actor', actor_group_graph_counts),
            ('critic', critic_group_graph_counts),
        ):
            if graph_counts:
                train_info[f'{prefix}_accumulation_graphs_mean'] = float(
                    np.mean(graph_counts)
                )
                train_info[f'{prefix}_accumulation_graphs_min'] = float(
                    min(graph_counts)
                )
                train_info[f'{prefix}_accumulation_graphs_max'] = float(
                    max(graph_counts)
                )
            else:
                train_info[f'{prefix}_accumulation_graphs_mean'] = 0.0
                train_info[f'{prefix}_accumulation_graphs_min'] = 0.0
                train_info[f'{prefix}_accumulation_graphs_max'] = 0.0

        if graph_batch_executor is not None:
            graph_batch_executor.shutdown(wait=True, cancel_futures=True)

        return train_info
    
    def prep_training(self):
        self.policy.ac.train()
        if self.policy.has_bc_reference():
            self.policy.bc_reference_ac.eval()

    def prep_rollout(self):
        self.policy.ac.eval()
        if self.policy.has_bc_reference():
            self.policy.bc_reference_ac.eval()
