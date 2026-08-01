import numpy as np
import math
import torch
import torch.nn as nn
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
        self.target_kl = max(0.0, float(getattr(args, 'target_kl', 0.0)))
        self.bc_reference_kl_coef = max(
            0.0, float(getattr(args, 'bc_reference_kl_coef', 0.0))
        )
        self.bc_reference_target_kl = max(
            0.0, float(getattr(args, 'bc_reference_target_kl', 0.0))
        )
        self.bc_reference_hard_gate = bool(
            getattr(args, 'bc_reference_hard_gate', False)
        )
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
        
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(input_shape=1, device=self.device)
        else:
            self.value_normalizer = None

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

    def _compute_rollout_advantages(self, returns, value_preds):
        """Subtract a critic baseline expressed on the same scale as returns."""
        returns = np.nan_to_num(
            np.asarray(returns), nan=0.0, posinf=1e4, neginf=-1e4
        )
        value_preds = np.nan_to_num(
            np.asarray(value_preds), nan=0.0, posinf=1e4, neginf=-1e4
        )
        if self._use_popart or self._use_valuenorm:
            value_baseline = self.value_normalizer.denormalize(value_preds)
        else:
            value_baseline = value_preds
        value_baseline = np.nan_to_num(
            value_baseline, nan=0.0, posinf=1e4, neginf=-1e4
        )
        return np.nan_to_num(returns - value_baseline, nan=0.0, posinf=1e4, neginf=-1e4), value_baseline

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch,
                       sample_weights=None):
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
            norm_return_batch = self.value_normalizer.normalize(return_batch)
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
        sample_weights = self._role_sample_weights(
            policy_masks_batch,
            agent_types_batch,
            base_weights=(
                policy_sample_weights_batch if self.case_balanced_loss else None
            ),
        )
        weight_denominator = sample_weights.sum().clamp_min(1e-8)
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
        )
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
        return results

    def update_policy_net(
        self,
        sample,
        update_actor=True,
        perform_step=True,
        loss_scale=1.0,
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
        if self.joint_team_ppo:
            plane_mask = policy_masks_batch * (
                agent_types_batch.unsqueeze(-1) == 0
            ).to(policy_masks_batch.dtype)
            plane_count = plane_mask.sum(dim=1).clamp_min(1.0)
            joint_valid = (plane_mask.sum(dim=1) > 0.0).to(
                policy_masks_batch.dtype
            )
            action_log_probs = (
                action_log_probs * plane_mask.squeeze(-1)
            ).sum(dim=1)
            old_action_log_probs_batch = (
                old_action_log_probs_batch.squeeze(-1)
                * plane_mask.squeeze(-1)
            ).sum(dim=1, keepdim=True)
            adv_targ = (
                adv_targ * plane_mask
            ).sum(dim=1) / plane_count
            if self.case_balanced_loss:
                sample_weights = (
                    policy_sample_weights_batch * plane_mask
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
                    * plane_mask.squeeze(-1)
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
        old_policy_kl_early_stop = gate['old_policy_kl_early_stop']
        reference_kl_early_stop = gate['bc_reference_kl_early_stop']
        kl_early_stop = gate['kl_early_stop']

        actor_update_skipped = 0.0
        actor_optimizer_step = 0.0
        actor_update_metrics = {}
        if update_actor and not kl_early_stop:
            # Accumulation is normalized by the actual decision mass of the
            # current group. A short final group therefore keeps full weight.
            loss = (
                policy_loss - dist_entropy * self.entropy_coef
            ) * float(loss_scale)
            if torch.isfinite(loss):
                loss.backward()
            else:
                actor_update_skipped = 1.0
        elif kl_early_stop:
            actor_update_skipped = 1.0
            # Discard any partial accumulation from this group.
            self.policy.actor_optimizer.zero_grad()

        actor_grad_norm = torch.tensor(0.0)
        
        actor_grad_norm_clipped = torch.tensor(0.0)
        actor_grad_clip_applied = 0.0
        if perform_step and update_actor and not kl_early_stop:
            actor_snapshot = self._snapshot_actor_parameters()
            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.ac.actor_param.parameters(), self.max_grad_norm)
                actor_grad_norm_clipped = torch.clamp(
                    actor_grad_norm.detach(), max=self.max_grad_norm
                )
                actor_grad_clip_applied = float(
                    actor_grad_norm.detach().item() > self.max_grad_norm
                )
            else:
                actor_grad_norm = get_gard_norm(self.policy.ac.actor_param.parameters())
                actor_grad_norm_clipped = actor_grad_norm.detach()

            if torch.isfinite(actor_grad_norm):
                self.policy.actor_optimizer.step()
                actor_optimizer_step = 1.0
                actor_update_metrics = self._actor_parameter_update_metrics(
                    actor_snapshot
                )
            else:
                actor_update_skipped = 1.0
                actor_grad_norm = torch.tensor(0.0, device=self.device)
                actor_grad_norm_clipped = torch.tensor(0.0, device=self.device)
                actor_grad_clip_applied = 0.0
            self.policy.actor_optimizer.zero_grad()

        results = {
            "policy_loss": _finite_item(policy_loss),
            "actor_grad_norm": _finite_item(actor_grad_norm),
            "dist_entropy": _finite_item(dist_entropy),
            "actor_grad_norm_clipped": _finite_item(actor_grad_norm_clipped),
            "actor_grad_clip_applied": actor_grad_clip_applied,
            "advantages": _finite_item((adv_targ * sample_weights).sum() / weight_denominator),
            "rewards": _finite_item(rewards),
            "actor_update_skipped": actor_update_skipped,
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
        }
        if self.joint_team_ppo:
            results['plane_trainable_decision_fraction'] = _finite_item(
                policy_masks_batch.mean()
            )
            results['plane_effective_decisions'] = _finite_item(
                policy_masks_batch.sum()
            )
            results['plane_effective_weight_mass'] = _finite_item(
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

        return {
            "value_loss": _finite_item(value_loss),
            "critic_grad_norm": _finite_item(critic_grad_norm),
            "value_mean": _finite_item(values.mean()),
            "critic_update_skipped": critic_update_skipped,
            "critic_optimizer_steps": critic_optimizer_step,
            "critic_effective_decisions": _finite_item(active_masks_batch.sum()),
        }

    def train(self, buffer, update_actor=True):
        # 🚨 在所有更新开始前，计算出固定的 Advantages。
        # 注意：整个 ppo_epoch 期间，Advantages 必须保持绝对固定！
        rollout_steps = int(getattr(buffer, 'filled_steps', buffer.episode_length))
        case_weight_info = {
            'policy_case_weight_max_error': 0.0,
            'value_case_weight_max_error': 0.0,
        }
        if self.case_balanced_loss:
            case_weight_info = buffer.build_case_balanced_weights(self.role_loss_coef)
        returns = np.nan_to_num(buffer.returns[:rollout_steps], nan=0.0, posinf=1e4, neginf=-1e4)
        value_preds = np.nan_to_num(buffer.value_preds[:rollout_steps], nan=0.0, posinf=1e4, neginf=-1e4)
        raw_advantages, value_baseline = self._compute_rollout_advantages(
            returns,
            value_preds,
        )
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
            'value_normalizer_mean': 0.0,
            'value_normalizer_std': 0.0,
        }
        if policy_valid.any():
            diagnostic_info.update({
                'return_mean_raw_scale': float(returns[..., 0][policy_valid].mean()),
                'value_pred_mean_normalized': float(value_preds[..., 0][policy_valid].mean()),
                'value_pred_mean_raw_scale': float(value_baseline[..., 0][policy_valid].mean()),
                'advantage_mean_raw_scale': float(raw_advantages[..., 0][policy_valid].mean()),
                'advantage_std_raw_scale': float(raw_advantages[..., 0][policy_valid].std()),
            })
        if self.value_normalizer is not None:
            normalizer_mean, normalizer_var = self.value_normalizer.running_mean_var()
            diagnostic_info.update({
                'value_normalizer_mean': _finite_item(normalizer_mean),
                'value_normalizer_std': _finite_item(torch.sqrt(normalizer_var)),
            })

        train_info = defaultdict(float)
        train_info.update(case_weight_info)
        train_info.update(diagnostic_info)
        train_info['actor_grad_accumulation_steps'] = float(
            self.actor_grad_accumulation_steps
        )
        train_info['critic_grad_accumulation_steps'] = float(self.grad_accumulation_steps)
        
        actor_sample_count = 0
        actor_data_sample_count = 0
        actor_accumulation_groups = 0
        actor_planned_optimizer_steps = 0
        post_update_probe_count = 0
        critic_sample_count = 0

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

            for epoch in range(self.ppo_epoch):
                # 每次 epoch 重新打乱数据
                data_generator = buffer.graph_recurrent_generator(advantages, self.mini_batch_size)
                data_samples = list(data_generator)
                total_steps = len(data_samples)
                if total_steps == 0:
                    continue

                stop_actor_epoch = False
                actor_data_sample_count += total_steps
                planned_groups_this_epoch = int(math.ceil(
                    total_steps / self.actor_grad_accumulation_steps
                ))
                actor_planned_optimizer_steps += planned_groups_this_epoch
                for group_start in range(0, total_steps, self.actor_grad_accumulation_steps):
                    actor_accumulation_groups += 1
                    group = data_samples[
                        group_start:group_start + self.actor_grad_accumulation_steps
                    ]
                    actor_mass_index = 13 if self.case_balanced_loss else 11
                    masses = [self._sample_mass(sample, actor_mass_index) for sample in group]
                    group_mass = max(sum(masses), 1e-8)
                    probe_sample = group[int(np.argmax(masses))]
                    group_optimizer_steps = 0.0
                    for group_idx, (sample, sample_mass) in enumerate(zip(group, masses)):
                        perform_step = group_idx == len(group) - 1
                        policy_results = self.update_policy_net(
                            sample,
                            update_actor=True,
                            perform_step=perform_step,
                            loss_scale=sample_mass / group_mass,
                        )

                        for k, v in policy_results.items():
                            train_info[k] += v
                        group_optimizer_steps += policy_results.get(
                            'actor_optimizer_steps', 0.0
                        )
                        actor_sample_count += 1
                        if policy_results.get('kl_early_stop', 0.0) > 0.0:
                            stop_actor_epoch = True
                            break
                    if group_optimizer_steps > 0.0:
                        post_update_results = self.measure_post_update_policy_shift(
                            probe_sample
                        )
                        for k, v in post_update_results.items():
                            train_info[k] += v
                        post_update_probe_count += 1
                        post_gate = self._kl_gate_decision(
                            post_update_results[
                                'post_update_probe_approx_kl'
                            ],
                            post_update_results[
                                'post_update_bc_reference_approx_kl'
                            ],
                        )
                        old_kl_exceeded = post_gate[
                            'old_policy_target_exceeded'
                        ]
                        reference_kl_exceeded = post_gate[
                            'bc_reference_target_exceeded'
                        ]
                        train_info['post_update_target_kl_exceeded'] += float(
                            old_kl_exceeded
                        )
                        train_info[
                            'post_update_bc_reference_target_kl_exceeded'
                        ] += float(reference_kl_exceeded)
                        if post_gate['old_policy_kl_early_stop']:
                            train_info['old_policy_kl_stop_events'] += 1.0
                        if post_gate['bc_reference_kl_early_stop']:
                            train_info['bc_reference_kl_stop_events'] += 1.0
                        if post_gate['kl_early_stop']:
                            train_info['actor_kl_stop_events'] += 1.0
                            stop_actor_epoch = True
                            print(
                                '[PPO] Post-update KL gate exceeded: '
                                f"old={post_update_results['post_update_probe_approx_kl']:.6g}/"
                                f'{self.target_kl:.6g}, '
                                f"bc_ref={post_update_results['post_update_bc_reference_approx_kl']:.6g}/"
                                f'{self.bc_reference_target_kl:.6g} '
                                f'(hard_gate={self.bc_reference_hard_gate}); '
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
            total_steps = len(data_samples)
            if total_steps == 0:
                continue

            for group_start in range(0, total_steps, self.grad_accumulation_steps):
                group = data_samples[
                    group_start:group_start + self.grad_accumulation_steps
                ]
                critic_mass_index = 14 if self.case_balanced_loss else 5
                masses = [self._sample_mass(sample, critic_mass_index) for sample in group]
                group_mass = max(sum(masses), 1e-8)
                for group_idx, (sample, sample_mass) in enumerate(zip(group, masses)):
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
        train_info['post_update_probe_count'] = float(post_update_probe_count)

        return train_info
    
    def prep_training(self):
        self.policy.ac.train()
        if self.policy.has_bc_reference():
            self.policy.bc_reference_ac.eval()

    def prep_rollout(self):
        self.policy.ac.eval()
        if self.policy.has_bc_reference():
            self.policy.bc_reference_ac.eval()
