"""Execution-only V4 acceleration; the original learner remains unchanged.

Canonical case groups, FP32 distributions, fresh current-weight recurrent
prefixes and every numerical guard are retained. Immutable replay inputs are
prepared once. Small diagnostic scalars stay on device until a TBPTT boundary
(guards) or a complete minibatch (statistics), always before an optimizer step.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from onpolicy.utils.stage2_resource_rl import resource_mask
from onpolicy.utils.stage2_resource_rl_v4 import (
    CaseMinibatchLearner, case_minibatches, minibatch_scale,
)

FAST_VERSION = 'v4_execution_cache_sync_v1'


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if hasattr(value, 'to_dict'):
        return tensor_bytes(value.to_dict())
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.nbytes
    return 0


def prepared_forward(policy, data, info, hidden, *, actions=None, encoded=None,
                     deterministic=False, reference=False, hungarian=False):
    ac = policy.bc_reference_ac if reference else policy.ac
    if ac is None:
        raise ValueError('Missing immutable BC reference.')
    # Never store or reuse a trainable recurrent state in the input cache.
    current = {**data, 'hidden_states': policy._to_tensor(hidden, torch.float32)}
    if encoded is None:
        encoded = ac._encode_graph(current['graph'], actor_grad=False)
    chosen = None if actions is None else torch.as_tensor(actions, device=policy.device, dtype=torch.long)
    out = ac(current, info, deterministic=deterministic, plane_deterministic=True,
        chosen_op=None if chosen is None else chosen[..., 0],
        chosen_site=None if chosen is None else chosen[..., 1],
        chosen_order=None if chosen is None or chosen.shape[-1] < 3 else chosen[..., 2],
        criticize=False, eval_action=actions is not None, resource_rl=not hungarian,
        return_actor_details=True, encoded_graph=encoded)
    return out, encoded


def device_terms(current_logp, old_logp, mask, advantage, clip, current, reference):
    """Original expressions and reduction order; no per-event host scalars."""
    log_ratio = ((current_logp - old_logp) * mask).sum(-1)
    ratio = log_ratio.exp()
    valid = mask.any(-1)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    loss = -surrogate[valid].sum()
    approx_kl = torch.expm1(log_ratio) - log_ratio
    support = torch.isfinite(current)
    cur = torch.where(support, current, torch.zeros_like(current))
    ref = torch.where(support, reference, torch.zeros_like(reference))
    pc = torch.where(support, current.exp(), torch.zeros_like(current))
    pr = torch.where(support, reference.exp(), torch.zeros_like(reference))
    count = mask.sum(-1).clamp_min(1)
    kl = (((pr * (ref - cur)).sum(-1) * mask).sum(-1) / count).sum()
    entropy = (((-pc * cur).sum(-1) * mask).sum(-1) / count).sum()
    flags = torch.stack((torch.isfinite(current_logp).all(), torch.isfinite(old_logp).all(),
        torch.isfinite(ratio).all(), (support[mask] == torch.isfinite(reference)[mask]).all(),
        torch.isclose(pc.sum(-1)[mask], torch.ones_like(pc.sum(-1)[mask]), atol=1e-5, rtol=0).all(),
        torch.isclose(pr.sum(-1)[mask], torch.ones_like(pr.sum(-1)[mask]), atol=1e-5, rtol=0).all(),
        torch.isfinite(kl), torch.isfinite(entropy)))
    # Python float accumulation in V4 is sequential binary64 addition. Preserve
    # that order on device, not an FP32 sum over the complete trajectory.
    stats = torch.stack((approx_kl[valid].detach().sum().double(),
        ((ratio - 1).abs()[valid] > clip).sum().double(), valid.sum().double(),
        loss.detach().double(), kl.detach().double(), entropy.detach().double()))
    return loss, kl, entropy, stats, flags


class FastCaseMinibatchLearner(CaseMinibatchLearner):
    def __init__(self, *args, cache_max_bytes=4 * 1024 ** 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_max_bytes = int(cache_max_bytes)
        if self.cache_max_bytes < 0:
            raise ValueError('Negative replay cache budget.')
        self._prepared = {}
        self._resident_bytes = 0
        self._collect_obs = None
        self._collect_inputs = {}
        self.fast_stats = {}

    def _inputs(self, graph, hidden, active, history, types):
        data, info = self.policy._build_inputs(graph, hidden, active,
            history[..., 0], history[..., 1], agent_types=types)
        data.pop('hidden_states')
        return data, info

    def collect_forward(self, policy, obs, hidden, active, history, types, *,
                        actions=None, encoded=None, hungarian=False, **kwargs):
        if self._collect_obs is not obs:
            self._collect_obs, self._collect_inputs = obs, {}
        size = len(obs) if hungarian else self.contract['minibatch_cases']
        outputs, encodings = [], {}
        for start in range(0, len(obs), size):
            indices = list(range(start, min(len(obs), start + size)))
            key = tuple(indices)
            if key not in self._collect_inputs:
                self._collect_inputs[key] = self._inputs(Batch.from_data_list([obs[i] for i in indices]),
                    hidden[indices], active[indices], history[indices], types[indices])
            data, info = self._collect_inputs[key]
            cached = None if encoded is None else encoded if hungarian else encoded['case_group_encodings'][key]
            out, enc = prepared_forward(policy, data, info, hidden[indices],
                actions=None if actions is None else actions[indices], encoded=cached,
                hungarian=hungarian, **kwargs)
            outputs.append(out)
            encodings[key] = enc
        if hungarian:
            return outputs[0], encodings[tuple(range(len(obs)))]
        return {k: torch.cat([out[k] for out in outputs], 0) for k in outputs[0]}, {
            'case_group_encodings': encodings}

    def collect(self, *args, **kwargs):
        self._collect_obs, self._collect_inputs = None, {}
        try:
            return super().collect(*args, **kwargs)
        finally:
            self._collect_obs, self._collect_inputs = None, {}

    def clear_replay_cache(self):
        self._prepared.clear()
        self._resident_bytes = 0

    def _entry(self, frame, indices):
        key = (id(frame), tuple(indices))
        if key not in self._prepared:
            group = tuple(indices)
            source = frame.get('encoded')
            if source is not None and ('case_group_encodings' not in source or group not in source['case_group_encodings']):
                raise ValueError('Replay changed the canonical behavior case group.')
            graph = Batch.from_data_list([frame['obs'][i] for i in indices])
            entry = dict(graph=graph, active=frame['active'][indices], history=frame['history'][indices],
                types=frame['types'][indices], actions=torch.as_tensor(frame['actions'][indices]),
                old_logp=frame['old_logp'][indices], mask=frame['mask'][indices],
                reference_logits=frame['reference_logits'][indices],
                dones=torch.as_tensor(frame['dones'][indices], dtype=torch.bool),
                encoded=None if source is None else source['case_group_encodings'][group],
                ended=bool(np.asarray(frame['dones'][indices]).all()))
            cost = tensor_bytes(entry)
            resident = self._resident_bytes + cost <= self.cache_max_bytes
            entry['resident'] = resident
            if resident:
                self._resident_bytes += cost
                entry = self._materialize(entry)
            self._prepared[key] = entry
            self.fast_stats['prepared_groups'] = len(self._prepared)
            self.fast_stats['resident_bytes'] = self._resident_bytes
            self.fast_stats['cpu_fallback_groups'] = self.fast_stats.get('cpu_fallback_groups', 0) + int(not resident)
        entry = self._prepared[key]
        return entry if entry['resident'] else self._materialize(entry)

    def _materialize(self, source):
        entry = dict(source)
        # PyG .to() mutates its containers. Clone only the CPU fallback so a
        # bounded cache can never accidentally retain unaccounted GPU graphs.
        graph = source['graph'] if source['resident'] else source['graph'].clone()
        entry['data'], entry['info'] = self._inputs(graph, self._hidden(len(source['actions'])),
            source['active'], source['history'], source['types'])
        for key in ('actions', 'old_logp', 'mask', 'reference_logits', 'dones'):
            entry[key] = source[key].to(self.device)
        if source['encoded'] is not None:
            entry['encoded'] = {k: v.to(self.device) for k, v in source['encoded'].items()}
        return entry

    def _forward_subset(self, frame, hidden, indices, *, bc=False, use_cache=True):
        if bc or not use_cache:
            return super()._forward_subset(frame, hidden, indices, bc=bc, use_cache=use_cache)
        entry = self._entry(frame, indices)
        return prepared_forward(self.policy, entry['data'], entry['info'], hidden,
            actions=entry['actions'], encoded=entry['encoded'])[0]

    def audit_minibatches(self, rollout):
        self.policy.ac.eval()
        for indices in case_minibatches(len(rollout['cases']), self.contract['minibatch_cases'], 0):
            hidden = self._hidden(len(indices))
            error = torch.zeros((), device=self.device)
            ratio_error, events = error.clone(), error.clone()
            equal = torch.ones((), device=self.device, dtype=torch.bool)
            with torch.no_grad():
                for frame in rollout['frames'][:self.contract['minibatch_audit_prefix']]:
                    entry = self._entry(frame, indices)
                    out = self._forward_subset(frame, hidden, indices)
                    mask = entry['mask']
                    equal &= (mask == resource_mask(out['decision_mask'], self.args.max_agent_num)).all()
                    equal &= (out['actions'] == entry['actions']).all()
                    delta = (out['log_probs'] - entry['old_logp']) * mask
                    error = torch.maximum(error, delta.abs().max())
                    ratio_error = torch.maximum(ratio_error, (delta.sum(-1).exp() - 1).abs().max())
                    events += mask.any(-1).sum()
                    hidden = out['rnn_states'] * (~entry['dones']).unsqueeze(-1).unsqueeze(-1)
            actual, ratio, count, same = torch.stack((error, ratio_error, events, equal.float())).cpu().tolist()
            if not same or not np.isfinite([actual, ratio]).all() or actual > 1e-6 or ratio > 1e-5:
                raise RuntimeError(f'Case-minibatch probability/action/mask mismatch: {actual}, {ratio}, {same}')
            self.minibatch_checks['events'] += int(count)
            self.minibatch_checks['max_logp_error'] = max(actual, self.minibatch_checks['max_logp_error'])
            self.minibatch_checks['max_ratio_error'] = max(ratio, self.minibatch_checks['max_ratio_error'])
        self.progress('minibatch_probability_audit', force=True, minibatch_checks=self.minibatch_checks)

    def _replay(self, rollout, indices, advantages, *, gradient, bc=False):
        if bc:
            return super()._replay(rollout, indices, advantages, gradient=gradient, bc=True)
        frames = rollout['frames']
        count = int(torch.stack([f['mask'].any(-1) for f in frames]).sum())
        scale = minibatch_scale(len(rollout['cases']), len(indices), count)
        selected_advantages = advantages[:, indices].to(self.device)
        hidden, chunk_loss = self._hidden(len(indices)), None
        sums = torch.zeros(6, dtype=torch.float64, device=self.device)
        guards = torch.ones(10, dtype=torch.bool, device=self.device)
        with torch.set_grad_enabled(gradient):
            for t, frame in enumerate(frames):
                entry = self._entry(frame, indices)
                out = self._forward_subset(frame, hidden, indices)
                mask = resource_mask(out['decision_mask'], self.args.max_agent_num)
                loss, kl, entropy, stats, flags = device_terms(out['log_probs'], entry['old_logp'], mask,
                    selected_advantages[t], self.contract['clip'], out['resource_logits'], entry['reference_logits'])
                regularizer = self.contract['reference_kl_coef'] * kl - self.contract['entropy_coef'] * entropy
                guards[:8] &= flags
                guards[8] &= (mask == entry['mask']).all()
                guards[9] &= torch.isfinite(loss + regularizer)
                sums += stats
                if gradient:
                    term = (loss + regularizer) * scale
                    chunk_loss = term if chunk_loss is None else chunk_loss + term
                hidden = out['rnn_states'] * (~entry['dones']).unsqueeze(-1).unsqueeze(-1)
                if (t + 1) % self.contract['chunk_length'] == 0 or t + 1 == len(frames) or entry['ended']:
                    # All original finite/support/normalization/mask guards run
                    # before backward and, critically, before any Adam step.
                    if not bool(guards.all()):
                        raise FloatingPointError(f'Invalid replay; guard vector: {guards.cpu().tolist()}')
                    if gradient and chunk_loss is not None and chunk_loss.requires_grad:
                        chunk_loss.backward()
                    chunk_loss, hidden = None, hidden.detach()
                    self.progress('actor_replay' if gradient else 'post_update_kl_replay',
                        replay_step=t + 1, replay_total=len(frames), actor_steps=self.actor_steps,
                        minibatch_cases=len(indices))
                if entry['ended']:
                    break
        values = sums.cpu().tolist()
        result = dict(zip(('kl_sum', 'clip_count', 'joint_count', 'policy_loss', 'reference_kl', 'entropy'), values))
        result['clip_count'], result['joint_count'] = int(result['clip_count']), int(result['joint_count'])
        result['approx_kl'] = result['kl_sum'] / max(1, result['joint_count'])
        result['normalization_scale'] = scale
        return result

    def update(self, rollout, **kwargs):
        self.clear_replay_cache()
        self.fast_stats = dict(version=FAST_VERSION, cache_limit_bytes=self.cache_max_bytes,
            dtype_changed=False, recurrent_hidden_cached=False, guard_boundary='before_each_TBPTT_backward')
        started = time.monotonic()
        try:
            result = super().update(rollout, **kwargs)
            result['acceleration'] = {**self.fast_stats, 'update_seconds': time.monotonic() - started}
            if self.device.type == 'cuda':
                result['acceleration'].update(cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device))
            return result
        finally:
            self.clear_replay_cache()
