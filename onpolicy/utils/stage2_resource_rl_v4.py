"""Case-minibatch resource PPO with value or frozen-rollout baselines.

V3 stays available. V4 changes optimization units, never the environment or
the normalized autoregressive action distribution. Every minibatch replays
whole cases from zero with current weights; TBPTT never steps midway through
a case. Frozen encodings are optional FP32 caches, not hidden-state caches.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch

from onpolicy.utils.stage2_resource_rl import (
    DEFAULTS as V3_DEFAULTS, ResourceLearner, RESOURCE_PREFIXES,
    conditional_regularizers, forward, joint_terms, learner_mode,
    require_finite, resource_mask, summary,
)

PROTOCOL = 'stage2_bc_resource_rl_v4'
ARMS = ('S2RL_PPO_R', 'S2RL_PPO_V')
DEFAULTS = {**V3_DEFAULTS, 'actor_lr': 3e-5, 'rounds': 20, 'epochs': 2,
    'rollout_cases': 24, 'minibatch_cases': 6, 'evaluation_workers': 12,
    'critic_warmup_steps': 200, 'critic_warmup_seconds': 300,
    'critic_steps_per_round': 50, 'critic_round_seconds': 60,
    'hard_timeout_seconds': 86400, 'check_timeout_seconds': 7200,
    'cache_frozen_encoding': True, 'encoding_cache_max_bytes': 8 * 1024 ** 3,
    'minibatch_audit_prefix': 64, 'evaluation_rounds': [10, 20],
    'baseline': 'rollout', 'rng_contract': 'separate_init_rollout_shuffle_v1',
    'forward_batch_contract': 'fixed_AR_case_groups_shuffle_groups_only_H_preserves_legacy_batch'}


def case_minibatches(count, size, seed):
    if count <= 0 or size <= 0:
        raise ValueError('Positive case/minibatch counts required.')
    # Fixed membership/row order matches behavior inference exactly. Only the
    # order of minibatches is shuffled; changing GEMM batch shapes/membership
    # otherwise exceeded the strict 1e-6 likelihood contract on the GPU.
    batches = [list(range(start, min(count, start + size))) for start in range(0, count, size)]
    return [batches[i] for i in np.random.default_rng(int(seed)).permutation(len(batches))]


def minibatch_scale(total_cases, selected_cases, total_events):
    """Expected minibatch gradient equals the full-rollout event-sum gradient.

    Do not divide each case by its own event count. Averaging these uniformly
    sampled equal-sized minibatch gradients reproduces sum(loss)/total_events.
    """
    if not 0 < selected_cases <= total_cases or total_events <= 0:
        raise ValueError('Invalid rollout/minibatch normalization.')
    return total_cases / (selected_cases * total_events)


def rollout_baseline_advantages(cases, reference_cmax, valid, scale=10000.):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid return scale.')
    values = []
    for row in cases:
        name = row['case_dir']
        if not row['completed'] or row['timeout'] or row['cycle_terminated']:
            raise ValueError('Rollout baseline cannot train an incomplete case.')
        baseline = float(reference_cmax[name])
        current = float(row['makespan'])
        if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(current) or current <= 0:
            raise ValueError('Invalid frozen baseline or student Cmax.')
        values.append((baseline - current) / scale)
    case_adv = torch.tensor(values, dtype=torch.float32)
    if valid.shape[1] != len(values) or not valid.any():
        raise ValueError('Rollout baseline case/mask coverage mismatch.')
    raw = case_adv.expand(valid.shape[0], -1)[valid]
    return raw, dict(raw_case_advantages=values,
        cases_better_than_frozen_baseline=sum(v > 1e-10 for v in values),
        cases_worse_than_frozen_baseline=sum(v < -1e-10 for v in values),
        baseline='frozen_B0_hungarian_complete_case_cmax',
        credit_assignment='same_MC_terminal_difference_for_each_resource_event')


def trace_event(obs, actions, active, logits, types, physical_time, step, plane_count):
    """Small, observable-only trace. Semantic IDs are stable within a case.

    No counterfactual execution, teacher facts or future readiness timestamps.
    Operation/site IDs refer to the immutable case graph, not request positions.
    """
    features = obs['request'].x.cpu()
    op_ids = obs.request_operation_indices.reshape(-1).cpu()
    site_ids = obs.request_site_indices.reshape(-1).cpu()
    lookahead = obs.request_is_lookahead.reshape(-1).cpu()
    kinds = obs.request_kind_ids.reshape(-1).cpu()
    identities = []
    for i, feat in enumerate(features):
        if i and int(op_ids[i]) < 0 and int(site_ids[i]) < 0:
            continue
        identities.append(dict(request_index=i, operation_index=int(op_ids[i]),
            site_index=int(site_ids[i]), plane_index=int(feat[6]),
            resource_type_id=int(feat[0]), lookahead=bool(lookahead[i]),
            kind=int(kinds[i]), observable_time_feature=float(feat[1]), noop=i == 0))
    resources = []
    for i in range(plane_count, len(actions)):
        if not bool(np.asarray(active[i]).any()):
            continue
        support = torch.nonzero(torch.isfinite(logits[i]), as_tuple=False).flatten().tolist()
        resources.append(dict(agent_index=i, role=int(np.asarray(types[i]).reshape(-1)[0]),
            request_index=int(actions[i, 0]), allowed_request_indices=support,
            forced=len(support) == 1))
    digest = hashlib.sha256()
    for value in (features, obs['device'].x.cpu(), obs.global_features.cpu()):
        digest.update(value.contiguous().numpy().tobytes())
    return dict(step=step, physical_time=physical_time, actions=np.asarray(actions).tolist(),
        resource_actions=resources, requests=identities,
        device_features=obs['device'].x.cpu().tolist(),
        observed_feature_sha256=digest.hexdigest(), hash_is_full_simulator_state=False)


def compare_cases(reference, candidate):
    from onpolicy.scripts.train.run_stage2_resource_rl import metrics
    ref = {r['case_sha256']: r for r in reference}
    cur = {r['case_sha256']: r for r in candidate}
    if len(ref) != len(reference) or len(cur) != len(candidate) or ref.keys() != cur.keys() or not ref:
        raise ValueError('Paired evaluation must have identical unique case hashes.')
    before, after = metrics(reference), metrics(candidate)
    rows = [dict(case_dir=ref[k]['case_dir'], case_sha256=k,
                 distribution=ref[k]['distribution'],
                 delta=cur[k]['makespan'] - ref[k]['makespan'],
                 relative_delta=cur[k]['makespan'] / ref[k]['makespan'] - 1)
            for k in sorted(ref)]
    delta = [r['delta'] for r in rows]
    return dict(mean_delta=float(np.mean(delta)),
        relative_improvement=1 - after['mean'] / before['mean'],
        wins=sum(d < -1e-6 for d in delta), ties=sum(abs(d) <= 1e-6 for d in delta),
        losses=sum(d > 1e-6 for d in delta),
        maximum_cmax_delta=after['maximum'] - before['maximum'],
        maximum_paired_delta=max(delta), maximum_positive_paired_delta=max(0., max(delta)),
        maximum_relative_paired_delta=max(r['relative_delta'] for r in rows),
        tail10_delta=after['tail10'] - before['tail10'],
        group_deltas={g: after['groups'][g] - before['groups'][g] for g in before['groups']},
        paired_cases=rows)


def screen_candidate(vs_ar, vs_hungarian):
    # Development-only investment screen; IGA180 still has zero case regret.
    return bool(all(x['relative_improvement'] >= .005
                    and max(x['group_deltas'].values()) <= 0
                    and x['tail10_delta'] <= 0
                    for x in (vs_ar, vs_hungarian))
                and vs_hungarian['maximum_relative_paired_delta'] <= .01)


class CaseMinibatchLearner(ResourceLearner):
    def __init__(self, policy, args, contract, progress):
        super().__init__(policy, args, contract, progress)
        if contract['baseline'] not in ('rollout', 'value', 'bc'):
            raise ValueError('Unknown V4 baseline.')
        if contract['baseline'] != 'value':
            for parameter in self.critic_params:
                parameter.requires_grad_(False)
        self.minibatch_checks = dict(events=0, max_logp_error=0., max_ratio_error=0.,
                                    current_weights_full_prefix=True)

    def collect_forward(self, policy, obs, hidden, active, history, types, *,
                        actions=None, encoded=None, hungarian=False, **kwargs):
        if hungarian:
            # Preserve historical deterministic Hungarian numerical batching.
            return forward(policy, obs, hidden, active, history, types,
                           actions=actions, encoded=encoded, hungarian=True, **kwargs)
        size = self.contract['minibatch_cases']
        groups = [list(range(start, min(len(obs), start + size))) for start in range(0, len(obs), size)]
        outputs, encodings = [], {}
        for indices in groups:
            cached = None if encoded is None else encoded['case_group_encodings'][tuple(indices)]
            out, enc = forward(policy, [obs[i] for i in indices], hidden[indices], active[indices],
                history[indices], types[indices], actions=None if actions is None else actions[indices],
                encoded=cached, **kwargs)
            outputs.append(out)
            encodings[tuple(indices)] = enc
        merged = {k: torch.cat([out[k] for out in outputs], dim=0) for k in outputs[0]}
        return merged, {'case_group_encodings': encodings}

    def cache_encoding(self, encoded):
        if 'case_group_encodings' not in encoded:
            return super().cache_encoding(encoded)
        groups, total = {}, 0
        for indices, value in encoded['case_group_encodings'].items():
            cached, count = super().cache_encoding(value)
            groups[indices] = cached
            total += count
        return {'case_group_encodings': groups}, total

    def _forward_subset(self, frame, hidden, indices, *, bc=False, use_cache=True):
        encoded = frame.get('encoded') if use_cache else None
        if encoded is not None:
            if 'case_group_encodings' in encoded:
                if tuple(indices) not in encoded['case_group_encodings']:
                    raise ValueError('Replay changed the canonical behavior case group.')
                encoded = {k: v.to(self.device) for k, v in encoded['case_group_encodings'][tuple(indices)].items()}
            else:
                encoded = {k: v[indices].to(self.device) for k, v in encoded.items()}
        return forward(self.policy, [frame['obs'][i] for i in indices], hidden,
            frame['active'][indices], frame['history'][indices], frame['types'][indices],
            actions=frame['teacher_actions'][indices] if bc else frame['actions'][indices],
            encoded=encoded)[0]

    def _hidden(self, n):
        return torch.zeros((n, self.args.max_agent_num + self.args.max_device_num,
                           self.args.recurrent_N, self.args.hidden_size), device=self.device)

    def audit_minibatches(self, rollout):
        """Before any update, verify current-weight prefixes in each case shard."""
        self.policy.ac.eval()
        indices_list = case_minibatches(len(rollout['cases']), self.contract['minibatch_cases'], 0)
        for indices in indices_list:
            hidden = self._hidden(len(indices))
            with torch.no_grad():
                for frame in rollout['frames'][:self.contract['minibatch_audit_prefix']]:
                    out = self._forward_subset(frame, hidden, indices)
                    old = frame['old_logp'][indices].to(self.device)
                    mask = frame['mask'][indices].to(self.device)
                    actual_mask = resource_mask(out['decision_mask'], self.args.max_agent_num)
                    if not torch.equal(mask, actual_mask) or not np.array_equal(
                            out['actions'].cpu().numpy(), frame['actions'][indices]):
                        raise RuntimeError('Case-minibatch replay changed executed actions or masks.')
                    error = float(((out['log_probs'] - old) * mask).abs().max())
                    ratio_error = float((((out['log_probs'] - old) * mask).sum(-1).exp() - 1).abs().max())
                    if error > 1e-6 or ratio_error > 1e-5:
                        raise RuntimeError(f'Case-minibatch likelihood mismatch: {error}, {ratio_error}')
                    self.minibatch_checks['events'] += int(mask.any(-1).sum())
                    self.minibatch_checks['max_logp_error'] = max(self.minibatch_checks['max_logp_error'], error)
                    self.minibatch_checks['max_ratio_error'] = max(self.minibatch_checks['max_ratio_error'], ratio_error)
                    done = torch.as_tensor(frame['dones'][indices], device=self.device, dtype=torch.bool)
                    hidden = out['rnn_states'] * (~done).unsqueeze(-1).unsqueeze(-1)
        self.progress('minibatch_probability_audit', force=True, minibatch_checks=self.minibatch_checks)

    def _replay(self, rollout, indices, advantages, *, gradient, bc=False):
        frames, n = rollout['frames'], len(rollout['cases'])
        count = int(torch.stack([f['mask'].any(-1) for f in frames]).sum())
        scale = minibatch_scale(n, len(indices), count)
        hidden, chunk_loss = self._hidden(len(indices)), None
        sums = dict(kl_sum=0., clip_count=0, joint_count=0, policy_loss=0.,
                    reference_kl=0., entropy=0.)
        with torch.set_grad_enabled(gradient):
            for t, frame in enumerate(frames):
                out = self._forward_subset(frame, hidden, indices, bc=bc)
                mask = resource_mask(out['decision_mask'], self.args.max_agent_num)
                if bc:
                    loss = -(out['log_probs'] * mask).sum(-1)[mask.any(-1)].sum()
                    regularizer = loss.new_zeros(())
                    sums['joint_count'] += int(mask.any(-1).sum())
                else:
                    if not torch.equal(mask, frame['mask'][indices].to(self.device)):
                        raise RuntimeError('Case-minibatch replay conditional decision mask changed.')
                    loss, info = joint_terms(out['log_probs'], frame['old_logp'][indices].to(self.device),
                        mask, advantages[t, indices].to(self.device), self.contract['clip'])
                    for k, value in info.items():
                        sums[k] += value
                    kl, entropy = conditional_regularizers(out['resource_logits'],
                        frame['reference_logits'][indices].to(self.device), mask)
                    regularizer = self.contract['reference_kl_coef'] * kl - self.contract['entropy_coef'] * entropy
                    sums['reference_kl'] += float(kl.detach())
                    sums['entropy'] += float(entropy.detach())
                require_finite(loss + regularizer, 'case-minibatch loss')
                sums['policy_loss'] += float(loss.detach())
                if gradient:
                    term = (loss + regularizer) * scale
                    chunk_loss = term if chunk_loss is None else chunk_loss + term
                done = torch.as_tensor(frame['dones'][indices], device=self.device, dtype=torch.bool)
                hidden = out['rnn_states'] * (~done).unsqueeze(-1).unsqueeze(-1)
                ended = bool(done.all())
                if (t + 1) % self.contract['chunk_length'] == 0 or t + 1 == len(frames) or ended:
                    if gradient and chunk_loss is not None and chunk_loss.requires_grad:
                        chunk_loss.backward()
                    chunk_loss = None
                    hidden = hidden.detach()
                    self.progress('actor_replay' if gradient else 'post_update_kl_replay',
                        replay_step=t + 1, replay_total=len(frames), actor_steps=self.actor_steps,
                        minibatch_cases=len(indices))
                if ended:
                    break
        sums['approx_kl'] = sums['kl_sum'] / max(1, sums['joint_count'])
        sums['normalization_scale'] = scale
        return sums

    @staticmethod
    def value_metrics(pred, targets):
        residual = targets - pred
        variance = float(targets.var(unbiased=False))
        return dict(mse=float(residual.square().mean()), value_mean=float(pred.mean()),
            return_mean=float(targets.mean()), value_std=float(pred.std(unbiased=False)),
            return_std=float(targets.std(unbiased=False)),
            explained_variance=float(1 - residual.var(unbiased=False) / variance) if variance > 1e-12 else None)

    def warmup_value(self, data, validation_indices):
        if self.contract['baseline'] != 'value':
            raise ValueError('Only PPO-V fits a value baseline.')
        valid_cases = torch.tensor(list(validation_indices), dtype=torch.long)
        held = torch.isin(data['case_indices'], valid_cases)
        if not held.any() or held.all():
            raise ValueError('Critic warmup needs case-disjoint fitting and validation samples.')
        x, y = data['embeddings'].to(self.device), data['targets'].to(self.device)
        held = held.to(self.device)
        started, losses = time.monotonic(), []
        before = self.actor_steps
        with torch.no_grad():
            initial = self.value_metrics(self.policy.ac.team_critic(x[held]).flatten(), y[held])
        while len(losses) < self.contract['critic_warmup_steps']:
            if time.monotonic() - started >= self.contract['critic_warmup_seconds']:
                break
            count = min(10, self.contract['critic_warmup_steps'] - len(losses))
            losses += self.critic_update(x[~held], y[~held], count)
            self.progress('critic_warmup', critic_steps=self.critic_steps,
                          critic_warmup_limit=self.contract['critic_warmup_steps'])
        with torch.no_grad():
            final = self.value_metrics(self.policy.ac.team_critic(x[held]).flatten(), y[held])
            fitted = self.value_metrics(self.policy.ac.team_critic(x[~held]).flatten(), y[~held])
        if before != self.actor_steps:
            raise RuntimeError('Critic warmup changed actor update count.')
        return dict(steps=len(losses), seconds=time.monotonic() - started,
            validation_case_indices=list(validation_indices), validation_before=initial,
            validation_after=final, fitted_after=fitted, losses=losses,
            validation_scope='case_disjoint_initial_warmup_only_not_independent_policy_evaluation')

    def update(self, rollout, *, bc=False, warmup=False, round_index=1, reference_cmax=None):
        if warmup:
            raise ValueError('V4 critic warmup uses separate frozen B0 trajectories, never current actor targets.')
        c = self.contract
        if bc != (c['baseline'] == 'bc'):
            raise ValueError('Baseline/BC objective mismatch.')
        self.audit_minibatches(rollout)
        frames = rollout['frames']
        valid = torch.stack([f['mask'].any(-1) for f in frames])
        denominator = int(valid.sum())
        if denominator <= 0:
            raise ValueError('No trainable resource joint events.')
        advantages = torch.zeros(valid.shape)
        signal = dict(valid_joint_events=denominator, case_episodes=len(rollout['cases']),
                      rollout_events=rollout['environment_events'], baseline=c['baseline'])
        embeddings = targets = None
        if not bc:
            if c['baseline'] == 'rollout':
                raw, baseline_info = rollout_baseline_advantages(rollout['cases'], reference_cmax or {}, valid, c['return_scale'])
                signal.update(baseline_info)
            else:
                embeddings, targets, _ = self._critic_batch(rollout)
                with torch.no_grad():
                    values = self.policy.ac.team_critic(embeddings).flatten()
                    require_finite(values, 'pre-fit value baseline')
                    raw = (targets - values).cpu()
                    signal.update(self.value_metrics(values, targets))
                    signal['value_snapshot'] = 'before_fitting_this_rollout'
            require_finite(raw, 'raw advantages')
            mean, std = raw.mean(), raw.std(unbiased=False).clamp_min(1e-6)
            advantages[valid] = (raw - mean) / std
            signal.update(raw_advantage_mean=float(mean), raw_advantage_std=float(std),
                          raw_advantage_min=float(raw.min()), raw_advantage_max=float(raw.max()),
                          normalization='one_fixed_event_weighted_normalization_per_rollout')
        learner_mode(self.policy.ac)
        records, stopped_for_kl = [], False
        for pass_index in range(c['ppo_passes']):
            batches = case_minibatches(len(rollout['cases']), c['minibatch_cases'],
                                      c['seed'] * 100000 + round_index * 100 + pass_index)
            for batch_index, indices in enumerate(batches):
                before = summary(self.policy.ac, RESOURCE_PREFIXES)
                self.actor_optim.zero_grad(set_to_none=True)
                started = time.monotonic()
                record = self._replay(rollout, indices, advantages, gradient=True, bc=bc)
                skip = not bc and record['approx_kl'] > c['target_kl']
                norm = torch.nn.utils.clip_grad_norm_(self.actor_params, c['max_grad_norm'])
                require_finite(norm, 'case-minibatch gradient')
                if not skip:
                    if float(norm) <= 0:
                        raise RuntimeError('No actor gradient; cannot claim an optimizer update.')
                    self.actor_optim.step()
                    self.actor_steps += 1
                    if summary(self.policy.ac, RESOURCE_PREFIXES) == before:
                        raise RuntimeError('Actor optimizer did not change tensors.')
                self.actor_optim.zero_grad(set_to_none=True)
                self.assert_protected()
                record.update(pass_index=pass_index + 1, minibatch_index=batch_index + 1,
                    case_indices=indices, gradient_norm=float(norm), actor_steps=self.actor_steps,
                    skipped_for_kl=skip, seconds=time.monotonic() - started,
                    kl_measurement='before_this_minibatch_optimizer_step')
                records.append(record)
                self.progress('actor_minibatch_completed', force=True, **record)
                if skip:
                    stopped_for_kl = True
                    break
            if stopped_for_kl:
                break
        post = None
        if not bc:
            self.policy.ac.eval()
            post = dict(kl_sum=0., clip_count=0, joint_count=0, policy_loss=0., reference_kl=0., entropy=0.)
            for indices in case_minibatches(len(rollout['cases']), c['minibatch_cases'], 0):
                part = self._replay(rollout, indices, advantages, gradient=False)
                for key in post:
                    post[key] += part[key]
            post['approx_kl'] = post['kl_sum'] / max(1, post['joint_count'])
            post['kl_measurement'] = 'post_all_updates_full_rollout_current_weight_canonical_case_groups'
        losses, started = [], time.monotonic()
        if c['baseline'] == 'value':
            while len(losses) < c['critic_steps_per_round']:
                if time.monotonic() - started >= c['critic_round_seconds']:
                    break
                count = min(10, c['critic_steps_per_round'] - len(losses))
                losses += self.critic_update(embeddings, targets, count)
                self.progress('critic_update', critic_steps=self.critic_steps)
        self.assert_protected()
        return dict(minibatches=records, actor_steps=self.actor_steps, critic_steps=self.critic_steps,
            critic_losses=losses, signal_metrics=signal, post_update=post,
            paused_for_kl=bool(post is not None and post['approx_kl'] > c['target_kl']),
            skipped_remaining_minibatches_for_kl=stopped_for_kl,
            frozen_parameters_unchanged=True, minibatch_checks=dict(self.minibatch_checks),
            recurrent_contract='current_weights_zero_hidden_each_complete_case_minibatch_TBPTT16',
            optimization_unit='one_step_per_complete_case_minibatch',
            actual_minibatch_size=c['minibatch_cases'])
