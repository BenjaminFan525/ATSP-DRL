"""Resource-only, on-policy Stage2 PPO; no cost branches or teacher execution.

This deliberately does not reuse the legacy per-agent PPO/BC-reference
estimators. Each environment event has one autoregressive resource likelihood.
Recurrent replay starts at zero under the current weights on EVERY pass;
optimizer steps happen only after the entire collection has been replayed.
Chunk boundaries truncate gradients, never substitute old-policy hidden states.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
from onpolicy.utils.training_stage import protected_parameter_summary


PROTOCOL = 'stage2_bc_resource_rl_v3'
RESOURCE_PREFIXES = ('device_sel_enc.', 'device_actor.',
                     'transporter_sel_enc.', 'transporter_actor.')
CRITIC_PREFIXES = ('team_critic.',)
DEFAULTS = dict(actor_lr=1e-5, critic_lr=1e-4, clip=0.1, ppo_passes=2,
                entropy_coef=0.001, reference_kl_coef=0.01, target_kl=0.01,
                return_scale=10000.0, gamma=1.0, critic_warmup_steps=10,
                chunk_length=16, max_grad_norm=1.0, seed=11, rounds=2,
                rollout_max_steps=4000, hard_timeout_seconds=14400,
                check_timeout_seconds=1800)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summary(model, prefixes=None, protected=False):
    state = model.state_dict() if hasattr(model, 'state_dict') else model
    if protected:
        state = {k: v for k, v in state.items()
                 if not k.startswith(RESOURCE_PREFIXES + CRITIC_PREFIXES)}
        prefixes = ('',)
    return protected_parameter_summary(state, prefixes=prefixes or ('',))


def require_finite(value, label):
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f'Non-finite {label}; no sanitizing or retry is permitted.')


def configure_trainability(policy, *, critic=True):
    ac = policy.ac
    if ac.device_policy_head_mode != 'shared' or ac.request_ready_policy_injection != 'none':
        raise ValueError('Initial resource-RL protocol is restricted to the verified B0 architecture.')
    actor_params, critic_params = [], []
    for name, parameter in ac.named_parameters():
        actor = name.startswith(RESOURCE_PREFIXES)
        value = critic and name.startswith(CRITIC_PREFIXES)
        parameter.requires_grad_(actor or value)
        if actor:
            actor_params.append(parameter)
        if value:
            critic_params.append(parameter)
    if not actor_params or (critic and not critic_params):
        raise ValueError('Empty resource actor or team critic parameter whitelist.')
    learner_mode(ac)
    return actor_params, critic_params


def learner_mode(ac):
    ac.eval()
    # cuDNN GRU backward needs train mode; pointer/encoder dropout stays OFF.
    for module in ac.modules():
        if isinstance(module, torch.nn.RNNBase) and any(p.requires_grad for p in module.parameters()):
            if module.dropout != 0:
                raise ValueError('Resource recurrent dropout must be zero.')
            module.train()


def reset_critic(ac):
    for module in ac.team_critic.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()


def forward(policy, obs, hidden, active, history, types, *, actions=None,
            deterministic=False, reference=False, encoded=None, hungarian=False):
    ac = policy.bc_reference_ac if reference else policy.ac
    if ac is None:
        raise ValueError('Missing frozen BC reference.')
    data, info = policy._build_inputs(Batch.from_data_list(list(obs)), hidden,
        active, history[..., 0], history[..., 1], agent_types=types)
    if encoded is None:
        encoded = ac._encode_graph(data['graph'], actor_grad=False)
    chosen = None if actions is None else torch.as_tensor(actions, device=policy.device, dtype=torch.long)
    result = ac(data, info, deterministic=deterministic, plane_deterministic=True,
        chosen_op=None if chosen is None else chosen[..., 0],
        chosen_site=None if chosen is None else chosen[..., 1],
        chosen_order=None if chosen is None or chosen.shape[-1] < 3 else chosen[..., 2],
        criticize=False, eval_action=actions is not None,
        resource_rl=not hungarian, return_actor_details=True, encoded_graph=encoded)
    return result, encoded


def resource_mask(decision_mask, plane_count):
    mask = decision_mask.bool().clone()
    mask[:, :plane_count] = False
    return mask


def joint_terms(new_logp, old_logp, mask, advantage, clip):
    require_finite(new_logp, 'new log-probability')
    require_finite(old_logp, 'behavior log-probability')
    log_ratio = ((new_logp - old_logp) * mask).sum(-1)
    ratio = log_ratio.exp()
    require_finite(ratio, 'joint importance ratio')
    valid = mask.any(-1)
    surrogate = torch.minimum(ratio * advantage,
        ratio.clamp(1 - clip, 1 + clip) * advantage)
    approx_kl = torch.expm1(log_ratio) - log_ratio
    return -surrogate[valid].sum(), {
        'kl_sum': float(approx_kl[valid].detach().sum()),
        'clip_count': int(((ratio - 1).abs()[valid] > clip).sum()),
        'joint_count': int(valid.sum()),
    }


def conditional_regularizers(current, reference, mask):
    """Exact KL(ref || current) on the SAME conditional action supports.

    These are event-averaged conditional regularizers, not a claim of exact
    joint KL under Hungarian or the reference's own state distribution.
    """
    support = torch.isfinite(current)
    if not torch.equal(support[mask], torch.isfinite(reference)[mask]):
        raise ValueError('Reference/current conditional legal masks differ.')
    cur = torch.where(support, current, torch.zeros_like(current))
    ref = torch.where(support, reference, torch.zeros_like(reference))
    pc = torch.where(support, current.exp(), torch.zeros_like(current))
    pr = torch.where(support, reference.exp(), torch.zeros_like(reference))
    if mask.any():
        for p in (pc, pr):
            if not torch.allclose(p.sum(-1)[mask], torch.ones_like(p.sum(-1)[mask]), atol=1e-5, rtol=0):
                raise ValueError('Conditional resource distribution is not normalized.')
    count = mask.sum(-1).clamp_min(1)
    kl = (((pr * (ref - cur)).sum(-1) * mask).sum(-1) / count).sum()
    entropy = (((-pc * cur).sum(-1) * mask).sum(-1) / count).sum()
    require_finite(kl, 'reference conditional KL')
    require_finite(entropy, 'conditional entropy')
    return kl, entropy


def returns_from_complete(cmax, times, completed, scale=10000.0):
    cmax, times = np.asarray(cmax, np.float64), np.asarray(times, np.float64)
    if (not np.asarray(completed, bool).all() or not np.isfinite(cmax).all()
            or not np.isfinite(times).all() or np.any(cmax <= 0)
            or not math.isfinite(scale) or scale <= 0
            or np.any(times < 0) or np.any(times > cmax + 1e-3)):
        raise ValueError('Returns require finite complete episodes and valid physical times.')
    return (-(cmax[None, :] - times) / scale).astype(np.float32)


class ResourceLearner:
    def __init__(self, policy, args, contract, progress):
        self.policy, self.args, self.contract, self.progress = policy, args, contract, progress
        self.device = policy.device
        self.actor_params, self.critic_params = configure_trainability(policy)
        self.protected = summary(policy.ac, protected=True)
        self.source_actor = summary(policy.ac, RESOURCE_PREFIXES)
        self.actor_optim = torch.optim.Adam(self.actor_params, lr=contract['actor_lr'])
        self.critic_optim = torch.optim.Adam(self.critic_params, lr=contract['critic_lr'])
        self.actor_steps = self.critic_steps = 0
        self.probability_checks = dict(events=0, max_logp_error=0.0, max_ratio_error=0.0,
                                      masks_equal=True, actions_equal=True)

    def assert_protected(self):
        if summary(self.policy.ac, protected=True) != self.protected:
            raise RuntimeError('Stage2 resource learning changed protected tensors.')

    def collect_forward(self, *args, **kwargs):
        return forward(*args, **kwargs)

    def cache_encoding(self, encoded):
        cached = {k: v.detach().cpu() for k, v in encoded.items()}
        return cached, sum(v.numel() * v.element_size() for v in cached.values())

    def collect(self, envs, *, training, bc=False, hungarian=False, audit=False,
                capture_values=False, trace_indices=()):
        policy, args = self.policy, self.args
        policy.ac.eval()
        if policy.bc_reference_ac is not None:
            policy.bc_reference_ac.eval()
        n, m = len(envs.ps), args.max_agent_num + args.max_device_num
        hidden = np.zeros((n, m, args.recurrent_N, args.hidden_size), np.float32)
        ref_hidden = hidden.copy()
        previous = np.full((n, m, 2), -1, np.int64)
        envs.call('reset_data_cursor')
        obs, dones, infos = envs.reset()
        frames, completed_steps = [], np.zeros(n, np.int64)
        value_frames, trace_rows = [], {int(i): [] for i in trace_indices}
        cache_enabled = bool(training and self.contract.get('cache_frozen_encoding', False))
        cache_bytes = 0
        cache_limit = int(self.contract.get('encoding_cache_max_bytes', 8 * 1024 ** 3))
        completed_cases = np.zeros(n, bool)
        teacher_queries, environment_events = 0, 0
        action_digest = hashlib.sha256()
        started = time.monotonic()
        for step in range(self.contract['rollout_max_steps']):
            active = np.asarray(infos['active_agents'], np.float32).reshape(n, m, 1)
            active[completed_cases] = 0
            history = HKBZ_Runner._authoritative_policy_history(infos, previous, args.max_agent_num)
            types = np.asarray(infos['agent_types']).copy()
            times = np.asarray(infos['env_total_time'], np.float64).reshape(n)
            with torch.no_grad():
                out, encoded = self.collect_forward(policy, obs, hidden, active, history, types,
                    deterministic=not training, hungarian=hungarian)
                actions = out['actions'].cpu().numpy()
                action_digest.update(actions.tobytes())
                mask = resource_mask(out['decision_mask'], args.max_agent_num)
                if ((training and step < 8) or (audit and step < 64)) and not hungarian:
                    replay, _ = self.collect_forward(policy, obs, hidden, active, history, types,
                        actions=actions, encoded=encoded)
                    error = float((out['log_probs'] - replay['log_probs']).abs().max())
                    ratio_error = float((((out['log_probs'] - replay['log_probs']) * mask).sum(-1).exp() - 1).abs().max())
                    if (error > 1e-6 or ratio_error > 1e-5
                            or not torch.equal(out['decision_mask'], replay['decision_mask'])
                            or not torch.equal(out['actions'], replay['actions'])
                            or not torch.equal(torch.isfinite(out['resource_logits']),
                                               torch.isfinite(replay['resource_logits']))):
                        raise RuntimeError(f'Zero-update probability/action/mask mismatch: {error}, {ratio_error}')
                    self.probability_checks['events'] += 1
                    self.probability_checks['max_logp_error'] = max(error, self.probability_checks['max_logp_error'])
                    self.probability_checks['max_ratio_error'] = max(ratio_error, self.probability_checks['max_ratio_error'])
                reference_logits = None
                if training and not bc:
                    ref, _ = self.collect_forward(policy, obs, ref_hidden, active, history, types,
                        actions=actions, reference=True, encoded=encoded)
                    ref_hidden = ref['rnn_states'].cpu().numpy()
                    reference_logits = ref['resource_logits'].cpu()
            teacher_actions = None
            if training and bc:
                labels = envs.call('resource_iga_teacher_actions', return_info=True,
                                  include_ready_targets=False, deployment_projection=True)
                teacher_actions = actions.copy()
                for i, label in enumerate(labels):
                    if mask[i].any() and label['info'].get('available') is not True:
                        raise RuntimeError('Missing live-state BC teacher.')
                    teacher_actions[i, args.max_agent_num:, :2] = np.asarray(label['actions'])[args.max_agent_num:, :2]
                teacher_queries += int(mask.any(-1).sum())
            if trace_rows:
                from onpolicy.utils.stage2_resource_rl_v4 import trace_event
                for i in trace_rows:
                    if not completed_cases[i]:
                        trace_rows[i].append(trace_event(obs[i], actions[i], active[i],
                            out['resource_logits'][i].cpu(), types[i], float(times[i]),
                            step, args.max_agent_num))
            if capture_values:
                value_frames.append((out['global_emb'].cpu(), mask.any(-1).cpu(), times.copy()))
            next_obs, _, dones, next_info = envs.step(actions)
            if training:
                frames.append(dict(obs=list(obs), active=active.copy(), history=history.copy(),
                    types=types, actions=actions.copy(), teacher_actions=teacher_actions,
                    old_logp=out['log_probs'].cpu(), mask=mask.cpu(),
                    reference_logits=reference_logits, global_emb=out['global_emb'].cpu(),
                    times=times, dones=np.asarray(dones).copy()))
                if cache_enabled:
                    cached, added_bytes = self.cache_encoding(encoded)
                    cache_bytes += added_bytes
                    if cache_bytes > cache_limit:
                        # Performance-only fallback, never a change in dtype or policy.
                        cache_enabled = False
                        for stored in frames:
                            stored.pop('encoded', None)
                        self.progress('encoding_cache_disabled_at_memory_cap', force=True,
                                      encoding_cache_bytes=cache_bytes, encoding_cache_limit=cache_limit)
                    else:
                        frames[-1]['encoded'] = cached
            environment_events += int((~completed_cases).sum())
            done_case = np.asarray(dones).reshape(n, -1).all(-1)
            completed_steps[~completed_cases] += 1
            completed_cases |= done_case
            hidden = out['rnn_states'].cpu().numpy()
            hidden[np.asarray(dones)] = 0
            ref_hidden[np.asarray(dones)] = 0
            obs, infos, previous = next_obs, next_info, actions
            self.progress('collect', event_step=step + 1, completed_cases=int(completed_cases.sum()),
                          total_cases=n, environment_events=environment_events)
            if completed_cases.all():
                break
        statuses = envs.call('stage2_cost_live_status')  # Read-only live status, NOT a cost branch.
        objectives = envs.call('get_training_objective')
        cmax = [row['makespan'] for row in statuses]
        complete = [bool(done and row['completed'] and not obj['cycle_terminated'])
                    for done, row, obj in zip(completed_cases, statuses, objectives)]
        rows = [dict(case_dir=Path(row['case']).name, makespan=row['makespan'],
                     completed=ok, cycle_terminated=bool(obj['cycle_terminated']),
                     timeout=not bool(done), steps=int(steps))
                for row, obj, ok, done, steps in zip(statuses, objectives, complete, completed_cases, completed_steps)]
        if not all(complete):
            self.progress('incomplete_rollout', force=True, cases=rows)
            raise RuntimeError('Incomplete/cycle/timeout rollout; never train partial Cmax.')
        for value, obj in zip(cmax, objectives):
            if obj['team_return'] != -value:
                raise ValueError('Unexpected auxiliary reward in the resource-RL objective.')
        targets = None if not training else returns_from_complete(cmax,
            [f['times'] for f in frames], complete, self.contract['return_scale'])
        result = dict(frames=frames, targets=targets, cases=rows,
                    teacher_queries=teacher_queries, cost_queries=0,
                    environment_events=environment_events, seconds=time.monotonic() - started,
                    trajectory_action_sha256=action_digest.hexdigest())
        if capture_values:
            value_valid = torch.stack([f[1] for f in value_frames])
            value_targets = torch.as_tensor(returns_from_complete(cmax,
                [f[2] for f in value_frames], complete, self.contract['return_scale']))
            result['value_data'] = dict(
                embeddings=torch.stack([f[0] for f in value_frames])[value_valid],
                targets=value_targets[value_valid],
                case_indices=torch.arange(n).expand(len(value_frames), n)[value_valid])
        if trace_rows:
            result['traces'] = {rows[i]['case_dir']: values for i, values in trace_rows.items()}
        if self.contract.get('cache_frozen_encoding', False) and training:
            result['encoding_cache'] = dict(enabled=cache_enabled, bytes_attempted=cache_bytes,
                                            limit_bytes=cache_limit, dtype_unchanged=True)
        return result

    def _critic_batch(self, rollout):
        masks = torch.stack([f['mask'].any(-1) for f in rollout['frames']])
        embeddings = torch.stack([f['global_emb'] for f in rollout['frames']])[masks].to(self.device)
        targets = torch.as_tensor(rollout['targets'])[masks].to(self.device)
        if targets.numel() == 0:
            raise ValueError('No non-forced resource decisions in collection.')
        return embeddings, targets, masks

    def critic_update(self, embeddings, targets, steps):
        actor_before = summary(self.policy.ac, RESOURCE_PREFIXES)
        losses = []
        for _ in range(steps):
            self.critic_optim.zero_grad(set_to_none=True)
            # Only the small team MLP sees all events; no actor/GNN replay here.
            pred = self.policy.ac.team_critic(embeddings.detach()).flatten()
            loss = (pred - targets).square().mean()
            require_finite(loss, 'critic loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.critic_params, self.contract['max_grad_norm'])
            require_finite(norm, 'critic gradient')
            self.critic_optim.step()
            self.critic_steps += 1
            losses.append(float(loss.detach()))
        if summary(self.policy.ac, RESOURCE_PREFIXES) != actor_before:
            raise RuntimeError('Critic-only update changed resource actor.')
        self.assert_protected()
        return losses

    def update(self, rollout, *, bc, warmup=False):
        c, policy = self.contract, self.policy
        frames = rollout['frames']
        n = len(rollout['cases'])
        m = self.args.max_agent_num + self.args.max_device_num
        valid = torch.stack([f['mask'].any(-1) for f in frames])
        denominator = int(valid.sum())
        if denominator <= 0:
            raise ValueError('No trainable resource joint events.')
        critic_losses = []
        critic_before = summary(policy.ac, CRITIC_PREFIXES)
        signal_metrics = dict(valid_joint_events=denominator, rollout_events=rollout['environment_events'],
                              trajectory_steps=len(frames), case_episodes=n)
        advantages = torch.zeros(valid.shape)
        if not bc:
            embeddings, targets, valid = self._critic_batch(rollout)
            if warmup:
                critic_losses = self.critic_update(embeddings, targets, c['critic_warmup_steps'])
            with torch.no_grad():
                values = policy.ac.team_critic(embeddings).flatten()
                advantage = targets - values
                std = advantage.std(unbiased=False).clamp_min(1e-6)
                target_variance = targets.var(unbiased=False)
                signal_metrics.update(return_mean=float(targets.mean()), return_std=float(targets.std(unbiased=False)),
                    value_mean=float(values.mean()), value_std=float(values.std(unbiased=False)),
                    raw_advantage_mean=float(advantage.mean()), raw_advantage_std=float(advantage.std(unbiased=False)),
                    explained_variance=(float(1 - advantage.var(unbiased=False) / target_variance)
                                        if float(target_variance) > 1e-12 else None))
                advantages[valid] = ((advantage - advantage.mean()) / std).cpu()
        learner_mode(policy.ac)
        records = []
        for pass_index in range(c['ppo_passes']):
            actor_before = summary(policy.ac, RESOURCE_PREFIXES)
            hidden = torch.zeros((n, m, self.args.recurrent_N, self.args.hidden_size), device=self.device)
            self.actor_optim.zero_grad(set_to_none=True)
            sums = dict(kl_sum=0.0, clip_count=0, joint_count=0,
                        policy_loss=0.0, reference_kl=0.0, entropy=0.0)
            chunk_loss = None
            for t, frame in enumerate(frames):
                out, _ = forward(policy, frame['obs'], hidden, frame['active'], frame['history'],
                    frame['types'], actions=frame['teacher_actions'] if bc else frame['actions'])
                mask = resource_mask(out['decision_mask'], self.args.max_agent_num)
                if bc:
                    # Exact conditional teacher NLL in the declared AR policy.
                    # No teacher execution, full-edge loss or cost preference.
                    joint_valid = mask.any(-1)
                    loss = -(out['log_probs'] * mask).sum(-1)[joint_valid].sum()
                    regularizer = loss.new_zeros(())
                    sums['joint_count'] += int(joint_valid.sum())
                else:
                    old_mask = frame['mask'].to(self.device)
                    if not torch.equal(mask, old_mask):
                        raise RuntimeError('PPO replay trainable resource mask changed.')
                    loss, metrics = joint_terms(out['log_probs'], frame['old_logp'].to(self.device),
                        mask, advantages[t].to(self.device), c['clip'])
                    for key, value in metrics.items():
                        sums[key] += value
                    kl, entropy = conditional_regularizers(out['resource_logits'],
                        frame['reference_logits'].to(self.device), mask)
                    regularizer = c['reference_kl_coef'] * kl - c['entropy_coef'] * entropy
                    sums['reference_kl'] += float(kl.detach())
                    sums['entropy'] += float(entropy.detach())
                require_finite(loss + regularizer, 'actor objective')
                sums['policy_loss'] += float(loss.detach())
                term = (loss + regularizer) / denominator
                chunk_loss = term if chunk_loss is None else chunk_loss + term
                done = torch.as_tensor(frame['dones'], device=self.device, dtype=torch.bool)
                hidden = out['rnn_states'] * (~done).unsqueeze(-1).unsqueeze(-1)
                if (t + 1) % c['chunk_length'] == 0 or t + 1 == len(frames):
                    if chunk_loss.requires_grad:
                        chunk_loss.backward()
                    chunk_loss = None
                    hidden = hidden.detach()
                    self.progress('actor_update', replay_step=t + 1, replay_total=len(frames),
                                  optimization_pass=pass_index + 1, actor_steps=self.actor_steps)
            kl_mean = sums['kl_sum'] / max(1, sums['joint_count'])
            skipped = not bc and kl_mean > c['target_kl']
            norm = torch.nn.utils.clip_grad_norm_(self.actor_params, c['max_grad_norm'])
            require_finite(norm, 'resource actor gradient')
            if not skipped:
                if float(norm) <= 0:
                    raise RuntimeError('No resource actor gradient; cannot claim learning.')
                self.actor_optim.step()
                self.actor_steps += 1
                if summary(policy.ac, RESOURCE_PREFIXES) == actor_before:
                    raise RuntimeError('Optimizer step did not change any resource actor tensor.')
            self.actor_optim.zero_grad(set_to_none=True)
            self.assert_protected()
            records.append(dict(pass_index=pass_index + 1, **sums, approx_kl=kl_mean,
                gradient_norm=float(norm), skipped_for_kl=skipped,
                kl_measurement='before_this_pass_optimizer_step', actor_steps=self.actor_steps))
            self.progress('actor_pass_completed', force=True, **records[-1])
            if skipped:
                break
        if not bc:
            critic_losses += self.critic_update(embeddings, targets, c['ppo_passes'])
        elif summary(policy.ac, CRITIC_PREFIXES) != critic_before:
            raise RuntimeError('BC control unexpectedly changed the team critic.')
        return dict(passes=records, critic_losses=critic_losses, actor_steps=self.actor_steps,
                    critic_steps=self.critic_steps, frozen_parameters_unchanged=True,
                    signal_metrics=signal_metrics,
                    recurrent_contract='current_weights_full_prefix_each_pass_chunked_gradient',
                    optimization_unit='one_step_per_full_collection_pass')
