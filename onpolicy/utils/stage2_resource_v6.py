"""Bounded V6 collection and offline BC. No automatic research promotion.

S1 replays complete current-weight device histories, TBPTT16, frozen plane/GNN.
S2 is stateless and batches teacher-forced conditional decisions directly.
Both optimize case-balanced and role-balanced CE on the SAME teacher episodes.
"""
from __future__ import annotations

import copy
import hashlib
import random
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from onpolicy.algorithms.utils.resource_actor_v6 import pair_inputs, resource_context
from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
from onpolicy.utils.stage2_resource_rl import (
    RESOURCE_PREFIXES, forward, learner_mode, require_finite, resource_mask, summary,
)
from onpolicy.utils.stage2_resource_rl_v4 import CaseMinibatchLearner
from onpolicy.utils.stage2_resource_rl_v4_fast import prepared_forward

PROTOCOL = 'stage2_semantic_pair_v6_pilot_v1'
ARMS = ('S1', 'S2')
DEFAULTS = dict(actor_lr=3e-5, critic_lr=1e-4, clip=.1, ppo_passes=2,
    entropy_coef=.001, reference_kl_coef=.01, target_kl=.01, return_scale=10000.,
    gamma=1., chunk_length=16, max_grad_norm=1., seed=11, rounds=10,
    rollout_max_steps=4000, minibatch_cases=6, evaluation_workers=12,
    bc_epochs=10, pilot_train_cases=48, pilot_dev_cases=12,
    evaluation_epochs=[1, 5, 10], reference_diagnostic_cases=4,
    supervised_microbatch=256, cache_bytes=4 * 1024**3)


def actor_prefixes(arm):
    return ('resource_v6.',) + (RESOURCE_PREFIXES if arm == 'S1' else ())


def protected_summary(ac, arm):
    state = {k: v for k, v in ac.state_dict().items()
             if not k.startswith(actor_prefixes(arm))}
    return summary(state)


def semantic_labels(obs, labels, plane_count):
    """Bind indexed teacher actions to identities and validate prefix supports."""
    bound = []
    for graph, label in zip(obs, labels):
        rows = []
        for decision in label['info'].get('decisions', []):
            agent = int(decision['agent_id'])
            req = int(decision['selected_request_id'])
            identity = graph.v6_request_keys[0, req].tolist() if req else None
            if ('selected_request_semantic_key' not in decision or
                    identity != decision['selected_request_semantic_key']):
                raise RuntimeError('Teacher live request identity differs from the saved observation.')
            rows.append(dict(agent=agent, role=decision['role'], request=req,
                identity=identity,
                legal=list(decision['legal_request_ids']), allow_noop=bool(decision['allow_noop']),
                reason=decision.get('reason'), noop_cause=decision.get('noop_cause'),
                score_margin=decision.get('selected_score_margin')))
        bound.append(dict(teacher_path=label['info'].get('teacher_path'),
            teacher_sha256=label['info'].get('teacher_sha256'), decisions=rows,
            projection_events=label['info'].get('teacher_projection_events', 0)))
    return bound


class V6Learner:
    # Retain the established six-case FP32 behavior grouping.
    collect_forward = CaseMinibatchLearner.collect_forward
    cache_encoding = CaseMinibatchLearner.cache_encoding

    def __init__(self, policy, args, contract, progress, arm=None):
        self.policy, self.args, self.contract, self.progress = policy, args, contract, progress
        self.arm, self.device = arm, policy.device
        self.actor_steps = 0
        self.checks = dict(zero_update_events=0, max_logp_error=0., max_ratio_error=0.,
            teacher_decisions=0, semantic_history_slots=0, source='current_runtime')
        self._prepared = {}
        self._prepared_bytes = 0
        self.actor_params = []
        for name, p in policy.ac.named_parameters():
            trainable = arm is not None and name.startswith(actor_prefixes(arm))
            p.requires_grad_(trainable)
            if trainable:
                self.actor_params.append(p)
        self.protected = protected_summary(policy.ac, arm) if arm else summary(policy.ac)
        self.actor_optim = torch.optim.Adam(self.actor_params, lr=contract['actor_lr']) if arm else None

    def _hidden(self, n):
        return torch.zeros((n, self.args.max_agent_num + self.args.max_device_num,
            self.args.recurrent_N, self.args.hidden_size), device=self.device)

    def assert_protected(self):
        actual = protected_summary(self.policy.ac, self.arm) if self.arm else summary(self.policy.ac)
        if actual != self.protected:
            raise RuntimeError('V6 learning changed frozen plane/GNN/Ready/critic tensors.')

    def collect(self, envs, *, store=False, teacher=False, archived=None, projected=True, hungarian=False):
        n, m, p = len(envs.ps), self.args.max_agent_num + self.args.max_device_num, self.args.max_agent_num
        hidden = np.zeros((n, m, self.args.recurrent_N, self.args.hidden_size), np.float32)
        previous = np.full((n, m, 2), -1, np.int64)
        complete = np.zeros(n, bool)
        steps = np.zeros(n, np.int64)
        envs.call('reset_data_cursor')
        obs, dones, info = envs.reset()
        frames, labels_report = [], []
        digest = hashlib.sha256()
        started, events = time.monotonic(), 0
        self.policy.ac.eval()
        for step in range(self.contract['rollout_max_steps']):
            active = np.asarray(info['active_agents'], np.float32).reshape(n, m, 1).copy()
            active[complete] = 0
            history = HKBZ_Runner._authoritative_policy_history(info, previous, p)
            types = np.asarray(info['agent_types']).copy()
            bound = None
            with torch.no_grad():
                # The initial proposal also computes the unchanged frozen-plane action.
                out, encoded = self.collect_forward(self.policy, obs, hidden, active, history, types,
                    deterministic=True, hungarian=hungarian)
                actions = out['actions'].cpu().numpy()
                if teacher:
                    if archived:
                        labels = envs.call('stage2_v6_reference_actions', archived['teacher_dir'],
                            archived['teacher_hashes'], deployment_projection=projected)
                    else:
                        labels = envs.call('resource_iga_teacher_actions', return_info=True,
                            include_ready_targets=False, deployment_projection=projected)
                    for i, label in enumerate(labels):
                        if not complete[i] and label['info'].get('available') is not True:
                            raise RuntimeError('Missing current-case teacher; no fallback or label fabrication.')
                        selected = np.asarray(label['actions'])
                        participating = active[i, p:, 0].astype(bool)
                        actions[i, p:, :2][participating] = selected[p:, :2][participating]
                    # Teacher-forced AR replay gives the exact executed prefix and hidden update.
                    out, _ = self.collect_forward(self.policy, obs, hidden, active, history, types,
                        actions=actions, encoded=encoded)
                    bound = semantic_labels(obs, labels, p)
                    for i, label in enumerate(bound):
                        for row in label['decisions']:
                            if not active[i, row['agent'], 0]:
                                continue
                            legal = set(row['legal']) | ({0} if row['allow_noop'] else set())
                            support = set(torch.isfinite(out['resource_logits'][i, row['agent']]).nonzero().flatten().cpu().tolist())
                            if legal != support or row['request'] not in support:
                                raise RuntimeError('Live teacher mask differs from the production AR prefix.')
                            self.checks['teacher_decisions'] += 1
                    labels_report.append(bound)
                mask = resource_mask(out['decision_mask'], p)
                if step < 8 and not hungarian:
                    replay, _ = self.collect_forward(self.policy, obs, hidden, active, history, types,
                        actions=actions, encoded=encoded)
                    delta = out['log_probs'] - replay['log_probs']
                    error, ratio = float(delta.abs().max()), float(((delta * mask).sum(-1).exp() - 1).abs().max())
                    if (error > 1e-6 or ratio > 1e-5 or not torch.equal(out['decision_mask'], replay['decision_mask'])
                            or not torch.equal(out['actions'], replay['actions'])):
                        raise RuntimeError(f'V6 zero-update replay failed: {error}, {ratio}')
                    self.checks['zero_update_events'] += int(mask.any(-1).sum())
                    self.checks['max_logp_error'] = max(error, self.checks['max_logp_error'])
                    self.checks['max_ratio_error'] = max(ratio, self.checks['max_ratio_error'])
            for graph in obs:
                self.checks['semantic_history_slots'] += int(graph.v6_history[0, :, -2].sum())
            digest.update(actions.tobytes())
            next_obs, _, dones, next_info = envs.step(actions)
            if store:
                cached = {'case_group_encodings': {key: {name: v.detach().cpu() for name, v in enc.items()}
                    for key, enc in encoded['case_group_encodings'].items()}}
                frames.append(dict(obs=list(obs), active=active, history=history.copy(), types=types,
                    actions=actions.copy(), teacher_actions=actions.copy(), encoded=cached,
                    teacher_labels=bound, mask=mask.cpu(), legal=torch.isfinite(out['resource_logits']).cpu(),
                    old_logp=out['log_probs'].cpu(), dones=np.asarray(dones).copy(),
                    times=np.asarray(info['env_total_time'], np.float64).reshape(n)))
            events += int((~complete).sum())
            steps[~complete] += 1
            complete |= np.asarray(dones).reshape(n, -1).all(-1)
            hidden = out['rnn_states'].cpu().numpy()
            hidden[np.asarray(dones)] = 0
            previous, obs, info = actions, next_obs, next_info
            self.progress('collect', event_step=step + 1, completed_cases=int(complete.sum()),
                total_cases=n, teacher_execution=teacher, environment_events=events)
            if complete.all():
                break
        statuses, objectives = envs.call('stage2_cost_live_status'), envs.call('get_training_objective')
        rows = []
        for idx, (status, objective) in enumerate(zip(statuses, objectives)):
            ok = bool(complete[idx] and status['completed'] and not objective['cycle_terminated'])
            if not ok or not np.isfinite(status['makespan']) or objective['team_return'] != -status['makespan']:
                raise RuntimeError('Incomplete/invalid trajectory; partial Cmax cannot be trained or selected.')
            rows.append(dict(case_dir=status['case'].rstrip('/').split('/')[-1], makespan=status['makespan'],
                steps=int(steps[idx]), completed=True, cycle_terminated=False, timeout=False))
        return dict(frames=frames, cases=rows, seconds=time.monotonic() - started,
            environment_events=events, teacher_execution=teacher, teacher_projected=projected if teacher else None,
            trajectory_action_sha256=digest.hexdigest(), label_audit=labels_report,
            checks=dict(self.checks), cost_label_queries=0, search_calls=0)

    def clear_cache(self):
        self._prepared.clear()
        self._prepared_bytes = 0

    def entry(self, frame):
        key = id(frame)
        if key not in self._prepared:
            graph = Batch.from_data_list(frame['obs'])
            data, info = self.policy._build_inputs(graph, self._hidden(len(frame['obs'])), frame['active'],
                frame['history'][..., 0], frame['history'][..., 1], agent_types=frame['types'])
            data.pop('hidden_states')
            group = tuple(range(len(frame['obs'])))
            encoded = {k: v.to(self.device) for k, v in frame['encoded']['case_group_encodings'][group].items()}
            result = dict(data=data, info=info, encoded=encoded,
                actions=torch.as_tensor(frame['actions'], device=self.device),
                legal=frame['legal'].to(self.device), mask=frame['mask'].to(self.device),
                dones=torch.as_tensor(frame['dones'], device=self.device))
            from onpolicy.utils.stage2_resource_rl_v4_fast import tensor_bytes
            size = tensor_bytes(result)
            if self._prepared_bytes + size <= self.contract['cache_bytes']:
                self._prepared[key] = result
                self._prepared_bytes += size
            return result
        return self._prepared[key]

    def case_role_weights(self, rollout):
        counts = torch.zeros((len(rollout['cases']), 2), dtype=torch.float64)
        for f in rollout['frames']:
            types = torch.as_tensor(f['types']).reshape(len(counts), -1)
            for role in range(2):
                counts[:, role] += (f['mask'] & (types == role + 1)).sum(-1)
        if not bool((counts.sum(-1) > 0).all()):
            raise ValueError('Every supervised case must contain a free resource decision.')
        present_roles = (counts > 0).sum(-1).clamp_min(1)
        return torch.where(counts > 0, 1. / counts.clamp_min(1) / present_roles[:, None] / len(counts),
                           torch.zeros_like(counts)).to(self.device, torch.float32)

    def bc_epoch_batch(self, rollout):
        """One update per six complete cases; no optimizer step within a case."""
        if self.arm not in ARMS or not rollout['teacher_execution']:
            raise ValueError('V6 offline BC needs the pinned shared teacher dataset.')
        if self.arm == 'S2':
            return self.bc_stateless_batch(rollout if 'stateless_records' in rollout else self.pack_stateless(rollout))
        self.clear_cache()
        learner_mode(self.policy.ac)
        before = summary(self.policy.ac, actor_prefixes(self.arm))
        self.actor_optim.zero_grad(set_to_none=True)
        weights = self.case_role_weights(rollout)
        hidden = self._hidden(len(rollout['cases']))
        chunk_loss, total, correct, count = None, 0., 0, 0
        by_role = {r: dict(count=0, correct=0, noop_count=0, noop_correct=0) for r in ('ordinary', 'transporter')}
        for step, frame in enumerate(rollout['frames']):
            e = self.entry(frame)
            types = torch.as_tensor(frame['types'], device=self.device).reshape(len(rollout['cases']), -1)
            if self.arm == 'S1':
                out, _ = prepared_forward(self.policy, e['data'], e['info'], hidden,
                    actions=e['actions'], encoded=e['encoded'])
                logits = out['resource_logits']
                hidden = out['rnn_states'] * (~e['dones']).unsqueeze(-1).unsqueeze(-1)
                if not torch.equal(torch.isfinite(logits)[e['mask']], e['legal'][e['mask']]):
                    raise RuntimeError('BC replay changed teacher conditional support.')
                parts = []
                for ri, role in enumerate(('ordinary', 'transporter')):
                    selected = e['mask'] & (types == ri + 1)
                    rows, agents = selected.nonzero(as_tuple=True)
                    if len(rows):
                        dist = logits[rows, agents]
                        target = e['actions'][rows, agents, 0]
                        parts.append(-(dist.gather(1, target[:, None]).squeeze(1) * weights[rows, ri]).sum())
                        hits = dist.argmax(-1) == target
                        self._stats(by_role[role], hits, target)
                        correct += int(hits.sum()); count += len(rows)
                loss = sum(parts) if parts else None
                if loss is not None:
                    total += float(loss.detach())
                    chunk_loss = loss if chunk_loss is None else chunk_loss + loss
                if (step + 1) % self.contract['chunk_length'] == 0 or step + 1 == len(rollout['frames']):
                    if chunk_loss is not None:
                        require_finite(chunk_loss, 'V6 BC loss')
                        chunk_loss.backward()
                    hidden, chunk_loss = hidden.detach(), None
            else:
                context = resource_context(e['data']['graph'])
                # Stateless frames need no recurrent graph and may be differentiated immediately.
                for agent in e['mask'].any(0).nonzero().flatten().tolist():
                    device_idx = agent - self.args.max_agent_num
                    for ri, role in enumerate(('ordinary', 'transporter')):
                        rows = (e['mask'][:, agent] & (types[:, agent] == ri + 1)).nonzero().flatten()
                        if not len(rows):
                            continue
                        features = pair_inputs(context, rows, device_idx, e['legal'][rows, agent])
                        dist = self.policy.ac.resource_v6.distribution(features, e['legal'][rows, agent], role, self.policy.ac.tau)
                        target = e['actions'][rows, agent, 0]
                        loss = -(dist.gather(1, target[:, None]).squeeze(1) * weights[rows, ri]).sum()
                        require_finite(loss, 'V6 feed-forward BC loss')
                        loss.backward()
                        total += float(loss.detach())
                        hits = dist.argmax(-1) == target
                        self._stats(by_role[role], hits, target)
                        correct += int(hits.sum()); count += len(rows)
            self.progress('bc_replay', arm=self.arm, replay_step=step + 1,
                replay_total=len(rollout['frames']), actor_steps=self.actor_steps)
        norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.contract['max_grad_norm'])
        require_finite(norm, 'V6 actor gradient')
        if float(norm) <= 0:
            raise RuntimeError('No actual V6 actor gradient.')
        self.actor_optim.step()
        self.actor_optim.zero_grad(set_to_none=True)
        self.actor_steps += 1
        self.assert_protected()
        if before == summary(self.policy.ac, actor_prefixes(self.arm)):
            raise RuntimeError('V6 optimizer did not change the selected actor.')
        self.clear_cache()
        return dict(loss=total, accuracy=correct / max(1, count), free_decisions=count,
            role_metrics=by_role, gradient_norm=float(norm), actor_steps=self.actor_steps,
            loss_weighting='equal_case_then_equal_present_role_then_free_decision',
            semantic_history=True, frozen_plane_and_encoder_unchanged=True)

    def pack_stateless(self, rollout):
        """Raw observable inputs only: valid across epochs, never learned activations."""
        weights = self.case_role_weights(rollout).cpu()
        pieces = {role: {k: [] for k in ('features', 'legal', 'target', 'weight')}
                  for role in ('ordinary', 'transporter')}
        for step, frame in enumerate(rollout['frames']):
            context = resource_context(Batch.from_data_list(frame['obs']))
            types = torch.as_tensor(frame['types']).reshape(len(rollout['cases']), -1)
            actions = torch.as_tensor(frame['actions'])
            for ri, role in enumerate(pieces):
                rows, agents = (frame['mask'] & (types == ri + 1)).nonzero(as_tuple=True)
                if not len(rows):
                    continue
                legal = frame['legal'][rows, agents]
                data = dict(features=pair_inputs(context, rows, agents - self.args.max_agent_num, legal),
                    legal=legal, target=actions[rows, agents, 0], weight=weights[rows, ri])
                for key, value in data.items():
                    pieces[role][key].append(value)
            self.progress('pack_stateless_teacher_inputs', replay_step=step + 1, replay_total=len(rollout['frames']))
        records = {role: {k: torch.cat(v) for k, v in values.items()}
                   for role, values in pieces.items() if values['target']}
        mass = sum(float(row['weight'].sum()) for row in records.values())
        if abs(mass - 1.) > 1e-5:
            raise ValueError(f'Stateless BC changed case/role normalization: {mass}')
        return dict(stateless_records=records, teacher_execution=True, cases=rollout['cases'],
            source='raw_observations_and_teacher_prefix_masks', learned_embeddings_cached=False,
            case_role_weight_mass=mass)

    def bc_stateless_batch(self, rollout):
        self.policy.ac.eval()
        before = summary(self.policy.ac, actor_prefixes(self.arm))
        self.actor_optim.zero_grad(set_to_none=True)
        total, count, correct = 0., 0, 0
        stats = {r: dict(count=0, correct=0, noop_count=0, noop_correct=0) for r in ('ordinary', 'transporter')}
        micro = self.contract['supervised_microbatch']
        for role, record in rollout['stateless_records'].items():
            for start in range(0, len(record['target']), micro):
                batch = {k: v[start:start + micro].to(self.device) for k, v in record.items()}
                dist = self.policy.ac.resource_v6.distribution(batch['features'], batch['legal'], role, self.policy.ac.tau)
                loss = -(dist.gather(1, batch['target'][:, None]).squeeze(1) * batch['weight']).sum()
                require_finite(loss, 'stateless BC loss')
                loss.backward()
                hits = dist.argmax(-1) == batch['target']
                self._stats(stats[role], hits, batch['target'])
                total += float(loss.detach()); count += len(hits); correct += int(hits.sum())
        norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.contract['max_grad_norm'])
        require_finite(norm, 'stateless actor gradient')
        if float(norm) <= 0 or count == 0:
            raise RuntimeError('No stateless BC update signal.')
        self.actor_optim.step()
        self.actor_optim.zero_grad(set_to_none=True)
        self.actor_steps += 1
        self.assert_protected()
        if before == summary(self.policy.ac, actor_prefixes(self.arm)):
            raise RuntimeError('Stateless optimizer changed no actor tensors.')
        return dict(loss=total, accuracy=correct / count, free_decisions=count, role_metrics=stats,
            gradient_norm=float(norm), actor_steps=self.actor_steps,
            loss_weighting='equal_case_then_equal_present_role_then_free_decision',
            stateless_vectorized=True, microbatch_records=micro, optimizer_steps_this_batch=1,
            frozen_plane_and_encoder_unchanged=True, learned_embeddings_cached=False)

    @staticmethod
    def _stats(stats, hits, target):
        stats['count'] += len(target)
        stats['correct'] += int(hits.sum())
        noop = target == 0
        stats['noop_count'] += int(noop.sum())
        stats['noop_correct'] += int((hits & noop).sum())

    def checkpoint_state(self):
        self.assert_protected()
        return dict(protocol=PROTOCOL, arm=self.arm, model=copy.deepcopy(self.policy.ac.state_dict()),
            actor_optimizer=copy.deepcopy(self.actor_optim.state_dict()), actor_steps=self.actor_steps,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            numpy_rng=copy.deepcopy(np.random.get_state()), python_rng=random.getstate(), protected=self.protected,
            checks=dict(self.checks), scientific_gate_passed=False,
            stage3_requires_fresh_critic_and_optimizers=True, deployment_decoder='autoregressive',
            automatic_promotion=False, stage3_started=False)
