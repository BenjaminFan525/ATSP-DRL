"""V5: independent graph views, matched case-minibatch PPO-V, bounded critic.

Only the immutable plane encoding may be reused by the V4 execution cache.
Resource views are recomputed by the actor on EVERY forward; value views are
recomputed here on EVERY critic microbatch. No learned hidden-state caching.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch.utils.checkpoint import checkpoint

from onpolicy.utils.stage2_resource_rl import (
    RESOURCE_PREFIXES, learner_mode, require_finite, resource_mask, summary,
)
from onpolicy.utils.stage2_resource_rl_v4 import DEFAULTS as V4_DEFAULTS
from onpolicy.utils.stage2_resource_rl_v4_fast import FastCaseMinibatchLearner
from onpolicy.utils.stage2_resource_rl_v5_zero import initial_reference_kernel_context

PROTOCOL = 'stage2_split_encoder_ppo_v5'
ARMS = ('E0', 'E1', 'E2', 'E3')
ACTOR = RESOURCE_PREFIXES + ('resource_encoder.',)
VALUE = ('team_critic.', 'value_encoder.')
DEFAULTS = {**V4_DEFAULTS, 'baseline': 'value', 'rounds': 10, 'epochs': 1,
    'evaluation_rounds': [5, 10], 'resource_encoder_lr': 3e-6,
    'value_encoder_lr': 1e-5, 'critic_batch_states': 48,
    'critic_microbatch_states': 6, 'reference_states_per_case': 32,
    'critic_warmup_seconds': 86400, 'critic_round_seconds': 86400,
    'resource_encoder_activation_checkpoint': True,
    'value_encoder_activation_checkpoint': True,
    'replay_cache_max_bytes': 4 * 1024**3}


def protected_summary(ac):
    return summary({k: v for k, v in ac.state_dict().items() if not k.startswith(ACTOR + VALUE)})


def install_encoders(ac, arm):
    if arm not in ARMS or hasattr(ac, 'split_encoder_arm'):
        raise ValueError('Unknown arm or duplicate encoder installation.')
    if (ac.request_ready_policy_injection != 'none' or ac.device_policy_head_mode != 'shared'
            or ac.resource_residual_adapter is not None):
        raise ValueError('V5 requires the immutable B0 architecture.')
    ac.split_encoder_arm = arm
    if arm in ('E1', 'E3'):
        ac.resource_encoder = copy.deepcopy(ac.encoder)
    if arm in ('E2', 'E3'):
        ac.value_encoder = copy.deepcopy(ac.encoder)
    ac.resource_encoder_activation_checkpoint = True
    # Each encoder is a real graph encoder copy, never a residual MLP adapter.
    original = summary(ac.encoder)
    for name in ('resource_encoder', 'value_encoder'):
        if hasattr(ac, name) and summary(getattr(ac, name)) != original:
            raise RuntimeError('Independent encoder was not initialized from B0.')


def assert_no_alias(ac, reference=None):
    # cuDNN flattens one GRU's *disjoint* parameters into a single allocation.
    # Allocation identity is therefore not parameter aliasing. Compare actual
    # occupied byte intervals, including storage offsets, on each device.
    seen = {}
    count = 0
    for label, model in (('current', ac), ('reference', reference)):
        if model is None:
            continue
        for name, tensor in model.state_dict().items():
            if tensor.numel() == 0:
                continue
            device = str(tensor.device)
            begin = tensor.untyped_storage().data_ptr() + tensor.storage_offset() * tensor.element_size()
            span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
            end = begin + span * tensor.element_size()
            for previous_begin, previous_end, previous_name in seen.setdefault(device, []):
                if begin < previous_end and previous_begin < end:
                    raise RuntimeError(f'Overlapping tensor memory: {previous_name} / {label}.{name}')
            seen[device].append((begin, end, label + '.' + name))
            count += 1
    return count


def architecture_manifest(ac, *, source_encoder_sha=None):
    arm = ac.split_encoder_arm
    return dict(protocol=PROTOCOL, arm=arm, encoder_roles={
        'plane': 'encoder', 'ready': 'encoder', 'B0_reference': 'reference.encoder',
        'ordinary_resource': 'resource_encoder' if arm in ('E1', 'E3') else 'encoder',
        'R014': 'resource_encoder' if arm in ('E1', 'E3') else 'encoder',
        'value': 'value_encoder' if arm in ('E2', 'E3') else 'encoder'},
        encoders={name: dict(parameters=sum(p.numel() for p in getattr(ac, name).parameters()),
            hash=summary(getattr(ac, name)), trainable=any(p.requires_grad for p in getattr(ac, name).parameters()))
            for name in ('encoder', 'resource_encoder', 'value_encoder') if hasattr(ac, name)},
        trainable_names=[n for n, p in ac.named_parameters() if p.requires_grad],
        actor_prefixes=list(ACTOR), value_prefixes=list(VALUE),
        initial_encoder_hash=source_encoder_sha or summary(ac.encoder),
        original_encoder_frozen=True, ready_policy_injection='none',
        activation_checkpoint=True, stage3_requires_split_architecture=True,
        stage3_requires_fresh_critic_and_optimizers=True)


def clean_cases(rows, records):
    """Never inherit old runtime diagnostics by a baseline/current dict merge."""
    by_name = {r['case_dir']: r for r in records}
    if len(by_name) != len(records) or len(rows) != len(records) or {r['case_dir'] for r in rows} != set(by_name):
        raise ValueError('Current case coverage differs from manifest.')
    result = []
    for row in rows:
        old = by_name[row['case_dir']]
        if row.get('case_sha256', old['case_sha256']) != old['case_sha256']:
            raise ValueError('Runtime/baseline content hash mismatch.')
        runtime = {k: v for k, v in row.items() if k not in ('case_dir', 'case_sha256', 'distribution')}
        baseline = {k: v for k, v in old.items() if k not in ('case_dir', 'case_sha256', 'distribution')}
        comparison = {}
        if old.get('makespan') is not None:
            comparison['makespan_delta_vs_manifest_baseline'] = row['makespan'] - old['makespan']
        # These six aliases ONLY come from the live rollout, for legacy summary functions.
        result.append(dict(case_dir=row['case_dir'], case_sha256=old['case_sha256'],
            distribution=old['distribution'], runtime_metrics=runtime,
            baseline_metrics=baseline, comparison_metrics=comparison,
            **{k: row[k] for k in ('makespan', 'completed', 'timeout', 'cycle_terminated', 'steps')}))
    return result


def balanced_indices(pool, count, rng, allowed_cases=None):
    cases = sorted(set(pool['case_indices'].tolist()))
    if allowed_cases is not None:
        cases = sorted(set(cases) & set(allowed_cases))
    if not cases or count <= 0:
        raise ValueError('Empty balanced critic sample.')
    # Uniform cases, then uniform trajectory positions within each case. Cycle
    # shuffled cases before reuse so long trajectories cannot dominate updates.
    chosen = []
    while len(chosen) < count:
        chosen.extend(rng.permutation(cases).tolist())
    lookup = pool.setdefault('_by_case', {c: np.flatnonzero(pool['case_indices'].numpy() == c) for c in
                                        sorted(set(pool['case_indices'].tolist()))})
    return [int(rng.choice(lookup[c])) for c in chosen[:count]]


class SplitEncoderLearner(FastCaseMinibatchLearner):
    def __init__(self, policy, args, contract, progress, *, arm):
        super().__init__(policy, args, contract, progress,
                         cache_max_bytes=contract['replay_cache_max_bytes'])
        install_encoders(policy.ac, arm)
        self.arm = arm
        self.actor_params, self.critic_params = [], []
        actor_groups, value_groups = [], []
        for prefixes, lr, groups, flat in (
                (RESOURCE_PREFIXES, contract['actor_lr'], actor_groups, self.actor_params),
                (('resource_encoder.',), contract['resource_encoder_lr'], actor_groups, self.actor_params),
                (('team_critic.',), contract['critic_lr'], value_groups, self.critic_params),
                (('value_encoder.',), contract['value_encoder_lr'], value_groups, self.critic_params)):
            params = [p for n, p in policy.ac.named_parameters() if n.startswith(prefixes)]
            if params:
                groups.append(dict(params=params, lr=lr, role=prefixes[0]))
                flat.extend(params)
        for n, p in policy.ac.named_parameters():
            p.requires_grad_(n.startswith(ACTOR + VALUE))
        if {id(p) for p in self.actor_params} & {id(p) for p in self.critic_params}:
            raise RuntimeError('Actor/critic optimizer ownership overlaps.')
        self.actor_optim = torch.optim.Adam(actor_groups)
        self.critic_optim = torch.optim.Adam(value_groups)
        self.protected = protected_summary(policy.ac)
        self.source_actor = summary(policy.ac, ACTOR)
        self.critic_rng = np.random.default_rng(contract['seed'] + 400000)
        self._value_pool = None
        self.critic_seconds = 0.
        self._record_reference = False
        self._snapshot_values = False
        self.gradient_checks = dict(actor_backwards=0, resource_encoder_nonzero=0,
                                    critic_updates=0, value_encoder_nonzero=0)
        self.initial_architecture = architecture_manifest(policy.ac)
        self.verify_initial_reference = False
        self.initial_reference_checks = dict(events=0, max_logp_error=0., max_hidden_error=0.)
        assert_no_alias(policy.ac, policy.bc_reference_ac)
        learner_mode(policy.ac)

    def assert_protected(self):
        if protected_summary(self.policy.ac) != self.protected:
            raise RuntimeError('V5 changed an original plane/ready/protected tensor.')

    def value_encoding(self, graph, *, gradient):
        ac = self.policy.ac
        encoder = getattr(ac, 'value_encoder', ac.encoder)
        with torch.set_grad_enabled(gradient and hasattr(ac, 'value_encoder')):
            if gradient and hasattr(ac, 'value_encoder') and self.contract['value_encoder_activation_checkpoint']:
                out = checkpoint(encoder, graph, use_reentrant=False, preserve_rng_state=True)
            else:
                out = encoder(graph)
        value = out['global_emb']
        require_finite(value, 'value graph encoding')
        return value

    def collect_forward(self, policy, obs, hidden, active, history, types, **kwargs):
        out, encoded = super().collect_forward(policy, obs, hidden, active, history, types, **kwargs)
        behavior = kwargs.get('actions') is None and not kwargs.get('reference', False)
        if behavior and self.verify_initial_reference:
            if kwargs.get('hungarian', False) or not kwargs.get('deterministic', False):
                raise ValueError('Initial reference equality audit requires deterministic AR.')
            with initial_reference_kernel_context(policy):
                reference, _ = super().collect_forward(policy, obs, hidden, active, history, types,
                    deterministic=True, reference=True, encoded=encoded)
            for name, result in (('current', out), ('B0 reference', reference)):
                require_finite(result['log_probs'], name + ' initial audit log probabilities')
                require_finite(result['rnn_states'], name + ' initial audit recurrent states')
            error = float((out['log_probs'] - reference['log_probs']).abs().max())
            hidden_error = float((out['rnn_states'] - reference['rnn_states']).abs().max())
            if (error > 1e-6 or hidden_error > 1e-6
                    or not torch.equal(out['actions'], reference['actions'])
                    or not torch.equal(out['decision_mask'], reference['decision_mask'])):
                raise RuntimeError('Copied architecture is not zero-update equivalent to B0: '
                    f'logp_error={error}, hidden_error={hidden_error}, '
                    f'actions_equal={torch.equal(out["actions"], reference["actions"])}, '
                    f'masks_equal={torch.equal(out["decision_mask"], reference["decision_mask"])}')
            self.initial_reference_checks['events'] += 1
            self.initial_reference_checks['max_logp_error'] = max(error, self.initial_reference_checks['max_logp_error'])
            self.initial_reference_checks['max_hidden_error'] = max(hidden_error, self.initial_reference_checks['max_hidden_error'])
        if behavior and self._snapshot_values and hasattr(policy.ac, 'value_encoder'):
            # Only the collector's value feature is replaced. Actor/plane and
            # frozen cache still use their own explicit graph views.
            parts = []
            size = len(obs) if kwargs.get('hungarian', False) else self.contract['minibatch_cases']
            for start in range(0, len(obs), size):
                parts.append(self.value_encoding(Batch.from_data_list(list(obs[start:start + size])).to(self.device), gradient=False))
            out = {**out, 'global_emb': torch.cat(parts)}
        if behavior and self._record_reference:
            valid = resource_mask(out['decision_mask'], self.args.max_agent_num).any(-1).cpu().tolist()
            for i, ok in enumerate(valid):
                if not ok:
                    continue
                count = self._reference_seen[i] + 1
                self._reference_seen[i] = count
                slot = count - 1 if count <= self.contract['reference_states_per_case'] else int(self._reference_rng.integers(count))
                if slot < self.contract['reference_states_per_case']:
                    record = dict(graph=obs[i].clone().cpu(), embedding=out['global_emb'][i].detach().cpu().clone(),
                                  flat_index=self._reference_flat, case_index=i)
                    if slot == len(self._reference_samples[i]):
                        self._reference_samples[i].append(record)
                    else:
                        self._reference_samples[i][slot] = record
                self._reference_flat += 1
        return out, encoded

    def collect(self, envs, **kwargs):
        if kwargs.get('bc', False):
            raise ValueError('V5 has no BC/teacher execution path.')
        self._record_reference = kwargs.get('capture_values', False)
        self._snapshot_values = kwargs.get('training', False)
        self._reference_seen = [0] * len(envs.ps)
        self._reference_samples = [[] for _ in envs.ps]
        self._reference_flat = 0
        self._reference_rng = np.random.default_rng(self.contract['seed'] + 500000)
        try:
            result = super().collect(envs, **kwargs)
            objectives = envs.call('get_training_objective')
            for row, current in zip(result['cases'], objectives):
                if (row['case_dir'] != Path(current['case_id']).name
                        or row['makespan'] != current['cmax']):
                    raise ValueError('Runtime objective belongs to a different case/result.')
                row.update({k: v for k, v in current.items() if k.startswith('resource_') or k == 'team_return'})
            if self._record_reference:
                data = result.pop('value_data')
                if self._reference_flat != len(data['targets']):
                    raise ValueError('Raw reference graph/MC target alignment failed.')
                samples = sorted((s for group in self._reference_samples for s in group), key=lambda s: s['flat_index'])
                result['raw_value_pool'] = dict(graphs=[s['graph'] for s in samples],
                    embeddings=torch.stack([s['embedding'] for s in samples]),
                    targets=data['targets'][[s['flat_index'] for s in samples]],
                    case_indices=torch.tensor([s['case_index'] for s in samples]),
                    observable_only=True, sampling='uniform_reservoir_per_case_max32')
            return result
        finally:
            self._record_reference = self._snapshot_values = False
            self._reference_samples = []

    def _replay(self, rollout, indices, advantages, *, gradient, bc=False):
        result = super()._replay(rollout, indices, advantages, gradient=gradient, bc=bc)
        if gradient:
            if any(p.grad is not None for p in self.critic_params):
                raise RuntimeError('Actor backward reached value-owned parameters.')
            if hasattr(self.policy.ac, 'resource_encoder'):
                grads = [p.grad for p in self.policy.ac.resource_encoder.parameters() if p.grad is not None]
                if not grads or not any(bool(g.abs().max() > 0) for g in grads):
                    raise RuntimeError('Detached/zero split resource encoder gradient.')
                for g in grads:
                    require_finite(g, 'resource encoder gradient')
                self.gradient_checks['resource_encoder_nonzero'] += 1
            self.gradient_checks['actor_backwards'] += 1
        return result

    def pool_from_rollout(self, rollout):
        graphs, embeddings, targets, cases = [], [], [], []
        for t, frame in enumerate(rollout['frames']):
            for i in torch.nonzero(frame['mask'].any(-1)).flatten().tolist():
                graphs.append(frame['obs'][i])  # Immutable CPU graph; never .to() it in place.
                embeddings.append(frame['global_emb'][i])
                targets.append(float(rollout['targets'][t, i]))
                cases.append(i)
        return dict(graphs=graphs, embeddings=torch.stack(embeddings),
                    targets=torch.tensor(targets), case_indices=torch.tensor(cases))

    def _pool_prediction(self, pool, indices, *, gradient):
        # Identical graph microbatch membership/shapes for all four arms.
        # E0/E1 encode without gradients; E2/E3 encode with their own value GNN.
        # In particular do not mix H24 reference embeddings with GNN6 training.
        graph = Batch.from_data_list([pool['graphs'][i] for i in indices]).to(self.device)
        features = self.value_encoding(graph, gradient=gradient)
        pred = self.policy.ac.team_critic(features).flatten()
        require_finite(pred, 'critic prediction')
        return pred

    def critic_update(self, embeddings, targets, steps):
        started = time.monotonic()
        pool = self._value_pool
        if pool is None:
            raise ValueError('Trainable graph critic needs raw observable states, not cached features.')
        actor_before = summary(self.policy.ac, ACTOR)
        losses = []
        self.policy.ac.eval()
        for _ in range(steps):
            selected = balanced_indices(pool, self.contract['critic_batch_states'], self.critic_rng,
                                        getattr(self, '_critic_allowed_cases', None))
            self.critic_optim.zero_grad(set_to_none=True)
            total = 0.
            micro = self.contract['critic_microbatch_states']
            for start in range(0, len(selected), micro):
                indices = selected[start:start + micro]
                pred = self._pool_prediction(pool, indices, gradient=True)
                target = pool['targets'][indices].to(self.device)
                loss = (pred - target).square().sum() / len(selected)
                require_finite(loss, 'accumulated critic loss')
                loss.backward()
                total += float(loss.detach())
            if any(p.grad is not None for p in self.actor_params):
                raise RuntimeError('Critic backward reached actor-owned parameters.')
            if hasattr(self.policy.ac, 'value_encoder'):
                grads = [p.grad for p in self.policy.ac.value_encoder.parameters() if p.grad is not None]
                if not grads or not any(bool(g.abs().max() > 0) for g in grads):
                    raise RuntimeError('Detached/zero split value encoder gradient.')
                self.gradient_checks['value_encoder_nonzero'] += 1
            norm = torch.nn.utils.clip_grad_norm_(self.critic_params, self.contract['max_grad_norm'])
            require_finite(norm, 'critic accumulated gradient')
            self.critic_optim.step()
            self.critic_optim.zero_grad(set_to_none=True)
            self.critic_steps += 1
            self.gradient_checks['critic_updates'] += 1
            losses.append(total)
            self.progress('critic_microbatch_update', critic_steps=self.critic_steps,
                logical_batch_states=len(selected), graph_microbatch_states=micro)
        if summary(self.policy.ac, ACTOR) != actor_before:
            raise RuntimeError('Critic-only update changed a resource actor/encoder tensor.')
        self.assert_protected()
        self.critic_seconds += time.monotonic() - started
        return losses

    def pool_metrics(self, pool, indices):
        parts = []
        with torch.no_grad():
            for start in range(0, len(indices), self.contract['critic_microbatch_states']):
                parts.append(self._pool_prediction(pool, indices[start:start + self.contract['critic_microbatch_states']], gradient=False).cpu())
        return self.value_metrics(torch.cat(parts), pool['targets'][indices])

    def warmup_value(self, data, validation_indices):
        self.policy.ac.eval()
        held = torch.isin(data['case_indices'], torch.tensor(list(validation_indices)))
        if not held.any() or held.all():
            raise ValueError('Warmup requires case-disjoint fit/validation graphs.')
        self._value_pool = data
        self._critic_allowed_cases = set(data['case_indices'][~held].tolist())
        validation = torch.nonzero(held).flatten().tolist()
        before_actor = summary(self.policy.ac, ACTOR)
        started = time.monotonic()
        try:
            initial = self.pool_metrics(data, validation)
            losses = self.critic_update(None, None, self.contract['critic_warmup_steps'])
            final = self.pool_metrics(data, validation)
            if summary(self.policy.ac, ACTOR) != before_actor:
                raise RuntimeError('Warmup changed actor parameters.')
            return dict(steps=len(losses), losses=losses, seconds=time.monotonic() - started,
                validation_case_indices=list(validation_indices), validation_before=initial,
                validation_after=final, actor_unchanged=True, logical_batch_states=48,
                graph_microbatch_states=6, validation_scope='case_disjoint_initial_warmup_only')
        finally:
            self._value_pool = None
            self._critic_allowed_cases = None

    def update(self, rollout, **kwargs):
        self._value_pool = self.pool_from_rollout(rollout)
        try:
            result = super().update(rollout, **kwargs)
            result.update(arm=self.arm, protocol=PROTOCOL, gradient_checks=dict(self.gradient_checks),
                critic_sampling='uniform_case_then_uniform_time', critic_logical_batch=48,
                critic_graph_microbatch=6, learned_encoder_cache=False)
            return result
        finally:
            self._value_pool = None


def training_state(learner):
    return dict(model={k: v.detach().cpu().clone() for k, v in learner.policy.ac.state_dict().items()},
        actor_optim=copy.deepcopy(learner.actor_optim.state_dict()),
        critic_optim=copy.deepcopy(learner.critic_optim.state_dict()),
        critic_rng=copy.deepcopy(learner.critic_rng.bit_generator.state),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(learner.device) if learner.device.type == 'cuda' else None,
        actor_steps=learner.actor_steps, critic_steps=learner.critic_steps,
        probability_checks=copy.deepcopy(learner.probability_checks),
        minibatch_checks=copy.deepcopy(learner.minibatch_checks),
        gradient_checks=copy.deepcopy(learner.gradient_checks), arm=learner.arm)


def restore_training_state(learner, state):
    if state['arm'] != learner.arm:
        raise ValueError('Cannot restore a different encoder arm.')
    learner.policy.ac.load_state_dict(state['model'], strict=True)
    # deepcopy is essential: CPU Adam step tensors otherwise alias across arms.
    learner.actor_optim.load_state_dict(copy.deepcopy(state['actor_optim']))
    learner.critic_optim.load_state_dict(copy.deepcopy(state['critic_optim']))
    learner.critic_rng.bit_generator.state = copy.deepcopy(state['critic_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if learner.device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), learner.device)
    for name in ('actor_steps', 'critic_steps', 'probability_checks', 'minibatch_checks', 'gradient_checks'):
        setattr(learner, name, copy.deepcopy(state[name]))
    learner.clear_replay_cache()
    learner.assert_protected()
    assert_no_alias(learner.policy.ac, learner.policy.bc_reference_ac)
