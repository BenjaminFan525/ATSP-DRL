"""Bounded, completed-Cmax policy improvement (not a PPO likelihood).

Counterfactuals retain the *entire* heterogeneous GPU batch, both GRUs and
the authoritative action history. Only one environment's first resource
action is intervened on; every companion environment continues normally.
This deliberately does not reuse the old CPU/S1-plus-teacher wait labels.
"""
from __future__ import annotations

import copy
import hashlib
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from onpolicy.utils.stage2_matching import solve_resource_matching, validate_teacher_matching


COST_CONTRACT = 'joint4_full_gpu_batch_frozen_student_cmax_v1'
LOSS_CONTRACT = 'cost_pair_hinge_whole_joint_bc_abstention_v1'


def validate_joint(legal, actions, lookahead):
    """Global uniqueness + maximum Blocking coverage + serialized last chance."""
    legal = np.asarray(legal, dtype=bool)
    actions = np.asarray(actions, dtype=np.int64)
    lookahead = np.asarray(lookahead, dtype=bool)
    validate_teacher_matching(np.where(legal, 0., -np.inf), actions, lookahead)
    remaining = legal.copy()
    for row, action in enumerate(actions):
        last = remaining[row] & ~lookahead & ~remaining[row + 1:].any(axis=0)
        last[0] = False
        allowed = last if last.any() else remaining[row]
        if not allowed[action]:
            raise ValueError('Candidate violates serialized last-chance legality.')
        if action:
            remaining[:, action] = False


def joint_candidates(legal, student, teacher, lookahead, limit=4):
    """Include the deployed incumbent, then teacher, wait/dispatch and swaps.

This is bounded LOCAL improvement, not exhaustive dispatch enumeration and
not a claim of optimality. Invalid neighbors are rejected, never repaired
inside the objective. Candidate order is fixed before observing any costs.
"""
    if limit != 4:
        raise ValueError('The preregistered candidate cap is exactly four.')
    legal = np.asarray(legal, dtype=bool)
    student, teacher = [np.asarray(x, dtype=np.int64) for x in (student, teacher)]
    result = []

    def add(action, origin, required=False):
        try:
            validate_joint(legal, action, lookahead)
        except ValueError:
            if required:
                raise
            return
        if len(result) < limit and not any(np.array_equal(x['action'], action) for x in result):
            result.append({'action': action.copy(), 'origin': origin})

    add(student, 'student', True)
    add(teacher, 'derived_teacher', True)
    # Prefer a wait/dispatch neighbor before identity-only substitutions.
    for row in range(len(student)):
        for action in ([0] if student[row] else np.flatnonzero(legal[row, 1:]) + 1):
            alternate = student.copy()
            alternate[row] = action
            add(alternate, 'wait_dispatch')
            if len(result) >= limit:
                return result
    for left in range(len(student)):
        for right in range(left + 1, len(student)):
            alternate = student.copy()
            alternate[left], alternate[right] = alternate[right], alternate[left]
            add(alternate, 'identity_swap')
            if len(result) >= limit:
                return result
    for row in range(len(student)):
        for action in np.flatnonzero(legal[row]):
            alternate = student.copy()
            alternate[row] = action
            add(alternate, 'local_substitution')
            if len(result) >= limit:
                return result
    return result


def cost_preference_loss(scores, candidates, costs, *, scale=120., clip=4., tie_seconds=1.):
    """Cost-sensitive ordered-pair hinge; ties never force a physical identity.

Scores are additive row scores, NOT probabilities of Hungarian matchings.
The scale and clipping constants must be locked using training data only.
An incomplete candidate invalidates the entire evidence group.
"""
    if not (np.isfinite([scale, clip, tie_seconds]).all() and scale > 0 and clip > 0 and tie_seconds >= 0):
        raise ValueError('Invalid fixed cost normalization.')
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[1] != scores.shape[0] or len(costs) != len(candidates):
        raise ValueError('Joint candidate/cost dimensions differ.')
    if len(candidates) < 2 or any(x is None or not np.isfinite(x) or x <= 0 for x in costs):
        return scores.new_zeros(()), {'covered': False, 'pairs': 0, 'regret': None}
    costs = np.asarray(costs, dtype=np.float64)
    indices = torch.as_tensor(candidates, dtype=torch.long, device=scores.device)
    values = scores[torch.arange(scores.shape[0], device=scores.device)[None, :], indices]
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Cost candidate includes an illegal model edge.')
    choice_rows = max(1, int((torch.isfinite(scores).sum(-1) > 1).sum()))
    joint = values.sum(-1) / choice_rows
    terms = []
    for good in range(len(costs)):
        for bad in range(len(costs)):
            gap = float(costs[bad] - costs[good])
            if gap > tie_seconds:
                weight = min(clip, gap / scale)
                terms.append(weight * torch.relu(weight - (joint[good] - joint[bad])))
    predicted = int(joint.detach().argmax())
    return (torch.stack(terms).mean() if terms else joint.sum() * 0.), {
        'covered': True, 'pairs': len(terms), 'regret': float(costs[predicted] - costs.min()),
    }


def apply_cost_evidence(logits, active, evidence, *, scale, clip, tie_seconds):
    """Return the cost loss and an anchor mask excluding EVERY covered row."""
    anchor = np.asarray(active, dtype=bool).copy()
    terms, pairs, covered = [], 0, 0
    regret = 0.
    for item in evidence or []:
        env, rows = item['env'], np.asarray(item['rows'], dtype=np.int64)
        if not np.array_equal(rows, np.flatnonzero(anchor[env])):
            raise ValueError('Cost evidence must cover the entire active resource joint action.')
        selected = logits[env, torch.as_tensor(rows, device=logits.device)]
        detached = selected.detach().cpu().double().numpy()
        lookup = np.ones(selected.shape[1], dtype=bool)
        lookup[:len(item['lookahead'])] = item['lookahead']
        incumbent = solve_resource_matching(detached, lookup)
        if not any(np.array_equal(incumbent, x) for x in item['candidates']):
            raise ValueError('Current decoded action escaped the scored candidate set.')
        loss, metrics = cost_preference_loss(selected, item['candidates'], item['costs'],
            scale=scale, clip=clip, tie_seconds=tie_seconds)
        if metrics['covered']:
            anchor[env] = False  # including ties: no arbitrary teacher-identity target
            covered += 1
            pairs += metrics['pairs']
            regret += metrics['regret']
            if metrics['pairs']:
                terms.append(loss)
    return (torch.stack(terms).mean() if terms else logits.new_zeros(())), anchor, {
        'device_bc_cost_covered_joints': covered, 'device_bc_cost_pairs': pairs,
        'device_bc_cost_anchor_suppressed_rows': int(np.asarray(active).sum() - anchor.sum()),
        'device_bc_cost_candidate_regret_sum': regret,
    }


def policy_digest(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.ac.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def actor_forward(policy, obs, rnn, active, history, types):
    with torch.no_grad():
        actions, state = policy.get_actor_actions(Batch.from_data_list(list(obs)), rnn,
            active, history[..., 0], history[..., 1], deterministic=True, agent_types=types)
    return actions.detach().cpu().numpy(), state.detach().cpu().numpy()


def deterministic_learner_mode(policy):
    """Disable dropout, but retain cuDNN's required resource-GRU backward reserve.

    cuDNN cannot backpropagate through an eval-mode GRU. Only resource GRU
    leaves are train-mode, and they must have zero recurrent dropout. Plane
    GRUs, encoder, pointer dropout and every other module stay eval-mode.
    """
    policy.ac.eval()
    roots = list(policy.ac.device_actor_param.modules) + [policy.ac.transporter_sel_enc,
                                                       policy.ac.transporter_actor]
    for root in roots:
        for module in root.modules():
            if isinstance(module, torch.nn.RNNBase) and any(p.requires_grad for p in module.parameters()):
                if module.dropout != 0:
                    raise ValueError('Deterministic resource GRU protocol requires zero recurrent dropout.')
                module.train()


class CostIteration:
    """One fixed target policy; live behavior can continue its BC/DAgger updates.

Only cost-bearing decisions use exact current-parameter prefix reconstruction;
the uncovered BC anchor retains the existing online truncated-BPTT convention.
Frozen target GRUs are NEVER fed to the trainable learner.
"""
    def __init__(self, runner, epoch):
        self.runner, self.epoch = runner, int(epoch)
        self.args = runner.all_args
        self.enabled = bool(self.args.stage2_cost_improvement)
        deterministic_learner_mode(runner.policy)
        if runner.device != torch.device('cuda:0') or not torch.are_deterministic_algorithms_enabled():
            raise ValueError('Student costs require deterministic GPU0, not a CPU proxy.')
        if (runner.request_ready_loss_coef != 0 or runner.device_bc_training_scope != 'policy_frozen_ready'
                or not self.args.device_bc_teacher_deployment_projection or runner.exact_resume_stage2_supervised):
            raise ValueError('Cost iteration requires a fresh frozen-S1/R1 fork and projected teachers.')
        self.policy = None
        if self.enabled:
            self.policy = copy.copy(runner.policy)
            self.policy.ac = copy.deepcopy(runner.policy.ac)
            self.policy.ac.eval()
            for module in self.policy.ac.modules():
                if isinstance(module, torch.nn.RNNBase):
                    module.flatten_parameters()
            for parameter in self.policy.ac.parameters():
                parameter.requires_grad_(False)
        self.model_sha256 = policy_digest(runner.policy)
        self.path = Path(runner.log_dir) / f'cost_evidence_epoch{epoch}.jsonl'
        if self.path.exists():
            raise FileExistsError(self.path)
        self.counts = dict(candidate_queries=0, continuation_vector_steps=0, completed=0,
            incomplete=0, selected_states=0, covered_joints=0, preference_pairs=0,
            continuation_wall_seconds=0., prefix_replay_steps=0, zero_intervention_replays=0,
            snapshot_bytes=0)
        self.case_counts = {}
        self.rollout = 0
        self.target_path = Path(runner.log_dir) / f'cost_target_epoch{epoch}.pt'
        self.target_file_sha256 = None
        if self.enabled:
            with self.target_path.open('xb') as handle:
                torch.save({'model': {k: v.detach().cpu() for k, v in self.policy.ac.state_dict().items()},
                            'policy_sha256': self.model_sha256, 'epoch': epoch}, handle)
            self.target_file_sha256 = hashlib.sha256(self.target_path.read_bytes()).hexdigest()
            self.counts['snapshot_bytes'] += self.target_path.stat().st_size

    def reset_rollout(self, rnn):
        self.rollout += 1
        self.rnn = np.zeros_like(rnn)
        self.prefix = []
        self.ordinals = np.zeros(len(rnn), dtype=np.int64)
        self.selected = []
        self.step = 0

    def prepare(self, obs, active, history, infos, learner_rnn):
        self.selected = []
        if self.step < self.args.stage2_cost_snapshot_horizon:
            for env, graph in enumerate(obs):
                rows = np.flatnonzero((active[env, :, 0] > 0)
                    & (np.arange(active.shape[1]) >= self.runner.policy.ac.max_plane_agents))
                if not len(rows):
                    continue
                legal = graph.request_mask_matrix.detach().cpu().numpy().astype(bool)[rows]
                if (legal.sum(axis=1) > 1).any():
                    self.ordinals[env] += 1
                    if self.ordinals[env] in (4, 16):
                        self.selected.append(env)
        if self.selected:
            # The weights have changed since the rollout started. Reconstruct
            # from raw prefix inputs rather than reusing stale hidden vectors.
            learner_rnn = np.zeros_like(learner_rnn)
            for old_obs, old_active, old_history, old_types, old_dones in self.prefix:
                _, learner_rnn = actor_forward(self.runner.policy, old_obs, learner_rnn,
                    old_active, old_history, old_types)
                learner_rnn[old_dones] = 0.
            self.counts['prefix_replay_steps'] += len(self.prefix)
        if self.enabled and self.step < self.args.stage2_cost_snapshot_horizon:
            self.frozen_actions, self.next_rnn = actor_forward(self.policy, obs, self.rnn,
                active, history, infos.get('agent_types'))
        self.learner_rnn = learner_rnn
        return learner_rnn

    def finish_step(self, obs, active, history, types, dones):
        if self.step < self.args.stage2_cost_snapshot_horizon:
            self.prefix.append(([graph.clone() for graph in obs], active.copy(),
                history.copy(), None if types is None else np.asarray(types).copy(), np.asarray(dones).copy()))
            if self.enabled:
                self.rnn = self.next_rnn.copy()
                self.rnn[np.asarray(dones, dtype=bool)] = 0.
        self.step += 1
        if self.step >= self.args.stage2_cost_snapshot_horizon:
            self.prefix.clear()

    def branch(self, obs, infos, history, first_actions, target_env):
        """One target intervention, all batch companions continue with pi_k."""
        started = time.monotonic()
        envs = self.runner.envs
        envs.call('stage2_cost_branch_restore')
        actions, hidden = first_actions.copy(), self.next_rnn.copy()
        last_actions = history.copy()
        completed, makespan, reason, steps = False, None, 'step_limit', 0
        for steps in range(1, self.runner.episode_length + 1):
            if time.monotonic() - started > self.args.stage2_cost_branch_timeout_seconds:
                reason = 'wall_timeout'
                break
            results = envs.call_each('stage2_cost_branch_step', [(a,) for a in actions])
            graphs, _, done_rows, info_rows, statuses = zip(*results)
            dones = np.stack(done_rows).astype(bool)
            status = statuses[target_env] if target_env is not None else None
            hidden[dones] = 0.
            last_actions = actions
            if target_env is None and np.all(dones):
                completed = all(x['completed'] for x in statuses)
                reason = 'completed' if completed else 'incomplete_terminal'
                break
            if target_env is not None and np.all(dones[target_env]):
                completed = status['completed']
                makespan = status['makespan'] if completed else None
                reason = 'completed' if completed else 'incomplete_terminal'
                break
            if steps % 100 == 0:
                print(f'[StudentCF] epoch={self.epoch} target={target_env} step={steps} '
                    f'elapsed={time.monotonic()-started:.1f}s', flush=True)
            infos = envs.stack_infos(info_rows)
            active = self.runner._active_masks_from_info(infos)
            live_history = self.runner._authoritative_policy_history(infos, last_actions,
                self.runner.policy.ac.max_plane_agents)
            actions, hidden = actor_forward(self.policy, graphs, hidden, active,
                live_history, infos.get('agent_types'))
        elapsed = time.monotonic() - started
        self.counts['zero_intervention_replays' if target_env is None else 'candidate_queries'] += 1
        self.counts['continuation_vector_steps'] += steps
        self.counts['continuation_wall_seconds'] += elapsed
        self.counts['completed' if completed else 'incomplete'] += 1
        result = dict(completed=completed, makespan=makespan, steps=steps, reason=reason, wall_seconds=elapsed)
        if target_env is None:
            result['case_statuses'] = list(statuses) if steps else []
        return result

    def evidence(self, obs, active, history, infos, student, teacher):
        if not self.enabled or not self.selected:
            return []
        evidence = []
        prepared = []
        for env in self.selected:
            case = str(np.asarray(infos['case_id']).reshape(-1)[env])
            if self.case_counts.get(case, 0) >= 2:
                raise ValueError('More than two cost states per case/iteration.')
            self.case_counts[case] = self.case_counts.get(case, 0) + 1
            rows = np.flatnonzero((active[env, :, 0] > 0)
                & (np.arange(active.shape[1]) >= self.runner.policy.ac.max_plane_agents))
            legal = obs[env].request_mask_matrix.detach().cpu().numpy().astype(bool)[rows]
            lookahead = obs[env].request_is_lookahead.detach().cpu().numpy().astype(bool).reshape(-1)
            candidates = joint_candidates(legal, student[env, rows, 0], teacher[env, rows, 0], lookahead)
            if len(candidates) >= 2:
                prepared.append((env, case, rows, lookahead, candidates))
        if not prepared:
            return []
        plane_count = self.runner.policy.ac.max_plane_agents
        if not np.array_equal(student[:, :plane_count], self.frozen_actions[:, :plane_count]):
            raise ValueError('Current learner and frozen target disagree on the protected plane action.')
        directory = Path(self.runner.log_dir) / f'cost_states_epoch{self.epoch}' / f'rollout{self.rollout:03}_step{self.step:06}'
        directory.mkdir(parents=True, exist_ok=False)
        snapshots = self.runner.envs.call_each('stage2_cost_branch_capture',
            [(str(directory / f'env{env:02}.pkl.gz'),) for env in range(len(obs))])
        state_path = directory / 'policy_history.pt.gz'
        with state_path.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as handle:
                torch.save({'obs': list(obs), 'active': active, 'history': history,
                    'agent_types': infos.get('agent_types'), 'target_rnn': self.rnn,
                    'target_post_forward_rnn': self.next_rnn, 'learner_rnn': self.learner_rnn,
                    'raw_prefix': self.prefix, 'student': student, 'teacher': teacher,
                    'learner_model': {k: v.detach().cpu() for k, v in self.runner.policy.ac.state_dict().items()},
                    'target_checkpoint': str(self.target_path), 'target_file_sha256': self.target_file_sha256}, handle)
        state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
        self.counts['snapshot_bytes'] += sum(x['bytes'] for x in snapshots) + state_path.stat().st_size
        if self.counts['snapshot_bytes'] > 16 * 1024**3:
            raise ValueError('Declared16-GiB per-cost-iteration evidence budget exceeded; outputs preserved.')
        try:
            for env, case, rows, lookahead, candidates in prepared:
                outcomes = []
                for candidate in candidates:
                    first = self.frozen_actions.copy()
                    # The plane head is frozen, so keep the actual deployment
                    # plane action. Inactive resources retain their history.
                    first[env] = student[env]
                    first[env, rows, 0] = candidate['action']
                    first[env, rows, 1] = 0
                    outcomes.append(self.branch(obs, infos, history, first, env))
                row = dict(contract=COST_CONTRACT, loss_contract=LOSS_CONTRACT,
                    epoch=self.epoch, step=self.step, case=case, env=env, rows=rows.tolist(),
                    policy_sha256=self.model_sha256, batch_width=len(obs),
                    batch_snapshot_sha256=[x['sha256'] for x in snapshots],
                    environment_snapshots=snapshots, state_path=str(state_path), state_sha256=state_sha256,
                    target_checkpoint=str(self.target_path), target_file_sha256=self.target_file_sha256,
                    recurrent_sha256=hashlib.sha256(self.rnn.tobytes()).hexdigest(),
                    post_forward_recurrent_sha256=hashlib.sha256(self.next_rnn.tobytes()).hexdigest(),
                    history_sha256=hashlib.sha256(history.tobytes()).hexdigest(),
                    active_sha256=hashlib.sha256(active.tobytes()).hexdigest(),
                    lookahead=lookahead.tolist(), candidates=[x['action'].tolist() for x in candidates],
                    origins=[x['origin'] for x in candidates], outcomes=outcomes,
                    costs=[x['makespan'] for x in outcomes])
                with self.path.open('a') as handle:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                evidence.append(row)
                self.counts['selected_states'] += 1
        finally:
            self.runner.envs.call('stage2_cost_branch_clear')
        return evidence

    def finish_epoch(self):
        if self.enabled and policy_digest(self.policy) != self.model_sha256:
            raise ValueError('The continuation target policy changed inside an iteration.')
        if self.counts['candidate_queries'] > 8 * len(self.case_counts):
            raise ValueError('Counterfactual budget exceeded.')
        if self.enabled and not self.counts['selected_states']:
            raise ValueError('No nontrivial cost evidence collected; no automatic BC-only fallback.')
        path = Path(self.runner.log_dir) / f'cost_summary_epoch{self.epoch}.json'
        path.write_text(json.dumps(dict(contract=COST_CONTRACT, epoch=self.epoch,
            policy_sha256=self.model_sha256, counts=self.counts, case_counts=self.case_counts,
            enabled=self.enabled,
            evidence_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else None),
            sort_keys=True, allow_nan=False) + '\n')
