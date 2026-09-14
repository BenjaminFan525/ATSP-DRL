"""One deployment matching contract and opt-in, identity-preserving BC terms.

The scores are row-normalized log probabilities, NOT the probability of a
Hungarian matching.  Nothing in this module supplies a PPO likelihood.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


MATCHING_CONTRACT = 'private_noop_blocking_1e6_v1'
SUPERVISION_CONTRACT = 'physical_edges_typed_empty_wait_v2'
TEACHER_PROJECTION_CONTRACT = 'iga_derived_max_blocking_identity_serial_v1'


def project_teacher_matching(legal, target, request_is_lookahead):
    """Explicitly derive a deployable teacher; never call inside a loss.

    Preserve every already compatible assignment. Otherwise maximize Blocking
    coverage, then reward unchanged physical edges (including private no-ops).
    If that LAP conflicts with serialized last-chance legality, pin the valid
    prefix and re-solve with the first conflicting row restricted. This is a
    deterministic repair, not an IGA reoptimization or a globally minimum-edit
    solution under the additional serialized constraints.
    """
    legal = np.asarray(legal, dtype=bool)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    lookahead = np.asarray(request_is_lookahead, dtype=bool).reshape(-1)
    if (legal.ndim != 2 or legal.shape[1] != lookahead.size
            or legal.shape[0] != target.size or not legal.shape[1]
            or np.any(target < 0) or np.any(target >= legal.shape[1])):
        raise ValueError('Invalid teacher projection dimensions/actions.')
    if not legal[:, 0].all() or not legal[np.arange(target.size), target].all():
        raise ValueError('Teacher projection requires legal identity edges and private no-ops.')
    positive = target[target > 0]
    if np.unique(positive).size != positive.size:
        raise ValueError('Teacher projection cannot hide duplicate original claims.')

    def serial_options(row, claimed):
        allowed = legal[row].copy()
        allowed[list(claimed)] = False
        last_chance = allowed & ~lookahead & ~legal[row + 1:].any(axis=0)
        last_chance[0] = False
        return last_chance if last_chance.any() else allowed

    claimed = set()
    for row, action in enumerate(target):
        if not serial_options(row, claimed)[action]:
            raise ValueError('Original teacher violates serialized last-chance legality.')
        if action:
            claimed.add(int(action))
    scores = np.where(legal, 0., -np.inf)
    maximum = solve_resource_matching(scores, lookahead)
    count = lambda actions: int(((actions > 0) & ~lookahead[actions]).sum())
    before, required = count(target), count(maximum)
    stats = {'teacher_projection_checked': 1, 'teacher_projection_events': 0,
             'teacher_projection_changed_rows': 0, 'teacher_projection_added_blocking': 0,
             'teacher_projection_serial_resolves': 0}
    if before == required:
        return target.copy(), stats
    # Total identity reward is < 1, so even large row counts cannot cancel the
    # deployed 1e6 Blocking bonus. IGA gene scores are not calibrated Q values.
    scores[np.arange(target.size), target] = 1. / (target.size + 1)
    projected = solve_resource_matching(scores, lookahead)
    claimed = set()
    for row in range(target.size):
        allowed = serial_options(row, claimed)
        if not allowed[projected[row]]:
            scores[row, ~allowed] = -np.inf
            projected = solve_resource_matching(scores, lookahead)
            stats['teacher_projection_serial_resolves'] += 1
        chosen = int(projected[row])
        if not allowed[chosen] or count(projected) != required:
            raise ValueError('Serialized teacher repair lost maximum Blocking coverage.')
        scores[row, np.arange(scores.shape[1]) != chosen] = -np.inf
        if chosen:
            claimed.add(chosen)
    validate_teacher_matching(np.where(legal, 0., -np.inf), projected, lookahead)
    stats.update(teacher_projection_events=1,
                 teacher_projection_changed_rows=int((projected != target).sum()),
                 teacher_projection_added_blocking=required - before)
    return projected, stats


def solve_resource_matching(scores, request_is_lookahead, *, adjustment=None):
    """Preserve the deployed float64 matrix, column order and tie-breaking.

``adjustment`` is a decomposable structured-margin cost, used only for
loss-augmented inference. It must never make an illegal edge legal.
"""
    scores = np.asarray(scores, dtype=np.float64)
    lookahead = np.asarray(request_is_lookahead, dtype=bool).reshape(-1)
    if scores.ndim != 2 or scores.shape[1] < 1 or lookahead.size != scores.shape[1]:
        raise ValueError('Matching score/lookup dimensions differ.')
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('Matching scores contain NaN or positive infinity.')
    row_count, request_count = scores.shape
    if not row_count:
        return np.empty(0, dtype=np.int64)
    values = scores
    if adjustment is not None:
        adjustment = np.asarray(adjustment, dtype=np.float64)
        if adjustment.shape != scores.shape or not np.isfinite(adjustment).all():
            raise ValueError('Invalid matching margin adjustment.')
        values = scores + adjustment
    real_count = request_count - 1
    matrix = np.full((row_count, real_count + row_count), -1.0e30, dtype=np.float64)
    if real_count:
        matrix[:, :real_count] = values[:, 1:]
        legal = np.isfinite(scores[:, 1:])
        matrix[:, :real_count][legal & ~lookahead[None, 1:]] += 1.0e6
    for row in range(row_count):
        matrix[row, real_count + row] = values[row, 0]
    matrix[~np.isfinite(matrix)] = -1.0e30
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    assignments = np.full(row_count, -1, dtype=np.int64)
    assignments[rows] = columns
    if np.any(assignments < 0):
        raise ValueError('Matching does not cover all active device rows.')
    actions = np.where(assignments < real_count, assignments + 1, 0)
    if not np.isfinite(scores[np.arange(row_count), actions]).all():
        raise ValueError('Matching has no feasible complete assignment.')
    return actions.astype(np.int64)


def validate_teacher_matching(scores, target, request_is_lookahead):
    """Reject impossible full teacher actions; do not silently rematch labels."""
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    lookahead = np.asarray(request_is_lookahead, dtype=bool).reshape(-1)
    if target.size != scores.shape[0] or np.any(target < 0) or np.any(target >= scores.shape[1]):
        raise ValueError('Teacher matching has out-of-range actions.')
    if not np.isfinite(scores[np.arange(target.size), target]).all():
        raise ValueError('Teacher identity edge is absent from the deployment legal mask.')
    positive = target[target > 0]
    if len(set(positive.tolist())) != positive.size:
        raise ValueError('Teacher joint action duplicates a real request.')
    feasibility_scores = np.where(np.isfinite(scores), 0.0, -np.inf)
    maximum = solve_resource_matching(feasibility_scores, lookahead)
    count = lambda actions: int(((actions > 0) & ~lookahead[actions]).sum())
    if count(target) != count(maximum):
        raise ValueError('Teacher joint action violates deployment Blocking priority.')
    return count(target)


def full_matching_supervision(scores, target, request_is_lookahead, groups,
                              noop_causes, *, edge_enabled, wait_enabled, margin=0.20):
    """Audit one complete resource joint action and calculate opt-in losses.

Groups partition the active rows by resource type, not by device identity.
Their FULL teacher row assignments are retained. Empty groups enter the
waiting term only when an actual legal choice has a typed temporal-defer
reason. Ambiguous/structural empty groups remain visible in the counters.
"""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('Structured edge margin must be finite and nonnegative.')
    detached = scores.detach().cpu().double().numpy()
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    lookahead = np.asarray(request_is_lookahead, dtype=bool).reshape(-1)
    causes = np.asarray(noop_causes, dtype=object).reshape(-1)
    if causes.size != target.size:
        raise ValueError('Teacher no-op reasons do not match active rows.')
    flattened = [int(row) for group in groups for row in group]
    if sorted(flattened) != list(range(target.size)):
        raise ValueError('Resource groups must partition every active device exactly once.')
    # A job/site request can require several resource types. Its global
    # one-claim constraint couples those types, so edge inference MUST remain
    # joint. Type groups are only metric/wait strata, never independent LAPs.
    blocking = validate_teacher_matching(detached, target, lookahead)
    predicted = solve_resource_matching(detached, lookahead)
    choice_rows = np.isfinite(detached).sum(axis=1) > 1
    metrics = {
        'device_bc_deploy_environments': 1,
        'device_bc_deploy_joint_correct': int(np.array_equal(predicted, target)),
        'device_bc_deploy_rows': int(target.size),
        'device_bc_deploy_edge_correct': int((predicted == target).sum()),
        'device_bc_deploy_choice_rows': int(choice_rows.sum()),
        'device_bc_deploy_choice_edge_correct': int(((predicted == target) & choice_rows).sum()),
        'device_bc_teacher_blocking_edges': blocking,
        'device_bc_matching_groups': len(groups),
        'device_bc_matching_nonempty_groups': 0,
        'device_bc_deploy_set_correct': 0,
        'device_bc_matching_empty_groups': 0,
        'device_bc_wait_eligible_groups': 0,
        'device_bc_wait_eligible_rows': 0,
        'device_bc_wait_structural_groups': 0,
        'device_bc_wait_untyped_choice_groups': 0,
        'device_bc_full_edge_groups': 0,
        'device_bc_empty_wait_loss_groups': 0,
        'device_bc_full_edge_loss_sum': 0.0,
        'device_bc_empty_wait_loss_sum': 0.0,
    }
    edge_terms, wait_terms = [], []
    if edge_enabled and np.any(target > 0) and choice_rows.any():
        costs = (np.arange(scores.shape[1])[None, :] != target[:, None]).astype(float) * margin
        competitor = solve_resource_matching(detached, lookahead, adjustment=costs)
        row_ids = torch.arange(target.size, device=scores.device)
        rival_ids = torch.as_tensor(competitor, dtype=torch.long, device=scores.device)
        target_ids = torch.as_tensor(target, dtype=torch.long, device=scores.device)
        # Forced rows contribute exactly zero, not dilution of the gradient.
        denominator = int(choice_rows.sum())
        delta = scores.new_tensor(float((competitor != target).sum()) * margin)
        edge_terms.append(F.relu(((scores[row_ids, rival_ids] - scores[row_ids, target_ids]).sum()
                                  + delta) / denominator))
    for group in groups:
        group = np.asarray(group, dtype=np.int64)
        teacher = target[group]
        actual = predicted[group]
        local = scores[torch.as_tensor(group, dtype=torch.long, device=scores.device)]
        expected_set = set(teacher[teacher > 0].tolist())
        actual_set = set(actual[actual > 0].tolist())
        metrics['device_bc_deploy_set_correct'] += int(expected_set == actual_set)
        if not expected_set:
            metrics['device_bc_matching_empty_groups'] += 1
            has_real = np.isfinite(detached[group, 1:]).any(axis=1)
            meaningful = has_real & (causes[group] == 'temporal_defer')
            if meaningful.any():
                metrics['device_bc_wait_eligible_groups'] += 1
                metrics['device_bc_wait_eligible_rows'] += int(meaningful.sum())
                if wait_enabled:
                    mask = torch.as_tensor(meaningful, dtype=torch.bool, device=scores.device)
                    wait_terms.append(-local[mask, 0].mean())
            elif has_real.any():
                metrics['device_bc_wait_untyped_choice_groups'] += 1
            else:
                metrics['device_bc_wait_structural_groups'] += 1
            continue
        metrics['device_bc_matching_nonempty_groups'] += 1
    zero = scores.new_zeros(())
    edge_loss = torch.stack(edge_terms).mean() if edge_terms else zero
    wait_loss = torch.stack(wait_terms).mean() if wait_terms else zero
    metrics['device_bc_full_edge_groups'] = len(edge_terms)
    metrics['device_bc_empty_wait_loss_groups'] = len(wait_terms)
    metrics['device_bc_full_edge_loss_sum'] = float(edge_loss.detach().cpu()) * len(edge_terms)
    metrics['device_bc_empty_wait_loss_sum'] = float(wait_loss.detach().cpu()) * (
        metrics['device_bc_wait_eligible_groups'] if wait_enabled else 0)
    return edge_loss, wait_loss, metrics


def finalize_matching_metrics(metrics):
    """Use sufficient-statistic denominators, never average empty-step rates."""
    for name, numerator, denominator in (
        ('device_bc_deploy_joint_exact', 'device_bc_deploy_joint_correct', 'device_bc_deploy_environments'),
        ('device_bc_deploy_edge_exact', 'device_bc_deploy_edge_correct', 'device_bc_deploy_rows'),
        ('device_bc_deploy_choice_edge_exact', 'device_bc_deploy_choice_edge_correct', 'device_bc_deploy_choice_rows'),
        ('device_bc_deploy_set_exact', 'device_bc_deploy_set_correct', 'device_bc_matching_groups'),
        ('device_bc_full_edge_loss_group_mean', 'device_bc_full_edge_loss_sum', 'device_bc_full_edge_groups'),
        ('device_bc_empty_wait_loss_group_mean', 'device_bc_empty_wait_loss_sum', 'device_bc_empty_wait_loss_groups'),
    ):
        count = metrics.get(denominator, 0)
        metrics[name] = metrics.get(numerator, 0) / count if count else None
    return metrics
