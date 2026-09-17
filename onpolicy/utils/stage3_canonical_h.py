"""Opt-in deterministic H decoder: exact raw objectives and physical-ID ties.

This is an evaluation contract, not a PPO likelihood or training policy. Positive
temperature scales and row log-normalizers cancel between complete matchings
with the same Blocking coverage. Comparing the binary raw scores directly avoids
softmax rounding. No epsilon, decimal quantization, or random tie perturbation is
used. Training and historical evaluation keep their existing default contract.
"""
from __future__ import annotations

import numpy as np


CONTRACT = 'h_exact_raw_blocking_device_request_lex_v1'
SPEC = {
    'contract': CONTRACT,
    'primary': 'maximum_blocking_coverage',
    'secondary': 'exact_sum_of_binary_pre_temperature_scores',
    'tertiary': 'device_id_ascending_then_request_id_ascending_with_noop_last',
    'plane': 'raw_joint_argmax_then_first_flat_operation_site_index',
    'probability': 'float64_log_softmax_then_cast_to_policy_output_dtype',
    'score_epsilon': None,
}


def _integer_assignment(cost):
    """Rectangular shortest-augmenting-path Hungarian with Python integers."""
    n, m = len(cost), len(cost[0])
    if n > m:
        raise ValueError('Assignment needs at least as many columns as rows')
    u, v, p, way = [0]*(n+1), [0]*(m+1), [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        distance, used = [None]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], None, 0
            row = cost[i0-1]
            for j in range(1, m+1):
                if used[j]:
                    continue
                current = row[j-1]-u[i0]-v[j]
                if distance[j] is None or current < distance[j]:
                    distance[j], way[j] = current, j0
                if delta is None or distance[j] < delta:
                    delta, j1 = distance[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                elif distance[j] is not None:
                    distance[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = [0]*n
    for j in range(1, m+1):
        if p[j]:
            result[p[j]-1] = j-1
    return result


def solve_canonical_matching(raw_scores, request_is_lookahead, *, device_ids,
                             request_ids=None):
    """Return column indices with lexicographic Blocking/raw/physical tie goals.

    Column zero is the private no-op choice. Device and request IDs are stable
    identities within the case, independent of matrix row/column order. The
    returned indices still refer to the caller's original matrix ordering.
    """
    scores = np.asarray(raw_scores)
    look = np.asarray(request_is_lookahead, dtype=bool)
    devices = list(device_ids)
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError('Expected device by request scores including no-op')
    n, width = scores.shape
    requests = list(range(width)) if request_ids is None else list(request_ids)
    if (look.shape != (width,) or len(devices) != n or len(set(devices)) != n
            or len(requests) != width or len(set(requests)) != width):
        raise ValueError('Matching identities or lookahead mask differ')
    if np.isnan(scores).any() or np.isposinf(scores).any() or not np.isfinite(scores[:, 0]).all():
        raise ValueError('Scores must be finite legal values with a legal private no-op')
    if not n:
        return np.empty(0, dtype=np.int64)
    order = sorted(range(n), key=lambda i: devices[i])
    # Drop globally illegal real columns; every private no-op remains available.
    real = sorted((j for j in range(1, width) if np.isfinite(scores[:, j]).any()),
                  key=lambda j: requests[j])
    choices = real + [0]
    fractions = {(i, j): float(scores[i, j]).as_integer_ratio()
                 for i in order for j in choices if np.isfinite(scores[i, j])}
    denominator = max(d for _, d in fractions.values())
    raw = {k: a*(denominator//b) for k, (a, b) in fractions.items()}
    bound = max(abs(x) for x in raw.values())
    blocking_unit = 2*n*bound+1
    base = len(real)+1
    tie_unit = base**n
    ranks = {j: rank for rank, j in enumerate(choices)}
    weights = {}
    for row, i in enumerate(order):
        place = base**(n-row-1)
        for j in choices:
            if (i, j) not in raw:
                continue
            objective = raw[i, j] + (blocking_unit if j and not look[j] else 0)
            column = real.index(j) if j else len(real)+row
            weights[row, column] = objective*tie_unit-ranks[j]*place
    maximum, minimum = max(weights.values()), min(weights.values())
    forbidden = (n+1)*(maximum-minimum+1)
    cost = [[forbidden]*(len(real)+n) for _ in range(n)]
    for (i, j), weight in weights.items():
        cost[i][j] = maximum-weight
    columns = _integer_assignment(cost)
    result = np.zeros(n, dtype=np.int64)
    for row, column in enumerate(columns):
        if (row, column) not in weights:
            raise RuntimeError('Exact matching selected an illegal edge')
        result[order[row]] = real[column] if column < len(real) else 0
    return result


def enable_canonical_h(policy):
    """Enable only supported deterministic heads; no parameter/buffer mutation."""
    from onpolicy.algorithms.utils.ptr_actor import JointPairPtrActor, DeviceRequestPtrActor
    ac = policy.ac
    if not isinstance(ac.actor, JointPairPtrActor):
        raise ValueError('Canonical H requires the joint operation/site actor')
    heads = [x for x in ac.modules() if isinstance(x, DeviceRequestPtrActor)]
    if not heads or any(x.timing_head is not None for x in heads):
        raise ValueError('Canonical raw matching excludes temperature-dependent timing gates')
    for head in [ac.actor, *heads]:
        head.canonical_h_decode = True
    ac.canonical_h_decode = True
    return dict(SPEC)
