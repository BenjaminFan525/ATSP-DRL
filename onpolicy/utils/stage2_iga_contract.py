"""Hash-paired Stage2 IGA diagnostics, without blessing historical budgets.

Historical tune references may screen a method, but cannot certify the new
IGA goal. Independent confirmation and runtime/search-budget validation are
separate prerequisites, not consequences of a favorable point estimate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


GROUPS = ('iid', 'ood_stress', 'ood_scale')


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def checked_cases(evaluation):
    rows = evaluation['cases'] if isinstance(evaluation, dict) else evaluation
    result = {}
    for row in rows:
        key = row.get('case_sha256')
        value = float(row.get('makespan', float('nan')))
        if (not key or key in result or not np.isfinite(value) or value <= 0
                or not row.get('completed') or row.get('cycle_terminated') or row.get('timeout')):
            raise ValueError('Missing, duplicate, incomplete or invalid paired evaluation case.')
        if row.get('distribution') not in GROUPS:
            raise ValueError('Unknown evaluation distribution.')
        result[key] = dict(row)
    if not result:
        raise ValueError('Empty paired evaluation.')
    return result


def load_historical_iga(directory, *, source_sha256, planning_contract, nominal_budget):
    directory = Path(directory).resolve()
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if (summary.get('status') != 'completed'
            or summary.get('resource_lookahead_contract') != planning_contract
            or summary.get('nominal_cumulative_budget_seconds') != nominal_budget):
        raise ValueError('Historical IGA summary has a different planning/budget contract.')
    files = {str(summary_path): file_digest(summary_path)}
    rows, searches = [], []
    for path in sorted((directory / 'cases').glob('case_*.json')):
        data = json.loads(path.read_text())
        completion = data.get('completion', {})
        search = data.get('search', {})
        replay = search.get('replay_verification', {})
        if (data.get('status') != 'completed' or data.get('completion_verified') is not True
                or data.get('frozen_plane_checkpoint_sha256') != source_sha256
                or data.get('resource_lookahead_contract') != planning_contract
                or data.get('nominal_cumulative_budget_seconds') != nominal_budget
                or replay.get('status') != 'matched_search_incumbent'):
            raise ValueError(f'Invalid historical IGA source or replay: {path}')
        if float(replay.get('trace_replay_makespan', float('nan'))) != float(data['makespan']):
            raise ValueError('Historical IGA replay Cmax differs from selected result.')
        rows.append({
            'case_sha256': data['case_sha256'], 'case_dir': data['case'],
            'makespan': data['makespan'], 'distribution': data['distribution'],
            'profile': data['profile'], 'completed': completion.get('completed', False),
            'cycle_terminated': completion.get('cycle_terminated', False),
            'timeout': completion.get('timeout', False),
        })
        searches.append({key: search.get(key) for key in (
            'configured_additional_budget_seconds', 'nominal_cumulative_budget_seconds',
            'optimization_wall_seconds', 'budget_overshoot_seconds', 'evaluated_candidates',
            'completed_candidates', 'inherited_verified_candidates', 'parallel_case_count',
            'warm_start', 'search_contract_version',
        )})
        files[str(path)] = file_digest(path)
    checked = checked_cases(rows)
    if len(checked) != summary['case_count']:
        raise ValueError('Historical IGA case count differs from its summary.')
    wall = np.array([float(row['optimization_wall_seconds']) for row in searches])
    budget = np.array([float(row['configured_additional_budget_seconds']) for row in searches])
    if not np.isfinite(wall).all() or not np.isfinite(budget).all() or np.any(budget <= 0):
        raise ValueError('Invalid IGA timing metadata.')
    if nominal_budget == 1800 and any(
        not row['warm_start'] or row['warm_start'].get('source_nominal_budget_seconds') != 180
        or row['configured_additional_budget_seconds'] != 1620 for row in searches
    ):
        raise ValueError('IGA1800 is not the declared nested 180 + 1620 budget.')
    return {
        'cases': list(checked.values()), 'source_files': files,
        'nominal_budget_seconds': nominal_budget,
        'source_sha256': source_sha256, 'planning_contract': planning_contract,
        'provenance_checked': True, 'current_runtime_replayed': False,
        'strict_budget_validated': False,
        'usage': 'historical_tune_reference_only',
        'budget_audit': {
            'case_count': len(rows), 'mean_additional_wall_seconds': float(wall.mean()),
            'max_additional_wall_seconds': float(wall.max()),
            'over_nominal_count': int((wall > budget).sum()),
            'maximum_overshoot_seconds': float(np.maximum(wall - budget, 0).max()),
            'mean_evaluated_candidates': float(np.mean([row['evaluated_candidates'] for row in searches])),
            'searches': searches,
            'finding': 'Historical v4 exempts the first feasible incumbent from the deadline; initialization and final replay are outside search timing.',
        },
    }


def distribution_metrics(evaluation):
    cases = list(checked_cases(evaluation).values())
    values = np.array([row['makespan'] for row in cases], dtype=float)
    result = {'raw': float(values.mean()),
              'tail10': float(np.sort(values)[-max(1, int(np.ceil(.1 * len(values)))):].mean())}
    for group in GROUPS:
        selected = [row['makespan'] for row in cases if row['distribution'] == group]
        result[group] = float(np.mean(selected)) if selected else None
    return result


def paired_iga_comparison(evaluation, baseline, *, samples=10000, seed=20260906):
    left, right = checked_cases(evaluation), checked_cases(baseline)
    if set(left) != set(right):
        raise ValueError('IGA comparison requires exactly the same case hashes.')
    keys = sorted(left)
    for key in keys:
        if any(left[key].get(field) != right[key].get(field) for field in ('distribution', 'profile')):
            raise ValueError('Paired IGA cases have different distribution/profile labels.')
    a = np.array([left[key]['makespan'] for key in keys], dtype=float)
    b = np.array([right[key]['makespan'] for key in keys], dtype=float)
    differences = a - b
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        indices = rng.integers(0, len(keys), (stop - start, len(keys)))
        boot[start:stop] = differences[indices].mean(axis=1)
    left_metrics, right_metrics = distribution_metrics(evaluation), distribution_metrics(baseline)
    group_deltas = {group: None if left_metrics[group] is None or right_metrics[group] is None
                    else left_metrics[group] - right_metrics[group] for group in (*GROUPS, 'tail10')}
    profile_deltas = {}
    for profile in sorted({str(row.get('profile')) for row in left.values()}):
        indices = [i for i, key in enumerate(keys) if str(left[key].get('profile')) == profile]
        profile_deltas[profile] = {'count': len(indices), 'mean_delta_seconds': float(differences[indices].mean())}
    return {
        'case_count': len(keys), 'mean_delta_seconds': float(differences.mean()),
        'relative_gap_fraction': float(a.mean() / b.mean() - 1),
        'wins': int((differences < -1e-6).sum()), 'ties': int((np.abs(differences) <= 1e-6).sum()),
        'losses': int((differences > 1e-6).sum()), 'max_regression_seconds': float(differences.max()),
        'strict_positive_delta_count': int((differences > 0).sum()),
        'regressed_cases': [{'case_sha256': key, 'delta_seconds': float(differences[i])}
                            for i, key in enumerate(keys) if differences[i] > 0],
        'case_bootstrap_95ci_seconds': np.quantile(boot, [.025, .975]).tolist(),
        'group_deltas_seconds': group_deltas,
        'profile_deltas_seconds': profile_deltas,
        'all_group_tail_point_nonworse': all(value is not None and value <= 0 for value in group_deltas.values()),
        'uncertainty_scope': 'Exploratory case bootstrap conditional on selected model and training seed; no multiplicity/selection correction.',
    }


def iga_target_diagnostics(evaluation, baselines, *, samples=10000):
    comparisons = {name: paired_iga_comparison(evaluation, baseline, samples=samples)
                   for name, baseline in baselines.items()}
    short, long = comparisons['IGA180'], comparisons['IGA1800']
    return {
        'comparisons': comparisons,
        'historical_iga180_point_requirement_met': bool(short['mean_delta_seconds'] < 0
            and short['strict_positive_delta_count'] == 0 and short['all_group_tail_point_nonworse']),
        'historical_iga1800_point_requirement_met': bool(long['mean_delta_seconds'] <= 0
            and long['all_group_tail_point_nonworse']),
        'confirmed_scientific_success': False,
        'confirmation_blockers': ['tune_selected_pilot', 'single_training_seed',
                                  'historical_search_budget_not_strictly_validated',
                                  'independent_confirmation_not_run'],
    }
