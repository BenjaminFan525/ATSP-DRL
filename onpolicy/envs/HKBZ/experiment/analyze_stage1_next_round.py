#!/usr/bin/env python
"""Aggregate the multi-seed next-round screen and sealed heldout results."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


SELECTION_WEIGHTS = {'iid': 0.50, 'ood_stress': 0.45, 'ood_scale': 0.05}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('screen', 'formal'), required=True)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=(1, 2))
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260808)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def case_key(row: dict) -> str:
    return str(row.get('case_key') or row.get('case_id') or row.get('case_path'))


def composite(records: list[dict], value_key='makespan') -> float:
    groups = {}
    for distribution, weight in SELECTION_WEIGHTS.items():
        values = [
            float(row[value_key]) for row in records
            if row.get('distribution') == distribution
        ]
        if not values:
            raise ValueError(f'Missing {distribution} cases in evaluation.')
        groups[distribution] = float(np.mean(values))
    return float(sum(SELECTION_WEIGHTS[name] * groups[name] for name in groups))


def screen_record(suite_dir: Path, variant: str, seed: int) -> dict:
    path = suite_dir / 'records' / f'screen_{variant}_seed{seed}.json'
    record = read_json(path)
    if record.get('status') != 'completed':
        raise RuntimeError(f'Incomplete trial record: {path}')
    best = read_json(Path(record['evaluation_json']))
    pre = read_json(Path(record['pre_ppo_evaluation_json']))
    best_rows = {case_key(row): row for row in best['cases']}
    pre_rows = {case_key(row): row for row in pre['cases']}
    if best_rows.keys() != pre_rows.keys() or len(best_rows) != 60:
        raise RuntimeError(f'Paired validation coverage mismatch: {path}')
    paired = []
    for key in sorted(best_rows):
        row = best_rows[key]
        paired.append({
            'case_key': key,
            'distribution': row['distribution'],
            'best': float(row['makespan']),
            'pre': float(pre_rows[key]['makespan']),
            'delta': float(row['makespan']) - float(pre_rows[key]['makespan']),
        })
    summary = best['summary']
    health = record['evaluation']['actor_update_health']
    return {
        'variant': variant,
        'seed': seed,
        'record_path': str(path.resolve()),
        'selection_score': float(summary['eval_selection_score']),
        'pre_selection_score': composite([
            {**row, 'makespan': row['pre']} for row in paired
        ]),
        'relative_improvement': (
            composite([{**row, 'makespan': row['pre']} for row in paired])
            - composite([{**row, 'makespan': row['best']} for row in paired])
        ) / composite([{**row, 'makespan': row['pre']} for row in paired]),
        'ood_stress_delta': float(np.mean([
            row['delta'] for row in paired
            if row['distribution'] == 'ood_stress'
        ])),
        'completion_rate': float(summary['eval_completion_rate']),
        'cycle_count': int(summary['eval_cycle_count']),
        'timeout_count': int(summary['eval_timeout_count']),
        'actor_step_completion_rate': float(health['step_completion_rate']),
        'actor_zero_update_shards': int(health['zero_update_shards']),
        'paired_cases': paired,
    }


def hierarchical_bootstrap(trials: list[dict], samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_trials = rng.integers(0, len(trials), size=len(trials))
        seed_deltas = []
        for trial_index in sampled_trials:
            rows = trials[int(trial_index)]['paired_cases']
            distribution_delta = {}
            for distribution in SELECTION_WEIGHTS:
                pool = [
                    row['delta'] for row in rows
                    if row['distribution'] == distribution
                ]
                picks = rng.integers(0, len(pool), size=len(pool))
                distribution_delta[distribution] = float(np.mean(
                    np.asarray(pool, dtype=np.float64)[picks]
                ))
            seed_deltas.append(sum(
                SELECTION_WEIGHTS[name] * distribution_delta[name]
                for name in SELECTION_WEIGHTS
            ))
        values[index] = float(np.mean(seed_deltas))
    return {
        'mean_delta_best_minus_pre': float(values.mean()),
        'ci95': [float(value) for value in np.quantile(values, [0.025, 0.975])],
        'probability_improvement': float(np.mean(values < 0.0)),
    }


def analyze_screen(args: argparse.Namespace) -> dict:
    variants = []
    for variant_index, variant in enumerate(args.variants):
        trials = [screen_record(args.suite_dir, variant, seed) for seed in args.seeds]
        bootstrap = hierarchical_bootstrap(
            trials,
            args.bootstrap_samples,
            args.bootstrap_seed + variant_index * 1009,
        )
        safety_pass = all(
            trial['completion_rate'] == 1.0
            and trial['cycle_count'] == 0
            and trial['timeout_count'] == 0
            and trial['actor_step_completion_rate'] >= 0.90
            and trial['actor_zero_update_shards'] == 0
            for trial in trials
        )
        mean_improvement = float(np.mean([
            trial['relative_improvement'] for trial in trials
        ]))
        worst_seed_regression = float(max(
            -trial['relative_improvement'] for trial in trials
        ))
        mean_ood_delta = float(np.mean([
            trial['ood_stress_delta'] for trial in trials
        ]))
        efficacy_pass = (
            mean_improvement >= 0.01
            and worst_seed_regression <= 0.005
            and mean_ood_delta <= 0.0
            and bootstrap['probability_improvement'] >= 0.90
        )
        variants.append({
            'variant': variant,
            'seed_results': [{
                key: value for key, value in trial.items()
                if key != 'paired_cases'
            } for trial in trials],
            'mean_selection_score': float(np.mean([
                trial['selection_score'] for trial in trials
            ])),
            'selection_score_std': float(np.std([
                trial['selection_score'] for trial in trials
            ], ddof=1)),
            'mean_relative_improvement': mean_improvement,
            'worst_seed_regression': worst_seed_regression,
            'mean_ood_stress_delta': mean_ood_delta,
            'bootstrap': bootstrap,
            'safety_pass': safety_pass,
            'efficacy_pass': efficacy_pass,
            'promotion_pass': bool(safety_pass and efficacy_pass),
        })
    ranked = sorted(
        variants,
        key=lambda row: (
            not row['promotion_pass'], row['mean_selection_score'], row['variant']
        ),
    )
    promoted = [
        row['variant'] for row in ranked if row['promotion_pass']
    ][:2]
    return {
        'schema_version': 1,
        'mode': 'screen',
        'created_unix_time': time.time(),
        'selection_weights': SELECTION_WEIGHTS,
        'hard_gates': {
            'completion_rate': 1.0,
            'cycle_count': 0,
            'timeout_count': 0,
            'actor_step_completion_rate_min': 0.90,
            'zero_update_shards': 0,
        },
        'efficacy_gates': {
            'two_seed_mean_improvement_min': 0.01,
            'worst_seed_regression_max': 0.005,
            'mean_ood_stress_delta_max': 0.0,
            'bootstrap_probability_improvement_min': 0.90,
        },
        'variants': ranked,
        'promoted_variants': promoted,
        'formal_authorized': len(promoted) == 2,
        'stop_reason': (
            None if len(promoted) == 2
            else 'Fewer than two methods passed the pre-registered screen gates.'
        ),
    }


def heldout_record(suite_dir: Path, variant: str, seed: int) -> dict:
    path = suite_dir / 'heldout' / f'{variant}_seed{seed}.json'
    payload = read_json(path)
    evaluation = payload['evaluation']
    rows = list(evaluation['records'])
    return {
        'variant': variant,
        'seed': seed,
        'path': str(path.resolve()),
        'score': composite(rows),
        'raw_makespan': float(evaluation['raw_makespan']),
        'completion_rate': float(evaluation['completion_rate']),
        'cycle_count': int(evaluation['cycle_count']),
        'timeout_count': int(evaluation['timeout_count']),
    }


def analyze_formal(args: argparse.Namespace) -> dict:
    variants = []
    for variant in args.variants:
        trials = [heldout_record(args.suite_dir, variant, seed) for seed in args.seeds]
        safety = all(
            trial['completion_rate'] == 1.0
            and trial['cycle_count'] == 0
            and trial['timeout_count'] == 0
            for trial in trials
        )
        variants.append({
            'variant': variant,
            'seed_results': trials,
            'mean_heldout_composite': float(np.mean([
                trial['score'] for trial in trials
            ])),
            'std_heldout_composite': float(np.std([
                trial['score'] for trial in trials
            ], ddof=1)),
            'mean_heldout_raw_makespan': float(np.mean([
                trial['raw_makespan'] for trial in trials
            ])),
            'safety_pass': safety,
        })
    ranked = sorted(
        variants,
        key=lambda row: (
            not row['safety_pass'], row['mean_heldout_composite'], row['variant']
        ),
    )
    winner = ranked[0]['variant'] if ranked and ranked[0]['safety_pass'] else None
    return {
        'schema_version': 1,
        'mode': 'formal_heldout',
        'created_unix_time': time.time(),
        'selection_weights': SELECTION_WEIGHTS,
        'variants': ranked,
        'winner': winner,
        'blind_test_authorized': winner is not None,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError('--bootstrap-samples must be positive.')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be unique.')
    result = analyze_screen(args) if args.mode == 'screen' else analyze_formal(args)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
