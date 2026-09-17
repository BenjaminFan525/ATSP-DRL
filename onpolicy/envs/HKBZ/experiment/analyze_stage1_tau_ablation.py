#!/usr/bin/env python
"""Aggregate the fixed-vs-annealed Stage-1 tau pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


WEIGHTS = {'iid': 0.50, 'ood_stress': 0.45, 'ood_scale': 0.05}
VARIANTS = ('fixed', 'annealed')
SEEDS = (1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-dir', type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + '\n',
    )


def distribution_summary(cases: list[dict]) -> dict:
    groups = defaultdict(list)
    for case in cases:
        groups[case['distribution']].append(float(case['makespan']))
    result = {
        name: {
            'n': len(groups[name]),
            'mean_makespan': float(np.mean(groups[name])),
        }
        for name in WEIGHTS
    }
    result['composite'] = float(sum(
        weight * result[name]['mean_makespan']
        for name, weight in WEIGHTS.items()
    ))
    return result


def scalar_map(accumulator: EventAccumulator, tag: str) -> dict[int, float]:
    return {int(item.step): float(item.value) for item in accumulator.Scalars(tag)}


def training_metrics(run_dir: Path) -> dict:
    accumulator = EventAccumulator(
        str(run_dir / 'logs'), size_guidance={'scalars': 0}
    )
    accumulator.Reload()
    shard_by_step = scalar_map(accumulator, 'global_shard_index')
    metric_tags = (
        'dist_entropy', 'actor_step_completion_rate',
        'post_update_probe_approx_kl', 'post_update_probe_entropy',
        'old_policy_kl_stop_events', 'actor_kl_stop_events',
    )
    rows = defaultdict(dict)
    for item in accumulator.Scalars('policy_tau'):
        shard = shard_by_step.get(int(item.step))
        if shard is not None:
            # Training metrics are logged before the optional epoch-end
            # evaluation, which may emit another policy_tau=0.3 at this step.
            rows[int(round(shard))].setdefault('policy_tau', float(item.value))
    for tag in metric_tags:
        for item in accumulator.Scalars(tag):
            shard = shard_by_step.get(int(item.step))
            if shard is not None:
                rows[int(round(shard))][tag] = float(item.value)
    ordered = [rows[index] for index in sorted(rows)]

    def phase(start: int, end: int) -> dict:
        selected = [rows[index] for index in sorted(rows) if start <= index <= end]
        result = {'shards': len(selected)}
        for tag in metric_tags:
            values = [row[tag] for row in selected if tag in row and math.isfinite(row[tag])]
            result[f'{tag}_mean'] = float(np.mean(values)) if values else None
            result[f'{tag}_max'] = float(np.max(values)) if values else None
        return result

    epoch_evals = []
    for path in sorted(
            (run_dir / 'evaluations').glob('epoch_*.json'),
            key=lambda item: int(item.stem.split('_')[1])):
        artifact = read_json(path)
        summary = artifact['summary']
        epoch_evals.append({
            'epoch': int(path.stem.split('_')[1]),
            'composite': float(summary['eval_composite_makespan']),
            'iid': float(summary['eval_distribution_iid_makespan']),
            'ood_stress': float(summary['eval_distribution_ood_stress_makespan']),
            'ood_scale': float(summary['eval_distribution_ood_scale_makespan']),
            'completion_rate': float(summary['eval_completion_rate']),
            'cycle_count': int(summary['eval_cycle_count']),
        })
    return {
        'shard_count': len(ordered),
        'early_epochs_1_2': phase(1, 16),
        'late_epochs_3_4': phase(17, 32),
        'observed_tau_by_epoch': [
            float(np.mean([
                rows[index]['policy_tau']
                for index in range(8 * epoch + 1, 8 * epoch + 9)
                if 'policy_tau' in rows[index]
            ]))
            for epoch in range(4)
        ],
        'epoch_evaluations': epoch_evals,
        'late_validation_range': float(
            max(row['composite'] for row in epoch_evals[-2:])
            - min(row['composite'] for row in epoch_evals[-2:])
        ),
    }


def bootstrap_delta(case_rows: list[dict], iterations=30000) -> list[float]:
    rng = np.random.default_rng(20260807)
    groups = {
        name: np.array([
            row['delta_annealed_minus_fixed']
            for row in case_rows if row['distribution'] == name
        ], dtype=float)
        for name in WEIGHTS
    }
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        samples[index] = sum(
            weight * rng.choice(values, size=len(values), replace=True).mean()
            for name, weight in WEIGHTS.items()
            for values in (groups[name],)
        )
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    records = {}
    evaluations = {}
    metrics = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            key = f'{variant}_seed{seed}'
            record_path = args.suite_dir / 'records' / f'{key}.json'
            if not record_path.is_file():
                raise FileNotFoundError(record_path)
            record = read_json(record_path)
            if record.get('status') != 'completed':
                raise RuntimeError(f'Incomplete record: {record_path}')
            records[key] = record
            evaluation = read_json(Path(record['heldout_evaluation']))
            drl = evaluation['methods']['DRL-G']
            evaluations[key] = {
                'summary': drl['summary'],
                'distribution': distribution_summary(drl['cases']),
                'cases': drl['cases'],
            }
            metrics[key] = training_metrics(Path(record['run_dir']))

    seed_comparisons = []
    for seed in SEEDS:
        fixed = evaluations[f'fixed_seed{seed}']['distribution']
        annealed = evaluations[f'annealed_seed{seed}']['distribution']
        delta = annealed['composite'] - fixed['composite']
        seed_comparisons.append({
            'seed': seed,
            'fixed_composite': fixed['composite'],
            'annealed_composite': annealed['composite'],
            'delta_annealed_minus_fixed': delta,
            'relative_delta': delta / fixed['composite'],
            'fixed_ood_stress': fixed['ood_stress']['mean_makespan'],
            'annealed_ood_stress': annealed['ood_stress']['mean_makespan'],
        })

    case_maps = {}
    for variant in VARIANTS:
        values = defaultdict(list)
        metadata = {}
        for seed in SEEDS:
            for case in evaluations[f'{variant}_seed{seed}']['cases']:
                key = case['case_sha256']
                values[key].append(float(case['makespan']))
                metadata[key] = case
        case_maps[variant] = {
            key: float(np.mean(case_values))
            for key, case_values in values.items()
        }
        if variant == 'fixed':
            case_metadata = metadata
    shared_cases = sorted(set(case_maps['fixed']) & set(case_maps['annealed']))
    case_rows = []
    for key in shared_cases:
        fixed = case_maps['fixed'][key]
        annealed = case_maps['annealed'][key]
        metadata = case_metadata[key]
        case_rows.append({
            'case_sha256': key,
            'case_id': metadata['case_id'],
            'profile': metadata['profile'],
            'distribution': metadata['distribution'],
            'fixed_mean': fixed,
            'annealed_mean': annealed,
            'delta_annealed_minus_fixed': annealed - fixed,
        })

    early_entropy_fixed = float(np.mean([
        metrics[f'fixed_seed{seed}']['early_epochs_1_2']['dist_entropy_mean']
        for seed in SEEDS
    ]))
    early_entropy_annealed = float(np.mean([
        metrics[f'annealed_seed{seed}']['early_epochs_1_2']['dist_entropy_mean']
        for seed in SEEDS
    ]))
    late_completion = float(np.mean([
        metrics[f'annealed_seed{seed}']['late_epochs_3_4'][
            'actor_step_completion_rate_mean'
        ] for seed in SEEDS
    ]))
    late_kl_max = float(max(
        metrics[f'annealed_seed{seed}']['late_epochs_3_4'][
            'post_update_probe_approx_kl_max'
        ] for seed in SEEDS
    ))
    fixed_mean = float(np.mean([
        row['fixed_composite'] for row in seed_comparisons
    ]))
    annealed_mean = float(np.mean([
        row['annealed_composite'] for row in seed_comparisons
    ]))
    mean_relative_improvement = (fixed_mean - annealed_mean) / fixed_mean
    ood_stress_delta = float(np.mean([
        row['annealed_ood_stress'] - row['fixed_ood_stress']
        for row in seed_comparisons
    ]))
    heldout_healthy = all(
        evaluations[key]['summary']['completion_rate'] == 1.0
        and evaluations[key]['summary']['cycle_count'] == 0
        for key in evaluations
    )
    scheduled_tune_healthy = all(
        row['completion_rate'] == 1.0 and row['cycle_count'] == 0
        for seed in SEEDS
        for row in metrics[f'annealed_seed{seed}']['epoch_evaluations']
    )
    scheduled_tau_observed = all(
        np.allclose(
            metrics[f'annealed_seed{seed}']['observed_tau_by_epoch'],
            [0.5, 0.4, 0.3, 0.3],
            rtol=0.0,
            atol=1e-6,
        )
        for seed in SEEDS
    )
    gates = {
        'early_entropy_increase_ge_10pct': (
            early_entropy_annealed >= 1.10 * early_entropy_fixed
        ),
        'heldout_composite_improvement_ge_0_5pct': (
            mean_relative_improvement >= 0.005
        ),
        'each_seed_regression_le_0_3pct': all(
            row['relative_delta'] <= 0.003 for row in seed_comparisons
        ),
        'ood_stress_no_regression': ood_stress_delta <= 0.0,
        'late_actor_step_completion_ge_90pct': late_completion >= 0.90,
        'late_old_policy_kl_le_target': late_kl_max <= 0.005,
        'heldout_completion_and_cycles_healthy': heldout_healthy,
        'tune_completion_and_cycles_healthy': scheduled_tune_healthy,
        'scheduled_tau_observed_exactly': scheduled_tau_observed,
    }
    payload = {
        'schema_version': 1,
        'status': 'completed',
        'fixed_composite_mean': fixed_mean,
        'annealed_composite_mean': annealed_mean,
        'delta_annealed_minus_fixed': annealed_mean - fixed_mean,
        'mean_relative_improvement': mean_relative_improvement,
        'paired_case_bootstrap_95ci': bootstrap_delta(case_rows),
        'early_entropy_fixed': early_entropy_fixed,
        'early_entropy_annealed': early_entropy_annealed,
        'early_entropy_relative_change': (
            early_entropy_annealed / early_entropy_fixed - 1.0
        ),
        'late_actor_step_completion': late_completion,
        'late_post_update_old_policy_kl_max': late_kl_max,
        'ood_stress_delta_annealed_minus_fixed': ood_stress_delta,
        'seed_comparisons': seed_comparisons,
        'training_metrics': metrics,
        'gates': gates,
        'passed_all_gates': all(gates.values()),
        'cases': case_rows,
        'created_unix_time': time.time(),
    }
    analysis_dir = args.suite_dir / 'analysis'
    atomic_json(analysis_dir / 'tau_ablation_summary.json', payload)

    lines = [
        '# Stage1 tau 调度消融结果', '',
        'Cmax 越低越好；blind test 未使用。', '',
        '| 指标 | Fixed 0.3 | 0.5→0.4→0.3→0.3 |',
        '|---|---:|---:|',
        f'| held-out composite | {fixed_mean:.2f} | {annealed_mean:.2f} |',
        f'| 前两 epoch entropy | {early_entropy_fixed:.6f} | {early_entropy_annealed:.6f} |',
        '',
        f'- Annealed−Fixed Cmax：{annealed_mean - fixed_mean:+.2f}',
        f'- 相对改善：{100.0 * mean_relative_improvement:.3f}%',
        f'- 配对案例 bootstrap 95% CI：{payload["paired_case_bootstrap_95ci"]}',
        f'- 后两 epoch actor step completion：{late_completion:.4f}',
        f'- 后两 epoch 最大 old-policy KL：{late_kl_max:.6g}',
        '', '## Seed 结果', '',
        '| Seed | Fixed | Annealed | Annealed−Fixed |',
        '|---:|---:|---:|---:|',
    ]
    for row in seed_comparisons:
        lines.append(
            f'| {row["seed"]} | {row["fixed_composite"]:.2f} | '
            f'{row["annealed_composite"]:.2f} | '
            f'{row["delta_annealed_minus_fixed"]:+.2f} |'
        )
    lines.extend(['', '## 预注册门槛', ''])
    for name, passed in gates.items():
        lines.append(f'- {name}: {"PASS" if passed else "FAIL"}')
    lines.extend([
        '',
        f'总判定：{"PASS" if payload["passed_all_gates"] else "FAIL"}',
        '',
    ])
    atomic_text(analysis_dir / 'tau_ablation_summary.md', '\n'.join(lines))
    print(json.dumps({
        key: payload[key] for key in (
            'fixed_composite_mean', 'annealed_composite_mean',
            'delta_annealed_minus_fixed', 'early_entropy_relative_change',
            'passed_all_gates',
        )
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
