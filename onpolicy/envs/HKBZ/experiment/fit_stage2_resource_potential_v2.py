#!/usr/bin/env python3
"""Fit a Stage2 resource potential from matched IGA decision-trace replays.

The replay is policy-free: every plane and mobile-resource action is restored
from the verified matched-IGA trace.  Consequently the fitted state features
use exactly the same frontier/ETA/reservation/departure semantics as DeviceBC,
without performing another neural-policy forward or another IGA search.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv  # noqa: E402


FEATURES = tuple(AircraftScheduleEnv.RESOURCE_POTENTIAL_FEATURES)
WORKER_CONFIG: dict | None = None
WORKER_SOURCE_TEACHERS: str | None = None
WORKER_MAX_STEPS = 4000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--teacher-index', required=True)
    parser.add_argument('--env-config', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--trajectory-jsonl', required=True)
    parser.add_argument('--workers', type=int, default=72)
    parser.add_argument('--max-cases', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--ridge', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20260828)
    parser.add_argument('--heldout-fraction', type=float, default=0.20)
    parser.add_argument('--min-heldout-r2', type=float, default=0.75)
    args = parser.parse_args()
    if args.workers < 1 or args.max_steps < 1:
        raise ValueError('workers and max_steps must be positive.')
    if args.ridge < 0.0:
        raise ValueError('ridge must be non-negative.')
    if not 0.05 <= args.heldout_fraction <= 0.50:
        raise ValueError('heldout_fraction must be in [0.05, 0.50].')
    return args


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def init_worker(config: dict, source_teachers: str, max_steps: int) -> None:
    global WORKER_CONFIG, WORKER_SOURCE_TEACHERS, WORKER_MAX_STEPS
    WORKER_CONFIG = config
    WORKER_SOURCE_TEACHERS = source_teachers
    WORKER_MAX_STEPS = int(max_steps)


def _case_config(case_dir: Path) -> dict:
    config = dict(WORKER_CONFIG or {})
    config.update({
        'jobs_path': str(case_dir / 'job.json'),
        'fixed_res_path': str(case_dir / 'fixed_resources.json'),
        'mobile_res_path': str(case_dir / 'mobile_resources.json'),
        'sites_path': str(case_dir / 'sites.json'),
        'flights_path': str(case_dir / 'flights.json'),
        'resource_policy': 'drl',
        'hindsight_reward_mode': 'team_time',
        'hindsight_terminal_cmax_coef': 1.0,
        'iga_potential_beta': 0.0,
        'use_domain_rand': False,
    })
    return config


def _trace_actions(env: AircraftScheduleEnv, step: dict) -> np.ndarray:
    actions = np.full((env.n_agents, 2), -1, dtype=np.int32)
    for decision in step.get('plane_decisions', ()):
        agent_id = int(decision['agent_id'])
        actions[agent_id] = (
            int(decision['operation_index']),
            int(decision['site_index']),
        )
    for decision in step.get('resource_decisions', ()):
        agent_id = int(decision['agent_id'])
        actions[agent_id] = (
            int(decision.get('selected_request_id', 0) or 0),
            0,
        )
    return actions


def replay_case(case_dir_text: str) -> dict:
    case_dir = Path(case_dir_text)
    teacher_path = Path(str(WORKER_SOURCE_TEACHERS)) / f'{case_dir.name}.json'
    teacher = json.loads(teacher_path.read_text(encoding='utf-8'))
    trace = list(teacher.get('decision_trace', ()))
    if not trace:
        raise RuntimeError(f'{teacher_path} has no decision_trace.')
    env = AircraftScheduleEnv(_case_config(case_dir))
    seed = int(teacher.get('seed', 1))
    env.seed(seed)
    env.reset(seed=seed)
    rows = []
    teacher_cmax = float(teacher['makespan'])
    for step_index, step in enumerate(trace):
        if step_index >= WORKER_MAX_STEPS:
            raise RuntimeError(
                f'{case_dir.name} exceeded {WORKER_MAX_STEPS} replay steps.'
            )
        expected_before = float(step['time_before'])
        if not math.isclose(
            float(env.total_time), expected_before, abs_tol=1e-6
        ):
            raise RuntimeError(
                f'{case_dir.name} trace drift before step {step_index}: '
                f'{env.total_time} != {expected_before}.'
            )
        features = env.get_resource_fitted_potential_features()
        if set(features) != set(FEATURES):
            raise RuntimeError('Resource potential feature schema drifted.')
        rows.append({
            'case': case_dir.name,
            'profile': str(teacher.get('profile', 'unknown')),
            'distribution': str(teacher.get('distribution', 'unknown')),
            'step': int(step_index),
            'time_before': expected_before,
            'time_to_go': max(0.0, teacher_cmax - expected_before),
            'features': features,
        })
        env.step(_trace_actions(env, step))
        expected_after = float(step['time_after'])
        if not math.isclose(
            float(env.total_time), expected_after, abs_tol=1e-6
        ):
            raise RuntimeError(
                f'{case_dir.name} trace drift after step {step_index}: '
                f'{env.total_time} != {expected_after}.'
            )
    completed = bool(env.done and env._is_schedule_complete())
    replay_cmax = float(env.total_time)
    if (
        not completed
        or env.cycle_terminated
        or not math.isclose(replay_cmax, teacher_cmax, abs_tol=1e-6)
    ):
        raise RuntimeError(
            f'{case_dir.name} replay failed: completed={completed}, '
            f'cycle={env.cycle_terminated}:{env.cycle_reason}, '
            f'cmax={replay_cmax}/{teacher_cmax}.'
        )
    return {
        'case': case_dir.name,
        'profile': str(teacher.get('profile', 'unknown')),
        'distribution': str(teacher.get('distribution', 'unknown')),
        'makespan': teacher_cmax,
        'steps': len(rows),
        'rows': rows,
    }


def _balanced_row_weights(rows: list[dict]) -> np.ndarray:
    target = {'iid': 0.50, 'ood_stress': 0.45, 'ood_scale': 0.05}
    row_counts = Counter(str(row['case']) for row in rows)
    case_distribution = {
        str(row['case']): str(row['distribution']) for row in rows
    }
    cases_by_distribution = Counter(case_distribution.values())
    unexpected = set(cases_by_distribution) - set(target)
    if unexpected:
        raise RuntimeError(
            f'Unexpected calibration distributions: {cases_by_distribution}.'
        )
    present_mass = sum(target[name] for name in cases_by_distribution)
    target = {
        name: target[name] / present_mass for name in cases_by_distribution
    }
    weights = np.asarray([
        target[str(row['distribution'])]
        / cases_by_distribution[str(row['distribution'])]
        / row_counts[str(row['case'])]
        for row in rows
    ], dtype=np.float64)
    return weights / weights.mean()


def _fit_nonnegative_ridge(
    x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, ridge: float
) -> np.ndarray:
    root_weight = np.sqrt(sample_weight / sample_weight.mean())
    xw = x * root_weight[:, None]
    yw = y * root_weight
    scale = np.sqrt(np.mean(np.square(xw), axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    xs = xw / scale
    gram = xs.T @ xs
    rhs = xs.T @ yw
    lipschitz = max(float(np.linalg.eigvalsh(gram).max()) + ridge, 1e-8)
    fitted = np.zeros(xs.shape[1], dtype=np.float64)
    for _ in range(10000):
        gradient = gram @ fitted - rhs + ridge * fitted
        updated = np.maximum(0.0, fitted - gradient / lipschitz)
        if np.max(np.abs(updated - fitted)) <= 1e-10:
            fitted = updated
            break
        fitted = updated
    weights = fitted / scale
    if not np.any(weights > 0.0):
        raise RuntimeError('Resource potential fit collapsed to all-zero weights.')
    return weights


def _metrics(y: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict:
    normalized = weight / weight.sum()
    mean = float(np.sum(normalized * y))
    residual = prediction - y
    mse = float(np.sum(normalized * np.square(residual)))
    total = float(np.sum(normalized * np.square(y - mean)))
    return {
        'samples': int(len(y)),
        'rmse': math.sqrt(mse),
        'mae': float(np.sum(normalized * np.abs(residual))),
        'r2': float(1.0 - mse / max(total, 1e-8)),
    }


def _heldout_cases(cases: list[str], fraction: float, seed: int) -> set[str]:
    if len(cases) < 2:
        return set()
    scored = sorted(
        cases,
        key=lambda case: hashlib.sha256(
            f'{seed}:{case}'.encode('utf-8')
        ).digest(),
    )
    count = max(1, int(round(len(scored) * fraction)))
    return set(scored[:count])


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    teacher_index_path = Path(args.teacher_index).resolve()
    index = json.loads(teacher_index_path.read_text(encoding='utf-8'))
    source_teachers = Path(index['source_teacher_dir']).resolve()
    with Path(args.env_config).resolve().open('r', encoding='utf-8') as handle:
        env_config = yaml.safe_load(handle) or {}
    env_config.update(dict(index.get('resource_lookahead_contract', {})))
    cases = sorted(path for path in dataset_dir.glob('case_*') if path.is_dir())
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    if not cases:
        raise FileNotFoundError(f'No cases under {dataset_dir}.')
    context = mp.get_context('spawn')
    with context.Pool(
        min(args.workers, len(cases)),
        initializer=init_worker,
        initargs=(env_config, str(source_teachers), args.max_steps),
    ) as pool:
        results = list(pool.imap_unordered(
            replay_case, [str(path) for path in cases], chunksize=1
        ))
    results.sort(key=lambda item: item['case'])
    rows = [row for result in results for row in result['rows']]
    x = np.asarray([
        [float(row['features'][name]) for name in FEATURES]
        for row in rows
    ], dtype=np.float64)
    y = np.asarray([float(row['time_to_go']) for row in rows], dtype=np.float64)
    weights = _balanced_row_weights(rows)
    all_cases = sorted(result['case'] for result in results)
    heldout_cases = _heldout_cases(
        all_cases, args.heldout_fraction, args.seed
    )
    heldout = np.asarray([
        row['case'] in heldout_cases for row in rows
    ], dtype=bool)
    fitted_train = _fit_nonnegative_ridge(
        x[~heldout], y[~heldout], weights[~heldout], args.ridge
    )
    heldout_metrics = (
        _metrics(y[heldout], x[heldout] @ fitted_train, weights[heldout])
        if heldout.any()
        else _metrics(y, x @ fitted_train, weights)
    )
    if heldout.any() and heldout_metrics['r2'] < args.min_heldout_r2:
        raise RuntimeError(
            'Resource Potential V2 heldout gate failed: '
            f"R2={heldout_metrics['r2']:.6f} < {args.min_heldout_r2:.6f}."
        )
    fitted = _fit_nonnegative_ridge(x, y, weights, args.ridge)
    full_metrics = _metrics(y, x @ fitted, weights)
    trajectory_path = Path(args.trajectory_jsonl).resolve()
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = trajectory_path.with_name(
        f'.{trajectory_path.name}.tmp.{os.getpid()}'
    )
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, trajectory_path)
    payload = {
        'schema_version': 1,
        'status': 'calibrated',
        'potential_schema_version': (
            AircraftScheduleEnv.RESOURCE_POTENTIAL_SCHEMA_VERSION
        ),
        'environment_semantics_version': AircraftScheduleEnv.SEMANTICS_VERSION,
        'feature_names': list(FEATURES),
        'weights': {
            name: float(value) for name, value in zip(FEATURES, fitted)
        },
        'calibration': {
            'case_count': len(results),
            'transition_count': len(rows),
            'distribution_counts': dict(Counter(
                result['distribution'] for result in results
            )),
            'profile_counts': dict(Counter(
                result['profile'] for result in results
            )),
            'ridge': float(args.ridge),
            'heldout_fraction': float(args.heldout_fraction),
            'heldout_cases': sorted(heldout_cases),
            'heldout_metrics': heldout_metrics,
            'full_metrics': full_metrics,
        },
        'dataset_dir': str(dataset_dir),
        'teacher_index': str(teacher_index_path),
        'source_teacher_dir': str(source_teachers),
        'trajectory_jsonl': str(trajectory_path),
    }
    atomic_json(Path(args.output_json).resolve(), payload)
    print(json.dumps({
        'status': 'calibrated',
        'cases': len(results),
        'transitions': len(rows),
        'heldout_r2': heldout_metrics['r2'],
        'full_r2': full_metrics['r2'],
        'output': str(Path(args.output_json).resolve()),
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
