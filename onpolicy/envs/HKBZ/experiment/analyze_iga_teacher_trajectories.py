#!/usr/bin/env python
"""Replay verified IGA teachers and calibrate a dense scheduling potential."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv


FEATURES = tuple(AircraftScheduleEnv.IGA_POTENTIAL_FEATURES)
WORKER_ENV_CONFIG: dict | None = None
WORKER_TEACHER_DIR: str | None = None
WORKER_MAX_STEPS = 4000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--teacher_dir', required=True)
    parser.add_argument('--env_config', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--trajectory_jsonl', required=True)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=4000)
    parser.add_argument('--max_cases', type=int, default=0)
    parser.add_argument('--ridge', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--reuse_trajectory_jsonl', default='',
        help='reuse verified transition rows instead of replaying every teacher',
    )
    parser.add_argument(
        '--source_analysis_json', default='',
        help='source replay metadata required with --reuse_trajectory_jsonl',
    )
    parser.add_argument(
        '--distribution_weights',
        default='iid=0.50,ood_stress=0.45,ood_scale=0.05',
        help=(
            'target distribution mass for case-balanced potential fitting'
        ),
    )
    parser.add_argument(
        '--profile_balance', action='store_true', default=False,
        help='give every profile equal mass inside each target distribution',
    )
    parser.add_argument('--tail_start_fraction', type=float, default=1.0)
    parser.add_argument('--tail_weight', type=float, default=1.0)
    parser.add_argument('--final_tail_start_fraction', type=float, default=1.0)
    parser.add_argument('--final_tail_weight', type=float, default=-1.0)
    parser.add_argument('--cv_folds', type=int, default=5)
    args = parser.parse_args()
    if args.workers <= 0 or args.max_steps <= 0:
        raise ValueError('workers and max_steps must be positive.')
    if args.ridge < 0.0:
        raise ValueError('ridge must be non-negative.')
    if not 0.0 <= args.tail_start_fraction <= 1.0:
        raise ValueError('tail_start_fraction must be in [0, 1].')
    if not args.tail_start_fraction <= args.final_tail_start_fraction <= 1.0:
        raise ValueError(
            'final_tail_start_fraction must be between tail_start_fraction and 1.'
        )
    if args.tail_weight < 1.0:
        raise ValueError('tail_weight must be at least 1.')
    if args.final_tail_weight < 0.0:
        args.final_tail_weight = args.tail_weight
    if args.final_tail_weight < args.tail_weight:
        raise ValueError('final_tail_weight must be at least tail_weight.')
    if args.cv_folds < 2:
        raise ValueError('cv_folds must be at least 2.')
    if bool(args.reuse_trajectory_jsonl) != bool(args.source_analysis_json):
        raise ValueError(
            'reuse_trajectory_jsonl and source_analysis_json must be supplied together.'
        )
    return args


def parse_distribution_weights(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(spec or '').split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(
                f'Distribution weight must use name=value, got {item!r}.'
            )
        name, value_text = (part.strip() for part in item.split('=', 1))
        if not name or name in weights:
            raise ValueError(f'Invalid or duplicate distribution {name!r}.')
        value = float(value_text)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f'Distribution weight {name!r} must be finite and positive.'
            )
        weights[name] = value
    if not weights:
        raise ValueError('At least one distribution weight is required.')
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def init_worker(env_config: dict, teacher_dir: str, max_steps: int) -> None:
    global WORKER_ENV_CONFIG, WORKER_TEACHER_DIR, WORKER_MAX_STEPS
    WORKER_ENV_CONFIG = env_config
    WORKER_TEACHER_DIR = teacher_dir
    WORKER_MAX_STEPS = int(max_steps)


def case_config(case_dir: Path) -> dict:
    config = dict(WORKER_ENV_CONFIG or {})
    config.update({
        'jobs_path': str(case_dir / 'job.json'),
        'fixed_res_path': str(case_dir / 'fixed_resources.json'),
        'mobile_res_path': str(case_dir / 'mobile_resources.json'),
        'sites_path': str(case_dir / 'sites.json'),
        'flights_path': str(case_dir / 'flights.json'),
        'iga_teacher_dir': str(WORKER_TEACHER_DIR),
        'hindsight_reward_mode': 'potential_cmax',
        'hindsight_cmax_coef': 1.0,
        'hindsight_shaping_coef': 0.0,
        'hindsight_terminal_cmax_coef': 0.0,
        'use_domain_rand': False,
        'resource_policy': 'heuristic',
    })
    return config


def action_diagnostics(env: AircraftScheduleEnv, actions: np.ndarray) -> dict:
    active = 0
    relocation = 0
    same_site = 0
    service_actions = 0
    staging_hold_actions = 0
    staging_move_actions = 0
    forced_departure_actions = 0
    trainable_pair_actions = 0
    legal_pairs = []
    for plane in env.planes.values():
        pid = int(plane.code.split('_')[-1])
        if (
            pid >= env.n_plane_agents
            or not plane.is_idle()
            or plane.is_completed_all_jobs()
            or int(actions[pid, 0]) < 0
            or int(actions[pid, 1]) < 0
        ):
            continue
        active += 1
        site_idx = int(actions[pid, 1])
        job_idx = int(actions[pid, 0])
        local_job_idx = (
            job_idx % len(env.job_code_list)
            if job_idx >= 0 and env.job_code_list else -1
        )
        job_code = (
            env.job_code_list[local_job_idx]
            if local_job_idx >= 0 else ''
        )
        current_idx = env.site_code_list.index(plane.site.code)
        if site_idx == current_idx:
            same_site += 1
        else:
            relocation += 1
        pair_count = int(
            np.asarray(env.agent_job_site_mask_matrix[pid]).sum()
        )
        legal_pairs.append(pair_count)
        trainable_pair_actions += int(pair_count > 1)
        if job_code == env.TRANSFER_JOB_CODE:
            if site_idx == current_idx:
                staging_hold_actions += 1
            else:
                staging_move_actions += 1
        elif job_code in set(env.departure_job_code_list):
            forced_departure_actions += 1
        else:
            service_actions += 1
    return {
        'active_actions': active,
        'relocations': relocation,
        'same_site_actions': same_site,
        'service_actions': service_actions,
        'staging_hold_actions': staging_hold_actions,
        'staging_move_actions': staging_move_actions,
        'forced_departure_actions': forced_departure_actions,
        'trainable_pair_actions': trainable_pair_actions,
        'mean_legal_pairs': float(np.mean(legal_pairs)) if legal_pairs else 0.0,
    }


def replay_case(case_dir_text: str) -> dict:
    case_dir = Path(case_dir_text)
    teacher_path = Path(str(WORKER_TEACHER_DIR)) / f'{case_dir.name}.json'
    teacher = json.loads(teacher_path.read_text(encoding='utf-8'))
    teacher_cmax = float(teacher['makespan'])
    env = AircraftScheduleEnv(case_config(case_dir))
    env.seed(int(teacher.get('seed', 1)))
    env.reset(seed=int(teacher.get('seed', 1)))
    rows = []
    steps = 0
    while not env.done and steps < WORKER_MAX_STEPS:
        features_before = env.get_iga_potential_features()
        time_before = float(env.total_time)
        teacher_step = env.iga_teacher_actions(return_info=True)
        if not teacher_step['info'].get('available', False):
            raise RuntimeError(f'No verified IGA teacher for {case_dir.name}.')
        actions = np.asarray(teacher_step['actions'], dtype=np.int32)
        diagnostics = action_diagnostics(env, actions)
        env.step(actions[:, :2])
        features_after = env.get_iga_potential_features()
        rows.append({
            'case': case_dir.name,
            'step': steps,
            'time_before': time_before,
            'time_after': float(env.total_time),
            'time_to_go': max(0.0, teacher_cmax - time_before),
            'before': features_before,
            'after': features_after,
            **diagnostics,
        })
        steps += 1
    completed = bool(env.done and env._is_schedule_complete())
    replay_cmax = float(env.total_time)
    stored_cmax_match = abs(replay_cmax - teacher_cmax) <= 1e-6
    if not completed or env.cycle_terminated:
        raise RuntimeError(
            f'IGA replay failed {case_dir.name}: completed={completed}, '
            f'teacher={teacher_cmax}, replay={replay_cmax}, steps={steps}, '
            f'cycle={env.cycle_terminated}:{env.cycle_reason}'
        )
    for row in rows:
        row['time_to_go'] = max(0.0, replay_cmax - float(row['time_before']))
        row['progress'] = float(
            np.clip(float(row['time_before']) / max(replay_cmax, 1e-8), 0.0, 1.0)
        )
    metadata_path = case_dir / 'metadata.json'
    metadata = (
        json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata_path.is_file() else {}
    )
    return {
        'case': case_dir.name,
        'profile': metadata.get('profile', teacher.get('profile', 'unknown')),
        'distribution': metadata.get(
            'distribution', teacher.get('distribution', 'unknown')
        ),
        'teacher_cmax': teacher_cmax,
        'replay_cmax': replay_cmax,
        'steps': steps,
        'reverified': True,
        'stored_cmax_match': stored_cmax_match,
        'cmax_drift': replay_cmax - teacher_cmax,
        'relative_cmax_drift': (replay_cmax - teacher_cmax) / max(teacher_cmax, 1e-8),
        'rows': rows,
    }


def fit_nonnegative_ridge(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Projected-gradient NNLS ridge fit with feature scaling and zero intercept."""
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != y.shape:
            raise ValueError(
                f'sample_weight shape {sample_weight.shape} != {y.shape}.'
            )
        if (
            not np.all(np.isfinite(sample_weight))
            or np.any(sample_weight <= 0.0)
        ):
            raise ValueError('Every calibration sample weight must be positive.')
        sample_weight = sample_weight / sample_weight.mean()
        root_weight = np.sqrt(sample_weight)
        x = x * root_weight[:, None]
        y = y * root_weight
    scale = np.sqrt(np.mean(np.square(x), axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    xs = x / scale
    gram = xs.T @ xs
    rhs = xs.T @ y
    lipschitz = max(float(np.linalg.eigvalsh(gram).max()) + ridge, 1e-8)
    weights_scaled = np.zeros(xs.shape[1], dtype=np.float64)
    for _ in range(5000):
        gradient = gram @ weights_scaled - rhs + ridge * weights_scaled
        updated = np.maximum(0.0, weights_scaled - gradient / lipschitz)
        if np.max(np.abs(updated - weights_scaled)) <= 1e-10:
            weights_scaled = updated
            break
        weights_scaled = updated
    weights = weights_scaled / scale
    if not np.any(weights > 0.0):
        remaining_index = FEATURES.index('remaining_work')
        denominator = float(x[:, remaining_index] @ x[:, remaining_index] + ridge)
        weights[remaining_index] = max(
            1e-8, float(x[:, remaining_index] @ y) / max(denominator, 1e-8)
        )
    return weights


def regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict:
    residual = prediction - y
    if sample_weight is None:
        normalized_weight = np.full(len(y), 1.0 / max(len(y), 1))
    else:
        normalized_weight = np.asarray(sample_weight, dtype=np.float64)
        normalized_weight = normalized_weight / normalized_weight.sum()
    target_mean = float(np.sum(normalized_weight * y))
    prediction_mean = float(np.sum(normalized_weight * prediction))
    mse = float(np.sum(normalized_weight * np.square(residual)))
    mae = float(np.sum(normalized_weight * np.abs(residual)))
    ss_total = float(
        np.sum(normalized_weight * np.square(y - target_mean))
    )
    ss_residual = float(
        np.sum(normalized_weight * np.square(residual))
    )
    return {
        'samples': int(len(y)),
        'rmse': float(np.sqrt(mse)),
        'mae': mae,
        'r2': float(1.0 - ss_residual / max(ss_total, 1e-8)),
        'target_mean': target_mean,
        'prediction_mean': prediction_mean,
        'weighted': sample_weight is not None,
    }


def case_balanced_row_weights(
    rows: list[dict],
    distribution_weights: dict[str, float],
    *,
    profile_balance: bool = False,
) -> np.ndarray:
    """Give each case equal mass inside a requested distribution mixture."""
    case_distribution = {}
    row_count_by_case = Counter()
    for row in rows:
        case = str(row['case'])
        distribution = str(row['distribution'])
        previous = case_distribution.setdefault(case, distribution)
        if previous != distribution:
            raise ValueError(
                f'Case {case} has inconsistent distributions '
                f'{previous!r}/{distribution!r}.'
            )
        row_count_by_case[case] += 1
    cases_by_distribution = Counter(case_distribution.values())
    unexpected = sorted(set(cases_by_distribution) - set(distribution_weights))
    missing = sorted(set(distribution_weights) - set(cases_by_distribution))
    if unexpected or missing:
        raise ValueError(
            'Calibration distribution mismatch: '
            f'unexpected={unexpected}, missing={missing}.'
        )
    if profile_balance:
        case_profile = {}
        profiles_by_distribution: dict[str, set[str]] = {}
        cases_by_distribution_profile = Counter()
        for row in rows:
            case = str(row['case'])
            profile = str(row['profile'])
            distribution = str(row['distribution'])
            previous = case_profile.setdefault(case, profile)
            if previous != profile:
                raise ValueError(
                    f'Case {case} has inconsistent profiles {previous!r}/{profile!r}.'
                )
            profiles_by_distribution.setdefault(distribution, set()).add(profile)
        for case, distribution in case_distribution.items():
            cases_by_distribution_profile[(distribution, case_profile[case])] += 1
        weights = np.asarray([
            distribution_weights[str(row['distribution'])]
            / len(profiles_by_distribution[str(row['distribution'])])
            / cases_by_distribution_profile[
                (str(row['distribution']), str(row['profile']))
            ]
            / row_count_by_case[str(row['case'])]
            for row in rows
        ], dtype=np.float64)
    else:
        weights = np.asarray([
            distribution_weights[str(row['distribution'])]
            / cases_by_distribution[str(row['distribution'])]
            / row_count_by_case[str(row['case'])]
            for row in rows
        ], dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError('Invalid case-balanced calibration weights.')
    return weights / weights.mean()


def row_progress(rows: list[dict]) -> np.ndarray:
    progress = []
    for row in rows:
        if 'progress' in row:
            value = float(row['progress'])
        else:
            before = float(row['time_before'])
            value = before / max(before + float(row['time_to_go']), 1e-8)
        progress.append(float(np.clip(value, 0.0, 1.0)))
    return np.asarray(progress, dtype=np.float64)


def temporal_tail_weights(
    progress: np.ndarray,
    *,
    tail_start: float,
    tail_weight: float,
    final_start: float,
    final_weight: float,
) -> np.ndarray:
    progress = np.asarray(progress, dtype=np.float64)
    if tail_start >= 1.0 or tail_weight <= 1.0:
        return np.ones_like(progress)
    first_ramp = np.clip(
        (progress - tail_start) / max(final_start - tail_start, 1e-12),
        0.0,
        1.0,
    )
    weights = 1.0 + (tail_weight - 1.0) * first_ramp
    if final_start < 1.0:
        final_ramp = np.clip(
            (progress - final_start) / max(1.0 - final_start, 1e-12),
            0.0,
            1.0,
        )
        weights += (final_weight - tail_weight) * final_ramp
    return weights


def phase_regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    progress: np.ndarray,
) -> dict:
    phases = {
        'all': np.ones(len(y), dtype=bool),
        'pre_tail_0_75': progress < 0.75,
        'tail_0_75_0_90': (progress >= 0.75) & (progress < 0.90),
        'final_tail_0_90_1_00': progress >= 0.90,
    }
    return {
        name: regression_metrics(y[mask], prediction[mask])
        for name, mask in phases.items()
        if np.any(mask)
    }


def departure_phase_regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    rows: list[dict],
) -> dict:
    service_remaining = np.asarray([
        float(row.get('before', {}).get('remaining_service_jobs', 0.0))
        for row in rows
    ])
    departure_remaining = np.asarray([
        float(row.get('before', {}).get('remaining_departure_jobs', 0.0))
        for row in rows
    ])
    ready_count = np.asarray([
        float(row.get('before', {}).get('departure_ready_count', 0.0))
        for row in rows
    ])
    phases = {
        'service_active': service_remaining > 0.0,
        'departure_only': (
            (service_remaining <= 0.0) & (departure_remaining > 0.0)
        ),
        'departure_queue_nonempty': ready_count > 0.0,
    }
    return {
        name: regression_metrics(y[mask], prediction[mask])
        for name, mask in phases.items()
        if np.any(mask)
    }


def case_level_cross_validation(
    rows: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    row_weights: np.ndarray,
    *,
    ridge: float,
    folds: int,
    seed: int,
) -> dict:
    """Profile-stratified folds prevent transitions from one case leaking."""
    cases_by_profile: dict[str, list[str]] = {}
    for row in rows:
        cases_by_profile.setdefault(str(row['profile']), []).append(str(row['case']))
    assignments = {}
    rng = np.random.default_rng(int(seed))
    for profile, repeated_cases in sorted(cases_by_profile.items()):
        cases = sorted(set(repeated_cases))
        rng.shuffle(cases)
        for index, case in enumerate(cases):
            assignments[case] = index % folds
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    fold_records = []
    row_cases = np.asarray([str(row['case']) for row in rows], dtype=object)
    for fold in range(folds):
        test_mask = np.asarray(
            [assignments[case] == fold for case in row_cases], dtype=bool
        )
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise RuntimeError(f'Empty calibration CV fold {fold}.')
        fold_weights = fit_nonnegative_ridge(
            x[train_mask], y[train_mask], ridge,
            sample_weight=row_weights[train_mask],
        )
        predictions[test_mask] = x[test_mask] @ fold_weights
        fold_records.append({
            'fold': fold,
            'train_cases': len(set(row_cases[train_mask])),
            'test_cases': len(set(row_cases[test_mask])),
            **regression_metrics(y[test_mask], predictions[test_mask]),
        })
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError('Case-level CV left non-finite predictions.')
    progress = row_progress(rows)
    return {
        'folds': folds,
        'stratified_by': 'profile',
        'case_disjoint': True,
        'metrics': regression_metrics(y, predictions),
        'phase_metrics': phase_regression_metrics(y, predictions, progress),
        'departure_phase_metrics': departure_phase_regression_metrics(
            y, predictions, rows
        ),
        'fold_records': fold_records,
    }


def grouped_summary(results: list[dict], key: str) -> dict:
    groups: dict[str, list[float]] = {}
    for result in results:
        groups.setdefault(str(result[key]), []).append(float(result['replay_cmax']))
    return {
        group: {
            'cases': len(values),
            'mean_cmax': float(np.mean(values)),
            'std_cmax': float(np.std(values)),
        }
        for group, values in sorted(groups.items())
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    dataset_dir = Path(args.dataset_dir).resolve()
    teacher_dir = Path(args.teacher_dir).resolve()
    env_config_path = Path(args.env_config).resolve()
    output_path = Path(args.output_json).resolve()
    trajectory_path = Path(args.trajectory_jsonl).resolve()
    source_payload = None
    reused_trajectory = bool(args.reuse_trajectory_jsonl)
    if reused_trajectory:
        source_path = Path(args.source_analysis_json).resolve()
        trajectory_path = Path(args.reuse_trajectory_jsonl).resolve()
        source_payload = json.loads(source_path.read_text(encoding='utf-8'))
        if source_payload.get('status') != 'completed':
            raise RuntimeError(f'Incomplete source analysis: {source_path}.')
        if (
            tuple(source_payload.get('feature_names', ())) != FEATURES
            or source_payload.get('environment_semantics_version')
            != AircraftScheduleEnv.SEMANTICS_VERSION
        ):
            raise RuntimeError(
                'Reused trajectories do not match the current departure-aware '
                'feature schema and environment semantics; replay is required.'
            )
        rows = [
            json.loads(line)
            for line in trajectory_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        results = list(source_payload.get('cases', []))
        replay_payload = dict(source_payload.get('replay', {}))
        action_payload = dict(source_payload.get('action_diagnostics', {}))
        if not rows or not results or not replay_payload:
            raise RuntimeError(
                'Reused trajectory analysis is missing rows, cases, or replay metadata.'
            )
        print(
            f'[IGA trajectory] reusing {len(rows)} verified rows from '
            f'{trajectory_path}.',
            flush=True,
        )
    else:
        with env_config_path.open('r', encoding='utf-8') as handle:
            env_config = yaml.safe_load(handle) or {}

        case_dirs = sorted(
            path for path in dataset_dir.glob('case_*') if path.is_dir()
        )
        if args.max_cases > 0:
            case_dirs = case_dirs[:args.max_cases]
        if not case_dirs:
            raise RuntimeError(f'No cases in {dataset_dir}.')
        missing = [
            case.name for case in case_dirs
            if not (teacher_dir / f'{case.name}.json').is_file()
        ]
        if missing:
            raise RuntimeError(
                f'Missing {len(missing)} teacher files: {missing[:10]}'
            )

        context = mp.get_context('spawn')
        with context.Pool(
            processes=min(args.workers, len(case_dirs)),
            initializer=init_worker,
            initargs=(env_config, str(teacher_dir), args.max_steps),
        ) as pool:
            results = []
            iterator = pool.imap_unordered(
                replay_case, map(str, case_dirs), chunksize=1
            )
            for completed, result in enumerate(iterator, start=1):
                results.append(result)
                if (
                    completed == 1
                    or completed % 10 == 0
                    or completed == len(case_dirs)
                ):
                    print(
                        f'[IGA trajectory] replay progress '
                        f'{completed}/{len(case_dirs)}',
                        flush=True,
                    )
        results.sort(key=lambda item: item['case'])
        if (
            len(results) != len(case_dirs)
            or not all(item['reverified'] for item in results)
        ):
            raise RuntimeError(
                'Not every requested IGA teacher replay completed safely.'
            )

        rows = []
        for result in results:
            for row in result.pop('rows'):
                row['profile'] = result['profile']
                row['distribution'] = result['distribution']
                rows.append(row)
        replay_payload = {
            'expected_cases': len(case_dirs),
            'reverified_cases': len(results),
            'stored_cmax_match_cases': sum(
                item['stored_cmax_match'] for item in results
            ),
            'transition_count': len(rows),
            'stored_mean_cmax': float(np.mean([
                item['teacher_cmax'] for item in results
            ])),
            'stored_std_cmax': float(np.std([
                item['teacher_cmax'] for item in results
            ])),
            'replay_mean_cmax': float(np.mean([
                item['replay_cmax'] for item in results
            ])),
            'replay_std_cmax': float(np.std([
                item['replay_cmax'] for item in results
            ])),
            'mean_relative_cmax_drift': float(np.mean([
                item['relative_cmax_drift'] for item in results
            ])),
            'max_abs_cmax_drift': float(max(
                abs(item['cmax_drift']) for item in results
            )),
            'profiles': grouped_summary(results, 'profile'),
            'distributions': grouped_summary(results, 'distribution'),
        }
        active_actions = sum(int(row['active_actions']) for row in rows)
        relocations = sum(int(row['relocations']) for row in rows)
        same_site = sum(int(row['same_site_actions']) for row in rows)
        phase_count_keys = (
            'service_actions',
            'staging_hold_actions',
            'staging_move_actions',
            'forced_departure_actions',
            'trainable_pair_actions',
        )
        action_payload = {
            'active_actions': active_actions,
            'relocations': relocations,
            'relocation_rate': float(relocations / max(active_actions, 1)),
            'same_site_actions': same_site,
            'same_site_rate': float(same_site / max(active_actions, 1)),
            'mean_legal_pairs_per_transition': float(
                np.mean([row['mean_legal_pairs'] for row in rows])
            ),
            **{
                key: sum(int(row.get(key, 0)) for row in rows)
                for key in phase_count_keys
            },
        }
    x = np.asarray(
        [[float(row['before'][name]) for name in FEATURES] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([float(row['time_to_go']) for row in rows], dtype=np.float64)
    target_distribution_weights = parse_distribution_weights(
        args.distribution_weights
    )
    row_weights = case_balanced_row_weights(
        rows,
        target_distribution_weights,
        profile_balance=args.profile_balance,
    )
    progress = row_progress(rows)
    tail_weights = temporal_tail_weights(
        progress,
        tail_start=args.tail_start_fraction,
        tail_weight=args.tail_weight,
        final_start=args.final_tail_start_fraction,
        final_weight=args.final_tail_weight,
    )
    row_weights = row_weights * tail_weights
    row_weights = row_weights / row_weights.mean()
    weights_array = fit_nonnegative_ridge(
        x, y, args.ridge, sample_weight=row_weights
    )
    prediction = x @ weights_array
    weights = {
        name: float(value) for name, value in zip(FEATURES, weights_array)
    }

    cross_validation = case_level_cross_validation(
        rows,
        x,
        y,
        row_weights,
        ridge=args.ridge,
        folds=args.cv_folds,
        seed=args.seed,
    )

    if not reused_trajectory:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_trajectory = trajectory_path.with_name(
            f'{trajectory_path.name}.tmp.{os.getpid()}'
        )
        with temporary_trajectory.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_trajectory, trajectory_path)

    payload = {
        'schema_version': 3,
        'potential_schema_version': (
            AircraftScheduleEnv.IGA_POTENTIAL_SCHEMA_VERSION
        ),
        'environment_semantics_version': (
            AircraftScheduleEnv.SEMANTICS_VERSION
        ),
        'status': 'completed',
        'dataset_dir': str(dataset_dir),
        'teacher_dir': str(teacher_dir),
        'env_config': str(env_config_path),
        'feature_names': list(FEATURES),
        'weights': weights,
        'calibration': {
            **regression_metrics(y, prediction, row_weights),
            'ridge': float(args.ridge),
            'zero_intercept': True,
            'nonnegative': True,
            'case_balanced': True,
            'profile_balanced': bool(args.profile_balance),
            'distribution_weights': target_distribution_weights,
            'temporal_weights': {
                'tail_start_fraction': float(args.tail_start_fraction),
                'tail_weight': float(args.tail_weight),
                'final_tail_start_fraction': float(
                    args.final_tail_start_fraction
                ),
                'final_tail_weight': float(args.final_tail_weight),
            },
            'unweighted_metrics': regression_metrics(y, prediction),
            'phase_metrics': phase_regression_metrics(y, prediction, progress),
            'departure_phase_metrics': departure_phase_regression_metrics(
                y, prediction, rows
            ),
            'case_level_cross_validation': cross_validation,
        },
        'replay': replay_payload,
        'action_diagnostics': action_payload,
        'cases': results,
        'trajectory_jsonl': str(trajectory_path),
        'reused_verified_trajectories': reused_trajectory,
        'source_analysis_json': (
            str(Path(args.source_analysis_json).resolve())
            if reused_trajectory else None
        ),
        'elapsed_seconds': float(time.time() - started),
        'seed': int(args.seed),
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in weights.values()):
        raise RuntimeError(f'Invalid calibrated weights: {weights}')
    atomic_json(output_path, payload)
    print(
        f"[IGA trajectory] reverified={replay_payload['reverified_cases']}/"
        f"{replay_payload['expected_cases']} "
        f"stored_match={replay_payload['stored_cmax_match_cases']}/"
        f"{replay_payload['expected_cases']} "
        f"transitions={len(rows)} weights={weights} "
        f"rmse={payload['calibration']['rmse']:.3f}",
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
