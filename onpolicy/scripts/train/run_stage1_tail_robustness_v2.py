#!/usr/bin/env python
"""Run one statically isolated lane of the Stage-1 tail-robustness v2 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
PYTHON = (ROOT.parent / 'conda/envs/maia-hkbz-cu124-20260903/bin/python').resolve()
TRAIN = ROOT / 'onpolicy/scripts/train/train_hkbz.py'
COMPARE = ROOT / 'onpolicy/envs/HKBZ/experiment/valid_fjsp_v2_comparison.py'
AC_CONFIG = ROOT / 'onpolicy/config/ac.yaml'
ENV_CONFIG = ROOT / 'onpolicy/config/env_plane_pretrain.yaml'
RESULT_ROOT = ROOT / 'onpolicy/scripts/results/HKBZ/simple/gnn_mappo'
PROFILE_CVAR_WEIGHTS = (
    'balanced=0.125,resource_sparse=0.10,bursty=0.10,'
    'high_flex=0.075,coupled=0.05,light=0.05,'
    'stress_joint=0.15,stress_arrival=0.20,resource_ood=0.10,'
    'low_load_ood=0.05'
)
TARGET_DISTRIBUTION_WEIGHTS = 'iid=0.50,ood_stress=0.45,ood_scale=0.05'
FORMAL_SAFE_ASYNC_GRAPH_CLONE_WORKERS = 4
ACTIVE_PROCESS: subprocess.Popen | None = None


def variants() -> dict[str, dict]:
    common = {
        'train_sampling_mode': 'distribution_balanced',
        'train_sampling_weights': TARGET_DISTRIBUTION_WEIGHTS,
        'new_bc': False,
        'new_potential': False,
    }
    return {
        'N0_r3_reproduction': {**common},
        'N1_tail_cv_potential': {**common, 'new_potential': True},
        'N2_safe_tail_dagger': {**common, 'new_bc': True},
        'N3_dagger_tail_cv': {
            **common, 'new_bc': True, 'new_potential': True,
        },
        'N4_dagger_tail_cv_cvar': {
            **common,
            'new_bc': True,
            'new_potential': True,
            'train_sampling_mode': 'profile_balanced',
            'train_sampling_weights': PROFILE_CVAR_WEIGHTS,
        },
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def relocate(value: str) -> str:
    return (
        str(value)
        .replace('/home/fanyx/HKBZ-environment', str(ROOT))
        .replace('/home/fanyx/conda/envs/maia-hkbz-cu124-20260903-hkbz-cu124-20260903/bin/python', str(PYTHON))
    )


def remove_option(command: list[str], flag: str, *, value: bool = True) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + (2 if value else 1)]


def set_option(command: list[str], flag: str, value) -> None:
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def set_switch(command: list[str], flag: str, enabled: bool) -> None:
    remove_option(command, flag, value=False)
    if enabled:
        command.append(flag)


def configure_safe_pipeline(command: list[str], *, enabled: bool) -> None:
    """Enable or disable the semantics-preserving throughput pipeline."""
    set_option(
        command,
        '--safe_async_graph_clone_workers',
        FORMAL_SAFE_ASYNC_GRAPH_CLONE_WORKERS if enabled else 0,
    )
    set_switch(command, '--safe_graph_batch_pipeline', enabled)
    set_switch(command, '--safe_dagger_teacher_overlap', enabled)


def model_digest(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location='cpu')
    digest = hashlib.sha256()
    for name, tensor in sorted(payload['model'].items()):
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def latest_checkpoint(experiment: str, filename: str) -> Path | None:
    candidates = sorted(
        (RESULT_ROOT / experiment).glob(f'run*/models/{filename}'),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def completed_training_checkpoint(experiment: str) -> Path | None:
    """Reuse a Best checkpoint only when its run atomically reports completion."""
    run_dirs = sorted(
        (RESULT_ROOT / experiment).glob('run*'),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for run_dir in run_dirs:
        checkpoint = run_dir / 'models/checkpoint_Best.pt'
        status_path = run_dir / 'run_status.json'
        if not checkpoint.is_file() or not status_path.is_file():
            continue
        try:
            status = read_json(status_path)
            payload = torch.load(checkpoint, map_location='cpu')
        except (
            OSError, ValueError, TypeError, EOFError, RuntimeError,
            json.JSONDecodeError,
        ):
            continue
        if status.get('status') != 'completed' or 'model' not in payload:
            continue
        return checkpoint
    return None


def interrupted_training_checkpoint(experiment: str) -> Path | None:
    """Find the newest valid post-shard recovery point from an unfinished run."""
    run_dirs = sorted(
        (RESULT_ROOT / experiment).glob('run*'),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for run_dir in run_dirs:
        checkpoint_path = run_dir / 'models/checkpoint_Recovery.pt'
        status_path = run_dir / 'run_status.json'
        if not checkpoint_path.is_file() or not status_path.is_file():
            continue
        try:
            status = read_json(status_path)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except (
            OSError, ValueError, TypeError, EOFError, RuntimeError,
            json.JSONDecodeError,
        ):
            continue
        if status.get('status') == 'completed':
            continue
        if checkpoint.get('stage') != 'post_shard_recovery':
            continue
        completed_shards = int(checkpoint.get('completed_shard', 0))
        total_shards = int(checkpoint.get('total_shards', 0))
        if not 0 < completed_shards <= total_shards:
            continue
        return checkpoint_path
    return None


def configure_exact_recovery_resume(
    command: list[str],
    checkpoint_path: Path,
) -> dict:
    """Point a generated command at a verified exact Stage-1 recovery cursor."""
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('stage') != 'post_shard_recovery':
        raise ValueError(f'Not a post-shard recovery checkpoint: {checkpoint_path}.')
    completed_shards = int(checkpoint.get('completed_shard', 0))
    total_shards = int(checkpoint.get('total_shards', 0))
    if not 0 < completed_shards <= total_shards:
        raise ValueError(
            'Invalid recovery cursor: '
            f'completed={completed_shards}, total={total_shards}.'
        )
    set_option(command, '--checkpoint_dir', checkpoint_path)
    set_option(command, '--plane_bc_pretrain_epochs', 0)
    remove_option(command, '--selection_checkpoint_dir')
    set_switch(command, '--resume_stage1', True)
    set_switch(command, '--reset_optimizers_on_resume', False)
    return {
        'checkpoint': str(checkpoint_path),
        'epoch': int(checkpoint.get('episodes', 0)),
        'completed_shards': completed_shards,
        'total_shards': total_shards,
        'total_num_steps': int(checkpoint.get('total_num_steps', 0)),
    }


def checkpoint_health(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location='cpu')
    health = dict(payload.get('actor_update_health', {}))
    return {
        'step_completion_rate': float(
            health.get('step_completion_rate', 1.0)
        ),
        'zero_update_shards': int(health.get('zero_update_shards', 0)),
        'old_policy_kl_stop_shards': int(
            health.get('old_policy_kl_stop_shards', 0)
        ),
        'canary_rejected': checkpoint.name == 'checkpoint_CanaryRejected.pt',
    }


def parse_seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f'Formal seeds must be comma-separated integers: {value!r}.'
        ) from error
    if not seeds or any(seed <= 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            'Formal seeds must contain at least one positive integer.'
        )
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError('Formal seeds must not contain duplicates.')
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', type=int, choices=(0, 1), required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument(
        '--status-id',
        default='',
        help='optional unique status-file suffix for parallel formal workers',
    )
    parser.add_argument('--run_tag', required=True)
    parser.add_argument('--suite_dir', type=Path, required=True)
    parser.add_argument('--source_command_json', type=Path, required=True)
    parser.add_argument('--old_bc_checkpoint', type=Path, required=True)
    parser.add_argument('--old_potential', type=Path, required=True)
    parser.add_argument('--new_potential', type=Path, required=True)
    parser.add_argument('--teacher_dir', type=Path, required=True)
    parser.add_argument('--validation_dir', type=Path, required=True)
    parser.add_argument('--blind_test_dir', type=Path, required=True)
    parser.add_argument('--screen_epochs', type=int, default=2)
    parser.add_argument('--formal_epochs', type=int, default=8)
    parser.add_argument(
        '--formal-only-variant',
        choices=('N1_tail_cv_potential', 'N2_safe_tail_dagger'),
        help=(
            'Skip screening and selection, then run the requested candidate as '
            'an explicitly post-selection formal extension.'
        ),
    )
    parser.add_argument(
        '--formal-seeds',
        type=parse_seed_list,
        default=(1, 2, 3),
        help='Comma-separated seeds for --formal-only-variant (default: 1,2,3).',
    )
    parser.add_argument('--partition_seed', type=int, default=20260803)
    parser.add_argument('--barrier_timeout', type=float, default=604800.0)
    parser.add_argument(
        '--start-delay-seconds',
        type=float,
        default=0.0,
        help='delay this isolated formal worker to stagger dataset/GPU startup',
    )
    parser.add_argument('--shared-eval-socket', default='')
    parser.add_argument('--shared-eval-cpu-set', default='')
    parser.add_argument(
        '--safe_pipeline_all_phases',
        action='store_true',
        help=(
            'Enable the verified graph batching, graph-clone overlap, and '
            'DAgger teacher-overlap paths for screening/audit as well as formal.'
        ),
    )
    args = parser.parse_args()
    if args.screen_epochs <= 0 or args.formal_epochs <= 0:
        raise ValueError('screen/formal epochs must be positive.')
    if args.start_delay_seconds < 0.0:
        raise ValueError('--start-delay-seconds must be non-negative.')
    if bool(args.shared_eval_socket) != bool(args.shared_eval_cpu_set):
        raise ValueError(
            '--shared-eval-socket and --shared-eval-cpu-set must be supplied '
            'together.'
        )
    if args.status_id and not all(
        character.isalnum() or character in ('-', '_')
        for character in args.status_id
    ):
        raise ValueError('--status-id may contain only letters, digits, - and _.')
    for name in (
        'source_command_json', 'old_bc_checkpoint', 'old_potential',
        'new_potential', 'teacher_dir', 'validation_dir', 'blind_test_dir',
    ):
        value = Path(getattr(args, name)).resolve()
        if not value.exists():
            raise FileNotFoundError(value)
        setattr(args, name, value)
    args.suite_dir = args.suite_dir.resolve()
    return args


def training_command(
    args: argparse.Namespace,
    variant_id: str,
    seed: int,
    *,
    formal: bool,
    audit_canary: bool = False,
) -> tuple[list[str], str]:
    variant = variants()[variant_id]
    phase = 'formal' if formal else 'screen'
    experiment = f'{args.run_tag}_{phase}_{variant_id}_seed{seed}'
    if audit_canary:
        experiment = f'{args.run_tag}_audit_N0_canary_on_seed{seed}'
    source = read_json(args.source_command_json)['command']
    command = [relocate(item) for item in source]
    command[0] = str(PYTHON)
    command[1] = str(TRAIN)
    set_option(command, '--experiment_name', experiment)
    set_option(command, '--ac_config', AC_CONFIG)
    set_option(command, '--env_config', ENV_CONFIG)
    set_option(command, '--seed', seed)
    set_option(command, '--num_episodes', args.formal_epochs if formal else args.screen_epochs)
    set_option(command, '--max_eval_cases', 60)
    set_option(command, '--eval_dataset_dir', args.validation_dir)
    set_option(command, '--eval_case_offset', 0)
    set_option(command, '--eval_partition_seed', args.partition_seed)
    set_option(command, '--eval_partition_stratify_by', 'profile')
    set_option(command, '--early_stop_patience', 4 if formal else 0)
    if args.shared_eval_socket:
        set_option(command, '--shared_eval_socket', args.shared_eval_socket)
        set_option(command, '--shared_eval_cpu_set', args.shared_eval_cpu_set)
    else:
        remove_option(command, '--shared_eval_socket')
        remove_option(command, '--shared_eval_cpu_set')
    set_option(command, '--train_sampling_mode', variant['train_sampling_mode'])
    set_option(command, '--train_sampling_weights', variant['train_sampling_weights'])
    set_option(command, '--train_sampling_size', 0)
    configure_safe_pipeline(
        command,
        enabled=bool(formal or args.safe_pipeline_all_phases),
    )
    set_option(
        command,
        '--iga_potential_weights_path',
        args.new_potential if variant['new_potential'] else args.old_potential,
    )
    for flag in ('--checkpoint_dir', '--canary_eval_interval_shards', '--canary_max_regression'):
        remove_option(command, flag)
    for flag in (
        '--resume_stage1', '--reset_optimizers_on_resume',
        '--canary_stop_on_regression',
    ):
        remove_option(command, flag, value=False)
    if formal or audit_canary:
        set_option(command, '--canary_eval_interval_shards', 3)
        set_option(command, '--canary_max_regression', 0.02)
        set_switch(command, '--canary_stop_on_regression', formal)
    else:
        set_option(command, '--canary_eval_interval_shards', 0)

    foundation = args.suite_dir / 'new_plane_bc_checkpoint.txt'
    is_foundation = variant_id == 'N2_safe_tail_dagger' and seed == 1 and not formal
    if is_foundation:
        set_option(command, '--plane_bc_pretrain_epochs', 4)
        set_option(command, '--plane_bc_teacher_dir', args.teacher_dir)
        set_option(command, '--plane_bc_dagger_schedule', '1.00,0.80,0.60,0.40')
        set_option(command, '--plane_bc_dagger_seed', seed + 73001)
        set_option(command, '--plane_bc_dagger_tail_start_fraction', 0.75)
        set_option(command, '--plane_bc_dagger_tail_teacher_rate', 0.80)
        set_option(command, '--plane_bc_lr', 0.0001)
        set_option(command, '--plane_bc_shared_lr_scale', 0.3)
        set_option(command, '--plane_bc_freeze_shared_epochs', 1)
        set_option(command, '--plane_bc_pair_loss_coef', 1.0)
        set_option(command, '--plane_bc_order_loss_coef', 0.0)
        set_option(command, '--plane_bc_rollouts_per_epoch', 0)
        set_option(command, '--plane_bc_initial_weight', 5.0)
        set_option(command, '--plane_bc_relocation_weight', 2.0)
        set_option(command, '--plane_bc_critical_op_weight', 2.0)
        set_option(command, '--plane_bc_tail_start_fraction', 0.75)
        set_option(command, '--plane_bc_tail_weight', 4.0)
        set_option(command, '--plane_bc_tail_final_start_fraction', 0.90)
        set_option(command, '--plane_bc_tail_final_weight', 8.0)
    else:
        set_option(command, '--plane_bc_pretrain_epochs', 0)
        checkpoint = args.old_bc_checkpoint
        if variant['new_bc']:
            if not foundation.is_file():
                raise FileNotFoundError(
                    f'New PlaneBC foundation is not ready: {foundation}.'
                )
            checkpoint = Path(foundation.read_text(encoding='utf-8').strip())
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        set_option(command, '--checkpoint_dir', checkpoint)
        set_switch(command, '--resume_stage1', True)
        set_switch(command, '--reset_optimizers_on_resume', True)
    return command, experiment


def comparison_command(
    args: argparse.Namespace,
    checkpoint: Path,
    output: Path,
    seed: int,
    *,
    blind: bool,
) -> list[str]:
    dataset = args.blind_test_dir if blind else args.validation_dir
    command = [
        str(PYTHON), str(COMPARE),
        '--env_name', 'HKBZ',
        '--algorithm_name', 'gnn_mappo',
        '--ac_config', str(AC_CONFIG),
        '--env_config', str(ENV_CONFIG),
        '--checkpoint_dir', str(checkpoint),
        '--dataset_test_dir', str(dataset),
        '--output_json', str(output),
        '--methods', 'drl_g',
        '--model_batch_size', '60',
        '--max_steps', '4000',
        '--seed', str(seed),
        '--policy_tau', '0.3',
        '--plane_order_mode', 'fixed',
        '--plane_pair_decoder', 'joint_pair',
        '--global_feature_mode', 'f1f2',
        '--max_cases', '60',
    ]
    if not blind:
        command.extend([
            '--case_offset', '60',
            '--partition_seed', str(args.partition_seed),
            '--partition_stratify_by', 'profile',
        ])
    return command


def process_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        'CUDA_VISIBLE_DEVICES': str(args.gpu),
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'PYTHONHASHSEED': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'MPLCONFIGDIR': '/tmp',
    })
    return environment


def run_command(
    command: list[str],
    log_path: Path,
    command_path: Path,
    environment: dict[str, str],
) -> None:
    global ACTIVE_PROCESS
    log_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(command_path, {
        'command': command,
        'shell_command': shlex.join(command),
        'environment': {
            key: environment[key]
            for key in (
                'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'PYTHONHASHSEED',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF',
            )
        },
    })
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f"\n[Command] {shlex.join(command)}\n")
        log.flush()
        ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        return_code = ACTIVE_PROCESS.wait()
    ACTIVE_PROCESS = None
    if return_code != 0:
        raise RuntimeError(
            f'Command exited with {return_code}; inspect {log_path}.'
        )


def training_record(
    args: argparse.Namespace,
    variant_id: str,
    seed: int,
    *,
    formal: bool = False,
    audit_canary: bool = False,
) -> dict:
    phase = 'formal' if formal else ('audit' if audit_canary else 'screen')
    label = (
        f'audit_N0_canary_on_seed{seed}'
        if audit_canary else f'{variant_id}_seed{seed}'
    )
    record_path = args.suite_dir / phase / f'{label}.json'
    if record_path.is_file():
        record = read_json(record_path)
        if record.get('status') == 'completed':
            print(f'[V2] Reusing completed {phase} {label}.', flush=True)
            return record
    command, experiment = training_command(
        args,
        variant_id,
        seed,
        formal=formal,
        audit_canary=audit_canary,
    )
    checkpoint = completed_training_checkpoint(experiment)
    if checkpoint is None:
        recovery = interrupted_training_checkpoint(experiment)
        if recovery is not None:
            cursor = configure_exact_recovery_resume(command, recovery)
            print(
                '[V2] Resuming exact post-shard checkpoint for '
                f'{phase} {label}: epoch={cursor["epoch"]}, '
                f'completed_shards={cursor["completed_shards"]}/'
                f'{cursor["total_shards"]}, '
                f'total_num_steps={cursor["total_num_steps"]}.',
                flush=True,
            )
        else:
            print(
                f'[V2] Starting a fresh run for unfinished {phase} {label}.',
                flush=True,
            )
        run_command(
            command,
            args.suite_dir / 'logs' / f'{phase}_{label}.log',
            args.suite_dir / 'commands' / f'{phase}_{label}.json',
            process_environment(args),
        )
        checkpoint = completed_training_checkpoint(experiment)
    if checkpoint is None:
        raise RuntimeError(
            f'No completed run with a Best checkpoint after {experiment}.'
        )
    if variant_id == 'N2_safe_tail_dagger' and seed == 1 and not formal:
        plane_bc = latest_checkpoint(experiment, 'checkpoint_PlaneBC.pt')
        if plane_bc is None:
            raise RuntimeError('N2 foundation completed without PlaneBC checkpoint.')
        foundation_file = args.suite_dir / 'new_plane_bc_checkpoint.txt'
        temporary = foundation_file.with_name(
            f'.{foundation_file.name}.tmp.{os.getpid()}'
        )
        temporary.write_text(str(plane_bc.resolve()) + '\n', encoding='utf-8')
        os.replace(temporary, foundation_file)
    evaluation_path = args.suite_dir / phase / f'{label}_evaluation.json'
    if not evaluation_path.is_file():
        run_command(
            comparison_command(
                args,
                checkpoint,
                evaluation_path,
                seed,
                blind=formal,
            ),
            args.suite_dir / 'logs' / f'{phase}_{label}_evaluation.log',
            args.suite_dir / 'commands' / f'{phase}_{label}_evaluation.json',
            process_environment(args),
        )
    evaluation = read_json(evaluation_path)
    drl = evaluation.get('methods', {}).get('DRL-G', {})
    if drl.get('status') != 'completed':
        raise RuntimeError(f'Incomplete evaluation {evaluation_path}.')
    record = {
        'status': 'completed',
        'phase': phase,
        'variant': variant_id,
        'seed': seed,
        'experiment_name': experiment,
        'checkpoint': str(checkpoint.resolve()),
        'model_sha256': model_digest(checkpoint),
        'actor_update_health': checkpoint_health(checkpoint),
        'safe_pipeline': {
            'safe_async_graph_clone_workers': (
                FORMAL_SAFE_ASYNC_GRAPH_CLONE_WORKERS
                if (formal or args.safe_pipeline_all_phases) else 0
            ),
            'safe_graph_batch_pipeline': bool(
                formal or args.safe_pipeline_all_phases
            ),
            'safe_dagger_teacher_overlap': bool(
                formal or args.safe_pipeline_all_phases
            ),
        },
        'evaluation_json': str(evaluation_path.resolve()),
        'evaluation_summary': drl.get('summary', {}),
        'completed_unix_time': time.time(),
    }
    atomic_json(record_path, record)
    return record


def eval_cases(record: dict) -> list[dict]:
    payload = read_json(Path(record['evaluation_json']))
    cases = payload['methods']['DRL-G']['cases']
    if len(cases) != 60:
        raise RuntimeError(
            f"Expected 60 evaluation cases for {record['variant']}, got {len(cases)}."
        )
    return cases


def composite(cases: list[dict]) -> float:
    values = {}
    for distribution, target_weight in (
        ('iid', 0.50), ('ood_stress', 0.45), ('ood_scale', 0.05)
    ):
        group = [
            float(case['makespan']) for case in cases
            if case['distribution'] == distribution and case.get('completed')
        ]
        if not group:
            raise RuntimeError(f'No completed {distribution} evaluation cases.')
        values[distribution] = target_weight * float(np.mean(group))
    return float(sum(values.values()))


def record_is_healthy(record: dict) -> bool:
    summary = record['evaluation_summary']
    health = record['actor_update_health']
    return bool(
        float(summary.get('completion_rate', 0.0)) == 1.0
        and int(summary.get('cycle_count', 1)) == 0
        and float(health.get('step_completion_rate', 0.0)) >= 0.90
        and int(health.get('zero_update_shards', 1)) == 0
        and not health.get('canary_rejected', False)
    )


def wait_for_file(args: argparse.Namespace, path: Path, label: str) -> None:
    started = time.monotonic()
    peer_status = args.suite_dir / f'lane{1 - args.lane}_status.json'
    while not path.is_file():
        if peer_status.is_file():
            peer = read_json(peer_status)
            if peer.get('status') == 'failed':
                raise RuntimeError(
                    f'Peer lane failed while waiting for {label}: '
                    f"{peer.get('error', 'unknown error')}"
                )
        if time.monotonic() - started > args.barrier_timeout:
            raise TimeoutError(f'Timed out waiting for {label}: {path}.')
        time.sleep(15.0)


def phase1_finalize(args: argparse.Namespace) -> dict:
    baseline = read_json(
        args.suite_dir / 'screen/N0_r3_reproduction_seed1.json'
    )
    baseline_score = composite(eval_cases(baseline))
    candidates = []
    for variant_id in variants():
        if variant_id == 'N0_r3_reproduction':
            continue
        record = read_json(args.suite_dir / 'screen' / f'{variant_id}_seed1.json')
        score = composite(eval_cases(record))
        candidates.append({
            'variant': variant_id,
            'selection_composite': score,
            'improvement_vs_n0': (baseline_score - score) / baseline_score,
            'health_passed': record_is_healthy(record),
        })
    eligible = [item for item in candidates if item['health_passed']]
    eligible.sort(key=lambda item: item['selection_composite'])
    finalists = [item['variant'] for item in eligible[:2]]
    audit = read_json(args.suite_dir / 'audit/audit_N0_canary_on_seed1.json')
    audit_cases = eval_cases(audit)
    baseline_cases = eval_cases(baseline)
    audit_delta = np.asarray([
        float(right['makespan']) - float(left['makespan'])
        for left, right in zip(baseline_cases, audit_cases)
    ])
    payload = {
        'status': 'completed',
        'baseline_composite': baseline_score,
        'candidates': candidates,
        'finalists': finalists,
        'successive_halving': 'top two healthy N1-N4 by seed1 select composite',
        'determinism_audit': {
            'model_sha256_equal': (
                baseline['model_sha256'] == audit['model_sha256']
            ),
            'max_abs_case_delta': float(np.max(np.abs(audit_delta))),
            'mean_case_delta': float(np.mean(audit_delta)),
            'canary_off_model_sha256': baseline['model_sha256'],
            'canary_on_model_sha256': audit['model_sha256'],
        },
        'completed_unix_time': time.time(),
    }
    atomic_json(args.suite_dir / 'halving.json', payload)
    return payload


def paired_bootstrap(values: np.ndarray, seed: int = 20260803) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(20000, dtype=np.float64)
    for start in range(0, len(means), 1000):
        count = min(1000, len(means) - start)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[sample].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def candidate_gate(
    args: argparse.Namespace,
    variant_id: str,
) -> dict:
    baseline_records = {
        seed: read_json(
            args.suite_dir / 'screen' / f'N0_r3_reproduction_seed{seed}.json'
        )
        for seed in (1, 2)
    }
    candidate_records = {
        seed: read_json(
            args.suite_dir / 'screen' / f'{variant_id}_seed{seed}.json'
        )
        for seed in (1, 2)
    }
    paired_deltas = []
    iid_base, iid_candidate = [], []
    stress_base, stress_candidate = [], []
    worst_base, worst_candidate = [], []
    seed_scores = {}
    complete = True
    for seed in (1, 2):
        baseline_cases = eval_cases(baseline_records[seed])
        candidate_cases = eval_cases(candidate_records[seed])
        left = {case['case_id']: case for case in baseline_cases}
        right = {case['case_id']: case for case in candidate_cases}
        if set(left) != set(right):
            raise RuntimeError(f'Unpaired select cases for seed {seed}.')
        ordered = sorted(left)
        base_values = np.asarray([float(left[key]['makespan']) for key in ordered])
        candidate_values = np.asarray([
            float(right[key]['makespan']) for key in ordered
        ])
        paired_deltas.extend((candidate_values - base_values).tolist())
        seed_scores[str(seed)] = {
            'baseline': composite(baseline_cases),
            'candidate': composite(candidate_cases),
        }
        iid_keys = [key for key in ordered if left[key]['distribution'] == 'iid']
        stress_keys = [
            key for key in ordered if left[key]['profile'] == 'stress_arrival'
        ]
        iid_base.extend(float(left[key]['makespan']) for key in iid_keys)
        iid_candidate.extend(float(right[key]['makespan']) for key in iid_keys)
        stress_base.extend(float(left[key]['makespan']) for key in stress_keys)
        stress_candidate.extend(float(right[key]['makespan']) for key in stress_keys)
        worst_count = max(1, int(math.ceil(0.10 * len(ordered))))
        worst_indices = np.argsort(base_values)[-worst_count:]
        worst_base.extend(base_values[worst_indices].tolist())
        worst_candidate.extend(candidate_values[worst_indices].tolist())
        complete &= record_is_healthy(baseline_records[seed])
        complete &= record_is_healthy(candidate_records[seed])
    baseline_composite = float(np.mean([
        value['baseline'] for value in seed_scores.values()
    ]))
    candidate_composite = float(np.mean([
        value['candidate'] for value in seed_scores.values()
    ]))
    relative_improvement = (
        baseline_composite - candidate_composite
    ) / baseline_composite
    iid_regression = (
        float(np.mean(iid_candidate)) - float(np.mean(iid_base))
    ) / float(np.mean(iid_base))
    stress_improvement = (
        float(np.mean(stress_base)) - float(np.mean(stress_candidate))
    ) / float(np.mean(stress_base))
    worst_improvement = (
        float(np.mean(worst_base)) - float(np.mean(worst_candidate))
    ) / float(np.mean(worst_base))
    ci_low, ci_high = paired_bootstrap(np.asarray(paired_deltas))
    seed_consistent = all(
        value['candidate'] < value['baseline']
        for value in seed_scores.values()
    )
    gates = {
        'composite_improvement_ge_1pct': relative_improvement >= 0.01,
        'paired_bootstrap_upper_lt_zero': ci_high < 0.0,
        'iid_regression_le_0_3pct': iid_regression <= 0.003,
        'stress_arrival_improvement_ge_2pct': stress_improvement >= 0.02,
        'worst10_improvement_ge_3pct': worst_improvement >= 0.03,
        'both_seeds_improve': seed_consistent,
        'training_and_completion_health': bool(complete),
    }
    return {
        'variant': variant_id,
        'baseline_composite': baseline_composite,
        'candidate_composite': candidate_composite,
        'relative_improvement': relative_improvement,
        'paired_mean_delta': float(np.mean(paired_deltas)),
        'paired_bootstrap_95ci': [ci_low, ci_high],
        'iid_relative_regression': iid_regression,
        'stress_arrival_relative_improvement': stress_improvement,
        'worst10_relative_improvement': worst_improvement,
        'seed_scores': seed_scores,
        'gates': gates,
        'passed': all(gates.values()),
    }


def final_selection(args: argparse.Namespace, finalists: list[str]) -> dict:
    records = [candidate_gate(args, variant_id) for variant_id in finalists]
    passed = [record for record in records if record['passed']]
    passed.sort(key=lambda record: record['candidate_composite'])
    selected = passed[0]['variant'] if passed else None
    payload = {
        'status': 'completed',
        'finalists': finalists,
        'candidate_gates': records,
        'selected': selected,
        'formal_skipped': None if selected else (
            'No two-seed candidate passed every pre-registered gate.'
        ),
        'completed_unix_time': time.time(),
    }
    atomic_json(args.suite_dir / 'selection.json', payload)
    return payload


def update_status(args: argparse.Namespace, **updates) -> None:
    suffix = args.status_id or f'lane{args.lane}'
    path = args.suite_dir / f'{suffix}_status.json'
    payload = read_json(path) if path.is_file() else {}
    if updates.get('status') in {'running', 'completed'}:
        for stale_key in ('error', 'traceback', 'reason'):
            payload.pop(stale_key, None)
    payload.update(
        lane=args.lane,
        gpu=args.gpu,
        status_id=suffix,
        pid=os.getpid(),
        updated_unix_time=time.time(),
        **updates,
    )
    atomic_json(path, payload)


def wait_for_peer_phase(args: argparse.Namespace, phase: str) -> None:
    peer = args.suite_dir / f'lane{1 - args.lane}_status.json'
    marker = args.suite_dir / f'lane{1 - args.lane}_{phase}.done.json'
    started = time.monotonic()
    while True:
        if marker.is_file():
            return
        if peer.is_file():
            payload = read_json(peer)
            if payload.get('status') == 'failed':
                raise RuntimeError(
                    f"Peer lane failed: {payload.get('error', 'unknown error')}"
                )
        if time.monotonic() - started > args.barrier_timeout:
            raise TimeoutError(f'Timed out waiting for peer phase {phase}.')
        time.sleep(15.0)


def mark_phase(args: argparse.Namespace, phase: str) -> None:
    update_status(args, status='running', phase=phase, current=None)
    atomic_json(
        args.suite_dir / f'lane{args.lane}_{phase}.done.json',
        {'lane': args.lane, 'phase': phase, 'completed_unix_time': time.time()},
    )


def terminate_active(signum, _frame) -> None:
    global ACTIVE_PROCESS
    if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
        try:
            os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, terminate_active)
    signal.signal(signal.SIGINT, terminate_active)
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    update_status(args, status='running', phase='starting')
    if args.start_delay_seconds > 0.0:
        update_status(
            args,
            status='running',
            phase='start_delay',
            start_delay_seconds=float(args.start_delay_seconds),
        )
        time.sleep(args.start_delay_seconds)

    if args.formal_only_variant is not None:
        for seed in args.formal_seeds:
            update_status(
                args,
                status='running',
                phase='formal_extension_running',
                current=f'{args.formal_only_variant}_seed{seed}',
            )
            training_record(
                args,
                args.formal_only_variant,
                seed,
                formal=True,
            )
        update_status(
            args,
            status='completed',
            phase='formal_extension_complete',
            current=None,
        )
        return 0

    if args.lane == 0:
        phase1 = [
            ('N0_r3_reproduction', 1, False),
            ('N0_r3_reproduction', 1, True),
            ('N1_tail_cv_potential', 1, False),
            ('N4_dagger_tail_cv_cvar', 1, False),
        ]
    else:
        phase1 = [
            ('N2_safe_tail_dagger', 1, False),
            ('N3_dagger_tail_cv', 1, False),
        ]
    for variant_id, seed, audit in phase1:
        if variant_id == 'N4_dagger_tail_cv_cvar':
            foundation = args.suite_dir / 'new_plane_bc_checkpoint.txt'
            if not foundation.is_file():
                update_status(
                    args,
                    status='running',
                    phase='phase1_waiting_foundation',
                    current=f'{variant_id}_seed{seed}',
                )
                wait_for_file(args, foundation, 'new PlaneBC foundation')
        update_status(
            args,
            status='running',
            phase='phase1_running',
            current=f'{variant_id}_seed{seed}' + ('_audit' if audit else ''),
        )
        training_record(
            args, variant_id, seed, audit_canary=audit
        )
    mark_phase(args, 'phase1_complete')
    wait_for_peer_phase(args, 'phase1_complete')
    halving_path = args.suite_dir / 'halving.json'
    if args.lane == 0 and not halving_path.is_file():
        phase1_finalize(args)
    wait_for_file(args, halving_path, 'successive-halving decision')
    finalists = list(read_json(halving_path).get('finalists', []))
    if len(finalists) != 2:
        update_status(
            args,
            status='completed',
            phase='stopped_after_phase1',
            reason=f'Expected two healthy finalists, got {finalists}.',
        )
        return 0

    phase2 = [('N0_r3_reproduction', 2)] if args.lane == 0 else []
    phase2.append((finalists[args.lane], 2))
    for variant_id, seed in phase2:
        update_status(
            args,
            status='running',
            phase='phase2_running',
            current=f'{variant_id}_seed{seed}',
        )
        training_record(args, variant_id, seed)
    mark_phase(args, 'phase2_complete')
    wait_for_peer_phase(args, 'phase2_complete')
    selection_path = args.suite_dir / 'selection.json'
    if args.lane == 0 and not selection_path.is_file():
        final_selection(args, finalists)
    wait_for_file(args, selection_path, 'two-seed selection gates')
    selected = read_json(selection_path).get('selected')
    if not selected:
        update_status(
            args,
            status='completed',
            phase='formal_skipped',
            current=None,
        )
        return 0

    formal_seeds = (1, 3) if args.lane == 0 else (2,)
    for seed in formal_seeds:
        update_status(
            args,
            status='running',
            phase='formal_running',
            current=f'{selected}_seed{seed}',
        )
        training_record(args, selected, seed, formal=True)
    update_status(args, status='completed', phase='formal_complete', current=None)
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
    except BaseException as error:
        try:
            parsed = parse_args()
            update_status(
                parsed,
                status='failed',
                phase='failed',
                current=None,
                error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(),
            )
        except BaseException:
            pass
        raise
    raise SystemExit(exit_code)
