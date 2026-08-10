#!/usr/bin/env python
"""Run one fixed-seed arm of the Stage-1 tau-range screen.

All arms are identical except for the policy-temperature schedule.  Validation
is delegated to one persistent per-GPU evaluator, so this process never creates
its own validation workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = (ROOT.parent / 'conda/envs/maia/bin/python3.11').resolve()
TRAIN = ROOT / 'onpolicy/scripts/train/train_hkbz.py'
RESULT_ROOT = ROOT / 'onpolicy/scripts/results/HKBZ/simple/gnn_mappo'
AC_CONFIG = ROOT / 'onpolicy/config/ac.yaml'
ENV_CONFIG = ROOT / 'onpolicy/config/env_plane_pretrain.yaml'


VARIANTS = {
    'fixed_030': {
        'anneal_original': 0.3,
        'anneal_final': 0.3,
        'tau_anneal_epochs': 0,
        'expected_tau': [0.3, 0.3, 0.3, 0.3],
    },
    'range_050_030': {
        'anneal_original': 0.5,
        'anneal_final': 0.3,
        'tau_anneal_epochs': 3,
        'expected_tau': [0.5, 0.4, 0.3, 0.3],
    },
    'range_080_030': {
        'anneal_original': 0.8,
        'anneal_final': 0.3,
        'tau_anneal_epochs': 3,
        'expected_tau': [0.8, 0.55, 0.3, 0.3],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=tuple(VARIANTS))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpu-set', required=True)
    parser.add_argument('--shared-eval-socket', required=True)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--source-command-json', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--potential-path', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    parser.add_argument('--start-delay-seconds', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--train-sampling-size', type=int, default=480)
    parser.add_argument('--partition-seed', type=int, default=20260803)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def set_option(command: list[str], flag: str, value) -> None:
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def remove_option(command: list[str], flag: str, *, takes_value=True) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + (2 if takes_value else 1)]


def set_switch(command: list[str], flag: str, enabled: bool) -> None:
    remove_option(command, flag, takes_value=False)
    if enabled:
        command.append(flag)


def experiment_name(run_tag: str, variant: str, seed: int) -> str:
    return f'{run_tag}_{variant}_seed{seed}'


def latest_completed_run(experiment: str) -> tuple[Path, Path] | None:
    directory = RESULT_ROOT / experiment
    candidates = []
    for run_dir in directory.glob('run*'):
        suffix = run_dir.name.removeprefix('run')
        if suffix.isdigit():
            candidates.append((int(suffix), run_dir))
    for _, run_dir in sorted(candidates, reverse=True):
        status_path = run_dir / 'run_status.json'
        checkpoint = run_dir / 'models/checkpoint_Best.pt'
        if not status_path.is_file() or not checkpoint.is_file():
            continue
        if read_json(status_path).get('status') == 'completed':
            return run_dir, checkpoint
    return None


def training_command(args: argparse.Namespace) -> list[str]:
    command = list(read_json(args.source_command_json)['command'])
    if len(command) < 2:
        raise ValueError(f'Invalid source command: {args.source_command_json}')
    command[0] = str(PYTHON)
    command[1] = str(TRAIN)
    schedule = VARIANTS[args.variant]
    set_option(
        command, '--experiment_name',
        experiment_name(args.run_tag, args.variant, args.seed),
    )
    set_option(command, '--ac_config', AC_CONFIG)
    set_option(command, '--env_config', ENV_CONFIG)
    set_option(command, '--seed', args.seed)
    set_option(command, '--num_episodes', args.epochs)
    set_option(command, '--train_sampling_size', args.train_sampling_size)
    set_option(command, '--n_rollout_threads', 60)
    set_option(command, '--n_eval_rollout_threads', 60)
    set_option(command, '--max_eval_cases', 60)
    set_option(command, '--eval_dataset_dir', args.validation_dir)
    set_option(command, '--eval_case_offset', 0)
    set_option(command, '--eval_partition_seed', args.partition_seed)
    set_option(command, '--eval_partition_stratify_by', 'profile')
    set_option(command, '--eval_interval', 1)
    set_option(command, '--early_stop_patience', 0)
    set_option(command, '--evaluation_tau', 0.3)
    set_option(command, '--canary_eval_interval_shards', 0)
    set_option(command, '--iga_potential_weights_path', args.potential_path)
    set_option(command, '--checkpoint_dir', args.initial_checkpoint)
    set_option(command, '--plane_bc_pretrain_epochs', 0)
    set_option(command, '--anneal_original', schedule['anneal_original'])
    set_option(command, '--anneal_final', schedule['anneal_final'])
    set_option(command, '--tau_anneal_epochs', schedule['tau_anneal_epochs'])
    set_option(command, '--shared_eval_socket', args.shared_eval_socket)
    set_option(command, '--shared_eval_cpu_set', args.cpu_set)
    set_option(command, '--shared_eval_timeout_seconds', 7200)
    remove_option(command, '--selection_checkpoint_dir')
    remove_option(command, '--canary_max_regression')
    set_switch(command, '--canary_stop_on_regression', False)
    set_switch(command, '--use_eval', True)
    set_switch(command, '--resume_stage1', True)
    set_switch(command, '--reset_optimizers_on_resume', True)
    set_switch(command, '--safe_graph_batch_pipeline', True)
    set_switch(command, '--safe_dagger_teacher_overlap', True)
    set_option(command, '--safe_async_graph_clone_workers', 4)
    return command


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_plan(args: argparse.Namespace) -> None:
    plan_path = args.suite_dir / 'experiment_plan.json'
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan.get('run_tag') != args.run_tag or plan.get('seed') != args.seed:
            raise ValueError(f'Existing plan does not match this run: {plan_path}')
        return
    atomic_json(plan_path, {
        'schema_version': 1,
        'status': 'running',
        'run_tag': args.run_tag,
        'objective': (
            'At fixed seed, isolate whether a wider early tau range improves '
            'exploration without sacrificing late-policy stability.'
        ),
        'base_variant': 'N1_tail_cv_potential',
        'seed': args.seed,
        'epochs': args.epochs,
        'train_sampling_size_per_epoch': args.train_sampling_size,
        'variants': VARIANTS,
        'controlled_variables': (
            'same seed, initial PlaneBC, N1 potential, PPO settings, training '
            'samples, tune partition, evaluation tau, and epoch count'
        ),
        'validation': {
            'service': 'one persistent 60-worker pool shared by all GPU0 arms',
            'tune_and_checkpoint': 'fixed validation partition offset 0, n=60',
            'heldout': (
                'deferred until all arms finish, then evaluated sequentially '
                'to avoid per-arm validation pools'
            ),
            'blind_test': 'sealed; not touched by this screen',
        },
        'cpu_policy': {
            'trainer_isolation': '12 physical cores / 24 logical CPUs per arm',
            'validation_borrowing': (
                'the shared evaluator temporarily binds to the requesting '
                'trainer slice while that trainer is blocked'
            ),
        },
        'created_unix_time': time.time(),
    })


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f"\n[Command] {shlex.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f'Command exited with {completed.returncode}; inspect {log_path}'
        )


def main() -> None:
    args = parse_args()
    for required in (
        PYTHON, TRAIN, args.source_command_json, args.initial_checkpoint,
        args.potential_path, args.validation_dir,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.seed <= 0:
        raise ValueError('--seed must be positive.')
    if args.epochs != 4:
        raise ValueError('This pre-registered pilot requires exactly 4 epochs.')
    if args.train_sampling_size <= 0:
        raise ValueError('--train-sampling-size must be positive.')
    if args.start_delay_seconds < 0:
        raise ValueError('--start-delay-seconds must be non-negative.')

    args.suite_dir.mkdir(parents=True, exist_ok=True)
    ensure_plan(args)
    status_path = args.suite_dir / 'trial_status' / f'{args.variant}.json'
    status = {
        'status': 'delayed' if args.start_delay_seconds else 'running',
        'variant': args.variant,
        'seed': args.seed,
        'gpu': args.gpu,
        'cpu_set': args.cpu_set,
        'shared_eval_socket': args.shared_eval_socket,
        'schedule': VARIANTS[args.variant],
        'started_unix_time': time.time(),
    }
    atomic_json(status_path, status)
    if args.start_delay_seconds:
        time.sleep(args.start_delay_seconds)

    try:
        status['status'] = 'running'
        status['training_started_unix_time'] = time.time()
        atomic_json(status_path, status)
        name = experiment_name(args.run_tag, args.variant, args.seed)
        command = training_command(args)
        atomic_json(
            args.suite_dir / 'commands' / f'{args.variant}_train.json',
            {'command': command, 'shell_command': shlex.join(command)},
        )
        completed = latest_completed_run(name)
        if completed is None:
            print(
                f'[TauRange] Training {args.variant} seed{args.seed} '
                f'on GPU{args.gpu}, CPUs={args.cpu_set}.',
                flush=True,
            )
            run_logged(
                command,
                args.suite_dir / 'logs' / f'{args.variant}_train.log',
            )
            completed = latest_completed_run(name)
        if completed is None:
            raise RuntimeError(f'No completed Best checkpoint for {name}.')
        run_dir, checkpoint = completed
        record = {
            'status': 'completed',
            'variant': args.variant,
            'seed': args.seed,
            'gpu': args.gpu,
            'cpu_set': args.cpu_set,
            'shared_eval_socket': args.shared_eval_socket,
            'experiment_name': name,
            'run_dir': str(run_dir.resolve()),
            'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': sha256(checkpoint),
            'initial_checkpoint': str(args.initial_checkpoint.resolve()),
            'initial_checkpoint_sha256': sha256(args.initial_checkpoint),
            'schedule': VARIANTS[args.variant],
            'completed_unix_time': time.time(),
        }
        atomic_json(
            args.suite_dir / 'records' / f'{args.variant}.json', record
        )
        atomic_json(status_path, record)
    except BaseException as exc:
        status.update({
            'status': 'failed',
            'error': repr(exc),
            'failed_unix_time': time.time(),
        })
        atomic_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
