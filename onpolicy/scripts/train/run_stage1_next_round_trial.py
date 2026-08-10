#!/usr/bin/env python
"""Run one isolated Stage-1 next-round training trial.

The six registered arms all start from the same frozen N2 PlaneBC snapshot.
Only the pre-registered mechanism fields differ.  Validation is delegated to
one persistent evaluator per GPU, so three colocated trainers never construct
three duplicate validation pools.
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

import torch


ROOT = Path(__file__).resolve().parents[3]
PYTHON = (ROOT.parent / 'conda/envs/maia/bin/python3.11').resolve()
TRAIN = ROOT / 'onpolicy/scripts/train/train_hkbz.py'
RESULT_ROOT = ROOT / 'onpolicy/scripts/results/HKBZ/simple/gnn_mappo'
AC_CONFIG = ROOT / 'onpolicy/config/ac.yaml'
ENV_CONFIG = ROOT / 'onpolicy/config/env_plane_pretrain.yaml'

TARGET_DISTRIBUTION_WEIGHTS = 'iid=0.50,ood_stress=0.45,ood_scale=0.05'
PROFILE_STRESS_WEIGHTS = (
    'balanced=0.125,resource_sparse=0.10,bursty=0.10,'
    'high_flex=0.075,coupled=0.05,light=0.05,'
    'stress_joint=0.15,stress_arrival=0.20,resource_ood=0.10,'
    'low_load_ood=0.05'
)


VARIANTS = {
    'M0_n2_control': {
        'hypothesis': 'Reproduce the robust N2 reference at fixed tau=0.3.',
    },
    'M1_potential_ramp': {
        'hypothesis': (
            'Delay IGA potential strength so exact-Cmax PPO establishes its '
            'own direction before the teacher-derived shaping reaches beta=0.1.'
        ),
        'potential_schedule': '0.0,0.05,0.10,0.10',
    },
    'M2_bc_kl_anneal': {
        'hypothesis': (
            'Use a strong early BC trust region, then release it so PPO can '
            'adapt without losing the safe DAgger initialization.'
        ),
        'bc_kl_schedule': '0.40,0.25,0.10,0.05',
    },
    'M3_stress_replay': {
        'hypothesis': (
            'Increase train-only stress-arrival and stress-joint exposure '
            'without changing validation or the number of sampled cases.'
        ),
        'sampling_mode': 'profile_balanced',
        'sampling_weights': PROFILE_STRESS_WEIGHTS,
    },
    'M4_tail_credit': {
        'hypothesis': (
            'Redistribute each case-balanced PPO mass toward the final 25% '
            'of environment time while keeping every case total equal.'
        ),
        'tail_policy_start_fraction': 0.75,
        'tail_policy_weight': 3.0,
    },
    'M5_ramp_tail_combo': {
        'hypothesis': (
            'Test the pre-registered interaction between staged potential '
            'direction and late-decision credit emphasis.'
        ),
        'potential_schedule': '0.0,0.05,0.10,0.10',
        'tail_policy_start_fraction': 0.75,
        'tail_policy_weight': 3.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('screen', 'formal'), required=True)
    parser.add_argument('--variant', choices=tuple(VARIANTS), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--cpu-set', required=True)
    parser.add_argument('--shared-eval-socket', required=True)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--source-command-json', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--potential-path', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    parser.add_argument('--start-delay-seconds', type=int, default=0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--train-sampling-size', type=int, required=True)
    parser.add_argument('--partition-seed', type=int, default=20260803)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
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


def set_optional(command: list[str], flag: str, value) -> None:
    remove_option(command, flag)
    if str(value or '').strip():
        command.extend([flag, str(value)])


def set_switch(command: list[str], flag: str, enabled: bool) -> None:
    remove_option(command, flag, takes_value=False)
    if enabled:
        command.append(flag)


def experiment_name(run_tag: str, phase: str, variant: str, seed: int) -> str:
    return f'{run_tag}_{phase}_{variant}_seed{seed}'


def expanded_schedule(raw: str, epochs: int, default: float) -> list[float]:
    values = [
        float(item.strip()) for item in str(raw or '').split(',') if item.strip()
    ]
    if not values:
        values = [float(default)]
    return [values[min(epoch, len(values) - 1)] for epoch in range(epochs)]


def method_manifest(variant: str, epochs: int) -> dict:
    registered = VARIANTS[variant]
    return {
        **registered,
        'potential_beta_by_epoch': expanded_schedule(
            registered.get('potential_schedule', ''), epochs, 0.10
        ),
        'bc_reference_kl_by_epoch': expanded_schedule(
            registered.get('bc_kl_schedule', ''), epochs, 0.20
        ),
        'sampling_mode': registered.get(
            'sampling_mode', 'distribution_balanced'
        ),
        'sampling_weights': registered.get(
            'sampling_weights', TARGET_DISTRIBUTION_WEIGHTS
        ),
        'tail_policy_start_fraction': float(registered.get(
            'tail_policy_start_fraction', 1.0
        )),
        'tail_policy_weight': float(registered.get('tail_policy_weight', 1.0)),
    }


def training_command(args: argparse.Namespace) -> list[str]:
    command = list(read_json(args.source_command_json)['command'])
    if len(command) < 2:
        raise ValueError(f'Invalid source command: {args.source_command_json}')
    command[0] = str(PYTHON)
    command[1] = str(TRAIN)
    variant = VARIANTS[args.variant]

    set_option(command, '--experiment_name', experiment_name(
        args.run_tag, args.phase, args.variant, args.seed
    ))
    set_option(command, '--ac_config', AC_CONFIG)
    set_option(command, '--env_config', ENV_CONFIG)
    set_option(command, '--seed', args.seed)
    set_option(command, '--num_episodes', args.epochs)
    set_option(command, '--train_sampling_size', args.train_sampling_size)
    set_option(command, '--train_sampling_mode', variant.get(
        'sampling_mode', 'distribution_balanced'
    ))
    set_option(command, '--train_sampling_weights', variant.get(
        'sampling_weights', TARGET_DISTRIBUTION_WEIGHTS
    ))
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
    set_option(command, '--anneal_original', 0.3)
    set_option(command, '--anneal_final', 0.3)
    set_option(command, '--tau_anneal_epochs', 0)
    set_option(command, '--iga_potential_beta', 0.10)
    set_optional(
        command,
        '--iga_potential_beta_schedule',
        variant.get('potential_schedule', ''),
    )
    set_option(command, '--bc_reference_kl_coef', 0.20)
    set_optional(
        command,
        '--bc_reference_kl_coef_schedule',
        variant.get('bc_kl_schedule', ''),
    )
    set_option(command, '--tail_policy_start_fraction', variant.get(
        'tail_policy_start_fraction', 1.0
    ))
    set_option(command, '--tail_policy_weight', variant.get(
        'tail_policy_weight', 1.0
    ))
    set_option(command, '--iga_potential_weights_path', args.potential_path)
    set_option(command, '--checkpoint_dir', args.initial_checkpoint)
    set_option(command, '--plane_bc_pretrain_epochs', 0)
    set_option(command, '--shared_eval_socket', args.shared_eval_socket)
    set_option(command, '--shared_eval_cpu_set', args.cpu_set)
    set_option(command, '--shared_eval_timeout_seconds', 10800)
    set_option(command, '--canary_eval_interval_shards', 0)
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


def latest_completed_run(experiment: str) -> tuple[Path, Path] | None:
    candidates = []
    for run_dir in (RESULT_ROOT / experiment).glob('run*'):
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_evaluation(run_dir: Path, checkpoint: Path) -> tuple[dict, Path]:
    payload = torch.load(checkpoint, map_location='cpu')
    epoch = int(payload['episodes'])
    if epoch == 0 or payload.get('stage') == 'pre_ppo_baseline':
        evaluation_path = run_dir / 'evaluations/pre_ppo.json'
    else:
        evaluation_path = run_dir / 'evaluations' / f'epoch_{epoch}.json'
    if not evaluation_path.is_file():
        raise FileNotFoundError(
            f'Best checkpoint refers to missing evaluation: {evaluation_path}'
        )
    evaluation = read_json(evaluation_path)
    summary = dict(evaluation['summary'])
    if abs(
        float(summary['eval_selection_score'])
        - float(payload['selection_score'])
    ) > 1e-5:
        raise RuntimeError('Best checkpoint and evaluation score disagree.')
    return {
        'checkpoint_episode': epoch,
        'selection_score': float(payload['selection_score']),
        'eval_raw_makespan': float(payload['eval_raw_makespan']),
        'eval_iid_makespan': float(payload['eval_iid_makespan']),
        'eval_composite_makespan': float(payload['eval_composite_makespan']),
        'summary': summary,
        'actor_update_health': dict(payload['actor_update_health']),
    }, evaluation_path


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n[Command] {shlex.join(command)}\n')
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
    if args.seed <= 0 or args.epochs <= 0:
        raise ValueError('seed and epochs must be positive.')
    if args.train_sampling_size < 0:
        raise ValueError('train sampling size must be non-negative.')
    if args.start_delay_seconds < 0:
        raise ValueError('start delay must be non-negative.')

    key = f'{args.phase}_{args.variant}_seed{args.seed}'
    status_path = args.suite_dir / 'trial_status' / f'{key}.json'
    status = {
        'status': 'delayed' if args.start_delay_seconds else 'running',
        'phase': args.phase,
        'variant': args.variant,
        'seed': args.seed,
        'gpu': args.gpu,
        'cpu_set': args.cpu_set,
        'shared_eval_socket': args.shared_eval_socket,
        'method': method_manifest(args.variant, args.epochs),
        'started_unix_time': time.time(),
    }
    atomic_json(status_path, status)
    if args.start_delay_seconds:
        time.sleep(args.start_delay_seconds)

    try:
        status['status'] = 'running'
        status['training_started_unix_time'] = time.time()
        atomic_json(status_path, status)
        name = experiment_name(args.run_tag, args.phase, args.variant, args.seed)
        command = training_command(args)
        atomic_json(
            args.suite_dir / 'commands' / f'{key}.json',
            {'command': command, 'shell_command': shlex.join(command)},
        )
        completed = latest_completed_run(name)
        if completed is None:
            print(
                f'[NextRound] {args.phase} {args.variant} seed{args.seed} '
                f'GPU{args.gpu} CPUs={args.cpu_set}.',
                flush=True,
            )
            run_logged(command, args.suite_dir / 'logs' / f'{key}.log')
            completed = latest_completed_run(name)
        if completed is None:
            raise RuntimeError(f'No completed Best checkpoint for {name}.')
        run_dir, checkpoint = completed
        evaluation, evaluation_path = checkpoint_evaluation(run_dir, checkpoint)
        pre_ppo_path = run_dir / 'evaluations/pre_ppo.json'
        if not pre_ppo_path.is_file():
            raise FileNotFoundError(pre_ppo_path)
        pre_ppo = read_json(pre_ppo_path)
        record = {
            **status,
            'status': 'completed',
            'experiment_name': name,
            'run_dir': str(run_dir.resolve()),
            'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': sha256(checkpoint),
            'initial_checkpoint': str(args.initial_checkpoint.resolve()),
            'initial_checkpoint_sha256': sha256(args.initial_checkpoint),
            'evaluation': evaluation,
            'evaluation_json': str(evaluation_path.resolve()),
            'pre_ppo_evaluation_json': str(pre_ppo_path.resolve()),
            'pre_ppo_summary': dict(pre_ppo['summary']),
            'completed_unix_time': time.time(),
        }
        atomic_json(args.suite_dir / 'records' / f'{key}.json', record)
        atomic_json(status_path, record)
    except BaseException as error:
        status.update({
            'status': 'failed',
            'error': f'{type(error).__name__}: {error}',
            'failed_unix_time': time.time(),
        })
        atomic_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
