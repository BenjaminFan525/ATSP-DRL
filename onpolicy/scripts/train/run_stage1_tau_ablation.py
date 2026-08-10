#!/usr/bin/env python
"""Run one GPU lane of the Stage-1 tau-schedule ablation.

The experiment deliberately keeps the blind test sealed.  Training and Best
checkpoint selection use the first fixed validation partition; the selected
checkpoint is then evaluated once on the disjoint second partition.
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
COMPARE = ROOT / 'onpolicy/envs/HKBZ/experiment/valid_fjsp_v2_comparison.py'
ANALYZE = ROOT / 'onpolicy/envs/HKBZ/experiment/analyze_stage1_tau_ablation.py'
RESULT_ROOT = ROOT / 'onpolicy/scripts/results/HKBZ/simple/gnn_mappo'
AC_CONFIG = ROOT / 'onpolicy/config/ac.yaml'
ENV_CONFIG = ROOT / 'onpolicy/config/env_plane_pretrain.yaml'


VARIANTS = {
    'fixed': {
        'anneal_original': 0.3,
        'anneal_final': 0.3,
        'tau_anneal_epochs': 0,
        'expected_tau': [0.3, 0.3, 0.3, 0.3],
    },
    'annealed': {
        'anneal_original': 0.5,
        'anneal_final': 0.3,
        'tau_anneal_epochs': 3,
        'expected_tau': [0.5, 0.4, 0.3, 0.3],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', type=int, required=True, choices=(0, 1))
    parser.add_argument('--gpu', type=int, required=True, choices=(0, 1))
    parser.add_argument('--cpu-set', required=True)
    parser.add_argument('--sequence', required=True)
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


def parse_sequence(value: str) -> list[tuple[str, int]]:
    result = []
    for item in value.split(','):
        variant, seed_text = item.split(':', 1)
        if variant not in VARIANTS:
            raise ValueError(f'Unknown tau variant: {variant}')
        seed = int(seed_text)
        if seed <= 0:
            raise ValueError('Seeds must be positive.')
        result.append((variant, seed))
    if not result:
        raise ValueError('The lane sequence is empty.')
    return result


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


def training_command(args: argparse.Namespace, variant: str, seed: int) -> list[str]:
    command = list(read_json(args.source_command_json)['command'])
    command[0] = str(PYTHON)
    command[1] = str(TRAIN)
    name = experiment_name(args.run_tag, variant, seed)
    set_option(command, '--experiment_name', name)
    set_option(command, '--ac_config', AC_CONFIG)
    set_option(command, '--env_config', ENV_CONFIG)
    set_option(command, '--seed', seed)
    set_option(command, '--num_episodes', args.epochs)
    set_option(command, '--train_sampling_size', args.train_sampling_size)
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
    schedule = VARIANTS[variant]
    set_option(command, '--anneal_original', schedule['anneal_original'])
    set_option(command, '--anneal_final', schedule['anneal_final'])
    set_option(command, '--tau_anneal_epochs', schedule['tau_anneal_epochs'])
    remove_option(command, '--shared_eval_socket')
    remove_option(command, '--shared_eval_cpu_set')
    remove_option(command, '--selection_checkpoint_dir')
    remove_option(command, '--canary_max_regression')
    set_switch(command, '--canary_stop_on_regression', False)
    set_switch(command, '--resume_stage1', True)
    set_switch(command, '--reset_optimizers_on_resume', True)
    set_switch(command, '--safe_graph_batch_pipeline', True)
    set_switch(command, '--safe_dagger_teacher_overlap', True)
    set_option(command, '--safe_async_graph_clone_workers', 4)
    return command


def selection_command(
        args: argparse.Namespace, checkpoint: Path, output: Path,
        seed: int) -> list[str]:
    return [
        str(PYTHON), str(COMPARE),
        '--env_name', 'HKBZ',
        '--algorithm_name', 'gnn_mappo',
        '--ac_config', str(AC_CONFIG),
        '--env_config', str(ENV_CONFIG),
        '--checkpoint_dir', str(checkpoint),
        '--dataset_test_dir', str(args.validation_dir),
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
        '--case_offset', '60',
        '--partition_seed', str(args.partition_seed),
        '--partition_stratify_by', 'profile',
    ]


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_plan(args: argparse.Namespace) -> None:
    plan_path = args.suite_dir / 'experiment_plan.json'
    if plan_path.exists():
        return
    atomic_json(plan_path, {
        'schema_version': 1,
        'status': 'running',
        'run_tag': args.run_tag,
        'objective': (
            'Test whether early high tau increases exploration while a final '
            'tau=0.3 plateau preserves late PPO stability and held-out Cmax.'
        ),
        'base_variant': 'N1_tail_cv_potential',
        'epochs': args.epochs,
        'train_sampling_size_per_epoch': args.train_sampling_size,
        'variants': VARIANTS,
        'seeds': [1, 2],
        'gpu_crossover': {
            'seed1': {'fixed': 0, 'annealed': 1},
            'seed2': {'fixed': 1, 'annealed': 0},
        },
        'data_policy': {
            'train': 'same 50/45/5 distribution-balanced generator',
            'tune_and_checkpoint': 'validation fixed partition offset 0, n=60',
            'heldout_selection': 'validation fixed partition offset 60, n=60',
            'blind_test': 'sealed; not touched by this screen',
        },
        'primary_gate': {
            'heldout_composite_mean_improvement': '>=0.5%',
            'per_seed_max_regression': '<=0.3%',
            'ood_stress_regression': '<=0.0%',
        },
        'mechanism_gate': {
            'early_entropy_increase': '>=10% over fixed tau',
            'late_tau': 0.3,
            'actor_step_completion': '>=90%',
            'post_update_old_policy_kl': '<=0.005',
            'completion_and_cycles': '100% completion and zero cycles',
        },
        'created_unix_time': time.time(),
    })


def run_one(args: argparse.Namespace, variant: str, seed: int) -> None:
    name = experiment_name(args.run_tag, variant, seed)
    record_path = args.suite_dir / 'records' / f'{variant}_seed{seed}.json'
    selection_path = (
        args.suite_dir / 'selection' / f'{variant}_seed{seed}.json'
    )
    if record_path.is_file() and read_json(record_path).get('status') == 'completed':
        print(f'[TauAblation] Reusing completed {variant} seed{seed}.', flush=True)
        return

    completed = latest_completed_run(name)
    train_command = training_command(args, variant, seed)
    atomic_json(
        args.suite_dir / 'commands' / f'{variant}_seed{seed}_train.json',
        {'command': train_command, 'shell_command': shlex.join(train_command)},
    )
    if completed is None:
        print(f'[TauAblation] Training {variant} seed{seed}.', flush=True)
        run_logged(
            train_command,
            args.suite_dir / 'logs' / f'{variant}_seed{seed}_train.log',
        )
        completed = latest_completed_run(name)
    if completed is None:
        raise RuntimeError(f'No completed Best checkpoint for {name}.')
    run_dir, checkpoint = completed

    eval_command = selection_command(args, checkpoint, selection_path, seed)
    atomic_json(
        args.suite_dir / 'commands' / f'{variant}_seed{seed}_selection.json',
        {'command': eval_command, 'shell_command': shlex.join(eval_command)},
    )
    if not selection_path.is_file():
        print(
            f'[TauAblation] Held-out validation {variant} seed{seed}.',
            flush=True,
        )
        run_logged(
            eval_command,
            args.suite_dir / 'logs' / f'{variant}_seed{seed}_selection.log',
        )
    evaluation = read_json(selection_path)
    drl = evaluation.get('methods', {}).get('DRL-G', {})
    if drl.get('status') != 'completed':
        raise RuntimeError(f'Incomplete held-out evaluation: {selection_path}')
    atomic_json(record_path, {
        'status': 'completed',
        'variant': variant,
        'seed': seed,
        'gpu': args.gpu,
        'cpu_set': args.cpu_set,
        'experiment_name': name,
        'run_dir': str(run_dir.resolve()),
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': sha256(checkpoint),
        'initial_checkpoint': str(args.initial_checkpoint.resolve()),
        'initial_checkpoint_sha256': sha256(args.initial_checkpoint),
        'schedule': VARIANTS[variant],
        'heldout_evaluation': str(selection_path.resolve()),
        'heldout_summary': drl.get('summary', {}),
        'completed_unix_time': time.time(),
    })


def main() -> None:
    args = parse_args()
    for required in (
        PYTHON, TRAIN, COMPARE, args.source_command_json,
        args.initial_checkpoint, args.potential_path, args.validation_dir,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.epochs != 4:
        raise ValueError('This pre-registered pilot requires exactly 4 epochs.')
    if args.train_sampling_size <= 0:
        raise ValueError('--train-sampling-size must be positive.')
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    ensure_plan(args)
    sequence = parse_sequence(args.sequence)
    lane_status = args.suite_dir / f'lane{args.lane}_status.json'
    atomic_json(lane_status, {
        'status': 'delayed' if args.start_delay_seconds else 'running',
        'lane': args.lane,
        'gpu': args.gpu,
        'cpu_set': args.cpu_set,
        'sequence': sequence,
        'started_unix_time': time.time(),
    })
    if args.start_delay_seconds:
        time.sleep(args.start_delay_seconds)
    try:
        for index, (variant, seed) in enumerate(sequence):
            atomic_json(lane_status, {
                'status': 'running',
                'lane': args.lane,
                'gpu': args.gpu,
                'cpu_set': args.cpu_set,
                'sequence': sequence,
                'current_index': index,
                'current_variant': variant,
                'current_seed': seed,
                'updated_unix_time': time.time(),
            })
            run_one(args, variant, seed)
        atomic_json(lane_status, {
            'status': 'completed',
            'lane': args.lane,
            'gpu': args.gpu,
            'cpu_set': args.cpu_set,
            'sequence': sequence,
            'completed_unix_time': time.time(),
        })
        expected_records = [
            args.suite_dir / 'records' / f'{variant}_seed{seed}.json'
            for variant in VARIANTS for seed in (1, 2)
        ]
        if ANALYZE.is_file() and all(path.is_file() for path in expected_records):
            analysis_command = [
                str(PYTHON), str(ANALYZE),
                '--suite-dir', str(args.suite_dir),
            ]
            run_logged(
                analysis_command,
                args.suite_dir / 'logs' / 'analysis.log',
            )
    except BaseException as exc:
        atomic_json(lane_status, {
            'status': 'failed',
            'lane': args.lane,
            'gpu': args.gpu,
            'cpu_set': args.cpu_set,
            'sequence': sequence,
            'error': repr(exc),
            'failed_unix_time': time.time(),
        })
        raise


if __name__ == '__main__':
    main()
