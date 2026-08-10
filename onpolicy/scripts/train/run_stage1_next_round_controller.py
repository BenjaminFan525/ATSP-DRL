#!/usr/bin/env python
"""Orchestrate the complete dual-A800 Stage-1 next-round experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from onpolicy.utils.shared_eval import SharedEvalClient


PYTHON = (ROOT.parent / 'conda/envs/maia/bin/python3.11').resolve()
TRIAL = ROOT / 'onpolicy/scripts/train/run_stage1_next_round_trial.py'
EVALUATOR = ROOT / 'onpolicy/scripts/train/shared_hkbz_evaluator.py'
OFFLINE_EVAL = ROOT / 'onpolicy/scripts/train/evaluate_stage1_shared_checkpoint.py'
ANALYZE = ROOT / 'onpolicy/envs/HKBZ/experiment/analyze_stage1_next_round.py'

SCREEN_WAVES = (
    ('M0_n2_control', 'M1_potential_ramp', 'M2_bc_kl_anneal'),
    ('M3_stress_replay', 'M4_tail_credit', 'M5_ramp_tail_combo'),
)
GPU_CPU_POOLS = {
    0: '0-35,72-107',
    1: '36-71,108-143',
}
GPU_CPU_SLICES = {
    0: ('0-11,72-83', '12-23,84-95', '24-35,96-107'),
    1: ('36-47,108-119', '48-59,120-131', '60-71,132-143'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--unit-prefix', required=True)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--source-command-json', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--potential-path', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--iga-180-json', type=Path, required=True)
    parser.add_argument('--iga-1800-json', type=Path, required=True)
    parser.add_argument('--screen-epochs', type=int, default=4)
    parser.add_argument('--formal-epochs', type=int, default=8)
    parser.add_argument('--screen-sampling-size', type=int, default=480)
    parser.add_argument('--formal-sampling-size', type=int, default=0)
    parser.add_argument('--partition-seed', type=int, default=20260803)
    parser.add_argument('--start-delay-step-seconds', type=int, default=30)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--phase-timeout-hours', type=float, default=96.0)
    parser.add_argument('--skip-blind-test', action='store_true')
    parser.add_argument('--predecessor-trial-unit', action='append', default=[])
    parser.add_argument('--predecessor-evaluator-unit', default='')
    parser.add_argument('--predecessor-timeout-hours', type=float, default=12.0)
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


def run(command: list[str], *, check=True, capture=False) -> subprocess.CompletedProcess:
    print(f'[ControllerCommand] {shlex.join(command)}', flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def unit_load_state(unit: str) -> str:
    completed = run(
        ['systemctl', '--user', 'show', unit, '--property=LoadState', '--value'],
        check=False,
        capture=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else 'not-found'


def require_new_unit(unit: str) -> None:
    state = unit_load_state(f'{unit}.service')
    if state not in ('', 'not-found'):
        raise RuntimeError(f'Refusing to reuse existing unit {unit}: {state}')


def transient_service(
    unit: str,
    cpu_set: str,
    gpu: int | None,
    log_path: Path,
    command: list[str],
) -> None:
    require_new_unit(unit)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    systemd = [
        'systemd-run', '--user', f'--unit={unit}', '--collect', '--same-dir',
        '--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID',
        '--setenv=PYTHONHASHSEED=0',
        '--setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
        '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1',
        '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
        '--property=Type=exec', '--property=Restart=no',
        '--property=KillMode=control-group', '--property=TimeoutStopSec=180',
        '--property=LimitNOFILE=65536',
        f'--property=AllowedCPUs={cpu_set}',
        f'--property=CPUAffinity={cpu_set}',
        '--property=CPUWeight=100', '--property=Nice=0',
        f'--property=StandardOutput=append:{log_path.resolve()}',
        f'--property=StandardError=append:{log_path.resolve()}',
    ]
    if gpu is not None:
        systemd.append(f'--setenv=CUDA_VISIBLE_DEVICES={gpu}')
    systemd.extend(['/usr/bin/taskset', '--cpu-list', cpu_set, *command])
    run(systemd)


def unit_active(unit: str) -> bool:
    completed = run(
        ['systemctl', '--user', 'is-active', '--quiet', f'{unit}.service'],
        check=False,
        capture=True,
    )
    return completed.returncode == 0


def stop_unit(unit: str) -> None:
    run(
        ['systemctl', '--user', 'stop', f'{unit}.service'],
        check=False,
        capture=True,
    )


def wait_for_predecessor(args: argparse.Namespace, status_path: Path, status: dict) -> None:
    trial_units = [
        unit.removesuffix('.service') for unit in args.predecessor_trial_unit
    ]
    evaluator_unit = args.predecessor_evaluator_unit.removesuffix('.service')
    if not trial_units and not evaluator_unit:
        return
    status.update({
        'stage': 'waiting_for_predecessor',
        'predecessor_trial_units': trial_units,
        'predecessor_evaluator_unit': evaluator_unit or None,
    })
    atomic_json(status_path, status)
    deadline = time.monotonic() + args.predecessor_timeout_hours * 3600.0
    while time.monotonic() < deadline:
        active = [unit for unit in trial_units if unit_active(unit)]
        if not active:
            break
        print(
            '[Handoff] Waiting for predecessor trials to finish naturally: '
            + ', '.join(active),
            flush=True,
        )
        time.sleep(args.poll_seconds)
    else:
        raise TimeoutError(
            f'Predecessor trials exceeded {args.predecessor_timeout_hours} hours.'
        )
    if evaluator_unit and unit_active(evaluator_unit):
        print(f'[Handoff] Stopping completed predecessor evaluator {evaluator_unit}.')
        stop_unit(evaluator_unit)
        stop_deadline = time.monotonic() + 300.0
        while unit_active(evaluator_unit) and time.monotonic() < stop_deadline:
            time.sleep(2.0)
        if unit_active(evaluator_unit):
            raise TimeoutError(f'Predecessor evaluator did not stop: {evaluator_unit}')


def wait_for_idle_gpus(timeout_seconds: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        completed = run(
            [
                'nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits',
            ],
            check=False,
            capture=True,
        )
        if completed.returncode != 0:
            raise RuntimeError('Cannot query GPU compute processes.')
        pids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not pids:
            return
        print(f'[Handoff] Waiting for GPU contexts to exit: {pids}', flush=True)
        time.sleep(2.0)
    raise TimeoutError('GPU compute contexts remained busy after predecessor cleanup.')


def evaluator_identity(args: argparse.Namespace, stage: str, gpu: int) -> tuple[str, str]:
    unit = f'{args.unit_prefix}-eval-{stage}-g{gpu}'
    socket = f'/tmp/{args.unit_prefix}-{stage}-g{gpu}.sock'
    if len(socket) >= 100:
        raise ValueError(f'AF_UNIX socket path is too long: {socket}')
    return unit, socket


def start_evaluators(
    args: argparse.Namespace,
    stage: str,
    dataset_dir: Path,
    offset: int,
    max_cases: int,
) -> dict[int, tuple[str, str]]:
    services = {}
    for gpu in (0, 1):
        unit, socket = evaluator_identity(args, stage, gpu)
        if Path(socket).exists():
            raise RuntimeError(f'Refusing to replace existing socket {socket}.')
        command = [
            str(PYTHON), '-u', str(EVALUATOR),
            '--source-command-json', str(args.source_command_json),
            '--socket-path', socket,
            '--cpu-pool', GPU_CPU_POOLS[gpu],
            '--run-dir', str(
                args.suite_dir / 'shared_evaluator' / f'{stage}_gpu{gpu}'
            ),
            '--eval-dataset-dir', str(dataset_dir),
            '--eval-case-offset', str(offset),
            '--max-eval-cases', str(max_cases),
            '--eval-partition-seed', str(args.partition_seed),
            '--eval-partition-stratify-by', 'profile',
        ]
        transient_service(
            unit,
            GPU_CPU_POOLS[gpu],
            gpu,
            args.suite_dir / 'service_logs' / f'{unit}.log',
            command,
        )
        services[gpu] = (unit, socket)

    deadline = time.monotonic() + 600.0
    pending = set(services)
    while pending and time.monotonic() < deadline:
        for gpu in list(pending):
            unit, socket = services[gpu]
            if not unit_active(unit):
                raise RuntimeError(f'Evaluator exited before readiness: {unit}')
            if not Path(socket).is_socket():
                continue
            try:
                reply = SharedEvalClient(socket, timeout_seconds=5.0).ping()
            except Exception:
                continue
            if int(reply.get('worker_count', -1)) != 60:
                raise RuntimeError(f'Expected 60 evaluator workers: {reply}')
            pending.remove(gpu)
            print(f'[Ready] {unit} socket={socket}', flush=True)
        if pending:
            time.sleep(2.0)
    if pending:
        raise TimeoutError(f'Evaluators did not become ready: {sorted(pending)}')
    return services


def stop_evaluators(services: dict[int, tuple[str, str]]) -> None:
    for unit, _ in services.values():
        stop_unit(unit)
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        if all(not unit_active(unit) for unit, _ in services.values()):
            return
        time.sleep(2.0)
    raise TimeoutError('Shared evaluators did not stop within 300 seconds.')


def trial_key(phase: str, variant: str, seed: int) -> str:
    return f'{phase}_{variant}_seed{seed}'


def launch_trial(
    args: argparse.Namespace,
    services: dict[int, tuple[str, str]],
    phase: str,
    variant: str,
    seed: int,
    gpu: int,
    lane: int,
    epochs: int,
    sampling_size: int,
) -> tuple[str, Path]:
    cpu_set = GPU_CPU_SLICES[gpu][lane]
    unit = f'{args.unit_prefix}-{phase[0]}-g{gpu}l{lane}-{variant[:2]}s{seed}'
    _, socket = services[gpu]
    key = trial_key(phase, variant, seed)
    command = [
        str(PYTHON), '-u', str(TRIAL),
        '--phase', phase,
        '--variant', variant,
        '--seed', str(seed),
        '--gpu', str(gpu),
        '--cpu-set', cpu_set,
        '--shared-eval-socket', socket,
        '--run-tag', args.run_tag,
        '--suite-dir', str(args.suite_dir),
        '--source-command-json', str(args.source_command_json),
        '--initial-checkpoint', str(args.initial_checkpoint),
        '--potential-path', str(args.potential_path),
        '--validation-dir', str(args.validation_dir),
        '--start-delay-seconds', str(lane * args.start_delay_step_seconds),
        '--epochs', str(epochs),
        '--train-sampling-size', str(sampling_size),
        '--partition-seed', str(args.partition_seed),
    ]
    transient_service(
        unit,
        cpu_set,
        gpu,
        args.suite_dir / 'service_logs' / f'{unit}.log',
        command,
    )
    return unit, args.suite_dir / 'trial_status' / f'{key}.json'


def wait_trials(
    args: argparse.Namespace,
    launched: list[tuple[str, Path]],
    label: str,
) -> None:
    deadline = time.monotonic() + args.phase_timeout_hours * 3600.0
    remaining = {unit: status for unit, status in launched}
    while remaining and time.monotonic() < deadline:
        for unit, status_path in list(remaining.items()):
            if status_path.is_file():
                status = read_json(status_path)
                if status.get('status') == 'completed':
                    print(f'[Completed] {unit}', flush=True)
                    del remaining[unit]
                    continue
                if status.get('status') == 'failed':
                    raise RuntimeError(f'{unit} failed: {status.get("error")}')
            if not unit_active(unit):
                raise RuntimeError(
                    f'{unit} is no longer active and has no completed record; '
                    f'inspect {status_path}.'
                )
        if remaining:
            print(
                f'[Wait] {label}: {len(remaining)}/{len(launched)} trials remain.',
                flush=True,
            )
            time.sleep(args.poll_seconds)
    if remaining:
        raise TimeoutError(f'{label} exceeded phase timeout: {sorted(remaining)}')


def run_screen(
    args: argparse.Namespace,
    services: dict[int, tuple[str, str]],
) -> Path:
    for wave_index, variants in enumerate(SCREEN_WAVES, start=1):
        launched = []
        for gpu, seed in ((0, 1), (1, 2)):
            for lane, variant in enumerate(variants):
                launched.append(launch_trial(
                    args, services, 'screen', variant, seed, gpu, lane,
                    args.screen_epochs, args.screen_sampling_size,
                ))
        wait_trials(args, launched, f'screen wave {wave_index}')

    summary_path = args.suite_dir / 'analysis/screen_summary.json'
    run([
        str(PYTHON), str(ANALYZE),
        '--mode', 'screen',
        '--suite-dir', str(args.suite_dir),
        '--variants', *[variant for wave in SCREEN_WAVES for variant in wave],
        '--seeds', '1', '2',
        '--output', str(summary_path),
    ])
    return summary_path


def formal_assignments(top: list[str]) -> list[tuple[str, int, int, int]]:
    first, second = top
    return [
        (first, 1, 0, 0),
        (first, 3, 0, 1),
        (second, 2, 0, 2),
        (first, 2, 1, 0),
        (second, 1, 1, 1),
        (second, 3, 1, 2),
    ]


def run_formal(
    args: argparse.Namespace,
    services: dict[int, tuple[str, str]],
    top: list[str],
) -> list[tuple[str, int, int, int]]:
    assignments = formal_assignments(top)
    launched = [
        launch_trial(
            args, services, 'formal', variant, seed, gpu, lane,
            args.formal_epochs, args.formal_sampling_size,
        )
        for variant, seed, gpu, lane in assignments
    ]
    wait_trials(args, launched, 'formal three-seed confirmation')
    return assignments


def evaluate_assignments(
    args: argparse.Namespace,
    services: dict[int, tuple[str, str]],
    assignments: list[tuple[str, int, int, int]],
    output_dir: Path,
    label_prefix: str,
) -> None:
    by_gpu = {0: [], 1: []}
    for assignment in assignments:
        by_gpu[assignment[2]].append(assignment)

    def evaluate_gpu(gpu: int) -> None:
        _, socket = services[gpu]
        for variant, seed, _, lane in by_gpu[gpu]:
            record = read_json(
                args.suite_dir / 'records' / f'formal_{variant}_seed{seed}.json'
            )
            output = output_dir / f'{variant}_seed{seed}.json'
            run([
                str(PYTHON), str(OFFLINE_EVAL),
                '--socket-path', socket,
                '--checkpoint', record['checkpoint'],
                '--cpu-set', GPU_CPU_SLICES[gpu][lane],
                '--seed', str(seed),
                '--output', str(output),
                '--label', f'{label_prefix}_{variant}_seed{seed}',
                '--evaluation-tau', '0.3',
            ])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(evaluate_gpu, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()


def blind_summary(args: argparse.Namespace, winner: str) -> dict:
    outputs = [
        read_json(args.suite_dir / 'blind_test' / f'{winner}_seed{seed}.json')
        for seed in (1, 2, 3)
    ]
    seed_maps = []
    seed_means = []
    for payload in outputs:
        evaluation = payload['evaluation']
        if (
            evaluation['completion_rate'] != 1.0
            or evaluation['cycle_count'] != 0
            or evaluation['timeout_count'] != 0
        ):
            raise RuntimeError('Blind-test checkpoint failed completion gates.')
        rows = {row['case_id']: row for row in evaluation['records']}
        seed_maps.append(rows)
        seed_means.append(float(np.mean([
            row['makespan'] for row in rows.values()
        ])))
    if any(rows.keys() != seed_maps[0].keys() for rows in seed_maps[1:]):
        raise RuntimeError('Blind-test seed coverage mismatch.')

    def iga_cases(path: Path) -> dict:
        payload = read_json(path)
        return {
            row['case_id']: row
            for row in payload['methods']['IGA']['cases']
        }

    iga180 = iga_cases(args.iga_180_json)
    iga1800 = iga_cases(args.iga_1800_json)
    if seed_maps[0].keys() != iga180.keys() or iga180.keys() != iga1800.keys():
        raise RuntimeError('Blind-test and IGA case identities differ.')
    rows = []
    for key in sorted(seed_maps[0]):
        seed_values = [float(seed_map[key]['makespan']) for seed_map in seed_maps]
        learned_mean = float(np.mean(seed_values))
        row180 = iga180[key]
        row1800 = iga1800[key]
        rows.append({
            'case_id': key,
            'profile': row1800['profile'],
            'distribution': row1800['distribution'],
            'seed_makespans': seed_values,
            'learned_mean': learned_mean,
            'iga_180': float(row180['makespan']),
            'iga_1800': float(row1800['makespan']),
            'delta_vs_iga_180': learned_mean - float(row180['makespan']),
            'delta_vs_iga_1800': learned_mean - float(row1800['makespan']),
        })
    learned_mean = float(np.mean(seed_means))
    iga180_mean = float(np.mean([row['iga_180'] for row in rows]))
    iga1800_mean = float(np.mean([row['iga_1800'] for row in rows]))
    rng = np.random.default_rng(20260808)
    bootstrap_delta = np.empty(10000, dtype=np.float64)
    learned_matrix = np.asarray(
        [row['seed_makespans'] for row in rows], dtype=np.float64
    ).T
    iga1800_values = np.asarray(
        [row['iga_1800'] for row in rows], dtype=np.float64
    )
    for index in range(bootstrap_delta.size):
        sampled_seeds = rng.integers(0, 3, size=3)
        sampled_cases = rng.integers(0, len(rows), size=len(rows))
        bootstrap_delta[index] = float(
            learned_matrix[sampled_seeds][:, sampled_cases].mean()
            - iga1800_values[sampled_cases].mean()
        )

    def grouped(field: str) -> dict:
        result = {}
        for name in sorted({row[field] for row in rows}):
            selected = [row for row in rows if row[field] == name]
            learned = float(np.mean([row['learned_mean'] for row in selected]))
            baseline = float(np.mean([row['iga_1800'] for row in selected]))
            result[name] = {
                'case_count': len(selected),
                'learned_mean': learned,
                'iga_1800_mean': baseline,
                'delta_vs_iga_1800': learned - baseline,
                'relative_gap_vs_iga_1800': learned / baseline - 1.0,
            }
        return result

    return {
        'schema_version': 1,
        'created_unix_time': time.time(),
        'winner': winner,
        'seed_means': seed_means,
        'learned_mean': learned_mean,
        'learned_std': float(np.std(seed_means, ddof=1)),
        'iga_180_mean': iga180_mean,
        'iga_1800_mean': iga1800_mean,
        'delta_vs_iga_180': learned_mean - iga180_mean,
        'delta_vs_iga_1800': learned_mean - iga1800_mean,
        'paired_hierarchical_bootstrap_delta_vs_iga_1800_ci95': [
            float(value) for value in np.quantile(
                bootstrap_delta, [0.025, 0.975]
            )
        ],
        'bootstrap_probability_at_or_better_than_iga_1800': float(
            np.mean(bootstrap_delta <= 0.0)
        ),
        'relative_gap_vs_iga_1800': learned_mean / iga1800_mean - 1.0,
        'near_iga_threshold': 1.03 * iga1800_mean,
        'near_iga_achieved': learned_mean <= 1.03 * iga1800_mean,
        'reach_iga_achieved': learned_mean <= iga1800_mean,
        'severe_gap_over_800_count': int(sum(
            row['delta_vs_iga_1800'] > 800.0 for row in rows
        )),
        'severe_gap_over_800_count_by_seed': [
            int(sum(
                float(seed_maps[seed_index][row['case_id']]['makespan'])
                - row['iga_1800'] > 800.0
                for row in rows
            ))
            for seed_index in range(3)
        ],
        'by_distribution': grouped('distribution'),
        'by_profile': grouped('profile'),
        'worst_10_by_gap_vs_iga_1800': sorted(
            rows,
            key=lambda row: row['delta_vs_iga_1800'],
            reverse=True,
        )[:10],
        'cases': rows,
    }


def write_plan(args: argparse.Namespace) -> None:
    atomic_json(args.suite_dir / 'experiment_plan.json', {
        'schema_version': 1,
        'status': 'running',
        'run_tag': args.run_tag,
        'objective': (
            'Determine whether staged shaping, an annealed BC trust region, '
            'stress replay, or tail-credit redistribution closes the sealed '
            'Stage-1 blind-test gap to IGA without OOD-stress regression.'
        ),
        'screen_waves': [list(wave) for wave in SCREEN_WAVES],
        'screen_design': 'six methods x seeds 1/2 x four PPO epochs',
        'formal_design': 'top two methods x seeds 1/2/3 x eight PPO epochs',
        'gpu_policy': 'three isolated trainers and one shared evaluator per GPU',
        'cpu_pools': GPU_CPU_POOLS,
        'cpu_slices': GPU_CPU_SLICES,
        'tune_partition': 'validation offset 0, 60 cases',
        'heldout_partition': 'validation offset 60, 60 cases; locks winner',
        'blind_test': (
            'test60 remains sealed until the heldout winner is locked; then '
            'three seeds are compared case-by-case with IGA-180/IGA-1800'
        ),
        'created_unix_time': time.time(),
    })


def main() -> None:
    args = parse_args()
    args.suite_dir = args.suite_dir.resolve()
    required = (
        PYTHON, TRIAL, EVALUATOR, OFFLINE_EVAL, ANALYZE,
        args.source_command_json, args.initial_checkpoint,
        args.potential_path, args.validation_dir, args.test_dir,
        args.iga_180_json, args.iga_1800_json,
    )
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.screen_epochs != 4 or args.formal_epochs != 8:
        raise ValueError('The pre-registered design requires screen=4, formal=8.')
    if args.screen_sampling_size != 480 or args.formal_sampling_size != 0:
        raise ValueError('The pre-registered sampling budgets are screen=480/formal=0.')

    args.suite_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.suite_dir / 'controller_status.json'
    write_plan(args)
    status = {'status': 'running', 'stage': 'initializing', 'pid': os.getpid()}
    atomic_json(status_path, status)
    services = {}
    try:
        wait_for_predecessor(args, status_path, status)
        wait_for_idle_gpus()
        status['stage'] = 'initializing_evaluators'
        atomic_json(status_path, status)
        services = start_evaluators(
            args, 'tune', args.validation_dir, offset=0, max_cases=60
        )
        status['stage'] = 'screen'
        atomic_json(status_path, status)
        screen_path = run_screen(args, services)
        screen = read_json(screen_path)
        if not screen['formal_authorized']:
            status.update({
                'status': 'stopped_by_gate',
                'stage': 'screen',
                'reason': screen['stop_reason'],
                'completed_unix_time': time.time(),
            })
            atomic_json(status_path, status)
            return

        top = list(screen['promoted_variants'])
        status.update({'stage': 'formal', 'promoted_variants': top})
        atomic_json(status_path, status)
        assignments = run_formal(args, services, top)
        stop_evaluators(services)
        services = start_evaluators(
            args, 'heldout', args.validation_dir, offset=60, max_cases=60
        )
        status['stage'] = 'formal_heldout'
        atomic_json(status_path, status)
        evaluate_assignments(
            args, services, assignments, args.suite_dir / 'heldout', 'heldout'
        )
        formal_path = args.suite_dir / 'analysis/formal_heldout_summary.json'
        run([
            str(PYTHON), str(ANALYZE),
            '--mode', 'formal',
            '--suite-dir', str(args.suite_dir),
            '--variants', *top,
            '--seeds', '1', '2', '3',
            '--output', str(formal_path),
        ])
        formal = read_json(formal_path)
        if not formal['blind_test_authorized']:
            status.update({
                'status': 'stopped_by_gate',
                'stage': 'formal_heldout',
                'reason': 'No formal candidate passed heldout safety gates.',
                'completed_unix_time': time.time(),
            })
            atomic_json(status_path, status)
            return
        winner = str(formal['winner'])
        status.update({'stage': 'blind_test', 'winner': winner})
        atomic_json(status_path, status)
        if args.skip_blind_test:
            status.update({
                'status': 'completed_before_blind',
                'completed_unix_time': time.time(),
            })
            atomic_json(status_path, status)
            return

        stop_evaluators(services)
        services = start_evaluators(
            args, 'blind', args.test_dir, offset=0, max_cases=60
        )
        blind_assignments = [
            (winner, 1, 0, 0),
            (winner, 3, 0, 1),
            (winner, 2, 1, 0),
        ]
        evaluate_assignments(
            args,
            services,
            blind_assignments,
            args.suite_dir / 'blind_test',
            'blind_test',
        )
        summary = blind_summary(args, winner)
        atomic_json(args.suite_dir / 'analysis/blind_test_vs_iga.json', summary)
        status.update({
            'status': 'completed',
            'stage': 'completed',
            'blind_summary': str(
                (args.suite_dir / 'analysis/blind_test_vs_iga.json').resolve()
            ),
            'near_iga_achieved': summary['near_iga_achieved'],
            'reach_iga_achieved': summary['reach_iga_achieved'],
            'completed_unix_time': time.time(),
        })
        atomic_json(status_path, status)
    except BaseException as error:
        status.update({
            'status': 'failed',
            'error': f'{type(error).__name__}: {error}',
            'failed_unix_time': time.time(),
        })
        atomic_json(status_path, status)
        raise
    finally:
        if services:
            try:
                stop_evaluators(services)
            except Exception as error:
                print(f'[CleanupWarning] {error}', flush=True)


if __name__ == '__main__':
    main()
