#!/usr/bin/env python3
"""Execute one preflighted departure-research command from a manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.utils.shared_eval import format_cpu_set, parse_cpu_set


def remove_option(command: list[str], flag: str, takes_value=True) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + (2 if takes_value else 1)]


def set_option(command: list[str], flag: str, value) -> None:
    remove_option(command, flag)
    command.extend([flag, str(value)])


def set_switch(command: list[str], flag: str, enabled=True) -> None:
    remove_option(command, flag, takes_value=False)
    if enabled:
        command.append(flag)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--command-key', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--cpu-set', required=True)
    parser.add_argument('--shared-eval-socket', required=True)
    parser.add_argument('--start-delay-seconds', type=float, default=0.0)
    parser.add_argument('--record', type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_delay_seconds < 0.0:
        raise ValueError('start delay must be non-negative.')
    requested_cpus = parse_cpu_set(args.cpu_set)
    allowed_cpus = frozenset(os.sched_getaffinity(0))
    if requested_cpus != allowed_cpus:
        raise RuntimeError(
            'Trainer cgroup/taskset mismatch: '
            f'requested={format_cpu_set(requested_cpus)}, '
            f'allowed={format_cpu_set(allowed_cpus)}'
        )
    visible = str(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    if visible != str(args.gpu):
        raise RuntimeError(
            f'CUDA_VISIBLE_DEVICES mismatch: expected {args.gpu}, got {visible!r}.'
        )
    payload = json.loads(args.manifest.read_text(encoding='utf-8'))
    entry = payload.get('commands', {}).get(args.command_key)
    if not entry:
        raise KeyError(f'Unknown command key {args.command_key!r}.')
    command = list(entry['argv'])
    set_option(command, '--shared_eval_socket', args.shared_eval_socket)
    set_option(command, '--shared_eval_cpu_set', args.cpu_set)
    set_option(command, '--shared_eval_timeout_seconds', 21600)
    set_option(command, '--n_eval_rollout_threads', 60)
    set_option(command, '--torch_mp_sharing_strategy', 'file_descriptor')
    set_option(command, '--ipc_timeout_seconds', 600)
    set_option(command, '--safe_async_graph_clone_workers', 4)
    remove_option(command, '--no_eval', takes_value=False)
    set_switch(command, '--use_eval', True)
    set_switch(command, '--safe_graph_batch_pipeline', True)
    set_switch(command, '--safe_dagger_teacher_overlap', True)
    record = {
        'schema_version': 1,
        'manifest': str(args.manifest.resolve()),
        'command_key': args.command_key,
        'gpu': args.gpu,
        'cpu_set': args.cpu_set,
        'shared_eval_socket': args.shared_eval_socket,
        'start_delay_seconds': args.start_delay_seconds,
        'launcher_pid': os.getpid(),
        'created_unix_time': time.time(),
        'command': command,
        'status': 'waiting' if args.start_delay_seconds else 'starting',
    }
    atomic_json(args.record, record)
    if args.start_delay_seconds:
        time.sleep(args.start_delay_seconds)
    record['status'] = 'exec'
    record['exec_unix_time'] = time.time()
    atomic_json(args.record, record)
    os.execv(command[0], command)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
