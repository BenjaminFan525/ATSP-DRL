#!/usr/bin/env python
"""Serve one persistent deterministic HKBZ validation pool per GPU."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from onpolicy.config.config import get_config
from onpolicy.scripts.train.train_hkbz import (
    apply_formal_safe_pipeline_manifest,
    make_eval_env,
    parse_args as parse_training_args,
)
from onpolicy.utils.shared_eval import (
    AUTHKEY,
    PROTOCOL_VERSION,
    format_cpu_set,
    parse_cpu_set,
    set_affinity,
    set_process_affinity,
)
from onpolicy.utils.checkpoint_contract import stage1_observation_metadata


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def remove_option(command: list[str], flag: str, has_value: bool = True) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + (2 if has_value else 1)]


def source_training_args(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    command = list(payload.get('command') or payload.get('evaluator_command') or ())
    if len(command) < 2:
        raise ValueError(f'Invalid source training command: {path}')
    args = command[2:]
    for flag in (
        '--checkpoint_dir', '--selection_checkpoint_dir',
        '--shared_eval_socket', '--shared_eval_cpu_set',
        '--shared_eval_timeout_seconds',
    ):
        remove_option(args, flag, has_value=True)
    for flag in (
        '--resume_stage1',
        '--reset_optimizers_on_resume',
        '--reset_value_normalizer_on_resume',
    ):
        remove_option(args, flag, has_value=False)
    return args


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-command-json', type=Path, required=True)
    parser.add_argument('--socket-path', type=Path, required=True)
    parser.add_argument('--cpu-pool', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--eval-dataset-dir', type=Path)
    parser.add_argument('--eval-case-offset', type=int)
    parser.add_argument('--max-eval-cases', type=int)
    parser.add_argument('--eval-partition-seed', type=int)
    parser.add_argument(
        '--eval-partition-stratify-by', choices=('', 'distribution', 'profile')
    )
    args = parser.parse_args()
    args.source_command_json = args.source_command_json.resolve()
    args.run_dir = args.run_dir.resolve()
    if not args.source_command_json.is_file():
        raise FileNotFoundError(args.source_command_json)
    if len(str(args.socket_path)) >= 100:
        raise ValueError(
            'AF_UNIX socket path must remain below 100 characters: '
            f'{args.socket_path}'
        )
    return args


def configure_runner(cli: argparse.Namespace):
    pool = parse_cpu_set(cli.cpu_pool)
    available = os.sched_getaffinity(0)
    if not pool.issubset(available):
        raise ValueError(
            'Evaluator CPU pool exceeds its systemd/taskset allowance: '
            f'pool={format_cpu_set(pool)}, '
            f'allowed={format_cpu_set(available)}'
        )
    set_affinity(0, pool)

    training_args = source_training_args(cli.source_command_json)
    all_args = parse_training_args(training_args, get_config())
    apply_formal_safe_pipeline_manifest(all_args, repository_root=ROOT)
    if cli.eval_dataset_dir is not None:
        all_args.eval_dataset_dir = str(cli.eval_dataset_dir.resolve())
    if cli.eval_case_offset is not None:
        all_args.eval_case_offset = int(cli.eval_case_offset)
    if cli.max_eval_cases is not None:
        all_args.max_eval_cases = int(cli.max_eval_cases)
    if cli.eval_partition_seed is not None:
        all_args.eval_partition_seed = int(cli.eval_partition_seed)
    if cli.eval_partition_stratify_by is not None:
        all_args.eval_partition_stratify_by = cli.eval_partition_stratify_by
    if all_args.hindsight_reward_mode == 'iga_potential':
        weights = Path(all_args.iga_potential_weights_path).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        all_args.iga_potential_weights_path = str(weights.resolve())
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False
    all_args.use_wandb = False
    all_args.use_eval = True
    all_args.checkpoint_dir = None
    all_args.selection_checkpoint_dir = None
    all_args.resume_stage1 = False
    all_args.reset_optimizers_on_resume = False
    all_args.reset_value_normalizer_on_resume = False
    all_args.plane_bc_pretrain_epochs = 0
    all_args.n_rollout_threads = all_args.n_eval_rollout_threads

    torch.multiprocessing.set_sharing_strategy(all_args.torch_mp_sharing_strategy)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device(str(all_args.device))
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
        torch.set_num_threads(all_args.n_training_threads)

    eval_envs = None
    try:
        eval_envs, case_counts = make_eval_env(all_args)
        with Path(all_args.ac_config).open('r', encoding='utf-8') as source:
            ac_config = yaml.safe_load(source)
        cli.run_dir.mkdir(parents=True, exist_ok=True)
        from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
        runner = HKBZ_Runner({
            'all_args': all_args,
            'envs': eval_envs,
            'eval_envs': eval_envs,
            'device': device,
            'run_dir': cli.run_dir,
            'ac_config': ac_config,
            'num_agents': all_args.max_agent_num + (
                all_args.max_device_num
                if all_args.resource_policy == 'drl' else 0
            ),
            'num_envs': max(case_counts),
            'eval_case_counts': case_counts,
            'eval_env_factory': None,
            'release_eval_envs_after_eval': False,
            'evaluation_only': True,
        })
        return runner, eval_envs, pool
    except BaseException:
        if eval_envs is not None:
            eval_envs.close()
        raise


def bind_evaluator(eval_envs, requested, pool) -> None:
    cpus = parse_cpu_set(requested)
    if not cpus.issubset(pool):
        raise ValueError(
            'Requested validation CPU set is outside the GPU-local pool: '
            f'requested={format_cpu_set(cpus)}, pool={format_cpu_set(pool)}'
        )
    suspended = eval_envs.suspend_async_graph_cloning()
    try:
        set_process_affinity(os.getpid(), cpus)
        for process in eval_envs.ps:
            if process.pid is None or not process.is_alive():
                raise RuntimeError(
                    f'Evaluation worker is not alive: pid={process.pid}, '
                    f'exitcode={process.exitcode}'
                )
            set_process_affinity(process.pid, cpus)
    finally:
        if suspended:
            eval_envs.resume_async_graph_cloning()


def evaluate_request(runner, eval_envs, pool, request: dict) -> dict:
    if int(request.get('protocol_version', -1)) != PROTOCOL_VERSION:
        raise ValueError('Shared-evaluator protocol version mismatch.')
    if int(request.get('n_eval_rollout_threads', -1)) != len(eval_envs.ps):
        raise ValueError(
            'Shared evaluator worker-count mismatch: '
            f'requested={request.get("n_eval_rollout_threads")}, '
            f'service={len(eval_envs.ps)}'
        )
    configured_arch = (
        runner.policy.ac.plane_order_mode,
        runner.policy.ac.plane_pair_decoder,
    )
    requested_arch = (
        request.get('plane_order_mode'),
        request.get('plane_pair_decoder'),
    )
    if requested_arch != configured_arch:
        raise ValueError(
            f'Shared evaluator architecture mismatch: '
            f'requested={requested_arch}, service={configured_arch}'
        )
    requested_global_mode = str(request.get('global_feature_mode', ''))
    expected_observation = stage1_observation_metadata(
        requested_global_mode
    )
    if request.get('observation_schema_id') != expected_observation[
        'observation_schema_id'
    ]:
        raise ValueError('Shared-evaluator observation schema mismatch.')
    if request.get('environment_semantics_version') != expected_observation[
        'environment_semantics_version'
    ]:
        raise ValueError('Shared-evaluator environment semantics mismatch.')
    supported_global_modes = set(
        runner.envs.call('set_global_feature_mode', requested_global_mode)
    )
    if supported_global_modes != {requested_global_mode}:
        raise RuntimeError(
            'Validation workers rejected global feature mode: '
            f'requested={requested_global_mode!r}, '
            f'observed={sorted(supported_global_modes)!r}'
        )
    bind_evaluator(eval_envs, request['cpu_set'], pool)

    seed = int(request['seed'])
    eval_envs.call_each(
        'seed',
        [((seed * 50000) + rank * 10000,) for rank in range(len(eval_envs.ps))],
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    checkpoint_path = Path(request['checkpoint_path'])
    if not checkpoint_path.is_absolute() or not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('request_id') != request.get('request_id'):
        raise ValueError('Shared-evaluator checkpoint request ID mismatch.')
    if checkpoint.get('global_feature_mode') != requested_global_mode:
        raise ValueError('Shared-evaluator global feature metadata mismatch.')
    if checkpoint.get('observation_schema_id') != expected_observation[
        'observation_schema_id'
    ]:
        raise ValueError('Shared-evaluator checkpoint schema mismatch.')
    if checkpoint.get(
        'environment_semantics_version'
    ) != expected_observation['environment_semantics_version']:
        raise ValueError('Shared-evaluator checkpoint semantics mismatch.')
    runner.policy.load_model_state(checkpoint['model'])
    runner.trainer.policy = runner.policy
    runner.policy.ac.tau = float(request['evaluation_tau'])
    runner.trainer.prep_rollout()
    return runner._eval_with_envs(
        evaluation_label=request.get('evaluation_label'),
        finalize=False,
    )


def serve(cli: argparse.Namespace) -> int:
    runner, eval_envs, pool = configure_runner(cli)
    socket_path = cli.socket_path
    status_path = cli.run_dir / 'service_status.json'
    listener = None
    if socket_path.exists():
        socket_path.unlink()
    try:
        listener = Listener(
            str(socket_path),
            family='AF_UNIX',
            backlog=8,
            authkey=AUTHKEY,
        )
        os.chmod(socket_path, 0o600)
        atomic_json(status_path, {
            'status': 'ready',
            'pid': os.getpid(),
            'socket_path': str(socket_path),
            'cpu_pool': format_cpu_set(pool),
            'worker_pids': [process.pid for process in eval_envs.ps],
            'worker_count': len(eval_envs.ps),
            'ready_unix_time': time.time(),
        })
        print(
            '[SharedEval] ready '
            f'socket={socket_path} workers={len(eval_envs.ps)} '
            f'cpu_pool={format_cpu_set(pool)}',
            flush=True,
        )
        while True:
            connection = listener.accept()
            request = None
            started = time.monotonic()
            try:
                request = connection.recv()
                if request.get('operation') == 'ping':
                    connection.send({
                        'ok': True,
                        'protocol_version': PROTOCOL_VERSION,
                        'pid': os.getpid(),
                        'worker_count': len(eval_envs.ps),
                        'cpu_pool': format_cpu_set(pool),
                    })
                    continue
                if request.get('operation') != 'evaluate':
                    raise ValueError(
                        f'Unsupported evaluator operation: '
                        f'{request.get("operation")!r}'
                    )
                request_id = str(request.get('request_id', ''))
                print(
                    f'[SharedEval] start request={request_id} '
                    f'seed={request.get("seed")} '
                    f'cpus={request.get("cpu_set")} '
                    f'label={request.get("evaluation_label")}',
                    flush=True,
                )
                evaluation = evaluate_request(runner, eval_envs, pool, request)
                elapsed = time.monotonic() - started
                connection.send({
                    'ok': True,
                    'protocol_version': PROTOCOL_VERSION,
                    'request_id': request_id,
                    'evaluation_seconds': float(elapsed),
                    'evaluation': evaluation,
                })
                print(
                    f'[SharedEval] completed request={request_id} '
                    f'elapsed={elapsed:.1f}s '
                    f'cases={evaluation["case_count"]}',
                    flush=True,
                )
            except BaseException as error:
                response = {
                    'ok': False,
                    'protocol_version': PROTOCOL_VERSION,
                    'request_id': (
                        request.get('request_id')
                        if isinstance(request, dict) else None
                    ),
                    'error': f'{type(error).__name__}: {error}',
                    'traceback': traceback.format_exc(),
                }
                try:
                    connection.send(response)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                print(
                    f'[SharedEval][Error] {response["error"]}\n'
                    f'{response["traceback"]}',
                    flush=True,
                )
            finally:
                connection.close()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        atomic_json(status_path, {
            'status': 'stopped',
            'pid': os.getpid(),
            'stopped_unix_time': time.time(),
        })
        if listener is not None:
            listener.close()
        eval_envs.close()
        writer = getattr(runner, 'writter', None)
        if writer is not None:
            writer.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
    return 0


def main() -> int:
    cli = parse_cli()

    def terminate(signum, _frame):
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    return serve(cli)


if __name__ == '__main__':
    raise SystemExit(main())
