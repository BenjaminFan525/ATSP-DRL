#!/usr/bin/env python
"""Serve one persistent deterministic HKBZ validation pool per GPU."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from multiprocessing import AuthenticationError
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
from onpolicy.utils.training_stage import protected_parameter_summary
from onpolicy.utils.stage2_bc_contract import ready_head_config, ready_head_signature


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
    parser.add_argument(
        '--cuda-memory-fraction',
        type=float,
        default=0.0,
        help='per-process CUDA allocator fraction for the shared evaluator',
    )
    parser.add_argument('--eval-partition-seed', type=int)
    parser.add_argument(
        '--eval-partition-stratify-by', choices=('', 'distribution', 'profile')
    )
    args = parser.parse_args()
    args.source_command_json = args.source_command_json.resolve()
    args.run_dir = args.run_dir.resolve()
    if not args.source_command_json.is_file():
        raise FileNotFoundError(args.source_command_json)
    if not np.isfinite(args.cuda_memory_fraction) or not (
        args.cuda_memory_fraction == 0.0
        or 0.0 < args.cuda_memory_fraction <= 1.0
    ):
        raise ValueError(
            '--cuda-memory-fraction must be 0 or a finite value in (0, 1].'
        )
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
    from onpolicy.utils.stage2_bc_contract import configure_bc_determinism
    configure_bc_determinism(all_args.stage2_bc_deterministic)
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
    # HKBZ_Runner validates recurrent *training* batches against the number of
    # environment slices during construction.  A shared evaluator never builds
    # or consumes PPO mini-batches, but small canary partitions can contain
    # fewer cases than the source training manifest's mini_batch_size.  Clamp
    # this otherwise-unused value so evaluation-only runners remain valid
    # without weakening the training-side safety check.
    all_args.mini_batch_size = min(
        int(all_args.mini_batch_size),
        int(all_args.n_rollout_threads),
    )

    torch.multiprocessing.set_sharing_strategy(all_args.torch_mp_sharing_strategy)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device(str(all_args.device))
        evaluator_fraction = float(cli.cuda_memory_fraction)
        if evaluator_fraction > 0.0:
            torch.cuda.set_per_process_memory_fraction(
                evaluator_fraction, device=device
            )
            total_gib = (
                torch.cuda.get_device_properties(device).total_memory
                / float(1024 ** 3)
            )
            print(
                '[SharedEvalMemoryLimit] '
                f'fraction={evaluator_fraction:.6f} '
                f'allocator_limit_gib={total_gib * evaluator_fraction:.3f}',
                flush=True,
            )
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


def _request_cache_key(request: dict) -> str:
    """Key deterministic validation by policy and evaluation semantics."""

    payload = {
        **ready_head_config(request),
        'model_sha256': str(request.get('model_sha256', '')),
        'evaluation_tau': float(request.get('evaluation_tau', 0.0)),
        'seed': int(request.get('seed', 0)),
        'max_eval_cases': int(request.get('max_eval_cases', 0)),
        'plane_order_mode': str(request.get('plane_order_mode', '')),
        'plane_pair_decoder': str(request.get('plane_pair_decoder', '')),
        'stage1_baseline': str(request.get('stage1_baseline', 'proposed')),
        'device_policy_head_mode': str(request.get(
            'device_policy_head_mode', ''
        )),
        'ordinary_device_type_count': int(request.get(
            'ordinary_device_type_count', 0
        )),
        'device_timing_head': bool(request.get(
            'device_timing_head', False
        )),
        'device_global_matching': bool(request.get(
            'device_global_matching', False
        )),
        'request_ready_prediction': bool(request.get(
            'request_ready_prediction', False
        )),
        'request_ready_time_scale': float(request.get(
            'request_ready_time_scale', 3600.0
        )),
        'request_ready_policy_injection': str(request.get(
            'request_ready_policy_injection', 'learned'
        )),
        'device_resource_adapter': bool(request.get(
            'device_resource_adapter', False
        )),
        'global_feature_mode': str(request.get('global_feature_mode', '')),
        'observation_schema_id': str(request.get('observation_schema_id', '')),
        'environment_semantics_version': str(request.get(
            'environment_semantics_version', ''
        )),
        'resource_planning_config': request.get('resource_planning_config'),
        'n_eval_rollout_threads': int(request.get(
            'n_eval_rollout_threads', 0
        )),
    }
    if not payload['model_sha256']:
        raise ValueError('Shared-evaluator request has no model digest.')
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def _limited_case_counts(case_counts, requested_max_cases):
    """Select a balanced prefix from each persistent validation worker."""

    original = [int(value) for value in case_counts]
    total = sum(original)
    requested = int(requested_max_cases)
    if requested <= 0 or requested >= total:
        return original
    if requested < len(original):
        raise ValueError(
            'Shared shard-canary cases cannot be fewer than evaluator workers: '
            f'cases={requested}, workers={len(original)}.'
        )
    limited = [1] * len(original)
    remaining = requested - len(original)
    while remaining > 0:
        progressed = False
        for rank, capacity in enumerate(original):
            if remaining <= 0:
                break
            if limited[rank] < capacity:
                limited[rank] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError('Unable to construct shared-eval case prefix.')
    return limited


def _runner_for_request(runner, eval_envs, request: dict, cli):
    """Rebuild only the lightweight model when an architecture changes.

    Validation workers and their prefetched graphs remain persistent.  This lets
    Scheduling-baseline and resource-head experiments on one GPU retain one
    common validation pool without silently evaluating a checkpoint through a
    different architecture.
    """

    requested_order = str(request.get('plane_order_mode', ''))
    requested_decoder = str(request.get('plane_pair_decoder', ''))
    requested_baseline = str(request.get(
        'stage1_baseline', 'proposed'
    )).strip().lower().replace('-', '_')
    requested_mode = str(request.get('device_policy_head_mode', ''))
    requested_count = int(request.get('ordinary_device_type_count', 0))
    requested_timing_head = bool(request.get('device_timing_head', False))
    requested_global_matching = bool(request.get(
        'device_global_matching', False
    ))
    requested_ready_prediction = bool(request.get(
        'request_ready_prediction', False
    ))
    requested_ready_scale = float(request.get(
        'request_ready_time_scale', 3600.0
    ))
    requested_ready_injection = str(request.get(
        'request_ready_policy_injection', 'learned'
    ))
    requested_resource_adapter = bool(request.get(
        'device_resource_adapter', False
    ))
    supported_modes = {'shared', 'type_adapter', 'per_type'}
    if requested_mode not in supported_modes:
        raise ValueError(
            'Unsupported shared-evaluator device head mode: '
            f'{requested_mode!r}.'
        )
    if requested_count <= 0:
        raise ValueError(
            'Shared-evaluator ordinary device type count must be positive.'
        )
    if requested_order not in {'fixed', 'learned'}:
        raise ValueError(
            f'Unsupported shared-evaluator plane order: {requested_order!r}.'
        )
    if requested_decoder not in {'cascade', 'joint_pair'}:
        raise ValueError(
            'Unsupported shared-evaluator plane decoder: '
            f'{requested_decoder!r}.'
        )
    if requested_baseline not in {
        'proposed', 'l2d', 'multi_ppo', 'fjsp_drl', 'daniel'
    }:
        raise ValueError(
            'Unsupported shared-evaluator Stage-1 baseline: '
            f'{requested_baseline!r}.'
        )
    configured = (
        str(runner.policy.ac.plane_order_mode),
        str(runner.policy.ac.plane_pair_decoder),
        str(getattr(runner.policy.ac, 'stage1_baseline', 'proposed')),
        str(runner.policy.ac.device_policy_head_mode),
        int(runner.policy.ac.ordinary_device_type_count),
        bool(runner.policy.ac.device_timing_head),
        bool(runner.policy.ac.device_global_matching),
        bool(runner.policy.ac.request_ready_prediction),
        float(runner.policy.ac.request_ready_time_scale),
        str(runner.policy.ac.request_ready_policy_injection),
        bool(runner.policy.ac.device_resource_adapter_enabled),
        *ready_head_signature(runner.policy.ac),
    )
    requested = (
        requested_order,
        requested_decoder,
        requested_baseline,
        requested_mode,
        requested_count,
        requested_timing_head,
        requested_global_matching,
        requested_ready_prediction,
        requested_ready_scale,
        requested_ready_injection,
        requested_resource_adapter,
        *ready_head_signature(request),
    )
    if configured == requested:
        return runner

    all_args = copy.deepcopy(runner.all_args)
    all_args.plane_order_mode = requested_order
    all_args.plane_pair_decoder = requested_decoder
    all_args.stage1_baseline = requested_baseline
    all_args.device_policy_head_mode = requested_mode
    all_args.ordinary_device_type_count = requested_count
    all_args.device_timing_head = requested_timing_head
    all_args.device_global_matching = requested_global_matching
    all_args.request_ready_prediction = requested_ready_prediction
    all_args.request_ready_time_scale = requested_ready_scale
    all_args.request_ready_policy_injection = requested_ready_injection
    all_args.device_resource_adapter = requested_resource_adapter
    for key, value in ready_head_config(request).items():
        setattr(all_args, key, value)
    ac_config = copy.deepcopy(runner.ac_config)
    case_counts = list(runner.eval_case_counts)
    device = runner.device
    num_agents = runner.num_agents

    writer = getattr(runner, 'writter', None)
    if writer is not None:
        writer.close()
    # Drop live CUDA tensors before constructing the replacement.  The worker
    # pool is deliberately untouched, so graph preprocessing is still shared.
    runner.policy = None
    runner.trainer = None
    runner.buffer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    architecture_run_dir = (
        cli.run_dir
        / (
            f'baseline_{requested_baseline}_order_{requested_order}_'
            f'decoder_{requested_decoder}_head_{requested_mode}'
        )
    )
    architecture_run_dir.mkdir(parents=True, exist_ok=True)
    from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
    replacement = HKBZ_Runner({
        'all_args': all_args,
        'envs': eval_envs,
        'eval_envs': eval_envs,
        'device': device,
        'run_dir': architecture_run_dir,
        'ac_config': ac_config,
        'num_agents': num_agents,
        'num_envs': max(case_counts),
        'eval_case_counts': case_counts,
        'eval_env_factory': None,
        'release_eval_envs_after_eval': False,
        'evaluation_only': True,
    })
    print(
        '[SharedEval] rebuilt evaluator model '
        f'configured={configured} requested={requested}',
        flush=True,
    )
    return replacement


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
        getattr(runner.policy.ac, 'stage1_baseline', 'proposed'),
    )
    requested_arch = (
        request.get('plane_order_mode'),
        request.get('plane_pair_decoder'),
        request.get('stage1_baseline', 'proposed'),
    )
    if requested_arch != configured_arch:
        raise ValueError(
            f'Shared evaluator architecture mismatch: '
            f'requested={requested_arch}, service={configured_arch}'
        )
    configured_device_arch = (
        str(runner.policy.ac.device_policy_head_mode),
        int(runner.policy.ac.ordinary_device_type_count),
        bool(runner.policy.ac.device_timing_head),
        bool(runner.policy.ac.device_global_matching),
        bool(runner.policy.ac.request_ready_prediction),
        float(runner.policy.ac.request_ready_time_scale),
        str(runner.policy.ac.request_ready_policy_injection),
        bool(runner.policy.ac.device_resource_adapter_enabled),
        *ready_head_signature(runner.policy.ac),
    )
    requested_device_arch = (
        str(request.get('device_policy_head_mode', '')),
        int(request.get('ordinary_device_type_count', 0)),
        bool(request.get('device_timing_head', False)),
        bool(request.get('device_global_matching', False)),
        bool(request.get('request_ready_prediction', False)),
        float(request.get('request_ready_time_scale', 3600.0)),
        str(request.get('request_ready_policy_injection', 'learned')),
        bool(request.get('device_resource_adapter', False)),
        *ready_head_signature(request),
    )
    if requested_device_arch != configured_device_arch:
        raise ValueError(
            'Shared evaluator device architecture mismatch: '
            f'requested={requested_device_arch}, '
            f'service={configured_device_arch}'
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
    requested_planning = request.get('resource_planning_config')
    if not isinstance(requested_planning, dict):
        raise ValueError(
            'Shared-evaluator resource planning metadata is missing.'
        )
    observed_planning = runner.envs.call(
        'set_resource_planning_config', requested_planning
    )
    if any(item != requested_planning for item in observed_planning):
        raise RuntimeError(
            'Validation workers rejected resource planning semantics: '
            f'requested={requested_planning!r}, '
            f'observed={observed_planning!r}'
        )
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
    if ready_head_config(checkpoint) != ready_head_config(request):
        raise ValueError('Shared-evaluator ready-head architecture/routing mismatch.')
    if checkpoint.get('request_id') != request.get('request_id'):
        raise ValueError('Shared-evaluator checkpoint request ID mismatch.')
    if checkpoint.get('global_feature_mode') != requested_global_mode:
        raise ValueError('Shared-evaluator global feature metadata mismatch.')
    if (
        checkpoint.get('plane_order_mode') != requested_arch[0]
        or checkpoint.get('plane_pair_decoder') != requested_arch[1]
        or checkpoint.get('stage1_baseline', 'proposed')
        != requested_arch[2]
    ):
        raise ValueError(
            'Shared-evaluator checkpoint scheduling architecture mismatch.'
        )
    if (
        checkpoint.get('device_policy_head_mode')
        != requested_device_arch[0]
        or int(checkpoint.get('ordinary_device_type_count', 0))
        != requested_device_arch[1]
        or bool(checkpoint.get('device_timing_head', False))
        != requested_device_arch[2]
        or bool(checkpoint.get('device_global_matching', False))
        != requested_device_arch[3]
        or bool(checkpoint.get('request_ready_prediction', False))
        != requested_device_arch[4]
        or float(checkpoint.get('request_ready_time_scale', 3600.0))
        != requested_device_arch[5]
        or str(checkpoint.get('request_ready_policy_injection', 'learned'))
        != requested_device_arch[6]
        or bool(checkpoint.get('device_resource_adapter', False))
        != requested_device_arch[7]
    ):
        raise ValueError(
            'Shared-evaluator checkpoint device architecture mismatch.'
        )
    if checkpoint.get('observation_schema_id') != expected_observation[
        'observation_schema_id'
    ]:
        raise ValueError('Shared-evaluator checkpoint schema mismatch.')
    if checkpoint.get(
        'environment_semantics_version'
    ) != expected_observation['environment_semantics_version']:
        raise ValueError('Shared-evaluator checkpoint semantics mismatch.')
    if checkpoint.get('resource_planning_config') != requested_planning:
        raise ValueError(
            'Shared-evaluator checkpoint resource planning mismatch.'
        )
    observed_model_sha256 = protected_parameter_summary(
        checkpoint['model'], prefixes=('',)
    )['sha256']
    if (
        checkpoint.get('model_sha256') != observed_model_sha256
        or request.get('model_sha256') != observed_model_sha256
    ):
        raise ValueError('Shared-evaluator model digest mismatch.')
    runner.policy.load_model_state(checkpoint['model'])
    runner.trainer.policy = runner.policy
    runner.policy.ac.tau = float(request['evaluation_tau'])
    runner.trainer.prep_rollout()
    original_case_counts = list(runner.eval_case_counts)
    runner.eval_case_counts = _limited_case_counts(
        original_case_counts,
        request.get('max_eval_cases', 0),
    )
    try:
        return runner._eval_with_envs(
            evaluation_label=request.get('evaluation_label'),
            finalize=False,
        )
    finally:
        runner.eval_case_counts = original_case_counts


def accept_connection(listener):
    """Ignore clients that disappear during the Listener auth handshake.

    ``multiprocessing.connection.Listener.accept`` authenticates a peer before
    returning its Connection.  A trainer stopped during a BC/PPO hand-off can
    therefore raise BrokenPipeError outside the normal per-request exception
    handler.  Only peer/handshake failures are retried; listener failures such
    as EBADF still propagate instead of causing a busy loop.
    """

    while True:
        try:
            return listener.accept()
        except (AuthenticationError, EOFError, ConnectionError) as error:
            print(
                '[SharedEval][Warning] discarded client during auth '
                f'handshake: {type(error).__name__}: {error}',
                flush=True,
            )


def serve(cli: argparse.Namespace) -> int:
    runner, eval_envs, pool = configure_runner(cli)
    socket_path = cli.socket_path
    status_path = cli.run_dir / 'service_status.json'
    listener = None
    if socket_path.exists():
        socket_path.unlink()
    evaluation_cache = {}
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
            'cuda_memory_fraction': float(cli.cuda_memory_fraction),
            'ready_unix_time': time.time(),
        })
        print(
            '[SharedEval] ready '
            f'socket={socket_path} workers={len(eval_envs.ps)} '
            f'cpu_pool={format_cpu_set(pool)}',
            flush=True,
        )
        while True:
            connection = accept_connection(listener)
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
                cache_key = _request_cache_key(request)
                cache_hit = cache_key in evaluation_cache
                if cache_hit:
                    evaluation = evaluation_cache[cache_key]
                else:
                    runner = _runner_for_request(
                        runner, eval_envs, request, cli
                    )
                    evaluation = evaluate_request(
                        runner, eval_envs, pool, request
                    )
                    evaluation_cache[cache_key] = evaluation
                elapsed = time.monotonic() - started
                connection.send({
                    'ok': True,
                    'protocol_version': PROTOCOL_VERSION,
                    'request_id': request_id,
                    'evaluation_seconds': float(elapsed),
                    'evaluation': evaluation,
                    'cache_hit': bool(cache_hit),
                })
                print(
                    f'[SharedEval] completed request={request_id} '
                    f'elapsed={elapsed:.1f}s '
                    f'cases={evaluation["case_count"]} '
                    f'cache_hit={cache_hit}',
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
