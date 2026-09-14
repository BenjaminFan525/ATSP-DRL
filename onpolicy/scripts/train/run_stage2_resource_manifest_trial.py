#!/usr/bin/env python3
"""Execute one preflighted Stage2 resource command from a suite manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.scripts.train.run_hkbz_two_stage_pipeline import (  # noqa: E402
    remove_option,
    set_option,
    set_switch,
)
from onpolicy.utils.shared_eval import format_cpu_set, parse_cpu_set  # noqa: E402
from onpolicy.utils.stage2_external_dispatch import guard_launch  # noqa: E402


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--command-key", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-set", required=True)
    parser.add_argument("--shared-eval-socket", required=True)
    parser.add_argument("--shared-eval-cpu-set")
    parser.add_argument("--eval-workers", type=int, required=True)
    parser.add_argument("--start-delay-seconds", type=float, default=0.0)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--external-owner-pid", type=int)
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--experiment-name")
    parser.add_argument("--n-rollout-threads", type=int)
    parser.add_argument("--mini-batch-size", type=int)
    parser.add_argument("--data-chunk-length", type=int)
    parser.add_argument("--max-graphs-per-forward", type=int)
    parser.add_argument("--grad-accumulation-steps", type=int)
    parser.add_argument("--actor-grad-accumulation-steps", type=int)
    parser.add_argument("--grad-accumulation-target-graphs", type=int)
    parser.add_argument("--actor-grad-accumulation-target-graphs", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resource-bc-checkpoint", type=Path)
    parser.add_argument(
        '--resource-bc-wait-timeout-seconds', type=float, default=0.0
    )
    return parser.parse_args()


def apply_runtime_overrides(command: list[str], args: argparse.Namespace) -> dict:
    """Apply an explicitly recorded one-lane throughput override."""

    overrides = {}
    memory_fraction = getattr(args, 'cuda_memory_fraction', None)
    if memory_fraction is not None:
        if not 0.0 < memory_fraction <= 1.0:
            raise ValueError('CUDA allocator fraction must be in (0, 1].')
        set_option(command, '--cuda_memory_fraction', memory_fraction)
        overrides['cuda_memory_fraction'] = memory_fraction
    if args.experiment_name:
        set_option(command, "--experiment_name", args.experiment_name)
        overrides["experiment_name"] = args.experiment_name
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"Stage2 recovery checkpoint does not exist: {resume_checkpoint}"
            )
        set_option(command, "--checkpoint_dir", str(resume_checkpoint))
        set_switch(command, "--resume_stage2", True)
        remove_option(command, "--resume_stage1", takes_value=False)
        remove_option(command, "--reset_optimizers_on_resume", takes_value=False)
        remove_option(
            command, "--reset_value_normalizer_on_resume", takes_value=False
        )
        overrides["resume_checkpoint"] = str(resume_checkpoint)
    resource_bc_checkpoint = getattr(args, "resource_bc_checkpoint", None)
    if resource_bc_checkpoint is not None:
        resource_bc_checkpoint = resource_bc_checkpoint.expanduser().resolve()
        if not resource_bc_checkpoint.is_file():
            raise FileNotFoundError(
                'Shared resource BC checkpoint does not exist: '
                f'{resource_bc_checkpoint}'
            )
        set_option(
            command,
            '--resource_bc_checkpoint',
            str(resource_bc_checkpoint),
        )
        set_option(command, '--device_bc_pretrain_epochs', 0)
        overrides['resource_bc_checkpoint'] = str(resource_bc_checkpoint)
    numeric_options = (
        ("n_rollout_threads", "--n_rollout_threads"),
        ("mini_batch_size", "--mini_batch_size"),
        ("data_chunk_length", "--data_chunk_length"),
        ("max_graphs_per_forward", "--max_graphs_per_forward"),
        ("grad_accumulation_steps", "--grad_accumulation_steps"),
        (
            "actor_grad_accumulation_steps",
            "--actor_grad_accumulation_steps",
        ),
        (
            "grad_accumulation_target_graphs",
            "--grad_accumulation_target_graphs",
        ),
        (
            "actor_grad_accumulation_target_graphs",
            "--actor_grad_accumulation_target_graphs",
        ),
    )
    for attribute, option in numeric_options:
        value = getattr(args, attribute, None)
        if value is None:
            continue
        if value <= 0:
            raise ValueError(f"{option} must be positive, got {value}.")
        set_option(command, option, value)
        overrides[attribute] = int(value)
    def effective_int(option: str):
        if option not in command:
            return None
        return int(command[command.index(option) + 1])

    n_rollout_threads = effective_int("--n_rollout_threads")
    mini_batch_size = effective_int("--mini_batch_size")
    data_chunk_length = effective_int("--data_chunk_length")
    max_graphs = effective_int("--max_graphs_per_forward")
    if (
        n_rollout_threads is not None
        and mini_batch_size is not None
        and mini_batch_size > n_rollout_threads
    ):
        raise ValueError(
            "Runtime PPO mini-batch exceeds rollout width: "
            f"mini_batch_size={mini_batch_size} > "
            f"n_rollout_threads={n_rollout_threads}."
        )
    if None not in (mini_batch_size, data_chunk_length, max_graphs):
        requested_graphs = mini_batch_size * data_chunk_length
        if requested_graphs > max_graphs:
            raise ValueError(
                "Runtime PPO forward exceeds --max-graphs-per-forward: "
                f"{mini_batch_size}*{data_chunk_length}={requested_graphs} "
                f"> {max_graphs}."
            )
    return overrides


def main() -> int:
    from onpolicy.utils.stage2_freeze_guard import reject_stage2_development
    reject_stage2_development()
    args = parse_args()
    adopted_code = guard_launch(args, atomic_json)
    if adopted_code is not None:
        return adopted_code
    requested_cpus = parse_cpu_set(args.cpu_set)
    allowed_cpus = frozenset(os.sched_getaffinity(0))
    if requested_cpus != allowed_cpus:
        raise RuntimeError(
            "Trainer cgroup affinity mismatch: "
            f"requested={format_cpu_set(requested_cpus)}, "
            f"allowed={format_cpu_set(allowed_cpus)}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match --gpu.")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    entry = payload.get("commands", {}).get(args.command_key)
    if not isinstance(entry, dict):
        raise KeyError(f"Unknown Stage2 method {args.command_key!r}.")
    command = list(entry["argv"])
    if (
        args.resource_bc_checkpoint is not None
        and args.resource_bc_wait_timeout_seconds > 0.0
    ):
        deadline = time.monotonic() + args.resource_bc_wait_timeout_seconds
        while (
            not args.resource_bc_checkpoint.expanduser().is_file()
            and time.monotonic() < deadline
        ):
            time.sleep(5.0)
        if not args.resource_bc_checkpoint.expanduser().is_file():
            raise TimeoutError(
                'Timed out waiting for shared resource BC checkpoint: '
                f'{args.resource_bc_checkpoint}'
            )
    runtime_overrides = apply_runtime_overrides(command, args)
    set_option(command, "--shared_eval_socket", args.shared_eval_socket)
    shared_eval_cpu_set = args.shared_eval_cpu_set or args.cpu_set
    parse_cpu_set(shared_eval_cpu_set)
    set_option(command, "--shared_eval_cpu_set", shared_eval_cpu_set)
    set_option(command, "--shared_eval_timeout_seconds", 21600)
    set_option(command, "--n_eval_rollout_threads", args.eval_workers)
    set_option(command, "--torch_mp_sharing_strategy", "file_descriptor")
    set_option(command, "--ipc_timeout_seconds", 600)
    set_option(command, "--safe_async_graph_clone_workers", 4)
    remove_option(command, "--no_eval", takes_value=False)
    set_switch(command, "--use_eval", True)
    set_switch(command, "--safe_graph_batch_pipeline", True)
    set_switch(command, "--safe_dagger_teacher_overlap", True)
    record = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "command_key": args.command_key,
        "gpu": args.gpu,
        "cpu_set": args.cpu_set,
        "shared_eval_socket": args.shared_eval_socket,
        "shared_eval_cpu_set": shared_eval_cpu_set,
        "eval_workers": args.eval_workers,
        "start_delay_seconds": args.start_delay_seconds,
        "launcher_pid": os.getpid(),
        "created_unix_time": time.time(),
        "command": command,
        "runtime_overrides": runtime_overrides,
        "status": "waiting" if args.start_delay_seconds else "starting",
    }
    atomic_json(args.record, record)
    if args.start_delay_seconds:
        time.sleep(args.start_delay_seconds)
    record["status"] = "exec"
    record["exec_unix_time"] = time.time()
    atomic_json(args.record, record)
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
