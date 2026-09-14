#!/usr/bin/env python3
"""Bind one immutable Stage3 manifest to a four-lane runtime allocation."""

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

from onpolicy.scripts.train.run_hkbz_two_stage_pipeline import (  # noqa: E402
    remove_option,
    set_option,
    set_switch,
)
from onpolicy.scripts.train.run_stage3_joint_manifest import (  # noqa: E402
    _sha256,
    _validate_pure_rl_command,
)
from onpolicy.utils.shared_eval import (  # noqa: E402
    format_cpu_set,
    parse_cpu_set,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-set", required=True)
    parser.add_argument("--shared-eval-socket", required=True)
    parser.add_argument("--shared-eval-cpu-set", required=True)
    parser.add_argument("--shared-gpu-phase-lock", required=True)
    parser.add_argument("--eval-workers", type=int, default=10)
    parser.add_argument("--start-delay-seconds", type=float, default=0.0)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if args.eval_workers <= 0:
        raise ValueError("--eval-workers must be positive.")
    if args.start_delay_seconds < 0.0:
        raise ValueError("--start-delay-seconds must be non-negative.")

    requested_cpus = parse_cpu_set(args.cpu_set)
    allowed_cpus = frozenset(os.sched_getaffinity(0))
    if requested_cpus != allowed_cpus:
        raise RuntimeError(
            "Trainer cgroup affinity mismatch: "
            f"requested={format_cpu_set(requested_cpus)}, "
            f"allowed={format_cpu_set(allowed_cpus)}"
        )
    evaluator_cpus = parse_cpu_set(args.shared_eval_cpu_set)
    if requested_cpus.intersection(evaluator_cpus):
        raise ValueError("Trainer and shared-evaluator CPU sets overlap.")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match --gpu.")
    gpu_phase_lock = Path(args.shared_gpu_phase_lock).expanduser()
    if not gpu_phase_lock.is_absolute():
        raise ValueError("--shared-gpu-phase-lock must be absolute.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        int(manifest.get("schema_version", 0)) != 2
        or manifest.get("training_stage") != "joint_finetune"
        or manifest.get("training_method") not in {
            "pure_joint_ppo", "role_clock_phase1", "role_clock_phase2",
            "shared_encoder_adaptation",
        }
    ):
        raise ValueError(f"Invalid Stage3 manifest: {manifest_path}")
    source = manifest.get("source_stage2", {})
    source_path = Path(str(source.get("path", ""))).resolve()
    if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
        raise ValueError(f"Stage3 source checkpoint changed: {source_path}")

    command = [str(value) for value in manifest.get("command", ())]
    if len(command) < 2:
        raise ValueError("Stage3 manifest has no argv command.")
    _validate_pure_rl_command(
        command,
        allow_shared_frozen=(
            manifest.get("training_method") in {
                "role_clock_phase1", "role_clock_phase2"
            }
            or (
                manifest.get("training_method") == "shared_encoder_adaptation"
                and manifest.get("architecture_contract", {}).get(
                    "shared_encoder_frozen"
                ) is True
            )
        ),
        allow_staged_shared=(
            manifest.get("training_method") == "shared_encoder_adaptation"
            and manifest.get("architecture_contract", {}).get(
                "shared_encoder_frozen"
            ) is False
        ),
    )
    runtime = manifest.get("four_lane_runtime_contract", {})
    if (
        int(runtime.get("trainer_count", 0)) != 4
        or int(runtime.get("n_eval_rollout_threads", 0)) != args.eval_workers
        or runtime.get("ppo_updates_serialized_per_gpu") is not True
        or runtime.get("gpu_phase_lock_scope") != "ppo_update"
        or runtime.get("clear_cuda_cache_after_update") is not True
        or command.count("--clear_cuda_cache_after_update") != 1
    ):
        raise ValueError("Stage3 four-lane runtime contract mismatch.")

    set_option(command, "--shared_eval_socket", args.shared_eval_socket)
    set_option(command, "--shared_eval_cpu_set", args.shared_eval_cpu_set)
    set_option(command, "--shared_eval_timeout_seconds", 21600)
    set_option(command, "--shared_gpu_phase_lock", str(gpu_phase_lock))
    set_option(command, "--n_eval_rollout_threads", args.eval_workers)
    set_option(command, "--torch_mp_sharing_strategy", "file_descriptor")
    set_option(command, "--ipc_timeout_seconds", 600)
    remove_option(command, "--no_eval", takes_value=False)
    set_switch(command, "--use_eval", True)
    set_switch(command, "--safe_graph_batch_pipeline", True)

    record = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "profile": manifest.get("profile"),
        "seed": int(command[command.index("--seed") + 1]),
        "experiment_name": command[command.index("--experiment_name") + 1],
        "gpu": args.gpu,
        "cpu_set": args.cpu_set,
        "shared_eval_socket": args.shared_eval_socket,
        "shared_eval_cpu_set": args.shared_eval_cpu_set,
        "shared_gpu_phase_lock": str(gpu_phase_lock),
        "eval_workers": args.eval_workers,
        "start_delay_seconds": args.start_delay_seconds,
        "launcher_pid": os.getpid(),
        "created_unix_time": time.time(),
        "command": command,
        "status": (
            "checked" if args.check_only
            else "waiting" if args.start_delay_seconds
            else "starting"
        ),
    }
    atomic_json(args.record.resolve(), record)
    if args.check_only:
        print(
            f"[Stage3Trial] runtime binding validated: {manifest_path}",
            flush=True,
        )
        return 0
    if args.start_delay_seconds:
        time.sleep(args.start_delay_seconds)
    record["status"] = "exec"
    record["exec_unix_time"] = time.time()
    atomic_json(args.record.resolve(), record)
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
