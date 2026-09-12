#!/usr/bin/env python3
"""Run one Stage-1 learning baseline across seeds on one physical GPU.

The immutable experiment manifest is produced by
``prepare_stage1_learning_baselines.py``.  This lane runner owns exactly one
baseline method and executes seeds 1, 2, and 3 sequentially.  Hardware-only
runtime changes (GPU cap, local evaluation parallelism, IPC safeguards, and
activation checkpointing) are recorded separately from the source commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.scripts.train.prepare_stage1_learning_baselines import (  # noqa: E402
    _atomic_json,
)


TRAIN = ROOT / "onpolicy/scripts/train/train_hkbz.py"
RESULT_ROOT = ROOT / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
METHODS = ("l2d", "multi_ppo", "fjsp_drl", "daniel")
SEEDS = (1, 2, 3)
RUNTIME_VALUE_OPTIONS = (
    "--shared_eval_socket",
    "--shared_eval_cpu_set",
    "--shared_eval_timeout_seconds",
    "--shared_gpu_phase_lock",
)
FORBIDDEN_TRAINING_STATE = (
    "--checkpoint_dir",
    "--selection_checkpoint_dir",
    "--plane_bc_teacher_dir",
    "--resume_stage1",
    "--resume_stage2",
    "--plane_bc_only",
    "--bc_reference_hard_gate",
    "--adaptive_bc_reference_kl",
)


class LaneInterrupted(RuntimeError):
    """Raised when systemd or the user stops the lane."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option(command: list[str], flag: str, default: str | None = None) -> str | None:
    count = command.count(flag)
    if count == 0:
        return default
    if count != 1:
        raise ValueError(f"Expected one {flag}, found {count}.")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"Missing value after {flag}.")
    return str(command[index + 1])


def remove_option(command: list[str], flag: str, *, takes_value: bool = True) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + (2 if takes_value else 1)]


def set_option(command: list[str], flag: str, value: object) -> None:
    remove_option(command, flag)
    command.extend((flag, str(value)))


def set_switch(command: list[str], flag: str, enabled: bool) -> None:
    remove_option(command, flag, takes_value=False)
    if enabled:
        command.append(flag)


def parse_cpu_set(specification: str) -> frozenset[int]:
    values: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        lower, separator, upper = item.partition("-")
        if separator:
            start, stop = int(lower), int(upper)
            if start > stop:
                raise ValueError(f"Reversed CPU range: {item}.")
            values.update(range(start, stop + 1))
        else:
            values.add(int(lower))
    if not values:
        raise ValueError("CPU set cannot be empty.")
    return frozenset(values)


def _require_file_option(command: list[str], flag: str) -> Path:
    value = option(command, flag)
    if not value:
        raise ValueError(f"Baseline command must pin {flag}.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _require_dir_option(command: list[str], flag: str) -> Path:
    value = option(command, flag)
    if not value:
        raise ValueError(f"Baseline command must pin {flag}.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def validate_training_command(command: list[str], method: str, seed: int) -> None:
    required = {
        "--stage1_baseline": method,
        "--seed": str(seed),
        "--training_stage": "plane_pretrain",
        "--resource_policy": "heuristic",
        "--plane_order_mode": "fixed",
        "--plane_pair_decoder": "joint_pair",
        "--plane_bc_pretrain_epochs": "0",
        "--bc_reference_kl_coef": "0.0",
        "--bc_reference_target_kl": "0.0",
        "--num_episodes": "8",
        "--n_rollout_threads": "60",
        "--ppo_epoch": "2",
        "--mini_batch_size": "7",
        "--data_chunk_length": "50",
        "--max_graphs_per_forward": "350",
    }
    for flag, expected in required.items():
        actual = option(command, flag)
        if actual != expected:
            raise ValueError(f"{flag}={actual!r}; expected {expected!r}.")
    present = [flag for flag in FORBIDDEN_TRAINING_STATE if flag in command]
    if present:
        raise ValueError(f"Baseline command contains forbidden state: {present}.")
    if "--use_eval" not in command:
        raise ValueError("Baseline command must enable validation.")
    _require_file_option(command, "--env_config")
    _require_file_option(command, "--ac_config")
    _require_dir_option(command, "--eval_dataset_dir")
    experiment = option(command, "--experiment_name")
    if not experiment:
        raise ValueError("Baseline command has no experiment name.")


def load_lane(
    manifest_path: Path,
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    if manifest.get("scope") != "stage1_learning_baselines":
        raise ValueError("Manifest is not a Stage-1 learning-baseline suite.")
    if tuple(manifest.get("methods", ())) != METHODS:
        raise ValueError(f"Unexpected methods: {manifest.get('methods')!r}.")
    if tuple(int(value) for value in manifest.get("seeds", ())) != SEEDS:
        raise ValueError(f"Unexpected seeds: {manifest.get('seeds')!r}.")
    if method not in METHODS:
        raise ValueError(f"Unknown Stage-1 baseline: {method}.")
    source = Path(str(manifest.get("source_command_json", ""))).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if sha256(source) != manifest.get("source_command_sha256"):
        raise ValueError("Source command SHA-256 changed after manifest creation.")

    entries: list[dict[str, Any]] = []
    for record in manifest.get("runs", ()):
        if record.get("stage1_baseline") != method:
            continue
        seed = int(record.get("seed", -1))
        command_path = Path(str(record.get("command_json", ""))).resolve()
        if not command_path.is_file():
            raise FileNotFoundError(command_path)
        payload = read_json(command_path)
        if payload.get("source_command_sha256") != manifest.get(
            "source_command_sha256"
        ):
            raise ValueError(f"Source lineage mismatch in {command_path}.")
        if payload.get("stage1_baseline") != method:
            raise ValueError(f"Method mismatch in {command_path}.")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Seed mismatch in {command_path}.")
        command = [str(value) for value in payload.get("command", ())]
        if len(command) < 2 or Path(command[1]).name != TRAIN.name:
            raise ValueError(f"Invalid train command in {command_path}.")
        validate_training_command(command, method, seed)
        experiment = str(option(command, "--experiment_name"))
        entries.append({
            "method": method,
            "seed": seed,
            "command_path": command_path,
            "command_sha256": sha256(command_path),
            "command": command,
            "experiment": experiment,
        })
    entries.sort(key=lambda item: int(item["seed"]))
    if tuple(int(item["seed"]) for item in entries) != SEEDS:
        raise ValueError(f"Lane {method} does not contain seeds {SEEDS}.")
    return manifest, entries


def runtime_command(
    source: Iterable[str],
    *,
    experiment: str,
    eval_workers: int,
    cuda_memory_fraction: float,
    canary: bool,
) -> list[str]:
    command = [str(value) for value in source]
    command[0] = str(Path(sys.executable).resolve())
    command[1] = str(TRAIN)
    for flag in RUNTIME_VALUE_OPTIONS:
        remove_option(command, flag)
    set_option(command, "--experiment_name", experiment)
    set_option(command, "--n_eval_rollout_threads", eval_workers)
    set_option(command, "--cuda_memory_fraction", cuda_memory_fraction)
    set_option(command, "--ipc_timeout_seconds", 900)
    set_option(command, "--safe_async_graph_clone_workers", 2)
    set_option(command, "--status_heartbeat_seconds", 30)
    set_switch(command, "--use_eval", True)
    set_switch(command, "--safe_graph_batch_pipeline", True)
    set_switch(command, "--safe_dagger_teacher_overlap", True)
    set_switch(command, "--shared_encoder_activation_checkpoint", True)
    set_switch(command, "--clear_cuda_cache_after_update", True)
    if canary:
        set_option(command, "--num_episodes", 1)
        set_option(command, "--train_sampling_size", 60)
        set_option(command, "--max_train_cases", 0)
        set_option(command, "--max_eval_cases", min(12, eval_workers))
        # A 12-case hardware canary need not contain the rare OOD-scale
        # stratum.  Select on the raw mean here so the canary reaches PPO;
        # formal runs retain the source command's composite selection metric
        # and full 60-case validation partition.
        set_option(command, "--selection_metric", "raw")
        set_option(command, "--actor_warmup_shards", 0)
        set_option(command, "--early_stop_patience", 0)
    return command


def latest_run_status(experiment: str) -> tuple[str, str | None, dict[str, Any]]:
    parent = RESULT_ROOT / experiment
    candidates: list[tuple[int, Path]] = []
    if parent.is_dir():
        for run_dir in parent.glob("run*"):
            suffix = run_dir.name.removeprefix("run")
            if suffix.isdigit():
                candidates.append((int(suffix), run_dir))
    if not candidates:
        return "missing", None, {}
    run_dir = max(candidates)[1]
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return "missing_run_status", str(run_dir), {}
    payload = read_json(status_path)
    return str(payload.get("status", "unknown")), str(run_dir), payload


def run_one(
    command: list[str],
    *,
    log_path: Path,
    state: dict[str, Any],
    status_path: Path,
    active_child: list[subprocess.Popen[Any]],
) -> tuple[int, str, str | None, dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active_child[:] = [process]
        state.update({
            "child_pid": process.pid,
            "heartbeat_unix_time": time.time(),
        })
        _atomic_json(status_path, state)
        while process.poll() is None:
            state["heartbeat_unix_time"] = time.time()
            _atomic_json(status_path, state)
            time.sleep(30)
        exit_code = int(process.returncode)
        active_child.clear()
    experiment = str(option(command, "--experiment_name"))
    run_status, run_dir, payload = latest_run_status(experiment)
    return exit_code, run_status, run_dir, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--cpu-affinity", required=True)
    parser.add_argument("--eval-workers", type=int, default=12)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.90)
    parser.add_argument("--start-delay", type=float, default=0.0)
    parser.add_argument("--canary-first", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.physical_gpu < 0:
        raise ValueError("Physical GPU must be non-negative.")
    if args.eval_workers <= 0 or args.eval_workers > 60:
        raise ValueError("Evaluation workers must be in [1, 60].")
    if not 0.0 < args.cuda_memory_fraction <= 1.0:
        raise ValueError("CUDA memory fraction must be in (0, 1].")
    if not 0.0 <= args.start_delay <= 300.0:
        raise ValueError("Start delay must be in [0, 300] seconds.")
    manifest_path = args.manifest.expanduser().resolve()
    suite_dir = args.suite_dir.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if manifest_path.parent != suite_dir / "commands":
        raise ValueError("Manifest must be stored in SUITE_DIR/commands.")
    requested_cpus = parse_cpu_set(args.cpu_affinity)
    observed_cpus = frozenset(os.sched_getaffinity(0))
    if not requested_cpus.issubset(observed_cpus):
        raise RuntimeError(
            f"Requested CPUs {sorted(requested_cpus)} are not available in "
            f"process affinity {sorted(observed_cpus)}."
        )
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible and visible != str(args.physical_gpu):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}; expected {args.physical_gpu}."
        )
    manifest, entries = load_lane(manifest_path, args.method)
    runtime_preview = runtime_command(
        entries[0]["command"],
        experiment=entries[0]["experiment"],
        eval_workers=args.eval_workers,
        cuda_memory_fraction=args.cuda_memory_fraction,
        canary=False,
    )
    validate_training_command(runtime_preview, args.method, 1)
    if option(runtime_preview, "--n_eval_rollout_threads") != str(
        args.eval_workers
    ):
        raise RuntimeError("Runtime evaluation worker override failed.")
    if any(flag in runtime_preview for flag in RUNTIME_VALUE_OPTIONS):
        raise RuntimeError("Runtime command retained stale shared-evaluator state.")
    if args.check_only:
        print(json.dumps({
            "status": "check_ok",
            "method": args.method,
            "physical_gpu": args.physical_gpu,
            "cpu_affinity": args.cpu_affinity,
            "seeds": list(SEEDS),
            "source_command_sha256": manifest["source_command_sha256"],
            "eval_workers": args.eval_workers,
            "cuda_memory_fraction": args.cuda_memory_fraction,
            "canary_first": bool(args.canary_first),
        }, indent=2))
        return 0

    formal_parents = [RESULT_ROOT / item["experiment"] for item in entries]
    canary_experiment = (
        f"{manifest['run_tag']}_canary_{args.method}_seed1"
    )
    parents = list(formal_parents)
    if args.canary_first:
        parents.insert(0, RESULT_ROOT / canary_experiment)
    existing = [str(path) for path in parents if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing existing experiment outputs: {existing}")
    status_path = suite_dir / "records" / f"{args.method}_lane.json"
    if status_path.exists():
        raise FileExistsError(f"Refusing existing lane status: {status_path}")
    runtime_dir = suite_dir / "runtime_commands"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "method": args.method,
        "physical_gpu": args.physical_gpu,
        "cpu_affinity": args.cpu_affinity,
        "controller_pid": os.getpid(),
        "created_unix_time": time.time(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_command_sha256": manifest["source_command_sha256"],
        "eval_workers": args.eval_workers,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "canary_first": bool(args.canary_first),
        "runs": {},
    }
    _atomic_json(status_path, state)
    active_child: list[subprocess.Popen[Any]] = []

    def handle_signal(signum, _frame):
        for process in active_child:
            if process.poll() is None:
                process.terminate()
        state.update({
            "status": "interrupted",
            "signal": int(signum),
            "finished_unix_time": time.time(),
        })
        _atomic_json(status_path, state)
        raise LaneInterrupted(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        if args.start_delay:
            state["status"] = "start_delay"
            _atomic_json(status_path, state)
            time.sleep(args.start_delay)
        if args.canary_first:
            command = runtime_command(
                entries[0]["command"],
                experiment=canary_experiment,
                eval_workers=args.eval_workers,
                cuda_memory_fraction=args.cuda_memory_fraction,
                canary=True,
            )
            runtime_path = runtime_dir / f"canary_{args.method}_seed1.json"
            _atomic_json(runtime_path, {
                "phase": "canary",
                "method": args.method,
                "seed": 1,
                "physical_gpu": args.physical_gpu,
                "cpu_affinity": args.cpu_affinity,
                "command": command,
            })
            state.update({
                "status": "canary_running",
                "current_run": "canary_seed1",
                "current_runtime_command": str(runtime_path),
                "run_started_unix_time": time.time(),
            })
            exit_code, run_status, run_dir, payload = run_one(
                command,
                log_path=suite_dir / "run_logs" / f"canary_{args.method}_seed1.log",
                state=state,
                status_path=status_path,
                active_child=active_child,
            )
            successful = exit_code == 0 and run_status == "completed"
            state["runs"]["canary_seed1"] = {
                "status": "completed" if successful else "failed",
                "exit_code": exit_code,
                "run_status": run_status,
                "run_dir": run_dir,
                "actor_step_completion_rate": payload.get(
                    "actor_step_completion_rate"
                ),
                "eval_makespan": payload.get("eval_makespan"),
                "finished_unix_time": time.time(),
            }
            _atomic_json(status_path, state)
            if not successful:
                raise RuntimeError(
                    f"{args.method} canary failed: exit={exit_code}, "
                    f"run_status={run_status}."
                )

        for entry in entries:
            seed = int(entry["seed"])
            run_key = f"seed{seed}"
            command = runtime_command(
                entry["command"],
                experiment=entry["experiment"],
                eval_workers=args.eval_workers,
                cuda_memory_fraction=args.cuda_memory_fraction,
                canary=False,
            )
            runtime_path = runtime_dir / f"formal_{args.method}_seed{seed}.json"
            _atomic_json(runtime_path, {
                "phase": "formal",
                "method": args.method,
                "seed": seed,
                "physical_gpu": args.physical_gpu,
                "cpu_affinity": args.cpu_affinity,
                "source_command_json": str(entry["command_path"]),
                "source_command_sha256": entry["command_sha256"],
                "command": command,
            })
            state.update({
                "status": "formal_running",
                "current_run": run_key,
                "current_runtime_command": str(runtime_path),
                "run_started_unix_time": time.time(),
            })
            _atomic_json(status_path, state)
            exit_code, run_status, run_dir, payload = run_one(
                command,
                log_path=suite_dir / "run_logs" / f"formal_{args.method}_seed{seed}.log",
                state=state,
                status_path=status_path,
                active_child=active_child,
            )
            successful = exit_code == 0 and run_status == "completed"
            state["runs"][run_key] = {
                "status": "completed" if successful else "failed",
                "exit_code": exit_code,
                "run_status": run_status,
                "run_dir": run_dir,
                "total_num_steps": payload.get("total_num_steps"),
                "eval_makespan": payload.get("eval_makespan"),
                "best_eval_makespan": payload.get("best_eval_makespan"),
                "finished_unix_time": time.time(),
            }
            _atomic_json(status_path, state)
            if not successful:
                raise RuntimeError(
                    f"{args.method} seed {seed} failed: exit={exit_code}, "
                    f"run_status={run_status}."
                )
        state.update({
            "status": "completed",
            "current_run": None,
            "completed_unix_time": time.time(),
        })
        _atomic_json(status_path, state)
        return 0
    except BaseException as error:
        if state.get("status") != "interrupted":
            state.update({
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "finished_unix_time": time.time(),
            })
            _atomic_json(status_path, state)
        raise
    finally:
        for process in active_child:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
