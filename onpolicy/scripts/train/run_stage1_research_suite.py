#!/usr/bin/env python
"""Run the Stage-1 causal ablation and formal-validation suite on one GPU."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable).resolve()
LEGACY_ROOT = "/home/fanyx/HKBZ-environment"
TRAIN = ROOT / "onpolicy/scripts/train/train_hkbz.py"
COMPARE = ROOT / "onpolicy/envs/HKBZ/experiment/valid_fjsp_v2_comparison.py"
IGA_PARALLEL = (
    ROOT / "onpolicy/envs/HKBZ/experiment/run_fjsp_v2_evolutionary_parallel.py"
)
IGA_TRAJECTORY_ANALYSIS = (
    ROOT / "onpolicy/envs/HKBZ/experiment/analyze_iga_teacher_trajectories.py"
)
AC_CONFIG = ROOT / "onpolicy/config/ac.yaml"
ENV_CONFIG = ROOT / "onpolicy/config/env_plane_pretrain.yaml"
TRAIN_DATA = (
    ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
)
TEST_DATA = (
    ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/test"
)
RESULT_ROOT = (
    ROOT / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
)
LOG_ROOT = ROOT / "result/hkbz_train_logs"
EVAL_ROOT = ROOT / "result/hkbz_eval_logs"
TARGET_DISTRIBUTION_WEIGHTS = "iid=0.50,ood_stress=0.45,ood_scale=0.05"
TAIL_RECOVERY_FOUNDATION_ID = "R0_balanced_dagger_global"

ACTIVE_PROCESS: subprocess.Popen | None = None
TEACHER_PROCESS: subprocess.Popen | None = None
TEACHER_LOG_HANDLE = None
STOP_REQUESTED = False


def relocate_saved_paths(value):
    """Map paths stored by the source host onto this checkout."""
    if isinstance(value, dict):
        return {key: relocate_saved_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_saved_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace(LEGACY_ROOT, str(ROOT))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_tag",
        default=f"stage1_research_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--study",
        choices=("representation", "ppo_recovery", "tail_recovery"),
        default="tail_recovery",
        help=(
            "representation reruns BC/DAgger ablations; ppo_recovery reuses "
            "one frozen PlaneBC checkpoint and changes one PPO factor at a "
            "time; tail_recovery trains one balanced DAgger/global-context "
            "PlaneBC foundation and runs tail-reward/team-credit ablations "
            "from that identical checkpoint"
        ),
    )
    parser.add_argument(
        "--plane_bc_checkpoint",
        default="",
        help="required initialization checkpoint for --study ppo_recovery",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--screen_epochs", type=int, default=3)
    parser.add_argument("--formal_epochs", type=int, default=8)
    parser.add_argument("--formal_seeds", default="1,2,3")
    parser.add_argument("--top_configs", type=int, default=1)
    parser.add_argument("--rollout_threads", type=int, default=80)
    parser.add_argument("--eval_threads", type=int, default=60)
    parser.add_argument("--mini_batch_size", type=int, default=7)
    parser.add_argument("--data_chunk_length", type=int, default=50)
    parser.add_argument("--max_graphs_per_forward", type=int, default=350)
    parser.add_argument("--teacher_workers", type=int, default=56)
    parser.add_argument("--teacher_time_budget", type=float, default=1800.0)
    parser.add_argument("--teacher_pop_size", type=int, default=20)
    parser.add_argument("--teacher_generations", type=int, default=20)
    parser.add_argument("--teacher_max_attempts", type=int, default=3)
    parser.add_argument(
        "--teacher_dir",
        default="",
        help="explicit verified teacher directory; default derives from IGA settings",
    )
    parser.add_argument("--plane_bc_epochs", type=int, default=4)
    parser.add_argument("--trajectory_workers", type=int, default=32)
    parser.add_argument("--potential_ridge", type=float, default=1e-3)
    parser.add_argument("--monitor_interval", type=float, default=30.0)
    parser.add_argument("--stall_timeout_seconds", type=float, default=900.0)
    parser.add_argument("--ipc_timeout_seconds", type=float, default=300.0)
    parser.add_argument(
        "--diagnostic_min_step_completion",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--diagnostic_min_relative_improvement",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--diagnostic_bootstrap_samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--continue_after_gate_pass",
        action="store_true",
        help=(
            "run all PPO-recovery ablations even after an earlier candidate "
            "passes every diagnostic gate"
        ),
    )
    parser.add_argument("--skip_formal", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing suite directory and skip completed units.",
    )
    args = parser.parse_args()
    if args.gpu not in (0, 1):
        raise ValueError("--gpu must select physical GPU 0 or 1 on this host.")
    if args.screen_epochs <= 0 or args.formal_epochs <= 0:
        raise ValueError("Screen and formal epochs must be positive.")
    if args.rollout_threads <= 0 or args.eval_threads <= 0:
        raise ValueError("Rollout and evaluation thread counts must be positive.")
    graphs = args.mini_batch_size * args.data_chunk_length
    if graphs != args.max_graphs_per_forward:
        raise ValueError(
            "The validated geometry requires mini_batch_size * data_chunk_length "
            f"== max_graphs_per_forward, got {graphs} != "
            f"{args.max_graphs_per_forward}."
        )
    if args.monitor_interval <= 0.0 or args.monitor_interval > 60.0:
        raise ValueError("--monitor_interval must be in (0, 60].")
    if args.teacher_workers <= 0 or args.teacher_time_budget <= 0.0:
        raise ValueError("IGA teacher worker count and time budget must be positive.")
    if args.teacher_max_attempts <= 0:
        raise ValueError("--teacher_max_attempts must be positive.")
    if args.trajectory_workers <= 0:
        raise ValueError("--trajectory_workers must be positive.")
    if args.potential_ridge < 0.0:
        raise ValueError("--potential_ridge must be non-negative.")
    if args.stall_timeout_seconds < 300.0:
        raise ValueError("--stall_timeout_seconds must be at least 300.")
    if args.ipc_timeout_seconds <= 0.0:
        raise ValueError("--ipc_timeout_seconds must be positive.")
    if not 0.0 < args.diagnostic_min_step_completion <= 1.0:
        raise ValueError(
            "--diagnostic_min_step_completion must be in (0, 1]."
        )
    if not 0.0 <= args.diagnostic_min_relative_improvement < 1.0:
        raise ValueError(
            "--diagnostic_min_relative_improvement must be in [0, 1)."
        )
    if args.diagnostic_bootstrap_samples <= 0:
        raise ValueError("--diagnostic_bootstrap_samples must be positive.")
    if args.study == "ppo_recovery":
        checkpoint = Path(args.plane_bc_checkpoint).expanduser().resolve()
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(
                "A non-empty --plane_bc_checkpoint is required for "
                f"--study ppo_recovery: {checkpoint}"
            )
        args.plane_bc_checkpoint = str(checkpoint)
    seeds = [int(item) for item in args.formal_seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--formal_seeds must contain at least one integer.")
    args.formal_seeds = seeds
    return args


def variants(plane_bc_epochs: int) -> tuple[list[dict], dict[str, str]]:
    """Causal teacher-forcing, DAgger, and global-context ablations."""
    if plane_bc_epochs <= 0:
        raise ValueError("The BC-to-PPO research suite requires --plane_bc_epochs > 0.")
    if plane_bc_epochs == 1:
        dagger_schedule = "1.0"
    else:
        dagger_schedule = ",".join(
            f"{1.0 - 0.9 * index / (plane_bc_epochs - 1):.2f}"
            for index in range(plane_bc_epochs)
        )
    current = {
        "id": "G0_local_teacher",
        "order": "fixed",
        "pair": "joint_pair",
        "reward": "potential_cmax",
        "global_features": "none",
        "dagger_schedule": "1.0",
        "adaptive_actor_kl": True,
        "adaptive_actor_lr_max_scale": 4.0,
        "adaptive_actor_min_step_completion": 0.90,
        "actor_accum": 8,
        "ppo_epoch": 1,
        "lr": 1e-5,
        "critic_lr": 1e-4,
        "shared_scale": 0.1,
        "max_grad_norm": 1.0,
        "entropy": 0.0005,
        "clip_param": 0.05,
        "target_kl": 0.005,
        "plane_bc_epochs": plane_bc_epochs,
        "plane_bc_shared_scale": 0.3,
        "plane_bc_freeze_shared_epochs": 1,
        "plane_bc_pair_loss_coef": 1.0,
        "plane_bc_order_loss_coef": 0.0,
        "bc_reference_kl_coef": 0.2,
        "bc_reference_target_kl": 0.0,
        "bc_reference_hard_gate": False,
        "potential_beta": 0.0,
        "bc_initial_weight": 5.0,
        "bc_relocation_weight": 2.0,
        "bc_critical_weight": 2.0,
        "bc_tail_start_fraction": 1.0,
        "bc_tail_weight": 1.0,
        "train_sampling_mode": "uniform",
        "train_sampling_weights": TARGET_DISTRIBUTION_WEIGHTS,
        "train_sampling_size": 0,
        "joint_team_ppo": False,
        "central_team_critic": False,
        "gnn_freeze_epochs": 0,
        "plane_order_freeze_epochs": 0,
    }
    local_dagger = {
        **current,
        "id": "G1_local_dagger",
        "dagger_schedule": dagger_schedule,
    }
    global_f1 = {
        **local_dagger,
        "id": "G2_global_f1_dagger",
        "global_features": "f1",
    }
    global_f1f2 = {
        **local_dagger,
        "id": "G3_global_f1f2_dagger",
        "global_features": "f1f2",
    }
    runs = [current, local_dagger, global_f1, global_f1f2]
    identifiers = [variant["id"] for variant in runs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"Duplicate experiment identifiers: {identifiers}")
    return runs, {}


def ppo_recovery_variants() -> tuple[list[dict], dict[str, str]]:
    """Cumulative one-factor PPO ablations initialized from one PlaneBC model."""
    base, _ = variants(1)
    diagnostic = {
        **base[0],
        "id": "P0_soft_gate_a8_e1_s01",
        "plane_bc_epochs": 0,
        "dagger_schedule": "reused_plane_bc",
        "actor_accum": 8,
        "ppo_epoch": 1,
        "shared_scale": 0.1,
        "bc_reference_kl_coef": 0.2,
        "bc_reference_target_kl": 0.0,
        "bc_reference_hard_gate": False,
        "adaptive_actor_lr_max_scale": 4.0,
        "adaptive_actor_min_step_completion": 0.90,
    }
    smaller_accumulation = {
        **diagnostic,
        "id": "P1_soft_gate_a4_e1_s01",
        "actor_accum": 4,
    }
    two_ppo_epochs = {
        **smaller_accumulation,
        "id": "P2_soft_gate_a4_e2_s01",
        "ppo_epoch": 2,
    }
    larger_shared_lr = {
        **two_ppo_epochs,
        "id": "P3_soft_gate_a4_e2_s03",
        "shared_scale": 0.3,
    }
    return [
        diagnostic,
        smaller_accumulation,
        two_ppo_epochs,
        larger_shared_lr,
    ], {}


def tail_recovery_variants(
    plane_bc_epochs: int,
) -> tuple[list[dict], dict[str, str]]:
    """Shared-BC factorial study for tail shaping and team-level PPO credit."""
    if plane_bc_epochs <= 1:
        raise ValueError(
            "tail_recovery requires at least two BC epochs for DAgger."
        )
    dagger_schedule = ",".join(
        f"{1.0 - 0.9 * index / (plane_bc_epochs - 1):.2f}"
        for index in range(plane_bc_epochs)
    )
    base = {
        **ppo_recovery_variants()[0][0],
        "id": TAIL_RECOVERY_FOUNDATION_ID,
        "plane_bc_epochs": int(plane_bc_epochs),
        "dagger_schedule": dagger_schedule,
        "global_features": "f1f2",
        "actor_accum": 4,
        "ppo_epoch": 2,
        "shared_scale": 0.1,
        "reward": "potential_cmax",
        "potential_beta": 0.0,
        "bc_tail_start_fraction": 0.75,
        "bc_tail_weight": 3.0,
        "train_sampling_mode": "distribution_balanced",
        "train_sampling_weights": TARGET_DISTRIBUTION_WEIGHTS,
        "train_sampling_size": 0,
        "joint_team_ppo": False,
        "central_team_critic": False,
    }
    tail_potential = {
        **base,
        "id": "R1_tail_potential",
        "plane_bc_epochs": 0,
        "dagger_schedule": "reused_shared_plane_bc",
        "reward": "iga_potential",
        "potential_beta": 0.10,
    }
    team_ratio = {
        **base,
        "id": "R2_team_ratio",
        "plane_bc_epochs": 0,
        "dagger_schedule": "reused_shared_plane_bc",
        "joint_team_ppo": True,
    }
    tail_team_ratio = {
        **tail_potential,
        "id": "R3_tail_team_ratio",
        "joint_team_ppo": True,
    }
    tail_team_critic = {
        **tail_team_ratio,
        "id": "R4_tail_team_critic",
        "central_team_critic": True,
    }
    runs = [
        base,
        tail_potential,
        team_ratio,
        tail_team_ratio,
        tail_team_critic,
    ]
    return runs, {}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_float(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def gpu_is_idle(gpu: int) -> tuple[bool, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    output = result.stdout.strip()
    return not bool(output), output


def terminate_group(process: subprocess.Popen | None, sig=signal.SIGTERM) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def handle_signal(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[Suite] Received signal {signum}; terminating child process groups.", flush=True)
    terminate_group(ACTIVE_PROCESS)
    terminate_group(TEACHER_PROCESS)


def write_command(path: Path, command: list[str], env: dict[str, str]) -> None:
    payload = {
        "command": command,
        "shell_command": shlex.join(command),
        "environment": {
            key: env[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MPLCONFIGDIR",
            )
            if key in env
        },
    }
    atomic_json(path, payload)


def archive_existing_artifact(path: Path) -> Path | None:
    """Preserve a failed attempt before a resumed unit reuses its log name."""
    if not path.exists():
        return None
    attempt = 1
    while True:
        archived = path.with_name(f"{path.name}.attempt{attempt}")
        if not archived.exists():
            os.replace(path, archived)
            return archived
        attempt += 1


def sample_gpu(metrics_path: Path, label: str, gpu: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=timestamp,index,memory.used,memory.total,"
            "utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    sample = result.stdout.strip() if result.returncode == 0 else (
        f"nvidia-smi-error: {result.stderr.strip()}"
    )
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.time():.3f}\t{label}\t{sample}\n")
    return sample


def terminate_then_kill(process: subprocess.Popen | None, grace_seconds=10.0) -> None:
    if process is None or process.poll() is not None:
        return
    terminate_group(process)
    try:
        process.wait(timeout=float(grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass
    terminate_group(process, sig=signal.SIGKILL)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def command_option(command: list[str], option: str) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def progress_signature(command: list[str], log_path: Path) -> tuple:
    log_signature = (
        log_path.stat().st_size, log_path.stat().st_mtime_ns
    ) if log_path.exists() else (0, 0)
    experiment = command_option(command, "--experiment_name")
    if not experiment:
        return (log_signature, None)
    status_paths = list((RESULT_ROOT / experiment).glob("run*/run_status.json"))
    if not status_paths:
        return (log_signature, None)
    status_path = max(status_paths, key=lambda path: path.stat().st_mtime_ns)
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (log_signature, None)
    semantic_keys = (
        "status", "event", "epoch", "shard", "completed_shard",
        "total_num_steps", "actor_optimizer_steps", "eval_makespan",
        "best_eval_makespan", "plane_bc_epoch", "plane_bc_rollout",
        "plane_bc_env_step", "plane_bc_updates",
    )
    return (
        log_signature,
        tuple((key, payload.get(key)) for key in semantic_keys),
    )


def run_monitored(
    command: list[str],
    label: str,
    log_path: Path,
    command_path: Path,
    env: dict[str, str],
    suite_status_path: Path,
    suite_state: dict,
    metrics_path: Path,
    gpu: int,
    monitor_interval: float,
    stall_timeout_seconds: float,
) -> None:
    global ACTIVE_PROCESS
    gpu_scripts = {str(TRAIN), str(COMPARE)}
    if any(str(item) in gpu_scripts for item in command):
        idle, users = gpu_is_idle(gpu)
        if not idle:
            raise RuntimeError(
                f"GPU {gpu} became occupied before {label}: {users}"
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archived_log = archive_existing_artifact(log_path)
    archived_command = archive_existing_artifact(command_path)
    if archived_log or archived_command:
        print(
            f"[Suite] Archived previous attempt for {label}: "
            f"log={archived_log}, command={archived_command}.",
            flush=True,
        )
    write_command(command_path, command, env)
    print(f"[Suite] Starting {label}", flush=True)
    print(f"[Suite] Command: {shlex.join(command)}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        suite_state.update(
            status="running",
            current=label,
            current_pid=ACTIVE_PROCESS.pid,
            current_started_unix_time=started,
            updated_unix_time=time.time(),
        )
        atomic_json(suite_status_path, suite_state)
        last_progress_signature = progress_signature(command, log_path)
        last_progress_unix_time = time.time()
        while ACTIVE_PROCESS.poll() is None:
            if STOP_REQUESTED:
                terminate_then_kill(ACTIVE_PROCESS)
                raise KeyboardInterrupt("Suite stop requested.")
            if TEACHER_PROCESS is not None:
                teacher_status = TEACHER_PROCESS.poll()
                if teacher_status not in (None, 0):
                    terminate_then_kill(ACTIVE_PROCESS)
                    raise RuntimeError(
                        f"IGA teacher generation exited with status {teacher_status}."
                    )
            sample = sample_gpu(metrics_path, label, gpu)
            current_progress_signature = progress_signature(command, log_path)
            if current_progress_signature != last_progress_signature:
                last_progress_signature = current_progress_signature
                last_progress_unix_time = time.time()
            no_progress_seconds = time.time() - last_progress_unix_time
            if no_progress_seconds >= float(stall_timeout_seconds):
                suite_state.update(
                    status="stall_timeout",
                    current=label,
                    current_pid=ACTIVE_PROCESS.pid,
                    no_progress_seconds=no_progress_seconds,
                    stall_timeout_seconds=float(stall_timeout_seconds),
                    updated_unix_time=time.time(),
                )
                atomic_json(suite_status_path, suite_state)
                terminate_then_kill(ACTIVE_PROCESS)
                raise TimeoutError(
                    f"{label} made no semantic/log progress for "
                    f"{no_progress_seconds:.1f}s; hard stall timeout is "
                    f"{stall_timeout_seconds:.1f}s."
                )
            suite_state.update(
                updated_unix_time=time.time(),
                current_gpu_sample=sample,
                no_progress_seconds=no_progress_seconds,
                teacher_pid=(
                    TEACHER_PROCESS.pid
                    if TEACHER_PROCESS is not None
                    and TEACHER_PROCESS.poll() is None
                    else None
                ),
            )
            atomic_json(suite_status_path, suite_state)
            print(
                f"[Suite] heartbeat label={label} elapsed="
                f"{time.time() - started:.1f}s gpu=({sample})",
                flush=True,
            )
            time.sleep(monitor_interval)
        return_code = ACTIVE_PROCESS.returncode
    ACTIVE_PROCESS = None
    if return_code != 0:
        raise RuntimeError(
            f"{label} exited with status {return_code}; inspect {log_path}."
        )
    elapsed = time.time() - started
    print(f"[Suite] Completed {label} in {elapsed:.1f}s", flush=True)
    suite_state.update(
        current=None,
        current_pid=None,
        updated_unix_time=time.time(),
    )
    atomic_json(suite_status_path, suite_state)


def training_command(
    args: argparse.Namespace,
    *,
    variant: dict,
    experiment_name: str,
    seed: int,
    epochs: int,
    formal: bool,
    teacher_dir: Path,
    potential_weights_path: Path,
    resume_checkpoint: Path | None = None,
    preserve_resume_state: bool = False,
) -> list[str]:
    potential = variant["reward"] in {"potential_cmax", "iga_potential"}
    command = [
        str(PYTHON),
        str(TRAIN),
        "--env_name", "HKBZ",
        "--scenario_name", "simple",
        "--algorithm_name", "gnn_mappo",
        "--experiment_name", experiment_name,
        "--ac_config", str(AC_CONFIG),
        "--env_config", str(ENV_CONFIG),
        "--seed", str(seed),
        "--n_training_threads", "1",
        "--n_rollout_threads", str(args.rollout_threads),
        "--n_eval_rollout_threads", str(args.eval_threads),
        "--max_train_cases", "0",
        "--max_eval_cases", "0",
        "--train_sampling_mode", variant["train_sampling_mode"],
        "--train_sampling_weights", variant["train_sampling_weights"],
        "--train_sampling_size", str(variant["train_sampling_size"]),
        "--training_stage", "plane_pretrain",
        "--num_episodes", str(epochs),
        "--rollout_until_done",
        "--rollout_max_steps", "4000",
        "--ppo_epoch", str(variant["ppo_epoch"]),
        "--mini_batch_size", str(args.mini_batch_size),
        "--data_chunk_length", str(args.data_chunk_length),
        "--grad_accumulation_steps", "11",
        "--actor_grad_accumulation_steps", str(variant["actor_accum"]),
        "--max_graphs_per_forward", str(args.max_graphs_per_forward),
        "--actor_warmup_shards", "1",
        "--plane_order_mode", variant["order"],
        "--plane_pair_decoder", variant["pair"],
        "--global_feature_mode", variant["global_features"],
        "--plane_bc_tail_start_fraction", str(
            variant["bc_tail_start_fraction"]
        ),
        "--plane_bc_tail_weight", str(variant["bc_tail_weight"]),
        "--evaluation_tau", "0.3",
        "--selection_metric", "composite",
        "--selection_iid_weight", "0.50",
        "--selection_ood_stress_weight", "0.45",
        "--selection_ood_scale_weight", "0.05",
        "--device_bc_pretrain_epochs", "0",
        "--gnn_freeze_epochs", str(variant["gnn_freeze_epochs"]),
        "--plane_freeze_epochs", "0",
        "--plane_order_freeze_epochs", str(
            variant["plane_order_freeze_epochs"]
        ),
        "--lr", str(variant["lr"]),
        "--critic_lr", str(variant["critic_lr"]),
        "--shared_actor_lr_scale", str(variant["shared_scale"]),
        "--plane_actor_lr_scale", "1.0",
        "--device_actor_lr_scale", "1.0",
        "--transporter_actor_lr_scale", "1.0",
        "--anneal_original", "0.3",
        "--anneal_final", "0.3",
        "--entropy_coef", str(variant["entropy"]),
        "--clip_param", str(variant["clip_param"]),
        "--target_kl", str(variant["target_kl"]),
        "--adaptive_actor_kl_low", "0.0001",
        "--adaptive_actor_kl_high", "0.0005",
        "--adaptive_actor_lr_min_scale", "0.25",
        "--adaptive_actor_lr_max_scale", str(
            variant["adaptive_actor_lr_max_scale"]
        ),
        "--adaptive_actor_lr_up", "1.5",
        "--adaptive_actor_lr_down", "0.5",
        "--adaptive_actor_min_step_completion", str(
            variant["adaptive_actor_min_step_completion"]
        ),
        "--bc_reference_kl_coef", str(
            variant["bc_reference_kl_coef"]
        ),
        "--bc_reference_target_kl", str(
            variant["bc_reference_target_kl"]
        ),
        "--max_grad_norm", str(variant["max_grad_norm"]),
        "--plane_loss_coef", "1.0",
        "--device_loss_coef", "0.5",
        "--transporter_loss_coef", "0.5",
        "--hindsight_reward_mode", variant["reward"],
        "--hindsight_cmax_coef", "1.0" if potential else "0.0",
        "--iga_potential_beta", str(variant["potential_beta"]),
        "--iga_potential_gamma", "0.99",
        "--hindsight_shaping_coef", "0.0",
        "--hindsight_terminal_cmax_coef", "0.0" if potential else "1.0",
        "--plane_cycle_repeat_limit", "8",
        "--plane_no_progress_limit", "120",
        "--plane_relocation_limit", "40",
        "--plane_first_completion_bonus", "120",
        "--plane_repeat_relocation_penalty", "300",
        "--plane_reset_job_penalty", "120",
        "--plane_no_progress_penalty", "60",
        "--plane_cycle_penalty", "20000",
        "--reward_coef", "0.01",
        "--use_valuenorm",
        "--use_eval",
        "--eval_interval", "1",
        "--eval_canary_rounds", "1",
        "--save_interval", "1",
        "--status_heartbeat_seconds", "60",
        "--recovery_checkpoint_interval_shards", "1",
        "--torch_mp_sharing_strategy", "file_descriptor",
        "--ipc_timeout_seconds", str(args.ipc_timeout_seconds),
        "--early_stop_patience", "4" if formal else "0",
    ]
    if variant["reward"] == "iga_potential":
        command.extend([
            "--iga_potential_weights_path", str(potential_weights_path),
        ])
    if variant["joint_team_ppo"]:
        command.append("--joint_team_ppo")
    if variant["central_team_critic"]:
        command.append("--central_team_critic")
    if variant["adaptive_actor_kl"]:
        command.append("--adaptive_actor_kl")
    if variant["bc_reference_hard_gate"]:
        command.append("--bc_reference_hard_gate")
    plane_bc_epochs = (
        0 if resume_checkpoint is not None else int(variant["plane_bc_epochs"])
    )
    if plane_bc_epochs > 0:
        command.extend(
            [
                "--plane_bc_pretrain_epochs", str(plane_bc_epochs),
                "--plane_bc_teacher_dir", str(teacher_dir),
                "--plane_bc_dagger_schedule",
                str(variant["dagger_schedule"]),
                "--plane_bc_lr", "0.0001",
                "--plane_bc_shared_lr_scale", str(
                    variant["plane_bc_shared_scale"]
                ),
                "--plane_bc_freeze_shared_epochs", str(
                    variant["plane_bc_freeze_shared_epochs"]
                ),
                "--plane_bc_pair_loss_coef", str(
                    variant["plane_bc_pair_loss_coef"]
                ),
                "--plane_bc_order_loss_coef", str(
                    variant["plane_bc_order_loss_coef"]
                ),
                "--plane_bc_rollouts_per_epoch", "0",
                "--plane_bc_initial_weight", str(
                    variant["bc_initial_weight"]
                ),
                "--plane_bc_relocation_weight", str(
                    variant["bc_relocation_weight"]
                ),
                "--plane_bc_critical_op_weight", str(
                    variant["bc_critical_weight"]
                ),
            ]
        )
    else:
        command.extend(["--plane_bc_pretrain_epochs", "0"])
    if resume_checkpoint is not None:
        command.extend([
            "--checkpoint_dir", str(resume_checkpoint),
            "--resume_stage1",
        ])
        if not preserve_resume_state:
            command.append("--reset_optimizers_on_resume")
    if formal:
        command.extend(
            [
                "--canary_eval_interval_shards", "3",
                "--canary_max_regression", "0.02",
                "--canary_stop_on_regression",
            ]
        )
    else:
        command.extend(["--canary_eval_interval_shards", "0"])
    return command


def comparison_command(
    variant: dict,
    checkpoint: Path,
    output_json: Path,
    seed: int,
) -> list[str]:
    return [
        str(PYTHON),
        str(COMPARE),
        "--env_name", "HKBZ",
        "--algorithm_name", "gnn_mappo",
        "--ac_config", str(AC_CONFIG),
        "--env_config", str(ENV_CONFIG),
        "--checkpoint_dir", str(checkpoint),
        "--dataset_test_dir", str(TEST_DATA),
        "--output_json", str(output_json),
        "--methods", "drl_g",
        "--model_batch_size", "60",
        "--max_steps", "4000",
        "--seed", str(seed),
        "--policy_tau", "0.3",
        "--plane_order_mode", variant["order"],
        "--plane_pair_decoder", variant["pair"],
        "--global_feature_mode", variant["global_features"],
    ]


def load_checkpoint_record(checkpoint_path: Path) -> dict:
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    makespan = safe_float(checkpoint.get("eval_makespan", math.inf))
    if makespan is None:
        raise RuntimeError(
            f"Checkpoint has no finite eval_makespan: {checkpoint_path}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "eval_makespan": makespan,
        "selection_score": safe_float(
            checkpoint.get("selection_score", makespan)
        ),
        "eval_raw_makespan": safe_float(
            checkpoint.get("eval_raw_makespan", makespan)
        ),
        "eval_iid_makespan": safe_float(
            checkpoint.get("eval_iid_makespan", makespan)
        ),
        "eval_composite_makespan": safe_float(
            checkpoint.get("eval_composite_makespan", makespan)
        ),
        "selection_metric": checkpoint.get("selection_metric", "iid"),
        "evaluation_tau": safe_float(
            checkpoint.get("evaluation_tau", checkpoint.get("tau", math.nan))
        ),
        "checkpoint_stage": checkpoint.get("stage", "unknown"),
        "episodes": int(checkpoint.get("episodes", 0)),
        "tau": safe_float(checkpoint.get("tau", math.nan)),
        "plane_order_mode": checkpoint.get("plane_order_mode"),
        "plane_pair_decoder": checkpoint.get("plane_pair_decoder"),
        "experiment_config": checkpoint.get("experiment_config", {}),
        "actor_update_health": checkpoint.get("actor_update_health", {}),
    }


def validate_plane_bc_checkpoint(
    checkpoint_path: Path,
    *,
    expected_global_feature_mode: str = "none",
) -> dict:
    """Reject an architecture-mismatched or non-BC initialization artifact."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    errors = []
    if not isinstance(checkpoint.get("model"), dict):
        errors.append("missing model state")
    if checkpoint.get("stage") != "plane_iga_bc_pretrain":
        errors.append(f"stage={checkpoint.get('stage')!r}")
    if checkpoint.get("plane_order_mode") != "fixed":
        errors.append(
            f"plane_order_mode={checkpoint.get('plane_order_mode')!r}"
        )
    if checkpoint.get("plane_pair_decoder") != "joint_pair":
        errors.append(
            f"plane_pair_decoder={checkpoint.get('plane_pair_decoder')!r}"
        )
    if (
        checkpoint.get("global_feature_mode", "none")
        != expected_global_feature_mode
    ):
        errors.append(
            f"global_feature_mode={checkpoint.get('global_feature_mode')!r}"
        )
    if errors:
        raise RuntimeError(
            f"Incompatible PlaneBC checkpoint {checkpoint_path}: "
            + "; ".join(errors)
        )
    return {
        "checkpoint": str(checkpoint_path),
        "stage": checkpoint["stage"],
        "plane_order_mode": checkpoint["plane_order_mode"],
        "plane_pair_decoder": checkpoint["plane_pair_decoder"],
        "global_feature_mode": checkpoint.get("global_feature_mode", "none"),
        "plane_bc_dagger_schedule": checkpoint.get(
            "plane_bc_dagger_schedule", []
        ),
    }


def _evaluation_payload(run_dir: Path, label: str) -> dict:
    path = run_dir / "evaluations" / f"{label}.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing evaluation artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_label") != label:
        raise RuntimeError(
            f"Evaluation label mismatch in {path}: "
            f"{payload.get('evaluation_label')} != {label}"
        )
    return payload


def paired_bootstrap_delta(
    baseline_payload: dict,
    candidate_payload: dict,
    *,
    bootstrap_samples: int,
    seed: int = 20260729,
) -> dict:
    """Paired candidate-minus-baseline Cmax statistics over identical cases."""
    def finite_cases(payload):
        cases = {}
        for record in payload.get("cases", []):
            makespan = safe_float(record.get("makespan"))
            if makespan is None:
                continue
            cases[str(record["case_key"])] = makespan
        return cases

    baseline_cases = finite_cases(baseline_payload)
    candidate_cases = finite_cases(candidate_payload)
    if set(baseline_cases) != set(candidate_cases) or not baseline_cases:
        raise RuntimeError(
            "Paired validation requires the same non-empty finite case set: "
            f"baseline={len(baseline_cases)}, candidate={len(candidate_cases)}."
        )
    case_keys = sorted(baseline_cases)
    deltas = np.asarray(
        [
            candidate_cases[case_key] - baseline_cases[case_key]
            for case_key in case_keys
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sample_count = int(bootstrap_samples)
    bootstrap_means = np.empty(sample_count, dtype=np.float64)
    # Keep peak memory bounded even if a future run requests many resamples.
    chunk_size = max(1, min(2048, sample_count))
    for start in range(0, sample_count, chunk_size):
        stop = min(sample_count, start + chunk_size)
        indices = rng.integers(
            0, len(deltas), size=(stop - start, len(deltas))
        )
        bootstrap_means[start:stop] = deltas[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "case_count": int(len(deltas)),
        "mean_delta_candidate_minus_pre_ppo": float(deltas.mean()),
        "median_delta_candidate_minus_pre_ppo": float(np.median(deltas)),
        "ci95_delta_lower": float(lower),
        "ci95_delta_upper": float(upper),
        "wins": int(np.sum(deltas < 0.0)),
        "ties": int(np.sum(deltas == 0.0)),
        "losses": int(np.sum(deltas > 0.0)),
    }


def latest_epoch_diagnostic(
    checkpoint: Path,
    *,
    variant: dict,
    min_step_completion: float,
    min_relative_improvement: float,
    bootstrap_samples: int,
) -> dict:
    """Evaluate the final PPO epoch against pre-PPO and optimizer-health gates."""
    run_dir = checkpoint.parent.parent
    epoch_files = []
    for path in (run_dir / "evaluations").glob("epoch_*.json"):
        try:
            epoch = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        epoch_files.append((epoch, path))
    if not epoch_files:
        raise RuntimeError(f"No epoch evaluation found under {run_dir}.")
    latest_epoch, _ = max(epoch_files)
    latest_checkpoint = run_dir / "models" / (
        f"checkpoint_Epoch{latest_epoch}.pt"
    )
    if (
        not latest_checkpoint.is_file()
        or latest_checkpoint.stat().st_size == 0
    ):
        raise RuntimeError(f"Missing checkpoint: {latest_checkpoint}")
    latest_checkpoint_payload = torch.load(
        latest_checkpoint, map_location="cpu"
    )
    baseline = _evaluation_payload(run_dir, "pre_ppo")
    candidate = _evaluation_payload(run_dir, f"epoch_{latest_epoch}")
    paired = paired_bootstrap_delta(
        baseline,
        candidate,
        bootstrap_samples=bootstrap_samples,
    )
    baseline_summary = baseline.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    # Evaluation JSON is the canonical source for per-epoch validation
    # metrics.  Legacy checkpoint_Epoch*.pt artifacts intentionally contain
    # optimizer/model state but may not contain eval_makespan.
    health = dict(
        latest_checkpoint_payload.get("actor_update_health", {})
    )
    pre_cmax = float(baseline_summary["eval_makespan"])
    candidate_cmax = float(candidate_summary["eval_makespan"])
    pre_ood_stress = float(
        baseline_summary["eval_distribution_ood_stress_makespan"]
    )
    candidate_ood_stress = float(
        candidate_summary["eval_distribution_ood_stress_makespan"]
    )
    checks = {
        "actor_step_completion": (
            float(health.get("step_completion_rate", 0.0))
            >= float(min_step_completion)
        ),
        "zero_update_shards": (
            int(health.get("zero_update_shards", -1)) == 0
        ),
        "old_policy_kl": (
            float(health.get("post_update_old_policy_kl_max", math.inf))
            <= float(variant["target_kl"])
        ),
        "validation_completion": (
            float(candidate_summary.get("eval_completion_rate", 0.0)) == 1.0
        ),
        "validation_cycles": (
            int(candidate_summary.get("eval_cycle_count", -1)) == 0
        ),
        "validation_cmax": (
            candidate_cmax
            <= pre_cmax * (1.0 - float(min_relative_improvement))
        ),
        "ood_stress_non_regression": (
            candidate_ood_stress <= pre_ood_stress
        ),
    }
    return {
        "latest_epoch": int(latest_epoch),
        "latest_checkpoint": str(latest_checkpoint),
        "pre_ppo_cmax": pre_cmax,
        "candidate_cmax": candidate_cmax,
        "relative_improvement": (
            (pre_cmax - candidate_cmax) / max(pre_cmax, 1e-8)
        ),
        "pre_ppo_ood_stress_cmax": pre_ood_stress,
        "candidate_ood_stress_cmax": candidate_ood_stress,
        "actor_update_health": health,
        "paired_bootstrap": paired,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def latest_checkpoint(experiment: str, filename: str) -> Path:
    candidates = list(
        (RESULT_ROOT / experiment).glob(f"run*/models/{filename}")
    )
    if not candidates:
        raise RuntimeError(
            f"No {filename} found for experiment {experiment}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def completed_training_checkpoint(experiment: str) -> Path | None:
    """Reuse training only after its atomic run_status says completed."""
    experiment_dir = RESULT_ROOT / experiment
    run_dirs = sorted(
        experiment_dir.glob("run*"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for run_dir in run_dirs:
        checkpoint = run_dir / "models/checkpoint_Best.pt"
        status_path = run_dir / "run_status.json"
        if not checkpoint.is_file() or not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") != "completed":
            continue
        load_checkpoint_record(checkpoint)
        return checkpoint
    return None


def interrupted_training_checkpoint(experiment: str) -> Path | None:
    """Return the newest valid post-shard Recovery checkpoint for an unfinished run."""
    experiment_dir = RESULT_ROOT / experiment
    run_dirs = sorted(
        experiment_dir.glob("run*"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for run_dir in run_dirs:
        status_path = run_dir / "run_status.json"
        checkpoint_path = run_dir / "models/checkpoint_Recovery.pt"
        if not status_path.is_file() or not checkpoint_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except (
            OSError, ValueError, TypeError, EOFError, RuntimeError,
            json.JSONDecodeError,
        ):
            continue
        if status.get("status") == "completed":
            continue
        if checkpoint.get("stage") != "post_shard_recovery":
            continue
        completed_shard = int(checkpoint.get("completed_shard", 0))
        total_shards = int(checkpoint.get("total_shards", 0))
        if completed_shard <= 0 or total_shards <= 0:
            continue
        return checkpoint_path
    return None


def pre_ppo_record(experiment: str) -> dict:
    try:
        checkpoint = latest_checkpoint(
            experiment, "checkpoint_PrePPO_tau03.pt"
        )
    except RuntimeError:
        checkpoint = latest_checkpoint(experiment, "checkpoint_PrePPO.pt")
    record = load_checkpoint_record(checkpoint)
    return {
        "pre_ppo_checkpoint": record["checkpoint"],
        "pre_ppo_eval_makespan": record["eval_makespan"],
        "pre_ppo_selection_score": record["selection_score"],
        "pre_ppo_eval_iid_makespan": record["eval_iid_makespan"],
        "pre_ppo_eval_composite_makespan": (
            record["eval_composite_makespan"]
        ),
        "pre_ppo_stage": record["checkpoint_stage"],
    }


def validate_resume_summary(
    summary: dict,
    args: argparse.Namespace,
    run_variants: list[dict],
    aliases: dict[str, str],
) -> None:
    if summary.get("run_tag") != args.run_tag:
        raise RuntimeError(
            f"Resume run_tag mismatch: {summary.get('run_tag')} != {args.run_tag}."
        )
    if summary.get("variants") != run_variants or summary.get("aliases") != aliases:
        raise RuntimeError("Resume variant definitions do not match the saved suite.")
    saved = summary.get("configuration", {})
    current = vars(args)
    ignored = {"resume", "monitor_interval", "stall_timeout_seconds"}
    if args.study == "tail_recovery":
        ignored.add("plane_bc_checkpoint")
    mismatches = []
    for key, old_value in saved.items():
        if key in ignored or key not in current:
            continue
        if current[key] != old_value:
            mismatches.append(
                f"{key}: saved={old_value!r}, current={current[key]!r}"
            )
    if mismatches:
        raise RuntimeError(
            "Resume configuration mismatch: " + "; ".join(mismatches)
        )


def validated_completed_screen_records(summary: dict) -> dict[str, dict]:
    completed = {}
    for record in summary.get("screen", []):
        if record.get("alias_of"):
            continue
        checkpoint = Path(str(record.get("checkpoint", "")))
        load_checkpoint_record(checkpoint)
        completed[str(record["id"])] = record
    return completed


def validated_completed_formal_records(
    summary: dict,
) -> dict[tuple[str, int], dict]:
    completed = {}
    for record in summary.get("formal", []):
        checkpoint = Path(str(record.get("checkpoint", "")))
        load_checkpoint_record(checkpoint)
        test_json = Path(str(record.get("test_json", "")))
        if not test_json.is_file() or test_json.stat().st_size == 0:
            raise RuntimeError(f"Missing completed formal evaluation: {test_json}")
        payload = json.loads(test_json.read_text(encoding="utf-8"))
        if payload.get("methods", {}).get("DRL-G", {}).get("status") != "completed":
            raise RuntimeError(f"Incomplete formal evaluation: {test_json}")
        completed[(str(record["id"]), int(record["seed"]))] = record
    return completed

def teacher_validation_errors(teacher_dir: Path, payload: dict) -> list[str]:
    expected_names = {path.name for path in TRAIN_DATA.glob("case_*")}
    teacher_paths = {path.stem: path for path in teacher_dir.glob("case_*.json")}
    method = payload.get("methods", {}).get("IGA", {})
    records = method.get("cases", [])
    summary = method.get("summary", {})
    errors = []
    if payload.get("status") != "completed":
        errors.append(f"generation status={payload.get('status')}")
    if len(records) != len(expected_names):
        errors.append(f"records={len(records)}/{len(expected_names)}")
    if set(teacher_paths) != expected_names:
        missing = sorted(expected_names - set(teacher_paths))
        extra = sorted(set(teacher_paths) - expected_names)
        errors.append(f"teacher files missing={missing[:10]} extra={extra[:10]}")
    if int(summary.get("completed_count", 0)) != len(expected_names):
        errors.append(f"completed_count={summary.get('completed_count')}")
    if int(summary.get("verified_count", 0)) != len(expected_names):
        errors.append(f"verified_count={summary.get('verified_count')}")
    if int(summary.get("error_count", 0)) != 0:
        errors.append(f"error_count={summary.get('error_count')}")

    seen = set()
    for record in records:
        case = str(record.get("case", ""))
        if case in seen:
            errors.append(f"duplicate record={case}")
        seen.add(case)
        if case not in expected_names:
            errors.append(f"unexpected record={case}")
            continue
        if not record.get("completed") or not record.get("completion_verified"):
            errors.append(f"unverified record={case}")
        if not str(record.get("case_id", "")).startswith("train_"):
            errors.append(f"wrong split metadata={case}:{record.get('case_id')}")
        if record.get("profile") in (None, "", "unknown"):
            errors.append(f"missing profile={case}")
        if record.get("distribution") in (None, "", "unknown"):
            errors.append(f"missing distribution={case}")
        if not record.get("case_sha256"):
            errors.append(f"missing sha256={case}")

    for case, path in teacher_paths.items():
        try:
            teacher = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid teacher JSON={case}:{error}")
            continue
        try:
            makespan = safe_float(teacher.get("makespan", math.inf))
        except (TypeError, ValueError):
            makespan = None
        if (
            int(teacher.get("schema_version", 0)) < 2
            or not teacher.get("completion_verified")
            or not str(teacher.get("case_id", "")).startswith("train_")
            or teacher.get("profile") in (None, "", "unknown")
            or teacher.get("distribution") in (None, "", "unknown")
            or not teacher.get("case_sha256")
            or makespan is None
            or makespan >= 100000.0
        ):
            errors.append(f"invalid teacher payload={case}")
    return errors


def find_verified_teacher_manifest(teacher_dir: Path):
    """Return the newest complete manifest, ignoring stale partial attempts."""
    candidates = sorted(
        teacher_dir.glob("generation*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    failures = {}
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures[candidate.name] = [f"{type(error).__name__}: {error}"]
            continue
        errors = teacher_validation_errors(teacher_dir, payload)
        if not errors:
            return candidate, payload, {}
        failures[candidate.name] = errors[:20]
    return None, None, failures


def start_teacher_generation(
    args: argparse.Namespace,
    suite_dir: Path,
    teacher_dir: Path,
    env: dict[str, str],
) -> None:
    global TEACHER_PROCESS, TEACHER_LOG_HANDLE
    result_path = teacher_dir / "generation.json"
    expected_cases = len(list(TRAIN_DATA.glob("case_*")))
    existing = list(teacher_dir.glob("case_*.json")) if teacher_dir.exists() else []
    manifest_path, _, failures = find_verified_teacher_manifest(teacher_dir)
    if manifest_path is not None:
        print(
            f"[Suite] Reusing {len(existing)} verified IGA teachers from "
            f"{teacher_dir} via {manifest_path.name}.",
            flush=True,
        )
        return
    if teacher_dir.exists() and existing:
        raise RuntimeError(
            f"Refusing to mix or overwrite IGA teachers in {teacher_dir}: "
            f"{len(existing)}/{expected_cases} files, manifests="
            f"{json.dumps(failures, ensure_ascii=False)}"
        )
    teacher_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(IGA_PARALLEL),
        "--dataset_test_dir", str(TRAIN_DATA),
        "--output_json", str(result_path),
        "--methods", "iga",
        "--workers", str(args.teacher_workers),
        "--max_cases", "0",
        "--time_budget", str(args.teacher_time_budget),
        "--iga_pop_size", str(args.teacher_pop_size),
        "--iga_generations", str(args.teacher_generations),
        "--iga_max_attempts", str(args.teacher_max_attempts),
        "--iga_teacher_dir", str(teacher_dir),
        "--seed", "1",
    ]
    write_command(suite_dir / "iga_teacher_generation.command.json", command, env)
    TEACHER_LOG_HANDLE = (suite_dir / "iga_teacher_generation.log").open(
        "w", encoding="utf-8"
    )
    TEACHER_PROCESS = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=TEACHER_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(
        f"[Suite] Started IGA teacher generation pid={TEACHER_PROCESS.pid}, "
        f"workers={args.teacher_workers}, budget={args.teacher_time_budget}s.",
        flush=True,
    )


def ensure_teachers(
    args: argparse.Namespace,
    suite_status_path: Path,
    suite_state: dict,
    teacher_dir: Path,
) -> None:
    global TEACHER_PROCESS, TEACHER_LOG_HANDLE
    if TEACHER_PROCESS is not None:
        result_path = teacher_dir / "generation.json"
        last_signature = None
        last_progress_unix_time = time.time()
        while TEACHER_PROCESS.poll() is None:
            if STOP_REQUESTED:
                terminate_then_kill(TEACHER_PROCESS)
                raise KeyboardInterrupt("Suite stop requested.")
            signature = (
                result_path.stat().st_size, result_path.stat().st_mtime_ns
            ) if result_path.exists() else (0, 0)
            if signature != last_signature:
                last_signature = signature
                last_progress_unix_time = time.time()
            no_progress_seconds = time.time() - last_progress_unix_time
            if no_progress_seconds >= float(args.stall_timeout_seconds):
                terminate_then_kill(TEACHER_PROCESS)
                raise TimeoutError(
                    "IGA teacher generation made no result progress for "
                    f"{no_progress_seconds:.1f}s."
                )
            suite_state.update(
                status="waiting_for_iga_teachers",
                current="IGA teacher generation",
                teacher_pid=TEACHER_PROCESS.pid,
                no_progress_seconds=no_progress_seconds,
                updated_unix_time=time.time(),
            )
            atomic_json(suite_status_path, suite_state)
            print(
                "[Suite] Waiting for IGA teacher generation; "
                f"no_progress={no_progress_seconds:.1f}s.",
                flush=True,
            )
            time.sleep(args.monitor_interval)
        return_code = TEACHER_PROCESS.returncode
        TEACHER_PROCESS = None
        if TEACHER_LOG_HANDLE is not None:
            TEACHER_LOG_HANDLE.close()
            TEACHER_LOG_HANDLE = None
        if return_code != 0:
            raise RuntimeError(
                f"IGA teacher generation exited with status {return_code}."
            )
    manifest_path, payload, failures = find_verified_teacher_manifest(teacher_dir)
    if manifest_path is None:
        raise RuntimeError(
            "No complete verified IGA teacher manifest in "
            f"{teacher_dir}: {json.dumps(failures, ensure_ascii=False)}"
        )
    teacher_count = len(list(teacher_dir.glob("case_*.json")))
    suite_state.update(
        status="teachers_verified",
        teacher_manifest=str(manifest_path),
        current=None,
        teacher_pid=None,
        no_progress_seconds=0.0,
        updated_unix_time=time.time(),
    )
    atomic_json(suite_status_path, suite_state)
    print(f"[Suite] Validated all {teacher_count} IGA teachers via {manifest_path.name}.", flush=True)


def trajectory_analysis_command(
    args: argparse.Namespace,
    teacher_dir: Path,
    output_json: Path,
    trajectory_jsonl: Path,
) -> list[str]:
    return [
        str(PYTHON),
        str(IGA_TRAJECTORY_ANALYSIS),
        "--dataset_dir", str(TRAIN_DATA),
        "--teacher_dir", str(teacher_dir),
        "--env_config", str(ENV_CONFIG),
        "--output_json", str(output_json),
        "--trajectory_jsonl", str(trajectory_jsonl),
        "--workers", str(args.trajectory_workers),
        "--max_steps", "4000",
        "--max_cases", "0",
        "--ridge", str(args.potential_ridge),
        "--seed", "1",
        "--distribution_weights", TARGET_DISTRIBUTION_WEIGHTS,
    ]


def validated_trajectory_analysis(path: Path, expected_cases: int) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing IGA trajectory analysis: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    replay = payload.get("replay", {})
    if (
        payload.get("status") != "completed"
        or int(replay.get("expected_cases", -1)) != expected_cases
        or int(replay.get("reverified_cases", -1)) != expected_cases
    ):
        raise RuntimeError(
            f"IGA trajectory replay is incomplete under current code: {replay}"
        )
    mean_drift = safe_float(replay.get("mean_relative_cmax_drift", math.nan))
    if mean_drift is None or mean_drift > 0.02:
        raise RuntimeError(
            f"Current IGA replay mean Cmax drift exceeds the 2% gate: {mean_drift}."
        )
    expected_features = [
        "remaining_work",
        "remaining_jobs",
        "waiting_age",
        "resource_queue",
        "relocation_debt",
        "max_plane_remaining_work",
        "max_waiting_age",
        "future_release_tail",
    ]
    if payload.get("feature_names") != expected_features:
        raise RuntimeError(f"Unexpected IGA potential features: {payload.get('feature_names')}")
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise RuntimeError("IGA trajectory analysis has no weight mapping.")
    for name in expected_features:
        value = safe_float(weights.get(name, math.nan))
        if value is None or value < 0.0:
            raise RuntimeError(f"Invalid IGA potential weight {name}={value}.")
    if not any(float(weights[name]) > 0.0 for name in expected_features):
        raise RuntimeError("All IGA potential weights are zero.")
    calibration = payload.get("calibration", {})
    if not calibration.get("case_balanced", False):
        raise RuntimeError("IGA potential calibration is not case-balanced.")
    expected_distribution_weights = {
        "iid": 0.50,
        "ood_stress": 0.45,
        "ood_scale": 0.05,
    }
    if calibration.get("distribution_weights") != expected_distribution_weights:
        raise RuntimeError(
            "Unexpected IGA calibration distribution weights: "
            f"{calibration.get('distribution_weights')}"
        )
    return payload


def main() -> int:
    global TEACHER_PROCESS, TEACHER_LOG_HANDLE
    args = parse_args()
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if args.teacher_workers > len(allowed_cpus):
        raise ValueError(
            "IGA teacher workers cannot exceed the suite CPU affinity: "
            f"workers={args.teacher_workers}, allowed_cpus={len(allowed_cpus)}."
        )
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)

    for required in (
        PYTHON, TRAIN, COMPARE, IGA_PARALLEL, IGA_TRAJECTORY_ANALYSIS,
        AC_CONFIG, ENV_CONFIG,
        TRAIN_DATA, TEST_DATA,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    suite_dir = LOG_ROOT / f"stage1_suite_{args.run_tag}"
    if suite_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to reuse existing suite directory without --resume: "
            f"{suite_dir}"
        )
    if args.resume and not suite_dir.exists():
        raise FileNotFoundError(
            f"Cannot resume missing suite directory: {suite_dir}"
        )
    suite_dir.mkdir(parents=True, exist_ok=args.resume)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = suite_dir / "gpu_metrics.tsv"
    if not metrics_path.exists():
        metrics_path.write_text(
            "unix_time\tlabel\tnvidia_smi_sample\n", encoding="utf-8"
        )
    status_path = suite_dir / "suite_status.json"
    summary_path = suite_dir / "suite_summary.json"

    lock_path = LOG_ROOT / f"gpu{args.gpu}_stage1_suite.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"Another Stage-1 suite holds {lock_path}."
        ) from error
    lock_handle.write(f"pid={os.getpid()} run_tag={args.run_tag}\n")
    lock_handle.flush()

    idle, users = gpu_is_idle(args.gpu)
    if not idle:
        raise RuntimeError(
            f"GPU {args.gpu} is not idle before suite launch: {users}"
        )

    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MPLCONFIGDIR": "/tmp",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    if args.study == "ppo_recovery":
        run_variants, aliases = ppo_recovery_variants()
        initialization_checkpoint = Path(
            args.plane_bc_checkpoint
        ).resolve()
        initialization_record = validate_plane_bc_checkpoint(
            initialization_checkpoint
        )
    elif args.study == "tail_recovery":
        run_variants, aliases = tail_recovery_variants(
            args.plane_bc_epochs
        )
        initialization_checkpoint = None
        initialization_record = None
    else:
        run_variants, aliases = variants(args.plane_bc_epochs)
        initialization_checkpoint = None
        initialization_record = None
    if args.teacher_dir:
        teacher_dir = Path(args.teacher_dir).resolve()
    else:
        teacher_dir = (
            LOG_ROOT / "iga_teachers"
            / (
                f"fjspv3_t600_train_p{args.teacher_pop_size}_g"
                f"{args.teacher_generations}_t{int(args.teacher_time_budget)}_"
                f"a{args.teacher_max_attempts}_s1_verified_v1"
            )
        )
    if args.resume:
        if not status_path.is_file() or not summary_path.is_file():
            raise RuntimeError(
                "Resume requires existing suite_status.json and "
                "suite_summary.json."
            )
        suite_state = relocate_saved_paths(
            json.loads(status_path.read_text(encoding="utf-8"))
        )
        summary = relocate_saved_paths(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        validate_resume_summary(summary, args, run_variants, aliases)
        resume_count = int(suite_state.get("resume_count", 0)) + 1
        for stale_key in (
            "error", "traceback", "completed_unix_time", "no_progress_seconds",
        ):
            suite_state.pop(stale_key, None)
        suite_state.update(
            status="resuming",
            current=None,
            current_pid=None,
            pid=os.getpid(),
            resume_count=resume_count,
            resumed_unix_time=time.time(),
            updated_unix_time=time.time(),
        )
        atomic_json(status_path, suite_state)
        print(
            f"[Suite] Resuming {args.run_tag}; completed screen="
            f"{len(summary.get('screen', []))}, formal="
            f"{len(summary.get('formal', []))}.",
            flush=True,
        )
    else:
        suite_state = {
            "status": "initializing",
            "run_tag": args.run_tag,
            "pid": os.getpid(),
            "gpu": args.gpu,
            "screen_epochs": args.screen_epochs,
            "formal_epochs": args.formal_epochs,
            "formal_seeds": args.formal_seeds,
            "rollout_threads": args.rollout_threads,
            "eval_threads": args.eval_threads,
            "graphs_per_forward": args.max_graphs_per_forward,
            "teacher_dir": str(teacher_dir),
            "initialization_checkpoint": initialization_record,
            "cpu_affinity": allowed_cpus,
            "variants": run_variants,
            "aliases": aliases,
            "started_unix_time": time.time(),
            "updated_unix_time": time.time(),
        }
        atomic_json(status_path, suite_state)
        summary = {
            "run_tag": args.run_tag,
            "configuration": vars(args),
            "teacher_dir": str(teacher_dir),
            "initialization_checkpoint": initialization_record,
            "cpu_affinity": allowed_cpus,
            "variants": run_variants,
            "aliases": aliases,
            "screen": [],
            "selected": [],
            "formal": [],
            "external_baselines": {},
            "external_baseline_note": (
                "FJSP-v2 baselines are not comparable to the regenerated "
                "FJSP-v3 test split."
            ),
        }
        atomic_json(summary_path, summary)

    potential_weights_path = suite_dir / "iga_trajectory_analysis.json"
    trajectory_jsonl = suite_dir / "iga_teacher_trajectories.jsonl"
    needs_iga_potential = any(
        variant["reward"] == "iga_potential" for variant in run_variants
    )
    needs_teachers = (
        needs_iga_potential
        or any(
            int(variant["plane_bc_epochs"]) > 0
            for variant in run_variants
        )
    )
    if needs_teachers:
        start_teacher_generation(args, suite_dir, teacher_dir, env)
        ensure_teachers(args, status_path, suite_state, teacher_dir)
        teacher_manifest, teacher_payload, _ = find_verified_teacher_manifest(
            teacher_dir
        )
        summary["teacher_manifest"] = str(teacher_manifest)
        summary["teacher_summary"] = teacher_payload.get(
            "methods", {}
        ).get("IGA", {}).get("summary", {})
        summary["teacher_selection"] = teacher_payload.get(
            "teacher_selection", {}
        )
    else:
        summary["teacher_manifest"] = None
        summary["teacher_summary"] = {
            "status": "skipped",
            "reason": "all runs reuse the supplied PlaneBC checkpoint",
        }
        summary["teacher_selection"] = {}
    atomic_json(summary_path, summary)

    if needs_iga_potential:
        expected_teacher_cases = len(list(TRAIN_DATA.glob("case_*")))
        try:
            trajectory_analysis = validated_trajectory_analysis(
                potential_weights_path, expected_teacher_cases
            )
            print(
                f"[Suite] Reusing exact IGA trajectory analysis "
                f"{potential_weights_path}.", flush=True
            )
        except (RuntimeError, OSError, ValueError, TypeError, json.JSONDecodeError):
            command = trajectory_analysis_command(
                args, teacher_dir, potential_weights_path, trajectory_jsonl
            )
            run_monitored(
                command, "IGA teacher trajectory calibration",
                suite_dir / "iga_trajectory_analysis.log",
                suite_dir / "iga_trajectory_analysis.command.json",
                env, status_path, suite_state, metrics_path, args.gpu,
                args.monitor_interval, args.stall_timeout_seconds,
            )
            trajectory_analysis = validated_trajectory_analysis(
                potential_weights_path, expected_teacher_cases
            )
        summary["iga_trajectory_analysis"] = trajectory_analysis
        summary["iga_trajectory_jsonl"] = str(trajectory_jsonl)
    else:
        summary["iga_trajectory_analysis"] = {
            "status": "skipped", "reason": "no iga_potential variant"
        }
    atomic_json(summary_path, summary)

    variant_by_id = {variant["id"]: variant for variant in run_variants}
    completed_screen = validated_completed_screen_records(summary)
    tail_foundation_checkpoint = None
    if (
        args.study == "tail_recovery"
        and TAIL_RECOVERY_FOUNDATION_ID in completed_screen
    ):
        foundation_experiment = completed_screen[
            TAIL_RECOVERY_FOUNDATION_ID
        ]["experiment_name"]
        tail_foundation_checkpoint = latest_checkpoint(
            foundation_experiment, "checkpoint_PlaneBC.pt"
        )
        summary["tail_recovery_foundation"] = validate_plane_bc_checkpoint(
            tail_foundation_checkpoint,
            expected_global_feature_mode="f1f2",
        )
        atomic_json(summary_path, summary)
    diagnostic_gate_passed = next(
        (
            record["id"]
            for record in completed_screen.values()
            if record.get("diagnostic_gate", {}).get("passed")
        ),
        None,
    )

    for index, variant in enumerate(run_variants, start=1):
        if (
            args.study == "ppo_recovery"
            and diagnostic_gate_passed is not None
            and not args.continue_after_gate_pass
        ):
            print(
                "[Suite] Stopping the sequential screen because "
                f"{diagnostic_gate_passed} already passed every gate.",
                flush=True,
            )
            break
        if int(variant["plane_bc_epochs"]) > 0:
            ensure_teachers(args, status_path, suite_state, teacher_dir)
        identifier = variant["id"]
        variant_initialization_checkpoint = initialization_checkpoint
        if (
            args.study == "tail_recovery"
            and identifier != TAIL_RECOVERY_FOUNDATION_ID
        ):
            if tail_foundation_checkpoint is None:
                raise RuntimeError(
                    "Tail-recovery PPO ablations require the completed shared "
                    "PlaneBC foundation."
                )
            variant_initialization_checkpoint = (
                tail_foundation_checkpoint
            )
        if identifier in completed_screen:
            print(
                f"[Suite] Skipping completed screen {index}/"
                f"{len(run_variants)} {identifier}.",
                flush=True,
            )
            continue
        experiment = f"{args.run_tag}_screen_{identifier}_seed1"
        checkpoint = (
            completed_training_checkpoint(experiment)
            if args.resume else None
        )
        resume_checkpoint = variant_initialization_checkpoint
        if checkpoint is not None:
            print(
                f"[Suite] Reusing completed screen training for "
                f"{identifier}: {checkpoint}",
                flush=True,
            )
        else:
            interrupted_checkpoint = (
                interrupted_training_checkpoint(experiment)
                if args.resume else None
            )
            preserve_resume_state = interrupted_checkpoint is not None
            resume_checkpoint = (
                interrupted_checkpoint
                if interrupted_checkpoint is not None
                else variant_initialization_checkpoint
            )
            if preserve_resume_state:
                print(
                    f"[Suite] Continuing interrupted screen {identifier} from "
                    f"{resume_checkpoint}.",
                    flush=True,
                )
            command = training_command(
                args,
                variant=variant,
                experiment_name=experiment,
                seed=1,
                epochs=args.screen_epochs,
                formal=False,
                teacher_dir=teacher_dir,
                potential_weights_path=potential_weights_path,
                resume_checkpoint=resume_checkpoint,
                preserve_resume_state=preserve_resume_state,
            )
            run_monitored(
                command,
                f"screen {index}/{len(run_variants)} {identifier}",
                suite_dir / f"screen_{identifier}.log",
                suite_dir / f"screen_{identifier}.command.json",
                env,
                status_path,
                suite_state,
                metrics_path,
                args.gpu,
                args.monitor_interval,
                args.stall_timeout_seconds,
            )
            checkpoint = completed_training_checkpoint(experiment)
            if checkpoint is None:
                raise RuntimeError(
                    "Screen training completed without a reusable Best "
                    f"checkpoint: {experiment}"
                )
        record = {
            "id": identifier,
            "variant": variant,
            "experiment_name": experiment,
            **load_checkpoint_record(checkpoint),
            **pre_ppo_record(experiment),
        }
        record["initialization_checkpoint"] = (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
        record["ppo_improvement_vs_bc"] = (
            record["pre_ppo_eval_makespan"] - record["eval_makespan"]
        )
        record["ppo_relative_improvement_vs_bc"] = (
            record["ppo_improvement_vs_bc"]
            / max(record["pre_ppo_eval_makespan"], 1e-8)
        )
        record["diagnostic_gate"] = latest_epoch_diagnostic(
            checkpoint,
            variant=variant,
            min_step_completion=args.diagnostic_min_step_completion,
            min_relative_improvement=(
                args.diagnostic_min_relative_improvement
            ),
            bootstrap_samples=args.diagnostic_bootstrap_samples,
        )
        summary["screen"].append(record)
        if (
            args.study == "tail_recovery"
            and identifier == TAIL_RECOVERY_FOUNDATION_ID
        ):
            tail_foundation_checkpoint = latest_checkpoint(
                experiment, "checkpoint_PlaneBC.pt"
            )
            summary["tail_recovery_foundation"] = (
                validate_plane_bc_checkpoint(
                    tail_foundation_checkpoint,
                    expected_global_feature_mode="f1f2",
                )
            )
        atomic_json(summary_path, summary)
        if record["diagnostic_gate"]["passed"]:
            diagnostic_gate_passed = identifier
            summary["screen_stopped_after_gate_pass"] = (
                None if args.continue_after_gate_pass else identifier
            )
            atomic_json(summary_path, summary)

    if needs_teachers:
        ensure_teachers(args, status_path, suite_state, teacher_dir)
    summary["screen"] = [
        record for record in summary["screen"]
        if not record.get("alias_of")
    ]
    for alias, source in aliases.items():
        source_record = next(
            record for record in summary["screen"] if record["id"] == source
        )
        summary["screen"].append(
            {
                **source_record,
                "id": alias,
                "alias_of": source,
                "experiment_name": source_record["experiment_name"],
            }
        )

    ranked = sorted(
        (
            record for record in summary["screen"]
            if "alias_of" not in record
        ),
        key=lambda record: record["selection_score"],
    )
    selection_pool = ranked
    if args.study in {"ppo_recovery", "tail_recovery"}:
        selection_pool = [
            record for record in ranked
            if record.get("diagnostic_gate", {}).get("passed")
        ]
    selected = [
        record["id"] for record in selection_pool[: args.top_configs]
    ]
    summary["selected"] = selected
    if args.study in {"ppo_recovery", "tail_recovery"} and not selected:
        summary["formal_skipped"] = {
            "reason": "no screen candidate passed every diagnostic gate"
        }
    atomic_json(summary_path, summary)
    if selected:
        print(
            "[Suite] Selected formal configurations: "
            + ", ".join(
                f"{identifier}="
                f"{next(x['selection_score'] for x in ranked if x['id'] == identifier):.3f}"
                for identifier in selected
            ),
            flush=True,
        )
    else:
        print(
            "[Suite] No screen candidate passed every diagnostic gate; "
            "formal training is intentionally skipped.",
            flush=True,
        )

    if not args.skip_formal:
        completed_formal = validated_completed_formal_records(summary)
        total_formal = len(selected) * len(args.formal_seeds)
        formal_index = 0
        for identifier in selected:
            variant = variant_by_id[identifier]
            if int(variant["plane_bc_epochs"]) > 0:
                ensure_teachers(args, status_path, suite_state, teacher_dir)
            for seed in args.formal_seeds:
                formal_index += 1
                if (identifier, seed) in completed_formal:
                    print(
                        f"[Suite] Skipping completed formal {formal_index}/"
                        f"{total_formal} {identifier} seed{seed}.",
                        flush=True,
                    )
                    continue
                experiment = f"{args.run_tag}_formal_{identifier}_seed{seed}"
                interrupted_checkpoint = (
                    interrupted_training_checkpoint(experiment)
                    if args.resume else None
                )
                formal_initialization_checkpoint = (
                    tail_foundation_checkpoint
                    if args.study == "tail_recovery"
                    else initialization_checkpoint
                )
                if (
                    args.study == "tail_recovery"
                    and formal_initialization_checkpoint is None
                ):
                    raise RuntimeError(
                        "Formal tail-recovery runs require the shared PlaneBC "
                        "foundation checkpoint."
                    )
                resume_checkpoint = (
                    interrupted_checkpoint
                    if interrupted_checkpoint is not None
                    else formal_initialization_checkpoint
                )
                command = training_command(
                    args,
                    variant=variant,
                    experiment_name=experiment,
                    seed=seed,
                    epochs=args.formal_epochs,
                    formal=True,
                    teacher_dir=teacher_dir,
                    potential_weights_path=potential_weights_path,
                    resume_checkpoint=resume_checkpoint,
                    preserve_resume_state=interrupted_checkpoint is not None,
                )
                checkpoint = completed_training_checkpoint(experiment)
                if checkpoint is None:
                    run_monitored(
                        command,
                        f"formal {formal_index}/{total_formal} {identifier} seed{seed}",
                        suite_dir / f"formal_{identifier}_seed{seed}.log",
                        suite_dir / f"formal_{identifier}_seed{seed}.command.json",
                        env,
                        status_path,
                        suite_state,
                        metrics_path,
                        args.gpu,
                        args.monitor_interval,
                        args.stall_timeout_seconds,
                    )
                    checkpoint = completed_training_checkpoint(experiment)
                    if checkpoint is None:
                        raise RuntimeError(
                            f"Training completed without a reusable Best checkpoint: {experiment}"
                        )
                else:
                    print(
                        f"[Suite] Reusing completed formal training for "
                        f"{identifier} seed{seed}: {checkpoint}",
                        flush=True,
                    )
                formal_record = {
                    "id": identifier,
                    "seed": seed,
                    "variant": variant,
                    "experiment_name": experiment,
                    **load_checkpoint_record(checkpoint),
                    **pre_ppo_record(experiment),
                }
                formal_record["ppo_improvement_vs_bc"] = (
                    formal_record["pre_ppo_eval_makespan"]
                    - formal_record["eval_makespan"]
                )
                formal_record["ppo_relative_improvement_vs_bc"] = (
                    formal_record["ppo_improvement_vs_bc"]
                    / max(formal_record["pre_ppo_eval_makespan"], 1e-8)
                )
                formal_record["initialization_checkpoint"] = str(
                    resume_checkpoint
                ) if resume_checkpoint is not None else None
                evaluation_json = (
                    EVAL_ROOT
                    / f"{args.run_tag}_{identifier}_seed{seed}_test60.json"
                )
                evaluation_command = comparison_command(
                    variant, checkpoint, evaluation_json, seed
                )
                run_monitored(
                    evaluation_command,
                    f"test60 {identifier} seed{seed}",
                    suite_dir / f"test60_{identifier}_seed{seed}.log",
                    suite_dir / f"test60_{identifier}_seed{seed}.command.json",
                    env,
                    status_path,
                    suite_state,
                    metrics_path,
                    args.gpu,
                    args.monitor_interval,
                    args.stall_timeout_seconds,
                )
                evaluation = json.loads(
                    evaluation_json.read_text(encoding="utf-8")
                )
                drl = evaluation.get("methods", {}).get("DRL-G", {})
                if drl.get("status") != "completed":
                    raise RuntimeError(
                        f"Incomplete test evaluation: {evaluation_json}"
                    )
                formal_record.update(
                    {
                        "test_json": str(evaluation_json),
                        "test_summary": drl.get("summary", {}),
                        "test_tau": drl.get("tau"),
                    }
                )
                summary["formal"].append(formal_record)
                atomic_json(summary_path, summary)

    summary["completed_unix_time"] = time.time()
    atomic_json(summary_path, summary)
    suite_state.update(
        status="completed",
        current=None,
        current_pid=None,
        completed_unix_time=time.time(),
        updated_unix_time=time.time(),
        summary_path=str(summary_path),
    )
    atomic_json(status_path, suite_state)
    print(f"[Suite] All requested experiments completed: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        terminate_then_kill(ACTIVE_PROCESS)
        terminate_then_kill(TEACHER_PROCESS)
        if TEACHER_LOG_HANDLE is not None:
            TEACHER_LOG_HANDLE.close()
        message = f"{type(error).__name__}: {error}"
        print(f"[Suite][Error] {message}", file=sys.stderr, flush=True)
        traceback.print_exc()
        for candidate in LOG_ROOT.glob("stage1_suite_*/suite_status.json"):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("pid") != os.getpid():
                continue
            payload.update(
                status="failed",
                error=message,
                traceback=traceback.format_exc(),
                updated_unix_time=time.time(),
            )
            atomic_json(candidate, payload)
            break
        raise
