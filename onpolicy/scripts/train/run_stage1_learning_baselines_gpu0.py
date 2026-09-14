#!/usr/bin/env python3
"""Queue and execute the Stage-1 learning baselines on one GPU.

The controller waits for explicitly registered predecessor services and for
the target GPU to become idle.  It then keeps one shared validation service
alive and executes the four methods x three seeds in four memory-safe waves.
Every runtime command, process transition, and hardware sample is persisted
under the suite directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.utils.shared_eval import SharedEvalClient  # noqa: E402


TRAIN = ROOT / "onpolicy/scripts/train/train_hkbz.py"
EVALUATOR = ROOT / "onpolicy/scripts/train/shared_hkbz_evaluator.py"
RESULT_ROOT = ROOT / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo"

METHODS = ("l2d", "multi_ppo", "fjsp_drl", "daniel")
SEEDS = (1, 2, 3)
WAVES = tuple(
    tuple((method, seed) for method in METHODS)
    for seed in SEEDS
)
TRAIN_CPU_SETS = (
    "0-13,72-85",
    "14-27,86-99",
    "28-41,100-113",
    "42-55,114-127",
)
EVALUATOR_CPU_SET = "56-69,128-141"
CONTROLLER_CPU_SET = "70-71,142-143"
CUDA_MEMORY_FRACTIONS = {
    "l2d": 0.15,
    "multi_ppo": 0.22,
    "fjsp_drl": 0.28,
    "daniel": 0.40,
}
EVALUATOR_CUDA_MEMORY_FRACTION = 0.08
BUSY_UNIT_STATES = {"active", "activating", "deactivating", "reloading"}


class ControllerInterrupted(RuntimeError):
    """Raised when systemd or the user asks the controller to stop."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_cpu_set(specification: str) -> frozenset[int]:
    values: set[int] = set()
    for item in specification.split(","):
        lower, separator, upper = item.partition("-")
        if separator:
            values.update(range(int(lower), int(upper) + 1))
        else:
            values.add(int(lower))
    return frozenset(values)


def validate_cpu_partition() -> None:
    specifications = (
        *TRAIN_CPU_SETS,
        EVALUATOR_CPU_SET,
        CONTROLLER_CPU_SET,
    )
    groups = [expand_cpu_set(item) for item in specifications]
    expected_widths = (28, 28, 28, 28, 28, 4)
    widths = tuple(len(group) for group in groups)
    if widths != expected_widths:
        raise RuntimeError(
            f"CPU partition widths are {widths}, expected {expected_widths}."
        )
    combined: set[int] = set()
    for group in groups:
        if combined.intersection(group):
            raise RuntimeError("CPU partition contains overlapping groups.")
        combined.update(group)
        for cpu in group:
            sibling = cpu + 72 if cpu < 72 else cpu - 72
            if sibling not in group:
                raise RuntimeError(
                    f"CPU partition splits SMT siblings {cpu}/{sibling}."
                )
    if combined != set(range(144)):
        raise RuntimeError("CPU partition must cover logical CPUs 0-143.")


def option(command: list[str], flag: str, default: str | None = None) -> str | None:
    count = command.count(flag)
    if count == 0:
        return default
    if count != 1:
        raise ValueError(f"Expected one {flag}, found {count}.")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"Missing value after {flag}.")
    return command[index + 1]


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


def command_contract(command: list[str]) -> dict[str, str]:
    required = {
        "--training_stage": "plane_pretrain",
        "--resource_policy": "heuristic",
        "--plane_order_mode": "fixed",
        "--plane_pair_decoder": "joint_pair",
        "--plane_bc_pretrain_epochs": "0",
        "--bc_reference_kl_coef": "0.0",
        "--bc_reference_target_kl": "0.0",
        "--n_rollout_threads": "60",
        "--n_eval_rollout_threads": "60",
        "--ppo_epoch": "2",
        "--mini_batch_size": "7",
        "--data_chunk_length": "50",
        "--max_graphs_per_forward": "350",
    }
    for flag, expected in required.items():
        actual = option(command, flag)
        if actual != expected:
            raise ValueError(f"{flag}={actual!r}; expected {expected!r}.")
    forbidden = (
        "--checkpoint_dir",
        "--selection_checkpoint_dir",
        "--resume_stage1",
        "--resume_stage2",
        "--plane_bc_only",
        "--bc_reference_hard_gate",
        "--adaptive_bc_reference_kl",
    )
    present = [flag for flag in forbidden if flag in command]
    if present:
        raise ValueError(f"Baseline command contains forbidden state: {present}.")
    if "--use_eval" not in command:
        raise ValueError("Baseline command must keep validation enabled.")
    environment = option(command, "--env_config")
    evaluation = option(command, "--eval_dataset_dir")
    experiment = option(command, "--experiment_name")
    if not environment or not Path(environment).is_file():
        raise FileNotFoundError(environment or "--env_config")
    if not evaluation or not Path(evaluation).is_dir():
        raise FileNotFoundError(evaluation or "--eval_dataset_dir")
    if not experiment:
        raise ValueError("Baseline command is missing --experiment_name.")
    return {
        "environment": str(Path(environment).resolve()),
        "evaluation": str(Path(evaluation).resolve()),
        "experiment": experiment,
    }


def load_plan(manifest_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payload = read_json(manifest_path)
    if payload.get("scope") != "stage1_learning_baselines":
        raise ValueError("Manifest is not a Stage-1 learning-baseline suite.")
    if tuple(payload.get("methods", ())) != METHODS:
        raise ValueError(f"Unexpected methods: {payload.get('methods')!r}.")
    if tuple(int(item) for item in payload.get("seeds", ())) != SEEDS:
        raise ValueError(f"Unexpected seeds: {payload.get('seeds')!r}.")
    plan: dict[tuple[str, int], dict[str, Any]] = {}
    common_contract: dict[str, str] | None = None
    for record in payload.get("runs", ()):
        method = str(record.get("stage1_baseline", ""))
        seed = int(record.get("seed", -1))
        key = (method, seed)
        if method not in METHODS or seed not in SEEDS or key in plan:
            raise ValueError(f"Invalid or duplicate run record {key!r}.")
        command_path = Path(str(record.get("command_json", ""))).resolve()
        if not command_path.is_file():
            raise FileNotFoundError(command_path)
        command_payload = read_json(command_path)
        if command_payload.get("stage1_baseline") != method:
            raise ValueError(f"Method mismatch in {command_path}.")
        if int(command_payload.get("seed", -1)) != seed:
            raise ValueError(f"Seed mismatch in {command_path}.")
        command = list(command_payload.get("command", ()))
        if len(command) < 2:
            raise ValueError(f"Invalid training command in {command_path}.")
        if option(command, "--stage1_baseline") != method:
            raise ValueError(f"CLI method mismatch in {command_path}.")
        if option(command, "--seed") != str(seed):
            raise ValueError(f"CLI seed mismatch in {command_path}.")
        contract = command_contract(command)
        comparable = {
            "environment": contract["environment"],
            "evaluation": contract["evaluation"],
            "num_episodes": str(option(command, "--num_episodes")),
            "hindsight_reward_mode": str(
                option(command, "--hindsight_reward_mode")
            ),
        }
        if common_contract is None:
            common_contract = comparable
        elif comparable != common_contract:
            raise ValueError(
                f"Fair-comparison contract differs for {key}: {comparable}."
            )
        experiment_dir = RESULT_ROOT / contract["experiment"]
        if experiment_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse experiment output {experiment_dir}."
            )
        plan[key] = {
            "method": method,
            "seed": seed,
            "command_path": command_path,
            "command_payload": command_payload,
            **contract,
        }
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(plan) != expected:
        raise ValueError(
            f"Manifest run set mismatch: missing={sorted(expected - set(plan))}, "
            f"extra={sorted(set(plan) - expected)}."
        )
    wave_keys = [key for wave in WAVES for key in wave]
    if len(wave_keys) != len(set(wave_keys)) or set(wave_keys) != expected:
        raise RuntimeError("Static wave plan does not cover every run exactly once.")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--phase-lock-path", type=Path, required=True)
    parser.add_argument("--predecessor-unit", action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stagger-seconds", type=float, default=480.0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--minimum-free-memory-mib", type=int, default=65536)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def unit_state(unit: str) -> str:
    normalized = unit if unit.endswith(".service") else f"{unit}.service"
    result = subprocess.run(
        ("systemctl", "--user", "is-active", normalized),
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    row = subprocess.run(
        (
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if row.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {row.stderr.strip()}")
    fields = [item.strip() for item in row.stdout.strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi row: {row.stdout!r}")
    processes = subprocess.run(
        (
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if processes.returncode != 0:
        raise RuntimeError(f"nvidia-smi process query failed: {processes.stderr}")
    pids = [
        int(line.strip())
        for line in processes.stdout.splitlines()
        if line.strip().isdigit()
    ]
    return {
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "memory_free_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
        "compute_pids": pids,
    }


def append_hardware_sample(path: Path, snapshot: dict[str, Any], wave: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow((
                "unix_time",
                "wave",
                "gpu_memory_used_mib",
                "gpu_memory_free_mib",
                "gpu_util_percent",
                "compute_pids",
            ))
        writer.writerow((
            time.time(),
            wave,
            snapshot["memory_used_mib"],
            snapshot["memory_free_mib"],
            snapshot["utilization_percent"],
            ";".join(str(pid) for pid in snapshot["compute_pids"]),
        ))


def process_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONHASHSEED": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/hkbz-mpl",
    })
    return environment


def stop_process(process: subprocess.Popen | None, *, timeout: float = 180.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def runtime_command(
    entry: dict[str, Any],
    *,
    socket_path: Path,
    phase_lock_path: Path,
) -> list[str]:
    command = list(entry["command_payload"]["command"])
    command[0] = str(Path(sys.executable).resolve())
    command[1] = str(TRAIN)
    set_option(command, "--shared_eval_socket", socket_path)
    set_option(command, "--shared_eval_cpu_set", EVALUATOR_CPU_SET)
    set_option(command, "--shared_eval_timeout_seconds", 21600)
    set_option(command, "--shared_gpu_phase_lock", phase_lock_path)
    set_option(
        command,
        "--cuda_memory_fraction",
        CUDA_MEMORY_FRACTIONS[entry["method"]],
    )
    set_option(command, "--ipc_timeout_seconds", 600)
    set_option(command, "--safe_async_graph_clone_workers", 4)
    set_switch(command, "--use_eval", True)
    set_switch(command, "--safe_graph_batch_pipeline", True)
    set_switch(command, "--safe_dagger_teacher_overlap", True)
    set_switch(command, "--shared_encoder_activation_checkpoint", True)
    set_switch(command, "--clear_cuda_cache_after_update", True)
    return command


def evaluator_command(
    source_command: Path,
    *,
    socket_path: Path,
    run_dir: Path,
) -> list[str]:
    return [
        "/usr/bin/taskset",
        "--cpu-list",
        EVALUATOR_CPU_SET,
        str(Path(sys.executable).resolve()),
        "-u",
        str(EVALUATOR),
        "--source-command-json",
        str(source_command),
        "--socket-path",
        str(socket_path),
        "--cpu-pool",
        EVALUATOR_CPU_SET,
        "--run-dir",
        str(run_dir),
        "--cuda-memory-fraction",
        str(EVALUATOR_CUDA_MEMORY_FRACTION),
    ]


def wait_for_evaluator(
    process: subprocess.Popen,
    socket_path: Path,
    *,
    timeout: float = 1200.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "socket not created"
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Shared evaluator exited before readiness with {exit_code}."
            )
        if socket_path.is_socket():
            try:
                response = SharedEvalClient(
                    str(socket_path), timeout_seconds=5.0
                ).ping()
                if int(response.get("worker_count", -1)) == 60:
                    return dict(response)
                last_error = f"unexpected ping response {response!r}"
            except Exception as error:  # readiness polling
                last_error = f"{type(error).__name__}: {error}"
        time.sleep(2)
    raise TimeoutError(f"Shared evaluator readiness timed out: {last_error}")


def latest_run_status(experiment: str) -> tuple[str, str | None]:
    experiment_dir = RESULT_ROOT / experiment
    candidates: list[tuple[int, Path]] = []
    if experiment_dir.is_dir():
        for run_dir in experiment_dir.glob("run*"):
            suffix = run_dir.name.removeprefix("run")
            if suffix.isdigit():
                candidates.append((int(suffix), run_dir))
    if not candidates:
        return "missing", None
    run_dir = max(candidates)[1]
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return "missing_run_status", str(run_dir)
    return str(read_json(status_path).get("status", "unknown")), str(run_dir)


def main() -> int:
    args = parse_args()
    if args.gpu != 0:
        raise ValueError("This registered controller is restricted to GPU0.")
    if args.poll_seconds < 5.0:
        raise ValueError("--poll-seconds must be at least 5 seconds.")
    if args.stagger_seconds < 0.0:
        raise ValueError("--stagger-seconds cannot be negative.")
    if args.minimum_free_memory_mib <= 0:
        raise ValueError("--minimum-free-memory-mib must be positive.")
    args.manifest = args.manifest.resolve()
    args.suite_dir = args.suite_dir.resolve()
    args.phase_lock_path = args.phase_lock_path.resolve()
    if not args.phase_lock_path.is_absolute():
        raise ValueError("--phase-lock-path must be absolute.")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.manifest.parent.parent != args.suite_dir:
        raise ValueError("Manifest must live under SUITE_DIR/commands.")
    if args.socket_path.exists():
        raise FileExistsError(f"Refusing stale socket {args.socket_path}.")
    for required in (TRAIN, EVALUATOR, Path(sys.executable)):
        if not required.exists():
            raise FileNotFoundError(required)
    validate_cpu_partition()
    plan = load_plan(args.manifest)
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_ok",
            "gpu": args.gpu,
            "run_count": len(plan),
            "waves": WAVES,
            "trainer_cpu_sets": TRAIN_CPU_SETS,
            "evaluator_cpu_set": EVALUATOR_CPU_SET,
            "cuda_memory_fractions": CUDA_MEMORY_FRACTIONS,
            "evaluator_cuda_memory_fraction": (
                EVALUATOR_CUDA_MEMORY_FRACTION
            ),
            "phase_lock_path": str(args.phase_lock_path),
            "allow_busy_gpu": bool(args.allow_busy_gpu),
            "minimum_free_memory_mib": args.minimum_free_memory_mib,
        }, ensure_ascii=False, indent=2))
        return 0

    args.suite_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.suite_dir / "controller_status.json"
    if status_path.exists():
        raise FileExistsError(f"Controller status already exists: {status_path}")
    log_dir = args.suite_dir / "service_logs"
    runtime_dir = args.suite_dir / "runtime_commands"
    hardware_path = args.suite_dir / "hardware/runtime_usage.csv"
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "gpu": args.gpu,
        "controller_pid": os.getpid(),
        "created_unix_time": time.time(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "predecessor_units": list(args.predecessor_unit),
        "poll_seconds": args.poll_seconds,
        "stagger_seconds": args.stagger_seconds,
        "waves": [[list(key) for key in wave] for wave in WAVES],
        "trainer_cpu_sets": list(TRAIN_CPU_SETS),
        "evaluator_cpu_set": EVALUATOR_CPU_SET,
        "cuda_memory_fractions": CUDA_MEMORY_FRACTIONS,
        "evaluator_cuda_memory_fraction": EVALUATOR_CUDA_MEMORY_FRACTION,
        "phase_lock_path": str(args.phase_lock_path),
        "allow_busy_gpu": bool(args.allow_busy_gpu),
        "minimum_free_memory_mib": args.minimum_free_memory_mib,
        "runs": {},
    }
    atomic_json(status_path, state)
    evaluator: subprocess.Popen | None = None
    evaluator_log = None
    running: dict[tuple[str, int], dict[str, Any]] = {}
    interrupted: list[int] = []

    def handle_signal(signum, _frame):
        interrupted.append(int(signum))
        raise ControllerInterrupted(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        os.sched_setaffinity(0, expand_cpu_set(CONTROLLER_CPU_SET))
        state["waiting_started_unix_time"] = time.time()
        if args.allow_busy_gpu:
            predecessor_states = {
                unit: unit_state(unit) for unit in args.predecessor_unit
            }
            snapshot = gpu_snapshot(args.gpu)
            active_predecessors = {
                unit: value
                for unit, value in predecessor_states.items()
                if value in BUSY_UNIT_STATES
            }
            if snapshot["memory_free_mib"] < args.minimum_free_memory_mib:
                raise RuntimeError(
                    "GPU0 does not have the registered immediate-launch "
                    f"headroom: free={snapshot['memory_free_mib']} MiB, "
                    f"required={args.minimum_free_memory_mib} MiB."
                )
            state.update({
                "status": "busy_gpu_overlap_accepted",
                "heartbeat_unix_time": time.time(),
                "predecessor_states": predecessor_states,
                "active_predecessors": active_predecessors,
                "gpu_snapshot": snapshot,
                "overlap_compute_pids": snapshot["compute_pids"],
            })
            atomic_json(status_path, state)
            append_hardware_sample(hardware_path, snapshot, wave=0)
        else:
            state["status"] = "waiting_for_gpu0"
            idle_observations = 0
            while idle_observations < 2:
                predecessor_states = {
                    unit: unit_state(unit) for unit in args.predecessor_unit
                }
                snapshot = gpu_snapshot(args.gpu)
                active_predecessors = {
                    unit: value
                    for unit, value in predecessor_states.items()
                    if value in BUSY_UNIT_STATES
                }
                idle = not active_predecessors and not snapshot["compute_pids"]
                idle_observations = idle_observations + 1 if idle else 0
                state.update({
                    "heartbeat_unix_time": time.time(),
                    "predecessor_states": predecessor_states,
                    "active_predecessors": active_predecessors,
                    "gpu_snapshot": snapshot,
                    "consecutive_idle_observations": idle_observations,
                })
                atomic_json(status_path, state)
                append_hardware_sample(hardware_path, snapshot, wave=0)
                if idle_observations < 2:
                    time.sleep(args.poll_seconds)

            final_idle = gpu_snapshot(args.gpu)
            if final_idle["compute_pids"]:
                raise RuntimeError(
                    "GPU0 became busy during handoff: "
                    f"{final_idle['compute_pids']}."
                )
        state["status"] = "starting_evaluator"
        state["gpu_acquired_unix_time"] = time.time()
        atomic_json(status_path, state)
        evaluator_source = plan[("l2d", 1)]["command_path"]
        evaluator_log_path = log_dir / "shared_evaluator_gpu0.log"
        evaluator_log = evaluator_log_path.open("a", encoding="utf-8")
        evaluator = subprocess.Popen(
            evaluator_command(
                evaluator_source,
                socket_path=args.socket_path,
                run_dir=args.suite_dir / "shared_evaluator/gpu0",
            ),
            cwd=ROOT,
            env=process_environment(args.gpu),
            stdout=evaluator_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        state["evaluator_pid"] = evaluator.pid
        state["evaluator_log"] = str(evaluator_log_path)
        atomic_json(status_path, state)
        ping = wait_for_evaluator(evaluator, args.socket_path)
        state["evaluator_ready"] = ping
        state["evaluator_ready_unix_time"] = time.time()
        failures: list[str] = []

        for wave_index, wave in enumerate(WAVES, start=1):
            state["status"] = "running"
            state["current_wave"] = wave_index
            state["wave_started_unix_time"] = time.time()
            atomic_json(status_path, state)
            wave_started = time.monotonic()
            pending = [
                {
                    "key": key,
                    "lane": lane,
                    "launch_at": wave_started + lane * args.stagger_seconds,
                }
                for lane, key in enumerate(wave)
            ]
            wave_running: dict[tuple[str, int], dict[str, Any]] = {}
            while pending or wave_running:
                if evaluator.poll() is not None:
                    raise RuntimeError(
                        "Shared evaluator exited during training with "
                        f"code {evaluator.returncode}."
                    )
                now = time.monotonic()
                due = [item for item in pending if item["launch_at"] <= now]
                for item in due:
                    pending.remove(item)
                    key = item["key"]
                    entry = plan[key]
                    lane = int(item["lane"])
                    command = runtime_command(
                        entry,
                        socket_path=args.socket_path,
                        phase_lock_path=args.phase_lock_path,
                    )
                    runtime_path = runtime_dir / (
                        f"wave{wave_index}_{entry['method']}_seed{entry['seed']}.json"
                    )
                    atomic_json(runtime_path, {
                        "schema_version": 1,
                        "wave": wave_index,
                        "lane": lane,
                        "method": entry["method"],
                        "seed": entry["seed"],
                        "cpu_set": TRAIN_CPU_SETS[lane],
                        "gpu": args.gpu,
                        "cuda_memory_fraction": CUDA_MEMORY_FRACTIONS[
                            entry["method"]
                        ],
                        "source_command_json": str(entry["command_path"]),
                        "command": command,
                    })
                    log_path = log_dir / (
                        f"wave{wave_index}_{entry['method']}_seed{entry['seed']}.log"
                    )
                    log_handle = log_path.open("a", encoding="utf-8")
                    process = subprocess.Popen(
                        [
                            "/usr/bin/taskset",
                            "--cpu-list",
                            TRAIN_CPU_SETS[lane],
                            *command,
                        ],
                        cwd=ROOT,
                        env=process_environment(args.gpu),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    record = {
                        "process": process,
                        "log_handle": log_handle,
                        "log_path": log_path,
                        "entry": entry,
                        "lane": lane,
                        "started_unix_time": time.time(),
                    }
                    wave_running[key] = record
                    running[key] = record
                    run_key = f"{entry['method']}_seed{entry['seed']}"
                    state["runs"][run_key] = {
                        "status": "running",
                        "wave": wave_index,
                        "lane": lane,
                        "pid": process.pid,
                        "cpu_set": TRAIN_CPU_SETS[lane],
                        "cuda_memory_fraction": CUDA_MEMORY_FRACTIONS[
                            entry["method"]
                        ],
                        "log": str(log_path),
                        "runtime_command": str(runtime_path),
                        "started_unix_time": record["started_unix_time"],
                    }

                for key, record in list(wave_running.items()):
                    process = record["process"]
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                    record["log_handle"].close()
                    del wave_running[key]
                    del running[key]
                    entry = record["entry"]
                    run_key = f"{entry['method']}_seed{entry['seed']}"
                    run_status, run_dir = latest_run_status(entry["experiment"])
                    successful = exit_code == 0 and run_status == "completed"
                    state["runs"][run_key].update({
                        "status": "completed" if successful else "failed",
                        "exit_code": exit_code,
                        "run_status": run_status,
                        "run_dir": run_dir,
                        "finished_unix_time": time.time(),
                    })
                    if not successful:
                        failures.append(run_key)

                snapshot = gpu_snapshot(args.gpu)
                append_hardware_sample(
                    hardware_path, snapshot, wave=wave_index
                )
                state.update({
                    "heartbeat_unix_time": time.time(),
                    "gpu_snapshot": snapshot,
                    "pending_in_wave": [
                        f"{item['key'][0]}_seed{item['key'][1]}"
                        for item in pending
                    ],
                    "running_in_wave": [
                        f"{key[0]}_seed{key[1]}" for key in wave_running
                    ],
                    "failures": failures,
                })
                atomic_json(status_path, state)
                if pending or wave_running:
                    next_due = min(
                        (item["launch_at"] for item in pending),
                        default=time.monotonic() + args.poll_seconds,
                    )
                    delay = min(
                        args.poll_seconds,
                        max(1.0, next_due - time.monotonic()),
                    )
                    time.sleep(delay)

            state.setdefault("completed_waves", []).append(wave_index)
            state["wave_finished_unix_time"] = time.time()
            atomic_json(status_path, state)

        state["status"] = "completed" if not failures else "completed_with_failures"
        state["completed_unix_time"] = time.time()
        state["failures"] = failures
        atomic_json(status_path, state)
        return 0 if not failures else 1
    except BaseException as error:
        state.update({
            "status": "interrupted" if interrupted else "failed",
            "error": f"{type(error).__name__}: {error}",
            "failed_unix_time": time.time(),
        })
        atomic_json(status_path, state)
        raise
    finally:
        for record in list(running.values()):
            stop_process(record.get("process"))
            handle = record.get("log_handle")
            if handle is not None and not handle.closed:
                handle.close()
        stop_process(evaluator)
        if evaluator_log is not None and not evaluator_log.closed:
            evaluator_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
