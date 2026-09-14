#!/usr/bin/env python3
"""Emit one immutable arm of the eight-GPU Stage3 credit factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
DEFAULT_PYTHON = (
    ROOT.parent / "conda/envs/maia-hkbz-cu124-20260903/bin/python3.11"
)
DEFAULT_SOURCE = RESULTS_ROOT / (
    "stage2_adaptive_trust_20260830_r4_wave1_"
    "T2_backtrack_adaptive_soft_bc_seed1/run1/models/checkpoint_Epoch3.pt"
)
DEFAULT_BASE_MANIFEST = ROOT / (
    "result/hkbz_train_logs/stage3_t2_gradconf_4090_20260903_r1/"
    "commands/resolved_wave1_g0_G0_seed1.json"
)
EXPECTED_SOURCE_SHA256 = (
    "41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030"
)
EXPECTED_MODEL_SHA256 = (
    "09854d261ea2fdfb6f79aeaf971c660f7d2b38cad7b4172ca41c4e10aa648148"
)
ARMS = tuple(f"W{counterfactual}{critical_path}{sequential}"
             for counterfactual in (0, 1)
             for critical_path in (0, 1)
             for sequential in (0, 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"Checkpoint has no model state: {path}")
    digest_parts = []
    for name, tensor in sorted(model.items()):
        if not torch.is_tensor(tensor):
            raise TypeError(f"Non-tensor model entry: {name}")
        tensor = tensor.detach().cpu().contiguous()
        header = (
            f"{tensor.dtype}|{tuple(int(dim) for dim in tensor.shape)}|"
        ).encode("utf-8")
        tensor_digest = hashlib.sha256(
            header + tensor.numpy().tobytes()
        ).hexdigest()
        digest_parts.append(f"{name}\0{tensor_digest}\0")
    return hashlib.sha256(
        "".join(digest_parts).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def option_value(command: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1:
        raise ValueError(f"Expected one {flag}; found {len(positions)}.")
    index = positions[0]
    if index + 1 >= len(command) or str(command[index + 1]).startswith("--"):
        raise ValueError(f"Missing value for {flag}.")
    return str(command[index + 1])


def set_option(command: list[str], flag: str, value: object) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) > 1:
        raise ValueError(f"Duplicate option in base command: {flag}")
    if positions:
        index = positions[0]
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ValueError(f"Base option has no value: {flag}")
        command[index + 1] = str(value)
    else:
        command.extend((flag, str(value)))


def remove_option(command: list[str], flag: str) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index]
        if index < len(command) and not command[index].startswith("--"):
            del command[index]


def set_switch(command: list[str], flag: str, enabled: bool) -> None:
    command[:] = [item for item in command if item != flag]
    if enabled:
        command.append(flag)


def arm_factors(arm: str) -> dict[str, bool]:
    if arm not in ARMS:
        raise ValueError(f"Unknown credit-factorial arm: {arm}")
    return {
        "counterfactual_q": arm[1] == "1",
        "critical_path_v2": arm[2] == "1",
        "role_sequential_ppo": arm[3] == "1",
    }


def build_command(
    base_command: Sequence[str], *, phase: str, arm: str,
    experiment_name: str, source_checkpoint: Path,
    python: Path = DEFAULT_PYTHON,
) -> list[str]:
    """Mutate the already-proven legacy-T2 handoff, then fail closed."""
    if phase not in {"canary", "wave1"}:
        raise ValueError(f"Unsupported phase: {phase}")
    factors = arm_factors(arm)
    command = [str(value) for value in base_command]
    if len(command) < 2:
        raise ValueError("Base command is empty.")
    command[0] = str(python.resolve())
    command[1] = str(
        (ROOT / "onpolicy/scripts/train/train_hkbz.py").resolve()
    )
    canary = phase == "canary"
    episodes = 1 if canary else 6
    valued = {
        "--experiment_name": experiment_name,
        "--checkpoint_dir": source_checkpoint.resolve(),
        "--training_stage": "auto",
        "--resource_policy": "drl",
        "--seed": 3,
        "--cuda_memory_fraction": 0.80,
        "--n_training_threads": 1,
        "--n_rollout_threads": 8,
        "--n_eval_rollout_threads": 4,
        "--max_train_cases": 24 if canary else 0,
        "--max_eval_cases": 12 if canary else 60,
        "--train_sampling_size": 24 if canary else 240,
        "--train_sampling_pool_size": 0 if canary else 960,
        "--num_episodes": episodes,
        "--ppo_epoch": 1,
        "--mini_batch_size": 4,
        "--data_chunk_length": 50,
        "--max_graphs_per_forward": 200,
        "--grad_accumulation_steps": 5,
        "--actor_grad_accumulation_steps": 2,
        "--grad_accumulation_target_graphs": 5000,
        "--actor_grad_accumulation_target_graphs": 2000,
        "--actor_warmup_shards": 0 if canary else 60,
        "--gnn_freeze_epochs": episodes,
        "--plane_freeze_epochs": 0,
        "--plane_order_freeze_epochs": 0,
        "--shared_actor_lr_scale": 0.0,
        "--shared_actor_lr_scale_schedule": (
            "0.0" if canary else "0.0,0.0,0.0,0.0,0.0,0.0"
        ),
        "--lr": "5e-6",
        "--critic_lr": "1e-4",
        "--plane_actor_lr_scale": 0.25,
        "--device_actor_lr_scale": 1.0,
        "--transporter_actor_lr_scale": 0.5,
        "--clip_param": 0.05,
        "--target_kl": 0.005,
        "--plane_target_kl": 0.0025,
        "--device_target_kl": 0.005,
        "--transporter_target_kl": 0.005,
        "--max_grad_norm": 1.0,
        "--actor_grad_clip_mode": "per_group",
        "--shared_actor_max_grad_norm": 1.0,
        "--plane_actor_max_grad_norm": 1.0,
        "--device_actor_max_grad_norm": 1.0,
        "--transporter_actor_max_grad_norm": 1.0,
        "--safe_async_graph_clone_workers": 2,
        "--role_event_gae_lambda": 1.0,
        "--plane_role_gae_lambda": 1.0,
        "--device_role_gae_lambda": 1.0,
        "--transporter_role_gae_lambda": 1.0,
        "--role_loss_weighting": "fixed",
        "--plane_loss_coef": 1.0,
        "--device_loss_coef": 1.0,
        "--transporter_loss_coef": 1.0,
        "--shared_gradient_method": "sum",
        "--role_event_credit_mode": (
            "critical_path_v2" if factors["critical_path_v2"] else "elapsed"
        ),
        "--role_event_credit_uniform_mix": 0.15,
        "--counterfactual_q_topk": 8,
        "--counterfactual_q_min_mass": 0.90,
        "--counterfactual_baseline_mix": 1.0,
        "--counterfactual_baseline_mix_schedule": (
            "1.0" if canary else "0.0,0.0,0.25,0.5,0.75,1.0"
        ),
        "--hindsight_reward_mode": "team_time",
        "--hindsight_cmax_coef": 0.0,
        "--hindsight_shaping_coef": 0.0,
        "--hindsight_terminal_cmax_coef": 1.0,
        "--resource_lateness_coef": 0.0,
        "--resource_critical_lateness_coef": 0.0,
        "--resource_earliness_coef": 0.0,
        "--resource_wait_constraint_target": 0.0,
        "--resource_wait_dual_lr": 0.0,
        "--iga_potential_beta": 0.0,
        "--iga_potential_beta_schedule": "0.0",
        "--device_future_intent_horizon": 3,
        "--device_frontier_max_requests": 4,
        "--device_request_capacity_per_plane": 5,
        "--device_lookahead_reservation_mode": "hard",
        # The selected legacy T2 checkpoint is admitted by the already-audited
        # ppo_gain_wave transition; the scientific arm identity lives in this
        # immutable manifest rather than widening the checkpoint contract.
        "--stage3_handoff_mode": "ppo_gain_wave",
        "--selection_metric": "raw",
        "--canary_eval_interval_shards": 0,
        "--canary_eval_max_per_epoch": 0,
        "--canary_eval_max_cases": 0,
        "--early_stop_patience": 0,
        "--save_interval": 1,
        "--status_heartbeat_seconds": 30,
        "--recovery_checkpoint_interval_shards": 1,
    }
    for flag, value in valued.items():
        set_option(command, flag, value)

    for flag in (
        "--rollout_until_done", "--use_valuenorm", "--use_eval",
        "--strict_checkpoint_contract", "--adaptive_actor_kl",
        "--joint_team_ppo", "--role_atomic_ppo", "--role_event_returns",
        "--role_valuenorm", "--device_lookahead_dispatch",
        "--device_deadline_aware_dispatch", "--resource_release_aware_eta",
        "--device_departure_lookahead", "--safe_graph_batch_pipeline",
        "--clear_cuda_cache_after_update", "--resume_stage1",
        "--reset_optimizers_on_resume", "--stage3_allow_shared_frozen",
    ):
        set_switch(command, flag, True)
    set_switch(
        command, "--counterfactual_q_baseline",
        factors["counterfactual_q"],
    )
    set_switch(
        command, "--role_sequential_ppo",
        factors["role_sequential_ppo"],
    )
    for flag in (
        "--shared_encoder_pcgrad", "--actor_kl_backtrack",
        "--request_ready_prediction", "--bc_reference_hard_gate",
        "--skip_pre_ppo_eval", "--canary_stop_on_regression",
    ):
        set_switch(command, flag, False)
    for flag in (
        "--joint_iga_teacher_dir", "--joint_iga_teacher_index",
        "--resource_iga_teacher_dir", "--resource_iga_teacher_index",
        "--plane_bc_teacher_dir", "--resource_bc_checkpoint",
    ):
        remove_option(command, flag)
    validate_command(command, phase=phase, arm=arm)
    return command


def validate_command(command: Sequence[str], *, phase: str, arm: str) -> None:
    command = [str(value) for value in command]
    factors = arm_factors(arm)
    episodes = 1 if phase == "canary" else 6
    exact = {
        "--training_stage": "auto",
        "--resource_policy": "drl",
        "--seed": "3",
        "--cuda_memory_fraction": "0.8",
        "--n_training_threads": "1",
        "--n_rollout_threads": "8",
        "--n_eval_rollout_threads": "4",
        "--num_episodes": str(episodes),
        "--max_train_cases": "24" if phase == "canary" else "0",
        "--max_eval_cases": "12" if phase == "canary" else "60",
        "--train_sampling_size": "24" if phase == "canary" else "240",
        "--train_sampling_pool_size": "0" if phase == "canary" else "960",
        "--max_graphs_per_forward": "200",
        "--mini_batch_size": "4",
        "--data_chunk_length": "50",
        "--safe_async_graph_clone_workers": "2",
        "--actor_warmup_shards": "0" if phase == "canary" else "60",
        "--gnn_freeze_epochs": str(episodes),
        "--shared_actor_lr_scale": "0.0",
        "--actor_grad_clip_mode": "per_group",
        "--role_event_credit_mode": (
            "critical_path_v2" if factors["critical_path_v2"] else "elapsed"
        ),
        "--hindsight_reward_mode": "team_time",
        "--resource_lateness_coef": "0.0",
        "--resource_critical_lateness_coef": "0.0",
        "--resource_earliness_coef": "0.0",
        "--resource_wait_constraint_target": "0.0",
        "--resource_wait_dual_lr": "0.0",
        "--iga_potential_beta": "0.0",
        "--device_future_intent_horizon": "3",
        "--device_frontier_max_requests": "4",
        "--device_lookahead_reservation_mode": "hard",
        "--joint_team_ppo_scope": "all",
        "--role_loss_weighting": "fixed",
    }
    for flag, expected in exact.items():
        actual = option_value(command, flag)
        if actual != expected:
            raise ValueError(f"{flag}={actual}; expected {expected}.")
    for flag in (
        "--plane_bc_pretrain_epochs", "--device_bc_pretrain_epochs",
    ):
        if option_value(command, flag) != "0":
            raise ValueError(f"Supervised update is enabled by {flag}.")
    for flag in (
        "--plane_bc_dagger_schedule",
        "--plane_bc_staging_dagger_schedule",
        "--device_bc_dagger_schedule", "--bc_reference_kl_coef",
        "--bc_reference_kl_coef_schedule", "--bc_reference_target_kl",
    ):
        if float(option_value(command, flag)) != 0.0:
            raise ValueError(f"Stage3 command is not pure RL: {flag}")
    required = (
        "--rollout_until_done", "--use_eval", "--strict_checkpoint_contract",
        "--joint_team_ppo", "--role_atomic_ppo", "--role_event_returns",
        "--role_valuenorm", "--resume_stage1",
        "--reset_optimizers_on_resume", "--stage3_allow_shared_frozen",
    )
    for flag in required:
        if command.count(flag) != 1:
            raise ValueError(f"Expected exactly one required switch {flag}.")
    switches = {
        "--counterfactual_q_baseline": factors["counterfactual_q"],
        "--role_sequential_ppo": factors["role_sequential_ppo"],
    }
    for flag, expected in switches.items():
        if (flag in command) != expected:
            raise ValueError(f"Factor switch mismatch: {flag}")
    if "--shared_encoder_pcgrad" in command:
        raise ValueError("Wave 1 must not mix shared-gradient surgery.")
    schedule = [
        float(value) for value in option_value(
            command, "--shared_actor_lr_scale_schedule"
        ).split(",")
    ]
    if len(schedule) != episodes or any(value != 0.0 for value in schedule):
        raise ValueError("Shared encoder is not frozen for the complete arm.")
    cf_schedule = [
        float(value) for value in option_value(
            command, "--counterfactual_baseline_mix_schedule"
        ).split(",")
    ]
    expected_cf = [1.0] if phase == "canary" else [
        0.0, 0.0, 0.25, 0.5, 0.75, 1.0
    ]
    if cf_schedule != expected_cf:
        raise ValueError("Counterfactual calibration schedule changed.")


def validate_manifest(manifest: Mapping[str, object]) -> None:
    if int(manifest.get("schema_version", 0)) != 3:
        raise ValueError("Unsupported credit-factorial manifest schema.")
    phase = str(manifest.get("phase"))
    arm = str(manifest.get("arm"))
    if phase not in {"canary", "wave1"} or arm not in ARMS:
        raise ValueError("Invalid phase or arm.")
    if dict(manifest.get("factors", {})) != arm_factors(arm):
        raise ValueError("Arm label and factors disagree.")
    source = manifest.get("source_stage2")
    if not isinstance(source, Mapping):
        raise ValueError("Missing Stage2 source record.")
    path = Path(str(source.get("path", ""))).resolve()
    if (
        not path.is_file()
        or sha256_file(path) != source.get("sha256")
        or model_sha256(path) != source.get("model_sha256")
        or source.get("sha256") != EXPECTED_SOURCE_SHA256
        or source.get("model_sha256") != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("Best Stage2 checkpoint identity changed.")
    command = manifest.get("command")
    if not isinstance(command, list):
        raise ValueError("Manifest command must be argv.")
    if Path(str(command[0])).resolve() != Path(str(manifest["python"])).resolve():
        raise ValueError("Manifest Python and argv disagree.")
    if Path(option_value(command, "--checkpoint_dir")).resolve() != path:
        raise ValueError("Manifest command uses another checkpoint.")
    if option_value(command, "--experiment_name") != manifest.get(
        "experiment_name"
    ):
        raise ValueError("Experiment name mismatch.")
    validate_command(command, phase=phase, arm=arm)
    if phase == "wave1":
        gate = manifest.get("canary_gate")
        if not isinstance(gate, Mapping) or gate.get("status") != "passed_8_of_8":
            raise ValueError("Formal Wave 1 requires a passed 8/8 canary gate.")


def git_snapshot() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args, cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()
    try:
        return {
            "git_head": run("git", "rev-parse", "HEAD"),
            "worktree_dirty": bool(run("git", "status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": "unknown", "worktree_dirty": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("canary", "wave1"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--cpu-affinity", required=True)
    parser.add_argument("--numa-node", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--canary-gate", type=Path)
    args = parser.parse_args()

    source = args.source_checkpoint.expanduser().resolve()
    base_path = args.base_manifest.expanduser().resolve()
    python = args.python.expanduser().resolve()
    if not source.is_file() or not base_path.is_file() or not python.is_file():
        raise FileNotFoundError("Source checkpoint, base manifest, or Python missing.")
    source_sha = sha256_file(source)
    source_model_sha = model_sha256(source)
    if (
        source_sha != EXPECTED_SOURCE_SHA256
        or source_model_sha != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("Requested source is not the selected best Stage2 checkpoint.")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("source_stage2", {}).get("sha256") != source_sha:
        raise ValueError("Proven base manifest used another Stage2 source.")
    experiment_name = f"{args.run_tag}_{args.phase}_{args.arm}_seed3"
    command = build_command(
        base["command"], phase=args.phase, arm=args.arm,
        experiment_name=experiment_name, source_checkpoint=source,
        python=python,
    )
    canary_gate = None
    if args.phase == "wave1":
        if args.canary_gate is None:
            raise ValueError("--canary-gate is required for formal Wave 1.")
        gate_path = args.canary_gate.expanduser().resolve()
        canary_gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            canary_gate.get("status") != "passed_8_of_8"
            or canary_gate.get("source_stage2_sha256") != source_sha
        ):
            raise ValueError("Canary gate is absent, failed, or uses another source.")
        canary_gate = {**canary_gate, "path": str(gate_path)}
    manifest = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "arm": args.arm,
        "factors": arm_factors(args.arm),
        "seed": 3,
        "physical_gpu": int(args.physical_gpu),
        "cpu_affinity": str(args.cpu_affinity),
        "numa_node": int(args.numa_node),
        "run_tag": args.run_tag,
        "experiment_name": experiment_name,
        "expected_result_parent": str((RESULTS_ROOT / experiment_name).resolve()),
        "python": str(python),
        "source_stage2": {
            "path": str(source),
            "sha256": source_sha,
            "model_sha256": source_model_sha,
        },
        "derived_from": {
            "path": str(base_path),
            "sha256": sha256_file(base_path),
        },
        "canary_gate": canary_gate,
        "code_snapshot": git_snapshot(),
        "contract_checks": {
            "single_gpu_local_train_and_eval": True,
            "cpu_affinity_isolated": True,
            "restart_policy": "no",
            "hard_timeout_hours": 12 if args.phase == "canary" else 96,
            "shared_encoder_frozen_all_epochs": True,
            "actor_grad_clip_mode": "per_group",
            "train_cases_per_epoch": 24 if args.phase == "canary" else 240,
            "rotating_pool": 0 if args.phase == "canary" else 960,
            "valid_cases_per_epoch": 12 if args.phase == "canary" else 60,
            "graph_cap": 200,
        },
        "command": command,
        "shell": shlex.join(command),
    }
    validate_manifest(manifest)
    atomic_json(args.output.expanduser().resolve(), manifest)
    print(f"[CreditFactorial] wrote {args.output} ({args.arm}/{args.phase})")


if __name__ == "__main__":
    main()
