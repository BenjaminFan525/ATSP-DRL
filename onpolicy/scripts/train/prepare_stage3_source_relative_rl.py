#!/usr/bin/env python3
"""Emit one immutable arm of the Stage3 source-relative RL factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onpolicy.scripts.train.prepare_stage3_credit_factorial import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_SOURCE_SHA256,
    ROOT,
    RESULTS_ROOT,
    atomic_json,
    git_snapshot,
    model_sha256,
    option_value,
    remove_option,
    set_option,
    set_switch,
    sha256_file,
)


DEFAULT_PYTHON = (
    ROOT.parent / "conda/envs/maia-hkbz-cu124-20260903/bin/python3.11"
)
DEFAULT_SOURCE = RESULTS_ROOT / (
    "stage2_adaptive_trust_20260830_r4_wave1_"
    "T2_backtrack_adaptive_soft_bc_seed1/run1/models/checkpoint_Epoch3.pt"
)
DEFAULT_BASE_MANIFEST = ROOT / (
    "result/hkbz_train_logs/stage3_credit_factorial_4090_20260903_r2/"
    "commands/wave1_g3_W001.json"
)
DEFAULT_TRAIN_DATASET = ROOT / (
    "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
)
EXPECTED_TRAIN_CASES = 600
BASELINE_REFERENCE_KIND = "source_checkpoint_deterministic_cmax"
BASELINE_SCHEMA_VERSION = 1

ARMS = tuple(
    f"U{unfreeze}K{reference_kl}R{paired_residual}"
    for unfreeze in (0, 1)
    for reference_kl in (0, 1)
    for paired_residual in (0, 1)
)

SOURCE_EVAL_CONTRACT = {
    "evaluation_tau": 0.3,
    "global_feature_mode": "f1f2",
    "resource_policy": "drl",
    "plane_order_mode": "fixed",
    "plane_pair_decoder": "joint_pair",
    "device_future_intent_horizon": 3,
    "device_future_intent_mode": "bounded_frontier",
    "device_frontier_max_requests": 4,
    "device_request_capacity_per_plane": 5,
    "device_lookahead_reservation_mode": "hard",
    "device_lookahead_safety_margin": 60.0,
    "device_reservation_grace_seconds": 300.0,
    "device_lookahead_dispatch": True,
    "device_deadline_aware_dispatch": True,
    "resource_release_aware_eta": True,
    "device_departure_lookahead": True,
}

IMPLEMENTATION_PATHS = (
    "onpolicy/config/config.py",
    "onpolicy/scripts/train/train_hkbz.py",
    "onpolicy/runner/shared/hkbz_runner.py",
    "onpolicy/algorithms/gnn_mappo/gnn_mappo.py",
    "onpolicy/algorithms/gnn_mappo/algorithm/MAPPOPolicy.py",
    "onpolicy/utils/shared_buffer.py",
    "onpolicy/scripts/train/prepare_stage3_source_relative_rl.py",
    "onpolicy/scripts/train/run_stage3_source_relative_manifest.py",
)


def arm_factors(arm: str) -> dict[str, bool]:
    if arm not in ARMS:
        raise ValueError(f"Unknown source-relative arm: {arm}")
    return {
        "shared_encoder_unfreeze": arm[1] == "1",
        "source_reference_kl": arm[3] == "1",
        "source_paired_residual": arm[5] == "1",
    }


def implementation_snapshot() -> dict[str, object]:
    files = {}
    for relative in IMPLEMENTATION_PATHS:
        path = (ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = sha256_file(path)
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _metadata_case_sha256(case_dir: Path) -> str:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    fingerprints = payload.get("fingerprints", {})
    value = str(
        payload.get("case_sha256")
        or (fingerprints.get("case_sha256") if isinstance(fingerprints, Mapping) else "")
        or ""
    ).strip()
    if len(value) != 64:
        raise ValueError(f"Missing immutable case SHA-256: {metadata_path}")
    return value


def dataset_contract(dataset_dir: Path) -> dict[str, object]:
    dataset = dataset_dir.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    case_dirs = sorted(
        path for path in dataset.glob("case_*") if path.is_dir()
    )
    if not case_dirs:
        raise ValueError(f"No cases found in {dataset}.")
    cases = {
        path.name: _metadata_case_sha256(path)
        for path in case_dirs
    }
    canonical = json.dumps(
        cases, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "path": str(dataset),
        "case_count": len(cases),
        "fingerprint_sha256": hashlib.sha256(canonical).hexdigest(),
        "cases": cases,
    }


def validate_source_baseline(
    baseline_path: Path,
    *,
    source_checkpoint: Path = DEFAULT_SOURCE,
    dataset_dir: Path = DEFAULT_TRAIN_DATASET,
) -> dict[str, object]:
    path = baseline_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != BASELINE_SCHEMA_VERSION:
        raise ValueError("Unsupported source-baseline schema.")
    if payload.get("reference_kind") != BASELINE_REFERENCE_KIND:
        raise ValueError("Paired baseline is not a source-checkpoint baseline.")

    source = source_checkpoint.expanduser().resolve()
    source_record = payload.get("source_checkpoint")
    if not isinstance(source_record, Mapping):
        raise ValueError("Source baseline has no checkpoint identity.")
    if (
        Path(str(source_record.get("path", ""))).resolve() != source
        or source_record.get("sha256") != EXPECTED_SOURCE_SHA256
        or source_record.get("model_sha256") != EXPECTED_MODEL_SHA256
        or sha256_file(source) != EXPECTED_SOURCE_SHA256
        or model_sha256(source) != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("Source baseline checkpoint identity changed.")
    if dict(payload.get("policy_environment_contract", {})) != SOURCE_EVAL_CONTRACT:
        raise ValueError("Source baseline was evaluated under another policy contract.")

    expected_dataset = dataset_contract(dataset_dir)
    dataset_record = payload.get("dataset")
    if not isinstance(dataset_record, Mapping):
        raise ValueError("Source baseline has no dataset identity.")
    for field in ("path", "case_count", "fingerprint_sha256"):
        if dataset_record.get(field) != expected_dataset[field]:
            raise ValueError(f"Source baseline dataset {field} changed.")
    if int(expected_dataset["case_count"]) != EXPECTED_TRAIN_CASES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_CASES} source cases; found "
            f"{expected_dataset['case_count']}."
        )

    cases = payload.get("cases")
    if not isinstance(cases, Mapping):
        raise ValueError("Source baseline has no case map.")
    expected_cases = expected_dataset["cases"]
    if set(cases) != set(expected_cases):
        missing = sorted(set(expected_cases) - set(cases))[:8]
        extra = sorted(set(cases) - set(expected_cases))[:8]
        raise ValueError(
            f"Source baseline coverage mismatch: missing={missing}, extra={extra}."
        )
    makespans = []
    for case_name, case_sha in expected_cases.items():
        record = cases[case_name]
        if not isinstance(record, Mapping):
            raise ValueError(f"Non-record source baseline for {case_name}.")
        if str(record.get("case_sha256", "")) != case_sha:
            raise ValueError(f"Source baseline lineage mismatch for {case_name}.")
        makespan = float(record.get("makespan", float("nan")))
        if not math.isfinite(makespan) or makespan <= 0.0:
            raise ValueError(f"Invalid source makespan for {case_name}: {makespan}.")
        makespans.append(makespan)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "case_count": len(cases),
        "dataset_fingerprint_sha256": expected_dataset["fingerprint_sha256"],
        "mean_makespan": sum(makespans) / len(makespans),
    }


def _source_unfreeze_schedule(phase: str, enabled: bool) -> tuple[int, str]:
    if phase == "canary":
        # Exercise the worst-memory shared update in the canary itself.
        return (0, "0.05") if enabled else (1, "0.0")
    if enabled:
        return 1, "0.0,0.01,0.025,0.05,0.025,0.01"
    return 6, "0.0,0.0,0.0,0.0,0.0,0.0"


def build_command(
    base_command: Sequence[str],
    *,
    phase: str,
    arm: str,
    experiment_name: str,
    source_checkpoint: Path,
    source_baseline: Path,
    python: Path = DEFAULT_PYTHON,
) -> list[str]:
    """Build one controlled U x K x R arm from the proven Stage3 command."""
    if phase not in {"canary", "wave1"}:
        raise ValueError(f"Unsupported phase: {phase}")
    factors = arm_factors(arm)
    command = [str(value) for value in base_command]
    if len(command) < 2:
        raise ValueError("Base command is empty.")
    command[0] = str(python.expanduser().resolve())
    command[1] = str(
        (ROOT / "onpolicy/scripts/train/train_hkbz.py").resolve()
    )
    canary = phase == "canary"
    episodes = 1 if canary else 6
    freeze_epochs, shared_schedule = _source_unfreeze_schedule(
        phase, factors["shared_encoder_unfreeze"]
    )
    reference_enabled = factors["source_reference_kl"]
    residual_enabled = factors["source_paired_residual"]
    valued = {
        "--experiment_name": experiment_name,
        "--checkpoint_dir": source_checkpoint.expanduser().resolve(),
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
        "--actor_warmup_shards": 0 if canary else 30,
        "--gnn_freeze_epochs": freeze_epochs,
        "--plane_freeze_epochs": 0,
        "--plane_order_freeze_epochs": 0,
        "--plane_order_mode": "fixed",
        "--plane_pair_decoder": "joint_pair",
        "--global_feature_mode": "f1f2",
        "--shared_actor_lr_scale": 0.0,
        "--shared_actor_lr_scale_schedule": shared_schedule,
        "--lr": "5e-6",
        "--critic_lr": "1e-4",
        "--plane_actor_lr_scale": 0.25,
        "--device_actor_lr_scale": 1.0,
        "--transporter_actor_lr_scale": 0.5,
        "--anneal_original": 0.3,
        "--anneal_final": 0.3,
        "--tau_anneal_epochs": 0,
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
        "--role_event_credit_mode": "elapsed",
        "--role_event_credit_uniform_mix": 0.15,
        "--counterfactual_baseline_mix": 0.0,
        "--counterfactual_baseline_mix_schedule": (
            "0.0" if canary else "0.0,0.0,0.0,0.0,0.0,0.0"
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
        "--device_future_intent_mode": "bounded_frontier",
        "--device_frontier_max_requests": 4,
        "--device_request_capacity_per_plane": 5,
        "--device_lookahead_reservation_mode": "hard",
        "--device_lookahead_safety_margin": 60.0,
        "--device_reservation_grace_seconds": 300.0,
        "--stage3_handoff_mode": "ppo_gain_wave",
        "--selection_metric": "raw",
        "--evaluation_tau": 0.3,
        "--canary_eval_interval_shards": 0,
        "--canary_eval_max_per_epoch": 0,
        "--canary_eval_max_cases": 0,
        "--early_stop_patience": 0,
        "--save_interval": 1,
        "--status_heartbeat_seconds": 30,
        "--recovery_checkpoint_interval_shards": 1,
        "--bc_reference_kl_coef": 0.1 if reference_enabled else 0.0,
        "--bc_reference_kl_coef_schedule": (
            "0.1" if reference_enabled else "0.0"
        ),
        "--bc_reference_target_kl": 0.03 if reference_enabled else 0.0,
        "--adaptive_bc_reference_target_kl": 0.03,
        "--adaptive_bc_reference_coef_min": 0.02,
        "--adaptive_bc_reference_coef_max": 1.0,
        "--adaptive_bc_reference_coef_up": 1.5,
        "--adaptive_bc_reference_coef_down": 0.8,
        "--paired_case_baseline_coef": 1.0 if residual_enabled else 0.0,
        "--paired_case_baseline_scope": "actor",
        "--cvar_case_metric": "paired_delta" if residual_enabled else "cmax",
        "--cvar_policy_fraction": 0.25 if residual_enabled else 1.0,
        "--cvar_policy_weight": 2.0 if residual_enabled else 1.0,
    }
    for flag, value in valued.items():
        set_option(command, flag, value)

    if reference_enabled:
        set_option(
            command,
            "--bc_reference_checkpoint",
            source_checkpoint.expanduser().resolve(),
        )
    else:
        remove_option(command, "--bc_reference_checkpoint")
    if residual_enabled:
        set_option(
            command,
            "--paired_case_baseline_dir",
            source_baseline.expanduser().resolve(),
        )
    else:
        remove_option(command, "--paired_case_baseline_dir")

    for flag in (
        "--rollout_until_done",
        "--use_valuenorm",
        "--use_eval",
        "--strict_checkpoint_contract",
        "--adaptive_actor_kl",
        "--joint_team_ppo",
        "--role_atomic_ppo",
        "--role_event_returns",
        "--role_valuenorm",
        "--role_sequential_ppo",
        "--device_lookahead_dispatch",
        "--device_deadline_aware_dispatch",
        "--resource_release_aware_eta",
        "--device_departure_lookahead",
        "--safe_graph_batch_pipeline",
        "--clear_cuda_cache_after_update",
        "--resume_stage1",
        "--reset_optimizers_on_resume",
        "--stage3_allow_shared_frozen",
        "--shared_gradient_diagnostics",
        "--shared_encoder_activation_checkpoint",
    ):
        set_switch(command, flag, True)
    set_switch(command, "--adaptive_bc_reference_kl", reference_enabled)
    for flag in (
        "--counterfactual_q_baseline",
        "--shared_encoder_pcgrad",
        "--actor_kl_backtrack",
        "--bc_reference_hard_gate",
        "--request_ready_prediction",
        "--reset_value_normalizer_before_ppo",
        "--skip_pre_ppo_eval",
        "--canary_stop_on_regression",
    ):
        set_switch(command, flag, False)
    for flag in (
        "--joint_iga_teacher_dir",
        "--joint_iga_teacher_index",
        "--resource_iga_teacher_dir",
        "--resource_iga_teacher_index",
        "--plane_bc_teacher_dir",
        "--resource_bc_checkpoint",
    ):
        remove_option(command, flag)
    validate_command(
        command,
        phase=phase,
        arm=arm,
        source_checkpoint=source_checkpoint,
        source_baseline=source_baseline,
    )
    return command


def validate_command(
    command: Sequence[str],
    *,
    phase: str,
    arm: str,
    source_checkpoint: Path = DEFAULT_SOURCE,
    source_baseline: Path = DEFAULT_TRAIN_DATASET,
) -> None:
    command = [str(value) for value in command]
    factors = arm_factors(arm)
    canary = phase == "canary"
    episodes = 1 if canary else 6
    freeze_epochs, shared_schedule = _source_unfreeze_schedule(
        phase, factors["shared_encoder_unfreeze"]
    )
    exact = {
        "--training_stage": "auto",
        "--resource_policy": "drl",
        "--seed": "3",
        "--cuda_memory_fraction": "0.8",
        "--n_training_threads": "1",
        "--n_rollout_threads": "8",
        "--n_eval_rollout_threads": "4",
        "--num_episodes": str(episodes),
        "--max_train_cases": "24" if canary else "0",
        "--max_eval_cases": "12" if canary else "60",
        "--train_sampling_size": "24" if canary else "240",
        "--train_sampling_pool_size": "0" if canary else "960",
        "--max_graphs_per_forward": "200",
        "--mini_batch_size": "4",
        "--data_chunk_length": "50",
        "--safe_async_graph_clone_workers": "2",
        "--actor_warmup_shards": "0" if canary else "30",
        "--gnn_freeze_epochs": str(freeze_epochs),
        "--shared_actor_lr_scale": "0.0",
        "--shared_actor_lr_scale_schedule": shared_schedule,
        "--actor_grad_clip_mode": "per_group",
        "--role_event_credit_mode": "elapsed",
        "--plane_order_mode": "fixed",
        "--plane_pair_decoder": "joint_pair",
        "--global_feature_mode": "f1f2",
        "--hindsight_reward_mode": "team_time",
        "--resource_lateness_coef": "0.0",
        "--resource_critical_lateness_coef": "0.0",
        "--resource_earliness_coef": "0.0",
        "--resource_wait_constraint_target": "0.0",
        "--resource_wait_dual_lr": "0.0",
        "--iga_potential_beta": "0.0",
        "--device_future_intent_horizon": "3",
        "--device_future_intent_mode": "bounded_frontier",
        "--device_frontier_max_requests": "4",
        "--device_request_capacity_per_plane": "5",
        "--device_lookahead_reservation_mode": "hard",
        "--device_lookahead_safety_margin": "60.0",
        "--device_reservation_grace_seconds": "300.0",
        "--joint_team_ppo_scope": "all",
        "--role_loss_weighting": "fixed",
        "--shared_gradient_method": "sum",
        "--anneal_original": "0.3",
        "--anneal_final": "0.3",
        "--tau_anneal_epochs": "0",
        "--evaluation_tau": "0.3",
        "--paired_case_baseline_scope": "actor",
        "--bc_reference_kl_coef": (
            "0.1" if factors["source_reference_kl"] else "0.0"
        ),
        "--bc_reference_target_kl": (
            "0.03" if factors["source_reference_kl"] else "0.0"
        ),
        "--paired_case_baseline_coef": (
            "1.0" if factors["source_paired_residual"] else "0.0"
        ),
        "--cvar_case_metric": (
            "paired_delta" if factors["source_paired_residual"] else "cmax"
        ),
        "--cvar_policy_fraction": (
            "0.25" if factors["source_paired_residual"] else "1.0"
        ),
        "--cvar_policy_weight": (
            "2.0" if factors["source_paired_residual"] else "1.0"
        ),
    }
    for flag, expected in exact.items():
        actual = option_value(command, flag)
        if actual != expected:
            raise ValueError(f"{flag}={actual}; expected {expected}.")
    if Path(option_value(command, "--checkpoint_dir")).resolve() != Path(
        source_checkpoint
    ).resolve():
        raise ValueError("Command does not start from the selected C0 checkpoint.")
    for flag in ("--plane_bc_pretrain_epochs", "--device_bc_pretrain_epochs"):
        if option_value(command, flag) != "0":
            raise ValueError(f"Supervised update is enabled by {flag}.")
    for flag in (
        "--plane_bc_dagger_schedule",
        "--plane_bc_staging_dagger_schedule",
        "--device_bc_dagger_schedule",
    ):
        if float(option_value(command, flag)) != 0.0:
            raise ValueError(f"Stage3 command consumes teacher actions via {flag}.")
    required = (
        "--rollout_until_done",
        "--use_eval",
        "--strict_checkpoint_contract",
        "--joint_team_ppo",
        "--role_atomic_ppo",
        "--role_event_returns",
        "--role_valuenorm",
        "--role_sequential_ppo",
        "--resume_stage1",
        "--reset_optimizers_on_resume",
        "--stage3_allow_shared_frozen",
    )
    for flag in required:
        if command.count(flag) != 1:
            raise ValueError(f"Expected exactly one required switch {flag}.")
    for forbidden in (
        "--counterfactual_q_baseline",
        "--shared_encoder_pcgrad",
        "--actor_kl_backtrack",
        "--bc_reference_hard_gate",
        "--request_ready_prediction",
    ):
        if forbidden in command:
            raise ValueError(f"Forbidden confound is enabled: {forbidden}.")

    reference_enabled = factors["source_reference_kl"]
    if ("--adaptive_bc_reference_kl" in command) != reference_enabled:
        raise ValueError("Adaptive source-reference KL factor mismatch.")
    if reference_enabled:
        reference_path = Path(
            option_value(command, "--bc_reference_checkpoint")
        ).resolve()
        if reference_path != Path(source_checkpoint).resolve():
            raise ValueError("K1 reference is not the initial C0 checkpoint.")
    elif "--bc_reference_checkpoint" in command:
        raise ValueError("K0 unexpectedly carries a source reference.")

    residual_enabled = factors["source_paired_residual"]
    if residual_enabled:
        baseline_path = Path(
            option_value(command, "--paired_case_baseline_dir")
        ).resolve()
        if baseline_path != Path(source_baseline).resolve():
            raise ValueError("R1 does not use the frozen C0 case baseline.")
    elif "--paired_case_baseline_dir" in command:
        raise ValueError("R0 unexpectedly carries a paired baseline.")

    schedule = option_value(command, "--shared_actor_lr_scale_schedule")
    if schedule != shared_schedule:
        raise ValueError("Shared-encoder unfreeze schedule changed.")
    cf_schedule = [
        float(item) for item in option_value(
            command, "--counterfactual_baseline_mix_schedule"
        ).split(",")
    ]
    if len(cf_schedule) != episodes or any(value != 0.0 for value in cf_schedule):
        raise ValueError("Counterfactual baseline must remain disabled.")


def validate_manifest(manifest: Mapping[str, object]) -> None:
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported source-relative manifest schema.")
    phase = str(manifest.get("phase"))
    arm = str(manifest.get("arm"))
    if phase not in {"canary", "wave1"} or arm not in ARMS:
        raise ValueError("Invalid phase or source-relative arm.")
    if dict(manifest.get("factors", {})) != arm_factors(arm):
        raise ValueError("Arm label and factors disagree.")
    if dict(manifest.get("implementation_snapshot", {})) != implementation_snapshot():
        raise ValueError("Stage3 implementation changed after manifest creation.")
    source = manifest.get("source_stage2")
    if not isinstance(source, Mapping):
        raise ValueError("Missing Stage2 source record.")
    source_path = Path(str(source.get("path", ""))).resolve()
    if (
        not source_path.is_file()
        or source.get("sha256") != EXPECTED_SOURCE_SHA256
        or source.get("model_sha256") != EXPECTED_MODEL_SHA256
        or sha256_file(source_path) != EXPECTED_SOURCE_SHA256
        or model_sha256(source_path) != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("Best Stage2 checkpoint identity changed.")
    baseline_record = manifest.get("source_baseline")
    if not isinstance(baseline_record, Mapping):
        raise ValueError("Missing source baseline record.")
    baseline = validate_source_baseline(
        Path(str(baseline_record.get("path", ""))),
        source_checkpoint=source_path,
        dataset_dir=Path(str(manifest.get("train_dataset", ""))),
    )
    for field in (
        "path", "sha256", "case_count", "dataset_fingerprint_sha256"
    ):
        if baseline_record.get(field) != baseline[field]:
            raise ValueError(f"Manifest source baseline {field} changed.")
    command = manifest.get("command")
    if not isinstance(command, list):
        raise ValueError("Manifest command must be argv.")
    if Path(str(command[0])).resolve() != Path(str(manifest["python"])).resolve():
        raise ValueError("Manifest Python and argv disagree.")
    if option_value(command, "--experiment_name") != manifest.get(
        "experiment_name"
    ):
        raise ValueError("Experiment name mismatch.")
    validate_command(
        command,
        phase=phase,
        arm=arm,
        source_checkpoint=source_path,
        source_baseline=Path(str(baseline["path"])),
    )
    if phase == "wave1":
        gate = manifest.get("canary_gate")
        if not isinstance(gate, Mapping) or gate.get("status") != "passed_8_of_8":
            raise ValueError("Formal Wave 1 requires a passed 8/8 canary gate.")
        if (
            gate.get("source_stage2_sha256") != EXPECTED_SOURCE_SHA256
            or gate.get("source_baseline_sha256") != baseline["sha256"]
            or gate.get("implementation_bundle_sha256")
            != manifest["implementation_snapshot"]["bundle_sha256"]
        ):
            raise ValueError("Canary gate source provenance differs from Wave 1.")


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
    parser.add_argument("--source-baseline", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--canary-gate", type=Path)
    args = parser.parse_args()

    source = args.source_checkpoint.expanduser().resolve()
    baseline_path = args.source_baseline.expanduser().resolve()
    dataset = args.train_dataset.expanduser().resolve()
    base_path = args.base_manifest.expanduser().resolve()
    python = args.python.expanduser().resolve()
    if not source.is_file() or not base_path.is_file() or not python.is_file():
        raise FileNotFoundError("Source checkpoint, base manifest, or Python missing.")
    if (
        sha256_file(source) != EXPECTED_SOURCE_SHA256
        or model_sha256(source) != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("Requested source is not the selected best Stage2 checkpoint.")
    baseline = validate_source_baseline(
        baseline_path,
        source_checkpoint=source,
        dataset_dir=dataset,
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("source_stage2", {}).get("sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("Proven base manifest used another Stage2 source.")
    experiment_name = f"{args.run_tag}_{args.phase}_{args.arm}_seed3"
    command = build_command(
        base["command"],
        phase=args.phase,
        arm=args.arm,
        experiment_name=experiment_name,
        source_checkpoint=source,
        source_baseline=baseline_path,
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
            or canary_gate.get("source_stage2_sha256") != EXPECTED_SOURCE_SHA256
            or canary_gate.get("source_baseline_sha256") != baseline["sha256"]
        ):
            raise ValueError("Canary gate is absent, failed, or has another source.")
        canary_gate = {**canary_gate, "path": str(gate_path)}
    manifest = {
        "schema_version": 1,
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
        "train_dataset": str(dataset),
        "source_stage2": {
            "path": str(source),
            "sha256": EXPECTED_SOURCE_SHA256,
            "model_sha256": EXPECTED_MODEL_SHA256,
        },
        "source_baseline": baseline,
        "derived_from": {
            "path": str(base_path),
            "sha256": sha256_file(base_path),
        },
        "canary_gate": canary_gate,
        "code_snapshot": git_snapshot(),
        "implementation_snapshot": implementation_snapshot(),
        "contract_checks": {
            "single_gpu_local_train_and_eval": True,
            "cpu_affinity_isolated": True,
            "restart_policy": "no",
            "hard_timeout_hours": 12 if args.phase == "canary" else 96,
            "source_relative_factorial": True,
            "actor_warmup_shards": 0 if args.phase == "canary" else 30,
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
    print(f"[SourceRelative] wrote {args.output} ({args.arm}/{args.phase})")


if __name__ == "__main__":
    main()
