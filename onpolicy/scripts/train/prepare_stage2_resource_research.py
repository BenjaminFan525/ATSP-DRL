#!/usr/bin/env python3
"""Prepare strict Stage2 resource teachers and the paired Wave-1 manifest.

This script never reruns IGA.  It compacts the already replay-verified IGA-1800
teachers to the fields required by live decoding, while preserving a SHA256
chain back to every original trajectory and binding each label to the current
dataset case fingerprint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.scripts.train.run_hkbz_two_stage_pipeline import (  # noqa: E402
    _load_source_command,
    build_stage2_command,
    remove_option,
    set_option,
    set_switch,
)


DEFAULT_HANDOFF = ROOT / "onpolicy/config/stage1_m2_handoff.json"
DEFAULT_SOURCE_TEACHERS = (
    ROOT
    / "result/hkbz_train_logs/"
    "stage2_resource_iga_p5_seed3_case_parallel_20260818_r3/"
    "iga1800/teachers"
)
DEFAULT_DATASET = (
    ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
)
DEFAULT_PYTHON = (
    ROOT.parent / "conda/envs/maia/bin/python3.11"
).resolve()


METHODS = {
    "M0_heuristic_uniform": {
        "teacher": "heuristic",
        "role_balanced": False,
        "hypothesis": "current Hungarian live teacher and pooled BC loss",
    },
    "M1_iga_uniform": {
        "teacher": "iga",
        "role_balanced": False,
        "hypothesis": "IGA-1800 live teacher with pooled BC loss",
    },
    "M2_iga_role_balanced": {
        "teacher": "iga",
        "role_balanced": True,
        "hypothesis": "IGA-1800 live teacher and equal ordinary/R014 BC loss",
    },
}

LATENESS_METHODS = {
    "P0_cmax": {
        "teacher": "heuristic",
        "role_balanced": True,
        "resource_lateness_coef": 0.0,
        "resource_critical_lateness_coef": 0.0,
        "resource_earliness_coef": 0.0,
        "hypothesis": "deadline/future-intent semantics with pure team Cmax",
    },
    "P1_total_lateness": {
        "teacher": "heuristic",
        "role_balanced": True,
        "resource_lateness_coef": 0.02,
        "resource_critical_lateness_coef": 0.0,
        "resource_earliness_coef": 0.0,
        "hypothesis": "team Cmax plus total observed mobile-resource wait",
    },
    "P2_critical_jit": {
        "teacher": "heuristic",
        "role_balanced": True,
        "resource_lateness_coef": 0.0,
        "resource_critical_lateness_coef": 0.15,
        "resource_earliness_coef": 0.01,
        "hypothesis": "team Cmax plus worst-aircraft wait and anti-hoarding",
    },
}

# Wave 3 changes one mechanism at a time while retaining the exact Wave-2
# observation and deadline-aware dispatch contract.  This is what permits one
# persistent validator and one shared DeviceBC checkpoint across all five arms.
CRITICAL_METHODS = {
    "C0_cmax_control": {
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_cmax",
        "potential_beta": 0.0,
        "forecast_seconds": 0.0,
        "tail_start": 1.0,
        "tail_weight": 1.0,
        "hypothesis": "exact Wave-2 P0 team-Cmax control",
    },
    "C1_team_time": {
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time",
        "potential_beta": 0.0,
        "forecast_seconds": 0.0,
        "tail_start": 1.0,
        "tail_weight": 1.0,
        "hypothesis": "exact dense remaining-Cmax credit without a surrogate",
    },
    "C2_critical_slack": {
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time_resource_potential",
        "potential_beta": 0.10,
        "forecast_seconds": 0.0,
        "tail_start": 1.0,
        "tail_weight": 1.0,
        "hypothesis": "policy-invariant credit from per-request critical slack",
    },
    "C3_slack_arrival": {
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time_resource_potential",
        "potential_beta": 0.10,
        "forecast_seconds": 300.0,
        "tail_start": 1.0,
        "tail_weight": 1.0,
        "hypothesis": "critical slack plus imminent-arrival stand pressure",
    },
    "C4_slack_arrival_tail": {
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time_resource_potential",
        "potential_beta": 0.10,
        "forecast_seconds": 300.0,
        "tail_start": 0.70,
        "tail_weight": 2.0,
        "hypothesis": "arrival-aware slack with normalized late-decision emphasis",
    },
}

# Wave 4 first fixes what the policy can observe/commit, then changes how the
# same resource actor is trained.  GPU0 (A*) is the mechanism ladder.  GPU1
# (B*) holds the best soft-reservation semantics fixed and tests supervision,
# constrained wait credit, and one explicit low-rate shared-encoder arm.
PLANNING_METHODS = {
    "A0_legacy_c1": {
        "gpu_group": 0,
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": False,
        "future_intent_mode": "legacy_one",
        "reservation_mode": "none",
        "hypothesis": "exact Wave-3 C1 semantics and reward control",
    },
    "A1_release_eta": {
        "gpu_group": 0,
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "legacy_one",
        "reservation_mode": "none",
        "hypothesis": "correct service-release ETA and alternative resources",
    },
    "A2_frontier": {
        "gpu_group": 0,
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "bounded_frontier",
        "reservation_mode": "none",
        "hypothesis": "two-request dependency frontier without commitment",
    },
    "A3_soft_reservation": {
        "gpu_group": 0,
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "bounded_frontier",
        "reservation_mode": "soft",
        "hypothesis": "bounded frontier plus stealable pre-position leases",
    },
    "A4_hard_reservation": {
        "gpu_group": 0,
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "bounded_frontier",
        "reservation_mode": "hard",
        "hypothesis": "bounded frontier plus exclusive expiring leases",
    },
    "B0_soft_control": {
        "gpu_group": 1,
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "bounded_frontier",
        "reservation_mode": "soft",
        "hypothesis": "GPU-replicated A3 learning control",
    },
    "B1_iga_flow_bc": {
        "gpu_group": 1,
        "teacher": "iga",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "bounded_frontier",
        "reservation_mode": "soft",
        "hypothesis": "trajectory-consistent IGA flow supervision",
    },
    "B2_wait_constraint": {
        "gpu_group": 1,
        "teacher": "heuristic",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "bounded_frontier",
        "reservation_mode": "soft",
        "wait_constraint_target": 21000.0,
        "wait_dual_lr": 0.01,
        "critical_wait_coef": 0.05,
        "earliness_coef": 0.002,
        "hypothesis": "adaptive wait constraint with bounded anti-hoarding",
    },
    "B3_iga_constraint": {
        "gpu_group": 1,
        "teacher": "iga",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "bounded_frontier",
        "reservation_mode": "soft",
        "wait_constraint_target": 21000.0,
        "wait_dual_lr": 0.01,
        "critical_wait_coef": 0.05,
        "earliness_coef": 0.002,
        "hypothesis": "IGA flow BC plus adaptive wait constraint",
    },
    "B4_gradual_shared": {
        "gpu_group": 1,
        "teacher": "iga",
        "role_balanced": True,
        "reward_mode": "team_time",
        "release_aware": True,
        "future_intent_mode": "bounded_frontier",
        "reservation_mode": "soft",
        "wait_constraint_target": 21000.0,
        "wait_dual_lr": 0.01,
        "critical_wait_coef": 0.05,
        "earliness_coef": 0.002,
        "shared_unfreeze_epoch": 2,
        "hypothesis": "B3 plus low-rate unsplit shared-encoder adaptation",
    },
}

CANARY_TRAINING_STEPS = 64
CANARY_ROLLOUT_THREADS = 60
FORMAL_ROLLOUT_THREADS = 60
FORMAL_MINI_BATCH_SIZE = 30
CANARY_STANDARD_GRAPHS = 1500
CANARY_LOWMEM_GRAPHS = 800
FORMAL_STANDARD_GRAPHS = 1500
FORMAL_ACTOR_ACCUMULATION_TARGET_GRAPHS = 2000
FORMAL_CRITIC_ACCUMULATION_TARGET_GRAPHS = 5000
GPU_MEMORY_SAFETY_LIMIT_MIB = 73728

CRITICAL_TRAINING_STEPS = 64
CRITICAL_ROLLOUT_THREADS = 36
CRITICAL_EVAL_THREADS = 60
CRITICAL_TRAIN_CASES = 180
CRITICAL_MINI_BATCH_SIZE = 20
CRITICAL_GRAPHS_PER_FORWARD = 1000
CRITICAL_ACTOR_ACCUMULATION_TARGET_GRAPHS = 2000
CRITICAL_CRITIC_ACCUMULATION_TARGET_GRAPHS = 5000
CRITICAL_GPU_MEMORY_SAFETY_LIMIT_MIB = 77824

PLANNING_TRAINING_STEPS = 64
PLANNING_ROLLOUT_THREADS = 20
PLANNING_EVAL_THREADS = 10
PLANNING_TRAIN_CASES = 180
PLANNING_MINI_BATCH_SIZE = 20
PLANNING_GRAPHS_PER_FORWARD = 1000
PLANNING_ACTOR_ACCUMULATION_TARGET_GRAPHS = 2000
PLANNING_CRITIC_ACCUMULATION_TARGET_GRAPHS = 5000
PLANNING_GPU_MEMORY_SAFETY_LIMIT_MIB = 77824


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def case_fingerprint(metadata: dict) -> str:
    return str(
        metadata.get("case_sha256")
        or metadata.get("fingerprints", {}).get("case_sha256")
        or ""
    )


def build_compact_teacher_index(
    source_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    index_path: Path,
) -> dict:
    source_files = sorted(source_dir.glob("case_*.json"))
    case_dirs = sorted(path for path in dataset_dir.glob("case_*") if path.is_dir())
    if len(source_files) != 600 or len(case_dirs) != 600:
        raise ValueError(
            "Stage2 Wave1 requires exactly 600 source teachers and 600 train "
            f"cases; got {len(source_files)} and {len(case_dirs)}."
        )
    source_by_name = {path.stem: path for path in source_files}
    if set(source_by_name) != {path.name for path in case_dirs}:
        raise ValueError("Stage2 teacher and training-dataset case IDs differ.")

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    semantics = None
    checkpoint_sha = None
    source_command_sha = None
    cmax_values = []
    for case_dir in case_dirs:
        case_name = case_dir.name
        source_path = source_by_name[case_name]
        source_sha = sha256_file(source_path)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        metadata = json.loads(
            (case_dir / "metadata.json").read_text(encoding="utf-8")
        )
        case_sha = case_fingerprint(metadata)
        search = source.get("search", {})
        chromosome = search.get("chromosome", [])
        cmax = float(source.get("makespan", math.inf))
        if (
            int(source.get("schema_version", 0)) < 1
            or source.get("teacher_scope") != "stage2_resource_policy"
            or source.get("teacher_method") != "resource_iga_all"
            or source.get("resource_policy") != "drl"
            or source.get("arm") != "iga_all"
            or source.get("backends")
            != {"ordinary": "iga", "transporter": "iga"}
            or source.get("case") != case_name
            or not source.get("completion_verified", False)
            or not source.get("completed", False)
            or not source.get("frozen_model_unchanged", False)
            or not case_sha
            or not math.isfinite(cmax)
            or cmax >= 100000.0
            or not isinstance(chromosome, list)
            or len(chromosome) != int(search.get("n_var", -1))
        ):
            raise ValueError(f"Invalid verified Stage2 teacher: {source_path}")
        current_semantics = str(source["environment_semantics_version"])
        current_checkpoint_sha = str(source["frozen_plane_checkpoint_sha256"])
        current_command_sha = str(source["frozen_plane_source_command_sha256"])
        if semantics is None:
            semantics = current_semantics
            checkpoint_sha = current_checkpoint_sha
            source_command_sha = current_command_sha
        if (
            current_semantics != semantics
            or current_checkpoint_sha != checkpoint_sha
            or current_command_sha != source_command_sha
        ):
            raise ValueError("Stage2 source teacher provenance is not uniform.")

        compact = {
            "status": "completed",
            "schema_version": 1,
            "teacher_scope": "stage2_resource_policy",
            "teacher_method": "resource_iga_all",
            "resource_policy": "drl",
            "arm": "iga_all",
            "backends": {"ordinary": "iga", "transporter": "iga"},
            "environment_semantics_version": semantics,
            "case": case_name,
            "case_sha256": case_sha,
            "frozen_plane_checkpoint_sha256": checkpoint_sha,
            "frozen_plane_source_command_sha256": source_command_sha,
            "source_teacher_path": str(source_path.resolve()),
            "source_teacher_sha256": source_sha,
            "completion_verified": True,
            "completed": True,
            "makespan": cmax,
            "search": {
                "search_contract_version": int(
                    search.get("search_contract_version", 0)
                ),
                "n_var": int(search["n_var"]),
                "chromosome": chromosome,
            },
        }
        compact_path = output_dir / f"{case_name}.json"
        existing_ok = False
        if compact_path.is_file():
            try:
                existing = json.loads(compact_path.read_text(encoding="utf-8"))
                existing_ok = existing == compact
            except (OSError, json.JSONDecodeError):
                existing_ok = False
        if not existing_ok:
            atomic_json(compact_path, compact)
        compact_sha = sha256_file(compact_path)
        entries[case_name] = {
            "case_sha256": case_sha,
            "teacher_sha256": compact_sha,
            "source_teacher_sha256": source_sha,
            "makespan": cmax,
            "profile": metadata.get("profile"),
            "distribution": metadata.get("distribution"),
        }
        cmax_values.append(cmax)

    index = {
        "schema_version": 1,
        "teacher_scope": "stage2_resource_policy",
        "teacher_method": "resource_iga_all",
        "environment_semantics_version": semantics,
        "teacher_dir": str(output_dir.resolve()),
        "source_teacher_dir": str(source_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "frozen_plane_checkpoint_sha256": checkpoint_sha,
        "frozen_plane_source_command_sha256": source_command_sha,
        "case_count": len(entries),
        "mean_teacher_cmax": sum(cmax_values) / len(cmax_values),
        "entries": entries,
    }
    atomic_json(index_path, index)
    return index


def source_contract(handoff_path: Path, seed: int) -> tuple[dict, list[str]]:
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff.get("stage1_status") != "closed":
        raise ValueError("Stage1 handoff is not closed.")
    record = handoff.get("checkpoints", {}).get(str(seed))
    if not record:
        raise KeyError(f"No Stage1 source seed {seed} in handoff.")
    checkpoint = (ROOT / record["path"]).resolve()
    source_command_path = (ROOT / record["source_command_path"]).resolve()
    if sha256_file(checkpoint) != record["sha256"]:
        raise ValueError("Stage1 source checkpoint SHA256 mismatch.")
    if sha256_file(source_command_path) != record["source_command_sha256"]:
        raise ValueError("Stage1 source command SHA256 mismatch.")
    command = _load_source_command(
        source_command_path, record.get("source_command_key")
    )
    return {
        **record,
        "path": str(checkpoint),
        "source_command_path": str(source_command_path),
        "source_seed": seed,
        "handoff_path": str(handoff_path.resolve()),
        "handoff_sha256": sha256_file(handoff_path),
    }, command


def configure_command(
    base: list[str],
    method_id: str,
    method: dict,
    *,
    run_tag: str,
    profile: str,
    teacher_dir: Path,
    teacher_index: Path,
    low_memory: bool,
) -> list[str]:
    command = list(base)
    canary = profile in {
        "canary", "lateness_canary", "critical_canary", "planning_canary"
    }
    critical_round = profile in {"critical_canary", "critical_wave"}
    planning_round = profile in {"planning_canary", "planning_wave"}
    lateness_round = profile in {
        "lateness_canary", "lateness_wave", "critical_canary", "critical_wave",
        "planning_canary", "planning_wave",
    }
    # Wave-4 specifically introduces stateful lookahead reservations.  Its
    # canary must therefore exercise one complete aircraft lifecycle per
    # worker; the old fixed 64-step slice could never reach the 400--600-step
    # departure tail where lease promotion occurs.  One 20-case natural
    # rollout is sufficient for both lifecycle coverage and the concurrent
    # 1000-graph memory/backward check without turning preflight into a screen.
    train_cases = 20 if profile == "planning_canary" else 60 if canary else 180
    rollout_threads = (
        PLANNING_ROLLOUT_THREADS
        if planning_round
        else CRITICAL_ROLLOUT_THREADS if critical_round
        else CANARY_ROLLOUT_THREADS if canary else FORMAL_ROLLOUT_THREADS
    )
    ppo_epochs = 1 if canary else 4
    bc_epochs = (1 if canary else 4) if lateness_round else (1 if canary else 2)
    bc_rollouts = math.ceil(train_cases / rollout_threads)
    set_option(
        command,
        "--experiment_name",
        f"{run_tag}_{profile}_{method_id}_seed1",
    )
    set_option(command, "--seed", 1)
    set_option(command, "--num_episodes", ppo_epochs)
    set_option(command, "--ppo_epoch", 2)
    set_option(command, "--actor_warmup_shards", 0 if canary else 1)
    shared_unfreeze_epoch = method.get("shared_unfreeze_epoch")
    set_option(
        command,
        "--gnn_freeze_epochs",
        shared_unfreeze_epoch if shared_unfreeze_epoch is not None else ppo_epochs,
    )
    set_option(command, "--plane_freeze_epochs", ppo_epochs)
    set_switch(
        command,
        "--stage2_allow_shared_unfreeze",
        shared_unfreeze_epoch is not None,
    )
    set_option(command, "--device_bc_pretrain_epochs", bc_epochs)
    set_option(command, "--device_bc_min_labels_per_epoch", 64)
    set_option(command, "--device_bc_min_rollouts_per_epoch", bc_rollouts)
    set_option(command, "--device_bc_max_rollouts_per_epoch", bc_rollouts)
    set_option(
        command,
        "--device_bc_dagger_schedule",
        "1.0" if canary or not lateness_round else "1.0,0.7,0.4,0.1",
    )
    set_option(command, "--device_bc_dagger_seed", 1701)
    set_option(command, "--device_bc_teacher", method["teacher"])
    set_switch(
        command,
        "--device_bc_role_balanced",
        method["role_balanced"],
    )
    if method["teacher"] == "iga":
        set_option(command, "--resource_iga_teacher_dir", teacher_dir)
        set_option(command, "--resource_iga_teacher_index", teacher_index)
    else:
        remove_option(command, "--resource_iga_teacher_dir")
        remove_option(command, "--resource_iga_teacher_index")
    set_option(command, "--resource_ppo_update_schedule", "joint")
    set_option(command, "--resource_ppo_warmup_epochs", 0)
    set_option(command, "--n_rollout_threads", rollout_threads)
    set_option(
        command,
        "--n_eval_rollout_threads",
        PLANNING_EVAL_THREADS
        if planning_round
        else CRITICAL_EVAL_THREADS if critical_round else 12 if canary else 60,
    )
    set_option(command, "--max_train_cases", train_cases)
    set_option(command, "--train_sampling_mode", "distribution_balanced")
    set_option(
        command,
        "--train_sampling_weights",
        "iid=0.50,ood_stress=0.45,ood_scale=0.05",
    )
    set_option(command, "--train_sampling_size", 0)
    set_option(
        command,
        "--max_eval_cases",
        60 if planning_round
        else CRITICAL_EVAL_THREADS if critical_round else 12 if canary else 60,
    )
    set_option(
        command,
        "--selection_metric",
        "composite_tail" if critical_round or planning_round else "composite",
    )
    set_option(command, "--selection_iid_weight", 0.50)
    set_option(command, "--selection_ood_stress_weight", 0.45)
    set_option(command, "--selection_ood_scale_weight", 0.05)
    set_option(command, "--selection_tail_fraction", 0.10)
    set_option(command, "--selection_tail_weight", 0.25)
    set_option(command, "--evaluation_tau", 0.3)
    set_option(command, "--early_stop_patience", 0)
    set_option(command, "--eval_interval", 1)
    set_option(command, "--status_heartbeat_seconds", 30)
    set_option(command, "--recovery_checkpoint_interval_shards", 1)
    set_option(
        command,
        "--hindsight_reward_mode",
        method.get("reward_mode", "team_cmax"),
    )
    set_option(command, "--hindsight_cmax_coef", 0.0)
    set_option(command, "--hindsight_shaping_coef", 0.0)
    set_option(command, "--hindsight_terminal_cmax_coef", 1.0)
    set_option(command, "--iga_potential_beta", method.get("potential_beta", 0.0))
    remove_option(command, "--iga_potential_beta_schedule")
    set_option(command, "--iga_potential_gamma", 1.0)
    set_option(command, "--resource_slack_criticality_seconds", 1800.0)
    set_option(
        command,
        "--resource_slack_forecast_seconds",
        method.get("forecast_seconds", 0.0),
    )
    set_option(command, "--tail_policy_start_fraction", method.get("tail_start", 1.0))
    set_option(command, "--tail_policy_weight", method.get("tail_weight", 1.0))
    set_option(
        command,
        "--resource_lateness_coef",
        method.get("resource_lateness_coef", 0.0),
    )
    set_option(
        command,
        "--resource_critical_lateness_coef",
        method.get(
            "critical_wait_coef",
            method.get("resource_critical_lateness_coef", 0.0),
        ),
    )
    set_option(
        command,
        "--resource_earliness_coef",
        method.get(
            "earliness_coef", method.get("resource_earliness_coef", 0.0)
        ),
    )
    set_option(
        command,
        "--resource_wait_constraint_target",
        method.get("wait_constraint_target", 0.0),
    )
    set_option(
        command,
        "--resource_wait_dual_lr",
        method.get("wait_dual_lr", 0.0),
    )
    set_option(command, "--resource_wait_dual_max", 0.05)
    set_switch(
        command,
        "--device_deadline_aware_dispatch",
        lateness_round,
    )
    set_option(
        command,
        "--device_future_intent_horizon",
        1 if lateness_round else 0,
    )
    set_option(
        command,
        "--device_future_intent_mode",
        method.get("future_intent_mode", "legacy_one"),
    )
    set_option(command, "--device_frontier_max_requests", 2)
    set_switch(
        command,
        "--resource_release_aware_eta",
        bool(method.get("release_aware", False)),
    )
    set_option(
        command,
        "--device_lookahead_reservation_mode",
        method.get("reservation_mode", "none"),
    )
    set_option(command, "--device_reservation_grace_seconds", 300.0)
    set_switch(
        command,
        "--device_departure_lookahead",
        lateness_round,
    )
    # Legacy canaries remain short compute/memory probes.  The planning canary
    # is deliberately natural-termination-only because reservation correctness
    # depends on the final staging -> R014 -> runway transition.
    remove_option(command, "--rollout_until_done", takes_value=False)
    remove_option(command, "--no_rollout_until_done", takes_value=False)
    if canary:
        set_option(
            command,
            "--episode_length",
            PLANNING_TRAINING_STEPS
            if planning_round
            else CRITICAL_TRAINING_STEPS if critical_round else CANARY_TRAINING_STEPS,
        )
        command.append(
            "--rollout_until_done"
            if profile == "planning_canary"
            else "--no_rollout_until_done"
        )
    else:
        command.append("--rollout_until_done")
    # C0 supplies the single shared Pre-PPO baseline for all five identical
    # DeviceBC policies.  C1--C4 enter PPO while it is evaluated, exercising
    # evaluator/backward overlap and avoiding four duplicate 60-case passes.
    skip_pre_eval = (
        critical_round and method_id != "C0_cmax_control"
    ) or canary
    set_switch(command, "--skip_pre_ppo_eval", skip_pre_eval)
    set_switch(command, "--skip_epoch_eval", canary)
    set_switch(command, "--use_eval", True)
    set_switch(command, "--safe_graph_batch_pipeline", True)
    set_switch(command, "--safe_dagger_teacher_overlap", True)
    if planning_round:
        set_option(command, "--mini_batch_size", PLANNING_MINI_BATCH_SIZE)
        set_option(command, "--data_chunk_length", 50)
        set_option(
            command,
            "--max_graphs_per_forward",
            PLANNING_GRAPHS_PER_FORWARD,
        )
        set_option(command, "--grad_accumulation_steps", 5)
        set_option(command, "--actor_grad_accumulation_steps", 2)
        set_option(
            command,
            "--grad_accumulation_target_graphs",
            PLANNING_CRITIC_ACCUMULATION_TARGET_GRAPHS,
        )
        set_option(
            command,
            "--actor_grad_accumulation_target_graphs",
            PLANNING_ACTOR_ACCUMULATION_TARGET_GRAPHS,
        )
    elif critical_round:
        # Five processes share GPU0.  Twenty time-major environments by fifty
        # steps produce exactly the requested 1000-graph physical forward;
        # graph-target accumulation keeps optimizer mass equal to Wave 2.
        set_option(command, "--mini_batch_size", CRITICAL_MINI_BATCH_SIZE)
        set_option(command, "--data_chunk_length", 50)
        set_option(
            command,
            "--max_graphs_per_forward",
            CRITICAL_GRAPHS_PER_FORWARD,
        )
        set_option(command, "--grad_accumulation_steps", 5)
        set_option(command, "--actor_grad_accumulation_steps", 2)
        set_option(
            command,
            "--grad_accumulation_target_graphs",
            CRITICAL_CRITIC_ACCUMULATION_TARGET_GRAPHS,
        )
        set_option(
            command,
            "--actor_grad_accumulation_target_graphs",
            CRITICAL_ACTOR_ACCUMULATION_TARGET_GRAPHS,
        )
    elif low_memory:
        # Exercise the exact fallback batch in the three-process canary.  A
        # 16-env x 50-step forward keeps the effective actor/critic batches
        # close to the standard profile while reducing peak graph residency.
        set_option(command, "--mini_batch_size", 16)
        set_option(command, "--data_chunk_length", 50)
        set_option(
            command, "--max_graphs_per_forward", CANARY_LOWMEM_GRAPHS
        )
        set_option(command, "--grad_accumulation_steps", 5)
        set_option(command, "--actor_grad_accumulation_steps", 2)
        remove_option(command, "--grad_accumulation_target_graphs")
        remove_option(command, "--actor_grad_accumulation_target_graphs")
    elif not canary:
        # Sixty rollout workers divide the 180-case screen exactly into three
        # shards and match the proven Stage1 concurrency.  Two 30 x 50
        # forwards cover all workers. Graph-target accumulation preserves the
        # proven ~2000-graph Actor and ~5000-graph Critic optimizer mass even
        # though an indivisible physical forward now contains 1500 graphs.
        set_option(command, "--mini_batch_size", FORMAL_MINI_BATCH_SIZE)
        set_option(command, "--data_chunk_length", 50)
        set_option(
            command, "--max_graphs_per_forward", FORMAL_STANDARD_GRAPHS
        )
        set_option(command, "--grad_accumulation_steps", 3)
        set_option(command, "--actor_grad_accumulation_steps", 2)
        set_option(
            command,
            "--grad_accumulation_target_graphs",
            FORMAL_CRITIC_ACCUMULATION_TARGET_GRAPHS,
        )
        set_option(
            command,
            "--actor_grad_accumulation_target_graphs",
            FORMAL_ACTOR_ACCUMULATION_TARGET_GRAPHS,
        )
    else:
        # Canary and formal use the same peak forward size, so success is
        # direct OOM evidence rather than a linear memory extrapolation.
        set_option(command, "--mini_batch_size", FORMAL_MINI_BATCH_SIZE)
        set_option(command, "--data_chunk_length", 50)
        set_option(
            command, "--max_graphs_per_forward", CANARY_STANDARD_GRAPHS
        )
        set_option(command, "--grad_accumulation_steps", 3)
        set_option(command, "--actor_grad_accumulation_steps", 2)
        set_option(
            command,
            "--grad_accumulation_target_graphs",
            FORMAL_CRITIC_ACCUMULATION_TARGET_GRAPHS,
        )
        set_option(
            command,
            "--actor_grad_accumulation_target_graphs",
            FORMAL_ACTOR_ACCUMULATION_TARGET_GRAPHS,
        )
    for flag in (
        "--shared_eval_socket",
        "--shared_eval_cpu_set",
        "--shared_eval_timeout_seconds",
    ):
        remove_option(command, flag)
    return command


def natural_evaluator_command(command: list[str]) -> list[str]:
    """Keep validation complete even when the training canary is truncated."""

    result = list(command)
    remove_option(result, "--rollout_until_done", takes_value=False)
    remove_option(result, "--no_rollout_until_done", takes_value=False)
    result.append("--rollout_until_done")
    return result


def formal_gpu_calibration(
    suite_dir: Path,
    *,
    profile: str,
    low_memory: bool,
) -> dict | None:
    """Fail closed if canary evidence cannot support the formal graph batch."""

    if profile not in {"wave1", "critical_wave", "planning_wave"}:
        return None
    if profile == "planning_wave":
        attempt = "dual_gpu_ten_lane_1000"
        memory_log = suite_dir / "hardware/planning_canary_gpu_memory.csv"
        canary_graphs = PLANNING_GRAPHS_PER_FORWARD
        formal_graphs = PLANNING_GRAPHS_PER_FORWARD
        safety_limit_mib = PLANNING_GPU_MEMORY_SAFETY_LIMIT_MIB
    elif profile == "critical_wave":
        attempt = "five_lane_1000"
        memory_log = suite_dir / "hardware/critical_canary_gpu_memory.csv"
        canary_graphs = CRITICAL_GRAPHS_PER_FORWARD
        formal_graphs = CRITICAL_GRAPHS_PER_FORWARD
        safety_limit_mib = CRITICAL_GPU_MEMORY_SAFETY_LIMIT_MIB
    else:
        attempt = "lowmem" if low_memory else "standard"
        memory_log = suite_dir / "hardware" / f"canary_{attempt}_gpu_memory.csv"
        canary_graphs = (
            CANARY_LOWMEM_GRAPHS if low_memory else CANARY_STANDARD_GRAPHS
        )
        formal_graphs = (
            CANARY_LOWMEM_GRAPHS if low_memory else FORMAL_STANDARD_GRAPHS
        )
        safety_limit_mib = GPU_MEMORY_SAFETY_LIMIT_MIB
    if not memory_log.is_file():
        raise FileNotFoundError(
            "Formal profile requires the completed matching canary memory "
            f"record: {memory_log}"
        )
    used_values = []
    for line in memory_log.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split(",")
        if len(fields) >= 2:
            try:
                used_values.append(int(fields[1]))
            except ValueError:
                continue
    if not used_values:
        raise ValueError(f"Canary memory record has no samples: {memory_log}")
    peak_mib = max(used_values)
    projected_mib = math.ceil(peak_mib * formal_graphs / canary_graphs)
    if projected_mib > safety_limit_mib:
        raise RuntimeError(
            "Formal Stage2 graph batch is not supported by canary memory "
            f"evidence: projected={projected_mib} MiB, "
            f"limit={safety_limit_mib} MiB."
        )
    return {
        "source": str(memory_log.resolve()),
        "observed_peak_mib": peak_mib,
        "canary_graphs_per_forward": canary_graphs,
        "formal_graphs_per_forward": formal_graphs,
        "conservative_linear_projection_mib": projected_mib,
        "safety_limit_mib": safety_limit_mib,
        "projected_headroom_mib": safety_limit_mib - projected_mib,
        "trainer_count": 10 if profile == "planning_wave" else 5 if profile == "critical_wave" else 3,
        "trainer_count_per_gpu": 5 if profile == "planning_wave" else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "canary", "wave1", "lateness_canary", "lateness_wave",
            "critical_canary", "critical_wave", "planning_canary",
            "planning_wave",
        ),
        required=True,
    )
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--source-seed", type=int, default=3)
    parser.add_argument("--source-teacher-dir", type=Path, default=DEFAULT_SOURCE_TEACHERS)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compact_dir = args.suite_dir / "artifacts/resource_iga1800_compact"
    teacher_index_path = args.suite_dir / "artifacts/resource_iga1800_index.json"
    index = build_compact_teacher_index(
        args.source_teacher_dir.resolve(),
        args.dataset_dir.resolve(),
        compact_dir.resolve(),
        teacher_index_path.resolve(),
    )
    source, source_command = source_contract(
        args.handoff.resolve(), args.source_seed
    )
    if source["sha256"] != index["frozen_plane_checkpoint_sha256"]:
        raise ValueError(
            "Stage1 source checkpoint differs from the resource IGA teacher."
        )

    if args.profile in {"planning_canary", "planning_wave"}:
        methods = PLANNING_METHODS
    elif args.profile in {"critical_canary", "critical_wave"}:
        methods = CRITICAL_METHODS
    elif args.profile in {"lateness_canary", "lateness_wave"}:
        methods = LATENESS_METHODS
    else:
        methods = METHODS
    commands = {}
    for method_id, method in methods.items():
        base = build_stage2_command(
            source["path"],
            run_tag=f"{args.run_tag}_{args.profile}_{method_id}",
            seed=1,
            bc_epochs=(
                1
                if args.profile in {
                    "canary", "lateness_canary", "critical_canary",
                    "planning_canary",
                }
                else 4 if args.profile in {
                    "lateness_wave", "critical_wave", "planning_wave"
                } else 2
            ),
            ppo_epochs=(
                1
                if args.profile in {
                    "canary", "lateness_canary", "critical_canary",
                    "planning_canary",
                }
                else 4
            ),
            ppo_epoch=2,
            bc_min_labels=64,
            bc_min_rollouts=1,
            bc_max_rollouts=8,
            source_command=source_command,
            python=args.python.resolve(),
        )
        command = configure_command(
            base,
            method_id,
            method,
            run_tag=args.run_tag,
            profile=args.profile,
            teacher_dir=compact_dir.resolve(),
            teacher_index=teacher_index_path.resolve(),
            low_memory=args.low_memory,
        )
        commands[method_id] = {
            "argv": command,
            "shell": shlex.join(command),
            **method,
        }
    gpu_calibration = formal_gpu_calibration(
        args.suite_dir.resolve(),
        profile=args.profile,
        low_memory=bool(args.low_memory),
    )
    evaluator_key = (
        next(iter(commands))
        if args.profile.startswith(("critical_", "planning_"))
        else next(reversed(commands))
    )
    manifest = {
        "schema_version": 1,
        "research_stage": "stage2_resource_joint",
        "wave": (
            4 if args.profile.startswith("planning_")
            else 3 if args.profile.startswith("critical_")
            else 2 if args.profile.startswith("lateness_")
            else 1
        ),
        "profile": args.profile,
        "low_memory": bool(args.low_memory),
        "run_tag": args.run_tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "teacher_index_path": str(teacher_index_path.resolve()),
        "teacher_index_sha256": sha256_file(teacher_index_path),
        "teacher_case_count": int(index["case_count"]),
        "evaluator_command": natural_evaluator_command(
            commands[evaluator_key]["argv"]
        ),
        "evaluator_source_method": evaluator_key,
        "canary_training_steps": (
            PLANNING_TRAINING_STEPS
            if args.profile == "planning_canary"
            else CRITICAL_TRAINING_STEPS if args.profile == "critical_canary"
            else CANARY_TRAINING_STEPS
            if args.profile in {"canary", "lateness_canary"}
            else None
        ),
        "gpu_calibration": gpu_calibration,
        "throughput_profile": {
            "n_rollout_threads": (
                PLANNING_ROLLOUT_THREADS
                if args.profile.startswith("planning_")
                else CRITICAL_ROLLOUT_THREADS if args.profile.startswith("critical_")
                else CANARY_ROLLOUT_THREADS
                if args.profile == "canary"
                else FORMAL_ROLLOUT_THREADS
            ),
            "mini_batch_size": (
                PLANNING_MINI_BATCH_SIZE
                if args.profile.startswith("planning_")
                else CRITICAL_MINI_BATCH_SIZE if args.profile.startswith("critical_")
                else 16
                if args.low_memory
                else FORMAL_MINI_BATCH_SIZE
            ),
            "data_chunk_length": 50,
            "max_graphs_per_forward": (
                PLANNING_GRAPHS_PER_FORWARD
                if args.profile.startswith("planning_")
                else CRITICAL_GRAPHS_PER_FORWARD if args.profile.startswith("critical_")
                else CANARY_LOWMEM_GRAPHS
                if args.low_memory
                else (
                    CANARY_STANDARD_GRAPHS
                    if args.profile == "canary"
                    else FORMAL_STANDARD_GRAPHS
                )
            ),
            "critic_grad_accumulation_steps": (
                5
                if args.profile.startswith(("critical_", "planning_"))
                else 5
                if args.low_memory
                else 3
            ),
            "actor_grad_accumulation_steps": (
                2
                if args.low_memory
                else 2
            ),
            "critic_grad_accumulation_target_graphs": (
                PLANNING_CRITIC_ACCUMULATION_TARGET_GRAPHS
                if args.profile.startswith("planning_")
                else CRITICAL_CRITIC_ACCUMULATION_TARGET_GRAPHS if args.profile.startswith("critical_")
                else 0
                if args.low_memory
                else FORMAL_CRITIC_ACCUMULATION_TARGET_GRAPHS
            ),
            "actor_grad_accumulation_target_graphs": (
                PLANNING_ACTOR_ACCUMULATION_TARGET_GRAPHS
                if args.profile.startswith("planning_")
                else CRITICAL_ACTOR_ACCUMULATION_TARGET_GRAPHS if args.profile.startswith("critical_")
                else 0
                if args.low_memory
                else FORMAL_ACTOR_ACCUMULATION_TARGET_GRAPHS
            ),
        },
        "methods": methods,
        "commands": commands,
    }
    atomic_json(args.output.resolve(), manifest)
    print(json.dumps({
        "manifest": str(args.output.resolve()),
        "profile": args.profile,
        "low_memory": bool(args.low_memory),
        "methods": list(commands),
        "teacher_cases": index["case_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
