#!/usr/bin/env python3
"""Validate the Stage2 hand-off and emit a pure joint-PPO Stage3 command."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import (  # noqa: E402
    GNN_MAPPOPolicy,
)
from onpolicy.config.config import get_config  # noqa: E402
from onpolicy.scripts.train.train_hkbz import parse_args as parse_train_args  # noqa: E402
from onpolicy.utils.training_stage import (  # noqa: E402
    STAGE2_SUPERVISION_CONTRACT,
    validate_stage2_joint_finetune_checkpoint,
)

DEFAULT_SOURCE = ROOT / (
    "onpolicy/scripts/results/HKBZ/simple/gnn_mappo/"
    "stage2_planning_wave4_20260822_r2_planning_wave_"
    "B2_wait_constraint_seed1/run1/models/checkpoint_Best.pt"
)
TARGET_PLANNING_CONTRACT = {
    "device_lookahead_dispatch": True,
    "device_lookahead_safety_margin": 60.0,
    "device_deadline_aware_dispatch": True,
    "device_future_intent_horizon": 1,
    "device_future_intent_mode": "bounded_frontier",
    "device_frontier_max_requests": 2,
    "resource_release_aware_eta": True,
    "device_lookahead_reservation_mode": "soft",
    "device_reservation_grace_seconds": 300.0,
    "device_departure_lookahead": True,
}
TARGET_REWARD_CONTRACT = {
    "hindsight_reward_mode": "team_time",
    "resource_critical_lateness_coef": 0.05,
    "resource_earliness_coef": 0.002,
    "resource_wait_constraint_target": 21000.0,
    "resource_wait_dual_lr": 0.01,
    "resource_wait_dual_max": 0.05,
}
HARD_PLANNING_CONTRACT = {
    **TARGET_PLANNING_CONTRACT,
    "device_lookahead_reservation_mode": "hard",
}
PURE_STAGE2_REWARD_CONTRACT = {
    "hindsight_reward_mode": "team_time",
    "resource_critical_lateness_coef": 0.0,
    "resource_earliness_coef": 0.0,
    "resource_wait_constraint_target": 0.0,
    "resource_wait_dual_lr": 0.0,
    # A zero learning rate makes the dual inactive.  Retaining the finite
    # bound is part of the latest Stage2 checkpoint contract and does not add
    # reward shaping.
    "resource_wait_dual_max": 0.05,
}
# A trainable shared encoder at graph=1000 needs more than the 55.48-GiB
# allocator budget imposed by a 0.70 fraction: both the plain E2 control and
# E3 reached that same cap while the physical A800 still had >22 GiB free.
# Updates are serialized by one GPU-local phase lock and unused cache is
# released before the lock changes owner.  A 0.85 cap therefore applies only
# to the single active updater; the evaluator plus three rollout-only trainers
# leave roughly 7 GiB of physical headroom even if that updater reaches its
# full 67.4-GiB allowance.
TRAINER_CUDA_MEMORY_FRACTION = 0.85
MINI_BATCH_SIZE = 16
DATA_CHUNK_LENGTH = 50
GRAPHS_PER_FORWARD = MINI_BATCH_SIZE * DATA_CHUNK_LENGTH


def profile_settings(profile: str) -> dict:
    """Return the immutable workload used by one of four GPU-local lanes."""

    settings = {
        "memory_canary": {
            "episodes": 1,
            # graph=1000 uses 20 x 50, so the memory profile retains twenty
            # rollout processes and exercises one real full-size forward.
            "n_rollout_threads": 20,
            "max_train_cases": 20,
            "max_eval_cases": 10,
            "ppo_epoch": 1,
            "train_sampling_size": 20,
        },
        "canary": {
            "episodes": 1,
            "n_rollout_threads": 20,
            "max_train_cases": 24,
            "max_eval_cases": 20,
            "ppo_epoch": 1,
            "train_sampling_size": 24,
        },
        "wave1": {
            "episodes": 2,
            "n_rollout_threads": 20,
            "max_train_cases": 120,
            "max_eval_cases": 60,
            "ppo_epoch": 1,
            "train_sampling_size": 120,
        },
        "encoder_wave1": {
            "episodes": 4,
            "n_rollout_threads": 20,
            "max_train_cases": 120,
            "max_eval_cases": 60,
            "ppo_epoch": 1,
            "train_sampling_size": 120,
        },
        "critical_wave1": {
            "episodes": 4,
            "n_rollout_threads": 20,
            "max_train_cases": 120,
            "max_eval_cases": 60,
            "ppo_epoch": 1,
            "train_sampling_size": 120,
            "train_sampling_pool_size": 0,
            "actor_warmup_shards": 0,
        },
        "ppo_gain_memory_canary": {
            "episodes": 1,
            "n_rollout_threads": 20,
            "max_train_cases": 20,
            "max_eval_cases": 10,
            "ppo_epoch": 1,
            "train_sampling_size": 20,
            "train_sampling_pool_size": 0,
            "actor_warmup_shards": 0,
        },
        "ppo_gain_wave1": {
            # Epoch 1 is critic-only calibration on 240 cases; epochs 2-5 are
            # four Actor PPO epochs over one complete 960-case rotating pool.
            "episodes": 5,
            "n_rollout_threads": 20,
            "max_train_cases": 0,
            "max_eval_cases": 60,
            "ppo_epoch": 1,
            "train_sampling_size": 240,
            "train_sampling_pool_size": 960,
            "actor_warmup_shards": 12,
        },
        "ppo_gain_wave2": {
            "episodes": 5,
            "n_rollout_threads": 20,
            "max_train_cases": 0,
            "max_eval_cases": 60,
            "ppo_epoch": 1,
            "train_sampling_size": 240,
            "train_sampling_pool_size": 960,
            "actor_warmup_shards": 12,
        },
        "credit_happo_memory_canary": {
            "episodes": 1,
            "n_rollout_threads": 20,
            "max_train_cases": 20,
            "max_eval_cases": 10,
            "ppo_epoch": 1,
            "train_sampling_size": 20,
            "train_sampling_pool_size": 0,
            "actor_warmup_shards": 0,
        },
        "credit_happo_wave1": {
            # One 240-case critic calibration epoch followed by four actor
            # epochs over the same deterministic 960-case rotating pool.
            "episodes": 5,
            "n_rollout_threads": 20,
            "max_train_cases": 0,
            "max_eval_cases": 60,
            "ppo_epoch": 1,
            "train_sampling_size": 240,
            "train_sampling_pool_size": 960,
            "actor_warmup_shards": 12,
        },
        "gradient_conflict_memory_canary": {
            # Exercise an unfrozen graph=1000 update in every lane before the
            # scientific screen is admitted.
            "episodes": 1,
            "n_rollout_threads": 20,
            "max_train_cases": 20,
            "max_eval_cases": 10,
            "ppo_epoch": 1,
            "train_sampling_size": 20,
            "train_sampling_pool_size": 0,
            "actor_warmup_shards": 0,
        },
        "gradient_conflict_wave1": {
            # Epoch 1 is critic-only, Epoch 2 adapts role heads with the shared
            # encoder frozen, and Epochs 3-4 isolate the gradient combiner at
            # the proven fixed shared LR scale of 0.01.
            "episodes": 4,
            "n_rollout_threads": 20,
            "max_train_cases": 0,
            "max_eval_cases": 60,
            "ppo_epoch": 1,
            "train_sampling_size": 240,
            "train_sampling_pool_size": 960,
            "actor_warmup_shards": 12,
        },
        "formal": {
            "episodes": 6,
            # Four trainers share 64 physical cores.  Forty workers per lane
            # retains large rollout batches without reproducing the severe
            # five-lane CPU oversubscription used by the Stage2 screen.
            "n_rollout_threads": 40,
            "max_train_cases": 0,
            "max_eval_cases": 60,
            "ppo_epoch": 2,
            "train_sampling_size": 960,
            "train_sampling_pool_size": 0,
            "actor_warmup_shards": 0,
        },
    }
    try:
        return dict(settings[profile])
    except KeyError as error:
        raise ValueError(f"Unknown Stage3 profile: {profile!r}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping) -> None:
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


def _updated(before: object, after: object) -> bool:
    return (
        isinstance(before, Mapping)
        and isinstance(after, Mapping)
        and before.get("sha256") != after.get("sha256")
    )


def _stage2_source_contracts(
    checkpoint: Mapping[str, object],
    *,
    inherit_source_contracts: bool,
) -> tuple[dict, dict]:
    """Return Stage2 planning plus the new Stage3 PPO reward contract.

    Supervised Stage2 has no reward/dual lineage.  The returned reward mapping
    is therefore an explicit Stage3 initialization, never inherited RL state.
    """

    planning = checkpoint.get("resource_lookahead_contract")
    if not isinstance(planning, Mapping):
        raise ValueError("Stage2 source has no planning contract.")
    planning = dict(planning)
    allowed_planning = (
        (TARGET_PLANNING_CONTRACT, HARD_PLANNING_CONTRACT)
        if inherit_source_contracts else (TARGET_PLANNING_CONTRACT,)
    )
    if planning not in allowed_planning:
        raise ValueError(
            "Unsupported Stage2 planning contract: "
            f"{planning!r}."
        )
    return planning, dict(TARGET_REWARD_CONTRACT)


def validate_stage2_source(
    path: Path,
    *,
    inherit_source_contracts: bool = False,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        ready_time_scale = float(checkpoint["request_ready_time_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Stage2 source has no numeric request-ready time scale."
        ) from error
    if not math.isfinite(ready_time_scale) or ready_time_scale <= 0.0:
        raise ValueError(
            "Stage2 source request-ready time scale must be finite and positive."
        )
    planning_contract, reward_contract = _stage2_source_contracts(
        checkpoint,
        inherit_source_contracts=inherit_source_contracts,
    )
    if (
        checkpoint.get("training_stage") != "resource_joint"
        or checkpoint.get("phase") != "resource_supervised_completed"
        or checkpoint.get("stage2_training_mode") != "supervised_only"
        or checkpoint.get("stage2_supervision_contract")
        != STAGE2_SUPERVISION_CONTRACT
        or checkpoint.get("request_ready_prediction") is not True
        or checkpoint.get("plane_order_mode") != "fixed"
        or checkpoint.get("plane_pair_decoder") != "joint_pair"
        or checkpoint.get("global_feature_mode") != "f1f2"
        or not isinstance(checkpoint.get("model"), Mapping)
        or len(checkpoint["model"]) < 1
    ):
        raise ValueError(f"Incompatible Stage2 source checkpoint: {path}")
    bc_updated = _updated(
        checkpoint.get("resource_actor_summary_before_bc"),
        checkpoint.get("resource_actor_summary_after_bc"),
    )
    predictor_updated = _updated(
        checkpoint.get("request_ready_predictor_summary_before"),
        checkpoint.get("request_ready_predictor_summary_after"),
    )
    if not bc_updated or not predictor_updated:
        raise ValueError(
            "Stage2 source lacks independent mobile-policy/predictor update evidence."
        )
    if int(checkpoint.get("resource_dense_ranking_total_labels", 0)) <= 0:
        raise ValueError("Stage2 source has no dense candidate-ranking labels.")
    if int(checkpoint.get("request_ready_total_labels", 0)) <= 0:
        raise ValueError("Stage2 source has no intrinsic-ready-time labels.")
    if any(
        checkpoint.get(key) is not None
        for key in (
            "actor_optim", "critic_optim", "value_normalizer",
            "role_value_normalizers", "shared_gradient_state",
        )
    ):
        raise ValueError("Supervised Stage2 source contains forbidden RL state.")
    source_m2_path = Path(str(checkpoint.get("source_m2_path", "")))
    if not source_m2_path.is_file():
        raise FileNotFoundError(
            f"Stage2 source lineage is missing: {source_m2_path}"
        )
    source_m2_sha = _sha256_file(source_m2_path)
    if source_m2_sha != checkpoint.get("source_m2_sha256"):
        raise ValueError("Stage2 source has a stale Stage1 lineage digest.")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "stage": checkpoint.get("stage"),
        "episodes": int(checkpoint.get("episodes", 0)),
        "model_tensor_count": len(checkpoint["model"]),
        "source_m2_path": str(source_m2_path.resolve()),
        "source_m2_sha256": source_m2_sha,
        "resource_bc_updated": bc_updated,
        "request_ready_predictor_updated": predictor_updated,
        "resource_ppo_updated": False,
        "request_ready_time_scale": float(
            ready_time_scale
        ),
        "planning_contract": planning_contract,
        "stage2_reward_contract": None,
        "reward_contract": reward_contract,
    }


def _add(command: list[str], flag: str, value: object | None = None) -> None:
    command.append(flag)
    if value is not None:
        command.append(str(value))


def _command_value(command: list[str], flag: str) -> str:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"Expected exactly one valued command flag {flag}.")
    return str(command[positions[0] + 1])


def _runtime_planning_contract(command: list[str]) -> dict:
    return {
        "device_lookahead_dispatch": "--device_lookahead_dispatch" in command,
        "device_lookahead_safety_margin": float(_command_value(
            command, "--device_lookahead_safety_margin"
        )),
        "device_deadline_aware_dispatch": (
            "--device_deadline_aware_dispatch" in command
        ),
        "device_future_intent_horizon": int(_command_value(
            command, "--device_future_intent_horizon"
        )),
        "device_future_intent_mode": _command_value(
            command, "--device_future_intent_mode"
        ),
        "device_frontier_max_requests": int(_command_value(
            command, "--device_frontier_max_requests"
        )),
        "resource_release_aware_eta": (
            "--resource_release_aware_eta" in command
        ),
        "device_lookahead_reservation_mode": _command_value(
            command, "--device_lookahead_reservation_mode"
        ),
        "device_reservation_grace_seconds": float(_command_value(
            command, "--device_reservation_grace_seconds"
        )),
        "device_departure_lookahead": (
            "--device_departure_lookahead" in command
        ),
    }


def _runtime_reward_contract(command: list[str]) -> dict:
    return {
        "hindsight_reward_mode": _command_value(
            command, "--hindsight_reward_mode"
        ),
        "resource_critical_lateness_coef": float(_command_value(
            command, "--resource_critical_lateness_coef"
        )),
        "resource_earliness_coef": float(_command_value(
            command, "--resource_earliness_coef"
        )),
        "resource_wait_constraint_target": float(_command_value(
            command, "--resource_wait_constraint_target"
        )),
        "resource_wait_dual_lr": float(_command_value(
            command, "--resource_wait_dual_lr"
        )),
        "resource_wait_dual_max": float(_command_value(
            command, "--resource_wait_dual_max"
        )),
    }


def build_command(args: argparse.Namespace) -> list[str]:
    profile = str(args.profile)
    settings = profile_settings(profile)
    graphs_per_forward = int(getattr(
        args, "max_graphs_per_forward", GRAPHS_PER_FORWARD
    ))
    if graphs_per_forward <= 0 or graphs_per_forward % DATA_CHUNK_LENGTH != 0:
        raise ValueError(
            "Stage3 max_graphs_per_forward must be a positive multiple of "
            f"data_chunk_length={DATA_CHUNK_LENGTH}."
        )
    mini_batch_size = graphs_per_forward // DATA_CHUNK_LENGTH
    if mini_batch_size > settings["n_rollout_threads"]:
        raise ValueError(
            "Stage3 PPO mini-batch exceeds the rollout worker count: "
            f"{mini_batch_size} > {settings['n_rollout_threads']}."
        )
    method = str(getattr(args, "method", "canonical"))
    source_planning_contract = dict(getattr(
        args, "source_planning_contract", TARGET_PLANNING_CONTRACT
    ))
    source_reward_contract = dict(getattr(
        args, "source_reward_contract", TARGET_REWARD_CONTRACT
    ))
    inherit_source_contracts = bool(getattr(
        args, "inherit_source_contracts", False
    ))
    standalone_eval = bool(getattr(args, "standalone_eval", False))
    gradient_conflict_method = method in {"G0", "G1", "G2", "G3"}
    new_credit_method = method in {
        "N0", "N1", "N2", "N3", "G0", "G1", "G2", "G3"
    }
    ppo_gain_method = method in {
        "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3"
    } or new_credit_method
    base_credit_method = (
        method if method.startswith("P")
        else str(getattr(args, "base_credit_method", "P3"))
        if method.startswith("Q")
        else ""
    )
    if method.startswith("Q") and base_credit_method not in {
        "P0", "P1", "P2", "P3"
    }:
        raise ValueError(
            "Wave-2 --base-credit-method must be one of P0/P1/P2/P3."
        )
    role_atomic = method in {
        "A1", "A2", "A3", "B0", "B1", "B2", "B3",
        "E0", "E1", "E2", "E3", "C0", "C1", "C2", "C3",
        "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3",
        "N0", "N1", "N2", "N3",
        "G0", "G1", "G2", "G3",
    }
    role_event_returns = (
        method in {"A2", "A3", "B2", "B3"}
        or base_credit_method in {"P2", "P3"}
        or new_credit_method
    )
    role_valuenorm = (
        method == "A3" or base_credit_method in {"P2", "P3"}
        or new_credit_method
    )
    role_event_gae_lambda = (
        0.90 if base_credit_method == "P2"
        else 0.95 if method in {"B2", "B3"}
        else 1.0
    )
    role_gae_lambdas = {
        "plane": 0.95,
        "device": 0.80,
        "transporter": 0.90,
    } if base_credit_method == "P3" else {
        role_name: role_event_gae_lambda
        for role_name in ("plane", "device", "transporter")
    }
    role_loss_weighting = (
        "sqrt_event"
        if method in {
            "B1", "B3", "E0", "E1", "E2", "E3",
            "C0", "C1", "C2", "C3",
            "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3",
        }
        else "fixed"
    )
    legacy_shared_frozen = method in {
        "A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3"
    }
    shared_frozen = legacy_shared_frozen or method == "E0"
    encoder_screen = method in {
        "E0", "E1", "E2", "E3", "C0", "C1", "C2", "C3",
        "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3",
        "N0", "N1", "N2", "N3",
        "G0", "G1", "G2", "G3",
    }
    if method == "E0":
        gnn_freeze_epochs = settings["episodes"]
        shared_lr_schedule = ",".join(["0.0"] * settings["episodes"])
    elif method == "E1":
        gnn_freeze_epochs = 0
        shared_lr_schedule = ",".join(["0.02"] * settings["episodes"])
    elif gradient_conflict_method:
        if settings["episodes"] >= 2:
            gnn_freeze_epochs = 2
            schedule = [0.0, 0.0, 0.01, 0.01]
            shared_lr_schedule = ",".join(
                str(schedule[min(index, len(schedule) - 1)])
                for index in range(settings["episodes"])
            )
        else:
            gnn_freeze_epochs = 0
            shared_lr_schedule = "0.01"
    elif ppo_gain_method:
        if settings["episodes"] >= 2:
            if method in {"Q2", "Q3"}:
                # Calibration has Actor updates disabled.  The four following
                # PPO epochs therefore see 0.01 -> 0.025 -> 0.05 -> 0.10 with
                # no artificial shared-encoder freeze.
                gnn_freeze_epochs = 0
                schedule = [0.0, 0.01, 0.025, 0.05, 0.10]
            else:
                # P0-P3 and Q0/Q1 retain the proven gentle-ramp control: the
                # calibration epoch plus first Actor epoch are shared-frozen.
                gnn_freeze_epochs = 2
                schedule = [0.0, 0.0, 0.01, 0.025, 0.05]
            shared_lr_schedule = ",".join(
                str(schedule[min(index, len(schedule) - 1)])
                for index in range(settings["episodes"])
            )
        else:
            gnn_freeze_epochs = 0
            shared_lr_schedule = "0.01"
    elif method in {"E2", "E3", "C0", "C1", "C2", "C3"}:
        # Short canaries must exercise the unfrozen update for a meaningful
        # VRAM test.  The scientific four-epoch arm preserves one complete
        # frozen stabilization epoch before the 0.01 -> 0.025 -> 0.05 ramp.
        if settings["episodes"] >= 2:
            gnn_freeze_epochs = 1
            schedule = [0.0, 0.01, 0.025, 0.05]
            shared_lr_schedule = ",".join(
                str(schedule[min(index, len(schedule) - 1)])
                for index in range(settings["episodes"])
            )
        else:
            gnn_freeze_epochs = 0
            shared_lr_schedule = "0.01"
    else:
        gnn_freeze_epochs = settings["episodes"] if shared_frozen else 0
        shared_lr_schedule = "0.1"
    critical_path_arm = method in {
        "C0", "C1", "C2", "C3",
        "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3",
        "N0", "N1", "N2", "N3",
        "G0", "G1", "G2", "G3",
    }
    critical_potential = method in {"C1", "C3"}
    extended_frontier = method in {"C2", "C3"} or ppo_gain_method
    legacy_wait_objective = (
        not critical_potential
        and (not ppo_gain_method or base_credit_method == "P0")
    )
    reward_mode = (
        "team_time_resource_potential" if critical_potential else "team_time"
    )
    potential_schedule = (
        "0.02,0.03,0.04,0.04" if critical_potential else "0.0"
    )
    future_horizon = 3 if extended_frontier else 1
    frontier_requests = 4 if extended_frontier else 2
    source_reward_is_pure = all(
        float(source_reward_contract[key]) == 0.0
        for key in (
            "resource_critical_lateness_coef",
            "resource_earliness_coef",
            "resource_wait_constraint_target",
            "resource_wait_dual_lr",
        )
    )
    preserve_pure_source_reward = (
        inherit_source_contracts
        and source_reward_is_pure
        and not critical_potential
    )
    command = [str(args.python), str(ROOT / "onpolicy/scripts/train/train_hkbz.py")]
    valued = {
        "--env_name": "HKBZ",
        "--scenario_name": "simple",
        "--algorithm_name": "gnn_mappo",
        "--experiment_name": args.experiment_name,
        "--ac_config": ROOT / "onpolicy/config/ac.yaml",
        "--env_config": ROOT / "onpolicy/config/env_joint_finetune.yaml",
        "--training_stage": "joint_finetune",
        "--resource_policy": "drl",
        "--checkpoint_dir": args.source_checkpoint,
        "--request_ready_time_scale": float(getattr(
            args, "request_ready_time_scale", 3600.0
        )),
        "--seed": args.seed,
        "--cuda_memory_fraction": TRAINER_CUDA_MEMORY_FRACTION,
        "--n_training_threads": 1,
        "--n_rollout_threads": settings["n_rollout_threads"],
        "--n_eval_rollout_threads": 10,
        "--max_train_cases": settings["max_train_cases"],
        "--max_eval_cases": settings["max_eval_cases"],
        "--eval_case_offset": 0,
        "--eval_partition_seed": 20260803,
        "--eval_partition_stratify_by": "profile",
        "--train_sampling_mode": "distribution_balanced",
        "--train_sampling_weights": "iid=0.50,ood_stress=0.45,ood_scale=0.05",
        "--train_sampling_size": settings["train_sampling_size"],
        "--train_sampling_pool_size": settings.get(
            "train_sampling_pool_size", 0
        ),
        "--num_episodes": settings["episodes"],
        "--rollout_max_steps": 4000,
        "--ppo_epoch": settings["ppo_epoch"],
        "--mini_batch_size": mini_batch_size,
        "--data_chunk_length": DATA_CHUNK_LENGTH,
        "--max_graphs_per_forward": graphs_per_forward,
        "--grad_accumulation_steps": 5,
        "--actor_grad_accumulation_steps": 2,
        "--grad_accumulation_target_graphs": 5000,
        "--actor_grad_accumulation_target_graphs": 2000,
        "--actor_warmup_shards": settings.get("actor_warmup_shards", 0),
        "--plane_bc_pretrain_epochs": 0,
        "--device_bc_pretrain_epochs": 0,
        "--plane_bc_dagger_schedule": "0.0",
        "--plane_bc_staging_dagger_schedule": "0.0",
        "--device_bc_dagger_schedule": "0.0",
        "--resource_ppo_update_schedule": "joint",
        "--resource_ppo_warmup_epochs": 0,
        "--gnn_freeze_epochs": gnn_freeze_epochs,
        "--plane_freeze_epochs": 0,
        "--plane_order_freeze_epochs": 0,
        "--plane_order_mode": "fixed",
        "--plane_pair_decoder": "joint_pair",
        "--global_feature_mode": "f1f2",
        "--lr": "5e-6",
        "--critic_lr": "1e-4",
        "--shared_actor_lr_scale": float(shared_lr_schedule.split(",")[0]),
        "--shared_actor_lr_scale_schedule": shared_lr_schedule,
        "--shared_actor_lr_max_multiplier": 2.0 if encoder_screen else 0.0,
        "--plane_actor_lr_scale": 0.25,
        "--device_actor_lr_scale": 1.0,
        "--transporter_actor_lr_scale": 0.5,
        "--anneal_original": 0.3,
        "--anneal_final": 0.3,
        "--tau_anneal_epochs": 0,
        "--entropy_coef": 0.0005,
        "--clip_param": 0.05,
        "--target_kl": 0.005,
        "--plane_target_kl": 0.0025,
        "--device_target_kl": 0.005,
        "--transporter_target_kl": 0.005,
        "--role_event_gae_lambda": role_event_gae_lambda,
        "--plane_role_gae_lambda": role_gae_lambdas["plane"],
        "--device_role_gae_lambda": role_gae_lambdas["device"],
        "--transporter_role_gae_lambda": role_gae_lambdas["transporter"],
        "--role_loss_weighting": role_loss_weighting,
        "--plane_loss_coef": 1.0,
        "--device_loss_coef": 1.0 if new_credit_method else 0.5,
        "--transporter_loss_coef": 1.0 if new_credit_method else 0.5,
        "--role_loss_min_share": 0.15,
        "--role_loss_max_share": 0.60,
        "--shared_gradient_method": {
            "G0": "sum",
            "G1": "norm_balance",
            "G2": "norm_pcgrad",
            "G3": "cagrad",
        }.get(method, "sum"),
        "--shared_grad_ema_beta": 0.97,
        "--shared_grad_norm_power": 0.5,
        "--shared_grad_min_scale": 0.5,
        "--shared_grad_max_scale": 2.0,
        "--shared_grad_conflict_threshold": -0.05,
        "--shared_cagrad_c": 0.2,
        "--adaptive_actor_kl_low": 0.0001,
        "--adaptive_actor_kl_high": 0.0005,
        "--adaptive_actor_lr_min_scale": 0.25,
        "--adaptive_actor_lr_max_scale": 4.0,
        "--adaptive_actor_lr_up": 1.5,
        "--adaptive_actor_lr_down": 0.5,
        "--adaptive_actor_min_step_completion": 0.9,
        "--bc_reference_kl_coef": 0.0,
        "--bc_reference_kl_coef_schedule": "0.0",
        "--bc_reference_target_kl": 0.0,
        "--max_grad_norm": 1.0,
        "--hindsight_reward_mode": reward_mode,
        "--hindsight_cmax_coef": 0.0,
        "--hindsight_shaping_coef": 0.0,
        "--hindsight_terminal_cmax_coef": 1.0,
        "--reward_coef": 0.01,
        "--resource_lateness_coef": 0.0,
        "--resource_critical_lateness_coef": (
            source_reward_contract["resource_critical_lateness_coef"]
            if preserve_pure_source_reward
            else 0.05 if legacy_wait_objective else 0.0
        ),
        "--resource_earliness_coef": (
            source_reward_contract["resource_earliness_coef"]
            if preserve_pure_source_reward
            else 0.002 if legacy_wait_objective else 0.0
        ),
        "--resource_wait_constraint_target": (
            source_reward_contract["resource_wait_constraint_target"]
            if preserve_pure_source_reward
            else 21000.0 if legacy_wait_objective else 0.0
        ),
        "--resource_wait_dual_lr": (
            source_reward_contract["resource_wait_dual_lr"]
            if preserve_pure_source_reward
            else 0.01 if legacy_wait_objective else 0.0
        ),
        "--resource_wait_dual_max": (
            source_reward_contract["resource_wait_dual_max"]
            if preserve_pure_source_reward
            else 0.05 if legacy_wait_objective else 0.0
        ),
        "--iga_potential_beta": 0.02 if critical_potential else 0.0,
        "--iga_potential_beta_schedule": potential_schedule,
        "--iga_potential_gamma": 1.0,
        "--resource_slack_criticality_seconds": 600.0,
        "--resource_slack_min_weight": 0.05,
        "--resource_slack_forecast_seconds": 0.0,
        "--tail_policy_start_fraction": (
            0.75 if method in {"Q1", "Q3"} else 1.0
        ),
        "--tail_policy_weight": (
            2.0 if method in {"Q1", "Q3"} else 1.0
        ),
        "--device_lookahead_safety_margin": source_planning_contract[
            "device_lookahead_safety_margin"
        ],
        "--device_future_intent_horizon": future_horizon,
        "--device_future_intent_mode": source_planning_contract[
            "device_future_intent_mode"
        ],
        "--device_frontier_max_requests": frontier_requests,
        "--device_request_capacity_per_plane": (
            5 if critical_path_arm else 0
        ),
        "--device_lookahead_reservation_mode": source_planning_contract[
            "device_lookahead_reservation_mode"
        ],
        "--device_reservation_grace_seconds": source_planning_contract[
            "device_reservation_grace_seconds"
        ],
        "--stage3_handoff_mode": (
            "ppo_gain_wave" if ppo_gain_method
            else "critical_path_wave1" if critical_path_arm else "strict"
        ),
        "--selection_metric": "raw" if new_credit_method else "composite_tail",
        "--selection_iid_weight": 0.50,
        "--selection_ood_stress_weight": 0.45,
        "--selection_ood_scale_weight": 0.05,
        "--selection_tail_fraction": 0.10,
        "--selection_tail_weight": 0.25,
        "--evaluation_tau": 0.3,
        "--eval_interval": 1,
        "--canary_eval_interval_shards": (
            1 if not standalone_eval and profile in {
                "canary", "wave1", "encoder_wave1", "critical_wave1",
                "ppo_gain_wave1", "ppo_gain_wave2", "credit_happo_wave1",
                "gradient_conflict_wave1",
            } else 0
        ),
        "--canary_eval_max_per_epoch": (
            1 if not standalone_eval and profile in {
                "wave1", "encoder_wave1", "critical_wave1",
                "ppo_gain_wave1", "ppo_gain_wave2", "credit_happo_wave1",
                "gradient_conflict_wave1",
            } else 0
        ),
        "--canary_eval_max_cases": (
            20 if not standalone_eval and profile in {
                "wave1", "encoder_wave1", "critical_wave1",
                "ppo_gain_wave1", "ppo_gain_wave2", "credit_happo_wave1",
                "gradient_conflict_wave1",
            } else 0
        ),
        "--canary_max_regression": 0.02,
        "--early_stop_patience": 0,
        "--save_interval": 1,
        "--status_heartbeat_seconds": 30,
        "--recovery_checkpoint_interval_shards": 1,
        "--ipc_timeout_seconds": 300.0,
        "--torch_mp_sharing_strategy": "file_descriptor",
        "--safe_async_graph_clone_workers": 4,
        "--joint_team_ppo_scope": "all",
        "--counterfactual_q_topk": 8,
        "--counterfactual_q_min_mass": 0.90,
        "--role_sequential_factor_clip": 2.0,
        "--role_sequential_min_ess": 0.50,
    }
    for flag, value in valued.items():
        _add(command, flag, value)
    for switch in (
        "--rollout_until_done",
        "--use_valuenorm",
        "--use_eval",
        "--strict_checkpoint_contract",
        "--request_ready_prediction",
        "--adaptive_actor_kl",
        "--joint_team_ppo",
        "--device_lookahead_dispatch",
        "--device_deadline_aware_dispatch",
        "--resource_release_aware_eta",
        "--device_departure_lookahead",
        "--safe_graph_batch_pipeline",
        "--clear_cuda_cache_after_update",
    ):
        _add(command, switch)
    if shared_frozen:
        _add(command, "--stage3_allow_shared_frozen")
    if role_atomic:
        _add(command, "--role_atomic_ppo")
    if role_event_returns:
        _add(command, "--role_event_returns")
    if role_valuenorm:
        _add(command, "--role_valuenorm")
    if encoder_screen and method != "E0":
        _add(command, "--shared_gradient_diagnostics")
        _add(command, "--shared_encoder_activation_checkpoint")
    if method in {"E3", "Q2", "Q3", "N0", "N1", "N2", "N3"}:
        _add(command, "--shared_encoder_pcgrad")
    if method in {"N1", "N3"}:
        _add(command, "--counterfactual_q_baseline")
    if method in {"N2", "N3"}:
        _add(command, "--role_sequential_ppo")
    if not standalone_eval and profile in {
        "canary", "wave1", "encoder_wave1", "critical_wave1",
        "ppo_gain_wave1", "ppo_gain_wave2", "credit_happo_wave1",
        "gradient_conflict_wave1",
    }:
        _add(command, "--canary_stop_on_regression")
    return command


def validate_runtime_handoff(
    path: Path,
    command: list[str],
    *,
    source_planning_contract: Mapping[str, object] = TARGET_PLANNING_CONTRACT,
    source_reward_contract: Mapping[str, object] = TARGET_REWARD_CONTRACT,
) -> dict:
    """Instantiate the production architecture and prove an exact hand-off.

    Metadata-only validation is not enough for a staged transition: a stale
    ``ac.yaml`` can leave every provenance field correct while changing a
    tensor name or shape.  The preparer therefore builds the same policy that
    ``train_hkbz.py`` will build, on CPU, and runs the runner's strict model
    contract before declaring the manifest launchable.
    """

    runtime_args = parse_train_args(command[2:], get_config())
    with Path(runtime_args.ac_config).open("r", encoding="utf-8") as source:
        ac_config = yaml.safe_load(source)
    policy = GNN_MAPPOPolicy(runtime_args, ac_config, device=torch.device("cpu"))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validation = validate_stage2_joint_finetune_checkpoint(
        checkpoint,
        policy.ac.state_dict(),
        plane_order_mode=policy.ac.plane_order_mode,
        plane_pair_decoder=policy.ac.plane_pair_decoder,
        global_feature_mode=runtime_args.global_feature_mode,
        planning_contract=source_planning_contract,
        reward_contract=None,
        request_ready_time_scale=(
            policy.ac.request_ready_time_scale
        ),
    )
    summary = dict(validation["model_summary"])
    del checkpoint, policy
    return {
        "max_plane_agents": int(runtime_args.max_agent_num),
        "max_device_agents": int(runtime_args.max_device_num),
        "model_tensor_count": int(summary["count"]),
        "model_numel": int(summary["numel"]),
        "model_sha256": str(summary["sha256"]),
        "stage3_rl_state_initialized_fresh": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "memory_canary", "canary", "wave1", "encoder_wave1",
            "critical_wave1", "ppo_gain_memory_canary",
            "ppo_gain_wave1", "ppo_gain_wave2",
            "credit_happo_memory_canary", "credit_happo_wave1",
            "gradient_conflict_memory_canary", "gradient_conflict_wave1",
            "formal",
        ),
        default="canary",
    )
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT.parent / "conda/envs/maia-hkbz-cu124-20260903/bin/python3.11",
    )
    parser.add_argument("--experiment-name", default="stage3_joint_finetune_canary_seed1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--method",
        choices=(
            "canonical", "A0", "A1", "A2", "A3",
            "B0", "B1", "B2", "B3",
            "E0", "E1", "E2", "E3",
            "C0", "C1", "C2", "C3",
            "P0", "P1", "P2", "P3",
            "Q0", "Q1", "Q2", "Q3",
            "N0", "N1", "N2", "N3",
            "G0", "G1", "G2", "G3",
        ),
        default="canonical",
        help=(
            "A0=legacy all-role ratio; A1=role-atomic ratios; "
            "A2=+role MC returns; A3=+role ValueNorm; "
            "B0=A1 reproduction; B1=+sqrt-event weights; "
            "B2=+role TD(lambda=.95); B3=B1+B2; "
            "E0=frozen B1 control; E1=immediate low-LR shared adaptation; "
            "E2=one-epoch freeze plus LR ramp; E3=E2+shared-only PCGrad; "
            "C0=E2 control; C1=critical-path potential; C2=horizon3/frontier4; "
            "C3=C1+C2; P0=legacy reward/team MC; P1=pure Cmax/team MC; "
            "P2=pure Cmax/role lambda=.90; P3=pure Cmax/heterogeneous role "
            "lambdas; Q0-Q3 form the selected-credit tail-weight/PCGrad 2x2; "
            "N0/N1 compare V versus action-conditioned Q under simultaneous "
            "PPO, while N2/N3 repeat that comparison with role-sequential PPO; "
            "G0=sum, G1=bounded norm balance, G2=norm balance plus threshold "
            "symmetric PCGrad, and G3=CAGrad on the shared encoder"
        ),
    )
    parser.add_argument(
        "--base-credit-method",
        choices=("P0", "P1", "P2", "P3"),
        default="P3",
        help="Wave-1 credit method inherited by Q0-Q3 in Wave 2.",
    )
    parser.add_argument(
        "--max-graphs-per-forward",
        dest="max_graphs_per_forward",
        type=int,
        default=GRAPHS_PER_FORWARD,
    )
    parser.add_argument(
        "--inherit-source-contracts",
        action="store_true",
        help=(
            "inherit the narrowly supported Stage2 hard-reservation and "
            "pure-team-time contracts instead of requiring the legacy soft "
            "Stage2 source"
        ),
    )
    parser.add_argument(
        "--standalone-eval",
        action="store_true",
        help=(
            "disable shard-level shared-evaluator canaries while retaining "
            "the normal local end-of-epoch validation"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_checkpoint = args.source_checkpoint.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    for required in (
        args.source_checkpoint,
        args.python,
        ROOT / "onpolicy/config/env_joint_finetune.yaml",
    ):
        if not required.exists():
            raise FileNotFoundError(f"Missing Stage3 input: {required}")
    source = validate_stage2_source(
        args.source_checkpoint,
        inherit_source_contracts=args.inherit_source_contracts,
    )
    args.request_ready_time_scale = source["request_ready_time_scale"]
    if args.inherit_source_contracts:
        args.source_planning_contract = source["planning_contract"]
        args.source_reward_contract = source["reward_contract"]
    command = build_command(args)
    source["runtime_handoff"] = validate_runtime_handoff(
        args.source_checkpoint,
        command,
        source_planning_contract=source["planning_contract"],
        source_reward_contract=source["reward_contract"],
    )
    research_method = args.method != "canonical"
    phase2_method = args.method in {"B0", "B1", "B2", "B3"}
    critical_method = args.method in {
        "C0", "C1", "C2", "C3",
        "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3",
        "N0", "N1", "N2", "N3",
        "G0", "G1", "G2", "G3",
    }
    encoder_method = args.method in {
        "E0", "E1", "E2", "E3", "C0", "C1", "C2", "C3",
        "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3",
        "N0", "N1", "N2", "N3",
        "G0", "G1", "G2", "G3",
    }
    command_freeze_epochs = int(
        command[command.index("--gnn_freeze_epochs") + 1]
    )
    command_episodes = int(command[command.index("--num_episodes") + 1])
    command_shared_schedule = command[
        command.index("--shared_actor_lr_scale_schedule") + 1
    ]
    payload = {
        "schema_version": 2,
        "training_stage": "joint_finetune",
        "training_method": (
            "shared_encoder_adaptation" if encoder_method
            else "role_clock_phase2" if phase2_method
            else "role_clock_phase1" if research_method
            else "pure_joint_ppo"
        ),
        "causal_arm": args.method,
        "profile": args.profile,
        "created_unix_time": time.time(),
        "source_stage2": source,
        "planning_contract": _runtime_planning_contract(command),
        "reward_contract": _runtime_reward_contract(command),
        "research_hypothesis": (
            (
                "strict_pre_ppo_gain"
                if args.method.startswith(("P", "Q", "N", "G"))
                else "critical_path_and_rendezvous"
            ) if critical_method else None
        ),
        "architecture_contract": {
            "shared_encoder_count": 1,
            "encoder_split": False,
            "joint_actor_roles": ["plane", "device", "transporter"],
            "joint_ppo_scope": "all",
            "supervised_updates": False,
            "source_policy_kl_penalty": False,
            "shared_encoder_frozen": command_freeze_epochs >= command_episodes,
            "shared_encoder_freeze_epochs": command_freeze_epochs,
            "shared_encoder_lr_scale_schedule": [
                float(value) for value in command_shared_schedule.split(",")
            ],
            "shared_encoder_pcgrad": args.method in {
                "E3", "Q2", "Q3", "N0", "N1", "N2", "N3"
            },
            "shared_gradient_method": _command_value(
                command, "--shared_gradient_method"
            ),
            "shared_gradient_hyperparameters": {
                "ema_beta": float(_command_value(
                    command, "--shared_grad_ema_beta"
                )),
                "norm_power": float(_command_value(
                    command, "--shared_grad_norm_power"
                )),
                "min_scale": float(_command_value(
                    command, "--shared_grad_min_scale"
                )),
                "max_scale": float(_command_value(
                    command, "--shared_grad_max_scale"
                )),
                "conflict_threshold": float(_command_value(
                    command, "--shared_grad_conflict_threshold"
                )),
                "cagrad_c": float(_command_value(
                    command, "--shared_cagrad_c"
                )),
            },
            "shared_encoder_activation_checkpoint": (
                encoder_method and args.method != "E0"
            ),
            "shared_gradient_diagnostics": (
                encoder_method and args.method != "E0"
            ),
            "role_atomic_ppo": args.method in {
                "A1", "A2", "A3", "B0", "B1", "B2", "B3",
                "E0", "E1", "E2", "E3",
                "C0", "C1", "C2", "C3",
                "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3",
                "N0", "N1", "N2", "N3",
                "G0", "G1", "G2", "G3",
            },
            "role_event_returns": "--role_event_returns" in command,
            "role_event_gae_lambda": float(_command_value(
                command, "--role_event_gae_lambda"
            )),
            "role_event_gae_lambdas": {
                role: float(_command_value(
                    command, f"--{role}_role_gae_lambda"
                ))
                for role in ("plane", "device", "transporter")
            },
            "role_loss_weighting": (
                "sqrt_event"
                if args.method in {
                    "B1", "B3", "E0", "E1", "E2", "E3",
                    "C0", "C1", "C2", "C3",
                    "P0", "P1", "P2", "P3", "Q0", "Q1", "Q2", "Q3",
                } else "fixed"
            ),
            "role_valuenorm": "--role_valuenorm" in command,
            "counterfactual_q_baseline": (
                "--counterfactual_q_baseline" in command
            ),
            "counterfactual_q_topk": int(_command_value(
                command, "--counterfactual_q_topk"
            )),
            "role_sequential_ppo": "--role_sequential_ppo" in command,
            "critic_only_calibration": {
                "actor_warmup_shards": int(_command_value(
                    command, "--actor_warmup_shards"
                )),
                "actor_hash_must_remain_unchanged": bool(
                    args.method.startswith(("P", "Q", "N", "G"))
                    and args.profile not in {
                        "ppo_gain_memory_canary",
                        "credit_happo_memory_canary",
                        "gradient_conflict_memory_canary",
                    }
                ),
            },
            "base_credit_method": (
                args.method if args.method.startswith("P")
                else args.base_credit_method
                if args.method.startswith("Q") else None
            ),
        },
        "four_lane_runtime_contract": {
            "trainer_count": 1 if args.standalone_eval else 4,
            "trainer_cuda_memory_fraction": TRAINER_CUDA_MEMORY_FRACTION,
            "shared_evaluator_count": 0 if args.standalone_eval else 1,
            "shared_evaluator_cuda_memory_fraction": (
                0.0 if args.standalone_eval else 0.06
            ),
            "ppo_updates_serialized_per_gpu": not args.standalone_eval,
            "gpu_phase_lock_scope": (
                "none" if args.standalone_eval else "ppo_update"
            ),
            "clear_cuda_cache_after_update": True,
            "max_graphs_per_forward": int(args.max_graphs_per_forward),
            "n_rollout_threads_per_trainer": profile_settings(args.profile)[
                "n_rollout_threads"
            ],
            "n_eval_rollout_threads": 10,
            "eval_partition_seed": 20260803,
            "eval_partition_stratify_by": "profile",
            "train_cases_per_epoch": int(_command_value(
                command, "--train_sampling_size"
            )),
            "train_sampling_pool_size": int(_command_value(
                command, "--train_sampling_pool_size"
            )),
        },
        "command": command,
        "shell": shlex.join(command),
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, payload)
    print(f"[Stage3Prepare] wrote {output}")
    print(payload["shell"])


if __name__ == "__main__":
    main()
