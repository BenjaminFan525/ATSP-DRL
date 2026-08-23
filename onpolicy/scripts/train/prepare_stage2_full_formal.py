#!/usr/bin/env python3
"""Build the five-arm, full-train600 Stage2 formal manifest.

The Wave-4 screen deliberately used 180 unique cases.  This formal profile
returns to the Stage-1 coverage contract: all 600 unique train cases are kept
and distribution balancing expands one deterministic epoch to 960 case slots.
Five resource methods share the same Stage-1 P5 source and validation service.
Methods with identical BC semantics are marked for exact checkpoint reuse by
the GPU0 launcher, while every PPO continuation remains independent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.scripts.train.prepare_stage2_resource_research import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_HANDOFF,
    DEFAULT_PYTHON,
    DEFAULT_SOURCE_TEACHERS,
    PLANNING_ACTOR_ACCUMULATION_TARGET_GRAPHS,
    PLANNING_CRITIC_ACCUMULATION_TARGET_GRAPHS,
    PLANNING_EVAL_THREADS,
    PLANNING_GRAPHS_PER_FORWARD,
    PLANNING_METHODS,
    PLANNING_MINI_BATCH_SIZE,
    PLANNING_ROLLOUT_THREADS,
    atomic_json,
    build_compact_teacher_index,
    configure_command,
    natural_evaluator_command,
    sha256_file,
    source_contract,
)
from onpolicy.scripts.train.run_hkbz_two_stage_pipeline import (  # noqa: E402
    build_stage2_command,
    set_option,
    set_switch,
)
from onpolicy.scripts.train.train_hkbz import select_train_case_dirs  # noqa: E402


FORMAL_PROFILE = "planning_formal_full"
FORMAL_BC_EPOCHS = 4
FORMAL_PPO_EPOCHS = 8
FORMAL_BC_ROLLOUTS = 48
FORMAL_RECOVERY_INTERVAL_SHARDS = 4
EXPECTED_SOURCE_CASES = 600
EXPECTED_CASE_SLOTS = 960
EXPECTED_UNIQUE_CASES = 600
EXPECTED_DISTRIBUTION_COUNTS = {
    "iid": 480,
    "ood_scale": 48,
    "ood_stress": 432,
}
DEFAULT_CALIBRATION_MANIFEST = (
    ROOT
    / "result/hkbz_train_logs/stage2_planning_wave4_20260822_r2/"
    "commands/planning_wave.json"
)


def _formal_method(source_id: str, *, bc_group: str, bc_leader: bool,
                   reuse_bc_from: str | None = None) -> dict:
    method = dict(PLANNING_METHODS[source_id])
    method.pop("gpu_group", None)
    method.update({
        "source_wave4_method": source_id,
        "bc_group": bc_group,
        "bc_leader": bool(bc_leader),
        "reuse_bc_from": reuse_bc_from,
    })
    return method


FORMAL_METHODS = {
    "F0_soft_control": _formal_method(
        "A3_soft_reservation",
        bc_group="heuristic_soft",
        bc_leader=True,
    ),
    "F1_hard_reservation": _formal_method(
        "A4_hard_reservation",
        bc_group="heuristic_hard",
        bc_leader=True,
    ),
    "F2_iga_flow_bc": _formal_method(
        "B1_iga_flow_bc",
        bc_group="iga_soft",
        bc_leader=True,
    ),
    "F3_wait_constraint": _formal_method(
        "B2_wait_constraint",
        bc_group="heuristic_soft",
        bc_leader=False,
        reuse_bc_from="F0_soft_control",
    ),
    "F4_iga_constraint": _formal_method(
        "B3_iga_constraint",
        bc_group="iga_soft",
        bc_leader=False,
        reuse_bc_from="F2_iga_flow_bc",
    ),
}


def full_coverage_audit(dataset_dir: Path, *, seed: int = 1) -> dict:
    case_dirs = sorted(path for path in dataset_dir.glob("case_*") if path.is_dir())
    selected, audit = select_train_case_dirs(
        case_dirs,
        mode="distribution_balanced",
        weights_spec="iid=0.50,ood_stress=0.45,ood_scale=0.05",
        seed=seed,
        max_cases=0,
        sample_size=0,
    )
    observed_counts = audit.get("selected_distribution_counts", {})
    expected = {
        "source_cases": EXPECTED_SOURCE_CASES,
        "selected_cases": EXPECTED_CASE_SLOTS,
        "unique_cases": EXPECTED_UNIQUE_CASES,
    }
    for key, value in expected.items():
        if int(audit.get(key, -1)) != value:
            raise ValueError(
                f"Full-data Stage2 contract mismatch for {key}: "
                f"{audit.get(key)!r} != {value}."
            )
    if {key: int(value) for key, value in observed_counts.items()} != (
        EXPECTED_DISTRIBUTION_COUNTS
    ):
        raise ValueError(
            "Full-data Stage2 distribution mismatch: "
            f"{observed_counts} != {EXPECTED_DISTRIBUTION_COUNTS}."
        )
    order_sha256 = hashlib.sha256(
        "\n".join(Path(path).name for path in selected).encode("utf-8")
    ).hexdigest()
    return {
        **audit,
        "dataset_dir": str(dataset_dir.resolve()),
        "sampling_seed": int(seed),
        "case_order_sha256": order_sha256,
        "n_rollout_threads": PLANNING_ROLLOUT_THREADS,
        "shards_per_epoch": math.ceil(
            len(selected) / PLANNING_ROLLOUT_THREADS
        ),
        "max_train_cases": 0,
        "train_sampling_size": 0,
    }


def configure_full_formal_command(
    base: list[str],
    method_id: str,
    method: dict,
    *,
    run_tag: str,
    teacher_dir: Path,
    teacher_index: Path,
) -> list[str]:
    command = configure_command(
        base,
        method_id,
        method,
        run_tag=run_tag,
        profile="planning_wave",
        teacher_dir=teacher_dir,
        teacher_index=teacher_index,
        low_memory=False,
    )
    set_option(
        command,
        "--experiment_name",
        f"{run_tag}_{FORMAL_PROFILE}_{method_id}_seed1",
    )
    set_option(command, "--seed", 1)
    set_option(command, "--num_episodes", FORMAL_PPO_EPOCHS)
    set_option(command, "--gnn_freeze_epochs", FORMAL_PPO_EPOCHS)
    set_option(command, "--plane_freeze_epochs", FORMAL_PPO_EPOCHS)
    set_switch(command, "--stage2_allow_shared_unfreeze", False)
    set_option(command, "--device_bc_pretrain_epochs", FORMAL_BC_EPOCHS)
    set_option(command, "--device_bc_min_rollouts_per_epoch", FORMAL_BC_ROLLOUTS)
    set_option(command, "--device_bc_max_rollouts_per_epoch", FORMAL_BC_ROLLOUTS)
    set_option(command, "--device_bc_dagger_schedule", "1.0,0.7,0.4,0.1")
    set_option(command, "--n_rollout_threads", PLANNING_ROLLOUT_THREADS)
    set_option(command, "--n_eval_rollout_threads", PLANNING_EVAL_THREADS)
    set_option(command, "--max_train_cases", 0)
    set_option(command, "--train_sampling_size", 0)
    set_option(command, "--max_eval_cases", 60)
    set_option(
        command,
        "--recovery_checkpoint_interval_shards",
        FORMAL_RECOVERY_INTERVAL_SHARDS,
    )
    set_switch(command, "--skip_pre_ppo_eval", False)
    set_switch(command, "--skip_epoch_eval", False)
    return command


def calibration_contract(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration = payload.get("gpu_calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"Missing Wave-4 GPU calibration in {path}.")
    if int(calibration.get("formal_graphs_per_forward", -1)) != (
        PLANNING_GRAPHS_PER_FORWARD
    ):
        raise ValueError("GPU calibration graph geometry differs from formal run.")
    if int(calibration.get("trainer_count_per_gpu", -1)) != 5:
        raise ValueError("GPU calibration did not exercise five trainers per GPU.")
    if int(calibration.get("projected_headroom_mib", -1)) <= 0:
        raise ValueError("GPU calibration has no positive safety headroom.")
    return {
        **calibration,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "reuse_reason": (
            "same five frozen-plane/frozen-encoder trainers and identical "
            "1000-graph forward; full coverage changes sequential shard count, "
            "not concurrent CUDA batch geometry"
        ),
    }


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--source-seed", type=int, default=3)
    parser.add_argument(
        "--source-teacher-dir", type=Path, default=DEFAULT_SOURCE_TEACHERS
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=DEFAULT_CALIBRATION_MANIFEST,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    compact_dir = suite_dir / "artifacts/resource_iga1800_compact"
    teacher_index = suite_dir / "artifacts/resource_iga1800_index.json"
    teacher_contract = build_compact_teacher_index(
        args.source_teacher_dir.resolve(),
        args.dataset_dir.resolve(),
        compact_dir,
        teacher_index,
    )
    source, source_command = source_contract(
        args.handoff.resolve(), args.source_seed
    )
    if source["sha256"] != teacher_contract["frozen_plane_checkpoint_sha256"]:
        raise ValueError(
            "Stage1 source checkpoint differs from the resource IGA teacher."
        )
    coverage = full_coverage_audit(args.dataset_dir.resolve(), seed=1)

    commands = {}
    for method_id, method in FORMAL_METHODS.items():
        base = build_stage2_command(
            source["path"],
            run_tag=f"{args.run_tag}_{FORMAL_PROFILE}_{method_id}",
            seed=1,
            bc_epochs=FORMAL_BC_EPOCHS,
            ppo_epochs=FORMAL_PPO_EPOCHS,
            ppo_epoch=2,
            bc_min_labels=64,
            bc_min_rollouts=FORMAL_BC_ROLLOUTS,
            bc_max_rollouts=FORMAL_BC_ROLLOUTS,
            source_command=source_command,
            python=args.python.resolve(),
        )
        command = configure_full_formal_command(
            base,
            method_id,
            method,
            run_tag=args.run_tag,
            teacher_dir=compact_dir,
            teacher_index=teacher_index,
        )
        commands[method_id] = {
            "argv": command,
            "shell": shlex.join(command),
            **method,
        }

    evaluator_key = "F0_soft_control"
    manifest = {
        "schema_version": 1,
        "research_stage": "stage2_resource_joint",
        "profile": FORMAL_PROFILE,
        "run_tag": args.run_tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_revision(),
        "source": source,
        "teacher_index_path": str(teacher_index),
        "teacher_index_sha256": sha256_file(teacher_index),
        "teacher_case_count": int(teacher_contract["case_count"]),
        "data_contract": coverage,
        "training_contract": {
            "seed": 1,
            "bc_epochs": FORMAL_BC_EPOCHS,
            "ppo_epochs": FORMAL_PPO_EPOCHS,
            "ppo_epoch": 2,
            "bc_rollouts_per_epoch": FORMAL_BC_ROLLOUTS,
            "recovery_interval_shards": FORMAL_RECOVERY_INTERVAL_SHARDS,
            "plane_frozen_for_all_ppo_epochs": True,
            "shared_encoder_frozen_for_all_ppo_epochs": True,
        },
        "cpu_contract": {
            "host_logical_cpus": 144,
            "maximum_experiment_logical_cpus": 72,
            "trainer_logical_cpus_each": 12,
            "shared_evaluator_logical_cpus": 10,
            "os_reserved_logical_cpus": 2,
            "physical_core_isolation": True,
            "numa_node": 0,
        },
        "gpu_contract": {
            "gpu": 0,
            "trainer_count": 5,
            "shared_evaluator_count": 1,
            "max_graphs_per_forward": PLANNING_GRAPHS_PER_FORWARD,
            "calibration": calibration_contract(
                args.calibration_manifest.resolve()
            ),
        },
        "throughput_profile": {
            "n_rollout_threads": PLANNING_ROLLOUT_THREADS,
            "n_eval_rollout_threads": PLANNING_EVAL_THREADS,
            "mini_batch_size": PLANNING_MINI_BATCH_SIZE,
            "data_chunk_length": 50,
            "max_graphs_per_forward": PLANNING_GRAPHS_PER_FORWARD,
            "critic_grad_accumulation_target_graphs": (
                PLANNING_CRITIC_ACCUMULATION_TARGET_GRAPHS
            ),
            "actor_grad_accumulation_target_graphs": (
                PLANNING_ACTOR_ACCUMULATION_TARGET_GRAPHS
            ),
        },
        "evaluator_command": natural_evaluator_command(
            commands[evaluator_key]["argv"]
        ),
        "evaluator_source_method": evaluator_key,
        "methods": FORMAL_METHODS,
        "commands": commands,
    }
    atomic_json(args.output.resolve(), manifest)
    print(json.dumps({
        "manifest": str(args.output.resolve()),
        "methods": list(commands),
        "unique_cases": coverage["unique_cases"],
        "case_slots": coverage["selected_cases"],
        "shards_per_epoch": coverage["shards_per_epoch"],
        "bc_leaders": [
            method_id for method_id, method in FORMAL_METHODS.items()
            if method["bc_leader"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
