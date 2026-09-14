#!/usr/bin/env python3
"""Evaluate a Stage2 checkpoint through Stage3 and compare case by case."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.config.config import get_config  # noqa: E402
from onpolicy.scripts.train.train_hkbz import (  # noqa: E402
    apply_formal_safe_pipeline_manifest,
    make_eval_env,
    parse_args as parse_training_args,
)


DEFAULT_MANIFEST = ROOT / (
    "result/hkbz_train_logs/stage3_joint_rl_20260823_r1/commands/canary.json"
)
DEFAULT_REFERENCE = ROOT / (
    "onpolicy/scripts/results/HKBZ/simple/gnn_mappo/"
    "stage2_planning_wave4_20260822_r2_planning_wave_"
    "B2_wait_constraint_seed1/run1/evaluations/epoch_2.json"
)
DEFAULT_OUTPUT_DIR = ROOT / (
    "result/hkbz_train_logs/stage3_joint_rl_20260823_r1/"
    "handoff_regression_s2_b2"
)

CASE_METRICS = (
    "makespan",
    "finish_steps",
    "total_relocations",
    "max_no_progress",
    "resource_wait_seconds",
    "resource_critical_wait_seconds",
    "resource_wait_p95_seconds",
    "resource_predicted_lateness_seconds",
    "resource_early_arrival_seconds",
    "resource_wait_before_dispatch_seconds",
    "resource_wait_travel_seconds",
    "resource_wait_post_arrival_seconds",
    "resource_lookahead_dispatch_count",
)
CASE_FLAGS = ("completed", "cycle_terminated", "timeout")
SUMMARY_METRICS = (
    "eval_makespan",
    "eval_raw_makespan",
    "eval_iid_makespan",
    "eval_composite_makespan",
    "eval_tail_makespan",
    "eval_selection_score",
    "eval_completed_count",
    "eval_completion_rate",
    "eval_timeout_count",
    "eval_cycle_count",
    "eval_mean_steps",
    "eval_max_no_progress",
    "eval_mean_relocations",
)


def _atomic_json(path: Path, payload: dict) -> None:
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


def _flag_value(command: list[str], flag: str) -> str:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one {flag}, found {len(positions)}")
    index = positions[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"Missing value for {flag}")
    return command[index + 1]


def _case_map(payload: dict) -> dict[str, dict]:
    records = payload.get("cases")
    if not isinstance(records, list) or not records:
        raise ValueError("Evaluation artifact contains no case records.")
    result = {}
    for record in records:
        key = str(record.get("case_key", ""))
        if not key or key in result:
            raise ValueError(f"Invalid or duplicate case_key: {key!r}")
        result[key] = record
    return result


def _compare(reference: dict, candidate: dict, tolerance: float) -> dict:
    reference_cases = _case_map(reference)
    candidate_cases = _case_map(candidate)
    reference_keys = set(reference_cases)
    candidate_keys = set(candidate_cases)
    missing = sorted(reference_keys - candidate_keys)
    extra = sorted(candidate_keys - reference_keys)

    identity_mismatches = []
    field_results = {}
    changed_cases = set(missing) | set(extra)
    for field in CASE_METRICS:
        deltas = []
        changed = []
        for case_key in sorted(reference_keys & candidate_keys):
            before = float(reference_cases[case_key][field])
            after = float(candidate_cases[case_key][field])
            delta = after - before
            deltas.append(delta)
            if abs(delta) > tolerance:
                changed.append(case_key)
                changed_cases.add(case_key)
        field_results[field] = {
            "changed_case_count": len(changed),
            "max_abs_delta": max((abs(value) for value in deltas), default=0.0),
            "mean_delta_s3_minus_s2": float(np.mean(deltas)) if deltas else 0.0,
            "changed_cases": changed,
        }

    for case_key in sorted(reference_keys & candidate_keys):
        before = reference_cases[case_key]
        after = candidate_cases[case_key]
        for field in ("case_sha256", *CASE_FLAGS):
            if before.get(field) != after.get(field):
                identity_mismatches.append({
                    "case_key": case_key,
                    "field": field,
                    "s2": before.get(field),
                    "s3": after.get(field),
                })
                changed_cases.add(case_key)

    reference_summary = reference.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    summary_deltas = {}
    for field in SUMMARY_METRICS:
        before = float(reference_summary[field])
        after = float(candidate_summary[field])
        summary_deltas[field] = {
            "s2": before,
            "s3": after,
            "delta_s3_minus_s2": after - before,
        }

    equivalent = (
        not missing
        and not extra
        and not identity_mismatches
        and all(
            item["changed_case_count"] == 0
            for item in field_results.values()
        )
    )
    return {
        "equivalent_within_tolerance": equivalent,
        "absolute_tolerance": tolerance,
        "reference_case_count": len(reference_cases),
        "candidate_case_count": len(candidate_cases),
        "missing_case_keys": missing,
        "extra_case_keys": extra,
        "changed_case_count": len(changed_cases),
        "changed_case_keys": sorted(changed_cases),
        "identity_mismatches": identity_mismatches,
        "case_metric_comparison": field_results,
        "summary_comparison": summary_deltas,
    }


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="stage3_valid_s2_b2")
    parser.add_argument("--max-eval-cases", type=int, default=60)
    parser.add_argument("--n-eval-rollout-threads", type=int, default=10)
    parser.add_argument("--eval-partition-seed", type=int, default=20260803)
    parser.add_argument(
        "--eval-partition-stratify-by",
        choices=("", "distribution", "profile"),
        default="profile",
    )
    parser.add_argument("--evaluation-tau", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    cli = _parse_cli()
    cli.stage3_manifest = cli.stage3_manifest.expanduser().resolve()
    cli.reference = cli.reference.expanduser().resolve()
    cli.output_dir = cli.output_dir.expanduser().resolve()
    for required in (cli.stage3_manifest, cli.reference):
        if not required.is_file():
            raise FileNotFoundError(required)
    if cli.max_eval_cases <= 0 or cli.n_eval_rollout_threads <= 0:
        raise ValueError("Evaluation case and worker counts must be positive.")
    if cli.tolerance < 0.0:
        raise ValueError("Comparison tolerance must be non-negative.")

    manifest = json.loads(cli.stage3_manifest.read_text(encoding="utf-8"))
    command = [str(value) for value in manifest.get("command", ())]
    if len(command) < 2:
        raise ValueError("Stage3 manifest has no training command.")
    if _flag_value(command, "--training_stage") != "joint_finetune":
        raise ValueError("The supplied manifest is not a Stage3 hand-off.")
    checkpoint = Path(_flag_value(command, "--checkpoint_dir")).resolve()
    source_record = manifest.get("source_stage2", {})
    if checkpoint != Path(str(source_record.get("path", ""))).resolve():
        raise ValueError("Manifest command and source_stage2 checkpoint differ.")

    all_args = parse_training_args(command[2:], get_config())
    apply_formal_safe_pipeline_manifest(all_args, repository_root=ROOT)
    all_args.use_wandb = False
    all_args.use_eval = True
    all_args.checkpoint_dir = str(checkpoint)
    all_args.selection_checkpoint_dir = None
    all_args.n_eval_rollout_threads = int(cli.n_eval_rollout_threads)
    all_args.n_rollout_threads = int(cli.n_eval_rollout_threads)
    all_args.max_eval_cases = int(cli.max_eval_cases)
    all_args.eval_case_offset = 0
    all_args.eval_partition_seed = int(cli.eval_partition_seed)
    all_args.eval_partition_stratify_by = cli.eval_partition_stratify_by
    all_args.evaluation_tau = float(cli.evaluation_tau)
    all_args.plane_bc_pretrain_epochs = 0
    all_args.device_bc_pretrain_epochs = 0

    torch.multiprocessing.set_sharing_strategy(
        all_args.torch_mp_sharing_strategy
    )
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device(str(all_args.device))
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads(cli.reference.read_text(encoding="utf-8"))
    eval_envs = None
    runner = None
    started = time.monotonic()
    try:
        eval_envs, case_counts = make_eval_env(all_args)
        with Path(all_args.ac_config).open("r", encoding="utf-8") as source:
            ac_config = yaml.safe_load(source)
        from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner

        runner = HKBZ_Runner({
            "all_args": all_args,
            "envs": eval_envs,
            "eval_envs": eval_envs,
            "device": device,
            "run_dir": cli.output_dir,
            "ac_config": ac_config,
            "num_agents": all_args.max_agent_num + all_args.max_device_num,
            "num_envs": max(case_counts),
            "eval_case_counts": case_counts,
            "eval_env_factory": None,
            "release_eval_envs_after_eval": False,
            "evaluation_only": True,
        })
        runner.eval(evaluation_label=cli.label)
    finally:
        if eval_envs is not None:
            eval_envs.close()
        writer = getattr(runner, "writter", None)
        if writer is not None:
            writer.close()

    candidate_path = cli.output_dir / "evaluations" / f"{cli.label}.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    comparison = _compare(reference, candidate, float(cli.tolerance))
    comparison.update({
        "stage3_manifest": str(cli.stage3_manifest),
        "source_stage2_checkpoint": str(checkpoint),
        "reference_evaluation": str(cli.reference),
        "candidate_evaluation": str(candidate_path),
        "device": str(device),
        "evaluation_tau": float(cli.evaluation_tau),
        "eval_partition_seed": int(cli.eval_partition_seed),
        "eval_partition_stratify_by": cli.eval_partition_stratify_by,
        "elapsed_seconds": float(time.monotonic() - started),
    })
    comparison_path = cli.output_dir / "comparison.json"
    _atomic_json(comparison_path, comparison)
    print(
        "[Stage3HandoffEval] "
        f"equivalent={comparison['equivalent_within_tolerance']} "
        f"cases={comparison['candidate_case_count']} "
        f"changed={comparison['changed_case_count']} "
        f"report={comparison_path}",
        flush=True,
    )
    return 0 if comparison["equivalent_within_tolerance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
