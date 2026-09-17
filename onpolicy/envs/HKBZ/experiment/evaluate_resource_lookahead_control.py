#!/usr/bin/env python3
"""Paired frozen-plane A/B test of demand versus lookahead Hungarian dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from onpolicy.envs.HKBZ.experiment.eval_common import list_case_folders
from onpolicy.envs.HKBZ.experiment.evaluate_resource_iga_ablation import (
    ARM_BACKENDS,
    CrossCaseBatchedEvaluator,
    FrozenPlaneEvaluator,
    _atomic_json,
    _metadata,
    _model_digest,
    _sha256_file,
    _validate_stage1_handoff,
    load_frozen_policy,
)


ROOT = Path(__file__).resolve().parents[4]
ARMS = {
    "A_demand_hungarian": False,
    "B_lookahead_hungarian": True,
}


def _derived_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _chunks(values: Sequence[Path], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _bootstrap_mean_ci(values: Sequence[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return [math.nan, math.nan]
    rng = np.random.default_rng(int(seed))
    samples = np.empty(100000, dtype=np.float64)
    for start in range(0, samples.size, 5000):
        count = min(5000, samples.size - start)
        indices = rng.integers(0, array.size, size=(count, array.size))
        samples[start : start + count] = array[indices].mean(axis=1)
    return [
        float(value) for value in np.quantile(samples, (0.025, 0.975))
    ]


def _paired_effect(
    rows: Sequence[Mapping],
    legacy_key: str,
    lookahead_key: str,
    *,
    seed: int,
) -> dict:
    legacy = np.asarray(
        [float(row[legacy_key]) for row in rows], dtype=np.float64
    )
    lookahead = np.asarray(
        [float(row[lookahead_key]) for row in rows], dtype=np.float64
    )
    delta = lookahead - legacy
    legacy_mean = float(legacy.mean()) if legacy.size else math.nan
    lookahead_mean = float(lookahead.mean()) if lookahead.size else math.nan
    return {
        "case_count": int(delta.size),
        "legacy_mean": legacy_mean,
        "lookahead_mean": lookahead_mean,
        "mean_delta_lookahead_minus_legacy": (
            float(delta.mean()) if delta.size else math.nan
        ),
        "median_delta_lookahead_minus_legacy": (
            float(np.median(delta)) if delta.size else math.nan
        ),
        "relative_delta_percent_of_legacy_mean": (
            100.0 * float(delta.mean()) / legacy_mean
            if delta.size and abs(legacy_mean) > 1e-12 else math.nan
        ),
        "bootstrap_mean_delta_95ci": _bootstrap_mean_ci(delta, seed),
        "lookahead_wins": int((delta < -1e-9).sum()),
        "ties": int((np.abs(delta) <= 1e-9).sum()),
        "legacy_wins": int((delta > 1e-9).sum()),
    }


def _delta_association(rows: Sequence[Mapping]) -> dict:
    cmax = np.asarray(
        [float(row["delta_makespan"]) for row in rows], dtype=np.float64
    )
    wait = np.asarray(
        [float(row["delta_resource_wait_seconds"]) for row in rows],
        dtype=np.float64,
    )
    correlation = math.nan
    if cmax.size > 1 and np.std(cmax) > 1e-12 and np.std(wait) > 1e-12:
        correlation = float(np.corrcoef(cmax, wait)[0, 1])
    return {
        "case_count": int(cmax.size),
        "pearson_correlation_delta_wait_vs_delta_makespan": correlation,
        "both_improved": int(((cmax < 0.0) & (wait < 0.0)).sum()),
        "wait_improved_makespan_not_improved": int(
            ((wait < 0.0) & (cmax >= 0.0)).sum()
        ),
        "makespan_improved_wait_not_improved": int(
            ((cmax < 0.0) & (wait >= 0.0)).sum()
        ),
        "neither_improved": int(((cmax >= 0.0) & (wait >= 0.0)).sum()),
    }


def _arm_summary(records: Sequence[Mapping]) -> dict:
    makespans = [float(record["makespan"]) for record in records]
    waits = [record["aircraft_resource_wait"] for record in records]
    per_aircraft_waits = [
        float(value)
        for item in waits
        for value in item.get("per_aircraft_wait_seconds", {}).values()
    ]
    total_wait = sum(float(item["total_wait_seconds"]) for item in waits)
    aircraft_count = sum(int(item["aircraft_count"]) for item in waits)
    positive_aircraft = sum(
        int(item["aircraft_with_positive_wait_count"]) for item in waits
    )
    category_totals: dict[str, float] = {}
    phase_totals: dict[str, float] = {}
    for item in waits:
        for name, metrics in item.get("by_category", {}).items():
            category_totals[name] = category_totals.get(name, 0.0) + float(
                metrics["total_wait_seconds"]
            )
        for name, metrics in item.get("by_phase", {}).items():
            phase_totals[name] = phase_totals.get(name, 0.0) + float(
                metrics["total_wait_seconds"]
            )
    return {
        "case_count": len(records),
        "mean_makespan": float(statistics.mean(makespans)),
        "median_makespan": float(statistics.median(makespans)),
        "total_aircraft_resource_wait_seconds": float(total_wait),
        "mean_case_aircraft_resource_wait_seconds": float(
            statistics.mean(
                float(item["total_wait_seconds"]) for item in waits
            )
        ),
        "median_case_aircraft_resource_wait_seconds": float(
            statistics.median(
                float(item["total_wait_seconds"]) for item in waits
            )
        ),
        "aircraft_count": int(aircraft_count),
        "mean_resource_wait_seconds_per_aircraft": (
            float(total_wait / aircraft_count) if aircraft_count else 0.0
        ),
        "median_resource_wait_seconds_per_aircraft": (
            float(np.median(per_aircraft_waits))
            if per_aircraft_waits else 0.0
        ),
        "p95_resource_wait_seconds_per_aircraft": (
            float(np.quantile(per_aircraft_waits, 0.95))
            if per_aircraft_waits else 0.0
        ),
        "max_resource_wait_seconds_per_aircraft": (
            float(max(per_aircraft_waits)) if per_aircraft_waits else 0.0
        ),
        "aircraft_with_positive_wait_count": int(positive_aircraft),
        "zero_wait_aircraft_count": int(aircraft_count - positive_aircraft),
        "zero_wait_aircraft_fraction": (
            float(aircraft_count - positive_aircraft) / aircraft_count
            if aircraft_count else 1.0
        ),
        "fully_eliminated_case_count": sum(
            bool(item["fully_eliminated"]) for item in waits
        ),
        "fully_eliminated_across_panel": bool(total_wait <= 1e-9),
        "wait_seconds_by_category": dict(sorted(category_totals.items())),
        "wait_seconds_by_phase": dict(sorted(phase_totals.items())),
    }


def _load_case_record(output_dir: Path, arm: str, case: str) -> dict:
    path = output_dir / "cases" / arm / f"{case}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_component_effects(
    records: Mapping[str, Sequence[Mapping]],
    component_key: str,
) -> dict:
    names = sorted(
        {
            str(name)
            for arm_records in records.values()
            for record in arm_records
            for name in record["aircraft_resource_wait"]
            .get(component_key, {})
        }
    )
    legacy_by_case = {
        record["case"]: record
        for record in records["A_demand_hungarian"]
    }
    lookahead_by_case = {
        record["case"]: record
        for record in records["B_lookahead_hungarian"]
    }
    effects = {}
    for name in names:
        rows = []
        for case in sorted(legacy_by_case):
            legacy = legacy_by_case[case]
            lookahead = lookahead_by_case[case]
            rows.append(
                {
                    "legacy": float(
                        legacy["aircraft_resource_wait"]
                        .get(component_key, {})
                        .get(name, {})
                        .get("total_wait_seconds", 0.0)
                    ),
                    "lookahead": float(
                        lookahead["aircraft_resource_wait"]
                        .get(component_key, {})
                        .get(name, {})
                        .get("total_wait_seconds", 0.0)
                    ),
                }
            )
        effects[name] = _paired_effect(
            rows,
            "legacy",
            "lookahead",
            seed=_derived_seed(component_key, name),
        )
    return effects


def _lookahead_trace_summary(output_dir: Path, cases: Sequence[str]) -> dict:
    request_observations = 0
    selected_dispatches = 0
    dispatch_records = []
    device_type_counts: dict[str, int] = {}
    for case in cases:
        path = (
            output_dir
            / "trajectories"
            / "B_lookahead_hungarian"
            / f"{case}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        for event in payload.get("decision_trace", []):
            requests = {
                int(request["id"]): request
                for request in event.get("requests", [])
            }
            request_observations += sum(
                bool(request.get("is_lookahead", False))
                for request in requests.values()
            )
            for decision in event.get("resource_decisions", []):
                selected = requests.get(
                    int(decision.get("selected_request_id") or 0)
                )
                if selected is not None and selected.get(
                    "is_lookahead", False
                ):
                    selected_dispatches += 1
        for record in payload.get("resource_trajectory", []):
            if not record.get("is_lookahead", False):
                continue
            lead = float(record.get("lead_time_at_dispatch", 0.0))
            travel = float(record.get("trans_time", 0.0))
            device_type = str(record.get("device_type") or "unknown")
            device_type_counts[device_type] = (
                device_type_counts.get(device_type, 0) + 1
            )
            dispatch_records.append(
                {
                    "lead": lead,
                    "travel": travel,
                    "earliness": max(0.0, lead - travel),
                    "lateness": max(0.0, travel - lead),
                }
            )
    count = len(dispatch_records)
    return {
        "case_count": len(cases),
        "lookahead_request_observations": int(request_observations),
        "selected_lookahead_dispatches": int(selected_dispatches),
        "trajectory_lookahead_dispatches": int(count),
        "mean_lookahead_dispatches_per_case": (
            float(count / len(cases)) if cases else 0.0
        ),
        "arrival_on_or_before_need_count": sum(
            item["travel"] <= item["lead"] + 1e-9
            for item in dispatch_records
        ),
        "arrival_after_need_count": sum(
            item["travel"] > item["lead"] + 1e-9
            for item in dispatch_records
        ),
        "mean_lead_time_seconds": (
            float(statistics.mean(item["lead"] for item in dispatch_records))
            if dispatch_records else 0.0
        ),
        "mean_travel_time_seconds": (
            float(
                statistics.mean(item["travel"] for item in dispatch_records)
            )
            if dispatch_records else 0.0
        ),
        "mean_arrival_earliness_seconds": (
            float(
                statistics.mean(
                    item["earliness"] for item in dispatch_records
                )
            )
            if dispatch_records else 0.0
        ),
        "mean_arrival_lateness_seconds": (
            float(
                statistics.mean(
                    item["lateness"] for item in dispatch_records
                )
            )
            if dispatch_records else 0.0
        ),
        "dispatches_by_device_type": dict(sorted(device_type_counts.items())),
    }


def _wait_effects_by_job(output_dir: Path, cases: Sequence[str]) -> dict:
    totals: dict[str, dict[str, dict[str, float]]] = {
        arm: {} for arm in ARMS
    }
    job_codes = set()
    for arm in ARMS:
        for case in cases:
            path = output_dir / "trajectories" / arm / f"{case}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_totals: dict[str, float] = {}
            for event in payload["aircraft_resource_wait"].get("events", []):
                job_code = str(event.get("job_code") or "unknown")
                job_codes.add(job_code)
                case_totals[job_code] = case_totals.get(job_code, 0.0) + float(
                    event.get("wait_seconds", 0.0)
                )
            totals[arm][case] = case_totals
    effects = {}
    for job_code in sorted(job_codes):
        rows = [
            {
                "legacy": totals["A_demand_hungarian"][case].get(
                    job_code, 0.0
                ),
                "lookahead": totals["B_lookahead_hungarian"][case].get(
                    job_code, 0.0
                ),
            }
            for case in cases
        ]
        effects[job_code] = _paired_effect(
            rows,
            "legacy",
            "lookahead",
            seed=_derived_seed("job_wait", job_code),
        )
    return effects


def _summarize(output_dir: Path, cases: Sequence[str]) -> dict:
    records = {
        arm: [_load_case_record(output_dir, arm, case) for case in cases]
        for arm in ARMS
    }
    legacy_by_case = {
        record["case"]: record for record in records["A_demand_hungarian"]
    }
    lookahead_by_case = {
        record["case"]: record
        for record in records["B_lookahead_hungarian"]
    }
    rows = []
    for case in cases:
        legacy = legacy_by_case[case]
        lookahead = lookahead_by_case[case]
        rows.append(
            {
                "case": case,
                "profile": legacy.get("profile"),
                "distribution": legacy.get("distribution"),
                "legacy_makespan": float(legacy["makespan"]),
                "lookahead_makespan": float(lookahead["makespan"]),
                "delta_makespan": float(lookahead["makespan"])
                - float(legacy["makespan"]),
                "legacy_resource_wait_seconds": float(
                    legacy["aircraft_resource_wait"]["total_wait_seconds"]
                ),
                "lookahead_resource_wait_seconds": float(
                    lookahead["aircraft_resource_wait"]["total_wait_seconds"]
                ),
                "delta_resource_wait_seconds": float(
                    lookahead["aircraft_resource_wait"]["total_wait_seconds"]
                )
                - float(
                    legacy["aircraft_resource_wait"]["total_wait_seconds"]
                ),
                "legacy_zero_wait_aircraft_fraction": float(
                    legacy["aircraft_resource_wait"][
                        "zero_wait_aircraft_fraction"
                    ]
                ),
                "lookahead_zero_wait_aircraft_fraction": float(
                    lookahead["aircraft_resource_wait"][
                        "zero_wait_aircraft_fraction"
                    ]
                ),
            }
        )

    paired = {
        "makespan": _paired_effect(
            rows,
            "legacy_makespan",
            "lookahead_makespan",
            seed=20260821,
        ),
        "aircraft_resource_wait_seconds": _paired_effect(
            rows,
            "legacy_resource_wait_seconds",
            "lookahead_resource_wait_seconds",
            seed=20260822,
        ),
    }
    grouped = {}
    for group_key in ("distribution", "profile"):
        grouped[group_key] = {}
        for name in sorted({str(row[group_key]) for row in rows}):
            selected = [row for row in rows if str(row[group_key]) == name]
            grouped[group_key][name] = {
                "makespan": _paired_effect(
                    selected,
                    "legacy_makespan",
                    "lookahead_makespan",
                    seed=_derived_seed(group_key, name, "cmax"),
                ),
                "aircraft_resource_wait_seconds": _paired_effect(
                    selected,
                    "legacy_resource_wait_seconds",
                    "lookahead_resource_wait_seconds",
                    seed=_derived_seed(group_key, name, "wait"),
                ),
            }

    legacy_total = sum(
        row["legacy_resource_wait_seconds"] for row in rows
    )
    lookahead_total = sum(
        row["lookahead_resource_wait_seconds"] for row in rows
    )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "contrast": (
            "B_lookahead_hungarian minus A_demand_hungarian; negative is better"
        ),
        "case_count": len(cases),
        "arm_summaries": {
            arm: _arm_summary(records[arm]) for arm in ARMS
        },
        "paired_effects": paired,
        "paired_delta_association": _delta_association(rows),
        "paired_effects_by_wait_category": _wait_component_effects(
            records, "by_category"
        ),
        "paired_effects_by_wait_phase": _wait_component_effects(
            records, "by_phase"
        ),
        "paired_effects_by_wait_job": _wait_effects_by_job(
            output_dir, cases
        ),
        "paired_effects_by_group": grouped,
        "lookahead_trace": _lookahead_trace_summary(output_dir, cases),
        "resource_wait_elimination": {
            "legacy_total_wait_seconds": float(legacy_total),
            "lookahead_total_wait_seconds": float(lookahead_total),
            "eliminated_wait_seconds": float(legacy_total - lookahead_total),
            "eliminated_fraction_of_legacy": (
                float((legacy_total - lookahead_total) / legacy_total)
                if legacy_total > 1e-9 else 0.0
            ),
            "fully_eliminated": bool(lookahead_total <= 1e-9),
        },
        "cases": rows,
        "completed_unix_time": time.time(),
    }
    _atomic_json(output_dir / "summary.json", payload)

    cmax = paired["makespan"]
    wait = paired["aircraft_resource_wait_seconds"]
    association = payload["paired_delta_association"]
    lines = [
        "# Demand versus lookahead Hungarian control",
        "",
        "The frozen plane policy, dataset, tau and Hungarian resource backend are matched.",
        "Negative paired deltas favor lookahead.",
        "",
        "| Arm | Mean Cmax | Total resource wait (s) | Mean wait/aircraft (s) | Zero-wait aircraft | Zero-wait cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        item = payload["arm_summaries"][arm]
        lines.append(
            f"| {arm} | {item['mean_makespan']:.2f} | "
            f"{item['total_aircraft_resource_wait_seconds']:.2f} | "
            f"{item['mean_resource_wait_seconds_per_aircraft']:.2f} | "
            f"{100.0 * item['zero_wait_aircraft_fraction']:.2f}% | "
            f"{item['fully_eliminated_case_count']}/{item['case_count']} |"
        )
    cmax_low, cmax_high = cmax["bootstrap_mean_delta_95ci"]
    wait_low, wait_high = wait["bootstrap_mean_delta_95ci"]
    lines.extend(
        [
            "",
            "| Paired metric | Mean delta | 95% bootstrap CI | W/T/L |",
            "| --- | ---: | ---: | ---: |",
            f"| Cmax | {cmax['mean_delta_lookahead_minus_legacy']:+.2f} | "
            f"[{cmax_low:+.2f}, {cmax_high:+.2f}] | "
            f"{cmax['lookahead_wins']}/{cmax['ties']}/{cmax['legacy_wins']} |",
            f"| Aircraft resource wait (s/case) | "
            f"{wait['mean_delta_lookahead_minus_legacy']:+.2f} | "
            f"[{wait_low:+.2f}, {wait_high:+.2f}] | "
            f"{wait['lookahead_wins']}/{wait['ties']}/{wait['legacy_wins']} |",
            "",
            f"Across cases, Pearson corr(delta wait, delta Cmax) = "
            f"{association['pearson_correlation_delta_wait_vs_delta_makespan']:.3f}; "
            f"both improved in {association['both_improved']}/{len(cases)} cases.",
            "",
            "Resource wait is fully eliminated only when the lookahead total is zero; "
            "runway/site queues and voluntary staging are excluded from this metric.",
            "",
            "## Wait decomposition",
            "",
            "| Category | Demand mean/case (s) | Lookahead mean/case (s) | "
            "Paired delta (s) | 95% CI | W/T/L |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, effect in payload["paired_effects_by_wait_category"].items():
        low, high = effect["bootstrap_mean_delta_95ci"]
        lines.append(
            f"| {name} | {effect['legacy_mean']:.2f} | "
            f"{effect['lookahead_mean']:.2f} | "
            f"{effect['mean_delta_lookahead_minus_legacy']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | "
            f"{effect['lookahead_wins']}/{effect['ties']}/"
            f"{effect['legacy_wins']} |"
        )
    lines.extend(
        [
            "",
            "| Job | Demand mean/case (s) | Lookahead mean/case (s) | "
            "Paired delta (s) | W/T/L |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    job_effects = sorted(
        payload["paired_effects_by_wait_job"].items(),
        key=lambda item: item[1]["legacy_mean"],
        reverse=True,
    )
    for name, effect in job_effects:
        lines.append(
            f"| {name} | {effect['legacy_mean']:.2f} | "
            f"{effect['lookahead_mean']:.2f} | "
            f"{effect['mean_delta_lookahead_minus_legacy']:+.2f} | "
            f"{effect['lookahead_wins']}/{effect['ties']}/"
            f"{effect['legacy_wins']} |"
        )
    trace = payload["lookahead_trace"]
    lines.extend(
        [
            "",
            "## Lookahead dispatch trace",
            "",
            f"- Request observations: {trace['lookahead_request_observations']}",
            f"- Executed early dispatches: "
            f"{trace['trajectory_lookahead_dispatches']} "
            f"({trace['mean_lookahead_dispatches_per_case']:.2f}/case)",
            f"- Arrived on/before need: "
            f"{trace['arrival_on_or_before_need_count']}; after need: "
            f"{trace['arrival_after_need_count']}",
            f"- Mean lead/travel/earliness/lateness: "
            f"{trace['mean_lead_time_seconds']:.2f} / "
            f"{trace['mean_travel_time_seconds']:.2f} / "
            f"{trace['mean_arrival_earliness_seconds']:.2f} / "
            f"{trace['mean_arrival_lateness_seconds']:.2f} seconds",
            f"- Dispatches by device type: "
            f"{json.dumps(trace['dispatches_by_device_type'], sort_keys=True)}",
            "",
        ]
    )
    (output_dir / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT
        / "onpolicy/envs/HKBZ/dataset/"
        "fjsp_v3_resource_joint_eval_s20260811/joint/tune",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo/"
        "stage1_departure_reward_formal_dual_20260816_r1_formal_"
        "P5_team_time_potential_fixed_seed3/run1/models/checkpoint_Best.pt",
    )
    parser.add_argument(
        "--source-command",
        type=Path,
        default=ROOT
        / "result/hkbz_train_logs/stage1_departure_reward_formal_dual_"
        "20260816_r1/commands/formal_P5.json",
    )
    parser.add_argument(
        "--source-command-key",
        default="P5_team_time_potential_fixed_seed3",
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        default=ROOT / "onpolicy/config/stage1_m2_handoff.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-tau", type=float, default=0.3)
    parser.add_argument("--device-lookahead-safety-margin", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--async-graph-clone-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.max_cases <= 0 or args.batch_size <= 0:
        parser.error("max-steps, max-cases and batch-size must be positive")
    if args.device_lookahead_safety_margin < 0.0:
        parser.error("device-lookahead-safety-margin must be non-negative")
    return args


def run(args) -> dict:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    command_path = args.source_command.resolve()
    handoff = _validate_stage1_handoff(
        args.handoff.resolve(),
        checkpoint_path,
        command_path,
        args.source_command_key,
    )
    cases = list_case_folders(str(dataset_dir), args.max_cases)
    case_paths = [dataset_dir / case for case in cases]
    policy, policy_args, _checkpoint = load_frozen_policy(
        checkpoint_path,
        command_path,
        torch.device(args.device),
        args.evaluation_tau,
        handoff.get("source_command_key"),
    )
    digest_before = _model_digest(policy)
    checkpoint_sha = _sha256_file(checkpoint_path)
    command_sha = _sha256_file(command_path)
    status = {
        "schema_version": 1,
        "status": "running",
        "dataset_dir": str(dataset_dir),
        "case_count": len(cases),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "source_command": str(command_path),
        "source_command_sha256": command_sha,
        "source_command_key": handoff.get("source_command_key"),
        "stage1_handoff": handoff,
        "evaluation_tau": float(args.evaluation_tau),
        "device_lookahead_safety_margin": float(
            args.device_lookahead_safety_margin
        ),
        "arms": ARMS,
        "completed_cases": {},
        "started_unix_time": time.time(),
    }
    _atomic_json(output_dir / "status.json", status)

    for arm, lookahead in ARMS.items():
        completed = {
            path.stem
            for path in (output_dir / "cases" / arm).glob("case_*.json")
        }
        status["completed_cases"][arm] = len(completed)
        status["heartbeat_unix_time"] = time.time()
        _atomic_json(output_dir / "status.json", status)
        missing = [path for path in case_paths if path.name not in completed]
        evaluator = FrozenPlaneEvaluator(
            policy,
            policy_args,
            max_steps=args.max_steps,
            device_lookahead_dispatch=lookahead,
            device_lookahead_safety_margin=(
                args.device_lookahead_safety_margin
            ),
        )
        for batch_index, batch_paths in enumerate(
            _chunks(missing, min(args.batch_size, len(cases)))
        ):
            print(
                f"[LookaheadControl] arm={arm} batch={batch_index + 1} "
                f"cases={len(batch_paths)}",
                flush=True,
            )
            batch = CrossCaseBatchedEvaluator(
                evaluator,
                batch_paths,
                async_graph_clone_workers=args.async_graph_clone_workers,
            )
            try:
                results = batch.run(
                    [
                        {
                            "backends": ARM_BACKENDS["hungarian_all"],
                            "layout": None,
                            "chromosome": None,
                            "record_trace": True,
                        }
                        for _ in batch_paths
                    ]
                )
            finally:
                batch.close()
            for case_path, result in zip(batch_paths, results):
                if not result.get("completed"):
                    raise RuntimeError(
                        f"{arm}/{case_path.name} failed: {result.get('error')}"
                    )
                metadata = _metadata(case_path)
                common = {
                    "schema_version": 1,
                    "status": "completed",
                    "case": case_path.name,
                    "case_id": metadata.get("case_id"),
                    "case_sha256": metadata.get("case_sha256"),
                    "profile": metadata.get("profile"),
                    "distribution": metadata.get("distribution"),
                    "arm": arm,
                    "resource_backend": ARM_BACKENDS["hungarian_all"],
                    "device_lookahead_dispatch": bool(lookahead),
                    "device_lookahead_safety_margin": float(
                        args.device_lookahead_safety_margin
                    ),
                    "frozen_plane_checkpoint": str(checkpoint_path),
                    "frozen_plane_checkpoint_sha256": checkpoint_sha,
                    "frozen_plane_source_command_sha256": command_sha,
                    "frozen_model_digest": digest_before,
                    "evaluation_tau": float(args.evaluation_tau),
                }
                trajectory_path = (
                    output_dir / "trajectories" / arm / f"{case_path.name}.json"
                )
                _atomic_json(trajectory_path, {**common, **result})
                wait_summary = dict(result["aircraft_resource_wait"])
                wait_summary.pop("events", None)
                _atomic_json(
                    output_dir / "cases" / arm / f"{case_path.name}.json",
                    {
                        **common,
                        "makespan": float(result["makespan"]),
                        "completion": result["completion"],
                        "aircraft_resource_wait": wait_summary,
                        "trajectory": str(trajectory_path),
                        "wall_seconds": float(result["wall_seconds"]),
                    },
                )
            status["completed_cases"][arm] = sum(
                1 for _ in (output_dir / "cases" / arm).glob("case_*.json")
            )
            status["heartbeat_unix_time"] = time.time()
            _atomic_json(output_dir / "status.json", status)

    digest_after = _model_digest(policy)
    if digest_after != digest_before:
        raise RuntimeError("Frozen plane model changed during A/B evaluation")
    payload = _summarize(output_dir, cases)
    status.update(
        {
            "status": "completed",
            "frozen_model_unchanged": True,
            "completed_unix_time": time.time(),
            "summary": str(output_dir / "summary.json"),
        }
    )
    _atomic_json(output_dir / "status.json", status)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mp.set_start_method("spawn", force=True)
    payload = run(args)
    print(json.dumps(payload["paired_effects"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
