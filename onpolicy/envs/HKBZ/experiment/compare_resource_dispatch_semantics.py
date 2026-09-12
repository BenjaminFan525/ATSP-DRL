#!/usr/bin/env python3
"""Paired old-versus-lookahead analysis for the resource IGA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ARMS = ("iga_all", "iga_r014", "iga_ordinary", "hungarian_all")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _derived_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _bootstrap_ci(values: Sequence[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(10000, array.size))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _paired_effect(rows: Sequence[Mapping], *, seed: int) -> dict:
    old = np.asarray([row["legacy_makespan"] for row in rows], dtype=np.float64)
    early = np.asarray(
        [row["lookahead_makespan"] for row in rows], dtype=np.float64
    )
    deltas = early - old
    return {
        "case_count": int(deltas.size),
        "legacy_mean_makespan": float(old.mean()),
        "lookahead_mean_makespan": float(early.mean()),
        "mean_delta_lookahead_minus_legacy": float(deltas.mean()),
        "median_delta_lookahead_minus_legacy": float(np.median(deltas)),
        "relative_delta_percent_of_legacy_mean": float(
            100.0 * deltas.mean() / old.mean()
        ),
        "bootstrap_mean_delta_95ci": _bootstrap_ci(deltas, seed),
        "lookahead_wins": int((deltas < 0.0).sum()),
        "ties": int((deltas == 0.0).sum()),
        "legacy_wins": int((deltas > 0.0).sum()),
    }


def _trace_metrics(payload: Mapping) -> dict:
    request_observations = 0
    decision_dispatches = 0
    trajectory_dispatches = []
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
            request = requests.get(
                int(decision.get("selected_request_id") or 0)
            )
            if request is not None and request.get("is_lookahead", False):
                decision_dispatches += 1
    for record in payload.get("resource_trajectory", []):
        if record.get("is_lookahead", False):
            lead = float(record.get("lead_time_at_dispatch", 0.0))
            travel = float(record.get("trans_time", 0.0))
            trajectory_dispatches.append(
                {
                    "lead_time": lead,
                    "travel_time": travel,
                    "arrival_earliness": max(0.0, lead - travel),
                    "arrival_lateness": max(0.0, travel - lead),
                }
            )
    return {
        "request_observations": int(request_observations),
        "decision_dispatches": int(decision_dispatches),
        "trajectory_dispatches": int(len(trajectory_dispatches)),
        "mean_lead_time": (
            statistics.mean(item["lead_time"] for item in trajectory_dispatches)
            if trajectory_dispatches
            else None
        ),
        "mean_travel_time": (
            statistics.mean(item["travel_time"] for item in trajectory_dispatches)
            if trajectory_dispatches
            else None
        ),
        "mean_arrival_earliness": (
            statistics.mean(
                item["arrival_earliness"] for item in trajectory_dispatches
            )
            if trajectory_dispatches
            else None
        ),
        "mean_arrival_lateness": (
            statistics.mean(
                item["arrival_lateness"] for item in trajectory_dispatches
            )
            if trajectory_dispatches
            else None
        ),
    }


def _same(left, right) -> bool:
    return left is None or right is None or left == right


def _validate_pair(legacy: Mapping, lookahead: Mapping, case: str, arm: str):
    checks = (
        "case_sha256",
        "frozen_plane_checkpoint_sha256",
        "frozen_plane_source_command_sha256",
        "evaluation_tau",
    )
    for key in checks:
        if not _same(legacy.get(key), lookahead.get(key)):
            raise ValueError(f"{case}/{arm}: unmatched {key}")
    old_search = legacy.get("search")
    new_search = lookahead.get("search")
    if (old_search is None) != (new_search is None):
        raise ValueError(f"{case}/{arm}: search presence differs")
    for key in (
        "population",
        "generations",
        "evaluated_candidates",
        "seed",
        "common_initial_population_seed",
        "n_var",
        "iga_device_count",
    ):
        if old_search is not None and old_search.get(key) != new_search.get(key):
            raise ValueError(f"{case}/{arm}: unmatched search field {key}")


def compare(legacy_dir: Path, lookahead_dir: Path, output_dir: Path) -> dict:
    legacy_summary = _read(legacy_dir / "summary.json")
    lookahead_summary = _read(lookahead_dir / "summary.json")
    legacy_cases = {item["case"]: item for item in legacy_summary["cases"]}
    lookahead_cases = {item["case"]: item for item in lookahead_summary["cases"]}
    if set(legacy_cases) != set(lookahead_cases):
        raise ValueError("Legacy and lookahead case sets differ")
    if bool(legacy_summary.get("device_lookahead_dispatch", False)):
        raise ValueError("Legacy summary unexpectedly enables lookahead")
    if not bool(lookahead_summary.get("device_lookahead_dispatch", False)):
        raise ValueError("Lookahead summary does not declare lookahead semantics")

    rows_by_arm = {arm: [] for arm in ARMS}
    trace_by_arm = {arm: {} for arm in ARMS}
    case_rows = []
    for case in sorted(legacy_cases):
        metadata = legacy_cases[case]
        case_row = {
            "case": case,
            "profile": metadata.get("profile"),
            "distribution": metadata.get("distribution"),
            "arms": {},
        }
        for arm in ARMS:
            old_result = _read(legacy_dir / "cases" / case / f"{arm}.json")
            new_result = _read(
                lookahead_dir / "cases" / case / f"{arm}.json"
            )
            _validate_pair(old_result, new_result, case, arm)
            old_makespan = float(old_result["makespan"])
            new_makespan = float(new_result["makespan"])
            row = {
                "case": case,
                "profile": metadata.get("profile"),
                "distribution": metadata.get("distribution"),
                "legacy_makespan": old_makespan,
                "lookahead_makespan": new_makespan,
                "delta_lookahead_minus_legacy": new_makespan - old_makespan,
            }
            rows_by_arm[arm].append(row)
            metrics = _trace_metrics(
                _read(lookahead_dir / "trajectories" / arm / f"{case}.json")
            )
            trace_by_arm[arm][case] = metrics
            case_row["arms"][arm] = {**row, "lookahead_trace": metrics}
        case_rows.append(case_row)

    effects = {}
    for arm, rows in rows_by_arm.items():
        item = _paired_effect(rows, seed=_derived_seed(arm, "all"))
        for group_key in ("distribution", "profile"):
            grouped = {}
            for group in sorted({str(row.get(group_key)) for row in rows}):
                selected = [
                    row for row in rows if str(row.get(group_key)) == group
                ]
                grouped[group] = _paired_effect(
                    selected, seed=_derived_seed(arm, group_key, group)
                )
            item[f"by_{group_key}"] = grouped
        effects[arm] = item

    trace_aggregates = {}
    for arm, case_metrics in trace_by_arm.items():
        metrics = list(case_metrics.values())
        trace_aggregates[arm] = {
            "case_count": len(metrics),
            "total_lookahead_dispatches": sum(
                item["trajectory_dispatches"] for item in metrics
            ),
            "mean_lookahead_dispatches_per_case": statistics.mean(
                item["trajectory_dispatches"] for item in metrics
            ),
            "mean_case_lead_time": statistics.mean(
                item["mean_lead_time"]
                for item in metrics
                if item["mean_lead_time"] is not None
            ) if any(item["mean_lead_time"] is not None for item in metrics) else None,
            "mean_case_arrival_earliness": statistics.mean(
                item["mean_arrival_earliness"]
                for item in metrics
                if item["mean_arrival_earliness"] is not None
            ) if any(item["mean_arrival_earliness"] is not None for item in metrics) else None,
            "mean_case_arrival_lateness": statistics.mean(
                item["mean_arrival_lateness"]
                for item in metrics
                if item["mean_arrival_lateness"] is not None
            ) if any(item["mean_arrival_lateness"] is not None for item in metrics) else None,
        }

    payload = {
        "schema_version": 1,
        "status": "completed",
        "contrast": "lookahead_minus_legacy; negative makespan delta is better",
        "legacy_dir": str(legacy_dir.resolve()),
        "lookahead_dir": str(lookahead_dir.resolve()),
        "case_count": len(case_rows),
        "paired_semantic_effects": effects,
        "lookahead_trace_aggregates": trace_aggregates,
        "cases": case_rows,
        "completed_unix_time": time.time(),
    }
    _write_json(output_dir / "lookahead_comparison.json", payload)

    lines = [
        "# Mobile-resource lookahead versus legacy semantics",
        "",
        "Negative delta means early decisions reduced Cmax.",
        "",
        "| Arm | Legacy mean | Lookahead mean | Delta | 95% CI | W/T/L | Early dispatches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        effect = effects[arm]
        low, high = effect["bootstrap_mean_delta_95ci"]
        trace = trace_aggregates[arm]
        lines.append(
            f"| {arm} | {effect['legacy_mean_makespan']:.2f} | "
            f"{effect['lookahead_mean_makespan']:.2f} | "
            f"{effect['mean_delta_lookahead_minus_legacy']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | "
            f"{effect['lookahead_wins']}/{effect['ties']}/{effect['legacy_wins']} | "
            f"{trace['total_lookahead_dispatches']} |"
        )
    lines.extend(
        [
            "",
            "The plane-network weights and per-arm IGA budgets are matched. "
            "Plane actions may still differ because the same frozen policy observes "
            "the resource state produced by each environment semantic.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lookahead_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--lookahead-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(
        args.legacy_dir.resolve(),
        args.lookahead_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(payload["paired_semantic_effects"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
