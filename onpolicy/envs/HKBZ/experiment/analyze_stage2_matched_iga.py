#!/usr/bin/env python3
"""Compare planning-contract-matched IGA with Stage-2 F1/F2/F3."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


EXPECTED_COMMON_CONTRACT = {
    "device_lookahead_dispatch": True,
    "device_lookahead_safety_margin": 60.0,
    "device_deadline_aware_dispatch": True,
    "device_future_intent_horizon": 1,
    "device_future_intent_mode": "bounded_frontier",
    "device_frontier_max_requests": 2,
    "resource_release_aware_eta": True,
    "device_reservation_grace_seconds": 300.0,
    "device_departure_lookahead": True,
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_iga(directory: Path, reservation_mode: str) -> tuple[dict, dict]:
    summary = _read_json(directory / "summary.json")
    expected = {
        **EXPECTED_COMMON_CONTRACT,
        "device_lookahead_reservation_mode": reservation_mode,
    }
    if summary.get("status") != "completed":
        raise ValueError(f"IGA summary is not complete: {directory}")
    if summary.get("search_contract_version") != 4:
        raise ValueError(f"IGA search contract is not v4: {directory}")
    if summary.get("resource_lookahead_contract") != expected:
        raise ValueError(
            f"IGA planning contract mismatch in {directory}: "
            f"{summary.get('resource_lookahead_contract')!r}"
        )
    records = {}
    for path in sorted((directory / "cases").glob("case_*.json")):
        payload = _read_json(path)
        if payload.get("status") != "completed":
            raise ValueError(f"Incomplete IGA case: {path}")
        if payload.get("resource_lookahead_contract") != expected:
            raise ValueError(f"Case planning contract mismatch: {path}")
        records[path.stem] = {
            "makespan": float(payload["makespan"]),
            "profile": payload.get("profile"),
            "distribution": payload.get("distribution"),
        }
    if len(records) != int(summary.get("case_count", -1)):
        raise ValueError(f"IGA case count mismatch: {directory}")
    return records, summary


def _load_legacy_iga(directory: Path | None) -> tuple[dict, dict | None]:
    if directory is None:
        return {}, None
    summary = _read_json(directory / "summary.json")
    records = {}
    for path in sorted((directory / "cases").glob("case_*.json")):
        payload = _read_json(path)
        records[path.stem] = {
            "makespan": float(payload["makespan"]),
            "profile": payload.get("profile"),
            "distribution": payload.get("distribution"),
        }
    return records, summary


def _load_policy_evaluation(path: Path) -> tuple[dict, dict]:
    payload = _read_json(path)
    records = {}
    for item in payload.get("cases", []):
        case = str(item.get("case_dir") or Path(item["case_path"]).name)
        if not item.get("completed") or item.get("cycle_terminated"):
            raise ValueError(f"Incomplete policy case {case} in {path}")
        records[case] = {
            "makespan": float(item["makespan"]),
            "profile": item.get("profile"),
            "distribution": item.get("distribution"),
            "resource_wait_seconds": float(
                item.get("resource_wait_seconds", 0.0)
            ),
            "resource_critical_wait_seconds": float(
                item.get("resource_critical_wait_seconds", 0.0)
            ),
        }
    return records, payload


def _basic(values: Sequence[float]) -> dict:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 20260827, samples: int = 20000
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _paired(
    left: Mapping[str, Mapping],
    right: Mapping[str, Mapping],
    cases: Sequence[str],
) -> dict:
    # Delta is left - right, so a negative value means left is better.
    deltas = [
        float(left[case]["makespan"] - right[case]["makespan"])
        for case in cases
    ]
    return {
        "delta_definition": "left_makespan_minus_right_makespan",
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "paired_bootstrap_mean_delta_95ci": _bootstrap_mean_ci(deltas),
        "left_wins": sum(value < 0.0 for value in deltas),
        "ties": sum(value == 0.0 for value in deltas),
        "right_wins": sum(value > 0.0 for value in deltas),
        "mean_relative_delta": statistics.mean(
            deltas[index] / float(right[case]["makespan"])
            for index, case in enumerate(cases)
        ),
    }


def _grouped_pair(
    left: Mapping[str, Mapping],
    right: Mapping[str, Mapping],
    cases: Sequence[str],
    key: str,
) -> dict:
    grouped = {}
    values = sorted({str(left[case].get(key)) for case in cases})
    for value in values:
        selected = [case for case in cases if str(left[case].get(key)) == value]
        grouped[value] = _paired(left, right, selected)
        grouped[value]["case_count"] = len(selected)
    return grouped


def _assert_same_cases(methods: Mapping[str, Mapping]) -> list[str]:
    names = list(methods)
    expected = set(methods[names[0]])
    for name in names[1:]:
        observed = set(methods[name])
        if observed != expected:
            raise ValueError(
                f"Case set mismatch for {name}: "
                f"missing={sorted(expected - observed)!r}, "
                f"extra={sorted(observed - expected)!r}"
            )
    return sorted(expected)


def analyze(args) -> dict:
    hard, hard_summary = _load_iga(args.hard_iga_dir, "hard")
    soft, soft_summary = _load_iga(args.soft_iga_dir, "soft")
    f1, f1_payload = _load_policy_evaluation(args.f1_evaluation)
    f2, f2_payload = _load_policy_evaluation(args.f2_evaluation)
    f3, f3_payload = _load_policy_evaluation(args.f3_evaluation)
    methods = {
        "matched_iga1800_hard_F1": hard,
        "matched_iga1800_soft_F2F3": soft,
        "F1_hard_reservation": f1,
        "F2_iga_flow_bc": f2,
        "F3_wait_constraint": f3,
    }
    legacy, legacy_summary = _load_legacy_iga(args.legacy_iga_dir)
    if legacy:
        methods["legacy_iga1800"] = legacy
    cases = _assert_same_cases(methods)
    if len(cases) != 60:
        raise ValueError(f"Expected tune60, found {len(cases)} cases.")

    pair_specs = {
        "F1_vs_matched_hard_IGA": (f1, hard),
        "F2_vs_matched_soft_IGA": (f2, soft),
        "F3_vs_matched_soft_IGA": (f3, soft),
        "matched_hard_vs_matched_soft_IGA": (hard, soft),
    }
    if legacy:
        pair_specs.update(
            {
                "matched_hard_vs_legacy_IGA": (hard, legacy),
                "matched_soft_vs_legacy_IGA": (soft, legacy),
            }
        )
    paired = {
        name: {
            **_paired(left, right, cases),
            "by_distribution": _grouped_pair(
                left, right, cases, "distribution"
            ),
            "by_profile": _grouped_pair(left, right, cases, "profile"),
        }
        for name, (left, right) in pair_specs.items()
    }

    rows = []
    for case in cases:
        row = {
            "case": case,
            "profile": hard[case].get("profile"),
            "distribution": hard[case].get("distribution"),
        }
        for name, records in methods.items():
            row[name] = float(records[case]["makespan"])
        row["F1_minus_matched_hard"] = row["F1_hard_reservation"] - row[
            "matched_iga1800_hard_F1"
        ]
        row["F2_minus_matched_soft"] = row["F2_iga_flow_bc"] - row[
            "matched_iga1800_soft_F2F3"
        ]
        row["F3_minus_matched_soft"] = row["F3_wait_constraint"] - row[
            "matched_iga1800_soft_F2F3"
        ]
        rows.append(row)

    payload = {
        "status": "completed",
        "schema_version": 1,
        "created_unix_time": time.time(),
        "comparison_scope": {
            "dataset": f1_payload.get("eval_dataset_dir"),
            "case_count": len(cases),
            "fairness": (
                "Optimizer comparison under the same frozen plane policy and "
                "the same frontier, ETA, reservation, and departure-lookahead "
                "semantics. F1 maps to hard IGA; F2/F3 map to one identical "
                "soft IGA arm."
            ),
        },
        "planning_contracts": {
            "hard_F1": hard_summary["resource_lookahead_contract"],
            "soft_F2F3": soft_summary["resource_lookahead_contract"],
        },
        "sources": {
            "hard_iga": str(args.hard_iga_dir.resolve()),
            "soft_iga": str(args.soft_iga_dir.resolve()),
            "legacy_iga": (
                str(args.legacy_iga_dir.resolve())
                if args.legacy_iga_dir is not None
                else None
            ),
            "F1_evaluation": str(args.f1_evaluation.resolve()),
            "F2_evaluation": str(args.f2_evaluation.resolve()),
            "F3_evaluation": str(args.f3_evaluation.resolve()),
            "F1_label": f1_payload.get("evaluation_label"),
            "F2_label": f2_payload.get("evaluation_label"),
            "F3_label": f3_payload.get("evaluation_label"),
        },
        "iga_search": {
            "hard": hard_summary,
            "soft": soft_summary,
            "legacy": legacy_summary,
        },
        "method_summaries": {
            name: _basic(
                [records[case]["makespan"] for case in cases]
            )
            for name, records in methods.items()
        },
        "paired_comparisons": paired,
        "per_case": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matched_iga_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "matched_iga_per_case.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Stage2 matched-information IGA comparison",
        "",
        "All values below use the same tune60 cases. Lower Cmax is better.",
        "",
        "| Method | Mean Cmax | Std | Median |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in payload["method_summaries"].items():
        lines.append(
            f"| {name} | {summary['mean']:.2f} | "
            f"{summary['std']:.2f} | {summary['median']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Paired comparison (left - right) | Mean delta | 95% CI | "
            "Left wins / ties / right wins |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, comparison in paired.items():
        low, high = comparison["paired_bootstrap_mean_delta_95ci"]
        lines.append(
            f"| {name} | {comparison['mean_delta']:.2f} | "
            f"[{low:.2f}, {high:.2f}] | {comparison['left_wins']} / "
            f"{comparison['ties']} / {comparison['right_wins']} |"
        )
    (args.output_dir / "matched_iga_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["method_summaries"], indent=2), flush=True)
    print(json.dumps(paired, indent=2), flush=True)
    return payload


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-iga-dir", type=Path, required=True)
    parser.add_argument("--soft-iga-dir", type=Path, required=True)
    parser.add_argument("--legacy-iga-dir", type=Path, default=None)
    parser.add_argument("--f1-evaluation", type=Path, required=True)
    parser.add_argument("--f2-evaluation", type=Path, required=True)
    parser.add_argument("--f3-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    analyze(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
