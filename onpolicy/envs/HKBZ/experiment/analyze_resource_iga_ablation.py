#!/usr/bin/env python3
"""Produce mechanism-level analysis for the frozen-plane resource IGA screen."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ARMS = ("iga_all", "iga_r014", "iga_ordinary", "hungarian_all")
PRIMARY_CONTRASTS = (
    ("iga_all", "iga_r014", "ordinary IGA given IGA R014"),
    ("iga_all", "iga_ordinary", "R014 IGA given ordinary IGA"),
    ("iga_r014", "iga_ordinary", "R014-only versus ordinary-only IGA"),
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(values) if values else None


def _trace_metrics(payload: dict) -> dict:
    role_counts = defaultdict(Counter)
    role_waiting = defaultdict(list)
    reason_counts = Counter()
    backend_counts = Counter()
    plane_decisions = 0
    resource_decisions = 0
    dispatches = 0
    preferred_dispatches = 0
    lookahead_request_observations = 0
    lookahead_dispatches = 0
    lookahead_lead_times = []
    lookahead_travel_times = []
    for event in payload.get("decision_trace", []):
        plane_decisions += len(event.get("plane_decisions", []))
        request_by_id = {
            int(request["id"]): request
            for request in event.get("requests", [])
        }
        lookahead_request_observations += sum(
            bool(request.get("is_lookahead", False))
            for request in event.get("requests", [])
        )
        for decision in event.get("resource_decisions", []):
            resource_decisions += 1
            role = str(decision.get("role") or "unknown")
            backend = str(decision.get("backend") or "unknown")
            reason = str(decision.get("reason") or "unknown")
            role_counts[role]["decision_turns"] += 1
            backend_counts[backend] += 1
            reason_counts[reason] += 1
            request_id = int(decision.get("selected_request_id") or 0)
            if request_id <= 0:
                role_counts[role]["noops"] += 1
                continue
            dispatches += 1
            role_counts[role]["dispatches"] += 1
            if request_id == decision.get("preferred_request_id"):
                preferred_dispatches += 1
                role_counts[role]["preferred_dispatches"] += 1
            request = request_by_id.get(request_id)
            if request is not None:
                role_waiting[role].append(
                    float(request.get("waiting_time", 0.0))
                )
                if request.get("is_lookahead", False):
                    lookahead_dispatches += 1
    for record in payload.get("resource_trajectory", []):
        if not record.get("is_lookahead", False):
            continue
        lookahead_lead_times.append(
            float(record.get("lead_time_at_dispatch", 0.0))
        )
        lookahead_travel_times.append(float(record.get("trans_time", 0.0)))
    role_metrics = {}
    for role in sorted(set(role_counts) | set(role_waiting)):
        counts = role_counts[role]
        role_metrics[role] = {
            **{key: int(value) for key, value in sorted(counts.items())},
            "mean_selected_waiting_time": _finite_mean(role_waiting[role]),
        }
    return {
        "makespan": float(payload["makespan"]),
        "environment_steps": len(payload.get("decision_trace", [])),
        "plane_decisions": int(plane_decisions),
        "resource_decision_turns": int(resource_decisions),
        "dispatches": int(dispatches),
        "preferred_dispatches": int(preferred_dispatches),
        "preferred_dispatch_rate": (
            float(preferred_dispatches / dispatches) if dispatches else None
        ),
        "lookahead_request_observations": int(
            lookahead_request_observations
        ),
        "lookahead_dispatches": int(lookahead_dispatches),
        "mean_lookahead_lead_time": _finite_mean(lookahead_lead_times),
        "mean_lookahead_travel_time": _finite_mean(lookahead_travel_times),
        "reason_counts": dict(sorted(reason_counts.items())),
        "backend_decision_turns": dict(sorted(backend_counts.items())),
        "by_role": role_metrics,
    }


def _aggregate_trace_metrics(case_metrics: dict[str, dict]) -> dict:
    scalar_keys = (
        "environment_steps",
        "plane_decisions",
        "resource_decision_turns",
        "dispatches",
        "preferred_dispatches",
        "lookahead_request_observations",
        "lookahead_dispatches",
    )
    aggregate = {"case_count": len(case_metrics)}
    aggregate.update(
        {
            f"mean_{key}": _finite_mean(
                metrics[key] for metrics in case_metrics.values()
            )
            for key in scalar_keys
        }
    )
    role_values = defaultdict(lambda: defaultdict(list))
    for metrics in case_metrics.values():
        for role, role_metric in metrics["by_role"].items():
            for key in ("decision_turns", "dispatches", "noops"):
                role_values[role][key].append(role_metric.get(key, 0))
            waiting = role_metric.get("mean_selected_waiting_time")
            if waiting is not None:
                role_values[role]["mean_selected_waiting_time"].append(waiting)
    aggregate["by_role_case_means"] = {
        role: {
            key: _finite_mean(values)
            for key, values in sorted(role_metrics.items())
        }
        for role, role_metrics in sorted(role_values.items())
    }
    return aggregate


def _evidence_label(comparison: dict) -> str:
    low, high = comparison["bootstrap_mean_95ci"]
    if high < 0.0:
        return "left_better_supported"
    if low > 0.0:
        return "right_better_supported"
    return "inconclusive"


def _stage_implication(summary: dict) -> dict:
    ordinary = summary["paired_comparisons"][
        "iga_all_minus_iga_r014"
    ]
    r014 = summary["paired_comparisons"][
        "iga_all_minus_iga_ordinary"
    ]
    ordinary_evidence = _evidence_label(ordinary)
    r014_evidence = _evidence_label(r014)
    if ordinary_evidence == r014_evidence == "left_better_supported":
        recommendation = (
            "Both roles have complementary scheduling leverage. Keep distinct "
            "role heads/loss accounting, but this ablation alone does not justify "
            "two temporal training stages; test gradient interference next."
        )
    elif r014_evidence == "left_better_supported":
        recommendation = (
            "R014 contributes clearly while ordinary-resource IGA does not. "
            "Prioritize an R014 warm-up/head and retain Hungarian ordinary "
            "dispatch until learned ordinary control passes a separate gate."
        )
    elif ordinary_evidence == "left_better_supported":
        recommendation = (
            "Ordinary resources contribute clearly while R014 IGA does not. "
            "Prioritize ordinary-resource learning and keep R014 Hungarian for "
            "the first Stage2 candidate."
        )
    elif (
        ordinary_evidence == "right_better_supported"
        or r014_evidence == "right_better_supported"
    ):
        recommendation = (
            "At least one IGA role replacement is reliably harmful under the "
            "frozen plane policy. Do not create a training stage for that role "
            "before diagnosing its dispatch/credit mismatch."
        )
    else:
        recommendation = (
            "The role effects are unresolved at this search budget. Retain one "
            "joint Stage2 with role-balanced heads and run a larger-budget or "
            "multi-plane-seed replication before adding a temporal stage split."
        )
    return {
        "ordinary_marginal_evidence": ordinary_evidence,
        "r014_marginal_evidence": r014_evidence,
        "recommendation": recommendation,
        "limitation": (
            "Per-case IGA is an oracle-style mechanism probe; a stage decision "
            "still requires learned-policy optimization and generalization tests."
        ),
    }


def analyze(output_dir: Path) -> dict:
    summary = _read(output_dir / "summary.json")
    cases = [item["case"] for item in summary["cases"]]
    per_case = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for case in cases:
            per_case[arm][case] = _trace_metrics(
                _read(output_dir / "trajectories" / arm / f"{case}.json")
            )

    extremes = {}
    case_summary = {item["case"]: item for item in summary["cases"]}
    for left, right, label in PRIMARY_CONTRASTS:
        ranked = sorted(
            (
                {
                    "case": case,
                    "profile": case_summary[case].get("profile"),
                    "distribution": case_summary[case].get("distribution"),
                    "left": left,
                    "right": right,
                    "left_makespan": float(case_summary[case][left]),
                    "right_makespan": float(case_summary[case][right]),
                    "delta": float(
                        case_summary[case][left] - case_summary[case][right]
                    ),
                }
                for case in cases
            ),
            key=lambda item: item["delta"],
        )
        extremes[f"{left}_minus_{right}"] = {
            "meaning": label,
            "largest_left_advantages": ranked[:10],
            "largest_right_advantages": ranked[-10:][::-1],
        }

    payload = {
        "schema_version": 1,
        "status": "completed",
        "summary_path": str((output_dir / "summary.json").resolve()),
        "trace_aggregates": {
            arm: _aggregate_trace_metrics(per_case[arm]) for arm in ARMS
        },
        "per_case_trace_metrics": per_case,
        "case_extremes": extremes,
        "stage_implication": _stage_implication(summary),
    }
    (output_dir / "detailed_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# Resource-role IGA detailed analysis",
        "",
        "## Makespan contrasts",
        "",
        "| Contrast | Mean delta | 95% bootstrap CI | W/T/L | Evidence |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for left, right, _ in PRIMARY_CONTRASTS:
        name = f"{left}_minus_{right}"
        item = summary["paired_comparisons"][name]
        low, high = item["bootstrap_mean_95ci"]
        lines.append(
            f"| {name} | {item['mean_delta']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | "
            f"{item['left_wins']}/{item['ties']}/{item['right_wins']} | "
            f"{_evidence_label(item)} |"
        )
    lines.extend(
        [
            "",
            "## Trace aggregates",
            "",
            "| Arm | Mean steps | Mean dispatches | Mean resource turns |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        item = payload["trace_aggregates"][arm]
        lines.append(
            f"| {arm} | {item['mean_environment_steps']:.2f} | "
            f"{item['mean_dispatches']:.2f} | "
            f"{item['mean_resource_decision_turns']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Stage implication",
            "",
            payload["stage_implication"]["recommendation"],
            "",
            payload["stage_implication"]["limitation"],
            "",
        ]
    )
    (output_dir / "detailed_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.output_dir.resolve())
    print(json.dumps(payload["stage_implication"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
