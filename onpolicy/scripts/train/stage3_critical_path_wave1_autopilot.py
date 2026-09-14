#!/usr/bin/env python3
"""Wait for Stage3 C0-C3, audit the causal screen and select one candidate."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np


METHODS = ("C0", "C1", "C2", "C3")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _terminal_status(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        status = _load(path)
    except (OSError, json.JSONDecodeError):
        return False
    return str(status.get("status", "")) in {
        "completed", "failed", "stopped", "canary_rejected"
    }


def _wait_for_runs(run_dirs: dict[str, Path], timeout_seconds: float) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        missing = [
            method for method, run_dir in run_dirs.items()
            if not _terminal_status(run_dir / "run_status.json")
        ]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Stage3 arms: {missing}")
        print(f"[CriticalWave1] waiting for {missing}", flush=True)
        time.sleep(30.0)


def _case_signature(payload: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(
        (
            str(record.get("case_key", record.get("case_id", ""))),
            str(record.get("case_sha256", "")),
        )
        for record in payload.get("cases", ())
    ))


def _valid_evaluation(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = _load(path)
    summary = payload.get("summary", {})
    cases = list(payload.get("cases", ()))
    if (
        int(summary.get("eval_valid", 0)) != 1
        or float(summary.get("eval_completion_rate", 0.0)) != 1.0
        or int(summary.get("eval_cycle_count", 1)) != 0
        or int(summary.get("eval_timeout_count", 1)) != 0
        or len(cases) != 60
    ):
        return None
    return payload


def _mean(cases: list[dict], field: str) -> float:
    return float(np.mean([
        float(record.get(field, 0.0)) for record in cases
    ]))


def _profile_mean(cases: list[dict], profile: str) -> float:
    values = [
        float(record["makespan"])
        for record in cases if record.get("profile") == profile
    ]
    return float(np.mean(values)) if values else math.inf


def _summarize_eval(payload: dict) -> dict:
    summary = dict(payload["summary"])
    cases = list(payload["cases"])
    makespans = sorted(float(record["makespan"]) for record in cases)
    return {
        "raw_cmax": float(np.mean(makespans)),
        "selection_score": float(summary["eval_selection_score"]),
        "tail_worst10": float(np.mean(makespans[-6:])),
        "iid_cmax": float(summary["eval_distribution_iid_makespan"]),
        "ood_stress_cmax": float(
            summary["eval_distribution_ood_stress_makespan"]
        ),
        "ood_scale_cmax": float(
            summary["eval_distribution_ood_scale_makespan"]
        ),
        "bursty_cmax": _profile_mean(cases, "bursty"),
        "stress_joint_cmax": _profile_mean(cases, "stress_joint"),
        "resource_ood_cmax": _profile_mean(cases, "resource_ood"),
        "resource_wait": _mean(cases, "resource_wait_seconds"),
        "slack_weighted_wait": _mean(
            cases, "resource_slack_weighted_wait_seconds"
        ),
        "avoidable_critical_lateness": _mean(
            cases, "resource_avoidable_critical_lateness_seconds"
        ),
        "avoidable_critical_lateness_p95": _mean(
            cases, "resource_avoidable_critical_lateness_p95_seconds"
        ),
        "rendezvous_spread": _mean(
            cases, "resource_rendezvous_spread_seconds"
        ),
        "rendezvous_spread_p95": _mean(
            cases, "resource_rendezvous_spread_p95_seconds"
        ),
        "predicted_lateness": _mean(
            cases, "resource_predicted_lateness_seconds"
        ),
        "case_signature": _case_signature(payload),
    }


def _paired_bootstrap(
    candidate: dict,
    control: dict,
    *,
    samples: int = 20000,
    seed: int = 20260825,
) -> dict:
    control_by_case = {
        str(item.get("case_key", item.get("case_id"))): float(item["makespan"])
        for item in control["cases"]
    }
    deltas = np.asarray([
        float(item["makespan"])
        - control_by_case[str(item.get("case_key", item.get("case_id")))]
        for item in candidate["cases"]
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, deltas.size, size=(samples, deltas.size))
    means = deltas[indices].mean(axis=1)
    return {
        "mean_delta_vs_C0": float(deltas.mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "probability_improves_C0": float(np.mean(means < 0.0)),
        "case_wins": int(np.sum(deltas < 0.0)),
        "case_losses": int(np.sum(deltas > 0.0)),
        "regressions_over_800": int(np.sum(deltas > 800.0)),
    }


def _arm_record(method: str, run_dir: Path, epochs: int) -> dict:
    status = _load(run_dir / "run_status.json")
    evaluations = []
    for epoch in range(1, epochs + 1):
        payload = _valid_evaluation(
            run_dir / "evaluations" / f"epoch_{epoch}.json"
        )
        if payload is not None:
            evaluations.append((epoch, payload, _summarize_eval(payload)))
    hard_reasons = []
    if status.get("status") != "completed":
        hard_reasons.append(f"run_status={status.get('status')}")
    health = status.get("actor_update_health", {})
    if float(health.get("step_completion_rate", 0.0)) < 0.95:
        hard_reasons.append("actor_step_completion_below_95pct")
    if int(health.get("zero_update_shards", 0)):
        hard_reasons.append("zero_update_shards")
    if float(health.get("empty_replay_fraction", 1.0)) > 0.01:
        hard_reasons.append("empty_replay_fraction_above_1pct")
    if bool(status.get("canary_rejected", False)):
        hard_reasons.append("canary_rejected")
    if len(evaluations) != epochs:
        hard_reasons.append(
            f"valid_evaluations={len(evaluations)}_expected={epochs}"
        )
    best = min(
        evaluations,
        key=lambda item: (item[2]["raw_cmax"], item[2]["tail_worst10"]),
        default=None,
    )
    return {
        "method": method,
        "run_dir": str(run_dir),
        "hard_reasons": hard_reasons,
        "health": health,
        "epochs": [
            {"epoch": epoch, **metrics}
            for epoch, _, metrics in evaluations
        ],
        "best_epoch": int(best[0]) if best else None,
        "best_metrics": best[2] if best else None,
        "_best_payload": best[1] if best else None,
    }


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Stage3 critical-path Wave-1 selection",
        "",
        "| Arm | Epoch | Cmax | worst-10 | bursty | stress-joint | "
        "critical lateness | rendezvous | Δ vs C0 | admitted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for arm in report["arms"]:
        metrics = arm.get("best_metrics") or {}
        paired = arm.get("paired_vs_C0") or {}
        lines.append(
            f"| {arm['method']} | {arm.get('best_epoch') or '-'} | "
            f"{metrics.get('raw_cmax', math.nan):.2f} | "
            f"{metrics.get('tail_worst10', math.nan):.2f} | "
            f"{metrics.get('bursty_cmax', math.nan):.2f} | "
            f"{metrics.get('stress_joint_cmax', math.nan):.2f} | "
            f"{metrics.get('avoidable_critical_lateness', math.nan):.2f} | "
            f"{metrics.get('rendezvous_spread', math.nan):.2f} | "
            f"{paired.get('mean_delta_vs_C0', 0.0):+.2f} | "
            f"{'yes' if arm.get('admitted') else 'no'} |"
        )
    lines.extend([
        "",
        f"Selected candidate: **{report.get('selected_method') or 'none'}**",
        "",
        "Selection uses raw Cmax first, then worst-10. Total resource wait is "
        "diagnostic only and is not an admission gate.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--profile", default="critical_wave1")
    parser.add_argument("--unit-prefix", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=7 * 86400)
    args = parser.parse_args()
    run_dirs = {
        method: args.results_root
        / f"{args.run_tag}_{args.profile}_{method}_seed1"
        / "run1"
        for method in METHODS
    }
    _wait_for_runs(run_dirs, args.timeout_seconds)
    try:
        subprocess.run(
            [
                "systemctl", "--user", "stop",
                f"{args.unit_prefix}-{args.profile}-eval.service",
            ],
            check=False,
        )
        arms = [
            _arm_record(method, run_dirs[method], args.epochs)
            for method in METHODS
        ]
        control = arms[0].get("_best_payload")
        signatures = {
            tuple(arm["best_metrics"]["case_signature"])
            for arm in arms if arm.get("best_metrics")
        }
        if len(signatures) != 1:
            raise ValueError("C0-C3 do not share the same Valid60 cases.")
        control_metrics = arms[0].get("best_metrics") or {}
        for arm in arms:
            payload = arm.pop("_best_payload", None)
            arm["paired_vs_C0"] = (
                _paired_bootstrap(payload, control)
                if payload is not None and control is not None else None
            )
            reasons = list(arm["hard_reasons"])
            if arm["method"] != "C0" and arm.get("best_metrics"):
                metrics = arm["best_metrics"]
                paired = arm["paired_vs_C0"]
                if paired["probability_improves_C0"] < 0.80:
                    reasons.append("bootstrap_improvement_probability_below_80pct")
                if paired["regressions_over_800"]:
                    reasons.append("new_case_regression_over_800")
                control_resource_ood = control_metrics.get(
                    "resource_ood_cmax", math.inf
                )
                if metrics["resource_ood_cmax"] > 1.005 * control_resource_ood:
                    reasons.append("resource_ood_regression_above_0.5pct")
            arm["admission_reasons"] = reasons
            arm["admitted"] = not reasons
        candidates = [arm for arm in arms if arm["admitted"]]
        selected = min(
            candidates,
            key=lambda arm: (
                arm["best_metrics"]["raw_cmax"],
                arm["best_metrics"]["tail_worst10"],
            ),
            default=None,
        )
        report = {
            "schema_version": 1,
            "suite": str(args.suite.resolve()),
            "created_unix_time": time.time(),
            "selection_objective": "raw_cmax_then_worst10",
            "total_wait_is_gate": False,
            "selected_method": selected["method"] if selected else None,
            "selected_epoch": selected["best_epoch"] if selected else None,
            "arms": arms,
        }
        output = args.suite / "analysis" / "critical_wave1_selection.json"
        _atomic_json(output, report)
        _write_markdown(
            args.suite / "analysis" / "critical_wave1_selection.md", report
        )
        print(
            f"[CriticalWave1] selected={report['selected_method']} "
            f"epoch={report['selected_epoch']} report={output}",
            flush=True,
        )
    finally:
        subprocess.run(
            [
                "systemctl", "--user", "stop",
                f"{args.unit_prefix}-{args.profile}-eval.service",
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
