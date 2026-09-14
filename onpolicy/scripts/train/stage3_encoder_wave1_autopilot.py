#!/usr/bin/env python3
"""Finish Stage3 encoder Wave-1, select a method, then launch joint IGA labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("E0", "E1", "E2", "E3")
JOINT_IGA_RUNNER = ROOT / "onpolicy/scripts/train/run_stage3_joint_iga_pipeline.py"
DEFAULT_JOINT_IGA_DATASET = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/"
    "fjsp_v3_resource_joint_eval_s20260811/joint/tune"
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_signature(cases: list[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(
        (str(case.get("case_key", "")), str(case.get("case_sha256", "")))
        for case in cases
    ))


def _mean(cases: list[Mapping[str, Any]], key: str) -> float:
    values = [float(case.get(key, 0.0)) for case in cases]
    return sum(values) / len(values) if values else math.inf


def _eval_record(path: Path) -> dict[str, Any]:
    payload = _load(path)
    cases = list(payload.get("cases", ()))
    summary = dict(payload.get("summary", {}))
    if len(cases) != 60 or int(summary.get("eval_case_count", 0)) != 60:
        raise ValueError(f"Expected Valid60 in {path}")
    return {
        "path": str(path.resolve()),
        "selection_score": float(summary["eval_selection_score"]),
        "raw_makespan": float(summary["eval_raw_makespan"]),
        "ood_stress_makespan": float(
            summary["eval_distribution_ood_stress_makespan"]
        ),
        "tail_makespan": float(summary["eval_tail_makespan"]),
        "completion_rate": float(summary["eval_completion_rate"]),
        "cycle_count": int(summary["eval_cycle_count"]),
        "timeout_count": int(summary["eval_timeout_count"]),
        "mean_resource_wait": _mean(cases, "resource_wait_seconds"),
        "mean_critical_wait": _mean(cases, "resource_critical_wait_seconds"),
        "mean_dispatch_wait": _mean(
            cases, "resource_wait_before_dispatch_seconds"
        ),
        "makespans": {
            str(case["case_key"]): float(case["makespan"]) for case in cases
        },
        "case_signature": _case_signature(cases),
    }


def _bootstrap_improvement_probability(
    pre: Mapping[str, float], post: Mapping[str, float], *, samples: int = 5000
) -> float:
    keys = sorted(pre)
    if keys != sorted(post) or not keys:
        raise ValueError("Paired bootstrap requires identical non-empty cases.")
    deltas = [float(pre[key]) - float(post[key]) for key in keys]
    rng = random.Random(20260824)
    positive = 0
    for _ in range(samples):
        estimate = sum(rng.choice(deltas) for _ in deltas) / len(deltas)
        positive += int(estimate > 0.0)
    return positive / samples


def _service_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", f"{unit}.service"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _service_load_state(unit: str) -> str:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            f"{unit}.service",
            "--property=LoadState",
            "--value",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "not-found"


def _stop_service(unit: str) -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", f"{unit}.service"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _launch_joint_iga(
    *,
    unit: str,
    dataset: Path,
    output_root: Path,
    python: Path,
    workers: int,
    cpu_set: str,
) -> dict[str, Any]:
    """Launch one independent service that serially completes IGA-180/1800."""
    dataset = dataset.resolve()
    output_root = output_root.resolve()
    state_path = output_root / "pipeline_state.json"
    if state_path.is_file():
        try:
            prior = _load(state_path)
        except (OSError, ValueError, TypeError):
            prior = {}
        if (
            prior.get("phase") == "completed"
            and prior.get("dataset_dir") == str(dataset)
            and float(prior.get("iga180_budget_seconds", -1.0)) == 180.0
            and float(prior.get("iga1800_budget_seconds", -1.0)) == 1800.0
        ):
            return {
                "status": "already_completed",
                "unit": f"{unit}.service",
                "dataset_dir": str(dataset),
                "output_root": str(output_root),
                "pipeline_state": str(state_path),
            }

    active_state = _service_state(unit)
    if active_state in {"active", "activating", "reloading", "deactivating"}:
        return {
            "status": "already_running",
            "unit": f"{unit}.service",
            "dataset_dir": str(dataset),
            "output_root": str(output_root),
            "pipeline_state": str(state_path),
        }
    load_state = _service_load_state(unit)
    if load_state != "not-found":
        raise RuntimeError(
            f"Refusing to replace existing inactive IGA unit {unit}.service "
            f"(LoadState={load_state}); inspect it before resuming."
        )

    log = output_root / "logs" / f"{unit}.log"
    mpl = output_root / "mplconfig"
    log.parent.mkdir(parents=True, exist_ok=True)
    mpl.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        "--same-dir",
        "--setenv=PYTHONHASHSEED=0",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1",
        "--setenv=MALLOC_ARENA_MAX=2",
        f"--setenv=MPLCONFIGDIR={mpl}",
        "--property=Type=exec",
        "--property=Restart=no",
        "--property=KillMode=control-group",
        "--property=TimeoutStopSec=60",
        "--property=LimitNOFILE=262144",
        "--property=TasksMax=2048",
        f"--property=AllowedCPUs={cpu_set}",
        f"--property=CPUAffinity={cpu_set}",
        f"--property=StandardOutput=append:{log}",
        f"--property=StandardError=append:{log}",
        str(python),
        "-u",
        str(JOINT_IGA_RUNNER),
        "--dataset-dir",
        str(dataset),
        "--output-root",
        str(output_root),
        "--python",
        str(python),
        "--workers",
        str(workers),
        "--cpu-set",
        cpu_set,
        "--iga180-budget-seconds",
        "180",
        "--iga1800-budget-seconds",
        "1800",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to launch {unit}.service ({result.returncode}): "
            f"{result.stdout.strip()}"
        )
    return {
        "status": "launched",
        "unit": f"{unit}.service",
        "dataset_dir": str(dataset),
        "output_root": str(output_root),
        "pipeline_state": str(state_path),
        "log": str(log.resolve()),
        "workers": workers,
        "cpu_set": cpu_set,
        "budgets_seconds": [180, 1800],
        "teacher_scope": "stage3_full_joint_policy",
        "backends": {"ordinary": "iga", "transporter": "iga"},
    }


def _method_record(
    suite: Path, results_root: Path, method: str
) -> dict[str, Any]:
    manifest_path = suite / "commands" / f"encoder_wave1_{method}.json"
    manifest = _load(manifest_path)
    command = [str(value) for value in manifest["command"]]
    experiment = command[command.index("--experiment_name") + 1]
    expected_epochs = int(command[command.index("--num_episodes") + 1])
    run_dir = results_root / experiment / "run1"
    hard_reasons: list[str] = []
    try:
        status = _load(run_dir / "run_status.json")
    except (FileNotFoundError, json.JSONDecodeError):
        status = {}
        hard_reasons.append("missing_run_status")
    if status.get("status") != "completed":
        hard_reasons.append("run_not_completed")
    if status.get("phase") != "joint_finetune_completed":
        hard_reasons.append("stage3_phase_not_completed")
    if bool(status.get("canary_rejected", False)):
        hard_reasons.append("canary_rejected")
    health = status.get("actor_update_health", {})
    if float(health.get("step_completion_rate", 0.0)) < 0.90:
        hard_reasons.append("actor_step_completion_below_90pct")
    if int(health.get("zero_update_shards", 0)) != 0:
        hard_reasons.append("zero_update_shards")
    if float(health.get("empty_replay_fraction", 0.0)) > 0.01:
        hard_reasons.append("empty_replay_fraction_above_1pct")

    try:
        pre = _eval_record(run_dir / "evaluations" / "pre_ppo.json")
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        pre = None
        hard_reasons.append("missing_or_invalid_pre_ppo_valid60")
    epochs = []
    for epoch in range(1, expected_epochs + 1):
        try:
            record = _eval_record(
                run_dir / "evaluations" / f"epoch_{epoch}.json"
            )
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            hard_reasons.append(f"missing_or_invalid_epoch_{epoch}_valid60")
        else:
            record["epoch"] = epoch
            epochs.append(record)
    signatures = {
        record["case_signature"] for record in ([pre] if pre else []) + epochs
    }
    if len(signatures) > 1:
        hard_reasons.append("within_arm_case_mismatch")
    for record in epochs:
        if (
            record["completion_rate"] != 1.0
            or record["cycle_count"] != 0
            or record["timeout_count"] != 0
        ):
            hard_reasons.append(f"epoch_{record['epoch']}_feasibility_failure")

    best = min(epochs, key=lambda item: item["selection_score"]) if epochs else None
    scientific_reasons: list[str] = []
    bootstrap_probability = None
    severe_regressions = None
    last_two_mean = None
    if pre is not None and best is not None and len(epochs) == expected_epochs:
        last_two_mean = sum(
            record["selection_score"] for record in epochs[-2:]
        ) / min(2, len(epochs))
        bootstrap_probability = _bootstrap_improvement_probability(
            pre["makespans"], best["makespans"]
        )
        severe_regressions = sum(
            best["makespans"][key] - pre["makespans"][key] > 800.0
            for key in pre["makespans"]
        )
        gates = (
            (best["selection_score"] <= pre["selection_score"] * 0.9975,
             "best_composite_tail_improvement_below_0.25pct"),
            (last_two_mean < pre["selection_score"],
             "last_two_epoch_mean_not_better_than_pre_ppo"),
            (best["raw_makespan"] <= pre["raw_makespan"],
             "raw_makespan_regressed"),
            (best["ood_stress_makespan"] <= pre["ood_stress_makespan"] * 1.003,
             "ood_stress_regressed_above_0.3pct"),
            (best["tail_makespan"] <= pre["tail_makespan"] * 1.003,
             "tail_regressed_above_0.3pct"),
            (best["mean_critical_wait"] <= pre["mean_critical_wait"],
             "critical_wait_regressed"),
            (best["mean_dispatch_wait"] <= pre["mean_dispatch_wait"],
             "dispatch_wait_regressed"),
            (severe_regressions == 0, "new_case_regression_above_800"),
            (bootstrap_probability > 0.80,
             "bootstrap_improvement_probability_not_above_0.80"),
        )
        scientific_reasons.extend(reason for passed, reason in gates if not passed)
    else:
        scientific_reasons.append("incomplete_scientific_metrics")

    def clean(record: Mapping[str, Any] | None) -> dict[str, Any]:
        if record is None:
            return {}
        return {
            key: value for key, value in record.items()
            if key not in {"case_signature", "makespans"}
        }

    return {
        "method": method,
        "manifest": str(manifest_path.resolve()),
        "experiment_name": experiment,
        "run_dir": str(run_dir.resolve()),
        "source_stage2_sha256": str(
            manifest.get("source_stage2", {}).get("sha256", "")
        ),
        "hard_gate_reasons": sorted(set(hard_reasons)),
        "scientific_gate_reasons": sorted(set(scientific_reasons)),
        "hard_eligible": not hard_reasons,
        "admitted": not hard_reasons and not scientific_reasons,
        "pre_ppo": clean(pre),
        "epochs": [clean(record) for record in epochs],
        "best_epoch": best.get("epoch") if best else None,
        "best_selection_score": best.get("selection_score") if best else None,
        "last_two_mean_selection_score": last_two_mean,
        "bootstrap_improvement_probability": bootstrap_probability,
        "severe_regressions_above_800": severe_regressions,
        "actor_update_health": health,
        "case_signature": pre.get("case_signature") if pre else None,
    }


def analyze(suite: Path, results_root: Path) -> dict[str, Any]:
    suite = suite.resolve()
    records = [_method_record(suite, results_root.resolve(), method) for method in METHODS]
    digests = {record["source_stage2_sha256"] for record in records}
    if len(digests) != 1 or not next(iter(digests), ""):
        raise ValueError("E0-E3 do not share one immutable Stage2 source.")
    signatures = {record["case_signature"] for record in records if record["case_signature"]}
    if len(signatures) != 1:
        raise ValueError("E0-E3 do not share identical Valid60 cases.")
    baselines = [record["pre_ppo"].get("selection_score") for record in records]
    if any(value is None for value in baselines) or max(baselines) - min(baselines) > 1e-6:
        raise ValueError("E0-E3 Pre-PPO baselines are inconsistent.")
    admitted = [record for record in records if record["admitted"]]
    ranking = sorted(
        admitted,
        key=lambda item: (
            float(item["last_two_mean_selection_score"]),
            float(item["best_selection_score"]),
            str(item["method"]),
        ),
    )
    selected = ranking[0]["method"] if ranking else None
    signature_digest = hashlib.sha256(
        json.dumps(next(iter(signatures)), sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "created_at": _now(),
        "suite_dir": str(suite),
        "source_stage2_sha256": next(iter(digests)),
        "valid60_case_signature_sha256": signature_digest,
        "pre_ppo_selection_score": float(baselines[0]),
        "admission_contract": {
            "best_composite_tail_improvement_min": 0.0025,
            "last_two_mean_must_beat_pre_ppo": True,
            "raw_must_not_regress": True,
            "ood_stress_and_tail_max_regression": 0.003,
            "critical_and_dispatch_wait_must_not_regress": True,
            "max_new_case_regression": 800.0,
            "bootstrap_improvement_probability_min_exclusive": 0.80,
            "max_empty_replay_fraction": 0.01,
        },
        "methods": [
            {key: value for key, value in record.items() if key != "case_signature"}
            for record in records
        ],
        "admitted_methods": [record["method"] for record in ranking],
        "ranking": [record["method"] for record in ranking],
        "selected_method": selected,
        "next_phase": "prepare_lateness_wave2" if selected else "stop_no_admission",
        "post_training_iga_contract": {
            "launch_regardless_of_admission": True,
            "teacher_scope": "stage3_full_joint_policy",
            "teacher_method": "joint_iga_all",
            "aircraft_policy": "iga",
            "ordinary_mobile_policy": "iga",
            "transporter_r014_policy": "iga",
            "budgets_seconds": [180, 1800],
            "nested_continuation": True,
        },
    }
    _atomic_json(suite / "analysis" / "encoder_wave1_selection.json", payload)
    # Avoid clever formatting expressions in the persisted scientific report.
    report_lines = [
        "# Stage3 Encoder Wave 1 自动分析", "",
        f"生成时间：`{payload['created_at']}`", "",
        "| 方法 | 硬门控 | 晋级 | 最佳分数 | 相对 Pre-PPO | 后两轮均值 | Bootstrap P(改善) | 未通过原因 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        pre = record["pre_ppo"].get("selection_score")
        best = record["best_selection_score"]
        delta = (best / pre - 1.0) if pre and best is not None else None
        report_lines.append(
            "| {method} | {hard} | {admitted} | {best} | {delta} | {last} | {prob} | {reasons} |".format(
                method=record["method"],
                hard="是" if record["hard_eligible"] else "否",
                admitted="是" if record["admitted"] else "否",
                best="—" if best is None else f"{best:.3f}",
                delta="—" if delta is None else f"{delta:+.3%}",
                last=("—" if record["last_two_mean_selection_score"] is None
                      else f"{record['last_two_mean_selection_score']:.3f}"),
                prob=("—" if record["bootstrap_improvement_probability"] is None
                      else f"{record['bootstrap_improvement_probability']:.3f}"),
                reasons=", ".join(
                    record["hard_gate_reasons"] + record["scientific_gate_reasons"]
                ) or "—",
            )
        )
    report_lines.extend((
        "", f"选择结果：**{selected or '无方法晋级'}**。", "",
        "本脚本不会在无方法超过共同 Pre-PPO 时兜底启动 Formal。", "",
        "训练与 Valid60 全部结束后，将独立启动全联合 IGA-180，随后以其为"
        "唯一 warm-start 自动继续 IGA-1800；该标注不受方法是否晋级影响。", "",
    ))
    _atomic_text(
        suite / "analysis" / "encoder_wave1_selection.md",
        "\n".join(report_lines),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--unit-prefix", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=172800.0)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-joint-iga", action="store_true")
    parser.add_argument(
        "--joint-iga-dataset", type=Path, default=DEFAULT_JOINT_IGA_DATASET
    )
    parser.add_argument("--joint-iga-output-root", type=Path)
    parser.add_argument("--joint-iga-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--joint-iga-workers", type=int, default=60)
    # Reuse the physical cores released by GPU1's Stage3 suite without
    # stealing GPU0's still-isolated Stage2 cores if that formal run remains.
    parser.add_argument(
        "--joint-iga-cpu-set", default="36-71,108-143"
    )
    parser.add_argument("--joint-iga-unit")
    args = parser.parse_args()
    if args.joint_iga_workers < 1:
        parser.error("--joint-iga-workers must be positive")
    if not args.joint_iga_dataset.is_dir():
        parser.error(f"joint IGA dataset does not exist: {args.joint_iga_dataset}")
    if not args.joint_iga_python.is_file():
        parser.error(f"joint IGA python does not exist: {args.joint_iga_python}")
    state_path = args.suite.resolve() / "autopilot" / "state.json"
    joint_iga_output = (
        args.joint_iga_output_root
        or args.suite.resolve() / "joint_iga_tune60"
    )
    joint_iga_unit = (
        args.joint_iga_unit or f"{args.unit_prefix}-joint-iga-tune60"
    )
    joint_iga_plan = {
        "enabled": not args.skip_joint_iga,
        "launch_after_all_trainers_terminal": True,
        "unit": f"{joint_iga_unit}.service",
        "dataset_dir": str(args.joint_iga_dataset.resolve()),
        "output_root": str(joint_iga_output.resolve()),
        "workers": args.joint_iga_workers,
        "cpu_set": args.joint_iga_cpu_set,
        "cpu_isolation": (
            "reuse_gpu1_stage3_partition_without_competing_with_gpu0_stage2"
        ),
        "teacher_scope": "stage3_full_joint_policy",
        "aircraft_policy": "iga",
        "ordinary_mobile_policy": "iga",
        "transporter_r014_policy": "iga",
        "sequence": ["iga180", "iga1800_from_verified_iga180"],
    }
    try:
        if not args.analyze_only:
            deadline = time.monotonic() + args.timeout_seconds
            units = [f"{args.unit_prefix}-encoder_wave1-{method}" for method in METHODS]
            while True:
                states = {unit: _service_state(unit) for unit in units}
                if not {"active", "activating", "reloading", "deactivating"}.intersection(states.values()):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for E0-E3: {states}")
                _atomic_json(state_path, {
                    "schema_version": 1, "updated_at": _now(),
                    "phase": "waiting_for_encoder_wave1", "service_states": states,
                    "post_training_joint_iga_plan": joint_iga_plan,
                })
                time.sleep(args.poll_seconds)
        payload = analyze(args.suite, args.results_root)
        joint_iga = None
        if not args.analyze_only:
            _stop_service(f"{args.unit_prefix}-encoder_wave1-eval")
            if not args.skip_joint_iga:
                joint_iga = _launch_joint_iga(
                    unit=joint_iga_unit,
                    dataset=args.joint_iga_dataset,
                    output_root=joint_iga_output,
                    python=args.joint_iga_python,
                    workers=args.joint_iga_workers,
                    cpu_set=args.joint_iga_cpu_set,
                )
        _atomic_json(state_path, {
            "schema_version": 1, "updated_at": _now(),
            "phase": (
                "joint_iga_labeling"
                if joint_iga and joint_iga["status"] != "already_completed"
                else (
                    "joint_iga_completed"
                    if joint_iga
                    else payload["next_phase"]
                )
            ),
            "selected_method": payload["selected_method"],
            "model_next_phase": payload["next_phase"],
            "analysis": str(
                args.suite.resolve() / "analysis/encoder_wave1_selection.json"
            ),
            "joint_iga": joint_iga,
            "post_training_joint_iga_plan": joint_iga_plan,
        })
        return 0
    except Exception as error:
        if not args.analyze_only:
            _stop_service(f"{args.unit_prefix}-encoder_wave1-eval")
        _atomic_json(state_path, {
            "schema_version": 1, "updated_at": _now(), "phase": "failed",
            "error_type": type(error).__name__, "error": str(error),
            "post_training_joint_iga_plan": joint_iga_plan,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
