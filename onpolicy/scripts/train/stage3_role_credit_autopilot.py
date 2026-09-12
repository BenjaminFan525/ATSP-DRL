#!/usr/bin/env python3
"""Select the best Stage3 Wave-1 role-credit arm and launch Formal.

The controller is intentionally independent from the Wave-1 launcher process.
It can therefore be restarted after a login/session loss, and it can attach to
an already-running Wave-1 suite.  Selection is based only on matched full
Valid60 evaluations; shard canaries are safety gates, never ranking inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("B0", "B1", "B2", "B3")
EPOCH_EVAL_RE = re.compile(r"epoch_(\d+)\.json$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            dict(payload), ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        ) + "\n",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _flag_value(command: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"Expected exactly one valued flag {flag}.")
    value = str(command[positions[0] + 1])
    if value.startswith("--"):
        raise ValueError(f"Missing value for {flag}.")
    return value


def _finite(summary: Mapping[str, Any], key: str) -> float:
    try:
        value = float(summary[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing numeric evaluation field {key}.") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite evaluation field {key}={value}.")
    return value


def _case_signature(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation has no per-case records.")
    signature = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Invalid evaluation case record.")
        key = str(case.get("case_key", ""))
        digest = str(case.get("case_sha256", ""))
        if not key or not digest:
            raise ValueError("Evaluation case lacks key or content digest.")
        signature.append((key, digest))
    if len(set(signature)) != len(signature):
        raise ValueError("Evaluation contains duplicate cases.")
    return tuple(sorted(signature))


def _eval_record(path: Path, expected_cases: int = 60) -> dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"Evaluation summary missing: {path}")
    case_count = int(_finite(summary, "eval_case_count"))
    if case_count != expected_cases:
        raise ValueError(
            f"Expected Valid{expected_cases}, got {case_count}: {path}"
        )
    completion = _finite(summary, "eval_completion_rate")
    cycles = int(_finite(summary, "eval_cycle_count"))
    timeouts = int(_finite(summary, "eval_timeout_count"))
    if completion != 1.0 or cycles != 0 or timeouts != 0:
        raise ValueError(
            f"Invalid evaluation completion in {path}: "
            f"completion={completion}, cycles={cycles}, timeouts={timeouts}"
        )
    cases = payload.get("cases", [])
    wait_values = [
        float(case["resource_wait_seconds"])
        for case in cases
        if isinstance(case, Mapping)
        and case.get("resource_wait_seconds") is not None
        and math.isfinite(float(case["resource_wait_seconds"]))
    ]
    return {
        "path": str(path.resolve()),
        "selection_score": _finite(summary, "eval_selection_score"),
        "raw_makespan": _finite(summary, "eval_raw_makespan"),
        "iid_makespan": _finite(summary, "eval_iid_makespan"),
        "composite_makespan": _finite(summary, "eval_composite_makespan"),
        "tail_makespan": _finite(summary, "eval_tail_makespan"),
        "ood_stress_makespan": _finite(
            summary, "eval_distribution_ood_stress_makespan"
        ),
        "ood_scale_makespan": _finite(
            summary, "eval_distribution_ood_scale_makespan"
        ),
        "completion_rate": completion,
        "cycle_count": cycles,
        "timeout_count": timeouts,
        "mean_resource_wait_seconds": (
            sum(wait_values) / len(wait_values) if wait_values else None
        ),
        "case_signature": _case_signature(payload),
    }


def _hard_gate_reasons(status: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if status.get("status") != "completed":
        reasons.append("status_not_completed")
    if status.get("training_stage") != "joint_finetune":
        reasons.append("wrong_training_stage")
    if status.get("phase") != "joint_finetune_completed":
        reasons.append("phase_not_completed")
    if status.get("canary_rejected") is not False:
        reasons.append("canary_rejected")
    health = status.get("actor_update_health")
    if not isinstance(health, Mapping):
        health = {}
    completion = health.get(
        "step_completion_rate", status.get("actor_step_completion_rate", 0.0)
    )
    try:
        completion = float(completion)
    except (TypeError, ValueError):
        completion = 0.0
    if not math.isfinite(completion) or completion < 0.90:
        reasons.append("actor_step_completion_below_90_percent")
    try:
        zero_updates = int(health.get("zero_update_shards", 0))
    except (TypeError, ValueError):
        zero_updates = -1
    if zero_updates != 0:
        reasons.append("zero_update_shards")
    for key, expected, reason in (
        ("eval_completion_rate", 1.0, "incomplete_validation"),
        ("eval_cycle_count", 0.0, "validation_cycle"),
        ("eval_timeout_count", 0.0, "validation_timeout"),
    ):
        try:
            actual = float(status.get(key, math.nan))
        except (TypeError, ValueError):
            actual = math.nan
        if not math.isfinite(actual) or actual != expected:
            reasons.append(reason)
    return reasons


def _signature_digest(signature: Sequence[tuple[str, str]]) -> str:
    encoded = json.dumps(list(signature), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _method_record(
    method: str,
    manifest_path: Path,
    results_root: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("profile") != "wave1" or manifest.get("causal_arm") != method:
        raise ValueError(f"Wave-1 manifest mismatch: {manifest_path}")
    command = [str(value) for value in manifest.get("command", ())]
    experiment = _flag_value(command, "--experiment_name")
    expected_epochs = int(_flag_value(command, "--num_episodes"))
    if expected_epochs < 1:
        raise ValueError("Wave-1 must contain at least one epoch.")
    run_dir = results_root / experiment / "run1"
    status_path = run_dir / "run_status.json"
    try:
        status = _load_json(status_path)
        hard_gate_reasons = _hard_gate_reasons(status)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        status = {}
        hard_gate_reasons = ["missing_or_invalid_run_status"]
    pre: dict[str, Any] | None = None
    try:
        pre = _eval_record(run_dir / "evaluations" / "pre_ppo.json")
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        hard_gate_reasons.append("missing_or_invalid_pre_ppo_valid60")
    epochs: list[dict[str, Any]] = []
    for epoch in range(1, expected_epochs + 1):
        try:
            record = _eval_record(
                run_dir / "evaluations" / f"epoch_{epoch}.json"
            )
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            hard_gate_reasons.append(
                f"missing_or_invalid_epoch_{epoch}_valid60"
            )
        else:
            record["epoch"] = epoch
            epochs.append(record)
    signatures = set()
    if pre is not None:
        signatures.add(pre["case_signature"])
    signatures.update(record["case_signature"] for record in epochs)
    if len(signatures) > 1:
        hard_gate_reasons.append("within_method_validation_case_mismatch")
    scores = [record["selection_score"] for record in epochs]
    best = epochs[min(range(len(scores)), key=scores.__getitem__)] if scores else None
    final = epochs[-1] if epochs else None
    health = status.get("actor_update_health", {})
    if not isinstance(health, Mapping):
        health = {}
    pre_score = pre["selection_score"] if pre is not None else None
    complete_metrics = pre_score is not None and best is not None and final is not None
    relative_delta = (
        float(best["selection_score"] / pre_score - 1.0)
        if complete_metrics and pre_score > 0.0
        else None
    )
    return {
        "method": method,
        "manifest": str(manifest_path.resolve()),
        "experiment_name": experiment,
        "run_dir": str(run_dir.resolve()),
        "source_stage2_sha256": str(
            manifest.get("source_stage2", {}).get("sha256", "")
        ),
        "eligible": not hard_gate_reasons,
        "hard_gate_reasons": hard_gate_reasons,
        "pre_ppo": (
            {key: value for key, value in pre.items() if key != "case_signature"}
            if pre is not None else {}
        ),
        "epochs": [
            {key: value for key, value in record.items() if key != "case_signature"}
            for record in epochs
        ],
        "best_epoch": int(best["epoch"]) if best is not None else None,
        "best_selection_score": (
            float(best["selection_score"]) if best is not None else None
        ),
        "mean_selection_score": (
            float(sum(scores) / len(scores)) if scores else None
        ),
        "final_selection_score": (
            float(final["selection_score"]) if final is not None else None
        ),
        "delta_vs_pre_ppo": (
            float(best["selection_score"] - pre_score)
            if complete_metrics else None
        ),
        "relative_delta_vs_pre_ppo": relative_delta,
        "beat_pre_ppo": bool(
            complete_metrics and best["selection_score"] < pre_score
        ),
        "best_iid_makespan": (
            float(best["iid_makespan"]) if best is not None else None
        ),
        "best_ood_stress_makespan": (
            float(best["ood_stress_makespan"]) if best is not None else None
        ),
        "best_ood_scale_makespan": (
            float(best["ood_scale_makespan"]) if best is not None else None
        ),
        "best_tail_makespan": (
            float(best["tail_makespan"]) if best is not None else None
        ),
        "best_mean_resource_wait_seconds": (
            best["mean_resource_wait_seconds"] if best is not None else None
        ),
        "actor_step_completion_rate": float(
            health.get(
                "step_completion_rate",
                status.get("actor_step_completion_rate", 0.0),
            )
        ),
        "zero_update_shards": int(health.get("zero_update_shards", 0)),
        "post_update_old_policy_kl_max": float(
            health.get("post_update_old_policy_kl_max", 0.0)
        ),
        "case_signature": pre["case_signature"] if pre is not None else None,
    }


def _ranking_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(record["best_selection_score"]),
        float(record["mean_selection_score"]),
        float(record["final_selection_score"]),
        float(record["best_ood_stress_makespan"]),
        float(record["best_tail_makespan"]),
        str(record["method"]),
    )


def _markdown_report(payload: Mapping[str, Any]) -> str:
    def number(value: Any, spec: str = ".3f") -> str:
        return "—" if value is None else format(float(value), spec)

    lines = [
        "# Stage3 Wave 1 自动方法选择",
        "",
        f"生成时间：`{payload['created_at']}`",
        "",
        "选择规则：仅使用完整且同案例的 Valid60；先执行硬门控，再按最佳 "
        "composite-tail、两轮均值、最终轮、OOD-stress、tail 顺序排序。",
        "",
        "| 排名 | 方法 | 合格 | Pre-PPO | Epoch 1 | Epoch 2 | 最佳轮 | 最佳分数 | 相对基线 | OOD-stress | Tail | 更新完成率 | Zero update |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranks = {method: index + 1 for index, method in enumerate(payload["ranking"])}
    for record in payload["methods"]:
        epoch_scores = [item["selection_score"] for item in record["epochs"]]
        epoch_scores += [None, None]
        rank = ranks.get(record["method"], "—")
        pre_score = record["pre_ppo"].get("selection_score")
        lines.append(
            f"| {rank} | {record['method']} | "
            f"{'是' if record['eligible'] else '否'} | "
            f"{number(pre_score)} | "
            f"{number(epoch_scores[0])} | {number(epoch_scores[1])} | "
            f"{record['best_epoch'] if record['best_epoch'] is not None else '—'} | "
            f"{number(record['best_selection_score'])} | "
            f"{number(record['relative_delta_vs_pre_ppo'], '+.3%')} | "
            f"{number(record['best_ood_stress_makespan'])} | "
            f"{number(record['best_tail_makespan'])} | "
            f"{record['actor_step_completion_rate']:.1%} | "
            f"{record['zero_update_shards']} |"
        )
    lines.extend((
        "",
        f"最终选择：**{payload['selected_method']}**。",
        "",
        str(payload["selection_reason"]),
        "",
    ))
    return "\n".join(lines)


def analyze_wave1(
    suite_dir: Path,
    results_root: Path,
) -> dict[str, Any]:
    suite_dir = suite_dir.resolve()
    records = [
        _method_record(
            method,
            suite_dir / "commands" / f"wave1_{method}.json",
            results_root.resolve(),
        )
        for method in METHODS
    ]
    source_digests = {record["source_stage2_sha256"] for record in records}
    if len(source_digests) != 1 or not next(iter(source_digests), ""):
        raise ValueError("Wave-1 methods do not share one immutable Stage2 source.")
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        reasons = {
            record["method"]: record["hard_gate_reasons"] for record in records
        }
        raise RuntimeError(f"No Wave-1 method passed hard gates: {reasons}")
    signatures = {record["case_signature"] for record in eligible}
    if None in signatures or len(signatures) != 1:
        raise ValueError("Wave-1 methods were not evaluated on identical Valid60 cases.")
    baselines = [record["pre_ppo"]["selection_score"] for record in eligible]
    if max(baselines) - min(baselines) > 1e-6:
        raise ValueError("Wave-1 methods have inconsistent Pre-PPO baselines.")
    ranking = sorted(eligible, key=_ranking_key)
    selected = ranking[0]
    baseline_note = (
        "该方法的最佳完整 Valid60 优于共同 Pre-PPO 基线。"
        if selected["beat_pre_ppo"]
        else "四组中没有方法优于共同 Pre-PPO 基线；按用户要求仍选择硬门控后分数最低的方法进入 Formal。"
    )
    common_signature = next(iter(signatures))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": _now(),
        "suite_dir": str(suite_dir),
        "selection_metric": "best_full_valid60_composite_tail",
        "selection_rule": [
            "hard_gates",
            "best_selection_score",
            "mean_selection_score",
            "final_selection_score",
            "best_ood_stress_makespan",
            "best_tail_makespan",
            "method_name",
        ],
        "source_stage2_sha256": next(iter(source_digests)),
        "valid60_case_signature_sha256": _signature_digest(common_signature),
        "pre_ppo_selection_score": float(baselines[0]),
        "methods": [
            {key: value for key, value in record.items() if key != "case_signature"}
            for record in records
        ],
        "eligible_methods": [record["method"] for record in ranking],
        "ranking": [record["method"] for record in ranking],
        "selected_method": selected["method"],
        "selection_reason": (
            f"{selected['method']} 在所有硬门控合格方法中取得最低的最佳完整 Valid60 "
            f"composite-tail={selected['best_selection_score']:.6f}（epoch "
            f"{selected['best_epoch']}，相对 Pre-PPO "
            f"{selected['relative_delta_vs_pre_ppo']:+.4%}）。{baseline_note}"
        ),
    }
    analysis_dir = suite_dir / "analysis"
    _atomic_json(analysis_dir / "wave1_method_selection.json", payload)
    _atomic_text(
        analysis_dir / "wave1_method_selection.md",
        _markdown_report(payload),
    )
    return payload


def _service_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", f"{unit}.service"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _write_state(path: Path, phase: str, **details: Any) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": _now(),
        "phase": phase,
        **details,
    }
    _atomic_json(path, payload)


def _wait_for_wave1(
    unit_prefix: str,
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, str]:
    deadline = time.monotonic() + timeout_seconds
    units = [f"{unit_prefix}-wave1-{method}" for method in METHODS]
    while True:
        states = {unit: _service_state(unit) for unit in units}
        active = {
            "active", "activating", "reloading", "deactivating"
        }.intersection(states.values())
        if not active:
            return states
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Wave-1 services: {states}")
        _write_state(state_path, "waiting_for_wave1", service_states=states)
        time.sleep(poll_seconds)


def _stop_service(unit: str) -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", f"{unit}.service"],
        check=False,
    )


def _wait_gpu_idle(gpu: int, timeout_seconds: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = subprocess.run(
            [
                "nvidia-smi", f"--id={gpu}",
                "--query-compute-apps=pid", "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0 and not result.stdout.strip():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"GPU{gpu} remained busy after Wave-1: {result.stdout.strip()}"
            )
        time.sleep(2.0)


def _launch_formal(args: argparse.Namespace, selected_method: str) -> None:
    trainer_unit = f"{args.formal_unit_prefix}-formal-{selected_method}"
    evaluator_unit = f"{args.formal_unit_prefix}-formal-eval"
    states = {
        trainer_unit: _service_state(trainer_unit),
        evaluator_unit: _service_state(evaluator_unit),
    }
    if states[trainer_unit] == "active" and states[evaluator_unit] == "active":
        print(f"[Autopilot] Formal already active: {states}", flush=True)
        return
    formal_experiment = (
        args.results_root
        / f"{args.formal_run_tag}_formal_{selected_method}_seed1"
    )
    if formal_experiment.exists():
        raise FileExistsError(
            "Refusing ambiguous Formal relaunch because output already exists: "
            f"{formal_experiment}"
        )
    environment = os.environ.copy()
    environment.update({
        "PYTHON": str(args.python.resolve()),
        "PROFILE": "formal",
        "START": "1",
        "WAIT_FOR_COMPLETION": "0",
        "METHOD_FILTER": selected_method,
        "AUTO_FORMAL": "0",
        "RUN_TAG": args.formal_run_tag,
        "UNIT_PREFIX": args.formal_unit_prefix,
        "SUITE_DIR": str(args.formal_suite.resolve()),
        "SOURCE_CHECKPOINT": str(args.source_checkpoint.resolve()),
        "VALIDATION_DIR": str(args.validation_dir.resolve()),
        "GPU_PHASE_LOCK": f"/tmp/{args.formal_unit_prefix}-formal.ppo.lock",
    })
    subprocess.run(
        [str(args.launcher.resolve())],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    states = {
        trainer_unit: _service_state(trainer_unit),
        evaluator_unit: _service_state(evaluator_unit),
    }
    if states[trainer_unit] != "active" or states[evaluator_unit] != "active":
        raise RuntimeError(f"Formal services did not become active: {states}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-suite", type=Path, required=True)
    parser.add_argument("--wave-unit-prefix", required=True)
    parser.add_argument("--formal-suite", type=Path, required=True)
    parser.add_argument("--formal-run-tag", required=True)
    parser.add_argument("--formal-unit-prefix", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument(
        "--results-root", type=Path,
        default=ROOT / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo",
    )
    parser.add_argument(
        "--launcher", type=Path,
        default=ROOT / "onpolicy/scripts/train/launch_stage3_role_clock_four_gpu1.sh",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise ValueError("Polling and timeout values must be positive.")
    state_path = args.wave_suite.resolve() / "autopilot" / "state.json"
    try:
        if not args.analyze_only:
            service_states = _wait_for_wave1(
                args.wave_unit_prefix,
                state_path,
                args.poll_seconds,
                args.timeout_seconds,
            )
        else:
            service_states = {}
        _write_state(
            state_path, "analyzing_wave1", service_states=service_states
        )
        selection = analyze_wave1(args.wave_suite, args.results_root)
        selected = str(selection["selected_method"])
        _write_state(
            state_path,
            "wave1_selected",
            service_states=service_states,
            selected_method=selected,
            selection_file=str(
                args.wave_suite.resolve()
                / "analysis/wave1_method_selection.json"
            ),
        )
        print(selection["selection_reason"], flush=True)
        if args.analyze_only:
            return 0
        _stop_service(f"{args.wave_unit_prefix}-wave1-eval")
        _wait_gpu_idle(args.gpu)
        _write_state(state_path, "launching_formal", selected_method=selected)
        _launch_formal(args, selected)
        _write_state(
            state_path,
            "formal_started",
            selected_method=selected,
            formal_suite=str(args.formal_suite.resolve()),
            formal_run_tag=args.formal_run_tag,
            formal_unit_prefix=args.formal_unit_prefix,
        )
        print(
            f"[Autopilot] Formal started method={selected} "
            f"suite={args.formal_suite.resolve()}",
            flush=True,
        )
        return 0
    except Exception as error:
        if not args.analyze_only:
            _stop_service(f"{args.wave_unit_prefix}-wave1-eval")
        _write_state(
            state_path,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
