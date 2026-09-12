#!/usr/bin/env python3
"""Strictly select Stage3 PPO-gain arms and optionally launch Wave 2.

Every arm is compared only with its own immutable Pre-PPO policy.  Epoch 1 is
critic-only calibration; scientific ranking uses Actor epochs 2-5 and fails
closed if calibration changes any actor tensor or deterministic Valid60 result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


WAVE_METHODS = {
    "ppo_gain_wave1": ("P0", "P1", "P2", "P3"),
    "ppo_gain_wave2": ("Q0", "Q1", "Q2", "Q3"),
    "credit_happo_wave1": ("N0", "N1", "N2", "N3"),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                dict(payload), ensure_ascii=False, indent=2,
                sort_keys=True, allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _flag_value(command: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"Expected one valued flag {flag}.")
    return str(command[positions[0] + 1])


def _terminal(path: Path) -> bool:
    try:
        status = _load(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return str(status.get("status", "")) in {
        "completed", "failed", "stopped", "canary_rejected",
    }


def _wait(run_dirs: Mapping[str, Path], timeout_seconds: float) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        pending = [
            method for method, run_dir in run_dirs.items()
            if not _terminal(run_dir / "run_status.json")
        ]
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Stage3 arms: {pending}")
        print(f"[PPOGain] waiting for {pending}", flush=True)
        time.sleep(30.0)


def _case_key(record: Mapping[str, Any]) -> str:
    return str(record.get("case_key", record.get("case_id", "")))


def _evaluation(path: Path) -> dict[str, Any]:
    payload = _load(path)
    summary = dict(payload.get("summary", {}))
    cases = list(payload.get("cases", ()))
    case_count = int(summary.get("eval_case_count", len(cases)))
    valid = int(summary.get("eval_valid", 1))
    if (
        case_count != 60
        or len(cases) != 60
        or valid != 1
        or float(summary.get("eval_completion_rate", 0.0)) != 1.0
        or int(summary.get("eval_cycle_count", 1)) != 0
        or int(summary.get("eval_timeout_count", 1)) != 0
    ):
        raise ValueError(f"Invalid deterministic Valid60 evaluation: {path}")
    makespans = {
        _case_key(record): float(record["makespan"]) for record in cases
    }
    if len(makespans) != 60 or not all(
        key and math.isfinite(value) for key, value in makespans.items()
    ):
        raise ValueError(f"Invalid per-case makespans: {path}")
    signature = tuple(sorted(
        (
            _case_key(record),
            str(record.get("case_sha256", "")),
        )
        for record in cases
    ))
    raw = float(summary.get(
        "eval_raw_makespan", np.mean(list(makespans.values()))
    ))
    ordered = sorted(makespans.values())
    return {
        "path": str(path.resolve()),
        "raw_cmax": raw,
        "selection_score": float(summary.get("eval_selection_score", raw)),
        "tail_worst10": float(np.mean(ordered[-6:])),
        "ood_stress_cmax": float(summary.get(
            "eval_distribution_ood_stress_makespan", math.inf
        )),
        "resource_wait": float(np.mean([
            float(record.get("resource_wait_seconds", 0.0))
            for record in cases
        ])),
        "critical_lateness": float(np.mean([
            float(record.get(
                "resource_avoidable_critical_lateness_seconds", 0.0
            ))
            for record in cases
        ])),
        "signature": signature,
        "makespans": makespans,
    }


def _paired_statistics(
    pre: Mapping[str, float],
    post: Mapping[str, float],
    *,
    samples: int = 20000,
    seed: int = 20260826,
) -> dict[str, Any]:
    keys = sorted(pre)
    if keys != sorted(post) or len(keys) != 60:
        raise ValueError("Paired statistics require identical Valid60 cases.")
    deltas = np.asarray(
        [float(post[key]) - float(pre[key]) for key in keys],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, deltas.size, size=(samples, deltas.size))
    means = deltas[indices].mean(axis=1)
    return {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "probability_improves": float(np.mean(means < 0.0)),
        "wins": int(np.sum(deltas < 0.0)),
        "ties": int(np.sum(deltas == 0.0)),
        "losses": int(np.sum(deltas > 0.0)),
        "regressions_over_800": int(np.sum(deltas > 800.0)),
    }


def _optimizer_max_step(state: Mapping[str, Any]) -> float:
    maximum = 0.0
    for item in dict(state.get("state", {})).values():
        if not isinstance(item, Mapping) or "step" not in item:
            continue
        step = item["step"]
        if isinstance(step, torch.Tensor):
            step = step.detach().cpu().item()
        maximum = max(maximum, float(step))
    return maximum


def _calibration_audit(run_dir: Path, pre: dict, epoch1: dict) -> dict:
    checkpoint_path = run_dir / "models" / "checkpoint_Epoch1.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    actor_pairs = {
        role: (
            checkpoint.get(f"{role}_actor_summary_before_ppo"),
            checkpoint.get(f"{role}_actor_summary_after_ppo"),
        )
        for role in ("plane", "resource", "shared")
    }
    actor_hash_unchanged = all(
        isinstance(before, Mapping)
        and isinstance(after, Mapping)
        and before.get("sha256") == after.get("sha256")
        for before, after in actor_pairs.values()
    )
    case_deltas = [
        abs(epoch1["makespans"][key] - value)
        for key, value in pre["makespans"].items()
    ]
    health = dict(checkpoint.get("actor_update_health", {}))
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "actor_hash_unchanged": bool(actor_hash_unchanged),
        "actor_hashes": {
            role: {
                "before": (before or {}).get("sha256"),
                "after": (after or {}).get("sha256"),
            }
            for role, (before, after) in actor_pairs.items()
        },
        "valid60_max_abs_delta": float(max(case_deltas, default=math.inf)),
        "critic_optimizer_max_step": _optimizer_max_step(
            checkpoint.get("critic_optim", {})
        ),
        "actor_planned_optimizer_steps": float(
            health.get("planned_optimizer_steps", 0.0)
        ),
        "actor_actual_optimizer_steps": float(
            health.get("actual_optimizer_steps", 0.0)
        ),
    }


def _clean_eval(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items()
        if key not in {"signature", "makespans"}
    }


def _arm_record(
    suite: Path,
    results_root: Path,
    profile: str,
    method: str,
) -> dict[str, Any]:
    manifest_path = suite / "commands" / f"{profile}_{method}.json"
    manifest = _load(manifest_path)
    command = [str(value) for value in manifest.get("command", ())]
    if manifest.get("profile") != profile or manifest.get("causal_arm") != method:
        raise ValueError(f"Manifest arm mismatch: {manifest_path}")
    experiment = _flag_value(command, "--experiment_name")
    epochs_expected = int(_flag_value(command, "--num_episodes"))
    run_dir = results_root / experiment / "run1"
    reasons: list[str] = []
    try:
        status = _load(run_dir / "run_status.json")
    except (OSError, ValueError, json.JSONDecodeError):
        status = {}
        reasons.append("missing_or_invalid_run_status")
    if status.get("status") != "completed":
        reasons.append("run_not_completed")
    if status.get("phase") != "joint_finetune_completed":
        reasons.append("stage3_phase_not_completed")
    if bool(status.get("canary_rejected", False)):
        reasons.append("canary_rejected")

    try:
        pre = _evaluation(run_dir / "evaluations" / "pre_ppo.json")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pre = None
        reasons.append("missing_or_invalid_pre_ppo_valid60")
    epoch_records = []
    for epoch in range(1, epochs_expected + 1):
        try:
            record = _evaluation(
                run_dir / "evaluations" / f"epoch_{epoch}.json"
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            reasons.append(f"missing_or_invalid_epoch_{epoch}_valid60")
        else:
            record["epoch"] = epoch
            epoch_records.append(record)
    signatures = {
        record["signature"]
        for record in ([pre] if pre is not None else []) + epoch_records
    }
    if len(signatures) > 1:
        reasons.append("within_arm_valid60_case_mismatch")

    calibration = None
    if pre is not None and epoch_records and epoch_records[0]["epoch"] == 1:
        try:
            calibration = _calibration_audit(run_dir, pre, epoch_records[0])
        except (OSError, ValueError, KeyError, RuntimeError):
            reasons.append("missing_or_invalid_critic_calibration_checkpoint")
        else:
            if not calibration["actor_hash_unchanged"]:
                reasons.append("critic_calibration_changed_actor_hash")
            if calibration["valid60_max_abs_delta"] > 1e-4:
                reasons.append("critic_calibration_changed_policy_output")
            if calibration["critic_optimizer_max_step"] <= 0.0:
                reasons.append("critic_calibration_executed_no_critic_step")
            if (
                calibration["actor_planned_optimizer_steps"] != 0.0
                or calibration["actor_actual_optimizer_steps"] != 0.0
            ):
                reasons.append("critic_calibration_executed_actor_step")
    else:
        reasons.append("missing_critic_calibration_epoch")

    actor_epochs = [record for record in epoch_records if record["epoch"] >= 2]
    expected_actor_epochs = max(0, epochs_expected - 1)
    if len(actor_epochs) != expected_actor_epochs:
        reasons.append("incomplete_actor_ppo_epochs")
    health = dict(status.get("actor_update_health", {}))
    if float(health.get("step_completion_rate", 0.0)) < 0.90:
        reasons.append("actor_step_completion_below_90pct")
    if int(health.get("zero_update_shards", 0)) != 0:
        reasons.append("zero_update_shards")
    if float(health.get("empty_replay_fraction", 1.0)) > 0.01:
        reasons.append("empty_replay_fraction_above_1pct")
    expected_shards = expected_actor_epochs * (
        int(_flag_value(command, "--train_sampling_size"))
        // int(_flag_value(command, "--n_rollout_threads"))
    )
    if int(health.get("update_shards", -1)) != expected_shards:
        reasons.append("unexpected_actor_update_shard_count")
    target_kl = float(_flag_value(command, "--target_kl"))
    if float(health.get("post_update_old_policy_kl_max", math.inf)) > (
        target_kl + 1e-9
    ):
        reasons.append("post_update_old_policy_kl_above_target")

    best = min(
        actor_epochs,
        key=lambda item: (item["raw_cmax"], item["tail_worst10"]),
        default=None,
    )
    paired = None
    last_two_raw = None
    if pre is not None and best is not None and len(actor_epochs) == 4:
        minimum_best_gain = (
            0.0025 if profile == "credit_happo_wave1" else 0.005
        )
        minimum_last_two_gain = (
            0.0015 if profile == "credit_happo_wave1" else 0.0
        )
        minimum_probability = (
            0.80 if profile == "credit_happo_wave1" else 0.85
        )
        paired = _paired_statistics(pre["makespans"], best["makespans"])
        last_two_raw = float(np.mean([
            record["raw_cmax"] for record in actor_epochs[-2:]
        ]))
        if best["raw_cmax"] > pre["raw_cmax"] * (1.0 - minimum_best_gain):
            reasons.append("best_raw_cmax_improvement_below_gate")
        if last_two_raw > pre["raw_cmax"] * (1.0 - minimum_last_two_gain):
            reasons.append("last_two_raw_cmax_improvement_below_gate")
        if paired["probability_improves"] < minimum_probability:
            reasons.append("bootstrap_improvement_probability_below_gate")
        if paired["median_delta"] > 0.0:
            reasons.append("paired_median_delta_above_zero")
        if paired["regressions_over_800"] != 0:
            reasons.append("new_case_regression_over_800")
    else:
        reasons.append("incomplete_scientific_comparison")

    return {
        "method": method,
        "manifest": str(manifest_path.resolve()),
        "experiment_name": experiment,
        "run_dir": str(run_dir.resolve()),
        "source_stage2_sha256": str(
            manifest.get("source_stage2", {}).get("sha256", "")
        ),
        "base_credit_method": manifest.get(
            "architecture_contract", {}
        ).get("base_credit_method"),
        "admitted": not reasons,
        "gate_reasons": sorted(set(reasons)),
        "pre_ppo": _clean_eval(pre) if pre else {},
        "calibration": calibration,
        "actor_epochs": [_clean_eval(record) for record in actor_epochs],
        "best_epoch": best.get("epoch") if best else None,
        "best_metrics": _clean_eval(best) if best else {},
        "last_two_raw_cmax_mean": last_two_raw,
        "paired_vs_own_pre": paired,
        "actor_update_health": health,
        "case_signature": pre.get("signature") if pre else None,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Stage3 PPO-gain {report['profile']} selection",
        "",
        "Every arm is gated against its own Pre-PPO Valid60; resource wait is "
        "diagnostic only.",
        "",
        "| Arm | admitted | Pre | best epoch | best Cmax | gain | last-2 | "
        "bootstrap P | median Δ | worst-10 | wait |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        pre = arm.get("pre_ppo", {})
        best = arm.get("best_metrics", {})
        paired = arm.get("paired_vs_own_pre") or {}
        pre_raw = float(pre.get("raw_cmax", math.nan))
        best_raw = float(best.get("raw_cmax", math.nan))
        gain = (pre_raw - best_raw) / pre_raw if pre_raw > 0 else math.nan
        lines.append(
            f"| {arm['method']} | {'yes' if arm['admitted'] else 'no'} | "
            f"{pre_raw:.2f} | {arm.get('best_epoch') or '-'} | "
            f"{best_raw:.2f} | {gain:.3%} | "
            f"{float(arm.get('last_two_raw_cmax_mean') or math.nan):.2f} | "
            f"{float(paired.get('probability_improves', math.nan)):.3f} | "
            f"{float(paired.get('median_delta', math.nan)):+.2f} | "
            f"{float(best.get('tail_worst10', math.nan)):.2f} | "
            f"{float(best.get('resource_wait', math.nan)):.2f} |"
        )
        if arm["gate_reasons"]:
            lines.append(
                f"\n{arm['method']} rejection: "
                + ", ".join(arm["gate_reasons"]) + "."
            )
    lines.extend((
        "",
        f"Selected: **{report.get('selected_method') or 'none'}**.",
        "",
    ))
    return "\n".join(lines)


def analyze_phase(
    suite: Path,
    results_root: Path,
    profile: str,
) -> dict[str, Any]:
    methods = WAVE_METHODS[profile]
    arms = [
        _arm_record(suite.resolve(), results_root.resolve(), profile, method)
        for method in methods
    ]
    source_digests = {arm["source_stage2_sha256"] for arm in arms}
    if len(source_digests) != 1 or not next(iter(source_digests), ""):
        raise ValueError(f"{methods} do not share one Stage2 source.")
    signatures = {arm["case_signature"] for arm in arms if arm["case_signature"]}
    if len(signatures) != 1:
        raise ValueError(f"{methods} do not share identical Valid60 cases.")
    pre_raw = [arm["pre_ppo"].get("raw_cmax") for arm in arms]
    if any(value is None for value in pre_raw) or max(pre_raw) - min(pre_raw) > 1e-6:
        raise ValueError(f"{methods} do not share an identical Pre-PPO baseline.")
    candidates = [arm for arm in arms if arm["admitted"]]
    selected = min(
        candidates,
        key=lambda arm: (
            arm["best_metrics"]["raw_cmax"],
            arm["last_two_raw_cmax_mean"],
            arm["best_metrics"]["tail_worst10"],
            arm["method"],
        ),
        default=None,
    )
    report = {
        "schema_version": 1,
        "created_unix_time": time.time(),
        "profile": profile,
        "strictly_compared_to_own_pre_ppo": True,
        "calibration_epoch": 1,
        "actor_ppo_epochs": [2, 3, 4, 5],
        "raw_cmax_minimum_improvement_fraction": (
            0.0025 if profile == "credit_happo_wave1" else 0.005
        ),
        "last_two_minimum_improvement_fraction": (
            0.0015 if profile == "credit_happo_wave1" else 0.0
        ),
        "bootstrap_minimum_probability": (
            0.80 if profile == "credit_happo_wave1" else 0.85
        ),
        "resource_wait_is_gate": False,
        "selected_method": selected["method"] if selected else None,
        "selected_epoch": selected["best_epoch"] if selected else None,
        "arms": arms,
    }
    output = suite / "analysis" / f"{profile}_selection.json"
    _atomic_json(output, report)
    markdown = suite / "analysis" / f"{profile}_selection.md"
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def _stop_unit(unit: str) -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", f"{unit}.service"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _launch_wave2(args: argparse.Namespace, selected: str) -> None:
    environment = dict(os.environ)
    environment.update({
        "PROFILE": "ppo_gain_wave2",
        "START": "1",
        "AUTO_ANALYZE": "1",
        "AUTO_NEXT_WAVE": "0",
        "METHOD_FILTER": "all",
        "BASE_CREDIT_METHOD": selected,
        "RUN_TAG": args.next_run_tag,
        "UNIT_PREFIX": args.next_unit_prefix,
        "SUITE_DIR": str(
            args.results_log_root.resolve() / args.next_run_tag
        ),
    })
    result = subprocess.run(
        [str(args.launcher.resolve())],
        cwd=str(args.launcher.resolve().parents[3]),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to launch Wave 2 ({result.returncode}): {result.stdout}"
        )
    print(result.stdout.strip(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--results-log-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--profile", choices=tuple(WAVE_METHODS), required=True)
    parser.add_argument("--unit-prefix", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7 * 86400)
    parser.add_argument("--auto-launch-next-wave", action="store_true")
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--next-run-tag", default="")
    parser.add_argument("--next-unit-prefix", default="")
    args = parser.parse_args()
    methods = WAVE_METHODS[args.profile]
    run_dirs = {
        method: args.results_root
        / f"{args.run_tag}_{args.profile}_{method}_seed1"
        / "run1"
        for method in methods
    }
    _wait(run_dirs, args.timeout_seconds)
    _stop_unit(f"{args.unit_prefix}-{args.profile}-eval")
    report = analyze_phase(args.suite, args.results_root, args.profile)
    selected = report.get("selected_method")
    print(
        f"[PPOGain] profile={args.profile} selected={selected} "
        f"report={args.suite / 'analysis'}",
        flush=True,
    )
    if args.auto_launch_next_wave:
        if args.profile != "ppo_gain_wave1":
            raise ValueError("Only Wave 1 can automatically launch Wave 2.")
        if not selected:
            print(
                "[PPOGain] no arm strictly beat its own Pre-PPO; Wave 2 is "
                "blocked by design.",
                flush=True,
            )
            return
        if not args.launcher or not args.next_run_tag or not args.next_unit_prefix:
            raise ValueError("Wave-2 launch metadata is incomplete.")
        _launch_wave2(args, str(selected))


if __name__ == "__main__":
    main()
