#!/usr/bin/env python3
"""Fail-closed admission gate for the eight source-relative RL canaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onpolicy.scripts.train.prepare_stage3_source_relative_rl import (  # noqa: E402
    ARMS,
    EXPECTED_SOURCE_SHA256,
    arm_factors,
    atomic_json,
    validate_manifest,
)


def finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite canary metric: {value!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-reserved-limit-gib", type=float, default=20.0)
    args = parser.parse_args()
    suite = args.suite_dir.expanduser().resolve()
    manifests = sorted((suite / "commands").glob("canary_g*_U*K*R*.json"))
    if len(manifests) != 8:
        raise RuntimeError(f"Expected 8 canary manifests, found {len(manifests)}.")

    observed_arms: set[str] = set()
    baseline_sha256 = ""
    implementation_bundle_sha256 = ""
    entries = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if manifest["phase"] != "canary":
            raise ValueError(f"Non-canary manifest in gate: {manifest_path}")
        arm = str(manifest["arm"])
        factors = arm_factors(arm)
        observed_arms.add(arm)
        current_baseline_sha = str(manifest["source_baseline"]["sha256"])
        if baseline_sha256 and current_baseline_sha != baseline_sha256:
            raise RuntimeError("Canary arms used different source baselines.")
        baseline_sha256 = current_baseline_sha
        current_implementation_sha = str(
            manifest["implementation_snapshot"]["bundle_sha256"]
        )
        if (
            implementation_bundle_sha256
            and current_implementation_sha != implementation_bundle_sha256
        ):
            raise RuntimeError("Canary arms used different implementation snapshots.")
        implementation_bundle_sha256 = current_implementation_sha

        status_paths = sorted(
            Path(manifest["expected_result_parent"]).glob("run*/run_status.json")
        )
        if len(status_paths) != 1:
            raise RuntimeError(
                f"{arm} needs exactly one run_status.json; found "
                f"{len(status_paths)}."
            )
        status_path = status_paths[0]
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed" or status.get("event") != "completed":
            raise RuntimeError(f"{arm} did not complete: {status.get('status')}")
        source_path = Path(manifest["source_stage2"]["path"]).resolve()
        if Path(str(status.get("checkpoint_dir", ""))).resolve() != source_path:
            raise RuntimeError(f"{arm} reports another initial checkpoint.")

        completion = finite(status.get("eval_completion_rate", 0.0))
        timeouts = int(status.get("eval_timeout_count", -1))
        cycles = int(status.get("eval_cycle_count", -1))
        peak = finite(status.get("cuda_peak_reserved_gib", 0.0))
        health = status.get("actor_update_health", {})
        actor_steps = finite(health.get("actual_optimizer_steps", 0.0))
        planned_steps = finite(health.get("planned_optimizer_steps", 0.0))
        completion_rate = finite(health.get("step_completion_rate", 0.0))
        zero_shards = int(health.get("zero_update_shards", -1))
        if completion != 1.0 or timeouts != 0 or cycles != 0:
            raise RuntimeError(
                f"{arm} invalid evaluation: completion={completion}, "
                f"timeouts={timeouts}, cycles={cycles}."
            )
        if actor_steps <= 0.0 or planned_steps <= 0.0 or completion_rate < 0.95:
            raise RuntimeError(f"{arm} did not complete healthy Actor updates.")
        if zero_shards != 0:
            raise RuntimeError(f"{arm} has {zero_shards} zero-update shards.")
        if not 0.0 < peak < args.peak_reserved_limit_gib:
            raise RuntimeError(
                f"{arm} peak reserved {peak:.3f} GiB violates the "
                f"(0,{args.peak_reserved_limit_gib}) gate."
            )

        sequential_steps = finite(status.get("role_sequential_macro_steps", 0.0))
        sequential_ess = finite(status.get("role_sequential_min_ess", 0.0))
        if sequential_steps <= 0.0:
            raise RuntimeError(f"{arm} did not exercise sequential role PPO.")
        if not 0.5 <= sequential_ess / sequential_steps <= 1.000001:
            raise RuntimeError(f"{arm} sequential ESS gate failed.")
        diagnostics = status.get("role_event_return_diagnostics", {})
        if finite(diagnostics.get("cost_conservation_max_abs_error", 1.0)) >= 2e-3:
            raise RuntimeError(f"{arm} failed role-return mass conservation.")

        group_norms = status.get("actor_group_grad_norms", {})
        shared_norm = finite(
            group_norms.get("shared_encoder", {}).get("raw", float("nan"))
        )
        observed_frozen = bool(status.get("shared_encoder_frozen", True))
        if factors["shared_encoder_unfreeze"]:
            if observed_frozen or shared_norm <= 0.0:
                raise RuntimeError(f"{arm} did not exercise a shared-encoder update.")
        elif not observed_frozen or shared_norm != 0.0:
            raise RuntimeError(f"{arm} did not keep the shared encoder frozen.")

        reference_path = str(status.get("bc_reference_resolved_path", ""))
        reference_health_kl = finite(
            health.get("post_update_bc_reference_kl_max", 0.0)
        )
        if factors["source_reference_kl"]:
            if Path(reference_path).resolve() != source_path:
                raise RuntimeError(f"{arm} did not resolve KL reference to C0.")
            if not bool(status.get("adaptive_bc_reference_enabled", False)):
                raise RuntimeError(f"{arm} did not exercise adaptive reference KL.")
            if reference_health_kl <= 0.0:
                raise RuntimeError(f"{arm} has no positive source-reference KL evidence.")
            coefficient = finite(status.get("bc_reference_kl_coef_after", -1.0))
            if not 0.02 <= coefficient <= 1.0:
                raise RuntimeError(f"{arm} adaptive reference coefficient escaped bounds.")
        else:
            if reference_path or reference_health_kl != 0.0:
                raise RuntimeError(f"{arm} unexpectedly used a KL reference.")

        baseline_path = str(status.get("paired_case_baseline_path", ""))
        paired_count = int(status.get("paired_case_baseline_count", -1))
        cvar_enabled = bool(status.get("cvar_policy_enabled", False))
        if factors["source_paired_residual"]:
            if Path(baseline_path).resolve() != Path(
                manifest["source_baseline"]["path"]
            ).resolve():
                raise RuntimeError(f"{arm} used another paired baseline.")
            if paired_count != int(manifest["source_baseline"]["case_count"]):
                raise RuntimeError(f"{arm} loaded incomplete paired baselines.")
            finite(status.get("paired_case_delta_mean", float("nan")))
            if (
                not cvar_enabled
                or not bool(status.get("cvar_case_metric_paired_delta", False))
                or finite(status.get("cvar_tail_case_count", 0.0)) <= 0.0
                or finite(status.get("cvar_mass_conservation_error", 1.0)) >= 1e-5
            ):
                raise RuntimeError(f"{arm} did not exercise paired-delta CVaR.")
        else:
            if baseline_path or paired_count != 0 or cvar_enabled:
                raise RuntimeError(f"{arm} unexpectedly used paired residual credit.")

        entries.append({
            "arm": arm,
            "factors": factors,
            "physical_gpu": int(manifest["physical_gpu"]),
            "status_path": str(status_path),
            "eval_makespan": finite(status["eval_makespan"]),
            "actor_optimizer_steps": actor_steps,
            "actor_step_completion_rate": completion_rate,
            "shared_encoder_grad_norm": shared_norm,
            "source_reference_kl_max": reference_health_kl,
            "paired_case_delta_mean": finite(
                status.get("paired_case_delta_mean", 0.0)
            ),
            "cuda_peak_reserved_gib": peak,
        })

    if observed_arms != set(ARMS):
        raise RuntimeError(f"Canary factorial incomplete: {sorted(observed_arms)}")
    gate = {
        "schema_version": 1,
        "status": "passed_8_of_8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_stage2_sha256": EXPECTED_SOURCE_SHA256,
        "source_baseline_sha256": baseline_sha256,
        "implementation_bundle_sha256": implementation_bundle_sha256,
        "peak_reserved_limit_gib": float(args.peak_reserved_limit_gib),
        "max_peak_reserved_gib": max(
            entry["cuda_peak_reserved_gib"] for entry in entries
        ),
        "completion_rate": 1.0,
        "cycle_count": 0,
        "timeout_count": 0,
        "arms": entries,
    }
    atomic_json(args.output.expanduser().resolve(), gate)
    print(f"[SourceRelativeCanaryGate] passed 8/8; wrote {args.output}")


if __name__ == "__main__":
    main()
