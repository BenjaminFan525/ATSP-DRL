#!/usr/bin/env python3
"""Fail-closed admission gate from eight real canaries to formal Wave 1."""

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

from onpolicy.scripts.train.prepare_stage3_credit_factorial import (  # noqa: E402
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
    manifests = sorted((suite / "commands").glob("canary_g*_W*.json"))
    if len(manifests) != 8:
        raise RuntimeError(f"Expected 8 canary manifests, found {len(manifests)}.")
    observed_arms = set()
    entries = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if manifest["phase"] != "canary":
            raise ValueError(f"Non-canary manifest in gate: {manifest_path}")
        arm = str(manifest["arm"])
        factors = arm_factors(arm)
        observed_arms.add(arm)
        status_paths = sorted(
            Path(manifest["expected_result_parent"]).glob("run*/run_status.json")
        )
        if len(status_paths) != 1:
            raise RuntimeError(
                f"{arm} needs exactly one run_status.json; found {len(status_paths)}."
            )
        status_path = status_paths[0]
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed" or status.get("event") != "completed":
            raise RuntimeError(f"{arm} did not complete: {status.get('status')}")
        if Path(str(status.get("checkpoint_dir", ""))).resolve() != Path(
            manifest["source_stage2"]["path"]
        ).resolve():
            raise RuntimeError(f"{arm} status reports another source checkpoint.")
        completion = finite(status.get("eval_completion_rate", 0.0))
        timeouts = int(status.get("eval_timeout_count", -1))
        cycles = int(status.get("eval_cycle_count", -1))
        actor_steps = finite(status.get("actor_optimizer_steps", 0.0))
        peak = finite(status.get("cuda_peak_reserved_gib", 0.0))
        if completion != 1.0 or timeouts != 0 or cycles != 0:
            raise RuntimeError(
                f"{arm} invalid evaluation: completion={completion}, "
                f"timeouts={timeouts}, cycles={cycles}."
            )
        if actor_steps <= 0.0:
            raise RuntimeError(f"{arm} did not exercise a real Actor update.")
        if not 0.0 < peak < args.peak_reserved_limit_gib:
            raise RuntimeError(
                f"{arm} peak reserved {peak:.3f} GiB violates the "
                f"(0,{args.peak_reserved_limit_gib}) gate."
            )
        if status.get("shared_encoder_frozen") is not True:
            raise RuntimeError(f"{arm} did not keep the shared encoder frozen.")
        diagnostics = status.get("role_event_return_diagnostics", {})
        if finite(diagnostics.get("cost_conservation_max_abs_error", 1.0)) >= 2e-3:
            raise RuntimeError(f"{arm} failed exact role-return mass conservation.")
        observed_v2 = finite(
            diagnostics.get("event_credit_mode_critical_path_v2", 0.0)
        ) == 1.0
        if observed_v2 != factors["critical_path_v2"]:
            raise RuntimeError(f"{arm} critical-path-v2 path was not exercised.")
        cf_count = finite(status.get("counterfactual_topk_mass_count", 0.0))
        if factors["counterfactual_q"]:
            if cf_count <= 0.0:
                raise RuntimeError(f"{arm} counterfactual path was not exercised.")
            mass_mean = finite(status.get("counterfactual_topk_mass_mean", 0.0))
            mass_min = finite(status.get("counterfactual_topk_mass_min", 0.0))
            if not 0.0 < mass_min <= mass_mean <= 1.000001:
                raise RuntimeError(f"{arm} invalid represented probability mass.")
        elif cf_count != 0.0:
            raise RuntimeError(f"{arm} unexpectedly evaluated counterfactual Q.")
        sequential_steps = finite(status.get("role_sequential_macro_steps", 0.0))
        if factors["role_sequential_ppo"]:
            if sequential_steps <= 0.0:
                raise RuntimeError(f"{arm} sequential macro step was not exercised.")
            # Trainer metrics are accumulated over macro optimizer groups;
            # completion already proves every individual group passed the
            # hard ESS gate, while this normalized value audits its scale.
            sequential_ess = finite(
                status.get("role_sequential_min_ess", 0.0)
            ) / sequential_steps
            if not 0.5 <= sequential_ess <= 1.000001:
                raise RuntimeError(f"{arm} sequential ESS gate failed.")
        elif sequential_steps != 0.0:
            raise RuntimeError(f"{arm} unexpectedly used sequential PPO.")
        else:
            sequential_ess = 1.0
        group_norms = status.get("actor_group_grad_norms")
        if not isinstance(group_norms, dict):
            raise RuntimeError(f"{arm} has no per-group gradient evidence.")
        for group in (
            "shared_encoder", "plane_actor", "device_actor",
            "transporter_actor",
        ):
            for key in ("raw", "clipped", "clip_applied"):
                finite(group_norms.get(group, {}).get(key, float("nan")))
        entries.append({
            "arm": arm,
            "physical_gpu": int(manifest["physical_gpu"]),
            "status_path": str(status_path),
            "eval_makespan": finite(status["eval_makespan"]),
            "eval_completion_rate": completion,
            "actor_optimizer_steps": actor_steps,
            "cuda_peak_reserved_gib": peak,
            "counterfactual_topk_mass_count": cf_count,
            "role_sequential_macro_steps": sequential_steps,
            "role_sequential_ess_per_macro": sequential_ess,
            "cost_conservation_max_abs_error": finite(
                diagnostics.get("cost_conservation_max_abs_error", 0.0)
            ),
        })
    if observed_arms != set(ARMS):
        raise RuntimeError(
            f"Canary factorial incomplete: {sorted(observed_arms)}"
        )
    gate = {
        "schema_version": 1,
        "status": "passed_8_of_8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_stage2_sha256": EXPECTED_SOURCE_SHA256,
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
    print(f"[CreditCanaryGate] passed 8/8; wrote {args.output}")


if __name__ == "__main__":
    main()
