#!/usr/bin/env python3
"""Run and verify the post-training Stage3 full-joint IGA-180/1800 chain.

This is deliberately a separate process from the training autopilot.  It lets
the four GPU trainers and the shared validator terminate first, then uses the
released CPU capacity for case-parallel, resumable IGA labeling.  The first
phase controls aircraft and every mobile-device role jointly.  The second
phase is a nested continuation from the replay-verified IGA-180 incumbent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels import (  # noqa: E402
    BACKENDS,
    NESTED_WARM_SEARCH_CONTRACT_VERSIONS,
    SEARCH_CONTRACT_VERSION,
    TEACHER_METHOD,
    TEACHER_SCOPE,
    _case_sha256,
    _contract,
    _planning_contract,
    _teacher_reusable,
)
from onpolicy.envs.HKBZ.experiment.eval_common import list_case_folders  # noqa: E402
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv  # noqa: E402


GENERATOR = (
    ROOT / "onpolicy/envs/HKBZ/experiment/generate_stage3_joint_iga_labels.py"
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_cases(dataset_dir: Path, max_cases: int, case_offset: int) -> list[str]:
    return list_case_folders(
        str(dataset_dir), int(max_cases), case_offset=int(case_offset)
    )


def _planning_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return _planning_contract(
        lookahead_margin=args.device_lookahead_safety_margin,
        future_intent_horizon=args.device_future_intent_horizon,
        future_intent_mode=args.device_future_intent_mode,
        frontier_max_requests=args.device_frontier_max_requests,
        request_capacity_per_plane=args.device_request_capacity_per_plane,
        release_aware_eta=args.resource_release_aware_eta,
        reservation_mode=args.device_lookahead_reservation_mode,
        reservation_grace_seconds=args.device_reservation_grace_seconds,
        slack_forecast_seconds=args.resource_slack_forecast_seconds,
    )


def _contract_planning_kwargs(planning: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "future_intent_horizon": int(planning["device_future_intent_horizon"]),
        "future_intent_mode": str(planning["device_future_intent_mode"]),
        "frontier_max_requests": int(planning["device_frontier_max_requests"]),
        "request_capacity_per_plane": int(
            planning["device_request_capacity_per_plane"]
        ),
        "release_aware_eta": bool(planning["resource_release_aware_eta"]),
        "reservation_mode": str(
            planning["device_lookahead_reservation_mode"]
        ),
        "reservation_grace_seconds": float(
            planning["device_reservation_grace_seconds"]
        ),
        "slack_forecast_seconds": float(
            planning["resource_slack_forecast_seconds"]
        ),
    }


def _validate_summary(
    output_dir: Path,
    dataset_dir: Path,
    cases: Sequence[str],
    *,
    cumulative_budget: float,
    lookahead_margin: float,
    configured_budget: float,
    planning_contract: Mapping[str, Any],
    accepted_search_contract_versions: Sequence[int] = (SEARCH_CONTRACT_VERSION,),
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    summary = _load(summary_path)
    required_summary = {
        "status": "completed",
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "dataset_dir": str(dataset_dir.resolve()),
        "expected_case_count": len(cases),
        "completed_case_count": len(cases),
        "missing_cases": [],
        "planning_contract": dict(planning_contract),
    }
    mismatch = {
        key: (summary.get(key), expected)
        for key, expected in required_summary.items()
        if summary.get(key) != expected
    }
    actual_budget = float(summary.get("nominal_cumulative_budget_seconds", -1.0))
    if not math.isclose(actual_budget, cumulative_budget, abs_tol=1e-9):
        mismatch["nominal_cumulative_budget_seconds"] = (
            actual_budget,
            cumulative_budget,
        )
    if mismatch:
        raise ValueError(f"Invalid Stage3 IGA summary {summary_path}: {mismatch}")

    makespans: dict[str, float] = {}
    for case in cases:
        case_dir = dataset_dir / case
        expected = _contract(
            case,
            _case_sha256(case_dir),
            cumulative_budget,
            lookahead_margin,
            **_contract_planning_kwargs(planning_contract),
        )
        teacher_path = output_dir / "teachers" / f"{case}.json"
        result_path = output_dir / "cases" / f"{case}.json"
        reusable = any(
            _teacher_reusable(
                teacher_path,
                {**expected, "search_contract_version": int(version)},
            )
            for version in accepted_search_contract_versions
        )
        if not reusable:
            raise ValueError(f"Unverified Stage3 full-joint teacher: {teacher_path}")
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing Stage3 IGA case result: {result_path}")
        teacher = _load(teacher_path)
        if teacher.get("backends") != BACKENDS:
            raise ValueError(f"IGA does not control every mobile role: {teacher_path}")
        search_contract = int(teacher.get("search_contract_version", -1))
        if search_contract not in set(accepted_search_contract_versions):
            raise ValueError(f"Stale Stage3 IGA search contract: {teacher_path}")
        search = teacher.get("search", {})
        nested_search_contract = int(search.get("search_contract_version", -1))
        if nested_search_contract != search_contract:
            raise ValueError(f"Inconsistent Stage3 IGA search contract: {teacher_path}")
        recorded_budget = float(
            search.get("configured_additional_budget_seconds", -1.0)
        )
        if not math.isclose(recorded_budget, configured_budget, abs_tol=1e-9):
            raise ValueError(f"Wrong Stage3 IGA configured budget: {teacher_path}")
        search_wall = float(search.get("optimization_wall_seconds", -1.0))
        max_generations = int(search.get("max_generations", -1))
        completed_generations = int(search.get("completed_generations", -1))
        budget_exhausted = search_wall >= 0.95 * configured_budget
        generation_limit_reached = (
            max_generations >= 0 and completed_generations >= max_generations
        )
        if not budget_exhausted and not generation_limit_reached:
            raise ValueError(
                f"Stage3 IGA stopped before exhausting its search contract: "
                f"{teacher_path} wall={search_wall:.3f}s budget={configured_budget:.3f}s"
            )
        makespans[case] = float(teacher["makespan"])
    return {
        "summary": str(summary_path.resolve()),
        "teacher_dir": str((output_dir / "teachers").resolve()),
        "case_count": len(cases),
        "makespan_mean": float(summary["makespan_mean"]),
        "makespan_median": float(summary["makespan_median"]),
        "makespans": makespans,
    }


def validate_nested_continuation(
    iga180_dir: Path,
    iga1800_dir: Path,
    dataset_dir: Path,
    cases: Sequence[str],
    *,
    iga180_budget: float,
    iga1800_budget: float,
    lookahead_margin: float,
    planning_contract: Mapping[str, Any],
) -> dict[str, Any]:
    first = _validate_summary(
        iga180_dir,
        dataset_dir,
        cases,
        cumulative_budget=iga180_budget,
        lookahead_margin=lookahead_margin,
        configured_budget=iga180_budget,
        planning_contract=planning_contract,
        accepted_search_contract_versions=tuple(
            NESTED_WARM_SEARCH_CONTRACT_VERSIONS
        ),
    )
    final = _validate_summary(
        iga1800_dir,
        dataset_dir,
        cases,
        cumulative_budget=iga1800_budget,
        lookahead_margin=lookahead_margin,
        configured_budget=iga1800_budget - iga180_budget,
        planning_contract=planning_contract,
    )
    regressions: list[dict[str, Any]] = []
    for case in cases:
        first_teacher = iga180_dir / "teachers" / f"{case}.json"
        final_teacher = iga1800_dir / "teachers" / f"{case}.json"
        first_payload = _load(first_teacher)
        final_payload = _load(final_teacher)
        warm = final_payload.get("search", {}).get("warm_start", {})
        if (
            warm.get("kind") != "verified_stage3_nested_incumbent"
            or not math.isclose(
                float(warm.get("source_nominal_budget_seconds", -1.0)),
                iga180_budget,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(f"IGA-1800 did not inherit verified IGA-180: {final_teacher}")
        first_makespan = float(first_payload["makespan"])
        final_makespan = float(final_payload["makespan"])
        if final_makespan > first_makespan + 1e-9:
            regressions.append(
                {
                    "case": case,
                    "iga180": first_makespan,
                    "iga1800": final_makespan,
                }
            )
    if regressions:
        raise ValueError(f"Nested IGA-1800 regressed versus IGA-180: {regressions}")
    first.pop("makespans")
    final.pop("makespans")
    return {"iga180": first, "iga1800": final, "nested_regression_count": 0}


def _generator_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    time_budget: float,
    cumulative_budget: float,
    warm_start_dir: Path | None,
) -> list[str]:
    command = [
        str(args.python),
        "-u",
        str(GENERATOR),
        "--dataset-dir",
        str(args.dataset_dir),
        "--output-dir",
        str(output_dir),
        "--time-budget-seconds",
        str(time_budget),
        "--cumulative-budget-seconds",
        str(cumulative_budget),
        "--workers",
        str(args.workers),
        "--population",
        str(args.population),
        "--max-generations",
        str(args.max_generations),
        "--max-steps",
        str(args.max_steps),
        "--max-plane-agents",
        str(args.max_plane_agents),
        "--max-device-num",
        str(args.max_device_num),
        "--device-lookahead-safety-margin",
        str(args.device_lookahead_safety_margin),
        "--device-future-intent-horizon",
        str(args.device_future_intent_horizon),
        "--device-future-intent-mode",
        str(args.device_future_intent_mode),
        "--device-frontier-max-requests",
        str(args.device_frontier_max_requests),
        "--device-request-capacity-per-plane",
        str(args.device_request_capacity_per_plane),
        (
            "--resource-release-aware-eta"
            if args.resource_release_aware_eta
            else "--no-resource-release-aware-eta"
        ),
        "--device-lookahead-reservation-mode",
        str(args.device_lookahead_reservation_mode),
        "--device-reservation-grace-seconds",
        str(args.device_reservation_grace_seconds),
        "--resource-slack-forecast-seconds",
        str(args.resource_slack_forecast_seconds),
        "--seed",
        str(args.seed),
    ]
    if args.max_cases:
        command.extend(("--max-cases", str(args.max_cases)))
    if args.case_offset:
        command.extend(("--case-offset", str(args.case_offset)))
    if warm_start_dir is not None:
        command.extend(
            (
                "--warm-start-dir",
                str(warm_start_dir),
                "--warm-start-budget-seconds",
                str(args.iga180_budget_seconds),
            )
        )
    elif args.plane_warm_dir is not None:
        command.extend(
            (
                "--plane-warm-dir",
                str(args.plane_warm_dir),
                "--resource-warm-dir",
                str(args.resource_warm_dir),
            )
        )
    return command


def _run_with_retries(
    command: Sequence[str],
    *,
    attempts: int,
    phase: str,
    state_path: Path,
    base_state: Mapping[str, Any],
) -> None:
    for attempt in range(1, attempts + 1):
        _atomic_json(
            state_path,
            {
                **base_state,
                "updated_at": _now(),
                "phase": f"{phase}_running",
                "attempt": attempt,
                "command": list(command),
            },
        )
        print(
            f"[Stage3IGA-Pipeline] phase={phase} attempt={attempt}/{attempts}",
            flush=True,
        )
        result = subprocess.run(list(command), cwd=ROOT, check=False)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, list(command))
        print(
            f"[Stage3IGA-Pipeline] phase={phase} exit={result.returncode}; "
            "resume incomplete cases",
            flush=True,
        )


def run(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    iga180_dir = output_root / "iga180"
    iga1800_dir = output_root / "iga1800"
    state_path = output_root / "pipeline_state.json"
    cases = _expected_cases(dataset_dir, args.max_cases, args.case_offset)
    if not cases:
        raise ValueError(f"No Stage3 IGA cases found in {dataset_dir}")
    additional_1800 = args.iga1800_budget_seconds - args.iga180_budget_seconds
    planning_contract = _planning_from_args(args)
    base_state = {
        "schema_version": 1,
        "started_at": _now(),
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "backends": BACKENDS,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "case_count": len(cases),
        "workers": args.workers,
        "cpu_set": args.cpu_set,
        "iga180_budget_seconds": args.iga180_budget_seconds,
        "iga1800_budget_seconds": args.iga1800_budget_seconds,
        "planning_contract": planning_contract,
    }
    command180 = _generator_command(
        args,
        output_dir=iga180_dir,
        time_budget=args.iga180_budget_seconds,
        cumulative_budget=args.iga180_budget_seconds,
        warm_start_dir=None,
    )
    command1800 = _generator_command(
        args,
        output_dir=iga1800_dir,
        time_budget=additional_1800,
        cumulative_budget=args.iga1800_budget_seconds,
        warm_start_dir=iga180_dir,
    )
    if args.check_only:
        print(
            json.dumps(
                {
                    **base_state,
                    "phase": "check_only",
                    "iga180_command": command180,
                    "iga1800_command": command1800,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        if not args.reuse_existing_iga180:
            _run_with_retries(
                command180,
                attempts=args.phase_attempts,
                phase="iga180",
                state_path=state_path,
                base_state=base_state,
            )
        first = _validate_summary(
            iga180_dir,
            dataset_dir,
            cases,
            cumulative_budget=args.iga180_budget_seconds,
            lookahead_margin=args.device_lookahead_safety_margin,
            configured_budget=args.iga180_budget_seconds,
            planning_contract=planning_contract,
            accepted_search_contract_versions=tuple(
                NESTED_WARM_SEARCH_CONTRACT_VERSIONS
            ),
        )
        first.pop("makespans")
        _atomic_json(
            state_path,
            {
                **base_state,
                "updated_at": _now(),
                "phase": (
                    "iga180_reused"
                    if args.reuse_existing_iga180
                    else "iga180_completed"
                ),
                "iga180": first,
            },
        )
        _run_with_retries(
            command1800,
            attempts=args.phase_attempts,
            phase="iga1800",
            state_path=state_path,
            base_state=base_state,
        )
        verification = validate_nested_continuation(
            iga180_dir,
            iga1800_dir,
            dataset_dir,
            cases,
            iga180_budget=args.iga180_budget_seconds,
            iga1800_budget=args.iga1800_budget_seconds,
            lookahead_margin=args.device_lookahead_safety_margin,
            planning_contract=planning_contract,
        )
        _atomic_json(
            state_path,
            {
                **base_state,
                "updated_at": _now(),
                "completed_at": _now(),
                "phase": "completed",
                "verification": verification,
            },
        )
        print(
            f"[Stage3IGA-Pipeline] completed {len(cases)} cases: "
            f"{iga180_dir} -> {iga1800_dir}",
            flush=True,
        )
        return 0
    except Exception as error:
        _atomic_json(
            state_path,
            {
                **base_state,
                "updated_at": _now(),
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--cpu-set", default="0-143")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--max-generations", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-plane-agents", type=int, default=24)
    parser.add_argument("--max-device-num", type=int, default=80)
    parser.add_argument("--device-lookahead-safety-margin", type=float, default=60.0)
    parser.add_argument("--device-future-intent-horizon", type=int, default=1)
    parser.add_argument(
        "--device-future-intent-mode",
        choices=("legacy_one", "bounded_frontier"),
        default="legacy_one",
    )
    parser.add_argument("--device-frontier-max-requests", type=int, default=2)
    parser.add_argument("--device-request-capacity-per-plane", type=int, default=0)
    parser.add_argument(
        "--resource-release-aware-eta",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--device-lookahead-reservation-mode",
        choices=("none", "soft", "hard"),
        default="none",
    )
    parser.add_argument("--device-reservation-grace-seconds", type=float, default=300.0)
    parser.add_argument("--resource-slack-forecast-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--iga180-budget-seconds", type=float, default=180.0)
    parser.add_argument("--iga1800-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--phase-attempts", type=int, default=3)
    parser.add_argument("--plane-warm-dir", type=Path)
    parser.add_argument("--resource-warm-dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--reuse-existing-iga180",
        action="store_true",
        help="Validate and reuse an existing compatible IGA-180 phase.",
    )
    args = parser.parse_args(argv)
    if not args.dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {args.dataset_dir}")
    if not args.python.is_file():
        parser.error(f"python executable does not exist: {args.python}")
    if args.workers < 1 or args.population < 2 or args.phase_attempts < 1:
        parser.error("workers>=1, population>=2, and phase-attempts>=1 are required")
    if args.iga180_budget_seconds <= 0:
        parser.error("IGA-180 budget must be positive")
    if args.iga1800_budget_seconds <= args.iga180_budget_seconds:
        parser.error("IGA-1800 budget must be greater than IGA-180 budget")
    if args.device_future_intent_horizon not in {0, 1, 2, 3}:
        parser.error("device-future-intent-horizon must be one of 0, 1, 2 or 3")
    if args.device_frontier_max_requests < 1:
        parser.error("device-frontier-max-requests must be positive")
    if args.device_request_capacity_per_plane < 0:
        parser.error("device-request-capacity-per-plane must be non-negative")
    if args.device_reservation_grace_seconds < 0:
        parser.error("device-reservation-grace-seconds must be non-negative")
    if args.resource_slack_forecast_seconds < 0:
        parser.error("resource-slack-forecast-seconds must be non-negative")
    if (args.plane_warm_dir is None) != (args.resource_warm_dir is None):
        parser.error("split warm start requires both plane and resource directories")
    for warm_dir in (args.plane_warm_dir, args.resource_warm_dir):
        if warm_dir is not None and not warm_dir.is_dir():
            parser.error(f"warm-start directory does not exist: {warm_dir}")
    return args


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
