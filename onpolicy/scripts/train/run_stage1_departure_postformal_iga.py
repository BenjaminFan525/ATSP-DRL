#!/usr/bin/env python3
"""Run the new-semantics test60 IGA baseline after Formal validation.

The controller is intentionally independent from the short-lived shell
launcher.  It waits for every preregistered Formal trial to publish both a
completed ``run_status.json`` and its final epoch validation artifact.  Only
then does it release the shared evaluators and submit the sequential
IGA-180 -> verification -> IGA-1800 labeling service.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_SEMANTICS = "progressive-departure-r014-pipeline-v2"
FAILED_TRIAL_STATES = {"failed", "interrupted"}
ACTIVE_UNIT_STATES = {"active", "activating", "deactivating", "reloading"}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def started_unix_time(status: dict[str, Any]) -> float:
    value = str(status.get("started_at", ""))
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def discover_current_status(
    results_root: Path,
    experiment_name: str,
    not_before_unix_time: float,
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    experiment_dir = results_root / experiment_name
    for path in experiment_dir.glob("run*/run_status.json"):
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        started = started_unix_time(payload)
        if started + 300.0 < not_before_unix_time:
            continue
        candidates.append((started, path, payload))
    if not candidates:
        return None
    _, path, payload = max(
        candidates,
        key=lambda item: (item[0], item[1].stat().st_mtime),
    )
    return path, payload


def validate_final_evaluation(
    status_path: Path,
    status: dict[str, Any],
    final_epoch: int,
    expected_cases: int,
    expected_validation_dir: Path,
) -> dict[str, Any]:
    if int(status.get("total_epochs", -1)) != final_epoch:
        raise RuntimeError(
            f"{status_path}: total_epochs={status.get('total_epochs')!r}, "
            f"expected {final_epoch}."
        )
    evaluation_path = status_path.parent / "evaluations" / f"epoch_{final_epoch}.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(str(evaluation_path))
    evaluation = read_json(evaluation_path)
    if evaluation.get("evaluation_label") != f"epoch_{final_epoch}":
        raise RuntimeError(f"Final evaluation label mismatch: {evaluation_path}")
    actual_dir = Path(str(evaluation.get("eval_dataset_dir", ""))).resolve()
    if actual_dir != expected_validation_dir.resolve():
        raise RuntimeError(
            f"Final evaluation dataset mismatch: {actual_dir} != "
            f"{expected_validation_dir.resolve()}"
        )
    cases = evaluation.get("cases")
    summary = evaluation.get("summary")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise RuntimeError(
            f"Final evaluation does not contain {expected_cases} cases: "
            f"{evaluation_path}"
        )
    if not isinstance(summary, dict):
        raise RuntimeError(f"Final evaluation summary is missing: {evaluation_path}")
    if int(summary.get("eval_case_count", -1)) != expected_cases:
        raise RuntimeError(f"Final evaluation case count mismatch: {evaluation_path}")
    if int(summary.get("eval_completed_count", -1)) != expected_cases:
        raise RuntimeError(f"Final evaluation is incomplete: {evaluation_path}")
    if float(summary.get("eval_completion_rate", -1.0)) != 1.0:
        raise RuntimeError(f"Final evaluation completion rate is not 1: {evaluation_path}")
    return {
        "run_status": str(status_path.resolve()),
        "final_evaluation": str(evaluation_path.resolve()),
        "final_eval_makespan": float(summary["eval_makespan"]),
        "case_count": len(cases),
    }


def inspect_trials(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    evidence: dict[str, Any] = {}
    all_complete = True
    for experiment_name in args.experiment_name:
        discovered = discover_current_status(
            args.results_root,
            experiment_name,
            args.not_before_unix_time,
        )
        if discovered is None:
            evidence[experiment_name] = {"status": "not_started"}
            all_complete = False
            continue
        status_path, status = discovered
        state = str(status.get("status", "unknown")).lower()
        if state in FAILED_TRIAL_STATES:
            raise RuntimeError(
                f"Formal trial {experiment_name} ended as {state}: "
                f"{status.get('reason', '')}"
            )
        trial = {
            "status": state,
            "run_status": str(status_path.resolve()),
            "epoch": status.get("epoch"),
            "shard": status.get("shard"),
            "heartbeat_timestamp": status.get("heartbeat_timestamp"),
        }
        if state != "completed":
            evidence[experiment_name] = trial
            all_complete = False
            continue
        try:
            trial.update(validate_final_evaluation(
                status_path,
                status,
                args.formal_epochs,
                args.expected_validation_cases,
                args.validation_dataset_dir,
            ))
        except FileNotFoundError:
            trial["status"] = "waiting_for_final_evaluation"
            evidence[experiment_name] = trial
            all_complete = False
            continue
        trial["status"] = "completed_and_validated"
        evidence[experiment_name] = trial
    return all_complete, evidence


def unit_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def wait_units_inactive(units: list[str], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        states = {unit: unit_state(unit) for unit in units}
        active = {
            unit: state for unit, state in states.items()
            if state in ACTIVE_UNIT_STATES
        }
        if not active:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Units did not settle: {active}")
        time.sleep(2.0)


def stop_evaluators(units: list[str]) -> None:
    if not units:
        return
    result = subprocess.run(
        ["systemctl", "--user", "stop", *units],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to stop shared evaluators: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    wait_units_inactive(units, timeout_seconds=300.0)


def validate_test_dataset(path: Path, expected_cases: int) -> None:
    cases = sorted(item for item in path.glob("case_*") if item.is_dir())
    if len(cases) != expected_cases:
        raise RuntimeError(
            f"test60 dataset has {len(cases)} cases, expected {expected_cases}: {path}"
        )
    required = {
        "flights.json", "job.json", "mobile_resources.json",
        "fixed_resources.json", "sites.json",
    }
    for case in cases:
        missing = sorted(name for name in required if not (case / name).is_file())
        if missing:
            raise RuntimeError(f"{case} is missing required files: {missing}")


def validate_iga_output(path: Path, dataset: Path, expected_cases: int) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("status") != "completed":
        raise RuntimeError(f"IGA output is not completed: {path}")
    if payload.get("environment_semantics_version") != EXPECTED_SEMANTICS:
        raise RuntimeError(f"IGA output uses the wrong semantics: {path}")
    if Path(str(payload.get("dataset_test_dir", ""))).resolve() != dataset.resolve():
        raise RuntimeError(f"IGA output uses the wrong dataset: {path}")
    if int(payload.get("case_count", -1)) != expected_cases:
        raise RuntimeError(f"IGA output case count mismatch: {path}")
    method = payload.get("methods", {}).get("IGA", {})
    summary = method.get("summary", {}) if isinstance(method, dict) else {}
    cases = method.get("cases", []) if isinstance(method, dict) else []
    if method.get("status") != "completed":
        raise RuntimeError(f"IGA method is not completed: {path}")
    if len(cases) != expected_cases:
        raise RuntimeError(f"IGA per-case record count mismatch: {path}")
    if int(summary.get("completed_count", -1)) != expected_cases:
        raise RuntimeError(f"IGA completed count mismatch: {path}")
    if int(summary.get("verified_count", -1)) != expected_cases:
        raise RuntimeError(f"IGA replay verification count mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "case_count": expected_cases,
        "verified_count": expected_cases,
        "environment_semantics_version": EXPECTED_SEMANTICS,
    }


def submit_iga(args: argparse.Namespace) -> None:
    environment = os.environ.copy()
    environment.update({
        "PYTHON": str(args.python.resolve()),
        "DATASET": str(args.test_dataset_dir.resolve()),
        "DATASET_TAG": "test60",
        "LABEL_VERSION": args.iga_label_version,
        "OUTPUT_ROOT": str(args.iga_output_root.resolve()),
        "TEACHER_ROOT": str(args.iga_teacher_root.resolve()),
        "WORKERS": str(args.iga_workers),
        "POPULATION": "20",
        "GENERATIONS": "20",
        "MAX_ATTEMPTS": "5",
        "IGA_SEED": "1",
        "RUNNER_PASSES": "3",
        "RETRY_DELAY_SECONDS": "30",
        "CPU_SET": args.iga_cpu_set,
        "UNIT_SEQUENCE": args.iga_unit,
    })
    subprocess.run(
        ["/bin/bash", str(args.iga_launcher.resolve())],
        check=True,
        env=environment,
    )


def wait_for_iga(args: argparse.Namespace, base_status: dict[str, Any]) -> None:
    outputs = {
        "iga180": args.iga_output_root / "iga180_test60.json",
        "iga1800": args.iga_output_root / "iga1800_test60.json",
    }
    deadline = time.monotonic() + args.iga_timeout_seconds
    while True:
        completed: dict[str, Any] = {}
        for label, path in outputs.items():
            if not path.is_file():
                continue
            try:
                completed[label] = validate_iga_output(
                    path,
                    args.test_dataset_dir,
                    args.expected_test_cases,
                )
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                continue
        if len(completed) == len(outputs):
            atomic_json(args.status_path, {
                **base_status,
                "status": "completed",
                "stage": "test60_iga_completed",
                "completed_unix_time": time.time(),
                "outputs": completed,
            })
            return
        state = unit_state(args.iga_unit)
        atomic_json(args.status_path, {
            **base_status,
            "status": "running",
            "stage": "test60_iga_running",
            "heartbeat_unix_time": time.time(),
            "iga_unit_state": state,
            "verified_outputs": completed,
        })
        if state not in ACTIVE_UNIT_STATES:
            raise RuntimeError(
                f"IGA unit {args.iga_unit} became {state!r} before both "
                "verified outputs were available."
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {args.iga_unit}")
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-contract", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--experiment-name", action="append", required=True)
    parser.add_argument("--trainer-unit", action="append", required=True)
    parser.add_argument("--evaluator-unit", action="append", required=True)
    parser.add_argument("--formal-epochs", type=int, required=True)
    parser.add_argument("--validation-dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-validation-cases", type=int, default=60)
    parser.add_argument("--test-dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-test-cases", type=int, default=60)
    parser.add_argument("--iga-launcher", type=Path, required=True)
    parser.add_argument("--iga-label-version", required=True)
    parser.add_argument("--iga-output-root", type=Path, required=True)
    parser.add_argument("--iga-teacher-root", type=Path, required=True)
    parser.add_argument("--iga-unit", required=True)
    parser.add_argument("--iga-cpu-set", default="0-143")
    parser.add_argument("--iga-workers", type=int, default=144)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--iga-timeout-seconds", type=float, default=172800.0)
    args = parser.parse_args()
    if args.formal_epochs < 1:
        parser.error("--formal-epochs must be positive")
    if args.expected_validation_cases < 1 or args.expected_test_cases < 1:
        parser.error("expected case counts must be positive")
    if args.poll_seconds <= 0.0 or args.iga_timeout_seconds <= 0.0:
        parser.error("poll and timeout values must be positive")
    if len(args.experiment_name) != len(args.trainer_unit):
        parser.error("experiment and trainer unit counts must match")
    return args


def run(args: argparse.Namespace) -> int:
    contract = read_json(args.launch_contract)
    args.not_before_unix_time = float(contract.get("created_unix_time", 0.0))
    if args.not_before_unix_time <= 0.0:
        raise RuntimeError("Launch contract lacks created_unix_time.")
    validate_test_dataset(args.test_dataset_dir, args.expected_test_cases)
    if not args.validation_dataset_dir.is_dir():
        raise FileNotFoundError(args.validation_dataset_dir)
    for required in (args.iga_launcher, args.python):
        if not required.is_file():
            raise FileNotFoundError(required)

    prior = read_json(args.status_path) if args.status_path.is_file() else {}
    if prior.get("status") == "completed":
        print("[PostFormal] Verified test60 IGA outputs already completed.", flush=True)
        return 0

    base_status = {
        "schema_version": 1,
        "launch_contract": str(args.launch_contract.resolve()),
        "formal_epochs": args.formal_epochs,
        "experiments": list(args.experiment_name),
        "test_dataset_dir": str(args.test_dataset_dir.resolve()),
        "expected_test_cases": args.expected_test_cases,
        "environment_semantics_version": EXPECTED_SEMANTICS,
        "iga_label_version": args.iga_label_version,
        "iga_unit": args.iga_unit,
        "iga_budgets_seconds": [180, 1800],
    }

    if prior.get("stage") == "test60_iga_running":
        wait_for_iga(args, base_status)
        return 0

    while True:
        all_complete, evidence = inspect_trials(args)
        atomic_json(args.status_path, {
            **base_status,
            "status": "waiting",
            "stage": "waiting_for_all_final_validations",
            "heartbeat_unix_time": time.time(),
            "trial_evidence": evidence,
        })
        if all_complete:
            break
        time.sleep(args.poll_seconds)

    atomic_json(args.status_path, {
        **base_status,
        "status": "running",
        "stage": "releasing_formal_resources",
        "heartbeat_unix_time": time.time(),
        "trial_evidence": evidence,
    })
    wait_units_inactive(args.trainer_unit, timeout_seconds=900.0)
    stop_evaluators(args.evaluator_unit)

    atomic_json(args.status_path, {
        **base_status,
        "status": "running",
        "stage": "submitting_test60_iga",
        "heartbeat_unix_time": time.time(),
        "trial_evidence": evidence,
    })
    submit_iga(args)
    wait_for_iga(args, {**base_status, "trial_evidence": evidence})
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except BaseException as error:
        try:
            atomic_json(args.status_path, {
                "schema_version": 1,
                "status": "failed",
                "stage": "postformal_iga_failed",
                "failed_unix_time": time.time(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
