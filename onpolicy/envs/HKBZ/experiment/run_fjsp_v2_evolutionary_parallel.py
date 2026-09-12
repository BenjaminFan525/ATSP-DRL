#!/usr/bin/env python
"""Parallel case-level runner for the legacy IGA and NSGA-II baselines."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))

from onpolicy.envs.HKBZ.experiment.eval_common import (
    build_case_env_config,
    completion_details,
    list_case_folders,
    load_case_metadata,
    write_json,
)
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv


METHOD_LABELS = {"iga": "IGA", "nsga2": "NSGA-II"}
IGA_TEACHER_SCOPE = "stage1_plane_policy"
IGA_TEACHER_RESOURCE_POLICY = "heuristic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_test_dir",
        default=(
            "/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/dataset/"
            "fjsp_v2_t480_v60_test60/test"
        ),
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHOD_LABELS), default=list(METHOD_LABELS)
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--time_budget", type=float, default=1800.0)
    parser.add_argument("--iga_pop_size", type=int, default=20)
    parser.add_argument("--iga_generations", type=int, default=20)
    parser.add_argument(
        "--iga_max_attempts",
        type=int,
        default=3,
        help="maximum deterministic seed attempts per IGA teacher case",
    )
    parser.add_argument("--iga_teacher_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse only completed, verified cases from an interrupted output; "
            "all failed or incomplete cases are submitted again"
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.time_budget <= 0.0:
        raise ValueError("--time_budget must be positive")
    if args.iga_pop_size <= 1 or args.iga_generations <= 0:
        raise ValueError("IGA population/generations are invalid")
    if args.iga_max_attempts <= 0:
        raise ValueError("--iga_max_attempts must be positive")
    if args.iga_teacher_dir and "iga" not in args.methods:
        raise ValueError("--iga_teacher_dir requires --methods iga")
    return args


def verify_iga_teacher(
    case_path,
    chromosome,
    n_agents,
    n_jobs,
    n_sites,
    policy,
    max_steps=2000,
):
    """Replay an exported chromosome through authoritative live masks."""

    from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv

    chromosome = np.asarray(chromosome, dtype=np.float64).reshape(-1)
    split = int(n_agents) * int(n_jobs)
    expected = split + int(n_agents) * int(n_sites)
    if chromosome.size != expected:
        raise ValueError(
            f"IGA chromosome has {chromosome.size} genes, expected {expected}."
        )
    job_priorities = chromosome[:split].reshape(int(n_agents), int(n_jobs))
    site_priorities = chromosome[split:].reshape(int(n_agents), int(n_sites))
    config = build_case_env_config(
        case_path,
        max_plane_agents=int(n_agents),
        max_device_num=80,
        resource_policy="heuristic",
        seed=42,
    )
    env = AircraftScheduleEnv(config)
    env.use_domain_rand = False
    step_count = 0
    try:
        _, done, info = env.reset()
        while not np.all(done) and step_count < int(max_steps):
            actions = policy(env, info, job_priorities, site_priorities)
            _, _, done, info = env.step(actions)
            step_count += 1
        details = completion_details(env, step_count, int(max_steps))
        details["timeout"] = bool(not np.all(done))
        details["completed"] = bool(
            details["completed"]
            and not details["cycle_terminated"]
            and not details["timeout"]
        )
        details["makespan"] = float(env.total_time)
        return details
    finally:
        env.close()


def evaluate_case(
    method, case_name, case_path, metadata, time_budget, iga_pop_size,
    iga_generations, seed, iga_max_attempts, iga_teacher_dir,
):
    # Apply the compatibility policy inside every worker so both fork and spawn
    # process starts are correct.
    from onpolicy.envs.HKBZ.experiment import valid_iga, valid_nsga
    from onpolicy.envs.HKBZ.experiment.valid_fjsp_v2_comparison import (
        mask_aware_ga_policy,
    )

    valid_iga.GA_Policy = mask_aware_ga_policy
    valid_nsga.GA_Policy = mask_aware_ga_policy
    record = {
        "case": case_name,
        "case_id": metadata.get("case_id", case_name),
        "seed": metadata.get("seed"),
        "profile": metadata.get("profile", "unknown"),
        "distribution": metadata.get("distribution", "unknown"),
        "case_sha256": metadata.get("case_sha256"),
    }
    wall_started = time.perf_counter()
    try:
        attempts = []
        completed = False
        completion_verified = False
        cycle_terminated = False
        timeout = False
        if method == "iga":
            export_teacher = bool(iga_teacher_dir)
            selected = None
            verification = None
            total_cpu_seconds = 0.0
            case_number = int(case_name.rsplit("_", 1)[-1])
            for attempt in range(int(iga_max_attempts)):
                attempt_seed = int(seed) + case_number * 1009 + attempt * 1000003
                candidate = valid_iga.test_iga_on_case(
                    case_path,
                    max_time_seconds=time_budget,
                    pop_size=iga_pop_size,
                    n_gen=iga_generations,
                    seed=attempt_seed,
                    return_solution=export_teacher,
                )
                if export_teacher:
                    candidate_cmax = candidate.get("makespan")
                    candidate_cpu = candidate.get("cpu_seconds")
                    chromosome = candidate.get("chromosome")
                    optimizer_valid = bool(
                        chromosome is not None
                        and candidate_cmax is not None
                        and np.isfinite(candidate_cmax)
                        and float(candidate_cmax) < 100000.0
                    )
                    verification = None
                    if optimizer_valid:
                        verification = verify_iga_teacher(
                            case_path,
                            chromosome,
                            candidate["n_agents"],
                            candidate["n_jobs"],
                            candidate["n_sites"],
                            mask_aware_ga_policy,
                        )
                    verified = bool(
                        optimizer_valid
                        and verification is not None
                        and verification["completed"]
                    )
                else:
                    candidate_cmax, candidate_cpu = candidate
                    verified = bool(
                        candidate_cmax is not None
                        and np.isfinite(candidate_cmax)
                        and float(candidate_cmax) < 100000.0
                    )
                if candidate_cpu is not None and np.isfinite(candidate_cpu):
                    total_cpu_seconds += float(candidate_cpu)
                attempts.append({
                    "attempt": attempt + 1,
                    "seed": attempt_seed,
                    "optimizer_makespan": (
                        float(candidate_cmax)
                        if candidate_cmax is not None else None
                    ),
                    "completion_verified": bool(verified),
                    "verified_makespan": (
                        float(verification["makespan"])
                        if verification is not None else None
                    ),
                })
                if verified:
                    selected = candidate
                    break

            cpu_seconds = total_cpu_seconds
            if selected is None:
                cmax = 100000.0
                cycle_terminated = bool(
                    verification and verification.get("cycle_terminated")
                )
                timeout = bool(
                    verification is None or verification.get("timeout")
                )
            elif export_teacher:
                chromosome = np.asarray(
                    selected["chromosome"], dtype=np.float64
                )
                split = selected["n_agents"] * selected["n_jobs"]
                cmax = float(verification["makespan"])
                completed = True
                completion_verified = True
                teacher = {
                    "schema_version": 3,
                    "teacher_scope": IGA_TEACHER_SCOPE,
                    "resource_policy": IGA_TEACHER_RESOURCE_POLICY,
                    "environment_semantics_version": verification.get(
                        "environment_semantics_version"
                    ),
                    "case": case_name,
                    "case_id": record["case_id"],
                    "seed": record["seed"],
                    "profile": record["profile"],
                    "distribution": record["distribution"],
                    "case_sha256": record["case_sha256"],
                    "makespan": cmax,
                    "completion_verified": True,
                    "verification": verification,
                    "job_priorities": chromosome[:split].reshape(
                        selected["n_agents"], selected["n_jobs"]
                    ).tolist(),
                    "site_priorities": chromosome[split:].reshape(
                        selected["n_agents"], selected["n_sites"]
                    ).tolist(),
                    "iga": {
                        key: selected[key]
                        for key in (
                            "cpu_seconds", "wall_seconds", "pop_size", "n_gen",
                            "seed", "time_budget_seconds", "incumbent_preserved"
                        )
                    },
                    "attempts": attempts,
                }
                write_json(
                    str(Path(iga_teacher_dir) / f"{case_name}.json"),
                    teacher,
                )
            else:
                cmax = float(selected[0])
                completed = True
        else:
            cmax, cpu_seconds = valid_nsga.test_nsga2_on_case(case_path)
            cmax = (
                float(np.asarray(cmax).reshape(-1)[0])
                if cmax is not None else math.nan
            )
            completed = bool(np.isfinite(cmax) and cmax < 100000.0)

        record.update({
            "makespan": float(cmax),
            "cpu_seconds": float(cpu_seconds),
            "wall_seconds": time.perf_counter() - wall_started,
            "completed": completed,
            "completion_verified": completion_verified,
            "cycle_terminated": cycle_terminated,
            "timeout": timeout,
            "attempts": attempts,
        })
    except Exception as error:
        record.update({
            "completed": False,
            "completion_verified": False,
            "wall_seconds": time.perf_counter() - wall_started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
    return method, record


def summarize(records):
    completed = [record for record in records if record.get("completed")]
    makespans = np.asarray([record["makespan"] for record in completed], dtype=float)
    wall = np.asarray([record["wall_seconds"] for record in completed], dtype=float)
    return {
        "case_count": len(records),
        "completed_count": len(completed),
        "verified_count": sum(
            bool(record.get("completion_verified")) for record in records
        ),
        "completion_rate": len(completed) / max(1, len(records)),
        "error_count": sum(bool(record.get("error")) for record in records),
        "mean_makespan": float(makespans.mean()) if len(makespans) else math.nan,
        "std_makespan": float(makespans.std()) if len(makespans) else math.nan,
        "median_makespan": float(np.median(makespans)) if len(makespans) else math.nan,
        "mean_wall_seconds": float(wall.mean()) if len(wall) else math.nan,
        "max_wall_seconds": float(wall.max()) if len(wall) else math.nan,
    }


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _teacher_is_reusable(teacher_path, case_name, case_metadata):
    """Reject partial files and labels generated under older semantics."""

    try:
        teacher = _load_json(teacher_path)
    except (OSError, ValueError, TypeError):
        return False
    if int(teacher.get("schema_version", 0)) < 3:
        return False
    if teacher.get("teacher_scope") != IGA_TEACHER_SCOPE:
        return False
    if teacher.get("resource_policy") != IGA_TEACHER_RESOURCE_POLICY:
        return False
    if (
        teacher.get("environment_semantics_version")
        != AircraftScheduleEnv.SEMANTICS_VERSION
    ):
        return False
    if teacher.get("case") != case_name:
        return False
    if not bool(teacher.get("completion_verified")):
        return False
    expected_hash = case_metadata.get("case_sha256")
    if expected_hash and teacher.get("case_sha256") != expected_hash:
        return False
    job_priorities = np.asarray(teacher.get("job_priorities", []))
    site_priorities = np.asarray(teacher.get("site_priorities", []))
    return bool(
        job_priorities.ndim == 2
        and site_priorities.ndim == 2
        and job_priorities.shape[0] == site_priorities.shape[0]
        and job_priorities.shape[1] == 20
    )


def _validate_resume_payload(payload, args, dataset_dir, cases):
    if (
        payload.get("environment_semantics_version")
        != AircraftScheduleEnv.SEMANTICS_VERSION
    ):
        raise RuntimeError(
            "Refusing to resume labels from a different environment semantics."
        )
    if payload.get("teacher_scope") != IGA_TEACHER_SCOPE:
        raise RuntimeError("Resume payload is not a Stage-1 plane-policy label set.")
    if payload.get("resource_policy") != IGA_TEACHER_RESOURCE_POLICY:
        raise RuntimeError("Stage-1 IGA labels require heuristic resources.")
    if str(Path(payload.get("dataset_test_dir", "")).resolve()) != dataset_dir:
        raise RuntimeError("Resume dataset directory does not match.")
    if int(payload.get("case_count", -1)) != len(cases):
        raise RuntimeError("Resume case count does not match.")
    requested = [METHOD_LABELS[method] for method in args.methods]
    if payload.get("methods_requested") != requested:
        raise RuntimeError("Resume method list does not match.")
    budget = payload.get("legacy_budget", {})
    expected_budget = {
        "population": int(args.iga_pop_size),
        "generations": int(args.iga_generations),
        "time_seconds": float(args.time_budget),
        "max_attempts": int(args.iga_max_attempts),
    }
    if budget != expected_budget:
        raise RuntimeError(
            f"Resume IGA budget does not match: {budget} != {expected_budget}."
        )
    if int(payload.get("seed", -1)) != int(args.seed):
        raise RuntimeError("Resume seed does not match.")
    stored_teacher_dir = str(payload.get("iga_teacher_dir", ""))
    if stored_teacher_dir:
        stored_teacher_dir = str(Path(stored_teacher_dir).resolve())
    if stored_teacher_dir != str(args.iga_teacher_dir):
        raise RuntimeError("Resume teacher directory does not match.")


def _record_is_reusable(method, record, args, metadata):
    if not bool(record.get("completed")):
        return False
    if method != "iga" or not args.iga_teacher_dir:
        return True
    if not bool(record.get("completion_verified")):
        return False
    case_name = str(record.get("case", ""))
    teacher_path = Path(args.iga_teacher_dir) / f"{case_name}.json"
    return _teacher_is_reusable(
        teacher_path, case_name, metadata.get(case_name, {})
    )


def main():
    args = parse_args()
    dataset_dir = str(Path(args.dataset_test_dir).resolve())
    args.output_json = str(Path(args.output_json).resolve())
    if args.iga_teacher_dir:
        args.iga_teacher_dir = str(Path(args.iga_teacher_dir).resolve())
    cases = list_case_folders(dataset_dir, args.max_cases)
    metadata = load_case_metadata(dataset_dir)
    if args.iga_teacher_dir:
        missing_metadata = [case for case in cases if case not in metadata]
        invalid_metadata = [
            case for case in cases
            if metadata.get(case, {}).get("profile") in (None, "", "unknown")
            or metadata.get(case, {}).get("distribution") in (None, "", "unknown")
        ]
        if missing_metadata or invalid_metadata:
            raise RuntimeError(
                "IGA teacher metadata is incomplete for dataset split: "
                f"missing={missing_metadata[:10]}, invalid={invalid_metadata[:10]}."
            )

    output_path = Path(args.output_json)
    teacher_path = Path(args.iga_teacher_dir) if args.iga_teacher_dir else None
    if teacher_path is not None:
        teacher_path.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists; pass --resume to reuse it: {output_path}"
        )
    if (
        teacher_path is not None
        and not args.resume
        and any(teacher_path.glob("case_*.json"))
    ):
        raise FileExistsError(
            "Teacher directory already contains labels; use a fresh directory "
            f"or pass --resume: {teacher_path}"
        )
    if (
        teacher_path is not None
        and args.resume
        and not output_path.exists()
        and any(teacher_path.glob("case_*.json"))
    ):
        raise FileNotFoundError(
            "Cannot safely resume a non-empty teacher directory without its "
            f"progress output: {output_path}"
        )

    if output_path.exists():
        payload = _load_json(output_path)
        _validate_resume_payload(payload, args, dataset_dir, cases)
        payload["status"] = "running"
        payload["workers"] = int(args.workers)
        payload["last_resumed_unix_time"] = time.time()
        for method in args.methods:
            label = METHOD_LABELS[method]
            previous = payload.get("methods", {}).get(label, {}).get(
                "cases", []
            )
            by_case = {
                record.get("case"): record
                for record in previous
                if record.get("case") in cases
                and _record_is_reusable(method, record, args, metadata)
            }
            records = [by_case[case] for case in cases if case in by_case]
            payload["methods"][label] = {
                "status": (
                    "completed" if len(records) == len(cases) else "running"
                ),
                "cases": records,
                "summary": summarize(records),
            }
    else:
        payload = {
            "status": "running",
            "teacher_scope": IGA_TEACHER_SCOPE,
            "resource_policy": IGA_TEACHER_RESOURCE_POLICY,
            "environment_semantics_version": (
                AircraftScheduleEnv.SEMANTICS_VERSION
            ),
            "created_unix_time": time.time(),
            "dataset_test_dir": dataset_dir,
            "workers": int(args.workers),
            "seed": int(args.seed),
            "iga_teacher_dir": args.iga_teacher_dir,
            "legacy_budget": {
                "population": int(args.iga_pop_size),
                "generations": int(args.iga_generations),
                "time_seconds": float(args.time_budget),
                "max_attempts": int(args.iga_max_attempts),
            },
            "methods_requested": [
                METHOD_LABELS[method] for method in args.methods
            ],
            "case_count": len(cases),
            "methods": {
                METHOD_LABELS[method]: {"status": "pending", "cases": []}
                for method in args.methods
            },
        }
    write_json(args.output_json, payload)
    reusable_cases = {
        method: {
            record["case"]
            for record in payload["methods"][METHOD_LABELS[method]]["cases"]
        }
        for method in args.methods
    }
    tasks = [
        (
            method,
            case,
            str(Path(dataset_dir) / case),
            metadata.get(case, {}),
            args.time_budget,
            args.iga_pop_size,
            args.iga_generations,
            args.seed,
            args.iga_max_attempts,
            args.iga_teacher_dir,
        )
        for method in args.methods
        for case in cases
        if case not in reusable_cases[method]
    ]

    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        future_map = {
            executor.submit(evaluate_case, *task): task[:2] for task in tasks
        }
        for future in as_completed(future_map):
            method, case_name = future_map[future]
            label = METHOD_LABELS[method]
            try:
                _, record = future.result()
            except Exception as error:
                record = {
                    "case": case_name,
                    "completed": False,
                    "error": f"worker {type(error).__name__}: {error}",
                }
            records = payload["methods"][label]["cases"]
            records.append(record)
            records.sort(key=lambda item: item["case"])
            payload["methods"][label]["status"] = (
                "completed" if len(records) == len(cases) else "running"
            )
            payload["methods"][label]["summary"] = summarize(records)
            write_json(args.output_json, payload)
            print(
                f"[{label}] {len(records)}/{len(cases)} {case_name} "
                f"Cmax={record.get('makespan')} completed={record.get('completed')}",
                flush=True,
            )

    fully_verified = True
    for method in args.methods:
        label = METHOD_LABELS[method]
        records = payload["methods"][label]["cases"]
        method_complete = bool(
            len(records) == len(cases)
            and all(record.get("completed") for record in records)
            and (
                method != "iga"
                or not args.iga_teacher_dir
                or all(record.get("completion_verified") for record in records)
            )
        )
        payload["methods"][label]["status"] = (
            "completed" if method_complete else "failed"
        )
        payload["methods"][label]["summary"] = summarize(records)
        fully_verified = fully_verified and method_complete
    payload["status"] = "completed" if fully_verified else "failed"
    payload["completed_unix_time"] = time.time()
    write_json(args.output_json, payload)
    print(json.dumps(
        {
            label: method_data.get("summary", {})
            for label, method_data in payload["methods"].items()
        },
        indent=2,
        ensure_ascii=False,
    ))
    if not fully_verified:
        raise RuntimeError(
            "One or more cases failed completion/teacher verification; rerun "
            "the same command with --resume to retry only those cases."
        )


if __name__ == "__main__":
    main()
