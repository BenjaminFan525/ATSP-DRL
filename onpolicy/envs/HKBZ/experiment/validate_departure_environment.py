#!/usr/bin/env python
"""Stress-test explicit departure semantics on a fixed dataset panel."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.experiment.eval_common import (
    build_case_env_config,
    list_case_folders,
    load_case_metadata,
    write_json,
)
from onpolicy.envs.HKBZ.experiment.valid_fjsp_v2_comparison import (
    mask_aware_dispatch_policy,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--rule", choices=("FIFO", "SPT", "MWKR"), default="FIFO")
    parser.add_argument(
        "--resource-policy", choices=("heuristic", "drl"),
        default="heuristic",
    )
    parser.add_argument("--device-lookahead-dispatch", action="store_true")
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--max-consecutive-zero-dt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if (
        args.max_cases <= 0
        or args.workers <= 0
        or args.max_steps <= 0
        or args.max_consecutive_zero_dt <= 0
    ):
        parser.error(
            "max-cases, workers, max-steps and max-consecutive-zero-dt "
            "must be positive"
        )
    return args


def _has_interval_overlap(intervals):
    previous_end = None
    for start, end in sorted(intervals):
        if previous_end is not None and start < previous_end - 1e-9:
            return True
        previous_end = max(end, previous_end or end)
    return False


def _trajectory_invariants(env, plane_count):
    errors = []
    trajectory = list(env.trajectory_log)
    by_plane = {}
    for record in trajectory:
        by_plane.setdefault(record.get("plane_id"), []).append(record)

    expected_planes = {f"Plane_0_{pid}" for pid in range(plane_count)}
    if set(by_plane).difference({None}) != expected_planes:
        errors.append(
            "trajectory_plane_set_mismatch:"
            f"actual={sorted(set(by_plane).difference({None}))},"
            f"expected={sorted(expected_planes)}"
        )

    transporter_intervals = {}
    runway_intervals = {}
    progressive_departures = 0
    global_last_service_end = max(
        (
            float(record.get("end_time", 0.0))
            for record in trajectory
            if record.get("action_phase") == "service"
        ),
        default=0.0,
    )
    for plane_id in sorted(expected_planes):
        records = by_plane.get(plane_id, [])
        stages = [
            record for record in records
            if record.get("target_job_code") == env.TRANSFER_JOB_CODE
        ]
        starts = [
            record for record in records
            if record.get("target_job_code") == "ZY-S"
        ]
        finishes = [
            record for record in records
            if record.get("target_job_code") == "ZY-F"
        ]
        if len(stages) != 1 or len(starts) != 1 or len(finishes) != 1:
            errors.append(
                f"{plane_id}:stage/start/finish="
                f"{len(stages)}/{len(starts)}/{len(finishes)}"
            )
            continue

        start_record, finish_record = starts[0], finishes[0]
        ready_time = start_record.get("departure_ready_time")
        if ready_time is None or float(ready_time) > float(
            start_record.get("start_time", 0.0)
        ) + 1e-9:
            errors.append(f"{plane_id}:invalid_departure_ready_time")
        service_end = max(
            (
                float(record.get("end_time", 0.0))
                for record in records
                if record.get("action_phase") == "service"
            ),
            default=0.0,
        )
        if ready_time is not None and service_end > float(ready_time) + 1e-9:
            errors.append(f"{plane_id}:departure_ready_before_service_end")
        if float(start_record.get("start_time", 0.0)) < global_last_service_end:
            progressive_departures += 1
        if start_record.get("origin_site_code") in env.takeoff_site_code_list:
            errors.append(f"{plane_id}:departure_started_on_runway")
        runway = start_record.get("target_site_code")
        if runway not in env.takeoff_site_code_list:
            errors.append(f"{plane_id}:invalid_runway={runway}")
            continue
        if finish_record.get("target_site_code") != runway:
            errors.append(f"{plane_id}:finish_changed_runway")
        if float(start_record.get("waiting_time", 0.0)) > 1e-9:
            errors.append(f"{plane_id}:waited_after_ZY-S_dispatch")

        device_ids = list(start_record.get("device_ids", []))
        if len(device_ids) != 1:
            errors.append(f"{plane_id}:departure_r014_count={len(device_ids)}")
        else:
            tow_end = float(start_record.get("start_time", 0.0)) + float(
                start_record.get("trans_time", 0.0)
            )
            transporter_intervals.setdefault(device_ids[0], []).append((
                float(start_record.get("start_time", 0.0)), tow_end,
            ))
        runway_intervals.setdefault(runway, []).append((
            float(start_record.get("start_time", 0.0)),
            float(finish_record.get("end_time", 0.0)),
        ))

    for device_id, intervals in transporter_intervals.items():
        if _has_interval_overlap(intervals):
            errors.append(f"overlapping_R014_use:{device_id}")
    for runway, intervals in runway_intervals.items():
        if _has_interval_overlap(intervals):
            errors.append(f"overlapping_runway_use:{runway}")

    if env.departure_transporter_by_plane:
        errors.append("stale_departure_plane_reservations")
    if env.departure_plane_by_transporter:
        errors.append("stale_departure_device_reservations")
    reserved_devices = [
        device.code for device in env.device_list
        if getattr(device, "reserved_for_plane", None) is not None
    ]
    if reserved_devices:
        errors.append(f"stale_reserved_devices={reserved_devices}")
    occupied_runways = [
        code for code in env.takeoff_site_code_list
        if env.sites[code].is_occupied
    ]
    if occupied_runways:
        errors.append(f"occupied_runways_after_completion={occupied_runways}")
    if any(
        device.is_transporting
        for device in env.mobile_devices.get(env.TRANSPORTER_RESOURCE_TYPE, [])
    ):
        errors.append("R014_transporting_after_completion")
    if len(env.departure_log) != plane_count:
        errors.append(
            f"departure_log={len(env.departure_log)} expected={plane_count}"
        )
    return errors, progressive_departures


def validate_case(
    case_path, rule, seed, max_steps, resource_policy,
    device_lookahead_dispatch, domain_rand, max_consecutive_zero_dt,
):
    case = Path(case_path)
    config = build_case_env_config(
        str(case),
        resource_policy=resource_policy,
        max_device_num=80,
        seed=42,
    )
    config["use_domain_rand"] = bool(domain_rand)
    config["device_lookahead_dispatch"] = bool(
        device_lookahead_dispatch
    )
    env = AircraftScheduleEnv(config)
    rng = random.Random(int(seed))
    started = time.perf_counter()
    zero_dt_steps = 0
    consecutive_zero_dt = 0
    max_zero_dt_run = 0
    positive_dt_steps = 0
    try:
        _, done, info = env.reset()
        steps = 0
        while not np.all(done) and steps < int(max_steps):
            previous_time = float(env.total_time)
            actions = mask_aware_dispatch_policy(env, info, rule, rng)
            if resource_policy == "drl":
                device_actions = env.heuristic_device_actions()
                actions[env.n_plane_agents:] = device_actions[
                    env.n_plane_agents:
                ]
            _, _, done, info = env.step(actions)
            if float(env.total_time) > previous_time:
                positive_dt_steps += 1
                consecutive_zero_dt = 0
            else:
                zero_dt_steps += 1
                consecutive_zero_dt += 1
                max_zero_dt_run = max(
                    max_zero_dt_run, consecutive_zero_dt
                )
            steps += 1

        departure_records = [
            record for record in env.trajectory_log
            if record.get("action_phase") == "departure"
        ]
        stage_records = [
            record for record in env.trajectory_log
            if record.get("action_phase") == "post_service_relocation"
        ]
        zy_s_records = [
            record for record in departure_records
            if record.get("target_job_code") == "ZY-S"
        ]
        zy_f_records = [
            record for record in departure_records
            if record.get("target_job_code") == "ZY-F"
        ]
        plane_count = len(env.flights_data)
        invariant_errors = []
        if not np.all(done):
            invariant_errors.append("episode_timeout")
        if env.cycle_terminated:
            invariant_errors.append(f"cycle:{env.cycle_reason}")
        if len(env.departed_agent_ids) != plane_count:
            invariant_errors.append(
                f"departed={len(env.departed_agent_ids)} expected={plane_count}"
            )
        if len(zy_s_records) != plane_count or len(zy_f_records) != plane_count:
            invariant_errors.append(
                f"departure_jobs=ZY-S:{len(zy_s_records)},ZY-F:{len(zy_f_records)}"
            )
        if any(not record.get("departure_eligible") for record in departure_records):
            invariant_errors.append("departure_before_own_service_completion")
        if any(
            record.get("departure_mode") != "progressive_per_aircraft"
            for record in departure_records
        ):
            invariant_errors.append("unexpected_departure_mode")
        if any(
            record.get("target_site_code") not in env.takeoff_site_code_list
            for record in departure_records
        ):
            invariant_errors.append("departure_targeted_non_runway")
        if any(not record.get("device_ids") for record in zy_s_records):
            invariant_errors.append("departure_without_r014")
        if env.planes:
            invariant_errors.append(f"planes_remaining={len(env.planes)}")
        trajectory_errors, progressive_departures = _trajectory_invariants(
            env, plane_count
        )
        invariant_errors.extend(trajectory_errors)
        if max_zero_dt_run > int(max_consecutive_zero_dt):
            invariant_errors.append(
                f"consecutive_zero_dt={max_zero_dt_run} "
                f"limit={max_consecutive_zero_dt}"
            )

        return {
            "case": case.name,
            "completed": not invariant_errors,
            "invariant_errors": invariant_errors,
            "steps": int(steps),
            "makespan": float(env.total_time),
            "positive_dt_steps": int(positive_dt_steps),
            "zero_dt_steps": int(zero_dt_steps),
            "max_consecutive_zero_dt": int(max_zero_dt_run),
            "progressive_departure_count": int(progressive_departures),
            "departed_plane_count": int(len(env.departed_agent_ids)),
            "departure_action_count": int(len(departure_records)),
            "staging_action_count": int(len(stage_records)),
            "total_relocations": int(
                env.departed_total_relocations
                + sum(plane.total_relocations for plane in env.planes.values())
            ),
            "cycle_terminated": bool(env.cycle_terminated),
            "wall_seconds": float(time.perf_counter() - started),
            "environment_semantics_version": env.SEMANTICS_VERSION,
            "resource_policy": resource_policy,
            "device_lookahead_dispatch": bool(
                device_lookahead_dispatch
            ),
            "domain_rand": bool(domain_rand),
        }
    except Exception as error:
        return {
            "case": case.name,
            "completed": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "wall_seconds": float(time.perf_counter() - started),
        }
    finally:
        env.close()


def main():
    args = parse_args()
    dataset = Path(args.dataset_dir).resolve()
    cases = list_case_folders(str(dataset), args.max_cases)
    metadata = load_case_metadata(str(dataset))
    payload = {
        "status": "running",
        "dataset_dir": str(dataset),
        "rule": args.rule,
        "resource_policy": args.resource_policy,
        "device_lookahead_dispatch": bool(
            args.device_lookahead_dispatch
        ),
        "domain_rand": bool(args.domain_rand),
        "seed": int(args.seed),
        "case_count": len(cases),
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "cases": [],
    }
    write_json(args.output_json, payload)
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(cases)), mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                validate_case,
                str(dataset / case),
                args.rule,
                args.seed + index * 1009,
                args.max_steps,
                args.resource_policy,
                args.device_lookahead_dispatch,
                args.domain_rand,
                args.max_consecutive_zero_dt,
            ): case
            for index, case in enumerate(cases)
        }
        for future in as_completed(futures):
            record = future.result()
            record.update({
                "profile": metadata.get(record["case"], {}).get(
                    "profile", "unknown"
                ),
                "distribution": metadata.get(record["case"], {}).get(
                    "distribution", "unknown"
                ),
            })
            payload["cases"].append(record)
            payload["cases"].sort(key=lambda item: item["case"])
            write_json(args.output_json, payload)
            print(
                f"[DepartureValidation] {len(payload['cases'])}/{len(cases)} "
                f"{record['case']} completed={record.get('completed')} "
                f"steps={record.get('steps')}",
                flush=True,
            )

    failures = [record for record in payload["cases"] if not record["completed"]]
    payload["status"] = "completed" if not failures else "failed"
    payload["summary"] = {
        "completed_count": len(cases) - len(failures),
        "failed_count": len(failures),
        "max_steps": max((row.get("steps", 0) for row in payload["cases"]), default=0),
        "max_zero_dt_steps": max(
            (row.get("zero_dt_steps", 0) for row in payload["cases"]), default=0
        ),
        "max_consecutive_zero_dt": max(
            (
                row.get("max_consecutive_zero_dt", 0)
                for row in payload["cases"]
            ),
            default=0,
        ),
        "progressive_departure_cases": sum(
            row.get("progressive_departure_count", 0) > 0
            for row in payload["cases"]
        ),
        "mean_wall_seconds": float(np.mean([
            row["wall_seconds"] for row in payload["cases"]
        ])) if payload["cases"] else 0.0,
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
