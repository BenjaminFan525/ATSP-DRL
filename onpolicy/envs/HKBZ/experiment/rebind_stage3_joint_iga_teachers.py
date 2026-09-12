#!/usr/bin/env python3
"""Replay-bind Stage3 joint IGA teachers to the selected Stage2 MDP.

The existing full-joint chromosomes were searched with the earlier Wave-3
request-opportunity contract.  Stage3 starts from the Wave-4 B2 policy, whose
bounded frontier, release-aware ETA, and soft reservations change the live
masks.  This tool does not rewrite the expensive IGA output.  It replays every
chromosome through the exact target environment and atomically emits a strict
sidecar index only when all requested cases complete without a cycle.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels import (
    BACKENDS,
    SCHEMA_VERSION,
    TEACHER_METHOD,
    TEACHER_SCOPE,
    _build_layout,
    _case_sha256,
    _env_config,
    evaluate_joint,
)


TARGET_PLANNING_CONTRACT = {
    "device_lookahead_dispatch": True,
    "device_lookahead_safety_margin": 60.0,
    "device_deadline_aware_dispatch": True,
    "device_future_intent_horizon": 1,
    "device_future_intent_mode": "bounded_frontier",
    "device_frontier_max_requests": 2,
    "resource_release_aware_eta": True,
    "device_lookahead_reservation_mode": "soft",
    "device_reservation_grace_seconds": 300.0,
    "device_departure_lookahead": True,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _case_names(dataset_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in dataset_dir.iterdir()
        if path.is_dir() and path.name.startswith("case_")
    )


def _validate_teacher(teacher: Mapping, case_name: str, case_sha: str) -> None:
    if (
        int(teacher.get("schema_version", 0)) != SCHEMA_VERSION
        or teacher.get("teacher_scope") != TEACHER_SCOPE
        or teacher.get("teacher_method") != TEACHER_METHOD
        or teacher.get("resource_policy") != "drl"
        or teacher.get("backends")
        not in (["ordinary", "transporter"], dict(BACKENDS))
        or teacher.get("case") != case_name
        or teacher.get("case_sha256") != case_sha
        or not teacher.get("completion_verified", False)
        or not teacher.get("completed", False)
    ):
        raise ValueError(f"Unverified Stage3 joint teacher: {case_name}")


def _replay_one(payload: tuple[str, str, str, int, int, int]) -> tuple[str, dict]:
    (
        case_name,
        dataset_dir_text,
        teacher_dir_text,
        max_plane_agents,
        max_device_num,
        max_steps,
    ) = payload
    dataset_dir = Path(dataset_dir_text)
    teacher_dir = Path(teacher_dir_text)
    case_path = dataset_dir / case_name
    teacher_path = teacher_dir / f"{case_name}.json"
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Missing joint teacher: {teacher_path}")
    case_sha = _case_sha256(case_path)
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    _validate_teacher(teacher, case_name, case_sha)

    config = _env_config(
        case_path,
        max_plane_agents=max_plane_agents,
        max_device_num=max_device_num,
        max_steps=max_steps,
        lookahead_margin=TARGET_PLANNING_CONTRACT[
            "device_lookahead_safety_margin"
        ],
    )
    config.update(TARGET_PLANNING_CONTRACT)
    chromosome = np.asarray(
        teacher.get("search", {}).get("chromosome", []), dtype=np.float64
    ).reshape(-1)
    layout = _build_layout(config)
    recorded_n_var = int(teacher.get("search", {}).get("n_var", -1))
    if chromosome.size != layout.n_var or recorded_n_var != layout.n_var:
        raise ValueError(
            f"Joint chromosome layout mismatch for {case_name}: "
            f"values={chromosome.size} recorded={recorded_n_var} "
            f"target={layout.n_var}"
        )

    replay = evaluate_joint(config, chromosome, max_steps=max_steps)
    completion = replay.get("completion", {})
    if (
        not replay.get("completed", False)
        or completion.get("cycle_terminated", False)
        or replay.get("error")
    ):
        raise RuntimeError(
            f"Target-contract replay failed for {case_name}: {replay}"
        )
    return case_name, {
        "case_sha256": case_sha,
        "teacher_sha256": _sha256_file(teacher_path),
        "teacher_n_var": layout.n_var,
        "original_makespan": float(teacher["makespan"]),
        "replay_makespan": float(replay["makespan"]),
        "replay_steps": int(completion.get("steps", 0)),
        "replay_cycle_terminated": False,
        "replay_verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-plane-agents", type=int, default=24)
    parser.add_argument("--max-device-num", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    teacher_dir = args.teacher_dir.expanduser().resolve()
    output_index = args.output_index.expanduser().resolve()
    if not dataset_dir.is_dir() or not teacher_dir.is_dir():
        raise FileNotFoundError(
            f"dataset/teacher directory missing: {dataset_dir}, {teacher_dir}"
        )
    names = _case_names(dataset_dir)
    if args.limit > 0:
        names = names[: args.limit]
    if not names:
        raise ValueError("No Stage3 cases selected for replay binding.")
    workers = min(max(1, int(args.workers)), len(names))
    payloads = [
        (
            name,
            str(dataset_dir),
            str(teacher_dir),
            int(args.max_plane_agents),
            int(args.max_device_num),
            int(args.max_steps),
        )
        for name in names
    ]
    started = time.monotonic()
    entries = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_replay_one, item): item[0] for item in payloads}
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            case_name, entry = future.result()
            entries[case_name] = entry
            if completed == 1 or completed % 25 == 0 or completed == len(names):
                print(
                    f"[Stage3TeacherRebind] {completed}/{len(names)} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

    payload = {
        "schema_version": 1,
        "status": "completed",
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "environment_semantics_version": (
            "progressive-departure-r014-pipeline-v2"
        ),
        "dataset_dir": str(dataset_dir),
        "teacher_dir": str(teacher_dir),
        "planning_contract": TARGET_PLANNING_CONTRACT,
        "expected_case_count": len(names),
        "replay_verified_case_count": len(entries),
        "workers": workers,
        "elapsed_seconds": time.monotonic() - started,
        "created_unix_time": time.time(),
        "entries": dict(sorted(entries.items())),
    }
    _atomic_json(output_index, payload)
    print(f"[Stage3TeacherRebind] wrote {output_index}", flush=True)


if __name__ == "__main__":
    main()
