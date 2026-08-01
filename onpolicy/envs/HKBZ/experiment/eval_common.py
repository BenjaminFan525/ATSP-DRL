"""Shared helpers for HKBZ test-set evaluation scripts.

The environment owns the authoritative operation, site, and joint
operation-site masks.  Baseline policies must consume those masks just like
the learned policy; reconstructing feasibility only from object state is not
enough after cycle and irreversible-progress guards are applied.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def list_case_folders(dataset_dir: str, max_cases: int = 0) -> List[str]:
    cases = sorted(
        entry.name
        for entry in Path(dataset_dir).iterdir()
        if entry.is_dir() and entry.name.startswith("case_")
    )
    if max_cases > 0:
        cases = cases[:max_cases]
    return cases


def load_case_metadata(dataset_dir: str) -> Dict[str, dict]:
    """Return manifest metadata keyed by case directory when available."""

    dataset_path = Path(dataset_dir).resolve()
    manifest_path = dataset_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    records: Iterable[dict] = ()
    for key in ("cases", "instances", "records"):
        value = manifest.get(key)
        if isinstance(value, list):
            records = value
            break
    if not records:
        splits = manifest.get("splits", {})
        value = None
        if isinstance(splits, dict):
            split_name = dataset_path.name.lower()
            split_key = next(
                (key for key in splits if str(key).lower() == split_name),
                None,
            )
            if split_key is not None:
                value = splits.get(split_key)
            elif len(splits) == 1:
                # Legacy one-split manifests may not name their sole split
                # after the dataset directory.  This fallback is safe only
                # when there is no ambiguity.
                value = next(iter(splits.values()))
        if isinstance(value, dict):
            value = value.get("cases")
        if isinstance(value, list):
            records = value

    result = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        case_dir = record.get("case_dir") or record.get("directory")
        if not case_dir and record.get("case_id"):
            suffix = str(record["case_id"]).rsplit("_", 1)[-1]
            case_dir = f"case_{suffix}"
        if case_dir:
            result[str(case_dir)] = record
    return result


def build_case_env_config(
    case_path: str,
    *,
    max_plane_agents: Optional[int] = None,
    max_device_num: int = 80,
    resource_policy: str = "heuristic",
    seed: int = 42,
    global_feature_mode: str = "none",
) -> dict:
    case = Path(case_path)
    with (case / "flights.json").open("r", encoding="utf-8") as handle:
        plane_count = len(json.load(handle))
    configured_agents = max(plane_count, int(max_plane_agents or plane_count))
    return {
        "batch_num": 1,
        "plane_num_per_batch": plane_count,
        "n_agents": configured_agents,
        "max_device_num": int(max_device_num),
        "resource_policy": resource_policy,
        "global_feature_mode": str(global_feature_mode),
        "jobs_path": str(case / "job.json"),
        "fixed_res_path": str(case / "fixed_resources.json"),
        "mobile_res_path": str(case / "mobile_resources.json"),
        "sites_path": str(case / "sites.json"),
        "flights_path": str(case / "flights.json"),
        "seed": int(seed),
        "interfere": [-1, [], 0],
        "force_chosen": [-1, "", 0],
    }


def legal_plane_candidates(env, pid: int) -> List[dict]:
    """Enumerate action pairs accepted by ``AircraftScheduleEnv.step``."""

    n_jobs = len(env.job_code_list)
    n_sites = len(env.site_code_list)
    op_mask = np.asarray(env.agent_op_mask, dtype=bool)
    site_mask = np.asarray(env.ptr_site_mask_matrix, dtype=bool)
    pair_mask = getattr(env, "agent_job_site_mask_matrix", None)
    if pair_mask is None:
        pair_mask = np.broadcast_to(
            np.asarray(env.job_site_mask_matrix, dtype=bool),
            (env.n_agents, n_jobs, n_sites),
        )
    else:
        pair_mask = np.asarray(pair_mask, dtype=bool)

    candidates = []
    for job_idx, site_idx in np.argwhere(pair_mask[pid]):
        op_global_idx = pid * n_jobs + int(job_idx)
        site_idx = int(site_idx)
        if not op_mask[pid, op_global_idx] or not site_mask[pid, site_idx]:
            continue
        candidates.append({
            "pid": int(pid),
            "job_idx": int(job_idx),
            "site_idx": site_idx,
            "op_global_idx": op_global_idx,
            "job": env.job_code_list[int(job_idx)],
            "site": env.site_code_list[site_idx],
        })
    return candidates


def completion_details(env, step_count: int, max_steps: int) -> dict:
    planes = list(env.planes.values())
    completed = all(plane.is_completed_all_jobs() for plane in planes)
    return {
        "completed": bool(completed),
        "steps": int(step_count),
        "max_steps": int(max_steps),
        "cycle_terminated": bool(getattr(env, "cycle_terminated", False)),
        "cycle_reason": str(getattr(env, "cycle_reason", "")),
        "max_no_progress": int(max(
            (getattr(plane, "no_progress_decisions", 0) for plane in planes),
            default=0,
        )),
        "total_relocations": int(sum(
            getattr(plane, "total_relocations", 0) for plane in planes
        )),
    }


def write_json(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output)
