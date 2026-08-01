"""Run deterministic legal rollouts to verify generated cases exercise devices."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv


def _case_config(case_dir: Path, base_config: dict) -> dict:
    config = copy.deepcopy(base_config)
    config.update(
        {
            "jobs_path": str(case_dir / "job.json"),
            "fixed_res_path": str(case_dir / "fixed_resources.json"),
            "mobile_res_path": str(case_dir / "mobile_resources.json"),
            "sites_path": str(case_dir / "sites.json"),
            "flights_path": str(case_dir / "flights.json"),
            "resource_policy": "drl",
            "use_domain_rand": False,
        }
    )
    return config


def _legal_joint_actions(env, observation, info):
    actions = env.heuristic_device_actions(return_info=True)
    joint_actions = actions["actions"]
    claimed_sites = set()
    active_agents = np.asarray(info["active_agents"], dtype=bool)
    n_ops = len(env.job_code_list)
    for plane_idx in range(env.n_plane_agents):
        if not active_agents[plane_idx]:
            continue
        op_indices = np.flatnonzero(observation.op_mask[plane_idx].cpu().numpy())
        selected = None
        for op_global_idx in op_indices:
            job_idx = int(op_global_idx) - plane_idx * n_ops
            compatible_sites = np.logical_and(
                observation.site_mask_matrix[plane_idx].cpu().numpy(),
                observation.job_site_mask_matrix[job_idx].cpu().numpy(),
            )
            for site_idx in np.flatnonzero(compatible_sites):
                if int(site_idx) not in claimed_sites:
                    selected = (int(op_global_idx), int(site_idx))
                    break
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f"No legal operation-site pair for active plane {plane_idx}")
        joint_actions[plane_idx] = selected
        claimed_sites.add(selected[1])
    return joint_actions, actions["info"]


def run_case(case_dir: Path, base_config: dict, max_steps: int) -> dict:
    env = AircraftScheduleEnv(_case_config(case_dir, base_config))
    observation, dones, info = env.reset()
    label_dispatches = 0
    steps = 0
    while not np.all(dones):
        actions, label_info = _legal_joint_actions(env, observation, info)
        label_dispatches += int(label_info["real_dispatches"])
        observation, _, dones, info = env.step(actions)
        steps += 1
        if steps >= max_steps:
            raise RuntimeError(f"{case_dir.name} exceeded {max_steps} environment steps")

    ordinary_dispatches = sum(
        record.get("device_type") != env.TRANSPORTER_RESOURCE_TYPE
        for record in env.device_trajectory_log
    )
    transporter_dispatches = sum(
        record.get("device_type") == env.TRANSPORTER_RESOURCE_TYPE
        for record in env.device_trajectory_log
    )
    result = {
        "case": case_dir.name,
        "steps": steps,
        "cmax": env.total_time,
        "label_dispatches": label_dispatches,
        "ordinary_device_dispatches": ordinary_dispatches,
        "transporter_dispatches": transporter_dispatches,
    }
    env.close()
    return result


def select_profile_cases(split_dir: Path, max_cases: int) -> list[Path]:
    selected = []
    seen_profiles = set()
    for case_dir in sorted(split_dir.glob("case_*")):
        metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["profile"] in seen_profiles:
            continue
        selected.append(case_dir)
        seen_profiles.add(metadata["profile"])
        if len(selected) >= max_cases:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, default=Path("onpolicy/config/env.yaml"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4000)
    args = parser.parse_args()

    base_config = yaml.safe_load(args.env_config.read_text(encoding="utf-8"))
    case_dirs = select_profile_cases(args.dataset_root / args.split, args.max_cases)
    results = [run_case(case_dir, base_config, args.max_steps) for case_dir in case_dirs]
    profile_counts = Counter()
    for case_dir in case_dirs:
        metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
        profile_counts[metadata["profile"]] += 1
    summary = {
        "status": "PASS" if results and all(row["ordinary_device_dispatches"] > 0 for row in results) else "FAIL",
        "profiles": dict(sorted(profile_counts.items())),
        "cases": results,
        "ordinary_device_dispatches_total": sum(row["ordinary_device_dispatches"] for row in results),
        "transporter_dispatches_total": sum(row["transporter_dispatches"] for row in results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
