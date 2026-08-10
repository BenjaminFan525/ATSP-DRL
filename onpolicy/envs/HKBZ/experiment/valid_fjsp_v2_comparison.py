#!/usr/bin/env python
"""Compare a trained HKBZ policy with legacy baselines on FJSP-v2 test data.

This entry point deliberately reuses the existing model vectorization and
evolutionary baseline implementations.  Its compatibility layer contributes
only the current environment's authoritative joint action masks, new case
naming, integrity metadata, and structured output.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import yaml
from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import Batch

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import (  # noqa: E402
    GNN_MAPPOPolicy as Policy,
)
from onpolicy.config.config import get_config  # noqa: E402
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv  # noqa: E402
from onpolicy.envs.HKBZ.experiment import valid_iga, valid_nsga  # noqa: E402
from onpolicy.envs.HKBZ.experiment.eval_common import (  # noqa: E402
    build_case_env_config,
    completion_details,
    legal_plane_candidates,
    list_case_folders,
    load_case_metadata,
    write_json,
)
from onpolicy.envs.HKBZ.experiment.valid_model_parallel import (  # noqa: E402
    _to_tensor,
    make_eval_envs_from_configs,
)


METHOD_ALIASES = {
    "drl_g": "DRL-G",
    "fifo": "FIFO",
    "spt": "SPT",
    "mwkr": "MWKR",
    "iga": "IGA",
    "nsga2": "NSGA-II",
}


def parse_args(argv):
    parser = get_config()
    parser.add_argument("--ac_config", type=str, default="onpolicy/config/ac.yaml")
    parser.add_argument(
        "--env_config",
        type=str,
        default="onpolicy/config/env_plane_pretrain.yaml",
    )
    parser.add_argument(
        "--dataset_test_dir",
        type=str,
        default=(
            "/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/dataset/"
            "fjsp_v2_t480_v60_test60/test"
        ),
    )
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument(
        "--case_offset", type=int, default=0,
        help="skip cases in the fixed partition order before applying max_cases",
    )
    parser.add_argument(
        "--partition_seed", type=int, default=None,
        help=(
            "optional fixed case-order seed, shared with train_hkbz.py's "
            "--eval_partition_seed for disjoint tune/select evaluation"
        ),
    )
    parser.add_argument(
        "--partition_stratify_by",
        choices=["distribution", "profile"],
        default=None,
        help="optional metadata key for deterministic stratified partitions",
    )
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--model_batch_size", type=int, default=20)
    parser.add_argument("--evolution_time_budget", type=float, default=1800.0)
    parser.add_argument("--iga_pop_size", type=int, default=20)
    parser.add_argument("--iga_generations", type=int, default=20)
    parser.add_argument(
        "--iga_teacher_dir",
        type=str,
        default="",
        help="optional output directory for replayable IGA chromosomes",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_ALIASES),
        choices=list(METHOD_ALIASES),
    )
    parser.add_argument(
        "--policy_tau",
        type=float,
        default=0.0,
        help="deterministic DRL evaluation temperature; <=0 uses checkpoint tau",
    )
    args = parser.parse_args(argv)

    with open(args.env_config, "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle) or {}
    args.max_agent_num = int(env_config.get("n_agents", args.max_agent_num))
    args.max_device_num = int(
        env_config.get("max_device_num", args.max_device_num)
    )
    args.resource_policy = str(
        env_config.get("resource_policy", args.resource_policy)
    )
    if args.resource_policy != "heuristic":
        raise ValueError(
            "Stage-1 comparison requires resource_policy=heuristic; "
            f"got {args.resource_policy!r}."
        )
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint_dir is required for DRL-G evaluation.")
    if args.max_steps <= 0 or args.model_batch_size <= 0:
        raise ValueError("--max_steps and --model_batch_size must be positive.")
    if args.policy_tau < 0.0:
        raise ValueError("--policy_tau must be non-negative.")
    return args


def _case_record(case_name, metadata):
    return {
        "case": case_name,
        "case_id": metadata.get("case_id", case_name),
        "seed": metadata.get("seed"),
        "profile": metadata.get("profile", "unknown"),
        "distribution": metadata.get("distribution", "unknown"),
        "case_sha256": metadata.get("case_sha256"),
    }


def _candidate_preferences(env, info, rule, rng):
    active = np.asarray(info["active_agents"], dtype=bool)
    preferences = {}
    pid_keys = {}
    for plane in env.planes.values():
        pid = int(plane.code.split("_")[-1])
        if pid >= len(active) or not active[pid]:
            continue
        remaining_work = sum(
            float(plane.jobs[job].time or 0.0) for job in plane.left_jobs
        )
        candidates = []
        for candidate in legal_plane_candidates(env, pid):
            enriched = dict(candidate)
            enriched["processing_time"] = float(
                env.jobs[candidate["job"]].time or 0.0
            )
            enriched["remaining_work"] = remaining_work
            enriched["wait_time"] = float(plane.waiting_time)
            enriched["random_key"] = rng.random()
            candidates.append(enriched)

        if not candidates:
            raise RuntimeError(
                f"Active plane {pid} has no legal joint action at step {env.steps}."
            )
        if rule == "Random":
            key = lambda item: (item["random_key"],)
        elif rule == "FIFO":
            key = lambda item: (-item["wait_time"], item["random_key"])
        elif rule == "SPT":
            key = lambda item: (item["processing_time"], item["random_key"])
        elif rule == "MWKR":
            key = lambda item: (-item["remaining_work"], item["random_key"])
        else:
            raise ValueError(f"Unsupported dispatching rule: {rule}")
        candidates.sort(key=key)
        preferences[pid] = candidates
        pid_keys[pid] = key(candidates[0])
    return preferences, sorted(preferences, key=lambda pid: pid_keys[pid])


def _match_unique_sites(preferences, pid_order, n_agents):
    """Find one legal action per active plane with unique site assignments."""

    site_matches = {}

    def augment(pid, seen_sites):
        for candidate in preferences[pid]:
            site_idx = candidate["site_idx"]
            if site_idx in seen_sites:
                continue
            seen_sites.add(site_idx)
            previous = site_matches.get(site_idx)
            if previous is None or augment(previous[0], seen_sites):
                site_matches[site_idx] = (pid, candidate)
                return True
        return False

    for pid in pid_order:
        if not augment(pid, set()):
            counts = {key: len(value) for key, value in preferences.items()}
            raise RuntimeError(
                "No collision-free site matching exists for active planes; "
                f"failed_pid={pid}, candidate_counts={counts}."
            )

    selected = {pid: candidate for pid, candidate in site_matches.values()}
    if set(selected) != set(preferences):
        raise RuntimeError(
            f"Incomplete action matching: selected={sorted(selected)}, "
            f"active={sorted(preferences)}."
        )
    actions = np.full((n_agents, 2), -1, dtype=np.int32)
    for pid, candidate in selected.items():
        actions[pid, 0] = candidate["op_global_idx"]
        actions[pid, 1] = candidate["site_idx"]
    return actions


def mask_aware_dispatch_policy(env, info, rule, rng):
    preferences, pid_order = _candidate_preferences(env, info, rule, rng)
    return _match_unique_sites(preferences, pid_order, env.n_agents)


def mask_aware_ga_policy(env, info, job_priorities, site_priorities):
    active = np.asarray(info["active_agents"], dtype=bool)
    preferences = {}
    pid_keys = {}
    for plane in env.planes.values():
        pid = int(plane.code.split("_")[-1])
        if pid >= len(active) or not active[pid]:
            continue
        candidates = []
        for candidate in legal_plane_candidates(env, pid):
            score = float(
                job_priorities[pid, candidate["job_idx"]]
                + site_priorities[pid, candidate["site_idx"]]
            )
            enriched = dict(candidate)
            enriched["score"] = score
            candidates.append(enriched)
        if not candidates:
            raise RuntimeError(
                f"Active plane {pid} has no legal joint action at step {env.steps}."
            )
        candidates.sort(
            key=lambda item: (-item["score"], item["job_idx"], item["site_idx"])
        )
        preferences[pid] = candidates
        pid_keys[pid] = -candidates[0]["score"]
    pid_order = sorted(preferences, key=lambda pid: (pid_keys[pid], pid))
    return _match_unique_sites(preferences, pid_order, env.n_agents)


def run_dispatch_case(case_path, rule, args, random_seed):
    config = build_case_env_config(
        case_path,
        max_plane_agents=args.max_agent_num,
        max_device_num=args.max_device_num,
        resource_policy="heuristic",
        seed=42,
        global_feature_mode=args.global_feature_mode,
    )
    env = AircraftScheduleEnv(config)
    env.use_domain_rand = False
    _, done, info = env.reset()
    rng = random.Random(int(random_seed))
    step_count = 0
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    while not np.all(done) and step_count < args.max_steps:
        actions = mask_aware_dispatch_policy(env, info, rule, rng)
        _, _, done, info = env.step(actions)
        step_count += 1
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    details = completion_details(env, step_count, args.max_steps)
    details["timeout"] = bool(not np.all(done))
    details["completed"] = bool(
        details["completed"]
        and not details["cycle_terminated"]
        and not details["timeout"]
    )
    details.update({
        "makespan": float(env.total_time),
        "cpu_seconds": float(cpu_seconds),
        "wall_seconds": float(wall_seconds),
    })
    return details


def _numpy(value):
    return value.detach().cpu().numpy()


def run_model_batch(case_names, dataset_dir, policy, args, metadata):
    configs = [
        build_case_env_config(
            str(Path(dataset_dir) / case_name),
            max_plane_agents=args.max_agent_num,
            max_device_num=args.max_device_num,
            resource_policy="heuristic",
            seed=42,
            global_feature_mode=args.global_feature_mode,
        )
        for case_name in case_names
    ]
    envs = make_eval_envs_from_configs(configs, base_seed=args.seed)
    num_envs = len(configs)
    total_agents = args.max_agent_num
    rnn_states = np.zeros(
        (num_envs, total_agents, args.recurrent_N, args.hidden_size),
        dtype=np.float32,
    )
    done_flags = np.zeros(num_envs, dtype=bool)
    cycle_flags = np.zeros(num_envs, dtype=bool)
    finish_steps = np.zeros(num_envs, dtype=np.int32)
    relocations = np.zeros(num_envs, dtype=np.float64)
    max_no_progress = np.zeros(num_envs, dtype=np.int32)
    actions = None
    wall_started = time.perf_counter()
    try:
        obs_dicts, dones, infos = envs.reset()
        for eval_step in range(args.max_steps):
            active_masks = np.zeros(
                (num_envs, total_agents, 1), dtype=np.float32
            )
            active_masks[np.asarray(infos["active_agents"], dtype=bool)] = 1.0
            rebuilt = [HeteroData.from_dict(_to_tensor(obs)) for obs in obs_dicts]
            graph_batch = Batch.from_data_list(rebuilt).to(
                next(policy.ac.parameters()).device,
                non_blocking=True,
            )
            with torch.inference_mode():
                action_tensor, rnn_tensor = policy.act(
                    graph_obs=graph_batch,
                    rnn_states=rnn_states,
                    active_agents=active_masks,
                    last_op_indices=(
                        actions[..., 0]
                        if actions is not None
                        else -np.ones((num_envs, total_agents), dtype=np.float32)
                    ),
                    last_site_indices=(
                        actions[..., 1]
                        if actions is not None
                        else -np.ones((num_envs, total_agents), dtype=np.float32)
                    ),
                    deterministic=True,
                )
            actions = _numpy(action_tensor)
            rnn_states = _numpy(rnn_tensor)
            obs_dicts, _, dones, infos = envs.step(actions)
            rnn_states[np.asarray(dones, dtype=bool)] = 0.0
            new_done = np.all(dones, axis=1)
            just_finished = new_done & ~done_flags
            finish_steps[just_finished] = eval_step + 1
            done_flags |= new_done
            cycle_flags |= np.asarray(
                infos.get("cycle_terminated", np.zeros(num_envs)), dtype=bool
            ).reshape(-1)
            relocations = np.asarray(
                infos.get("total_relocations", relocations), dtype=float
            ).reshape(-1)
            max_no_progress = np.maximum(
                max_no_progress,
                np.asarray(
                    infos.get("max_no_progress", np.zeros(num_envs)), dtype=np.int32
                ).reshape(-1),
            )
            if np.all(done_flags):
                break
        makespans = np.asarray(envs.get_episode_rewards(), dtype=float).reshape(-1)
    finally:
        envs.close()
    wall_seconds = time.perf_counter() - wall_started

    records = []
    for index, case_name in enumerate(case_names):
        timeout = bool(not done_flags[index])
        cycle = bool(cycle_flags[index])
        record = _case_record(case_name, metadata.get(case_name, {}))
        record.update({
            "makespan": float(makespans[index]),
            "wall_seconds": float(wall_seconds / max(1, num_envs)),
            "batch_wall_seconds": float(wall_seconds),
            "steps": int(finish_steps[index] if done_flags[index] else args.max_steps),
            "total_relocations": float(relocations[index]),
            "max_no_progress": int(max_no_progress[index]),
            "completed": bool(not timeout and not cycle),
            "cycle_terminated": cycle,
            "timeout": timeout,
        })
        records.append(record)
    return records


def load_policy(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(args.ac_config, "r", encoding="utf-8") as handle:
        ac_config = yaml.safe_load(handle) or {}
    policy = Policy(args, ac_config, device=device)
    checkpoint = torch.load(args.checkpoint_dir, map_location=device)
    policy.load_model_state(checkpoint["model"])
    checkpoint_tau = float(checkpoint.get("tau", 1.0))
    policy.ac.tau = (
        float(args.policy_tau)
        if float(args.policy_tau) > 0.0
        else checkpoint_tau
    )
    policy.ac.eval()
    return policy, checkpoint, device


def run_evolutionary_case(method, case_path, args):
    teacher = None
    if method == "iga":
        export_teacher = bool(args.iga_teacher_dir)
        iga_result = valid_iga.test_iga_on_case(
            case_path,
            max_time_seconds=args.evolution_time_budget,
            pop_size=args.iga_pop_size,
            n_gen=args.iga_generations,
            seed=args.seed,
            return_solution=export_teacher,
        )
        if export_teacher:
            cmax = iga_result.get("makespan")
            cpu_seconds = iga_result.get("cpu_seconds")
            chromosome = iga_result.get("chromosome")
            if chromosome is not None:
                chromosome = np.asarray(chromosome, dtype=np.float64)
                split = iga_result["n_agents"] * iga_result["n_jobs"]
                teacher = {
                    "schema_version": 1,
                    "case": Path(case_path).name,
                    "makespan": cmax,
                    "job_priorities": chromosome[:split].reshape(
                        iga_result["n_agents"], iga_result["n_jobs"]
                    ).tolist(),
                    "site_priorities": chromosome[split:].reshape(
                        iga_result["n_agents"], iga_result["n_sites"]
                    ).tolist(),
                    "iga": {
                        key: iga_result[key]
                        for key in (
                            "cpu_seconds", "wall_seconds", "pop_size", "n_gen",
                            "seed", "time_budget_seconds"
                        )
                    },
                }
        else:
            cmax, cpu_seconds = iga_result
    elif method == "nsga2":
        cmax, cpu_seconds = valid_nsga.test_nsga2_on_case(case_path)
    else:
        raise ValueError(method)
    cmax = float(np.asarray(cmax).reshape(-1)[0]) if cmax is not None else math.nan
    valid = bool(np.isfinite(cmax) and cmax < 100000.0)
    result = {
        "makespan": cmax,
        "cpu_seconds": float(cpu_seconds) if cpu_seconds is not None else math.nan,
        "completed": valid,
        "cycle_terminated": False,
        "timeout": False,
        "completion_verified": False,
    }
    return result, teacher


def summarize(records):
    completed = [record for record in records if record.get("completed")]
    makespans = np.asarray([record["makespan"] for record in completed], dtype=float)
    wall = np.asarray(
        [record.get("wall_seconds", math.nan) for record in completed], dtype=float
    )
    return {
        "case_count": len(records),
        "completed_count": len(completed),
        "completion_rate": len(completed) / max(1, len(records)),
        "cycle_count": sum(bool(record.get("cycle_terminated")) for record in records),
        "timeout_count": sum(bool(record.get("timeout")) for record in records),
        "error_count": sum(bool(record.get("error")) for record in records),
        "mean_makespan": float(np.mean(makespans)) if len(makespans) else math.nan,
        "std_makespan": float(np.std(makespans)) if len(makespans) else math.nan,
        "median_makespan": float(np.median(makespans)) if len(makespans) else math.nan,
        "mean_wall_seconds": (
            float(np.nanmean(wall)) if len(wall) and np.isfinite(wall).any() else math.nan
        ),
    }


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dataset_dir = str(Path(args.dataset_test_dir).resolve())
    case_names = list_case_folders(
        dataset_dir,
        args.max_cases,
        case_offset=args.case_offset,
        partition_seed=args.partition_seed,
        stratify_by=args.partition_stratify_by,
    )
    metadata = load_case_metadata(dataset_dir)
    if not case_names:
        raise RuntimeError(f"No test cases found in {dataset_dir}")

    # Existing evolutionary problems resolve GA_Policy from their module globals.
    valid_iga.GA_Policy = mask_aware_ga_policy
    valid_nsga.GA_Policy = mask_aware_ga_policy

    payload = {
        "status": "running",
        "created_unix_time": time.time(),
        "dataset_test_dir": dataset_dir,
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "env_config": str(Path(args.env_config).resolve()),
        "ac_config": str(Path(args.ac_config).resolve()),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "methods_requested": [METHOD_ALIASES[name] for name in args.methods],
        "requested_policy_tau": float(args.policy_tau),
        "case_count": len(case_names),
        "case_offset": int(args.case_offset),
        "partition_seed": args.partition_seed,
        "partition_stratify_by": args.partition_stratify_by,
        "methods": {},
    }
    write_json(args.output_json, payload)

    if "drl_g" in args.methods:
        print(f"[DRL-G] loading checkpoint {args.checkpoint_dir}", flush=True)
        policy, checkpoint, device = load_policy(args)
        records = []
        for start in range(0, len(case_names), args.model_batch_size):
            batch_names = case_names[start:start + args.model_batch_size]
            batch_records = run_model_batch(
                batch_names, dataset_dir, policy, args, metadata
            )
            records.extend(batch_records)
            print(
                f"[DRL-G] completed {len(records)}/{len(case_names)} cases; "
                f"batch_mean={np.mean([x['makespan'] for x in batch_records]):.3f}",
                flush=True,
            )
            payload["methods"]["DRL-G"] = {
                "status": "running",
                "device": str(device),
                "checkpoint_tau": float(checkpoint.get("tau", 1.0)),
                "tau": float(policy.ac.tau),
                "cases": records,
                "summary": summarize(records),
            }
            write_json(args.output_json, payload)
        payload["methods"]["DRL-G"]["status"] = "completed"
        write_json(args.output_json, payload)

    dispatch_rules = {"fifo": "FIFO", "spt": "SPT", "mwkr": "MWKR"}
    for method, rule in dispatch_rules.items():
        if method not in args.methods:
            continue
        records = []
        print(f"[{rule}] starting {len(case_names)} cases", flush=True)
        for index, case_name in enumerate(case_names):
            case_path = str(Path(dataset_dir) / case_name)
            record = _case_record(case_name, metadata.get(case_name, {}))
            try:
                result = run_dispatch_case(
                    case_path,
                    rule,
                    args,
                    random_seed=args.seed * 100000 + index,
                )
                record.update(result)
            except Exception as error:
                record.update({
                    "completed": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                })
            records.append(record)
            print(
                f"[{rule}] {index + 1}/{len(case_names)} {case_name} "
                f"Cmax={record.get('makespan')} completed={record.get('completed')}",
                flush=True,
            )
            payload["methods"][rule] = {
                "status": "running",
                "cases": records,
                "summary": summarize(records),
            }
            write_json(args.output_json, payload)
        payload["methods"][rule]["status"] = "completed"
        write_json(args.output_json, payload)

    for method in ("iga", "nsga2"):
        if method not in args.methods:
            continue
        label = METHOD_ALIASES[method]
        records = []
        print(f"[{label}] starting {len(case_names)} cases", flush=True)
        for index, case_name in enumerate(case_names):
            case_path = str(Path(dataset_dir) / case_name)
            record = _case_record(case_name, metadata.get(case_name, {}))
            wall_started = time.perf_counter()
            try:
                result, teacher = run_evolutionary_case(
                    method, case_path, args
                )
                if teacher is not None:
                    write_json(
                        str(Path(args.iga_teacher_dir) / f"{case_name}.json"),
                        teacher,
                    )
                result["wall_seconds"] = time.perf_counter() - wall_started
                record.update(result)
            except Exception as error:
                record.update({
                    "completed": False,
                    "wall_seconds": time.perf_counter() - wall_started,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                })
            records.append(record)
            print(
                f"[{label}] {index + 1}/{len(case_names)} {case_name} "
                f"Cmax={record.get('makespan')} completed={record.get('completed')}",
                flush=True,
            )
            payload["methods"][label] = {
                "status": "running",
                "legacy_budget": {
                    "population": args.iga_pop_size if method == "iga" else 20,
                    "generations": args.iga_generations if method == "iga" else 20,
                    "time_seconds": args.evolution_time_budget if method == "iga" else None,
                },
                "cases": records,
                "summary": summarize(records),
            }
            write_json(args.output_json, payload)
        payload["methods"][label]["status"] = "completed"
        write_json(args.output_json, payload)

    payload["status"] = "completed"
    payload["completed_unix_time"] = time.time()
    write_json(args.output_json, payload)
    print(json.dumps(
        {name: value["summary"] for name, value in payload["methods"].items()},
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
