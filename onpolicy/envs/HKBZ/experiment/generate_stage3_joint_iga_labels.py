#!/usr/bin/env python3
"""Generate replay-verified full-joint (Stage-3) IGA teachers.

The chromosome controls both decision families executed by the final neural
policy: aircraft operation/site pairs and all mobile-device requests.  Search
and final replay deliberately use the same authoritative environment masks and
the production resource decoder.  Completed teachers are written atomically
per case, so a systemd cgroup can pre-empt a run without corrupting or losing
already completed cases.

IGA-1800 is a nested continuation: pass ``--warm-start-dir`` pointing at the
verified IGA-180 output and use an additional budget of 1620 seconds.  Existing
S1/S2 teachers can separately seed the two halves of the first joint search.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import signal
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv  # noqa: E402
from onpolicy.envs.HKBZ.experiment.eval_common import (  # noqa: E402
    build_case_env_config,
    completion_details,
    list_case_folders,
    write_json,
)
from onpolicy.envs.HKBZ.experiment.resource_wait_metrics import (  # noqa: E402
    summarize_aircraft_resource_wait,
)
from onpolicy.envs.HKBZ.experiment.valid_iga import GA_Policy  # noqa: E402
from onpolicy.envs.HKBZ.resource_teacher import (  # noqa: E402
    ResourceGenomeLayout,
    mixed_resource_actions,
)


TEACHER_SCOPE = "stage3_full_joint_policy"
TEACHER_METHOD = "joint_iga_all"
SCHEMA_VERSION = 1
SEARCH_CONTRACT_VERSION = 4
# Contract v2 IGA-180 teachers remain valid warm incumbents.  Version 3 only
# changes continuation accounting so that a replay-verified incumbent skipped
# in generation zero still counts as a resolved population member.  Version 4
# binds every Stage3 planning field that changes the search/replay MDP.  A warm
# incumbent is accepted only when those recorded fields match as well.
NESTED_WARM_SEARCH_CONTRACT_VERSIONS = frozenset(
    (2, 3, SEARCH_CONTRACT_VERSION)
)
BACKENDS = {"ordinary": "iga", "transporter": "iga"}
PLANNING_CONTRACT_KEYS = (
    "device_lookahead_dispatch",
    "device_lookahead_safety_margin",
    "device_deadline_aware_dispatch",
    "device_future_intent_horizon",
    "device_future_intent_mode",
    "device_frontier_max_requests",
    "device_request_capacity_per_plane",
    "resource_release_aware_eta",
    "device_lookahead_reservation_mode",
    "device_reservation_grace_seconds",
    "device_departure_lookahead",
    "resource_slack_forecast_seconds",
)
CASE_FILES = (
    "job.json",
    "fixed_resources.json",
    "mobile_resources.json",
    "sites.json",
    "flights.json",
)
_STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[Stage3IGA] received signal {signum}; stop submitting cases", flush=True)


def _atomic_json(path: Path, payload: Mapping) -> None:
    write_json(str(path), dict(payload))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_sha256(case_path: Path) -> str:
    metadata_path = case_path / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value = metadata.get("fingerprints", {}).get("case_sha256")
        if value:
            return str(value)
    digest = hashlib.sha256()
    for name in CASE_FILES:
        path = case_path / name
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _metadata(case_path: Path) -> dict:
    path = case_path / "metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _derived_seed(base_seed: int, case: str, phase: str) -> int:
    data = f"{int(base_seed)}:{case}:{phase}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def _planning_contract(
    *,
    lookahead_margin: float,
    future_intent_horizon: int,
    future_intent_mode: str,
    frontier_max_requests: int,
    request_capacity_per_plane: int,
    release_aware_eta: bool,
    reservation_mode: str,
    reservation_grace_seconds: float,
    slack_forecast_seconds: float,
) -> dict:
    """Return the exact decision-opportunity contract used by search/replay."""

    return {
        "device_lookahead_dispatch": True,
        "device_lookahead_safety_margin": float(lookahead_margin),
        "device_deadline_aware_dispatch": True,
        "device_future_intent_horizon": int(future_intent_horizon),
        "device_future_intent_mode": str(future_intent_mode),
        "device_frontier_max_requests": int(frontier_max_requests),
        "device_request_capacity_per_plane": int(request_capacity_per_plane),
        "resource_release_aware_eta": bool(release_aware_eta),
        "device_lookahead_reservation_mode": str(reservation_mode),
        "device_reservation_grace_seconds": float(reservation_grace_seconds),
        "device_departure_lookahead": True,
        "resource_slack_forecast_seconds": float(slack_forecast_seconds),
    }


def _env_config(
    case_path: Path,
    *,
    max_plane_agents: int,
    max_device_num: int,
    max_steps: int,
    lookahead_margin: float,
    future_intent_horizon: int,
    future_intent_mode: str,
    frontier_max_requests: int,
    request_capacity_per_plane: int,
    release_aware_eta: bool,
    reservation_mode: str,
    reservation_grace_seconds: float,
    slack_forecast_seconds: float,
) -> dict:
    config = build_case_env_config(
        str(case_path),
        max_plane_agents=max_plane_agents,
        max_device_num=max_device_num,
        resource_policy="drl",
        seed=42,
        global_feature_mode="f1f2",
    )
    planning = _planning_contract(
        lookahead_margin=lookahead_margin,
        future_intent_horizon=future_intent_horizon,
        future_intent_mode=future_intent_mode,
        frontier_max_requests=frontier_max_requests,
        request_capacity_per_plane=request_capacity_per_plane,
        release_aware_eta=release_aware_eta,
        reservation_mode=reservation_mode,
        reservation_grace_seconds=reservation_grace_seconds,
        slack_forecast_seconds=slack_forecast_seconds,
    )
    config.update(
        {
            "use_domain_rand": False,
            "plane_cycle_repeat_limit": 8,
            "plane_no_progress_limit": 120,
            "plane_relocation_limit": 40,
            # Match the currently trained Wave-3 decision-opportunity
            # contract.  These controls change which future requests are
            # exposed and legal; omitting them would create labels for a
            # different MDP even though the base semantics version is equal.
            **planning,
            # Kept as provenance even though the loop, rather than the env,
            # enforces this limit.
            "max_episode_steps": int(max_steps),
        }
    )
    return config


@dataclass(frozen=True)
class JointGenomeLayout:
    n_plane_agents: int
    n_jobs: int
    n_sites: int
    resource: ResourceGenomeLayout

    @classmethod
    def from_env(cls, env: AircraftScheduleEnv) -> "JointGenomeLayout":
        return cls(
            n_plane_agents=int(env.n_plane_agents),
            n_jobs=len(env.job_code_list),
            n_sites=len(env.site_code_list),
            resource=ResourceGenomeLayout.from_env(env, BACKENDS),
        )

    @property
    def plane_n_var(self) -> int:
        return self.n_plane_agents * (self.n_jobs + self.n_sites)

    @property
    def n_var(self) -> int:
        return self.plane_n_var + self.resource.n_var

    def decode(self, chromosome: np.ndarray):
        values = np.asarray(chromosome, dtype=np.float64).reshape(-1)
        if values.size != self.n_var:
            raise ValueError(
                f"Joint chromosome has {values.size} variables; expected {self.n_var}."
            )
        job_end = self.n_plane_agents * self.n_jobs
        site_end = self.plane_n_var
        jobs = values[:job_end].reshape(self.n_plane_agents, self.n_jobs)
        sites = values[job_end:site_end].reshape(
            self.n_plane_agents, self.n_sites
        )
        resources = self.resource.decode(values[site_end:])
        return jobs, sites, resources

    def record(self) -> dict:
        return {
            "n_plane_agents": self.n_plane_agents,
            "n_jobs": self.n_jobs,
            "n_sites": self.n_sites,
            "plane_n_var": self.plane_n_var,
            "resource_n_var": self.resource.n_var,
            "n_var": self.n_var,
            "iga_device_count": len(self.resource.selected_device_indices),
            "resource_row_width": self.resource.row_width,
        }


def _build_layout(config: Mapping) -> JointGenomeLayout:
    env = AircraftScheduleEnv(dict(config))
    try:
        return JointGenomeLayout.from_env(env)
    finally:
        env.close()


def _plain_request(request: Mapping) -> dict:
    result = {}
    for key in (
        "id",
        "plane_idx",
        "plane_id",
        "job_code",
        "site_code",
        "needed_res_types",
        "waiting_time",
        "lead_time",
        "is_lookahead",
        "intrinsic_ready_time",
    ):
        value = request.get(key)
        if key == "intrinsic_ready_time" and value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _plane_decisions(env: AircraftScheduleEnv, actions: np.ndarray) -> list[dict]:
    records = []
    for plane in env.planes.values():
        pid = int(plane.code.split("_")[-1])
        # ``GA_Policy`` leaves non-deciding plane rows at [-1, -1].  Reading
        # that output is more robust than depending on a transient env field
        # which is populated only inside ``step``.
        if (
            pid >= env.n_plane_agents
            or int(actions[pid, 0]) < 0
            or int(actions[pid, 1]) < 0
        ):
            continue
        op_idx = int(actions[pid, 0])
        site_idx = int(actions[pid, 1])
        job_idx = op_idx - pid * len(env.job_code_list)
        records.append(
            {
                "agent_id": pid,
                "plane_id": plane.code,
                "operation_index": op_idx,
                "job_code": (
                    env.job_code_list[job_idx]
                    if 0 <= job_idx < len(env.job_code_list)
                    else None
                ),
                "site_index": site_idx,
                "site_code": (
                    env.site_code_list[site_idx]
                    if 0 <= site_idx < len(env.site_code_list)
                    else None
                ),
            }
        )
    return records


def evaluate_joint(
    config: Mapping,
    chromosome: np.ndarray,
    *,
    max_steps: int,
    deadline: float | None = None,
    record_trace: bool = False,
) -> dict:
    """Evaluate one chromosome; a deadline may abort only search candidates."""

    started = time.monotonic()
    env = AircraftScheduleEnv(dict(config))
    trace = []
    error = None
    interrupted = False
    step_count = 0
    try:
        layout = JointGenomeLayout.from_env(env)
        job_priorities, site_priorities, decoded_resource = layout.decode(chromosome)
        _, dones, info = env.reset()
        while not np.all(dones) and step_count < int(max_steps):
            if deadline is not None and time.monotonic() >= deadline:
                interrupted = True
                break
            plane_actions = GA_Policy(
                env, info, job_priorities, site_priorities
            )
            resource_actions, resource_decisions = mixed_resource_actions(
                env, BACKENDS, decoded_resource, record=record_trace
            )
            plane_actions[env.n_plane_agents :] = resource_actions[
                env.n_plane_agents :
            ]
            event = None
            if record_trace:
                event = {
                    "step": step_count,
                    "env_step": int(env.steps),
                    "time_before": float(env.total_time),
                    "requests": [
                        _plain_request(request) for request in env.request_list[1:]
                    ],
                    "plane_decisions": _plane_decisions(env, plane_actions),
                    "resource_decisions": resource_decisions,
                }
            _, rewards, dones, info = env.step(plane_actions)
            if event is not None:
                event["time_after"] = float(env.total_time)
                event["elapsed"] = event["time_after"] - event["time_before"]
                event["reward_sum"] = float(np.asarray(rewards).sum())
                trace.append(event)
            step_count += 1
    except Exception as exc:  # failed candidates are penalized, not fatal
        error = f"{type(exc).__name__}: {exc}"

    details = completion_details(env, step_count, int(max_steps))
    completed = bool(details["completed"] and error is None and not interrupted)
    result = {
        "completed": completed,
        "makespan": float(env.total_time),
        "wall_seconds": float(time.monotonic() - started),
        "error": error,
        "deadline_interrupted": interrupted,
        "completion": details,
    }
    if record_trace:
        result.update(
            {
                "decision_trace": trace,
                "plane_trajectory": env.trajectory_log,
                "resource_trajectory": env.device_trajectory_log,
                "resource_decision_log": env.device_decision_log,
                "aircraft_resource_wait": summarize_aircraft_resource_wait(
                    env.trajectory_log,
                    env.device_trajectory_log,
                    {
                        code: env._needed_mobile_types(code)
                        for code in env.job_code_list
                    },
                    transporter_type=env.TRANSPORTER_RESOURCE_TYPE,
                    aircraft_count=len(env.flights_data),
                    include_events=True,
                ),
            }
        )
    env.close()
    return result


def _valid_vector(values, expected_size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != expected_size:
        raise ValueError(f"Warm vector has {vector.size} values; expected {expected_size}.")
    if not np.isfinite(vector).all() or np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError("Warm chromosome must be finite and inside [0, 1].")
    return vector


def _load_split_warm(
    case: str,
    layout: JointGenomeLayout,
    plane_dir: Path | None,
    resource_dir: Path | None,
    expected_case_sha: str,
) -> tuple[np.ndarray | None, dict | None]:
    if plane_dir is None or resource_dir is None:
        return None, None
    plane_path = plane_dir / f"{case}.json"
    resource_path = resource_dir / "teachers" / f"{case}.json"
    if not plane_path.is_file() or not resource_path.is_file():
        return None, None
    plane = json.loads(plane_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    required_plane = {
        "teacher_scope": "stage1_plane_policy",
        "case": case,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "resource_policy": "heuristic",
        "completion_verified": True,
    }
    required_resource = {
        "teacher_scope": "stage2_resource_policy",
        "teacher_method": "resource_iga_all",
        "case": case,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "resource_policy": "drl",
        "arm": "iga_all",
        "completion_verified": True,
    }
    if any(plane.get(key) != value for key, value in required_plane.items()):
        raise ValueError(f"Incompatible S1 split warm start for {case}.")
    if plane.get("case_sha256") != expected_case_sha:
        raise ValueError(f"S1 split warm-start case hash mismatch for {case}.")
    if any(resource.get(key) != value for key, value in required_resource.items()):
        raise ValueError(f"Incompatible S2 split warm start for {case}.")
    if resource.get("backends") != BACKENDS:
        raise ValueError(f"Unverified split warm start for {case}.")
    jobs = np.full((layout.n_plane_agents, layout.n_jobs), 0.5)
    sites = np.full((layout.n_plane_agents, layout.n_sites), 0.5)
    source_jobs = np.asarray(plane.get("job_priorities"), dtype=np.float64)
    source_sites = np.asarray(plane.get("site_priorities"), dtype=np.float64)
    if source_jobs.ndim != 2 or source_jobs.shape[1] != layout.n_jobs:
        raise ValueError(f"Incompatible S1 job priorities for {case}: {source_jobs.shape}")
    if source_sites.ndim != 2 or source_sites.shape[1] != layout.n_sites:
        raise ValueError(f"Incompatible S1 site priorities for {case}: {source_sites.shape}")
    rows = min(layout.n_plane_agents, source_jobs.shape[0], source_sites.shape[0])
    jobs[:rows] = source_jobs[:rows]
    sites[:rows] = source_sites[:rows]
    resources = _valid_vector(
        resource.get("search", {}).get("chromosome", []), layout.resource.n_var
    )
    chromosome = np.concatenate((jobs.reshape(-1), sites.reshape(-1), resources))
    return chromosome, {
        "kind": "verified_s1_plane_plus_s2_resource",
        "plane_path": str(plane_path.resolve()),
        "plane_sha256": _sha256_file(plane_path),
        "plane_makespan": float(plane["makespan"]),
        "resource_path": str(resource_path.resolve()),
        "resource_sha256": _sha256_file(resource_path),
        "resource_makespan": float(resource["makespan"]),
    }


def _load_joint_warm(
    case: str,
    layout: JointGenomeLayout,
    warm_start_dir: Path | None,
    expected_budget: float,
    expected_contract: Mapping,
) -> tuple[np.ndarray | None, float | None, dict | None]:
    if warm_start_dir is None:
        return None, None, None
    source = warm_start_dir / "teachers" / f"{case}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing nested Stage3 warm start: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "status": "completed",
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "case": case,
        "completion_verified": True,
        **{
            key: expected_contract[key]
            for key in PLANNING_CONTRACT_KEYS
            if key in expected_contract
        },
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in required.items()
        if payload.get(key) != value
    }
    source_contract = int(payload.get("search_contract_version", -1))
    source_budget = float(payload.get("nominal_cumulative_budget_seconds", -1))
    if (
        mismatch
        or source_contract not in NESTED_WARM_SEARCH_CONTRACT_VERSIONS
        or not math.isclose(source_budget, expected_budget, abs_tol=1e-9)
    ):
        raise ValueError(f"Incompatible nested warm start {source}: {mismatch}")
    chromosome = _valid_vector(
        payload.get("search", {}).get("chromosome", []), layout.n_var
    )
    return chromosome, float(payload["makespan"]), {
        "kind": "verified_stage3_nested_incumbent",
        "path": str(source.resolve()),
        "sha256": _sha256_file(source),
        "source_search_contract_version": source_contract,
        "source_nominal_budget_seconds": source_budget,
        "source_makespan": float(payload["makespan"]),
    }


def _polynomial_mutation(
    rng: np.random.Generator, child: np.ndarray, eta: float = 20.0
) -> np.ndarray:
    child = child.copy()
    mask = rng.random(child.size) < (1.0 / max(1, child.size))
    if not mask.any():
        mask[int(rng.integers(0, child.size))] = True
    values = child[mask]
    u = rng.random(values.size)
    delta = np.where(
        u < 0.5,
        np.power(2.0 * u, 1.0 / (eta + 1.0)) - 1.0,
        1.0 - np.power(2.0 * (1.0 - u), 1.0 / (eta + 1.0)),
    )
    child[mask] = np.clip(values + delta, 0.0, 1.0)
    return child


def _sbx_children(rng, left, right, eta: float = 15.0):
    u = rng.random(left.size)
    beta = np.where(
        u <= 0.5,
        np.power(2.0 * u, 1.0 / (eta + 1.0)),
        np.power(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0)),
    )
    a = np.clip(0.5 * ((1.0 + beta) * left + (1.0 - beta) * right), 0, 1)
    b = np.clip(0.5 * ((1.0 - beta) * left + (1.0 + beta) * right), 0, 1)
    return a, b


def _next_population(rng, population, fitness):
    size = len(population)
    result = [population[int(np.argmin(fitness))].copy()]

    def tournament():
        left, right = rng.integers(0, size, size=2)
        return int(left if fitness[left] <= fitness[right] else right)

    while len(result) < size:
        first, second = _sbx_children(
            rng, population[tournament()], population[tournament()]
        )
        result.append(_polynomial_mutation(rng, first))
        if len(result) < size:
            result.append(_polynomial_mutation(rng, second))
    return np.asarray(result)


def _search_case(task: Mapping) -> dict:
    case_path = Path(task["case_path"])
    case = case_path.name
    config = _env_config(
        case_path,
        max_plane_agents=int(task["max_plane_agents"]),
        max_device_num=int(task["max_device_num"]),
        max_steps=int(task["max_steps"]),
        lookahead_margin=float(task["lookahead_margin"]),
        future_intent_horizon=int(task["future_intent_horizon"]),
        future_intent_mode=str(task["future_intent_mode"]),
        frontier_max_requests=int(task["frontier_max_requests"]),
        request_capacity_per_plane=int(task["request_capacity_per_plane"]),
        release_aware_eta=bool(task["release_aware_eta"]),
        reservation_mode=str(task["reservation_mode"]),
        reservation_grace_seconds=float(task["reservation_grace_seconds"]),
        slack_forecast_seconds=float(task["slack_forecast_seconds"]),
    )
    layout = _build_layout(config)
    joint_warm, joint_objective, warm_evidence = _load_joint_warm(
        case,
        layout,
        Path(task["warm_start_dir"]) if task.get("warm_start_dir") else None,
        float(task["warm_start_budget"]),
        task["expected"],
    )
    if joint_warm is None:
        joint_warm, warm_evidence = _load_split_warm(
            case,
            layout,
            Path(task["plane_warm_dir"]) if task.get("plane_warm_dir") else None,
            Path(task["resource_warm_dir"])
            if task.get("resource_warm_dir")
            else None,
            str(task["expected"]["case_sha256"]),
        )

    seed = _derived_seed(int(task["seed"]), case, str(task["cumulative_budget"]))
    rng = np.random.default_rng(seed)
    population_size = int(task["population"])
    population = rng.random((population_size, layout.n_var))
    heuristic = np.full(layout.n_var, 0.5, dtype=np.float64)
    heuristic[-4:] = np.asarray([1.0, 1.0, 0.5, 1.0])
    population[0] = joint_warm if joint_warm is not None else heuristic
    if joint_warm is not None and population_size > 1:
        population[1] = heuristic

    # A split S1+S2 chromosome is only a promising candidate: the two source
    # teachers were optimized under different complementary policies, so the
    # concatenation is not considered feasible until this joint evaluator has
    # completed it.  Only a prior Stage3 teacher may be inherited directly.
    best = (
        joint_warm.copy()
        if joint_warm is not None and joint_objective is not None
        else None
    )
    best_objective = float(joint_objective) if joint_objective is not None else math.inf
    inherited = int(joint_objective is not None)
    evaluated = 0
    completed_candidates = 0
    generation = 0
    partial = 0
    history = []
    started = time.monotonic()
    deadline = started + float(task["time_budget"])
    while generation < int(task["max_generations"]):
        fitness = np.full(population_size, math.inf, dtype=np.float64)
        evaluated_this_generation = 0
        resolved_this_generation = 0
        inherited_this_generation = 0
        for index, chromosome in enumerate(population):
            # A nested warm incumbent is already replay verified.  Do not
            # spend a second evaluation on its exact elite in generation zero.
            if generation == 0 and index == 0 and joint_objective is not None:
                fitness[index] = joint_objective
                resolved_this_generation += 1
                inherited_this_generation += 1
                continue
            if best is not None and time.monotonic() >= deadline:
                break
            episode = evaluate_joint(
                config,
                chromosome,
                max_steps=int(task["max_steps"]),
                deadline=deadline if best is not None else None,
                record_trace=False,
            )
            if episode["deadline_interrupted"]:
                break
            evaluated += 1
            evaluated_this_generation += 1
            resolved_this_generation += 1
            if episode["completed"]:
                completed_candidates += 1
                objective = float(episode["makespan"])
                fitness[index] = objective
                if objective < best_objective:
                    best_objective = objective
                    best = np.asarray(chromosome, dtype=np.float64).copy()
        partial = evaluated_this_generation
        history.append(
            {
                "generation": generation,
                "evaluated": evaluated_this_generation,
                "resolved": resolved_this_generation,
                "inherited": inherited_this_generation,
                "best_makespan": (
                    float(best_objective) if math.isfinite(best_objective) else None
                ),
                "wall_seconds": float(time.monotonic() - started),
            }
        )
        if best is None:
            raise RuntimeError(f"{case} produced no completed joint candidate.")
        # A verified nested incumbent is deliberately not reevaluated.  It is
        # nevertheless a resolved member of generation zero; comparing only
        # the number of newly evaluated candidates to population_size made
        # every continuation stop at 19/20 before crossover and mutation.
        if (
            resolved_this_generation < population_size
            or time.monotonic() >= deadline
        ):
            break
        generation += 1
        population = _next_population(rng, population, fitness)

    optimization_wall = time.monotonic() - started
    final = evaluate_joint(
        config,
        best,
        max_steps=int(task["max_steps"]),
        deadline=None,
        record_trace=True,
    )
    if not final["completed"]:
        raise RuntimeError(f"{case} best joint incumbent failed replay: {final['error']}")
    if not math.isclose(
        float(final["makespan"]), float(best_objective), rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            f"{case} non-deterministic replay: search={best_objective}, "
            f"replay={final['makespan']}"
        )

    metadata = _metadata(case_path)
    case_sha = _case_sha256(case_path)
    expected = dict(task["expected"])
    search = {
        "search_contract_version": SEARCH_CONTRACT_VERSION,
        "algorithm": "real_coded_ga_sbx15_pm20_elitist_tournament2_anytime",
        "parallel_axis": "independent_cases",
        "population": population_size,
        "max_generations": int(task["max_generations"]),
        "completed_generations": generation,
        "partial_generation_evaluations": partial,
        "partial_generation_resolved_members": resolved_this_generation,
        "evaluated_candidates": evaluated,
        "completed_candidates": completed_candidates,
        "inherited_verified_candidates": inherited,
        "seed": seed,
        **layout.record(),
        "configured_additional_budget_seconds": float(task["time_budget"]),
        "nominal_cumulative_budget_seconds": float(task["cumulative_budget"]),
        "optimization_wall_seconds": float(optimization_wall),
        "budget_overshoot_seconds": max(
            0.0, float(optimization_wall - float(task["time_budget"]))
        ),
        "incumbent_preserved": True,
        "warm_start": warm_evidence,
        "candidate_history": history,
        "chromosome": best.tolist(),
    }
    teacher = {
        **expected,
        "case_id": metadata.get("case_id"),
        "profile": metadata.get("profile"),
        "distribution": metadata.get("distribution"),
        "dataset_dir": str(Path(task["dataset_dir"]).resolve()),
        "case_sha256": case_sha,
        "global_feature_mode": "f1f2",
        "max_plane_agents": int(task["max_plane_agents"]),
        "max_device_num": int(task["max_device_num"]),
        "search": search,
        **final,
        "completion_verified": True,
        "generated_unix_time": time.time(),
    }
    teacher_path = Path(task["teacher_path"])
    result_path = Path(task["result_path"])
    _atomic_json(teacher_path, teacher)
    _atomic_json(
        result_path,
        {
            **expected,
            "case_id": metadata.get("case_id"),
            "profile": metadata.get("profile"),
            "distribution": metadata.get("distribution"),
            "case_sha256": case_sha,
            "makespan": float(final["makespan"]),
            "completion": final["completion"],
            "aircraft_resource_wait": final["aircraft_resource_wait"],
            "search": {key: value for key, value in search.items() if key != "chromosome"},
            "teacher": str(teacher_path.resolve()),
        },
    )
    return {
        "case": case,
        "makespan": float(final["makespan"]),
        "evaluated_candidates": evaluated,
        "optimization_wall_seconds": optimization_wall,
        "teacher": str(teacher_path.resolve()),
    }


def _contract(
    case: str,
    case_sha: str,
    cumulative_budget: float,
    lookahead_margin: float,
    *,
    future_intent_horizon: int = 1,
    future_intent_mode: str = "legacy_one",
    frontier_max_requests: int = 2,
    request_capacity_per_plane: int = 0,
    release_aware_eta: bool = False,
    reservation_mode: str = "none",
    reservation_grace_seconds: float = 300.0,
    slack_forecast_seconds: float = 0.0,
) -> dict:
    return {
        "status": "completed",
        "schema_version": SCHEMA_VERSION,
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "search_contract_version": SEARCH_CONTRACT_VERSION,
        "resource_policy": "drl",
        "backends": BACKENDS,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "case": case,
        "case_sha256": case_sha,
        "nominal_cumulative_budget_seconds": float(cumulative_budget),
        **_planning_contract(
            lookahead_margin=lookahead_margin,
            future_intent_horizon=future_intent_horizon,
            future_intent_mode=future_intent_mode,
            frontier_max_requests=frontier_max_requests,
            request_capacity_per_plane=request_capacity_per_plane,
            release_aware_eta=release_aware_eta,
            reservation_mode=reservation_mode,
            reservation_grace_seconds=reservation_grace_seconds,
            slack_forecast_seconds=slack_forecast_seconds,
        ),
        "completion_verified": True,
    }


def _teacher_reusable(path: Path, expected: Mapping) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    return bool(
        payload.get("completed")
        and payload.get("completion", {}).get("completed")
        and payload.get("decision_trace")
        and payload.get("search", {}).get("chromosome")
    )


def _summary(output_dir: Path, expected_cases: list[str], args) -> dict:
    records = []
    missing = []
    for case in expected_cases:
        path = output_dir / "cases" / f"{case}.json"
        if not path.is_file():
            missing.append(case)
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    makespans = [float(record["makespan"]) for record in records]
    payload = {
        "status": "completed" if not missing else "partial",
        "schema_version": SCHEMA_VERSION,
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "expected_case_count": len(expected_cases),
        "completed_case_count": len(records),
        "missing_cases": missing,
        "nominal_cumulative_budget_seconds": float(args.cumulative_budget_seconds),
        "planning_contract": _planning_contract(
            lookahead_margin=args.device_lookahead_safety_margin,
            future_intent_horizon=args.device_future_intent_horizon,
            future_intent_mode=args.device_future_intent_mode,
            frontier_max_requests=args.device_frontier_max_requests,
            request_capacity_per_plane=args.device_request_capacity_per_plane,
            release_aware_eta=args.resource_release_aware_eta,
            reservation_mode=args.device_lookahead_reservation_mode,
            reservation_grace_seconds=args.device_reservation_grace_seconds,
            slack_forecast_seconds=args.resource_slack_forecast_seconds,
        ),
        "workers": int(args.workers),
        "makespan_mean": statistics.fmean(makespans) if makespans else None,
        "makespan_median": statistics.median(makespans) if makespans else None,
        "generated_unix_time": time.time(),
    }
    _atomic_json(output_dir / "summary.json", payload)
    return payload


def run(args) -> int:
    if getattr(args, "case_list_json", None):
        if args.max_cases or args.case_offset:
            raise ValueError("case-list-json cannot be combined with max-cases/case-offset")
        cases = json.loads(Path(args.case_list_json).read_text(encoding="utf-8"))
        if (not isinstance(cases, list) or not cases or len(set(cases)) != len(cases)
                or any(not isinstance(case, str) or Path(case).name != case
                       or not case.startswith("case_")
                       or not (args.dataset_dir / case).is_dir() for case in cases)):
            raise ValueError("case-list-json must name unique existing case directories")
    else:
        cases = list_case_folders(
            str(args.dataset_dir),
            args.max_cases,
            case_offset=args.case_offset,
        )
    output_dir = args.output_dir.resolve()
    output_dir.joinpath("teachers").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("cases").mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summary = _summary(output_dir, cases, args)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0 if summary["status"] == "completed" else 2

    pending = []
    reused = 0
    for case in cases:
        case_path = args.dataset_dir / case
        case_sha = _case_sha256(case_path)
        expected = _contract(
            case,
            case_sha,
            args.cumulative_budget_seconds,
            args.device_lookahead_safety_margin,
            future_intent_horizon=args.device_future_intent_horizon,
            future_intent_mode=args.device_future_intent_mode,
            frontier_max_requests=args.device_frontier_max_requests,
            request_capacity_per_plane=args.device_request_capacity_per_plane,
            release_aware_eta=args.resource_release_aware_eta,
            reservation_mode=args.device_lookahead_reservation_mode,
            reservation_grace_seconds=args.device_reservation_grace_seconds,
            slack_forecast_seconds=args.resource_slack_forecast_seconds,
        )
        teacher_path = output_dir / "teachers" / f"{case}.json"
        result_path = output_dir / "cases" / f"{case}.json"
        if _teacher_reusable(teacher_path, expected) and result_path.is_file():
            reused += 1
            continue
        pending.append(
            {
                "case_path": str(case_path.resolve()),
                "dataset_dir": str(args.dataset_dir.resolve()),
                "teacher_path": str(teacher_path),
                "result_path": str(result_path),
                "expected": expected,
                "population": args.population,
                "max_generations": args.max_generations,
                "time_budget": args.time_budget_seconds,
                "cumulative_budget": args.cumulative_budget_seconds,
                "warm_start_dir": str(args.warm_start_dir.resolve())
                if args.warm_start_dir
                else None,
                "warm_start_budget": args.warm_start_budget_seconds,
                "plane_warm_dir": str(args.plane_warm_dir.resolve())
                if args.plane_warm_dir
                else None,
                "resource_warm_dir": str(args.resource_warm_dir.resolve())
                if args.resource_warm_dir
                else None,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "max_plane_agents": args.max_plane_agents,
                "max_device_num": args.max_device_num,
                "lookahead_margin": args.device_lookahead_safety_margin,
                "future_intent_horizon": args.device_future_intent_horizon,
                "future_intent_mode": args.device_future_intent_mode,
                "frontier_max_requests": args.device_frontier_max_requests,
                "request_capacity_per_plane": args.device_request_capacity_per_plane,
                "release_aware_eta": args.resource_release_aware_eta,
                "reservation_mode": args.device_lookahead_reservation_mode,
                "reservation_grace_seconds": args.device_reservation_grace_seconds,
                "slack_forecast_seconds": args.resource_slack_forecast_seconds,
            }
        )
    print(
        f"[Stage3IGA] cases={len(cases)} reusable={reused} pending={len(pending)} "
        f"workers={args.workers} additional_budget={args.time_budget_seconds:.0f}s "
        f"cumulative_budget={args.cumulative_budget_seconds:.0f}s",
        flush=True,
    )
    if not pending:
        _summary(output_dir, cases, args)
        return 0

    completed = reused
    failures = []
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=min(args.workers, len(pending)),
        mp_context=__import__("multiprocessing").get_context("spawn"),
    )
    futures = {}
    try:
        iterator = iter(pending)
        while len(futures) < args.workers:
            try:
                task = next(iterator)
            except StopIteration:
                break
            futures[executor.submit(_search_case, task)] = task
        while futures and not _STOP_REQUESTED:
            done, _ = concurrent.futures.wait(
                futures, timeout=2.0, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                task = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "case": Path(task["case_path"]).name,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(
                        f"[Stage3IGA] FAILED {failures[-1]['case']}: {exc}", flush=True
                    )
                else:
                    completed += 1
                    print(
                        f"[Stage3IGA] {result['case']} cmax={result['makespan']:.1f} "
                        f"evals={result['evaluated_candidates']} "
                        f"search_wall={result['optimization_wall_seconds']:.1f}s "
                        f"({completed}/{len(cases)})",
                        flush=True,
                    )
                if not _STOP_REQUESTED:
                    try:
                        next_task = next(iterator)
                    except StopIteration:
                        pass
                    else:
                        futures[executor.submit(_search_case, next_task)] = next_task
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=not _STOP_REQUESTED, cancel_futures=True)

    summary = _summary(output_dir, cases, args)
    # Always refresh this file so a clean resumed run cannot leave stale
    # pre-emption diagnostics looking like current failures.
    _atomic_json(
        output_dir / "failures.json",
        {"failures": failures, "generated_unix_time": time.time()},
    )
    if _STOP_REQUESTED:
        return 143
    return 0 if not failures and summary["status"] == "completed" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plane-warm-dir", type=Path)
    parser.add_argument("--resource-warm-dir", type=Path)
    parser.add_argument("--warm-start-dir", type=Path)
    parser.add_argument("--warm-start-budget-seconds", type=float, default=180.0)
    parser.add_argument("--time-budget-seconds", type=float, default=180.0)
    parser.add_argument("--cumulative-budget-seconds", type=float, default=180.0)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--max-generations", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260821)
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
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-list-json", type=Path,
                        help="Explicit outcome-independent subset; no offset or truncation")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.population < 2 or args.workers < 1 or args.max_generations < 1:
        parser.error("population>=2, workers>=1 and max-generations>=1 are required")
    if args.time_budget_seconds <= 0 or args.cumulative_budget_seconds <= 0:
        parser.error("time budgets must be positive")
    if args.warm_start_dir and args.warm_start_budget_seconds <= 0:
        parser.error("nested warm-start budget must be positive")
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
    if not args.dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {args.dataset_dir}")
    return args


def main() -> int:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
