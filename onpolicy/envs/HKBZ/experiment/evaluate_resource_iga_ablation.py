#!/usr/bin/env python3
"""Frozen-plane ablation for IGA versus Hungarian mobile-resource dispatch.

The historical HKBZ IGA chromosome controls only plane operation/site
priorities.  Mobile resources in those runs are always dispatched by the
Hungarian backend.  This experiment adds the missing, symmetric resource IGA:
for each case it evolves device-request priorities while replaying one frozen
Stage-1 M2 plane policy.  The final incumbent is then replayed once with a full
decision trace.

Three predeclared arms are supported:

* ``iga_all``: IGA dispatches ordinary devices and R014 transporters.
* ``iga_r014``: IGA dispatches R014; Hungarian dispatches ordinary devices.
* ``iga_ordinary``: IGA dispatches ordinary devices; Hungarian dispatches R014.

``hungarian_all`` is evaluated as an audit baseline and is never optimized.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import shlex
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import (  # noqa: E402
    GNN_MAPPOPolicy,
)
from onpolicy.config.config import get_config  # noqa: E402
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv  # noqa: E402
from onpolicy.envs.HKBZ.resource_teacher import (  # noqa: E402
    DecodedResourceGenome as ProductionDecodedGenome,
    ResourceGenomeLayout as ProductionGenomeLayout,
    iga_preferences as production_iga_preferences,
    mixed_resource_actions as production_mixed_resource_actions,
    resource_role as production_resource_role,
)
from onpolicy.envs.env_wrappers import GraphSubprocVecEnv  # noqa: E402
from onpolicy.envs.HKBZ.experiment.eval_common import (  # noqa: E402
    build_case_env_config,
    completion_details,
    list_case_folders,
    write_json,
)
from onpolicy.envs.HKBZ.experiment.resource_wait_metrics import (  # noqa: E402
    summarize_aircraft_resource_wait,
)
from onpolicy.utils.checkpoint_contract import (  # noqa: E402
    validate_stage1_checkpoint_contract,
)
from onpolicy.utils.training_stage import (  # noqa: E402
    protected_parameter_summary,
    validate_stage1_m2_checkpoint,
)


ARM_BACKENDS = {
    "iga_all": {"ordinary": "iga", "transporter": "iga"},
    "iga_r014": {"ordinary": "hungarian", "transporter": "iga"},
    "iga_ordinary": {"ordinary": "iga", "transporter": "hungarian"},
    "hungarian_all": {
        "ordinary": "hungarian",
        "transporter": "hungarian",
    },
}
PRIMARY_ARMS = ("iga_all", "iga_r014", "iga_ordinary")
TRANSPORTER_TYPE = AircraftScheduleEnv.TRANSPORTER_RESOURCE_TYPE

RESOURCE_LOOKAHEAD_CONTRACT_DEFAULTS = {
    "device_lookahead_dispatch": False,
    "device_lookahead_safety_margin": 60.0,
    "device_deadline_aware_dispatch": False,
    "device_future_intent_horizon": 0,
    "device_future_intent_mode": "legacy_one",
    "device_frontier_max_requests": 2,
    "resource_release_aware_eta": False,
    "device_lookahead_reservation_mode": "none",
    "device_reservation_grace_seconds": 300.0,
    "device_departure_lookahead": False,
}


def normalize_resource_lookahead_contract(
    contract: Mapping | None,
    *,
    device_lookahead_dispatch: bool = False,
    device_lookahead_safety_margin: float = 60.0,
) -> dict:
    """Return one complete, validated resource-planning contract.

    Legacy callers may omit ``contract`` and keep the historical two-field
    interface. Contract-aware callers must provide every field so a search,
    replay, or warm start can never silently inherit environment defaults.
    """

    if contract is None:
        values = dict(RESOURCE_LOOKAHEAD_CONTRACT_DEFAULTS)
        values.update(
            {
                "device_lookahead_dispatch": bool(
                    device_lookahead_dispatch
                ),
                "device_lookahead_safety_margin": float(
                    device_lookahead_safety_margin
                ),
            }
        )
    else:
        expected = set(RESOURCE_LOOKAHEAD_CONTRACT_DEFAULTS)
        observed = set(contract)
        if observed != expected:
            raise ValueError(
                "Resource lookahead contract must be complete; "
                f"missing={sorted(expected - observed)!r}, "
                f"unknown={sorted(observed - expected)!r}."
            )
        values = {
            "device_lookahead_dispatch": bool(
                contract["device_lookahead_dispatch"]
            ),
            "device_lookahead_safety_margin": float(
                contract["device_lookahead_safety_margin"]
            ),
            "device_deadline_aware_dispatch": bool(
                contract["device_deadline_aware_dispatch"]
            ),
            "device_future_intent_horizon": int(
                contract["device_future_intent_horizon"]
            ),
            "device_future_intent_mode": str(
                contract["device_future_intent_mode"]
            ),
            "device_frontier_max_requests": int(
                contract["device_frontier_max_requests"]
            ),
            "resource_release_aware_eta": bool(
                contract["resource_release_aware_eta"]
            ),
            "device_lookahead_reservation_mode": str(
                contract["device_lookahead_reservation_mode"]
            ),
            "device_reservation_grace_seconds": float(
                contract["device_reservation_grace_seconds"]
            ),
            "device_departure_lookahead": bool(
                contract["device_departure_lookahead"]
            ),
        }

    margin = values["device_lookahead_safety_margin"]
    grace = values["device_reservation_grace_seconds"]
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "device_lookahead_safety_margin must be finite and non-negative."
        )
    if not math.isfinite(grace) or grace < 0.0:
        raise ValueError(
            "device_reservation_grace_seconds must be finite and non-negative."
        )
    if values["device_future_intent_horizon"] not in {0, 1, 2, 3}:
        raise ValueError(
            "device_future_intent_horizon must be one of 0, 1, 2 or 3."
        )
    if values["device_future_intent_mode"] not in {
        "legacy_one",
        "bounded_frontier",
    }:
        raise ValueError(
            "device_future_intent_mode must be legacy_one or bounded_frontier."
        )
    if values["device_frontier_max_requests"] < 1:
        raise ValueError("device_frontier_max_requests must be positive.")
    if values["device_lookahead_reservation_mode"] not in {
        "none",
        "soft",
        "hard",
    }:
        raise ValueError(
            "device_lookahead_reservation_mode must be none, soft or hard."
        )
    return values


def _atomic_json(path: Path, payload: Mapping) -> None:
    write_json(str(path), dict(payload))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_digest(policy: GNN_MAPPOPolicy) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.ac.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_stage1_handoff(
    handoff_path: Path,
    checkpoint_path: Path,
    command_path: Path,
    command_key: str | None = None,
) -> dict:
    """Require the frozen policy to be an authoritative closed-Stage1 M2."""

    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    if payload.get("stage1_status") != "closed":
        raise ValueError(f"Stage1 handoff is not closed: {handoff_path}")
    checkpoint_path = checkpoint_path.resolve()
    command_path = command_path.resolve()
    for seed, record in payload.get("checkpoints", {}).items():
        expected_checkpoint = (ROOT / record["path"]).resolve()
        expected_command = (ROOT / record["source_command_path"]).resolve()
        if (
            checkpoint_path != expected_checkpoint
            or command_path != expected_command
        ):
            continue
        expected_command_key = record.get("source_command_key")
        if command_key is not None and command_key != expected_command_key:
            raise ValueError(
                f"Stage1 source-command key mismatch for seed {seed}: "
                f"{command_key!r} != {expected_command_key!r}"
            )
        actual_checkpoint_sha = _sha256_file(checkpoint_path)
        actual_command_sha = _sha256_file(command_path)
        if actual_checkpoint_sha != record.get("sha256"):
            raise ValueError(
                f"Stage1 checkpoint SHA256 mismatch for seed {seed}: "
                f"{actual_checkpoint_sha} != {record.get('sha256')}"
            )
        if actual_command_sha != record.get("source_command_sha256"):
            raise ValueError(
                f"Stage1 source-command SHA256 mismatch for seed {seed}: "
                f"{actual_command_sha} != {record.get('source_command_sha256')}"
            )
        return {
            "source_seed": int(seed),
            "winner": payload.get("winner"),
            "training_stage": payload.get("training_stage"),
            "semantic_contract": payload.get("semantic_contract", {}),
            "handoff_path": str(handoff_path.resolve()),
            "handoff_sha256": _sha256_file(handoff_path),
            "checkpoint_sha256": actual_checkpoint_sha,
            "source_command_sha256": actual_command_sha,
            "source_command_key": expected_command_key,
        }
    raise ValueError(
        "Checkpoint/source-command pair is not present in the authoritative "
        f"Stage1 handoff: {checkpoint_path}, {command_path}"
    )


def _role(device) -> str:
    return (
        "transporter"
        if device.resource.type == TRANSPORTER_TYPE
        else "ordinary"
    )


def _request_payload(request: Mapping) -> dict:
    payload = {
        "id": int(request.get("id", 0)),
        "plane_id": request.get("plane_id"),
        "plane_idx": int(request.get("plane_idx", -1)),
        "job_code": request.get("job_code"),
        "site_code": request.get("site_code"),
        "needed_res_types": list(request.get("needed_res_types", [])),
        "waiting_time": float(request.get("waiting_time", 0.0)),
        "is_lookahead": bool(request.get("is_lookahead", False)),
        "urgent": bool(request.get("urgent", True)),
        "request_kind": request.get("request_kind", "blocking"),
        "lead_time": float(request.get("lead_time", 0.0)),
    }
    # Label generation is a separate, replay-aware pass.  Preserve its exact
    # absolute target when present, but never synthesize it from ``lead_time``:
    # the latter is only the dependency-graph lower bound and was the source
    # of the old late-dispatch blind spot.
    if request.get("intrinsic_ready_time") is not None:
        payload["intrinsic_ready_time"] = float(
            request["intrinsic_ready_time"]
        )
    return payload


def _plane_only_graph(graph, n_plane_agents: int = 24):
    """Remove device-agent rows before the frozen Stage-1 plane forward.

    Device/resource nodes and their graph edges remain intact, so the shared
    encoder sees the authoritative resource state.  Only agent-indexed decoder
    tensors are sliced back to the exact Stage-1 width.  This avoids running
    80 untrained, externally overridden resource decoder slots on every step.
    """

    for name in (
        "pair_features",
        "agent_current_site_indices",
        "op_mask",
        "site_mask_matrix",
        "agent_job_site_mask_matrix",
        "request_mask_matrix",
        "agent_types",
    ):
        value = getattr(graph, name, None)
        if value is not None and value.shape[0] > n_plane_agents:
            setattr(graph, name, value[:n_plane_agents])
    return graph


@dataclass(frozen=True)
class GenomeLayout:
    """Stable per-case chromosome layout for selected IGA device roles."""

    selected_device_indices: tuple[int, ...]
    n_planes: int
    site_codes: tuple[str, ...]
    job_codes: tuple[str, ...]

    @classmethod
    def from_env(cls, env: AircraftScheduleEnv, backends: Mapping[str, str]):
        selected = tuple(
            index
            for index, device in enumerate(
                env.device_list[: env.max_device_num]
            )
            if backends[_role(device)] == "iga"
        )
        return cls(
            selected_device_indices=selected,
            n_planes=int(env.n_plane_agents),
            site_codes=tuple(env.site_code_list),
            job_codes=tuple(sorted(env.jobs)),
        )

    @property
    def row_width(self) -> int:
        return self.n_planes + len(self.site_codes) + len(self.job_codes)

    @property
    def n_var(self) -> int:
        # Four global, sign-constrained feature weights follow static genes:
        # waiting age, travel cost, remaining work and same-site service.
        return len(self.selected_device_indices) * self.row_width + 4

    def decode(self, chromosome: np.ndarray) -> "DecodedGenome":
        chromosome = np.asarray(chromosome, dtype=np.float64).reshape(-1)
        if chromosome.size != self.n_var:
            raise ValueError(
                f"Resource chromosome has {chromosome.size} variables; "
                f"expected {self.n_var}."
            )
        static_size = len(self.selected_device_indices) * self.row_width
        rows = chromosome[:static_size].reshape(
            len(self.selected_device_indices), self.row_width
        )
        return DecodedGenome(
            row_by_device={
                device_idx: rows[row_idx]
                for row_idx, device_idx in enumerate(
                    self.selected_device_indices
                )
            },
            feature_weights=chromosome[static_size:],
            site_index={code: index for index, code in enumerate(self.site_codes)},
            job_index={code: index for index, code in enumerate(self.job_codes)},
            n_planes=self.n_planes,
            n_sites=len(self.site_codes),
        )


@dataclass(frozen=True)
class DecodedGenome:
    row_by_device: Mapping[int, np.ndarray]
    feature_weights: np.ndarray
    site_index: Mapping[str, int]
    job_index: Mapping[str, int]
    n_planes: int
    n_sites: int

    def score(
        self,
        env: AircraftScheduleEnv,
        device_idx: int,
        device,
        request: Mapping,
    ) -> float:
        row = self.row_by_device[device_idx]
        plane_idx = int(request.get("plane_idx", -1))
        plane_gene = row[plane_idx] if 0 <= plane_idx < self.n_planes else 0.0
        site_idx = self.site_index.get(str(request.get("site_code", "")), -1)
        site_gene = (
            row[self.n_planes + site_idx] if site_idx >= 0 else 0.0
        )
        job_idx = self.job_index.get(str(request.get("job_code", "")), -1)
        job_gene = (
            row[self.n_planes + self.n_sites + job_idx]
            if job_idx >= 0
            else 0.0
        )

        target = env.sites[str(request["site_code"])]
        distance = abs(float(device.site.pos[0]) - float(target.pos[0])) + abs(
            float(device.site.pos[1]) - float(target.pos[1])
        )
        travel_seconds = distance / max(float(device.velocity), 1.0)
        waiting = max(0.0, float(request.get("waiting_time", 0.0)))
        plane = env.planes.get(request.get("plane_id"))
        remaining = 0.0
        if plane is not None:
            remaining = (
                len(plane.left_jobs) + len(plane.current_jobs)
            ) / max(1.0, float(len(env.job_code_list)))

        wait_weight, travel_weight, remaining_weight, local_weight = (
            float(value) for value in self.feature_weights
        )
        score = (
            float(plane_gene + site_gene + job_gene)
            + 2.0 * wait_weight * min(waiting / 3600.0, 4.0)
            - 2.0 * travel_weight * min(travel_seconds / 600.0, 4.0)
            + remaining_weight * remaining
            + local_weight * float(device.site.code == request["site_code"])
        )
        # Stable tie breaking must not depend on dictionary traversal.
        score -= 1e-9 * int(request.get("id", 0))
        return float(score)


def _iga_preferences(
    env: AircraftScheduleEnv,
    decoded: DecodedGenome,
) -> tuple[dict[str, int], dict[tuple[str, int], float]]:
    """Return maximum-score one-to-one preferences for all IGA roles."""

    preferences: dict[str, int] = {}
    scores: dict[tuple[str, int], float] = {}
    selected = set(decoded.row_by_device)
    by_type: dict[str, list[tuple[int, object]]] = {}
    for device_idx, device in enumerate(env.device_list[: env.max_device_num]):
        if (
            device_idx in selected
            and env._device_is_dispatchable(device)
        ):
            by_type.setdefault(device.resource.type, []).append(
                (device_idx, device)
            )

    for resource_type in sorted(by_type):
        devices = by_type[resource_type]
        requests = [
            request
            for request in env.request_list[1:]
            if resource_type in request.get("needed_res_types", [])
        ]
        if not requests:
            continue
        infeasible = -1e12
        matrix = np.full(
            (len(devices), len(requests)), infeasible, dtype=np.float64
        )
        for row_idx, (device_idx, device) in enumerate(devices):
            for col_idx, request in enumerate(requests):
                if request.get("is_lookahead", False):
                    target = env.sites[str(request["site_code"])]
                    distance = (
                        abs(float(device.site.pos[0]) - float(target.pos[0]))
                        + abs(float(device.site.pos[1]) - float(target.pos[1]))
                    )
                    travel_seconds = distance / max(
                        float(device.velocity), 1.0
                    )
                    lead_time = max(
                        0.0, float(request.get("lead_time", 0.0))
                    )
                    if (
                        travel_seconds
                        + env.device_lookahead_safety_margin
                        < lead_time
                    ):
                        continue
                value = decoded.score(
                    env, device_idx, device, request
                )
                matrix[row_idx, col_idx] = value
                scores[(device.code, int(request["id"]))] = value
        row_indices, col_indices = linear_sum_assignment(matrix, maximize=True)
        for row_idx, col_idx in zip(row_indices, col_indices):
            if matrix[int(row_idx), int(col_idx)] <= infeasible:
                continue
            preferences[devices[int(row_idx)][1].code] = int(
                requests[int(col_idx)]["id"]
            )
    return preferences, scores


def mixed_resource_actions(
    env: AircraftScheduleEnv,
    backends: Mapping[str, str],
    decoded: DecodedGenome | None,
    *,
    record: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Build one legal resource joint action from role-specific backends."""

    env._refresh_request_pool()
    hungarian_preferences = env._heuristic_device_assignment_preferences()
    iga_preferences: dict[str, int] = {}
    iga_scores: dict[tuple[str, int], float] = {}
    if "iga" in backends.values():
        if decoded is None:
            raise ValueError("IGA resource actions require a decoded genome.")
        iga_preferences, iga_scores = _iga_preferences(env, decoded)

    actions = np.zeros((env.n_agents, 2), dtype=np.int64)
    claimed_requests: set[int] = set()
    decisions: list[dict] = []
    for device_idx, device in enumerate(env.device_list[: env.max_device_num]):
        agent_id = env.n_plane_agents + device_idx
        if agent_id >= env.n_agents or not env._device_is_dispatchable(device):
            continue
        initial_requests = [
            request
            for request in env.request_list[1:]
            if env._device_can_dispatch(device, request)
        ]
        if not initial_requests:
            continue
        valid_requests, allow_noop = env._sequential_device_options(
            device_idx, claimed_requests
        )
        role = _role(device)
        backend = backends[role]
        preference_map = (
            iga_preferences if backend == "iga" else hungarian_preferences
        )
        preferred_id = preference_map.get(device.code)
        chosen = next(
            (
                request
                for request in valid_requests
                if int(request["id"]) == preferred_id
            ),
            None,
        )
        reason = "preferred"
        if chosen is None and not allow_noop and valid_requests:
            if backend == "iga":
                chosen = max(
                    valid_requests,
                    key=lambda request: decoded.score(
                        env, device_idx, device, request
                    ),
                )
            else:
                chosen = env._nearest_request_for_device(
                    device, valid_requests
                )
            reason = "last_chance_fallback"
        elif chosen is None:
            reason = "deferred_noop"

        selected_id = 0
        selected_score = None
        if chosen is not None:
            selected_id = int(chosen["id"])
            actions[agent_id] = [selected_id, 0]
            claimed_requests.add(selected_id)
            if backend == "iga":
                selected_score = decoded.score(
                    env, device_idx, device, chosen
                )
        if record:
            decisions.append(
                {
                    "agent_id": int(agent_id),
                    "device_id": device.code,
                    "device_type": device.resource.type,
                    "role": role,
                    "backend": backend,
                    "from_site": device.site.code,
                    "candidate_request_ids": [
                        int(request["id"]) for request in initial_requests
                    ],
                    "legal_request_ids": [
                        int(request["id"]) for request in valid_requests
                    ],
                    "allow_noop": bool(allow_noop),
                    "preferred_request_id": (
                        int(preferred_id) if preferred_id is not None else None
                    ),
                    "selected_request_id": selected_id,
                    "selected_score": selected_score,
                    "reason": reason,
                }
            )
    return actions, decisions


# Search and training deliberately execute the same production decoder.  The
# definitions above remain temporarily as source-compatible documentation for
# old serialized experiment imports, but no live path below uses them.
GenomeLayout = ProductionGenomeLayout
DecodedGenome = ProductionDecodedGenome
_iga_preferences = production_iga_preferences
mixed_resource_actions = production_mixed_resource_actions
_role = production_resource_role


class ResourceAblationEnv(AircraftScheduleEnv):
    """Subprocess-friendly environment with configured hybrid dispatch.

    Terminal workers return their last transition unchanged while slower
    population members finish.  This makes a heterogeneous candidate
    population safe to run in one fixed vector-environment pool.
    """

    def __init__(self, config):
        super().__init__(config)
        self._ablation_backends = dict(ARM_BACKENDS["hungarian_all"])
        self._ablation_decoded = None
        self._ablation_record = False
        self._ablation_trace = []
        self._ablation_pending_event = None
        self._ablation_terminal_payload = None
        self._ablation_step_count = 0

    def configure_resource_ablation(
        self,
        backends,
        layout,
        chromosome,
        record_trace=False,
    ):
        self._ablation_backends = dict(backends)
        self._ablation_decoded = (
            layout.decode(np.asarray(chromosome, dtype=np.float64))
            if layout is not None and chromosome is not None
            else None
        )
        self._ablation_record = bool(record_trace)
        return True

    def reset(self, *args, **kwargs):
        self._ablation_trace = []
        self._ablation_pending_event = None
        self._ablation_terminal_payload = None
        self._ablation_step_count = 0
        return super().reset(*args, **kwargs)

    def ablation_resource_actions(self):
        actions, decisions = mixed_resource_actions(
            self,
            self._ablation_backends,
            self._ablation_decoded,
            record=self._ablation_record,
        )
        if self._ablation_record:
            self._ablation_pending_event = {
                "step": int(self._ablation_step_count),
                "env_step": int(self.steps),
                "time_before": float(self.total_time),
                "requests": [
                    _request_payload(request) for request in self.request_list[1:]
                ],
                "resource_decisions": decisions,
            }
        return actions

    def _plane_trace(self, action):
        records = []
        active_pids = sorted(
            int(plane.code.split("_")[-1])
            for plane in self.planes.values()
            if plane.is_idle() and not plane.is_completed_all_jobs()
        )
        for pid in active_pids:
            op_idx = int(action[pid][0])
            site_idx = int(action[pid][1])
            job_idx = op_idx - int(pid) * len(self.job_code_list)
            records.append(
                {
                    "agent_id": int(pid),
                    "plane_id": f"Plane_0_{int(pid)}",
                    "operation_index": op_idx,
                    "job_code": (
                        self.job_code_list[job_idx]
                        if 0 <= job_idx < len(self.job_code_list)
                        else None
                    ),
                    "site_index": site_idx,
                    "site_code": (
                        self.site_code_list[site_idx]
                        if 0 <= site_idx < len(self.site_code_list)
                        else None
                    ),
                }
            )
        return records

    def step(self, action):
        if self._ablation_terminal_payload is not None:
            return self._ablation_terminal_payload
        event = self._ablation_pending_event
        if event is not None:
            event["plane_decisions"] = self._plane_trace(action)
        payload = super().step(action)
        obs, rewards, dones, info = payload
        if event is not None:
            event["time_after"] = float(self.total_time)
            event["elapsed"] = float(
                event["time_after"] - event["time_before"]
            )
            event["reward_sum"] = float(np.asarray(rewards).sum())
            self._ablation_trace.append(event)
        self._ablation_pending_event = None
        self._ablation_step_count += 1
        if np.all(dones):
            self._ablation_terminal_payload = payload
        return payload

    def ablation_result(self):
        details = completion_details(
            self, self._ablation_step_count, max(1, self._ablation_step_count)
        )
        result = {
            "completed": bool(details["completed"]),
            "makespan": float(self.total_time),
            "error": None,
            "completion": details,
            "aircraft_resource_wait": summarize_aircraft_resource_wait(
                self.trajectory_log,
                self.device_trajectory_log,
                {
                    code: self._needed_mobile_types(code)
                    for code in self.job_code_list
                },
                transporter_type=self.TRANSPORTER_RESOURCE_TYPE,
                aircraft_count=len(self.flights_data),
                include_events=self._ablation_record,
            ),
        }
        if self._ablation_record:
            trace, ready_labels, ready_coverage = (
                self.attach_intrinsic_ready_time_labels(
                    self._ablation_trace
                )
            )
            result.update(
                {
                    "decision_trace": trace,
                    "intrinsic_ready_time_label_schema_version": (
                        self.INTRINSIC_READY_TIME_LABEL_SCHEMA_VERSION
                    ),
                    "intrinsic_ready_time_semantics": (
                        self.INTRINSIC_READY_TIME_SEMANTICS
                    ),
                    "intrinsic_ready_time_labels": ready_labels,
                    "intrinsic_ready_time_label_coverage": ready_coverage,
                    "plane_trajectory": copy.deepcopy(self.trajectory_log),
                    "resource_trajectory": copy.deepcopy(
                        self.device_trajectory_log
                    ),
                    "resource_decision_log": copy.deepcopy(
                        self.device_decision_log
                    ),
                }
            )
        return result


class BatchedCaseEvaluator:
    """Persistent CPU environment pool for one case and many GA candidates."""

    def __init__(
        self,
        frozen_evaluator: "FrozenPlaneEvaluator",
        case_path: Path,
        pool_size: int,
    ):
        self.frozen_evaluator = frozen_evaluator
        self.case_path = case_path
        self.pool_size = int(pool_size)
        config = frozen_evaluator._env_config(case_path)

        def make_env():
            return ResourceAblationEnv(copy.deepcopy(config))

        self.envs = GraphSubprocVecEnv(
            [make_env for _ in range(self.pool_size)],
            ipc_timeout_seconds=1800.0,
            async_graph_clone_workers=min(4, self.pool_size),
        )

    def close(self):
        self.envs.close()

    def run(self, specs: Sequence[Mapping]) -> list[dict]:
        if not specs or len(specs) > self.pool_size:
            raise ValueError(
                f"Batch has {len(specs)} specs for pool_size={self.pool_size}."
            )
        padded = list(specs)
        while len(padded) < self.pool_size:
            duplicate = dict(specs[-1])
            duplicate["record_trace"] = False
            padded.append(duplicate)
        self.envs.call_each(
            "configure_resource_ablation",
            [
                (
                    spec["backends"],
                    spec.get("layout"),
                    spec.get("chromosome"),
                    bool(spec.get("record_trace", False)),
                )
                for spec in padded
            ],
        )
        obs, dones, infos = self.envs.reset()
        policy = self.frozen_evaluator.policy
        policy_args = self.frozen_evaluator.policy_args
        env_n_agents = int(infos["active_agents"].shape[1])
        n_plane_agents = 24
        rnn_states = np.zeros(
            (
                self.pool_size,
                n_plane_agents,
                int(policy_args.recurrent_N),
                int(policy_args.hidden_size),
            ),
            dtype=np.float32,
        )
        done_flags = np.all(dones, axis=1)
        step_count = 0
        started = time.perf_counter()
        while not np.all(done_flags) and step_count < self.frozen_evaluator.max_steps:
            active = np.asarray(infos["active_agents"], dtype=bool).copy()
            active[:, n_plane_agents:] = False
            policy_rows = np.flatnonzero(active[:, :n_plane_agents].any(axis=1))
            actions = np.zeros(
                (self.pool_size, env_n_agents, 2), dtype=np.int64
            )
            if policy_rows.size:
                with torch.inference_mode():
                    plane_actions, next_rnn = policy.act(
                        graph_obs=[
                            _plane_only_graph(obs[int(index)], n_plane_agents)
                            for index in policy_rows
                        ],
                        rnn_states=rnn_states[policy_rows],
                        active_agents=active[
                            policy_rows, :n_plane_agents, None
                        ],
                        last_op_indices=np.asarray(
                            infos["last_op_indices"]
                        )[policy_rows, :n_plane_agents],
                        last_site_indices=np.asarray(
                            infos["last_site_indices"]
                        )[policy_rows, :n_plane_agents],
                        deterministic=True,
                        agent_types=np.asarray(infos["agent_types"])[
                            policy_rows, :n_plane_agents
                        ],
                    )
                actions[policy_rows, :n_plane_agents] = (
                    plane_actions.detach().cpu().numpy()
                )
                rnn_states[policy_rows] = next_rnn.detach().cpu().numpy()
            resource_actions = self.envs.call("ablation_resource_actions")
            for index, resource_action in enumerate(resource_actions):
                actions[index, n_plane_agents:, :2] = np.asarray(
                    resource_action
                )[n_plane_agents:, :2]
            obs, _, dones, infos = self.envs.step(actions)
            newly_done = np.all(dones, axis=1)
            rnn_states[newly_done] = 0.0
            done_flags |= newly_done
            step_count += 1
        results = self.envs.call("ablation_result")
        wall_seconds = float(time.perf_counter() - started)
        selected = []
        for result in results[: len(specs)]:
            result = dict(result)
            result["wall_seconds"] = wall_seconds
            if step_count >= self.frozen_evaluator.max_steps and not result["completed"]:
                result["error"] = "max_steps_exceeded"
                result["completion"]["max_steps"] = self.frozen_evaluator.max_steps
            selected.append(result)
        return selected


class CrossCaseBatchedEvaluator:
    """Evaluate one candidate for many *different* cases in one GPU batch.

    The Stage-1 IGA runner parallelizes independent cases, not candidates from
    one case.  Stage-2 still needs the frozen plane network at every simulator
    step, so this class keeps that case-level parallelism while collecting all
    currently active plane observations into one policy forward pass.  Each
    environment process owns exactly one case and one candidate at a time.

    A deadline may interrupt an unfinished candidate between environment
    steps.  Already completed rows remain valid, and the caller can retain its
    explicit anytime incumbent exactly as the Stage-1 optimizer does.
    """

    def __init__(
        self,
        frozen_evaluator: "FrozenPlaneEvaluator",
        case_paths: Sequence[Path],
        *,
        async_graph_clone_workers: int = 4,
    ):
        self.frozen_evaluator = frozen_evaluator
        self.case_paths = tuple(Path(path) for path in case_paths)
        if not self.case_paths:
            raise ValueError("Cross-case evaluator requires at least one case.")
        env_fns = []
        for case_path in self.case_paths:
            config = frozen_evaluator._env_config(case_path)

            def make_env(config=config):
                return ResourceAblationEnv(copy.deepcopy(config))

            env_fns.append(make_env)
        self.envs = GraphSubprocVecEnv(
            env_fns,
            ipc_timeout_seconds=1800.0,
            async_graph_clone_workers=min(
                int(async_graph_clone_workers), len(env_fns)
            ),
        )

    def close(self):
        self.envs.close()

    def run(
        self,
        specs: Sequence[Mapping],
        *,
        deadline: float | None = None,
        abort_at_deadline: bool = True,
    ) -> list[dict]:
        if len(specs) != len(self.case_paths):
            raise ValueError(
                f"Cross-case batch has {len(specs)} specs for "
                f"{len(self.case_paths)} cases."
            )
        self.envs.call_each(
            "configure_resource_ablation",
            [
                (
                    spec["backends"],
                    spec.get("layout"),
                    spec.get("chromosome"),
                    bool(spec.get("record_trace", False)),
                )
                for spec in specs
            ],
        )
        obs, dones, infos = self.envs.reset()
        policy = self.frozen_evaluator.policy
        policy_args = self.frozen_evaluator.policy_args
        pool_size = len(self.case_paths)
        env_n_agents = int(infos["active_agents"].shape[1])
        n_plane_agents = 24
        rnn_states = np.zeros(
            (
                pool_size,
                n_plane_agents,
                int(policy_args.recurrent_N),
                int(policy_args.hidden_size),
            ),
            dtype=np.float32,
        )
        done_flags = np.all(dones, axis=1)
        step_count = 0
        policy_batches = 0
        policy_rows = 0
        deadline_interrupted = False
        started = time.perf_counter()
        while (
            not np.all(done_flags)
            and step_count < self.frozen_evaluator.max_steps
        ):
            if (
                abort_at_deadline
                and deadline is not None
                and time.monotonic() >= float(deadline)
            ):
                deadline_interrupted = True
                break
            active = np.asarray(infos["active_agents"], dtype=bool).copy()
            active[done_flags, :] = False
            active[:, n_plane_agents:] = False
            active_rows = np.flatnonzero(
                active[:, :n_plane_agents].any(axis=1)
            )
            actions = np.zeros(
                (pool_size, env_n_agents, 2), dtype=np.int64
            )
            if active_rows.size:
                with torch.inference_mode():
                    plane_actions, next_rnn = policy.act(
                        graph_obs=[
                            _plane_only_graph(obs[int(index)], n_plane_agents)
                            for index in active_rows
                        ],
                        rnn_states=rnn_states[active_rows],
                        active_agents=active[
                            active_rows, :n_plane_agents, None
                        ],
                        last_op_indices=np.asarray(
                            infos["last_op_indices"]
                        )[active_rows, :n_plane_agents],
                        last_site_indices=np.asarray(
                            infos["last_site_indices"]
                        )[active_rows, :n_plane_agents],
                        deterministic=True,
                        agent_types=np.asarray(infos["agent_types"])[
                            active_rows, :n_plane_agents
                        ],
                    )
                actions[active_rows, :n_plane_agents] = (
                    plane_actions.detach().cpu().numpy()
                )
                rnn_states[active_rows] = next_rnn.detach().cpu().numpy()
                policy_batches += 1
                policy_rows += int(active_rows.size)
            resource_actions = self.envs.call("ablation_resource_actions")
            for index, resource_action in enumerate(resource_actions):
                if not done_flags[index]:
                    actions[index, n_plane_agents:, :2] = np.asarray(
                        resource_action
                    )[n_plane_agents:, :2]
            obs, _, dones, infos = self.envs.step(actions)
            newly_done = np.all(dones, axis=1)
            rnn_states[newly_done] = 0.0
            done_flags |= newly_done
            step_count += 1

        results = self.envs.call("ablation_result")
        wall_seconds = float(time.perf_counter() - started)
        selected = []
        for index, result in enumerate(results):
            result = dict(result)
            result["wall_seconds"] = wall_seconds
            result["batch_step_count"] = int(step_count)
            result["batch_policy_forwards"] = int(policy_batches)
            result["batch_policy_rows"] = int(policy_rows)
            result["deadline_interrupted"] = bool(
                deadline_interrupted and not result["completed"]
            )
            if result["deadline_interrupted"]:
                result["error"] = "search_deadline_exceeded"
            elif (
                step_count >= self.frozen_evaluator.max_steps
                and not result["completed"]
            ):
                result["error"] = "max_steps_exceeded"
                result["completion"]["max_steps"] = (
                    self.frozen_evaluator.max_steps
                )
            result["case"] = self.case_paths[index].name
            selected.append(result)
        return selected


def _sbx_children(
    rng: np.random.Generator,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    eta: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random(parent_a.size)
    beta = np.where(
        u <= 0.5,
        np.power(2.0 * u, 1.0 / (eta + 1.0)),
        np.power(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0)),
    )
    child_a = 0.5 * ((1.0 + beta) * parent_a + (1.0 - beta) * parent_b)
    child_b = 0.5 * ((1.0 - beta) * parent_a + (1.0 + beta) * parent_b)
    return np.clip(child_a, 0.0, 1.0), np.clip(child_b, 0.0, 1.0)


def _polynomial_mutation(
    rng: np.random.Generator,
    child: np.ndarray,
    eta: float = 20.0,
) -> np.ndarray:
    child = child.copy()
    mutation_mask = rng.random(child.size) < (1.0 / max(1, child.size))
    if not mutation_mask.any():
        mutation_mask[int(rng.integers(0, child.size))] = True
    values = child[mutation_mask]
    u = rng.random(values.size)
    delta = np.where(
        u < 0.5,
        np.power(2.0 * u, 1.0 / (eta + 1.0)) - 1.0,
        1.0 - np.power(2.0 * (1.0 - u), 1.0 / (eta + 1.0)),
    )
    child[mutation_mask] = np.clip(values + delta, 0.0, 1.0)
    return child


def _next_population(
    rng: np.random.Generator,
    population: np.ndarray,
    fitness: np.ndarray,
) -> np.ndarray:
    size = len(population)
    next_population = [population[int(np.argmin(fitness))].copy()]

    def tournament():
        left, right = rng.integers(0, size, size=2)
        return int(left if fitness[left] <= fitness[right] else right)

    while len(next_population) < size:
        parent_a = population[tournament()]
        parent_b = population[tournament()]
        child_a, child_b = _sbx_children(rng, parent_a, parent_b)
        next_population.append(_polynomial_mutation(rng, child_a))
        if len(next_population) < size:
            next_population.append(_polynomial_mutation(rng, child_b))
    return np.asarray(next_population)


def _common_initial_populations(
    layouts: Mapping[str, GenomeLayout],
    population: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Project one common super-population into all three causal arms.

    Candidate ``i`` has identical role and feature genes wherever those genes
    exist in two arms.  This common-random-number design removes avoidable
    first-generation search luck while retaining role-specific evolution after
    generation zero.
    """

    full = layouts["iga_all"]
    rng = np.random.default_rng(int(seed))
    full_values = rng.random((int(population), full.n_var))
    full_values[0, :] = 0.5
    full_values[0, -4:] = np.asarray([1.0, 1.0, 0.5, 1.0])
    full_row = {
        device_idx: row_idx
        for row_idx, device_idx in enumerate(full.selected_device_indices)
    }
    projected = {}
    for arm, layout in layouts.items():
        if (
            layout.row_width != full.row_width
            or layout.site_codes != full.site_codes
            or layout.job_codes != full.job_codes
        ):
            raise ValueError(f"Incompatible genome layout for {arm}.")
        blocks = [
            full_values[
                :,
                full_row[device_idx] * full.row_width :
                (full_row[device_idx] + 1) * full.row_width,
            ]
            for device_idx in layout.selected_device_indices
        ]
        values = np.concatenate([*blocks, full_values[:, -4:]], axis=1)
        if values.shape != (int(population), layout.n_var):
            raise RuntimeError(
                f"Projected population for {arm} has shape {values.shape}; "
                f"expected {(int(population), layout.n_var)}."
            )
        projected[arm] = values.copy()
    return projected


def optimize_all_arms_batched(
    evaluator: "FrozenPlaneEvaluator",
    case_path: Path,
    *,
    population: int,
    generations: int,
    base_seed: int,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Search all three arms concurrently with equal evaluation budgets."""

    layouts = {
        arm: evaluator.build_layout(case_path, ARM_BACKENDS[arm])
        for arm in PRIMARY_ARMS
    }
    rng_by_arm = {
        arm: np.random.default_rng(_derived_seed(base_seed, case_path.name, arm))
        for arm in PRIMARY_ARMS
    }
    initial_population_seed = _derived_seed(
        base_seed, case_path.name, "common_initial_population"
    )
    populations = _common_initial_populations(
        layouts, population, initial_population_seed
    )

    best_objective = {arm: math.inf for arm in PRIMARY_ARMS}
    best_chromosome = {arm: None for arm in PRIMARY_ARMS}
    evaluated = {arm: 0 for arm in PRIMARY_ARMS}
    generation_seconds = []
    batch = BatchedCaseEvaluator(evaluator, case_path, population * len(PRIMARY_ARMS))
    started = time.monotonic()
    try:
        for generation in range(generations):
            specs = []
            identities = []
            for arm in PRIMARY_ARMS:
                for index, chromosome in enumerate(populations[arm]):
                    specs.append(
                        {
                            "backends": ARM_BACKENDS[arm],
                            "layout": layouts[arm],
                            "chromosome": chromosome,
                            "record_trace": False,
                        }
                    )
                    identities.append((arm, index))
            generation_started = time.monotonic()
            episodes = batch.run(specs)
            generation_seconds.append(time.monotonic() - generation_started)
            fitness = {
                arm: np.full(population, 1e9, dtype=np.float64)
                for arm in PRIMARY_ARMS
            }
            for (arm, index), episode in zip(identities, episodes):
                objective = (
                    float(episode["makespan"])
                    if episode["completed"]
                    else 1e8 + float(episode["makespan"])
                )
                fitness[arm][index] = objective
                evaluated[arm] += 1
                if objective < best_objective[arm]:
                    best_objective[arm] = objective
                    best_chromosome[arm] = populations[arm][index].copy()
            if generation + 1 < generations:
                for arm in PRIMARY_ARMS:
                    populations[arm] = _next_population(
                        rng_by_arm[arm], populations[arm], fitness[arm]
                    )

        final_specs = [
            {
                "backends": ARM_BACKENDS[arm],
                "layout": layouts[arm],
                "chromosome": best_chromosome[arm],
                "record_trace": True,
            }
            for arm in PRIMARY_ARMS
        ]
        final_specs.append(
            {
                "backends": ARM_BACKENDS["hungarian_all"],
                "layout": None,
                "chromosome": None,
                "record_trace": True,
            }
        )
        finals_list = batch.run(final_specs)
    finally:
        batch.close()

    finals = {
        arm: result
        for arm, result in zip((*PRIMARY_ARMS, "hungarian_all"), finals_list)
    }
    searches = {}
    total_seconds = time.monotonic() - started
    for arm in PRIMARY_ARMS:
        final = finals[arm]
        if not final["completed"]:
            raise RuntimeError(
                f"Best {arm}/{case_path.name} incumbent failed replay: "
                f"{final.get('error')}"
            )
        if not math.isclose(
            float(final["makespan"]),
            float(best_objective[arm]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                f"Non-deterministic incumbent replay for {arm}/{case_path.name}: "
                f"search={best_objective[arm]} replay={final['makespan']}"
            )
        searches[arm] = {
            "algorithm": "real_coded_ga_sbx15_pm20_elitist_tournament2",
            "population": int(population),
            "generations": int(generations),
            "seed": _derived_seed(base_seed, case_path.name, arm),
            "common_initial_population_seed": int(initial_population_seed),
            "n_var": int(layouts[arm].n_var),
            "iga_device_count": len(layouts[arm].selected_device_indices),
            "evaluated_candidates": int(evaluated[arm]),
            "generation_wall_seconds": [float(x) for x in generation_seconds],
            "joint_three_arm_wall_seconds": float(total_seconds),
            "chromosome": best_chromosome[arm].tolist(),
        }
    return searches, finals


def _resolve_source_command(
    command_path: Path,
    command_key: str | None = None,
) -> list[str]:
    payload = json.loads(command_path.read_text(encoding="utf-8"))
    command_payload = payload
    if command_key is not None:
        commands = payload.get("commands")
        if not isinstance(commands, Mapping) or command_key not in commands:
            raise ValueError(
                f"No command key {command_key!r} in {command_path}"
            )
        command_payload = commands[command_key]
        if not isinstance(command_payload, Mapping):
            raise ValueError(
                f"Command entry {command_key!r} is not a mapping in "
                f"{command_path}"
            )
    command = command_payload.get("command")
    if not isinstance(command, list):
        command = command_payload.get("argv")
    if not isinstance(command, list):
        shell_command = command_payload.get("shell_command")
        if not shell_command:
            shell_command = command_payload.get("shell")
        if not shell_command:
            suffix = f" key={command_key!r}" if command_key else ""
            raise ValueError(f"No command in {command_path}{suffix}")
        command = shlex.split(shell_command)
    if len(command) < 2:
        raise ValueError(f"Malformed source command in {command_path}")
    return [str(value) for value in command]


def _load_policy_args(
    command_path: Path,
    command_key: str | None = None,
):
    command = _resolve_source_command(command_path, command_key)
    parser = get_config()
    parser.add_argument("--scenario_name", default="simple")
    parser.add_argument("--ac_config", default=str(ROOT / "onpolicy/config/ac.yaml"))
    parser.add_argument("--env_config", default=str(ROOT / "onpolicy/config/env_plane_pretrain.yaml"))
    args = parser.parse_known_args(command[2:])[0]
    # The checkpoint was trained with heuristic resource execution, but the
    # resource ablation needs device-agent graph slots.  This flag changes no
    # parameter shape; it only enables the already-present role backends.
    args.max_agent_num = 24
    args.max_device_num = 80
    args.resource_policy = "drl"
    return args


def load_frozen_policy(
    checkpoint_path: Path,
    command_path: Path,
    device: torch.device,
    evaluation_tau: float,
    command_key: str | None = None,
) -> tuple[GNN_MAPPOPolicy, object, dict]:
    args = _load_policy_args(command_path, command_key)
    ac_config = yaml.safe_load(
        Path(args.ac_config).read_text(encoding="utf-8")
    )
    policy = GNN_MAPPOPolicy(args, ac_config, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    observation_contract = validate_stage1_checkpoint_contract(
        checkpoint,
        global_feature_mode=args.global_feature_mode,
        plane_order_mode=args.plane_order_mode,
        plane_pair_decoder=args.plane_pair_decoder,
        strict_metadata=True,
    )
    handoff_contract = validate_stage1_m2_checkpoint(
        checkpoint,
        policy.ac.state_dict(),
        plane_order_mode=args.plane_order_mode,
        plane_pair_decoder=args.plane_pair_decoder,
        global_feature_mode=args.global_feature_mode,
    )
    policy.load_model_state(checkpoint["model"])
    loaded_summary = protected_parameter_summary(policy.ac.state_dict())
    if loaded_summary != handoff_contract["source_summary"]:
        raise ValueError(
            "Frozen Stage1 protected tensors were not loaded bit-for-bit."
        )
    checkpoint["_resource_iga_validation"] = {
        "observation_contract": observation_contract,
        "handoff_contract": handoff_contract,
        "loaded_protected_summary": loaded_summary,
    }
    policy.ac.tau = float(evaluation_tau)
    policy.ac.eval()
    for parameter in policy.ac.parameters():
        parameter.requires_grad_(False)
    return policy, args, checkpoint


class FrozenPlaneEvaluator:
    def __init__(
        self,
        policy: GNN_MAPPOPolicy,
        policy_args,
        *,
        max_steps: int,
        device_lookahead_dispatch: bool = False,
        device_lookahead_safety_margin: float = 60.0,
        resource_lookahead_contract: Mapping | None = None,
    ):
        self.policy = policy
        self.policy_args = policy_args
        self.max_steps = int(max_steps)
        self.resource_lookahead_contract = normalize_resource_lookahead_contract(
            resource_lookahead_contract,
            device_lookahead_dispatch=device_lookahead_dispatch,
            device_lookahead_safety_margin=device_lookahead_safety_margin,
        )
        self.device_lookahead_dispatch = self.resource_lookahead_contract[
            "device_lookahead_dispatch"
        ]
        self.device_lookahead_safety_margin = self.resource_lookahead_contract[
            "device_lookahead_safety_margin"
        ]

    def _env_config(self, case_path: Path) -> dict:
        config = build_case_env_config(
            str(case_path),
            max_plane_agents=24,
            max_device_num=80,
            resource_policy="drl",
            seed=42,
            global_feature_mode="f1f2",
        )
        config.update(
            {
                "use_domain_rand": False,
                "plane_cycle_repeat_limit": 8,
                "plane_no_progress_limit": 120,
                "plane_relocation_limit": 40,
                "device_lookahead_dispatch": self.device_lookahead_dispatch,
                "device_lookahead_safety_margin": (
                    self.device_lookahead_safety_margin
                ),
                **self.resource_lookahead_contract,
            }
        )
        return config

    def build_layout(
        self, case_path: Path, backends: Mapping[str, str]
    ) -> GenomeLayout:
        env = AircraftScheduleEnv(self._env_config(case_path))
        try:
            return GenomeLayout.from_env(env, backends)
        finally:
            env.close()

    def run(
        self,
        case_path: Path,
        backends: Mapping[str, str],
        chromosome: np.ndarray | None,
        layout: GenomeLayout | None,
        *,
        record_trace: bool,
    ) -> dict:
        env = AircraftScheduleEnv(self._env_config(case_path))
        env.use_domain_rand = False
        trace: list[dict] = []
        error = None
        started = time.perf_counter()
        try:
            obs, dones, info = env.reset()
            decoded = (
                layout.decode(chromosome)
                if chromosome is not None and layout is not None
                else None
            )
            rnn_states = np.zeros(
                (
                    1,
                    env.n_agents,
                    int(self.policy_args.recurrent_N),
                    int(self.policy_args.hidden_size),
                ),
                dtype=np.float32,
            )
            step_count = 0
            while not np.all(dones) and step_count < self.max_steps:
                active = np.asarray(info["active_agents"], dtype=bool).copy()
                # Resource actions come exclusively from the experimental
                # backend; random, untrained M2 resource heads are never run.
                active[env.n_plane_agents :] = False
                actions = np.zeros((env.n_agents, 2), dtype=np.int64)
                if active[: env.n_plane_agents].any():
                    with torch.inference_mode():
                        plane_actions, next_rnn = self.policy.act(
                            graph_obs=[_plane_only_graph(obs, env.n_plane_agents)],
                            rnn_states=rnn_states[:, : env.n_plane_agents],
                            active_agents=active[: env.n_plane_agents].reshape(
                                1, -1, 1
                            ),
                            last_op_indices=np.asarray(
                                info["last_op_indices"]
                            )[: env.n_plane_agents].reshape(1, -1),
                            last_site_indices=np.asarray(
                                info["last_site_indices"]
                            )[: env.n_plane_agents].reshape(1, -1),
                            deterministic=True,
                            agent_types=np.asarray(info["agent_types"])[
                                : env.n_plane_agents
                            ].reshape(1, -1),
                        )
                    actions[: env.n_plane_agents] = (
                        plane_actions.detach().cpu().numpy()[0]
                    )
                    rnn_states[:, : env.n_plane_agents] = (
                        next_rnn.detach().cpu().numpy()
                    )
                resource_actions, resource_decisions = mixed_resource_actions(
                    env, backends, decoded, record=record_trace
                )
                actions[env.n_plane_agents :, :2] = resource_actions[
                    env.n_plane_agents :, :2
                ]
                event = None
                if record_trace:
                    plane_decisions = []
                    for pid in np.flatnonzero(active[: env.n_plane_agents]):
                        op_idx = int(actions[pid, 0])
                        site_idx = int(actions[pid, 1])
                        job_idx = op_idx - int(pid) * len(env.job_code_list)
                        plane_decisions.append(
                            {
                                "agent_id": int(pid),
                                "plane_id": f"Plane_0_{int(pid)}",
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
                    event = {
                        "step": int(step_count),
                        "env_step": int(env.steps),
                        "time_before": float(env.total_time),
                        "requests": [
                            _request_payload(request)
                            for request in env.request_list[1:]
                        ],
                        "plane_decisions": plane_decisions,
                        "resource_decisions": resource_decisions,
                    }
                obs, rewards, dones, info = env.step(actions)
                if event is not None:
                    event["time_after"] = float(env.total_time)
                    event["elapsed"] = float(
                        event["time_after"] - event["time_before"]
                    )
                    event["reward_sum"] = float(np.asarray(rewards).sum())
                    trace.append(event)
                step_count += 1
        except Exception as exc:  # candidate failures are finite GA penalties
            error = f"{type(exc).__name__}: {exc}"
            step_count = int(getattr(env, "steps", 0))
            dones = np.zeros(env.n_agents, dtype=bool)

        details = completion_details(env, step_count, self.max_steps)
        completed = bool(details["completed"] and error is None)
        result = {
            "completed": completed,
            "makespan": float(env.total_time),
            "wall_seconds": float(time.perf_counter() - started),
            "error": error,
            "completion": details,
        }
        if record_trace:
            trace, ready_labels, ready_coverage = (
                env.attach_intrinsic_ready_time_labels(trace)
            )
            result.update(
                {
                    "decision_trace": trace,
                    "intrinsic_ready_time_label_schema_version": (
                        env.INTRINSIC_READY_TIME_LABEL_SCHEMA_VERSION
                    ),
                    "intrinsic_ready_time_semantics": (
                        env.INTRINSIC_READY_TIME_SEMANTICS
                    ),
                    "intrinsic_ready_time_labels": ready_labels,
                    "intrinsic_ready_time_label_coverage": ready_coverage,
                    "plane_trajectory": copy.deepcopy(env.trajectory_log),
                    "resource_trajectory": copy.deepcopy(
                        env.device_trajectory_log
                    ),
                    "resource_decision_log": copy.deepcopy(
                        env.device_decision_log
                    ),
                }
            )
        env.close()
        return result


class ResourceIGAProblem(ElementwiseProblem):
    def __init__(
        self,
        evaluator: FrozenPlaneEvaluator,
        case_path: Path,
        backends: Mapping[str, str],
        layout: GenomeLayout,
        deadline: float | None,
    ):
        self.evaluator = evaluator
        self.case_path = case_path
        self.backends = dict(backends)
        self.layout = layout
        self.deadline = deadline
        self.evaluation_count = 0
        self.skipped_after_deadline = 0
        self.best_objective = math.inf
        self.best_chromosome: np.ndarray | None = None
        self.best_episode: dict | None = None
        super().__init__(
            n_var=layout.n_var,
            n_obj=1,
            n_ieq_constr=0,
            xl=np.zeros(layout.n_var),
            xu=np.ones(layout.n_var),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.skipped_after_deadline += 1
            out["F"] = [1e9]
            return
        episode = self.evaluator.run(
            self.case_path,
            self.backends,
            np.asarray(x, dtype=np.float64),
            self.layout,
            record_trace=False,
        )
        self.evaluation_count += 1
        objective = (
            float(episode["makespan"])
            if episode["completed"]
            else 1e8 + float(episode["makespan"])
        )
        out["F"] = [objective]
        if objective < self.best_objective:
            self.best_objective = objective
            self.best_chromosome = np.asarray(x, dtype=np.float64).copy()
            self.best_episode = episode


class DeadlineCallback(Callback):
    def __init__(self, deadline: float | None):
        super().__init__()
        self.deadline = deadline

    def notify(self, algorithm):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            algorithm.termination.force_termination = True


def optimize_arm(
    evaluator: FrozenPlaneEvaluator,
    case_path: Path,
    arm: str,
    *,
    population: int,
    generations: int,
    time_budget_seconds: float,
    seed: int,
) -> tuple[dict, dict]:
    backends = ARM_BACKENDS[arm]
    layout = evaluator.build_layout(case_path, backends)
    if not layout.selected_device_indices:
        raise RuntimeError(f"{arm} selects no IGA-controlled devices.")
    started = time.monotonic()
    deadline = (
        started + float(time_budget_seconds)
        if time_budget_seconds > 0.0
        else None
    )
    problem = ResourceIGAProblem(
        evaluator, case_path, backends, layout, deadline
    )
    rng = np.random.default_rng(int(seed))
    initial = rng.random((int(population), layout.n_var))
    # A distance/wait-aware incumbent prevents an unlucky all-random first
    # generation while retaining exactly the same population budget.
    initial[0, :] = 0.5
    initial[0, -4:] = np.asarray([1.0, 1.0, 0.5, 1.0])
    algorithm = GA(
        pop_size=int(population),
        sampling=initial,
        eliminate_duplicates=True,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
    )
    minimize(
        problem,
        algorithm,
        ("n_gen", int(generations)),
        callback=DeadlineCallback(deadline),
        seed=int(seed),
        verbose=False,
    )
    if problem.best_chromosome is None:
        raise RuntimeError(
            f"{arm}/{case_path.name} produced no evaluated IGA candidate."
        )
    final = evaluator.run(
        case_path,
        backends,
        problem.best_chromosome,
        layout,
        record_trace=True,
    )
    if not final["completed"]:
        raise RuntimeError(
            f"Best {arm}/{case_path.name} incumbent failed replay: "
            f"{final['error']}"
        )
    if not math.isclose(
        float(final["makespan"]),
        float(problem.best_episode["makespan"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"Non-deterministic incumbent replay for {arm}/{case_path.name}: "
            f"search={problem.best_episode['makespan']} replay={final['makespan']}"
        )
    search = {
        "population": int(population),
        "generations": int(generations),
        "time_budget_seconds": float(time_budget_seconds),
        "seed": int(seed),
        "n_var": int(layout.n_var),
        "iga_device_count": len(layout.selected_device_indices),
        "evaluated_candidates": int(problem.evaluation_count),
        "skipped_after_deadline": int(problem.skipped_after_deadline),
        "wall_seconds": float(time.monotonic() - started),
        "chromosome": problem.best_chromosome.tolist(),
    }
    return search, final


def _metadata(case_path: Path) -> dict:
    path = case_path / "metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _derived_seed(base_seed: int, case: str, arm: str) -> int:
    payload = f"{int(base_seed)}:{case}:{arm}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def run_shard(args) -> None:
    checkpoint_path = args.checkpoint.resolve()
    command_path = args.source_command.resolve()
    handoff = _validate_stage1_handoff(
        args.handoff.resolve(),
        checkpoint_path,
        command_path,
        args.source_command_key,
    )
    output_dir = args.output_dir.resolve()
    cases = list_case_folders(str(args.dataset_dir), args.max_cases)
    cases = [
        case
        for index, case in enumerate(cases)
        if index % args.shard_count == args.shard_index
    ]
    device = torch.device(args.device)
    policy, policy_args, checkpoint = load_frozen_policy(
        checkpoint_path,
        command_path,
        device,
        args.evaluation_tau,
        handoff.get("source_command_key"),
    )
    evaluator = FrozenPlaneEvaluator(
        policy,
        policy_args,
        max_steps=args.max_steps,
        device_lookahead_dispatch=args.device_lookahead_dispatch,
        device_lookahead_safety_margin=args.device_lookahead_safety_margin,
    )
    digest_before = _model_digest(policy)
    checkpoint_sha = _sha256_file(checkpoint_path)
    command_sha = _sha256_file(command_path)
    print(
        f"[ResourceIGA] shard={args.shard_index}/{args.shard_count} "
        f"device={device} cases={len(cases)} model={digest_before[:12]}",
        flush=True,
    )

    if args.time_budget_seconds > 0.0:
        raise ValueError(
            "The batched three-arm experiment uses an equal fixed candidate "
            "budget; --time-budget-seconds must be zero."
        )

    for local_index, case in enumerate(cases):
        case_path = args.dataset_dir / case
        case_output = output_dir / "cases" / case
        trajectory_dir = output_dir / "trajectories"
        expected_paths = [
            case_output / f"{arm}.json"
            for arm in (*PRIMARY_ARMS, "hungarian_all")
        ]
        if all(path.is_file() for path in expected_paths):
            print(
                f"[ResourceIGA] shard={args.shard_index} {case} already complete",
                flush=True,
            )
            continue

        searches, finals = optimize_all_arms_batched(
            evaluator,
            case_path,
            population=args.population,
            generations=args.generations,
            base_seed=args.seed,
        )
        metadata = _metadata(case_path)
        for arm in (*PRIMARY_ARMS, "hungarian_all"):
            final = finals[arm]
            if not final["completed"]:
                raise RuntimeError(
                    f"Final {arm} replay failed {case}: {final.get('error')}"
                )
            search = searches.get(arm)
            trajectory_path = trajectory_dir / arm / f"{case}.json"
            _atomic_json(
                trajectory_path,
                {
                    "schema_version": 1,
                    "case": case,
                    "arm": arm,
                    "backends": ARM_BACKENDS[arm],
                    "frozen_plane_checkpoint": str(checkpoint_path),
                    "frozen_plane_checkpoint_sha256": checkpoint_sha,
                    "frozen_plane_source_command": str(command_path),
                    "frozen_plane_source_command_sha256": command_sha,
                    "stage1_handoff": handoff,
                    "frozen_model_digest": digest_before,
                    "frozen_model_unchanged": True,
                    "plane_actions_deterministic": True,
                    "evaluation_tau": float(args.evaluation_tau),
                    "device_lookahead_dispatch": bool(
                        args.device_lookahead_dispatch
                    ),
                    "device_lookahead_safety_margin": float(
                        args.device_lookahead_safety_margin
                    ),
                    "case_sha256": metadata.get("case_sha256"),
                    "search": search,
                    **final,
                },
            )
            result_path = case_output / f"{arm}.json"
            _atomic_json(
                result_path,
                {
                    "status": "completed",
                    "case": case,
                    "case_id": metadata.get("case_id"),
                    "case_sha256": metadata.get("case_sha256"),
                    "profile": metadata.get("profile"),
                    "distribution": metadata.get("distribution"),
                    "arm": arm,
                    "backends": ARM_BACKENDS[arm],
                    "makespan": final["makespan"],
                    "completion": final["completion"],
                    "search": (
                        None
                        if search is None
                        else {
                            key: value
                            for key, value in search.items()
                            if key != "chromosome"
                        }
                    ),
                    "trajectory": str(trajectory_path),
                    "frozen_plane_checkpoint": str(checkpoint_path),
                    "frozen_plane_checkpoint_sha256": checkpoint_sha,
                    "frozen_plane_source_command_sha256": command_sha,
                    "stage1_source_seed": handoff["source_seed"],
                    "frozen_model_digest": digest_before,
                    "evaluation_tau": float(args.evaluation_tau),
                    "device_lookahead_dispatch": bool(
                        args.device_lookahead_dispatch
                    ),
                    "device_lookahead_safety_margin": float(
                        args.device_lookahead_safety_margin
                    ),
                },
            )
        cmax_text = " ".join(
            f"{arm}={finals[arm]['makespan']:.1f}"
            for arm in (*PRIMARY_ARMS, "hungarian_all")
        )
        print(
            f"[ResourceIGA] shard={args.shard_index} {case} {cmax_text} "
            f"evals_per_arm={args.population * args.generations}",
            flush=True,
        )

    digest_after = _model_digest(policy)
    if digest_after != digest_before:
        raise RuntimeError(
            "Frozen plane/model parameters changed during resource IGA search."
        )
    _atomic_json(
        output_dir / "workers" / f"shard_{args.shard_index:02d}.json",
        {
            "status": "completed",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "device": str(device),
            "cases": cases,
            "model_digest_before": digest_before,
            "model_digest_after": digest_after,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "source_command": str(command_path),
            "source_command_sha256": command_sha,
            "stage1_handoff": handoff,
            "evaluation_tau": float(args.evaluation_tau),
            "device_lookahead_dispatch": bool(
                args.device_lookahead_dispatch
            ),
            "device_lookahead_safety_margin": float(
                args.device_lookahead_safety_margin
            ),
            "checkpoint_episode": int(checkpoint.get("episodes", -1)),
            "completed_unix_time": time.time(),
        },
    )


def _bootstrap_mean_ci(values: Sequence[float], seed: int = 20260811):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(10000, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _paired_summary(
    records: Mapping[str, Mapping[str, Mapping]],
    cases: Sequence[str],
    left: str,
    right: str,
    *,
    seed: int,
) -> dict:
    deltas = np.asarray(
        [
            float(records[case][left]["makespan"])
            - float(records[case][right]["makespan"])
            for case in cases
        ],
        dtype=np.float64,
    )
    return {
        "case_count": int(deltas.size),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "bootstrap_mean_95ci": _bootstrap_mean_ci(deltas, seed=seed),
        "left_wins": int((deltas < 0).sum()),
        "ties": int((deltas == 0).sum()),
        "right_wins": int((deltas > 0).sum()),
    }


def summarize(args) -> dict:
    cases = list_case_folders(str(args.dataset_dir), args.max_cases)
    records: dict[str, dict[str, dict]] = {}
    missing = []
    for case in cases:
        records[case] = {}
        for arm in (*PRIMARY_ARMS, "hungarian_all"):
            path = args.output_dir / "cases" / case / f"{arm}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            records[case][arm] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise RuntimeError(
            f"Cannot summarize; {len(missing)} arm/case results are missing. "
            f"First missing: {missing[0]}"
        )

    arm_summaries = {}
    for arm in (*PRIMARY_ARMS, "hungarian_all"):
        values = [float(records[case][arm]["makespan"]) for case in cases]
        arm_summaries[arm] = {
            "case_count": len(values),
            "completed_count": sum(
                bool(records[case][arm]["completion"]["completed"])
                for case in cases
            ),
            "mean_makespan": statistics.mean(values),
            "std_makespan": statistics.pstdev(values),
            "median_makespan": statistics.median(values),
            "min_makespan": min(values),
            "max_makespan": max(values),
        }
        for group_key in ("distribution", "profile"):
            grouped = {}
            for case, value in zip(cases, values):
                group = str(records[case][arm].get(group_key) or "unknown")
                grouped.setdefault(group, []).append(value)
            arm_summaries[arm][f"by_{group_key}"] = {
                group: {"n": len(group_values), "mean": statistics.mean(group_values)}
                for group, group_values in sorted(grouped.items())
            }

    comparisons = {}
    pairs = [
        ("iga_all", "iga_r014"),
        ("iga_all", "iga_ordinary"),
        ("iga_r014", "iga_ordinary"),
        ("iga_all", "hungarian_all"),
        ("iga_r014", "hungarian_all"),
        ("iga_ordinary", "hungarian_all"),
    ]
    for left, right in pairs:
        name = f"{left}_minus_{right}"
        item = _paired_summary(
            records,
            cases,
            left,
            right,
            seed=_derived_seed(args.seed, name, "all"),
        )
        for group_key in ("distribution", "profile"):
            grouped = {}
            group_values = sorted(
                {
                    str(records[case][left].get(group_key) or "unknown")
                    for case in cases
                }
            )
            for group in group_values:
                group_cases = [
                    case
                    for case in cases
                    if str(records[case][left].get(group_key) or "unknown")
                    == group
                ]
                grouped[group] = _paired_summary(
                    records,
                    group_cases,
                    left,
                    right,
                    seed=_derived_seed(
                        args.seed, name, f"{group_key}:{group}"
                    ),
                )
            item[f"by_{group_key}"] = grouped
        comparisons[name] = item

    payload = {
        "status": "completed",
        "schema_version": 1,
        "objective": (
            "Frozen-M2 plane policy resource-dispatch ablation: IGA for all, "
            "IGA for R014 only, and IGA for ordinary resources only."
        ),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "case_count": len(cases),
        "arms": ARM_BACKENDS,
        "device_lookahead_dispatch": bool(args.device_lookahead_dispatch),
        "device_lookahead_safety_margin": float(
            args.device_lookahead_safety_margin
        ),
        "arm_summaries": arm_summaries,
        "paired_comparisons": comparisons,
        "cases": [
            {
                "case": case,
                "profile": records[case][PRIMARY_ARMS[0]].get("profile"),
                "distribution": records[case][PRIMARY_ARMS[0]].get(
                    "distribution"
                ),
                **{
                    arm: float(records[case][arm]["makespan"])
                    for arm in (*PRIMARY_ARMS, "hungarian_all")
                },
            }
            for case in cases
        ],
        "completed_unix_time": time.time(),
    }
    _atomic_json(args.output_dir / "summary.json", payload)

    lines = [
        "# Frozen-plane resource IGA ablation",
        "",
        "| Arm | Ordinary | R014 | Mean Cmax | Std | Median |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for arm in (*PRIMARY_ARMS, "hungarian_all"):
        item = arm_summaries[arm]
        lines.append(
            f"| {arm} | {ARM_BACKENDS[arm]['ordinary']} | "
            f"{ARM_BACKENDS[arm]['transporter']} | "
            f"{item['mean_makespan']:.2f} | {item['std_makespan']:.2f} | "
            f"{item['median_makespan']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Paired contrast | Mean delta | 95% bootstrap CI | W/T/L |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, item in comparisons.items():
        low, high = item["bootstrap_mean_95ci"]
        lines.append(
            f"| {name} | {item['mean_delta']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | "
            f"{item['left_wins']}/{item['ties']}/{item['right_wins']} |"
        )
    lines.extend(
        [
            "",
            "Every final per-case decision trace is stored under `trajectories/`.",
            "Negative paired deltas favor the left-hand arm.",
            "",
        ]
    )
    report_path = args.output_dir / "comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT
        / "onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo/"
        "stage1_next_round_20260808_r1_formal_M2_bc_kl_anneal_seed1/"
        "run1/models/checkpoint_Best.pt",
    )
    parser.add_argument(
        "--source-command",
        type=Path,
        default=ROOT
        / "result/hkbz_train_logs/stage1_next_round_20260808_r1/commands/"
        "formal_M2_bc_kl_anneal_seed1.json",
    )
    parser.add_argument("--source-command-key", default=None)
    parser.add_argument(
        "--handoff",
        type=Path,
        default=ROOT / "onpolicy/config/stage1_m2_handoff.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-tau", type=float, default=0.3)
    parser.add_argument(
        "--device-lookahead-dispatch", action="store_true", default=False
    )
    parser.add_argument(
        "--device-lookahead-safety-margin", type=float, default=60.0
    )
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--time-budget-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.population < 2 or args.generations < 1:
        parser.error("population must be >=2 and generations must be >=1")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    if args.device_lookahead_safety_margin < 0.0:
        parser.error("device-lookahead-safety-margin must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize_only:
        payload = summarize(args)
        print(json.dumps(payload["arm_summaries"], indent=2), flush=True)
        return 0
    # Environment workers are CPU-only. Spawn avoids inheriting an initialized
    # CUDA runtime from the frozen-policy parent process.
    mp.set_start_method("spawn", force=True)
    run_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
