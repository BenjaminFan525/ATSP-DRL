"""Production decoder for Stage-2 mobile-resource IGA teachers.

The search program and the training environment must decode a chromosome with
exactly the same live legality masks.  Keeping the decoder in this dependency-
light module avoids importing pymoo (and the experiment CLI) in rollout
workers, while still giving both call sites one authoritative implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment


TRANSPORTER_RESOURCE_TYPE = "R014"


def resource_role(device: Any) -> str:
    return (
        "transporter"
        if device.resource.type == TRANSPORTER_RESOURCE_TYPE
        else "ordinary"
    )


@dataclass(frozen=True)
class ResourceGenomeLayout:
    """Stable, per-case chromosome layout for selected IGA resource roles."""

    selected_device_indices: tuple[int, ...]
    n_planes: int
    site_codes: tuple[str, ...]
    job_codes: tuple[str, ...]

    @classmethod
    def from_env(cls, env: Any, backends: Mapping[str, str]):
        selected = tuple(
            index
            for index, device in enumerate(
                env.device_list[: env.max_device_num]
            )
            if backends[resource_role(device)] == "iga"
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
        # Four global feature weights follow the device-specific static genes:
        # waiting age, travel cost, remaining work and same-site service.
        return len(self.selected_device_indices) * self.row_width + 4

    def decode(self, chromosome: np.ndarray) -> "DecodedResourceGenome":
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
        return DecodedResourceGenome(
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
class DecodedResourceGenome:
    row_by_device: Mapping[int, np.ndarray]
    feature_weights: np.ndarray
    site_index: Mapping[str, int]
    job_index: Mapping[str, int]
    n_planes: int
    n_sites: int

    def score(self, env: Any, device_idx: int, device: Any, request: Mapping) -> float:
        row = self.row_by_device[device_idx]
        plane_idx = int(request.get("plane_idx", -1))
        plane_gene = row[plane_idx] if 0 <= plane_idx < self.n_planes else 0.0
        site_idx = self.site_index.get(str(request.get("site_code", "")), -1)
        site_gene = row[self.n_planes + site_idx] if site_idx >= 0 else 0.0
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


def iga_preferences(
    env: Any,
    decoded: DecodedResourceGenome,
) -> tuple[dict[str, int], dict[tuple[str, int], float]]:
    """Return maximum-score one-to-one preferences for all IGA roles."""

    preferences: dict[str, int] = {}
    scores: dict[tuple[str, int], float] = {}
    selected = set(decoded.row_by_device)
    by_type: dict[str, list[tuple[int, Any]]] = {}
    for device_idx, device in enumerate(env.device_list[: env.max_device_num]):
        if device_idx in selected and env._device_is_dispatchable(device):
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
                    travel_seconds = distance / max(float(device.velocity), 1.0)
                    lead_time = max(0.0, float(request.get("lead_time", 0.0)))
                    if (
                        travel_seconds + env.device_lookahead_safety_margin
                        < lead_time
                    ):
                        continue
                value = decoded.score(env, device_idx, device, request)
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
    env: Any,
    backends: Mapping[str, str],
    decoded: DecodedResourceGenome | None,
    *,
    record: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Build one legal resource joint action from role-specific backends."""

    env._refresh_request_pool()
    hungarian_preferences = env._heuristic_device_assignment_preferences()
    iga_preference_map: dict[str, int] = {}
    if "iga" in backends.values():
        if decoded is None:
            raise ValueError("IGA resource actions require a decoded genome.")
        iga_preference_map, _ = iga_preferences(env, decoded)

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
        role = resource_role(device)
        backend = backends[role]
        preference_map = (
            iga_preference_map if backend == "iga" else hungarian_preferences
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
                chosen = env._nearest_request_for_device(device, valid_requests)
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
                selected_score = decoded.score(env, device_idx, device, chosen)
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


# Backwards-compatible names used by existing experiment scripts/tests.
GenomeLayout = ResourceGenomeLayout
DecodedGenome = DecodedResourceGenome
_role = resource_role
_iga_preferences = iga_preferences
