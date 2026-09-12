"""Dependency-light identifiers for HKBZ environment/checkpoint contracts."""

from __future__ import annotations

import hashlib
import json


ENVIRONMENT_SEMANTICS_VERSION = "progressive-departure-r014-pipeline-v2"
GLOBAL_FEATURE_MODES = frozenset({
    "none",
    "f1",
    "f1f2",
    "f1f2_departure",
})
GLOBAL_FEATURE_DIM = 24
GLOBAL_FEATURE_SCHEMA_VERSION = 1
GLOBAL_FEATURE_NORMALIZATION_VERSION = "hkbz-global-normalization-v1"
GLOBAL_FEATURE_F1_NAMES = (
    "active_plane_fraction",
    "future_plane_fraction",
    "completed_plane_fraction",
    "idle_active_plane_fraction",
    "remaining_operation_fraction",
    "mean_remaining_operation_fraction",
    "remaining_work_hours_per_plane",
    "future_work_hours_per_plane",
    "next_arrival_hours",
    "third_arrival_hours",
    "free_service_site_fraction",
    "interfered_service_site_fraction",
)
GLOBAL_FEATURE_RESOURCE_COMMON_NAMES = (
    "idle_device_fraction",
    "mean_device_unavailable_hours",
    "max_device_unavailable_hours",
    "request_pool_fraction",
    "mean_request_wait_hours",
    "max_request_wait_hours",
)
GLOBAL_FEATURE_F2_NAMES = (
    "mean_resource_demand_fraction",
    "max_resource_demand_fraction",
    "mean_resource_capacity_fraction",
    "min_capacity_demand_ratio",
    "mean_capacity_demand_ratio",
    "idle_r014_fraction",
)
GLOBAL_FEATURE_DEPARTURE_NAMES = (
    "departure_ready_plane_fraction",
    "mean_departure_ready_age_hours",
    "max_departure_ready_age_hours",
    "busy_runway_fraction",
    "unavailable_r014_fraction",
    "idle_unreserved_r014_fraction",
)


def global_feature_contract(mode: object) -> dict[str, object]:
    """Return the exact versioned meaning of all 24 global feature slots."""

    value = str(mode)
    if value not in GLOBAL_FEATURE_MODES:
        raise ValueError(f"Unsupported global_feature_mode={value!r}.")
    zero_names = tuple(
        f"constant_zero_{index:02d}" for index in range(GLOBAL_FEATURE_DIM)
    )
    if value == "none":
        feature_names = zero_names
    elif value == "f1":
        feature_names = GLOBAL_FEATURE_F1_NAMES + zero_names[12:]
    elif value == "f1f2":
        feature_names = (
            GLOBAL_FEATURE_F1_NAMES
            + GLOBAL_FEATURE_RESOURCE_COMMON_NAMES
            + GLOBAL_FEATURE_F2_NAMES
        )
    else:
        feature_names = (
            GLOBAL_FEATURE_F1_NAMES
            + GLOBAL_FEATURE_RESOURCE_COMMON_NAMES
            + GLOBAL_FEATURE_DEPARTURE_NAMES
        )
    if len(feature_names) != GLOBAL_FEATURE_DIM:
        raise RuntimeError(
            f"Global feature schema {value!r} has {len(feature_names)} slots, "
            f"expected {GLOBAL_FEATURE_DIM}."
        )
    payload: dict[str, object] = {
        "schema_version": GLOBAL_FEATURE_SCHEMA_VERSION,
        "mode": value,
        "dimension": GLOBAL_FEATURE_DIM,
        "normalization_version": GLOBAL_FEATURE_NORMALIZATION_VERSION,
        "feature_names": list(feature_names),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    payload["schema_id"] = (
        "hkbz-global-" + hashlib.sha256(encoded).hexdigest()[:20]
    )
    return payload


__all__ = [
    "ENVIRONMENT_SEMANTICS_VERSION",
    "GLOBAL_FEATURE_MODES",
    "GLOBAL_FEATURE_DIM",
    "GLOBAL_FEATURE_SCHEMA_VERSION",
    "GLOBAL_FEATURE_NORMALIZATION_VERSION",
    "GLOBAL_FEATURE_F1_NAMES",
    "GLOBAL_FEATURE_RESOURCE_COMMON_NAMES",
    "GLOBAL_FEATURE_F2_NAMES",
    "GLOBAL_FEATURE_DEPARTURE_NAMES",
    "global_feature_contract",
]
