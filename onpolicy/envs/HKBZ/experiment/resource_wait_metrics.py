"""Aircraft waiting-time metrics attributable to mobile resources.

The generic plane ``waiting_time`` field is intentionally broader than the
metric required by the Stage-2 lookahead study.  This module keeps only waits
for a mobile resource needed by the selected job or by an aircraft transfer.
Departure pickup is recorded before the ZY-S plane action becomes legal, so it
is recovered separately from the resource trajectory.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _non_negative(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, number)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _phase(record: Mapping) -> str:
    explicit = record.get("action_phase")
    if explicit:
        return str(explicit)
    job_code = record.get("target_job_code")
    if job_code == "ZY-T":
        return "post_service_relocation"
    if job_code in {"ZY-S", "ZY-F"}:
        return "departure"
    return "service"


def _category(resource_types: Iterable[str], transporter_type: str) -> str:
    selected = {str(value) for value in resource_types if value}
    has_transporter = transporter_type in selected
    has_ordinary = bool(selected.difference({transporter_type}))
    if has_transporter and has_ordinary:
        return "mixed_transport_and_ordinary"
    if has_transporter:
        return "transporter"
    return "ordinary"


def summarize_aircraft_resource_wait(
    plane_trajectory: Sequence[Mapping],
    resource_trajectory: Sequence[Mapping],
    job_mobile_resource_types: Mapping[str, Sequence[str]],
    *,
    transporter_type: str = "R014",
    aircraft_count: int | None = None,
    include_events: bool = False,
    episode_cmax: float | None = None,
    criticality_scale_seconds: float = 600.0,
    criticality_min_weight: float = 0.0,
) -> dict:
    """Summarize non-overlapping aircraft waits caused by mobile resources.

    Plane-action waits cover service and transfer actions.  A transfer from
    the landing site ``Z`` does not require R014 in the current environment.
    Departure-pickup waits are measured from the first blocking pickup request
    until the reserved R014 arrives; they do not overlap a ZY-S action wait.
    """

    events: list[dict] = []
    resource_by_plane_job = defaultdict(list)
    for resource_record in resource_trajectory:
        plane_id = resource_record.get('plane_id')
        job_code = resource_record.get('job_code')
        if plane_id is None or job_code is None:
            continue
        resource_by_plane_job[(str(plane_id), str(job_code))].append(
            resource_record
        )
    aircraft_ids = {
        str(record.get("plane_id"))
        for record in plane_trajectory
        if record.get("plane_id") is not None
    }

    for record in plane_trajectory:
        plane_id = record.get("plane_id")
        if plane_id is None:
            continue
        plane_id = str(plane_id)
        job_code = record.get("target_job_code")
        resource_types = set(job_mobile_resource_types.get(str(job_code), ()))
        origin = record.get("origin_site_code")
        target = record.get("target_site_code")
        if (
            origin is not None
            and target is not None
            and str(origin) != str(target)
            and str(origin) != "Z"
        ):
            resource_types.add(transporter_type)
        if not resource_types:
            continue
        action_start = _non_negative(record.get("start_time"))
        wait_seconds = _non_negative(record.get("waiting_time"))
        matching_dispatches = [
            item
            for item in resource_by_plane_job.get(
                (plane_id, str(job_code)), ()
            )
            if (
                target is None
                or item.get('site_id') is None
                or str(item.get('site_id')) == str(target)
            )
        ]
        if matching_dispatches:
            selected_dispatch = min(
                matching_dispatches,
                key=lambda item: (
                    _non_negative(item.get('start_time'))
                    + _non_negative(item.get('trans_time')),
                    str(item.get('device_id', '')),
                ),
            )
            dispatch_time = _non_negative(
                selected_dispatch.get('start_time')
            )
            arrival_time = dispatch_time + _non_negative(
                selected_dispatch.get('trans_time')
            )
            before_dispatch = min(
                wait_seconds, max(0.0, dispatch_time - action_start)
            )
            travel_after_dispatch = min(
                max(0.0, wait_seconds - before_dispatch),
                max(0.0, arrival_time - max(action_start, dispatch_time)),
            )
        else:
            before_dispatch = wait_seconds
            travel_after_dispatch = 0.0
        post_arrival = max(
            0.0, wait_seconds - before_dispatch - travel_after_dispatch
        )
        events.append(
            {
                "source": "plane_action",
                "plane_id": plane_id,
                "job_code": job_code,
                "phase": _phase(record),
                "category": _category(resource_types, transporter_type),
                "resource_types": sorted(resource_types),
                "start_time": action_start,
                "end_time": _non_negative(record.get("end_time")),
                "wait_seconds": wait_seconds,
                "waiting_before_dispatch_seconds": before_dispatch,
                "travel_after_dispatch_seconds": travel_after_dispatch,
                "post_arrival_synchronization_seconds": post_arrival,
            }
        )

    departure_request_ready = {}
    for record in plane_trajectory:
        if record.get("target_job_code") != "ZY-T":
            continue
        plane_id = record.get("plane_id")
        if plane_id is None:
            continue
        departure_request_ready[str(plane_id)] = max(
            departure_request_ready.get(str(plane_id), 0.0),
            _non_negative(record.get("end_time")),
        )

    # The progressive-departure mask intentionally hides ZY-S until its R014
    # is present.  That aircraft wait therefore lives in the device record,
    # not in the later zero-wait ZY-S plane record.
    for record in resource_trajectory:
        if record.get("request_kind") != "departure_pickup":
            continue
        plane_id = record.get("plane_id")
        if plane_id is None:
            continue
        plane_id = str(plane_id)
        aircraft_ids.add(plane_id)
        travel_time = _non_negative(record.get("trans_time"))
        dispatch_time = _non_negative(record.get("start_time"))
        request_ready_time = departure_request_ready.get(plane_id)
        if request_ready_time is None:
            already_waited = _non_negative(
                record.get("waiting_time_at_dispatch")
            )
            request_ready_time = max(0.0, dispatch_time - already_waited)
        else:
            already_waited = max(0.0, dispatch_time - request_ready_time)
        arrival_time = dispatch_time + travel_time
        # A lookahead dispatcher may send R014 before the aircraft becomes
        # departure-ready.  Only the portion after request readiness blocks
        # the aircraft; counting the complete trip would turn successful
        # pre-positioning into artificial wait.
        wait_seconds = max(0.0, arrival_time - request_ready_time)
        travel_while_waiting = max(0.0, wait_seconds - already_waited)
        events.append(
            {
                "source": "departure_pickup",
                "plane_id": plane_id,
                "job_code": record.get("job_code"),
                "phase": "departure_pickup",
                "category": "departure_pickup",
                "resource_types": [transporter_type],
                "start_time": request_ready_time,
                "end_time": arrival_time,
                "wait_seconds": wait_seconds,
                "waiting_before_dispatch_seconds": already_waited,
                "travel_after_dispatch_seconds": travel_while_waiting,
                "post_arrival_synchronization_seconds": 0.0,
            }
        )

    if aircraft_count is None:
        aircraft_count = len(aircraft_ids)
    aircraft_count = max(int(aircraft_count), len(aircraft_ids))

    criticality_scale_seconds = max(
        1e-9, _non_negative(criticality_scale_seconds)
    )
    criticality_min_weight = min(
        1.0, _non_negative(criticality_min_weight)
    )
    plane_completion = defaultdict(float)
    for record in plane_trajectory:
        plane_id = record.get("plane_id")
        if plane_id is None:
            continue
        plane_completion[str(plane_id)] = max(
            plane_completion[str(plane_id)],
            _non_negative(record.get("end_time")),
        )
    observed_cmax = max(plane_completion.values(), default=0.0)
    if episode_cmax is None:
        episode_cmax = observed_cmax
    episode_cmax = max(observed_cmax, _non_negative(episode_cmax))

    per_aircraft = defaultdict(float)
    critical_wait_per_aircraft = defaultdict(float)
    critical_avoidable_per_aircraft = defaultdict(float)
    by_phase = defaultdict(lambda: {
        "opportunity_count": 0,
        "positive_wait_event_count": 0,
        "total_wait_seconds": 0.0,
    })
    by_category = defaultdict(lambda: {
        "opportunity_count": 0,
        "positive_wait_event_count": 0,
        "total_wait_seconds": 0.0,
    })
    positive_events = 0
    decomposition = {
        'waiting_before_dispatch_seconds': 0.0,
        'travel_after_dispatch_seconds': 0.0,
        'post_arrival_synchronization_seconds': 0.0,
    }
    rendezvous_spreads = []
    for event in events:
        wait = float(event["wait_seconds"])
        plane_id = event["plane_id"]
        per_aircraft[plane_id] += wait
        completion_time = plane_completion.get(plane_id, 0.0)
        completion_slack = max(0.0, episode_cmax - completion_time)
        critical_weight = (
            criticality_min_weight
            + (1.0 - criticality_min_weight)
            * math.exp(-completion_slack / criticality_scale_seconds)
        )
        conservative_avoidable = (
            float(event.get("waiting_before_dispatch_seconds", 0.0))
            + float(event.get(
                "post_arrival_synchronization_seconds", 0.0
            ))
        )
        event["plane_completion_time"] = float(completion_time)
        event["completion_slack_seconds"] = float(completion_slack)
        event["criticality_weight"] = float(critical_weight)
        event["critical_wait_seconds"] = float(critical_weight * wait)
        event["avoidable_lateness_seconds"] = float(
            conservative_avoidable
        )
        event["critical_avoidable_lateness_seconds"] = float(
            critical_weight * conservative_avoidable
        )
        critical_wait_per_aircraft[plane_id] += critical_weight * wait
        critical_avoidable_per_aircraft[plane_id] += (
            critical_weight * conservative_avoidable
        )
        rendezvous_spreads.append(float(event.get(
            "post_arrival_synchronization_seconds", 0.0
        )))
        if wait > 1e-9:
            positive_events += 1
        for field in decomposition:
            decomposition[field] += float(event.get(field, 0.0))
        for group, key in (
            (by_phase, event["phase"]),
            (by_category, event["category"]),
        ):
            group[key]["opportunity_count"] += 1
            group[key]["positive_wait_event_count"] += int(wait > 1e-9)
            group[key]["total_wait_seconds"] += wait

    waits = list(per_aircraft.values())
    critical_waits = list(critical_wait_per_aircraft.values())
    critical_avoidable_waits = list(
        critical_avoidable_per_aircraft.values()
    )
    if aircraft_count > len(waits):
        waits.extend([0.0] * (aircraft_count - len(waits)))
    if aircraft_count > len(critical_waits):
        critical_waits.extend(
            [0.0] * (aircraft_count - len(critical_waits))
        )
    if aircraft_count > len(critical_avoidable_waits):
        critical_avoidable_waits.extend(
            [0.0] * (aircraft_count - len(critical_avoidable_waits))
        )
    total_wait = float(sum(waits))
    positive_aircraft = sum(value > 1e-9 for value in waits)
    result = {
        "schema_version": 3,
        "unit": "seconds",
        "definition": (
            "time an aircraft is blocked by a required mobile resource; "
            "runway/site queues and voluntary staging are excluded"
        ),
        "aircraft_count": int(aircraft_count),
        "resource_wait_opportunity_count": int(len(events)),
        "positive_wait_event_count": int(positive_events),
        "zero_wait_event_count": int(len(events) - positive_events),
        "total_wait_seconds": total_wait,
        "mean_wait_seconds_per_aircraft": (
            total_wait / aircraft_count if aircraft_count else 0.0
        ),
        "median_wait_seconds_per_aircraft": (
            float(statistics.median(waits)) if waits else 0.0
        ),
        "p95_wait_seconds_per_aircraft": _quantile(waits, 0.95),
        "max_wait_seconds_per_aircraft": max(waits, default=0.0),
        "critical_wait_seconds": float(sum(critical_waits)),
        "critical_wait_p95_seconds_per_aircraft": _quantile(
            critical_waits, 0.95
        ),
        "critical_wait_max_seconds_per_aircraft": max(
            critical_waits, default=0.0
        ),
        "critical_avoidable_lateness_seconds": float(
            sum(critical_avoidable_waits)
        ),
        "critical_avoidable_lateness_p95_seconds_per_aircraft": _quantile(
            critical_avoidable_waits, 0.95
        ),
        "rendezvous_spread_seconds": float(sum(rendezvous_spreads)),
        "rendezvous_spread_p95_seconds": _quantile(
            rendezvous_spreads, 0.95
        ),
        "rendezvous_spread_max_seconds": max(
            rendezvous_spreads, default=0.0
        ),
        "criticality_scale_seconds": float(criticality_scale_seconds),
        "criticality_min_weight": float(criticality_min_weight),
        "episode_cmax": float(episode_cmax),
        "aircraft_with_positive_wait_count": int(positive_aircraft),
        "zero_wait_aircraft_count": int(aircraft_count - positive_aircraft),
        "zero_wait_aircraft_fraction": (
            float(aircraft_count - positive_aircraft) / aircraft_count
            if aircraft_count else 1.0
        ),
        "fully_eliminated": bool(total_wait <= 1e-9),
        "wait_decomposition": decomposition,
        "by_phase": dict(sorted(by_phase.items())),
        "by_category": dict(sorted(by_category.items())),
        "per_aircraft_wait_seconds": dict(sorted(per_aircraft.items())),
    }
    if include_events:
        result["events"] = events
    return result
