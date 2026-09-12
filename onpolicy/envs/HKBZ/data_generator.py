"""Reproducible HKBZ benchmark generator inspired by open FJSP datasets.

The original generator varied geometry and release times but kept the effective
scheduling problem almost unchanged: every service cluster had complete fixed
resource coverage.  This module treats a case as a benchmark instance and
varies the dimensions that change the decision problem while preserving the
fixed tensor widths required by the current MARL implementation.

Design references are recorded in the generated manifest.  No external data is
copied; the FJSP benchmark families only motivate the stratification scheme:

* operation precedence and processing-time heterogeneity;
* sparse/mixed/rich eligible-resource alternatives;
* workload and release-time regimes;
* explicit train/validation/test splits and deterministic instance seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


GENERATOR_VERSION = "hkbz-fjsp-v3.0"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "dataset" / "fjsp_v2"
SERVICE_STANDS = 40
TAKEOFF_SITES = 3
MAX_PLANES = 24
MOBILE_DEVICE_BUDGET = 44
MAP_SIZE = 1200
MIN_SITE_DISTANCE = 50.0

FIXED_RESOURCE_TYPES = (
    "R001",
    "R002",
    "R003",
    "R005",
    "R006",
    "R007",
    "R008",
)
ORDINARY_MOBILE_TYPES = (
    "R001",
    "R002",
    "R003",
    "R005",
    "R006",
    "R007",
    "R008",
    "R011",
    "R012",
    "R013",
)
TRANSPORTER_TYPE = "R014"
ALL_MOBILE_TYPES = ORDINARY_MOBILE_TYPES + (TRANSPORTER_TYPE,)

REFERENCE_DESIGN = (
    {
        "name": "FJSPLib",
        "url": "https://scheduleopt.github.io/benchmarks/fjsplib",
        "used_for": (
            "instance-family stratification, general precedence graphs, "
            "eligible-machine alternatives, and difficulty-aware evaluation"
        ),
    },
    {
        "name": "Hurink sdata/edata/rdata/vdata families via FJSPLib",
        "url": "https://scheduleopt.github.io/benchmarks/fjsplib",
        "used_for": "sparse-to-rich flexibility classes",
    },
    {
        "name": "LOS online FJSP benchmark",
        "url": "https://www.dfki.de/web/forschung/projekte-publikationen/publikation/15372",
        "used_for": "release-time regimes and explicit transportation effects",
    },
    {
        "name": "MO-FJSPW benchmark data",
        "url": "https://data.mendeley.com/datasets/hpp82wtxfr/1",
        "used_for": "partial resource eligibility rather than universal coverage",
    },
    {
        "name": "FJSP with multiple AGVs replication data",
        "url": "https://doi.org/10.34810/DATA2426",
        "used_for": "transport-resource scarcity and spatial travel-time variation",
    },
)


BASE_JOB_DATA: list[dict[str, Any]] = [
    {
        "作业编号": "ZY-Z",
        "需要设备类型": [],
        "作业时间": 1,
        "前置作业": [],
        "互斥作业": [],
        "分组": "进场",
    },
    {
        "作业编号": "ZY-M",
        "需要设备类型": [],
        "作业时间": "根据距离计算",
        "前置作业": ["ZY-Z"],
        "互斥作业": [],
        "分组": "进场",
    },
    {
        "作业编号": "ZY01",
        "需要设备类型": [],
        "作业时间": 1,
        "前置作业": ["ZY-T", "ZY-M"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY02",
        "需要设备类型": ["R008"],
        "作业时间": 1,
        "前置作业": [],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY03",
        "需要设备类型": ["R002"],
        "作业时间": 1,
        "前置作业": [],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY04",
        "需要设备类型": ["R005", "R013"],
        "作业时间": 3,
        "前置作业": [],
        "互斥作业": ["ZY10"],
        "分组": "保障",
    },
    {
        "作业编号": "ZY05",
        "需要设备类型": ["R003", "R012"],
        "作业时间": 3,
        "前置作业": [],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY06",
        "需要设备类型": ["R007"],
        "作业时间": 4,
        "前置作业": [],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY07",
        "需要设备类型": ["R006", "R011"],
        "作业时间": 2,
        "前置作业": ["ZY02"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY08",
        "需要设备类型": [],
        "作业时间": 1,
        "前置作业": ["ZY02", "ZY03"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY09",
        "需要设备类型": ["R007"],
        "作业时间": 2,
        "前置作业": ["ZY06"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY10",
        "需要设备类型": ["R001"],
        "作业时间": "根据需要确定",
        "前置作业": ["ZY03"],
        "互斥作业": ["ZY04"],
        "分组": "保障",
    },
    {
        "作业编号": "ZY11",
        "需要设备类型": [],
        "作业时间": 1,
        "前置作业": ["ZY02", "ZY08"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY12",
        "需要设备类型": [],
        "作业时间": 20,
        "前置作业": ["ZY03", "ZY08"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY13",
        "需要设备类型": [],
        "作业时间": 1,
        "前置作业": ["ZY02", "ZY03"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY14",
        "需要设备类型": [],
        "作业时间": 2,
        "前置作业": [
            "ZY02",
            "ZY03",
            "ZY04",
            "ZY05",
            "ZY07",
            "ZY09",
            "ZY10",
            "ZY11",
            "ZY12",
            "ZY13",
        ],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY15",
        "需要设备类型": [],
        "作业时间": 5,
        "前置作业": ["ZY02", "ZY03", "ZY14"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY16",
        "需要设备类型": [],
        "作业时间": 1,
        "前置作业": ["ZY15"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY17",
        "需要设备类型": [],
        "作业时间": 1,
        "前置作业": ["ZY16"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY18",
        "需要设备类型": [],
        "作业时间": 3,
        "前置作业": ["ZY17"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY-L",
        "需要设备类型": ["R014"],
        "作业时间": 1,
        "前置作业": ["ZY-Z", "ZY-L"],
        "互斥作业": [],
        "分组": "保障",
    },
    {
        "作业编号": "ZY-T",
        "需要设备类型": ["R014"],
        "作业时间": "根据距离计算",
        "前置作业": ["ZY-Z", "ZY-L"],
        "互斥作业": [],
        "分组": "转移",
    },
    {
        "作业编号": "ZY-S",
        "需要设备类型": [],
        "作业时间": 3,
        "前置作业": ["ZY-L"],
        "互斥作业": [],
        "分组": "出场",
    },
    {
        "作业编号": "ZY-F",
        "需要设备类型": [],
        "作业时间": 1,
        "前置作业": ["ZY-S"],
        "互斥作业": [],
        "分组": "出场",
    },
]


PRECEDENCE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "parallel": {
        "ZY02": [],
        "ZY03": [],
        "ZY04": [],
        "ZY05": [],
        "ZY06": [],
        "ZY07": ["ZY02"],
        "ZY08": ["ZY02", "ZY03"],
        "ZY09": ["ZY06"],
        "ZY10": ["ZY03"],
        "ZY11": ["ZY08"],
        "ZY12": ["ZY08"],
        "ZY13": ["ZY02", "ZY03"],
        "ZY14": ["ZY04", "ZY05", "ZY07", "ZY09", "ZY10", "ZY11", "ZY12", "ZY13"],
        "ZY15": ["ZY14"],
        "ZY16": ["ZY15"],
        "ZY17": ["ZY16"],
        "ZY18": ["ZY17"],
    },
    "balanced": {
        item["作业编号"]: list(item["前置作业"])
        for item in BASE_JOB_DATA
        if item["分组"] == "保障" and item["作业编号"] not in {"ZY01", "ZY-L"}
    },
    "coupled": {
        "ZY02": [],
        "ZY03": [],
        "ZY04": ["ZY02"],
        "ZY05": ["ZY03"],
        "ZY06": ["ZY02"],
        "ZY07": ["ZY02", "ZY04"],
        "ZY08": ["ZY02", "ZY03"],
        "ZY09": ["ZY06"],
        "ZY10": ["ZY03", "ZY05"],
        "ZY11": ["ZY08"],
        "ZY12": ["ZY04", "ZY08"],
        "ZY13": ["ZY06", "ZY08"],
        "ZY14": ["ZY07", "ZY09", "ZY10", "ZY11", "ZY12", "ZY13"],
        "ZY15": ["ZY14"],
        "ZY16": ["ZY15"],
        "ZY17": ["ZY16"],
        "ZY18": ["ZY17"],
    },
    "inspection_first": {
        "ZY02": [],
        "ZY03": [],
        "ZY04": ["ZY02", "ZY03"],
        "ZY05": ["ZY02", "ZY03"],
        "ZY06": ["ZY02", "ZY03"],
        "ZY07": ["ZY02"],
        "ZY08": ["ZY02", "ZY03"],
        "ZY09": ["ZY06"],
        "ZY10": ["ZY03"],
        "ZY11": ["ZY04", "ZY08"],
        "ZY12": ["ZY05", "ZY08"],
        "ZY13": ["ZY06", "ZY08"],
        "ZY14": ["ZY07", "ZY09", "ZY10", "ZY11", "ZY12", "ZY13"],
        "ZY15": ["ZY14"],
        "ZY16": ["ZY15"],
        "ZY17": ["ZY16"],
        "ZY18": ["ZY17"],
    },
}


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    plane_range: tuple[int, int]
    cluster_choices: tuple[int, ...]
    fixed_coverage_range: tuple[float, float]
    flexibility_choices: tuple[str, ...]
    precedence_choices: tuple[str, ...]
    arrival_choices: tuple[str, ...]
    duration_scale_range: tuple[float, float]
    depot_bias_range: tuple[float, float]
    transporter_range: tuple[int, int]
    distribution: str = "iid"


PROFILES: dict[str, ScenarioProfile] = {
    "balanced": ScenarioProfile(
        "balanced", (20, 24), (6, 7, 8), (0.40, 0.58), ("edata", "rdata"),
        ("parallel", "balanced"), ("mixed",), (0.90, 1.20), (0.25, 0.50), (7, 9)
    ),
    "resource_sparse": ScenarioProfile(
        "resource_sparse", (22, 24), (6, 7, 8), (0.15, 0.32), ("sdata", "edata"),
        ("balanced", "coupled"), ("mixed", "bursty"), (1.00, 1.30), (0.45, 0.70), (6, 8)
    ),
    "bursty": ScenarioProfile(
        "bursty", (22, 24), (6, 7, 8), (0.35, 0.52), ("edata", "rdata"),
        ("parallel", "balanced"), ("bursty", "waves"), (0.90, 1.20), (0.30, 0.60), (7, 9)
    ),
    "high_flex": ScenarioProfile(
        "high_flex", (18, 24), (5, 6, 7), (0.55, 0.72), ("rdata", "vdata"),
        ("parallel", "balanced"), ("mixed",), (0.80, 1.15), (0.20, 0.45), (8, 10)
    ),
    "coupled": ScenarioProfile(
        "coupled", (20, 24), (6, 7, 8), (0.28, 0.50), ("edata", "rdata"),
        ("coupled", "inspection_first"), ("mixed", "waves"), (1.10, 1.40), (0.35, 0.60), (6, 8)
    ),
    "light": ScenarioProfile(
        "light", (16, 20), (5, 6), (0.55, 0.70), ("rdata", "vdata"),
        ("parallel",), ("steady", "mixed"), (0.75, 1.00), (0.15, 0.35), (8, 10)
    ),
    "stress_joint": ScenarioProfile(
        "stress_joint", (24, 24), (7, 8), (0.08, 0.20), ("sdata", "edata"),
        ("coupled", "inspection_first"), ("bursty", "waves"), (1.15, 1.45), (0.60, 0.80), (5, 7),
        distribution="ood_stress",
    ),
    "stress_arrival": ScenarioProfile(
        "stress_arrival", (24, 24), (6, 7, 8), (0.35, 0.50), ("rdata",),
        ("parallel", "balanced"), ("waves",), (0.95, 1.20), (0.40, 0.60), (6, 8),
        distribution="ood_stress",
    ),
    "resource_ood": ScenarioProfile(
        "resource_ood", (20, 24), (7, 8), (0.06, 0.16), ("sdata",),
        ("coupled", "inspection_first"), ("mixed",), (1.00, 1.35), (0.55, 0.75), (5, 7),
        distribution="ood_stress",
    ),
    "low_load_ood": ScenarioProfile(
        "low_load_ood", (14, 17), (5, 6), (0.62, 0.78), ("vdata",),
        ("parallel",), ("steady",), (0.70, 0.95), (0.10, 0.30), (8, 10),
        distribution="ood_scale",
    ),
}


SPLIT_PROFILE_WEIGHTS: dict[str, tuple[tuple[str, float], ...]] = {
    "train": (
        # 600-case production split: 480 IID + 120 OOD augmentation.
        ("balanced", 0.20),
        ("resource_sparse", 0.16),
        ("bursty", 0.16),
        ("high_flex", 0.12),
        ("coupled", 0.08),
        ("light", 0.08),
        ("stress_joint", 0.08),
        ("stress_arrival", 0.06),
        ("resource_ood", 0.04),
        ("low_load_ood", 0.02),
    ),
    "validation": (
        # 120 cases: 60 IID + 54 OOD-stress + 6 OOD-scale.
        ("balanced", 0.125),
        ("resource_sparse", 0.10),
        ("bursty", 0.10),
        ("high_flex", 0.075),
        ("coupled", 0.05),
        ("light", 0.05),
        ("stress_joint", 0.20),
        ("stress_arrival", 0.15),
        ("resource_ood", 0.10),
        ("low_load_ood", 0.05),
    ),
    "test": (
        ("balanced", 0.10),
        ("resource_sparse", 0.10),
        ("bursty", 0.10),
        ("high_flex", 0.10),
        ("coupled", 0.05),
        ("light", 0.05),
        ("stress_joint", 0.20),
        ("stress_arrival", 0.15),
        ("resource_ood", 0.10),
        ("low_load_ood", 0.05),
    ),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _derived_seed(base_seed: int, split: str, case_index: int, profile_name: str) -> int:
    payload = f"{GENERATOR_VERSION}:{base_seed}:{split}:{case_index}:{profile_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def _allocate_profile_schedule(split: str, count: int, base_seed: int) -> list[str]:
    weighted = SPLIT_PROFILE_WEIGHTS[split]
    raw = [(name, weight * count) for name, weight in weighted]
    allocated = {name: int(math.floor(value)) for name, value in raw}
    remaining = count - sum(allocated.values())
    remainders = sorted(
        ((value - math.floor(value), name) for name, value in raw),
        reverse=True,
    )
    for _, name in remainders[:remaining]:
        allocated[name] += 1

    schedule = [name for name, _ in weighted for _ in range(allocated[name])]
    random.Random(_derived_seed(base_seed, split, 0, "profile_schedule")).shuffle(schedule)
    return schedule


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _farthest_first_kmeans(
    positions: Sequence[Sequence[float]],
    k: int,
    rng: random.Random,
) -> list[list[list[float]]]:
    points = [list(point) for point in positions]
    centroids = [list(rng.choice(points))]
    while len(centroids) < k:
        candidate = max(
            points,
            key=lambda point: min(_distance(point, centroid) for centroid in centroids),
        )
        centroids.append(list(candidate))

    assignments: list[int] = [0] * len(points)
    for _ in range(25):
        next_assignments = [
            min(range(k), key=lambda idx: _distance(point, centroids[idx]))
            for point in points
        ]
        if next_assignments == assignments:
            break
        assignments = next_assignments
        for cluster_idx in range(k):
            members = [point for point, idx in zip(points, assignments) if idx == cluster_idx]
            if members:
                centroids[cluster_idx] = [
                    sum(point[axis] for point in members) / len(members)
                    for axis in (0, 1)
                ]

    clusters = [
        [point for point, idx in zip(points, assignments) if idx == cluster_idx]
        for cluster_idx in range(k)
    ]
    if any(not cluster for cluster in clusters):
        raise ValueError("K-means produced an empty service cluster")
    clusters.sort(
        key=lambda cluster: (
            sum(point[0] for point in cluster) / len(cluster),
            sum(point[1] for point in cluster) / len(cluster),
        )
    )
    for cluster in clusters:
        cluster.sort(key=lambda point: (point[1], point[0]))
    return clusters


def _service_jobs(jobs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        job
        for job in jobs
        if job["分组"] == "保障" and job["作业编号"] not in {"ZY01", "ZY-L"}
    ]


def _resource_jobs(jobs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [job for job in _service_jobs(jobs) if job["需要设备类型"]]


def _fixed_types_by_stand(
    fixed_resources: Sequence[Mapping[str, str]],
    num_stands: int = SERVICE_STANDS,
) -> dict[str, set[str]]:
    result = {str(index): set() for index in range(1, num_stands + 1)}
    for resource in fixed_resources:
        lower, upper = (int(value) for value in resource["支持停机位"].split("-"))
        for stand_idx in range(lower, upper + 1):
            result[str(stand_idx)].add(resource["类型"])
    return result


def _case_statistics(case: Mapping[str, Any]) -> dict[str, Any]:
    jobs = case["jobs"]
    flights = case["flights"]
    fixed_resources = case["fixed_resources"]
    mobile_resources = case["mobile_resources"]
    gaps = [
        flights[idx]["到达时间"] - flights[idx - 1]["到达时间"]
        for idx in range(1, len(flights))
    ]
    fixed_by_stand = _fixed_types_by_stand(fixed_resources)
    mobile_types = {resource["类型"] for resource in mobile_resources}
    ordinary_jobs = [
        job
        for job in _resource_jobs(jobs)
        if TRANSPORTER_TYPE not in set(job["需要设备类型"])
    ]
    stand_job_pairs = max(1, SERVICE_STANDS * len(ordinary_jobs))
    mobile_required_pairs = 0
    eligible_counts: list[int] = []
    mobile_counts = Counter(resource["类型"] for resource in mobile_resources)

    for fixed_types in fixed_by_stand.values():
        for job in ordinary_jobs:
            required = set(job["需要设备类型"])
            fixed_eligible = required.intersection(fixed_types)
            mobile_eligible = required.intersection(mobile_types)
            if not fixed_eligible and mobile_eligible:
                mobile_required_pairs += 1
            eligible_count = len(fixed_eligible) + sum(
                mobile_counts[resource_type] for resource_type in mobile_eligible
            )
            eligible_counts.append(eligible_count)

    fixed_coverage = sum(len(types) for types in fixed_by_stand.values()) / max(
        1, len(fixed_by_stand) * len(FIXED_RESOURCE_TYPES)
    )
    return {
        "num_planes": len(flights),
        "num_service_stands": SERVICE_STANDS,
        "num_fixed_resources": len(fixed_resources),
        "num_mobile_resources": len(mobile_resources),
        "num_service_jobs": len(_service_jobs(jobs)),
        "num_resource_jobs": len(_resource_jobs(jobs)),
        "arrival_horizon": flights[-1]["到达时间"] if flights else 0,
        "arrival_gap_mean": round(statistics.fmean(gaps), 6) if gaps else 0.0,
        "arrival_gap_min": min(gaps) if gaps else 0,
        "arrival_gap_max": max(gaps) if gaps else 0,
        "short_arrival_gap_fraction": round(
            sum(gap <= 90 for gap in gaps) / max(1, len(gaps)), 6
        ),
        "fixed_coverage_ratio": round(fixed_coverage, 6),
        "ordinary_mobile_required_pair_fraction": round(
            mobile_required_pairs / stand_job_pairs, 6
        ),
        "mean_eligible_resource_instances": round(
            statistics.fmean(eligible_counts), 6
        ) if eligible_counts else 0.0,
        "mobile_type_counts": dict(sorted(mobile_counts.items())),
    }


class AirportScenarioGenerator:
    """Generate one fixed-width but structurally heterogeneous HKBZ instance."""

    def __init__(
        self,
        *,
        profile: ScenarioProfile,
        seed: int,
        split: str,
        case_id: str,
        num_stands: int = SERVICE_STANDS,
        max_planes: int = MAX_PLANES,
    ) -> None:
        if num_stands != SERVICE_STANDS:
            raise ValueError(
                f"Current MARL batching requires exactly {SERVICE_STANDS} service stands"
            )
        if max_planes != MAX_PLANES:
            raise ValueError(
                f"Current policy configuration requires max_planes={MAX_PLANES}"
            )
        self.profile = profile
        self.seed = int(seed)
        self.split = split
        self.case_id = case_id
        self.num_stands = num_stands
        self.max_planes = max_planes
        self.rng = random.Random(self.seed)

    def _generate_layout(self, cluster_count: int) -> tuple[dict[str, Any], list[str]]:
        candidates = []
        for col in range(8):
            for row in range(7):
                candidates.append([180 + col * 110, 120 + row * 150])
        self.rng.shuffle(candidates)
        selected = candidates[: self.num_stands]
        positions = [
            [point[0] + self.rng.randint(-25, 25), point[1] + self.rng.randint(-25, 25)]
            for point in selected
        ]
        clusters = _farthest_first_kmeans(positions, cluster_count, self.rng)

        codes = ["Z"]
        ordered_positions: list[list[float]] = [[50.0, MAP_SIZE / 2.0]]
        cluster_ranges: list[str] = []
        next_code = 1
        for cluster in clusters:
            start = next_code
            for position in cluster:
                codes.append(str(next_code))
                ordered_positions.append([round(position[0], 1), round(position[1], 1)])
                next_code += 1
            cluster_ranges.append(f"{start}-{next_code - 1}")

        for runway_offset in range(TAKEOFF_SITES):
            codes.append(str(self.num_stands + runway_offset + 1))
            ordered_positions.append(
                [MAP_SIZE - 50.0, MAP_SIZE * (runway_offset + 1) / (TAKEOFF_SITES + 1)]
            )
        return {"sites_codes": codes, "sites_positions": ordered_positions}, cluster_ranges

    def _generate_jobs(
        self,
        flexibility: str,
        precedence_template: str,
        duration_scale: float,
    ) -> list[dict[str, Any]]:
        jobs = copy.deepcopy(BASE_JOB_DATA)
        by_code = {job["作业编号"]: job for job in jobs}
        for code, predecessors in PRECEDENCE_TEMPLATES[precedence_template].items():
            by_code[code]["前置作业"] = list(predecessors)

        alternative_pairs = {
            "ZY04": ("R005", "R013"),
            "ZY05": ("R003", "R012"),
            "ZY07": ("R006", "R011"),
        }
        if flexibility == "sdata":
            for code, (_, mobile_only_type) in alternative_pairs.items():
                by_code[code]["需要设备类型"] = [mobile_only_type]
        elif flexibility == "edata":
            forced_mobile = self.rng.choice(sorted(alternative_pairs))
            for code, pair in alternative_pairs.items():
                by_code[code]["需要设备类型"] = (
                    [pair[1]] if code == forced_mobile or self.rng.random() < 0.45 else list(pair)
                )
        elif flexibility == "rdata":
            forced_mobile = self.rng.choice(sorted(alternative_pairs))
            for code, pair in alternative_pairs.items():
                by_code[code]["需要设备类型"] = [pair[1]] if code == forced_mobile else list(pair)
        elif flexibility == "vdata":
            for code, pair in alternative_pairs.items():
                by_code[code]["需要设备类型"] = list(pair)
        else:
            raise ValueError(f"Unknown flexibility class: {flexibility}")

        if precedence_template in {"coupled", "inspection_first"}:
            by_code["ZY05"]["互斥作业"] = ["ZY06"]
            by_code["ZY06"]["互斥作业"] = ["ZY05"]

        for job in _service_jobs(jobs):
            duration = job["作业时间"]
            if isinstance(duration, (int, float)):
                noise = self.rng.uniform(0.85, 1.20)
                job["作业时间"] = max(1, int(round(duration * duration_scale * noise)))
        return jobs

    def _generate_fixed_resources(
        self,
        cluster_ranges: Sequence[str],
        coverage: float,
    ) -> list[dict[str, str]]:
        fixed_resources: list[dict[str, str]] = []
        resource_idx = 1
        type_bias = {
            "R001": 0.90,
            "R002": 1.00,
            "R003": 0.95,
            "R005": 0.95,
            "R006": 0.85,
            "R007": 1.05,
            "R008": 1.00,
        }
        selected_by_cluster: list[set[str]] = []
        for _ in cluster_ranges:
            selected = {
                resource_type
                for resource_type in FIXED_RESOURCE_TYPES
                if self.rng.random() < min(0.88, coverage * type_bias[resource_type])
            }
            if len(selected) == len(FIXED_RESOURCE_TYPES):
                selected.remove(self.rng.choice(sorted(selected)))
            selected_by_cluster.append(selected)

        for resource_type in FIXED_RESOURCE_TYPES:
            covered_clusters = [
                idx for idx, selected in enumerate(selected_by_cluster) if resource_type in selected
            ]
            if len(covered_clusters) == len(cluster_ranges):
                selected_by_cluster[self.rng.choice(covered_clusters)].remove(resource_type)

        for cluster_range, selected in zip(cluster_ranges, selected_by_cluster):
            for resource_type in sorted(selected):
                fixed_resources.append(
                    {
                        "设备编号": f"FR{resource_idx:03d}",
                        "类型": resource_type,
                        "支持停机位": cluster_range,
                    }
                )
                resource_idx += 1
        return fixed_resources

    def _generate_mobile_resources(
        self,
        jobs: Sequence[Mapping[str, Any]],
        transporter_count: int,
        depot_bias: float,
    ) -> list[dict[str, str]]:
        counts = {resource_type: 2 for resource_type in ORDINARY_MOBILE_TYPES}
        demand_weights = Counter()
        for job in _resource_jobs(jobs):
            for resource_type in job["需要设备类型"]:
                if resource_type in ORDINARY_MOBILE_TYPES:
                    demand_weights[resource_type] += 1
        remaining = MOBILE_DEVICE_BUDGET - transporter_count - sum(counts.values())
        weighted_types = list(ORDINARY_MOBILE_TYPES)
        weights = [1.0 + 2.0 * demand_weights[resource_type] for resource_type in weighted_types]
        for resource_type in self.rng.choices(weighted_types, weights=weights, k=remaining):
            counts[resource_type] += 1
        counts[TRANSPORTER_TYPE] = transporter_count

        resources: list[dict[str, str]] = []
        resource_idx = 1
        for resource_type in ALL_MOBILE_TYPES:
            for _ in range(counts[resource_type]):
                initial_site = (
                    "Z"
                    if self.rng.random() < depot_bias
                    else str(self.rng.randint(1, self.num_stands))
                )
                resources.append(
                    {
                        "设备编号": f"MR{resource_idx:03d}",
                        "类型": resource_type,
                        "初始停机位": initial_site,
                    }
                )
                resource_idx += 1
        return resources

    def _generate_flights(self, plane_count: int, arrival_mode: str) -> list[dict[str, Any]]:
        arrivals = [0]
        while len(arrivals) < plane_count:
            flight_idx = len(arrivals)
            if arrival_mode == "steady":
                gap = self.rng.randint(150, 260)
            elif arrival_mode == "mixed":
                gap = self.rng.randint(25, 90) if self.rng.random() < 0.38 else self.rng.randint(160, 300)
            elif arrival_mode == "bursty":
                gap = self.rng.randint(20, 65) if flight_idx % self.rng.randint(3, 6) else self.rng.randint(230, 400)
            elif arrival_mode == "waves":
                position_in_wave = flight_idx % 5
                gap = self.rng.randint(15, 55) if position_in_wave else self.rng.randint(280, 450)
            else:
                raise ValueError(f"Unknown arrival mode: {arrival_mode}")
            arrivals.append(arrivals[-1] + gap)

        return [
            {
                "飞机编号": f"F{index + 1:02d}",
                "到达时间": arrival,
                "初始燃油状态": f"{self.rng.randint(15, 95)}%",
            }
            for index, arrival in enumerate(arrivals)
        ]

    def generate(self) -> tuple[dict[str, Any], dict[str, Any]]:
        plane_count = self.rng.randint(*self.profile.plane_range)
        cluster_count = self.rng.choice(self.profile.cluster_choices)
        coverage = self.rng.uniform(*self.profile.fixed_coverage_range)
        flexibility = self.rng.choice(self.profile.flexibility_choices)
        precedence_template = self.rng.choice(self.profile.precedence_choices)
        arrival_mode = self.rng.choice(self.profile.arrival_choices)
        duration_scale = self.rng.uniform(*self.profile.duration_scale_range)
        depot_bias = self.rng.uniform(*self.profile.depot_bias_range)
        transporter_count = self.rng.randint(*self.profile.transporter_range)

        sites, cluster_ranges = self._generate_layout(cluster_count)
        jobs = self._generate_jobs(flexibility, precedence_template, duration_scale)
        fixed_resources = self._generate_fixed_resources(cluster_ranges, coverage)
        mobile_resources = self._generate_mobile_resources(jobs, transporter_count, depot_bias)
        flights = self._generate_flights(plane_count, arrival_mode)
        case = {
            "jobs": jobs,
            "fixed_resources": fixed_resources,
            "mobile_resources": mobile_resources,
            "sites": sites,
            "flights": flights,
        }
        self._validate(case)
        stats = _case_statistics(case)
        metadata = {
            "schema_version": GENERATOR_VERSION,
            "case_id": self.case_id,
            "split": self.split,
            "seed": self.seed,
            "profile": self.profile.name,
            "distribution": self.profile.distribution,
            "parameters": {
                "plane_count": plane_count,
                "max_plane_agents": self.max_planes,
                "service_stands": self.num_stands,
                "takeoff_sites": TAKEOFF_SITES,
                "cluster_count": cluster_count,
                "target_fixed_coverage": round(coverage, 6),
                "flexibility_class": flexibility,
                "precedence_template": precedence_template,
                "arrival_mode": arrival_mode,
                "duration_scale": round(duration_scale, 6),
                "depot_bias": round(depot_bias, 6),
                "mobile_device_budget": MOBILE_DEVICE_BUDGET,
                "transporter_count": transporter_count,
            },
            "statistics": stats,
        }
        return case, metadata

    def _validate(self, case: Mapping[str, Any]) -> None:
        jobs = case["jobs"]
        sites = case["sites"]
        flights = case["flights"]
        mobile_resources = case["mobile_resources"]
        if not (0 < len(flights) <= self.max_planes):
            raise ValueError("Flight count exceeds fixed plane-agent capacity")
        if len(sites["sites_codes"]) != self.num_stands + TAKEOFF_SITES + 1:
            raise ValueError("Unexpected number of site nodes")
        if len(set(sites["sites_codes"])) != len(sites["sites_codes"]):
            raise ValueError("Site codes must be unique")
        for left_idx, left in enumerate(sites["sites_positions"]):
            for right in sites["sites_positions"][left_idx + 1 :]:
                if _distance(left, right) < MIN_SITE_DISTANCE:
                    raise ValueError("Site positions violate the minimum distance")

        service_jobs = _service_jobs(jobs)
        service_codes = {job["作业编号"] for job in service_jobs}
        predecessors = {
            job["作业编号"]: [pred for pred in job["前置作业"] if pred in service_codes]
            for job in service_jobs
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(code: str) -> None:
            if code in visiting:
                raise ValueError(f"Cyclic job precedence detected at {code}")
            if code in visited:
                return
            visiting.add(code)
            for predecessor in predecessors[code]:
                visit(predecessor)
            visiting.remove(code)
            visited.add(code)

        for code in predecessors:
            visit(code)

        mobile_types = {resource["类型"] for resource in mobile_resources}
        required_types = {
            resource_type
            for job in _resource_jobs(jobs)
            for resource_type in job["需要设备类型"]
        }
        if not required_types.issubset(mobile_types):
            raise ValueError(
                f"All demanded resource types need a mobile fallback; missing {sorted(required_types - mobile_types)}"
            )
        if TRANSPORTER_TYPE not in mobile_types:
            raise ValueError("At least one transporter is required")
        if len(mobile_resources) != MOBILE_DEVICE_BUDGET:
            raise ValueError("Mobile device count must stay fixed for batched MARL")

        stats = _case_statistics(case)
        if stats["ordinary_mobile_required_pair_fraction"] < 0.15:
            raise ValueError("Instance still starves the ordinary-device policy")
        if stats["fixed_coverage_ratio"] > 0.85:
            raise ValueError("Fixed-resource coverage is too close to universal")
        if self.profile.name in {"bursty", "stress_joint", "stress_arrival"}:
            if stats["short_arrival_gap_fraction"] < 0.25:
                raise ValueError("Bursty profile did not create enough short release gaps")


def _write_case(case_dir: Path, case: Mapping[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=False)
    file_values = {
        "job.json": case["jobs"],
        "fixed_resources.json": case["fixed_resources"],
        "mobile_resources.json": case["mobile_resources"],
        "sites.json": case["sites"],
        "flights.json": case["flights"],
    }
    file_hashes = {}
    for filename, value in file_values.items():
        _write_json(case_dir / filename, value)
        file_hashes[filename] = _sha256(value)
    case_hash = _sha256(file_values)
    metadata = copy.deepcopy(metadata)
    metadata["fingerprints"] = {"case_sha256": case_hash, "files": file_hashes}
    _write_json(case_dir / "metadata.json", metadata)
    return {
        "case_id": metadata["case_id"],
        "seed": metadata["seed"],
        "profile": metadata["profile"],
        "distribution": metadata["distribution"],
        "case_sha256": case_hash,
    }


def _read_case(case_dir: Path) -> dict[str, Any]:
    def read(name: str) -> Any:
        return json.loads((case_dir / name).read_text(encoding="utf-8"))

    return {
        "jobs": read("job.json"),
        "fixed_resources": read("fixed_resources.json"),
        "mobile_resources": read("mobile_resources.json"),
        "sites": read("sites.json"),
        "flights": read("flights.json"),
        "metadata": read("metadata.json"),
    }


def _job_structure(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "code": job["作业编号"],
            "resources": sorted(job["需要设备类型"]),
            "predecessors": sorted(job["前置作业"]),
            "exclusive": sorted(job["互斥作业"]),
            "group": job["分组"],
        }
        for job in jobs
    ]


def audit_dataset(dataset_root: Path, *, write_report: bool = True) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_reports: dict[str, Any] = {}
    fingerprints_by_split: dict[str, set[str]] = {}
    all_failures: list[str] = []
    for split in ("train", "validation", "test"):
        case_dirs = sorted((dataset_root / split).glob("case_*"))
        expected_count = int(manifest["splits"][split]["count"])
        if len(case_dirs) != expected_count:
            all_failures.append(
                f"{split}: expected {expected_count} cases, found {len(case_dirs)}"
            )

        case_hashes: list[str] = []
        structure_hashes: list[str] = []
        job_hashes: list[str] = []
        profiles = Counter()
        distributions = Counter()
        stats_rows: list[dict[str, Any]] = []
        for case_dir in case_dirs:
            loaded = _read_case(case_dir)
            metadata = loaded.pop("metadata")
            case_hash = _sha256(
                {
                    "job.json": loaded["jobs"],
                    "fixed_resources.json": loaded["fixed_resources"],
                    "mobile_resources.json": loaded["mobile_resources"],
                    "sites.json": loaded["sites"],
                    "flights.json": loaded["flights"],
                }
            )
            if case_hash != metadata["fingerprints"]["case_sha256"]:
                all_failures.append(f"{split}/{case_dir.name}: fingerprint mismatch")
            case_hashes.append(case_hash)
            structure_hashes.append(_sha256(_job_structure(loaded["jobs"])))
            job_hashes.append(_sha256(loaded["jobs"]))
            profiles[metadata["profile"]] += 1
            distributions[metadata["distribution"]] += 1
            stats_rows.append(_case_statistics(loaded))

        if len(set(case_hashes)) != len(case_hashes):
            all_failures.append(f"{split}: duplicate full cases detected")
        short_gap_values = [row["short_arrival_gap_fraction"] for row in stats_rows]
        mobile_required_values = [row["ordinary_mobile_required_pair_fraction"] for row in stats_rows]
        fixed_coverage_values = [row["fixed_coverage_ratio"] for row in stats_rows]
        plane_counts = [row["num_planes"] for row in stats_rows]
        split_report = {
            "case_count": len(case_dirs),
            "unique_full_cases": len(set(case_hashes)),
            "unique_job_files": len(set(job_hashes)),
            "unique_job_structures": len(set(structure_hashes)),
            "profiles": dict(sorted(profiles.items())),
            "distributions": dict(sorted(distributions.items())),
            "plane_count": {
                "min": min(plane_counts) if plane_counts else 0,
                "max": max(plane_counts) if plane_counts else 0,
                "mean": round(statistics.fmean(plane_counts), 6) if plane_counts else 0.0,
            },
            "short_arrival_gap_fraction_mean": round(
                statistics.fmean(short_gap_values), 6
            ) if short_gap_values else 0.0,
            "ordinary_mobile_required_pair_fraction_mean": round(
                statistics.fmean(mobile_required_values), 6
            ) if mobile_required_values else 0.0,
            "fixed_coverage_ratio_mean": round(
                statistics.fmean(fixed_coverage_values), 6
            ) if fixed_coverage_values else 0.0,
        }
        if split_report["unique_job_structures"] < 4:
            all_failures.append(f"{split}: fewer than four job structures")
        if split_report["short_arrival_gap_fraction_mean"] < 0.12:
            all_failures.append(f"{split}: release-time distribution is still too smooth")
        mobile_mean = split_report["ordinary_mobile_required_pair_fraction_mean"]
        if not 0.20 <= mobile_mean <= 0.90:
            all_failures.append(
                f"{split}: ordinary mobile-required fraction {mobile_mean:.3f} outside [0.20, 0.90]"
            )
        split_reports[split] = split_report
        fingerprints_by_split[split] = set(case_hashes)

    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = fingerprints_by_split[left].intersection(fingerprints_by_split[right])
        overlaps[f"{left}__{right}"] = len(overlap)
        if overlap:
            all_failures.append(f"{left}/{right}: {len(overlap)} exact case overlaps")

    report = {
        "schema_version": GENERATOR_VERSION,
        "dataset_root": str(dataset_root),
        "status": "PASS" if not all_failures else "FAIL",
        "splits": split_reports,
        "cross_split_exact_overlaps": overlaps,
        "quality_gates": {
            "no_duplicate_cases": True,
            "no_cross_split_leakage": True,
            "min_unique_job_structures_per_split": 4,
            "min_mean_short_arrival_gap_fraction": 0.12,
            "ordinary_mobile_required_pair_fraction_range": [0.20, 0.90],
        },
        "failures": all_failures,
    }
    if write_report:
        _write_json(dataset_root / "audit.json", report)
        lines = [
            "# HKBZ FJSP-v2 Dataset Audit",
            "",
            f"- Status: **{report['status']}**",
            f"- Generator: `{GENERATOR_VERSION}`",
            "- Exact train/validation/test overlap: "
            + ", ".join(f"{name}={count}" for name, count in overlaps.items()),
            "",
            "| Split | Cases | Job structures | Plane range | Short-gap fraction | Mobile-required pairs | Fixed coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for split, split_report in split_reports.items():
            plane = split_report["plane_count"]
            lines.append(
                f"| {split} | {split_report['case_count']} | "
                f"{split_report['unique_job_structures']} | {plane['min']}-{plane['max']} | "
                f"{split_report['short_arrival_gap_fraction_mean']:.3f} | "
                f"{split_report['ordinary_mobile_required_pair_fraction_mean']:.3f} | "
                f"{split_report['fixed_coverage_ratio_mean']:.3f} |"
            )
        if all_failures:
            lines.extend(["", "## Failures", ""] + [f"- {failure}" for failure in all_failures])
        (dataset_root / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_benchmark_dataset(
    output_root: Path,
    *,
    train_cases: int = 600,
    validation_cases: int = 120,
    test_cases: int = 60,
    seed: int = 20260715,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty dataset directory: {output_root}"
        )
    if output_root.exists():
        output_root.rmdir()

    building_root = output_root.with_name(output_root.name + ".building")
    if building_root.exists():
        raise FileExistsError(
            f"Stale build directory exists; inspect it before retrying: {building_root}"
        )
    building_root.mkdir(parents=True)

    split_counts = {
        "train": int(train_cases),
        "validation": int(validation_cases),
        "test": int(test_cases),
    }
    manifest_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, count in split_counts.items():
        split_dir = building_root / split
        split_dir.mkdir()
        profile_schedule = _allocate_profile_schedule(split, count, seed)
        for index, profile_name in enumerate(profile_schedule, start=1):
            case_id = f"{split}_{index:04d}"
            case_seed = _derived_seed(seed, split, index, profile_name)
            generator = AirportScenarioGenerator(
                profile=PROFILES[profile_name],
                seed=case_seed,
                split=split,
                case_id=case_id,
            )
            case, metadata = generator.generate()
            manifest_cases[split].append(
                _write_case(split_dir / f"case_{index:04d}", case, metadata)
            )

    manifest = {
        "schema_version": GENERATOR_VERSION,
        "base_seed": int(seed),
        "static_model_contract": {
            "max_plane_agents": MAX_PLANES,
            "service_stands": SERVICE_STANDS,
            "takeoff_sites": TAKEOFF_SITES,
            "site_nodes": SERVICE_STANDS + TAKEOFF_SITES + 1,
            "mobile_device_nodes": MOBILE_DEVICE_BUDGET,
            "job_slots": len(BASE_JOB_DATA),
        },
        "split_semantics": {
            "train": "parameter fitting; 80% IID plus 20% OOD augmentation",
            "validation": "checkpoint selection and early stopping only; 50% IID, 45% OOD stress, 5% OOD scale",
            "test": "final reporting only; 50% IID and 50% held-out stress/scale profiles",
        },
        "references": list(REFERENCE_DESIGN),
        "profiles": {name: asdict(profile) for name, profile in PROFILES.items()},
        "splits": {
            split: {
                "count": split_counts[split],
                "profile_weights": dict(SPLIT_PROFILE_WEIGHTS[split]),
                "cases": manifest_cases[split],
            }
            for split in ("train", "validation", "test")
        },
    }
    _write_json(building_root / "manifest.json", manifest)
    audit = audit_dataset(building_root, write_report=True)
    if audit["status"] != "PASS":
        raise RuntimeError(
            "Generated dataset failed quality gates; build kept at "
            f"{building_root}: {audit['failures']}"
        )
    building_root.replace(output_root)
    return audit_dataset(output_root, write_report=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Generate all dataset splits")
    build_parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_ROOT)
    build_parser.add_argument("--train-cases", type=int, default=600)
    build_parser.add_argument("--validation-cases", type=int, default=120)
    build_parser.add_argument("--test-cases", type=int, default=60)
    build_parser.add_argument("--seed", type=int, default=20260715)

    audit_parser = subparsers.add_parser("audit", help="Audit an existing generated dataset")
    audit_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        report = build_benchmark_dataset(
            args.output,
            train_cases=args.train_cases,
            validation_cases=args.validation_cases,
            test_cases=args.test_cases,
            seed=args.seed,
        )
    else:
        report = audit_dataset(args.dataset_root, write_report=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
