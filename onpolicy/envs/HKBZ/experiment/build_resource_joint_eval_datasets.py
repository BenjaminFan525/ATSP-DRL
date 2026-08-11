#!/usr/bin/env python
"""Build the tune/gate/final-blind resource-joint evaluation partitions.

The builder deliberately keeps the production partition sizes in this module
and exposes ``partition_counts`` only through the Python API.  This makes the
CLI unambiguous while still allowing fast, small, temporary builds in tests.
Every role has its own seed namespace, and the manifest is written before an
atomic directory publication.  A failed build therefore leaves a diagnostic
``.building`` directory but can never publish an incomplete collection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.data_generator import (
    GENERATOR_VERSION,
    MAX_PLANES,
    MOBILE_DEVICE_BUDGET,
    PROFILES,
    SERVICE_STANDS,
    SPLIT_PROFILE_WEIGHTS,
    TAKEOFF_SITES,
    AirportScenarioGenerator,
    _allocate_profile_schedule,
    _derived_seed,
    _sha256,
    _write_case,
)


# This is the public production contract.  Keep it as a plain mapping so
# callers/tests can compare it directly and so a future schema migration is
# obvious in code review.
PARTITIONS: dict[str, dict[str, int]] = {
    "joint": {"tune": 60, "gate": 120, "finalblind": 60},
}
PRODUCTION_PARTITIONS = PARTITIONS

ROLE_POLICIES: dict[str, dict[str, Any]] = {
    "tune": {
        "sealed": False,
        "selection_policy": "calibration_and_screen",
        "purpose": "calibration and research screen only",
    },
    "gate": {
        "sealed": True,
        "selection_policy": "formal_selection_only",
        "purpose": "formal selection only",
    },
    "finalblind": {
        "sealed": True,
        "selection_policy": "winner_lock_then_final_reporting",
        "purpose": "report only after the resource-joint winner is locked",
    },
}

# The validation profile schedule in data_generator.py is the authoritative
# 50/45/5 schedule.  Exposing the target here makes the contract inspectable
# without duplicating profile-specific weights.
DISTRIBUTION_TARGETS: dict[str, float] = {
    "iid": 0.50,
    "ood_stress": 0.45,
    "ood_scale": 0.05,
}
SELECTION_POLICIES = ROLE_POLICIES
ROLE_NAMES = tuple(sorted({role for family in PARTITIONS.values() for role in family}))
PROFILE_SCHEDULE_SPLIT = "validation"
MANIFEST_SCHEMA_SUFFIX = "-resource-joint-eval"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON file using a same-directory temporary and replace."""

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _clone_partition_counts(
    partition_counts: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, dict[str, int]]:
    source = PARTITIONS if partition_counts is None else partition_counts
    result: dict[str, dict[str, int]] = {}
    if set(source) != set(PARTITIONS):
        raise ValueError(
            "partition_counts must contain exactly the joint family"
        )
    for family, expected_roles in PARTITIONS.items():
        supplied = source[family]
        if set(supplied) != set(expected_roles):
            raise ValueError(
                f"partition_counts[{family!r}] must contain exactly "
                f"{sorted(expected_roles)}"
            )
        result[family] = {}
        for role in expected_roles:
            count = supplied[role]
            try:
                normalized_count = int(count)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Invalid case count for {family}/{role}: {count!r}"
                ) from exc
            if (
                isinstance(count, bool)
                or normalized_count != count
                or normalized_count < 0
            ):
                raise ValueError(f"Invalid case count for {family}/{role}: {count!r}")
            result[family][role] = normalized_count
    return result


def _role_namespace(family: str, role: str) -> str:
    return f"resource_joint/{family}/{role}"


def _schedule_for_role(
    family: str,
    role: str,
    count: int,
    base_seed: int,
) -> list[str]:
    """Allocate the validation 50/45/5 profile schedule in a role namespace."""

    namespace = _role_namespace(family, role)
    schedule_seed = _derived_seed(
        base_seed, f"{namespace}/profile_schedule", 0, "profile_schedule"
    )
    # _allocate_profile_schedule owns the deterministic largest-remainder and
    # shuffle behavior.  ``validation`` is the existing 50/45/5 profile table.
    return _allocate_profile_schedule(PROFILE_SCHEDULE_SPLIT, count, schedule_seed)


def _profile_payloads() -> dict[str, dict[str, Any]]:
    return {name: asdict(profile) for name, profile in PROFILES.items()}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_fingerprints(case_dir: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Recompute the generator's content fingerprints from a written case."""

    filenames = (
        "job.json",
        "fixed_resources.json",
        "mobile_resources.json",
        "sites.json",
        "flights.json",
    )
    values: dict[str, Any] = {}
    file_hashes: dict[str, str] = {}
    for filename in filenames:
        path = case_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing case file: {path}")
        value = _read_json(path)
        values[filename] = value
        file_hashes[filename] = _sha256(value)
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing case metadata: {metadata_path}")
    metadata = _read_json(metadata_path)
    return _sha256(values), file_hashes, metadata


def _record_for_case(
    *,
    case_dir: Path,
    family: str,
    role: str,
    case_index: int,
    derived_seed: int,
    profile_name: str,
    written: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _read_json(case_dir / "metadata.json")
    fingerprints = metadata.get("fingerprints", {})
    file_hashes = dict(fingerprints.get("files", {}))
    if fingerprints.get("case_sha256") != written.get("case_sha256"):
        raise RuntimeError(f"Generator fingerprint mismatch in {case_dir}")
    if not file_hashes:
        raise RuntimeError(f"Generator did not write file fingerprints in {case_dir}")
    namespace = _role_namespace(family, role)
    return {
        "case_id": written["case_id"],
        "case_index": int(case_index),
        "path": f"{family}/{role}/case_{case_index:04d}",
        "family": family,
        "role": role,
        "seed": int(written["seed"]),
        "derived_seed": int(derived_seed),
        "seed_namespace": namespace,
        "profile": profile_name,
        "distribution": written["distribution"],
        "case_sha256": written["case_sha256"],
        "file_sha256": file_hashes,
    }


def _build_collection(
    building_root: Path,
    *,
    family: str,
    role: str,
    count: int,
    base_seed: int,
) -> dict[str, Any]:
    if family not in PARTITIONS or role not in PARTITIONS[family]:
        raise ValueError(f"Unknown resource-joint collection: {family}/{role}")
    collection_dir = building_root / family / role
    collection_dir.mkdir(parents=True, exist_ok=False)
    schedule = _schedule_for_role(family, role, count, base_seed)
    namespace = _role_namespace(family, role)
    records: list[dict[str, Any]] = []
    for case_index, profile_name in enumerate(schedule, start=1):
        case_id = f"{family}_{role}_{case_index:04d}"
        case_seed = _derived_seed(base_seed, namespace, case_index, profile_name)
        generator = AirportScenarioGenerator(
            profile=PROFILES[profile_name],
            seed=case_seed,
            split=f"{family}_{role}",
            case_id=case_id,
        )
        case, metadata = generator.generate()
        written = _write_case(
            collection_dir / f"case_{case_index:04d}", case, metadata
        )
        records.append(
            _record_for_case(
                case_dir=collection_dir / f"case_{case_index:04d}",
                family=family,
                role=role,
                case_index=case_index,
                derived_seed=case_seed,
                profile_name=profile_name,
                written=written,
            )
        )

    profile_counts: Counter[str] = Counter({name: 0 for name in PROFILES})
    profile_counts.update(record["profile"] for record in records)
    distribution_counts: Counter[str] = Counter({name: 0 for name in DISTRIBUTION_TARGETS})
    distribution_counts.update(record["distribution"] for record in records)
    policy = ROLE_POLICIES[role]
    return {
        "family": family,
        "role": role,
        "count": int(count),
        "sealed": bool(policy["sealed"]),
        "selection_policy": policy["selection_policy"],
        "purpose": policy["purpose"],
        "seed_namespace": namespace,
        "profile_weights": dict(SPLIT_PROFILE_WEIGHTS[PROFILE_SCHEDULE_SPLIT]),
        "distribution_targets": dict(DISTRIBUTION_TARGETS),
        "profile_counts": dict(sorted(profile_counts.items())),
        "distribution_counts": dict(sorted(distribution_counts.items())),
        "profiles": dict(sorted(profile_counts.items())),
        "distributions": dict(sorted(distribution_counts.items())),
        "cases": records,
    }


def _iter_case_hashes(value: Any) -> Iterable[str]:
    """Yield content hashes from current and legacy manifest shapes."""

    if isinstance(value, Mapping):
        for key in ("case_sha256", "case_hash", "case_content_sha256"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                yield candidate
        for child in value.values():
            yield from _iter_case_hashes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_case_hashes(child)


def _manifest_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Exclude manifest does not exist: {path}")
    return path


def _load_excluded_hashes(paths: Sequence[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for supplied_path in paths:
        path = _manifest_path(Path(supplied_path))
        payload = _read_json(path)
        found = set(_iter_case_hashes(payload))
        hashes.update(found)
        summaries.append({"case_sha256_count": len(found)})
    return hashes, summaries


def _validate_collections(
    building_root: Path,
    collections: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    base_seed: int,
    excluded_hashes: set[str],
) -> dict[str, Any]:
    """Validate the complete candidate before its directory is published."""

    all_hashes: dict[str, str] = {}
    all_seeds: dict[int, str] = {}
    aggregate_profiles: Counter[str] = Counter({name: 0 for name in PROFILES})
    aggregate_distributions: Counter[str] = Counter({name: 0 for name in DISTRIBUTION_TARGETS})
    for family, family_collections in collections.items():
        for role, collection in family_collections.items():
            expected_count = int(collection["count"])
            case_dir = building_root / family / role
            if not case_dir.is_dir():
                raise RuntimeError(f"Missing collection directory: {case_dir}")
            case_dirs = sorted(path for path in case_dir.glob("case_*") if path.is_dir())
            if len(case_dirs) != expected_count:
                raise RuntimeError(
                    f"{family}/{role}: expected {expected_count} cases, found {len(case_dirs)}"
                )
            records = collection["cases"]
            if len(records) != expected_count:
                raise RuntimeError(
                    f"{family}/{role}: manifest case count does not match count"
                )
            profile_counts: Counter[str] = Counter({name: 0 for name in PROFILES})
            distribution_counts: Counter[str] = Counter({name: 0 for name in DISTRIBUTION_TARGETS})
            for case_index, record in enumerate(records, start=1):
                expected_case_dir = case_dir / f"case_{case_index:04d}"
                if expected_case_dir not in case_dirs:
                    raise RuntimeError(
                        f"{family}/{role}: missing ordered case {expected_case_dir.name}"
                    )
                case_hash, file_hashes, metadata = _case_fingerprints(expected_case_dir)
                if case_hash != record.get("case_sha256"):
                    raise RuntimeError(
                        f"{family}/{role}/{expected_case_dir.name}: case hash mismatch"
                    )
                if file_hashes != record.get("file_sha256"):
                    raise RuntimeError(
                        f"{family}/{role}/{expected_case_dir.name}: file hash mismatch"
                    )
                if metadata.get("seed") != record.get("derived_seed"):
                    raise RuntimeError(
                        f"{family}/{role}/{expected_case_dir.name}: derived seed mismatch"
                    )
                expected_seed = _derived_seed(
                    base_seed,
                    _role_namespace(family, role),
                    case_index,
                    record.get("profile", ""),
                )
                if expected_seed != record.get("derived_seed"):
                    raise RuntimeError(
                        f"{family}/{role}/{expected_case_dir.name}: seed namespace mismatch"
                    )
                if case_hash in all_hashes:
                    raise RuntimeError(
                        "Generated resource-joint collections contain duplicate content: "
                        f"{all_hashes[case_hash]} and {family}/{role}/{expected_case_dir.name}"
                    )
                if case_hash in excluded_hashes:
                    raise RuntimeError(
                        f"Generated resource-joint case overlaps an excluded manifest: "
                        f"{family}/{role}/{expected_case_dir.name} ({case_hash})"
                    )
                all_hashes[case_hash] = f"{family}/{role}/{expected_case_dir.name}"
                seed = int(record["derived_seed"])
                if seed in all_seeds:
                    raise RuntimeError(
                        "Generated resource-joint collections reused a derived seed: "
                        f"{all_seeds[seed]} and {family}/{role}/{expected_case_dir.name}"
                    )
                all_seeds[seed] = f"{family}/{role}/{expected_case_dir.name}"
                profile_counts[record["profile"]] += 1
                distribution_counts[record["distribution"]] += 1

            if dict(sorted(profile_counts.items())) != collection["profile_counts"]:
                raise RuntimeError(f"{family}/{role}: profile counts mismatch")
            if dict(sorted(distribution_counts.items())) != collection["distribution_counts"]:
                raise RuntimeError(f"{family}/{role}: distribution counts mismatch")
            aggregate_profiles.update(profile_counts)
            aggregate_distributions.update(distribution_counts)

    return {
        "case_count": len(all_hashes),
        "unique_case_sha256": len(all_hashes),
        "excluded_case_sha256_count": len(excluded_hashes),
        "profile_counts": dict(sorted(aggregate_profiles.items())),
        "distribution_counts": dict(sorted(aggregate_distributions.items())),
    }


def build(
    output: Path,
    *,
    seed: int = 20260811,
    exclude_manifests: Sequence[Path] = (),
    exclude_manifest: Path | None = None,
    partition_counts: Mapping[str, Mapping[str, int]] | None = None,
    counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Build and atomically publish all three resource-joint collections.

    ``partition_counts``/``counts`` are intentionally Python-only test hooks;
    the CLI always uses :data:`PARTITIONS`.
    """

    if partition_counts is not None and counts is not None:
        raise ValueError("Specify only one of partition_counts or counts")
    if counts is not None:
        partition_counts = counts
    if isinstance(exclude_manifests, (str, Path)):
        requested_excludes = [Path(exclude_manifests)]
    else:
        requested_excludes = [Path(path) for path in exclude_manifests]
    if exclude_manifest is not None:
        if isinstance(exclude_manifest, (str, Path)):
            requested_excludes.append(Path(exclude_manifest))
        else:
            requested_excludes.extend(Path(path) for path in exclude_manifest)
    resolved_counts = _clone_partition_counts(partition_counts)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
        output.rmdir()
    building_root = output.with_name(output.name + ".building")
    if building_root.exists():
        raise FileExistsError(
            f"Stale build directory exists; inspect it before retrying: {building_root}"
        )

    excluded_hashes, excluded_summaries = _load_excluded_hashes(requested_excludes)
    building_root.mkdir(parents=True)
    # Do not catch generation errors: retaining .building is useful evidence,
    # while output remains absent and therefore cannot look complete.
    collections: dict[str, dict[str, dict[str, Any]]] = {"joint": {}}
    for family in ("joint",):
        for role, count in resolved_counts[family].items():
            collections[family][role] = _build_collection(
                building_root,
                family=family,
                role=role,
                count=count,
                base_seed=int(seed),
            )

    manifest: dict[str, Any] = {
        "schema_version": f"{GENERATOR_VERSION}{MANIFEST_SCHEMA_SUFFIX}",
        "generator_version": GENERATOR_VERSION,
        "base_seed": int(seed),
        "static_model_contract": {
            "max_plane_agents": MAX_PLANES,
            "service_stands": SERVICE_STANDS,
            "takeoff_sites": TAKEOFF_SITES,
            "site_nodes": SERVICE_STANDS + TAKEOFF_SITES + 1,
            "mobile_device_nodes": MOBILE_DEVICE_BUDGET,
        },
        "partition_contract": {
            family: dict(roles) for family, roles in resolved_counts.items()
        },
        "role_policies": {role: dict(policy) for role, policy in ROLE_POLICIES.items()},
        "partition_semantics": {
            family: {
                role: {
                    "sealed": bool(ROLE_POLICIES[role]["sealed"]),
                    "selection_policy": ROLE_POLICIES[role]["selection_policy"],
                    "purpose": ROLE_POLICIES[role]["purpose"],
                }
                for role in roles
            }
            for family, roles in resolved_counts.items()
        },
        "seed_namespaces": {
            family: {
                role: _role_namespace(family, role)
                for role in roles
            }
            for family, roles in resolved_counts.items()
        },
        "distribution_targets": dict(DISTRIBUTION_TARGETS),
        "profile_weights": dict(SPLIT_PROFILE_WEIGHTS[PROFILE_SCHEDULE_SPLIT]),
        "profiles": _profile_payloads(),
        "exclude_manifests": excluded_summaries,
        "excluded_case_sha256_count": len(excluded_hashes),
        "partitions": collections,
    }
    validation = _validate_collections(
        building_root,
        collections,
        base_seed=int(seed),
        excluded_hashes=excluded_hashes,
    )
    manifest["validation"] = validation
    manifest["case_sha256_count"] = validation["case_count"]
    manifest["cross_collection_exact_overlap_count"] = 0
    manifest["excluded_overlap_count"] = 0
    manifest["legacy_dataset_overlap_count"] = 0
    _atomic_json(building_root / "manifest.json", manifest)
    # Validation has checked every case and the manifest is complete.  The
    # final rename is the only publication step and is atomic on one filesystem.
    building_root.replace(output)
    return manifest


def build_resource_joint_eval_datasets(
    output: Path,
    *,
    seed: int = 20260811,
    exclude_manifests: Sequence[Path] = (),
    partition_counts: Mapping[str, Mapping[str, int]] | None = None,
    counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Named wrapper for callers that prefer an explicit builder function."""

    return build(
        output,
        seed=seed,
        exclude_manifests=exclude_manifests,
        partition_counts=partition_counts,
        counts=counts,
    )


# Short aliases keep the API discoverable for experiment scripts while the
# descriptive wrapper above remains the documented entry point.
build_research_datasets = build_resource_joint_eval_datasets
build_datasets = build_resource_joint_eval_datasets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--exclude-manifest",
        dest="exclude_manifests",
        action="append",
        type=Path,
        default=[],
        help="Manifest (or dataset directory) whose case content hashes must be excluded; repeatable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build(
        args.output,
        seed=args.seed,
        exclude_manifests=args.exclude_manifests,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "base_seed": manifest["base_seed"],
                "partitions": manifest["partition_contract"],
                "case_sha256_count": manifest["case_sha256_count"],
                "excluded_case_sha256_count": manifest["excluded_case_sha256_count"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
