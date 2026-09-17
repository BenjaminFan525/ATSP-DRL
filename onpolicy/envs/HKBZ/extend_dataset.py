"""Safely append deterministic cases to an existing HKBZ FJSP-v2 split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from onpolicy.envs.HKBZ.data_generator import (
    GENERATOR_VERSION,
    PROFILES,
    AirportScenarioGenerator,
    _allocate_profile_schedule,
    _derived_seed,
    _write_case,
    _write_json,
    audit_dataset,
)


def extend_test_split(dataset_root: Path, target_count: int) -> dict:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != GENERATOR_VERSION:
        raise ValueError("Dataset generator version does not match this extension tool")

    split_manifest = manifest["splits"]["test"]
    current_count = int(split_manifest["count"])
    existing_dirs = sorted((dataset_root / "test").glob("case_*"))
    if len(existing_dirs) != current_count:
        raise ValueError(
            f"Manifest records {current_count} test cases but filesystem has {len(existing_dirs)}"
        )
    if target_count <= current_count:
        raise ValueError(
            f"target_count must exceed current test count {current_count}, got {target_count}"
        )

    additional_count = target_count - current_count
    base_seed = int(manifest["base_seed"])
    extension_seed = _derived_seed(
        base_seed,
        "test_extension",
        current_count,
        f"target_{target_count}",
    )
    profiles = _allocate_profile_schedule("test", additional_count, extension_seed)
    stage_dir = dataset_root / f".extend_test_{current_count + 1:04d}_{target_count:04d}"
    if stage_dir.exists():
        raise FileExistsError(f"Stale extension staging directory exists: {stage_dir}")
    stage_dir.mkdir()

    new_entries = []
    profile_counts = Counter()
    for offset, profile_name in enumerate(profiles, start=1):
        case_index = current_count + offset
        case_id = f"test_{case_index:04d}"
        case_seed = _derived_seed(
            base_seed,
            f"test_extension_{current_count}_{target_count}",
            case_index,
            profile_name,
        )
        generator = AirportScenarioGenerator(
            profile=PROFILES[profile_name],
            seed=case_seed,
            split="test",
            case_id=case_id,
        )
        case, metadata = generator.generate()
        new_entries.append(
            _write_case(stage_dir / f"case_{case_index:04d}", case, metadata)
        )
        profile_counts[profile_name] += 1

    test_dir = dataset_root / "test"
    for case_index in range(current_count + 1, target_count + 1):
        source = stage_dir / f"case_{case_index:04d}"
        destination = test_dir / source.name
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing case: {destination}")
        source.replace(destination)
    stage_dir.rmdir()

    split_manifest["count"] = target_count
    split_manifest["cases"].extend(new_entries)
    manifest.setdefault("extension_history", []).append(
        {
            "split": "test",
            "previous_count": current_count,
            "new_count": target_count,
            "extension_seed": extension_seed,
            "profile_counts": dict(sorted(profile_counts.items())),
        }
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    _write_json(temporary_manifest, manifest)
    temporary_manifest.replace(manifest_path)
    return audit_dataset(dataset_root, write_report=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--target-test-cases", type=int, required=True)
    args = parser.parse_args()
    report = extend_test_split(args.dataset_root, args.target_test_cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
