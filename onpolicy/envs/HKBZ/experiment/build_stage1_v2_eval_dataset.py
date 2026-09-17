#!/usr/bin/env python
"""Build independent Stage-1 tune/select and blind-test benchmark splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.data_generator import (
    GENERATOR_VERSION,
    PROFILES,
    AirportScenarioGenerator,
    _allocate_profile_schedule,
    _derived_seed,
    _write_case,
)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def _build_split(root: Path, split: str, count: int, seed: int) -> list[dict]:
    split_dir = root / split
    split_dir.mkdir()
    records = []
    schedule = _allocate_profile_schedule(split, count, seed)
    for index, profile_name in enumerate(schedule, start=1):
        case_id = f'{split}_{index:04d}'
        case_seed = _derived_seed(seed, split, index, profile_name)
        generator = AirportScenarioGenerator(
            profile=PROFILES[profile_name],
            seed=case_seed,
            split=split,
            case_id=case_id,
        )
        case, metadata = generator.generate()
        records.append(
            _write_case(split_dir / f'case_{index:04d}', case, metadata)
        )
    return records


def build(output: Path, *, validation_cases: int, test_cases: int, seed: int) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty {output}.')
    if output.exists():
        output.rmdir()
    building = output.with_name(output.name + '.building')
    if building.exists():
        raise FileExistsError(f'Stale build directory exists: {building}.')
    building.mkdir(parents=True)
    try:
        validation = _build_split(
            building, 'validation', validation_cases, seed
        )
        blind_test = _build_split(building, 'test', test_cases, seed)
        old_hashes = set()
        legacy_root = output.parent / 'fjsp_v3_t600_v120_test60'
        legacy_manifest = legacy_root / 'manifest.json'
        if legacy_manifest.is_file():
            legacy = json.loads(legacy_manifest.read_text(encoding='utf-8'))
            for split in legacy.get('splits', {}).values():
                for record in split.get('cases', []):
                    value = record.get('case_sha256')
                    if value:
                        old_hashes.add(value)
        generated_hashes = {
            record.get('case_sha256') for record in validation + blind_test
        }
        generated_hashes.discard(None)
        overlaps = sorted(generated_hashes & old_hashes)
        if overlaps:
            raise RuntimeError(
                f'Independent evaluation build overlaps {len(overlaps)} legacy cases.'
            )
        if len(generated_hashes) != validation_cases + test_cases:
            raise RuntimeError('Generated evaluation cases are not unique.')
        manifest = {
            'schema_version': f'{GENERATOR_VERSION}-stage1-v2-eval',
            'base_seed': int(seed),
            'purpose': {
                'validation': (
                    'fixed-seed partition: first half tune/canary, second half '
                    'independent screen selection'
                ),
                'test': 'blind final reporting only',
            },
            'legacy_dataset_overlap_count': 0,
            'profiles': {
                name: {
                    **profile.__dict__,
                    'plane_range': list(profile.plane_range),
                    'cluster_choices': list(profile.cluster_choices),
                    'flexibility_choices': list(profile.flexibility_choices),
                    'precedence_choices': list(profile.precedence_choices),
                    'arrival_choices': list(profile.arrival_choices),
                    'duration_scale_range': list(profile.duration_scale_range),
                    'depot_bias_range': list(profile.depot_bias_range),
                    'transporter_range': list(profile.transporter_range),
                }
                for name, profile in PROFILES.items()
            },
            'splits': {
                'validation': {
                    'count': len(validation),
                    'cases': validation,
                },
                'test': {
                    'count': len(blind_test),
                    'cases': blind_test,
                },
            },
        }
        _atomic_json(building / 'manifest.json', manifest)
        building.replace(output)
        return manifest
    except BaseException:
        # Keep a failed build for diagnosis; never expose it as the final path.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation_cases', type=int, default=120)
    parser.add_argument('--test_cases', type=int, default=60)
    parser.add_argument('--seed', type=int, default=20260803)
    args = parser.parse_args()
    if args.validation_cases < 120 or args.test_cases < 60:
        raise ValueError('Production v2 evaluation requires at least 120/60 cases.')
    manifest = build(
        args.output,
        validation_cases=args.validation_cases,
        test_cases=args.test_cases,
        seed=args.seed,
    )
    print(
        json.dumps({
            'output': str(args.output.resolve()),
            'base_seed': manifest['base_seed'],
            'validation_cases': manifest['splits']['validation']['count'],
            'blind_test_cases': manifest['splits']['test']['count'],
            'legacy_overlap_count': manifest['legacy_dataset_overlap_count'],
        }, ensure_ascii=False),
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
