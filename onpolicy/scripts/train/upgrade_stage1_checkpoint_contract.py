#!/usr/bin/env python3
"""Create an immutable, metadata-complete copy of a trusted Stage-1 model."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.utils.checkpoint_contract import (
    stage1_observation_metadata,
    validate_stage1_checkpoint_contract,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--global-feature-mode', required=True)
    parser.add_argument('--plane-order-mode', default='fixed')
    parser.add_argument('--plane-pair-decoder', default='joint_pair')
    parser.add_argument('--reuse-if-valid', action='store_true')
    return parser.parse_args()


def validate_output(
    output: Path,
    *,
    source_sha256: str,
    global_feature_mode: str,
    plane_order_mode: str,
    plane_pair_decoder: str,
) -> dict:
    payload = torch.load(output, map_location='cpu')
    validate_stage1_checkpoint_contract(
        payload,
        global_feature_mode=global_feature_mode,
        plane_order_mode=plane_order_mode,
        plane_pair_decoder=plane_pair_decoder,
        strict_metadata=True,
    )
    upgrade = payload.get('source_checkpoint_contract_upgrade', {})
    if upgrade.get('source_sha256') != source_sha256:
        raise ValueError(
            'Existing upgraded checkpoint was produced from a different '
            'source artifact.'
        )
    if not isinstance(payload.get('model'), dict):
        raise ValueError('Upgraded checkpoint is missing its model mapping.')
    return payload


def main() -> int:
    args = parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.source == args.output:
        raise ValueError('Contract upgrade must not overwrite its source.')
    source_sha = sha256(args.source)
    if args.output.exists():
        if not args.reuse_if_valid:
            raise FileExistsError(args.output)
        validate_output(
            args.output,
            source_sha256=source_sha,
            global_feature_mode=args.global_feature_mode,
            plane_order_mode=args.plane_order_mode,
            plane_pair_decoder=args.plane_pair_decoder,
        )
        print(
            f'[CheckpointContract] Reusing validated {args.output} '
            f'(sha256={sha256(args.output)}).'
        )
        return 0

    checkpoint = torch.load(args.source, map_location='cpu')
    validate_stage1_checkpoint_contract(
        checkpoint,
        global_feature_mode=args.global_feature_mode,
        plane_order_mode=args.plane_order_mode,
        plane_pair_decoder=args.plane_pair_decoder,
        strict_metadata=False,
    )
    metadata = stage1_observation_metadata(args.global_feature_mode)
    upgraded = dict(checkpoint)
    upgraded.update(metadata)
    upgraded['source_checkpoint_contract_upgrade'] = {
        'schema_version': 1,
        'source_path': str(args.source),
        'source_sha256': source_sha,
        'metadata_only': True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f'.{args.output.name}.tmp.{os.getpid()}')
    try:
        torch.save(upgraded, temporary)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    validate_output(
        args.output,
        source_sha256=source_sha,
        global_feature_mode=args.global_feature_mode,
        plane_order_mode=args.plane_order_mode,
        plane_pair_decoder=args.plane_pair_decoder,
    )
    print(
        f'[CheckpointContract] Wrote {args.output} '
        f'(source_sha256={source_sha}, output_sha256={sha256(args.output)}).'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
