#!/usr/bin/env python
"""Evaluate one ordinary HKBZ checkpoint through a shared evaluator service."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from onpolicy.utils.shared_eval import PROTOCOL_VERSION, SharedEvalClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket-path', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cpu-set', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--evaluation-tau', type=float, default=0.3)
    parser.add_argument('--n-eval-rollout-threads', type=int, default=60)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location='cpu')
    request_id = f'offline-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
    request_checkpoint = args.output.with_suffix(f'.{request_id}.pt')
    request_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'protocol_version': PROTOCOL_VERSION,
        'request_id': request_id,
        'model': payload['model'],
        'plane_order_mode': payload['plane_order_mode'],
        'plane_pair_decoder': payload['plane_pair_decoder'],
    }, request_checkpoint)
    try:
        response = SharedEvalClient(
            args.socket_path, timeout_seconds=10800.0
        ).request({
            'operation': 'evaluate',
            'request_id': request_id,
            'checkpoint_path': str(request_checkpoint.resolve()),
            'evaluation_label': args.label,
            'evaluation_tau': float(args.evaluation_tau),
            'seed': int(args.seed),
            'cpu_set': args.cpu_set,
            'plane_order_mode': payload['plane_order_mode'],
            'plane_pair_decoder': payload['plane_pair_decoder'],
            'n_eval_rollout_threads': int(args.n_eval_rollout_threads),
        })
        atomic_json(args.output, {
            'status': 'completed',
            'label': args.label,
            'seed': args.seed,
            'checkpoint': str(checkpoint_path),
            'checkpoint_episode': int(payload['episodes']),
            'evaluation_tau': float(args.evaluation_tau),
            'evaluation_seconds': float(response.get('evaluation_seconds', 0.0)),
            'evaluation': response['evaluation'],
            'completed_unix_time': time.time(),
        })
    finally:
        try:
            request_checkpoint.unlink()
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()
