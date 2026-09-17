#!/usr/bin/env python
"""Evaluate one HKBZ checkpoint through a shared evaluator service.

Despite the historical filename, the request now carries the complete model,
observation, device-head, and resource-planning contract required by Stage2
and Stage3 evaluators as well as Stage1.
"""

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
from onpolicy.utils.training_stage import protected_parameter_summary
from onpolicy.utils.checkpoint_contract import checkpoint_stage1_baseline
from onpolicy.utils.stage2_bc_contract import ready_head_config


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
    model = {
        name: tensor.detach().cpu().clone()
        for name, tensor in payload['model'].items()
    }
    model_sha256 = protected_parameter_summary(
        model, prefixes=('',)
    )['sha256']
    global_feature_mode = str(payload.get('global_feature_mode', 'none'))
    observation_schema_id = str(payload.get('observation_schema_id', ''))
    environment_semantics_version = str(
        payload.get('environment_semantics_version', '')
    )
    device_policy_head_mode = str(
        payload.get('device_policy_head_mode', 'shared')
    )
    ordinary_device_type_count = int(
        payload.get('ordinary_device_type_count', 10)
    )
    stage1_baseline = checkpoint_stage1_baseline(payload)
    device_contract = {
        'device_timing_head': bool(payload.get('device_timing_head', False)),
        'device_global_matching': bool(payload.get('device_global_matching', False)),
        'request_ready_prediction': bool(payload.get('request_ready_prediction', False)),
        'request_ready_time_scale': float(payload.get('request_ready_time_scale', 3600.0)),
        'request_ready_policy_injection': str(payload.get('request_ready_policy_injection', 'learned')),
        'device_resource_adapter': bool(payload.get('device_resource_adapter', False)),
        **ready_head_config(payload),
    }
    lookahead = payload.get('resource_lookahead_contract', {})
    if not isinstance(lookahead, dict):
        lookahead = {}
    resource_planning_config = {
        'resource_release_aware_eta': bool(
            lookahead.get('resource_release_aware_eta', False)
        ),
        'device_future_intent_horizon': int(
            lookahead.get('device_future_intent_horizon', 0)
        ),
        'device_future_intent_mode': str(
            lookahead.get('device_future_intent_mode', 'legacy_one')
        ),
        'device_frontier_max_requests': int(
            lookahead.get('device_frontier_max_requests', 2)
        ),
        'device_request_capacity_per_plane': int(
            lookahead.get('device_request_capacity_per_plane', 0)
        ),
        'device_lookahead_reservation_mode': str(
            lookahead.get('device_lookahead_reservation_mode', 'none')
        ),
        'device_reservation_grace_seconds': float(
            lookahead.get('device_reservation_grace_seconds', 300.0)
        ),
    }
    required_metadata = {
        'observation_schema_id': observation_schema_id,
        'environment_semantics_version': environment_semantics_version,
    }
    missing = [key for key, value in required_metadata.items() if not value]
    if missing:
        raise ValueError(
            f'Checkpoint lacks shared-evaluator metadata {missing}: '
            f'{checkpoint_path}'
        )
    request_id = f'offline-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
    request_checkpoint = args.output.with_suffix(f'.{request_id}.pt')
    request_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **device_contract,
        'protocol_version': PROTOCOL_VERSION,
        'request_id': request_id,
        'model': model,
        'model_sha256': model_sha256,
        'plane_order_mode': payload['plane_order_mode'],
        'plane_pair_decoder': payload['plane_pair_decoder'],
        'stage1_baseline': stage1_baseline,
        'device_policy_head_mode': device_policy_head_mode,
        'ordinary_device_type_count': ordinary_device_type_count,
        'global_feature_mode': global_feature_mode,
        'observation_schema_id': observation_schema_id,
        'environment_semantics_version': environment_semantics_version,
        'resource_planning_config': resource_planning_config,
    }, request_checkpoint)
    try:
        response = SharedEvalClient(
            args.socket_path, timeout_seconds=10800.0
        ).request({
            **device_contract,
            'operation': 'evaluate',
            'request_id': request_id,
            'checkpoint_path': str(request_checkpoint.resolve()),
            'evaluation_label': args.label,
            'evaluation_tau': float(args.evaluation_tau),
            'seed': int(args.seed),
            'cpu_set': args.cpu_set,
            'plane_order_mode': payload['plane_order_mode'],
            'plane_pair_decoder': payload['plane_pair_decoder'],
            'stage1_baseline': stage1_baseline,
            'device_policy_head_mode': device_policy_head_mode,
            'ordinary_device_type_count': ordinary_device_type_count,
            'global_feature_mode': global_feature_mode,
            'observation_schema_id': observation_schema_id,
            'environment_semantics_version': environment_semantics_version,
            'resource_planning_config': resource_planning_config,
            'n_eval_rollout_threads': int(args.n_eval_rollout_threads),
            'model_sha256': model_sha256,
            'max_eval_cases': 0,
        })
        atomic_json(args.output, {
            'status': 'completed',
            'label': args.label,
            'seed': args.seed,
            'checkpoint': str(checkpoint_path),
            'checkpoint_episode': int(payload['episodes']),
            'evaluation_tau': float(args.evaluation_tau),
            'model_sha256': model_sha256,
            'stage1_baseline': stage1_baseline,
            'global_feature_mode': global_feature_mode,
            'observation_schema_id': observation_schema_id,
            'environment_semantics_version': environment_semantics_version,
            'resource_planning_config': resource_planning_config,
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
