"""Contracts shared by frozen-predictor BC, checkpointing and evaluation."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch


READY_HEAD_DEFAULTS = {
    'request_ready_hard_blocking': False,
    'request_ready_context_features': False,
    'request_ready_head_mode': 'shared',
    'request_ready_quantile_head': False,
}


def configure_bc_determinism(enabled):
    if not enabled:
        return
    if torch.cuda.is_initialized():
        raise RuntimeError('BC determinism must be configured before CUDA initialization.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in {':4096:8', ':16:8'}:
        raise ValueError('BC determinism requires a deterministic cuBLAS workspace configuration.')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('[Stage2Determinism] strict deterministic algorithms enabled.', flush=True)


def json_safe(value):
    """Keep absent auxiliary strata JSON-null without accepting invalid scores."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def supervised_optimizer_step(loss, optimizer, *, enabled, max_grad_norm):
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite supervised loss.')
    # Matching-only batches can contain labels but no active matching groups.
    # Ready regression previously hid this constant-loss boundary.
    if not enabled or not loss.requires_grad:
        return loss.new_zeros(()), False
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [param for group in optimizer.param_groups for param in group['params']], max_grad_norm
    )
    if not torch.isfinite(norm):
        optimizer.zero_grad()
        raise RuntimeError('Non-finite supervised gradient norm.')
    optimizer.step()
    return norm, True


def ready_head_config(source):
    """Include shape-independent routing flags, not only tensor shapes."""
    getter = source.get if isinstance(source, dict) else (
        lambda key, default: getattr(source, key, default)
    )
    return {key: type(default)(getter(key, default))
            for key, default in READY_HEAD_DEFAULTS.items()}


def ready_head_signature(source):
    return tuple(ready_head_config(source).values())


def validate_frozen_ready_checkpoint(checkpoint, ac, *, source_sha256,
                                     planning_contract):
    if checkpoint.get('stage2_training_mode') != 'ready_predictor_only':
        raise ValueError('Ready source must be a predictor-only checkpoint.')
    if checkpoint.get('source_m2_sha256') != source_sha256:
        raise ValueError('Ready predictor and resource BC have different Stage1 sources.')
    if checkpoint.get('resource_lookahead_contract') != planning_contract:
        raise ValueError('Ready predictor planning contract differs from resource BC.')
    if ready_head_config(checkpoint) != ready_head_config(ac):
        raise ValueError('Ready predictor architecture/routing metadata mismatch.')
    if float(checkpoint.get('request_ready_time_scale', -1)) != float(ac.request_ready_time_scale):
        raise ValueError('Ready predictor time scale mismatch.')
    before = checkpoint.get('protected_parameter_summary_before_bc')
    if not before or before != checkpoint.get('protected_parameter_summary_after_bc'):
        raise ValueError('Ready source lacks unchanged protected-parameter evidence.')
    resources = checkpoint.get('resource_actor_summary_before_bc')
    if not resources or resources != checkpoint.get('resource_actor_summary_after_bc'):
        raise ValueError('Ready source changed its frozen resource actors.')
    prefix = 'request_ready_head.'
    state = {key[len(prefix):]: value for key, value in checkpoint['model'].items()
             if key.startswith(prefix)}
    expected = ac.request_ready_head.state_dict()
    if set(state) != set(expected):
        raise ValueError('Ready predictor tensor keys mismatch.')
    for key, value in state.items():
        if value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid ready predictor tensor: {key}')
    return state


def supervised_gate(pre, post, *, max_raw_regression, max_stress_regression):
    """Fail closed on invalid evaluation, including NaN comparisons."""
    if not isinstance(pre, dict) or not isinstance(post, dict):
        raise ValueError('Supervised selection requires pre/post evaluation metrics.')
    reasons = []
    raw_pre, raw_post = (float(row.get('eval_raw_makespan', float('nan')))
                         for row in (pre, post))
    stress_pre, stress_post = (float(row.get('eval_distribution_ood_stress_makespan', float('nan')))
                               for row in (pre, post))
    valid = (post.get('eval_valid') == 1.0
             and post.get('eval_completion_rate') == 1.0
             and post.get('eval_cycle_count') == 0
             and post.get('eval_timeout_count') == 0)
    if not valid or not all(math.isfinite(x) for x in (raw_pre, raw_post)):
        reasons.append('invalid_or_incomplete_validation')
    if raw_post - raw_pre > max_raw_regression:
        reasons.append('raw_validation_regression')
    if math.isfinite(max_stress_regression):
        if not all(math.isfinite(x) for x in (stress_pre, stress_post)):
            reasons.append('missing_ood_stress_metric')
        elif stress_post - stress_pre > max_stress_regression:
            reasons.append('ood_stress_regression')
    return json_safe({
        'passed': not reasons, 'reasons': reasons,
        'pre_raw_makespan': raw_pre, 'post_raw_makespan': raw_post,
        'raw_delta_seconds': raw_post - raw_pre,
        'pre_ood_stress_makespan': stress_pre,
        'post_ood_stress_makespan': stress_post,
        'ood_stress_delta_seconds': stress_post - stress_pre,
        'completion_rate': post.get('eval_completion_rate'),
        'cycle_count': post.get('eval_cycle_count'),
        'timeout_count': post.get('eval_timeout_count'),
        'max_raw_regression_seconds': max_raw_regression,
        'max_stress_regression_seconds': max_stress_regression,
    })


def portable_rng_state(state):
    """Encode NumPy state as builtins so weights-only checkpoint loading works."""
    result = copy.deepcopy(state)
    numpy_state = result['numpy']
    if not isinstance(numpy_state, dict):
        result['numpy'] = {
            'name': str(numpy_state[0]), 'keys': numpy_state[1].tolist(),
            'position': int(numpy_state[2]), 'has_gauss': int(numpy_state[3]),
            'cached_gaussian': float(numpy_state[4]),
        }
    return result


def capture_rng_state():
    return portable_rng_state({
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    })


def restore_rng_state(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    if isinstance(numpy_state, dict):
        numpy_state = (numpy_state['name'], np.asarray(numpy_state['keys'], dtype=np.uint32),
                       numpy_state['position'], numpy_state['has_gauss'], numpy_state['cached_gaussian'])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def choose_best_epoch(records):
    eligible = [row for row in records if row['gate']['passed']]
    return copy.deepcopy(min(eligible, key=lambda row: (
        row['metrics']['eval_raw_makespan'], row['epoch']
    ))) if eligible else None


def compare_reference_cases(expected, actual, *, case_count, tolerance_seconds):
    """Require a complete hash-paired replay, not merely a similar mean."""
    def index(rows):
        result = {}
        for row in rows:
            key = row.get('case_sha256')
            value = float(row.get('makespan', float('nan')))
            if (not key or key in result or not math.isfinite(value)
                    or not row.get('completed') or row.get('cycle_terminated')
                    or row.get('timeout')):
                raise ValueError('Reference replay has invalid, duplicate or incomplete cases.')
            result[key] = value
        if len(result) != case_count:
            raise ValueError('Reference replay case count differs from the declared contract.')
        return result

    left, right = index(expected), index(actual)
    if set(left) != set(right):
        raise ValueError('Reference replay case hashes differ from the historical evaluation.')
    deltas = {key: right[key] - left[key] for key in sorted(left)}
    mismatches = {key: delta for key, delta in deltas.items()
                  if abs(delta) > tolerance_seconds}
    return {
        'passed': not mismatches, 'case_count': case_count,
        'expected_mean_makespan': sum(left.values()) / case_count,
        'actual_mean_makespan': sum(right.values()) / case_count,
        'max_absolute_case_delta_seconds': max(abs(value) for value in deltas.values()),
        'tolerance_seconds': tolerance_seconds, 'mismatches': mismatches,
    }


def copy_bc_epoch_evidence(source_run, destination_run, completed_epochs):
    """Retain only completed-boundary evidence when recovering into a new run."""
    source_run, destination_run = Path(source_run), Path(destination_run)
    if source_run.resolve() == destination_run.resolve():
        return
    artifacts = [Path('evaluations/pre_supervised.json')]
    for epoch in range(1, completed_epochs + 1):
        artifacts += [Path(f'evaluations/bc_epoch_{epoch}.json'),
                      Path(f'logs/request_ready_cases_epoch{epoch}.json')]
    metrics_name = Path('logs/request_ready_epoch_metrics.jsonl')
    # Validate everything first; never silently drop a prior selected epoch.
    for relative in [*artifacts, metrics_name]:
        if not (source_run / relative).is_file():
            raise FileNotFoundError(f'BC recovery lacks boundary evidence: {source_run / relative}')
        if (destination_run / relative).exists():
            raise FileExistsError(f'BC recovery cannot overwrite evidence: {destination_run / relative}')
    rows = [json.loads(line) for line in (source_run / metrics_name).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if int(row['device_bc_epoch']) <= completed_epochs]
    if [int(row['device_bc_epoch']) for row in rows] != list(range(1, completed_epochs + 1)):
        raise ValueError('BC recovery epoch metrics are incomplete or duplicated.')
    for relative in artifacts:
        target = destination_run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_run / relative, target)
    (destination_run / metrics_name).write_text(
        ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows)
    )
