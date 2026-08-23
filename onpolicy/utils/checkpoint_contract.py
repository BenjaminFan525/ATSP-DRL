"""Versioned Stage-1 observation and reward contracts.

Tensor-shape compatibility is not sufficient for HKBZ global observations:
``f1f2`` and ``f1f2_departure`` both contain 24 values but assign different
meanings to the final six slots.  This module centralizes fail-closed metadata
checks used by training, shared evaluation, and experiment preflight.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from onpolicy.utils.hkbz_semantics import (
    ENVIRONMENT_SEMANTICS_VERSION,
    global_feature_contract,
)


ACTION_CMAX_REWARD_MODES = frozenset({
    'cmax_delta', 'potential_cmax', 'iga_potential',
})
TEAM_CMAX_REWARD_MODES = frozenset({
    'team_cmax', 'team_time', 'team_time_potential',
    'team_time_resource_potential',
})
STRICT_STAGE1_REWARD_MODES = (
    ACTION_CMAX_REWARD_MODES | TEAM_CMAX_REWARD_MODES
)
STAGE1_REWARD_SCALE = 0.01


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return prefix + hashlib.sha256(encoded).hexdigest()[:20]


def checkpoint_global_feature_mode(checkpoint: Mapping[str, object]):
    value = checkpoint.get('global_feature_mode')
    if value is not None:
        return str(value)
    experiment_config = checkpoint.get('experiment_config', {})
    if isinstance(experiment_config, Mapping):
        value = experiment_config.get('global_feature_mode')
        if value is not None:
            return str(value)
    return None


def stage1_observation_metadata(global_feature_mode: object) -> dict:
    contract = global_feature_contract(str(global_feature_mode))
    return {
        'global_feature_mode': contract['mode'],
        'observation_schema_id': contract['schema_id'],
        'observation_schema': contract,
        'environment_semantics_version': ENVIRONMENT_SEMANTICS_VERSION,
    }


def validate_stage1_checkpoint_contract(
    checkpoint: Mapping[str, object],
    *,
    global_feature_mode: object,
    plane_order_mode: object | None = None,
    plane_pair_decoder: object | None = None,
    strict_metadata: bool = True,
) -> dict:
    """Validate architecture plus observation semantics before loading.

    Mode mismatches are rejected even in compatibility mode.  Compatibility
    mode only permits older artifacts that predate schema/semantics fields.
    New research launchers always request ``strict_metadata=True``.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint payload must be a mapping.')
    expected = stage1_observation_metadata(global_feature_mode)
    observed_mode = checkpoint_global_feature_mode(checkpoint)
    if observed_mode is None:
        if strict_metadata:
            raise ValueError('Checkpoint is missing global_feature_mode.')
    elif observed_mode != expected['global_feature_mode']:
        raise ValueError(
            'Checkpoint global feature semantic mismatch: '
            f'checkpoint={observed_mode!r}, '
            f'configured={expected["global_feature_mode"]!r}.'
        )

    observed_schema = checkpoint.get('observation_schema_id')
    observed_semantics = checkpoint.get('environment_semantics_version')
    missing = []
    if observed_schema is None:
        missing.append('observation_schema_id')
    elif str(observed_schema) != expected['observation_schema_id']:
        raise ValueError(
            'Checkpoint observation schema mismatch: '
            f'checkpoint={observed_schema!r}, '
            f'configured={expected["observation_schema_id"]!r}.'
        )
    if observed_semantics is None:
        missing.append('environment_semantics_version')
    elif str(observed_semantics) != expected['environment_semantics_version']:
        raise ValueError(
            'Checkpoint environment semantics mismatch: '
            f'checkpoint={observed_semantics!r}, '
            f'configured={expected["environment_semantics_version"]!r}.'
        )
    if strict_metadata and missing:
        raise ValueError(
            f'Checkpoint is missing strict contract metadata: {missing}.'
        )

    architecture = {
        'plane_order_mode': plane_order_mode,
        'plane_pair_decoder': plane_pair_decoder,
    }
    for field, configured in architecture.items():
        if configured is None:
            continue
        observed = checkpoint.get(field)
        if observed is None and strict_metadata:
            raise ValueError(f'Checkpoint is missing {field}.')
        if observed is not None and str(observed) != str(configured):
            raise ValueError(
                f'Checkpoint {field} mismatch: checkpoint={observed!r}, '
                f'configured={configured!r}.'
            )
    return {
        **expected,
        'strict_metadata': bool(strict_metadata),
        'checkpoint_global_feature_mode': observed_mode,
        'checkpoint_observation_schema_id': observed_schema,
        'checkpoint_environment_semantics_version': observed_semantics,
    }


def stage1_reward_contract(
    *,
    reward_mode: object,
    reward_coef: object,
    hindsight_cmax_coef: object,
    terminal_cmax_coef: object,
    gamma: object,
    potential_gamma: object,
) -> dict:
    """Validate that every causal arm still optimizes scaled ``-Cmax``."""
    mode = str(reward_mode)
    if mode not in STRICT_STAGE1_REWARD_MODES:
        raise ValueError(
            f'Unsupported strict Stage-1 reward mode {mode!r}.'
        )
    values = {
        'reward_coef': float(reward_coef),
        'hindsight_cmax_coef': float(hindsight_cmax_coef),
        'terminal_cmax_coef': float(terminal_cmax_coef),
        'gamma': float(gamma),
        'potential_gamma': float(potential_gamma),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Stage-1 reward contract contains NaN or Inf.')
    expected_terminal = 0.0 if mode in ACTION_CMAX_REWARD_MODES else 1.0
    expected = {
        'reward_coef': STAGE1_REWARD_SCALE,
        'hindsight_cmax_coef': 1.0,
        'terminal_cmax_coef': expected_terminal,
        'gamma': 1.0,
        'potential_gamma': 1.0,
    }
    incompatible = {
        name: {'configured': values[name], 'expected': expected_value}
        for name, expected_value in expected.items()
        if not math.isclose(
            values[name], expected_value, rel_tol=0.0, abs_tol=1e-12
        )
    }
    if incompatible:
        raise ValueError(
            'Stage-1 reward contract is not an equivalent scaled -Cmax '
            f'objective for mode {mode!r}: {incompatible}.'
        )
    payload = {
        'version': 1,
        'reward_mode': mode,
        **expected,
        'terminal_objective': 'negative_cmax',
    }
    payload['contract_id'] = _stable_id('hkbz-stage1-reward-', payload)
    return payload
