"""Versioned Stage2 resource-RL handoff, retaining strict historical BC checks."""
import math
from pathlib import Path

import torch

from onpolicy.utils.training_stage import (
    PROTECTED_RESOURCE_JOINT_PREFIXES, protected_parameter_summary, sha256_file,
    validate_stage2_joint_finetune_checkpoint,
)

RESOURCE = ('device_sel_enc.', 'device_actor.', 'transporter_sel_enc.', 'transporter_actor.')
ALLOWED = RESOURCE + ('team_critic.',)


def validate_frozen_lineage(checkpoint, expected_source=None):
    """Compare actual plane/shared tensors with immutable Stage1, not metadata alone."""
    path = Path(checkpoint.get('source_m2_path', ''))
    digest = checkpoint.get('source_m2_sha256')
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError('Frozen Stage1 source is missing or changed.')
    if expected_source is not None and (
            path.resolve() != Path(expected_source['path']).resolve()
            or digest != expected_source['sha256']):
        raise ValueError('Stage1 lineage differs from the locked manifest.')
    source = torch.load(path, map_location='cpu', weights_only=True)
    model = checkpoint['model']
    original = source['model']
    names = {k for k in original if k.startswith(PROTECTED_RESOURCE_JOINT_PREFIXES)}
    if not names or names != {k for k in model if k.startswith(PROTECTED_RESOURCE_JOINT_PREFIXES)}:
        raise ValueError('Stage1 protected tensor set differs.')
    for name in sorted(names):
        if not torch.equal(model[name].cpu(), original[name].cpu()):
            raise ValueError(f'Stage1 protected tensor changed: {name}')
    return dict(passed=True, source_sha256=digest, tensor_count=len(names),
                protected=protected_parameter_summary(model, prefixes=PROTECTED_RESOURCE_JOINT_PREFIXES))


def validate_rl_handoff(checkpoint, target_state, *, resource_decoder, **kwargs):
    contract = checkpoint.get('resource_rl_contract') or {}
    v4 = contract.get('protocol') == 'stage2_bc_resource_rl_v4'
    allowed_arm = (contract.get('arm') in ('S2RL_PPO_R', 'S2RL_PPO_V') if v4
                   else contract.get('protocol') == 'stage2_bc_resource_rl_v3'
                   and contract.get('arm') == 'S2RL_PPO')
    if (checkpoint.get('training_stage') != 'resource_joint'
            or checkpoint.get('phase') != 'resource_rl_completed'
            or not allowed_arm):
        raise ValueError('Stage3 needs a completed, versioned resource PPO artifact.')
    if (resource_decoder != 'autoregressive'
            or checkpoint.get('resource_deployment_decoder') != 'autoregressive'
            or checkpoint.get('device_global_matching') is not False
            or contract.get('behavior_decoder') != 'autoregressive'
            or contract.get('evaluation_decoder') != 'autoregressive_argmax'):
        raise ValueError('Resource-RL handoff requires explicit identical autoregressive decoding.')
    training = contract.get('training') or {}
    rollout_baseline = v4 and contract.get('arm') == 'S2RL_PPO_R'
    if (int(contract.get('actor_steps', 0)) <= 0
            or (int(contract.get('critic_steps', 0)) != 0 if rollout_baseline
                else int(contract.get('critic_steps', 0)) <= 0)
            or int(contract.get('completed_rounds', 0)) != int(training.get('rounds', -1))
            or int(contract.get('case_episodes', 0)) <= 0
            or training.get('gamma') != 1.0 or training.get('return_scale') != 10000.0
            or contract.get('teacher_execution') is not False or contract.get('cost_queries') != 0):
        raise ValueError('Resource-RL training/effect/reward evidence is incomplete.')
    checks = contract.get('probability_checks') or {}
    logp_error = float(checks.get('max_logp_error', float('inf')))
    ratio_error = float(checks.get('max_ratio_error', float('inf')))
    if (int(checks.get('events', 0)) <= 0 or not math.isfinite(logp_error)
            or not math.isfinite(ratio_error) or not 0 <= logp_error <= 1e-6
            or not 0 <= ratio_error <= 1e-5
            or checks.get('masks_equal') is not True or checks.get('actions_equal') is not True):
        raise ValueError('Resource-RL likelihood checks did not pass.')
    if v4:
        minibatch = contract.get('minibatch_checks') or {}
        errors = [float(minibatch.get(k, float('inf'))) for k in ('max_logp_error', 'max_ratio_error')]
        expected_value = 'not_used_untrained' if rollout_baseline else 'trained_value_baseline'
        if (contract.get('value_head_status') != expected_value
                or contract.get('stage3_requires_fresh_critic') is not True
                or training.get('baseline') != ('rollout' if rollout_baseline else 'value')
                or int(contract.get('case_episodes', 0)) != int(training.get('rounds', 0)) * int(training.get('rollout_cases', 0))
                or int(minibatch.get('events', 0)) <= 0
                or minibatch.get('current_weights_full_prefix') is not True
                or not all(math.isfinite(v) and 0 <= v <= limit for v, limit in zip(errors, (1e-6, 1e-5)))):
            raise ValueError('V4 minibatch/baseline/critic-reset evidence is incomplete.')
    source = contract.get('source') or {}
    source_path = Path(source.get('path', ''))
    if not source_path.is_file() or sha256_file(source_path) != source.get('sha256'):
        raise ValueError('Resource-RL immutable BC source changed or is missing.')
    base = torch.load(source_path, map_location='cpu', weights_only=True)
    if base.get('stage2_training_mode') != 'supervised_only':
        raise ValueError('Resource-RL source must be a historical supervised artifact.')
    baseline = validate_stage2_joint_finetune_checkpoint(base, target_state, **kwargs)
    lineage = validate_frozen_lineage(base)
    model = checkpoint.get('model') or {}
    if set(model) != set(base['model']) or set(model) != set(target_state):
        raise ValueError('Resource-RL model tensor set differs from source/target.')
    for name, tensor in model.items():
        original = base['model'][name]
        if (tensor.shape != original.shape or tensor.dtype != original.dtype
                or not torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid resource-RL tensor: {name}')
        if not name.startswith(ALLOWED) and not torch.equal(tensor.cpu(), original.cpu()):
            raise ValueError(f'Resource-RL modified a protected tensor: {name}')
    for field in ('source_m2_path', 'source_m2_sha256', 'resource_lookahead_contract',
                  'frozen_ready_source', 'plane_order_mode', 'plane_pair_decoder',
                  'global_feature_mode', 'request_ready_time_scale', 'request_ready_policy_injection'):
        if checkpoint.get(field) != base.get(field):
            raise ValueError(f'Resource-RL changed inherited semantic/source field: {field}')
    if not rollout_baseline and not any(not torch.equal(model[k].cpu(), base['model'][k].cpu())
               for k in model if k.startswith('team_critic.')):
        raise ValueError('Resource-RL has no changed team critic tensors.')
    before = protected_parameter_summary(base['model'], prefixes=RESOURCE)
    after = protected_parameter_summary(model, prefixes=RESOURCE)
    protected = protected_parameter_summary(
        {k: v for k, v in model.items() if not k.startswith(ALLOWED)}, prefixes=('',))
    if (before == after or contract.get('actor_before') != before or contract.get('actor_after') != after
            or contract.get('protected_before') != protected or contract.get('protected_after') != protected):
        raise ValueError('Resource-RL bitwise update/freeze evidence differs from actual tensors.')
    # Optimizers are nested archival training state; Stage3 never inherits them.
    if any(checkpoint.get(k) is not None for k in ('actor_optim', 'critic_optim', 'value_normalizer',
            'role_value_normalizers', 'lagrangmdvrpn_optimizer', 'shared_gradient_state')):
        raise ValueError('Do not expose old RL optimizer/value-normalizer state to Stage3.')
    return {**baseline, 'phase': checkpoint['phase'], 'resource_ppo_updated': True,
            'model_summary': protected_parameter_summary(model, prefixes=('',)),
            'resource_decoder': resource_decoder, 'rl_contract': contract,
            'reward_contract': {'mode': 'team_time', 'gamma': 1.0, 'scale': 10000.0},
            'frozen_stage1_verified': lineage,
            'stage3_requires_fresh_critic': bool(v4),
            'value_head_status': contract.get('value_head_status', 'trained_value_baseline'),
            'scientific_goal_confirmed': False}
