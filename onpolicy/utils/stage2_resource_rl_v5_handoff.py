"""Strict, opt-in V5 restore/handoff. Legacy V3/V4 validators are unchanged."""
import math

import torch

from onpolicy.utils.stage2_resource_rl import file_sha, summary
from onpolicy.utils.stage2_resource_rl_handoff import validate_frozen_lineage
from onpolicy.utils.stage2_resource_rl_v5 import (
    ACTOR, VALUE, ARMS, PROTOCOL, architecture_manifest, protected_summary,
)
from onpolicy.utils.training_stage import validate_stage2_joint_finetune_checkpoint


def validate_split_handoff(checkpoint, target_ac, *, require_completed=True):
    contract = checkpoint.get('split_encoder_contract', {})
    arm = contract.get('arm')
    if contract.get('protocol') != PROTOCOL or arm not in ARMS or target_ac.split_encoder_arm != arm:
        raise ValueError('Explicit identical V5 architecture required; no legacy fallback.')
    if require_completed and (checkpoint.get('phase') != 'split_encoder_stage_b_completed'
                             or contract.get('completed_rounds') != 10):
        raise ValueError('Handoff requires the complete predeclared stage B endpoint.')
    source = contract['source']
    if file_sha(source['path']) != source['sha256']:
        raise ValueError('Immutable B0 lineage changed.')
    base = torch.load(source['path'], map_location='cpu', weights_only=True)
    # Validate the original B0 contract against its exact old tensor set; new
    # encoder keys are separately validated below, never accepted by legacy code.
    validate_stage2_joint_finetune_checkpoint(base, base['model'],
        plane_order_mode=target_ac.plane_order_mode, plane_pair_decoder=target_ac.plane_pair_decoder,
        global_feature_mode=base['global_feature_mode'], planning_contract=base['resource_lookahead_contract'],
        request_ready_time_scale=target_ac.request_ready_time_scale)
    validate_frozen_lineage(base)
    model = checkpoint['model']
    if set(model) != set(target_ac.state_dict()):
        raise ValueError('V5 target tensor set differs from checkpoint.')
    expected = set(base['model'])
    for role in ('resource_encoder', 'value_encoder'):
        if hasattr(target_ac, role):
            expected |= {role + n[len('encoder'):] for n in base['model'] if n.startswith('encoder.')}
    if expected != set(model):
        raise ValueError('V5 tensor set has unapproved additions/removals.')
    for name, tensor in model.items():
        original_name = 'encoder.' + name.split('.', 1)[1] if name.startswith(('resource_encoder.', 'value_encoder.')) else name
        original = base['model'][original_name]
        if tensor.shape != original.shape or tensor.dtype != original.dtype or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid V5 tensor {name}')
        if not name.startswith(ACTOR + VALUE) and not torch.equal(tensor.cpu(), original.cpu()):
            raise ValueError(f'Protected B0 tensor changed: {name}')
    for key in ('source_m2_path', 'source_m2_sha256', 'resource_lookahead_contract', 'frozen_ready_source',
                'plane_order_mode', 'plane_pair_decoder', 'global_feature_mode', 'request_ready_time_scale',
                'request_ready_policy_injection'):
        if checkpoint.get(key) != base.get(key):
            raise ValueError(f'Inherited semantic/lineage field changed: {key}')
    if contract['architecture']['encoder_roles'] != architecture_manifest(target_ac)['encoder_roles']:
        raise ValueError('Encoder role map differs.')
    expected_trainable = sorted(n for n in model if n.startswith(ACTOR + VALUE)
                                and n in dict(target_ac.named_parameters()))
    if sorted(contract['architecture']['trainable_names']) != expected_trainable:
        raise ValueError('Checkpoint trainable whitelist differs from approved Stage2 roles.')
    for role in ('encoder', 'resource_encoder', 'value_encoder'):
        if hasattr(target_ac, role):
            actual_hash = summary({n[len(role) + 1:]: v for n, v in model.items() if n.startswith(role + '.')})
            if contract['architecture']['encoders'][role]['hash'] != actual_hash:
                raise ValueError('Checkpoint encoder hash differs from actual tensors.')
    if contract['initial_architecture']['initial_encoder_hash'] != summary(
            {n[len('encoder.'):]: v for n, v in base['model'].items() if n.startswith('encoder.')}):
        raise ValueError('Encoder copy lineage hash differs from B0.')
    actual_protected = summary({n: v for n, v in model.items() if not n.startswith(ACTOR + VALUE)})
    if contract['protected_before'] != actual_protected or contract['protected_after'] != actual_protected:
        raise ValueError('Recorded freeze evidence disagrees with actual tensors.')
    for checks in (contract['probability_checks'], contract['minibatch_checks']):
        if require_completed and checks.get('events', 0) <= 0:
            raise ValueError('Missing probability audits.')
        for key, limit in (('max_logp_error', 1e-6), ('max_ratio_error', 1e-5)):
            value = checks.get(key, math.inf)
            if not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError('Strict probability audit failed.')
    if (checkpoint.get('resource_deployment_decoder') != 'autoregressive'
            or checkpoint.get('device_global_matching') is not False
            or contract.get('stage3_requires_fresh_critic_and_optimizers') is not True):
        raise ValueError('Decoder/Stage3 reset contract missing.')
    if any(checkpoint.get(k) is not None for k in ('actor_optim', 'critic_optim', 'value_normalizer',
                                                 'role_value_normalizers', 'shared_gradient_state')):
        raise ValueError('Stage3 must not inherit optimization/normalization state.')
    if require_completed and (contract['case_episodes'] != 240 or not 0 < contract['actor_steps'] <= 80
                             or contract['critic_steps'] != 700):
        raise ValueError('Stage B update/coverage evidence incomplete.')
    if require_completed:
        checks = contract['gradient_checks']
        if checks['actor_backwards'] < contract['actor_steps'] or checks['critic_updates'] != contract['critic_steps']:
            raise ValueError('Gradient/update evidence incomplete.')
        for role, count in (('resource_encoder', checks['resource_encoder_nonzero']),
                            ('value_encoder', checks['value_encoder_nonzero'])):
            if hasattr(target_ac, role) and (count <= 0 or not any(
                    not torch.equal(model[n].cpu(), base['model']['encoder.' + n.split('.', 1)[1]].cpu())
                    for n in model if n.startswith(role + '.'))):
                raise ValueError('Independent graph encoder did not learn.')
    return dict(passed=True, protocol=PROTOCOL, arm=arm, frozen_lineage_verified=True,
                dynamic_stage3_transfer_verified=False, stage3_started=False,
                scientific_gate_passed=False, stage3_requires_fresh_critic_and_optimizers=True)
