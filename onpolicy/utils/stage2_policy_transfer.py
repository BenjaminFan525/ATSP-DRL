"""Strict, weights-only forks of selected Stage2 policies (never resume)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from onpolicy.utils.stage2_bc_contract import compare_reference_cases, ready_head_config
from onpolicy.utils.training_stage import (
    PROTECTED_RESOURCE_JOINT_PREFIXES, STAGE2_SUPERVISION_CONTRACT,
    protected_parameter_summary, source_checkpoint_metadata,
)


FROZEN_POLICY_PREFIXES = PROTECTED_RESOURCE_JOINT_PREFIXES + (
    'plane_critic.', 'device_critic.', 'transporter_critic.', 'team_critic.',
    'request_ready_head.',
)


def validate_ready_target_collection(*, skip, stage, scope, ready_loss, minimum_labels, teacher, teacher_rates):
    if skip and (stage != 'resource_joint' or scope != 'policy_frozen_ready'
                 or ready_loss != 0.0 or minimum_labels != 0 or teacher != 'iga'):
        raise ValueError('Skipping ready targets requires frozen-ready IGA policy-only training, zero ready loss and zero ready-label minimum.')
    if stage == 'resource_joint' and not skip and any(abs(rate - 1.0) > 1e-12 for rate in teacher_rates):
        raise ValueError('Exact intrinsic-ready labels are bound to teacher trajectories; DAgger requires explicitly disabled ready targets with a verified frozen head.')


def verified_frozen_ready_source(checkpoint, *, provenance_resolver=None):
    """Verify actual source weights; projection changes are not predictor training."""
    provenance = checkpoint.get('frozen_ready_source') or {}
    if not provenance.get('path') or not provenance.get('sha256'):
        raise ValueError('Frozen-ready handoff lacks source provenance.')
    metadata = source_checkpoint_metadata(provenance['path'])
    if metadata['sha256'] != provenance['sha256']:
        raise ValueError('Frozen-ready handoff source digest changed.')
    source = torch.load(metadata['path'], map_location='cpu', weights_only=True)
    if provenance_resolver is not None:
        # The portable frozen bundle first binds immutable bytes by SHA, then
        # rebases only known provenance fields in memory. Never rewrite R1.
        source = provenance_resolver.rebase_checkpoint(source)
    if source.get('stage2_training_mode') != 'ready_predictor_only':
        raise ValueError('Frozen-ready source is not a trained predictor artifact.')
    before = source.get('request_ready_predictor_summary_before') or {}
    after = source.get('request_ready_predictor_summary_after') or {}
    before_head = before.get('groups', {}).get('request_ready_head', {}).get('sha256')
    after_head = after.get('groups', {}).get('request_ready_head', {}).get('sha256')
    actual_predictor = protected_parameter_summary(
        source['model'], prefixes=('request_ready_head.', 'request_ready_feature.'))
    if (int(source.get('request_ready_total_labels', 0)) <= 0 or not before_head
            or not after_head or before_head == after_head or after != actual_predictor):
        raise ValueError('Frozen-ready source lacks verified predictor-head training evidence.')
    if source.get('source_m2_sha256') != checkpoint.get('source_m2_sha256'):
        raise ValueError('Frozen-ready source has different Stage1 lineage.')
    for prefix in ('protected_parameter_summary', 'resource_actor_summary'):
        before, after = source.get(prefix + '_before_bc'), source.get(prefix + '_after_bc')
        if not before or before != after:
            raise ValueError('Predictor source changed protected policy tensors.')
    for key, value in ready_head_config(checkpoint).items():
        if ready_head_config(source)[key] != value:
            raise ValueError(f'Frozen-ready source routing differs: {key}.')
    if source.get('request_ready_time_scale') != checkpoint.get('request_ready_time_scale'):
        raise ValueError('Frozen-ready source time scale differs.')
    if source.get('resource_lookahead_contract') != checkpoint.get('resource_lookahead_contract'):
        raise ValueError('Frozen-ready source planning contract differs.')
    teacher = Path(checkpoint.get('resource_iga_teacher_index', ''))
    if not teacher.is_file() or teacher.resolve() != Path(source.get('resource_iga_teacher_index', '')).resolve():
        raise ValueError('Frozen-ready handoff teacher differs.')
    if hashlib.sha256(teacher.read_bytes()).hexdigest() != provenance.get('teacher_index_sha256'):
        raise ValueError('Frozen-ready handoff teacher index digest changed.')
    model = checkpoint['model']
    names = {name for name in model if name.startswith('request_ready_head.')}
    if not names or names != {name for name in source['model'] if name.startswith('request_ready_head.')}:
        raise ValueError('Frozen-ready source head tensor set differs.')
    for name in names:
        if not torch.equal(model[name].cpu(), source['model'][name].cpu()):
            raise ValueError(f'Frozen-ready handoff changed predictor tensor {name}.')
    summary = protected_parameter_summary(model, prefixes=('request_ready_head.',))
    if summary != checkpoint.get('frozen_ready_head_summary'):
        raise ValueError('Frozen-ready handoff head summary is inconsistent.')
    return {**metadata, 'head_sha256': summary['sha256'], 'verified': True,
            'source_ready_labels': int(source['request_ready_total_labels'])}


def validate_policy_warmstart(checkpoint, current_model, *, source_sha256,
                              ready_sha256, planning_contract, injection,
                              head_config, teacher_index):
    if checkpoint.get('training_stage') != 'resource_joint' or checkpoint.get('stage2_training_mode') != 'supervised_only':
        raise ValueError('Policy warm start requires a supervised Stage2 artifact.')
    if checkpoint.get('device_bc_training_scope') != 'policy_frozen_ready':
        raise ValueError('Policy warm start requires a frozen-ready source.')
    if checkpoint.get('stage2_supervision_contract') != STAGE2_SUPERVISION_CONTRACT:
        raise ValueError('Policy warm start supervision contract differs.')
    if not checkpoint.get('stage2_scientific_gate', {}).get('passed') or not checkpoint.get('selected_bc_epoch'):
        raise ValueError('Policy warm start requires the selected eligible Best, not Last.')
    if checkpoint.get('source_m2_sha256') != source_sha256:
        raise ValueError('Policy warm start Stage1 lineage differs.')
    if checkpoint.get('frozen_ready_source', {}).get('sha256') != ready_sha256:
        raise ValueError('Policy warm start predictor source differs.')
    if checkpoint.get('resource_lookahead_contract') != planning_contract:
        raise ValueError('Policy warm start planning contract differs.')
    if checkpoint.get('request_ready_policy_injection') != injection:
        raise ValueError('Policy warm start injection mode differs.')
    if ready_head_config(checkpoint) != head_config:
        raise ValueError('Policy warm start predictor architecture/routing differs.')
    if Path(checkpoint.get('resource_iga_teacher_index', '')).resolve() != Path(teacher_index).resolve():
        raise ValueError('Policy warm start teacher index differs.')
    model = checkpoint.get('model', {})
    if set(model) != set(current_model):
        raise ValueError('Policy warm start model tensor set differs.')
    for name, value in model.items():
        target = current_model[name]
        if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
            raise ValueError(f'Policy warm start invalid tensor: {name}.')
        if name.startswith(FROZEN_POLICY_PREFIXES) and not torch.equal(value.cpu(), target.cpu()):
            raise ValueError(f'Policy warm start changed frozen tensor: {name}.')
    if not checkpoint.get('device_global_matching'):
        raise ValueError('Policy warm start requires the deployed global-matching decoder.')
    verified_frozen_ready_source(checkpoint)
    return protected_parameter_summary(model, prefixes=('',))


def warmstart_replay(expected_path, expected_sha256, actual_rows):
    path = Path(expected_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Policy warm start reference evaluation digest changed.')
    expected = json.loads(path.read_text())['cases']
    return compare_reference_cases(expected, actual_rows, case_count=len(expected), tolerance_seconds=0.0)


def selection_with_fallback(records, baseline):
    """Keep the unchanged source in analysis without hiding failed candidates."""
    candidates = [(0, baseline)] + [(row['epoch'], row['metrics'])
                                    for row in records if row['gate']['passed']]
    if any(not math.isfinite(float(metrics['eval_raw_makespan'])) for _, metrics in candidates):
        raise ValueError('Policy selection cannot accept nonfinite Cmax.')
    epoch, metrics = min(candidates, key=lambda row: (row[1]['eval_raw_makespan'], row[0]))
    return {'epoch': epoch, 'metrics': metrics, 'retained_source': epoch == 0}
