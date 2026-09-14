"""Portable, hash-bound Stage2 B0 handoff; no training or source rewrites.

The original B0/R1/Stage1 bytes remain immutable. Only provenance paths in
loaded dictionaries are rebased, after checking their original path and SHA
bindings against the committed bundle manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from onpolicy.utils.stage2_bc_contract import ready_head_config
from onpolicy.utils.training_stage import (
    PROTECTED_RESOURCE_JOINT_PREFIXES, protected_parameter_summary,
    validate_stage2_joint_finetune_checkpoint,
)

PROTOCOL = 'stage2_frozen_b0_v1'
PLANNING = {
    'device_lookahead_dispatch': True,
    'device_lookahead_safety_margin': 60.0,
    'device_deadline_aware_dispatch': True,
    'device_future_intent_horizon': 2,
    'device_future_intent_mode': 'bounded_frontier',
    'device_frontier_max_requests': 4,
    'resource_release_aware_eta': True,
    'device_lookahead_reservation_mode': 'soft',
    'device_reservation_grace_seconds': 300.0,
    'device_departure_lookahead': True,
}
REQUIRED_FILES = ('b0', 'stage1', 'r1', 'teacher_index', 'ac_config')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class FrozenStage2Bundle:
    """An explicit trusted manifest; never fall back to host-specific paths."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text())
        if (self.manifest.get('protocol') != PROTOCOL
                or self.manifest.get('scientific_goal_confirmed') is not False
                or self.manifest.get('planning_contract') != PLANNING):
            raise ValueError('Frozen Stage2 manifest protocol/goal/planning contract differs.')
        self.files = self.manifest.get('files', {})
        if any(key not in self.files for key in REQUIRED_FILES):
            raise ValueError('Frozen Stage2 bundle is missing required files.')
        self._paths = {}
        for key, entry in self.files.items():
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Bundle path must stay relative and contained: {key}')
            path = (self.root / relative).resolve(strict=True)
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f'Bundle path escapes its root: {key}')
            if sha256(path) != entry.get('sha256'):
                raise ValueError(f'Frozen Stage2 SHA mismatch: {key}')
            self._paths[key] = path
        for key in ('b0', 'stage1', 'r1', 'teacher_index'):
            if not self.files[key].get('original_path'):
                raise ValueError(f'Bundle lacks original provenance: {key}')
        self._raw = {key: torch.load(self.path(key), map_location='cpu', weights_only=True)
                     for key in ('b0', 'stage1', 'r1')}
        # Check original bindings before any in-memory path transformation.
        for key in ('b0', 'r1'):
            self._check_original_bindings(self._raw[key])
        source = self._raw['b0']
        if (source.get('resource_lookahead_contract') != PLANNING
                or source.get('request_ready_policy_injection') != 'none'
                or source.get('device_global_matching') is not True
                or source.get('device_bc_training_scope') != 'policy_frozen_ready'):
            raise ValueError('Frozen B0 deployment contract differs.')

    def path(self, key):
        return self._paths[key]

    def _check_reference(self, value, key, *, allow_rebased=False):
        allowed = {self.files[key]['original_path']}
        if allow_rebased:
            allowed.add(str(self.path(key)))
        if value not in allowed:
            raise ValueError(f'Unexpected original provenance path: {key}')

    def _check_original_bindings(self, checkpoint, *, allow_rebased=False):
        self._check_reference(checkpoint.get('source_m2_path'), 'stage1', allow_rebased=allow_rebased)
        if checkpoint.get('source_m2_sha256') != self.files['stage1']['sha256']:
            raise ValueError('Frozen Stage1 provenance SHA differs.')
        nested = checkpoint.get('source_m2_checkpoint')
        if nested:
            self._check_reference(nested.get('path'), 'stage1', allow_rebased=allow_rebased)
            if nested.get('sha256') != self.files['stage1']['sha256']:
                raise ValueError('Nested frozen Stage1 provenance SHA differs.')
        self._check_reference(checkpoint.get('resource_iga_teacher_index'), 'teacher_index',
                              allow_rebased=allow_rebased)
        ready = checkpoint.get('frozen_ready_source')
        if ready:
            self._check_reference(ready.get('path'), 'r1', allow_rebased=allow_rebased)
            if (ready.get('sha256') != self.files['r1']['sha256']
                    or ready.get('teacher_index_sha256') != self.files['teacher_index']['sha256']):
                raise ValueError('Frozen ready/teacher provenance SHA differs.')

    def rebase_checkpoint(self, checkpoint):
        """Return a metadata copy; never alter tensor values or source files."""
        self._check_original_bindings(checkpoint, allow_rebased=True)
        result = dict(checkpoint)
        result['source_m2_path'] = str(self.path('stage1'))
        if checkpoint.get('source_m2_checkpoint'):
            result['source_m2_checkpoint'] = {
                **checkpoint['source_m2_checkpoint'], 'path': str(self.path('stage1'))}
        result['resource_iga_teacher_index'] = str(self.path('teacher_index'))
        if checkpoint.get('frozen_ready_source'):
            result['frozen_ready_source'] = {
                **checkpoint['frozen_ready_source'], 'path': str(self.path('r1'))}
        return result

    def checkpoint(self):
        return self.rebase_checkpoint(self._raw['b0'])

    def configure_args(self, args=None):
        """Set exact model/planning flags without importing historical launchers.

        Existing Stage3 learning rates, budget and dataset choices are retained.
        Stage2 BC and Ready training are disabled; PPO optimizers start fresh.
        """
        if args is None:
            from onpolicy.config.config import get_config
            args = get_config().parse_args([])
        values = dict(PLANNING)
        values.update(ready_head_config(self._raw['b0']))
        values.update(training_stage='joint_finetune', stage3_handoff_mode='strict',
            resource_policy='drl', max_agent_num=24, max_device_num=80,
            plane_order_mode='fixed', plane_pair_decoder='joint_pair',
            global_feature_mode='f1f2', stage1_baseline='proposed',
            device_policy_head_mode='shared', device_global_matching=True,
            central_team_critic=True, request_ready_prediction=True,
            request_ready_policy_injection='none', request_ready_time_scale=3600.0,
            device_request_capacity_per_plane=5, request_ready_loss_coef=0.0,
            plane_bc_pretrain_epochs=0, device_bc_pretrain_epochs=0,
            evaluation_tau=0.3, anneal_original=0.3, anneal_final=0.3,
            tau_anneal_epochs=0, strict_checkpoint_contract=True,
            stage2_resource_v6_observations=False, stage2_bc_deterministic=True,
            checkpoint_dir=str(self.path('b0')), ac_config=str(self.path('ac_config')),
            stage2_frozen_manifest=str(self.manifest_path))
        # Dataset selection belongs to Stage3. Do not replace a caller's
        # environment YAML (and its train/eval paths) with Stage2 defaults.
        env_config = getattr(args, 'env_config', '')
        if not env_config and 'env_config' in self.files:
            env_config = str(self.path('env_config'))
            values['env_config'] = env_config
        if env_config:
            import yaml
            environment = yaml.safe_load(Path(env_config).read_text()) or {}
            if (environment.get('n_agents') != 24
                    or environment.get('max_device_num') != 80
                    or environment.get('resource_policy') != 'drl'):
                raise ValueError('Stage3 environment must retain frozen 24-plane/80-device DRL dimensions.')
        for field in ('resource_iga_teacher_dir', 'resource_iga_teacher_index',
                      'joint_iga_teacher_dir', 'joint_iga_teacher_index',
                      'plane_bc_teacher_dir', 'plane_bc_teacher_index',
                      'request_ready_checkpoint', 'request_ready_checkpoint_sha256',
                      'stage2_policy_warmstart_checkpoint'):
            values[field] = ''
        for key, value in values.items():
            setattr(args, key, value)
        return args

    def validate(self, target_state=None):
        source = self.checkpoint()
        model = source['model']
        for role in ('b0', 'r1', 'stage1'):
            if not all(torch.isfinite(value).all() for value in self._raw[role]['model'].values()):
                raise ValueError(f'Nonfinite frozen model tensor: {role}')
        original = self._raw['stage1']['model']
        names = {name for name in original if name.startswith(PROTECTED_RESOURCE_JOINT_PREFIXES)}
        if not names:
            raise ValueError('Frozen Stage1 has no protected tensors.')
        for role in ('b0', 'r1'):
            candidate = self._raw[role]['model']
            if names != {name for name in candidate if name.startswith(PROTECTED_RESOURCE_JOINT_PREFIXES)}:
                raise ValueError(f'Frozen Stage1 tensor set differs: {role}')
            for name in sorted(names):
                if (candidate[name].dtype != original[name].dtype
                        or not torch.equal(candidate[name], original[name])):
                    raise ValueError(f'Frozen Stage1 tensor changed: {role}/{name}')
        if target_state is not None:
            for name in model.keys() & target_state.keys():
                if model[name].dtype != target_state[name].dtype:
                    raise ValueError(f'Frozen model dtype differs: {name}')
        validation = validate_stage2_joint_finetune_checkpoint(source,
            model if target_state is None else target_state,
            plane_order_mode='fixed', plane_pair_decoder='joint_pair',
            global_feature_mode='f1f2', planning_contract=PLANNING,
            request_ready_time_scale=3600.0, resource_decoder='hungarian',
            provenance_resolver=self)
        return {**validation, 'bundle_protocol': PROTOCOL,
                'frozen_stage1_tensor_count': len(names), 'scientific_goal_confirmed': False,
                'original_b0_sha256': self.files['b0']['sha256'],
                'model_summary': protected_parameter_summary(model, prefixes=('',))}


def load_frozen_stage2(manifest_path, args=None):
    bundle = FrozenStage2Bundle(manifest_path)
    bundle.validate()
    return bundle.checkpoint(), bundle.configure_args(args), bundle
