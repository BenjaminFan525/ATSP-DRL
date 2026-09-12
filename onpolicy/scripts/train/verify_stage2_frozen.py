#!/usr/bin/env python3
"""CPU-only strict frozen Stage2 bundle verification; no rollouts or training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.utils.checkpoint_contract import validate_stage1_checkpoint_contract
from onpolicy.utils.stage2_frozen import load_frozen_stage2, sha256
from onpolicy.utils.training_stage import protected_parameter_summary


def _verify_once(manifest_path):
    torch.set_num_threads(1)
    checkpoint, args, bundle = load_frozen_stage2(manifest_path)
    policy = GNN_MAPPOPolicy(args, yaml.safe_load(Path(args.ac_config).read_text()),
                            device=torch.device('cpu'))
    validation = bundle.validate(policy.ac.state_dict())
    observation = validate_stage1_checkpoint_contract(checkpoint,
        global_feature_mode=args.global_feature_mode, plane_order_mode=policy.ac.plane_order_mode,
        plane_pair_decoder=policy.ac.plane_pair_decoder, stage1_baseline=policy.ac.stage1_baseline,
        strict_metadata=True)
    # Exercise the actual production Stage3 boundary without initializing an
    # environment, logger, rollout collector, trainer or data loader.
    from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
    calls = []
    def env_call(name, value):
        if name != 'set_resource_lateness_coef':
            raise AssertionError(f'Unexpected environment action: {name}')
        calls.append(name)
        return [value]
    runner = object.__new__(HKBZ_Runner)
    runner.policy, runner.all_args, runner.device = policy, args, torch.device('cpu')
    runner.stage3_handoff_mode = 'strict'
    runner.envs = SimpleNamespace(call=env_call)
    log = io.StringIO()
    with redirect_stdout(log):
        runner._restore_stage2_joint_finetune(str(bundle.path('b0')))
    if protected_parameter_summary(policy.ac.state_dict(), prefixes=('',)) != validation['model_summary']:
        raise ValueError('Frozen B0 model did not load bit for bit.')
    if policy.actor_optimizer.state or policy.critic_optimizer.state:
        raise ValueError('Stage3 optimizer state must be fresh.')
    if (runner._pending_value_normalizer_state is not None
            or runner._pending_role_value_normalizer_states is not None
            or calls != ['set_resource_lateness_coef']):
        raise ValueError('Stage3 reset/stub contract differs.')
    validation = {**validation, 'model_summary': {
        key: value for key, value in validation['model_summary'].items()
        if key != 'parameters'}}
    return dict(passed=True, device='cpu', actor_updates=0, critic_updates=0,
        teacher_queries=0, rollout_episodes=0, source_files_modified=False,
        production_runner_handoff=True, environment_rollout_verified=False,
        fresh_stage3_optimizers=True, observation_contract=observation,
        validation=validation, scientific_goal_confirmed=False)


@contextmanager
def _forbid_original_artifact_reads(paths):
    """Test instrumentation only; never remaps paths or model payloads."""
    import builtins
    original_open, original_io_open = builtins.open, io.open
    forbidden = {str(Path(path).resolve()) for path in paths}
    def checked(opener):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, bytes, Path)):
                resolved = str(Path(os.fsdecode(path)).resolve())
                if resolved in forbidden:
                    raise AssertionError(f'Portable handoff opened an original artifact: {resolved}')
            return opener(path, *args, **kwargs)
        return call
    with patch('builtins.open', checked(original_open)), patch('io.open', checked(original_io_open)):
        yield


def verify(manifest_path, *, check_relocation=True):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    original_paths = [manifest['files'][key]['original_path']
                      for key in ('b0', 'stage1', 'r1', 'teacher_index')]
    before = {key: sha256(manifest_path.parent / value['path'])
              for key, value in manifest['files'].items()}
    with _forbid_original_artifact_reads(original_paths):
        report = _verify_once(manifest_path)
        report['original_artifact_reads_forbidden'] = True
        if check_relocation:
            with TemporaryDirectory(prefix='stage2-frozen-relocation-') as temporary:
                relocated = Path(temporary) / 'bundle'
                relocated.mkdir()
                for value in manifest['files'].values():
                    target = relocated / value['path']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(manifest_path.parent / value['path'], target)
                shutil.copyfile(manifest_path, relocated / 'manifest.json')
                second = _verify_once(relocated / 'manifest.json')
                if second['validation']['model_summary'] != report['validation']['model_summary']:
                    raise ValueError('Relocated bundle loaded different model weights.')
                report['relocated_bundle_verified'] = True
    after = {key: sha256(manifest_path.parent / value['path'])
             for key, value in manifest['files'].items()}
    if before != after:
        raise ValueError('Verification modified source bundle bytes.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
        default=ROOT / 'artifacts/stage2_frozen/20260912/manifest.json')
    parser.add_argument('--skip-relocation', action='store_true',
        help='Skip the temporary-copy portability check; never changes source files.')
    cli = parser.parse_args()
    print(json.dumps(verify(cli.manifest, check_relocation=not cli.skip_relocation),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
