"""Portable B0 contracts, immutable provenance and fail-closed corruption checks."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from onpolicy.utils.stage2_bc_contract import ready_head_config
from onpolicy.utils.stage2_frozen import FrozenStage2Bundle, PLANNING, PROTOCOL, sha256
from onpolicy.utils.training_stage import STAGE2_SUPERVISION_CONTRACT, protected_parameter_summary


class FrozenStage2Test(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'portable'
        self.root.mkdir()
        self.original = {name: '/unavailable/original/' + name for name in
                         ('b0', 'stage1', 'r1', 'teacher_index')}
        self.paths = {name: self.root / (name + '.pt') for name in ('b0', 'stage1', 'r1')}
        self.paths.update(teacher_index=self.root / 'teacher.json', ac_config=self.root / 'ac.yaml')
        self.paths['teacher_index'].write_text('{}')
        self.paths['ac_config'].write_text('common_cfg: {}\n')
        base = {'encoder.weight': torch.ones(2, 2), 'actor.weight': torch.ones(2, 2),
                'device_actor.weight': torch.ones(2, 2)}
        self.stage1 = {'model': copy.deepcopy(base)}
        torch.save(self.stage1, self.paths['stage1'])
        common = dict(source_m2_path=self.original['stage1'],
            source_m2_sha256=sha256(self.paths['stage1']),
            resource_iga_teacher_index=self.original['teacher_index'],
            resource_lookahead_contract=dict(PLANNING), request_ready_time_scale=3600.,
            **ready_head_config({}))
        common['source_m2_checkpoint'] = dict(path=self.original['stage1'], sha256=common['source_m2_sha256'])
        model = {**base, 'request_ready_head.weight': torch.ones(1, 2),
                 'request_ready_feature.weight': torch.zeros(2, 2)}
        old = copy.deepcopy(model)
        old['request_ready_head.weight'].zero_()
        self.r1 = dict(common, model=copy.deepcopy(model), stage2_training_mode='ready_predictor_only',
            request_ready_total_labels=100,
            request_ready_predictor_summary_before=protected_parameter_summary(
                old, prefixes=('request_ready_head.', 'request_ready_feature.')),
            request_ready_predictor_summary_after=protected_parameter_summary(
                model, prefixes=('request_ready_head.', 'request_ready_feature.')))
        for field in ('protected_parameter_summary', 'resource_actor_summary'):
            self.r1[field + '_before_bc'] = {'sha256': 'frozen'}
            self.r1[field + '_after_bc'] = {'sha256': 'frozen'}
        torch.save(self.r1, self.paths['r1'])
        self.b0 = dict(common, model=copy.deepcopy(model), training_stage='resource_joint',
            phase='resource_supervised_completed', stage2_training_mode='supervised_only',
            stage2_supervision_contract=STAGE2_SUPERVISION_CONTRACT,
            device_bc_training_scope='policy_frozen_ready', device_global_matching=True,
            request_ready_policy_injection='none', request_ready_prediction=True,
            plane_order_mode='fixed', plane_pair_decoder='joint_pair', global_feature_mode='f1f2',
            resource_bc_total_labels=100, request_ready_total_labels=0,
            resource_assignment_total_labels=10, device_bc_categorical_loss_coef=0.,
            resource_actor_summary_before_bc={'sha256': 'before'},
            resource_actor_summary_after_bc={'sha256': 'after'},
            frozen_ready_head_summary=protected_parameter_summary(model, prefixes=('request_ready_head.',)),
            frozen_ready_source=dict(path=self.original['r1'], sha256=sha256(self.paths['r1']),
                teacher_index_sha256=sha256(self.paths['teacher_index'])))
        self.b0['model']['device_actor.weight'] = torch.full((2, 2), 2.)
        torch.save(self.b0, self.paths['b0'])
        self.manifest = dict(protocol=PROTOCOL, scientific_goal_confirmed=False,
            planning_contract=dict(PLANNING), files={key: dict(path=path.name,
                sha256=sha256(path), original_path=self.original.get(key, '/original/ac.yaml'))
                for key, path in self.paths.items()})
        self.manifest_path = self.root / 'manifest.json'
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def rebind_b0(self):
        torch.save(self.b0, self.paths['b0'])
        self.manifest['files']['b0']['sha256'] = sha256(self.paths['b0'])
        self.write_manifest()

    def test_portable_handoff_keeps_original_bytes_and_accepts_frozen_ready(self):
        before = {key: sha256(path) for key, path in self.paths.items()}
        bundle = FrozenStage2Bundle(self.manifest_path)
        result = bundle.validate(self.b0['model'])
        self.assertTrue(result['frozen_ready_source_verified']['verified'])
        self.assertFalse(result['scientific_goal_confirmed'])
        checkpoint = bundle.checkpoint()
        self.assertEqual(checkpoint['source_m2_path'], str(self.paths['stage1']))
        self.assertEqual(self.b0['source_m2_path'], self.original['stage1'])
        self.assertEqual(before, {key: sha256(path) for key, path in self.paths.items()})

    def test_corrupted_artifact_sha_rejected(self):
        self.paths['teacher_index'].write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            FrozenStage2Bundle(self.manifest_path)

    def test_path_escape_and_absolute_path_rejected(self):
        for relative in ('../teacher.json', '/tmp/teacher.json'):
            self.manifest['files']['teacher_index']['path'] = relative
            self.write_manifest()
            with self.assertRaisesRegex(ValueError, 'relative and contained'):
                FrozenStage2Bundle(self.manifest_path)

    def test_symlink_escape_rejected(self):
        external = self.root.parent / 'outside.json'
        external.write_text('{}')
        link = self.root / 'link.json'
        link.symlink_to(external)
        self.manifest['files']['teacher_index']['path'] = link.name
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            FrozenStage2Bundle(self.manifest_path)

    def test_unknown_original_path_rejected_even_after_sha_rebinding(self):
        self.b0['source_m2_path'] = '/unexpected/stage1.pt'
        self.rebind_b0()
        with self.assertRaisesRegex(ValueError, 'original provenance path'):
            FrozenStage2Bundle(self.manifest_path)

    def test_frozen_tensor_change_rejected_even_after_sha_rebinding(self):
        self.b0['model']['encoder.weight'][0, 0] = 9.
        self.rebind_b0()
        with self.assertRaisesRegex(ValueError, 'Frozen Stage1 tensor changed'):
            FrozenStage2Bundle(self.manifest_path).validate()

    def test_planning_contract_change_rejected(self):
        self.b0['resource_lookahead_contract']['device_frontier_max_requests'] = 6
        self.rebind_b0()
        with self.assertRaisesRegex(ValueError, 'deployment contract'):
            FrozenStage2Bundle(self.manifest_path)

    def test_changed_ready_head_rejected_even_after_metadata_rebinding(self):
        self.b0['model']['request_ready_head.weight'][0, 0] = 7.
        self.b0['frozen_ready_head_summary'] = protected_parameter_summary(
            self.b0['model'], prefixes=('request_ready_head.',))
        self.rebind_b0()
        with self.assertRaisesRegex(ValueError, 'changed predictor tensor'):
            FrozenStage2Bundle(self.manifest_path).validate()

    def test_old_optimizer_state_is_not_admitted(self):
        self.b0['actor_optim'] = {'state': {'old': 1}}
        self.rebind_b0()
        with self.assertRaisesRegex(ValueError, 'PPO/critic/ValueNorm'):
            FrozenStage2Bundle(self.manifest_path).validate()

    def test_matching_only_supervision_passes_but_missing_labels_fail(self):
        self.assertTrue(FrozenStage2Bundle(self.manifest_path).validate()['resource_bc_updated'])
        self.b0['resource_assignment_total_labels'] = 0
        self.rebind_b0()
        with self.assertRaisesRegex(ValueError, 'neither final-matching'):
            FrozenStage2Bundle(self.manifest_path).validate()

    def test_target_model_shape_and_dtype_fail_closed(self):
        bundle = FrozenStage2Bundle(self.manifest_path)
        target = copy.deepcopy(self.b0['model'])
        target['device_actor.weight'] = torch.ones(1, 2)
        with self.assertRaisesRegex(ValueError, 'shape/type'):
            bundle.validate(target)
        target['device_actor.weight'] = torch.ones(2, 2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'dtype differs'):
            bundle.validate(target)

    def test_configuration_retains_stage3_budget_and_removes_teachers(self):
        from types import SimpleNamespace
        args = SimpleNamespace(num_episodes=7, lr=0.000123, resource_iga_teacher_dir='/stale')
        args = FrozenStage2Bundle(self.manifest_path).configure_args(args)
        self.assertEqual((args.num_episodes, args.lr), (7, 0.000123))
        self.assertEqual(args.training_stage, 'joint_finetune')
        self.assertEqual(args.resource_iga_teacher_dir, '')
        self.assertEqual(args.device_bc_pretrain_epochs, 0)
        self.assertEqual(args.device_request_capacity_per_plane, 5)

    def test_configuration_preserves_custom_stage3_dataset_and_environment(self):
        from types import SimpleNamespace
        config = self.root / 'stage3.yaml'
        config.write_text('n_agents: 24\nmax_device_num: 80\nresource_policy: drl\ndataset_dir: custom-stage3\n')
        args = SimpleNamespace(env_config=str(config), eval_dataset_dir='stage3-eval')
        args = FrozenStage2Bundle(self.manifest_path).configure_args(args)
        self.assertEqual(args.env_config, str(config))
        self.assertEqual(args.eval_dataset_dir, 'stage3-eval')
        config.write_text('n_agents: 12\nmax_device_num: 80\nresource_policy: drl\n')
        with self.assertRaisesRegex(ValueError, '24-plane/80-device'):
            FrozenStage2Bundle(self.manifest_path).configure_args(args)


if __name__ == '__main__':
    unittest.main()
