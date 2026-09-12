"""Administrative freeze regressions; no training, environment, or GPU work."""

from contextlib import ExitStack
import unittest
from unittest import mock
import warnings

from onpolicy.utils.stage2_freeze_guard import (
    guard_training_stage,
    reject_stage2_development,
)


class Stage2FreezeGuardTest(unittest.TestCase):
    def test_rejection_explains_supported_verification_and_handoff(self):
        with self.assertRaisesRegex(RuntimeError, 'Stage2 development is frozen') as caught:
            reject_stage2_development()
        self.assertIn('verify_stage2_frozen.py', str(caught.exception))
        self.assertIn('STAGE2_FROZEN.md', str(caught.exception))

    def test_all_stage2_aliases_are_frozen_after_normalization(self):
        for stage in ('resource_joint', 'device_bc', 'frozen_joint', ' RESOURCE_JOINT '):
            with self.subTest(stage=stage), warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                with self.assertRaisesRegex(RuntimeError, 'Stage2 development is frozen'):
                    guard_training_stage(stage)

    def test_stage1_stage3_and_unresolved_auto_are_not_frozen(self):
        # The caller must guard the effective stage again after resolving auto.
        for stage in ('plane_pretrain', 'joint_finetune', ' JOINT_FINETUNE ', 'auto', None):
            with self.subTest(stage=stage):
                self.assertIsNone(guard_training_stage(stage))

    def test_retired_and_unknown_stage_names_remain_invalid(self):
        for stage in ('full_joint', 'unrecognized_stage'):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                guard_training_stage(stage)

    def test_resource_rl_entry_rejects_before_manifest_or_model_work(self):
        from onpolicy.scripts.train import run_stage2_resource_rl as script

        with ExitStack() as stack:
            guarded = [stack.enter_context(mock.patch.object(script, name))
                       for name in ('atomic_json', 'make_train_env', 'GNN_MAPPOPolicy', 'seed')]
            with self.assertRaisesRegex(RuntimeError, 'Stage2 development is frozen'):
                script.run(None)
            for operation in guarded:
                operation.assert_not_called()

    def test_manifest_runner_rejects_before_argument_parsing_or_dispatch(self):
        from onpolicy.scripts.train import run_stage2_resource_manifest_trial as script

        with mock.patch.object(script, 'parse_args') as parse, \
                mock.patch.object(script, 'guard_launch') as dispatch, \
                mock.patch.object(script, 'atomic_json') as write:
            with self.assertRaisesRegex(RuntimeError, 'Stage2 development is frozen'):
                script.main()
            parse.assert_not_called()
            dispatch.assert_not_called()
            write.assert_not_called()

    def test_hf_controller_rejects_before_manifest_or_resource_checks(self):
        from onpolicy.scripts.train import run_stage2_bc_hf as script

        with mock.patch.object(script, 'verify_resources') as resources, \
                mock.patch.object(script, 'verify_files') as files, \
                mock.patch.object(script, 'atomic_json') as write:
            with self.assertRaisesRegex(RuntimeError, 'Stage2 development is frozen'):
                script.run_controller(None)
            resources.assert_not_called()
            files.assert_not_called()
            write.assert_not_called()

    def test_pipeline_actual_stage2_rejects_before_planning_or_io(self):
        from onpolicy.scripts.train import run_hkbz_two_stage_pipeline as pipeline

        runner = mock.Mock()
        with mock.patch.object(pipeline, 'plan_stage2') as plan, \
                mock.patch.object(pipeline, 'atomic_json') as write:
            with self.assertRaisesRegex(RuntimeError, 'Stage2 development is frozen'):
                pipeline.run_stage2(
                    None, run_tag='blocked_stage2', manifest_path=None, runner=runner,
                )
            plan.assert_not_called()
            write.assert_not_called()
            runner.assert_not_called()

    def test_teacher_generator_rejects_before_arguments_or_worker_start(self):
        from onpolicy.envs.HKBZ.experiment import generate_stage2_resource_iga_labels as script

        with mock.patch.object(script, 'parse_args') as parse, \
                mock.patch.object(script, 'run_shard') as generate, \
                mock.patch.object(script.mp, 'set_start_method') as workers:
            with self.assertRaisesRegex(RuntimeError, 'Stage2 development is frozen'):
                script.main(None)
            parse.assert_not_called()
            generate.assert_not_called()
            workers.assert_not_called()


if __name__ == '__main__':
    unittest.main()
