"""Fail-closed candidate, objective, snapshot and research-gate regressions."""
import copy
import gzip
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from onpolicy.config.config import get_config
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.utils.stage2_cost_improvement import (
    apply_cost_evidence, cost_preference_loss, joint_candidates, validate_joint,
)


class CostImprovementTest(unittest.TestCase):
    def test_disabled_by_default(self):
        args = get_config().parse_args([])
        self.assertFalse(args.stage2_cost_improvement)
        self.assertFalse(args.stage2_policy_improvement_protocol)

    def test_incumbent_first_distinct_and_bounded(self):
        legal = np.ones((3, 5), bool)
        result = joint_candidates(legal, [1, 2, 0], [2, 1, 3], np.ones(5, bool))
        self.assertLessEqual(len(result), 4)
        self.assertEqual(result[0]['origin'], 'student')
        self.assertEqual(result[0]['action'].tolist(), [1, 2, 0])
        self.assertEqual(len({tuple(x['action']) for x in result}), len(result))
        for row in result:
            validate_joint(legal, row['action'], np.ones(5, bool))

    def test_global_claim_not_separate_resource_types(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_joint(np.ones((2, 3), bool), [1, 1], np.ones(3, bool))

    def test_blocking_priority_and_serial_rule_both_required(self):
        with self.assertRaisesRegex(ValueError, 'Blocking'):
            validate_joint(np.ones((1, 2), bool), [0], [True, False])
        # Maximum coverage can be one but this row is last eligible for TWO
        # requests; consuming one at an earlier row can violate serial replay.
        legal = np.array([[1, 1, 1], [1, 1, 0]], bool)
        with self.assertRaises(ValueError):
            validate_joint(legal, [1, 0], [True, False, False])

    def test_invalid_incumbent_or_teacher_fails_not_silent_repair(self):
        with self.assertRaises(ValueError):
            joint_candidates(np.ones((1, 2), bool), [0], [1], [True, False])

    def test_wait_better_and_dispatch_better_have_opposite_gradients(self):
        gradients = []
        for costs in ([100., 160.], [160., 100.]):
            scores = torch.zeros((1, 2), requires_grad=True)
            loss, metrics = cost_preference_loss(scores, [[0], [1]], costs)
            loss.backward()
            gradients.append(scores.grad)
            self.assertEqual(metrics['pairs'], 1)
        torch.testing.assert_close(gradients[0], -gradients[1])
        self.assertLess(float(gradients[0][0, 0]), 0.)

    def test_tied_identity_does_not_receive_gradient(self):
        scores = torch.tensor([[0., 2., 0.], [0., 0., 2.]], requires_grad=True)
        loss, metrics = cost_preference_loss(scores, [[1, 2], [2, 1]], [100., 100.])
        loss.backward()
        self.assertEqual(metrics['pairs'], 0)
        self.assertEqual(float(scores.grad.abs().sum()), 0.)

    def test_incomplete_or_nan_abstains_entire_joint(self):
        for costs in ([100., None], [100., float('nan')], [100., 0.]):
            loss, metrics = cost_preference_loss(torch.zeros((1, 2)), [[0], [1]], costs)
            self.assertFalse(metrics['covered'])
            self.assertEqual(float(loss), 0.)

    def test_whole_joint_suppression_including_ties(self):
        logits = torch.tensor([[[0., 2., 0.], [0., 0., 2.]],
                               [[0., 1., 0.], [0., 0., 1.]]], requires_grad=True)
        evidence = [{'env': 0, 'rows': [0, 1], 'lookahead': [True]*3,
                     'candidates': [[1, 2], [2, 1]], 'costs': [100., 100.]}]
        _, anchor, stats = apply_cost_evidence(logits, np.ones((2, 2), bool), evidence,
            scale=120., clip=4., tie_seconds=1.)
        self.assertFalse(anchor[0].any())
        self.assertTrue(anchor[1].all())
        self.assertEqual(stats['device_bc_cost_anchor_suppressed_rows'], 2)

    def test_decoder_escape_rejected(self):
        evidence = [{'env': 0, 'rows': [0], 'lookahead': [True]*3,
                     'candidates': [[0], [1]], 'costs': [100., 120.]}]
        with self.assertRaisesRegex(ValueError, 'escaped'):
            apply_cost_evidence(torch.tensor([[[0., 0., 3.]]]), np.ones((1, 1), bool),
                evidence, scale=120., clip=4., tie_seconds=1.)

    def test_snapshot_restores_without_touching_live_state(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.total_time, env.current_case_path = 15., 'train/case_test'
        env.nested = {'history': [1, 2]}
        digest = env.stage2_cost_branch_capture()['sha256']
        env.stage2_cost_branch_restore()
        env._stage2_cost_shadow.nested['history'].append(3)
        env._stage2_cost_shadow.total_time = 20.
        self.assertEqual(env.nested['history'], [1, 2])
        self.assertEqual(env.total_time, 15.)
        env.stage2_cost_branch_restore()
        self.assertEqual(env._stage2_cost_shadow.nested['history'], [1, 2])
        self.assertEqual(env.stage2_cost_branch_capture()['sha256'], digest)
        env.stage2_cost_branch_clear()
        with self.assertRaises(RuntimeError):
            env.stage2_cost_branch_restore()

    def test_cost_scale_and_clip_validation(self):
        for scale, clip in [(0., 1.), (float('nan'), 1.), (120., -1.)]:
            with self.assertRaises(ValueError):
                cost_preference_loss(torch.zeros((1, 2)), [[0], [1]], [100., 120.],
                    scale=scale, clip=clip)

    def test_persisted_full_environment_is_replayable_and_never_overwritten(self):
        import cloudpickle
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.total_time, env.current_case_path = 15., 'train/case_test'
        env.nested = {'history': [1, 2], 'mask': np.array([[True, False]])}
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'env.pkl.gz'
            row = env.stage2_cost_branch_capture(str(path))
            raw = gzip.decompress(path.read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
            restored = cloudpickle.loads(raw)
            np.testing.assert_array_equal(restored.nested['mask'], env.nested['mask'])
            self.assertEqual(restored.nested['history'], [1, 2])
            with self.assertRaises(FileExistsError):
                env.stage2_cost_branch_capture(str(path))


if __name__ == '__main__':
    unittest.main()
