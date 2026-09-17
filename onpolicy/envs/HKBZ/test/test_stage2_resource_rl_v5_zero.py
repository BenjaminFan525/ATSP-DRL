"""No-environment regressions for the strictly no-grad initialization audit."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from onpolicy.utils.stage2_resource_rl_v5_zero import initial_reference_kernel_context, non_audit_code_sha
from onpolicy.utils.stage2_resource_rl_v5 import SplitEncoderLearner
from onpolicy.utils.stage2_resource_rl_v4_fast import FastCaseMinibatchLearner


def models():
    current = torch.nn.Module()
    current.device_actor = torch.nn.Linear(3, 3)
    current.transporter_actor = torch.nn.Linear(3, 3)
    current.plane = torch.nn.Linear(3, 3)
    current.team_critic = torch.nn.Linear(3, 1)
    for p in current.plane.parameters():
        p.requires_grad_(False)
    current.eval()
    reference = copy.deepcopy(current)
    for p in reference.parameters():
        p.requires_grad_(False)
    return SimpleNamespace(ac=current, bc_reference_ac=reference)


class InitialReferenceTests(unittest.TestCase):
    def test_only_resource_metadata_matches_and_no_graph_or_tensor_changes(self):
        policy = models()
        old = copy.deepcopy(policy.bc_reference_ac.state_dict())
        with torch.no_grad(), initial_reference_kernel_context(policy):
            self.assertTrue(all(p.requires_grad for p in policy.bc_reference_ac.device_actor.parameters()))
            self.assertTrue(all(p.requires_grad for p in policy.bc_reference_ac.transporter_actor.parameters()))
            self.assertFalse(any(p.requires_grad for p in policy.bc_reference_ac.plane.parameters()))
            self.assertFalse(any(p.requires_grad for p in policy.bc_reference_ac.team_critic.parameters()))
            result = policy.bc_reference_ac.device_actor(torch.ones(2, 3))
            self.assertFalse(result.requires_grad)
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in policy.bc_reference_ac.parameters()))
        self.assertTrue(all(torch.equal(old[n], p) for n, p in policy.bc_reference_ac.state_dict().items()))

    def test_exception_always_restores_frozen_reference(self):
        policy = models()
        with self.assertRaisesRegex(ValueError, 'audit failed'):
            with torch.no_grad(), initial_reference_kernel_context(policy):
                raise ValueError('audit failed')
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in policy.bc_reference_ac.parameters()))

    def test_grad_enabled_is_forbidden(self):
        with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, 'no_grad'):
            with initial_reference_kernel_context(models()):
                pass

    def test_training_mode_is_forbidden(self):
        policy = models()
        policy.ac.train()
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'eval-mode'):
            with initial_reference_kernel_context(policy):
                pass

    def test_unfrozen_or_preexisting_gradient_reference_is_rejected(self):
        for kind in ('flag', 'gradient'):
            policy = models()
            p = next(policy.bc_reference_ac.parameters())
            if kind == 'flag':
                p.requires_grad_(True)
            else:
                p.grad = torch.zeros_like(p)
            with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'fully frozen'):
                with initial_reference_kernel_context(policy):
                    pass

    def test_non_audit_signature_rejects_any_learning_change(self):
        source = Path('onpolicy/utils/stage2_resource_rl_v5.py').read_text()
        signature = non_audit_code_sha(source)
        self.assertEqual(signature, non_audit_code_sha(source.replace('error > 1e-6', 'error > 2e-6')))
        self.assertNotEqual(signature, non_audit_code_sha(source.replace("'resource_encoder_lr': 3e-6", "'resource_encoder_lr': 9e-6")))
        with self.assertRaises(ValueError):
            non_audit_code_sha(source.replace('behavior and self.verify_initial_reference', 'behavior'))

    def test_probability_tolerance_is_not_relaxed(self):
        for error, accepted in ((1e-7, True), (1.01e-6, False), (float('nan'), False)):
            policy = models()
            learner = SplitEncoderLearner.__new__(SplitEncoderLearner)
            learner.verify_initial_reference = True
            learner._snapshot_values = learner._record_reference = False
            learner.initial_reference_checks = dict(events=0, max_logp_error=0., max_hidden_error=0.)
            out = dict(log_probs=torch.zeros(1, 1), rnn_states=torch.zeros(1, 1, 1, 3),
                       actions=torch.zeros(1, 1, 2, dtype=torch.long), decision_mask=torch.ones(1, 1, dtype=torch.bool))
            ref = {**out, 'log_probs': torch.full((1, 1), error)}
            with patch.object(FastCaseMinibatchLearner, 'collect_forward', side_effect=[(out, {}), (ref, {})]):
                with torch.no_grad():
                    if accepted:
                        learner.collect_forward(policy, None, None, None, None, None, deterministic=True)
                        self.assertEqual(learner.initial_reference_checks['events'], 1)
                    else:
                        with self.assertRaises((RuntimeError, FloatingPointError)):
                            learner.collect_forward(policy, None, None, None, None, None, deterministic=True)
            self.assertFalse(any(p.requires_grad for p in policy.bc_reference_ac.parameters()))


if __name__ == '__main__':
    unittest.main()
