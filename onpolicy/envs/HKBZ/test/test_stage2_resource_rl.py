"""Numerical and scope regressions for the opt-in Stage2 resource RL path."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from onpolicy.utils.stage2_resource_rl import (
    configure_trainability, conditional_regularizers, joint_terms,
    forward, learner_mode, resource_mask, returns_from_complete, summary,
)
from onpolicy.utils.training_stage import validate_stage2_joint_finetune_checkpoint


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device_policy_head_mode = 'shared'
        self.request_ready_policy_injection = 'none'
        for name in ('encoder', 'actor', 'plane_sel_enc', 'request_ready_head',
                     'request_ready_feature', 'device_actor', 'device_sel_enc',
                     'transporter_actor', 'transporter_sel_enc', 'team_critic', 'device_critic'):
            setattr(self, name, torch.nn.Linear(3, 2))


class ResourceRLTests(unittest.TestCase):
    def test_resource_recurrent_graph_does_not_enter_frozen_plane_gru(self):
        from onpolicy.envs.HKBZ.test.test_joint_training import _make_env, _make_policy
        env, policy = _make_env(), _make_policy()
        try:
            policy.ac.request_ready_policy_injection = 'none'
            configure_trainability(policy)
            learner_mode(policy.ac)
            self.assertFalse(policy.ac.plane_sel_enc.seq_encoder.training)
            self.assertTrue(policy.ac.device_sel_enc.seq_encoder.training)
            self.assertTrue(policy.ac.transporter_sel_enc.seq_encoder.training)
            observed = []
            hook = policy.ac.plane_sel_enc.register_forward_hook(
                lambda module, inputs, outputs: observed.extend(t.requires_grad for t in outputs))
            obs, _, info = env.reset()
            hidden = torch.zeros((1, env.n_agents, 1, 64), requires_grad=True)
            active = np.asarray(info['active_agents'])[None, :, None]
            history = np.full((1, env.n_agents, 2), -1, np.int64)
            forward(policy, [obs], hidden, active, history,
                    np.asarray(info['agent_types'])[None], deterministic=True)
            hook.remove()
            self.assertTrue(observed, 'The test must actually execute a plane GRU.')
            self.assertFalse(any(observed))
        finally:
            env.close()

    def test_exact_parameter_whitelist_excludes_ready_shared_plane_and_other_critics(self):
        ac = TinyModel()
        actor, critic = configure_trainability(SimpleNamespace(ac=ac))
        allowed = {id(p) for p in actor + critic}
        self.assertTrue(all(p.requires_grad == (id(p) in allowed) for p in ac.parameters()))
        for name in ('encoder', 'actor', 'plane_sel_enc', 'request_ready_head', 'request_ready_feature', 'device_critic'):
            self.assertFalse(any(p.requires_grad for p in getattr(ac, name).parameters()))
        before = summary(ac, protected=True)
        opt = torch.optim.Adam(actor, lr=.01)
        (ac.device_actor(torch.ones(1, 3)).sum() + ac.transporter_actor(torch.ones(1, 3)).sum()).backward()
        opt.step()
        self.assertEqual(before, summary(ac, protected=True))
        actor_before = summary(ac, ('device_actor.', 'transporter_actor.'))
        opt = torch.optim.Adam(critic, lr=.01)
        ac.team_critic(torch.ones(1, 3)).square().mean().backward()
        opt.step()
        self.assertEqual(actor_before, summary(ac, ('device_actor.', 'transporter_actor.')))

    def test_joint_ratio_excludes_planes_and_forced_slots(self):
        old = torch.zeros((1, 4))
        new = torch.tensor([[100., .1, .2, 100.]], requires_grad=True)
        mask = resource_mask(torch.tensor([[True, True, True, False]]), 1)
        loss, info = joint_terms(new, old, mask, torch.ones(1), clip=1.)
        self.assertAlmostEqual(float(loss), -float(torch.exp(torch.tensor(.3))), places=6)
        loss.backward()
        self.assertEqual(float(new.grad[0, 0]), 0.)
        self.assertEqual(float(new.grad[0, 3]), 0.)
        self.assertEqual(info['joint_count'], 1)

    def test_zero_update_ratio_kl(self):
        logits = torch.log_softmax(torch.randn(3, 4), -1)
        loss, info = joint_terms(logits, logits, torch.ones((3, 4), dtype=torch.bool), torch.ones(3), .1)
        self.assertEqual(float(loss), -3.)
        self.assertEqual(info['kl_sum'], 0.)

    def test_masked_conditional_kl_has_finite_gradients(self):
        logits = torch.tensor([[[0., 1., -torch.inf], [-torch.inf, -torch.inf, -torch.inf]]], requires_grad=True)
        current = torch.cat((torch.log_softmax(logits[:, :1], -1), logits[:, 1:]), 1)
        reference = torch.tensor([[[0.5, 0.5, 0.], [0., 0., 0.]]]).log()
        kl, ent = conditional_regularizers(current, reference, torch.tensor([[True, False]]))
        (kl - .01 * ent).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(kl), 0.)
        bad = reference.clone()
        bad[0, 0, 2] = -3.
        with self.assertRaisesRegex(ValueError, 'masks'):
            conditional_regularizers(current, bad, torch.tensor([[True, False]]))

    def test_small_autoregressive_joint_normalizes_without_duplicate_claims(self):
        # Two resources; each has its private wait action and two real requests.
        # Enumerate conditional supports, then audit the implemented joint sum.
        total = 0.
        for first in range(3):
            legal_second = [j for j in range(3) if j == 0 or j != first]
            for second in legal_second:
                logs = torch.tensor([[-np.log(3), -np.log(len(legal_second))]])
                loss, _ = joint_terms(logs, torch.zeros_like(logs), torch.ones((1, 2), dtype=torch.bool), torch.ones(1), 1.)
                total -= float(loss)
                self.assertFalse(first == second and first > 0)
        self.assertAlmostEqual(total, 1., places=6)

    def test_returns_are_complete_undiscounted_physical_remaining_time(self):
        np.testing.assert_allclose(returns_from_complete([100., 200.], [[0., 0.], [20., 50.]], [True, True], 100.),
                                   [[-1., -2.], [-.8, -1.5]])
        for complete, time_values in [([False, True], [[0., 0.]]), ([True, True], [[101., 0.]])]:
            with self.assertRaises(ValueError):
                returns_from_complete([100., 200.], time_values, complete)

    def test_nonfinite_probabilities_are_not_sanitized(self):
        with self.assertRaises(FloatingPointError):
            joint_terms(torch.tensor([[float('nan')]]), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool), torch.ones(1), .1)

    def test_rl_handoff_cannot_masquerade_as_supervised_or_change_decoder(self):
        checkpoint = dict(stage2_training_mode='bc_resource_rl', training_stage='resource_joint',
            phase='resource_rl_completed', resource_rl_contract=dict(protocol='stage2_bc_resource_rl_v3', arm='S2RL_PPO'))
        with self.assertRaisesRegex(ValueError, 'decoding'):
            validate_stage2_joint_finetune_checkpoint(checkpoint, {}, plane_order_mode='fixed',
                plane_pair_decoder='joint_pair', global_feature_mode='f1f2', resource_decoder='hungarian')
        checkpoint['phase'] = 'resource_rl_running'
        with self.assertRaisesRegex(ValueError, 'completed'):
            validate_stage2_joint_finetune_checkpoint(checkpoint, {}, plane_order_mode='fixed',
                plane_pair_decoder='joint_pair', global_feature_mode='f1f2', resource_decoder='autoregressive')

    def test_rl_handoff_rejects_nonfinite_probability_evidence(self):
        checkpoint = dict(stage2_training_mode='bc_resource_rl', training_stage='resource_joint',
            phase='resource_rl_completed', resource_deployment_decoder='autoregressive',
            device_global_matching=False, resource_rl_contract=dict(
                protocol='stage2_bc_resource_rl_v3', arm='S2RL_PPO',
                behavior_decoder='autoregressive', evaluation_decoder='autoregressive_argmax',
                actor_steps=2, critic_steps=10, completed_rounds=2, case_episodes=48,
                teacher_execution=False, cost_queries=0,
                training=dict(rounds=2, gamma=1.0, return_scale=10000.0),
                probability_checks=dict(events=1, max_logp_error=float('nan'), max_ratio_error=0.,
                                        masks_equal=True, actions_equal=True)))
        with self.assertRaisesRegex(ValueError, 'likelihood'):
            validate_stage2_joint_finetune_checkpoint(checkpoint, {}, plane_order_mode='fixed',
                plane_pair_decoder='joint_pair', global_feature_mode='f1f2', resource_decoder='autoregressive')


if __name__ == '__main__':
    unittest.main()
