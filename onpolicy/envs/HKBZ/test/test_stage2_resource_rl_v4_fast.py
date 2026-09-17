"""Execution equivalence, guard retention, checkpoint and restart budget tests."""
import copy
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData
from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy

from onpolicy.envs.HKBZ.test.test_stage2_resource_rl import TinyModel
from onpolicy.utils.stage2_resource_rl import conditional_regularizers, joint_terms, summary, RESOURCE_PREFIXES
from onpolicy.utils.stage2_resource_rl_v4 import ARMS, DEFAULTS, PROTOCOL
from onpolicy.utils.stage2_resource_rl_v4_fast import device_terms, FastCaseMinibatchLearner, tensor_bytes
from onpolicy.utils.stage2_resource_rl_v4_resume import (
    assert_tree_equal, continuation_budget, pending_evaluations, restore_learner,
)


class FastTests(unittest.TestCase):
    def terms(self, dtype=torch.float32, empty=False):
        torch.manual_seed(4)
        raw = torch.randn(6, 4, 8, dtype=dtype, requires_grad=True)
        old = torch.randn(6, 4, 8, dtype=dtype).log_softmax(-1)
        current = raw.log_softmax(-1)
        mask = torch.rand(6, 4) > .5
        if empty:
            mask[:] = False
        advantage = torch.randn(6, dtype=dtype)
        legacy, stats = joint_terms(current[..., 0], old[..., 0], mask, advantage, .1)
        kl, entropy = conditional_regularizers(current, old, mask)
        fast, fast_kl, fast_entropy, numbers, flags = device_terms(
            current[..., 0], old[..., 0], mask, advantage, .1, current, old)
        self.assertTrue(bool(flags.all()))
        for a, b in ((legacy, fast), (kl, fast_kl), (entropy, fast_entropy)):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(numbers.tolist(), [stats['kl_sum'], stats['clip_count'], stats['joint_count'],
            float(legacy.detach()), float(kl.detach()), float(entropy.detach())])
        original_grad = torch.autograd.grad(legacy + .01 * kl - .001 * entropy, raw, retain_graph=True)[0]
        fast_grad = torch.autograd.grad(fast + .01 * fast_kl - .001 * fast_entropy, raw)[0]
        self.assertTrue(torch.equal(original_grad, fast_grad))

    def test_scalar_and_gradient_bitwise_equivalence(self):
        self.terms()
        self.terms(torch.float64)

    def test_forced_only_frames_keep_empty_valid_reductions(self):
        self.terms(empty=True)

    def test_invalid_support_normalization_and_nan_still_fail_guards(self):
        current = torch.full((2, 2, 3), -np.log(3), dtype=torch.float32)
        mask = torch.ones(2, 2, dtype=torch.bool)
        for change in ('support', 'normalization', 'nan'):
            ref = current.clone()
            ref[0, 0, 0] = float('-inf') if change == 'support' else 10. if change == 'normalization' else float('nan')
            *_, flags = device_terms(current[..., 0], current[..., 0], mask, torch.ones(2), .1, current, ref)
            self.assertFalse(bool(flags.all()))

    def test_sequential_double_accumulation_matches_python(self):
        values = [float(x) for x in torch.randn(2000)]
        total = torch.zeros((), dtype=torch.float64)
        for value in values:
            total += torch.tensor(value, dtype=torch.float64)
        self.assertEqual(float(total), sum(values))

    def learner(self, baseline='rollout'):
        model = TinyModel()
        policy = SimpleNamespace(ac=model, device=torch.device('cpu'),
            load_model_state=lambda state: model.load_state_dict(state))
        args = SimpleNamespace(max_agent_num=1, max_device_num=1, recurrent_N=1, hidden_size=1)
        return FastCaseMinibatchLearner(policy, args, {**DEFAULTS, 'baseline': baseline}, lambda *a, **k: None)

    def checkpoint(self, learner, arm):
        learner.policy.ac.device_actor.weight.sum().backward()
        learner.actor_optim.step()
        learner.actor_steps = 1
        c = dict(protocol=PROTOCOL, arm=arm, training=learner.contract, completed_rounds=1,
            case_episodes=24, actor_steps=1, critic_steps=0, protected_before=learner.protected,
            protected_after=summary(learner.policy.ac, protected=True), actor_before=learner.source_actor,
            actor_after=summary(learner.policy.ac, RESOURCE_PREFIXES), probability_checks=learner.probability_checks,
            minibatch_checks=learner.minibatch_checks)
        return copy.deepcopy(dict(model=learner.policy.ac.state_dict(), resource_rl_contract=c,
            resource_rl_training_state=dict(actor_optim=learner.actor_optim.state_dict(),
                critic_optim=learner.critic_optim.state_dict(), collection_round=1,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.zeros(1, dtype=torch.uint8))))

    def test_restore_keeps_model_optimizer_counters_and_rng(self):
        checkpoint = self.checkpoint(self.learner(), ARMS[0])
        resumed = self.learner()
        report = restore_learner(resumed, checkpoint, ARMS[0])
        self.assertTrue(report['actor_adam_exact'])
        self.assertFalse(report['critic_warmup_repeated'])
        self.assertEqual(resumed.actor_steps, 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), checkpoint['resource_rl_training_state']['torch_rng']))
        checkpoint['resource_rl_training_state']['collection_round'] = 2
        with self.assertRaisesRegex(ValueError, 'complete'):
            restore_learner(resumed, checkpoint, ARMS[0])

    def test_restore_rejects_changed_contract_and_protected_weights(self):
        checkpoint = self.checkpoint(self.learner(), ARMS[0])
        with self.assertRaisesRegex(ValueError, 'contract'):
            restore_learner(self.learner('value'), checkpoint, ARMS[0])
        checkpoint['resource_rl_contract']['protected_after'] = {}
        with self.assertRaisesRegex(ValueError, 'frozen'):
            restore_learner(self.learner(), checkpoint, ARMS[0])

    def test_independent_restores_do_not_share_adam_steps_or_mutate_checkpoint(self):
        checkpoint = self.checkpoint(self.learner(), ARMS[0])
        snapshot = copy.deepcopy(checkpoint)
        first, second = self.learner(), self.learner()
        restore_learner(first, checkpoint, ARMS[0])
        restore_learner(second, checkpoint, ARMS[0])
        p, q = first.policy.ac.device_actor.weight, second.policy.ac.device_actor.weight
        self.assertNotEqual(first.actor_optim.state[p]['step'].data_ptr(), second.actor_optim.state[q]['step'].data_ptr())
        first.actor_optim.zero_grad()
        p.sum().backward()
        first.actor_optim.step()
        self.assertEqual(float(first.actor_optim.state[p]['step']), 2.)
        self.assertEqual(float(second.actor_optim.state[q]['step']), 1.)
        assert_tree_equal(snapshot, checkpoint, 'immutable checkpoint')

    def test_missing_midpoint_evaluation_is_not_skipped_or_relabelled(self):
        self.assertEqual(pending_evaluations(ARMS[0], 10, {}, DEFAULTS), [ARMS[0] + '_round_10'])
        with self.assertRaisesRegex(ValueError, 'earlier'):
            pending_evaluations(ARMS[0], 11, {}, DEFAULTS)
        labels = {ARMS[0] + '_round_10'}
        self.assertEqual(len(pending_evaluations(ARMS[0], 20, labels, DEFAULTS)), 2)

    def test_budget_records_interrupted_attempts_without_extra_optimization(self):
        manifest = dict(training_contract=DEFAULTS, execution=dict(total_case_episode_limit=1708,
            total_training_case_episodes=960))
        checkpoints = {arm: dict(state=dict(completed_rounds=9, case_episodes=216, status='running')) for arm in ARMS}
        evaluations = {'preflight': dict(cases=[None] * 148)}
        budget = continuation_budget(manifest, checkpoints, evaluations, 844)
        self.assertEqual(budget['restart_uncommitted_case_attempts'], 24)
        self.assertEqual(budget['remaining_by_arm'][ARMS[0]], dict(training=264, evaluation=180, total=444))
        self.assertEqual(budget['physical_limit_including_restart'], 1732)
        self.assertEqual(budget['extra_optimization_rounds'], 0)
        with self.assertRaises(ValueError):
            continuation_budget(manifest, checkpoints, evaluations, 1000)

    def test_tensor_accounting_and_restore_tree_reject_changes(self):
        graph = HeteroData()
        graph['node'].x = torch.zeros(2, 3)
        self.assertEqual(tensor_bytes(graph), 24)
        self.assertEqual(tensor_bytes(dict(a=np.zeros(3, dtype=np.float32), b=graph)), 36)
        with self.assertRaises(ValueError):
            assert_tree_equal({'a': torch.ones(1)}, {'a': torch.zeros(1)})

    def test_prepared_inputs_are_reused_without_hidden_state_or_source_mutation(self):
        for cache_limit in (0, 1024 ** 2):
            learner = self.learner()
            learner.cache_max_bytes = cache_limit
            learner.policy._build_inputs = MethodType(GNN_MAPPOPolicy._build_inputs, learner.policy)
            learner.policy._to_tensor = MethodType(GNN_MAPPOPolicy._to_tensor, learner.policy)
            graph = HeteroData()
            graph['node'].x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            original = graph['node'].x.clone()
            frame = dict(obs=[graph.clone(), graph.clone()], active=np.ones((2, 2, 1)),
                history=np.zeros((2, 2, 2), np.int64), types=np.zeros((2, 2), np.int64),
                actions=np.zeros((2, 2, 2), np.int64), old_logp=torch.zeros(2, 2),
                mask=torch.tensor([[False, True]] * 2), reference_logits=torch.zeros(2, 2, 2),
                dones=np.ones((2, 2), bool),
                encoded={'case_group_encodings': {(0, 1): {'global_emb': torch.zeros(2, 3)}}})
            first, second = learner._entry(frame, [0, 1]), learner._entry(frame, [0, 1])
            self.assertEqual(len(learner._prepared), 1)
            self.assertEqual(first['resident'], cache_limit > 0)
            self.assertNotIn('hidden_states', first['data'])
            self.assertTrue(first['ended'])
            self.assertTrue(torch.equal(frame['obs'][0]['node'].x, original))
            self.assertTrue(torch.equal(first['data']['graph']['node'].x, second['data']['graph']['node'].x))
            if cache_limit:
                self.assertIs(first, second)
            else:
                self.assertEqual(learner._resident_bytes, 0)
            learner.clear_replay_cache()
            self.assertFalse(learner._prepared)


if __name__ == '__main__':
    unittest.main()
