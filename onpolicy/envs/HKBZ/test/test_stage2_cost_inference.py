"""Exact matching outputs/gradients including padding and adversarial ties."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import GNN_Actor_Critic
from onpolicy.utils.stage2_cost_inference import apply_batched_matching, install_frozen_matching


class CostInferenceTest(unittest.TestCase):
    def compare(self, scores, active, lookahead, planes=2):
        actor = SimpleNamespace(max_plane_agents=planes)
        results = []
        for function in (GNN_Actor_Critic._apply_device_global_matching, apply_batched_matching):
            logits = scores.clone().requires_grad_()
            choices = torch.full(scores.shape[:2], -1, dtype=torch.long, device=scores.device)
            log_prob = torch.zeros(scores.shape[:2], device=scores.device)
            valid = torch.zeros(scores.shape[:2], dtype=torch.bool, device=scores.device)
            function(actor, logits, lookahead, active, choices, log_prob, valid)
            if log_prob.requires_grad:
                log_prob.sum().backward()
            results.append((choices, log_prob, valid, logits.grad))
        for left, right in zip(*results):
            if left is None:
                self.assertIsNone(right)
            else:
                self.assertTrue(torch.equal(left, right))

    def test_random_and_tied_matching_with_inactive_padding(self):
        for seed in range(12):
            generator = torch.Generator().manual_seed(seed)
            scores = torch.randint(-3, 1, (24, 12, 9), generator=generator).float()
            scores[torch.rand(scores.shape, generator=generator) < .4] = -torch.inf
            scores[:, :, 0] = 0
            active = torch.rand((24, 12), generator=generator) > .4
            active[0] = False
            # Garbage in inactive padding must not be mistaken for a legal row.
            scores[~active] = torch.nan
            lookahead = torch.rand((24, 9), generator=generator) > .5
            self.compare(scores, active, lookahead)

    def test_no_active_resources_leaves_outputs_untouched(self):
        self.compare(torch.zeros(2, 4, 3), torch.zeros(2, 4, dtype=torch.bool),
                     torch.ones(2, 3, dtype=torch.bool))

    def test_invalid_scores_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                apply_batched_matching(SimpleNamespace(max_plane_agents=0),
                    torch.full((1, 1, 1), invalid), torch.ones(1, 1, dtype=torch.bool),
                    torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long),
                    torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.bool))

    def test_install_rejects_trainable_or_training_actor(self):
        for training in (False, True):
            ac = torch.nn.Linear(2, 2)
            ac.train(training)
            with self.assertRaises(ValueError):
                install_frozen_matching(SimpleNamespace(ac=ac))
        ac.eval().requires_grad_(False)
        install_frozen_matching(SimpleNamespace(ac=ac))
        self.assertIs(ac._apply_device_global_matching.__self__, ac)

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU0 exactness checked separately on host')
    def test_cuda_bitwise_outputs_and_gradients(self):
        self.compare(torch.zeros(24, 10, 8, device='cuda:0'),
                     torch.ones(24, 10, dtype=torch.bool, device='cuda:0'),
                     torch.ones(24, 8, dtype=torch.bool, device='cuda:0'))


if __name__ == '__main__':
    unittest.main()
