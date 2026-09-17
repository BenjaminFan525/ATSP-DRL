"""Opt-in exact host matching for frozen cost actors, not a new policy.

The float64 LAP solver, row/column order and score tensors are unchanged.
Only transfers and independent output writes are batched. The ordinary actor
class remains the reference implementation used by trajectory verification.
"""
from types import MethodType

import numpy as np
import torch

from onpolicy.utils.stage2_matching import solve_resource_matching


INFERENCE_CONTRACT = 'full_batch_exact_host_matching_v1'


def apply_batched_matching(self, resource_action_logits, request_is_lookahead,
                           active_agents, op_choice, log_prob, decision_validated):
    if resource_action_logits is None:
        raise RuntimeError('Global resource matching requires per-device action logits.')
    # Convert AFTER the CPU copy, just like the reference; no CUDA arithmetic
    # is changed. Inactive rows are never passed to the solver or validated.
    scores = resource_action_logits.detach().cpu().double().numpy()
    lookahead = request_is_lookahead.detach().cpu().numpy()
    active = active_agents.detach().cpu().numpy().astype(bool)
    batch_ids, agent_ids, request_ids = [], [], []
    for batch in range(scores.shape[0]):
        rows = np.flatnonzero(active[batch, self.max_plane_agents:]) + self.max_plane_agents
        if not rows.size:
            continue
        assignments = solve_resource_matching(scores[batch, rows], lookahead[batch])
        batch_ids.extend([batch] * len(rows))
        agent_ids.extend(rows.tolist())
        request_ids.extend(assignments.tolist())
    if not batch_ids:
        return
    indices = torch.as_tensor([batch_ids, agent_ids, request_ids],
                              dtype=torch.long, device=resource_action_logits.device)
    batches, agents, requests = indices.unbind(0)
    # Every (batch, agent) pair is unique. This also retains the reference
    # selected-score gradient, although production installs only on no-grad
    # private cost actors, never on the learner or the serial proof actor.
    op_choice[batches, agents] = requests
    log_prob[batches, agents] = resource_action_logits[batches, agents, requests]
    decision_validated[batches, agents] = True


def install_frozen_matching(policy):
    if policy.ac.training or any(p.requires_grad for p in policy.ac.parameters()):
        raise ValueError('Fast cost matching is restricted to frozen evaluation actors.')
    policy.ac._apply_device_global_matching = MethodType(apply_batched_matching, policy.ac)


def install_frozen_inference(policy):
    """Named extension point; only the measured matching optimization is used."""
    install_frozen_matching(policy)
