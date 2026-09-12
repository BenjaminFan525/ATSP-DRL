"""Focused contracts for the Stage3 counterfactual/HAPPO screen."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import (
    GNN_Actor_Critic,
)


class _MaskValueCritic(nn.Module):
    """Return selected operation id plus selected-site id / 10."""

    def forward(
        self, query, op_nodes, site_nodes, op_pad_mask=None,
        site_pad_mask=None,
    ):
        op_id = (~op_pad_mask).float().argmax(dim=1).float()
        site_id = (
            torch.zeros_like(op_id)
            if site_pad_mask is None
            else (~site_pad_mask).float().argmax(dim=1).float()
        )
        return (op_id + site_id / 10.0).view(-1, 1, 1)


def _counterfactual_owner(topk=8, mix=1.0):
    owner = SimpleNamespace(
        counterfactual_q_topk=topk,
        counterfactual_q_min_mass=0.9,
        counterfactual_baseline_mix=mix,
        _counterfactual_represented_masses=[],
    )
    for name in (
        '_counterfactual_step_value',
        'set_counterfactual_baseline_mix',
        'reset_counterfactual_diagnostics',
        'consume_counterfactual_diagnostics',
    ):
        setattr(owner, name, getattr(GNN_Actor_Critic, name).__get__(owner))
    return owner


def test_counterfactual_q_replay_and_expectation():
    owner = _counterfactual_owner(topk=2)
    critic = _MaskValueCritic()
    query = torch.zeros(1, 3)
    ops = torch.zeros(1, 2, 2)
    sites = torch.zeros(1, 2, 2)
    log_probs = torch.log(torch.tensor([[0.75, 0.0, 0.0, 0.25]]))

    expected = owner._counterfactual_step_value(
        critic,
        query=query,
        op_nodes=ops,
        site_nodes=sites,
        legal_log_probs=log_probs,
        n_sites=2,
    )
    replayed = owner._counterfactual_step_value(
        critic,
        query=query,
        op_nodes=ops,
        site_nodes=sites,
        legal_log_probs=log_probs,
        chosen_action=torch.tensor([3]),
        n_sites=2,
    )
    assert torch.allclose(expected, torch.tensor([0.275]))
    assert torch.allclose(replayed, torch.tensor([1.1]))


def test_counterfactual_topk_renormalizes_represented_mass():
    owner = _counterfactual_owner(topk=1)
    critic = _MaskValueCritic()
    expected = owner._counterfactual_step_value(
        critic,
        query=torch.zeros(1, 3),
        op_nodes=torch.zeros(1, 3, 2),
        site_nodes=torch.zeros(1, 1, 2),
        legal_log_probs=torch.log(torch.tensor([[0.2, 0.7, 0.1]])),
    )
    assert torch.allclose(expected, torch.tensor([1.0]))


def test_counterfactual_mix_scales_only_rollout_baseline_and_audits_mass():
    owner = _counterfactual_owner(topk=1, mix=0.25)
    critic = _MaskValueCritic()
    common = dict(
        critic=critic,
        query=torch.zeros(1, 3),
        op_nodes=torch.zeros(1, 3, 2),
        site_nodes=torch.zeros(1, 1, 2),
        legal_log_probs=torch.log(torch.tensor([[0.2, 0.7, 0.1]])),
    )
    rollout = owner._counterfactual_step_value(**common)
    replay = owner._counterfactual_step_value(
        **common, chosen_action=torch.tensor([2])
    )
    assert torch.allclose(rollout, torch.tensor([0.25]))
    assert torch.allclose(replay, torch.tensor([2.0]))
    diagnostics = owner.consume_counterfactual_diagnostics()
    assert diagnostics['counterfactual_topk_mass_count'] == 1.0
    assert np.isclose(diagnostics['counterfactual_topk_mass_mean'], 0.7)
    assert diagnostics['counterfactual_topk_mass_target_fraction'] == 0.0
    assert owner._counterfactual_represented_masses == []


def test_sequential_factor_ess_contract():
    factors = torch.tensor([1.0, 1.0, 1.0, 1.0])
    ess = factors.sum().square() / (factors.numel() * factors.square().sum())
    assert np.isclose(float(ess), 1.0)
    skewed = torch.tensor([2.0, 0.5, 0.5, 0.5])
    skewed_ess = (
        skewed.sum().square() / (skewed.numel() * skewed.square().sum())
    )
    assert 0.5 < float(skewed_ess) < 1.0
