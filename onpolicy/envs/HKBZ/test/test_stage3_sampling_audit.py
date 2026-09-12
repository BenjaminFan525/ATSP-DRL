from types import SimpleNamespace

import numpy as np
import pytest
import torch

from onpolicy.utils.stage3_sampling_audit import (ARMS, DecisionStats, action_digest,
    compare_actor_api, history_inputs, restricted_logits, sampling_heads)


def actors():
    from onpolicy.algorithms.utils.ptr_actor import JointPairPtrActor, DeviceRequestPtrActor
    torch.set_num_threads(1)
    torch.manual_seed(35)
    return SimpleNamespace(actor=JointPairPtrActor(8, embed_dim=8),
        device_actor=DeviceRequestPtrActor(8, embed_dim=8, nhead=2),
        transporter_actor=DeviceRequestPtrActor(8, embed_dim=8, nhead=2),
        device_policy_head_mode="shared", device_global_matching=False)


def inputs(role):
    torch.manual_seed(64)
    query = torch.randn(3, 1, 8)
    if role == 0:
        return dict(query=query, op_nodes=torch.randn(3, 2, 8), site_nodes=torch.randn(3, 5, 8),
            op_valid_mask=torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool),
            site_valid_mask=torch.ones(3, 2, 5, dtype=torch.bool), deterministic=False, tau=.3)
    return dict(query=query, request_nodes=torch.randn(3, 5, 8),
        request_valid_mask=torch.tensor([[1, 0, 1, 1, 1], [1, 0, 0, 0, 0], [0, 1, 0, 1, 1]],
                                        dtype=torch.bool), deterministic=False, tau=.3)


def test_topk_legality_ties_normalization_and_noop():
    logits = torch.tensor([[0., 0., 0., float("-inf")], [float("-inf"), 4., float("-inf"), 1.]])
    result = restricted_logits(logits, 2)
    assert torch.isfinite(result).tolist() == [[True, True, False, False], [False, True, False, True]]
    assert torch.allclose(result.exp().sum(-1), torch.ones(2))
    assert restricted_logits(logits, 0) is logits
    assert torch.isfinite(restricted_logits(logits, 50)).sum().item() == 5
    for bad in (torch.tensor([[float("nan"), 0.]]), torch.tensor([[float("-inf")]])):
        with pytest.raises(ValueError):
            restricted_logits(bad, 4)


@pytest.mark.parametrize("role", [0, 1, 2])
def test_native_hook_preserves_outputs_and_rng_and_restores(role):
    from onpolicy.algorithms.utils import ptr_actor
    ac = actors()
    actor = (ac.actor, ac.device_actor, ac.transporter_actor)[role]
    kwargs = inputs(role)
    original_selector = ptr_actor._select_masked_index
    torch.manual_seed(992)
    expected = actor(**kwargs)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(992)
    stats = DecisionStats()
    with sampling_heads(ac, ARMS["full_t030"], stats):
        actual = actor(**kwargs)
    assert all(torch.equal(x, y) for x, y in zip(actual, expected))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert ptr_actor._select_masked_index is original_selector
    assert "forward" not in actor.__dict__
    assert stats.result()[str(role)]["head_calls"] == 3


@pytest.mark.parametrize("arm,greedy_roles", [("plane_t030", [1, 2]), ("resource_t030", [0])])
def test_nonselected_roles_are_greedy_without_rng_draws(arm, greedy_roles):
    ac = actors()
    for role in greedy_roles:
        actor = (ac.actor, ac.device_actor, ac.transporter_actor)[role]
        kwargs = inputs(role)
        expected = actor(**{**kwargs, "deterministic": True})
        before = torch.get_rng_state()
        with sampling_heads(ac, ARMS[arm]):
            actual = actor(**kwargs)
        assert all(torch.equal(x, y) for x, y in zip(actual, expected))
        assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("role", [0, 1, 2])
def test_topk_head_returns_effective_logprob(role):
    ac = actors()
    actor = (ac.actor, ac.device_actor, ac.transporter_actor)[role]
    kwargs = inputs(role)
    with sampling_heads(ac, ARMS["full_top4_t030"]):
        out = actor(**kwargs)
    selected = out[0] * 5 + out[1] if role == 0 else out[0]
    assert torch.isfinite(out[-1]).sum(-1).max() <= 4
    assert torch.allclose(out[-1].exp().sum(-1), torch.ones(3))
    assert torch.equal(out[-2], out[-1].gather(1, selected[:, None]).squeeze(-1))
    assert torch.isfinite(out[-2]).all()


def test_hooks_restore_after_error():
    from onpolicy.algorithms.utils import ptr_actor
    ac = actors()
    selector = ptr_actor._select_masked_index
    with pytest.raises(RuntimeError):
        with sampling_heads(ac, ARMS["full_t030"]):
            raise RuntimeError("test")
    assert ptr_actor._select_masked_index is selector
    assert all("forward" not in actor.__dict__ for actor in (ac.actor, ac.device_actor, ac.transporter_actor))


def test_api_comparison_restores_rng_and_method():
    class Policy:
        def act(self):
            return torch.randint(100, (3,)), torch.zeros(3)

        def get_actions(self, return_decision_mask=True):
            action, hidden = self.act()
            return None, action, None, hidden, None

    policy = Policy()
    torch.manual_seed(925)
    expected = policy.get_actions()[1]
    after = torch.get_rng_state()
    torch.manual_seed(925)
    metrics = {}
    with compare_actor_api(policy, torch.device("cpu"), metrics):
        assert torch.equal(policy.get_actions()[1], expected)
    assert torch.equal(after, torch.get_rng_state())
    assert "get_actions" not in policy.__dict__
    assert metrics == {"calls": 1, "hidden_max_abs": 0., "actions_equal": True, "rng_equal": True}


@pytest.mark.parametrize("different_action,hidden_delta", [(True, 0.), (False, 2e-5)])
def test_api_guard_still_rejects_action_or_material_hidden_mismatch(different_action, hidden_delta):
    class Policy:
        def act(self):
            return torch.tensor([int(different_action)]), torch.tensor([hidden_delta])

        def get_actions(self):
            return None, torch.tensor([0]), None, torch.tensor([0.]), None

    policy = Policy()
    with pytest.raises(RuntimeError, match="mismatch"):
        with compare_actor_api(policy, torch.device("cpu"), {}):
            policy.get_actions()
    assert "get_actions" not in policy.__dict__


def test_history_intervention_does_not_mutate_inputs():
    previous = np.arange(12).reshape(1, 4, 3)
    op, site = np.zeros((1, 4)), np.ones((1, 4))
    assert history_inputs(previous, None, op, site, "authoritative")[0] is op
    new_op, new_site = history_inputs(previous, None, op, site, "previous_issued")
    assert np.array_equal(new_op, previous[:, :, 0])
    assert np.array_equal(new_site, previous[:, :, 1])
    new_op[:] = -1
    assert previous.min() == 0


def test_action_hash_normalizes_legacy_padding():
    a = np.zeros((3, 104, 2), dtype=np.int64)
    b = np.concatenate([a, np.full((3, 104, 1), -1)], -1)
    assert action_digest(a) == action_digest(b)
    b[1, 42, 1] = 1
    assert action_digest(a) != action_digest(b)


def test_save_group_keeps_same_seed_different_cases_and_is_immutable(tmp_path):
    from onpolicy.scripts.train.run_stage3_sampling_audit import save_group
    from onpolicy.utils.stage3_research import read_json
    trajectories = [dict(seed=42, actions=np.full((2, 104, 3), n).tolist(),
                         states=[], completed=True, makespan=10 + n) for n in (0, 1)]
    path = tmp_path / "group.json"
    save_group(path, trajectories, {})
    saved = read_json(path)
    with np.load(saved["trace"]["path"]) as trace:
        assert len(trace.files) == 2
        assert np.all(trace["trajectory_0_seed_42"] == 0)
        assert np.all(trace["trajectory_1_seed_42"] == 1)
    with pytest.raises(FileExistsError):
        save_group(path, trajectories, {})


def test_manifest_identity_covers_cases_arms_and_no_rl():
    from onpolicy.scripts.train.run_stage3_sampling_audit import manifest_identity
    manifest = {"arms": ARMS, "automatic_rl": False, "cases": ["a", "b"]}
    sha = manifest_identity(manifest)
    manifest["manifest_sha256"] = sha
    assert manifest_identity(manifest) == sha
    manifest["automatic_rl"] = True
    assert manifest_identity(manifest) != sha
