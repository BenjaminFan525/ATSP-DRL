"""Independent objective oracle and numerical/identity tests for canonical H."""
from fractions import Fraction
import itertools

import numpy as np
import pytest
import torch

from onpolicy.utils.stage3_canonical_h import solve_canonical_matching
from onpolicy.algorithms.utils.ptr_actor import DeviceRequestPtrActor, JointPairPtrActor


def oracle(scores, look, devices, requests):
    order = sorted(range(len(devices)), key=lambda i: devices[i])
    ranks = {j: k for k, j in enumerate(sorted(range(1, len(requests)), key=lambda j: requests[j])+[0])}
    candidates = []
    for choice in itertools.product(*(np.flatnonzero(np.isfinite(row)) for row in scores)):
        real = [x for x in choice if x]
        if len(real) != len(set(real)):
            continue
        coverage = sum(j != 0 and not look[j] for j in choice)
        value = sum((Fraction.from_float(float(scores[i, j])) for i, j in enumerate(choice)), Fraction())
        tie = tuple(-ranks[choice[i]] for i in order)
        candidates.append(((coverage, value, tie), choice))
    return np.asarray(max(candidates)[1])


def test_exact_objective_against_exhaustive_oracle_and_permutations():
    rng = np.random.default_rng(813)
    for trial in range(180):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 6))
        scores = rng.integers(-8, 9, size=(n, m)).astype(np.float32)/8
        if trial % 3 == 0:
            scores = np.nextafter(scores, np.float32(np.inf))
        mask = rng.random((n, m)) < .25; mask[:, 0] = False
        scores[mask] = -np.inf
        look = rng.integers(0, 2, size=m).astype(bool)
        devices = rng.permutation(n)+100
        requests = np.r_[0, rng.permutation(m-1)+10]
        expected = oracle(scores, look, devices, requests)
        actual = solve_canonical_matching(scores, look, device_ids=devices, request_ids=requests)
        assert np.array_equal(actual, expected)
        rows = rng.permutation(n); columns = np.r_[0, rng.permutation(m-1)+1]
        changed = solve_canonical_matching(scores[rows][:, columns], look[columns],
            device_ids=devices[rows], request_ids=requests[columns])
        restored = np.empty(n, dtype=np.int64); restored[rows] = columns[changed]
        assert np.array_equal(restored, expected)


def test_one_ulp_is_not_an_epsilon_tie_and_blocking_is_primary():
    raw = np.array([[-.8458806276321411, -.6777406930923462],
                    [-.8458806276321411, -.6777406334877014]], dtype=np.float32)
    assert solve_canonical_matching(raw, [False, True], device_ids=[42, 45]).tolist() == [0, 1]
    assert solve_canonical_matching([[100, -100]], [False, False], device_ids=[1]).tolist() == [1]
    assert solve_canonical_matching([[100, -100]], [False, True], device_ids=[1]).tolist() == [0]


def test_true_ties_choose_lowest_physical_device_then_request():
    raw = np.zeros((3, 3), dtype=np.float32)
    result = solve_canonical_matching(raw, [False, True, True], device_ids=[30, 10, 20],
                                     request_ids=[0, 8, 3])
    assert result.tolist() == [0, 2, 1]
    assert solve_canonical_matching(np.empty((0, 1)), [False], device_ids=[]).shape == (0,)


@pytest.mark.parametrize('scores,devices', [([[np.nan, 0]], [1]), ([[0, np.inf]], [1]),
    ([[-np.inf, 0]], [1]), ([[0], [0]], [1, 1])])
def test_invalid_inputs_rejected(scores, devices):
    with pytest.raises(ValueError):
        solve_canonical_matching(scores, [False]*len(scores[0]), device_ids=devices)


def test_heads_keep_actions_fixed_but_probabilities_follow_temperature():
    torch.manual_seed(17)
    device = DeviceRequestPtrActor(8, embed_dim=8, nhead=2).eval()
    plane = JointPairPtrActor(8, embed_dim=8, nhead=2, pair_feature_dim=0).eval()
    device.canonical_h_decode = plane.canonical_h_decode = True
    query = torch.randn(3, 1, 8)
    req = torch.randn(3, 4, 8)
    req_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    args = dict(query=query, op_nodes=torch.randn(3, 2, 8), site_nodes=torch.randn(3, 3, 8),
                op_valid_mask=torch.ones(3, 2, dtype=torch.bool),
                site_valid_mask=torch.ones(3, 3, dtype=torch.bool), deterministic=True)
    outputs = []
    with torch.no_grad():
        for tau in [.03, .1, .3, .5, 1., 2.]:
            d = device(query, req, req_mask, deterministic=True, tau=tau)
            p = plane(**args, tau=tau)
            outputs.append((d, p, device._canonical_h_raw.clone()))
    for d, p, raw in outputs[1:]:
        assert torch.equal(d[0], outputs[0][0][0])
        assert torch.equal(p[0], outputs[0][1][0]) and torch.equal(p[1], outputs[0][1][1])
        assert torch.equal(raw, outputs[0][2])
        assert not torch.equal(d[2], outputs[0][0][2])
        assert not torch.equal(p[3], outputs[0][1][3])
        assert torch.allclose(d[2].exp().sum(-1), torch.ones(3))
        assert torch.allclose(p[3].exp().sum(-1), torch.ones(3))
    with pytest.raises(ValueError):
        device(query, req, req_mask, deterministic=False, tau=.3)


def test_canonical_full_policy_integration_and_environment_history():
    from types import SimpleNamespace
    from pathlib import Path
    import yaml
    from torch_geometric.data import Batch
    from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import GNN_Actor_Critic
    from onpolicy.envs.HKBZ.test.test_action_masks import _make_env
    from onpolicy.utils.stage3_canonical_h import enable_canonical_h
    env = _make_env()
    try:
        torch.manual_seed(41)
        obs, _, info = env.reset()
        config = yaml.safe_load((Path(__file__).resolve().parents[4]/'onpolicy/config/ac.yaml').read_text())
        config['device_global_matching'] = True
        policy = GNN_Actor_Critic(**config, max_plane_agents=env.n_plane_agents,
                                 max_device_agents=env.max_device_num).eval()
        before = {k:v.clone() for k,v in policy.state_dict().items()}
        enable_canonical_h(SimpleNamespace(ac=policy))
        hidden = {t:torch.zeros(2, env.n_agents, 1, 64) for t in [.03, .3, .5]}
        resource_decisions = 0
        with torch.no_grad():
            for _ in range(16):
                pi = {key:torch.as_tensor(np.stack([info[key], info[key]]),
                    dtype=torch.bool if key=='active_agents' else torch.long)
                    for key in ['active_agents','last_op_indices','last_site_indices']}
                resource_decisions += int(pi['active_agents'][:,env.n_plane_agents:].sum())
                outputs = []
                for tau in hidden:
                    policy.tau = tau
                    result = policy({'graph':Batch.from_data_list([obs, obs.clone()]),
                                     'hidden_states':hidden[tau]},pi,deterministic=True)
                    hidden[tau] = result[3]
                    outputs.append(result)
                for result in outputs[1:]:
                    assert torch.equal(result[1],outputs[0][1])
                    assert torch.equal(result[3],outputs[0][3])
                obs,_,done,info = env.step(outputs[0][1][0].numpy())
                for h in hidden.values():
                    h[:,np.asarray(done).reshape(-1).astype(bool)] = 0
        assert resource_decisions > 0
        assert all(torch.equal(v, before[k]) for k,v in policy.state_dict().items())
    finally:
        env.close()
