"""CPU mathematical/identity tests; synthetic costs are never research evidence."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from onpolicy.utils.stage3_local_improvement import (ENVIRONMENT, HISTORY, TaskQueue, candidate_indices,
    remap_history, local_advantages, validate_group, fit_gate)
from onpolicy.runner.shared.stage3_local_improvement_engine import LocalScore, LocalEngine, distribution, masked_kl, save_tensor


def test_main_contract_is_user_selected_soft_h2():
    assert ENVIRONMENT == {"device_future_intent_horizon": 2, "device_frontier_max_requests": 4,
                           "device_lookahead_reservation_mode": "soft"}


def test_history_maps_identity_not_request_slot_and_preserves_planes():
    previous = np.full((28, 3), -1)
    previous[0] = [3, 4, -1]
    previous[24:, 0] = [1, 2, 0, 3]
    old = [None, (1, 0, 1, 0), (2, 0, 2, 1), (3, 0, 3, 2)]
    new = [None, old[2], old[1], (4, 0, 4, 3)]
    mapped, audit = remap_history(previous, old, new)
    assert mapped[24:, 0].tolist() == [2, 1, 0, -1]
    assert np.array_equal(mapped[:24], previous[:24])
    assert previous[24:, 0].tolist() == [1, 2, 0, 3]
    assert audit == {"history_uses": 3, "index_aliases": 3, "expired": 1, "noop": 1}


def test_history_duplicate_live_identity_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        remap_history(np.full((25, 3), -1), [], [None, (1,), (1,)])


def test_history_audit_counts_only_active_decoders():
    p = np.full((25,3), -1)
    p[24,0] = 1
    remapped, audit = remap_history(p, [None, (1,)], [None], active=np.zeros(25, dtype=bool))
    assert remapped[24,0] == -1 and not audit["history_uses"]


def test_candidates_have_legal_support_incumbent_and_reproducible_sampling():
    lp = np.array([-1., -2., -np.inf, -3.])
    mask = np.array([1,1,0,1], dtype=bool)
    assert candidate_indices(lp, mask, 8, 1) == [0,1,3]
    a = candidate_indices(lp, mask, 1000, 32, sampled=True)
    assert a == candidate_indices(lp, mask, 1000, 32, sampled=True)
    assert set(a) == {0,1,3} and a.count(0) > a.count(1) > a.count(3)


def test_advantages_preserve_negative_sign_and_absolute_cost_scale():
    assert local_advantages(100, [90,100,120]).tolist() == [0.1,0.,-0.2]
    for costs in ([np.nan], [0], [-1]):
        with pytest.raises(ValueError):
            local_advantages(100, costs)


def test_zero_residual_is_function_preserving_and_trainable():
    model = LocalScore(4, 3, 8)
    q, n, p = torch.randn(2,4), torch.randn(2,5,3), torch.randn(2,5,8)
    d = model(q,n,p)
    assert torch.equal(d, torch.zeros_like(d))
    mask = torch.tensor([[1,1,0,1,0], [0,1,1,0,0]], dtype=torch.bool)
    base = torch.log_softmax(torch.randn(2,5).masked_fill(~mask, -torch.inf), -1)
    actual = distribution(base,d,mask)
    assert torch.equal(base.argmax(-1), actual.argmax(-1))
    assert torch.allclose(base[mask], actual[mask], atol=1e-6)
    assert torch.isneginf(actual[~mask]).all()
    (-actual[0,0]-actual[1,1]).backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert torch.isfinite(masked_kl(base,actual,mask)).all()


def learner_fixture():
    torch.manual_seed(29)
    runner = LocalEngine.__new__(LocalEngine)
    runner.device = torch.device("cpu")
    runner.scores = nn.ModuleDict({"1": LocalScore(4, 3, 8), "2": LocalScore(4, 3, 8)})
    runner.optimizer = torch.optim.Adam(runner.scores.parameters(), lr=1e-4, eps=1e-5)
    runner.behavior_sha, runner.policy_updates = "frozen", 0
    record = {"role": 1, "query": torch.randn(4), "nodes": torch.randn(3,3), "physical": torch.randn(3,8),
        "base": torch.log(torch.tensor([.5,.3,.2])), "mask": torch.ones(3, dtype=torch.bool), "action": 0}
    record["old_logp"] = record["base"].clone()
    group = {"case_id": "case", "completed": True, "behavior_sha256": "frozen", "query_count": 4,
        "kind": "local_on_policy", "states": [{"record": record, "reference_cost": 100.,
        "sampling": "on_policy_with_replacement", "actions": [0,1,2], "costs": [100.,80.,120.]}]}
    return runner, group


@pytest.mark.parametrize("mode", ["local_rl", "rank"])
def test_real_autograd_local_losses_and_greedy_margin(mode):
    runner, group = learner_fixture()
    before = runner.record_distribution([group["states"][0]["record"]])[0].detach()
    stats = runner.learn([group], mode, lr=.001, epochs=2)
    after = runner.record_distribution([group["states"][0]["record"]])[0].detach()
    assert stats["actor_updates"] == 2
    assert stats["epochs"][0]["preclip_gradient_norm"] > 0
    assert (after[0,1]-after[0,2]) > (before[0,1]-before[0,2])


def test_local_policy_gradient_matches_manual_at_ratio_one():
    runner, group = learner_fixture()
    record = group["states"][0]["record"]
    lp, _ = runner.record_distribution([record])
    advantage = torch.tensor([0.,.2,-.2])
    manual = -(lp[0]*advantage).mean()
    expected = torch.autograd.grad(manual, runner.scores["1"].output.weight, retain_graph=True)[0]
    ratio = (lp[0]-record["old_logp"]).exp()
    actual = -torch.minimum(ratio*advantage, ratio.clamp(.8,1.2)*advantage).mean()
    observed = torch.autograd.grad(actual, runner.scores["1"].output.weight)[0]
    assert torch.allclose(expected, observed, atol=1e-7)


@pytest.mark.parametrize("kind,mode", [("local_search","local_rl"), ("local_on_policy","ppo"), ("full_on_policy","local_rl")])
def test_reject_wrong_behavior_class(kind, mode):
    with pytest.raises(ValueError):
        validate_group([{"case_id":"a", "kind":kind, "behavior_sha256":"x", "completed":True, "query_count":1}], "x", mode)


def test_stale_policy_and_duplicate_case_rejected():
    _, group = learner_fixture()
    with pytest.raises(ValueError):
        validate_group([group], "new", "local_rl")
    with pytest.raises(ValueError):
        validate_group([group, group], "frozen", "local_rl")


def test_queue_claim_is_unique_across_shared_consumers(tmp_path):
    queue = TaskQueue(tmp_path/"validator")
    for i in range(20):
        queue.submit(f"task{i:02d}", {"kind":"evaluate", "model":str(i)})
    def consume(_):
        ids = []
        while (row := queue.claim()) is not None:
            ids.append(row["id"])
            queue.finish(row, result={"ok":True})
        return ids
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = sum(pool.map(consume, range(4)), [])
    assert len(claimed) == len(set(claimed)) == 20
    assert queue.outstanding() == 0
    assert queue.poll("task00") == {"ok":True}
    with pytest.raises(FileExistsError):
        queue.submit("task00", {})


def test_targeted_worker_and_failed_task_no_retry(tmp_path):
    q = TaskQueue(tmp_path)
    q.submit("a", {"worker":"gpu7"})
    assert q.claim("gpu0") is None
    r = q.claim("gpu7")
    q.finish(r, error="intentional synthetic failure")
    with pytest.raises(RuntimeError):
        q.poll("a")
    assert q.claim("gpu7") is None


def test_atomic_checkpoint_refuses_overwrite(tmp_path):
    path = tmp_path/"a.pt"
    save_tensor(path, {"x":torch.ones(2)})
    with pytest.raises(FileExistsError):
        save_tensor(path, {"x":torch.zeros(2)})
    assert torch.equal(torch.load(path, weights_only=False)["x"], torch.ones(2))


def test_gate_includes_cases_without_teacher_gain():
    cases = [{"path":str(i)} for i in range(16)]
    source = {str(i):100. for i in range(16)}
    teacher = {str(i):98. if i<8 else 100. for i in range(16)}
    probe = {"gain_fraction":.001, "completion_rate":1., "distributions":{"ood_stress":{"regression_fraction":0}},
        "profiles":{"stress_joint":{"regression_fraction":0}}, "tail_makespan":100., "source_tail_makespan":100.,
        "regression_over_5pct_fraction":0.}
    g = fit_gate(cases, source, teacher, {"delta_seconds":-.5, "completion_rate":1.}, probe, .8)
    assert g["passed"] and g["recovery"] == .5 and g["cases_gt1pct"] == 8
    assert not fit_gate(cases, source, teacher, {"delta_seconds":.1, "completion_rate":1.}, probe, .9)["passed"]
    with pytest.raises(ValueError):
        fit_gate(cases, source, {"0":98.}, {}, probe, 1.)


def test_native_actor_class_not_globally_patched():
    from onpolicy.algorithms.utils.ptr_actor import DeviceRequestPtrActor
    assert DeviceRequestPtrActor.forward.__module__ == "onpolicy.algorithms.utils.ptr_actor"


@pytest.mark.parametrize("arm,warm,expected_groups", [("T_PPO",False,48), ("L_RL",False,48), ("PRE",True,30)])
def test_orchestrator_budget_commits_and_case_boundaries(tmp_path, arm, warm, expected_groups):
    from onpolicy.scripts.train.run_stage3_local_improvement import Suite
    class FakeSuite(Suite):
        def submit(self, label, payload, *, validator=False):
            key = str(len(self.payloads))
            self.payloads[key] = payload
            if payload["kind"] == "collect":
                n = 4 if payload["mode"] == "warm" else 5
                self.results[key] = {"query_count":n, "case_id":payload["case"]["path"], "path":key, "sha256":key}
            elif payload["kind"] == "learn":
                assert len({d["case_id"] for d in payload["data"]}) == 4
                self.results[key] = {"checkpoint":key, "checkpoint_sha256":key, "metadata":payload["metadata"]}
            else:
                self.results[key] = {"not_executed":"synthetic controller fixture"}
            return key
        def wait(self, keys, *, validator=False):
            return [self.results[k] for k in keys]
    cases = [{"path":f"case_{i}", "content_sha256":str(i)} for i in range(120)]
    manifest = {"root":str(tmp_path), "splits":{"train_pilot120":cases, "tune":cases[:2]},
                "training":{"lr":1e-4, "ppo_epochs":2}}
    suite = FakeSuite(manifest, tmp_path/"manifest.json")
    suite.payloads, suite.results = {}, {}
    suite.source_costs = {c["path"]:100. for c in cases}
    initial = {"checkpoint":{"checkpoint":"initial", "checkpoint_sha256":"initial"}, "queries":0, "cursor":0}
    budget = 480 if warm else 960
    result = suite.train_to(arm, initial, budget, warm=warm)
    assert result["queries"] == budget and result["cursor"] == 4*expected_groups
    assert len(list((tmp_path/"train"/arm).glob("commit_q*.json"))) == expected_groups
    updates = [v for v in suite.payloads.values() if v["kind"] == "learn"]
    assert len(updates) == expected_groups
    assert updates[-1]["metadata"]["query_count"] == budget
    assert not updates[-1]["metadata"]["diagnostic_only"]
