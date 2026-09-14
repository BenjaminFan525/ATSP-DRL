"""Math and orchestration tests; synthetic labels are not research evidence."""
import copy
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from onpolicy.runner.shared.stage3_conditional_engine import (
    ConditionalEngine, ConditionalScore, CachedRanking, conditional_distribution,
    residual_ablation, install_decision_identity)
from onpolicy.runner.shared.stage3_local_improvement_engine import LocalScore, distribution
from onpolicy.utils.stage3_conditional_improvement import (
    admission_route, meaningful_epsilon, cost_labels, select_stratified_states,
    aggregate_evaluations, learnability_gate, evaluation_summary, closed_loop_gate)


@pytest.mark.parametrize("mask", [[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1]])
def test_conditional_zero_exact_mask_and_finite_backward(mask):
    mask = torch.tensor([mask], dtype=torch.bool)
    base = torch.log_softmax(torch.tensor([[1., 2., 3.]]).masked_fill(~mask, -torch.inf), -1)
    gate = torch.zeros(1, requires_grad=True)
    rank = torch.zeros(1, 3, requires_grad=True)
    actual = conditional_distribution(base, gate, rank, mask)
    assert torch.allclose(actual[mask], base[mask], atol=1e-6)
    assert torch.isneginf(actual[~mask]).all()
    assert torch.equal(actual.argmax(-1), base.argmax(-1))
    (-actual[mask].sum()).backward()
    assert torch.isfinite(gate.grad).all() and torch.isfinite(rank.grad).all()


def test_global_argmax_is_not_binary_threshold_gate():
    base = torch.log(torch.tensor([[.4, .3, .3]]))
    actual = conditional_distribution(base, torch.zeros(1), torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool))
    assert actual.argmax(-1).item() == 0


def test_residual_decomposition_reconstructs_full_distribution():
    mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.bool)
    base = torch.log_softmax(torch.randn(2, 4).masked_fill(~mask, -torch.inf), -1)
    delta = torch.randn(2, 4)
    wait = residual_ablation(delta, mask, "wait_only")
    ranking = residual_ablation(delta, mask, "ranking_only")
    assert torch.allclose(distribution(base, delta, mask)[mask], distribution(base, wait+ranking, mask)[mask], atol=1e-6)


def fixture(architecture="conditional", pair=False):
    torch.manual_seed(12)
    engine = ConditionalEngine.__new__(ConditionalEngine)
    engine.device, engine.architecture, engine.ablation = torch.device("cpu"), architecture, "full"
    cls = LocalScore if architecture == "score" else ConditionalScore
    kwargs = {} if architecture == "score" else {"pair": pair}
    engine.scores = nn.ModuleDict({str(r): cls(4, 3, 8, **kwargs) for r in (1, 2)})
    engine.optimizer = torch.optim.Adam(engine.scores.parameters(), lr=.001, eps=1e-5)
    engine.policy_updates = 0
    record = {"role": 1, "key": [3, 1, 0], "query": torch.randn(4), "nodes": torch.randn(3, 3),
              "physical": torch.randn(3, 12 if pair else 8), "mask": torch.ones(3, dtype=torch.bool),
              "base": torch.log(torch.tensor([.6, .3, .1])), "action": 0}
    record["old_logp"] = record["base"].clone()
    state = {"record": record, "reference_cost": 100., "actions": [0, 1, 2], "costs": [100., 70., 150.]}
    return engine, [{"case_id": "synthetic", "states": [state]}]


@pytest.mark.parametrize("architecture", ["score", "conditional", "conditional_pair"])
def test_cached_fit_learns_and_has_zero_simulator_queries(architecture):
    engine, groups = fixture(architecture, pair=architecture == "conditional_pair")
    trainer = CachedRanking(engine, groups)
    result = trainer.fit(lr=.01, steps=200)
    assert result["metrics"]["meaningful"]["beneficial_hit_rate"] == 1.
    assert result["metrics"]["loss"] < result["initial"]["loss"]
    assert learnability_gate(result["metrics"])["passed"]
    assert result["executed_updates"] <= 200


def test_independent_logits_are_diagnostic_only_and_fit_labels():
    engine, groups = fixture()
    original = copy.deepcopy(engine.scores.state_dict())
    result = CachedRanking(engine, groups, lookup=True).fit(lr=.1, steps=200)
    assert result["lookup_only"] and result["metrics"]["meaningful"]["beneficial_hit_rate"] == 1.
    assert all(torch.equal(v, original[k]) for k, v in engine.scores.state_dict().items())


def test_pair_architecture_rejects_historical_missing_features():
    engine, groups = fixture("conditional_pair", pair=True)
    groups[0]["states"][0]["record"]["physical"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="fresh explicit-device"):
        CachedRanking(engine, groups)


def test_unknown_action_never_counts_as_zero_regret_or_pass():
    engine, groups = fixture()
    state = groups[0]["states"][0]
    state["actions"], state["costs"] = [1, 2], [70., 150.]
    metrics = CachedRanking(engine, groups).metrics()
    assert metrics["meaningful"]["unknown_actions"] == 1
    assert not learnability_gate(metrics)["passed"]


def test_cost_labels_reject_conflicting_repeated_draws():
    assert meaningful_epsilon(100.) == 5.
    assert meaningful_epsilon(10000.) == 10.
    with pytest.raises(ValueError, match="inconsistent"):
        cost_labels({"reference_cost": 100., "actions": [1, 1], "costs": [90., 100.]})


def test_admission_does_not_conflate_fit_and_safe_warm():
    assert admission_route(False, True, True) == "implementation_blocked"
    assert admission_route(True, False, True) == "learnability_budget_exhausted"
    assert admission_route(True, True, False) == "c0_rl_only_480"
    assert admission_route(True, True, True) == "four_arm"


def test_stratified_selector_is_unique_reproducible_and_outcome_independent():
    _, groups = fixture()
    records = []
    for i in range(20):
        record = copy.deepcopy(groups[0]["states"][0]["record"])
        record["key"], record["role"] = [i*10, 1+i%2, 0], 1+i%2
        records.append(record)
    a = select_stratified_states(records, 8, 77)
    b = select_stratified_states(records, 8, 77)
    assert [r["key"] for r in a] == [r["key"] for r in b]
    assert len({tuple(r["key"]) for r in a}) == 8
    assert {r["role"] for r in a} == {1, 2}


def test_identity_hook_passes_actual_agent_index_not_ordinal():
    class Dummy:
        def forward(self, actor_head):
            agent_idx = 57
            a = actor_head()
            agent_idx = 31
            b = actor_head()
            return a, b
    dummy = Dummy()
    info = install_decision_identity(dummy)
    assert info["resource_call_sites"] == 2
    assert dummy.forward(lambda **kw: kw["stage3_agent_index"]) == (57, 31)


def test_validator_rejects_duplicate_missing_and_mixed_checkpoint_shards():
    cases = [{"path": "a"}, {"path": "b"}]
    row = {"case_id": "a", "makespan": 100., "completed": True, "profile": "iid", "distribution": "iid"}
    parts = [{"checkpoint_sha256": "x", "cases": [row]}, {"checkpoint_sha256": "x", "cases": [row]}]
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        aggregate_evaluations(parts, cases, {"a": 100., "b": 100.}, {})
    parts[1] = {"checkpoint_sha256": "y", "cases": [dict(row, case_id="b")]}
    with pytest.raises(ValueError, match="different checkpoints"):
        aggregate_evaluations(parts, cases, {"a": 100., "b": 100.}, {})


def controller_fixture(tmp_path):
    from onpolicy.scripts.train.run_stage3_conditional_improvement import Suite
    from onpolicy.utils.stage3_research import read_json
    cases = [{"path": f"case_{i}", "content_sha256": str(i), "profile": "stress_joint" if i%2 else "light",
              "distribution": "ood_stress" if i%2 else "iid"} for i in range(120)]
    manifest = {"root": str(tmp_path), "splits": {"train_pilot120": cases, "train_probe64": cases[:8], "tune": cases[:8]},
                "resources": {"validator_shard_cases": 6}, "learning_protocol_sha256": "test-protocol",
                "training": {"lr": 1e-4, "ppo_epochs": 2, "warm_lr": 3e-4},
                "timeouts_seconds": {"diagnostic_budget": 10000, "formal_budget": 10000}}
    class FakeSuite(Suite):
        def health(self):
            pass
        def confirm(self, winner, states, initial, warm, hb):
            return {"opened": False, "test_only_candidate_boundary": winner}
        def submit(self, label, payload, *, validator=False):
            with self.fake_lock:
                return self.immediate_submit(label, payload, validator=validator)
        def immediate_submit(self, label, payload, *, validator=False):
            self.payloads.append(payload)
            key = super().submit(label, payload, validator=validator)
            queue = self.validator if validator else self.work
            row = queue.claim()
            assert row["id"] == key
            kind = payload["kind"]
            if kind == "evaluate":
                cost = self.warm_cost if payload["checkpoint"] == "warm" else 97.
                result = {"checkpoint_sha256": payload["checkpoint_sha256"], "cases": [dict(
                    case_id=c["path"], makespan=cost, completed=True, profile=c["profile"], distribution=c["distribution"])
                    for c in payload["cases"]]}
            elif kind == "collect":
                count = 4 if payload["mode"] == "warm" else 5
                if payload["mode"] == "warm":
                    assert payload["states"] == 3 and payload["candidates"] == 2
                result = {"case_id": payload["case"]["path"], "query_count": count, "path": key, "sha256": key}
            elif kind == "formal_warm_fit":
                assert len(payload["data"]) == 120 and sum(r["query_count"] for r in payload["data"]) == 480
                assert len({r["case_id"] for r in payload["data"]}) == 120
                result = {"checkpoint": "warm", "checkpoint_sha256": "warm-sha", "metadata": {"diagnostic_only": False}}
            elif kind == "learn":
                result = {"checkpoint": key, "checkpoint_sha256": key, "metadata": payload["metadata"]}
            else:
                raise AssertionError(kind)
            queue.finish(row, result=result)
            return key
    suite = FakeSuite(manifest, tmp_path/"manifest.json")
    suite.fake_lock = threading.Lock()
    suite.payloads, suite.warm_cost = [], 98.
    suite.source_costs = {c["path"]: 100. for c in cases}
    suite.legacy_source_costs = dict(suite.source_costs)
    return suite, cases


def test_shared_validator_shards_deduplicate_and_aggregate_complete_cases(tmp_path):
    suite, cases = controller_fixture(tmp_path)
    checkpoint = {"checkpoint": "x", "checkpoint_sha256": "x-sha"}
    a = suite.evaluate(checkpoint, cases[:13], "one")
    b = suite.evaluate(checkpoint, cases[:13], "another-label")
    assert a == b and len(a["shards"]) == 3
    assert len(suite.payloads) == 3
    result = suite.wait([a], validator=True)[0]
    assert len(result["cases"]) == result["evaluation_queries"] == 13
    assert result["summary"]["gain_fraction"] == pytest.approx(.03)


def test_formal_warm_collects_three_positions_then_fits_complete_corpus(tmp_path):
    suite, _ = controller_fixture(tmp_path)
    blank = {"checkpoint": {"checkpoint": "zero", "checkpoint_sha256": "zero-sha"}, "queries": 0, "cursor": 0}
    result = suite.train_to("PRE", blank, 480, warm=True)
    assert result["queries"] == 480 and result["cursor"] == 120
    collects = [p for p in suite.payloads if p["kind"] == "collect"]
    assert len(collects) == 120 and {p["checkpoint_sha256"] for p in collects} == {"zero-sha"}
    assert [p["kind"] for p in suite.payloads].index("formal_warm_fit") == 120
    assert not any(p["kind"] == "learn" for p in suite.payloads)


def test_limited_rl_route_never_uses_unsafe_warm_or_exceeds480(tmp_path):
    suite, _ = controller_fixture(tmp_path)
    class Heartbeat:
        def update(self, **kwargs):
            pass
    result = suite.training({"checkpoint": "zero", "checkpoint_sha256": "zero-sha"}, Heartbeat(), limited=True)
    assert result["formal_training_started"]
    assert result["status"] == "limited_c0_rl_480_complete_review_required"
    assert not any(p["kind"] == "formal_warm_fit" or p.get("mode") == "warm" for p in suite.payloads)
    updates = [p["metadata"] for p in suite.payloads if p["kind"] == "learn"]
    assert {p["arm"] for p in updates} == {"T_PPO", "L_RL"}
    assert max(p["query_count"] for p in updates) == 480


def test_unsafe_formal_warm_never_initializes_mixed_rl(tmp_path):
    suite, _ = controller_fixture(tmp_path)
    suite.warm_cost = 102.
    class Heartbeat:
        def update(self, **kwargs):
            pass
    result = suite.training({"checkpoint": "zero", "checkpoint_sha256": "zero-sha"}, Heartbeat())
    assert "L_RANK_RL" not in result["screen"]
    updates = [p["metadata"] for p in suite.payloads if p["kind"] == "learn"]
    assert "L_RANK_RL" not in {p["arm"] for p in updates}
    assert max(p["query_count"] for p in updates) <= 1920
    assert (tmp_path/"mixed_arm_not_admitted.json").exists()


def test_shared_warm_both_arms_have_equal_initial_checkpoint_and_budget(tmp_path):
    suite, _ = controller_fixture(tmp_path)
    class Heartbeat:
        def update(self, **kwargs):
            pass
    suite.training({"checkpoint": "zero", "checkpoint_sha256": "zero-sha"}, Heartbeat())
    first_updates = {p["metadata"]["arm"]: p for p in suite.payloads if p["kind"] == "learn"
                     and p["metadata"]["query_count"] == 500}
    assert first_updates["L_RANK"]["checkpoint_sha256"] == first_updates["L_RANK_RL"]["checkpoint_sha256"] == "warm-sha"
    assert sum(p["kind"] == "formal_warm_fit" for p in suite.payloads) == 1
    assert first_updates["L_RANK"]["metadata"]["case_cursor"] == first_updates["L_RANK_RL"]["metadata"]["case_cursor"] == 124


def test_no_signal_is_a_diagnostic_result_not_a_runtime_crash():
    engine, groups = fixture()
    groups[0]["states"][0]["costs"] = [100., 100., 100.]
    result = CachedRanking(engine, groups).fit(lr=.001, steps=200)
    assert result["stop_reason"] == "no_informative_preferences" and result["executed_updates"] == 0
    assert not learnability_gate(result["metrics"])["passed"]


def test_incomplete_validation_never_reports_partial_time_as_cost_gain():
    rows = [{"case_id": "x", "completed": False, "makespan": 2., "cycle_terminated": True}]
    summary = evaluation_summary(rows, {"x": 100.})
    assert not summary["metric_valid"] and summary["makespan"] is None
    assert summary["paired_case_bootstrap_gain_ci95"] is None
    assert summary["gain_is_rejection_sentinel"] and summary["completion_rate"] == 0
    gate = closed_loop_gate({"cases": rows, "summary": summary}, {}, {"x": 100.}, {"x": 90.})
    assert not gate["passed"] and gate["recovery"] is None


@pytest.fixture
def checkpoint_worker(tmp_path, monkeypatch):
    """Real heads/checkpoint IO, without the simulator or the large frozen GNN."""
    from onpolicy.scripts.train import stage3_conditional_worker as module
    from onpolicy.utils.stage3_local_improvement import HISTORY
    from onpolicy.utils.stage3_research import digest_file

    class TinyEngine(ConditionalEngine):
        def __init__(self, source, *, architecture="score", protocol="test", history=HISTORY, **kwargs):
            self.device = torch.device("cpu")
            self.source, self.source_sha = str(source), digest_file(source)
            self.protocol, self.history = protocol, history
            actors = [SimpleNamespace(req_query_ff=nn.Linear(4, 4), embed_dim=3) for _ in range(2)]
            self.policy = SimpleNamespace(ac=SimpleNamespace(device_actor=actors[0], transporter_actor=actors[1]))
            self.ablation, self.checkpoint_meta, self.behavior_sha = "full", {}, self.source_sha
            self.set_architecture(architecture)

        def close(self):
            pass

    def canary(worker, task, output, hb):
        worker.close()
        engine = worker.engine()
        assert engine.architecture == task["architecture"]
        if task.get("fail"):
            raise RuntimeError("synthetic canary failure")
        return {"passed": True, "architecture": engine.architecture}

    source = tmp_path/"source.pt"
    source.write_bytes(b"synthetic frozen source")
    monkeypatch.setattr(module, "ConditionalEngine", TinyEngine)
    monkeypatch.setattr(module.Worker, "canary", canary)
    manifest = {"source": {"path": str(source), "sha256": digest_file(source)},
                "training": {"seed": 42}, "manifest_sha256": "new-protocol"}
    worker = module.ConditionalWorker(manifest, width=0, device="cpu")
    yield worker, TinyEngine, source
    worker.close()


def assert_checkpoint_trees_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_checkpoint_trees_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_checkpoint_trees_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("previous_architecture", ["score", "conditional", "conditional_pair"])
def test_historical_import_after_canary_preserves_scores_and_populated_adam(checkpoint_worker, tmp_path, previous_architecture):
    from onpolicy.runner.shared.stage3_local_improvement_engine import LocalEngine
    worker, engine_class, source = checkpoint_worker
    historical = engine_class(source, protocol="historical-protocol")
    # Populate Adam for both roles; empty-optimizer tests miss this regression.
    sum(p.square().sum() for p in historical.scores.parameters()).backward()
    historical.optimizer.step()
    historical.policy_updates = 17
    path = tmp_path/"historical.pt"
    sha = LocalEngine.save(historical, path, diagnostic_only=True)  # Legacy metadata has no architecture.
    payload = torch.load(path, weights_only=False)
    worker.run({"kind": "canary", "architecture": previous_architecture}, tmp_path/"canary", None)
    result = worker.run({"kind": "import_previous_fit", "previous_checkpoint": str(path),
                         "previous_sha256": sha}, tmp_path/"import", None)
    imported = torch.load(result["checkpoint"], weights_only=False)
    assert imported["metadata"]["architecture"] == "score"
    assert imported["metadata"]["diagnostic_only"] is True
    assert imported["protocol"] == "new-protocol" and imported["policy_updates"] == 17
    assert_checkpoint_trees_equal(imported["scores"], payload["scores"])
    assert_checkpoint_trees_equal(imported["optimizer"], payload["optimizer"])
    assert result["physical_query_ledger"] == {"started": 0, "completed": 0}
    worker.close()
    assert worker.engine().architecture == "score"


def test_failed_canary_does_not_leak_architecture_to_next_task(checkpoint_worker, tmp_path):
    worker, _, _ = checkpoint_worker
    with pytest.raises(RuntimeError, match="synthetic canary failure"):
        worker.run({"kind": "canary", "architecture": "conditional_pair", "fail": True}, tmp_path/"failed", None)
    # No manual close: a resident engine must respect the new task's default too.
    assert worker.engine().architecture == "score"


def test_mixed_checkpoint_tasks_restore_architecture_without_rebuilding_pool(checkpoint_worker, tmp_path):
    worker, _, _ = checkpoint_worker
    checkpoints = [worker.run({"kind": "initialize_model", "architecture": architecture}, tmp_path/architecture, None)
                   for architecture in ("score", "conditional", "conditional_pair")]
    resident = worker.runner
    for checkpoint in reversed(checkpoints):
        runner = worker.engine(checkpoint=checkpoint["checkpoint"])
        assert runner is resident
        assert runner.architecture == checkpoint["architecture"]
        assert_checkpoint_trees_equal(runner.scores.state_dict(),
                                     torch.load(checkpoint["checkpoint"], weights_only=False)["scores"])


@pytest.mark.parametrize("field,value", [("schema", "other-schema"), ("source_sha256", "changed"),
                                       ("history", "legacy"), ("environment", {"h": 3, "f": 4, "reservation": "hard"}),
                                       ("metadata", {"architecture": "conditional"})])
def test_historical_import_rejects_incompatible_contract(checkpoint_worker, tmp_path, field, value):
    from onpolicy.runner.shared.stage3_local_improvement_engine import LocalEngine, save_tensor
    worker, engine_class, source = checkpoint_worker
    path = tmp_path/"historical.pt"
    LocalEngine.save(engine_class(source), path, diagnostic_only=True)
    payload = torch.load(path, weights_only=False)
    payload[field] = value
    path = tmp_path/"incompatible.pt"
    sha = save_tensor(path, payload)
    with pytest.raises(ValueError, match="Historical"):
        worker.run({"kind": "import_previous_fit", "previous_checkpoint": str(path),
                    "previous_sha256": sha}, tmp_path/"rejected", None)
    assert not (tmp_path/"rejected/historical_diagnostic.pt").exists()


@pytest.mark.skipif(not os.environ.get("HKBZ_STAGE3_RECOVERY_MANIFEST"), reason="explicit real-checkpoint/free-GPU check only")
def test_real_historical_import_gpu_preserves_scores_adam_and_cached_predictions(tmp_path):
    from onpolicy.scripts.train.stage3_conditional_worker import ConditionalWorker
    from onpolicy.utils.stage3_research import read_json, digest_file
    manifest = read_json(os.environ["HKBZ_STAGE3_RECOVERY_MANIFEST"])
    previous_path = manifest["reuse"]["fit_checkpoint"]
    previous_sha = manifest["reuse"]["fit_checkpoint_sha256"]
    assert digest_file(previous_path) == previous_sha
    previous = torch.load(previous_path, map_location="cpu", weights_only=False)
    teacher = read_json(manifest["reuse"]["teachers"])["data"][0]
    assert digest_file(teacher["path"]) == teacher["sha256"]
    group = torch.load(teacher["path"], map_location="cpu", weights_only=False)
    worker = ConditionalWorker(manifest, width=0, device="cuda:0")
    try:
        for architecture in ("score", "conditional", "conditional_pair"):
            output = tmp_path/architecture
            worker.run({"kind": "initialize_model", "architecture": architecture}, output/"initial", None)
            result = worker.run({"kind": "import_previous_fit", "previous_checkpoint": previous_path,
                                 "previous_sha256": previous_sha}, output/"import", None)
            imported = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
            assert_checkpoint_trees_equal(imported["scores"], previous["scores"])
            assert_checkpoint_trees_equal(imported["optimizer"], previous["optimizer"])
            assert imported["policy_updates"] == previous["policy_updates"]
            runner = worker.engine(checkpoint=result["checkpoint"])
            assert runner.architecture == "score" and runner.checkpoint_meta["diagnostic_only"]
            # Compare the legacy score formula on real cached observations.
            with torch.no_grad():
                for state in group["states"]:
                    actual, tensors = runner.record_distribution([state["record"]])
                    delta = runner.scores[str(state["record"]["role"])](
                        tensors["query"], tensors["nodes"], tensors["physical"])
                    expected = distribution(tensors["base"], delta, tensors["mask"])
                    mask = tensors["mask"]
                    assert torch.allclose(actual[mask], expected[mask], atol=1e-6)
            assert result["physical_query_ledger"] == {"started": 0, "completed": 0}
        assert digest_file(previous_path) == previous_sha
    finally:
        worker.close()
