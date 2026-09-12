import copy
import numpy as np
import pytest
import torch

from onpolicy.utils.stage3_local_exploration import (MODES, ARMS, RoleExploration,
    exploration_config, training_mask, choose_elite, case_schedule, exploration_gate,
    imitation_gate, LocalEvaluationQueue, EVAL_PROTOCOL)
from onpolicy.envs.HKBZ.test.test_stage3_sampling_audit import actors, inputs


@pytest.mark.parametrize("mode", ["R", "J"])
@pytest.mark.parametrize("role", [0, 1, 2])
def test_differentiable_native_sampling_and_forced_replay(mode, role):
    ac = actors()
    actor = (ac.actor, ac.device_actor, ac.transporter_actor)[role]
    kwargs = inputs(role)
    names = list(actor.state_dict())
    expected_kwargs = {**kwargs, "tau": MODES[mode]["taus"][role],
                       "deterministic": role not in MODES[mode]["roles"]}
    torch.manual_seed(51)
    expected = actor(**expected_kwargs)
    rng = torch.get_rng_state()
    hooks = RoleExploration(ac, MODES[mode])
    try:
        torch.manual_seed(51)
        sampled = actor(**kwargs)
        assert all(torch.equal(a, b) for a, b in zip(expected, sampled))
        assert torch.equal(rng, torch.get_rng_state())
        forced = {"chosen_op": sampled[0], "chosen_site": sampled[1]} if role == 0 else {"chosen_request": sampled[0]}
        replay = actor(**kwargs, **forced)
        assert torch.equal(replay[-2], sampled[-2])
        assert torch.allclose(replay[-1].exp().sum(-1), torch.ones(3))
        (-replay[-2].mean()).backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
        assert names == list(actor.state_dict())
    finally:
        hooks.close()
    assert not actor._forward_pre_hooks and not actor._forward_hooks


@pytest.mark.parametrize("mode", ["R", "J"])
@pytest.mark.parametrize("role", [0, 1, 2])
def test_explicit_greedy_overrides_sampling(mode, role):
    ac = actors()
    actor = (ac.actor, ac.device_actor, ac.transporter_actor)[role]
    kwargs = {**inputs(role), "deterministic": True}
    expected = actor(**{**kwargs, "tau": MODES[mode]["taus"][role]})
    before = torch.get_rng_state()
    hooks = RoleExploration(ac, mode)
    actual = actor(**kwargs)
    hooks.close()
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("bad", [{"roles": [1], "taus": [.3] * 3}, {"roles": [0, 1, 2], "taus": [0, .3, .3]},
    {"roles": [1, 2], "taus": [float("nan"), .3, .3]}, {"roles": [1, 2], "taus": [.3] * 3, "top_k": 4}])
def test_invalid_training_contract(bad):
    with pytest.raises(ValueError):
        exploration_config(bad)


def test_masks_separate_legal_and_trainable_roles():
    mask = np.array([[1, 1, 0, 1]])
    roles = np.array([[0, 1, 1, 2]])
    assert training_mask(mask, roles, MODES["R"]).tolist() == [[0, 1, 0, 1]]
    assert training_mask(mask, roles, MODES["J"]).tolist() == mask.tolist()
    assert mask.tolist() == [[1, 1, 0, 1]]


def test_schedule_paired_shuffled_and_equal_case_visits():
    cases = [{"content_sha256": str(i), "path": str(i)} for i in range(120)]
    a = case_schedule(cases, 6)
    assert len(a) == 240 and a == case_schedule(cases, 6)
    assert a[:120] != [(0, c) for c in cases]
    assert {c["path"] for _, c in a[:120]} == {c["path"] for _, c in a[120:]}


def test_elite_selection_has_no_every_fourth_case_bias():
    entries = {str(i): {} for i in range(16)}
    usage = {}
    for step in range(16):
        key = choose_elite(entries, usage, 8, step * 4)
        usage[key] = usage.get(key, 0) + 1
    assert set(usage.values()) == {1} and len(usage) == 16
    assert choose_elite({}, {}, 1, 1) is None


def test_exploration_gate_is_complete_and_does_not_require_bc():
    source = {str(i): 100. for i in range(96)}
    diag = [{"case_id": str(i), "prefix_best": {"8": 98., "32": 97.}} for i in range(32)]
    probe = [{"case_id": str(i), "prefix_best": {"8": 99.}} for i in range(32, 96)]
    assert exploration_gate(diag, probe, source)["passed"]
    with pytest.raises(ValueError):
        exploration_gate(diag[:8], probe, source)
    probe[0]["prefix_best"]["8"] = 200.
    assert not exploration_gate(diag, probe, source)["passed"]


def test_imitation_gate_requires_closed_loop_and_teacher_quality():
    probe = {"gain_fraction": .001, "completion_rate": 1., "distributions": {"ood_stress": {"regression_fraction": 0}},
        "profiles": {"stress_joint": {"regression_fraction": 0}}, "tail_makespan": 100,
        "source_tail_makespan": 100, "regression_over_5pct_fraction": 0.}
    assert imitation_gate(.02, {"gain_fraction": .012}, probe)["passed"]
    assert not imitation_gate(.02, {"gain_fraction": -.001}, probe)["passed"]
    assert not imitation_gate(.001, {"gain_fraction": .001}, probe)["passed"]


def test_queue_identity_includes_decoder_and_behavior():
    request = {k: "x" for k in ("checkpoint_sha256", "cases_sha256", "contract_sha256", "code_sha256")}
    request.update(tau=.3, seed=42, evaluation_protocol=EVAL_PROTOCOL, training_exploration=MODES["R"])
    first = LocalEvaluationQueue.identity(request)
    request["training_exploration"] = MODES["J"]
    assert LocalEvaluationQueue.identity(request) != first


@pytest.mark.parametrize("field,value", [("behavior_deterministic", True), ("forced_replay", True), ("policy_updates", 7)])
def test_ppo_rejects_greedy_forced_and_stale_data(field, value):
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    engine = ResearchEngine.__new__(ResearchEngine)
    engine.exploration, engine.policy_updates = MODES["R"], 0
    trajectory = {"exploration": MODES["R"], "behavior_deterministic": False, "forced_replay": False, "policy_updates": 0}
    trajectory[field] = value
    with pytest.raises(ValueError):
        engine.update([trajectory], "source", 100)


@pytest.mark.parametrize("advantage", [1., -1.])
def test_source_advantage_has_correct_gradient_direction(advantage):
    logits = torch.tensor([.2, -.2], requires_grad=True)
    old = logits.detach().log_softmax(0)[0]
    before = logits.detach().softmax(0)[0]
    ratio = (logits.log_softmax(0)[0] - old).exp()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
    loss.backward()
    after = (logits.detach() - .1 * logits.grad).softmax(0)[0]
    assert (after - before) * advantage > 0


def test_trace_integrity_and_immutability(tmp_path):
    from onpolicy.scripts.train.run_stage3_sampling_audit import save_group
    from onpolicy.scripts.train.stage3_local_worker import load_group
    from onpolicy.utils.stage3_research import digest_file
    path = tmp_path / "group.json"
    row = {"seed": 42, "actions": np.zeros((2, 104, 3), np.int32).tolist(), "states": [],
           "steps": 2, "completed": True, "makespan": 100., "case_id": "example"}
    save_group(path, [row], {})
    reference = {"path": str(path), "sha256": digest_file(path)}
    restored, _ = load_group(reference)
    assert restored[0]["actions"] == row["actions"]
    with pytest.raises(FileExistsError):
        save_group(path, [row], {})
    reference["sha256"] = "corrupt"
    with pytest.raises(ValueError):
        load_group(reference)


def test_manifest_identity_and_four_arm_bound():
    from onpolicy.scripts.train.run_stage3_local_exploration import identity
    assert len(ARMS) == 4 and all(a["mode"] in MODES for a in ARMS.values())
    manifest = {"automatic_finalblind": False, "automatic_confirmation": False, "arms": ARMS}
    manifest["manifest_sha256"] = identity(manifest)
    assert identity(manifest) == manifest["manifest_sha256"]
    manifest["automatic_finalblind"] = True
    assert identity(manifest) != manifest["manifest_sha256"]


def test_actual_engine_actor_loss_excludes_deterministic_plane():
    from types import SimpleNamespace
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    ac = torch.nn.Module()
    ac.actors = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.)) for _ in range(3)])
    ac.critic_param = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(ac.critic_param.weight)
    def likelihood(graph, h, active, op, site, actions, **kwargs):
        lp = torch.nn.functional.logsigmoid(torch.stack(list(ac.actors))).expand(len(graph), -1)
        return lp, torch.zeros_like(lp), torch.ones_like(lp), h
    def values(graph, h, active, op, site, actions, **kwargs):
        return ac.critic_param.weight[0, 0].expand(len(graph), 3)
    policy = SimpleNamespace(ac=ac, evaluate_actions=likelihood, evaluate_values=values,
        actor_optimizer=torch.optim.Adam([{"params": [p], "name": str(i)} for i, p in enumerate(ac.actors)], lr=.001),
        critic_optimizer=torch.optim.Adam(ac.critic_param.parameters(), lr=.001))
    engine = ResearchEngine.__new__(ResearchEngine)
    engine.policy, engine.device = policy, torch.device("cpu")
    engine.exploration, engine.policy_updates = MODES["R"], 0
    engine.norms = {i: SimpleNamespace(normalize=lambda x: x) for i in range(3)}
    state = {"graph": None, "hidden": np.zeros((3, 1, 1), np.float32), "active": np.ones(3),
        "op": np.zeros(3), "site": np.zeros(3), "roles": np.arange(3), "action": np.zeros((3, 3)),
        "old_logp": np.full(3, np.log(.5)), "mask": np.ones(3), "value": np.zeros(3), "time": 0.}
    trajectory = {"case_id": "test", "makespan": 90., "completed": True, "states": [state],
        "exploration": MODES["R"], "policy_updates": 0, "behavior_deterministic": False, "forced_replay": False}
    result = engine.update([trajectory], "source", 100., epochs=1)
    assert ac.actors[0].item() == 0  # excluded even though this toy parameter is trainable
    assert all(p.item() > 0 for p in ac.actors[1:])
    assert ac.critic_param.weight.item() < 0
    assert result["epochs"][0]["post_update"]["roles"]["0"]["decisions"] == 0


def test_shared_validator_durable_cache_and_tune_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from onpolicy.scripts.train import stage3_local_worker as worker
    from onpolicy.utils.stage3_research import digest_file, digest_json, atomic_json
    cases = [{"path": "case", "profile": "stress_joint", "distribution": "ood_stress"}]
    manifest = {"splits": {"tune": cases}, "source_costs": {"case": 100.},
                "manifest_sha256": "protocol", "source": {"sha256": "c0"},
                "code": {"sha256": "code"}, "contract_sha256": "contract"}
    checkpoint = tmp_path / "model.pt"
    torch.save({"exploration": MODES["R"], "diagnostic_only": False,
                "protocol_sha256": "protocol", "source_sha256": "c0", "training_episodes": 480}, checkpoint)
    queue = LocalEvaluationQueue(tmp_path / "queue")
    request = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest_file(checkpoint),
               "training_episodes": 480, "cases": cases, "cases_sha256": digest_json(cases),
               "code_sha256": "code", "contract_sha256": "contract", "tau": .3, "seed": 42,
               "evaluation_protocol": EVAL_PROTOCOL, "training_exploration": MODES["R"]}
    queue.submit({**request, "request_id": "first"})
    queue.submit({**request, "request_id": "second"})
    calls = []
    runner = SimpleNamespace(load=lambda path: None, configure_exploration=lambda mode: None,
                             policy=SimpleNamespace(ac=SimpleNamespace(tau=.3)), close=lambda: None)
    monkeypatch.setattr(worker, "verify", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "verify_cases", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "engine", lambda *args: runner)
    def evaluate(*args):
        calls.append(1)
        return [{"case_id": "case", "completed": True, "makespan": 98.,
                 "profile": "stress_joint", "distribution": "ood_stress"}]
    monkeypatch.setattr(worker, "evaluate", evaluate)
    atomic_json(queue.root / "STOP", {})
    heartbeat = SimpleNamespace(update=lambda **kwargs: None)
    worker.validator(manifest, queue.root, heartbeat)
    assert len(calls) == 1 and queue.poll("first")["ok"] and queue.poll("second")["ok"]
    wrong = [{"path": "finalblind", "profile": "stress_joint", "distribution": "ood_stress"}]
    queue.submit({**request, "request_id": "forbidden", "cases": wrong, "cases_sha256": digest_json(wrong)})
    with pytest.raises(ValueError, match="Tune-only"):
        worker.validator(manifest, queue.root, heartbeat)
    assert (queue.root / "failed/forbidden.json").exists()


def test_completed_fixed_endpoint_and_tail_guard_are_required():
    from onpolicy.utils.stage3_research import pilot_gate
    good = {"gain_fraction": .012, "completion_rate": 1.,
            "distributions": {"ood_stress": {"regression_fraction": 0}},
            "profiles": {"stress_joint": {"regression_fraction": 0}},
            "tail_makespan": 99., "source_tail_makespan": 100.}
    bad = copy.deepcopy(good)
    bad["gain_fraction"] = 0.
    assert pilot_gate([bad, good, good])
    assert not pilot_gate([good, good, bad])
    bad = copy.deepcopy(good)
    bad["tail_makespan"] = 102.
    assert not pilot_gate([good, bad])


def test_training_checkpoint_elite_and_async_request_integration(tmp_path, monkeypatch):
    """Tiny mocked simulator tests orchestration; NOT an experiment admission."""
    from types import SimpleNamespace
    from onpolicy.scripts.train import stage3_local_worker as worker
    from onpolicy.utils.stage3_research import atomic_json, read_json
    cases = [{"path": name, "name": name, "content_sha256": name,
              "profile": "stress_joint", "distribution": "ood_stress"} for name in ("a", "b")]
    training = {"pilot_seed": 4, "passes": 2, "episodes": 32, "queue_pending_per_arm": 2,
                "ppo_epochs": 1, "tbptt_steps": 8, "post_update_hard_kl": .04, "elite_interval": 1,
                "elite_lr": 1e-6, "save_every_groups": 1, "eval_every_groups": 1}
    manifest = {"root": str(tmp_path), "splits": {"train_pilot120": cases}, "training": training,
                "source_costs": {"a": 100., "b": 100.}, "manifest_sha256": "test-only",
                "source": {"sha256": "c0"}}
    atomic_json(tmp_path / "pilot_admission.json", {"eligible_arms": ["R_PPO_E"]})
    atomic_json(tmp_path / "contract/result.json", {"actor_lr": 5e-6})
    responses, updates = {}, []
    summary = {"gain_fraction": .012, "completion_rate": 1.,
               "regression_over_5pct_fraction": 0.,
               "distributions": {"ood_stress": {"regression_fraction": 0}},
               "profiles": {"stress_joint": {"regression_fraction": 0}},
               "tail_makespan": 99., "source_tail_makespan": 100.}
    class Runner:
        width = 8
        last_decision_stats = {}
        policy = SimpleNamespace(ac=torch.nn.Module())
        def set_actor_lr(self, lr):
            pass
        def rollout(self, batch, seeds, **kwargs):
            return [{"case_id": c["path"], "profile": c["profile"], "distribution": c["distribution"],
                     "actions": np.zeros((2, 104, 3), np.int32).tolist(), "states": [{}, {}],
                     "seed": s, "steps": 2, "completed": True, "makespan": 99.} for c, s in zip(batch, seeds)]
        def likelihood_metrics(self, probe):
            return {"1": {"accuracy": .5, "nll": 1.}}
        def update(self, group, mode, *args, **kwargs):
            updates.append(mode)
            return {"costs": [r["makespan"] for r in group], "epochs": [{"post_update": {
                    "kl": 0., "roles": {str(r): {"kl": 0.} for r in range(3)}}}]}
        def save(self, path, **kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(kwargs, path)
        def close(self):
            pass
    def submit(manifest, checkpoint, arm, episodes, suffix):
        name = f"{arm}_e{episodes:06d}{suffix}"
        responses[name] = {"evaluation": {"summary": copy.deepcopy(summary)}}
        return name
    monkeypatch.setattr(worker, "engine", lambda *args: Runner())
    monkeypatch.setattr(worker, "LocalEvaluationQueue", lambda *args: SimpleNamespace(poll=lambda r: responses.get(r)))
    monkeypatch.setattr(worker, "submit", submit)
    output = tmp_path / "pilot/R_PPO_E"
    worker.train(manifest, output, "R_PPO_E", SimpleNamespace(update=lambda **kwargs: None))
    result = read_json(output / "result.json")
    assert result["passed"] and result["training_episodes"] == 32
    assert result["auxiliary_steps"] == 4 and updates.count("source") == 4 and updates.count("bc") == 4
    state = torch.load(output / "models/episodes_000032.pt", weights_only=False)
    assert state["next_group"] == 4 and len(state["elite_buffer"]) == 2 and sum(state["elite_usage"].values()) == 4
    assert state["validation_requests"] == result["requests"]
