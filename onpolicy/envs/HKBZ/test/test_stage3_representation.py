from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import os
import numpy as np
import pytest
import torch

from onpolicy.utils.stage3_representation import (ARMS, schedule, allocate_resources,
    RepresentationQueue, screen_candidates, endpoint_pass)
from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json
from onpolicy.algorithms.utils.stage3_encoder import Stage3Encoder


def graph_encoder():
    from onpolicy.envs.HKBZ.test.test_stage1_learning_baselines import _graph, _ac_config
    from onpolicy.algorithms.utils.gnn import HeteroGraphEncoder
    config = _ac_config()
    enc = HeteroGraphEncoder(
        config["common_cfg"], config["encoder_cfg"]["gnn_cfg"], config["encoder_cfg"]["gff_cfg"])
    graph = _graph(.2)
    graph["operation", "needs", "device"].edge_index = torch.tensor([[0, 1], [0, 1]])
    graph["operation", "needs", "device"].edge_attr = torch.ones(2, 1)
    graph["device", "can_serve", "request"].edge_index = torch.tensor([[0, 1], [0, 1]])
    graph["device", "can_serve", "request"].edge_attr = torch.ones(2, 1)
    graph.global_features = torch.ones(1, 24)
    return enc.eval(), graph


@pytest.mark.parametrize("variant", Stage3Encoder.variants)
def test_initial_encoder_exact_and_state_round_trip(variant):
    enc, graph = graph_encoder()
    original = enc(graph)
    rng = torch.get_rng_state().clone()
    candidate = Stage3Encoder(enc, variant).eval()
    assert torch.equal(rng, torch.get_rng_state())
    output = candidate(graph)
    for key in original:
        assert torch.equal(output[key], original[key]), key
    if variant == "E2":
        for role in output["role_encodings"].values():
            assert all(torch.equal(role[k], original[k]) for k in original)
    restored = Stage3Encoder(enc, variant).eval()
    restored.load_state_dict(candidate.state_dict(), strict=True)
    assert all(torch.equal(restored(graph)[k], output[k]) for k in original)
    ptrs = [p.data_ptr() for p in candidate.parameters()]
    assert len(ptrs) == len(set(ptrs))
    assert not {p.data_ptr() for p in enc.parameters()} & set(ptrs)
    assert not any(p.requires_grad for p in candidate.prefix.parameters())
    if variant == "E3":
        assert candidate.report()["e3_vs_e2_trainable_relative_error"] <= .05


def test_private_tail_gradient_isolation_and_residual_capacity_alive():
    enc, graph = graph_encoder()
    private = Stage3Encoder(enc, "E2").eval()
    private(graph)["role_encodings"]["1"]["request_nodes"].square().mean().backward()
    for i, tail in enumerate(private.tails):
        nonzero = any(p.grad is not None and p.grad.abs().sum() > 0 for p in tail.parameters())
        assert bool(nonzero) == (i == 1)
    control = Stage3Encoder(enc, "E3").eval()
    optimizer = torch.optim.Adam(control.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        control(graph)["global_emb"].square().sum().backward()
        if step == 1:
            assert control.residuals["global_emb"].input.weight.grad.abs().sum() > 0
        optimizer.step()


def test_paired_schedule_equal_exposure_not_stale_macro_batch():
    cases = [{"content_sha256": str(i), "path": str(i)} for i in range(120)]
    left, right = [schedule(cases, 2026090803, b) for b in ("T0", "T1")]
    assert len(ARMS) == 8 and len(left) == len(right) == 240
    for first in range(0, 240, 4):
        exposures = []
        for side, expected_cases in ((left, 1), (right, 4)):
            block = side[first:first+4]
            assert all(len(row["cases"]) == 8 and len(set(c["path"] for c in row["cases"])) == expected_cases for row in block)
            exposures.append(Counter((c["path"], s) for row in block for c, s in zip(row["cases"], row["seeds"])))
        assert exposures[0] == exposures[1]
        assert all(value == 1 for value in exposures[0].values())


def summary(gain=.011):
    return {"gain_fraction": gain, "completion_rate": 1., "regression_over_5pct_fraction": .05,
        "distributions": {"ood_stress": {"regression_fraction": .0}},
        "profiles": {"stress_joint": {"regression_fraction": .0}},
        "tail_makespan": 100., "source_tail_makespan": 100.}


def test_screen_limit_and_both_endpoints_required():
    results = {arm: {"training_episodes": 960, "summary": summary(.006+i*.001)}
               for i, arm in enumerate(ARMS)}
    assert screen_candidates(results) == ["E3_T1", "E3_T0"]
    assert not endpoint_pass([{"training_episodes": 1920, "summary": summary()}])
    assert endpoint_pass([{"training_episodes": e, "summary": summary()} for e in (1440, 1920)])
    bad = summary(); bad["regression_over_5pct_fraction"] = .051
    assert not endpoint_pass([{"training_episodes": e, "summary": bad} for e in (1440, 1920)])


@pytest.mark.parametrize("reserved,ngpu,ntrain", [(set(), 8, 8), (set(range(50,64)) | set(range(114,128)), 6, 6)])
def test_full_resources_preserve_baseline_and_share_validator_gpu(reserved, ngpu, ntrain):
    plan = allocate_resources([[i, i+64] for i in range(64)], reserved, list(range(ngpu)))
    assert len(plan["trainers"]) == ntrain and len(plan["validators"]) == 2
    occupied = set(plan["controller"])
    for p in plan["trainers"] + plan["validators"]:
        assert not occupied & set(p["cpus"]) and not reserved & set(p["cpus"])
        occupied.update(p["cpus"])
    assert occupied | reserved == set(range(128))
    assert {p["gpu"] for p in plan["validators"]} <= {p["gpu"] for p in plan["trainers"]}


def test_shared_queue_concurrent_claims_and_structure_identity(tmp_path):
    checkpoint = tmp_path/"model.pt"; checkpoint.touch()
    q = RepresentationQueue(tmp_path/"queue")
    req = {k: "x" for k in ("contract_sha256", "code_sha256", "protocol_sha256")}
    req.update(checkpoint=str(checkpoint), checkpoint_sha256=digest_file(checkpoint), cases=[], cases_sha256=digest_json([]),
               tau=.3, seed=42, representation="E0", evaluation_protocol={}, training_exploration={}, training_episodes=480)
    for i in range(12):
        q.submit({**req, "request_id": f"E0_T0_{i}"})
    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(lambda _: q.claim(), range(12)))
    assert len({path.name for path, _ in claims}) == 12
    assert q.claim() is None
    assert q.identity(req) != q.identity({**req, "representation": "E2"})


def test_initial_real_joint_decoder_equivalence():
    from onpolicy.runner.shared.stage3_research_engine import policy_args, environment_config, ResearchEngine
    from onpolicy.utils.stage3_research import SOURCE, read_json
    from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
    from onpolicy.algorithms.utils.stage3_encoder import install_encoder
    from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
    import yaml
    args = policy_args()
    policy = GNN_MAPPOPolicy(args, yaml.safe_load(open(args.ac_config)), device=torch.device("cpu"))
    policy.ac.load_state_dict(torch.load(SOURCE, map_location="cpu", weights_only=False)["model"], strict=True)
    policy.ac.eval()
    candidates = []
    for variant in Stage3Encoder.variants:
        other = copy.deepcopy(policy)
        install_encoder(other, variant); other.ac.eval()
        candidates.append(other)
    prior = read_json("result/hkbz_train_logs/stage3_local_exploration_half_20260906_r1/manifest.json")
    case = prior["splits"]["train_fit16"][0]
    env = AircraftScheduleEnv(environment_config(case["path"]))
    helper = ResearchEngine.__new__(ResearchEngine)
    hidden = np.zeros((1,104,1,64), np.float32)
    previous = np.full((1,104,3), -1, np.int64)
    try:
        obs, _, info = env.reset()
        resource_seen = False
        for _ in range(16):
            data = helper._inputs([obs], hidden, [info], previous)
            graph, h, active, op, site, roles = data
            with torch.no_grad():
                base = policy.get_actions(graph, h, active, op, site, agent_types=roles,
                                          deterministic=True, return_decision_mask=True)
                for other in candidates:
                    candidate = other.get_actions(graph, h, active, op, site, agent_types=roles,
                                                 deterministic=True, return_decision_mask=True)
                    for a,b in zip(base, candidate):
                        assert torch.allclose(a, b, atol=2e-6, rtol=2e-5)
            resource_seen |= bool(np.any(active.reshape(1,-1).astype(bool) & (roles != 0)))
            hidden = base[3].numpy(); previous = base[1].numpy().astype(np.int64)
            obs, _, done, info = env.step(previous[0])
            if np.all(done):
                break
        assert resource_seen
    finally:
        env.close()


def test_mixed_case_objective_is_case_mean_trajectory_mean_time_sum():
    from types import SimpleNamespace
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    from onpolicy.utils.stage3_local_exploration import MODES
    ac = torch.nn.Module()
    ac.theta = torch.nn.Parameter(torch.tensor(0.))
    ac.critic_param = torch.nn.Linear(1,1,bias=False)
    def likelihood(graph,h,active,op,site,actions,**kwargs):
        lp = torch.nn.functional.logsigmoid(ac.theta).expand(len(graph),1)
        return lp, torch.zeros_like(lp), torch.ones_like(lp), h
    runner = ResearchEngine.__new__(ResearchEngine)
    runner.device,runner.policy_updates,runner.exploration = torch.device("cpu"),0,MODES["J"]
    runner.norms = {i:SimpleNamespace(normalize=lambda x:x) for i in range(3)}
    runner.policy = SimpleNamespace(ac=ac, evaluate_actions=likelihood,
        evaluate_values=lambda graph,*args,**kwargs: ac.critic_param.weight.expand(len(graph),1),
        actor_optimizer=torch.optim.SGD([{"params":[ac.theta],"name":"actor"}],lr=.001),
        critic_optimizer=torch.optim.SGD(ac.critic_param.parameters(),lr=.001))
    state = {"graph":None,"hidden":np.zeros((1,1,1),np.float32),"active":np.ones(1),"op":np.zeros(1),
        "site":np.zeros(1),"roles":np.zeros(1),"action":np.zeros((1,3)),"old_logp":np.array([np.log(.5)]),
        "mask":np.ones(1),"value":np.zeros(1),"time":0.}
    # Case A has one two-step trajectory, B has three one-step trajectories.
    # Advantages A=1, B=-1; expected derivative = -.5*(2*.5)+.5*.5 = -.25.
    trajectories = [{"case_id":case,"makespan":cost,"completed":True,"states":[state]*length,
        "exploration":MODES["J"],"policy_updates":0,"behavior_deterministic":False,"forced_replay":False}
        for case,cost,length in [("A",100.,2),("B",200.,1),("B",200.,1),("B",200.,1)]]
    captured = []
    class Captured(Exception): pass
    def capture(epoch, ac, stats):
        captured.append(float(ac.theta.grad)); raise Captured()
    with pytest.raises(Captured):
        runner.update(trajectories,"source",{"A":200.,"B":100.},epochs=1,allow_multi_case=True,gradient_callback=capture)
    assert captured == pytest.approx([-.25])
    assert runner.policy_updates == 0 and float(ac.theta) == 0.
    with pytest.raises(ValueError, match="exactly one case"):
        runner.update(trajectories,"source",100.,epochs=1)


@pytest.mark.skipif(os.environ.get("HKBZ_STAGE3_GPU_TEST") != "1", reason="explicit free-GPU canary only")
@pytest.mark.parametrize("variant", Stage3Encoder.variants)
def test_gpu_checkpoint_backward_and_resume(variant, tmp_path):
    from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
    from onpolicy.runner.shared.stage3_research_engine import environment_config, gradient_mode
    from onpolicy.utils.stage3_research import read_json
    from onpolicy.utils.stage3_local_exploration import MODES
    from onpolicy.scripts.train.stage3_local_worker import nested_close
    runner = RepresentationEngine(variant, width=1, exploration="J", cuda_memory_fraction=.50)
    manifest = read_json("result/hkbz_train_logs/stage3_local_exploration_half_20260906_r1/manifest.json")
    case = manifest["splits"]["train_fit16"][0]
    hidden = np.zeros((1,104,1,64),np.float32)
    previous = np.full((1,104,3),-1,np.int64)
    try:
        obs,_,info = runner.pool.call([(0,"reset",environment_config(case["path"]))])[0]
        selected = None
        for _ in range(20):
            inputs = runner._inputs([obs],hidden,[info],previous)
            graph,h,active,op,site,roles = inputs
            with torch.no_grad():
                result = runner.policy.get_actions(graph,h,active,op,site,agent_types=roles,
                    deterministic=True,return_decision_mask=True)
            actions = result[1].cpu().numpy()
            if bool((result[-1] > 0).any()):
                selected = (inputs,actions)
                break
            hidden = result[3].cpu().numpy(); previous = actions
            obs,_,_,info = runner.pool.call([(0,"step",actions[0])])[0]
        assert selected is not None
        initial = tmp_path/f"{variant}.pt"
        runner.save(initial,protocol_sha256="test",next_group=0,elite_buffer={},elite_usage={},auxiliary_steps=0,
                    diagnostic_only=True)
        graph,h,active,op,site,roles = selected[0]
        def step():
            gradient_mode(runner.policy.ac)
            runner.policy.actor_optimizer.zero_grad(set_to_none=True)
            lp,_,mask = runner.policy.evaluate_actions(graph,h,active,op,site,selected[1],agent_types=roles,return_decision_mask=True)
            loss = -(lp*mask).sum(); loss.backward()
            grads = [p.grad for p in runner.policy.ac.encoder.parameters() if p.grad is not None]
            if variant == "E0": assert not grads
            else: assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
            assert all(p.grad is None for p in runner.policy.ac.encoder.prefix.parameters())
            runner.policy.actor_optimizer.step()
        step()
        expected = {k:v.detach().cpu().clone() for k,v in runner.policy.ac.state_dict().items()}
        optimizer = copy.deepcopy(runner.policy.actor_optimizer.state_dict())
        runner.resume(initial,protocol_sha256="test",exploration="J")
        step()
        assert nested_close(expected, runner.policy.ac.state_dict())
        assert nested_close(optimizer,runner.policy.actor_optimizer.state_dict())
        runner.assert_optimizer_ownership()
    finally:
        runner.close()
