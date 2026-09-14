"""Contract tests for full coverage and full-depth execution-only acceleration."""
import copy
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta

from onpolicy.algorithms.utils.stage3_encoder import FullStage3Encoder
from onpolicy.envs.HKBZ.test.test_stage3_representation import graph_encoder
from onpolicy.utils.stage3_distributed import TrajectoryParallel, state_digest
from onpolicy.utils.stage3_full_data import ARMS, COUNTS, QUOTAS, schedule
from onpolicy.utils.stage3_performance import GroupExecutionCache


def cases():
    return [{"path":f"{d}/{i}","content_sha256":f"{d}/{i}","distribution":d}
            for d,n in COUNTS.items() for i in range(n)]


def test_full_coverage_quota_budget_and_fresh_repeat_seeds():
    plan = schedule(cases(),2026091201)
    assert plan == schedule(list(reversed(cases())),2026091201)
    assert len(plan)==240 and plan[-1]["training_episodes"]==7680
    streams=[]
    for epoch in range(8):
        rows=plan[epoch*30:(epoch+1)*30]
        visits=[c for r in rows for c in r["cases"]]
        assert len({c["path"] for c in visits})==600
        assert Counter(c["distribution"] for c in visits)==QUOTAS
        assert {r["data_epoch"] for r in rows}=={epoch+1}
        streams.extend((c["path"],s) for r in rows for c,s in zip(r["cases"],r["seeds"]))
    assert len(set(streams))==7680
    assert len(ARMS)==2 and [len(a["gpus"]) for a in ARMS.values()]==[4,4]
    assert sorted(g for a in ARMS.values() for g in a["gpus"])==list(range(8))


def test_pilot_or_changed_global_batch_rejected():
    with pytest.raises(ValueError): schedule(cases()[:120],1)
    with pytest.raises(ValueError): schedule(cases(),1,batch_size=16)


def test_visit_weighting_does_not_deduplicate_oversampled_cases(monkeypatch):
    sync=TrajectoryParallel()
    sync.world_size=4; sync.rank=0
    shards=[["stress","stress"],["stress","iid"],["iid2","iid3"],["iid4","iid5"]]
    monkeypatch.setattr(sync,"gather",lambda v: shards if isinstance(v,list) else [2.]*4)
    weights, denominator=sync.objective_weights(shards[0],2,visit_balanced=True)
    assert np.array_equal(weights,np.array([.5,.5])) and denominator==2
    legacy,_=sync.objective_weights(shards[0],2)
    assert not np.array_equal(weights,legacy)


def _four_rank_visit_gradient(rank,init_file):
    dist.init_process_group("gloo",init_method="file://"+init_file,rank=rank,world_size=4,
                            timeout=timedelta(seconds=60))
    try:
        sync=TrajectoryParallel()
        ids=["oversampled"]*8+[f"case{i}" for i in range(24)]
        indices=list(range(rank,32,4))
        weight,denominator=sync.objective_weights([ids[i] for i in indices],len(indices),visit_balanced=True)
        actor=torch.nn.Parameter(torch.tensor([.2,-.3]))
        critic=torch.nn.Parameter(torch.tensor([.5]))
        features=torch.arange(64,dtype=torch.float32).reshape(32,2)/64
        (features[indices].mv(actor).square()*torch.as_tensor(weight)).sum().backward()
        (critic.square()*len(indices)/denominator).backward()
        policy=SimpleNamespace(actor_optimizer=torch.optim.Adam([actor]),critic_optimizer=torch.optim.Adam([critic]))
        stats=dict(kl=0.,clip=0.,logp=0.,decisions=len(indices),mask_mismatch=0)
        sync.synchronize_update(policy,stats)
        expected=torch.nn.Parameter(torch.tensor([.2,-.3]))
        features.mv(expected).square().mean().backward()
        assert torch.allclose(actor.grad,expected.grad,atol=1e-7)
        assert torch.allclose(critic.grad,torch.ones(1)) and stats["decisions"]==32
    finally:
        dist.destroy_process_group()


def test_real_four_rank_visit_mean(tmp_path):
    mp.spawn(_four_rank_visit_gradient,args=(str(tmp_path/"gloo_store"),),nprocs=4,join=True)


def test_full_data_controller_uses_four_ranks_per_arm(tmp_path,monkeypatch):
    from onpolicy.scripts.train.run_stage3_full_policy import FullSuite
    from onpolicy.utils.stage3_full_policy import resource_plan
    from onpolicy.utils.stage3_research import atomic_json
    manifest={"root":str(tmp_path),"arms":ARMS,
              "resources":{"plan":resource_plan([[i,i+64] for i in range(64)])}}
    suite=FullSuite(manifest,tmp_path/"manifest.json")
    seen=[]
    def start(name,phase,placement,**options):
        key=f"{phase}/{name}"; output=tmp_path/key
        atomic_json(output/"result.json",{"completed":True})
        suite.jobs[key]={"process":SimpleNamespace(poll=lambda:0),"output":output,"options":options}
        seen.append((placement["gpu"],options["world_size"],options["until"]))
        return key
    monkeypatch.setattr(suite,"start",start);monkeypatch.setattr(suite,"check",lambda hb:None)
    result=suite.run_group("train","to7680",None,batch_size=32,until=7680)
    assert len(result["B_SHARED"])==len(result["C_PRIVATE"])==4
    assert seen==[(i,4,7680) for i in range(8)]


def test_shared_validation_request_rejects_contract_or_protocol_changes(tmp_path):
    from onpolicy.utils.stage3_full_data import publish, validate_request
    from onpolicy.utils.stage3_representation import RepresentationQueue
    from onpolicy.utils.stage3_research import digest_file
    checkpoint=tmp_path/"c0.pt"; checkpoint.write_bytes(b"source-fixture")
    manifest={"root":str(tmp_path),"source":{"path":str(checkpoint),"sha256":digest_file(checkpoint)},
        "splits":{"train_full600":[{"path":"case0"}]},"manifest_sha256":"protocol0",
        "contract_sha256":"H2F4soft","code":{"sha256":"code0"}}
    publish(manifest,checkpoint,"C0",0,"F_SHARED",split="train_full600",baseline=True)
    queue=RepresentationQueue(tmp_path/"validator")
    path,request=queue.claim()
    validate_request(manifest,request)
    assert queue.claim() is None
    for key,value in (("contract_sha256","H3hard"),("protocol_sha256","old"),
                      ("representation","F_PRIVATE"),("baseline",False)):
        changed=dict(request,**{key:value})
        with pytest.raises(ValueError): validate_request(manifest,changed)
    queue.finish(path,{"fixture":True})
    assert queue.pending_count()==0 and queue.poll(request["request_id"])["ok"]


def output_loss(out):
    outputs=list(out["role_encodings"].values()) if "role_encodings" in out else [out]
    return sum(v.square().mean() for role in outputs for v in role.values()
               if torch.is_tensor(v) and v.is_floating_point())


@pytest.mark.parametrize("variant",FullStage3Encoder.variants)
def test_full_depth_input_cache_keeps_all_gradients_and_invalidates_activations(variant):
    original,graph=graph_encoder()
    encoder=FullStage3Encoder(original,variant).eval()
    policy=SimpleNamespace(ac=SimpleNamespace(encoder=encoder),device=torch.device("cpu"))
    cache=GroupExecutionCache(policy,budget_mib=1,frozen_features=False)
    from torch_geometric.data import Batch
    expected=encoder(Batch.from_data_list([graph]))
    output_loss(expected).backward()
    expected_grad={n:p.grad.clone() for n,p in encoder.named_parameters() if p.grad is not None}
    encoder.zero_grad(set_to_none=True)
    with cache.group():
        batch=cache.prepare_graph([graph])
        out=cache.encode(batch,actor_grad=True)
        critic=cache.encode(cache.prepare_graph([graph]),actor_grad=False)
        assert state_digest(expected)==state_digest(out)==state_digest(critic)
        assert not output_loss(critic).requires_grad
        output_loss(out).backward()
        assert all(torch.equal(p.grad,expected_grad[n]) for n,p in encoder.named_parameters() if n in expected_grad)
        with torch.no_grad():
            for e in encoder.encoders: next(e.op_embedding.parameters()).add_(.01)
        fresh=cache.encode(cache.prepare_graph([graph]),actor_grad=True)
        assert state_digest(fresh)==state_digest(encoder(batch))
        assert state_digest(fresh)!=state_digest(expected)
        assert cache.stats["critic_reuses"]==1 and cache.stats["frozen_hits"]==0
        assert cache.bytes<=cache.limit
    assert not cache.active and not cache.entries and cache.last_actor is None


def test_full_depth_graph_cache_has_bounded_admission_and_exception_cleanup():
    original,graph=graph_encoder()
    encoder=FullStage3Encoder(original,"F_PRIVATE").eval()
    cache=GroupExecutionCache(SimpleNamespace(ac=SimpleNamespace(encoder=encoder),device="cpu"),
                              budget_mib=.00001,frozen_features=False)
    with pytest.raises(RuntimeError,match="fixture"):
        with cache.group():
            cache.encode(cache.prepare_graph([graph]),actor_grad=True)
            assert cache.bytes==0
            raise RuntimeError("fixture")
    assert not cache.active and not cache.entries and cache.last_actor is None


@pytest.mark.parametrize("variant",FullStage3Encoder.variants)
def test_gpu_real_semantic_history_decoder_cache_equivalence(variant,tmp_path):
    if os.environ.get("HKBZ_STAGE3_GPU_TEST")!="1":
        pytest.skip("requires an explicitly coordinated GPU canary")
    from onpolicy.scripts.train.stage3_full_data_worker import engine
    from onpolicy.utils.stage3_research import read_json, WORKSPACE_ROOT
    from onpolicy.utils.stage3_numerics import RUNTIME, training_state
    from onpolicy.envs.HKBZ.test.test_stage3_performance import nested_close
    manifest=read_json(WORKSPACE_ROOT / "result/hkbz_train_logs/stage3_conditional_improvement_all_20260911_r1_recovery1/manifest.json")
    manifest=copy.deepcopy(manifest)
    manifest["training"]["actor_lr"]=1e-5; manifest["numerics"]=RUNTIME
    runner=engine(manifest,variant,width=1)
    original_call=runner.pool.call
    def fixture_call(commands):
        rows=original_call(commands)
        for index,command,_ in commands:
            if command=="summary":
                rows[index]={**rows[index],"completed":True,"synthetic_prefix_fixture":True}
        return rows
    runner.pool.call=fixture_call
    try:
        case=manifest["splits"]["train_fit16"][0]
        trajectory=runner.rollout([case],[2026091201],retain=True,max_steps=16)[0]
        assert trajectory["history"]=="stable_request_identity_v1"
        assert any(np.any((s["mask"]>0)&(s["roles"]!=0)) for s in trajectory["states"])
        # Unit fixture ONLY; partial costs never enter research artifacts.
        group=[dict(trajectory,states=trajectory["states"][:16-i%3],case_id=f"fixture{i//2}") for i in range(8)]
        sources={t["case_id"]:t["makespan"]+100 for t in group}
        start=tmp_path/"cold.pt"
        runner.save(start,protocol_sha256="fixture",next_group=0,elite_buffer={},elite_usage={},auxiliary_steps=0)
        saved_cache=runner.execution_cache
        states=[]; updates=[]
        for fast in (False,True):
            runner.resume(start,protocol_sha256="fixture",exploration="J")
            runner.execution_cache=saved_cache if fast else None
            update=runner.update(group,"source",sources,epochs=2)
            updates.append(update)
            states.append(copy.deepcopy(training_state(runner)))
        assert nested_close(states[0],states[1],atol=2e-6,rtol=2e-5)
        assert updates[1]["execution_cache"]["critic_reuses"]>0
        assert updates[1]["execution_cache"]["frozen_hits"]==0
        for left,right in zip(updates[0]["epochs"],updates[1]["epochs"]):
            assert nested_close(left,right,atol=2e-6,rtol=2e-5)
    finally:
        runner.close()
