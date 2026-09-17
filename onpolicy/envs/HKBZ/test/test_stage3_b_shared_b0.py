"""B0-specific invariants; no historical C0 artifacts or GPU required."""
import copy
import random
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
import pytest
import torch
from torch import nn

from onpolicy.runner.shared.stage3_b_shared_b0_engine import BSharedB0Engine, FROZEN_PREFIXES
from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
from onpolicy.utils.stage3_b_shared_b0 import (
    recipe, cpus, schedule, choose_candidate, normalization_moments, read_commit,
    check_resources, SOURCE_SHA,
)
from onpolicy.utils.stage3_research import atomic_json, digest_file


@pytest.fixture
def engine():
    r = BSharedB0Engine(device='cpu', create_pool=False)
    yield r
    r.close()


def test_strict_b0_native_shared_initialization_and_optimizer_ownership(engine):
    source = engine.bundle.checkpoint()['model']
    assert len(source) == 508
    assert all(torch.equal(v.cpu(), source[k]) for k, v in engine.policy.ac.state_dict().items())
    from onpolicy.algorithms.utils.gnn import HeteroGraphEncoder
    assert sum(isinstance(m, HeteroGraphEncoder) for m in engine.policy.ac.modules()) == 1
    assert all(p.requires_grad for p in engine.policy.ac.encoder.parameters())
    assert all(not p.requires_grad for n, p in engine.policy.ac.named_parameters() if n.startswith(FROZEN_PREFIXES))
    actors = {id(p) for g in engine.policy.actor_optimizer.param_groups for p in g['params']}
    critics = {id(p) for g in engine.policy.critic_optimizer.param_groups for p in g['params']}
    assert actors.isdisjoint(critics) and critics == {id(p) for p in engine.policy.ac.team_critic.parameters()}
    assert not engine.policy.actor_optimizer.state and not engine.policy.critic_optimizer.state
    assert engine.norm.debiasing_term == 0
    assert [g['lr'] for g in engine.policy.actor_optimizer.param_groups] == [1e-5, 2.5e-6, 1e-5, 5e-6]


def test_deployment_context_restores_trainability_tau_and_matching(engine):
    flags = {n:p.requires_grad for n,p in engine.policy.ac.named_parameters()}
    state = copy.deepcopy(engine.policy.ac.state_dict())
    assert not torch.backends.cudnn.allow_tf32
    with engine.evaluation_mode('AR'):
        assert engine.policy.ac.tau == .3 and not engine.policy.ac.device_global_matching
        assert not any(p.requires_grad for p in engine.policy.ac.encoder.parameters())
        assert torch.backends.cudnn.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert engine.policy.ac.tau == .03 and engine.policy.ac.device_global_matching
    assert {n:p.requires_grad for n,p in engine.policy.ac.named_parameters()} == flags
    assert all(torch.equal(v, state[k]) for k,v in engine.policy.ac.state_dict().items())


def test_only_gpu0_and_unoccupied_cpu_half(monkeypatch):
    r = recipe()
    allocations = [cpus(r[k+'_cpus']) for k in ('trainer','validator','controller')]
    assert set.union(*allocations) == set(range(32)) | set(range(64,96))
    assert sum(len(a) for a in allocations) == 64
    assert all(not a & b for i,a in enumerate(allocations) for b in allocations[i+1:])
    monkeypatch.setattr('os.sched_getaffinity', lambda _: allocations[0])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    with pytest.raises(ValueError, match='GPU0'):
        check_resources({'recipe':r},'trainer',cuda=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',r['gpu_uuid'])
    check_resources({'recipe':r},'trainer',cuda=False)
    monkeypatch.setattr('os.sched_getaffinity', lambda _: {32})
    with pytest.raises(ValueError,match='affinity'):
        check_resources({'recipe':r},'trainer',cuda=False)


def test_global32_schedule_keeps_every_visit_and_fresh_rng():
    cases=[]
    for distribution, n in (('iid',480),('ood_stress',108),('ood_scale',12)):
        cases.extend({'path':f'{distribution}/{i}','content_sha256':f'{distribution}-{i}',
                      'distribution':distribution} for i in range(n))
    batches=schedule(cases,recipe()['seed'],32)
    assert len(batches)==240 and batches[-1]['training_episodes']==7680
    for epoch in range(8):
        rows=batches[epoch*30:(epoch+1)*30]
        visits=[(c['path'],seed) for row in rows for c,seed in zip(row['cases'],row['seeds'])]
        assert len(visits)==len(set(visits))==960
        assert len({c for c,_ in visits})==600
        assert all(len(row['cases'])==32 for row in rows)


def test_normalization_uses_declared_training_visits_not_104_agents():
    ts=[{'distribution':'iid','makespan':100.,'times':[0.,50.]},
        {'distribution':'ood_stress','makespan':200.,'times':[0.]}]
    assert normalization_moments(ts)==[6.,-9.5,17.25]


def _risk_summary(gain):
    return dict(makespan=100*(1-gain),gain_fraction=gain,completion_rate=1.,
        distributions={'ood_stress':{'regression_fraction':0.}},
        profiles={'stress_joint':{'regression_fraction':0.}},
        tail_makespan=100.,source_tail_makespan=100.,regression_over_5pct_fraction=0.)


def test_transient_best_does_not_imply_stable_endpoint():
    curves=[{'epoch':i,'validation':_risk_summary(.03 if i==3 else (-.001 if i==8 else .005)),
             'tune':_risk_summary(.01)} for i in range(1,9)]
    result=choose_candidate(curves)
    assert result['selected']['epoch']==3 and result['open_confirmation']
    assert not result['last_two_epochs_stable']
    with pytest.raises(ValueError,match='eight'):
        choose_candidate(curves[:-1])
    curves[2]['tune']['regression_over_5pct_fraction']=.1
    result=choose_candidate(curves)
    assert result['selected']['epoch']==1 and not result['open_confirmation']


class ToyCache:
    def prepare_graph(self, graph): return graph


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(torch.tensor([.1,-.2,.3]))
        self.team_critic=nn.Linear(1,1)
    def _encode_graph(self, graph, actor_grad=False):
        # Deliberately has an actor dependency: the critic must detach it.
        return {'global_emb':torch.tensor(graph)[:,None]*(1+self.theta[0])}


def _toy_replay(self, group):
    for step in range(max(len(t['states']) for t in group)):
        ids=[i for i,t in enumerate(group) if step<len(t['states'])]
        states=[group[i]['states'][step] for i in ids]
        graph=[s['x'] for s in states]
        lp=torch.nn.functional.logsigmoid(torch.tensor(graph)[:,None]*self.policy.ac.theta)
        mask=torch.tensor(np.stack([s['mask'] for s in states]))
        yield step,ids,states,graph,lp,mask


@pytest.mark.parametrize('microbatch',[1,2,3])
def test_visit_mean_negative_advantages_and_critic_detach_against_analytic_gradient(microbatch):
    r=object.__new__(BSharedB0Engine)
    r.config=recipe();r.device=torch.device('cpu');r.cache=ToyCache()
    ac=ToyActor();r.policy=SimpleNamespace(ac=ac)
    from onpolicy.utils.valuenorm import ValueNorm
    r.norm=ValueNorm(1);r._replay=MethodType(_toy_replay,r)
    theta=ac.theta.detach().clone()
    records=[];expected=torch.zeros_like(theta)
    for name,cost,features in [('a',800.,[1.]),('a',800.,[2.,3.,4.]),('b',1400.,[1.,2.])]:
        states=[]
        advantage=.01*(1000.-cost)
        for x in features:
            states.append(dict(x=x,old_logp=torch.nn.functional.logsigmoid(theta*x).numpy(),
                mask=np.ones(3,np.float32),roles=np.arange(3),time=0.))
            expected-=advantage*(1-torch.sigmoid(theta*x))*x/3
        records.append(dict(case_id=name,makespan=cost,states=states))
    r._backward(records,{'a':1000.,'b':1000.},microbatch=microbatch)
    assert torch.allclose(ac.theta.grad,expected,atol=2e-6,rtol=0)
    assert ac.team_critic.weight.grad is not None


def _optimizer_step(r):
    r.policy.actor_optimizer.zero_grad(set_to_none=True)
    r.policy.critic_optimizer.zero_grad(set_to_none=True)
    loss=sum(g['params'][0].square().mean() for g in r.policy.actor_optimizer.param_groups)
    loss+=next(r.policy.ac.team_critic.parameters()).square().mean()
    loss.backward()
    r.policy.actor_optimizer.step();r.policy.critic_optimizer.step();r.policy_updates+=1


def test_checkpoint_restores_nonempty_adam_norm_rng_and_rejects_partial(engine,tmp_path):
    engine.set_normalization([10.,-20.,80.])
    _optimizer_step(engine)
    path=tmp_path/'before.pt'
    engine.save(path,manifest_sha256='test',next_batch=1)
    random_before=random.getstate();numpy_before=np.random.get_state();torch_before=torch.get_rng_state()
    _optimizer_step(engine)
    expected=copy.deepcopy(engine.policy.ac.state_dict())
    expected_adam=copy.deepcopy(engine.policy.actor_optimizer.state_dict())
    random.random();np.random.random();torch.rand(3)
    assert engine.resume(path,manifest_sha256='test')==1
    compare_states(random.getstate(),random_before,exact=True)
    compare_states(np.random.get_state(),numpy_before,exact=True)
    assert torch.equal(torch.get_rng_state(),torch_before)
    _optimizer_step(engine)
    compare_states(expected,engine.policy.ac.state_dict(),exact=True)
    compare_states(expected_adam,engine.policy.actor_optimizer.state_dict(),exact=True)
    assert engine.policy.actor_optimizer.state
    partial=tmp_path/'partial.pt'
    engine.save(partial,manifest_sha256='test',next_batch=1,diagnostic=True)
    with pytest.raises(ValueError,match='identity'):
        engine.resume(partial,manifest_sha256='test')
    with pytest.raises(ValueError,match='identity'):
        engine.resume(path,manifest_sha256='other')


def test_committed_batch_requires_checkpoint_update_and_exact_cursor(tmp_path):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'owned-test-checkpoint')
    update=tmp_path/'update.json';atomic_json(update,{'actor_updates':60})
    path=tmp_path/'commit.json'
    commit=dict(manifest_sha256='m',schedule_sha256='s',next_batch=30,training_episodes=960,
        actor_updates=60,checkpoint=str(checkpoint),checkpoint_sha256=digest_file(checkpoint),
        update=str(update),update_sha256=digest_file(update))
    atomic_json(path,commit)
    assert read_commit({'manifest_sha256':'m','schedule_sha256':'s'},path)['next_batch']==30
    checkpoint.write_bytes(b'changed-after-commit')
    with pytest.raises(ValueError,match='commit'):
        read_commit({'manifest_sha256':'m','schedule_sha256':'s'},path)


def test_resume_reconciles_auxiliary_ar_after_epoch_commit(monkeypatch,tmp_path):
    import onpolicy.utils.stage3_b_shared_b0 as contract
    submissions=[]
    monkeypatch.setattr(contract,'read_commit',lambda m,p:{'checkpoint':str(p)+'.pt'})
    def publish(m,checkpoint,episodes,split,**kwargs):
        value=(episodes,split,kwargs.get('decoder','H'))
        submissions.append(value)
        return value
    monkeypatch.setattr(contract,'submit_evaluation',publish)
    ids=contract.reconcile_epoch_requests({'root':str(tmp_path)},120)
    assert len(ids)==9
    assert (3840,'validation','AR') in ids
    assert (3840,'tune','H') in ids
    assert all(ep<=3840 for ep,_,_ in ids)


def test_replay_applies_native_per_agent_terminal_reset():
    r=object.__new__(BSharedB0Engine)
    r.config=recipe();r.device=torch.device('cpu')
    def evaluate(graph,h,active,op,site,actions,**kwargs):
        lp=h.reshape(len(graph),2)
        return lp,torch.tensor(0.),torch.ones_like(lp),h+1
    r.policy=SimpleNamespace(evaluate_actions=evaluate)
    states=[]
    for i in range(3):
        states.append(dict(graph=None,hidden=np.zeros((2,1,1),np.float32),active=np.ones(2),
            op=np.zeros(2),site=np.zeros(2),action=np.zeros((2,3)),roles=np.arange(2),
            done=np.asarray([i==0,False])))
    actual=[lp.detach().numpy().tolist() for _,_,_,_,lp,_ in r._replay([{'states':states}])]
    assert actual==[[[0.,0.]],[[0.,1.]],[[1.,2.]]]


def test_native_evaluation_retains_finished_slots_and_original_steps():
    r=object.__new__(BSharedB0Engine)
    r.config=recipe();r.device=torch.device('cpu');r.policy_updates=0
    r.environment_config=lambda case,seed: {'seed':seed}
    def info(step):
        return dict(active_agents=np.ones(104),agent_types=np.zeros(104),
            last_op_indices=np.zeros(24),last_site_indices=np.zeros(24),env_total_time=step)
    steps=[0,0];forward_batches=[];step_batches=[];active_masks=[]
    class Pool:
        def call(self, commands):
            if commands[0][1]=='reset':
                assert [c[2]['_stage3_native_reset_seed'] for c in commands]==[50000,60000]
                return {i:(None,None,info(0)) for i,_,_ in commands}
            if commands[0][1]=='summary':
                return {i:dict(completed=True,makespan=100.,steps=steps[i]) for i,_,_ in commands}
            step_batches.append([i for i,_,_ in commands])
            result={}
            for i,_,_ in commands:
                steps[i]+=1
                result[i]=(None,None,np.full(104,steps[i]>=i+1),info(steps[i]))
            return result
    def get_actions(graph,hidden,active,*args,**kw):
        forward_batches.append(len(graph));active_masks.append(active.copy())
        n=len(graph)
        return (None,torch.zeros((n,104,3)),torch.zeros((n,104)),
                torch.tensor(hidden)+1,torch.ones((n,104)))
    r.pool=Pool();r.policy=SimpleNamespace(ac=SimpleNamespace(tau=.3),get_actions=get_actions)
    cases=[dict(path=f'case{i}',profile='balanced',distribution='iid') for i in range(2)]
    result=r._collect(cases,[1,1],deterministic=True,retain=True,decoder='H',native=False,
                      heartbeat=None,record_times=True)
    assert forward_batches==[2,2] and step_batches==[[0,1],[0,1]]
    assert not active_masks[1][0].any() and active_masks[1][1].all()
    assert [len(t['states']) for t in result]==[1,2]
    assert [t['steps'] for t in result]==[1,2]


def test_interrupted_evaluation_replays_original_fixed_groups(tmp_path):
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import evaluate_cases
    r=recipe();calls=[]
    cases=[dict(path=f'case{i}',content_sha256=str(i)) for i in range(24)]
    class Runner:
        width=16
        def rollout(self, group, seeds, **kw):
            calls.append([c['path'] for c in group])
            return [dict(case_id=c['path'],makespan=100.,completed=True) for c in group]
    m=dict(manifest_sha256='m',recipe=r)
    hb=SimpleNamespace(update=lambda **kw:None)
    evaluate_cases(m,Runner(),cases,tmp_path,hb)
    assert calls==[[f'case{i}' for i in range(12)],[f'case{i}' for i in range(12,24)]]
    (tmp_path/'0003.json').unlink();(tmp_path/'0016.json').unlink()
    calls.clear()
    evaluate_cases(m,Runner(),cases,tmp_path,hb)
    assert calls==[[f'case{i}' for i in range(12)],[f'case{i}' for i in range(12,24)]]
