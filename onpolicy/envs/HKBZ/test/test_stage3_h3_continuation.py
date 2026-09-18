"""Causal comparison, probability provenance, full-state fork and promotion contracts."""
import copy
import math
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from onpolicy.envs.HKBZ.test.test_stage3_h3_frozen import cases, toy
from onpolicy.runner.shared.stage3_h3_continuation_engine import H3ContinuationEngine, DecisionObserver
from onpolicy.runner.shared.stage3_h3_frozen_engine import model_digest
from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
from onpolicy.utils.stage3_b_shared_b0 import recipe as b0_recipe
from onpolicy.utils.stage3_h3_frozen import recipe as r0_recipe
from onpolicy.utils.stage3_h3_continuation import (
    check_resources, choose_parent, eligible, extend_arms, harmful, recipe, resources,
    schedule, select_candidate, validate_parent_payload, validate_recipe, reusable_evaluation,
)
from onpolicy.utils.stage3_research import digest_file, read_json


def config(arm='C03', gpu=0, cpu='0-31,64-95'):
    return recipe(r0_recipe(b0_recipe()), resources(gpu, f'GPU-unit-{gpu}', cpu), arm)


def metric(gap=-.005, **extra):
    return dict(completed=True, gap_fraction=gap, makespan=100*(1+gap), tail_ratio=.995,
                regression_over_5pct_fraction=.05, distributions={'iid':gap,'ood_stress':gap}, **extra)


def test_both_temperatures_have_identical_visits_and_fresh_replica_seeds():
    a,b=schedule(cases(),config()),schedule(cases(),config('T10'))
    assert a==b
    assert len({v for row in a for v in row['visit_ids']})==3072
    assert all(len(row['cases'])==384 and len({c['path'] for c in row['cases']})==240 for row in a)
    assert all(sum(c['distribution']=='iid' for c in row['cases'])==192 for row in a)
    changed=config();changed['seed']+=1
    assert schedule(cases(),changed)!=a


@pytest.mark.parametrize('gpu,cpu',[(0,'0-31,64-95'),(1,'32-63,96-127')])
def test_physical_gpu_and_cpu_half_are_explicit_not_cuda_index_zero(gpu,cpu,monkeypatch):
    from onpolicy.utils.stage3_b_shared_b0 import cpus
    r=config(gpu=gpu,cpu=cpu)
    monkeypatch.setattr('os.sched_getaffinity',lambda _:cpus(cpu))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',r['gpu_uuid'])
    check_resources(r,cuda=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',str(gpu))
    with pytest.raises(ValueError,match='UUID'):check_resources(r,cuda=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',r['gpu_uuid'])
    monkeypatch.setattr('os.sched_getaffinity',lambda _:set(range(128)))
    with pytest.raises(ValueError,match='escaped'):check_resources(r,cuda=False)


@pytest.mark.parametrize('field,value',[('train_tau',.3),('microbatch',32),('teacher_queries_allowed',1),
                                      ('optimizer_minibatch',240),('actor_lr',1e-4)])
def test_unregistered_algorithm_or_solver_changes_rejected(field,value):
    r=config();r[field]=value
    with pytest.raises(ValueError):validate_recipe(r)


def test_decision_observer_excludes_forced_choices_and_preserves_rng_and_inputs():
    obs=DecisionObserver(torch.device('cpu'));obs.begin_batch(['iid','ood_stress','iid'])
    logits=torch.tensor([[math.log(.9),math.log(.1)],[0.,-torch.inf],[math.log(.25),math.log(.75)]])
    before=logits.clone();rng=torch.get_rng_state().clone()
    obs.record(1,torch.arange(3),logits,torch.tensor([0,0,0]))
    stats=obs.result()['1']
    assert torch.equal(before,logits) and torch.equal(rng,torch.get_rng_state())
    assert stats['iid']['multi_candidate_decisions']==2
    assert stats['ood_stress']['multi_candidate_decisions']==0
    assert stats['ood_stress']['single_candidate_decisions']==1
    assert stats['iid']['pmax_mean']==pytest.approx(.825)
    assert stats['iid']['non_greedy_fraction']==.5
    assert 0<stats['iid']['normalized_entropy_mean']<1


def _temperature_replay(self,group):
    for step in range(max(len(t['states']) for t in group)):
        ids=[i for i,t in enumerate(group) if step<len(t['states'])]
        states=[group[i]['states'][step] for i in ids]
        graph=[s['x'] for s in states]
        lp=torch.nn.functional.logsigmoid(torch.tensor(graph)[:,None]*self.policy.ac.theta/self.policy.ac.tau)
        yield step,ids,states,graph,lp,torch.tensor(np.stack([s['mask'] for s in states]))


def temperature_toy(tau=.1):
    r,rows=toy();r.__class__=H3ContinuationEngine;r.config=config('T10' if tau==.1 else 'C03')
    r._replay=MethodType(_temperature_replay,r);r.policy.ac.tau=tau
    r.value_diagnostics=None;r.initialization_sha256='parent'
    for row in rows:
        row.update(tau=tau,behavior_tau=tau,advantage_mode='source_relative',behavior_model_sha256=model_digest(r.policy.ac))
        state=row['states'][0]
        state['old_logp']=torch.nn.functional.logsigmoid(r.policy.ac.theta.detach()*state['x']/tau).numpy()
    return r,rows


def test_temperature_replay_uses_actual_behavior_distribution_and_updates():
    r,rows=temperature_toy()
    result=r.update_logical(rows,{t['case_id']:100. for t in rows},logical_id='toy',shuffle_seed=7,
                            diagnostic=True,minibatch=2,microbatch=2)
    assert result['actual_ppo_steps']==4 and result['train_tau']==.1
    assert result['behavior_replay']['max_logp_error']==0
    assert r.policy.ac.tau==.1


@pytest.mark.parametrize('field,value',[('tau',.03),('behavior_tau',.03),('advantage_mode','state_value')])
def test_mismatched_temperature_or_estimator_never_reaches_adam(field,value):
    r,rows=temperature_toy();rows[0][field]=value
    with pytest.raises((ValueError,RuntimeError)):
        r.update_logical(rows,{t['case_id']:100. for t in rows},logical_id='toy',shuffle_seed=1,
                         diagnostic=True,minibatch=2,microbatch=2)
    assert r.policy_updates==0 and not r.policy.actor_optimizer.state


def test_state_value_advantage_is_fixed_and_detached_during_ppo():
    r,rows=temperature_toy();r.config['advantage']='state_value'
    for i,row in enumerate(rows):row['states'][0]['actor_advantage']=float(i-2)
    states=[t['states'][0] for t in rows]
    a=r.actor_advantage(rows,list(range(4)),states,{})
    assert torch.equal(a,torch.tensor([-2.,-1.,0.,1.])) and not a.requires_grad
    with torch.no_grad():r.policy.ac.team_critic.weight.add_(100)
    assert torch.equal(a,r.actor_advantage(rows,list(range(4)),states,{}))


def test_parent_choice_is_validation_only_with_registered_fallback():
    assert choose_parent(metric(),metric(-.004),metric(-.001))=='epoch8'
    assert choose_parent(metric(),metric(-.004),metric(.003))=='epoch6'
    bad=metric();bad['tail_ratio']=1.02
    assert choose_parent(bad,metric(),metric(-.001))=='epoch6'
    with pytest.raises(ValueError):choose_parent(bad,bad,metric())


def test_plateau_and_risk_gates_do_not_reward_mean_at_any_cost():
    assert extend_arms({'C03':metric(-.001),'T10':metric(-.002)})==[]
    assert extend_arms({'C03':metric(-.003),'T10':metric(-.008)})==['C03','T10']
    assert extend_arms({'C03':metric(-.006),'T10':metric(-.0065)})==['C03']
    bad=metric(-.02);bad['tail_ratio']=1.1
    assert not eligible(bad)
    assert harmful([bad,bad]) and not harmful([metric(),bad])


def test_champion_ties_choose_tail_then_earlier_without_tune_scores():
    a=dict(arm='C03',epoch=4,checkpoint='a',versus_parent=metric(-.005),tune_score=999)
    b=dict(arm='T10',epoch=8,checkpoint='b',versus_parent=metric(-.0055),tune_score=1)
    b['versus_parent']['tail_ratio']=1.001
    assert select_candidate([a,b])['checkpoint']=='a'


def test_eval_cache_requires_identical_checkpoint_cases_rules_and_suite():
    q=dict(expected=[],manifest_sha256='suite',checkpoint={'sha256':'a'},cases=['a','b'],decoder='canonical',label='a')
    assert reusable_evaluation(q,dict(q,label='b'))
    for key,value in [('checkpoint',{'sha256':'b'}),('cases',['b','a']),('decoder','legacy'),
                      ('manifest_sha256','other'),('expected',[{'case':'a'}])]:
        assert not reusable_evaluation(q,dict(q,**{key:value}))


PARENT=Path(__file__).resolve().parents[4]/'result/hkbz_train_logs/stage3_b_shared_h3_r0_frozen_iga_20260915_r1_gpu0'


@pytest.mark.skipif(not (PARENT/'manifest.json').exists(),reason='Local immutable H3 environment fixture')
def test_native_policy_observer_preserves_real_sampled_actions_probabilities_and_rng(tmp_path):
    from onpolicy.utils.stage3_b0_single_model import EnvironmentSlot
    from onpolicy.runner.shared.stage3_b_shared_b0_engine import authoritative_history
    old=read_json(PARENT/'manifest.json')
    r=recipe(old['recipe'],resources(0,old['recipe']['gpu_uuid'],'0-31,64-95'),'T10')
    runner=H3ContinuationEngine(old['frozen_manifest']['path'],config=r,device='cpu',
        runtime={'sampling_output':str(tmp_path/'sampling')})
    slots=[EnvironmentSlot(),EnvironmentSlot()]
    try:
        selected=[next(c for c in old['splits']['train'] if c['distribution']==d) for d in ('iid','ood_stress')]
        initial=[slot.execute('reset',runner.environment_config(c['path'],123+i)) for i,(slot,c) in enumerate(zip(slots,selected))]
        infos=[x[2] for x in initial];graph=[x[0] for x in initial]
        previous=np.full((2,104,3),-1,np.int64);hidden=np.zeros((2,104,1,64),np.float32)
        hist=authoritative_history(previous,infos)
        active=np.stack([i['active_agents'] for i in infos]).reshape(2,104)
        roles=np.stack([i['agent_types'] for i in infos]).reshape(2,104)
        def forward():
            with torch.no_grad():
                return runner.policy.get_actions(graph,hidden,active,hist[...,0],hist[...,1],
                    deterministic=False,agent_types=roles,return_decision_mask=True)
        before=torch.get_rng_state().clone();plain=forward();after=torch.get_rng_state().clone()
        torch.set_rng_state(before)
        observer=DecisionObserver(torch.device('cpu'));observer.begin_batch([c['distribution'] for c in selected])
        runner.policy.ac.stage3_decision_observer=observer
        measured=forward();del runner.policy.ac.stage3_decision_observer
        assert torch.equal(after,torch.get_rng_state())
        for a,b in zip(plain,measured):assert torch.equal(a,b)
        assert sum(v['head_decisions'] for role in observer.result().values() for v in role.values())>0
    finally:
        for slot in slots:slot.close()
        runner.close()


@pytest.mark.skipif(not (PARENT/'manifest.json').exists(),reason='Local immutable R0 checkpoint integration fixture')
@pytest.mark.parametrize('arm',['C03','T10','fresh128'])
def test_real_checkpoint_fork_and_roundtrip_preserve_nonempty_adam_and_value_norm(tmp_path,arm):
    from onpolicy.utils.stage3_h3_continuation import MB128_FRESH_PROFILE, FIXED_EPOCHS_POLICY
    import random
    old=read_json(PARENT/'manifest.json')
    checkpoint=PARENT/'attempts/20260915T012204_2081156/train/models/epoch_0008.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    extra = dict(execution='single',single_profile=MB128_FRESH_PROFILE,
                 stopping_policy=FIXED_EPOCHS_POLICY) if arm == 'fresh128' else {}
    r=recipe(old['recipe'],resources(0,old['recipe']['gpu_uuid'],'0-31,64-95'),
             'C03' if arm == 'fresh128' else arm,**extra)
    runner=H3ContinuationEngine(old['frozen_manifest']['path'],config=r,device='cpu',
        runtime={'sampling_output':str(tmp_path/'sampling')})
    try:
        receipt=runner.fork_parent(checkpoint,expected_sha256=digest_file(checkpoint),parent_manifest=old,parent_epoch=8)
        assert receipt['inherited_updates']==96 and receipt['actor_optimizer_entries']>0
        assert runner.physical_microbatch == r['microbatch'] == receipt['physical_microbatch']
        assert receipt['parent_physical_microbatch'] == 64
        assert random.getstate() == payload['rng_python']
        compare_states(np.random.get_state(),payload['rng_numpy'],exact=True)
        assert torch.equal(torch.get_rng_state(),payload['rng_torch'])
        for key,actual in [('model',runner.policy.ac.state_dict()),('actor_optim',runner.policy.actor_optimizer.state_dict()),
                           ('critic_optim',runner.policy.critic_optimizer.state_dict()),('value_normalizer',runner.norm.state_dict())]:
            compare_states(payload[key],actual,exact=True)
        path=tmp_path/'fork.pt';runner.save(path,manifest_sha256='new-arm',next_batch=0)
        runner.policy.ac.tau=.99;runner.policy.actor_optimizer.state.clear();runner.norm.running_mean.zero_()
        assert runner.resume(path,manifest_sha256='new-arm')==0
        assert runner.policy.ac.tau==r['train_tau']
        compare_states(payload['actor_optim'],runner.policy.actor_optimizer.state_dict(),exact=True)
        compare_states(payload['value_normalizer'],runner.norm.state_dict(),exact=True)
        broken=copy.deepcopy(payload);broken['complete_logical_rollout']=False
        with pytest.raises(ValueError):validate_parent_payload(broken,old,8)
    finally:runner.close()
