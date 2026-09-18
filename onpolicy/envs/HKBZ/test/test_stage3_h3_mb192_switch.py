"""Checkpoint-boundary 128-to-192 accounting, admission and full-state migration."""
import copy
from pathlib import Path

import pytest
import torch

from onpolicy.envs.HKBZ.test.test_stage3_h3_fresh_fixed8 import fresh
from onpolicy.envs.HKBZ.test.test_stage3_h3_frozen import cases, toy
from onpolicy.utils.stage3_b_shared_b0 import recipe as b0_recipe
from onpolicy.utils.stage3_h3_frozen import recipe as parent_recipe, bind
from onpolicy.utils.stage3_h3_continuation import (
    MB192_PROFILE, FIXED_EPOCHS_POLICY, recipe, resources, schedule, planned_new_steps,
)
from onpolicy.utils.stage3_h3_resize import verify_recipe_change, import_committed_state, rebind_payload
from onpolicy.utils.stage3_h3_checkpoint_switch import boundary_commit, verify_request, request_identity
from onpolicy.utils.stage3_research import atomic_json, digest_json, read_json


def target(boundary=1):
    return recipe(parent_recipe(b0_recipe()),resources(0,'GPU-test','0-31,64-95'),'C03',
        execution='single',single_profile=MB192_PROFILE,optimizer_resize_after_epoch=boundary,
        stopping_policy=FIXED_EPOCHS_POLICY)


def test_192_preserves_six_historical_steps_and_counts_four_new_steps_per_epoch():
    old,new=fresh(),target()
    verify_recipe_change(old,new)
    assert [planned_new_steps(new,e) for e in range(9)] == [0,6,10,14,18,22,26,30,34]
    assert 96+planned_new_steps(new,8)==130
    assert schedule(cases(),old)==schedule(cases(),new)
    assert new['stopping_policy']==old['stopping_policy']==FIXED_EPOCHS_POLICY


@pytest.mark.parametrize('boundary',[None,0,8,1.0,True])
def test_192_requires_a_real_committed_boundary(boundary):
    with pytest.raises(ValueError):target(boundary)


@pytest.mark.parametrize('key,value',[('microbatch',128),('optimizer_minibatch',196),
    ('ppo_epochs',3),('rollout_workers',192),('stopping_policy',None),('seed',1),('actor_lr',2e-5)])
def test_192_rejects_unrequested_changes(key,value):
    new=target();new[key]=value
    with pytest.raises(ValueError):verify_recipe_change(fresh(),new)


def test_formal_192_batch_executes_four_adam_steps_with_complete_coverage():
    runner,small=toy();runner.config=target();runner.physical_microbatch=192
    rows=[]
    for i in range(384):
        row=copy.deepcopy(small[i%4]);row['visit_id']=f'visit-{i}';rows.append(row)
    result=runner.update_logical(rows,{r['case_id']:100. for r in rows},logical_id='toy',shuffle_seed=19)
    assert result['actual_ppo_steps']==result['planned_ppo_steps']==runner.policy_updates==4
    assert result['completed_passes']==2
    assert all(int(v['step'])==4 for v in runner.policy.actor_optimizer.state.values())
    for pass_id in (1,2):
        parts=[r['visit_indices'] for r in result['minibatches'] if r['ppo_pass']==pass_id]
        assert [len(p) for p in parts]==[192,192]
        assert sorted(i for p in parts for i in p)==list(range(384))


@pytest.fixture
def cpu_commit(tmp_path):
    # Real CPU Adam updates and serialization on a toy network; CUDA RNG is a fixture.
    runner,small=toy();runner.config=fresh();runner.physical_microbatch=128
    runner.initialization_sha256='parent';runner.policy_updates=96
    rows=[]
    for i in range(384):
        row=copy.deepcopy(small[i%4]);row.update(visit_id=f'visit-{i}',policy_updates=96);rows.append(row)
    runner.update_logical(rows,{r['case_id']:100. for r in rows},logical_id='toy',shuffle_seed=19)
    old_root=tmp_path/'old';arm_root=old_root/'arms/C03'
    old=dict(root=str(arm_root),recipe=fresh(),parent_checkpoint={'sha256':'parent'})
    old['manifest_sha256']=digest_json(old)
    atomic_json(arm_root/'manifest.json',old)
    atomic_json(old_root/'manifest.json',dict(root=str(old_root),execution_mode='single',
        recipe=fresh(),resources=resources(0,'GPU-test','0-31,64-95')))
    path=old_root/'checkpoint.pt';runner.save(path,manifest_sha256=old['manifest_sha256'],next_batch=1)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    payload['rng_cuda']=torch.arange(16,dtype=torch.uint8);torch.save(payload,path)
    row=dict(epoch=1,manifest_sha256=old['manifest_sha256'],ppo_budget_complete=True,
        new_ppo_steps=6,inherited_ppo_steps=96,cumulative_ppo_steps=102,training_episodes=384,
        checkpoint=bind(path))
    atomic_json(old_root/'update.json',row);row['update']=bind(old_root/'update.json')
    atomic_json(arm_root/'commits/epoch_0001.json',row)
    new=dict(old,root=str(tmp_path/'new/arms/C03'),recipe=target(),manifest_sha256='new-arm')
    return old_root,old,new,row,payload


def test_192_import_preserves_model_adam_rng_and_historical_cursor(cpu_commit,tmp_path):
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    root,old,new,commit,payload=cpu_commit
    suite=dict(root=str(tmp_path/'new'),recipe=target(),resume_origin=dict(epoch=1,
        arm_manifest=bind(root/'arms/C03/manifest.json'),
        commits=[bind(root/'arms/C03/commits/epoch_0001.json')]))
    mapping=import_committed_state(suite,new)
    restored=torch.load(mapping[commit['checkpoint']['sha256']]['path'],map_location='cpu',weights_only=False)
    metadata={'manifest_sha256','recipe_sha256','physical_microbatch'}
    compare_states({k:v for k,v in payload.items() if k not in metadata},
                   {k:v for k,v in restored.items() if k not in metadata},exact=True)
    assert restored['physical_microbatch']==192 and restored['policy_updates']==102
    assert restored['next_batch']==1
    receipt=read_json(tmp_path/'new/resume_migration.json')
    assert receipt['passed'] and receipt['next_epoch']==2 and receipt['remaining_epochs']==7
    assert read_json(Path(new['root'])/'commits/epoch_0001.json')['new_ppo_steps']==6


@pytest.mark.parametrize('field,value',[('next_batch',2),('physical_microbatch',64),
    ('policy_updates',108),('complete_logical_rollout',False),('diagnostic_only',True),('rng_cuda',None)])
def test_192_rejects_wrong_checkpoint_state(cpu_commit,field,value):
    _,old,new,row,payload=cpu_commit;payload[field]=value
    with pytest.raises(ValueError):rebind_payload(payload,old,new,row)


def test_192_switch_request_and_boundary_bind_the_fresh128_origin(cpu_commit):
    root,old,_,row,_=cpu_commit
    q=dict(origin=bind(root/'manifest.json'),target_profile=MB192_PROFILE,after_epoch=1,
        requested_minibatch=192,requested_microbatch=192,resources=resources(0,'GPU-test','0-31,64-95'),
        tests=bind(root/'update.json'),plan=bind(root/'update.json'),source_root=str(root),source_files={})
    q['request_sha256']=request_identity(q)
    assert verify_request(q)['recipe']==fresh()
    assert boundary_commit(q)['cumulative_ppo_steps']==102
    q['requested_microbatch']=128;q['request_sha256']=request_identity(q)
    with pytest.raises(ValueError,match='dimensions'):verify_request(q)


@pytest.mark.parametrize('capacity_width',[128,192])
def test_192_capacity_admission_precedes_fixed_eight_epoch_training(tmp_path,monkeypatch,capacity_width):
    import onpolicy.scripts.train.run_stage3_h3_continuation as driver
    import onpolicy.utils.stage3_memory_runtime as memory
    c=driver.Controller.__new__(driver.Controller)
    c.m={'recipe':target(),'execution_mode':'single','source_root':str(tmp_path),'python':'python'}
    c.root=tmp_path;c.attempt=tmp_path/'attempt';c.arms=('C03',)
    arm={'root':str(tmp_path/'arms/C03'),'manifest_sha256':'arm','recipe':target()}
    calls=[];c.baseline=lambda:{'epoch':8};c.arm_manifest=lambda a,p:(tmp_path/'arm.json',arm)
    c.canaries=lambda arms:calls.append('canary')
    def job(command,out,label):
        calls.append(label)
        atomic_json(out/'result.json',dict(passed=True,manifest_sha256='arm',backward_microbatch=capacity_width))
    c.job=job;c.train_to=lambda a,item,e,p:calls.append(('train',e)) or True
    c.finish=lambda p,active:calls.append(('finish',active))
    monkeypatch.setattr(memory,'verify_unprotected_memory',lambda:{'passed':True})
    monkeypatch.setattr(driver,'extend_arms',lambda _:pytest.fail('Effect gate must stay disabled'))
    if capacity_width==128:
        with pytest.raises(ValueError,match='192-wide'):c.run()
        assert calls==['canary','C03_capacity192']
    else:
        c.run()
        assert calls==['canary','C03_capacity192',('train',8),('finish',['C03'])]
