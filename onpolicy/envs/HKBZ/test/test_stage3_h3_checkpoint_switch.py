"""Batch-change accounting, real optimizer updates, and checkpoint-only switching."""
import copy
from pathlib import Path

import pytest
import torch

from onpolicy.utils.stage3_h3_continuation import MB128_PROFILE, recipe, planned_new_steps, schedule, resources
from onpolicy.utils.stage3_h3_frozen import recipe as parent_recipe, bind
from onpolicy.utils.stage3_b_shared_b0 import recipe as b0_recipe
from onpolicy.utils.stage3_h3_resize import rebind_payload, verify_recipe_change
from onpolicy.utils.stage3_h3_checkpoint_switch import boundary_commit
from onpolicy.utils.stage3_research import atomic_json, digest_json, read_json


def recipes():
    parent = parent_recipe(b0_recipe())
    args = (parent,resources(0,'GPU-test','0-31,64-95'),'C03')
    old = recipe(*args,execution='single',single_profile='single_parent_env384_v1')
    new = recipe(*args,execution='single',single_profile=MB128_PROFILE,optimizer_resize_after_epoch=2)
    return old,new


def test_historical_steps_and_data_schedule_are_not_recomputed():
    from onpolicy.envs.HKBZ.test.test_stage3_h3_frozen import cases
    old,new = recipes()
    assert [planned_new_steps(new,e) for e in range(9)] == [0,12,24,30,36,42,48,54,60]
    assert planned_new_steps(old,8) == 96
    assert schedule(cases(),old) == schedule(cases(),new)
    verify_recipe_change(old,new)


@pytest.mark.parametrize('boundary',[None,0,8,2.0,True])
def test_resize_needs_an_explicit_valid_boundary(boundary):
    with pytest.raises(ValueError):
        recipe(parent_recipe(b0_recipe()),resources(0,'GPU-test','0-31,64-95'),'C03',
               execution='single',single_profile=MB128_PROFILE,optimizer_resize_after_epoch=boundary)


@pytest.mark.parametrize('field,value',[('optimizer_minibatch',96),('microbatch',64),
    ('train_tau',.1),('actor_lr',2e-5),('rollout_workers',192),('ppo_epochs',4)])
def test_batch_resize_rejects_unrequested_recipe_changes(field,value):
    old,new = recipes();new[field]=value
    with pytest.raises(ValueError):verify_recipe_change(old,new)


def test_formal_128_batch_applies_six_real_adam_steps():
    from onpolicy.envs.HKBZ.test.test_stage3_h3_frozen import toy
    runner,small = toy()
    _,runner.config = recipes();runner.physical_microbatch=128
    rows=[]
    for i in range(384):
        row=copy.deepcopy(small[i%4]);row['visit_id']=f'visit-{i}';rows.append(row)
    result=runner.update_logical(rows,{r['case_id']:100. for r in rows},logical_id='toy',shuffle_seed=19)
    assert result['actual_ppo_steps']==result['planned_ppo_steps']==runner.policy_updates==6
    assert result['completed_passes']==2
    assert all(int(v['step'])==6 for v in runner.policy.actor_optimizer.state.values())
    assert all(len(row['visit_indices'])==128 for row in result['minibatches'])


def test_rebind_native_checkpoint_changes_only_execution_metadata():
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    root=Path(__file__).resolve().parents[4]/'result/hkbz_train_logs/stage3_h3_single_parent_20260918_r3_env384_gpu0'
    if not (root/'arms/C03/commits/epoch_0001.json').exists():
        pytest.skip('Native committed fixture unavailable')
    old=read_json(root/'arms/C03/manifest.json');new=copy.deepcopy(old)
    new['recipe'].update(execution_profile=MB128_PROFILE,optimizer_minibatch=128,microbatch=128,
                         physical_microbatch_candidates=[128],optimizer_resize_after_epoch=2)
    new['manifest_sha256']='new-arm'
    commit=read_json(root/'arms/C03/commits/epoch_0001.json')
    before=torch.load(commit['checkpoint']['path'],map_location='cpu',weights_only=False)
    after=rebind_payload(before,old,new,commit)
    excluded={'manifest_sha256','recipe_sha256','physical_microbatch'}
    compare_states({k:v for k,v in before.items() if k not in excluded},
                   {k:v for k,v in after.items() if k not in excluded},exact=True)
    assert after['physical_microbatch']==128 and before['physical_microbatch']==64
    assert after['next_batch']==1 and after['policy_updates']==108


@pytest.fixture
def boundary(tmp_path):
    old,_=recipes();root=tmp_path/'old';root.mkdir()
    atomic_json(root/'manifest.json',{'root':str(root)})
    arm={'recipe':old};arm['manifest_sha256']=digest_json(arm)
    atomic_json(root/'arms/C03/manifest.json',arm)
    checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'fixture-checkpoint')
    row=dict(epoch=2,manifest_sha256=arm['manifest_sha256'],ppo_budget_complete=True,
             new_ppo_steps=24,inherited_ppo_steps=96,cumulative_ppo_steps=120,training_episodes=768,
             checkpoint=bind(checkpoint))
    atomic_json(root/'update.json',row)
    row['update']=bind(root/'update.json')
    request=dict(origin=bind(root/'manifest.json'),after_epoch=2)
    return request,row,root/'arms/C03/commits/epoch_0002.json'


def test_checkpoint_file_without_atomic_commit_does_not_trigger(boundary):
    request,row,path=boundary
    assert boundary_commit(request) is None
    atomic_json(path,row)
    assert boundary_commit(request)['cumulative_ppo_steps']==120


@pytest.mark.parametrize('field,value',[('ppo_budget_complete',False),('new_ppo_steps',12),
    ('epoch',1),('cumulative_ppo_steps',114),('manifest_sha256','foreign')])
def test_partial_or_foreign_commit_cannot_trigger_switch(boundary,field,value):
    request,row,path=boundary;row[field]=value;atomic_json(path,row)
    with pytest.raises(ValueError):boundary_commit(request)


def test_missing_boundary_never_signals_the_training_process(tmp_path,monkeypatch):
    import onpolicy.scripts.train.stage3_h3_checkpoint_switch as switch
    import onpolicy.utils.stage3_memory_runtime as memory
    request=dict(root=str(tmp_path),request_sha256='test',after_epoch=2,deadline_unix=0)
    path=tmp_path/'request.json';atomic_json(path,request)
    monkeypatch.setattr(switch,'verify_request',lambda _:None)
    monkeypatch.setattr(switch,'origin_alive',lambda _:True)
    monkeypatch.setattr(switch,'boundary_commit',lambda _:None)
    monkeypatch.setattr(memory,'verify_unprotected_memory',lambda :{})
    monkeypatch.setattr(switch.os,'kill',lambda *a:pytest.fail('No signal before a committed checkpoint'))
    with pytest.raises(TimeoutError):switch.run(path)
    assert read_json(tmp_path/'switch_status.json')['original_controller_running']
