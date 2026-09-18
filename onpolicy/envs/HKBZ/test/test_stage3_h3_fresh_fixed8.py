"""Fresh E8 fork with 128-wide PPO and a fixed eight-epoch outcome budget."""
import copy
import time

import pytest

from onpolicy.envs.HKBZ.test.test_stage3_h3_checkpoint_switch import recipes
from onpolicy.envs.HKBZ.test.test_stage3_h3_continuation import metric
from onpolicy.envs.HKBZ.test.test_stage3_h3_frozen import cases, toy
from onpolicy.utils.stage3_b_shared_b0 import recipe as b0_recipe
from onpolicy.utils.stage3_h3_frozen import recipe as parent_recipe, bind
from onpolicy.utils.stage3_h3_continuation import (
    recipe, resources, schedule, planned_new_steps, validate_recipe,
    MB128_FRESH_PROFILE, FIXED_EPOCHS_POLICY,
)
from onpolicy.utils.stage3_research import atomic_json, read_json


def fresh():
    return recipe(parent_recipe(b0_recipe()),resources(0,'GPU-test','0-31,64-95'),'C03',
        execution='single',single_profile=MB128_FRESH_PROFILE,stopping_policy=FIXED_EPOCHS_POLICY)


def test_fresh_budget_and_schedule_keep_original_parent_continuation():
    r=fresh();old,_=recipes()
    assert 'optimizer_resize_after_epoch' not in r
    assert [planned_new_steps(r,e) for e in range(9)] == list(range(0,49,6))
    assert schedule(cases(),r) == schedule(cases(),old)
    assert r['microbatch'] == r['optimizer_minibatch'] == 128
    assert (r['hard_kl'],r['soft_kl'],r['gradient_clip']) == (.04,.02,1.)


@pytest.mark.parametrize('field,value', [('optimizer_resize_after_epoch',2),('microbatch',64),
    ('stopping_policy','disable_all_safety'),('execution_profile','single_parent_env384_mb128_v1')])
def test_fresh_contract_rejects_boundary_or_safety_drift(field,value):
    r=fresh();r[field]=value
    with pytest.raises(ValueError):validate_recipe(r)


def test_fresh_formal_epoch_executes_six_adam_updates():
    runner,small=toy();runner.config=fresh();runner.physical_microbatch=128
    rows=[]
    for i in range(384):
        row=copy.deepcopy(small[i%4]);row.update(visit_id=f'visit-{i}',logical_id='fresh');rows.append(row)
    result=runner.update_logical(rows,{r['case_id']:100. for r in rows},logical_id='fresh',shuffle_seed=19)
    assert result['actual_ppo_steps'] == result['planned_ppo_steps'] == runner.policy_updates == 6
    assert result['completed_passes'] == 2
    assert all(int(v['step'])==6 for v in runner.policy.actor_optimizer.state.values())


@pytest.mark.parametrize('incomplete_epoch', [None,3])
def test_fixed_budget_runs_past_bad_validation_but_stops_incomplete_ppo(tmp_path,incomplete_epoch):
    import onpolicy.scripts.train.run_stage3_h3_continuation as driver
    c=driver.Controller.__new__(driver.Controller)
    c.root=tmp_path;c.started=time.time();c.attempt=tmp_path/'attempt'
    c.m={'budget':{},'python':'python','source_root':str(tmp_path)}
    r=fresh();m={'root':str(tmp_path/'arms/C03'),'manifest_sha256':'arm','recipe':r}
    evaluated=[]
    for epoch in range(1,9):
        out=tmp_path/f'payload-{epoch}';out.write_text(str(epoch))
        atomic_json(tmp_path/'arms/C03/commits'/f'epoch_{epoch:04d}.json',
            dict(checkpoint=bind(out),update=bind(out),epoch=epoch,manifest_sha256='arm',
                 ppo_budget_complete=epoch!=incomplete_epoch,new_ppo_steps=6*epoch))
    c.validation=lambda arm,epoch,row,parent: evaluated.append(epoch) or {'versus_parent':metric(.1)}
    assert c.train_to('C03',(tmp_path/'arm.json',m),8,{}) == (incomplete_epoch is None)
    if incomplete_epoch is None:
        assert evaluated == [1,2,4,6,8]
        assert not (tmp_path/'arms/C03/early_stop.json').exists()
    else:
        assert evaluated == [1,2]
        assert read_json(tmp_path/'arms/C03/early_stop.json')['reason']=='incomplete_registered_PPO_budget'


def test_fixed_controller_keeps_capacity_gate_and_skips_epoch4_promotion(tmp_path,monkeypatch):
    import onpolicy.scripts.train.run_stage3_h3_continuation as driver
    import onpolicy.utils.stage3_memory_runtime as memory
    c=driver.Controller.__new__(driver.Controller)
    c.m={'recipe':fresh(),'execution_mode':'single','source_root':str(tmp_path),'python':'python'}
    c.root=tmp_path;c.attempt=tmp_path/'attempt';c.arms=('C03',)
    arm={'root':str(tmp_path/'arms/C03'),'manifest_sha256':'arm'}
    calls=[]
    c.baseline=lambda: {'epoch':8}
    c.arm_manifest=lambda a,p:(tmp_path/'arm.json',arm)
    c.canaries=lambda arms:calls.append('canary')
    def job(command,out,label):
        calls.append('capacity')
        atomic_json(out/'result.json',dict(passed=True,manifest_sha256='arm',backward_microbatch=128))
    c.job=job
    c.train_to=lambda a,item,e,p:calls.append(('train',e)) or True
    c.finish=lambda p,active:calls.append(('finish',active))
    monkeypatch.setattr(memory,'verify_unprotected_memory',lambda:{'passed':True})
    monkeypatch.setattr(driver,'extend_arms',lambda _:pytest.fail('Fixed budget must bypass outcome promotion'))
    c.run()
    assert calls==['canary','capacity',('train',8),('finish',['C03'])]
