from types import SimpleNamespace
import numpy as np
import pytest
import torch

from onpolicy.runner.shared.stage3_b_shared_b0_engine import BSharedB0Engine, PPOContractError
from onpolicy.utils.stage3_b0_throughput import (
    ThroughputEngine, WholeUpdateCache, lane_cpus, terminal_rng_index, collect_with_rng,
)


def test_lane_affinity_preserves_half_and_smt_pairs():
    allowed=set(range(30)) | set(range(64,94))
    for n in (1,2,4,8,16):
        groups=lane_cpus(allowed,n)
        assert set.union(*groups)==allowed
        assert sum(map(len,groups))==len(allowed)
        assert all((c+64 if c<64 else c-64) in g for g in groups for c in g)
    with pytest.raises(ValueError):
        lane_cpus({0,1,64},2)
    assert len(lane_cpus(set(range(32)) | set(range(64,96)),32))==32


def test_capacity_window_stops_only_after_measured_events(tmp_path):
    from onpolicy.utils.stage3_b0_throughput import LaneProgress,CapacityWindowComplete
    progress=LaneProgress(tmp_path/'lane.json',stop_step=26)
    progress.update(rollout_step=1,live_trajectories=16)
    assert progress.events==0
    with pytest.raises(CapacityWindowComplete):
        progress.update(rollout_step=26,live_trajectories=16)
    assert progress.events==400 and progress.measured_start is not None


def test_tuned_deployment_uses_the_same_canonical_identity_encoder():
    from onpolicy.scripts.train.deploy_stage3_b_shared_b0_tuned import digest
    from onpolicy.utils.stage3_research import digest_json
    value=dict(name='完全共享',parameters=dict(global_batch=128,actor_lr=1e-5))
    assert digest(value)==digest_json(value)
    with pytest.raises(ValueError):digest(dict(value=float('nan')))


def test_complete_trial_selection_serializes_without_mutating_or_admitting_failed_trials():
    import copy,json
    from onpolicy.scripts.train.autotune_stage3_b_shared_b0 import choose_complete_trial
    rows=[dict(passed=True,seconds_per_visit=9.,name='safe'),
          dict(passed=True,seconds_per_visit=12.,name='slower'),
          dict(passed=False,seconds_per_visit=1.,name='invalid')]
    original=copy.deepcopy(rows)
    selected=choose_complete_trial(rows,parent_manifest_sha256='study')
    assert selected['name']=='safe' and rows==original
    assert json.loads(json.dumps(selected))['trials']==rows
    with pytest.raises(ValueError):choose_complete_trial(rows[2:])


def test_full_trial_rejects_capacity_or_numerical_failures_but_surfaces_other_errors(tmp_path,monkeypatch):
    from onpolicy.scripts.train import autotune_stage3_b_shared_b0 as tuner
    fixture=tmp_path/'reference.pt';fixture.write_bytes(b'synthetic unit fixture')
    args=SimpleNamespace(output=tmp_path,module=tmp_path/'module.py',launcher=tmp_path/'launcher.py',
        manifest=tmp_path/'manifest.json',reference=tmp_path,trajectories=fixture)
    params=dict(environments=32,lanes=16,microbatch=16,cache_mib=4096)
    for index,marker in enumerate(('OutOfMemoryError','AssertionError','unexpected ValueError')):
        def invoke(args,command,log_path,*,result_path,**kwargs):
            result_path.parent.mkdir(parents=True)
            if 'sample' in command:
                (result_path.parent/'trajectories.pt').write_bytes(b'synthetic retained input')
                result=dict(passed=True,seconds=1.)
                tuner.write(result_path,result)
                return result
            log_path.write_text(marker)
            raise RuntimeError('synthetic child rejection')
        monkeypatch.setattr(tuner,'run_command',invoke)
        if index==2:
            with pytest.raises(RuntimeError,match='synthetic'):tuner.full_trial(args,params,index)
        else:
            result=tuner.full_trial(args,params,index)
            assert not result['passed']
            assert not (tmp_path/f'full_{index}_32/sample/trajectories.pt').exists()


def test_capacity_window_tracks_joint_late_peak_and_keeps_behavior_hidden_states():
    from onpolicy.scripts.train.autotune_stage3_b_shared_b0 import dense_window
    template=[dict(states=[dict(graph=(100 if 48<=i<56 else 1),hidden=i) for i in range(80)]),
              dict(states=[dict(graph=3,hidden=-i) for i in range(40)])]
    group,window=dense_window(template,4,lambda graph:graph)
    assert window['start_step']==48 and window['peak_tbptt_step']==48
    assert window['active_microbatch']==2 and window['events']==64
    assert group[0]['states'][0]['hidden']==48
    group[0]['states'][0]['hidden']=999
    assert template[0]['states'][48]['hidden']==48


@pytest.mark.parametrize('size',[32,64,128,256,512])
def test_capacity_schedule_preserves_all_visits_and_partial_epoch_commits(size):
    from onpolicy.utils.stage3_b_shared_b0 import schedule,budget,episodes_at_cursor
    from onpolicy.utils.stage3_full_data import schedule as original
    cases=[]
    for distribution,n in (('iid',480),('ood_stress',108),('ood_scale',12)):
        cases.extend(dict(path=f'{distribution}/{i}',content_sha256=f'{distribution}-{i}',
                          distribution=distribution) for i in range(n))
    plan=schedule(cases,42,size)
    baseline=original(cases,42)
    def flatten(rows):
        return [(c['path'],s,v) for row in rows for c,s,v in zip(row['cases'],row['seeds'],row['visit_ids'])]
    assert flatten(plan)==flatten(baseline)
    manifest=dict(recipe=dict(global_batch=size))
    limits=budget(manifest)
    assert len(plan)==limits['global_batches']
    assert plan[-1]['training_episodes']==7680
    assert limits['actor_updates']==2*len(plan)
    for i,row in enumerate(plan,1):
        assert len(row['cases'])==len(row['visit_ids'])<=size
        assert row['training_episodes']==episodes_at_cursor(manifest,i)
    assert all(plan[i*limits['batches_per_epoch']-1]['training_episodes']==i*960 for i in range(1,9))
    if size!=32:
        with pytest.raises(ValueError):original(cases,42,batch_size=size)


def test_terminal_rng_matches_last_original_group_and_ties():
    rows=[{'steps':3000} for _ in range(16)]+[{'steps':10} for _ in range(16)]
    rows[18]['steps']=rows[27]['steps']=40
    assert terminal_rng_index(rows)==27
    assert terminal_rng_index([{'steps':9},{'steps':8}])==0


def test_observer_keeps_all_live_environments_and_each_terminal_rng(monkeypatch):
    from onpolicy.utils import stage3_b0_throughput as mod
    counter=[0]
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda *_:torch.tensor([counter[0]]))
    monkeypatch.setattr(mod,'capture_rng',lambda:dict(rng_cuda=torch.tensor([-1]),cpu_marker=7))
    def forward(*args,**kw):counter[0]+=1;return counter[0]
    def pool_call(commands):
        if commands[0][1]=='reset':return {i:None for i,_,_ in commands}
        return {i:(None,None,np.array([done]),None) for i,_,done in commands}
    runner=SimpleNamespace(pool=SimpleNamespace(call=pool_call),policy=SimpleNamespace(get_actions=forward))
    def rollout(cases,seeds,**kwargs):
        runner.pool.call([(0,'reset',None),(1,'reset',None)])
        runner.policy.get_actions();runner.policy.get_actions()
        runner.pool.call([(0,'step',True),(1,'step',False)])
        runner.policy.get_actions()
        runner.pool.call([(1,'step',True)])
        return [dict(steps=1),dict(steps=2)]
    runner.rollout=rollout
    rows,states=collect_with_rng(runner,[{},{}],[1,2],None)
    assert [s['rng_cuda'].item() for s in states]==[1,3]
    assert runner.pool.call is pool_call and runner.policy.get_actions is forward


def metric_runner(invalid=None):
    runner=ThroughputEngine.__new__(ThroughputEngine)
    runner.device=torch.device('cpu')
    runner.config=dict(microbatch=8,tbptt=8)
    runner.policy=SimpleNamespace(ac=torch.nn.Linear(1,1))
    def replay(group):
        for step in range(11):
            mask=torch.tensor([[1.,1.,0.,1.]])
            lp=torch.tensor([[.01*step,-.03*step,0.,.007*step]])
            if invalid=='nonfinite' and step==10:lp[0,0]=float('nan')
            expected=mask.clone()
            if invalid=='mask' and step==10:mask[0,1]=0.
            states=[dict(old_logp=np.zeros(4,np.float32),mask=expected[0].numpy(),
                         roles=np.array([0,1,2,1]))]
            yield step,[0],states,None,lp,mask
    runner._replay=replay
    return runner


def test_deferred_metrics_preserve_exact_scalars_and_empty_roles():
    original=BSharedB0Engine.replay_metrics(metric_runner(),[{}])
    actual=ThroughputEngine.replay_metrics(metric_runner(),[{}])
    assert actual==original
    assert actual['roles']['2']['decisions']==0


@pytest.mark.parametrize('invalid',['nonfinite','mask'])
def test_deferred_checks_reject_invalid_last_partial_window(invalid):
    with pytest.raises(PPOContractError):
        ThroughputEngine.replay_metrics(metric_runner(invalid),[{}])


def test_whole_update_cache_preserves_nested_inputs_then_releases():
    cache=WholeUpdateCache(None,4096,frozen_features=False)
    with cache.group():
        cache.entries['sentinel']=object()
        with cache.group():
            assert 'sentinel' in cache.entries
            cache.stats['input_hits']=12
        assert cache.active and 'sentinel' in cache.entries
    assert not cache.active and not cache.entries
    assert cache.last_report['input_hits']==12


def test_native_evaluation_cannot_use_parallel_collector():
    runner=ThroughputEngine.__new__(ThroughputEngine)
    runner.training=False
    with pytest.raises(ValueError,match='fixed12'):
        runner.parallel_collect([{}],[1])


def test_runtime_admission_binds_both_complete_proofs_and_code(tmp_path):
    from onpolicy.scripts.train.run_stage3_b_shared_b0_throughput import verify_runtime
    from onpolicy.utils.stage3_research import atomic_json,digest_file,digest_json
    module,launcher,tests=[tmp_path/name for name in ('module.py','launch.py','tests.xml')]
    for path in (module,launcher,tests):path.write_text(path.name)
    module_sha=digest_file(module)
    proof={}
    for kind in ('sample','update'):
        p=tmp_path/(kind+'.json')
        fields=(dict(visits=32,rows_exact=True,rng_exact=True,sampling_lanes=16) if kind=='sample'
                else dict(global_visits=32,ppo_epochs=2,microbatch=16))
        atomic_json(p,dict(passed=True,manifest_sha256='study',extension_sha256=module_sha,**fields))
        proof[kind]=dict(path=str(p),sha256=digest_file(p))
    runtime=dict(study_manifest_sha256='study',module_sha256=module_sha,
        launcher_sha256=digest_file(launcher),global_batch=32,ppo_epochs=2,
        sampling_environments=32,sampling_lanes=16,microbatch=16,input_cache_mib=4096,
        proofs=proof,tests=dict(path=str(tests),sha256=digest_file(tests)))
    runtime['runtime_sha256']=digest_json(runtime)
    p=tmp_path/'runtime.json';atomic_json(p,runtime)
    assert verify_runtime(p,{'manifest_sha256':'study'},module,launcher)==runtime
    module.write_text('changed')
    with pytest.raises(ValueError,match='identity'):
        verify_runtime(p,{'manifest_sha256':'study'},module,launcher)
    module.write_text(module.name)
    proof_path=tmp_path/'sample.json'
    value=__import__('json').loads(proof_path.read_text());value['rng_exact']=False
    atomic_json(proof_path,value)
    runtime['proofs']['sample']['sha256']=digest_file(proof_path)
    runtime['runtime_sha256']=digest_json({k:v for k,v in runtime.items() if k!='runtime_sha256'})
    atomic_json(p,runtime)
    with pytest.raises(ValueError,match='RNG'):
        verify_runtime(p,{'manifest_sha256':'study'},module,launcher)
