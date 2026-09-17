"""Requested 240-environment continuation and multiplexed CPU contracts."""
import copy
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from onpolicy.envs.HKBZ.test.test_stage3_b0_restart import continuation
from onpolicy.utils.stage3_b_shared_b0 import training_schedule, budget, schedule
from onpolicy.utils.stage3_b0_single_model import EnvironmentSlot, CpuEnvironmentPool


@pytest.mark.parametrize('extra_completed', [False, True])
def test_requested_240_batch_preserves_mixed_history_and_every_visit(extra_completed):
    m = continuation()
    old = training_schedule(m)
    a = m['resume_amendment']
    if extra_completed:
        a.update(completed_batch_sizes=[32]*9+[64], completed_batches=10,
                 completed_visits=352, completed_actor_updates=20)
    m['recipe'].update(global_batch=240, microbatch=80)
    new = training_schedule(m)
    cursor = a['completed_batches']
    assert new[:cursor] == old[:cursor]
    flatten = lambda rows: [(c['path'],s,v) for row in rows
                           for c,s,v in zip(row['cases'],row['seeds'],row['visit_ids'])]
    assert flatten(new) == flatten(old)
    assert [len(row['cases']) for row in new[cursor:cursor+3]] == [240,240,128 if extra_completed else 192]
    assert len(new) == cursor+31
    assert budget(m)['actor_updates'] == 2*(cursor+31)
    assert new[-1]['training_episodes'] == 7680
    assert budget(m)['epoch_end_batches'] == list(range(cursor+3,cursor+32,4))
    assert len(schedule(m['splits']['train'],42,240)) == 32


def test_multiplexed_slots_keep_environment_and_global_rng_independent(monkeypatch):
    class Env:
        def __init__(self, config): self.closed=False;self.steps=0
        def reset(self, seed):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            return self.step(None)
        def step(self, action):
            self.steps += 1
            return (self.steps,random.random(),np.random.random(),torch.rand(()).item())
        def close(self): self.closed=True
    monkeypatch.setattr('onpolicy.envs.HKBZ.environment.AircraftScheduleEnv', Env)
    expected=[]
    for seed in (11,22):
        env=Env({});expected.append([env.reset(seed)]+[env.step(None) for _ in range(3)])
    slots=[EnvironmentSlot(),EnvironmentSlot()]
    actual=[[slots[i].execute('reset',dict(seed=seed))] for i,seed in enumerate((11,22))]
    for _ in range(3):
        for i in (1,0): actual[i].append(slots[i].execute('step',None))
    assert actual==expected
    old=slots[0].env
    slots[0].execute('reset',dict(seed=33))
    assert old.closed and slots[0].env is not old and slots[1].env.steps==4


def test_multiplexed_ipc_routes_sparse_live_slots_and_preserves_order():
    class Connection:
        def __init__(self): self.commands=[]
        def send_bytes(self, payload): self.commands=pickle.loads(payload)
        def poll(self, timeout): return True
        def recv_bytes(self):
            return pickle.dumps((True,{i:(c,p) for i,c,p in self.commands}))
    pool=object.__new__(CpuEnvironmentPool)
    pool.width=240;pool.worker_count=64;pool.multiplexed=True;pool.timeout=1
    pool.connections=[Connection() for _ in range(64)]
    commands=[(i,'step',f'action-{i}') for i in [239,0,64,128,192,65,129]]
    result=pool.call(commands)
    assert result=={i:(c,p) for i,c,p in commands}
    assert [i for i,_,_ in pool.connections[0].commands]==[0,64,128,192]
    assert [i for i,_,_ in pool.connections[1].commands]==[65,129]
    with pytest.raises(ValueError,match='outside'):
        pool.call([(240,'step',None)])


def test_requested_capacity_requires_explicit_new_parameters(tmp_path):
    from onpolicy.utils.stage3_b0_restart import MODE, verify_requested_capacity
    from onpolicy.utils.stage3_research import atomic_json,digest_file
    f=tmp_path/'commit.json';atomic_json(f,dict(checkpoint_sha256='checkpoint',next_batch=9))
    params=dict(environments=240,microbatch=80,lanes=1,cache_mib=4096,environment_processes=64)
    m=dict(recipe=dict(memory_max_gib=114),resume_amendment=dict(mode=MODE,
        numerical_equivalence_gate='waived_by_user',user_instruction='240/80',completed_batches=9,
        parent_commit=dict(path=str(f),sha256=digest_file(f)),requested_parameters=params,
        sampling_probe=dict(path='proof',sha256='proof'),parent_manifest_sha256='parent',
        capacity_admission='requested_capacity_window_v1'))
    runtime=dict(global_batch=240,microbatch=80,input_cache_mib=4096,environment_processes=64,
        sampling_lanes=1,proofs=dict(sample=m['resume_amendment']['sampling_probe']))
    for broken in ({},dict(passed=True,scope='requested_capacity_window_v1',parameters=dict(params,microbatch=120))):
        with pytest.raises(ValueError,match='capacity evidence failed'):
            verify_requested_capacity(m,runtime,broken)


@pytest.mark.parametrize('microbatch', [80, 120, 240])
@pytest.mark.parametrize('subset', [False, True])
def test_capacity_admits_only_the_matching_measured_microbatch(tmp_path, microbatch, subset):
    from onpolicy.utils.stage3_b0_restart import MODE, verify_requested_capacity
    from onpolicy.utils.stage3_research import atomic_json, digest_file
    from onpolicy.scripts.train.probe_stage3_b0_requested_capacity import CODE_FILES
    source = tmp_path / 'source'
    checksums = {}
    for relative in CODE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
        checksums[relative] = digest_file(path)
    def bound(name, payload):
        path = tmp_path / name
        atomic_json(path, payload)
        return dict(path=str(path), sha256=digest_file(path))
    params = dict(environments=240, microbatch=microbatch, lanes=1, cache_mib=4096,
                  environment_processes=64, encoder_activation_checkpoint=True)
    parent = bound('parent.json', dict(source_root=str(source)))
    commit = bound('commit.json', dict(checkpoint_sha256='checkpoint', next_batch=11))
    fixture = bound('fixture.json', dict(capacity=True))
    sample_binding = dict(path='probe.json', sha256='probe')
    m = dict(source_root=str(source), recipe=dict(memory_max_gib=114),
        resume_amendment=dict(mode=MODE, numerical_equivalence_gate='waived_by_user',
            user_instruction='switch after durable checkpoint', completed_batches=11,
            capacity_admission='requested_capacity_window_v1', requested_parameters=params,
            parent_manifest=parent, parent_manifest_sha256='parent', parent_commit=commit,
            sampling_probe=sample_binding))
    runtime = dict(global_batch=240, microbatch=microbatch, sampling_lanes=1,
        input_cache_mib=4096, environment_processes=64, encoder_activation_checkpoint=True,
        proofs=dict(sample=sample_binding), module=str(source/'onpolicy/utils/stage3_b0_single_model.py'))
    proof = dict(passed=True, scope='requested_capacity_window_v1', parameters=dict(params),
        architecture='single_model_batched_v1', rng_contract='checkpoint_global_torch_live_slot_order_v1',
        parent_manifest_sha256='parent', checkpoint_sha256='checkpoint', next_batch=11,
        checkpoint_restore_exact=True, cuda_rng_restored_exact=True, updates_executed=0,
        environment_parity=dict(passed=True), peak_gpu_mib=40000, gpu_total_mib=49140,
        code_files=checksums, fixture=fixture,
        sampling=dict(environment_count=240, environment_processes=64, model_copies=1,
            repeated_forward_and_rng_exact=True, window=dict(environment_steps=6240),
            full_trajectories_tested=False, replay=dict(decisions=100, max_logp_error=1e-5),
            projected_full_memory_bytes=80*2**30),
        update=dict(finite_gradients=True, optimizer_updates=0, full_update_tested=False,
            window=dict(active_microbatch=microbatch, events=microbatch*32)))
    if subset:
        from onpolicy.envs.HKBZ.test.test_stage3_b0_subset import subset_study
        from onpolicy.utils.stage3_b0_subset import epoch_rows
        subset_root = tmp_path/'subset'
        subset_root.mkdir()
        _, study, selected, _ = subset_study(subset_root)
        m['training_subset'] = study['training_subset']
        rows = epoch_rows(selected, 0)[:240]
        proof['training_subset'] = dict(selection=study['training_subset']['selection'],
            cases_sha256=selected['cases_sha256'], sampling_cases=[r[0]['path'] for r in rows],
            sampling_seeds=[r[1] for r in rows], formal_training_visits=0)
        relative = 'onpolicy/utils/stage3_b0_subset.py'
        (source/relative).write_text(relative)
        checksums[relative] = digest_file(source/relative)
    verify_requested_capacity(m, runtime, proof)
    if subset:
        for key in ('sampling_cases', 'sampling_seeds'):
            stale = copy.deepcopy(proof)
            stale['training_subset'][key].reverse()
            with pytest.raises(ValueError, match='selected Train240 data'):
                verify_requested_capacity(m, runtime, stale)
        stale = copy.deepcopy(proof)
        del stale['training_subset']
        with pytest.raises(ValueError, match='selected Train240 data'):
            verify_requested_capacity(m, runtime, stale)
    stale = copy.deepcopy(proof)
    stale['parameters']['microbatch'] = 120 if microbatch == 80 else 80
    with pytest.raises(ValueError, match='capacity evidence failed'):
        verify_requested_capacity(m, runtime, stale)
    over = dict(proof, peak_gpu_mib=48000)
    with pytest.raises(ValueError, match='capacity evidence failed'):
        verify_requested_capacity(m, runtime, over)


def test_lower_cache_retains_the_same_update_implementation(monkeypatch):
    from onpolicy.utils.stage3_b0_single_model import ThroughputEngine
    from onpolicy.utils.stage3_b0_throughput import ThroughputEngine as Base
    def initialize(self,*args,runtime,**kwargs):
        assert runtime['input_cache_mib']==4096
        self.cache=SimpleNamespace(limit=4096*2**20)
    monkeypatch.setattr(Base,'__init__',initialize)
    r=ThroughputEngine(runtime=dict(input_cache_mib=1024,microbatch=80))
    assert r.runtime['microbatch']==80 and r.runtime['input_cache_mib']==1024
    assert r.cache.limit==1024*2**20
    assert ThroughputEngine.update is Base.update


def test_encoder_recomputation_preserves_outputs_gradients_and_rng():
    from onpolicy.utils.stage3_b0_single_model import checkpoint_encoder
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(5,7);self.calls=0
        def forward(self,x):
            self.calls+=1
            y=self.layer(x).tanh()
            return dict(global_emb=y,op_nodes=y.square())
    torch.manual_seed(443)
    reference=Encoder().eval();actual=copy.deepcopy(reference);checkpoint_encoder(actual)
    x=torch.randn(13,5)
    rng=torch.get_rng_state()
    before=reference(x);after=actual(x)
    assert all(torch.equal(before[k],after[k]) for k in before)
    sum(v.sum() for v in before.values()).backward()
    sum(v.sum() for v in after.values()).backward()
    assert all(torch.equal(a.grad,b.grad) for a,b in zip(reference.parameters(),actual.parameters()))
    assert actual.calls>reference.calls and torch.equal(rng,torch.get_rng_state())
    calls=actual.calls
    with torch.no_grad():
        values=actual(x)
    assert actual.calls==calls+1 and all(torch.equal(values[k],before[k]) for k in before)
