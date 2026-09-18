"""Same learning budget, disjoint resources and exact deferred replay reductions."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from onpolicy.envs.HKBZ.test.test_stage3_h3_continuation import config, temperature_toy
from onpolicy.envs.HKBZ.test.test_stage3_h3_frozen import cases
from onpolicy.utils.stage3_h3_continuation import (
    recipe, resources, schedule, validate_recipe, check_resources, DUAL_ENV96_PROFILE,
)
from onpolicy.utils.stage3_b_shared_b0 import cpus
from onpolicy.scripts.train.stage3_h3_dual_controller import DualController
from onpolicy.scripts.train.stage3_h3_cpu_guard import enforce_tree, process_identity
from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states


def dual(arm='C03'):
    return recipe(config(),resources(0,'GPU-unit-0','0-31,64-95'),arm,execution='dual')


def test_two_arms_partition_the_authorized_half_and_keep_the_same_learning_budget():
    a,b=dual(),dual('T10')
    left,right=cpus(a['trainer_cpus']),cpus(b['trainer_cpus'])
    assert not left&right and left|right==cpus(a['cpus'])
    assert schedule(cases(),a)==schedule(cases(),b)==schedule(cases(),config())
    for r in (a,b):
        assert r['global_batch']//r['optimizer_minibatch']*r['ppo_epochs']==12
        assert r['rollout_workers']==128 and r['environment_processes']==32
        assert r['cuda_allocator_mib']==18*1024


def test_env96_profile_preserves_case_seed_schedule_and_optimizer_budget():
    allocation = resources(0, 'GPU-unit-0', '0-31,64-95')
    records = [recipe(config(), allocation, arm, execution='dual', dual_profile=DUAL_ENV96_PROFILE)
               for arm in ('C03', 'T10')]
    for row in records:
        assert row['rollout_workers'] == 96 and row['environment_processes'] == 32
        assert row['global_batch']//row['rollout_workers'] == 4
        assert row['global_batch']//row['optimizer_minibatch']*row['ppo_epochs'] == 12
        assert row['memory_protection'] == 'disabled'
        assert all(row[k] is None for k in ('memory_high_gib','memory_max_gib','memory_swap_gib'))
        assert schedule(cases(), row) == schedule(cases(), dual())
    assert not cpus(records[0]['trainer_cpus']) & cpus(records[1]['trainer_cpus'])


@pytest.mark.parametrize('field,value', [('rollout_workers',128), ('memory_high_gib',164),
                                       ('memory_swap_gib',0), ('memory_protection','bounded')])
def test_env96_memory_policy_cannot_silently_revert(field, value):
    row = recipe(config(), resources(0,'GPU-unit-0','0-31,64-95'), 'C03',
                 execution='dual', dual_profile=DUAL_ENV96_PROFILE)
    row[field] = value
    with pytest.raises(ValueError):
        validate_recipe(row)


@pytest.mark.parametrize('field,value',[('trainer_cpus','0-31,64-95'),('cuda_allocator_mib',32768),
                                      ('input_cache_mib',4096),('rollout_workers',240),('optimizer_minibatch',128)])
def test_unbounded_parallel_resources_or_changed_learning_budget_are_rejected(field,value):
    r=dual();r[field]=value
    with pytest.raises(ValueError):validate_recipe(r)


def test_worker_must_stay_in_its_own_cpu_partition(monkeypatch):
    r=dual();monkeypatch.setenv('CUDA_VISIBLE_DEVICES',r['gpu_uuid'])
    monkeypatch.setattr(os,'sched_getaffinity',lambda _:cpus(r['trainer_cpus']))
    check_resources(r,cuda=False)
    monkeypatch.setattr(os,'sched_getaffinity',lambda _:cpus(r['cpus']))
    with pytest.raises(ValueError,match='partition'):check_resources(r,cuda=False)


def test_deferred_metrics_preserve_exact_reductions_rng_and_full_optimizer_update():
    torch.manual_seed(731);a,rows=temperature_toy()
    torch.manual_seed(731);b,other=temperature_toy()
    b.config['deferred_replay_metrics']=True
    with torch.no_grad():
        a.policy.ac.theta.add_(.007);b.policy.ac.theta.add_(.007)
    rng=torch.get_rng_state().clone()
    expected=a.replay_metrics(rows,microbatch=2)
    actual=b.replay_metrics(other,microbatch=2)
    assert actual==expected and actual['max_logp_error']>0
    assert torch.equal(rng,torch.get_rng_state())
    torch.manual_seed(731);a,rows=temperature_toy()
    torch.manual_seed(731);b,other=temperature_toy()
    b.config['deferred_replay_metrics']=True
    costs={r['case_id']:100. for r in rows}
    x=a.update_logical(rows,costs,logical_id='toy',shuffle_seed=7,diagnostic=True,minibatch=2,microbatch=2)
    y=b.update_logical(other,costs,logical_id='toy',shuffle_seed=7,diagnostic=True,minibatch=2,microbatch=2)
    assert x==y
    compare_states(a.policy.actor_optimizer.state_dict(),b.policy.actor_optimizer.state_dict(),exact=True)
    compare_states(a.policy.ac.state_dict(),b.policy.ac.state_dict(),exact=True)


def controller(tmp_path):
    x=DualController.__new__(DualController)
    x.m={'resources':{'gpu_uuid':'GPU-unit-0'},'workspace_root':str(tmp_path),'source_root':str(tmp_path),
         'budget':{'max_wall_seconds':60}}
    x.root=tmp_path;x.started=time.time();x.child=None;x.children={};x.poll_seconds=.01
    x.wait_gpu=lambda:None;x.status=lambda **kw:None;x.sample=lambda label:None
    return x


def test_peer_processes_really_overlap_instead_of_serial_gpu_wait(tmp_path):
    x=controller(tmp_path);entries=[]
    for arm in ('C03','T10'):
        out=tmp_path/arm;out.mkdir()
        script="import json,time,pathlib; a=time.time(); time.sleep(.3); pathlib.Path('timing.json').write_text(json.dumps([a,time.time()]))"
        # Distinct absolute result paths; both processes share controller cwd.
        script=script.replace("'timing.json'",repr(str(out/'timing.json')))
        entries.append((arm,[sys.executable,'-c',script],out))
    x.jobs(entries,'test')
    a,b=[json.loads((tmp_path/arm/'timing.json').read_text()) for arm in ('C03','T10')]
    assert max(a[0],b[0])<min(a[1],b[1]) and not x.children


def test_failed_peer_cancels_other_process_group(tmp_path):
    x=controller(tmp_path)
    entries=[('C03',[sys.executable,'-c','import time; time.sleep(.15); raise SystemExit(2)'],tmp_path/'a'),
             ('T10',[sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'b')]
    began=time.time()
    with pytest.raises(RuntimeError,match='failed'):x.jobs(entries,'failure')
    assert time.time()-began<5 and not x.children


def test_launcher_binds_native_helpers_before_python_initialization(tmp_path):
    cpu = min(os.sched_getaffinity(0))
    x = controller(tmp_path)
    x.m['python'] = sys.executable
    item = (tmp_path/'manifest.json', {'recipe': {'trainer_cpus': str(cpu)}})
    _, command, _ = x.entry('C03', item, 'train', tmp_path/'out')
    worker = Path(x.m['source_root'])/'onpolicy/scripts/train/stage3_h3_continuation_worker.py'
    worker.parent.mkdir(parents=True)
    worker.write_text('''import json, os, subprocess, sys, threading
rows = [sorted(os.sched_getaffinity(0))]
thread = threading.Thread(target=lambda: rows.append(sorted(os.sched_getaffinity(0))))
thread.start()
thread.join()
rows.append(json.loads(subprocess.check_output([sys.executable, '-c',
    'import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))'], text=True)))
print(json.dumps(rows))
''')
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [[cpu], [cpu], [cpu]]


def test_guard_repairs_existing_helper_threads_and_preserves_pinned_children(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip('Needs two CPUs to expose inherited helper-thread affinity')
    cpu = allowed[0]
    ready = tmp_path/'ready.json'
    script = '''import json, os, pathlib, subprocess, sys, threading, time
cpu = int(sys.argv[1])
helper = threading.Thread(target=lambda: time.sleep(30), daemon=True)
helper.start()
os.sched_setaffinity(0, {cpu})
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
pathlib.Path(sys.argv[2]).write_text(json.dumps({'child': child.pid, 'helper': helper.native_id}))
time.sleep(30)
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(cpu), str(ready)],
                               start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        row = json.loads(ready.read_text())
        assert len(os.sched_getaffinity(row['helper'])) > 1
        identities = {pid: process_identity(pid) for pid in (process.pid, row['child'])}
        result = enforce_tree(process.pid, identities, {cpu})
        assert {x['tid'] for x in result['changes']} == {row['helper']}
        assert set(result['processes']) == set(identities)
        for tid in (process.pid, row['child'], row['helper']):
            assert os.sched_getaffinity(tid) == {cpu}
        assert not enforce_tree(process.pid, identities, {cpu})['changes']
        stale = {pid: (value[0], value[1] - 1) for pid, value in identities.items()}
        assert enforce_tree(process.pid, stale, {cpu})['threads'] == 0
    finally:
        import signal
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
