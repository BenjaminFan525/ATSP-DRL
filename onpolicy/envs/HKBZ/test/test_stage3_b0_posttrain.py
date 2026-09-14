"""CPU-only contracts for the deferred all-CPU evaluation handoff."""
import copy
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[3] / 'scripts/train/run_stage3_b_shared_b0_posttrain.py'
SPEC = importlib.util.spec_from_file_location('posttrain_adapter', SOURCE)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


@pytest.fixture
def study(tmp_path):
    cases = [dict(path=f'/fixed/case_{i:04d}', content_sha256=f'case-hash-{i}') for i in range(24)]
    m = dict(root=str(tmp_path / 'run'), manifest_sha256='manifest', schedule_sha256='schedule',
             splits={'validation':cases}, recipe=dict(validation_workers=12,
                evaluation_batch_contract='b0_fixed12_keep_completed_slots_v1',
                evaluation_cudnn_allow_tf32=True, gpu_uuid='GPU0', controller_cpus='30-31,94-95',
                cpus='0-31,64-95'))
    root = Path(m['root'])
    root.mkdir()
    manifest = root / 'manifest.json'
    adapter.write(manifest, m)
    tests = tmp_path / 'tests.xml'
    tests.write_text('<testsuite tests="1" failures="0"/>')
    plan = dict(root=m['root'], manifest_sha256=m['manifest_sha256'],
        manifest=dict(path=str(manifest), sha256=adapter.checksum(manifest)),
        adapter=dict(path=str(SOURCE), sha256=adapter.checksum(SOURCE)),
        tests=dict(path=str(tests), sha256=adapter.checksum(tests), passed=1, failures=0),
        fixed_batch_size=12, gpu_uuid='GPU0', all_cpus='0-127',
        lane_cpus=['0-15,64-79','16-31,80-95','32-47,96-111','48-63,112-127'],
        authorization='训练结束后，将全部的CPU资源分配给验证器',
        training_budget=dict(training_episodes=3664, actor_updates=54, data_epochs=8,
            global_batches=27, epoch_end_batches=[13,15,17,19,21,23,25,27]),
        control=str(tmp_path / 'control'), original_service='hkbz-test.service',
        controller=dict(pid=99991, create_time=1), trainer=dict(pid=99992, create_time=2))
    request = dict(request_id='e003664_H_validation', request_sha256='request-hash',
                   split='validation', seed=1, tau=.3, history='b0_authoritative_native_v1', decoder='H')
    path = tmp_path / 'plan.json'
    adapter.write(path, plan)
    return m, plan, request, path


def row(m, request, index):
    case = m['splits'][request['split']][index]
    return dict(binding=adapter.binding(m, request), case_sha256=case['content_sha256'],
        result=dict(case_id=case['path'], completed=True, makespan=8000.0 + index,
                    seed=1, tau=.3, history=request['history'], decoder=request['decoder'],
                    behavior_deterministic=True, forced_replay=False))


def completed(study):
    m, plan, _, _ = study
    root = Path(m['root'])
    for batch in (19, 27):
        checkpoint, update = root / f'checkpoint_{batch}.pt', root / f'update_{batch}.json'
        checkpoint.write_bytes(b'test checkpoint')
        update.write_text('{}')
        commit = dict(next_batch=batch, training_episodes=3664 if batch==27 else 2128,
            actor_updates=batch*2, manifest_sha256=m['manifest_sha256'],
            schedule_sha256=m['schedule_sha256'], checkpoint=str(checkpoint),
            checkpoint_sha256=adapter.checksum(checkpoint), update=str(update),
            update_sha256=adapter.checksum(update))
        adapter.write(root / 'commits' / f'batch_{batch:04d}.json', commit)
        epoch = 4 if batch==19 else 8
        adapter.write(root / 'diagnostics' / f'epoch_{epoch}.json',
            dict(manifest_sha256=m['manifest_sha256'], checkpoint_sha256=commit['checkpoint_sha256'],
                 epoch=epoch, optimizer_steps=0))
    adapter.write(root / 'training_completed.json', dict(completed=True,
        manifest_sha256=m['manifest_sha256'], training_episodes=3664, actor_updates=54,
        data_epochs=8, last_commit=str(root / 'commits/batch_0027.json')))


def test_plan_binds_manifest_adapter_and_authorization(study):
    _, plan, _, path = study
    assert adapter.load_plan(path)[0] == plan
    Path(plan['tests']['path']).write_text('changed')
    with pytest.raises(ValueError, match='Changed post-training input'):
        adapter.load_plan(path)


def test_plan_rejects_changed_manifest(study):
    _, plan, _, path = study
    Path(plan['manifest']['path']).write_text('{}')
    with pytest.raises(ValueError, match='Changed post-training input'):
        adapter.load_plan(path)


def test_cpu_lanes_cover_all_cores_without_overlap(study):
    _, plan, _, _ = study
    adapter.validate_cpu_lanes(plan)
    lanes = [adapter.cpu_set(s) for s in plan['lane_cpus']]
    assert set().union(*lanes) == set(range(128))
    assert sum(map(len, lanes)) == 128


def test_cpu_lanes_reject_overlap(study):
    _, plan, _, _ = study
    plan['lane_cpus'][3] = plan['lane_cpus'][2]
    with pytest.raises(ValueError):
        adapter.validate_cpu_lanes(plan)


def test_process_binding_uses_kernel_ticks_and_rejects_pid_reuse():
    pid=os.getpid()
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    identity=dict(pid=pid,start_ticks=int(fields[19]),create_time=0)
    assert adapter.same_process(identity)
    assert not adapter.same_process({**identity,'start_ticks':identity['start_ticks']+1})
    assert not adapter.same_process(identity,required_tokens=('not-an-existing-argv-token',))


@pytest.mark.parametrize('start', [-12, 1, 13, 24])
def test_fixed_group_rejects_rebatching(study, start):
    m, _, request, _ = study
    with pytest.raises(ValueError, match='fixed12'):
        adapter.case_group(m, request, start)


def test_fixed_group_retains_original_case_order(study):
    m, _, request, _ = study
    assert adapter.case_group(m, request, 12) == m['splits']['validation'][12:24]


def test_partial_cache_is_not_a_complete_result(study):
    m, _, request, _ = study
    adapter.write(adapter.cache_path(m, request, 0), row(m, request, 0))
    assert not adapter.cached_group(m, request, 0)
    assert not adapter.publish_result(m, request)
    assert not (Path(m['root']) / 'validator/results').exists()


def test_groups_can_finish_out_of_order_but_result_is_canonical(study):
    m, _, request, _ = study
    adapter.publish_group(m, request, 12, [row(m, request, i) for i in range(12,24)])
    assert not adapter.publish_result(m, request)
    adapter.publish_group(m, request, 0, [row(m, request, i) for i in range(12)])
    assert adapter.publish_result(m, request)
    result = adapter.read(Path(m['root']) / 'validator/results' / (request['request_id']+'.json'))
    assert [r['case_id'] for r in result['rows']] == [c['path'] for c in m['splits']['validation']]
    assert result['request'] == request


def test_partial_cache_replay_must_match_before_any_publication(study):
    m, _, request, _ = study
    original = row(m, request, 11)
    adapter.write(adapter.cache_path(m, request, 11), original)
    rows = [row(m, request, i) for i in range(12)]
    rows[11]['result']['makespan'] += 1
    with pytest.raises(ValueError, match='previously completed'):
        adapter.publish_group(m, request, 0, rows)
    assert not adapter.cache_path(m, request, 0).exists()
    assert adapter.read(adapter.cache_path(m, request, 11)) == original


@pytest.mark.parametrize('bad', ['checkpoint', 'case', 'decoder', 'nonfinite'])
def test_cache_identity_is_checked(study, bad):
    m, _, request, _ = study
    r = row(m, request, 0)
    if bad=='checkpoint':r['binding']['request_sha256']='other'
    if bad=='case':r['case_sha256']='other'
    if bad=='decoder':r['result']['decoder']='AR'
    if bad=='nonfinite':r['result']['makespan']=float('nan')
    with pytest.raises(ValueError):
        adapter.validate_row(m, request, m['splits']['validation'][0], r)


def test_completion_requires_receipt_and_both_diagnostics(study):
    m, plan, _, _ = study
    assert adapter.completion_proof(plan, m) is None
    completed(study)
    assert adapter.completion_proof(plan, m)
    (Path(m['root']) / 'diagnostics/epoch_8.json').unlink()
    assert adapter.completion_proof(plan, m) is None


def test_completion_rejects_changed_checkpoint(study):
    m, plan, _, _ = study
    completed(study)
    (Path(m['root']) / 'checkpoint_27.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='incomplete or changed'):
        adapter.completion_proof(plan, m)


def test_completion_rejects_wrong_final_budget(study):
    m, plan, _, _ = study
    completed(study)
    path = Path(m['root']) / 'training_completed.json'
    value=adapter.read(path);value['actor_updates']=52;adapter.write(path,value)
    with pytest.raises(ValueError, match='full budget'):
        adapter.completion_proof(plan, m)


class EndWatch(Exception):
    pass


def test_watcher_never_stops_live_trainer_even_with_completion_receipt(study, monkeypatch):
    m, plan, _, path = study
    completed(study)
    commands=[]
    monkeypatch.setattr(adapter.os,'sched_setaffinity',lambda *_:None)
    monkeypatch.setattr(adapter,'same_process',lambda *a,**k:True)
    monkeypatch.setattr(adapter.subprocess,'run',lambda *a,**k:commands.append(a))
    monkeypatch.setattr(adapter.time,'sleep',lambda *_:(_ for _ in ()).throw(EndWatch()))
    with pytest.raises(EndWatch):
        adapter.watch_main(SimpleNamespace(plan=path,dry_run=False),plan,m)
    assert not commands
    assert adapter.read(Path(plan['control'])/'status.json')['status']=='waiting_training'


def test_watcher_stops_only_bound_service_after_completed_training(study, monkeypatch):
    m, plan, _, path = study
    completed(study)
    commands=[];controllers=[]
    monkeypatch.setattr(adapter.os,'sched_setaffinity',lambda *_:None)
    monkeypatch.setattr(adapter,'same_process',lambda identity,**k:identity==plan['controller'] and not commands)
    monkeypatch.setattr(adapter.subprocess,'run',lambda command,**k:commands.append(command))
    monkeypatch.setattr(adapter.subprocess,'check_output',lambda cmd,**k:
        str(plan['controller']['pid']) if '--property=MainPID' in cmd else 'inactive\n')
    monkeypatch.setattr(adapter,'controller_main',lambda *a:controllers.append(a))
    adapter.watch_main(SimpleNamespace(plan=path,dry_run=False),plan,m)
    assert commands==[['systemctl','--user','stop','--no-block','hkbz-test.service']]
    assert len(controllers)==1
    assert adapter.read(Path(plan['control'])/'handoff.json')['trainer_exited'] is True


def test_watcher_refuses_unexpected_controller_exit(study, monkeypatch):
    m, plan, _, path = study
    commands=[]
    monkeypatch.setattr(adapter.os,'sched_setaffinity',lambda *_:None)
    monkeypatch.setattr(adapter,'same_process',lambda *a,**k:False)
    monkeypatch.setattr(adapter.subprocess,'run',lambda *a,**k:commands.append(a))
    with pytest.raises(RuntimeError,match='Original controller exited'):
        adapter.watch_main(SimpleNamespace(plan=path,dry_run=False),plan,m)
    assert not commands


def test_watcher_refuses_to_stop_reassigned_service(study, monkeypatch):
    m, plan, _, path = study
    completed(study)
    commands=[]
    monkeypatch.setattr(adapter.os,'sched_setaffinity',lambda *_:None)
    monkeypatch.setattr(adapter,'same_process',lambda identity,**k:identity==plan['controller'])
    monkeypatch.setattr(adapter.subprocess,'run',lambda *a,**k:commands.append(a))
    monkeypatch.setattr(adapter.subprocess,'check_output',lambda *a,**k:'123456')
    with pytest.raises(RuntimeError,match='different controller'):
        adapter.watch_main(SimpleNamespace(plan=path,dry_run=False),plan,m)
    assert not commands


def test_completed_study_needs_no_service_stop(study, monkeypatch):
    m, plan, _, path = study
    adapter.write(Path(m['root'])/'result.json', {'completed':True})
    commands=[]
    monkeypatch.setattr(adapter.os,'sched_setaffinity',lambda *_:None)
    monkeypatch.setattr(adapter.subprocess,'run',lambda *a,**k:commands.append(a))
    adapter.watch_main(SimpleNamespace(plan=path,dry_run=False),plan,m)
    assert not commands


def test_posttrain_controller_preserves_scientific_manifest_and_never_trains(study, monkeypatch):
    m, plan, _, path = study
    completed(study)
    m.update(source_root=m['root'], python=sys.executable)
    original_m = copy.deepcopy(m)
    launched=[];finalized=[];admitted=[]

    class Heartbeat:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def update(self,**k):pass

    class Supervisor:
        def __init__(self,resources,attempt,hb):
            self.m,self.output,self.hb=resources,attempt,hb
            self.children=[];self.started=time.monotonic();self.gpu_peak=0
            self.thread=SimpleNamespace(start=lambda:None)
        def drain(self):
            assert self.m['recipe']['cpus']=='0-127'
            assert all(phase=='validator' for phase,_ in self.children)
        def close(self):pass

    def finish(scientific,supervisor,hb):
        assert scientific==original_m
        finalized.append(scientific)
        adapter.write(Path(m['root'])/'result.json',{'scientific_target_passed':False})

    contract=SimpleNamespace(PROTOCOL='test',verify_admission=lambda m:admitted.append(m),
        check_resources=lambda *a,**k:None,reconcile_epoch_requests=lambda *a:None)
    original=SimpleNamespace(Supervisor=Supervisor,finish=finish,gpu_snapshot=lambda r:{'used_mib':0})
    monkeypatch.setattr(adapter,'context',lambda *a:(contract,None,original,SimpleNamespace(Heartbeat=Heartbeat)))
    monkeypatch.setattr(adapter.os,'sched_setaffinity',lambda *_:None)
    monkeypatch.setattr(adapter,'same_process',lambda *a,**k:False)
    monkeypatch.setattr(adapter.subprocess,'Popen',lambda cmd,**k:launched.append(cmd) or SimpleNamespace(pid=88))
    adapter.controller_main(SimpleNamespace(plan=path),plan,m)
    assert len(finalized)==len(admitted)==len(launched)==1
    assert 'validator' in launched[0]
    assert 'train' not in launched[0]
    assert m==original_m
