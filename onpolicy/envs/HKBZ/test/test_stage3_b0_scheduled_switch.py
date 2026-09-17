import copy
from pathlib import Path
import sys

import pytest

from onpolicy.envs.HKBZ.test.test_stage3_b0_restart import continuation
from onpolicy.scripts.train.switch_stage3_b0_microbatch import boundary_batch, ready_commit, require_process
from onpolicy.utils.stage3_b_shared_b0 import training_schedule
from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json


def parent(tmp_path):
    m = continuation()
    m['resume_amendment'].update(completed_batch_sizes=[32]*9+[64], completed_batches=10,
                                completed_visits=352, completed_actor_updates=20)
    m['recipe'].update(global_batch=240, microbatch=80)
    m.update(root=str(tmp_path), manifest_sha256='parent')
    m['schedule_sha256'] = digest_json(training_schedule(m))
    return m


def commit_prefix(m, count):
    directory = Path(m['root']) / 'commits'
    directory.mkdir(exist_ok=True)
    for i, batch in enumerate(training_schedule(m)[:count], 1):
        checkpoint = Path(m['root']) / f'model_{i}.pt'
        update = Path(m['root']) / f'update_{i}.json'
        checkpoint.write_bytes(b'complete checkpoint')
        atomic_json(update, dict(complete_ppo_epochs=2))
        atomic_json(directory / f'batch_{i:04d}.json', dict(
            manifest_sha256=m['manifest_sha256'], schedule_sha256=m['schedule_sha256'],
            next_batch=i, training_episodes=batch['training_episodes'], actor_updates=2*i,
            checkpoint=str(checkpoint), checkpoint_sha256=digest_file(checkpoint),
            update=str(update), update_sha256=digest_file(update)))


def test_switch_distinguishes_checkpoint_from_data_epoch(tmp_path):
    m = parent(tmp_path)
    assert boundary_batch(m, 11, 'checkpoint') == 11
    assert boundary_batch(m, 11, 'data_epoch') == 13
    assert boundary_batch(m, 14, 'data_epoch') == 17


def test_switch_waits_for_commit_and_rejects_corrupt_payload(tmp_path):
    m = parent(tmp_path)
    commit_prefix(m, 10)
    (tmp_path / 'model_11.pt').write_bytes(b'checkpoint saved but no commit yet')
    assert ready_commit(m, 11) is None
    commit_prefix(m, 11)
    committed = ready_commit(m, 11)
    assert committed['training_episodes'] == 592 and committed['actor_updates'] == 22
    (tmp_path / 'model_11.pt').write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='incomplete or changed'):
        ready_commit(m, 11)


def test_switch_refuses_missing_or_already_passed_boundary(tmp_path):
    m = parent(tmp_path)
    commit_prefix(m, 12)
    with pytest.raises(ValueError, match='already been passed'):
        ready_commit(m, 11)
    (tmp_path / 'commits/batch_0007.json').unlink()
    with pytest.raises(ValueError, match='not contiguous'):
        ready_commit(m, 12)


def test_switch_rejects_reused_pid_before_signalling(monkeypatch):
    expected = dict(pid=123, start_ticks='100', argv=['python', 'the parent manifest'])
    import onpolicy.scripts.train.switch_stage3_b0_microbatch as switch
    monkeypatch.setattr(switch, 'process_identity', lambda _: dict(expected, start_ticks='101'))
    with pytest.raises(ValueError, match='identity changed'):
        require_process(expected)


@pytest.mark.parametrize('microbatch, completed', [(120, 11), (240, 12), (240, 13)])
def test_microbatch_switch_preserves_every_scheduled_visit(tmp_path, microbatch, completed):
    m = parent(tmp_path)
    before = training_schedule(m)
    after = copy.deepcopy(m)
    after['recipe']['microbatch'] = microbatch
    prefix = before[:completed]
    after['resume_amendment'].update(completed_batch_sizes=[len(x['cases']) for x in prefix],
        completed_batches=completed, completed_visits=prefix[-1]['training_episodes'],
        completed_actor_updates=2*completed)
    assert training_schedule(after) == before


def test_continuation_launch_preserves_supervisor_launch_receipt(tmp_path):
    from onpolicy.scripts.train.switch_stage3_b0_microbatch import run_logged_command
    receipt = tmp_path / 'launch_command.json'
    receipt.write_text('{"supervisor": true}\n')
    before = receipt.read_bytes()
    run_logged_command(tmp_path, 'launch', [sys.executable, '-c', 'print("child launched")'],
                       source=tmp_path)
    assert receipt.read_bytes() == before
    assert (tmp_path / 'step_launch_command.json').exists()
    assert (tmp_path / 'launch.log').read_text().strip() == 'child launched'
    with pytest.raises(FileExistsError):
        run_logged_command(tmp_path, 'launch', [sys.executable, '-c', 'raise SystemExit(0)'],
                           source=tmp_path)


@pytest.mark.parametrize('old_microbatch, new_microbatch, target, fail_capacity', [
    (80, 120, 11, False), (120, 240, 12, False), (120, 240, 12, True),
])
def test_scheduled_switch_runs_requested_parameters_or_restores_parent(
        tmp_path, monkeypatch, old_microbatch, new_microbatch, target, fail_capacity):
    import subprocess
    import onpolicy.scripts.train.switch_stage3_b0_microbatch as switch
    from onpolicy.utils.stage3_research import read_json

    original_root = tmp_path / 'parent'
    original_root.mkdir()
    m = parent(original_root)
    m['recipe'].update(microbatch=old_microbatch, gpu_uuid='gpu0')
    m.update(python=sys.executable, source_root=str(original_root))
    commit_prefix(m, target)
    manifest_path = original_root / 'manifest.json'
    atomic_json(manifest_path, m)
    atomic_json(original_root / 'launch_command.json', dict(argv=[
        'systemd-run', '--user', '--unit=parent', sys.executable, str(manifest_path)]))
    control = tmp_path / 'control'
    control.mkdir()
    launch_receipt = dict(supervisor=True)
    atomic_json(control / 'launch_command.json', launch_receipt)
    output = tmp_path / 'continuation'
    request = dict(parent_manifest=dict(path=str(manifest_path), sha256=digest_file(manifest_path)),
        code_files={}, after_batch=target, next_batch=target+1, global_batch=240,
        microbatch=new_microbatch, parent_process=dict(pid=123), parent_service='parent.service',
        output=str(output), probe_output=str(tmp_path / 'capacity'), trajectories='fixture',
        candidate='candidate', pytest_python=sys.executable, user_instruction='after this checkpoint')
    atomic_json(control / 'request.json', request)
    calls = []
    monkeypatch.setattr(switch, 'verify_manifest', lambda *a, **kw: None)
    monkeypatch.setattr(switch, 'require_process', lambda _: None)
    monkeypatch.setattr(switch, 'service_pid', lambda unit: 123 if unit == 'parent.service' else 456)
    monkeypatch.setattr(switch.os, 'sched_getaffinity', lambda _: {30, 31, 94, 95})

    def unexpected_sleep(_):
        raise AssertionError('Ready checkpoint or tail-batch restoration was not recognized')
    monkeypatch.setattr(switch.time, 'sleep', unexpected_sleep)

    def gpu_status(argv, **kwargs):
        assert argv[0] == 'nvidia-smi' and '--id=gpu0' in argv
        return '0\n'
    monkeypatch.setattr(switch.subprocess, 'check_output', gpu_status)

    def execute(argv, **kwargs):
        argv = list(map(str, argv))
        calls.append(argv)
        is_probe = any(x.endswith('/probe_stage3_b0_requested_capacity.py') for x in argv)
        if is_probe or 'prepare' in argv:
            assert argv[argv.index('--microbatch')+1] == str(new_microbatch)
            assert argv[argv.index('--environment-processes')+1] == '64'
        if is_probe and fail_capacity:
            raise subprocess.CalledProcessError(1, argv)
        if 'prepare' in argv:
            output.mkdir()
            resumed = copy.deepcopy(m)
            resumed.update(root=str(output), source_root=str(output / 'source'))
            resumed['recipe']['microbatch'] = new_microbatch
            prefix = training_schedule(m)[:target]
            resumed['resume_amendment'].update(completed_batches=target,
                completed_batch_sizes=[len(row['cases']) for row in prefix],
                completed_visits=prefix[-1]['training_episodes'], completed_actor_updates=2*target)
            atomic_json(output / 'manifest.json', resumed)
            atomic_json(output / 'runtime.json', dict(global_batch=240, microbatch=new_microbatch))
        if 'launch' in argv:
            (output / 'last_service_unit.txt').write_text('continued.service\n')
            attempt = output / 'attempts/active'
            (attempt / 'train').mkdir(parents=True)
            atomic_json(output / 'run_status.json', dict(attempt=str(attempt), status='running', pid=456))
            atomic_json(attempt / 'train/resume_verified.json', dict(passed=True,
                loaded_training_state_exact=True, cuda_rng_checked=True, next_batch=target))
            atomic_json(attempt / 'train/status.json', dict(status='running', batch=target+1,
                environment_count=len(training_schedule(m)[target]['cases']), model_copies=1))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(switch.subprocess, 'run', execute)

    if fail_capacity:
        with pytest.raises(subprocess.CalledProcessError):
            switch.run(control / 'request.json')
        fallback = read_json(control / 'fallback.json')
        assert fallback['microbatch'] == old_microbatch
        assert fallback['service'].startswith(f'hkbz-b0-fallback{old_microbatch}-')
        assert calls[-1][0] == 'systemd-run' and str(manifest_path) in calls[-1]
        assert not (control / 'result.json').exists()
        assert read_json(control / 'failure.json')['parent_stopped'] is True
    else:
        switch.run(control / 'request.json')
        result = read_json(control / 'result.json')
        assert result['completed'] and result['microbatch'] == new_microbatch
        assert result['next_batch_visits'] == (128 if target == 12 else 240)
        assert (control / 'step_launch_command.json').exists()
        assert read_json(control / 'status.json')['status'] == 'completed'
    assert read_json(control / 'launch_command.json') == launch_receipt
    assert calls[0] == ['systemctl', '--user', 'stop', 'parent.service']
    assert (control / 'boundary_checkpoint.json').exists()
