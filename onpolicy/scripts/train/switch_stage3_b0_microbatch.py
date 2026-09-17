#!/usr/bin/env python3
"""Apply an authorized microbatch change after a durable global checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from onpolicy.utils.stage3_research import atomic_json, digest_file, read_json
from onpolicy.utils.stage3_b_shared_b0 import budget, read_commit, training_schedule, verify_manifest
from onpolicy.scripts.train.restart_stage3_b_shared_b0 import environment


def boundary_batch(manifest, current_batch, boundary):
    if boundary == 'checkpoint':
        return current_batch
    if boundary == 'data_epoch':
        return next(i for i in budget(manifest)['epoch_end_batches'] if i >= current_batch)
    raise ValueError('Unknown checkpoint boundary')

def process_identity(pid):
    directory = Path('/proc') / str(pid)
    fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), start_ticks=fields[19],
                argv=(directory / 'cmdline').read_text().rstrip('\0').split('\0'))

def require_process(expected):
    if process_identity(expected['pid']) != expected:
        raise ValueError('Parent process identity changed; no signal will be sent')

def ready_commit(manifest, target):
    """Checkpoint files alone are insufficient; require the final commit hashes."""
    directory = Path(manifest['root']) / 'commits'
    paths = sorted(directory.glob('batch_*.json'))
    if paths and int(paths[-1].stem.split('_')[-1]) > target:
        raise ValueError('Requested switch boundary has already been passed')
    path = directory / f'batch_{target:04d}.json'
    if not path.exists():
        return None
    if [p.name for p in paths] != [f'batch_{i:04d}.json' for i in range(1, target + 1)]:
        raise ValueError('The committed prefix is not contiguous')
    return read_commit(manifest, path)

def service_pid(unit):
    return int(subprocess.check_output(
        ['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'], text=True).strip() or 0)


def run_logged_command(control, name, argv, *, source=ROOT, gpu='', timeout=1800):
    # Keep step commands distinct from the supervisor's own launch receipt.
    atomic_json(control / f'step_{name}_command.json', dict(argv=list(map(str, argv)),
                cwd=str(source), visible_gpu=gpu), overwrite=False)
    with (control / f'{name}.log').open('w') as stream:
        subprocess.run(list(map(str, argv)), cwd=source,
            env=environment(source, gpu), stdout=stream, stderr=subprocess.STDOUT,
            timeout=timeout, check=True)

def run(request_path):
    request = read_json(request_path)
    control = request_path.parent
    manifest_path = Path(request['parent_manifest']['path'])
    if digest_file(manifest_path) != request['parent_manifest']['sha256']:
        raise ValueError('Scheduled parent manifest changed')
    for relative, checksum in request['code_files'].items():
        if digest_file(ROOT / relative) != checksum:
            raise ValueError(f'Scheduled source changed: {relative}')
    m = read_json(manifest_path)
    verify_manifest(m, inputs=True)
    root = Path(m['root'])
    target = request['after_batch']
    microbatch = request['microbatch']
    parent_microbatch = m['recipe']['microbatch']
    if (m['recipe']['global_batch'] != 240 or request['global_batch'] != 240
            or (parent_microbatch, microbatch) not in ((80, 120), (80, 240), (120, 240))
            or request['next_batch'] != target + 1):
        raise ValueError('Unsupported requested 240-environment microbatch continuation')
    plan = training_schedule(m)
    if not 1 <= target < len(plan):
        raise ValueError('The switch must leave a subsequent training batch')
    next_batch_visits = len(plan[target]['cases'])
    allocation = set(range(32)) | set(range(64, 96))
    if not set(os.sched_getaffinity(0)).issubset(allocation):
        raise ValueError('Switch supervisor escaped the authorized CPU half')
    lock = (root / '.scheduled_microbatch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='running', phase='waiting_checkpoint', pid=os.getpid(),
                 started_unix=time.time(), after_batch=target, next_batch=target + 1,
                 microbatch=microbatch, parent_microbatch=parent_microbatch,
                 global_batch=240, next_batch_visits=next_batch_visits, parent_root=str(root))

    def progress(phase, **values):
        state.update(phase=phase, heartbeat_unix=time.time(), **values)
        atomic_json(control / 'status.json', state)

    def command(name, argv, *, gpu='', timeout=1800):
        run_logged_command(control, name, argv, gpu=gpu, timeout=timeout)

    stopped = False
    launched = False
    try:
        last_health = 0
        deadline = time.monotonic() + 24 * 3600
        while True:
            commit = ready_commit(m, target)
            if commit is not None:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Checkpoint boundary was not reached within 24 hours')
            if time.monotonic() - last_health >= 10:
                require_process(request['parent_process'])
                if service_pid(request['parent_service']) != request['parent_process']['pid']:
                    raise ValueError('Parent service no longer owns the recorded controller')
                running = read_json(root / 'run_status.json')
                if (running['status'] != 'running'
                        or running['pid'] != request['parent_process']['pid']
                        or time.time() - running['heartbeat_unix'] > 120):
                    raise RuntimeError('Parent training is not healthy; leaving it untouched')
                progress('waiting_checkpoint', committed_batches=len(list((root / 'commits').glob('batch_*.json'))))
                last_health = time.monotonic()
            time.sleep(.5)
        require_process(request['parent_process'])
        if service_pid(request['parent_service']) != request['parent_process']['pid']:
            raise ValueError('Parent service identity changed before the boundary')
        atomic_json(control / 'boundary_checkpoint.json', commit, overwrite=False)
        progress('stopping_after_checkpoint', checkpoint=commit['checkpoint'],
                 checkpoint_sha256=commit['checkpoint_sha256'], training_episodes=commit['training_episodes'])
        # SIGTERM may discard the beginning of the next rollout, never the
        # complete two-PPO update referenced by the durable commit above.
        subprocess.run(['systemctl', '--user', 'stop', request['parent_service']], check=True, timeout=90)
        stopped = True
        if ready_commit(m, target) != commit:
            raise ValueError('The boundary checkpoint changed while stopping')
        gpu = m['recipe']['gpu_uuid']
        progress('waiting_gpu_release')
        deadline = time.monotonic() + 60
        while True:
            used = int(subprocess.check_output(['nvidia-smi', '--id=' + gpu,
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if used <= 1024:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('GPU0 is not free after the parent stopped')
            time.sleep(2)
        probe = Path(request['probe_output'])
        output = Path(request['output'])
        progress('capacity_probe')
        command('capacity_probe', [m['python'], '-B', '-u',
            ROOT / 'onpolicy/scripts/train/probe_stage3_b0_requested_capacity.py', manifest_path,
            '--output', probe, '--trajectories', request['trajectories'],
            '--environments', '240', '--microbatch', str(microbatch), '--environment-processes', '64',
            '--input-cache-mib', '4096', '--encoder-checkpoint', '--memory-limit-gib', '109'], gpu=gpu)
        progress('preparing_continuation')
        command('prepare', [m['python'], '-B', '-u',
            ROOT / 'onpolicy/scripts/train/restart_stage3_b_shared_b0.py', 'prepare', manifest_path,
            '--output', output, '--candidate', request['candidate'], '--pytest-python', request['pytest_python'],
            '--ignore-numerical-equivalence', '--capacity-window', probe / 'result.json',
            '--global-batch', '240', '--microbatch', str(microbatch), '--environment-processes', '64',
            '--input-cache-mib', '4096', '--encoder-checkpoint', '--user-instruction', request['user_instruction']])
        resumed = read_json(output / 'manifest.json')
        runtime = read_json(output / 'runtime.json')
        if (resumed['resume_amendment']['completed_batches'] != target
                or resumed['recipe']['microbatch'] != microbatch
                or resumed['recipe']['global_batch'] != 240
                or runtime['microbatch'] != microbatch or runtime['global_batch'] != 240
                or training_schedule(resumed) != plan):
            raise ValueError('Prepared continuation changed the requested checkpoint, parameters or visits')
        progress('launching_continuation')
        command('launch', [m['python'], '-B', '-u',
            Path(resumed['source_root']) / 'onpolicy/scripts/train/restart_stage3_b_shared_b0.py',
            'launch', output / 'manifest.json'])
        launched = True
        unit = (output / 'last_service_unit.txt').read_text().strip()
        progress('verifying_live_restore', service=unit, output=str(output))
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status_path = output / 'run_status.json'
            if status_path.exists():
                running = read_json(status_path)
                attempt = Path(running['attempt'])
                if (attempt / 'failure.json').exists() or running['status'] == 'failed':
                    raise RuntimeError('The new trainer failed during startup; inspect its attempt')
                proof_path = attempt / 'train/resume_verified.json'
                train_path = attempt / 'train/status.json'
                if proof_path.exists() and train_path.exists():
                    proof, train = read_json(proof_path), read_json(train_path)
                    if (proof.get('passed') and proof.get('loaded_training_state_exact')
                            and proof.get('cuda_rng_checked') and proof['next_batch'] == target
                            and train.get('environment_count') == next_batch_visits
                            and train.get('model_copies') == 1
                            and train.get('batch') == target + 1
                            and train.get('status') == 'running'
                            and service_pid(unit) == running['pid']):
                        receipt = dict(completed=True, verified_unix=time.time(), service=unit,
                            output=str(output), after_batch=target, next_batch=target + 1,
                            global_batch=240, microbatch=microbatch, next_batch_visits=next_batch_visits,
                            resume_verified=proof,
                            training_status=train, capacity_probe=str(probe / 'result.json'))
                        atomic_json(control / 'result.json', receipt, overwrite=False)
                        atomic_json(output / 'scheduled_switch_verified.json', receipt, overwrite=False)
                        progress('switched', status='completed')
                        return
            time.sleep(2)
        raise TimeoutError('The new trainer did not confirm CUDA checkpoint restoration in 180 seconds')
    except BaseException as error:
        progress('failed', status='failed', error=str(error))
        atomic_json(control / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(),
                    parent_stopped=stopped, continuation_launched=launched), overwrite=False)
        if stopped and not launched:
            # A failed capacity probe must not leave this long-running study
            # idle. Continue the parent's original recipe from the saved point.
            original = read_json(root / 'launch_command.json')['argv']
            unit = f'hkbz-b0-fallback{parent_microbatch}-{int(time.time())}'
            argv = ['--unit=' + unit if str(x).startswith('--unit=') else x for x in original]
            atomic_json(control / 'fallback_command.json', dict(argv=argv), overwrite=False)
            subprocess.run(argv, check=True, timeout=60)
            (root / 'last_service_unit.txt').write_text(unit + '.service\n')
            atomic_json(control / 'fallback.json', dict(service=unit + '.service',
                        microbatch=parent_microbatch, reason=str(error), checkpoint=commit['checkpoint']), overwrite=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    run(parser.parse_args().request.resolve())


if __name__ == '__main__':
    main()
