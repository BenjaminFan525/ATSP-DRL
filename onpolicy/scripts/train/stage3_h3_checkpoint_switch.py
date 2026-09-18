#!/usr/bin/env python3
"""Wait for a complete checkpoint, then replace the controller at that boundary."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from onpolicy.utils.stage3_h3_checkpoint_switch import verify_request, boundary_commit, prepare_at_boundary
from onpolicy.utils.stage3_research import atomic_json, read_json


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


def origin_alive(request):
    pid = request['controller_pid']
    if process_start(pid) != request['controller_start_ticks']:
        return False
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    return os.fsencode(request['origin']['path']) in command


def run(path):
    request = read_json(path)
    root = Path(request['root'])
    lock = (root/'switch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase, **values):
        atomic_json(root/'switch_status.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),
            request_sha256=request['request_sha256'],after_epoch=request['after_epoch'],**values))
    try:
        verify_request(request)
        # Validate the cgroup BEFORE the old controller can be signalled.
        from onpolicy.utils.stage3_memory_runtime import verify_unprotected_memory
        atomic_json(root/'switch_memory_runtime.json',verify_unprotected_memory(),overwrite=False)
        while True:
            if not origin_alive(request):
                raise RuntimeError('Original controller exited; no automatic restart will be attempted')
            if boundary_commit(request) is not None:
                break
            if time.time() > request['deadline_unix']:
                raise TimeoutError('Checkpoint switch wait expired; original experiment was left running')
            status('waiting_for_checkpoint',original_controller_running=True)
            time.sleep(5)
        status('preparing_checkpoint_migration')
        m = prepare_at_boundary(request)
        # Check again after copying, before terminating any process.
        commit = boundary_commit(request)
        if commit is None or not origin_alive(request):
            raise RuntimeError('Origin changed during checkpoint migration')
        status('stopping_old_controller',checkpoint=commit['checkpoint'])
        os.kill(request['controller_pid'],signal.SIGTERM)
        deadline = time.monotonic()+65
        while origin_alive(request):
            if time.monotonic() > deadline:
                raise TimeoutError('Original controller did not stop; new training was not started')
            time.sleep(1)
        status('starting_continuation',next_epoch=request['after_epoch']+1,
               restored_updates=commit['cumulative_ppo_steps'],
               minibatch=m['recipe']['optimizer_minibatch'],microbatch=m['recipe']['microbatch'])
        command = [m['python'],'-B','-u',str(Path(m['source_root'])/'onpolicy/scripts/train/run_stage3_h3_continuation.py'),
                   'run',str(root/'manifest.json')]
        atomic_json(root/'switch_applied.json',dict(applied_unix=time.time(),checkpoint=commit['checkpoint'],
            next_epoch=request['after_epoch']+1,restored_updates=commit['cumulative_ppo_steps'],
            manifest_sha256=m['manifest_sha256'],command=command),overwrite=False)
        os.execv(command[0],command)
    except BaseException as exc:
        status('failed',error=str(exc),original_controller_running=origin_alive(request))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path)
    run(parser.parse_args().request.resolve())
