"""Exactly-once handoff of an explicitly expedited Stage2 manifest job."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time

from onpolicy.utils.shared_eval import parse_cpu_set


def marker_path(manifest, key):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', key):
        raise ValueError('Invalid manifest command key.')
    return Path(manifest).resolve().parent / 'external_jobs' / f'{key}.json'


def acquire_job_lock(manifest, key):
    """Keep this descriptor open, including across exec, for the job lifetime."""
    path = marker_path(manifest, key).with_suffix('.lock')
    path.parent.mkdir(exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def process_identity(pid):
    """A PID alone is insufficient: also bind to boot and /proc start ticks."""
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def read_marker(manifest, key):
    state = json.loads(marker_path(manifest, key).read_text())
    expected_sha = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if (state.get('manifest') != str(Path(manifest).resolve())
            or state.get('manifest_sha256') != expected_sha
            or state.get('command_key') != key):
        raise RuntimeError('External job is not bound to this frozen manifest command.')
    return state


def follow_external_job(args, write_json):
    """The original queue waits for the external result; it never relaunches it."""
    last_log = 0.0
    while True:
        state = read_marker(args.manifest, args.command_key)
        if state.get('monitor_cpu_set'):
            os.sched_setaffinity(0, parse_cpu_set(state['monitor_cpu_set']))
        if state.get('status') in {'completed', 'failed'}:
            code = int(state['exit_code'])
            if (state['status'] == 'completed') != (code == 0):
                raise RuntimeError('Inconsistent external completion status.')
            write_json(args.record, {
                'status': 'adopted_completed' if code == 0 else 'adopted_failed',
                'command_key': args.command_key, 'manifest': state['manifest'],
                'external_job': str(marker_path(args.manifest, args.command_key)),
                'trainer_pid': state.get('trainer_pid'), 'exit_code': code,
                'adopter_pid': os.getpid(), 'updated_unix_time': time.time(),
            })
            print(f'[ExternalJob] adopted terminal result: exit_code={code}', flush=True)
            return code
        owner = state.get('supervisor_identity')
        if not owner or process_identity(owner['pid']) != owner:
            raise RuntimeError('External supervisor disappeared; automatic retry is forbidden.')
        if time.time() >= state['deadline_unix_time']:
            raise TimeoutError('External job reached the original suite deadline.')
        if time.monotonic() - last_log >= 30:
            last_log = time.monotonic()
            write_json(args.record, {
                'status': 'adopting_external_job', 'command_key': args.command_key,
                'manifest': state['manifest'], 'trainer_pid': state.get('trainer_pid'),
                'external_job': str(marker_path(args.manifest, args.command_key)),
                'adopter_pid': os.getpid(), 'updated_unix_time': time.time(),
            })
            print(f'[ExternalJob] waiting for existing trainer={state.get("trainer_pid")}; '
                  'duplicate launch suppressed.', flush=True)
        time.sleep(2)


def guard_launch(args, write_json):
    path = marker_path(args.manifest, args.command_key)
    owner_pid = getattr(args, 'external_owner_pid', None)
    if owner_pid is not None:
        state = read_marker(args.manifest, args.command_key)
        identity = state.get('supervisor_identity')
        if (owner_pid != os.getppid() or identity != process_identity(owner_pid)
                or state.get('status') != 'launching'
                or state.get('cpu_set') != args.cpu_set):
            raise RuntimeError('External execution must be a child of the claiming supervisor.')
        return None
    try:
        fd = acquire_job_lock(args.manifest, args.command_key)
    except BlockingIOError:
        # A claiming supervisor publishes its marker immediately after locking.
        for _ in range(100):
            if path.exists():
                return follow_external_job(args, write_json)
            time.sleep(0.1)
        raise RuntimeError('Manifest job already claimed; refusing a duplicate launch.')
    if path.exists():
        try:
            return follow_external_job(args, write_json)
        finally:
            os.close(fd)
    # No external handoff: retain the inheritable lock through the normal exec.
    return None
