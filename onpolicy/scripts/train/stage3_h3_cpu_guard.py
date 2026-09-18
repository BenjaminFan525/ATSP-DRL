#!/usr/bin/env python3
"""Repair all-thread affinity for an already-running, immutable dual-arm suite.

This external guard changes neither training files nor memory limits. New launchers
use taskset before Python starts; the guard also covers later epochs of older runs.
It enforces scheduler affinity, not an administrator-owned cgroup cpuset partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def cpu_set(spec):
    result = set()
    for part in spec.split(','):
        first, sep, last = part.partition('-')
        result.update(range(int(first), int(last) + 1) if sep else [int(first)])
    if not result:
        raise ValueError('Empty CPU partition')
    return result


def process_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[1]), int(fields[19])  # Parent PID and start ticks.
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(root, identities):
    found = {root}
    while True:
        children = {pid for pid, identity in identities.items() if identity[0] in found}
        if children <= found:
            return found
        found |= children


def enforce_tree(root, identities, allowed):
    """Preserve narrower worker pinning and include native helper threads."""
    changes = []
    checked_threads = 0
    checked_processes = []
    for pid in sorted(descendants(root, identities)):
        if process_identity(pid) != identities.get(pid):
            continue
        try:
            tasks = list((Path('/proc') / str(pid) / 'task').iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        checked_processes.append(pid)
        for task in tasks:
            tid = int(task.name)
            try:
                before = set(os.sched_getaffinity(tid))
                if not before <= allowed:
                    os.sched_setaffinity(tid, before & allowed or allowed)
                    changes.append(dict(pid=pid, tid=tid, before=sorted(before),
                                        after=sorted(os.sched_getaffinity(tid))))
                if not set(os.sched_getaffinity(tid)) <= allowed:
                    raise RuntimeError(f'Thread {tid} escaped its arm partition')
                checked_threads += 1
            except (FileNotFoundError, ProcessLookupError):
                continue
    return dict(root_pid=root, processes=checked_processes, threads=checked_threads,
                changes=changes, allowed_cpus=sorted(allowed))


def verified_json(path):
    value = json.loads(path.read_text())
    payload = {k: v for k, v in value.items() if k != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    if digest != value['manifest_sha256']:
        raise ValueError(f'Manifest identity changed: {path}')
    return value


def audit(suite, controller, expected_identity, groups):
    if process_identity(controller) != expected_identity:
        return None
    line = next(x for x in (Path('/proc') / str(controller) / 'cgroup').read_text().splitlines()
                if x.startswith('0::'))
    cgroup = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
    pids = {int(x) for file in cgroup.rglob('cgroup.procs') for x in file.read_text().split()}
    identities = {pid: identity for pid in pids if (identity := process_identity(pid)) is not None}
    worker = str(Path(suite['source_root']) / 'onpolicy/scripts/train/stage3_h3_continuation_worker.py')
    arms = {}
    for pid, identity in identities.items():
        if identity[0] != controller:
            continue
        try:
            args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if worker not in args:
            continue
        for arm, (manifest, allowed) in groups.items():
            if str(manifest) in args:
                if arm in arms:
                    raise RuntimeError(f'Multiple live trainers for {arm}')
                arms[arm] = enforce_tree(pid, identities, allowed)
    return dict(unix=time.time(), controller_pid=controller,
                manifest_sha256=suite['manifest_sha256'],
                mode='all_thread_scheduler_affinity', arms=arms,
                memory_limits_changed=False, cgroup_cpuset=False)


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < .5:
        parser.error('Polling interval must be at least half a second')
    root = args.suite.resolve()
    suite = verified_json(root / 'manifest.json')
    if suite['execution_mode'] != 'dual':
        raise ValueError('Expected a registered dual-arm suite')
    controller_args = (Path('/proc') / str(args.controller_pid) / 'cmdline').read_bytes().decode().split('\0')
    if str(root / 'manifest.json') not in controller_args:
        raise ValueError('Controller PID belongs to another suite')
    expected = process_identity(args.controller_pid)
    if expected is None:
        raise ValueError('Controller is not running')
    groups = {}
    for arm in ('C03', 'T10'):
        path = root / 'arms' / arm / 'manifest.json'
        manifest = verified_json(path)
        if manifest['suite']['path'] != str(root / 'manifest.json'):
            raise ValueError('Arm belongs to another suite')
        groups[arm] = (path, cpu_set(manifest['recipe']['trainer_cpus']))
    left, right = (groups[a][1] for a in ('C03', 'T10'))
    if left & right or left | right != cpu_set(suite['resources']['cpus']):
        raise ValueError('Arms must partition exactly the authorized CPU half')
    for allowed in (left, right):
        for cpu in allowed:
            siblings = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
            if not cpu_set(siblings.read_text().strip()) <= allowed:
                raise ValueError('Arms must not share physical cores through SMT')
    stop = False

    def finish(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, finish)
    signal.signal(signal.SIGINT, finish)
    last_write = 0.
    total_changes = 0
    while not stop:
        sample = audit(suite, args.controller_pid, expected, groups)
        if sample is None:
            break
        changed = sum(len(row['changes']) for row in sample['arms'].values())
        total_changes += changed
        sample['guard_pid'] = os.getpid()
        sample['total_thread_corrections'] = total_changes
        sample['status'] = 'running'
        if changed:
            with (root / 'cpu_isolation_events.jsonl').open('a') as file:
                file.write(json.dumps(sample) + '\n')
        if changed or time.monotonic() - last_write >= 30 or args.once:
            publish(root / 'cpu_isolation_status.json', sample)
            last_write = time.monotonic()
        if args.once:
            print(json.dumps(sample))
            return
        time.sleep(args.interval)
    publish(root / 'cpu_isolation_status.json', dict(status='stopped', unix=time.time(),
            controller_pid=args.controller_pid, total_thread_corrections=total_changes))


if __name__ == '__main__':
    main()
