#!/usr/bin/env python3
"""Parallel preflight using the existing study's immutable evaluator and caches.

This adapter never trains, changes a twelve-case batch, or edits frozen source.
After both shards finish it resumes the original controller, whose original
baseline/zero validators still decide whether numerical admission may proceed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

SHARDS = ('0-14,64-78', '15-29,79-93')


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_id(task):
    return f"{task['kind']}_{task['split']}_{task['start']:04d}"


def tasks(m):
    width = m['recipe']['validation_workers']
    if width != 12:
        raise ValueError('Only the registered fixed twelve-case evaluator is supported')
    return [dict(kind=kind, split=split, start=start)
            for kind, splits in (('baseline', ('tune', 'validation', 'train')),
                                 ('zero', ('tune', 'validation')))
            for split in splits for start in range(0, len(m['splits'][split]), width)]


def cases_for(m, task):
    if task not in tasks(m):
        raise ValueError('Unregistered preflight task or batch boundary')
    return m['splits'][task['split']][task['start']:task['start'] + 12]


def destination(m, task, local_index):
    folder = 'baseline_cases' if task['kind'] == 'baseline' else 'zero_check_cases'
    return Path(m['root']) / folder / task['split'] / f"{task['start'] + local_index:04d}.json"


def validate_row(m, task, case, row):
    expected = dict(manifest_sha256=m['manifest_sha256'], source_sha256=m['source']['sha256'],
        decoder='H', native=task['kind'] == 'baseline',
        batch_contract=m['recipe']['evaluation_batch_contract'],
        evaluation_cudnn_allow_tf32=m['recipe']['evaluation_cudnn_allow_tf32'])
    r = row['result']
    if (row['binding'] != expected or row['case_sha256'] != case['content_sha256']
            or r['case_id'] != case['path'] or not r['completed'] or r.get('cycle_terminated')
            or not math.isfinite(r['makespan']) or r['steps'] <= 0
            or r['seed'] != 1 or r['tau'] != .3 or r['history'] != m['recipe']['history']
            or r['decoder'] != 'H' or r['policy_updates'] != 0
            or not r['behavior_deterministic'] or r['forced_replay']
            or len(r['actions_sha256']) != 64 or len(r['history_sha256']) != 64):
        raise ValueError('Preflight cache identity or trajectory contract differs')
    if task['kind'] == 'baseline' and task['split'] == 'train':
        moments = row.get('normalization_moments', [])
        if len(moments) != 3 or moments[0] <= 0 or not all(map(math.isfinite, moments)):
            raise ValueError('Training baseline is missing its normalization moments')


def cached_count(m, task):
    count = 0
    for i, c in enumerate(cases_for(m, task)):
        p = destination(m, task, i)
        if p.exists():
            validate_row(m, task, c, read(p))
            count += 1
    return count


def compare_zero_reference(m, task, rows, *, required=False):
    if task['kind'] != 'zero':
        return 0
    matched = 0
    for i, (c, row) in enumerate(zip(cases_for(m, task), rows)):
        baseline_task = {**task, 'kind': 'baseline'}
        p = destination(m, baseline_task, i)
        if not p.exists():
            if required:
                raise ValueError('Benchmark requires a complete native baseline reference')
            continue
        old = read(p)
        validate_row(m, baseline_task, c, old)
        for key in ('makespan', 'steps', 'actions_sha256', 'history_sha256'):
            if row['result'][key] != old['result'][key]:
                raise ValueError(f'Concurrent zero-update reference differs: {c["path"]}: {key}')
        matched += 1
    return matched


def context(manifest):
    m = read(manifest)
    source = Path(m['source_root']).resolve()
    os.environ['HKBZ_STAGE3_WORKSPACE_ROOT'] = m['workspace_root']
    sys.path.insert(0, str(source))
    from onpolicy.utils import stage3_b_shared_b0 as contract
    from onpolicy.scripts.train import stage3_b_shared_b0_worker as worker
    from onpolicy.scripts.train import run_stage3_b_shared_b0 as controller
    from onpolicy.utils import stage3_research as common
    for module in (contract, worker, controller, common):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError('Preflight must import the original immutable source')
    contract.verify_manifest(m, inputs=True)
    return m, contract, worker, controller, common


def publish(m, task, rows, common):
    group = cases_for(m, task)
    if len(rows) != len(group):
        raise ValueError('Cannot publish a partial task')
    for c, row in zip(group, rows):
        validate_row(m, task, c, row)
    compare_zero_reference(m, task, rows)
    for i, row in enumerate(rows):
        p = destination(m, task, i)
        if p.exists():
            if read(p) != row:
                raise ValueError('Refusing to replace an existing evaluation result')
        else:
            common.atomic_json(p, row, overwrite=False)


def worker_main(args):
    m, contract, worker, _, common = context(args.manifest)
    os.sched_setaffinity(0, contract.cpus(args.cpus))
    contract.check_resources(m)
    spec = read(args.spec)
    if (spec['manifest_sha256'] != m['manifest_sha256']
            or spec['adapter_sha256'] != checksum(__file__)
            or len({task_id(t) for t in spec['tasks']}) != len(spec['tasks'])):
        raise ValueError('Invalid immutable preflight assignment')
    for t in spec['tasks']:
        cases_for(m, t)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'status.json').exists():
        raise ValueError('Worker attempts must have fresh status directories')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    with worker.Progress(args.output / 'status.json', phase='parallel_preflight',
            cpus=sorted(os.sched_getaffinity(0)), manifest_sha256=m['manifest_sha256']) as hb:
        runner = worker.engine(m, training=False, width=12)
        receipts = []
        try:
            for task in spec['tasks']:
                started = time.monotonic()
                ident = task_id(task)
                group = cases_for(m, task)
                staging = args.output / 'cases' / ident
                for i, c in enumerate(group):
                    source = destination(m, task, i)
                    if source.exists():
                        row = read(source)
                        validate_row(m, task, c, row)
                        common.atomic_json(staging / f'{i:04d}.json', row, overwrite=False)
                initial = len(list(staging.glob('*.json')))
                hb.update(event='task_start', task=ident, completed_tasks=len(receipts), total_tasks=len(spec['tasks']))
                rows = worker.evaluate_cases(m, runner, group, staging, hb,
                    native=task['kind'] == 'baseline',
                    times=task['kind'] == 'baseline' and task['split'] == 'train')
                for c, row in zip(group, rows):
                    validate_row(m, task, c, row)
                matched = compare_zero_reference(m, task, rows, required=args.require_reference)
                if args.publish:
                    publish(m, task, rows, common)
                receipt = dict(task=task, seconds=time.monotonic() - started,
                    completed_cases=len(rows), queried_cases=0 if initial == len(group) else len(group),
                    existing_cases=initial, zero_reference_exact_cases=matched,
                    row_files={f'{i:04d}.json':checksum(staging/f'{i:04d}.json') for i in range(len(rows))})
                common.atomic_json(args.output / 'receipts' / f'{ident}.json', receipt, overwrite=False)
                receipts.append(receipt)
                hb.update(event='task_completed', task=ident, completed_tasks=len(receipts),
                          total_tasks=len(spec['tasks']), task_seconds=receipt['seconds'])
            common.atomic_json(args.output / 'result.json', dict(passed=True,
                manifest_sha256=m['manifest_sha256'], adapter_sha256=checksum(__file__),
                receipts=receipts, queried_cases=sum(r['queried_cases'] for r in receipts)), overwrite=False)
        finally:
            runner.close()


def run_main(args):
    import psutil
    import fcntl
    m, contract, _, controller, common = context(args.manifest)
    root = Path(m['root'])
    plan = read(args.plan)
    if (plan['manifest_sha256'] != m['manifest_sha256']
            or plan['adapter_sha256'] != checksum(__file__) or plan['shard_cpus'] != list(SHARDS)
            or plan['tests']['failures'] != 0 or plan['tests']['passed'] < 1):
        raise ValueError('Preflight plan/source/tests changed')
    if checksum(plan['tests']['path']) != plan['tests']['sha256']:
        raise ValueError('Preflight regression receipt changed')
    bench = read(plan['benchmark']['path'])
    if (checksum(plan['benchmark']['path']) != plan['benchmark']['sha256'] or not bench['passed']
            or bench['adapter_sha256'] != checksum(__file__) or bench['manifest_sha256'] != m['manifest_sha256']):
        raise ValueError('Concurrent numerical benchmark changed')
    if (root/'training_admission.json').exists() or list((root/'commits').glob('*.json')):
        raise ValueError('This adapter is restricted to pre-training evaluation')
    os.sched_setaffinity(0, contract.cpus(m['recipe']['controller_cpus']))
    contract.check_resources(m, 'controller', cuda=False)
    lock = (root/'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if controller.gpu_snapshot(m['recipe'])['used_mib'] > 1024:
        raise ValueError('GPU0 must be idle after the original controller stops')
    if psutil.virtual_memory().available < (m['recipe']['memory_max_gib']+32)*2**30:
        raise ValueError('Host memory does not cover the original cap and headroom')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'started.json').exists():
        raise ValueError('Use a fresh accelerator attempt')
    common.atomic_json(args.output/'started.json', dict(unix=time.time(), plan_sha256=checksum(args.plan)), overwrite=False)
    # The source controller has released its lock; only now import benchmark
    # results into its canonical cache. No evaluator races with publication.
    bench_dir = Path(plan['benchmark']['path']).parent
    for receipt in bench['receipts']:
        task = receipt['task']
        if receipt['zero_reference_exact_cases'] != len(cases_for(m, task)):
            raise ValueError('Benchmark did not compare every action/history/cost/step')
        staging = bench_dir/'cases'/task_id(task)
        for name, sha in receipt['row_files'].items():
            if checksum(staging/name) != sha:
                raise ValueError('Benchmark trajectory file changed')
        publish(m, task, [read(staging/f'{i:04d}.json') for i in range(len(cases_for(m, task)))], common)
    pending = [t for t in tasks(m) if cached_count(m, t) != len(cases_for(m, t))]
    common.atomic_json(args.output/'assignments.json', dict(tasks=pending, workers=2), overwrite=False)
    started = time.monotonic()
    children = []
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    passed = False
    with common.Heartbeat(root/'run_status.json', protocol=contract.PROTOCOL,
            phase='parallel_preflight', attempt=str(args.output), manifest_sha256=m['manifest_sha256'],
            resource='GPU0; CPUs 0-31,64-95; two fixed12 evaluators') as hb:
        try:
            for index, cpus in enumerate(SHARDS):
                output = args.output/f'worker{index}'
                output.mkdir()
                spec = args.output/f'assignment{index}.json'
                common.atomic_json(spec, dict(manifest_sha256=m['manifest_sha256'], adapter_sha256=checksum(__file__),
                    tasks=pending[index::2]), overwrite=False)
                command = [m['python'], '-B', '-u', str(Path(__file__).resolve()), 'worker',
                    str(args.manifest.resolve()), '--spec', str(spec.resolve()), '--cpus', cpus,
                    '--output', str(output.resolve()), '--publish']
                with (output/'terminal.log').open('x') as log:
                    p = subprocess.Popen(command, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                                         cwd=m['source_root'], start_new_session=True)
                children.append(p)
            while any(p.poll() is None for p in children):
                for p in children:
                    if p.poll() is not None and p.returncode != 0:
                        raise RuntimeError(f'Parallel evaluator {p.pid} failed: {p.returncode}')
                gpu = controller.gpu_snapshot(m['recipe'])
                if gpu['used_mib'] > gpu['total_mib']-m['recipe']['gpu_headroom_mib']:
                    raise RuntimeError('GPU0 no longer has its required 6 GiB headroom')
                family = [psutil.Process()]+psutil.Process().children(recursive=True)
                processes = []
                for p in family:
                    try:
                        affinity = p.cpu_affinity()
                        if not set(affinity) <= contract.cpus(m['recipe']['cpus']):
                            raise RuntimeError('Preflight process escaped the allowed CPU half')
                        processes.append(dict(pid=p.pid, cpus=affinity, rss=p.memory_info().rss))
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
                with (args.output/'resources.jsonl').open('a') as f:
                    f.write(json.dumps(dict(unix=time.time(), gpu=gpu, processes=processes))+'\n')
                done = sum(len(list((args.output/f'worker{i}'/'receipts').glob('*.json'))) for i in range(2))
                hb.update(completed_tasks=done, total_tasks=len(pending))
                if time.monotonic()-started > m['recipe']['evaluation_timeout_seconds']:
                    raise TimeoutError('Parallel preflight exceeded six hours')
                time.sleep(10)
            if any(p.returncode != 0 for p in children):
                raise RuntimeError('Parallel evaluator failed on exit')
            for t in tasks(m):
                if cached_count(m, t) != len(cases_for(m, t)):
                    raise ValueError('Original 960-case preflight coverage is incomplete')
            passed = True
            common.atomic_json(args.output/'completed.json', dict(passed=True, all_preflight_cases=960,
                manifest_sha256=m['manifest_sha256'], elapsed_seconds=time.monotonic()-started), overwrite=False)
            hb.update(phase='preflight_cached_resume_pending')
        except BaseException as e:
            common.atomic_json(args.output/'failure.json', dict(error=str(e), traceback=traceback.format_exc()))
            raise
        finally:
            for p in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            for p in children:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=15)
            common.atomic_json(args.output/'accounting.json', dict(elapsed_seconds=time.monotonic()-started,
                reserved_gpu_hours=(time.monotonic()-started)/3600, gpu_count=1,
                source_controller_stopped=True, passed=passed))
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    # Original code aggregates moments, checks historical Tune and compares all
    # 180 zero-update trajectories. Its global32 and fresh-process resume gates
    # remain mandatory. The resumed service gets the original resource recipe.
    launcher = Path(m['source_root'])/'onpolicy/scripts/train/launch_stage3_b_shared_b0_local.sh'
    result = subprocess.run(['bash', str(launcher), str(args.manifest.resolve()), 'resume'],
                            text=True, capture_output=True)
    common.atomic_json(args.output/'resume_receipt.json', dict(returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, unit=(root/'last_service_unit.txt').read_text().strip()))
    if result.returncode:
        raise RuntimeError('Original controller resume failed; see resume_receipt.json')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    worker = sub.add_parser('worker')
    worker.add_argument('manifest', type=Path)
    worker.add_argument('--spec', type=Path, required=True)
    worker.add_argument('--cpus', required=True)
    worker.add_argument('--output', type=Path, required=True)
    worker.add_argument('--publish', action='store_true')
    worker.add_argument('--require-reference', action='store_true')
    run = sub.add_parser('run')
    run.add_argument('manifest', type=Path)
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'worker':
        worker_main(args)
    else:
        run_main(args)


if __name__ == '__main__':
    main()
