#!/usr/bin/env python3
"""Hand a completed, frozen B_SHARED run to four fixed12 CPU evaluation lanes.

The existing training service is stopped only after its final commit, both
gradient diagnostics, completion receipt, and trainer exit have been checked.
Scientific inputs and frozen source stay unchanged; the CPU amendment is a
separate, hash-bound execution plan. No training entry point is called here.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cpu_set(spec):
    result = set()
    for part in spec.split(','):
        bounds = [int(x) for x in part.split('-')]
        if len(bounds) == 1:
            result.add(bounds[0])
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            result.update(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError('Invalid CPU specification')
    return result


def validate_cpu_lanes(plan):
    lanes = [cpu_set(s) for s in plan['lane_cpus']]
    if (len(lanes) != 4 or any(len(s) != 32 for s in lanes)
            or set().union(*lanes) != cpu_set(plan['all_cpus'])
            or sum(map(len, lanes)) != len(set().union(*lanes))
            or cpu_set(plan['all_cpus']) != set(range(128))):
        raise ValueError('Four disjoint lanes must cover all 128 logical CPUs')
    if any({c ^ 64 for c in s} != s for s in lanes):
        raise ValueError('Each lane must retain complete SMT sibling pairs')


def load_plan(path):
    plan = read(path)
    for key in ('manifest', 'adapter', 'tests'):
        if checksum(plan[key]['path']) != plan[key]['sha256']:
            raise ValueError(f'Changed post-training input: {key}')
    if Path(plan['adapter']['path']).resolve() != Path(__file__).resolve():
        raise ValueError('Execute the frozen post-training adapter')
    m = read(plan['manifest']['path'])
    if (plan['manifest_sha256'] != m['manifest_sha256']
            or plan['root'] != m['root'] or plan['tests']['failures'] != 0
            or plan['tests']['passed'] < 1 or plan['fixed_batch_size'] != 12
            or m['recipe']['validation_workers'] != 12
            or plan['gpu_uuid'] != m['recipe']['gpu_uuid']
            or plan['authorization'] != '训练结束后，将全部的CPU资源分配给验证器'):
        raise ValueError('Post-training plan does not match the authorized run')
    validate_cpu_lanes(plan)
    return plan, m


def same_process(identity, *, required_tokens=()):
    import psutil
    try:
        process = psutil.Process(identity['pid'])
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        if 'start_ticks' in identity:
            # Kernel start ticks survive wall-clock/NTP adjustments.
            fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')', 1)[1].split()
            if int(fields[19]) != identity['start_ticks']:
                return False
        elif abs(process.create_time() - identity['create_time']) > .01:
            return False
        argv = process.cmdline()
        return all(token in argv for token in required_tokens)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, FileNotFoundError):
        return False


def completion_proof(plan, m):
    """Return None while incomplete; reject inconsistent completion evidence."""
    root = Path(m['root'])
    path = root / 'training_completed.json'
    if not path.exists():
        return None
    receipt = read(path)
    expected = plan['training_budget']
    if (receipt.get('completed') is not True
            or receipt.get('manifest_sha256') != m['manifest_sha256']
            or any(receipt.get(k) != expected[k] for k in
                   ('training_episodes', 'actor_updates', 'data_epochs'))):
        raise ValueError('Training completion receipt does not match the full budget')
    last = root / 'commits' / f"batch_{expected['global_batches']:04d}.json"
    if Path(receipt['last_commit']).resolve() != last.resolve():
        raise ValueError('Training receipt points to a different final commit')
    commit = read(last)
    if (commit['next_batch'] != expected['global_batches']
            or commit['training_episodes'] != expected['training_episodes']
            or commit['actor_updates'] != expected['actor_updates']
            or commit['manifest_sha256'] != m['manifest_sha256']
            or commit['schedule_sha256'] != m['schedule_sha256']
            or checksum(commit['checkpoint']) != commit['checkpoint_sha256']
            or checksum(commit['update']) != commit['update_sha256']):
        raise ValueError('Final checkpoint/update commit is incomplete or changed')
    diagnostics = {}
    for epoch in (4, 8):
        diagnostic = root / 'diagnostics' / f'epoch_{epoch}.json'
        if not diagnostic.exists():
            return None
        d = read(diagnostic)
        boundary = expected['epoch_end_batches'][epoch - 1]
        c = read(root / 'commits' / f'batch_{boundary:04d}.json')
        if (d['manifest_sha256'] != m['manifest_sha256']
                or d['checkpoint_sha256'] != c['checkpoint_sha256']
                or d['epoch'] != epoch or d['optimizer_steps'] != 0):
            raise ValueError('Gradient diagnostic is bound to a different checkpoint')
        diagnostics[str(epoch)] = checksum(diagnostic)
    return dict(receipt_sha256=checksum(path), final_commit_sha256=checksum(last),
                checkpoint_sha256=commit['checkpoint_sha256'], diagnostics=diagnostics)


def context(plan, m):
    source = Path(m['source_root']).resolve()
    os.environ['HKBZ_STAGE3_WORKSPACE_ROOT'] = m['workspace_root']
    sys.path.insert(0, str(source))
    from onpolicy.utils import stage3_b_shared_b0 as contract
    from onpolicy.scripts.train import stage3_b_shared_b0_worker as worker
    from onpolicy.scripts.train import run_stage3_b_shared_b0 as controller
    from onpolicy.utils import stage3_research as common
    for module in (contract, worker, controller, common):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError('Post-training evaluation must import frozen run source')
    contract.verify_manifest(m, inputs=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['gpu_uuid']:
        raise ValueError('Only the registered physical GPU0 may be visible')
    return contract, worker, controller, common


def case_group(m, request, start):
    cases = m['splits'][request['split']]
    if not isinstance(start, int) or start < 0 or start >= len(cases) or start % 12:
        raise ValueError('Task must start at an original fixed12 boundary')
    return cases[start:start + 12]


def binding(m, request):
    return dict(request_sha256=request['request_sha256'],
                batch_contract=m['recipe']['evaluation_batch_contract'],
                evaluation_cudnn_allow_tf32=m['recipe']['evaluation_cudnn_allow_tf32'])


def validate_row(m, request, case, row):
    r = row['result']
    if (row['binding'] != binding(m, request)
            or row['case_sha256'] != case['content_sha256']
            or r['case_id'] != case['path'] or not r['completed']
            or not math.isfinite(r['makespan']) or r['makespan'] <= 0
            or r['seed'] != request['seed'] or r['tau'] != request['tau']
            or r['history'] != request['history'] or r['decoder'] != request['decoder']
            or not r['behavior_deterministic'] or r['forced_replay']):
        raise ValueError('Evaluation cache changed case, checkpoint, or decoder identity')


def cache_path(m, request, index):
    return Path(m['root']) / 'validator/cases' / request['request_id'] / f'{index:04d}.json'


def cached_group(m, request, start):
    count = 0
    for i, case in enumerate(case_group(m, request, start), start):
        path = cache_path(m, request, i)
        if path.exists():
            validate_row(m, request, case, read(path))
            count += 1
    return count == len(case_group(m, request, start))


def publish_group(m, request, start, rows):
    cases = case_group(m, request, start)
    if len(rows) != len(cases):
        raise ValueError('Cannot publish a partial fixed12 task')
    # Validate the whole group, including old partial results, before any write.
    for i, (case, row) in enumerate(zip(cases, rows), start):
        validate_row(m, request, case, row)
        path = cache_path(m, request, i)
        if path.exists() and read(path) != row:
            raise ValueError('Refusing to replace a previously completed case')
    for i, row in enumerate(rows, start):
        path = cache_path(m, request, i)
        if not path.exists():
            write(path, row, overwrite=False)


def publish_result(m, request):
    rows = []
    for i, case in enumerate(m['splits'][request['split']]):
        path = cache_path(m, request, i)
        if not path.exists():
            return False
        row = read(path)
        validate_row(m, request, case, row)
        rows.append(row['result'])
    path = Path(m['root']) / 'validator/results' / (request['request_id'] + '.json')
    if not path.exists():
        write(path, dict(request=request, rows=rows, completed=True, finished_unix=time.time()),
              overwrite=False)
    return True


def signal_handler(*_):
    raise KeyboardInterrupt('Service stop')


def stop_children(children):
    for process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    for process in children:
        try:
            process.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def lane_main(args, plan, m):
    contract, worker, _, common = context(plan, m)
    assigned = cpu_set(plan['lane_cpus'][args.lane])
    os.sched_setaffinity(0, assigned)
    if set(os.sched_getaffinity(0)) != assigned:
        raise ValueError('Service cpuset did not admit the full assigned CPU lane')
    # Keep the original CUDA/recipe checks, with the separately authorized CPUs.
    resources = copy.deepcopy(m)
    resources['recipe']['cpus'] = plan['all_cpus']
    contract.check_resources(resources)
    from onpolicy.runner.shared.stage3_research_engine import seed_all
    seed_all(m['recipe']['seed'])
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, signal_handler)
    with worker.Progress(args.output / 'status.json', phase='posttrain_validator_lane',
                         lane=args.lane, cpus=sorted(os.sched_getaffinity(0)),
                         manifest_sha256=m['manifest_sha256']) as hb:
        runner = worker.engine(m, training=False, width=12)
        loaded, sequence = None, 0
        try:
            write(args.output / 'ready.json', dict(pid=os.getpid(), cpus=sorted(os.sched_getaffinity(0)),
                                                 fixed_batch_size=12), overwrite=False)
            while not (Path(m['root']) / 'validator/stop.json').exists():
                job_path = args.output / 'jobs' / f'{sequence:06d}.json'
                if not job_path.exists():
                    time.sleep(.25)
                    continue
                job = read(job_path)
                request, start = job['request'], job['start']
                if job['plan_sha256'] != checksum(args.plan):
                    raise ValueError('Worker assignment changed execution plan')
                contract.validate_request(m, request)
                cases = case_group(m, request, start)
                if loaded != request['checkpoint_sha256']:
                    runner.load_weights(request['checkpoint'], manifest_sha256=m['manifest_sha256'])
                    loaded = request['checkpoint_sha256']
                staging = args.output / 'cases' / f'{sequence:06d}'
                for local, (index, case) in enumerate(zip(range(start, start + len(cases)), cases)):
                    old = cache_path(m, request, index)
                    if old.exists():
                        row = read(old)
                        validate_row(m, request, case, row)
                        write(staging / f'{local:04d}.json', row, overwrite=False)
                hb.update(event='fixed12_task', request_id=request['request_id'], start=start)
                began = time.monotonic()
                rows = worker.evaluate_cases(m, runner, cases, staging, hb,
                    decoder=request['decoder'], binding={'request_sha256':request['request_sha256']},
                    timeout=m['recipe']['evaluation_timeout_seconds'])
                publish_group(m, request, start, rows)
                common.atomic_json(args.output / 'done' / f'{sequence:06d}.json',
                    dict(request_id=request['request_id'], start=start, completed_cases=len(rows),
                         seconds=time.monotonic()-began), overwrite=False)
                sequence += 1
        finally:
            runner.close()


def validator_main(args, plan, m):
    contract, worker, _, _ = context(plan, m)
    os.sched_setaffinity(0, cpu_set(m['recipe']['controller_cpus']))
    signal.signal(signal.SIGTERM, signal_handler)
    args.output.mkdir(parents=True, exist_ok=True)
    lanes, busy, validated = [], set(), {}
    done_groups = 0
    try:
        for lane in range(len(plan['lane_cpus'])):
            output = args.output / f'lane_{lane}'
            output.mkdir()
            command = [m['python'], '-B', '-u', str(Path(__file__).resolve()), 'lane',
                       '--plan', str(args.plan), '--output', str(output), '--lane', str(lane)]
            write(output / 'command.json', dict(argv=command), overwrite=False)
            with (output / 'terminal.log').open('x') as log:
                process = subprocess.Popen(command, cwd=m['source_root'], env=os.environ.copy(),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            lanes.append(dict(process=process, output=output, sequence=0, task=None))
        with worker.Progress(args.output / 'status.json', phase='posttrain_validator',
                             manifest_sha256=m['manifest_sha256'], lanes=4,
                             cpu_allocation=plan['all_cpus'], fixed_batch_size=12) as hb:
            while not (Path(m['root']) / 'validator/stop.json').exists():
                for lane in lanes:
                    if lane['process'].poll() is not None:
                        if (Path(m['root']) / 'validator/stop.json').exists():
                            return
                        raise RuntimeError(f"Validation lane exited: {lane['output']}")
                    marker = lane['output'] / 'done' / f"{lane['sequence']:06d}.json"
                    if lane['task'] and marker.exists():
                        receipt = read(marker)
                        if (receipt['request_id'], receipt['start']) != lane['task']:
                            raise ValueError('Lane receipt does not match assigned group')
                        busy.remove(lane['task'])
                        lane['task'] = None
                        lane['sequence'] += 1
                        done_groups += 1
                tasks = []
                pending = sorted(contract.pending_evaluations(m['root']), key=lambda p:p.stat().st_mtime_ns)
                for path in pending:
                    request = read(path)
                    if path.name != request['request_id'] + '.json':
                        raise ValueError('Request filename/identity mismatch')
                    if validated.get(path.name) != request:
                        contract.validate_request(m, request)
                        validated[path.name] = request
                    if publish_result(m, request):
                        continue
                    for start in range(0, len(m['splits'][request['split']]), 12):
                        key = (request['request_id'], start)
                        if key not in busy and not cached_group(m, request, start):
                            tasks.append((key, request))
                for lane in lanes:
                    if lane['task'] is not None or not tasks:
                        continue
                    key, request = tasks.pop(0)
                    job = dict(plan_sha256=checksum(args.plan), request=request, start=key[1])
                    write(lane['output'] / 'jobs' / f"{lane['sequence']:06d}.json", job, overwrite=False)
                    lane['task'] = key
                    busy.add(key)
                hb.update(event='parallel_evaluation' if busy else 'validator_idle',
                          active_groups=len(busy), completed_groups=done_groups,
                          pending_requests=len(contract.pending_evaluations(m['root'])))
                time.sleep(2)
    finally:
        stop_children([lane['process'] for lane in lanes])


def controller_main(args, plan, m):
    contract, _, original, common = context(plan, m)
    contract.verify_admission(m)
    proof = completion_proof(plan, m)
    if not proof or same_process(plan['trainer']):
        raise ValueError('Post-training controller cannot run before trainer exit')
    root = Path(m['root'])
    if (root / 'result.json').exists():
        return
    if (root / 'validator/stop.json').exists():
        raise ValueError('Validator is terminally stopped without a final result')
    os.sched_setaffinity(0, cpu_set(m['recipe']['controller_cpus']))
    contract.check_resources(m, 'controller', cuda=False)
    gpu = original.gpu_snapshot(m['recipe'])
    release_deadline = time.monotonic() + 30
    while gpu['used_mib'] > 1024:
        if time.monotonic() > release_deadline:
            raise ValueError('GPU0 has not been released by the original workers')
        time.sleep(2)
        gpu = original.gpu_snapshot(m['recipe'])
    lock = (root / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = root / 'attempts' / f'{time.strftime("%Y%m%dT%H%M%S")}_{os.getpid()}_posttrain_cpu128'
    attempt.mkdir(parents=True)
    resource_m = copy.deepcopy(m)
    resource_m['recipe']['cpus'] = plan['all_cpus']

    class Supervisor(original.Supervisor):
        def launch(self, phase, **options):
            if phase != 'validator' or options:
                raise ValueError('Post-training controller may launch only the validator')
            output = self.output / phase
            output.mkdir()
            command = [m['python'], '-B', '-u', str(Path(__file__).resolve()), 'validator',
                       '--plan', str(args.plan), '--output', str(output)]
            write(output / 'command.json', dict(argv=command, plan_sha256=checksum(args.plan)), overwrite=False)
            with (output / 'terminal.log').open('x') as log:
                process = subprocess.Popen(command, cwd=m['source_root'], env=os.environ.copy(),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append((phase, process))
            return process

    signal.signal(signal.SIGTERM, signal_handler)
    write(attempt / 'resource_admission.json', dict(plan=plan, plan_sha256=checksum(args.plan),
        gpu=gpu, completion_proof=proof, cpu_allocation=plan['all_cpus'],
        frozen_manifest_unchanged=True, additional_training_updates=0), overwrite=False)
    with common.Heartbeat(root / 'run_status.json', protocol=contract.PROTOCOL, attempt=str(attempt),
            manifest_sha256=m['manifest_sha256'], resource='GPU0; CPUs 0-127; four fixed12 validators',
            posttrain_plan=str(args.plan)) as hb:
        supervisor = Supervisor(resource_m, attempt, hb)
        supervisor.thread.start()
        try:
            contract.reconcile_epoch_requests(m, plan['training_budget']['global_batches'])
            supervisor.launch('validator')
            supervisor.drain()
            hb.update(phase='candidate_selection')
            original.finish(m, supervisor, hb)
            hb.update(phase='completed', training_episodes=plan['training_budget']['training_episodes'],
                actor_updates=plan['training_budget']['actor_updates'],
                scientific_target_passed=read(root / 'result.json')['scientific_target_passed'])
        except BaseException as exc:
            write(attempt / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(),
                  manifest_sha256=m['manifest_sha256']))
            raise
        finally:
            supervisor.close()
            write(attempt / 'accounting.json', dict(elapsed_seconds=time.monotonic()-supervisor.started,
                peak_gpu0_mib=supervisor.gpu_peak, cpu_allocation=plan['all_cpus'],
                plan_sha256=checksum(args.plan), additional_training_updates=0))
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def watch_main(args, plan, m):
    control = Path(plan['control'])
    control.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        proof = completion_proof(plan, m)
        print(json.dumps(dict(plan_valid=True, training_complete=bool(proof),
            trainer_alive=same_process(plan['trainer']),
            action='wait_for_training' if not proof or same_process(plan['trainer']) else 'ready_for_handoff')))
        return
    lock = (control / 'watcher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, cpu_set(m['recipe']['controller_cpus']))
    signal.signal(signal.SIGTERM, signal_handler)
    root = Path(m['root'])
    try:
        while True:
            if (root / 'result.json').exists():
                write(control / 'status.json', dict(status='already_completed', pid=os.getpid()))
                return
            # Revalidate the exact authorized adapter/manifest before any action.
            plan, m = load_plan(args.plan)
            if not same_process(plan['controller'], required_tokens=(plan['manifest']['path'],)):
                raise RuntimeError('Original controller exited before the scheduled handoff')
            proof = completion_proof(plan, m)
            trainer_alive = same_process(plan['trainer'])
            write(control / 'status.json', dict(status='waiting_training', pid=os.getpid(),
                checked_unix=time.time(), training_complete=bool(proof), trainer_alive=trainer_alive,
                pending_cpu_allocation=plan['all_cpus'], active_cpu_allocation=m['recipe']['cpus']))
            if proof and not trainer_alive:
                break
            time.sleep(5)
        # Training and its final gradient diagnostic have finished. Stop the old
        # evaluation controller gracefully, retaining every committed case.
        write(control / 'handoff.json', dict(started_unix=time.time(), proof=proof,
            original_controller=plan['controller'], trainer_exited=True,
            reason=plan['authorization']), overwrite=False)
        service_pid = int(subprocess.check_output(['systemctl', '--user', 'show',
            plan['original_service'], '--property=MainPID', '--value'], text=True).strip())
        if service_pid != plan['controller']['pid']:
            raise RuntimeError('Original service now belongs to a different controller')
        subprocess.run(['systemctl', '--user', 'stop', '--no-block', plan['original_service']], check=True)
        deadline = time.monotonic() + 75
        while same_process(plan['controller']):
            if time.monotonic() > deadline:
                raise TimeoutError('Original controller did not exit after its completed training')
            time.sleep(1)
        # Wait for all processes in the old service, including CUDA env workers.
        while True:
            active = subprocess.check_output(['systemctl', '--user', 'show', plan['original_service'],
                '--property=ActiveState', '--value'], text=True).strip()
            if active in ('inactive', 'failed'):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Old service still owns evaluation workers')
            time.sleep(1)
        write(control / 'status.json', dict(status='posttrain_evaluation', pid=os.getpid(),
            checked_unix=time.time(), cpu_allocation=plan['all_cpus'], lanes=4,
            original_service_stopped=True, trainer_exited=True))
        controller_main(args, plan, m)
        write(control / 'status.json', dict(status='completed', pid=os.getpid(), finished_unix=time.time(),
                                           result=str(root / 'result.json')))
    except BaseException as exc:
        write(control / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(), unix=time.time()))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('watch', 'controller', 'validator', 'lane'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lane', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    plan, m = load_plan(args.plan)
    if args.mode in ('validator', 'lane') and args.output is None:
        parser.error('Worker modes require --output')
    if args.mode == 'lane' and args.lane not in range(4):
        parser.error('Lane must be in 0..3')
    if args.mode in ('controller', 'validator', 'lane'):
        if not completion_proof(plan, m) or same_process(plan['trainer']):
            raise ValueError('All-CPU evaluation is restricted to completed training')
    {'watch':watch_main, 'controller':controller_main,
     'validator':validator_main, 'lane':lane_main}[args.mode](args, plan, m)


if __name__ == '__main__':
    main()
