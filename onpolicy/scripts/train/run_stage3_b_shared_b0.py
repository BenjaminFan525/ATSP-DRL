#!/usr/bin/env python3
"""Prepare, supervise, explicitly resume, and report the local B_SHARED study."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.utils.stage3_b_shared_b0 import (
    PROTOCOL, SOURCE_SHA, FROZEN, WORKSPACE, SPLITS, COUNTS, recipe, cpus, physical_hash,
    manifest_identity, verify_manifest, check_resources, choose_candidate, risk_pass,
    submit_evaluation, pending_evaluations, validate_request, read_commit, schedule,
    reconcile_epoch_requests, verify_admission, budget,
)
from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json, case_record, paired_summary, Heartbeat


def gpu_snapshot(r):
    text = subprocess.check_output(['nvidia-smi', '--id=' + r['gpu_uuid'],
        '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    index, uuid, total, used, utilization = [s.strip() for s in text.strip().split(',')]
    if int(index) != 0 or uuid != r['gpu_uuid']:
        raise ValueError('The registered GPU0 identity changed')
    return dict(index=0, uuid=uuid, total_mib=int(total), used_mib=int(used), utilization=int(utilization))


def check_junit(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
    failures = sum(int(s.get('failures', 0)) + int(s.get('errors', 0)) for s in suites)
    tests = sum(int(s.get('tests', 0)) for s in suites)
    skipped = sum(int(s.get('skipped', 0)) for s in suites)
    passed_names = {c.get('name') for c in root.iter('testcase')
                    if not any(c.find(k) is not None for k in ('failure', 'error', 'skipped'))}
    required = {'test_strict_b0_native_shared_initialization_and_optimizer_ownership',
                'test_checkpoint_restores_nonempty_adam_norm_rng_and_rejects_partial',
                'test_resume_reconciles_auxiliary_ar_after_epoch_commit',
                'test_replay_applies_native_per_agent_terminal_reset',
                'test_native_evaluation_retains_finished_slots_and_original_steps',
                'test_interrupted_evaluation_replays_original_fixed_groups'}
    if failures or tests <= skipped or not required.issubset(passed_names):
        raise ValueError('A nonempty passing regression report is required')
    return dict(path=str(Path(path).resolve()), sha256=digest_file(path), tests=tests,
                passed=tests-skipped, skipped=skipped, failures=failures)


def prepare(args):
    import yaml
    from onpolicy.utils.stage2_frozen import FrozenStage2Bundle
    from onpolicy.envs.HKBZ.data_generator import (
        AirportScenarioGenerator, PROFILES, _allocate_profile_schedule, _derived_seed, _write_case,
    )
    r, output = recipe(), args.output.resolve()
    if output.exists():
        raise FileExistsError('New study output must not exist; use resume for an existing manifest')
    tests = check_junit(args.tests)
    bundle = FrozenStage2Bundle(FROZEN)
    bundle.validate()
    if bundle.files['b0']['sha256'] != SOURCE_SHA:
        raise ValueError('Unexpected Stage2 initialization checkpoint')
    output.mkdir(parents=True)
    splits = {name: [case_record(p) for p in sorted((WORKSPACE / 'onpolicy/envs/HKBZ/dataset' / relative).glob('case_*'))]
              for name, relative in SPLITS.items()}
    for name, records in splits.items():
        if Counter(c['distribution'] for c in records) != COUNTS[name]:
            raise ValueError(f'Dataset coverage differs: {name}')
    # Published B0 scores are bound to its original fixed twelve-case groups.
    # Canonical train/validation groups are fixed here; Tune preserves the
    # handed-off order instead of silently sorting or regrouping close scores.
    tune = {c['name']: c for c in splits['tune']}
    native_order = [c['case_dir'] for c in read_json(bundle.path('hf_baseline'))['cases']]
    if set(native_order) != set(tune) or len(native_order) != 60:
        raise ValueError('Frozen Tune60 order/coverage differs')
    splits['tune'] = [tune[name] for name in native_order]
    physical = {name: {physical_hash(c) for c in records} for name, records in splits.items()}
    if any(len(physical[s]) != len(splits[s]) for s in splits):
        raise ValueError('Duplicate physical cases within a split')
    if any(physical[a] & physical[b] for a, b in (('train', 'validation'), ('train', 'tune'), ('validation', 'tune'))):
        raise ValueError('Train/validation/Tune physical overlap')
    known = set()
    # Read fingerprints only, never historical holdout outcomes.
    names = subprocess.check_output(['rg', '--files', str(WORKSPACE / 'onpolicy/envs/HKBZ/dataset'),
                                     '-g', 'metadata.json'], text=True).splitlines()
    for path in names:
        value = read_json(path).get('fingerprints', {}).get('case_sha256')
        if value:
            known.add(value)
    confirmation = []
    physical_known = set().union(*physical.values())
    for i, profile in enumerate(_allocate_profile_schedule('validation', 120, r['confirmation_seed'])):
        name = f'case_{i:04d}'
        seed = _derived_seed(r['confirmation_seed'], r['confirmation_namespace'], i, profile)
        generator = AirportScenarioGenerator(profile=PROFILES[profile], seed=seed, split='confirmation', case_id=name)
        case, metadata = generator.generate()
        path = output / 'confirmation_data' / name
        _write_case(path, case, metadata)
        record = case_record(path)
        ph = physical_hash(record)
        if record['case_sha256'] in known or ph in physical_known:
            raise ValueError('Fresh confirmation overlaps a known case')
        known.add(record['case_sha256'])
        physical_known.add(ph)
        confirmation.append(record)
    if Counter(c['profile'] for c in confirmation) != Counter(c['profile'] for c in splits['validation']):
        raise ValueError('Confirmation profile coverage differs from validation120')
    splits['confirmation'] = confirmation
    # Four fixed profiles plus boundaries selected using metadata, never costs.
    canaries = []
    for profile in ('balanced', 'low_load_ood', 'resource_ood', 'stress_joint'):
        canaries.append(min((c for c in splits['train'] if c['profile'] == profile), key=lambda c: c['content_sha256']))
    remaining = [c for c in splits['train'] if c not in canaries]
    for key in ('num_planes', 'num_mobile_resources', 'arrival_horizon', 'mean_eligible_resource_instances'):
        selected = max(remaining, key=lambda c: (read_json(Path(c['path']) / 'metadata.json')['statistics'].get(key, 0), c['content_sha256']))
        canaries.append(selected)
        remaining.remove(selected)
    files = {}
    names = subprocess.check_output(['rg', '--files', 'onpolicy', '-g', '*.py', '-g', '*.yaml', '-g', '*.json', '-g', '*.sh',
                                     '-g', '!**/dataset/**', '-g', '!**/results/**', '-g', '!**/__pycache__/**'], cwd=WORKSPACE, text=True).splitlines()
    for relative in sorted(names):
        source, target = WORKSPACE / relative, output / 'source' / relative
        checksum = digest_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest_file(target) != checksum:
            raise RuntimeError('Workspace changed while creating the source snapshot')
        target.chmod(0o444)
        files[relative] = checksum
    shutil.copyfile(args.tests, output / 'tests.xml')
    tests.update(path=str(output / 'tests.xml'), sha256=digest_file(output / 'tests.xml'))
    plan = schedule(splits['train'], r['seed'])
    m = dict(protocol=PROTOCOL, material_passport={'mode': 'implementation -> run',
        'user_authorized': '2026-09-12; B0, GPU0 and remaining CPU half', 'scientific_result': 'unverified'},
        root=str(output), workspace_root=str(WORKSPACE), source_root=str(output / 'source'),
        source={'path':str(bundle.path('b0')), 'sha256':SOURCE_SHA},
        frozen_manifest={'path':str(FROZEN), 'sha256':digest_file(FROZEN)},
        recipe=r, splits=splits, canary_cases=canaries, code_files=files, code_sha256=digest_json(files),
        schedule_sha256=digest_json(plan), packages={name:importlib.metadata.version(name)
            for name in ('torch', 'torch-geometric', 'numpy', 'scipy', 'PyYAML', 'psutil')},
        python=sys.executable, tests=tests, created_unix=time.time(),
        exposure={'confirmation_costs_opened':False, 'known_case_fingerprints_checked':len(known) - 120,
                  'historical_holdout_outcomes_opened':False},
        resource_amendment='one physical GPU0, one policy rank, 32 physical cores; microbatch8 accumulates global32')
    m['manifest_sha256'] = manifest_identity(m)
    for relative in ('commits', 'validator/requests', 'validator/results', 'validator/cases', 'attempts', 'diagnostics'):
        (output / relative).mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'manifest.json', m, overwrite=False)
    atomic_json(output / 'preparation.json', {'prepared':True, 'training_started':False,
        'manifest_sha256':m['manifest_sha256'], 'source_sha256':SOURCE_SHA, 'case_counts':{k:len(v) for k,v in splits.items()},
        'code_files':len(files), 'resource_contract':r}, overwrite=False)
    print(json.dumps({'manifest':str(output / 'manifest.json'), 'source_files':len(files), 'training_started':False}), flush=True)


class Supervisor:
    def __init__(self, m, attempt, hb):
        self.m, self.root, self.output, self.hb = m, Path(m['root']), attempt, hb
        self.children = []
        self.started = time.monotonic()
        self.gpu_peak = 0
        self.resource_error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def monitor(self):
        import psutil
        own = psutil.Process()
        path = self.output / 'resources.jsonl'
        try:
            while not self.stop.is_set():
                gpu = gpu_snapshot(self.m['recipe'])
                self.gpu_peak = max(self.gpu_peak, gpu['used_mib'])
                family = [own] + own.children(recursive=True)
                processes = []
                for p in family:
                    try:
                        affinity = p.cpu_affinity()
                        if not set(affinity).issubset(cpus(self.m['recipe']['cpus'])):
                            raise RuntimeError(f'Study process {p.pid} escaped CPU half')
                        processes.append({'pid':p.pid,'rss':p.memory_info().rss,'cpus':affinity})
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
                row = dict(unix=time.time(), gpu=gpu, rss_sum_bytes=sum(p['rss'] for p in processes),
                           available_memory_bytes=psutil.virtual_memory().available, processes=processes)
                with path.open('a') as f:
                    f.write(json.dumps(row) + '\n')
                if gpu['used_mib'] > gpu['total_mib'] - self.m['recipe']['gpu_headroom_mib']:
                    raise RuntimeError('Whole GPU0 usage exceeded the admitted 6 GiB headroom')
                self.stop.wait(10)
        except BaseException as e:
            self.resource_error = str(e)

    def launch(self, phase, **options):
        output = self.output / phase
        output.mkdir(parents=True)
        command = [self.m['python'], '-B', '-u', str(Path(self.m['source_root']) /
            'onpolicy/scripts/train/stage3_b_shared_b0_worker.py'), str(self.root / 'manifest.json'),
            '--phase', phase, '--output', str(output)]
        for name, value in options.items():
            command.extend(['--' + name.replace('_', '-'), str(value)])
        atomic_json(output / 'command.json', {'argv':command, 'cwd':self.m['source_root']}, overwrite=False)
        log = (output / 'terminal.log').open('x')
        process = subprocess.Popen(command, cwd=self.m['source_root'], env=os.environ.copy(),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.children.append((phase, process))
        return process

    def health(self):
        if self.resource_error:
            raise RuntimeError(self.resource_error)
        if time.monotonic() - self.started > self.m['recipe']['wall_timeout_seconds']:
            raise TimeoutError('Study hard timeout reached')
        for phase, p in self.children:
            if p.poll() is not None and p.returncode != 0:
                raise RuntimeError(f'{phase} worker failed with exit {p.returncode}; see {self.output/phase/"terminal.log"}')

    def run_phase(self, phase, **options):
        self.hb.update(phase=phase, phase_started_unix=time.time())
        p = self.launch(phase, **options)
        while p.poll() is None:
            self.health()
            time.sleep(5)
        self.health()

    def drain(self):
        while pending_evaluations(self.root):
            self.health()
            self.hb.update(phase='validation_drain', pending_evaluations=len(pending_evaluations(self.root)))
            time.sleep(5)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)
        if (self.root / 'validator/stop.json').exists():
            for phase, process in self.children:
                if phase == 'validator':
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
        for _, process in self.children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process in self.children:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def evaluation(m, ident):
    data = read_json(Path(m['root']) / 'validator/results' / f'{ident}.json')
    request, rows = data['request'], data['rows']
    validate_request(m, request)
    cases = m['splits'][request['split']]
    if not data['completed'] or [r['case_id'] for r in rows] != [c['path'] for c in cases]:
        raise ValueError('Incomplete or reordered evaluation result')
    return rows


def finish(m, supervisor, hb):
    from onpolicy.utils.stage3_b_shared_b0 import epoch_episodes
    root = Path(m['root'])
    limits=budget(m)
    baseline = read_json(root / 'baselines.json')
    curves = []
    for epoch in range(1, 9):
        c = read_commit(m, root / 'commits' / f'batch_{limits["epoch_end_batches"][epoch-1]:04d}.json')
        episodes = epoch_episodes(m, epoch)
        curves.append(dict(epoch=epoch, episodes=episodes, checkpoint=c['checkpoint'], checkpoint_sha256=c['checkpoint_sha256'],
            **{split:paired_summary(evaluation(m, f'e{episodes:06d}_H_{split}'), baseline['costs'])
               for split in ('validation', 'tune')}))
    choice = {**choose_candidate(curves), 'manifest_sha256':m['manifest_sha256'], 'curves_sha256':digest_json(curves)}
    lock_path = root / 'selection_locked.json'
    if lock_path.exists():
        if read_json(lock_path) != choice:
            raise ValueError('Locked selection differs from complete epoch results')
    else:
        atomic_json(lock_path, choice, overwrite=False)
    selected = choice['selected']
    submit_evaluation(m, selected['checkpoint'], selected['episodes'], 'validation', decoder='AR')
    result = dict(protocol=PROTOCOL, manifest_sha256=m['manifest_sha256'], curves=curves, selection=choice,
                  scientific_target_passed=False, ci_scope='case uncertainty conditional on one trained seed')
    if choice['open_confirmation']:
        hb.update(phase='confirmation', selected_epoch=selected['epoch'])
        submit_evaluation(m, m['source']['path'], 0, 'confirmation', baseline=True)
        submit_evaluation(m, selected['checkpoint'], selected['episodes'], 'confirmation')
        supervisor.drain()
        control = evaluation(m, 'B0_H_confirmation')
        candidate = evaluation(m, f'e{selected["episodes"]:06d}_H_confirmation')
        summary = paired_summary(candidate, {r['case_id']:r['makespan'] for r in control})
        result['confirmation'] = summary
        result['scientific_target_passed'] = bool(choice['risk_qualified'] and choice['last_two_epochs_stable']
            and selected['validation']['gain_fraction'] >= .02 and summary['gain_fraction'] >= .02
            and summary['paired_case_bootstrap_gain_ci95'][0] > 0 and risk_pass(summary))
    else:
        result['confirmation'] = {'opened':False, 'reason':'no risk-qualified validation >=2% candidate'}
    supervisor.drain()
    ar_control = {r['case_id']:r['makespan'] for r in evaluation(m, 'B0_AR_validation')}
    result['autoregressive_diagnostic'] = {str(ep):paired_summary(evaluation(m, f'e{epoch_episodes(m, ep):06d}_AR_validation'), ar_control)
                                          for ep in sorted({4, 8, selected['epoch']})}
    updates = [read_json(read_commit(m, p)['update']) for p in sorted((root / 'commits').glob('batch_*.json'))]
    result['budget'] = dict(training_episodes=limits['training_episodes'], actor_updates=limits['actor_updates'], global_batches=limits['global_batches'],
        training_batch_seconds=sum(r['seconds'] for r in updates), gpu_count=1,
        source_baseline_episodes=780, zero_update_episodes=180,
        epoch_H_evaluation_episodes=1440, AR_evaluation_episodes=120*(1+len(result['autoregressive_diagnostic'])),
        confirmation_episodes=240 if choice['open_confirmation'] else 0,
        canary_completed_episodes=36*len(list((root/'attempts').glob('*/canary/result.json')))
                                  +4*len(list((root/'attempts').glob('*/resume-check/result.json'))),
        epoch_gradient_diagnostic_episodes=8,
        gpu_hour_scope='one reserved GPU; shared-validator time is not counted twice; attempts have separate resource logs')
    if m.get('resume_amendment'):
        result['budget'].update(continuation=m['resume_amendment'],
            canary_completed_episodes=0, source_baseline_fresh_queries=0,
            zero_update_fresh_queries=0, numerical_equivalence_gate='waived_by_user',
            inherited_batches=m['resume_amendment']['completed_batches'])
        if m.get('training_subset'):
            result['budget'].update(training_subset=m['training_subset'], training_cases=len(m['splits']['train']),
                inherited_training_episodes=m['training_subset']['inherited_visits'],
                subset_training_episodes=m['training_subset']['new_training_visits'])
    elif 'throughput_amendment' in m:
        selected=read_json(m['throughput_amendment']['selection'])
        result['budget'].update(canary_completed_episodes=8,source_baseline_fresh_queries=0,
            zero_update_fresh_queries=0,capacity_complete_diagnostic_visits=sum(
                trial['parameters']['environments'] for trial in selected['trials']),
            capacity_accounting=str(Path(m['throughput_amendment']['selection']).parent),
            capacity_visits_excluded_from_training_budget=True)
    atomic_json(root / 'result.json', result, overwrite=False)
    lines = ['# B0 → B_SHARED 本机实验结果', '', f"科学门通过：{result['scientific_target_passed']}。单训练 seed 的案例证据。", '',
        '| Epoch | Validation 秒 | 相对 B0 收益 | Tune 秒 | 相对 B0 收益 |', '| --- | ---: | ---: | ---: | ---: |']
    for row in curves:
        lines.append(f"| {row['epoch']} | {row['validation']['makespan']:.4f} | {row['validation']['gain_fraction']:.3%} | {row['tune']['makespan']:.4f} | {row['tune']['gain_fraction']:.3%} |")
    lines += ['', f"选择 epoch {selected['epoch']}；最后两个 epoch 稳定性：{choice['last_two_epochs_stable']}。", '',
        '完整逐案例结果、确认集置信区间、AR 诊断、预算及恢复记录见 result.json、validator/、commits/ 和 attempts/。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    atomic_json(root / 'validator/stop.json', {'completed':True})


def run(args, *, resume=False):
    import psutil
    m = read_json(args.manifest)
    root = Path(m['root'])
    if ROOT.resolve() != Path(m['source_root']).resolve():
        raise ValueError('Run must execute from its frozen source copy')
    verify_manifest(m, inputs=True)
    os.sched_setaffinity(0, cpus(m['recipe']['controller_cpus']))
    check_resources(m, 'controller', cuda=False)
    if (root / 'result.json').exists():
        raise ValueError('Study already completed')
    if not resume and (root / 'run_status.json').exists():
        raise ValueError('Existing study requires explicit resume')
    if (root / 'validator/stop.json').exists():
        raise ValueError('Validator was terminally stopped')
    gpu = gpu_snapshot(m['recipe'])
    if gpu['used_mib'] > 1024:
        raise ValueError('GPU0 has another workload; no preemption or oversubscription')
    if psutil.virtual_memory().available < (m['recipe']['memory_max_gib'] + 32) * 2**30:
        raise ValueError('Available host memory does not cover the declared cap plus 32 GiB headroom')
    lock = (root / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = root / 'attempts' / f'{time.strftime("%Y%m%dT%H%M%S")}_{os.getpid()}'
    attempt.mkdir()
    atomic_json(attempt / 'resource_admission.json', dict(gpu=gpu, cpu_allocation=m['recipe']['cpus'],
        available_memory_bytes=psutil.virtual_memory().available, resume=resume), overwrite=False)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    with Heartbeat(root / 'run_status.json', protocol=PROTOCOL, attempt=str(attempt),
                   manifest_sha256=m['manifest_sha256'], resource='GPU0; CPUs 0-31,64-95') as hb:
        s = Supervisor(m, attempt, hb)
        s.thread.start()
        try:
            if not (root / 'baselines.json').exists():
                s.run_phase('baseline')
            if not (root / 'zero_update_verified.json').exists():
                s.run_phase('zero')
            validator = s.launch('validator')
            submit_evaluation(m, m['source']['path'], 0, 'validation', decoder='AR', baseline=True)
            if not (root / 'training_admission.json').exists():
                s.run_phase('canary')
                s.run_phase('resume-check', canary_dir=attempt / 'canary')
                proof = read_json(attempt / 'canary/result.json')
                restored = read_json(attempt / 'resume-check/result.json')
                if not proof['passed'] or not restored['passed'] or validator.poll() is not None:
                    raise ValueError('Numerical/resume/resident validator admission failed')
                s.health()
                admission = dict(passed=True, manifest_sha256=m['manifest_sha256'],
                    baseline_sha256=digest_file(root / 'baselines.json'),
                    zero_check_sha256=digest_file(root / 'zero_update_verified.json'),
                    canary=str(attempt / 'canary/result.json'), canary_sha256=digest_file(attempt / 'canary/result.json'),
                    resume_check=str(attempt / 'resume-check/result.json'), resume_check_sha256=digest_file(attempt / 'resume-check/result.json'),
                    gpu0_peak_observed_mib=s.gpu_peak, resident_validator_pid=validator.pid,
                    full_batch_seconds=proof['batch_seconds'], estimated_training_hours=proof['batch_seconds'] * 240 / 3600,
                    global_batch=32, physical_microbatch=8, gpu_count=1, tests=m['tests'])
                atomic_json(root / 'training_admission.json', admission, overwrite=False)
            verify_admission(m)
            commit = args.commit if resume else None
            commits = sorted((root / 'commits').glob('batch_*.json'))
            if resume and commit is None and commits:
                commit = commits[-1]
            if commit:
                read_commit(m, commit)
                if commits and Path(commit).resolve() != commits[-1].resolve():
                    raise ValueError('Resume must use the latest complete global commit')
            if not (root / 'training_completed.json').exists():
                s.run_phase('train', **({'resume_commit':commit} if commit else {}))
            reconcile_epoch_requests(m, budget(m)['global_batches'])
            s.run_phase('diagnostics')
            s.drain()
            hb.update(phase='candidate_selection')
            finish(m, s, hb)
            hb.update(phase='completed', training_episodes=budget(m)['training_episodes'], actor_updates=budget(m)['actor_updates'],
                      scientific_target_passed=read_json(root / 'result.json')['scientific_target_passed'])
        except BaseException as e:
            atomic_json(attempt / 'failure.json', {'error':str(e), 'traceback':traceback.format_exc(),
                'manifest_sha256':m['manifest_sha256'], 'scientific_result':'unverified; technical interruption'})
            raise
        finally:
            s.close()
            atomic_json(attempt / 'accounting.json', {'elapsed_seconds':time.monotonic()-s.started,
                'reserved_gpu_hours':(time.monotonic()-s.started)/3600, 'gpu_count':1,
                'peak_gpu0_mib':s.gpu_peak, 'validator_colocated_not_double_counted':True})
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    prepare_p = sub.add_parser('prepare')
    prepare_p.add_argument('--output', type=Path, required=True)
    prepare_p.add_argument('--tests', type=Path, required=True)
    for name in ('run', 'resume', 'verify', 'status'):
        sp = sub.add_parser(name)
        sp.add_argument('manifest', type=Path)
        if name == 'resume': sp.add_argument('--commit', type=Path)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command in ('run', 'resume'): run(args, resume=args.command == 'resume')
    elif args.command == 'verify':
        verify_manifest(read_json(args.manifest), inputs=True)
        print('Source, data, packages, schedule and resource recipe verified')
    else:
        m = read_json(args.manifest)
        print(json.dumps(read_json(Path(m['root']) / 'run_status.json'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
