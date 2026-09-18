#!/usr/bin/env python3
"""Frozen-IGA H3 continuation: canonical admission, paired temperatures, bounded promotion."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from onpolicy.utils.stage3_b_shared_b0 import cpus
from onpolicy.utils.stage3_h3_continuation import (
    PROTOCOL, ARMS, choose_parent, eligible, extend_arms, harmful, recipe, resources,
    schedule, select_candidate, reusable_evaluation, DUAL_PROFILES, SINGLE_PROFILES, active_arms,
    planned_new_steps, WIDE_BATCH_PROFILES, MB128_FRESH_PROFILE, FIXED_EPOCHS_POLICY,
)
from onpolicy.utils.stage3_h3_frozen import bind, checked, paired_metrics, verify_manifest as verify_parent
from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json, read_json

WORKER = 'onpolicy/scripts/train/stage3_h3_continuation_worker.py'
EVALUATOR = 'onpolicy/scripts/train/stage3_h3_continuation_eval.py'


def identity(value):
    return digest_json({k:v for k,v in value.items() if k != 'manifest_sha256'})


def verify(m):
    if m['protocol'] != PROTOCOL or identity(m) != m['manifest_sha256']:
        raise ValueError('Continuation suite identity changed')
    for k in ('parent_manifest','baselines','frozen_iga','plan','diagnostic_train24','canonical_references','tests'):
        checked(m[k])
    for record in m['checkpoints'].values():
        checked(record)
    if m.get('resume_origin'):
        checked(m['resume_origin']['suite'])
        checked(m['resume_origin']['arm_manifest'])
        for record in m['resume_origin']['commits']:
            checked(record)
    for root_key, files_key in [('source_root','source_files'),('evaluation_source_root','evaluation_source_files')]:
        for name, checksum in m[files_key].items():
            if digest_file(Path(m[root_key])/name) != checksum:
                raise ValueError('Frozen source changed: '+name)


def prepare(args):
    from onpolicy.scripts.train.run_stage3_h3_frozen import check_tests
    parent = read_json(args.parent); verify_parent(parent, inputs=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new experiment directory')
    tests = check_tests(args.tests)
    canonical = read_json(args.canonical_matrix)
    cached, evidence, source_root, source_files, legacy = {}, [], None, None, None
    for label in ('s0','epoch6'):
        old_manifest = read_json(checked(canonical['models'][label]))
        if source_files is not None:
            # The historical study patched its command-line orchestration entry
            # for S0. This adapter never imports/executes that entry; every
            # environment, policy, numeric and engine source must still match.
            unused_driver = 'onpolicy/scripts/train/run_stage3_h3_canonical_tau.py'
            computation = lambda files:{k:v for k,v in files.items() if k != unused_driver}
            if computation(source_files) != computation(old_manifest['source_files']):
                raise ValueError('Cached canonical models do not share the same evaluation computation')
        source_root, source_files = old_manifest['source_root'], old_manifest['source_files']
        rows = []
        for i in range(10):
            path = Path(old_manifest['root'])/'groups'/f'batch_{i:02d}_tau_0p3.json'
            group = read_json(path)
            if (group['manifest_sha256'] != old_manifest['manifest_sha256'] or not group['weights_unchanged']
                    or group['tau'] != .3 or group['batch_index'] != i):
                raise ValueError('Invalid cached canonical group')
            evidence.append(bind(path)); rows.extend(group['rows'])
        if ([r['case_id'] for r in rows] != [c['path'] for c in parent['splits']['validation']]
                or any(not r['completed'] or r.get('cycle_terminated') for r in rows)):
            raise ValueError('Cached canonical case coverage changed')
        cached[label] = rows
        if label == 'epoch6':
            legacy_record = old_manifest['original_results']['0p3']
            legacy = read_json(checked(legacy_record))['rows']; evidence.append(legacy_record)
    for name, checksum in source_files.items():
        if digest_file(Path(source_root)/name) != checksum:
            raise ValueError('Canonical execution snapshot changed')
    for package, version in parent['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Runtime dependency differs from frozen evaluation: '+package)
    allocation = resources(args.gpu, args.gpu_uuid, args.cpus)
    execution = getattr(args, 'execution', 'serial')
    template = recipe(parent['recipe'], allocation, 'C03', execution=execution,
                      dual_profile=getattr(args, 'dual_profile', None),
                      single_profile=getattr(args, 'single_profile', None),
                      optimizer_resize_after_epoch=getattr(args, 'optimizer_resize_after_epoch', None),
                      stopping_policy=getattr(args, 'stopping_policy', None))
    if template.get('execution_profile') == MB128_FRESH_PROFILE and getattr(args, 'resume_suite', None):
        raise ValueError('Fresh E8 continuation cannot import another continuation state')
    reused_suite = None
    reuse_path = getattr(args, 'resume_suite', None) or getattr(args, 'reuse_suite', None)
    if getattr(args, 'resume_suite', None) and getattr(args, 'reuse_suite', None):
        raise ValueError('Choose baseline reuse or committed resume, not both')
    if reuse_path:
        reused_suite = read_json(reuse_path); verify(reused_suite)
        if list(Path(reused_suite['root']).glob('arms/*/commits/*.json')) and not getattr(args,'resume_suite',None):
            raise ValueError('Baseline-only migration cannot discard formal training commits')
        if (reused_suite['evaluation_source_files'] != source_files
                or reused_suite['source_files'][EVALUATOR] != digest_file(ROOT/EVALUATOR)
                or reused_suite['resources'] != allocation or reused_suite['packages'] != parent['packages']
                or reused_suite['splits'] != parent['splits']):
            raise ValueError('Baseline migration changed evaluation computation, inputs or resources')
    diagnostic = read_json(args.diagnostic)
    train_ids = {c['content_sha256'] for c in parent['splits']['train']}
    if len(diagnostic['cases']) != 24 or not {c['content_sha256'] for c in diagnostic['cases']} <= train_ids:
        raise ValueError('Diagnostic cases must be the pre-registered Train24')
    final = read_json(Path(parent['root'])/'final_result.json')
    selected = final['selected']['checkpoint']
    if final['selected']['epoch'] != 8:
        raise ValueError('Expected the planned E8 parent')
    checkpoints = dict(epoch8=selected,
        epoch6=bind(Path(selected['path']).with_name('epoch_0006.pt')), s0=parent['initialization'])
    output.mkdir(parents=True)
    atomic_json(output/'canonical_references.json', dict(rows=cached, legacy_epoch6=legacy,
        evidence=evidence, evaluation_source_sha256=digest_json(source_files)), overwrite=False)
    code = {}
    names = subprocess.check_output(['rg','--files','onpolicy','-g','*.py','-g','*.yaml','-g','*.json','-g','*.sh',
        '-g','!**/dataset/**','-g','!**/results/**','-g','!**/__pycache__/**'], cwd=ROOT, text=True).splitlines()
    for name in sorted(names):
        src, dst = ROOT/name, output/'source'/name
        checksum = digest_file(src); dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst); dst.chmod(0o444)
        if digest_file(dst) != checksum:
            raise ValueError('Source changed during preparation')
        code[name] = checksum
    shutil.copyfile(args.tests, output/'tests.xml')
    m = dict(protocol=PROTOCOL, root=str(output), source_root=str(output/'source'), source_files=code,
        workspace_root=str(ROOT), workspace_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        parent_manifest=bind(args.parent.resolve()), baselines=bind(Path(parent['root'])/'baselines.json'),
        frozen_iga=parent['frozen_iga'], frozen_manifest=parent['frozen_manifest'], checkpoints=checkpoints,
        plan=bind(args.plan.resolve()), diagnostic_train24=bind(args.diagnostic.resolve()),
        canonical_references=bind(output/'canonical_references.json'),
        evaluation_source_root=source_root, evaluation_source_files=source_files,
        cached_source_difference=dict(unused_cli_only='onpolicy/scripts/train/run_stage3_h3_canonical_tau.py',
            adapter_imports_legacy_cli=False, evaluation_computation_files_identical=True),
        splits=parent['splits'], resources=allocation, recipe=template, python=sys.executable,
        execution_mode=execution,
        packages=parent['packages'], tests=dict(bind(output/'tests.xml'), passed=tests['passed']),
        budget=dict(screen_training_visits=768 if execution=='single' else 1536,
                    screen_ppo_steps=planned_new_steps(template,2) if execution=='single' else 48,
                    max_training_visits=3072 if execution=='single' else 6144,
                    max_new_ppo_steps=planned_new_steps(template,8) if execution=='single' else 192,
                    screen_wall_seconds=36*3600, max_wall_seconds=120*3600,
                    physical_microbatch=template['microbatch'], full_microbatch_benchmark=False),
        material_passport=dict(mode='run', verification_status='UNVERIFIED',
            authorization='User authorized implementation and launch; GPU0 confirmed after host resource clarification; CPU 0-31,64-95',
            solver_queries=0, conditional_next_studies='state-value/annealing/data coverage require their separate matched phase registration'),
        exposure=dict(confirmation_opened=False, tune_used_for_training=False, tune_used_for_selection=False),
        created_unix=time.time())
    if reused_suite is not None:
        m['reuse_suite'] = bind(reuse_path.resolve())
    if getattr(args, 'resume_suite', None):
        from onpolicy.utils.stage3_h3_resize import describe_origin
        m['resume_origin'] = describe_origin(reused_suite, m, reuse_path)
    m['manifest_sha256'] = identity(m)
    atomic_json(output/'manifest.json', m, overwrite=False); verify(m)
    if reused_suite is not None:
        migrate_baselines(m, reused_suite)
    if m.get('resume_origin'):
        from onpolicy.utils.stage3_h3_resize import import_committed_state
        parent_selection = read_json(Path(reused_suite['root'])/'parent_selection.json')
        parent_selection['manifest_sha256'] = m['manifest_sha256']
        atomic_json(output/'parent_selection.json',parent_selection,overwrite=False)
        controller = Controller.__new__(Controller)
        controller.m, controller.path, controller.root = m, output/'manifest.json', output
        _, arm = controller.arm_manifest('C03',parent_selection)
        mapping = import_committed_state(m, arm)
        migrate_baselines(m, reused_suite, labels=[f'C03_epoch_{i:04d}'
            for i in range(1,m['resume_origin']['epoch']+1)], checkpoint_map=mapping)
    atomic_json(output/'prepared.json', dict(prepared=True, manifest_sha256=m['manifest_sha256'],
                resources=allocation, solver_queries=0), overwrite=False)
    print(json.dumps(dict(manifest=str(output/'manifest.json'), resources=allocation, budget=m['budget']),ensure_ascii=False))


def migrate_baselines(m, previous, *, labels=None, checkpoint_map=None):
    """Rebind complete results only when the frozen evaluator is byte-identical."""
    root, old = Path(m['root']), Path(previous['root'])
    for label in labels or ('admission_legacy','admission_canonical','epoch8_validation','s0_tune','epoch8_tune','epoch6_tune'):
        source = old/'evaluations'/label/'result.json'
        if not source.exists():
            continue
        q = read_json(old/'requests'/f'{label}.json'); result = read_json(source)
        if (q['manifest_sha256'] != previous['manifest_sha256']
                or digest_json({k:v for k,v in q.items() if k != 'request_sha256'}) != q['request_sha256']
                or result['request_sha256'] != q['request_sha256']
                or result['manifest_sha256'] != previous['manifest_sha256']
                or result['evaluation_source_sha256'] != digest_json(m['evaluation_source_files'])
                or not result['completed'] or not result['weights_unchanged']
                or (label.startswith('admission_') and not result['reproduction_passed'])
                or any(not r['completed'] or r.get('cycle_terminated') for r in result['rows'])
                or [r['case_id'] for r in result['rows']] != [c['path'] for c in q['cases']]):
            raise ValueError('Invalid complete baseline for migration: '+label)
        checked(q['checkpoint'])
        if checkpoint_map:
            q['checkpoint'] = checkpoint_map[q['checkpoint']['sha256']]
        q.update(manifest_sha256=m['manifest_sha256'], output=str(root/'evaluations'/label))
        q['request_sha256'] = digest_json({k:v for k,v in q.items() if k != 'request_sha256'})
        atomic_json(root/'requests'/f'{label}.json',q,overwrite=False)
        atomic_json(root/'evaluations'/label/'result.json',dict(result,
            manifest_sha256=m['manifest_sha256'],request_sha256=q['request_sha256'],
            checkpoint=q['checkpoint'],
            reused_from=bind(source),original_seconds=result['seconds'],seconds=0.),overwrite=False)


class Controller:
    def __init__(self, m, path):
        self.m, self.path, self.root = m, path, Path(m['root'])
        self.arms = active_arms(m)
        self.child = None
        anchor = self.root/'budget_clock.json'
        if not anchor.exists():
            atomic_json(anchor,dict(started_unix=m.get('resume_origin',{}).get('started_unix',time.time()),
                manifest_sha256=m['manifest_sha256']),overwrite=False)
        clock = read_json(anchor)
        if clock['manifest_sha256'] != m['manifest_sha256']:
            raise ValueError('Budget clock belongs to a different experiment')
        self.started = clock['started_unix']
        self.attempt = self.root/'attempts'/f'{time.strftime("%Y%m%dT%H%M%S")}_{os.getpid()}'
        self.attempt.mkdir(parents=True)

    def status(self, **values):
        atomic_json(self.root/'run_status.json', dict(controller_pid=os.getpid(),
            updated_unix=time.time(), manifest_sha256=self.m['manifest_sha256'], solver_queries=0, **values))

    def wait_gpu(self):
        uuid = self.m['resources']['gpu_uuid']
        while True:
            raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                           '--format=csv,noheader,nounits'], text=True)
            occupied = [line.strip() for line in raw.splitlines() if uuid in line]
            if not occupied:
                break
            self.status(status='waiting_for_gpu', gpu_uuid=uuid, competing_processes=occupied)
            if time.time()-self.started > 24*3600:
                raise TimeoutError('Registered GPU remained occupied for 24 hours')
            time.sleep(30)

    def job(self, command, output, label, limit=21600):
        output.mkdir(parents=True, exist_ok=True)
        self.wait_gpu()
        if time.time()-self.started > self.m['budget']['max_wall_seconds']:
            raise TimeoutError('Continuation reached its 120-hour wall budget')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.m['resources']['gpu_uuid'],
                   HKBZ_STAGE3_WORKSPACE_ROOT=self.m['workspace_root'], PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
        started = time.time(); last_sample = 0.
        # Bind before Python initializes native helper threads. The child may
        # narrow its own main thread further, but cannot inherit the whole host.
        command = ['/usr/bin/taskset','--cpu-list',self.m['resources']['cpus'],*command]
        with (output/'worker.log').open('a') as log:
            self.child = subprocess.Popen(command, cwd=self.m['source_root'], env=env,
                                          stdout=log, stderr=subprocess.STDOUT)
            while self.child.poll() is None:
                self.status(status='running', phase=label, child_pid=self.child.pid,
                            phase_started_unix=started, output=str(output), resources=self.m['resources'])
                if time.time()-last_sample >= 30:
                    import psutil
                    raw = subprocess.check_output(['nvidia-smi','--id='+self.m['resources']['gpu_uuid'],
                        '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                    try:
                        process = psutil.Process(self.child.pid)
                        rss, affinity = process.memory_info().rss, process.cpu_affinity()
                    except psutil.NoSuchProcess:
                        break
                    cgroup = {}
                    try:
                        for line in Path(f'/proc/{self.child.pid}/cgroup').read_text().splitlines():
                            if line.startswith('0::'):
                                base = Path('/sys/fs/cgroup')/line[3:].lstrip('/')
                                for name in ('memory.current','memory.peak','memory.swap.current',
                                             'memory.high','memory.max','memory.swap.max',
                                             'memory.events','memory.pressure'):
                                    if (base/name).exists():
                                        value=(base/name).read_text().strip()
                                        cgroup[name]=int(value) if value.isdigit() else value
                    except FileNotFoundError:
                        pass
                    with (self.root/'resource_samples.jsonl').open('a') as f:
                        f.write(json.dumps(dict(unix=time.time(),phase=label,gpu=raw,child_pid=self.child.pid,
                            cpu_affinity=affinity,rss_bytes=rss,cgroup=cgroup,
                            host_available_bytes=psutil.virtual_memory().available))+'\n')
                    last_sample = time.time()
                if (time.time()-started > limit or time.time()-self.started > self.m['budget']['max_wall_seconds']):
                    self.stop_child(); raise TimeoutError('Phase exceeded its registered wall budget: '+label)
                time.sleep(5)
            code = self.child.wait(); self.child = None
            if code:
                raise RuntimeError(f'{label} exited {code}; see {output / "worker.log"}')

    def stop_child(self):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                self.child.kill(); self.child.wait()

    def evaluate(self, label, checkpoint, cases, *, decoder='canonical', expected=None):
        root = self.root/'evaluations'/label
        q = dict(manifest_sha256=self.m['manifest_sha256'],label=label,checkpoint=checkpoint,cases=cases,
                 decoder=decoder,expected=expected or [],output=str(root))
        q['request_sha256'] = digest_json(q)
        request = self.root/'requests'/f'{label}.json'
        if request.exists():
            if read_json(request) != q:
                raise ValueError('Evaluation request changed')
        else:
            atomic_json(request,q,overwrite=False)
        result = root/'result.json'
        if not result.exists() and not expected:
            for previous_path in sorted((self.root/'evaluations').glob('*/result.json')):
                previous_request = self.root/'requests'/f'{previous_path.parent.name}.json'
                if not previous_request.exists():continue
                prior_q = read_json(previous_request)
                if not reusable_evaluation(q, prior_q):continue
                prior = read_json(previous_path)
                if (not prior['completed'] or not prior['weights_unchanged']
                        or prior['request_sha256'] != prior_q['request_sha256']
                        or digest_json({k:v for k,v in prior_q.items() if k != 'request_sha256'}) != prior_q['request_sha256']):
                    raise ValueError('Invalid reusable complete evaluation')
                atomic_json(result,dict(prior,label=label,request_sha256=q['request_sha256'],
                                       reused_from=bind(previous_path),seconds=0.),overwrite=False)
                break
        if not result.exists():
            self.job([self.m['python'],'-B','-u',str(Path(self.m['source_root'])/EVALUATOR),
                      str(self.path),str(request)],root,'eval_'+label)
        row = read_json(result)
        if (not row['completed'] or row['request_sha256'] != q['request_sha256']
                or [r['case_id'] for r in row['rows']] != [c['path'] for c in cases]):
            raise ValueError('Evaluation completion identity differs')
        return row

    def baseline(self):
        cached = read_json(checked(self.m['canonical_references']))
        for record in cached['evidence']:
            checked(record)
        cases = self.m['splits']['validation'][12:24]
        wanted = {c['path'] for c in cases}
        for decoder, records in [('legacy',cached['legacy_epoch6']),('canonical',cached['rows']['epoch6'])]:
            self.evaluate('admission_'+decoder,self.m['checkpoints']['epoch6'],cases,decoder=decoder,
                          expected=[r for r in records if r['case_id'] in wanted])
        e8 = self.evaluate('epoch8_validation',self.m['checkpoints']['epoch8'],self.m['splits']['validation'])['rows']
        refs = lambda rows:{r['case_id']:r['makespan'] for r in rows}
        e8s0 = paired_metrics(e8,refs(cached['rows']['s0']))
        e6s0 = paired_metrics(cached['rows']['epoch6'],refs(cached['rows']['s0']))
        versus6 = paired_metrics(e8,refs(cached['rows']['epoch6']))
        choice = choose_parent(e8s0,e6s0,versus6)
        if self.m.get('execution_mode') == 'single' and choice != 'epoch8':
            raise ValueError('User selected the previous best E8; automatic parent fallback is disabled')
        selected = dict(label=choice,epoch=8 if choice=='epoch8' else 6,
            checkpoint=self.m['checkpoints'][choice],validation=e8 if choice=='epoch8' else cached['rows']['epoch6'],
            selection_uses_tune=False,e8_vs_s0=e8s0,e6_vs_s0=e6s0,e8_vs_e6=versus6,
            manifest_sha256=self.m['manifest_sha256'])
        path = self.root/'parent_selection.json'
        if path.exists():
            if read_json(path) != selected:
                raise ValueError('Parent selection changed')
        else:
            atomic_json(path,selected,overwrite=False)
        # Parent decision is frozen before any new Tune result is opened.
        for label in ('s0','epoch8') + (('epoch6',) if choice=='epoch6' else ()):
            self.evaluate(label+'_tune',self.m['checkpoints'][label],self.m['splits']['tune'])
        return selected

    def arm_manifest(self, arm, parent):
        path = self.root/'arms'/arm/'manifest.json'
        previous = read_json(checked(self.m['parent_manifest']))
        r = recipe(previous['recipe'],self.m['resources'],arm,parent_epoch=parent['epoch'],
                   execution=self.m.get('execution_mode','serial'),
                   dual_profile=self.m['recipe'].get('execution_profile')
                       if self.m.get('execution_mode') == 'dual' else None,
                   single_profile=self.m['recipe'].get('execution_profile')
                       if self.m.get('execution_mode') == 'single' else None,
                   optimizer_resize_after_epoch=self.m['recipe'].get('optimizer_resize_after_epoch'),
                   stopping_policy=self.m['recipe'].get('stopping_policy'))
        m = dict(protocol=PROTOCOL,root=str(path.parent),suite=bind(self.path),recipe=r,
            parent_checkpoint=parent['checkpoint'],parent_manifest=self.m['parent_manifest'],
            parent_selection=bind(self.root/'parent_selection.json'),baselines=self.m['baselines'],
            frozen_manifest=self.m['frozen_manifest'],schedule_sha256=digest_json(schedule(self.m['splits']['train'],r)))
        if self.m.get('resume_origin'):
            m['resume_epoch'] = self.m['resume_origin']['epoch']
        m['manifest_sha256'] = identity(m)
        if path.exists():
            if read_json(path) != m:
                raise ValueError('Arm manifest changed')
        else:
            atomic_json(path,m,overwrite=False)
        return path,m

    def canaries(self, arms):
        for arm,(path,m) in arms.items():
            out = Path(m['root'])/'canary'
            if (out/'passed.json').exists():
                proof=read_json(out/'passed.json')
                if not proof['passed'] or proof['manifest_sha256'] != m['manifest_sha256']:
                    raise ValueError('Foreign canary proof')
                continue
            for phase in ('canary','canary-resume'):
                if phase=='canary' and (out/'expected.pt').exists():
                    continue
                self.job([self.m['python'],'-B','-u',str(Path(self.m['source_root'])/WORKER),
                    phase,str(path),'--output',str(out)],out,f'{arm}_{phase}')

    def validation(self, arm, epoch, commit, parent):
        rows = self.evaluate(f'{arm}_epoch_{epoch:04d}',commit['checkpoint'],self.m['splits']['validation'])['rows']
        result = dict(arm=arm,epoch=epoch,checkpoint=commit['checkpoint'],rows=rows,
            new_ppo_steps=commit['new_ppo_steps'],training_episodes=commit['training_episodes'],
            ppo_budget_complete=commit['ppo_budget_complete'],
            versus_parent=paired_metrics(rows,{r['case_id']:r['makespan'] for r in parent['validation']}),
            manifest_sha256=self.m['manifest_sha256'])
        path=self.root/'arms'/arm/'validation'/f'epoch_{epoch:04d}.json'
        if path.exists():
            if read_json(path)!=result:raise ValueError('Validation summary changed')
        else:atomic_json(path,result,overwrite=False)
        return result

    def train_to(self, arm, item, until, parent):
        path,m=item;root=Path(m['root']); curve=[]
        for epoch in range(1,until+1):
            commit=root/'commits'/f'epoch_{epoch:04d}.json'
            if not commit.exists():
                if until==2 and time.time()-self.started>self.m['budget']['screen_wall_seconds']:
                    raise TimeoutError('Two-arm screen reached the 36-hour wall budget')
                out=self.attempt/arm/f'epoch_{epoch:04d}'
                self.job([self.m['python'],'-B','-u',str(Path(self.m['source_root'])/WORKER),
                    'train',str(path),'--epoch',str(epoch),'--output',str(out)],out,f'{arm}_epoch_{epoch:04d}',limit=12*3600)
            row=read_json(commit);checked(row['checkpoint']);checked(row['update'])
            if row['manifest_sha256']!=m['manifest_sha256'] or row['epoch']!=epoch:
                raise ValueError('Training commit ledger changed')
            if epoch in m['recipe']['evaluation_epochs']:
                result=self.validation(arm,epoch,row,parent);curve.append(result['versus_parent'])
                if m['recipe'].get('stopping_policy') != FIXED_EPOCHS_POLICY and harmful(curve):
                    stop=root/'early_stop.json'
                    if not stop.exists():atomic_json(stop,dict(epoch=epoch,reason='two_consecutive_harmful_validations'),overwrite=False)
                    return False
            if not row['ppo_budget_complete']:
                stop=root/'early_stop.json'
                if not stop.exists():atomic_json(stop,dict(epoch=epoch,reason='incomplete_registered_PPO_budget',
                    actual_new_steps=row['new_ppo_steps'],planned_new_steps=planned_new_steps(m['recipe'],epoch)),overwrite=False)
                return False
        return True

    def finish(self,parent,active):
        reference={r['case_id']:r['makespan'] for r in parent['validation']}
        candidates=[dict(arm='parent',epoch=0,checkpoint=parent['checkpoint'],rows=parent['validation'],
                         versus_parent=paired_metrics(parent['validation'],reference))]
        for arm in self.arms:
            candidates.extend(r for r in (read_json(p) for p in sorted((self.root/'arms'/arm/'validation').glob('epoch_*.json')))
                              if r['ppo_budget_complete'])
        selected=select_candidate(candidates)
        path=self.root/'selection.json'
        if path.exists():
            if read_json(path)!=selected:raise ValueError('Frozen champion changed')
        else:atomic_json(path,selected,overwrite=False)
        rows=self.evaluate('selected_tune',selected['checkpoint'],self.m['splits']['tune'])['rows']
        iga=read_json(checked(self.m['frozen_iga']))['references']['tune']
        parent_tune=read_json(self.root/'evaluations'/f'{parent["label"]}_tune'/'result.json')['rows']
        tune=dict(rows=rows,versus_iga1800=paired_metrics(rows,{k:v['iga1800']['makespan'] for k,v in iga.items()}),
                  versus_parent=paired_metrics(rows,{r['case_id']:r['makespan'] for r in parent_tune}))
        final=dict(completed=True,selected=selected,tune=tune,manifest_sha256=self.m['manifest_sha256'],
            solver_queries=0,confirmation_opened=False,scientific_target_confirmed=False,
            extension_arms=active,seconds=time.time()-self.started,
            next_research='state_value_matched_screen' if not active else 'lock_recipe_then_replicate_seeds',
            limitations=['One continuation seed; frozen IGA only covers exposed Tune60',
                         'Conditional P2/P3 studies need their own paired recipe and budget registration'])
        if self.m['recipe'].get('stopping_policy') == FIXED_EPOCHS_POLICY:
            final.update(stopping_policy=FIXED_EPOCHS_POLICY,
                training_budget_completed=bool(active),
                completed_training_epochs={a:len(list((self.root/'arms'/a/'commits').glob('epoch_*.json')))
                                           for a in self.arms})
        atomic_json(self.root/'final_result.json',final,overwrite=False)
        self.status(status='completed',selected_arm=selected['arm'],selected_epoch=selected['epoch'])

    def run(self):
        if self.m['recipe'].get('memory_protection') == 'disabled':
            from onpolicy.utils.stage3_memory_runtime import verify_unprotected_memory
            atomic_json(self.attempt/'memory_runtime.json',verify_unprotected_memory(),overwrite=False)
        parent=self.baseline()
        arms={a:self.arm_manifest(a,parent) for a in self.arms}
        self.canaries(arms)
        if self.m['recipe'].get('execution_profile') in WIDE_BATCH_PROFILES:
            path,m=arms['C03'];out=Path(m['root'])/'capacity'
            width=self.m['recipe']['microbatch']
            if not (out/'result.json').exists():
                self.job([self.m['python'],'-B','-u',str(Path(self.m['source_root'])/WORKER),
                    'capacity',str(path),'--output',str(out)],out,f'C03_capacity{width}')
            proof=read_json(out/'result.json')
            if (not proof['passed'] or proof['manifest_sha256'] != m['manifest_sha256']
                    or proof['backward_microbatch'] != width):
                raise ValueError(f'{width}-wide capacity probe was not admitted')
        if self.m['recipe'].get('stopping_policy') == FIXED_EPOCHS_POLICY:
            # Scores remain available for selection, but do not allocate the training budget.
            active=[a for a in self.arms if self.train_to(a,arms[a],self.m['recipe']['epochs'],parent)]
            self.finish(parent,active)
            return
        live=[a for a in self.arms if self.train_to(a,arms[a],2,parent)]
        live=[a for a in live if self.train_to(a,arms[a],4,parent)]
        summaries={a:read_json(self.root/'arms'/a/'validation/epoch_0004.json')['versus_parent'] for a in live}
        active=extend_arms(summaries)
        decision=dict(stage=4,extend_arms=active,summaries=summaries,
                      rule='validation mean gain >=0.3%, stress nonworse and risk gates',solver_queries=0)
        path=self.root/'promotion_epoch4.json'
        if not path.exists():atomic_json(path,decision,overwrite=False)
        elif read_json(path)!=decision:raise ValueError('Promotion decision changed')
        for a in active:self.train_to(a,arms[a],self.m['recipe']['epochs'],parent)
        self.finish(parent,active)


def run(path):
    path=path.resolve();m=read_json(path);verify(m)
    if (Path(m['root'])/'final_result.json').exists():
        return
    os.sched_setaffinity(0,cpus(m['resources']['cpus']))
    lock=(Path(m['root'])/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if m.get('execution_mode') == 'dual':
        from onpolicy.scripts.train.stage3_h3_dual_controller import DualController
        controller=DualController(m,path)
    else:
        controller=Controller(m,path)
    def terminate(*_):raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM,terminate)
    try:controller.run()
    except BaseException as exc:
        controller.stop_child();controller.status(status='failed',error=str(exc));raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare')
    for name in ('parent','canonical-matrix','tests','plan','diagnostic','output'):
        q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--gpu',type=int,choices=(0,1),required=True)
    q.add_argument('--gpu-uuid',required=True);q.add_argument('--cpus',required=True)
    q.add_argument('--execution',choices=('serial','dual','single'),default='serial')
    q.add_argument('--dual-profile',choices=tuple(DUAL_PROFILES))
    q.add_argument('--single-profile',choices=tuple(SINGLE_PROFILES))
    q.add_argument('--optimizer-resize-after-epoch',type=int)
    q.add_argument('--stopping-policy',choices=(FIXED_EPOCHS_POLICY,))
    q.add_argument('--reuse-suite',type=Path)
    q.add_argument('--resume-suite',type=Path)
    q=sub.add_parser('run');q.add_argument('manifest',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    else:run(a.manifest)


if __name__=='__main__':main()
