#!/usr/bin/env python3
"""Prepare and run H3 B_SHARED with frozen IGA references and no solver path."""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from onpolicy.runner.shared.stage3_b_shared_b0_engine import BSharedB0Engine
from onpolicy.runner.shared.stage3_h3_frozen_engine import H3FrozenEngine, model_digest
from onpolicy.runner.shared.stage3_research_engine import seed_all
from onpolicy.scripts.train.stage3_b_shared_b0_worker import Progress, compare_states, evaluate_cases
from onpolicy.utils.stage3_b0_single_model import CpuEnvironmentPool
from onpolicy.utils.stage3_b_shared_b0 import check_resources, cpus, normalization_moments, record_summary
from onpolicy.utils.stage3_h3_frozen import (
    PROTOCOL, PLANNING, SOURCE_SHA, bind, budget, checked, freeze_iga, manifest_identity,
    paired_metrics, recipe, require_training_teachers, schedule, selection_eligible, verify_manifest,
)
from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json, read_json

BEST6_SHA = 'd8a99daa66e7cf89cc9c83a11b4ace2735e3a01c59020e80e58deb4c911fc87e'


def save_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(0))


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch']); torch.cuda.set_rng_state(value['cuda'], 0)


def resource_snapshot(m, pid):
    import psutil
    raw = subprocess.check_output(['nvidia-smi', '--id=' + m['recipe']['gpu_uuid'],
        '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    index, uuid, total, used, utilization = [v.strip() for v in raw.strip().split(',')]
    if int(index) != 0 or uuid != m['recipe']['gpu_uuid']:
        raise ValueError('Resource monitor observed a different GPU')
    process = psutil.Process(pid)
    cg = Path(f'/proc/{pid}/cgroup').read_text().strip().splitlines()
    memory_current = None
    for line in cg:
        if line.startswith('0::'):
            file = Path('/sys/fs/cgroup') / line[3:].lstrip('/') / 'memory.current'
            if file.exists():
                memory_current = int(file.read_text())
    return dict(unix=time.time(), pid=pid, gpu_index=0, gpu_uuid=uuid,
                gpu_total_mib=int(total), gpu_used_mib=int(used), gpu_utilization=int(utilization),
                allowed_cpus=process.cpu_affinity(), child_processes=len(process.children(recursive=True)),
                process_rss_bytes=process.memory_info().rss, cgroup_memory_bytes=memory_current,
                host_available_bytes=psutil.virtual_memory().available)


def check_tests(path):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag == 'testsuite' else list(tree.iter('testsuite'))
    failures = sum(int(s.get('failures', 0)) + int(s.get('errors', 0)) for s in suites)
    count = sum(int(s.get('tests', 0)) - int(s.get('skipped', 0)) for s in suites)
    if failures or count < 10:
        raise ValueError('At least ten passing non-skipped contract tests are required')
    return dict(**bind(path), passed=count, failures=failures)


def prepare(args):
    parent_path = args.parent.resolve()
    previous = read_json(parent_path)
    r = recipe(previous['recipe'])
    r['arm'] = args.arm
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new run directory; existing experiments are immutable')
    tests = check_tests(args.tests)
    init = Path(previous['root']) / 'attempts/20260913T164751_1357885/train/models/batch_0023.pt'
    if digest_file(init) != BEST6_SHA:
        raise ValueError('Expected the selected Best6 checkpoint')
    frozen = previous['frozen_manifest']
    checked(frozen)
    b0 = Path(frozen['path']).parent / 'checkpoints/b0.pt'
    if digest_file(b0) != SOURCE_SHA:
        raise ValueError('B0 changed')
    index = freeze_iga(args.iga_root, previous['splits'])
    if args.arm == 'R1':
        require_training_teachers(index, previous['splits']['train'])
        raise ValueError('R1 execution requires a separately validated teacher adapter and pilot receipt')
    output.mkdir(parents=True)
    atomic_json(output / 'frozen_iga.json', index, overwrite=False)
    code = {}
    names = subprocess.check_output([
        'rg', '--files', 'onpolicy', '-g', '*.py', '-g', '*.yaml', '-g', '*.json', '-g', '*.sh',
        '-g', '!**/dataset/**', '-g', '!**/results/**', '-g', '!**/__pycache__/**'],
        cwd=ROOT, text=True).splitlines()
    for name in sorted(names):
        src, dst = ROOT / name, output / 'source' / name
        checksum = digest_file(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if digest_file(dst) != checksum:
            raise ValueError('Source changed during snapshot')
        dst.chmod(0o444)
        code[name] = checksum
    shutil.copyfile(args.tests, output / 'tests.xml')
    tests.update(bind(output / 'tests.xml'))
    canaries = [min((c for c in previous['splits']['train'] if c['profile'] == profile),
                    key=lambda c: c['content_sha256'])
                for profile in ('balanced', 'low_load_ood', 'resource_ood', 'stress_joint')]
    m = dict(protocol=PROTOCOL, root=str(output), workspace_root=str(ROOT),
             source_root=str(output / 'source'), python=sys.executable,
             parent=bind(parent_path), b0=bind(b0), initialization=bind(init),
             frozen_manifest=frozen, frozen_iga=bind(output / 'frozen_iga.json'),
             recipe=r, splits=previous['splits'], canary_cases=canaries,
             code_files=code, code_sha256=digest_json(code), tests=tests,
             schedule_sha256=digest_json(schedule(previous['splits']['train'], r)),
             packages={p: importlib.metadata.version(p) for p in
                       ('torch', 'torch-geometric', 'numpy', 'scipy', 'PyYAML', 'psutil')},
             created_unix=time.time(), budget=budget(r),
             material_passport=dict(mode='implementation -> run', scientific_result='unverified',
                 authorization='User requested implementation and launch; same GPU0/half-CPU resources; IGA frozen, no solve'),
             scope_amendment=dict(arm='R0', reason='No matching frozen H3 Train240 teachers; no IGA solving authorized',
                 frozen_iga_splits=[s for s, missing in index['missing'].items() if not missing],
                 r1='not launched; missing frozen H3 training demonstrations',
                 confirmation='closed; no matching frozen IGA reference, no independent IGA success claim'),
             exposure=dict(confirmation_costs_opened=False, tune_used_for_gradients=False,
                           validation_used_for_gradients=False))
    m['manifest_sha256'] = manifest_identity(m)
    atomic_json(output / 'manifest.json', m, overwrite=False)
    verify_manifest(m, inputs=True)
    atomic_json(output / 'preparation.json', dict(prepared=True, manifest_sha256=m['manifest_sha256'],
                solver_queries=0, training_started=False, budget=m['budget']), overwrite=False)
    print(json.dumps({'manifest': str(output / 'manifest.json'), 'budget': m['budget'],
                      'frozen_iga_missing': {s: len(v) for s, v in index['missing'].items()}}), flush=True)


def engine(m, output, *, training=True):
    return H3FrozenEngine(m['frozen_manifest']['path'], config=m['recipe'], training=training,
                          runtime=dict(sampling_cpus=sorted(cpus(m['recipe']['cpus'])),
                                       sampling_output=str(output / 'sampling')))


def eval_rows(m, runner, cases, output, hb, *, label, checkpoint_sha, times=False):
    if runner.pool is not None:
        raise ValueError('Evaluation cannot replace a live sampling pool')
    rng = save_rng()
    runner.pool = CpuEnvironmentPool(12, timeout=m['recipe']['ipc_timeout_seconds'],
                                    affinity=cpus(m['recipe']['validator_cpus']))
    try:
        rows = evaluate_cases(m, runner, cases, output, hb, times=times,
            binding=dict(protocol=PROTOCOL, model_sha256=checkpoint_sha, planning=PLANNING,
                         code_sha256=m['code_sha256'], cases_sha256=digest_json(cases),
                         history=m['recipe']['history'], decoder='H', tau=.3, seed=1, label=label),
            timeout=m['recipe']['evaluation_timeout_seconds'])
        return rows
    finally:
        runner.pool.close(); runner.pool = None
        restore_rng(rng)


def canary(m, output, hb, *, resumed=False):
    runner = engine(m, output / ('fresh_process_resume' if resumed else 'first_process'))
    try:
        runner.warm_start(m['initialization']['path'], expected_sha256=m['initialization']['sha256'])
        cases = m['canary_cases']
        receipt_path = output / 'canary_expected.json'
        if not resumed:
            raw = eval_rows(m, runner, cases, output / 'reference', hb, label='canary_s0_h3',
                            checkpoint_sha=m['initialization']['sha256'], times=True)
            costs = {x['result']['case_id']: x['result']['makespan'] for x in raw}
            moments = np.asarray([x['normalization_moments'] for x in raw]).sum(axis=0).tolist()
            runner.set_normalization(moments)
            initial = output / 'initial.pt'
            runner.save(initial, manifest_sha256=m['manifest_sha256'], next_batch=0, diagnostic=True)
            initial_model = model_digest(runner.policy.ac)
            rows = runner.collect_logical(cases, [2026091510 + i for i in range(4)],
                       [f'canary-1-{i}' for i in range(4)], logical_id='canary-1', heartbeat=hb)
            update = runner.update_logical(rows, costs, logical_id='canary-1', shuffle_seed=1,
                                          diagnostic=True, minibatch=2, microbatch=2, heartbeat=hb)
            if update['actual_ppo_steps'] != 4 or model_digest(runner.policy.ac) == initial_model:
                raise ValueError('Canary did not apply actual full-shared optimizer updates')
            if not runner.policy.actor_optimizer.state or not runner.policy.critic_optimizer.state:
                raise ValueError('Canary optimizer state is empty')
            original = torch.load(initial, map_location='cpu', weights_only=False)['model']
            changed = [name for name, value in runner.policy.ac.state_dict().items()
                       if not torch.equal(original[name], value.cpu())]
            for prefix in ('encoder.op_embedding.', 'encoder.convs.0.', 'encoder.convs.3.',
                           'actor.', 'device_actor.', 'transporter_actor.'):
                if not any(name.startswith(prefix) for name in changed):
                    raise ValueError(f'Canary did not demonstrate an update for {prefix}')
            atomic_json(output / 'shared_updates.json', dict(changed_parameters=changed,
                        passed=True, manifest_sha256=m['manifest_sha256']), overwrite=False)
            del original
            before = output / 'before_resume.pt'
            runner.save(before, manifest_sha256=m['manifest_sha256'], next_batch=1, diagnostic=True)
            atomic_json(output / 'canary_first_update.json', update, overwrite=False)
            del rows
        else:
            receipt = read_json(receipt_path)
            costs = receipt['source_costs']
            before = checked(receipt['before'])
        cursor = runner.resume(before, manifest_sha256=m['manifest_sha256'], allow_diagnostic=True)
        if cursor != 1 or runner.policy_updates != 4:
            raise ValueError('Resume cursor or actual optimizer step differs')
        # Both paths restore the committed RNG before collecting genuinely new visits.
        rows = runner.collect_logical(cases, [2026091520 + i for i in range(4)],
                   [f'canary-2-{i}' for i in range(4)], logical_id='canary-2', heartbeat=hb)
        update = runner.update_logical(rows, costs, logical_id='canary-2', shuffle_seed=2,
                                      diagnostic=True, minibatch=2, microbatch=2, heartbeat=hb)
        path = output / ('actual_after_resume.pt' if resumed else 'expected_after_resume.pt')
        runner.save(path, manifest_sha256=m['manifest_sha256'], next_batch=2, diagnostic=True)
        summaries = [record_summary(x) for x in rows]
        if not resumed:
            atomic_json(receipt_path, dict(before=bind(before), expected=bind(path), source_costs=costs,
                         rows=summaries, first_steps=4, next_steps=update['actual_ppo_steps']), overwrite=False)
        else:
            a = torch.load(checked(receipt['expected']), map_location='cpu', weights_only=False)
            b = torch.load(path, map_location='cpu', weights_only=False)
            if summaries != receipt['rows']:
                raise ValueError('Restored process changed new sampled actions/history/costs')
            for key in ('model', 'actor_optim', 'critic_optim', 'value_normalizer', 'policy_updates',
                        'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda'):
                compare_states(a[key], b[key], exact=True)
            atomic_json(output / 'canary_passed.json', dict(passed=True, fresh_process_resume=True,
                actual_updates=runner.policy_updates, trajectory_matches=4, solver_queries=0,
                numerical_equivalence='No cross-microbatch equivalence claim',
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                manifest_sha256=m['manifest_sha256']), overwrite=False)
    finally:
        runner.close()


def baseline(m, output, hb):
    root = Path(m['root'])
    if (root / 'baselines.json').exists():
        return read_json(root / 'baselines.json')
    parent = read_json(checked(m['parent']))
    h2 = BSharedB0Engine(m['frozen_manifest']['path'], config=m['recipe'], width=12,
                        training=False, create_pool=False)
    h2.pool = CpuEnvironmentPool(12, timeout=m['recipe']['ipc_timeout_seconds'],
                                affinity=cpus(m['recipe']['validator_cpus']))
    try:
        hb.update(event='h2_zero_update_canary', cases=12)
        cases = m['splits']['tune'][:12]
        rows = h2.rollout(cases, [1] * 12, deterministic=True, heartbeat=hb)
        old = {x['case_id']: x for x in read_json(Path(parent['root']) / 'baselines.json')['records']['tune']}
        for row in rows:
            reference = old[row['case_id']]
            if any(row[k] != reference[k] for k in ('makespan', 'actions_sha256', 'history_sha256')):
                raise ValueError(f'H2 zero-update reproduction failed: {row["case_id"]}')
        atomic_json(output / 'h2_zero_update.json', dict(passed=True, cases=12, optimizer_steps=0,
                    manifest_sha256=m['manifest_sha256']), overwrite=False)
    finally:
        h2.close()
    del h2
    gc.collect(); torch.cuda.empty_cache()
    runner = engine(m, output)
    data, moments = {}, None
    try:
        for label, source in [('B0', m['b0']), ('Best6', m['initialization'])]:
            runner.warm_start(source['path'], expected_sha256=source['sha256'])
            data[label] = {}
            for split in ('tune', 'validation', 'train'):
                if label == 'Best6' and split == 'train' and m['recipe']['arm'] == 'R0':
                    # R0 only needs B0 training costs/ValueNorm. S0 train
                    # trajectories were for the now-unrequested teacher arm.
                    continue
                hb.update(event='h3_baseline', model=label, split=split)
                raw = eval_rows(m, runner, m['splits'][split], root / 'baseline_cases' / label / split,
                                hb, label=label, checkpoint_sha=source['sha256'],
                                times=label == 'B0' and split == 'train')
                data[label][split] = [row['result'] for row in raw]
                if label == 'B0' and split == 'train':
                    moments = np.asarray([row['normalization_moments'] for row in raw]).sum(axis=0).tolist()
        b0_vs_best = paired_metrics(data['B0']['validation'],
                                    {r['case_id']: r['makespan'] for r in data['Best6']['validation']})
        choose_b0 = b0_vs_best['completed'] and b0_vs_best['gap_fraction'] <= -.003 and b0_vs_best['tail_ratio'] <= 1
        chosen = 'B0' if choose_b0 else 'Best6'
        source = m['b0'] if choose_b0 else m['initialization']
        payload = dict(manifest_sha256=m['manifest_sha256'], planning=PLANNING, records=data,
                       initialization=source, initialization_label=chosen,
                       initialization_rule='B0 if >=0.3% better on Validation and tail nonworse; otherwise Best6',
                       b0_vs_best_validation=b0_vs_best,
                       source_costs={r['case_id']: r['makespan'] for r in data['B0']['train']},
                       normalization_moments=moments, normalization_sha256=digest_json(moments),
                       completed_case_episodes=sum(len(rows) for splits in data.values() for rows in splits.values()),
                       optimizer_steps=0, solver_queries=0)
        atomic_json(root / 'baselines.json', payload, overwrite=False)
        return payload
    finally:
        runner.close()


def latest_commit(m):
    root = Path(m['root'])
    commits = sorted((root / 'commits').glob('epoch_*.json'))
    if not commits:
        return None
    applied = 0
    for expected, path in enumerate(commits, 1):
        row = read_json(path)
        if (row['manifest_sha256'] != m['manifest_sha256'] or row['next_epoch'] != expected
                or row['training_episodes'] != expected * 384):
            raise ValueError('Noncontiguous or invalid committed epoch')
        checked(row['checkpoint']); checked(row['update'])
        checked(row['baselines'])
        update = read_json(row['update']['path'])
        step_count = update['update']['actual_ppo_steps']
        applied += step_count
        if (not 0 <= step_count <= 12 or row['actual_ppo_steps'] != applied
                or update['actual_ppo_steps'] != applied or update['epoch'] != expected
                or update['training_episodes'] != expected * 384):
            raise ValueError('Actual Adam steps do not match the committed update ledger')
    return read_json(commits[-1])


def evaluate_checkpoint(m, runner, base, epoch, checkpoint, hb):
    root = Path(m['root'])
    target = root / 'validator/results' / f'epoch_{epoch:04d}.json'
    if target.exists():
        old = read_json(target)
        if old['checkpoint'] != bind(checkpoint) or old['manifest_sha256'] != m['manifest_sha256']:
            raise ValueError('Completed evaluation checkpoint identity changed')
        return old
    rows = eval_rows(m, runner, m['splits']['validation'],
                    root / 'validator/cases' / f'epoch_{epoch:04d}', hb,
                    label=f'epoch_{epoch}', checkpoint_sha=digest_file(checkpoint))
    records = [r['result'] for r in rows]
    chosen = base['initialization_label']
    result = dict(completed=True, checkpoint=bind(checkpoint), epoch=epoch, rows=records,
                  manifest_sha256=m['manifest_sha256'],
                  versus_s0=paired_metrics(records, {r['case_id']: r['makespan'] for r in base['records'][chosen]['validation']}),
                  versus_b0=paired_metrics(records, {r['case_id']: r['makespan'] for r in base['records']['B0']['validation']}),
                  iga_comparison=None, iga_missing=True, solver_queries=0)
    atomic_json(target, result, overwrite=False)
    return result


def first_cycle_capacity(m, runner, trajectories, costs, logical_id, output, hb, rollout_seconds):
    """Benchmark registered physical chunks; retain the fastest valid update.

    All candidates use the same frozen rollout and identical Adam/RNG start.
    Selection observes timing and validity only, never validation reward. The
    retained candidate is the first formal update; other computation is logged
    separately. No cross-microbatch numerical-equivalence claim is made.
    """
    root = Path(m['root'])
    configuration = root / 'physical_configuration.json'
    if configuration.exists():
        chosen = read_json(configuration)
        if (chosen['manifest_sha256'] != m['manifest_sha256']
                or chosen['microbatch'] not in m['recipe']['physical_microbatch_candidates']):
            raise ValueError('Physical configuration identity changed')
        runner.physical_microbatch = chosen['microbatch']
        return runner.update_logical(trajectories, costs, logical_id=logical_id,
            shuffle_seed=m['recipe']['optimizer_shuffle_seed'] + 1, heartbeat=hb)
    initial = root / 'models/initial.pt'
    after_collection_rng = save_rng()
    measured = []
    for microbatch in m['recipe']['physical_microbatch_candidates']:
        runner.resume(initial, manifest_sha256=m['manifest_sha256'])
        restore_rng(after_collection_rng)
        runner.physical_microbatch = microbatch
        runner.policy.actor_optimizer.zero_grad(set_to_none=True)
        runner.policy.critic_optimizer.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        hb.update(event='first_cycle_capacity', physical_microbatch=microbatch,
                  optimizer_minibatch=64, retained_visits=384)
        started = time.monotonic()
        try:
            update = runner.update_logical(trajectories, costs, logical_id=logical_id,
                shuffle_seed=m['recipe']['optimizer_shuffle_seed'] + 1, heartbeat=hb)
            update_seconds = time.monotonic() - started
            checkpoint = output / 'capacity' / f'microbatch_{microbatch}.pt'
            runner.save(checkpoint, manifest_sha256=m['manifest_sha256'], next_batch=1, diagnostic=True)
            full_budget = update['actual_ppo_steps'] == 12 and update['completed_passes'] == 2
            sample = dict(microbatch=microbatch, valid=full_budget, update=update, checkpoint=bind(checkpoint),
                          validity_reason='complete_12_step_update' if full_budget else 'partial_ppo_budget',
                          update_seconds=update_seconds, rollout_seconds=rollout_seconds,
                          complete_cycle_seconds=time.monotonic() - started + rollout_seconds,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        except torch.cuda.OutOfMemoryError as exc:
            sample = dict(microbatch=microbatch, valid=False, error_type='cuda_out_of_memory',
                          error=str(exc), discarded_ppo_steps=runner.policy_updates,
                          measured_seconds=time.monotonic() - started)
            runner.policy.actor_optimizer.zero_grad(set_to_none=True)
            runner.policy.critic_optimizer.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
        atomic_json(output / 'capacity' / f'microbatch_{microbatch}.json', sample, overwrite=False)
        measured.append(sample)
    valid = [row for row in measured if row['valid']]
    if not valid:
        raise RuntimeError('Neither registered physical microbatch fits the resource limit')
    chosen = min(valid, key=lambda row: row['complete_cycle_seconds'])
    runner.resume(checked(chosen['checkpoint']), manifest_sha256=m['manifest_sha256'], allow_diagnostic=True)
    discarded = sum(row.get('update', {}).get('actual_ppo_steps', row.get('discarded_ppo_steps', 0))
                    for row in measured if row is not chosen)
    atomic_json(configuration, dict(manifest_sha256=m['manifest_sha256'], microbatch=chosen['microbatch'],
        optimizer_minibatch=64, selection='lowest_complete_cycle_time_among_valid_registered_candidates',
        candidates=[bind(output / 'capacity' / f'microbatch_{row["microbatch"]}.json') for row in measured],
        discarded_diagnostic_ppo_steps=discarded, selected_candidate=chosen['checkpoint'],
        solver_queries=0, full_microbatch_numerical_equivalence='waived_not_claimed'), overwrite=False)
    result = copy.deepcopy(chosen['update'])
    result['capacity_selection'] = dict(microbatch=chosen['microbatch'],
        selected_update_seconds=chosen['update_seconds'], discarded_diagnostic_ppo_steps=discarded,
        selected_complete_cycle_seconds=chosen['complete_cycle_seconds'],
        selected_peak_allocated_mib=chosen['peak_allocated_mib'],
        all_candidates_peak_allocated_mib=max(row.get('peak_allocated_mib', 0) for row in measured),
        all_candidates_peak_reserved_mib=max(row.get('peak_reserved_mib', 0) for row in measured),
        record=bind(configuration))
    return result


def train(m, output, hb):
    root = Path(m['root'])
    if (root / 'training_completed.json').exists():
        return
    base = read_json(root / 'baselines.json')
    if base['manifest_sha256'] != m['manifest_sha256'] or base['planning'] != PLANNING:
        raise ValueError('Training baseline identity differs')
    runner = engine(m, output)
    cursor = 0
    start_all = time.monotonic()
    try:
        runner.warm_start(base['initialization']['path'], expected_sha256=base['initialization']['sha256'])
        runner.set_normalization(base['normalization_moments'], expected_sha=base['normalization_sha256'])
        commit = latest_commit(m)
        if commit:
            cursor = runner.resume(checked(commit['checkpoint']), manifest_sha256=m['manifest_sha256'])
            if cursor != commit['next_epoch'] or runner.policy_updates != commit['actual_ppo_steps']:
                raise ValueError('Checkpoint/commit optimizer cursor mismatch')
            if cursor in m['recipe']['evaluation_epochs']:
                evaluate_checkpoint(m, runner, base, cursor, commit['checkpoint']['path'], hb)
        else:
            initial = root / 'models/initial.pt'
            if initial.exists():
                runner.resume(initial, manifest_sha256=m['manifest_sha256'])
            else:
                seed_all(m['recipe']['seed'])
                runner.save(initial, manifest_sha256=m['manifest_sha256'], next_batch=0)
        for row in schedule(m['splits']['train'], m['recipe'])[cursor:]:
            epoch = row['data_epoch']
            if time.monotonic() - start_all > m['recipe']['train_wall_timeout_seconds']:
                raise TimeoutError('Training phase reached its 72-hour budget')
            hb.update(event='logical_collection_start', epoch=epoch, completed_visits=cursor * 384,
                      actual_ppo_steps=runner.policy_updates, environment_slots=240)
            begin = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            logical_id = f'{m["manifest_sha256"]}:epoch:{epoch}'
            trajectories = runner.collect_logical(row['cases'], row['seeds'], row['visit_ids'],
                                                  logical_id=logical_id, heartbeat=hb)
            rollout_seconds = time.monotonic() - begin
            rollout_peak_allocated = torch.cuda.max_memory_allocated() / 2**20
            rollout_peak_reserved = torch.cuda.max_memory_reserved() / 2**20
            atomic_json(output / 'rollouts' / f'epoch_{epoch:04d}.json',
                        [record_summary(t) for t in trajectories], overwrite=False)
            update_start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            if epoch == 1:
                update = first_cycle_capacity(m, runner, trajectories, base['source_costs'], logical_id,
                                              output, hb, rollout_seconds)
            else:
                update = runner.update_logical(trajectories, base['source_costs'], logical_id=logical_id,
                    shuffle_seed=m['recipe']['optimizer_shuffle_seed'] + epoch, heartbeat=hb)
            update_seconds = time.monotonic() - update_start
            checkpoint = output / 'models' / f'epoch_{epoch:04d}.pt'
            runner.save(checkpoint, manifest_sha256=m['manifest_sha256'], next_batch=epoch)
            update_path = output / 'updates' / f'epoch_{epoch:04d}.json'
            atomic_json(update_path, dict(epoch=epoch, training_episodes=epoch * 384,
                actual_ppo_steps=runner.policy_updates, update=update,
                rollout_seconds=rollout_seconds, update_seconds=update_seconds,
                seconds=time.monotonic() - begin,
                rollout_peak_allocated_mib=rollout_peak_allocated,
                rollout_peak_reserved_mib=rollout_peak_reserved,
                update_peak_allocated_mib=max(torch.cuda.max_memory_allocated() / 2**20,
                    update.get('capacity_selection', {}).get('all_candidates_peak_allocated_mib', 0)),
                update_peak_reserved_mib=max(torch.cuda.max_memory_reserved() / 2**20,
                    update.get('capacity_selection', {}).get('all_candidates_peak_reserved_mib', 0))), overwrite=False)
            commit = dict(manifest_sha256=m['manifest_sha256'], next_epoch=epoch,
                          training_episodes=epoch * 384, actual_ppo_steps=runner.policy_updates,
                          checkpoint=bind(checkpoint), update=bind(update_path),
                          baselines=bind(root / 'baselines.json'), committed_unix=time.time())
            atomic_json(root / 'commits' / f'epoch_{epoch:04d}.json', commit, overwrite=False)
            cursor = epoch
            hb.update(event='epoch_committed', epoch=epoch, completed_visits=epoch * 384,
                      actual_ppo_steps=runner.policy_updates, checkpoint=str(checkpoint))
            del trajectories
            gc.collect(); torch.cuda.empty_cache()
            if epoch in m['recipe']['evaluation_epochs']:
                evaluated = evaluate_checkpoint(m, runner, base, epoch, checkpoint, hb)
                earlier = sorted((root / 'validator/results').glob('epoch_*.json'))
                if len(earlier) >= 2 and all(read_json(p)['versus_s0']['gap_fraction'] > .01 for p in earlier[-2:]):
                    atomic_json(root / 'early_stop.json', dict(reason='two_validation_regressions_over_1pct',
                                epoch=epoch, manifest_sha256=m['manifest_sha256']), overwrite=False)
                    break
                if epoch == 4 and evaluated['versus_s0']['gap_fraction'] > -.003:
                    # No frozen Validation IGA exists, so the IGA-gap branch of
                    # the original gate is unavailable. Require observable RL gain.
                    best = min(read_json(p)['versus_s0']['gap_fraction'] for p in earlier)
                    if best > -.003:
                        atomic_json(root / 'early_stop.json', dict(reason='no_0p3pct_dev_gain_by_epoch4',
                                    epoch=epoch, iga_gap_gate='unavailable_frozen_reference',
                                    manifest_sha256=m['manifest_sha256']), overwrite=False)
                        break
        atomic_json(root / 'training_completed.json', dict(completed=True, epochs=cursor,
            planned_budget_completed=cursor == 8, training_episodes=cursor * 384,
            actual_ppo_steps=runner.policy_updates, solver_queries=0,
            manifest_sha256=m['manifest_sha256']), overwrite=False)
    finally:
        runner.close()


def report(m, output, hb):
    root = Path(m['root'])
    if (root / 'final_result.json').exists():
        return
    base = read_json(root / 'baselines.json')
    candidates = [read_json(p) for p in sorted((root / 'validator/results').glob('epoch_*.json'))]
    eligible = [r for r in candidates if selection_eligible(r['versus_s0'])]
    selected = min(eligible, key=lambda r: (r['versus_s0']['makespan'], r['epoch'])) if eligible else None
    if selected is not None:
        checkpoint = selected['checkpoint']
        runner = engine(m, output)
        try:
            runner.warm_start(base['initialization']['path'], expected_sha256=base['initialization']['sha256'])
            runner.resume(checked(checkpoint), manifest_sha256=m['manifest_sha256'])
            raw = eval_rows(m, runner, m['splits']['tune'], root / 'validator/cases/selected_tune', hb,
                            label='selected_tune', checkpoint_sha=checkpoint['sha256'])
            rows = [r['result'] for r in raw]
            iga = read_json(checked(m['frozen_iga']))['references']['tune']
            tune = dict(rows=rows, versus_iga1800=paired_metrics(rows,
                        {c: r['iga1800']['makespan'] for c, r in iga.items()}),
                        versus_s0=paired_metrics(rows, {r['case_id']: r['makespan']
                            for r in base['records'][base['initialization_label']]['tune']}))
        finally:
            runner.close()
    else:
        tune = None
    verify_manifest(m)
    atomic_json(root / 'final_result.json', dict(completed=True, selected=selected, tune=tune,
        training=read_json(root / 'training_completed.json'), manifest_sha256=m['manifest_sha256'],
        scientific_target_confirmed=False, confirmation_opened=False, solver_queries=0,
        limitations=['R0 only; frozen H3 training teachers absent',
                     'No frozen H3 IGA Validation/Confirmation references; Tune comparison is descriptive',
                     'One training seed; independent IGA near-parity not confirmed']), overwrite=False)


def run_phase(m, phase, output):
    os.sched_setaffinity(0, cpus(m['recipe']['cpus']))
    check_resources(m)
    verify_manifest(m)
    for name, version in m['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Package changed: {name}')
    output.mkdir(parents=True, exist_ok=True)
    seed_all(m['recipe']['seed'])
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    with Progress(output / f'{phase}_status.json', phase=phase, protocol=PROTOCOL,
                  manifest_sha256=m['manifest_sha256']) as hb:
        if phase in ('canary', 'canary-resume'):
            canary(m, output, hb, resumed=phase == 'canary-resume')
        elif phase == 'baseline':
            baseline(m, output, hb)
        elif phase == 'train':
            train(m, output, hb)
        elif phase == 'report':
            report(m, output, hb)


def supervise(m, manifest_path):
    root = Path(m['root'])
    verify_manifest(m, inputs=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    phases = ['canary', 'canary-resume', 'baseline', 'train', 'report']
    attempt = root / 'attempts' / f'{time.strftime("%Y%m%dT%H%M%S")}_{os.getpid()}'
    attempt.mkdir(parents=True)
    child = None
    try:
        for phase in phases:
            proof = root / 'proofs/canary_passed.json'
            if phase.startswith('canary') and proof.exists():
                if not read_json(proof).get('passed') or read_json(proof)['manifest_sha256'] != m['manifest_sha256']:
                    raise ValueError('Invalid GPU/resume proof')
                continue
            if phase == 'canary' and (root / 'proofs/canary_expected.json').exists():
                continue
            output = root / 'proofs' if phase.startswith('canary') else attempt / phase
            output.mkdir(parents=True, exist_ok=True)
            command = [m['python'], '-B', '-u', str(Path(m['source_root']) /
                       'onpolicy/scripts/train/run_stage3_h3_frozen.py'),
                       'phase', str(manifest_path), '--phase', phase, '--output', str(output)]
            with (output / f'{phase}.log').open('a') as log:
                child = subprocess.Popen(command, cwd=m['source_root'], stdout=log, stderr=subprocess.STDOUT)
                started = time.time()
                last_resources, resources = 0., None
                while child.poll() is None:
                    if time.time() - last_resources >= 30:
                        try:
                            resources = resource_snapshot(m, child.pid)
                        except Exception:
                            if child.poll() is not None:
                                break
                            raise
                        with (root / 'resource_samples.jsonl').open('a') as samples:
                            samples.write(json.dumps(dict(phase=phase, **resources)) + '\n')
                        last_resources = time.time()
                    atomic_json(root / 'run_status.json', dict(status='running', phase=phase,
                        child_pid=child.pid, controller_pid=os.getpid(), phase_started_unix=started,
                        updated_unix=time.time(), manifest_sha256=m['manifest_sha256'], output=str(output),
                        solver_queries=0, budget=m['budget'], resources=resources))
                    limit = m['recipe']['train_wall_timeout_seconds'] if phase == 'train' else 21600
                    if time.time() - started > limit:
                        child.terminate()
                        try:
                            child.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
                        raise TimeoutError(f'{phase} exceeded its wall-clock budget')
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'{phase} failed with exit code {child.returncode}; see {output / (phase + ".log")}')
        atomic_json(root / 'run_status.json', dict(status='completed', updated_unix=time.time(),
                    manifest_sha256=m['manifest_sha256'], solver_queries=0))
    except BaseException as exc:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        atomic_json(root / 'run_status.json', dict(status='failed', error=str(exc),
                    updated_unix=time.time(), manifest_sha256=m['manifest_sha256'], solver_queries=0))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--iga-root', required=True, type=Path)
    p.add_argument('--tests', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--arm', choices=['R0', 'R1'], default='R0')
    p = commands.add_parser('run'); p.add_argument('manifest', type=Path)
    p = commands.add_parser('phase'); p.add_argument('manifest', type=Path)
    p.add_argument('--phase', required=True, choices=['canary', 'canary-resume', 'baseline', 'train', 'report'])
    p.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'run':
        supervise(read_json(args.manifest), args.manifest.resolve())
    else:
        run_phase(read_json(args.manifest), args.phase, args.output.resolve())


if __name__ == '__main__':
    main()
