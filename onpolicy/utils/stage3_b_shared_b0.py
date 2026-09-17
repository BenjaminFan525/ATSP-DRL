"""Portable contracts for the single-GPU, fully shared B0 study.

No old C0 manifest, CUDA initialization, or launch side effects on import.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import copy
import math
import os

from onpolicy.utils.stage3_research import (
    atomic_json, read_json, digest_file, digest_json, case_record, verify_cases,
    paired_summary, safety_gate,
)
from onpolicy.utils.stage3_full_data import schedule as full_data_schedule

PROTOCOL = 'stage3_b_shared_b0_native_v1'
SOURCE_SHA = 'b7740b2b688c83dc7fa65bed6381243cdf2f6675f4b96808ff05a41d4e44e7e8'
HISTORY = 'b0_authoritative_native_v1'
WORKSPACE = Path(os.environ.get('HKBZ_STAGE3_WORKSPACE_ROOT', Path(__file__).resolve().parents[2])).resolve()
FROZEN = WORKSPACE / 'artifacts/stage2_frozen/20260912/manifest.json'
CONFIG = Path(__file__).resolve().parents[1] / 'config/env_stage3_b_shared_b0.yaml'
SPLITS = {
    'train': 'fjsp_v3_t600_v120_test60/train',
    'validation': 'fjsp_v3_stage1_v2_eval_s20260803/validation',
    'tune': 'fjsp_v3_resource_joint_eval_s20260811/joint/tune',
}
COUNTS = {'train': {'iid': 480, 'ood_stress': 108, 'ood_scale': 12},
          'validation': {'iid': 60, 'ood_stress': 54, 'ood_scale': 6},
          'tune': {'iid': 30, 'ood_stress': 27, 'ood_scale': 3}}


def cpus(spec):
    values = set()
    for part in str(spec).split(','):
        lo, sep, hi = part.partition('-')
        values.update(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return values


def recipe():
    import yaml
    value = yaml.safe_load(CONFIG.read_text())
    if (value['protocol'] != PROTOCOL or value['arm'] != 'B_SHARED'
            or value['gpu'] != 0 or value['cpus'] != '0-31,64-95'
            or value['validation_workers'] != 12
            or value['evaluation_batch_contract'] != 'b0_fixed12_keep_completed_slots_v1'
            or value['evaluation_cudnn_allow_tf32'] is not True
            or (value['epochs'], value['ppo_epochs']) != (8, 2)
            or value['global_batch'] not in (32,64,128,240,256,512)
            or value['history'] != HISTORY or value['primary_decoder'] != 'hungarian'):
        raise ValueError('B0 single-arm budget/resource/semantics contract changed')
    return value


def schedule(cases, seed, batch_size=None):
    return full_data_schedule(cases,seed,batch_size=batch_size or recipe()['global_batch'],
                              allow_variable_batch=True)


def batch_sizes(m):
    """Keep committed groups intact while rebatching only unvisited examples."""
    if m.get('training_subset'):
        from onpolicy.utils.stage3_b0_subset import subset_schedule
        return [len(row['cases']) for row in subset_schedule(m)]
    size = m.get('recipe', {}).get('global_batch', 32)
    if size not in (32, 64, 128, 240, 256, 512):
        raise ValueError('Unregistered global batch')
    amendment = m.get('resume_amendment', {})
    sizes = list(amendment.get('completed_batch_sizes', []))
    episodes = 0
    for count in sizes:
        if type(count) is not int or not 0 < count <= 512 or count > 960 - episodes % 960:
            raise ValueError('Committed batch sizes cross an epoch boundary')
        episodes += count
    if episodes > 7680 or (amendment and (
            episodes != amendment['completed_visits']
            or len(sizes) != amendment['completed_batches']
            or 2 * len(sizes) != amendment['completed_actor_updates'])):
        raise ValueError('Committed continuation cursor is inconsistent')
    while episodes < 7680:
        count = min(size, 960 - episodes % 960)
        sizes.append(count)
        episodes += count
    return sizes


def training_schedule(m):
    if m.get('training_subset'):
        from onpolicy.utils.stage3_b0_subset import subset_schedule
        return subset_schedule(m)
    if not m.get('resume_amendment'):
        return schedule(m['splits']['train'], m['recipe']['seed'], m['recipe']['global_batch'])
    canonical = schedule(m['splits']['train'], m['recipe']['seed'], 32)
    cases = [c for row in canonical for c in row['cases']]
    seeds = [s for row in canonical for s in row['seeds']]
    visits = [v for row in canonical for v in row['visit_ids']]
    result, start = [], 0
    for count in batch_sizes(m):
        stop = start + count
        epoch = start // 960
        result.append(dict(cases=cases[start:stop], seeds=seeds[start:stop],
            visit_ids=visits[start:stop], visit=epoch, data_epoch=epoch + 1,
            group=len(result) + 1, training_episodes=stop))
        start = stop
    return result


def epoch_episodes(m, epoch):
    subset = m.get('training_subset')
    return subset['inherited_visits'] + epoch*subset['epoch_visits'] if subset else epoch*960


def budget(m):
    size=m.get('recipe',{}).get('global_batch',32)
    if m.get('training_subset'):
        a = m['training_subset']
        plan = training_schedule(m)
        ends = {epoch_episodes(m, ep) for ep in range(1, a['epochs']+1)}
        return dict(global_batch=size, batches_per_epoch=(a['epoch_visits']+size-1)//size,
            global_batches=len(plan), actor_updates=2*len(plan), training_episodes=plan[-1]['training_episodes'],
            epoch_end_batches=[i for i, row in enumerate(plan, 1) if row['training_episodes'] in ends],
            training_cases=len(m['splits']['train']), epoch_visits=a['epoch_visits'],
            inherited_training_episodes=a['inherited_visits'], subset_training_episodes=a['new_training_visits'])
    per_epoch=(960+size-1)//size
    sizes = batch_sizes(m)
    episodes, boundaries = 0, []
    for index, count in enumerate(sizes, 1):
        episodes += count
        if episodes % 960 == 0:
            boundaries.append(index)
    return dict(global_batch=size,batches_per_epoch=per_epoch,global_batches=len(sizes),
                actor_updates=2*len(sizes),training_episodes=7680,epoch_end_batches=boundaries)


def episodes_at_cursor(m,cursor):
    b=budget(m)
    if not isinstance(cursor,int) or not 0<=cursor<=b['global_batches']:
        raise ValueError('Cursor outside the registered coverage budget')
    return sum(batch_sizes(m)[:cursor])


def physical_hash(record):
    # Deliberately excludes names, paths, metadata seed and profile labels.
    return digest_json({name: read_json(Path(record['path']) / name)
                        for name in ('job.json', 'fixed_resources.json', 'mobile_resources.json',
                                     'sites.json', 'flights.json')})


def manifest_identity(value):
    return digest_json({k: v for k, v in value.items() if k != 'manifest_sha256'})


def verify_manifest(m, *, inputs=False, source_root=None):
    if (m['protocol'] != PROTOCOL or m['manifest_sha256'] != manifest_identity(m)
            or m['recipe'] != recipe() or m['source']['sha256'] != SOURCE_SHA):
        raise ValueError('Invalid B0 study manifest')
    if digest_file(m['source']['path']) != SOURCE_SHA:
        raise ValueError('Frozen B0 checkpoint changed')
    if digest_file(m['frozen_manifest']['path']) != m['frozen_manifest']['sha256']:
        raise ValueError('Frozen Stage2 manifest changed')
    root = Path(source_root or m['source_root'])
    for relative, checksum in m['code_files'].items():
        if digest_file(root / relative) != checksum:
            raise ValueError(f'Frozen study code changed: {relative}')
    if m.get('training_subset'):
        from onpolicy.utils.stage3_b0_subset import verify_subset
        verify_subset(m)
    if inputs:
        for name, records in m['splits'].items():
            verify_cases(records, training=name == 'train')
        if digest_json(training_schedule(m)) != m['schedule_sha256']:
            raise ValueError('Visit schedule changed')
        import importlib.metadata
        for name, version in m['packages'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f'Runtime package changed: {name}')


def check_resources(m, role=None, *, cuda=True):
    r = m['recipe']
    allowed = cpus(r[f'{role}_cpus'] if role else r['cpus'])
    if not set(os.sched_getaffinity(0)).issubset(allowed):
        raise ValueError(f'CPU affinity escaped the registered {role or "study"} allocation')
    # UUID pinning prevents CUDA index order changes from touching GPU1.
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', r['gpu_uuid']):
        raise ValueError('Only physical GPU0 is authorized')
    if cuda:
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one CUDA device must be visible')
        actual = str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        if actual != r['gpu_uuid'].removeprefix('GPU-'):
            raise ValueError('Visible CUDA device is not the admitted GPU0 UUID')


def record_summary(trajectory):
    keys = ('case_id', 'profile', 'distribution', 'seed', 'makespan', 'completed', 'steps',
            'cycle_terminated', 'policy_updates', 'behavior_deterministic', 'forced_replay',
            'history', 'decoder', 'tau', 'visit_id', 'actions_sha256', 'history_sha256',
            'sampling_architecture', 'sampling_rng_contract')
    result = {key: trajectory[key] for key in keys if key in trajectory}
    if 'actions' in trajectory:
        result['actions_sha256'] = digest_json(trajectory['actions'])
    return result


def risk_pass(summary):
    return safety_gate(summary) and summary['regression_over_5pct_fraction'] <= .05


def choose_candidate(curves):
    if [r['epoch'] for r in curves] != list(range(1, 9)):
        raise ValueError('Selection requires all eight complete epoch evaluations')
    eligible = [r for r in curves if risk_pass(r['validation']) and risk_pass(r['tune'])]
    selected = min(eligible, key=lambda r: (r['validation']['makespan'], r['epoch'])) if eligible else curves[-1]
    stable = all(r['validation']['gain_fraction'] > 0 and risk_pass(r['validation'])
                 and risk_pass(r['tune']) for r in curves[-2:])
    return {'selected': copy.deepcopy(selected), 'risk_qualified': bool(eligible),
            'open_confirmation': bool(eligible and selected['validation']['gain_fraction'] >= .02),
            'last_two_epochs_stable': stable,
            'rule': 'lowest validation mean among risk-qualified epochs; ties earlier; no confirmation reselection'}


def request_identity(request):
    return digest_json({k: v for k, v in request.items() if k != 'request_sha256'})


def submit_evaluation(m, checkpoint, episodes, split, *, decoder='H', baseline=False):
    if split not in ('validation', 'tune', 'confirmation') or decoder not in ('H', 'AR'):
        raise ValueError('Unregistered evaluation')
    checkpoint = Path(checkpoint).resolve()
    source_sha = digest_file(checkpoint)
    if baseline and source_sha != SOURCE_SHA:
        raise ValueError('Baseline evaluation must load B0')
    if split == 'confirmation':
        lock = read_json(Path(m['root']) / 'selection_locked.json')
        if (not lock['open_confirmation'] or lock['manifest_sha256'] != m['manifest_sha256']
                or (not baseline and source_sha != lock['selected']['checkpoint_sha256'])):
            raise ValueError('Confirmation is restricted to B0 and the locked candidate')
    label = 'B0' if baseline else f'e{episodes:06d}'
    ident = f'{label}_{decoder}_{split}'
    request = {'request_id': ident, 'manifest_sha256': m['manifest_sha256'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': source_sha,
        'episodes': episodes, 'split': split, 'decoder': decoder, 'baseline': baseline,
        'cases_sha256': digest_json(m['splits'][split]), 'tau': .3, 'seed': 1, 'history': HISTORY}
    request['request_sha256'] = request_identity(request)
    path = Path(m['root']) / 'validator/requests' / f'{ident}.json'
    if path.exists():
        if read_json(path) != request:
            raise ValueError('Evaluation request ID reused with different content')
    else:
        atomic_json(path, request, overwrite=False)
    return ident


def validate_request(m, request):
    if (request['manifest_sha256'] != m['manifest_sha256']
            or request_identity(request) != request['request_sha256']
            or request['tau'] != .3 or request['seed'] != 1 or request['history'] != HISTORY
            or request['decoder'] not in ('H', 'AR')
            or digest_file(request['checkpoint']) != request['checkpoint_sha256']
            or digest_json(m['splits'][request['split']]) != request['cases_sha256']):
        raise ValueError('Invalid evaluation request identity')
    if request['baseline'] and request['checkpoint_sha256'] != SOURCE_SHA:
        raise ValueError('Invalid B0 control')
    if request['split'] == 'confirmation':
        lock = read_json(Path(m['root']) / 'selection_locked.json')
        if (not lock['open_confirmation'] or lock['manifest_sha256'] != m['manifest_sha256']
                or (not request['baseline'] and request['checkpoint_sha256'] != lock['selected']['checkpoint_sha256'])):
            raise ValueError('Unselected checkpoint attempted confirmation')


def pending_evaluations(root):
    root = Path(root) / 'validator'
    return [p for p in sorted((root / 'requests').glob('*.json'))
            if not (root / 'results' / p.name).exists()]


def read_commit(m, path):
    commit = read_json(path)
    if (commit['manifest_sha256'] != m['manifest_sha256']
            or not isinstance(commit['next_batch'], int) or not 1 <= commit['next_batch'] <= budget(m)['global_batches']
            or episodes_at_cursor(m,commit['next_batch']) != commit['training_episodes']
            or commit['actor_updates'] != commit['next_batch'] * 2
            or commit['schedule_sha256'] != m['schedule_sha256']
            or digest_file(commit['checkpoint']) != commit['checkpoint_sha256']
            or digest_file(commit['update']) != commit['update_sha256']):
        raise ValueError('Global commit is incomplete or changed')
    return commit


def reconcile_epoch_requests(m, cursor):
    """Recover every post-commit publication, including auxiliary AR endpoints."""
    ids = []
    for ep, boundary in enumerate(budget(m)['epoch_end_batches'], 1):
        if boundary > cursor:
            break
        commit = read_commit(m, Path(m['root']) / 'commits' / f'batch_{boundary:04d}.json')
        for split in ('validation', 'tune'):
            ids.append(submit_evaluation(m, commit['checkpoint'], epoch_episodes(m, ep), split))
        if ep in (4, 8):
            ids.append(submit_evaluation(m, commit['checkpoint'], epoch_episodes(m, ep), 'validation', decoder='AR'))
    return ids


def verify_admission(m):
    root = Path(m['root'])
    a = read_json(root / 'training_admission.json')
    if (not a['passed'] or a['manifest_sha256'] != m['manifest_sha256']
            or a['global_batch'] != m['recipe']['global_batch']
            or a['physical_microbatch'] != m['recipe']['microbatch'] or a['gpu_count'] != 1
            or a['baseline_sha256'] != digest_file(root / 'baselines.json')
            or a['zero_check_sha256'] != digest_file(root / 'zero_update_verified.json')
            or a['tests'] != m['tests'] or digest_file(m['tests']['path']) != m['tests']['sha256']):
        raise ValueError('Formal admission identity changed')
    if m.get('resume_amendment'):
        from onpolicy.utils.stage3_b0_restart import verify_restart_admission
        verify_restart_admission(m, a)
        return a
    for name in ('canary', 'resume_check'):
        if digest_file(a[name]) != a[name + '_sha256']:
            raise ValueError(f'Admission evidence changed: {name}')
        proof = read_json(a[name])
        if not proof['passed'] or proof['manifest_sha256'] != m['manifest_sha256']:
            raise ValueError(f'Admission evidence failed: {name}')
    return a


def normalization_moments(trajectories):
    sums = [0., 0., 0.]
    for t in trajectories:
        weight = 1 if t['distribution'] == 'iid' else 4
        for time in t['times']:
            value = -.01 * (t['makespan'] - time)
            if not math.isfinite(value):
                raise ValueError('Nonfinite training-only normalization target')
            sums[0] += weight
            sums[1] += weight * value
            sums[2] += weight * value * value
    return sums
