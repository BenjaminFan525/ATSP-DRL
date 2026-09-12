"""Explicit frozen-policy H/F experiments; never a training/handoff override."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

PROTOCOL = 'stage2_bc_hf_sensitivity_v1'
GRID = tuple(f'H{h}_F{f}' for h in (1, 2, 3) for f in (1, 2, 3, 4))
ANCHOR = 'H2_F4'
TRAIN_ARMS = ('H2_F4', 'H2_F2', 'H1_F2', 'H3_F4')
GPU_UUID = 'GPU-744c1334-98c8-5318-e799-7ad15eea1fbf'
ALLOWED_CPUS = '0-31,64-95'
SLOTS = ('0-14,64-78', '15-29,79-93')
CAPACITY = 5
WORKERS = 12
SOURCE_SHA = 'b7740b2b688c83dc7fa65bed6381243cdf2f6675f4b96808ff05a41d4e44e7e8'
PARENT = 'result/hkbz_train_logs/stage2_bc_rl_v4_20260908_gpu0_half_train240_r1/manifest.json'
PLAN = 'STAGE2_BC_HF_RESEARCH_PLAN_20260912.md'
ROLES = ('ordinary', 'transporter')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def parse_arm(arm):
    if arm not in GRID:
        raise ValueError(f'H/F arm outside the declared grid: {arm}')
    h, f = re.fullmatch(r'H([1-3])_F([1-4])', arm).groups()
    return int(h), int(f)


def planning_for(source, arm):
    h, f = parse_arm(arm)
    if (source.get('device_future_intent_horizon') != 2
            or source.get('device_frontier_max_requests') != 4
            or source.get('device_future_intent_mode') != 'bounded_frontier'
            or source.get('device_lookahead_reservation_mode') != 'soft'):
        raise ValueError('Expected the original H2/F4 soft planning source.')
    result = copy.deepcopy(source)
    result.update(device_future_intent_horizon=h, device_frontier_max_requests=f)
    return result


def configure_args(args, arm, *, native=False):
    """Only the evaluation environment changes; original weights stay untouched."""
    h, f = parse_arm(arm)
    if native and arm != ANCHOR:
        raise ValueError('Native-capacity replay exists only for H2/F4.')
    args.device_future_intent_horizon = h
    args.device_frontier_max_requests = f
    args.device_request_capacity_per_plane = 0 if native else CAPACITY
    args.device_global_matching = True
    args.request_ready_policy_injection = 'none'
    args.stage2_resource_v6_observations = False
    # Inference never consults teacher files, even if inherited from a trainer.
    for name in ('resource_iga_teacher_dir', 'resource_iga_teacher_index',
                 'joint_iga_teacher_dir', 'joint_iga_teacher_index',
                 'plane_bc_teacher_dir', 'plane_bc_teacher_index'):
        setattr(args, name, '')
    return args


def evaluation_jobs():
    jobs = [dict(label='B0_native_H2_F4', arm=ANCHOR, native=True),
            dict(label=ANCHOR, arm=ANCHOR, native=False)]
    jobs.extend(dict(label=a, arm=a, native=False) for a in GRID if a != ANCHOR)
    jobs.append(dict(label='H2_F4_repeat', arm=ANCHOR, native=False))
    return jobs


def validate_manifest(manifest):
    if (manifest.get('protocol') != PROTOCOL or manifest.get('jobs') != evaluation_jobs()
            or manifest.get('train_anchors') != list(TRAIN_ARMS)
            or manifest.get('source', {}).get('sha256') != SOURCE_SHA):
        raise ValueError('Unapproved source, grid, order or training-anchor change.')
    c = manifest['execution']
    expected = dict(evaluation_workers=WORKERS, fixed_request_capacity=CAPACITY,
        max_parallel_evaluators=2, total_case_episode_limit=840, evaluation_jobs=14,
        actor_updates=0, critic_updates=0, teacher_queries=0, search_calls=0,
        automatic_training=False, automatic_retry=False, automatic_stage3=False,
        rollout_max_steps=4000, runtime_max_seconds=None)
    if c != expected:
        raise ValueError('Frozen sensitivity execution contract changed.')
    r = manifest['resource_contract']
    if (r != dict(gpu=0, gpu_uuid=GPU_UUID, allowed_cpus=ALLOWED_CPUS,
            slots=list(SLOTS), monitor_cpus='30-31,94-95', cuda_memory_fraction=.40,
            blas_threads=1, heartbeat_seconds=30)):
        raise ValueError('GPU0/original CPU half or concurrency contract changed.')
    rows = manifest['evaluation']['cases']
    if (len(rows) != 60 or len({r['case_sha256'] for r in rows}) != 60
            or dict(Counter(r['distribution'] for r in rows))
            != {'iid': 30, 'ood_stress': 27, 'ood_scale': 3}):
        raise ValueError('Expected the original full tune60 composition.')
    training = manifest['train']['cases']
    if (len(training) != 240 or len({r['case_sha256'] for r in training}) != 240
            or dict(Counter(r['distribution'] for r in training))
            != {'iid': 120, 'ood_stress': 108, 'ood_scale': 12}) or (
            {r['case_sha256'] for r in rows}
            & {r['case_sha256'] for r in manifest['train']['cases']}):
        raise ValueError('Training alignment/overlap failure.')
    if manifest['evaluation'].get('independent') is not False:
        raise ValueError('Previously used tune60 is not independent confirmation.')
    planning_for(manifest['source_planning_contract'], ANCHOR)


def current_cases(rows, records):
    """Only copy identifying fields, never old runtime diagnostics."""
    lookup = {r['case_dir']: r for r in records}
    if len(lookup) != len(records) or len(rows) != len(records):
        raise ValueError('Missing/duplicate case coverage.')
    if len({r['case_dir'] for r in rows}) != len(rows) or {r['case_dir'] for r in rows} != set(lookup):
        raise ValueError('Different evaluated cases.')
    if any(r.get('case_sha256', lookup[r['case_dir']]['case_sha256'])
           != lookup[r['case_dir']]['case_sha256'] for r in rows):
        raise ValueError('Runtime case fingerprint differs from the manifest.')
    return [{**{k: lookup[r['case_dir']][k] for k in
                ('case_id', 'case_dir', 'case_sha256', 'distribution', 'profile')
                if k in lookup[r['case_dir']]}, **r} for r in rows]


def check_replay(expected, actual):
    def index(rows):
        indexed = {r['case_sha256']: r for r in rows}
        if len(indexed) != len(rows) or not rows:
            raise ValueError('Empty/duplicate replay cases.')
        for r in rows:
            if not r['completed'] or r['timeout'] or r['cycle_terminated'] or not math.isfinite(r['makespan']):
                raise ValueError('Invalid/incomplete replay cases.')
        return indexed
    left, right = index(expected), index(actual)
    if left.keys() != right.keys():
        raise ValueError('Replay case hashes differ.')
    errors = []
    for key in left:
        a, b = left[key], right[key]
        old_steps = a.get('steps', a.get('finish_steps'))
        if old_steps is None or b.get('steps') is None:
            raise ValueError('Missing replay completion-step evidence.')
        if a['makespan'] != b['makespan'] or old_steps != b['steps']:
            errors.append(dict(case_sha256=key, expected=a['makespan'], actual=b['makespan'],
                               expected_steps=old_steps, actual_steps=b['steps']))
    return dict(passed=not errors, cases=len(left), tolerance=0., errors=errors)


def paired_analysis(reference, candidate):
    import numpy as np
    from onpolicy.scripts.train.run_stage2_resource_rl import compare, metrics
    comparison = compare(reference, candidate)
    ref = {r['case_sha256']: r for r in reference}
    cur = {r['case_sha256']: r for r in candidate}
    differences = np.array([cur[k]['makespan'] - ref[k]['makespan'] for k in sorted(ref)])
    indices = np.random.default_rng(20260912).integers(0, len(ref), (20000, len(ref)))
    comparison.update(case_bootstrap_95ci=np.quantile(differences[indices].mean(1), [.025, .975]).tolist(),
        uncertainty_scope='cases conditional on one frozen model; no selection/multiplicity correction')
    return dict(summary=metrics(candidate), versus_H2_F4=comparison,
                training_gain=False, independent_confirmation=False)


def report_suite(suite):
    suite = Path(suite)
    reference = json.loads((suite / 'evaluations/H2_F4.json').read_text())
    records = {}
    for arm in GRID:
        result = json.loads((suite / 'evaluations' / (arm + '.json')).read_text())
        records[arm] = paired_analysis(reference['cases'], result['cases'])
        records[arm].update(wall_seconds=result['seconds'], diagnostics=result['diagnostics'],
                            peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes'])
    ranked = sorted(GRID, key=lambda a: (records[a]['summary']['mean'],
                    records[a]['summary']['tail10'], *parse_arm(a)))
    return dict(protocol=PROTOCOL, complete=True, development_only=True,
        result_type='frozen_B0_HF_sensitivity_not_retrained_BC', arms=records,
        observed_deployment_ranking=ranked, predeclared_training_anchors=list(TRAIN_ARMS),
        anchors_selected_from_this_ranking=False, confirmed_scientific_success=False,
        automatic_training=False, automatic_stage3=False,
        next_phase='train_only_teacher_compatibility_and_cost_audit_then_independent_BC',
        limitations=['one H2/F4-trained source', 'tune60 repeatedly used',
                     '11 unadjusted exploratory contrasts', 'OOD-scale only 3 cases',
                     'does not identify independently retrained BC optimum'])
