"""Locked, explicitly requested V6 train240/tune60 extension contracts."""
from __future__ import annotations

import copy
import math

from onpolicy.utils.stage2_resource_v6 import ARMS, DEFAULTS

PROTOCOL = 'stage2_semantic_pair_v6_train240_v1'
USER_REQUEST = '启动train240／tune60 全量训练'
CONTRACT = copy.deepcopy(DEFAULTS)
CONTRACT.pop('pilot_train_cases')
CONTRACT.pop('pilot_dev_cases')
CONTRACT.update(training_cases=240, development_cases=60)
RESOURCE = dict(gpu=0, gpu_uuid='GPU-744c1334-98c8-5318-e799-7ad15eea1fbf',
    allowed_cpus='0-31,64-95', train_cpus='0-23,64-87', eval_cpus='24-29,88-93',
    monitor_cpus='30-31,94-95', max_concurrent_trainers=1, actual_trainers=1,
    scheduling='alternate_S1_S2_same_shared_teacher_batch', runtime_max_seconds=None)


def case_batches(rows, size):
    if size <= 0 or not rows or len(rows) % size:
        raise ValueError('Only complete fixed-size case groups are permitted.')
    if len({r['case_dir'] for r in rows}) != len(rows):
        raise ValueError('Duplicate case in the locked order.')
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def execution_contract():
    return dict(arms=list(ARMS), phases=['semantic_reference_audit', 'shared_teacher240', 'full_bc10'],
        total_case_episode_limit=748, logical_case_episode_budget_including_reuse=796,
        teacher_case_episodes=240, reused_teacher_case_episodes=48, new_teacher_case_episodes=192,
        reference_case_episodes=16, b0_development_case_episodes=60,
        candidate_development_case_episodes=480, actor_steps_per_arm=400,
        total_supervised_case_presentations=4800, automatic_retry=False,
        automatic_ppo=False, automatic_stage3=False, automatic_further_training=False,
        search_calls=0, cost_label_queries=0, wall_clock_limit_enabled=False)


def pilot_observation(report):
    if report.get('endpoint') != 10 or set(report.get('results', {})) != set(ARMS):
        raise ValueError('A completed two-arm epoch10 pilot report is required.')
    result = {}
    for arm in ARMS:
        row = report['results'][arm]
        if row['actor_steps'] != 80 or type(row['screen_passed']) is not bool:
            raise ValueError('Incomplete or invalid pilot result.')
        gain = row['comparison']['relative_improvement']
        if not math.isfinite(gain):
            raise ValueError('Invalid pilot Cmax comparison.')
        result[arm] = dict(screen_passed=row['screen_passed'], actor_steps=80,
            relative_improvement_vs_b0=gain,
            first_epoch_training_loss=row['first_epoch_online_training_loss'],
            final_epoch_training_loss=row['final_epoch_online_training_loss'])
    return result


def validate_teacher_parts(parts, records):
    groups = case_batches(records, 6)
    if len(parts) != len(groups):
        raise ValueError('Teacher part coverage differs from the fixed case groups.')
    for part, group in zip(parts, groups):
        if part['cases'] != [r['case_dir'] for r in group]:
            raise ValueError('Teacher case order or six-case membership changed.')
        for field in ('sha256', 'stateless_sha256'):
            digest = part[field]
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Missing or invalid teacher artifact digest.')


def validate_plan(manifest, pilot, report):
    if (manifest['protocol'] != PROTOCOL or manifest['training_contract'] != CONTRACT
            or manifest['execution'] != execution_contract() or manifest['resource_contract'] != RESOURCE):
        raise ValueError('Full-data training, resource or episode contract changed.')
    authority = manifest.get('authorization', {})
    if (authority.get('user_request') != USER_REQUEST
            or authority.get('mode') != 'explicit_manual_exploratory_extension'
            or authority.get('automatic_promotion') is not False):
        raise ValueError('Explicit manual full-data launch authorization is required.')
    if manifest['pilot_observation'] != pilot_observation(report):
        raise ValueError('Pilot failures or results must not be relabelled.')
    for field in ('train', 'evaluation', 'source', 'stage1_source', 'ready_source',
                  'teacher_index', 'teacher_index_sha256', 'teacher_files',
                  'environment_argv', 'archived_teachers', 'diagnostic_cases', 'iga_reference'):
        if manifest[field] != pilot[field]:
            raise ValueError(f'Historical lineage or data contract changed: {field}')
    train, dev = manifest['train']['cases'], manifest['evaluation']['cases']
    if len(train) != 240 or len(dev) != 60:
        raise ValueError('Exactly train240/tune60 are authorized, not train600.')
    if {r['case_sha256'] for r in train} & {r['case_sha256'] for r in dev}:
        raise ValueError('Training and development contents overlap.')
    case_batches(train, 6)
    case_batches(dev, 12)
    if pilot['pilot_train'] != train[:48]:
        raise ValueError('Reused pilot cache is not the first48 of this training order.')
    reuse = manifest['teacher_reuse']
    if (reuse.get('case_episodes') != 48 or reuse.get('read_only') is not True
            or reuse.get('counted_as_new_environment_episodes') is not False):
        raise ValueError('Teacher cache reuse must be read-only and separately accounted.')
    validate_teacher_parts(manifest['teacher_reuse']['parts'], train[:48])
    if (manifest['data_roles'] != dict(train_cases=240, development_cases=60,
            historical_case_order_preserved=True, development_only=True,
            gate120_accessed=False, finalblind60_accessed=False, selection_uses_outcomes=False)
            or manifest['screen'] != (pilot['screen'] | {'minimum_gradient_steps': 400,
                'scientific_goal_certified_by_full_run': False})):
        raise ValueError('Development-only interpretation or review criteria changed.')


def checkpoint_progress(actor_steps):
    if not 0 <= actor_steps <= 400:
        raise ValueError('Actor update count exceeds this fixed experiment.')
    return dict(completed_bc_epochs=actor_steps // 40, updates_in_current_epoch=actor_steps % 40,
                total_planned_actor_steps=400)


def assert_replayed_subset(previous, current):
    lookup = {r['case_sha256']: r for r in current}
    if len(lookup) != len(current):
        raise ValueError('Duplicate current baseline hashes.')
    deltas = {}
    for old in previous:
        new = lookup.get(old['case_sha256'])
        if (new is None or new['case_dir'] != old['case_dir'] or new['steps'] != old['steps']
                or abs(new['makespan'] - old['makespan']) > 1e-5):
            raise RuntimeError('Full-data evaluation changed the pilot B0 replay.')
        deltas[old['case_dir']] = new['makespan'] - old['makespan']
    return dict(passed=True, checked_cases=len(previous), cmax_tolerance=1e-5,
                equal_event_steps=True, cmax_deltas=deltas)
