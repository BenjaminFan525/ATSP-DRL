"""H3 continuation contracts. IGA is an immutable input, never a solver task."""
from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path

import numpy as np

from onpolicy.utils.stage3_b_shared_b0 import HISTORY, SOURCE_SHA, cpus
from onpolicy.utils.stage3_research import (
    atomic_json, digest_file, digest_json, read_json, trajectory_seed, verify_cases,
)

PROTOCOL = 'stage3_b_shared_h3_frozen_iga_v1'
COUNTS = {'iid': 192, 'ood_stress': 43, 'ood_scale': 5}
# Registered alternative to the two post-pass full replays: the pass gate is
# derived from the applied minibatch pre-step records instead of re-replaying
# all trajectories. It never affects the gradient path or the collection.
POST_PASS_REPLAY_SKIP = 'skip_minibatch_derived_v1'
PLANNING = {
    'device_lookahead_dispatch': True,
    'device_lookahead_safety_margin': 60.0,
    'device_deadline_aware_dispatch': True,
    'device_future_intent_horizon': 3,
    'device_future_intent_mode': 'bounded_frontier',
    'device_frontier_max_requests': 4,
    'device_request_capacity_per_plane': 5,
    'resource_release_aware_eta': True,
    'device_lookahead_reservation_mode': 'soft',
    'device_reservation_grace_seconds': 300.0,
    'device_departure_lookahead': True,
    'resource_slack_forecast_seconds': 0.0,
}


class MissingFrozenTeacher(ValueError):
    pass


def recipe(previous):
    r = copy.deepcopy(previous)
    r.update(protocol=PROTOCOL, arm='R0', epochs=8, seed=2026091502,
             global_batch=384, optimizer_minibatch=64, microbatch=64,
             physical_microbatch_candidates=[64, 32], ppo_epochs=2,
             rollout_workers=240, environment_processes=64,
             encoder_activation_checkpoint=True, input_cache_mib=4096,
             memory_high_gib=102, memory_max_gib=114,
             planning=copy.deepcopy(PLANNING), history=HISTORY, iga_mode='frozen_only',
             teacher_queries_allowed=0, train_wall_timeout_seconds=72 * 3600,
             evaluation_epochs=[1, 2, 4, 6, 8],
             initialization_mode='weights_only_fresh_optimizers',
             checkpoint_boundary='complete_logical_rollout',
             numerical_equivalence='waived_by_user_not_claimed',
             optimizer_shuffle_seed=2026091506)
    validate_recipe(r)
    return r


def validate_recipe(r):
    if (r['protocol'] != PROTOCOL or r['planning'] != PLANNING
            or r['iga_mode'] != 'frozen_only' or r['teacher_queries_allowed'] != 0
            or r['history'] != HISTORY or r['arm'] not in ('R0', 'R1')):
        raise ValueError('H3/frozen-IGA protocol changed')
    if (r['global_batch'], r['rollout_workers'], r['environment_processes'],
            r['optimizer_minibatch'], r['ppo_epochs'], r['epochs']) != (384, 240, 64, 64, 2, 8):
        raise ValueError('Unregistered rollout/optimizer budget')
    if (r['microbatch'] not in (32, 64) or r['physical_microbatch_candidates'] != [64, 32]
            or r['optimizer_minibatch'] % r['microbatch']):
        raise ValueError('Physical microbatch must divide the optimizer minibatch')
    if (r['gpu'] != 0 or cpus(r['cpus']) != set(range(32)) | set(range(64, 96))
            or r['validation_workers'] != 12 or r['train_tau'] != .03
            or r['evaluation_tau'] != .3 or r['gradient_clip'] != 1.
            or r['clip'] != .2 or r['actor_lr'] != 1e-5):
        raise ValueError('Previous resource/deployment/PPO settings changed')


def schedule(cases, r):
    validate_recipe(r)
    if (len(cases) != 240 or Counter(c['distribution'] for c in cases) != COUNTS
            or len({c['content_sha256'] for c in cases}) != 240):
        raise ValueError('Expected the fixed unique Train240')
    rows = []
    for epoch in range(r['epochs']):
        visits = []
        for c in cases:
            for replica in range(1 if c['distribution'] == 'iid' else 4):
                seed = trajectory_seed(r['seed'], c['content_sha256'], epoch, replica)
                identity = digest_json([PROTOCOL, c['content_sha256'], epoch, replica, seed])
                visits.append((identity, c, seed))
        visits.sort(key=lambda x: digest_json([r['seed'], 'order', x[0]]))
        if len(visits) != 384 or len({x[0] for x in visits}) != 384:
            raise ValueError('Visit coverage changed')
        rows.append(dict(data_epoch=epoch + 1, cases=[x[1] for x in visits],
                         seeds=[x[2] for x in visits], visit_ids=[x[0] for x in visits],
                         training_episodes=384 * (epoch + 1)))
    return rows


def optimizer_orders(n, minibatch, passes, seed):
    if n <= 0 or minibatch <= 0 or n % minibatch or passes <= 0:
        raise ValueError('Optimizer minibatches must partition complete trajectories')
    rng = np.random.default_rng(seed)
    return [[order[i:i + minibatch] for i in range(0, n, minibatch)]
            for order in (rng.permutation(n).tolist() for _ in range(passes))]


def budget(r):
    validate_recipe(r)
    steps = r['global_batch'] // r['optimizer_minibatch'] * r['ppo_epochs']
    return dict(unique_train_cases=240, visits_per_epoch=384, epochs=8,
                new_training_visits=3072, optimizer_steps_per_epoch=steps,
                max_new_ppo_steps=steps * 8, environment_slots=240,
                environment_cpu_workers=64, bc_steps=0 if r['arm'] == 'R0' else None)


def bind(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': digest_file(p)}


def checked(record):
    if digest_file(record['path']) != record['sha256']:
        raise ValueError(f'Frozen input changed: {record["path"]}')
    return Path(record['path'])


def freeze_iga(iga_root, splits):
    """Index only existing results with matching physical case and H3 contract.

    A missing split remains missing. No extrapolation across cases or horizons,
    no labeling command and no call into a search implementation is permitted.
    """
    root = Path(iga_root).resolve()
    result = dict(mode='frozen_only', solver_queries=0, root=str(root), files=[],
                  references={name: {} for name in splits}, missing={},
                  teacher_scope='stage3_full_joint_policy', planning=PLANNING)
    for tier in ('iga180', 'iga1800'):
        summary = root / tier / 'summary.json'
        if not summary.is_file():
            raise MissingFrozenTeacher(f'Missing frozen IGA summary: {summary}')
        result['files'].append(bind(summary))
        for path in sorted((root / tier / 'teachers').glob('case_*.json')):
            teacher = read_json(path)
            if (teacher.get('teacher_scope') != 'stage3_full_joint_policy'
                    or teacher.get('teacher_method') != 'joint_iga_all'
                    or not teacher.get('completed') or not teacher.get('completion_verified')
                    or any(teacher.get(k) != v for k, v in PLANNING.items())):
                raise ValueError(f'Frozen H3 IGA contract mismatch: {path}')
            cost = float(teacher['makespan'])
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError('Frozen IGA cost must be finite and positive')
            matched = [(name, c) for name, cases in splits.items() for c in cases
                       if c['case_sha256'] == teacher['case_sha256']]
            if len(matched) > 1:
                raise ValueError('Frozen IGA physical case overlaps splits')
            if not matched:
                continue
            split, case = matched[0]
            record = bind(path)
            result['files'].append(record)
            tiers = result['references'][split].setdefault(case['path'], {})
            if tier in tiers:
                raise ValueError('Duplicate frozen reference for physical case')
            tiers[tier] = dict(makespan=cost, teacher=record,
                               case_sha256=case['case_sha256'],
                               content_sha256=case['content_sha256'])
    for split, cases in splits.items():
        result['missing'][split] = [c['path'] for c in cases
                                   if 'iga1800' not in result['references'][split].get(c['path'], {})]
    if result['missing']['tune']:
        raise MissingFrozenTeacher('Frozen H3 Tune60 reference is incomplete')
    result['identity_sha256'] = digest_json({k: v for k, v in result.items() if k != 'identity_sha256'})
    return result


def require_training_teachers(index, cases):
    refs = index['references'].get('train', {})
    missing = [c['path'] for c in cases if 'iga1800' not in refs.get(c['path'], {})]
    if missing:
        raise MissingFrozenTeacher(f'R1 requires {len(missing)} missing frozen H3 training teachers; '
                                   'IGA solving and Tune/Validation label substitution are forbidden')
    return [refs[c['path']]['iga1800'] for c in cases]


def manifest_identity(m):
    return digest_json({k: v for k, v in m.items() if k != 'manifest_sha256'})


def verify_manifest(m, *, inputs=False):
    validate_recipe(m['recipe'])
    if m['protocol'] != PROTOCOL or m['manifest_sha256'] != manifest_identity(m):
        raise ValueError('H3 manifest identity mismatch')
    checked(m['b0']); checked(m['initialization']); checked(m['frozen_manifest'])
    if m['b0']['sha256'] != SOURCE_SHA:
        raise ValueError('Frozen B0 identity changed')
    index = read_json(checked(m['frozen_iga']))
    if index['mode'] != 'frozen_only' or index['solver_queries'] != 0:
        raise ValueError('Frozen-only IGA constraint changed')
    for record in index['files']:
        checked(record)
    if m['recipe']['arm'] == 'R1':
        require_training_teachers(index, m['splits']['train'])
    source = Path(m['source_root'])
    for name, checksum in m['code_files'].items():
        if digest_file(source / name) != checksum:
            raise ValueError(f'Snapshot source changed: {name}')
    if digest_json(schedule(m['splits']['train'], m['recipe'])) != m['schedule_sha256']:
        raise ValueError('Frozen visit schedule changed')
    if inputs:
        for split, cases in m['splits'].items():
            verify_cases(cases, training=split == 'train')
        for a, cases in m['splits'].items():
            for b, other in m['splits'].items():
                if a < b and {c['content_sha256'] for c in cases} & {c['content_sha256'] for c in other}:
                    raise ValueError('Physical split overlap')


def paired_metrics(rows, references, *, seed=2026091505, draws=20000):
    if not rows or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Evaluation rows must be nonempty and unique')
    if set(references) != {r['case_id'] for r in rows}:
        raise ValueError('Paired comparison requires exact case coverage')
    values = np.asarray([r['makespan'] for r in rows], dtype=np.float64)
    base = np.asarray([references[r['case_id']] for r in rows], dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Invalid paired costs')
    tail = max(1, math.ceil(len(rows) * .1))
    distribution = {d: float(values[idx].mean() / base[idx].mean() - 1)
                    for d in sorted({r['distribution'] for r in rows})
                    for idx in [[i for i, r in enumerate(rows) if r['distribution'] == d]]}
    groups = [np.asarray([i for i, r in enumerate(rows) if r['profile'] == p])
              for p in sorted({r['profile'] for r in rows})]
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, draws, 1000):
        n = min(1000, draws - start)
        ids = np.concatenate([rng.choice(g, size=(n, len(g))) for g in groups], axis=1)
        boot.extend((values[ids].mean(axis=1) / base[ids].mean(axis=1) - 1).tolist())
    return dict(case_count=len(rows), makespan=float(values.mean()), reference_makespan=float(base.mean()),
                gap_fraction=float(values.mean() / base.mean() - 1),
                paired_gap_ci95=np.quantile(boot, [.025, .975]).tolist(),
                ci_scope='case uncertainty conditional on fixed trained seed',
                wins=int(np.sum(values < base - 1e-6)), ties=int(np.sum(np.abs(values - base) <= 1e-6)),
                losses=int(np.sum(values > base + 1e-6)),
                regression_over_5pct_fraction=float(np.mean(values > base * 1.05)),
                regression_over_10pct_fraction=float(np.mean(values > base * 1.10)),
                tail_ratio=float(np.sort(values)[-tail:].mean() / np.sort(base)[-tail:].mean()),
                distributions=distribution,
                completed=all(r.get('completed') and not r.get('cycle_terminated') for r in rows))


def selection_eligible(summary):
    return (summary['completed'] and summary['gap_fraction'] <= .002
            and summary['tail_ratio'] <= 1.01
            and all(v <= .01 for v in summary['distributions'].values()))
