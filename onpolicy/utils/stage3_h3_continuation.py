"""Registered H3 continuation experiment, independent of the immutable R0 protocol."""
from __future__ import annotations

import copy
from collections import Counter
import os

from onpolicy.utils.stage3_b_shared_b0 import HISTORY, cpus
from onpolicy.utils.stage3_h3_frozen import (
    COUNTS, PLANNING, POST_PASS_REPLAY_SKIP, PROTOCOL as PARENT_PROTOCOL,
)
from onpolicy.utils.stage3_research import digest_json, trajectory_seed

PROTOCOL = 'stage3_h3_temperature_continuation_v1'
ARMS = {'C03': .03, 'T10': .1}
HALVES = ('0-31,64-95', '32-63,96-127')
EVALUATION_EPOCHS = [1, 2, 4, 6, 8]
EVALUATION_EPOCHS_EXTENDED = [1, 2, 4, 6, 8, 10]
EPOCH_BUDGETS = {8: EVALUATION_EPOCHS, 10: EVALUATION_EPOCHS_EXTENDED}
DUAL_PROFILE = 'dual_env128_mb64_cache1024_v1'
DUAL_ENV96_PROFILE = 'dual_env96_mb64_no_memory_protection_v1'
SINGLE_PROFILES = {'single_parent_v1': 240, 'single_parent_env192_v1': 192,
                   'single_parent_env384_v1': 384}
MB128_PROFILE = 'single_parent_env384_mb128_v1'
SINGLE_PROFILES[MB128_PROFILE] = 384
MB128_FRESH_PROFILE = 'single_parent_env384_mb128_fresh_v1'
SINGLE_PROFILES[MB128_FRESH_PROFILE] = 384
MB128_PROFILES = (MB128_PROFILE, MB128_FRESH_PROFILE)
MB192_PROFILE = 'single_parent_env384_mb192_v1'
SINGLE_PROFILES[MB192_PROFILE] = 384
MB192_NOPOST_PROFILE = 'single_parent_env384_mb192_nopost_v1'
SINGLE_PROFILES[MB192_NOPOST_PROFILE] = 384
MB192_FAMILY = (MB192_PROFILE, MB192_NOPOST_PROFILE)
BATCH_RESIZE_PROFILES = (MB128_PROFILE, MB192_PROFILE)
PROTOCOL_CHANGE_PROFILES = (MB192_NOPOST_PROFILE,)
WIDE_BATCH_PROFILES = (*MB128_PROFILES, *MB192_FAMILY)
FIXED_EPOCHS_POLICY = 'fixed_eight_epochs_numeric_gates_v1'
DUAL_PROFILES = {
    DUAL_PROFILE: dict(rollout_workers=128, memory_high_gib=164, memory_max_gib=176),
    DUAL_ENV96_PROFILE: dict(rollout_workers=96, memory_high_gib=None, memory_max_gib=None,
                            memory_swap_gib=None, memory_protection='disabled'),
}


def active_arms(suite):
    """A single-parent continuation must never create the temperature arm."""
    return ('C03',) if suite.get('execution_mode') == 'single' else tuple(ARMS)


def arm_cpus(cpu_half, arm):
    if cpu_half not in HALVES or arm not in ARMS:
        raise ValueError('Unknown CPU half or temperature arm')
    offset = (0 if cpu_half == HALVES[0] else 32) + (0 if arm == 'C03' else 16)
    return f'{offset}-{offset+15},{offset+64}-{offset+79}'


def resources(gpu, uuid, cpu_half):
    if gpu not in (0, 1) or not uuid.startswith('GPU-') or cpu_half not in HALVES:
        raise ValueError('Register one physical GPU and one complete CPU half')
    offset = 0 if cpu_half == HALVES[0] else 32
    return dict(gpu=gpu, gpu_uuid=uuid, cpus=cpu_half,
                trainer_cpus=f'{offset}-{offset+21},{offset+64}-{offset+85}',
                validator_cpus=f'{offset+22}-{offset+29},{offset+86}-{offset+93}',
                controller_cpus=f'{offset+30}-{offset+31},{offset+94}-{offset+95}')


def recipe(previous, allocation, arm, *, parent_epoch=8, advantage='source_relative', execution='serial',
           dual_profile=None, single_profile=None, optimizer_resize_after_epoch=None, stopping_policy=None,
           epochs=8):
    if arm not in ARMS or advantage not in ('source_relative', 'state_value'):
        raise ValueError('Unregistered continuation arm/advantage')
    if single_profile is not None and execution != 'single':
        raise ValueError('Single profile requires single execution')
    if epochs not in EPOCH_BUDGETS:
        raise ValueError('Unregistered epoch budget')
    r = copy.deepcopy(previous)
    r.update(allocation)
    r.update(protocol=PROTOCOL, arm=arm, train_tau=ARMS[arm], advantage=advantage,
             initialization_mode='full_state_fork', parent_epoch=parent_epoch,
             epochs=epochs, seed=2026091702, optimizer_shuffle_seed=2026091706,
             physical_microbatch_candidates=[64], microbatch=64,
             evaluation_epochs=list(EPOCH_BUDGETS[epochs]),
             train_wall_timeout_seconds=120*3600, screen_wall_timeout_seconds=36*3600,
             canonical_contract='h_exact_raw_blocking_device_request_lex_v1',
             decision_observer=True, checkpoint_boundary='complete_logical_rollout')
    if execution == 'dual':
        profile = dual_profile or DUAL_PROFILE
        if profile not in DUAL_PROFILES:
            raise ValueError('Unknown dual-arm execution profile')
        r.update(execution_mode='dual', execution_profile=profile, environment_processes=32,
                 trainer_cpus=arm_cpus(allocation['cpus'], arm),
                 input_cache_mib=1024, cuda_allocator_mib=18432,
                 deferred_replay_metrics=True, trim_host_heap=True)
        r.update(DUAL_PROFILES[profile])
    elif execution == 'single':
        if arm != 'C03' or dual_profile is not None or parent_epoch != 8:
            raise ValueError('Single continuation requires the selected E8 C03 parent')
        profile = single_profile or 'single_parent_v1'
        if profile not in SINGLE_PROFILES:
            raise ValueError('Unknown single-parent execution profile')
        r.update(execution_mode='single', execution_profile=profile,
                 rollout_workers=SINGLE_PROFILES[profile],
                 seed=previous['seed'],
                 optimizer_shuffle_seed=previous['optimizer_shuffle_seed']+parent_epoch,
                 sampling_schedule='parent_absolute_epoch_v1', rng_initialization='parent_checkpoint',
                 decision_observer=False, memory_high_gib=None, memory_max_gib=None,
                 memory_swap_gib=None, memory_protection='disabled')
        if profile in BATCH_RESIZE_PROFILES:
            if type(optimizer_resize_after_epoch) is not int or not 1 <= optimizer_resize_after_epoch < 8:
                raise ValueError('Batch resize requires an explicit completed epoch boundary')
            width = 192 if profile == MB192_PROFILE else 128
            r.update(optimizer_minibatch=width, microbatch=width, physical_microbatch_candidates=[width],
                     optimizer_resize_after_epoch=optimizer_resize_after_epoch)
        elif profile == MB128_FRESH_PROFILE:
            r.update(optimizer_minibatch=128, microbatch=128, physical_microbatch_candidates=[128])
        elif profile in PROTOCOL_CHANGE_PROFILES:
            if type(optimizer_resize_after_epoch) is not int or not 1 <= optimizer_resize_after_epoch < 8:
                raise ValueError('Post-pass replay removal requires the inherited historical boundary')
            r.update(optimizer_minibatch=192, microbatch=192, physical_microbatch_candidates=[192],
                     optimizer_resize_after_epoch=optimizer_resize_after_epoch,
                     post_pass_replay=POST_PASS_REPLAY_SKIP)
    elif execution != 'serial':
        raise ValueError('Unknown execution mode')
    elif dual_profile is not None:
        raise ValueError('Dual-arm profile requires dual execution')
    if (optimizer_resize_after_epoch is not None
            and r.get('execution_profile') not in (*BATCH_RESIZE_PROFILES, *PROTOCOL_CHANGE_PROFILES)):
        raise ValueError('Optimizer boundary requires a registered batch resize profile')
    if stopping_policy is not None:
        r['stopping_policy'] = stopping_policy
    validate_recipe(r)
    return r


def validate_recipe(r):
    epochs = r.get('epochs')
    if epochs not in EPOCH_BUDGETS:
        raise ValueError('Unregistered epoch budget')
    fixed = dict(protocol=PROTOCOL, planning=PLANNING, history=HISTORY,
                 global_batch=384, rollout_workers=240, environment_processes=64,
                 optimizer_minibatch=64, microbatch=64, ppo_epochs=2, epochs=epochs,
                 physical_microbatch_candidates=[64], evaluation_tau=.3, validation_workers=12,
                 gradient_clip=1., clip=.2, actor_lr=1e-5, critic_lr=1e-4,
                 actor_lr_scales=[1., .25, 1., .5], soft_kl=.02, hard_kl=.04,
                 iga_mode='frozen_only', teacher_queries_allowed=0,
                 evaluation_epochs=EPOCH_BUDGETS[epochs], tbptt=8,
                 initialization_mode='full_state_fork',
                 canonical_contract='h_exact_raw_blocking_device_request_lex_v1')
    dual = r.get('execution_mode', 'serial') == 'dual'
    if dual:
        profile = r.get('execution_profile')
        if profile not in DUAL_PROFILES:
            raise ValueError('Unknown dual-arm execution profile')
        fixed.update(execution_profile=profile, environment_processes=32,
                     input_cache_mib=1024, cuda_allocator_mib=18432,
                     deferred_replay_metrics=True, trim_host_heap=True)
        fixed.update(DUAL_PROFILES[profile])
    elif r.get('execution_mode') == 'single':
        profile = r.get('execution_profile')
        if profile not in SINGLE_PROFILES:
            raise ValueError('Unknown single-parent execution profile')
        fixed.update(execution_profile=profile, rollout_workers=SINGLE_PROFILES[profile],
                     arm='C03', parent_epoch=8,
                     train_tau=.03, seed=2026091502, optimizer_shuffle_seed=2026091506+8,
                     sampling_schedule='parent_absolute_epoch_v1', rng_initialization='parent_checkpoint',
                     input_cache_mib=4096, decision_observer=False,
                     memory_high_gib=None, memory_max_gib=None, memory_swap_gib=None,
                     memory_protection='disabled')
        if profile in (*BATCH_RESIZE_PROFILES, *PROTOCOL_CHANGE_PROFILES):
            boundary = r.get('optimizer_resize_after_epoch')
            if type(boundary) is not int or not 1 <= boundary < 8:
                raise ValueError('Invalid optimizer resize boundary')
            width = 128 if profile == MB128_PROFILE else 192
            fixed.update(optimizer_minibatch=width, microbatch=width, physical_microbatch_candidates=[width])
            if profile in MB192_FAMILY:
                fixed.update(stopping_policy=FIXED_EPOCHS_POLICY)
            if profile in PROTOCOL_CHANGE_PROFILES:
                fixed.update(post_pass_replay=POST_PASS_REPLAY_SKIP)
        elif profile == MB128_FRESH_PROFILE:
            fixed.update(optimizer_minibatch=128, microbatch=128, physical_microbatch_candidates=[128])
        if r.get('cuda_allocator_mib') is not None or r.get('deferred_replay_metrics', False):
            raise ValueError('Single continuation must preserve parent execution parameters')
    elif r.get('execution_mode', 'serial') != 'serial':
        raise ValueError('Unknown execution mode')
    bad = [k for k, v in fixed.items() if r.get(k) != v]
    if (bad or r.get('arm') not in ARMS or r.get('train_tau') != ARMS.get(r.get('arm'))
            or r.get('advantage') not in ('source_relative', 'state_value')):
        raise ValueError(f'Continuation contract changed: {bad}')
    expected = resources(r['gpu'], r['gpu_uuid'], r['cpus'])
    if dual:
        expected['trainer_cpus'] = arm_cpus(r['cpus'], r['arm'])
    if any(r[k] != v for k, v in expected.items()):
        raise ValueError('CPU allocation is not the registered disjoint half')
    if (r.get('execution_profile') not in (*BATCH_RESIZE_PROFILES, *PROTOCOL_CHANGE_PROFILES)
            and 'optimizer_resize_after_epoch' in r):
        raise ValueError('Unexpected optimizer resize boundary')
    if r.get('post_pass_replay') not in (None, POST_PASS_REPLAY_SKIP):
        raise ValueError('Unregistered post-pass replay mode')
    if r.get('execution_profile') not in PROTOCOL_CHANGE_PROFILES and r.get('post_pass_replay') is not None:
        raise ValueError('Post-pass replay removal requires its registered profile')
    if 'stopping_policy' in r and (r['stopping_policy'] != FIXED_EPOCHS_POLICY
            or r.get('execution_profile') not in (MB128_FRESH_PROFILE, *WIDE_BATCH_PROFILES)):
        raise ValueError('Fixed epoch policy requires the registered fresh E8 continuation')


def planned_new_steps(r, epoch):
    """Preserve updates before the declared batch boundary; count new steps afterward."""
    if type(epoch) is not int or not 0 <= epoch <= r['epochs']:
        raise ValueError('Epoch is outside the registered continuation')
    width = r['optimizer_minibatch']
    per_epoch = r['global_batch'] // width * r['ppo_epochs']
    boundary = r.get('optimizer_resize_after_epoch', 0)
    historical = 6 if r.get('execution_profile') in MB192_FAMILY else 12
    return historical * min(epoch, boundary) + per_epoch * max(0, epoch-boundary)


def check_resources(r, *, cuda=True):
    validate_recipe(r)
    if not set(os.sched_getaffinity(0)).issubset(cpus(r['cpus'])):
        raise ValueError('Process escaped the registered CPU half')
    if r.get('execution_mode') == 'dual' and not set(os.sched_getaffinity(0)).issubset(cpus(r['trainer_cpus'])):
        raise ValueError('Process escaped its temperature arm CPU partition')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != r['gpu_uuid']:
        raise ValueError('CUDA must expose only the registered physical GPU UUID')
    if cuda:
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one GPU must be visible')
        actual = str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        if actual != r['gpu_uuid'].removeprefix('GPU-'):
            raise ValueError('Visible GPU identity differs')


def schedule(cases, r):
    validate_recipe(r)
    if (len(cases) != 240 or Counter(c['distribution'] for c in cases) != COUNTS
            or len({c['content_sha256'] for c in cases}) != 240):
        raise ValueError('Expected the same unique Train240')
    result = []
    for epoch in range(1, r['epochs'] + 1):
        visits = []
        for c in cases:
            for replica in range(1 if c['distribution'] == 'iid' else 4):
                seed = trajectory_seed(r['seed'], c['content_sha256'], r['parent_epoch']+epoch-1, replica)
                if r.get('sampling_schedule') == 'parent_absolute_epoch_v1':
                    identity = digest_json([PARENT_PROTOCOL, c['content_sha256'],
                                            r['parent_epoch']+epoch-1, replica, seed])
                else:
                    identity = digest_json([PROTOCOL, c['content_sha256'], epoch, replica, seed])
                visits.append((identity, c, seed))
        visits.sort(key=lambda x: digest_json([r['seed'], 'order', x[0]]))
        result.append(dict(epoch=epoch, cases=[x[1] for x in visits], seeds=[x[2] for x in visits],
                           visit_ids=[x[0] for x in visits], training_episodes=384*epoch))
    return result


def eligible(summary):
    return (summary['completed'] and summary['gap_fraction'] <= .002
            and summary['tail_ratio'] <= 1.01
            and summary['regression_over_5pct_fraction'] <= .10
            and all(v <= .01 for v in summary['distributions'].values()))


def parent_eligible(summary):
    return eligible(summary) and summary['gap_fraction'] <= 0


def choose_parent(e8, e6, e8_vs_e6):
    if parent_eligible(e8) and e8_vs_e6['gap_fraction'] <= .002:
        return 'epoch8'
    if parent_eligible(e6):
        return 'epoch6'
    raise ValueError('Neither pre-registered parent passes canonical admission')


def harmful(curve):
    latest = curve[-2:]
    return len(latest) == 2 and (all(s['gap_fraction'] > .01 for s in latest)
                                or all(not eligible(s) for s in latest))


def extension_eligible(summary):
    return (eligible(summary) and summary['gap_fraction'] <= -.003
            and summary['distributions'].get('ood_stress', 1) <= 0)


def extend_arms(summaries):
    """E4 budget allocation; never inspect Tune scores."""
    qualified = {a for a, s in summaries.items() if extension_eligible(s)}
    if not qualified:
        return []
    a, b = summaries.get('C03'), summaries.get('T10')
    if b is not None and 'T10' in qualified:
        if a is None:
            return ['T10']
        if b['makespan']/a['makespan'] - 1 <= -.002:
            return [k for k in ('C03', 'T10') if eligible(summaries[k])]
    return ['C03'] if 'C03' in qualified else ['T10']


def select_candidate(candidates):
    good = [r for r in candidates if eligible(r['versus_parent'])]
    if not good:
        raise ValueError('Parent must be included as a safe candidate')
    lowest = min(r['versus_parent']['makespan'] for r in good)
    near = [r for r in good if r['versus_parent']['makespan'] <= lowest*1.001]
    return min(near, key=lambda r: (r['versus_parent']['tail_ratio'], r['epoch'],
                                   {'parent': -1, 'C03': 0, 'T10': 1}.get(r['arm'], 2)))


def reusable_evaluation(request, previous):
    """Only the label/output may differ; admission always executes independently."""
    return (not request['expected'] and not previous['expected']
            and all(request[k] == previous[k] for k in
                    ('manifest_sha256','checkpoint','cases','decoder')))


def validate_parent_payload(payload, parent_manifest, expected_epoch):
    from onpolicy.utils.stage3_h3_frozen import PROTOCOL as OLD_PROTOCOL, SOURCE_SHA
    expected = dict(protocol=OLD_PROTOCOL, source_sha256=SOURCE_SHA,
                    manifest_sha256=parent_manifest['manifest_sha256'],
                    recipe_sha256=digest_json(parent_manifest['recipe']), history=HISTORY,
                    planning=PLANNING, initialization_sha256=parent_manifest['initialization']['sha256'],
                    next_batch=expected_epoch, complete_logical_rollout=True, diagnostic_only=False)
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError('Parent checkpoint lineage/recipe/commit identity mismatch')
    if payload.get('physical_microbatch') != 64 or not 0 < payload.get('policy_updates', 0) <= 12*expected_epoch:
        raise ValueError('Parent physical batch or cumulative optimizer steps differ')
    for k in ('actor_optim', 'critic_optim'):
        if not payload[k]['state']:
            raise ValueError('Full-state fork requires nonempty optimizer moments')
    if not payload.get('normalization_sha256') or 'value_normalizer' not in payload:
        raise ValueError('Full-state fork requires the parent ValueNorm')
    return payload
