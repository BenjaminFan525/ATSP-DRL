"""Registered profile that drops the two post-pass full replays at a checkpoint boundary."""
import copy
from pathlib import Path

import pytest

from onpolicy.envs.HKBZ.test.test_stage3_h3_frozen import cases, toy
from onpolicy.utils.stage3_b_shared_b0 import recipe as b0_recipe
from onpolicy.utils.stage3_h3_frozen import POST_PASS_REPLAY_SKIP, recipe as parent_recipe, bind
from onpolicy.utils.stage3_h3_continuation import (
    EVALUATION_EPOCHS_EXTENDED, FIXED_EPOCHS_POLICY, MB192_NOPOST_PROFILE, MB192_PROFILE,
    recipe, resources, planned_new_steps, schedule, validate_recipe,
)
from onpolicy.utils.stage3_h3_resize import rebind_payload, verify_recipe_change
from onpolicy.utils.stage3_h3_checkpoint_switch import request_identity, verify_request
from onpolicy.utils.stage3_research import atomic_json, digest_json, read_json


def origin():
    return recipe(parent_recipe(b0_recipe()), resources(0, 'GPU-test', '0-31,64-95'), 'C03',
                  execution='single', single_profile=MB192_PROFILE, optimizer_resize_after_epoch=1,
                  stopping_policy=FIXED_EPOCHS_POLICY)


def target():
    return recipe(parent_recipe(b0_recipe()), resources(0, 'GPU-test', '0-31,64-95'), 'C03',
                  execution='single', single_profile=MB192_NOPOST_PROFILE,
                  optimizer_resize_after_epoch=1, stopping_policy=FIXED_EPOCHS_POLICY)


def test_registered_change_touches_only_the_execution_profile_and_replay_mode():
    before, after = origin(), target()
    changes = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changes == {'execution_profile', 'post_pass_replay'}
    assert after['post_pass_replay'] == POST_PASS_REPLAY_SKIP
    verify_recipe_change(before, after)
    assert schedule(cases(), before) == schedule(cases(), after)
    assert [planned_new_steps(after, e) for e in range(9)] == [planned_new_steps(before, e) for e in range(9)]
    assert [planned_new_steps(after, e) for e in range(9)] == [0, 6, 10, 14, 18, 22, 26, 30, 34]


@pytest.mark.parametrize('key,value', [('microbatch', 128), ('optimizer_minibatch', 64),
    ('optimizer_resize_after_epoch', 3), ('stopping_policy', None), ('post_pass_replay', 'nonsense')])
def test_unregistered_edits_are_rejected(key, value):
    after = target(); after[key] = value
    with pytest.raises(ValueError):
        verify_recipe_change(origin(), after)


def test_engine_skips_both_post_pass_replays_and_derives_the_pass_record():
    runner, small = toy(); runner.config = target(); runner.physical_microbatch = 192
    calls = []
    original = runner.replay_metrics
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    runner.replay_metrics = counted
    rows = []
    for i in range(384):
        row = copy.deepcopy(small[i % 4]); row.update(visit_id=f'visit-{i}', logical_id='nopost')
        rows.append(row)
    result = runner.update_logical(rows, {r['case_id']: 100. for r in rows},
                                   logical_id='nopost', shuffle_seed=19)
    assert len(calls) == 1, 'only the pre-update full replay may run'
    assert result['actual_ppo_steps'] == 4 and result['completed_passes'] == 2
    assert len(result['pass_metrics']) == 2
    for index, post in enumerate(result['pass_metrics'], 1):
        applied = [m for m in result['minibatches'] if m['ppo_pass'] == index]
        assert post['excludes_final_step'] is True and post['post_pass_replay'] == 'skipped'
        assert post['decisions'] == sum(m['pre_step']['decisions'] for m in applied)
        assert post['kl'] == max(m['pre_step']['kl'] for m in applied)


def test_per_minibatch_kl_gate_is_unchanged(monkeypatch):
    runner, small = toy(); runner.config = target(); runner.physical_microbatch = 192
    from onpolicy.runner.shared.stage3_b_shared_b0_engine import PPOContractError
    def exploding(group, source_costs, microbatch, heartbeat=None):
        kl = runner.config['hard_kl'] + .01
        return dict(actor_loss=0., critic_loss=0.,
                    pre_step=dict(decisions=1, kl=kl,
                                  roles={'0': dict(decisions=1, kl=kl, kl_sum=0., clip_count=0.)}))
    monkeypatch.setattr(runner, '_minibatch_backward', exploding)
    rows = []
    for i in range(384):
        row = copy.deepcopy(small[i % 4]); row.update(visit_id=f'visit-{i}', logical_id='nopost')
        rows.append(row)
    with pytest.raises(PPOContractError, match='Hard KL'):
        runner.update_logical(rows, {r['case_id']: 100. for r in rows},
                              logical_id='nopost', shuffle_seed=19)


def test_rebind_keeps_every_state_field_and_the_physical_batch():
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    old, new = origin(), target()
    old_arm = dict(recipe=old, manifest_sha256='old-arm', parent_checkpoint={'sha256': 'parent'})
    new_arm = dict(recipe=new, manifest_sha256='new-arm', parent_checkpoint={'sha256': 'parent'})
    payload = dict(protocol=old['protocol'], source_sha256=__import__(
        'onpolicy.utils.stage3_b_shared_b0', fromlist=['SOURCE_SHA']).SOURCE_SHA,
        manifest_sha256='old-arm', recipe_sha256=digest_json(old), initialization_sha256='parent',
        planning=old['planning'], history=old['history'], next_batch=3, policy_updates=110,
        physical_microbatch=192, train_tau=.03, complete_logical_rollout=True, diagnostic_only=False,
        model={'w': 1}, actor_optim={'state': {}}, critic_optim={'state': {}},
        value_normalizer={'mean': 0.}, normalization_sha256='norm',
        rng_python='py', rng_numpy='np', rng_torch='torch', rng_cuda='cuda')
    commit = dict(epoch=3, new_ppo_steps=planned_new_steps(new, 3), cumulative_ppo_steps=110)
    rebound = rebind_payload(payload, old_arm, new_arm, commit)
    assert rebound['manifest_sha256'] == 'new-arm'
    assert rebound['recipe_sha256'] == digest_json(new)
    assert rebound['physical_microbatch'] == 192
    metadata = {'manifest_sha256', 'recipe_sha256'}
    compare_states({k: v for k, v in payload.items() if k not in metadata},
                   {k: v for k, v in rebound.items() if k not in metadata}, exact=True)


def test_switch_request_binds_the_mb192_origin(tmp_path):
    old = origin()
    allocation = resources(0, 'GPU-test', '0-31,64-95')
    root = tmp_path/'origin'; root.mkdir()
    manifest = dict(root=str(root), execution_mode='single', recipe=old,
                    resources=allocation)
    atomic_json(root/'manifest.json', manifest)
    for name in ('tests.txt', 'plan.md'):
        (root/name).write_text(name)
    request = dict(origin=bind(root/'manifest.json'), target_profile=MB192_NOPOST_PROFILE,
                   after_epoch=3, requested_minibatch=192, requested_microbatch=192,
                   resources=allocation, tests=bind(root/'tests.txt'), plan=bind(root/'plan.md'),
                   source_root=str(root), source_files={})
    request['request_sha256'] = request_identity(request)
    assert verify_request(request)['recipe'] == old
    request['requested_microbatch'] = 128
    request['request_sha256'] = request_identity(request)
    with pytest.raises(ValueError, match='dimensions'):
        verify_request(request)


def extended(epochs=10):
    return recipe(parent_recipe(b0_recipe()), resources(0, 'GPU-test', '0-31,64-95'), 'C03',
                  execution='single', single_profile=MB192_NOPOST_PROFILE,
                  optimizer_resize_after_epoch=1, stopping_policy=FIXED_EPOCHS_POLICY, epochs=epochs)


def test_epoch_extension_to_ten_extends_the_schedule_without_reordering():
    before, after = origin(), extended()
    verify_recipe_change(before, after)
    assert after['evaluation_epochs'] == EVALUATION_EPOCHS_EXTENDED
    assert [planned_new_steps(after, e) for e in range(11)] == [0, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42]
    old_rows, new_rows = schedule(cases(), before), schedule(cases(), after)
    assert len(new_rows) == 10 and new_rows[:8] == old_rows


def test_epoch_ten_requires_the_matching_evaluation_list():
    after = extended(); after['evaluation_epochs'] = [1, 2, 4, 6, 8]
    with pytest.raises(ValueError):
        validate_recipe(after)


@pytest.mark.parametrize('epochs', [9, 12, 16])
def test_unregistered_epoch_budgets_are_rejected(epochs):
    with pytest.raises(ValueError, match='epoch budget'):
        extended(epochs)


def test_epoch_extension_beyond_plus_two_is_rejected_by_the_switch_contract():
    after = extended(epochs=8)
    after['epochs'] = 12
    after['evaluation_epochs'] = [1, 2, 4, 6, 8, 12]
    with pytest.raises(ValueError):
        verify_recipe_change(origin(), after)


def test_same_profile_epoch_extension_is_allowed():
    before, after = extended(epochs=8), extended(epochs=10)
    changes = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changes == {'epochs', 'evaluation_epochs'}
    verify_recipe_change(before, after)


def test_plain_wide_batch_profile_cannot_extend_the_budget():
    before, after = origin(), origin()
    after['epochs'] = 10
    after['evaluation_epochs'] = [1, 2, 4, 6, 8, 10]
    with pytest.raises(ValueError):
        verify_recipe_change(before, after)


def test_switch_request_extends_from_the_same_profile(tmp_path):
    allocation = resources(0, 'GPU-test', '0-31,64-95')
    old = extended(epochs=8)
    root = tmp_path/'origin'; root.mkdir()
    atomic_json(root/'manifest.json', dict(root=str(root), execution_mode='single',
                                          recipe=old, resources=allocation))
    for name in ('tests.txt', 'plan.md'):
        (root/name).write_text(name)
    request = dict(origin=bind(root/'manifest.json'), target_profile=MB192_NOPOST_PROFILE,
                   after_epoch=4, requested_minibatch=192, requested_microbatch=192,
                   requested_epochs=10, resources=allocation, tests=bind(root/'tests.txt'),
                   plan=bind(root/'plan.md'), source_root=str(root), source_files={})
    request['request_sha256'] = request_identity(request)
    assert verify_request(request)['recipe'] == old
    late = dict(request, after_epoch=8)          # boundary 8 is valid for a 10-epoch budget
    late['request_sha256'] = request_identity(late)
    assert verify_request(late)['recipe'] == old
    too_late = dict(request, after_epoch=10)
    too_late['request_sha256'] = request_identity(too_late)
    with pytest.raises(ValueError, match='boundary'):
        verify_request(too_late)
    request['requested_epochs'] = 12
    request['request_sha256'] = request_identity(request)
    with pytest.raises(ValueError, match='epoch budget'):
        verify_request(request)


def test_arm_manifest_inherits_the_suite_epoch_budget(tmp_path):
    """The arm recipe must carry the suite budget; a stale default kills epoch 9."""
    import onpolicy.scripts.train.run_stage3_h3_continuation as driver
    suite_root = tmp_path/'suite'
    (suite_root/'arms/C03').mkdir(parents=True)
    parent_manifest = tmp_path/'parent.json'
    atomic_json(parent_manifest, {'recipe': parent_recipe(b0_recipe())})
    atomic_json(suite_root/'parent_selection.json', {'epoch': 8})
    suite = dict(root=str(suite_root), resources=resources(0, 'GPU-test', '0-31,64-95'),
                 splits={'train': cases()}, recipe=extended(epochs=10), execution_mode='single',
                 parent_manifest=bind(parent_manifest),
                 baselines={'path': 'x', 'sha256': 'y'}, frozen_manifest={'path': 'x', 'sha256': 'y'})
    manifest_path = suite_root/'manifest.json'
    atomic_json(manifest_path, suite)
    controller = driver.Controller.__new__(driver.Controller)
    controller.m, controller.path, controller.root = suite, manifest_path, suite_root
    _, arm = controller.arm_manifest('C03', {'epoch': 8, 'checkpoint': {'path': 'e8.pt', 'sha256': 'abc'}})
    assert arm['recipe']['epochs'] == 10
    assert arm['recipe']['evaluation_epochs'] == [1, 2, 4, 6, 8, 10]
    assert len(schedule(cases(), arm['recipe'])) == 10
