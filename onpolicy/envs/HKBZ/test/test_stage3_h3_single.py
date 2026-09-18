"""Only the selected parent is continued, with the original rollout schedule."""
from types import SimpleNamespace

import pytest

from onpolicy.envs.HKBZ.test.test_stage3_h3_continuation import metric
from onpolicy.envs.HKBZ.test.test_stage3_h3_frozen import cases
from onpolicy.utils.stage3_b_shared_b0 import recipe as b0_recipe
from onpolicy.utils.stage3_h3_frozen import recipe as parent_recipe
from onpolicy.utils.stage3_h3_continuation import active_arms, recipe, resources, schedule, validate_recipe


def single():
    parent = parent_recipe(b0_recipe())
    return recipe(parent, resources(0, 'GPU-test', '0-31,64-95'), 'C03', execution='single')


def test_single_preserves_parent_training_parameters_and_restores_rng():
    old, new = parent_recipe(b0_recipe()), single()
    for key in ('planning','train_tau','rollout_workers','environment_processes','global_batch',
                'optimizer_minibatch','microbatch','ppo_epochs','actor_lr','critic_lr',
                'actor_lr_scales','gradient_clip','clip','tbptt','input_cache_mib','seed'):
        assert new[key] == old[key]
    assert new['optimizer_shuffle_seed']+1 == old['optimizer_shuffle_seed']+9
    assert new['rng_initialization'] == 'parent_checkpoint'
    assert active_arms({'execution_mode':'single'}) == ('C03',)
    assert active_arms({'execution_mode':'dual'}) == ('C03','T10')
    assert new['memory_protection'] == 'disabled'


def test_single_schedule_matches_original_epochs_9_to_16(monkeypatch):
    import onpolicy.utils.stage3_h3_frozen as old_module
    old = parent_recipe(b0_recipe())
    old_module.validate_recipe(old)
    # Extend only the historical loop bound to compare its unchanged scheduler.
    old['epochs'] = 16
    monkeypatch.setattr(old_module, 'validate_recipe', lambda _: None)
    expected = old_module.schedule(cases(), old)[8:]
    actual = schedule(cases(), single())
    for before, after in zip(expected, actual):
        for key in ('cases','seeds','visit_ids'):
            assert before[key] == after[key]
    assert len(actual) == 8
    assert sum(len(row['cases']) for row in actual) == 3072


@pytest.mark.parametrize('field,value', [('arm','T10'),('train_tau',.1),('rollout_workers',96),
    ('input_cache_mib',1024),('seed',2026091702),('deferred_replay_metrics',True)])
def test_single_rejects_temperature_or_execution_drift(field, value):
    r = single(); r[field] = value
    with pytest.raises(ValueError):
        validate_recipe(r)


def test_single_controller_never_schedules_t10(tmp_path, monkeypatch):
    import onpolicy.scripts.train.run_stage3_h3_continuation as driver
    controller = driver.Controller.__new__(driver.Controller)
    controller.m = {'recipe':{'epochs':8},'execution_mode':'single'}
    controller.arms = active_arms(controller.m)
    controller.root = tmp_path
    calls = []
    controller.baseline = lambda: {'epoch':8}
    controller.arm_manifest = lambda a,p: (tmp_path/a, {'arm':a})
    controller.canaries = lambda arms: calls.append(('canaries',tuple(arms)))
    controller.train_to = lambda a,item,e,p: calls.append(('train',a,e)) or True
    controller.finish = lambda p,arms: calls.append(('finish',tuple(arms)))
    monkeypatch.setattr(driver, 'read_json', lambda _: {'versus_parent':metric(-.006)})
    controller.run()
    assert calls == [('canaries',('C03',)),('train','C03',2),('train','C03',4),
                     ('train','C03',8),('finish',('C03',))]


def test_single_worker_does_not_reseed_after_parent_restore(monkeypatch, tmp_path):
    import onpolicy.scripts.train.stage3_h3_continuation_worker as worker
    restored = []
    def fork(*args, **kwargs):
        restored.append(kwargs['parent_epoch'])
        return {'restored':True}
    fake = SimpleNamespace(fork_parent=fork)
    monkeypatch.setattr(worker, 'H3ContinuationEngine', lambda *a,**kw: fake)
    monkeypatch.setattr(worker, 'checked', lambda record: record['path'])
    monkeypatch.setattr(worker, 'read_json', lambda path: {})
    def reseed(_):
        pytest.fail('Parent RNG must not be replaced with a new experiment seed')
    monkeypatch.setattr(worker, 'seed_all', reseed)
    m = dict(recipe=single(),frozen_manifest={'path':'frozen'},
             parent_manifest={'path':'parent'},parent_checkpoint={'path':'e8','sha256':'sha'})
    runner, receipt = worker.engine(m, tmp_path)
    assert runner is fake and receipt['restored'] and restored == [8]


def test_env192_preserves_single_flow_and_training_schedule():
    old = parent_recipe(b0_recipe())
    r = recipe(old, resources(0,'GPU-test','0-31,64-95'), 'C03', execution='single',
               single_profile='single_parent_env192_v1')
    before = single()
    assert {k for k in r if r[k] != before[k]} == {'rollout_workers','execution_profile'}
    assert r['rollout_workers'] == 192 and r['environment_processes'] == 64
    assert r['global_batch']//r['rollout_workers'] == 2
    assert schedule(cases(),r) == schedule(cases(),before)
    r['rollout_workers'] = 240
    with pytest.raises(ValueError):
        validate_recipe(r)


def test_single_profile_cannot_be_used_to_launch_two_arms():
    with pytest.raises(ValueError, match='Single profile'):
        recipe(parent_recipe(b0_recipe()), resources(0,'GPU-test','0-31,64-95'),
               'C03', execution='dual', single_profile='single_parent_env192_v1')
