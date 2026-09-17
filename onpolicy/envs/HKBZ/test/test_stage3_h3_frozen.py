"""Contracts and real Adam/recurrent interfaces for the frozen-IGA H3 branch."""
from contextlib import contextmanager
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from onpolicy.runner.shared.stage3_b_shared_b0_engine import PPOContractError
from onpolicy.runner.shared.stage3_h3_frozen_engine import H3FrozenEngine, model_digest
from onpolicy.utils.stage3_b_shared_b0 import recipe as old_recipe, HISTORY, FROZEN
from onpolicy.utils.stage3_h3_frozen import (
    PLANNING, MissingFrozenTeacher, budget, freeze_iga, optimizer_orders,
    paired_metrics, recipe, require_training_teachers, schedule, selection_eligible,
)
from onpolicy.utils.stage3_research import atomic_json, read_json
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.envs.HKBZ.test.test_stage3_b_shared_b0 import ToyActor, _toy_replay


def cases():
    return [dict(path=f'/train/{d}/{i}', distribution=d, profile=d,
                 content_sha256=f'{d}-{i}', case_sha256=f'case-{d}-{i}')
            for d, n in [('iid', 192), ('ood_stress', 43), ('ood_scale', 5)] for i in range(n)]


def test_schedule_keeps_train240_384_visits_and_96_real_steps():
    r = recipe(old_recipe())
    result = schedule(cases(), r)
    assert len(result) == 8
    seen = set()
    for row in result:
        assert len(row['cases']) == 384 and len({c['path'] for c in row['cases']}) == 240
        assert not seen.intersection(row['visit_ids'])
        seen.update(row['visit_ids'])
        counts = {d: sum(c['distribution'] == d for c in row['cases'])
                  for d in ('iid', 'ood_stress', 'ood_scale')}
        assert counts == {'iid': 192, 'ood_stress': 172, 'ood_scale': 20}
    assert len(seen) == budget(r)['new_training_visits'] == 3072
    assert budget(r)['max_new_ppo_steps'] == 96
    assert result == schedule(cases(), r)


@pytest.mark.parametrize('n,batch', [(384, 64), (4, 2)])
def test_optimizer_shuffle_covers_each_visit_once_per_pass(n, batch):
    result = optimizer_orders(n, batch, 2, 11)
    assert all(sorted(i for part in order for i in part) == list(range(n)) for order in result)
    assert result == optimizer_orders(n, batch, 2, 11)
    assert len(result[0]) == n // batch


def test_invalid_partial_optimizer_group_is_rejected():
    with pytest.raises(ValueError, match='partition'):
        optimizer_orders(144, 64, 2, 0)


def test_recipe_preserves_previous_resources_and_forbids_solver():
    before = old_recipe()
    original = copy.deepcopy(before)
    r = recipe(before)
    assert r['gpu_uuid'] == before['gpu_uuid'] and r['cpus'] == before['cpus']
    assert r['rollout_workers'] == 240 and r['environment_processes'] == 64
    assert r['teacher_queries_allowed'] == 0 and r['planning'] == PLANNING
    assert before == original
    r['teacher_queries_allowed'] = 1
    with pytest.raises(ValueError, match='protocol'):
        budget(r)


def test_recipe_planning_is_not_a_mutable_alias_of_contract():
    r = recipe(old_recipe())
    r['planning']['device_future_intent_horizon'] = 2
    assert PLANNING['device_future_intent_horizon'] == 3
    with pytest.raises(ValueError, match='protocol'):
        budget(r)


def write_iga(root, case_hash='tune-hash', horizon=3):
    for tier in ('iga180', 'iga1800'):
        atomic_json(root / tier / 'summary.json', {'completed': True})
        atomic_json(root / tier / 'teachers/case_0001.json', dict(
            PLANNING, device_future_intent_horizon=horizon,
            teacher_scope='stage3_full_joint_policy', teacher_method='joint_iga_all',
            completed=True, completion_verified=True, makespan=100., case_sha256=case_hash))


def test_frozen_iga_cannot_leak_tune_into_training(tmp_path):
    write_iga(tmp_path)
    splits = {'train': cases()[:1], 'tune': [dict(path='/tune/case_0001',
              case_sha256='tune-hash', content_sha256='tune-content')]}
    index = freeze_iga(tmp_path, splits)
    assert index['solver_queries'] == 0 and index['references']['train'] == {}
    assert len(index['references']['tune']) == 1
    with pytest.raises(MissingFrozenTeacher, match='substitution are forbidden'):
        require_training_teachers(index, splits['train'])


def test_frozen_h1_teacher_cannot_be_relabelled_h3(tmp_path):
    write_iga(tmp_path, horizon=1)
    with pytest.raises(ValueError, match='contract mismatch'):
        freeze_iga(tmp_path, {'train': [], 'tune': []})


def test_missing_iga_is_an_error_not_a_solver_request(tmp_path):
    with pytest.raises(MissingFrozenTeacher):
        freeze_iga(tmp_path, {'train': [], 'tune': []})


def test_paired_mean_ratio_and_missing_case_guard():
    rows = [dict(case_id='a', makespan=110., profile='a', distribution='iid', completed=True),
            dict(case_id='b', makespan=190., profile='b', distribution='ood_stress', completed=True)]
    result = paired_metrics(rows, {'a': 100., 'b': 200.}, draws=100)
    assert result['gap_fraction'] == 0 and result['wins'] == result['losses'] == 1
    assert result['regression_over_5pct_fraction'] == .5
    assert result['tail_ratio'] == .95
    with pytest.raises(ValueError, match='coverage'):
        paired_metrics(rows, {'a': 100.}, draws=10)


def test_selection_rejects_distribution_regression_even_if_mean_improves():
    good = dict(completed=True, gap_fraction=-.01, tail_ratio=.99,
                distributions={'iid': -.01, 'ood_stress': 0.})
    assert selection_eligible(good)
    assert not selection_eligible(dict(good, distributions={'ood_stress': .02}))


class Cache:
    last_actor = None
    last_report = {}

    def prepare_graph(self, graph):
        return graph

    @contextmanager
    def group(self):
        yield self


def toy():
    r = object.__new__(H3FrozenEngine)
    r.config = recipe(old_recipe()); r.device = torch.device('cpu')
    ac = ToyActor()
    r.policy = SimpleNamespace(ac=ac,
        actor_optimizer=torch.optim.Adam([ac.theta], lr=1e-5),
        critic_optimizer=torch.optim.Adam(ac.team_critic.parameters(), lr=1e-4))
    r.cache = Cache(); r.norm = ValueNorm(1); r.norm.debiasing_term.fill_(1.)
    r.training = True; r.commit_ready = True; r.normalization_sha256 = 'test'
    r.physical_microbatch = 32; r.policy_updates = 0
    r.assert_frozen = lambda: None
    r._replay = MethodType(_toy_replay, r)
    rows = []
    identity = model_digest(ac)
    for i in range(4):
        x = float(i + 1)
        rows.append(dict(case_id=f'/train/{i}', makespan=99., completed=True,
            behavior_deterministic=False, forced_replay=False, history=HISTORY, decoder='AR_sample', tau=.03,
            policy_updates=0, logical_id='toy', behavior_model_sha256=identity, visit_id=f'visit-{i}',
            states=[dict(x=x, old_logp=torch.nn.functional.logsigmoid(ac.theta.detach() * x).numpy(),
                         mask=np.ones(3, np.float32), roles=np.arange(3), time=0.)]))
    return r, rows


def test_minibatch_update_steps_adam_each_partition_not_each_microbatch():
    r, rows = toy()
    result = r.update_logical(rows, {t['case_id']: 100. for t in rows}, logical_id='toy',
                              shuffle_seed=19, diagnostic=True, minibatch=2, microbatch=1)
    assert result['actual_ppo_steps'] == r.policy_updates == 4
    assert len(result['minibatches']) == 4 and result['completed_passes'] == 2
    assert r.commit_ready and r.policy.actor_optimizer.state
    assert all(int(v['step']) == 4 for v in r.policy.actor_optimizer.state.values())
    # All passes still refer to the frozen initial behavior version.
    assert result['behavior_version'] == 0


@pytest.mark.parametrize('field,value', [('policy_updates', 1), ('forced_replay', True),
                                       ('behavior_model_sha256', 'wrong'), ('logical_id', 'different')])
def test_update_rejects_mixed_or_offpolicy_visits(field, value):
    r, rows = toy()
    rows[1][field] = value
    with pytest.raises(PPOContractError, match='same fresh behavior'):
        r.update_logical(rows, {t['case_id']: 100. for t in rows}, logical_id='toy',
                         shuffle_seed=1, diagnostic=True, minibatch=2, microbatch=1)
    assert r.policy_updates == 0 and not r.policy.actor_optimizer.state


def test_unreproducible_behavior_cannot_update_or_commit(tmp_path):
    r, rows = toy()
    rows[0]['states'][0]['old_logp'] += .1
    with pytest.raises(PPOContractError, match='cannot be reproduced'):
        r.update_logical(rows, {t['case_id']: 100. for t in rows}, logical_id='toy',
                         shuffle_seed=1, diagnostic=True, minibatch=2, microbatch=1)
    assert r.policy_updates == 0 and not r.commit_ready
    with pytest.raises(PPOContractError, match='complete logical collection'):
        r.save(tmp_path / 'bad.pt', manifest_sha256='test', next_batch=1)


def test_soft_kl_records_actual_step_count_without_claiming_full_budget():
    r, rows = toy()
    r.config['soft_kl'] = 1e-12
    result = r.update_logical(rows, {t['case_id']: 100. for t in rows}, logical_id='toy',
                              shuffle_seed=19, diagnostic=True, minibatch=2, microbatch=1)
    assert result['stop_reason'] == 'soft_kl'
    assert result['actual_ppo_steps'] < result['planned_ppo_steps'] == 4
    assert result['actual_ppo_steps'] == r.policy_updates and r.commit_ready


def test_real_shared_model_h3_environment_and_checkpoint_identity(tmp_path):
    r = H3FrozenEngine(FROZEN, config=recipe(old_recipe()), device='cpu',
                        runtime={'sampling_output': str(tmp_path)})
    try:
        config = r.environment_config('/example/case_0001', 42)
        assert all(config[k] == v for k, v in PLANNING.items())
        assert config['resource_policy'] == 'drl' and config['joint_iga_teacher_dir'] == ''
        r.set_normalization([10., -20., 80.])
        loss = sum(g['params'][0].square().mean() for g in r.policy.actor_optimizer.param_groups)
        loss += next(r.policy.ac.team_critic.parameters()).square().mean()
        loss.backward(); r.policy.actor_optimizer.step(); r.policy.critic_optimizer.step()
        r.policy_updates = 1
        path = tmp_path / 'committed.pt'
        r.save(path, manifest_sha256='test', next_batch=1)
        original = copy.deepcopy(r.policy.actor_optimizer.state_dict())
        r.policy.actor_optimizer.state.clear()
        assert r.resume(path, manifest_sha256='test') == 1
        assert r.policy.actor_optimizer.state and r.policy_updates == 1
        assert r.policy.actor_optimizer.state_dict()['param_groups'] == original['param_groups']
        with pytest.raises(PPOContractError, match='identity'):
            r.resume(path, manifest_sha256='different')
    finally:
        r.close()


def test_first_cycle_selects_complete_budget_and_restores_selected_candidate(monkeypatch, tmp_path):
    from onpolicy.scripts.train import run_stage3_h3_frozen as driver
    from onpolicy.utils.stage3_research import read_json
    class Optimizer:
        def zero_grad(self, **kwargs): pass
    class Runner:
        physical_microbatch = 64
        policy = SimpleNamespace(actor_optimizer=Optimizer(), critic_optimizer=Optimizer())
        policy_updates = 0
        def resume(self, path, **kwargs):
            self.physical_microbatch = int(Path(path).read_text())
        def update_logical(self, *args, **kwargs):
            # A faster partial candidate must not win a throughput comparison.
            return dict(actual_ppo_steps=12 if self.physical_microbatch == 64 else 6,
                        completed_passes=2 if self.physical_microbatch == 64 else 1)
        def save(self, path, **kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(self.physical_microbatch))
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/initial.pt').write_text('64')
    monkeypatch.setattr(driver, 'save_rng', lambda: 'post-rollout-rng')
    restored = []
    monkeypatch.setattr(driver, 'restore_rng', restored.append)
    for name in ('empty_cache', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda: 1024)
    ticks = iter([0., 10., 10., 20., 21., 21.])
    monkeypatch.setattr(driver.time, 'monotonic', lambda: next(ticks))
    runner = Runner()
    m = {'root': str(tmp_path), 'manifest_sha256': 'manifest', 'recipe': recipe(old_recipe())}
    hb = SimpleNamespace(update=lambda **kwargs: None)
    out = driver.first_cycle_capacity(m, runner, [], {}, 'logical', tmp_path / 'attempt', hb, 5.)
    assert runner.physical_microbatch == 64 and out['actual_ppo_steps'] == 12
    assert restored == ['post-rollout-rng', 'post-rollout-rng']
    selected = read_json(tmp_path / 'physical_configuration.json')
    assert selected['microbatch'] == 64 and selected['discarded_diagnostic_ppo_steps'] == 6
