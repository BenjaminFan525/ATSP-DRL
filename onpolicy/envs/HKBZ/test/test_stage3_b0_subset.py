import copy
from collections import Counter
from pathlib import Path

import pytest

from onpolicy.envs.HKBZ.test.test_stage3_b0_restart import continuation
from onpolicy.utils.stage3_b0_subset import (
    binding, create_selection, epoch_rows, make_amendment, select_cases, verify_selection, verify_subset,
)
from onpolicy.utils.stage3_b_shared_b0 import budget, epoch_episodes, training_schedule
from onpolicy.utils.stage3_research import atomic_json, digest_json, read_json


def subset_study(tmp_path, completed=11):
    parent = continuation()
    for row in parent['splits']['train']:
        row['profile'] = row['distribution'] + '-' + str(int(row['path'].split('/')[-1]) % 3)
    parent['splits']['validation'] = [dict(path='validation/case', content_sha256='held-out')]
    parent['splits']['tune'] = [dict(path='tune/case', content_sha256='held-out-tune')]
    parent['recipe'].update(global_batch=240, microbatch=120, epochs=8)
    parent['resume_amendment'].update(completed_batch_sizes=[32]*9+[64], completed_batches=10,
        completed_visits=352, completed_actor_updates=20)
    parent.update(root=str(tmp_path/'parent'), source_root=str(tmp_path/'source'), manifest_sha256='parent')
    old = training_schedule(parent)
    parent_path = tmp_path/'parent.json'
    atomic_json(parent_path, parent)
    selection_path = tmp_path/'selection.json'
    selected = create_selection(parent_path, selection_path, 2026091307)
    prefix = old[:completed]
    m = copy.deepcopy(parent)
    m.update(root=str(tmp_path), manifest_sha256='subset')
    m['recipe']['microbatch'] = 240
    m['splits']['train'] = selected['cases']
    m['resume_amendment'].update(completed_batch_sizes=[len(row['cases']) for row in prefix],
        completed_batches=completed, completed_visits=prefix[-1]['training_episodes'], completed_actor_updates=2*completed)
    m['training_subset'] = make_amendment(parent, prefix, binding(selection_path), tmp_path)
    return parent, m, selected, old


def test_selection_is_unique_stratified_order_independent_and_score_free(tmp_path):
    parent, m, selected, _ = subset_study(tmp_path)
    rows = selected['cases']
    assert len(rows) == len({r['path'] for r in rows}) == 240
    assert Counter(r['distribution'] for r in rows) == dict(iid=192, ood_stress=43, ood_scale=5)
    assert select_cases(list(reversed(parent['splits']['train'])), selected['seed']) == rows
    assert select_cases(parent['splits']['train'], selected['seed']+1) != rows
    assert all('makespan' not in row for row in rows)
    verify_selection(parent, m['training_subset']['selection'])
    verify_subset(m)


@pytest.mark.parametrize('completed', [11, 12])
def test_subset_retains_history_then_runs_eight_complete_384_visit_epochs(tmp_path, completed):
    parent, m, selected, old = subset_study(tmp_path, completed)
    plan = training_schedule(m)
    assert plan[:completed] == old[:completed]
    assert [len(row['cases']) for row in plan[completed:]] == [240,144]*8
    limits = budget(m)
    assert limits['global_batches'] == completed+16
    assert limits['actor_updates'] == 2*completed+32
    assert limits['training_episodes'] == old[completed-1]['training_episodes']+3072
    assert limits['epoch_end_batches'] == list(range(completed+2,completed+17,2))
    seen_ids = set()
    for epoch in range(8):
        group = plan[completed+epoch*2:completed+epoch*2+2]
        cases = [c for row in group for c in row['cases']]
        counts = Counter(c['path'] for c in cases)
        assert len(cases) == 384 and len(counts) == 240
        assert all(counts[c['path']] == (1 if c['distribution']=='iid' else 4) for c in cases)
        ids = [v for row in group for v in row['visit_ids']]
        assert len(set(ids)) == 384 and not set(ids) & seen_ids
        seen_ids.update(ids)
        assert group[-1]['training_episodes'] == epoch_episodes(m, epoch+1)
    first = epoch_rows(selected, 0)[:240]
    assert plan[completed]['cases'] == [r[0] for r in first]
    assert plan[completed]['seeds'] == [r[1] for r in first]


def test_later_restart_keeps_trained_subset_prefix_and_remaining_seed_order(tmp_path):
    _, m, _, _ = subset_study(tmp_path)
    before = training_schedule(m)
    n = m['training_subset']['inherited_batches']+1
    after = copy.deepcopy(m)
    after['resume_amendment'].update(completed_batch_sizes=[len(r['cases']) for r in before[:n]],
        completed_batches=n, completed_visits=before[n-1]['training_episodes'], completed_actor_updates=2*n)
    after['recipe']['global_batch'] = 128
    after['recipe']['microbatch'] = 128
    plan = training_schedule(after)
    assert plan[:n] == before[:n]
    flat = lambda rows: [(c['path'],s,v) for row in rows for c,s,v in zip(row['cases'],row['seeds'],row['visit_ids'])]
    assert flat(plan) == flat(before)
    assert [len(r['cases']) for r in plan[n:n+2]] == [128,16]


def test_subset_rejects_changed_selection_prefix_and_cursor(tmp_path):
    _, m, _, _ = subset_study(tmp_path)
    wrong = copy.deepcopy(m)
    wrong['resume_amendment']['completed_visits'] += 1
    with pytest.raises(ValueError, match='cursor'):
        training_schedule(wrong)
    wrong = copy.deepcopy(m)
    wrong['splits']['train'].pop()
    with pytest.raises(ValueError, match='schedule identity'):
        training_schedule(wrong)
    prefix = Path(m['training_subset']['prefix_schedule']['path'])
    prefix.write_text('[]')
    with pytest.raises(ValueError, match='evidence changed'):
        training_schedule(m)


def test_subset_rejects_held_out_overlap(tmp_path):
    parent, m, selected, _ = subset_study(tmp_path)
    parent['splits']['validation'] = [selected['cases'][0]]
    parent_path = tmp_path/'bad_parent.json'
    atomic_json(parent_path, parent)
    selection_path = tmp_path/'bad_selection.json'
    create_selection(parent_path, selection_path, selected['seed'])
    with pytest.raises(ValueError, match='held-out'):
        verify_selection(parent, binding(selection_path))


def test_subset_reuses_exact_original_baseline_coverage(tmp_path):
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import read_baseline, SOURCE_SHA
    parent, m, _, _ = subset_study(tmp_path)
    costs = {c['path']:100. for rows in parent['splits'].values() for c in rows}
    baseline = dict(manifest_sha256=m['manifest_sha256'], source_sha256=SOURCE_SHA,
                    costs=costs, costs_sha256=digest_json(costs))
    path = tmp_path/'baselines.json'
    atomic_json(path, baseline)
    assert read_baseline(m) == baseline
    selected_costs = {c['path']:100. for rows in m['splits'].values() for c in rows}
    atomic_json(path, dict(baseline, costs=selected_costs, costs_sha256=digest_json(selected_costs)))
    with pytest.raises(ValueError, match='Baseline identity/coverage'):
        read_baseline(m)
    costs['unregistered'] = 100.
    atomic_json(path, dict(baseline, costs=costs, costs_sha256=digest_json(costs)))
    with pytest.raises(ValueError, match='Baseline identity/coverage'):
        read_baseline(m)


def test_subset_epoch_reconciliation_uses_offset_visits_not_legacy_960(monkeypatch, tmp_path):
    import onpolicy.utils.stage3_b_shared_b0 as contract
    _, m, _, _ = subset_study(tmp_path)
    submissions = []
    monkeypatch.setattr(contract, 'read_commit', lambda m,p: dict(checkpoint=str(p)+'.pt'))
    monkeypatch.setattr(contract, 'submit_evaluation',
        lambda m,p,episodes,split,**kw: submissions.append((episodes,split,kw.get('decoder','H'))))
    contract.reconcile_epoch_requests(m, budget(m)['epoch_end_batches'][3])
    assert len(submissions) == 9
    assert [r[0] for r in submissions if r[1]=='tune'] == [976,1360,1744,2128]
    assert submissions[-1] == (2128,'validation','AR')


def test_subset_final_report_uses_real_epoch_endpoints_and_total(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import onpolicy.scripts.train.run_stage3_b_shared_b0 as controller
    _, m, _, _ = subset_study(tmp_path)
    m.update(source=dict(path='B0'), protocol='stage3_b_shared_b0_native_v1')
    plan = training_schedule(m)
    (tmp_path/'commits').mkdir()
    (tmp_path/'validator').mkdir()
    atomic_json(tmp_path/'baselines.json',dict(costs={'case':100.}))
    for i,row in enumerate(plan,1):
        update = tmp_path/f'update_{i}.json'
        atomic_json(update,dict(seconds=1.))
        atomic_json(tmp_path/'commits'/f'batch_{i:04d}.json',dict(checkpoint=f'checkpoint_{i}',
            checkpoint_sha256=f'sha_{i}',update=str(update),training_episodes=row['training_episodes']))
    monkeypatch.setattr(controller,'read_commit',lambda m,p:read_json(p))
    monkeypatch.setattr(controller,'risk_pass',lambda s:True)
    calls=[]
    def evaluation(m,ident):
        calls.append(ident)
        if ident!='B0_AR_validation':
            assert int(ident.split('_')[0][1:]) in {epoch_episodes(m,e) for e in range(1,9)}
        return [dict(case_id='case',makespan=100.)]
    monkeypatch.setattr(controller,'evaluation',evaluation)
    monkeypatch.setattr(controller,'paired_summary',lambda *a:dict(makespan=100.,gain_fraction=0.))
    monkeypatch.setattr(controller,'choose_candidate',lambda curves:dict(selected=curves[0],
        risk_qualified=True,open_confirmation=False,last_two_epochs_stable=False))
    monkeypatch.setattr(controller,'submit_evaluation',lambda *a,**kw:None)
    controller.finish(m,SimpleNamespace(drain=lambda:None),SimpleNamespace(update=lambda **kw:None))
    result=read_json(tmp_path/'result.json')
    assert result['budget']['training_episodes']==3664
    assert result['budget']['subset_training_episodes']==3072
    assert result['budget']['inherited_training_episodes']==592
    assert result['curves'][0]['episodes']==976 and result['curves'][-1]['episodes']==3664
    assert 'e002128_AR_validation' in calls and 'e003664_AR_validation' in calls
