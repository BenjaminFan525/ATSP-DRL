"""Protect checkpoint attribution and incomplete legacy temperature references."""
import copy
from types import SimpleNamespace

import pytest

from onpolicy.scripts.train.run_stage3_h3_canonical_tau import (
    bind, collect_records, report, validate_rows, write,
)


def fixture(tmp_path):
    cases = [dict(path=f'/case_{i:04d}', name=f'case_{i:04d}',
                  profile='balanced', distribution='iid') for i in range(120)]
    rows = [dict(case_id=c['path'], profile=c['profile'], distribution=c['distribution'],
                 tau=.3, seed=1, decoder='H', completed=True, behavior_deterministic=True,
                 makespan=100.+i, steps=3, actions_sha256=str(i), history_sha256=str(i))
            for i, c in enumerate(cases)]
    write(tmp_path/'reference.json', dict(rows=rows))
    manifest = dict(root=str(tmp_path), cases=cases, temperatures=[.3, .03, .5],
                    original_results={'0p3':bind(tmp_path/'reference.json')},
                    batch_order=list(range(10)), fixed_batch_size=12,
                    manifest_sha256='checkpoint-specific-binding')
    return manifest, rows


def group(manifest, rows, tau=.3, batch=0):
    selected = copy.deepcopy(rows[12*batch:12*batch+12])
    for row in selected:
        row['tau'] = tau
    return dict(batch_index=batch, tau=tau, rows=selected, weights_unchanged=True,
                manifest_sha256=manifest['manifest_sha256'])


def test_report_accepts_only_tau03_legacy_reference_without_inventing_other_scores(tmp_path):
    m, rows = fixture(tmp_path)
    for batch in range(10):
        for tau in m['temperatures']:
            write(tmp_path/'groups'/f'{batch}_{tau}.json', group(m, rows, tau, batch))
    summary = report(m)
    assert summary['completed'] and summary['completed_case_episodes'] == 360
    assert all(c['n'] == 120 and not any(c['temperature_mismatches'].values())
               for c in summary['comparisons'])
    import csv
    with (tmp_path/'per_case.csv').open() as f:
        record = next(csv.DictReader(f))
    assert record['legacy_makespan_0p03'] == record['legacy_makespan_0p5'] == ''
    assert float(record['legacy_makespan_0p3']) == rows[0]['makespan']


@pytest.mark.parametrize('mutation', ['foreign_checkpoint', 'wrong_case_order', 'duplicate_group', 'incomplete'])
def test_invalid_group_is_rejected_before_reporting(tmp_path, mutation):
    m, rows = fixture(tmp_path)
    value = group(m, rows)
    if mutation == 'foreign_checkpoint':
        value['manifest_sha256'] = 'another-checkpoint'
    elif mutation == 'wrong_case_order':
        value['rows'].reverse()
    elif mutation == 'incomplete':
        value['rows'][0]['completed'] = False
    write(tmp_path/'groups'/'a.json', value)
    if mutation == 'duplicate_group':
        write(tmp_path/'groups'/'b.json', value)
    with pytest.raises(ValueError):
        collect_records(m)


def test_temperature_and_forced_replay_are_not_interchangeable(tmp_path):
    m, rows = fixture(tmp_path)
    with pytest.raises(ValueError):
        validate_rows(rows, m['cases'], .03)
    rows[0]['forced_replay'] = True
    with pytest.raises(ValueError):
        validate_rows(rows, m['cases'], .3)


def test_prepare_s0_uses_bound_cached_reference_without_new_run_metadata(tmp_path, monkeypatch):
    from onpolicy.scripts.train import run_stage3_h3_canonical_tau as driver
    m, rows = fixture(tmp_path)
    source = tmp_path/'source'; source.mkdir()
    (source/'core.py').write_text('frozen = True\n')
    checkpoint = tmp_path/'s0.pt'; checkpoint.write_text('frozen S0')
    parent = tmp_path/'parent.json'; write(parent, {})
    study = tmp_path/'study.json'; write(study, {})
    selected = dict(checkpoint=bind(checkpoint), epoch=0, policy_updates=0,
                    reference=bind(tmp_path/'reference.json'))
    original = dict(root=str(tmp_path), source_root=str(source), parent=bind(parent),
                    models={'s0':selected}, cases=m['cases'], gpu_uuid='GPU-0',cpu_affinity='22')
    # A reused reference legitimately has no fresh-run weights_unchanged field.
    write(tmp_path/'results'/'s0_tau_0p3.json', dict(model='s0', checkpoint=bind(checkpoint),
                                                   tau=.3, reused=True, rows=rows))
    monkeypatch.setattr(driver,'load_original',lambda path:(original,{}))
    monkeypatch.setattr(driver,'OVERLAYS',[])
    output=tmp_path/'prepared'
    driver.prepare(SimpleNamespace(study=study,output=output,model='s0',all_trajectories=True))
    result=driver.load_manifest(output/'manifest.json')
    assert result['epoch']==0 and result['checkpoint']==bind(checkpoint)
    assert result['original_results']=={'0p3':selected['reference']}


@pytest.mark.parametrize('changed', ['policy', 'case_order', 'epoch_label', None])
def test_matrix_only_allows_driver_changes_between_checkpoints(tmp_path, changed):
    from onpolicy.scripts.train.run_stage3_h3_canonical_matrix import check_contract, LEAF
    reference, _ = fixture(tmp_path)
    reference.update(parent={'sha256':'parent'}, seed=1, gpu_uuid='GPU0',
                     cpu_affinity='22', allocator_limit_mib=4096,
                     source_files={'actor.py':'fixed-actor',LEAF:'original-driver'})
    models={name:dict(copy.deepcopy(reference),epoch=epoch)
            for name,epoch in [('s0',0),('epoch1',1),('epoch6',6)]}
    models['epoch6']['source_files'][LEAF]='extended-driver'
    if changed=='policy':models['epoch6']['source_files']['actor.py']='changed-actor'
    elif changed=='case_order':models['s0']['cases'].reverse()
    elif changed=='epoch_label':models['epoch6']['epoch']=7
    if changed:
        with pytest.raises(ValueError):check_contract(models)
    else:
        check_contract(models)
