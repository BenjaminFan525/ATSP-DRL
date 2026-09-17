"""Parallel preflight must retain canonical batches and immutable evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onpolicy.scripts.train.run_stage3_b_shared_b0_preflight import (
    tasks, cases_for, cached_count, compare_zero_reference, destination,
    publish, validate_row, task_id, SHARDS,
)
from onpolicy.utils.stage3_b_shared_b0 import recipe, cpus
from onpolicy.utils.stage3_research import atomic_json


@pytest.fixture
def study(tmp_path):
    return dict(root=str(tmp_path), manifest_sha256='study', source={'sha256':'source'}, recipe=recipe(),
        splits={split:[dict(path=f'{split}/{i}',content_sha256=f'{split}-{i}') for i in range(n)]
                for split,n in (('train',600),('validation',120),('tune',60))})


def row(m,t,c):
    return dict(binding=dict(manifest_sha256=m['manifest_sha256'],source_sha256=m['source']['sha256'],
        decoder='H',native=t['kind']=='baseline',batch_contract=m['recipe']['evaluation_batch_contract'],
        evaluation_cudnn_allow_tf32=True),case_sha256=c['content_sha256'],
        result=dict(case_id=c['path'],completed=True,makespan=123.,steps=10,seed=1,tau=.3,
            history=m['recipe']['history'],decoder='H',policy_updates=0,
            behavior_deterministic=True,forced_replay=False,actions_sha256='a'*64,history_sha256='b'*64))


def test_two_shards_preserve_every_original_case_and_batch(study):
    all_tasks=tasks(study)
    assert len(all_tasks)==80
    assert sum(len(cases_for(study,t)) for t in all_tasks)==960
    a,b=all_tasks[::2],all_tasks[1::2]
    assert not {task_id(t) for t in a}&{task_id(t) for t in b}
    assert cpus(SHARDS[0]).isdisjoint(cpus(SHARDS[1]))
    assert cpus(SHARDS[0])|cpus(SHARDS[1])|cpus(study['recipe']['controller_cpus'])==cpus(study['recipe']['cpus'])
    with pytest.raises(ValueError,match='boundary'):
        cases_for(study,dict(kind='baseline',split='tune',start=1))


def test_partial_batch_reuse_keeps_global_case_indices_and_refuses_overwrite(study):
    t=dict(kind='baseline',split='tune',start=24)
    cases=cases_for(study,t);rows=[row(study,t,c) for c in cases]
    atomic_json(destination(study,t,3),rows[3],overwrite=False)
    assert cached_count(study,t)==1
    common=SimpleNamespace(atomic_json=atomic_json)
    publish(study,t,rows,common)
    assert cached_count(study,t)==12
    assert destination(study,t,3).name=='0027.json'
    publish(study,t,rows,common)
    changed=copy.deepcopy(rows);changed[3]['result']['makespan']+=1
    with pytest.raises(ValueError,match='replace'):
        publish(study,t,changed,common)
    assert json.loads(destination(study,t,3).read_text())==rows[3]


def test_zero_requires_exact_cost_steps_actions_history(study):
    native=dict(kind='baseline',split='tune',start=0)
    zero={**native,'kind':'zero'}
    cases=cases_for(study,native)
    rows=[row(study,zero,c) for c in cases]
    for i,c in enumerate(cases):atomic_json(destination(study,native,i),row(study,native,c))
    assert compare_zero_reference(study,zero,rows,required=True)==12
    for key,value in (('makespan',124.),('steps',11),('actions_sha256','c'*64),('history_sha256','d'*64)):
        changed=copy.deepcopy(rows);changed[5]['result'][key]=value
        with pytest.raises(ValueError,match=key):
            compare_zero_reference(study,zero,changed,required=True)


def test_cached_training_requires_moments_and_correct_source(study):
    t=dict(kind='baseline',split='train',start=0);c=cases_for(study,t)[0]
    value=row(study,t,c)
    with pytest.raises(ValueError,match='moments'):validate_row(study,t,c,value)
    value['normalization_moments']=[10.,-50.,300.]
    validate_row(study,t,c,value)
    value['binding']['source_sha256']='other'
    with pytest.raises(ValueError,match='identity'):validate_row(study,t,c,value)
