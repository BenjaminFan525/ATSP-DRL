import pytest
import torch

from onpolicy.utils.stage3_b_shared_b0 import training_schedule, schedule, budget, episodes_at_cursor
from onpolicy.utils.stage3_b0_restart import MODE, require_waiver, rebind_checkpoint


def cases():
    return [dict(path=f'{kind}/{i}', content_sha256=f'{kind}-{i}', distribution=kind)
            for kind, count in [('iid',480),('ood_stress',108),('ood_scale',12)] for i in range(count)]


def continuation():
    return dict(recipe=dict(global_batch=64,seed=42),splits=dict(train=cases()),
        resume_amendment=dict(completed_batch_sizes=[32]*9, completed_batches=9,
                              completed_visits=288,completed_actor_updates=18))


def test_rebatch_continuation_preserves_committed_prefix_and_every_unvisited_seed():
    m=continuation(); old=schedule(m['splits']['train'],42,32); new=training_schedule(m)
    assert new[:9]==old[:9]
    flatten=lambda rows:[(c['path'],s,v) for row in rows for c,s,v in zip(row['cases'],row['seeds'],row['visit_ids'])]
    assert flatten(new)==flatten(old)
    assert len(new)==125 and len(new[9]['cases'])==64 and len(new[19]['cases'])==32
    assert sum(len(row['cases']) for row in new[9:])==7392
    for i,row in enumerate(new,1):assert episodes_at_cursor(m,i)==row['training_episodes']
    assert budget(m)['epoch_end_batches']==[20,35,50,65,80,95,110,125]
    assert budget(m)['actor_updates']==250


@pytest.mark.parametrize('field,value',[('completed_visits',320),('completed_batches',10),
    ('completed_actor_updates',20),('completed_batch_sizes',[512,512])])
def test_rebatch_rejects_inconsistent_or_cross_epoch_history(field,value):
    m=continuation();m['resume_amendment'][field]=value
    with pytest.raises(ValueError):budget(m)


def test_equivalence_waiver_requires_explicit_manifest_authorization():
    with pytest.raises(ValueError):require_waiver(continuation())
    m=continuation();m['resume_amendment'].update(mode=MODE,
        numerical_equivalence_gate='waived_by_user',user_instruction='resume preferred throughput configuration')
    assert require_waiver(m)['completed_batches']==9


def test_rebind_changes_only_identity_and_rejects_diagnostic_checkpoint():
    p=dict(actor_optim=dict(state={0:dict(exp_avg=torch.tensor([.1]))}),
        critic_optim=dict(state={0:dict(exp_avg=torch.tensor([.2]))}),
        rng_cuda=torch.tensor([1,2],dtype=torch.uint8),diagnostic_only=False,
        manifest_sha256='old',recipe_sha256='old-recipe',next_batch=9,policy_updates=18)
    result=rebind_checkpoint(p,'new',dict(global_batch=64))
    assert p['manifest_sha256']=='old' and result['manifest_sha256']=='new'
    assert result['actor_optim'] is p['actor_optim'] and result['rng_cuda'] is p['rng_cuda']
    assert result['next_batch']==9 and result['policy_updates']==18
    with pytest.raises(ValueError):rebind_checkpoint(dict(p,diagnostic_only=True),'new',{})
