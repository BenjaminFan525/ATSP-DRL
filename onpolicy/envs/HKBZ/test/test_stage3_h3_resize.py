"""Full-state preservation and strict width-only migration of real checkpoints."""
import copy
from pathlib import Path

import pytest
import torch

from onpolicy.utils.stage3_h3_resize import rebind_payload, import_committed_state, verify_recipe_change
from onpolicy.utils.stage3_h3_continuation import schedule
from onpolicy.utils.stage3_h3_frozen import bind
from onpolicy.utils.stage3_research import read_json, digest_json

RUN = Path(__file__).resolve().parents[4]/'result/hkbz_train_logs/stage3_h3_single_parent_20260918_r2_env192_gpu0'


@pytest.fixture
def native(tmp_path):
    if not (RUN/'arms/C03/commits/epoch_0001.json').exists():
        pytest.skip('Local committed continuation fixture absent')
    old = read_json(RUN/'arms/C03/manifest.json')
    new = copy.deepcopy(old)
    new['root'] = str(tmp_path/'arms/C03')
    new['recipe'].update(execution_profile='single_parent_env384_v1',rollout_workers=384)
    new['resume_epoch'] = 1
    new['manifest_sha256'] = digest_json({k:v for k,v in new.items() if k!='manifest_sha256'})
    commit = read_json(RUN/'arms/C03/commits/epoch_0001.json')
    payload = torch.load(commit['checkpoint']['path'],map_location='cpu',weights_only=False)
    return old,new,commit,payload


def test_real_checkpoint_rebinding_preserves_every_state_and_cursor(native,tmp_path):
    old,new,commit,payload = native
    from onpolicy.scripts.train.stage3_b_shared_b0_worker import compare_states
    suite = dict(root=str(tmp_path),recipe={'epochs':8},resume_origin=dict(epoch=1,
        arm_manifest=bind(RUN/'arms/C03/manifest.json'),
        commits=[bind(RUN/'arms/C03/commits/epoch_0001.json')]))
    mapping = import_committed_state(suite,new)
    actual = torch.load(mapping[commit['checkpoint']['sha256']]['path'],map_location='cpu',weights_only=False)
    expected = dict(payload,manifest_sha256=new['manifest_sha256'],recipe_sha256=digest_json(new['recipe']))
    compare_states(expected,actual,exact=True)
    assert actual['policy_updates']==108 and actual['next_batch']==1
    assert actual['actor_optim']['state'] and actual['critic_optim']['state']
    assert read_json(tmp_path/'resume_migration.json')['next_epoch']==2
    source = read_json(RUN/'manifest.json')
    assert schedule(source['splits']['train'],old['recipe'])==schedule(source['splits']['train'],new['recipe'])


@pytest.mark.parametrize('key,value',[('next_batch',2),('policy_updates',96),
    ('diagnostic_only',True),('complete_logical_rollout',False),('train_tau',.1),
    ('manifest_sha256','foreign'),('rng_cuda',None)])
def test_partial_or_foreign_checkpoint_cannot_be_relabelled(native,key,value):
    old,new,commit,payload=native
    payload[key]=value
    with pytest.raises(ValueError):rebind_payload(payload,old,new,commit)


def test_resize_rejects_optimizer_or_seed_change(native):
    old,new,_,_=native
    new['recipe']['optimizer_shuffle_seed']+=1
    with pytest.raises(ValueError):verify_recipe_change(old['recipe'],new['recipe'])
