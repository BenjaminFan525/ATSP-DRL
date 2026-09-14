"""Recovery must never become an unregistered change of scientific protocol."""
import copy
import pytest
from onpolicy.utils.stage3_full_data_restart import assert_same_study, OVERLAYS


def parent():
    return {"root":"old", "created_unix":1, "manifest_sha256":"old-id",
        "training":{"seed":2026091201,"actor_lr":1e-5,"global_batch":32,"max_training_episodes":7680},
        "splits":{"train_full600":["case1"]}, "contract":{"H":2,"F":4,"reservation":"soft"},
        "execution":{"code_root":"old/source","python":"python3","packages":{"torch":"same"}},
        "code":{"files":{"model.py":"original",OVERLAYS[0]:"old-controller"}}}


def test_recovery_allows_only_administrative_changes():
    before=parent(); after=copy.deepcopy(before)
    after.update(root="recovery",created_unix=2,manifest_sha256="new-id",recovery={"explicit":True})
    after["execution"]["code_root"]="recovery/source"
    after["code"]["files"].update({p:"new-administration" for p in OVERLAYS})
    assert_same_study(before,after)


@pytest.mark.parametrize("field,value",[("seed",2),("actor_lr",2e-5),("global_batch",16),("max_training_episodes",960)])
def test_recovery_rejects_training_changes(field,value):
    before=parent(); after=copy.deepcopy(before); after["training"][field]=value
    with pytest.raises(ValueError,match="research protocol"): assert_same_study(before,after)


def test_recovery_rejects_physics_changes():
    before=parent(); after=copy.deepcopy(before); after["contract"]["reservation"]="hard"
    with pytest.raises(ValueError,match="research protocol"): assert_same_study(before,after)


def test_recovery_rejects_dataset_changes():
    before=parent(); after=copy.deepcopy(before); after["splits"]["train_full600"].append("case2")
    with pytest.raises(ValueError,match="research protocol"): assert_same_study(before,after)


def test_recovery_rejects_dependency_changes():
    before=parent(); after=copy.deepcopy(before); after["execution"]["packages"]["torch"]="different"
    with pytest.raises(ValueError,match="dependencies"): assert_same_study(before,after)


def test_recovery_rejects_policy_code_changes():
    before=parent(); after=copy.deepcopy(before); after["code"]["files"]["model.py"]="changed"
    with pytest.raises(ValueError,match="source"): assert_same_study(before,after)
