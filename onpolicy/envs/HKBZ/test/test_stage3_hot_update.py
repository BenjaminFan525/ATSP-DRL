import copy
from pathlib import Path
from types import SimpleNamespace
import signal

import pytest

from onpolicy.utils.stage3_hot_update import (execution_identities, resume_payload_allowed,
    numa_plan, OPTIONS, validate_hot_update, identity, OVERLAYS)
from onpolicy.utils.stage3_representation import ARMS
from onpolicy.utils.stage3_research import atomic_json, digest_json, read_json, digest_file
from onpolicy.scripts.train import hot_update_stage3_representation as hot


def resume_fixture():
    manifest = dict(manifest_sha256="new",code={"sha256":"new-code"},source={"sha256":"C0"},
        arms=ARMS,training={"seed":3},hot_update={"parent_execution_identities":{"old":"old-code"}})
    payload = dict(training_episodes=480,next_group=60,arm="E1_T1",source_sha256="C0",seed=3,
        schedule_sha256=digest_json([]),batch_composition="T1",optimizer_recipe="O0",protocol_sha256="old")
    return manifest,payload


def test_explicit_old_new_identity_and_intrascreen_resume_only():
    manifest,payload = resume_fixture()
    assert execution_identities(manifest) == {"old":"old-code","new":"new-code"}
    assert resume_payload_allowed(payload,manifest,"E1_T1",960,[]) == "old"
    for episodes in (240,480,720):
        candidate = {**payload,"training_episodes":episodes,"next_group":episodes//8}
        assert resume_payload_allowed(candidate,manifest,"E1_T1",960,[]) == "old"
    for endpoint in (1440,1920):
        with pytest.raises(ValueError):
            resume_payload_allowed(payload,manifest,"E1_T1",endpoint,[])
        candidate = {**payload,"training_episodes":endpoint-480,"next_group":(endpoint-480)//8}
        assert resume_payload_allowed(candidate,manifest,"E1_T1",endpoint,[]) == "old"


@pytest.mark.parametrize("field,value",[("training_episodes",488),("next_group",59),
    ("diagnostic_only",True),("not_resumable_partial_group",True),("forbidden_as_rl_initialization",True),
    ("source_sha256","other"),("arm","E0_T0"),("seed",4),("schedule_sha256","other"),
    ("batch_composition","T0"),("optimizer_recipe","O1"),("protocol_sha256","unknown")])
def test_resume_rejects_partial_or_wrong_scientific_lineage(field,value):
    manifest,payload = resume_fixture()
    payload[field] = value
    with pytest.raises(ValueError):
        resume_payload_allowed(payload,manifest,"E1_T1",960,[])


def test_numa_no_overlapping_physical_cores_and_single_shared_pool():
    plan = numa_plan([[i,i+64] for i in range(64)],[{"index":i} for i in range(8)])
    occupied = set(plan["controller"])
    for row in plan["trainers"]+plan["validators"]:
        assert not occupied.intersection(row["cpus"])
        assert {cpu%64//32 for cpu in row["cpus"]} == {row["gpu"]//4}
        occupied.update(row["cpus"])
    assert occupied == set(range(128))
    assert [v["gpu"] for v in plan["validators"]] == [0,4]
    with pytest.raises(ValueError):
        numa_plan([[i] for i in range(64)],[{"index":i} for i in range(8)])


def test_checkpoint_selection_never_rewinds_committed_updates(tmp_path):
    atomic_json(tmp_path/"updates/group_0060.json",{"training_episodes":480})
    checkpoint = tmp_path/"models/episodes_000480.pt"
    checkpoint.parent.mkdir(); checkpoint.touch()
    assert hot.checkpoint_candidate(tmp_path) == checkpoint
    atomic_json(tmp_path/"updates/group_0061.json",{"training_episodes":488})
    assert hot.checkpoint_candidate(tmp_path) is None
    atomic_json(tmp_path/"updates/group_0120.json",{"training_episodes":960})
    (checkpoint.parent/"episodes_000960.pt").touch()
    assert hot.checkpoint_candidate(tmp_path) is None


def test_migration_pause_race_resumes_old_worker_without_mutation(tmp_path,monkeypatch):
    suite = hot.RollingSuite.__new__(hot.RollingSuite)
    calls = []
    process = SimpleNamespace(alive=lambda:True,pause=lambda:calls.append("pause"),send=lambda sig:calls.append(sig))
    monkeypatch.setattr(hot,"checkpoint_candidate",lambda _:tmp_path/"episodes_000480.pt")
    monkeypatch.setattr(hot,"latest_committed",lambda _:488)
    assert not suite.migrate("E0_T0",dict(process=process,output=tmp_path),None)
    assert calls == ["pause",signal.SIGCONT]


def test_gpu_gate_failure_resumes_legacy_without_stopping_any_trainer(tmp_path,monkeypatch):
    # Real exception path, without real OS signals or GPU work.
    from contextlib import contextmanager
    calls = []
    suite = hot.RollingSuite.__new__(hot.RollingSuite)
    suite.m = {}; suite.root = tmp_path; suite.attempt = tmp_path/"attempt"
    suite.manifest_path = tmp_path/"manifest.json"
    suite.upgrade = {"attempt_id":"test"}; suite.irreversible = False
    suite.legacy = SimpleNamespace(pid=99,pause=lambda:calls.append("pause"),send=lambda sig:calls.append(sig))
    suite.stop_owned = lambda:calls.append("owned_cleanup")
    suite.resources = lambda:None
    suite.canary = lambda hb:(_ for _ in ()).throw(RuntimeError("GPU gate failed"))
    monkeypatch.setattr(hot,"verify",lambda _:None)
    class Heartbeat:
        def __init__(self,*a,**k): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
    monkeypatch.setattr(hot,"ProgressHeartbeat",Heartbeat)
    atomic_json(tmp_path/"status.json",{"status":"running","pid":99})
    with pytest.raises(RuntimeError,match="GPU gate failed"):
        suite.run()
    assert calls == ["pause",signal.SIGCONT,"owned_cleanup"]
    assert read_json(tmp_path/"status.json")["pid"] == 99
    assert not read_json(suite.attempt/"activation_failure.json")["irreversible"]


def test_hot_update_preserves_scientific_fields_and_parent_hash(tmp_path):
    prior = dict(manifest_sha256="",root=str(tmp_path),code={"files":{},"sha256":digest_json({})},
        execution={"code_root":str(tmp_path/"old"),"python":"python"},training={"seed":3},
        created_unix=1,material_passport={})
    prior["manifest_sha256"] = identity(prior)
    parent = tmp_path/"parent.json"; atomic_json(parent,prior)
    evidence = tmp_path/"evidence.json"; atomic_json(evidence,{"passed":True})
    manifest = copy.deepcopy(prior)
    manifest["execution"]["code_root"] = str(tmp_path/"new")
    manifest["hot_update"] = dict(parent_manifest=str(parent),parent_manifest_file_sha256=digest_file(parent),
        options=OPTIONS,parent_execution_identities=execution_identities(prior),
        cpu_verification={"path":str(evidence),"sha256":digest_file(evidence)})
    manifest["manifest_sha256"] = identity(manifest)
    assert validate_hot_update(manifest) == prior
    manifest["training"]["seed"] = 4; manifest["manifest_sha256"] = identity(manifest)
    with pytest.raises(ValueError,match="scientific"):
        validate_hot_update(manifest)
