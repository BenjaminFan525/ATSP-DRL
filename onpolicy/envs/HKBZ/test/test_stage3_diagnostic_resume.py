import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from onpolicy.utils.stage3_research import (atomic_json, best_of_n, digest_file,
    digest_json, read_json, trajectory_seed, validate_sample_trajectories)


def test_snapshot_does_not_follow_shared_workspace_edits(tmp_path):
    from onpolicy.scripts.train.prepare_stage3_diagnostic_resume import snapshot_sources
    workspace, snapshot = tmp_path / "workspace", tmp_path / "snapshot"
    source = workspace / "onpolicy/module.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    excluded = workspace / "onpolicy/dataset/data.json"
    excluded.parent.mkdir()
    excluded.write_text("{}")
    code = snapshot_sources(workspace, snapshot,
        listing=["onpolicy/module.py", "onpolicy/dataset/data.json"])
    source.write_text("value = 2\n")
    assert (snapshot / "onpolicy/module.py").read_text() == "value = 1\n"
    assert not (snapshot / "onpolicy/dataset").exists()
    assert code["files"] == {"onpolicy/module.py": digest_file(snapshot / "onpolicy/module.py")}
    with pytest.raises(FileExistsError):
        snapshot_sources(workspace, snapshot, listing=["onpolicy/module.py"])
    with pytest.raises(ValueError, match="Unsafe"):
        snapshot_sources(workspace, tmp_path / "unsafe", listing=["../escape.py"])


def isolated_manifest(tmp_path, monkeypatch):
    import onpolicy.utils.stage3_research as protocol
    snapshot, workspace = tmp_path / "snapshot", tmp_path / "workspace"
    snapshot.mkdir()
    workspace.mkdir()
    for directory in (snapshot, workspace):
        (directory / "module.py").write_text("original")
    checkpoint = tmp_path / "c0.pt"
    checkpoint.write_bytes(b"c0")
    input_path = tmp_path / "original_manifest.json"
    input_path.write_text("{}")
    monkeypatch.setattr(protocol, "ROOT", snapshot)
    manifest = {"source": {"path": str(checkpoint), "sha256": digest_file(checkpoint)},
        "code": {"files": {"module.py": digest_file(snapshot / "module.py")}},
        "splits": {}, "contract": {}, "contract_sha256": digest_json({}),
        "execution": {"code_root": str(snapshot), "workspace_root": str(workspace),
            "workspace_code_policy": "advisory", "input_files": {str(input_path): digest_file(input_path)}}}
    return protocol, manifest, snapshot, workspace


def test_workspace_drift_is_advisory_but_execution_code_is_checked(tmp_path, monkeypatch):
    protocol, manifest, snapshot, workspace = isolated_manifest(tmp_path, monkeypatch)
    assert protocol.verify_protocol(manifest) == {}
    (workspace / "module.py").write_text("baseline experiment edit")
    assert "module.py" in protocol.verify_protocol(manifest)
    (workspace / "module.py").unlink()
    assert protocol.verify_protocol(manifest)["module.py"]["actual"] is None
    (snapshot / "module.py").write_text("snapshot corruption")
    with pytest.raises(ValueError, match="Frozen implementation"):
        protocol.verify_protocol(manifest)


def test_checkpoint_and_experiment_inputs_remain_strict(tmp_path, monkeypatch):
    protocol, manifest, _, _ = isolated_manifest(tmp_path, monkeypatch)
    path = Path(manifest["source"]["path"])
    path.write_bytes(b"changed c0")
    with pytest.raises(ValueError, match="C0"):
        protocol.verify_protocol(manifest)
    path.write_bytes(b"c0")
    Path(next(iter(manifest["execution"]["input_files"]))).write_text("changed manifest")
    with pytest.raises(ValueError, match="experiment input"):
        protocol.verify_protocol(manifest)


def case(name):
    return {"name": name, "path": "/cases/" + name, "content_sha256": name, "profile": "iid"}


def trajectories(record, count=32):
    return [{"case_id": record["path"], "completed": True, "makespan": 90., "steps": 1,
        "actions": [[[1, 2, 3]]], "seed": trajectory_seed(20260909, record["content_sha256"], 0, k)}
        for k in range(count)]


@pytest.mark.parametrize("corruption", ("seed", "case", "partial", "actions", "incomplete"))
def test_recovery_rejects_invalid_sample_prefixes(corruption):
    record = case("a")
    saved = trajectories(record)
    if corruption == "seed":
        saved[0]["seed"] += 1
    elif corruption == "case":
        saved[0]["case_id"] = "wrong"
    elif corruption == "partial":
        saved.pop()
    elif corruption == "actions":
        saved[0]["actions"] = []
    else:
        saved[0]["completed"] = False
    with pytest.raises(ValueError):
        validate_sample_trajectories(saved, record, complete=True)


def test_sampling_resumes_complete_cases_and_committed_groups(tmp_path, monkeypatch):
    from onpolicy.scripts.train import stage3_diagnostic_worker as worker
    first, second = case("a"), case("b")
    output = tmp_path / "sample"
    saved = {"case_id": first["path"], "profile": "iid", "source_cost": 100.,
        **best_of_n([90.] * 32, 100.), "trajectories": trajectories(first)}
    atomic_json(output / "cases/a.json", saved)
    original_sha = digest_file(output / "cases/a.json")
    atomic_json(output / "partial/b.json", {"case_content_sha256": second["content_sha256"],
        "trajectories": trajectories(second, 8)})
    calls = []

    def rollout(cases, seeds, **kwargs):
        calls.append((cases, seeds))
        by_seed = {row["seed"]: row for row in trajectories(second)}
        return [by_seed[seed] for seed in seeds]

    monkeypatch.setattr(worker, "engine", lambda **kwargs: SimpleNamespace(rollout=rollout, close=lambda: None))
    manifest = {"resume": {"authorized": True}, "splits": {"train_diag32": [first, second]},
        "source_costs": {first["path"]: 100., second["path"]: 100.}, "code": {"sha256": "copy"},
        "diagnostic": {"sampling_improvable_case_fraction_min": .25}}
    worker.sample(manifest, output, SimpleNamespace(update=lambda **kwargs: None))
    assert len(calls) == 3  # Only the 24 missing trajectories; not 64 from scratch.
    assert all(row == second for batch, _ in calls for row in batch)
    assert digest_file(output / "cases/a.json") == original_sha
    assert len(read_json(output / "partial/b.json")["trajectories"]) == 32
    assert read_json(output / "result.json")["passed"]


def test_sampler_commits_before_next_group_can_fail(tmp_path, monkeypatch):
    from onpolicy.scripts.train import stage3_diagnostic_worker as worker
    record = case("a")
    calls = 0

    def rollout(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interruption")
        return trajectories(record, 8)

    monkeypatch.setattr(worker, "engine", lambda **kwargs: SimpleNamespace(rollout=rollout, close=lambda: None))
    manifest = {"splits": {"train_diag32": [record]}, "source_costs": {record["path"]: 100.},
        "code": {"sha256": "copy"}}
    with pytest.raises(RuntimeError, match="interruption"):
        worker.sample(manifest, tmp_path, SimpleNamespace(update=lambda **kwargs: None))
    assert len(read_json(tmp_path / "partial/a.json")["trajectories"]) == 8
    assert not (tmp_path / "cases/a.json").exists()


def test_completed_phase_skip_never_starts_process(monkeypatch):
    from onpolicy.scripts.train.run_stage3_diagnostic_suite import Scheduler
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.resume = {"completed_phases": ["counterfactual"]}
    scheduler.jobs = {}
    scheduler.start("counterfactual", "counterfactual", 2, 100)
    assert scheduler.jobs["counterfactual"]["process"] is None
    scheduler.check = lambda heartbeat: None
    scheduler.wait(["counterfactual"], None)
    scheduler.stop_job(scheduler.jobs["counterfactual"])
