import copy
import json
from pathlib import Path

import numpy as np
import pytest

from onpolicy.utils.stage3_research import (EvaluationQueue, atomic_json, best_of_n, cost_baselines,
    digest_file, digest_json, half_cpu_topology, paired_summary, pilot_gate, stratified_cases,
    trajectory_seed)


def test_baselines_replace_not_add_and_keep_negative_advantages():
    costs = np.array([100., 110., 90., 100.])
    source = cost_baselines(costs, "source", 98.)
    np.testing.assert_array_equal(source, [98.] * 4)
    loo = cost_baselines(costs, "leave_one_out")
    np.testing.assert_allclose(loo, [100., 290/3, 310/3, 100.])
    assert (.01 * (source - costs) < 0).sum() == 3
    np.testing.assert_array_equal(cost_baselines([100., 100.], "leave_one_out") - 100, [0., 0.])


@pytest.mark.parametrize("values", ([100.], [100., float("nan")], [0., 5.], [-1., 2.]))
def test_invalid_loo_group_rejected(values):
    with pytest.raises(ValueError):
        cost_baselines(values, "leave_one_out")


def test_group_seed_and_prefix_protocol():
    seeds = [trajectory_seed(42, "abc", 0, i) for i in range(32)]
    assert len(set(seeds)) == 32
    assert seeds[0] == trajectory_seed(42, "abc", 0, 0)
    assert seeds[0] != trajectory_seed(42, "abc", 1, 0)
    result = best_of_n([100., 90., 95., 105.], 98.)
    assert result["prefix_best"] == {"1": 100., "4": 90.}
    assert result["better_than_source_fraction"] == .5


def test_half_resources_do_not_split_smt_siblings():
    topology = [(i, i % 64, (i % 64) // 32, (i % 64) // 32) for i in range(128)]
    cores = half_cpu_topology(topology)
    assert len(cores) == 32
    assert cores == [[i, i + 64] for i in range(32)]


def test_outcome_independent_case_selection():
    records = [{"distribution": "iid", "profile": str(i % 2), "content_sha256": str(i)} for i in range(20)]
    selected = stratified_cases(records, {"iid": 6}, 42)
    assert selected == stratified_cases(records[::-1], {"iid": 6}, 42)
    assert sum(r["profile"] == "0" for r in selected) == 3


def request(tmp_path, name="a"):
    checkpoint = tmp_path / "checkpoint.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"immutable checkpoint")
    cases = [{"path": "fake-case"}]
    return {"request_id": name, "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest_file(checkpoint), "cases": cases,
        "cases_sha256": digest_json(cases), "contract_sha256": "contract",
        "code_sha256": "code", "tau": .3, "seed": 42, "training_episodes": 480}


def test_nonblocking_queue_identity_and_cache_binding(tmp_path):
    queue = EvaluationQueue(tmp_path / "queue")
    original = request(tmp_path)
    queue.submit(original)
    assert queue.poll("a") is None
    path, claimed = queue.claim()
    assert claimed["request_id"] == "a"
    assert queue.claim() is None
    queue.finish(path, {"makespan": 100})
    result = queue.poll("a")
    assert result["training_episodes"] == 480
    assert result["checkpoint_sha256"] == original["checkpoint_sha256"]
    changed = copy.deepcopy(original)
    changed["tau"] = 1.
    assert queue.identity(original) != queue.identity(changed)
    changed = copy.deepcopy(original)
    changed["code_sha256"] = "changed"
    assert queue.identity(original) != queue.identity(changed)
    with pytest.raises(FileExistsError):
        queue.submit(original)


def test_validator_crash_is_failure_not_silent_retry(tmp_path):
    queue = EvaluationQueue(tmp_path / "queue")
    queue.submit(request(tmp_path))
    queue.claim()
    queue.fail_interrupted()
    with pytest.raises(RuntimeError, match="automatic retry disabled"):
        queue.poll("a")
    assert queue.claim() is None


def test_reject_stale_checkpoint_and_unsafe_ids(tmp_path):
    queue = EvaluationQueue(tmp_path / "queue")
    stale = request(tmp_path)
    Path(stale["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        queue.submit(stale)
    bad = request(tmp_path, "../escape")
    with pytest.raises(ValueError, match="Unsafe"):
        queue.submit(bad)


def test_atomic_manifest_publication_cannot_replace_existing(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"version": 1}, overwrite=False)
    with pytest.raises(FileExistsError):
        atomic_json(path, {"version": 2}, overwrite=False)
    assert json.loads(path.read_text()) == {"version": 1}


def test_evaluation_is_paired_unique_case_not_episode_statistics():
    records = [{"case_id": "a", "makespan": 98., "completed": True, "profile": "stress_joint",
                "distribution": "ood_stress"}]
    summary = paired_summary(records, {"a": 100.})
    assert summary["gain_fraction"] == pytest.approx(.02)
    assert pilot_gate([summary, summary])
    bad = copy.deepcopy(summary)
    bad["gain_fraction"] = 0.
    assert not pilot_gate([summary, summary, bad])  # Early best does not qualify endpoint.
    with pytest.raises(ValueError, match="independent"):
        paired_summary(records * 8, {"a": 100.})


def test_authoritative_plane_history_preserves_resource_history():
    from onpolicy.runner.shared.stage3_research_engine import authoritative_history
    actions = np.zeros((2, 104, 3), dtype=np.int64) + 100
    infos = [{"last_op_indices": np.arange(104), "last_site_indices": np.arange(104) + 2} for _ in range(2)]
    history = authoritative_history(actions, infos)
    assert history[0, 2, 0] == 2
    assert history[0, 2, 1] == 4
    assert history[0, 30, 0] == 100
    assert actions[0, 2, 0] == 100


def test_recurrent_backward_mode_does_not_enable_encoder_dropout():
    import torch
    from onpolicy.runner.shared.stage3_research_engine import gradient_mode
    module = torch.nn.ModuleDict({"dropout": torch.nn.Dropout(.1), "gru": torch.nn.GRU(4, 4)})
    gradient_mode(module)
    assert not module["dropout"].training
    assert module["gru"].training
    with pytest.raises(ValueError, match="dropout"):
        gradient_mode(torch.nn.GRU(4, 4, num_layers=2, dropout=.1))


def test_real_dataset_hash_uses_canonical_utf8_json():
    from onpolicy.utils.stage3_research import TRAIN_ROOT, case_record
    case = case_record(TRAIN_ROOT / "case_0004")
    assert case["name"] == "case_0004"
    assert len(case["content_sha256"]) == 64
