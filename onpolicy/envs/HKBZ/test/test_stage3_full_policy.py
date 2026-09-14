from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from onpolicy.algorithms.utils.stage3_encoder import FullStage3Encoder
from onpolicy.envs.HKBZ.test.test_stage3_representation import graph_encoder
from onpolicy.utils.stage3_distributed import TrajectoryParallel, global_case_weights, state_digest
from onpolicy.utils.stage3_full_policy import ARMS, schedule, shard, resource_plan


@pytest.mark.parametrize("variant", FullStage3Encoder.variants)
def test_full_encoder_preserves_c0_and_trains_bottom(variant):
    original, graph = graph_encoder()
    before = torch.get_rng_state().clone()
    encoder = FullStage3Encoder(original, variant).eval()
    assert torch.equal(before, torch.get_rng_state())
    expected, actual = original(graph), encoder(graph)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert all(p.requires_grad for p in encoder.parameters())
    all_pointers = [p.data_ptr() for p in encoder.parameters()]
    assert len(set(all_pointers)) == len(all_pointers)
    assert not set(all_pointers) & {p.data_ptr() for p in original.parameters()}
    if variant == "F_PRIVATE":
        for role in actual["role_encodings"].values():
            assert all(torch.equal(expected[k], role[k]) for k in expected)
        output = actual["role_encodings"]["1"]
    else:
        output = actual
    sum(v.square().sum() for v in output.values() if torch.is_tensor(v)).backward()
    for i, private in enumerate(encoder.encoders):
        active = variant == "F_SHARED" or i == 1
        assert bool(any(p.grad is not None and p.grad.abs().sum() > 0 for p in private.op_embedding.parameters())) == active
        for block in private.convs:
            assert bool(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())) == active
    rebuilt = FullStage3Encoder(original, variant).eval()
    rebuilt.load_state_dict(encoder.state_dict(), strict=True)
    assert state_digest(rebuilt.state_dict()) == state_digest(encoder.state_dict())


@pytest.mark.parametrize("batch", (12, 24))
def test_schedule_matches_exposure_across_two_and_three_gpu_arms(batch):
    cases = [{"path": str(i), "content_sha256": str(i)} for i in range(120)]
    plan = schedule(cases, 2026090903, batch)
    assert len(plan) * batch == 1920
    for row in plan:
        assert len(set(c["path"] for c in row["cases"])) == 4
        expected = Counter((c["path"], s) for c, s in zip(row["cases"], row["seeds"]))
        assert all(v == 1 for v in expected.values())
        for world in (2, 3):
            gathered = Counter()
            pieces = [shard(row, rank, world) for rank in range(world)]
            ids = [[c["path"] for c in x["cases"]] for x in pieces]
            weighted = Counter()
            for rank, piece in enumerate(pieces):
                gathered.update(zip(ids[rank], piece["seeds"]))
                for case, weight in zip(ids[rank], global_case_weights(ids, rank)):
                    weighted[case] += weight / world
            assert gathered == expected
            assert all(np.isclose(value, .25) for value in weighted.values())
    assert all(any(r["training_episodes"] == e for r in plan) for e in (480, 960, 1440, 1920))


def test_all_hardware_one_shared_validator_pool_and_no_extra_seeds():
    plan = resource_plan([[i, i + 64] for i in range(64)])
    assert len(ARMS) == 3
    assert sorted(g for arm in ARMS.values() for g in arm["gpus"]) == list(range(8))
    assert len(plan["validators"]) == 2
    assert {x["gpu"] for x in plan["validators"]} <= {x["gpu"] for x in plan["trainers"].values()}
    assert all(len(x["cpus"]) == 14 for x in plan["trainers"].values())


def _collective_test(rank, world, init_path, output):
    dist.init_process_group("gloo", init_method="file://" + init_path,
                            rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        sync = TrajectoryParallel()
        cases = ["a", "a", "b", "b", "b", "c", "c", "c"]
        features = torch.arange(1., 17.).reshape(8, 2)
        index = list(range(rank, len(cases), world))
        weights, denominator = sync.objective_weights([cases[i] for i in index], len(index))
        actor = torch.nn.Parameter(torch.tensor([.2, -.3]))
        critic = torch.nn.Parameter(torch.tensor([.5]))
        # An unused parameter on one rank must not shift collective ordering.
        conditional = torch.nn.Parameter(torch.tensor([.7]))
        policy = SimpleNamespace(actor_optimizer=torch.optim.Adam([actor, conditional], lr=.01),
                                 critic_optimizer=torch.optim.Adam([critic], lr=.01))
        (features[index].mv(actor).square() * torch.tensor(weights)).sum().backward()
        if rank == 0:
            conditional.square().backward()
        (critic.square() * len(index) / denominator).sum().backward()
        stats = {"kl": rank + 1., "clip": 0., "logp": -1., "decisions": len(index), "mask_mismatch": 0}
        sync.synchronize_update(policy, stats)
        expected_actor = torch.nn.Parameter(torch.tensor([.2, -.3]))
        expected_weights = global_case_weights([cases], 0)
        (features.mv(expected_actor).square() * torch.tensor(expected_weights)).sum().backward()
        assert torch.allclose(actor.grad, expected_actor.grad, atol=1e-5)
        assert torch.allclose(critic.grad, torch.ones(1), atol=1e-6)
        assert torch.allclose(conditional.grad, torch.tensor([1.4 / world]))
        assert stats["decisions"] == 8
        torch.nn.utils.clip_grad_norm_([actor, conditional], 1.)
        policy.actor_optimizer.step()
        checksum = state_digest(policy.actor_optimizer.state_dict())
        assert len(set(sync.gather(checksum))) == 1
        Path(output, f"rank{rank}.ok").touch()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", (2, 3))
def test_real_gloo_global_objective_and_preclip_gradient_sync(tmp_path, world):
    mp.spawn(_collective_test, args=(world, str(tmp_path / "store"), str(tmp_path)),
             nprocs=world, join=True)
    assert len(list(tmp_path.glob("*.ok"))) == world


def test_checkpoint_parity_checks_real_engine_payload_keys(tmp_path):
    from onpolicy.scripts.train.run_stage3_full_policy import compare_checkpoints
    payload = {"model": {"weight": torch.ones(2)}, "actor_optim": {"state": {0: {"exp_avg": torch.ones(2)}}},
               "critic_optim": {}, "role_value_normalizers": {"0": {"mean": torch.zeros(1)}}, "policy_updates": 2}
    reference, candidate = tmp_path / "reference.pt", tmp_path / "candidate.pt"
    torch.save(payload, reference)
    torch.save(payload, candidate)
    assert compare_checkpoints(reference, candidate, 2e-6)["passed"]
    payload["actor_optim"]["state"][0]["exp_avg"] += .1
    torch.save(payload, candidate)
    assert not compare_checkpoints(reference, candidate, 2e-6)["passed"]


def test_numpy_rng_state_is_compared_without_ambiguous_array_truth():
    from copy import deepcopy
    original = np.random.get_state()
    assert state_digest(original) == state_digest(deepcopy(original))
    changed = deepcopy(original)
    changed[1][0] ^= np.uint32(1)
    assert state_digest(original) != state_digest(changed)


@pytest.mark.parametrize("phase,name", [("fit", "x"), ("train", "../x"), ("train", "/x"), ("train", "x;bad")])
def test_controller_rejects_unsafe_or_unplanned_task_identity(phase, name):
    from onpolicy.scripts.train.run_stage3_full_policy import FullSuite
    with pytest.raises(ValueError):
        FullSuite.task_key(phase, name)


def test_controller_preserves_global_rank_resume_commit(tmp_path, monkeypatch):
    from onpolicy.scripts.train.run_stage3_full_policy import FullSuite
    from onpolicy.utils.stage3_research import atomic_json
    manifest = {"root": str(tmp_path), "resources": {"plan": resource_plan([[i, i + 64] for i in range(64)])}}
    suite = FullSuite(manifest, tmp_path / "manifest.json")
    calls = []

    def start(name, phase, placement, **options):
        key = f"{phase}/{name}"
        output = tmp_path / key
        atomic_json(output / "result.json", {"completed": True})
        suite.jobs[key] = {"process": SimpleNamespace(poll=lambda: 0), "output": output, "options": options}
        calls.append({"name": name, "placement": placement, **options})
        return key

    monkeypatch.setattr(suite, "start", start)
    monkeypatch.setattr(suite, "check", lambda hb: None)
    result = suite.run_group("train", "to1440", None, arms=["C_PRIVATE"], batch_size=24, until=1440, previous=960)
    assert len(result["C_PRIVATE"]) == 3
    assert [x["placement"]["gpu"] for x in calls] == [5, 6, 7]
    assert [x["rank"] for x in calls] == [0, 1, 2]
    assert all(x["world_size"] == 3 and x["batch_size"] == 24 and x["until"] == 1440 for x in calls)
    assert all(x["resume_commit"] == tmp_path / "train/to960/C_PRIVATE/commits/episodes_000960.json" for x in calls)
