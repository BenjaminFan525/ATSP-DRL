"""Execution equivalence, not a new training experiment or seed replicate."""
import copy
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from onpolicy.algorithms.utils.stage3_encoder import Stage3Encoder
from onpolicy.envs.HKBZ.test.test_stage3_representation import graph_encoder
from onpolicy.utils.stage3_performance import GroupExecutionCache, DeferredScalars


def nested_close(a, b, atol=2e-7, rtol=2e-6):
    if isinstance(a, torch.Tensor):
        return torch.allclose(a, b, atol=atol, rtol=rtol)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(nested_close(a[k], b[k], atol, rtol) for k in a)
    if isinstance(a, (tuple,list)):
        return len(a) == len(b) and all(nested_close(x,y,atol,rtol) for x,y in zip(a,b))
    if isinstance(a, float):
        return abs(a-b) <= atol+rtol*abs(b)
    return a == b


@pytest.mark.parametrize("variant", Stage3Encoder.variants)
def test_frozen_cache_outputs_gradients_invalidation_and_budget(variant):
    base, graph = graph_encoder()
    enc = Stage3Encoder(base, variant).eval()
    ac = SimpleNamespace(encoder=enc)
    cache = GroupExecutionCache(SimpleNamespace(ac=ac, device=torch.device("cpu")), budget_mib=1)
    batch = Batch.from_data_list([graph])
    expected = enc(batch)
    def loss(out):
        return sum(v.square().sum() for k, v in out.items() if isinstance(v, torch.Tensor) and v.is_floating_point())
    if variant != "E0":
        loss(expected).backward()
    gradients = {n:p.grad.clone() for n,p in enc.named_parameters() if p.grad is not None}
    enc.zero_grad(set_to_none=True)
    with cache.group():
        first = cache.prepare_graph([graph])
        actual = cache.encode(first, actor_grad=True)
        for k,v in expected.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, actual[k])
        critic = cache.encode(cache.prepare_graph([graph]), actor_grad=False)
        assert all(not v.requires_grad for v in critic.values() if isinstance(v, torch.Tensor))
        if variant != "E0":
            loss(actual).backward()
        assert all(torch.equal(dict(enc.named_parameters())[n].grad, v) for n,v in gradients.items())
        assert not any(p.grad is not None for p in enc.prefix.parameters())
        with torch.no_grad():
            if variant != "E0":
                next(enc.tails.parameters()).add_(.01)
        after = cache.encode(cache.prepare_graph([graph]), actor_grad=True)
        fresh = enc(first)
        assert all(torch.equal(v,after[k]) for k,v in fresh.items() if isinstance(v, torch.Tensor))
        assert cache.stats["critic_reuses"] == 1
        assert cache.bytes <= cache.limit
    assert not cache.entries and cache.last_input is None and not cache.active
    assert cache.last_report["frozen_hits"] >= 1


def test_deferred_scalars_preserve_sequential_double_sum_and_counts():
    expected = 0.
    row = dict(total=0., count=0, peak=0.)
    buffer = DeferredScalars()
    for value in [1e8, 1., -1e8, .1]:
        value = torch.tensor(value, dtype=torch.float32)
        expected += float(value)
        buffer.add(row, "total", value)
        buffer.add(row, "count", torch.tensor(1))
        buffer.add(row, "peak", value.abs(), maximum=True)
    buffer.flush()
    assert row == dict(total=expected, count=4, peak=1e8)
    assert not buffer.pending


def test_cache_released_on_failure_and_no_wrong_observation_hit():
    enc, graph = graph_encoder()
    ac = SimpleNamespace(encoder=Stage3Encoder(enc,"E0").eval())
    cache = GroupExecutionCache(SimpleNamespace(ac=ac,device="cpu"),budget_mib=.00001)
    with pytest.raises(RuntimeError, match="intentional"):
        with cache.group():
            first = cache.prepare_graph([graph])
            cache.encode(first, actor_grad=True)
            different = copy.deepcopy(graph)
            different["operation"].x.add_(.1)
            assert cache.prepare_graph([different]) is not first
            assert cache.bytes == 0
            raise RuntimeError("intentional")
    assert not cache.active and not cache.entries


@pytest.mark.parametrize("variant", Stage3Encoder.variants)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_real_decoder_update_and_warm_resume_equivalence(variant, device, tmp_path):
    if device.startswith("cuda") and os.environ.get("HKBZ_STAGE3_GPU_TEST") != "1":
        pytest.skip("explicit coordinated GPU canary only")
    from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
    from onpolicy.utils.stage3_hot_update import OPTIONS
    from onpolicy.utils.stage3_research import read_json, atomic_json
    from onpolicy.utils.stage3_sampling_audit import action_digest
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    runner = RepresentationEngine(variant, width=1, exploration="J", device=device, performance=OPTIONS)
    real_call = runner.pool.call
    def fixture_call(commands):
        rows = real_call(commands)
        for index,command,_ in commands:
            if command == "summary":
                rows[index] = {**rows[index], "completed":True, "synthetic_prefix_fixture":True}
        return rows
    runner.pool.call = fixture_call
    manifest = read_json(Path(os.environ.get("HKBZ_STAGE3_TEST_PARENT_MANIFEST",
        "result/hkbz_train_logs/stage3_local_exploration_half_20260906_r1/manifest.json")))
    case = manifest["splits"]["train_fit16"][0]
    try:
        # A bounded real-observation prefix is deliberately a synthetic unit
        # fixture; its costs are not reported as experimental performance.
        trajectory = runner.rollout([case],[2026090803],retain=True,max_steps=16)[0]
        assert any(np.any((s["mask"] > 0) & (s["roles"] != 0)) for s in trajectory["states"])
        trajectory["completed"] = True
        group = []
        for index in range(8):
            item = dict(trajectory, states=trajectory["states"][:16-index%3], case_id=f"fixture{index//2}")
            group.append(item)
        sources = {t["case_id"]:t["makespan"]+100 for t in group}
        checkpoint = tmp_path/"cold.pt"
        runner.save(checkpoint,protocol_sha256="fixture",next_group=0,elite_buffer={},elite_usage={},auxiliary_steps=0)
        # Warm Adam on a BC fixture before testing a real PPO move/resume.
        runner.update(group,"bc",epochs=1)
        for t in group:
            t["policy_updates"] = runner.policy_updates
        warm = tmp_path/"warm.pt"
        runner.save(warm,protocol_sha256="fixture",next_group=0,elite_buffer={},elite_usage={},auxiliary_steps=0)
        reference_cache = runner.execution_cache
        results = []
        for fast in (False,True):
            runner.resume(warm,protocol_sha256="fixture",exploration="J")
            runner.execution_cache = reference_cache if fast else None
            runner.defer_statistics = fast
            runner.policy.ac.shared_encoder_activation_checkpoint = not fast
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            update = runner.update(group,"source",sources,epochs=2,chunk=8)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            results.append(dict(seconds=time.perf_counter()-begin,update=update,
                model=copy.deepcopy(runner.policy.ac.state_dict()),
                actor=copy.deepcopy(runner.policy.actor_optimizer.state_dict()),
                critic=copy.deepcopy(runner.policy.critic_optimizer.state_dict()),
                gradients={n:p.grad.detach().clone() for n,p in runner.policy.ac.named_parameters() if p.grad is not None},
                rng=torch.cuda.get_rng_state() if device.startswith("cuda") else torch.get_rng_state()))
        left,right = results
        error = max(float((left["model"][k]-v).abs().max()) for k,v in right["model"].items())
        assert nested_close(left["model"],right["model"],atol=2e-7,rtol=2e-6)
        assert nested_close(left["actor"],right["actor"],atol=2e-7,rtol=2e-6)
        assert nested_close(left["critic"],right["critic"],atol=2e-7,rtol=2e-6)
        assert nested_close(left["gradients"],right["gradients"],atol=2e-7,rtol=2e-6)
        assert torch.equal(left["rng"],right["rng"])
        assert nested_close(left["update"]["epochs"],right["update"]["epochs"],atol=2e-7,rtol=2e-6)
        runner.assert_optimizer_ownership()
        assert reference_cache.last_report["input_hits"] > 0
        # Sampling remains the unchanged batch-one/per-trajectory-RNG path.
        actions = []
        for fast in (False,True):
            runner.resume(warm,protocol_sha256="fixture",exploration="J")
            runner.execution_cache = reference_cache if fast else None
            sample = runner.rollout([case],[2026090803],retain=False,max_steps=16)[0]
            actions.append(action_digest(sample["actions"]))
        assert actions[0] == actions[1]
        report = dict(passed=True,variant=variant,device=device,model_max_abs_error=error,
            legacy_seconds=left["seconds"],optimized_seconds=right["seconds"],
            speedup=left["seconds"]/right["seconds"],cache=reference_cache.last_report,
            peak_reserved_gib=right["update"]["peak_reserved_gib"],
            masks_actions_rng_gradients_warm_adam_and_kl_checked=True,
            scope="deterministic 16-step real-observation synthetic unit fixture, B8 ragged; not end-to-end speedup")
        destination = os.environ.get("HKBZ_STAGE3_CANARY_REPORT_DIR")
        if destination:
            atomic_json(Path(destination)/f"{variant}_{device.replace(':','_')}.json",report,overwrite=False)
    finally:
        runner.close()
        torch.use_deterministic_algorithms(deterministic)
