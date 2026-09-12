from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.nn import global_max_pool

from onpolicy.utils.stage3_numerics import (RUNTIME, configure_runtime,
    stable_global_max_pool, install_stable_pool, model_difference)


@pytest.fixture
def runtime_guard(monkeypatch):
    values = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield
    torch.use_deterministic_algorithms(values[0], warn_only=values[1])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = values[2:4]
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = values[4:]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stable_max_preserves_values_and_first_tie_gradient_with_empty_groups(dtype):
    x = torch.tensor([[3., -2., 4.], [3., -1., 4.], [-5., -1., 3.], [2., 0., 1.], [2., 0., 1.]],
                     dtype=dtype, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 2, 2])
    output = stable_global_max_pool(x, batch, size=4)
    assert torch.equal(output, global_max_pool(x, batch, size=4))
    weights = torch.arange(1, 13, dtype=dtype).view(4, 3)
    (output * weights).sum().backward()
    expected = torch.tensor([[1., 0., 3.], [0., 2., 0.], [0., 0., 0.], [7., 8., 9.], [0., 0., 0.]], dtype=dtype)
    assert torch.equal(x.grad, expected)


def test_stable_max_empty_batch_has_valid_zero_gradient():
    x = torch.empty(0, 4, requires_grad=True)
    out = stable_global_max_pool(x, torch.empty(0, dtype=torch.long), size=3)
    assert torch.equal(out, torch.zeros(3, 4))
    out.sum().backward()
    assert x.grad.shape == x.shape


def test_stable_max_unique_values_gradcheck():
    x = torch.tensor([[1.1, -2.], [3., -.5], [-5., -1.], [2., .4]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: stable_global_max_pool(value, torch.tensor([0, 1, 0, 1]), 2), (x,))


def test_install_is_opt_in_and_checkpoint_neutral():
    from onpolicy.envs.HKBZ.test.test_stage3_representation import graph_encoder
    from onpolicy.algorithms.utils.stage3_encoder import FullStage3Encoder
    encoder, graph = graph_encoder()
    before = deepcopy(encoder.state_dict())
    expected = encoder(graph)
    candidate = FullStage3Encoder(encoder, "F_PRIVATE")
    assert not hasattr(encoder, "_stage3_max_pool")
    assert install_stable_pool(SimpleNamespace(ac=SimpleNamespace(encoder=candidate))) == 3
    for full in candidate.encoders:
        assert model_difference(before, full.state_dict())["exact"]
    actual = candidate(graph)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert not hasattr(encoder, "_stage3_max_pool")


def test_runtime_rejects_wrong_recipe_and_late_cublas_change(runtime_guard, monkeypatch):
    configure_runtime(RUNTIME)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(ValueError):
        configure_runtime({**RUNTIME, "warn_only": True})
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA"):
        configure_runtime(RUNTIME)


def test_difference_report_preserves_original_threshold_diagnostics():
    difference = model_difference({"a": torch.zeros(2), "b": torch.ones(2)},
                                 {"a": torch.tensor([0., 4e-6]), "b": torch.ones(2)})
    assert difference["tensors_changed"] == 1
    assert difference["worst_tensors"][0]["parameter"] == "a"
    assert difference["max_abs_error"] > 2e-6


@pytest.mark.skipif(os.environ.get("HKBZ_STAGE3_GPU_TEST") != "1", reason="explicit free-GPU check only")
def test_full_encoder_cuda_gradient_repeatability(runtime_guard):
    from torch_geometric.data import Batch
    from onpolicy.envs.HKBZ.test.test_stage3_representation import graph_encoder
    from onpolicy.algorithms.utils.stage3_encoder import FullStage3Encoder
    from onpolicy.utils.stage3_distributed import state_digest
    configure_runtime(RUNTIME)
    encoder, graph = graph_encoder()
    candidate = FullStage3Encoder(encoder, "F_PRIVATE").cuda().eval()
    install_stable_pool(SimpleNamespace(ac=SimpleNamespace(encoder=candidate)))
    graph = Batch.from_data_list([graph, graph.clone()]).cuda()
    checksums = []
    for _ in range(4):
        candidate.zero_grad(set_to_none=True)
        outputs = candidate(graph)["role_encodings"]
        sum(v.square().sum() for role in outputs.values() for v in role.values() if v.is_floating_point()).backward()
        checksums.append(state_digest({k:p.grad for k,p in candidate.named_parameters() if p.grad is not None}))
    assert len(set(checksums)) == 1


@pytest.mark.parametrize("field", ["training", "gates", "diagnostic", "source_costs", "resources"])
def test_recovery_rejects_changed_science_or_tolerances(field):
    from onpolicy.utils.stage3_full_policy import RECOVERY_FIELDS, assert_recovery_protocol
    parent = {key: {"unchanged": True} for key in RECOVERY_FIELDS}
    child = deepcopy(parent)
    child["numerics"] = RUNTIME
    assert_recovery_protocol(parent, child)
    child[field] = {"changed": True}
    with pytest.raises(ValueError, match="protocol or tolerances"):
        assert_recovery_protocol(parent, child)
