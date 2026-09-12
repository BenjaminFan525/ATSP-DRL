"""Opt-in numerical contract for the full-policy study, not global baseline policy."""
import os

import torch

RUNTIME = {
    "deterministic_algorithms": True,
    "warn_only": False,
    "cublas_workspace_config": ":4096:8",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "allow_tf32": False,
    "max_pool_tie_break": "first_node_index_v1",
}


def configure_runtime(expected=None):
    if expected is not None and expected != RUNTIME:
        raise ValueError("Numerical execution recipe changed")
    workspace = RUNTIME["cublas_workspace_config"]
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace:
        raise RuntimeError("Set the cuBLAS reproducibility environment before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(RUNTIME)


def stable_global_max_pool(x, batch, size=None):
    """Same max values, deterministic first-index tie gradient, including empty groups.

    torch-scatter's CUDA max picks an argmax through atomic operations outside
    torch's deterministic-algorithm guard. Equal maxima can therefore route
    gradients to different nodes across identical backward calls. Compute the
    value/argmax without autograd, select the minimum matching node index, and
    gather from the live input. The gradient matches per-graph torch.max(dim=0)
    (one first winner), not scatter_reduce's different equal-tie averaging rule.
    """
    if x.dim() != 2 or batch.dim() != 1 or len(batch) != len(x):
        raise ValueError("Expected [nodes, features] and [nodes] batch indices")
    if size is None:
        size = int(batch.max()) + 1 if batch.numel() else 0
    n, d = x.shape
    with torch.no_grad():
        indices = batch[:, None].expand(n, d)
        maxima = x.new_zeros((size, d)).scatter_reduce_(0, indices, x.detach(),
            reduce="amax", include_self=False)
        order = torch.arange(n, device=x.device)[:, None].expand(n, d)
        winners = torch.where(x.detach() == maxima.index_select(0, batch), order, n)
        first = torch.full((size, d), n, dtype=torch.long, device=x.device)
        first.scatter_reduce_(0, indices, winners, reduce="amin", include_self=True)
    # The extra row defines zero output for missing/empty graph groups.
    padded = torch.cat((x, x.new_zeros((1, d))), dim=0)
    return padded[first, torch.arange(d, device=x.device)[None, :]]


def install_stable_pool(policy):
    from onpolicy.algorithms.utils.gnn import HeteroGraphEncoder
    count = 0
    for module in policy.ac.encoder.modules():
        if isinstance(module, HeteroGraphEncoder):
            module._stage3_max_pool = stable_global_max_pool
            count += 1
    if not count:
        raise ValueError("No Stage3 graph readout found")
    return count


def training_state(runner):
    return {"model": runner.policy.ac.state_dict(),
        "actor_optim": runner.policy.actor_optimizer.state_dict(),
        "critic_optim": runner.policy.critic_optimizer.state_dict(),
        "role_value_normalizers": {str(k): v.state_dict() for k,v in runner.norms.items()},
        "policy_updates": runner.policy_updates}


def model_difference(expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError("Model parameter/buffer identities changed")
    differences = []
    for name, before in expected.items():
        after = actual[name].detach().cpu()
        before = before.detach().cpu()
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Model tensor contract changed: {name}")
        if not torch.equal(before, after):
            differences.append({"parameter": name,
                "max_abs_error": float((before.to(torch.float64)-after.to(torch.float64)).abs().max())})
    differences.sort(key=lambda r: (-r["max_abs_error"], r["parameter"]))
    return {"exact": not differences, "tensors_changed": len(differences),
        "max_abs_error": differences[0]["max_abs_error"] if differences else 0., "worst_tensors": differences[:10]}
