#!/usr/bin/env python3
"""Bounded backend diagnosis using the failed run's immutable code and inputs.

No optimizer step or experimental reward is used: the same real graph and C0
weights are differentiated repeatedly under a fixed synthetic probe objective.
Only the deterministic backend setting differs between the two invocations.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("legacy", "deterministic", "deterministic_native", "deterministic_stable_pool"), required=True)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["execution"]["code_root"])
    sys.path.insert(0, str(source))
    os.environ["HKBZ_STAGE3_WORKSPACE_ROOT"] = manifest["execution"]["workspace_root"]
    if args.backend != "legacy":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    import torch
    import torch_geometric.typing
    if args.backend == "deterministic_native":
        # Diagnostic intervention only: isolates custom torch-scatter kernels.
        torch_geometric.typing.WITH_TORCH_SCATTER = False
    from torch_geometric.data import Batch
    from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
    from onpolicy.runner.shared.stage3_research_engine import seed_all, environment_config
    from onpolicy.utils.stage3_distributed import state_digest
    from onpolicy.utils.stage3_research import atomic_json, digest_file, code_changes
    from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat
    from onpolicy.scripts.train.stage3_full_policy_worker import diagnostic_batch
    from onpolicy.utils.stage3_full_policy import shard
    if args.backend == "deterministic_stable_pool":
        # Isolate only the pooling replacement while retaining the old frozen
        # encoder/optimizer implementation. Never edit the failed snapshot.
        import importlib.util
        numerical_path = Path(__file__).resolve().parents[2] / "utils/stage3_numerics.py"
        spec = importlib.util.spec_from_file_location("stage3_diagnosis_numerics", numerical_path)
        numerical = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(numerical)
        import onpolicy.algorithms.utils.gnn as gnn_module
        gnn_module.global_max_pool = numerical.stable_global_max_pool

    if code_changes(manifest["code"]["files"], source):
        raise ValueError("Diagnostic must use the intact failed execution snapshot")
    if args.output.exists():
        raise FileExistsError("No overwrite or implicit diagnostic retry")
    torch.use_deterministic_algorithms(args.backend != "legacy", warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.backend != "legacy"
    seed_all(manifest["training"]["seed"])
    local = shard(diagnostic_batch(manifest, 8), 0, 3)
    checkpoint = Path(manifest["root"]) / "contract/c0/B_SHARED/rank0/resume_start.pt"
    with ProgressHeartbeat(args.output / "status.json", phase="fixed_graph_backend_diagnosis", backend=args.backend) as hb:
        runner = RepresentationEngine("F_SHARED", source=manifest["source"]["path"], width=len(local["cases"]),
            exploration="J", diagnostics=True, performance={"defer_statistics": True, "disable_activation_checkpoint": True})
        try:
            payload = runner.resume(checkpoint, protocol_sha256=manifest["manifest_sha256"], exploration="J")
            restore = {
                "model": state_digest(payload["model"]) == state_digest(runner.policy.ac.state_dict()),
                "actor": state_digest(payload["actor_optim"]) == state_digest(runner.policy.actor_optimizer.state_dict()),
                "critic": state_digest(payload["critic_optim"]) == state_digest(runner.policy.critic_optimizer.state_dict()),
                "norms": state_digest(payload["role_value_normalizers"]) == state_digest({str(k): v.state_dict() for k,v in runner.norms.items()}),
            }
            if not all(restore.values()):
                raise RuntimeError(f"Checkpoint load itself is inexact: {restore}")
            rows = runner.pool.call([(i, "reset", environment_config(c["path"], s))
                for i, (c, s) in enumerate(zip(local["cases"], local["seeds"]))])
            graph = Batch.from_data_list([rows[i][0] for i in range(len(rows))]).to(runner.device)
            encoder = runner.policy.ac.encoder.eval()
            initial = state_digest(encoder.state_dict())
            with torch.no_grad():
                example = encoder(graph)
                probes = {k: torch.randn_like(v) for k,v in example.items() if torch.is_tensor(v) and v.is_floating_point()}
            reference, results = None, []
            for repeat in range(args.repeats):
                hb.update(event="backward_probe", repeat=repeat)
                encoder.zero_grad(set_to_none=True)
                begin = time.perf_counter()
                output = encoder(graph)
                sum((output[k] * probe).sum() for k, probe in probes.items()).backward()
                gradients = {k: p.grad.detach().cpu().clone() for k,p in encoder.named_parameters() if p.grad is not None}
                if reference is None:
                    reference = gradients
                errors = {k: float((v-reference[k]).abs().max()) for k,v in gradients.items()}
                worst = max(errors, key=errors.get)
                results.append({"repeat": repeat, "gradient_sha256": state_digest(gradients),
                    "max_gradient_abs_error": errors[worst], "worst_parameter": worst,
                    "changed_tensors": sum(not torch.equal(v,reference[k]) for k,v in gradients.items()),
                    "seconds": time.perf_counter()-begin})
                if state_digest(encoder.state_dict()) != initial:
                    raise RuntimeError("Read-only gradient probe mutated model weights/buffers")
            exact = len({r["gradient_sha256"] for r in results}) == 1
            result = {"completed": True, "backend": args.backend, "checkpoint_restore_exact": restore,
                "fixed_weight_and_input_gradient_repeat_exact": exact, "results": results,
                "code_sha256": manifest["code"]["sha256"], "checkpoint_sha256": digest_file(checkpoint),
                "diagnostic_script_sha256": digest_file(__file__), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "pyg_with_torch_scatter": torch_geometric.typing.WITH_TORCH_SCATTER,
                "stable_pool": args.backend == "deterministic_stable_pool",
                "case_ids": [c["path"] for c in local["cases"]], "scientific_performance_claim": False,
                "scope": "real initial observation graphs; fixed synthetic gradient objective, no optimizer or learning"}
            atomic_json(args.output / "result.json", result, overwrite=False)
            print(json.dumps(result), flush=True)
            if args.backend != "legacy" and not exact:
                raise RuntimeError("Strict backend did not eliminate repeated-gradient differences")
        finally:
            runner.close()


if __name__ == "__main__":
    main()
