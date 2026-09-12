"""Immutable three-arm full-depth study, sampling and resource contracts."""
from collections import Counter
from pathlib import Path

from onpolicy.utils.stage3_research import digest_json, digest_file, code_changes, verify_cases

ARMS = {
    "A_E2": {"encoder": "E2", "description": "frozen shared prefix, private trainable tails", "gpus": [0, 1]},
    "B_SHARED": {"encoder": "F_SHARED", "description": "full-depth shared trainable graph", "gpus": [2, 3, 4]},
    "C_PRIVATE": {"encoder": "F_PRIVATE", "description": "three full-depth private trainable graphs", "gpus": [5, 6, 7]},
}

RECOVERY_FIELDS = ("source", "source_costs", "splits", "contract", "contract_sha256", "arms", "training", "gates",
    "diagnostic", "capacity", "resources", "automatic_multi_seed", "automatic_4800", "automatic_finalblind")


def assert_recovery_protocol(parent, child):
    changed = [key for key in RECOVERY_FIELDS if child[key] != parent[key]]
    if changed:
        raise ValueError(f"Recovery changed scientific/resource protocol or tolerances: {changed}")


def identity(manifest):
    return digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def verify(manifest, inputs=True):
    from onpolicy.utils.stage3_research import ROOT
    from onpolicy.utils.stage3_numerics import RUNTIME
    import importlib.metadata
    if identity(manifest) != manifest["manifest_sha256"] or manifest["arms"] != ARMS:
        raise ValueError("Full-policy study identity changed")
    if manifest.get("numerics") != RUNTIME:
        raise ValueError("Full-policy numerical runtime must match the verified deterministic recipe")
    if Path(manifest["execution"]["code_root"]).resolve() != ROOT.resolve():
        raise ValueError("Run only the immutable full-policy source snapshot")
    if code_changes(manifest["code"]["files"], ROOT):
        raise ValueError("Frozen source snapshot changed")
    if digest_file(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("Original C0 changed")
    if digest_json(manifest["contract"]) != manifest["contract_sha256"]:
        raise ValueError("Environment contract changed")
    if inputs:
        for path, checksum in manifest["input_files"].items():
            if digest_file(path) != checksum:
                raise ValueError(f"Frozen input changed: {path}")
        for key, rows in manifest["splits"].items():
            verify_cases(rows, training=key.startswith("train"))
        for package, version in manifest["execution"]["packages"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Package version changed: {package}")
        if manifest.get("recovery"):
            from onpolicy.utils.stage3_research import read_json
            parent = read_json(manifest["recovery"]["parent_manifest"])
            assert_recovery_protocol(parent, manifest)
    return code_changes(manifest["code"]["files"], manifest["execution"]["workspace_root"])


def schedule(cases, seed, batch_size, until=1920):
    if batch_size not in (12, 24) or until % batch_size or len(cases) % 4:
        raise ValueError("Need balanced four-case batches at exact 480-episode boundaries")
    replicas = batch_size // 4
    result, visit = [], 0
    while len(result) * batch_size < until:
        ordered = sorted(cases, key=lambda c: digest_json([seed, "full-policy-cases", visit, c["content_sha256"]]))
        for first in range(0, len(ordered), 4):
            block = ordered[first:first + 4]
            pairs = [(case, replica) for case in block for replica in range(replicas)]
            result.append({"cases": [c for c, _ in pairs],
                "seeds": [int(digest_json([seed, "full-policy-replica", visit, c["content_sha256"], r])[:15], 16) % (2**31-1)
                          for c, r in pairs],
                "visit": visit, "group": len(result) + 1,
                "training_episodes": (len(result) + 1) * batch_size})
            if len(result) * batch_size == until:
                break
        visit += 1
    return result


def shard(batch, rank, world_size):
    if not 0 <= rank < world_size <= len(batch["cases"]):
        raise ValueError("Invalid trajectory partition")
    indices = list(range(rank, len(batch["cases"]), world_size))
    return {"cases": [batch["cases"][i] for i in indices],
            "seeds": [batch["seeds"][i] for i in indices], "global_indices": indices}


def trace_identity(rows):
    return digest_json(sorted((r["case_id"], r["seed"], r["action_sha256"], r["makespan"])
                              for r in rows))


def resource_plan(core_groups):
    if len(core_groups) != 64 or len({cpu for c in core_groups for cpu in c}) != 128:
        raise ValueError("This study expects 64 physical/128 logical CPUs")
    # NUMA-local placement on this 8x4090 host. SMT siblings are never split.
    result = {"trainers": {}, "validators": [], "controller": []}
    for base, gpu_base in ((0, 0), (32, 4)):
        for slot in range(4):
            cores = core_groups[base + slot * 7:base + (slot + 1) * 7]
            result["trainers"][str(gpu_base + slot)] = {
                "gpu": gpu_base + slot, "cpus": sum(cores, []), "cuda_memory_fraction": 1.0}
        result["validators"].append({"gpu": gpu_base,
            "cpus": sum(core_groups[base + 28:base + 31], []), "cuda_memory_fraction": .15})
        result["controller"].extend(core_groups[base + 31])
    cpus = result["controller"] + [c for x in result["validators"] for c in x["cpus"]]
    cpus += [c for x in result["trainers"].values() for c in x["cpus"]]
    if len(cpus) != len(set(cpus)) or len(cpus) != 128:
        raise ValueError("CPU lease overlap or missing CPUs")
    return result
