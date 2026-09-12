"""Resource-only placement for the frozen Stage3 local-exploration protocol.

Sharing never changes rollout width, seeds, optimizer settings or evaluation
identity. Old manifests retain their dedicated-validator placement.
"""
from __future__ import annotations

import copy
import math


SHARED_VALIDATOR = {
    "mode": "colocate_if_four",
    "gpu_lane": 3,
    "trainer_cpus": "24-29,88-93",
    "validator_cpus": "30-31,94-95",
    "trainer_cuda_memory_fraction": .60,
    "validator_cuda_memory_fraction": .20,
}


def cpu_set(value):
    result = set()
    for part in value.split(","):
        ends = [int(x) for x in part.split("-")]
        if len(ends) not in (1, 2) or min(ends) < 0 or ends[-1] < ends[0]:
            raise ValueError("Invalid CPU set")
        current = set(range(ends[0], ends[-1] + 1))
        if result & current:
            raise ValueError("Duplicate CPU in resource assignment")
        result |= current
    if not result:
        raise ValueError("Empty CPU set")
    return result


def memory_fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= .80:
        raise ValueError("CUDA allocator fraction must be finite and in (0, 0.80]")
    return value


def pilot_resource_plan(resources, arm_count):
    if not 1 <= arm_count <= 4:
        raise ValueError("One to four admitted arms required")
    if resources["physical_gpus"] != [0, 1, 2, 3] or len(resources["lanes"]) != 4:
        raise ValueError("This half-resource protocol owns GPU0-3 and four CPU lanes")
    parent = cpu_set(resources["parent_cpuset"])
    lanes = [cpu_set(s) for s in resources["lanes"]]
    if set.union(*lanes) != parent or sum(map(len, lanes)) != len(parent):
        raise ValueError("CPU lanes must partition the fixed parent CPU budget")
    if len(parent) != resources["logical_cpu_count"]:
        raise ValueError("Logical CPU budget mismatch")
    config = resources.get("pilot_validator", {"mode": "dedicated"})
    if config["mode"] not in ("dedicated", "colocate_if_four"):
        raise ValueError("Unknown validator placement policy")
    shared = config["mode"] == "colocate_if_four" and arm_count == 4
    def assignment(lane):
        return {"lane": lane, "gpu": resources["physical_gpus"][lane],
                "cpuset": resources["lanes"][lane], "cuda_memory_fraction": .80}
    result = {"profile": "four_trainers_shared_validator" if shared else "dedicated_validator",
              "trainers": {i: assignment(i) for i in range(4 if shared else min(arm_count, 3))},
              "validator": assignment(3)}
    if shared:
        if config["gpu_lane"] != 3:
            raise ValueError("Only GPU3 may be shared in this protocol")
        trainer_cpus, validator_cpus = [cpu_set(config[k]) for k in ("trainer_cpus", "validator_cpus")]
        if trainer_cpus & validator_cpus or trainer_cpus | validator_cpus != lanes[3]:
            raise ValueError("Shared workers must disjointly partition the GPU3 CPU lane")
        fractions = [memory_fraction(config[k]) for k in (
            "trainer_cuda_memory_fraction", "validator_cuda_memory_fraction")]
        if sum(fractions) > .80 + 1e-12:
            raise ValueError("Shared CUDA allocator fractions exceed the existing 80% budget")
        for role, cpus, fraction in zip((result["trainers"][3], result["validator"]),
                                       (config["trainer_cpus"], config["validator_cpus"]), fractions):
            role.update(cpuset=cpus, cuda_memory_fraction=fraction)
    return copy.deepcopy(result)


def validate_coexistence(plan, phase, placement, live_jobs):
    """Allow exactly one trainer + one validator, never arbitrary GPU overbooking."""
    peers = [j for j in live_jobs if j["gpu"] == placement["gpu"]]
    if not peers:
        return
    if (plan is None or plan["profile"] != "four_trainers_shared_validator"
            or placement["lane"] != 3 or len(peers) != 1
            or {phase, peers[0]["phase"]} != {"train", "validator"}):
        raise RuntimeError("GPU lane already owned by an incompatible study worker")
    peer = peers[0]
    if (cpu_set(placement["cpuset"]) & cpu_set(peer["cpuset"])
            or placement["cuda_memory_fraction"] + peer["cuda_memory_fraction"] > .80 + 1e-12):
        raise RuntimeError("Shared workers exceed their disjoint CPU / CUDA allocator budget")


def validate_worker_placement(placement, *, affinity, visible_devices, cuda_memory_fraction):
    if set(affinity) != cpu_set(placement["cpuset"]):
        raise ValueError("Worker CPU affinity differs from its frozen placement")
    if visible_devices != str(placement["gpu"]):
        raise ValueError("Worker CUDA_VISIBLE_DEVICES differs from its frozen placement")
    if abs(memory_fraction(cuda_memory_fraction) - placement["cuda_memory_fraction"]) > 1e-12:
        raise ValueError("Worker CUDA allocator fraction differs from its frozen placement")
