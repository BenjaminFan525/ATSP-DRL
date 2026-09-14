"""Versioned enlarged-batch continuation; keep T0/T1 exposure contrasts explicit."""
from collections import Counter


def phase_schedule(cases, seed, batch, start, until, batch_size=32):
    from onpolicy.utils.stage3_research import digest_json, trajectory_seed
    if batch not in ("T0", "T1") or len(cases) % 4 or not cases:
        raise ValueError("Need four-case blocks and a T0/T1 design")
    if batch_size != 32 or not 0 <= start < until or start % 8 or until % 480:
        raise ValueError("Expected a full old group and a canonical evaluation endpoint")
    ordered = sorted(cases, key=lambda c: digest_json([seed, "capacity-phase-case-order", c["content_sha256"]]))
    result, episode, block_index = [], start, 0
    # End a full balanced exposure block at every 480-trajectory evaluation.
    for endpoint in range((start // 480 + 1) * 480, until + 1, 480):
        while episode < endpoint:
            exposure = min(4 * batch_size, endpoint - episode)
            if exposure % 4:
                raise ValueError("Unbalanced endpoint")
            per_case = exposure // 4
            first = (4 * block_index) % len(ordered)
            block = ordered[first:first + 4]
            visit = 10000 + (until // 480) * 1000 + block_index
            all_pairs = [(case, replica) for replica in range(per_case) for case in block]
            for update in range(4):
                pairs = ([(block[update], replica) for replica in range(per_case)] if batch == "T0"
                         else all_pairs[update * per_case:(update + 1) * per_case])
                episode += len(pairs)
                result.append({"macroblock": block_index, "training_episodes": episode,
                    "cases": [case for case, _ in pairs],
                    "seeds": [trajectory_seed(seed, case["content_sha256"], visit, replica) for case, replica in pairs]})
            block_index += 1
    if episode != until:
        raise ValueError("Continuation schedule did not reach the exact budget")
    return result


def capacity_gate(normal, dense, config, dense_config):
    def passed(result):
        peaks = result.get("resources_peaks", {})
        return (result.get("completed") and result.get("full_update_measured", result.get("full_rollout_and_update_measured"))
                and result.get("post_update_kl_guard_passed")
                and peaks.get("probe_gpu_used_bytes", float("inf")) <= 20.5 * 2**30
                and peaks.get("cgroup_memory_bytes", float("inf")) <= 23 * 2**30)
    settings = ("batch_trajectories", "tbptt_steps", "performance", "cuda_memory_fraction", "checkpoint_sha256")
    return (passed(normal) and passed(dense) and dense.get("dense_case_memory_stress")
            and not normal.get("dense_case_memory_stress")
            and config.get("batch_trajectories") == 32 and config.get("tbptt_steps") in (8, 16)
            and config.get("cuda_memory_fraction") == 1.0
            and all(config.get(key) == dense_config.get(key) for key in settings))


def exposures(plan):
    return Counter((case["path"], seed) for batch in plan for case, seed in zip(batch["cases"], batch["seeds"]))
