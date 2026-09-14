#!/usr/bin/env python3
"""Full-policy workers: C0 contracts, capacity checks, synchronized PPO, validation."""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import gc
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
from onpolicy.runner.shared.stage3_research_engine import seed_all
from onpolicy.utils.stage3_distributed import TrajectoryParallel, state_digest
from onpolicy.utils.stage3_full_policy import ARMS, verify, schedule, shard, trace_identity
from onpolicy.utils.stage3_numerics import configure_runtime, install_stable_pool, training_state, model_difference
from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json
from onpolicy.utils.stage3_representation import RepresentationQueue
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat, save_group, trajectory_record
from onpolicy.scripts.train.stage3_local_worker import frozen_state, assert_frozen, nested_close
from onpolicy.scripts.train.stage3_representation_worker import submit, validator, max_role_kl


def engine(manifest, variant, width=8, checkpoint=None, *, validation=False):
    configure_runtime(manifest["numerics"])
    fraction = .15 if validation else 1.0
    runner = RepresentationEngine(variant, checkpoint=checkpoint, source=manifest["source"]["path"],
        width=width, exploration="J", diagnostics=True, cuda_memory_fraction=min(.8, fraction),
        performance={"cache_frozen_features": variant == "E2" and not validation,
                     "cache_mib": 2048, "defer_statistics": True, "disable_activation_checkpoint": True})
    torch.cuda.set_per_process_memory_fraction(fraction, runner.device)
    install_stable_pool(runner.policy)
    return runner


def metadata(manifest, arm, **extra):
    return {"protocol_sha256": manifest["manifest_sha256"], "source_sha256": manifest["source"]["sha256"],
        "execution_code_sha256": manifest["code"]["sha256"], "arm": arm,
        "numerics": manifest["numerics"],
        "seed": manifest["training"]["seed"], "next_group": 0, "elite_buffer": {},
        "elite_usage": {}, "auxiliary_steps": 0, **extra}


def diagnostic_batch(manifest, count, dense_case=None):
    cases = manifest["diagnostic"]["cases"]
    if dense_case is not None:
        cases = [next(c for c in manifest["splits"].get("train_full600", cases) if c["path"] == dense_case)]
    if count % len(cases):
        raise ValueError("Diagnostic batch must be balanced")
    pairs = [(c, replica) for c in cases for replica in range(count // len(cases))]
    return {"cases": [c for c, _ in pairs],
            "seeds": [int(digest_json([manifest["training"]["seed"], "full-policy-canary", c["content_sha256"], r])[:15], 16)
                      % (2**31 - 1) for c, r in pairs]}


def collect(runner, batch, hb):
    rows = []
    version = runner.policy_updates
    for first in range(0, len(batch["cases"]), runner.width):
        rows.extend(runner.rollout(batch["cases"][first:first + runner.width],
            batch["seeds"][first:first + runner.width], retain=True, heartbeat=hb))
        if runner.policy_updates != version:
            raise ValueError("Behavior policy changed during accumulation")
    return rows


def records(group):
    from onpolicy.utils.stage3_performance import tensor_bytes
    rows = []
    for trajectory in group:
        row = trajectory_record(trajectory)
        row["max_graph_tensor_bytes"] = max(tensor_bytes(s["graph"].to_dict()) for s in trajectory["states"])
        rows.append(row)
    return rows


def model_delta(before, runner):
    return max(float((v.detach().cpu() - before[k]).abs().max()) for k, v in runner.policy.ac.state_dict().items())


def contract(args, manifest, sync, hb, *, engine_fn=None, submit_fn=None):
    make_engine = engine if engine_fn is None else engine_fn
    publish = submit if submit_fn is None else submit_fn
    variant = manifest.get("arms", ARMS)[args.arm]["encoder"]
    batch = diagnostic_batch(manifest, args.batch_size, args.dense_case)
    local = shard(batch, args.rank, args.world_size)
    runner = make_engine(manifest, variant, width=min(8, len(local["cases"])))
    try:
        sync.assert_replicas(runner)
        if args.phase == "reference":
            initial = args.output / "initial_c0.pt"
            runner.save(initial, **metadata(manifest, args.arm, training_episodes=0,
                initialization_only=True, diagnostic_only=False))
            publish(manifest, initial, args.arm, 0, variant)
        if args.phase != "capacity":
            for case in manifest["diagnostic"]["cases"]:
                row = runner.rollout([case], [42], deterministic=True, heartbeat=hb)[0]
                if abs(row["makespan"] - manifest["source_costs"][case["path"]]) > 1e-6:
                    raise RuntimeError("C0 greedy changed during full-policy migration")
        begin = time.time()
        torch.cuda.reset_peak_memory_stats()
        group = collect(runner, local, hb)
        save_group(args.output / "rollout.json", group, {"arm": args.arm, "diagnostic_only": True,
                   "rank": args.rank, "world_size": args.world_size, "global_indices": local["global_indices"]})
        global_records = [r for rows in sync.gather(records(group)) for r in rows]
        rollout_seconds = time.time() - begin
        replay = runner.replay_metrics(group, hb, distributed=sync)
        if replay["max_logp_error"] > .002:
            raise RuntimeError("Full-history likelihood replay mismatch")
        start = args.output / "resume_start.pt"
        runner.save(start, **metadata(manifest, args.arm, diagnostic_only=True, training_episodes=0))
        initial_state_sha = state_digest(training_state(runner))
        frozen = frozen_state(runner)
        initial_model = {k: v.detach().cpu().clone() for k, v in runner.policy.ac.state_dict().items()}
        begin = time.time()
        update = runner.update(group, "source", manifest["source_costs"], epochs=2, chunk=8,
                               distributed=sync, heartbeat=hb)
        assert_frozen(runner, frozen)
        replica_digest = sync.assert_replicas(runner)
        update_seconds = time.time() - begin
        expected = {k: v.detach().cpu().clone() for k, v in runner.policy.ac.state_dict().items()}
        expected_actor = copy.deepcopy(runner.policy.actor_optimizer.state_dict())
        expected_critic = copy.deepcopy(runner.policy.critic_optimizer.state_dict())
        expected_norms = state_digest({str(k): v.state_dict() for k,v in runner.norms.items()})
        expected_updates = runner.policy_updates
        after = args.output / "after_update.pt"
        runner.save(after, **metadata(manifest, args.arm, diagnostic_only=True, training_episodes=args.batch_size))
        resume_report = None
        repeat_update_seconds = None
        if args.phase != "capacity":
            payload = runner.resume(start, protocol_sha256=manifest["manifest_sha256"], exploration="J")
            rng_checks = {"torch": torch.equal(torch.get_rng_state(), payload["rng_torch"]),
                "cuda": torch.equal(torch.cuda.get_rng_state(runner.device), payload["rng_cuda"]),
                "numpy": state_digest(np.random.get_state()) == state_digest(payload["rng_numpy"]),
                "python": random.getstate() == payload["rng_python"]}
            rng_ok = all(rng_checks.values())
            restored_exact = state_digest(training_state(runner)) == initial_state_sha
            if not restored_exact or not rng_ok or payload.get("numerics") != manifest["numerics"]:
                atomic_json(args.output / "resume_comparison.json", {"passed": False,
                    "restored_state_exact": restored_exact, "rng": rng_checks, "phase": "before_repeated_update"}, overwrite=False)
                raise RuntimeError("Model/Adam/ValueNorm/count/RNG checkpoint restore was not exact")
            del payload
            repeat_start = time.time()
            runner.update(group, "source", manifest["source_costs"], epochs=2, chunk=8,
                          distributed=sync, heartbeat=hb)
            repeat_update_seconds = time.time() - repeat_start
            difference = model_difference(expected, runner.policy.ac.state_dict())
            resume_error = difference["max_abs_error"]
            actor_ok = nested_close(expected_actor, runner.policy.actor_optimizer.state_dict())
            critic_ok = nested_close(expected_critic, runner.policy.critic_optimizer.state_dict())
            norms_ok = expected_norms == state_digest({str(k): v.state_dict() for k,v in runner.norms.items()})
            updates_ok = expected_updates == runner.policy_updates
            repeated_state_sha = sync.assert_replicas(runner)
            resume_ok = (rng_ok and resume_error <= 2e-6
                         and actor_ok and critic_ok and norms_ok and updates_ok)
            resume_report = {"passed": resume_ok, "restored_state_exact": restored_exact,
                "rng": rng_checks, "model": difference, "actor_optimizer_close": actor_ok,
                "critic_optimizer_close": critic_ok, "complete_repeated_state_exact": repeated_state_sha == replica_digest,
                "normalizers_exact": norms_ok, "update_count_exact": updates_ok,
                "atol_unchanged": 2e-6, "numerics": manifest["numerics"]}
            atomic_json(args.output / "resume_comparison.json", resume_report, overwrite=False)
            if not resume_ok:
                raise RuntimeError(f"Model/Adam/RNG resume mismatch: {resume_error}")
        else:
            resume_error, resume_ok = None, None
        changed = {}
        if variant != "E2":
            for i, full in enumerate(runner.policy.ac.encoder.encoders):
                for name in ["op_embedding", *[f"convs.{j}" for j in range(len(full.convs))]]:
                    prefix = f"encoder.encoders.{i}.{name}."
                    changed[f"role{i}/{name}"] = any(
                        not torch.equal(initial_model[k], v.detach().cpu())
                        for k, v in runner.policy.ac.state_dict().items() if k.startswith(prefix))
            if not all(changed.values()):
                raise RuntimeError(f"A supposedly trainable bottom layer did not update: {changed}")
        peak = torch.cuda.max_memory_reserved() / 2**30
        if max_role_kl(update) > manifest["training"]["post_update_hard_kl"] or len(update["epochs"]) != 2 or not all(
                e["actor_step_applied"] for e in update["epochs"]):
            raise RuntimeError("Canary update did not satisfy the original PPO/KL contract")
        atomic_json(args.output / "result.json", {"passed": True, "arm": args.arm, "rank": args.rank,
            "world_size": args.world_size, "batch_size": args.batch_size, "dense_case": args.dense_case,
            "initial_greedy_checked": args.phase != "capacity", "trace_identity": trace_identity(global_records),
            "records": global_records, "replay": replay, "update": update,
            "replica_state_sha256": replica_digest, "resume_passed": resume_ok, "resume_model_max_abs_error": resume_error,
            "resume_comparison": resume_report, "numerics": manifest["numerics"],
            "full_bottom_layers_changed": changed, "peak_reserved_gib": peak,
            "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
            "repeat_update_seconds": repeat_update_seconds,
            "execution_cache_equivalence": bool(getattr(runner, "compare_uncached_on_resume", False)),
            "after_update_checkpoint": str(after), "representation": runner.representation_report,
            "diagnostic_only": True, "forbidden_as_rl_initialization": True}, overwrite=False)
    finally:
        runner.close()


def train(args, manifest, sync, hb):
    root, variant = Path(manifest["root"]), ARMS[args.arm]["encoder"]
    admission_path = root / "training_admission.json"
    admission = read_json(admission_path)
    if not admission["passed"] or args.arm not in admission["eligible_arms"]:
        raise ValueError("Training admission missing")
    batch_size = admission["batch_size"]
    plan = schedule(manifest["splits"]["train_pilot120"], manifest["training"]["seed"], batch_size)
    schedule_sha = digest_json(plan)
    recipe_sha = digest_file(admission_path)
    if args.until not in (960, 1440, 1920):
        raise ValueError("Unsupported endpoint")
    if args.until > 960:
        gate, key = ("screen_admission.json", "selected") if args.until == 1440 else ("pilot_admission_1440.json", "eligible_next_chunk")
        if args.arm not in read_json(root / gate)[key]:
            raise ValueError("This arm did not pass the previous stage")
    runner = engine(manifest, variant, width=min(8, (batch_size + args.world_size - 1) // args.world_size))
    cursor, episodes = 0, 0
    queue = RepresentationQueue(root / "validator")
    try:
        if args.resume_commit:
            commit = read_json(args.resume_commit)
            if (commit["arm"] != args.arm or commit["world_size"] != args.world_size
                    or commit["schedule_sha256"] != schedule_sha or commit["recipe_sha256"] != recipe_sha):
                raise ValueError("Resume commit identity changed")
            entry = commit["ranks"][args.rank]
            if digest_file(entry["checkpoint"]) != entry["sha256"]:
                raise ValueError("Committed checkpoint changed")
            payload = runner.resume(entry["checkpoint"], protocol_sha256=manifest["manifest_sha256"], exploration="J")
            if (payload.get("diagnostic_only") or payload.get("arm") != args.arm
                    or payload.get("numerics") != manifest["numerics"]
                    or payload.get("rank") != args.rank or payload.get("world_size") != args.world_size
                    or payload.get("schedule_sha256") != schedule_sha or payload.get("recipe_sha256") != recipe_sha):
                raise ValueError("Invalid rank checkpoint for resume")
            cursor, episodes = payload["next_group"], payload["training_episodes"]
            if episodes != cursor * batch_size or episodes != commit["training_episodes"]:
                raise ValueError("Resume cursor mismatch")
            del payload
        sync.assert_replicas(runner)
        frozen = frozen_state(runner)
        for index in range(cursor, args.until // batch_size):
            sync.barrier()
            while queue.pending_count(args.arm) >= 2 or queue.pending_count() >= 8:
                hb.update(event="validator_backpressure", pending=queue.pending_count())
                time.sleep(5)
            batch = plan[index]
            local = shard(batch, args.rank, args.world_size)
            hb.update(event="group_start", training_episodes=episodes, group=index + 1, target_episodes=args.until)
            torch.cuda.reset_peak_memory_stats()
            begin = time.time()
            group = collect(runner, local, hb)
            save_group(args.output / f"rollouts/group_{index + 1:04d}.json", group,
                {"arm": args.arm, "rank": args.rank, "world_size": args.world_size,
                 "global_indices": local["global_indices"], "policy_updates": runner.policy_updates})
            update = runner.update(group, "source", manifest["source_costs"], epochs=2, chunk=8,
                                   distributed=sync, heartbeat=hb)
            if len(update["epochs"]) != 2 or not all(e["actor_step_applied"] for e in update["epochs"]) or max_role_kl(update) > .04:
                raise RuntimeError("PPO guard fired; do not silently continue with unequal update budgets")
            assert_frozen(runner, frozen)
            replica_sha = sync.assert_replicas(runner)
            episodes = batch["training_episodes"]
            atomic_json(args.output / f"updates/group_{index + 1:04d}.json", {
                "arm": args.arm, "rank": args.rank, "training_episodes": episodes,
                "global_batch_size": batch_size, "local_trajectories": len(group),
                "seconds": time.time() - begin, "update": update}, overwrite=False)
            checkpoint = args.output / f"models/episodes_{episodes:06d}.pt"
            runner.save(checkpoint, **metadata(manifest, args.arm, diagnostic_only=False,
                training_episodes=episodes, next_group=index + 1, rank=args.rank, world_size=args.world_size,
                schedule_sha256=schedule_sha, recipe_sha256=recipe_sha, replica_state_sha256=replica_sha))
            entries = sync.gather({"rank": args.rank, "checkpoint": str(checkpoint), "sha256": digest_file(checkpoint)})
            if args.rank == 0:
                atomic_json(args.output.parent / f"commits/episodes_{episodes:06d}.json", {
                    "arm": args.arm, "training_episodes": episodes, "world_size": args.world_size,
                    "schedule_sha256": schedule_sha, "recipe_sha256": recipe_sha,
                    "replica_state_sha256": replica_sha, "ranks": entries}, overwrite=False)
                if episodes % 480 == 0:
                    submit(manifest, checkpoint, args.arm, episodes, variant)
            sync.barrier()
            hb.update(event="group_committed", training_episodes=episodes, group=index + 1,
                      actor_updates=runner.policy_updates)
            del group
            gc.collect()
        atomic_json(args.output / "result.json", {"completed": True, "training_episodes": episodes,
            "arm": args.arm, "rank": args.rank, "policy_updates": runner.policy_updates}, overwrite=False)
    except BaseException:
        runner.save(args.output / "failure_state.pt", **metadata(manifest, args.arm,
            diagnostic_only=True, not_resumable_partial_group=True, completed_training_episodes=episodes,
            rank=args.rank, world_size=args.world_size))
        raise
    finally:
        runner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", choices=("reference", "contract", "capacity", "train", "validator"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--init-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dense-case")
    parser.add_argument("--until", type=int, default=960)
    parser.add_argument("--resume-commit", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    verify(manifest, inputs=False)
    configure_runtime(manifest["numerics"])
    if (args.output / "status.json").exists():
        raise FileExistsError("No implicit retry/overwrite of worker output")
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    seed_all(manifest["training"]["seed"])
    try:
        if args.world_size > 1:
            dist.init_process_group("gloo", init_method="file://" + str(args.init_file.resolve()),
                rank=args.rank, world_size=args.world_size, timeout=timedelta(hours=2))
        sync = TrajectoryParallel()
        with ProgressHeartbeat(args.output / "status.json", phase=args.phase, arm=args.arm, rank=args.rank,
                world_size=args.world_size, cpuset=sorted(os.sched_getaffinity(0)),
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), numerics=manifest["numerics"]) as hb:
            if args.phase == "validator":
                validator(manifest, args.output, hb, verify_fn=verify,
                    engine_fn=lambda m, v, width: engine(m, v, width, validation=True), arms=ARMS,
                    execution_identity_fn=lambda m: {m["manifest_sha256"]: m["code"]["sha256"]})
            elif args.phase == "train":
                train(args, manifest, sync, hb)
            else:
                contract(args, manifest, sync, hb)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
