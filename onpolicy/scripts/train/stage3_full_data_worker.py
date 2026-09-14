#!/usr/bin/env python3
"""Two synchronous full-data policies; one suite-wide asynchronous validator pool."""
import argparse
import gc
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
from onpolicy.runner.shared.stage3_research_engine import seed_all
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat, save_group
from onpolicy.scripts.train.stage3_full_policy_worker import contract, collect, metadata
from onpolicy.scripts.train.stage3_local_worker import evaluate, frozen_state, assert_frozen
from onpolicy.scripts.train.stage3_representation_worker import max_role_kl
from onpolicy.utils.stage3_distributed import TrajectoryParallel
from onpolicy.utils.stage3_full_policy import shard
from onpolicy.utils.stage3_full_data import (ARMS, HISTORY, EVAL_PROTOCOL, schedule, verify,
    source_costs, publish_epoch, validate_request)
from onpolicy.utils.stage3_representation import RepresentationQueue
from onpolicy.utils.stage3_numerics import configure_runtime, install_stable_pool
from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json, paired_summary


class FullDataEngine(RepresentationEngine):
    def resume(self, *args, **kwargs):
        payload = super().resume(*args, **kwargs)
        if getattr(self, "compare_uncached_on_resume", False):
            # Reference canary: same stored model/RNG/Adam and same real traces,
            # compare the optimized first update with an uncached repeated one.
            self.execution_cache = None
        return payload

    def update(self, *args, **kwargs):
        kwargs.setdefault("visit_balanced", True)
        return super().update(*args, **kwargs)

    def save(self, path, **kwargs):
        kwargs.setdefault("history", HISTORY)
        kwargs.setdefault("environment_overrides", self.environment_overrides)
        return super().save(path, **kwargs)


def engine(manifest, variant, width=8, checkpoint=None, *, validation=False, optimized=True):
    configure_runtime(manifest["numerics"])
    runner = FullDataEngine(variant, checkpoint=checkpoint, source=manifest["source"]["path"],
        width=width, exploration="J", diagnostics=True, semantic_history=True,
        environment_overrides=manifest["contract"], cuda_memory_fraction=.15 if validation else .8,
        performance={"cache_inputs": optimized and not validation, "cache_frozen_features": False,
                     "cache_mib": 1024, "defer_statistics": True, "disable_activation_checkpoint": True})
    torch.cuda.set_per_process_memory_fraction(.15 if validation else 1., runner.device)
    install_stable_pool(runner.policy)
    # Do not reset restored Adam moments; new C0 engines already have fresh optimizers.
    if checkpoint is None:
        runner.set_actor_lr(manifest["training"]["actor_lr"])
    return runner


def validator(manifest, output, hb):
    queue = RepresentationQueue(Path(manifest["root"]) / "validator")
    runner, current = None, None
    try:
        while True:
            claim = queue.claim()
            if claim is None:
                hb.update(request_id=None, request_started_unix=None, event="validator_idle")
                if (queue.root / "STOP").exists():
                    return
                time.sleep(2)
                continue
            path, request = claim
            try:
                verify(manifest, inputs=False)
                validate_request(manifest, request)
                checkpoint = None if request["baseline"] else request["checkpoint"]
                if checkpoint:
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    if (payload.get("diagnostic_only") or payload.get("forbidden_as_rl_initialization")
                            or payload.get("history") != HISTORY
                            or payload.get("environment_overrides") != manifest["contract"]
                            or payload.get("protocol_sha256") != manifest["manifest_sha256"]
                            or payload.get("source_sha256") != manifest["source"]["sha256"]
                            or payload.get("arm") != request["arm"]
                            or payload.get("training_episodes") != request["training_episodes"]
                            or payload.get("representation") != request["representation"]
                            or payload.get("exploration") != request["training_exploration"]):
                        raise ValueError("Unqualified full-data checkpoint")
                    del payload
                key = (request["representation"], request["baseline"])
                if current != key:
                    if runner is not None:
                        runner.close()
                        del runner
                        gc.collect()
                        torch.cuda.empty_cache()
                    runner = engine(manifest, request["representation"],
                                    width=manifest["resources"]["validator_width"], validation=True)
                    current = key
                if checkpoint:
                    runner.load(checkpoint)
                hb.update(event="greedy_evaluation", request_id=request["request_id"], request_started_unix=time.time())
                rows = evaluate(runner, request["cases"], hb)
                costs = ({r["case_id"]: r["makespan"] for r in rows} if request["baseline"]
                         else source_costs(manifest, confirmation=request["split"] == "confirmation"))
                result = {"cases": rows, "summary": paired_summary(rows, costs), "history": HISTORY,
                          "split": request["split"], "evaluation_protocol": EVAL_PROTOCOL}
                queue.finish(path, result)
                hb.update(event="validation_complete", request_id=None, request_started_unix=None)
            except BaseException:
                queue.finish(path, None, traceback.format_exc())
                raise
    finally:
        if runner is not None:
            runner.close()


def train(args, manifest, sync, hb):
    root, variant = Path(manifest["root"]), ARMS[args.arm]["encoder"]
    admission_path = root / "training_admission.json"
    admission = read_json(admission_path)
    if (not admission["passed"] or args.arm not in admission["eligible_arms"]
            or admission["protocol_sha256"] != manifest["manifest_sha256"]
            or admission["baseline_sha256"] != digest_file(root / "baselines.json")):
        raise ValueError("Full-data technical admission missing/changed")
    batch_size = admission["batch_size"]
    plan = schedule(manifest["splits"]["train_full600"], manifest["training"]["seed"], batch_size)
    schedule_sha, recipe_sha = digest_json(plan), digest_file(admission_path)
    if args.until != 7680 or args.world_size != 4 or batch_size != 32:
        raise ValueError("This round fixes 7680 trajectories, batch32 and four synchronized ranks per arm")
    runner = engine(manifest, variant, width=8)
    costs = source_costs(manifest)
    cursor, episodes = 0, 0
    queue = RepresentationQueue(root / "validator")
    try:
        if args.resume_commit:
            commit = read_json(args.resume_commit)
            if (commit["arm"] != args.arm or commit["world_size"] != args.world_size
                    or commit["schedule_sha256"] != schedule_sha or commit["recipe_sha256"] != recipe_sha):
                raise ValueError("Resume commit identity mismatch")
            entry = commit["ranks"][args.rank]
            if digest_file(entry["checkpoint"]) != entry["sha256"]:
                raise ValueError("Committed checkpoint changed")
            payload = runner.resume(entry["checkpoint"], protocol_sha256=manifest["manifest_sha256"], exploration="J")
            if (payload.get("diagnostic_only") or payload.get("rank") != args.rank
                    or payload.get("world_size") != args.world_size or payload.get("arm") != args.arm
                    or payload.get("history") != HISTORY or payload.get("environment_overrides") != manifest["contract"]
                    or payload.get("schedule_sha256") != schedule_sha or payload.get("recipe_sha256") != recipe_sha):
                raise ValueError("Invalid full-data rank checkpoint")
            cursor, episodes = payload["next_group"], payload["training_episodes"]
            if episodes != cursor * batch_size or episodes != commit["training_episodes"]:
                raise ValueError("Resume cursor mismatch")
            del payload
        sync.assert_replicas(runner)
        frozen = frozen_state(runner)
        for index in range(cursor, len(plan)):
            sync.barrier()
            while queue.pending_count(args.arm) >= 4 or queue.pending_count() >= 8:
                hb.update(event="validator_backpressure", pending=queue.pending_count())
                time.sleep(5)
            batch = plan[index]
            local = shard(batch, args.rank, args.world_size)
            hb.update(event="group_start", training_episodes=episodes, group=index+1,
                      data_epoch=batch["data_epoch"], target_episodes=args.until)
            torch.cuda.reset_peak_memory_stats()
            begin = time.time()
            group = collect(runner, local, hb)
            rollout_seconds = time.time() - begin
            save_group(args.output / f"rollouts/group_{index+1:04d}.json", group,
                {"arm": args.arm, "rank": args.rank, "world_size": args.world_size,
                 "global_indices": local["global_indices"], "policy_updates": runner.policy_updates,
                 "visit_ids": [batch["visit_ids"][i] for i in local["global_indices"]]})
            update_start = time.time()
            update = runner.update(group, "source", costs, epochs=2, chunk=8, distributed=sync, heartbeat=hb)
            update_seconds = time.time() - update_start
            if len(update["epochs"]) != 2 or not all(e["actor_step_applied"] for e in update["epochs"]) or max_role_kl(update) > .04:
                raise RuntimeError("PPO guard fired; keep artifacts, no silent recipe change")
            assert_frozen(runner, frozen)
            replica_sha = sync.assert_replicas(runner)
            episodes = batch["training_episodes"]
            checkpoint = args.output / f"models/episodes_{episodes:06d}.pt"
            checkpoint_start = time.time()
            runner.save(checkpoint, **metadata(manifest, args.arm, diagnostic_only=False,
                training_episodes=episodes, next_group=index+1, rank=args.rank, world_size=args.world_size,
                schedule_sha256=schedule_sha, recipe_sha256=recipe_sha, replica_state_sha256=replica_sha))
            entries = sync.gather({"rank": args.rank, "checkpoint": str(checkpoint), "sha256": digest_file(checkpoint)})
            if args.rank == 0:
                atomic_json(args.output.parent / f"commits/episodes_{episodes:06d}.json", {
                    "arm": args.arm, "training_episodes": episodes, "world_size": args.world_size,
                    "schedule_sha256": schedule_sha, "recipe_sha256": recipe_sha,
                    "replica_state_sha256": replica_sha, "ranks": entries}, overwrite=False)
                if episodes % 960 == 0:
                    publish_epoch(manifest, checkpoint, args.arm, episodes, variant)
            sync.barrier()
            atomic_json(args.output / f"updates/group_{index+1:04d}.json", {
                "arm": args.arm, "rank": args.rank, "training_episodes": episodes, "data_epoch": batch["data_epoch"],
                "global_batch_size": batch_size, "local_trajectories": len(group), "seconds": time.time()-begin,
                "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
                "checkpoint_and_commit_seconds": time.time()-checkpoint_start,
                "update": update}, overwrite=False)
            hb.update(event="group_committed", training_episodes=episodes, group=index+1,
                      actor_updates=runner.policy_updates, completed_data_epochs=episodes//960)
            del group
            gc.collect()
        atomic_json(args.output / "result.json", {"completed": True, "training_episodes": episodes,
            "arm": args.arm, "rank": args.rank, "policy_updates": runner.policy_updates,
            "data_epochs": episodes//960}, overwrite=False)
    except BaseException:
        runner.save(args.output / "failure_state.pt", **metadata(manifest, args.arm, diagnostic_only=True,
            not_resumable_partial_group=True, completed_training_episodes=episodes,
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
    parser.add_argument("--until", type=int, default=7680)
    parser.add_argument("--resume-commit", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    verify(manifest, inputs=False)
    configure_runtime(manifest["numerics"])
    if (args.output / "status.json").exists():
        raise FileExistsError("No implicit worker retry")
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
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
                validator(manifest, args.output, hb)
            elif args.phase == "train":
                train(args, manifest, sync, hb)
            else:
                runtime = {**manifest, "source_costs": source_costs(manifest)}
                def make_reference(m, variant, **options):
                    runner = engine(m, variant, **options)
                    runner.compare_uncached_on_resume = args.phase == "reference"
                    return runner
                contract(args, runtime, sync, hb, engine_fn=make_reference, submit_fn=publish_epoch)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
