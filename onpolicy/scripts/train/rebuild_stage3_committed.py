#!/usr/bin/env python3
"""Reproduce committed groups to obtain a current full checkpoint without rewind.

Original paused processes and outputs remain untouched. Every sampled action,
cost and PPO metric is checked against the original committed artifacts before
publishing a full checkpoint. A mismatch fails closed; no failure-state promotion.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def equivalent(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        ignored = {"execution_cache", "peak_reserved_gib"}
        keys = a.keys() - ignored
        if keys != b.keys() - ignored:
            return False, path + ": keys"
        for key in keys:
            ok, detail = equivalent(a[key], b[key], path + "/" + key)
            if not ok:
                return ok, detail
        return True, ""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, path + ": length"
        for index, (x, y) in enumerate(zip(a, b)):
            ok, detail = equivalent(x, y, path + f"/{index}")
            if not ok:
                return ok, detail
        return True, ""
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-6)), path + f": {a} != {b}"
    return a == b, path + f": {a} != {b}"


def run(args, manifest):
    from onpolicy.scripts.train.run_stage3_representation import verify
    from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat
    from onpolicy.scripts.train.stage3_local_worker import load_group
    from onpolicy.scripts.train.stage3_representation_worker import metadata
    from onpolicy.utils.stage3_research import read_json, atomic_json, digest_file, digest_json
    from onpolicy.utils.stage3_sampling_audit import action_digest
    from onpolicy.utils.stage3_representation import ARMS, schedule
    from onpolicy.utils.stage3_hot_update import resume_payload_allowed
    from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
    import torch

    root, output = Path(manifest["root"]), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = read_json(args.pause_receipt)
    record = next(r for r in receipt["workers"] if r["task"].startswith(f"train/{args.arm}_to960_"))
    state = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(")", 1)[1].split()
    if state[19] != record["start_ticks"] or state[0] not in ("T", "t"):
        raise ValueError("Original source worker must remain paused")
    old_output = Path(record["output"])
    updates = sorted((old_output / "updates").glob("group_*.json"))
    target = read_json(updates[-1])["training_episodes"]
    candidates = [p for d in (root / "train").glob(f"{args.arm}_to960*")
                  for p in (d / "models").glob("episodes_*.pt") if int(p.stem.split("_")[-1]) <= target]
    checkpoint = max(candidates, key=lambda p: int(p.stem.split("_")[-1]))
    config = ARMS[args.arm]
    plan = schedule(manifest["splits"]["train_pilot120"], manifest["training"]["seed"], config["batch"])
    runner = None
    with ProgressHeartbeat(output / "status.json", phase="reconstruct_committed", arm=args.arm,
                           target_episodes=target, diagnostic_reconstruction=True) as hb:
        try:
            verify(manifest, inputs=True)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            protocol = resume_payload_allowed(payload, manifest, args.arm, 960, plan)
            cursor = payload["next_group"]
            requests = payload["validation_requests"]
            runner = RepresentationEngine(config["encoder"], source=manifest["source"]["path"], width=8,
                exploration="J", diagnostics=True, performance=manifest["hot_update"]["options"], cuda_memory_fraction=.65)
            runner.resume(checkpoint, protocol_sha256=protocol, exploration="J")
            del payload
            checks = []
            for index in range(cursor, target // 8):
                reference = old_output / f"rollouts/group_{index + 1:04d}.json"
                expected, _ = load_group({"path": str(reference), "sha256": digest_file(reference)})
                batch = plan[index]
                trajectories = runner.rollout(batch["cases"], batch["seeds"], retain=True, heartbeat=hb)
                if (len(trajectories) != len(expected) or any(a["case_id"] != b["case_id"]
                        or a["seed"] != b["seed"] or action_digest(a["actions"]) != action_digest(b["actions"])
                        or abs(a["makespan"] - b["makespan"]) > 1e-6 for a, b in zip(trajectories, expected))):
                    raise ValueError(f"Action/cost reproduction mismatch at {args.arm} group {index + 1}")
                result = runner.update(trajectories, "source", manifest["source_costs"],
                    epochs=manifest["training"]["ppo_epochs"], chunk=manifest["training"]["tbptt_steps"],
                    clip_mode=manifest["training"]["clip_mode"], heartbeat=hb)
                old_update = old_output / f"updates/group_{index + 1:04d}.json"
                ok, detail = equivalent(result, read_json(old_update)["update"])
                if not ok:
                    raise ValueError(f"PPO reproduction mismatch at group {index + 1}: {detail}")
                checks.append({"group": index + 1, "training_episodes": (index + 1) * 8,
                    "actions_exact": True, "costs_equal_atol_1e_6": True, "ppo_metrics_equivalent": True,
                    "original_rollout": str(reference), "original_rollout_sha256": digest_file(reference),
                    "original_update": str(old_update), "original_update_sha256": digest_file(old_update)})
                atomic_json(output / "checks.json", checks)
                hb.update(event="reproduced_group", training_episodes=(index + 1) * 8)
                del trajectories, expected
            verify(manifest, inputs=True)
            proof = {"verified": True, "arm": args.arm, "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest_file(checkpoint), "old_output": str(old_output),
                "training_episodes": target, "checks": checks, "committed_groups_discarded": 0,
                "numeric_tolerance": {"rtol": 2e-5, "atol": 2e-6},
                "comparison_scope": "exact actions and costs; PPO metrics within CUDA numerical tolerance, not original live parameter bytes",
                "code_sha256": manifest["code"]["sha256"], "script_sha256": digest_file(__file__)}
            proof_path = output / "reconstruction_proof.json"
            atomic_json(proof_path, proof, overwrite=False)
            saved = output / f"episodes_{target:06d}.pt"
            checksum = runner.save(saved, **metadata(manifest, arm=args.arm, diagnostic_only=False,
                training_episodes=target, next_group=target // 8, seed=manifest["training"]["seed"],
                schedule_sha256=digest_json(plan), validation_requests=requests,
                batch_composition=config["batch"], optimizer_recipe="O0",
                reconstruction_proof={"path": str(proof_path), "sha256": digest_file(proof_path)}))
            atomic_json(output / "result.json", {"completed": True, "checkpoint": str(saved),
                "checkpoint_sha256": checksum, "training_episodes": target, "committed_groups_discarded": 0}, overwrite=False)
        finally:
            if runner is not None:
                runner.close()


def suite(args, manifest):
    from onpolicy.utils.stage3_research import atomic_json, read_json
    from onpolicy.utils.stage3_hot_update import numa_plan
    from onpolicy.utils.stage3_representation import ARMS, cpu_text
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    placements = numa_plan(manifest["resources"]["core_groups"], manifest["resources"]["gpus"])["trainers"]
    slots = [placements[i] for i in (0, 2, 4, 6)]
    pending, jobs, results = list(ARMS), {}, {}
    begin = time.time()
    try:
        while pending or jobs:
            for index, job in list(jobs.items()):
                rc = job["process"].poll()
                if rc is not None:
                    if rc:
                        raise RuntimeError(f"Checkpoint reconstruction failed: {job['arm']} rc={rc}")
                    results[job["arm"]] = read_json(output / job["arm"] / "result.json")
                    del jobs[index]
            for index, placement in enumerate(slots):
                if index in jobs or not pending:
                    continue
                arm = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["resources"]["gpus"][placement["gpu"]]["uuid"])
                command = ["taskset", "--cpu-list", cpu_text(placement["cpus"]), sys.executable, "-B", "-u", __file__, "worker",
                    "--manifest", str(args.manifest), "--pause-receipt", str(args.pause_receipt),
                    "--output", str(output / arm), "--arm", arm]
                with (output / (arm + ".log")).open("x") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                jobs[index] = {"arm": arm, "process": process}
            atomic_json(output / "status.json", {"status": "running", "pid": os.getpid(), "updated_unix": time.time(),
                "running": {str(i): {"arm": j["arm"], "pid": j["process"].pid} for i, j in jobs.items()},
                "pending": pending, "completed": list(results)})
            if time.time() - begin > 7200:
                raise TimeoutError("Checkpoint reproduction exceeded two hours")
            time.sleep(5)
        atomic_json(output / "result.json", {"completed": True, "arms": results}, overwrite=False)
        atomic_json(output / "status.json", {"status": "completed", "pid": os.getpid(), "updated_unix": time.time()})
    finally:
        import signal
        for job in jobs.values():
            if job["process"].poll() is None:
                os.killpg(job["process"].pid, signal.SIGTERM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("suite", "worker"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pause-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sys.path.insert(0, manifest["execution"]["code_root"])
    (suite if args.mode == "suite" else run)(args, manifest)
