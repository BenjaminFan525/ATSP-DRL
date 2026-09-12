#!/usr/bin/env python3
"""Certify the actual paused trainer state at a completed-group boundary.

Finish only the in-flight group, stop before the next group's optimizer steps,
and use the engine's existing signal-time serialization. Certification checks
ALL Adam state steps, policy steps, frozen parameters and normalizer bytes against
the full parent and committed update ledger. Never promotes an arbitrary failure
state, a partial optimizer step, or a fit checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time


def certify_steps(parent, exported, updates):
    actor_steps = sum(int(epoch["actor_step_applied"]) for row in updates for epoch in row["update"]["epochs"])
    critic_steps = sum(len(row["update"]["epochs"]) for row in updates)
    if exported["policy_updates"] != parent["policy_updates"] + actor_steps:
        raise ValueError("Uncommitted actor step detected")
    for key, increment in (("actor_optim", actor_steps), ("critic_optim", critic_steps)):
        old, now = parent[key], exported[key]
        if old["param_groups"] != now["param_groups"] or old["state"].keys() != now["state"].keys():
            raise ValueError("Optimizer ownership or state set changed")
        if not old["state"]:
            raise ValueError("Warm optimizer state required")
        for param in old["state"]:
            if int(now["state"][param]["step"]) != int(old["state"][param]["step"]) + increment:
                raise ValueError(f"Partial/uncommitted {key} step at parameter {param}")
    return {"actor_steps_since_parent": actor_steps, "critic_steps_since_parent": critic_steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pause-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Capture interrupted by signal {signum}; pause any live source mains")
    signal.signal(signal.SIGTERM, interrupted)
    manifest = json.loads(args.manifest.read_text())
    os.environ.setdefault("HKBZ_STAGE3_WORKSPACE_ROOT", manifest["execution"]["workspace_root"])
    sys.path.insert(0, manifest["execution"]["code_root"])
    import torch
    from onpolicy.scripts.train.hot_update_stage3_representation import ProcessIdentity, latest_committed
    from onpolicy.scripts.train.run_stage3_representation import verify
    from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json
    from onpolicy.utils.stage3_representation import ARMS, schedule
    from onpolicy.utils.stage3_hot_update import resume_payload_allowed
    receipt = read_json(args.pause_receipt)
    controller = ProcessIdentity(receipt["controller"]["pid"], "resume_stage3_after_probe.py")
    if controller.ticks != receipt["controller"]["start_ticks"] or controller.stat()[0] not in ("T", "t"):
        raise ValueError("Original controller must remain paused")
    verify(manifest, inputs=True)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for arm in args.arms:
        if arm not in ARMS or (args.output / arm).exists():
            raise ValueError("Unknown or already captured arm")
        record = next(r for r in receipt["workers"] if r["task"].startswith(f"train/{arm}_to960_"))
        old_output = Path(record["output"])
        process = ProcessIdentity(record["pid"], str(old_output))
        if process.ticks != record["start_ticks"] or process.stat()[0] not in ("T", "t"):
            raise ValueError("Original worker identity/pause changed")
        if "--phase train" not in process.command:
            raise ValueError("Only the actual training producer is eligible")
        folder = args.output / arm
        folder.mkdir()
        committed = latest_committed(old_output)
        jobs[arm] = dict(process=process, record=record, folder=folder, old_output=old_output,
                         committed_before=committed, target=committed + 8)
        atomic_json(folder / "intent.json", {"arm": arm, "record": record,
            "committed_before": committed, "target": committed + 8, "created_unix": time.time()}, overwrite=False)
    try:
        for job in jobs.values():
            job["process"].send(signal.SIGCONT)
        deadline = time.time() + 1800
        waiting = dict(jobs)
        while waiting:
            for arm, job in list(waiting.items()):
                if not job["process"].alive():
                    raise RuntimeError(f"Original {arm} exited before the requested boundary")
                if latest_committed(job["old_output"]) >= job["target"]:
                    job["process"].pause()
                    job["target"] = latest_committed(job["old_output"])
                    job["boundary_status"] = read_json(job["old_output"] / "status.json")
                    del waiting[arm]
            if time.time() > deadline:
                raise TimeoutError("Live boundary capture exceeded 30 minutes")
            time.sleep(.1)
        for arm, job in jobs.items():
            state_path = job["old_output"] / "failure_state.pt"
            if state_path.exists():
                raise FileExistsError("Do not overwrite a previous exported failure state")
            job["process"].send(signal.SIGTERM)
            job["process"].send(signal.SIGCONT)
        deadline = time.time() + 60
        while any(job["process"].alive() for job in jobs.values()):
            if time.time() > deadline:
                raise TimeoutError("Actual trainer state export did not finish")
            time.sleep(.1)
        for arm, job in jobs.items():
            exported_path = job["old_output"] / "failure_state.pt"
            exported = torch.load(exported_path, map_location="cpu", weights_only=False)
            resume = read_json(job["old_output"] / "resume_receipt.json")
            parent_path = Path(resume["checkpoint"])
            if digest_file(parent_path) != resume["checkpoint_sha256"]:
                raise ValueError("Full parent checkpoint changed")
            parent = torch.load(parent_path, map_location="cpu", weights_only=False)
            plan = schedule(manifest["splits"]["train_pilot120"], manifest["training"]["seed"], ARMS[arm]["batch"])
            resume_payload_allowed(parent, manifest, arm, 960, plan)
            if (exported.get("arm") != arm or exported.get("representation") != ARMS[arm]["encoder"]
                    or exported.get("source_sha256") != manifest["source"]["sha256"]
                    or exported.get("protocol_sha256") != manifest["manifest_sha256"]
                    or exported.get("completed_training_episodes") != job["target"]
                    or not exported.get("diagnostic_only") or not exported.get("not_resumable_partial_group")):
                raise ValueError("Export producer/committed boundary mismatch")
            updates = [read_json(job["old_output"] / f"updates/group_{index:04d}.json")
                       for index in range(parent["next_group"] + 1, job["target"] // 8 + 1)]
            if any(row["training_episodes"] != row["group"] * 8 for row in updates):
                raise ValueError("Committed ledger is not contiguous")
            counts = certify_steps(parent, exported, updates)
            for name, value in exported["model"].items():
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite model state")
                if name.startswith("encoder.prefix.") or (arm.startswith("E0_") and name.startswith("encoder.")):
                    if not torch.equal(value, parent["model"][name]):
                        raise ValueError("Frozen encoder changed")
            for key in ("value_normalizer", "role_value_normalizers"):
                def exact(a, b):
                    if isinstance(a, dict):
                        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
                    return torch.equal(a, b)
                if not exact(exported[key], parent[key]):
                    raise ValueError("Normalizer buffers changed outside optimizer steps")
            proof = {"verified": True, "kind": "actual_live_completed_group_state", "arm": arm,
                "export_path": str(exported_path), "export_sha256": digest_file(exported_path),
                "parent_checkpoint": str(parent_path), "parent_checkpoint_sha256": digest_file(parent_path),
                "committed_before_pause": job["committed_before"], "training_episodes": job["target"],
                "all_adam_steps_match_committed_ledger": True, "frozen_and_normalizer_bytes_equal": True,
                "model_optimizer_rng_copied_without_numeric_replay": True, "committed_groups_discarded": 0,
                "boundary_status": job["boundary_status"], "script_sha256": digest_file(__file__), **counts}
            proof_path = job["folder"] / "boundary_proof.json"
            atomic_json(proof_path, proof, overwrite=False)
            checkpoint = job["folder"] / f"episodes_{job['target']:06d}.pt"
            exported.update(diagnostic_only=False, not_resumable_partial_group=False,
                training_episodes=job["target"], next_group=job["target"] // 8,
                seed=manifest["training"]["seed"], schedule_sha256=digest_json(plan),
                validation_requests=parent["validation_requests"], batch_composition=ARMS[arm]["batch"], optimizer_recipe="O0",
                safe_boundary_proof={"path": str(proof_path), "sha256": digest_file(proof_path)})
            temporary = checkpoint.with_suffix(".tmp.pt")
            torch.save(exported, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.link(temporary, checkpoint)
            temporary.unlink()
            atomic_json(job["folder"] / "result.json", {"completed": True, "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest_file(checkpoint), "training_episodes": job["target"],
                "committed_groups_discarded": 0}, overwrite=False)
    finally:
        for job in jobs.values():
            if job["process"].alive():
                job["process"].pause()


if __name__ == "__main__":
    main()
