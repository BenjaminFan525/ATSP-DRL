#!/usr/bin/env python3
"""Explicit recovery after the 2026-09-08 probe triggered the external-job guard.

Only the recorded probe PID/birth token/cgroup may share resources. Reuses frozen
workers, the existing validator queue and the unchanged screening/extension gates.
Never resumes diagnostic failure states or overwrites historical arm outputs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback


def allowed_probe(row, lease):
    """An exact process identity, not a name/GPU-wide external-job exemption."""
    return (row["pid"] == lease["pid"] and row["start_ticks"] == lease["start_ticks"]
            and row["gpu"] == 1 and row["command"] == lease["command"]
            and row["cpus"] == lease["cpus"])


def interrupted_validation(row, checkpoint_sha):
    return (row.get("request_id") == "E3_T1_e000480" and not row.get("ok")
            and row.get("evaluation") is None and row.get("checkpoint_sha256") == checkpoint_sha
            and "KeyboardInterrupt: Signal 15; preserve diagnostic state" in row.get("error", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--probe-output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["execution"]["code_root"]).resolve()
    sys.path.insert(0, str(source))
    sys.dont_write_bytecode = True
    import torch
    from onpolicy.scripts.train.run_stage3_representation import Suite, verify
    from onpolicy.scripts.train.hot_update_stage3_representation import RollingSuite, latest_committed
    from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat
    from onpolicy.scripts.train.stage3_representation_worker import submit
    from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file
    from onpolicy.utils.stage3_hot_update import validate_hot_update, numa_plan, resume_payload_allowed
    from onpolicy.utils.stage3_representation import ARMS, external_resources, schedule

    root, attempt = Path(manifest["root"]), args.attempt.resolve()
    if attempt.parent != root / "recovery" or attempt.name != "probe_guard_resume_20260908_r1":
        raise ValueError("Use this incident's explicit, new recovery attempt")
    attempt.mkdir(exist_ok=False)

    class RecoverySuite(RollingSuite):
        def __init__(self):
            Suite.__init__(self, manifest, manifest_path=args.manifest.resolve())
            self.parent = validate_hot_update(manifest)
            self.upgrade = {"attempt_id": attempt.name}
            self.plan = numa_plan(manifest["resources"]["core_groups"], manifest["resources"]["gpus"])
            self.outputs = {}
            probe_state = read_json(args.probe_output / "status.json")
            probe_config = read_json(args.probe_output / "configuration.json")
            pid = probe_state["pid"]
            row = next((r for r in external_resources(manifest["resources"]["gpus"], os.getpid())
                        if r["pid"] == pid), None)
            expected_script = args.probe_output.parent / "probe_stage3_throughput.py"
            if (row is None or row["gpu"] != 1 or probe_state["status"] != "running"
                    or probe_config["manifest"] != str(args.manifest.resolve())
                    or digest_file(expected_script) != probe_config["probe_script_sha256"]
                    or str(expected_script) not in row["command"]):
                raise ValueError("Expected the exact user-authorized, monitored GPU1 probe")
            cgroup_text = Path(f"/proc/{pid}/cgroup").read_text()
            expected_unit = "hkbz-s3probe-b32-t16-20260908-r2.service"
            if not any(line.endswith("/" + expected_unit) for line in cgroup_text.splitlines()):
                raise ValueError("Probe cgroup identity changed")
            self.probe_lease = {**row, "cgroup": cgroup_text, "configuration": str(args.probe_output / "configuration.json")}
            atomic_json(attempt / "probe_lease.json", self.probe_lease, overwrite=False)

        def resources(self):
            external = external_resources(self.m["resources"]["gpus"], os.getpid())
            unexpected = []
            for row in external:
                accepted = allowed_probe(row, self.probe_lease)
                if accepted:
                    try:
                        accepted = Path(f"/proc/{row['pid']}/cgroup").read_text() == self.probe_lease["cgroup"]
                    except FileNotFoundError:
                        continue
                if not accepted:
                    unexpected.append(row)
            if unexpected:
                raise RuntimeError(f"Unapproved external GPU job; no preemption: {unexpected}")
            atomic_json(self.root / "resource_leases.json", dict(updated_unix=time.time(), plan=self.plan,
                external_reservations=[], authorized_colocated_probe=external, recovery_attempt=attempt.name))
            return self.plan

        def recover(self):
            verify(self.m, inputs=True)
            canary = read_json(Path(self.m["hot_update"]["attempt_dir"]) / "gpu_canary/result.json")
            if not canary.get("passed"):
                raise ValueError("Existing v1 execution-equivalence canary did not pass")
            if (self.root / "screen_admission.json").exists() or (self.queue.root / "STOP").exists():
                raise ValueError("Study already passed the recovery boundary")
            self.resources()  # No surviving old CUDA worker/controller may conflict.
            if list((self.queue.root / "running").glob("*.json")):
                raise ValueError("Unexpected claimed validation; inspect before recovery")
            checkpoints, audit = {}, {}
            for arm in ARMS:
                outputs = list((self.root / "train").glob(f"{arm}_to960*"))
                for output in outputs:
                    state = read_json(output / "status.json")
                    if Path(f"/proc/{state['pid']}/cmdline").exists():
                        command = Path(f"/proc/{state['pid']}/cmdline").read_text()
                        if str(output) in command:
                            raise ValueError(f"Original worker still exists: {arm}")
                candidates = [p for output in outputs for p in (output / "models").glob("episodes_*.pt")]
                checkpoint = max(candidates, key=lambda p: int(p.stem.split("_")[-1]))
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                plan = schedule(self.m["splits"]["train_pilot120"], self.m["training"]["seed"], ARMS[arm]["batch"])
                protocol = resume_payload_allowed(payload, self.m, arm, 960, plan)
                required = ("actor_optim", "critic_optim", "role_value_normalizers", "rng_torch", "rng_numpy",
                            "rng_python", "rng_cuda", "elite_buffer", "elite_usage", "auxiliary_steps", "validation_requests")
                if any(k not in payload for k in required) or payload["rng_cuda"] is None:
                    raise ValueError(f"Incomplete checkpoint: {arm}")
                committed = max(latest_committed(output) for output in outputs)
                checkpoints[arm] = checkpoint
                audit[arm] = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest_file(checkpoint),
                    "resume_episodes": payload["training_episodes"], "last_committed_episodes": committed,
                    "committed_episodes_to_recompute": committed - payload["training_episodes"],
                    "parent_protocol_sha256": protocol, "old_outputs_preserved": [str(p) for p in outputs]}
                del payload
            atomic_json(attempt / "recovery_audit.json", dict(reason="Authorized probe misclassified as external GPU job",
                arms=audit, training_recipe_changed=False, frozen_worker_code_changed=False,
                performance="Previously verified v1 options for all eight resumed arms",
                diagnostic_failure_states_never_resumed=True, probe_lease=self.probe_lease,
                controller_script_sha256=digest_file(__file__), created_unix=time.time(),
                material_passport={"origin_skill":"academic-research-suite/experiment-agent", "origin_mode":"run",
                    "origin_date":"2026-09-08", "version_label":"probe_guard_recovery_v1",
                    "verification_status":"checkpoint lineage verified; resumed progress pending"}), overwrite=False)
            for failed in (self.queue.root / "failed").glob("*.json"):
                row = read_json(failed)
                if not interrupted_validation(row, audit["E3_T1"]["checkpoint_sha256"]):
                    raise ValueError("Only this incident's administrative validation interruption may be retried")
                failed.rename(attempt / ("archived_failed_" + failed.name))
            for arm, checkpoint in checkpoints.items():
                episodes = audit[arm]["resume_episodes"]
                if episodes % 480:
                    continue
                request = f"{arm}_e{episodes:06d}"
                result = self.queue.poll(request)
                if result is not None:
                    if result["checkpoint_sha256"] != audit[arm]["checkpoint_sha256"]:
                        raise ValueError("Validation checkpoint mismatch")
                elif not (self.queue.root / "pending" / (request + ".json")).exists():
                    parent = self.m if audit[arm]["parent_protocol_sha256"] == self.m["manifest_sha256"] else self.parent
                    submit(parent, checkpoint, arm, episodes, ARMS[arm]["encoder"])
            with ProgressHeartbeat(self.root / "status.json", phase="screen_960", active_manifest=str(self.manifest_path),
                    recovery_attempt=attempt.name, recovery_audit=str(attempt / "recovery_audit.json"),
                    authorized_probe_pid=self.probe_lease["pid"]) as hb:
                atomic_json(self.root / "active_execution.json", dict(manifest=str(self.manifest_path), code_root=str(source),
                    recovery_attempt=attempt.name, controller_script=str(Path(__file__).resolve()), started_unix=time.time()))
                os.sched_setaffinity(0, self.plan["controller"])
                for index, placement in enumerate(self.plan["validators"]):
                    self.start(f"pool{index}_{attempt.name}", "validator", placement)
                for index, (arm, checkpoint) in enumerate(checkpoints.items()):
                    name = f"{arm}_to960_{attempt.name}"
                    self.start(name, "train", self.plan["trainers"][index], arm=arm, until=960,
                               resume_checkpoint=str(checkpoint))
                    self.outputs[arm] = self.root / "train" / name
                self.finish_study(hb)  # Existing screening + <=2 extension arms + shared validator lifecycle.

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Recovery controller signal {signum}; owned study only")
    signal.signal(signal.SIGTERM, stop)
    suite = None
    with (root / "controller.lock").open("a") as lock, (root / "hot_update.lock").open("a") as hot_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(hot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("status.json", "resource_leases.json", "active_execution.json"):
            if (root / name).exists():
                shutil.copy2(root / name, attempt / ("pre_recovery_" + name))
        try:
            suite = RecoverySuite()
            suite.recover()
        except BaseException:
            atomic_json(attempt / "failure.json", {"traceback": traceback.format_exc(), "created_unix": time.time()}, overwrite=False)
            raise
        finally:
            if suite is not None:
                suite.stop_owned()


if __name__ == "__main__":
    main()
