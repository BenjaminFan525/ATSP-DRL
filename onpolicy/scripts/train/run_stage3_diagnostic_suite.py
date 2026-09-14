#!/usr/bin/env python3
"""Fail-closed, half-resource stage scheduler. No unattended configuration search."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onpolicy.utils.stage3_research import (SOURCE, EvaluationQueue, Heartbeat, read_json,
    atomic_json, verify_protocol, pilot_gate, digest_file)
from onpolicy.scripts.train.stage3_diagnostic_worker import submit_eval


class Scheduler:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.manifest = read_json(self.path)
        self.root = Path(self.manifest["root"])
        self.resume = self.manifest.get("resume")
        self.attempt = Path(self.resume["attempt_dir"]) if self.resume else self.root
        self.lock_file = (self.root / "suite.lock").open("a")
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.resume:
            for relative, sha in self.resume["reused_artifacts"].items():
                if digest_file(self.root / relative) != sha:
                    raise ValueError(f"Recovery artifact changed after audit: {relative}")
        self.jobs = {}
        self.last_code_check = 0
        self.advisories = {}
        self.progress = {}
        actual = sorted(os.sched_getaffinity(0))
        if actual != self.manifest["resources"]["logical_cpus"]:
            raise ValueError(f"Parent CPU allocation is not the declared half: {actual}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
            raise ValueError("Suite must see exactly physical GPUs 0,1,2,3")
        (self.attempt / "logs").mkdir(parents=True, exist_ok=True)
        atomic_json(self.attempt / "runtime.json", {"started_unix": time.time(),
            "pid": os.getpid(), "cpu_affinity": actual,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_inventory": subprocess.check_output(["nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader"], text=True),
            "python": sys.executable, "code_root": str(ROOT),
            "code_sha256": self.manifest["code"]["sha256"]}, overwrite=False)
        atomic_json(self.root / "active_execution.json", {"manifest": str(self.path),
            "attempt_dir": str(self.attempt), "pid": os.getpid(),
            "code_root": str(ROOT), "code_sha256": self.manifest["code"]["sha256"]})

    def start(self, name, phase, lane, timeout, arm=None):
        if self.resume and phase in self.resume["completed_phases"]:
            print(f"[Recovery] reuse completed {phase}; artifacts verified", flush=True)
            self.jobs[name] = {"process": None, "lane": lane, "phase": phase}
            return
        if any(job["lane"] == lane and job["process"] is not None
               and job["process"].poll() is None for job in self.jobs.values()):
            raise RuntimeError(f"Lane {lane} is occupied")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), HKBZ_PHYSICAL_GPU=str(lane))
        worker = "shared_stage3_validator.py" if phase == "validator" else "stage3_diagnostic_worker.py"
        command = ["/usr/bin/taskset", "--cpu-list", self.manifest["resources"]["lanes"][lane],
            sys.executable, "-u", str(ROOT / "onpolicy/scripts/train" / worker), str(self.path)]
        if phase != "validator":
            command += ["--phase", phase]
            if arm:
                command += ["--arm", arm]
        log = (self.attempt / "logs" / f"{name}.log").open("xb")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        log.close()
        status = self.root / (f"pilot_{arm}" if phase == "train" else phase) / "status.json"
        if phase == "resume_check":
            status = self.attempt / "resume_check" / "status.json"
        self.jobs[name] = {"process": process, "lane": lane, "started": time.time(),
            "timeout": timeout, "status": status, "phase": phase}
        atomic_json(self.attempt / "commands" / f"{name}.json", {"pid": process.pid,
            "command": command, "physical_gpu": lane, "cpuset": self.manifest["resources"]["lanes"][lane],
            "started_unix": time.time(), "timeout_seconds": timeout,
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
            "code_sha256": self.manifest["code"]["sha256"]}, overwrite=False)

    def check(self, heartbeat):
        now = time.time()
        if now - self.last_code_check > 120:
            changes = verify_protocol(self.manifest)
            if changes:
                self.advisories["workspace_code"] = {"kind": "shared_workspace_code_changed",
                    "files": changes, "action": "advisory only; isolated execution unaffected"}
            else:
                self.advisories.pop("workspace_code", None)
            self.last_code_check = now
        alive = []
        for name, job in self.jobs.items():
            if job["process"] is None:
                continue
            code = job["process"].poll()
            if code is not None:
                if code:
                    raise RuntimeError(f"{name} failed with exit {code}; see logs/{name}.log. No automatic retry.")
                continue
            alive.append(name)
            if now - job["started"] > job["timeout"]:
                self.stop_job(job)
                raise TimeoutError(f"{name} exceeded its declared {job['timeout']}s hard timeout")
            if job["status"].exists():
                status = read_json(job["status"])
                age = now - status["heartbeat_unix"]
                signature = tuple(status.get(key) for key in ("event", "rollout_step", "update_epoch",
                    "update_step", "completed_cases", "completed_groups", "completed_requests", "request_id",
                    "completed_replicas"))
                if job["phase"] == "iga":
                    signature += (len(list((self.root / "iga/cases").glob("*.json"))),)
                previous_signature, changed_at = self.progress.get(name, (None, now))
                if signature != previous_signature:
                    changed_at = now
                self.progress[name] = (signature, changed_at)
                progress_limit = 2700 if job["phase"] == "iga" else 300
                if age > 120:
                    self.advisories[name] = {"kind": "heartbeat_stall", "seconds": age,
                        "action": "advisory only; no kill/retry"}
                elif now - changed_at > progress_limit and not (job["phase"] == "validator" and not status.get("request_id")):
                    self.advisories[name] = {"kind": "output_progress_stall", "seconds": now - changed_at,
                        "action": "advisory only; check log and resource use; no kill/retry"}
                else:
                    self.advisories.pop(name, None)
                if job["phase"] == "validator" and status.get("request_started_unix"):
                    if now - status["request_started_unix"] > self.manifest["timeouts_seconds"]["evaluation_job"]:
                        self.stop_job(job)
                        raise TimeoutError("Shared validator request exceeded hard timeout")
        heartbeat.update(alive_jobs=alive, advisories=self.advisories)

    def wait(self, names, heartbeat):
        while any(self.jobs[name]["process"] is not None
                  and self.jobs[name]["process"].poll() is None for name in names):
            self.check(heartbeat)
            time.sleep(10)
        self.check(heartbeat)

    def require(self, relative):
        result = read_json(self.root / relative)
        if not result["passed"]:
            raise RuntimeError(f"Research gate did not pass: {relative}. Review diagnostics before changing training/architecture.")
        return result

    @staticmethod
    def stop_job(job):
        process = job["process"]
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)

    def run(self):
        manifest = self.manifest
        with Heartbeat(self.root / "status.json", phase="contract",
                       attempt_dir=str(self.attempt), code_root=str(ROOT),
                       workspace_code_policy=manifest.get("execution", {}).get("workspace_code_policy", "strict")) as heartbeat:
            try:
                verify_protocol(manifest)
                self.start("validator", "validator", 3, manifest["timeouts_seconds"]["suite"])
                baseline_id = f"c0_tune60_{self.resume['id']}" if self.resume else "c0_tune60"
                submit_eval(manifest, SOURCE, baseline_id, 0)
                if self.resume:
                    heartbeat.update(phase="resume_compatibility_check",
                        reused_progress=self.resume["progress"])
                    self.start("resume_check", "resume_check", 0, manifest["timeouts_seconds"]["contract"])
                    self.wait(["resume_check"], heartbeat)
                    self.require(str(self.attempt.relative_to(self.root) / "resume_check/result.json"))
                self.start("contract", "contract", 0, manifest["timeouts_seconds"]["contract"])
                self.wait(["contract"], heartbeat)
                self.require("contract/result.json")
                queue = EvaluationQueue(self.root / "validator")
                while queue.poll(baseline_id) is None:
                    self.check(heartbeat)
                    time.sleep(10)
                baseline = queue.poll(baseline_id)["evaluation"]
                errors = [abs(r["makespan"] - manifest["source_costs"][r["case_id"]]) for r in baseline["cases"]]
                if max(errors) > .1:
                    raise RuntimeError(f"Full Tune60 C0 reproduction failed: max cost error {max(errors)}")
                heartbeat.update(phase="diagnostics", c0_tune60_max_cost_error=max(errors))
                timeout = manifest["timeouts_seconds"]["diagnostics"]
                self.start("iga", "iga", 0, timeout)
                self.start("sample", "sample", 1, timeout)
                self.start("counterfactual", "counterfactual", 2, timeout)
                self.wait(["iga"], heartbeat)
                self.start("replay", "replay", 0, timeout)
                self.wait(["replay", "counterfactual"], heartbeat)
                self.require("replay/result.json")
                self.require("counterfactual/result.json")
                self.start("bc_heads", "bc_heads", 0, timeout)
                self.start("bc_full", "bc_full", 2, timeout)
                self.wait(["bc_heads", "bc_full", "sample"], heartbeat)
                sampling = read_json(self.root / "sample/result.json")
                heads = read_json(self.root / "bc_heads/result.json")
                full = read_json(self.root / "bc_full/result.json")
                passed = sampling["passed"] and (heads["passed"] or full["passed"])
                gate = {"passed": passed, "sampling_passed": sampling["passed"],
                    "bc_heads_passed": heads["passed"], "bc_full_passed": full["passed"],
                    "unfreeze_shared": not heads["passed"] and full["passed"],
                    "decision": "start fresh-C0 paired pilot" if passed else "stop for diagnosis; do not scale training"}
                atomic_json(self.root / "diagnostic_gate.json", gate, overwrite=False)
                if not passed:
                    heartbeat.update(phase="diagnostic_gate_not_met", gate=gate)
                    atomic_json(self.root / "report.json", {"status": "diagnostic_gate_not_met", **gate}, overwrite=False)
                    return
                heartbeat.update(phase="pilot")
                for lane, arm in enumerate(("R0", "R1", "R2")):
                    self.start(f"pilot_{arm}", "train", lane, manifest["timeouts_seconds"]["pilot"], arm)
                # Whichever training lane finishes first takes the fourth arm.
                while all(self.jobs[f"pilot_{arm}"]["process"].poll() is None for arm in ("R0", "R1", "R2")):
                    self.check(heartbeat)
                    time.sleep(10)
                self.check(heartbeat)
                lane = next(self.jobs[f"pilot_{arm}"]["lane"] for arm in ("R0", "R1", "R2")
                            if self.jobs[f"pilot_{arm}"]["process"].poll() == 0)
                self.start("pilot_R3", "train", lane, manifest["timeouts_seconds"]["pilot"], "R3")
                self.wait([f"pilot_{arm}" for arm in ("R0", "R1", "R2", "R3")], heartbeat)
                requests = [f"pilot_{arm}_e{n:06d}" for arm in ("R0", "R1", "R2", "R3")
                            for n in (480, 960, 1440, 1920)]
                while any(queue.poll(name) is None for name in requests):
                    self.check(heartbeat)
                    time.sleep(10)
                report = {"status": "pilot_completed", "arms": {}, "confirmation_started": False,
                    "finalblind_opened": False, "endpoint": "fixed 1920 fresh trajectories; not best checkpoint"}
                for arm in ("R0", "R1", "R2", "R3"):
                    evaluations = [queue.poll(f"pilot_{arm}_e{n:06d}")["evaluation"]["summary"]
                                   for n in (480, 960, 1440, 1920)]
                    report["arms"][arm] = {"evaluations": evaluations, "pilot_gate_passed": pilot_gate(evaluations)}
                atomic_json(self.root / "report.json", report, overwrite=False)
                heartbeat.update(phase="pilot_completed_requires_review")
            finally:
                # Clean up only child process groups started by this scheduler.
                for job in self.jobs.values():
                    self.stop_job(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    Scheduler(parser.parse_args().manifest).run()
