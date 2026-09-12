#!/usr/bin/env python3
"""Frozen local-exploration study: contract -> sampling -> fit/CF -> four pilots.

Confirmation and Finalblind are intentionally NOT executable in this entry.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onpolicy.utils.stage3_research import (BASE_MANIFEST, atomic_json, read_json,
    digest_json, digest_file, verify_cases, code_changes, pilot_gate)
from onpolicy.utils.stage3_local_exploration import ARMS, MODES, exploration_gate, LocalEvaluationQueue
from onpolicy.utils.stage3_local_resources import SHARED_VALIDATOR, pilot_resource_plan, validate_coexistence
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat

OVERLAYS = (
    "onpolicy/runner/shared/stage3_research_engine.py",
    "onpolicy/utils/stage3_local_exploration.py",
    "onpolicy/utils/stage3_local_resources.py",
    "onpolicy/scripts/train/run_stage3_local_exploration.py",
    "onpolicy/scripts/train/stage3_local_worker.py",
    "onpolicy/scripts/train/launch_stage3_local_exploration_half.sh",
    "onpolicy/envs/HKBZ/test/test_stage3_local_exploration.py",
    "onpolicy/envs/HKBZ/test/test_stage3_local_resources.py",
    "STAGE3_LOCAL_EXPLORATION_20260906.md",
)


def identity(manifest):
    return digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def prepare(args):
    audit_path = args.audit.resolve()
    audit = read_json(audit_path)
    prior_path = Path(audit["prior_manifest"]["path"])
    if digest_file(prior_path) != audit["prior_manifest"]["sha256"]:
        raise ValueError("Historical diagnostic identity changed")
    prior = read_json(prior_path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("A new suite directory is required")
    if read_json(Path(audit["root"]) / "status.json")["status"] != "completed":
        raise ValueError("Sampling audit has not completed")
    if digest_file(audit["source"]["path"]) != audit["source"]["sha256"]:
        raise ValueError("C0 changed")
    splits = {key: prior["splits"][key] for key in
              ("train_diag32", "train_fit16", "train_probe64", "train_pilot120", "tune")}
    for key, rows in splits.items():
        verify_cases(rows, training=key.startswith("train"))
    disjoint = [splits[k] for k in ("train_diag32", "train_probe64", "train_pilot120", "tune")]
    hashes = [r["content_sha256"] for rows in disjoint for r in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Cross-split content overlap")
    lineage = [r["case_sha256"] for rows in disjoint for r in rows if r.get("case_sha256")]
    if len(set(lineage)) != len(lineage):
        raise ValueError("Cross-split canonical case identity overlap")
    if not {c["content_sha256"] for c in splits["train_fit16"]} <= {
            c["content_sha256"] for c in splits["train_diag32"]}:
        raise ValueError("Fit must be a subset of Diag")
    files = {}
    for relative, sha in audit["code"]["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe snapshot entry")
        source = Path(audit["execution"]["code_root"]) / relative
        if digest_file(source) != sha:
            raise ValueError(f"Verified base snapshot changed: {relative}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative] = sha
    for relative in OVERLAYS:
        source, target = ROOT / relative, output / "source" / relative
        sha = digest_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest_file(target) != sha:
            raise ValueError("Overlay changed while snapshotting")
        files[relative] = sha
    archives = {}
    for mode, old_arm in (("R", "resource_t030"), ("J", "full_t003")):
        archives[mode] = {}
        for case in audit["cases"]:
            directory = Path(audit["root"]) / "sampling" / old_arm / case["name"]
            groups = []
            for first in range(0, 32, 8):
                path = directory / f"group_{first:02d}.json"
                group = read_json(path)
                if digest_file(group["trace"]["path"]) != group["trace"]["sha256"]:
                    raise ValueError("Archived trace changed")
                groups.append({"path": str(path), "sha256": digest_file(path), "trace": group["trace"]})
            archives[mode][case["name"]] = groups
    manifest = {
        "schema": "stage3-local-exploration-v1", "root": str(output), "created_unix": time.time(),
        "material_passport": {"origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run", "origin_date": "2026-09-06", "verification_status": "prepared",
            "version_label": "local_exploration_v1", "data_classification": "internal; no upload"},
        "source": audit["source"], "contract": audit["contract"],
        "contract_sha256": digest_json(audit["contract"]), "splits": splits,
        "source_costs": {r["path"]: prior["source_costs"][r["path"]] for rows in splits.values() for r in rows},
        "archives": archives, "archive_cases": audit["cases"], "audit_manifest": str(audit_path),
        "code": {"files": files, "sha256": digest_json(files)},
        "execution": {"code_root": str(output / "source"), "workspace_root": str(ROOT),
            "workspace_code_policy": "advisory", "python": sys.executable,
            "python_version": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torch-geometric", "numpy", "scipy", "gymnasium", "PyYAML")},
            "base_snapshot_sha256": audit["code"]["sha256"], "overlays": list(OVERLAYS)},
        "input_files": {str(p): digest_file(p) for p in (BASE_MANIFEST, audit_path, prior_path)},
        "resources": {**copy.deepcopy(audit["resources"]), "pilot_validator": copy.deepcopy(SHARED_VALIDATOR)},
        "modes": MODES, "arms": ARMS,
        "sampling": {"diag_replicas": 32, "probe_replicas": 8, "seed": 20260909,
            "fresh_trajectories": 2560, "reused_trajectories": 512},
        "fit": {"passes": 10, "teacher_min_potential": .005, "recovery_min": .5, "lr": 1e-5},
        "counterfactual": {"cases": audit["cases"], "positions": [.25, .50, .75, .9],
            "branches_per_position": 8, "short_window_steps": 4, "seed": 2026090607},
        "training": {"pilot_seed": 2026090611, "group_size": 8, "passes": 2, "episodes": 1920,
            "actor_lr_candidates": [5e-6, 1e-6], "critic_lr": 1e-4, "ppo_epochs": 2,
            "clip": .2, "kl_limit": .02, "post_update_hard_kl": .04,
            "tbptt_steps": 8, "gamma": 1, "reward_coef": .01, "advantage": "source",
            "save_every_groups": 30, "eval_every_groups": 60, "queue_pending_per_arm": 2,
            "elite_interval": 4, "elite_lr": 1e-6, "elite_per_case": 1},
        "gates": {"exploration": "Diag32 >=8 cases improve >1%; best8 gain >=1%; Probe64 best8 >0",
            "pure_ppo_requires_bc": False, "hybrid_requires_bc": True,
            "pilot_last_two_gain": .01, "early_stop_two_regression": .02,
            "formal_mean_gain": .02, "formal_all_seeds_positive": True,
            "formal_win_fraction": .60, "formal_bad5_fraction": .05,
            "ood_stress_and_joint_regression_max": .005, "tail_regression_max": .01},
        "timeouts_seconds": {"contract": 21600, "sample": 129600, "fit": 129600,
            "counterfactual": 43200, "validator_request": 21600, "pilot_cap": 432000,
            "suite": 864000, "progress_warning": 600},
        "automatic_pilot": True, "automatic_confirmation": False, "automatic_finalblind": False,
        "confirmation_plan": {"seeds": [20260921, 20260922, 20260923], "episodes": 4800,
            "training_set": "train_pilot120", "passes": 5, "requires_separate_authorization": True},
        "exposure": "Probe/Tune/Gate historically exposed; Finalblind not opened; no validation labels",
    }
    manifest["manifest_sha256"] = identity(manifest)
    atomic_json(output / "manifest.json", manifest, overwrite=False)
    print(str(output / "manifest.json"), flush=True)


def verify(manifest, *, inputs=True):
    if identity(manifest) != manifest["manifest_sha256"]:
        raise ValueError("Frozen protocol changed")
    if ROOT.resolve() != Path(manifest["execution"]["code_root"]).resolve():
        raise ValueError("Execute only from the isolated snapshot")
    if code_changes(manifest["code"]["files"], ROOT):
        raise ValueError("Execution snapshot changed")
    if digest_file(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("C0 changed")
    if inputs:
        if any(importlib.metadata.version(name) != version
               for name, version in manifest["execution"]["packages"].items()):
            raise ValueError("Execution dependency versions changed")
        for path, sha in manifest["input_files"].items():
            if digest_file(path) != sha:
                raise ValueError(f"Frozen input changed: {path}")
        for name, cases in manifest["splits"].items():
            verify_cases(cases, training=name.startswith("train"))
    return code_changes(manifest["code"]["files"], manifest["execution"]["workspace_root"])


class Suite:
    def __init__(self, manifest):
        self.m = manifest
        self.root = Path(manifest["root"])
        self.jobs = {}
        self.pilot_plan = None

    def start(self, name, phase, lane, *, mode=None, arm=None, timeout=None):
        placement = {"lane": lane, "gpu": self.m["resources"]["physical_gpus"][lane],
                     "cpuset": self.m["resources"]["lanes"][lane], "cuda_memory_fraction": .80}
        if phase in ("train", "validator") and self.pilot_plan is not None:
            placement = (self.pilot_plan["validator"] if phase == "validator"
                         else self.pilot_plan["trainers"][lane])
            if placement["lane"] != lane:
                raise ValueError("Worker lane differs from its frozen resource placement")
        validate_coexistence(self.pilot_plan, phase, placement,
                             [j for j in self.jobs.values() if j["process"].poll() is None])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(placement["gpu"]), PYTHONPATH=str(ROOT),
                   HKBZ_STAGE3_WORKSPACE_ROOT=self.m["execution"]["workspace_root"],
                   HKBZ_STAGE3_CUDA_MEMORY_FRACTION=str(placement["cuda_memory_fraction"]))
        command = ["/usr/bin/taskset", "--cpu-list", placement["cpuset"],
                   self.m["execution"]["python"], "-u", str(ROOT / "onpolicy/scripts/train/stage3_local_worker.py"),
                   str(self.root / "manifest.json"), "--phase", phase, "--lane", str(lane)]
        if mode:
            command.extend(("--mode", mode))
        if arm:
            command.extend(("--arm", arm))
        log_path = self.root / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.root / "commands" / f"{name}.json", {"command": command, **placement,
                    "created_unix": time.time()}, overwrite=False)
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.jobs[name] = {"process": process, "phase": phase, **placement, "start": time.time(),
                          "timeout": timeout or self.m["timeouts_seconds"].get(phase, 21600),
                          "status_path": self.root / ({"contract": f"contract/{mode}",
                              "sample": f"sampling/lane{lane}", "fit": f"fit/{mode}",
                              "counterfactual": "counterfactual", "train": f"pilot/{arm}",
                              "validator": "validator"}[phase]) / "status.json"}

    def check(self, heartbeat):
        now = time.time()
        warnings = []
        for name, job in self.jobs.items():
            rc = job["process"].poll()
            if rc is not None and rc != 0:
                raise RuntimeError(f"Worker {name} exited {rc}; no automatic retry")
            if rc is None and now - job["start"] > job["timeout"]:
                heartbeat.update(event="hard_timeout", timed_out_worker=name)
                raise TimeoutError(f"Hard timeout: {name}")
            if rc is None and job["status_path"].exists():
                state = read_json(job["status_path"])
                idle = name == "validator" and not state.get("request_id")
                if not idle and now - state.get("last_progress_unix", job["start"]) > self.m["timeouts_seconds"]["progress_warning"]:
                    warnings.append({"worker": name, "kind": "no_actual_progress_advisory"})
                if name == "validator" and state.get("request_started_unix") and now - state["request_started_unix"] > self.m["timeouts_seconds"]["validator_request"]:
                    raise TimeoutError("Shared validator request exceeded hard timeout")
        heartbeat.update(worker_exit_codes={n: j["process"].poll() for n, j in self.jobs.items()},
                         progress_advisories=warnings)

    def wait(self, names, heartbeat):
        while any(self.jobs[n]["process"].poll() is None for n in names):
            self.check(heartbeat)
            time.sleep(10)
        self.check(heartbeat)

    def stop_owned(self):
        for job in self.jobs.values():
            p = job["process"]
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        deadline = time.time() + 10
        while any(j["process"].poll() is None for j in self.jobs.values()) and time.time() < deadline:
            time.sleep(.2)
        for job in self.jobs.values():
            if job["process"].poll() is None:
                os.killpg(job["process"].pid, signal.SIGKILL)
            job["process"].wait(timeout=10)

    def run(self):
        with ProgressHeartbeat(self.root / "status.json", phase="contract") as hb:
            try:
                hb.update(workspace_advisory=verify(self.m))
                for lane, mode in enumerate(MODES):
                    self.start(f"contract_{mode}", "contract", lane, mode=mode)
                self.wait(["contract_R", "contract_J"], hb)
                contracts = {mode: read_json(self.root / "contract" / mode / "result.json") for mode in MODES}
                if not all(r["passed"] and not r.get("smoke_only") for r in contracts.values()):
                    raise RuntimeError("Implementation contract failed")
                eligible_lrs = [lr for lr in self.m["training"]["actor_lr_candidates"] if all(
                    contracts[mode]["lr_checks"][str(lr)]["passed"] for mode in MODES)]
                if not eligible_lrs:
                    raise RuntimeError("No common stable actor LR passed numerical canaries")
                chosen_lr = eligible_lrs[0]
                atomic_json(self.root / "contract/result.json", {"passed": True, "actor_lr": chosen_lr,
                            "modes": contracts, "selection": "numerical checks only; no Tune gain tuning"}, overwrite=False)
                hb.update(phase="expanded_sampling", actor_lr=chosen_lr)
                for lane in range(4):
                    self.start(f"sample_{lane}", "sample", lane)
                self.wait([f"sample_{n}" for n in range(4)], hb)
                admission = {}
                for mode in MODES:
                    rows = {split: [read_json(self.root / "sampling" / mode / split / c["name"] / "result.json")
                        for c in self.m["splits"][split]] for split in ("train_diag32", "train_probe64")}
                    admission[mode] = exploration_gate(rows["train_diag32"], rows["train_probe64"], self.m["source_costs"])
                atomic_json(self.root / "exploration_gate.json", admission, overwrite=False)
                hb.update(phase="fit_and_counterfactual", exploration_admission=admission)
                for lane, mode in enumerate(MODES):
                    self.start(f"fit_{mode}", "fit", lane, mode=mode)
                self.start("counterfactual", "counterfactual", 2)
                self.wait(["fit_R", "fit_J", "counterfactual"], hb)
                fits = {mode: read_json(self.root / "fit" / mode / "result.json") for mode in MODES}
                eligible = [arm for arm, config in ARMS.items() if admission[config["mode"]]["passed"]
                            and (not config["elite"] or fits[config["mode"]]["gate"]["passed"])]
                gate = {"eligible_arms": eligible, "exploration": admission,
                        "imitation": {m: f["gate"] for m, f in fits.items()},
                        "old_diagnostic_gate_unchanged": True, "pure_ppo_requires_bc": False}
                atomic_json(self.root / "pilot_admission.json", gate, overwrite=False)
                if not eligible:
                    hb.update(phase="no_pilot_admitted")
                    atomic_json(self.root / "report.json", {"status": "no_pilot_admitted", **gate}, overwrite=False)
                    return
                hb.update(phase="pilot", eligible_arms=eligible)
                self.pilot_plan = pilot_resource_plan(self.m["resources"], len(eligible))
                atomic_json(self.root / "pilot_resources.json", self.pilot_plan, overwrite=False)
                hb.update(pilot_resources=self.pilot_plan)
                self.start("validator", "validator", 3, timeout=self.m["timeouts_seconds"]["suite"])
                pending, running = list(eligible), {}
                while pending or running:
                    self.check(hb)
                    for arm, lane in list(running.items()):
                        if self.jobs[f"pilot_{arm}"]["process"].poll() is not None:
                            del running[arm]
                    for lane in self.pilot_plan["trainers"]:
                        if pending and lane not in running.values():
                            arm = pending.pop(0)
                            mode = ARMS[arm]["mode"]
                            seconds = contracts[mode]["group_seconds"]
                            timeout = min(self.m["timeouts_seconds"]["pilot_cap"], max(86400, seconds * 240 * 2.5))
                            self.start(f"pilot_{arm}", "train", lane, arm=arm, timeout=timeout)
                            running[arm] = lane
                    hb.update(queued_arms=pending, running_arms=running)
                    if pending or running:
                        time.sleep(10)
                queue = LocalEvaluationQueue(self.root / "validator")
                while list((queue.root / "pending").glob("*.json")) or list((queue.root / "running").glob("*.json")):
                    self.check(hb)
                    time.sleep(10)
                atomic_json(queue.root / "STOP", {"reason": "all admitted pilots completed"}, overwrite=False)
                self.wait(["validator"], hb)
                results = {arm: read_json(self.root / "pilot" / arm / "result.json") for arm in eligible}
                atomic_json(self.root / "report.json", {"status": "pilot_completed", "admission": gate,
                    "pilots": results, "automatic_confirmation": False, "automatic_finalblind": False}, overwrite=False)
                hb.update(phase="pilot_completed", passed_arms=[a for a, r in results.items() if r["passed"]])
            except BaseException:
                atomic_json(self.root / "failure.json", {"traceback": traceback.format_exc(), "unix": time.time()}, overwrite=False)
                raise
            finally:
                self.stop_owned()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--audit", type=Path, default=ROOT / "result/hkbz_train_logs/stage3_sampling_audit_half_20260906_r2/manifest.json")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("run")
    p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        manifest = read_json(args.manifest)
        with (Path(manifest["root"]) / "suite.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (Path(manifest["root"]) / "status.json").exists():
                raise FileExistsError("No implicit suite retry; preserve the original execution")
            Suite(manifest).run()


if __name__ == "__main__":
    main()
