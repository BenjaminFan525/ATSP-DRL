#!/usr/bin/env python3
"""Three logical policies on eight GPUs, one shared asynchronous validator pool."""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json, verify_cases, BASE_MANIFEST
from onpolicy.utils.stage3_full_policy import ARMS, identity, verify, resource_plan, RECOVERY_FIELDS, assert_recovery_protocol
from onpolicy.utils.stage3_numerics import RUNTIME
from onpolicy.utils.stage3_representation import (RepresentationQueue, discover_topology, external_resources,
    cpu_text, screen_candidates, risk_pass, endpoint_pass)
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat
from onpolicy.scripts.train.run_stage3_representation import Suite

OVERLAYS = (
    "onpolicy/algorithms/utils/gnn.py",
    "onpolicy/algorithms/utils/stage3_encoder.py",
    "onpolicy/runner/shared/stage3_representation_engine.py",
    "onpolicy/runner/shared/stage3_research_engine.py",
    "onpolicy/scripts/train/stage3_representation_worker.py",
    "onpolicy/utils/stage3_distributed.py", "onpolicy/utils/stage3_full_policy.py",
    "onpolicy/utils/stage3_numerics.py",
    "onpolicy/scripts/train/check_stage3_full_policy_resume.py",
    "onpolicy/scripts/train/stage3_full_policy_worker.py",
    "onpolicy/scripts/train/run_stage3_full_policy.py",
    "onpolicy/scripts/train/launch_stage3_full_policy_all.sh",
    "onpolicy/envs/HKBZ/test/test_stage3_full_policy.py",
    "onpolicy/envs/HKBZ/test/test_stage3_numerics.py",
    "STAGE3_FULL_POLICY_20260909.md",
    "STAGE3_NUMERICAL_RECOVERY_20260909.md",
)


def prepare(args):
    prior = read_json(args.prior)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new study directory; retain all historical artifacts")
    recovery = bool(getattr(args, "recovery", False))
    if recovery:
        if (prior.get("schema") != "stage3-full-policy-single-seed-v1"
                or read_json(Path(prior["root"]) / "status.json")["status"] != "failed"
                or (Path(prior["root"]) / "training_admission.json").exists()):
            raise ValueError("This recovery is restricted to the failed pre-training numerical admission")
    if digest_file(prior["source"]["path"]) != prior["source"]["sha256"]:
        raise ValueError("Original C0 changed")
    if digest_json(prior["contract"]) != prior["contract_sha256"]:
        raise ValueError("Historical environment contract changed")
    for key, cases in prior["splits"].items():
        verify_cases(cases, training=key.startswith("train"))
    gpus = []
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits"], text=True).splitlines():
        index, uuid, memory = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(index), "uuid": uuid, "memory_mib": int(memory)})
    if [g["index"] for g in gpus] != list(range(8)):
        raise ValueError("Expected all eight GPUs")
    cores = discover_topology()
    plan = resource_plan(cores)
    files = {}
    # Inherit the verified historical runtime. Do not capture unrelated baseline edits.
    for relative, checksum in prior["code"]["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe snapshot path")
        source = Path(prior["execution"]["code_root"]) / relative
        if digest_file(source) != checksum:
            raise ValueError(f"Historical snapshot changed: {relative}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative] = checksum
    for relative in OVERLAYS:
        source, target = ROOT / relative, output / "source" / relative
        checksum = digest_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest_file(target) != checksum:
            raise ValueError("Source changed while snapshotting")
        files[relative] = checksum
    diagnostic = []
    for distribution in ("iid", "ood_stress"):
        diagnostic.extend([c for c in prior["splits"]["train_pilot120"] if c["distribution"] == distribution][:2])
    if len(diagnostic) != 4:
        raise ValueError("Need four training-only canary cases")
    training = {**prior["training"], "seed": 2026090903, "seeds": [2026090903],
        "save_every_episodes": "every complete global group, all rank states",
        "optimizer_recipe": "O0; globally case-balanced gradient mean before global clipping/Adam",
        "global_batch_candidates": [24, 12], "batch_immutable_after_admission": True,
        "distributed_world_sizes": {a: len(v["gpus"]) for a, v in ARMS.items()}}
    manifest = {"schema": "stage3-full-policy-single-seed-v1", "root": str(output), "created_unix": time.time(),
        "material_passport": {"origin_skill": "academic-research-suite/experiment-agent", "origin_mode": "run",
            "origin_date": "2026-09-09", "verification_status": "prepared; outcomes unverified",
            "version_label": "full_policy_v1_single_seed", "data_classification": "internal; no upload"},
        "prior_manifest": str(args.prior.resolve()), "prior_manifest_sha256": prior["manifest_sha256"],
        "source": prior["source"], "source_costs": prior["source_costs"], "splits": prior["splits"],
        "contract": prior["contract"], "contract_sha256": prior["contract_sha256"],
        "input_files": {str(args.prior.resolve()): digest_file(args.prior), str(BASE_MANIFEST): digest_file(BASE_MANIFEST),
            prior["source"]["baseline"]["path"]: prior["source"]["baseline"]["sha256"]},
        "code": {"files": files, "sha256": digest_json(files)},
        "execution": {"code_root": str(output / "source"), "workspace_root": str(ROOT),
            "workspace_code_policy": "advisory; unrelated baseline edits never stop this study",
            "python": sys.executable, "packages": {n: importlib.metadata.version(n) for n in prior["execution"]["packages"]}},
        "resources": {"gpus": gpus, "core_groups": cores, "plan": plan, "validator_workers": 2,
            "validator_width": 3, "trainer_max_width": 8, "memory_high_gib": 96, "memory_max_gib": 108,
            "validator_queue_scope": "one durable queue, two shared consumers across all GPUs and arms",
            "external_job_policy": "wait for available GPUs before start; never preempt external jobs"},
        "arms": ARMS, "training": training, "gates": prior["gates"], "numerics": RUNTIME,
        "diagnostic": {"cases": diagnostic, "global_batch": 8, "resume_model_atol": 2e-6,
            "single_vs_distributed_model_atol": 2e-6, "optimizer_atol": 2e-6, "optimizer_rtol": 2e-5,
            "full_history_logp_atol": .002, "teacher_fit_rerun": False,
            "scope": "C0 migration, full-depth updates, all state/RNG resume and single/multi-GPU objective parity"},
        "capacity": {"candidates": [24, 12], "rank_reserved_max_gib": 19, "service_sampled_max_gib": 96,
            "dense_case_selection": "largest observed retained graph in the four C_PRIVATE canary cases, not a global worst-case guarantee",
            "fallback": "only a preregistered capacity rejection; implementation/algorithm failures stop admission"},
        "timeouts_seconds": {"suite": 864000, "reference": 43200, "contract": 43200, "capacity": 43200,
            "train": 259200, "validator": 864000, "validator_request": 21600, "progress_warning": 900},
        "automatic_multi_seed": False, "automatic_4800": False, "automatic_finalblind": False,
        "exposure": "same exposed Train/Tune; no validation labels used by PPO; Finalblind unopened",
        "comparison_scope": "controlled A/B/C within this study, not matched-update causal comparison with previous variable-batch studies"}
    if recovery:
        assert_recovery_protocol(prior, manifest)
        if manifest["execution"]["packages"] != prior["execution"]["packages"]:
            raise ValueError("A numerical bug recovery cannot silently upgrade dependencies")
        evidence = {}
        for name in ("legacy_gradients", "deterministic_same_gpu", "deterministic_native", "deterministic_stable_pool"):
            path = Path(prior["root"]) / "diagnosis" / name / "result.json"
            row = read_json(path)
            if row["code_sha256"] != prior["code"]["sha256"] or not all(row["checkpoint_restore_exact"].values()):
                raise ValueError("Backend diagnostic evidence does not match the failed run")
            expected = name in ("deterministic_native", "deterministic_stable_pool")
            if row["fixed_weight_and_input_gradient_repeat_exact"] != expected:
                raise ValueError("Recovery diagnosis not supported by fixed-input checks")
            evidence[name] = {"path": str(path), "sha256": digest_file(path)}
            manifest["input_files"][str(path)] = digest_file(path)
        status_path = Path(prior["root"]) / "status.json"
        manifest["input_files"][str(status_path)] = digest_file(status_path)
        failure_log = Path(prior["root"]) / "logs/contract/c0/B_SHARED/rank0.log"
        manifest["input_files"][str(failure_log)] = digest_file(failure_log)
        manifest["recovery"] = {"parent_manifest": str(args.prior.resolve()),
            "parent_manifest_file_sha256": digest_file(args.prior), "parent_root": prior["root"],
            "reason": "CUDA numerical nondeterminism including max-pool tie argmax; strict backend plus stable first-node pooling",
            "diagnosis": evidence, "preserved_fields": list(RECOVERY_FIELDS), "training_episodes_before_failure": 0,
            "reuse_diagnostic_weights_for_training": False, "all_admissions_reexecuted": True,
            "thresholds_unchanged": True}
    manifest["manifest_sha256"] = identity(manifest)
    atomic_json(output / "manifest.json", manifest, overwrite=False)
    print(output / "manifest.json", flush=True)


def host_usage():
    relative = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    memory = {parts[0].rstrip(":"): int(parts[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if len(parts := line.split()) >= 3 and parts[2] == "kB"}
    return {"unix": time.time(), "cgroup_memory_bytes": int((group / "memory.current").read_text()),
        "host_available_bytes": memory["MemAvailable"], "memory_events": (group / "memory.events").read_text(),
        "scope": "whole study service including children; sampled, not exact peak"}


def compare_checkpoints(reference, distributed, atol):
    import torch
    from onpolicy.scripts.train.stage3_local_worker import nested_close
    before = torch.load(reference, map_location="cpu", weights_only=False)
    after = torch.load(distributed, map_location="cpu", weights_only=False)
    # ResearchEngine.save uses the same complete payload for every architecture.
    keys = ("model", "actor_optim", "critic_optim", "role_value_normalizers")
    missing = [k for k in keys if k not in before or k not in after]
    if missing:
        raise ValueError(f"Missing checkpoint parity fields: {missing}")
    error = max(float((before["model"][k] - after["model"][k]).abs().max()) for k in before["model"])
    optimizer_ok = all(nested_close(before[k], after[k]) for k in keys[1:])
    result = {"model_max_abs_error": error, "optimizer_and_normalizer_close": optimizer_ok,
        "same_policy_updates": before["policy_updates"] == after["policy_updates"],
        "reference_sha256": digest_file(reference), "distributed_sha256": digest_file(distributed)}
    result["passed"] = error <= atol and optimizer_ok and result["same_policy_updates"]
    return result


class FullSuite(Suite):
    def __init__(self, manifest, manifest_path):
        super().__init__(manifest, manifest_path=manifest_path)
        self.plan = manifest["resources"]["plan"]
        self.sampled_peak = 0
        self.phase_sampled_peak = 0
        self.last_resource_sample = 0

    @staticmethod
    def task_key(phase, name):
        if phase not in ("reference", "contract", "capacity", "train", "validator") or not re.fullmatch(r"[A-Za-z0-9_/-]+", name) or ".." in name or name.startswith("/"):
            raise ValueError("Unsafe worker identity")
        return f"{phase}/{name}"

    def start(self, name, phase, placement, **options):
        key = self.task_key(phase, name)
        output = self.root / ("validator/workers" if phase == "validator" else phase) / name
        command_path, log_path = self.root / "commands" / f"{key}.json", self.root / "logs" / f"{key}.log"
        if key in self.jobs or output.exists() or command_path.exists() or log_path.exists():
            raise FileExistsError(f"No implicit retry/overwrite: {output}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=next(g["uuid"] for g in self.m["resources"]["gpus"] if g["index"] == placement["gpu"]),
            HKBZ_STAGE3_WORKSPACE_ROOT=self.m["execution"]["workspace_root"], PYTHONPATH=str(ROOT),
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MALLOC_ARENA_MAX="2",
            CUBLAS_WORKSPACE_CONFIG=self.m["numerics"]["cublas_workspace_config"],
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        command = ["/usr/bin/taskset", "--cpu-list", cpu_text(placement["cpus"]), self.m["execution"]["python"], "-B", "-u",
            str(ROOT / self.m["execution"].get("worker", "onpolicy/scripts/train/stage3_full_policy_worker.py")), str(self.manifest_path),
            "--phase", phase, "--output", str(output)]
        for option, value in options.items():
            command.extend(("--" + option.replace("_", "-"), str(value)))
        atomic_json(command_path, {"command": command, "placement": placement, "created_unix": time.time()}, overwrite=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.jobs[key] = {"process": process, "phase": phase, "name": name, "placement": placement,
            "output": output, "started_unix": time.time(), "options": options}
        return key

    def check(self, hb):
        super().check(hb)
        if time.time() - self.last_resource_sample >= 10:
            usage = host_usage()
            self.sampled_peak = max(self.sampled_peak, usage["cgroup_memory_bytes"])
            self.phase_sampled_peak = max(self.phase_sampled_peak, usage["cgroup_memory_bytes"])
            self.reservations = external_resources(self.m["resources"]["gpus"], os.getpid(), self.reservations)
            usage["external_gpu_jobs_advisory"] = self.reservations
            with (self.root / "resource_samples.jsonl").open("a") as out:
                out.write(json.dumps(usage) + "\n")
            atomic_json(self.root / "resource_leases.json", {"plan": self.plan,
                "external_reservations_advisory": self.reservations, "usage": usage,
                "sampled_peak_gib": self.sampled_peak / 2**30})
            hb.update(sampled_memory_peak_gib=self.sampled_peak / 2**30,
                external_gpu_jobs_advisory=self.reservations)
            self.last_resource_sample = time.time()

    def wait_available(self, hb):
        os.sched_setaffinity(0, self.plan["controller"])
        while True:
            self.reservations = external_resources(self.m["resources"]["gpus"], os.getpid(), self.reservations)
            if not self.reservations:
                break
            hb.update(event="waiting_for_external_gpu_jobs", reservations=self.reservations)
            time.sleep(10)

    def run_group(self, phase, label, hb, *, arms=None, reference=False, batch_size=8, dense_case=None,
                  until=960, previous=None):
        keys = []
        arm_configs = self.m.get("arms", ARMS)
        for arm in (arms if arms is not None else arm_configs):
            gpus = arm_configs[arm]["gpus"][:1] if reference else arm_configs[arm]["gpus"]
            rendezvous = self.root / "rendezvous" / f"{phase}_{label}_{arm}"
            rendezvous.parent.mkdir(parents=True, exist_ok=True)
            for rank, gpu in enumerate(gpus):
                options = {"arm": arm, "rank": rank, "world_size": len(gpus), "init_file": rendezvous,
                    "batch_size": batch_size, "until": until}
                if dense_case:
                    options["dense_case"] = dense_case
                if previous:
                    options["resume_commit"] = self.root / f"train/to{previous}/{arm}/commits/episodes_{previous:06d}.json"
                keys.append(self.start(f"{label}/{arm}/rank{rank}", phase, self.plan["trainers"][str(gpu)], **options))
        while any(self.jobs[key]["process"].poll() is None for key in keys):
            self.check(hb)
            time.sleep(5)
        self.check(hb)
        results = {arm: [read_json(self.jobs[key]["output"] / "result.json") for key in keys
                        if self.jobs[key]["options"]["arm"] == arm]
                   for arm in (arms if arms is not None else arm_configs)}
        return results

    def contracts(self, hb):
        hb.update(phase="single_gpu_reference")
        references = self.run_group("reference", "c0", hb, reference=True)
        hb.update(phase="distributed_contract")
        distributed = self.run_group("contract", "c0", hb)
        comparisons = {}
        for arm in ARMS:
            ref, rows = references[arm][0], distributed[arm]
            if any(not r["passed"] or r["trace_identity"] != ref["trace_identity"] for r in rows):
                raise RuntimeError(f"Single/multi-GPU trajectory parity failed: {arm}")
            comparison = compare_checkpoints(ref["after_update_checkpoint"], rows[0]["after_update_checkpoint"],
                self.m["diagnostic"]["single_vs_distributed_model_atol"])
            comparisons[arm] = comparison
        atomic_json(self.root / "contract/parity.json", {"arms": comparisons,
            "passed": all(c["passed"] for c in comparisons.values())}, overwrite=False)
        if not all(c["passed"] for c in comparisons.values()):
            raise RuntimeError("Single/multi-GPU model/Adam update parity failed")
        for arm in ARMS:
            self.evaluation(arm, 0, hb)
            rows = self.queue.poll(f"{arm}_e000000")["evaluation"]["cases"]
            if len(rows) != len(self.m["splits"]["tune"]) or any(
                    abs(r["makespan"] - self.m["source_costs"][r["case_id"]]) > 1e-6 for r in rows):
                raise RuntimeError(f"C0 Tune60 changed for {arm}")
        atomic_json(self.root / "contract/admission.json", {"passed": True, "c0_tune60_arms": list(ARMS),
            "source": self.m["source"], "diagnostic_checkpoints_never_initialize_rl": True}, overwrite=False)
        return max(distributed["C_PRIVATE"][0]["records"], key=lambda r: r["max_graph_tensor_bytes"])["case_id"]

    def capacity(self, dense_case, hb):
        candidates = []
        for batch in self.m["capacity"]["candidates"]:
            self.phase_sampled_peak = 0
            hb.update(phase=f"capacity_b{batch}", dense_case=dense_case)
            normal = self.run_group("capacity", f"b{batch}_normal", hb, batch_size=batch)
            dense = self.run_group("capacity", f"b{batch}_dense", hb, batch_size=batch, dense_case=dense_case)
            rows = [row for result in (normal, dense) for group in result.values() for row in group]
            peak = max(row["peak_reserved_gib"] for row in rows)
            passed = (all(row["passed"] for row in rows) and peak <= self.m["capacity"]["rank_reserved_max_gib"]
                      and self.phase_sampled_peak / 2**30 <= self.m["capacity"]["service_sampled_max_gib"])
            candidate = {"batch_size": batch, "passed": passed, "max_rank_reserved_gib": peak,
                "service_sampled_peak_gib": self.phase_sampled_peak / 2**30, "tested_with_shared_validator_pool": True}
            candidates.append(candidate)
            atomic_json(self.root / f"capacity/b{batch}_decision.json", candidate, overwrite=False)
            if passed:
                result = {"passed": True, "eligible_arms": list(ARMS), "batch_size": batch,
                    "candidates": candidates, "dense_case": dense_case, "created_unix": time.time(),
                    "protocol_sha256": self.m["manifest_sha256"], "source": self.m["source"],
                    "updates_at_960_per_arm": 960 // batch * 2, "ppo_epochs": 2, "tbptt_steps": 8,
                    "logical_batch_and_world_sizes_immutable": True, "initialization": "original C0, never canary weights"}
                atomic_json(self.root / "training_admission.json", result, overwrite=False)
                return batch
        raise RuntimeError("No capacity candidate admitted; no formal training launched")

    def run(self):
        with ProgressHeartbeat(self.root / "status.json", phase="preflight", active_manifest=str(self.manifest_path)) as hb:
            try:
                hb.update(workspace_advisory=verify(self.m))
                self.wait_available(hb)
                for i, placement in enumerate(self.plan["validators"]):
                    self.start(f"pool{i}", "validator", placement)
                dense_case = self.contracts(hb)
                batch = self.capacity(dense_case, hb)
                hb.update(phase="screen_960", batch_size=batch)
                self.run_group("train", "to960", hb, batch_size=batch)
                screen = {arm: self.evaluation(arm, 960, hb) for arm in ARMS}
                chosen = screen_candidates(screen)
                atomic_json(self.root / "screen_admission.json", {"results": screen, "selected": chosen,
                    "threshold": .005, "max_extensions": 2, "created_unix": time.time()}, overwrite=False)
                outcomes = {}
                for endpoint in (1440, 1920):
                    if not chosen:
                        break
                    hb.update(phase=f"pilot_to_{endpoint}")
                    self.run_group("train", f"to{endpoint}", hb, arms=chosen, batch_size=batch,
                        until=endpoint, previous=endpoint - 480)
                    survivors = []
                    for arm in chosen:
                        result = self.evaluation(arm, endpoint, hb)
                        outcomes.setdefault(arm, []).append(result)
                        if result["summary"]["gain_fraction"] >= .01 and risk_pass(result["summary"]):
                            survivors.append(arm)
                    atomic_json(self.root / f"pilot_admission_{endpoint}.json", {"results": outcomes,
                        "eligible_next_chunk": survivors if endpoint == 1440 else []}, overwrite=False)
                    chosen = survivors
                atomic_json(self.root / "result.json", {"completed": True, "screen": screen, "pilot": outcomes,
                    "passed_arms": [a for a, r in outcomes.items() if endpoint_pass(r)],
                    "single_seed_only": True, "finalblind_opened": False, "automatic_extension_4800": False,
                    "evidence_scope": "single-seed architecture screening, not cross-seed robustness"}, overwrite=False)
                if self.queue.pending_count():
                    raise RuntimeError("Cannot close validator pool with pending requests")
                atomic_json(self.root / "validator/STOP", {"reason": "all validations drained"}, overwrite=False)
                while any(j["phase"] == "validator" and j["process"].poll() is None for j in self.jobs.values()):
                    self.check(hb)
                    time.sleep(2)
            except BaseException:
                hb.update(event="suite_failed", traceback=traceback.format_exc())
                raise
            finally:
                self.stop_owned()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recovery", action="store_true", help="explicit recovery of a failed pre-training full-policy admission")
    r = sub.add_parser("run")
    r.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    manifest = read_json(args.manifest)
    if (Path(manifest["root"]) / "status.json").exists():
        raise FileExistsError("No implicit suite restart; failed artifacts must remain intact")
    with (Path(manifest["root"]) / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        FullSuite(manifest, args.manifest).run()


if __name__ == "__main__":
    main()
