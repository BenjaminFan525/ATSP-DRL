#!/usr/bin/env python3
"""Single-seed representation study: bounded diagnosis, eight arms, <=2 pilots.

Only this suite's children are controlled. Existing GPU/CPU jobs retain leases.
All evaluation requests use one durable queue and a two-worker shared pool.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
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
from onpolicy.utils.stage3_research import (atomic_json, read_json, digest_file, digest_json,
    code_changes, verify_cases, BASE_MANIFEST)
from onpolicy.utils.stage3_representation import (ARMS, RepresentationQueue, discover_topology,
    allocate_resources, external_resources, cpu_text, screen_candidates, risk_pass, endpoint_pass)
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat

OVERLAYS = (
    "onpolicy/algorithms/utils/gnn.py", "onpolicy/algorithms/utils/stage3_encoder.py",
    "onpolicy/algorithms/gnn_mappo/algorithm/gnn_actor_critic.py",
    "onpolicy/runner/shared/stage3_research_engine.py",
    "onpolicy/runner/shared/stage3_representation_engine.py",
    "onpolicy/utils/stage3_representation.py",
    "onpolicy/scripts/train/run_stage3_representation.py",
    "onpolicy/scripts/train/stage3_representation_worker.py",
    "onpolicy/scripts/train/launch_stage3_representation_all.sh",
    "onpolicy/envs/HKBZ/test/test_stage3_representation.py",
    "onpolicy/envs/HKBZ/test/test_stage3_representation_orchestration.py",
    "onpolicy/utils/stage3_representation_recovery.py",
    "onpolicy/scripts/train/launch_stage3_representation_resume.sh",
    "STAGE3_REPRESENTATION_20260908.md",
    "onpolicy/algorithms/gnn_mappo/algorithm/MAPPOPolicy.py",
    "onpolicy/utils/stage3_performance.py",
    "onpolicy/utils/stage3_hot_update.py",
    "onpolicy/scripts/train/hot_update_stage3_representation.py",
    "onpolicy/scripts/train/launch_stage3_representation_hot_update.sh",
    "onpolicy/scripts/train/check_stage3_performance.py",
    "onpolicy/envs/HKBZ/test/test_stage3_performance.py",
    "onpolicy/envs/HKBZ/test/test_stage3_hot_update.py",
    "STAGE3_PERFORMANCE_20260908.md",
)


def identity(manifest):
    return digest_json({k:v for k,v in manifest.items() if k != "manifest_sha256"})


def prepare(args):
    prior = read_json(args.prior)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; old runs are preserved")
    if not (Path(prior["root"])/"administrative_stop.json").exists():
        raise ValueError("Expected the explicitly stopped historical study")
    if digest_file(prior["source"]["path"]) != prior["source"]["sha256"]:
        raise ValueError("C0 changed")
    for key, rows in prior["splits"].items():
        verify_cases(rows, training=key.startswith("train"))
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total",
                                       "--format=csv,noheader,nounits"], text=True)
    gpus = []
    for line in gpu_text.splitlines():
        index, uuid, memory = [v.strip() for v in line.split(",")]
        gpus.append({"index": int(index), "uuid": uuid, "memory_mib": int(memory)})
    if len(gpus) != 8:
        raise ValueError("This preregistered resource geometry expects eight GPUs")
    files = {}
    for relative, sha in prior["code"]["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe snapshot path")
        source = Path(prior["execution"]["code_root"])/relative
        if digest_file(source) != sha:
            raise ValueError(f"Historical snapshot changed: {relative}")
        target = output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative] = sha
    for relative in OVERLAYS:
        source, target = ROOT/relative, output/"source"/relative
        sha = digest_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest_file(target) != sha:
            raise ValueError("Source changed while snapshotting")
        files[relative] = sha
    inputs = {str(args.prior.resolve()): digest_file(args.prior), str(BASE_MANIFEST): digest_file(BASE_MANIFEST)}
    teachers = {}
    for case in prior["splits"]["train_fit16"]:
        path = Path(prior["root"])/"sampling/J/train_diag32"/case["name"]/"result.json"
        result = read_json(path)
        inputs[str(path)] = digest_file(path)
        for group in result["groups"]:
            inputs[group["path"]] = group["sha256"]
            trace = read_json(group["path"])["trace"]
            inputs[trace["path"]] = trace["sha256"]
        teachers[case["path"]] = {"path": str(path), "sha256": digest_file(path)}
    mechanisms = {}
    for mode in ("R", "J"):
        for group in (1, 121, 181):
            path = Path(prior["root"])/f"pilot/{mode}_PPO/rollouts/group_{group:04d}.json"
            trace = read_json(path)["trace"]
            checkpoint = (prior["source"]["path"] if group == 1 else str(Path(prior["root"])/
                          f"pilot/{mode}_PPO/models/episodes_{(group-1)*8:06d}.pt"))
            inputs.update({str(path): digest_file(path), trace["path"]: trace["sha256"], checkpoint: digest_file(checkpoint)})
            mechanisms[f"{mode}_g{group:04d}"] = {"mode": mode, "group": group,
                "checkpoint": checkpoint, "archive": {"path": str(path), "sha256": digest_file(path)}}
    manifest = {"schema": "stage3-representation-single-seed-v1", "root": str(output), "created_unix": time.time(),
        "material_passport": {"origin_skill": "academic-research-suite/experiment-agent", "origin_mode": "run",
            "origin_date": "2026-09-08", "verification_status": "prepared; outcomes unverified",
            "version_label": "representation_v1_single_seed", "data_classification": "internal; no upload"},
        "prior_manifest": str(args.prior.resolve()), "prior_manifest_sha256": prior["manifest_sha256"],
        "source": prior["source"], "source_costs": prior["source_costs"],
        "splits": prior["splits"], "contract": prior["contract"], "contract_sha256": prior["contract_sha256"],
        "teachers": teachers, "mechanisms": mechanisms, "input_files": inputs,
        "code": {"files": files, "sha256": digest_json(files)},
        "execution": {"code_root": str(output/"source"), "workspace_root": str(ROOT), "workspace_code_policy": "advisory",
            "python": sys.executable, "packages": {name: importlib.metadata.version(name)
                for name in ("torch", "torch-geometric", "numpy", "scipy", "gymnasium", "PyYAML")}},
        "resources": {"gpus": gpus, "core_groups": discover_topology(), "validator_workers": 2,
            "trainer_width": 8, "validator_width": 3, "trainer_memory_fraction": .65,
            "validator_memory_fraction": .15, "cpu_policy": "2 controller + 6 validator + 7 per trainer physical cores; SMT siblings together",
            "external_job_policy": "reserve live external GPU PIDs and their CPU affinities; never preempt",
            "memory_high_gib": 72, "memory_max_gib": 88},
        "arms": ARMS, "training": {"seed": 2026090803, "seeds": [2026090803], "mode": "J",
            "actor_lr": 5e-6, "critic_lr": 1e-4, "ppo_epochs": 2, "clip_mode": "global", "clip_norm": 1.,
            "ppo_clip": .2, "tbptt_steps": 8, "gamma": 1., "reward_coef": .01,
            "kl_limit": .02, "post_update_hard_kl": .04, "screen_episodes": 960, "pilot_episodes": 1920,
            "save_every_episodes": 240, "evaluate_every_episodes": 480, "max_extensions": 2,
            "advantage": "source", "reduction": "case mean -> trajectory mean -> time/role sum",
            "optimizer_recipe": "O0 unchanged; diagnostic interventions never silently select the training recipe"},
        "fit": {"single_case_count": 2, "single_case_updates": 16, "fit16_passes": 5, "lr": 1e-5,
            "recovery_min": .5, "diagnostic_only": True, "blocks_pure_ppo": False},
        "diagnostic": {"fixed_batch_interventions": ["O0", "lr2", "lr4", "per_group_clip", "tbptt32"],
            "gradient_components": ["all", "positive", "negative", "winner", "plane", "device", "transporter"]},
        "gates": {"screen_gain_min": .005, "pilot_gain_min_both_1440_1920": .01,
            "desired_gain": .02, "ood_stress_and_joint_regression_max": .005,
            "tail_regression_max": .01, "bad5_fraction_max": .05},
        "timeouts_seconds": {"suite": 864000, "contract": 43200, "mechanism": 86400, "fit": 172800,
            "train": 259200, "validator_request": 21600, "progress_warning": 900},
        "automatic_multi_seed": False, "automatic_4800": False, "automatic_finalblind": False,
        "iga": {"automatic_rerun": False, "reason": "Mechanism/representation screen first; no historical soft-cost comparison with this hard-contract run"},
        "exposure": "Train/Probe/Tune historically exposed; Finalblind not opened; no validation labels used for training"}
    manifest["manifest_sha256"] = identity(manifest)
    atomic_json(output/"manifest.json", manifest, overwrite=False)
    print(output/"manifest.json", flush=True)


def verify(manifest, inputs=True):
    if identity(manifest) != manifest["manifest_sha256"] or manifest["arms"] != ARMS:
        raise ValueError("Protocol identity changed")
    if ROOT.resolve() != Path(manifest["execution"]["code_root"]).resolve():
        raise ValueError("Execute only from the frozen source snapshot")
    if code_changes(manifest["code"]["files"], ROOT):
        raise ValueError("Frozen execution source changed")
    if digest_file(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("C0 changed")
    if inputs:
        if manifest.get("hot_update"):
            from onpolicy.utils.stage3_hot_update import validate_hot_update
            validate_hot_update(manifest)
        elif manifest.get("recovery"):
            from onpolicy.utils.stage3_representation_recovery import validate_recovery
            validate_recovery(manifest)
        for path, sha in manifest["input_files"].items():
            if digest_file(path) != sha:
                raise ValueError(f"Frozen input changed: {path}")
        for key, cases in manifest["splits"].items():
            verify_cases(cases, training=key.startswith("train"))
        for key, version in manifest["execution"]["packages"].items():
            if importlib.metadata.version(key) != version:
                raise ValueError(f"Package version changed: {key}")
    return code_changes(manifest["code"]["files"], manifest["execution"]["workspace_root"])


class Suite:
    def __init__(self, manifest, *, manifest_path=None, resume_fit=False):
        self.m, self.root = manifest, Path(manifest["root"])
        self.manifest_path = Path(manifest_path or self.root/"manifest.json").resolve()
        self.resume_fit = resume_fit
        self.jobs, self.reservations = {}, []
        self.queue = RepresentationQueue(self.root/"validator")

    @staticmethod
    def task_key(phase, name):
        if phase not in ("contract", "mechanism", "fit", "train", "validator") or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unsafe or unknown phase/task name")
        return f"{phase}/{name}"

    def resources(self):
        self.reservations = external_resources(self.m["resources"]["gpus"], os.getpid(), self.reservations)
        busy = {r["gpu"] for r in self.reservations}
        excluded = {c for r in self.reservations for c in r["cpus"]}
        for name, job in self.jobs.items():
            if job["process"].poll() is None and (job["placement"]["gpu"] in busy
                    or set(job["placement"]["cpus"]) & excluded):
                raise RuntimeError(f"New external job conflicts with {name}; stop owned study only, retain artifacts")
        free = [g["index"] for g in self.m["resources"]["gpus"] if g["index"] not in busy]
        plan = allocate_resources(self.m["resources"]["core_groups"], excluded, free)
        if plan["controller"]:
            os.sched_setaffinity(0, plan["controller"])
        atomic_json(self.root/"resource_leases.json", {"updated_unix": time.time(),
            "external_reservations": self.reservations, "plan": plan})
        return plan

    def start(self, name, phase, placement, **options):
        key = self.task_key(phase, name)
        output = self.root/({"validator": "validator/workers"}.get(phase, phase))/name
        command_path = self.root/"commands"/f"{key}.json"
        log_path = self.root/"logs"/f"{key}.log"
        if key in self.jobs or (output/"status.json").exists() or command_path.exists() or log_path.exists():
            raise FileExistsError(f"No implicit overwrite/retry: {output}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=next(g["uuid"] for g in self.m["resources"]["gpus"] if g["index"] == placement["gpu"]),
            HKBZ_STAGE3_CUDA_MEMORY_FRACTION=str(placement["cuda_memory_fraction"]),
            HKBZ_STAGE3_WORKSPACE_ROOT=self.m["execution"]["workspace_root"], PYTHONPATH=str(ROOT),
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        command = ["/usr/bin/taskset", "--cpu-list", cpu_text(placement["cpus"]), self.m["execution"]["python"], "-u",
            str(ROOT/"onpolicy/scripts/train/stage3_representation_worker.py"), str(self.manifest_path),
            "--phase", phase, "--output", str(output)]
        for option_name, value in options.items():
            command.extend(("--"+option_name.replace("_", "-"), str(value)))
        atomic_json(command_path, {"command": command, "placement": placement, "task_key": key,
            "created_unix": time.time()}, overwrite=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.jobs[key] = {"process": process, "phase": phase, "name": name, "placement": placement,
            "output": output, "started_unix": time.time(), "options": options}

    def check(self, heartbeat):
        warnings = []
        for name, job in self.jobs.items():
            rc = job["process"].poll()
            if rc not in (None, 0):
                raise RuntimeError(f"Worker {name} failed rc={rc}; artifacts retained, no automatic retry")
            if rc is None:
                if time.time()-job["started_unix"] > self.m["timeouts_seconds"].get(job["phase"], 864000):
                    raise TimeoutError(f"Worker hard timeout: {name}")
                status = job["output"]/"status.json"
                if status.exists():
                    state = read_json(status)
                    if state.get("request_started_unix") and time.time()-state["request_started_unix"] > self.m["timeouts_seconds"]["validator_request"]:
                        raise TimeoutError(f"Validator request timed out: {name}")
                    idle = job["phase"] == "validator" and not state.get("request_id")
                    if not idle and time.time()-state.get("last_progress_unix", time.time()) > self.m["timeouts_seconds"]["progress_warning"]:
                        warnings.append({"worker": name, "kind": "no_actual_progress_advisory"})
        heartbeat.update(worker_exit_codes={n:j["process"].poll() for n,j in self.jobs.items()},
                         progress_advisories=warnings, pending_validations=self.queue.pending_count())

    def run_tasks(self, tasks, heartbeat):
        waiting = list(tasks)
        names = [self.task_key(t["phase"], t["name"]) for t in tasks]
        if len(set(names)) != len(names) or any(name in self.jobs for name in names):
            raise ValueError("Repeated phase/task identity; no implicit retry")
        while waiting or any(n in self.jobs and self.jobs[n]["process"].poll() is None for n in names):
            self.check(heartbeat)
            plan = self.resources()
            # Validator pool persists across arms/phases. CPU/GPU leases are
            # checked against *actual* placements if external leases change.
            live = [j for j in self.jobs.values() if j["process"].poll() is None]
            if not any(j["phase"] == "validator" for j in live) and plan["validators"]:
                for i, placement in enumerate(plan["validators"]):
                    suffix = "_" + self.m["recovery"]["attempt_id"] if self.resume_fit else ""
                    self.start(f"pool{i}{suffix}", "validator", placement)
                live = [j for j in self.jobs.values() if j["process"].poll() is None]
            trainer_gpus = {j["placement"]["gpu"] for j in live if j["phase"] != "validator"}
            busy_cpus = {c for j in live for c in j["placement"]["cpus"]}
            unavailable = busy_cpus | set(plan["controller"]) | {c for r in self.reservations for c in r["cpus"]}
            spare_cores = [core for core in self.m["resources"]["core_groups"] if not set(core) & unavailable]
            for placement in plan["trainers"]:
                if not waiting:
                    break
                if placement["gpu"] in trainer_gpus or len(spare_cores) < 7:
                    continue
                placement = {**placement, "cpus": sum(spare_cores[:7], [])}
                spare_cores = spare_cores[7:]
                task = waiting.pop(0)
                self.start(task["name"], task["phase"], placement, **task["options"])
                trainer_gpus.add(placement["gpu"]); busy_cpus.update(placement["cpus"])
            heartbeat.update(event="schedule", queued_tasks=[self.task_key(t["phase"], t["name"]) for t in waiting],
                external_gpu_reservations=sorted({r["gpu"] for r in self.reservations}))
            time.sleep(10)
        self.check(heartbeat)

    def evaluation(self, arm, episodes, heartbeat):
        request = f"{arm}_e{episodes:06d}"
        while self.queue.poll(request) is None:
            self.check(heartbeat)
            heartbeat.update(event="await_admission_evaluation", request=request)
            time.sleep(10)
        result = self.queue.poll(request)
        return {"training_episodes": episodes, "summary": result["evaluation"]["summary"], "request_id": request}

    def stop_owned(self):
        for job in self.jobs.values():
            if job["process"].poll() is None:
                os.killpg(job["process"].pid, signal.SIGTERM)
        deadline = time.time()+10
        while any(j["process"].poll() is None for j in self.jobs.values()) and time.time()<deadline:
            time.sleep(.2)
        for job in self.jobs.values():
            if job["process"].poll() is None:
                os.killpg(job["process"].pid, signal.SIGKILL)
            job["process"].wait(timeout=10)

    def run(self):
        phase = "teacher_fit_diagnostic" if self.resume_fit else "diagnostic_admission"
        with ProgressHeartbeat(self.root/"status.json", phase=phase, active_manifest=str(self.manifest_path),
                               recovery_attempt=self.m.get("recovery", {}).get("attempt_id")) as hb:
            try:
                hb.update(workspace_advisory=verify(self.m))
                if list((self.root/"validator/running").glob("*.json")):
                    raise RuntimeError("Interrupted validation requests require explicit reconciliation")
                if self.resume_fit:
                    hb.update(event="reuse_verified_diagnostics", reused_contracts=4, reused_mechanisms=6,
                              recovery_audit=self.m["recovery"]["audit_path"])
                else:
                    tasks = [{"name": e, "phase": "contract", "options": {"encoder": e}} for e in ("E0","E1","E2","E3")]
                    tasks += [{"name": key, "phase": "mechanism", "options": {"mechanism": key}} for key in self.m["mechanisms"]]
                    self.run_tasks(tasks, hb)
                for encoder in ("E0","E1","E2","E3"):
                    if not read_json(self.root/f"contract/{encoder}/result.json")["passed"]:
                        raise RuntimeError(f"Representation contract failed: {encoder}")
                c0 = self.evaluation("C0", 0, hb)
                c0_cases = self.queue.poll("C0_e000000")["evaluation"]["cases"]
                if any(abs(row["makespan"]-self.m["source_costs"][row["case_id"]]) > 1e-6 for row in c0_cases):
                    raise RuntimeError("C0 Tune reference no longer reproduces")
                hb.update(phase="teacher_fit_diagnostic")
                self.run_tasks([{"name": e, "phase": "fit", "options": {"encoder": e}} for e in ("E0","E1","E2","E3")], hb)
                atomic_json(self.root/"training_admission.json", {"passed": True, "eligible_arms": list(ARMS),
                    "single_seed": True, "source": self.m["source"], "fit_is_not_ppo_veto": True,
                    "recipe": "O0 unchanged", "created_unix": time.time()}, overwrite=False)
                hb.update(phase="screen_960")
                self.run_tasks([{"name": f"{arm}_to960", "phase": "train", "options": {"arm": arm, "until": 960}}
                                for arm in ARMS], hb)
                screen = {arm: self.evaluation(arm, 960, hb) for arm in ARMS
                          if read_json(self.root/f"train/{arm}_to960/result.json").get("completed")}
                chosen = screen_candidates(screen)
                atomic_json(self.root/"screen_admission.json", {"results": screen, "selected": chosen,
                    "threshold": .005, "max_extensions": 2, "created_unix": time.time()}, overwrite=False)
                outcomes = {}
                for endpoint in (1440, 1920):
                    if not chosen:
                        break
                    hb.update(phase=f"pilot_to_{endpoint}")
                    previous = endpoint-480
                    self.run_tasks([{"name": f"{arm}_to{endpoint}", "phase": "train", "options": {"arm": arm,
                        "until": endpoint, "resume_checkpoint": str(self.root/f"train/{arm}_to{previous}/models/episodes_{previous:06d}.pt")}}
                        for arm in chosen], hb)
                    survivors = []
                    for arm in chosen:
                        if not read_json(self.root/f"train/{arm}_to{endpoint}/result.json").get("completed"):
                            continue
                        result = self.evaluation(arm, endpoint, hb)
                        outcomes.setdefault(arm, []).append(result)
                        if result["summary"]["gain_fraction"] >= .01 and risk_pass(result["summary"]):
                            survivors.append(arm)
                    atomic_json(self.root/f"pilot_admission_{endpoint}.json", {"results": outcomes,
                        "eligible_next_chunk": survivors if endpoint == 1440 else [],
                        "failed_1440_not_called_completed_1920": True}, overwrite=False)
                    chosen = survivors
                atomic_json(self.root/"result.json", {"completed": True, "screen": screen, "pilot": outcomes,
                    "passed_arms": [a for a,r in outcomes.items() if endpoint_pass(r)],
                    "single_seed_only": True, "finalblind_opened": False, "automatic_extension_4800": False,
                    "evidence_scope": "single-seed screening, not cross-seed robustness"}, overwrite=False)
                atomic_json(self.root/"validator/STOP", {"reason": "all requested validations drained"}, overwrite=False)
                while any(j["phase"] == "validator" and j["process"].poll() is None for j in self.jobs.values()):
                    self.check(hb); time.sleep(2)
            except BaseException:
                hb.update(event="suite_failed", traceback=traceback.format_exc())
                raise
            finally:
                self.stop_owned()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--prior", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    recovery = sub.add_parser("prepare-recovery")
    recovery.add_argument("--manifest", type=Path, required=True)
    recovery.add_argument("--attempt-id", required=True)
    r = sub.add_parser("run"); r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--resume-fit", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args); return
    if args.command == "prepare-recovery":
        from onpolicy.utils.stage3_representation_recovery import prepare_recovery
        prepare_recovery(args.manifest, args.attempt_id, ROOT)
        return
    manifest = read_json(args.manifest)
    lock_path = Path(manifest["root"])/"controller.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume_fit:
            from onpolicy.utils.stage3_representation_recovery import activate_recovery
            # Validate before replacing the root status. Historical statuses,
            # source snapshots and command/log files remain archived intact.
            verify(manifest)
            activate_recovery(manifest, args.manifest)
        elif manifest.get("recovery") or (Path(manifest["root"])/"status.json").exists():
            raise FileExistsError("No implicit suite restart")
        def stop(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}; owned jobs only")
        signal.signal(signal.SIGTERM, stop)
        Suite(manifest, manifest_path=args.manifest, resume_fit=args.resume_fit).run()


if __name__ == "__main__":
    main()
