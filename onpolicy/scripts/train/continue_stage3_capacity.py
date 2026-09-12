#!/usr/bin/env python3
"""Auditable B32 continuation from certified live-state boundaries.

Frozen model/environment code is reused unchanged. New sampling/update recipe,
artifact namespace and checkpoint ancestry are explicit. All eight arms share
one validator queue, with the original screening and extension thresholds.
"""
import argparse
import copy
import ctypes
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def bootstrap(path):
    manifest = json.loads(Path(path).read_text())
    parent_path = manifest.get("capacity_parent_manifest", str(path))
    parent = json.loads(Path(parent_path).read_text())
    os.environ.setdefault("HKBZ_STAGE3_WORKSPACE_ROOT", parent["execution"]["workspace_root"])
    sys.path.insert(0, parent["execution"]["code_root"])
    return manifest, parent


def verify_capacity(manifest, parent, inputs=True):
    from onpolicy.scripts.train.run_stage3_representation import verify
    from onpolicy.utils.stage3_research import digest_file, digest_json, read_json
    if digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Capacity manifest identity changed")
    verify(parent, inputs=inputs)
    if digest_file(manifest["capacity_parent_manifest"]) != manifest["capacity_parent_file_sha256"]:
        raise ValueError("Parent protocol file changed")
    for path, checksum in manifest["capacity_code_files"].items():
        if digest_file(path) != checksum:
            raise ValueError("Capacity driver source changed")
    if inputs:
        for path, checksum in manifest["capacity_inputs"].items():
            if digest_file(path) != checksum:
                raise ValueError(f"Capacity evidence changed: {path}")
        for arm, row in manifest["initial_checkpoints"].items():
            proof = read_json(row["proof"])
            if (not proof["verified"] or proof["kind"] != "actual_live_completed_group_state"
                    or proof["arm"] != arm or proof["training_episodes"] != row["training_episodes"]):
                raise ValueError("Initial state lacks a certified real training boundary")


def prepare(args, parent):
    from onpolicy.utils.stage3_research import read_json, atomic_json, digest_file, digest_json
    from onpolicy.utils.stage3_representation import ARMS
    from stage3_capacity_protocol import capacity_gate
    normal_path, dense_path = args.normal.resolve(), args.dense.resolve()
    normal, dense = read_json(normal_path), read_json(dense_path)
    config_path, dense_config_path = normal_path.parent / "configuration.json", dense_path.parent / "configuration.json"
    config, dense_config = read_json(config_path), read_json(dense_config_path)
    if not capacity_gate(normal, dense, config, dense_config):
        raise ValueError("Normal + dense-case complete-update capacity/KL/headroom gate did not pass")
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError("Capacity output already exists")
    manifest = copy.deepcopy(parent)
    for key in ("hot_update", "recovery", "manifest_sha256"):
        manifest.pop(key, None)
    initial, inputs = {}, {str(p): digest_file(p) for p in (normal_path, dense_path, config_path, dense_config_path)}
    inputs[str(args.pause_receipt.resolve())] = digest_file(args.pause_receipt)
    for arm in ARMS:
        folder = args.boundaries.resolve() / arm
        row = read_json(folder / "result.json")
        proof_path = folder / "boundary_proof.json"
        if not row["completed"] or row["committed_groups_discarded"]:
            raise ValueError("Every arm needs a zero-rewind real-state boundary")
        initial[arm] = {**row, "proof": str(proof_path)}
        for path in (Path(row["checkpoint"]), folder / "result.json", proof_path):
            inputs[str(path)] = digest_file(path)
    driver = Path(__file__).resolve()
    protocol = driver.parent / "stage3_capacity_protocol.py"
    files = {str(p): digest_file(p) for p in (driver, protocol)}
    manifest.update(root=str(root), schema="stage3-capacity-continuation-v1", created_unix=time.time(),
        capacity_parent_manifest=str(args.manifest.resolve()), capacity_parent_file_sha256=digest_file(args.manifest),
        capacity_code_files=files, capacity_inputs=inputs, initial_checkpoints=initial,
        capacity_driver=str(driver), pause_receipt=str(args.pause_receipt.resolve()),
        capacity_recipe={"batch_size": 32, "tbptt_steps": config["tbptt_steps"],
            "performance": config["performance"], "cuda_memory_fraction": 1.0,
            "rollout_width": 8, "save_every_complete_group": True,
            "T0": "one case per update", "T1": "four cases per update",
            "exposure": "matched 4-case macroblocks; smaller balanced boundary batches land exactly on evaluation milestones",
            "normal_evidence": str(normal_path), "dense_evidence": str(dense_path),
            "interpretation": "Adaptive parameter continuation with recorded per-arm switch points; not a fresh fixed-batch factorial study"},
        code={"sha256": digest_json({"base": parent["code"]["sha256"], "capacity_files": files})},
        material_passport={"origin_skill": "academic-research-suite/experiment-agent", "origin_mode": "run",
            "origin_date": "2026-09-08", "version_label": "capacity_continuation_v1",
            "verification_status": "capacity and live boundary checks passed; RL outcome unverified"})
    manifest["training"].update(tbptt_steps=config["tbptt_steps"])
    manifest["resources"].update(trainer_memory_fraction=1.0, memory_high_gib=96, memory_max_gib=108)
    manifest["manifest_sha256"] = digest_json(manifest)
    verify_capacity(manifest, parent)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "manifest.json", manifest, overwrite=False)
    print(root / "manifest.json", flush=True)


def engine(manifest, parent, variant, width, fraction):
    import torch
    from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
    runner = RepresentationEngine(variant, source=parent["source"]["path"], width=width,
        exploration="J", diagnostics=True, performance=manifest["capacity_recipe"]["performance"],
        cuda_memory_fraction=min(fraction, .80))
    torch.cuda.set_per_process_memory_fraction(fraction, runner.device)
    return runner


def checkpoint_metadata(manifest, arm, **extra):
    return {"protocol_sha256": manifest["manifest_sha256"], "source_sha256": manifest["source"]["sha256"],
        "execution_code_sha256": manifest["code"]["sha256"], "arm": arm,
        "seed": manifest["training"]["seed"], "elite_buffer": {}, "elite_usage": {}, "auxiliary_steps": 0,
        "capacity_recipe": manifest["capacity_recipe"], "diagnostic_only": False, **extra}


def host_usage():
    """Read-only whole-service RAM accounting, including all environment children."""
    relative = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    memory = {parts[0].rstrip(":"): int(parts[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if len(parts := line.split()) >= 3 and parts[2] == "kB"}
    return {"unix": time.time(), "cgroup_memory_bytes": int((group / "memory.current").read_text()),
        "host_available_bytes": memory["MemAvailable"], "scope": "whole continuation service, sampled",
        "memory_events": (group / "memory.events").read_text()}


def submit(manifest, checkpoint, arm, episodes):
    from onpolicy.utils.stage3_representation import RepresentationQueue, ARMS
    from onpolicy.utils.stage3_research import digest_file, digest_json
    from onpolicy.utils.stage3_local_exploration import EVAL_PROTOCOL, MODES
    cases = manifest["splits"]["tune"]
    return RepresentationQueue(Path(manifest["root"]) / "validator").submit({
        "request_id": f"{arm}_e{episodes:06d}", "checkpoint": str(checkpoint), "checkpoint_sha256": digest_file(checkpoint),
        "cases": cases, "cases_sha256": digest_json(cases), "contract_sha256": manifest["contract_sha256"],
        "code_sha256": manifest["code"]["sha256"], "protocol_sha256": manifest["manifest_sha256"],
        "tau": .3, "seed": 42, "evaluation_protocol": EVAL_PROTOCOL, "training_exploration": MODES["J"],
        "representation": ARMS[arm]["encoder"], "arm": arm, "training_episodes": episodes})


def train(args, manifest, parent, hb):
    import torch
    from onpolicy.utils.stage3_representation import ARMS, RepresentationQueue
    from onpolicy.utils.stage3_research import read_json, atomic_json, digest_file, digest_json
    from onpolicy.scripts.train.run_stage3_sampling_audit import save_group
    from onpolicy.scripts.train.stage3_local_worker import frozen_state, assert_frozen
    from onpolicy.scripts.train.stage3_representation_worker import max_role_kl
    from stage3_capacity_protocol import phase_schedule
    arm, until, output = args.arm, args.until, args.output
    if until not in (960, 1440, 1920):
        raise ValueError("Only existing stage boundaries are allowed")
    if until > 960:
        path, key = (("screen_admission.json", "selected") if until == 1440
                     else ("pilot_admission_1440.json", "eligible_next_chunk"))
        if arm not in read_json(Path(manifest["root"]) / path)[key]:
            raise ValueError("Arm has not passed the extension gate")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("diagnostic_only") or payload.get("not_resumable_partial_group")
            or payload.get("forbidden_as_rl_initialization") or payload.get("arm") != arm
            or payload.get("source_sha256") != manifest["source"]["sha256"]):
        raise ValueError("Invalid training state")
    new_protocol = payload["protocol_sha256"] == manifest["manifest_sha256"]
    if not new_protocol and (str(args.checkpoint.resolve()) != manifest["initial_checkpoints"][arm]["checkpoint"]
            or digest_file(args.checkpoint) != manifest["initial_checkpoints"][arm]["checkpoint_sha256"]
            or not payload.get("safe_boundary_proof")):
        raise ValueError("Only the certified initial boundary may cross recipe versions")
    continuing = new_protocol and payload.get("capacity_stage_until") == until
    start = payload["capacity_stage_start"] if continuing else payload["training_episodes"]
    plan = phase_schedule(manifest["splits"]["train_pilot120"], manifest["training"]["seed"], ARMS[arm]["batch"], start, until)
    cursor = payload["capacity_next_group"] if continuing else 0
    episodes = payload["training_episodes"]
    if continuing and (payload["capacity_schedule_sha256"] != digest_json(plan)
                       or episodes != (plan[cursor - 1]["training_episodes"] if cursor else start)):
        raise ValueError("Capacity resume cursor changed")
    runner = engine(manifest, parent, ARMS[arm]["encoder"], width=8, fraction=1.0)
    queue = RepresentationQueue(Path(manifest["root"]) / "validator")
    requests = payload.get("capacity_validation_requests", []) if new_protocol else []
    guard = None
    try:
        runner.resume(args.checkpoint, protocol_sha256=payload["protocol_sha256"], exploration="J")
        del payload
        frozen = frozen_state(runner)
        atomic_json(output / "resume_receipt.json", {"checkpoint": str(args.checkpoint),
            "checkpoint_sha256": digest_file(args.checkpoint), "training_episodes": episodes,
            "model_optimizer_normalizers_rng_restored": True, "new_capacity_recipe": manifest["capacity_recipe"]}, overwrite=False)
        if new_protocol and episodes % 480 == 0:
            request = f"{arm}_e{episodes:06d}.json"
            if not any((queue.root / state / request).exists() for state in ("pending", "running", "results", "failed")):
                submit(manifest, args.checkpoint, arm, episodes)
        for index in range(cursor, len(plan)):
            while queue.pending_count(arm) >= 2 or queue.pending_count() >= 16:
                hb.update(event="validator_backpressure", pending=queue.pending_count())
                time.sleep(10)
            batch, group = plan[index], []
            begin = time.time()
            policy_updates = runner.policy_updates
            torch.cuda.reset_peak_memory_stats()
            for first in range(0, len(batch["cases"]), 8):
                hb.update(event="collect_wave", capacity_group=index + 1, retained_trajectories=len(group),
                          training_episodes=episodes, target_episodes=until)
                group.extend(runner.rollout(batch["cases"][first:first + 8], batch["seeds"][first:first + 8], retain=True, heartbeat=hb))
                if runner.policy_updates != policy_updates:
                    raise ValueError("Policy changed during batch accumulation")
            save_group(output / f"rollouts/group_{index + 1:04d}.json", group,
                {"arm": arm, "capacity_group": index + 1, "macroblock": batch["macroblock"], "policy_updates": policy_updates})
            update = runner.update(group, "source", manifest["source_costs"], epochs=manifest["training"]["ppo_epochs"],
                chunk=manifest["capacity_recipe"]["tbptt_steps"], clip_mode=manifest["training"]["clip_mode"], heartbeat=hb)
            assert_frozen(runner, frozen)
            if max_role_kl(update) > manifest["training"]["post_update_hard_kl"]:
                guard = {"reason": "post_update_role_kl", "attempted_episodes": batch["training_episodes"], "update": update}
                atomic_json(output / "guard_stop.json", guard, overwrite=False)
                runner.save(output / "guard_stop_state.pt", **checkpoint_metadata(manifest, arm,
                    diagnostic_only=True, not_resumable_partial_group=True, next_group=index))
                break
            episodes = batch["training_episodes"]
            atomic_json(output / f"updates/group_{index + 1:04d}.json", {"arm": arm, "capacity_group": index + 1,
                "training_episodes": episodes, "batch_trajectories": len(group), "seconds": time.time() - begin,
                "update": update, "macroblock": batch["macroblock"]}, overwrite=False)
            due = episodes % 480 == 0
            next_requests = requests + ([f"{arm}_e{episodes:06d}"] if due else [])
            checkpoint = output / f"models/episodes_{episodes:06d}.pt"
            runner.save(checkpoint, **checkpoint_metadata(manifest, arm, training_episodes=episodes,
                next_group=index + 1, capacity_stage_start=start, capacity_stage_until=until,
                capacity_next_group=index + 1, capacity_schedule_sha256=digest_json(plan),
                capacity_validation_requests=next_requests, batch_composition=ARMS[arm]["batch"], optimizer_recipe="O0"))
            if due:
                requests.append(submit(manifest, checkpoint, arm, episodes))
            hb.update(event="group_completed", training_episodes=episodes, capacity_group=index + 1)
            del group
            gc.collect()
            try:
                ctypes.CDLL(None).malloc_trim(0)
            except AttributeError:
                pass
        atomic_json(output / "result.json", {"completed": guard is None and episodes == until, "guard": guard,
            "arm": arm, "training_episodes": episodes, "target_episodes": until, "requests": requests}, overwrite=False)
    finally:
        runner.close()


def validator(args, manifest, parent, hb):
    import torch
    from onpolicy.utils.stage3_representation import RepresentationQueue, ARMS
    from onpolicy.utils.stage3_research import digest_file, digest_json, read_json, atomic_json, paired_summary
    from onpolicy.utils.stage3_local_exploration import EVAL_PROTOCOL, MODES
    from onpolicy.scripts.train.stage3_local_worker import evaluate
    queue = RepresentationQueue(Path(manifest["root"]) / "validator")
    runner, current = None, None
    try:
        while True:
            claim = queue.claim()
            if claim is None:
                if (queue.root / "STOP").exists():
                    return
                time.sleep(2)
                continue
            path, request = claim
            try:
                verify_capacity(manifest, parent, inputs=False)
                if (request["cases"] != manifest["splits"]["tune"] or request["evaluation_protocol"] != EVAL_PROTOCOL
                        or request["tau"] != .3 or request["seed"] != 42 or request["training_exploration"] != MODES["J"]
                        or request["protocol_sha256"] != manifest["manifest_sha256"]
                        or request["code_sha256"] != manifest["code"]["sha256"]
                        or request["contract_sha256"] != manifest["contract_sha256"]
                        or request["cache_key"] != queue.identity(request)
                        or request["cases_sha256"] != digest_json(request["cases"])
                        or request["checkpoint_sha256"] != digest_file(request["checkpoint"])):
                    raise ValueError("Capacity validation request identity mismatch")
                payload = torch.load(request["checkpoint"], map_location="cpu", weights_only=False)
                arm = request["arm"]
                if (arm not in ARMS or payload.get("diagnostic_only") or payload.get("not_resumable_partial_group")
                        or payload.get("forbidden_as_rl_initialization")
                        or request.get("representation") != ARMS[arm]["encoder"]
                        or payload.get("arm") != arm or payload.get("representation") != ARMS[arm]["encoder"]
                        or payload.get("protocol_sha256") != manifest["manifest_sha256"]
                        or payload.get("source_sha256") != manifest["source"]["sha256"]
                        or payload.get("execution_code_sha256") != manifest["code"]["sha256"]
                        or payload.get("training_episodes") != request["training_episodes"]
                        or payload.get("exploration") != MODES["J"]
                        or request["request_id"] != f"{arm}_e{request['training_episodes']:06d}"):
                    raise ValueError("Checkpoint is not an eligible capacity training result")
                del payload
                hb.update(event="evaluate", request_id=request["request_id"], request_started_unix=time.time())
                cache = queue.root / "cache" / f"{request['cache_key']}.json"
                if cache.exists():
                    cached = read_json(cache)
                    if cached["cache_key"] != request["cache_key"]:
                        raise ValueError("Validation cache identity changed")
                    queue.finish(path, cached["evaluation"])
                    hb.update(event="idle", request_id=None, request_started_unix=None)
                    continue
                if current != request["representation"]:
                    if runner is not None:
                        runner.close()
                        del runner
                        gc.collect()
                        torch.cuda.empty_cache()
                    runner = engine(manifest, parent, request["representation"], width=3, fraction=.15)
                    current = request["representation"]
                runner.load(request["checkpoint"])
                runner.configure_exploration(None)
                runner.policy.ac.tau = .3
                rows = evaluate(runner, request["cases"], hb)
                result = {"cases": rows, "summary": paired_summary(rows, manifest["source_costs"]), "evaluation_protocol": EVAL_PROTOCOL}
                atomic_json(cache,
                    {"cache_key": request["cache_key"], "evaluation": result}, overwrite=False)
                queue.finish(path, result)
                hb.update(event="idle", request_id=None, request_started_unix=None)
            except BaseException:
                queue.finish(path, None, traceback.format_exc())
                raise
    finally:
        if runner is not None:
            runner.close()


def run_suite(args, manifest, parent):
    from onpolicy.scripts.train.run_stage3_representation import Suite
    from onpolicy.scripts.train.hot_update_stage3_representation import RollingSuite, ProcessIdentity
    from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat
    from onpolicy.utils.stage3_research import atomic_json, read_json
    from onpolicy.utils.stage3_hot_update import numa_plan
    from onpolicy.utils.stage3_representation import ARMS, cpu_text, external_resources

    class CapacitySuite(RollingSuite):
        def __init__(self):
            Suite.__init__(self, manifest, manifest_path=args.manifest.resolve())
            self.parent = parent
            self.upgrade = {"attempt_id": "capacity_20260908_r1"}
            self.plan = numa_plan(manifest["resources"]["core_groups"], manifest["resources"]["gpus"])
            for placement in self.plan["trainers"]:
                placement["cuda_memory_fraction"] = 1.0
            self.outputs = {}
            self.hb = None
            self.memory_peak = 0

        def start(self, name, phase, placement, **options):
            key = self.task_key(phase, name)
            output = self.root / ("validator/workers" if phase == "validator" else phase) / name
            if key in self.jobs or output.exists():
                raise FileExistsError("No implicit capacity worker overwrite/retry")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.m["resources"]["gpus"][placement["gpu"]]["uuid"])
            command = ["taskset", "--cpu-list", cpu_text(placement["cpus"]), sys.executable, "-B", "-u", __file__, "worker",
                "--manifest", str(self.manifest_path), "--phase", phase, "--output", str(output)]
            for option, value in options.items():
                command.extend(["--checkpoint" if option == "resume_checkpoint" else "--" + option.replace("_", "-"), str(value)])
            log = self.root / "logs" / phase / (name + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(self.root / "commands" / phase / (name + ".json"),
                        {"command": command, "placement": placement}, overwrite=False)
            with log.open("x") as handle:
                process = subprocess.Popen(command, env=env, cwd=parent["execution"]["code_root"],
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            self.jobs[key] = {"process": process, "phase": phase, "name": name, "placement": placement,
                "output": output, "started_unix": time.time(), "options": options}

        def resources(self):
            external = external_resources(self.m["resources"]["gpus"], os.getpid())
            if external:
                # A new external job must not kill all eight arms as happened
                # in the previous controller. Reversibly pause mains and wait.
                paused = []
                try:
                    for job in self.jobs.values():
                        if job["process"].poll() is None:
                            token = ProcessIdentity(job["process"].pid)
                            token.pause()
                            paused.append(token)
                    while external:
                        if self.hb:
                            self.hb.update(status="paused", event="external_gpu_conflict", external_reservations=external)
                        atomic_json(self.root / "external_resource_pause.json", {"created_unix": time.time(),
                            "external": external, "workers": [{"pid": p.pid, "start_ticks": p.ticks} for p in paused]})
                        time.sleep(10)
                        external = external_resources(self.m["resources"]["gpus"], os.getpid())
                finally:
                    for token in paused:
                        token.send(signal.SIGCONT)
                    if self.hb:
                        self.hb.update(status="running", event="external_gpu_conflict_cleared")
            atomic_json(self.root / "resource_leases.json", {"updated_unix": time.time(), "plan": self.plan})
            usage = host_usage()
            self.memory_peak = max(self.memory_peak, usage["cgroup_memory_bytes"])
            usage["sampled_cgroup_peak_bytes"] = self.memory_peak
            atomic_json(self.root / "resources_latest.json", usage)
            with (self.root / "resources.jsonl").open("a") as handle:
                handle.write(json.dumps(usage) + "\n")
            return self.plan

        def retire_original(self):
            receipt = read_json(self.m["pause_receipt"])
            # All actual trainer states have already been captured, certified
            # and retired individually. Only old validators/controller remain.
            for record in receipt["workers"]:
                try:
                    token = ProcessIdentity(record["pid"])
                except FileNotFoundError:
                    continue
                if token.ticks != record["start_ticks"]:
                    raise ValueError("Old PID was reused; do not signal")
                if not token.alive():
                    continue
                if record["task"].startswith("train/"):
                    raise ValueError("A source trainer still lives; capture its true boundary first")
                token.send(signal.SIGTERM)
                token.send(signal.SIGCONT)
            record = receipt["controller"]
            token = ProcessIdentity(record["pid"], "resume_stage3_after_probe.py")
            if token.ticks != record["start_ticks"]:
                raise ValueError("Old controller identity changed")
            token.send(signal.SIGTERM)
            token.send(signal.SIGCONT)
            deadline = time.time() + 60
            while token.alive():
                if time.time() > deadline:
                    raise TimeoutError("Original controller cleanup did not finish")
                time.sleep(.2)
            atomic_json(self.root / "handoff_receipt.json", {"old_controller": record,
                "actual_boundary_checkpoints": self.m["initial_checkpoints"], "committed_groups_discarded": 0,
                "old_artifacts_preserved": True, "created_unix": time.time()}, overwrite=False)

        def run(self):
            verify_capacity(self.m, self.parent)
            self.retire_original()
            old_root = Path(self.parent["root"])
            atomic_json(old_root / "status.json", {"status": "continued", "phase": "capacity_continuation",
                "active_manifest": str(self.manifest_path), "active_root": str(self.root), "pid": os.getpid(),
                "capacity_recipe": self.m["capacity_recipe"], "handoff_receipt": str(self.root / "handoff_receipt.json")})
            atomic_json(old_root / "active_execution.json", {"manifest": str(self.manifest_path),
                "active_root": str(self.root), "code_root": self.parent["execution"]["code_root"],
                "capacity_driver": str(Path(__file__).resolve()), "created_unix": time.time()})
            with ProgressHeartbeat(self.root / "status.json", phase="screen_960", active_manifest=str(self.manifest_path),
                                   capacity_recipe=self.m["capacity_recipe"]) as hb:
                self.hb = hb
                self.resources()
                os.sched_setaffinity(0, self.plan["controller"])
                for index, placement in enumerate(self.plan["validators"]):
                    self.start(f"pool{index}", "validator", placement)
                for index, arm in enumerate(ARMS):
                    name = f"{arm}_to960"
                    self.start(name, "train", self.plan["trainers"][index], arm=arm, until=960,
                               resume_checkpoint=self.m["initial_checkpoints"][arm]["checkpoint"])
                    self.outputs[arm] = self.root / "train" / name
                self.finish_study(hb)

    suite = CapacitySuite()
    with (suite.root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            suite.run()
        finally:
            suite.stop_owned()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--normal", type=Path, required=True)
    p.add_argument("--dense", type=Path, required=True)
    p.add_argument("--boundaries", type=Path, required=True)
    p.add_argument("--pause-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--phase", choices=("train", "validator"), required=True)
    worker.add_argument("--arm")
    worker.add_argument("--until", type=int)
    worker.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    manifest, parent = bootstrap(args.manifest)
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Capacity stop signal {signum}; completed checkpoints retained")
    signal.signal(signal.SIGTERM, stop)
    if args.command == "prepare":
        prepare(args, parent)
    elif args.command == "run":
        run_suite(args, manifest, parent)
    else:
        from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat
        verify_capacity(manifest, parent)
        args.output.mkdir(parents=True, exist_ok=False)
        with ProgressHeartbeat(args.output / "status.json", phase=args.phase, arm=args.arm) as hb:
            (train if args.phase == "train" else validator)(args, manifest, parent, hb)


if __name__ == "__main__":
    main()
