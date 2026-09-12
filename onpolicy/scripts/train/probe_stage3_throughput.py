#!/usr/bin/env python3
"""Isolated, bounded resource probe. Never modifies or submits to the live study.

Imports the parent's immutable execution snapshot, not workspace training code.
Collects fresh trajectories in width-8 waves under ONE policy, then updates once.
The changed batch/TBPTT recipe is diagnostic only, not an execution-equivalent arm.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import threading
import time
import traceback

GIB = 2**30


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def macroblock_batches(plan, next_group, trajectories=32):
    if trajectories not in (8, 16, 32) or next_group < 0:
        raise ValueError("Probe supports 8/16/32 trajectories and a nonnegative cursor")
    start = ((next_group + 3) // 4) * 4
    batches = plan[start:start + trajectories // 8]
    if len(batches) != trajectories // 8 or len({b["macroblock"] for b in batches}) != 1:
        raise ValueError("No complete aligned probe macroblock remains")
    pairs = [(c["path"], s) for b in batches for c, s in zip(b["cases"], b["seeds"])]
    counts = Counter(c for c, _ in pairs)
    if (any(len(b["cases"]) != 8 or len(b["seeds"]) != 8 for b in batches)
            or len(pairs) != trajectories or len(set(pairs)) != trajectories
            or len(counts) != 4 or set(counts.values()) != {trajectories // 4}):
        raise ValueError("Expected balanced, fresh T1 exposures to four cases")
    return start, batches


def memory_kib(text):
    return {parts[0].rstrip(":"): int(parts[1]) * 1024
            for line in text.splitlines() if len(parts := line.split()) >= 3 and parts[2] == "kB"}


def guard_reason(sample, elapsed, minimum_available, maximum_seconds):
    if sample["host_available_bytes"] < minimum_available:
        return "host MemAvailable below reserved headroom"
    if sample["cgroup_memory_bytes"] >= 23 * GIB:
        return "probe cgroup approaching its 24 GiB hard limit"
    if elapsed >= maximum_seconds:
        return "bounded probe runtime reached"
    return None


def save_archive(path, trajectories, metadata):
    """Stream the retained CPU trajectories, without building a duplicate bytes blob."""
    temporary = path.with_suffix(".tmp.pkl")
    with temporary.open("xb") as handle:
        pickle.dump(trajectories, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    write_json(path.with_suffix(".json"), {**metadata, "sha256": sha256(path)})


def load_archive(path, expected):
    metadata = json.loads(path.with_suffix(".json").read_text())
    if any(metadata.get(k) != value for k, value in expected.items()) or sha256(path) != metadata["sha256"]:
        raise ValueError("Retained rollout archive identity mismatch")
    with path.open("rb") as handle:
        return pickle.load(handle)  # Own hash-verified local artifact only.


def service_cgroup():
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next((line[3:] for line in lines if line.startswith("0::")), None)
    if relative is None:
        raise RuntimeError("Resource probe requires a dedicated cgroup v2 service")
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    limit = (group / "memory.max").read_text().strip()
    if not group.name.startswith("hkbz-s3probe-") or limit == "max" or not 0 < int(limit) <= 24 * GIB:
        raise RuntimeError("Launch only in hkbz-s3probe-* with MemoryMax <= 24 GiB")
    return group


def cgroup_memory(group):
    result = {"cgroup_memory_bytes": int((group / "memory.current").read_text())}
    try:
        result["cgroup_memory_peak_bytes"] = int((group / "memory.peak").read_text())
        result["cgroup_peak_source"] = "kernel memory.peak"
    except FileNotFoundError:
        # Older cgroup-v2 kernels have the hard limit but no high-water counter.
        result["cgroup_peak_source"] = "sampled memory.current; short peaks may be missed"
    return result


class Resources:
    def __init__(self, output, group, heartbeat, gpu, seconds=3300, headroom=24 * GIB):
        self.output, self.group, self.heartbeat, self.gpu = output, group, heartbeat, gpu
        self.seconds, self.headroom = seconds, headroom
        self.begin = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.peaks, self.latest, self.errors = {}, {}, []
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        with self.lock:
            with self.heartbeat.lock:
                phase = self.heartbeat.state.get("phase", "startup")
                event = self.heartbeat.state.get("event")
            result = {"unix": time.time(), "elapsed_seconds": time.monotonic() - self.begin,
                      "phase": phase, "event": event,
                      "host_available_bytes": memory_kib(Path("/proc/meminfo").read_text())["MemAvailable"],
                      **cgroup_memory(self.group),
                      "process_rss_sum_bytes": 0, "process_pss_sum_bytes": 0, "processes": []}
            for pid in (self.group / "cgroup.procs").read_text().split():
                try:
                    values = memory_kib(Path(f"/proc/{pid}/smaps_rollup").read_text())
                    row = {"pid": int(pid), "rss_bytes": values["Rss"], "pss_bytes": values["Pss"]}
                    result["processes"].append(row)
                    result["process_rss_sum_bytes"] += row["rss_bytes"]
                    result["process_pss_sum_bytes"] += row["pss_bytes"]
                except (FileNotFoundError, ProcessLookupError):
                    continue
            torch = sys.modules.get("torch")
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_initialized():
                result.update(torch_allocated_bytes=torch.cuda.memory_allocated(0),
                              torch_reserved_bytes=torch.cuda.memory_reserved(0),
                              torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                              torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            try:
                gpu = subprocess.check_output(["nvidia-smi", "-i", self.gpu,
                    "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                    text=True, timeout=3).strip().split(",")
                result.update(gpu_total_used_bytes=int(gpu[0]) * 2**20,
                              gpu_total_utilization_percent=float(gpu[1]), gpu_total_power_watts=float(gpu[2]))
                processes = subprocess.check_output(["nvidia-smi", "-i", self.gpu,
                    "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                    text=True, timeout=3)
                result["probe_gpu_used_bytes"] = sum(int(row.split(",")[1]) * 2**20
                    for row in processes.splitlines() if row.split(",")[0].strip() == str(os.getpid()))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                result["gpu_sampling_error"] = str(error)
            for key, value in result.items():
                if key.endswith("_bytes") and key != "host_available_bytes":
                    self.peaks[key] = max(self.peaks.get(key, 0), value)
            self.peaks["minimum_host_available_bytes"] = min(
                self.peaks.get("minimum_host_available_bytes", result["host_available_bytes"]),
                result["host_available_bytes"])
            self.latest = result
            with (self.output / "resources.jsonl").open("a") as handle:
                handle.write(json.dumps(result, allow_nan=False) + "\n")
            write_json(self.output / "resources_latest.json", {"latest": result, "peaks": self.peaks})
            reason = guard_reason(result, result["elapsed_seconds"], self.headroom, self.seconds)
            if reason:
                write_json(self.output / "guard_stop.json", {"reason": reason, "sample": result})
                self.stop.set()
                os.kill(os.getpid(), signal.SIGTERM)  # This isolated probe only.
            return result

    def loop(self):
        while not self.stop.wait(5):
            try:
                self.sample()
            except Exception as error:
                self.errors.append(str(error))
                write_json(self.output / "monitor_error.json", {"errors": self.errors})
                self.stop.set()
                os.kill(os.getpid(), signal.SIGTERM)  # Fail closed if monitoring is lost.

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["execution"]["code_root"]).resolve()
    sys.path.insert(0, str(source))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("Pin CUDA_VISIBLE_DEVICES to the requested single GPU UUID")
    group = service_cgroup()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("Probe stopped by guard/service")))

    from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat
    from onpolicy.scripts.train.run_stage3_representation import verify
    from onpolicy.utils.stage3_representation import schedule
    from onpolicy.utils.stage3_hot_update import resume_payload_allowed
    from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
    from onpolicy.scripts.train.stage3_local_worker import frozen_state, assert_frozen
    from onpolicy.scripts.train.stage3_representation_worker import max_role_kl
    import torch

    runner, monitor = None, None
    with ProgressHeartbeat(output / "status.json", phase="verification", diagnostic_only=True) as hb:
        try:
            monitor = Resources(output, group, hb, args.gpu_uuid, args.max_seconds, args.host_headroom_gib * GIB)
            monitor.sample()
            monitor.thread.start()
            advisories = verify(manifest, inputs=True)
            checkpoint_sha = sha256(args.checkpoint)
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            plan = schedule(manifest["splits"]["train_pilot120"], manifest["training"]["seed"], "T1")
            # Check the ORIGINAL continuation contract before making a diagnostic branch.
            parent_protocol = resume_payload_allowed(payload, manifest, "E2_T1", 960, plan)
            start, batches = macroblock_batches(plan, payload["next_group"], args.batch_trajectories)
            performance = dict(manifest["hot_update"]["options"], cache_mib=args.cache_mib)
            configuration = {"diagnostic_only": True, "forbidden_as_rl_initialization": True,
                "representation": "E2", "parent_arm": "E2_T1", "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha, "parent_episodes": payload["training_episodes"],
                "parent_protocol_sha256": parent_protocol, "manifest": str(args.manifest.resolve()),
                "manifest_file_sha256": sha256(args.manifest), "execution_code_sha256": manifest["code"]["sha256"],
                "code_root": str(source), "probe_script_sha256": sha256(__file__),
                "batch_trajectories": args.batch_trajectories, "environment_processes": 8,
                "tbptt_steps": args.tbptt_steps, "ppo_epochs": manifest["training"]["ppo_epochs"],
                "actor_lr": manifest["training"]["actor_lr"], "critic_lr": manifest["training"]["critic_lr"],
                "performance": performance, "gpu_uuid": args.gpu_uuid, "cuda_memory_fraction": args.cuda_memory_fraction,
                "retained_rollout_source": str(args.reuse_archive) if args.reuse_archive else None,
                "dense_case_memory_stress": args.dense_stress,
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "cgroup": str(group),
                "memory_max_bytes": int((group / "memory.max").read_text()), "max_seconds": args.max_seconds,
                "host_headroom_gib": args.host_headroom_gib, "probe_schedule_start_group": start,
                "schedule_batches": batches, "workspace_advisories": advisories,
                "throughput_caveat": "Co-located GPU and CPUs; not a matched speedup comparison",
                "material_passport": {"origin_skill": "academic-research-suite/experiment-agent",
                    "origin_mode": "run", "origin_date": "2026-09-08", "version_label": "resource_probe_v1",
                    "verification_status": "inputs verified; full rollout/update resource peaks pending"}}
            write_json(output / "configuration.json", configuration)
            hb.update(phase="initialization", event="construct_engine")
            runner = RepresentationEngine("E2", source=manifest["source"]["path"], width=8,
                exploration="J", diagnostics=True, performance=performance,
                cuda_memory_fraction=min(args.cuda_memory_fraction, .80))
            # The frozen engine validates <=80% at construction. Explicitly lift
            # the allocator cap after its small model initialization, before any
            # rollout/update. No source snapshot or another process is modified.
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, runner.device)
            runner.resume(args.checkpoint, protocol_sha256=parent_protocol, exploration="J")
            del payload
            frozen = frozen_state(runner)
            policy_updates = runner.policy_updates
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            trajectories, timing, rollout_rows = [], {}, []
            rollout_begin = time.monotonic()
            archive_metadata = {"checkpoint_sha256": checkpoint_sha, "execution_code_sha256": manifest["code"]["sha256"],
                "policy_updates": policy_updates, "schedule_batches": batches, "diagnostic_only": True}
            if args.reuse_archive:
                hb.update(phase="archive_load", event="load_same_policy_trajectories")
                trajectories = load_archive(args.reuse_archive, archive_metadata)
                pairs = [(c["path"], seed) for b in batches for c, seed in zip(b["cases"], b["seeds"])]
                if ([(t["case_id"], t["seed"]) for t in trajectories] != pairs
                        or any(t["policy_updates"] != policy_updates or not t["completed"]
                               or t["forced_replay"] or t["behavior_deterministic"] for t in trajectories)):
                    raise ValueError("Archive is not the original complete on-policy group")
            for index, batch in enumerate([] if args.reuse_archive else batches):
                hb.update(phase="rollout", event="wave_started", wave=index + 1, waves=len(batches),
                          retained_trajectories=len(trajectories))
                begin = time.monotonic()
                trajectories.extend(runner.rollout(batch["cases"], batch["seeds"], retain=True, heartbeat=hb))
                if runner.policy_updates != policy_updates:
                    raise RuntimeError("Policy changed while accumulating fresh trajectories")
                rollout_rows.append({"wave": index + 1, "seconds": time.monotonic() - begin,
                                     "retained_trajectories": len(trajectories), "resources": monitor.sample()})
                write_json(output / "rollout_waves.json", rollout_rows)
            torch.cuda.synchronize()
            timing["rollout_seconds"] = time.monotonic() - rollout_begin
            timing["rollout_reused"] = bool(args.reuse_archive)
            if args.archive_rollouts and not args.reuse_archive:
                hb.update(phase="archive_save", event="retain_completed_rollouts_for_parameter_retest")
                save_archive(output / "trajectories.pkl", trajectories, archive_metadata)
            if args.dense_stress:
                if not args.reuse_archive:
                    raise ValueError("Dense memory stress requires the verified original on-policy archive")
                def graph_bytes(trajectory):
                    return max(sum(value.numel() * value.element_size() for store in state["graph"].stores
                        for value in store.values() if torch.is_tensor(value)) for state in trajectory["states"])
                case = max(trajectories, key=graph_bytes)["case_id"]
                selected = [t for t in trajectories if t["case_id"] == case]
                hb.update(phase="stress_prepare", event="repeat_dense_real_case_for_memory_only")
                trajectories = [copy.deepcopy(selected[i % len(selected)]) for i in range(args.batch_trajectories)]
                del selected
                write_json(output / "dense_stress.json", {"case_id": case,
                    "selection": "largest input graph tensor bytes among four sampled cases; not global worst case",
                    "repeated_trajectories": True, "for_memory_capacity_only": True})
            write_json(output / "trajectory_summary.json", [{"case_id": t["case_id"], "seed": t["seed"],
                "steps": len(t["states"]), "makespan": t["makespan"], "policy_updates": t["policy_updates"],
                "actions_sha256": hashlib.sha256(json.dumps(t["actions"]).encode()).hexdigest()}
                for t in trajectories])
            hb.update(phase="update", event="ppo_started", retained_trajectories=len(trajectories))
            begin = time.monotonic()
            update = runner.update(trajectories, "source", manifest["source_costs"],
                epochs=manifest["training"]["ppo_epochs"], chunk=args.tbptt_steps,
                clip_mode=manifest["training"]["clip_mode"], heartbeat=hb)
            torch.cuda.synchronize()
            timing["update_seconds"] = time.monotonic() - begin
            monitor.sample()
            assert_frozen(runner, frozen)
            if sha256(args.checkpoint) != checkpoint_sha:
                raise RuntimeError("Parent checkpoint changed")
            verify(manifest, inputs=True)
            guard_passed = max_role_kl(update) <= manifest["training"]["post_update_hard_kl"]
            saved = output / "diagnostic_after_update.pt"
            saved_sha = runner.save(saved, diagnostic_only=True, forbidden_as_rl_initialization=True,
                not_resumable_partial_group=True, probe_configuration=configuration,
                parent_checkpoint_sha256=checkpoint_sha, protocol_sha256="diagnostic-resource-probe-v1")
            timing["rollout_and_update_seconds"] = timing["rollout_seconds"] + timing["update_seconds"]
            write_json(output / "result.json", {"completed": True, "diagnostic_only": True,
                "full_rollout_and_update_measured": not args.reuse_archive, "full_update_measured": True,
                "dense_case_memory_stress": args.dense_stress, "formal_study_modified": False,
                "timing": timing, "trajectories_per_hour": (None if args.reuse_archive else
                    len(trajectories) * 3600 / timing["rollout_and_update_seconds"]),
                "resources_peaks": monitor.peaks, "update": update, "post_update_kl_guard_passed": guard_passed,
                "peak_scope": "One real group; not a worst-case bound across all cases or later optimizer steps",
                "checkpoint": str(saved), "checkpoint_sha256": saved_sha,
                "material_passport": {**configuration["material_passport"],
                    "verification_status": (("one complete update measured on a repeated dense-case memory stress batch"
                        if args.dense_stress else "one complete update measured on hash-verified archived on-policy trajectories")
                        if args.reuse_archive else "one full fresh rollout/update measured")
                        + "; frozen source/checkpoint reverified; no RL-effect conclusion"}})
            hb.update(phase="finished", event="probe_completed")
        except BaseException as error:
            write_json(output / "failure.json", {"completed": False, "error": str(error),
                "traceback": traceback.format_exc(), "resources_peaks": monitor.peaks if monitor else {},
                "execution_cache": getattr(runner.execution_cache, "last_report", None) if runner else None,
                "torch_peak_allocated_bytes_at_failure": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                "torch_peak_reserved_bytes_at_failure": torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None})
            raise
        finally:
            if monitor is not None:
                monitor.close()
            if runner is not None:
                runner.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--gpu-uuid", required=True)
    result.add_argument("--batch-trajectories", type=int, choices=(8, 16, 32), default=32)
    result.add_argument("--tbptt-steps", type=int, choices=(8, 16, 32), default=16)
    result.add_argument("--cache-mib", type=int, choices=(1024, 2048, 4096), default=4096)
    result.add_argument("--cuda-memory-fraction", type=float, choices=(.75, 1.0), default=.75)
    result.add_argument("--archive-rollouts", action="store_true")
    result.add_argument("--reuse-archive", type=Path)
    result.add_argument("--dense-stress", action="store_true")
    result.add_argument("--host-headroom-gib", type=int, choices=range(24, 65), default=24)
    result.add_argument("--max-seconds", type=int, choices=range(60, 3301), default=3300)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
