#!/usr/bin/env python3
"""Prepare/run the bounded, inference-only sampling follow-up on isolated code."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.utils.stage3_research import (BASE_MANIFEST, Heartbeat, atomic_json,
    best_of_n, digest_file, digest_json, read_json, stratified_cases, trajectory_seed,
    verify_cases)
from onpolicy.utils.stage3_sampling_audit import ARMS, action_digest

NEW_FILES = (
    "onpolicy/utils/stage3_sampling_audit.py",
    "onpolicy/scripts/train/run_stage3_sampling_audit.py",
    "onpolicy/scripts/train/launch_stage3_sampling_audit_half.sh",
    "onpolicy/envs/HKBZ/test/test_stage3_sampling_audit.py",
    "onpolicy/scripts/train/check_stage3_sampling_api_precision.py",
    "STAGE3_SAMPLING_AUDIT_20260906.md",
)


class ProgressHeartbeat(Heartbeat):
    def update(self, **values):
        if values:
            values["last_progress_unix"] = time.time()
        super().update(**values)


def manifest_identity(manifest):
    return digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def prepare(args):
    prior_path = args.prior.resolve()
    prior = read_json(prior_path)
    old_root = Path(prior["root"])
    old_code = Path(prior["execution"]["code_root"])
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("New audit directory required; old results are never overwritten")
    precision = read_json(args.precision_evidence)
    if (not precision.get("bounded_probe_completed") or precision["action_mismatches"]
            or precision["rng_mismatches"]
            or not all(r["actions_all_equal"] for r in precision["same_state_repeats"])):
        raise ValueError("Numerical tolerance requires clean same-state repeatability evidence")
    if read_json(old_root / "status.json")["status"] != "completed":
        raise ValueError("Prior diagnostic suite is not completed")
    if read_json(old_root / "diagnostic_gate.json")["passed"]:
        raise ValueError("This protocol is scoped to the failed sampling diagnostic")
    if digest_file(prior["source"]["path"]) != prior["source"]["sha256"]:
        raise ValueError("C0 checkpoint changed")
    cases = stratified_cases(prior["splits"]["train_diag32"],
        {"iid": 4, "ood_stress": 3, "ood_scale": 1}, 2026090601)
    verify_cases(cases, training=True)
    files = {}
    # Copy the previous execution's verified code, NOT today's shared worktree.
    for relative, expected in prior["code"]["files"].items():
        source = old_code / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe snapshot entry")
        if digest_file(source) != expected:
            raise ValueError(f"Previous source snapshot changed: {relative}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest_file(target) != expected:
            raise ValueError("Snapshot copy mismatch")
        files[relative] = expected
    for relative in NEW_FILES:
        if relative in files:
            raise ValueError(f"Follow-up may not overwrite an old source file: {relative}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        sha = digest_file(ROOT / relative)
        shutil.copy2(ROOT / relative, target)
        if digest_file(target) != sha:
            raise ValueError("Audit file changed while copying")
        files[relative] = sha
    archives = {case["name"]: {"path": str(old_root / "sample/cases" / f"{case['name']}.json"),
        "sha256": digest_file(old_root / "sample/cases" / f"{case['name']}.json")} for case in cases}
    manifest = {
        "schema": "stage3-sampling-audit-v1", "created_unix": time.time(), "root": str(output),
        "material_passport": {"origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run", "origin_date": "2026-09-06", "verification_status": "prepared",
            "version_label": "sampling_audit_v1", "data_classification": "internal; no upload"},
        "prior_manifest": {"path": str(prior_path), "sha256": digest_file(prior_path)},
        "source": prior["source"], "contract": prior["contract"],
        "execution": {"code_root": str(output / "source"), "workspace_root": str(ROOT),
            "workspace_code_policy": "advisory; shared edits do not abort this run",
            "snapshot_base_sha256": prior["code"]["sha256"],
            "python": sys.executable, "python_version": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in
                ("torch", "torch-geometric", "numpy", "scipy", "gymnasium", "PyYAML")}},
        "code": {"files": files, "sha256": digest_json(files)},
        "input_files": {str(BASE_MANIFEST): digest_file(BASE_MANIFEST)},
        "cases": cases, "case_selection_seed": 2026090601,
        "source_costs": {case["path"]: prior["source_costs"][case["path"]] for case in cases},
        "archives": archives, "arms": ARMS, "replicas": 32, "seed_base": 20260909,
        "resources": {"physical_gpus": [0, 1, 2, 3], "parent_cpuset": "0-31,64-95",
            "lanes": ["0-7,64-71", "8-15,72-79", "16-23,80-87", "24-31,88-95"],
            "physical_cpu_cores": 32, "logical_cpu_count": 64, "workers_per_lane": 8,
            "torch_threads": 1, "memory_max_gib": 40},
        "timeouts_seconds": {"preflight": 7200, "sampling": 86400,
            "suite": 93600, "rollout_steps": 4000, "progress_warning": 600},
        "preflight": {"greedy_cases": 8, "stochastic_archive_cases": 8,
            "cost_abs_tolerance": 1e-6, "hidden_abs_tolerance": 1e-5,
            "archive_actions": "exact SHA256 of int32 actions including shape validation",
            "api_check": "every visited state: identical inputs, act/get_actions, same RNG",
            "gate": "all complete, exact greedy cost/archive actions, API/RNG match, no weight change"},
        "numerical_tolerance_evidence": {"path": str(args.precision_evidence.resolve()),
            "sha256": digest_file(args.precision_evidence),
            "reason": "Same act API repeats vary by 2.58e-6 in FP32; use 1e-5 hidden-only tolerance. "
                      "No relaxation of action equality, RNG equality, completion or cost reproduction."},
        "reporting": {"prefix_best": [1, 4, 8, 16, 32], "primary": "all-case mean and best-of-32",
            "incumbent": "C0 retained separately; never confuse with a learned improvement",
            "inference": "8-case diagnostic only; no confirmatory model/architecture claim",
            "seeds": "paired seed labels; role ablations consume different RNG draws",
            "historical_scope": "same C0 and contract, not unidentified historical checkpoint/run",
            "completion": "incomplete/cyclic trajectories fail task, never rank as cheap solutions"},
        "automatic_rl": False, "automatic_finalblind": False,
    }
    manifest["manifest_sha256"] = manifest_identity(manifest)
    atomic_json(output / "manifest.json", manifest, overwrite=False)
    print(json.dumps({"manifest": str(output / "manifest.json"),
        "cases": [c["name"] for c in cases], "arms": list(ARMS),
        "fresh_sampling_trajectories": len(cases) * 32 * len(ARMS)}, ensure_ascii=False), flush=True)


def verify_manifest(manifest, *, archives=False):
    if manifest_identity(manifest) != manifest["manifest_sha256"]:
        raise ValueError("Frozen audit manifest changed")
    if ROOT != Path(manifest["execution"]["code_root"]):
        raise ValueError("Execution must import only the isolated source snapshot")
    for relative, sha in manifest["code"]["files"].items():
        if digest_file(ROOT / relative) != sha:
            raise ValueError(f"Isolated execution file changed: {relative}")
    if digest_file(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("C0 checkpoint changed")
    for path, sha in manifest["input_files"].items():
        if digest_file(path) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    verify_cases(manifest["cases"], training=True)
    if archives:
        for record in manifest["archives"].values():
            if digest_file(record["path"]) != record["sha256"]:
                raise ValueError("Prior sampling archive changed")


def model_digest(engine):
    digest = hashlib.sha256()
    for name, tensor in sorted(engine.policy.ac.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_engine(manifest):
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    from onpolicy.utils.stage3_sampling_audit import history_inputs

    class AuditEngine(ResearchEngine):
        history_mode = "authoritative"

        def _inputs(self, observations, hidden, infos, previous):
            graph, h, active, op, site, roles = super()._inputs(observations, hidden, infos, previous)
            op, site = history_inputs(previous, active, op, site, self.history_mode)
            return graph, h, active, op, site, roles

    engine = AuditEngine(manifest["source"]["path"], width=manifest["resources"]["workers_per_lane"])
    for parameter in engine.policy.ac.parameters():
        parameter.requires_grad_(False)
    return engine


def trajectory_record(trajectory):
    return {**{k: v for k, v in trajectory.items() if k not in ("actions", "states")},
            "actions_shape": list(__import__("numpy").asarray(trajectory["actions"]).shape),
            "action_sha256": action_digest(trajectory["actions"])}


def save_group(path, trajectories, metadata):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = path.with_suffix(".npz")
    if path.exists() or trace.exists():
        raise FileExistsError("Committed group/trace must not be overwritten")
    temporary = trace.with_name(trace.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **{f"trajectory_{i}_seed_{t['seed']}": np.asarray(t["actions"], dtype=np.int32)
                                      for i, t in enumerate(trajectories)})
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, trace)
    temporary.unlink()
    atomic_json(path, {**metadata, "trajectories": [trajectory_record(t) for t in trajectories],
        "trace": {"path": str(trace), "sha256": digest_file(trace)}}, overwrite=False)


def run_preflight(manifest, lane, engine, heartbeat):
    from onpolicy.utils.stage3_sampling_audit import sampling_heads, compare_actor_api
    cases = manifest["cases"][lane::4]
    output = Path(manifest["root"]) / "preflight" / f"lane{lane}"
    metrics = {}
    original_weights = model_digest(engine)
    with sampling_heads(engine.policy.ac, manifest["arms"]["full_t030"]), \
            compare_actor_api(engine.policy, engine.device, metrics,
                hidden_abs_tolerance=manifest["preflight"]["hidden_abs_tolerance"]):
        heartbeat.update(event="greedy_reproduction", cases=[c["name"] for c in cases])
        greedy = engine.rollout(cases, [42] * len(cases), deterministic=True, heartbeat=heartbeat)
        save_group(output / "greedy.json", greedy, {"kind": "greedy_reproduction"})
        for case, trajectory in zip(cases, greedy):
            expected = manifest["source_costs"][case["path"]]
            if abs(trajectory["makespan"] - expected) > 1e-6:
                raise RuntimeError(f"Greedy C0 reproduction failed: {case['name']}: {trajectory['makespan']} != {expected}")
        heartbeat.update(event="stochastic_archive_reproduction", api_calls=metrics.get("calls", 0))
        seeds = [trajectory_seed(manifest["seed_base"], c["content_sha256"], 0, 0) for c in cases]
        sampled = engine.rollout(cases, seeds, heartbeat=heartbeat)
        save_group(output / "sample.json", sampled, {"kind": "archive_reproduction"})
    comparisons = []
    for case, trajectory in zip(cases, sampled):
        archive = read_json(manifest["archives"][case["name"]]["path"])
        old = archive["trajectories"][0]
        same = (old["seed"] == trajectory["seed"]
            and abs(old["makespan"] - trajectory["makespan"]) <= 1e-6
            and old["completed"] and len(old["actions"]) == len(trajectory["actions"])
            and action_digest(old["actions"]) == action_digest(trajectory["actions"]))
        comparisons.append({"case": case["name"], "passed": bool(same),
            "archive_cost": old["makespan"], "fresh_cost": trajectory["makespan"],
            "fresh_actions_sha256": action_digest(trajectory["actions"])})
        atomic_json(output / "archive_comparison.json", comparisons)
        if not same:
            raise RuntimeError(f"Archived stochastic trajectory mismatch: {case['name']}; investigate before sampling grid")
    if model_digest(engine) != original_weights:
        raise RuntimeError("Inference changed model state")
    atomic_json(output / "result.json", {"passed": True, "api": metrics,
        "cases": comparisons, "greedy": [trajectory_record(t) for t in greedy],
        "weights_unchanged": True}, overwrite=False)


def run_sampling(manifest, lane, engine, heartbeat):
    from onpolicy.utils.stage3_sampling_audit import DecisionStats, sampling_heads
    tasks = [(case, name) for case in manifest["cases"] for name in manifest["arms"]]
    tasks = tasks[lane::4]
    weights = model_digest(engine)
    root = Path(manifest["root"])
    for task_index, (case, arm) in enumerate(tasks):
        config = manifest["arms"][arm]
        engine.history_mode = config.get("history", "authoritative")
        output = root / "sampling" / arm / case["name"]
        records = []
        started = time.time()
        for first in range(0, manifest["replicas"], engine.width):
            count = min(engine.width, manifest["replicas"] - first)
            seeds = [trajectory_seed(manifest["seed_base"], case["content_sha256"], 0, k)
                     for k in range(first, first + count)]
            heartbeat.update(event="sampling_group", arm=arm, case=case["name"],
                completed_tasks=task_index, total_tasks=len(tasks), completed_replicas=first,
                last_progress_unix=time.time())
            stats = DecisionStats()
            with sampling_heads(engine.policy.ac, config, stats):
                trajectories = engine.rollout([case] * count, seeds, heartbeat=heartbeat)
            save_group(output / f"group_{first:02d}.json", trajectories,
                {"arm": arm, "case_content_sha256": case["content_sha256"],
                 "manifest_sha256": manifest["manifest_sha256"], "decision_stats": stats.result()})
            records.extend(trajectory_record(t) for t in trajectories)
            heartbeat.update(event="group_committed", completed_replicas=len(records),
                last_progress_unix=time.time())
        if model_digest(engine) != weights:
            raise RuntimeError("Inference changed model state")
        source_cost = manifest["source_costs"][case["path"]]
        result = {"arm": arm, "case_id": case["path"], "profile": case["profile"],
            "distribution": case["distribution"], "source_cost": source_cost,
            "elapsed_seconds": time.time() - started, "trajectories": records,
            **best_of_n([t["makespan"] for t in records], source_cost)}
        # Every native sample, not just preflight replica 0, is checked against the archive.
        if arm == "full_t030":
            old = read_json(manifest["archives"][case["name"]]["path"])["trajectories"]
            result["archive_all_32_exact"] = len(old) == len(records) and all(
                a["seed"] == b["seed"] and abs(a["makespan"] - b["makespan"]) <= 1e-6
                and action_digest(a["actions"]) == b["action_sha256"] for a, b in zip(old, records))
            if not result["archive_all_32_exact"]:
                atomic_json(output / "archive_mismatch.json", result, overwrite=False)
                raise RuntimeError(f"Native sampling archive mismatch: {case['name']}")
        atomic_json(output / "result.json", result, overwrite=False)
        print(json.dumps({"event": "case_completed", "lane": lane, "arm": arm, "case": case["name"],
            "mean": result["mean"], "best32": result["prefix_best"]["32"],
            "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        heartbeat.update(completed_tasks=task_index + 1, last_progress_unix=time.time())


def worker(args, manifest):
    verify_manifest(manifest)
    lane, phase = args.lane, args.phase
    path = Path(manifest["root"]) / "workers" / f"{phase}_lane{lane}.json"
    with ProgressHeartbeat(path, phase=phase, lane=lane, last_progress_unix=time.time()) as heartbeat:
        engine = make_engine(manifest)
        try:
            if phase == "preflight":
                run_preflight(manifest, lane, engine, heartbeat)
            else:
                if not read_json(Path(manifest["root"]) / "preflight/result.json")["passed"]:
                    raise ValueError("Preflight did not pass")
                run_sampling(manifest, lane, engine, heartbeat)
        finally:
            engine.close()


def collect_report(manifest):
    import numpy as np
    root = Path(manifest["root"])
    report = {"schema": manifest["schema"], "completed_unix": time.time(),
        "manifest_sha256": manifest["manifest_sha256"], "arms": {}, "automatic_rl": False,
        "notes": manifest["reporting"], "material_passport": {**manifest["material_passport"],
            "verification_status": "completed; diagnostic evidence only"}}
    for arm in manifest["arms"]:
        cases = [read_json(root / "sampling" / arm / c["name"] / "result.json") for c in manifest["cases"]]
        if any(len(c["trajectories"]) != manifest["replicas"] or
               any(not t["completed"] for t in c["trajectories"]) for c in cases):
            raise ValueError("Cannot rank incomplete case/replica grid")
        base = np.asarray([c["source_cost"] for c in cases])
        means = np.asarray([c["mean"] for c in cases])
        best = np.asarray([c["prefix_best"]["32"] for c in cases])
        summary = {"case_count": len(cases), "trajectory_count": len(cases) * manifest["replicas"],
            "source_mean": float(base.mean()), "sample_mean": float(means.mean()),
            "sample_mean_gain_fraction": float(1 - means.mean() / base.mean()),
            "prefix_best_mean": {str(n): float(np.mean([c["prefix_best"][str(n)] for c in cases]))
                for n in manifest["reporting"]["prefix_best"]},
            "best32_gain_fraction": float(1 - best.mean() / base.mean()),
            "incumbent_plus_32_mean": float(np.minimum(base, best).mean()),
            "incumbent_plus_32_gain_fraction": float(1 - np.minimum(base, best).mean() / base.mean()),
            "cases_with_1pct_better_sample": int((best < base * .99).sum()),
            "single_sample_win_fraction": float(np.mean([c["better_than_source_fraction"] for c in cases])),
            "cases": [{k: v for k, v in c.items() if k != "trajectories"} for c in cases]}
        report["arms"][arm] = summary
    atomic_json(root / "report.json", report, overwrite=False)
    lines = ["# Stage3 sampling follow-up", "", "## Material Passport", "",
        "- Origin Skill: academic-research-suite/experiment-agent", "- Origin Mode: run",
        "- Verification Status: completed; 8-case diagnostic only", "- Version Label: sampling_audit_v1",
        "", "No weights updated; no RL or Finalblind evaluation launched.", "",
        "| Arm | Sample mean | Best-32 mean | Best-32 gain vs C0 | C0+32 gain | >1% better cases |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, row in report["arms"].items():
        lines.append(f"| {arm} | {row['sample_mean']:.3f} | {row['prefix_best_mean']['32']:.3f} | "
            f"{row['best32_gain_fraction']:.2%} | {row['incumbent_plus_32_gain_fraction']:.2%} | "
            f"{row['cases_with_1pct_better_sample']}/8 |")
    lines.extend(["", "Positive gain means lower cost. C0+32 retains greedy as an extra incumbent;",
        "it is not an RL improvement. Same seed labels do not imply identical draws after role masking.",
        "Historical API/input ablation uses this C0/contract, not an unidentified historical run.",
        "No statistical significance or architecture verdict is inferred from this screening set.", ""])
    with (root / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_phase(manifest, phase, heartbeat):
    root = Path(manifest["root"])
    processes, logs = [], []
    started = time.time()
    try:
        for lane, cpus in enumerate(manifest["resources"]["lanes"]):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(manifest["resources"]["physical_gpus"][lane])}
            log = (root / "logs" / f"{phase}_lane{lane}.log").open("x")
            logs.append(log)
            command = ["/usr/bin/taskset", "--cpu-list", cpus, sys.executable, "-u", str(Path(__file__)),
                "worker", "--manifest", str(root / "manifest.json"), "--phase", phase, "--lane", str(lane)]
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        while True:
            codes = [p.poll() for p in processes]
            if any(code is not None and code != 0 for code in codes):
                heartbeat.update(worker_exit_codes=codes)
                raise RuntimeError(f"{phase} worker failed: exit_codes={codes}; logs retained; no automatic retry")
            if all(code == 0 for code in codes):
                return
            if time.time() - started > manifest["timeouts_seconds"][phase]:
                raise TimeoutError(f"{phase} timeout")
            warnings = []
            for lane, code in enumerate(codes):
                state_path = root / "workers" / f"{phase}_lane{lane}.json"
                if code is None and state_path.exists():
                    state = read_json(state_path)
                    age = time.time() - state.get("heartbeat_unix", started)
                    if age > manifest["timeouts_seconds"]["progress_warning"]:
                        warnings.append(f"lane{lane}: heartbeat age {age:.0f}s")
                    progress_age = time.time() - state.get("last_progress_unix", started)
                    if progress_age > manifest["timeouts_seconds"]["progress_warning"]:
                        warnings.append(f"lane{lane}: no rollout progress for {progress_age:.0f}s; advisory only")
            heartbeat.update(phase=phase, worker_pids=[p.pid for p in processes],
                worker_exit_codes=codes, warnings=warnings, phase_elapsed_seconds=time.time() - started)
            time.sleep(30)
    finally:
        # Only subprocess groups created by this controller, never unrelated jobs.
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for log in logs:
            log.close()


def run_suite(manifest):
    root = Path(manifest["root"])
    with (root / "suite.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "status.json").exists():
            raise FileExistsError("Existing attempt: explicit recovery plan required")
        (root / "logs").mkdir(exist_ok=True)
        with Heartbeat(root / "status.json", phase="input_verification", automatic_rl=False) as heartbeat:
            verify_manifest(manifest, archives=True)
            run_phase(manifest, "preflight", heartbeat)
            results = [read_json(root / "preflight" / f"lane{i}/result.json") for i in range(4)]
            if not all(r["passed"] for r in results):
                raise RuntimeError("Preflight failed; no sampling grid")
            atomic_json(root / "preflight/result.json", {"passed": True, "lanes": results}, overwrite=False)
            run_phase(manifest, "sampling", heartbeat)
            verify_manifest(manifest)
            collect_report(manifest)
            heartbeat.update(phase="sampling_audit_completed", report=str(root / "report.json"),
                completed_sampling_trajectories=len(manifest["cases"]) * len(manifest["arms"]) * manifest["replicas"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--prior", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--precision-evidence", required=True, type=Path)
    for command in ("run", "worker"):
        cli = sub.add_parser(command)
        cli.add_argument("--manifest", required=True, type=Path)
        if command == "worker":
            cli.add_argument("--phase", choices=("preflight", "sampling"), required=True)
            cli.add_argument("--lane", choices=range(4), type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        manifest = read_json(args.manifest)
        if args.command == "run":
            run_suite(manifest)
        else:
            worker(args, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
