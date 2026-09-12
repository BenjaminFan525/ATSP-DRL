#!/usr/bin/env python3
"""Explicit, audited recovery of an interrupted pre-RL Stage3 diagnostic run.

Never restores or changes another experiment's source files. The original
manifest, outputs and logs remain intact; execution gets its own source copy.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.utils.stage3_research import (BASE_MANIFEST, SOURCE_BASELINE, SOURCE_EVAL,
    atomic_json, best_of_n, code_changes, digest_file, digest_json, protocol_identity,
    read_json, validate_sample_trajectories, verify_cases)


def snapshot_sources(workspace, destination, listing=None):
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Source snapshot must be a new directory")
    if listing is None:
        listing = subprocess.check_output(["git", "ls-files", "--cached", "--others",
            "--exclude-standard", "onpolicy", "utils", "arrangement.py"],
            cwd=workspace, text=True).splitlines()
    files = {}
    for relative in sorted(set(listing)):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe source snapshot path")
        if (path.suffix not in (".py", ".yaml", ".yml", ".json", ".sh")
                or any(part in ("dataset", "results", "result", "__pycache__") for part in path.parts)):
            continue
        source = workspace / path
        if not source.is_file():
            continue
        source.resolve().relative_to(workspace)
        sha = digest_file(source)
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest_file(target) != sha or digest_file(source) != sha:
            raise RuntimeError(f"Source changed while taking snapshot: {relative}; prepare a new attempt")
        files[relative] = sha
    if not files:
        raise ValueError("Empty source snapshot")
    return {"files": files, "sha256": digest_json(files)}


def audit_recovery(manifest):
    root = Path(manifest["root"])
    status = read_json(root / "status.json")
    if status["status"] != "failed" or status["phase"] != "diagnostics":
        raise ValueError("This recovery entry is for a failed pre-RL diagnostic attempt only")
    try:
        os.kill(status["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("Original controller PID is still alive; refusing a second controller")
    # Do not pretend model training can resume from a diagnostic case boundary.
    for phase in ("replay", "bc_heads", "bc_full", "pilot_R0", "pilot_R1", "pilot_R2", "pilot_R3"):
        if (root / phase).exists() and any((root / phase).iterdir()):
            raise ValueError(f"Recovery of partial {phase} needs its own checkpoint audit")
    if (root / "diagnostic_gate.json").exists() or (root / "report.json").exists():
        raise ValueError("Do not resume a finished/gated study through this entry")
    for state in ("pending", "running", "failed"):
        if list((root / "validator" / state).glob("*.json")):
            raise ValueError(f"Validator {state} jobs need explicit recovery decisions")
    if digest_file(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("C0 checkpoint changed")
    if digest_json(manifest["contract"]) != manifest["contract_sha256"]:
        raise ValueError("Environment contract changed")
    for name, records in manifest["splits"].items():
        verify_cases(records, training=name.startswith("train"))

    artifacts = {}

    def keep(relative):
        artifacts[str(relative)] = digest_file(root / relative)
        return read_json(root / relative) if str(relative).endswith(".json") else None

    contract = keep("contract/result.json")
    if not contract["passed"] or max(contract["source_cost_errors"]) > .1:
        raise ValueError("Original contract did not pass")
    for arm in ("R0", "R1", "R2", "R3"):
        keep(f"contract/canary_{arm}.pt")
    baseline = keep("validator/results/c0_tune60.json")
    expected_tune = {case["path"] for case in manifest["splits"]["tune"]}
    rows = baseline["evaluation"]["cases"]
    if (not baseline["ok"] or baseline["checkpoint_sha256"] != manifest["source"]["sha256"]
            or len(rows) != len(expected_tune) or {r["case_id"] for r in rows} != expected_tune
            or any(not r["completed"] or abs(r["makespan"] - manifest["source_costs"][r["case_id"]]) > .1
                   for r in rows)):
        raise ValueError("Original full C0 Tune evaluation is not reusable")
    cf = keep("counterfactual/result.json")
    expected_cf = {case["path"] for case in manifest["splits"]["train_diag32"][:8]}
    if (not cf["passed"] or len(cf["cases"]) != 8
            or {row["case_id"] for row in cf["cases"]} != expected_cf
            or any(len(row["branches"]) != 8 or not all(b["completed"] for b in row["branches"])
                   for row in cf["cases"])):
        raise ValueError("Counterfactual outputs are incomplete")
    for case in manifest["splits"]["train_diag32"][:8]:
        row = keep(f"counterfactual/cases/{case['name']}.json")
        if row != next(r for r in cf["cases"] if r["case_id"] == case["path"]):
            raise ValueError("Counterfactual case/summary mismatch")

    from onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels import _contract, _teacher_reusable
    cases = manifest["splits"]["train_diag32"]
    if keep("iga/cases.json") != [case["name"] for case in cases]:
        raise ValueError("IGA case ordering changed")
    progress = {"contract": "completed", "c0_tune60": "completed", "counterfactual": "8/8",
                "iga_completed": 0, "sample_completed": 0, "sample_partial_trajectories": 0}
    for case in cases:
        name = case["name"]
        teacher_path, result_path = root / f"iga/teachers/{name}.json", root / f"iga/cases/{name}.json"
        if teacher_path.exists() or result_path.exists():
            expected = _contract(name, case["case_sha256"], manifest["diagnostic"]["iga_seconds_per_case"], 60.,
                future_intent_horizon=3, future_intent_mode="bounded_frontier", frontier_max_requests=4,
                request_capacity_per_plane=5, release_aware_eta=True, reservation_mode="hard",
                reservation_grace_seconds=300., slack_forecast_seconds=0.)
            if not _teacher_reusable(teacher_path, expected) or not result_path.is_file():
                raise ValueError(f"Incomplete/incompatible IGA artifact pair: {name}")
            teacher = keep(teacher_path.relative_to(root))
            result = keep(result_path.relative_to(root))
            if (result["status"] != "completed" or result["case_sha256"] != case["case_sha256"]
                    or abs(result["makespan"] - teacher["makespan"]) > .1):
                raise ValueError(f"IGA teacher/result disagreement: {name}")
            progress["iga_completed"] += 1
        sample_path = root / f"sample/cases/{name}.json"
        if sample_path.exists():
            saved = keep(sample_path.relative_to(root))
            validate_sample_trajectories(saved["trajectories"], case, complete=True)
            c0 = manifest["source_costs"][case["path"]]
            expected = best_of_n([t["makespan"] for t in saved["trajectories"]], c0)
            if saved["source_cost"] != c0 or any(saved[key] != value for key, value in expected.items()):
                raise ValueError(f"Stored sample summary disagrees with trajectories: {name}")
            progress["sample_completed"] += 1
        partial_path = root / f"sample/partial/{name}.json"
        if partial_path.exists():
            saved = keep(partial_path.relative_to(root))
            if saved["case_content_sha256"] != case["content_sha256"]:
                raise ValueError(f"Partial sample case identity mismatch: {name}")
            validate_sample_trajectories(saved["trajectories"], case)
            if not sample_path.exists():
                progress["sample_partial_trajectories"] += len(saved["trajectories"])
    return artifacts, progress


def prepare(manifest_path, attempt_id):
    manifest_path = Path(manifest_path).resolve()
    original = read_json(manifest_path)
    root = Path(original["root"]).resolve()
    if manifest_path != root / "manifest.json":
        raise ValueError("Use the original study manifest as the recovery reference")
    if not re.fullmatch(r"resume_[A-Za-z0-9_-]+", attempt_id):
        raise ValueError("Attempt ID must be a simple resume_* name")
    with (root / "suite.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        artifacts, progress = audit_recovery(original)
        attempt = root / "attempts" / attempt_id
        attempt.mkdir(parents=True, exist_ok=False)
        shutil.copy2(root / "status.json", attempt / "previous_status.json")
        manifest = copy.deepcopy(original)
        changes = code_changes(original["code"]["files"], ROOT)
        manifest["code"] = snapshot_sources(ROOT, attempt / "source")
        inputs = (manifest_path, BASE_MANIFEST, SOURCE_BASELINE, SOURCE_EVAL)
        manifest["execution"] = {"code_root": str(attempt / "source"), "workspace_root": str(ROOT),
            "workspace_code_policy": "advisory", "snapshot_policy": "isolated",
            "protocol_sha256": protocol_identity(original),
            "input_files": {str(path): digest_file(path) for path in inputs}}
        manifest["resume"] = {"id": attempt_id, "attempt_dir": str(attempt),
            "authorized_action": "user explicitly requested resume and relaxed shared-source protection",
            "prepared_unix": time.time(), "parent_manifest": str(manifest_path),
            "original_code_sha256": original["code"]["sha256"], "workspace_changes_since_original": changes,
            "completed_phases": ["contract", "counterfactual"],
            "reused_artifacts": artifacts, "progress": progress,
            "compatibility_gate": "six C0 greedy cases, forced-action/logp replay, and fresh full Tune60",
            "partial_search_policy": "reuse completed IGA cases; uncommitted searches restart their per-case budget",
            "review": "Stage1 baseline builders are inactive for proposed; legacy hkbz_runner is not this engine."}
        atomic_json(attempt / "manifest.json", manifest, overwrite=False)
        atomic_json(attempt / "recovery_audit.json", {"progress": progress, "artifacts": artifacts,
            "workspace_changes": changes, "source_snapshot": manifest["code"]["sha256"],
            "original_protocol": protocol_identity(original), "resumed_protocol": protocol_identity(manifest),
            "compatibility_status": "pending runtime verification"}, overwrite=False)
        print(str(attempt / "manifest.json"), flush=True)
        print(progress, flush=True)
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.attempt_id)
