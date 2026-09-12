"""Explicit, pre-training recovery after a host driver failure.

Only completed C0 evaluations are reused. All GPU admission checks are rerun;
neither diagnostic models nor partial PPO batches initialize formal training.
"""
import copy
import fcntl
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import time

from onpolicy.utils.stage3_research import (
    atomic_json, code_changes, digest_file, digest_json, read_json,
)
from onpolicy.utils.stage3_full_data import (
    EVAL_PROTOCOL, HISTORY, identity, source_costs,
)
from onpolicy.utils.stage3_local_exploration import MODES
from onpolicy.utils.stage3_representation import RepresentationQueue

OVERLAYS = (
    "onpolicy/scripts/train/run_stage3_full_data.py",
    "onpolicy/utils/stage3_full_data_restart.py",
    "onpolicy/envs/HKBZ/test/test_stage3_full_data_restart.py",
)
ADMIN_FIELDS = {"root", "created_unix", "material_passport", "execution", "code",
                "input_files", "manifest_sha256", "recovery"}


def assert_same_study(parent, child):
    if {k:v for k,v in parent.items() if k not in ADMIN_FIELDS} != {
            k:v for k,v in child.items() if k not in ADMIN_FIELDS}:
        raise ValueError("Recovery changed the research protocol")
    if {k:v for k,v in parent["execution"].items() if k != "code_root"} != {
            k:v for k,v in child["execution"].items() if k != "code_root"}:
        raise ValueError("Recovery changed execution dependencies/options")
    left, right = parent["code"]["files"], child["code"]["files"]
    if any(left.get(k) != right.get(k) for k in set(left) | set(right) if k not in OVERLAYS):
        raise ValueError("Recovery changed training/environment source")


def checked_baselines(parent):
    root = Path(parent["root"])
    payload = read_json(root / "baselines.json")
    costs = source_costs(parent)
    expected_ids, rows, evidence = [], [], {}
    for split in ("train_full600", "validation", "tune"):
        for start in range(0, len(parent["splits"][split]), 60):
            cases = parent["splits"][split][start:start+60]
            request_id = f"C0_{split}_e000000_s{start:04d}"
            path = root / "validator/results" / f"{request_id}.json"
            result = read_json(path)
            expected = {"checkpoint_sha256": parent["source"]["sha256"],
                "cases_sha256": digest_json(cases), "contract_sha256": parent["contract_sha256"],
                "code_sha256": parent["code"]["sha256"], "tau": .3, "seed": 42,
                "evaluation_protocol": EVAL_PROTOCOL, "training_exploration": MODES["J"],
                "representation": "F_SHARED", "protocol_sha256": parent["manifest_sha256"]}
            evaluation = result["evaluation"]
            if (not result["ok"] or result["error"] is not None
                    or result["request_id"] != request_id or result["training_episodes"] != 0
                    or result["cache_key"] != RepresentationQueue.identity(expected)
                    or any(result[k] != expected[k] for k in
                           ("checkpoint_sha256", "cases_sha256", "contract_sha256", "code_sha256"))
                    or evaluation["history"] != HISTORY or evaluation["split"] != split
                    or evaluation["evaluation_protocol"] != EVAL_PROTOCOL
                    or [r["case_id"] for r in evaluation["cases"]] != [c["path"] for c in cases]
                    or any(not r["completed"] or not r["behavior_deterministic"]
                           or r["seed"] != 42 or r["makespan"] != costs[r["case_id"]]
                           for r in evaluation["cases"])):
                raise ValueError(f"Invalid reusable C0 evaluation: {request_id}")
            expected_ids.append(request_id)
            rows.extend(evaluation["cases"])
            evidence[str(path)] = digest_file(path)
    if payload["requests"] != expected_ids or payload["cases"] != rows:
        raise ValueError("Baseline summary differs from completed evaluation evidence")
    evidence[str(root / "baselines.json")] = digest_file(root / "baselines.json")
    return payload, evidence


def prepare(args, workspace):
    parent_path, output = args.prior.resolve(), args.output.resolve()
    parent = read_json(parent_path)
    root = Path(parent["root"])
    if output.exists() or output.parent != root.parent:
        raise ValueError("Recovery requires a new sibling output directory")
    if parent.get("recovery") or identity(parent) != parent["manifest_sha256"]:
        raise ValueError("Expected the original, unchanged full-data manifest")
    status_path = root / "status.json"
    state = read_json(status_path)
    if (state["status"] != "failed" or state.get("formal_training_started")
            or "nvidia-smi" not in state.get("error", "")
            or any((root / "train").glob("**/*"))
            or (root / "training_admission.json").exists() or (root / "result.json").exists()):
        raise ValueError("This restart only handles the pre-training NVML interruption")
    if (root / "validator/STOP").exists() or any(
            p for name in ("pending", "running") for p in (root / "validator" / name).glob("*.json")):
        raise ValueError("Unexpected closed or undrained validator pool")
    # Never steal a live controller or any of its original workers.
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        statuses = [status_path, *root.glob("reference/**/status.json"),
                    *root.glob("contract/**/status.json"), *root.glob("validator/workers/*/status.json")]
        for path in statuses:
            pid = read_json(path).get("pid")
            if not isinstance(pid, int) or pid <= 1:
                raise ValueError(f"Invalid historical worker PID: {path}")
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(f"Historical PID still exists; refusing restart: {pid}")
        if code_changes(parent["code"]["files"], parent["execution"]["code_root"]):
            raise ValueError("Parent frozen source changed")
        if digest_file(parent["source"]["path"]) != parent["source"]["sha256"]:
            raise ValueError("Original C0 changed")
        if {n:importlib.metadata.version(n) for n in parent["execution"]["packages"]} != parent["execution"]["packages"]:
            raise ValueError("Python dependencies changed")
        baseline, evidence = checked_baselines(parent)
        for path, checksum in parent["input_files"].items():
            if digest_file(path) != checksum:
                raise ValueError(f"Parent input changed: {path}")
        gpu_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version",
                                              "--format=csv,noheader,nounits"], text=True)
        gpu_rows = [[x.strip() for x in line.split(",")] for line in gpu_output.splitlines()]
        if [(int(r[0]), r[1]) for r in gpu_rows] != [(g["index"], g["uuid"]) for g in parent["resources"]["gpus"]]:
            raise ValueError("GPU inventory differs from the registered study")
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader,nounits"], text=True).strip():
            raise RuntimeError("GPU jobs are active; do not preempt them")
        files = dict(parent["code"]["files"])
        for relative, checksum in files.items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe frozen source path")
            source, target = Path(parent["execution"]["code_root"]) / relative, output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest_file(target) != checksum:
                raise ValueError("Frozen source changed while copying")
        for relative in OVERLAYS:
            source, target = Path(workspace) / relative, output / "source" / relative
            checksum = digest_file(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest_file(target) != checksum:
                raise ValueError("Recovery overlay changed while copying")
            files[relative] = checksum
        child = copy.deepcopy(parent)
        child.update(root=str(output), created_unix=time.time(), code={"files":files,"sha256":digest_json(files)})
        child["execution"]["code_root"] = str(output / "source")
        evidence.update({str(parent_path):digest_file(parent_path), str(status_path):digest_file(status_path)})
        child["input_files"].update(evidence)
        child["recovery"] = {"parent_manifest":str(parent_path), "baseline_evidence":evidence,
            "reason":"User requested restart after repairing NVIDIA/NVML driver mismatch",
            "observed_driver_versions":gpu_rows, "reused_c0_cases":len(baseline["cases"]),
            "training_episodes_before_failure":0, "formal_initialization":"original C0",
            "all_gpu_admissions_reexecuted":True, "thresholds_unchanged":True,
            "historical_outputs_retained":True}
        child["material_passport"].update(version_label="full_data_driver_recovery1",
            verification_status="prepared; driver-revalidated tests and GPU admissions pending")
        assert_same_study(parent, child)
        child["manifest_sha256"] = identity(child)
        old_schedule = read_json(root / "schedule.json")
        if old_schedule["sha256"] != child["training"]["schedule_sha256"] or digest_json(old_schedule["groups"]) != old_schedule["sha256"]:
            raise ValueError("Original full-data schedule changed")
        atomic_json(output / "schedule.json", old_schedule, overwrite=False)
        atomic_json(output / "manifest.json", child, overwrite=False)
        print(output / "manifest.json", flush=True)


def reuse_baselines(manifest):
    recovery = manifest["recovery"]
    parent = read_json(recovery["parent_manifest"])
    assert_same_study(parent, manifest)
    for path, checksum in recovery["baseline_evidence"].items():
        if digest_file(path) != checksum:
            raise ValueError(f"Recovery input changed: {path}")
    baseline, _ = checked_baselines(parent)
    result = {**baseline, "protocol_sha256":manifest["manifest_sha256"],
        "reuse_origin":str(Path(parent["root"])/"baselines.json"),
        "reuse_origin_sha256":digest_file(Path(parent["root"])/"baselines.json"),
        "reuse_original_protocol_sha256":parent["manifest_sha256"],
        "reuse_note":"Historical completed C0 evaluations; not reevaluated under the new driver. GPU admission is rerun."}
    atomic_json(Path(manifest["root"])/"baselines.json", result, overwrite=False)
    source_costs(manifest)
    return result["cases"]
