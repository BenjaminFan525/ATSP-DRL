"""Explicit recovery of the pre-Fit orchestration failure, without retraining.

Recovery may change only orchestration/tests/docs. Model, optimizer, data,
environment, evaluation and training budgets remain byte-for-byte unchanged.
Old execution snapshots and scientific artifacts are never overwritten.
"""
from __future__ import annotations
import copy
import fcntl
import os
from pathlib import Path
import re
import shutil
import time

from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json, code_changes

RECOVERY_OVERLAYS = (
    "onpolicy/scripts/train/run_stage3_representation.py",
    "onpolicy/utils/stage3_representation_recovery.py",
    "onpolicy/scripts/train/launch_stage3_representation_resume.sh",
    "onpolicy/envs/HKBZ/test/test_stage3_representation_orchestration.py",
    "STAGE3_REPRESENTATION_20260908.md",
)
ADMIN_FIELDS = {"code", "execution", "input_files", "material_passport", "created_unix", "manifest_sha256", "recovery"}


def identity(value):
    return digest_json({k:v for k,v in value.items() if k != "manifest_sha256"})


def require_dead(pid):
    if not pid:
        return
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return
    raise RuntimeError(f"Old worker PID still exists; recovery will not preempt it: {pid}")


def check_fit_not_started(root):
    root = Path(root)
    for phase in ("fit", "train"):
        if any(p.is_file() for p in (root/phase).rglob("*")):
            raise ValueError(f"{phase} already has artifacts; pre-Fit recovery cannot discard them")
    for name in ("training_admission.json", "screen_admission.json", "result.json", "validator/STOP"):
        if (root/name).exists():
            raise ValueError(f"Study has advanced beyond the recoverable boundary: {name}")
    for state in ("pending", "running", "failed"):
        if list((root/"validator"/state).glob("*.json")):
            raise ValueError("Validation queue is not clean; reconcile requests explicitly")


def diagnostic_evidence(manifest):
    root = Path(manifest["root"])
    evidence = {}
    for phase, names in (("contract", ["E0", "E1", "E2", "E3"]), ("mechanism", manifest["mechanisms"])):
        for name in names:
            result, status = root/phase/name/"result.json", root/phase/name/"status.json"
            row, state = read_json(result), read_json(status)
            if state.get("status") != "completed":
                raise ValueError(f"Unfinished diagnostic: {phase}/{name}")
            if phase == "contract":
                if row.get("passed") is not True or row["representation"]["variant"] != name:
                    raise ValueError(f"Failed architecture contract: {name}")
            elif row.get("completed") is not True or row.get("historical_trace_exact") is not True:
                raise ValueError(f"Historical replay did not complete exactly: {name}")
            evidence.update({str(result):digest_file(result), str(status):digest_file(status)})
    path = root/"validator/results/C0_e000000.json"
    baseline = read_json(path)
    if (baseline.get("ok") is not True or baseline.get("training_episodes") != 0
            or baseline.get("request_id") != "C0_e000000"
            or baseline.get("code_sha256") != manifest["code"]["sha256"]
            or baseline.get("contract_sha256") != manifest["contract_sha256"]
            or baseline.get("cases_sha256") != digest_json(manifest["splits"]["tune"])):
        raise ValueError("C0 validation identity is not reusable")
    cases = baseline["evaluation"]["cases"]
    expected = {c["path"] for c in manifest["splits"]["tune"]}
    if len(cases) != len(expected) or {c["case_id"] for c in cases} != expected or any(
            not c.get("completed") or abs(c["makespan"]-manifest["source_costs"][c["case_id"]]) > 1e-6 for c in cases):
        raise ValueError("C0 Tune cases did not reproduce individually")
    checkpoint = root/"contract/E0/initial_c0.pt"
    if digest_file(checkpoint) != baseline["checkpoint_sha256"]:
        raise ValueError("Validated C0 checkpoint changed")
    evidence.update({str(path):digest_file(path), str(checkpoint):digest_file(checkpoint)})
    return evidence


def validate_recovery(manifest):
    """Static scientific equivalence; safe to call after Fit has started."""
    recovery = manifest["recovery"]
    prior_path = Path(recovery["prior_manifest"])
    if digest_file(prior_path) != recovery["prior_manifest_file_sha256"]:
        raise ValueError("Original manifest changed")
    prior = read_json(prior_path)
    if identity(prior) != prior["manifest_sha256"] or prior["manifest_sha256"] != recovery["prior_manifest_sha256"]:
        raise ValueError("Original manifest identity corrupted")
    if {k:v for k,v in prior.items() if k not in ADMIN_FIELDS} != {k:v for k,v in manifest.items() if k not in ADMIN_FIELDS}:
        raise ValueError("Recovery changed scientific settings, data or budgets")
    if {k:v for k,v in prior["execution"].items() if k != "code_root"} != {
            k:v for k,v in manifest["execution"].items() if k != "code_root"}:
        raise ValueError("Recovery changed execution dependencies/settings")
    old, new = prior["code"]["files"], manifest["code"]["files"]
    if not old.keys() <= new.keys() or any(old.get(k) != new[k] and k not in RECOVERY_OVERLAYS for k in new):
        raise ValueError("Only the reviewed orchestration/test/documentation fix may change")
    if code_changes(old, prior["execution"]["code_root"]):
        raise ValueError("Original frozen source changed")
    for path, sha in recovery["evidence"].items():
        if digest_file(path) != sha:
            raise ValueError(f"Reused diagnostic evidence changed: {path}")
    if diagnostic_evidence(prior) != recovery["evidence"]:
        raise ValueError("Diagnostic admission evidence is incomplete")
    if digest_file(recovery["audit_path"]) != recovery["audit_sha256"]:
        raise ValueError("Recovery audit changed")


def prepare_recovery(manifest_path, attempt_id, workspace):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", attempt_id):
        raise ValueError("Unsafe recovery attempt ID")
    manifest_path, workspace = Path(manifest_path).resolve(), Path(workspace).resolve()
    prior = read_json(manifest_path)
    if prior.get("recovery") or identity(prior) != prior["manifest_sha256"]:
        raise ValueError("Expected the intact original manifest")
    root = Path(prior["root"]).resolve()
    attempt = root/"recovery"/attempt_id
    if attempt.exists():
        raise FileExistsError("Recovery attempt exists; use a new explicit ID")
    with (root/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(root/"status.json")
        if (state.get("status") != "failed" or state.get("phase") != "teacher_fit_diagnostic"
                or "File exists" not in state.get("error", "") or "commands/E0.json" not in state.get("error", "")):
            raise ValueError("This entry only recovers the audited pre-Fit name collision")
        require_dead(state.get("pid"))
        for status in list(root.glob("contract/*/status.json")) + list(root.glob("mechanism/*/status.json")) + list(root.glob("validator/workers/*/status.json")):
            require_dead(read_json(status).get("pid"))
        check_fit_not_started(root)
        evidence = diagnostic_evidence(prior)
        if code_changes(prior["code"]["files"], prior["execution"]["code_root"]):
            raise ValueError("Original source snapshot changed")
        for path, sha in prior["input_files"].items():
            if digest_file(path) != sha:
                raise ValueError(f"Original input changed: {path}")
        # Inventory before creating any recovery output. Source/artifacts remain
        # in place; the two mutable root dashboard files also get exact copies.
        inventory = {str(p.relative_to(root)):{"sha256":digest_file(p), "bytes":p.stat().st_size}
                     for p in root.rglob("*") if p.is_file() and "recovery" not in p.relative_to(root).parts
                     and not p.name.endswith(".lock")}
        attempt.mkdir(parents=True)
        preserved = {}
        for name in ("status.json", "resource_leases.json"):
            target = attempt/"pre_recovery"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/name, target)
            preserved[name] = {"path":str(target), "sha256":digest_file(target)}
        files = {}
        for relative, sha in prior["code"]["files"].items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe frozen source path")
            target = attempt/"source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(prior["execution"]["code_root"])/relative, target)
            if digest_file(target) != sha:
                raise ValueError("Source copy mismatch")
            files[relative] = sha
        for relative in RECOVERY_OVERLAYS:
            target = attempt/"source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            sha = digest_file(workspace/relative)
            shutil.copy2(workspace/relative, target)
            if digest_file(target) != sha:
                raise ValueError("Recovery overlay changed during copy")
            files[relative] = sha
        audit_path = attempt/"recovery_audit.json"
        atomic_json(audit_path, {"reason":"cross-phase task names collided in commands/logs/job table",
            "authorized_action":"user requested bug fix and experiment recovery",
            "resume_phase":"teacher_fit_diagnostic", "created_unix":time.time(),
            "reused_contracts":4, "reused_mechanisms":6, "reused_c0_cases":60,
            "training_or_environment_code_changed":False, "scientific_settings_changed":False,
            "overlays":list(RECOVERY_OVERLAYS), "original_inventory":inventory,
            "original_inventory_sha256":digest_json(inventory), "preserved_root_statuses":preserved,
            "material_passport":{"origin_skill":"academic-research-suite/experiment-agent", "origin_mode":"run",
                "origin_date":"2026-09-08", "verification_status":"diagnostic reuse verified; recovery not yet launched",
                "version_label":"representation_fit_recovery_v1", "data_classification":"internal; no upload"}}, overwrite=False)
        manifest = copy.deepcopy(prior)
        manifest.update(created_unix=time.time(), code={"files":files,"sha256":digest_json(files)})
        manifest["execution"]["code_root"] = str(attempt/"source")
        manifest["input_files"].update({str(manifest_path):digest_file(manifest_path), **evidence})
        manifest["recovery"] = {"attempt_id":attempt_id, "attempt_dir":str(attempt), "resume_phase":"teacher_fit_diagnostic",
            "prior_manifest":str(manifest_path), "prior_manifest_file_sha256":digest_file(manifest_path),
            "prior_manifest_sha256":prior["manifest_sha256"], "evidence":evidence,
            "prior_status":preserved["status.json"], "audit_path":str(audit_path), "audit_sha256":digest_file(audit_path)}
        manifest["material_passport"].update(version_label="representation_v1_fit_recovery", verification_status="prepared recovery; outcomes pending")
        manifest["manifest_sha256"] = identity(manifest)
        validate_recovery(manifest)
        atomic_json(attempt/"manifest.json", manifest, overwrite=False)
    print(attempt/"manifest.json", flush=True)


def activate_recovery(manifest, manifest_path):
    """Caller holds the original controller lock. Exactly one activation."""
    if not manifest.get("recovery") or manifest["recovery"]["resume_phase"] != "teacher_fit_diagnostic":
        raise ValueError("No explicit pre-Fit recovery manifest")
    validate_recovery(manifest)
    root, attempt = Path(manifest["root"]), Path(manifest["recovery"]["attempt_dir"])
    if Path(manifest_path).resolve() != attempt/"manifest.json":
        raise ValueError("Recovery manifest path differs from its frozen identity")
    if digest_file(root/"status.json") != manifest["recovery"]["prior_status"]["sha256"]:
        raise ValueError("Root status changed since recovery preparation; do not overwrite")
    require_dead(read_json(root/"status.json").get("pid"))
    check_fit_not_started(root)
    atomic_json(attempt/"started.json", {"pid":os.getpid(), "started_unix":time.time(),
        "manifest":str(Path(manifest_path).resolve()), "manifest_sha256":manifest["manifest_sha256"]}, overwrite=False)
    atomic_json(root/"active_execution.json", {"manifest":str(Path(manifest_path).resolve()),
        "code_root":manifest["execution"]["code_root"], "recovery_attempt":manifest["recovery"]["attempt_id"],
        "original_manifest":manifest["recovery"]["prior_manifest"], "started_unix":time.time()}, overwrite=False)
