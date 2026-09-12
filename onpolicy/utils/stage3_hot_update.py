"""Auditable execution-only migration of the running representation study."""
import copy
from pathlib import Path
import re
import shutil
import time

from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json, code_changes

OPTIONS = dict(disable_activation_checkpoint=True, cache_frozen_features=True,
               cache_mib=1024, defer_statistics=True)
OVERLAYS = (
    "onpolicy/algorithms/gnn_mappo/algorithm/MAPPOPolicy.py",
    "onpolicy/algorithms/gnn_mappo/algorithm/gnn_actor_critic.py",
    "onpolicy/algorithms/utils/stage3_encoder.py",
    "onpolicy/runner/shared/stage3_research_engine.py",
    "onpolicy/runner/shared/stage3_representation_engine.py",
    "onpolicy/utils/stage3_performance.py",
    "onpolicy/utils/stage3_hot_update.py",
    "onpolicy/scripts/train/run_stage3_representation.py",
    "onpolicy/scripts/train/stage3_representation_worker.py",
    "onpolicy/scripts/train/hot_update_stage3_representation.py",
    "onpolicy/scripts/train/launch_stage3_representation_hot_update.sh",
    "onpolicy/scripts/train/check_stage3_performance.py",
    "onpolicy/envs/HKBZ/test/test_stage3_performance.py",
    "onpolicy/envs/HKBZ/test/test_stage3_hot_update.py",
    "STAGE3_PERFORMANCE_20260908.md",
)
ADMIN = {"created_unix", "material_passport", "code", "execution", "manifest_sha256", "hot_update"}


def identity(manifest):
    return digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def execution_identities(manifest):
    identities = {manifest["manifest_sha256"]: manifest["code"]["sha256"]}
    if manifest.get("hot_update"):
        identities.update(manifest["hot_update"]["parent_execution_identities"])
    return identities


def validate_hot_update(manifest):
    upgrade = manifest["hot_update"]
    prior_path = Path(upgrade["parent_manifest"])
    if digest_file(prior_path) != upgrade["parent_manifest_file_sha256"]:
        raise ValueError("Hot-update parent manifest changed")
    prior = read_json(prior_path)
    if identity(prior) != prior["manifest_sha256"] or identity(manifest) != manifest["manifest_sha256"]:
        raise ValueError("Execution manifest identity changed")
    if prior.get("hot_update"):
        raise ValueError("A second migration needs its own reviewed continuation protocol")
    if {k: v for k, v in manifest.items() if k not in ADMIN} != {k: v for k, v in prior.items() if k not in ADMIN}:
        raise ValueError("Hot update changed scientific settings, inputs, resources or budgets")
    if {k: v for k, v in manifest["execution"].items() if k != "code_root"} != {
            k: v for k, v in prior["execution"].items() if k != "code_root"}:
        raise ValueError("Hot update changed dependencies or execution settings")
    if upgrade["options"] != OPTIONS or upgrade["parent_execution_identities"] != execution_identities(prior):
        raise ValueError("Unreviewed performance options or parent identities")
    if code_changes(prior["code"]["files"], prior["execution"]["code_root"]):
        raise ValueError("Original frozen source changed")
    old, new = prior["code"]["files"], manifest["code"]["files"]
    if not old.keys() <= new.keys() or any(old.get(k) != new[k] and k not in OVERLAYS for k in new):
        raise ValueError("Execution update extends beyond reviewed overlay files")
    evidence = upgrade["cpu_verification"]
    if digest_file(evidence["path"]) != evidence["sha256"] or not read_json(evidence["path"])["passed"]:
        raise ValueError("Missing successful CPU verification")
    if prior.get("recovery"):
        from onpolicy.utils.stage3_representation_recovery import validate_recovery
        validate_recovery(prior)
    return prior


def prepare(parent_path, attempt_id, workspace, verification, legacy_service):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", attempt_id):
        raise ValueError("Unsafe attempt ID")
    if not re.fullmatch(r"hkbz-s3repr-[A-Za-z0-9_-]+\.service", legacy_service):
        raise ValueError("Expected this study's explicit systemd unit")
    parent_path, workspace, verification = map(lambda p: Path(p).resolve(), (parent_path, workspace, verification))
    prior = read_json(parent_path)
    root = Path(prior["root"]).resolve()
    state = read_json(root/"status.json")
    if state.get("phase") != "screen_960" or state.get("status") != "running" or state.get("active_manifest") != str(parent_path):
        raise ValueError("Only the active screen_960 execution can be rolled forward")
    if (root/"screen_admission.json").exists() or (root/"hot_update_request.json").exists():
        raise ValueError("Study advanced or already has a hot-update request")
    evidence = read_json(verification)
    if not evidence.get("passed"):
        raise ValueError("CPU verification has not passed")
    for relative in OVERLAYS:
        if evidence["overlay_hashes"][relative] != digest_file(workspace/relative):
            raise ValueError(f"Code changed since verification: {relative}")
    attempt = root/"hot_updates"/attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    files = {}
    for relative, sha in prior["code"]["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe snapshot path")
        target = attempt/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(prior["execution"]["code_root"])/relative
        if digest_file(source) != sha:
            raise ValueError("Parent snapshot changed")
        shutil.copy2(source, target)
        files[relative] = sha
    for relative in OVERLAYS:
        target = attempt/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace/relative, target)
        files[relative] = digest_file(target)
        if files[relative] != evidence["overlay_hashes"][relative]:
            raise ValueError("Overlay changed while copying")
    manifest = copy.deepcopy(prior)
    manifest.update(created_unix=time.time(), code={"files": files, "sha256": digest_json(files)})
    manifest["execution"]["code_root"] = str(attempt/"source")
    manifest["hot_update"] = dict(attempt_id=attempt_id, attempt_dir=str(attempt),
        parent_manifest=str(parent_path), parent_manifest_file_sha256=digest_file(parent_path),
        parent_execution_identities=execution_identities(prior), options=OPTIONS,
        cpu_verification={"path": str(verification), "sha256": digest_file(verification)},
        legacy_service=legacy_service, legacy_controller_pid=state["pid"],
        migration="next complete checkpoint; no committed updates discarded; independent source/artifacts",
        validation="one shared queue; parent and updated identities explicitly admitted")
    manifest["material_passport"].update(version_label="representation_execution_perf_v1",
        verification_status="CPU verified; GPU canary and rolling activation pending")
    manifest["manifest_sha256"] = identity(manifest)
    validate_hot_update(manifest)
    atomic_json(attempt/"manifest.json", manifest, overwrite=False)
    return attempt/"manifest.json"


def resume_payload_allowed(payload, manifest, arm, until, plan):
    expected = until - 480
    episodes = payload.get("training_episodes", -1)
    if manifest.get("hot_update") and until == 960:
        expected = episodes
        if not isinstance(episodes, int) or episodes <= 0 or episodes >= until or episodes % 240:
            raise ValueError("Rolling migration requires a completed 240-trajectory checkpoint")
    if (payload.get("diagnostic_only") or payload.get("forbidden_as_rl_initialization")
            or payload.get("not_resumable_partial_group") or payload.get("arm") != arm
            or payload.get("source_sha256") != manifest["source"]["sha256"]
            or payload.get("schedule_sha256") != digest_json(plan)
            or payload.get("seed") != manifest["training"]["seed"] or episodes != expected
            or payload.get("next_group", -1) * 8 != episodes
            or payload.get("batch_composition") != manifest["arms"][arm]["batch"]
            or payload.get("optimizer_recipe") != "O0"
            or payload.get("protocol_sha256") not in execution_identities(manifest)):
        raise ValueError("Checkpoint arm/schedule/seed/recipe/lineage/cursor mismatch")
    return payload["protocol_sha256"]


def numa_plan(core_groups, gpus):
    """Explicit reviewed 2 x 32-core / 4-GPU topology; never silently guess."""
    if core_groups != [[i, i+64] for i in range(64)] or [g["index"] for g in gpus] != list(range(8)):
        raise ValueError("NUMA mapping requires the audited 64-core, eight-GPU host")
    trainers, validators, controller = [], [], []
    for node in range(2):
        first = node*32
        controller += core_groups[first]
        validators.append(dict(gpu=node*4, cpus=sum(core_groups[first+1:first+4], []), cuda_memory_fraction=.15))
        for local in range(4):
            begin = first+4+local*7
            trainers.append(dict(gpu=node*4+local, cpus=sum(core_groups[begin:begin+7], []), cuda_memory_fraction=.65))
    return dict(controller=controller, validators=validators, trainers=trainers)
