#!/usr/bin/env python3
"""Export the frozen Stage2 handoff and byte-exact historical result records.

This maintenance tool does not train, evaluate, remove source artifacts, inspect
credentials, or mutate Git. Run it with a CPU affinity/BLAS cap. Existing bundle
files may only be reused when their bytes are identical. Retired development
archives belong to the separate retirement procedure and are never touched.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile


PROTOCOL = "stage2_frozen_b0_v1"
DATE = "20260912"
HF_RUN = "stage2_bc_hf_20260912_gpu0_half_tune60_r1"
IGA_RUN = "stage2_supervised_hf_labels_20260901_r1"
MODEL_ROOT = "onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
B0 = MODEL_ROOT + "/stage2_bc_injection_20260905_gpu0_half_pilot_r2_B0_none_seed11/run1/models/checkpoint_Best.pt"
STAGE1 = MODEL_ROOT + "/stage1_departure_reward_formal_dual_20260816_r1_formal_P5_team_time_potential_fixed_seed3/run1/models/checkpoint_Best.pt"
R1 = MODEL_ROOT + "/stage2_ready_reliable_20260904_local_a6000_r2_ready_reliable_R1_symmetric_hybrid_seed1/run1/models/checkpoint_BestPredictor.pt"
TEACHER_INDEX = "result/hkbz_train_logs/" + IGA_RUN + "/stage2_bc_labels/iga1800/stage2_bc_teacher_index.json"
EXPECTED = {
    "b0": "b7740b2b688c83dc7fa65bed6381243cdf2f6675f4b96808ff05a41d4e44e7e8",
    "stage1": "8ce0244f9b0b3877e2cc581a2479a6e5edab1854c1ed643749712612d0c5379d",
    "r1": "c0c4825946e469b680e764a4c4b9f3bcbcfdaa66ccfb2fee221633b5d3afb1f1",
    "teacher_index": "3b2cbe592d22fb2d4610a8edec3d6f10baec7b5258e3e6ca069e7efa6cfa9d35",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_bytes())


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def source_bytes(path: Path, root: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink source: {path}")
    path.resolve().relative_to(root.resolve())
    return path.read_bytes()


def write_new_or_identical(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite differing artifact: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    if path.read_bytes() != data:
        raise ValueError(f"Artifact byte verification failed: {path}")


def describe(path: str, data: bytes, original_path: str | None = None):
    item = {"path": path, "sha256": digest(data), "size_bytes": len(data)}
    if original_path is not None:
        item["original_path"] = original_path
    return item


def selected_history(root: Path) -> list[Path]:
    """Explicit report paths only; no teacher trajectories, weights, or secrets."""
    selected = set()
    log_root = root / "result/hkbz_train_logs"
    for run in sorted(log_root.iterdir()):
        if not run.is_dir() or run.is_symlink() or not run.name.startswith("stage2"):
            continue
        for path in run.rglob("*.json"):
            rel = path.relative_to(run)
            named_summary = path.name in {
                "comparison.json", "full_report.json", "summary.json", "analysis.json"
            }
            report_directory = any(
                part in rel.parts for part in ("evaluations", "commands", "records", "analysis", "checks")
            )
            iga_hf_case = (
                run.name == IGA_RUN and rel.parts[0] == "hf_search" and "cases" in rel.parts
            )
            if len(rel.parts) == 1 or named_summary or report_directory or iga_hf_case:
                selected.add(path)
    models = root / MODEL_ROOT
    for run in sorted(models.iterdir()):
        if not run.is_dir() or run.is_symlink() or not run.name.startswith("stage2"):
            continue
        for path in run.rglob("*.json"):
            if "evaluations" in path.relative_to(run).parts:
                selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def archive_results(root: Path, bundle: Path):
    relative_archive = "history/experiment_results.tar.gz"
    archive = bundle / relative_archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    records = []
    fd, temporary = tempfile.mkstemp(prefix="stage2-results-", suffix=".tar.gz", dir=archive.parent)
    os.close(fd)
    temp = Path(temporary)
    try:
        with temp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for path in selected_history(root):
                        data = source_bytes(path, root)
                        # Validate that selected result records are JSON, without reserializing.
                        json.loads(data)
                        relative = path.relative_to(root).as_posix()
                        entry = tarfile.TarInfo(relative)
                        entry.size = len(data)
                        entry.mode = 0o644
                        entry.uid = entry.gid = 0
                        entry.uname = entry.gname = ""
                        entry.mtime = 0
                        tar.addfile(entry, io.BytesIO(data))
                        records.append({
                            "member": relative, "original_path": str(path.resolve()),
                            "size_bytes": len(data), "sha256": digest(data),
                        })
        archive_data = temp.read_bytes()
        if len(archive_data) >= 90 * 1024 * 1024:
            raise ValueError("Result archive exceeds 90 MiB; split by history group before export")
        write_new_or_identical(archive, archive_data)
    finally:
        # Only the explicit temporary file created above is removed.
        temp.unlink(missing_ok=True)
    index = {
        "protocol": PROTOCOL, "original_bytes_preserved": True,
        "parameter_projection": False, "record_count": len(records),
        "uncompressed_source_bytes": sum(row["size_bytes"] for row in records),
        "archive": describe(relative_archive, archive_data),
        "selection": {
            "run_prefix": "stage2",
            "run_logs": "root JSON; evaluations/commands/records/analysis/checks; named summaries; historical HF IGA cases",
            "model_runs": "all nested evaluations JSON in Stage2 model runs",
            "large_engineering_json": "included without truncation or projection",
            "excluded": "raw logs, tensorboard, teacher trajectories, model candidates other than frozen handoff, datasets; retained unchanged locally",
        },
        "records": records,
    }
    index_data = json_bytes(index)
    write_new_or_identical(bundle / "history/results_index.json", index_data)
    return index, index_data, archive_data


def verify(bundle: Path):
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("scientific_goal_confirmed") is not False:
        raise ValueError("Unexpected frozen protocol or scientific claim")
    for name, entry in manifest["files"].items():
        path = bundle / entry["path"]
        data = source_bytes(path, bundle)
        if len(data) != entry["size_bytes"] or digest(data) != entry["sha256"]:
            raise ValueError(f"Bundled file digest mismatch: {name}")
    index = read_json(bundle / manifest["files"]["results_index"]["path"])
    expected = {record["member"]: record for record in index["records"]}
    if len(expected) != index["record_count"]:
        raise ValueError("Duplicate indexed archive member")
    observed = set()
    with tarfile.open(bundle / index["archive"]["path"], "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in observed:
                raise ValueError(f"Unexpected archive entry: {member.name}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive entry: {member.name}")
            data = stream.read()
            row = expected[member.name]
            if digest(data) != row["sha256"] or len(data) != row["size_bytes"]:
                raise ValueError(f"Archive byte mismatch: {member.name}")
            observed.add(member.name)
    if observed != set(expected):
        raise ValueError("Archive is missing indexed results")
    return {
        "protocol": PROTOCOL, "passed": True,
        "bundled_files_verified": len(manifest["files"]),
        "archived_result_files_verified": len(observed),
        "archived_source_bytes": index["uncompressed_source_bytes"],
        "compressed_result_bytes": index["archive"]["size_bytes"],
        "checkpoint_bytes": sum(manifest["files"][key]["size_bytes"] for key in ("b0", "stage1", "r1")),
        "verification_scope": "byte identity and archive completeness; no model rollout or new scientific experiment",
    }


def export(root: Path, bundle: Path):
    hf_root = root / "result/hkbz_train_logs" / HF_RUN
    hf = read_json(hf_root / "manifest.json")
    analysis = read_json(hf_root / "analysis.json")
    status = read_json(hf_root / "run_status.json")
    if not analysis.get("complete") or status["completed_physical_case_episodes"] != 840:
        raise ValueError("Source HF experiment is not complete")
    files = {}
    sources = {
        "b0": (B0, "checkpoints/b0.pt"),
        "stage1": (STAGE1, "checkpoints/stage1.pt"),
        "r1": (R1, "checkpoints/r1.pt"),
        "teacher_index": (TEACHER_INDEX, "provenance/teacher_index.json"),
        "stage1_source_command": (hf["stage1_source"]["source_command_path"], "provenance/stage1_source_command.json"),
        "stage1_handoff": (hf["stage1_source"]["handoff_path"], "provenance/stage1_m2_handoff.json"),
        "ac_config": ("onpolicy/config/ac.yaml", "config/ac.yaml"),
        "env_config": ("onpolicy/config/env_resource_joint.yaml", "config/env_resource_joint.yaml"),
        "hf_manifest": (str(hf_root / "manifest.json"), "provenance/hf_manifest.json"),
        "hf_analysis": (str(hf_root / "analysis.json"), "results/hf_analysis.json"),
        "hf_status": (str(hf_root / "run_status.json"), "results/hf_status.json"),
        "b0_evaluation": (hf["source"]["evaluation"], "results/b0_source_evaluation.json"),
        "hf_baseline": (str(hf_root / "evaluations/H2_F4.json"), "results/h2_f4_evaluation.json"),
    }
    for path in sorted((hf_root / "checks").glob("*.json")):
        sources["hf_check_" + path.stem] = (str(path), "results/checks/" + path.name)
    for key, (original, target) in sources.items():
        path = root / original
        data = source_bytes(path, root)
        expected = EXPECTED.get(key)
        if key == "stage1_source_command":
            expected = hf["stage1_source"]["source_command_sha256"]
        elif key == "stage1_handoff":
            expected = hf["stage1_source"]["handoff_sha256"]
        elif key == "b0_evaluation":
            expected = hf["source"]["evaluation_sha256"]
        if key in {"ac_config", "env_config"}:
            expected = hf["code_fingerprint"][path.relative_to(root).as_posix()]
        if expected and digest(data) != expected:
            raise ValueError(f"Original provenance mismatch: {key}")
        write_new_or_identical(bundle / target, data)
        files[key] = describe(target, data, str(path.resolve()))
    index, index_data, archive_data = archive_results(root, bundle)
    files["results_index"] = describe("history/results_index.json", index_data)
    files["results_archive"] = describe("history/experiment_results.tar.gz", archive_data)
    summary = """# Stage2 frozen handoff — 2026-09-12

Stage2 development is frozen, not scientifically declared successful. The
handoff is the exactly preserved B0 checkpoint with Hungarian decoding, H2/F4,
soft reservations (grace 300 s, safety 60 s), request capacity 5 per plane,
Ready injection `none`, release/deadline/departure-aware lookahead enabled.

The 60-case historical B0 mean makespan is 8352.922222 seconds. Historical
same-H/F IGA180 and IGA1800 means are 8200.438889 and 8094.988889 seconds:
B0 remains 1.86% and 3.19% worse respectively. No Stage2 configuration has
established the requested IGA superiority. Those IGA comparisons are historical
solution-quality references, not a new hardware-equal runtime certification.

The latest frozen-model HF sweep completed 14 jobs / 840 case-episodes (12 HF
configurations and two additional controls) on the same 60 cases. H2/F4 had the
lowest observed mean. Source and terminal replays passed. It performed zero
actor/critic updates and zero teacher queries. Repeated cases are not 840
independent samples; this is not independent BC retraining or confirmation.

`history/experiment_results.tar.gz` retains every selected original JSON record
byte-for-byte, including large engineering records. `history/results_index.json`
records the exact original path, SHA256, size and tar member for each. The
selection includes all Stage2 run-root JSON, evaluations, commands, records,
analyses, checks, named summaries, model-run evaluations, and historical HF IGA
per-case results. No metrics, parameter summaries, or failed runs were silently
projected away. Old physical rules and development gates differ across runs;
do not pool their raw scores as a common benchmark.

All original raw artifacts remain locally unchanged in the ignored `result/`
and `onpolicy/scripts/results/` trees. Dataset content, teacher trajectories,
TensorBoard/log streams and non-selected candidate checkpoints are not copied
into the portable bundle. The teacher index is preserved for provenance only;
its referenced training label files are not required for frozen inference.
Stage3 supplies its own data and training protocol. Original command paths are
historical evidence, not portable executable commands.

Use the frozen Stage2 helper and verification entry point for portable loading.
The embedded old BC-improvement `stage2_scientific_gate.passed` is not an IGA-goal
pass; the release manifest explicitly records `scientific_goal_confirmed=false`.
"""
    summary_data = summary.encode()
    write_new_or_identical(bundle / "results/README.md", summary_data)
    files["results_readme"] = describe("results/README.md", summary_data)
    manifest = {
        "protocol": PROTOCOL, "schema_version": 1, "frozen_date": "2026-09-12",
        "status": "frozen_for_stage3_handoff", "scientific_goal_confirmed": False,
        "automatic_stage2_training": False, "automatic_stage3_training": False,
        "checkpoint_modification": "none; exact original bytes retained",
        "files": files, "planning_contract": hf["source_planning_contract"],
        "runtime": {
            "ac_config": files["ac_config"]["path"], "env_config": files["env_config"]["path"],
            "decoder": "hungarian", "request_capacity": 5,
            "request_ready_policy_injection": "none",
            "evaluation_workers": 12, "evaluation_seed": 1, "evaluation_tau": 0.3,
            "environment_semantics_version": "progressive-departure-r014-pipeline-v2",
            "observation_schema_id": "hkbz-global-a9ef97bf5c338262910d",
            "original_environment_argv": hf["model_arguments"]["environment_argv"],
            "original_argv_is_executable_handoff": False,
            "stage3_datasets_and_budgets": "supplied by Stage3; no training labels or queues inherited",
        },
        "provenance": {
            "original_hf_run": str(hf_root), "stage1_source": hf["stage1_source"],
            "source_b0": hf["source"], "source_bc_epoch": 1,
            "source_bc_seed": 11, "stage1_seed": 3,
            "old_checkpoint_gate_scope": "historical BC improvement, not current IGA superiority target",
        },
        "results": {
            "b0_tune60_mean": 8352.922222222222,
            "historical_iga180_h2_f4_mean": 8200.43888888889,
            "historical_iga1800_h2_f4_mean": 8094.988888888889,
            "hf_completed_jobs": 14, "hf_completed_case_episodes": 840,
            "independent_confirmation": False, "result_archive_records": index["record_count"],
        },
        "preservation": {
            "raw_sources_deleted": False, "history_parameter_projection": False,
            "raw_results_remain_ignored_locally": True,
            "datasets_and_teacher_trajectories_bundled": False,
            "retired_development_archives": "owned by separate retirement procedure under history/",
        },
    }
    write_new_or_identical(bundle / "manifest.json", json_bytes(manifest))
    report = verify(bundle)
    write_new_or_identical(bundle / "export_verification.json", json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    bundle = (args.bundle or root / "artifacts/stage2_frozen" / DATE).resolve()
    bundle.relative_to(root)
    print(json.dumps(verify(bundle) if args.verify_only else export(root, bundle), indent=2))


if __name__ == "__main__":
    main()
