#!/usr/bin/env python
"""Build a verified per-case IGA incumbent set from multiple time budgets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))

from onpolicy.envs.HKBZ.experiment.eval_common import (
    list_case_folders,
    load_case_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--primary_dir", required=True)
    parser.add_argument("--prefix_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--effective_budget", type=float, default=1800.0)
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_verified_candidate(path, case, metadata):
    payload = json.loads(path.read_text(encoding="utf-8"))
    verification = payload.get("verification", {})
    iga = payload.get("iga", {})
    makespan = float(payload.get("makespan", math.inf))
    checks = {
        "schema_version": int(payload.get("schema_version", 0)) >= 2,
        "case": payload.get("case") == case,
        "case_sha256": payload.get("case_sha256") == metadata.get("case_sha256"),
        "profile": payload.get("profile") == metadata.get("profile"),
        "distribution": payload.get("distribution") == metadata.get("distribution"),
        "completion_verified": payload.get("completion_verified") is True,
        "verification_completed": verification.get("completed") is True,
        "verification_timeout": not bool(verification.get("timeout")),
        "verification_cycle": not bool(verification.get("cycle_terminated")),
        "makespan": math.isfinite(makespan) and makespan < 100000.0,
        "verification_makespan": (
            float(verification.get("makespan", math.inf)) == makespan
        ),
        "population": int(iga.get("pop_size", -1)) > 1,
        "generations": int(iga.get("n_gen", -1)) > 0,
        "time_budget": float(iga.get("time_budget_seconds", -1.0)) > 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{path}: failed checks {failed}")
    return payload


def validate_existing(output_dir, cases, metadata, effective_budget):
    manifest_path = output_dir / "generation.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or int(manifest.get("case_count", 0)) != len(cases)
        or float(
            manifest.get("teacher_selection", {}).get(
                "effective_budget_seconds", -1.0
            )
        ) != float(effective_budget)
    ):
        return False
    for case in cases:
        path = output_dir / f"{case}.json"
        if not path.is_file():
            return False
        load_verified_candidate(path, case, metadata[case])
    return True


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    primary_dir = Path(args.primary_dir).resolve()
    prefix_dirs = [Path(path).resolve() for path in args.prefix_dirs]
    output_dir = Path(args.output_dir).resolve()
    cases = list_case_folders(str(dataset_dir))
    metadata = load_case_metadata(str(dataset_dir))
    if set(cases) - set(metadata):
        raise RuntimeError("Dataset manifest is missing training-case metadata.")

    if output_dir.exists():
        if validate_existing(output_dir, cases, metadata, args.effective_budget):
            print(f"[TeacherSelect] Reusing verified incumbent set: {output_dir}")
            return 0
        raise RuntimeError(
            f"Refusing to overwrite incomplete or incompatible output: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    records = []
    source_counts = Counter()
    selected_values = []
    primary_values = []
    for case in cases:
        candidates = []
        for source_dir in [primary_dir, *prefix_dirs]:
            path = source_dir / f"{case}.json"
            candidate = load_verified_candidate(path, case, metadata[case])
            candidates.append((source_dir, candidate))

        primary = candidates[0][1]
        primary_budget = float(primary["iga"]["time_budget_seconds"])
        if primary_budget + 1e-9 < float(args.effective_budget):
            raise ValueError(
                f"{case}: primary budget {primary_budget} < "
                f"effective budget {args.effective_budget}"
            )
        reference = (
            int(primary["iga"]["pop_size"]),
            int(primary["iga"]["n_gen"]),
            int(primary["iga"]["seed"]),
        )
        for source_dir, candidate in candidates[1:]:
            current = (
                int(candidate["iga"]["pop_size"]),
                int(candidate["iga"]["n_gen"]),
                int(candidate["iga"]["seed"]),
            )
            if current != reference:
                raise ValueError(
                    f"{case}: incompatible IGA search configuration in {source_dir}"
                )
            if float(candidate["iga"]["time_budget_seconds"]) > primary_budget:
                raise ValueError(f"{case}: prefix budget exceeds primary budget.")

        selected_dir, selected = min(
            candidates,
            key=lambda item: (
                float(item[1]["makespan"]),
                -float(item[1]["iga"]["time_budget_seconds"]),
            ),
        )
        output = copy.deepcopy(selected)
        candidate_records = [
            {
                "source_dir": str(source_dir),
                "time_budget_seconds": float(candidate["iga"]["time_budget_seconds"]),
                "makespan": float(candidate["makespan"]),
            }
            for source_dir, candidate in candidates
        ]
        output["teacher_selection"] = {
            "schema_version": 1,
            "rule": "minimum_verified_makespan_then_longer_budget",
            "effective_budget_seconds": float(args.effective_budget),
            "selected_source_dir": str(selected_dir),
            "selected_source_budget_seconds": float(
                selected["iga"]["time_budget_seconds"]
            ),
            "candidates": candidate_records,
        }
        atomic_json(output_dir / f"{case}.json", output)
        source_budget = float(selected["iga"]["time_budget_seconds"])
        source_counts[str(int(source_budget))] += 1
        makespan = float(selected["makespan"])
        selected_values.append(makespan)
        primary_values.append(float(primary["makespan"]))
        records.append(
            {
                "case": case,
                "case_id": selected.get("case_id"),
                "seed": selected.get("seed"),
                "profile": selected.get("profile"),
                "distribution": selected.get("distribution"),
                "case_sha256": selected.get("case_sha256"),
                "makespan": makespan,
                "cpu_seconds": float(selected["iga"].get("cpu_seconds", 0.0)),
                "wall_seconds": float(selected["iga"].get("wall_seconds", 0.0)),
                "completed": True,
                "completion_verified": True,
                "cycle_terminated": False,
                "timeout": False,
                "selected_source_budget_seconds": source_budget,
            }
        )

    summary = {
        "case_count": len(records),
        "completed_count": len(records),
        "verified_count": len(records),
        "completion_rate": 1.0,
        "error_count": 0,
        "mean_makespan": statistics.mean(selected_values),
        "std_makespan": statistics.pstdev(selected_values),
        "median_makespan": statistics.median(selected_values),
        "primary_mean_makespan": statistics.mean(primary_values),
        "mean_gain_vs_primary": (
            statistics.mean(primary_values) - statistics.mean(selected_values)
        ),
    }
    manifest = {
        "status": "completed",
        "created_unix_time": time.time(),
        "dataset_test_dir": str(dataset_dir),
        "case_count": len(records),
        "methods_requested": ["IGA"],
        "teacher_selection": {
            "schema_version": 1,
            "rule": "minimum_verified_makespan_then_longer_budget",
            "effective_budget_seconds": float(args.effective_budget),
            "primary_dir": str(primary_dir),
            "prefix_dirs": [str(path) for path in prefix_dirs],
            "selected_source_budget_counts": dict(source_counts),
        },
        "methods": {
            "IGA": {
                "status": "completed",
                "cases": records,
                "summary": summary,
            }
        },
    }
    atomic_json(output_dir / "generation.json", manifest)
    print(json.dumps(manifest["teacher_selection"], indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
