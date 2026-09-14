#!/usr/bin/env python3
"""Freeze the next Stage3 study without starting jobs or reading blind outcomes."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onpolicy.utils.stage3_research import (ROOT, TRAIN_ROOT, EVAL_ROOT, SOURCE, SOURCE_SHA,
    BASE_MANIFEST, SOURCE_BASELINE, SOURCE_EVAL, SCHEMA, read_json, digest_file,
    digest_json, atomic_json, case_record, stratified_cases, half_cpu_topology)


def prepare(output):
    from onpolicy.scripts.train.prepare_stage3_source_relative_rl import (
        validate_source_baseline, SOURCE_EVAL_CONTRACT, EXPECTED_MODEL_SHA256)
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Refusing to change an existing experiment manifest")
    baseline_identity = validate_source_baseline(SOURCE_BASELINE, source_checkpoint=SOURCE,
                                                dataset_dir=TRAIN_ROOT)
    train = [case_record(p) for p in sorted(TRAIN_ROOT.glob("case_*"))]
    diag = stratified_cases(train, {"iid": 16, "ood_stress": 14, "ood_scale": 2}, 20260905)
    fit = stratified_cases(diag, {"iid": 8, "ood_stress": 6, "ood_scale": 2}, 20260906)
    used = {r["path"] for r in diag}
    probe = stratified_cases([r for r in train if r["path"] not in used],
        {"iid": 32, "ood_stress": 28, "ood_scale": 4}, 20260907)
    used.update(r["path"] for r in probe)
    pilot = stratified_cases([r for r in train if r["path"] not in used],
        {"iid": 60, "ood_stress": 54, "ood_scale": 6}, 20260908)
    splits = {"train600": train, "train_diag32": diag, "train_fit16": fit,
              "train_probe64": probe, "train_pilot120": pilot}
    for split, expected in (("tune", 60), ("gate", 120), ("finalblind", 60)):
        records = [case_record(p) for p in sorted((EVAL_ROOT / split).glob("case_*"))]
        if len(records) != expected:
            raise ValueError(f"{split} expected {expected}, found {len(records)}")
        splits[split] = records
    # Use case content fingerprints, not just directory names, for leakage checks.
    seen = {}
    for split in ("train600", "tune", "gate", "finalblind"):
        for record in splits[split]:
            sha = record["case_sha256"]
            if sha in seen:
                raise ValueError(f"Content leakage: {seen[sha]} / {split}: {record['path']}")
            seen[sha] = split
    source_costs = {str((TRAIN_ROOT / name).resolve()): value["makespan"]
                    for name, value in read_json(SOURCE_BASELINE)["cases"].items()}
    evaluation = read_json(SOURCE_EVAL)
    if evaluation["policy_tau"] != .3:
        raise ValueError("C0 evaluation temperature changed")
    tune_map = {str(Path(r["case_path"]).resolve()): r for r in evaluation["cases"]}
    for case in splits["tune"]:
        record = tune_map[case["path"]]
        if not record["completed"] or record["case_sha256"] != case["case_sha256"]:
            raise ValueError("C0 Tune lineage/completion mismatch")
        source_costs[case["path"]] = record["makespan"]
    from onpolicy.scripts.train.prepare_stage3_diagnostic_resume import snapshot_sources
    from onpolicy.utils.stage3_research import protocol_identity
    code = snapshot_sources(ROOT, output / "source")
    topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    rows = [tuple(map(int, line.split(","))) for line in topology.splitlines() if not line.startswith("#")]
    cores = half_cpu_topology(rows)
    cpu_list = sorted(cpu for siblings in cores for cpu in siblings)
    if cpu_list != list(range(32)) + list(range(64, 96)):
        raise ValueError("Hardware topology differs from the approved half-resource mapping")
    manifest = {"schema": SCHEMA, "created_unix": time.time(), "root": str(output),
        "material_passport": {"id": output.name, "type": "Code Experiment", "status": "prepared",
            "data_classification": "internal; no upload", "purpose": "diagnose and improve RL over fixed C0"},
        "software": {"python": sys.version, "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in
                ("torch", "torch-geometric", "numpy", "scipy", "pymoo", "gymnasium", "PyYAML")}},
        "source": {"path": str(SOURCE), "sha256": SOURCE_SHA, "model_sha256": EXPECTED_MODEL_SHA256,
                   "baseline": baseline_identity, "tune_sha256": digest_file(SOURCE_EVAL)},
        "code": code, "splits": splits,
        "source_costs": source_costs, "contract": SOURCE_EVAL_CONTRACT,
        "contract_sha256": digest_json(SOURCE_EVAL_CONTRACT),
        "resources": {"physical_gpus": [0, 1, 2, 3], "physical_cpu_cores": 32,
            "logical_cpus": cpu_list, "parent_cpuset": "0-31,64-95",
            "lanes": ["0-7,64-71", "8-15,72-79", "16-23,80-87", "24-31,88-95"],
            "training_gpu_count": 3, "validator_gpu": 3, "workers_per_lane": 8,
            "torch_threads": 1, "cuda_memory_fraction": .8, "canary_reserved_gib_limit": 18.0},
        "diagnostic": {"samples_per_case": 32, "iga_seconds_per_case": 1800,
            "iga_population": 20, "bc_passes": 10, "bc_lr": .0001,
            "bc_shared_lr": .00001, "counterfactual_cases": 8, "counterfactual_branches": 8,
            "branch_role_cycle": [0, 1, 2], "source_reproduction_abs_tolerance": .1,
            "replay_logp_abs_tolerance": .002, "bc_nll_ratio_max": .5,
            "bc_fit_gain_min": .01, "sampling_improvable_case_fraction_min": .25},
        "training": {"arms": ["R0", "R1", "R2", "R3"], "group_size": 8,
            "pilot_passes": 2, "pilot_episodes": 1920, "pilot_seed": 20260911,
            "eval_every_groups": 60, "ppo_epochs": 2, "tbptt_steps": 8,
            "actor_lr": .000005, "critic_lr": .0001, "clip": .2, "target_kl": .02,
            "reward_coef": .01, "gamma": 1.0, "advantage_normalization": "none",
            "objective_reduction": "equal case mean, trajectory mean, time/agent sum",
            "r0_advantage": "Monte Carlo remaining cost return - pretrained role value",
            "r1_advantage": "0.01 * (C0 same-case cost - sampled trajectory cost)",
            "r2_advantage": "0.01 * (leave-one-out group cost - sampled trajectory cost)",
            "r3_auxiliary": "one self-winning trajectory BC step every 4 groups; no IGA labels",
            "elite_per_case": 1, "confirmation_seeds": [20260921, 20260922, 20260923],
            "confirmation_episodes": 4800, "automatic_confirmation": False,
            "automatic_finalblind": False},
        "gates": {"pilot_consecutive_gain_min": .01, "pilot_consecutive_evaluations": 2,
            "formal_seed_mean_gain_min": .02, "formal_all_seeds_positive": True,
            "ood_stress_and_joint_regression_max": .005, "tail_regression_max": .01,
            "formal_win_fraction_min": .60, "formal_regression_over_5pct_fraction_max": .05},
        "timeouts_seconds": {"contract": 21600, "diagnostics": 259200,
            "pilot": 604800, "evaluation_job": 21600, "suite": 1209600},
        "notes": ["Gate has historical exposure: confirmation, not untouched test.",
            "Finalblind metadata only; outcomes remain unopened.",
            "Architecture changes require diagnostic failure review, never automatic.",
            "All four arms restart C0 and reset optimizers; BC-fit models cannot seed RL."]}
    manifest["execution"] = {"code_root": str(output / "source"), "workspace_root": str(ROOT),
        "workspace_code_policy": "advisory", "snapshot_policy": "isolated",
        "protocol_sha256": protocol_identity(manifest),
        "input_files": {str(path): digest_file(path) for path in
            (BASE_MANIFEST, SOURCE_BASELINE, SOURCE_EVAL)}}
    atomic_json(output / "manifest.json", manifest, overwrite=False)
    print(str(output / "manifest.json"), flush=True)
    print({name: dict(Counter(r["distribution"] for r in records)) for name, records in splits.items()})
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output)
