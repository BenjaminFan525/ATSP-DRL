#!/usr/bin/env python3
"""Prepare and run the full Stage1-data shared/private Stage3 comparison."""
import argparse
from collections import Counter
import fcntl
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
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.utils.stage3_research import (TRAIN_ROOT, case_record, atomic_json,
    read_json, digest_json, digest_file, paired_summary)
from onpolicy.utils.stage3_full_data import (ARMS, COUNTS, HISTORY, identity, schedule, verify,
    publish, source_costs)
from onpolicy.utils.stage3_full_policy import resource_plan
from onpolicy.utils.stage3_representation import discover_topology, risk_pass
from onpolicy.utils.stage3_numerics import RUNTIME
from onpolicy.scripts.train.run_stage3_full_policy import FullSuite, compare_checkpoints
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat

OVERLAYS = (
    "onpolicy/utils/stage3_full_data.py", "onpolicy/utils/stage3_performance.py",
    "onpolicy/utils/stage3_distributed.py", "onpolicy/runner/shared/stage3_research_engine.py",
    "onpolicy/runner/shared/stage3_representation_engine.py",
    "onpolicy/scripts/train/stage3_full_policy_worker.py", "onpolicy/scripts/train/run_stage3_full_policy.py",
    "onpolicy/scripts/train/stage3_full_data_worker.py", "onpolicy/scripts/train/run_stage3_full_data.py",
    "onpolicy/scripts/train/launch_stage3_full_data_all.sh",
    "onpolicy/envs/HKBZ/test/test_stage3_full_data.py", "STAGE3_FULL_DATA_20260912.md",
    "onpolicy/utils/stage3_full_data_restart.py", "onpolicy/envs/HKBZ/test/test_stage3_full_data_restart.py",
)


def physical_identity(case):
    return digest_json({name: read_json(Path(case["path"]) / name)
                        for name in case["files"] if name != "metadata.json"})


def prepare(args):
    prior = read_json(args.prior)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new full-data run directory; never overwrite artifacts")
    if digest_file(prior["source"]["path"]) != prior["source"]["sha256"]:
        raise ValueError("Original C0 changed")
    contract = prior["contract"]
    if (contract["device_future_intent_horizon"], contract["device_frontier_max_requests"],
            contract["device_lookahead_reservation_mode"], prior["history"]) != (2, 4, "soft", HISTORY):
        raise ValueError("Expected the verified H2/F4/soft semantic-history parent")
    prior_result = read_json(Path(prior["root"]) / "result.json")
    if not prior_result.get("completed") or prior_result.get("target"):
        raise ValueError("Confirmation exposure requires review before reuse")
    confirmation_paths = [p for p in (Path(prior["root"])/"validator/results").glob("*.json")
                          if "confirmation" in p.name.lower()]
    if confirmation_paths:
        raise ValueError("Prior confirmation outcomes already exposed")
    splits = {"train_full600": [case_record(p) for p in sorted(TRAIN_ROOT.glob("case_*"))],
        "validation": [case_record(p) for p in sorted((ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation").glob("case_*"))],
        "tune": prior["splits"]["tune"], "confirmation": prior["splits"]["confirmation"]}
    if {k: len(v) for k, v in splits.items()} != {"train_full600":600,"validation":120,"tune":60,"confirmation":120}:
        raise ValueError("Unexpected full-data/evaluation split sizes")
    physical = {k: {physical_identity(c) for c in v} for k, v in splits.items()}
    if any(len(physical[k]) != len(splits[k]) for k in splits):
        raise ValueError("Duplicate physical cases within a split")
    for name in ("validation", "tune", "confirmation"):
        if physical["train_full600"] & physical[name]:
            raise ValueError(f"Train/evaluation leakage: {name}")
    if physical["confirmation"] & (physical["validation"] | physical["tune"]):
        raise ValueError("Confirmation overlaps exposed validation cases")
    plan = schedule(splits["train_full600"], 2026091201)
    gpus = []
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"], text=True).splitlines():
        index, uuid, memory = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(index), "uuid": uuid, "memory_mib": int(memory)})
    if [g["index"] for g in gpus] != list(range(8)):
        raise ValueError("Expected eight GPUs")
    cores = discover_topology()
    placement = resource_plan(cores)
    files = {}
    for relative, checksum in prior["code"]["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe frozen source path")
        source = Path(prior["execution"]["code_root"]) / relative
        if digest_file(source) != checksum:
            raise ValueError(f"Historical snapshot changed: {relative}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative] = checksum
    for relative in OVERLAYS:
        source, target = ROOT / relative, output / "source" / relative
        checksum = digest_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest_file(target) != checksum:
            raise ValueError("Workspace changed while snapshotting")
        files[relative] = checksum
    diagnostic = []
    for profile in ("balanced", "low_load_ood", "resource_ood", "stress_joint"):
        candidates = [c for c in splits["train_full600"] if c["profile"] == profile]
        if not candidates:
            raise ValueError(f"No canary case for {profile}")
        diagnostic.append(sorted(candidates, key=lambda c: c["content_sha256"])[0])
    manifest = {"schema": "stage3-full-data-two-arm-v1", "root": str(output), "created_unix": time.time(),
        "material_passport": {"origin_skill":"academic-research-suite/experiment-agent", "origin_mode":"run",
            "origin_date":"2026-09-12", "verification_status":"prepared; admission and scientific outcomes unverified",
            "version_label":"full_data_shared_private_v1"},
        "source": prior["source"], "contract": contract, "contract_sha256": digest_json(contract), "history": HISTORY,
        "splits": splits, "arms": ARMS, "code": {"files": files, "sha256": digest_json(files)},
        "input_files": {str(args.prior.resolve()):digest_file(args.prior)},
        "execution": {"code_root":str(output / "source"), "workspace_root":str(ROOT), "python":sys.executable,
            "worker":"onpolicy/scripts/train/stage3_full_data_worker.py",
            "workspace_code_policy":"advisory; frozen runtime strict, parallel baseline edits allowed",
            "packages":{n:importlib.metadata.version(n) for n in prior["execution"]["packages"]}},
        "training": {"seed":2026091201, "seeds":[2026091201], "data_epochs":8, "episodes_per_data_epoch":960,
            "max_training_episodes":7680, "global_batch":32, "samples_per_visit":1, "actor_lr":1e-5,
            "critic_lr":1e-4, "ppo_epochs":2, "tbptt_steps":8, "mode":"J", "advantage":"source",
            "post_update_hard_kl":.04, "soft_kl":.02, "clip_norm":1., "ppo_clip":.2,
            "objective":"visit mean -> time/role sum", "schedule_sha256":digest_json(plan),
            "coverage_counts":COUNTS, "sampling_weights":{"iid":.5,"ood_stress":.45,"ood_scale":.05}},
        "resources": {"gpus":gpus, "core_groups":cores, "plan":placement, "validator_workers":2,
            "validator_width":3, "memory_high_gib":96, "memory_max_gib":108,
            "validator_queue_scope":"one global durable queue and two consumers across BOTH arms"},
        "diagnostic": {"cases":diagnostic, "global_batch":8, "single_vs_distributed_model_atol":2e-6},
        "capacity": {"global_batch":32, "rank_reserved_max_gib":19, "service_sampled_max_gib":96},
        "numerics":RUNTIME, "timeouts_seconds": {"suite":1209600,"reference":43200,"contract":43200,
            "capacity":43200,"train":1209600,"validator":1209600,"validator_request":21600,"progress_warning":900},
        "exposure": {"old_train_diagnostic_splits_now_training":True, "validation_and_tune_previously_exposed":True,
            "confirmation_costs_unopened":True, "prior_result_sha256":digest_file(Path(prior["root"])/"result.json"),
            "physical_train_overlap_counts":{s:0 for s in ("validation","tune","confirmation")}},
        "scientific_target": {"greedy_gain":.02,"private_incremental_gain":.005,"single_seed_only":True},
        "automatic_multi_seed":False, "teacher_fit_gate":False, "automatic_finalblind":False}
    manifest["manifest_sha256"] = identity(manifest)
    atomic_json(output / "manifest.json", manifest, overwrite=False)
    atomic_json(output / "schedule.json", {"sha256":digest_json(plan),"groups":plan}, overwrite=False)
    print(output / "manifest.json", flush=True)


class FullDataSuite(FullSuite):
    def wait_requests(self, requests, hb):
        while any(self.queue.poll(r) is None for r in requests):
            self.check(hb)
            hb.update(event="await_shared_validation", pending_requests=[r for r in requests if self.queue.poll(r) is None])
            time.sleep(5)
        return [self.queue.poll(r)["evaluation"] for r in requests]

    def baselines(self, hb, *, confirmation=False):
        if not confirmation and self.m.get("recovery"):
            from onpolicy.utils.stage3_full_data_restart import reuse_baselines
            rows = reuse_baselines(self.m)
            hb.update(event="reused_verified_c0_baselines", reused_c0_cases=len(rows))
            return rows
        splits = ("confirmation",) if confirmation else ("train_full600", "validation", "tune")
        requests = []
        for split in splits:
            for start in range(0, len(self.m["splits"][split]), 60):
                requests.append(publish(self.m, self.m["source"]["path"], "C0", 0, "F_SHARED",
                    split=split, start=start, count=min(60,len(self.m["splits"][split])-start), baseline=True))
        results = self.wait_requests(requests, hb)
        rows = [r for result in results for r in result["cases"]]
        costs = {r["case_id"]:r["makespan"] for r in rows}
        atomic_json(self.root / ("confirmation_baseline.json" if confirmation else "baselines.json"), {
            "protocol_sha256":self.m["manifest_sha256"], "source_sha256":self.m["source"]["sha256"],
            "contract_sha256":self.m["contract_sha256"], "history":HISTORY,
            "costs":costs, "costs_sha256":digest_json(costs), "cases":rows, "requests":requests}, overwrite=False)
        source_costs(self.m, confirmation=confirmation)
        return rows

    def eval_at(self, arm, episodes, split, hb):
        request = f"{arm}_{split}_e{episodes:06d}_s0000"
        return self.wait_requests([request], hb)[0]

    def contracts(self, hb):
        hb.update(phase="single_gpu_reference_and_cache_equivalence")
        references = self.run_group("reference", "c0", hb, reference=True)
        hb.update(phase="four_gpu_contract")
        distributed = self.run_group("contract", "c0", hb)
        comparisons = {}
        for arm in ARMS:
            ref, rows = references[arm][0], distributed[arm]
            if any(not r["passed"] or r["trace_identity"] != ref["trace_identity"] for r in rows):
                raise RuntimeError(f"Single/four-GPU trace parity failed: {arm}")
            comparisons[arm] = compare_checkpoints(ref["after_update_checkpoint"], rows[0]["after_update_checkpoint"], 2e-6)
            for split in ("validation", "tune"):
                result = self.eval_at(arm, 0, split, hb)
                if any(abs(r["makespan"]-self.costs[r["case_id"]]) > 1e-6 for r in result["cases"]):
                    raise RuntimeError(f"Migrated C0 changed on {split}: {arm}")
        receipt = {"arms":comparisons, "passed":all(r["passed"] for r in comparisons.values()),
            "cache_equivalence":{a:{k:references[a][0][k] for k in
                ("execution_cache_equivalence","resume_passed","update_seconds","repeat_update_seconds")} for a in ARMS}}
        atomic_json(self.root / "contract/parity.json", receipt, overwrite=False)
        if not receipt["passed"]:
            raise RuntimeError("Single/four-GPU model/Adam parity failed")

    def capacity(self, dense_case, hb):
        self.phase_sampled_peak = 0
        results = []
        for label, case in (("normal",None),("dense",dense_case)):
            hb.update(phase=f"capacity_batch32_{label}", dense_case=dense_case)
            rows = self.run_group("capacity", f"b32_{label}", hb, batch_size=32, dense_case=case)
            results.extend(row for group in rows.values() for row in group)
        peak = max(r["peak_reserved_gib"] for r in results)
        passed = all(r["passed"] for r in results) and peak <= 19 and self.phase_sampled_peak/2**30 <= 96
        receipt = {"passed":passed,"eligible_arms":list(ARMS) if passed else [], "batch_size":32,
            "max_rank_reserved_gib":peak,"service_sampled_peak_gib":self.phase_sampled_peak/2**30,
            "baseline_sha256":digest_file(self.root/"baselines.json"),"protocol_sha256":self.m["manifest_sha256"],
            "dense_case":dense_case,"dense_scope":"largest graph observed in the full C0 train600 baseline, not all future policies",
            "shared_validator_pool_resident":True,"source":self.m["source"],
            "initialization":"original C0; no diagnostic weights", "teacher_fit_gate":False,
            "created_unix":time.time()}
        atomic_json(self.root / "training_admission.json", receipt, overwrite=False)
        if not passed:
            raise RuntimeError("Batch32 capacity not admitted; preserve global batch and diagnose before training")

    def finish_study(self, hb):
        curves, selected = {}, {}
        for arm in ARMS:
            curves[arm] = []
            for epoch in range(1,9):
                episodes = epoch*960
                values = {s:self.eval_at(arm,episodes,s,hb) for s in ("validation","tune")}
                curves[arm].append({"data_epoch":epoch,"training_episodes":episodes,
                    **{s:v["summary"] for s,v in values.items()}})
            eligible = [r for r in curves[arm] if risk_pass(r["validation"]) and risk_pass(r["tune"])]
            best = min(eligible, key=lambda r:(r["validation"]["makespan"],r["training_episodes"])) if eligible else curves[arm][-1]
            checkpoint = self.root / f"train/to7680/{arm}/rank0/models/episodes_{best['training_episodes']:06d}.pt"
            selected[arm] = {"checkpoint":str(checkpoint), "checkpoint_sha256":digest_file(checkpoint),
                "training_episodes":best["training_episodes"],"risk_qualified":bool(eligible)}
        atomic_json(self.root/"learning_curves.json",curves,overwrite=False)
        atomic_json(self.root/"selection_locked.json",{"protocol_sha256":self.m["manifest_sha256"],
            "selected":selected,"rule":"lowest validation120 mean among validation and Tune risk-qualified epoch checkpoints; else Last marked unqualified",
            "locked_before_confirmation":True,"created_unix":time.time()},overwrite=False)
        hb.update(phase="locked_confirmation")
        self.baselines(hb, confirmation=True)
        requests = [publish(self.m, row["checkpoint"], arm, row["training_episodes"], ARMS[arm]["encoder"], split="confirmation")
                    for arm,row in selected.items()]
        confirmation = dict(zip(ARMS,self.wait_requests(requests,hb)))
        passed = []
        for arm, result in confirmation.items():
            summary = result["summary"]
            chosen_validation = next(r["validation"] for r in curves[arm]
                if r["training_episodes"] == selected[arm]["training_episodes"])
            if (selected[arm]["risk_qualified"] and summary["gain_fraction"] >= .02 and risk_pass(summary)
                    and chosen_validation["gain_fraction"] >= .02
                    and summary["paired_case_bootstrap_gain_ci95"][0] > 0
                    and all(r["validation"]["gain_fraction"] > 0 for r in curves[arm][-2:])):
                passed.append(arm)
        private_comparison = paired_summary(confirmation["C_PRIVATE"]["cases"],
            {r["case_id"]:r["makespan"] for r in confirmation["B_SHARED"]["cases"]})
        matched = {}
        for epoch in range(1,9):
            shared = self.eval_at("B_SHARED",epoch*960,"validation",hb)["cases"]
            private = self.eval_at("C_PRIVATE",epoch*960,"validation",hb)["cases"]
            matched[str(epoch)] = paired_summary(private,{r["case_id"]:r["makespan"] for r in shared})
        atomic_json(self.root/"result.json",{"completed":True,"training_episodes_per_arm":7680,
            "scientific_target_passed":bool(passed),"passed_arms":passed,"selected":selected,
            "confirmation":confirmation,"private_vs_shared":private_comparison,
            "matched_budget_validation_private_vs_shared":matched,
            "single_seed_only":True,"all_artifacts_retained":True,"historical_finalblind_opened":False},overwrite=False)

    def run(self):
        with ProgressHeartbeat(self.root/"status.json",phase="preflight",active_manifest=str(self.manifest_path),
                               formal_training_started=False) as hb:
            try:
                hb.update(workspace_advisory=verify(self.m))
                self.wait_available(hb)
                for i, placement in enumerate(self.plan["validators"]):
                    self.start(f"pool{i}","validator",placement)
                hb.update(phase="full_data_c0_baselines")
                rows = self.baselines(hb)
                self.costs = source_costs(self.m)
                train_names = {c["path"] for c in self.m["splits"]["train_full600"]}
                dense = max((r for r in rows if r["case_id"] in train_names), key=lambda r:r["max_graph_tensor_bytes"])["case_id"]
                self.contracts(hb)
                self.capacity(dense,hb)
                hb.update(phase="full_data_training",formal_training_started=True,target_episodes_per_arm=7680)
                self.run_group("train","to7680",hb,batch_size=32,until=7680)
                self.finish_study(hb)
                if self.queue.pending_count():
                    raise RuntimeError("Cannot close pool with pending evaluations")
                atomic_json(self.root/"validator/STOP",{"reason":"all full-data evaluations drained"},overwrite=False)
                while any(j["phase"]=="validator" and j["process"].poll() is None for j in self.jobs.values()):
                    self.check(hb)
                    time.sleep(2)
                hb.update(phase="complete")
            except BaseException:
                hb.update(event="suite_failed",traceback=traceback.format_exc())
                raise
            finally:
                self.stop_owned()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare"); p.add_argument("--prior",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("prepare-recovery"); p.add_argument("--prior",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    r=sub.add_parser("run"); r.add_argument("--manifest",type=Path,required=True)
    v=sub.add_parser("seal-tests"); v.add_argument("--manifest",type=Path,required=True)
    v.add_argument("--reports",type=Path,nargs="+",required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        prepare(args)
        return
    if args.command=="prepare-recovery":
        from onpolicy.utils.stage3_full_data_restart import prepare as prepare_restart
        prepare_restart(args, ROOT)
        return
    manifest=read_json(args.manifest)
    if args.command=="seal-tests":
        verify(manifest)
        tests=[]
        for path in args.reports:
            tests.extend(ET.parse(path).getroot().iter("testcase"))
        if not tests or any(t.find("failure") is not None or t.find("error") is not None for t in tests):
            raise ValueError("Cannot seal failed/empty test reports")
        passed=[t.attrib.get("name","") for t in tests if t.find("skipped") is None]
        required=("test_real_four_rank_visit_mean", "test_gpu_real_semantic_history_decoder_cache_equivalence[F_SHARED]",
                  "test_gpu_real_semantic_history_decoder_cache_equivalence[F_PRIVATE]")
        if any(name not in passed for name in required):
            raise ValueError("Missing four-rank or full-depth GPU test evidence")
        atomic_json(Path(manifest["root"])/"verification.json",{"passed":True,"passed_tests":len(passed),
            "code_sha256":manifest["code"]["sha256"],"manifest_sha256":manifest["manifest_sha256"],
            "reports":{str(p.resolve()):digest_file(p) for p in args.reports},
            "scope":"CPU contracts, real Gloo and bounded GPU decoder/update equivalence; full-trajectory and capacity admissions still required",
            "created_unix":time.time()},overwrite=False)
        print("Verification sealed",flush=True)
        return
    verification=read_json(Path(manifest["root"])/"verification.json")
    if (not verification["passed"] or verification["code_sha256"]!=manifest["code"]["sha256"]
            or verification["manifest_sha256"]!=manifest["manifest_sha256"]
            or any(digest_file(p)!=s for p,s in verification["reports"].items())):
        raise ValueError("Source-matched test receipt required before launch")
    if (Path(manifest["root"])/"status.json").exists():
        raise FileExistsError("No implicit suite restart")
    with (Path(manifest["root"])/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        FullDataSuite(manifest,args.manifest).run()


if __name__=="__main__":
    main()
