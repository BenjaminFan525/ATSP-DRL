#!/usr/bin/env python3
"""Conditional Stage3: bounded fit, closed-loop gates, four-arm attribution."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.scripts.train.run_stage3_local_improvement import Suite as BaseSuite
from onpolicy.utils.stage3_local_improvement import ARMS, SEED, HISTORY, ENVIRONMENT, identity, verify
from onpolicy.utils.stage3_conditional_improvement import (STUDY, ARCHITECTURES, PAIR_FEATURES,
    metric_rank, learnability_gate, closed_loop_gate, admission_route, aggregate_evaluations)
from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json, case_record, paired_summary
from onpolicy.utils.stage3_representation import external_resources, risk_pass, discover_topology
from onpolicy.utils.stage3_full_policy import resource_plan
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat

OVERLAYS = (
    "onpolicy/runner/shared/stage3_local_improvement_engine.py",
    "onpolicy/utils/stage3_conditional_improvement.py",
    "onpolicy/runner/shared/stage3_conditional_engine.py",
    "onpolicy/scripts/train/stage3_conditional_worker.py",
    "onpolicy/scripts/train/run_stage3_conditional_improvement.py",
    "onpolicy/scripts/train/launch_stage3_conditional_improvement_all.sh",
    "onpolicy/envs/HKBZ/test/test_stage3_conditional_improvement.py",
    "STAGE3_CONDITIONAL_IMPROVEMENT_20260911.md",
)


class BudgetPause(RuntimeError):
    pass


def prepare(args):
    prior_path = args.prior.resolve()
    prior = read_json(prior_path)
    verify(prior, code_root=prior["execution"]["code_root"])
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("New immutable attempt directory required")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total",
                                       "--format=csv,noheader,nounits"], text=True)
    gpus = []
    for row in gpu_rows.splitlines():
        index, uuid, memory = [v.strip() for v in row.split(",")]
        gpus.append({"index": int(index), "uuid": uuid, "memory_mib": int(memory)})
    if [g["index"] for g in gpus] != list(range(8)):
        raise ValueError("Expected all eight GPUs")
    files = {}
    for relative, expected in prior["code"]["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe snapshot path")
        source = Path(prior["execution"]["code_root"])/relative
        if digest_file(source) != expected:
            raise ValueError("Previous frozen execution copy changed")
        target = output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative] = expected
    for relative in OVERLAYS:
        source, target = ROOT/relative, output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest_file(source)
        shutil.copy2(source, target)
        if digest_file(target) != before:
            raise ValueError("Workspace changed during snapshot")
        files[relative] = before
    resume = getattr(args, "resume_from", None)
    previous_root = Path(prior["root"])
    if resume:
        previous = read_json(resume)
        if previous.get("study") != STUDY or previous["code"]["files"] != files:
            raise ValueError("Explicit resume requires identical execution code/protocol; no silent bugfix migration")
        reuse = previous["reuse"]
    else:
        fit = read_json(previous_root/"fit_admission.json")
        reuse = {"manifest": str(prior_path), "baselines": str(previous_root/"baselines.json"),
                 "teachers": str(previous_root/"teachers.json"), "fit_checkpoint": fit["checkpoint"]["checkpoint"],
                 "fit_checkpoint_sha256": fit["checkpoint"]["checkpoint_sha256"],
                 "iga": str(previous_root/"iga_diagnostic.json"),
                 "historical_benchmark": str(previous_root/"fixed_checkpoint_benchmark.json")}
    inputs = {**prior["input_files"], str(prior_path): digest_file(prior_path)}
    for key in ("baselines", "teachers", "fit_checkpoint", "iga", "historical_benchmark"):
        inputs[reuse[key]] = digest_file(reuse[key])
    for entry in read_json(reuse["teachers"])["data"]:
        if digest_file(entry["path"]) != entry["sha256"]:
            raise ValueError("Cached teacher changed")
        inputs[entry["path"]] = entry["sha256"]
    m = copy.deepcopy(prior)
    m.update(study=STUDY, root=str(output), created_unix=time.time(), prior_manifest=str(prior_path),
        reuse=reuse, input_files=inputs, code={"files": files, "sha256": digest_json(files)},
        material_passport={"origin_skill": "academic-research-suite/experiment-agent", "origin_mode": "run",
            "origin_date": "2026-09-11", "verification_status": "prepared; scientific results unverified", "version_label": STUDY})
    m["execution"] = {**prior["execution"], "python": sys.executable, "code_root": str(output/"source"),
                       "workspace_root": str(ROOT), "workspace_code_policy": "advisory; strict frozen execution copy"}
    cores = discover_topology()
    plan = resource_plan(cores)
    for p in plan["validators"]:
        p["cuda_memory_fraction"] = 1.
    m["resources"].update(gpus=gpus, core_groups=cores, plan=plan, validator_shard_cases=6,
        total_card_reserved_limit_gib=22., host_available_min_gib=16.)
    m["fit"] = {"architectures": list(ARCHITECTURES), "learning_rates": [1e-4, 3e-4, 1e-3],
        "max_updates": 2000, "diagnostic_epochs_are_not_formal_epochs": True, "min_plateau_step": 200,
        "plateau_patience": 200, "metric_interval": 50, "weighted_followup_lr": 3e-4,
        "meaningful_epsilon": "max(5 seconds, .001 * reference case cost)",
        "cost_weight_cap": 50., "neutral_kl_coefficient": .1, "max_teacher_refresh_rounds": 3,
        "max_teacher_queries_per_round": 320, "max_new_teacher_queries": 960,
        "diagnostic_weights_forbidden_as_formal_initialization": True}
    m["model"] = {"frozen": ["GNN", "GRU", "plane actor", "original resource actors"],
        "width": 64, "pair_features": list(PAIR_FEATURES), "input_mode": "residual side channel, unchanged base GNN",
        "greedy_decoder": "global argmax over reconstructed full legal distribution; never binary .5 gate"}
    m["baseline_policy"] = "reuse verified same-physics dual baselines after native/annotated canaries; no hard costs"
    m["timeouts_seconds"].update(suite=30*3600, diagnostic_budget=6*3600, formal_budget=24*3600)
    m["confirmation"] = {"seed": 2026091107, "count": 120, "opened_after_candidate_lock_only": True,
        "scope": "new case confirmation conditional on one trained seed, not the historical FinalBlind"}
    m["automatic_confirmation"] = True
    m["training"]["rank_objective"] = "cost-weighted preferences + neutral reference preservation"
    m["training"]["warm_protocol"] = "C0-frozen 120 cases, 3 distinct decision positions each, one alternative per position; cached fit max2000"
    m["training"]["warm_lr"] = 3e-4
    m["training"]["limited_rl_queries"] = 480
    m["selection"] = {"head_selection": "training-cache metrics first; closed-loop diagnostics explicitly adaptive",
        "maximum_tune_endpoints_per_arm": 4, "confirmation_winner_count": 1}
    # New cases are generated under an independent, predeclared namespace, no
    # costs are evaluated or used during generation. Historical FinalBlind is not opened.
    if resume:
        m["splits"] = previous["splits"]
        m["resume_from"] = str(Path(resume).resolve())
        m["exposure_registry"] = previous["exposure_registry"]
    else:
        from onpolicy.envs.HKBZ.data_generator import AirportScenarioGenerator, PROFILES, _allocate_profile_schedule, _derived_seed, _write_case
        schedule = _allocate_profile_schedule("validation", 120, m["confirmation"]["seed"])
        known = {c["case_sha256"] for split in prior["splits"].values() for c in split}
        confirmation = []
        for i, profile in enumerate(schedule):
            name = f"case_{i:04d}"
            seed = _derived_seed(m["confirmation"]["seed"], "stage3_conditional_confirmation", i, profile)
            generator = AirportScenarioGenerator(profile=PROFILES[profile], seed=seed, split="confirmation", case_id=name)
            case, metadata = generator.generate()
            path = output/"confirmation_data"/name
            _write_case(path, case, metadata)
            record = case_record(path)
            if record["case_sha256"] in known:
                raise ValueError("Confirmation duplicates an existing registered case")
            known.add(record["case_sha256"])
            confirmation.append(record)
        m["splits"]["confirmation"] = confirmation
        m["exposure_registry"] = {"historically_exposed": list(prior["splits"]),
            "confirmation_costs_unopened": [c["content_sha256"] for c in confirmation],
            "generation_seed_namespace": "stage3_conditional_confirmation/2026091107",
            "historical_finalblind_opened": False, "training_forbidden": ["train_probe64", "tune", "confirmation"]}
    m["learning_protocol_sha256"] = digest_json({k: m[k] for k in
        ("study", "source", "contract", "history", "code", "training", "fit", "model", "gates")})
    if resume and m["learning_protocol_sha256"] != previous["learning_protocol_sha256"]:
        raise ValueError("Resume learning protocol changed")
    m["manifest_sha256"] = identity(m)
    atomic_json(output/"manifest.json", m, overwrite=False)
    atomic_json(output/"exposure_registry.json", m["exposure_registry"], overwrite=False)
    print(output/"manifest.json", flush=True)


class Suite(BaseSuite):
    def __init__(self, manifest, path):
        super().__init__(manifest, path)
        self.phase_started, self.formal_started = time.time(), False
        self.new_teacher_queries = 0
        self.eval_registry = {}
        self.eval_lock = threading.Lock()
        self.resume_root = None
        if manifest.get("resume_from"):
            self.resume_root = Path(read_json(manifest["resume_from"])["root"])

    def start_workers(self, hb):
        reservations = external_resources(self.m["resources"]["gpus"], os.getpid())
        while reservations:
            hb.update(phase="waiting_external_gpu_jobs", reservations=reservations)
            if time.time()-self.started > self.m["timeouts_seconds"]["suite"]:
                raise TimeoutError("Waiting for resources exhausted hard timeout")
            time.sleep(10)
            reservations = external_resources(self.m["resources"]["gpus"], os.getpid(), reservations)
        plan = self.m["resources"]["plan"]
        os.sched_setaffinity(0, plan["controller"])
        placements = [(f"gpu{i}", "work", p, self.m["resources"]["worker_width"]) for i, p in plan["trainers"].items()]
        placements += [(f"validator{i}", "validator", p, self.m["resources"]["validator_width"]) for i, p in enumerate(plan["validators"])]
        for name, queue, placement, width in placements:
            gpu = self.m["resources"]["gpus"][placement["gpu"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONPATH=str(ROOT),
                HKBZ_STAGE3_WORKSPACE_ROOT=self.m["execution"]["workspace_root"], OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MALLOC_ARENA_MAX="2",
                PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            command = ["taskset", "--cpu-list", ",".join(map(str, placement["cpus"])), self.m["execution"]["python"],
                "-B", "-u", str(ROOT/"onpolicy/scripts/train/stage3_conditional_worker.py"), "serve", "--manifest", str(self.path),
                "--output", str(self.root/"workers"/name), "--queue", queue, "--width", str(width), "--worker-id", name]
            atomic_json(self.root/"commands"/f"{name}.json", {"command": command, "placement": placement}, overwrite=False)
            log_path = self.root/"logs"/f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("x") as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.jobs.append({"name": name, "process": proc, "queue": queue})
        self.phase_started = time.time()
        atomic_json(self.root/"resource_plan.json", self.m["resources"], overwrite=False)

    def health(self):
        super().health()
        budget = self.m["timeouts_seconds"]["formal_budget" if self.formal_started else "diagnostic_budget"]
        if time.time()-self.phase_started > budget:
            raise BudgetPause("Registered phase wall-clock budget exhausted; not a scientific failure")

    def submit(self, label, payload, *, validator=False):
        if not validator and payload["kind"] in ("cached_fit", "formal_warm_fit"):
            # Cheap sufficient-fitting checks must not sit behind an entire
            # wave of long causal rollouts. Parent-checkpoint barriers remain.
            with self.lock:
                self.counter += 1
                key = f"000000_fit{self.counter:06d}_{label}"
            self.work.submit(key, payload)
            return key
        return super().submit(label, payload, validator=validator)

    def evaluate(self, checkpoint, cases, label):
        from threading import Lock
        key = digest_json({"checkpoint": checkpoint["checkpoint_sha256"], "cases": [c["content_sha256"] for c in cases],
                           "protocol": self.m["learning_protocol_sha256"], "decoder": "greedy"})
        with self.eval_lock:
            if key in self.eval_registry:
                return self.eval_registry[key]
            size = self.m["resources"]["validator_shard_cases"]
            shards = []
            for first in range(0, len(cases), size):
                part = cases[first:first+size]
                shards.append(self.submit(f"{label}_s{first:03d}", self.checkpoint_task(checkpoint, kind="evaluate", cases=part,
                    source_costs={c["path"]: self.source_costs[c["path"]] for c in part},
                    legacy_source_costs={c["path"]: self.legacy_source_costs[c["path"]] for c in part}), validator=True))
            descriptor = {"evaluation_group": key, "shards": shards, "cases": cases,
                          "path": str(self.root/"evaluations"/f"{key}.json")}
            atomic_json(self.root/"evaluation_requests"/f"{key}.json", descriptor, overwrite=False)
            self.eval_registry[key] = descriptor
            return descriptor

    def wait(self, keys, *, validator=False):
        if not any(isinstance(k, dict) for k in keys):
            return super().wait(keys, validator=validator)
        results = []
        for descriptor in keys:
            if not isinstance(descriptor, dict):
                results.extend(super().wait([descriptor], validator=validator))
                continue
            path = Path(descriptor["path"])
            if path.exists():
                results.append(read_json(path))
                continue
            parts = super().wait(descriptor["shards"], validator=True)
            result = aggregate_evaluations(parts, descriptor["cases"], self.source_costs, self.legacy_source_costs)
            # An evaluation can be awaited by both prewarm and control readers.
            with self.eval_lock:
                if not path.exists():
                    atomic_json(path, result, overwrite=False)
            results.append(result)
        return results

    def phase0(self, hb):
        hb.update(phase="P0_implementation_and_cache_contract", formal_training_started=False)
        initial = self.wait([self.submit("initial_score", {"kind": "initialize_model", "architecture": "score"})])[0]
        atomic_json(self.root/"initial_checkpoint.json", initial, overwrite=False)
        warmups = [self.submit(f"validator_warmup_{i}", self.checkpoint_task(initial, kind="warmup",
                    case=self.m["splits"]["train_fit16"][i], worker=f"validator{i}"), validator=True) for i in range(2)]
        self.wait(warmups, validator=True)
        architectures = ("score", "score", "score", "score", "conditional", "conditional", "conditional_pair", "conditional_pair")
        ids = [self.submit(f"canary_gpu{i}", {"kind": "canary", "case": self.m["splits"]["train_fit16"][i],
                "worker": f"gpu{i}", "architecture": architectures[i]}) for i in range(8)]
        results = self.wait(ids)
        if any(not r["passed"] for r in results):
            raise RuntimeError("Implementation canary failed")
        baseline = read_json(self.m["reuse"]["baselines"])
        self.source_costs, self.legacy_source_costs = baseline["source_costs"], baseline["legacy_source_costs"]
        cases = self.m["splits"]["train_fit16"][:4]
        ids = [self.submit(f"cache_replay_{i}", self.checkpoint_task(initial, kind="baseline", cases=[case])) for i, case in enumerate(cases)]
        checks = self.wait(ids)
        for batch in checks:
            for row in batch["cases"]:
                for mode, costs in (("semantic", self.source_costs), ("legacy", self.legacy_source_costs)):
                    if abs(row[mode]["makespan"]-costs[row[mode]["case_id"]]) > 1e-6:
                        raise ValueError("Baseline cache failed exact annotated/native environment replay")
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        card_memory = {int(row.split(",")[0]): float(row.split(",")[1])/1024 for row in memory.splitlines()}
        available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
        if max(card_memory.values()) > 22 or self.sampled_peak_bytes > 96*2**30 or available < 16*2**30:
            raise RuntimeError("Combined-card/host capacity admission failed")
        atomic_json(self.root/"baselines.json", {**baseline, "reuse_origin": self.m["reuse"]["baselines"],
                    "exact_replayed_cases": len(cases)}, overwrite=False)
        atomic_json(self.root/"technical_admission.json", {"passed": True, "canaries": results,
                    "baseline_cache_replays": checks, "combined_gpu_memory_gib": card_memory,
                    "all_gpu_count": 8, "validator_pool_count": 1, "validator_consumers": 2}, overwrite=False)
        return initial

    def causal_ablation(self, hb):
        hb.update(phase="P1_wait_causal_ablation")
        old = self.wait([self.submit("import_previous_fit", {"kind": "import_previous_fit",
            "previous_checkpoint": self.m["reuse"]["fit_checkpoint"], "previous_sha256": self.m["reuse"]["fit_checkpoint_sha256"]})])[0]
        cases, ids = self.m["splits"]["train_diag32"], {}
        for mode in ("full", "wait_only", "ranking_only"):
            ids[mode] = [self.submit(f"ablation_{mode}_{i:02d}", self.checkpoint_task(old, kind="ablation", cases=cases[i:i+4],
                         ablation=mode)) for i in range(0, len(cases), 4)]
        result = {mode: {"cases": [r for part in self.wait(keys) for r in part["cases"]]} for mode, keys in ids.items()}
        for value in result.values():
            value["summary"] = paired_summary(value["cases"], self.source_costs)
        result["C0"] = {"case_costs": {c["path"]: self.source_costs[c["path"]] for c in cases}, "reused": True}
        atomic_json(self.root/"causal_wait_ablation.json", {"results": result, "new_terminal_queries": 96,
            "scope": "component intervention; not proof the first divergence alone caused total cost"}, overwrite=False)

    def cached_fits(self, initial, teachers, hb):
        hb.update(phase="P1_bounded_cached_fitting")
        conditional = self.wait([self.submit("initial_conditional", {"kind": "initialize_model", "architecture": "conditional"})])[0]
        initial_by_arch = {"score": initial, "conditional": conditional}
        ids = []
        for arch, checkpoint in initial_by_arch.items():
            for lr in self.m["fit"]["learning_rates"]:
                ids.append(self.submit(f"fit_{arch}_{lr:g}", self.checkpoint_task(checkpoint, kind="cached_fit", data=teachers,
                    lr=lr, max_updates=self.m["fit"]["max_updates"], weighted=False)))
        lookup_id = self.submit("lookup_fit", self.checkpoint_task(initial, kind="cached_fit", data=teachers,
                    lr=.03, max_updates=2000, lookup=True))
        fits, lookup = self.wait(ids), self.wait([lookup_id])[0]
        if not learnability_gate(lookup["fit"]["metrics"])["passed"]:
            atomic_json(self.root/"lookup_failure.json", lookup, overwrite=False)
            raise RuntimeError("Independent logits failed fixed-label computational check")
        # One registered loss ablation at the same LR as the train-cache winner.
        best = max(fits, key=lambda r: metric_rank(r["fit"]["metrics"]))
        weighted_id = self.submit("weighted_fit", self.checkpoint_task(initial_by_arch[best["architecture"]], kind="cached_fit",
            data=teachers, lr=best["fit"]["lr"], max_updates=2000, weighted=True))
        fits.extend(self.wait([weighted_id]))
        for fit in fits:
            fit["learnability_gate"] = learnability_gate(fit["fit"]["metrics"], fit["single"]["metrics"])
        atomic_json(self.root/"fit_convergence.json", {"fits": fits, "lookup": lookup,
            "optimizer_budget_is_not_simulator_budget": True}, overwrite=False)
        return fits, initial_by_arch

    def closed_check(self, fit, teachers, label):
        fit_key = self.evaluate(fit, self.m["splits"]["train_fit16"], label+"_fit16")
        probe_key = self.evaluate(fit, self.m["splits"]["train_probe64"], label+"_probe64")
        fit_eval, probe_eval = self.wait([fit_key, probe_key], validator=True)
        gate = closed_loop_gate(fit_eval, probe_eval, self.source_costs, {r["case_id"]: r["teacher_cost"] for r in teachers})
        return {"fit": fit_eval, "probe": probe_eval, "gate": gate, "checkpoint": fit}

    def diagnose(self, initial, hb):
        teachers = read_json(self.m["reuse"]["teachers"])["data"]
        atomic_json(self.root/"teachers.json", {"data": teachers, "reused": True, "diagnostic_only": True}, overwrite=False)
        # Independent work proceeds together on the shared pool; no duplicated
        # large historical checkpoint benchmark precedes these cheap fits.
        with ThreadPoolExecutor(max_workers=2) as pool:
            causal = pool.submit(self.causal_ablation, hb)
            fits, initials = self.cached_fits(initial, teachers, hb)
            causal.result()
        eligible = [r for r in fits if r["learnability_gate"]["passed"]]
        selected = max(eligible or fits, key=lambda r: metric_rank(r["fit"]["metrics"]))
        checks = []
        if eligible:
            hb.update(phase="P2_closed_loop_check", selected_architecture=selected["architecture"])
            checks.append(self.closed_check(selected, teachers, "closed0"))
        # Missing conditional features are tested only after the original /
        # same-information conditional heads have had a real fit budget.
        for round_index in range(1, 4):
            if eligible and checks and checks[-1]["gate"]["passed"]:
                break
            hb.update(phase=f"P2_teacher_refresh_{round_index}", new_teacher_queries=self.new_teacher_queries)
            if round_index == 1:
                pair_initial = self.wait([self.submit("initial_pair", {"kind": "initialize_model", "architecture": "conditional_pair"})])[0]
                initials["conditional_pair"] = pair_initial
                reference_complete = bool(checks and checks[-1]["fit"]["summary"].get("metric_valid", True))
                reference = selected if eligible and reference_complete else initial
                start = pair_initial
            else:
                reference, start = selected, selected
            ids = [self.submit(f"refresh{round_index}_{i:02d}", self.checkpoint_task(reference, kind="branch", case=c,
                   mode="diagnostic", seed=SEED+round_index*100+i, states=2, candidates=8))
                   for i, c in enumerate(self.m["splits"]["train_fit16"])]
            fresh = self.wait(ids)
            queries = sum(r["query_count"] for r in fresh)
            # 16*(1+2*7)=240 max, within the registered 320/960 budgets.
            self.new_teacher_queries += queries
            if queries > 320 or self.new_teacher_queries > 960:
                raise ValueError("Teacher query budget exceeded")
            atomic_json(self.root/f"teachers_refresh_{round_index}.json", {"data": fresh, "queries": queries,
                "reference": reference, "old_reference_labels_not_pooled_as_fresh": True}, overwrite=False)
            fit = self.wait([self.submit(f"refit{round_index}", self.checkpoint_task(start, kind="cached_fit", data=fresh,
                lr=3e-4, max_updates=2000, weighted=True))])[0]
            fit["learnability_gate"] = learnability_gate(fit["fit"]["metrics"], fit["single"]["metrics"])
            atomic_json(self.root/f"refit_{round_index}.json", fit, overwrite=False)
            if fit["learnability_gate"]["passed"]:
                selected, teachers, eligible = fit, fresh, [fit]
                checks.append(self.closed_check(fit, fresh, f"closed{round_index}"))
            elif not eligible:
                selected = fit
        learning_ok = bool(eligible)
        closed_ok = bool(checks and checks[-1]["gate"]["passed"])
        route = admission_route(True, learning_ok, closed_ok)
        result = {"route": route, "learnability_passed": learning_ok, "safe_diagnostic_warm": closed_ok,
            "selected": selected, "initial": initials[selected["architecture"]], "closed_loop_checks": checks,
            "new_teacher_queries": self.new_teacher_queries, "formal_training_started": False,
            "diagnostic_weights_forbidden_as_formal_initialization": True}
        atomic_json(self.root/"closed_loop_admission.json", result, overwrite=False)
        return result

    def resumed_state(self, arm, blank):
        if self.resume_root is None:
            return copy.deepcopy(blank)
        path = self.resume_root/"train"/arm/"latest_commit.json"
        if not path.exists():
            return copy.deepcopy(blank)
        state = read_json(path)
        prior_evaluations = state.get("evaluations", {})
        state["evaluations"] = {}
        for endpoint in (480, 960, 1440, 1920):
            key = str(endpoint)
            if key in prior_evaluations and Path(prior_evaluations[key]["path"]).exists():
                state["evaluations"][key] = prior_evaluations[key]
        if state["queries"] in (480, 960, 1440, 1920) and str(state["queries"]) not in state["evaluations"]:
            state["evaluations"][str(state["queries"])] = self.evaluate(state["checkpoint"], self.m["splits"]["tune"], f"resume_{arm}")
        return state

    def train_to(self, arm, state, until, *, warm=False):
        if not warm:
            return super().train_to(arm, state, until, warm=False)
        if until != 480 or state["queries"] not in (0, 480):
            raise ValueError("Formal warm is one atomic complete-corpus fit")
        if state["queries"] == 480:
            return state
        initial = state["checkpoint"]
        cases = sorted(self.m["splits"]["train_pilot120"], key=lambda c: digest_json([SEED, "warm", c["content_sha256"]]))
        ids = [self.submit(f"formal_warm_c{i:03d}", self.checkpoint_task(initial, kind="collect", mode="warm", case=case,
            seed=SEED+i, states=3, candidates=2)) for i, case in enumerate(cases)]
        data = self.wait(ids)
        if len(data) != 120 or sum(r["query_count"] for r in data) != 480:
            raise ValueError("Formal warm simulation ledger mismatch")
        atomic_json(self.root/"formal_warm_data.json", {"data": data, "initial": initial,
            "physical_queries": 480, "reference_frozen_for_entire_corpus": True}, overwrite=False)
        checkpoint = self.wait([self.submit("formal_warm_fit", self.checkpoint_task(initial, kind="formal_warm_fit",
            data=data, lr=self.m["training"]["warm_lr"]))])[0]
        evaluation = self.evaluate(checkpoint, self.m["splits"]["tune"], "PRE_eval_q0480")
        result = {"checkpoint": checkpoint, "queries": 480, "cursor": 120, "evaluations": {"480": evaluation}}
        atomic_json(self.root/"train"/"PRE"/"commit_q0480.json", result, overwrite=False)
        atomic_json(self.root/"train"/"PRE"/"latest_commit.json", result)
        return result

    def training(self, initial, hb, *, limited=False):
        self.formal_started, self.phase_started = True, time.time()
        hb.update(phase="P3_formal_480", formal_training_started=True, limited_c0_rl=limited)
        atomic_json(self.root/"formal_plan.json", {"initial": initial, "limited": limited,
            "protocol": self.m["learning_protocol_sha256"], "diagnostic_weights_reused": False}, overwrite=False)
        blank = {"checkpoint": initial, "queries": 0, "cursor": 0, "evaluations": {}}
        states = {a: self.resumed_state(a, blank) for a in ("T_PPO", "L_RL")}
        warm_state = self.resumed_state("PRE", blank)
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {a: pool.submit(self.train_to, a, s, 480) for a, s in states.items()}
            warm_job = None if limited else pool.submit(self.train_to, "PRE", warm_state, 480, warm=True)
            states.update({a: job.result() for a, job in jobs.items()})
            warm = warm_job.result() if warm_job else None
        first = {a: self.wait([s["evaluations"]["480"]], validator=True)[0] for a, s in states.items()}
        if limited:
            return {"status": "limited_c0_rl_480_complete_review_required", "screen480": first,
                    "formal_training_started": True, "pilot_passed": [], "target": {}}
        warm_eval = self.wait([warm["evaluations"]["480"]], validator=True)[0]
        warm_probe = self.wait([self.evaluate(warm["checkpoint"], self.m["splits"]["train_probe64"], "formal_warm_probe")], validator=True)[0]
        warm_safe = (warm_eval["summary"]["gain_fraction"] >= 0 and risk_pass(warm_eval["summary"])
                     and warm_probe["summary"]["gain_fraction"] >= 0 and risk_pass(warm_probe["summary"]))
        atomic_json(self.root/"prewarm.json", {"state": warm, "evaluation": warm_eval, "probe": warm_probe,
            "safe_for_rl_initialization": warm_safe, "physical_queries_shared_once": 480,
            "standalone_queries_charged_each_arm": 480}, overwrite=False)
        # Unsafe warm weights never initialize the mixed RL arm. Ranking can
        # remain a clearly labelled supervised control, not a safe candidate.
        states["L_RANK"] = self.resumed_state("L_RANK", warm)
        if warm_safe:
            states["L_RANK_RL"] = self.resumed_state("L_RANK_RL", warm)
        else:
            atomic_json(self.root/"mixed_arm_not_admitted.json", {"reason": "formal warm failed safety", "warm": warm}, overwrite=False)
        active = [a for a in states if a == "L_RANK" or a == "L_RANK_RL" or risk_pass(first[a]["summary"])]
        hb.update(phase="P3_formal_960", active_arms=active)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {a: pool.submit(self.train_to, a, states[a], 960) for a in active}
            states.update({a: job.result() for a, job in jobs.items()})
        evaluations = {a: self.wait([states[a]["evaluations"]["960"]], validator=True)[0] for a in active}
        chosen = sorted([a for a, r in evaluations.items() if r["summary"]["gain_fraction"] >= .005 and risk_pass(r["summary"])],
                        key=lambda a: (-evaluations[a]["summary"]["gain_fraction"], a))[:2]
        atomic_json(self.root/"screen_admission.json", {"selected": chosen, "results": evaluations,
            "stopped_at480": [a for a in first if a not in active]}, overwrite=False)
        endpoints = {}
        for until in (1440, 1920):
            if not chosen:
                break
            hb.update(phase=f"P3_formal_{until}", selected=chosen)
            running = list(chosen)
            if "L_RANK_RL" in chosen and "L_RANK" not in running:
                running.append("L_RANK")
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = {a: pool.submit(self.train_to, a, states[a], until) for a in running}
                states.update({a: job.result() for a, job in jobs.items()})
            rows = {a: self.wait([states[a]["evaluations"][str(until)]], validator=True)[0] for a in running}
            endpoints[str(until)] = rows
            chosen = [a for a in chosen if rows[a]["summary"]["gain_fraction"] >= .01 and risk_pass(rows[a]["summary"])]
            atomic_json(self.root/f"admission_{until}.json", {"selected": chosen, "evaluations": rows}, overwrite=False)
        target = {}
        for arm, row in endpoints.get("1920", {}).items():
            summary = row["summary"]
            passed = (arm in chosen and arm != "L_RANK" and summary["gain_fraction"] >= .02
                      and row["legacy_summary"]["gain_fraction"] >= .02 and risk_pass(summary))
            extra = {}
            if arm == "L_RANK_RL":
                extra["post_warm_rl_gain"] = 1-summary["makespan"]/warm_eval["summary"]["makespan"]
                control = endpoints["1920"].get("L_RANK")
                extra["rank_only_comparison"] = paired_summary(row["cases"], {r["case_id"]: r["makespan"] for r in control["cases"]}) if control else None
                passed = passed and extra["post_warm_rl_gain"] >= .01 and control is not None and extra["rank_only_comparison"]["gain_fraction"] > 0
            target[arm] = {"practical_rl_target_passed": bool(passed), "total_gain": summary["gain_fraction"],
                           "gain_vs_legacy_c0": row["legacy_summary"]["gain_fraction"], **extra}
        result = {"status": "formal_pilot_complete", "formal_training_started": True, "screen480": first,
            "screen": evaluations, "endpoints": endpoints, "pilot_passed": chosen, "target": target,
            "single_training_seed": True, "historical_finalblind_opened": False}
        eligible = [a for a in target if target[a]["practical_rl_target_passed"]]
        if eligible:
            winner = max(eligible, key=lambda a: target[a]["total_gain"])
            result["confirmation"] = self.confirm(winner, states, initial, warm, hb)
        else:
            result["confirmation"] = {"opened": False, "reason": "no candidate met registered Tune practical target"}
        return result

    def confirm(self, winner, states, initial, warm, hb):
        hb.update(phase="P4_frozen_candidate_confirmation", winner=winner)
        candidates = {"CANDIDATE": states[winner]["checkpoint"], "WARM": warm["checkpoint"]}
        if winner == "L_RANK_RL":
            candidates["RANK_CONTROL"] = states["L_RANK"]["checkpoint"]
        atomic_json(self.root/"candidate_lock.json", {"winner": winner, "checkpoints": candidates,
            "no_further_model_selection_on_confirmation": True}, overwrite=False)
        cases = self.m["splits"]["confirmation"]
        ids = [self.submit(f"confirm_baseline_{i:03d}", self.checkpoint_task(initial, kind="baseline", cases=cases[i:i+4]))
               for i in range(0, len(cases), 4)]
        baselines = [r for part in self.wait(ids) for r in part["cases"]]
        self.source_costs.update({r["semantic"]["case_id"]: r["semantic"]["makespan"] for r in baselines})
        self.legacy_source_costs.update({r["legacy"]["case_id"]: r["legacy"]["makespan"] for r in baselines})
        refs = {name: self.evaluate(checkpoint, cases, "confirm_"+name) for name, checkpoint in candidates.items()}
        evaluations = {name: self.wait([ref], validator=True)[0] for name, ref in refs.items()}
        candidate = evaluations["CANDIDATE"]
        passed = (candidate["summary"]["gain_fraction"] >= .02 and candidate["legacy_summary"]["gain_fraction"] >= .02
            and risk_pass(candidate["summary"]) and candidate["summary"]["paired_case_bootstrap_gain_ci95"][0] > 0
            and candidate["legacy_summary"]["paired_case_bootstrap_gain_ci95"][0] > 0)
        increment = {}
        if winner == "L_RANK_RL":
            for name in ("WARM", "RANK_CONTROL"):
                increment[name] = paired_summary(candidate["cases"], {r["case_id"]: r["makespan"] for r in evaluations[name]["cases"]})
            passed = (passed and increment["WARM"]["gain_fraction"] >= .01 and increment["WARM"]["paired_case_bootstrap_gain_ci95"][0] > 0
                      and increment["RANK_CONTROL"]["paired_case_bootstrap_gain_ci95"][0] > 0)
        result = {"opened": True, "passed": bool(passed), "baselines": baselines, "evaluations": evaluations,
            "rl_increment": increment, "scope": "single-seed case confirmation; not cross-seed robustness"}
        atomic_json(self.root/"confirmation_result.json", result, overwrite=False)
        # Only now pay for an additional small sampling/IGA-matched comparison.
        benchmark_ids = {}
        for label, checkpoint in (("C0", initial), (winner, states[winner]["checkpoint"])):
            benchmark_ids[label] = [self.submit(f"candidate_benchmark_{label}_{i}", self.checkpoint_task(checkpoint,
                kind="benchmark_candidate", case=case)) for i, case in enumerate(self.m["iga"]["cases"])]
        atomic_json(self.root/"candidate_iga_comparison.json", {"benchmarks": {a: self.wait(ids) for a, ids in benchmark_ids.items()},
            "iga_reuse": self.m["reuse"]["iga"], "same_cases": self.m["iga"]["cases"],
            "scope": "same H2/F4/soft eight diagnostic cases; not confirmation-set IGA evidence"}, overwrite=False)
        return result

    def run(self):
        with ProgressHeartbeat(self.root/"status.json", phase="preflight", active_manifest=str(self.path)) as hb:
            try:
                hb.update(workspace_advisory=verify(self.m, code_root=ROOT))
                self.start_workers(hb)
                initial = self.phase0(hb)
                if self.resume_root:
                    plan = read_json(self.resume_root/"formal_plan.json")
                    initial, limited = plan["initial"], plan["limited"]
                    atomic_json(self.root/"resume_audit.json", {"previous_root": str(self.resume_root),
                        "protocol_unchanged": True, "only_committed_groups_resumed": True,
                        "uncommitted_previous_queries_not_erased": True}, overwrite=False)
                    result = self.training(initial, hb, limited=limited)
                else:
                    admission = self.diagnose(initial, hb)
                    if admission["route"] == "learnability_budget_exhausted":
                        result = {"status": "learnability_budget_exhausted", "formal_training_started": False,
                                  "admission": admission, "scientific_target_passed": False}
                    else:
                        atomic_json(self.root/"training_admission.json", admission, overwrite=False)
                        result = self.training(admission["initial"], hb, limited=admission["route"] == "c0_rl_only_480")
                if self.work.outstanding() or self.validator.outstanding():
                    raise RuntimeError("Cannot finish with undrained tasks")
                result.update(completed=True, all_artifacts_retained=True)
                result["scientific_target_passed"] = bool(result.get("confirmation", {}).get("passed", False))
                atomic_json(self.root/"result.json", result, overwrite=False)
                for queue in (self.work, self.validator):
                    atomic_json(queue.root/"STOP.json", {"reason": "registered stages complete"}, overwrite=False)
                for row in self.jobs:
                    row["process"].wait(timeout=60)
                hb.update(phase="complete", result_status=result["status"])
            except BudgetPause as exc:
                atomic_json(self.root/"result.json", {"completed": False, "status": "budget_exhausted_review_required",
                    "formal_training_started": self.formal_started, "reason": str(exc), "all_artifacts_retained": True}, overwrite=False)
                hb.update(phase="budget_pause", reason=str(exc), scientific_failure=False)
            except BaseException:
                hb.update(event="suite_failed", traceback=traceback.format_exc())
                raise
            finally:
                self.stop_owned()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--prior", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume-from", type=Path)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    m = read_json(args.manifest)
    if m.get("study") != STUDY:
        raise ValueError("Wrong study manifest")
    if (Path(m["root"])/"status.json").exists():
        raise FileExistsError("Use an explicit new attempt; no automatic retry")
    with (Path(m["root"])/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        Suite(m, args.manifest).run()


if __name__ == "__main__":
    main()
