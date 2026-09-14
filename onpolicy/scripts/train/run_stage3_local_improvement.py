#!/usr/bin/env python3
"""Gated H2/F4/soft local-policy study; eight workers, one validator pool."""
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

from onpolicy.utils.stage3_local_improvement import (ARMS, ENVIRONMENT, HISTORY, SCHEMA, SEED, TaskQueue,
                                                    identity, fit_gate, verify)
from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json, read_json, verify_cases, BASE_MANIFEST
from onpolicy.utils.stage3_representation import discover_topology, external_resources, risk_pass
from onpolicy.utils.stage3_full_policy import resource_plan
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat

OVERLAYS = (
    "onpolicy/utils/stage3_local_improvement.py",
    "onpolicy/runner/shared/stage3_local_improvement_engine.py",
    "onpolicy/scripts/train/stage3_local_improvement_worker.py",
    "onpolicy/scripts/train/run_stage3_local_improvement.py",
    "onpolicy/scripts/train/launch_stage3_local_improvement_all.sh",
    "onpolicy/envs/HKBZ/test/test_stage3_local_improvement.py",
    "STAGE3_LOCAL_IMPROVEMENT_20260910.md",
)


def prepare(args):
    prior = read_json(args.prior)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new suite directory; no overwrite/retry")
    for key, cases in prior["splits"].items():
        verify_cases(cases, training=key.startswith("train"))
    if digest_file(prior["source"]["path"]) != prior["source"]["sha256"]:
        raise ValueError("C0 changed")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total",
                                       "--format=csv,noheader,nounits"], text=True)
    gpus = []
    for row in gpu_rows.splitlines():
        index, uuid, memory = [v.strip() for v in row.split(",")]
        gpus.append({"index": int(index), "uuid": uuid, "memory_mib": int(memory)})
    if [g["index"] for g in gpus] != list(range(8)):
        raise ValueError("Expected all eight GPUs")
    cores = discover_topology()
    plan = resource_plan(cores)
    for placement in plan["validators"]:
        placement["cuda_memory_fraction"] = 1.
    inputs = {str(args.prior.resolve()): digest_file(args.prior), str(BASE_MANIFEST): digest_file(BASE_MANIFEST),
              prior["source"]["path"]: prior["source"]["sha256"]}
    references = {"C0": {"name": "C0", "path": prior["source"]["path"], "variant": None,
                          "sha256": prior["source"]["sha256"]}}
    for arm, variant in (("B_SHARED", "F_SHARED"), ("C_PRIVATE", "F_PRIVATE")):
        path = Path(prior["root"])/"train/to960"/arm/"rank0/models/episodes_000960.pt"
        sha = digest_file(path)
        inputs[str(path)] = sha
        references[arm] = {"name": arm, "path": str(path), "sha256": sha, "variant": variant}
    files = {}
    for relative, expected in prior["code"]["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe snapshot path")
        source = Path(prior["execution"]["code_root"])/relative
        if digest_file(source) != expected:
            raise ValueError(f"Historical snapshot changed: {relative}")
        target = output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative] = expected
    for relative in OVERLAYS:
        source, target = ROOT/relative, output/"source"/relative
        expected = digest_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest_file(target) != expected:
            raise ValueError("Workspace changed while snapshotting")
        files[relative] = expected
    contract = {**prior["contract"], **ENVIRONMENT}
    manifest = {"schema": SCHEMA, "root": str(output), "created_unix": time.time(),
        "material_passport": {"origin_skill": "academic-research-suite/experiment-agent", "origin_mode": "run",
            "origin_date": "2026-09-10", "verification_status": "prepared; outcomes unverified", "version_label": SCHEMA},
        "prior_manifest": str(args.prior.resolve()), "source": {"path": prior["source"]["path"],
            "sha256": prior["source"]["sha256"], "legacy_baseline_not_for_evaluation": prior["source"].get("baseline")},
        "splits": prior["splits"],
        "contract": contract, "contract_sha256": digest_json(contract), "history": HISTORY,
        "input_files": inputs, "benchmark_checkpoints": references,
        "code": {"files": files, "sha256": digest_json(files)},
        "execution": {"python": sys.executable, "code_root": str(output/"source"), "workspace_root": str(ROOT),
            "workspace_code_policy": "advisory; do not affect parallel baselines",
            "packages": {n: importlib.metadata.version(n) for n in prior["execution"]["packages"]}},
        "resources": {"gpus": gpus, "core_groups": cores, "plan": plan, "worker_width": 7,
            "validator_width": 3, "validator_workers": 2, "memory_high_gib": 96, "memory_max_gib": 108,
            "queue_scope": "one shared asynchronous validation queue for all GPUs/checkpoints/arms"},
        "training": {"seed": SEED, "seeds": [SEED], "arms": list(ARMS), "lr": 1e-4,
            "ppo_epochs": 2, "case_batch": 4, "samples_per_local_state": 4, "ppo_samples_per_case": 5,
            "warm_queries": 480, "endpoints": [480, 960, 1440, 1920], "max_queries_per_arm": 1920,
            "budget_unit": "completed terminal simulator evaluations; baseline and forced branches included",
            "actor_reduction": "case mean -> sample mean; full PPO sums decisions, local RL one factor",
            "case_schedule": "seeded outcome-independent permutations, complete 4-case groups",
            "reward_coef": .01, "ppo_clip": .2, "grad_clip": 1., "soft_kl": .02, "hard_kl": .04,
            "local_objective": "one-factor contextual surrogate with frozen greedy continuation, not full-MDP unbiased PPO"},
        "fit": {"passes": 10, "cases_per_update": 4, "epochs": 2, "lr": 1e-4,
            "states_per_case": 8, "candidates_per_state": 8, "single_case_updates": 16,
            "scope": "diagnostic weights forbidden as formal RL initialization"},
        "gates": {"fit_cases_gt1pct": 8, "fit_recovery": .5, "fit_top1": .8, "screen_gain": .005,
            "pilot_gain": .01, "target_gain": .02, "post_warm_rl_gain": .01,
            "stress_regression": .005, "tail_regression": .01, "badcase_fraction": .05,
            "max_surviving_arms": 2},
        "baseline_policy": "recompute C0_legacy and C0_semantic on H2/F4/soft; never use H3/hard costs as denominator",
        "timeouts_seconds": {"suite": 864000, "task": 43200, "validator": 21600, "progress_warning": 900},
        "automatic_finalblind": False, "automatic_multi_seed": False, "automatic_extension": False,
        "exposure": "existing Train/Tune exposed historically; no Tune gradients/teachers; FinalBlind unopened",
        "iga": {"cases": prior["splits"]["train_diag32"][:8], "cold_budgets_seconds": [180, 1800],
                "population": 20, "scope": "same H2/F4/soft, training diagnostic only; no warm-start claim"}}
    manifest["manifest_sha256"] = identity(manifest)
    atomic_json(output/"manifest.json", manifest, overwrite=False)
    print(output/"manifest.json", flush=True)


class Suite:
    def __init__(self, manifest, path):
        self.m, self.path = manifest, Path(path).resolve()
        self.root = Path(manifest["root"])
        self.work, self.validator = TaskQueue(self.root/"work"), TaskQueue(self.root/"validator")
        self.jobs, self.counter, self.lock = [], 0, threading.Lock()
        self.started = time.time()
        self.source_costs = {}
        self.legacy_source_costs = {}
        self.last_sample = 0.
        self.sample_lock = threading.Lock()
        self.sampled_peak_bytes = 0
        self.last_source_check = 0.

    def start_workers(self, hb):
        reservations = external_resources(self.m["resources"]["gpus"], os.getpid())
        while reservations:
            hb.update(phase="waiting_external_gpu_jobs", reservations=reservations)
            if time.time()-self.started > self.m["timeouts_seconds"]["suite"]:
                raise TimeoutError("Waiting for GPUs exhausted the suite timeout")
            time.sleep(10)
            reservations = external_resources(self.m["resources"]["gpus"], os.getpid(), reservations)
        plan = self.m["resources"]["plan"]
        os.sched_setaffinity(0, plan["controller"])
        placements = [(f"gpu{i}", "work", p, self.m["resources"]["worker_width"])
                      for i,p in plan["trainers"].items()]
        placements += [(f"validator{i}", "validator", p, self.m["resources"]["validator_width"])
                       for i,p in enumerate(plan["validators"])]
        for name, queue, placement, width in placements:
            gpu = self.m["resources"]["gpus"][placement["gpu"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONPATH=str(ROOT),
                HKBZ_STAGE3_WORKSPACE_ROOT=self.m["execution"]["workspace_root"], OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MALLOC_ARENA_MAX="2",
                PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            command = ["taskset", "--cpu-list", ",".join(map(str, placement["cpus"])), self.m["execution"]["python"],
                "-B", "-u", str(ROOT/"onpolicy/scripts/train/stage3_local_improvement_worker.py"), "serve",
                "--manifest", str(self.path), "--output", str(self.root/"workers"/name),
                "--queue", queue, "--width", str(width), "--worker-id", name]
            atomic_json(self.root/"commands"/f"{name}.json", {"command": command, "placement": placement}, overwrite=False)
            log_path = self.root/"logs"/f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("x") as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.jobs.append({"name": name, "process": proc, "queue": queue})
        atomic_json(self.root/"resource_plan.json", self.m["resources"], overwrite=False)

    def submit(self, label, payload, *, validator=False):
        with self.lock:
            self.counter += 1
            key = f"{self.counter:06d}_{label}"
        (self.validator if validator else self.work).submit(key, payload)
        return key

    def health(self):
        if time.time()-self.started > self.m["timeouts_seconds"]["suite"]:
            raise TimeoutError("Suite hard timeout")
        for row in self.jobs:
            code = row["process"].poll()
            if code is not None:
                raise RuntimeError(f"Worker {row['name']} exited unexpectedly: {code}")
        for queue, timeout in ((self.work, self.m["timeouts_seconds"]["task"]),
                               (self.validator, self.m["timeouts_seconds"]["validator"])):
            failures = list((queue.root/"failed").glob("*.json"))
            if failures:
                raise RuntimeError(f"Worker task failed; artifacts retained: {failures[0]}")
            for path in (queue.root/"running").glob("*.json"):
                try:
                    task = read_json(path)
                except FileNotFoundError:
                    continue
                if time.time()-task.get("started_unix", time.time()) > timeout:
                    raise TimeoutError(f"Task hard timeout: {task['id']}")
        with self.sample_lock:
            if time.time()-self.last_sample >= 10:
                import json
                relative = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
                group = Path("/sys/fs/cgroup")/relative.lstrip("/")
                usage = int((group/"memory.current").read_text())
                self.sampled_peak_bytes = max(self.sampled_peak_bytes, usage)
                statuses, alerts = {}, []
                for worker in self.jobs:
                    path = self.root/"workers"/worker["name"]/"status.json"
                    if path.exists():
                        row = read_json(path)
                        statuses[worker["name"]] = {k:row.get(k) for k in ("phase", "task", "event", "rollout_step", "last_progress_unix")}
                        if row.get("phase") != "idle" and time.time()-row.get("last_progress_unix", time.time()) > 900:
                            alerts.append(worker["name"])
                sample = {"unix": time.time(), "service_memory_bytes": usage,
                    "sampled_peak_gib": self.sampled_peak_bytes/2**30, "workers": statuses,
                    "progress_warnings_advisory": alerts,
                    "external_gpu_jobs_advisory": external_resources(self.m["resources"]["gpus"], os.getpid())}
                with (self.root/"resource_samples.jsonl").open("a") as handle:
                    handle.write(json.dumps(sample)+"\n")
                atomic_json(self.root/"resource_status.json", sample)
                if time.time()-self.last_source_check >= 600:
                    changes = verify(self.m, code_root=ROOT, inputs=False)
                    atomic_json(self.root/"workspace_advisory.json", {"changes": changes, "unix": time.time(),
                        "policy": "advisory only; frozen execution copy verified"})
                    self.last_source_check = time.time()
                self.last_sample = time.time()

    def wait(self, keys, *, validator=False):
        queue = self.validator if validator else self.work
        while True:
            self.health()
            rows = [queue.poll(k) for k in keys]
            if all(r is not None for r in rows):
                return rows
            time.sleep(2)

    def checkpoint_task(self, checkpoint, **extra):
        return {"checkpoint": checkpoint["checkpoint"], "checkpoint_sha256": checkpoint["checkpoint_sha256"], **extra}

    def evaluate(self, checkpoint, cases, label):
        return self.submit(label, self.checkpoint_task(checkpoint, kind="evaluate", cases=cases,
            source_costs={c["path"]: self.source_costs[c["path"]] for c in cases},
            legacy_source_costs={c["path"]: self.legacy_source_costs[c["path"]] for c in cases}
                if self.legacy_source_costs else {}), validator=True)

    def phase0(self, hb):
        hb.update(phase="P0_contract_and_semantic_baselines")
        initial = self.wait([self.submit("initialize", {"kind": "initialize"})])[0]
        # Persist references so no control process needs to inspect training tensors.
        atomic_json(self.root/"initial_checkpoint.json", initial, overwrite=False)
        warmups = [self.submit(f"validator_warmup_{i}", self.checkpoint_task(initial, kind="warmup",
            case=self.m["splits"]["train_fit16"][i], worker=f"validator{i}"), validator=True) for i in range(2)]
        self.wait(warmups, validator=True)
        canaries = [self.submit(f"canary_gpu{i}", {"kind": "canary", "case": self.m["splits"]["train_fit16"][i],
                     "worker": f"gpu{i}"}) for i in range(8)]
        results = self.wait(canaries)
        if (any(not r["passed"] or r["peak_reserved_gib"] > 21 for r in results)
                or self.sampled_peak_bytes > 96*2**30):
            raise RuntimeError("All-GPU capacity/implementation admission failed")
        unique = {c["path"]: c for cases in self.m["splits"].values() for c in cases}
        cases = list(unique.values())
        width = self.m["resources"]["worker_width"]
        ids = [self.submit(f"baseline_{i:03d}", self.checkpoint_task(initial, kind="baseline", cases=cases[i:i+width]))
               for i in range(0,len(cases),width)]
        baselines = [case for batch in self.wait(ids) for case in batch["cases"]]
        self.source_costs = {r["semantic"]["case_id"]: r["semantic"]["makespan"] for r in baselines}
        self.legacy_source_costs = {r["legacy"]["case_id"]:r["legacy"]["makespan"] for r in baselines}
        atomic_json(self.root/"baselines.json", {"environment": ENVIRONMENT, "history": HISTORY,
            "cases": baselines, "source_costs": self.source_costs, "legacy_source_costs": self.legacy_source_costs,
            "legacy_hard_costs_used": False}, overwrite=False)
        atomic_json(self.root/"technical_admission.json", {"passed": True, "canaries": results,
            "all_gpu_count": 8, "shared_validator_count": 2, "baseline_case_count": len(baselines),
            "scientific_admission": False}, overwrite=False)
        return initial

    def phase1(self, initial, hb):
        hb.update(phase="P1_fixed_checkpoint_and_local_branches")
        small = [self.submit(f"branch_probe_{i:02d}", self.checkpoint_task(initial, kind="branch", mode="diagnostic",
            case=c, seed=SEED+i, states=4, candidates=8)) for i,c in enumerate(self.m["splits"]["train_diag32"][:8])]
        self.wait(small)
        windows = [self.submit(f"window_probe_{i:02d}", self.checkpoint_task(initial,
            kind="window_diagnostic", case=c)) for i,c in enumerate(self.m["splits"]["train_diag32"][:8])]
        atomic_json(self.root/"window_diagnostic.json", {"cases": self.wait(windows),
            "diagnostic_only": True}, overwrite=False)
        teacher_ids = [self.submit(f"teacher_{i:02d}", self.checkpoint_task(initial, kind="branch", mode="diagnostic",
            case=c, seed=SEED+i, states=8, candidates=8)) for i,c in enumerate(self.m["splits"]["train_fit16"])]
        teachers = self.wait(teacher_ids)
        atomic_json(self.root/"teachers.json", {"data": teachers, "diagnostic_only": True}, overwrite=False)
        benchmark_ids = [self.submit(f"benchmark_{ref}_{i:02d}", {"kind": "benchmark", "reference": ref, "case": c})
            for ref in self.m["benchmark_checkpoints"] for i,c in enumerate(self.m["splits"]["train_probe64"])]
        benchmark = self.wait(benchmark_ids)
        atomic_json(self.root/"fixed_checkpoint_benchmark.json", {"cases": benchmark,
            "scope": "same H2/F4/soft and legacy observation; B/C transfer evaluation, not matched training"}, overwrite=False)
        iga_ids = [self.submit(f"iga_{budget}_{i:02d}", {"kind": "iga", "case": c, "budget_seconds": budget})
                   for budget in self.m["iga"]["cold_budgets_seconds"] for i,c in enumerate(self.m["iga"]["cases"])]
        atomic_json(self.root/"iga_diagnostic.json", {"results": self.wait(iga_ids),
            "scope": "cold search, H2/F4/soft, TrainDiag only; warm IGA not executed"}, overwrite=False)
        return teachers

    def fit(self, initial, teachers, hb):
        hb.update(phase="P2_local_fit")
        checkpoint = initial
        # Fixed single-case diagnostic, never used to initialize the multi-case fit.
        single = initial
        for step in range(self.m["fit"]["single_case_updates"]):
            single = self.wait([self.submit(f"single_fit_{step:02d}", self.checkpoint_task(single, kind="learn",
                data=teachers[:1], mode="rank", diagnostic=True, epochs=1, lr=self.m["fit"]["lr"],
                metadata={"diagnostic_only": True, "forbidden_as_rl_initialization": True, "single_step": step+1}))])[0]
        single_eval = self.evaluate(single, self.m["splits"]["train_fit16"][:1], "single_fit_greedy")
        for visit in range(self.m["fit"]["passes"]):
            order = sorted(teachers, key=lambda r: digest_json([SEED, "fit", visit, r["case_id"]]))
            for first in range(0, len(order), 4):
                checkpoint = self.wait([self.submit(f"fit_v{visit:02d}_b{first:02d}", self.checkpoint_task(checkpoint,
                    kind="learn", data=order[first:first+4], mode="rank", diagnostic=True,
                    epochs=self.m["fit"]["epochs"], lr=self.m["fit"]["lr"],
                    metadata={"diagnostic_only": True, "forbidden_as_rl_initialization": True, "fit_pass": visit+1}))])[0]
        ids = [self.evaluate(checkpoint, self.m["splits"][split], "fit_"+split)
               for split in ("train_fit16", "train_probe64")]
        fit, probe = self.wait(ids, validator=True)
        metrics = self.wait([self.submit("fit_metrics", self.checkpoint_task(checkpoint, kind="fit_metrics", data=teachers))])[0]
        teacher_costs = {r["case_id"]: r["teacher_cost"] for r in teachers}
        gate = fit_gate(self.m["splits"]["train_fit16"], self.source_costs, teacher_costs,
                        fit["summary"], probe["summary"], metrics["beneficial_top1"])
        atomic_json(self.root/"fit_admission.json", {**gate, "fit": fit, "probe": probe, "metrics": metrics,
            "single_case": self.wait([single_eval], validator=True)[0], "checkpoint": checkpoint,
            "diagnostic_weights_never_initialize_training": True}, overwrite=False)
        return gate

    def train_to(self, arm, state, until, *, warm=False):
        checkpoint = state["checkpoint"]
        queries, cursor = state.get("queries", 0), state.get("cursor", 0)
        cases = self.m["splits"]["train_pilot120"]
        evaluations = dict(state.get("evaluations", {}))
        while queries < until:
            visit, offset = divmod(cursor, len(cases))
            order = sorted(cases, key=lambda c: digest_json([SEED, "train_order", visit, c["content_sha256"]]))
            block = order[offset:offset+4]
            if len(block) != 4:
                raise ValueError("Case-batch cursor not at a complete group")
            mode = "warm" if warm else {"T_PPO": "ppo", "L_RANK": "rank", "L_RL": "local_rl", "L_RANK_RL": "local_rl"}[arm]
            # After prewarm both rank-only and local-RL see the same sampled-candidate protocol.
            collection_mode = "local_rl" if mode in ("rank", "local_rl") else mode
            ids = [self.submit(f"{arm}_q{queries:04d}_c{i}", self.checkpoint_task(checkpoint, kind="collect",
                mode=collection_mode, case=c, seed=SEED+cursor+i, candidates=4,
                source_cost=self.source_costs[c["path"]])) for i,c in enumerate(block)]
            data = self.wait(ids)
            actual = sum(r["query_count"] for r in data)
            expected = 16 if warm else 20
            if actual != expected or queries+actual > until:
                raise ValueError(f"Simulation budget mismatch: {actual}, expected {expected}")
            queries += actual
            cursor += len(block)
            meta = {"diagnostic_only": False, "arm": arm, "query_count": queries, "case_cursor": cursor,
                    "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"], "complete_global_group": True,
                    "warm_complete": warm and queries == 480}
            checkpoint = self.wait([self.submit(f"{arm}_update_q{queries:04d}", self.checkpoint_task(checkpoint,
                kind="learn", data=data, mode="rank" if warm else mode, metadata=meta,
                lr=self.m["training"]["lr"], epochs=self.m["training"]["ppo_epochs"]))])[0]
            if queries % 480 == 0:
                key = self.evaluate(checkpoint, self.m["splits"]["tune"], f"{arm}_eval_q{queries:04d}")
                evaluations[str(queries)] = key
            commit = {"checkpoint": checkpoint, "queries": queries, "cursor": cursor, "evaluations": evaluations}
            atomic_json(self.root/"train"/arm/f"commit_q{queries:04d}.json", commit, overwrite=False)
            atomic_json(self.root/"train"/arm/"latest_commit.json", commit)
        return {"checkpoint": checkpoint, "queries": queries, "cursor": cursor, "evaluations": evaluations}

    def training(self, initial, hb):
        hb.update(phase="P3_formal_warm_and_screen")
        blank = {"checkpoint": initial, "queries": 0, "cursor": 0, "evaluations": {}}
        with ThreadPoolExecutor(max_workers=3) as pool:
            warm_future = pool.submit(self.train_to, "PRE", copy.deepcopy(blank), 480, warm=True)
            pure = {arm: pool.submit(self.train_to, arm, copy.deepcopy(blank), 960) for arm in ("T_PPO", "L_RL")}
            warm = warm_future.result()
            warm_eval = self.wait([warm["evaluations"]["480"]], validator=True)[0]
            atomic_json(self.root/"prewarm.json", {"state": warm, "evaluation": warm_eval,
                "physical_cost_shared_once": 480, "standalone_budget_charged_each_arm": 480}, overwrite=False)
            states = {a: f.result() for a,f in pure.items()}
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = {a: pool.submit(self.train_to, a, copy.deepcopy(warm), 960) for a in ("L_RANK", "L_RANK_RL")}
            states.update({a: f.result() for a,f in future.items()})
        evaluations = {a: self.wait([s["evaluations"]["960"]], validator=True)[0] for a,s in states.items()}
        chosen = sorted([a for a,r in evaluations.items() if r["summary"]["gain_fraction"] >= .005 and risk_pass(r["summary"])],
                        key=lambda a: (-evaluations[a]["summary"]["gain_fraction"], a))[:2]
        atomic_json(self.root/"screen_admission.json", {"selected": chosen, "results": evaluations,
            "budget_unit": self.m["training"]["budget_unit"]}, overwrite=False)
        endpoints = {}
        for until in (1440, 1920):
            if not chosen:
                break
            hb.update(phase=f"P3_to_{until}", selected=chosen)
            running = list(chosen)
            if "L_RANK_RL" in chosen and "L_RANK" not in running:
                running.append("L_RANK")  # Required equal-budget attribution control, not an admitted candidate.
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {a: pool.submit(self.train_to, a, states[a], until) for a in running}
                states.update({a: f.result() for a,f in futures.items()})
            current = {a: self.wait([states[a]["evaluations"][str(until)]], validator=True)[0] for a in running}
            endpoints[str(until)] = current
            chosen = [a for a in chosen if current[a]["summary"]["gain_fraction"] >= .01 and risk_pass(current[a]["summary"])]
            atomic_json(self.root/f"pilot_admission_{until}.json", {"selected": chosen, "results": current,
                "attribution_controls_not_admitted": [a for a in running if a not in chosen]}, overwrite=False)
        target = {}
        for arm, row in endpoints.get("1920", {}).items():
            summary = row["summary"]
            passed = arm in chosen and summary["gain_fraction"] >= .02 and risk_pass(summary) and arm != "L_RANK"
            extra = {}
            if arm == "L_RANK_RL":
                from onpolicy.utils.stage3_research import paired_summary
                pre_cost = warm_eval["summary"]["makespan"]
                extra["post_warm_rl_gain"] = 1-summary["makespan"]/pre_cost
                control = endpoints["1920"].get("L_RANK")
                extra["rank_only_comparison"] = paired_summary(row["cases"],
                    {r["case_id"]:r["makespan"] for r in control["cases"]}) if control else None
                passed = (passed and extra["post_warm_rl_gain"] >= .01 and control is not None
                          and extra["rank_only_comparison"]["gain_fraction"] > 0)
            target[arm] = {"practical_rl_target_passed": bool(passed), "total_gain": summary["gain_fraction"], **extra}
            target[arm]["gain_vs_legacy_c0"] = row["legacy_summary"]["gain_fraction"]
            target[arm]["original_checkpoint_target_passed"] = bool(passed and row["legacy_summary"]["gain_fraction"] >= .02)
        return {"screen": evaluations, "endpoints": endpoints, "pilot_passed": chosen if "1920" in endpoints else [],
                "target": target, "prewarm": warm_eval, "single_seed_only": True, "finalblind_opened": False}

    def stop_owned(self):
        for row in self.jobs:
            proc = row["process"]
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for row in self.jobs:
            proc = row["process"]
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)

    def run(self):
        with ProgressHeartbeat(self.root/"status.json", phase="preflight", active_manifest=str(self.path)) as hb:
            try:
                hb.update(workspace_advisory=verify(self.m, code_root=ROOT))
                self.start_workers(hb)
                initial = self.phase0(hb)
                teachers = self.phase1(initial, hb)
                gate = self.fit(initial, teachers, hb)
                if not gate["passed"]:
                    result = {"completed": True, "status": "scientific_gate_not_passed", "formal_training_started": False,
                              "fit_gate": gate, "all_artifacts_retained": True}
                else:
                    atomic_json(self.root/"training_admission.json", {"passed": True, "arms": list(ARMS),
                        "source": initial, "diagnostic_weights_reused": False}, overwrite=False)
                    result = {"completed": True, "formal_training_started": True, **self.training(initial, hb)}
                if self.work.outstanding() or self.validator.outstanding():
                    raise RuntimeError("Refusing to finish with undrained tasks")
                atomic_json(self.root/"result.json", result, overwrite=False)
                for queue in (self.work, self.validator):
                    atomic_json(queue.root/"STOP.json", {"reason": "all tasks drained"}, overwrite=False)
                for row in self.jobs:
                    row["process"].wait(timeout=60)
                hb.update(phase="complete", result_status=result.get("status", "pilot_complete"))
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
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    manifest = read_json(args.manifest)
    if (Path(manifest["root"])/"status.json").exists():
        raise FileExistsError("No automatic retry; preserve failed suite and use an explicit new attempt")
    with (Path(manifest["root"])/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda s,f: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        Suite(manifest, args.manifest).run()


if __name__ == "__main__":
    main()
