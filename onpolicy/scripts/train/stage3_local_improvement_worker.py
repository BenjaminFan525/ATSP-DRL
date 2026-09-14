#!/usr/bin/env python3
"""Persistent GPU consumer for the all-GPU local improvement study."""
from __future__ import annotations

import argparse
import copy
import gc
import os
from pathlib import Path
import random
import sys
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from onpolicy.runner.shared.stage3_local_improvement_engine import LocalEngine, save_tensor
from onpolicy.runner.shared.stage3_research_engine import seed_all
from onpolicy.utils.stage3_local_improvement import HISTORY, SEED, TaskQueue, candidate_indices, verify
from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json, read_json, paired_summary, trajectory_seed
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat


def compact(row):
    return {k: v for k, v in row.items() if k not in ("records", "actions")}


def select_states(records, count, seed):
    if not records:
        raise ValueError("No genuine resource choice in this case")
    if count == 1:
        role = 1 + seed % 2
        candidates = [r for r in records if r["role"] == role] or records
        return [candidates[(seed // 2) % len(candidates)]]
    chosen = []
    for role in (1, 2):
        candidates = [r for r in records if r["role"] == role]
        for q in np.linspace(.1, .9, count // 2):
            if candidates:
                row = candidates[min(len(candidates)-1, int(q*len(candidates)))]
                if row["key"] not in [r["key"] for r in chosen]:
                    chosen.append(row)
    return chosen


def branch_group(engine, case, *, mode, seed, state_count=1, candidates=4, hb=None):
    start = time.time()
    anchor = engine.rollout([case], [42], collect=True, heartbeat=hb)[0]
    selected = select_states(anchor["records"], state_count, seed)
    states, queries, branch_rows = [], 1, []
    best_cost, best_actions = anchor["makespan"], anchor["actions"]
    for position, record in enumerate(selected):
        local_seed = trajectory_seed(seed, case["content_sha256"], position, 0)
        sampled = mode == "local_rl"
        actions = candidate_indices(record["old_logp"].numpy(), record["mask"].numpy(), candidates,
                                    local_seed, sampled=sampled)
        if mode == "warm":
            # Every formal warm case is exactly one baseline + three complete branches.
            while len(actions) < candidates:
                actions.append(actions[-1])
        reference_action = int(record["old_logp"].argmax())
        costs = [] if sampled else [anchor["makespan"]]
        if not sampled and actions[0] != reference_action:
            raise ValueError("Search candidates must include greedy incumbent first")
        to_run = actions if sampled else actions[1:]
        for first in range(0, len(to_run), engine.width):
            batch = to_run[first:first+engine.width]
            interventions = [{"key": record["key"], "action": action, "record": record} for action in batch]
            prefix = anchor["actions"][:record["key"][0]]
            rows = engine.rollout([case]*len(batch), [42]*len(batch), forced=[prefix]*len(batch),
                                 interventions=interventions, heartbeat=hb)
            queries += len(rows)
            for action, row in zip(batch, rows):
                costs.append(row["makespan"])
                branch_rows.append({"key": record["key"], "action": action, "cost": row["makespan"],
                    "prefix_exact": True, "intervention_hits": row["intervention_hits"],
                    "action_sha256": digest_json(row["actions"].tolist()), "diagnostics": row.get("local_diagnostics", {})})
                if row["makespan"] < best_cost:
                    best_cost, best_actions = row["makespan"], row["actions"]
            del rows
        states.append({"record": record, "actions": actions, "costs": costs,
            "reference_cost": anchor["makespan"], "reference_action": reference_action,
            "sampling": "on_policy_with_replacement" if sampled else "enumerated_search",
            "reference_suffix_sha256": engine.behavior_sha})
    return {"case_id": case["path"], "completed": True, "behavior_sha256": engine.behavior_sha,
        "kind": "local_on_policy" if mode == "local_rl" else "local_search",
        "states": states, "reference_cost": anchor["makespan"], "teacher_cost": best_cost,
        "teacher_actions": best_actions, "anchor_actions": anchor["actions"], "branches": branch_rows,
        "query_count": queries, "wall_seconds": time.time()-start,
        "scope": "single autoregressive resource factor, recomputed legal suffix; not fixed-other-agent COMA"}


class Worker:
    def __init__(self, manifest, width=7, device="cuda:0"):
        self.m, self.width, self.device = manifest, width, device
        self.runner, self.runner_key = None, None

    def engine(self, *, checkpoint=None, history=HISTORY, exploration="R", reference=None):
        key = (history, exploration, None if reference is None else reference["name"])
        if self.runner_key != key:
            self.close()
            if digest_file(self.m["source"]["path"]) != self.m["source"]["sha256"]:
                raise ValueError("Original C0 changed since worker startup")
            if reference and digest_file(reference["path"]) != reference["sha256"]:
                raise ValueError("Historical benchmark checkpoint changed")
            seed_all(self.m["training"]["seed"])
            self.runner = LocalEngine(self.m["source"]["path"], width=self.width, device=self.device,
                history=history, protocol=self.m["manifest_sha256"], exploration=exploration,
                reference_variant=reference.get("variant") if reference else None,
                reference_checkpoint=reference.get("path") if reference else None)
            self.runner_key = key
        if checkpoint is not None:
            self.runner.restore(checkpoint)
        return self.runner

    def close(self):
        if self.runner is not None:
            self.runner.close()
            self.runner = None
            self.runner_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self, task, output, hb):
        kind = task["kind"]
        output.mkdir(parents=True, exist_ok=False)
        start = time.time()
        if task.get("checkpoint") and digest_file(task["checkpoint"]) != task["checkpoint_sha256"]:
            raise ValueError("Task checkpoint changed")
        if kind == "initialize":
            self.close()
            runner = self.engine()
            path = output / "initial.pt"
            sha = runner.save(path, diagnostic_only=False, query_count=0, case_cursor=0, arm="C0_semantic")
            result = {"checkpoint": str(path), "checkpoint_sha256": sha}
        elif kind == "baseline":
            runner = self.engine(checkpoint=task["checkpoint"])
            cases = task.get("cases", [task.get("case")])
            runner.history = "legacy"
            try:
                legacy = runner.rollout(cases, [42]*len(cases), heartbeat=hb, audit_graph=True)
            finally:
                runner.history = HISTORY
            rows = runner.rollout(cases, [42]*len(cases), heartbeat=hb)
            result = {"cases": [{"legacy": compact(old), "semantic": compact(row),
                "semantic_change_seconds": row["makespan"]-old["makespan"]} for old,row in zip(legacy,rows)],
                "query_count": 2*len(cases), "scope": "both H2/F4/soft; observation change is not RL gain"}
            save_tensor(output / "actions.pt", {"legacy": [r["actions"] for r in legacy], "semantic": [r["actions"] for r in rows]})
        elif kind == "benchmark":
            reference = self.m["benchmark_checkpoints"][task["reference"]]
            runner = self.engine(history="legacy", exploration="J", reference=reference)
            case = task["case"]
            greedy = runner.rollout([case], [42], heartbeat=hb)[0]
            costs, queries = [], 1
            for first in range(0, 8, self.width):
                n = min(self.width, 8-first)
                seeds = [trajectory_seed(SEED, case["content_sha256"], 0, i) for i in range(first, first+n)]
                rows = runner.rollout([case]*n, seeds, deterministic=False, heartbeat=hb)
                costs.extend(r["makespan"] for r in rows)
                save_tensor(output/f"samples_{first}.pt", [{"summary": compact(r), "actions": r["actions"]} for r in rows])
                queries += len(rows)
            result = {"case_id": case["path"], "reference": task["reference"], "greedy": greedy["makespan"],
                "sample_costs": costs, "sample_mean": float(np.mean(costs)), "best8": min(costs), "query_count": queries,
                "environment": "H2/F4/soft", "history": "legacy",
                "scope": "matched evaluation; historical B/C were trained under H3/F4/hard"}
        elif kind in ("branch", "collect"):
            runner = self.engine(checkpoint=task["checkpoint"])
            case, mode = task["case"], task["mode"]
            allowed = (self.m["splits"]["train_pilot120"] if kind == "collect" else
                       self.m["splits"]["train_diag32"] + self.m["splits"]["train_fit16"])
            if case not in allowed or (kind == "collect" and runner.checkpoint_meta.get("diagnostic_only")):
                raise ValueError("Formal/diagnostic data or checkpoint boundary violation")
            if mode == "ppo":
                rows = []
                for first in range(0, 5, self.width):
                    n = min(self.width, 5-first)
                    rows.extend(runner.rollout([case]*n, [trajectory_seed(task["seed"], case["content_sha256"], 0, i)
                        for i in range(first, first+n)], deterministic=False, collect=True, heartbeat=hb))
                group = {"case_id": case["path"], "kind": "full_on_policy", "completed": True,
                    "behavior_sha256": runner.behavior_sha, "query_count": 5, "source_cost": task["source_cost"],
                    "trajectories": rows}
            else:
                group = branch_group(runner, case, mode=mode, seed=task["seed"],
                    state_count=task.get("states", 1), candidates=task.get("candidates", 4), hb=hb)
            path = output/"data.pt"
            sha = save_tensor(path, group)
            result = {"path": str(path), "sha256": sha, "case_id": case["path"], "query_count": group["query_count"],
                "reference_cost": group.get("reference_cost"), "teacher_cost": group.get("teacher_cost"),
                "state_count": len(group.get("states", []))}
        elif kind == "learn":
            runner = self.engine(checkpoint=task["checkpoint"])
            if not task.get("diagnostic", False) and runner.checkpoint_meta.get("diagnostic_only"):
                raise ValueError("Diagnostic weights cannot initialize formal learning")
            groups = []
            for entry in task["data"]:
                if digest_file(entry["path"]) != entry["sha256"]:
                    raise ValueError("Training data identity changed")
                groups.append(torch.load(entry["path"], map_location="cpu", weights_only=False))
            split = "train_fit16" if task.get("diagnostic", False) else "train_pilot120"
            if any(g["case_id"] not in {c["path"] for c in self.m["splits"][split]} for g in groups):
                raise ValueError("Learning cases escaped their registered split")
            update = runner.learn(groups, task["mode"], lr=task.get("lr", 1e-4), epochs=task.get("epochs", 2),
                                  diagnostic=task.get("diagnostic", False), heartbeat=hb)
            path = output/"checkpoint.pt"
            sha = runner.save(path, **task["metadata"])
            result = {"checkpoint": str(path), "checkpoint_sha256": sha, "update": update,
                      "metadata": task["metadata"]}
        elif kind == "warmup":
            runner = self.engine(checkpoint=task["checkpoint"])
            row = runner.rollout([task["case"]], [42], heartbeat=hb)[0]
            result = {"case": compact(row), "validator_resident": True}
        elif kind == "evaluate":
            runner = self.engine(checkpoint=task["checkpoint"])
            rows = []
            for first in range(0, len(task["cases"]), self.width):
                cases = task["cases"][first:first+self.width]
                batch = runner.rollout(cases, [42]*len(cases), heartbeat=hb)
                rows.extend(compact(r) for r in batch)
            result = {"cases": rows, "summary": paired_summary(rows, task["source_costs"]),
                "checkpoint_sha256": task["checkpoint_sha256"], "decoder": "single_greedy",
                "history": HISTORY, "environment": "H2/F4/soft", "evaluation_queries": len(rows)}
            if task.get("legacy_source_costs"):
                result["legacy_summary"] = paired_summary(rows, task["legacy_source_costs"])
        elif kind == "fit_metrics":
            runner = self.engine(checkpoint=task["checkpoint"])
            hits, eligible = 0, 0
            for entry in task["data"]:
                if digest_file(entry["path"]) != entry["sha256"]:
                    raise ValueError("Teacher data changed")
                group = torch.load(entry["path"], map_location="cpu", weights_only=False)
                for state in group["states"]:
                    winners = {a for a, c in zip(state["actions"], state["costs"]) if c < state["reference_cost"]-1e-6}
                    if winners:
                        with torch.no_grad():
                            lp, _ = runner.record_distribution([state["record"]])
                        hits += int(int(lp.argmax(-1).item()) in winners)
                        eligible += 1
            result = {"hits": hits, "eligible": eligible, "beneficial_top1": hits/eligible if eligible else 0.}
        elif kind == "canary":
            result = self.canary(task, output, hb)
        elif kind == "window_diagnostic":
            comparisons = []
            for exploration in ("R", "J"):
                runner = self.engine(checkpoint=task["checkpoint"], exploration=exploration)
                anchor = runner.rollout([task["case"]], [42], collect=True, heartbeat=hb)[0]
                positions = sorted({r["key"][0] for r in select_states(anchor["records"], 4, SEED)})
                for position in positions:
                    for width in (1, 4):
                        seed = trajectory_seed(SEED, task["case"]["content_sha256"], position, width)
                        row = runner.rollout([task["case"]], [seed], forced=[anchor["actions"][:position]],
                            sampling_window=(position, position+width), heartbeat=hb)[0]
                        save_tensor(output/f"{exploration}_p{position}_w{width}.pt", {"summary": compact(row), "actions": row["actions"]})
                        comparisons.append({"mode": exploration, "position": position, "window": width,
                            "cost_delta": row["makespan"]-anchor["makespan"], "makespan": row["makespan"],
                            "reference_cost": anchor["makespan"], "prefix_exact": True})
            result = {"case_id": task["case"]["path"], "comparisons": comparisons, "diagnostic_only": True,
                "scope": "role-window intervention, not single-factor labels and never PPO training data"}
        elif kind == "iga":
            # This worker owns its CPU lease; solver does not borrow other jobs' cores.
            self.close()
            case = task["case"]
            case_list = output/"case_list.json"
            atomic_json(case_list, [case["name"]], overwrite=False)
            budget = task["budget_seconds"]
            command = [sys.executable, "-B", str(ROOT/"onpolicy/envs/HKBZ/experiment/generate_stage3_joint_iga_labels.py"),
                "--dataset-dir", str(Path(case["path"]).parent), "--output-dir", str(output/"search"),
                "--case-list-json", str(case_list), "--workers", "1", "--population", "20",
                "--time-budget-seconds", str(budget), "--cumulative-budget-seconds", str(budget),
                "--seed", str(SEED), "--device-future-intent-horizon", "2",
                "--device-future-intent-mode", "bounded_frontier", "--device-frontier-max-requests", "4",
                "--device-request-capacity-per-plane", "5", "--resource-release-aware-eta",
                "--device-lookahead-reservation-mode", "soft", "--device-reservation-grace-seconds", "300",
                "--device-lookahead-safety-margin", "60", "--max-plane-agents", "24", "--max-device-num", "80"]
            atomic_json(output/"command.json", {"command": command, "cold_start": True}, overwrite=False)
            with (output/"solver.log").open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    hb.update(event="iga_search", budget_seconds=budget, elapsed_seconds=time.time()-start)
                    if time.time()-start > budget+1800:
                        process.terminate()
                        process.wait(timeout=30)
                        raise TimeoutError("IGA hard timeout including final replay")
                    time.sleep(10)
                if process.returncode:
                    raise RuntimeError(f"IGA failed with code {process.returncode}; solver.log retained")
            result = {"summary": read_json(output/"search/summary.json"), "case_id": case["path"],
                "budget_seconds": budget, "cold_start": True, "environment": "H2/F4/soft",
                "scope": "diagnostic comparison only; not training teachers"}
        else:
            raise ValueError(kind)
        result.update(wall_seconds=time.time()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if self.device.startswith("cuda") else 0,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if self.device.startswith("cuda") else 0)
        atomic_json(output/"result.json", result, overwrite=False)
        return result

    def canary(self, task, output, hb):
        from onpolicy.scripts.train.stage3_local_worker import nested_close
        self.close()
        runner = self.engine()
        initial = output/"initial.pt"
        runner.save(initial, diagnostic_only=True, query_count=0, case_cursor=0)
        frozen = {k: v.cpu().clone() for k,v in runner.policy.ac.state_dict().items()}
        group = branch_group(runner, task["case"], mode="local_rl", seed=SEED, candidates=2, hb=hb)
        runner.residual_enabled = False
        try:
            native = runner.rollout([task["case"]], [42], heartbeat=hb)[0]
        finally:
            runner.residual_enabled = True
        if (abs(native["makespan"]-group["reference_cost"]) > 1e-6
                or not np.array_equal(native["actions"], group["anchor_actions"])):
            raise RuntimeError("Zero residual does not preserve the native C0 greedy policy")
        # Actual sampled-factor update and exact optimizer/RNG restoration.
        first = runner.learn([group], "local_rl", heartbeat=hb)
        expected = copy.deepcopy(runner.scores.state_dict())
        optimizer = copy.deepcopy(runner.optimizer.state_dict())
        runner.restore(initial)
        runner.learn([group], "local_rl", heartbeat=hb)
        error = max(float((v-expected[k]).abs().max()) for k,v in runner.scores.state_dict().items())
        optimizer_ok = nested_close(optimizer, runner.optimizer.state_dict())
        unchanged = all(torch.equal(v.cpu(), frozen[k]) for k,v in runner.policy.ac.state_dict().items())
        if error > 2e-6 or not optimizer_ok or not unchanged:
            raise RuntimeError("Residual learner restore/frozen-state contract failed")
        runner.restore(initial)
        search = branch_group(runner, task["case"], mode="warm", seed=SEED, candidates=4, hb=hb)
        rank = runner.learn([search], "rank", heartbeat=hb)
        runner.restore(initial)
        traces = runner.rollout([task["case"]], [SEED], deterministic=False, collect=True, heartbeat=hb)
        ppo = {"case_id": task["case"]["path"], "completed": True, "kind": "full_on_policy",
            "behavior_sha256": runner.behavior_sha, "query_count": 1,
            "source_cost": group["reference_cost"], "trajectories": traces}
        ppo_update = runner.learn([ppo], "ppo", heartbeat=hb)
        # Nonzero rank gradients and a populated-Adam next-update restore are
        # checked with explicitly SYNTHETIC labels on real observed features.
        # They are not rollout costs, teachers, or scientific results.
        synthetic = copy.deepcopy(group)
        state = synthetic["states"][0]
        legal = torch.nonzero(state["record"]["mask"]).flatten().tolist()
        state["actions"], state["costs"] = legal[:2], [101., 99.]
        state["reference_cost"] = 100.
        synthetic["synthetic_math_only"] = True
        populated = output/"populated_optimizer.pt"
        runner.save(populated, diagnostic_only=True, synthetic_math_only=True, query_count=0, case_cursor=0)
        expected_rng = (torch.rand(3), np.random.random(), random.random())
        runner.restore(populated)
        actual_rng = (torch.rand(3), np.random.random(), random.random())
        if not nested_close(expected_rng, actual_rng):
            raise RuntimeError("Saved RNG state did not restore")
        synthetic_rank = runner.learn([synthetic], "rank", diagnostic=True, epochs=1, heartbeat=hb)
        expected_scores, expected_optim = copy.deepcopy(runner.scores.state_dict()), copy.deepcopy(runner.optimizer.state_dict())
        runner.restore(populated)
        runner.learn([synthetic], "rank", diagnostic=True, epochs=1, heartbeat=hb)
        nonzero_error = max(float((v-expected_scores[k]).abs().max()) for k,v in runner.scores.state_dict().items())
        if (nonzero_error > 2e-6 or not nested_close(expected_optim, runner.optimizer.state_dict())
                or synthetic_rank["epochs"][0]["preclip_gradient_norm"] <= 0):
            raise RuntimeError("Populated optimizer next-update contract failed")
        save_tensor(output/"local_group.pt", group)
        save_tensor(output/"rank_group.pt", search)
        return {"passed": True, "zero_residual_native_greedy_exact": True,
            "resume_model_max_abs_error": error, "optimizer_restore": optimizer_ok,
            "populated_adam_next_update_error": nonzero_error, "rng_restore_exact": True,
            "synthetic_rank_math_test": synthetic_rank,
            "frozen_c0_unchanged": unchanged, "local_update": first, "rank_update": rank, "ppo_update": ppo_update,
            "scope": "real simulation and three actor objectives; not scientific success"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve", "canary", "smoke"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queue", choices=("work", "validator"), default="work")
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-id", default="development")
    args = parser.parse_args()
    if args.command in ("canary", "smoke") and args.manifest is None:
        prior = read_json(ROOT/"result/hkbz_train_logs/stage3_full_policy_all_20260909_r1_recovery1/manifest.json")
        manifest = {"source": prior["source"], "training": {"seed": SEED}, "manifest_sha256": "development",
                    "splits": prior["splits"]}
    else:
        manifest = read_json(args.manifest)
        verify(manifest, code_root=ROOT)
    worker = Worker(manifest, width=args.width, device=args.device)
    try:
        if args.command == "canary":
            args.output.mkdir(parents=True, exist_ok=False)
            with ProgressHeartbeat(args.output/"status.json", phase="canary") as hb:
                worker.run({"kind": "canary", "case": manifest["splits"]["train_fit16"][0]}, args.output/"run", hb)
            return
        if args.command == "smoke":
            args.output.mkdir(parents=True, exist_ok=False)
            with ProgressHeartbeat(args.output/"status.json", phase="smoke") as hb:
                initial = worker.run({"kind": "initialize"}, args.output/"initialize", hb)
                baseline = worker.run({"kind": "baseline", **initial,
                    "cases": manifest["splits"]["train_fit16"][:min(args.width, 2)]}, args.output/"baseline", hb)
                costs = {r["semantic"]["case_id"]:r["semantic"]["makespan"] for r in baseline["cases"]}
                worker.run({"kind": "evaluate", **initial, "cases": manifest["splits"]["train_fit16"][:min(args.width, 2)],
                    "source_costs": costs}, args.output/"evaluate", hb)
            return
        args.output.mkdir(parents=True, exist_ok=False)
        queue = TaskQueue(Path(manifest["root"])/args.queue)
        with ProgressHeartbeat(args.output/"status.json", phase="idle", queue=args.queue) as hb:
            while True:
                task = queue.claim(args.worker_id)
                if task is None:
                    if (queue.root/"STOP.json").exists():
                        break
                    time.sleep(1)
                    continue
                hb.update(phase=task["payload"]["kind"], task=task["id"])
                try:
                    result = worker.run(task["payload"], queue.root/"artifacts"/task["id"], hb)
                    queue.finish(task, result=result)
                except BaseException:
                    queue.finish(task, error=traceback.format_exc())
                    raise
                hb.update(phase="idle", last_task=task["id"])
    finally:
        worker.close()


if __name__ == "__main__":
    main()
