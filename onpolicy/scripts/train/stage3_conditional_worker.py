#!/usr/bin/env python3
"""Persistent consumer for one shared work queue / validator pool."""
from __future__ import annotations

import argparse
import copy
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from onpolicy.scripts.train import stage3_local_improvement_worker as base_worker
from onpolicy.scripts.train.stage3_local_improvement_worker import Worker, compact, branch_group
from onpolicy.runner.shared.stage3_conditional_engine import ConditionalEngine, CachedRanking
from onpolicy.runner.shared.stage3_local_improvement_engine import save_tensor, masked_kl
from onpolicy.runner.shared.stage3_research_engine import seed_all
from onpolicy.utils.stage3_conditional_improvement import select_stratified_states, cost_labels, evaluation_summary, STUDY
from onpolicy.utils.stage3_local_improvement import HISTORY, SEED, TaskQueue, verify, validate_group
from onpolicy.utils.stage3_research import atomic_json, digest_file, read_json
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat


# This imported helper is private to this dedicated consumer process; legacy
# services keep their own module instances and original selection protocol.
base_worker.select_states = select_stratified_states


class ConditionalWorker(Worker):
    def engine(self, *, checkpoint=None, history=HISTORY, exploration="R", reference=None):
        key = (history, exploration, None if reference is None else reference["name"])
        architecture = getattr(self, "_task_architecture", "score")
        # Checkpoints select their own head in restore(). A checkpoint-free
        # request must not inherit a previous task's initialized/restored head.
        if (self.runner_key != key or (checkpoint is None and self.runner is not None
                                       and self.runner.architecture != architecture)):
            self.close()
            if digest_file(self.m["source"]["path"]) != self.m["source"]["sha256"]:
                raise ValueError("C0 identity changed")
            seed_all(self.m["training"]["seed"])
            self.runner = ConditionalEngine(self.m["source"]["path"], width=self.width, device=self.device,
                architecture=architecture, history=history,
                protocol=self.m.get("learning_protocol_sha256", self.m["manifest_sha256"]), exploration=exploration,
                reference_variant=reference.get("variant") if reference else None,
                reference_checkpoint=reference.get("path") if reference else None)
            self.runner_key = key
        self.runner.ablation = "full"
        self.runner.query_observer = self.record_queries
        if checkpoint:
            self.runner.restore(checkpoint)
        return self.runner

    def record_queries(self, kind, count):
        self.query_counts[kind] += count

    def run(self, task, output, hb):
        self.query_counts = {"started": 0, "completed": 0}
        self._task_architecture = task.get("architecture", "score") if task["kind"] == "canary" else "score"
        try:
            result = self.execute(task, output, hb)
        finally:
            # Includes failed canaries: initialization choices are task-local.
            self._task_architecture = "score"
            if output.exists():
                atomic_json(output/"query_ledger.json", {"kind": task["kind"], **self.query_counts,
                    "incomplete_attempts": self.query_counts["started"]-self.query_counts["completed"]}, overwrite=False)
        result["physical_query_ledger"] = dict(self.query_counts)
        return result

    def load_groups(self, entries, allowed):
        groups = []
        permitted = {c["path"] for c in allowed}
        for entry in entries:
            if digest_file(entry["path"]) != entry["sha256"]:
                raise ValueError("Teacher identity changed")
            group = torch.load(entry["path"], map_location="cpu", weights_only=False)
            if group["case_id"] not in permitted:
                raise ValueError("Learning data escaped its registered split")
            groups.append(group)
        return groups

    def execute(self, task, output, hb):
        kind = task["kind"]
        custom = {"initialize_model", "import_previous_fit", "ablation", "cached_fit", "cache_metrics",
                  "rank_learn", "benchmark_candidate", "formal_warm_fit", "evaluate"}
        if kind == "learn" and task["mode"] == "rank":
            task = dict(task, kind="rank_learn")
            kind = "rank_learn"
        if kind not in custom:
            return super().run(task, output, hb)
        start = time.time()
        output.mkdir(parents=True, exist_ok=False)
        if task.get("checkpoint") and digest_file(task["checkpoint"]) != task["checkpoint_sha256"]:
            raise ValueError("Task checkpoint changed")
        if kind == "initialize_model":
            self.close()
            runner = self.engine()
            seed_all(SEED)
            runner.set_architecture(task["architecture"])
            path = output/"initial.pt"
            sha = runner.save(path, diagnostic_only=False, query_count=0, case_cursor=0, arm="C0_semantic")
            result = {"checkpoint": str(path), "checkpoint_sha256": sha, "architecture": runner.architecture}
        elif kind == "import_previous_fit":
            self.close()
            runner = self.engine()
            if digest_file(task["previous_checkpoint"]) != task["previous_sha256"]:
                raise ValueError("Historical fitted checkpoint changed")
            payload = torch.load(task["previous_checkpoint"], map_location="cpu", weights_only=False)
            if (payload.get("schema") != "stage3-local-residual-checkpoint-v1"
                    or payload["source_sha256"] != runner.source_sha or payload["history"] != HISTORY
                    or payload["environment"] != {"h": 2, "f": 4, "reservation": "soft"}):
                raise ValueError("Historical fitted model contract mismatch")
            architecture = payload.get("metadata", {}).get("architecture", "score")
            if architecture != "score":
                raise ValueError("Historical WAIT/ranking ablation requires a score checkpoint")
            # Legacy residual checkpoints omit architecture metadata. Rebuild
            # the exact score head/Adam before strict loading; never drop keys.
            runner.set_architecture(architecture)
            runner.scores.load_state_dict(payload["scores"], strict=True)
            runner.optimizer.load_state_dict(payload["optimizer"])
            runner.policy_updates = payload["policy_updates"]
            path = output/"historical_diagnostic.pt"
            sha = runner.save(path, diagnostic_only=True, previous_sha256=task["previous_sha256"],
                              query_count=0, case_cursor=0)
            result = {"checkpoint": str(path), "checkpoint_sha256": sha, "architecture": architecture,
                      "previous_sha256": task["previous_sha256"], "diagnostic_only": True,
                      "policy_updates": runner.policy_updates}
        elif kind == "evaluate":
            runner = self.engine(checkpoint=task["checkpoint"])
            rows = []
            for first in range(0, len(task["cases"]), self.width):
                cases = task["cases"][first:first+self.width]
                rows.extend(compact(r) for r in runner.rollout(cases, [42]*len(cases), heartbeat=hb, allow_incomplete=True))
            result = {"cases": rows, "summary": evaluation_summary(rows, task["source_costs"]),
                "legacy_summary": evaluation_summary(rows, task["legacy_source_costs"]) if task.get("legacy_source_costs") else None,
                "checkpoint_sha256": task["checkpoint_sha256"], "decoder": "single_greedy", "evaluation_queries": len(rows),
                "history": HISTORY, "environment": "H2/F4/soft"}
        elif kind == "ablation":
            runner = self.engine(checkpoint=task["checkpoint"])
            runner.ablation = task["ablation"]
            try:
                rows = runner.rollout(task["cases"], [42]*len(task["cases"]), heartbeat=hb)
            finally:
                runner.ablation = "full"
            save_tensor(output/"actions.pt", [r["actions"] for r in rows])
            result = {"cases": [compact(r) for r in rows], "query_count": len(rows),
                      "ablation": task["ablation"], "diagnostic_only": True}
        elif kind in ("cached_fit", "cache_metrics"):
            runner = self.engine(checkpoint=task["checkpoint"])
            groups = self.load_groups(task["data"], self.m["splits"]["train_fit16"])
            if kind == "cache_metrics":
                result = {"metrics": CachedRanking(runner, groups, weighted=task.get("weighted", False)).metrics()}
            else:
                weighted, lookup = task.get("weighted", False), task.get("lookup", False)
                single = None
                if not lookup:
                    informative = [g for g in groups if any(
                        max(cost_labels(s, meaningful=weighted)[0].values())-min(cost_labels(s, meaningful=weighted)[0].values())
                        > cost_labels(s, meaningful=weighted)[1] for s in g["states"])]
                    single_group = (informative or groups)[0]
                    single = CachedRanking(runner, [single_group], weighted=weighted).fit(
                        lr=task["lr"], steps=task.get("max_updates", 2000), heartbeat=hb)
                    single["case_id"] = single_group["case_id"]
                    path = output/"single.pt"
                    runner.save(path, diagnostic_only=True, forbidden_as_rl_initialization=True,
                                query_count=0, case_cursor=0, fit_scope="single")
                    runner.restore(task["checkpoint"])
                trainer = CachedRanking(runner, groups, weighted=weighted, lookup=lookup)
                fit = trainer.fit(lr=task["lr"], steps=task.get("max_updates", 2000), heartbeat=hb)
                result = {"fit": fit, "single": single, "architecture": runner.architecture,
                          "query_count": 0, "weighted": weighted, "lookup_only": lookup}
                if lookup:
                    save_tensor(output/"lookup_diagnostic_only.pt", trainer.lookup_logits.state_dict())
                else:
                    path = output/"checkpoint.pt"
                    sha = runner.save(path, diagnostic_only=True, forbidden_as_rl_initialization=True,
                        query_count=0, case_cursor=0, weighted=weighted, fit_data=[e["sha256"] for e in task["data"]])
                    result.update(checkpoint=str(path), checkpoint_sha256=sha)
        elif kind == "formal_warm_fit":
            runner = self.engine(checkpoint=task["checkpoint"])
            if runner.checkpoint_meta.get("diagnostic_only"):
                raise ValueError("Diagnostic weights cannot initialize formal warm")
            groups = self.load_groups(task["data"], self.m["splits"]["train_pilot120"])
            if len(groups) != 120 or sum(g["query_count"] for g in groups) != 480:
                raise ValueError("Formal warm requires the complete registered 120-case/480-query corpus")
            validate_group(groups, runner.behavior_sha, "rank")
            fit = CachedRanking(runner, groups, weighted=True).fit(lr=task["lr"], steps=2000, heartbeat=hb)
            path = output/"checkpoint.pt"
            metadata = {"diagnostic_only": False, "arm": "PRE", "query_count": 480, "case_cursor": 120,
                "warm_complete": True, "complete_global_group": True, "parent_checkpoint_sha256": task["checkpoint_sha256"],
                "fit_data": [e["sha256"] for e in task["data"]]}
            sha = runner.save(path, **metadata)
            result = {"checkpoint": str(path), "checkpoint_sha256": sha, "fit": fit, "metadata": metadata}
        elif kind == "rank_learn":
            runner = self.engine(checkpoint=task["checkpoint"])
            if task.get("diagnostic", False) or runner.checkpoint_meta.get("diagnostic_only"):
                raise ValueError("Formal ranking rejects diagnostic initialization")
            groups = self.load_groups(task["data"], self.m["splits"]["train_pilot120"])
            validate_group(groups, runner.behavior_sha, "rank")
            try:
                trainer = CachedRanking(runner, groups, weighted=True)
                if not trainer.pairs:
                    trainer = None
            except ValueError as exc:
                if "No informative cost preferences" not in str(exc):
                    raise
                trainer = None
            update = {"mode": "cost_weighted_rank", "actor_updates": 0, "epochs": []}
            if trainer is not None:
                with torch.no_grad():
                    before = [p.detach().clone() for p in trainer.predictions()]
                for param in runner.optimizer.param_groups:
                    param["lr"] = task.get("lr", 1e-4)
                for epoch in range(task.get("epochs", 2)):
                    runner.optimizer.zero_grad(set_to_none=True)
                    loss = trainer.loss(trainer.predictions())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite formal ranking loss")
                    loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(runner.scores.parameters(), 1., error_if_nonfinite=True))
                    runner.optimizer.step()
                    runner.policy_updates += 1
                    update["actor_updates"] += 1
                    with torch.no_grad():
                        after = trainer.predictions()
                        kl = [float(masked_kl(a[None], b[None], s["record"]["mask"].to(a.device)[None]))
                              for a, b, (_, s) in zip(before, after, trainer.rows)]
                    mean, maximum = float(np.mean(kl)), max(kl)
                    update["epochs"].append({"loss": float(loss), "gradient_norm": norm,
                                              "mean_kl": mean, "max_state_kl": maximum})
                    if mean > .04 or maximum > .2:
                        raise RuntimeError("Formal rank hard KL guard; no silent rollback")
                    if mean > .02:
                        update["soft_kl_stop"] = True
                        break
            path = output/"checkpoint.pt"
            sha = runner.save(path, **task["metadata"])
            result = {"checkpoint": str(path), "checkpoint_sha256": sha, "update": update, "metadata": task["metadata"]}
        elif kind == "benchmark_candidate":
            from onpolicy.utils.stage3_research import trajectory_seed
            runner = self.engine(checkpoint=task["checkpoint"], exploration="R")
            case = task["case"]
            greedy = runner.rollout([case], [42], heartbeat=hb)[0]
            costs = []
            for first in range(0, 8, self.width):
                n = min(self.width, 8-first)
                rows = runner.rollout([case]*n, [trajectory_seed(SEED, case["content_sha256"], 0, i)
                    for i in range(first, first+n)], deterministic=False, heartbeat=hb)
                costs.extend(r["makespan"] for r in rows)
            result = {"case_id": case["path"], "greedy": greedy["makespan"], "sample_costs": costs,
                      "best8": min(costs), "query_count": 9, "history": HISTORY, "environment": "H2/F4/soft"}
        result.update(wall_seconds=time.time()-start,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if self.device.startswith("cuda") else 0)
        atomic_json(output/"result.json", result, overwrite=False)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve", "canary", "smoke"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue", choices=("work", "validator"), default="work")
    parser.add_argument("--worker-id", default="development")
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("study") != STUDY:
        raise ValueError("Wrong conditional-study manifest")
    verify(manifest, code_root=ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    worker = ConditionalWorker(manifest, width=args.width, device=args.device)
    try:
        with ProgressHeartbeat(args.output/"status.json", phase="idle", queue=args.queue) as hb:
            if args.command == "canary":
                worker.run({"kind": "canary", "case": manifest["splits"]["train_fit16"][0]}, args.output/"run", hb)
                return
            if args.command == "smoke":
                for architecture in ("score", "conditional", "conditional_pair"):
                    initial = worker.run({"kind": "initialize_model", "architecture": architecture}, args.output/architecture/"initial", hb)
                    case = manifest["splits"]["train_fit16"][0]
                    branch = worker.run({"kind": "branch", **initial, "case": case, "mode": "warm",
                        "seed": SEED, "states": 2, "candidates": 4}, args.output/architecture/"branch", hb)
                    worker.run({"kind": "cached_fit", **initial, "data": [branch], "lr": .001,
                        "max_updates": 50}, args.output/architecture/"fit", hb)
                return
            queue = TaskQueue(Path(manifest["root"])/args.queue)
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
