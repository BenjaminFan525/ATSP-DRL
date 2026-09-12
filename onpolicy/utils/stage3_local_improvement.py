"""Contracts for local policy improvement. No CUDA or environment side effects."""
from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json, read_json

SCHEMA = "stage3-local-improvement-v1"
ARMS = ("T_PPO", "L_RANK", "L_RL", "L_RANK_RL")
HISTORY = "stable_request_identity_v1"
SEED = 2026091001
ENVIRONMENT = {"device_future_intent_horizon": 2, "device_frontier_max_requests": 4,
               "device_lookahead_reservation_mode": "soft"}


def identity(manifest):
    return digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def remap_history(previous, old_keys, current_keys, active=None):
    result = np.asarray(previous).copy()
    lookup = {}
    for index, key in enumerate(current_keys):
        if key is not None:
            if key in lookup:
                raise ValueError("Ambiguous live request identity")
            lookup[key] = index
    audit = {"history_uses": 0, "index_aliases": 0, "expired": 0, "noop": 0}
    for agent in range(24, len(result)):
        old_index = int(result[agent, 0])
        counted = active is None or bool(active[agent])
        if old_index == 0:
            audit["noop"] += int(counted)
        elif old_index >= 0:
            audit["history_uses"] += int(counted)
            key = old_keys[old_index] if old_index < len(old_keys) else None
            found = lookup.get(key, -1) if key is not None else -1
            audit["expired"] += int(counted and found < 0)
            audit["index_aliases"] += int(counted and (old_index >= len(current_keys) or current_keys[old_index] != key))
            result[agent, 0] = found
    return result, audit


def candidate_indices(log_probs, mask, count, seed, *, sampled=False):
    legal = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(legal):
        raise ValueError("No legal local action")
    values = np.asarray(log_probs, dtype=np.float64)[legal]
    probs = np.exp(values - values.max())
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    if sampled:
        return [int(x) for x in rng.choice(legal, size=count, replace=True, p=probs)]
    ordered = legal[np.argsort(-values, kind="stable")].tolist()
    # Include the incumbent and one high-probability alternative; retain tail exploration.
    chosen = ordered[:min(2, count)]
    remaining = [i for i in legal if i not in chosen]
    if remaining and len(chosen) < count:
        chosen.extend(int(x) for x in rng.choice(remaining, min(count-len(chosen), len(remaining)), replace=False))
    return chosen


def local_advantages(reference_cost, branch_costs):
    costs = np.asarray(branch_costs, dtype=np.float64)
    if not math.isfinite(reference_cost) or reference_cost <= 0 or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Only completed finite positive costs are valid")
    return .01 * (reference_cost - costs)


def fit_gate(cases, source, teacher, fit_summary, probe_summary, top1):
    from onpolicy.utils.stage3_representation import risk_pass
    if len(cases) != 16 or set(teacher) != {c["path"] for c in cases}:
        raise ValueError("Fit gate requires every Fit16 case, including no-winner cases")
    potential = sum(source[c["path"]] - min(source[c["path"]], teacher[c["path"]]) for c in cases)
    gained = -fit_summary["delta_seconds"] * len(cases)
    recovery = gained / potential if potential > 1e-6 else 0.
    wins = sum(teacher[c["path"]] <= source[c["path"]] * .99 for c in cases)
    passed = (wins >= 8 and recovery >= .5 and top1 >= .8 and probe_summary["gain_fraction"] >= 0
              and risk_pass(probe_summary) and fit_summary["completion_rate"] == 1)
    return {"passed": passed, "cases_gt1pct": wins, "recovery": recovery,
            "teacher_potential_seconds_sum": potential, "beneficial_top1": top1,
            "scope": "gate for this local-learning protocol; not a theorem of learnability"}


def validate_group(rows, checkpoint_sha, mode):
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("One equal-weight group per distinct case required")
    for row in rows:
        if row["behavior_sha256"] != checkpoint_sha:
            raise ValueError("Stale behavior policy in a fresh training group")
        if mode == "local_rl" and row["kind"] != "local_on_policy":
            raise ValueError("Enumerated/forced search is not an on-policy local sample")
        if mode == "ppo" and row["kind"] != "full_on_policy":
            raise ValueError("Forced prefixes are not full-trajectory PPO samples")
        if not row.get("completed") or row["query_count"] <= 0:
            raise ValueError("Incomplete simulation group")


class TaskQueue:
    """One durable queue, many consumers. No implicit retry after a crash."""
    def __init__(self, root):
        self.root = Path(root)
        for state in ("pending", "running", "results", "failed"):
            (self.root / state).mkdir(parents=True, exist_ok=True)

    def submit(self, key, payload):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError("Unsafe task name")
        row = {"id": key, "payload": payload, "sha256": digest_json(payload), "submitted_unix": time.time()}
        with (self.root / "claim.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for state in ("pending", "running", "results", "failed"):
                if (self.root / state / f"{key}.json").exists():
                    raise FileExistsError(f"No implicit task overwrite: {key}")
            atomic_json(self.root / "pending" / f"{key}.json", row, overwrite=False)
        return key

    def claim(self, worker_id=None):
        with (self.root / "claim.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for path in sorted((self.root / "pending").glob("*.json")):
                pending = read_json(path)
                if pending["payload"].get("worker") not in (None, worker_id):
                    continue
                target = self.root / "running" / path.name
                os.rename(path, target)
                row = read_json(target)
                if row["sha256"] != digest_json(row["payload"]):
                    raise ValueError("Task identity mismatch")
                row.update(worker_pid=os.getpid(), started_unix=time.time())
                atomic_json(target, row)
                return row
        return None

    def finish(self, row, result=None, error=None):
        target = self.root / ("failed" if error else "results") / f"{row['id']}.json"
        atomic_json(target, {**row, "result": result, "error": error, "finished_unix": time.time()}, overwrite=False)
        (self.root / "running" / f"{row['id']}.json").unlink()

    def poll(self, key):
        bad = self.root / "failed" / f"{key}.json"
        if bad.exists():
            raise RuntimeError(f"Task {key} failed: {read_json(bad)['error']}")
        path = self.root / "results" / f"{key}.json"
        return read_json(path)["result"] if path.exists() else None

    def outstanding(self):
        return sum(len(list((self.root / state).glob("*.json"))) for state in ("pending", "running"))


def verify(manifest, *, code_root=None, inputs=True):
    from onpolicy.utils.stage3_research import code_changes, verify_cases
    if manifest["schema"] != SCHEMA or identity(manifest) != manifest["manifest_sha256"]:
        raise ValueError("Manifest identity changed")
    if any(manifest["contract"].get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError("Main study requires H2/F4/soft reservation")
    if digest_json(manifest["contract"]) != manifest["contract_sha256"]:
        raise ValueError("Environment contract changed")
    root = Path(code_root or manifest["execution"]["code_root"]).resolve()
    if code_changes(manifest["code"]["files"], root):
        raise ValueError("Frozen source snapshot changed")
    if inputs:
        import importlib.metadata
        for package, expected in manifest["execution"]["packages"].items():
            if importlib.metadata.version(package) != expected:
                raise ValueError(f"Runtime package changed: {package}")
        for path, sha in manifest["input_files"].items():
            if digest_file(path) != sha:
                raise ValueError(f"Frozen input changed: {path}")
        for key, cases in manifest["splits"].items():
            verify_cases(cases, training=key.startswith("train"))
    return code_changes(manifest["code"]["files"], manifest["execution"]["workspace_root"])
