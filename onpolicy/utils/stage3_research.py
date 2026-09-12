"""Auditable, dependency-light contracts for the Stage3 diagnostic study.

This module deliberately does not import the environment or initialize CUDA.
Old Stage3 launchers remain unchanged; the new study has an explicit protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA = "stage3-diagnostic-v1"
ROOT = Path(__file__).resolve().parents[2]
# Execution code may be an isolated copy; data/checkpoints stay in the workspace.
WORKSPACE_ROOT = Path(os.environ.get("HKBZ_STAGE3_WORKSPACE_ROOT", ROOT)).resolve()
TRAIN_ROOT = WORKSPACE_ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"
EVAL_ROOT = WORKSPACE_ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint"
SOURCE = WORKSPACE_ROOT / ("onpolicy/scripts/results/HKBZ/simple/gnn_mappo/"
    "stage2_adaptive_trust_20260830_r4_wave1_T2_backtrack_adaptive_soft_bc_seed1/"
    "run1/models/checkpoint_Epoch3.pt")
SOURCE_SHA = "41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030"
BASE_MANIFEST = WORKSPACE_ROOT / ("result/hkbz_train_logs/stage3_source_relative_rl_4090_20260904_r1/"
    "commands/wave1_g0_U0K0R0.json")
SOURCE_BASELINE = WORKSPACE_ROOT / ("result/hkbz_train_logs/stage3_source_relative_rl_4090_20260904_r1/"
    "artifacts/source_checkpoint_train600.json")
SOURCE_EVAL = WORKSPACE_ROOT / ("onpolicy/scripts/results/HKBZ/simple/gnn_mappo/"
    "stage3_source_relative_rl_4090_20260904_r1_wave1_U0K0R0_seed3/"
    "run1/evaluations/pre_ppo.json")
CASE_FILES = ("job.json", "fixed_resources.json", "mobile_resources.json",
              "sites.json", "flights.json", "metadata.json")
ARMS = {"R0": "value", "R1": "source", "R2": "leave_one_out", "R3": "leave_one_out"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # link is atomic and refuses to replace an existing publication.
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def case_record(path):
    path = Path(path).resolve()
    metadata = read_json(path / "metadata.json")
    files = {name: digest_file(path / name) for name in CASE_FILES}
    recorded = metadata.get("fingerprints", {}).get("files", {})
    # Dataset provenance hashes canonical UTF-8 JSON, not pretty-printed file bytes.
    semantic = {name: hashlib.sha256(json.dumps(read_json(path / name), sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() for name in recorded}
    if semantic != recorded:
        raise ValueError(f"Case files differ from their recorded lineage: {path}")
    return {"path": str(path), "name": path.name,
            "case_sha256": metadata.get("case_sha256", metadata.get("fingerprints", {}).get("case_sha256")),
            "content_sha256": digest_json(files), "files": files,
            "profile": metadata["profile"], "distribution": metadata["distribution"]}


def verify_cases(records, *, training=False):
    seen = set()
    for record in records:
        path = Path(record["path"]).resolve()
        if training and path.parent != TRAIN_ROOT.resolve():
            raise ValueError(f"Training data escaped Train600: {path}")
        if str(path) in seen:
            raise ValueError(f"Duplicate case in immutable split: {path}")
        seen.add(str(path))
        if case_record(path) != record:
            raise ValueError(f"Case content or metadata changed: {path}")


def stratified_cases(records, counts, seed):
    """Outcome-independent, deterministic ordering, balanced over profiles."""
    result = []
    for distribution, count in counts.items():
        candidates = [r for r in records if r["distribution"] == distribution]
        profiles = sorted({r["profile"] for r in candidates})
        queues = {p: sorted([r for r in candidates if r["profile"] == p],
                            key=lambda r: digest_json([seed, r["content_sha256"]])) for p in profiles}
        ordered = []
        while any(queues.values()):
            for profile in profiles:
                if queues[profile]:
                    ordered.append(queues[profile].pop(0))
        if len(ordered) < count:
            raise ValueError(f"Not enough unique {distribution} cases: {len(ordered)} < {count}")
        result.extend(ordered[:count])
    return sorted(result, key=lambda r: digest_json([seed, "order", r["content_sha256"]]))


def trajectory_seed(seed, case_hash, visit, replica):
    return int(digest_json([seed, case_hash, visit, replica])[:8], 16) % (2**31 - 1)


def cost_baselines(costs, mode, source_cost=None):
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Only finite, positive, completed trajectory costs are admissible")
    if mode == "source":
        if source_cost is None or not math.isfinite(source_cost) or source_cost <= 0:
            raise ValueError("Source baseline is missing or invalid")
        return np.full_like(values, source_cost)
    if mode == "leave_one_out":
        if len(values) < 2:
            raise ValueError("Leave-one-out requires at least two independent trajectories")
        return (values.sum() - values) / (len(values) - 1)
    raise ValueError(f"Unsupported cost baseline: {mode}")


def best_of_n(costs, source_cost, prefixes=(1, 4, 8, 16, 32)):
    values = np.asarray(costs, dtype=np.float64)
    cost_baselines(values, "source", source_cost)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "better_than_source_fraction": float(np.mean(values < source_cost - 1e-6)),
            "prefix_best": {str(n): float(values[:n].min()) for n in prefixes if n <= len(values)}}


def paired_summary(records, source_costs):
    if not records or any(not r.get("completed", False) for r in records):
        raise ValueError("Partial/failed evaluations cannot be ranked")
    names = [r["case_id"] for r in records]
    if len(set(names)) != len(names):
        raise ValueError("Repeated trajectories are not independent evaluation cases")
    costs = np.asarray([r["makespan"] for r in records], dtype=np.float64)
    base = np.asarray([source_costs[name] for name in names], dtype=np.float64)
    if not np.isfinite(costs).all() or not np.isfinite(base).all() or (base <= 0).any() or (costs <= 0).any():
        raise ValueError("Invalid paired costs")
    delta = costs - base
    tail_n = max(1, math.ceil(len(costs) * 0.1))
    result = {"case_count": len(records), "completion_rate": 1.0,
              "makespan": float(costs.mean()), "source_makespan": float(base.mean()),
              "delta_seconds": float(delta.mean()), "gain_fraction": float(1 - costs.mean() / base.mean()),
              "win_fraction": float(np.mean(costs < base * 0.999)),
              "regression_over_5pct_fraction": float(np.mean(costs > base * 1.05)),
              "tail_makespan": float(np.sort(costs)[-tail_n:].mean()),
              "source_tail_makespan": float(np.sort(base)[-tail_n:].mean())}
    result["profiles"] = {}
    for profile in sorted({r["profile"] for r in records}):
        selected = np.asarray([r["profile"] == profile for r in records])
        result["profiles"][profile] = {"count": int(selected.sum()),
            "delta_seconds": float(delta[selected].mean()),
            "regression_fraction": float(costs[selected].mean() / base[selected].mean() - 1)}
    result["distributions"] = {}
    for distribution in sorted({r.get("distribution", "unknown") for r in records}):
        selected = np.asarray([r.get("distribution", "unknown") == distribution for r in records])
        result["distributions"][distribution] = {"count": int(selected.sum()),
            "regression_fraction": float(costs[selected].mean() / base[selected].mean() - 1)}
    # Paired, profile-stratified resampling. This is conditional on ONE seed;
    # it must not be advertised as uncertainty across training seeds.
    rng = np.random.default_rng(20260905)
    resampled = []
    for profile in sorted(result["profiles"]):
        indices = np.flatnonzero([r["profile"] == profile for r in records])
        resampled.append(rng.choice(indices, size=(2000, len(indices)), replace=True))
    indices = np.concatenate(resampled, axis=1)
    gains = 1 - costs[indices].mean(axis=1) / base[indices].mean(axis=1)
    result["paired_case_bootstrap_gain_ci95"] = np.quantile(gains, [.025, .975]).tolist()
    result["ci_scope"] = "case uncertainty conditional on this trained seed; not cross-seed evidence"
    return result


def safety_gate(summary):
    stress = summary["distributions"].get("ood_stress", {}).get("regression_fraction", float("inf"))
    joint = summary["profiles"].get("stress_joint", {}).get("regression_fraction", float("inf"))
    return (summary["completion_rate"] == 1.0 and stress <= .005 and joint <= .005
            and summary["tail_makespan"] <= summary["source_tail_makespan"] * 1.01)


def pilot_gate(evaluations):
    # Endpoint must itself pass; an early lucky checkpoint cannot select a winner.
    return len(evaluations) >= 2 and all(safety_gate(e) and e["gain_fraction"] >= .01
                                       for e in evaluations[-2:])


def code_changes(files, root):
    """Missing/changed workspace files are reportable, not process failures."""
    changes = {}
    for relative, expected in files.items():
        path = Path(root) / relative
        try:
            actual = digest_file(path)
        except OSError:
            actual = None
        if actual != expected:
            changes[relative] = {"expected": expected, "actual": actual}
    return changes


def protocol_identity(manifest):
    """Identity of the study, independent of execution attempts and source copies."""
    return digest_json({key: manifest[key] for key in ("schema", "source", "splits",
        "source_costs", "contract", "contract_sha256", "resources", "diagnostic",
        "training", "gates", "timeouts_seconds")})


def verify_protocol(manifest):
    if digest_file(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("Frozen C0 checkpoint changed")
    execution = manifest.get("execution", {})
    code_root = Path(execution.get("code_root", ROOT)).resolve()
    if execution and code_root != ROOT.resolve():
        raise ValueError("Stage3 must execute from its declared source snapshot")
    changes = code_changes(manifest["code"]["files"], code_root)
    if changes:
        raise ValueError(f"Frozen implementation changed: {next(iter(changes))}")
    if manifest.get("contract_sha256") != digest_json(manifest["contract"]):
        raise ValueError("Environment contract identity changed")
    if "protocol_sha256" in execution and protocol_identity(manifest) != execution["protocol_sha256"]:
        raise ValueError("Experiment configuration/data identity changed")
    for path, sha in execution.get("input_files", {}).items():
        if digest_file(path) != sha:
            raise ValueError(f"Frozen experiment input changed: {path}")
    for name, records in manifest["splits"].items():
        verify_cases(records, training=name.startswith("train"))
    # Other experiments own the shared working tree. Never kill this isolated
    # study because they edit it; callers publish this as an advisory instead.
    if execution.get("workspace_code_policy") == "advisory":
        return code_changes(manifest["code"]["files"], execution["workspace_root"])
    return {}


def validate_sample_trajectories(trajectories, case, *, complete=False):
    count = len(trajectories)
    if count > 32 or count % 8 or (complete and count != 32):
        raise ValueError("Sample recovery requires complete K=8 groups (32 for a finished case)")
    for replica, trajectory in enumerate(trajectories):
        if (trajectory["case_id"] != case["path"] or not trajectory["completed"]
                or trajectory["seed"] != trajectory_seed(20260909, case["content_sha256"], 0, replica)
                or not math.isfinite(trajectory["makespan"]) or trajectory["makespan"] <= 0
                or len(trajectory["actions"]) != trajectory["steps"]):
            raise ValueError("Sample recovery case/seed/completion/action identity mismatch")


def half_cpu_topology(rows):
    """Select half the physical cores, keeping every SMT sibling together."""
    cores = {}
    for cpu, core, socket, node in rows:
        cores.setdefault((socket, core, node), []).append(cpu)
    if len(cores) < 2:
        raise ValueError("Need at least two physical CPU cores")
    chosen = sorted(cores, key=lambda key: (key[2], key[0], key[1]))[:len(cores) // 2]
    return [sorted(cores[key]) for key in chosen]


class Heartbeat:
    def __init__(self, path, **initial):
        self.path = Path(path)
        self.state = {"schema": SCHEMA, "pid": os.getpid(), "started_unix": time.time(), **initial}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            atomic_json(self.path, {**self.state, "heartbeat_unix": time.time()})

    def _loop(self):
        while not self.stop.wait(30):
            self.update()

    def __enter__(self):
        self.update(status="running")
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join(timeout=2)
        self.update(status="failed" if kind else "completed", error=str(value) if kind else None)


class EvaluationQueue:
    """Durable nonblocking submit/poll protocol; single evaluator claims jobs.

    A crashed running job is marked failed, never silently replayed. The
    coordinator may make an explicit, recorded recovery decision later.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        for name in ("pending", "running", "results", "failed", "cache"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def identity(request):
        return digest_json({key: request[key] for key in
            ("checkpoint_sha256", "cases_sha256", "contract_sha256", "code_sha256", "tau", "seed")})

    def submit(self, request):
        request = dict(request)
        request_id = str(request["request_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", request_id):
            raise ValueError("Unsafe request ID")
        checkpoint = Path(request["checkpoint"]).resolve()
        if digest_file(checkpoint) != request["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed before queue publication")
        if digest_json(request["cases"]) != request["cases_sha256"]:
            raise ValueError("Case set identity mismatch")
        request.update(cache_key=self.identity(request), submitted_unix=time.time())
        for state in ("pending", "running", "results", "failed"):
            if (self.root / state / f"{request_id}.json").exists():
                raise FileExistsError(f"Request already published: {request_id}")
        atomic_json(self.root / "pending" / f"{request_id}.json", request, overwrite=False)
        return request_id

    def poll(self, request_id):
        for state in ("results", "failed"):
            path = self.root / state / f"{request_id}.json"
            if path.exists():
                result = read_json(path)
                if result["request_id"] != request_id:
                    raise ValueError("Out-of-order response identity corrupted")
                if state == "failed":
                    raise RuntimeError(f"Validator failed: {result}")
                return result
        return None

    def claim(self):
        for path in sorted((self.root / "pending").glob("*.json"), key=lambda p: p.stat().st_mtime_ns):
            target = self.root / "running" / path.name
            try:
                os.rename(path, target)
            except FileNotFoundError:
                continue
            return target, read_json(target)
        return None

    def finish(self, path, result, error=None):
        request = read_json(path)
        payload = {"request_id": request["request_id"], "checkpoint_sha256": request["checkpoint_sha256"],
                   "training_episodes": request["training_episodes"], "finished_unix": time.time(),
                   "ok": error is None, "error": error, "evaluation": result}
        payload.update({key: request[key] for key in
            ("code_sha256", "cases_sha256", "contract_sha256", "cache_key")})
        atomic_json(self.root / ("results" if error is None else "failed") / Path(path).name,
                    payload, overwrite=False)
        Path(path).unlink()

    def fail_interrupted(self):
        for path in (self.root / "running").glob("*.json"):
            self.finish(path, None, "Evaluator interrupted; automatic retry disabled")
