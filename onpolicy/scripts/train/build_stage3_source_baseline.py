#!/usr/bin/env python3
"""Evaluate C0 on all training cases and emit a verified paired baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.config.config import get_config  # noqa: E402
from onpolicy.scripts.train.prepare_stage3_source_relative_rl import (  # noqa: E402
    BASELINE_REFERENCE_KIND,
    BASELINE_SCHEMA_VERSION,
    DEFAULT_BASE_MANIFEST,
    DEFAULT_SOURCE,
    DEFAULT_TRAIN_DATASET,
    EXPECTED_MODEL_SHA256,
    EXPECTED_SOURCE_SHA256,
    SOURCE_EVAL_CONTRACT,
    atomic_json,
    dataset_contract,
    model_sha256,
    option_value,
    sha256_file,
    validate_source_baseline,
)
from onpolicy.scripts.train.train_hkbz import (  # noqa: E402
    make_eval_env,
    parse_args as parse_training_args,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StatusHeartbeat:
    def __init__(self, path: Path, interval_seconds: float = 30.0):
        self.path = path
        self.interval_seconds = float(interval_seconds)
        self.started_at = _utc_now()
        self.state: dict[str, object] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._loop,
            name="source-baseline-heartbeat",
            daemon=True,
        )

    def write(self, status: str, **extra: object) -> None:
        with self.lock:
            self.state.update(extra)
            payload = {
                "status": status,
                "timestamp_utc": _utc_now(),
                "heartbeat_timestamp_utc": _utc_now(),
                "started_at_utc": self.started_at,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                **self.state,
            }
        atomic_json(self.path, payload)

    def start(self, **extra: object) -> None:
        self.write("running", **extra)
        self.thread.start()

    def finish(self, status: str, **extra: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        self.write(status, **extra)

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.write("running")


def _assert_eval_contract(all_args: argparse.Namespace) -> None:
    observed = {
        "evaluation_tau": float(all_args.evaluation_tau),
        "global_feature_mode": str(all_args.global_feature_mode),
        "resource_policy": str(all_args.resource_policy),
        "plane_order_mode": str(all_args.plane_order_mode),
        "plane_pair_decoder": str(all_args.plane_pair_decoder),
        "device_future_intent_horizon": int(
            all_args.device_future_intent_horizon
        ),
        "device_future_intent_mode": str(all_args.device_future_intent_mode),
        "device_frontier_max_requests": int(
            all_args.device_frontier_max_requests
        ),
        "device_request_capacity_per_plane": int(
            all_args.device_request_capacity_per_plane
        ),
        "device_lookahead_reservation_mode": str(
            all_args.device_lookahead_reservation_mode
        ),
        "device_lookahead_safety_margin": float(
            all_args.device_lookahead_safety_margin
        ),
        "device_reservation_grace_seconds": float(
            all_args.device_reservation_grace_seconds
        ),
        "device_lookahead_dispatch": bool(all_args.device_lookahead_dispatch),
        "device_deadline_aware_dispatch": bool(
            all_args.device_deadline_aware_dispatch
        ),
        "resource_release_aware_eta": bool(
            all_args.resource_release_aware_eta
        ),
        "device_departure_lookahead": bool(
            all_args.device_departure_lookahead
        ),
    }
    if observed != SOURCE_EVAL_CONTRACT:
        raise ValueError(
            "Source checkpoint evaluation contract mismatch: "
            f"observed={observed}, expected={SOURCE_EVAL_CONTRACT}."
        )


def _case_map(evaluation: dict, dataset: dict[str, object]) -> dict[str, dict]:
    records = evaluation.get("cases")
    if not isinstance(records, list):
        raise ValueError("Evaluation artifact has no per-case records.")
    result: dict[str, dict] = {}
    expected_cases = dataset["cases"]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Evaluation contains a non-object case record.")
        case_name = Path(str(record.get("case_path", ""))).name
        if not case_name or case_name in result:
            raise ValueError(f"Invalid or duplicate evaluated case: {case_name!r}.")
        if case_name not in expected_cases:
            raise ValueError(f"Evaluation escaped the training split: {case_name}.")
        case_sha = str(record.get("case_sha256", ""))
        if case_sha != expected_cases[case_name]:
            raise ValueError(f"Evaluation lineage mismatch for {case_name}.")
        makespan = float(record.get("makespan", float("nan")))
        if not math.isfinite(makespan) or makespan <= 0.0:
            raise ValueError(f"Invalid C0 makespan for {case_name}: {makespan}.")
        if (
            not bool(record.get("completed", False))
            or bool(record.get("cycle_terminated", False))
            or bool(record.get("timeout", False))
        ):
            raise ValueError(f"C0 did not safely complete {case_name}.")
        result[case_name] = {
            "makespan": makespan,
            "case_sha256": case_sha,
        }
    if set(result) != set(expected_cases):
        missing = sorted(set(expected_cases) - set(result))[:12]
        raise ValueError(f"C0 evaluation missed training cases: {missing}.")
    return dict(sorted(result.items()))


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--label", default="source_checkpoint_train600")
    parser.add_argument("--expected-cases", type=int, default=600)
    parser.add_argument("--n-eval-rollout-threads", type=int, default=30)
    parser.add_argument("--eval-partition-seed", type=int, default=20260803)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    cli = parse_cli()
    base_path = cli.base_manifest.expanduser().resolve()
    source = cli.source_checkpoint.expanduser().resolve()
    dataset_dir = cli.dataset_dir.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    output = cli.output.expanduser().resolve()
    status_path = cli.status.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite source baseline: {output}")
    if not base_path.is_file() or not source.is_file() or not dataset_dir.is_dir():
        raise FileNotFoundError("Base manifest, source checkpoint, or dataset missing.")
    if cli.expected_cases <= 0 or cli.n_eval_rollout_threads <= 0:
        raise ValueError("Expected cases and evaluation workers must be positive.")
    if cli.n_eval_rollout_threads > cli.expected_cases:
        raise ValueError("Evaluation workers cannot exceed expected cases.")
    if (
        sha256_file(source) != EXPECTED_SOURCE_SHA256
        or model_sha256(source) != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("C0 checkpoint identity changed.")

    dataset = dataset_contract(dataset_dir)
    if int(dataset["case_count"]) != int(cli.expected_cases):
        raise ValueError(
            f"Training split has {dataset['case_count']} cases; expected "
            f"{cli.expected_cases}."
        )
    manifest = json.loads(base_path.read_text(encoding="utf-8"))
    command = [str(value) for value in manifest.get("command", ())]
    if len(command) < 2:
        raise ValueError("Base manifest has no training command.")
    if Path(option_value(command, "--checkpoint_dir")).resolve() != source:
        raise ValueError("Base manifest and requested C0 checkpoint differ.")

    heartbeat = StatusHeartbeat(status_path, cli.heartbeat_seconds)
    heartbeat.start(
        event="initializing",
        source_checkpoint=str(source),
        source_checkpoint_sha256=EXPECTED_SOURCE_SHA256,
        source_model_sha256=EXPECTED_MODEL_SHA256,
        dataset=str(dataset_dir),
        expected_cases=int(cli.expected_cases),
        evaluation_workers=int(cli.n_eval_rollout_threads),
        output=str(output),
    )
    started = time.monotonic()
    eval_envs = None
    runner = None
    try:
        all_args = parse_training_args(command[2:], get_config())
        all_args.use_wandb = False
        all_args.use_eval = True
        all_args.checkpoint_dir = str(source)
        all_args.selection_checkpoint_dir = None
        all_args.eval_dataset_dir = str(dataset_dir)
        manifest_path = dataset_dir.parent / "manifest.json"
        all_args.dataset_manifest = (
            str(manifest_path.resolve()) if manifest_path.is_file() else ""
        )
        all_args.max_eval_cases = int(cli.expected_cases)
        all_args.eval_case_offset = 0
        all_args.eval_partition_seed = int(cli.eval_partition_seed)
        all_args.eval_partition_stratify_by = ""
        all_args.evaluation_tau = 0.3
        all_args.selection_metric = "raw"
        all_args.n_eval_rollout_threads = int(cli.n_eval_rollout_threads)
        all_args.n_rollout_threads = int(cli.n_eval_rollout_threads)
        all_args.plane_bc_pretrain_epochs = 0
        all_args.device_bc_pretrain_epochs = 0
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
        _assert_eval_contract(all_args)

        torch.multiprocessing.set_sharing_strategy(
            all_args.torch_mp_sharing_strategy
        )
        torch.manual_seed(all_args.seed)
        torch.cuda.manual_seed_all(all_args.seed)
        np.random.seed(all_args.seed)
        if all_args.cuda and torch.cuda.is_available():
            device = torch.device(str(all_args.device))
            if float(all_args.cuda_memory_fraction) > 0.0:
                torch.cuda.set_per_process_memory_fraction(
                    float(all_args.cuda_memory_fraction), device=device
                )
            torch.set_num_threads(all_args.n_training_threads)
            if all_args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        else:
            device = torch.device("cpu")
            torch.set_num_threads(all_args.n_training_threads)

        output_dir.mkdir(parents=True, exist_ok=False)
        heartbeat.write("running", event="evaluating", device=str(device))
        eval_envs, case_counts = make_eval_env(all_args)
        with Path(all_args.ac_config).open("r", encoding="utf-8") as handle:
            ac_config = yaml.safe_load(handle)
        from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner

        runner = HKBZ_Runner({
            "all_args": all_args,
            "envs": eval_envs,
            "eval_envs": eval_envs,
            "device": device,
            "run_dir": output_dir,
            "ac_config": ac_config,
            "num_agents": (
                all_args.max_agent_num + all_args.max_device_num
            ),
            "num_envs": max(case_counts),
            "eval_case_counts": case_counts,
            "eval_env_factory": None,
            "release_eval_envs_after_eval": False,
            "evaluation_only": True,
        })
        runner.eval(evaluation_label=cli.label)
        evaluation_path = output_dir / "evaluations" / f"{cli.label}.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        summary = evaluation.get("summary", {})
        if (
            int(summary.get("eval_case_count", -1)) != cli.expected_cases
            or float(summary.get("eval_completion_rate", 0.0)) != 1.0
            or int(summary.get("eval_timeout_count", -1)) != 0
            or int(summary.get("eval_cycle_count", -1)) != 0
        ):
            raise RuntimeError(f"Unsafe or incomplete C0 evaluation: {summary}.")
        cases = _case_map(evaluation, dataset)
        baseline = {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "reference_kind": BASELINE_REFERENCE_KIND,
            "created_at_utc": _utc_now(),
            "source_checkpoint": {
                "path": str(source),
                "sha256": EXPECTED_SOURCE_SHA256,
                "model_sha256": EXPECTED_MODEL_SHA256,
            },
            "policy_environment_contract": SOURCE_EVAL_CONTRACT,
            "evaluation": {
                "artifact": str(evaluation_path.resolve()),
                "artifact_sha256": sha256_file(evaluation_path),
                "seed": int(all_args.seed),
                "partition_seed": int(cli.eval_partition_seed),
                "deterministic": True,
                "case_count": int(cli.expected_cases),
                "completion_rate": 1.0,
                "timeout_count": 0,
                "cycle_count": 0,
                "mean_makespan": float(summary["eval_raw_makespan"]),
            },
            "dataset": {
                "path": dataset["path"],
                "case_count": dataset["case_count"],
                "fingerprint_sha256": dataset["fingerprint_sha256"],
            },
            "cases": cases,
        }
        atomic_json(output, baseline)
        verified = validate_source_baseline(
            output,
            source_checkpoint=source,
            dataset_dir=dataset_dir,
        )
        elapsed = time.monotonic() - started
        heartbeat.finish(
            "completed",
            event="completed",
            elapsed_seconds=float(elapsed),
            evaluation_artifact=str(evaluation_path.resolve()),
            baseline_sha256=verified["sha256"],
            case_count=verified["case_count"],
            mean_makespan=verified["mean_makespan"],
        )
        print(
            "[SourceBaseline] "
            f"completed cases={verified['case_count']} "
            f"mean={verified['mean_makespan']:.6f} "
            f"output={output}",
            flush=True,
        )
        return 0
    except BaseException as error:
        heartbeat.finish(
            "failed",
            event="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            elapsed_seconds=float(time.monotonic() - started),
        )
        raise
    finally:
        if eval_envs is not None:
            eval_envs.close()
        writer = getattr(runner, "writter", None)
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
