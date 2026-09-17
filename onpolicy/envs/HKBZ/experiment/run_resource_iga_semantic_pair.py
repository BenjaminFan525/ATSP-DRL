#!/usr/bin/env python3
"""Hand legacy resource-IGA shards to matched lookahead shards, then analyze."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EVALUATOR = Path(__file__).with_name("evaluate_resource_iga_ablation.py")
ANALYZER = Path(__file__).with_name("analyze_resource_iga_ablation.py")
SEMANTIC_ANALYZER = Path(__file__).with_name(
    "compare_resource_dispatch_semantics.py"
)
DEFAULT_LEGACY = (
    ROOT / "result/hkbz_train_logs/resource_iga_role_ablation_20260811_r1"
)
DEFAULT_LOOKAHEAD = (
    ROOT
    / "result/hkbz_train_logs/"
    "resource_iga_role_ablation_lookahead_20260811_r1"
)
DATASET = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/"
    "fjsp_v3_resource_joint_eval_s20260811/joint/tune"
)
SHARDS = (
    (0, 0, "0-11,72-83"),
    (1, 0, "12-23,84-95"),
    (2, 0, "24-35,96-107"),
    (3, 1, "36-47,108-119"),
    (4, 1, "48-59,120-131"),
    (5, 1, "60-71,132-143"),
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _worker_completed(output_dir: Path, shard: int) -> bool:
    path = output_dir / "workers" / f"shard_{shard:02d}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and int(payload.get("shard_index", -1)) == shard
        and int(payload.get("shard_count", -1)) == len(SHARDS)
    )


def _shard_command(output_dir: Path, shard: int, cpu_set: str) -> list[str]:
    return [
        "taskset",
        "--cpu-list",
        cpu_set,
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--dataset-dir",
        str(DATASET),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda:0",
        "--evaluation-tau",
        "0.3",
        "--population",
        "6",
        "--generations",
        "6",
        "--seed",
        "20260811",
        "--max-steps",
        "4000",
        "--shard-index",
        str(shard),
        "--shard-count",
        str(len(SHARDS)),
        "--device-lookahead-dispatch",
        "--device-lookahead-safety-margin",
        "60",
    ]


def _environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "MPLCONFIGDIR": "/tmp/hkbz-matplotlib",
            "PYTORCH_CUDA_ALLOC_CONF": (
                "expandable_segments:True,max_split_size_mb:512"
            ),
        }
    )
    return environment


def _run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "MPLCONFIGDIR": "/tmp/hkbz-matplotlib"},
        )


def _postprocess(legacy_dir: Path, lookahead_dir: Path) -> None:
    common = [
        sys.executable,
        str(EVALUATOR),
        "--dataset-dir",
        str(DATASET),
        "--population",
        "6",
        "--generations",
        "6",
        "--seed",
        "20260811",
        "--summarize-only",
    ]
    _run_checked(
        [*common, "--output-dir", str(legacy_dir)],
        legacy_dir / "logs/postprocess.log",
    )
    _run_checked(
        [
            *common,
            "--output-dir",
            str(lookahead_dir),
            "--device-lookahead-dispatch",
            "--device-lookahead-safety-margin",
            "60",
        ],
        lookahead_dir / "logs/postprocess.log",
    )
    _run_checked(
        [sys.executable, str(ANALYZER), "--output-dir", str(legacy_dir)],
        legacy_dir / "logs/postprocess.log",
    )
    _run_checked(
        [sys.executable, str(ANALYZER), "--output-dir", str(lookahead_dir)],
        lookahead_dir / "logs/postprocess.log",
    )
    _run_checked(
        [
            sys.executable,
            str(SEMANTIC_ANALYZER),
            "--legacy-dir",
            str(legacy_dir),
            "--lookahead-dir",
            str(lookahead_dir),
            "--output-dir",
            str(lookahead_dir / "lookahead_comparison"),
        ],
        lookahead_dir / "logs/postprocess.log",
    )


def orchestrate(
    legacy_dir: Path,
    lookahead_dir: Path,
    *,
    poll_seconds: float,
) -> int:
    lookahead_dir.mkdir(parents=True, exist_ok=True)
    (lookahead_dir / "logs").mkdir(parents=True, exist_ok=True)
    Path("/tmp/hkbz-matplotlib").mkdir(parents=True, exist_ok=True)
    active = {}
    failures = {}
    launched = set()
    last_heartbeat = 0.0
    started = time.time()

    while True:
        legacy_done = {
            shard for shard, _, _ in SHARDS
            if _worker_completed(legacy_dir, shard)
        }
        lookahead_done = {
            shard for shard, _, _ in SHARDS
            if _worker_completed(lookahead_dir, shard)
        }
        for shard, gpu, cpu_set in SHARDS:
            if (
                shard not in legacy_done
                or shard in lookahead_done
                or shard in active
                or shard in failures
            ):
                continue
            command = _shard_command(lookahead_dir, shard, cpu_set)
            log_path = lookahead_dir / "logs" / f"shard_{shard:02d}.log"
            log = log_path.open("a", encoding="utf-8")
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_environment(gpu),
                start_new_session=True,
            )
            active[shard] = (process, log)
            launched.add(shard)
            print(
                f"[SemanticPair] launched lookahead shard={shard} "
                f"gpu={gpu} cpus={cpu_set} pid={process.pid}",
                flush=True,
            )

        for shard, (process, log) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log.close()
            del active[shard]
            if return_code != 0 or not _worker_completed(lookahead_dir, shard):
                failures[shard] = int(return_code)
                print(
                    f"[SemanticPair] lookahead shard={shard} failed "
                    f"return_code={return_code}",
                    flush=True,
                )
            else:
                print(
                    f"[SemanticPair] lookahead shard={shard} completed",
                    flush=True,
                )

        lookahead_done = {
            shard for shard, _, _ in SHARDS
            if _worker_completed(lookahead_dir, shard)
        }
        status = {
            "status": (
                "failed" if failures else
                "completed" if len(lookahead_done) == len(SHARDS) else
                "running"
            ),
            "legacy_completed_shards": sorted(legacy_done),
            "lookahead_launched_shards": sorted(launched),
            "lookahead_active_shards": sorted(active),
            "lookahead_completed_shards": sorted(lookahead_done),
            "failures": failures,
            "started_unix_time": started,
            "updated_unix_time": time.time(),
        }
        _atomic_json(lookahead_dir / "orchestrator_status.json", status)

        now = time.time()
        if now - last_heartbeat >= 60.0:
            print(
                "[SemanticPair] heartbeat "
                f"legacy={len(legacy_done)}/6 "
                f"lookahead={len(lookahead_done)}/6 "
                f"active={sorted(active)}",
                flush=True,
            )
            last_heartbeat = now
        if failures and not active:
            return 1
        if len(lookahead_done) == len(SHARDS):
            break
        time.sleep(poll_seconds)

    print("[SemanticPair] all shards complete; postprocessing", flush=True)
    _postprocess(legacy_dir, lookahead_dir)
    status["status"] = "completed_with_analysis"
    status["updated_unix_time"] = time.time()
    _atomic_json(lookahead_dir / "orchestrator_status.json", status)
    print("[SemanticPair] paired analysis complete", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--lookahead-dir", type=Path, default=DEFAULT_LOOKAHEAD)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0.0:
        parser.error("poll-seconds must be positive")
    if args.dry_run:
        for shard, gpu, cpu_set in SHARDS:
            print(
                json.dumps(
                    {
                        "shard": shard,
                        "gpu": gpu,
                        "cpu_set": cpu_set,
                        "command": _shard_command(
                            args.lookahead_dir.resolve(), shard, cpu_set
                        ),
                    }
                )
            )
        return 0
    return orchestrate(
        args.legacy_dir.resolve(),
        args.lookahead_dir.resolve(),
        poll_seconds=float(args.poll_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
