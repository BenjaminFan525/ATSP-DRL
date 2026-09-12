#!/usr/bin/env python3
"""Validate and exec one CPU-isolated Stage3 credit-factorial manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onpolicy.scripts.train.prepare_stage3_credit_factorial import (  # noqa: E402
    ROOT,
    atomic_json,
    sha256_file,
    validate_manifest,
)


def parse_cpu_set(raw: str) -> set[int]:
    cpus: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = token.split("-", 1)
            start, stop = int(first), int(last)
            if start > stop:
                raise ValueError(f"Reversed CPU range: {token}")
            cpus.update(range(start, stop + 1))
        else:
            cpus.add(int(token))
    if not cpus:
        raise ValueError("CPU affinity cannot be empty.")
    return cpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-gpu", type=int, required=True)
    parser.add_argument("--expected-cpu-affinity", required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--start-delay", type=float, default=0.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if int(manifest["physical_gpu"]) != args.expected_gpu:
        raise ValueError("CLI GPU and immutable manifest disagree.")
    if str(manifest["cpu_affinity"]) != args.expected_cpu_affinity:
        raise ValueError("CLI CPU set and immutable manifest disagree.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(args.expected_gpu):
        raise RuntimeError(
            f"Expected CUDA_VISIBLE_DEVICES={args.expected_gpu}, got {visible!r}."
        )
    requested_cpus = parse_cpu_set(args.expected_cpu_affinity)
    observed_cpus = set(os.sched_getaffinity(0))
    if observed_cpus != requested_cpus:
        raise RuntimeError(
            "Runtime CPU affinity differs from manifest: "
            f"observed={sorted(observed_cpus)}, "
            f"expected={sorted(requested_cpus)}."
        )
    for variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable}=1 is required for CPU isolation.")
    if Path(os.environ.get("PYTHON", manifest["python"])).resolve() != Path(
        manifest["python"]
    ).resolve():
        raise RuntimeError("Runtime Python differs from immutable manifest.")
    print(
        f"[CreditRun] validated {manifest['phase']} {manifest['arm']} "
        f"GPU{args.expected_gpu} CPUs={args.expected_cpu_affinity}",
        flush=True,
    )
    if args.check_only:
        return
    expected_parent = Path(manifest["expected_result_parent"])
    if expected_parent.exists():
        raise FileExistsError(
            f"Refusing to mix with an existing experiment: {expected_parent}"
        )
    if args.start_delay < 0.0 or args.start_delay > 300.0:
        raise ValueError("start-delay must be in [0, 300] seconds.")
    if args.start_delay:
        time.sleep(args.start_delay)
    if args.record is not None:
        atomic_json(args.record.expanduser().resolve(), {
            "status": "exec",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "phase": manifest["phase"],
            "arm": manifest["arm"],
            "physical_gpu": args.expected_gpu,
            "cpu_affinity": args.expected_cpu_affinity,
            "observed_cpus": sorted(observed_cpus),
            "source_stage2": manifest["source_stage2"],
        })
    os.chdir(ROOT)
    command = [str(value) for value in manifest["command"]]
    os.execve(command[0], command, dict(os.environ))


if __name__ == "__main__":
    main()
