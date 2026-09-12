#!/usr/bin/env python3
"""One GPU, one model, durable asynchronous validation for all Stage3 arms."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onpolicy.utils.stage3_research import (EvaluationQueue, Heartbeat, read_json, atomic_json,
    digest_file, digest_json, verify_protocol, verify_cases, paired_summary)
from onpolicy.scripts.train.stage3_diagnostic_worker import evaluate, engine


def serve(manifest):
    queue = EvaluationQueue(Path(manifest["root"]) / "validator")
    verify_protocol(manifest)
    # No automatic retry or pretending interrupted jobs completed.
    queue.fail_interrupted()
    runner = None
    with Heartbeat(queue.root / "status.json", phase="validator", completed_requests=0) as heartbeat:
        completed = 0
        try:
            while True:
                claim = queue.claim()
                if claim is None:
                    if (queue.root / "STOP").exists():
                        return
                    time.sleep(2)
                    continue
                path, request = claim
                try:
                    verify_protocol(manifest)
                    verify_cases(request["cases"])
                    if request["code_sha256"] != manifest["code"]["sha256"] or request["contract_sha256"] != manifest["contract_sha256"]:
                        raise ValueError("Request implementation/environment contract differs")
                    if digest_file(request["checkpoint"]) != request["checkpoint_sha256"]:
                        raise ValueError("Immutable checkpoint changed before evaluation")
                    if digest_json(request["cases"]) != request["cases_sha256"] or queue.identity(request) != request["cache_key"]:
                        raise ValueError("Request/cache identity mismatch")
                    if request["tau"] != .3 or request["seed"] != 42:
                        raise ValueError("Evaluation temperature/seed differs")
                    allowed = {r["path"] for r in manifest["splits"]["tune"]}
                    if any(r["path"] not in allowed for r in request["cases"]):
                        raise ValueError("Automatic validator is Tune-only; no Gate or Finalblind access")
                    heartbeat.update(event="evaluate", request_id=request["request_id"],
                                     checkpoint_sha256=request["checkpoint_sha256"],
                                     request_started_unix=time.time())
                    cache_path = queue.root / "cache" / f"{request['cache_key']}.json"
                    if cache_path.exists():
                        cached = read_json(cache_path)
                        if cached["cache_key"] != request["cache_key"]:
                            raise ValueError("Durable cache corrupted")
                        result = cached["evaluation"]
                    else:
                        if runner is None:
                            runner = engine(checkpoint=request["checkpoint"], width=8)
                        else:
                            runner.load(request["checkpoint"])
                        records = evaluate(runner, request["cases"], heartbeat)
                        result = {"cases": records, "summary": paired_summary(records, manifest["source_costs"])}
                        atomic_json(cache_path, {"cache_key": request["cache_key"], "evaluation": result}, overwrite=False)
                    queue.finish(path, result)
                    completed += 1
                    heartbeat.update(event="idle", completed_requests=completed, request_id=None,
                                     request_started_unix=None)
                except Exception:
                    queue.finish(path, None, traceback.format_exc())
                    raise
        finally:
            if runner is not None:
                runner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    serve(read_json(parser.parse_args().manifest))
