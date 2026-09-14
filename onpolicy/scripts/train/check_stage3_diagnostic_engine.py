#!/usr/bin/env python3
"""Small live smoke check, not formal experimental evidence."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onpolicy.utils.stage3_research import (TRAIN_ROOT, SOURCE_BASELINE, case_record, read_json,
                                          atomic_json, Heartbeat)


def main():
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--unfreeze-shared", action="store_true")
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    with Heartbeat(args.output / "status.json", phase="smoke") as heartbeat:
        runner = ResearchEngine(width=2, freeze_shared=not args.unfreeze_shared)
        try:
            case = case_record(TRAIN_ROOT / "case_0004")
            trajectories = runner.rollout([case], [42], deterministic=True, retain=True, heartbeat=heartbeat)
            expected = read_json(SOURCE_BASELINE)["cases"][case["name"]]["makespan"]
            actual = trajectories[0]["makespan"]
            print(f"Source reproduction: actual={actual}, expected={expected}", flush=True)
            if abs(actual - expected) > .1:
                raise RuntimeError("Source C0 reproduction failed")
            metrics = runner.likelihood_metrics(trajectories[0])
            print(f"Likelihood metrics: {metrics}", flush=True)
            result = {"actual": actual, "expected": expected, "likelihood": metrics}
            if args.teacher:
                from onpolicy.scripts.train.stage3_diagnostic_worker import teacher_actions
                teacher = read_json(args.teacher)
                actions = teacher_actions(case, teacher)
                replay = runner.rollout([case], [42], deterministic=True, forced=[actions],
                                        retain=True, heartbeat=heartbeat)[0]
                result["teacher_replay"] = {"makespan": replay["makespan"],
                    "expected": teacher["makespan"], "likelihood": runner.likelihood_metrics(replay)}
                if abs(replay["makespan"] - teacher["makespan"]) > .1:
                    raise RuntimeError("Teacher action reconstruction failed")
                print(f"Teacher replay: {result['teacher_replay']}", flush=True)
                del replay
            if args.update:
                del trajectories
                group = runner.rollout([case] * 2, [20260913, 20260914], retain=True, heartbeat=heartbeat)
                result["update"] = runner.update(group, "leave_one_out", expected, epochs=2, heartbeat=heartbeat)
                print(f"Update: {result['update']}", flush=True)
                result["bc"] = runner.update([group[0]], "bc", epochs=1, heartbeat=heartbeat)
                print(f"BC update: {result['bc']}", flush=True)
            runner.save(args.output / "smoke.pt", diagnostic_only=True)
            atomic_json(args.output / "result.json", result)
        finally:
            runner.close()


if __name__ == "__main__":
    main()
