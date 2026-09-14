#!/usr/bin/env python3
"""Bounded same-state GPU repeatability probe; does not evaluate partial costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


class ProbeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=150)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sys.path.insert(0, manifest["execution"]["code_root"])
    import torch
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    from onpolicy.utils.stage3_research import atomic_json, digest_file
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest_file(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("C0 changed")
    engine = ResearchEngine(manifest["source"]["path"], width=2)
    original = engine.policy.get_actions
    result = {"purpose": "same-state FP32 GPU repeatability; no partial-trajectory cost reported",
              "calls": 0, "hidden_max_abs": 0., "action_mismatches": 0, "rng_mismatches": 0,
              "same_state_repeats": []}

    def checked(*inputs, **kwargs):
        before = torch.cuda.get_rng_state(engine.device)
        output = original(*inputs, **kwargs)
        after = torch.cuda.get_rng_state(engine.device)
        act_kwargs = {k: v for k, v in kwargs.items() if k != "return_decision_mask"}
        try:
            torch.cuda.set_rng_state(before, engine.device)
            action, hidden = engine.policy.act(*inputs, **act_kwargs)
            delta = float((hidden - output[3]).abs().max())
            result["calls"] += 1
            result["hidden_max_abs"] = max(result["hidden_max_abs"], delta)
            result["action_mismatches"] += int(not torch.equal(action, output[1]))
            result["rng_mismatches"] += int(not torch.equal(torch.cuda.get_rng_state(engine.device), after))
            if result["calls"] in (1, 95, args.calls) or (delta > 1e-6 and len(result["same_state_repeats"]) < 6):
                repeated = {"call": result["calls"], "cross_api_hidden_abs": delta,
                    "same_get_hidden_abs": [], "same_act_hidden_abs": [], "actions_all_equal": True}
                for _ in range(8):
                    torch.cuda.set_rng_state(before, engine.device)
                    again = original(*inputs, **kwargs)
                    repeated["same_get_hidden_abs"].append(float((again[3] - output[3]).abs().max()))
                    repeated["actions_all_equal"] &= torch.equal(again[1], output[1])
                    torch.cuda.set_rng_state(before, engine.device)
                    again_act, again_hidden = engine.policy.act(*inputs, **act_kwargs)
                    repeated["same_act_hidden_abs"].append(float((again_hidden - hidden).abs().max()))
                    repeated["actions_all_equal"] &= torch.equal(again_act, action)
                result["same_state_repeats"].append(repeated)
                atomic_json(args.output, result)
        finally:
            torch.cuda.set_rng_state(after, engine.device)
        if result["calls"] >= args.calls:
            raise ProbeComplete()
        return output

    engine.policy.get_actions = checked
    try:
        cases = manifest["cases"][0::4]
        engine.rollout(cases, [42] * len(cases), deterministic=True)
    except ProbeComplete:
        result["bounded_probe_completed"] = True
    finally:
        engine.close()
        atomic_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
