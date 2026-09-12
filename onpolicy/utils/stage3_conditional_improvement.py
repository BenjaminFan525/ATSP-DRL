"""Pre-registered contracts for the conditional Stage3 study (no CUDA)."""
from __future__ import annotations

import math

import numpy as np

from onpolicy.utils.stage3_local_improvement import HISTORY, ENVIRONMENT, SEED
from onpolicy.utils.stage3_representation import risk_pass
from onpolicy.utils.stage3_research import digest_json

STUDY = "stage3-conditional-improvement-v1"
ARCHITECTURES = ("score", "conditional", "conditional_pair")
PAIR_FEATURES = ("release_eta_hours", "travel_eta_hours", "arrival_minus_lead_hours",
                 "legal_competing_requests_log")
FIT_ENDPOINTS = (50, 200, 500, 1000, 2000)


def meaningful_epsilon(reference_cost):
    return max(5., .001 * float(reference_cost))


def cost_labels(state, *, meaningful=False):
    """Coalesce repeated on-policy draws without inventing unobserved costs."""
    costs = {}
    for action, cost in zip(state["actions"], state["costs"]):
        action, cost = int(action), float(cost)
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("Only finite completed terminal costs are labels")
        if action in costs and abs(costs[action]-cost) > 1e-5:
            raise ValueError("Same action/reference has inconsistent deterministic costs")
        costs[action] = cost
    if not costs:
        raise ValueError("Empty cost labels")
    epsilon = meaningful_epsilon(state["reference_cost"]) if meaningful else 1e-6
    return costs, epsilon


def metric_rank(metrics):
    """Train-cache-only selection, never consult Probe/Tune outcomes here."""
    return (metrics["meaningful"]["beneficial_hit_rate"],
            -metrics["meaningful"]["unknown_actions"],
            -metrics["meaningful"]["regret_ratio"],
            metrics["legacy"]["beneficial_hit_rate"], -metrics["loss"])


def learnability_gate(metrics, single=None):
    m = metrics["meaningful"]
    enough = m["beneficial_states"] > 0 and m["unknown_actions"] == 0
    fitted = enough and m["beneficial_hit_rate"] >= .8 and m["regret_ratio"] <= .2
    single_ok = (single is None or (single["meaningful"]["beneficial_states"] > 0
                 and single["meaningful"]["beneficial_hit_rate"] >= .95
                 and single["meaningful"]["unknown_actions"] == 0))
    return {"passed": bool(fitted and single_ok), "cache_fitted": bool(fitted),
            "single_fitted": bool(single_ok), "coverage_sufficient": bool(enough),
            "scope": "fixed-label learnability within registered optimizer budget, not closed-loop safety"}


def closed_loop_gate(fit, probe, source_costs, teacher_costs):
    rows = fit["cases"]
    if not fit["summary"].get("metric_valid", True) or fit["summary"]["completion_rate"] != 1:
        return {"passed": False, "checks": {"fit_complete": False}, "recovery": None,
                "leave_largest_winner_out_gain_seconds_sum": None, "teacher_potential_seconds_sum": None,
                "reason": "incomplete candidate rollout; partial elapsed time is not terminal cost"}
    gains = [source_costs[r["case_id"]]-r["makespan"] for r in rows]
    potential = sum(max(0., source_costs[r["case_id"]]-teacher_costs[r["case_id"]]) for r in rows)
    recovery = sum(gains)/potential if potential > 1e-6 else 0.
    leave_largest_out = (sum(gains)-max(gains)) if len(gains) > 1 else 0.
    checks = {"fit_complete": fit["summary"]["completion_rate"] == 1,
              "recovery_at_least_half": recovery >= .5,
              "leave_largest_winner_out_nonnegative": leave_largest_out >= -1e-6,
              "probe_nonregression": probe["summary"]["gain_fraction"] >= 0,
              "probe_risk": risk_pass(probe["summary"])}
    return {"passed": all(checks.values()), "checks": checks, "recovery": recovery,
            "leave_largest_winner_out_gain_seconds_sum": leave_largest_out,
            "teacher_potential_seconds_sum": potential}


def admission_route(technical, learnability, closed_loop):
    if not technical:
        return "implementation_blocked"
    if not learnability:
        return "learnability_budget_exhausted"
    return "four_arm" if closed_loop else "c0_rl_only_480"


def select_stratified_states(records, count, seed):
    """Outcome-independent strata: role, phase, future/ready, decision margin.

    All labels are collected afterwards; there is no future-cost based PPO
    selection. A single formal state alternates roles by the schedule seed.
    """
    if not records:
        raise ValueError("No nontrivial resource decision")
    if count == 1:
        role = 1 + seed % 2
        pool = [r for r in records if r["role"] == role] or records
        ordered = sorted(pool, key=lambda r: digest_json([seed, r["key"]]))
        return [ordered[0]]
    last_step = max(r["key"][0] for r in records) + 1
    buckets = {}
    for r in records:
        mask = np.asarray(r["mask"], dtype=bool)
        values = np.asarray(r["old_logp"])[mask]
        gap = np.sort(values)[-1]-np.sort(values)[-2]
        physical = np.asarray(r["physical"])
        real = mask.copy()
        real[0] = False
        future = bool(real.any() and np.all(physical[real, 4] > .5))
        key = (int(r["role"]), min(2, int(3*r["key"][0]/last_step)), future, bool(gap < .3))
        buckets.setdefault(key, []).append(r)
    for key, pool in buckets.items():
        pool.sort(key=lambda r: digest_json([seed, r["key"]]))
    chosen, keys = [], sorted(buckets, key=lambda k: digest_json([seed, list(k)]))
    while len(chosen) < min(count, len(records)):
        for role in (1, 2):
            for key in keys:
                if key[0] == role and buckets[key] and len(chosen) < count:
                    chosen.append(buckets[key].pop(0))
                    break
    return chosen


def aggregate_evaluations(parts, cases, source, legacy):
    rows = [r for part in parts for r in part["cases"]]
    expected = {c["path"] for c in cases}
    if len(rows) != len(expected) or {r["case_id"] for r in rows} != expected:
        raise ValueError("Incomplete or duplicated validator shards")
    shas = {part["checkpoint_sha256"] for part in parts}
    if len(shas) != 1:
        raise ValueError("Validator shards belong to different checkpoints")
    rows.sort(key=lambda r: r["case_id"])
    result = {"cases": rows, "summary": evaluation_summary(rows, source),
              "checkpoint_sha256": shas.pop(), "evaluation_queries": len(rows),
              "history": HISTORY, "environment": "H2/F4/soft", "decoder": "single_greedy"}
    if legacy:
        result["legacy_summary"] = evaluation_summary(rows, legacy)
    return result


def evaluation_summary(rows, source):
    from onpolicy.utils.stage3_research import paired_summary
    if all(r["completed"] and not r.get("cycle_terminated", False) for r in rows):
        return {**paired_summary(rows, source), "metric_valid": True}
    # Negative gain is a rejection sentinel, NOT an estimated effect. Keep
    # unavailable cost means/CIs null instead of rewarding unfinished jobs.
    return {"metric_valid": False, "reason": "incomplete policy rollout; terminal-cost metrics unavailable",
        "case_count": len(rows), "completion_rate": sum(bool(r["completed"] and not r.get("cycle_terminated", False)) for r in rows)/len(rows),
        "gain_fraction": -1., "gain_is_rejection_sentinel": True, "makespan": None,
        "source_makespan": float(np.mean([source[r["case_id"]] for r in rows])),
        "delta_seconds": None, "distributions": {}, "profiles": {}, "tail_makespan": None,
        "source_tail_makespan": None, "regression_over_5pct_fraction": 1.,
        "paired_case_bootstrap_gain_ci95": None}
