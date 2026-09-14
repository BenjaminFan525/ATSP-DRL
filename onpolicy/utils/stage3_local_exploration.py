"""Opt-in, trainable role exploration and gates for Stage3 local learning.

No global selector patch, candidate truncation, or replacement of actor modules.
The original differentiable heads compute BOTH sampled and forced likelihoods.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from onpolicy.utils.stage3_research import digest_json, EvaluationQueue, safety_gate

MODES = {
    "R": {"roles": [1, 2], "taus": [.3, .3, .3]},
    "J": {"roles": [0, 1, 2], "taus": [.03, .03, .03]},
}
ARMS = {
    "R_PPO": {"mode": "R", "elite": False},
    "J_PPO": {"mode": "J", "elite": False},
    "R_PPO_E": {"mode": "R", "elite": True},
    "J_PPO_E": {"mode": "J", "elite": True},
}
EVAL_PROTOCOL = {"decoder": "greedy", "tau": .3, "seed": 42,
                 "history": "environment_authoritative", "incumbent": False}


def exploration_config(config):
    if config is None:
        return None
    config = copy.deepcopy(MODES[config] if isinstance(config, str) else config)
    if set(config) != {"roles", "taus"}:
        raise ValueError("Exploration requires exactly roles and taus")
    if config["roles"] not in ([1, 2], [0, 1, 2]):
        raise ValueError("Only resource-only or joint training is admitted")
    if len(config["taus"]) != 3 or any(not math.isfinite(t) or t <= 0 for t in config["taus"]):
        raise ValueError("Three finite positive role temperatures required")
    return config


def training_mask(mask, roles, config):
    if config is None:
        return np.asarray(mask).copy()
    return np.asarray(mask) * np.isin(roles, config["roles"])


class RoleExploration:
    """Instance-local PyTorch hooks; support autograd and legal forced replay.

    No module/state_dict names change. Explicit global greedy still wins over
    sampling. Deterministic role likelihoods are excluded by training_mask.
    """
    def __init__(self, ac, config):
        self.config = exploration_config(config)
        self.handles = []
        self.stats = None
        if (ac.actor.__class__.__name__ != "JointPairPtrActor"
                or ac.device_policy_head_mode != "shared" or ac.device_global_matching
                or ac.device_actor.timing_head is not None
                or ac.transporter_actor.timing_head is not None):
            raise ValueError("Local exploration requires the frozen C0 head contract")
        for role, actor in enumerate((ac.actor, ac.device_actor, ac.transporter_actor)):
            self.handles.append(actor.register_forward_pre_hook(self._before(role), with_kwargs=True))
            self.handles.append(actor.register_forward_hook(self._after(role), with_kwargs=True))

    def _before(self, role):
        def hook(module, args, kwargs):
            if args:
                raise ValueError("Role heads must use explicit keyword inputs")
            kwargs = dict(kwargs)
            kwargs["tau"] = self.config["taus"][role]
            kwargs["deterministic"] = bool(kwargs.get("deterministic", False)
                                           or role not in self.config["roles"])
            return args, kwargs
        return hook

    def _after(self, role):
        def hook(module, args, kwargs, output):
            if self.stats is not None:
                selected = (output[0] * kwargs["site_nodes"].shape[1] + output[1]
                            if role == 0 else output[0])
                self.stats.add(role, output[-1], output[-1], selected)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def case_schedule(cases, seed, passes=2):
    return [(visit, case) for visit in range(passes) for case in sorted(cases,
        key=lambda c: digest_json([seed, "case-order", visit, c["content_sha256"]]))]


def choose_elite(entries, usage, seed, step):
    """Least-used case, deterministic tie break; not every fourth current case."""
    if not entries:
        return None
    return min(entries, key=lambda key: (usage.get(key, 0), digest_json([seed, step, key])))


def exploration_gate(diag, probe, source_costs):
    if len(diag) != 32 or len(probe) != 64:
        raise ValueError("Exploration admission requires complete Diag32 and Probe64")
    def gain(rows, n):
        return 1 - np.mean([r["prefix_best"][str(n)] for r in rows]) / np.mean([
            source_costs[r["case_id"]] for r in rows])
    wins = sum(r["prefix_best"]["32"] < source_costs[r["case_id"]] * .99 for r in diag)
    dg, pg = float(gain(diag, 8)), float(gain(probe, 8))
    return {"passed": wins >= 8 and dg >= .01 and pg > 0,
            "diag_cases_gt1pct": wins, "diag_best8_gain": dg, "probe_best8_gain": pg,
            "scope": "exploration signal only; not trained-checkpoint improvement"}


def imitation_gate(teacher_potential, fit_summary, probe_summary):
    capture = fit_summary["gain_fraction"] / teacher_potential if teacher_potential > 0 else 0.
    passed = (teacher_potential >= .005 and capture >= .5
              and probe_summary["gain_fraction"] >= 0 and safety_gate(probe_summary)
              and probe_summary["regression_over_5pct_fraction"] <= .05)
    return {"passed": passed, "teacher_potential": teacher_potential,
            "recovered_fraction": capture,
            "scope": "admits auxiliary imitation only; never blocks pure PPO"}


class LocalEvaluationQueue(EvaluationQueue):
    @staticmethod
    def identity(request):
        return digest_json({key: request[key] for key in (
            "checkpoint_sha256", "cases_sha256", "contract_sha256", "code_sha256",
            "tau", "seed", "evaluation_protocol", "training_exploration")})
