"""Inference-only interventions for the isolated C0 sampling audit.

Nothing imports this module in production training. Hooks are local to one audit
process and restored on exit. Native (top_k=0) sampling keeps the original RNG
calls and logits byte-for-byte; top-k returns its *renormalized* likelihoods.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib

import numpy as np
import torch


ARMS = {
    "full_t030": {"roles": [0, 1, 2], "tau": .3, "top_k": 0},
    "plane_t030": {"roles": [0], "tau": .3, "top_k": 0},
    "resource_t030": {"roles": [1, 2], "tau": .3, "top_k": 0},
    "full_t010": {"roles": [0, 1, 2], "tau": .1, "top_k": 0},
    "full_t003": {"roles": [0, 1, 2], "tau": .03, "top_k": 0},
    "full_top4_t030": {"roles": [0, 1, 2], "tau": .3, "top_k": 4},
    "full_previous_t030": {
        "roles": [0, 1, 2], "tau": .3, "top_k": 0, "history": "previous_issued"},
}


def restricted_logits(logits, top_k):
    """Keep at most k finite legal logits; resolve ties by action index."""
    if logits.ndim != 2 or top_k < 0:
        raise ValueError("Expected 2D logits and nonnegative top_k")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("Non-finite legal score")
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("No legal action")
    if not top_k:
        return logits
    indices = torch.argsort(logits, dim=-1, descending=True, stable=True)
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(1, indices[:, :min(top_k, logits.shape[-1])], True)
    return torch.log_softmax(logits.masked_fill(~keep, float("-inf")), dim=-1)


def action_digest(actions):
    values = np.asarray(actions, dtype="<i4")
    if values.ndim != 3 or values.shape[-1] not in (2, 3):
        raise ValueError("Expected time x agent x action")
    if values.shape[-1] == 2:
        values = np.concatenate([values, np.full((*values.shape[:-1], 1), -1, dtype="<i4")], -1)
    digest = hashlib.sha256(str(values.shape).encode())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


class DecisionStats:
    """Decision-weighted stats, with single-candidate decisions kept separate."""
    def __init__(self):
        self.rows = {}

    def add(self, role, base, effective, selected):
        legal = torch.isfinite(base).sum(-1)
        meaningful = legal > 1
        base_p, effective_p = base.softmax(-1), effective.softmax(-1)
        entropy = -(effective_p * effective.nan_to_num(neginf=0)).sum(-1)
        values = torch.stack([
            torch.ones_like(legal), meaningful, legal * meaningful,
            base_p.max(-1).values * meaningful,
            effective_p.max(-1).values * meaningful,
            entropy * meaningful,
            (selected != base.argmax(-1)) * meaningful,
            ((selected == 0) & meaningful) if role else torch.zeros_like(meaningful),
        ], -1).sum(0).detach().cpu().double().numpy()
        current = self.rows.setdefault(str(role), np.zeros(8, dtype=np.float64))
        current += values

    def result(self):
        result = {}
        for role, values in self.rows.items():
            n = max(1., values[1])
            result[role] = dict(head_calls=int(values[0]), multi_candidate_decisions=int(values[1]),
                legal_candidates_mean=values[2] / n, base_pmax_mean=values[3] / n,
                effective_pmax_mean=values[4] / n, effective_entropy_mean=values[5] / n,
                non_greedy_fraction=values[6] / n,
                resource_noop_fraction=values[7] / n if role != "0" else None)
        return result


@contextmanager
def sampling_heads(ac, config, stats=None):
    """Override sampling roles/temperature while retaining sequential legal masks."""
    from onpolicy.algorithms.utils import ptr_actor
    if (ac.actor.__class__.__name__ != "JointPairPtrActor"
            or ac.device_policy_head_mode != "shared" or ac.device_global_matching
            or ac.device_actor.timing_head is not None
            or ac.transporter_actor.timing_head is not None):
        raise ValueError("Audit hooks only support the frozen C0 architecture")
    if config["tau"] <= 0 or not set(config["roles"]).issubset({0, 1, 2}):
        raise ValueError("Invalid sampling arm")
    original_selector = ptr_actor._select_masked_index
    top_k = int(config["top_k"])
    saved = []

    def selector(logits, valid, deterministic):
        if not top_k:
            return original_selector(logits, valid, deterministic)
        scores = restricted_logits(logits.masked_fill(~valid, float("-inf")), top_k)
        return original_selector(scores, valid & torch.isfinite(scores), deterministic)

    def wrap(original, role):
        def forward(*args, **kwargs):
            if args:
                raise ValueError("Frozen actor caller must use keyword arguments")
            forced = kwargs.get("chosen_op" if role == 0 else "chosen_request") is not None
            if forced and top_k:
                raise ValueError("Top-k audit is not an off-policy replay/training implementation")
            kwargs["deterministic"] = bool(kwargs.get("deterministic", False) or role not in config["roles"])
            kwargs["tau"] = config["tau"]
            output = list(original(**kwargs))
            base = output[-1]
            effective = restricted_logits(base, top_k) if top_k else base
            selected = (output[0] * kwargs["site_nodes"].shape[1] + output[1]
                        if role == 0 else output[0])
            if top_k:
                output[-1] = effective
                output[-2] = effective.gather(1, selected[:, None]).squeeze(-1)
            if stats is not None:
                stats.add(role, base, effective, selected)
            return tuple(output)
        return forward

    try:
        ptr_actor._select_masked_index = selector
        for role, actor in enumerate((ac.actor, ac.device_actor, ac.transporter_actor)):
            saved.append((actor, actor.__dict__.get("forward"), "forward" in actor.__dict__))
            actor.forward = wrap(actor.forward, role)
        yield
    finally:
        ptr_actor._select_masked_index = original_selector
        for actor, original, existed in saved:
            if existed:
                actor.forward = original
            else:
                del actor.forward


@contextmanager
def compare_actor_api(policy, device, metrics, *, hidden_abs_tolerance=1e-5):
    """On every visited state compare act/get_actions with identical inputs/RNG.

    Leaves the RNG stream exactly where the original get_actions call left it.
    This checks retained entry-point equivalence, NOT an unavailable historical
    checkpoint/environment, and NOT equivalence between batch sizes.
    """
    original = policy.get_actions
    existed = "get_actions" in policy.__dict__
    get_rng = lambda: torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    set_rng = lambda value: (torch.cuda.set_rng_state(value, device)
                            if device.type == "cuda" else torch.set_rng_state(value))

    def checked(*args, **kwargs):
        before = get_rng()
        output = original(*args, **kwargs)
        after = get_rng()
        try:
            set_rng(before)
            act_kwargs = {k: v for k, v in kwargs.items() if k != "return_decision_mask"}
            actions, hidden = policy.act(*args, **act_kwargs)
            rng_equal = torch.equal(get_rng(), after)
        finally:
            set_rng(after)
        delta = float((hidden - output[3]).abs().max())
        metrics["calls"] = metrics.get("calls", 0) + 1
        metrics["hidden_max_abs"] = max(metrics.get("hidden_max_abs", 0.), delta)
        actions_equal = torch.equal(actions, output[1])
        metrics["actions_equal"] = metrics.get("actions_equal", True) and actions_equal
        metrics["rng_equal"] = metrics.get("rng_equal", True) and rng_equal
        if not actions_equal or delta > hidden_abs_tolerance or not rng_equal:
            raise RuntimeError(f"act/get_actions mismatch at call {metrics['calls']}: "
                f"hidden={delta}, actions_equal={actions_equal}, rng={rng_equal}")
        return output

    try:
        policy.get_actions = checked
        yield
    finally:
        if existed:
            policy.get_actions = original
        else:
            del policy.get_actions


def history_inputs(previous, active, op, site, mode):
    if mode == "authoritative":
        return op, site
    if mode != "previous_issued":
        raise ValueError(mode)
    # Exactly the retained sampling script's issued-action history, all roles.
    return previous[:, :, 0].copy(), previous[:, :, 1].copy()
