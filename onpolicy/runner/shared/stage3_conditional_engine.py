"""Instance-local conditional actors and bounded, cached-feature diagnostics.

The inherited simulator, recurrent policy and physical contract are unchanged.
An audited AST hook only forwards the existing decoder's actual agent index;
it does not infer identity from call order or modify the global actor class.
"""
from __future__ import annotations

import ast
import copy
import inspect
import multiprocessing as mp
import textwrap
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from onpolicy.runner.shared.stage3_local_improvement_engine import (
    LocalEngine, LocalScore, LocalEnvironmentPool, distribution, masked_kl, _environment_worker)
from onpolicy.utils.stage3_conditional_improvement import (
    ARCHITECTURES, cost_labels, meaningful_epsilon, metric_rank, FIT_ENDPOINTS)


def install_decision_identity(ac):
    original = ac.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed = []
    class IdentityHook(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "actor_head":
                node.keywords.append(ast.keyword(arg="stage3_agent_index", value=ast.Name(id="agent_idx", ctx=ast.Load())))
                changed.append(node.lineno)
            return node
    tree = IdentityHook().visit(tree)
    if len(changed) != 2:
        raise ValueError("Decoder identity hook expected exactly the two resource actor call sites")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__func__.__globals__)
    exec(compile(tree, "<stage3_explicit_device_identity>", "exec"), namespace)
    ac.forward = types.MethodType(namespace[original.__name__], ac)
    return {"resource_call_sites": len(changed), "identity_source": "decoder agent_idx, not actor call ordinal"}


def conditional_distribution(base, gate, rank, mask):
    """Exact WAIT marginal + conditional real ranking; global argmax at decode."""
    if not mask.any(-1).all():
        raise ValueError("Empty legal support")
    real = mask.clone()
    real[:, 0] = False
    has_real = real.any(-1)
    mixed = mask[:, 0] & has_real
    # Safe nonempty support even for forced WAIT rows, so backward never sees
    # softmax(-inf,...,-inf). Such rows are replaced by the forced base below.
    safe_real = real.clone()
    safe_real[~has_real, 0] = True
    real_base = base.masked_fill(~safe_real, -torch.inf)
    conditional = F.log_softmax(real_base + rank.masked_fill(~safe_real, 0), -1)
    wait_base = base[:, 0].masked_fill(~mixed, 0)
    real_mass = torch.logsumexp(real_base, -1).masked_fill(~mixed, 0)
    odds = wait_base - real_mass + gate
    result = conditional + F.logsigmoid(-odds)[:, None]
    result = torch.where(mixed[:, None], result, conditional)
    result = torch.cat((torch.where(mixed, F.logsigmoid(odds), result[:, 0])[:, None], result[:, 1:]), -1)
    result = torch.where(has_real[:, None], result, base)
    return result.masked_fill(~mask, -torch.inf)


def residual_ablation(delta, mask, mode):
    if mode == "full":
        return delta
    real = mask.clone()
    real[:, 0] = False
    mean = (delta*real).sum(-1)/real.sum(-1).clamp_min(1)
    centered = delta-mean[:, None]
    result = torch.zeros_like(delta)
    if mode == "wait_only":
        result[:, 0] = centered[:, 0]
    elif mode == "ranking_only":
        result = centered*real
    elif mode != "zero":
        raise ValueError(mode)
    return result


class ConditionalScore(nn.Module):
    def __init__(self, query_dim, node_dim=64, width=64, pair=False):
        super().__init__()
        self.pair = pair
        physical_dim = 12 if pair else 8
        self.rank_input = nn.Linear(query_dim+node_dim+physical_dim, width)
        self.rank_output = nn.Linear(width, 1)
        self.gate_input = nn.Linear(query_dim+node_dim+physical_dim+3, width)
        self.gate_output = nn.Linear(width, 1)
        for layer in (self.rank_output, self.gate_output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def log_probs(self, q, nodes, physical, base, mask):
        p = physical[..., :12 if self.pair else 8]
        real = mask.clone()
        real[:, 0] = False
        features = torch.cat((nodes, p), -1)
        pooled = (features*real[..., None]).sum(1)/real.sum(-1).clamp_min(1)[:, None]
        extra = torch.stack((base[:, 0].exp(), base.masked_fill(~real, -torch.inf).exp().max(-1).values,
                             torch.log1p(real.sum(-1).float())/np.log(121)), -1)
        gate = self.gate_output(F.silu(self.gate_input(torch.cat((q, pooled, extra), -1)))).squeeze(-1)
        rank = self.rank_output(F.silu(self.rank_input(torch.cat((q[:, None].expand(-1, nodes.shape[1], -1), features), -1)))).squeeze(-1)
        return conditional_distribution(base, gate, rank, mask)


def _conditional_environment_worker(connection):
    from onpolicy.envs.HKBZ import environment as module
    original = module.AircraftScheduleEnv
    class AnnotatedEnvironment(original):
        def annotate(self, result):
            result = list(result)
            graph = result[0]
            count = len(graph["request"].x)
            if [r["id"] for r in self.request_list] != list(range(len(self.request_list))):
                raise ValueError("Request identity/index disagreement")
            keys = [None if r.get("is_noop", False) else tuple(str(v) for v in self._request_identity(r))
                    for r in self.request_list]
            pair = np.zeros((104, count, 4), dtype=np.float32)
            # All values are current estimates used by the existing dispatcher.
            # No request_ready_targets / future realized timestamps are read.
            for local, dev in enumerate(self.device_list[:80]):
                for req in self.request_list[1:]:
                    if not self._device_can_dispatch(dev, req):
                        continue
                    release = self._device_release_seconds(dev, request=req)
                    travel = self._device_travel_seconds(dev, self.sites[req["site_code"]])
                    pair[24+local, req["id"], :3] = np.clip(
                        [release/3600, travel/3600, (release+travel-req.get("lead_time", 0.))/3600], -10, 10)
            result[-1] = dict(result[-1], stage3_request_keys=keys+[None]*(count-len(keys)),
                              stage3_pair_features=pair)
            return tuple(result)
        def reset(self, *args, **kwargs):
            return self.annotate(super().reset(*args, **kwargs))
        def step(self, *args, **kwargs):
            return self.annotate(super().step(*args, **kwargs))
    module.AircraftScheduleEnv = AnnotatedEnvironment
    _environment_worker(connection)


class ConditionalPool(LocalEnvironmentPool):
    def __init__(self, width, timeout=300):
        self.timeout, self.connections, self.processes = timeout, [], []
        context = mp.get_context("spawn")
        for _ in range(width):
            parent, child = context.Pipe()
            process = context.Process(target=_conditional_environment_worker, args=(child,), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)


class ConditionalEngine(LocalEngine):
    def __init__(self, source, *, architecture="score", **kwargs):
        checkpoint = kwargs.pop("checkpoint", None)
        width = kwargs.pop("width", 7)
        self.architecture, self.ablation = architecture, "full"
        super().__init__(source, width=0, **kwargs)
        self.pool = ConditionalPool(width)
        self.width = width
        self.identity_hook = install_decision_identity(self.policy.ac)
        self.set_architecture(architecture)
        if checkpoint:
            self.restore(checkpoint)

    def set_architecture(self, architecture):
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        self.architecture = architecture
        self.scores = nn.ModuleDict()
        for role, actor in ((1, self.policy.ac.device_actor), (2, self.policy.ac.transporter_actor)):
            qdim = next(m.in_features for m in actor.req_query_ff.modules() if isinstance(m, nn.Linear))
            cls = LocalScore if architecture == "score" else ConditionalScore
            args = {} if architecture == "score" else {"pair": architecture == "conditional_pair"}
            self.scores[str(role)] = cls(qdim, actor.embed_dim, **args).to(self.device)
        self.optimizer = torch.optim.Adam(self.scores.parameters(), lr=1e-4, eps=1e-5)
        self.policy_updates = 0

    def _inputs(self, observations, hidden, infos, previous):
        self.current_info = infos[0]
        return super()._inputs(observations, hidden, infos, previous)

    def rollout(self, cases, seeds, **kwargs):
        observer = getattr(self, "query_observer", None)
        if observer:
            observer("started", len(cases))
        rows = super().rollout(cases, seeds, **kwargs)
        if observer:
            observer("completed", sum(bool(r["completed"] and not r.get("cycle_terminated", False)) for r in rows))
        return rows

    def probabilities(self, role, q, nodes, physical, base, mask):
        model = self.scores[str(role)]
        if isinstance(model, ConditionalScore):
            return model.log_probs(q, nodes, physical, base, mask)
        delta = model(q, nodes, physical[..., :8])
        return distribution(base, residual_ablation(delta, mask, self.ablation), mask)

    def _head(self, original, role, kwargs):
        kwargs = dict(kwargs)
        agent = int(kwargs.pop("stage3_agent_index"))
        if not 24 <= agent < 104:
            raise ValueError("Resource decision has invalid actual agent index")
        if not self.residual_enabled:
            return original(**kwargs)
        expected_role = int(np.asarray(self.current_info["agent_types"]).reshape(-1)[agent])
        if expected_role != role:
            raise ValueError("Decoder device identity/role mismatch")
        with torch.no_grad():
            baseline = original(**dict(kwargs, deterministic=True, chosen_request=None))[-1]
        ctx, mask = self._context, kwargs["request_valid_mask"]
        if ctx is None or kwargs["query"].shape[0] != 1:
            raise RuntimeError("Explicit per-replica context required")
        ordinal = ctx["calls"][role]
        ctx["calls"][role] += 1
        q, nodes = kwargs["query"].squeeze(1).detach(), kwargs["request_nodes"].detach()
        pair = torch.as_tensor(self.current_info["stage3_pair_features"][agent], device=self.device).clone()
        pair[:, 3] = np.log1p(max(0, int(mask.sum())-int(mask[0, 0])))/np.log(121)
        physical = torch.cat((ctx["physical"], pair), -1).unsqueeze(0)
        lp = self.probabilities(role, q, nodes, physical, baseline, mask)
        key, chosen = [ctx["step"], role, ordinal], kwargs.get("chosen_request")
        intervention = ctx.get("intervention")
        if chosen is None and intervention and key == intervention["key"]:
            if ctx["hits"]:
                raise ValueError("Repeated intervention")
            chosen = torch.tensor([intervention["action"]], device=self.device)
            ctx["hits"] += 1
            expected = intervention.get("record")
            if expected is not None:
                legal = mask[0].cpu()
                if not torch.equal(legal, expected["mask"]) or ("agent_index" in expected and expected["agent_index"] != agent):
                    raise ValueError("Prefix legal support/device identity changed")
                if (lp[0].detach().cpu()[legal]-expected["old_logp"][legal]).abs().max() > .002:
                    raise ValueError("Prefix conditional probability changed")
        if chosen is not None:
            action = chosen.long().reshape(-1)
            if ((action < 0) | (action >= mask.shape[-1])).any() or not mask.gather(1, action[:, None]).all():
                raise ValueError("Illegal forced action")
        else:
            action = lp.argmax(-1) if kwargs.get("deterministic", False) else torch.multinomial(lp.exp(), 1).squeeze(-1)
        if ctx["collect"] and int(mask.sum()) >= 2:
            ctx["records"].append({"key": key, "role": role, "agent_index": agent,
                "query": q[0].cpu().clone(), "nodes": nodes[0].cpu().clone(), "physical": physical[0].cpu().clone(),
                "base": baseline[0].cpu().clone(), "mask": mask[0].cpu().clone(),
                "old_logp": lp[0].detach().cpu().clone(), "action": int(action.item())})
        return action, lp.gather(1, action[:, None]).squeeze(-1), lp

    def record_distribution(self, records):
        role = records[0]["role"]
        if any(r["role"] != role for r in records):
            raise ValueError("Batch role mismatch")
        if self.architecture == "conditional_pair" and any(r["physical"].shape[-1] != 12 for r in records):
            raise ValueError("Pair model requires fresh explicit-device features; old cache cannot supply them")
        size = max(len(r["mask"]) for r in records)
        tensors = {}
        for key in ("query", "nodes", "physical", "base", "mask", "old_logp"):
            values = []
            for r in records:
                v = r[key]
                if key == "physical" and self.architecture != "conditional_pair":
                    v = v[..., :8]
                if key != "query" and len(v) != size:
                    shape = (size-len(v), *v.shape[1:])
                    fill = -torch.inf if key in ("base", "old_logp") else 0
                    v = torch.cat((v, v.new_full(shape, fill)))
                values.append(v)
            tensors[key] = torch.stack(values).to(self.device)
        lp = self.probabilities(role, tensors["query"], tensors["nodes"], tensors["physical"], tensors["base"], tensors["mask"])
        return lp, tensors

    def save(self, path, **metadata):
        return super().save(path, **dict(metadata, architecture=self.architecture))

    def restore(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        architecture = payload.get("metadata", {}).get("architecture", "score")
        if architecture != self.architecture:
            self.set_architecture(architecture)
        return super().restore(path)


class CachedRanking:
    """One detached feature batch per role; no environment calls in the loop."""
    def __init__(self, engine, groups, *, weighted=False, lookup=False):
        self.engine, self.groups, self.weighted, self.lookup = engine, groups, weighted, lookup
        self.rows = [(g, s) for g in groups for s in g["states"]]
        self.role_indices = {role: [i for i, (_, s) in enumerate(self.rows) if s["record"]["role"] == role] for role in (1, 2)}
        self.cached_tensors = {}
        if not lookup:
            with torch.no_grad():
                for role, indices in self.role_indices.items():
                    if indices:
                        _, self.cached_tensors[role] = engine.record_distribution([self.rows[i][1]["record"] for i in indices])
        self.lookup_logits = nn.ParameterList()
        if lookup:
            for _, s in self.rows:
                self.lookup_logits.append(nn.Parameter(s["record"]["base"].masked_fill(~s["record"]["mask"], 0).to(engine.device)))
        self.pairs, self.neutral = [], []
        for index, (g, s) in enumerate(self.rows):
            costs, eps = cost_labels(s, meaningful=weighted)
            weight = 1/(len(groups)*len(g["states"]))
            pairs = [(a, b, min(50., (cb-ca)/eps) if weighted else 1.)
                     for a, ca in costs.items() for b, cb in costs.items() if ca < cb-eps]
            total = sum(v for _, _, v in pairs)
            self.pairs.extend((index, a, b, weight*v/total) for a, b, v in pairs)
            if weighted and min(costs.values()) >= s["reference_cost"]-eps:
                self.neutral.append((index, weight*.1))

    def predictions(self):
        if self.lookup:
            return [F.log_softmax(v.masked_fill(~s["record"]["mask"].to(v.device), -torch.inf), -1)
                    for v, (_, s) in zip(self.lookup_logits, self.rows)]
        values = [None]*len(self.rows)
        for role, indices in self.role_indices.items():
            if not indices:
                continue
            t = self.cached_tensors[role]
            lp = self.engine.probabilities(role, t["query"], t["nodes"], t["physical"], t["base"], t["mask"])
            for j, i in enumerate(indices):
                values[i] = lp[j, :len(self.rows[i][1]["record"]["mask"])]
        return values

    def loss(self, predictions):
        legal = self.rows[0][1]["record"]["mask"].to(self.engine.device)
        zero = predictions[0][legal].sum()*0
        loss = sum((F.softplus(predictions[i][b]-predictions[i][a])*weight for i, a, b, weight in self.pairs), zero)
        for i, weight in self.neutral:
            r = self.rows[i][1]["record"]
            mask = r["mask"].to(self.engine.device)
            old = r["old_logp"].to(self.engine.device)
            loss = loss + weight*masked_kl(old[None], predictions[i][None], mask[None]).sum()
        return loss

    @torch.no_grad()
    def metrics(self):
        lp = self.predictions()
        result = {"loss": float(self.loss(lp)), "states": len(lp), "roles": {}}
        for name, meaningful in (("legacy", False), ("meaningful", True)):
            eligible = hits = unknown = pairs = correct = 0
            regret = initial_regret = 0.
            by_role, types = {}, {}
            for pred, (g, s) in zip(lp, self.rows):
                costs, eps = cost_labels(s, meaningful=meaningful)
                action = int(pred.argmax())
                best, ref = min(costs.values()), float(s["reference_cost"])
                good = {a for a, c in costs.items() if c < ref-eps}
                unknown += int(action not in costs)
                if action in costs:
                    regret += max(0., costs[action]-best)/len(g["states"])/len(self.groups)
                else:
                    # A gate must not interpret unknown cost as zero regret.
                    regret += ref/len(g["states"])/len(self.groups)
                initial_regret += max(0., ref-best)/len(g["states"])/len(self.groups)
                if good:
                    eligible += 1
                    hits += int(action in good)
                    role = str(s["record"]["role"])
                    r = by_role.setdefault(role, {"eligible": 0, "hits": 0})
                    r["eligible"] += 1
                    r["hits"] += int(action in good)
                    kind = "wait" if 0 in good else "dispatch"
                    t = types.setdefault(kind, {"eligible": 0, "hits": 0})
                    t["eligible"] += 1
                    t["hits"] += int(action in good)
                for a, ca in costs.items():
                    for b, cb in costs.items():
                        if ca < cb-eps:
                            pairs += 1
                            correct += int(pred[a] > pred[b])
            result[name] = {"beneficial_states": eligible, "beneficial_hits": hits,
                "beneficial_hit_rate": hits/eligible if eligible else 0., "unknown_actions": unknown,
                "pair_accuracy": correct/pairs if pairs else 0., "pairs": pairs,
                "regret_seconds": regret, "initial_regret_seconds": initial_regret,
                "regret_ratio": regret/initial_regret if initial_regret > 1e-8 else 0.,
                "by_role": by_role, "by_action_type": types}
        return result

    def fit(self, *, lr, steps=2000, heartbeat=None, patience=200):
        if not 1 <= steps <= 2000:
            raise ValueError("Unregistered fit budget")
        params = list(self.lookup_logits.parameters()) if self.lookup else list(self.engine.scores.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5) if self.lookup else self.engine.optimizer
        for group in optimizer.param_groups:
            group["lr"] = lr
        initial = self.metrics()
        initial_updates = self.engine.policy_updates
        if not self.pairs:
            return {"initial": initial, "metrics": initial, "trace": [{"step": 0, "metrics": initial}],
                    "executed_updates": 0, "selected_update": 0, "lr": lr, "lookup_only": self.lookup,
                    "weighted": self.weighted, "stop_reason": "no_informative_preferences"}
        best, best_key, trace = initial, metric_rank(initial), [{"step": 0, "metrics": initial}]
        best_state = copy.deepcopy(self.lookup_logits.state_dict() if self.lookup else self.engine.scores.state_dict())
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        best_step, plateau_start, last_loss = 0, 0, initial["loss"]
        max_norm, max_kl, clipped = 0., 0., 0
        for step in range(1, steps+1):
            optimizer.zero_grad(set_to_none=True)
            before = self.predictions()
            loss = self.loss(before)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite cached fit loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            max_norm, clipped = max(max_norm, norm), clipped+int(norm > 1.)
            optimizer.step()
            if not self.lookup:
                self.engine.policy_updates += 1
            if step % 50 == 0 or step == steps:
                metrics = self.metrics()
                key = metric_rank(metrics)
                with torch.no_grad():
                    after = self.predictions()
                    kls = [float(masked_kl(s["record"]["old_logp"].to(p.device)[None], p[None],
                                s["record"]["mask"].to(p.device)[None])) for p, (_, s) in zip(after, self.rows)]
                    max_kl = max(max_kl, max(kls))
                trace.append({"step": step, "metrics": metrics, "max_cumulative_kl": max(kls)})
                if key > best_key:
                    best, best_key, best_step = metrics, key, step
                    best_state = copy.deepcopy(self.lookup_logits.state_dict() if self.lookup else self.engine.scores.state_dict())
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                if last_loss-metrics["loss"] > max(1e-6, abs(last_loss)*1e-4):
                    plateau_start = step
                last_loss = metrics["loss"]
                if heartbeat:
                    heartbeat.update(event="cached_fit", optimizer_step=step, loss=metrics["loss"],
                                     beneficial_hit_rate=metrics["meaningful"]["beneficial_hit_rate"])
                if step >= 200 and step-plateau_start >= patience:
                    break
        if self.lookup:
            self.lookup_logits.load_state_dict(best_state)
        else:
            self.engine.scores.load_state_dict(best_state)
            self.engine.policy_updates = initial_updates+best_step
        optimizer.load_state_dict(best_optimizer)
        return {"initial": initial, "metrics": best, "trace": trace, "executed_updates": step,
                "selected_update": best_step, "max_gradient_norm": max_norm, "clipped_updates": clipped,
                "max_cumulative_kl": max_kl, "lr": lr, "lookup_only": self.lookup,
                "weighted": self.weighted, "stop_reason": "plateau" if step < steps else "budget_exhausted",
                "selection_scope": "cached training labels only; diagnostic weights forbidden as formal initialization"}
