"""Frozen C0 + stateless local residuals, exact-prefix interventions and learners.

The residual never changes the recurrent transition. Detached feature replay is
therefore the exact conditional actor computation, not truncated recurrent RL.
All installation is instance-local; legacy trainers/observations are untouched.
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
from pathlib import Path
import random
import types
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from onpolicy.runner.shared.stage3_research_engine import (ResearchEngine, EnvironmentPool,
    _environment_worker, environment_config, seed_all)
from onpolicy.utils.stage3_local_improvement import HISTORY, remap_history, local_advantages
from onpolicy.utils.stage3_research import digest_file, digest_json


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest_file(path)


class LocalScore(nn.Module):
    def __init__(self, query_dim, node_dim=64, width=64):
        super().__init__()
        self.input = nn.Linear(query_dim + node_dim + 8, width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, nodes, physical):
        q = query.unsqueeze(1).expand(-1, nodes.shape[1], -1)
        return self.output(F.silu(self.input(torch.cat((q, nodes, physical), -1)))).squeeze(-1)


def _local_environment_worker(connection):
    # Process-local observation annotation, no edits to the legacy environment.
    # Actual plane IDs are necessary: graph plane slots could be reused.
    from onpolicy.envs.HKBZ import environment as module
    original = module.AircraftScheduleEnv
    class ObservedEnvironment(original):
        def annotate(self, result):
            result = list(result)
            if [r["id"] for r in self.request_list] != list(range(len(self.request_list))):
                raise ValueError("Request pool positions no longer match action indices")
            keys = [None if r.get("is_noop", False) else tuple(str(v) for v in self._request_identity(r))
                    for r in self.request_list]
            count = len(result[0]["request"].x)
            if len(keys) > count:
                raise ValueError("Request identity metadata exceeds graph capacity")
            result[-1] = dict(result[-1], stage3_request_keys=keys+[None]*(count-len(keys)))
            return tuple(result)

        def reset(self, *args, **kwargs):
            return self.annotate(super().reset(*args, **kwargs))

        def step(self, *args, **kwargs):
            return self.annotate(super().step(*args, **kwargs))
    module.AircraftScheduleEnv = ObservedEnvironment
    _environment_worker(connection)


class LocalEnvironmentPool(EnvironmentPool):
    def __init__(self, width, timeout=300):
        self.timeout, self.connections, self.processes = timeout, [], []
        context = mp.get_context("spawn")
        for _ in range(width):
            parent, child = context.Pipe()
            process = context.Process(target=_local_environment_worker, args=(child,), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)


def distribution(base, delta, mask):
    if not mask.any(-1).all():
        raise ValueError("Empty legal support")
    return F.log_softmax((base + delta).masked_fill(~mask, -torch.inf), -1)


def masked_kl(old, new, mask):
    safe_old, safe_new = old.masked_fill(~mask, 0), new.masked_fill(~mask, 0)
    return (safe_old.exp() * mask * (safe_old-safe_new)).sum(-1)


def physical_features(graph, device):
    raw = graph["request"].x.to(device)
    valid = raw[:, 7] < .5
    scale = raw[:, 2:4].abs().max().clamp_min(1.)
    look = graph.request_is_lookahead.to(device).float()
    degree = torch.zeros(len(raw), device=device)
    edge = graph[("device", "can_serve", "request")].edge_index if ("device", "can_serve", "request") in graph.edge_types else None
    if edge is not None and edge.numel():
        degree = torch.bincount(edge[1].to(device), minlength=len(raw)).float()
    return torch.stack((raw[:, 1].clamp(0, 36000)/3600, (-raw[:, 1]).clamp(0, 36000)/3600,
        raw[:, 2]/scale, raw[:, 3]/scale, look, raw[:, 7], valid.float(), torch.log1p(degree)/np.log(81)), -1)


class LocalEngine(ResearchEngine):
    def __init__(self, source, *, checkpoint=None, width=7, device="cuda:0", history=HISTORY,
                 protocol="development", exploration="R", reference_variant=None, reference_checkpoint=None):
        from onpolicy.utils.stage3_numerics import configure_runtime, install_stable_pool
        configure_runtime()
        super().__init__(source, width=0, device=device, freeze_shared=True,
                         exploration=exploration, diagnostics=True, cuda_memory_fraction=.8)
        self.pool = LocalEnvironmentPool(width)
        self.width = width
        if self.device.type == "cuda":
            # Remove the old fractional training cap; the service has a host-RAM budget.
            torch.cuda.set_per_process_memory_fraction(1., self.device)
        if reference_variant:
            from onpolicy.algorithms.utils.stage3_encoder import install_encoder
            install_encoder(self.policy, reference_variant)
            payload = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
            if payload.get("representation") != reference_variant:
                raise ValueError("Historical checkpoint representation mismatch")
            self.policy.ac.load_state_dict(payload["model"], strict=True)
        install_stable_pool(self.policy)
        self.policy.ac.requires_grad_(False)
        self.policy.ac.eval()
        self.policy.ac.shared_encoder_activation_checkpoint = False
        self.source = str(Path(source).resolve())
        self.source_sha = digest_file(source)
        self.protocol, self.history = protocol, history
        if history not in (HISTORY, "legacy"):
            raise ValueError(history)
        self.scores = nn.ModuleDict()
        for role, actor in ((1, self.policy.ac.device_actor), (2, self.policy.ac.transporter_actor)):
            qdim = next(m.in_features for m in actor.req_query_ff.modules() if isinstance(m, nn.Linear))
            self.scores[str(role)] = LocalScore(qdim, actor.embed_dim).to(self.device)
            original = actor.forward
            def forward(module, *, _original=original, _role=role, **kwargs):
                return self._head(_original, _role, kwargs)
            actor.forward = types.MethodType(forward, actor)
        self.optimizer = torch.optim.Adam(self.scores.parameters(), lr=1e-4, eps=1e-5)
        self.policy_updates = 0
        self._context = None
        self.residual_enabled = True
        self.checkpoint_meta = {}
        self.behavior_sha = self.source_sha
        if checkpoint is not None:
            self.restore(checkpoint)

    def _head(self, original, role, kwargs):
        if not self.residual_enabled:
            return original(**kwargs)
        chosen = kwargs.get("chosen_request")
        deterministic = kwargs.get("deterministic", False)
        with torch.no_grad():
            baseline = original(**dict(kwargs, deterministic=True, chosen_request=None))[-1]
        context = self._context
        if context is None or kwargs["query"].shape[0] != 1:
            raise RuntimeError("Local decoder requires an explicit per-replica context")
        ordinal = context["calls"][role]
        context["calls"][role] += 1
        mask = kwargs["request_valid_mask"]
        q = kwargs["query"].squeeze(1).detach()
        nodes = kwargs["request_nodes"].detach()
        physical = context["physical"].unsqueeze(0)
        logp = distribution(baseline, self.scores[str(role)](q, nodes, physical), mask)
        key = [context["step"], role, ordinal]
        intervention = context.get("intervention")
        if chosen is None and intervention and key == intervention["key"]:
            if context["hits"]:
                raise ValueError("Intervention applied more than once")
            chosen = torch.tensor([intervention["action"]], device=self.device)
            context["hits"] += 1
            expected = intervention.get("record")
            if expected is not None:
                if not torch.equal(mask[0].cpu(), expected["mask"]):
                    raise ValueError("Intervention prefix changed the legal mask")
                legal = mask[0].cpu()
                error = (logp[0].cpu()[legal] - expected["old_logp"][legal]).abs().max().item()
                if error > .002:
                    raise ValueError(f"Intervention full-history probability mismatch: {error}")
        if chosen is not None:
            action = chosen.long().reshape(-1)
            if ((action < 0) | (action >= mask.shape[-1])).any() or not mask.gather(1, action[:, None]).all():
                raise ValueError("Illegal forced local action")
        elif deterministic:
            action = logp.argmax(-1)
        else:
            action = torch.multinomial(logp.exp(), 1).squeeze(-1)
        if context["collect"] and int(mask.sum()) >= 2:
            context["records"].append({"key": key, "role": role, "query": q[0].cpu().clone(),
                "nodes": nodes[0].cpu().clone(), "physical": physical[0].cpu().clone(),
                "base": baseline[0].cpu().clone(), "mask": mask[0].cpu().clone(),
                "old_logp": logp[0].cpu().clone(), "action": int(action.item())})
        return action, logp.gather(1, action[:, None]).squeeze(-1), logp

    @torch.no_grad()
    def rollout(self, cases, seeds, *, deterministic=True, collect=False, forced=None,
                interventions=None, heartbeat=None, audit_graph=False, sampling_window=None,
                allow_incomplete=False, **unused):
        if not 0 < len(cases) <= self.width or len(cases) != len(seeds):
            raise ValueError("Invalid rollout width")
        from onpolicy.utils.stage3_research import verify_cases
        verify_cases(list({c["path"]:c for c in cases}.values()))
        count = len(cases)
        resets = []
        for i, (case, seed) in enumerate(zip(cases, seeds)):
            config = environment_config(case["path"], seed)
            config.update(device_future_intent_horizon=2, device_frontier_max_requests=4,
                          device_lookahead_reservation_mode="soft", _stage3_diagnostics=True)
            resets.append((i, "reset", config))
        initial = self.pool.call(resets)
        obs, infos = [initial[i][0] for i in range(count)], [initial[i][2] for i in range(count)]
        hidden = np.zeros((count, 104, 1, 64), np.float32)
        previous = np.full((count, 104, 3), -1, np.int64)
        old_keys = [[] for _ in cases]
        live = set(range(count))
        trajectories = [{"case_id": c["path"], "profile": c["profile"], "distribution": c["distribution"],
            "seed": int(seed), "actions": [], "records": [], "history_audit": {}, "intervention_hits": 0,
            "behavior_sha256": self.behavior_sha, "behavior_deterministic": deterministic,
            "forced_replay": forced is not None, "history": self.history} for c, seed in zip(cases, seeds)]
        rng = []
        for seed in seeds:
            seed_all(seed)
            rng.append(torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else torch.get_rng_state())
        for step in range(4000):
            if not live:
                break
            commands = []
            for i in sorted(live):
                keys = infos[i]["stage3_request_keys"]
                mapped, audit = remap_history(previous[i], old_keys[i], keys,
                                               np.asarray(infos[i]["active_agents"]).reshape(-1))
                for name, value in audit.items():
                    trajectories[i]["history_audit"][name] = trajectories[i]["history_audit"].get(name, 0) + value
                history = mapped if self.history == HISTORY else previous[i]
                graph, h, active, op, site, roles = self._inputs([obs[i]], hidden[i:i+1], [infos[i]], history[None])
                context = {"step": step, "calls": {1: 0, 2: 0}, "physical": physical_features(obs[i], self.device),
                    "records": [], "collect": collect, "hits": 0,
                    "intervention": interventions[i] if interventions else None}
                self._context = context
                if audit_graph and bool((obs[i]["request"].x[:, 7] < .5).any()):
                    probes = trajectories[i].setdefault("graph_audit", [])
                    if len(probes) < 3 and (not probes or step-probes[-1]["step"] >= 80):
                        probes.append({"step": step, **self.graph_audit(obs[i])})
                if self.device.type == "cuda":
                    torch.cuda.set_rng_state(rng[i], self.device)
                else:
                    torch.set_rng_state(rng[i])
                if forced is not None and step < len(forced[i]):
                    actions = np.asarray(forced[i][step], dtype=np.int64)[None]
                    _, _, _, new_h = self.policy.evaluate_actions(graph, h, active, op, site, actions,
                        agent_types=roles, return_decision_mask=True, return_rnn_states=True)
                else:
                    _, actions, _, new_h, _ = self.policy.get_actions(graph, h, active, op, site,
                        deterministic=(deterministic if sampling_window is None else
                                       not sampling_window[0] <= step < sampling_window[1]),
                        agent_types=roles, return_decision_mask=True)
                    actions = actions.cpu().numpy().astype(np.int64)
                    if actions.shape[-1] == 2:
                        actions = np.concatenate((actions, np.full((*actions.shape[:-1], 1), -1, np.int64)), -1)
                rng[i] = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else torch.get_rng_state()
                trajectories[i]["records"].extend(context["records"])
                trajectories[i]["intervention_hits"] += context["hits"]
                trajectories[i]["actions"].append(actions[0].copy())
                previous[i], old_keys[i], hidden[i] = actions[0], keys, new_h[0].cpu().numpy()
                commands.append((i, "step", actions[0]))
            results = self.pool.call(commands)
            for i, (observation, _, done, info) in results.items():
                obs[i], infos[i] = observation, info
                if np.all(done):
                    live.remove(i)
            if heartbeat and step % 40 == 0:
                heartbeat.update(event="local_rollout", rollout_step=step, live_trajectories=len(live))
        summaries = self.pool.call([(i, "summary", None) for i in range(count)])
        self._context = None
        for i, row in enumerate(trajectories):
            row.update(summaries[i])
            row["actions"] = np.asarray(row["actions"], dtype=np.int32)
            if not row["completed"] or row.get("cycle_terminated"):
                if not allow_incomplete:
                    raise RuntimeError(f"Incomplete trajectory: {row['case_id']}")
            if interventions and row["intervention_hits"] != 1:
                raise ValueError("Requested intervention was not reached exactly once")
            if forced is not None and not np.array_equal(row["actions"][:len(forced[i])], forced[i]):
                raise ValueError("Forced prefix changed")
        return trajectories

    @torch.no_grad()
    def graph_audit(self, graph):
        original = graph.clone().to(self.device)
        padded = original.clone()
        padded["request"].x = torch.cat((padded["request"].x,
            padded["request"].x.new_tensor([[0, 0, 0, 0, 0, 0, -1, 1]]).repeat(5, 1)))
        a, b = self.policy.ac.encoder(original), self.policy.ac.encoder(padded)
        raw = original["request"].x
        return {"padding_global_max_abs_change": float((a["global_emb"]-b["global_emb"]).abs().max()),
            "feature_abs_max": raw.abs().max(0).values.cpu().tolist(),
            "feature_q50": raw.quantile(.5, dim=0).cpu().tolist(),
            "feature_q95": raw.quantile(.95, dim=0).cpu().tolist(),
            "scope": "encoder capacity-invariance and scale audit; no performance causality claim"}

    def record_distribution(self, records):
        role = records[0]["role"]
        if any(r["role"] != role for r in records):
            raise ValueError("Batch role mismatch")
        tensors = {k: torch.stack([r[k] for r in records]).to(self.device)
                   for k in ("query", "nodes", "physical", "base", "mask", "old_logp")}
        delta = self.scores[str(role)](tensors["query"], tensors["nodes"], tensors["physical"])
        return distribution(tensors["base"], delta, tensors["mask"]), tensors

    def learn(self, groups, mode, *, lr=1e-4, epochs=2, heartbeat=None, diagnostic=False):
        from onpolicy.utils.stage3_local_improvement import validate_group
        if mode not in ("rank", "local_rl", "ppo"):
            raise ValueError(mode)
        if not diagnostic:
            validate_group(groups, self.behavior_sha, mode)
        items = []
        for group in groups:
            if mode == "ppo":
                for trace in group["trajectories"]:
                    if trace["behavior_deterministic"] or trace["forced_replay"]:
                        raise ValueError("Off-policy full trajectory")
                    advantage = float(local_advantages(group["source_cost"], [trace["makespan"]])[0])
                    for record in trace["records"]:
                        items.append({"record": record, "advantage": advantage, "action": record["action"],
                            "weight": 1 / (len(groups)*len(group["trajectories"]))})
            else:
                for state in group["states"]:
                    record, actions, costs = state["record"], state["actions"], state["costs"]
                    weight = 1 / (len(groups)*len(group["states"]))
                    if mode == "rank":
                        pairs = [(i,j) for i in range(len(actions)) for j in range(len(actions))
                                 if costs[i] < costs[j]-1e-6 and actions[i] != actions[j]]
                        for i,j in pairs:
                            items.append({"record": record, "better": actions[i], "worse": actions[j],
                                          "weight": weight/max(1, len(pairs))})
                    else:
                        if state.get("sampling") != "on_policy_with_replacement":
                            raise ValueError("Local PG requires genuine behavior-policy sampling")
                        for a, advantage in zip(actions, local_advantages(state["reference_cost"], costs)):
                            items.append({"record": record, "advantage": float(advantage), "action": a,
                                          "weight": weight/len(actions)})
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        result = {"mode": mode, "items": len(items), "epochs": [], "actor_updates": 0}
        if not items:
            result["reason"] = "no non-tied local preferences; no update"
            return result
        if mode == "rank":
            # Diagnostic teachers can be old; the step-size guard compares to
            # this update's starting policy, not permanently to their C0 logits.
            cached = {}
            with torch.no_grad():
                for item in items:
                    original = item["record"]
                    if id(original) not in cached:
                        lp, _ = self.record_distribution([original])
                        cached[id(original)] = dict(original, old_logp=lp[0].cpu())
                    item["record"] = cached[id(original)]
        batches = []
        for role in (1, 2):
            role_items = [x for x in items if x["record"]["role"] == role]
            batches.extend(role_items[start:start+64] for start in range(0, len(role_items), 64))
        for epoch in range(epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum, replay_error = 0., 0.
            for batch in batches:
                lp, tensors = self.record_distribution([x["record"] for x in batch])
                weight = lp.new_tensor([x["weight"] for x in batch])
                index = torch.arange(len(batch), device=self.device)
                if mode == "rank":
                    better = torch.tensor([x["better"] for x in batch], device=self.device)
                    worse = torch.tensor([x["worse"] for x in batch], device=self.device)
                    losses = F.softplus(lp[index, worse]-lp[index, better])
                else:
                    action = torch.tensor([x["action"] for x in batch], device=self.device)
                    log_ratio = lp[index, action]-tensors["old_logp"][index, action]
                    if epoch == 0:
                        replay_error = max(replay_error, float(log_ratio.detach().abs().max()))
                    ratio = log_ratio.exp()
                    advantage = lp.new_tensor([x["advantage"] for x in batch])
                    losses = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)
                loss = (losses*weight).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite actor loss")
                loss.backward()
                loss_sum += float(loss.detach())
            if mode != "rank" and epoch == 0 and replay_error > .002:
                raise ValueError(f"Fresh conditional feature replay mismatch: {replay_error}")
            norm = torch.nn.utils.clip_grad_norm_(self.scores.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.policy_updates += 1
            result["actor_updates"] += 1
            role_kl = {1: [], 2: []}
            with torch.no_grad():
                for batch in batches:
                    lp, tensors = self.record_distribution([x["record"] for x in batch])
                    role_kl[batch[0]["record"]["role"]].extend(masked_kl(tensors["old_logp"], lp, tensors["mask"]).cpu().tolist())
            kl = {str(role): float(np.mean(values)) for role, values in role_kl.items() if values}
            max_kl = max(kl.values(), default=0.)
            result["epochs"].append({"loss": loss_sum, "preclip_gradient_norm": float(norm),
                "conditional_replay_error": replay_error, "role_kl": kl, "max_role_kl": max_kl})
            if heartbeat:
                heartbeat.update(event="local_update", actor_updates=self.policy_updates, epoch=epoch+1)
            if max_kl > .04:
                raise RuntimeError(f"Hard KL gate failed; no rollback or silent retry: {max_kl}")
            if max_kl > .02:
                result["soft_kl_stop"] = True
                break
        return result

    def save(self, path, **metadata):
        payload = {"schema": "stage3-local-residual-checkpoint-v1", "protocol": self.protocol,
            "source": self.source, "source_sha256": self.source_sha, "history": self.history,
            "environment": {"h": 2, "f": 4, "reservation": "soft"},
            "scores": self.scores.state_dict(), "optimizer": self.optimizer.state_dict(),
            "policy_updates": self.policy_updates, "metadata": metadata,
            "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "rng_cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}
        sha = save_tensor(path, payload)
        self.behavior_sha = sha
        self.checkpoint_meta = metadata
        return sha

    def restore(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("schema") != "stage3-local-residual-checkpoint-v1" or payload["protocol"] != self.protocol
                or payload["source_sha256"] != self.source_sha or payload["history"] != self.history
                or payload["environment"] != {"h": 2, "f": 4, "reservation": "soft"}):
            raise ValueError("Checkpoint protocol/source/observation mismatch")
        self.scores.load_state_dict(payload["scores"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.policy_updates, self.checkpoint_meta = payload["policy_updates"], payload["metadata"]
        torch.set_rng_state(payload["rng_torch"])
        np.random.set_state(payload["rng_numpy"])
        random.setstate(payload["rng_python"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(payload["rng_cuda"], self.device)
        self.behavior_sha = digest_file(path)
        return payload
