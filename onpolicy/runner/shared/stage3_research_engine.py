"""Common rollout/replay engine for the diagnostic-first Stage3 protocol.

This deliberately does not alter the legacy runner. Every new arm uses the
same action history, recurrence, terminal objective and case-group reduction.
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from onpolicy.utils.stage3_research import (BASE_MANIFEST, SOURCE, ROOT, WORKSPACE_ROOT, read_json,
    cost_baselines, digest_file)


def policy_args():
    from onpolicy.config.config import get_config
    from onpolicy.scripts.train.train_hkbz import parse_args
    args = parse_args(read_json(BASE_MANIFEST)["command"][2:], get_config())
    # Old command provenance contains an absolute shared-workspace config path.
    # Bind it to this attempt's source copy, not a concurrently edited file.
    config_path = Path(args.ac_config)
    if config_path.is_absolute():
        args.ac_config = str(ROOT / config_path.relative_to(WORKSPACE_ROOT))
    else:
        args.ac_config = str(ROOT / config_path)
    # Preserve the source architecture, not the previous run's auxiliary losses.
    args.shared_actor_lr_scale = 1.0
    args.plane_actor_lr_scale = 0.25
    args.device_actor_lr_scale = 1.0
    args.transporter_actor_lr_scale = 0.5
    args.lr = 5e-6
    args.critic_lr = 1e-4
    args.shared_encoder_activation_checkpoint = True
    return args


def environment_config(case, seed=42):
    from onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels import _env_config
    config = yaml.safe_load((ROOT / "onpolicy/config/env_joint_finetune.yaml").read_text())
    config.update(_env_config(Path(case), max_plane_agents=24, max_device_num=80,
        max_steps=4000, lookahead_margin=60.0, future_intent_horizon=3,
        future_intent_mode="bounded_frontier", frontier_max_requests=4,
        request_capacity_per_plane=5, release_aware_eta=True, reservation_mode="hard",
        reservation_grace_seconds=300.0, slack_forecast_seconds=0.0))
    config.update(seed=int(seed), use_domain_rand=False, resource_policy="drl",
        iga_teacher_dir="", resource_iga_teacher_dir="", resource_iga_teacher_index="",
        joint_iga_teacher_dir="", joint_iga_teacher_index="", pair_feature_storage="sparse_legal")
    return config


def _environment_worker(connection):
    """Byte IPC avoids retaining thousands of torch shared-storage descriptors."""
    from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
    from onpolicy.envs.HKBZ.experiment.eval_common import completion_details
    torch.set_num_threads(1)
    env = None
    steps = 0
    diagnostics = False
    actual_resets = 0
    try:
        while True:
            command, payload = pickle.loads(connection.recv_bytes())
            if command == "close":
                return
            try:
                if command == "reset":
                    if env is not None:
                        env.close()
                    payload = dict(payload)
                    diagnostics = bool(payload.pop("_stage3_diagnostics", False))
                    actual_resets = 0
                    env = AircraftScheduleEnv(payload)
                    steps = 0
                    result = env.reset(seed=payload["seed"])
                elif command == "step":
                    before_jobs = [(p, set(p.finished_jobs) & set(p.LONG_OCCUPANCY_JOBS))
                                   for p in env.planes.values()] if diagnostics else []
                    result = env.step(payload)
                    actual_resets += sum(len(jobs - set(p.finished_jobs)) for p, jobs in before_jobs)
                    steps += 1
                elif command == "summary":
                    result = completion_details(env, steps, 4000)
                    result["makespan"] = float(env.total_time)
                    if diagnostics:
                        resource = env.get_resource_lateness_metrics()
                        result["local_diagnostics"] = {
                            "actual_long_job_reset_count": actual_resets,
                            "repeat_relocation_count": sum(bool(r.get("repeat_relocation")) for r in env.trajectory_log),
                            "relocations_without_irreversible_progress": sum(int(r.get("relocation_count", 0))
                                for r in env.trajectory_log if not r.get("irreversible_progress_delta", 0)),
                            **{key: float(resource.get(key, 0.)) for key in (
                                "total_wait_seconds", "p95_wait_seconds_per_aircraft", "critical_wait_seconds",
                                "policy_defer_seconds", "predicted_lateness_seconds", "early_arrival_seconds")}}
                else:
                    raise ValueError(command)
                connection.send_bytes(pickle.dumps((True, result), protocol=5))
            except Exception as exc:
                import traceback
                connection.send_bytes(pickle.dumps((False, traceback.format_exc()), protocol=5))
                return
    except EOFError:
        pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class EnvironmentPool:
    def __init__(self, width=8, timeout=300):
        self.timeout = timeout
        self.connections, self.processes = [], []
        context = mp.get_context("spawn")
        for _ in range(width):
            parent, child = context.Pipe()
            process = context.Process(target=_environment_worker, args=(child,), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def call(self, commands):
        for index, command, payload in commands:
            self.connections[index].send_bytes(pickle.dumps((command, payload), protocol=5))
        results = {}
        for index, _, _ in commands:
            connection = self.connections[index]
            if not connection.poll(self.timeout):
                raise TimeoutError(f"Environment {index} stalled for {self.timeout}s")
            ok, payload = pickle.loads(connection.recv_bytes())
            if not ok:
                raise RuntimeError(f"Environment {index} failed:\n{payload}")
            results[index] = payload
        return results

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            try:
                connection.send_bytes(pickle.dumps(("close", None)))
            except (BrokenPipeError, OSError):
                pass
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()  # Only this pool's own stuck child.
                process.join(timeout=2)
            connection.close()


def authoritative_history(previous, infos, planes=24):
    history = np.asarray(previous).copy()
    for column, name in enumerate(("last_op_indices", "last_site_indices")):
        history[:, :planes, column] = np.stack([i[name] for i in infos]).reshape(len(infos), -1)[:, :planes]
    return history


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_mode(module):
    """Disable stochastic dropout but request cuDNN's recurrent backward buffers.

    A cuDNN GRU evaluated with training=False has no backward reserve space.
    Switching only dropout-free RNN modules preserves the rollout distribution.
    """
    module.eval()
    for child in module.modules():
        if isinstance(child, torch.nn.RNNBase):
            if child.num_layers > 1 and child.dropout:
                raise ValueError("Recurrent dropout would break likelihood replay consistency")
            child.train(True)


class ResearchEngine:
    def __init__(self, checkpoint=SOURCE, *, width=8, device="cuda:0", freeze_shared=True,
                 exploration=None, diagnostics=False, cuda_memory_fraction=.80,
                 environment_overrides=None, semantic_history=False):
        cuda_memory_fraction = float(cuda_memory_fraction)
        if not 0 < cuda_memory_fraction <= .80:
            raise ValueError("CUDA allocator fraction must be finite and in (0, 0.80]")
        from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
        from onpolicy.utils.checkpoint_contract import validate_stage1_checkpoint_contract
        from onpolicy.utils.valuenorm import ValueNorm
        self.freeze_shared = freeze_shared
        self.diagnostics = bool(diagnostics)
        self.exploration = None
        self._role_exploration = None
        self.last_decision_stats = {}
        self.environment_overrides = dict(environment_overrides or {})
        self.semantic_history = bool(semantic_history)
        if self.semantic_history:
            from onpolicy.runner.shared.stage3_local_improvement_engine import LocalEnvironmentPool
            self.pool = LocalEnvironmentPool(width)
        else:
            self.pool = EnvironmentPool(width)
        self.width = width
        self.device = torch.device(device)
        torch.set_num_threads(1)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, self.device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.args = policy_args()
        self.policy = GNN_MAPPOPolicy(self.args,
            yaml.safe_load(Path(self.args.ac_config).read_text()), device=self.device)
        self.norms = {role: ValueNorm(1, device=self.device) for role in range(3)}
        self.load(checkpoint)
        validate_stage1_checkpoint_contract(self.source_metadata,
            global_feature_mode="f1f2", plane_order_mode="fixed", plane_pair_decoder="joint_pair",
            stage1_baseline="proposed", strict_metadata=True)
        self.policy.set_resource_joint_training_stage(freeze_plane=False, freeze_shared=freeze_shared)
        self.configure_exploration(exploration if exploration is not None else self.exploration)
        self.policy.ac.eval()  # Same dropout mode for collection and likelihood replay.

    def configure_exploration(self, config):
        from onpolicy.utils.stage3_local_exploration import exploration_config, RoleExploration
        if self._role_exploration is not None:
            self._role_exploration.close()
            self._role_exploration = None
        self.exploration = exploration_config(config)
        if self.exploration is not None:
            self._role_exploration = RoleExploration(self.policy.ac, self.exploration)
            self.policy.ac.tau = self.exploration["taus"][0]
            self.policy.set_resource_joint_training_stage(
                freeze_plane=0 not in self.exploration["roles"], freeze_shared=self.freeze_shared)

    def set_actor_lr(self, lr):
        self.policy.lr = float(lr)
        for group in self.policy.actor_optimizer.param_groups:
            group["lr"] = float(lr) * group.get("lr_scale", 1.)

    def load(self, checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.policy.ac.load_state_dict(payload["model"], strict=True)
        self.policy_updates = int(payload.get("policy_updates", 0))
        self.policy.ac.tau = 0.3
        for role, norm in self.norms.items():
            norm.load_state_dict(payload.get("role_value_normalizers", {}).get(str(role),
                                 payload["value_normalizer"]))
        self.source_metadata = {k: copy.deepcopy(payload[k]) for k in
            ("global_feature_mode", "observation_schema_id", "observation_schema",
             "environment_semantics_version", "plane_order_mode", "plane_pair_decoder") if k in payload}
        self.checkpoint = str(Path(checkpoint).resolve())
        self.configure_exploration(payload.get("exploration"))

    def resume(self, checkpoint, *, protocol_sha256, exploration):
        """Explicit full training resume, not initialization from a model file."""
        from onpolicy.utils.stage3_local_exploration import exploration_config
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload.get("protocol_sha256") != protocol_sha256
                or payload.get("exploration") != exploration_config(exploration)
                or payload.get("freeze_shared") != self.freeze_shared):
            raise ValueError("Resume protocol/exploration/trainability mismatch")
        required = ("actor_optim", "critic_optim", "rng_torch", "rng_numpy", "rng_python",
                    "rng_cuda", "next_group", "elite_buffer", "elite_usage", "auxiliary_steps")
        if any(key not in payload for key in required):
            raise ValueError("Incomplete training checkpoint cannot resume")
        self.load(checkpoint)
        self.policy.actor_optimizer.load_state_dict(payload["actor_optim"])
        self.policy.critic_optimizer.load_state_dict(payload["critic_optim"])
        self.policy.lr = payload["actor_base_lr"]
        torch.set_rng_state(payload["rng_torch"])
        np.random.set_state(payload["rng_numpy"])
        random.setstate(payload["rng_python"])
        if self.device.type == "cuda":
            if payload["rng_cuda"] is None:
                raise ValueError("CUDA resume requires a saved CUDA RNG")
            torch.cuda.set_rng_state(payload["rng_cuda"], self.device)
        return payload

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Immutable checkpoint exists: {path}")
        temporary = path.with_suffix(".tmp.pt")
        payload = {**self.source_metadata, **metadata, "model": self.policy.ac.state_dict(),
            "value_normalizer": self.norms[0].state_dict(),
            "role_value_normalizers": {str(r): n.state_dict() for r, n in self.norms.items()},
            "actor_optim": self.policy.actor_optimizer.state_dict(),
            "critic_optim": self.policy.critic_optimizer.state_dict(), "tau": self.policy.ac.tau,
            "exploration": self.exploration, "freeze_shared": self.freeze_shared,
            "policy_updates": self.policy_updates,
            "actor_base_lr": self.policy.lr,
            "rng_cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate()}
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        return digest_file(path)

    def _inputs(self, observations, hidden, infos, previous):
        history = authoritative_history(previous, infos)
        active = np.stack([i["active_agents"] for i in infos]).reshape(len(infos), -1)
        roles = np.stack([i["agent_types"] for i in infos]).reshape(len(infos), -1)
        return observations, hidden, active, history[:, :, 0], history[:, :, 1], roles

    @torch.no_grad()
    def rollout(self, cases, seeds, *, deterministic=False, retain=False, forced=None,
                heartbeat=None, max_steps=4000, sampling_window=None):
        """Independent per-trajectory RNG streams, invariant to ragged batch lengths.

        Stochastic action calls use per-replica RNG state (batch size one);
        greedy evaluation batches all live replicas. Update always batches them.
        forced is a list of action prefixes; normal policy completes each suffix.
        """
        if not 0 < len(cases) <= self.width or len(cases) != len(seeds):
            raise ValueError("Invalid rollout batch")
        self.policy.ac.eval()
        if self._role_exploration is not None:
            from onpolicy.utils.stage3_sampling_audit import DecisionStats
            self._role_exploration.stats = DecisionStats()
        count, agents = len(cases), 104
        if self.diagnostics:
            from onpolicy.utils.stage3_research import verify_cases
            verify_cases(list({c["path"]: c for c in cases}.values()))
        initial = self.pool.call([(i, "reset", {**environment_config(c["path"], s),
                                               **self.environment_overrides,
                                               "_stage3_diagnostics": self.diagnostics})
                                for i, (c, s) in enumerate(zip(cases, seeds))])
        obs = [initial[i][0] for i in range(count)]
        infos = [initial[i][2] for i in range(count)]
        hidden = np.zeros((count, agents, 1, 64), np.float32)
        previous = np.full((count, agents, 3), -1, np.int64)
        old_request_keys = [[] for _ in cases]
        live = np.ones(count, dtype=bool)
        trajectories = [{"case_id": c["path"], "profile": c["profile"],
            "distribution": c["distribution"], "seed": int(s), "actions": [], "states": [],
            "exploration": copy.deepcopy(self.exploration),
            "policy_updates": self.policy_updates,
            "behavior_deterministic": deterministic, "forced_replay": forced is not None}
            for c, s in zip(cases, seeds)]
        if self.semantic_history:
            for trajectory in trajectories:
                trajectory.update(history="stable_request_identity_v1", history_audit={})
        rng = []
        for seed in seeds:
            seed_all(seed)
            rng.append(torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else torch.get_rng_state())
        for step in range(max_steps):
            indices = np.flatnonzero(live).tolist()
            if not indices:
                break
            if self.semantic_history:
                from onpolicy.utils.stage3_local_improvement import remap_history
                for i in indices:
                    from onpolicy.utils.stage3_performance import tensor_bytes
                    trajectories[i]["max_graph_tensor_bytes"] = max(
                        trajectories[i].get("max_graph_tensor_bytes", 0), tensor_bytes(obs[i].to_dict()))
                    keys = infos[i]["stage3_request_keys"]
                    previous[i], audit = remap_history(previous[i], old_request_keys[i], keys,
                        np.asarray(infos[i]["active_agents"]).reshape(-1))
                    old_request_keys[i] = keys
                    for name, value in audit.items():
                        audits = trajectories[i]["history_audit"]
                        audits[name] = audits.get(name, 0) + value
            groups = [indices] if deterministic and forced is None else [[i] for i in indices]
            actions_to_step = {}
            for group in groups:
                if not deterministic:
                    if self.device.type == "cuda":
                        torch.cuda.set_rng_state(rng[group[0]], self.device)
                    else:
                        torch.set_rng_state(rng[group[0]])
                inputs = self._inputs([obs[i] for i in group], hidden[group],
                                      [infos[i] for i in group], previous[group])
                graph, h, active, op, site, roles = inputs
                values, actions, logp, new_h, mask = self.policy.get_actions(
                    graph, h, active, op, site, deterministic=(deterministic or
                        (sampling_window is not None and not sampling_window[0] <= step < sampling_window[1])),
                    agent_types=roles, return_decision_mask=True)
                if not deterministic:
                    rng[group[0]] = (torch.cuda.get_rng_state(self.device)
                        if self.device.type == "cuda" else torch.get_rng_state())
                actions = actions.cpu().numpy().astype(np.int64)
                if actions.shape[-1] == 2:
                    actions = np.concatenate([actions, np.full((*actions.shape[:-1], 1), -1, np.int64)], axis=-1)
                if forced is not None and step < len(forced[group[0]]):
                    given = np.asarray(forced[group[0]][step], dtype=np.int64)
                    if given.shape[-1] == 2:
                        given = np.concatenate([given, np.full((agents, 1), -1, np.int64)], axis=-1)
                    actions[0] = given
                    logp, _, mask, new_h = self.policy.evaluate_actions(
                        graph, h, active, op, site, actions, agent_types=roles,
                        return_decision_mask=True, return_rnn_states=True)
                raw_values = values.detach().cpu().numpy().reshape(len(group), agents)
                for role, norm in self.norms.items():
                    selected = roles == role
                    raw_values[selected] = norm.denormalize(raw_values[selected, None]).reshape(-1)
                for j, i in enumerate(group):
                    trajectories[i]["actions"].append(actions[j].tolist())
                    if retain:
                        trajectories[i]["states"].append({"graph": obs[i], "hidden": hidden[i].copy(),
                            "active": active[j].copy(), "op": op[j].copy(), "site": site[j].copy(),
                            "roles": roles[j].copy(), "action": actions[j].copy(),
                            "old_logp": logp[j].cpu().numpy().reshape(agents),
                            "mask": mask[j].cpu().numpy().reshape(agents),
                            "value": raw_values[j].copy(), "time": float(infos[i]["env_total_time"])})
                    actions_to_step[i] = actions[j]
                    hidden[i] = new_h[j].cpu().numpy()
                    previous[i] = actions[j]
            results = self.pool.call([(i, "step", actions_to_step[i]) for i in indices])
            for i in indices:
                obs[i], _, done, infos[i] = results[i]
                if np.all(done):
                    live[i] = False
            if heartbeat and step % 50 == 0:
                heartbeat.update(event="rollout", rollout_step=step, live_trajectories=int(live.sum()))
        summaries = self.pool.call([(i, "summary", None) for i in range(count)])
        for i, trajectory in enumerate(trajectories):
            trajectory.update(summaries[i])
            if not trajectory["completed"] or trajectory.get("cycle_terminated"):
                raise RuntimeError(f"Incomplete trajectory, group rejected: {trajectory['case_id']}: {summaries[i]}")
        if self._role_exploration is not None:
            self.last_decision_stats = self._role_exploration.stats.result()
            self._role_exploration.stats = None
        return trajectories

    def update(self, trajectories, mode, source_cost=None, *, epochs=2, chunk=8,
               imitation_weight=0.0, heartbeat=None, allow_multi_case=False,
               clip_mode="global", gradient_callback=None, diagnostic_role=None,
               distributed=None, visit_balanced=False):
        """PPO with case -> trajectory mean, time/agent SUM (no length weighting).

        R0 is the MC-return value-baseline PPO control, gamma=1. R1/R2 replace
        its actor baseline entirely; the critic still fits raw cost-to-go.
        No advantage centering, case-wise standardization, or positive filtering.
        """
        count = len(trajectories)
        from onpolicy.utils.stage3_local_exploration import training_mask
        if self.exploration is not None:
            if mode not in ("source", "bc"):
                raise ValueError("Local pilot fixes the source baseline")
            for trajectory in trajectories:
                if trajectory.get("exploration") != self.exploration:
                    raise ValueError("Rollout and replay exploration contracts differ")
                if mode != "bc" and (trajectory.get("behavior_deterministic") or trajectory.get("forced_replay")):
                    raise ValueError("Greedy/forced trajectories are not on-policy PPO samples")
                if mode != "bc" and trajectory.get("policy_updates") != self.policy_updates:
                    raise ValueError("Stale policy-version trajectories cannot start a new PPO update")
        case_ids = [t["case_id"] for t in trajectories]
        if len(set(case_ids)) != 1 and not allow_multi_case:
            raise ValueError("An update group must contain exactly one case")
        if allow_multi_case and mode not in ("source", "bc"):
            raise ValueError("Multi-case updates are defined only for source-relative PPO or BC")
        if clip_mode not in ("global", "per_group"):
            raise ValueError("Unknown clipping intervention")
        # Uniform cases, then uniform trajectories within each case. Keep the
        # historical time/role SUM; do not introduce episode-length weighting.
        weights = np.asarray([1. / (len(set(case_ids)) * case_ids.count(case)) for case in case_ids])
        if visit_balanced:
            weights = np.full(count, 1. / count)
        costs = np.asarray([t["makespan"] for t in trajectories])
        if mode == "source" and isinstance(source_cost, dict):
            baselines = np.asarray([cost_baselines([cost], mode, source_cost[case])[0]
                                    for cost, case in zip(costs, case_ids)])
        else:
            baselines = None if mode in ("value", "bc") else cost_baselines(costs, mode, source_cost)
        if any(not t["states"] or not t["completed"] for t in trajectories):
            raise ValueError("Only complete retained trajectories can train")
        device = self.device
        tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=device)
        steps = max(len(t["states"]) for t in trajectories)
        if diagnostic_role is not None and gradient_callback is None:
            raise ValueError("Role-only objectives are restricted to non-stepping gradient diagnostics")
        actor_mask = lambda state: (training_mask(state["mask"], state["roles"], self.exploration)
            * (state["roles"] == diagnostic_role) if diagnostic_role is not None
            else training_mask(state["mask"], state["roles"], self.exploration))
        total_decisions = sum(float(actor_mask(s).sum()) for t in trajectories for s in t["states"])
        total_env_decisions = sum(float(s["mask"].sum()) for t in trajectories for s in t["states"])
        if distributed is not None:
            if mode != "source" or imitation_weight or diagnostic_role is not None:
                raise ValueError("Distributed update is restricted to unmodified source-relative PPO")
            # Averaged replica gradients must equal the global case/trajectory
            # objective, even with uneven shards or unequal trajectory lengths.
            weights, total_env_decisions = distributed.objective_weights(case_ids, total_env_decisions,
                                                                         visit_balanced=visit_balanced)
        if total_decisions <= 0:
            raise ValueError("Group has no trainable decisions")
        metrics = []
        deferred = None
        if getattr(self, "defer_statistics", False):
            from onpolicy.utils.stage3_performance import DeferredScalars
            deferred = DeferredScalars()
        for epoch in range(epochs):
            gradient_mode(self.policy.ac)
            self.policy.actor_optimizer.zero_grad(set_to_none=True)
            self.policy.critic_optimizer.zero_grad(set_to_none=True)
            hidden = tensor(np.stack([t["states"][0]["hidden"] for t in trajectories]))
            actor_loss = critic_loss = None
            stats = dict(kl=0.0, clip=0.0, logp=0.0, decisions=0.0, mask_mismatch=0)
            health = dict(nonfinite=0)
            def statistic(key, value):
                if deferred is None:
                    stats[key] += float(value.detach())
                else:
                    deferred.add(stats, key, value)
            for step in range(steps):
                indices = [i for i, t in enumerate(trajectories) if step < len(t["states"])]
                states = [trajectories[i]["states"][step] for i in indices]
                stack = lambda key: np.stack([s[key] for s in states])
                graph, active, op, site = [s["graph"] for s in states], stack("active"), stack("op"), stack("site")
                roles, actions = stack("roles"), stack("action")
                logp, entropy, decision_mask, new_h = self.policy.evaluate_actions(
                    graph, hidden[indices], active, op, site, actions, agent_types=roles,
                    return_decision_mask=True, return_rnn_states=True)
                logp = logp.reshape(len(indices), -1)
                env_mask = tensor(stack("mask"))
                mask = tensor(np.stack([actor_mask(s) for s in states]))
                mismatch = (decision_mask.reshape_as(env_mask) != env_mask).sum()
                if deferred is None:
                    if not torch.isfinite(logp).all():
                        raise FloatingPointError("Nonfinite action likelihood")
                    stats["mask_mismatch"] += int(mismatch.item())
                else:
                    deferred.add(health, "nonfinite", (~torch.isfinite(logp)).sum())
                    deferred.add(stats, "mask_mismatch", mismatch)
                # Index-copy preserves recurrent gradients inside a TBPTT chunk.
                hidden = hidden.index_copy(0, tensor(indices, torch.long), new_h)
                remaining = tensor(np.asarray([-.01 * (costs[i] - s["time"])
                                              for i, s in zip(indices, states)]))[:, None]
                if mode == "bc":
                    loss = -(logp * mask).sum() / total_decisions
                else:
                    advantage = (remaining - tensor(stack("value"))) if mode == "value" else tensor(
                        .01 * (baselines[indices] - costs[indices]))[:, None]
                    ratio_log = logp - tensor(stack("old_logp"))
                    # Clamp only numerical exponent range, not the advantage sign.
                    ratio = torch.exp(ratio_log.clamp(-40, 40))
                    surrogate = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                    loss = (-(surrogate * mask).sum() / count if len(set(case_ids)) == 1 and distributed is None
                            else -(surrogate * mask * tensor(weights[indices])[:, None]).sum())
                    if imitation_weight:
                        loss -= float(imitation_weight) * (logp * mask).sum() / total_decisions
                    statistic("kl", (((ratio - 1) - ratio_log) * mask).sum())
                    statistic("clip", (((ratio - 1).abs() > .2) * mask).sum())
                statistic("logp", (logp.detach() * mask).sum())
                statistic("decisions", mask.sum())
                actor_loss = loss if actor_loss is None else actor_loss + loss
                if mode != "bc":
                    values = self.policy.evaluate_values(graph, tensor(stack("hidden")), active,
                        op, site, actions, agent_types=roles).reshape(len(indices), -1)
                    targets = torch.zeros_like(values)
                    for role, norm in self.norms.items():
                        target = norm.normalize(remaining.reshape(-1, 1)).reshape(-1, 1)
                        targets = torch.where(tensor(roles == role, torch.bool), target, targets)
                    c_loss = ((values - targets.detach()).square() * env_mask).sum() / total_env_decisions
                    critic_loss = c_loss if critic_loss is None else critic_loss + c_loss
                if (step + 1) % chunk == 0 or step + 1 == steps:
                    if deferred is not None:
                        deferred.flush()
                        if health["nonfinite"]:
                            raise FloatingPointError("Nonfinite action likelihood")
                        if stats["mask_mismatch"]:
                            raise RuntimeError(f"Replay decision-mask mismatch: {stats['mask_mismatch']}")
                    if actor_loss.requires_grad:
                        actor_loss.backward()
                    if critic_loss is not None and critic_loss.requires_grad:
                        critic_loss.backward()
                    actor_loss = critic_loss = None
                    hidden = hidden.detach()
                if heartbeat and step % 50 == 0:
                    heartbeat.update(event="update", update_epoch=epoch, update_step=step)
            if stats["mask_mismatch"]:
                raise RuntimeError(f"Replay decision-mask mismatch: {stats['mask_mismatch']}")
            if distributed is not None:
                # Synchronize BEFORE global clipping, gradient diagnostics,
                # Adam steps and KL decisions; not a local-SGD approximation.
                distributed.synchronize_update(self.policy, stats)
            stats["kl"] /= stats["decisions"]
            stats["clip"] /= stats["decisions"]
            stats["nll"] = -stats.pop("logp") / stats["decisions"]
            stats["gradient_norm_by_group"] = {}
            for group in self.policy.actor_optimizer.param_groups:
                squares = [p.grad.detach().square().sum() for p in group["params"] if p.grad is not None]
                stats["gradient_norm_by_group"][group["name"]] = (
                    float(torch.stack(squares).sum().sqrt()) if squares else 0.0)
            parameters = [p for group in self.policy.actor_optimizer.param_groups
                          for p in group["params"] if p.requires_grad and p.grad is not None]
            if gradient_callback is not None:
                gradient_callback(epoch, self.policy.ac, stats)
                if diagnostic_role is not None:
                    raise RuntimeError("A role-gradient capture must abort before any optimizer step")
            if clip_mode == "global":
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            else:
                norm = sum(v*v for v in stats["gradient_norm_by_group"].values()) ** .5
                for group in self.policy.actor_optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_([p for p in group["params"] if p.grad is not None],
                                                   1.0, error_if_nonfinite=True)
            stats["clip_mode"] = clip_mode
            stats["actor_gradient_norm"] = float(norm)
            # Abort an epoch before applying a second excessive PPO move. Never rollback silently.
            stats["actor_step_applied"] = mode == "bc" or stats["kl"] <= .02
            if stats["actor_step_applied"]:
                self.policy.actor_optimizer.step()
                self.policy_updates += 1
            if mode != "bc":
                torch.nn.utils.clip_grad_norm_(self.policy.ac.critic_param.parameters(), 1.0,
                                               error_if_nonfinite=True)
                self.policy.critic_optimizer.step()
            if self.exploration is not None:
                stats["post_update"] = (self.replay_metrics(trajectories, heartbeat=heartbeat)
                    if distributed is None else self.replay_metrics(
                        trajectories, heartbeat=heartbeat, distributed=distributed))
            metrics.append(stats)
            if not stats["actor_step_applied"] or stats.get("post_update", {}).get("kl", 0.) > .02:
                break
        advantages = [.01 * (b - c) for b, c in zip(baselines, costs)] if baselines is not None else None
        return {"epochs": metrics, "costs": costs.tolist(), "baseline_mode": mode,
                "case_ids": case_ids, "case_trajectory_weights": weights.tolist(),
                "trajectory_advantages": advantages,
                "positive_advantage_fraction": float(np.mean(np.asarray(advantages) > 0)) if advantages is not None else None,
                "source_cost": {key:source_cost[key] for key in set(case_ids)} if isinstance(source_cost, dict) else source_cost,
                "decisions": total_decisions,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0}

    def close(self):
        if self._role_exploration is not None:
            self._role_exploration.close()
        self.pool.close()

    @torch.no_grad()
    def replay_metrics(self, trajectories, heartbeat=None, *, distributed=None):
        """Rebuild current recurrent history; report trainable-role post-step KL."""
        from onpolicy.utils.stage3_local_exploration import training_mask
        self.policy.ac.eval()
        device = self.device
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, device=device, dtype=dtype)
        hidden = tensor(np.stack([t["states"][0]["hidden"] for t in trajectories]))
        sums = {r: {"decisions": 0, "kl_sum": 0., "nll_sum": 0., "max_logp_error": 0.}
                for r in range(3)}
        deferred = None
        if getattr(self, "defer_statistics", False):
            from onpolicy.utils.stage3_performance import DeferredScalars
            deferred = DeferredScalars()
        health = dict(nonfinite=0, mask_mismatch=0)
        steps = max(len(t["states"]) for t in trajectories)
        for step in range(steps):
            indices = [i for i, t in enumerate(trajectories) if step < len(t["states"])]
            states = [trajectories[i]["states"][step] for i in indices]
            stack = lambda key: np.stack([s[key] for s in states])
            lp, _, decision, new_h = self.policy.evaluate_actions(
                [s["graph"] for s in states], hidden[indices], stack("active"), stack("op"),
                stack("site"), stack("action"), agent_types=stack("roles"),
                return_decision_mask=True, return_rnn_states=True)
            hidden = hidden.index_copy(0, tensor(indices, torch.long), new_h)
            lp = lp.reshape(len(indices), -1)
            if deferred is None:
                if not torch.isfinite(lp).all() or not np.array_equal(
                        decision.cpu().numpy().reshape(len(indices), -1), stack("mask")):
                    raise RuntimeError("Nonfinite likelihood or legal decision-mask drift")
            else:
                deferred.add(health, "nonfinite", (~torch.isfinite(lp)).sum())
                deferred.add(health, "mask_mismatch", (decision.reshape_as(lp) != tensor(stack("mask"))).sum())
            diff = lp - tensor(stack("old_logp"))
            mask = training_mask(stack("mask"), stack("roles"), self.exploration).astype(bool)
            for role, row in sums.items():
                selected_cpu = mask & (stack("roles") == role)
                selected = tensor(selected_cpu, torch.bool)
                n = int(selected_cpu.sum()) if deferred is not None else int(selected.sum())
                if n:
                    d = diff[selected]
                    row["decisions"] += n
                    if deferred is None:
                        row["kl_sum"] += float((d.clamp(-40, 40).exp() - 1 - d).sum())
                        row["nll_sum"] -= float(lp[selected].sum())
                        row["max_logp_error"] = max(row["max_logp_error"], float(d.abs().max()))
                    else:
                        deferred.add(row, "kl_sum", (d.clamp(-40, 40).exp() - 1 - d).sum())
                        deferred.add(row, "nll_sum", -lp[selected].sum())
                        deferred.add(row, "max_logp_error", d.abs().max(), maximum=True)
            if deferred is not None and ((step + 1) % 8 == 0 or step + 1 == steps):
                deferred.flush()
                if any(health.values()):
                    raise RuntimeError("Nonfinite likelihood or legal decision-mask drift")
            if heartbeat and step % 50 == 0:
                heartbeat.update(event="post_update_replay", replay_step=step)
        if distributed is not None:
            sums = distributed.replay_sums(sums)
        n = sum(r["decisions"] for r in sums.values())
        return {"kl": sum(r["kl_sum"] for r in sums.values()) / max(n, 1),
                "nll": sum(r["nll_sum"] for r in sums.values()) / max(n, 1),
                "max_logp_error": max(r["max_logp_error"] for r in sums.values()),
                "roles": {str(k): {**r, "kl": r["kl_sum"] / max(r["decisions"], 1)} for k, r in sums.items()}}

    @torch.no_grad()
    def likelihood_metrics(self, trajectory):
        sums = {role: {"decisions": 0, "nll_sum": 0., "matches": 0,
                       "impossible": 0, "logp_replay_max_error": 0.} for role in range(3)}
        for state in trajectory["states"]:
            graph, h = [state["graph"]], state["hidden"][None]
            active, op, site, roles, actions = [state[k][None] for k in
                                               ("active", "op", "site", "roles", "action")]
            logp, _, mask = self.policy.evaluate_actions(graph, h, active, op, site, actions,
                agent_types=roles, return_decision_mask=True)
            greedy, _ = self.policy.act(graph, h, active, op, site, agent_types=roles,
                                         deterministic=True)
            lp = logp.cpu().numpy().reshape(-1)
            decision = mask.cpu().numpy().reshape(-1).astype(bool)
            predicted = greedy.cpu().numpy()[0]
            for role, result in sums.items():
                chosen = (roles[0] == role) & decision
                if not chosen.any():
                    continue
                result["decisions"] += int(chosen.sum())
                result["nll_sum"] -= float(lp[chosen].sum())
                dimensions = 2 if role == 0 else 1
                result["matches"] += int(np.all(predicted[chosen, :dimensions]
                    == actions[0, chosen, :dimensions], axis=-1).sum())
                result["impossible"] += int((lp[chosen] < -1e6).sum())
                result["logp_replay_max_error"] = max(result["logp_replay_max_error"],
                    float(np.abs(lp[chosen] - state["old_logp"][chosen]).max()))
        for result in sums.values():
            result["nll"] = result["nll_sum"] / max(1, result["decisions"])
            result["accuracy"] = result["matches"] / max(1, result["decisions"])
        return sums
