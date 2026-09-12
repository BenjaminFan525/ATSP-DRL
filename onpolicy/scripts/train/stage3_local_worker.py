#!/usr/bin/env python3
"""Workers for the local-exploration protocol; every phase has bounded output."""
from __future__ import annotations
import argparse
import copy
import gc
import math
import os
import random
import time
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from onpolicy.utils.stage3_research import (atomic_json, read_json, digest_file, digest_json,
    trajectory_seed, best_of_n, paired_summary, pilot_gate, verify_cases)
from onpolicy.utils.stage3_sampling_audit import action_digest, sampling_heads
from onpolicy.utils.stage3_local_exploration import (MODES, ARMS, EVAL_PROTOCOL,
    exploration_config, case_schedule, choose_elite, imitation_gate, LocalEvaluationQueue)
from onpolicy.scripts.train.run_stage3_sampling_audit import (ProgressHeartbeat, model_digest,
    save_group, trajectory_record)
from onpolicy.scripts.train.run_stage3_local_exploration import verify
from onpolicy.utils.stage3_local_resources import memory_fraction, pilot_resource_plan, validate_worker_placement


class ArmGuardStop(RuntimeError):
    """A prespecified learning safety stop, not a corrupted shared protocol."""


def engine(manifest, mode=None, width=8):
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    return ResearchEngine(manifest["source"]["path"], width=width, exploration=mode, diagnostics=True,
                          cuda_memory_fraction=memory_fraction(os.environ.get("HKBZ_STAGE3_CUDA_MEMORY_FRACTION", ".80")))


def reset_c0(runner, manifest, mode, lr):
    runner.load(manifest["source"]["path"])
    runner.configure_exploration(mode)
    runner.policy.reset_optimizers()
    runner.set_actor_lr(lr)


def evaluate(runner, cases, heartbeat):
    rows = []
    for first in range(0, len(cases), runner.width):
        batch = cases[first:first + runner.width]
        rows.extend(trajectory_record(t) for t in runner.rollout(batch, [42] * len(batch),
                                                               deterministic=True, heartbeat=heartbeat))
        heartbeat.update(event="greedy_evaluation", completed_cases=len(rows), total_cases=len(cases))
    return rows


def load_group(reference, case=None, first=None):
    path = Path(reference["path"])
    if digest_file(path) != reference["sha256"]:
        raise ValueError("Committed trajectory metadata changed")
    group = read_json(path)
    trace = group["trace"]
    if digest_file(trace["path"]) != trace["sha256"]:
        raise ValueError("Committed trajectory trace changed")
    rows = []
    with np.load(trace["path"], allow_pickle=False) as arrays:
        if len(arrays.files) != len(group["trajectories"]):
            raise ValueError("Trace count mismatch")
        for i, row in enumerate(group["trajectories"]):
            actions = arrays[f"trajectory_{i}_seed_{row['seed']}"]
            if (list(actions.shape) != row["actions_shape"] or action_digest(actions) != row["action_sha256"]
                    or not row["completed"] or row.get("cycle_terminated")
                    or len(actions) != row["steps"] or not math.isfinite(row["makespan"]) or row["makespan"] <= 0):
                raise ValueError("Invalid completed action trace")
            if case is not None and (row["case_id"] != case["path"] or row["seed"] != trajectory_seed(
                    20260909, case["content_sha256"], 0, first + i)):
                raise ValueError("Archive case/seed schedule mismatch")
            rows.append({**row, "actions": actions.tolist(), "states": []})
    return rows, group.get("decision_stats", {})


def compact(trajectory):
    return {**{k: v for k, v in trajectory.items() if k not in ("states", "actions")},
            "actions": np.asarray(trajectory["actions"], dtype=np.int32)}


def frozen_state(runner):
    return {name: p.detach().cpu().clone() for name, p in runner.policy.ac.named_parameters() if not p.requires_grad}


def assert_frozen(runner, before):
    now = dict(runner.policy.ac.named_parameters())
    if any(not torch.equal(value, now[name].detach().cpu()) for name, value in before.items()):
        raise RuntimeError("Frozen policy parameter changed")


def nested_close(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.allclose(a.cpu(), b.cpu(), atol=2e-6, rtol=2e-5)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(nested_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(nested_close(x, y) for x, y in zip(a, b))
    return a == b


def resume_metadata(manifest, **extra):
    return {"protocol_sha256": manifest["manifest_sha256"],
            "source_sha256": manifest["source"]["sha256"], "next_group": 0,
            "elite_buffer": {}, "elite_usage": {}, "auxiliary_steps": 0, **extra}


def contract(manifest, output, mode, heartbeat, smoke=False):
    width = 2 if smoke else 8
    runner = engine(manifest, mode, width)
    started = time.time()
    try:
        heartbeat.update(check="greedy_and_archived_sampling")
        actor_ids = {id(p) for group in runner.policy.actor_optimizer.param_groups for p in group["params"]}
        critic_ids = {id(p) for group in runner.policy.critic_optimizer.param_groups for p in group["params"]}
        if actor_ids & critic_ids:
            raise RuntimeError("Actor/critic optimizer parameter ownership overlaps")
        case = manifest["archive_cases"][0]
        greedy = runner.rollout([case], [42], deterministic=True, heartbeat=heartbeat)[0]
        if abs(greedy["makespan"] - manifest["source_costs"][case["path"]]) > 1e-6:
            raise RuntimeError("C0 greedy did not reproduce")
        archived, _ = load_group(manifest["archives"][mode][case["name"]][0], case, 0)
        begin = time.time()
        group = runner.rollout([case] * width, [r["seed"] for r in archived[:width]], retain=True, heartbeat=heartbeat)
        rollout_seconds = time.time() - begin
        if any(action_digest(a["actions"]) != action_digest(b["actions"])
               or abs(a["makespan"] - b["makespan"]) > 1e-6 for a, b in zip(group, archived)):
            raise RuntimeError("Trainable role configuration differs from the frozen inference audit")
        save_group(output / "c0_sampling.json", group, {"mode": mode, "archive_exact": True})
        replay = runner.replay_metrics(group, heartbeat)
        if replay["max_logp_error"] > .002:
            raise RuntimeError("Initial batched/recurrent log-probability replay mismatch")
        checks = {}
        for lr in manifest["training"]["actor_lr_candidates"]:
            heartbeat.update(check="real_ppo_update", actor_lr=lr)
            reset_c0(runner, manifest, mode, lr)
            before = frozen_state(runner)
            begin = time.time()
            update = runner.update(group, "source", manifest["source_costs"][case["path"]],
                                   epochs=2, heartbeat=heartbeat)
            post_kl = max(max(e["post_update"]["kl"], *(r["kl"] for r in e["post_update"]["roles"].values()))
                          for e in update["epochs"])
            assert_frozen(runner, before)
            checks[str(lr)] = {"passed": post_kl <= .04 and update["peak_reserved_gib"] < 18
                and update["epochs"][0]["actor_step_applied"] and update["epochs"][0]["actor_gradient_norm"] > 0,
                "update": update,
                "seconds": time.time() - begin, "max_role_post_kl": post_kl, "frozen_unchanged": True}
        if not any(c["passed"] for c in checks.values()):
            raise RuntimeError("No stable LR; bounded canary stopped")
        # Exercise current-state forced replay and auxiliary gradients separately.
        heartbeat.update(check="auxiliary_forced_replay")
        winner = min(group, key=lambda t: t["makespan"])
        replayed = runner.rollout([case], [42], deterministic=True, forced=[winner["actions"]],
                                  retain=True, heartbeat=heartbeat)[0]
        if abs(replayed["makespan"] - winner["makespan"]) > 1e-6:
            raise RuntimeError("Forced replay changed terminal cost")
        before = frozen_state(runner)
        runner.set_actor_lr(manifest["training"]["elite_lr"])
        auxiliary = runner.update([replayed], "bc", epochs=1, heartbeat=heartbeat)
        assert_frozen(runner, before)
        del group, replayed, winner
        gc.collect()
        # Compare the SAME next update before/after a full optimizer/RNG restore.
        checkpoint = output / "resume_start.pt"
        heartbeat.update(check="full_resume_next_update")
        metadata = resume_metadata(manifest, diagnostic_only=True, next_group=1,
                                   elite_usage={case["path"]: 1}, auxiliary_steps=1)
        runner.save(checkpoint, **metadata)
        random_expected = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, device=runner.device))
        seeds = [trajectory_seed(2026090612, case["content_sha256"], 0, k) for k in range(2)]
        fresh = runner.rollout([case] * 2, seeds, retain=True, heartbeat=heartbeat)
        runner.update(fresh, "source", manifest["source_costs"][case["path"]], epochs=1, heartbeat=heartbeat)
        expected_model = {k: v.detach().cpu().clone() for k, v in runner.policy.ac.state_dict().items()}
        expected_actor = copy.deepcopy(runner.policy.actor_optimizer.state_dict())
        expected_critic = copy.deepcopy(runner.policy.critic_optimizer.state_dict())
        restored = runner.resume(checkpoint, protocol_sha256=manifest["manifest_sha256"], exploration=mode)
        random_actual = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, device=runner.device))
        if not all(torch.equal(a, b) if torch.is_tensor(a) else a == b for a, b in zip(random_expected, random_actual)):
            raise RuntimeError("RNG resume mismatch")
        if restored["elite_usage"] != metadata["elite_usage"] or restored["next_group"] != 1:
            raise RuntimeError("Schedule/elite state resume mismatch")
        runner.update(fresh, "source", manifest["source_costs"][case["path"]], epochs=1, heartbeat=heartbeat)
        max_error = max(float((v.cpu() - expected_model[k]).abs().max()) for k, v in runner.policy.ac.state_dict().items())
        if max_error > 2e-6 or not nested_close(expected_actor, runner.policy.actor_optimizer.state_dict()) or not nested_close(
                expected_critic, runner.policy.critic_optimizer.state_dict()):
            raise RuntimeError(f"Next-update restore mismatch: model max abs {max_error}")
        del fresh, expected_model, expected_actor, expected_critic
        reset_c0(runner, manifest, mode, 5e-6)
        tune = None
        if mode == "R" and not smoke:
            heartbeat.update(check="full_c0_tune60_reproduction")
            rows = evaluate(runner, manifest["splits"]["tune"], heartbeat)
            errors = [abs(r["makespan"] - manifest["source_costs"][r["case_id"]]) for r in rows]
            if max(errors) > .1:
                raise RuntimeError("Full C0 Tune60 reproduction failed")
            tune = {"cases": rows, "max_abs_error": max(errors)}
        atomic_json(output / "result.json", {"passed": True, "smoke_only": smoke,
            "mode": mode, "archive_exact": True, "initial_replay": replay, "lr_checks": checks,
            "auxiliary_path": auxiliary, "resume_model_max_abs_error": max_error,
            "resume_optimizer_close": True, "resume_rng_exact": True, "c0_tune": tune,
            "group_seconds": rollout_seconds + max(r["seconds"] for r in checks.values()),
            "elapsed_seconds": time.time() - started}, overwrite=False)
    finally:
        runner.close()


def sample(manifest, output, lane, heartbeat):
    tasks = [(mode, split, case) for split in ("train_diag32", "train_probe64")
             for case in manifest["splits"][split] for mode in MODES][lane::4]
    runner = engine(manifest)
    initial = model_digest(runner)
    try:
        for task_n, (mode, split, case) in enumerate(tasks):
            runner.configure_exploration(mode)
            directory = Path(manifest["root"]) / "sampling" / mode / split / case["name"]
            replicas = 32 if split == "train_diag32" else 8
            records, groups = [], []
            begin = time.time()
            for first in range(0, replicas, 8):
                heartbeat.update(event="sample_group", completed_tasks=task_n, total_tasks=len(tasks),
                                 mode=mode, split=split, case=case["name"], replica=first)
                reference = None
                if split == "train_diag32" and case["name"] in manifest["archives"][mode]:
                    reference = manifest["archives"][mode][case["name"]][first // 8]
                    rows, stats = load_group(reference, case, first)
                else:
                    seeds = [trajectory_seed(manifest["sampling"]["seed"], case["content_sha256"], 0, k)
                             for k in range(first, first + 8)]
                    rows = runner.rollout([case] * 8, seeds, heartbeat=heartbeat)
                    stats = runner.last_decision_stats
                path = directory / f"group_{first:02d}.json"
                save_group(path, rows, {"mode": mode, "split": split, "case_content_sha256": case["content_sha256"],
                    "manifest_sha256": manifest["manifest_sha256"], "decision_stats": stats,
                    "reused_from": reference, "exploration": MODES[mode]})
                groups.append({"path": str(path), "sha256": digest_file(path)})
                records.extend(trajectory_record(t) for t in rows)
            if model_digest(runner) != initial:
                raise RuntimeError("Frozen sampling changed model state")
            c0 = manifest["source_costs"][case["path"]]
            atomic_json(directory / "result.json", {"case_id": case["path"], "mode": mode,
                "profile": case["profile"], "distribution": case["distribution"], "source_cost": c0,
                "groups": groups, "trajectories": records, "elapsed_seconds": time.time() - begin,
                **best_of_n([r["makespan"] for r in records], c0)}, overwrite=False)
            heartbeat.update(completed_tasks=task_n + 1)
            print(f"sample {mode} {split} {case['name']} complete", flush=True)
        atomic_json(output / "result.json", {"completed": True, "tasks": len(tasks)}, overwrite=False)
    finally:
        runner.close()


def fit(manifest, output, mode, heartbeat):
    runner = engine(manifest, mode)
    fit_cases = manifest["splits"]["train_fit16"]
    teachers, teacher_costs = {}, []
    try:
        for case in fit_cases:
            result = read_json(Path(manifest["root"]) / "sampling" / mode / "train_diag32" / case["name"] / "result.json")
            winner = min(result["trajectories"], key=lambda t: t["makespan"])
            c0 = manifest["source_costs"][case["path"]]
            teacher_costs.append(min(c0, winner["makespan"]))
            if winner["makespan"] < c0 - 1e-6:
                for reference in result["groups"]:
                    rows, _ = load_group(reference)
                    match = next((r for r in rows if r["seed"] == winner["seed"]), None)
                    if match is not None:
                        teachers[case["path"]] = compact(match)
                        break
                if case["path"] not in teachers:
                    raise RuntimeError("Winning trace missing")
        potential = float(1 - np.mean(teacher_costs) / np.mean([manifest["source_costs"][c["path"]] for c in fit_cases]))
        initial_nll, final_nll, initial_likelihood, final_likelihood = {}, {}, {}, {}
        for case in fit_cases:
            teacher = teachers.get(case["path"])
            if teacher is not None:
                replay = runner.rollout([case], [42], deterministic=True, forced=[teacher["actions"]],
                                        retain=True, heartbeat=heartbeat)
                if abs(replay[0]["makespan"] - teacher["makespan"]) > 1e-6:
                    raise RuntimeError("Initial teacher replay changed cost")
                initial_nll[case["path"]] = runner.replay_metrics(replay, heartbeat)["nll"]
                initial_likelihood[case["path"]] = runner.likelihood_metrics(replay[0])
                del replay
        runner.set_actor_lr(manifest["fit"]["lr"])
        before = frozen_state(runner)
        for visit in range(manifest["fit"]["passes"]):
            for _, case in case_schedule(fit_cases, 2026090613 + visit, passes=1):
                teacher = teachers.get(case["path"])
                if teacher is None:
                    continue
                replay = runner.rollout([case], [42], deterministic=True, forced=[teacher["actions"]],
                                        retain=True, heartbeat=heartbeat)
                if abs(replay[0]["makespan"] - teacher["makespan"]) > 1e-6:
                    raise RuntimeError("Teacher replay changed cost")
                update = runner.update(replay, "bc", epochs=1, heartbeat=heartbeat)
                assert_frozen(runner, before)
                atomic_json(output / "updates" / f"pass{visit:02d}_{case['name']}.json", update, overwrite=False)
                del replay
                gc.collect()
            heartbeat.update(event="fit_pass_completed", completed_passes=visit + 1,
                             total_passes=manifest["fit"]["passes"])
        for case in fit_cases:
            if case["path"] in teachers:
                replay = runner.rollout([case], [42], deterministic=True, forced=[teachers[case["path"]]["actions"]],
                                        retain=True, heartbeat=heartbeat)
                final_nll[case["path"]] = runner.replay_metrics(replay, heartbeat)["nll"]
                final_likelihood[case["path"]] = runner.likelihood_metrics(replay[0])
                del replay
        fit_rows = evaluate(runner, fit_cases, heartbeat)
        probe_rows = evaluate(runner, manifest["splits"]["train_probe64"], heartbeat)
        fit_summary, probe_summary = [paired_summary(rows, manifest["source_costs"]) for rows in (fit_rows, probe_rows)]
        runner.save(output / "diagnostic_fit.pt", diagnostic_only=True, forbidden_as_rl_initialization=True,
                    protocol_sha256=manifest["manifest_sha256"], teacher_cases=len(teachers))
        atomic_json(output / "result.json", {"mode": mode, "teacher_cases": len(teachers),
            "teacher_potential": potential, "initial_nll": initial_nll, "final_nll": final_nll,
            "initial_teacher_forced_by_role": initial_likelihood, "final_teacher_forced_by_role": final_likelihood,
            "fit": {"cases": fit_rows, "summary": fit_summary},
            "probe": {"cases": probe_rows, "summary": probe_summary},
            "gate": imitation_gate(potential, fit_summary, probe_summary),
            "diagnostic_only": True, "forbidden_as_rl_initialization": True}, overwrite=False)
    finally:
        runner.close()


def counterfactual(manifest, output, heartbeat):
    runner = engine(manifest)
    conditions = [([1, 2], .3, 1), ([1, 2], .3, 1), ([0], .03, 1), ([0], .03, 1),
                  ([0, 1, 2], .03, 1), ([0, 1, 2], .03, 1),
                  ([0, 1, 2], .03, 4), ([0, 1, 2], .03, 4)]
    results = []
    try:
        for case in manifest["counterfactual"]["cases"]:
            baseline = runner.rollout([case], [42], deterministic=True, retain=True, heartbeat=heartbeat)[0]
            decisions = [i for i, s in enumerate(baseline["states"]) if np.asarray(s["mask"]).any()]
            positions = [decisions[min(len(decisions) - 1, int(q * len(decisions)))]
                         for q in manifest["counterfactual"]["positions"]]
            branches = []
            for position_n, step in enumerate(positions):
                for k, (roles, tau, window) in enumerate(conditions):
                    seed = trajectory_seed(manifest["counterfactual"]["seed"], case["content_sha256"], position_n, k)
                    with sampling_heads(runner.policy.ac, {"roles": roles, "tau": tau, "top_k": 0}):
                        branch = runner.rollout([case], [seed], forced=[baseline["actions"][:step]],
                            sampling_window=(step, step + window), heartbeat=heartbeat)[0]
                    prefix_equal = np.array_equal(np.asarray(branch["actions"][:step]), np.asarray(baseline["actions"][:step]))
                    if not prefix_equal:
                        raise RuntimeError("Counterfactual prefix changed")
                    row = {"position": step, "roles": roles, "tau": tau, "window_steps": window,
                        "cost_delta": branch["makespan"] - baseline["makespan"], "prefix_exact": True,
                        "relocation_delta": branch.get("total_relocations", 0) - baseline.get("total_relocations", 0),
                        "diagnostic_deltas": {key: value - baseline["local_diagnostics"][key]
                            for key, value in branch["local_diagnostics"].items()},
                        "changed_agents_at_branch": int(np.any(np.asarray(branch["actions"][step]) !=
                            np.asarray(baseline["actions"][step]), axis=-1).sum())}
                    save_group(output / case["name"] / f"position{position_n}_branch{k}.json", [branch], row)
                    branches.append({**row, **trajectory_record(branch)})
            result = {"case_id": case["path"], "source_cost": baseline["makespan"], "branches": branches,
                "scope": "targeted diagnostic; role-dependent legal masks may induce downstream changes; not isolated agent Q"}
            atomic_json(output / case["name"] / "result.json", result, overwrite=False)
            results.append(result)
            del baseline
            gc.collect()
        atomic_json(output / "result.json", {"completed": True, "cases": results}, overwrite=False)
    finally:
        runner.close()


def submit(manifest, checkpoint, arm, episodes, suffix=""):
    cases = manifest["splits"]["tune"]
    request_id = f"{arm}_e{episodes:06d}{suffix}"
    return LocalEvaluationQueue(Path(manifest["root"]) / "validator").submit({
        "request_id": request_id, "checkpoint": str(checkpoint), "checkpoint_sha256": digest_file(checkpoint),
        "training_episodes": episodes, "cases": cases, "cases_sha256": digest_json(cases),
        "code_sha256": manifest["code"]["sha256"], "contract_sha256": manifest["contract_sha256"],
        "tau": .3, "seed": 42, "evaluation_protocol": EVAL_PROTOCOL,
        "training_exploration": MODES[ARMS[arm]["mode"]]})


def train(manifest, output, arm, heartbeat, resume_checkpoint=None, suffix=""):
    admission = read_json(Path(manifest["root"]) / "pilot_admission.json")
    if arm not in admission["eligible_arms"]:
        raise ValueError("This training arm was not admitted")
    config, training = ARMS[arm], manifest["training"]
    runner = engine(manifest, config["mode"])
    queue = LocalEvaluationQueue(Path(manifest["root"]) / "validator")
    schedule = case_schedule(manifest["splits"]["train_pilot120"], training["pilot_seed"], training["passes"])
    elite, usage, auxiliary_steps, start, requests = {}, {}, 0, 0, []
    episodes, checkpoint = 0, None
    try:
        if len(schedule) * 8 != training["episodes"]:
            raise ValueError("Pilot episode budget differs from fixed case schedule")
        lr = read_json(Path(manifest["root"]) / "contract/result.json")["actor_lr"]
        runner.set_actor_lr(lr)
        if resume_checkpoint:
            state = runner.resume(resume_checkpoint, protocol_sha256=manifest["manifest_sha256"], exploration=config["mode"])
            if state.get("arm") != arm or state.get("diagnostic_only") or state.get("schedule_sha256") != digest_json(schedule):
                raise ValueError("Resume arm/schedule mismatch or diagnostic checkpoint")
            elite, usage, auxiliary_steps, start = [state[k] for k in ("elite_buffer", "elite_usage", "auxiliary_steps", "next_group")]
            requests = state.get("validation_requests", [])
            if not 0 <= start <= len(schedule):
                raise ValueError("Resume group outside schedule")
            episodes, checkpoint = start * 8, resume_checkpoint
            validator_state = read_json(queue.root / "status.json")
            import os
            os.kill(int(validator_state["pid"]), 0)
            if validator_state.get("status") != "running" or (queue.root / "STOP").exists():
                raise ValueError("Explicit resume requires a live shared validator")
            for request_id in requests:
                if not any((queue.root / state_name / f"{request_id}.json").exists()
                           for state_name in ("pending", "running", "results", "failed")):
                    raise ValueError(f"Saved validation publication missing: {request_id}; reconcile explicitly before resume")
        before = frozen_state(runner)
        early_stopped = False
        for index in range(start, len(schedule)):
            while sum(queue.poll(r) is None for r in requests) >= training["queue_pending_per_arm"]:
                heartbeat.update(event="validator_backpressure", pending_requests=requests)
                time.sleep(10)
            evaluations = [queue.poll(r) for r in requests]
            finished = [e["evaluation"]["summary"] for e in evaluations if e is not None]
            if len(finished) >= 2 and all(e["gain_fraction"] <= -.02 for e in finished[-2:]):
                early_stopped = True
                break
            visit, case = schedule[index]
            begin = time.time()
            group = runner.rollout([case] * 8, [trajectory_seed(training["pilot_seed"], case["content_sha256"], visit, k)
                for k in range(8)], retain=True, heartbeat=heartbeat)
            decision_stats = copy.deepcopy(runner.last_decision_stats)
            save_group(output / "rollouts" / f"group_{index + 1:04d}.json", group,
                       {"arm": arm, "case_id": case["path"], "decision_stats": decision_stats})
            c0 = manifest["source_costs"][case["path"]]
            winner = min(group, key=lambda t: t["makespan"])
            probe = {**winner, "states": winner["states"][::max(1, math.ceil(len(winner["states"]) / 16))]}
            probe_before = runner.likelihood_metrics(probe)
            update = runner.update(group, "source", c0, epochs=training["ppo_epochs"],
                                   chunk=training["tbptt_steps"], heartbeat=heartbeat)
            post_kl = max(max(e["post_update"]["kl"], *(r["kl"] for r in e["post_update"]["roles"].values()))
                          for e in update["epochs"])
            if post_kl > training["post_update_hard_kl"]:
                atomic_json(output / "guard_stop.json", {"group": index + 1, "case_id": case["path"],
                    "attempted_training_episodes": (index + 1) * 8, "update": update}, overwrite=False)
                raise ArmGuardStop(f"Post-update role KL exceeded hard bound: {post_kl}")
            if config["elite"] and winner["makespan"] < c0 - 1e-6:
                old = elite.get(case["path"])
                if old is None or winner["makespan"] < old["makespan"]:
                    elite[case["path"]] = compact(winner)
            del group, winner
            if config["elite"] and (index + 1) % training["elite_interval"] == 0:
                key = choose_elite(elite, usage, training["pilot_seed"], index + 1)
                if key is not None:
                    selected = next(c for c in manifest["splits"]["train_pilot120"] if c["path"] == key)
                    replay = runner.rollout([selected], [42], deterministic=True, forced=[elite[key]["actions"]],
                                            retain=True, heartbeat=heartbeat)
                    if abs(replay[0]["makespan"] - elite[key]["makespan"]) > 1e-6 or replay[0]["makespan"] >= manifest["source_costs"][key] - 1e-6:
                        raise RuntimeError("Elite is not a reproducible genuine C0 win")
                    runner.set_actor_lr(training["elite_lr"])
                    update["auxiliary"] = runner.update(replay, "bc", epochs=1, heartbeat=heartbeat)
                    runner.set_actor_lr(lr)
                    auxiliary_kl = max(r["kl"] for e in update["auxiliary"]["epochs"]
                                       for r in e["post_update"]["roles"].values())
                    if auxiliary_kl > training["post_update_hard_kl"]:
                        atomic_json(output / "guard_stop.json", {"group": index + 1, "case_id": key,
                            "attempted_training_episodes": (index + 1) * 8, "update": update}, overwrite=False)
                        raise ArmGuardStop(f"Auxiliary update role KL exceeded hard bound: {auxiliary_kl}")
                    usage[key] = usage.get(key, 0) + 1
                    auxiliary_steps += 1
                    update["auxiliary_case"] = key
                    del replay
            assert_frozen(runner, before)
            update["fixed_input_greedy_probe"] = {"before": probe_before, "after": runner.likelihood_metrics(probe),
                "sample_beats_c0": probe["makespan"] < c0 - 1e-6,
                "scope": "up to16 fixed observations and stored hidden states; greedy agreement with sampled actions, not closed-loop gain"}
            del probe
            episodes = (index + 1) * 8
            record = {"group": index + 1, "training_episodes": episodes, "case_id": case["path"],
                "update": update, "auxiliary_steps": auxiliary_steps, "seconds": time.time() - begin,
                "mean_cost": float(np.mean(update["costs"])), "decision_stats": decision_stats}
            atomic_json(output / "updates" / f"group_{index + 1:04d}.json", record, overwrite=False)
            if (index + 1) % training["save_every_groups"] == 0:
                checkpoint = output / "models" / f"episodes_{episodes:06d}.pt"
                # Include the upcoming request in resume state; publication follows immutable checkpoint.
                due = (index + 1) % training["eval_every_groups"] == 0
                next_requests = requests + ([f"{arm}_e{episodes:06d}{suffix}"] if due else [])
                runner.save(checkpoint, **resume_metadata(manifest, arm=arm, diagnostic_only=False,
                    training_episodes=episodes, next_group=index + 1, seed=training["pilot_seed"],
                    schedule_sha256=digest_json(schedule), elite_buffer=elite, elite_usage=usage,
                    auxiliary_steps=auxiliary_steps, validation_requests=next_requests))
                if due:
                    requests.append(submit(manifest, checkpoint, arm, episodes, suffix))
            heartbeat.update(event="group_completed", completed_groups=index + 1, total_groups=len(schedule),
                             training_episodes=episodes, auxiliary_steps=auxiliary_steps)
            gc.collect()
        while any(queue.poll(r) is None for r in requests):
            heartbeat.update(event="waiting_final_validation", requests=requests)
            time.sleep(10)
        evaluations = [queue.poll(r)["evaluation"] for r in requests]
        passed = (not early_stopped and episodes == training["episodes"] and len(evaluations) == 4
                  and pilot_gate([e["summary"] for e in evaluations])
                  and all(e["summary"]["regression_over_5pct_fraction"] <= .05 for e in evaluations[-2:]))
        atomic_json(output / "result.json", {"arm": arm, "completed": not early_stopped,
            "early_stopped_for_regression": early_stopped, "passed": passed,
            "training_episodes": episodes, "auxiliary_steps": auxiliary_steps,
            "evaluations": evaluations, "requests": requests,
            "endpoint_checkpoint": str(checkpoint) if not early_stopped else None}, overwrite=False)
    except ArmGuardStop as exc:
        runner.save(output / "guard_stop_state.pt", **resume_metadata(manifest, arm=arm, diagnostic_only=True,
            not_resumable_partial_group=True, elite_buffer=elite, elite_usage=usage, auxiliary_steps=auxiliary_steps))
        atomic_json(output / "result.json", {"arm": arm, "completed": False, "passed": False,
            "guard_stopped": True, "reason": str(exc), "training_episodes": episodes,
            "auxiliary_steps": auxiliary_steps, "endpoint_checkpoint": None, "requests": requests}, overwrite=False)
        heartbeat.update(event="prespecified_arm_guard_stop", reason=str(exc))
    except BaseException:
        runner.save(output / "failure_state.pt", **resume_metadata(manifest, arm=arm, diagnostic_only=True,
            not_resumable_partial_group=True, elite_buffer=elite, elite_usage=usage, auxiliary_steps=auxiliary_steps))
        raise
    finally:
        runner.close()


def validator(manifest, output, heartbeat):
    queue = LocalEvaluationQueue(output)
    queue.fail_interrupted()
    runner = None
    try:
        while True:
            claim = queue.claim()
            if claim is None:
                if (output / "STOP").exists():
                    return
                time.sleep(2)
                continue
            path, request = claim
            try:
                verify(manifest, inputs=False)
                if (request["evaluation_protocol"] != EVAL_PROTOCOL or request["tau"] != .3 or request["seed"] != 42
                        or request["cases"] != manifest["splits"]["tune"]
                        or request["cases_sha256"] != digest_json(request["cases"])
                        or request["code_sha256"] != manifest["code"]["sha256"]
                        or request["contract_sha256"] != manifest["contract_sha256"]
                        or request["cache_key"] != queue.identity(request)
                        or digest_file(request["checkpoint"]) != request["checkpoint_sha256"]):
                    raise ValueError("Validator request contract mismatch; Tune-only")
                checkpoint = torch.load(request["checkpoint"], map_location="cpu", weights_only=False)
                if (checkpoint.get("exploration") != request["training_exploration"] or checkpoint.get("diagnostic_only")
                        or checkpoint.get("protocol_sha256") != manifest["manifest_sha256"]
                        or checkpoint.get("source_sha256") != manifest["source"]["sha256"]
                        or checkpoint.get("training_episodes") != request["training_episodes"]):
                    raise ValueError("Checkpoint behavior identity mismatch")
                del checkpoint
                verify_cases(request["cases"])
                heartbeat.update(event="evaluate", request_id=request["request_id"], request_started_unix=time.time())
                cache = queue.root / "cache" / f"{request['cache_key']}.json"
                if cache.exists():
                    cached = read_json(cache)
                    if cached["cache_key"] != request["cache_key"]:
                        raise ValueError("Cache identity mismatch")
                    result = cached["evaluation"]
                else:
                    if runner is None:
                        runner = engine(manifest)
                    runner.load(request["checkpoint"])
                    runner.configure_exploration(None)
                    runner.policy.ac.tau = .3
                    rows = evaluate(runner, request["cases"], heartbeat)
                    result = {"cases": rows, "summary": paired_summary(rows, manifest["source_costs"]),
                              "evaluation_protocol": EVAL_PROTOCOL}
                    atomic_json(cache, {"cache_key": request["cache_key"], "evaluation": result}, overwrite=False)
                queue.finish(path, result)
                heartbeat.update(event="idle", request_id=None, request_started_unix=None)
            except BaseException:
                queue.finish(path, None, traceback.format_exc())
                raise
    finally:
        if runner is not None:
            runner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", choices=("contract", "sample", "fit", "counterfactual", "train", "validator"), required=True)
    parser.add_argument("--mode", choices=tuple(MODES))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--attempt-id")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    verify(manifest)
    root = Path(manifest["root"])
    if args.phase in ("train", "validator"):
        plan = pilot_resource_plan(manifest["resources"], len(read_json(root / "pilot_admission.json")["eligible_arms"]))
        placement = plan["validator"] if args.phase == "validator" else plan["trainers"][args.lane]
        if placement["lane"] != args.lane:
            raise ValueError("Worker lane differs from its frozen placement")
        validate_worker_placement(placement, affinity=os.sched_getaffinity(0),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            cuda_memory_fraction=os.environ.get("HKBZ_STAGE3_CUDA_MEMORY_FRACTION", ".80"))
    sub = {"contract": f"contract/{args.mode}", "sample": f"sampling/lane{args.lane}",
           "fit": f"fit/{args.mode}", "counterfactual": "counterfactual", "train": f"pilot/{args.arm}",
           "validator": "validator"}[args.phase]
    output = args.output.resolve() if args.output else root / sub
    if args.resume_checkpoint:
        import re
        if args.phase != "train" or not args.attempt_id or not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt_id):
            raise ValueError("Explicit resume requires train phase and a new safe attempt ID")
        output = root / sub / "attempts" / args.attempt_id
    if (output / "status.json").exists():
        raise FileExistsError("Worker output already exists; no implicit retry")
    with ProgressHeartbeat(output / "status.json", phase=args.phase, mode=args.mode, arm=args.arm,
                           lane=args.lane, smoke_only=args.smoke,
                           cpuset=sorted(os.sched_getaffinity(0)),
                           cuda_memory_fraction=memory_fraction(os.environ.get("HKBZ_STAGE3_CUDA_MEMORY_FRACTION", ".80"))) as heartbeat:
        if args.phase == "contract":
            contract(manifest, output, args.mode, heartbeat, args.smoke)
        elif args.phase == "sample":
            sample(manifest, output, args.lane, heartbeat)
        elif args.phase == "fit":
            fit(manifest, output, args.mode, heartbeat)
        elif args.phase == "counterfactual":
            counterfactual(manifest, output, heartbeat)
        elif args.phase == "train":
            train(manifest, output, args.arm, heartbeat, args.resume_checkpoint,
                  f"_{args.attempt_id}" if args.attempt_id else "")
        else:
            validator(manifest, output, heartbeat)


if __name__ == "__main__":
    main()
