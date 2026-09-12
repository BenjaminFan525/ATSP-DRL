#!/usr/bin/env python3
"""Bounded workers: contract, hard-IGA, sampling, BC fit, counterfactual and PPO."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from onpolicy.utils.stage3_research import (SOURCE, TRAIN_ROOT, ARMS, EvaluationQueue, Heartbeat,
    read_json, atomic_json, digest_json, digest_file, verify_protocol, trajectory_seed,
    best_of_n, paired_summary, validate_sample_trajectories)


def compact(trajectory):
    return {key: value for key, value in trajectory.items() if key != "states"}


def engine(**kwargs):
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    return ResearchEngine(**kwargs)


def evaluate(runner, cases, heartbeat):
    results = []
    for start in range(0, len(cases), runner.width):
        batch = cases[start:start + runner.width]
        trajectories = runner.rollout(batch, [42] * len(batch), deterministic=True, heartbeat=heartbeat)
        results.extend({key: t[key] for key in ("case_id", "profile", "distribution",
                                              "makespan", "completed", "steps")} for t in trajectories)
        heartbeat.update(completed_cases=len(results), total_cases=len(cases))
    return results


def submit_eval(manifest, checkpoint, request_id, episodes, cases=None):
    cases = manifest["splits"]["tune"] if cases is None else cases
    return EvaluationQueue(Path(manifest["root"]) / "validator").submit({
        "request_id": request_id, "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": digest_file(checkpoint), "training_episodes": int(episodes),
        "cases": cases, "cases_sha256": digest_json(cases), "tau": .3, "seed": 42,
        "contract_sha256": manifest["contract_sha256"], "code_sha256": manifest["code"]["sha256"]})


def contract(manifest, output, heartbeat):
    """Gate 0: C0 identity, greedy reproducibility, policy likelihood, real update."""
    runner = engine(width=8)
    try:
        cases = manifest["splits"]["train_diag32"][:3] + manifest["splits"]["tune"][:3]
        trajectories = runner.rollout(cases, [42] * len(cases), deterministic=True,
                                      retain=True, heartbeat=heartbeat)
        errors = [abs(t["makespan"] - manifest["source_costs"][t["case_id"]]) for t in trajectories]
        if max(errors) > manifest["diagnostic"]["source_reproduction_abs_tolerance"]:
            raise RuntimeError(f"C0 reproduction mismatch in seconds: {errors}")
        # Same recorded joint actions must produce the same makespan from reset.
        replay = runner.rollout(cases[:1], [42], deterministic=True, retain=True,
                                forced=[trajectories[0]["actions"]], heartbeat=heartbeat)[0]
        if abs(replay["makespan"] - trajectories[0]["makespan"]) > 1e-6:
            raise RuntimeError("Action replay changed makespan")
        likelihood = runner.likelihood_metrics(replay)
        if max(v["logp_replay_max_error"] for v in likelihood.values()) > .002:
            raise RuntimeError("Collection / replay likelihood mismatch")
        case = manifest["splits"]["train_diag32"][0]
        del replay, trajectories
        group = runner.rollout([case] * 8,
            [trajectory_seed(20260910, case["content_sha256"], 0, k) for k in range(8)],
            retain=True, heartbeat=heartbeat)
        updates = {}
        import torch
        for arm in ARMS:
            runner.load(SOURCE)
            runner.policy.reset_optimizers()
            # Retained old-policy likelihoods belong to the exact same C0 in every arm.
            updates[arm] = runner.update(group, ARMS[arm], manifest["source_costs"][case["path"]],
                                         epochs=1, heartbeat=heartbeat)
            if arm == "R3":
                # Exercise the off-policy BC code path, but this canary is NOT training evidence.
                winner = min(group, key=lambda t: t["makespan"])
                updates[arm]["bc_path_smoke"] = runner.update([winner], "bc", epochs=1, heartbeat=heartbeat)
            if updates[arm]["peak_reserved_gib"] > 18.0:
                raise RuntimeError(f"{arm} memory exceeds 18 GiB canary gate")
            if not updates[arm]["epochs"][0]["actor_step_applied"]:
                raise RuntimeError(f"{arm} did not apply its canary actor update")
            runner.save(output / f"canary_{arm}.pt", arm=arm, training_episodes=8,
                        diagnostic_only=True, source_sha256=manifest["source"]["sha256"])
        atomic_json(output / "result.json", {"passed": True, "source_cost_errors": errors,
            "likelihood": likelihood, "updates": updates, "diagnostic_only": True})
    finally:
        runner.close()


def hard_iga(manifest, output, heartbeat):
    from onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels import parse_args, run
    cases = manifest["splits"]["train_diag32"]
    names = [r["name"] for r in cases]
    if (output / "cases.json").exists():
        if not manifest.get("resume") or read_json(output / "cases.json") != names:
            raise ValueError("Existing IGA case selection differs or resume was not authorized")
    else:
        atomic_json(output / "cases.json", names, overwrite=False)
    args = parse_args(["--dataset-dir", str(TRAIN_ROOT), "--output-dir", str(output),
        "--case-list-json", str(output / "cases.json"), "--workers", "8",
        "--time-budget-seconds", str(manifest["diagnostic"]["iga_seconds_per_case"]),
        "--cumulative-budget-seconds", str(manifest["diagnostic"]["iga_seconds_per_case"]),
        "--population", "20", "--seed", "20260905", "--max-steps", "4000",
        "--max-plane-agents", "24", "--max-device-num", "80",
        "--device-future-intent-horizon", "3", "--device-future-intent-mode", "bounded_frontier",
        "--device-frontier-max-requests", "4", "--device-request-capacity-per-plane", "5",
        "--resource-release-aware-eta", "--device-lookahead-reservation-mode", "hard"])
    heartbeat.update(event="hard_iga_search", total_cases=len(cases))
    code = run(args)
    if code:
        raise RuntimeError(f"Hard-contract IGA search exited {code}")


def teacher_actions(case, teacher):
    from onpolicy.runner.shared.stage3_research_engine import environment_config
    from onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels import (
        AircraftScheduleEnv, JointGenomeLayout, GA_Policy, BACKENDS, mixed_resource_actions)
    from onpolicy.envs.HKBZ.experiment.eval_common import completion_details
    env = AircraftScheduleEnv(environment_config(case["path"]))
    actions = []
    try:
        layout = JointGenomeLayout.from_env(env)
        jobs, sites, resources = layout.decode(np.asarray(teacher["search"]["chromosome"]))
        _, done, info = env.reset(seed=42)
        while not np.all(done) and len(actions) < 4000:
            chosen = GA_Policy(env, info, jobs, sites)
            resource, _ = mixed_resource_actions(env, BACKENDS, resources, record=False)
            chosen[env.n_plane_agents:] = resource[env.n_plane_agents:]
            actions.append(np.asarray(chosen, dtype=np.int64).tolist())
            _, _, done, info = env.step(chosen)
        if not completion_details(env, len(actions), 4000)["completed"]:
            raise RuntimeError("Hard-IGA gene failed complete action reconstruction")
        if abs(env.total_time - teacher["makespan"]) > .1:
            raise RuntimeError("Hard-IGA label / current environment mismatch")
        return actions
    finally:
        env.close()


def replay_teachers(manifest, output, heartbeat):
    runner = engine(width=1)
    records = []
    try:
        for case in manifest["splits"]["train_diag32"]:
            teacher = read_json(Path(manifest["root"]) / "iga" / "teachers" / f"{case['name']}.json")
            if teacher["case_sha256"] != case["case_sha256"]:
                raise ValueError("Teacher belongs to a different case")
            actions = teacher_actions(case, teacher)
            trajectory = runner.rollout([case], [42], deterministic=True, forced=[actions],
                                        retain=True, heartbeat=heartbeat)[0]
            likelihood = runner.likelihood_metrics(trajectory)
            record = {"case_id": case["path"], "profile": case["profile"],
                "distribution": case["distribution"], "makespan": trajectory["makespan"],
                "teacher_makespan": teacher["makespan"], "completed": trajectory["completed"],
                "cost_error": abs(trajectory["makespan"] - teacher["makespan"]),
                "roles": likelihood, "actions": trajectory["actions"]}
            atomic_json(output / "cases" / f"{case['name']}.json", record, overwrite=False)
            records.append({k: v for k, v in record.items() if k != "actions"})
            heartbeat.update(completed_cases=len(records), total_cases=32)
            del trajectory
        passed = all(r["cost_error"] <= .1 and not any(v["impossible"] for v in r["roles"].values())
                     for r in records)
        atomic_json(output / "result.json", {"passed": passed, "cases": records,
            "summary": paired_summary(records, manifest["source_costs"])})
    finally:
        runner.close()


def sample(manifest, output, heartbeat):
    runner = engine(width=8)
    records = []
    try:
        for case in manifest["splits"]["train_diag32"]:
            final_path = output / "cases" / f"{case['name']}.json"
            partial_path = output / "partial" / f"{case['name']}.json"
            if final_path.exists():
                if not manifest.get("resume"):
                    raise FileExistsError(final_path)
                saved = read_json(final_path)
                validate_sample_trajectories(saved["trajectories"], case, complete=True)
                if saved["source_cost"] != manifest["source_costs"][case["path"]]:
                    raise ValueError("Recovered sample C0 baseline differs")
                records.append({k: v for k, v in saved.items() if k != "trajectories"})
                heartbeat.update(event="sample_case_reused", completed_cases=len(records), total_cases=32)
                continue
            trajectories = []
            if partial_path.exists():
                if not manifest.get("resume"):
                    raise FileExistsError(partial_path)
                saved = read_json(partial_path)
                if saved["case_content_sha256"] != case["content_sha256"]:
                    raise ValueError("Partial sample case content changed")
                trajectories = saved["trajectories"]
                validate_sample_trajectories(trajectories, case)
            for start in range(len(trajectories), 32, 8):
                seeds = [trajectory_seed(20260909, case["content_sha256"], 0, k)
                         for k in range(start, start + 8)]
                trajectories.extend(compact(t) for t in runner.rollout([case] * 8, seeds, heartbeat=heartbeat))
                atomic_json(partial_path, {"case_content_sha256": case["content_sha256"],
                    "code_sha256": manifest["code"]["sha256"], "trajectories": trajectories})
                heartbeat.update(event="sample_group_committed", case_id=case["path"],
                    completed_replicas=len(trajectories), completed_cases=len(records), total_cases=32)
            c0 = manifest["source_costs"][case["path"]]
            record = {"case_id": case["path"], "profile": case["profile"], "source_cost": c0,
                **best_of_n([t["makespan"] for t in trajectories], c0)}
            atomic_json(final_path,
                {**record, "trajectories": trajectories}, overwrite=False)
            records.append(record)
            heartbeat.update(completed_cases=len(records), total_cases=32)
        fraction = float(np.mean([r["prefix_best"]["32"] < r["source_cost"] * .99 for r in records]))
        atomic_json(output / "result.json", {"cases": records,
            "cases_with_1pct_better_sample_fraction": fraction,
            "passed": fraction >= manifest["diagnostic"]["sampling_improvable_case_fraction_min"]})
    finally:
        runner.close()


def resume_check(manifest, output, heartbeat):
    """Check the isolated C0 execution path before reusing old diagnostics."""
    runner = engine(width=8)
    try:
        cases = manifest["splits"]["train_diag32"][:3] + manifest["splits"]["tune"][:3]
        trajectories = runner.rollout(cases, [42] * len(cases), deterministic=True,
                                      retain=True, heartbeat=heartbeat)
        errors = [abs(t["makespan"] - manifest["source_costs"][t["case_id"]]) for t in trajectories]
        replay = runner.rollout(cases[:1], [42], deterministic=True, retain=True,
            forced=[trajectories[0]["actions"]], heartbeat=heartbeat)[0]
        replay_error = abs(replay["makespan"] - trajectories[0]["makespan"])
        likelihood = runner.likelihood_metrics(replay)
        passed = (max(errors) <= .1 and replay_error <= 1e-6
                  and max(v["logp_replay_max_error"] for v in likelihood.values()) <= .002)
        imports = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
            if (name == "onpolicy" or name.startswith("onpolicy.")) and getattr(module, "__file__", None)}
        if any(not Path(path).is_relative_to(ROOT) for path in imports.values()):
            raise RuntimeError("Stage3 imported shared-workspace code outside its source snapshot")
        atomic_json(output / "result.json", {"passed": passed, "source_cost_errors": errors,
            "replay_error": replay_error, "likelihood": likelihood,
            "code_sha256": manifest["code"]["sha256"], "diagnostic_only": True,
            "policy_variant": runner.policy.ac.stage1_baseline, "ac_config": runner.args.ac_config,
            "runtime_imports": imports}, overwrite=False)
        if not passed:
            raise RuntimeError("Resumed C0 execution path did not reproduce the original contract")
    finally:
        runner.close()


def bc_fit(manifest, output, heartbeat, full=False):
    """IGA labels are confined to this diagnostic; checkpoints cannot seed PPO."""
    runner = engine(width=1, freeze_shared=not full)
    for group in runner.policy.actor_optimizer.param_groups:
        group["lr"] = .00001 if group["name"] == "shared_encoder" else .0001
    fit = manifest["splits"]["train_fit16"]
    history = []
    try:
        for visit in range(manifest["diagnostic"]["bc_passes"] + 1):
            nll_sum = decisions = correct = 0
            if visit > 0:
                for case in fit:
                    actions = read_json(Path(manifest["root"]) / "replay" / "cases" / f"{case['name']}.json")["actions"]
                    trajectory = runner.rollout([case], [42], deterministic=True, forced=[actions],
                                                 retain=True, heartbeat=heartbeat)[0]
                    update = runner.update([trajectory], "bc", epochs=1, heartbeat=heartbeat)
                    if update["peak_reserved_gib"] > 18:
                        raise RuntimeError("BC fit exceeded 18 GiB memory gate")
                    del trajectory
            if visit in (0, manifest["diagnostic"]["bc_passes"]):
                # Measure a SINGLE frozen policy across every teacher case. In
                # particular visit zero is pristine C0, not an online BC mixture.
                for case in fit:
                    actions = read_json(Path(manifest["root"]) / "replay" / "cases" / f"{case['name']}.json")["actions"]
                    trajectory = runner.rollout([case], [42], deterministic=True, forced=[actions],
                                                 retain=True, heartbeat=heartbeat)[0]
                    measured = runner.likelihood_metrics(trajectory)
                    nll_sum += sum(r["nll_sum"] for r in measured.values())
                    decisions += sum(r["decisions"] for r in measured.values())
                    correct += sum(r["matches"] for r in measured.values())
                    del trajectory
            record = {"visit": visit, "nll": nll_sum / max(decisions, 1) if decisions else None,
                      "teacher_forced_accuracy": correct / max(decisions, 1) if decisions else None}
            if visit in (0, manifest["diagnostic"]["bc_passes"]):
                runner.save(output / f"diagnostic_only_visit{visit}.pt", diagnostic_only=True,
                            shared_unfrozen=full, visit=visit)
                fit_records = evaluate(runner, fit, heartbeat)
                record["fit"] = paired_summary(fit_records, manifest["source_costs"])
                atomic_json(output / f"fit_visit{visit}.json", fit_records, overwrite=False)
            history.append(record)
            atomic_json(output / "progress.json", {"shared_unfrozen": full, "history": history})
        probe_records = evaluate(runner, manifest["splits"]["train_probe64"], heartbeat)
        initial, final = history[0], history[-1]
        passed = (final["nll"] <= initial["nll"] * .5 and final["fit"]["gain_fraction"] >= .01)
        atomic_json(output / "result.json", {"passed": passed, "shared_unfrozen": full,
            "history": history, "probe": paired_summary(probe_records, manifest["source_costs"]),
            "probe_cases": probe_records, "diagnostic_only": True})
    finally:
        runner.close()


def counterfactual(manifest, output, heartbeat):
    """Replay a real prefix, intervene at one joint decision, finish under C0.

    These are JOINT-action interventions, not mislabeled single-device effects.
    Report changed roles/agents and terminal cost; never substitute a Q estimate.
    """
    from onpolicy.runner.shared.stage3_research_engine import seed_all
    import torch
    runner = engine(width=8)
    records = []
    try:
        for case in manifest["splits"]["train_diag32"][:8]:
            reference = runner.rollout([case], [42], deterministic=True, retain=True, heartbeat=heartbeat)[0]
            candidates = [(i, s) for i, s in enumerate(reference["states"])
                          if np.asarray(s["mask"]).sum() >= 2]
            if not candidates:
                raise RuntimeError("No nontrivial branch state")
            step, state = candidates[len(candidates) // 2]
            actions = []
            logs = []
            for replica in range(8):
                seed_all(trajectory_seed(20260912, case["content_sha256"], step, replica))
                with torch.no_grad():
                    _, action, lp, _, mask = runner.policy.get_actions([state["graph"]],
                        state["hidden"][None], state["active"][None], state["op"][None],
                        state["site"][None], agent_types=state["roles"][None], return_decision_mask=True)
                action = action[0].cpu().numpy()
                actions.append(reference["actions"][:step] + [action.tolist()])
                changed = np.any(action[:, :2] != state["action"][:, :2], axis=1) & state["active"]
                logs.append({"changed_agents": np.flatnonzero(changed).tolist(),
                    "changed_roles": sorted(set(state["roles"][changed].astype(int).tolist())),
                    "joint_logp": float((lp * mask).sum()), "branch_action": action.tolist()})
            branches = runner.rollout([case] * 8, [42] * 8, deterministic=True, forced=actions, heartbeat=heartbeat)
            costs = np.asarray([branch["makespan"] for branch in branches])
            loo = (costs.sum() - costs) / (len(costs) - 1)
            for branch, record in zip(branches, logs):
                role_advantages = {}
                for role in range(3):
                    selected = (state["roles"] == role) & np.asarray(state["mask"], dtype=bool)
                    if selected.any():
                        advantages = -.01 * (branch["makespan"] - state["time"]) - state["value"][selected]
                        role_advantages[str(role)] = {"mean": float(advantages.mean()),
                            "positive_fraction": float(np.mean(advantages > 0)),
                            "mean_value": float(state["value"][selected].mean())}
                record.update(makespan=branch["makespan"], completed=branch["completed"],
                    delta_seconds=branch["makespan"] - reference["makespan"],
                    source_advantage=.01 * (reference["makespan"] - branch["makespan"]),
                    value_baseline_advantages=role_advantages)
            for record, baseline in zip(logs, loo):
                record["leave_one_out_advantage"] = .01 * (float(baseline) - record["makespan"])
            record = {"case_id": case["path"], "step": step, "reference_cost": reference["makespan"],
                "intervention": "joint_action; untouched prefix; deterministic C0 suffix", "branches": logs}
            records.append(record)
            atomic_json(output / "cases" / f"{case['name']}.json", record, overwrite=False)
        informative = sum(any(abs(b["delta_seconds"]) > 1e-6 for b in r["branches"]) for r in records)
        atomic_json(output / "result.json", {"passed": True, "informative_cases": informative, "cases": records,
            "interpretation": "Causal joint-decision diagnostic only; not learned Q or isolated role attribution"})
    finally:
        runner.close()


def train(manifest, output, heartbeat, arm):
    gate = read_json(Path(manifest["root"]) / "diagnostic_gate.json")
    if not gate["passed"]:
        raise ValueError("Training is forbidden before diagnostic admission")
    runner = engine(width=8, freeze_shared=not gate["unfreeze_shared"])
    seed = manifest["training"]["pilot_seed"]
    cases = manifest["splits"]["train_pilot120"]
    elite = {}
    trajectory_count = 0
    auxiliary_steps = 0
    metrics = []
    try:
        for visit in range(2):
            for index, case in enumerate(cases):
                group_number = visit * len(cases) + index + 1
                group = runner.rollout([case] * 8,
                    [trajectory_seed(seed, case["content_sha256"], visit, k) for k in range(8)],
                    retain=True, heartbeat=heartbeat)
                c0 = manifest["source_costs"][case["path"]]
                update = runner.update(group, ARMS[arm], c0, epochs=2, heartbeat=heartbeat)
                trajectory_count += 8
                winner = min(group, key=lambda t: t["makespan"])
                if arm == "R3" and winner["makespan"] < c0 - 1e-6:
                    old = elite.get(case["path"])
                    if old is None or winner["makespan"] < old["makespan"]:
                        elite[case["path"]] = compact(winner)
                del winner, group
                if arm == "R3" and group_number % 4 == 0 and case["path"] in elite:
                    # Fresh replay with CURRENT policy hidden states. Never on-policy PPO ratios here.
                    replay = runner.rollout([case], [42], deterministic=True,
                        forced=[elite[case["path"]]["actions"]], retain=True, heartbeat=heartbeat)
                    if (abs(replay[0]["makespan"] - elite[case["path"]]["makespan"]) > .1
                            or replay[0]["makespan"] >= c0 - 1e-6):
                        raise RuntimeError("Self-imitation elite no longer replays as a genuine C0 win")
                    update["auxiliary_bc"] = runner.update(replay, "bc", epochs=1, heartbeat=heartbeat)
                    auxiliary_steps += 1
                    del replay
                record = {"arm": arm, "group": group_number, "training_episodes": trajectory_count,
                    "case_id": case["path"], "visit": visit, "update": update,
                    "auxiliary_steps": auxiliary_steps, "auxiliary_replayed_trajectories": auxiliary_steps,
                    "unix": time.time()}
                atomic_json(output / "updates" / f"group_{group_number:04d}.json", record, overwrite=False)
                metrics.append({"group": group_number, "mean_cost": float(np.mean(update["costs"]))})
                heartbeat.update(event="group_completed", completed_groups=group_number,
                    total_groups=240, training_episodes=trajectory_count, auxiliary_steps=auxiliary_steps)
                if group_number % 30 == 0:
                    checkpoint = output / "models" / f"episodes_{trajectory_count:06d}.pt"
                    runner.save(checkpoint, arm=arm, training_episodes=trajectory_count, seed=seed,
                        protocol_sha256=digest_json(manifest), source_sha256=manifest["source"]["sha256"],
                        diagnostic_only=False, next_group=group_number, elite_buffer=elite,
                        auxiliary_steps=auxiliary_steps)
                    if group_number % 60 == 0:
                        submit_eval(manifest, checkpoint, f"pilot_{arm}_e{trajectory_count:06d}", trajectory_count)
                # Poll errors without waiting for validator; no stale result can affect an update.
                queue = EvaluationQueue(Path(manifest["root"]) / "validator")
                for name in (queue.root / "failed").glob(f"pilot_{arm}_*.json"):
                    raise RuntimeError(f"Shared validator reported failure: {name}")
                gc.collect()
        atomic_json(output / "result.json", {"arm": arm, "completed": True,
            "training_episodes": trajectory_count, "auxiliary_steps": auxiliary_steps,
            "auxiliary_replayed_trajectories": auxiliary_steps,
            "metrics": metrics, "endpoint_checkpoint": str(checkpoint)})
    finally:
        runner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", required=True,
                        choices=("contract", "resume_check", "iga", "replay", "sample", "bc_heads", "bc_full", "counterfactual", "train"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    verify_protocol(manifest)
    output = Path(manifest["root"]) / (f"pilot_{args.arm}" if args.phase == "train" else args.phase)
    if args.phase == "resume_check":
        output = Path(manifest["resume"]["attempt_dir"]) / "resume_check"
    output.mkdir(parents=True, exist_ok=True)
    with Heartbeat(output / "status.json", phase=args.phase, arm=args.arm) as heartbeat:
        if args.phase == "train":
            if not args.arm:
                raise ValueError("Train requires arm")
            train(manifest, output, heartbeat, args.arm)
        elif args.phase in ("bc_heads", "bc_full"):
            bc_fit(manifest, output, heartbeat, full=args.phase == "bc_full")
        else:
            {"contract": contract, "resume_check": resume_check, "iga": hard_iga, "replay": replay_teachers,
             "sample": sample, "counterfactual": counterfactual}[args.phase](manifest, output, heartbeat)


if __name__ == "__main__":
    main()
