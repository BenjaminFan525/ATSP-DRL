#!/usr/bin/env python3
"""Bounded workers; diagnostic checkpoints can never initialize PPO."""
from __future__ import annotations
import argparse
import copy
import gc
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from onpolicy.runner.shared.stage3_research_engine import ResearchEngine, seed_all
from onpolicy.runner.shared.stage3_representation_engine import RepresentationEngine
from onpolicy.utils.stage3_research import (atomic_json, read_json, digest_file, digest_json,
    trajectory_seed, paired_summary, verify_cases)
from onpolicy.utils.stage3_local_exploration import MODES, EVAL_PROTOCOL
from onpolicy.utils.stage3_representation import ARMS, schedule, RepresentationQueue
from onpolicy.utils.stage3_hot_update import execution_identities, resume_payload_allowed
from onpolicy.utils.stage3_sampling_audit import action_digest
from onpolicy.scripts.train.run_stage3_sampling_audit import ProgressHeartbeat, save_group, trajectory_record
from onpolicy.scripts.train.stage3_local_worker import (load_group, frozen_state, assert_frozen,
    evaluate, nested_close)
from onpolicy.scripts.train.run_stage3_representation import verify


def engine(manifest, encoder, width=8, checkpoint=None):
    return RepresentationEngine(encoder, checkpoint=checkpoint, source=manifest["source"]["path"],
        width=width, exploration="J", diagnostics=True,
        performance=manifest.get("hot_update", {}).get("options"),
        cuda_memory_fraction=float(os.environ.get("HKBZ_STAGE3_CUDA_MEMORY_FRACTION", ".65")))


def metadata(manifest, **extra):
    return {"protocol_sha256": manifest["manifest_sha256"], "source_sha256": manifest["source"]["sha256"],
            "execution_code_sha256": manifest["code"]["sha256"],
            "execution_performance": manifest.get("hot_update", {}).get("options"),
            "next_group": 0, "elite_buffer": {}, "elite_usage": {}, "auxiliary_steps": 0, **extra}


def submit(manifest, checkpoint, arm, episodes, encoder):
    queue = RepresentationQueue(Path(manifest["root"])/"validator")
    cases = manifest["splits"]["tune"]
    return queue.submit({"request_id": f"{arm}_e{episodes:06d}", "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest_file(checkpoint), "cases": cases, "cases_sha256": digest_json(cases),
        "contract_sha256": manifest["contract_sha256"], "code_sha256": manifest["code"]["sha256"],
        "protocol_sha256": manifest["manifest_sha256"], "tau": .3, "seed": 42,
        "evaluation_protocol": EVAL_PROTOCOL, "training_exploration": MODES["J"],
        "training_episodes": episodes, "representation": encoder, "arm": arm})


def max_role_kl(update):
    return max(max(e["post_update"]["kl"], *(v["kl"] for v in e["post_update"]["roles"].values()))
               for e in update["epochs"])


def contract(manifest, output, encoder, hb):
    seed_all(manifest["training"]["seed"])
    runner = engine(manifest, encoder)
    try:
        report = runner.representation_report
        if encoder == "E3" and report["e3_vs_e2_trainable_relative_error"] > .05:
            raise RuntimeError("Capacity control differs from E2 by more than 5%")
        initial = output/"initial_c0.pt"
        runner.save(initial, **metadata(manifest, training_episodes=0, diagnostic_only=False,
                    initialization_only=True, source_migration=True))
        if encoder == "E0":
            submit(manifest, initial, "C0", 0, encoder)
        case = manifest["splits"]["train_fit16"][0]
        greedy = runner.rollout([case], [42], deterministic=True, heartbeat=hb)[0]
        if abs(greedy["makespan"] - manifest["source_costs"][case["path"]]) > 1e-6:
            raise RuntimeError("Migrated C0 greedy cost changed")
        save_group(output/"greedy.json", [greedy], {"source_reproduced": True})
        batch = schedule(manifest["splits"]["train_pilot120"], manifest["training"]["seed"], "T1")[0]
        begin = time.time()
        trajectories = runner.rollout(batch["cases"], batch["seeds"], retain=True, heartbeat=hb)
        rollout_seconds = time.time()-begin
        save_group(output/"multi_case_canary.json", trajectories, {"encoder": encoder, "batch": "4x2"})
        replay = runner.replay_metrics(trajectories, hb)
        if replay["max_logp_error"] > .002:
            raise RuntimeError(f"Full-history replay mismatch: {replay}")
        start = output/"resume_start.pt"
        runner.save(start, **metadata(manifest, diagnostic_only=True))
        hb.update(event="encoder_role_and_cross_case_gradients")
        components = gradient_components(runner, trajectories, manifest["source_costs"], hb)
        atomic_json(output/"gradient_components.json", components, overwrite=False)
        rng_expected = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, device=runner.device))
        frozen = frozen_state(runner)
        begin = time.time()
        update = runner.update(trajectories, "source", manifest["source_costs"], epochs=2, heartbeat=hb)
        update_seconds = time.time()-begin
        assert_frozen(runner, frozen)
        expected = {k:v.detach().cpu().clone() for k,v in runner.policy.ac.state_dict().items()}
        expected_actor = copy.deepcopy(runner.policy.actor_optimizer.state_dict())
        expected_critic = copy.deepcopy(runner.policy.critic_optimizer.state_dict())
        runner.resume(start, protocol_sha256=manifest["manifest_sha256"], exploration="J")
        rng_actual = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, device=runner.device))
        if not nested_close(rng_expected, rng_actual):
            raise RuntimeError("RNG restoration mismatch")
        runner.update(trajectories, "source", manifest["source_costs"], epochs=2, heartbeat=hb)
        error = max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in runner.policy.ac.state_dict().items())
        if error > 2e-6 or not nested_close(expected_actor, runner.policy.actor_optimizer.state_dict()) or not nested_close(
                expected_critic, runner.policy.critic_optimizer.state_dict()):
            raise RuntimeError(f"Next-update resume mismatch: {error}")
        passed = (max_role_kl(update) <= .04 and update["epochs"][0]["actor_step_applied"]
                  and update["epochs"][0]["actor_gradient_norm"] > 0)
        atomic_json(output/"result.json", {"passed": passed, "representation": report,
            "greedy_source_exact": True, "replay": replay, "update": update,
            "resume_model_max_abs_error": error, "resume_optimizer_close": True, "resume_rng_close": True,
            "frozen_unchanged": True, "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
            "gradient_components": components,
            "scope": "real 4-case x2 on-policy canary, not evidence of improvement"}, overwrite=False)
    finally:
        runner.close()


def teachers(manifest):
    selected = {}
    for case in manifest["splits"]["train_fit16"]:
        reference = manifest["teachers"][case["path"]]
        if digest_file(reference["path"]) != reference["sha256"]:
            raise ValueError("Frozen teacher evidence changed")
        result = read_json(reference["path"])
        winner = min(result["trajectories"], key=lambda t:t["makespan"])
        if winner["makespan"] >= manifest["source_costs"][case["path"]] - 1e-6:
            continue
        for group in result["groups"]:
            rows, _ = load_group(group)
            row = next((r for r in rows if r["seed"] == winner["seed"]), None)
            if row is not None:
                selected[case["path"]] = row
                break
        if case["path"] not in selected:
            raise ValueError("Winning teacher trace not found")
    return selected


def teacher_replay(runner, case, teacher, hb):
    data = runner.rollout([case], [42], deterministic=True, forced=[teacher["actions"]], retain=True, heartbeat=hb)
    if abs(data[0]["makespan"] - teacher["makespan"]) > 1e-6:
        raise RuntimeError("Teacher no longer reproduces under the hard contract")
    return data


def fit(manifest, output, encoder, hb):
    teacher = teachers(manifest)
    cases = manifest["splits"]["train_fit16"]
    chosen = [c for c in cases if c["path"] in teacher][:manifest["fit"]["single_case_count"]]
    single = []
    for i, case in enumerate(chosen):
        # Each single-case diagnostic starts independently from original C0.
        seed_all(manifest["training"]["seed"])
        runner = engine(manifest, encoder, width=1)
        try:
            runner.set_actor_lr(manifest["fit"]["lr"])
            initial = runner.replay_metrics(teacher_replay(runner, case, teacher[case["path"]], hb), hb)
            for step in range(manifest["fit"]["single_case_updates"]):
                data = teacher_replay(runner, case, teacher[case["path"]], hb)
                update = runner.update(data, "bc", epochs=1, heartbeat=hb)
                atomic_json(output/f"single{i}/updates/{step:03d}.json", update, overwrite=False)
                del data
            final = runner.replay_metrics(teacher_replay(runner, case, teacher[case["path"]], hb), hb)
            row = evaluate(runner, [case], hb)[0]
            c0, tc = manifest["source_costs"][case["path"]], teacher[case["path"]]["makespan"]
            single.append({"case": case, "initial_replay": initial, "final_replay": final, "greedy": row,
                           "recovered_fraction": (c0-row["makespan"])/(c0-tc)})
            runner.save(output/f"single{i}/diagnostic_fit.pt", **metadata(manifest, diagnostic_only=True,
                        forbidden_as_rl_initialization=True))
        finally:
            runner.close()
    seed_all(manifest["training"]["seed"])
    runner = engine(manifest, encoder)
    try:
        runner.set_actor_lr(manifest["fit"]["lr"])
        before = frozen_state(runner)
        initial, final = {}, {}
        for case in cases:
            if case["path"] in teacher:
                initial[case["path"]] = runner.replay_metrics(teacher_replay(runner, case, teacher[case["path"]], hb), hb)
        for visit in range(manifest["fit"]["fit16_passes"]):
            ordered = sorted(cases, key=lambda c:digest_json([manifest["training"]["seed"], "fit", visit, c["content_sha256"]]))
            for case in ordered:
                if case["path"] not in teacher:
                    continue
                data = teacher_replay(runner, case, teacher[case["path"]], hb)
                update = runner.update(data, "bc", epochs=1, heartbeat=hb)
                assert_frozen(runner, before)
                atomic_json(output/f"fit16/updates/pass{visit}_{case['name']}.json", update, overwrite=False)
                del data; gc.collect()
            hb.update(event="fit_pass_complete", completed_passes=visit+1)
        for case in cases:
            if case["path"] in teacher:
                final[case["path"]] = runner.replay_metrics(teacher_replay(runner, case, teacher[case["path"]], hb), hb)
        rows = evaluate(runner, cases, hb)
        probe = evaluate(runner, manifest["splits"]["train_probe64"], hb)
        summary = paired_summary(rows, manifest["source_costs"])
        potential_seconds = sum(manifest["source_costs"][c["path"]] - teacher.get(c["path"],
            {"makespan": manifest["source_costs"][c["path"]]})["makespan"] for c in cases)
        recovered = -summary["delta_seconds"]*len(cases)/potential_seconds if potential_seconds > 0 else 0.
        runner.save(output/"fit16/diagnostic_fit.pt", **metadata(manifest, diagnostic_only=True,
                    forbidden_as_rl_initialization=True))
        atomic_json(output/"result.json", {"completed": True, "encoder": encoder, "single_case": single,
            "initial_replay": initial, "final_replay": final, "teacher_cases": len(teacher),
            "fit": {"cases": rows, "summary": summary}, "probe": {"cases": probe, "summary": paired_summary(probe, manifest["source_costs"])},
            "recovered_fraction": recovered, "recovery_target_met": recovered >= .5,
            "diagnostic_only": True, "forbidden_as_rl_initialization": True, "blocks_pure_ppo": False}, overwrite=False)
    finally:
        runner.close()


class GradientCaptured(Exception):
    pass


def gradient_components(runner, group, source, hb):
    baseline = lambda t: source[t["case_id"]] if isinstance(source, dict) else source
    components = {"all": group, "positive": [t for t in group if t["makespan"] < baseline(t)],
        "negative": [t for t in group if t["makespan"] > baseline(t)],
        "winner": [max(group, key=lambda t:baseline(t)-t["makespan"])],
        "plane": group, "device": group, "transporter": group}
    case_keys = sorted({t["case_id"] for t in group})
    if len(case_keys) > 1:
        components.update({f"case{i}": [t for t in group if t["case_id"] == case]
                           for i,case in enumerate(case_keys)})
    gradients, metrics = {}, {}
    for label, rows in components.items():
        role = {"plane":0, "device":1, "transporter":2}.get(label)
        if not rows or (role == 0 and 0 not in runner.exploration["roles"]):
            metrics[label] = {"available": False}; continue
        captured = {}
        def callback(epoch, ac, stats):
            for pg in runner.policy.actor_optimizer.param_groups:
                captured[pg["name"]] = torch.cat([(p.grad.detach().cpu().flatten() if p.grad is not None
                    else torch.zeros(p.numel())) for p in pg["params"] if p.requires_grad]) if any(
                    p.requires_grad for p in pg["params"]) else torch.zeros(0)
            raise GradientCaptured()
        try:
            runner.update(rows, "source", source, epochs=1, heartbeat=hb,
                          gradient_callback=callback, diagnostic_role=role)
        except GradientCaptured:
            pass
        if not captured:
            raise RuntimeError("Gradient diagnostic unexpectedly applied an optimizer step")
        # Retain normalized subgroup objectives. Unlike a single-case group,
        # arbitrary positive/negative subsets of T1 need not retain equal case
        # counts; their gradients are directional probes, not an exact sum.
        gradients[label] = captured
        metrics[label] = {"available": True, "trajectories": len(rows), "group_norms": {k:float(v.norm()) for k,v in captured.items()}}
    cosines = {}
    pairs = [("positive","negative"),("winner","all"),("plane","device"),("plane","transporter"),("device","transporter")]
    if len(case_keys) > 1:
        pairs += [(f"case{i}",f"case{j}") for i in range(len(case_keys)) for j in range(i+1,len(case_keys))]
    for a,b in pairs:
        if a not in gradients or b not in gradients:
            continue
        cosines[f"{a}__{b}"] = {}
        for key in gradients[a]:
            x,y = gradients[a][key], gradients[b][key]
            denominator = float(x.norm()*y.norm())
            cosines[f"{a}__{b}"][key] = float(x.dot(y))/denominator if denominator else None
    runner.policy.actor_optimizer.zero_grad(set_to_none=True)
    runner.policy.critic_optimizer.zero_grad(set_to_none=True)
    return {"components": metrics, "cosines": cosines, "cases":case_keys,
            "scope": "pre-clip normalized subgroup objectives; role heads disjoint; a frozen graph gives null cosine, not evidence of no conflict"}


def mechanism(manifest, output, key, hb):
    config = manifest["mechanisms"][key]
    mode = config["mode"]
    runner = ResearchEngine(config["checkpoint"], width=8, exploration=mode, diagnostics=True,
        cuda_memory_fraction=float(os.environ.get("HKBZ_STAGE3_CUDA_MEMORY_FRACTION", ".65")))
    try:
        if config["group"] != 1:
            runner.resume(config["checkpoint"], protocol_sha256=manifest["prior_manifest_sha256"], exploration=mode)
        archived, _ = load_group(config["archive"])
        case = next(c for c in manifest["splits"]["train_pilot120"] if c["path"] == archived[0]["case_id"])
        source = manifest["source_costs"][case["path"]]
        # Regenerate ON POLICY with the original seeds. Never relabel a forced
        # teacher replay as on-policy data to bypass the replay admission guard.
        group = runner.rollout([case]*8, [t["seed"] for t in archived], retain=True, heartbeat=hb)
        if any(action_digest(a["actions"]) != action_digest(b["actions"]) or abs(a["makespan"]-b["makespan"]) > 1e-6
               for a,b in zip(group, archived)):
            raise RuntimeError("Historical behavior regeneration differs from archived actions")
        start = output/"fixed_batch_start.pt"
        runner.save(start, **metadata(manifest, diagnostic_only=True))
        full = runner.replay_metrics(group, hb)
        gradients = gradient_components(runner, group, source, hb)
        atomic_json(output/"gradient_components.json", gradients, overwrite=False)
        results = {}
        for intervention in manifest["diagnostic"]["fixed_batch_interventions"]:
            runner.resume(start, protocol_sha256=manifest["manifest_sha256"], exploration=mode)
            multiplier = {"lr2":2., "lr4":4.}.get(intervention, 1.)
            runner.set_actor_lr(5e-6*multiplier)
            winner = min(group, key=lambda t:t["makespan"])
            probe = {**winner, "states": winner["states"][::max(1,len(winner["states"])//16)]}
            old_hidden_before = runner.likelihood_metrics(probe)
            update = runner.update(group, "source", source, epochs=2,
                chunk=32 if intervention == "tbptt32" else 8,
                clip_mode="per_group" if intervention == "per_group_clip" else "global", heartbeat=hb)
            old_hidden_after = runner.likelihood_metrics(probe)
            closed_loop = evaluate(runner, [case], hb)[0]
            results[intervention] = {"update": update, "stored_hidden_probe_before": old_hidden_before,
                "stored_hidden_probe_after": old_hidden_after, "current_history_after": runner.replay_metrics(group, hb),
                "closed_loop_greedy": closed_loop, "source_cost": source}
            atomic_json(output/f"interventions/{intervention}.json", results[intervention], overwrite=False)
        atomic_json(output/"result.json", {"completed": True, "historical_trace_exact": True,
            "before_full_history": full, "gradient_components": gradients, "interventions": results,
            "diagnostic_only": True, "training_recipe_changed": False}, overwrite=False)
    finally:
        runner.close()


def train(manifest, output, arm, until, resume_checkpoint, hb):
    if until not in (960,1440,1920):
        raise ValueError("Only preregistered screening/pilot boundaries are allowed")
    admission = read_json(Path(manifest["root"])/"training_admission.json")
    if not admission["passed"] or arm not in admission["eligible_arms"]:
        raise ValueError("No training admission")
    config, training = ARMS[arm], manifest["training"]
    if until > 960:
        if arm not in read_json(Path(manifest["root"])/"screen_admission.json")["selected"]:
            raise ValueError("Arm not selected for single-seed extension")
        if until == 1920 and arm not in read_json(Path(manifest["root"])/"pilot_admission_1440.json")["eligible_next_chunk"]:
            raise ValueError("1440 gate failed; 1920 cannot be entered")
    seed_all(training["seed"])
    runner = engine(manifest, config["encoder"])
    queue = RepresentationQueue(Path(manifest["root"])/"validator")
    plan = schedule(manifest["splits"]["train_pilot120"], training["seed"], config["batch"])
    cursor, episodes, requests = 0, 0, []
    try:
        runner.set_actor_lr(training["actor_lr"])
        if resume_checkpoint:
            payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            protocol = resume_payload_allowed(payload, manifest, arm, until, plan)
            payload = runner.resume(resume_checkpoint, protocol_sha256=protocol, exploration="J")
            cursor, episodes, requests = payload["next_group"], payload["training_episodes"], payload["validation_requests"]
            if cursor*8 != episodes:
                raise ValueError("Checkpoint case cursor inconsistent")
            atomic_json(output/"resume_receipt.json", {"checkpoint":str(resume_checkpoint),
                "checkpoint_sha256":digest_file(resume_checkpoint), "training_episodes":episodes,
                "next_group":cursor, "parent_protocol_sha256":protocol,
                "active_protocol_sha256":manifest["manifest_sha256"],
                "model_optimizer_normalizers_rng_restored":True,
                "performance":manifest.get("hot_update", {}).get("options")}, overwrite=False)
            hb.update(event="resumed", training_episodes=episodes, completed_groups=cursor, target_episodes=until)
        elif until != 960:
            raise ValueError("Extension requires a complete checkpoint")
        frozen = frozen_state(runner)
        guard = None
        for index in range(cursor, until//8):
            while queue.pending_count(arm) >= 2 or queue.pending_count() >= 16:
                # No evaluation point is dropped: explicit bounded backpressure.
                for request in requests:
                    queue.poll(request)
                hb.update(event="validator_backpressure", pending=queue.pending_count())
                time.sleep(10)
            begin = time.time()
            batch = plan[index]
            group = runner.rollout(batch["cases"], batch["seeds"], retain=True, heartbeat=hb)
            save_group(output/f"rollouts/group_{index+1:04d}.json", group,
                       {"arm":arm, "group":index+1, "macroblock":batch["macroblock"], "policy_updates":runner.policy_updates})
            update = runner.update(group, "source", manifest["source_costs"], epochs=training["ppo_epochs"],
                                   chunk=training["tbptt_steps"], clip_mode=training["clip_mode"], heartbeat=hb)
            assert_frozen(runner, frozen)
            if max_role_kl(update) > training["post_update_hard_kl"]:
                guard = {"reason":"post_update_role_kl", "attempted_episodes":(index+1)*8, "update":update}
                atomic_json(output/"guard_stop.json", guard, overwrite=False)
                runner.save(output/"guard_stop_state.pt", **metadata(manifest, diagnostic_only=True,
                            not_resumable_partial_group=True, arm=arm))
                break
            episodes = (index+1)*8
            atomic_json(output/f"updates/group_{index+1:04d}.json", {"arm":arm, "group":index+1,
                "training_episodes":episodes, "seconds":time.time()-begin, "update":update,
                "mean_cost":float(np.mean(update["costs"])), "macroblock":batch["macroblock"]}, overwrite=False)
            del group; gc.collect()
            if episodes % 240 == 0:
                checkpoint = output/f"models/episodes_{episodes:06d}.pt"
                due = episodes % 480 == 0
                next_requests = requests + ([f"{arm}_e{episodes:06d}"] if due else [])
                runner.save(checkpoint, **metadata(manifest, arm=arm, diagnostic_only=False, training_episodes=episodes,
                    next_group=index+1, seed=training["seed"], schedule_sha256=digest_json(plan),
                    validation_requests=next_requests, batch_composition=config["batch"], optimizer_recipe="O0"))
                if due:
                    requests.append(submit(manifest, checkpoint, arm, episodes, config["encoder"]))
            hb.update(event="group_completed", completed_groups=index+1, training_episodes=episodes, target_episodes=until)
        atomic_json(output/"result.json", {"completed": guard is None and episodes == until, "arm":arm,
            "training_episodes":episodes, "target_episodes":until, "guard":guard, "requests":requests,
            "validation_pending": [r for r in requests if queue.poll(r) is None],
            "endpoint_checkpoint":str(output/f"models/episodes_{episodes:06d}.pt") if guard is None else None}, overwrite=False)
    except BaseException:
        runner.save(output/"failure_state.pt", **metadata(manifest, arm=arm, diagnostic_only=True,
            not_resumable_partial_group=True, completed_training_episodes=episodes))
        raise
    finally:
        runner.close()


def validator(manifest, output, hb, *, verify_fn=None, engine_fn=None,
              arms=None, execution_identity_fn=None):
    verify_fn = verify if verify_fn is None else verify_fn
    engine_fn = engine if engine_fn is None else engine_fn
    arms = ARMS if arms is None else arms
    execution_identity_fn = execution_identities if execution_identity_fn is None else execution_identity_fn
    queue = RepresentationQueue(Path(manifest["root"])/"validator")
    runner, current = None, None
    try:
        while True:
            claim = queue.claim()
            if claim is None:
                if (queue.root/"STOP").exists():
                    return
                time.sleep(2); continue
            path, request = claim
            try:
                verify_fn(manifest, inputs=False)
                if (request["cases"] != manifest["splits"]["tune"] or request["evaluation_protocol"] != EVAL_PROTOCOL
                        or request["tau"] != .3 or request["seed"] != 42
                        or request["training_exploration"] != MODES["J"]
                        or execution_identity_fn(manifest).get(request["protocol_sha256"]) != request["code_sha256"]
                        or request["contract_sha256"] != manifest["contract_sha256"]
                        or request["cache_key"] != queue.identity(request)
                        or digest_json(request["cases"]) != request["cases_sha256"]
                        or digest_file(request["checkpoint"]) != request["checkpoint_sha256"]):
                    raise ValueError("Shared validator request identity/contract mismatch")
                payload = torch.load(request["checkpoint"], map_location="cpu", weights_only=False)
                if (payload.get("diagnostic_only") or payload.get("forbidden_as_rl_initialization")
                        or payload.get("representation") != request["representation"]
                        or payload.get("exploration") != request["training_exploration"]
                        or payload.get("training_episodes") != request["training_episodes"]
                        or payload.get("protocol_sha256") != request["protocol_sha256"]
                        or payload.get("execution_code_sha256", request["code_sha256"]) != request["code_sha256"]
                        or payload.get("source_sha256") != manifest["source"]["sha256"]):
                    raise ValueError("Checkpoint is not an admissible training/C0 artifact")
                if request["arm"] == "C0":
                    if request["training_episodes"] != 0 or not payload.get("initialization_only"):
                        raise ValueError("Invalid C0 request")
                elif (request["arm"] not in arms or payload.get("arm") != request["arm"]
                      or arms[request["arm"]]["encoder"] != request["representation"]):
                    raise ValueError("Validation arm/representation identity mismatch")
                if request["request_id"] != f"{request['arm']}_e{request['training_episodes']:06d}":
                    raise ValueError("Validation request ID is not canonical")
                del payload
                verify_cases(request["cases"])
                hb.update(event="evaluate", request_id=request["request_id"], request_started_unix=time.time())
                cache = queue.root/"cache"/f"{request['cache_key']}.json"
                if cache.exists():
                    cached = read_json(cache)
                    if cached["cache_key"] != request["cache_key"]:
                        raise ValueError("Cached evaluation identity mismatch")
                    result = cached["evaluation"]
                else:
                    if current != request["representation"]:
                        if runner is not None:
                            runner.close(); del runner; gc.collect(); torch.cuda.empty_cache()
                        runner = engine_fn(manifest, request["representation"], width=manifest["resources"]["validator_width"])
                        current = request["representation"]
                    runner.load(request["checkpoint"])
                    runner.configure_exploration(None)
                    runner.policy.ac.tau = .3
                    rows = evaluate(runner, request["cases"], hb)
                    result = {"cases":rows, "summary":paired_summary(rows, manifest["source_costs"]),
                              "evaluation_protocol":EVAL_PROTOCOL}
                    atomic_json(cache, {"cache_key":request["cache_key"], "evaluation":result}, overwrite=False)
                queue.finish(path, result)
                hb.update(event="idle", request_id=None, request_started_unix=None)
            except BaseException:
                queue.finish(path, None, traceback.format_exc())
                raise
    finally:
        if runner is not None:
            runner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", required=True, choices=("contract","mechanism","fit","train","validator"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoder", choices=("E0","E1","E2","E3"))
    parser.add_argument("--mechanism")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--until", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}; preserve diagnostic state")
    signal.signal(signal.SIGTERM, stop)
    manifest = read_json(args.manifest); verify(manifest)
    if (args.output/"status.json").exists():
        raise FileExistsError("Worker output already exists")
    with ProgressHeartbeat(args.output/"status.json", phase=args.phase, encoder=args.encoder, arm=args.arm,
        cpuset=sorted(os.sched_getaffinity(0)), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_memory_fraction=os.environ.get("HKBZ_STAGE3_CUDA_MEMORY_FRACTION")) as hb:
        if args.phase == "contract": contract(manifest,args.output,args.encoder,hb)
        elif args.phase == "mechanism": mechanism(manifest,args.output,args.mechanism,hb)
        elif args.phase == "fit": fit(manifest,args.output,args.encoder,hb)
        elif args.phase == "train": train(manifest,args.output,args.arm,args.until,args.resume_checkpoint,hb)
        else: validator(manifest,args.output,hb)


if __name__ == "__main__":
    main()
