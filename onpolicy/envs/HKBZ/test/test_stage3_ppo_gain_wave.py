"""Contracts for the strict Stage3 PPO-gain two-wave experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.scripts.train.prepare_stage3_joint_finetune import build_command
from onpolicy.scripts.train.run_stage3_joint_manifest import (
    _validate_pure_rl_command,
)
from onpolicy.scripts.train.stage3_ppo_gain_autopilot import analyze_phase


ROOT = Path(__file__).resolve().parents[4]


def _value(command, flag):
    return command[command.index(flag) + 1]


def _command(method, profile, *, base="P3"):
    return build_command(SimpleNamespace(
        profile=profile,
        python=ROOT.parent / "conda/envs/maia-hkbz-cu124-20260903/bin/python3.11",
        experiment_name=f"contract_{profile}_{method}",
        source_checkpoint=Path("/tmp/stage2_source.pt"),
        seed=1,
        method=method,
        base_credit_method=base,
        max_graphs_per_forward=1000,
    ))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation(offset):
    cases = [{
        "case_key": f"tune/case_{index:04d}",
        "case_sha256": f"sha-{index:04d}",
        "makespan": 10000.0 + index + offset,
        "resource_wait_seconds": 1000.0 + index,
        "resource_avoidable_critical_lateness_seconds": 100.0 + index,
    } for index in range(60)]
    raw = float(np.mean([record["makespan"] for record in cases]))
    return {
        "summary": {
            "eval_case_count": 60,
            "eval_valid": 1,
            "eval_completion_rate": 1.0,
            "eval_cycle_count": 0,
            "eval_timeout_count": 0,
            "eval_raw_makespan": raw,
            "eval_selection_score": raw,
            "eval_distribution_ood_stress_makespan": raw,
        },
        "cases": cases,
    }


class Stage3PPOGainWaveTest(unittest.TestCase):
    def test_wave1_isolated_credit_arms_and_common_calibration(self):
        commands = {
            method: _command(method, "ppo_gain_wave1")
            for method in ("P0", "P1", "P2", "P3")
        }
        for command in commands.values():
            _validate_pure_rl_command(command, allow_staged_shared=True)
            self.assertEqual(_value(command, "--max_graphs_per_forward"), "1000")
            self.assertEqual(_value(command, "--num_episodes"), "5")
            self.assertEqual(_value(command, "--actor_warmup_shards"), "12")
            self.assertEqual(_value(command, "--train_sampling_size"), "240")
            self.assertEqual(
                _value(command, "--train_sampling_pool_size"), "960"
            )
            self.assertEqual(
                _value(command, "--shared_actor_lr_scale_schedule"),
                "0.0,0.0,0.01,0.025,0.05",
            )
            self.assertEqual(_value(command, "--gnn_freeze_epochs"), "2")
            self.assertEqual(
                _value(command, "--device_future_intent_horizon"), "3"
            )
            self.assertEqual(
                _value(command, "--device_frontier_max_requests"), "4"
            )
            self.assertEqual(
                _value(command, "--stage3_handoff_mode"), "ppo_gain_wave"
            )

        self.assertNotIn("--role_event_returns", commands["P0"])
        self.assertNotIn("--role_event_returns", commands["P1"])
        self.assertEqual(
            _value(commands["P0"], "--resource_wait_constraint_target"),
            "21000.0",
        )
        self.assertEqual(
            _value(commands["P1"], "--resource_wait_constraint_target"),
            "0.0",
        )
        self.assertIn("--role_event_returns", commands["P2"])
        self.assertIn("--role_valuenorm", commands["P2"])
        for role in ("plane", "device", "transporter"):
            self.assertEqual(
                _value(commands["P2"], f"--{role}_role_gae_lambda"),
                "0.9",
            )
        self.assertEqual(
            _value(commands["P3"], "--plane_role_gae_lambda"), "0.95"
        )
        self.assertEqual(
            _value(commands["P3"], "--device_role_gae_lambda"), "0.8"
        )
        self.assertEqual(
            _value(commands["P3"], "--transporter_role_gae_lambda"), "0.9"
        )

    def test_wave2_is_tail_and_pcgrad_two_by_two(self):
        commands = {
            method: _command(method, "ppo_gain_wave2", base="P3")
            for method in ("Q0", "Q1", "Q2", "Q3")
        }
        for method, command in commands.items():
            _validate_pure_rl_command(command, allow_staged_shared=True)
            self.assertEqual(
                "--shared_encoder_pcgrad" in command,
                method in {"Q2", "Q3"},
            )
            self.assertEqual(
                _value(command, "--tail_policy_start_fraction"),
                "0.75" if method in {"Q1", "Q3"} else "1.0",
            )
            self.assertEqual(
                _value(command, "--tail_policy_weight"),
                "2.0" if method in {"Q1", "Q3"} else "1.0",
            )
        for method in ("Q2", "Q3"):
            self.assertEqual(
                _value(commands[method], "--shared_actor_lr_scale_schedule"),
                "0.0,0.01,0.025,0.05,0.1",
            )
            self.assertEqual(
                _value(commands[method], "--gnn_freeze_epochs"), "0"
            )

    def test_rotating_worker_pool_uses_each_case_once_per_cycle(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.data_list = list(range(4))
        env.data_idx = 0
        env._train_epoch_case_budget_per_worker = 1
        env._training_case_cycle_initialized = False
        env.np_random = np.random.default_rng(7)
        observed = []
        for _ in range(4):
            env.shuffer_data()
            observed.append(env.data_list[env.data_idx])
            env.data_idx = (env.data_idx + 1) % len(env.data_list)
        self.assertEqual(set(observed), {0, 1, 2, 3})
        self.assertEqual(len(observed), len(set(observed)))
        env.reset_training_case_cycle()
        self.assertEqual(env.data_idx, 0)
        self.assertFalse(env._training_case_cycle_initialized)

    def test_autopilot_requires_gain_against_each_arms_own_pre(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "suite"
            results = root / "results"
            offsets = {"P0": -60.0, "P1": -80.0, "P2": -100.0, "P3": -90.0}
            for method, best_offset in offsets.items():
                experiment = f"fixture_ppo_gain_wave1_{method}_seed1"
                manifest = {
                    "profile": "ppo_gain_wave1",
                    "causal_arm": method,
                    "source_stage2": {"sha256": "common-digest"},
                    "architecture_contract": {"base_credit_method": method},
                    "command": [
                        "python", "train.py",
                        "--experiment_name", experiment,
                        "--num_episodes", "5",
                        "--train_sampling_size", "240",
                        "--n_rollout_threads", "20",
                        "--target_kl", "0.005",
                    ],
                }
                _write_json(
                    suite / "commands" / f"ppo_gain_wave1_{method}.json",
                    manifest,
                )
                run = results / experiment / "run1"
                _write_json(run / "evaluations" / "pre_ppo.json", _evaluation(0))
                _write_json(run / "evaluations" / "epoch_1.json", _evaluation(0))
                for epoch, scale in zip((2, 3, 4, 5), (0.8, 1.0, 0.9, 0.95)):
                    _write_json(
                        run / "evaluations" / f"epoch_{epoch}.json",
                        _evaluation(best_offset * scale),
                    )
                _write_json(run / "run_status.json", {
                    "status": "completed",
                    "phase": "joint_finetune_completed",
                    "canary_rejected": False,
                    "actor_update_health": {
                        "update_shards": 48,
                        "step_completion_rate": 1.0,
                        "zero_update_shards": 0,
                        "empty_replay_fraction": 0.0,
                        "post_update_old_policy_kl_max": 0.001,
                    },
                })
                digest = {"sha256": "same-actor-hash"}
                checkpoint = {
                    "plane_actor_summary_before_ppo": digest,
                    "plane_actor_summary_after_ppo": digest,
                    "resource_actor_summary_before_ppo": digest,
                    "resource_actor_summary_after_ppo": digest,
                    "shared_actor_summary_before_ppo": digest,
                    "shared_actor_summary_after_ppo": digest,
                    "actor_update_health": {
                        "planned_optimizer_steps": 0.0,
                        "actual_optimizer_steps": 0.0,
                    },
                    "critic_optim": {"state": {0: {"step": torch.tensor(1)}}},
                }
                checkpoint_path = run / "models" / "checkpoint_Epoch1.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)

            report = analyze_phase(suite, results, "ppo_gain_wave1")
            self.assertEqual(report["selected_method"], "P2")
            self.assertTrue(all(arm["admitted"] for arm in report["arms"]))
            self.assertTrue(
                suite.joinpath(
                    "analysis/ppo_gain_wave1_selection.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
