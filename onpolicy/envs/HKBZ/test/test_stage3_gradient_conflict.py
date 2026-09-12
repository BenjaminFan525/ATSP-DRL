"""Contracts for the Stage3 shared-encoder gradient-conflict screen."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from onpolicy.algorithms.gnn_mappo.gnn_mappo import MAPPO_Trainer
from onpolicy.scripts.train.prepare_stage3_joint_finetune import build_command
from onpolicy.scripts.train.run_stage3_joint_manifest import (
    _validate_pure_rl_command,
)


ROOT = Path(__file__).resolve().parents[4]


def _value(command, flag):
    return command[command.index(flag) + 1]


def _command(method, profile="gradient_conflict_wave1"):
    return build_command(SimpleNamespace(
        profile=profile,
        python=ROOT.parent / "conda/envs/maia-hkbz-cu124-20260903/bin/python3.11",
        experiment_name=f"contract_{profile}_{method}",
        source_checkpoint=Path("/tmp/stage2_source.pt"),
        seed=1,
        method=method,
        max_graphs_per_forward=1000,
    ))


def _tiny_trainer(method):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = SimpleNamespace(param_groups=[{
        "name": "shared_encoder", "params": [parameter]
    }])
    trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
    trainer.policy = SimpleNamespace(actor_optimizer=optimizer)
    trainer.role_names = {0: "plane", 1: "device", 2: "transporter"}
    trainer.shared_gradient_method = method
    trainer.shared_grad_ema_beta = 0.97
    trainer.shared_grad_norm_power = 0.5
    trainer.shared_grad_min_scale = 0.5
    trainer.shared_grad_max_scale = 2.0
    trainer.shared_grad_conflict_threshold = -0.05
    trainer.shared_cagrad_c = 0.2
    trainer.shared_grad_norm_ema = {}
    trainer.shared_grad_norm_ema_updates = 0
    return trainer, parameter


class Stage3GradientConflictTest(unittest.TestCase):
    def test_per_group_clipping_prevents_cross_role_norm_suppression(self):
        plane = torch.nn.Parameter(torch.tensor([0.0]))
        transporter = torch.nn.Parameter(torch.tensor([0.0]))
        optimizer = torch.optim.SGD([
            {'name': 'plane_actor', 'params': [plane]},
            {'name': 'transporter_actor', 'params': [transporter]},
        ], lr=1.0)
        trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
        trainer.policy = SimpleNamespace(actor_optimizer=optimizer)
        trainer._use_max_grad_norm = True
        trainer.max_grad_norm = 1.0
        trainer.actor_grad_clip_mode = 'per_group'
        trainer.actor_group_max_grad_norm = {
            'plane_actor': 1.0,
            'transporter_actor': 1.0,
        }
        plane.grad = torch.tensor([1.0])
        transporter.grad = torch.tensor([100.0])
        metrics = trainer._clip_actor_gradients()
        self.assertAlmostEqual(float(plane.grad.item()), 1.0, places=5)
        self.assertAlmostEqual(float(transporter.grad.item()), 1.0, places=5)
        self.assertAlmostEqual(
            metrics['actor_plane_actor_grad_norm'], 1.0, places=6
        )
        self.assertGreater(
            metrics['actor_transporter_actor_grad_norm'], 99.0
        )
        self.assertEqual(
            metrics['actor_plane_actor_grad_clip_applied'], 0.0
        )
        self.assertEqual(
            metrics['actor_transporter_actor_grad_clip_applied'], 1.0
        )

    def test_four_arms_change_only_gradient_combiner(self):
        expected = {
            "G0": "sum",
            "G1": "norm_balance",
            "G2": "norm_pcgrad",
            "G3": "cagrad",
        }
        commands = {method: _command(method) for method in expected}
        for method, command in commands.items():
            _validate_pure_rl_command(command, allow_staged_shared=True)
            self.assertEqual(
                _value(command, "--shared_gradient_method"), expected[method]
            )
            self.assertEqual(_value(command, "--max_graphs_per_forward"), "1000")
            self.assertEqual(_value(command, "--num_episodes"), "4")
            self.assertEqual(_value(command, "--actor_warmup_shards"), "12")
            self.assertEqual(_value(command, "--gnn_freeze_epochs"), "2")
            self.assertEqual(
                _value(command, "--shared_actor_lr_scale_schedule"),
                "0.0,0.0,0.01,0.01",
            )
            self.assertEqual(_value(command, "--role_loss_weighting"), "fixed")
            self.assertIn("--role_atomic_ppo", command)
            self.assertIn("--role_event_returns", command)
            self.assertIn("--role_valuenorm", command)
            self.assertIn("--shared_gradient_diagnostics", command)
            self.assertIn("--shared_encoder_activation_checkpoint", command)
            self.assertNotIn("--shared_encoder_pcgrad", command)

        ignored = {"--experiment_name", "--shared_gradient_method"}
        reference = commands["G0"]
        for method in ("G1", "G2", "G3"):
            command = commands[method]
            for index in range(0, len(reference)):
                if reference[index] in ignored:
                    continue
                if index and reference[index - 1] in ignored:
                    continue
                self.assertEqual(reference[index], command[index])

    def test_memory_canary_exercises_unfrozen_graph_1000_update(self):
        for method in ("G0", "G1", "G2", "G3"):
            command = _command(method, "gradient_conflict_memory_canary")
            _validate_pure_rl_command(command, allow_staged_shared=True)
            self.assertEqual(_value(command, "--num_episodes"), "1")
            self.assertEqual(_value(command, "--gnn_freeze_epochs"), "0")
            self.assertEqual(
                _value(command, "--shared_actor_lr_scale_schedule"), "0.01"
            )

    def test_norm_balance_is_bounded_and_checkpointable(self):
        trainer, parameter = _tiny_trainer("norm_balance")
        losses = {
            0: 10.0 * parameter[0],
            1: -parameter[0] + parameter[1],
            2: -0.5 * parameter[1],
        }
        payload, metrics = trainer._shared_role_gradient_payload(
            losses, gradient_method="norm_balance"
        )
        sum(losses.values()).backward()
        trainer._apply_shared_gradient_correction(payload)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(metrics["shared_plane_grad_scale"], 0.5)
        self.assertGreaterEqual(metrics["shared_device_grad_scale"], 0.5)
        self.assertLessEqual(metrics["shared_device_grad_scale"], 2.0)
        state = trainer.shared_gradient_state_dict()
        restored, _ = _tiny_trainer("norm_balance")
        restored.load_shared_gradient_state_dict(state)
        self.assertEqual(restored.shared_grad_norm_ema, trainer.shared_grad_norm_ema)
        self.assertEqual(restored.shared_grad_norm_ema_updates, 1)

    def test_symmetric_pcgrad_and_cagrad_are_finite(self):
        for method in ("norm_pcgrad", "cagrad"):
            trainer, parameter = _tiny_trainer(method)
            losses = {
                0: parameter[0],
                1: -parameter[0] + parameter[1],
                2: -parameter[1],
            }
            payload, metrics = trainer._shared_role_gradient_payload(
                losses, gradient_method=method
            )
            sum(losses.values()).backward()
            trainer._apply_shared_gradient_correction(payload)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue(
                torch.isfinite(torch.tensor(metrics["shared_combined_grad_norm"]))
            )
            if method == "norm_pcgrad":
                self.assertEqual(metrics["shared_pcgrad_applied"], 1.0)
                self.assertGreater(metrics["shared_pcgrad_projection_rate"], 0.0)
            else:
                total_weight = sum(
                    metrics[f"shared_cagrad_{role}_weight"]
                    for role in ("plane", "device", "transporter")
                )
                self.assertAlmostEqual(total_weight, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
