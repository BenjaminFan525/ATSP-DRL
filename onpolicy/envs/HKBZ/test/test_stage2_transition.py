"""Regression tests for canonical staged-training transitions."""

from pathlib import Path
import unittest
import warnings

import torch
import yaml

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.config.config import get_config
from onpolicy.utils.training_stage import (
    CANONICAL_JOINT_FINETUNE,
    CANONICAL_RESOURCE_JOINT,
    STAGE2_SUPERVISION_CONTRACT,
    protected_parameter_summary,
    normalize_training_stage,
    validate_stage1_m2_checkpoint,
    validate_stage2_joint_finetune_checkpoint,
)


class ResourceJointTransitionTest(unittest.TestCase):
    def test_resource_joint_freezes_exact_protected_scope(self):
        with (Path(__file__).resolve().parents[3] / "config/ac.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            ac_config = yaml.safe_load(stream)

        class Args:
            lr = 1e-4
            critic_lr = 2e-4
            opti_eps = 1e-5
            weight_decay = 0.0
            anneal_final = 1.0
            anneal_original = 1.0
            max_agent_num = 24
            max_device_num = 80
            resource_policy = "drl"
            shared_actor_lr_scale = 0.1
            plane_actor_lr_scale = 0.2
            device_actor_lr_scale = 1.0
            transporter_actor_lr_scale = 0.5

        policy = GNN_MAPPOPolicy(Args, ac_config)
        policy.set_resource_supervised_training_stage()
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in policy.ac.shared_actor_param.parameters()
        ))
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in policy.ac.plane_actor_param.parameters()
        ))
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in policy.ac.critic_param.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in policy.ac.device_actor_param.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in policy.ac.transporter_actor_param.parameters()
        ))

        # Stage3 later re-enables the critics and uses the existing joint
        # trainability controls.
        policy.set_resource_joint_training_stage()
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in policy.ac.shared_actor_param.parameters()
        ))
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in policy.ac.plane_actor_param.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in policy.ac.device_actor_param.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in policy.ac.transporter_actor_param.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in policy.ac.critic_param.parameters()
        ))

        policy.set_resource_joint_training_stage(
            train_device=False,
            train_transporter=True,
        )
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in policy.ac.device_actor_param.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in policy.ac.transporter_actor_param.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in policy.ac.critic_param.parameters()
        ))

    def test_deprecated_stage_aliases_normalize_to_resource_joint(self):
        for alias in ("device_bc", "frozen_joint"):
            with self.subTest(alias=alias):
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    args = get_config().parse_args(
                        ["--training_stage", alias]
                    )
                self.assertEqual(args.training_stage, CANONICAL_RESOURCE_JOINT)
                self.assertTrue(
                    any(issubclass(item.category, DeprecationWarning) for item in captured)
                )

    def test_full_joint_is_retired(self):
        with self.assertRaises(SystemExit):
            get_config().parse_args(["--training_stage", "full_joint"])
        with self.assertRaisesRegex(ValueError, "full_joint.*retired"):
            normalize_training_stage("full_joint", warn_deprecated=False)

    def test_joint_finetune_is_the_canonical_stage3_name(self):
        args = get_config().parse_args(
            ["--training_stage", "joint_finetune"]
        )
        self.assertEqual(args.training_stage, CANONICAL_JOINT_FINETUNE)
        self.assertEqual(
            normalize_training_stage("joint_finetune"),
            CANONICAL_JOINT_FINETUNE,
        )

    def test_stage3_handoff_requires_exact_model_and_planning_contract(self):
        target = {
            "encoder.weight": torch.ones(2, 2),
            "actor.weight": torch.ones(2, 2),
            "device_actor.weight": torch.ones(2, 2),
        }
        planning = {
            "device_lookahead_dispatch": True,
            "device_future_intent_mode": "bounded_frontier",
        }
        checkpoint = {
            "training_stage": "resource_joint",
            "phase": "resource_supervised_completed",
            "stage2_training_mode": "supervised_only",
            "stage2_supervision_contract": dict(
                STAGE2_SUPERVISION_CONTRACT
            ),
            "request_ready_prediction": True,
            "request_ready_time_scale": 3600.0,
            "plane_order_mode": "fixed",
            "plane_pair_decoder": "joint_pair",
            "global_feature_mode": "f1f2",
            "resource_lookahead_contract": dict(planning),
            "source_m2_path": "/tmp/source.pt",
            "source_m2_sha256": "a" * 64,
            "resource_actor_summary_before_bc": {"sha256": "before"},
            "resource_actor_summary_after_bc": {"sha256": "after"},
            "request_ready_predictor_summary_before": {
                "sha256": "predictor-before"
            },
            "request_ready_predictor_summary_after": {
                "sha256": "predictor-after"
            },
            "resource_bc_total_labels": 100,
            "resource_dense_ranking_total_labels": 80,
            "request_ready_total_labels": 100,
            "model": {key: value.clone() for key, value in target.items()},
        }
        result = validate_stage2_joint_finetune_checkpoint(
            checkpoint,
            target,
            plane_order_mode="fixed",
            plane_pair_decoder="joint_pair",
            global_feature_mode="f1f2",
            planning_contract=planning,
            request_ready_time_scale=3600.0,
        )
        self.assertTrue(result["resource_bc_updated"])
        self.assertEqual(result["model_summary"]["count"], 3)

        with self.assertRaisesRegex(ValueError, "time-scale mismatch"):
            validate_stage2_joint_finetune_checkpoint(
                checkpoint,
                target,
                plane_order_mode="fixed",
                plane_pair_decoder="joint_pair",
                global_feature_mode="f1f2",
                planning_contract=planning,
                request_ready_time_scale=1.0,
            )

        incompatible = dict(checkpoint)
        incompatible["resource_lookahead_contract"] = {
            **planning,
            "device_future_intent_mode": "legacy_one",
        }
        with self.assertRaisesRegex(ValueError, "planning contract mismatch"):
            validate_stage2_joint_finetune_checkpoint(
                incompatible,
                target,
                plane_order_mode="fixed",
                plane_pair_decoder="joint_pair",
                global_feature_mode="f1f2",
                planning_contract=planning,
                request_ready_time_scale=3600.0,
            )

        contaminated = dict(checkpoint)
        contaminated["shared_gradient_state"] = {"updates": 1}
        with self.assertRaisesRegex(ValueError, "PPO/critic/ValueNorm state"):
            validate_stage2_joint_finetune_checkpoint(
                contaminated,
                target,
                plane_order_mode="fixed",
                plane_pair_decoder="joint_pair",
                global_feature_mode="f1f2",
                planning_contract=planning,
                request_ready_time_scale=3600.0,
            )

    def test_strict_handoff_requires_every_protected_tensor(self):
        target = {
            "encoder.weight": torch.ones(2, 2),
            "plane_sel_enc.weight": torch.ones(2, 2),
            "actor.weight": torch.ones(2, 2),
            "plane_order_actor.weight": torch.ones(2, 2),
            "device_actor.weight": torch.ones(2, 2),
        }
        source = {
            key: value.clone()
            for key, value in target.items()
            if key != "plane_sel_enc.weight"
        }
        checkpoint = {
            "training_stage": "plane_pretrain",
            "plane_order_mode": "learned",
            "plane_pair_decoder": "joint_pair",
            "global_feature_mode": "none",
            "model": source,
        }
        with self.assertRaisesRegex(ValueError, "missing protected tensors"):
            validate_stage1_m2_checkpoint(
                checkpoint,
                target,
                plane_order_mode="learned",
                plane_pair_decoder="joint_pair",
                global_feature_mode="none",
            )

    def test_protected_summary_is_bitwise(self):
        state = {
            "encoder.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "device_actor.weight": torch.zeros(2, 2),
        }
        same = protected_parameter_summary(state)
        changed = dict(state)
        changed["encoder.weight"] = state["encoder.weight"].clone()
        changed["encoder.weight"][0, 0] += 1.0
        changed_summary = protected_parameter_summary(changed)
        self.assertEqual(same["count"], 1)
        self.assertNotEqual(same["sha256"], changed_summary["sha256"])


if __name__ == "__main__":
    unittest.main()
