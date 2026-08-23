"""Regression tests for the canonical Stage-1 M2 -> resource-joint path."""

from pathlib import Path
import unittest
import warnings

import torch
import yaml

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.config.config import get_config
from onpolicy.utils.training_stage import (
    CANONICAL_RESOURCE_JOINT,
    protected_parameter_summary,
    normalize_training_stage,
    validate_stage1_m2_checkpoint,
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
