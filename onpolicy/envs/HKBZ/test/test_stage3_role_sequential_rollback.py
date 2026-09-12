"""Regression coverage for role-sequential post-update KL rollback."""

from types import MethodType, SimpleNamespace

import numpy as np
import torch

from onpolicy.algorithms.gnn_mappo.gnn_mappo import MAPPO_Trainer


class _ModeStub:
    def eval(self):
        return self

    def train(self):
        return self


class _OptimizerStub:
    def zero_grad(self):
        return None


class _ReplayStub:
    def __init__(self):
        self.filled_steps = 1
        self.episode_length = 1
        self.n_rollout_threads = 1
        self.returns = np.zeros((1, 1, 3, 1), dtype=np.float32)
        self.value_preds = np.zeros_like(self.returns)
        self.active_masks = np.ones_like(self.returns)
        self.policy_masks = np.ones_like(self.returns)
        self.value_sample_weights = np.ones_like(self.returns)
        self.agent_types = np.asarray([[[0, 1, 2]]], dtype=np.int64)
        self.actor_case_baseline_offsets = np.zeros(1, dtype=np.float32)

        sample = [None] * 13
        sample[0] = [object()]
        sample[5] = np.ones((1, 3, 1), dtype=np.float32)
        sample[11] = np.ones((1, 3, 1), dtype=np.float32)
        sample[12] = np.asarray([[0, 1, 2]], dtype=np.int64)
        self.sample = tuple(sample)

    def graph_recurrent_generator(self, advantages, mini_batch_size):
        del advantages, mini_batch_size
        return iter((self.sample,))


def test_sequential_helper_returns_optimizer_metrics_for_macro_rollback():
    trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
    parameters = {
        name: torch.nn.Parameter(torch.tensor([1.0]))
        for name in (
            "shared_encoder",
            "plane_actor",
            "device_actor",
            "transporter_actor",
        )
    }
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter], "name": name}
            for name, parameter in parameters.items()
        ],
        lr=0.01,
    )
    trainer.policy = SimpleNamespace(actor_optimizer=optimizer)
    trainer.role_names = {0: "plane", 1: "device", 2: "transporter"}
    trainer.role_sequential_min_ess = 0.5
    trainer.actor_grad_clip_mode = "global"
    trainer.actor_group_max_grad_norm = {
        name: 1.0 for name in parameters
    }
    trainer._use_max_grad_norm = False
    trainer.max_grad_norm = 1.0
    trainer.role_atomic_ppo = True
    trainer.bc_reference_hard_gate = False
    trainer.actor_kl_backtrack = False
    trainer._last_actor_step_state = None

    trainer._preceding_role_importance = MethodType(
        lambda self, sample, preceding_roles, target_role: (
            torch.ones(1, 1), 1.0, 1
        ),
        trainer,
    )

    def update_policy(self, sample, **kwargs):
        del sample, kwargs
        for group in self.policy.actor_optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad:
                    parameter.grad = torch.ones_like(parameter)
        return {
            "actor_replay_valid": 1.0,
            "actor_optimizer_steps": 0.0,
            "actor_update_skipped": 0.0,
        }

    trainer.update_policy_net = MethodType(update_policy, trainer)
    before = {
        name: parameter.detach().clone()
        for name, parameter in parameters.items()
    }

    metrics, optimizer_metrics = trainer._sequential_actor_group_update(
        [object()], [1.0], order=(0, 1, 2)
    )

    assert metrics["actor_optimizer_steps"] == 1.0
    assert metrics["actor_optimizer_substeps"] == 4.0
    assert optimizer_metrics["actor_optimizer_steps"] == 1.0
    assert optimizer_metrics["actor_optimizer_substeps"] == 4.0
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in parameters.items()
    )

    trainer.rollback_last_actor_step()
    for name, parameter in parameters.items():
        assert torch.equal(parameter.detach(), before[name])


def test_role_sequential_kl_rejection_rolls_back_without_unbound_metrics():
    trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
    trainer.case_balanced_loss = False
    trainer._use_popart = False
    trainer._use_valuenorm = False
    trainer.role_valuenorm = False
    trainer.value_normalizer = None
    trainer.normalize_advantages = False
    trainer.role_loss_coef = {0: 1.0, 1: 1.0, 2: 1.0}
    trainer.actor_grad_accumulation_steps = 1
    trainer.grad_accumulation_steps = 1
    trainer.actor_grad_accumulation_target_graphs = 0
    trainer.grad_accumulation_target_graphs = 0
    trainer.ppo_epoch = 1
    trainer.mini_batch_size = 1
    trainer.safe_graph_batch_pipeline = False
    trainer.role_sequential_ppo = True
    trainer.shared_gradient_diagnostics = False
    trainer.role_names = {0: "plane", 1: "device", 2: "transporter"}
    trainer.role_atomic_ppo = True
    trainer.role_target_kl = {0: 0.01, 1: 0.01, 2: 0.01}
    trainer.target_kl = 0.01
    trainer.bc_reference_target_kl = 1.0
    trainer.bc_reference_hard_gate = False
    trainer.actor_kl_backtrack = False
    trainer.actor_kl_backtrack_scales = ()
    trainer.bc_reference_kl_coef = 0.0
    trainer.adaptive_bc_reference_kl = False

    ac = _ModeStub()
    ac.actor_param_without_gnn = _ModeStub()
    ac.critic_param = _ModeStub()
    trainer.policy = SimpleNamespace(
        ac=ac,
        actor_optimizer=_OptimizerStub(),
        critic_optimizer=_OptimizerStub(),
    )

    trainer._compute_rollout_advantages = MethodType(
        lambda self, returns, value_preds, agent_types=None: (
            returns.copy(), value_preds.copy()
        ),
        trainer,
    )
    trainer._normalize_rollout_advantages = MethodType(
        lambda self, advantages, buffer, rollout_steps: advantages,
        trainer,
    )
    trainer._trainable_actor_roles = MethodType(
        lambda self: frozenset((0, 1, 2)), trainer
    )
    trainer._trainable_shared_actor_parameters = MethodType(
        lambda self: [], trainer
    )

    sequential_metrics = {
        "actor_optimizer_steps": 1.0,
        "actor_optimizer_substeps": 3.0,
        "actor_grad_norm": 3.0,
        "actor_param_update_l2": 0.5,
        "actor_replay_valid": 1.0,
        "actor_empty_replay_samples": 0.0,
    }
    optimizer_metrics = {
        "actor_optimizer_steps": 1.0,
        "actor_optimizer_substeps": 3.0,
        "actor_grad_norm": 3.0,
        "actor_param_update_l2": 0.5,
    }
    trainer._sequential_actor_group_update = MethodType(
        lambda self, group, masses, **kwargs: (
            dict(sequential_metrics), dict(optimizer_metrics)
        ),
        trainer,
    )
    trainer.measure_post_update_policy_shift = MethodType(
        lambda self, sample: {
            "post_update_probe_approx_kl": 0.02,
            "post_update_bc_reference_approx_kl": 0.0,
            "post_update_plane_approx_kl": 0.02,
            "post_update_device_approx_kl": 0.0,
            "post_update_transporter_approx_kl": 0.0,
        },
        trainer,
    )

    rollback_calls = []
    trainer.rollback_last_actor_step = MethodType(
        lambda self: rollback_calls.append(True), trainer
    )
    trainer.update_value_net = MethodType(
        lambda self, sample, perform_step=True, loss_scale=1.0: {
            "value_loss": 1.0,
            "value_mean": 0.0,
            "critic_update_skipped": 0.0,
            "critic_optimizer_steps": 1.0,
            "critic_grad_norm": 1.0,
            "critic_effective_decisions": 3.0,
        },
        trainer,
    )

    train_info = trainer.train(_ReplayStub())

    assert rollback_calls == [True]
    assert train_info["actor_optimizer_rollbacks"] == 1.0
    assert train_info["actor_optimizer_steps"] == 0.0
    assert train_info["actor_optimizer_substeps"] == 0.0
    assert train_info["actor_grad_norm"] == 0.0
    assert train_info["actor_param_update_l2"] == 0.0
    assert train_info["actor_step_completion_rate"] == 0.0
