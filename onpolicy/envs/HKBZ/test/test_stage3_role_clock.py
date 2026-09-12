from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from onpolicy.algorithms.gnn_mappo.gnn_mappo import MAPPO_Trainer
from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import (
    GNN_Actor_Critic,
)
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.valuenorm import ValueNorm


def _buffer_args():
    return SimpleNamespace(
        episode_length=4,
        n_rollout_threads=1,
        hidden_size=2,
        recurrent_N=1,
        gamma=1.0,
        gae_lambda=1.0,
        data_chunk_length=1,
        use_gae=True,
        use_popart=False,
        use_valuenorm=True,
        use_proper_time_limits=False,
        algorithm_name="gnn_mappo",
    )


def test_role_event_returns_preserve_each_physical_clock():
    buffer = SharedReplayBuffer(_buffer_args(), 3, None, None, None)
    buffer.filled_steps = 3
    buffer.decision_times[:4, 0] = [0.0, 2.0, 5.0, 10.0]
    buffer.agent_types[:3, 0] = np.asarray([0, 1, 2])
    buffer.active_masks[0, 0, 0, 0] = 1.0
    buffer.active_masks[1, 0, 1, 0] = 1.0
    buffer.active_masks[2, 0, 0, 0] = 1.0
    buffer.active_masks[2, 0, 2, 0] = 1.0
    buffer.policy_masks[:] = buffer.active_masks

    diagnostics = buffer.compute_role_event_time_returns(
        team_returns=np.asarray([-10.0], dtype=np.float32),
        cmax_values=np.asarray([10.0], dtype=np.float32),
        cmax_coef=1.0,
        next_value=np.zeros((1, 3, 1), dtype=np.float32),
    )

    assert buffer.returns[0, 0, 0, 0] == -10.0
    assert buffer.returns[1, 0, 1, 0] == -8.0
    assert buffer.returns[2, 0, 0, 0] == -5.0
    assert buffer.returns[2, 0, 2, 0] == -5.0
    assert buffer.rewards[:, 0, 0, 0].sum() == -10.0
    assert buffer.rewards[:, 0, 1, 0].sum() == -8.0
    assert buffer.rewards[:, 0, 2, 0].sum() == -5.0
    assert diagnostics["plane_event_count"] == 2
    assert diagnostics["device_event_count"] == 1
    assert diagnostics["transporter_event_count"] == 1
    assert diagnostics["return_identity_max_abs_error"] < 1e-6
    assert diagnostics["cost_conservation_max_abs_error"] < 1e-6


def test_role_event_td_lambda_uses_next_same_role_value():
    buffer = SharedReplayBuffer(_buffer_args(), 3, None, None, None)
    buffer.filled_steps = 3
    buffer.decision_times[:4, 0] = [0.0, 2.0, 5.0, 10.0]
    buffer.agent_types[:3, 0] = np.asarray([0, 1, 2])
    buffer.active_masks[0, 0, 0, 0] = 1.0
    buffer.active_masks[1, 0, 1, 0] = 1.0
    buffer.active_masks[2, 0, 0, 0] = 1.0
    buffer.active_masks[2, 0, 2, 0] = 1.0
    buffer.policy_masks[:] = buffer.active_masks
    baselines = np.zeros_like(buffer.value_preds[:3])
    baselines[0, 0, 0, 0] = -8.0
    baselines[2, 0, 0, 0] = -4.0

    diagnostics = buffer.compute_role_event_time_returns(
        team_returns=np.asarray([-10.0], dtype=np.float32),
        cmax_values=np.asarray([10.0], dtype=np.float32),
        cmax_coef=1.0,
        next_value=np.zeros((1, 3, 1), dtype=np.float32),
        gae_lambda=0.95,
        value_baselines=baselines,
    )

    # Plane delta sequence: (-5 + -4 - -8)=-1, then (-5 + 0 - -4)=-1.
    # GAE0=-1 + .95*(-1)=-1.95, so the raw-scale target is -9.95.
    assert np.isclose(buffer.returns[0, 0, 0, 0], -9.95)
    assert np.isclose(buffer.returns[2, 0, 0, 0], -5.0)
    assert diagnostics["gae_lambda"] == 0.95
    assert diagnostics["td_mc_max_abs_difference"] > 0.0
    assert diagnostics["cost_conservation_max_abs_error"] < 1e-6


def test_role_event_td_lambda_can_differ_by_role():
    buffer = SharedReplayBuffer(_buffer_args(), 3, None, None, None)
    buffer.filled_steps = 3
    buffer.decision_times[:4, 0] = [0.0, 2.0, 5.0, 10.0]
    buffer.agent_types[:3, 0] = np.asarray([0, 1, 2])
    buffer.active_masks[0, 0, 0, 0] = 1.0
    buffer.active_masks[1, 0, 1, 0] = 1.0
    buffer.active_masks[2, 0, 0, 0] = 1.0
    buffer.active_masks[2, 0, 2, 0] = 1.0
    buffer.policy_masks[:] = buffer.active_masks
    baselines = np.zeros_like(buffer.value_preds[:3])
    baselines[0, 0, 0, 0] = -8.0
    baselines[2, 0, 0, 0] = -4.0

    diagnostics = buffer.compute_role_event_time_returns(
        team_returns=np.asarray([-10.0], dtype=np.float32),
        cmax_values=np.asarray([10.0], dtype=np.float32),
        cmax_coef=1.0,
        next_value=np.zeros((1, 3, 1), dtype=np.float32),
        gae_lambda=1.0,
        role_gae_lambdas={
            "plane": 0.50,
            "device": 0.80,
            "transporter": 0.90,
        },
        value_baselines=baselines,
    )

    assert np.isclose(buffer.returns[0, 0, 0, 0], -9.5)
    assert np.isclose(buffer.returns[2, 0, 0, 0], -5.0)
    assert diagnostics["heterogeneous_gae_lambda"] == 1.0
    assert diagnostics["plane_gae_lambda"] == 0.50
    assert diagnostics["device_gae_lambda"] == 0.80
    assert diagnostics["transporter_gae_lambda"] == 0.90
    assert diagnostics["mc_identity_enforced"] == 0.0


def test_role_event_critical_path_redistributes_without_changing_mass():
    buffer = SharedReplayBuffer(_buffer_args(), 3, None, None, None)
    buffer.filled_steps = 3
    buffer.decision_times[:4, 0] = [0.0, 2.0, 5.0, 10.0]
    buffer.agent_types[:3, 0] = np.asarray([0, 1, 2])
    buffer.active_masks[0, 0, 0, 0] = 1.0
    buffer.active_masks[2, 0, 0, 0] = 1.0
    buffer.policy_masks[:] = buffer.active_masks
    event_weights = np.zeros((3, 1, 3), dtype=np.float32)
    event_weights[0, 0, 0] = 9.0
    event_weights[2, 0, 0] = 1.0

    diagnostics = buffer.compute_role_event_time_returns(
        team_returns=np.asarray([-10.0], dtype=np.float32),
        cmax_values=np.asarray([10.0], dtype=np.float32),
        cmax_coef=1.0,
        next_value=np.zeros((1, 3, 1), dtype=np.float32),
        event_credit_weights=event_weights,
        event_credit_mode="critical_path",
        event_credit_uniform_mix=0.20,
    )

    # 80% * [0.9, 0.1] + 20% * elapsed [0.5, 0.5].
    assert np.isclose(buffer.rewards[0, 0, 0, 0], -8.2)
    assert np.isclose(buffer.rewards[2, 0, 0, 0], -1.8)
    assert np.isclose(buffer.returns[0, 0, 0, 0], -10.0)
    assert np.isclose(buffer.returns[2, 0, 0, 0], -1.8)
    assert diagnostics["event_credit_mode_critical_path"] == 1.0
    assert diagnostics["event_credit_fallback_sequence_count"] == 0
    assert diagnostics["cost_conservation_max_abs_error"] < 1e-6


def test_role_event_critical_path_v2_preserves_exact_cost_mass():
    buffer = SharedReplayBuffer(_buffer_args(), 3, None, None, None)
    buffer.filled_steps = 3
    buffer.decision_times[:4, 0] = [0.0, 2.0, 5.0, 10.0]
    buffer.agent_types[:3, 0] = np.asarray([0, 1, 2])
    buffer.active_masks[0, 0, 0, 0] = 1.0
    buffer.active_masks[2, 0, 0, 0] = 1.0
    buffer.policy_masks[:] = buffer.active_masks
    event_weights = np.zeros((3, 1, 3), dtype=np.float32)
    event_weights[0, 0, 0] = 2.0
    event_weights[2, 0, 0] = 8.0
    diagnostics = buffer.compute_role_event_time_returns(
        team_returns=np.asarray([-10.0], dtype=np.float32),
        cmax_values=np.asarray([10.0], dtype=np.float32),
        cmax_coef=1.0,
        next_value=np.zeros((1, 3, 1), dtype=np.float32),
        event_credit_weights=event_weights,
        event_credit_mode='critical_path_v2',
        event_credit_uniform_mix=0.25,
    )
    assert diagnostics['event_credit_mode_critical_path_v2'] == 1.0
    assert diagnostics['event_credit_fallback_sequence_count'] == 0
    assert diagnostics['cost_conservation_max_abs_error'] < 1e-6
    assert np.isclose(buffer.rewards[[0, 2], 0, 0, 0].sum(), -10.0)


def test_role_event_potential_composes_and_telescopes():
    buffer = SharedReplayBuffer(_buffer_args(), 3, None, None, None)
    buffer.filled_steps = 3
    buffer.decision_times[:4, 0] = [0.0, 2.0, 5.0, 10.0]
    buffer.potential_values[:3, 0] = [-10.0, -7.0, -4.0]
    buffer.agent_types[:3, 0] = np.asarray([0, 1, 2])
    buffer.active_masks[0, 0, 0, 0] = 1.0
    buffer.active_masks[2, 0, 0, 0] = 1.0
    buffer.policy_masks[:] = buffer.active_masks

    diagnostics = buffer.compute_role_event_time_returns(
        team_returns=np.asarray([-10.0], dtype=np.float32),
        cmax_values=np.asarray([10.0], dtype=np.float32),
        cmax_coef=1.0,
        next_value=np.zeros((1, 3, 1), dtype=np.float32),
        potential_coef=0.1,
    )

    # Potential rewards are .1*(-4 - -10)=.6 and .1*(0 - -4)=.4.
    assert np.isclose(buffer.rewards[0, 0, 0, 0], -4.4)
    assert np.isclose(buffer.rewards[2, 0, 0, 0], -4.6)
    assert np.isclose(buffer.returns[0, 0, 0, 0], -9.0)
    assert np.isclose(buffer.returns[2, 0, 0, 0], -4.6)
    assert diagnostics["return_identity_max_abs_error"] < 1e-6
    assert diagnostics["potential_telescoping_max_abs_error"] < 1e-6


def test_sqrt_event_weights_are_case_balanced_and_bounded():
    buffer = SharedReplayBuffer(_buffer_args(), 3, None, None, None)
    buffer.filled_steps = 4
    buffer.agent_types[:4, 0] = np.asarray([0, 1, 2])
    buffer.active_masks[:4, 0, :, 0] = 1.0
    buffer.policy_masks[:4, 0, :, 0] = 0.0
    buffer.policy_masks[:4, 0, 0, 0] = 1.0
    buffer.policy_masks[:2, 0, 1, 0] = 1.0
    buffer.policy_masks[0, 0, 2, 0] = 1.0

    diagnostics = buffer.build_case_balanced_weights(
        {0: 1.0, 1: 0.5, 2: 0.5},
        role_loss_weighting="sqrt_event",
        role_loss_min_share=0.15,
        role_loss_max_share=0.60,
    )

    assert np.isclose(buffer.policy_sample_weights.sum(), 1.0)
    masses = [
        buffer.policy_sample_weights[:4, 0, role, 0].sum()
        for role in range(3)
    ]
    expected = np.sqrt([4.0, 2.0, 1.0])
    expected /= expected.sum()
    assert np.allclose(masses, expected)
    assert diagnostics["policy_case_weight_max_error"] < 1e-6


def test_role_joint_views_do_not_mix_log_probabilities():
    trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
    trainer.role_names = {0: "plane", 1: "device", 2: "transporter"}
    trainer.role_loss_coef = {0: 1.0, 1: 0.5, 2: 0.5}
    trainer.case_balanced_loss = True
    trainer.role_target_kl = {0: 0.0025, 1: 0.005, 2: 0.005}
    new = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    old = torch.zeros((2, 3, 1))
    advantages = torch.tensor(
        [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]
    )
    masks = torch.ones((2, 3, 1))
    agent_types = torch.tensor([[0, 1, 2], [0, 1, 2]])
    base_weights = torch.full((2, 3, 1), 1.0 / 6.0)

    views = trainer._role_joint_views(
        new, old, advantages, masks, agent_types, base_weights
    )
    assert [view["name"] for view in views] == [
        "plane", "device", "transporter"
    ]
    assert torch.allclose(views[0]["new_log_prob"].squeeze(-1), new[:, 0])
    assert torch.allclose(views[1]["new_log_prob"].squeeze(-1), new[:, 1])
    assert torch.allclose(views[2]["new_log_prob"].squeeze(-1), new[:, 2])
    assert torch.allclose(views[0]["advantage"].squeeze(-1), advantages[:, 0, 0])
    assert trainer._role_kl_gate({"plane_approx_kl": 0.003}) == {
        "plane": True,
        "device": False,
        "transporter": False,
    }


def test_sqrt_event_role_views_preserve_buffer_case_mass():
    trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
    views = [
        {
            "coefficient": 1.0,
            "weights": torch.tensor([[0.45], [0.15]]),
        },
        {
            "coefficient": 0.5,
            "weights": torch.tensor([[0.25]]),
        },
        {
            "coefficient": 0.5,
            "weights": torch.tensor([[0.15]]),
        },
    ]
    trainer.role_loss_weighting = "sqrt_event"
    observed = trainer._role_view_loss_weights(views)
    for actual, expected in zip(observed, views):
        assert torch.equal(actual, expected["weights"])
    assert torch.isclose(sum(value.sum() for value in observed), torch.tensor(1.0))

    trainer.role_loss_weighting = "fixed"
    fixed = trainer._role_view_loss_weights(views)
    assert torch.isclose(fixed[0].sum(), torch.tensor(0.5))
    assert torch.isclose(fixed[1].sum(), torch.tensor(0.25))
    assert torch.isclose(fixed[2].sum(), torch.tensor(0.25))


def test_role_valuenorm_clones_shared_stage2_state():
    trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
    trainer.value_normalizer = ValueNorm(1)
    trainer.role_valuenorm = True
    trainer.role_names = {0: "plane", 1: "device", 2: "transporter"}
    trainer.role_value_normalizers = {
        role: ValueNorm(1) for role in trainer.role_names
    }
    source = ValueNorm(1)
    source.update(np.asarray([[10.0], [20.0]], dtype=np.float32))
    trainer.load_value_normalizer_state(source.state_dict())

    for normalizer in trainer.role_value_normalizers.values():
        for key, value in source.state_dict().items():
            assert torch.equal(normalizer.state_dict()[key], value)


def test_atomic_actor_rollback_restores_parameters_and_adam_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam([
        {"params": list(model.parameters()), "name": "plane_actor"}
    ], lr=0.01)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
    trainer.policy = SimpleNamespace(actor_optimizer=optimizer)
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    before_optimizer = deepcopy(optimizer.state_dict())
    parameter_snapshot = trainer._snapshot_actor_parameters()
    trainer._last_actor_step_state = trainer._snapshot_actor_step_state(
        parameter_snapshot
    )

    optimizer.zero_grad()
    model(torch.full((1, 2), 2.0)).sum().backward()
    optimizer.step()
    trainer.rollback_last_actor_step()

    for parameter, expected in zip(model.parameters(), before_parameters):
        assert torch.equal(parameter, expected)
    restored = optimizer.state_dict()
    assert restored["param_groups"] == before_optimizer["param_groups"]
    for parameter_id, expected_state in before_optimizer["state"].items():
        for key, expected in expected_state.items():
            observed = restored["state"][parameter_id][key]
            if torch.is_tensor(expected):
                assert torch.equal(observed, expected)
            else:
                assert observed == expected


def test_shared_lr_schedule_is_independently_capped_from_head_adaptation():
    shared = torch.nn.Parameter(torch.ones(1))
    head = torch.nn.Parameter(torch.ones(1))
    policy = GNN_MAPPOPolicy.__new__(GNN_MAPPOPolicy)
    policy.lr = 5e-6
    policy.actor_lr_multiplier = 4.0
    policy.actor_lr_decay_factor = 1.0
    policy.shared_actor_lr_scale = 0.1
    policy.shared_actor_lr_max_multiplier = 2.0
    policy.actor_optimizer = torch.optim.Adam([
        {
            "params": [shared], "name": "shared_encoder",
            "lr_scale": 0.1, "lr": 5e-7,
        },
        {
            "params": [head], "name": "device_actor",
            "lr_scale": 1.0, "lr": 5e-6,
        },
    ])

    metrics = policy.set_shared_actor_lr_scale(0.025)

    assert np.isclose(metrics["shared_actor_lr"], 2.5e-7)
    assert np.isclose(policy._actor_group_lr("device_actor"), 2.0e-5)
    assert metrics["shared_actor_lr_multiplier_effective"] == 2.0


def test_pcgrad_projects_only_conflicting_shared_role_gradients():
    shared = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
    trainer.role_names = {0: "plane", 1: "device", 2: "transporter"}
    trainer.shared_grad_conflict_threshold = -0.05
    trainer.shared_grad_norm_ema = {}
    trainer.policy = SimpleNamespace(
        actor_optimizer=torch.optim.Adam([
            {"params": [shared], "name": "shared_encoder"}
        ])
    )
    role_losses = {
        0: shared[0],
        1: -shared[0] + shared[1],
        2: shared[1],
    }

    payload, metrics = trainer._shared_role_gradient_payload(
        role_losses, apply_pcgrad=True
    )
    sum(role_losses.values()).backward()
    raw_gradient = shared.grad.detach().clone()
    trainer._apply_shared_gradient_correction(payload)

    assert metrics["shared_plane_device_grad_cosine"] < 0.0
    assert metrics["shared_pcgrad_applied"] == 1.0
    assert metrics["shared_pcgrad_projection_fraction"] > 0.0
    assert not torch.allclose(shared.grad, raw_gradient)


def test_shared_encoder_checkpoint_preserves_gradient_and_detaches_critic_path():
    class ToyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2, bias=False)

        def forward(self, graph):
            return {"encoded": torch.sin(self.linear(graph))}

    baseline = GNN_Actor_Critic.__new__(GNN_Actor_Critic)
    torch.nn.Module.__init__(baseline)
    baseline.encoder = ToyEncoder()
    baseline.shared_encoder_activation_checkpoint = False

    checkpointed = GNN_Actor_Critic.__new__(GNN_Actor_Critic)
    torch.nn.Module.__init__(checkpointed)
    checkpointed.encoder = ToyEncoder()
    checkpointed.encoder.load_state_dict(baseline.encoder.state_dict())
    checkpointed.shared_encoder_activation_checkpoint = True

    graph = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10.0
    baseline._encode_graph(graph, actor_grad=True)["encoded"].sum().backward()
    checkpointed._encode_graph(
        graph, actor_grad=True
    )["encoded"].sum().backward()
    assert torch.allclose(
        baseline.encoder.linear.weight.grad,
        checkpointed.encoder.linear.weight.grad,
        atol=1e-7,
        rtol=1e-6,
    )

    checkpointed.encoder.zero_grad(set_to_none=True)
    critic_encoding = checkpointed._encode_graph(graph, actor_grad=False)
    assert not critic_encoding["encoded"].requires_grad
    assert checkpointed.encoder.linear.weight.grad is None
