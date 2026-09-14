from pathlib import Path
from types import SimpleNamespace
import unittest
from collections import Counter

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import GNN_Actor_Critic
from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.algorithms.gnn_mappo.gnn_mappo import MAPPO_Trainer
from onpolicy.config.config import get_config
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.data_generator import _allocate_profile_schedule
from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.valuenorm import ValueNorm


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train/case_0046"
)


class _PolicyArgs:
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


def _make_policy():
    with open(ROOT / "onpolicy/config/ac.yaml", "r", encoding="utf-8") as stream:
        ac_config = yaml.safe_load(stream)
    return GNN_MAPPOPolicy(_PolicyArgs, ac_config)


def _make_env(global_feature_mode="none"):
    return AircraftScheduleEnv({
        "jobs_path": str(CASE_DIR / "job.json"),
        "fixed_res_path": str(CASE_DIR / "fixed_resources.json"),
        "mobile_res_path": str(CASE_DIR / "mobile_resources.json"),
        "sites_path": str(CASE_DIR / "sites.json"),
        "flights_path": str(CASE_DIR / "flights.json"),
        "seed": 42,
        "use_domain_rand": False,
        "resource_policy": "drl",
        "n_agents": 24,
        "max_device_num": 80,
        "global_feature_mode": global_feature_mode,
    })


class JointTrainingRegressionTest(unittest.TestCase):
    def test_tail_policy_weighting_preserves_each_case_mass(self):
        args = get_config().parse_args([])
        args.episode_length = 4
        args.n_rollout_threads = 2
        args.hidden_size = 8
        args.recurrent_N = 1
        buffer = SharedReplayBuffer(args, 2, None, None, None)
        buffer.filled_steps = 4
        buffer.policy_sample_weights[:4, :, :, 0] = 0.125
        buffer.decision_times[:4, 0] = [0.0, 500.0, 750.0, 1000.0]
        buffer.decision_times[:4, 1] = [0.0, 100.0, 200.0, 300.0]
        buffer.set_team_cmax_values([1000.0, 300.0])

        metrics = buffer.apply_time_tail_policy_weights(0.75, 3.0)

        np.testing.assert_allclose(
            buffer.policy_sample_weights[:4].sum(axis=(0, 2, 3)),
            [1.0, 1.0],
            rtol=0.0,
            atol=1e-6,
        )
        self.assertGreater(
            buffer.policy_sample_weights[3, 0, 0, 0],
            buffer.policy_sample_weights[0, 0, 0, 0],
        )
        self.assertEqual(metrics['tail_policy_enabled'], 1.0)
        self.assertLess(metrics['tail_policy_case_weight_max_error'], 1e-6)

    def test_epoch_method_schedule_holds_final_value(self):
        self.assertEqual(
            HKBZ_Runner._parse_epoch_schedule('0, 0.05, 0.1', 'test'),
            (0.0, 0.05, 0.1),
        )
        self.assertAlmostEqual(
            HKBZ_Runner._epoch_schedule_value((0.0, 0.05, 0.1), 7, 9.0),
            0.1,
        )

    def test_plane_bc_temporal_weight_smoothly_emphasizes_tail(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.plane_bc_tail_start_fraction = 0.75
        runner.plane_bc_tail_weight = 3.0

        weights = runner._plane_bc_temporal_weights(
            np.asarray([0.0, 750.0, 875.0, 1000.0]),
            np.asarray([1000.0, 1000.0, 1000.0, 1000.0]),
        )

        np.testing.assert_allclose(weights, [1.0, 1.0, 2.0, 3.0])

    def test_plane_bc_two_stage_tail_weight_reaches_final_tail_target(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.plane_bc_tail_start_fraction = 0.75
        runner.plane_bc_tail_weight = 4.0
        runner.plane_bc_tail_final_start_fraction = 0.90
        runner.plane_bc_tail_final_weight = 8.0

        weights = runner._plane_bc_temporal_weights(
            np.asarray([750.0, 825.0, 900.0, 950.0, 1000.0]),
            np.full(5, 1000.0),
        )

        np.testing.assert_allclose(weights, [1.0, 2.5, 4.0, 6.0, 8.0])

    def test_dagger_tail_risk_floor_preserves_more_teacher_states(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.plane_bc_dagger_tail_start_fraction = 0.75
        runner.plane_bc_dagger_tail_teacher_rate = 0.80

        rates = runner._dagger_execution_rates(
            0.40, np.asarray([0.20, 0.74, 0.75, 0.95])
        )

        np.testing.assert_allclose(rates, [0.40, 0.40, 0.80, 0.80])

    def test_graph_batch_memory_guard(self):
        self.assertEqual(
            HKBZ_Runner._validate_graph_batch_memory_config(1, 50, 64),
            50,
        )
        with self.assertRaisesRegex(ValueError, "Unsafe graph PPO batch"):
            HKBZ_Runner._validate_graph_batch_memory_config(4, 200, 64)
        with self.assertRaisesRegex(
            ValueError, "mini_batch_size cannot exceed n_rollout_threads"
        ):
            HKBZ_Runner._validate_graph_batch_memory_config(30, 50, 1500, 20)
        self.assertEqual(
            HKBZ_Runner._validate_graph_batch_memory_config(
                30, 50, 1500, 30
            ),
            1500,
        )

    def test_policy_does_not_rebatch_pyg_batch(self):
        env = _make_env()
        try:
            obs, _, info = env.reset()
            graph_batch = Batch.from_data_list([obs.clone(), obs.clone()])
            policy = _make_policy()
            active_agents = np.repeat(
                info['active_agents'][None, :], 2, axis=0
            )
            last_indices = -np.ones_like(active_agents, dtype=np.int64)
            rnn_states = np.zeros(
                (2, env.n_agents, 1, 64), dtype=np.float32
            )

            data, _ = policy._build_inputs(
                graph_batch,
                rnn_states,
                active_agents,
                last_indices,
                last_indices,
            )

            self.assertEqual(data['graph'].num_graphs, 2)
            self.assertEqual(
                int(data['graph']['operation'].batch.max().item()), 1
            )
        finally:
            env.close()

    def test_graph_batch_preparation_preserves_rng_and_policy_outputs(self):
        env = _make_env()
        try:
            obs, _, info = env.reset()
            graph_list = [obs.clone(), obs.clone()]
            sample = (graph_list, 'unchanged')
            torch_state = torch.random.get_rng_state().clone()
            numpy_state = np.random.get_state()

            prepared = MAPPO_Trainer._prepare_graph_sample(sample)

            self.assertIsInstance(prepared[0], Batch)
            self.assertEqual(prepared[0].num_graphs, 2)
            self.assertEqual(prepared[1], 'unchanged')
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            observed_numpy_state = np.random.get_state()
            self.assertEqual(observed_numpy_state[0], numpy_state[0])
            np.testing.assert_array_equal(
                observed_numpy_state[1], numpy_state[1]
            )
            self.assertEqual(observed_numpy_state[2:], numpy_state[2:])

            policy = _make_policy()
            policy.ac.eval()
            active_agents = np.repeat(
                info['active_agents'][None, :], 2, axis=0
            )
            agent_types = np.repeat(
                info['agent_types'][None, :], 2, axis=0
            )
            last_indices = -np.ones_like(active_agents, dtype=np.int64)
            rnn_states = np.zeros(
                (2, env.n_agents, 1, 64), dtype=np.float32
            )
            with torch.inference_mode():
                list_outputs = policy.get_actions(
                    graph_list,
                    rnn_states,
                    active_agents,
                    last_indices,
                    last_indices,
                    agent_types=agent_types,
                    deterministic=True,
                )
                batch_outputs = policy.get_actions(
                    prepared[0],
                    rnn_states,
                    active_agents,
                    last_indices,
                    last_indices,
                    agent_types=agent_types,
                    deterministic=True,
                )
            for list_output, batch_output in zip(list_outputs, batch_outputs):
                torch.testing.assert_close(
                    list_output, batch_output, rtol=0.0, atol=0.0
                )
        finally:
            env.close()

    def test_sparse_pair_graph_matches_dense_policy_and_reduces_bytes(self):
        dense_env = _make_env()
        sparse_env = _make_env()
        try:
            dense_env.config['pair_feature_storage'] = 'dense_legacy'
            sparse_env.config['pair_feature_storage'] = 'sparse_legal'
            dense_obs, _, dense_info = dense_env.reset()
            sparse_obs, _, sparse_info = sparse_env.reset()

            plane_count = dense_env.n_plane_agents
            agent_count = dense_env.n_agents
            job_count = len(dense_env.job_code_list)
            site_count = len(dense_env.site_code_list)
            local_op_mask = torch.stack([
                dense_obs.op_mask.view(agent_count, -1)[
                    plane_idx,
                    plane_idx * job_count:(plane_idx + 1) * job_count,
                ]
                for plane_idx in range(plane_count)
            ])
            legal_pair_mask = (
                local_op_mask.unsqueeze(-1)
                & dense_obs.agent_job_site_mask_matrix[:plane_count]
                & dense_obs.site_mask_matrix[
                    :plane_count, None, :
                ]
            )
            reconstructed = torch.zeros(
                plane_count * job_count * site_count,
                dense_obs.pair_features.shape[-1],
            )
            reconstructed[sparse_obs.pair_feature_flat_ids.long()] = (
                sparse_obs.pair_feature_values
            )
            reconstructed = reconstructed.view(
                plane_count,
                job_count,
                site_count,
                -1,
            )
            torch.testing.assert_close(
                reconstructed[legal_pair_mask],
                dense_obs.pair_features[:plane_count][legal_pair_mask],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                sparse_obs.pair_feature_values.shape[0],
                int(legal_pair_mask.sum()),
            )

            def tensor_bytes(graph):
                return sum(
                    value.numel() * value.element_size()
                    for store in graph.stores
                    for value in store.values()
                    if torch.is_tensor(value)
                )

            self.assertLess(
                tensor_bytes(sparse_obs), 0.10 * tensor_bytes(dense_obs)
            )

            policy = _make_policy()
            policy.ac.eval()
            rnn_states = np.zeros(
                (1, agent_count, 1, 64), dtype=np.float32
            )
            last_indices = -np.ones(
                (1, agent_count), dtype=np.int64
            )
            common = {
                'rnn_states': rnn_states,
                'active_agents': dense_info['active_agents'][None, :],
                'last_op_indices': last_indices,
                'last_site_indices': last_indices,
                'agent_types': dense_info['agent_types'][None, :],
                'deterministic': True,
                'return_decision_mask': True,
            }
            with torch.inference_mode():
                dense_output = policy.get_actions(
                    Batch.from_data_list([dense_obs]), **common
                )
                sparse_output = policy.get_actions(
                    Batch.from_data_list([sparse_obs]), **common
                )
            for dense_value, sparse_value in zip(
                dense_output, sparse_output
            ):
                torch.testing.assert_close(
                    sparse_value, dense_value, rtol=1e-6, atol=1e-6
                )
            np.testing.assert_array_equal(
                sparse_info['active_agents'], dense_info['active_agents']
            )
        finally:
            dense_env.close()
            sparse_env.close()

    def test_sparse_pair_variable_length_batch_matches_dense_policy(self):
        dense_env = _make_env()
        sparse_env = _make_env()
        try:
            dense_env.config['pair_feature_storage'] = 'dense_legacy'
            sparse_env.config['pair_feature_storage'] = 'sparse_legal'
            dense_obs, _, dense_info = dense_env.reset()
            sparse_obs, _, sparse_info = sparse_env.reset()
            policy = _make_policy()
            policy.ac.eval()
            agent_count = dense_env.n_agents
            dense_graphs = []
            sparse_graphs = []
            infos = []

            for _ in range(4):
                dense_graphs.append(dense_obs.clone())
                sparse_graphs.append(sparse_obs.clone())
                infos.append(dense_info)
                rnn_states = np.zeros(
                    (1, agent_count, 1, 64), dtype=np.float32
                )
                last_indices = -np.ones(
                    (1, agent_count), dtype=np.int64
                )
                kwargs = {
                    'rnn_states': rnn_states,
                    'active_agents': dense_info['active_agents'][None, :],
                    'last_op_indices': last_indices,
                    'last_site_indices': last_indices,
                    'agent_types': dense_info['agent_types'][None, :],
                    'deterministic': True,
                    'return_decision_mask': True,
                }
                with torch.inference_mode():
                    dense_step = policy.get_actions(
                        Batch.from_data_list([dense_obs]), **kwargs
                    )
                    sparse_step = policy.get_actions(
                        Batch.from_data_list([sparse_obs]), **kwargs
                    )
                torch.testing.assert_close(
                    sparse_step[1], dense_step[1], rtol=0.0, atol=0.0
                )
                action = dense_step[1][0].cpu().numpy()
                dense_obs, _, dense_done, dense_info = dense_env.step(action)
                sparse_obs, _, sparse_done, sparse_info = sparse_env.step(action)
                np.testing.assert_array_equal(sparse_done, dense_done)
                np.testing.assert_array_equal(
                    sparse_info['active_agents'], dense_info['active_agents']
                )

            sparse_counts = [
                int(graph.pair_feature_counts[0])
                for graph in sparse_graphs
            ]
            self.assertGreater(len(set(sparse_counts)), 1)
            batch_size = len(dense_graphs)
            active_agents = np.stack([
                info['active_agents'] for info in infos
            ])
            agent_types = np.stack([
                info['agent_types'] for info in infos
            ])
            rnn_states = np.zeros(
                (batch_size, agent_count, 1, 64), dtype=np.float32
            )
            last_indices = -np.ones(
                (batch_size, agent_count), dtype=np.int64
            )
            kwargs = {
                'rnn_states': rnn_states,
                'active_agents': active_agents,
                'last_op_indices': last_indices,
                'last_site_indices': last_indices,
                'agent_types': agent_types,
                'deterministic': True,
                'return_decision_mask': True,
            }
            with torch.inference_mode():
                dense_output = policy.get_actions(
                    Batch.from_data_list(dense_graphs), **kwargs
                )
                sparse_output = policy.get_actions(
                    Batch.from_data_list(sparse_graphs), **kwargs
                )
            for dense_value, sparse_value in zip(
                dense_output, sparse_output
            ):
                torch.testing.assert_close(
                    sparse_value, dense_value, rtol=1e-6, atol=1e-6
                )
        finally:
            dense_env.close()
            sparse_env.close()

    def test_global_feature_modes_have_stable_batched_shape(self):
        observations = {}
        for mode in ("none", "f1", "f1f2", "f1f2_departure"):
            env = _make_env(global_feature_mode=mode)
            try:
                obs, _, _ = env.reset()
                features = obs.global_features.detach().cpu().numpy()
                self.assertEqual(features.shape, (1, 24))
                self.assertTrue(np.isfinite(features).all())
                observations[mode] = obs.clone()
            finally:
                env.close()

        none_features = observations["none"].global_features
        f1_features = observations["f1"].global_features
        f1f2_features = observations["f1f2"].global_features
        departure_features = observations[
            "f1f2_departure"
        ].global_features
        self.assertEqual(torch.count_nonzero(none_features).item(), 0)
        self.assertGreater(torch.count_nonzero(f1_features[:, :12]).item(), 0)
        self.assertEqual(torch.count_nonzero(f1_features[:, 12:]).item(), 0)
        self.assertGreater(torch.count_nonzero(f1f2_features[:, 12:]).item(), 0)
        self.assertGreater(
            torch.count_nonzero(departure_features[:, 12:]).item(), 0
        )
        batch = Batch.from_data_list([
            observations["f1"], observations["f1f2_departure"]
        ])
        self.assertEqual(tuple(batch.global_features.shape), (2, 24))
        policy = _make_policy()
        policy.ac.eval()
        with torch.inference_mode():
            encoded = policy.ac.encoder(batch)
        self.assertEqual(tuple(encoded["global_emb"].shape), (2, 64))
        self.assertTrue(torch.isfinite(encoded["global_emb"]).all())

    def test_global_plane_history_is_converted_to_local_index(self):
        last_op = torch.tensor([0, 17, 18, 35, 36, -1])
        safe, valid = GNN_Actor_Critic._local_plane_history_index(
            last_op,
            op_start=18,
            n_jobs=18,
        )
        self.assertEqual(safe.tolist(), [0, 0, 0, 17, 17, 0])
        self.assertEqual(valid.tolist(), [False, False, True, True, False, False])

    def test_actor_and_critic_optimizers_have_disjoint_ownership(self):
        policy = _make_policy()
        actor_ids = [
            id(param)
            for group in policy.actor_optimizer.param_groups
            for param in group["params"]
        ]
        critic_ids = [
            id(param)
            for group in policy.critic_optimizer.param_groups
            for param in group["params"]
        ]
        self.assertEqual(len(actor_ids), len(set(actor_ids)))
        self.assertEqual(len(critic_ids), len(set(critic_ids)))
        self.assertFalse(set(actor_ids) & set(critic_ids))
        self.assertEqual(
            [group["name"] for group in policy.actor_optimizer.param_groups],
            ["shared_encoder", "plane_actor", "device_actor", "transporter_actor"],
        )
        self.assertEqual(
            [group["lr"] for group in policy.actor_optimizer.param_groups],
            [1e-5, 2e-5, 1e-4, 5e-5],
        )

    def test_joint_stage_freezes_only_pretrained_actor_modules(self):
        policy = _make_policy()
        policy.set_joint_training_stage(freeze_plane=True, freeze_shared=True)
        self.assertFalse(any(param.requires_grad for param in policy.ac.shared_actor_param.parameters()))
        self.assertFalse(any(param.requires_grad for param in policy.ac.plane_actor_param.parameters()))
        self.assertTrue(all(param.requires_grad for param in policy.ac.device_actor_param.parameters()))
        self.assertTrue(all(param.requires_grad for param in policy.ac.transporter_actor_param.parameters()))
        self.assertTrue(all(param.requires_grad for param in policy.ac.critic_param.parameters()))

        policy.set_joint_training_stage(freeze_plane=False, freeze_shared=False)
        self.assertTrue(all(param.requires_grad for param in policy.ac.actor_param.parameters()))

    def test_plane_pretraining_freezes_non_plane_backends(self):
        policy = _make_policy()
        policy.set_plane_pretraining_stage()
        self.assertTrue(all(param.requires_grad for param in policy.ac.shared_actor_param.parameters()))
        self.assertTrue(all(param.requires_grad for param in policy.ac.plane_actor_param.parameters()))
        self.assertFalse(any(param.requires_grad for param in policy.ac.device_actor_param.parameters()))
        self.assertFalse(any(param.requires_grad for param in policy.ac.transporter_actor_param.parameters()))
        self.assertTrue(all(param.requires_grad for param in policy.ac.plane_critic_param.parameters()))
        self.assertFalse(any(param.requires_grad for param in policy.ac.device_critic_param.parameters()))
        self.assertFalse(any(param.requires_grad for param in policy.ac.transporter_critic_param.parameters()))

        policy.set_joint_training_stage(freeze_plane=False, freeze_shared=False)
        self.assertTrue(all(param.requires_grad for param in policy.ac.critic_param.parameters()))

    def test_gradient_accumulation_mass_uses_actual_valid_decisions(self):
        trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
        sample = [None] * 15
        sample[11] = np.array([[[1.0], [1.0], [0.0]]], dtype=np.float32)
        sample[5] = np.array([[[1.0], [1.0], [1.0]]], dtype=np.float32)
        sample[13] = np.array([[[0.125], [0.25], [0.0]]], dtype=np.float32)

        self.assertEqual(trainer._sample_mass(sample, 11), 2.0)
        self.assertEqual(trainer._sample_mass(sample, 5), 3.0)
        self.assertAlmostEqual(trainer._sample_mass(sample, 13), 0.375)

    def test_role_balancing_is_independent_of_agent_count(self):
        trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
        trainer.role_balanced_loss = True
        trainer.role_loss_coef = {0: 1.0, 1: 0.5, 2: 0.25}
        masks = torch.ones(7, 1)
        agent_types = torch.tensor([0, 0, 0, 0, 1, 1, 2])
        weights = trainer._role_sample_weights(masks, agent_types)
        self.assertAlmostEqual(float(weights[agent_types == 0].sum()), 1.0)
        self.assertAlmostEqual(float(weights[agent_types == 1].sum()), 0.5)
        self.assertAlmostEqual(float(weights[agent_types == 2].sum()), 0.25)

    def test_actor_sample_mass_only_counts_trainable_roles(self):
        trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
        sample = [None] * 15
        sample[12] = np.array([[[0, 0, 1, 2]]], dtype=np.int64)
        sample[13] = np.array(
            [[[[0.4], [0.6], [0.2], [0.3]]]],
            dtype=np.float32,
        )

        self.assertAlmostEqual(
            trainer._role_filtered_sample_mass(sample, 13, {1, 2}),
            0.5,
        )
        self.assertEqual(
            trainer._role_filtered_sample_mass(sample, 13, set()),
            0.0,
        )

    def test_nondifferentiable_actor_loss_is_skipped_without_backward(self):
        frozen_loss = torch.tensor(1.0)
        self.assertFalse(MAPPO_Trainer._backward_actor_loss(frozen_loss))

        parameter = torch.nn.Parameter(torch.tensor(2.0))
        trainable_loss = parameter.square()
        self.assertTrue(MAPPO_Trainer._backward_actor_loss(trainable_loss))
        self.assertAlmostEqual(float(parameter.grad), 4.0)

    def test_resource_joint_trainable_roles_exclude_frozen_plane(self):
        policy = _make_policy()
        policy.set_resource_joint_training_stage(
            freeze_plane=True,
            freeze_shared=True,
            train_device=True,
            train_transporter=True,
        )
        trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
        trainer.policy = policy
        trainer.joint_team_ppo = False

        self.assertEqual(trainer._trainable_actor_roles(), {1, 2})

    def test_stage3_joint_ratio_includes_every_trainable_role(self):
        policy = _make_policy()
        policy.set_joint_training_stage(
            freeze_plane=False, freeze_shared=False
        )
        trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
        trainer.policy = policy
        trainer.joint_team_ppo = True
        trainer.joint_team_ppo_scope = 'all'
        self.assertEqual(trainer._trainable_actor_roles(), {0, 1, 2})

        trainer.joint_team_ppo_scope = 'plane'
        self.assertEqual(trainer._trainable_actor_roles(), {0})

    def test_terminal_cmax_is_assigned_once_per_agent(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.hindsight_terminal_cmax_coef = 1.0
        env.total_time = 100.0
        step_rewards = {
            (0, 0): {"reward": 1.0},
            (4, 0): {"reward": 2.0},
            (1, 24): {"reward": 3.0},
            (8, 24): {"reward": 4.0},
        }
        result = env._apply_terminal_cmax_penalty(step_rewards)

        self.assertNotIn("terminal_cmax_penalty", result[(0, 0)])
        self.assertNotIn("terminal_cmax_penalty", result[(1, 24)])
        self.assertEqual(result[(4, 0)]["terminal_cmax_penalty"], -100.0)
        self.assertEqual(result[(8, 24)]["terminal_cmax_penalty"], -100.0)
        self.assertEqual(result[(4, 0)]["reward"], -98.0)
        self.assertEqual(result[(8, 24)]["reward"], -96.0)

    def test_team_cmax_is_one_case_return_without_action_replication(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.hindsight_reward_mode = "team_cmax"
        env.hindsight_terminal_cmax_coef = 1.0
        env.total_time = 123.0
        env.cycle_terminated = False
        env.plane_cycle_penalty = 20000.0
        env.current_case_path = "case_001"

        self.assertEqual(env.calculate_hindsight_rewards(), {})
        objective = env.get_training_objective()
        self.assertEqual(objective["cmax"], 123.0)
        self.assertEqual(objective["team_return"], -123.0)
        self.assertFalse(objective["cycle_terminated"])

        env.cycle_terminated = True
        cycled = env.get_training_objective()
        self.assertEqual(cycled["team_return"], -20123.0)
        self.assertEqual(cycled["cycle_penalty"], 20000.0)

    def test_team_time_returns_are_global_remaining_makespan(self):
        args = get_config().parse_args([])
        args.episode_length = 3
        args.n_rollout_threads = 1
        args.data_chunk_length = 3
        buffer = SharedReplayBuffer(args, 2, None, None, None)
        buffer.filled_steps = 3
        buffer.decision_times[:, 0] = np.array(
            [0.0, 20.0, 70.0, 100.0], dtype=np.float32
        )
        buffer.active_masks[:3, 0, :, 0] = np.array(
            [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        buffer.policy_masks[:3] = buffer.active_masks[:3]
        buffer.compute_team_time_returns(
            np.array([-100.0], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            1.0,
            np.zeros((1, 2, 1), dtype=np.float32),
        )

        expected = (-100.0, -80.0, -30.0)
        for step, target in enumerate(expected):
            active = buffer.active_masks[step, 0, :, 0] > 0.0
            np.testing.assert_allclose(
                buffer.returns[step, 0, :, 0][active],
                target,
            )
        np.testing.assert_allclose(
            buffer.rewards[:3, 0].sum(axis=(1, 2)),
            np.array([-20.0, -50.0, -30.0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(buffer.rewards[:3, 0].sum()), -100.0)

    def test_team_time_potential_returns_telescope_exactly(self):
        args = get_config().parse_args([])
        args.episode_length = 3
        args.n_rollout_threads = 1
        args.data_chunk_length = 3
        buffer = SharedReplayBuffer(args, 2, None, None, None)
        buffer.filled_steps = 3
        buffer.decision_times[:, 0] = [0.0, 20.0, 70.0, 100.0]
        buffer.potential_values[:, 0] = [-100.0, -70.0, -20.0, -5.0]
        buffer.active_masks[:3, 0, :, 0] = np.array([
            [1.0, 1.0], [1.0, 0.0], [1.0, 0.0]
        ])
        buffer.policy_masks[:3] = buffer.active_masks[:3]

        diagnostics = buffer.compute_team_time_potential_returns(
            np.array([-100.0], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            1.0,
            0.1,
            np.zeros((1, 2, 1), dtype=np.float32),
        )

        np.testing.assert_allclose(
            buffer.returns[:3, 0, 0, 0], [-90.0, -73.0, -28.0]
        )
        np.testing.assert_allclose(
            buffer.rewards[:3, 0].sum(axis=(1, 2)),
            [-17.0, -45.0, -28.0],
        )
        self.assertAlmostEqual(float(buffer.rewards[:3, 0].sum()), -90.0)
        self.assertLess(diagnostics['telescoping_max_abs_error'], 1e-8)
        self.assertEqual(diagnostics['raw_terminal_potential_abs_max'], 5.0)

    def test_team_time_potential_uses_each_cases_own_terminal_step(self):
        args = get_config().parse_args([])
        args.episode_length = 3
        args.n_rollout_threads = 2
        buffer = SharedReplayBuffer(args, 1, None, None, None)
        buffer.filled_steps = 3
        buffer.decision_times[:, 0] = [0.0, 10.0, 10.0, 10.0]
        buffer.decision_times[:, 1] = [0.0, 10.0, 20.0, 30.0]
        buffer.potential_values[:, 0] = [-10.0, -4.0, -4.0, -4.0]
        buffer.potential_values[:, 1] = [-30.0, -20.0, -10.0, -2.0]
        buffer.active_masks[0, :, 0, 0] = 1.0
        buffer.active_masks[1:3, 1, 0, 0] = 1.0
        buffer.policy_masks[:3] = buffer.active_masks[:3]
        buffer.masks[1:, 0, 0, 0] = 0.0

        diagnostics = buffer.compute_team_time_potential_returns(
            np.array([-10.0, -30.0], dtype=np.float32),
            np.array([10.0, 30.0], dtype=np.float32),
            1.0,
            1.0,
            np.zeros((2, 1, 1), dtype=np.float32),
        )

        self.assertAlmostEqual(buffer.rewards[0, 0, 0, 0], 0.0)
        self.assertEqual(diagnostics['terminal_step_min'], 1)
        self.assertEqual(diagnostics['terminal_step_max'], 3)
        self.assertLess(diagnostics['telescoping_max_abs_error'], 1e-8)

    def test_team_returns_and_case_balancing_are_decision_count_invariant(self):
        args = get_config().parse_args([])
        args.episode_length = 3
        args.n_rollout_threads = 2
        args.data_chunk_length = 3
        buffer = SharedReplayBuffer(args, 3, None, None, None)
        buffer.filled_steps = 3
        buffer.agent_types[:3, :, :] = np.array([0, 1, 2], dtype=np.int64)

        # Case 0 has five decisions; case 1 has only two. Both must still
        # contribute total policy/value mass one.
        buffer.active_masks[:3, 0, :, 0] = np.array(
            [[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32
        )
        buffer.active_masks[:3, 1, :, 0] = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32
        )
        buffer.policy_masks[:3] = buffer.active_masks[:3]

        buffer.compute_team_returns(
            np.array([-100.0, -200.0], dtype=np.float32),
            np.zeros((2, 3, 1), dtype=np.float32),
        )
        for env_idx, expected in enumerate((-100.0, -200.0)):
            active = buffer.active_masks[:3, env_idx, :, 0] > 0.0
            self.assertTrue(
                np.all(buffer.returns[:3, env_idx, :, 0][active] == expected)
            )
            self.assertAlmostEqual(
                float(buffer.rewards[:3, env_idx].sum()), expected
            )

        diagnostics = buffer.build_case_balanced_weights(
            {0: 1.0, 1: 0.5, 2: 0.25}
        )
        self.assertLess(diagnostics["policy_case_weight_max_error"], 1e-6)
        self.assertLess(diagnostics["value_case_weight_max_error"], 1e-6)
        case_masses = buffer.policy_sample_weights[:3].sum(axis=(0, 2, 3))
        np.testing.assert_allclose(case_masses, np.ones(2), atol=1e-6)

        case0 = buffer.policy_sample_weights[:3, 0, :, 0]
        self.assertAlmostEqual(float(case0[:, 0].sum()), 1.0 / 1.75)
        self.assertAlmostEqual(float(case0[:, 1].sum()), 0.5 / 1.75)
        self.assertAlmostEqual(float(case0[:, 2].sum()), 0.25 / 1.75)
        case1 = buffer.policy_sample_weights[:3, 1, :, 0]
        self.assertAlmostEqual(float(case1[:, 0].sum()), 1.0 / 1.5)
        self.assertAlmostEqual(float(case1[:, 1].sum()), 0.5 / 1.5)

    def test_valuenorm_respects_case_weights(self):
        normalizer = ValueNorm(input_shape=1, beta=0.0)
        normalizer.update(
            np.array([[10.0], [100.0]], dtype=np.float32),
            weights=np.array([[1.0], [0.0]], dtype=np.float32),
        )
        mean, _ = normalizer.running_mean_var()
        self.assertAlmostEqual(float(mean.item()), 10.0)

    def test_valuenorm_statistics_round_trip_through_state_dict(self):
        normalizer = ValueNorm(input_shape=1, beta=0.0)
        normalizer.update(
            np.array([[-100.0], [-200.0]], dtype=np.float32)
        )
        state = normalizer.state_dict()
        self.assertEqual(
            set(state), {"running_mean", "running_mean_sq", "debiasing_term"}
        )
        restored = ValueNorm(input_shape=1, beta=0.0)
        restored.load_state_dict(state)
        for name in state:
            torch.testing.assert_close(restored.state_dict()[name], state[name])

    def test_valuenorm_advantage_uses_denormalized_critic_baseline(self):
        trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
        trainer._use_popart = False
        trainer._use_valuenorm = True
        normalizer = ValueNorm(input_shape=1, beta=0.0)
        normalizer.update(
            np.array([[100.0], [200.0]], dtype=np.float32)
        )
        trainer.value_normalizer = normalizer
        raw_baseline = np.array([[120.0], [170.0]], dtype=np.float32)
        normalized_baseline = normalizer.normalize(raw_baseline).cpu().numpy()
        returns = np.array([[130.0], [160.0]], dtype=np.float32)
        advantages, restored_baseline = trainer._compute_rollout_advantages(
            returns, normalized_baseline
        )
        np.testing.assert_allclose(restored_baseline, raw_baseline, atol=1e-5)
        np.testing.assert_allclose(advantages, [[10.0], [-10.0]], atol=1e-5)

    def test_eval_and_valuenorm_flags_are_positive_switches(self):
        args = get_config().parse_args([
            "--use_eval",
            "--use_valuenorm",
            "--reset_optimizers_on_resume",
            "--canary_eval_interval_shards", "6",
            "--canary_max_regression", "0.03",
            "--canary_stop_on_regression",
        ])
        self.assertTrue(args.use_eval)
        self.assertTrue(args.use_valuenorm)
        self.assertTrue(args.reset_optimizers_on_resume)
        self.assertEqual(args.canary_eval_interval_shards, 6)
        self.assertAlmostEqual(args.canary_max_regression, 0.03)
        self.assertTrue(args.canary_stop_on_regression)
        self.assertEqual(args.hindsight_reward_mode, "team_cmax")
        self.assertEqual(args.hindsight_cmax_coef, 0.0)
        self.assertEqual(args.hindsight_shaping_coef, 0.0)
        self.assertEqual(args.hindsight_terminal_cmax_coef, 1.0)
        self.assertTrue(args.case_balanced_loss)
        self.assertFalse(args.train_domain_rand)
        self.assertEqual(args.training_stage, "auto")
        self.assertEqual(args.plane_order_mode, "fixed")
        self.assertEqual(args.plane_pair_decoder, "joint_pair")
        self.assertEqual(args.plane_bc_order_loss_coef, 0.0)
        self.assertEqual(args.plane_order_freeze_epochs, 0)
        self.assertFalse(args.bc_reference_hard_gate)
        self.assertAlmostEqual(
            args.adaptive_actor_min_step_completion, 0.9
        )

    def test_stage1_research_variants_disable_plane_order_head(self):
        from onpolicy.scripts.train.run_stage1_research_suite import variants

        run_variants, aliases = variants(4)

        self.assertFalse(aliases)
        self.assertEqual(
            [variant["global_features"] for variant in run_variants],
            ["none", "none", "f1", "f1f2"],
        )
        self.assertEqual(run_variants[0]["dagger_schedule"], "1.0")
        self.assertEqual(
            run_variants[-1]["dagger_schedule"], "1.00,0.70,0.40,0.10"
        )
        for variant in run_variants:
            with self.subTest(variant=variant["id"]):
                self.assertEqual(variant["order"], "fixed")
                self.assertEqual(variant["pair"], "joint_pair")
                self.assertEqual(variant["plane_bc_pair_loss_coef"], 1.0)
                self.assertEqual(variant["plane_bc_order_loss_coef"], 0.0)
                self.assertEqual(variant["plane_order_freeze_epochs"], 0)
                self.assertTrue(variant["adaptive_actor_kl"])
                self.assertEqual(variant["bc_reference_target_kl"], 0.0)
                self.assertFalse(variant["bc_reference_hard_gate"])
                self.assertEqual(
                    variant["adaptive_actor_lr_max_scale"], 4.0
                )

    def test_ppo_recovery_variants_change_one_factor_at_a_time(self):
        from onpolicy.scripts.train.run_stage1_research_suite import (
            ppo_recovery_variants,
        )

        run_variants, aliases = ppo_recovery_variants()

        self.assertFalse(aliases)
        self.assertEqual(
            [variant["id"] for variant in run_variants],
            [
                "P0_soft_gate_a8_e1_s01",
                "P1_soft_gate_a4_e1_s01",
                "P2_soft_gate_a4_e2_s01",
                "P3_soft_gate_a4_e2_s03",
            ],
        )
        self.assertEqual(
            [variant["actor_accum"] for variant in run_variants],
            [8, 4, 4, 4],
        )
        self.assertEqual(
            [variant["ppo_epoch"] for variant in run_variants],
            [1, 1, 2, 2],
        )
        self.assertEqual(
            [variant["shared_scale"] for variant in run_variants],
            [0.1, 0.1, 0.1, 0.3],
        )
        for variant in run_variants:
            self.assertEqual(variant["plane_bc_epochs"], 0)
            self.assertFalse(variant["bc_reference_hard_gate"])
            self.assertEqual(variant["bc_reference_target_kl"], 0.0)

    def test_plane_bc_drops_teacher_order_without_order_head(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.policy = SimpleNamespace(
            ac=SimpleNamespace(
                max_plane_agents=2,
                plane_order_actor=None,
            )
        )
        policy_actions = np.asarray(
            [[[-1, -1], [-1, -1], [-1, -1]]],
            dtype=np.int64,
        )
        teacher_results = [
            {
                "actions": np.asarray(
                    [[3, 4, 1], [5, 6, 0], [-1, -1, -1]],
                    dtype=np.int64,
                ),
                "info": {"available": True},
            }
        ]

        actions, labels, stats = runner._merge_plane_bc_actions(
            policy_actions, teacher_results
        )

        self.assertEqual(actions.shape[-1], 2)
        self.assertEqual(labels.shape[-1], 2)
        np.testing.assert_array_equal(
            actions,
            np.asarray([[[3, 4], [5, 6], [-1, -1]]], dtype=np.int64),
        )
        self.assertEqual(stats["order_labels"], 0)
        self.assertEqual(stats["order_correct"], 0)

    def test_dagger_keeps_teacher_labels_on_student_executed_envs(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.policy = SimpleNamespace(
            ac=SimpleNamespace(max_plane_agents=1, plane_order_actor=None)
        )
        policy_actions = np.asarray(
            [[[1, 2], [-1, -1]], [[3, 4], [-1, -1]]],
            dtype=np.int64,
        )
        teacher_results = [
            {
                "actions": np.asarray([[5, 6], [-1, -1]]),
                "info": {"available": True},
            },
            {
                "actions": np.asarray([[7, 8], [-1, -1]]),
                "info": {"available": True},
            },
        ]
        actions, labels, stats = runner._merge_plane_bc_actions(
            policy_actions,
            teacher_results,
            teacher_execution_mask=np.asarray([True, False]),
        )
        np.testing.assert_array_equal(actions[:, 0], [[5, 6], [3, 4]])
        np.testing.assert_array_equal(labels[:, 0], [[5, 6], [7, 8]])
        self.assertEqual(stats["teacher_executed_envs"], 1)
        self.assertEqual(stats["student_executed_envs"], 1)

    def test_per_agent_dagger_repairs_crossed_site_preferences(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.policy = SimpleNamespace(
            ac=SimpleNamespace(max_plane_agents=2, plane_order_actor=None)
        )
        policy_actions = np.asarray(
            [[[10, 1], [11, 2], [-1, -1]]], dtype=np.int64
        )
        teacher_results = [{
            'actions': np.asarray(
                [[20, 2], [21, 1], [-1, -1]], dtype=np.int64
            ),
            'info': {'available': True},
        }]

        actions, labels, stats = runner._merge_plane_bc_actions(
            policy_actions,
            teacher_results,
            teacher_execution_mask=np.asarray([[True, False]]),
        )

        self.assertEqual(len(set(actions[0, :2, 1].tolist())), 2)
        for plane_idx in range(2):
            candidates = {
                tuple(policy_actions[0, plane_idx]),
                tuple(teacher_results[0]['actions'][plane_idx]),
            }
            self.assertIn(tuple(actions[0, plane_idx]), candidates)
        np.testing.assert_array_equal(
            labels[0, :2], teacher_results[0]['actions'][:2]
        )
        self.assertEqual(stats['dagger_conflict_repairs'], 1)

    def test_composite_validation_selection_is_split_safe(self):
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.selection_metric = "composite"
        runner.selection_weights = {
            "iid": 0.50,
            "ood_stress": 0.45,
            "ood_scale": 0.05,
        }
        runner.last_eval_records = [
            {"split": "validation", "distribution": "iid", "makespan": 10.0},
            {
                "split": "validation",
                "distribution": "ood_stress",
                "makespan": 20.0,
            },
            {
                "split": "validation",
                "distribution": "ood_scale",
                "makespan": 30.0,
            },
        ]
        score = runner._selection_metrics_from_records(
            raw_makespan=20.0, evaluation_label="epoch_1"
        )
        self.assertAlmostEqual(score, 15.5)
        self.assertAlmostEqual(runner.last_eval_iid_makespan, 10.0)
        runner.last_eval_records[0]["split"] = "test"
        with self.assertRaisesRegex(RuntimeError, "final-reporting only"):
            runner._selection_metrics_from_records(
                raw_makespan=20.0, evaluation_label="epoch_2"
            )

    def test_adaptive_actor_lr_preserves_role_scales(self):
        policy = _make_policy()
        before = [
            group["lr"] for group in policy.actor_optimizer.param_groups
        ]
        metrics = policy.adapt_actor_lr(
            0.0,
            low=1e-4,
            high=5e-4,
            min_scale=0.25,
            max_scale=8.0,
            up=1.5,
            down=0.5,
        )
        after = [
            group["lr"] for group in policy.actor_optimizer.param_groups
        ]
        self.assertAlmostEqual(metrics["actor_lr_multiplier"], 1.5)
        np.testing.assert_allclose(after, np.asarray(before) * 1.5)
        np.testing.assert_allclose(
            np.asarray(after) / after[0],
            np.asarray(before) / before[0],
        )

    def test_incomplete_actor_update_downshifts_lr_and_preserves_scales(self):
        policy = _make_policy()
        policy.adapt_actor_lr(
            0.0,
            low=1e-4,
            high=5e-4,
            min_scale=0.25,
            max_scale=4.0,
            up=2.0,
            down=0.5,
        )
        before = np.asarray([
            group["lr"] for group in policy.actor_optimizer.param_groups
        ])

        metrics = policy.downshift_actor_lr(min_scale=0.25, down=0.5)
        after = np.asarray([
            group["lr"] for group in policy.actor_optimizer.param_groups
        ])

        self.assertAlmostEqual(metrics["actor_lr_multiplier"], 1.0)
        self.assertEqual(metrics["actor_lr_incomplete_downshift"], 1.0)
        np.testing.assert_allclose(after, before * 0.5)
        np.testing.assert_allclose(after / after[0], before / before[0])

    def test_reference_kl_monitor_is_not_a_gate_unless_enabled(self):
        trainer = MAPPO_Trainer.__new__(MAPPO_Trainer)
        trainer.target_kl = 0.005
        trainer.bc_reference_target_kl = 0.02
        trainer.bc_reference_hard_gate = False

        soft = trainer._kl_gate_decision(0.001, 0.03)
        self.assertTrue(soft["bc_reference_target_exceeded"])
        self.assertFalse(soft["bc_reference_kl_early_stop"])
        self.assertFalse(soft["kl_early_stop"])

        trainer.bc_reference_hard_gate = True
        hard = trainer._kl_gate_decision(0.001, 0.03)
        self.assertTrue(hard["bc_reference_kl_early_stop"])
        self.assertTrue(hard["kl_early_stop"])
        self.assertEqual(hard["kl_stop_reason_code"], 2)

    def test_adaptive_lr_requires_complete_unblocked_actor_update(self):
        complete = HKBZ_Runner._actor_lr_adaptation_decision(
            12, 12, 0, 0.9
        )
        partial = HKBZ_Runner._actor_lr_adaptation_decision(
            12, 4, 0, 0.9
        )
        blocked = HKBZ_Runner._actor_lr_adaptation_decision(
            12, 12, 1, 0.9
        )

        self.assertEqual(complete["actor_lr_update_eligible"], 1.0)
        self.assertEqual(complete[
            "actor_lr_incomplete_downshift_requested"
        ], 0.0)
        self.assertEqual(partial["actor_lr_update_eligible"], 0.0)
        self.assertEqual(partial[
            "actor_lr_incomplete_downshift_requested"
        ], 1.0)
        self.assertEqual(blocked["actor_lr_update_eligible"], 0.0)
        self.assertEqual(blocked[
            "actor_lr_incomplete_downshift_requested"
        ], 1.0)

    def test_default_dataset_profile_counts_match_split_contract(self):
        train = Counter(_allocate_profile_schedule("train", 600, 20260729))
        validation = Counter(
            _allocate_profile_schedule("validation", 120, 20260729)
        )
        self.assertEqual(sum(train.values()), 600)
        self.assertEqual(
            sum(train[name] for name in (
                "balanced", "resource_sparse", "bursty", "high_flex",
                "coupled", "light",
            )),
            480,
        )
        self.assertEqual(sum(validation.values()), 120)
        self.assertEqual(
            sum(validation[name] for name in (
                "balanced", "resource_sparse", "bursty", "high_flex",
                "coupled", "light",
            )),
            60,
        )
        self.assertEqual(validation["low_load_ood"], 6)

    def test_bc_rollout_limits_are_explicit(self):
        self.assertEqual(
            HKBZ_Runner._resolve_bc_rollout_limits(24, 24, 24),
            (24, 24),
        )
        self.assertEqual(
            HKBZ_Runner._resolve_bc_rollout_limits(1, 5, 24),
            (1, 5),
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            HKBZ_Runner._resolve_bc_rollout_limits(6, 5, 24)

    def test_evaluation_rounds_cover_each_case_once(self):
        case_counts = [3, 3, 2]
        masks = [
            HKBZ_Runner._evaluation_rank_mask(case_counts, round_idx)
            for round_idx in range(max(case_counts))
        ]
        observed_per_rank = np.stack(masks).sum(axis=0)
        self.assertEqual(observed_per_rank.tolist(), case_counts)
        self.assertEqual(int(np.stack(masks).sum()), sum(case_counts))

    def test_one_step_role_atomic_replay_and_ppo_update_is_finite(self):
        env = _make_env()
        try:
            args = get_config().parse_args([])
            args.n_rollout_threads = 1
            args.episode_length = 1
            args.data_chunk_length = 1
            args.mini_batch_size = 1
            args.ppo_epoch = 1
            args.grad_accumulation_steps = 1
            args.max_agent_num = env.n_plane_agents
            args.max_device_num = env.max_device_num
            args.resource_policy = "drl"
            args.use_valuenorm = True
            args.safe_graph_batch_pipeline = True
            # Production Stage-2 uses role-atomic PPO.  Keep this enabled in
            # the end-to-end replay regression so the post-step BC-reference
            # probe is tested after role-event aggregation, not only in the
            # simpler per-agent layout.
            args.role_atomic_ppo = True
            args.joint_team_ppo = True
            args.joint_team_ppo_scope = "all"
            args.plane_target_kl = 1.0
            args.device_target_kl = 1.0
            args.transporter_target_kl = 1.0

            with open(ROOT / "onpolicy/config/ac.yaml", "r", encoding="utf-8") as stream:
                ac_config = yaml.safe_load(stream)
            args.bc_reference_kl_coef = 0.2
            args.bc_reference_target_kl = 1.0
            policy = GNN_MAPPOPolicy(args, ac_config)
            policy.capture_bc_reference()
            trainer = MAPPO_Trainer(args, policy)
            buffer = SharedReplayBuffer(args, env.n_agents, None, None, None)

            obs, _, info = env.reset()
            buffer.graph_obs[0][0] = obs.clone()
            buffer.active_masks[0, 0, :, 0] = info["active_agents"].astype(np.float32)
            buffer.agent_types[0, 0] = info["agent_types"]

            active_masks = buffer.active_masks[0]
            last_actions = -np.ones((1, env.n_agents), dtype=np.int64)
            # Match HKBZ_Runner.prep_rollout(): collection is deterministic
            # with respect to network Dropout, even though actions are sampled.
            policy.ac.eval()
            values, actions, log_probs, next_rnn, policy_masks = policy.get_actions(
                Batch.from_data_list([obs]),
                buffer.rnn_states[0],
                active_masks,
                last_actions,
                last_actions,
                agent_types=info["agent_types"][None, :],
                return_decision_mask=True,
            )
            actions_np = actions.detach().cpu().numpy()
            next_rnn_np = next_rnn.detach().cpu().numpy()
            next_obs, _, dones, next_info = env.step(actions_np[0])
            next_rnn_np[0, dones] = 0.0

            masks = np.ones((1, env.n_agents), dtype=np.float32)
            masks[0, dones] = 0.0
            next_active_masks = np.zeros((1, env.n_agents), dtype=np.float32)
            next_active_masks[0, next_info["active_agents"]] = 1.0
            buffer.graph_insert(
                [next_obs],
                next_rnn_np,
                actions_np,
                log_probs.detach().cpu().numpy(),
                values.detach().cpu().numpy(),
                np.zeros((1, env.n_agents, 1), dtype=np.float32),
                masks,
                next_active_masks,
                policy_masks=policy_masks.detach().cpu().numpy()[..., None],
            )
            buffer.rewards[0, 0, :, 0] = -buffer.active_masks[0, 0, :, 0]

            next_values = policy.get_values(
                Batch.from_data_list([next_obs]),
                next_rnn_np,
                next_active_masks[..., None],
                actions_np[..., 0],
                actions_np[..., 1],
                agent_types=next_info["agent_types"][None, :],
            )
            buffer.compute_returns(
                next_values.detach().cpu().numpy().reshape(1, env.n_agents, 1),
                trainer.value_normalizer,
            )
            train_info = trainer.train(buffer)
            self.assertEqual(train_info["safe_graph_batch_pipeline"], 1.0)

            for key in (
                "policy_loss",
                "value_loss",
                "actor_grad_norm",
                "critic_grad_norm",
                "approx_kl",
                "clip_fraction",
                "trainable_decision_fraction",
                "plane_trainable_decision_fraction",
                "device_trainable_decision_fraction",
                "transporter_trainable_decision_fraction",
                "post_update_probe_approx_kl",
                "post_update_probe_clip_fraction",
                "post_update_probe_abs_log_ratio_max",
                "bc_reference_approx_kl",
                "bc_reference_approx_kl_p95",
                "post_update_bc_reference_approx_kl",
                "post_update_bc_reference_approx_kl_p95",
                "actor_param_update_l2",
                "actor_param_update_relative",
                "post_update_probe_count",
            ):
                self.assertTrue(np.isfinite(train_info[key]), key)

            # PPO replay must match rollout before the first optimizer step.
            # A stochastic encoder here causes a false KL stop and zero Actor
            # updates even though the training loop otherwise completes.
            self.assertEqual(train_info["kl_early_stop"], 0.0)
            self.assertGreater(train_info["actor_optimizer_steps"], 0.0)
            self.assertEqual(
                train_info["actor_planned_optimizer_steps"],
                train_info["actor_optimizer_steps"],
            )
            self.assertEqual(train_info["actor_step_completion_rate"], 1.0)
            self.assertEqual(train_info["actor_zero_update"], 0.0)
            self.assertGreater(train_info["actor_grad_norm"], 0.0)
            self.assertLess(abs(train_info["approx_kl"]), 1e-6)
            self.assertEqual(train_info["post_update_probe_count"], 1.0)
            self.assertGreater(train_info["actor_param_update_l2"], 0.0)
            self.assertGreater(train_info["actor_param_update_relative"], 0.0)
            self.assertGreater(train_info["post_update_probe_abs_log_ratio_max"], 0.0)
            self.assertLess(abs(train_info["bc_reference_approx_kl"]), 1e-6)
            self.assertGreaterEqual(
                train_info["post_update_bc_reference_approx_kl"], 0.0
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
