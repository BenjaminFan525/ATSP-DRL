from pathlib import Path
import unittest

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch, HeteroData

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.algorithms.utils.stage1_baselines import (
    DANIELActor,
    DANIELEncoder,
    FJSPDRLActor,
    FJSPDRLEncoder,
    L2DActor,
    MultiPPOActor,
    OperationGINEncoder,
    build_stage1_baseline_actor,
    build_stage1_baseline_encoder,
)
from onpolicy.config.config import get_config
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.scripts.train.prepare_stage1_learning_baselines import (
    baseline_command,
)
from onpolicy.utils.checkpoint_contract import (
    stage1_observation_metadata,
    validate_stage1_checkpoint_contract,
)


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train/case_0012"
)


def _ac_config():
    with (ROOT / "onpolicy/config/ac.yaml").open(
        "r", encoding="utf-8"
    ) as source:
        return yaml.safe_load(source)


def _graph(device_marker=0.0):
    graph = HeteroData()
    graph["operation"].x = torch.tensor([
        [1 / 3, 0.20, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.15, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.10, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.05, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ], dtype=torch.float32)
    sites = torch.zeros(3, 22)
    sites[:, 3:5] = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    graph["site"].x = sites
    graph["device"].x = torch.full((2, 5), float(device_marker))
    graph["request"].x = torch.full((2, 8), float(device_marker))
    graph["operation", "precedes", "operation"].edge_index = torch.tensor(
        [[0, 1, 1], [1, 2, 3]], dtype=torch.long
    )
    op_ids = torch.arange(4).repeat_interleave(3)
    site_ids = torch.arange(3).repeat(4)
    graph["operation", "assignable_to", "site"].edge_index = torch.stack(
        (op_ids, site_ids)
    )
    graph["operation", "assignable_to", "site"].edge_attr = (
        site_ids.float().unsqueeze(-1) * 0.1
    )
    return graph


class _PolicyArgs:
    lr = 1e-4
    critic_lr = 2e-4
    opti_eps = 1e-5
    weight_decay = 0.0
    anneal_final = 1.0
    anneal_original = 1.0
    max_agent_num = 24
    max_device_num = 80
    resource_policy = "heuristic"
    plane_order_mode = "fixed"
    plane_pair_decoder = "joint_pair"
    shared_actor_lr_scale = 1.0
    plane_actor_lr_scale = 1.0
    device_actor_lr_scale = 1.0
    transporter_actor_lr_scale = 1.0


class Stage1LearningBaselineTest(unittest.TestCase):
    def test_cli_exposes_all_stage1_baselines(self):
        parser = get_config()
        for name in ("proposed", "l2d", "multi_ppo", "fjsp_drl", "daniel"):
            args = parser.parse_args(["--stage1_baseline", name])
            self.assertEqual(args.stage1_baseline, name)

    def test_checkpoint_contract_is_baseline_specific(self):
        checkpoint = {
            **stage1_observation_metadata("none"),
            "plane_order_mode": "fixed",
            "plane_pair_decoder": "joint_pair",
            "stage1_baseline": "l2d",
        }
        validate_stage1_checkpoint_contract(
            checkpoint,
            global_feature_mode="none",
            plane_order_mode="fixed",
            plane_pair_decoder="joint_pair",
            stage1_baseline="l2d",
            strict_metadata=True,
        )
        with self.assertRaisesRegex(ValueError, "stage1_baseline mismatch"):
            validate_stage1_checkpoint_contract(
                checkpoint,
                global_feature_mode="none",
                plane_order_mode="fixed",
                plane_pair_decoder="joint_pair",
                stage1_baseline="daniel",
                strict_metadata=True,
            )
        legacy_proposed = dict(checkpoint)
        legacy_proposed.pop("stage1_baseline")
        validate_stage1_checkpoint_contract(
            legacy_proposed,
            global_feature_mode="none",
            plane_order_mode="fixed",
            plane_pair_decoder="joint_pair",
            stage1_baseline="proposed",
            strict_metadata=True,
        )

    def test_fair_command_generator_removes_bc_and_resume_state(self):
        source = [
            "python",
            "onpolicy/scripts/train/train_hkbz.py",
            "--env_config",
            str(ROOT / "onpolicy/config/env_plane_pretrain.yaml"),
            "--num_episodes",
            "8",
            "--plane_bc_pretrain_epochs",
            "4",
            "--plane_bc_teacher_dir",
            "/tmp/teacher",
            "--bc_reference_kl_coef",
            "0.2",
            "--bc_reference_hard_gate",
            "--resume_stage1",
            "--checkpoint_dir",
            "/tmp/checkpoint.pt",
        ]
        command = baseline_command(
            source, method="daniel", seed=3, run_tag="fair"
        )
        self.assertNotIn("--resume_stage1", command)
        self.assertNotIn("--checkpoint_dir", command)
        self.assertNotIn("--plane_bc_teacher_dir", command)
        self.assertNotIn("--bc_reference_hard_gate", command)
        self.assertEqual(
            command[command.index("--stage1_baseline") + 1], "daniel"
        )
        self.assertEqual(
            command[command.index("--plane_bc_pretrain_epochs") + 1], "0"
        )
        self.assertEqual(
            command[command.index("--bc_reference_kl_coef") + 1], "0.0"
        )

    def test_encoder_families_have_expected_shapes_and_no_device_leakage(self):
        config = _ac_config()
        expected_types = {
            "l2d": OperationGINEncoder,
            "multi_ppo": OperationGINEncoder,
            "fjsp_drl": FJSPDRLEncoder,
            "daniel": DANIELEncoder,
        }
        clean = Batch.from_data_list([_graph(0.0), _graph(0.0)])
        changed_devices = Batch.from_data_list([_graph(9.0), _graph(-7.0)])
        for name, expected_type in expected_types.items():
            with self.subTest(name=name):
                torch.manual_seed(20260901)
                encoder = build_stage1_baseline_encoder(
                    name,
                    config["common_cfg"],
                    config["encoder_cfg"],
                )
                encoder.eval()
                self.assertIsInstance(encoder, expected_type)
                with torch.no_grad():
                    output = encoder(clean)
                    changed = encoder(changed_devices)
                self.assertEqual(output["global_emb"].shape, (2, 64))
                self.assertEqual(output["op_nodes"].shape, (2, 4, 64))
                self.assertEqual(output["site_nodes"].shape, (2, 3, 64))
                self.assertEqual(output["device_nodes"].shape, (2, 2, 64))
                self.assertEqual(output["request_nodes"].shape, (2, 2, 64))
                for key in ("global_emb", "op_nodes", "site_nodes"):
                    self.assertTrue(torch.isfinite(output[key]).all())
                    torch.testing.assert_close(
                        output[key], changed[key], rtol=0.0, atol=0.0
                    )

    def test_encoders_accept_empty_edges_devices_and_requests(self):
        graph = HeteroData()
        graph["operation"].x = torch.zeros(4, 11)
        graph["site"].x = torch.zeros(3, 22)
        graph["device"].x = torch.empty(0, 5)
        graph["request"].x = torch.empty(0, 8)
        graph[
            "operation", "precedes", "operation"
        ].edge_index = torch.empty(2, 0, dtype=torch.long)
        relation = graph["operation", "assignable_to", "site"]
        relation.edge_index = torch.empty(2, 0, dtype=torch.long)
        relation.edge_attr = torch.empty(0, 1)
        batch = Batch.from_data_list([graph, graph])
        config = _ac_config()
        for name in ("l2d", "multi_ppo", "fjsp_drl", "daniel"):
            with self.subTest(name=name):
                encoder = build_stage1_baseline_encoder(
                    name,
                    config["common_cfg"],
                    config["encoder_cfg"],
                )
                encoder.eval()
                with torch.no_grad():
                    output = encoder(batch)
                self.assertEqual(output["device_nodes"].shape, (2, 0, 64))
                self.assertEqual(output["request_nodes"].shape, (2, 0, 64))
                self.assertTrue(torch.isfinite(output["global_emb"]).all())

    def test_actor_masks_sampling_and_ppo_replay(self):
        actor_types = {
            "l2d": L2DActor,
            "multi_ppo": MultiPPOActor,
            "fjsp_drl": FJSPDRLActor,
            "daniel": DANIELActor,
        }
        batch_size, op_count, site_count, embed_dim = 2, 3, 4, 8
        torch.manual_seed(71)
        query = torch.randn(batch_size, 1, 3 * embed_dim)
        op_nodes = torch.randn(batch_size, op_count, embed_dim)
        site_nodes = torch.randn(batch_size, site_count, embed_dim)
        op_mask = torch.tensor([
            [True, True, False],
            [False, True, True],
        ])
        pair_mask = torch.zeros(
            batch_size, op_count, site_count, dtype=torch.bool
        )
        pair_mask[0, 0, [0, 2]] = True
        pair_mask[0, 1, [1, 3]] = True
        pair_mask[1, 1, [0, 3]] = True
        pair_mask[1, 2, [1, 2]] = True
        pair_features = torch.rand(
            batch_size, op_count, site_count, 10
        )
        pair_features[..., 2] += 0.1

        for name, expected_type in actor_types.items():
            with self.subTest(name=name):
                torch.manual_seed(99)
                actor = build_stage1_baseline_actor(
                    name,
                    {"embed_dim": embed_dim, "activation": "relu"},
                    {
                        "query_dim": 3 * embed_dim,
                        "nhead": 2,
                        "pair_feature_dim": 10,
                    },
                )
                self.assertIsInstance(actor, expected_type)
                kwargs = (
                    {"pair_features": pair_features}
                    if actor.requires_pair_features else {}
                )
                output = actor(
                    query,
                    op_nodes,
                    site_nodes,
                    op_mask,
                    pair_mask,
                    deterministic=True,
                    **kwargs,
                )
                op_index, site_index, selected, log_probs = output
                rows = torch.arange(batch_size)
                self.assertTrue(pair_mask[rows, op_index, site_index].all())
                self.assertTrue(torch.isfinite(selected).all())
                self.assertEqual(
                    log_probs.shape, (batch_size, op_count * site_count)
                )
                probability_mass = torch.exp(log_probs).sum(dim=-1)
                torch.testing.assert_close(
                    probability_mass, torch.ones_like(probability_mass)
                )
                replay = actor(
                    query,
                    op_nodes,
                    site_nodes,
                    op_mask,
                    pair_mask,
                    chosen_op=op_index,
                    chosen_site=site_index,
                    **kwargs,
                )
                torch.testing.assert_close(replay[2], selected)
                torch.testing.assert_close(replay[3], log_probs)
                actor.zero_grad(set_to_none=True)
                (-selected.mean()).backward()
                gradient = sum(
                    float(parameter.grad.abs().sum())
                    for parameter in actor.parameters()
                    if parameter.grad is not None
                )
                self.assertGreater(gradient, 0.0)

    def test_l2d_uses_earliest_feasible_completion_site(self):
        actor = L2DActor(
            query_dim=24,
            embed_dim=8,
            nhead=2,
            pair_feature_dim=10,
        )
        for parameter in actor.operation_score.parameters():
            torch.nn.init.zeros_(parameter)
        pair_features = torch.zeros(1, 2, 3, 10)
        # op 0: completion costs are 1.1, 0.6, 0.9 -> site 1.
        pair_features[0, 0, :, 1] = torch.tensor([1.0, 0.2, 0.8])
        pair_features[0, 0, :, 2] = 0.1
        pair_features[0, 0, :, 3] = torch.tensor([0.0, 0.5, 0.0])
        pair_features[0, 0, :, 4] = torch.tensor([0.0, 0.1, 0.0])
        output = actor(
            torch.zeros(1, 1, 24),
            torch.zeros(1, 2, 8),
            torch.zeros(1, 3, 8),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, 3, dtype=torch.bool),
            deterministic=True,
            pair_features=pair_features,
        )
        self.assertEqual(int(output[0].item()), 0)
        self.assertEqual(int(output[1].item()), 1)

    def test_policy_factory_switches_encoder_and_actor_together(self):
        expected = {
            "l2d": (OperationGINEncoder, L2DActor),
            "multi_ppo": (OperationGINEncoder, MultiPPOActor),
            "fjsp_drl": (FJSPDRLEncoder, FJSPDRLActor),
            "daniel": (DANIELEncoder, DANIELActor),
        }
        for name, (encoder_type, actor_type) in expected.items():
            with self.subTest(name=name):
                args = _PolicyArgs()
                args.stage1_baseline = name
                policy = GNN_MAPPOPolicy(args, _ac_config())
                self.assertEqual(policy.ac.stage1_baseline, name)
                self.assertIsInstance(policy.ac.encoder, encoder_type)
                self.assertIsInstance(policy.ac.actor, actor_type)
                self.assertEqual(policy.max_device_agents, 0)

    def test_real_sparse_stage1_graph_batches_and_replays(self):
        environment = AircraftScheduleEnv({
            "jobs_path": str(CASE_DIR / "job.json"),
            "fixed_res_path": str(CASE_DIR / "fixed_resources.json"),
            "mobile_res_path": str(CASE_DIR / "mobile_resources.json"),
            "sites_path": str(CASE_DIR / "sites.json"),
            "flights_path": str(CASE_DIR / "flights.json"),
            "seed": 42,
            "use_domain_rand": False,
            "resource_policy": "heuristic",
            "n_agents": 24,
            "max_device_num": 80,
            "pair_feature_storage": "sparse_legal",
        })
        try:
            observation, _, info = environment.reset()
            job_count = len(environment.job_code_list)
            for _ in range(100):
                active_ids = np.flatnonzero(info["active_agents"])
                if active_ids.size >= 2:
                    break
                dispatch = np.full(
                    (environment.n_agents, 2), -1, dtype=np.int64
                )
                claimed_sites = set()
                for plane_id in active_ids:
                    local_mask = environment.agent_op_mask[
                        plane_id,
                        plane_id * job_count:(plane_id + 1) * job_count,
                    ]
                    pair_mask = (
                        local_mask[:, None]
                        & environment.agent_job_site_mask_matrix[plane_id]
                        & environment.ptr_site_mask_matrix[
                            plane_id
                        ][None, :]
                    )
                    for site_id in claimed_sites:
                        pair_mask[:, site_id] = False
                    candidates = np.argwhere(pair_mask)
                    self.assertGreater(candidates.shape[0], 0)
                    job_id, site_id = candidates[0]
                    dispatch[plane_id] = [
                        plane_id * job_count + job_id,
                        site_id,
                    ]
                    claimed_sites.add(int(site_id))
                observation, _, done, info = environment.step(dispatch)
                self.assertFalse(np.all(done))
            else:
                self.fail("Could not reach a simultaneous-plane decision state.")
            self.assertGreaterEqual(
                int(np.asarray(info["active_agents"]).sum()), 2
            )
            graphs = [observation.clone(), observation.clone()]
            active = np.repeat(
                np.asarray(info["active_agents"])[None, :], 2, axis=0
            )
            agent_types = np.repeat(
                np.asarray(info["agent_types"])[None, :], 2, axis=0
            )
            last = np.full_like(active, -1, dtype=np.int64)
            recurrent = np.zeros(
                (2, environment.n_agents, 1, 64), dtype=np.float32
            )
            active_tensor = torch.as_tensor(active, dtype=torch.bool)
            for name in ("l2d", "multi_ppo", "fjsp_drl", "daniel"):
                with self.subTest(name=name):
                    args = _PolicyArgs()
                    args.stage1_baseline = name
                    torch.manual_seed(123)
                    policy = GNN_MAPPOPolicy(args, _ac_config())
                    policy.ac.eval()
                    policy.actor_optimizer.zero_grad(set_to_none=True)
                    values, actions, log_probs, _ = policy.get_actions(
                        graphs,
                        recurrent,
                        active,
                        last,
                        last,
                        deterministic=True,
                        agent_types=agent_types,
                    )
                    with torch.no_grad():
                        replay_log_probs, entropy = policy.evaluate_actions(
                            graphs,
                            recurrent,
                            active,
                            last,
                            last,
                            actions,
                            agent_types=agent_types,
                        )
                    torch.testing.assert_close(
                        log_probs[active_tensor],
                        replay_log_probs[active_tensor],
                    )
                    action_array = actions.detach().cpu().numpy()
                    for batch_id in range(action_array.shape[0]):
                        active_ids = np.flatnonzero(active[batch_id])
                        selected_sites = action_array[
                            batch_id, active_ids, 1
                        ].astype(int)
                        self.assertEqual(
                            len(selected_sites), len(set(selected_sites.tolist()))
                        )
                        for plane_id in active_ids:
                            operation_id, site_id = action_array[
                                batch_id, plane_id, :2
                            ].astype(int)
                            local_id = operation_id - plane_id * job_count
                            self.assertTrue(environment.agent_op_mask[
                                plane_id, operation_id
                            ])
                            self.assertTrue(
                                environment.agent_job_site_mask_matrix[
                                    plane_id, local_id, site_id
                                ]
                            )
                    self.assertTrue(torch.isfinite(values[active_tensor]).all())
                    self.assertTrue(torch.isfinite(entropy))
                    (-log_probs[active_tensor].mean()).backward()
                    encoder_gradient = sum(
                        float(parameter.grad.abs().sum())
                        for parameter in policy.ac.encoder.parameters()
                        if parameter.grad is not None
                    )
                    actor_gradient = sum(
                        float(parameter.grad.abs().sum())
                        for parameter in policy.ac.actor.parameters()
                        if parameter.grad is not None
                    )
                    self.assertGreater(encoder_gradient, 0.0)
                    self.assertGreater(actor_gradient, 0.0)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
