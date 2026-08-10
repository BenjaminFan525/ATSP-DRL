from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import GNN_Actor_Critic
from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy
from onpolicy.algorithms.utils.ptr_actor import (
    CascadePtrActor,
    DeviceRequestPtrActor,
    JointPairPtrActor,
    PlaneOrderPointer,
)
from onpolicy.envs.HKBZ.core import Job, Plane, Site
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train/case_0012"
)


def _make_env():
    config = {
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
    }
    return AircraftScheduleEnv(config)


def _masked_actions(env, info, force_remote_pid=None):
    actions = np.full((env.n_agents, 2), -1, dtype=np.int64)
    n_jobs = len(env.job_code_list)
    claimed_sites = set()

    for pid in range(env.n_plane_agents):
        if not info["active_agents"][pid]:
            continue
        op_mask = env.agent_op_mask[pid, pid * n_jobs:(pid + 1) * n_jobs]
        pair_mask = (
            op_mask[:, None]
            & env.agent_job_site_mask_matrix[pid]
        )
        if pid == force_remote_pid:
            plane = next(
                plane for plane in env.planes.values()
                if int(plane.code.split("_")[-1]) == pid
            )
            current_site_idx = env.site_code_list.index(plane.site.code)
            pair_mask[:, current_site_idx] = False
        for site_idx in claimed_sites:
            pair_mask[:, site_idx] = False
        legal_pairs = np.argwhere(pair_mask)
        assert legal_pairs.size > 0
        job_idx, site_idx = legal_pairs[0]
        actions[pid] = [pid * n_jobs + job_idx, site_idx]
        claimed_sites.add(int(site_idx))

    claimed_requests = set()
    for dev_idx in range(min(len(env.device_list), env.max_device_num)):
        agent_id = env.n_plane_agents + dev_idx
        if not info["active_agents"][agent_id]:
            continue
        request_ids = (
            np.flatnonzero(env.request_mask_matrix[agent_id, 1:]) + 1
        ).astype(int).tolist()
        available = [req_id for req_id in request_ids if req_id not in claimed_requests]
        req_idx = available[0] if available else 0
        actions[agent_id] = [req_idx, 0]
        if req_idx > 0:
            claimed_requests.add(req_idx)

    return actions


class ActionMaskTest(unittest.TestCase):
    def test_learned_plane_order_replays_exactly_and_is_permutation_equivariant(self):
        torch.manual_seed(13)
        selector = PlaneOrderPointer(embed_dim=8)
        global_emb = torch.randn(2, 8)
        plane_nodes = torch.randn(2, 4, 8)
        active = torch.tensor([
            [True, True, True, False],
            [False, True, True, True],
        ])
        outputs = selector(
            global_emb, plane_nodes, active, deterministic=True
        )
        order, ranks, log_prob, entropy, trainable = outputs
        replay = selector(
            global_emb,
            plane_nodes,
            active,
            chosen_ranks=ranks,
        )
        self.assertTrue(torch.equal(replay[0], order))
        self.assertTrue(torch.equal(replay[1], ranks))
        torch.testing.assert_close(replay[2], log_prob)
        self.assertTrue(torch.isfinite(entropy[active]).all())
        self.assertTrue(trainable.any())

        permutation = torch.tensor([2, 0, 3, 1])
        permuted = selector(
            global_emb,
            plane_nodes[:, permutation],
            active[:, permutation],
            deterministic=True,
        )
        for batch_idx in range(2):
            count = int(active[batch_idx].sum())
            original_order = order[batch_idx, :count]
            restored_order = permutation[permuted[0][batch_idx, :count]]
            self.assertTrue(torch.equal(restored_order, original_order))

        loss = -log_prob[trainable].mean()
        loss.backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in selector.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_joint_pair_actor_masks_and_replays_exact_probability(self):
        torch.manual_seed(17)
        actor = JointPairPtrActor(query_dim=12, embed_dim=8)
        query = torch.randn(2, 1, 12)
        op_nodes = torch.randn(2, 3, 8)
        site_nodes = torch.randn(2, 4, 8)
        op_mask = torch.tensor([
            [True, True, False],
            [False, True, True],
        ])
        pair_mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        pair_mask[0, 0, [0, 2]] = True
        pair_mask[0, 1, [1, 3]] = True
        pair_mask[1, 1, [0, 3]] = True
        pair_mask[1, 2, [1, 2]] = True

        op_idx, site_idx, log_prob, logits = actor(
            query,
            op_nodes,
            site_nodes,
            op_mask,
            pair_mask,
            deterministic=True,
        )
        self.assertTrue(
            pair_mask[torch.arange(2), op_idx, site_idx].all()
        )
        self.assertTrue(
            torch.isneginf(
                logits.masked_select(~pair_mask.reshape(2, -1))
            ).all()
        )
        replay = actor(
            query,
            op_nodes,
            site_nodes,
            op_mask,
            pair_mask,
            chosen_op=op_idx,
            chosen_site=site_idx,
        )
        torch.testing.assert_close(replay[2], log_prob)
        torch.testing.assert_close(replay[3], logits)

    def test_pointer_actors_never_sample_outside_masks(self):
        plane_actor = CascadePtrActor(query_dim=12, embed_dim=8, nhead=2)
        query = torch.randn(2, 1, 12)
        op_nodes = torch.randn(2, 3, 8)
        site_nodes = torch.randn(2, 4, 8)
        op_mask = torch.tensor([[True, True, False], [False, True, True]])
        pair_mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        pair_mask[0, 0, 2] = True
        pair_mask[0, 1, 1] = True
        pair_mask[1, 1, 3] = True
        pair_mask[1, 2, 0] = True

        for deterministic in (False, True):
            for _ in range(500):
                op_idx, site_idx, _, logits = plane_actor(
                    query,
                    op_nodes,
                    site_nodes,
                    op_mask,
                    pair_mask,
                    deterministic=deterministic,
                )
                self.assertTrue(pair_mask[torch.arange(2), op_idx, site_idx].all())
                self.assertTrue(
                    torch.isneginf(logits.masked_select(~pair_mask.view(2, -1))).all()
                )

        with self.assertRaisesRegex(RuntimeError, "rejected by the current mask"):
            plane_actor(
                query,
                op_nodes,
                site_nodes,
                op_mask,
                pair_mask,
                chosen_op=torch.tensor([2, 0]),
                chosen_site=torch.tensor([0, 0]),
            )

        device_actor = DeviceRequestPtrActor(query_dim=12, embed_dim=8, nhead=2)
        request_mask = torch.tensor([
            [False, True, False, True],
            [True, False, False, False],
        ])
        for deterministic in (False, True):
            for _ in range(500):
                req_idx, _, logits = device_actor(
                    query,
                    torch.randn(2, 4, 8),
                    request_mask,
                    deterministic=deterministic,
                )
                self.assertTrue(
                    request_mask.gather(1, req_idx.unsqueeze(-1)).all()
                )
                self.assertTrue(
                    torch.isneginf(logits.masked_select(~request_mask)).all()
                )

        with self.assertRaisesRegex(RuntimeError, "no legal request"):
            device_actor(
                query,
                torch.randn(2, 4, 8),
                torch.zeros(2, 4, dtype=torch.bool),
            )

    def test_plane_joint_log_prob_matches_masked_categorical(self):
        torch.manual_seed(7)
        actor = CascadePtrActor(query_dim=12, embed_dim=8, nhead=2)
        query = torch.randn(2, 1, 12)
        op_nodes = torch.randn(2, 3, 8)
        site_nodes = torch.randn(2, 4, 8)
        op_mask = torch.tensor([
            [True, True, False],
            [False, True, True],
        ])
        # Deliberately non-Cartesian: the legal site set depends on operation.
        pair_mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        pair_mask[0, 0, [0, 2]] = True
        pair_mask[0, 1, [1, 2, 3]] = True
        pair_mask[1, 1, [0, 3]] = True
        pair_mask[1, 2, [1, 2]] = True

        op_idx, site_idx, old_selected_log_prob, old_log_probs = actor(
            query,
            op_nodes,
            site_nodes,
            op_mask,
            pair_mask,
            deterministic=True,
        )
        flat_idx = op_idx * site_nodes.shape[1] + site_idx
        old_dist = torch.distributions.Categorical(logits=old_log_probs)

        self.assertTrue(torch.allclose(
            old_log_probs.exp().sum(dim=-1),
            torch.ones(2),
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertTrue(torch.isneginf(
            old_log_probs.masked_select(~pair_mask.view(2, -1))
        ).all())
        self.assertTrue(torch.allclose(
            old_selected_log_prob,
            old_dist.log_prob(flat_idx),
            atol=1e-7,
            rtol=1e-7,
        ))

        # Perturb only the new operation-conditioning path. The replayed PPO
        # ratio must equal the ratio of the actual masked Categorical policies.
        with torch.no_grad():
            actor.site_op_condition.weight.add_(
                0.05 * torch.eye(actor.embed_dim)
            )
        _, _, new_selected_log_prob, new_log_probs = actor(
            query,
            op_nodes,
            site_nodes,
            op_mask,
            pair_mask,
            chosen_op=op_idx,
            chosen_site=site_idx,
        )
        new_dist = torch.distributions.Categorical(logits=new_log_probs)
        ppo_ratio = torch.exp(new_selected_log_prob - old_selected_log_prob)
        actual_ratio = torch.exp(
            new_dist.log_prob(flat_idx) - old_dist.log_prob(flat_idx)
        )
        self.assertTrue(torch.allclose(ppo_ratio, actual_ratio, atol=1e-7, rtol=1e-6))

        # The PPO k3 estimator equals exact KL when integrated over all old
        # actions; this guards the normalization term used by KL early stop.
        log_ratio = new_log_probs - old_log_probs
        legal = pair_mask.view(2, -1)
        k3 = torch.zeros_like(log_ratio)
        k3[legal] = torch.expm1(log_ratio[legal]) - log_ratio[legal]
        expected_kl = (old_log_probs.exp() * k3).sum(dim=-1)
        exact_kl = torch.distributions.kl_divergence(old_dist, new_dist)
        self.assertTrue(torch.allclose(expected_kl, exact_kl, atol=1e-6, rtol=1e-5))

    def test_site_accepts_any_fjsp_alternative_resource(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.mobile_devices = {"R001": []}
        site = SimpleNamespace(
            resources={
                "fixed": SimpleNamespace(type="R002"),
            }
        )

        two_supported_alternatives = SimpleNamespace(resources=["R001", "R002"])
        one_supported_alternative = SimpleNamespace(resources=["R001", "R003"])
        unsupported = SimpleNamespace(resources=["R003", "R004"])
        no_resources = SimpleNamespace(resources=[])

        self.assertTrue(env._site_can_eventually_support_job(site, two_supported_alternatives))
        self.assertTrue(env._site_can_eventually_support_job(site, one_supported_alternative))
        self.assertFalse(env._site_can_eventually_support_job(site, unsupported))
        self.assertTrue(env._site_can_eventually_support_job(site, no_resources))

    def test_long_occupancy_jobs_are_forced_to_current_supported_site(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.site_code_list = ["1", "2"]
        env.job_code_list = ["ZY02", "ZY03", "ZY08"]
        plane = SimpleNamespace(
            LONG_OCCUPANCY_JOBS={"ZY02", "ZY03"},
            site=SimpleNamespace(code="1"),
            relocations_since_progress=0,
            last_departed_site_code=None,
        )
        pair_mask = env._build_plane_pair_mask(
            plane,
            own_op_mask=np.ones(3, dtype=bool),
            ptr_site_mask=np.ones(2, dtype=bool),
            job_site_mask=np.ones((3, 2), dtype=bool),
        )

        self.assertEqual(pair_mask[0].tolist(), [True, False])
        self.assertEqual(pair_mask[1].tolist(), [True, False])
        self.assertEqual(pair_mask[2].tolist(), [True, True])

    def test_immediate_return_site_is_masked_when_an_alternative_exists(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.site_code_list = ["1", "2", "3"]
        env.job_code_list = ["ZY02", "ZY03", "ZY08"]
        plane = SimpleNamespace(
            LONG_OCCUPANCY_JOBS={"ZY02", "ZY03"},
            site=SimpleNamespace(code="2"),
            relocations_since_progress=1,
            last_departed_site_code="1",
        )
        pair_mask = env._build_plane_pair_mask(
            plane,
            own_op_mask=np.array([False, False, True]),
            ptr_site_mask=np.ones(3, dtype=bool),
            job_site_mask=np.ones((3, 3), dtype=bool),
        )

        self.assertFalse(pair_mask[2, 0])
        self.assertTrue(pair_mask[2, 1:].any())

    def test_repeated_plane_state_triggers_cycle_termination(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.plane_cycle_repeat_limit = 2
        env.plane_no_progress_limit = 100
        env.plane_relocation_limit = 100
        env.cycle_terminated = False
        env.cycle_agent_ids = set()
        env.cycle_reason = ""
        plane = SimpleNamespace(
            code="Plane_0_0",
            irreversible_progress_count=0,
            total_relocations=1,
            ever_finished_jobs=set(),
            last_transport_reset_jobs=("ZY02",),
            no_progress_decisions=0,
            decision_signature_counts={},
            site=SimpleNamespace(code="2"),
            left_jobs=["ZY02", "ZY03"],
            current_jobs=[],
            finished_jobs=[],
            relocations_since_progress=1,
        )
        record = {
            "irreversible_progress_before": 0,
            "relocations_before": 0,
            "ever_finished_before": 0,
            "target_site_code": "2",
            "previous_departed_site_code": "1",
        }

        for _ in range(3):
            env._finalize_plane_action_record(plane, dict(record))

        self.assertTrue(env.cycle_terminated)
        self.assertEqual(env.cycle_agent_ids, {0})
        self.assertIn("repeated post-decision state", env.cycle_reason)

    def test_repeated_long_job_completion_cannot_farm_progress_bonus(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.plane_first_completion_bonus = 120.0
        env.plane_reset_job_penalty = 120.0
        env.plane_no_progress_penalty = 60.0
        env.plane_repeat_relocation_penalty = 300.0
        reward = env._plane_hindsight_shaping_reward({
            "new_first_completions": 0,
            "irreversible_progress_delta": 0,
            "relocation_count": 1,
            "reset_long_jobs": ["ZY02"],
            "repeat_relocation": True,
            "total_job_time": 1000.0,
            "job_time": 100.0,
            "waiting_time": 0.0,
            "trans_time": 100.0,
        })

        self.assertEqual(reward, -580.0)

    def test_relocation_restores_full_predecessor_chain(self):
        jobs = {
            code: Job(
                code=code,
                time=1,
                group="保障",
                resources=[],
                predecessor=predecessors,
                exclusive=[],
            )
            for code, predecessors in (
                ("ZY02", []),
                ("ZY03", []),
                ("ZY08", ["ZY02", "ZY03"]),
            )
        }
        site = Site("1", {
            "position": [0, 0],
            "jobs": jobs,
            "fixed_resources": [],
            "mobile_resources": [],
        })
        plane = Plane("Plane_0_0", {
            "velocity": 1,
            "site": site,
            "fuel": 100,
            "jobs": list(jobs.values()),
        })
        site.add_plane(plane)
        plane.finished_jobs.extend(["ZY02", "ZY03"])
        plane.left_jobs = ["ZY08"]
        plane.choosed_job = "ZY08"
        plane.is_transporting = True
        plane.left_trans_time = 1

        plane.update(1)
        self.assertTrue(plane.is_busy)
        self.assertEqual(set(plane.current_jobs), {"ZY02", "ZY03"})
        self.assertEqual(plane.choosed_job, "ZY08")

        plane.update(60)
        self.assertTrue(plane.is_busy)
        self.assertIn("ZY08", plane.current_jobs)
        self.assertIsNone(plane.choosed_job)

    def test_waiting_queue_tracks_planes_and_excludes_inflight_dispatch(self):
        env = _make_env()
        try:
            env.reset()
            plane = env.planes["Plane_0_0"]
            plane.start_waiting("ZY02")
            plane.choosed_job = "ZY02"
            env.waiting_sites["ZY-T"] = ["5", "16"]

            env._sync_waiting_sites()
            self.assertEqual(env.waiting_sites["ZY-T"], [])
            self.assertEqual(env.waiting_sites["ZY02"], [plane.site.code])

            device = next(
                dev for dev in env.device_list
                if dev.resource.type == "R008" and dev.resource.is_available()
            )
            device.start_transport(plane.site)
            env._sync_waiting_sites()
            self.assertEqual(env.waiting_sites["ZY02"], [])

            if device.left_trans_time == 0:
                device.finish_transport()
            else:
                device.update(device.left_trans_time)
            env._sync_waiting_sites()
            self.assertEqual(env.waiting_sites["ZY02"], [plane.site.code])
        finally:
            env.close()

    def test_environment_accepts_masked_actions_without_repair(self):
        env = _make_env()
        try:
            _, dones, info = env.reset()
            for _ in range(30):
                actions = _masked_actions(env, info)
                _, _, dones, info = env.step(actions)
                if np.all(dones):
                    break
        finally:
            env.close()

    def test_environment_rejects_masked_plane_action(self):
        env = _make_env()
        try:
            _, _, info = env.reset()
            actions = _masked_actions(env, info)
            active_planes = np.flatnonzero(info["active_agents"][:env.n_plane_agents])
            self.assertGreater(active_planes.size, 0)
            actions[int(active_planes[0])] = [-1, -1]
            with self.assertRaisesRegex(RuntimeError, "outside its operation block"):
                env.step(actions)
        finally:
            env.close()

    def test_device_contention_uses_explicit_masked_noop(self):
        env = _make_env()
        try:
            _, _, info = env.reset()
            active_non_z = []
            for _ in range(20):
                active_non_z = [
                    int(plane.code.split("_")[-1])
                    for plane in env.planes.values()
                    if plane.site.code != "Z"
                    and info["active_agents"][int(plane.code.split("_")[-1])]
                ]
                if active_non_z:
                    break
                actions = _masked_actions(env, info)
                _, _, _, info = env.step(actions)
            self.assertTrue(active_non_z)
            actions = _masked_actions(
                env,
                info,
                force_remote_pid=min(active_non_z),
            )
            _, _, _, info = env.step(actions)

            active_devices = np.flatnonzero(
                info["active_agents"][env.n_plane_agents:]
            ) + env.n_plane_agents
            self.assertGreater(active_devices.size, 1)

            actions = _masked_actions(env, info)
            selected_requests = actions[active_devices, 0]
            self.assertEqual(np.count_nonzero(selected_requests > 0), 1)
            self.assertGreater(np.count_nonzero(selected_requests == 0), 0)
            env.step(actions)
        finally:
            env.close()

    def test_device_joint_action_is_planned_before_local_side_effects(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        site_a = SimpleNamespace(code="A")
        site_b = SimpleNamespace(code="B")

        def make_resource(resource_type):
            resource = SimpleNamespace(type=resource_type, available=True)
            resource.is_available = lambda resource=resource: resource.available
            return resource

        resource_y = make_resource("Y")
        resource_x_defer = make_resource("X")
        resource_x_remote = make_resource("X")
        device_local = SimpleNamespace(
            code="local_y",
            resource=resource_y,
            site=site_a,
            is_idle=lambda: True,
        )
        device_defer = SimpleNamespace(
            code="defer_x",
            resource=resource_x_defer,
            site=site_a,
            is_idle=lambda: True,
        )
        device_remote = SimpleNamespace(
            code="remote_x",
            resource=resource_x_remote,
            site=site_a,
            is_idle=lambda: True,
        )

        noop = {
            "id": 0,
            "is_noop": True,
            "needed_res_types": [],
            "site_code": "A",
        }
        remote_request = {
            "id": 1,
            "is_noop": False,
            "needed_res_types": ["X"],
            "site_code": "B",
        }
        local_request = {
            "id": 2,
            "is_noop": False,
            "needed_res_types": ["Y"],
            "site_code": "A",
        }

        env.n_plane_agents = 0
        env.max_device_num = 3
        env.n_agents = 3
        env.device_list = [device_local, device_defer, device_remote]
        env.request_list = [noop, remote_request, local_request]
        env.request_pool = {request["id"]: request for request in env.request_list}
        env.request_mask_matrix = np.array(
            [
                [True, False, True],
                [True, True, False],
                [True, True, False],
            ],
            dtype=bool,
        )
        env.sites = {"A": site_a, "B": site_b}

        execution_order = []

        def fake_start_device_request(self, device, request, action_idx):
            if not self._device_can_dispatch(device, request):
                return False
            execution_order.append(device.code)
            if device is device_remote:
                device.site = site_b
            elif device is device_local and device_remote.site == site_a:
                # A local job can consume another idle resource at the same
                # site. This reproduces the pre-fix validation/execution race.
                resource_x_remote.available = False
            return True

        env._start_device_request = MethodType(fake_start_device_request, env)
        actions = np.array([[2, 0], [0, 0], [1, 0]], dtype=np.int64)

        env._dispatch_device_actions(actions)

        self.assertEqual(execution_order, ["remote_x", "local_y"])
        self.assertTrue(resource_x_remote.available)

    def test_heuristic_device_labels_are_step_legal(self):
        env = _make_env()
        try:
            _, _, info = env.reset()
            active_non_z = []
            for _ in range(20):
                active_non_z = [
                    int(plane.code.split("_")[-1])
                    for plane in env.planes.values()
                    if plane.site.code != "Z"
                    and info["active_agents"][int(plane.code.split("_")[-1])]
                ]
                if active_non_z:
                    break
                actions = _masked_actions(env, info)
                _, _, _, info = env.step(actions)
            self.assertTrue(active_non_z)

            actions = _masked_actions(
                env,
                info,
                force_remote_pid=min(active_non_z),
            )
            _, _, _, info = env.step(actions)

            active_devices = np.flatnonzero(
                info["active_agents"][env.n_plane_agents:]
            ) + env.n_plane_agents
            self.assertGreater(active_devices.size, 0)

            labels = env.heuristic_device_actions(return_info=True)
            actions = _masked_actions(env, info)
            device_slice = slice(env.n_plane_agents, env.n_agents)
            actions[device_slice] = labels["actions"][device_slice]
            env.step(actions)
            self.assertGreater(labels["info"]["real_dispatches"], 0)
        finally:
            env.close()

    def test_heuristic_device_labels_complete_natural_rollout(self):
        env = _make_env()
        try:
            _, dones, info = env.reset()
            for _ in range(12000):
                actions = _masked_actions(env, info)
                labels = env.heuristic_device_actions(return_info=True)
                device_slice = slice(env.n_plane_agents, env.n_agents)
                actions[device_slice] = labels["actions"][device_slice]
                _, _, dones, info = env.step(actions)
                if np.all(dones):
                    break

            self.assertTrue(np.all(dones), "heuristic device deferrals must not deadlock")
            self.assertTrue(np.isfinite(env.total_time))
            self.assertGreater(env.total_time, 0.0)
        finally:
            env.close()

    def test_same_site_device_request_settles_immediately(self):
        env = _make_env()
        try:
            env.reset()
            plane = env.planes["Plane_0_0"]
            old_site = plane.site
            target_site = env.sites["26"]
            if old_site.is_occupied:
                old_site.remove_plane()
            plane.site = target_site
            target_site.add_plane(plane)

            device = next(
                dev for dev in env.device_list
                if dev.resource.type == "R008" and dev.site == target_site
            )
            for resource in list(target_site.resources.values()):
                if resource.type == "R008":
                    target_site.remove_resource(resource)
            self.assertNotIn("R008", target_site.res_avail)

            plane.start_waiting("ZY02")
            plane.choosed_job = "ZY02"
            env.waiting_sites["ZY02"].append(target_site.code)
            env._refresh_request_pool()
            request = next(req for req in env.request_list if req.get("job_code") == "ZY02")

            self.assertTrue(env._start_device_request(device, request, request["id"]))
            self.assertFalse(plane.is_waiting)
            self.assertTrue(plane.is_busy)
            self.assertEqual(device.left_trans_time, 0)
            self.assertFalse(device.is_transporting)
            self.assertIn(device.resource.code, target_site.resources)

            env._refresh_request_pool()
            self.assertFalse(any(req.get("job_code") == "ZY02" for req in env.request_list[1:]))
        finally:
            env.close()

    def test_pending_job_request_settles_when_queued_job_differs(self):
        env = _make_env()
        try:
            env.reset()
            plane = env.planes["Plane_0_0"]
            old_site = plane.site
            target_site = env.sites["26"]
            if old_site.is_occupied:
                old_site.remove_plane()
            plane.site = target_site
            target_site.add_plane(plane)

            device = next(
                dev for dev in env.device_list
                if dev.resource.type == "R008" and dev.site == target_site
            )
            for resource in list(target_site.resources.values()):
                if resource.type == "R008":
                    target_site.remove_resource(resource)
            self.assertNotIn("R008", target_site.res_avail)

            plane.start_waiting("ZY02")
            plane.choosed_job = "ZY04"
            env.waiting_sites["ZY02"].append(target_site.code)
            env._refresh_request_pool()
            request = next(req for req in env.request_list if req.get("job_code") == "ZY02")

            self.assertTrue(env._start_device_request(device, request, request["id"]))
            self.assertFalse(plane.is_waiting)
            self.assertTrue(plane.is_busy)
            self.assertIn("ZY02", plane.current_jobs)
            if "ZY04" in plane.current_jobs:
                self.assertIsNone(plane.choosed_job)
            else:
                self.assertEqual(plane.choosed_job, "ZY04")

            env._refresh_request_pool()
            self.assertFalse(any(req.get("job_code") == "ZY02" for req in env.request_list[1:]))
        finally:
            env.close()

    def test_device_bc_loss_accepts_heuristic_labels(self):
        env = _make_env()
        try:
            obs, _, info = env.reset()
            last_actions = -np.ones((1, env.n_agents, 2), dtype=np.int64)
            active_non_z = []
            for _ in range(20):
                active_non_z = [
                    int(plane.code.split("_")[-1])
                    for plane in env.planes.values()
                    if plane.site.code != "Z"
                    and info["active_agents"][int(plane.code.split("_")[-1])]
                ]
                if active_non_z:
                    break
                actions = _masked_actions(env, info)
                obs, _, _, info = env.step(actions)
                last_actions[0] = actions
            self.assertTrue(active_non_z)

            actions = _masked_actions(
                env,
                info,
                force_remote_pid=min(active_non_z),
            )
            obs, _, _, info = env.step(actions)
            last_actions[0] = actions

            labels = env.heuristic_device_actions(return_info=True)
            actions = _masked_actions(env, info)
            device_slice = slice(env.n_plane_agents, env.n_agents)
            actions[device_slice] = labels["actions"][device_slice]

            with open(ROOT / "onpolicy/config/ac.yaml", "r", encoding="utf-8") as stream:
                ac_config = yaml.safe_load(stream)

            class Args:
                lr = 1e-4
                critic_lr = 1e-4
                opti_eps = 1e-5
                weight_decay = 0.0
                anneal_final = 1.0
                anneal_original = 1.0
                max_agent_num = env.n_plane_agents
                max_device_num = env.max_device_num
                resource_policy = "drl"

            wrapped = GNN_MAPPOPolicy(Args, ac_config)
            active_masks = np.zeros((1, env.n_agents, 1), dtype=np.float32)
            active_masks[0, info["active_agents"]] = 1.0
            agent_types = info["agent_types"][None, :]
            log_probs, _, decision_mask = wrapped.evaluate_actions(
                Batch.from_data_list([obs]),
                np.zeros((1, env.n_agents, 1, 64), dtype=np.float32),
                active_masks,
                last_actions[..., 0],
                last_actions[..., 1],
                actions[None, :, :],
                agent_types=agent_types,
                return_decision_mask=True,
            )
            active_device_mask = torch.as_tensor(
                (active_masks.squeeze(-1) > 0.0)
                & (agent_types != AircraftScheduleEnv.AGENT_TYPE_PLANE),
                dtype=torch.bool,
            )
            trainable_device_mask = active_device_mask & decision_mask
            self.assertGreater(trainable_device_mask.sum().item(), 0)
            self.assertGreater(
                (active_device_mask & ~decision_mask).sum().item(),
                0,
                "contention-created forced no-ops must be excluded from BC/PPO",
            )
            trainable_device_mask = trainable_device_mask.float()
            loss = -(log_probs * trainable_device_mask).sum() / trainable_device_mask.sum()
            self.assertTrue(torch.isfinite(loss))
            wrapped.actor_optimizer.zero_grad()
            loss.backward()
            device_grad = sum(
                float(param.grad.abs().sum())
                for param in wrapped.ac.device_actor.parameters()
                if param.grad is not None
            )
            transporter_grad = sum(
                float(param.grad.abs().sum())
                for param in wrapped.ac.transporter_actor.parameters()
                if param.grad is not None
            )
            if float(loss.detach()) > 1e-8:
                self.assertGreater(device_grad + transporter_grad, 0.0)
        finally:
            env.close()

    def test_full_policy_actions_are_accepted_by_environment(self):
        env = _make_env()
        try:
            obs, _, info = env.reset()
            with open(ROOT / "onpolicy/config/ac.yaml", "r", encoding="utf-8") as stream:
                ac_config = yaml.safe_load(stream)
            policy = GNN_Actor_Critic(
                **ac_config,
                max_plane_agents=env.n_plane_agents,
                max_device_agents=env.max_device_num,
            )
            self.assertIsNone(policy.plane_order_actor)
            self.assertIsInstance(policy.actor, JointPairPtrActor)
            data = {
                "graph": Batch.from_data_list([obs, obs.clone()]),
                "hidden_states": torch.zeros(2, env.n_agents, 1, 64),
            }
            policy_info = {
                "active_agents": torch.as_tensor(
                    np.stack([info["active_agents"], info["active_agents"]]),
                    dtype=torch.bool,
                ),
                "last_op_indices": torch.as_tensor(
                    np.stack([info["last_op_indices"], info["last_op_indices"]]),
                    dtype=torch.long,
                ),
                "last_site_indices": torch.as_tensor(
                    np.stack([info["last_site_indices"], info["last_site_indices"]]),
                    dtype=torch.long,
                ),
            }
            actions = None
            with torch.no_grad():
                for deterministic in (False, True):
                    for _ in range(10):
                        _, actions, _, _ = policy(
                            data,
                            policy_info,
                            deterministic=deterministic,
                        )
                self.assertEqual(actions.shape[-1], 2)
                log_probs, entropy = policy(
                    data,
                    policy_info,
                    chosen_op=actions[..., 0],
                    chosen_site=actions[..., 1],
                    eval_action=True,
                )
            self.assertTrue(torch.isfinite(log_probs).all())
            self.assertTrue(torch.isfinite(entropy))
            env.step(actions[0].cpu().numpy())
        finally:
            env.close()

    def test_role_backends_only_share_encoder(self):
        with open(ROOT / "onpolicy/config/ac.yaml", "r", encoding="utf-8") as stream:
            ac_config = yaml.safe_load(stream)
        policy = GNN_Actor_Critic(
            **ac_config,
            max_plane_agents=24,
            max_device_agents=80,
        )

        role_modules = [
            policy.plane_sel_enc,
            policy.device_sel_enc,
            policy.transporter_sel_enc,
            policy.actor,
            policy.device_actor,
            policy.transporter_actor,
            policy.plane_critic,
            policy.device_critic,
            policy.transporter_critic,
        ]
        role_param_ids = []
        for module in role_modules:
            ids = {id(param) for param in module.parameters()}
            self.assertTrue(ids)
            role_param_ids.append(ids)

        for idx, lhs in enumerate(role_param_ids):
            for rhs in role_param_ids[idx + 1:]:
                self.assertFalse(lhs & rhs)

        state_keys = policy.state_dict().keys()
        self.assertFalse(any(key.startswith("actor_param.") for key in state_keys))
        self.assertFalse(any(key.startswith("critic_param.") for key in state_keys))

    def test_legacy_shared_backend_checkpoint_initializes_all_roles(self):
        with open(ROOT / "onpolicy/config/ac.yaml", "r", encoding="utf-8") as stream:
            ac_config = yaml.safe_load(stream)

        class Args:
            lr = 1e-4
            critic_lr = 1e-4
            opti_eps = 1e-5
            weight_decay = 0.0
            anneal_final = 1.0
            anneal_original = 1.0
            max_agent_num = 24
            max_device_num = 80
            resource_policy = "drl"

        legacy = GNN_Actor_Critic(
            **ac_config,
            max_plane_agents=24,
            max_device_agents=80,
        ).state_dict()
        old_state = {
            key.replace("plane_sel_enc.", "sel_enc.", 1): value
            for key, value in legacy.items()
            if key.startswith("plane_sel_enc.")
        }
        old_state.update({
            key.replace("plane_critic.", "critic.", 1): value
            for key, value in legacy.items()
            if key.startswith("plane_critic.")
        })
        old_state.update({
            key: value for key, value in legacy.items()
            if key.startswith("encoder.")
            or key.startswith("actor.")
            or key.startswith("device_actor.")
        })

        wrapped = GNN_MAPPOPolicy(Args, ac_config)
        wrapped.load_model_state(old_state)
        loaded = wrapped.ac.state_dict()

        for role_prefix in ("plane_sel_enc.", "device_sel_enc.", "transporter_sel_enc."):
            for key, value in loaded.items():
                if key.startswith(role_prefix):
                    legacy_key = key.replace(role_prefix, "sel_enc.", 1)
                    self.assertTrue(torch.equal(value, old_state[legacy_key]))

        for role_prefix in ("plane_critic.", "device_critic.", "transporter_critic."):
            for key, value in loaded.items():
                if key.startswith(role_prefix):
                    legacy_key = key.replace(role_prefix, "critic.", 1)
                    self.assertTrue(torch.equal(value, old_state[legacy_key]))


if __name__ == "__main__":
    unittest.main()
