from pathlib import Path
import unittest

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import (
    GNN_Actor_Critic,
)
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.test.test_action_masks import _masked_actions


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train/case_0012"
)


def _make_env(enabled, safety_margin=60.0):
    return AircraftScheduleEnv(
        {
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
            "device_lookahead_dispatch": bool(enabled),
            "device_lookahead_safety_margin": float(safety_margin),
        }
    )


def _commit_plane_to_future_mobile_job(env):
    env.reset()
    plane = env.planes["Plane_0_0"]
    target_site = env.sites["26"]
    job_code = "ZY02"
    needed_types = set(env._needed_mobile_types(job_code))
    assert needed_types
    for resource in list(target_site.resources.values()):
        if resource.type in needed_types:
            target_site.remove_resource(resource)

    origin_site = plane.site.code
    plane.choosed_job = job_code
    plane.start_transport(target_site, None)
    env.pending_actions[plane.code] = {
        "step_idx": env.steps,
        "agent_id": 0,
        "action": [0, env.site_code_list.index(target_site.code)],
        "start_time": env.total_time,
        "plane_id": plane.code,
        "site_id": target_site.code,
        "origin_site_code": origin_site,
        "target_site_code": target_site.code,
        "target_job_code": job_code,
        "device_ids": [],
    }
    env._get_obs()
    return plane, target_site, job_code


class DeviceLookaheadDispatchTest(unittest.TestCase):
    def test_default_off_preserves_waiting_only_request_contract(self):
        env = _make_env(False)
        try:
            _commit_plane_to_future_mobile_job(env)
            self.assertEqual(env.max_request_num, env.n_plane_agents + 1)
            self.assertEqual(env.request_list[1:], [])
        finally:
            env.close()

    def test_committed_future_job_exposes_deferrable_request_and_horizon(self):
        env = _make_env(True)
        try:
            plane, target_site, job_code = _commit_plane_to_future_mobile_job(env)
            self.assertEqual(env.max_request_num, 2 * env.n_plane_agents + 1)
            requests = [
                request
                for request in env.request_list[1:]
                if request["plane_id"] == plane.code
            ]
            self.assertEqual(len(requests), 1)
            request = requests[0]
            self.assertEqual(request["job_code"], job_code)
            self.assertEqual(request["site_code"], target_site.code)
            self.assertTrue(request["is_lookahead"])
            self.assertFalse(request["urgent"])
            self.assertGreater(request["lead_time"], 0.0)

            observation = env._get_obs()
            self.assertEqual(
                float(observation["request"].x[request["id"], 1]),
                -float(request["lead_time"]),
            )
            serving_device = next(
                device
                for device in env.device_list
                if env._device_can_dispatch(device, request)
            )
            device_idx = env.device_list.index(serving_device)
            valid, allow_noop = env._sequential_device_options(device_idx, set())
            self.assertIn(request["id"], [item["id"] for item in valid])
            self.assertTrue(allow_noop)
        finally:
            env.close()

    def test_deferred_lookahead_is_suppressed_until_time_advances(self):
        env = _make_env(True)
        try:
            _commit_plane_to_future_mobile_job(env)
            request = env.request_list[1]
            actions = np.zeros((env.n_agents, 2), dtype=np.int64)
            env._dispatch_device_actions(actions)
            key = env._request_identity(request)
            self.assertIn(key, env._deferred_lookahead_requests)

            env._refresh_request_pool()
            self.assertEqual(env.request_list[1:], [])
            env.total_time += 1.0
            env._refresh_request_pool()
            self.assertTrue(env.request_list[1:])
            self.assertEqual(
                env._request_identity(env.request_list[1]), key
            )
        finally:
            env.close()

    def test_hungarian_teacher_can_dispatch_before_plane_waits(self):
        env = _make_env(True, safety_margin=1e9)
        try:
            plane, _, _ = _commit_plane_to_future_mobile_job(env)
            self.assertTrue(plane.is_transporting)
            self.assertFalse(plane.is_waiting)
            request_id = int(env.request_list[1]["id"])
            actions = env.heuristic_device_actions()
            self.assertIn(
                request_id,
                actions[env.n_plane_agents :, 0].tolist(),
            )
            env._dispatch_device_actions(actions)
            records = list(env.pending_device_actions.values()) + list(
                env.device_trajectory_log
            )
            self.assertTrue(records)
            self.assertTrue(any(record.get("is_lookahead") for record in records))
        finally:
            env.close()

    def test_intent_ledger_attributes_dispatch_delay_to_four_timestamps(self):
        env = _make_env(True, safety_margin=1e9)
        try:
            _commit_plane_to_future_mobile_job(env)
            request = env.request_list[1]
            identity = env._request_identity(request)
            intent = env.resource_intent_ledger[identity]
            self.assertIn('first_visible_time', intent)
            self.assertIn('first_legal_time', intent)
            self.assertIn('first_compatible_idle_time', intent)
            env.total_time += 7.0
            actions = env.heuristic_device_actions()
            env._dispatch_device_actions(actions)
            intent = env.resource_intent_ledger[identity]
            self.assertEqual(intent['dispatch_time'], env.total_time)
            metrics = env.get_resource_lateness_metrics()
            self.assertGreaterEqual(metrics['visibility_to_legal_seconds'], 0.0)
            self.assertGreaterEqual(metrics['legal_to_idle_seconds'], 0.0)
            self.assertGreaterEqual(metrics['policy_defer_seconds'], 7.0)
            self.assertGreater(metrics['policy_defer_count'], 0)
        finally:
            env.close()

    def test_hungarian_teacher_does_not_arrive_excessively_early(self):
        env = _make_env(True)
        try:
            _, target_site, _ = _commit_plane_to_future_mobile_job(env)
            request = env.request_list[1]
            request_id = int(request["id"])
            actions = env.heuristic_device_actions()
            selected = [
                device
                for device_idx, device in enumerate(env.device_list)
                if actions[env.n_plane_agents + device_idx, 0] == request_id
            ]
            self.assertTrue(selected)
            for device in selected:
                distance = (
                    abs(float(device.site.pos[0]) - float(target_site.pos[0]))
                    + abs(float(device.site.pos[1]) - float(target_site.pos[1]))
                )
                travel_time = distance / max(float(device.velocity), 1.0)
                self.assertGreaterEqual(
                    travel_time + env.device_lookahead_safety_margin,
                    float(request["lead_time"]),
                )
        finally:
            env.close()

    def test_resource_actor_accepts_doubled_request_capacity(self):
        env = _make_env(True)
        try:
            _commit_plane_to_future_mobile_job(env)
            observation = env._get_obs()
            info = env._get_info()
            with (ROOT / "onpolicy/config/ac.yaml").open(
                "r", encoding="utf-8"
            ) as stream:
                ac_config = yaml.safe_load(stream)
            policy = GNN_Actor_Critic(
                **ac_config,
                max_plane_agents=env.n_plane_agents,
                max_device_agents=env.max_device_num,
            )
            data = {
                "graph": Batch.from_data_list([observation]),
                "hidden_states": torch.zeros(1, env.n_agents, 1, 64),
            }
            policy_info = {
                "active_agents": torch.as_tensor(
                    info["active_agents"][None, :], dtype=torch.bool
                ),
                "last_op_indices": torch.as_tensor(
                    info["last_op_indices"][None, :], dtype=torch.long
                ),
                "last_site_indices": torch.as_tensor(
                    info["last_site_indices"][None, :], dtype=torch.long
                ),
            }
            with torch.no_grad():
                _, actions, _, _ = policy(
                    data, policy_info, deterministic=True
                )
                log_probs, entropy = policy(
                    data,
                    policy_info,
                    chosen_op=actions[..., 0],
                    chosen_site=actions[..., 1],
                    eval_action=True,
                )
            self.assertTrue(torch.isfinite(log_probs).all())
            self.assertTrue(torch.isfinite(entropy))
            env._plan_device_actions(actions[0].cpu().numpy())
        finally:
            env.close()

    def test_resource_actor_replays_deferred_lookahead_noop(self):
        """The last compatible device may still defer a lookahead request."""
        env = _make_env(True)
        try:
            _commit_plane_to_future_mobile_job(env)
            observation = env._get_obs()
            info = env._get_info()
            with (ROOT / "onpolicy/config/ac.yaml").open(
                "r", encoding="utf-8"
            ) as stream:
                ac_config = yaml.safe_load(stream)
            policy = GNN_Actor_Critic(
                **ac_config,
                max_plane_agents=env.n_plane_agents,
                max_device_agents=env.max_device_num,
            )
            data = {
                "graph": Batch.from_data_list([observation]),
                "hidden_states": torch.zeros(1, env.n_agents, 1, 64),
            }
            policy_info = {
                "active_agents": torch.as_tensor(
                    info["active_agents"][None, :], dtype=torch.bool
                ),
                "last_op_indices": torch.as_tensor(
                    info["last_op_indices"][None, :], dtype=torch.long
                ),
                "last_site_indices": torch.as_tensor(
                    info["last_site_indices"][None, :], dtype=torch.long
                ),
            }
            with torch.no_grad():
                _, actions, _, _ = policy(
                    data, policy_info, deterministic=True
                )
                actions[:, env.n_plane_agents :, 0] = 0
                log_probs, entropy = policy(
                    data,
                    policy_info,
                    chosen_op=actions[..., 0],
                    chosen_site=actions[..., 1],
                    eval_action=True,
                )
            self.assertTrue(torch.isfinite(log_probs).all())
            self.assertTrue(torch.isfinite(entropy))
        finally:
            env.close()

    def test_hungarian_lookahead_completes_a_natural_rollout(self):
        env = _make_env(True)
        try:
            _, dones, info = env.reset()
            for _ in range(12000):
                actions = _masked_actions(env, info)
                labels = env.heuristic_device_actions()
                actions[env.n_plane_agents :] = labels[env.n_plane_agents :]
                _, _, dones, info = env.step(actions)
                if np.all(dones):
                    break
            self.assertTrue(np.all(dones))
            self.assertTrue(np.isfinite(env.total_time))
            self.assertFalse(env.cycle_terminated)
            self.assertTrue(
                any(
                    record.get("is_lookahead", False)
                    for record in env.device_trajectory_log
                )
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
