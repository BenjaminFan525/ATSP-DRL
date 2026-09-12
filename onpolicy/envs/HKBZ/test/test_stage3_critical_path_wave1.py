"""Focused contracts for the Stage3 critical-path/rendezvous Wave-1."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.experiment.resource_wait_metrics import (
    summarize_aircraft_resource_wait,
)
from onpolicy.scripts.train.prepare_stage3_joint_finetune import build_command


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train/case_0012"
)


def _config(*, horizon: int, frontier: int) -> dict:
    return {
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
        "device_lookahead_dispatch": True,
        "device_lookahead_safety_margin": 60.0,
        "device_deadline_aware_dispatch": True,
        "device_future_intent_horizon": horizon,
        "device_future_intent_mode": "bounded_frontier",
        "device_frontier_max_requests": frontier,
        "device_request_capacity_per_plane": 5,
        "resource_release_aware_eta": True,
        "device_lookahead_reservation_mode": "soft",
        "device_reservation_grace_seconds": 300.0,
        "device_departure_lookahead": True,
    }


def _commit_zy02(env: AircraftScheduleEnv) -> None:
    env.reset()
    plane = env.planes["Plane_0_0"]
    target_site = env.sites["26"]
    # Make every mobile type absent from the forecast stand so the test
    # observes the frontier rather than the case's initial resource placement.
    for resource in list(target_site.resources.values()):
        if resource.type in env.mobile_devices:
            target_site.remove_resource(resource)
    origin_site = plane.site.code
    plane.choosed_job = "ZY02"
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
        "target_job_code": "ZY02",
        "device_ids": [],
    }
    env._refresh_request_pool()


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class Stage3CriticalPathWave1Test(unittest.TestCase):
    def test_fixed_request_capacity_allows_shared_evaluator_horizon_switch(self):
        env = AircraftScheduleEnv(_config(horizon=1, frontier=2))
        try:
            env.reset()
            self.assertEqual(env.max_request_num, 5 * 24 + 1)
            observed = env.set_resource_planning_config({
                **env.resource_planning_config(),
                "device_future_intent_horizon": 3,
                "device_frontier_max_requests": 4,
            })
            self.assertEqual(observed["device_future_intent_horizon"], 3)
            self.assertEqual(observed["device_frontier_max_requests"], 4)
            self.assertEqual(env.max_request_num, 5 * 24 + 1)
        finally:
            env.close()

    def test_horizon_three_exposes_a_larger_soft_forecast_frontier(self):
        counts = {}
        depths = {}
        for horizon, frontier in ((1, 2), (3, 4)):
            env = AircraftScheduleEnv(_config(
                horizon=horizon, frontier=frontier
            ))
            try:
                _commit_zy02(env)
                forecasts = [
                    request for request in env.request_list[1:]
                    if str(request.get("request_kind", "")).startswith(
                        "bounded_mobile_frontier"
                    )
                ]
                counts[horizon] = len(forecasts)
                depths[horizon] = {
                    int(request.get("dependency_depth", 0))
                    for request in forecasts
                }
                self.assertTrue(all(
                    request.get("is_lookahead") for request in forecasts
                ))
            finally:
                env.close()
        self.assertGreater(counts[3], counts[1])
        self.assertLessEqual(counts[3], 4)
        self.assertTrue(depths[3].issubset({1, 2, 3}))

    def test_critical_wait_metrics_weight_only_cmax_relevant_delay(self):
        result = summarize_aircraft_resource_wait(
            [{
                "plane_id": "P0",
                "target_job_code": "J",
                "waiting_time": 100.0,
                "start_time": 0.0,
                "end_time": 100.0,
            }, {
                "plane_id": "P1",
                "target_job_code": "J",
                "waiting_time": 40.0,
                "start_time": 160.0,
                "end_time": 200.0,
            }],
            [],
            {"J": ["R001"]},
            aircraft_count=2,
            episode_cmax=200.0,
            criticality_scale_seconds=50.0,
            criticality_min_weight=0.0,
            include_events=True,
        )
        self.assertEqual(result["schema_version"], 3)
        self.assertLess(result["critical_wait_seconds"], 140.0)
        self.assertGreater(result["critical_wait_seconds"], 40.0)
        critical = max(
            result["events"], key=lambda item: item["criticality_weight"]
        )
        self.assertEqual(critical["plane_id"], "P1")
        self.assertAlmostEqual(critical["criticality_weight"], 1.0)

    def test_critical_path_v2_slack_gates_identical_plane_wait(self):
        env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
        env.total_time = 200.0
        env.resource_slack_criticality_seconds = 50.0
        env.resource_slack_min_weight = 0.0
        env.job_code_list = []
        env.flights_data = [{}, {}]
        env.device_trajectory_log = []
        env.trajectory_log = [{
            'step_idx': 0,
            'agent_id': 0,
            'plane_id': 'P0',
            'target_job_code': 'J0',
            'start_time': 0.0,
            'end_time': 100.0,
            'duration': 100.0,
            'waiting_time': 40.0,
        }, {
            'step_idx': 1,
            'agent_id': 1,
            'plane_id': 'P1',
            'target_job_code': 'J1',
            'start_time': 100.0,
            'end_time': 200.0,
            'duration': 100.0,
            'waiting_time': 40.0,
        }]
        payload = env.get_role_event_credit_weights('critical_path_v2')
        early = payload['weights'][(0, 0)]
        critical = payload['weights'][(1, 1)]
        self.assertEqual(payload['schema_version'], 2)
        self.assertEqual(payload['credit_mode'], 'critical_path_v2')
        self.assertLess(early['criticality_weight'], 0.14)
        self.assertAlmostEqual(critical['criticality_weight'], 1.0)
        self.assertLess(
            early['critical_wait_seconds'],
            critical['critical_wait_seconds'],
        )
        self.assertIn('critical_avoidable_lateness_seconds', critical)
        self.assertGreater(critical['weight'], early['weight'])

    def test_causal_arm_commands_isolate_the_two_hypotheses(self):
        source = ROOT / (
            "onpolicy/scripts/results/HKBZ/simple/gnn_mappo/"
            "stage2_planning_wave4_20260822_r2_planning_wave_"
            "B2_wait_constraint_seed1/run1/models/checkpoint_Best.pt"
        )
        python = ROOT.parent / "conda/envs/maia-hkbz-cu124-20260903/bin/python3.11"
        commands = {}
        for method in ("C0", "C1", "C2", "C3"):
            commands[method] = build_command(SimpleNamespace(
                profile="critical_wave1",
                source_checkpoint=source,
                python=python,
                experiment_name=f"test_{method}",
                seed=1,
                method=method,
                max_graphs_per_forward=1000,
            ))

        for method, command in commands.items():
            self.assertEqual(_value(command, "--max_graphs_per_forward"), "1000")
            self.assertEqual(
                _value(command, "--device_request_capacity_per_plane"), "5"
            )
            self.assertEqual(
                _value(command, "--stage3_handoff_mode"),
                "critical_path_wave1",
            )
            expected_horizon = "3" if method in {"C2", "C3"} else "1"
            expected_frontier = "4" if method in {"C2", "C3"} else "2"
            self.assertEqual(
                _value(command, "--device_future_intent_horizon"),
                expected_horizon,
            )
            self.assertEqual(
                _value(command, "--device_frontier_max_requests"),
                expected_frontier,
            )
        for method in ("C1", "C3"):
            command = commands[method]
            self.assertEqual(
                _value(command, "--hindsight_reward_mode"),
                "team_time_resource_potential",
            )
            self.assertEqual(
                _value(command, "--resource_wait_constraint_target"), "0.0"
            )
            self.assertEqual(
                _value(command, "--resource_wait_dual_lr"), "0.0"
            )
        for method in ("C0", "C2"):
            self.assertEqual(
                _value(commands[method], "--hindsight_reward_mode"),
                "team_time",
            )


if __name__ == "__main__":
    unittest.main()
