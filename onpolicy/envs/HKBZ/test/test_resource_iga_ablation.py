from pathlib import Path
import unittest

import numpy as np

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.experiment.evaluate_resource_iga_ablation import (
    ARM_BACKENDS,
    GenomeLayout,
    PRIMARY_ARMS,
    TRANSPORTER_TYPE,
    _common_initial_populations,
    _iga_preferences,
    mixed_resource_actions,
)
from onpolicy.envs.HKBZ.test.test_device_lookahead_dispatch import (
    _commit_plane_to_future_mobile_job,
)


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train/case_0012"
)


def _make_env(lookahead=False):
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
            "device_lookahead_dispatch": bool(lookahead),
            "device_lookahead_safety_margin": 60.0,
        }
    )


def _create_r008_request(env):
    """Create one authoritative ordinary-device request without stepping time."""

    env.reset()
    plane = env.planes["Plane_0_0"]
    target_site = env.sites["26"]
    if plane.site is not target_site:
        if plane.site.is_occupied:
            plane.site.remove_plane()
        plane.site = target_site
        target_site.add_plane(plane)

    # The waiting operation must really require a mobile resource at this site.
    for resource in list(target_site.resources.values()):
        if resource.type == "R008":
            target_site.remove_resource(resource)
    plane.start_waiting("ZY02")
    plane.choosed_job = "ZY02"
    if target_site.code not in env.waiting_sites["ZY02"]:
        env.waiting_sites["ZY02"].append(target_site.code)
    env._get_obs()


class ResourceIGAAblationTest(unittest.TestCase):
    def test_three_primary_arms_are_the_requested_role_combinations(self):
        self.assertEqual(
            PRIMARY_ARMS,
            ("iga_all", "iga_r014", "iga_ordinary"),
        )
        self.assertEqual(
            ARM_BACKENDS["iga_all"],
            {"ordinary": "iga", "transporter": "iga"},
        )
        self.assertEqual(
            ARM_BACKENDS["iga_r014"],
            {"ordinary": "hungarian", "transporter": "iga"},
        )
        self.assertEqual(
            ARM_BACKENDS["iga_ordinary"],
            {"ordinary": "iga", "transporter": "hungarian"},
        )

    def test_genome_contains_only_devices_owned_by_each_iga_arm(self):
        env = _make_env()
        try:
            env.reset()
            all_layout = GenomeLayout.from_env(env, ARM_BACKENDS["iga_all"])
            r014_layout = GenomeLayout.from_env(env, ARM_BACKENDS["iga_r014"])
            ordinary_layout = GenomeLayout.from_env(
                env, ARM_BACKENDS["iga_ordinary"]
            )

            all_indices = set(all_layout.selected_device_indices)
            r014_indices = set(r014_layout.selected_device_indices)
            ordinary_indices = set(ordinary_layout.selected_device_indices)
            self.assertTrue(r014_indices)
            self.assertTrue(ordinary_indices)
            self.assertFalse(r014_indices & ordinary_indices)
            self.assertEqual(all_indices, r014_indices | ordinary_indices)
            self.assertTrue(
                all(
                    env.device_list[index].resource.type == TRANSPORTER_TYPE
                    for index in r014_indices
                )
            )
            self.assertTrue(
                all(
                    env.device_list[index].resource.type != TRANSPORTER_TYPE
                    for index in ordinary_indices
                )
            )
        finally:
            env.close()

    def test_initial_candidates_share_identical_overlapping_role_genes(self):
        env = _make_env()
        try:
            env.reset()
            layouts = {
                arm: GenomeLayout.from_env(env, ARM_BACKENDS[arm])
                for arm in PRIMARY_ARMS
            }
            populations = _common_initial_populations(layouts, 3, 20260811)
            for candidate in range(3):
                full = layouts["iga_all"].decode(
                    populations["iga_all"][candidate]
                )
                for arm in ("iga_r014", "iga_ordinary"):
                    partial = layouts[arm].decode(populations[arm][candidate])
                    np.testing.assert_array_equal(
                        partial.feature_weights, full.feature_weights
                    )
                    for device_idx, row in partial.row_by_device.items():
                        np.testing.assert_array_equal(
                            row, full.row_by_device[device_idx]
                        )
        finally:
            env.close()

    def test_all_hungarian_dispatch_is_exactly_the_environment_baseline(self):
        env = _make_env()
        try:
            _create_r008_request(env)
            expected = env.heuristic_device_actions()
            observed, decisions = mixed_resource_actions(
                env,
                ARM_BACKENDS["hungarian_all"],
                None,
                record=True,
            )
            np.testing.assert_array_equal(observed, expected)
            self.assertTrue(decisions)
            self.assertTrue(
                all(item["backend"] == "hungarian" for item in decisions)
            )
            self.assertTrue(env._plan_device_actions(observed))
        finally:
            env.close()

    def test_every_hybrid_arm_emits_a_legal_non_deadlocking_joint_action(self):
        for arm in PRIMARY_ARMS:
            with self.subTest(arm=arm):
                env = _make_env()
                try:
                    _create_r008_request(env)
                    layout = GenomeLayout.from_env(env, ARM_BACKENDS[arm])
                    chromosome = np.linspace(0.05, 0.95, layout.n_var)
                    decoded = layout.decode(chromosome)
                    actions, decisions = mixed_resource_actions(
                        env,
                        ARM_BACKENDS[arm],
                        decoded,
                        record=True,
                    )
                    plan = env._plan_device_actions(actions)
                    self.assertTrue(plan)
                    self.assertTrue(decisions)
                    self.assertGreater(
                        np.count_nonzero(
                            actions[env.n_plane_agents :, 0]
                        ),
                        0,
                    )
                finally:
                    env.close()

    def test_iga_does_not_assign_a_lookahead_device_excessively_early(self):
        env = _make_env(lookahead=True)
        try:
            _, target_site, _ = _commit_plane_to_future_mobile_job(env)
            request = env.request_list[1]
            layout = GenomeLayout.from_env(env, ARM_BACKENDS["iga_all"])
            decoded = layout.decode(np.full(layout.n_var, 0.5))
            preferences, _ = _iga_preferences(env, decoded)
            selected = [
                device
                for device in env.device_list
                if preferences.get(device.code) == int(request["id"])
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


if __name__ == "__main__":
    unittest.main()
