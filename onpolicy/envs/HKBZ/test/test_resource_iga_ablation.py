from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.experiment.evaluate_resource_iga_ablation import (
    ARM_BACKENDS,
    FrozenPlaneEvaluator,
    GenomeLayout,
    PRIMARY_ARMS,
    TRANSPORTER_TYPE,
    _common_initial_populations,
    _iga_preferences,
    _resolve_source_command,
    mixed_resource_actions,
    normalize_resource_lookahead_contract,
)
from onpolicy.envs.HKBZ.experiment.generate_stage2_resource_iga_labels import (
    SEARCH_CONTRACT_VERSION,
    _AnytimeCaseGA,
    _initial_population,
    _load_warm_start,
    _load_verified_warm_episode,
    _teacher_contract,
    _teacher_is_reusable,
)
from onpolicy.envs.HKBZ.test.test_device_lookahead_dispatch import (
    _commit_plane_to_future_mobile_job,
)


from onpolicy.envs.HKBZ.test.regression_fixtures import case_dir

ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = case_dir('train_case_0012')

MATCHED_PLANNING_CONTRACT = {
    "device_lookahead_dispatch": True,
    "device_lookahead_safety_margin": 60.0,
    "device_deadline_aware_dispatch": True,
    "device_future_intent_horizon": 1,
    "device_future_intent_mode": "bounded_frontier",
    "device_frontier_max_requests": 2,
    "resource_release_aware_eta": True,
    "device_lookahead_reservation_mode": "hard",
    "device_reservation_grace_seconds": 300.0,
    "device_departure_lookahead": True,
}


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
    def test_matched_planning_contract_is_injected_without_defaults(self):
        evaluator = FrozenPlaneEvaluator(
            None,
            None,
            max_steps=4000,
            resource_lookahead_contract=MATCHED_PLANNING_CONTRACT,
        )
        config = evaluator._env_config(CASE_DIR)
        self.assertEqual(
            evaluator.resource_lookahead_contract,
            MATCHED_PLANNING_CONTRACT,
        )
        for key, value in MATCHED_PLANNING_CONTRACT.items():
            self.assertEqual(config[key], value, key)

    def test_matched_planning_contract_must_be_complete(self):
        incomplete = dict(MATCHED_PLANNING_CONTRACT)
        incomplete.pop("device_departure_lookahead")
        with self.assertRaisesRegex(ValueError, "must be complete"):
            normalize_resource_lookahead_contract(incomplete)

    def test_hard_teacher_cannot_be_reused_as_soft_warm_start(self):
        hard = dict(MATCHED_PLANNING_CONTRACT)
        soft = {
            **MATCHED_PLANNING_CONTRACT,
            "device_lookahead_reservation_mode": "soft",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_dir = root / "teachers"
            teacher_dir.mkdir()
            path = teacher_dir / "case_0001.json"
            contract = _teacher_contract(
                case="case_0001",
                case_sha="case-sha",
                checkpoint_sha="checkpoint-sha",
                command_sha="command-sha",
                command_key="command-key",
                cumulative_budget=180.0,
                resource_lookahead_contract=hard,
            )
            payload = {
                **contract,
                "completed": True,
                "makespan": 100.0,
                "completion": {"completed": True},
                "decision_trace": [{"step": 0}],
                "intrinsic_ready_time_labels": [{
                    "observation_time": 0.0,
                    "intrinsic_ready_time": 1.0,
                    "plane_id": "P1",
                    "job_code": "J1",
                    "site_code": "S1",
                }],
                "intrinsic_ready_time_label_coverage": {
                    "labeled_request_count": 1,
                },
                "search": {
                    "nominal_cumulative_budget_seconds": 180.0,
                    "chromosome": [0.5],
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertTrue(_teacher_is_reusable(path, contract))
            soft_contract = {
                **contract,
                "resource_lookahead_contract": soft,
            }
            self.assertFalse(_teacher_is_reusable(path, soft_contract))
            with self.assertRaisesRegex(ValueError, "Incompatible warm-start"):
                _load_warm_start(
                    root,
                    "case_0001",
                    checkpoint_sha="checkpoint-sha",
                    command_sha="command-sha",
                    command_key="command-key",
                    case_sha="case-sha",
                    expected_budget=180.0,
                    resource_lookahead_contract=soft,
                )

    def test_new_multi_command_manifest_resolves_the_exact_source_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "commands.json"
            path.write_text(
                json.dumps(
                    {
                        "commands": {
                            "seed1": {"argv": ["python", "train.py", "--seed", "1"]},
                            "seed3": {"shell": "python train.py --seed 3"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _resolve_source_command(path, "seed1"),
                ["python", "train.py", "--seed", "1"],
            )
            self.assertEqual(
                _resolve_source_command(path, "seed3"),
                ["python", "train.py", "--seed", "3"],
            )
            with self.assertRaisesRegex(ValueError, "No command key"):
                _resolve_source_command(path, "missing")

    def test_stage2_population_keeps_verified_warm_incumbent(self):
        env = _make_env()
        try:
            env.reset()
            layout = GenomeLayout.from_env(env, ARM_BACKENDS["iga_all"])
            warm = np.linspace(0.0, 1.0, layout.n_var)
            _, population = _initial_population(layout, 3, 20260818, warm)
            np.testing.assert_array_equal(population[0], warm)
            np.testing.assert_array_equal(
                population[1, -4:],
                np.asarray([1.0, 1.0, 0.5, 1.0]),
            )
        finally:
            env.close()

    def test_case_parallel_ga_preserves_incumbent_mid_generation(self):
        layout = GenomeLayout(
            selected_device_indices=(0,),
            n_planes=2,
            site_codes=("s0",),
            job_codes=("j0",),
        )
        warm = np.linspace(0.0, 1.0, layout.n_var)
        state = _AnytimeCaseGA(
            layout,
            population=3,
            max_generations=2,
            seed=7,
            warm_chromosome=warm,
            warm_start={"source_makespan": 20.0},
        )
        self.assertEqual(state.cursor, 1)
        self.assertEqual(state.evaluated_candidates, 0)
        np.testing.assert_array_equal(state.best_chromosome, warm)
        state.observe({"completed": True, "makespan": 18.0, "error": None})
        self.assertEqual(state.cursor, 2)
        self.assertEqual(state.best_objective, 18.0)
        record = state.search_record(
            optimization_wall_seconds=5.0,
            time_budget_seconds=10.0,
            cumulative_budget_seconds=1800.0,
            batch_wall_seconds=[5.0],
            parallel_case_count=72,
        )
        self.assertEqual(
            record["search_contract_version"], SEARCH_CONTRACT_VERSION
        )
        self.assertEqual(record["evaluated_candidates"], 1)
        self.assertEqual(record["inherited_verified_candidates"], 1)
        self.assertEqual(record["partial_generation_evaluations"], 1)
        self.assertEqual(record["parallel_axis"], "independent_cases")

    def test_trace_replay_adjustment_keeps_a_verified_improvement(self):
        layout = GenomeLayout(
            selected_device_indices=(0,),
            n_planes=2,
            site_codes=("s0",),
            job_codes=("j0",),
        )
        warm = np.linspace(0.0, 1.0, layout.n_var)
        state = _AnytimeCaseGA(
            layout,
            population=3,
            max_generations=2,
            seed=7,
            warm_chromosome=warm,
            warm_start={"source_makespan": 20.0},
        )
        candidate = state.candidate.copy()
        state.observe({"completed": True, "makespan": 18.0, "error": None})

        source = state.reconcile_verified_replay(
            {"completed": True, "makespan": 19.0, "error": None}
        )

        self.assertEqual(source, "replayed_search_incumbent")
        self.assertEqual(state.best_objective, 19.0)
        np.testing.assert_array_equal(state.best_chromosome, candidate)
        self.assertEqual(
            state.replay_verification["status"],
            "accepted_replay_adjustment",
        )

    def test_trace_replay_regression_restores_verified_warm_teacher(self):
        layout = GenomeLayout(
            selected_device_indices=(0,),
            n_planes=2,
            site_codes=("s0",),
            job_codes=("j0",),
        )
        warm = np.linspace(0.0, 1.0, layout.n_var)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "teacher.json"
            payload = {
                "completion_verified": True,
                "completed": True,
                "makespan": 20.0,
                "wall_seconds": 1.0,
                "error": None,
                "completion": {"completed": True},
                "decision_trace": [{"step": 0}],
                "plane_trajectory": [],
                "resource_trajectory": [],
                "resource_decision_log": [],
                "search": {"chromosome": warm.tolist()},
            }
            source.write_text(json.dumps(payload), encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            state = _AnytimeCaseGA(
                layout,
                population=3,
                max_generations=2,
                seed=7,
                warm_chromosome=warm,
                warm_start={
                    "source_makespan": 20.0,
                    "path": str(source),
                    "sha256": digest,
                },
            )
            state.observe(
                {"completed": True, "makespan": 18.0, "error": None}
            )

            selected = state.reconcile_verified_replay(
                {"completed": True, "makespan": 21.0, "error": None}
            )
            episode = _load_verified_warm_episode(state)

            self.assertEqual(selected, "verified_warm_start")
            self.assertEqual(state.best_objective, 20.0)
            np.testing.assert_array_equal(state.best_chromosome, warm)
            self.assertEqual(episode["makespan"], 20.0)
            self.assertTrue(episode["reused_verified_warm_start"])
            self.assertNotIn("search", episode)

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
