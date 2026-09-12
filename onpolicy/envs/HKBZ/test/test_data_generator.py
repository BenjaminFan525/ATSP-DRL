import copy
import tempfile
import unittest
from pathlib import Path

from torch_geometric.data import Batch

from onpolicy.envs.HKBZ.data_generator import (
    MAX_PLANES,
    MOBILE_DEVICE_BUDGET,
    PROFILES,
    SERVICE_STANDS,
    AirportScenarioGenerator,
    _case_statistics,
    _sha256,
    _write_case,
    build_benchmark_dataset,
)
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.experiment.eval_common import load_case_metadata


class AirportScenarioGeneratorTest(unittest.TestCase):
    def _generate(self, profile_name, seed, case_id="case"):
        return AirportScenarioGenerator(
            profile=PROFILES[profile_name],
            seed=seed,
            split="train",
            case_id=case_id,
        ).generate()

    def test_same_seed_is_exactly_reproducible(self):
        first, first_metadata = self._generate("balanced", 12345)
        second, second_metadata = self._generate("balanced", 12345)
        self.assertEqual(_sha256(first), _sha256(second))
        self.assertEqual(first_metadata, second_metadata)

    def test_profiles_create_real_device_demand(self):
        for index, profile_name in enumerate(PROFILES):
            with self.subTest(profile=profile_name):
                case, _ = self._generate(profile_name, 8000 + index)
                stats = _case_statistics(case)
                self.assertLessEqual(stats["num_planes"], MAX_PLANES)
                self.assertEqual(stats["num_mobile_resources"], MOBILE_DEVICE_BUDGET)
                self.assertGreaterEqual(
                    stats["ordinary_mobile_required_pair_fraction"], 0.15
                )
                self.assertLessEqual(stats["fixed_coverage_ratio"], 0.85)

    def test_environment_and_pyg_batch_accept_variable_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            observations = []
            for index, profile_name in enumerate(("light", "stress_joint"), start=1):
                case, metadata = self._generate(
                    profile_name, 9100 + index, case_id=f"integration_{index}"
                )
                case_dir = root / f"case_{index:04d}"
                _write_case(case_dir, case, metadata)
                config = {
                    "jobs_path": str(case_dir / "job.json"),
                    "fixed_res_path": str(case_dir / "fixed_resources.json"),
                    "mobile_res_path": str(case_dir / "mobile_resources.json"),
                    "sites_path": str(case_dir / "sites.json"),
                    "flights_path": str(case_dir / "flights.json"),
                    "n_agents": MAX_PLANES,
                    "max_device_num": 80,
                    "resource_policy": "drl",
                    "use_domain_rand": False,
                    "seed": 42,
                }
                env = AircraftScheduleEnv(copy.deepcopy(config))
                observation, dones, info = env.reset()
                self.assertEqual(observation["site"].x.shape[0], SERVICE_STANDS + 4)
                self.assertEqual(observation["device"].x.shape[0], MOBILE_DEVICE_BUDGET)
                self.assertEqual(observation.site_mask_matrix.shape[1], SERVICE_STANDS + 4)
                self.assertEqual(len(dones), MAX_PLANES + 80)
                self.assertEqual(len(info["agent_types"]), MAX_PLANES + 80)
                observations.append(observation)
                env.close()

            batch = Batch.from_data_list(observations)
            self.assertEqual(batch.num_graphs, 2)

    def test_split_builder_has_no_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "dataset"
            report = build_benchmark_dataset(
                output,
                train_cases=24,
                validation_cases=12,
                test_cases=16,
                seed=20260715,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(
                report["cross_split_exact_overlaps"],
                {
                    "train__validation": 0,
                    "train__test": 0,
                    "validation__test": 0,
                },
            )
            for split, prefix, expected_count in (
                ("train", "train_", 24),
                ("validation", "validation_", 12),
                ("test", "test_", 16),
            ):
                metadata = load_case_metadata(str(output / split))
                self.assertEqual(len(metadata), expected_count)
                self.assertTrue(
                    all(item["case_id"].startswith(prefix) for item in metadata.values())
                )
                self.assertTrue(
                    all(item["profile"] != "unknown" for item in metadata.values())
                )


if __name__ == "__main__":
    unittest.main()
