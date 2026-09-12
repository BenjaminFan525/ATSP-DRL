import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


from onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels import (
    BACKENDS,
    SEARCH_CONTRACT_VERSION,
    _contract,
    _planning_contract,
    _search_case,
)
from onpolicy.scripts.train.run_stage3_joint_iga_pipeline import (
    _generator_command,
    validate_nested_continuation,
)


MATCHED_H3F4_PLANNING = _planning_contract(
    lookahead_margin=60.0,
    future_intent_horizon=3,
    future_intent_mode="bounded_frontier",
    frontier_max_requests=4,
    request_capacity_per_plane=5,
    release_aware_eta=True,
    reservation_mode="soft",
    reservation_grace_seconds=300.0,
    slack_forecast_seconds=0.0,
)
MATCHED_H3F4_KWARGS = {
    "future_intent_horizon": 3,
    "future_intent_mode": "bounded_frontier",
    "frontier_max_requests": 4,
    "request_capacity_per_plane": 5,
    "release_aware_eta": True,
    "reservation_mode": "soft",
    "reservation_grace_seconds": 300.0,
    "slack_forecast_seconds": 0.0,
}
from onpolicy.scripts.train.stage3_encoder_wave1_autopilot import (
    JOINT_IGA_RUNNER,
    _launch_joint_iga,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class Stage3JointIgaPipelineTest(unittest.TestCase):
    def _phase(
        self,
        output: Path,
        dataset: Path,
        *,
        budget: float,
        makespan: float,
        warm: bool,
    ) -> None:
        case = "case_0001"
        case_sha = "fixture-case-sha"
        payload = {
            **_contract(
                case,
                case_sha,
                budget,
                60.0,
                **MATCHED_H3F4_KWARGS,
            ),
            "completed": True,
            "completion": {"completed": True},
            "decision_trace": [{"step": 0}],
            "makespan": makespan,
            "search": {
                "search_contract_version": SEARCH_CONTRACT_VERSION,
                "chromosome": [0.5],
                "configured_additional_budget_seconds": (
                    budget - 180.0 if warm else budget
                ),
                "optimization_wall_seconds": (
                    budget - 180.0 if warm else budget
                ),
                "max_generations": 100000,
                "completed_generations": 1,
            },
        }
        if warm:
            payload["search"]["warm_start"] = {
                "kind": "verified_stage3_nested_incumbent",
                "source_nominal_budget_seconds": 180.0,
            }
        _write(output / "teachers" / f"{case}.json", payload)
        _write(output / "cases" / f"{case}.json", {"makespan": makespan})
        _write(
            output / "summary.json",
            {
                "status": "completed",
                "teacher_scope": "stage3_full_joint_policy",
                "teacher_method": "joint_iga_all",
                "environment_semantics_version": payload[
                    "environment_semantics_version"
                ],
                "dataset_dir": str(dataset.resolve()),
                "expected_case_count": 1,
                "completed_case_count": 1,
                "missing_cases": [],
                "nominal_cumulative_budget_seconds": budget,
                "planning_contract": MATCHED_H3F4_PLANNING,
                "makespan_mean": makespan,
                "makespan_median": makespan,
            },
        )

    def test_validates_full_joint_nested_non_regressing_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            _write(
                dataset / "case_0001" / "metadata.json",
                {"fingerprints": {"case_sha256": "fixture-case-sha"}},
            )
            self._phase(root / "iga180", dataset, budget=180.0, makespan=100.0, warm=False)
            self._phase(root / "iga1800", dataset, budget=1800.0, makespan=90.0, warm=True)
            result = validate_nested_continuation(
                root / "iga180",
                root / "iga1800",
                dataset,
                ["case_0001"],
                iga180_budget=180.0,
                iga1800_budget=1800.0,
                lookahead_margin=60.0,
                planning_contract=MATCHED_H3F4_PLANNING,
            )
            self.assertEqual(result["nested_regression_count"], 0)
            self.assertEqual(result["iga1800"]["case_count"], 1)

    def test_rejects_non_iga_mobile_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            _write(
                dataset / "case_0001" / "metadata.json",
                {"fingerprints": {"case_sha256": "fixture-case-sha"}},
            )
            self._phase(root / "iga180", dataset, budget=180.0, makespan=100.0, warm=False)
            self._phase(root / "iga1800", dataset, budget=1800.0, makespan=90.0, warm=True)
            teacher = root / "iga1800" / "teachers" / "case_0001.json"
            payload = json.loads(teacher.read_text(encoding="utf-8"))
            payload["backends"] = {**BACKENDS, "transporter": "hungarian"}
            _write(teacher, payload)
            with self.assertRaisesRegex(ValueError, "Unverified|every mobile role"):
                validate_nested_continuation(
                    root / "iga180",
                    root / "iga1800",
                    dataset,
                    ["case_0001"],
                    iga180_budget=180.0,
                    iga1800_budget=1800.0,
                    lookahead_margin=60.0,
                    planning_contract=MATCHED_H3F4_PLANNING,
                )

    def test_iga1800_command_is_nested_and_adds_only_1620_seconds(self):
        args = SimpleNamespace(
            python=Path("/python"),
            dataset_dir=Path("/dataset"),
            workers=60,
            population=20,
            max_generations=100000,
            max_steps=4000,
            max_plane_agents=24,
            max_device_num=80,
            device_lookahead_safety_margin=60.0,
            device_future_intent_horizon=3,
            device_future_intent_mode="bounded_frontier",
            device_frontier_max_requests=4,
            device_request_capacity_per_plane=5,
            resource_release_aware_eta=True,
            device_lookahead_reservation_mode="soft",
            device_reservation_grace_seconds=300.0,
            resource_slack_forecast_seconds=0.0,
            seed=20260824,
            max_cases=0,
            case_offset=0,
            iga180_budget_seconds=180.0,
            plane_warm_dir=None,
            resource_warm_dir=None,
        )
        command = _generator_command(
            args,
            output_dir=Path("/output/iga1800"),
            time_budget=1620.0,
            cumulative_budget=1800.0,
            warm_start_dir=Path("/output/iga180"),
        )
        self.assertEqual(command[command.index("--time-budget-seconds") + 1], "1620.0")
        self.assertEqual(
            command[command.index("--cumulative-budget-seconds") + 1], "1800.0"
        )
        self.assertEqual(
            command[command.index("--warm-start-dir") + 1], "/output/iga180"
        )
        self.assertEqual(
            command[command.index("--device-future-intent-horizon") + 1], "3"
        )
        self.assertEqual(
            command[command.index("--device-frontier-max-requests") + 1], "4"
        )
        self.assertIn("--resource-release-aware-eta", command)
        self.assertEqual(
            command[command.index("--device-lookahead-reservation-mode") + 1],
            "soft",
        )

    @mock.patch(
        "onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels._env_config",
        return_value={},
    )
    @mock.patch(
        "onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels._build_layout"
    )
    @mock.patch(
        "onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels._load_joint_warm"
    )
    @mock.patch(
        "onpolicy.envs.HKBZ.experiment.generate_stage3_joint_iga_labels.evaluate_joint"
    )
    def test_nested_incumbent_counts_as_resolved_population_member(
        self, evaluate_mock, warm_mock, layout_mock, _config_mock
    ):
        class Layout:
            n_var = 4

            @staticmethod
            def record():
                return {"n_var": 4}

        layout_mock.return_value = Layout()
        warm_mock.return_value = (
            np.full(4, 0.5),
            100.0,
            {"kind": "verified_stage3_nested_incumbent"},
        )

        def evaluate(_config, chromosome, *, record_trace, **_kwargs):
            objective = 90.0 + float(chromosome[0])
            result = {
                "completed": True,
                "makespan": objective,
                "wall_seconds": 0.0,
                "error": None,
                "deadline_interrupted": False,
                "completion": {"completed": True},
            }
            if record_trace:
                result.update(
                    decision_trace=[{"step": 0}],
                    plane_trajectory=[],
                    resource_trajectory=[],
                    resource_decision_log=[],
                    aircraft_resource_wait={"total_wait_seconds": 0.0},
                )
            return result

        evaluate_mock.side_effect = evaluate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case_0001"
            _write(
                case / "metadata.json",
                {
                    "case_id": "fixture",
                    "fingerprints": {"case_sha256": "fixture-case-sha"},
                },
            )
            result = _search_case(
                {
                    "case_path": str(case),
                    "dataset_dir": str(root),
                    "teacher_path": str(root / "teachers" / "case_0001.json"),
                    "result_path": str(root / "cases" / "case_0001.json"),
                    "expected": _contract(
                        "case_0001",
                        "fixture-case-sha",
                        1800.0,
                        60.0,
                        **MATCHED_H3F4_KWARGS,
                    ),
                    "population": 3,
                    "max_generations": 2,
                    "time_budget": 1000.0,
                    "cumulative_budget": 1800.0,
                    "warm_start_dir": str(root / "iga180"),
                    "warm_start_budget": 180.0,
                    "plane_warm_dir": None,
                    "resource_warm_dir": None,
                    "seed": 7,
                    "max_steps": 10,
                    "max_plane_agents": 2,
                    "max_device_num": 2,
                    "lookahead_margin": 60.0,
                    "future_intent_horizon": 3,
                    "future_intent_mode": "bounded_frontier",
                    "frontier_max_requests": 4,
                    "request_capacity_per_plane": 5,
                    "release_aware_eta": True,
                    "reservation_mode": "soft",
                    "reservation_grace_seconds": 300.0,
                    "slack_forecast_seconds": 0.0,
                }
            )
            teacher = json.loads(
                (root / "teachers" / "case_0001.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["evaluated_candidates"], 5)
        self.assertEqual(teacher["search"]["completed_generations"], 2)
        self.assertEqual(teacher["search"]["candidate_history"][0]["evaluated"], 2)
        self.assertEqual(teacher["search"]["candidate_history"][0]["resolved"], 3)

    @mock.patch(
        "onpolicy.scripts.train.stage3_encoder_wave1_autopilot._service_load_state",
        return_value="not-found",
    )
    @mock.patch(
        "onpolicy.scripts.train.stage3_encoder_wave1_autopilot._service_state",
        return_value="unknown",
    )
    @mock.patch(
        "onpolicy.scripts.train.stage3_encoder_wave1_autopilot.subprocess.run"
    )
    def test_autopilot_launches_independent_full_cpu_service(
        self, run_mock, _state_mock, _load_mock
    ):
        run_mock.return_value = subprocess.CompletedProcess([], 0, stdout="ok")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            result = _launch_joint_iga(
                unit="fixture-joint-iga",
                dataset=dataset,
                output_root=root / "output",
                python=Path("/python"),
                workers=60,
                cpu_set="0-143",
            )
        command = run_mock.call_args.args[0]
        self.assertEqual(result["status"], "launched")
        self.assertIn("--property=AllowedCPUs=0-143", command)
        self.assertIn("--property=CPUAffinity=0-143", command)
        self.assertIn(str(JOINT_IGA_RUNNER), command)
        self.assertIn("1800", command)


if __name__ == "__main__":
    unittest.main()
