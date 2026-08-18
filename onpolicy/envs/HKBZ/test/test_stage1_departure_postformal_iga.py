import json
import tempfile
import unittest
from pathlib import Path

from onpolicy.scripts.train.run_stage1_departure_postformal_iga import (
    EXPECTED_SEMANTICS,
    discover_current_status,
    validate_final_evaluation,
    validate_iga_output,
)


class Stage1DeparturePostFormalIgaTest(unittest.TestCase):
    def test_final_validation_gate_requires_the_exact_epoch_and_case_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status_path = root / "results" / "formal_seed1" / "run1" / "run_status.json"
            status_path.parent.mkdir(parents=True)
            status = {
                "status": "completed",
                "started_at": "2026-08-17T01:00:00",
                "total_epochs": 8,
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            validation = root / "validation"
            validation.mkdir()
            evaluation_path = status_path.parent / "evaluations" / "epoch_8.json"
            evaluation_path.parent.mkdir()
            evaluation_path.write_text(json.dumps({
                "evaluation_label": "epoch_8",
                "eval_dataset_dir": str(validation),
                "summary": {
                    "eval_makespan": 9000.0,
                    "eval_case_count": 2,
                    "eval_completed_count": 2,
                    "eval_completion_rate": 1.0,
                },
                "cases": [{"case_id": "a"}, {"case_id": "b"}],
            }), encoding="utf-8")

            evidence = validate_final_evaluation(
                status_path,
                status,
                final_epoch=8,
                expected_cases=2,
                expected_validation_dir=validation,
            )
            self.assertEqual(evidence["case_count"], 2)
            self.assertEqual(evidence["final_eval_makespan"], 9000.0)

            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            payload["summary"]["eval_completed_count"] = 1
            evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                validate_final_evaluation(
                    status_path,
                    status,
                    final_epoch=8,
                    expected_cases=2,
                    expected_validation_dir=validation,
                )

    def test_status_discovery_rejects_a_run_older_than_the_launch_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "exp" / "run1" / "run_status.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "status": "completed",
                "started_at": "2026-08-16T01:00:00",
            }), encoding="utf-8")
            self.assertIsNone(discover_current_status(
                root,
                "exp",
                not_before_unix_time=1786986000.0,
            ))

    def test_iga_gate_requires_new_semantics_and_replay_verified_test60(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "test"
            dataset.mkdir()
            output = root / "iga180_test60.json"
            payload = {
                "status": "completed",
                "environment_semantics_version": EXPECTED_SEMANTICS,
                "dataset_test_dir": str(dataset),
                "case_count": 2,
                "methods": {
                    "IGA": {
                        "status": "completed",
                        "summary": {"completed_count": 2, "verified_count": 2},
                        "cases": [
                            {"case": "case_0001", "completion_verified": True},
                            {"case": "case_0002", "completion_verified": True},
                        ],
                    },
                },
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            evidence = validate_iga_output(output, dataset, expected_cases=2)
            self.assertEqual(evidence["verified_count"], 2)

            payload["environment_semantics_version"] = "legacy"
            output.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "wrong semantics"):
                validate_iga_output(output, dataset, expected_cases=2)


if __name__ == "__main__":
    unittest.main()
