import json
import tempfile
import unittest
from pathlib import Path


from onpolicy.scripts.train.stage3_role_credit_autopilot import (
    METHODS,
    analyze_wave1,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation(score, case_suffix=""):
    cases = [
        {
            "case_key": f"tune/case_{index:04d}{case_suffix}",
            "case_sha256": f"sha-{index:04d}{case_suffix}",
            "resource_wait_seconds": 1000.0 + index,
        }
        for index in range(60)
    ]
    return {
        "cases": cases,
        "summary": {
            "eval_case_count": 60,
            "eval_completion_rate": 1.0,
            "eval_cycle_count": 0,
            "eval_timeout_count": 0,
            "eval_selection_score": score,
            "eval_raw_makespan": score - 100.0,
            "eval_iid_makespan": score - 200.0,
            "eval_composite_makespan": score - 100.0,
            "eval_tail_makespan": score + 300.0,
            "eval_distribution_ood_stress_makespan": score + 100.0,
            "eval_distribution_ood_scale_makespan": score - 300.0,
        },
    }


def _status(*, zero_updates=0):
    return {
        "status": "completed",
        "training_stage": "joint_finetune",
        "phase": "joint_finetune_completed",
        "canary_rejected": False,
        "eval_completion_rate": 1.0,
        "eval_cycle_count": 0,
        "eval_timeout_count": 0,
        "actor_update_health": {
            "step_completion_rate": 1.0,
            "zero_update_shards": zero_updates,
            "post_update_old_policy_kl_max": 0.001,
        },
    }


class Stage3RoleCreditAutopilotTest(unittest.TestCase):
    def _fixture(self, root, scores, *, zero_update_method=None):
        suite = root / "suite"
        results = root / "results"
        for method in METHODS:
            experiment = f"fixture_wave1_{method}_seed1"
            manifest = {
                "profile": "wave1",
                "causal_arm": method,
                "source_stage2": {"sha256": "common-stage2-digest"},
                "command": [
                    "python", "train.py",
                    "--experiment_name", experiment,
                    "--num_episodes", "2",
                ],
            }
            _write_json(suite / "commands" / f"wave1_{method}.json", manifest)
            run = results / experiment / "run1"
            _write_json(
                run / "run_status.json",
                _status(zero_updates=int(method == zero_update_method)),
            )
            _write_json(run / "evaluations" / "pre_ppo.json", _evaluation(100.0))
            for epoch, score in enumerate(scores[method], start=1):
                _write_json(
                    run / "evaluations" / f"epoch_{epoch}.json",
                    _evaluation(score),
                )
        return suite, results

    def test_selects_best_eligible_full_valid60_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite, results = self._fixture(
                root,
                {
                    "B0": (99.0, 98.0),
                    "B1": (97.0, 96.0),
                    "B2": (90.0, 89.0),
                    "B3": (96.0, 97.0),
                },
                zero_update_method="B2",
            )
            report = analyze_wave1(suite, results)
            self.assertEqual(report["selected_method"], "B1")
            self.assertEqual(report["ranking"], ["B1", "B3", "B0"])
            by_method = {item["method"]: item for item in report["methods"]}
            self.assertFalse(by_method["B2"]["eligible"])
            self.assertIn("zero_update_shards", by_method["B2"]["hard_gate_reasons"])
            self.assertTrue(
                (suite / "analysis" / "wave1_method_selection.json").is_file()
            )
            self.assertIn(
                "最终选择：**B1**",
                (suite / "analysis" / "wave1_method_selection.md").read_text(
                    encoding="utf-8"
                ),
            )

    def test_identical_best_score_uses_mean_then_final_tie_breakers(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite, results = self._fixture(
                Path(temporary),
                {
                    "B0": (98.0, 98.0),
                    "B1": (96.0, 99.0),
                    "B2": (97.0, 96.0),
                    "B3": (99.0, 99.0),
                },
            )
            report = analyze_wave1(suite, results)
            self.assertEqual(report["selected_method"], "B2")

    def test_rejects_cross_method_valid60_case_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite, results = self._fixture(
                root,
                {method: (99.0, 98.0) for method in METHODS},
            )
            run = results / "fixture_wave1_B3_seed1" / "run1" / "evaluations"
            _write_json(run / "pre_ppo.json", _evaluation(100.0, "-different"))
            _write_json(run / "epoch_1.json", _evaluation(99.0, "-different"))
            _write_json(run / "epoch_2.json", _evaluation(98.0, "-different"))
            with self.assertRaisesRegex(ValueError, "identical Valid60"):
                analyze_wave1(suite, results)

    def test_all_hard_gate_failures_block_formal_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite, results = self._fixture(
                Path(temporary),
                {method: (99.0, 98.0) for method in METHODS},
            )
            for method in METHODS:
                status = (
                    results / f"fixture_wave1_{method}_seed1" / "run1"
                    / "run_status.json"
                )
                _write_json(status, _status(zero_updates=1))
            with self.assertRaisesRegex(RuntimeError, "No Wave-1 method"):
                analyze_wave1(suite, results)

    def test_early_rejected_arm_with_missing_epoch_does_not_block_others(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite, results = self._fixture(
                Path(temporary),
                {
                    "B0": (99.0, 98.0),
                    "B1": (97.0, 96.0),
                    "B2": (95.0, 94.0),
                    "B3": (93.0, 92.0),
                },
            )
            run = results / "fixture_wave1_B3_seed1" / "run1"
            status = _status()
            status["canary_rejected"] = True
            _write_json(run / "run_status.json", status)
            (run / "evaluations" / "epoch_2.json").unlink()
            report = analyze_wave1(suite, results)
            self.assertEqual(report["selected_method"], "B2")
            by_method = {item["method"]: item for item in report["methods"]}
            self.assertFalse(by_method["B3"]["eligible"])
            self.assertIn("canary_rejected", by_method["B3"]["hard_gate_reasons"])
            self.assertIn(
                "missing_or_invalid_epoch_2_valid60",
                by_method["B3"]["hard_gate_reasons"],
            )


if __name__ == "__main__":
    unittest.main()
