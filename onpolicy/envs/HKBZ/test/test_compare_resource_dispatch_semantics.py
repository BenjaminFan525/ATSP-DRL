import json
import tempfile
import unittest
from pathlib import Path

from onpolicy.envs.HKBZ.experiment.compare_resource_dispatch_semantics import (
    ARMS,
    compare,
)


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CompareResourceDispatchSemanticsTest(unittest.TestCase):
    def test_paired_comparison_validates_budget_and_counts_lookahead(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            lookahead = root / "lookahead"
            output = root / "comparison"
            cases = [
                {
                    "case": "case_0001",
                    "profile": "p1",
                    "distribution": "iid",
                    **{arm: 100.0 for arm in ARMS},
                },
                {
                    "case": "case_0002",
                    "profile": "p2",
                    "distribution": "ood_stress",
                    **{arm: 200.0 for arm in ARMS},
                },
            ]
            _write(
                legacy / "summary.json",
                {"cases": cases, "device_lookahead_dispatch": False},
            )
            _write(
                lookahead / "summary.json",
                {"cases": cases, "device_lookahead_dispatch": True},
            )
            search = {
                "population": 6,
                "generations": 6,
                "evaluated_candidates": 36,
                "seed": 1,
                "common_initial_population_seed": 2,
                "n_var": 10,
                "iga_device_count": 4,
            }
            for case_index, case_item in enumerate(cases):
                case = case_item["case"]
                for arm in ARMS:
                    common = {
                        "case": case,
                        "arm": arm,
                        "case_sha256": "case-sha",
                        "frozen_plane_checkpoint_sha256": "model-sha",
                        "frozen_plane_source_command_sha256": "command-sha",
                        "evaluation_tau": 0.3,
                        "search": None if arm == "hungarian_all" else search,
                    }
                    _write(
                        legacy / "cases" / case / f"{arm}.json",
                        {**common, "makespan": float(100 + 100 * case_index)},
                    )
                    _write(
                        lookahead / "cases" / case / f"{arm}.json",
                        {**common, "makespan": float(90 + 100 * case_index)},
                    )
                    _write(
                        lookahead / "trajectories" / arm / f"{case}.json",
                        {
                            "decision_trace": [
                                {
                                    "requests": [
                                        {"id": 1, "is_lookahead": True}
                                    ],
                                    "resource_decisions": [
                                        {"selected_request_id": 1}
                                    ],
                                }
                            ],
                            "resource_trajectory": [
                                {
                                    "is_lookahead": True,
                                    "lead_time_at_dispatch": 100.0,
                                    "trans_time": 60.0,
                                }
                            ],
                        },
                    )

            payload = compare(legacy, lookahead, output)
            for arm in ARMS:
                effect = payload["paired_semantic_effects"][arm]
                self.assertEqual(effect["mean_delta_lookahead_minus_legacy"], -10.0)
                self.assertEqual(effect["lookahead_wins"], 2)
                self.assertEqual(
                    payload["lookahead_trace_aggregates"][arm][
                        "total_lookahead_dispatches"
                    ],
                    2,
                )
            self.assertTrue((output / "lookahead_comparison.json").is_file())
            self.assertTrue((output / "lookahead_comparison.md").is_file())


if __name__ == "__main__":
    unittest.main()
