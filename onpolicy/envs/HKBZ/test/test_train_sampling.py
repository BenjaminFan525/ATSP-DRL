import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from onpolicy.scripts.train.train_hkbz import select_train_case_dirs


class TrainSamplingTest(unittest.TestCase):
    def _dataset(self, root, counts):
        case_dirs = []
        index = 0
        for distribution, count in counts.items():
            for _ in range(count):
                index += 1
                case_dir = root / f"case_{index:04d}"
                case_dir.mkdir()
                (case_dir / "metadata.json").write_text(
                    json.dumps({"distribution": distribution}),
                    encoding="utf-8",
                )
                case_dirs.append(str(case_dir))
        return case_dirs

    def test_balanced_sampler_retains_unique_cases_and_oversamples_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = self._dataset(
                Path(temporary),
                {"iid": 8, "ood_stress": 2, "ood_scale": 1},
            )
            selected, audit = select_train_case_dirs(
                cases,
                mode="distribution_balanced",
                weights_spec="iid=0.5,ood_stress=0.4,ood_scale=0.1",
                seed=7,
            )
            counts = Counter(
                json.loads(
                    (Path(path) / "metadata.json").read_text(
                        encoding="utf-8"
                    )
                )["distribution"]
                for path in selected
            )

            self.assertEqual(len(selected), 16)
            self.assertEqual(
                counts,
                Counter({"iid": 8, "ood_stress": 6, "ood_scale": 2}),
            )
            self.assertEqual(len(set(selected)), len(cases))
            self.assertTrue(audit["expanded_for_full_coverage"])

    def test_balanced_sampler_is_seed_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = self._dataset(
                Path(temporary),
                {"iid": 8, "ood_stress": 2, "ood_scale": 1},
            )
            kwargs = {
                "mode": "distribution_balanced",
                "weights_spec": "iid=0.5,ood_stress=0.4,ood_scale=0.1",
                "seed": 19,
                "sample_size": 10,
            }
            first, _ = select_train_case_dirs(cases, **kwargs)
            second, _ = select_train_case_dirs(cases, **kwargs)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 10)

    def test_balanced_sampler_rejects_unweighted_distribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = self._dataset(
                Path(temporary), {"iid": 2, "held_out": 1}
            )
            with self.assertRaisesRegex(ValueError, "does not cover"):
                select_train_case_dirs(
                    cases,
                    mode="distribution_balanced",
                    weights_spec="iid=1.0",
                    seed=1,
                )


if __name__ == "__main__":
    unittest.main()
