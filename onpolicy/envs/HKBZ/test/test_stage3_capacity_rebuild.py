import unittest
from onpolicy.scripts.train.rebuild_stage3_committed import equivalent


class RebuildTests(unittest.TestCase):
    def test_only_resource_metrics_excluded(self):
        self.assertTrue(equivalent({"kl": .01, "peak_reserved_gib": 1, "execution_cache": {}},
                                   {"kl": .01000001, "peak_reserved_gib": 3})[0])
        self.assertFalse(equivalent({"mask_mismatch": 0}, {"mask_mismatch": 1})[0])
        self.assertFalse(equivalent({"costs": [1., 2.]}, {"costs": [1., 3.]})[0])

    def test_nonfinite_and_schema_changes_rejected(self):
        self.assertFalse(equivalent({"kl": float("nan")}, {"kl": float("nan")})[0])
        self.assertFalse(equivalent({"decisions": [1, 2]}, {"decisions": [1]})[0])
        self.assertFalse(equivalent({"decisions": 1}, {"missing": 1})[0])


if __name__ == "__main__":
    unittest.main()
