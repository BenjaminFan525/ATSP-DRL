import unittest
from onpolicy.scripts.train.resume_stage3_after_probe import allowed_probe, interrupted_validation


class RecoveryTests(unittest.TestCase):
    def test_only_exact_authorized_probe(self):
        lease = {"pid": 123, "start_ticks": "456", "gpu": 1, "command": "probe", "cpus": [11, 75]}
        self.assertTrue(allowed_probe(lease, lease))
        for key, value in (("pid", 124), ("start_ticks", "457"), ("gpu", 0),
                           ("command", "other"), ("cpus", [12, 76])):
            with self.subTest(key=key):
                self.assertFalse(allowed_probe(dict(lease, **{key: value}), lease))

    def test_only_known_administrative_validation_failure(self):
        row = {"request_id": "E3_T1_e000480", "ok": False, "evaluation": None,
               "checkpoint_sha256": "sha", "error": "KeyboardInterrupt: Signal 15; preserve diagnostic state"}
        self.assertTrue(interrupted_validation(row, "sha"))
        self.assertFalse(interrupted_validation(row, "different"))
        for key, value in (("request_id", "E0_T0_e000480"), ("ok", True),
                           ("evaluation", {}), ("error", "CUDA out of memory")):
            with self.subTest(key=key):
                self.assertFalse(interrupted_validation(dict(row, **{key: value}), "sha"))


if __name__ == "__main__":
    unittest.main()
