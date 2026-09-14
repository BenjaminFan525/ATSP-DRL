import unittest
from onpolicy.scripts.train.stage3_capacity_protocol import phase_schedule, exposures, capacity_gate


class CapacityProtocolTests(unittest.TestCase):
    def cases(self):
        return [{"path": str(i), "content_sha256": str(i)} for i in range(120)]

    def test_matched_exposure_preserves_distinct_batch_composition(self):
        for start, until in ((256, 960), (264, 960), (272, 960), (504, 960), (512, 960), (960, 1440), (1440, 1920)):
            with self.subTest(start=start, until=until):
                t0 = phase_schedule(self.cases(), 1, "T0", start, until)
                t1 = phase_schedule(self.cases(), 1, "T1", start, until)
                self.assertEqual(exposures(t0), exposures(t1))
                self.assertTrue(all(v == 1 for v in exposures(t0).values()))
                self.assertEqual(sum(len(b["cases"]) for b in t0), until - start)
                self.assertTrue(all(len({c["path"] for c in b["cases"]}) == 1 for b in t0))
                self.assertTrue(all(len({c["path"] for c in b["cases"]}) == 4 for b in t1))
                self.assertTrue(all(0 < len(b["cases"]) <= 32 for b in t1))
                for endpoint in range((start // 480 + 1) * 480, until + 1, 480):
                    self.assertIn(endpoint, [b["training_episodes"] for b in t1])

    def test_incomplete_oom_or_no_headroom_cannot_pass(self):
        config = dict(batch_trajectories=32, tbptt_steps=8, performance={"cache_mib": 2048},
                      cuda_memory_fraction=1.0, checkpoint_sha256="sha")
        result = dict(completed=True, full_update_measured=True, post_update_kl_guard_passed=True,
                      resources_peaks=dict(probe_gpu_used_bytes=18 * 2**30, cgroup_memory_bytes=12 * 2**30))
        dense = dict(result, dense_case_memory_stress=True)
        self.assertTrue(capacity_gate(result, dense, config, config))
        self.assertFalse(capacity_gate(dict(result, completed=False), dense, config, config))
        self.assertFalse(capacity_gate(result, dense, config, dict(config, tbptt_steps=16)))
        self.assertFalse(capacity_gate(dict(result, resources_peaks=dict(probe_gpu_used_bytes=22 * 2**30)), dense, config, config))


if __name__ == "__main__":
    unittest.main()
