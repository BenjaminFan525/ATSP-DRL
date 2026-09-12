"""CPU-only tests: resource guards and independent diagnostic scheduling."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from onpolicy.scripts.train.probe_stage3_throughput import (
    GIB, Resources, cgroup_memory, guard_reason, macroblock_batches, memory_kib,
    parser, service_cgroup, write_json, save_archive, load_archive)


class ProbeTests(unittest.TestCase):
    def plan(self):
        return [{"macroblock": group // 4,
                 "cases": [{"path": str(case)} for case in range(4) for _ in range(2)],
                 "seeds": [100 * group + index for index in range(8)]} for group in range(40)]

    def test_aligned_macroblock_fresh_and_balanced(self):
        start, batches = macroblock_batches(self.plan(), 30)
        self.assertEqual(start, 32)
        self.assertEqual(len(batches), 4)
        self.assertEqual({b["macroblock"] for b in batches}, {8})

    def test_supported_batch_sizes(self):
        for size in (8, 16, 32):
            with self.subTest(size=size):
                start, batches = macroblock_batches(self.plan(), 32, size)
                self.assertEqual((start, len(batches)), (32, size // 8))

    def test_invalid_or_exhausted_schedule(self):
        for cursor, size in ((-1, 32), (40, 32), (0, 64)):
            with self.subTest(cursor=cursor, size=size), self.assertRaises(ValueError):
                macroblock_batches(self.plan(), cursor, size)

    def test_duplicate_seed_or_imbalanced_cases_rejected(self):
        plan = self.plan()
        plan[33]["seeds"] = plan[32]["seeds"]
        with self.assertRaises(ValueError):
            macroblock_batches(plan, 30)
        plan = self.plan()
        plan[32]["cases"][0] = {"path": "wrong"}
        with self.assertRaises(ValueError):
            macroblock_batches(plan, 30)

    def test_memory_parser(self):
        self.assertEqual(memory_kib("Rss: 100 kB\nPss: 40 kB\nTHPeligible: 0\n"),
                         {"Rss": 102400, "Pss": 40960})

    def test_cgroup_peak_optional_on_old_kernel(self):
        with patch.object(Path, "read_text", side_effect=["1234", FileNotFoundError()]):
            measured = cgroup_memory(Path("/test"))
            self.assertEqual(measured["cgroup_memory_bytes"], 1234)
            self.assertNotIn("cgroup_memory_peak_bytes", measured)
            self.assertIn("sampled", measured["cgroup_peak_source"])
        with patch.object(Path, "read_text", side_effect=["1234", "5678"]):
            self.assertEqual(cgroup_memory(Path("/test"))["cgroup_memory_peak_bytes"], 5678)

    def test_monitor_close_before_start(self):
        monitor = Resources(Path("/unused"), Path("/unused"), None, "gpu")
        monitor.close()
        self.assertTrue(monitor.stop.is_set())

    def test_archive_reuse_requires_identical_parent_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "group.pkl"
            save_archive(path, [{"test": [1, 2, 3]}], {"checkpoint": "same"})
            self.assertEqual(load_archive(path, {"checkpoint": "same"}), [{"test": [1, 2, 3]}])
            with self.assertRaises(ValueError):
                load_archive(path, {"checkpoint": "different"})
            path.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                load_archive(path, {"checkpoint": "same"})

    def test_independent_memory_and_time_guards(self):
        sample = {"host_available_bytes": 30 * GIB, "cgroup_memory_bytes": 10 * GIB}
        self.assertIsNone(guard_reason(sample, 0, 24 * GIB, 3300))
        self.assertIn("runtime", guard_reason(sample, 3300, 24 * GIB, 3300))
        self.assertIn("headroom", guard_reason(dict(sample, host_available_bytes=23 * GIB), 0, 24 * GIB, 3300))
        self.assertIn("hard limit", guard_reason(dict(sample, cgroup_memory_bytes=23 * GIB), 0, 24 * GIB, 3300))

    def test_unbounded_or_production_cgroup_rejected(self):
        for cgroup, limit in (("/hkbz-s3repr-live.service", str(24 * GIB)),
                              ("/hkbz-s3probe-test.service", "max"),
                              ("/hkbz-s3probe-test.service", str(25 * GIB))):
            with self.subTest(cgroup=cgroup, limit=limit):
                with patch.object(Path, "read_text", side_effect=["0::" + cgroup, limit]):
                    with self.assertRaises(RuntimeError):
                        service_cgroup()

    def test_bounded_probe_cgroup_accepted(self):
        with patch.object(Path, "read_text", side_effect=["0::/hkbz-s3probe-test.service", str(24 * GIB)]):
            self.assertEqual(service_cgroup().name, "hkbz-s3probe-test.service")

    def test_atomic_json_and_default_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            write_json(path, {"complete": False})
            self.assertIn('"complete": false', path.read_text())
            self.assertFalse(path.with_suffix(".json.tmp").exists())
        args = parser().parse_args(["--manifest", "m", "--checkpoint", "c", "--output", "o", "--gpu-uuid", "gpu"])
        self.assertEqual((args.batch_trajectories, args.tbptt_steps, args.cache_mib), (32, 16, 4096))


if __name__ == "__main__":
    unittest.main()
