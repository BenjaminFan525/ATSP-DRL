"""Real-fuser regression checks; never change services, drivers, or APT config."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("server_gpu_maintenance.sh")


class IdleCheckTests(unittest.TestCase):
    def probe(self, *paths, prelude=""):
        return subprocess.run(
            ["bash", "-c", 'source "$1"; shift; ' + prelude
             + 'assert_no_open_files "$@"', "probe", str(SCRIPT), *map(str, paths)],
            capture_output=True, text=True, timeout=15,
        )

    def test_empty_input_is_rejected(self):
        result = self.probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No files/devices supplied", result.stderr)

    def test_option_or_relative_path_is_rejected(self):
        for path in ("--help", "relative-file"):
            with self.subTest(path=path):
                result = self.probe(path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("require absolute paths", result.stderr)

    @unittest.skipUnless(shutil.which("fuser"), "fuser is not installed")
    def test_real_fuser_accepts_idle_file(self):
        with tempfile.TemporaryDirectory(prefix="hkbz-fuser-test-") as directory:
            path = Path(directory) / "idle"
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDONLY, 0o600)
            os.close(fd)
            result = self.probe(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Usage:", result.stderr)

    @unittest.skipUnless(shutil.which("fuser"), "fuser is not installed")
    def test_real_fuser_rejects_open_file(self):
        with tempfile.TemporaryDirectory(prefix="hkbz-fuser-test-") as directory:
            path = Path(directory) / "busy"
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDONLY, 0o600)
            try:
                result = self.probe(path)
            finally:
                os.close(fd)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Files/devices are in use", result.stderr)
            self.assertIn(str(os.getpid()), result.stderr)

    @unittest.skipUnless(shutil.which("fuser"), "fuser is not installed")
    def test_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="hkbz-fuser-test-") as directory:
            result = self.probe(Path(directory) / "missing")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cannot reliably establish", result.stderr)

    @unittest.skipUnless(shutil.which("fuser"), "fuser is not installed")
    def test_multiple_files_rejects_any_busy_file(self):
        with tempfile.TemporaryDirectory(prefix="hkbz-fuser-test-") as directory:
            idle, busy = Path(directory) / "idle", Path(directory) / "busy"
            fd = os.open(idle, os.O_CREAT | os.O_EXCL | os.O_RDONLY, 0o600)
            os.close(fd)
            fd = os.open(busy, os.O_CREAT | os.O_EXCL | os.O_RDONLY, 0o600)
            try:
                result = self.probe(idle, busy)
            finally:
                os.close(fd)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Files/devices are in use", result.stderr)

    def test_probe_errors_fail_closed(self):
        for code in (1, 2):
            with self.subTest(code=code):
                result = self.probe(
                    "/dummy", prelude=f'fuser() {{ echo "probe failed" >&2; return {code}; }}; ',
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Cannot reliably establish", result.stderr)


if __name__ == "__main__":
    unittest.main()
