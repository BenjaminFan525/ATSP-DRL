import json
from pathlib import Path
import tempfile
import unittest

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.experiment.run_fjsp_v2_evolutionary_parallel import (
    IGA_TEACHER_RESOURCE_POLICY,
    IGA_TEACHER_SCOPE,
    _teacher_is_reusable,
)


class Stage1IgaTeacherContractTest(unittest.TestCase):
    def _write_teacher(self, root, **overrides):
        payload = {
            "schema_version": 3,
            "teacher_scope": IGA_TEACHER_SCOPE,
            "resource_policy": IGA_TEACHER_RESOURCE_POLICY,
            "environment_semantics_version": (
                AircraftScheduleEnv.SEMANTICS_VERSION
            ),
            "case": "case_0001",
            "case_sha256": "test-sha256",
            "completion_verified": True,
            "job_priorities": [[0.0] * 20 for _ in range(6)],
            "site_priorities": [[0.0] * 43 for _ in range(6)],
        }
        payload.update(overrides)
        path = Path(root) / "case_0001.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_verified_stage1_plane_teacher_is_reusable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_teacher(temp_dir)
            self.assertTrue(_teacher_is_reusable(
                path, "case_0001", {"case_sha256": "test-sha256"}
            ))

    def test_resource_policy_teacher_cannot_be_used_for_stage1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_teacher(temp_dir, resource_policy="drl")
            self.assertFalse(_teacher_is_reusable(
                path, "case_0001", {"case_sha256": "test-sha256"}
            ))

    def test_unscoped_teacher_cannot_be_used_for_stage1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_teacher(temp_dir, teacher_scope=None)
            self.assertFalse(_teacher_is_reusable(
                path, "case_0001", {"case_sha256": "test-sha256"}
            ))


if __name__ == "__main__":
    unittest.main()
