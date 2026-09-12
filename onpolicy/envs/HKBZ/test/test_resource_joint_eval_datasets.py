"""Contract tests for the three-way resource-joint evaluation builder."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

builder_module = importlib.import_module(
    "onpolicy.envs.HKBZ.experiment.build_resource_joint_eval_datasets"
)
from onpolicy.envs.HKBZ.experiment.build_resource_joint_eval_datasets import (
    DISTRIBUTION_TARGETS,
    PARTITIONS,
    ROLE_POLICIES,
    build,
)


SMALL_COUNTS = {
    "joint": {"tune": 2, "gate": 3, "finalblind": 2},
}


class ResourceJointEvalDatasetContractTest(unittest.TestCase):
    def test_production_partition_contract_is_explicit(self):
        self.assertEqual(
            PARTITIONS,
            {
                "joint": {"tune": 60, "gate": 120, "finalblind": 60},
            },
        )

    def test_small_build_is_deterministic_and_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build(
                root / "first",
                seed=31001,
                partition_counts=SMALL_COUNTS,
            )
            second = build(
                root / "second",
                seed=31001,
                partition_counts=SMALL_COUNTS,
            )
            self.assertEqual(first, second)
            self.assertFalse((root / "first.building").exists())
            self.assertFalse((root / "second.building").exists())
            self.assertEqual(first["case_sha256_count"], 7)
            self.assertEqual(first["cross_collection_exact_overlap_count"], 0)

            for family, roles in SMALL_COUNTS.items():
                for role, count in roles.items():
                    collection = first["partitions"][family][role]
                    self.assertEqual(collection["count"], count)
                    self.assertEqual(len(collection["cases"]), count)
                    self.assertEqual(
                        collection["sealed"], ROLE_POLICIES[role]["sealed"]
                    )
                    self.assertEqual(
                        collection["selection_policy"],
                        ROLE_POLICIES[role]["selection_policy"],
                    )
                    self.assertEqual(
                        sum(collection["profile_counts"].values()), count
                    )
                    self.assertEqual(
                        sum(collection["distribution_counts"].values()), count
                    )
                    for record in collection["cases"]:
                        self.assertEqual(
                            set(record["file_sha256"]),
                            {
                                "job.json",
                                "fixed_resources.json",
                                "mobile_resources.json",
                                "sites.json",
                                "flights.json",
                            },
                        )

    def test_production_distribution_contract_is_50_45_5(self):
        self.assertEqual(DISTRIBUTION_TARGETS, {
            "iid": 0.50,
            "ood_stress": 0.45,
            "ood_scale": 0.05,
        })

    def test_exclude_manifest_rejects_legacy_hash_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            build(source, seed=31002, partition_counts=SMALL_COUNTS)
            legacy = root / "legacy_manifest.json"
            source_manifest = json.loads(
                (source / "manifest.json").read_text(encoding="utf-8")
            )
            first_case = source_manifest["partitions"]["joint"]["tune"]["cases"][0]
            legacy.write_text(
                json.dumps({"splits": {"validation": {"cases": [first_case]}}}),
                encoding="utf-8",
            )
            target = root / "target"
            with self.assertRaisesRegex(RuntimeError, "excluded manifest"):
                build(
                    target,
                    seed=31002,
                    partition_counts=SMALL_COUNTS,
                    exclude_manifests=[legacy],
                )
            self.assertFalse(target.exists())
            self.assertTrue((root / "target.building").exists())

    def test_repeated_exclude_manifest_is_harmless_but_still_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            build(source, seed=31003, partition_counts=SMALL_COUNTS)
            manifest_path = source / "manifest.json"
            target = root / "target"
            with self.assertRaisesRegex(RuntimeError, "excluded manifest"):
                build(
                    target,
                    seed=31003,
                    partition_counts=SMALL_COUNTS,
                    exclude_manifests=[manifest_path, manifest_path],
                )
            self.assertFalse(target.exists())

    def test_cross_collection_content_duplicate_is_rejected_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "duplicate"
            with mock.patch.object(
                builder_module, "_role_namespace", return_value="resource_joint/duplicate"
            ):
                with self.assertRaisesRegex(RuntimeError, "duplicate content"):
                    build(target, seed=31004, partition_counts=SMALL_COUNTS)
            self.assertFalse(target.exists())
            self.assertTrue((root / "duplicate.building").exists())


if __name__ == "__main__":
    unittest.main()
