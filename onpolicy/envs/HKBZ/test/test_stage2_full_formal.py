from pathlib import Path
import unittest

from onpolicy.scripts.train.prepare_stage2_full_formal import (
    EXPECTED_CASE_SLOTS,
    EXPECTED_DISTRIBUTION_COUNTS,
    EXPECTED_UNIQUE_CASES,
    FORMAL_BC_EPOCHS,
    FORMAL_BC_ROLLOUTS,
    FORMAL_METHODS,
    FORMAL_PPO_EPOCHS,
    configure_full_formal_command,
    full_coverage_audit,
)


ROOT = Path(__file__).resolve().parents[4]
DATASET = ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train"


class Stage2FullFormalTest(unittest.TestCase):
    @staticmethod
    def _value(command, flag):
        return command[command.index(flag) + 1]

    def test_full_formal_keeps_five_stable_wave4_methods(self):
        self.assertEqual(
            list(FORMAL_METHODS),
            [
                "F0_soft_control",
                "F1_hard_reservation",
                "F2_iga_flow_bc",
                "F3_wait_constraint",
                "F4_iga_constraint",
            ],
        )
        self.assertNotIn(
            "B4_gradual_shared",
            {
                method["source_wave4_method"]
                for method in FORMAL_METHODS.values()
            },
        )
        self.assertEqual(
            FORMAL_METHODS["F3_wait_constraint"]["reuse_bc_from"],
            "F0_soft_control",
        )
        self.assertEqual(
            FORMAL_METHODS["F4_iga_constraint"]["reuse_bc_from"],
            "F2_iga_flow_bc",
        )

    def test_commands_use_full_stage1_coverage_and_frozen_stage2(self):
        base = ["python", "train_hkbz.py", "--rollout_until_done"]
        for method_id, method in FORMAL_METHODS.items():
            command = configure_full_formal_command(
                base,
                method_id,
                method,
                run_tag="formal-test",
                teacher_dir=Path("/tmp/teacher"),
                teacher_index=Path("/tmp/index.json"),
            )
            value = lambda flag: self._value(command, flag)
            self.assertEqual(int(value("--max_train_cases")), 0)
            self.assertEqual(int(value("--train_sampling_size")), 0)
            self.assertEqual(int(value("--num_episodes")), FORMAL_PPO_EPOCHS)
            self.assertEqual(
                int(value("--gnn_freeze_epochs")), FORMAL_PPO_EPOCHS
            )
            self.assertEqual(
                int(value("--plane_freeze_epochs")), FORMAL_PPO_EPOCHS
            )
            self.assertEqual(
                int(value("--device_bc_pretrain_epochs")), FORMAL_BC_EPOCHS
            )
            self.assertEqual(
                int(value("--device_bc_min_rollouts_per_epoch")),
                FORMAL_BC_ROLLOUTS,
            )
            self.assertEqual(
                int(value("--device_bc_max_rollouts_per_epoch")),
                FORMAL_BC_ROLLOUTS,
            )
            self.assertEqual(value("--max_graphs_per_forward"), "1000")
            self.assertEqual(value("--n_rollout_threads"), "20")
            self.assertEqual(value("--selection_metric"), "composite_tail")
            self.assertNotIn("--stage2_allow_shared_unfreeze", command)
            self.assertNotIn("--skip_pre_ppo_eval", command)
            self.assertNotIn("--skip_epoch_eval", command)

    def test_full_coverage_audit_is_exactly_stage1_train600(self):
        audit = full_coverage_audit(DATASET, seed=1)
        self.assertEqual(audit["unique_cases"], EXPECTED_UNIQUE_CASES)
        self.assertEqual(audit["selected_cases"], EXPECTED_CASE_SLOTS)
        self.assertEqual(audit["shards_per_epoch"], 48)
        self.assertEqual(
            audit["selected_distribution_counts"],
            EXPECTED_DISTRIBUTION_COUNTS,
        )
        self.assertTrue(audit["expanded_for_full_coverage"])


if __name__ == "__main__":
    unittest.main()
