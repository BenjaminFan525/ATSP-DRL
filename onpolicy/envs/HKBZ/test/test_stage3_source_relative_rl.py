"""Contracts for the U x K x R Stage3 source-relative RL experiment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from onpolicy.config.config import get_config
from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
from onpolicy.scripts.train.prepare_stage3_source_relative_rl import (
    ARMS,
    DEFAULT_BASE_MANIFEST,
    DEFAULT_PYTHON,
    DEFAULT_SOURCE,
    arm_factors,
    build_command,
    option_value,
    validate_command,
)
from onpolicy.scripts.train.run_stage3_source_relative_manifest import (
    parse_cpu_set,
)
from onpolicy.scripts.train.train_hkbz import parse_args as parse_train_args


def command_mapping(command: list[str]) -> dict[str, object]:
    mapping: dict[str, object] = {"python": command[0], "script": command[1]}
    index = 2
    while index < len(command):
        flag = command[index]
        assert flag.startswith("--")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            mapping[flag] = command[index + 1]
            index += 2
        else:
            mapping[flag] = True
            index += 1
    return mapping


def _base_command() -> list[str]:
    return json.loads(
        DEFAULT_BASE_MANIFEST.read_text(encoding="utf-8")
    )["command"]


def test_wave1_commands_form_complete_source_relative_factorial():
    baseline = Path("/tmp/immutable_source_train600.json")
    mappings = {}
    for arm in ARMS:
        command = build_command(
            _base_command(),
            phase="wave1",
            arm=arm,
            experiment_name=f"contract_wave1_{arm}_seed3",
            source_checkpoint=DEFAULT_SOURCE,
            source_baseline=baseline,
            python=DEFAULT_PYTHON,
        )
        validate_command(
            command,
            phase="wave1",
            arm=arm,
            source_checkpoint=DEFAULT_SOURCE,
            source_baseline=baseline,
        )
        parsed = parse_train_args(command[2:], get_config())
        factors = arm_factors(arm)
        assert parsed.seed == 3
        assert parsed.num_episodes == 6
        assert parsed.actor_warmup_shards == 30
        assert parsed.role_sequential_ppo
        assert not parsed.counterfactual_q_baseline
        assert parsed.role_event_credit_mode == "elapsed"
        assert parsed.gnn_freeze_epochs == (
            1 if factors["shared_encoder_unfreeze"] else 6
        )
        assert parsed.adaptive_bc_reference_kl == factors["source_reference_kl"]
        assert bool(parsed.bc_reference_checkpoint) == factors["source_reference_kl"]
        assert (
            parsed.cvar_case_metric == "paired_delta"
        ) == factors["source_paired_residual"]
        assert bool(parsed.paired_case_baseline_dir) == factors[
            "source_paired_residual"
        ]
        mappings[arm] = command_mapping(command)

    ignored = {
        "--experiment_name",
        "--gnn_freeze_epochs",
        "--shared_actor_lr_scale_schedule",
        "--bc_reference_checkpoint",
        "--bc_reference_kl_coef",
        "--bc_reference_kl_coef_schedule",
        "--bc_reference_target_kl",
        "--adaptive_bc_reference_kl",
        "--paired_case_baseline_dir",
        "--paired_case_baseline_coef",
        "--cvar_case_metric",
        "--cvar_policy_fraction",
        "--cvar_policy_weight",
    }
    reference = {
        key: value for key, value in mappings["U0K0R0"].items()
        if key not in ignored
    }
    for mapping in mappings.values():
        assert {
            key: value for key, value in mapping.items() if key not in ignored
        } == reference
    assert len({tuple(sorted(arm_factors(arm).items())) for arm in ARMS}) == 8


def test_canary_exercises_actor_and_worst_case_shared_update():
    baseline = Path("/tmp/immutable_source_train600.json")
    unfrozen = build_command(
        _base_command(),
        phase="canary",
        arm="U1K1R1",
        experiment_name="contract_canary_U1K1R1_seed3",
        source_checkpoint=DEFAULT_SOURCE,
        source_baseline=baseline,
        python=DEFAULT_PYTHON,
    )
    frozen = build_command(
        _base_command(),
        phase="canary",
        arm="U0K0R0",
        experiment_name="contract_canary_U0K0R0_seed3",
        source_checkpoint=DEFAULT_SOURCE,
        source_baseline=baseline,
        python=DEFAULT_PYTHON,
    )
    assert option_value(unfrozen, "--actor_warmup_shards") == "0"
    assert option_value(unfrozen, "--gnn_freeze_epochs") == "0"
    assert option_value(unfrozen, "--shared_actor_lr_scale_schedule") == "0.05"
    assert option_value(frozen, "--gnn_freeze_epochs") == "1"
    assert option_value(frozen, "--shared_actor_lr_scale_schedule") == "0.0"


def test_explicit_reference_checkpoint_overrides_legacy_sibling_resolution():
    class Policy:
        def __init__(self):
            self.reference = None

        def has_bc_reference(self):
            return self.reference is not None

        def capture_bc_reference(self, state_dict):
            self.reference = state_dict

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        explicit = root / "source.pt"
        torch.save({"model": {"weight": torch.tensor([7.0])}}, explicit)
        save_dir = root / "models"
        save_dir.mkdir()
        runner = HKBZ_Runner.__new__(HKBZ_Runner)
        runner.bc_reference_kl_coef = 0.1
        runner.bc_reference_target_kl = 0.03
        runner.bc_reference_kl_coef_schedule = ()
        runner.bc_reference_checkpoint = str(explicit)
        runner.bc_reference_resolved_path = ""
        runner.resource_bc_checkpoint = ""
        runner.training_stage = "auto"
        runner.checkpoint_dir = str(root / "unrelated_checkpoint.pt")
        runner.save_dir = str(save_dir)
        runner.policy = Policy()
        runner._ensure_bc_reference_policy()
        assert torch.equal(
            runner.policy.reference["weight"], torch.tensor([7.0])
        )
        assert Path(runner.bc_reference_resolved_path) == explicit.resolve()
        assert (save_dir / "checkpoint_ExplicitPolicyReference.pt").is_file()


def test_cpu_sets_are_disjoint_and_balance_factors_per_numa_node():
    lanes = [
        parse_cpu_set(value) for value in (
            "0-6,64-70", "7-13,71-77", "14-20,78-84",
            "21-27,85-91", "32-38,96-102", "39-45,103-109",
            "46-52,110-116", "53-59,117-123",
        )
    ]
    assert all(len(lane) == 14 for lane in lanes)
    assert all(
        lanes[left].isdisjoint(lanes[right])
        for left in range(len(lanes))
        for right in range(left + 1, len(lanes))
    )
    arm_order = (
        "U0K0R0", "U0K1R1", "U1K0R1", "U1K1R0",
        "U0K0R1", "U0K1R0", "U1K0R0", "U1K1R1",
    )
    for node_arms in (arm_order[:4], arm_order[4:]):
        factors = [arm_factors(arm) for arm in node_arms]
        for name in factors[0]:
            assert sum(int(item[name]) for item in factors) == 2
