"""Contracts for the eight-arm Stage3 credit-assignment factorial."""

from __future__ import annotations

import json
from pathlib import Path

from onpolicy.config.config import get_config
from onpolicy.scripts.train.prepare_stage3_credit_factorial import (
    ARMS,
    DEFAULT_BASE_MANIFEST,
    DEFAULT_PYTHON,
    DEFAULT_SOURCE,
    arm_factors,
    build_command,
    option_value,
    validate_command,
)
from onpolicy.scripts.train.run_stage3_credit_manifest import parse_cpu_set
from onpolicy.scripts.train.train_hkbz import parse_args as parse_train_args


def command_mapping(command: list[str]) -> dict[str, object]:
    mapping = {"python": command[0], "script": command[1]}
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


def test_wave1_commands_form_a_complete_isolated_two_by_two_by_two():
    base = json.loads(DEFAULT_BASE_MANIFEST.read_text(encoding="utf-8"))["command"]
    mappings = {}
    for arm in ARMS:
        command = build_command(
            base,
            phase="wave1",
            arm=arm,
            experiment_name=f"contract_wave1_{arm}_seed3",
            source_checkpoint=DEFAULT_SOURCE,
            python=DEFAULT_PYTHON,
        )
        validate_command(command, phase="wave1", arm=arm)
        parsed = parse_train_args(command[2:], get_config())
        assert parsed.seed == 3
        assert parsed.n_rollout_threads == 8
        assert parsed.n_eval_rollout_threads == 4
        assert parsed.actor_warmup_shards == 60
        assert parsed.gnn_freeze_epochs == parsed.num_episodes == 6
        assert parsed.actor_grad_clip_mode == "per_group"
        factors = arm_factors(arm)
        assert parsed.counterfactual_q_baseline == factors["counterfactual_q"]
        assert (
            parsed.role_event_credit_mode == "critical_path_v2"
        ) == factors["critical_path_v2"]
        assert parsed.role_sequential_ppo == factors["role_sequential_ppo"]
        mappings[arm] = command_mapping(command)

    ignored = {
        "--experiment_name", "--counterfactual_q_baseline",
        "--role_event_credit_mode", "--role_sequential_ppo",
    }
    reference = {k: v for k, v in mappings["W000"].items() if k not in ignored}
    for arm, mapping in mappings.items():
        assert {k: v for k, v in mapping.items() if k not in ignored} == reference
    assert len({tuple(sorted(arm_factors(arm).items())) for arm in ARMS}) == 8


def test_canary_exercises_actor_and_all_new_paths_at_reduced_memory_shape():
    base = json.loads(DEFAULT_BASE_MANIFEST.read_text(encoding="utf-8"))["command"]
    command = build_command(
        base,
        phase="canary",
        arm="W111",
        experiment_name="contract_canary_W111_seed3",
        source_checkpoint=DEFAULT_SOURCE,
        python=DEFAULT_PYTHON,
    )
    validate_command(command, phase="canary", arm="W111")
    assert option_value(command, "--actor_warmup_shards") == "0"
    assert option_value(command, "--max_graphs_per_forward") == "200"
    assert option_value(command, "--train_sampling_size") == "24"
    assert option_value(command, "--max_eval_cases") == "12"
    assert "--counterfactual_q_baseline" in command
    assert "--role_sequential_ppo" in command
    assert option_value(command, "--role_event_credit_mode") == "critical_path_v2"


def test_cpu_sets_are_exact_disjoint_smt_lanes():
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
