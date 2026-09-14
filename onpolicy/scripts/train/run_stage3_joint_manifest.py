#!/usr/bin/env python3
"""Execute one immutable Stage3 command manifest without shell evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flag_value(command: list[str], flag: str) -> str:
    """Return exactly one value for a required argv flag."""

    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1:
        raise ValueError(
            f"Stage3 manifest requires exactly one {flag}; found {len(positions)}."
        )
    position = positions[0]
    if position + 1 >= len(command) or command[position + 1].startswith("--"):
        raise ValueError(f"Stage3 manifest flag has no value: {flag}")
    return command[position + 1]


def _validate_pure_rl_command(
    command: list[str],
    *,
    allow_shared_frozen: bool = False,
    allow_staged_shared: bool = False,
) -> None:
    """Fail closed if a Stage3 manifest reintroduces supervised training."""

    exact_values = {
        "--training_stage": "joint_finetune",
        "--resource_policy": "drl",
        "--plane_bc_pretrain_epochs": "0",
        "--device_bc_pretrain_epochs": "0",
        "--plane_bc_dagger_schedule": "0.0",
        "--plane_bc_staging_dagger_schedule": "0.0",
        "--device_bc_dagger_schedule": "0.0",
        "--bc_reference_kl_coef": "0.0",
        "--bc_reference_kl_coef_schedule": "0.0",
        "--bc_reference_target_kl": "0.0",
        "--resource_ppo_update_schedule": "joint",
        "--joint_team_ppo_scope": "all",
    }
    for flag, expected in exact_values.items():
        actual = _flag_value(command, flag)
        if actual != expected:
            raise ValueError(
                f"Stage3 pure-RL contract requires {flag}={expected}, got {actual}."
            )
    for forbidden in (
        "--joint_iga_teacher_dir",
        "--joint_iga_teacher_index",
        "--resource_iga_teacher_dir",
        "--resource_iga_teacher_index",
        "--plane_bc_teacher_dir",
        "--resource_bc_checkpoint",
        "--bc_reference_hard_gate",
        "--skip_pre_ppo_eval",
    ):
        if forbidden in command:
            raise ValueError(
                f"Stage3 pure-RL manifest contains forbidden flag: {forbidden}"
            )
    for required_switch in (
        "--joint_team_ppo",
        "--use_eval",
        "--strict_checkpoint_contract",
    ):
        if command.count(required_switch) != 1:
            raise ValueError(
                f"Stage3 pure-RL manifest requires one {required_switch}."
            )
    freeze_epochs = int(_flag_value(command, "--gnn_freeze_epochs"))
    total_epochs = int(_flag_value(command, "--num_episodes"))
    shared_gradient_method = _flag_value(
        command, "--shared_gradient_method"
    )
    if shared_gradient_method not in {
        "sum", "norm_balance", "norm_pcgrad", "cagrad"
    }:
        raise ValueError(
            f"Unsupported shared gradient method: {shared_gradient_method}"
        )
    if (
        shared_gradient_method != "sum"
        and "--role_atomic_ppo" not in command
    ):
        raise ValueError(
            "Shared gradient surgery requires role-atomic PPO."
        )
    if (
        shared_gradient_method != "sum"
        and "--shared_encoder_pcgrad" in command
    ):
        raise ValueError(
            "New shared gradient methods cannot be mixed with legacy PCGrad."
        )
    for flag in (
        "--shared_grad_ema_beta",
        "--shared_grad_norm_power",
        "--shared_grad_min_scale",
        "--shared_grad_max_scale",
        "--shared_grad_conflict_threshold",
        "--shared_cagrad_c",
    ):
        float(_flag_value(command, flag))
    if allow_shared_frozen and allow_staged_shared:
        raise ValueError('Stage3 shared mode cannot be both frozen and staged.')
    if allow_shared_frozen:
        if freeze_epochs < total_epochs:
            raise ValueError(
                "Stage3 shared-frozen research requires gnn_freeze_epochs "
                ">= num_episodes."
            )
        if command.count("--stage3_allow_shared_frozen") != 1:
            raise ValueError(
                "Stage3 shared-frozen research requires its explicit switch."
            )
    elif allow_staged_shared:
        if not 0 <= freeze_epochs < total_epochs:
            raise ValueError(
                'Stage3 staged shared adaptation requires 0 <= '
                'gnn_freeze_epochs < num_episodes.'
            )
        if "--stage3_allow_shared_frozen" in command:
            raise ValueError(
                'Stage3 staged shared adaptation cannot use the frozen switch.'
            )
        schedule = [
            float(value.strip())
            for value in _flag_value(
                command, '--shared_actor_lr_scale_schedule'
            ).split(',')
            if value.strip()
        ]
        if not schedule or not any(
            schedule[min(epoch, len(schedule) - 1)] > 0.0
            for epoch in range(freeze_epochs, total_epochs)
        ):
            raise ValueError(
                'Stage3 staged shared adaptation requires a positive '
                'post-freeze shared LR scale.'
            )
        if '--shared_encoder_pcgrad' in command and '--role_atomic_ppo' not in command:
            raise ValueError('Shared PCGrad requires role-atomic PPO.')
        if (
            shared_gradient_method != "sum"
            and command.count("--shared_gradient_diagnostics") != 1
        ):
            raise ValueError(
                "Shared gradient research requires gradient diagnostics."
            )
        if command.count('--shared_encoder_activation_checkpoint') != 1:
            raise ValueError(
                'Stage3 graph=1000 shared adaptation requires activation '
                'checkpointing for the trainable shared encoder.'
            )
    elif freeze_epochs != 0 or "--stage3_allow_shared_frozen" in command:
        raise ValueError(
            "Canonical Stage3 pure RL requires gnn_freeze_epochs=0."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate immutable inputs without executing the training command",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    architecture = manifest.get("architecture_contract", {})
    if (
        int(manifest.get("schema_version", 0)) != 2
        or manifest.get("training_stage") != "joint_finetune"
        or manifest.get("training_method") not in {
            "pure_joint_ppo", "role_clock_phase1", "role_clock_phase2",
            "shared_encoder_adaptation",
        }
        or architecture.get("encoder_split") is not False
        or architecture.get("shared_encoder_count") != 1
        or architecture.get("joint_ppo_scope") != "all"
        or architecture.get("supervised_updates") is not False
        or architecture.get("source_policy_kl_penalty") is not False
    ):
        raise ValueError(f"Invalid Stage3 manifest: {manifest_path}")
    if "joint_teacher_index" in manifest:
        raise ValueError("Pure-RL Stage3 manifest must not contain a teacher index.")
    record = manifest["source_stage2"]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise ValueError(f"Stage3 manifest input changed: {path}")
    command = manifest.get("command")
    if not isinstance(command, list) or len(command) < 2:
        raise ValueError("Stage3 manifest has no argv command.")
    command = [str(value) for value in command]
    _validate_pure_rl_command(
        command,
        allow_shared_frozen=(
            manifest.get("training_method") in {
                "role_clock_phase1", "role_clock_phase2"
            }
            or (
                manifest.get("training_method") == "shared_encoder_adaptation"
                and architecture.get("shared_encoder_frozen") is True
            )
        ),
        allow_staged_shared=(
            manifest.get("training_method") == "shared_encoder_adaptation"
            and architecture.get("shared_encoder_frozen") is False
        ),
    )
    if Path(command[0]).resolve() != Path(os.environ.get("PYTHON", command[0])).resolve():
        raise ValueError("Stage3 launcher Python differs from the manifest.")
    if args.check_only:
        print(f"[Stage3Run] manifest validated: {manifest_path}", flush=True)
        return
    print(f"[Stage3Run] exec manifest={manifest_path}", flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
