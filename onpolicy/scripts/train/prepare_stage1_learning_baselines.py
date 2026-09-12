#!/usr/bin/env python3
"""Materialize equal-budget Stage-1 learning-baseline training commands.

The source command is the single source of truth for dataset, PPO budget,
reward, evaluation split and hardware settings.  This generator changes only
the seed, experiment name and scheduling architecture, and removes any IGA
behavior-cloning/resume state that would contaminate a published PPO baseline.
It prepares commands but never launches jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.algorithms.utils.stage1_baselines import (
    normalize_stage1_baseline,
)


DEFAULT_METHODS = ("l2d", "multi_ppo", "fjsp_drl", "daniel")
TRAIN_SCRIPT = ROOT / "onpolicy/scripts/train/train_hkbz.py"
VALUE_OPTIONS_TO_REMOVE = (
    "--checkpoint_dir",
    "--selection_checkpoint_dir",
    "--plane_bc_teacher_dir",
    "--plane_bc_lr",
    "--plane_bc_shared_lr_scale",
    "--plane_bc_freeze_shared_epochs",
    "--plane_bc_pair_loss_coef",
    "--plane_bc_order_loss_coef",
    "--plane_bc_dagger_schedule",
    "--bc_reference_kl_coef_schedule",
    # Shared evaluators and GPU-local phase locks belong to the launcher that
    # owns the target hardware.  Carrying these values from an archived source
    # command can silently connect a new baseline to a stale socket/CPU pool.
    "--shared_eval_socket",
    "--shared_eval_cpu_set",
    "--shared_eval_timeout_seconds",
    "--shared_gpu_phase_lock",
)
SWITCHES_TO_REMOVE = (
    "--resume_stage1",
    "--resume_stage2",
    "--reset_optimizers_on_resume",
    "--reset_value_normalizer_on_resume",
    "--plane_bc_only",
    "--bc_reference_hard_gate",
    "--adaptive_bc_reference_kl",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remove_option(command: list[str], flag: str, *, takes_value=True) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + (2 if takes_value else 1)]


def _set_option(command: list[str], flag: str, value) -> None:
    while command.count(flag) > 1:
        index = len(command) - 1 - command[::-1].index(flag)
        del command[index:index + 2]
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))


def _option(command: list[str], flag: str, default=None):
    if flag not in command:
        return default
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"Command option {flag} is missing its value.")
    return command[index + 1]


def _resolve_command_path(value: str, source_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        marker = "/onpolicy/"
        if marker in str(path):
            relocated = (
                ROOT / "onpolicy" / str(path).split(marker, 1)[1]
            )
            if relocated.exists():
                return relocated.resolve()
        return resolved
    candidates = (
        (source_path.parent / path).resolve(),
        (ROOT / path).resolve(),
    )
    return next((item for item in candidates if item.exists()), candidates[-1])


def _relocate_source_command(command: list[str]) -> tuple[list[str], str | None]:
    """Bind an archived command to this checkout and active interpreter."""

    if len(command) < 2:
        return list(command), None
    normalized_train = str(command[1]).replace("\\", "/")
    suffix = "/onpolicy/scripts/train/train_hkbz.py"
    legacy_root = (
        normalized_train[: -len(suffix)]
        if normalized_train.endswith(suffix) else None
    )
    relocated = []
    for token in command:
        value = str(token)
        if legacy_root and value.startswith(legacy_root + "/"):
            value = str(ROOT / value[len(legacy_root) + 1:])
        relocated.append(value)
    relocated[0] = str(Path(sys.executable).resolve())
    relocated[1] = str(TRAIN_SCRIPT)
    return relocated, legacy_root


def _relocate_environment(environment: dict, legacy_root: str | None) -> dict:
    if not legacy_root:
        return dict(environment)
    return {
        key: (
            str(ROOT) + str(value)[len(legacy_root):]
            if str(value).startswith(legacy_root + "/") else value
        )
        for key, value in environment.items()
    }


def _validate_source(command: list[str], source_path: Path) -> dict:
    if len(command) < 2:
        raise ValueError("Source command must include Python and train_hkbz.py.")
    if Path(command[1]).name != "train_hkbz.py":
        raise ValueError(
            "Source command must invoke onpolicy/scripts/train/train_hkbz.py."
        )
    env_config_value = _option(command, "--env_config")
    if not env_config_value:
        raise ValueError("Source command must pin --env_config.")
    env_config_path = _resolve_command_path(env_config_value, source_path)
    if not env_config_path.is_file():
        raise FileNotFoundError(env_config_path)
    environment = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
    if str(environment.get("resource_policy", "heuristic")) != "heuristic":
        raise ValueError(
            "The pinned environment config must use resource_policy=heuristic."
        )
    dataset_manifest = _option(
        command, "--dataset_manifest", environment.get("dataset_manifest")
    )
    if dataset_manifest in {None, ""}:
        raise ValueError(
            "Source command must pin --dataset_manifest for paired evaluation."
        )
    budget = {
        "num_env_steps": _option(command, "--num_env_steps"),
        "num_episodes": _option(command, "--num_episodes"),
        "episode_length": _option(command, "--episode_length"),
        "n_rollout_threads": _option(command, "--n_rollout_threads"),
    }
    if budget["num_env_steps"] is None and budget["num_episodes"] is None:
        raise ValueError(
            "Source command must explicitly pin --num_env_steps or "
            "--num_episodes."
        )
    return {
        "env_config": str(env_config_path),
        "dataset_manifest": str(dataset_manifest),
        "budget": budget,
    }


def baseline_command(
    source_command: list[str],
    *,
    method: str,
    seed: int,
    run_tag: str,
) -> list[str]:
    method = normalize_stage1_baseline(method)
    if method == "proposed":
        raise ValueError("This generator is for the four external baselines.")
    command = list(source_command)
    for flag in VALUE_OPTIONS_TO_REMOVE:
        _remove_option(command, flag, takes_value=True)
    for flag in SWITCHES_TO_REMOVE:
        _remove_option(command, flag, takes_value=False)
    _set_option(command, "--stage1_baseline", method)
    _set_option(command, "--training_stage", "plane_pretrain")
    _set_option(command, "--resource_policy", "heuristic")
    _set_option(command, "--plane_order_mode", "fixed")
    _set_option(command, "--plane_pair_decoder", "joint_pair")
    _set_option(command, "--plane_bc_pretrain_epochs", 0)
    _set_option(command, "--bc_reference_kl_coef", 0.0)
    _set_option(command, "--bc_reference_target_kl", 0.0)
    _set_option(command, "--bc_reference_kl_coef_schedule", "")
    _set_option(command, "--seed", int(seed))
    _set_option(
        command,
        "--experiment_name",
        f"{run_tag}_{method}_seed{int(seed)}",
    )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-command-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default="1,2,3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source_command_json.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_payload = _read_json(source_path)
    source_command, legacy_root = _relocate_source_command(
        list(source_payload.get("command", ()))
    )
    source_audit = _validate_source(source_command, source_path)
    methods = tuple(
        normalize_stage1_baseline(item)
        for item in args.methods.split(",") if item.strip()
    )
    if not methods or "proposed" in methods or len(methods) != len(set(methods)):
        raise ValueError(
            "--methods must contain unique external Stage-1 baselines."
        )
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("Fair comparison requires at least three unique seeds.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    records = []
    source_environment = _relocate_environment(
        dict(source_payload.get("environment", {})), legacy_root
    )
    for method in methods:
        for seed in seeds:
            command = baseline_command(
                source_command,
                method=method,
                seed=seed,
                run_tag=args.run_tag,
            )
            command_path = output_dir / f"{method}_seed{seed}.command.json"
            payload = {
                "schema_version": 1,
                "stage1_baseline": method,
                "seed": seed,
                "command": command,
                "shell_command": shlex.join(command),
                "environment": source_environment,
                "source_command_json": str(source_path),
                "source_command_sha256": source_sha256,
                "fair_comparison_contract": source_audit,
            }
            _atomic_json(command_path, payload)
            records.append({
                "stage1_baseline": method,
                "seed": seed,
                "command_json": str(command_path),
            })
    manifest_path = output_dir / "experiment_manifest.json"
    _atomic_json(manifest_path, {
        "schema_version": 1,
        "run_tag": args.run_tag,
        "scope": "stage1_learning_baselines",
        "source_command_json": str(source_path),
        "source_command_sha256": source_sha256,
        "methods": list(methods),
        "seeds": list(seeds),
        "run_count": len(records),
        "fair_comparison_contract": source_audit,
        "runs": records,
    })
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
