#!/usr/bin/env python3
"""Plan, audit, and execute the Stage-1 M2 -> Stage-2 HKBZ hand-off.

The controller is intentionally small and dependency-light.  It never treats
``checkpoint_DeviceBC.pt`` as a new training stage: that file is the
in-process resource-BC warm-up boundary produced by the Stage-2 runner.  A
new Stage-2 run always restores the immutable Stage-1 M2 source and performs
the warm-up again.

The command line has three operational modes::

    register  Record an externally completed Stage-1 M2 checkpoint.
    plan      Record the unique Stage-2 command (``--dry-run`` is implied by
              the absence of an execute request).
    run       Plan and execute the command, then audit warm-up/Best/Last
              checkpoint lineage.

All manifest writes are atomic.  The functions in this module are also used
by the unit tests, where subprocess execution is injected rather than
started.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = Path(
    os.environ.get(
        "PYTHON",
        str((ROOT.parent / "conda/envs/maia/bin/python3.11").resolve()),
    )
)
TRAIN_SCRIPT = ROOT / "onpolicy/scripts/train/train_hkbz.py"
AC_CONFIG = ROOT / "onpolicy/config/ac.yaml"
ENV_CONFIG = ROOT / "onpolicy/config/env_resource_joint.yaml"
DEFAULT_STAGE1_HANDOFF = ROOT / "onpolicy/config/stage1_m2_handoff.json"
DEFAULT_MANIFEST_ROOT = ROOT / "result/hkbz_train_logs/two_stage"
DEFAULT_RESULTS_ROOT = ROOT / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo"

CANONICAL_STAGE = "resource_joint"
STAGE1_KIND = "stage1_m2"
MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_HANDOFF_SCHEMA_VERSIONS = frozenset({1, 2})
DEFAULT_BC_EPOCHS = 2
DEFAULT_PPO_EPOCHS = 8
DEFAULT_PPO_EPOCH = 3
DEFAULT_BC_MIN_LABELS = 64
DEFAULT_BC_MIN_ROLLOUTS = 1
DEFAULT_BC_MAX_ROLLOUTS = 20

_RUN_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# These options describe Stage 1's plane/teacher or same-stage recovery
# machinery.  They are removed when an old Stage-1 command is supplied as a
# migration hint; a Stage-2 command is otherwise built from a clean baseline.
STAGE1_ONLY_OPTIONS = frozenset(
    {
        "--resume_stage1",
        "--reset_optimizers_on_resume",
        "--selection_checkpoint_dir",
        "--plane_bc_pretrain_epochs",
        "--plane_bc_teacher_dir",
        "--plane_bc_lr",
        "--plane_bc_shared_lr_scale",
        "--plane_bc_freeze_shared_epochs",
        "--plane_bc_pair_loss_coef",
        "--plane_bc_order_loss_coef",
        "--plane_bc_rollouts_per_epoch",
        "--plane_bc_dagger_schedule",
        "--plane_bc_dagger_seed",
        "--plane_bc_dagger_tail_start_fraction",
        "--plane_bc_dagger_tail_teacher_rate",
        "--plane_bc_initial_weight",
        "--plane_bc_relocation_weight",
        "--plane_bc_critical_op_weight",
        "--plane_bc_tail_start_fraction",
        "--plane_bc_tail_weight",
        "--plane_bc_tail_final_start_fraction",
        "--plane_bc_tail_final_weight",
        "--bc_reference_kl_coef",
        "--bc_reference_kl_coef_schedule",
        "--bc_reference_target_kl",
        "--bc_reference_hard_gate",
        "--iga_potential_beta",
        "--iga_potential_beta_schedule",
        "--iga_potential_weights_path",
        "--iga_potential_gamma",
        # A completed Stage-1 command may point at an evaluator service that
        # no longer exists. Stage 2 either evaluates locally or receives a
        # fresh socket from its own launcher.
        "--shared_eval_socket",
        "--shared_eval_cpu_set",
        "--shared_eval_timeout_seconds",
    }
)

# General training knobs that are safe to carry from the Stage-1 command
# supplied by an operator.  Every transition-critical option is overwritten
# below, so stale heuristic/recovery settings cannot leak into Stage 2.
INHERITED_OPTIONS = frozenset(
    {
        "--seed",
        "--n_training_threads",
        "--n_rollout_threads",
        "--n_eval_rollout_threads",
        "--num_env_steps",
        "--episode_length",
        "--rollout_max_steps",
        "--num_mini_batch",
        "--mini_batch_size",
        "--data_chunk_length",
        "--max_graphs_per_forward",
        "--grad_accumulation_steps",
        "--actor_grad_accumulation_steps",
        "--grad_accumulation_target_graphs",
        "--actor_grad_accumulation_target_graphs",
        "--actor_warmup_shards",
        "--lr",
        "--critic_lr",
        "--opti_eps",
        "--weight_decay",
        "--entropy_coef",
        "--value_loss_coef",
        "--max_grad_norm",
        "--clip_param",
        "--target_kl",
        "--adaptive_actor_kl",
        "--adaptive_actor_kl_low",
        "--adaptive_actor_kl_high",
        "--adaptive_actor_lr_min_scale",
        "--adaptive_actor_lr_max_scale",
        "--adaptive_actor_lr_up",
        "--adaptive_actor_lr_down",
        "--adaptive_actor_min_step_completion",
        "--hindsight_reward_mode",
        "--hindsight_cmax_coef",
        "--hindsight_shaping_coef",
        "--hindsight_terminal_cmax_coef",
        "--global_feature_mode",
        "--plane_order_mode",
        "--plane_pair_decoder",
        "--max_train_cases",
        "--max_eval_cases",
        "--train_sampling_mode",
        "--train_sampling_weights",
        "--train_sampling_size",
        "--eval_case_offset",
        "--eval_partition_seed",
        "--eval_partition_stratify_by",
        "--eval_episodes",
        "--evaluation_tau",
        "--anneal_original",
        "--anneal_final",
        "--tau_anneal_epochs",
        "--save_interval",
        "--status_heartbeat_seconds",
        "--recovery_checkpoint_interval_shards",
        "--torch_mp_sharing_strategy",
        "--ipc_timeout_seconds",
        "--safe_async_graph_clone_workers",
        "--early_stop_patience",
        "--reward_coef",
        "--rollout_until_done",
        "--use_valuenorm",
        "--safe_graph_batch_pipeline",
        "--safe_dagger_teacher_overlap",
        "--device_lookahead_safety_margin",
    }
)

SWITCH_OPTIONS = frozenset(
    {
        "--adaptive_actor_kl",
        "--bc_reference_hard_gate",
        "--use_eval",
        "--no_eval",
        "--train_domain_rand",
        "--joint_team_ppo",
        "--central_team_critic",
        "--rollout_until_done",
        "--use_valuenorm",
        "--safe_graph_batch_pipeline",
        "--safe_dagger_teacher_overlap",
        "--device_lookahead_dispatch",
        "--strict_checkpoint_contract",
        "--device_bc_role_balanced",
    }
)

# Value-taking options in the allow-list.  Unknown switches are not copied;
# this keeps migration deterministic even when a historical launcher grows a
# private option that the current parser does not understand.
VALUE_OPTIONS = INHERITED_OPTIONS | STAGE1_ONLY_OPTIONS | frozenset(
    {
        "--env_name",
        "--scenario_name",
        "--algorithm_name",
        "--experiment_name",
        "--ac_config",
        "--env_config",
        "--checkpoint_dir",
        "--resource_policy",
        "--training_stage",
        "--device_bc_pretrain_epochs",
        "--device_bc_lr",
        "--device_bc_min_labels_per_epoch",
        "--device_bc_min_rollouts_per_epoch",
        "--device_bc_max_rollouts_per_epoch",
        "--device_bc_teacher",
        "--resource_iga_teacher_dir",
        "--resource_iga_teacher_index",
        "--device_bc_dagger_schedule",
        "--device_bc_dagger_seed",
        "--resource_ppo_update_schedule",
        "--resource_ppo_warmup_epochs",
        "--ppo_epoch",
        "--num_episodes",
        "--gnn_freeze_epochs",
        "--plane_freeze_epochs",
        "--plane_order_freeze_epochs",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_run_tag(run_tag: str) -> str:
    value = str(run_tag or "").strip()
    if not _RUN_TAG_RE.fullmatch(value):
        raise ValueError(
            "run_tag must contain only letters, digits, '.', '_', and '-'; "
            f"got {run_tag!r}."
        )
    return value


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash an artifact without deserializing its (possibly huge) tensors."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_checkpoint_metadata(path: str | os.PathLike[str]) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Stage-1 M2 checkpoint does not exist: {source}")
    if source.name == "checkpoint_DeviceBC.pt":
        raise ValueError(
            "checkpoint_DeviceBC.pt is the Stage-2 resource-BC warm-up boundary; "
            "register the externally completed Stage-1 M2 checkpoint instead."
        )
    return {"path": str(source), "sha256": sha256_file(source)}


def atomic_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    refuse_existing: bool = False,
) -> Path:
    """Write JSON through fsync + replace, preserving a complete manifest."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and target.exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {target}")
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{id(payload)}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _option_indices(command: Sequence[str], flag: str) -> list[int]:
    return [index for index, value in enumerate(command) if value == flag]


def remove_option(command: list[str], flag: str, *, takes_value: bool = True) -> None:
    """Remove every occurrence of a CLI option from a mutable command."""

    while flag in command:
        index = command.index(flag)
        del command[index : index + (2 if takes_value else 1)]


def set_option(command: list[str], flag: str, value: Any) -> None:
    remove_option(command, flag, takes_value=True)
    command.extend([flag, str(value)])


def set_switch(command: list[str], flag: str, enabled: bool) -> None:
    remove_option(command, flag, takes_value=False)
    if enabled:
        command.append(flag)


def sanitize_stage1_command(source_command: Sequence[str]) -> list[str]:
    """Keep safe general options while dropping Stage-1-only controls.

    This helper exists for migration tooling and tests.  ``build_stage2_command``
    still overwrites every transition-critical setting after sanitization.
    """

    tokens = list(map(str, source_command))
    if len(tokens) < 2:
        raise ValueError("A source command must contain an executable and script.")
    result = tokens[:2]
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        takes_value = token in VALUE_OPTIONS and token not in SWITCH_OPTIONS
        if token in STAGE1_ONLY_OPTIONS:
            index += 2 if takes_value else 1
            continue
        if token not in INHERITED_OPTIONS:
            index += 2 if takes_value and index + 1 < len(tokens) else 1
            continue
        result.append(token)
        if takes_value:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"Option {token} in source command has no value.")
            result.append(tokens[index + 1])
            index += 2
        else:
            index += 1
    return result


def _load_source_command(
    path: str | os.PathLike[str] | None,
    command_key: str | None = None,
) -> list[str] | None:
    """Load either a legacy one-command record or a keyed suite manifest."""

    if not path:
        return None
    payload = read_json(path)
    command = payload.get("command")
    if command is None:
        commands = payload.get("commands")
        if not isinstance(commands, Mapping):
            raise ValueError(
                f"Source command JSON has neither command nor commands: {path}"
            )
        key = str(command_key or "").strip()
        if not key:
            if len(commands) != 1:
                available = ", ".join(sorted(map(str, commands)))
                raise ValueError(
                    "A multi-command source manifest requires source_command_key; "
                    f"available keys: {available}."
                )
            key = str(next(iter(commands)))
        entry = commands.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Source command key {key!r} is missing from {path}."
            )
        command = entry.get("argv", entry.get("command"))
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError(
            f"Source command JSON must resolve to a string list: {path}"
        )
    return command


def _append_if_missing(command: list[str], flag: str, value: Any) -> None:
    if flag not in command:
        command.extend([flag, str(value)])


def _option_value(command: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(command):
        if token == flag and index + 1 < len(command):
            candidate = str(command[index + 1])
            if not candidate.startswith("--"):
                return candidate
    return None


def build_stage2_command(
    source_m2: str | os.PathLike[str],
    *,
    run_tag: str,
    seed: int = 1,
    bc_epochs: int = DEFAULT_BC_EPOCHS,
    ppo_epochs: int = DEFAULT_PPO_EPOCHS,
    ppo_epoch: int = DEFAULT_PPO_EPOCH,
    bc_min_labels: int = DEFAULT_BC_MIN_LABELS,
    bc_min_rollouts: int = DEFAULT_BC_MIN_ROLLOUTS,
    bc_max_rollouts: int = DEFAULT_BC_MAX_ROLLOUTS,
    bc_lr: float = 0.0,
    plane_order_mode: str | None = None,
    plane_pair_decoder: str | None = None,
    global_feature_mode: str | None = None,
    source_command: Sequence[str] | None = None,
    python: str | os.PathLike[str] | None = None,
    train_script: str | os.PathLike[str] | None = None,
    env_config: str | os.PathLike[str] | None = None,
    ac_config: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Return exactly one fresh Stage-2 command for the supplied M2 source."""

    source = source_checkpoint_metadata(source_m2)
    tag = validate_run_tag(run_tag)
    positive = {
        "bc_epochs": bc_epochs,
        "ppo_epochs": ppo_epochs,
        "ppo_epoch": ppo_epoch,
        "bc_min_labels": bc_min_labels,
        "bc_min_rollouts": bc_min_rollouts,
        "bc_max_rollouts": bc_max_rollouts,
    }
    for name, raw in positive.items():
        if int(raw) <= 0:
            raise ValueError(f"{name} must be positive for resource_joint: {raw}")
    if float(bc_lr) < 0.0:
        raise ValueError("bc_lr must be non-negative.")
    if source_command is not None:
        # The protected M2 contract is semantic as well as tensor-level.  If
        # an operator supplies the original Stage-1 command, preserve its
        # architecture fields unless an explicit Stage-2 override is given.
        plane_order_mode = plane_order_mode or _option_value(
            source_command, "--plane_order_mode"
        )
        plane_pair_decoder = plane_pair_decoder or _option_value(
            source_command, "--plane_pair_decoder"
        )
        global_feature_mode = global_feature_mode or _option_value(
            source_command, "--global_feature_mode"
        )
    plane_order_mode = plane_order_mode or "fixed"
    plane_pair_decoder = plane_pair_decoder or "joint_pair"
    global_feature_mode = global_feature_mode or "none"
    if plane_order_mode not in {"fixed", "learned"}:
        raise ValueError(f"Unsupported plane_order_mode={plane_order_mode!r}.")
    if plane_pair_decoder not in {"cascade", "joint_pair"}:
        raise ValueError(f"Unsupported plane_pair_decoder={plane_pair_decoder!r}.")
    if global_feature_mode not in {"none", "f1", "f1f2"}:
        raise ValueError(f"Unsupported global_feature_mode={global_feature_mode!r}.")

    if source_command is None:
        command = [str(python or PYTHON), "-u", str(train_script or TRAIN_SCRIPT)]
    else:
        command = sanitize_stage1_command(source_command)
        command[0] = str(python or PYTHON)
        command[1] = str(train_script or TRAIN_SCRIPT)

    experiment_name = f"{tag}_resource_joint"
    fixed = {
        "--env_name": "HKBZ",
        "--scenario_name": "simple",
        "--algorithm_name": "gnn_mappo",
        "--experiment_name": experiment_name,
        "--ac_config": str(ac_config or AC_CONFIG),
        "--env_config": str(env_config or ENV_CONFIG),
        "--training_stage": CANONICAL_STAGE,
        "--resource_policy": "drl",
        "--checkpoint_dir": source["path"],
        "--seed": int(seed),
        "--device_bc_pretrain_epochs": int(bc_epochs),
        "--device_bc_min_labels_per_epoch": int(bc_min_labels),
        "--device_bc_min_rollouts_per_epoch": int(bc_min_rollouts),
        "--device_bc_max_rollouts_per_epoch": int(bc_max_rollouts),
        "--device_bc_lr": float(bc_lr),
        "--num_episodes": int(ppo_epochs),
        "--ppo_epoch": int(ppo_epoch),
        # Runner interprets both freeze counters as initial PPO epochs.  Set
        # them to the full run length to make the all-epoch freeze auditable.
        "--gnn_freeze_epochs": int(ppo_epochs),
        "--plane_freeze_epochs": int(ppo_epochs),
        "--plane_order_mode": plane_order_mode,
        "--plane_pair_decoder": plane_pair_decoder,
        "--global_feature_mode": global_feature_mode,
        # Stage 2 trains independent role-balanced resource ratios.  The
        # Stage-1 joint-team ratio intentionally aggregates plane actions only
        # and would give frozen-plane loss no resource-actor gradient.
        "--hindsight_reward_mode": "team_cmax",
        "--hindsight_cmax_coef": 0.0,
        "--hindsight_shaping_coef": 0.0,
        "--hindsight_terminal_cmax_coef": 1.0,
        "--device_lookahead_safety_margin": 60.0,
    }
    for flag, value in fixed.items():
        set_option(command, flag, value)

    # Evaluation is needed for the runner's Best checkpoint lineage.  It is a
    # plain switch and does not re-enable any Stage-1 regularizer.
    set_switch(command, "--use_eval", True)
    set_switch(command, "--device_lookahead_dispatch", True)
    set_switch(command, "--strict_checkpoint_contract", True)
    set_option(command, "--eval_interval", 1)

    # A positive Stage-2 contract must not accidentally inherit any old
    # recovery/teacher flags from a source command.  The parser's defaults
    # are intentionally the desired values: device_bc_train_gnn=False and
    # device_bc_reset_optim=True.  Their explicit values are recorded in the
    # manifest contract because argparse has no ``--foo=false`` spelling for
    # store_true/store_false options.
    for flag in (
        "--resume_stage1",
        "--reset_optimizers_on_resume",
        "--selection_checkpoint_dir",
        "--device_bc_train_gnn",
        "--no_device_bc_reset_optim",
        "--no_device_bc_save",
        "--joint_team_ppo",
        "--central_team_critic",
    ):
        remove_option(command, flag, takes_value=False)
    for flag in STAGE1_ONLY_OPTIONS:
        # Some options take values, while switches such as hard_gate do not.
        remove_option(command, flag, takes_value=flag not in {"--bc_reference_hard_gate"})
    return command


def stage2_contract(
    *,
    source: Mapping[str, str],
    command: Sequence[str],
    bc_epochs: int,
    ppo_epochs: int,
    ppo_epoch: int,
    plane_order_mode: str,
    plane_pair_decoder: str,
    global_feature_mode: str,
) -> dict[str, Any]:
    from onpolicy.utils.checkpoint_contract import stage1_observation_metadata

    observation = stage1_observation_metadata(global_feature_mode)
    return {
        "from": STAGE1_KIND,
        "to": CANONICAL_STAGE,
        "resource_policy": "drl",
        "device_lookahead_dispatch": True,
        "device_lookahead_safety_margin": 60.0,
        "resource_request_time_semantics": (
            "negative_lead_time_for_lookahead_nonnegative_wait_for_blocking"
        ),
        "checkpoint_contract": "strict_stage1_m2_protected_plane_shared",
        "strict_checkpoint_contract": True,
        "environment_semantics_version": observation[
            "environment_semantics_version"
        ],
        "observation_schema_id": observation["observation_schema_id"],
        "source_m2_checkpoint": dict(source),
        "device_bc_pretrain_epochs": int(bc_epochs),
        "ppo_epochs": int(ppo_epochs),
        "ppo_epoch": int(ppo_epoch),
        "device_bc_train_gnn": False,
        "device_bc_reset_optim": True,
        "device_bc_save": True,
        "hindsight_reward_mode": "team_cmax",
        "joint_team_ppo": False,
        "central_team_critic": False,
        "ppo_ratio_scope": "role_balanced_resource_actions",
        "gnn_freeze_epochs": int(ppo_epochs),
        "plane_freeze_epochs": int(ppo_epochs),
        "freeze_scope": "shared_encoder_and_plane_actor_for_every_ppo_epoch",
        "stage1_only_regularizers_cleared": True,
        "stage1_recovery_switches_cleared": True,
        "checkpoint_DeviceBC_scope": "internal_resource_bc_warmup_boundary_only",
        "shard_resume": "unsupported; rerun_warmup_from_verified_m2",
        "plane_order_mode": str(plane_order_mode),
        "plane_pair_decoder": str(plane_pair_decoder),
        "global_feature_mode": str(global_feature_mode),
        "command": list(command),
        "command_text": shlex.join(list(command)),
    }


def _base_manifest(
    *,
    run_tag: str,
    source: Mapping[str, str],
    command: Sequence[str] | None = None,
    contract: Mapping[str, Any] | None = None,
    status: str = "registered",
    artifact_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    tag = validate_run_tag(run_tag)
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline": "hkbz_two_stage",
        "run_tag": tag,
        "status": status,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "training_stage": CANONICAL_STAGE,
        "source_kind": STAGE1_KIND,
        "source_m2": dict(source),
        # Flat aliases make shell/audit consumers independent of nested JSON
        # details and bind the manifest to the exact source artifact.
        "source_m2_path": source["path"],
        "source_m2_sha256": source["sha256"],
        "migration_contract": dict(contract or {
            "from": STAGE1_KIND,
            "to": CANONICAL_STAGE,
            "checkpoint_DeviceBC_scope": "internal_only",
            "shard_resume": "unsupported",
        }),
        "operation_log": [
            {
                "at": utc_now(),
                "operation": "register_stage1_m2",
                "status": "accepted",
                "source_m2_path": source["path"],
                "source_m2_sha256": source["sha256"],
            }
        ],
    }
    if command is not None:
        payload["command"] = list(command)
        payload["command_text"] = shlex.join(list(command))
    if artifact_dir:
        models = Path(artifact_dir).expanduser().resolve()
        if models.name != "models":
            models = models / "models"
        payload["artifacts"] = {
            "models_dir": str(models),
            "warmup_checkpoint": str(models / "checkpoint_DeviceBC.pt"),
            "final_best_checkpoint": str(models / "checkpoint_Best.pt"),
            "final_last_checkpoint": str(models / "checkpoint_Last.pt"),
            "run_status_path": str(models.parent / "run_status.json"),
        }
        payload["run_status_path"] = str(models.parent / "run_status.json")
    else:
        payload["artifacts"] = {
            "models_dir": None,
            "warmup_checkpoint": None,
            "final_best_checkpoint": None,
            "final_last_checkpoint": None,
            "run_status_path": None,
            "models_glob": str(
                DEFAULT_RESULTS_ROOT / "<experiment_name>" / "run*" / "models"
            ),
        }
    return payload


def _resource_joint_target_state(
    semantics: Mapping[str, str],
) -> Mapping[str, Any]:
    """Build the current Stage-2 network once for strict shape preflight."""

    from types import SimpleNamespace

    import torch
    import yaml

    from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import (
        GNN_MAPPOPolicy,
    )

    with AC_CONFIG.open("r", encoding="utf-8") as stream:
        ac_config = yaml.safe_load(stream)
    with ENV_CONFIG.open("r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)
    args = SimpleNamespace(
        lr=1e-5,
        critic_lr=1e-4,
        opti_eps=1e-5,
        weight_decay=0.0,
        anneal_original=0.3,
        anneal_final=0.3,
        tau_anneal_epochs=0,
        max_agent_num=int(env_config["n_agents"]),
        max_device_num=int(env_config["max_device_num"]),
        resource_policy="drl",
        shared_actor_lr_scale=0.1,
        plane_actor_lr_scale=1.0,
        device_actor_lr_scale=1.0,
        transporter_actor_lr_scale=1.0,
        plane_order_mode=semantics["plane_order_mode"],
        plane_pair_decoder=semantics["plane_pair_decoder"],
        central_team_critic=False,
    )
    policy = GNN_MAPPOPolicy(args, ac_config, device=torch.device("cpu"))
    return policy.ac.state_dict()


def _validate_stage1_handoff_checkpoint_payload(
    checkpoint_path: Path,
    *,
    semantics: Mapping[str, str],
    entry: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate checkpoint metadata and protected tensor compatibility."""

    from onpolicy.utils.checkpoint_contract import (
        validate_stage1_checkpoint_contract,
    )
    from onpolicy.utils.training_stage import validate_stage1_m2_checkpoint

    checkpoint = _load_checkpoint(checkpoint_path)
    observation = validate_stage1_checkpoint_contract(
        checkpoint,
        global_feature_mode=semantics["global_feature_mode"],
        plane_order_mode=semantics["plane_order_mode"],
        plane_pair_decoder=semantics["plane_pair_decoder"],
        strict_metadata=True,
    )
    transition = validate_stage1_m2_checkpoint(
        checkpoint,
        target_state,
        plane_order_mode=semantics["plane_order_mode"],
        plane_pair_decoder=semantics["plane_pair_decoder"],
        global_feature_mode=semantics["global_feature_mode"],
    )
    if str(checkpoint.get("stage", "")) != "best_eval":
        raise ValueError(
            f"Stage-1 hand-off must use a validation Best checkpoint: {checkpoint_path}"
        )
    expected_episode = int(entry.get("selected_episode", -1))
    if expected_episode <= 0 or int(checkpoint.get("episodes", -1)) != expected_episode:
        raise ValueError(
            "Stage-1 hand-off selected_episode disagrees with checkpoint: "
            f"entry={expected_episode}, checkpoint={checkpoint.get('episodes')}."
        )
    for field, checkpoint_field in (
        ("selection_score", "selection_score"),
        ("validation_raw_makespan", "eval_raw_makespan"),
    ):
        expected = float(entry.get(field, math.nan))
        observed = float(checkpoint.get(checkpoint_field, math.nan))
        if not (
            math.isfinite(expected)
            and math.isfinite(observed)
            and math.isclose(expected, observed, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError(
                f"Stage-1 hand-off {field} mismatch: "
                f"entry={expected!r}, checkpoint={observed!r}."
            )
    protected_sha = str(transition["source_summary"]["sha256"])
    expected_protected_sha = str(
        entry.get("protected_parameter_sha256", "")
    ).lower()
    if expected_protected_sha != protected_sha:
        raise ValueError(
            "Stage-1 hand-off protected tensor digest mismatch: "
            f"expected={expected_protected_sha!r}, observed={protected_sha!r}."
        )
    return {
        "training_stage": transition["training_stage"],
        "protected_parameter_sha256": protected_sha,
        "protected_parameter_count": int(
            transition["source_summary"]["count"]
        ),
        "environment_semantics_version": observation[
            "environment_semantics_version"
        ],
        "observation_schema_id": observation["observation_schema_id"],
        "selected_episode": expected_episode,
        "selection_score": float(entry["selection_score"]),
        "validation_raw_makespan": float(entry["validation_raw_makespan"]),
    }


def load_stage1_handoff(
    path: str | os.PathLike[str] = DEFAULT_STAGE1_HANDOFF,
) -> dict[str, Any]:
    """Load and verify the repository's immutable three-seed M2 hand-off."""

    handoff_path = Path(path).expanduser().resolve()
    payload = read_json(handoff_path)
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in SUPPORTED_HANDOFF_SCHEMA_VERSIONS:
        raise ValueError("Unsupported Stage-1 M2 hand-off schema.")
    if payload.get("stage1_status") != "closed":
        raise ValueError("Stage-1 hand-off must declare stage1_status='closed'.")
    if payload.get("training_stage") != "plane_pretrain":
        raise ValueError(
            "Stage-1 hand-off must identify the successful plane_pretrain stage."
        )
    if payload.get("decision", {}).get("resource_joint_transition_authorized") is not True:
        raise ValueError("Stage-1 hand-off does not authorize resource_joint.")

    semantics = payload.get("semantic_contract")
    if not isinstance(semantics, Mapping):
        raise ValueError("Stage-1 hand-off is missing semantic_contract.")
    allowed_semantics = {
        "plane_order_mode": {"fixed", "learned"},
        "plane_pair_decoder": {"cascade", "joint_pair"},
        "global_feature_mode": {"none", "f1", "f1f2"},
    }
    normalized_semantics: dict[str, str] = {}
    for field, allowed in allowed_semantics.items():
        value = str(semantics.get(field, ""))
        if value not in allowed:
            raise ValueError(
                f"Invalid Stage-1 hand-off semantic {field}={value!r}."
            )
        normalized_semantics[field] = value
    if schema_version >= 2:
        from onpolicy.utils.checkpoint_contract import stage1_observation_metadata

        expected_observation = stage1_observation_metadata(
            normalized_semantics["global_feature_mode"]
        )
        for field in (
            "environment_semantics_version",
            "observation_schema_id",
        ):
            value = str(semantics.get(field, ""))
            if value != str(expected_observation[field]):
                raise ValueError(
                    f"Stage-1 hand-off semantic {field} is stale: "
                    f"handoff={value!r}, current={expected_observation[field]!r}."
                )
            normalized_semantics[field] = value

    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("Stage-1 hand-off has no checkpoints.")
    normalized_checkpoints: dict[str, dict[str, Any]] = {}
    target_state = (
        _resource_joint_target_state(normalized_semantics)
        if schema_version >= 2
        else None
    )
    for raw_seed, raw_entry in checkpoints.items():
        seed = str(int(raw_seed))
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Stage-1 hand-off seed {seed} is not a mapping.")
        raw_path = Path(str(raw_entry.get("path", ""))).expanduser()
        checkpoint_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Stage-1 M2 seed {seed} checkpoint is missing: {checkpoint_path}"
            )
        expected_sha = str(raw_entry.get("sha256", "")).lower()
        if len(expected_sha) != 64:
            raise ValueError(
                f"Stage-1 M2 seed {seed} requires a 64-character SHA256."
            )
        observed_sha = sha256_file(checkpoint_path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"Stage-1 M2 seed {seed} SHA256 mismatch: "
                f"expected={expected_sha}, observed={observed_sha}."
            )
        expected_size = int(raw_entry.get("size_bytes", -1))
        observed_size = checkpoint_path.stat().st_size
        if expected_size != observed_size:
            raise ValueError(
                f"Stage-1 M2 seed {seed} size mismatch: "
                f"expected={expected_size}, observed={observed_size}."
            )
        raw_command_path = Path(
            str(raw_entry.get("source_command_path", ""))
        ).expanduser()
        command_path = (
            raw_command_path
            if raw_command_path.is_absolute()
            else ROOT / raw_command_path
        ).resolve()
        if not command_path.is_file():
            raise FileNotFoundError(
                f"Stage-1 M2 seed {seed} source command is missing: "
                f"{command_path}"
            )
        expected_command_sha = str(
            raw_entry.get("source_command_sha256", "")
        ).lower()
        observed_command_sha = sha256_file(command_path)
        if observed_command_sha != expected_command_sha:
            raise ValueError(
                f"Stage-1 M2 seed {seed} source-command SHA256 mismatch: "
                f"expected={expected_command_sha}, "
                f"observed={observed_command_sha}."
            )
        expected_command_size = int(
            raw_entry.get("source_command_size_bytes", -1)
        )
        observed_command_size = command_path.stat().st_size
        if expected_command_size != observed_command_size:
            raise ValueError(
                f"Stage-1 M2 seed {seed} source-command size mismatch: "
                f"expected={expected_command_size}, "
                f"observed={observed_command_size}."
            )
        source_command_key = raw_entry.get("source_command_key")
        if schema_version >= 2 and not str(source_command_key or "").strip():
            raise ValueError(
                f"Stage-1 hand-off seed {seed} requires source_command_key."
            )
        source_command = _load_source_command(
            command_path,
            None if source_command_key is None else str(source_command_key),
        )
        if _option_value(source_command or (), "--seed") != seed:
            raise ValueError(
                f"Stage-1 M2 seed {seed} source command has a different seed."
            )
        for field, flag in (
            ("plane_order_mode", "--plane_order_mode"),
            ("plane_pair_decoder", "--plane_pair_decoder"),
            ("global_feature_mode", "--global_feature_mode"),
        ):
            if _option_value(source_command or (), flag) != normalized_semantics[field]:
                raise ValueError(
                    f"Stage-1 M2 seed {seed} source command disagrees with "
                    f"semantic_contract.{field}."
                )
        checkpoint_contract = None
        if schema_version >= 2:
            checkpoint_contract = _validate_stage1_handoff_checkpoint_payload(
                checkpoint_path,
                semantics=normalized_semantics,
                entry=raw_entry,
                target_state=target_state or {},
            )
        normalized_checkpoints[seed] = {
            "path": str(checkpoint_path),
            "sha256": observed_sha,
            "size_bytes": observed_size,
            "source_command_path": str(command_path),
            "source_command_sha256": observed_command_sha,
            "source_command_size_bytes": observed_command_size,
            "source_command_key": (
                None if source_command_key is None else str(source_command_key)
            ),
            "checkpoint_contract": checkpoint_contract,
        }

    return {
        **payload,
        "handoff_path": str(handoff_path),
        "semantic_contract": normalized_semantics,
        "checkpoints": normalized_checkpoints,
    }


def resolve_stage1_handoff_checkpoint(
    seed: int,
    path: str | os.PathLike[str] = DEFAULT_STAGE1_HANDOFF,
) -> tuple[str, dict[str, str]]:
    """Resolve one verified M2 checkpoint and its architecture semantics."""

    entry, semantics = resolve_stage1_handoff_entry(seed, path)
    return (
        str(entry["path"]),
        semantics,
    )


def resolve_stage1_handoff_entry(
    seed: int,
    path: str | os.PathLike[str] = DEFAULT_STAGE1_HANDOFF,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve one verified M2 checkpoint plus its source command."""

    handoff = load_stage1_handoff(path)
    seed_key = str(int(seed))
    if seed_key not in handoff["checkpoints"]:
        available = ", ".join(sorted(handoff["checkpoints"]))
        raise ValueError(
            f"Stage-1 hand-off has no seed {seed_key}; available seeds: {available}."
        )
    return (
        dict(handoff["checkpoints"][seed_key]),
        dict(handoff["semantic_contract"]),
    )


def _append_operation(manifest: MutableMapping[str, Any], operation: str, **fields: Any) -> None:
    operations = manifest.setdefault("operation_log", [])
    if not isinstance(operations, list):
        operations = []
        manifest["operation_log"] = operations
    operations.append({"at": utc_now(), "operation": operation, **fields})
    manifest["updated_at"] = utc_now()


def _assert_manifest_source(manifest: Mapping[str, Any], source: Mapping[str, str]) -> None:
    observed = manifest.get("source_m2")
    observed_path = manifest.get("source_m2_path")
    observed_sha = manifest.get("source_m2_sha256")
    if isinstance(observed, Mapping):
        observed_path = observed.get("path", observed_path)
        observed_sha = observed.get("sha256", observed_sha)
    if str(observed_path) != str(source["path"]) or str(observed_sha) != str(source["sha256"]):
        raise ValueError(
            "Manifest source M2 identity mismatch; refusing to mix checkpoints "
            f"(manifest={observed_path!r}/{observed_sha!r}, "
            f"requested={source['path']!r}/{source['sha256']!r})."
        )


def register_stage1_m2(
    source_m2: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str],
    run_tag: str,
) -> dict[str, Any]:
    """Register an external, completed Stage-1 M2 artifact exactly once."""

    source = source_checkpoint_metadata(source_m2)
    target = Path(manifest_path).expanduser().resolve()
    if target.exists():
        existing = read_json(target)
        _assert_manifest_source(existing, source)
        if str(existing.get("run_tag")) != validate_run_tag(run_tag):
            raise ValueError("Existing manifest run_tag differs from requested run_tag.")
        # Idempotent registration is useful for a shell retry; it does not
        # create a second command or alter the source identity.
        return existing
    payload = _base_manifest(run_tag=run_tag, source=source)
    atomic_json(target, payload, refuse_existing=True)
    return payload


def _artifact_manifest_for_command(
    payload: MutableMapping[str, Any],
    *,
    run_tag: str,
    artifact_dir: str | os.PathLike[str] | None,
) -> None:
    artifacts = payload.setdefault("artifacts", {})
    if artifact_dir:
        models = Path(artifact_dir).expanduser().resolve()
        if models.name != "models":
            models = models / "models"
        artifacts.update({
            "models_dir": str(models),
            "warmup_checkpoint": str(models / "checkpoint_DeviceBC.pt"),
            "final_best_checkpoint": str(models / "checkpoint_Best.pt"),
            "final_last_checkpoint": str(models / "checkpoint_Last.pt"),
            "run_status_path": str(models.parent / "run_status.json"),
        })
        payload["run_status_path"] = str(models.parent / "run_status.json")
        return
    experiment_name = f"{validate_run_tag(run_tag)}_resource_joint"
    artifacts["experiment_name"] = experiment_name
    artifacts["models_glob"] = str(
        DEFAULT_RESULTS_ROOT / experiment_name / "run*" / "models"
    )
    models_glob = str(DEFAULT_RESULTS_ROOT / experiment_name / "run*" / "models")
    artifacts["warmup_checkpoint_pattern"] = models_glob + "/checkpoint_DeviceBC.pt"
    artifacts["final_best_checkpoint_pattern"] = models_glob + "/checkpoint_Best.pt"
    artifacts["final_last_checkpoint_pattern"] = models_glob + "/checkpoint_Last.pt"
    artifacts["run_status_pattern"] = models_glob.removesuffix("/models") + "/run_status.json"


def _mark_lineage_pending(payload: MutableMapping[str, Any]) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return
    lineage = {}
    for role, field in (
        ("warmup", "warmup_checkpoint"),
        ("best", "final_best_checkpoint"),
        ("last", "final_last_checkpoint"),
    ):
        path = artifacts.get(field)
        pattern = artifacts.get(
            {
                "warmup": "warmup_checkpoint_pattern",
                "best": "final_best_checkpoint_pattern",
                "last": "final_last_checkpoint_pattern",
            }[role]
        )
        lineage[role] = {
            "status": "pending",
            "path": str(path) if path else None,
            "path_pattern": str(pattern) if pattern else None,
        }
    run_status_path = artifacts.get(
        "run_status_path",
        artifacts.get("run_status", payload.get("run_status_path")),
    )
    if isinstance(run_status_path, Mapping):
        run_status_path = run_status_path.get("path")
    run_status_pattern = artifacts.get("run_status_pattern")
    lineage["run_status"] = {
        "status": "pending",
        "path": str(run_status_path) if run_status_path else None,
        "path_pattern": str(run_status_pattern) if run_status_pattern else None,
    }
    payload["lineage_status"] = "pending"
    payload["lineage"] = lineage


def plan_stage2(
    source_m2: str | os.PathLike[str],
    *,
    run_tag: str,
    manifest_path: str | os.PathLike[str],
    dry_run: bool = False,
    artifact_dir: str | os.PathLike[str] | None = None,
    source_command: Sequence[str] | None = None,
    seed: int = 1,
    bc_epochs: int = DEFAULT_BC_EPOCHS,
    ppo_epochs: int = DEFAULT_PPO_EPOCHS,
    ppo_epoch: int = DEFAULT_PPO_EPOCH,
    bc_min_labels: int = DEFAULT_BC_MIN_LABELS,
    bc_min_rollouts: int = DEFAULT_BC_MIN_ROLLOUTS,
    bc_max_rollouts: int = DEFAULT_BC_MAX_ROLLOUTS,
    bc_lr: float = 0.0,
    plane_order_mode: str | None = None,
    plane_pair_decoder: str | None = None,
    global_feature_mode: str | None = None,
) -> dict[str, Any]:
    """Register M2 and atomically publish one canonical Stage-2 command."""

    source = source_checkpoint_metadata(source_m2)
    tag = validate_run_tag(run_tag)
    command = build_stage2_command(
        source["path"],
        run_tag=tag,
        seed=seed,
        bc_epochs=bc_epochs,
        ppo_epochs=ppo_epochs,
        ppo_epoch=ppo_epoch,
        bc_min_labels=bc_min_labels,
        bc_min_rollouts=bc_min_rollouts,
        bc_max_rollouts=bc_max_rollouts,
        bc_lr=bc_lr,
        plane_order_mode=plane_order_mode,
        plane_pair_decoder=plane_pair_decoder,
        global_feature_mode=global_feature_mode,
        source_command=source_command,
    )
    plane_order_mode = plane_order_mode or _option_value(command, "--plane_order_mode") or "fixed"
    plane_pair_decoder = plane_pair_decoder or _option_value(command, "--plane_pair_decoder") or "joint_pair"
    global_feature_mode = global_feature_mode or _option_value(command, "--global_feature_mode") or "none"
    contract = stage2_contract(
        source=source,
        command=command,
        bc_epochs=bc_epochs,
        ppo_epochs=ppo_epochs,
        ppo_epoch=ppo_epoch,
        plane_order_mode=plane_order_mode,
        plane_pair_decoder=plane_pair_decoder,
        global_feature_mode=global_feature_mode,
    )
    target = Path(manifest_path).expanduser().resolve()
    if target.exists():
        payload = read_json(target)
        _assert_manifest_source(payload, source)
        if str(payload.get("run_tag")) != tag:
            raise ValueError("Existing manifest run_tag differs from requested run_tag.")
        status = str(payload.get("status", "registered"))
        if status in {"running", "completed"}:
            raise FileExistsError(
                f"Manifest already owns a {status} Stage-2 command: {target}"
            )
        _append_operation(payload, "plan_stage2", status="dry_run" if dry_run else "planned")
    else:
        payload = _base_manifest(
            run_tag=tag,
            source=source,
            command=command,
            contract=contract,
            status="dry_run" if dry_run else "planned",
            artifact_dir=artifact_dir,
        )
        _append_operation(payload, "plan_stage2", status=payload["status"])
    payload["status"] = "dry_run" if dry_run else "planned"
    payload["training_stage"] = CANONICAL_STAGE
    payload["command"] = list(command)
    payload["command_text"] = shlex.join(command)
    payload["migration_contract"] = contract
    payload["source_m2"] = dict(source)
    payload["source_m2_path"] = source["path"]
    payload["source_m2_sha256"] = source["sha256"]
    _artifact_manifest_for_command(payload, run_tag=tag, artifact_dir=artifact_dir)
    _mark_lineage_pending(payload)
    payload["updated_at"] = utc_now()
    atomic_json(target, payload)
    return payload


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load torch checkpoints without importing torch during planning."""

    try:
        import torch

        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch versions before weights_only
            value = torch.load(path, map_location="cpu")
    except Exception as torch_error:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as json_error:
            raise ValueError(
                f"Cannot load checkpoint lineage payload {path}: "
                f"torch={torch_error}; json={json_error}"
            ) from torch_error
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint lineage payload must be a mapping: {path}")
    return value


FINAL_RESOURCE_JOINT_PHASES = frozenset(
    {"resource_joint_ppo", "resource_joint_completed"}
)


def _summary_digest(value: object, label: str) -> str:
    """Require a non-empty protected/resource summary digest."""

    if not isinstance(value, Mapping):
        raise ValueError(f"Missing {label} digest evidence.")
    digest = str(value.get("sha256", "")).strip()
    if not digest:
        raise ValueError(f"Missing {label} sha256 digest evidence.")
    # Runner summaries contain a positive count.  Keep the check optional for
    # small audit fixtures that only carry the digest, while rejecting an
    # explicitly empty scope.
    if value.get("count") is not None and int(value["count"]) <= 0:
        raise ValueError(f"{label} digest evidence has an empty scope.")
    return digest


def _require_same_digest(left: object, right: object, label: str) -> str:
    left_digest = _summary_digest(left, f"{label} (left)")
    right_digest = _summary_digest(right, f"{label} (right)")
    if left_digest != right_digest:
        raise ValueError(
            f"{label} changed unexpectedly: {left_digest} != {right_digest}."
        )
    return left_digest


def _require_different_digest(left: object, right: object, label: str) -> tuple[str, str]:
    left_digest = _summary_digest(left, f"{label} (before)")
    right_digest = _summary_digest(right, f"{label} (after)")
    if left_digest == right_digest:
        raise ValueError(f"{label} has no bitwise actor update evidence.")
    return left_digest, right_digest


def _checkpoint_source_identity(checkpoint: Mapping[str, Any]) -> tuple[str, str]:
    observed_source = checkpoint.get("source_m2_checkpoint")
    observed_path = checkpoint.get("source_m2_path")
    observed_sha = checkpoint.get("source_m2_sha256")
    if isinstance(observed_source, Mapping):
        observed_path = observed_source.get("path", observed_path)
        observed_sha = observed_source.get("sha256", observed_sha)
    return str(observed_path or ""), str(observed_sha or "")


def _validate_warmup_evidence(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    phase = str(checkpoint.get("phase", checkpoint.get("stage", ""))).lower()
    if "resource_bc_warmup_completed" not in phase:
        raise ValueError(f"Stage-2 warm-up checkpoint has unexpected phase={phase!r}.")
    labels = int(checkpoint.get("resource_bc_total_labels", 0))
    if labels <= 0:
        raise ValueError("Stage-2 warm-up checkpoint has no positive BC label evidence.")
    if checkpoint.get("resource_bc_optimizer_reset") is not True:
        raise ValueError("Stage-2 warm-up checkpoint did not record fresh PPO optimizers.")
    protected_digest = _require_same_digest(
        checkpoint.get("protected_parameter_summary_before_bc"),
        checkpoint.get("protected_parameter_summary_after_bc"),
        "Stage-2 warm-up protected parameters",
    )
    actor_before, actor_after = _require_different_digest(
        checkpoint.get("resource_actor_summary_before_bc"),
        checkpoint.get("resource_actor_summary_after_bc"),
        "Stage-2 warm-up resource actor",
    )
    current_digest = _summary_digest(
        checkpoint.get("protected_parameter_summary_after_bc"),
        "Stage-2 warm-up protected-after-BC",
    )
    if current_digest != protected_digest:
        raise ValueError("Stage-2 warm-up protected digest is internally inconsistent.")
    return {
        "phase": phase,
        "protected_sha256": protected_digest,
        "resource_actor_before_bc_sha256": actor_before,
        "resource_actor_after_bc_sha256": actor_after,
        "resource_bc_total_labels": str(labels),
    }


def _validate_final_checkpoint_evidence(
    checkpoint: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, str | None]:
    phase = str(checkpoint.get("phase", checkpoint.get("stage", ""))).lower()
    if phase not in FINAL_RESOURCE_JOINT_PHASES:
        raise ValueError(
            f"Stage-2 {role} checkpoint has non-PPO/final phase={phase!r}."
        )
    labels = int(checkpoint.get("resource_bc_total_labels", 0))
    if labels <= 0:
        raise ValueError(f"Stage-2 {role} checkpoint has no positive BC label evidence.")
    if checkpoint.get("resource_bc_optimizer_reset") is not True:
        raise ValueError(f"Stage-2 {role} checkpoint did not record fresh PPO optimizers.")
    protected_before_bc = checkpoint.get("protected_parameter_summary_before_bc")
    protected_after_bc = checkpoint.get("protected_parameter_summary_after_bc")
    protected_current = checkpoint.get("protected_parameter_summary")
    protected_digest = _require_same_digest(
        protected_before_bc,
        protected_after_bc,
        f"Stage-2 {role} protected parameters during BC",
    )
    _require_same_digest(
        protected_after_bc,
        protected_current,
        f"Stage-2 {role} protected parameters at final",
    )
    protected_after_ppo = checkpoint.get("protected_parameter_summary_after_ppo")
    if protected_after_ppo is not None:
        _require_same_digest(
            protected_current,
            protected_after_ppo,
            f"Stage-2 {role} protected parameters after PPO",
        )
    actor_before_bc, actor_after_bc = _require_different_digest(
        checkpoint.get("resource_actor_summary_before_bc"),
        checkpoint.get("resource_actor_summary_after_bc"),
        f"Stage-2 {role} resource actor during BC",
    )
    actor_before_ppo = checkpoint.get("resource_actor_summary_before_ppo")
    actor_after_ppo = checkpoint.get("resource_actor_summary_after_ppo")
    actor_before_ppo_sha = None
    actor_after_ppo_sha = None
    if actor_before_ppo is not None and actor_after_ppo is None and role == "best":
        # Best may be the deterministic post-BC baseline captured after the
        # runner enters resource_joint_ppo but before the first PPO update.
        actor_before_ppo_sha = _summary_digest(
            actor_before_ppo,
            f"Stage-2 {role} resource actor before PPO",
        )
    elif actor_before_ppo is not None or actor_after_ppo is not None:
        if actor_before_ppo is None or actor_after_ppo is None:
            raise ValueError(
                f"Stage-2 {role} has incomplete PPO resource actor evidence."
            )
        actor_before_ppo_sha, actor_after_ppo_sha = _require_different_digest(
            actor_before_ppo,
            actor_after_ppo,
            f"Stage-2 {role} resource actor during PPO",
        )
    if role == "last" and actor_before_ppo_sha is None:
        raise ValueError("Stage-2 Last checkpoint lacks PPO resource actor update evidence.")
    return {
        "phase": phase,
        "protected_sha256": protected_digest,
        "resource_actor_before_bc_sha256": actor_before_bc,
        "resource_actor_after_bc_sha256": actor_after_bc,
        "resource_actor_before_ppo_sha256": actor_before_ppo_sha,
        "resource_actor_after_ppo_sha256": actor_after_ppo_sha,
        "resource_bc_total_labels": str(labels),
    }


def validate_checkpoint_lineage(
    checkpoint_path: str | os.PathLike[str],
    source_m2: Mapping[str, str],
    *,
    role: str,
) -> dict[str, Any]:
    """Validate a warm-up/Best/Last checkpoint's Stage-2 source lineage."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage-2 {role} checkpoint: {path}")
    if role not in {"warmup", "best", "last"}:
        raise ValueError(f"Unknown Stage-2 checkpoint role: {role}")
    checkpoint = _load_checkpoint(path)
    stage = str(checkpoint.get("training_stage", checkpoint.get("stage", ""))).strip().lower()
    if stage != CANONICAL_STAGE:
        raise ValueError(
            f"Stage-2 {role} checkpoint has training_stage={stage!r}; "
            f"expected {CANONICAL_STAGE!r}."
        )
    observed_path, observed_sha = _checkpoint_source_identity(checkpoint)
    if str(observed_path) != str(source_m2["path"]):
        raise ValueError(
            f"Stage-2 {role} source path does not match registered M2: "
            f"{observed_path!r} != {source_m2['path']!r}"
        )
    if str(observed_sha) != str(source_m2["sha256"]):
        raise ValueError(
            f"Stage-2 {role} source SHA256 does not match registered M2: "
            f"{observed_sha!r} != {source_m2['sha256']!r}"
        )
    if role == "warmup":
        evidence = _validate_warmup_evidence(checkpoint)
    else:
        evidence = _validate_final_checkpoint_evidence(checkpoint, role=role)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "role": role,
        "training_stage": stage,
        **evidence,
        "source_m2_path": str(observed_path),
        "source_m2_sha256": str(observed_sha),
        "resource_bc_optimizer_reset": checkpoint.get("resource_bc_optimizer_reset"),
    }


def _discover_models_dir(manifest: Mapping[str, Any]) -> Path | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    explicit = artifacts.get("models_dir")
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        return path if path.is_dir() else None
    glob_pattern = artifacts.get("models_glob")
    if not glob_pattern:
        return None
    pattern = str(glob_pattern)
    parent = Path(pattern)
    # Locate the final run* directory without globbing through arbitrary
    # user paths; the pattern is controller-generated and always absolute.
    marker = "/run*/models"
    if marker not in pattern:
        return None
    root_text, _ = pattern.split(marker, 1)
    base = Path(root_text)
    if not base.is_dir():
        return None
    matches = sorted(
        (path for path in base.glob("run*/models") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _discover_run_status_path(models_dir: Path) -> Path | None:
    candidates = (
        models_dir.parent / "run_status.json",
        models_dir / "run_status.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def _validate_run_status_evidence(
    status_path: Path,
    source_m2: Mapping[str, str],
    *,
    last_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    status = read_json(status_path)
    if str(status.get("status", "")).lower() != "completed":
        raise ValueError(
            f"Stage-2 run_status.json is not completed: {status.get('status')!r}."
        )
    if str(status.get("training_stage", "")).strip().lower() != CANONICAL_STAGE:
        raise ValueError(
            "Stage-2 run_status.json has a non-canonical training_stage: "
            f"{status.get('training_stage')!r}."
        )
    phase = str(status.get("phase", "")).strip().lower()
    if phase != "resource_joint_completed":
        raise ValueError(
            "Stage-2 run_status.json does not record the final resource_joint "
            f"phase: {phase!r}."
        )
    observed_path, observed_sha = _checkpoint_source_identity(status)
    if observed_path != str(source_m2["path"]):
        raise ValueError(
            "Stage-2 run_status.json source M2 path mismatch: "
            f"{observed_path!r} != {source_m2['path']!r}."
        )
    if observed_sha != str(source_m2["sha256"]):
        raise ValueError(
            "Stage-2 run_status.json source M2 SHA256 mismatch: "
            f"{observed_sha!r} != {source_m2['sha256']!r}."
        )
    labels = int(status.get("resource_bc_total_labels", 0))
    if labels <= 0:
        raise ValueError("Stage-2 run_status.json lacks positive BC label evidence.")
    if status.get("resource_bc_optimizer_reset") is not True:
        raise ValueError(
            "Stage-2 run_status.json lacks fresh-optimizer evidence after BC."
        )
    protected_sha = _summary_digest(
        status.get("protected_parameter_summary"),
        "Stage-2 run_status protected parameters",
    )
    if protected_sha != str(last_lineage.get("protected_sha256")):
        raise ValueError(
            "Stage-2 run_status protected digest differs from checkpoint_Last.pt."
        )
    actor_before_bc, actor_after_bc = _require_different_digest(
        status.get("resource_actor_summary_before_bc"),
        status.get("resource_actor_summary_after_bc"),
        "Stage-2 run_status resource actor during BC",
    )
    if actor_before_bc != str(last_lineage.get("resource_actor_before_bc_sha256")):
        raise ValueError(
            "Stage-2 run_status BC-before actor digest differs from checkpoint_Last.pt."
        )
    if actor_after_bc != str(last_lineage.get("resource_actor_after_bc_sha256")):
        raise ValueError(
            "Stage-2 run_status BC-after actor digest differs from checkpoint_Last.pt."
        )
    if labels != int(last_lineage.get("resource_bc_total_labels", 0)):
        raise ValueError(
            "Stage-2 run_status BC label count differs from checkpoint_Last.pt."
        )
    actor_before_ppo, actor_after_ppo = _require_different_digest(
        status.get("resource_actor_summary_before_ppo"),
        status.get("resource_actor_summary_after_ppo"),
        "Stage-2 run_status resource actor during PPO",
    )
    if actor_before_ppo != str(last_lineage.get("resource_actor_before_ppo_sha256")):
        raise ValueError(
            "Stage-2 run_status PPO-before actor digest differs from checkpoint_Last.pt."
        )
    if actor_after_ppo != str(last_lineage.get("resource_actor_after_ppo_sha256")):
        raise ValueError(
            "Stage-2 run_status PPO-after actor digest differs from checkpoint_Last.pt."
        )
    return {
        "path": str(status_path),
        "sha256": sha256_file(status_path),
        "status": "completed",
        "training_stage": CANONICAL_STAGE,
        "phase": phase,
        "source_m2_path": observed_path,
        "source_m2_sha256": observed_sha,
        "protected_sha256": protected_sha,
        "resource_actor_before_bc_sha256": actor_before_bc,
        "resource_actor_after_bc_sha256": actor_after_bc,
        "resource_actor_before_ppo_sha256": actor_before_ppo,
        "resource_actor_after_ppo_sha256": actor_after_ppo,
        "resource_bc_total_labels": labels,
        "resource_bc_optimizer_reset": True,
    }


def audit_stage2_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    require_artifacts: bool = True,
) -> dict[str, Any]:
    """Audit warm-up/Best/Last lineage and atomically persist the evidence."""

    target = Path(manifest_path).expanduser().resolve()
    payload = read_json(target)
    source = payload.get("source_m2")
    if not isinstance(source, Mapping) or not source.get("path") or not source.get("sha256"):
        raise ValueError(f"Manifest lacks source M2 identity: {target}")
    current = source_checkpoint_metadata(str(source["path"]))
    _assert_manifest_source(payload, current)
    if current["sha256"] != str(source["sha256"]):
        raise ValueError("Registered Stage-1 M2 changed since manifest creation.")
    models_dir = _discover_models_dir(payload)
    artifacts = payload.setdefault("artifacts", {})
    if models_dir is not None:
        artifacts["models_dir"] = str(models_dir)
        artifacts["warmup_checkpoint"] = str(models_dir / "checkpoint_DeviceBC.pt")
        artifacts["final_best_checkpoint"] = str(models_dir / "checkpoint_Best.pt")
        artifacts["final_last_checkpoint"] = str(models_dir / "checkpoint_Last.pt")
        discovered_status = _discover_run_status_path(models_dir)
        if discovered_status is not None:
            artifacts["run_status_path"] = str(discovered_status)
    if require_artifacts:
        preflight_status_path = artifacts.get(
            "run_status_path",
            artifacts.get("run_status", payload.get("run_status_path")),
        )
        if isinstance(preflight_status_path, Mapping):
            preflight_status_path = preflight_status_path.get("path")
        if not preflight_status_path:
            raise FileNotFoundError("Manifest has no path for Stage-2 run_status.json.")
        if not Path(str(preflight_status_path)).expanduser().resolve().is_file():
            raise FileNotFoundError(
                "Missing Stage-2 run_status.json: "
                f"{Path(str(preflight_status_path)).expanduser().resolve()}"
            )
    lineage: dict[str, Any] = {}
    for role, field in (
        ("warmup", "warmup_checkpoint"),
        ("best", "final_best_checkpoint"),
        ("last", "final_last_checkpoint"),
    ):
        path = artifacts.get(field)
        if not path:
            if require_artifacts:
                raise FileNotFoundError(f"Manifest has no path for Stage-2 {role} checkpoint.")
            lineage[role] = {"status": "pending"}
            continue
        try:
            lineage[role] = {
                "status": "validated",
                **validate_checkpoint_lineage(path, current, role=role),
            }
        except (FileNotFoundError, ValueError) as error:
            if require_artifacts:
                raise
            lineage[role] = {"status": "pending", "path": str(path), "reason": str(error)}
    if require_artifacts:
        warmup_protected = str(lineage["warmup"].get("protected_sha256"))
        best_protected = str(lineage["best"].get("protected_sha256"))
        last_protected = str(lineage["last"].get("protected_sha256"))
        if len({warmup_protected, best_protected, last_protected}) != 1:
            raise ValueError(
                "Stage-2 warm-up/Best/Last protected digests are not identical."
            )
    run_status_path = artifacts.get(
        "run_status_path",
        artifacts.get("run_status", payload.get("run_status_path")),
    )
    if isinstance(run_status_path, Mapping):
        run_status_path = run_status_path.get("path")
    run_status_evidence = None
    if run_status_path:
        status_candidate = Path(str(run_status_path)).expanduser().resolve()
        if status_candidate.is_file():
            if require_artifacts:
                run_status_evidence = _validate_run_status_evidence(
                    status_candidate,
                    current,
                    last_lineage=lineage.get("last", {}),
                )
                # Keep the lineage wrapper status distinct from the
                # run_status.json payload's own ``status=completed`` field.
                lineage["run_status"] = {**run_status_evidence, "status": "validated"}
            else:
                lineage["run_status"] = {
                    "status": "pending",
                    "path": str(status_candidate),
                }
        elif require_artifacts:
            raise FileNotFoundError(f"Missing Stage-2 run_status.json: {status_candidate}")
        else:
            lineage["run_status"] = {
                "status": "pending",
                "path": str(status_candidate),
            }
    elif require_artifacts:
        raise FileNotFoundError("Manifest has no path for Stage-2 run_status.json.")
    else:
        lineage["run_status"] = {"status": "pending"}
    if require_artifacts:
        paths = [lineage[role].get("path") for role in ("warmup", "best", "last")]
        if len(set(paths)) != 3:
            raise ValueError("Warm-up, Best, and Last checkpoints must be distinct artifacts.")
        if lineage.get("run_status", {}).get("status") != "validated":
            raise ValueError("Stage-2 run_status.json lineage evidence is not validated.")
        payload["lineage_status"] = "validated"
        payload["run_status"] = dict(run_status_evidence or {})
        artifacts["run_status"] = dict(run_status_evidence or {})
        payload["run_status_path"] = str(run_status_evidence["path"])
        payload["run_status_sha256"] = str(run_status_evidence["sha256"])
    else:
        payload["lineage_status"] = "pending"
    payload["lineage"] = lineage
    payload["updated_at"] = utc_now()
    _append_operation(payload, "audit_stage2_lineage", status=payload["lineage_status"])
    atomic_json(target, payload)
    return payload


def _invoke(command: Sequence[str], runner: Callable[..., Any] | None = None) -> Any:
    invoke = runner or subprocess.run
    print(f"[Stage2Command] {shlex.join(list(command))}", flush=True)
    return invoke(list(command), cwd=str(ROOT), check=True, text=True)


def run_stage2(
    source_m2: str | os.PathLike[str],
    *,
    run_tag: str,
    manifest_path: str | os.PathLike[str],
    dry_run: bool = False,
    runner: Callable[..., Any] | None = None,
    **plan_kwargs: Any,
) -> dict[str, Any]:
    """Plan, optionally execute, and require complete checkpoint lineage."""

    payload = plan_stage2(
        source_m2,
        run_tag=run_tag,
        manifest_path=manifest_path,
        dry_run=dry_run,
        **plan_kwargs,
    )
    target = Path(manifest_path).expanduser().resolve()
    if dry_run:
        _append_operation(payload, "execute_stage2", status="dry_run", subprocess_started=False)
        payload["status"] = "dry_run"
        payload["updated_at"] = utc_now()
        atomic_json(target, payload)
        return payload

    source = source_checkpoint_metadata(source_m2)
    _assert_manifest_source(payload, source)
    payload["status"] = "running"
    _append_operation(payload, "execute_stage2", status="running", subprocess_started=False)
    atomic_json(target, payload)
    try:
        _invoke(payload["command"], runner=runner)
        payload = audit_stage2_manifest(target, require_artifacts=True)
        payload["status"] = "completed"
        _append_operation(payload, "execute_stage2", status="completed", subprocess_started=True)
        payload["updated_at"] = utc_now()
        atomic_json(target, payload)
    except Exception as error:
        try:
            payload = read_json(target)
            payload["status"] = "failed"
            payload["failure"] = {"class": type(error).__name__, "message": str(error)}
            _append_operation(payload, "execute_stage2", status="failed", subprocess_started=True)
            atomic_json(target, payload)
        finally:
            raise
    return payload


def default_manifest_path(run_tag: str) -> Path:
    return DEFAULT_MANIFEST_ROOT / f"{validate_run_tag(run_tag)}.json"


# Small compatibility aliases keep the controller convenient for notebooks and
# downstream audit scripts while preserving one implementation of each
# operation.
atomic_write_json = atomic_json
generate_stage2_command = build_stage2_command
register_external_stage1_m2 = register_stage1_m2
create_stage2_manifest = plan_stage2
validate_lineage = audit_stage2_manifest
run_controller = run_stage2


def _common_cli(parser: argparse.ArgumentParser, *, source_required: bool = True) -> None:
    if source_required:
        parser.add_argument(
            "--source-m2",
            "--stage1-m2",
            "--stage1-m2-checkpoint",
            "--source-m2-checkpoint",
            "--checkpoint-dir",
            required=False,
            default=None,
            dest="source_m2",
        )
        parser.add_argument(
            "--stage1-handoff",
            default=str(DEFAULT_STAGE1_HANDOFF),
            dest="stage1_handoff",
            help=(
                "verified Stage-1 M2 hand-off used when --source-m2 is omitted"
            ),
        )
        parser.add_argument(
            "--source-seed",
            type=int,
            default=None,
            help=(
                "Stage-1 seed selected from --stage1-handoff; defaults to "
                "handoff.default_source_seed, then --seed for legacy hand-offs"
            ),
        )
    parser.add_argument("--manifest", "--manifest-path", default=None, dest="manifest_path")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--artifact-dir", "--stage2-artifact-dir", default=None, dest="artifact_dir")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bc-epochs", type=int, default=DEFAULT_BC_EPOCHS)
    parser.add_argument("--ppo-epochs", type=int, default=DEFAULT_PPO_EPOCHS)
    parser.add_argument("--ppo-epoch", type=int, default=DEFAULT_PPO_EPOCH)
    parser.add_argument("--bc-min-labels", type=int, default=DEFAULT_BC_MIN_LABELS)
    parser.add_argument("--bc-min-rollouts", type=int, default=DEFAULT_BC_MIN_ROLLOUTS)
    parser.add_argument("--bc-max-rollouts", type=int, default=DEFAULT_BC_MAX_ROLLOUTS)
    parser.add_argument("--bc-lr", type=float, default=0.0)
    parser.add_argument("--plane-order-mode", choices=("fixed", "learned"), default=None)
    parser.add_argument("--plane-pair-decoder", choices=("cascade", "joint_pair"), default=None)
    parser.add_argument("--global-feature-mode", choices=("none", "f1", "f1f2"), default=None)
    parser.add_argument("--source-command-json", default=None)
    parser.add_argument(
        "--source-command-key",
        default=None,
        help="entry key when --source-command-json is a multi-command suite manifest",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(argv if argv is not None else sys.argv[1:])
    # Direct ``--source-m2 ...`` invocation is treated as a plan for shell
    # compatibility.  Explicit subcommands remain clearer and are used by
    # the service launcher.
    if not raw or raw[0].startswith("-"):
        raw.insert(0, "plan")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    register = subparsers.add_parser("register", aliases=("register-m2",))
    _common_cli(register)
    plan = subparsers.add_parser("plan")
    _common_cli(plan)
    plan.add_argument("--dry-run", action="store_true", default=False)
    run = subparsers.add_parser("run")
    _common_cli(run)
    run.add_argument("--dry-run", action="store_true", default=False)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", "--manifest-path", required=True, dest="manifest_path")
    audit.add_argument("--allow-pending", action="store_true", default=False)
    return parser.parse_args(raw)


def _plan_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "artifact_dir": args.artifact_dir,
        "source_command": _load_source_command(
            args.source_command_json,
            args.source_command_key,
        ),
        "seed": args.seed,
        "bc_epochs": args.bc_epochs,
        "ppo_epochs": args.ppo_epochs,
        "ppo_epoch": args.ppo_epoch,
        "bc_min_labels": args.bc_min_labels,
        "bc_min_rollouts": args.bc_min_rollouts,
        "bc_max_rollouts": args.bc_max_rollouts,
        "bc_lr": args.bc_lr,
        "plane_order_mode": args.plane_order_mode,
        "plane_pair_decoder": args.plane_pair_decoder,
        "global_feature_mode": args.global_feature_mode,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action != "audit" and args.source_m2 is None:
        handoff = load_stage1_handoff(args.stage1_handoff)
        source_seed = (
            int(handoff.get("default_source_seed", args.seed))
            if args.source_seed is None
            else args.source_seed
        )
        source_seed_key = str(int(source_seed))
        if source_seed_key not in handoff["checkpoints"]:
            available = ", ".join(sorted(handoff["checkpoints"]))
            raise ValueError(
                f"Stage-1 hand-off has no seed {source_seed_key}; "
                f"available seeds: {available}."
            )
        source_entry = dict(handoff["checkpoints"][source_seed_key])
        semantics = dict(handoff["semantic_contract"])
        args.source_m2 = source_entry["path"]
        if args.source_command_json is None:
            args.source_command_json = source_entry["source_command_path"]
            args.source_command_key = source_entry.get("source_command_key")
        if args.plane_order_mode is None:
            args.plane_order_mode = semantics["plane_order_mode"]
        if args.plane_pair_decoder is None:
            args.plane_pair_decoder = semantics["plane_pair_decoder"]
        if args.global_feature_mode is None:
            args.global_feature_mode = semantics["global_feature_mode"]
    if args.action != "audit" and not args.manifest_path:
        args.manifest_path = str(default_manifest_path(args.run_tag))
    if args.action.startswith("register"):
        payload = register_stage1_m2(
            args.source_m2,
            manifest_path=args.manifest_path,
            run_tag=args.run_tag,
        )
    elif args.action == "plan":
        payload = plan_stage2(
            args.source_m2,
            run_tag=args.run_tag,
            manifest_path=args.manifest_path,
            dry_run=args.dry_run,
            **_plan_kwargs(args),
        )
    elif args.action == "run":
        payload = run_stage2(
            args.source_m2,
            run_tag=args.run_tag,
            manifest_path=args.manifest_path,
            dry_run=args.dry_run,
            **_plan_kwargs(args),
        )
    else:
        payload = audit_stage2_manifest(
            args.manifest_path,
            require_artifacts=not args.allow_pending,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
