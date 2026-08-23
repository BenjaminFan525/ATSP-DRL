"""Runtime contracts shared by the HKBZ staged trainer.

The stage-2 hand-off is intentionally kept in a small dependency-free module.
It is used by configuration/tests as well as by :class:`HKBZ_Runner`, so a
checkpoint cannot accidentally be accepted merely because a permissive model
loader found a few compatible tensors.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence


CANONICAL_RESOURCE_JOINT = "resource_joint"
RESOURCE_JOINT_STAGE = CANONICAL_RESOURCE_JOINT
STAGE_AUTO = "auto"
STAGE_PLANE_PRETRAIN = "plane_pretrain"
DEPRECATED_STAGE_ALIASES = {
    "device_bc": CANONICAL_RESOURCE_JOINT,
    "frozen_joint": CANONICAL_RESOURCE_JOINT,
}
RETIRED_STAGES = {"full_joint"}
SUPPORTED_TRAINING_STAGES = (
    STAGE_AUTO,
    STAGE_PLANE_PRETRAIN,
    CANONICAL_RESOURCE_JOINT,
    *DEPRECATED_STAGE_ALIASES,
)

# These prefixes are deliberately exact.  In particular, resource selectors
# are not in this set: a Stage-1 checkpoint may contain an untrained or absent
# resource backend, while the plane/shared hand-off must remain fail-closed.
PROTECTED_RESOURCE_JOINT_PREFIXES = (
    "encoder.",
    "plane_sel_enc.",
    "actor.",
    "plane_order_actor.",
)


def normalize_training_stage(stage: object, *, warn_deprecated: bool = True) -> str:
    """Return the canonical stage name and reject retired/unknown names.

    ``device_bc`` and ``frozen_joint`` remain parseable for old launchers, but
    they now denote the complete resource-joint transition.  Keeping the
    warning here (rather than in argument parsing) also covers callers that
    construct ``Namespace`` objects directly in tests or service wrappers.
    """

    value = str(stage if stage is not None else STAGE_AUTO).strip().lower()
    if value in RETIRED_STAGES:
        raise ValueError(
            "training_stage='full_joint' is retired; Stage 2 ends at "
            "frozen resource_joint PPO and no Stage-4/full-joint transition "
            "is supported."
        )
    if value in DEPRECATED_STAGE_ALIASES:
        if warn_deprecated:
            warnings.warn(
                f"training_stage='{value}' is deprecated; use "
                f"'{CANONICAL_RESOURCE_JOINT}'.",
                DeprecationWarning,
                stacklevel=2,
            )
        return CANONICAL_RESOURCE_JOINT
    if value not in {STAGE_AUTO, STAGE_PLANE_PRETRAIN, CANONICAL_RESOURCE_JOINT}:
        allowed = ", ".join(
            (STAGE_AUTO, STAGE_PLANE_PRETRAIN, CANONICAL_RESOURCE_JOINT)
        )
        raise ValueError(
            f"Unsupported training_stage={value!r}; expected one of {allowed}."
        )
    return value


# Common spelling used by small callers and older tests.
canonicalize_training_stage = normalize_training_stage
canonical_training_stage = normalize_training_stage


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a checkpoint without loading its tensors."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_checkpoint_metadata(path: str | Path) -> dict[str, str]:
    """Describe the exact Stage-1 source artifact used for a hand-off."""

    resolved = str(Path(path).expanduser().resolve())
    return {"path": resolved, "sha256": sha256_file(resolved)}


def _is_tensor_like(value: object) -> bool:
    # Avoid importing torch in this utility: numpy arrays and tensor-like
    # objects both expose the attributes needed by the digest helper.
    return hasattr(value, "shape") and hasattr(value, "detach")


def _tensor_digest(value: object) -> str:
    """Digest a tensor's exact bytes, dtype and shape.

    The tensor is moved to CPU only for the digest; no model parameter is
    mutated.  ``detach`` is intentionally required so a malformed checkpoint
    cannot be mistaken for a valid parameter mapping.
    """

    if not _is_tensor_like(value):
        raise TypeError("Protected checkpoint entries must be tensor-like.")
    tensor = value.detach().cpu().contiguous()
    payload = tensor.numpy().tobytes()
    header = f"{tensor.dtype}|{tuple(int(dim) for dim in tensor.shape)}|".encode()
    return hashlib.sha256(header + payload).hexdigest()


def protected_parameter_names(
    state_dict: Mapping[str, object],
    prefixes: Sequence[str] = PROTECTED_RESOURCE_JOINT_PREFIXES,
) -> tuple[str, ...]:
    """List protected state keys in deterministic order."""

    return tuple(
        sorted(
            str(name)
            for name in state_dict
            if any(str(name).startswith(prefix) for prefix in prefixes)
        )
    )


def protected_parameter_summary(
    state_dict: Mapping[str, object],
    prefixes: Sequence[str] = PROTECTED_RESOURCE_JOINT_PREFIXES,
) -> dict[str, object]:
    """Create a compact bitwise identity for protected parameters.

    The combined digest is suitable for run-status/checkpoint evidence while
    ``parameters`` gives useful per-key diagnostics when a hand-off fails.
    """

    names = protected_parameter_names(state_dict, prefixes)
    parameters: MutableMapping[str, dict[str, object]] = {}
    groups: MutableMapping[str, list[str]] = {
        str(prefix).rstrip("."): [] for prefix in prefixes
    }
    total_numel = 0
    digest_parts = []
    for name in names:
        value = state_dict[name]
        if not _is_tensor_like(value):
            raise TypeError(f"Protected state entry {name!r} is not a tensor.")
        shape = tuple(int(dim) for dim in value.shape)
        digest = _tensor_digest(value)
        total_numel += int(value.numel())
        parameters[name] = {
            "shape": list(shape),
            "dtype": str(value.dtype),
            "sha256": digest,
        }
        for prefix in prefixes:
            if name.startswith(prefix):
                groups[str(prefix).rstrip(".")].append(digest)
                break
        digest_parts.append(f"{name}\0{digest}\0")
    combined = hashlib.sha256("".join(digest_parts).encode("utf-8")).hexdigest()
    group_summaries = {
        name: {
            "count": len(digests),
            "sha256": hashlib.sha256(
                "".join(digests).encode("utf-8")
            ).hexdigest(),
        }
        for name, digests in groups.items()
    }
    return {
        "scope": list(prefixes),
        "count": len(names),
        "numel": total_numel,
        "sha256": combined,
        "groups": group_summaries,
        "parameters": dict(parameters),
    }


def protected_parameters_equal(
    left: Mapping[str, object],
    right: Mapping[str, object],
    prefixes: Sequence[str] = PROTECTED_RESOURCE_JOINT_PREFIXES,
) -> bool:
    """Compare protected tensors bit-for-bit, including key/shape identity."""

    return protected_parameter_summary(left, prefixes) == protected_parameter_summary(
        right, prefixes
    )


def _checkpoint_global_feature_mode(checkpoint: Mapping[str, object]) -> str | None:
    experiment_config = checkpoint.get("experiment_config", {})
    if isinstance(experiment_config, Mapping):
        value = experiment_config.get("global_feature_mode")
        if value is not None:
            return str(value)
    value = checkpoint.get("global_feature_mode")
    return None if value is None else str(value)


def validate_stage1_m2_checkpoint(
    checkpoint: Mapping[str, object],
    target_state: Mapping[str, object],
    *,
    plane_order_mode: object,
    plane_pair_decoder: object,
    global_feature_mode: object,
) -> dict[str, object]:
    """Validate the strict Stage-1 M2 -> resource-joint hand-off.

    Only the protected plane/shared state is required.  Resource actor and
    critic tensors are intentionally optional, but *any* missing/mismatched
    protected tensor or semantic architecture field rejects the hand-off.
    """

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage-1 M2 checkpoint must contain a mapping payload.")
    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Stage-1 M2 checkpoint is missing its model state mapping.")

    training_stage = str(
        checkpoint.get("training_stage", checkpoint.get("stage", ""))
    ).strip().lower()
    if training_stage not in {
        STAGE_PLANE_PRETRAIN,
        "stage1",
        "stage1_m2",
        "stage1_plane",
        "plane_iga_bc_pretrain",
        "m2",
    }:
        raise ValueError(
            "resource_joint requires a Stage-1 M2 checkpoint with "
            "training_stage='plane_pretrain'; got "
            f"{checkpoint.get('training_stage', checkpoint.get('stage'))!r}."
        )

    expected_semantics = {
        "plane_order_mode": str(plane_order_mode),
        "plane_pair_decoder": str(plane_pair_decoder),
        "global_feature_mode": str(global_feature_mode),
    }
    observed_semantics = {
        "plane_order_mode": checkpoint.get("plane_order_mode"),
        "plane_pair_decoder": checkpoint.get("plane_pair_decoder"),
        "global_feature_mode": _checkpoint_global_feature_mode(checkpoint),
    }
    incompatible = {
        key: {"checkpoint": observed_semantics[key], "configured": value}
        for key, value in expected_semantics.items()
        if observed_semantics[key] is None
        or str(observed_semantics[key]) != value
    }
    if incompatible:
        raise ValueError(
            "Stage-1 M2 semantic checkpoint mismatch for "
            f"{sorted(incompatible)}: {incompatible}."
        )

    missing = []
    mismatched = []
    for name in protected_parameter_names(target_state):
        if name not in model:
            missing.append(name)
            continue
        source_value = model[name]
        target_value = target_state[name]
        if not _is_tensor_like(source_value) or not _is_tensor_like(target_value):
            mismatched.append((name, "non_tensor"))
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            mismatched.append(
                (name, f"checkpoint={tuple(source_value.shape)} target={tuple(target_value.shape)}")
            )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing protected tensors={missing[:8]}")
        if mismatched:
            details.append(f"shape/type mismatches={mismatched[:8]}")
        raise ValueError("Strict Stage-1 M2 hand-off rejected: " + "; ".join(details))

    # Summarize the exact hand-off intersection.  A legacy fixed-order
    # checkpoint can carry stale optional ``plane_order_actor`` tensors even
    # when the configured target has no such module; those extras are not part
    # of the target's protected scope and must not make a valid hand-off look
    # bitwise different after loading.
    target_protected_names = protected_parameter_names(target_state)
    source_protected = {
        name: model[name] for name in target_protected_names
    }
    source_summary = protected_parameter_summary(source_protected)
    target_summary = protected_parameter_summary(target_state)
    return {
        "training_stage": training_stage,
        "source_summary": source_summary,
        "target_summary": target_summary,
        "semantic_config": expected_semantics,
    }


def validate_stage2_recovery_checkpoint(
    checkpoint: Mapping[str, object],
    target_state: Mapping[str, object],
    *,
    plane_order_mode: object,
    plane_pair_decoder: object,
    global_feature_mode: object,
    protected_prefixes: Sequence[str] = PROTECTED_RESOURCE_JOINT_PREFIXES,
) -> dict[str, object]:
    """Validate a complete, cursor-bearing Stage-2 recovery checkpoint.

    Stage-2 recovery must be exact: unlike the Stage-1 hand-off, every model
    tensor is required and shape-compatible.  Only a checkpoint written after
    a completed shard is accepted, so an emergency snapshot taken halfway
    through an update can never be mistaken for a resumable cursor.
    """

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage-2 recovery checkpoint must contain a mapping payload.")
    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Stage-2 recovery checkpoint is missing its model state mapping.")

    training_stage = str(checkpoint.get("training_stage", "")).strip().lower()
    stage = str(checkpoint.get("stage", "")).strip().lower()
    phase = str(checkpoint.get("phase", "")).strip().lower()
    if training_stage != CANONICAL_RESOURCE_JOINT:
        raise ValueError(
            "Stage-2 recovery requires training_stage='resource_joint'; got "
            f"{checkpoint.get('training_stage')!r}."
        )
    if stage != "post_shard_recovery":
        raise ValueError(
            "Stage-2 recovery requires a post_shard_recovery checkpoint; got "
            f"stage={checkpoint.get('stage')!r}."
        )
    if phase != "resource_joint_ppo":
        raise ValueError(
            "Stage-2 recovery checkpoint is not in resource_joint_ppo; got "
            f"phase={checkpoint.get('phase')!r}."
        )

    total_shards = int(checkpoint.get("total_shards", 0))
    completed_shards = int(checkpoint.get("completed_shard", 0))
    completed_epochs = int(checkpoint.get("episodes", 0))
    total_num_steps = int(checkpoint.get("total_num_steps", -1))
    if total_shards <= 0 or not 0 < completed_shards <= total_shards:
        raise ValueError(
            "Stage-2 recovery checkpoint has an invalid shard cursor: "
            f"completed={completed_shards}, total={total_shards}."
        )
    if completed_epochs <= 0 or total_num_steps < 0:
        raise ValueError(
            "Stage-2 recovery checkpoint has invalid progress counters: "
            f"episodes={completed_epochs}, total_num_steps={total_num_steps}."
        )

    expected_semantics = {
        "plane_order_mode": str(plane_order_mode),
        "plane_pair_decoder": str(plane_pair_decoder),
        "global_feature_mode": str(global_feature_mode),
    }
    observed_semantics = {
        "plane_order_mode": checkpoint.get("plane_order_mode"),
        "plane_pair_decoder": checkpoint.get("plane_pair_decoder"),
        "global_feature_mode": _checkpoint_global_feature_mode(checkpoint),
    }
    incompatible = {
        key: {"checkpoint": observed_semantics[key], "configured": value}
        for key, value in expected_semantics.items()
        if observed_semantics[key] is None
        or str(observed_semantics[key]) != value
    }
    if incompatible:
        raise ValueError(
            "Stage-2 recovery semantic mismatch for "
            f"{sorted(incompatible)}: {incompatible}."
        )

    missing = []
    mismatched = []
    for name, target_value in target_state.items():
        if name not in model:
            missing.append(str(name))
            continue
        source_value = model[name]
        if not _is_tensor_like(source_value) or not _is_tensor_like(target_value):
            mismatched.append((str(name), "non_tensor"))
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            mismatched.append(
                (
                    str(name),
                    f"checkpoint={tuple(source_value.shape)} "
                    f"target={tuple(target_value.shape)}",
                )
            )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing model tensors={missing[:8]}")
        if mismatched:
            details.append(f"shape/type mismatches={mismatched[:8]}")
        raise ValueError("Strict Stage-2 recovery rejected: " + "; ".join(details))

    source_path = str(checkpoint.get("source_m2_path", "") or "")
    source_sha256 = str(checkpoint.get("source_m2_sha256", "") or "")
    if not source_path or len(source_sha256) != 64:
        raise ValueError(
            "Stage-2 recovery checkpoint is missing its immutable Stage-1 M2 binding."
        )

    model_summary = protected_parameter_summary(model, prefixes=("",))
    protected_summary = protected_parameter_summary(
        model, prefixes=protected_prefixes
    )
    recorded_protected = checkpoint.get("protected_parameter_summary")
    if recorded_protected != protected_summary:
        raise ValueError(
            "Stage-2 recovery protected-parameter evidence does not match its model."
        )

    return {
        "training_stage": training_stage,
        "stage": stage,
        "phase": phase,
        "completed_epochs": completed_epochs,
        "completed_shards": completed_shards,
        "total_shards": total_shards,
        "total_num_steps": total_num_steps,
        "semantic_config": expected_semantics,
        "source_m2_path": source_path,
        "source_m2_sha256": source_sha256,
        "model_summary": model_summary,
        "protected_summary": protected_summary,
    }


__all__ = [
    "CANONICAL_RESOURCE_JOINT",
    "RESOURCE_JOINT_STAGE",
    "STAGE_AUTO",
    "STAGE_PLANE_PRETRAIN",
    "DEPRECATED_STAGE_ALIASES",
    "RETIRED_STAGES",
    "SUPPORTED_TRAINING_STAGES",
    "PROTECTED_RESOURCE_JOINT_PREFIXES",
    "normalize_training_stage",
    "canonicalize_training_stage",
    "canonical_training_stage",
    "sha256_file",
    "source_checkpoint_metadata",
    "protected_parameter_names",
    "protected_parameter_summary",
    "protected_parameters_equal",
    "validate_stage1_m2_checkpoint",
    "validate_stage2_recovery_checkpoint",
]
