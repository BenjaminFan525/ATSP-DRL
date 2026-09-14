"""Administrative Stage2 freeze, independent of scientific acceptance gates."""
from onpolicy.utils.training_stage import CANONICAL_RESOURCE_JOINT, normalize_training_stage


def reject_stage2_development():
    raise RuntimeError(
        'Stage2 development is frozen (2026-09-12). No new Stage2 training or '
        'automatic experiment suites are permitted by this snapshot. '
        'Use onpolicy/scripts/train/verify_stage2_frozen.py for verification '
        'and STAGE2_FROZEN.md for the explicit Stage3 handoff.'
    )


def guard_training_stage(stage):
    if normalize_training_stage(stage) == CANONICAL_RESOURCE_JOINT:
        reject_stage2_development()
