#!/usr/bin/env python3
"""Fail-closed preflight and command generator for departure-aware Stage 1.

This module deliberately does not select a winner from blind-test data.  It
materializes the pre-registered B0--B2 or P0--P5 commands after verifying the
new teacher/potential contract; the existing shared-evaluator/systemd launcher
can then place three generated commands on each GPU with isolated CPU slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.utils.checkpoint_contract import (
    stage1_reward_contract,
    validate_stage1_checkpoint_contract,
)


BC_VARIANTS = {
    'B0_legacy_dagger_warm': {
        'warm_start': True,
        'global_feature_mode': 'f1f2',
        'phase_aware': False,
        'per_agent_dagger': False,
    },
    'B1_phase_dagger_warm': {
        'warm_start': True,
        'global_feature_mode': 'f1f2_departure',
        'phase_aware': True,
        'per_agent_dagger': True,
    },
    'B2_phase_dagger_cold': {
        'warm_start': False,
        'global_feature_mode': 'f1f2_departure',
        'phase_aware': True,
        'per_agent_dagger': True,
    },
}


PPO_VARIANTS = {
    'P0_action_cmax': {
        'global_feature_mode': 'f1f2',
        'reward': 'cmax_delta',
        'potential_schedule': '',
        'terminal_cmax_coef': 0.0,
    },
    'P1_action_potential': {
        'global_feature_mode': 'f1f2',
        'reward': 'iga_potential',
        'potential_schedule': '0.10',
        'terminal_cmax_coef': 0.0,
    },
    'P2_team_cmax': {
        'global_feature_mode': 'f1f2',
        'reward': 'team_cmax',
        'potential_schedule': '',
        'terminal_cmax_coef': 1.0,
    },
    'P3_team_time': {
        'global_feature_mode': 'f1f2',
        'reward': 'team_time',
        'potential_schedule': '',
        'terminal_cmax_coef': 1.0,
    },
    'P4_team_time_potential_ramp': {
        'global_feature_mode': 'f1f2',
        'reward': 'team_time_potential',
        'potential_schedule': '0.10,0.10,0.05,0.00',
        'terminal_cmax_coef': 1.0,
    },
    'P5_team_time_potential_fixed': {
        'global_feature_mode': 'f1f2',
        'reward': 'team_time_potential',
        'potential_schedule': '0.10',
        'terminal_cmax_coef': 1.0,
    },
}

PPO_WAVES = {
    1: tuple(list(PPO_VARIANTS)[:3]),
    2: tuple(list(PPO_VARIANTS)[3:]),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_bc_checkpoint_contract(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    validation = validate_stage1_checkpoint_contract(
        checkpoint,
        global_feature_mode='f1f2',
        plane_order_mode='fixed',
        plane_pair_decoder='joint_pair',
        strict_metadata=True,
    )
    if checkpoint.get('training_stage') != 'plane_pretrain':
        raise ValueError(
            'PPO source checkpoint must use training_stage=plane_pretrain.'
        )
    if checkpoint.get('stage') != 'plane_iga_bc_pretrain':
        raise ValueError(
            'PPO source checkpoint must be the frozen PlaneBC artifact.'
        )
    if not isinstance(checkpoint.get('model'), dict):
        raise ValueError('PPO source checkpoint is missing its model mapping.')
    return {
        'path': str(checkpoint_path.resolve()),
        'sha256': sha256(checkpoint_path),
        'global_feature_mode': validation['global_feature_mode'],
        'observation_schema_id': validation['observation_schema_id'],
        'environment_semantics_version': validation[
            'environment_semantics_version'
        ],
        'plane_order_mode': str(checkpoint['plane_order_mode']),
        'plane_pair_decoder': str(checkpoint['plane_pair_decoder']),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def set_option(command: list[str], flag: str, value) -> None:
    while command.count(flag) > 1:
        index = len(command) - 1 - command[::-1].index(flag)
        del command[index:index + 2]
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def remove_option(command: list[str], flag: str, *, takes_value=True) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + (2 if takes_value else 1)]


def set_switch(command: list[str], flag: str, enabled: bool) -> None:
    remove_option(command, flag, takes_value=False)
    if enabled:
        command.append(flag)


def validate_teacher_contract(
    dataset_dir: Path,
    teacher_dir: Path,
    expected_cases: int,
) -> dict:
    cases = sorted(path for path in dataset_dir.glob('case_*') if path.is_dir())
    if len(cases) != expected_cases:
        raise RuntimeError(
            f'Expected {expected_cases} dataset cases, found {len(cases)}.'
        )
    missing = []
    invalid = []
    makespans = []
    for case in cases:
        teacher_path = teacher_dir / f'{case.name}.json'
        if not teacher_path.is_file():
            missing.append(case.name)
            continue
        payload = read_json(teacher_path)
        makespan = float(payload.get('makespan', math.inf))
        metadata_path = case / 'metadata.json'
        expected_sha = None
        metadata_split = None
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            expected_sha = metadata.get('case_sha256')
            metadata_split = metadata.get('split')
        if (
            int(payload.get('schema_version', 0)) < 2
            or payload.get('teacher_scope') != 'stage1_plane_policy'
            or payload.get('resource_policy') != 'heuristic'
            or not payload.get('completion_verified', False)
            or payload.get('environment_semantics_version')
            != AircraftScheduleEnv.SEMANTICS_VERSION
            or payload.get('case') != case.name
            or not payload.get('case_sha256')
            or (
                expected_sha is not None
                and payload.get('case_sha256') != expected_sha
            )
            or not math.isfinite(makespan)
            or not 0.0 < makespan < 100000.0
            or (
                metadata_split is not None
                and str(metadata_split) != 'train'
            )
        ):
            invalid.append(case.name)
            continue
        makespans.append(makespan)
    if missing or invalid:
        raise RuntimeError(
            'Teacher contract failed: '
            f'missing={missing[:10]} ({len(missing)}), '
            f'invalid={invalid[:10]} ({len(invalid)}).'
        )
    return {
        'case_count': len(cases),
        'teacher_count': len(makespans),
        'mean_teacher_cmax': sum(makespans) / max(1, len(makespans)),
        'environment_semantics_version': AircraftScheduleEnv.SEMANTICS_VERSION,
    }


def validate_potential_contract(
    potential_path: Path,
    expected_cases: int,
) -> dict:
    payload = read_json(potential_path)
    replay = payload.get('replay', {})
    if payload.get('status') != 'completed':
        raise RuntimeError('Potential analysis is not completed.')
    if (
        int(payload.get('potential_schema_version', 0))
        != AircraftScheduleEnv.IGA_POTENTIAL_SCHEMA_VERSION
    ):
        raise RuntimeError('Potential schema version mismatch.')
    if (
        payload.get('environment_semantics_version')
        != AircraftScheduleEnv.SEMANTICS_VERSION
    ):
        raise RuntimeError('Potential environment semantics mismatch.')
    if tuple(payload.get('feature_names', ())) != tuple(
        AircraftScheduleEnv.IGA_POTENTIAL_FEATURES
    ):
        raise RuntimeError('Potential feature order mismatch.')
    if (
        int(replay.get('expected_cases', -1)) != expected_cases
        or int(replay.get('reverified_cases', -1)) != expected_cases
        or int(replay.get('stored_cmax_match_cases', -1)) != expected_cases
        or float(replay.get('max_abs_cmax_drift', float('inf'))) > 1e-6
    ):
        raise RuntimeError(
            'Potential replay did not reverify every teacher with zero drift.'
        )
    weights = payload.get('weights', {})
    if set(weights) != set(AircraftScheduleEnv.IGA_POTENTIAL_FEATURES):
        raise RuntimeError('Potential weights do not cover the exact V2 schema.')
    weight_values = [float(weights[name]) for name in weights]
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in weight_values)
        or not any(value > 0.0 for value in weight_values)
    ):
        raise RuntimeError('Potential weights must be finite and non-negative.')
    cross_validation = payload.get('calibration', {}).get(
        'case_level_cross_validation', {}
    )
    if (
        not cross_validation.get('case_disjoint', False)
        or not math.isfinite(float(
            cross_validation.get('metrics', {}).get('rmse', math.inf)
        ))
    ):
        raise RuntimeError(
            'Potential calibration lacks finite case-disjoint cross-validation.'
        )
    return {
        'potential_schema_version': int(payload['potential_schema_version']),
        'transition_count': int(replay.get('transition_count', 0)),
        'calibration': payload.get('calibration', {}),
    }


def common_command(
    source_command: list[str],
    *,
    experiment_name: str,
    seed: int,
    epochs: int,
    teacher_dir: Path,
    potential_path: Path,
) -> list[str]:
    command = list(source_command)
    if len(command) < 2:
        raise ValueError('Source command must contain Python and train_hkbz.py.')
    set_option(command, '--experiment_name', experiment_name)
    set_option(command, '--seed', seed)
    set_option(command, '--num_episodes', epochs)
    set_option(command, '--training_stage', 'plane_pretrain')
    set_option(command, '--resource_policy', 'heuristic')
    set_option(command, '--device_bc_pretrain_epochs', 0)
    set_option(command, '--gnn_freeze_epochs', 0)
    set_option(command, '--plane_freeze_epochs', 0)
    set_option(command, '--plane_order_mode', 'fixed')
    set_option(command, '--plane_pair_decoder', 'joint_pair')
    set_option(command, '--plane_bc_teacher_dir', teacher_dir)
    set_option(command, '--iga_potential_weights_path', potential_path)
    set_option(command, '--iga_potential_gamma', 1.0)
    set_option(command, '--gamma', 1.0)
    set_option(command, '--reward_coef', 0.01)
    set_option(command, '--hindsight_terminal_cmax_coef', 1.0)
    set_option(command, '--tail_policy_start_fraction', 1.0)
    set_option(command, '--tail_policy_weight', 1.0)
    set_option(command, '--selection_metric', 'composite')
    set_option(command, '--selection_iid_weight', 0.50)
    set_option(command, '--selection_ood_stress_weight', 0.45)
    set_option(command, '--selection_ood_scale_weight', 0.05)
    set_option(command, '--train_sampling_mode', 'distribution_balanced')
    set_option(
        command,
        '--train_sampling_weights',
        'iid=0.50,ood_stress=0.45,ood_scale=0.05',
    )
    set_option(command, '--train_sampling_size', 0)
    set_option(command, '--max_train_cases', 0)
    set_option(command, '--evaluation_tau', 0.3)
    set_option(command, '--anneal_original', 0.3)
    set_option(command, '--anneal_final', 0.3)
    set_option(command, '--tau_anneal_epochs', 0)
    set_option(command, '--eval_interval', 1)
    set_option(command, '--early_stop_patience', 0)
    set_option(command, '--canary_eval_interval_shards', 0)
    set_switch(command, '--canary_stop_on_regression', False)
    remove_option(command, '--selection_checkpoint_dir')
    remove_option(command, '--no_eval', takes_value=False)
    remove_option(command, '--shared_eval_socket')
    remove_option(command, '--shared_eval_cpu_set')
    remove_option(command, '--shared_eval_timeout_seconds')
    set_option(command, '--bc_reference_kl_coef', 0.40)
    set_option(
        command,
        '--bc_reference_kl_coef_schedule',
        '0.40,0.25,0.10,0.05,0.00',
    )
    set_switch(command, '--joint_team_ppo', True)
    set_switch(command, '--central_team_critic', False)
    set_switch(command, '--use_eval', True)
    return command


def bc_command(
    source_command: list[str],
    variant_id: str,
    *,
    run_tag: str,
    seed: int,
    epochs: int,
    teacher_dir: Path,
    potential_path: Path,
    warm_start_checkpoint: Path,
) -> list[str]:
    variant = BC_VARIANTS[variant_id]
    command = common_command(
        source_command,
        experiment_name=f'{run_tag}_bc_screen_{variant_id}_seed{seed}',
        seed=seed,
        epochs=epochs,
        teacher_dir=teacher_dir,
        potential_path=potential_path,
    )
    set_option(command, '--global_feature_mode', variant['global_feature_mode'])
    set_option(command, '--plane_bc_pretrain_epochs', 4)
    set_switch(command, '--plane_bc_only', True)
    set_option(command, '--plane_bc_dagger_schedule', '1.00,0.70,0.40,0.10')
    set_option(command, '--plane_bc_dagger_tail_start_fraction', 1.0)
    set_option(command, '--plane_bc_dagger_tail_teacher_rate', 0.0)
    remove_option(command, '--plane_bc_staging_dagger_schedule')
    set_option(command, '--plane_bc_service_weight', 1.0)
    set_option(command, '--plane_bc_staging_hold_weight', 1.0)
    set_option(command, '--plane_bc_staging_move_weight', 2.0)
    set_option(command, '--plane_bc_service_tail_start_fraction', 0.75)
    set_option(command, '--plane_bc_service_tail_weight', 3.0)
    set_option(
        command,
        '--plane_bc_tail_start_fraction',
        1.0 if variant['phase_aware'] else 0.75,
    )
    set_option(
        command,
        '--plane_bc_tail_weight',
        1.0 if variant['phase_aware'] else 3.0,
    )
    set_switch(command, '--plane_bc_phase_aware', variant['phase_aware'])
    set_switch(
        command, '--plane_bc_per_agent_dagger', variant['per_agent_dagger']
    )
    set_option(command, '--hindsight_reward_mode', 'team_time')
    set_option(command, '--iga_potential_beta', 0.0)
    remove_option(command, '--iga_potential_beta_schedule')
    if variant['warm_start']:
        set_option(command, '--checkpoint_dir', warm_start_checkpoint)
        set_switch(command, '--resume_stage1', True)
        set_switch(command, '--reset_optimizers_on_resume', True)
    else:
        remove_option(command, '--checkpoint_dir')
        set_switch(command, '--resume_stage1', False)
        set_switch(command, '--reset_optimizers_on_resume', False)
    return command


def ppo_command(
    source_command: list[str],
    variant_id: str,
    *,
    run_tag: str,
    phase: str,
    seed: int,
    epochs: int,
    teacher_dir: Path,
    potential_path: Path,
    bc_checkpoint: Path,
) -> list[str]:
    variant = PPO_VARIANTS[variant_id]
    command = common_command(
        source_command,
        experiment_name=f'{run_tag}_{phase}_{variant_id}_seed{seed}',
        seed=seed,
        epochs=epochs,
        teacher_dir=teacher_dir,
        potential_path=potential_path,
    )
    set_option(command, '--checkpoint_dir', bc_checkpoint)
    set_switch(command, '--resume_stage1', True)
    set_switch(command, '--reset_optimizers_on_resume', True)
    set_switch(command, '--reset_value_normalizer_on_resume', True)
    set_switch(command, '--strict_checkpoint_contract', True)
    set_switch(command, '--strict_stage1_reward_contract', True)
    set_option(command, '--plane_bc_pretrain_epochs', 0)
    set_switch(command, '--plane_bc_only', False)
    set_switch(command, '--plane_bc_phase_aware', False)
    set_switch(command, '--plane_bc_per_agent_dagger', False)
    set_option(command, '--global_feature_mode', variant['global_feature_mode'])
    set_option(command, '--hindsight_reward_mode', variant['reward'])
    set_option(command, '--hindsight_cmax_coef', 1.0)
    set_option(command, '--hindsight_shaping_coef', 0.0)
    set_option(
        command,
        '--hindsight_terminal_cmax_coef',
        variant['terminal_cmax_coef'],
    )
    schedule = variant['potential_schedule']
    set_option(command, '--iga_potential_beta', 0.10 if schedule else 0.0)
    if schedule:
        set_option(command, '--iga_potential_beta_schedule', schedule)
    else:
        remove_option(command, '--iga_potential_beta_schedule')
    stage1_reward_contract(
        reward_mode=variant['reward'],
        reward_coef=0.01,
        hindsight_cmax_coef=1.0,
        terminal_cmax_coef=variant['terminal_cmax_coef'],
        gamma=1.0,
        potential_gamma=1.0,
    )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--phase', choices=('bc_screen', 'ppo_screen', 'formal'), required=True
    )
    parser.add_argument(
        '--ppo-wave',
        choices=('1', '2', 'all'),
        default='all',
        help='materialize one three-method PPO wave or all six methods',
    )
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--source-command-json', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--teacher-dir', type=Path, required=True)
    parser.add_argument('--potential-path', type=Path, required=True)
    parser.add_argument('--warm-start-checkpoint', type=Path)
    parser.add_argument('--bc-checkpoint', type=Path)
    parser.add_argument('--formal-variant', choices=tuple(PPO_VARIANTS))
    parser.add_argument('--expected-cases', type=int, default=600)
    parser.add_argument('--screen-seed', type=int, default=1)
    parser.add_argument(
        '--screen-epochs',
        type=int,
        default=4,
        help=(
            'screen length; PPO screening requires at least four epochs so '
            'P3 beta ramp differs causally from fixed-beta P5'
        ),
    )
    parser.add_argument('--formal-epochs', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.screen_epochs <= 0 or args.formal_epochs <= 0:
        raise ValueError('Screen and formal epoch counts must be positive.')
    if args.phase == 'ppo_screen' and args.screen_epochs < 4:
        raise ValueError(
            '--screen-epochs must be at least 4 for PPO screening; shorter '
            'runs make the P3 ramp identical to fixed-beta P5.'
        )
    required = [
        args.source_command_json,
        args.dataset_dir,
        args.teacher_dir,
        args.potential_path,
    ]
    if args.phase == 'bc_screen':
        required.append(args.warm_start_checkpoint)
    else:
        required.append(args.bc_checkpoint)
    if any(path is None or not path.exists() for path in required):
        raise FileNotFoundError(
            [str(path) for path in required if path is None or not path.exists()]
        )
    source_payload = read_json(args.source_command_json)
    source_command = list(source_payload.get('command', ()))
    teacher_audit = validate_teacher_contract(
        args.dataset_dir, args.teacher_dir, args.expected_cases
    )
    potential_audit = validate_potential_contract(
        args.potential_path, args.expected_cases
    )
    bc_checkpoint_audit = None
    if args.phase != 'bc_screen':
        bc_checkpoint_audit = validate_bc_checkpoint_contract(
            args.bc_checkpoint
        )

    commands = {}
    if args.phase == 'bc_screen':
        for variant_id in BC_VARIANTS:
            commands[variant_id] = bc_command(
                source_command,
                variant_id,
                run_tag=args.run_tag,
                seed=args.screen_seed,
                epochs=args.screen_epochs,
                teacher_dir=args.teacher_dir,
                potential_path=args.potential_path,
                warm_start_checkpoint=args.warm_start_checkpoint,
            )
    elif args.phase == 'ppo_screen':
        variant_ids = (
            tuple(PPO_VARIANTS)
            if args.ppo_wave == 'all'
            else PPO_WAVES[int(args.ppo_wave)]
        )
        for variant_id in variant_ids:
            commands[variant_id] = ppo_command(
                source_command,
                variant_id,
                run_tag=args.run_tag,
                phase='ppo_screen',
                seed=args.screen_seed,
                epochs=args.screen_epochs,
                teacher_dir=args.teacher_dir,
                potential_path=args.potential_path,
                bc_checkpoint=args.bc_checkpoint,
            )
    else:
        if not args.formal_variant:
            raise ValueError('--formal-variant is required for formal commands.')
        for seed in (1, 2, 3):
            key = f'{args.formal_variant}_seed{seed}'
            commands[key] = ppo_command(
                source_command,
                args.formal_variant,
                run_tag=args.run_tag,
                phase='formal',
                seed=seed,
                epochs=args.formal_epochs,
                teacher_dir=args.teacher_dir,
                potential_path=args.potential_path,
                bc_checkpoint=args.bc_checkpoint,
            )
    payload = {
        'schema_version': 1,
        'phase': args.phase,
        'run_tag': args.run_tag,
        'teacher_audit': teacher_audit,
        'potential_audit': potential_audit,
        'bc_checkpoint_audit': bc_checkpoint_audit,
        'ppo_wave': args.ppo_wave if args.phase == 'ppo_screen' else None,
        'reward_contracts': {
            key: stage1_reward_contract(
                reward_mode=value['reward'],
                reward_coef=0.01,
                hindsight_cmax_coef=1.0,
                terminal_cmax_coef=value['terminal_cmax_coef'],
                gamma=1.0,
                potential_gamma=1.0,
            )
            for key, value in PPO_VARIANTS.items()
            if key in commands
        },
        'hardware_contract': {
            'trainers_per_gpu': 3,
            'shared_evaluator_per_gpu': 1,
            'cpu_isolation_required': True,
        },
        'evaluator_command': next(iter(commands.values())),
        'commands': {
            key: {
                'argv': command,
                'shell': shlex.join(command),
            }
            for key, command in commands.items()
        },
    }
    atomic_json(args.output, payload)
    print(f'[DepartureResearch] wrote {len(commands)} commands to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
