import time
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from onpolicy.runner.shared.base_runner import Runner
from torch_geometric.loader.dataloader import Batch
import wandb
import imageio
import copy
import gc
import shutil
from tqdm import tqdm
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer
import cProfile
import pstats
import math
import hashlib
import uuid
from types import SimpleNamespace
from pathlib import Path
from typing import Mapping
from scipy.optimize import linear_sum_assignment

from onpolicy.utils.shared_eval import (
    PROTOCOL_VERSION,
    SharedEvalClient,
    parse_cpu_set,
)
from onpolicy.utils.gpu_phase_lock import exclusive_gpu_phase
from onpolicy.utils.stage2_bc_contract import (
    capture_rng_state, restore_rng_state, choose_best_epoch, ready_head_config,
    supervised_gate, validate_frozen_ready_checkpoint,
    supervised_optimizer_step,
    copy_bc_epoch_evidence,
    portable_rng_state,
)
from onpolicy.utils.stage2_policy_transfer import validate_policy_warmstart, warmstart_replay
from onpolicy.utils.stage2_matching import (
    full_matching_supervision, finalize_matching_metrics,
    MATCHING_CONTRACT, SUPERVISION_CONTRACT, TEACHER_PROJECTION_CONTRACT,
)
from onpolicy.utils.training_stage import (
    CANONICAL_JOINT_FINETUNE,
    CANONICAL_RESOURCE_JOINT,
    PROTECTED_RESOURCE_JOINT_PREFIXES,
    STAGE2_SUPERVISION_CONTRACT,
    normalize_training_stage,
    protected_parameter_summary,
    source_checkpoint_metadata,
    validate_stage1_m2_checkpoint,
    validate_stage2_joint_finetune_checkpoint,
    validate_stage2_recovery_checkpoint,
)
from onpolicy.utils.checkpoint_contract import (
    stage1_observation_metadata,
    stage1_reward_contract,
    validate_stage1_checkpoint_contract,
)

def _t2n(x):
    return x.detach().cpu().numpy()


def _tensor_rounding_tolerance(dtype, *values):
    """Return a small tolerance for comparisons against tensor scalars.

    Stage2 labels are decoded as Python ``float`` values, while dependency
    bounds are produced by the network in ``float32``.  Comparing the two with
    a fixed microsecond tolerance can reject the same value after its normal
    float32 rounding.  Four units of relative rounding cover the conversion
    and the few arithmetic operations used to form the DAG bound while
    remaining well below the scheduling time resolution.
    """

    absolute_floor = 1e-6
    if not getattr(dtype, 'is_floating_point', False):
        return absolute_floor
    magnitude = max(
        1.0,
        *(abs(float(value)) for value in values if np.isfinite(value)),
    )
    return max(
        absolute_floor,
        4.0 * float(torch.finfo(dtype).eps) * magnitude,
    )


class HKBZ_Runner(Runner):
    """Runner class to perform training, evaluation. and data collection for the IAs. See parent class for details."""
    REQUEST_READY_KIND_OTHER = 0
    REQUEST_READY_KIND_H1 = 1
    REQUEST_READY_KIND_H2 = 2
    REQUEST_READY_KIND_BLOCKING = 3
    REQUEST_READY_KIND_DEPARTURE = 4
    REQUEST_READY_KIND_NAMES = {
        REQUEST_READY_KIND_OTHER: 'other',
        REQUEST_READY_KIND_H1: 'h1',
        REQUEST_READY_KIND_H2: 'h2',
        REQUEST_READY_KIND_BLOCKING: 'blocking',
        REQUEST_READY_KIND_DEPARTURE: 'departure',
    }
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.ac_config = config['ac_config']
        self.num_envs = config['num_envs']
        self.evaluation_only = bool(config.get('evaluation_only', False))
        self.eval_case_counts = config.get('eval_case_counts')
        self.eval_env_factory = config.get('eval_env_factory')
        self.shared_eval_socket = str(
            config.get('shared_eval_socket', '') or ''
        )
        self.shared_eval_cpu_set = str(
            config.get('shared_eval_cpu_set', '') or ''
        )
        self.shared_eval_client = None
        if self.shared_eval_socket:
            parse_cpu_set(self.shared_eval_cpu_set)
            self.shared_eval_client = SharedEvalClient(
                self.shared_eval_socket,
                timeout_seconds=float(config.get(
                    'shared_eval_timeout_seconds', 7200.0
                )),
            )
        self.release_eval_envs_after_eval = bool(
            config.get('release_eval_envs_after_eval', False)
        )
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.num_episodes = self.all_args.num_episodes
        self.nominal_episode_length = self.all_args.episode_length
        self.rollout_until_done = bool(getattr(self.all_args, 'rollout_until_done', True))
        self.rollout_max_steps = int(getattr(self.all_args, 'rollout_max_steps', self.nominal_episode_length))
        self.allow_incomplete_rollout = bool(getattr(self.all_args, 'allow_incomplete_rollout', False))
        self.episode_length = self.rollout_max_steps if self.rollout_until_done else self.nominal_episode_length
        self.all_args.episode_length = self.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.safe_async_graph_clone_workers = int(
            getattr(self.all_args, 'safe_async_graph_clone_workers', 0)
        )
        if self.safe_async_graph_clone_workers < 0:
            raise ValueError(
                '--safe_async_graph_clone_workers must be non-negative.'
            )
        self.safe_graph_batch_pipeline = bool(
            getattr(self.all_args, 'safe_graph_batch_pipeline', False)
        )
        self.safe_dagger_teacher_overlap = bool(
            getattr(self.all_args, 'safe_dagger_teacher_overlap', False)
        )
        if (
            self.safe_async_graph_clone_workers
            or self.safe_graph_batch_pipeline
            or self.safe_dagger_teacher_overlap
        ):
            print(
                '[SafePipeline] '
                f'graph_clone_workers={self.safe_async_graph_clone_workers} '
                f'graph_batch_pipeline={self.safe_graph_batch_pipeline} '
                f'dagger_teacher_overlap={self.safe_dagger_teacher_overlap}',
                flush=True,
            )
        self.max_graphs_per_forward = max(0, int(getattr(self.all_args, 'max_graphs_per_forward', 0)))
        self.actor_warmup_shards = max(
            0,
            int(getattr(self.all_args, 'actor_warmup_shards', 0)),
        )
        self.clear_cuda_cache_after_update = bool(
            getattr(self.all_args, 'clear_cuda_cache_after_update', False)
        )
        self.shared_gpu_phase_lock = str(
            getattr(self.all_args, 'shared_gpu_phase_lock', '') or ''
        )
        if self.shared_gpu_phase_lock and not os.path.isabs(
            self.shared_gpu_phase_lock
        ):
            raise ValueError('--shared_gpu_phase_lock must be absolute.')
        self.graphs_per_forward = self._validate_graph_batch_memory_config(
            self.all_args.mini_batch_size,
            self.all_args.data_chunk_length,
            self.max_graphs_per_forward,
            self.n_rollout_threads,
        )
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.use_anneal = self.all_args.use_anneal
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.obj = self.all_args.obj
        self.reward_coef = self.all_args.reward_coef
        self.fuse_s = self.all_args.fuse_s
        self.fuse_epoch = self.all_args.fuse_epoch
        self.start_epoch = self.all_args.start_epoch
        self.auto_fuse = self.all_args.auto_fuse
        self.training_stage = normalize_training_stage(
            getattr(self.all_args, 'training_stage', 'auto')
        )
        # The launcher writes this field directly into run_status.json.  Keep
        # it canonical even when a deprecated alias was supplied by an old
        # service wrapper.
        self.all_args.training_stage = self.training_stage
        self.resource_joint_phase = {
            CANONICAL_RESOURCE_JOINT: 'stage1_m2_restore_pending',
            CANONICAL_JOINT_FINETUNE: 'stage2_handoff_pending',
        }.get(self.training_stage, 'not_applicable')
        self.stage1_m2_source = None
        self.stage1_m2_checkpoint_summary = None
        self.stage2_source = None
        self.stage2_checkpoint_summary = None
        self.protected_parameter_summary_before_bc = None
        self.protected_parameter_summary_after_bc = None
        self.protected_parameter_summary_after_ppo = None
        self.resource_actor_summary_before_ppo = None
        self.resource_actor_summary_after_ppo = None
        self.resource_actor_summary_before_bc = None
        self.resource_actor_summary_after_bc = None
        self.request_ready_predictor_summary_before = None
        self.request_ready_predictor_summary_after = None
        self.plane_actor_summary_before_bc = None
        self.plane_actor_summary_after_bc = None
        self.plane_actor_summary_before_ppo = None
        self.plane_actor_summary_after_ppo = None
        self.shared_actor_summary_before_ppo = None
        self.shared_actor_summary_after_ppo = None
        self.resource_bc_optimizer_reset = False
        self.resource_supervised_optimizer_state = None
        self.frozen_ready_source = None
        self.frozen_ready_head_summary = None
        self.stage2_frozen_state_before = None
        self.stage2_pre_supervised_info = None
        self.stage2_bc_epoch_records = []
        self.stage2_bc_completed_epochs = 0
        self.stage2_bc_at_boundary = False
        self.stage2_bc_pending_rng = None
        self.resource_bc_total_labels = 0
        self.resource_dense_ranking_total_labels = 0
        self.resource_assignment_total_labels = 0
        self.request_ready_total_labels = 0
        self.resource_bc_phase_events = []
        self.hindsight_reward_mode = str(
            getattr(self.all_args, 'hindsight_reward_mode', '')
        )
        self.team_return_mode = self.hindsight_reward_mode in {
            'team_cmax', 'team_time', 'team_time_potential',
            'team_time_resource_potential',
            'team_time_resource_fitted_potential',
        }
        self.team_time_return_mode = self.hindsight_reward_mode in {
            'team_time', 'team_time_potential',
            'team_time_resource_potential',
            'team_time_resource_fitted_potential',
        }
        self.team_time_potential_mode = self.hindsight_reward_mode in {
            'team_time_potential', 'team_time_resource_potential',
            'team_time_resource_fitted_potential',
        }
        self.role_event_credit_mode = str(getattr(
            self.all_args, 'role_event_credit_mode', 'elapsed'
        ))
        self.role_event_credit_uniform_mix = float(getattr(
            self.all_args, 'role_event_credit_uniform_mix', 0.15
        ))
        self.paired_case_baseline_path = str(getattr(
            self.all_args, 'paired_case_baseline_dir', ''
        ) or '')
        self.paired_case_baseline_coef = float(getattr(
            self.all_args, 'paired_case_baseline_coef', 0.0
        ))
        self.paired_case_baseline_scope = str(getattr(
            self.all_args, 'paired_case_baseline_scope', 'returns'
        ))
        self.cvar_case_metric = str(getattr(
            self.all_args, 'cvar_case_metric', 'cmax'
        ))
        self.paired_case_baselines = self._load_paired_case_baselines(
            self.paired_case_baseline_path
        )
        if self.role_event_credit_mode not in {
            'elapsed', 'critical_path', 'critical_path_v2'
        }:
            raise ValueError(
                '--role_event_credit_mode must be elapsed, critical_path, '
                'or critical_path_v2.'
            )
        if not 0.0 <= self.role_event_credit_uniform_mix <= 1.0:
            raise ValueError(
                '--role_event_credit_uniform_mix must be in [0, 1].'
            )
        if (
            not np.isfinite(self.paired_case_baseline_coef)
            or self.paired_case_baseline_coef < 0.0
        ):
            raise ValueError(
                '--paired_case_baseline_coef must be finite and non-negative.'
            )
        if self.paired_case_baseline_scope not in {'returns', 'actor'}:
            raise ValueError(
                '--paired_case_baseline_scope must be returns or actor.'
            )
        if self.cvar_case_metric not in {'cmax', 'paired_delta'}:
            raise ValueError('--cvar_case_metric must be cmax or paired_delta.')
        if (
            self.paired_case_baseline_coef > 0.0
            or self.cvar_case_metric == 'paired_delta'
        ) and not self.paired_case_baselines:
            raise ValueError(
                'Paired terminal training requires a non-empty '
                '--paired_case_baseline_dir.'
            )
        role_event_returns = bool(getattr(
            self.all_args, 'role_event_returns', False
        ))
        if role_event_returns and not self.team_time_return_mode:
            raise ValueError(
                '--role_event_returns requires a team_time reward mode.'
            )
        if (
            self.role_event_credit_mode != 'elapsed'
            and not role_event_returns
        ):
            raise ValueError(
                'non-elapsed role event credit requires '
                '--role_event_returns.'
            )
        self.recovery_checkpoint_interval_shards = max(
            0,
            int(getattr(self.all_args, 'recovery_checkpoint_interval_shards', 1)),
        )
        self.progress_callback = config.get('progress_callback')
        self.current_epoch = -1
        self.current_shard = -1
        self.last_team_cmax_mean = 0.0
        self.last_team_return_mean_raw = 0.0
        self.last_resource_wait_mean = 0.0
        self.last_resource_critical_wait_mean = 0.0
        self.last_resource_slack_weighted_wait_mean = 0.0
        self.last_resource_avoidable_critical_lateness_mean = 0.0
        self.last_resource_rendezvous_spread_mean = 0.0
        self.last_paired_case_delta_mean = 0.0
        self.last_resource_early_arrival_mean = 0.0
        self.last_resource_predicted_lateness_mean = 0.0
        self.last_team_cycle_count = 0
        self.resource_wait_constraint_target = float(getattr(
            self.all_args, 'resource_wait_constraint_target', 0.0
        ))
        self.resource_wait_dual_lr = float(getattr(
            self.all_args, 'resource_wait_dual_lr', 0.0
        ))
        self.resource_wait_dual_max = float(getattr(
            self.all_args, 'resource_wait_dual_max', 0.05
        ))
        self.resource_wait_dual_value = float(getattr(
            self.all_args, 'resource_lateness_coef', 0.0
        ))
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (
                self.resource_wait_constraint_target,
                self.resource_wait_dual_lr,
                self.resource_wait_dual_max,
                self.resource_wait_dual_value,
            )
        ):
            raise ValueError('Resource-wait dual settings must be finite and non-negative.')
        self.resource_wait_dual_enabled = bool(
            self.resource_wait_constraint_target > 0.0
            and self.resource_wait_dual_lr > 0.0
        )
        self.plane_bc_pretrain_epochs = int(
            getattr(self.all_args, 'plane_bc_pretrain_epochs', 0)
        )
        self.plane_bc_only = bool(
            getattr(self.all_args, 'plane_bc_only', False)
        )
        self.plane_bc_teacher_dir = str(
            getattr(self.all_args, 'plane_bc_teacher_dir', '') or ''
        )
        self.plane_bc_lr = float(
            getattr(self.all_args, 'plane_bc_lr', 0.0)
        )
        self.plane_bc_rollouts_per_epoch = int(
            getattr(self.all_args, 'plane_bc_rollouts_per_epoch', 0)
        )
        self.plane_bc_shared_lr_scale = float(
            getattr(self.all_args, 'plane_bc_shared_lr_scale', 0.3)
        )
        self.plane_bc_freeze_shared_epochs = max(
            0, int(getattr(self.all_args, 'plane_bc_freeze_shared_epochs', 0))
        )
        self.plane_bc_pair_loss_coef = float(
            getattr(self.all_args, 'plane_bc_pair_loss_coef', 1.0)
        )
        self.plane_bc_order_loss_coef = float(
            getattr(self.all_args, 'plane_bc_order_loss_coef', 0.0)
        )
        self.plane_bc_initial_weight = float(
            getattr(self.all_args, 'plane_bc_initial_weight', 5.0)
        )
        self.plane_bc_relocation_weight = float(
            getattr(self.all_args, 'plane_bc_relocation_weight', 2.0)
        )
        self.plane_bc_critical_op_weight = float(
            getattr(self.all_args, 'plane_bc_critical_op_weight', 2.0)
        )
        self.plane_bc_tail_start_fraction = float(
            getattr(
                self.all_args, 'plane_bc_tail_start_fraction', 1.0
            )
        )
        self.plane_bc_tail_weight = float(
            getattr(self.all_args, 'plane_bc_tail_weight', 1.0)
        )
        self.plane_bc_tail_final_start_fraction = float(
            getattr(
                self.all_args, 'plane_bc_tail_final_start_fraction', 1.0
            )
        )
        requested_final_weight = float(
            getattr(self.all_args, 'plane_bc_tail_final_weight', -1.0)
        )
        self.plane_bc_tail_final_weight = (
            self.plane_bc_tail_weight
            if requested_final_weight < 0.0
            else requested_final_weight
        )
        dagger_schedule = str(
            getattr(self.all_args, 'plane_bc_dagger_schedule', '') or ''
        ).strip()
        self.plane_bc_dagger_schedule = tuple(
            float(item.strip())
            for item in dagger_schedule.split(',')
            if item.strip()
        )
        staging_dagger_schedule = str(getattr(
            self.all_args, 'plane_bc_staging_dagger_schedule', ''
        ) or '').strip()
        self.plane_bc_staging_dagger_schedule = tuple(
            float(item.strip())
            for item in staging_dagger_schedule.split(',')
            if item.strip()
        )
        self.plane_bc_per_agent_dagger = bool(getattr(
            self.all_args, 'plane_bc_per_agent_dagger', False
        ))
        self.plane_bc_phase_aware = bool(getattr(
            self.all_args, 'plane_bc_phase_aware', False
        ))
        self.plane_bc_service_weight = float(getattr(
            self.all_args, 'plane_bc_service_weight', 1.0
        ))
        self.plane_bc_staging_hold_weight = float(getattr(
            self.all_args, 'plane_bc_staging_hold_weight', 1.0
        ))
        self.plane_bc_staging_move_weight = float(getattr(
            self.all_args, 'plane_bc_staging_move_weight', 2.0
        ))
        self.plane_bc_service_tail_start_fraction = float(getattr(
            self.all_args, 'plane_bc_service_tail_start_fraction', 1.0
        ))
        self.plane_bc_service_tail_weight = float(getattr(
            self.all_args, 'plane_bc_service_tail_weight', 1.0
        ))
        self.plane_bc_dagger_rng = np.random.default_rng(
            int(getattr(self.all_args, 'plane_bc_dagger_seed', 0))
            or (int(self.all_args.seed) + 73001)
        )
        self.plane_bc_dagger_tail_start_fraction = float(
            getattr(
                self.all_args,
                'plane_bc_dagger_tail_start_fraction',
                1.0,
            )
        )
        self.plane_bc_dagger_tail_teacher_rate = float(
            getattr(
                self.all_args,
                'plane_bc_dagger_tail_teacher_rate',
                0.0,
            )
        )
        self.bc_reference_kl_coef = float(
            getattr(self.all_args, 'bc_reference_kl_coef', 0.0)
        )
        self.bc_reference_checkpoint = str(getattr(
            self.all_args, 'bc_reference_checkpoint', ''
        ) or '').strip()
        self.bc_reference_resolved_path = ''
        self.bc_reference_kl_coef_schedule = self._parse_epoch_schedule(
            getattr(self.all_args, 'bc_reference_kl_coef_schedule', ''),
            'bc_reference_kl_coef_schedule',
        )
        self.shared_actor_lr_scale_schedule = self._parse_epoch_schedule(
            getattr(self.all_args, 'shared_actor_lr_scale_schedule', ''),
            'shared_actor_lr_scale_schedule',
        )
        self.iga_potential_beta_schedule = self._parse_epoch_schedule(
            getattr(self.all_args, 'iga_potential_beta_schedule', ''),
            'iga_potential_beta_schedule',
        )
        self.current_iga_potential_beta = float(
            getattr(self.all_args, 'iga_potential_beta', 0.0)
        )
        self.counterfactual_baseline_mix_schedule = (
            self._parse_epoch_schedule(
                getattr(
                    self.all_args,
                    'counterfactual_baseline_mix_schedule',
                    '',
                ),
                'counterfactual_baseline_mix_schedule',
            )
        )
        if any(
            value > 1.0
            for value in self.counterfactual_baseline_mix_schedule
        ):
            raise ValueError(
                '--counterfactual_baseline_mix_schedule values must be in '
                '[0, 1].'
            )
        self.current_counterfactual_baseline_mix = float(getattr(
            self.all_args, 'counterfactual_baseline_mix', 1.0
        ))
        if not 0.0 <= self.current_counterfactual_baseline_mix <= 1.0:
            raise ValueError('--counterfactual_baseline_mix must be in [0, 1].')
        self.bc_reference_target_kl = float(
            getattr(self.all_args, 'bc_reference_target_kl', 0.0)
        )
        self.bc_reference_hard_gate = bool(
            getattr(self.all_args, 'bc_reference_hard_gate', False)
        )
        self.adaptive_bc_reference_kl = bool(getattr(
            self.all_args, 'adaptive_bc_reference_kl', False
        ))
        self.adaptive_actor_kl = bool(
            getattr(self.all_args, 'adaptive_actor_kl', False)
        )
        self.adaptive_actor_kl_low = float(
            getattr(self.all_args, 'adaptive_actor_kl_low', 1e-4)
        )
        self.adaptive_actor_kl_high = float(
            getattr(self.all_args, 'adaptive_actor_kl_high', 5e-4)
        )
        self.adaptive_actor_lr_min_scale = float(
            getattr(self.all_args, 'adaptive_actor_lr_min_scale', 0.25)
        )
        self.adaptive_actor_lr_max_scale = float(
            getattr(self.all_args, 'adaptive_actor_lr_max_scale', 8.0)
        )
        self.adaptive_actor_lr_up = float(
            getattr(self.all_args, 'adaptive_actor_lr_up', 1.5)
        )
        self.adaptive_actor_lr_down = float(
            getattr(self.all_args, 'adaptive_actor_lr_down', 0.5)
        )
        self.adaptive_actor_min_step_completion = float(
            getattr(
                self.all_args,
                'adaptive_actor_min_step_completion',
                0.9,
            )
        )
        self.device_bc_pretrain_epochs = int(getattr(self.all_args, 'device_bc_pretrain_epochs', 0))
        self.device_bc_training_scope = str(getattr(
            self.all_args, 'device_bc_training_scope', 'policy_and_ready'
        ))
        self.device_bc_eval_each_epoch = bool(getattr(
            self.all_args, 'device_bc_eval_each_epoch', False
        ))
        self.device_bc_only = bool(
            getattr(self.all_args, 'device_bc_only', False)
        )
        self.resource_bc_checkpoint = str(
            getattr(self.all_args, 'resource_bc_checkpoint', '') or ''
        )
        self.resource_bc_checkpoint_source = None
        self.device_bc_lr = float(getattr(self.all_args, 'device_bc_lr', 0.0))
        self.device_bc_teacher = str(
            getattr(self.all_args, 'device_bc_teacher', 'heuristic')
        )
        self.device_bc_role_balanced = bool(
            getattr(self.all_args, 'device_bc_role_balanced', False)
        )
        self.device_bc_timing_balanced = bool(
            getattr(self.all_args, 'device_bc_timing_balanced', False)
        )
        self.device_bc_legacy_noop_timing = bool(
            getattr(self.all_args, 'device_bc_legacy_noop_timing', False)
        )
        self.device_bc_min_teacher_score_margin = float(getattr(
            self.all_args, 'device_bc_min_teacher_score_margin', -1.0
        ))
        self.device_bc_ranking_loss_coef = float(getattr(
            self.all_args, 'device_bc_ranking_loss_coef', 0.0
        ))
        self.device_bc_categorical_loss_coef = float(getattr(
            self.all_args, 'device_bc_categorical_loss_coef', 1.0
        ))
        self.device_bc_min_ranking_labels_per_epoch = int(getattr(
            self.all_args, 'device_bc_min_ranking_labels_per_epoch', 0
        ))
        self.device_bc_ranking_temperature = float(getattr(
            self.all_args, 'device_bc_ranking_temperature', 0.10
        ))
        self.device_bc_assignment_loss_coef = float(getattr(
            self.all_args, 'device_bc_assignment_loss_coef', 0.0
        ))
        self.device_bc_assignment_margin_loss_coef = float(getattr(
            self.all_args, 'device_bc_assignment_margin_loss_coef', 0.0
        ))
        self.device_bc_assignment_margin = float(getattr(
            self.all_args, 'device_bc_assignment_margin', 0.20
        ))
        self.device_bc_matching_audit = bool(getattr(self.all_args, 'device_bc_matching_audit', False))
        self.device_bc_full_edge_loss_coef = float(getattr(self.all_args, 'device_bc_full_edge_loss_coef', 0.0))
        self.device_bc_empty_wait_loss_coef = float(getattr(self.all_args, 'device_bc_empty_wait_loss_coef', 0.0))
        self.device_bc_min_wait_groups_per_epoch = int(getattr(self.all_args, 'device_bc_min_wait_groups_per_epoch', 0))
        self.device_bc_min_assignment_labels_per_epoch = int(getattr(
            self.all_args, 'device_bc_min_assignment_labels_per_epoch', 0
        ))
        self.device_bc_timing_loss_coef = float(getattr(
            self.all_args, 'device_bc_timing_loss_coef', 0.0
        ))
        self.request_ready_loss_coef = float(getattr(
            self.all_args, 'request_ready_loss_coef', 1.0
        ))
        self.request_ready_seconds_loss_coef = float(getattr(
            self.all_args, 'request_ready_seconds_loss_coef', 0.0
        ))
        self.request_ready_seconds_loss_scale = float(getattr(
            self.all_args, 'request_ready_seconds_loss_scale', 600.0
        ))
        self.request_ready_h1_weight = float(getattr(
            self.all_args, 'request_ready_h1_weight', 1.0
        ))
        self.request_ready_h2_weight = float(getattr(
            self.all_args, 'request_ready_h2_weight', 1.0
        ))
        self.request_ready_departure_weight = float(getattr(
            self.all_args, 'request_ready_departure_weight', 1.0
        ))
        self.request_ready_underprediction_weight = float(getattr(
            self.all_args, 'request_ready_underprediction_weight', 1.0
        ))
        self.request_ready_blocking_weight = float(getattr(
            self.all_args, 'request_ready_blocking_weight', 1.0
        ))
        self.request_ready_exclude_blocking_loss = bool(getattr(
            self.all_args, 'request_ready_exclude_blocking_loss', False
        ))
        self.request_ready_kind_balanced_loss = bool(getattr(
            self.all_args, 'request_ready_kind_balanced_loss', False
        ))
        self.request_ready_quantile_loss_coef = float(getattr(
            self.all_args, 'request_ready_quantile_loss_coef', 0.0
        ))
        self.request_ready_holdout_folds = int(getattr(
            self.all_args, 'request_ready_holdout_folds', 0
        ))
        self.request_ready_holdout_fold = int(getattr(
            self.all_args, 'request_ready_holdout_fold', 0
        ))
        self.request_ready_min_labels_per_epoch = int(getattr(
            self.all_args, 'request_ready_min_labels_per_epoch', 0
        ))
        self.device_bc_dagger_schedule = self._parse_epoch_schedule(
            getattr(self.all_args, 'device_bc_dagger_schedule', '1.0'),
            'device_bc_dagger_schedule',
        ) or (1.0,)
        self.device_bc_dagger_rng = np.random.default_rng(
            int(getattr(self.all_args, 'device_bc_dagger_seed', 0))
            or (int(self.all_args.seed) + 97001)
        )
        self.device_bc_min_labels_per_epoch = int(getattr(self.all_args, 'device_bc_min_labels_per_epoch', 0))
        self.stage2_max_raw_regression_seconds = float(getattr(
            self.all_args, 'stage2_max_raw_regression_seconds', float('inf')
        ))
        self.stage2_max_stress_regression_seconds = float(getattr(
            self.all_args, 'stage2_max_stress_regression_seconds', float('inf')
        ))
        self.device_bc_min_rollouts_per_epoch = int(getattr(self.all_args, 'device_bc_min_rollouts_per_epoch', 1))
        self.device_bc_max_rollouts_per_epoch = int(getattr(self.all_args, 'device_bc_max_rollouts_per_epoch', 0))
        self.device_bc_train_gnn = bool(getattr(self.all_args, 'device_bc_train_gnn', False))
        self.device_bc_plane_deterministic = bool(getattr(self.all_args, 'device_bc_plane_deterministic', True))
        self.device_bc_reset_optim = bool(getattr(self.all_args, 'device_bc_reset_optim', True))
        self.device_bc_save = bool(getattr(self.all_args, 'device_bc_save', True))
        self.resource_ppo_update_schedule = str(getattr(
            self.all_args, 'resource_ppo_update_schedule', 'joint'
        ))
        self.resource_ppo_warmup_epochs = max(0, int(getattr(
            self.all_args, 'resource_ppo_warmup_epochs', 1
        )))
        self.gnn_freeze_epochs = max(0, int(getattr(self.all_args, 'gnn_freeze_epochs', 0)))
        self.plane_freeze_epochs = max(0, int(getattr(self.all_args, 'plane_freeze_epochs', 0)))
        self.stage2_allow_shared_unfreeze = bool(getattr(
            self.all_args, 'stage2_allow_shared_unfreeze', False
        ))
        self.stage3_allow_shared_frozen = bool(getattr(
            self.all_args, 'stage3_allow_shared_frozen', False
        ))
        self.stage3_handoff_mode = str(getattr(
            self.all_args, 'stage3_handoff_mode', 'strict'
        ))
        self.stage3_contract_transition = None
        self.plane_order_freeze_epochs = max(
            0, int(getattr(self.all_args, 'plane_order_freeze_epochs', 0))
        )
        self.early_stop_patience = max(0, int(getattr(self.all_args, 'early_stop_patience', 0)))
        self.reset_optimizers_on_resume = bool(getattr(self.all_args, 'reset_optimizers_on_resume', False))
        self.reset_value_normalizer_on_resume = bool(getattr(
            self.all_args, 'reset_value_normalizer_on_resume', False
        ))
        self.reset_value_normalizer_before_ppo = bool(getattr(
            self.all_args, 'reset_value_normalizer_before_ppo', False
        ))
        self.strict_checkpoint_contract = bool(getattr(
            self.all_args, 'strict_checkpoint_contract', False
        ))
        self.strict_stage1_reward_contract = bool(getattr(
            self.all_args, 'strict_stage1_reward_contract', False
        ))
        self.stage1_reward_contract_metadata = None
        self.canary_eval_interval_shards = int(getattr(self.all_args, 'canary_eval_interval_shards', 0))
        self.canary_eval_max_per_epoch = int(getattr(
            self.all_args, 'canary_eval_max_per_epoch', 0
        ))
        self.canary_eval_max_cases = int(getattr(
            self.all_args, 'canary_eval_max_cases', 0
        ))
        self.canary_max_regression = float(getattr(self.all_args, 'canary_max_regression', 0.0))
        self.canary_stop_on_regression = bool(getattr(self.all_args, 'canary_stop_on_regression', False))
        self.canary_rejected = False
        self.canary_rejection_info = {}
        self.canary_baseline_makespan = np.inf
        self.evaluation_tau = float(
            getattr(self.all_args, 'evaluation_tau', 0.3)
        )
        self.selection_metric = str(
            getattr(self.all_args, 'selection_metric', 'iid')
        )
        self.selection_weights = {
            'iid': float(getattr(
                self.all_args, 'selection_iid_weight', 0.50
            )),
            'ood_stress': float(getattr(
                self.all_args, 'selection_ood_stress_weight', 0.45
            )),
            'ood_scale': float(getattr(
                self.all_args, 'selection_ood_scale_weight', 0.05
            )),
        }
        self.selection_tail_fraction = float(getattr(
            self.all_args, 'selection_tail_fraction', 0.10
        ))
        self.selection_tail_weight = float(getattr(
            self.all_args, 'selection_tail_weight', 0.25
        ))
        if self.evaluation_tau <= 0.0:
            raise ValueError("--evaluation_tau must be positive.")
        if self.selection_metric not in {
            'raw', 'iid', 'composite', 'composite_tail'
        }:
            raise ValueError(
                f"Unsupported checkpoint selection metric: {self.selection_metric}"
            )
        if any(weight < 0.0 for weight in self.selection_weights.values()):
            raise ValueError("Composite selection weights must be non-negative.")
        if not np.isclose(sum(self.selection_weights.values()), 1.0):
            raise ValueError(
                "Composite selection weights must sum to 1.0, got "
                f"{self.selection_weights}."
            )
        if not 0.0 < self.selection_tail_fraction <= 1.0:
            raise ValueError('--selection_tail_fraction must be in (0, 1].')
        if not 0.0 <= self.selection_tail_weight <= 1.0:
            raise ValueError('--selection_tail_weight must be in [0, 1].')
        self.best_eval_makespan = np.inf
        self.best_eval_iid_makespan = np.inf
        self.best_eval_composite_makespan = np.inf
        self.eval_epochs_without_improvement = 0
        self.exact_resume_stage1 = False
        self.exact_resume_stage2 = False
        self.exact_resume_stage2_supervised = False
        self.device_bc_resume_epoch = 0
        self.device_bc_resume_completed_rollouts = 0
        self.resume_epoch = 0
        self.resume_completed_shards = 0
        self.resume_total_shards = 0
        self.resume_total_num_steps = 0
        self.actor_update_shards = 0
        self.actor_planned_optimizer_steps_total = 0.0
        self.actor_optimizer_steps_total = 0.0
        self.actor_zero_update_shards = 0
        self.actor_incomplete_update_shards = 0
        self.actor_kl_stop_shards = 0
        self.actor_old_policy_kl_stop_shards = 0
        self.actor_bc_reference_kl_stop_shards = 0
        self.actor_bc_reference_target_exceeded_shards = 0
        self.actor_post_update_old_policy_kl_max = 0.0
        self.actor_post_update_bc_reference_kl_max = 0.0
        self.actor_backtrack_failed_groups_total = 0.0
        self.actor_backtrack_groups_total = 0.0
        self.actor_backtrack_retry_attempts_total = 0.0
        self.actor_replay_samples_total = 0.0
        self.actor_empty_replay_samples_total = 0.0
        self._pending_value_normalizer_state = None
        self._pending_role_value_normalizer_states = None
        self.last_eval_case_count = 0
        self.last_eval_case_ids = []
        self.last_eval_records = []
        self.pre_ppo_case_makespan = {}
        self.dataset_manifest_path = str(getattr(self.all_args, 'dataset_manifest', ''))
        self.eval_dataset_dir = str(getattr(self.all_args, 'eval_dataset_dir', ''))
        self.last_eval_completed_count = 0
        self.last_eval_completion_rate = 0.0
        self.last_eval_timeout_count = 0
        self.last_eval_cycle_count = 0
        self.last_eval_mean_steps = 0.0
        self.last_eval_max_no_progress = 0
        self.last_eval_mean_relocations = 0.0
        self.last_eval_raw_makespan = np.inf
        self.last_eval_iid_makespan = np.inf
        self.last_eval_composite_makespan = np.inf
        self.last_eval_tail_makespan = np.inf
        self.last_eval_selection_score = np.inf
        self.eval_canary_rounds = max(
            1,
            int(getattr(self.all_args, 'eval_canary_rounds', 1)),
        )

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.skip_pre_ppo_eval = bool(
            getattr(self.all_args, 'skip_pre_ppo_eval', False)
        )
        self.skip_epoch_eval = bool(
            getattr(self.all_args, 'skip_epoch_eval', False)
        )
        self.eval_interval = max(1, int(self.all_args.eval_interval))
        self.log_interval = self.all_args.log_interval

        # dir
        self.checkpoint_dir = self.all_args.checkpoint_dir
        self.selection_checkpoint_dir = getattr(
            self.all_args, 'selection_checkpoint_dir', None
        )
        if not self.evaluation_only:
            self._validate_training_stage_config()

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.render_dir = str(self.run_dir / 'renders')
            if not os.path.exists(self.render_dir):
                os.makedirs(self.render_dir)
        self.evaluation_dir = os.path.join(str(self.run_dir), 'evaluations')
        os.makedirs(self.evaluation_dir, exist_ok=True)
        self.dataset_case_metadata = self._load_dataset_case_metadata(
            self.dataset_manifest_path
        )

        from onpolicy.algorithms.gnn_mappo.gnn_mappo import MAPPO_Trainer as TrainAlgo
        from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy as Policy

        # share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        
        # policy network
        self.policy = Policy(self.all_args, self.ac_config,
                            device = self.device)

        if self.checkpoint_dir is not None:
            self.restore(self.checkpoint_dir)
        if (
            self.resource_bc_checkpoint
            and not self.evaluation_only
            and not self.exact_resume_stage2
        ):
            self._restore_shared_resource_bc(self.resource_bc_checkpoint)

        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        if self.reset_value_normalizer_before_ppo:
            self._pending_value_normalizer_state = None
            self._pending_role_value_normalizer_states = None
            print(
                '[Info] Starting PPO with fresh ValueNorm statistics '
                '(--reset_value_normalizer_before_ppo).',
                flush=True,
            )
        if self._pending_value_normalizer_state is not None and self.trainer.value_normalizer is not None:
            if len(self._pending_value_normalizer_state) == 0:
                print(
                    "[Warning] Loaded a legacy checkpoint without persisted ValueNorm "
                    "statistics; starting ValueNorm from a fresh state."
                )
            else:
                self.trainer.load_value_normalizer_state(
                    self._pending_value_normalizer_state,
                    self._pending_role_value_normalizer_states,
                )
                mode = 'role-specific' if self.trainer.role_valuenorm else 'shared'
                print(
                    f"[Info] Restored persisted {mode} ValueNorm statistics "
                    "from checkpoint."
                )
        
        # Deterministic evaluation and supervised-only Stage2 need only the
        # recurrent-state tail shape.  Avoid allocating a multi-gigabyte PPO
        # replay buffer for a stage that can never collect a PPO rollout.
        if (
            self.evaluation_only
            or self.training_stage == CANONICAL_RESOURCE_JOINT
        ):
            self.buffer = SimpleNamespace(
                rnn_states=np.zeros(
                    (
                        1, 1, self.num_agents,
                        self.recurrent_N, self.hidden_size,
                    ),
                    dtype=np.float32,
                )
            )
        else:
            self.buffer = SharedReplayBuffer(
                self.all_args, self.num_agents, None, None, None
            )

    @staticmethod
    def _load_paired_case_baselines(path_value):
        """Load a compact map or verified per-case reference-result directory."""
        if not str(path_value or '').strip():
            return {}
        path = Path(path_value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        def add_record(target, payload, fallback):
            if not isinstance(payload, Mapping):
                return
            case = str(
                payload.get('case')
                or payload.get('case_id')
                or fallback
            )
            case = Path(case).name
            makespan = payload.get('makespan')
            if makespan is None:
                makespan = payload.get('cmax')
            if makespan is None:
                return
            makespan = float(makespan)
            if not np.isfinite(makespan) or makespan <= 0.0:
                raise ValueError(
                    f'Invalid paired baseline for {case}: {makespan!r}.'
                )
            previous = target.get(case)
            if previous is not None and not np.isclose(previous, makespan):
                raise ValueError(
                    f'Conflicting paired baselines for {case}: '
                    f'{previous} != {makespan}.'
                )
            target[case] = makespan

        baselines = {}
        if path.is_dir():
            files = sorted(path.glob('case_*.json'))
            for file_path in files:
                add_record(
                    baselines,
                    json.loads(file_path.read_text(encoding='utf-8')),
                    file_path.stem,
                )
        else:
            payload = json.loads(path.read_text(encoding='utf-8'))
            raw_cases = (
                payload.get('cases', payload)
                if isinstance(payload, Mapping) else payload
            )
            if isinstance(raw_cases, Mapping):
                for case, value in raw_cases.items():
                    if isinstance(value, Mapping):
                        add_record(baselines, value, case)
                    else:
                        add_record(
                            baselines,
                            {'case': case, 'makespan': value},
                            case,
                        )
            elif isinstance(raw_cases, list):
                for index, record in enumerate(raw_cases):
                    add_record(baselines, record, f'row_{index}')
        if not baselines:
            raise ValueError(f'No paired case baselines found at {path}.')
        return baselines

    def _validate_training_stage_config(self):
        resource_policy = getattr(self.all_args, 'resource_policy', 'heuristic')
        stage = normalize_training_stage(self.training_stage)
        self.training_stage = stage
        self.all_args.training_stage = stage
        stage1_baseline = str(getattr(
            self.all_args, 'stage1_baseline', 'proposed'
        )).strip().lower().replace('-', '_')
        if stage1_baseline != 'proposed':
            if stage not in {'auto', 'plane_pretrain'}:
                raise ValueError(
                    'L2D, Multi-PPO, FJSP-DRL and DANIEL adapters are '
                    'Stage-1-only; use --training_stage plane_pretrain.'
                )
            if resource_policy != 'heuristic':
                raise ValueError(
                    'Stage-1 learning baselines require '
                    '--resource_policy heuristic so every arm uses the common '
                    'Hungarian resource backend.'
                )
            if str(getattr(
                self.all_args, 'plane_order_mode', 'fixed'
            )) != 'fixed':
                raise ValueError(
                    'Stage-1 learning baselines require '
                    '--plane_order_mode fixed.'
                )
            if str(getattr(
                self.all_args, 'plane_pair_decoder', 'joint_pair'
            )) != 'joint_pair':
                raise ValueError(
                    'Stage-1 learning baselines require '
                    '--plane_pair_decoder joint_pair for the common action '
                    'contract.'
                )
            if (
                self.plane_bc_pretrain_epochs > 0
                and getattr(self, 'baseline_initialization_protocol', '')
                != 'new_semantics_budget_matched'
            ):
                raise ValueError(
                    'Published learning-baseline arms use PPO without the '
                    'proposed IGA behavior-cloning warm start. '
                    'An explicitly selected P5-aligned runner is required '
                    'for budget-matched baseline initialization.'
                )
        resume_stage1 = bool(getattr(self.all_args, 'resume_stage1', False))
        resume_stage2 = bool(getattr(self.all_args, 'resume_stage2', False))
        device_bc_resume_epoch = int(getattr(
            self.all_args, 'device_bc_resume_epoch', 0
        ))
        device_bc_resume_rollouts = int(getattr(
            self.all_args, 'device_bc_resume_completed_rollouts', 0
        ))
        if resume_stage1 and resume_stage2:
            raise ValueError(
                "--resume_stage1 and --resume_stage2 are mutually exclusive."
            )
        if device_bc_resume_epoch < 0 or device_bc_resume_rollouts < 0:
            raise ValueError('DeviceBC recovery cursors must be non-negative.')
        if (
            (device_bc_resume_epoch or device_bc_resume_rollouts)
            and not resume_stage2
        ):
            raise ValueError(
                'DeviceBC recovery cursors require --resume_stage2.'
            )
        if (
            self.reset_optimizers_on_resume
            and not (resume_stage1 or resume_stage2)
        ):
            raise ValueError(
                "--reset_optimizers_on_resume requires an explicit resume flag."
            )
        if (
            self.reset_value_normalizer_on_resume
            and not (resume_stage1 or resume_stage2)
        ):
            raise ValueError(
                "--reset_value_normalizer_on_resume requires an explicit resume flag."
            )
        if self.plane_bc_pretrain_epochs < 0:
            raise ValueError("--plane_bc_pretrain_epochs must be non-negative.")
        if self.plane_bc_rollouts_per_epoch < 0:
            raise ValueError("--plane_bc_rollouts_per_epoch must be non-negative.")
        if self.device_bc_pretrain_epochs < 0:
            raise ValueError("--device_bc_pretrain_epochs must be non-negative.")
        from onpolicy.utils.stage2_policy_transfer import validate_ready_target_collection
        validate_ready_target_collection(
            skip=bool(getattr(self.all_args, 'device_bc_skip_ready_targets', False)),
            stage=self.training_stage, scope=self.device_bc_training_scope,
            ready_loss=self.request_ready_loss_coef,
            minimum_labels=self.request_ready_min_labels_per_epoch,
            teacher=self.device_bc_teacher, teacher_rates=self.device_bc_dagger_schedule,
        )
        if self.device_bc_training_scope not in {
            'policy_and_ready', 'ready_only', 'policy_frozen_ready'
        }:
            raise ValueError(
                'Unsupported --device_bc_training_scope.'
            )
        if self.device_bc_eval_each_epoch and (
            self.training_stage != CANONICAL_RESOURCE_JOINT
            or self.device_bc_training_scope == 'ready_only'
            or not self.use_eval
        ):
            raise ValueError('Per-epoch BC selection requires Stage2 policy BC and --use_eval.')
        if self.device_bc_training_scope == 'policy_frozen_ready':
            if self.training_stage != CANONICAL_RESOURCE_JOINT:
                raise ValueError('Frozen-ready policy BC is a canonical Stage2 mode.')
            if not getattr(self.all_args, 'request_ready_checkpoint', '') or not getattr(
                self.all_args, 'request_ready_checkpoint_sha256', ''
            ):
                raise ValueError('Frozen-ready BC requires a predictor source and SHA256.')
            if self.request_ready_loss_coef != 0.0:
                raise ValueError('Frozen-ready BC must disable the ready regression objective.')
        elif getattr(self.all_args, 'request_ready_checkpoint', ''):
            raise ValueError('A ready source is only supported by policy_frozen_ready BC.')
        warmstart = getattr(self.all_args, 'stage2_policy_warmstart_checkpoint', '')
        warmstart_sha = getattr(self.all_args, 'stage2_policy_warmstart_sha256', '')
        replay_path = getattr(self.all_args, 'stage2_policy_warmstart_evaluation', '')
        replay_sha = getattr(self.all_args, 'stage2_policy_warmstart_evaluation_sha256', '')
        if bool(warmstart) != bool(warmstart_sha) or bool(replay_path) != bool(replay_sha):
            raise ValueError('Policy warm-start files require their immutable SHA256 digests.')
        if (replay_path and not warmstart) or (warmstart and (
                self.training_stage != CANONICAL_RESOURCE_JOINT
                or self.device_bc_training_scope != 'policy_frozen_ready'
                or not self.device_bc_eval_each_epoch
                or not getattr(self.all_args, 'stage2_bc_deterministic', False))):
            raise ValueError('Policy warm start requires deterministic frozen-ready Stage2 with epoch evaluation.')
        if not np.isfinite(self.device_bc_min_teacher_score_margin):
            raise ValueError(
                "--device_bc_min_teacher_score_margin must be finite."
            )
        if (
            not np.isfinite(self.device_bc_ranking_loss_coef)
            or self.device_bc_ranking_loss_coef < 0.0
        ):
            raise ValueError(
                '--device_bc_ranking_loss_coef must be finite and non-negative.'
            )
        for name, value in (
            ('device_bc_categorical_loss_coef', self.device_bc_categorical_loss_coef),
            ('device_bc_assignment_loss_coef', self.device_bc_assignment_loss_coef),
            (
                'device_bc_assignment_margin_loss_coef',
                self.device_bc_assignment_margin_loss_coef,
            ),
            ('device_bc_assignment_margin', self.device_bc_assignment_margin),
            ('device_bc_full_edge_loss_coef', self.device_bc_full_edge_loss_coef),
            ('device_bc_empty_wait_loss_coef', self.device_bc_empty_wait_loss_coef),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f'--{name} must be finite and non-negative.'
                )
        if self.device_bc_min_wait_groups_per_epoch < 0:
            raise ValueError('Minimum meaningful wait groups must be nonnegative.')
        if (self.device_bc_full_edge_loss_coef > 0 or self.device_bc_empty_wait_loss_coef > 0
                or self.device_bc_min_wait_groups_per_epoch > 0) and not self.device_bc_matching_audit:
            raise ValueError('New full-edge/wait objectives require --device_bc_matching_audit.')
        if self.device_bc_matching_audit and (
                self.training_stage != CANONICAL_RESOURCE_JOINT
                or self.device_bc_training_scope != 'policy_frozen_ready'
                or not getattr(self.all_args, 'device_global_matching', False)
                or self.device_bc_teacher != 'iga'):
            raise ValueError('Matching audit requires frozen-ready resource_joint with IGA and global matching.')
        if getattr(self.all_args, 'device_bc_teacher_deployment_projection', False) and (
                not self.device_bc_matching_audit
                or not getattr(self.all_args, 'device_bc_skip_ready_targets', False)):
            raise ValueError('Derived teacher requires matching audit and disabled factual Ready labels.')
        if self.device_bc_min_assignment_labels_per_epoch < 0:
            raise ValueError(
                '--device_bc_min_assignment_labels_per_epoch must be non-negative.'
            )
        if (
            not np.isfinite(self.device_bc_ranking_temperature)
            or self.device_bc_ranking_temperature <= 0.0
        ):
            raise ValueError(
                '--device_bc_ranking_temperature must be finite and positive.'
            )
        if self.device_bc_min_ranking_labels_per_epoch < 0:
            raise ValueError(
                '--device_bc_min_ranking_labels_per_epoch must be non-negative.'
            )
        if (
            not np.isfinite(self.device_bc_timing_loss_coef)
            or self.device_bc_timing_loss_coef < 0.0
        ):
            raise ValueError(
                '--device_bc_timing_loss_coef must be finite and non-negative.'
            )
        if (
            not np.isfinite(self.request_ready_loss_coef)
            or self.request_ready_loss_coef < 0.0
        ):
            raise ValueError(
                '--request_ready_loss_coef must be finite and non-negative.'
            )
        if (
            not np.isfinite(self.request_ready_seconds_loss_coef)
            or self.request_ready_seconds_loss_coef < 0.0
        ):
            raise ValueError(
                '--request_ready_seconds_loss_coef must be finite and '
                'non-negative.'
            )
        if (
            not np.isfinite(self.request_ready_seconds_loss_scale)
            or self.request_ready_seconds_loss_scale <= 0.0
        ):
            raise ValueError(
                '--request_ready_seconds_loss_scale must be finite and '
                'positive.'
            )
        if (
            not np.isfinite(self.request_ready_quantile_loss_coef)
            or self.request_ready_quantile_loss_coef < 0.0
        ):
            raise ValueError(
                '--request_ready_quantile_loss_coef must be finite and '
                'non-negative.'
            )
        if self.request_ready_holdout_folds < 0:
            raise ValueError(
                '--request_ready_holdout_folds must be non-negative.'
            )
        if self.request_ready_holdout_folds:
            if not 0 <= self.request_ready_holdout_fold < self.request_ready_holdout_folds:
                raise ValueError(
                    '--request_ready_holdout_fold must be in '
                    '[0, request_ready_holdout_folds).'
                )
            if self.device_bc_training_scope != 'ready_only':
                raise ValueError(
                    'request-ready holdout validation is only supported for '
                    'ready_only Stage2.'
                )
        if self.request_ready_exclude_blocking_loss and not bool(getattr(
            self.all_args, 'request_ready_hard_blocking', False
        )):
            raise ValueError(
                '--request_ready_exclude_blocking_loss requires '
                '--request_ready_hard_blocking.'
            )
        if self.request_ready_quantile_loss_coef > 0.0 and not bool(getattr(
            self.all_args, 'request_ready_quantile_head', False
        )):
            raise ValueError(
                '--request_ready_quantile_loss_coef requires '
                '--request_ready_quantile_head.'
            )
        for name, value in (
            (
                'request_ready_underprediction_weight',
                self.request_ready_underprediction_weight,
            ),
            ('request_ready_blocking_weight', self.request_ready_blocking_weight),
            ('request_ready_h1_weight', self.request_ready_h1_weight),
            ('request_ready_h2_weight', self.request_ready_h2_weight),
            (
                'request_ready_departure_weight',
                self.request_ready_departure_weight,
            ),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'--{name} must be finite and positive.')
        for name, value in (
            ('stage2_max_raw_regression_seconds', self.stage2_max_raw_regression_seconds),
            (
                'stage2_max_stress_regression_seconds',
                self.stage2_max_stress_regression_seconds,
            ),
        ):
            if np.isnan(value) or value < 0.0:
                raise ValueError(f'--{name} must be non-negative or infinity.')
        if self.request_ready_min_labels_per_epoch < 0:
            raise ValueError(
                '--request_ready_min_labels_per_epoch must be non-negative.'
            )
        if self.device_bc_min_labels_per_epoch < 0:
            raise ValueError(
                "--device_bc_min_labels_per_epoch must be non-negative."
            )
        if self.device_bc_min_rollouts_per_epoch < 1:
            raise ValueError(
                "--device_bc_min_rollouts_per_epoch must be at least one."
            )
        if self.device_bc_max_rollouts_per_epoch < 0:
            raise ValueError(
                "--device_bc_max_rollouts_per_epoch must be non-negative."
            )
        if self.device_bc_teacher not in {'heuristic', 'iga', 'joint_iga'}:
            raise ValueError(
                "--device_bc_teacher must be heuristic, iga, or joint_iga."
            )
        if any(rate > 1.0 for rate in self.device_bc_dagger_schedule):
            raise ValueError(
                'Every --device_bc_dagger_schedule rate must be in [0, 1].'
            )
        if self.resource_ppo_update_schedule not in {
            'joint', 'ordinary_then_joint', 'r014_then_joint'
        }:
            raise ValueError('Invalid --resource_ppo_update_schedule.')
        if (
            self.resource_ppo_update_schedule != 'joint'
            and self.resource_ppo_warmup_epochs >= self.num_episodes
        ):
            raise ValueError(
                'Role-specific resource PPO warmup must leave a joint epoch.'
            )
        if self.plane_bc_shared_lr_scale < 0.0:
            raise ValueError("--plane_bc_shared_lr_scale must be non-negative.")
        if self.plane_bc_pair_loss_coef < 0.0 or self.plane_bc_order_loss_coef < 0.0:
            raise ValueError("Plane BC loss coefficients must be non-negative.")
        bc_stratum_weights = (
            self.plane_bc_initial_weight,
            self.plane_bc_relocation_weight,
            self.plane_bc_critical_op_weight,
        )
        if any(weight <= 0.0 for weight in bc_stratum_weights):
            raise ValueError('All plane BC stratum weights must be positive.')
        if not 0.0 <= self.plane_bc_tail_start_fraction <= 1.0:
            raise ValueError(
                '--plane_bc_tail_start_fraction must be in [0, 1].'
            )
        if self.plane_bc_tail_weight < 1.0:
            raise ValueError('--plane_bc_tail_weight must be at least 1.')
        if not (
            self.plane_bc_tail_start_fraction
            <= self.plane_bc_tail_final_start_fraction
            <= 1.0
        ):
            raise ValueError(
                '--plane_bc_tail_final_start_fraction must be between '
                '--plane_bc_tail_start_fraction and 1.'
            )
        if self.plane_bc_tail_final_weight < self.plane_bc_tail_weight:
            raise ValueError(
                '--plane_bc_tail_final_weight must be at least '
                '--plane_bc_tail_weight.'
            )
        if not 0.0 <= self.plane_bc_dagger_tail_start_fraction <= 1.0:
            raise ValueError(
                '--plane_bc_dagger_tail_start_fraction must be in [0, 1].'
            )
        if not 0.0 <= self.plane_bc_dagger_tail_teacher_rate <= 1.0:
            raise ValueError(
                '--plane_bc_dagger_tail_teacher_rate must be in [0, 1].'
            )
        if any(
            rate < 0.0 or rate > 1.0
            for rate in self.plane_bc_dagger_schedule
        ):
            raise ValueError(
                "Every --plane_bc_dagger_schedule rate must be in [0, 1]."
            )
        if self.plane_bc_pair_loss_coef + self.plane_bc_order_loss_coef <= 0.0:
            raise ValueError("At least one plane BC loss coefficient must be positive.")
        plane_order_mode = str(
            getattr(self.all_args, 'plane_order_mode', 'fixed')
        )
        if plane_order_mode == 'fixed':
            if self.plane_bc_order_loss_coef != 0.0:
                raise ValueError(
                    "fixed plane order requires --plane_bc_order_loss_coef 0."
                )
            if self.plane_order_freeze_epochs != 0:
                raise ValueError(
                    "fixed plane order requires --plane_order_freeze_epochs 0."
                )
        if self.bc_reference_kl_coef < 0.0 or self.bc_reference_target_kl < 0.0:
            raise ValueError("BC-reference KL settings must be non-negative.")
        reference_enabled = (
            self.bc_reference_kl_coef > 0.0
            or self.bc_reference_target_kl > 0.0
            or any(value > 0.0 for value in self.bc_reference_kl_coef_schedule)
        )
        if self.bc_reference_checkpoint:
            reference_path = Path(
                self.bc_reference_checkpoint
            ).expanduser().resolve()
            if not reference_enabled:
                raise ValueError(
                    '--bc_reference_checkpoint requires an enabled '
                    'BC-reference KL coefficient or monitoring target.'
                )
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            self.bc_reference_checkpoint = str(reference_path)
            self.all_args.bc_reference_checkpoint = str(reference_path)
        if self.bc_reference_hard_gate and self.bc_reference_target_kl <= 0.0:
            raise ValueError(
                "--bc_reference_hard_gate requires a positive "
                "--bc_reference_target_kl."
            )
        if not (
            0.0 < self.adaptive_actor_kl_low
            < self.adaptive_actor_kl_high
        ):
            raise ValueError(
                "Adaptive actor KL requires 0 < low < high."
            )
        if not (
            0.0 < self.adaptive_actor_lr_min_scale
            <= self.adaptive_actor_lr_max_scale
        ):
            raise ValueError("Adaptive actor LR scale bounds are invalid.")
        if (
            self.adaptive_actor_lr_up <= 1.0
            or not 0.0 < self.adaptive_actor_lr_down < 1.0
        ):
            raise ValueError(
                "Adaptive actor LR multipliers require up > 1 and 0 < down < 1."
            )
        if not 0.0 < self.adaptive_actor_min_step_completion <= 1.0:
            raise ValueError(
                "--adaptive_actor_min_step_completion must be in (0, 1]."
            )
        if self.plane_bc_pretrain_epochs > 0:
            if stage not in {'auto', 'plane_pretrain'}:
                raise ValueError(
                    "Plane BC is only valid for plane_pretrain; canonical "
                    "joint_finetune is pure RL."
                )
            if not self.plane_bc_teacher_dir:
                raise ValueError(
                    "--plane_bc_teacher_dir is required when Stage1 plane "
                    "BC is enabled."
                )
            if not os.path.isdir(self.plane_bc_teacher_dir):
                raise ValueError(
                    "IGA teacher directory does not exist: "
                    f"{self.plane_bc_teacher_dir}"
                )
        if self.canary_eval_interval_shards < 0:
            raise ValueError("--canary_eval_interval_shards must be non-negative.")
        if self.canary_eval_max_per_epoch < 0:
            raise ValueError('--canary_eval_max_per_epoch must be non-negative.')
        if self.canary_eval_max_cases < 0:
            raise ValueError('--canary_eval_max_cases must be non-negative.')
        if (
            self.canary_eval_max_cases > 0
            and getattr(self, 'shared_eval_client', None) is None
        ):
            raise ValueError(
                '--canary_eval_max_cases requires the shared evaluator.'
            )
        if self.canary_max_regression < 0.0:
            raise ValueError("--canary_max_regression must be non-negative.")
        if self.canary_eval_interval_shards > 0 and not self.use_eval:
            raise ValueError("shard canary evaluation requires --use_eval.")
        if self.skip_pre_ppo_eval and not self.use_eval:
            raise ValueError("--skip_pre_ppo_eval requires --use_eval.")
        if self.canary_stop_on_regression and self.canary_eval_interval_shards <= 0:
            raise ValueError(
                "--canary_stop_on_regression requires "
                "--canary_eval_interval_shards > 0."
            )
        if self.plane_bc_only:
            if self.plane_bc_pretrain_epochs <= 0:
                raise ValueError('--plane_bc_only requires PlaneBC epochs.')
            if not self.use_eval:
                raise ValueError('--plane_bc_only requires --use_eval.')
        if self.device_bc_only:
            if self.device_bc_pretrain_epochs <= 0:
                raise ValueError('--device_bc_only requires DeviceBC epochs.')
            if not self.use_eval:
                raise ValueError('--device_bc_only requires --use_eval.')
            if stage != CANONICAL_RESOURCE_JOINT:
                raise ValueError(
                    '--device_bc_only is valid only for resource_joint.'
                )
        if stage == 'auto':
            if self.device_bc_pretrain_epochs > 0:
                raise ValueError(
                    "Resource BC settings require the explicit canonical "
                    "--training_stage resource_joint; auto cannot infer the "
                    "Stage-1 M2 hand-off contract."
                )
            return
        if stage == 'plane_pretrain':
            if resource_policy != 'heuristic':
                raise ValueError("plane_pretrain requires resource_policy='heuristic'.")
            if resume_stage2:
                raise ValueError(
                    "--resume_stage2 is only valid for resource_joint recovery."
                )
            if self.checkpoint_dir is not None and not resume_stage1:
                raise ValueError(
                    "plane_pretrain checkpoint restore requires explicit --resume_stage1."
                )
            if resume_stage1 and self.checkpoint_dir is None:
                raise ValueError("--resume_stage1 requires --checkpoint_dir.")
            if (
                self.strict_stage1_reward_contract
                and self.checkpoint_dir is not None
                and not self.reset_value_normalizer_on_resume
            ):
                raise ValueError(
                    'Strict Stage-1 reward ablations must reset ValueNorm '
                    'when restoring the shared BC checkpoint.'
                )
            if self.strict_stage1_reward_contract:
                self.stage1_reward_contract_metadata = stage1_reward_contract(
                    reward_mode=self.hindsight_reward_mode,
                    reward_coef=self.reward_coef,
                    hindsight_cmax_coef=getattr(
                        self.all_args, 'hindsight_cmax_coef', 0.0
                    ),
                    terminal_cmax_coef=getattr(
                        self.all_args, 'hindsight_terminal_cmax_coef', 0.0
                    ),
                    gamma=getattr(self.all_args, 'gamma', 1.0),
                    potential_gamma=getattr(
                        self.all_args, 'iga_potential_gamma', 0.99
                    ),
                )
            if self.selection_checkpoint_dir is not None and not resume_stage1:
                raise ValueError(
                    "--selection_checkpoint_dir is only valid with --resume_stage1."
                )
            if self.device_bc_pretrain_epochs != 0:
                raise ValueError("plane_pretrain cannot run device BC.")
            if self.plane_freeze_epochs != 0:
                raise ValueError("plane_pretrain must not freeze the complete plane actor.")
            if (
                (self.bc_reference_kl_coef > 0.0 or self.bc_reference_target_kl > 0.0)
                and self.plane_bc_pretrain_epochs <= 0
                and self.checkpoint_dir is None
            ):
                raise ValueError(
                    "BC-reference PPO requires plane BC in the current run or a "
                    "checkpoint with a sibling checkpoint_PlaneBC.pt."
                )
            if self.num_episodes <= 0:
                raise ValueError("plane_pretrain requires at least one PPO epoch.")
            return
        if stage == CANONICAL_RESOURCE_JOINT:
            if resource_policy != 'drl':
                raise ValueError(
                    "resource_joint requires resource_policy='drl'."
                )
            if self.checkpoint_dir is None:
                raise ValueError(
                    "resource_joint requires the Stage-1 M2 checkpoint "
                    "via --checkpoint_dir."
                )
            if self.device_bc_pretrain_epochs <= 0:
                raise ValueError(
                    'resource_joint is supervised-only and requires positive '
                    '--device_bc_pretrain_epochs.'
                )
            if self.resource_bc_checkpoint:
                raise ValueError(
                    'resource_joint no longer branches into PPO and does not '
                    'accept --resource_bc_checkpoint.'
                )
            if self.num_episodes != 0:
                raise ValueError(
                    'resource_joint has no PPO phase; --num_episodes must be 0.'
                )
            if int(self.all_args.ppo_epoch) != 0:
                raise ValueError(
                    'resource_joint has no PPO update loop; --ppo_epoch must be 0.'
                )
            if not bool(getattr(self.all_args, 'request_ready_prediction', False)):
                raise ValueError(
                    'resource_joint requires --request_ready_prediction.'
                )
            if (self.request_ready_loss_coef <= 0.0
                    and self.device_bc_training_scope != 'policy_frozen_ready'):
                raise ValueError(
                    'resource_joint requires --request_ready_loss_coef > 0.'
                )
            if (self.request_ready_min_labels_per_epoch <= 0
                    and not getattr(self.all_args, 'device_bc_skip_ready_targets', False)):
                raise ValueError(
                    'resource_joint requires a positive '
                    '--request_ready_min_labels_per_epoch.'
                )
            if self.device_bc_training_scope == 'ready_only':
                policy_coefficients = {
                    'device_bc_categorical_loss_coef': (
                        self.device_bc_categorical_loss_coef
                    ),
                    'device_bc_ranking_loss_coef': (
                        self.device_bc_ranking_loss_coef
                    ),
                    'device_bc_timing_loss_coef': (
                        self.device_bc_timing_loss_coef
                    ),
                    'device_bc_assignment_loss_coef': (
                        self.device_bc_assignment_loss_coef
                    ),
                    'device_bc_assignment_margin_loss_coef': (
                        self.device_bc_assignment_margin_loss_coef
                    ),
                }
                nonzero = {
                    name: value for name, value in policy_coefficients.items()
                    if value > 0.0
                }
                if nonzero:
                    raise ValueError(
                        'ready_only Stage2 must disable every resource-policy '
                        f'supervision coefficient; got {nonzero}.'
                    )
                if str(getattr(
                    self.all_args, 'request_ready_policy_injection', 'none'
                )) != 'none':
                    raise ValueError(
                        'ready_only Stage2 requires '
                        '--request_ready_policy_injection none so predictor '
                        'changes cannot alter the behavior trajectory.'
                    )
            elif (
                self.device_bc_categorical_loss_coef <= 0.0
                and self.device_bc_assignment_loss_coef <= 0.0
            ):
                raise ValueError(
                    'resource_joint requires categorical or final-matching '
                    'mobile-policy supervision.'
                )
            if (
                self.device_bc_assignment_loss_coef > 0.0
                and self.device_bc_min_assignment_labels_per_epoch <= 0
            ):
                raise ValueError(
                    'final-matching Stage2 supervision requires a positive '
                    '--device_bc_min_assignment_labels_per_epoch.'
                )
            if (
                self.device_bc_assignment_loss_coef > 0.0
                and not bool(getattr(self.all_args, 'device_global_matching', False))
            ):
                raise ValueError(
                    'permutation-invariant matching supervision requires '
                    '--device_global_matching at training and evaluation.'
                )
            if not self.device_bc_plane_deterministic:
                raise ValueError(
                    'resource_joint exact intrinsic-ready labels are bound to '
                    'the frozen deterministic plane trajectory; remove '
                    '--device_bc_stochastic_plane.'
                )
            if self.device_bc_teacher == 'iga' and (
                not str(getattr(
                    self.all_args, 'resource_iga_teacher_dir', ''
                ) or '').strip()
                or not str(getattr(
                    self.all_args, 'resource_iga_teacher_index', ''
                ) or '').strip()
            ):
                raise ValueError(
                    'resource_joint with --device_bc_teacher iga requires '
                    'both --resource_iga_teacher_dir and '
                    '--resource_iga_teacher_index containing the exact '
                    'intrinsic-ready label schema.'
                )
            if self.device_bc_teacher == 'joint_iga' and (
                not str(getattr(
                    self.all_args, 'joint_iga_teacher_dir', ''
                ) or '').strip()
                or not str(getattr(
                    self.all_args, 'joint_iga_teacher_index', ''
                ) or '').strip()
            ):
                raise ValueError(
                    'resource_joint with --device_bc_teacher joint_iga '
                    'requires both --joint_iga_teacher_dir and '
                    '--joint_iga_teacher_index containing the exact '
                    'intrinsic-ready label schema.'
                )
            if self.device_bc_train_gnn:
                raise ValueError(
                    "resource_joint supervision must not train the shared GNN; "
                    "remove --device_bc_train_gnn."
                )
            if self.device_bc_reset_optim:
                raise ValueError(
                    'resource_joint has no PPO optimizer boundary; pass '
                    '--no_device_bc_reset_optim.'
                )
            if not self.device_bc_save:
                raise ValueError(
                    "resource_joint requires its final supervised checkpoint; "
                    "--no_device_bc_save is rejected."
                )
            if (
                self.device_bc_pretrain_epochs > 0
                and self.device_bc_min_labels_per_epoch <= 0
            ):
                raise ValueError(
                    "resource_joint requires a positive "
                    "--device_bc_min_labels_per_epoch; insufficient BC labels "
                    "must fail closed."
                )
            if (
                self.device_bc_pretrain_epochs > 0
                and self.device_bc_max_rollouts_per_epoch <= 0
            ):
                raise ValueError(
                    "resource_joint requires --device_bc_max_rollouts_per_epoch > 0."
                )
            if self.plane_freeze_epochs != 0 or self.gnn_freeze_epochs != 0:
                raise ValueError(
                    'resource_joint has no PPO freeze schedule; set both '
                    '--plane_freeze_epochs and --gnn_freeze_epochs to 0.'
                )
            if self.stage2_allow_shared_unfreeze:
                raise ValueError(
                    'resource_joint always freezes the shared encoder during '
                    'supervised training; remove --stage2_allow_shared_unfreeze.'
                )
            if resume_stage1:
                raise ValueError(
                    "resource_joint consumes an immutable Stage-1 M2 source; "
                    "--resume_stage1 is only valid for plane_pretrain recovery."
                )
            if resume_stage2:
                if device_bc_resume_epoch >= self.device_bc_pretrain_epochs:
                    raise ValueError(
                        '--device_bc_resume_epoch must identify an unfinished '
                        'supervised epoch.'
                    )
                if (
                    device_bc_resume_rollouts
                    >= self.device_bc_max_rollouts_per_epoch
                ):
                    raise ValueError(
                        '--device_bc_resume_completed_rollouts must be smaller '
                        'than the per-epoch rollout budget.'
                    )
            if self.selection_checkpoint_dir is not None:
                raise ValueError(
                    "resource_joint does not accept --selection_checkpoint_dir; "
                    "checkpoint selection must start from the strict M2 source."
                )
            return
        if stage == CANONICAL_JOINT_FINETUNE:
            plane_teacher_dir = str(self.plane_bc_teacher_dir or '')
            resource_teacher_dir = str(getattr(
                self.all_args, 'resource_iga_teacher_dir', ''
            ) or '')
            resource_teacher_index = str(getattr(
                self.all_args, 'resource_iga_teacher_index', ''
            ) or '')
            joint_teacher_dir = str(getattr(
                self.all_args, 'joint_iga_teacher_dir', ''
            ) or '')
            joint_teacher_index = str(getattr(
                self.all_args, 'joint_iga_teacher_index', ''
            ) or '')
            if resource_policy != 'drl':
                raise ValueError(
                    "joint_finetune requires resource_policy='drl'."
                )
            if self.checkpoint_dir is None:
                raise ValueError(
                    "joint_finetune requires a selected Stage-2 checkpoint "
                    "via --checkpoint_dir."
                )
            if not bool(getattr(self.all_args, 'request_ready_prediction', False)):
                raise ValueError(
                    'joint_finetune must instantiate --request_ready_prediction '
                    'to load the supervised Stage2 architecture exactly.'
                )
            if resume_stage1 or resume_stage2:
                raise ValueError(
                    "joint_finetune is a strict Stage-2 hand-off, not a "
                    "Stage-1/Stage-2 exact recovery run."
                )
            if self.selection_checkpoint_dir is not None:
                raise ValueError(
                    "joint_finetune selects its source before launch and does "
                    "not accept --selection_checkpoint_dir."
                )
            if self.resource_bc_checkpoint:
                raise ValueError(
                    "joint_finetune is a pure-RL transition and does not "
                    "accept --resource_bc_checkpoint."
                )
            if self.plane_bc_pretrain_epochs != 0:
                raise ValueError(
                    "joint_finetune starts directly from the trained Stage2 "
                    "policy; --plane_bc_pretrain_epochs must be 0."
                )
            if self.device_bc_pretrain_epochs != 0:
                raise ValueError(
                    "joint_finetune uses pure joint PPO; "
                    "--device_bc_pretrain_epochs must be 0."
                )
            if any((
                plane_teacher_dir,
                resource_teacher_dir,
                resource_teacher_index,
                joint_teacher_dir,
                joint_teacher_index,
            )):
                raise ValueError(
                    "joint_finetune does not consume teacher supervision; "
                    "remove all Plane/Resource/Joint IGA teacher paths."
                )
            if any(
                value > 0.0
                for value in (
                    *self.plane_bc_dagger_schedule,
                    *self.plane_bc_staging_dagger_schedule,
                    *self.device_bc_dagger_schedule,
                )
            ):
                raise ValueError(
                    "joint_finetune pure RL requires every BC/DAgger teacher "
                    "schedule to be 0."
                )
            if (
                self.bc_reference_kl_coef > 0.0
                or self.bc_reference_target_kl > 0.0
                or self.bc_reference_hard_gate
                or any(
                    value > 0.0
                    for value in self.bc_reference_kl_coef_schedule
                )
            ):
                raise ValueError(
                    "joint_finetune pure RL must not use the BC-reference "
                    "penalty or gate."
                )
            if not self.use_eval or self.skip_pre_ppo_eval:
                raise ValueError(
                    "joint_finetune requires deterministic Pre-PPO validation "
                    "to measure the unmodified Stage2 hand-off."
                )
            if self.num_episodes <= 0:
                raise ValueError(
                    "joint_finetune requires at least one joint PPO epoch."
                )
            if self.plane_freeze_epochs >= self.num_episodes:
                raise ValueError(
                    "joint_finetune must update the plane actor during PPO."
                )
            if (
                self.gnn_freeze_epochs >= self.num_episodes
                and not self.stage3_allow_shared_frozen
            ):
                raise ValueError(
                    "joint_finetune must eventually update the shared encoder."
                )
            if (
                self.stage3_allow_shared_frozen
                and self.gnn_freeze_epochs < self.num_episodes
            ):
                raise ValueError(
                    '--stage3_allow_shared_frozen requires '
                    '--gnn_freeze_epochs >= --num_episodes.'
                )
            shared_pcgrad = bool(getattr(
                self.all_args, 'shared_encoder_pcgrad', False
            ))
            shared_gradient_method = str(getattr(
                self.all_args, 'shared_gradient_method', 'sum'
            ))
            if (
                shared_pcgrad or shared_gradient_method != 'sum'
            ) and not bool(getattr(
                self.all_args, 'role_atomic_ppo', False
            )):
                raise ValueError(
                    'Shared gradient surgery requires --role_atomic_ppo.'
                )
            if not self.stage3_allow_shared_frozen:
                post_freeze_scales = [
                    self._epoch_schedule_value(
                        self.shared_actor_lr_scale_schedule,
                        episode,
                        getattr(self.all_args, 'shared_actor_lr_scale', 1.0),
                    )
                    for episode in range(
                        self.gnn_freeze_epochs, self.num_episodes
                    )
                ]
                if not any(value > 0.0 for value in post_freeze_scales):
                    raise ValueError(
                        'Stage3 eventually-unfrozen encoder requires a '
                        'positive shared Actor LR scale after the freeze window.'
                    )
            if self.resource_ppo_update_schedule != 'joint':
                raise ValueError(
                    "joint_finetune requires simultaneous resource PPO updates."
                )
            if not bool(getattr(self.all_args, 'joint_team_ppo', False)):
                raise ValueError(
                    "joint_finetune requires --joint_team_ppo."
                )
            if str(getattr(
                self.all_args, 'joint_team_ppo_scope', 'plane'
            )) != 'all':
                raise ValueError(
                    "joint_finetune requires --joint_team_ppo_scope all."
                )
            if bool(getattr(
                self.all_args, 'role_event_returns', False
            )) and not self.team_time_return_mode:
                raise ValueError(
                    '--role_event_returns requires '
                    'a team_time reward mode.'
                )
            return

    @staticmethod
    def _validate_graph_batch_memory_config(
        mini_batch_size,
        data_chunk_length,
        max_graphs,
        n_rollout_threads=None,
    ):
        mini_batch_size = int(mini_batch_size)
        data_chunk_length = int(data_chunk_length)
        max_graphs = int(max_graphs)
        if mini_batch_size <= 0 or data_chunk_length <= 0:
            raise ValueError("mini_batch_size and data_chunk_length must both be positive.")
        if n_rollout_threads is not None:
            n_rollout_threads = int(n_rollout_threads)
            if n_rollout_threads <= 0:
                raise ValueError("n_rollout_threads must be positive.")
            if mini_batch_size > n_rollout_threads:
                raise ValueError(
                    "Unsafe recurrent PPO batch: mini_batch_size cannot "
                    "exceed n_rollout_threads because environment slices "
                    "are the recurrent mini-batch axis: "
                    f"{mini_batch_size} > {n_rollout_threads}."
                )
        graphs_per_forward = mini_batch_size * data_chunk_length
        if max_graphs > 0 and graphs_per_forward > max_graphs:
            raise ValueError(
                "Unsafe graph PPO batch: mini_batch_size * data_chunk_length "
                f"= {graphs_per_forward}, configured limit = {max_graphs}."
            )
        return graphs_per_forward

    def _reset_cuda_peak_memory(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def _record_and_clear_cuda_memory(self, train_infos, episode, shard_idx):
        if self.device.type != 'cuda':
            return
        gib = float(1024 ** 3)
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / gib
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / gib
        train_infos['cuda_peak_allocated_gib'] = peak_allocated
        train_infos['cuda_peak_reserved_gib'] = peak_reserved
        train_infos['graphs_per_forward'] = float(self.graphs_per_forward)
        print(
            f"[Memory] epoch={episode + 1}, shard={shard_idx + 1}/{self.num_envs}, "
            f"graphs_per_forward={self.graphs_per_forward}, "
            f"peak_allocated_gib={peak_allocated:.3f}, "
            f"peak_reserved_gib={peak_reserved:.3f}."
        )
        if self.clear_cuda_cache_after_update:
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _actor_lr_adaptation_decision(
        planned_steps,
        actual_steps,
        kl_stop_reason_code,
        min_step_completion,
    ):
        """Return a deterministic LR-control decision for one PPO shard."""
        planned_steps = max(0.0, float(planned_steps))
        actual_steps = max(0.0, float(actual_steps))
        min_step_completion = float(min_step_completion)
        completion_rate = (
            actual_steps / planned_steps if planned_steps > 0.0 else 1.0
        )
        blocked = int(kl_stop_reason_code) != 0
        eligible = bool(
            planned_steps > 0.0
            and actual_steps > 0.0
            and completion_rate >= min_step_completion
            and not blocked
        )
        downshift = bool(
            planned_steps > 0.0
            and (completion_rate < min_step_completion or blocked)
        )
        return {
            'actor_step_completion_rate': float(completion_rate),
            'actor_lr_update_eligible': float(eligible),
            'actor_lr_incomplete_downshift_requested': float(downshift),
        }

    def _record_actor_update_health(self, train_infos):
        """Accumulate optimizer health over all non-warmup PPO shards."""
        planned = max(
            0.0, float(train_infos.get('actor_planned_optimizer_steps', 0.0))
        )
        actual = max(
            0.0, float(train_infos.get('actor_optimizer_steps', 0.0))
        )
        if planned <= 0.0:
            return
        self.actor_update_shards += 1
        self.actor_planned_optimizer_steps_total += planned
        self.actor_optimizer_steps_total += actual
        self.actor_zero_update_shards += int(actual <= 0.0)
        self.actor_incomplete_update_shards += int(actual < planned)
        self.actor_backtrack_failed_groups_total += max(
            0.0,
            float(train_infos.get('actor_backtrack_failed_groups', 0.0)),
        )
        self.actor_backtrack_groups_total += max(
            0.0,
            float(train_infos.get('actor_accumulation_groups', 0.0)),
        )
        self.actor_backtrack_retry_attempts_total += max(
            0.0,
            float(train_infos.get('actor_backtrack_retry_attempts', 0.0)),
        )
        replay_samples = max(
            0.0, float(train_infos.get('actor_data_sample_count', 0.0))
        )
        empty_replay_samples = max(
            0.0, float(train_infos.get('actor_empty_replay_samples', 0.0))
        )
        self.actor_replay_samples_total += replay_samples
        self.actor_empty_replay_samples_total += min(
            empty_replay_samples, replay_samples
        )
        stop_reason = int(
            train_infos.get('actor_kl_stop_reason_code', 0.0)
        )
        self.actor_kl_stop_shards += int(stop_reason != 0)
        self.actor_old_policy_kl_stop_shards += int(stop_reason in {1, 3})
        self.actor_bc_reference_kl_stop_shards += int(stop_reason in {2, 3})
        reference_exceeded = (
            float(train_infos.get('bc_reference_target_exceeded', 0.0)) > 0.0
            or float(train_infos.get(
                'post_update_bc_reference_target_kl_exceeded', 0.0
            )) > 0.0
        )
        self.actor_bc_reference_target_exceeded_shards += int(
            reference_exceeded
        )
        old_kl = float(
            train_infos.get('post_update_probe_approx_kl', 0.0)
        )
        reference_kl = float(
            train_infos.get('post_update_bc_reference_approx_kl', 0.0)
        )
        if np.isfinite(old_kl):
            self.actor_post_update_old_policy_kl_max = max(
                self.actor_post_update_old_policy_kl_max, old_kl
            )
        if np.isfinite(reference_kl):
            self.actor_post_update_bc_reference_kl_max = max(
                self.actor_post_update_bc_reference_kl_max, reference_kl
            )

    def _actor_update_health_snapshot(self):
        planned = float(self.actor_planned_optimizer_steps_total)
        actual = float(self.actor_optimizer_steps_total)
        return {
            'update_shards': int(self.actor_update_shards),
            'planned_optimizer_steps': planned,
            'actual_optimizer_steps': actual,
            'step_completion_rate': (
                actual / planned if planned > 0.0 else 1.0
            ),
            'zero_update_shards': int(self.actor_zero_update_shards),
            'incomplete_update_shards': int(
                self.actor_incomplete_update_shards
            ),
            'kl_stop_shards': int(self.actor_kl_stop_shards),
            'old_policy_kl_stop_shards': int(
                self.actor_old_policy_kl_stop_shards
            ),
            'bc_reference_kl_stop_shards': int(
                self.actor_bc_reference_kl_stop_shards
            ),
            'bc_reference_target_exceeded_shards': int(
                self.actor_bc_reference_target_exceeded_shards
            ),
            'post_update_old_policy_kl_max': float(
                self.actor_post_update_old_policy_kl_max
            ),
            'post_update_bc_reference_kl_max': float(
                self.actor_post_update_bc_reference_kl_max
            ),
            'backtrack_failed_groups': float(
                self.actor_backtrack_failed_groups_total
            ),
            'backtrack_groups': float(self.actor_backtrack_groups_total),
            'backtrack_retry_attempts': float(
                self.actor_backtrack_retry_attempts_total
            ),
            'backtrack_failed_group_fraction': (
                self.actor_backtrack_failed_groups_total
                / self.actor_backtrack_groups_total
                if self.actor_backtrack_groups_total > 0.0 else 0.0
            ),
            'replay_samples': float(self.actor_replay_samples_total),
            'empty_replay_samples': float(
                self.actor_empty_replay_samples_total
            ),
            'empty_replay_fraction': (
                self.actor_empty_replay_samples_total
                / self.actor_replay_samples_total
                if self.actor_replay_samples_total > 0.0 else 0.0
            ),
        }

    def _restore_actor_update_health(self, health):
        """Restore cumulative diagnostics only for an exact recovery resume."""
        health = dict(health or {})
        self.actor_update_shards = int(health.get('update_shards', 0))
        self.actor_planned_optimizer_steps_total = float(
            health.get('planned_optimizer_steps', 0.0)
        )
        self.actor_optimizer_steps_total = float(
            health.get('actual_optimizer_steps', 0.0)
        )
        self.actor_zero_update_shards = int(
            health.get('zero_update_shards', 0)
        )
        self.actor_incomplete_update_shards = int(
            health.get('incomplete_update_shards', 0)
        )
        self.actor_kl_stop_shards = int(health.get('kl_stop_shards', 0))
        self.actor_old_policy_kl_stop_shards = int(
            health.get('old_policy_kl_stop_shards', 0)
        )
        self.actor_bc_reference_kl_stop_shards = int(
            health.get('bc_reference_kl_stop_shards', 0)
        )
        self.actor_bc_reference_target_exceeded_shards = int(
            health.get('bc_reference_target_exceeded_shards', 0)
        )
        self.actor_replay_samples_total = float(
            health.get('replay_samples', 0.0)
        )
        self.actor_empty_replay_samples_total = float(
            health.get('empty_replay_samples', 0.0)
        )
        self.actor_post_update_old_policy_kl_max = float(
            health.get('post_update_old_policy_kl_max', 0.0)
        )
        self.actor_post_update_bc_reference_kl_max = float(
            health.get('post_update_bc_reference_kl_max', 0.0)
        )
        self.actor_backtrack_failed_groups_total = float(
            health.get('backtrack_failed_groups', 0.0)
        )
        self.actor_backtrack_groups_total = float(
            health.get('backtrack_groups', 0.0)
        )
        self.actor_backtrack_retry_attempts_total = float(
            health.get('backtrack_retry_attempts', 0.0)
        )

    @staticmethod
    def _resolve_bc_rollout_limits(min_rollouts, max_rollouts, fallback_rollouts):
        fallback_rollouts = max(1, int(fallback_rollouts))
        max_rollouts = int(max_rollouts)
        if max_rollouts <= 0:
            max_rollouts = fallback_rollouts
        min_rollouts = max(1, int(min_rollouts))
        if min_rollouts > max_rollouts:
            raise ValueError(
                "device_bc_min_rollouts_per_epoch cannot exceed "
                "device_bc_max_rollouts_per_epoch: "
                f"min={min_rollouts}, max={max_rollouts}"
            )
        return min_rollouts, max_rollouts

    @staticmethod
    def _evaluation_rank_mask(case_counts, round_idx):
        counts = np.asarray(case_counts, dtype=np.int64)
        if counts.ndim != 1 or np.any(counts <= 0):
            raise ValueError(f"Invalid evaluation case counts: {case_counts}")
        return round_idx < counts

    @staticmethod
    def _load_dataset_case_metadata(manifest_path):
        if not manifest_path:
            return {}
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Dataset manifest does not exist: {manifest_path}"
            )
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)

        metadata_by_path_key = {}
        for split_name, split_info in manifest.get('splits', {}).items():
            for case_info in split_info.get('cases', []):
                metadata = dict(case_info)
                manifest_case_id = str(metadata.get('case_id', ''))
                suffix = manifest_case_id.rsplit('_', 1)[-1]
                case_dir = f'case_{suffix}'
                metadata['split'] = str(split_name)
                metadata_by_path_key[f'{split_name}/{case_dir}'] = metadata
        return metadata_by_path_key

    def _case_metadata(self, case_path):
        normalized_path = os.path.abspath(str(case_path))
        metadata_cache = getattr(self, '_case_metadata_cache', None)
        if metadata_cache is None:
            metadata_cache = {}
            self._case_metadata_cache = metadata_cache
        if normalized_path in metadata_cache:
            return copy.deepcopy(metadata_cache[normalized_path])

        case_dir = os.path.basename(normalized_path)
        split_name = os.path.basename(os.path.dirname(normalized_path))
        path_key = f'{split_name}/{case_dir}'

        # Evaluation datasets are allowed to use a manifest schema that is
        # different from the legacy top-level ``splits`` schema.  The case
        # directory metadata is the common, immutable lineage contract across
        # both schemas, so load it before applying any manifest override.
        metadata = {}
        metadata_path = os.path.join(normalized_path, 'metadata.json')
        if os.path.isfile(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                case_metadata = json.load(metadata_file)
            if not isinstance(case_metadata, dict):
                raise ValueError(
                    'Case metadata must be a JSON object: '
                    f'{metadata_path}'
                )
            metadata.update(case_metadata)

        metadata.update(dict(
            getattr(self, 'dataset_case_metadata', {}).get(path_key, {})
        ))
        fingerprints = metadata.get('fingerprints', {})
        if isinstance(fingerprints, dict):
            case_sha256 = str(fingerprints.get('case_sha256', '')).strip()
            if case_sha256:
                metadata.setdefault('case_sha256', case_sha256)
        metadata.setdefault('case_id', f"{split_name}_{case_dir.rsplit('_', 1)[-1]}")
        metadata.setdefault('split', split_name)
        metadata['case_dir'] = case_dir
        metadata['case_path'] = normalized_path
        metadata['case_key'] = path_key
        metadata_cache[normalized_path] = copy.deepcopy(metadata)
        return copy.deepcopy(metadata)
    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {
                str(key): HKBZ_Runner._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [HKBZ_Runner._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return HKBZ_Runner._json_safe(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            value = float(value)
            return value if np.isfinite(value) else None
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    @staticmethod
    def _atomic_json_dump(payload, destination):
        temporary = f'{destination}.tmp.{os.getpid()}'
        with open(temporary, 'w', encoding='utf-8') as output_file:
            json.dump(
                HKBZ_Runner._json_safe(payload),
                output_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)

    def _evaluation_group_metrics(self):
        metrics = {}
        for field_name in ('profile', 'distribution'):
            groups = {}
            for record in self.last_eval_records:
                group_name = str(record.get(field_name, 'unknown') or 'unknown')
                groups.setdefault(group_name, []).append(record)
            for group_name, records in groups.items():
                safe_group = ''.join(
                    character if character.isalnum() else '_'
                    for character in group_name
                ).strip('_') or 'unknown'
                makespans = [
                    float(record['makespan'])
                    for record in records
                    if np.isfinite(record.get('makespan', np.inf))
                ]
                deltas = [
                    float(record['delta_vs_pre_ppo'])
                    for record in records
                    if record.get('delta_vs_pre_ppo') is not None
                    and np.isfinite(record['delta_vs_pre_ppo'])
                ]
                prefix = f'eval_{field_name}_{safe_group}'
                metrics[f'{prefix}_case_count'] = float(len(records))
                if makespans:
                    metrics[f'{prefix}_makespan'] = float(np.mean(makespans))
                if deltas:
                    metrics[f'{prefix}_delta_vs_pre_ppo'] = float(np.mean(deltas))
        return metrics

    def _selection_metrics_from_records(
        self,
        raw_makespan,
        evaluation_label=None,
    ):
        """Derive IID/composite/tail scores without letting test select."""
        raw_makespan = float(raw_makespan)
        self.last_eval_raw_makespan = raw_makespan
        if not np.isfinite(raw_makespan):
            self.last_eval_iid_makespan = np.inf
            self.last_eval_composite_makespan = np.inf
            self.last_eval_tail_makespan = np.inf
            self.last_eval_selection_score = np.inf
            return np.inf

        selection_labels = (
            str(evaluation_label or '') == 'pre_ppo'
            or str(evaluation_label or '').startswith('epoch_')
            or str(evaluation_label or '').startswith('canary_')
        )
        record_splits = {
            str(record.get('split', '')).strip().lower()
            for record in self.last_eval_records
        }
        if selection_labels and 'test' in record_splits:
            raise RuntimeError(
                "The test split is final-reporting only and cannot be used "
                "for checkpoint selection."
            )

        grouped = {}
        for distribution in ('iid', 'ood_stress', 'ood_scale'):
            values = [
                float(record['makespan'])
                for record in self.last_eval_records
                if str(record.get('distribution', '')).strip().lower()
                == distribution
                and np.isfinite(record.get('makespan', np.inf))
            ]
            grouped[distribution] = (
                float(np.mean(values)) if values else np.inf
            )

        # Legacy validation directories may not have a manifest.  They remain
        # valid strict-IID controls, but can never silently become composite.
        self.last_eval_iid_makespan = (
            grouped['iid']
            if np.isfinite(grouped['iid'])
            else raw_makespan
        )
        if all(np.isfinite(grouped[name]) for name in self.selection_weights):
            self.last_eval_composite_makespan = float(sum(
                self.selection_weights[name] * grouped[name]
                for name in self.selection_weights
            ))
        else:
            self.last_eval_composite_makespan = np.inf

        finite_makespans = sorted((
            float(record['makespan'])
            for record in self.last_eval_records
            if np.isfinite(record.get('makespan', np.inf))
        ), reverse=True)
        if finite_makespans:
            tail_count = max(1, int(math.ceil(
                float(getattr(self, 'selection_tail_fraction', 0.10))
                * len(finite_makespans)
            )))
            self.last_eval_tail_makespan = float(np.mean(
                finite_makespans[:tail_count]
            ))
        else:
            self.last_eval_tail_makespan = np.inf

        if self.selection_metric == 'raw':
            score = raw_makespan
        elif self.selection_metric in {'composite', 'composite_tail'}:
            if not np.isfinite(self.last_eval_composite_makespan):
                missing = [
                    name
                    for name, value in grouped.items()
                    if not np.isfinite(value)
                ]
                raise RuntimeError(
                    "Composite checkpoint selection requires IID, OOD-stress, "
                    f"and OOD-scale validation cases; missing={missing}."
                )
            if self.selection_metric == 'composite_tail':
                if not np.isfinite(self.last_eval_tail_makespan):
                    raise RuntimeError(
                        'composite_tail selection requires finite validation '
                        'case makespans.'
                    )
                score = (
                    (1.0 - float(getattr(
                        self, 'selection_tail_weight', 0.25
                    )))
                    * self.last_eval_composite_makespan
                    + float(getattr(
                        self, 'selection_tail_weight', 0.25
                    ))
                    * self.last_eval_tail_makespan
                )
            else:
                score = self.last_eval_composite_makespan
        else:
            score = self.last_eval_iid_makespan
        self.last_eval_selection_score = float(score)
        return self.last_eval_selection_score

    def _update_best_checkpoints(self, episode, stage):
        """Maintain IID, composite, and active-selection best snapshots."""
        improved = False
        common = {
            'selection_metric': self.selection_metric,
            'selection_score': float(self.last_eval_selection_score),
            'eval_raw_makespan': float(self.last_eval_raw_makespan),
            'eval_iid_makespan': float(self.last_eval_iid_makespan),
            'eval_composite_makespan': float(
                self.last_eval_composite_makespan
            ),
            'eval_tail_makespan': float(self.last_eval_tail_makespan),
            'evaluation_tau': float(self.evaluation_tau),
            'stage': str(stage),
        }
        if (
            np.isfinite(self.last_eval_iid_makespan)
            and self.last_eval_iid_makespan < self.best_eval_iid_makespan
        ):
            self.best_eval_iid_makespan = self.last_eval_iid_makespan
            self.save(
                episode,
                filename='checkpoint_Best_IID.pt',
                extra={**common, 'eval_makespan': self.last_eval_iid_makespan},
            )
        if (
            np.isfinite(self.last_eval_composite_makespan)
            and self.last_eval_composite_makespan
            < self.best_eval_composite_makespan
        ):
            self.best_eval_composite_makespan = (
                self.last_eval_composite_makespan
            )
            self.save(
                episode,
                filename='checkpoint_Best_Composite.pt',
                extra={
                    **common,
                    'eval_makespan': self.last_eval_composite_makespan,
                },
            )
        if (
            np.isfinite(self.last_eval_selection_score)
            and self.last_eval_selection_score < self.best_eval_makespan
        ):
            self.best_eval_makespan = self.last_eval_selection_score
            self.save(
                episode,
                filename='checkpoint_Best.pt',
                extra={
                    **common,
                    'eval_makespan': self.last_eval_selection_score,
                },
            )
            improved = True
        return improved

    def _write_evaluation_records(self, evaluation_label, eval_makespan):
        if not evaluation_label:
            return None
        safe_label = ''.join(
            character if character.isalnum() or character in ('-', '_') else '_'
            for character in str(evaluation_label)
        ).strip('_') or 'evaluation'
        destination = os.path.join(
            self.evaluation_dir,
            f'{safe_label}.json',
        )
        payload = {
            'evaluation_label': str(evaluation_label),
            'created_unix_time': time.time(),
            'policy_tau': float(self.policy.ac.tau),
            'dataset_manifest': self.dataset_manifest_path,
            'eval_dataset_dir': self.eval_dataset_dir,
            'summary': self._evaluation_log_info(eval_makespan),
            'cases': self.last_eval_records,
        }
        self._atomic_json_dump(payload, destination)
        print(
            f"[Eval] Saved {len(self.last_eval_records)} per-case records to "
            f"{destination}."
        )
        return destination

    def _evaluation_log_info(self, eval_makespan):
        eval_valid = float(
            np.isfinite(eval_makespan)
            and self.last_eval_completion_rate >= 1.0
            and self.last_eval_cycle_count == 0
            and self.last_eval_timeout_count == 0
        )
        info = {
            'eval_makespan': eval_makespan,
            'best_eval_makespan': min(self.best_eval_makespan, eval_makespan),
            'eval_valid': eval_valid,
            'eval_case_count': self.last_eval_case_count,
            'eval_completed_count': self.last_eval_completed_count,
            'eval_completion_rate': self.last_eval_completion_rate,
            'eval_timeout_count': self.last_eval_timeout_count,
            'eval_cycle_count': self.last_eval_cycle_count,
            'eval_mean_steps': self.last_eval_mean_steps,
            'eval_max_no_progress': self.last_eval_max_no_progress,
            'eval_mean_relocations': self.last_eval_mean_relocations,
            'policy_tau': float(self.policy.ac.tau),
            'eval_raw_makespan': self.last_eval_raw_makespan,
            'eval_iid_makespan': self.last_eval_iid_makespan,
            'eval_composite_makespan': self.last_eval_composite_makespan,
            'eval_tail_makespan': self.last_eval_tail_makespan,
            'eval_selection_score': self.last_eval_selection_score,
            'best_eval_iid_makespan': self.best_eval_iid_makespan,
            'best_eval_composite_makespan': (
                self.best_eval_composite_makespan
            ),
        }
        info.update(self._evaluation_group_metrics())
        return info

    def _protected_resource_joint_summary(self):
        """Return the immutable subset for the configured Stage-2 arm."""
        if not hasattr(self, 'policy') or self.policy is None:
            return None
        return protected_parameter_summary(
            self.policy.ac.state_dict(),
            prefixes=self._resource_joint_protected_prefixes(),
        )

    def _resource_joint_protected_prefixes(self):
        """Plane-only protection is explicit for shared-encoder adaptation."""
        return (
            ('plane_sel_enc.', 'actor.', 'plane_order_actor.')
            if self.stage2_allow_shared_unfreeze
            else PROTECTED_RESOURCE_JOINT_PREFIXES
        )

    def _stage1_handoff_summary(self):
        """Always validate the complete immutable Stage-1 hand-off."""
        if not hasattr(self, 'policy') or self.policy is None:
            return None
        return protected_parameter_summary(
            self.policy.ac.state_dict(),
            prefixes=PROTECTED_RESOURCE_JOINT_PREFIXES,
        )

    def _resource_actor_summary(self):
        """Return a bitwise digest for the Stage-2 trainable actor scope."""
        if not hasattr(self, 'policy') or self.policy is None:
            return None
        mode = self.policy.ac.device_policy_head_mode
        if mode == 'per_type':
            prefixes = [
                'device_type_sel_encs.',
                'device_type_actors.',
            ]
        else:
            # Preserve the historical ordinary-before-R014 scope order so
            # old audited DeviceBC summaries remain byte-for-byte comparable.
            prefixes = [
                'device_sel_enc.',
                'device_actor.',
            ]
            if mode == 'type_adapter':
                prefixes.append('device_type_adapter.')
        prefixes.extend((
            'transporter_sel_enc.',
            'transporter_actor.',
        ))
        if self.policy.ac.device_resource_adapter_enabled:
            prefixes.append('resource_residual_adapter.')
        return protected_parameter_summary(
            self.policy.ac.state_dict(),
            prefixes=tuple(prefixes),
        )

    def _request_ready_predictor_summary(self):
        """Digest the new request predictor independently from policy heads."""

        if (
            not hasattr(self, 'policy')
            or self.policy is None
            or not self.policy.ac.request_ready_prediction
        ):
            return None
        return protected_parameter_summary(
            self.policy.ac.state_dict(),
            prefixes=('request_ready_head.', 'request_ready_feature.'),
        )

    def _assert_frozen_ready_unchanged(self):
        if self.device_bc_training_scope != 'policy_frozen_ready':
            return
        observed = protected_parameter_summary(
            self.policy.ac.state_dict(), prefixes=('request_ready_head.',)
        )
        if not self.frozen_ready_head_summary or observed != self.frozen_ready_head_summary:
            raise RuntimeError('Frozen request-ready head changed during policy BC.')
        if self.stage2_frozen_state_before is not None and (
            self._stage2_frozen_state_summary() != self.stage2_frozen_state_before
        ):
            raise RuntimeError('Policy BC changed a tensor outside its registered optimizer scope.')

    def _stage2_frozen_state_summary(self):
        trainable_ids = {id(p) for module in self._device_bc_trainable_modules()
                         for p in module.parameters()}
        trainable_names = {name for name, param in self.policy.ac.named_parameters()
                           if id(param) in trainable_ids}
        frozen = {name: value for name, value in self.policy.ac.state_dict().items()
                  if name not in trainable_names}
        return protected_parameter_summary(frozen, prefixes=('',))

    def _stage2_bc_checkpoint_metadata(self):
        parameter_names = {id(param): name for name, param in self.policy.ac.named_parameters()}
        return {
            'stage2_matching_contract': self._stage2_matching_contract(),
            'stage2_bc_run_contract': self._stage2_bc_run_contract(),
            'stage2_policy_warmstart': getattr(self, 'stage2_policy_warmstart', None),
            'stage2_policy_warmstart_replay': getattr(self, 'stage2_policy_warmstart_replay', None),
            'device_bc_skip_ready_targets': bool(getattr(self.all_args, 'device_bc_skip_ready_targets', False)),
            'frozen_ready_source': self.frozen_ready_source,
            'frozen_ready_head_summary': self.frozen_ready_head_summary,
            'stage2_frozen_state_before': self.stage2_frozen_state_before,
            'request_ready_feature_summary': protected_parameter_summary(
                self.policy.ac.state_dict(), prefixes=('request_ready_feature.',)
            ) if self.policy.ac.request_ready_prediction else None,
            'stage2_bc_state': {
                'schema_version': 1,
                'at_epoch_boundary': self.stage2_bc_at_boundary,
                'completed_epochs': self.stage2_bc_completed_epochs,
                'pre_supervised_info': copy.deepcopy(self.stage2_pre_supervised_info),
                'epoch_records': copy.deepcopy(self.stage2_bc_epoch_records),
                'rng': {**capture_rng_state(), 'device_bc_dagger': copy.deepcopy(
                    self.device_bc_dagger_rng.bit_generator.state
                )},
                'eval_each_epoch': self.device_bc_eval_each_epoch,
                'sampling_seed': getattr(self.all_args, 'train_sampling_seed', None),
                'seed': self.all_args.seed,
                'rollout_threads': self.n_rollout_threads,
                'rollouts_per_epoch': self.device_bc_max_rollouts_per_epoch,
                'optimizer_parameter_names': [
                    parameter_names[id(param)] for module in self._device_bc_trainable_modules()
                    for param in module.parameters()
                ],
            },
        }

    def _stage2_bc_run_contract(self):
        names = (
            'seed', 'train_sampling_seed', 'train_sampling_mode',
            'train_sampling_weights', 'train_sampling_size', 'train_sampling_pool_size',
            'max_train_cases', 'n_rollout_threads', 'env_config', 'eval_dataset_dir',
            'max_eval_cases', 'evaluation_tau', 'device_bc_training_scope',
            'device_bc_eval_each_epoch', 'device_bc_lr', 'weight_decay', 'opti_eps',
            'stage2_bc_deterministic',
            'max_grad_norm', 'device_bc_categorical_loss_coef',
            'device_bc_assignment_loss_coef', 'device_bc_assignment_margin_loss_coef',
            'device_bc_assignment_margin', 'device_global_matching',
            'request_ready_checkpoint_sha256', 'request_ready_policy_injection',
            'request_ready_time_scale', 'device_bc_min_rollouts_per_epoch',
            'device_bc_max_rollouts_per_epoch', 'rollout_until_done',
        )
        contract = {**{key: getattr(self.all_args, key, None) for key in names},
                    **ready_head_config(self.policy.ac)}
        if getattr(self.all_args, 'stage2_policy_improvement_protocol', False):
            contract['cost_improvement'] = {name: getattr(self.all_args, name) for name in (
                'stage2_policy_improvement_protocol', 'stage2_cost_improvement',
                'stage2_cost_scale_seconds', 'stage2_cost_weight_clip', 'stage2_cost_tie_seconds',
                'stage2_cost_loss_coef', 'stage2_cost_snapshot_horizon',
                'stage2_cost_branch_timeout_seconds')}
        # Absent for legacy runs: do not invalidate their recovery contracts.
        if getattr(self.all_args, 'device_bc_matching_audit', False):
            contract['full_matching_supervision'] = self._stage2_matching_contract()
        if (getattr(self.all_args, 'stage2_policy_warmstart_checkpoint', '')
                or getattr(self.all_args, 'device_bc_skip_ready_targets', False)):
            contract.update({key: getattr(self.all_args, key, None) for key in (
                'stage2_policy_warmstart_checkpoint', 'stage2_policy_warmstart_sha256',
                'stage2_policy_warmstart_evaluation', 'stage2_policy_warmstart_evaluation_sha256',
                'device_bc_dagger_schedule', 'device_bc_plane_deterministic',
                'device_bc_skip_ready_targets',
            )})
        return contract

    def _stage2_matching_contract(self):
        return {
            'decoder': MATCHING_CONTRACT,
            'supervision': SUPERVISION_CONTRACT,
            'teacher_decoder': (TEACHER_PROJECTION_CONTRACT if getattr(
                self.all_args, 'device_bc_teacher_deployment_projection', False) else 'legacy_iga_serial'),
            'audit': bool(getattr(self.all_args, 'device_bc_matching_audit', False)),
            'full_edge_coef': float(getattr(self.all_args, 'device_bc_full_edge_loss_coef', 0.0)),
            'empty_wait_coef': float(getattr(self.all_args, 'device_bc_empty_wait_loss_coef', 0.0)),
            'minimum_wait_groups': int(getattr(self.all_args, 'device_bc_min_wait_groups_per_epoch', 0)),
            'edge_margin': float(getattr(self.all_args, 'device_bc_assignment_margin', 0.20)),
            'wait_causes': ['temporal_defer'],
            'independent_matching_probability': False,
        }

    def _evaluate_stage2_bc_epoch(self, epoch, optimizer):
        self.protected_parameter_summary_after_bc = self._assert_resource_joint_protected(
            self.protected_parameter_summary_before_bc, 'resource_bc_epoch'
        )
        self._assert_frozen_ready_unchanged()
        self.resource_actor_summary_after_bc = self._resource_actor_summary()
        self.request_ready_predictor_summary_after = self._request_ready_predictor_summary()
        training_flags = [(module, module.training) for module in self.policy.ac.modules()]
        try:
            score = float(self.eval(evaluation_label=f'bc_epoch_{epoch}'))
            metrics = self._evaluation_log_info(score)
        finally:
            for module, was_training in training_flags:
                module.training = was_training
        gate = supervised_gate(
            self.stage2_pre_supervised_info, metrics,
            max_raw_regression=self.stage2_max_raw_regression_seconds,
            max_stress_regression=self.stage2_max_stress_regression_seconds,
        )
        filename = f'checkpoint_BCEpoch{epoch}.pt'
        self.stage2_bc_epoch_records.append({
            'epoch': epoch, 'filename': filename, 'metrics': metrics, 'gate': gate,
        })
        self.stage2_bc_completed_epochs = epoch
        self.stage2_bc_at_boundary = True
        self.resource_supervised_optimizer_state = copy.deepcopy(optimizer.state_dict())
        self.save(episode=epoch - 1, filename=filename, extra={
            'stage': 'resource_supervised_epoch',
            'device_bc_epoch': epoch, 'selection_score': score,
            'eval_makespan': metrics['eval_raw_makespan'],
            'stage2_scientific_gate': gate,
        })
        self._report_progress('resource_bc_epoch_evaluated',
                              device_bc_epoch=epoch, **metrics)

    def _finish_stage2_epoch_selection(self, started_at):
        best = choose_best_epoch(self.stage2_bc_epoch_records)
        last = self.stage2_bc_epoch_records[-1]
        self.save(episode=self.stage2_bc_completed_epochs - 1,
                  filename='checkpoint_Last.pt', extra={
                      'stage': 'resource_supervised_final',
                      'device_bc_epoch': self.stage2_bc_completed_epochs,
                      'selection_score': last['metrics']['eval_raw_makespan'],
                      'stage2_scientific_gate': last['gate'],
                  })
        gate = best['gate'] if best is not None else last['gate']
        if best is not None:
            # Promote the actual selected epoch, including its optimizer/RNG,
            # not the final epoch under a misleading Best filename.
            selected = torch.load(os.path.join(self.save_dir, best['filename']), map_location='cpu')
            selected['stage'] = 'resource_supervised_final'
            selected['phase'] = 'resource_supervised_completed'
            selected['selected_bc_epoch'] = best['epoch']
            for filename in ('checkpoint_Best.pt', 'checkpoint_Stage2.pt'):
                self._atomic_torch_save(selected, os.path.join(self.save_dir, filename))
        self._report_progress(
            'resource_supervised_pipeline_completed',
            device_bc_epoch=self.stage2_bc_completed_epochs,
            resource_bc_total_labels=int(self.resource_bc_total_labels),
            resource_assignment_total_labels=int(self.resource_assignment_total_labels),
            request_ready_total_labels=int(self.request_ready_total_labels),
            selected_bc_epoch=best['epoch'] if best is not None else None,
            selected_evaluation=best['metrics'] if best is not None else None,
            stage2_scientific_gate=gate,
            elapsed_seconds=float(time.time() - started_at),
        )
        print(f'[Stage2] Epoch-selected pure BC completed; selected={best and best["epoch"]}.', flush=True)

    def _plane_actor_summary(self):
        """Return a bitwise digest for every plane-policy actor tensor."""
        if not hasattr(self, 'policy') or self.policy is None:
            return None
        return protected_parameter_summary(
            self.policy.ac.state_dict(),
            prefixes=('plane_sel_enc.', 'actor.', 'plane_order_actor.'),
        )

    def _shared_actor_summary(self):
        """Return a bitwise digest for the single shared graph encoder."""
        if not hasattr(self, 'policy') or self.policy is None:
            return None
        return protected_parameter_summary(
            self.policy.ac.state_dict(), prefixes=('encoder.',)
        )

    def _resource_lookahead_contract(self):
        return {
            'device_lookahead_dispatch': bool(getattr(
                self.all_args, 'device_lookahead_dispatch', False
            )),
            'device_lookahead_safety_margin': float(getattr(
                self.all_args, 'device_lookahead_safety_margin', 60.0
            )),
            'device_deadline_aware_dispatch': bool(getattr(
                self.all_args, 'device_deadline_aware_dispatch', False
            )),
            'device_future_intent_horizon': int(getattr(
                self.all_args, 'device_future_intent_horizon', 0
            )),
            'device_future_intent_mode': str(getattr(
                self.all_args, 'device_future_intent_mode', 'legacy_one'
            )),
            'device_frontier_max_requests': int(getattr(
                self.all_args, 'device_frontier_max_requests', 2
            )),
            'resource_release_aware_eta': bool(getattr(
                self.all_args, 'resource_release_aware_eta', False
            )),
            'device_lookahead_reservation_mode': str(getattr(
                self.all_args, 'device_lookahead_reservation_mode', 'none'
            )),
            'device_reservation_grace_seconds': float(getattr(
                self.all_args, 'device_reservation_grace_seconds', 300.0
            )),
            'device_departure_lookahead': bool(getattr(
                self.all_args, 'device_departure_lookahead', False
            )),
        }

    def _stage3_reward_contract(self):
        """Objective fields that must remain continuous across S2 -> S3."""

        return {
            'hindsight_reward_mode': str(self.hindsight_reward_mode),
            'resource_critical_lateness_coef': float(getattr(
                self.all_args, 'resource_critical_lateness_coef', 0.0
            )),
            'resource_earliness_coef': float(getattr(
                self.all_args, 'resource_earliness_coef', 0.0
            )),
            'resource_wait_constraint_target': float(
                self.resource_wait_constraint_target
            ),
            'resource_wait_dual_lr': float(self.resource_wait_dual_lr),
            'resource_wait_dual_max': float(self.resource_wait_dual_max),
        }

    def _critical_path_wave1_source_contracts(self, checkpoint):
        """Validate the narrow, preregistered Stage3 Wave-1 transition.

        The model/observation tensors remain bit compatible.  Only the
        dependency horizon/frontier or the policy-invariant online potential
        may differ from the Stage2 source.  This explicit audit prevents the
        research flag from becoming a general-purpose contract bypass.
        """
        source_planning = checkpoint.get('resource_lookahead_contract')
        experiment_config = checkpoint.get('experiment_config')
        if not isinstance(source_planning, Mapping):
            raise ValueError('Critical-path handoff has no source planning contract.')
        if not isinstance(experiment_config, Mapping):
            raise ValueError('Critical-path handoff has no source reward contract.')
        source_planning = dict(source_planning)
        if (
            int(source_planning.get('device_future_intent_horizon', -1)) != 1
            or int(source_planning.get('device_frontier_max_requests', -1)) != 2
            or source_planning.get('device_future_intent_mode')
            != 'bounded_frontier'
        ):
            raise ValueError(
                'Critical-path Wave-1 requires the selected horizon=1, '
                f'frontier=2 Stage2 source, got {source_planning!r}.'
            )
        runtime_planning = self._resource_lookahead_contract()
        mutable_planning = {
            'device_future_intent_horizon',
            'device_frontier_max_requests',
        }
        fixed_mismatches = {
            key: {
                'source': source_planning.get(key),
                'runtime': runtime_planning.get(key),
            }
            for key in set(source_planning).union(runtime_planning)
            if key not in mutable_planning
            and source_planning.get(key) != runtime_planning.get(key)
        }
        if fixed_mismatches:
            raise ValueError(
                'Critical-path Wave-1 changed a non-preregistered planning '
                f'field: {fixed_mismatches!r}.'
            )
        horizon_frontier = (
            int(runtime_planning['device_future_intent_horizon']),
            int(runtime_planning['device_frontier_max_requests']),
        )
        if horizon_frontier not in {(1, 2), (3, 4)}:
            raise ValueError(
                'Critical-path Wave-1 permits only horizon/frontier 1/2 or '
                f'3/4, got {horizon_frontier!r}.'
            )

        reward_keys = tuple(self._stage3_reward_contract())
        source_reward = {
            key: experiment_config.get(key) for key in reward_keys
        }
        if any(value is None for value in source_reward.values()):
            raise ValueError(
                f'Critical-path source reward fields are incomplete: '
                f'{source_reward!r}.'
            )
        runtime_reward = self._stage3_reward_contract()
        if self.hindsight_reward_mode == 'team_time':
            if runtime_reward != source_reward:
                raise ValueError(
                    'The Wave-1 control/planning arms must preserve the exact '
                    f'Stage2 reward contract: source={source_reward!r}, '
                    f'runtime={runtime_reward!r}.'
                )
            reset_wait_dual = False
        elif self.hindsight_reward_mode == 'team_time_resource_potential':
            expected_zero = {
                'resource_critical_lateness_coef': 0.0,
                'resource_earliness_coef': 0.0,
                'resource_wait_constraint_target': 0.0,
                'resource_wait_dual_lr': 0.0,
            }
            invalid = {
                key: runtime_reward[key]
                for key, expected in expected_zero.items()
                if not np.isclose(float(runtime_reward[key]), expected)
            }
            if invalid:
                raise ValueError(
                    'The critical-path potential arm must disable non-telescoping '
                    f'wait penalties: {invalid!r}.'
                )
            if (
                not np.isclose(float(self.all_args.resource_lateness_coef), 0.0)
                or not np.isclose(float(self.all_args.iga_potential_gamma), 1.0)
                or float(self.all_args.iga_potential_beta) <= 0.0
            ):
                raise ValueError(
                    'Critical-path potential requires zero total-wait penalty, '
                    'gamma=1 and positive beta.'
                )
            reset_wait_dual = True
        else:
            raise ValueError(
                'critical_path_wave1 permits only team_time or '
                'team_time_resource_potential.'
            )
        return {
            'source_planning_contract': source_planning,
            'runtime_planning_contract': runtime_planning,
            'source_reward_contract': source_reward,
            'runtime_reward_contract': runtime_reward,
            'reset_wait_dual': bool(reset_wait_dual),
        }

    def _ppo_gain_source_contracts(self, checkpoint):
        """Audit the narrow reward transition used by the PPO-gain screen.

        The checkpoint tensors and all non-frontier planning fields remain
        unchanged.  Runtime planning is fixed to horizon=3/frontier=4.  The
        reward must be either the exact Stage2 contract (P0 control) or pure
        terminal team Cmax with every wait/shaping coefficient disabled
        (P1-P3 and Wave 2).  No other contract drift is accepted.
        """

        source_planning = checkpoint.get('resource_lookahead_contract')
        experiment_config = checkpoint.get('experiment_config')
        if not isinstance(source_planning, Mapping):
            raise ValueError('PPO-gain handoff has no source planning contract.')
        if not isinstance(experiment_config, Mapping):
            raise ValueError('PPO-gain handoff has no source reward contract.')
        source_planning = dict(source_planning)
        if (
            int(source_planning.get('device_future_intent_horizon', -1)) != 1
            or int(source_planning.get('device_frontier_max_requests', -1)) != 2
            or source_planning.get('device_future_intent_mode')
            != 'bounded_frontier'
        ):
            raise ValueError(
                'PPO-gain requires the selected horizon=1/frontier=2 Stage2 '
                f'source, got {source_planning!r}.'
            )
        runtime_planning = self._resource_lookahead_contract()
        mutable_planning = {
            'device_future_intent_horizon',
            'device_frontier_max_requests',
        }
        fixed_mismatches = {
            key: {
                'source': source_planning.get(key),
                'runtime': runtime_planning.get(key),
            }
            for key in set(source_planning).union(runtime_planning)
            if key not in mutable_planning
            and source_planning.get(key) != runtime_planning.get(key)
        }
        if fixed_mismatches:
            raise ValueError(
                'PPO-gain changed a non-preregistered planning field: '
                f'{fixed_mismatches!r}.'
            )
        horizon_frontier = (
            int(runtime_planning['device_future_intent_horizon']),
            int(runtime_planning['device_frontier_max_requests']),
        )
        if horizon_frontier != (3, 4):
            raise ValueError(
                'PPO-gain requires runtime horizon/frontier 3/4, got '
                f'{horizon_frontier!r}.'
            )

        reward_keys = tuple(self._stage3_reward_contract())
        source_reward = {
            key: experiment_config.get(key) for key in reward_keys
        }
        if any(value is None for value in source_reward.values()):
            raise ValueError(
                f'PPO-gain source reward fields are incomplete: '
                f'{source_reward!r}.'
            )
        runtime_reward = self._stage3_reward_contract()
        pure_cmax_reward = {
            'hindsight_reward_mode': 'team_time',
            'resource_critical_lateness_coef': 0.0,
            'resource_earliness_coef': 0.0,
            'resource_wait_constraint_target': 0.0,
            'resource_wait_dual_lr': 0.0,
            'resource_wait_dual_max': 0.0,
        }
        if runtime_reward == source_reward:
            transition = 'exact_stage2_reward'
            reset_wait_dual = False
        elif runtime_reward == pure_cmax_reward:
            if (
                not np.isclose(float(self.all_args.resource_lateness_coef), 0.0)
                or not np.isclose(float(self.all_args.iga_potential_beta), 0.0)
                or not np.isclose(float(self.all_args.hindsight_shaping_coef), 0.0)
                or not np.isclose(float(self.all_args.hindsight_cmax_coef), 0.0)
                or not np.isclose(
                    float(self.all_args.hindsight_terminal_cmax_coef), 1.0
                )
            ):
                raise ValueError(
                    'PPO-gain pure-Cmax arm contains a hidden shaping or wait '
                    'coefficient.'
                )
            transition = 'pure_terminal_cmax'
            reset_wait_dual = True
        else:
            raise ValueError(
                'PPO-gain runtime reward must be exact Stage2 or fully pure '
                f'Cmax: source={source_reward!r}, runtime={runtime_reward!r}.'
            )
        return {
            'source_planning_contract': source_planning,
            'runtime_planning_contract': runtime_planning,
            'source_reward_contract': source_reward,
            'runtime_reward_contract': runtime_reward,
            'reward_transition': transition,
            'reset_wait_dual': bool(reset_wait_dual),
        }

    def _assert_resource_joint_protected(self, expected, phase):
        """Fail closed if any protected plane/shared tensor changed."""
        if self.training_stage != CANONICAL_RESOURCE_JOINT:
            return None
        observed = self._protected_resource_joint_summary()
        if expected is None or observed is None:
            raise RuntimeError(
                f"resource_joint protected-parameter evidence is missing at {phase}."
            )
        if observed != expected:
            raise RuntimeError(
                "resource_joint protected parameters changed during "
                f"{phase}; expected sha256={expected.get('sha256')} "
                f"observed sha256={observed.get('sha256')}."
            )
        return observed

    def _set_resource_joint_phase(self, phase, event=None, **extra):
        """Publish a canonical Stage-2 phase and its immutable evidence."""
        self.resource_joint_phase = str(phase)
        self.resource_bc_phase_events.append(self.resource_joint_phase)
        payload = {
            'phase': self.resource_joint_phase,
            'source_m2_checkpoint': dict(self.stage1_m2_source or {}),
            'source_m2_path': str((self.stage1_m2_source or {}).get('path', '')),
            'source_m2_sha256': str(
                (self.stage1_m2_source or {}).get('sha256', '')
            ),
            'stage1_m2_checkpoint_summary': self.stage1_m2_checkpoint_summary,
            'source_stage2_checkpoint': dict(self.stage2_source or {}),
            'source_stage2_path': str(
                (self.stage2_source or {}).get('path', '')
            ),
            'source_stage2_sha256': str(
                (self.stage2_source or {}).get('sha256', '')
            ),
            'stage2_checkpoint_summary': self.stage2_checkpoint_summary,
            'role_event_credit_mode': self.role_event_credit_mode,
            'role_event_credit_uniform_mix': float(
                self.role_event_credit_uniform_mix
            ),
            'counterfactual_q_baseline': bool(getattr(
                self.all_args, 'counterfactual_q_baseline', False
            )),
            'counterfactual_baseline_mix': float(
                self.current_counterfactual_baseline_mix
            ),
            'counterfactual_baseline_mix_schedule': list(
                self.counterfactual_baseline_mix_schedule
            ),
            'protected_parameter_summary': (
                self._protected_resource_joint_summary()
                if hasattr(self, 'policy') else None
            ),
            **extra,
        }
        self._report_progress(event or f'resource_joint_{phase}', **payload)

    def _set_joint_finetune_phase(self, phase, event=None, **extra):
        """Publish the canonical Stage3 phase and its source evidence."""
        self.resource_joint_phase = str(phase)
        self.resource_bc_phase_events.append(self.resource_joint_phase)
        self._report_progress(
            event or f'joint_finetune_{phase}',
            phase=self.resource_joint_phase,
            source_stage2_checkpoint=dict(self.stage2_source or {}),
            stage2_checkpoint_summary=self.stage2_checkpoint_summary,
            plane_actor_summary=self._plane_actor_summary(),
            resource_actor_summary=self._resource_actor_summary(),
            shared_actor_summary=self._shared_actor_summary(),
            **extra,
        )

    def _stage2_status_metadata(self):
        stage2_training_mode = (
            'ready_predictor_only'
            if self.device_bc_training_scope == 'ready_only'
            else 'supervised_only'
        )
        return {
            'training_stage': self.training_stage,
            'frozen_ready_source': self.frozen_ready_source,
            'frozen_ready_head_summary': self.frozen_ready_head_summary,
            'bc_epoch_evaluation_enabled': self.device_bc_eval_each_epoch,
            'phase': self.resource_joint_phase,
            'source_m2_checkpoint': dict(self.stage1_m2_source or {}),
            'source_m2_path': str((self.stage1_m2_source or {}).get('path', '')),
            'source_m2_sha256': str(
                (self.stage1_m2_source or {}).get('sha256', '')
            ),
            'stage1_m2_checkpoint_summary': self.stage1_m2_checkpoint_summary,
            'source_stage2_checkpoint': dict(self.stage2_source or {}),
            'source_stage2_path': str(
                (self.stage2_source or {}).get('path', '')
            ),
            'source_stage2_sha256': str(
                (self.stage2_source or {}).get('sha256', '')
            ),
            'stage2_checkpoint_summary': self.stage2_checkpoint_summary,
            'stage2_training_mode': (
                stage2_training_mode
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'device_bc_training_scope': self.device_bc_training_scope,
            'request_ready_seconds_loss_coef': float(
                self.request_ready_seconds_loss_coef
            ),
            'request_ready_seconds_loss_scale': float(
                self.request_ready_seconds_loss_scale
            ),
            'request_ready_kind_weights': {
                'h1': float(self.request_ready_h1_weight),
                'h2': float(self.request_ready_h2_weight),
                'departure': float(self.request_ready_departure_weight),
                'blocking': float(self.request_ready_blocking_weight),
            },
            'stage2_supervision_contract': (
                dict(STAGE2_SUPERVISION_CONTRACT)
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'request_ready_prediction': bool(
                self.policy.ac.request_ready_prediction
            ),
            'request_ready_time_scale': float(
                self.policy.ac.request_ready_time_scale
            ),
            'request_ready_policy_injection': str(
                self.policy.ac.request_ready_policy_injection
            ),
            'request_ready_hard_blocking': bool(
                self.policy.ac.request_ready_hard_blocking
            ),
            'request_ready_context_features': bool(
                self.policy.ac.request_ready_context_features
            ),
            'request_ready_head_mode': str(
                self.policy.ac.request_ready_head_mode
            ),
            'request_ready_quantile_head': bool(
                self.policy.ac.request_ready_quantile_head
            ),
            'request_ready_exclude_blocking_loss': bool(
                self.request_ready_exclude_blocking_loss
            ),
            'request_ready_kind_balanced_loss': bool(
                self.request_ready_kind_balanced_loss
            ),
            'request_ready_quantile_loss_coef': float(
                self.request_ready_quantile_loss_coef
            ),
            'device_resource_adapter': bool(
                self.policy.ac.device_resource_adapter_enabled
            ),
            'resource_bc_optimizer_reset': bool(
                self.resource_bc_optimizer_reset
            ),
            'resource_bc_total_labels': int(self.resource_bc_total_labels),
            'resource_dense_ranking_total_labels': int(
                self.resource_dense_ranking_total_labels
            ),
            'resource_assignment_total_labels': int(
                self.resource_assignment_total_labels
            ),
            'request_ready_total_labels': int(
                self.request_ready_total_labels
            ),
            'resource_actor_summary_before_bc': (
                self.resource_actor_summary_before_bc
            ),
            'resource_actor_summary_after_bc': (
                self.resource_actor_summary_after_bc
            ),
            'request_ready_predictor_summary_before': (
                self.request_ready_predictor_summary_before
            ),
            'request_ready_predictor_summary_after': (
                self.request_ready_predictor_summary_after
            ),
            'protected_parameter_summary_before_bc': (
                self.protected_parameter_summary_before_bc
            ),
            'protected_parameter_summary_after_bc': (
                self.protected_parameter_summary_after_bc
            ),
            'protected_parameter_summary': (
                self._protected_resource_joint_summary()
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'resource_actor_summary': (
                self._resource_actor_summary()
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
        }

    def _joint_finetune_status_metadata(self):
        return {
            'training_stage': self.training_stage,
            'phase': self.resource_joint_phase,
            'source_stage2_checkpoint': dict(self.stage2_source or {}),
            'stage2_checkpoint_summary': self.stage2_checkpoint_summary,
            'stage3_handoff_mode': self.stage3_handoff_mode,
            'stage3_contract_transition': self.stage3_contract_transition,
            'plane_actor_summary': self._plane_actor_summary(),
            'resource_actor_summary': self._resource_actor_summary(),
            'shared_actor_summary': self._shared_actor_summary(),
            'role_atomic_ppo': bool(getattr(
                self.all_args, 'role_atomic_ppo', False
            )),
            'role_event_returns': bool(getattr(
                self.all_args, 'role_event_returns', False
            )),
            'role_event_credit_mode': self.role_event_credit_mode,
            'role_event_credit_uniform_mix': float(
                self.role_event_credit_uniform_mix
            ),
            'counterfactual_q_baseline': bool(getattr(
                self.all_args, 'counterfactual_q_baseline', False
            )),
            'counterfactual_baseline_mix': float(
                self.current_counterfactual_baseline_mix
            ),
            'counterfactual_baseline_mix_schedule': list(
                self.counterfactual_baseline_mix_schedule
            ),
            'actor_grad_clip_mode': str(getattr(
                self.all_args, 'actor_grad_clip_mode', 'global'
            )),
            'role_event_gae_lambda': float(getattr(
                self.all_args, 'role_event_gae_lambda', 1.0
            )),
            'role_event_gae_lambdas': {
                role_name: float(value)
                for role_name, value in zip(
                    ('plane', 'device', 'transporter'),
                    getattr(
                        self.trainer,
                        'role_event_gae_lambdas',
                        {
                            0: float(getattr(
                                self.all_args,
                                'role_event_gae_lambda',
                                1.0,
                            )),
                            1: float(getattr(
                                self.all_args,
                                'role_event_gae_lambda',
                                1.0,
                            )),
                            2: float(getattr(
                                self.all_args,
                                'role_event_gae_lambda',
                                1.0,
                            )),
                        },
                    ).values(),
                )
            },
            'role_loss_weighting': str(getattr(
                self.all_args, 'role_loss_weighting', 'fixed'
            )),
            'shared_actor_lr_scale_schedule': list(
                self.shared_actor_lr_scale_schedule
            ),
            'shared_gradient_diagnostics': bool(getattr(
                self.all_args, 'shared_gradient_diagnostics', False
            )),
            'shared_encoder_pcgrad': bool(getattr(
                self.all_args, 'shared_encoder_pcgrad', False
            )),
            'shared_gradient_method': str(getattr(
                self.all_args, 'shared_gradient_method', 'sum'
            )),
            'shared_grad_ema_beta': float(getattr(
                self.all_args, 'shared_grad_ema_beta', 0.97
            )),
            'shared_grad_norm_power': float(getattr(
                self.all_args, 'shared_grad_norm_power', 0.5
            )),
            'shared_grad_min_scale': float(getattr(
                self.all_args, 'shared_grad_min_scale', 0.5
            )),
            'shared_grad_max_scale': float(getattr(
                self.all_args, 'shared_grad_max_scale', 2.0
            )),
            'shared_grad_conflict_threshold': float(getattr(
                self.all_args, 'shared_grad_conflict_threshold', -0.05
            )),
            'shared_cagrad_c': float(getattr(
                self.all_args, 'shared_cagrad_c', 0.2
            )),
            'shared_encoder_activation_checkpoint': bool(getattr(
                self.all_args, 'shared_encoder_activation_checkpoint', False
            )),
        }

    def _report_progress(self, event, **extra):
        callback = getattr(self, 'progress_callback', None)
        if callback is None:
            return

        # A few resume/selection paths construct a minimal runner via
        # ``__new__`` before the normal Stage2 attributes are initialized.
        # Keep those legacy callbacks useful while retaining the complete
        # resource_joint evidence on a fully initialized runner.
        training_stage = getattr(self, 'training_stage', 'auto') or 'auto'
        resource_joint_phase = (
            getattr(self, 'resource_joint_phase', 'not_applicable')
            or 'not_applicable'
        )
        stage1_m2_source = getattr(self, 'stage1_m2_source', None) or {}
        stage1_m2_checkpoint_summary = getattr(
            self, 'stage1_m2_checkpoint_summary', None
        )
        resource_actor_summary_before_ppo = getattr(
            self, 'resource_actor_summary_before_ppo', None
        )
        resource_actor_summary_after_ppo = getattr(
            self, 'resource_actor_summary_after_ppo', None
        )
        resource_actor_summary_before_bc = getattr(
            self, 'resource_actor_summary_before_bc', None
        )
        resource_actor_summary_after_bc = getattr(
            self, 'resource_actor_summary_after_bc', None
        )
        stage2_source = getattr(self, 'stage2_source', None) or {}
        stage2_checkpoint_summary = getattr(
            self, 'stage2_checkpoint_summary', None
        )
        protected_parameter_summary = None
        if (
            training_stage == CANONICAL_RESOURCE_JOINT
            and getattr(self, 'policy', None) is not None
        ):
            protected_parameter_summary = (
                self._protected_resource_joint_summary()
            )
        payload = {
            'event': str(event),
            'training_stage': training_stage,
            'phase': resource_joint_phase,
            'source_m2_checkpoint': dict(stage1_m2_source),
            'source_m2_path': str(stage1_m2_source.get('path', '')),
            'source_m2_sha256': str(
                stage1_m2_source.get('sha256', '')
            ),
            'stage1_m2_checkpoint_summary': stage1_m2_checkpoint_summary,
            'protected_parameter_summary': protected_parameter_summary,
            'resource_actor_summary_before_ppo': resource_actor_summary_before_ppo,
            'resource_actor_summary_after_ppo': resource_actor_summary_after_ppo,
            'resource_actor_summary_before_bc': resource_actor_summary_before_bc,
            'resource_actor_summary_after_bc': resource_actor_summary_after_bc,
            'source_stage2_checkpoint': dict(stage2_source),
            'stage2_checkpoint_summary': stage2_checkpoint_summary,
            'stage2_policy_warmstart': getattr(self, 'stage2_policy_warmstart', None),
            'stage2_policy_warmstart_replay': getattr(self, 'stage2_policy_warmstart_replay', None),
            'bc_reference_checkpoint': str(getattr(
                self, 'bc_reference_checkpoint', ''
            ) or ''),
            'bc_reference_resolved_path': str(getattr(
                self, 'bc_reference_resolved_path', ''
            ) or ''),
            'paired_case_baseline_path': str(getattr(
                self, 'paired_case_baseline_path', ''
            ) or ''),
            'paired_case_baseline_count': int(len(getattr(
                self, 'paired_case_baselines', {}
            ))),
            'paired_case_baseline_coef': float(getattr(
                self, 'paired_case_baseline_coef', 0.0
            )),
            'paired_case_baseline_scope': str(getattr(
                self, 'paired_case_baseline_scope', 'returns'
            )),
            'cvar_case_metric': str(getattr(
                self, 'cvar_case_metric', 'cmax'
            )),
            'epoch': int(getattr(self, 'current_epoch', -1)),
            'shard': int(getattr(self, 'current_shard', -1)),
            'total_num_steps': int(getattr(self, 'total_num_steps', 0)),
            **extra,
        }
        try:
            callback(**payload)
        except Exception as error:
            print(f"[Warning] Failed to publish training progress: {error}")

    def _seed_best_from_selection_checkpoint(self):
        """Preserve a prior global Best while continuing from a Last model."""
        if not self.selection_checkpoint_dir:
            return False
        checkpoint_path = str(self.selection_checkpoint_dir)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Selection checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        selection_makespan = float(checkpoint.get(
            'selection_score',
            checkpoint.get('eval_makespan', np.inf),
        ))
        if not np.isfinite(selection_makespan):
            raise ValueError(
                "Selection checkpoint must contain a finite eval_makespan: "
                f"{checkpoint_path}"
            )
        if selection_makespan >= self.best_eval_makespan:
            return False

        checkpoint = dict(checkpoint)
        checkpoint.update({
            'selection_source_checkpoint': checkpoint_path,
            'selection_resume_checkpoint': str(self.checkpoint_dir or ''),
            'selection_seeded_across_resume': True,
        })
        self._atomic_torch_save(
            checkpoint,
            os.path.join(self.save_dir, 'checkpoint_Best.pt'),
        )
        self.best_eval_makespan = selection_makespan
        restored_iid = float(checkpoint.get(
            'eval_iid_makespan', np.inf
        ))
        restored_composite = float(checkpoint.get(
            'eval_composite_makespan', np.inf
        ))
        if np.isfinite(restored_iid):
            self.best_eval_iid_makespan = min(
                self.best_eval_iid_makespan, restored_iid
            )
        if np.isfinite(restored_composite):
            self.best_eval_composite_makespan = min(
                self.best_eval_composite_makespan, restored_composite
            )
        self.log_train(
            {
                'best_eval_makespan': selection_makespan,
                'selection_baseline_makespan': selection_makespan,
            },
            self.total_num_steps,
        )
        print(
            f"[Info] Preserved cross-run global Best Cmax "
            f"{selection_makespan:.4f} from {checkpoint_path}."
        )
        self._report_progress(
            'selection_baseline_seeded',
            selection_baseline_makespan=selection_makespan,
            best_eval_makespan=selection_makespan,
            selection_checkpoint_dir=checkpoint_path,
        )
        return True

    def _run_shard_canary(self, episode, completed_shards):
        canary_makespan = float(self.eval(
            evaluation_label=f'canary_epoch_{episode + 1}_shard_{completed_shards}'
        ))
        best_makespan = float(
            self.canary_baseline_makespan
            if np.isfinite(self.canary_baseline_makespan)
            else self.best_eval_makespan
        )
        if np.isfinite(best_makespan) and best_makespan > 0.0:
            rejection_threshold = best_makespan * (1.0 + self.canary_max_regression)
            relative_regression = canary_makespan / best_makespan - 1.0
        else:
            rejection_threshold = np.inf
            relative_regression = 0.0

        rejected = bool(
            not np.isfinite(canary_makespan)
            or (
                np.isfinite(best_makespan)
                and canary_makespan > rejection_threshold
            )
        )
        canary_info = self._evaluation_log_info(canary_makespan)
        canary_info.update({
            'canary_eval_makespan': canary_makespan,
            'canary_best_makespan': best_makespan,
            'canary_relative_regression': relative_regression,
            'canary_rejection_threshold': rejection_threshold,
            'canary_rejected': float(rejected),
        })
        self.log_train(canary_info, self.total_num_steps)
        self._report_progress(
            'canary_evaluation_completed',
            epoch=int(episode),
            completed_shard=int(completed_shards),
            **{
                **canary_info,
                'canary_rejected': bool(rejected),
            },
        )
        print(
            f"[Canary] epoch={episode + 1}, completed_shards={completed_shards}, "
            f"Cmax={canary_makespan:.4f}, Best={best_makespan:.4f}, "
            f"relative_regression={relative_regression:.4%}, rejected={rejected}."
        )
        if not rejected or not self.canary_stop_on_regression:
            return False

        self.canary_rejected = True
        self.canary_rejection_info = {
            'canary_epoch': int(episode),
            'canary_completed_shard': int(completed_shards),
            'canary_eval_makespan': canary_makespan,
            'canary_best_makespan': best_makespan,
            'canary_relative_regression': float(relative_regression),
            'canary_rejection_threshold': float(rejection_threshold),
        }
        self.save(
            episode,
            filename='checkpoint_CanaryRejected.pt',
            extra={'stage': 'canary_rejected', **self.canary_rejection_info},
        )
        print(
            "[Warning] Shard canary rejected the updated policy; "
            "stopping cleanly while preserving checkpoint_Best.pt."
        )
        self._report_progress(
            'canary_regression_stop_requested',
            **self.canary_rejection_info,
        )
        return True

    def _ensure_bc_reference_policy(self):
        enabled = (
            self.bc_reference_kl_coef > 0.0
            or self.bc_reference_target_kl > 0.0
            or any(value > 0.0 for value in self.bc_reference_kl_coef_schedule)
        )
        if not enabled or self.policy.has_bc_reference():
            return
        explicit_reference = str(
            getattr(self, 'bc_reference_checkpoint', '') or ''
        ).strip()
        resource_reference = (
            self.resource_bc_checkpoint
            if self.training_stage == CANONICAL_RESOURCE_JOINT
            and self.resource_bc_checkpoint
            else None
        )
        if not explicit_reference and not resource_reference and not self.checkpoint_dir:
            raise RuntimeError(
                'BC-reference PPO is enabled but no BC snapshot is available.'
            )
        checkpoint_path = str(
            explicit_reference or resource_reference or self.checkpoint_dir
        )
        if explicit_reference:
            # Stage3 source-relative trust regions must refer to the exact
            # hand-off checkpoint, not the historical PlaneBC sibling.
            candidates = [str(Path(explicit_reference).expanduser().resolve())]
            reference_name = 'checkpoint_ExplicitPolicyReference.pt'
        elif resource_reference:
            # Resource PPO must stay close to the selected post-DeviceBC
            # policy, not the much earlier Stage1 PlaneBC sibling.
            candidates = [checkpoint_path]
            reference_name = 'checkpoint_DeviceBCReference.pt'
        else:
            candidates = []
            if os.path.basename(checkpoint_path) == 'checkpoint_PlaneBC.pt':
                candidates.append(checkpoint_path)
            candidates.append(os.path.join(
                os.path.dirname(checkpoint_path), 'checkpoint_PlaneBC.pt'
            ))
            reference_name = 'checkpoint_PlaneBC.pt'
        reference_path = next(
            (candidate for candidate in candidates if os.path.isfile(candidate)),
            None,
        )
        if reference_path is None:
            raise RuntimeError(
                'BC-reference PPO could not resolve its immutable reference; '
                f'checked {candidates}.'
            )
        payload = torch.load(reference_path, map_location='cpu')
        self.policy.capture_bc_reference(payload['model'])
        self.bc_reference_resolved_path = str(
            Path(reference_path).expanduser().resolve()
        )
        local_reference_path = os.path.join(
            self.save_dir, reference_name
        )
        if (
            os.path.abspath(reference_path)
            != os.path.abspath(local_reference_path)
            and not os.path.exists(local_reference_path)
        ):
            shutil.copy2(reference_path, local_reference_path)
            print(
                '[Info] Copied the BC reference into the current run for '
                f'resumable PPO: {local_reference_path}.'
            )
        print(
            '[Info] Restored frozen BC reference from '
            f'{self.bc_reference_resolved_path}.'
        )

    @staticmethod
    def _parse_epoch_schedule(raw, name):
        values = tuple(
            float(item.strip())
            for item in str(raw or '').split(',')
            if item.strip()
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f'--{name} values must be finite and non-negative.')
        return values

    @staticmethod
    def _epoch_schedule_value(values, epoch, default):
        if not values:
            return float(default)
        return float(values[min(int(epoch), len(values) - 1)])

    def _apply_epoch_method_schedules(self, episode):
        potential_beta = self._epoch_schedule_value(
            self.iga_potential_beta_schedule,
            episode,
            getattr(self.all_args, 'iga_potential_beta', 0.0),
        )
        if self.adaptive_bc_reference_kl:
            # The trainer updates this coefficient after every rollout shard.
            # Do not reset it to the command-line seed at epoch boundaries.
            bc_kl_coef = float(self.trainer.bc_reference_kl_coef)
        else:
            bc_kl_coef = self._epoch_schedule_value(
                self.bc_reference_kl_coef_schedule,
                episode,
                getattr(self.all_args, 'bc_reference_kl_coef', 0.0),
            )
        shared_actor_lr_scale = self._epoch_schedule_value(
            self.shared_actor_lr_scale_schedule,
            episode,
            getattr(self.all_args, 'shared_actor_lr_scale', 1.0),
        )
        shared_lr_metrics = self.policy.set_shared_actor_lr_scale(
            shared_actor_lr_scale
        )
        counterfactual_mix = self._epoch_schedule_value(
            self.counterfactual_baseline_mix_schedule,
            episode,
            getattr(self.all_args, 'counterfactual_baseline_mix', 1.0),
        )
        if not 0.0 <= counterfactual_mix <= 1.0:
            raise RuntimeError(
                'Scheduled counterfactual baseline mix left [0, 1].'
            )
        self.current_counterfactual_baseline_mix = float(
            self.policy.set_counterfactual_baseline_mix(
                counterfactual_mix
            )
        )
        observed = self.envs.call('set_iga_potential_beta', potential_beta)
        if any(not np.isclose(float(value), potential_beta) for value in observed):
            raise RuntimeError('Training workers rejected potential-beta schedule.')
        self.current_iga_potential_beta = float(potential_beta)
        wait_coefficients = self.envs.call(
            'set_resource_lateness_coef', self.resource_wait_dual_value
        )
        if any(
            not np.isclose(float(value), self.resource_wait_dual_value)
            for value in wait_coefficients
        ):
            raise RuntimeError(
                'Training workers rejected resource-wait dual coefficient.'
            )
        self.all_args.resource_lateness_coef = float(
            self.resource_wait_dual_value
        )
        self.bc_reference_kl_coef = bc_kl_coef
        self.trainer.bc_reference_kl_coef = bc_kl_coef
        self._report_progress(
            'epoch_method_schedule_applied',
            iga_potential_beta=float(potential_beta),
            bc_reference_kl_coef=float(bc_kl_coef),
            resource_wait_dual_value=float(self.resource_wait_dual_value),
            counterfactual_baseline_mix=float(
                self.current_counterfactual_baseline_mix
            ),
            **shared_lr_metrics,
        )
        print(
            f'[MethodSchedule] epoch={episode + 1} '
            f'iga_potential_beta={potential_beta:.6g} '
            f'bc_reference_kl_coef={bc_kl_coef:.6g} '
            f'resource_wait_dual={self.resource_wait_dual_value:.6g} '
            f'counterfactual_mix='
            f'{self.current_counterfactual_baseline_mix:.6g} '
            f'shared_lr_scale={shared_actor_lr_scale:.6g} '
            f"shared_lr={shared_lr_metrics['shared_actor_lr']:.6g}.",
            flush=True,
        )

    def _update_resource_wait_dual(self, observed_wait, episode):
        if not self.resource_wait_dual_enabled:
            return self.resource_wait_dual_value
        observed_wait = float(observed_wait)
        if not np.isfinite(observed_wait):
            raise RuntimeError('Observed resource wait for dual update is non-finite.')
        violation = (
            observed_wait - self.resource_wait_constraint_target
        ) / self.resource_wait_constraint_target
        previous = float(self.resource_wait_dual_value)
        self.resource_wait_dual_value = float(np.clip(
            previous + self.resource_wait_dual_lr * violation,
            0.0,
            self.resource_wait_dual_max,
        ))
        self._report_progress(
            'resource_wait_dual_updated',
            resource_wait_observed=float(observed_wait),
            resource_wait_target=float(self.resource_wait_constraint_target),
            resource_wait_normalized_violation=float(violation),
            resource_wait_dual_previous=previous,
            resource_wait_dual_value=float(self.resource_wait_dual_value),
            epoch=int(episode + 1),
        )
        print(
            f'[ResourceWaitDual] epoch={episode + 1} '
            f'observed={observed_wait:.3f} '
            f'target={self.resource_wait_constraint_target:.3f} '
            f'lambda={previous:.6g}->{self.resource_wait_dual_value:.6g}.',
            flush=True,
        )
        return self.resource_wait_dual_value

    def run(self):   
        from onpolicy.utils.stage2_freeze_guard import guard_training_stage
        guard_training_stage(self.training_stage)
        
        start = time.time()
        episodes = self.num_episodes
        self.total_num_steps = int(self.resume_total_num_steps)
        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            # restore() runs during construction, before the launcher can
            # attach its run_status callback.  Publish the completed hand-off
            # as the first runtime event and carry its immutable digest into
            # every later status/checkpoint.
            if not (
                self.exact_resume_stage2
                or self.exact_resume_stage2_supervised
            ):
                self.resource_joint_phase = 'stage1_m2_restored'
            if self.protected_parameter_summary_before_bc is None:
                self.protected_parameter_summary_before_bc = (
                    self._protected_resource_joint_summary()
                )
        elif self.training_stage == CANONICAL_JOINT_FINETUNE:
            self.resource_joint_phase = 'stage2_handoff_restored'
        exact_resume = self.exact_resume_stage1 or self.exact_resume_stage2
        if self.exact_resume_stage2_supervised:
            self._report_progress(
                'supervised_training_resumed',
                total_epochs=int(self.device_bc_pretrain_epochs),
                device_bc_epoch=int(self.device_bc_resume_epoch + 1),
                device_bc_rollout=int(
                    self.device_bc_resume_completed_rollouts
                ),
                device_bc_total_rollouts=int(
                    self.device_bc_max_rollouts_per_epoch
                ),
                exact_resume_stage2_supervised=True,
                **self._stage2_status_metadata(),
            )
        elif exact_resume:
            self._copy_resume_artifacts()
            self._report_progress(
                'training_resumed',
                total_epochs=int(episodes),
                resume_epoch=int(self.resume_epoch),
                resume_completed_shards=int(self.resume_completed_shards),
                total_num_steps=int(self.total_num_steps),
                exact_resume_stage2=bool(self.exact_resume_stage2),
            )
        else:
            self._report_progress(
                'training_started',
                total_epochs=int(episodes),
                **(
                    self._stage2_status_metadata()
                    if self.training_stage == CANONICAL_RESOURCE_JOINT
                    else self._joint_finetune_status_metadata()
                    if self.training_stage == CANONICAL_JOINT_FINETUNE
                    else {}
                ),
            )

        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            # Canonical Stage2 is a closed supervised transition.  Establish
            # the unmodified S1 boundary, train the ready-time/resource heads,
            # validate once more, and return without constructing a PPO
            # rollout buffer update or touching critic/ValueNorm state.
            pre_supervised_info = self.stage2_pre_supervised_info
            if self.use_eval and not self.exact_resume_stage2_supervised:
                pre_supervised = float(self.eval(
                    evaluation_label='pre_supervised'
                ))
                pre_supervised_info = self._evaluation_log_info(
                    pre_supervised
                )
                self.stage2_pre_supervised_info = copy.deepcopy(pre_supervised_info)
                self._check_stage2_policy_warmstart_replay()
                self.log_train(
                    pre_supervised_info,
                    self.total_num_steps,
                )
                self.save(
                    episode=-1,
                    filename='checkpoint_PreSupervised.pt',
                    extra={
                        'stage': 'pre_supervised_baseline',
                        'eval_makespan': float(
                            pre_supervised_info['eval_raw_makespan']
                        ),
                        'selection_score': pre_supervised,
                        'evaluation_tau': self.evaluation_tau,
                    },
                )
                self._report_progress(
                    'pre_supervised_evaluated',
                    **self._evaluation_log_info(pre_supervised),
                )
            elif self.use_eval:
                print(
                    '[Info] Supervised Stage2 recovery reuses the previously '
                    'completed pre-supervised validation and resumes directly '
                    'at the DeviceBC cursor.',
                    flush=True,
                )
            self._set_resource_joint_phase(
                'resource_supervised',
                event='resource_supervised_started',
            )
            self.device_bc_pretrain()
            if self.device_bc_eval_each_epoch:
                self._finish_stage2_epoch_selection(start)
                return
            post_supervised = float('nan')
            post_supervised_info = None
            if self.use_eval:
                post_supervised = float(self.eval(
                    evaluation_label='post_supervised'
                ))
                post_supervised_info = self._evaluation_log_info(
                    post_supervised
                )
                self.log_train(
                    post_supervised_info,
                    self.total_num_steps,
                )
            self._set_resource_joint_phase(
                'resource_supervised_completed',
                event='resource_supervised_validation_completed',
                **(
                    post_supervised_info
                    if self.use_eval else {}
                ),
            )
            gate = {
                'passed': True,
                'reasons': [],
                'max_raw_regression_seconds': float(
                    self.stage2_max_raw_regression_seconds
                ),
                'max_stress_regression_seconds': float(
                    self.stage2_max_stress_regression_seconds
                ),
            }
            if self.use_eval:
                if pre_supervised_info is None:
                    raise RuntimeError(
                        'Stage2 scientific gate cannot resume without the '
                        'pre-supervised validation metrics.'
                    )
                pre_raw = float(pre_supervised_info['eval_raw_makespan'])
                post_raw = float(post_supervised_info['eval_raw_makespan'])
                raw_delta = post_raw - pre_raw
                pre_stress = float(pre_supervised_info.get(
                    'eval_distribution_ood_stress_makespan', float('nan')
                ))
                post_stress = float(post_supervised_info.get(
                    'eval_distribution_ood_stress_makespan', float('nan')
                ))
                stress_delta = post_stress - pre_stress
                gate.update({
                    'pre_raw_makespan': pre_raw,
                    'post_raw_makespan': post_raw,
                    'raw_delta_seconds': raw_delta,
                    'pre_ood_stress_makespan': pre_stress,
                    'post_ood_stress_makespan': post_stress,
                    'ood_stress_delta_seconds': stress_delta,
                    'completion_rate': float(
                        post_supervised_info['eval_completion_rate']
                    ),
                    'cycle_count': int(
                        post_supervised_info['eval_cycle_count']
                    ),
                    'timeout_count': int(
                        post_supervised_info['eval_timeout_count']
                    ),
                })
                if float(post_supervised_info['eval_valid']) != 1.0:
                    gate['reasons'].append('invalid_or_incomplete_validation')
                if raw_delta > self.stage2_max_raw_regression_seconds:
                    gate['reasons'].append('raw_validation_regression')
                if np.isfinite(self.stage2_max_stress_regression_seconds):
                    if not (np.isfinite(pre_stress) and np.isfinite(post_stress)):
                        gate['reasons'].append('missing_ood_stress_metric')
                    elif stress_delta > self.stage2_max_stress_regression_seconds:
                        gate['reasons'].append('ood_stress_regression')
                gate['passed'] = not gate['reasons']
            predictor_only = self.device_bc_training_scope == 'ready_only'
            final_extra = {
                'stage': (
                    'request_ready_predictor_final'
                    if predictor_only else 'resource_supervised_final'
                ),
                'stage2_training_mode': (
                    'ready_predictor_only'
                    if predictor_only else 'supervised_only'
                ),
                'device_bc_training_scope': self.device_bc_training_scope,
                'eval_makespan': (
                    float(post_supervised_info['eval_raw_makespan'])
                    if post_supervised_info is not None else post_supervised
                ),
                'selection_score': post_supervised,
                'evaluation_tau': self.evaluation_tau,
                'stage2_scientific_gate': gate,
            }
            self.save(
                episode=-1,
                filename='checkpoint_Last.pt',
                extra=final_extra,
            )
            if gate['passed']:
                final_filenames = (
                    ('checkpoint_RequestReady.pt', 'checkpoint_BestPredictor.pt')
                    if predictor_only
                    else ('checkpoint_Stage2.pt', 'checkpoint_Best.pt')
                )
                for filename in final_filenames:
                    self.save(episode=-1, filename=filename, extra=final_extra)
            else:
                self.save(
                    episode=-1,
                    filename='checkpoint_ScientificRejected.pt',
                    extra={**final_extra, 'stage': 'scientific_rejected'},
                )
            self._report_progress(
                'resource_supervised_pipeline_completed',
                device_bc_epoch=int(self.device_bc_pretrain_epochs),
                resource_bc_total_labels=int(self.resource_bc_total_labels),
                resource_dense_ranking_total_labels=int(
                    self.resource_dense_ranking_total_labels
                ),
                resource_assignment_total_labels=int(
                    self.resource_assignment_total_labels
                ),
                request_ready_total_labels=int(
                    self.request_ready_total_labels
                ),
                stage2_scientific_gate=gate,
                elapsed_seconds=float(time.time() - start),
            )
            print(
                '[Stage2] Supervised '
                f'{"request-ready predictor only" if predictor_only else "ready-time and resource-policy"} training '
                'completed; no PPO update was executed; scientific_gate='
                f'{"passed" if gate["passed"] else "rejected"}.',
                flush=True,
            )
            return

        if self.plane_bc_pretrain_epochs > 0 and not self.exact_resume_stage2:
            self.plane_bc_pretrain()

        if self.device_bc_pretrain_epochs > 0 and not self.exact_resume_stage2:
            if self.training_stage == CANONICAL_JOINT_FINETUNE:
                self._set_joint_finetune_phase(
                    'joint_resource_bc',
                    event='joint_resource_bc_started',
                )
            self.device_bc_pretrain()

        self._ensure_bc_reference_policy()

        if self.training_stage == CANONICAL_JOINT_FINETUNE:
            self._set_joint_finetune_phase(
                'joint_finetune_ppo',
                event='joint_finetune_ppo_started',
            )
            self.plane_actor_summary_before_ppo = (
                self._plane_actor_summary()
            )
            self.resource_actor_summary_before_ppo = (
                self._resource_actor_summary()
            )
            self.shared_actor_summary_before_ppo = (
                self._shared_actor_summary()
            )

        # Establish a pre-PPO baseline so checkpoint selection cannot silently
        # discard a stronger loaded/BC-warmed policy after the first update.
        if (
            self.use_eval
            and not exact_resume
            and not self.skip_pre_ppo_eval
        ):
            baseline_makespan = float(self.eval(evaluation_label='pre_ppo'))
            self.log_train(
                self._evaluation_log_info(baseline_makespan),
                self.total_num_steps,
            )
            if np.isfinite(baseline_makespan):
                self.save(
                    episode=-1,
                    filename='checkpoint_PrePPO_tau03.pt',
                    extra={
                        'eval_makespan': baseline_makespan,
                        'selection_score': baseline_makespan,
                        'eval_iid_makespan': (
                            self.last_eval_iid_makespan
                        ),
                        'eval_composite_makespan': (
                            self.last_eval_composite_makespan
                        ),
                        'evaluation_tau': self.evaluation_tau,
                        'stage': 'pre_ppo_baseline',
                    },
                )
                # Keep the historical filename for downstream evaluation
                # scripts while making the fixed-temperature artifact explicit.
                self.save(
                    episode=-1,
                    filename='checkpoint_PrePPO.pt',
                    extra={
                        'eval_makespan': baseline_makespan,
                        'selection_score': baseline_makespan,
                        'evaluation_tau': self.evaluation_tau,
                        'stage': 'pre_ppo_baseline',
                    },
                )
                self._update_best_checkpoints(
                    episode=-1,
                    stage='pre_ppo_baseline',
                )
                print(
                    f"[Info] Pre-PPO deterministic evaluation selection score "
                    f"{baseline_makespan:.4f} at tau={self.evaluation_tau:.3f}; "
                    "saved as the initial Best candidate."
                )
                self._report_progress(
                    'baseline_evaluated',
                    **self._evaluation_log_info(baseline_makespan),
                )
                self._seed_best_from_selection_checkpoint()
                if (
                    self.canary_eval_interval_shards > 0
                    and self.canary_eval_max_cases > 0
                ):
                    self.canary_baseline_makespan = float(self.eval(
                        evaluation_label='canary_pre_ppo'
                    ))
                    if not np.isfinite(self.canary_baseline_makespan):
                        raise RuntimeError(
                            'Shard-canary baseline evaluation is non-finite.'
                        )
                    self._report_progress(
                        'canary_baseline_evaluated',
                        canary_baseline_makespan=float(
                            self.canary_baseline_makespan
                        ),
                        canary_eval_max_cases=int(
                            self.canary_eval_max_cases
                        ),
                    )
                    print(
                        '[Canary] Established matched subset baseline '
                        f'Cmax={self.canary_baseline_makespan:.4f} '
                        f'cases={self.canary_eval_max_cases}.',
                        flush=True,
                    )

        if self.plane_bc_only or self.device_bc_only:
            bc_kind = 'DeviceBC' if self.device_bc_only else 'PlaneBC'
            self._report_progress(
                'device_bc_only_completed'
                if self.device_bc_only else 'plane_bc_only_completed',
                **(
                    {'device_bc_epoch': int(self.device_bc_pretrain_epochs)}
                    if self.device_bc_only
                    else {'plane_bc_epoch': int(self.plane_bc_pretrain_epochs)}
                ),
                best_eval_makespan=float(self.best_eval_makespan),
            )
            print(
                f'[{bc_kind}] BC-only screening completed after deterministic '
                'Pre-PPO validation; no PPO update was executed.',
                flush=True,
            )
            return

        first_episode = int(self.resume_epoch) if exact_resume else 0
        pbar = tqdm(range(first_episode, episodes),
              desc="Training",    
              unit="episode",     
              total=episodes,       
              initial=first_episode,
              ncols=160)
        for episode in pbar:
            # profiler = cProfile.Profile()
            # profiler.enable()
            if (
                self.actor_warmup_shards > 0
                and episode * self.num_envs == self.actor_warmup_shards
                and int(getattr(
                    self.all_args, 'train_sampling_pool_size', 0
                )) > int(getattr(
                    self.all_args, 'train_sampling_size', 0
                )) > 0
            ):
                self.envs.call('reset_training_case_cycle')
                self._report_progress(
                    'actor_training_case_cycle_reset',
                    actor_warmup_shards=int(self.actor_warmup_shards),
                    train_sampling_pool_size=int(getattr(
                        self.all_args, 'train_sampling_pool_size', 0
                    )),
                    train_sampling_size=int(getattr(
                        self.all_args, 'train_sampling_size', 0
                    )),
                )
            self.envs.shuffer_data()
            self.episode = episode
            self.current_epoch = episode
            self.current_shard = -1
            self._apply_epoch_method_schedules(episode)
            epoch_resource_wait_values = []
            first_shard = (
                int(self.resume_completed_shards)
                if exact_resume and episode == first_episode
                else 0
            )
            self._report_progress(
                'epoch_started',
                total_epochs=int(episodes),
                resume_from_shard=int(first_shard),
            )

            freeze_shared = episode < self.gnn_freeze_epochs
            freeze_plane = episode < self.plane_freeze_epochs
            freeze_order = episode < self.plane_order_freeze_epochs
            if self.training_stage == 'plane_pretrain':
                self.policy.set_plane_pretraining_stage(
                    freeze_shared=freeze_shared,
                    freeze_order=freeze_order,
                )
            elif self.training_stage == CANONICAL_RESOURCE_JOINT:
                train_device = True
                train_transporter = True
                if episode < self.resource_ppo_warmup_epochs:
                    if self.resource_ppo_update_schedule == 'ordinary_then_joint':
                        train_transporter = False
                    elif self.resource_ppo_update_schedule == 'r014_then_joint':
                        train_device = False
                self.policy.set_resource_joint_training_stage(
                    freeze_plane=freeze_plane,
                    freeze_shared=freeze_shared,
                    train_device=train_device,
                    train_transporter=train_transporter,
                )
                self._report_progress(
                    'resource_ppo_trainability_configured',
                    resource_ppo_update_schedule=(
                        self.resource_ppo_update_schedule
                    ),
                    resource_ppo_warmup_epochs=int(
                        self.resource_ppo_warmup_epochs
                    ),
                    train_device_actor=bool(train_device),
                    train_transporter_actor=bool(train_transporter),
                )
            elif self.training_stage == CANONICAL_JOINT_FINETUNE:
                self.policy.set_joint_training_stage(
                    freeze_plane=freeze_plane,
                    freeze_shared=freeze_shared,
                )
            else:
                self.policy.set_joint_training_stage(
                    freeze_plane=freeze_plane,
                    freeze_shared=freeze_shared,
                )

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if self.use_anneal:
                self.trainer.policy.hyperparams_anneal(episode, episodes)

            canary_requested_stop = False
            for shard_idx in range(first_shard, self.num_envs):
                shard_started_at = time.monotonic()
                rollout_started_at = shard_started_at
                self.current_shard = shard_idx
                self._report_progress('shard_started', total_shards=int(self.num_envs))
                self.warmup()
                self.policy.reset_counterfactual_diagnostics()
                training_rewards = []
                rollout_steps = 0
                env_done_flags = np.zeros(self.n_rollout_threads, dtype=bool)
                for step in range(self.episode_length):
                    # Sample actions
                    values, actions, action_log_probs, rnn_states, policy_masks = self.collect(step)
                        
                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions)

                    data = (
                        obs,
                        rewards,
                        dones,
                        infos,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        policy_masks,
                    )

                    # insert data into buffer
                    self.insert(data)
                    rollout_steps = step + 1

                    env_done_flags = np.all(dones, axis=1)
                    if self.rollout_until_done and np.all(env_done_flags):
                        break

                rollout_seconds = max(time.monotonic() - rollout_started_at, 1e-9)

                if self.rollout_until_done and not np.all(env_done_flags):
                    unfinished_envs = np.where(~env_done_flags)[0]
                    active_counts = infos['active_agents'][unfinished_envs].sum(axis=1).astype(int).tolist()
                    active_indices = []
                    active_types = []
                    active_actions = []
                    agent_types_arr = infos.get(
                        'agent_types',
                        np.zeros_like(infos['active_agents'], dtype=np.int64),
                    )
                    for env_idx in unfinished_envs:
                        indices = np.flatnonzero(infos['active_agents'][env_idx]).astype(int).tolist()
                        active_indices.append(indices)
                        active_types.append([int(agent_types_arr[env_idx][idx]) for idx in indices])
                        active_actions.append(actions[env_idx, indices].astype(int).tolist())
                    case_ids = infos.get('case_id', np.array([''] * self.n_rollout_threads))[unfinished_envs].tolist()
                    env_steps = infos.get('env_steps', np.zeros(self.n_rollout_threads, dtype=np.int32))[unfinished_envs].tolist()
                    env_total_time = infos.get('env_total_time', np.zeros(self.n_rollout_threads, dtype=np.float32))[unfinished_envs].tolist()
                    active_device_debug = infos.get(
                        'active_device_debug',
                        np.array([None] * self.n_rollout_threads, dtype=object),
                    )[unfinished_envs].tolist()
                    msg = (
                        "Natural rollout did not finish within "
                        f"{self.episode_length} decision steps. "
                        f"unfinished_envs={unfinished_envs.tolist()}, "
                        f"active_counts={active_counts}, "
                        f"active_indices={active_indices}, "
                        f"active_types={active_types}, "
                        f"active_actions={active_actions}, "
                        f"case_ids={case_ids}, "
                        f"env_steps={env_steps}, "
                        f"env_total_time={env_total_time}, "
                        f"active_device_debug={active_device_debug}. "
                        "Increase --rollout_max_steps or inspect the policy for deadlock."
                    )
                    if not self.allow_incomplete_rollout:
                        raise RuntimeError(msg)
                    print(f"[Warning] {msg} Continuing because --allow_incomplete_rollout is set.")

                # compute return and update network
                update_started_at = time.monotonic()
                global_shard_idx = episode * self.num_envs + shard_idx
                actor_update_enabled = global_shard_idx >= self.actor_warmup_shards
                phase_label = (
                    f'{self.experiment_name}:epoch{episode + 1}:'
                    f'shard{shard_idx + 1}'
                )
                with exclusive_gpu_phase(
                    self.shared_gpu_phase_lock, phase_label
                ) as gpu_phase_wait_seconds:
                    self._reset_cuda_peak_memory()
                    self.compute()
                    if self.team_return_mode:
                        epoch_resource_wait_values.append(
                            float(self.last_resource_wait_mean)
                        )
                    train_infos = self.train(
                        update_actor=actor_update_enabled
                    )
                    train_infos['gpu_phase_wait_seconds'] = float(
                        gpu_phase_wait_seconds
                    )
                    # Record the true in-lock peak, then return unused cache to
                    # the card before the next trainer enters its PPO phase.
                    self._record_and_clear_cuda_memory(
                        train_infos, episode, shard_idx
                    )
                lr_decision = self._actor_lr_adaptation_decision(
                    train_infos.get(
                        'actor_planned_optimizer_steps', 0.0
                    ),
                    train_infos.get('actor_optimizer_steps', 0.0),
                    train_infos.get('actor_kl_stop_reason_code', 0.0),
                    self.adaptive_actor_min_step_completion,
                )
                train_infos.update(lr_decision)
                train_infos.update(
                    self.policy.consume_counterfactual_diagnostics()
                )
                train_infos['actor_lr_incomplete_downshift'] = 0.0
                if self.adaptive_actor_kl and actor_update_enabled:
                    if lr_decision['actor_lr_update_eligible'] > 0.0:
                        train_infos.update(self.policy.adapt_actor_lr(
                            train_infos.get(
                                'post_update_probe_approx_kl', np.nan
                            ),
                            low=self.adaptive_actor_kl_low,
                            high=self.adaptive_actor_kl_high,
                            min_scale=self.adaptive_actor_lr_min_scale,
                            max_scale=self.adaptive_actor_lr_max_scale,
                            up=self.adaptive_actor_lr_up,
                            down=self.adaptive_actor_lr_down,
                        ))
                    elif (
                        lr_decision[
                            'actor_lr_incomplete_downshift_requested'
                        ] > 0.0
                    ):
                        train_infos.update(self.policy.downshift_actor_lr(
                            min_scale=self.adaptive_actor_lr_min_scale,
                            down=self.adaptive_actor_lr_down,
                        ))
                train_infos.setdefault(
                    'actor_lr_multiplier',
                    float(self.policy.actor_lr_multiplier),
                )
                train_infos.setdefault(
                    'actor_lr_multiplier_previous',
                    float(self.policy.actor_lr_multiplier),
                )
                train_infos.setdefault('actor_lr_adapted', 0.0)
                train_infos.setdefault(
                    'actor_lr',
                    float(self.policy.actor_optimizer.param_groups[0]['lr']),
                )
                train_infos.setdefault(
                    'shared_actor_lr_scale',
                    float(self.policy.shared_actor_lr_scale),
                )
                train_infos.setdefault(
                    'shared_actor_lr',
                    float(self.policy._actor_group_lr('shared_encoder')),
                )
                train_infos.setdefault(
                    'shared_actor_lr_multiplier_effective',
                    float(self.policy._shared_actor_lr_multiplier()),
                )
                if actor_update_enabled:
                    self._record_actor_update_health(train_infos)
                train_infos['actor_warmup_active'] = float(not actor_update_enabled)
                train_infos['shared_encoder_frozen'] = float(freeze_shared)
                train_infos['plane_order_frozen'] = float(freeze_order)
                train_infos['global_shard_index'] = float(global_shard_idx + 1)
                train_infos['policy_tau'] = float(self.policy.ac.tau)
                update_seconds = max(time.monotonic() - update_started_at, 1e-9)
                shard_seconds = max(time.monotonic() - shard_started_at, 1e-9)
                train_infos['rollout_seconds'] = float(rollout_seconds)
                train_infos['update_seconds'] = float(update_seconds)
                train_infos['shard_seconds'] = float(shard_seconds)
                train_infos['rollout_env_steps_per_second'] = float(
                    self.n_rollout_threads * rollout_steps / rollout_seconds
                )
                train_infos['cases_per_second'] = float(
                    self.n_rollout_threads / shard_seconds
                )
                train_infos['plane_policy_frozen'] = float(freeze_plane)
                if self.team_return_mode:
                    train_infos['team_cmax_mean'] = self.last_team_cmax_mean
                    train_infos['team_return_mean_raw'] = self.last_team_return_mean_raw
                    train_infos['paired_case_delta_mean'] = (
                        self.last_paired_case_delta_mean
                    )
                    train_infos['resource_wait_seconds_mean'] = (
                        self.last_resource_wait_mean
                    )
                    train_infos['resource_critical_wait_seconds_mean'] = (
                        self.last_resource_critical_wait_mean
                    )
                    train_infos[
                        'resource_slack_weighted_wait_seconds_mean'
                    ] = self.last_resource_slack_weighted_wait_mean
                    train_infos[
                        'resource_avoidable_critical_lateness_seconds_mean'
                    ] = self.last_resource_avoidable_critical_lateness_mean
                    train_infos[
                        'resource_rendezvous_spread_seconds_mean'
                    ] = self.last_resource_rendezvous_spread_mean
                    train_infos['resource_early_arrival_seconds_mean'] = (
                        self.last_resource_early_arrival_mean
                    )
                    train_infos[
                        'resource_predicted_lateness_seconds_mean'
                    ] = self.last_resource_predicted_lateness_mean
                    train_infos['team_cycle_count'] = float(self.last_team_cycle_count)
                
                self.total_num_steps += self.n_rollout_threads * rollout_steps
                self.log_train(train_infos, self.total_num_steps)
                if self.team_return_mode:
                    training_rewards.append(self.last_team_return_mean_raw)
                else:
                    training_rewards.append(train_infos["rewards"] / self.reward_coef)

                if (
                    self.recovery_checkpoint_interval_shards > 0
                    and (shard_idx + 1) % self.recovery_checkpoint_interval_shards == 0
                ):
                    self.save(
                        episode,
                        filename='checkpoint_Recovery.pt',
                        extra={
                            'stage': 'post_shard_recovery',
                            'completed_shard': int(shard_idx + 1),
                            'total_shards': int(self.num_envs),
                            'total_num_steps': int(self.total_num_steps),
                            'best_eval_makespan': float(self.best_eval_makespan),
                            'best_eval_iid_makespan': float(
                                self.best_eval_iid_makespan
                            ),
                            'best_eval_composite_makespan': float(
                                self.best_eval_composite_makespan
                            ),
                            'eval_epochs_without_improvement': int(
                                self.eval_epochs_without_improvement
                            ),
                        },
                    )
                self._report_progress(
                    'shard_completed',
                    completed_shard=int(shard_idx + 1),
                    total_shards=int(self.num_envs),
                    rollout_steps=int(rollout_steps),
                    rollout_seconds=float(rollout_seconds),
                    update_seconds=float(update_seconds),
                    shard_seconds=float(shard_seconds),
                    rollout_env_steps_per_second=float(
                        self.n_rollout_threads * rollout_steps / rollout_seconds
                    ),
                    cases_per_second=float(self.n_rollout_threads / shard_seconds),
                    policy_tau=float(self.policy.ac.tau),
                    actor_optimizer_steps=float(
                        train_infos.get('actor_optimizer_steps', 0.0)
                    ),
                    actor_optimizer_rollbacks=float(
                        train_infos.get('actor_optimizer_rollbacks', 0.0)
                    ),
                    actor_backtrack_retry_attempts=float(
                        train_infos.get('actor_backtrack_retry_attempts', 0.0)
                    ),
                    actor_backtrack_failed_groups=float(
                        train_infos.get('actor_backtrack_failed_groups', 0.0)
                    ),
                    actor_backtrack_failed_group_fraction=float(
                        train_infos.get(
                            'actor_backtrack_failed_group_fraction', 0.0
                        )
                    ),
                    actor_planned_optimizer_steps=float(
                        train_infos.get(
                            'actor_planned_optimizer_steps', 0.0
                        )
                    ),
                    actor_step_completion_rate=float(
                        train_infos.get('actor_step_completion_rate', 1.0)
                    ),
                    actor_accumulation_graphs_mean=float(
                        train_infos.get(
                            'actor_accumulation_graphs_mean', 0.0
                        )
                    ),
                    actor_accumulation_graphs_min=float(
                        train_infos.get(
                            'actor_accumulation_graphs_min', 0.0
                        )
                    ),
                    actor_accumulation_graphs_max=float(
                        train_infos.get(
                            'actor_accumulation_graphs_max', 0.0
                        )
                    ),
                    critic_accumulation_graphs_mean=float(
                        train_infos.get(
                            'critic_accumulation_graphs_mean', 0.0
                        )
                    ),
                    critic_accumulation_graphs_min=float(
                        train_infos.get(
                            'critic_accumulation_graphs_min', 0.0
                        )
                    ),
                    critic_accumulation_graphs_max=float(
                        train_infos.get(
                            'critic_accumulation_graphs_max', 0.0
                        )
                    ),
                    actor_kl_stop_reason_code=int(
                        train_infos.get('actor_kl_stop_reason_code', 0.0)
                    ),
                    actor_lr_update_eligible=bool(
                        train_infos.get('actor_lr_update_eligible', 0.0)
                    ),
                    actor_lr_incomplete_downshift=bool(
                        train_infos.get(
                            'actor_lr_incomplete_downshift', 0.0
                        )
                    ),
                    actor_update_health=self._actor_update_health_snapshot(),
                    actor_grad_norm=float(train_infos.get('actor_grad_norm', 0.0)),
                    actor_grad_norm_clipped=float(train_infos.get(
                        'actor_grad_norm_clipped', 0.0
                    )),
                    actor_grad_clip_applied=float(train_infos.get(
                        'actor_grad_clip_applied', 0.0
                    )),
                    actor_group_grad_norms={
                        group: {
                            'raw': float(train_infos.get(
                                f'actor_{group}_grad_norm', 0.0
                            )),
                            'clipped': float(train_infos.get(
                                f'actor_{group}_grad_norm_clipped', 0.0
                            )),
                            'clip_applied': float(train_infos.get(
                                f'actor_{group}_grad_clip_applied', 0.0
                            )),
                        }
                        for group in (
                            'shared_encoder', 'plane_actor',
                            'device_actor', 'transporter_actor',
                        )
                    },
                    counterfactual_baseline_mix=float(train_infos.get(
                        'counterfactual_baseline_mix',
                        self.current_counterfactual_baseline_mix,
                    )),
                    counterfactual_topk_mass_count=float(train_infos.get(
                        'counterfactual_topk_mass_count', 0.0
                    )),
                    counterfactual_topk_mass_mean=float(train_infos.get(
                        'counterfactual_topk_mass_mean', 0.0
                    )),
                    counterfactual_topk_mass_min=float(train_infos.get(
                        'counterfactual_topk_mass_min', 0.0
                    )),
                    counterfactual_topk_mass_p10=float(train_infos.get(
                        'counterfactual_topk_mass_p10', 0.0
                    )),
                    counterfactual_topk_mass_target_fraction=float(
                        train_infos.get(
                            'counterfactual_topk_mass_target_fraction', 0.0
                        )
                    ),
                    role_sequential_macro_steps=float(train_infos.get(
                        'role_sequential_macro_steps', 0.0
                    )),
                    role_sequential_min_ess=float(train_infos.get(
                        'role_sequential_min_ess', 1.0
                    )),
                    actor_optimizer_substeps=float(train_infos.get(
                        'actor_optimizer_substeps', 0.0
                    )),
                    shared_encoder_frozen=bool(freeze_shared),
                    cuda_peak_allocated_gib=float(train_infos.get(
                        'cuda_peak_allocated_gib', 0.0
                    )),
                    cuda_peak_reserved_gib=float(train_infos.get(
                        'cuda_peak_reserved_gib', 0.0
                    )),
                    post_update_probe_approx_kl=float(
                        train_infos.get('post_update_probe_approx_kl', 0.0)
                    ),
                    post_update_probe_clip_fraction=float(
                        train_infos.get('post_update_probe_clip_fraction', 0.0)
                    ),
                    critic_calibration_ratio=float(
                        train_infos.get('critic_calibration_ratio', 0.0)
                    ),
                    bc_reference_kl_coef_after=float(
                        train_infos.get(
                            'bc_reference_kl_coef_after',
                            self.bc_reference_kl_coef,
                        )
                    ),
                    post_update_bc_reference_approx_kl=float(
                        train_infos.get(
                            'post_update_bc_reference_approx_kl', 0.0
                        )
                    ),
                    adaptive_bc_reference_enabled=bool(
                        train_infos.get(
                            'adaptive_bc_reference_enabled', 0.0
                        )
                    ),
                    paired_case_delta_mean=float(
                        train_infos.get(
                            'paired_case_delta_mean',
                            self.last_paired_case_delta_mean,
                        )
                    ),
                    cvar_policy_enabled=bool(train_infos.get(
                        'cvar_policy_enabled', 0.0
                    )),
                    cvar_tail_case_count=float(train_infos.get(
                        'cvar_tail_case_count', 0.0
                    )),
                    cvar_mass_conservation_error=float(train_infos.get(
                        'cvar_mass_conservation_error', 0.0
                    )),
                    cvar_case_metric_paired_delta=bool(train_infos.get(
                        'cvar_case_metric_paired_delta', 0.0
                    )),
                    post_update_plane_approx_kl=float(
                        train_infos.get('post_update_plane_approx_kl', 0.0)
                    ),
                    post_update_device_approx_kl=float(
                        train_infos.get('post_update_device_approx_kl', 0.0)
                    ),
                    post_update_transporter_approx_kl=float(
                        train_infos.get(
                            'post_update_transporter_approx_kl', 0.0
                        )
                    ),
                    role_event_return_diagnostics=dict(getattr(
                        self.buffer,
                        'last_role_event_return_diagnostics',
                        {},
                    )),
                    actor_param_update_relative=float(
                        train_infos.get('actor_param_update_relative', 0.0)
                    ),
                    team_cmax_mean=(
                        float(self.last_team_cmax_mean)
                        if self.team_return_mode else None
                    ),
                )
                print(
                    f"[Timing] epoch={episode + 1}, shard={shard_idx + 1}/{self.num_envs}, "
                    f"rollout_seconds={rollout_seconds:.3f}, "
                    f"update_seconds={update_seconds:.3f}, "
                    f"shard_seconds={shard_seconds:.3f}, "
                    f"rollout_env_steps_per_second="
                    f"{self.n_rollout_threads * rollout_steps / rollout_seconds:.3f}, "
                    f"cases_per_second={self.n_rollout_threads / shard_seconds:.4f}."
                )
                pbar.set_description(f"[Episode {episode+1}]")
                pbar.set_postfix(
                    mean_decision_reward=np.mean(training_rewards),
                    total_num_steps=self.total_num_steps,
                    fps=int(self.total_num_steps / (time.time() - start)))
                completed_shards = shard_idx + 1
                canary_due = (
                    self.use_eval
                    and self.canary_eval_interval_shards > 0
                    and completed_shards < self.num_envs
                    and completed_shards % self.canary_eval_interval_shards == 0
                    and (
                        self.canary_eval_max_per_epoch <= 0
                        or completed_shards
                        // self.canary_eval_interval_shards
                        <= self.canary_eval_max_per_epoch
                    )
                )
                if canary_due:
                    canary_requested_stop = self._run_shard_canary(
                        episode,
                        completed_shards,
                    )
                    if canary_requested_stop:
                        break

            if epoch_resource_wait_values:
                self._update_resource_wait_dual(
                    float(np.mean(epoch_resource_wait_values)), episode
                )

            if self.training_stage == CANONICAL_RESOURCE_JOINT:
                # Bind the checkpoint written below to the protected-state
                # evidence from this PPO epoch, not to a stale pre-update
                # snapshot.
                self.protected_parameter_summary_after_ppo = (
                    self._assert_resource_joint_protected(
                        self.protected_parameter_summary_before_bc,
                        f'resource_joint_ppo_epoch_{episode + 1}',
                    )
                )
                self.resource_actor_summary_after_ppo = (
                    self._resource_actor_summary()
                )
            elif self.training_stage == CANONICAL_JOINT_FINETUNE:
                self.plane_actor_summary_after_ppo = (
                    self._plane_actor_summary()
                )
                self.resource_actor_summary_after_ppo = (
                    self._resource_actor_summary()
                )
                self.shared_actor_summary_after_ppo = (
                    self._shared_actor_summary()
                )

            # Always save a resumable post-train checkpoint before the
            # potentially expensive deterministic validation.
            self.save(
                episode,
                filename='checkpoint_Last.pt',
                extra={'stage': 'post_train_pre_eval'},
            )

            should_stop = canary_requested_stop
            epoch_checkpoint_extra = {
                'stage': 'epoch_completed_without_evaluation',
            }
            should_eval = (
                self.use_eval
                and not self.skip_epoch_eval
                and not canary_requested_stop
                and (
                    episode == 0
                    or (episode + 1) % self.eval_interval == 0
                    or episode == episodes - 1
                )
            )
            if should_eval:
                eval_makespan = float(self.eval(
                    evaluation_label=f'epoch_{episode + 1}'
                ))
                eval_info = self._evaluation_log_info(eval_makespan)
                self.log_train(eval_info, self.total_num_steps)
                if self._update_best_checkpoints(
                    episode,
                    stage='best_eval',
                ):
                    self.eval_epochs_without_improvement = 0
                    print(
                        f"[Info] New best deterministic selection score "
                        f"{eval_makespan:.4f} at epoch {episode + 1}."
                    )
                else:
                    self.eval_epochs_without_improvement += 1
                epoch_checkpoint_extra = {
                    'stage': 'epoch_eval',
                    'eval_makespan': float(eval_makespan),
                    'selection_score': float(
                        self.last_eval_selection_score
                    ),
                    'eval_raw_makespan': float(
                        self.last_eval_raw_makespan
                    ),
                    'eval_iid_makespan': float(
                        self.last_eval_iid_makespan
                    ),
                    'eval_composite_makespan': float(
                        self.last_eval_composite_makespan
                    ),
                    'eval_tail_makespan': float(
                        self.last_eval_tail_makespan
                    ),
                    'evaluation_tau': float(self.evaluation_tau),
                    'selection_metric': self.selection_metric,
                    'best_eval_makespan': float(
                        self.best_eval_makespan
                    ),
                    'best_eval_iid_makespan': float(
                        self.best_eval_iid_makespan
                    ),
                    'best_eval_composite_makespan': float(
                        self.best_eval_composite_makespan
                    ),
                }
                self._report_progress(
                    'evaluation_completed',
                    **eval_info,
                    eval_epochs_without_improvement=int(
                        self.eval_epochs_without_improvement
                    ),
                )

                if (
                    self.early_stop_patience > 0
                    and self.eval_epochs_without_improvement >= self.early_stop_patience
                ):
                    should_stop = True
                    print(
                        "[Info] Early stopping after "
                        f"{self.eval_epochs_without_improvement} evaluations without Cmax improvement."
                    )

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode, extra=epoch_checkpoint_extra)

            self._report_progress(
                'epoch_completed',
                best_eval_makespan=float(self.best_eval_makespan),
            )
            if should_stop:
                break

            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats('cumtime')
            # stats.print_stats(30) # 打印耗时前20的函数

        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            self.protected_parameter_summary_after_ppo = (
                self._assert_resource_joint_protected(
                    self.protected_parameter_summary_before_bc,
                    'resource_joint_ppo',
                )
            )
            self.resource_actor_summary_after_ppo = self._resource_actor_summary()
            if (
                self.resource_actor_summary_before_ppo is None
                or self.resource_actor_summary_after_ppo is None
                or self.resource_actor_summary_before_ppo
                == self.resource_actor_summary_after_ppo
            ):
                raise RuntimeError(
                    "resource_joint PPO completed without a bitwise resource "
                    "actor update."
                )
            self._set_resource_joint_phase(
                'resource_joint_completed',
                event='resource_joint_completed',
            )
        elif self.training_stage == CANONICAL_JOINT_FINETUNE:
            self.plane_actor_summary_after_ppo = self._plane_actor_summary()
            self.resource_actor_summary_after_ppo = (
                self._resource_actor_summary()
            )
            self.shared_actor_summary_after_ppo = self._shared_actor_summary()
            unchanged = []
            for role, before, after in (
                (
                    'plane', self.plane_actor_summary_before_ppo,
                    self.plane_actor_summary_after_ppo,
                ),
                (
                    'resource', self.resource_actor_summary_before_ppo,
                    self.resource_actor_summary_after_ppo,
                ),
                (
                    'shared', self.shared_actor_summary_before_ppo,
                    self.shared_actor_summary_after_ppo,
                ),
            ):
                if role == 'shared' and self.stage3_allow_shared_frozen:
                    if before is None or after is None or before != after:
                        raise RuntimeError(
                            'The Stage3 shared-frozen research arm changed the '
                            'shared encoder.'
                        )
                    continue
                if role == 'shared' and (
                    before is None or after is None or before == after
                ):
                    # A canary/early-stop may end before the scheduled shared
                    # LR ramp, and a KL-rejected macro step is rolled back by
                    # design.  Treat the digest as scientific evidence rather
                    # than crashing after all validation artifacts are ready.
                    print(
                        '[Stage3] shared encoder remained bitwise unchanged; '
                        'recording shared_encoder_updated=false.  Role-head '
                        'updates remain mandatory.'
                    )
                    continue
                if before is None or after is None or before == after:
                    unchanged.append(role)
            if unchanged:
                raise RuntimeError(
                    'joint_finetune PPO completed without bitwise updates for '
                    f'{unchanged}.'
                )
            self._set_joint_finetune_phase(
                'joint_finetune_completed',
                event='joint_finetune_completed',
            )

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        last_step = int(getattr(self.buffer, 'filled_steps', self.episode_length))
        if last_step <= 0:
            raise RuntimeError("Cannot compute returns before collecting any rollout steps.")
        next_values = self.trainer.policy.get_values(
                            Batch.from_data_list(self.buffer.graph_obs[last_step]),
                            self.buffer.rnn_states[last_step],
                            self.buffer.active_masks[last_step],
                            self.buffer.actions[last_step - 1, ..., 0],
                            self.buffer.actions[last_step - 1, ..., 1],
                            )
        next_values = _t2n(next_values).reshape(self.n_rollout_threads, self.num_agents, 1)

        if self.team_return_mode:
            objectives = self.envs.call('get_training_objective')
            if len(objectives) != self.n_rollout_threads:
                raise RuntimeError(
                    "Expected one team objective per rollout thread, got "
                    f"{len(objectives)} for {self.n_rollout_threads} threads."
                )
            cmax_values = np.asarray(
                [objective['cmax'] for objective in objectives],
                dtype=np.float32,
            )
            team_returns_raw = np.asarray(
                [objective['team_return'] for objective in objectives],
                dtype=np.float32,
            )
            if not np.isfinite(cmax_values).all() or not np.isfinite(team_returns_raw).all():
                raise RuntimeError("Non-finite team Cmax objective returned by environment.")
            self.last_team_cmax_mean = float(cmax_values.mean())
            if self.paired_case_baselines:
                reference_values = []
                missing_cases = []
                for objective in objectives:
                    case_key = Path(str(objective.get('case_id', ''))).name
                    reference = self.paired_case_baselines.get(case_key)
                    if reference is None:
                        missing_cases.append(case_key)
                    else:
                        reference_values.append(float(reference))
                if missing_cases:
                    raise RuntimeError(
                        'Paired terminal baseline is missing rollout cases: '
                        f'{sorted(set(missing_cases))[:16]}.'
                    )
                reference_values = np.asarray(
                    reference_values, dtype=np.float32
                )
                paired_deltas = cmax_values - reference_values
                self.buffer.set_team_case_deltas(paired_deltas)
                self.last_paired_case_delta_mean = float(
                    paired_deltas.mean()
                )
                if self.paired_case_baseline_coef > 0.0:
                    paired_offset = (
                        self.paired_case_baseline_coef * reference_values
                    )
                    if self.paired_case_baseline_scope == 'returns':
                        team_returns_raw = team_returns_raw + paired_offset
                    else:
                        self.buffer.set_actor_case_baseline_offsets(
                            paired_offset * self.reward_coef
                        )
            self.last_team_return_mean_raw = float(team_returns_raw.mean())
            self.last_resource_wait_mean = float(np.mean([
                objective.get('resource_wait_seconds', 0.0)
                for objective in objectives
            ]))
            self.last_resource_critical_wait_mean = float(np.mean([
                objective.get('resource_critical_wait_seconds', 0.0)
                for objective in objectives
            ]))
            self.last_resource_slack_weighted_wait_mean = float(np.mean([
                objective.get('resource_slack_weighted_wait_seconds', 0.0)
                for objective in objectives
            ]))
            self.last_resource_avoidable_critical_lateness_mean = float(
                np.mean([
                    objective.get(
                        'resource_avoidable_critical_lateness_seconds', 0.0
                    )
                    for objective in objectives
                ])
            )
            self.last_resource_rendezvous_spread_mean = float(np.mean([
                objective.get('resource_rendezvous_spread_seconds', 0.0)
                for objective in objectives
            ]))
            self.last_resource_early_arrival_mean = float(np.mean([
                objective.get('resource_early_arrival_seconds', 0.0)
                for objective in objectives
            ]))
            self.last_resource_predicted_lateness_mean = float(np.mean([
                objective.get('resource_predicted_lateness_seconds', 0.0)
                for objective in objectives
            ]))
            self.last_team_cycle_count = int(
                sum(bool(objective['cycle_terminated']) for objective in objectives)
            )
            self.buffer.set_team_cmax_values(cmax_values)
            scaled_team_returns = team_returns_raw * self.reward_coef
            if self.team_time_return_mode:
                cmax_coef = (
                    float(self.all_args.hindsight_terminal_cmax_coef)
                    * self.reward_coef
                )
                if bool(getattr(self.all_args, 'role_event_returns', False)):
                    role_event_gae_lambda = float(getattr(
                        self.all_args, 'role_event_gae_lambda', 1.0
                    ))
                    role_event_gae_lambdas = dict(getattr(
                        self.trainer,
                        'role_event_gae_lambdas',
                        {role: role_event_gae_lambda for role in (0, 1, 2)},
                    ))
                    value_baselines = (
                        self.trainer.denormalize_value_predictions(
                            self.buffer.value_preds[:last_step],
                            agent_types=self.buffer.agent_types[:last_step],
                        )
                    )
                    event_credit_weights = None
                    if self.role_event_credit_mode != 'elapsed':
                        event_credit_payloads = self.envs.call(
                            'get_role_event_credit_weights',
                            self.role_event_credit_mode,
                        )
                        event_credit_weights = (
                            self._role_event_credit_tensor(
                                event_credit_payloads,
                                cmax_values,
                                last_step,
                                self.n_rollout_threads,
                                self.num_agents,
                            )
                        )
                    diagnostics = self.buffer.compute_role_event_time_returns(
                        scaled_team_returns,
                        cmax_values,
                        cmax_coef,
                        next_values,
                        gae_lambda=role_event_gae_lambda,
                        role_gae_lambdas=role_event_gae_lambdas,
                        value_baselines=value_baselines,
                        event_credit_weights=event_credit_weights,
                        event_credit_mode=self.role_event_credit_mode,
                        event_credit_uniform_mix=(
                            self.role_event_credit_uniform_mix
                        ),
                        potential_coef=(
                            self.current_iga_potential_beta * self.reward_coef
                            if self.team_time_potential_mode else 0.0
                        ),
                    )
                    for name, value in diagnostics.items():
                        self.writter.add_scalar(
                            f'role_event_returns/{name}',
                            value,
                            max(0, int(self.current_epoch) + 1),
                        )
                elif self.team_time_potential_mode:
                    diagnostics = (
                        self.buffer.compute_team_time_potential_returns(
                            scaled_team_returns,
                            cmax_values,
                            cmax_coef,
                            self.current_iga_potential_beta
                            * self.reward_coef,
                            next_values,
                        )
                    )
                    for name, value in diagnostics.items():
                        self.writter.add_scalar(
                            f'team_time_potential/{name}',
                            value,
                            max(0, int(self.current_epoch) + 1),
                        )
                else:
                    self.buffer.compute_team_time_returns(
                        scaled_team_returns,
                        cmax_values,
                        cmax_coef,
                        next_values,
                    )
            else:
                self.buffer.compute_team_returns(
                    scaled_team_returns,
                    next_values,
                )
            return

        if (
            float(getattr(self.all_args, 'tail_policy_start_fraction', 1.0)) < 1.0
            and float(getattr(self.all_args, 'tail_policy_weight', 1.0)) > 1.0
        ):
            objectives = self.envs.call('get_training_objective')
            cmax_values = np.asarray(
                [objective['cmax'] for objective in objectives],
                dtype=np.float32,
            )
            self.buffer.set_team_cmax_values(cmax_values)

        hindsight_rewards = self.envs.get_rewards()
        self.buffer.rewards.fill(0.0)
        rollout_steps = int(getattr(self.buffer, 'filled_steps', self.episode_length))
        for env_idx, env_rewards in enumerate(hindsight_rewards):
            for (step_idx, agent_id), data in env_rewards.items():
                if step_idx >= rollout_steps:
                    raise RuntimeError(
                        f"Hindsight reward step {step_idx} is outside collected rollout length {rollout_steps}."
                    )
                self.buffer.rewards[step_idx, env_idx, agent_id, 0] = data['reward'] * self.reward_coef

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    @staticmethod
    def _role_event_credit_tensor(
        payloads, cmax_values, rollout_steps, n_envs, n_agents
    ):
        """Validate environment event-credit payloads and align them to replay."""
        if len(payloads) != int(n_envs):
            raise RuntimeError(
                'Expected one event-credit payload per rollout thread, got '
                f'{len(payloads)} for {n_envs} threads.'
            )
        result = np.zeros(
            (int(rollout_steps), int(n_envs), int(n_agents)),
            dtype=np.float32,
        )
        for env_idx, payload in enumerate(payloads):
            if not isinstance(payload, Mapping):
                raise TypeError(
                    'Role-event credit payload must be a mapping.'
                )
            reported_cmax = float(payload.get('cmax', cmax_values[env_idx]))
            tolerance = max(1e-3, 1e-5 * abs(float(cmax_values[env_idx])))
            if (
                not np.isfinite(reported_cmax)
                or abs(reported_cmax - float(cmax_values[env_idx])) > tolerance
            ):
                raise RuntimeError(
                    'Role-event credit Cmax does not match the team objective: '
                    f'env={env_idx}, credit={reported_cmax}, '
                    f'objective={float(cmax_values[env_idx])}.'
                )
            weights = payload.get('weights', {})
            if not isinstance(weights, Mapping):
                raise TypeError('Role-event payload weights must be a mapping.')
            for key, data in weights.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError(
                        'Role-event credit keys must be (step, agent) tuples.'
                    )
                step_idx, agent_id = (int(key[0]), int(key[1]))
                if not 0 <= step_idx < int(rollout_steps):
                    raise ValueError(
                        f'Role-event credit step {step_idx} is outside '
                        f'rollout length {rollout_steps}.'
                    )
                if not 0 <= agent_id < int(n_agents):
                    raise ValueError(
                        f'Role-event credit agent {agent_id} is outside '
                        f'agent count {n_agents}.'
                    )
                value = (
                    data.get('weight', 0.0)
                    if isinstance(data, Mapping) else data
                )
                value = float(value)
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(
                        'Role-event credit weights must be finite and '
                        'non-negative.'
                    )
                result[step_idx, env_idx, agent_id] += value
        return result

    def train(self, update_actor=True):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, update_actor=update_actor)
        self.bc_reference_kl_coef = float(
            self.trainer.bc_reference_kl_coef
        )
        self.all_args.bc_reference_kl_coef = self.bc_reference_kl_coef
        return train_infos

    @staticmethod
    def _graph_potential_values(obs):
        values = []
        for graph in obs:
            value = getattr(graph, 'iga_potential_value', None)
            if value is None:
                values.append(0.0)
                continue
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if array.size != 1 or not np.isfinite(array[0]):
                raise ValueError(
                    'Each graph must contain one finite iga_potential_value.'
                )
            values.append(float(array[0]))
        return np.asarray(values, dtype=np.float32)

    def warmup(self):
        self.trainer.prep_rollout()
        self.buffer.reset_rollout()
        # reset env
        obs, dones, infos = self.envs.reset()

        for thread_idx in range(self.n_rollout_threads):
            self.buffer.graph_obs[0][thread_idx] = obs[thread_idx].clone()
            graph_agent_types = getattr(obs[thread_idx], 'agent_types', None)
            if graph_agent_types is not None:
                if torch.is_tensor(graph_agent_types):
                    graph_agent_types = graph_agent_types.detach().cpu().numpy()
                self.buffer.agent_types[0, thread_idx] = np.asarray(
                    graph_agent_types,
                    dtype=np.int64,
                ).reshape(self.num_agents)

        initial_times = np.asarray(
            infos['env_total_time'], dtype=np.float32
        ).reshape(self.n_rollout_threads)
        self.buffer.decision_times[0] = initial_times
        self.buffer.potential_values[0] = self._graph_potential_values(obs)

        self.buffer.rnn_states[0] = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        
        self.buffer.masks[0] = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)
        self.buffer.masks[0][dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.active_masks[0] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)
        self.buffer.active_masks[0][infos['active_agents'] == True] = np.ones(((infos['active_agents'] == True).sum(), 1), dtype=np.float32)

    def _active_masks_from_info(self, infos):
        active_masks = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[infos['active_agents'] == True] = 1.0
        return active_masks

    @staticmethod
    def _authoritative_policy_history(infos, previous_actions, plane_count):
        """Combine authoritative plane history with persistent device history.

        Plane actions use global operation ids, whereas the environment owns
        and reports each plane's local last-job id.  IGA replay and deployment
        consume that environment history.  Device requests have no equivalent
        environment field, so their recurrent history remains the last action
        actually executed by that device.  Keeping these two sources explicit
        prevents sparse global events from silently changing policy semantics.
        """

        history = np.asarray(previous_actions).copy()
        if history.ndim != 3 or history.shape[-1] < 2:
            raise RuntimeError(
                'Policy history must have shape [env, agent, >=2], got '
                f'{history.shape}.'
            )
        plane_count = min(int(plane_count), history.shape[1])
        expected_shape = history.shape[:2]
        for column, key in enumerate(
            ('last_op_indices', 'last_site_indices')
        ):
            if key not in infos:
                raise RuntimeError(
                    f'Environment info is missing authoritative {key}.'
                )
            values = np.asarray(infos[key])
            if values.size != int(np.prod(expected_shape)):
                raise RuntimeError(
                    f'Environment {key} has shape {values.shape}, expected '
                    f'{expected_shape}.'
                )
            values = values.reshape(expected_shape)
            history[:, :plane_count, column] = values[:, :plane_count]
        return history

    def _dagger_teacher_rate(self, epoch):
        if not self.plane_bc_dagger_schedule:
            return 1.0
        return float(self.plane_bc_dagger_schedule[
            min(int(epoch), len(self.plane_bc_dagger_schedule) - 1)
        ])

    def _dagger_staging_teacher_rate(self, epoch):
        if not self.plane_bc_staging_dagger_schedule:
            return self._dagger_teacher_rate(epoch)
        return float(self.plane_bc_staging_dagger_schedule[
            min(int(epoch), len(self.plane_bc_staging_dagger_schedule) - 1)
        ])

    def _per_agent_dagger_mask(
        self,
        obs,
        teacher_results,
        base_rate,
        staging_rate,
        done_flags,
    ):
        plane_count = self.policy.ac.max_plane_agents
        rates = np.full(
            (self.n_rollout_threads, plane_count),
            float(base_rate),
            dtype=np.float64,
        )
        active = np.zeros_like(rates, dtype=bool)
        for env_idx, (graph, result) in enumerate(zip(obs, teacher_results)):
            teacher_actions = np.asarray(result['actions'], dtype=np.int64)
            teacher_plane = teacher_actions[:plane_count]
            active[env_idx] = (
                (teacher_plane[:, 0] >= 0)
                & (teacher_plane[:, 1] >= 0)
            )
            phase_codes = np.asarray(
                graph.job_phase_codes.detach().cpu(), dtype=np.int64
            )
            n_jobs = len(phase_codes)
            local_ops = np.mod(
                np.maximum(teacher_plane[:, 0], 0), max(1, n_jobs)
            )
            staging = phase_codes[local_ops] == 1
            rates[env_idx, staging & active[env_idx]] = float(
                staging_rate
            )
        active[np.asarray(done_flags, dtype=bool), :] = False
        return (
            self.plane_bc_dagger_rng.random(rates.shape) < rates
        ) & active

    @staticmethod
    def _plane_bc_progress(env_times, teacher_cmaxes):
        env_times = np.asarray(env_times, dtype=np.float64).reshape(-1)
        teacher_cmaxes = np.asarray(
            teacher_cmaxes, dtype=np.float64
        ).reshape(-1)
        if env_times.shape != teacher_cmaxes.shape:
            raise ValueError(
                'BC progress inputs have different shapes: '
                f'{env_times.shape} != {teacher_cmaxes.shape}.'
            )
        if (
            not np.all(np.isfinite(env_times))
            or not np.all(np.isfinite(teacher_cmaxes))
            or np.any(teacher_cmaxes <= 0.0)
        ):
            raise ValueError('BC progress inputs must be finite.')
        return np.clip(env_times / teacher_cmaxes, 0.0, 1.0)

    def _dagger_execution_rates(self, base_rate, progress):
        """Raise teacher execution only in high-risk teacher-tail states."""
        progress = np.asarray(progress, dtype=np.float64).reshape(-1)
        rates = np.full(progress.shape, float(base_rate), dtype=np.float64)
        tail_start = float(getattr(
            self, 'plane_bc_dagger_tail_start_fraction', 1.0
        ))
        tail_rate = float(getattr(
            self, 'plane_bc_dagger_tail_teacher_rate', 0.0
        ))
        if (
            tail_start < 1.0
            and tail_rate > float(base_rate)
        ):
            rates[progress >= tail_start] = tail_rate
        return rates

    def _plane_bc_temporal_weights(self, env_times, teacher_cmaxes):
        """Smoothly emphasize teacher states in the final schedule quarter."""
        progress = self._plane_bc_progress(env_times, teacher_cmaxes)
        if (
            self.plane_bc_tail_weight <= 1.0
            or self.plane_bc_tail_start_fraction >= 1.0
        ):
            return np.ones_like(progress, dtype=np.float32)
        first_end = float(getattr(
            self, 'plane_bc_tail_final_start_fraction', 1.0
        ))
        first_ramp = np.clip(
            (progress - self.plane_bc_tail_start_fraction)
            / max(first_end - self.plane_bc_tail_start_fraction, 1e-12),
            0.0,
            1.0,
        )
        weights = 1.0 + (self.plane_bc_tail_weight - 1.0) * first_ramp
        if first_end < 1.0:
            final_ramp = np.clip(
                (progress - first_end) / max(1.0 - first_end, 1e-12),
                0.0,
                1.0,
            )
            weights += (
                float(getattr(
                    self,
                    'plane_bc_tail_final_weight',
                    self.plane_bc_tail_weight,
                ))
                - self.plane_bc_tail_weight
            ) * final_ramp
        return weights.astype(np.float32)

    def _plane_bc_trainable_modules(self):
        modules = [
            self.policy.ac.encoder,
            self.policy.ac.plane_sel_enc,
            self.policy.ac.actor,
        ]
        if self.policy.ac.plane_order_actor is not None:
            modules.append(self.policy.ac.plane_order_actor)
        return modules

    @staticmethod
    def _mix_feasible_plane_actions(
        policy_actions,
        teacher_actions,
        teacher_preferences,
        active_teacher,
    ):
        """Mix per-plane DAgger choices without creating site contention."""
        policy_actions = np.asarray(policy_actions, dtype=np.int64)
        teacher_actions = np.asarray(teacher_actions, dtype=np.int64)
        teacher_preferences = np.asarray(
            teacher_preferences, dtype=bool
        ).reshape(-1)
        active_teacher = np.asarray(active_teacher, dtype=bool).reshape(-1)
        mixed = policy_actions.copy()
        active_indices = np.flatnonzero(active_teacher)
        if active_indices.size == 0:
            return mixed, np.zeros_like(active_teacher), 0

        preferred = np.where(
            teacher_preferences[:, None], teacher_actions, policy_actions
        )
        alternate = np.where(
            teacher_preferences[:, None], policy_actions, teacher_actions
        )
        preferred_sites = [
            int(preferred[plane_idx, 1]) for plane_idx in active_indices
        ]
        if (
            all(site >= 0 for site in preferred_sites)
            and len(set(preferred_sites)) == len(preferred_sites)
        ):
            mixed[active_indices] = preferred[active_indices]
            realized_teacher = np.zeros_like(active_teacher)
            realized_teacher[active_indices] = np.all(
                preferred[active_indices, :2]
                == teacher_actions[active_indices, :2],
                axis=1,
            )
            return mixed, realized_teacher, 0
        candidate_sites = sorted(set(
            int(action[1])
            for plane_idx in active_indices
            for action in (preferred[plane_idx], alternate[plane_idx])
            if int(action[1]) >= 0
        ))
        if len(candidate_sites) < len(active_indices):
            raise RuntimeError(
                'Per-agent DAgger action union cannot form a conflict-free '
                'site assignment.'
            )
        site_to_column = {
            site: column for column, site in enumerate(candidate_sites)
        }
        unavailable = 1e6
        costs = np.full(
            (len(active_indices), len(candidate_sites)),
            unavailable,
            dtype=np.float64,
        )
        for row, plane_idx in enumerate(active_indices):
            preferred_site = int(preferred[plane_idx, 1])
            alternate_site = int(alternate[plane_idx, 1])
            if preferred_site >= 0:
                costs[row, site_to_column[preferred_site]] = 1e-9 * row
            if alternate_site >= 0:
                alternate_column = site_to_column[alternate_site]
                costs[row, alternate_column] = min(
                    costs[row, alternate_column], 1.0 + 1e-9 * row
                )
        rows, columns = linear_sum_assignment(costs)
        if len(rows) != len(active_indices) or np.any(
            costs[rows, columns] >= unavailable
        ):
            raise RuntimeError(
                'Failed to repair per-agent DAgger site contention.'
            )

        realized_teacher = np.zeros_like(active_teacher)
        conflict_repairs = 0
        for row, column in zip(rows, columns):
            plane_idx = int(active_indices[int(row)])
            assigned_site = candidate_sites[int(column)]
            preferred_action = preferred[plane_idx]
            alternate_action = alternate[plane_idx]
            if int(preferred_action[1]) == assigned_site:
                selected = preferred_action
            elif int(alternate_action[1]) == assigned_site:
                selected = alternate_action
                conflict_repairs += 1
            else:
                raise RuntimeError('DAgger assignment selected an unknown site.')
            mixed[plane_idx] = selected
            realized_teacher[plane_idx] = bool(
                np.array_equal(selected[:2], teacher_actions[plane_idx, :2])
            )
        return mixed, realized_teacher, conflict_repairs

    def _merge_plane_bc_actions(
        self,
        policy_actions,
        teacher_results,
        teacher_execution_mask=None,
    ):
        learned_order = self.policy.ac.plane_order_actor is not None
        action_width = 3 if learned_order else 2
        if policy_actions.shape[-1] != action_width:
            raise ValueError(
                f'Expected plane actions with width {action_width}, '
                f'got {policy_actions.shape[-1]}.'
            )
        actions = policy_actions[..., :action_width].astype(np.int64, copy=True)
        labels = actions.copy()
        if teacher_execution_mask is None:
            teacher_execution_mask = np.ones(
                policy_actions.shape[0], dtype=bool
            )
        teacher_execution_mask = np.asarray(
            teacher_execution_mask, dtype=bool
        )
        plane_count = self.policy.ac.max_plane_agents
        per_agent_execution = teacher_execution_mask.ndim == 2
        if per_agent_execution:
            expected_shape = (policy_actions.shape[0], plane_count)
            if teacher_execution_mask.shape != expected_shape:
                raise ValueError(
                    'Per-agent teacher_execution_mask must have shape '
                    f'{expected_shape}, got {teacher_execution_mask.shape}.'
                )
            if learned_order:
                raise ValueError(
                    'Per-agent DAgger requires fixed plane order.'
                )
        else:
            teacher_execution_mask = teacher_execution_mask.reshape(
                policy_actions.shape[0]
            )
        available_count = 0
        active_plane_labels = 0
        pair_correct = 0
        order_correct = 0
        order_labels = 0
        teacher_executed_planes = 0
        student_executed_planes = 0
        teacher_executed_envs = 0
        student_executed_envs = 0
        conflict_repairs = 0
        for env_idx, result in enumerate(teacher_results):
            info = result.get('info', {}) if isinstance(result, dict) else {}
            if not bool(info.get('available', False)):
                case_ids = getattr(self, '_plane_bc_case_ids', None)
                case_id = case_ids[env_idx] if case_ids is not None else env_idx
                raise RuntimeError(
                    f'No IGA teacher is available for training case {case_id}.'
                )
            teacher_actions = np.asarray(
                result['actions'], dtype=np.int64
            )
            labels[env_idx, :plane_count, :action_width] = teacher_actions[
                :plane_count, :action_width
            ]
            available_count += 1
            teacher_plane_actions = teacher_actions[:plane_count, :action_width]
            active_teacher = (
                (teacher_plane_actions[:, 0] >= 0)
                & (teacher_plane_actions[:, 1] >= 0)
            )
            active_count = int(active_teacher.sum())
            active_plane_labels += active_count
            if per_agent_execution:
                mixed, realized_teacher, repairs = (
                    self._mix_feasible_plane_actions(
                        policy_actions[env_idx, :plane_count, :action_width],
                        teacher_plane_actions,
                        teacher_execution_mask[env_idx],
                        active_teacher,
                    )
                )
                actions[env_idx, :plane_count, :action_width] = mixed
                teacher_executed_planes += int(
                    (realized_teacher & active_teacher).sum()
                )
                student_executed_planes += int(
                    ((~realized_teacher) & active_teacher).sum()
                )
                teacher_executed_envs += int(
                    bool((realized_teacher & active_teacher).any())
                )
                student_executed_envs += int(
                    bool(((~realized_teacher) & active_teacher).any())
                )
                conflict_repairs += int(repairs)
            elif teacher_execution_mask[env_idx]:
                actions[env_idx, :plane_count, :action_width] = teacher_actions[
                    :plane_count, :action_width
                ]
                teacher_executed_planes += active_count
                teacher_executed_envs += int(active_count > 0)
            else:
                student_executed_planes += active_count
                student_executed_envs += int(active_count > 0)
            if active_count > 0:
                predicted = policy_actions[env_idx, :plane_count, :action_width]
                pair_correct += int(
                    (
                        (predicted[:, 0] == teacher_plane_actions[:, 0])
                        & (predicted[:, 1] == teacher_plane_actions[:, 1])
                        & active_teacher
                    ).sum()
                )
                if learned_order:
                    valid_order = active_teacher & (
                        teacher_plane_actions[:, 2] >= 0
                    )
                    order_labels += int(valid_order.sum())
                    order_correct += int(
                        (
                            (predicted[:, 2] == teacher_plane_actions[:, 2])
                            & valid_order
                        ).sum()
                    )
        return actions, labels, {
            'teacher_envs': available_count,
            'teacher_active_planes': active_plane_labels,
            'pair_correct': pair_correct,
            'order_correct': order_correct,
            'order_labels': order_labels,
            'teacher_executed_envs': teacher_executed_envs,
            'student_executed_envs': student_executed_envs,
            'teacher_executed_planes': teacher_executed_planes,
            'student_executed_planes': student_executed_planes,
            'dagger_conflict_repairs': conflict_repairs,
        }

    def _plane_bc_update(
        self,
        obs,
        rnn_states,
        active_masks,
        last_actions,
        label_actions,
        optimizer,
        state_weights=None,
        graph_batch=None,
    ):
        plane_count = self.policy.ac.max_plane_agents
        plane_mask_np = (
            active_masks.squeeze(-1)[:, :plane_count] > 0.0
        )
        if not plane_mask_np.any():
            return None
        (
            action_log_probs,
            _,
            decision_mask,
            log_prob_components,
        ) = self.policy.evaluate_actions(
            (
                graph_batch
                if graph_batch is not None
                else Batch.from_data_list(obs)
            ),
            rnn_states,
            active_masks,
            last_actions[..., 0],
            last_actions[..., 1],
            label_actions,
            return_decision_mask=True,
            return_log_prob_components=True,
        )
        plane_mask = torch.zeros_like(
            decision_mask, dtype=torch.bool, device=self.device
        )
        plane_mask[:, :plane_count] = torch.as_tensor(
            plane_mask_np, dtype=torch.bool, device=self.device
        )
        plane_mask &= decision_mask.bool()
        label_count = int(plane_mask.sum().item())
        if label_count == 0:
            return None
        pair_mask = plane_mask & log_prob_components[
            'pair_decision_mask'
        ].bool()
        order_mask = plane_mask & log_prob_components[
            'order_decision_mask'
        ].bool()
        current_sites_np = np.stack([
            np.asarray(
                graph.agent_current_site_indices.detach().cpu(),
                dtype=np.int64,
            )[:plane_count]
            for graph in obs
        ])
        critical_ops_np = np.stack([
            np.asarray(
                graph.critical_op_mask.detach().cpu(),
                dtype=bool,
            )
            for graph in obs
        ])
        phase_codes_np = np.stack([
            np.asarray(
                getattr(
                    graph,
                    'job_phase_codes',
                    torch.zeros(
                        critical_ops_np.shape[1], dtype=torch.long
                    ),
                ).detach().cpu(),
                dtype=np.int64,
            )
            for graph in obs
        ])
        service_progress_np = np.stack([
            np.asarray(
                getattr(
                    graph,
                    'agent_service_progress',
                    torch.zeros(plane_count, dtype=torch.float32),
                ).detach().cpu(),
                dtype=np.float32,
            )[:plane_count]
            for graph in obs
        ])
        n_jobs = critical_ops_np.shape[1]
        label_ops_np = np.asarray(
            label_actions[:, :plane_count, 0], dtype=np.int64
        )
        label_sites_np = np.asarray(
            label_actions[:, :plane_count, 1], dtype=np.int64
        )
        local_ops_np = np.mod(np.maximum(label_ops_np, 0), n_jobs)
        initial_np = np.asarray(
            last_actions[:, :plane_count, 0] < 0, dtype=bool
        )
        relocation_np = (
            (current_sites_np >= 0) & (label_sites_np != current_sites_np)
        )
        same_site_np = (
            (current_sites_np >= 0) & (label_sites_np == current_sites_np)
        )
        critical_np = np.take_along_axis(
            critical_ops_np, local_ops_np, axis=1
        )
        selected_phase_np = np.take_along_axis(
            phase_codes_np, local_ops_np, axis=1
        )
        service_np = selected_phase_np == 0
        staging_np = selected_phase_np == 1
        forced_departure_np = selected_phase_np == 2
        staging_hold_np = staging_np & same_site_np
        staging_move_np = staging_np & relocation_np
        stratum_weights_np = np.ones_like(
            label_sites_np, dtype=np.float32
        )
        if self.plane_bc_phase_aware:
            stratum_weights_np[service_np] *= self.plane_bc_service_weight
            stratum_weights_np[staging_hold_np] *= (
                self.plane_bc_staging_hold_weight
            )
            stratum_weights_np[staging_move_np] *= (
                self.plane_bc_staging_move_weight
            )
            stratum_weights_np[initial_np & service_np] *= (
                self.plane_bc_initial_weight
            )
            stratum_weights_np[relocation_np & service_np] *= (
                self.plane_bc_relocation_weight
            )
            stratum_weights_np[critical_np & service_np] *= (
                self.plane_bc_critical_op_weight
            )
            if (
                self.plane_bc_service_tail_start_fraction < 1.0
                and self.plane_bc_service_tail_weight > 1.0
            ):
                service_ramp = np.clip(
                    (
                        service_progress_np
                        - self.plane_bc_service_tail_start_fraction
                    ) / max(
                        1.0 - self.plane_bc_service_tail_start_fraction,
                        1e-12,
                    ),
                    0.0,
                    1.0,
                )
                service_multiplier = 1.0 + (
                    self.plane_bc_service_tail_weight - 1.0
                ) * service_ramp
                stratum_weights_np[service_np] *= service_multiplier[
                    service_np
                ]
        else:
            stratum_weights_np[initial_np] *= self.plane_bc_initial_weight
            stratum_weights_np[relocation_np] *= self.plane_bc_relocation_weight
            stratum_weights_np[critical_np] *= self.plane_bc_critical_op_weight
        stratum_weights = torch.ones_like(
            action_log_probs, dtype=action_log_probs.dtype
        )
        stratum_weights[:, :plane_count] = torch.as_tensor(
            stratum_weights_np,
            dtype=action_log_probs.dtype,
            device=self.device,
        )
        pair_weight = pair_mask.to(action_log_probs.dtype) * stratum_weights
        phase_state_scale = torch.ones(
            (pair_weight.shape[0], 1),
            dtype=pair_weight.dtype,
            device=pair_weight.device,
        )
        if self.plane_bc_phase_aware:
            # Preserve the requested phase multiplier between environment
            # states.  Row normalization alone would cancel a staging weight
            # whenever that state contains only one trainable plane.
            phase_state_scale = (
                pair_weight.sum(dim=1, keepdim=True)
                / pair_mask.to(pair_weight.dtype).sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0)
            ).clamp_min(1.0)
        # Each environment state receives unit base mass regardless of its
        # number of simultaneous teacher decisions; important strata retain
        # their requested relative multipliers.
        pair_weight = pair_weight / pair_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pair_weight = pair_weight * phase_state_scale
        order_weight = order_mask.to(action_log_probs.dtype)
        order_weight = order_weight / order_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        if state_weights is None:
            state_weights = np.ones(
                pair_weight.shape[0], dtype=np.float32
            )
        state_weights = torch.as_tensor(
            state_weights,
            dtype=action_log_probs.dtype,
            device=self.device,
        ).reshape(-1, 1)
        if (
            state_weights.shape[0] != pair_weight.shape[0]
            or not torch.isfinite(state_weights).all()
            or bool((state_weights < 1.0).any())
        ):
            raise ValueError('Invalid plane BC temporal state weights.')
        # Apply temporal emphasis after per-state normalization; applying it
        # before normalization would cancel the intended late-state weight.
        pair_weight = pair_weight * state_weights
        order_weight = order_weight * state_weights
        pair_loss = -(
            log_prob_components['pair_log_probs'] * pair_weight
        ).sum() / pair_weight.sum().clamp_min(1e-8)
        order_loss = -(
            log_prob_components['order_log_probs'] * order_weight
        ).sum() / order_weight.sum().clamp_min(1e-8)
        def category_mask(mask_np, pair_only=True):
            result = torch.zeros_like(pair_mask, dtype=torch.bool)
            result[:, :plane_count] = torch.as_tensor(
                mask_np, dtype=torch.bool, device=self.device
            )
            return result & (pair_mask if pair_only else plane_mask)

        initial_mask = category_mask(initial_np)
        relocation_mask = category_mask(relocation_np)
        same_site_mask = category_mask(same_site_np)
        critical_mask = category_mask(critical_np)
        service_mask = category_mask(service_np)
        staging_hold_mask = category_mask(staging_hold_np)
        staging_move_mask = category_mask(staging_move_np)
        forced_departure_mask = category_mask(
            forced_departure_np, pair_only=False
        )
        pair_nll = -log_prob_components['pair_log_probs']

        def category_nll(mask):
            weight = mask.to(pair_nll.dtype)
            return float((
                (pair_nll * weight).sum()
                / weight.sum().clamp_min(1.0)
            ).detach().cpu().item())

        loss = (
            self.plane_bc_pair_loss_coef * pair_loss
            + self.plane_bc_order_loss_coef * order_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f'Plane BC loss is not finite: {loss.item()}.')
        optimizer.zero_grad()
        loss.backward()
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group['params']
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.all_args.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad()
            raise RuntimeError(
                f'Plane BC gradient norm is not finite: {grad_norm.item()}.'
            )
        optimizer.step()
        return {
            'plane_bc_loss': float(loss.detach().cpu().item()),
            'plane_bc_pair_loss': float(pair_loss.detach().cpu().item()),
            'plane_bc_order_loss': float(order_loss.detach().cpu().item()),
            'plane_bc_grad_norm': float(grad_norm.detach().cpu().item()),
            'plane_bc_labels': label_count,
            'plane_bc_initial_labels': int(initial_mask.sum().item()),
            'plane_bc_relocation_labels': int(relocation_mask.sum().item()),
            'plane_bc_same_site_labels': int(same_site_mask.sum().item()),
            'plane_bc_critical_labels': int(critical_mask.sum().item()),
            'plane_bc_initial_nll': category_nll(initial_mask),
            'plane_bc_relocation_nll': category_nll(relocation_mask),
            'plane_bc_same_site_nll': category_nll(same_site_mask),
            'plane_bc_critical_nll': category_nll(critical_mask),
            'plane_bc_service_labels': int(service_mask.sum().item()),
            'plane_bc_staging_hold_labels': int(
                staging_hold_mask.sum().item()
            ),
            'plane_bc_staging_move_labels': int(
                staging_move_mask.sum().item()
            ),
            'plane_bc_forced_departure_labels': int(
                forced_departure_mask.sum().item()
            ),
            'plane_bc_service_nll': category_nll(service_mask),
            'plane_bc_staging_hold_nll': category_nll(
                staging_hold_mask
            ),
            'plane_bc_staging_move_nll': category_nll(
                staging_move_mask
            ),
            'plane_bc_effective_case_mass': float(
                pair_weight.sum().detach().cpu().item()
            ),
            'plane_bc_mean_temporal_weight': float(
                state_weights.mean().detach().cpu().item()
            ),
            'plane_bc_pair_labels': int(pair_mask.sum().item()),
            'plane_bc_order_labels': int(order_mask.sum().item()),
        }

    def plane_bc_pretrain(self):
        resource_policy = getattr(
            self.all_args, 'resource_policy', 'heuristic'
        )
        if self.training_stage == CANONICAL_JOINT_FINETUNE:
            if resource_policy != 'drl':
                raise RuntimeError(
                    'Stage3 joint plane BC requires neural resource actors.'
                )
            teacher_method = 'joint_iga_plane_teacher_actions'
            self.plane_actor_summary_before_bc = self._plane_actor_summary()
        else:
            if resource_policy != 'heuristic':
                raise RuntimeError(
                    'Stage1 IGA plane BC requires heuristic resource policy.'
                )
            teacher_method = 'iga_teacher_actions'
        trainable_modules = self._plane_bc_trainable_modules()
        shared_params = list(self.policy.ac.encoder.parameters())
        backend_modules = [
            self.policy.ac.plane_sel_enc,
            self.policy.ac.actor,
        ]
        if self.policy.ac.plane_order_actor is not None:
            backend_modules.append(self.policy.ac.plane_order_actor)
        backend_params = [
            parameter
            for module in backend_modules
            for parameter in module.parameters()
        ]
        trainable_params = shared_params + backend_params
        previous_requires_grad = self._set_device_bc_requires_grad(
            trainable_params
        )
        bc_lr = (
            self.plane_bc_lr if self.plane_bc_lr > 0.0
            else self.all_args.lr
        )
        optimizer = torch.optim.Adam(
            [
                {
                    'params': shared_params,
                    'lr': bc_lr * self.plane_bc_shared_lr_scale,
                    'name': 'shared_encoder',
                },
                {
                    'params': backend_params,
                    'lr': bc_lr,
                    'name': 'plane_backend',
                },
            ],
            lr=bc_lr,
            eps=self.all_args.opti_eps,
            weight_decay=self.all_args.weight_decay,
        )
        rollout_target = (
            self.plane_bc_rollouts_per_epoch
            if self.plane_bc_rollouts_per_epoch > 0
            else max(1, int(self.num_envs))
        )
        bc_step = 0
        progress_interval = max(
            10.0,
            min(
                60.0,
                float(getattr(self.all_args, 'status_heartbeat_seconds', 60.0)),
            ),
        )
        last_progress_report = 0.0
        try:
            self.policy.ac.eval()
            for module in trainable_modules:
                module.train()
            self._report_progress(
                'plane_bc_started',
                plane_bc_epoch=0,
                plane_bc_total_epochs=int(self.plane_bc_pretrain_epochs),
                plane_bc_rollout=0,
                plane_bc_total_rollouts=int(rollout_target),
                plane_bc_env_step=0,
                plane_bc_updates=0,
            )
            for epoch in range(self.plane_bc_pretrain_epochs):
                teacher_rate = self._dagger_teacher_rate(epoch)
                freeze_shared = epoch < self.plane_bc_freeze_shared_epochs
                for parameter in shared_params:
                    parameter.requires_grad_(not freeze_shared)
                self.envs.shuffer_data()
                epoch_loss = 0.0
                epoch_pair_loss = 0.0
                epoch_order_loss = 0.0
                epoch_updates = 0
                epoch_labels = 0
                epoch_pair_labels = 0
                epoch_order_labels = 0
                epoch_teacher_envs = 0
                epoch_teacher_active_planes = 0
                epoch_pair_correct = 0
                epoch_order_correct = 0
                epoch_order_targets = 0
                epoch_teacher_executed_envs = 0
                epoch_student_executed_envs = 0
                epoch_dagger_conflict_repairs = 0
                epoch_phase_labels = {
                    phase: 0 for phase in (
                        'service', 'staging_hold', 'staging_move',
                        'forced_departure',
                    )
                }
                epoch_phase_nll_mass = {
                    phase: 0.0 for phase in (
                        'service', 'staging_hold', 'staging_move',
                    )
                }
                for rollout_idx in range(rollout_target):
                    rollout_started_at = time.monotonic()
                    obs, dones, infos = self.envs.reset()
                    self._plane_bc_case_ids = np.asarray(
                        infos.get('case_id', np.arange(self.n_rollout_threads)),
                        dtype=object,
                    ).reshape(-1)
                    rnn_states = np.zeros(
                        (
                            self.n_rollout_threads,
                            self.num_agents,
                            self.recurrent_N,
                            self.hidden_size,
                        ),
                        dtype=np.float32,
                    )
                    action_width = (
                        3
                        if self.policy.ac.plane_order_actor is not None
                        else 2
                    )
                    last_actions = -np.ones(
                        (self.n_rollout_threads, self.num_agents, action_width),
                        dtype=np.int64,
                    )
                    done_flags = np.zeros(self.n_rollout_threads, dtype=bool)
                    for _step in range(self.episode_length):
                        active_masks = self._active_masks_from_info(infos)
                        policy_history = self._authoritative_policy_history(
                            infos,
                            last_actions,
                            self.policy.ac.max_plane_agents,
                        )
                        teacher_rpc_pending = False
                        if self.safe_dagger_teacher_overlap:
                            self.envs.call_async(
                                teacher_method, return_info=True
                            )
                            teacher_rpc_pending = True
                        graph_batch = (
                            Batch.from_data_list(obs)
                            if self.safe_graph_batch_pipeline else None
                        )
                        try:
                            with torch.no_grad():
                                _, policy_actions, _, next_rnn_states = (
                                    self.policy.get_actions(
                                        (
                                            graph_batch
                                            if graph_batch is not None
                                            else Batch.from_data_list(obs)
                                        ),
                                        rnn_states,
                                        active_masks,
                                        policy_history[..., 0],
                                        policy_history[..., 1],
                                        deterministic=True,
                                    )
                                )
                        except BaseException:
                            # Drain the already-dispatched read-only RPC so the
                            # vector environment never remains in a pending
                            # state while the original inference error escapes.
                            if teacher_rpc_pending:
                                try:
                                    self.envs.call_wait()
                                except BaseException:
                                    pass
                            raise
                        teacher_results = (
                            self.envs.call_wait()
                            if teacher_rpc_pending
                            else self.envs.call(
                                teacher_method, return_info=True
                            )
                        )
                        env_times = np.asarray(infos['env_total_time'])
                        teacher_cmaxes = np.asarray([
                                result.get('info', {}).get(
                                    'teacher_cmax', math.nan
                                )
                                for result in teacher_results
                            ])
                        progress = self._plane_bc_progress(
                            env_times, teacher_cmaxes
                        )
                        temporal_weights = self._plane_bc_temporal_weights(
                            env_times, teacher_cmaxes
                        )
                        if self.plane_bc_phase_aware:
                            # Phase-local service progress is applied inside
                            # _plane_bc_update; absolute episode-time tail
                            # weighting is intentionally disabled here.
                            temporal_weights = np.ones_like(
                                temporal_weights, dtype=np.float32
                            )
                        if self.plane_bc_per_agent_dagger:
                            teacher_execution_mask = self._per_agent_dagger_mask(
                                obs,
                                teacher_results,
                                teacher_rate,
                                self._dagger_staging_teacher_rate(epoch),
                                done_flags,
                            )
                        else:
                            execution_rates = self._dagger_execution_rates(
                                teacher_rate, progress
                            )
                            teacher_execution_mask = (
                                self.plane_bc_dagger_rng.random(
                                    self.n_rollout_threads
                                ) < execution_rates
                            )
                            teacher_execution_mask &= ~done_flags
                        policy_actions_np = _t2n(policy_actions)
                        actions, labels, label_stats = self._merge_plane_bc_actions(
                            policy_actions_np,
                            teacher_results,
                            teacher_execution_mask=teacher_execution_mask,
                        )
                        # Teacher action arrays contain sentinels for agents
                        # that did not decide at this event.  Keep the actor's
                        # carry-forward values there so ``last_actions`` stays
                        # a persistent decision history rather than merely
                        # the previous global event's sparse action vector.
                        inactive_agents = active_masks[..., 0] <= 0.0
                        actions[inactive_agents] = policy_actions_np[
                            inactive_agents
                        ]
                        expected_teacher_mask = (
                            active_masks[:, :self.policy.ac.max_plane_agents, 0]
                            > 0.0
                        )
                        actual_teacher_mask = (
                            labels[:, :self.policy.ac.max_plane_agents, 0] >= 0
                        )
                        if not np.array_equal(
                            expected_teacher_mask, actual_teacher_mask
                        ):
                            expected_debug = [
                                np.flatnonzero(row).tolist()
                                for row in expected_teacher_mask
                            ]
                            actual_debug = [
                                np.flatnonzero(row).tolist()
                                for row in actual_teacher_mask
                            ]
                            raise RuntimeError(
                                'IGA teacher labels do not match active planes; '
                                f'case_ids={self._plane_bc_case_ids.tolist()}, '
                                f'expected={expected_debug}, '
                                f'actual={actual_debug}.'
                            )
                        update = self._plane_bc_update(
                            obs,
                            rnn_states,
                            active_masks,
                            policy_history,
                            labels,
                            optimizer,
                            state_weights=temporal_weights,
                            graph_batch=graph_batch,
                        )
                        if update is not None:
                            epoch_updates += 1
                            bc_step += 1
                            epoch_loss += update['plane_bc_loss']
                            epoch_pair_loss += update['plane_bc_pair_loss']
                            epoch_order_loss += update['plane_bc_order_loss']
                            epoch_labels += int(update['plane_bc_labels'])
                            epoch_pair_labels += int(
                                update['plane_bc_pair_labels']
                            )
                            epoch_order_labels += int(
                                update['plane_bc_order_labels']
                            )
                            for phase in epoch_phase_labels:
                                labels_key = f'plane_bc_{phase}_labels'
                                label_total = int(update[labels_key])
                                epoch_phase_labels[phase] += label_total
                                nll_key = f'plane_bc_{phase}_nll'
                                if nll_key in update:
                                    epoch_phase_nll_mass[phase] += (
                                        float(update[nll_key]) * label_total
                                    )
                            self.writter.add_scalar(
                                'plane_bc/loss',
                                update['plane_bc_loss'],
                                bc_step,
                            )
                            self.writter.add_scalar(
                                'plane_bc/pair_loss',
                                update['plane_bc_pair_loss'],
                                bc_step,
                            )
                            self.writter.add_scalar(
                                'plane_bc/order_loss',
                                update['plane_bc_order_loss'],
                                bc_step,
                            )
                            for metric_name in (
                                'plane_bc_initial_nll',
                                'plane_bc_relocation_nll',
                                'plane_bc_same_site_nll',
                                'plane_bc_critical_nll',
                                'plane_bc_initial_labels',
                                'plane_bc_relocation_labels',
                                'plane_bc_same_site_labels',
                                'plane_bc_critical_labels',
                                'plane_bc_effective_case_mass',
                                'plane_bc_service_nll',
                                'plane_bc_staging_hold_nll',
                                'plane_bc_staging_move_nll',
                                'plane_bc_service_labels',
                                'plane_bc_staging_hold_labels',
                                'plane_bc_staging_move_labels',
                                'plane_bc_forced_departure_labels',
                            ):
                                self.writter.add_scalar(
                                    'plane_bc/' + metric_name.replace(
                                        'plane_bc_', ''
                                    ),
                                    update[metric_name],
                                    bc_step,
                                )
                        epoch_teacher_envs += label_stats['teacher_envs']
                        epoch_teacher_active_planes += label_stats[
                            'teacher_active_planes'
                        ]
                        epoch_pair_correct += label_stats['pair_correct']
                        epoch_order_correct += label_stats['order_correct']
                        epoch_order_targets += label_stats['order_labels']
                        epoch_teacher_executed_envs += int(
                            label_stats['teacher_executed_planes']
                        )
                        epoch_student_executed_envs += int(
                            label_stats['student_executed_planes']
                        )
                        epoch_dagger_conflict_repairs += int(
                            label_stats['dagger_conflict_repairs']
                        )
                        obs, _, dones, infos = self.envs.step(actions)
                        next_rnn_states = _t2n(next_rnn_states)
                        next_rnn_states[dones == True] = 0.0
                        rnn_states = next_rnn_states
                        last_actions = actions
                        done_flags |= np.all(dones, axis=1)
                        now = time.monotonic()
                        if (
                            now - last_progress_report >= progress_interval
                            or np.all(done_flags)
                        ):
                            self._report_progress(
                                'plane_bc_progress',
                                plane_bc_epoch=int(epoch + 1),
                                plane_bc_total_epochs=int(
                                    self.plane_bc_pretrain_epochs
                                ),
                                plane_bc_rollout=int(rollout_idx + 1),
                                plane_bc_total_rollouts=int(rollout_target),
                                plane_bc_env_step=int(_step + 1),
                                plane_bc_updates=int(bc_step),
                                plane_bc_epoch_updates=int(epoch_updates),
                                plane_bc_epoch_labels=int(epoch_labels),
                            )
                            last_progress_report = now
                        if self.rollout_until_done and np.all(done_flags):
                            break
                    if self.rollout_until_done and not np.all(done_flags):
                        unfinished = np.where(~done_flags)[0].tolist()
                        active_debug = [
                            np.flatnonzero(
                                np.asarray(infos['active_agents'])[index]
                            ).tolist()
                            for index in unfinished
                        ]
                        raise RuntimeError(
                            'IGA plane BC rollout did not finish; '
                            f'unfinished_envs={unfinished}, '
                            f'case_ids={self._plane_bc_case_ids[unfinished].tolist()}, '
                            f'env_steps={np.asarray(infos.get("env_steps", []))[unfinished].tolist()}, '
                            f'env_total_time={np.asarray(infos.get("env_total_time", []))[unfinished].tolist()}, '
                            f'active_agents={active_debug}, '
                            f'last_actions={last_actions[unfinished].tolist()}.'
                        )
                    rollout_seconds = time.monotonic() - rollout_started_at
                    print(
                        f'[PlaneBC] epoch={epoch + 1}/'
                        f'{self.plane_bc_pretrain_epochs} rollout='
                        f'{rollout_idx + 1}/{rollout_target} '
                        f'env_steps={_step + 1} updates={epoch_updates} '
                        f'elapsed={rollout_seconds:.1f}s.',
                        flush=True,
                    )
                mean_loss = epoch_loss / max(1, epoch_updates)
                mean_pair_loss = epoch_pair_loss / max(1, epoch_updates)
                mean_order_loss = epoch_order_loss / max(1, epoch_updates)
                pair_accuracy = epoch_pair_correct / max(
                    1, epoch_teacher_active_planes
                )
                order_accuracy = epoch_order_correct / max(
                    1, epoch_order_targets
                )
                self.writter.add_scalar(
                    'plane_bc_epoch/loss', mean_loss, epoch + 1
                )
                self.writter.add_scalar(
                    'plane_bc_epoch/pair_loss', mean_pair_loss, epoch + 1
                )
                self.writter.add_scalar(
                    'plane_bc_epoch/order_loss', mean_order_loss, epoch + 1
                )
                self.writter.add_scalar(
                    'plane_bc_epoch/pair_accuracy', pair_accuracy, epoch + 1
                )
                self.writter.add_scalar(
                    'plane_bc_epoch/order_accuracy', order_accuracy, epoch + 1
                )
                self.writter.add_scalar(
                    'plane_bc_epoch/labels', epoch_labels, epoch + 1
                )
                executed_envs = (
                    epoch_teacher_executed_envs
                    + epoch_student_executed_envs
                )
                realized_teacher_rate = (
                    epoch_teacher_executed_envs / max(1, executed_envs)
                )
                self.writter.add_scalar(
                    'plane_bc_epoch/teacher_execution_rate',
                    realized_teacher_rate,
                    epoch + 1,
                )
                epoch_phase_nll = {
                    phase: epoch_phase_nll_mass[phase]
                    / max(1, epoch_phase_labels[phase])
                    for phase in epoch_phase_nll_mass
                }
                for phase, value in epoch_phase_nll.items():
                    self.writter.add_scalar(
                        f'plane_bc_epoch/{phase}_nll', value, epoch + 1
                    )
                for phase, value in epoch_phase_labels.items():
                    self.writter.add_scalar(
                        f'plane_bc_epoch/{phase}_labels', value, epoch + 1
                    )
                print(
                    f'[PlaneBC] epoch={epoch + 1}/'
                    f'{self.plane_bc_pretrain_epochs} loss={mean_loss:.6f} '
                    f'pair_loss={mean_pair_loss:.6f} '
                    f'order_loss={mean_order_loss:.6f} '
                    f'pair_acc={pair_accuracy:.4f} '
                    f'order_acc={order_accuracy:.4f} '
                    f'labels={epoch_labels} teacher_envs={epoch_teacher_envs} '
                    f'teacher_rate={realized_teacher_rate:.3f}/'
                    f'{teacher_rate:.3f} tail_teacher_rate='
                    f'{self.plane_bc_dagger_tail_teacher_rate:.3f} '
                    f'phase_labels={epoch_phase_labels} '
                    f'conflict_repairs={epoch_dagger_conflict_repairs} '
                    f'freeze_shared={freeze_shared}.',
                    flush=True,
                )
                self._report_progress(
                    'plane_bc_epoch_completed',
                    plane_bc_epoch=int(epoch + 1),
                    plane_bc_total_epochs=int(self.plane_bc_pretrain_epochs),
                    plane_bc_rollout=int(rollout_target),
                    plane_bc_total_rollouts=int(rollout_target),
                    plane_bc_env_step=0,
                    plane_bc_updates=int(bc_step),
                    plane_bc_epoch_updates=int(epoch_updates),
                    plane_bc_epoch_labels=int(epoch_labels),
                    plane_bc_epoch_pair_labels=int(epoch_pair_labels),
                    plane_bc_epoch_order_labels=int(epoch_order_labels),
                    plane_bc_epoch_loss=float(mean_loss),
                    plane_bc_epoch_pair_loss=float(mean_pair_loss),
                    plane_bc_epoch_order_loss=float(mean_order_loss),
                    plane_bc_epoch_pair_accuracy=float(pair_accuracy),
                    plane_bc_epoch_order_accuracy=float(order_accuracy),
                    plane_bc_teacher_execution_rate=float(
                        realized_teacher_rate
                    ),
                    plane_bc_teacher_execution_target=float(teacher_rate),
                    plane_bc_phase_labels=dict(epoch_phase_labels),
                    plane_bc_phase_nll=dict(epoch_phase_nll),
                    plane_bc_dagger_conflict_repairs=int(
                        epoch_dagger_conflict_repairs
                    ),
                    plane_bc_shared_frozen=bool(freeze_shared),
                )
        finally:
            self._restore_requires_grad(previous_requires_grad)
            self.policy.ac.train()
            self._plane_bc_case_ids = None
        self.policy.reset_optimizers()
        self.trainer.policy = self.policy
        if self.training_stage == CANONICAL_JOINT_FINETUNE:
            self.plane_actor_summary_after_bc = self._plane_actor_summary()
            if (
                self.plane_actor_summary_before_bc is None
                or self.plane_actor_summary_after_bc
                == self.plane_actor_summary_before_bc
            ):
                raise RuntimeError(
                    'joint_finetune plane BC completed without a bitwise '
                    'plane-actor update.'
                )
        if self.bc_reference_kl_coef > 0.0 or self.bc_reference_target_kl > 0.0:
            self.policy.capture_bc_reference()
            print('[Info] Captured frozen post-BC reference policy for PPO.')
        self.save_plane_bc_checkpoint()
        self._report_progress(
            'plane_bc_completed',
            plane_bc_epoch=int(self.plane_bc_pretrain_epochs),
            plane_bc_total_epochs=int(self.plane_bc_pretrain_epochs),
            plane_bc_rollout=int(rollout_target),
            plane_bc_total_rollouts=int(rollout_target),
            plane_bc_env_step=0,
            plane_bc_updates=int(bc_step),
        )

    def _device_bc_trainable_modules(self):
        if self.device_bc_training_scope == 'ready_only':
            if not self.policy.ac.request_ready_prediction:
                raise RuntimeError(
                    'ready_only Stage2 requires the request-ready predictor.'
                )
            # The policy-injection projection is intentionally excluded.  It
            # is part of the later policy-distillation phase, while this wave
            # measures whether the intrinsic-ready target itself is learnable.
            return [self.policy.ac.request_ready_head]
        modules = list(self.policy.ac.device_actor_param.modules) + [
            self.policy.ac.transporter_sel_enc,
            self.policy.ac.transporter_actor,
        ]
        if self.device_bc_training_scope == 'policy_frozen_ready':
            modules = [module for module in modules
                       if module is not self.policy.ac.request_ready_head
                       and not (self.policy.ac.request_ready_policy_injection == 'none'
                                and module is self.policy.ac.request_ready_feature)]
        # Canonical Stage 2 always keeps the shared encoder and every plane
        # module immutable during resource BC.  The legacy switch remains
        # available for non-Stage-2 callers, but cannot widen this scope.
        if (
            self.device_bc_train_gnn
            and self.training_stage
            not in {CANONICAL_RESOURCE_JOINT, CANONICAL_JOINT_FINETUNE}
        ):
            modules.insert(0, self.policy.ac.encoder)
        return modules

    def _set_device_bc_requires_grad(self, trainable_params):
        trainable_ids = {id(param) for param in trainable_params}
        previous = []
        for param in self.policy.ac.parameters():
            previous.append((param, param.requires_grad))
            param.requires_grad_(id(param) in trainable_ids)
        return previous

    @staticmethod
    def _restore_requires_grad(previous):
        for param, requires_grad in previous:
            param.requires_grad_(requires_grad)

    def _merge_device_bc_actions(
        self,
        policy_actions,
        label_results,
        teacher_execution_mask=None,
        return_metadata=False,
        active_env_mask=None,
    ):
        actions = policy_actions.astype(np.int64, copy=True)
        label_actions = policy_actions.astype(np.int64, copy=True)
        if teacher_execution_mask is None:
            teacher_execution_mask = np.ones(len(label_results), dtype=bool)
        teacher_execution_mask = np.asarray(
            teacher_execution_mask, dtype=bool
        ).reshape(-1)
        if teacher_execution_mask.size != len(label_results):
            raise ValueError(
                'Stage2 DAgger execution mask must have one value per env.'
            )
        active_env_mask = (np.ones(len(label_results), dtype=bool) if active_env_mask is None
                           else np.asarray(active_env_mask, dtype=bool).reshape(-1))
        if active_env_mask.size != len(label_results):
            raise ValueError('Stage2 active execution mask must have one value per env.')
        stats = {
            'preferred_assignments': 0,
            'greedy_legal_fills': 0,
            'real_dispatches': 0,
            'noop_after_claimed': 0,
            'deferred_noop': 0,
            'temporal_defer': 0,
            'capacity_unmatched': 0,
            'claimed_noop': 0,
            'no_demand': 0,
            'non_unique_candidate_masks': 0,
            'ordinary_labels': 0,
            'transporter_labels': 0,
            'teacher_executed_envs': int((teacher_execution_mask & active_env_mask).sum()),
            'student_executed_envs': int((~teacher_execution_mask & active_env_mask).sum()),
        }
        projection_keys = ('teacher_projection_checked', 'teacher_projection_events',
                           'teacher_projection_changed_rows', 'teacher_projection_added_blocking',
                           'teacher_projection_serial_resolves', 'blocking_projection_noop')
        if getattr(getattr(self, 'all_args', None), 'device_bc_teacher_deployment_projection', False):
            stats.update(dict.fromkeys(projection_keys, 0))
        score_margins = np.full(
            (len(label_results), self.num_agents), np.nan, dtype=np.float32
        )
        candidate_scores = [
            [None for _ in range(self.num_agents)]
            for _ in range(len(label_results))
        ]
        noop_causes = np.full(
            (len(label_results), self.num_agents), '', dtype=object
        )
        request_ready_targets = []
        for env_idx, result in enumerate(label_results):
            env_actions = result['actions'] if isinstance(result, dict) else result
            env_info = result.get('info', {}) if isinstance(result, dict) else {}
            if getattr(getattr(self, 'all_args', None), 'device_bc_teacher_deployment_projection', False):
                if env_info.get('teacher_decoder_contract') != TEACHER_PROJECTION_CONTRACT:
                    raise ValueError('IGA-derived teacher RPC contract is missing or changed.')
                for key in projection_keys:
                    if active_env_mask[env_idx]:
                        stats[key] += int(env_info.get(key, 0))
                if active_env_mask[env_idx] and env_info.get('teacher_projection_events', 0):
                    path = Path(self.run_dir) / 'logs/teacher_projection_events.jsonl'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    event = {key: env_info.get(key) for key in (
                        'teacher_path', 'teacher_sha256', 'teacher_decoder_contract',
                        'teacher_projection_case',
                        'original_teacher_cmax', 'derived_teacher_cmax', 'decisions', *projection_keys)}
                    event['env_index'] = env_idx
                    with path.open('a') as handle:
                        handle.write(json.dumps(event) + '\n')
            request_ready_targets.append(
                copy.deepcopy(env_info.get('request_ready_targets', []))
            )
            env_actions = np.asarray(env_actions, dtype=np.int64)
            device_slice = slice(self.policy.ac.max_plane_agents, self.num_agents)
            # DAgger mixes complete resource joint actions per environment.
            # Per-device mixing could duplicate request claims and create an
            # action that neither teacher nor student actually produced.
            if teacher_execution_mask[env_idx]:
                actions[env_idx, device_slice, :2] = env_actions[
                    device_slice, :2
                ]
            # Labels are always queried on the state actually visited, even
            # when the student joint action is executed.
            label_actions[env_idx, device_slice, :2] = env_actions[device_slice, :2]
            for key in (
                'preferred_assignments',
                'greedy_legal_fills',
                'real_dispatches',
                'noop_after_claimed',
                'deferred_noop',
                'temporal_defer',
                'capacity_unmatched',
                'claimed_noop',
                'no_demand',
                'non_unique_candidate_masks',
                'ordinary_labels',
                'transporter_labels',
            ):
                stats[key] += int(env_info.get(key, 0))
            for decision in env_info.get('decisions', ()):
                agent_id = int(decision.get('agent_id', -1))
                margin = decision.get('selected_score_margin')
                if (
                    self.policy.ac.max_plane_agents <= agent_id < self.num_agents
                    and margin is not None
                ):
                    score_margins[env_idx, agent_id] = float(margin)
                raw_scores = decision.get('candidate_scores')
                if (
                    self.policy.ac.max_plane_agents <= agent_id < self.num_agents
                    and isinstance(raw_scores, dict)
                ):
                    candidate_scores[env_idx][agent_id] = {
                        int(request_id): float(score)
                        for request_id, score in raw_scores.items()
                        if np.isfinite(float(score))
                    }
                if (
                    self.policy.ac.max_plane_agents <= agent_id < self.num_agents
                    and int(decision.get('selected_request_id', 0)) == 0
                ):
                    noop_causes[env_idx, agent_id] = str(
                        decision.get('noop_cause')
                        or decision.get('reason')
                        or 'no_demand'
                    )
        if return_metadata:
            return actions, label_actions, stats, {
                'teacher_score_margins': score_margins,
                'teacher_candidate_scores': candidate_scores,
                'teacher_noop_causes': noop_causes,
                'request_ready_targets': request_ready_targets,
            }
        return actions, label_actions, stats

    def _request_ready_supervision_tensors(
        self,
        label_metadata,
        predictions,
        dag_lower_bounds,
        prediction_valid,
        *,
        return_context=False,
        return_kind=False,
    ):
        """Normalize the baseline label schema into dense request tensors."""

        targets = torch.full_like(predictions, float('nan'))
        mask = torch.zeros_like(prediction_valid, dtype=torch.bool)
        blocking = torch.zeros_like(prediction_valid, dtype=torch.bool)
        kind_ids = torch.full_like(
            prediction_valid,
            HKBZ_Runner.REQUEST_READY_KIND_OTHER,
            dtype=torch.long,
        )
        if label_metadata is None:
            result = (targets, mask, blocking) if return_context else (targets, mask)
            return result + (kind_ids,) if return_kind else result
        per_env = label_metadata.get('request_ready_targets', [])
        if not isinstance(per_env, (list, tuple)) or len(per_env) != predictions.shape[0]:
            raise RuntimeError(
                'request_ready_targets must contain one label collection per '
                f'environment; got {type(per_env).__name__} with length '
                f'{len(per_env) if hasattr(per_env, "__len__") else "?"}, '
                f'expected {predictions.shape[0]}.'
            )
        for env_idx, raw_labels in enumerate(per_env):
            if raw_labels is None:
                continue
            if isinstance(raw_labels, Mapping):
                records = []
                for request_id, value in raw_labels.items():
                    if isinstance(value, Mapping):
                        record = dict(value)
                        record.setdefault('request_id', request_id)
                    else:
                        raise RuntimeError(
                            'Scalar ready-lead labels are forbidden: Stage2 '
                            'requires the exact absolute intrinsic_ready_time '
                            'schema, not a DAG/lead-time proxy.'
                        )
                    records.append(record)
            elif isinstance(raw_labels, (list, tuple)):
                records = raw_labels
            else:
                raise RuntimeError(
                    'Each request ready-time label collection must be a '
                    f'mapping or list, got {type(raw_labels).__name__}.'
                )
            for raw_record in records:
                if not isinstance(raw_record, Mapping):
                    raise RuntimeError(
                        'Request ready-time label records must be mappings.'
                    )
                try:
                    request_id = int(raw_record['request_id'])
                    intrinsic_ready_time = float(
                        raw_record['intrinsic_ready_time']
                    )
                    observation_time = float(
                        raw_record['observation_time']
                    )
                    label_schema_version = int(
                        raw_record['label_schema_version']
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        'Request ready-time labels require request_id, numeric '
                        'absolute intrinsic_ready_time/observation_time, and '
                        'label_schema_version.'
                    ) from error
                target_semantics = raw_record.get('target_semantics')
                if (
                    label_schema_version
                    != STAGE2_SUPERVISION_CONTRACT[
                        'ready_label_schema_version'
                    ]
                    or target_semantics
                    != STAGE2_SUPERVISION_CONTRACT[
                        'ready_target_semantics'
                    ]
                ):
                    raise RuntimeError(
                        'Request ready-time label schema/semantics mismatch: '
                        f'version={label_schema_version!r}, '
                        f'semantics={target_semantics!r}.'
                    )
                if not (
                    np.isfinite(intrinsic_ready_time)
                    and np.isfinite(observation_time)
                ):
                    raise RuntimeError(
                        'Request intrinsic-ready and observation times must be '
                        'finite.'
                    )
                target = max(
                    0.0, intrinsic_ready_time - observation_time
                )
                if raw_record.get('ready_lead_seconds') is not None:
                    supplied_lead = float(
                        raw_record['ready_lead_seconds']
                    )
                    if not np.isclose(
                        target, supplied_lead, rtol=0.0, atol=1e-6
                    ):
                        raise RuntimeError(
                            'Request ready_lead_seconds does not equal '
                            'intrinsic_ready_time - observation_time: '
                            f'{supplied_lead} != {target}.'
                        )
                if not 0 <= request_id < predictions.shape[1]:
                    raise RuntimeError(
                        f'Request ready-time label id {request_id} is outside '
                        f'[0, {predictions.shape[1]}).'
                    )
                if not bool(prediction_valid[env_idx, request_id].item()):
                    raise RuntimeError(
                        f'Request ready-time label targets invalid/padded '
                        f'request {request_id} in env {env_idx}.'
                    )
                lower_bound = float(
                    dag_lower_bounds[env_idx, request_id].detach().cpu().item()
                )
                comparison_tolerance = _tensor_rounding_tolerance(
                    dag_lower_bounds.dtype,
                    target,
                    lower_bound,
                )
                if (
                    not np.isfinite(target)
                    or target < lower_bound - comparison_tolerance
                ):
                    raise RuntimeError(
                        'Request ready-time target violates its dependency '
                        f'lower bound in env {env_idx}, request {request_id}: '
                        f'{target} < {lower_bound} beyond '
                        f'tolerance {comparison_tolerance}.'
                    )
                if mask[env_idx, request_id]:
                    previous = float(targets[env_idx, request_id].item())
                    duplicate_tolerance = _tensor_rounding_tolerance(
                        targets.dtype,
                        previous,
                        target,
                    )
                    if not np.isclose(
                        previous,
                        target,
                        rtol=0.0,
                        atol=duplicate_tolerance,
                    ):
                        raise RuntimeError(
                            'Conflicting request ready-time labels for env '
                            f'{env_idx}, request {request_id}: '
                            f'{previous} != {target}.'
                        )
                    continue
                targets[env_idx, request_id] = target
                mask[env_idx, request_id] = True
                request_kind = str(raw_record.get('request_kind', ''))
                blocking[env_idx, request_id] = request_kind == 'blocking_wait'
                kind_ids[env_idx, request_id] = {
                    'bounded_mobile_frontier': HKBZ_Runner.REQUEST_READY_KIND_H1,
                    'bounded_mobile_frontier_h1': HKBZ_Runner.REQUEST_READY_KIND_H1,
                    'bounded_mobile_frontier_h2': HKBZ_Runner.REQUEST_READY_KIND_H2,
                    'blocking_wait': HKBZ_Runner.REQUEST_READY_KIND_BLOCKING,
                    'departure_pickup': HKBZ_Runner.REQUEST_READY_KIND_DEPARTURE,
                }.get(request_kind, HKBZ_Runner.REQUEST_READY_KIND_OTHER)
        result = (targets, mask, blocking) if return_context else (targets, mask)
        return result + (kind_ids,) if return_kind else result

    @classmethod
    def _request_ready_metric_sums(
        cls, predictions, targets, mask, kind_ids
    ):
        """Return additive, label-weighted regression sufficient statistics.

        These values can be summed across decisions and workers without the
        update-size bias of averaging per-update MAEs.  The derived epoch
        metrics are therefore comparable even when different states expose a
        different number of future requests.
        """

        payload = {}
        error = predictions.detach() - targets.detach()
        for code, name in (
            (-1, 'all'),
            *cls.REQUEST_READY_KIND_NAMES.items(),
        ):
            stratum = mask if code == -1 else (mask & (kind_ids == code))
            values = error[stratum]
            prefix = f'request_ready_{name}'
            count = int(values.numel())
            payload[f'{prefix}_count'] = count
            if count == 0:
                for suffix in (
                    'abs_error_sum',
                    'squared_error_sum',
                    'signed_error_sum',
                    'positive_error_sum',
                    'negative_error_sum',
                    'under_count',
                    'over_count',
                    'within_60_count',
                    'within_120_count',
                    'within_300_count',
                    'within_600_count',
                    'late_60_count',
                    'late_120_count',
                    'late_300_count',
                    'late_600_count',
                ):
                    payload[f'{prefix}_{suffix}'] = 0.0
                continue
            absolute = values.abs()
            payload[f'{prefix}_abs_error_sum'] = float(
                absolute.sum().cpu().item()
            )
            payload[f'{prefix}_squared_error_sum'] = float(
                values.square().sum().cpu().item()
            )
            payload[f'{prefix}_signed_error_sum'] = float(
                values.sum().cpu().item()
            )
            payload[f'{prefix}_positive_error_sum'] = float(
                values.clamp_min(0.0).sum().cpu().item()
            )
            payload[f'{prefix}_negative_error_sum'] = float(
                (-values).clamp_min(0.0).sum().cpu().item()
            )
            payload[f'{prefix}_under_count'] = float(
                (values < -1e-6).sum().cpu().item()
            )
            payload[f'{prefix}_over_count'] = float(
                (values > 1e-6).sum().cpu().item()
            )
            for threshold in (60, 120, 300, 600):
                payload[f'{prefix}_within_{threshold}_count'] = float(
                    (absolute <= float(threshold)).sum().cpu().item()
                )
                payload[f'{prefix}_late_{threshold}_count'] = float(
                    (values > float(threshold)).sum().cpu().item()
                )
        return payload

    @classmethod
    def _finalize_request_ready_epoch_metrics(cls, epoch_info):
        """Derive exact per-stratum epoch metrics from additive statistics."""

        additive_suffixes = (
            'count',
            'abs_error_sum',
            'squared_error_sum',
            'signed_error_sum',
            'positive_error_sum',
            'negative_error_sum',
            'under_count',
            'over_count',
            'within_60_count',
            'within_120_count',
            'within_300_count',
            'within_600_count',
            'late_60_count',
            'late_120_count',
            'late_300_count',
            'late_600_count',
        )
        for suffix in additive_suffixes:
            difference = (
                float(epoch_info.get(f'request_ready_all_{suffix}', 0.0))
                - float(epoch_info.get(
                    f'request_ready_blocking_{suffix}', 0.0
                ))
            )
            epoch_info[f'request_ready_nonblocking_{suffix}'] = (
                difference if suffix == 'signed_error_sum'
                else max(0.0, difference)
            )

        for name in (
            'all', 'nonblocking', *cls.REQUEST_READY_KIND_NAMES.values()
        ):
            prefix = f'request_ready_{name}'
            count = int(epoch_info.get(f'{prefix}_count', 0))
            denominator = max(1, count)
            epoch_info[f'{prefix}_mae_seconds'] = float(
                epoch_info.get(f'{prefix}_abs_error_sum', 0.0)
                / denominator
            )
            epoch_info[f'{prefix}_rmse_seconds'] = math.sqrt(max(
                0.0,
                float(epoch_info.get(
                    f'{prefix}_squared_error_sum', 0.0
                )) / denominator,
            ))
            epoch_info[f'{prefix}_bias_seconds'] = float(
                epoch_info.get(f'{prefix}_signed_error_sum', 0.0)
                / denominator
            )
            epoch_info[f'{prefix}_underprediction_rate'] = float(
                epoch_info.get(f'{prefix}_under_count', 0.0)
                / denominator
            )
            epoch_info[f'{prefix}_overprediction_rate'] = float(
                epoch_info.get(f'{prefix}_over_count', 0.0)
                / denominator
            )
            epoch_info[f'{prefix}_late_mae_seconds'] = float(
                epoch_info.get(f'{prefix}_positive_error_sum', 0.0)
                / max(1.0, epoch_info.get(f'{prefix}_over_count', 0.0))
            )
            epoch_info[f'{prefix}_early_mae_seconds'] = float(
                epoch_info.get(f'{prefix}_negative_error_sum', 0.0)
                / max(1.0, epoch_info.get(f'{prefix}_under_count', 0.0))
            )
            for threshold in (60, 120, 300, 600):
                epoch_info[
                    f'{prefix}_within_{threshold}_rate'
                ] = float(
                    epoch_info.get(
                        f'{prefix}_within_{threshold}_count', 0.0
                    ) / denominator
                )
                epoch_info[f'{prefix}_late_{threshold}_rate'] = float(
                    epoch_info.get(
                        f'{prefix}_late_{threshold}_count', 0.0
                    ) / denominator
                )
        # Stable compatibility names now use exact label-weighted values.
        epoch_info['request_ready_mae_seconds'] = epoch_info[
            'request_ready_all_mae_seconds'
        ]
        epoch_info['request_ready_blocking_mae_seconds'] = epoch_info[
            'request_ready_blocking_mae_seconds'
        ]
        epoch_info['request_ready_underprediction_rate'] = epoch_info[
            'request_ready_all_underprediction_rate'
        ]
        # H2 is the long-horizon frontier that dominated the previous error;
        # H1 and departure predictions remain explicit guardrails.  This
        # score is independent of each arm's training loss and can therefore
        # rank loss/weight ablations fairly.
        epoch_info['request_ready_selection_score_seconds'] = float(
            0.60 * epoch_info['request_ready_h2_mae_seconds']
            + 0.30 * epoch_info['request_ready_h1_mae_seconds']
            + 0.10 * epoch_info['request_ready_departure_mae_seconds']
        )

    @staticmethod
    def _request_ready_case_fold(case_id, folds):
        case_name = Path(str(case_id)).name
        digest = hashlib.sha256(case_name.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], byteorder='big') % int(folds)

    @staticmethod
    def _request_ready_case_metric_sums(
        predictions, targets, mask, blocking, case_ids, kind_ids=None
    ):
        """Return non-Blocking sufficient statistics grouped by case."""

        payload = {}
        error = (predictions.detach() - targets.detach()).cpu()
        selected = (mask & ~blocking).detach().cpu()
        kinds = kind_ids.detach().cpu() if kind_ids is not None else None
        for env_idx, raw_case_id in enumerate(case_ids):
            values = error[env_idx][selected[env_idx]]
            if not values.numel():
                continue
            case_id = str(raw_case_id)
            absolute = values.abs()
            stats = payload.setdefault(case_id, {
                'count': 0.0,
                'abs_error_sum': 0.0,
                'squared_error_sum': 0.0,
                'signed_error_sum': 0.0,
                'within_300_count': 0.0,
                'late_300_count': 0.0,
            })
            stats['count'] += float(values.numel())
            stats['abs_error_sum'] += float(absolute.sum().item())
            stats['squared_error_sum'] += float(values.square().sum().item())
            stats['signed_error_sum'] += float(values.sum().item())
            stats['within_300_count'] += float(
                (absolute <= 300.0).sum().item()
            )
            stats['late_300_count'] += float(
                (values > 300.0).sum().item()
            )
            if kinds is not None:
                for code, name in HKBZ_Runner.REQUEST_READY_KIND_NAMES.items():
                    stratum = selected[env_idx] & (kinds[env_idx] == code)
                    errors = error[env_idx][stratum]
                    for suffix, value in {
                        'count': errors.numel(),
                        'abs_error_sum': errors.abs().sum().item(),
                        'squared_error_sum': errors.square().sum().item(),
                        'signed_error_sum': errors.sum().item(),
                        'within_300_count': (errors.abs() <= 300).sum().item(),
                        'late_300_count': (errors > 300).sum().item(),
                    }.items():
                        key = f'{name}_{suffix}'
                        stats[key] = stats.get(key, 0.0) + float(value)
        return payload

    @staticmethod
    def _request_ready_quantile_metric_sums(
        quantiles, targets, mask, blocking
    ):
        selected = mask & ~blocking
        count = int(selected.sum().item())
        payload = {
            'request_ready_quantile_count': count,
            'request_ready_q20_coverage_count': 0.0,
            'request_ready_q50_coverage_count': 0.0,
            'request_ready_q80_coverage_count': 0.0,
            'request_ready_q20_q80_interval_count': 0.0,
            'request_ready_q20_q80_width_sum': 0.0,
        }
        if not count:
            return payload
        predicted = quantiles[selected]
        observed = targets[selected]
        payload['request_ready_q20_coverage_count'] = float(
            (observed <= predicted[:, 0]).sum().item()
        )
        payload['request_ready_q50_coverage_count'] = float(
            (observed <= predicted[:, 1]).sum().item()
        )
        payload['request_ready_q80_coverage_count'] = float(
            (observed <= predicted[:, 2]).sum().item()
        )
        payload['request_ready_q20_q80_interval_count'] = float((
            (observed >= predicted[:, 0])
            & (observed <= predicted[:, 2])
        ).sum().item())
        payload['request_ready_q20_q80_width_sum'] = float(
            (predicted[:, 2] - predicted[:, 0]).sum().item()
        )
        return payload

    @staticmethod
    def _finalize_request_ready_quantile_metrics(epoch_info):
        count = max(1.0, float(epoch_info.get(
            'request_ready_quantile_count', 0.0
        )))
        for quantile in ('q20', 'q50', 'q80'):
            epoch_info[f'request_ready_{quantile}_coverage'] = float(
                epoch_info.get(
                    f'request_ready_{quantile}_coverage_count', 0.0
                ) / count
            )
        epoch_info['request_ready_q20_q80_interval_coverage'] = float(
            epoch_info.get(
                'request_ready_q20_q80_interval_count', 0.0
            ) / count
        )
        epoch_info['request_ready_q20_q80_mean_width_seconds'] = float(
            epoch_info.get(
                'request_ready_q20_q80_width_sum', 0.0
            ) / count
        )

    @staticmethod
    def _merge_request_ready_case_stats(destination, source):
        for case_id, observed in source.items():
            current = destination.setdefault(
                case_id, {key: 0.0 for key in observed}
            )
            for key, value in observed.items():
                current[key] = float(current.get(key, 0.0)) + float(value)

    @staticmethod
    def _finalize_request_ready_case_metrics(epoch_info, case_stats):
        rows = [value for value in case_stats.values() if value['count'] > 0]
        epoch_info['request_ready_case_count'] = int(len(rows))
        if not rows:
            for name in (
                'mae_seconds', 'rmse_seconds', 'bias_seconds',
                'within_300_rate', 'late_300_rate',
            ):
                epoch_info[f'request_ready_case_macro_{name}'] = 0.0
            return
        epoch_info['request_ready_case_macro_mae_seconds'] = float(np.mean([
            row['abs_error_sum'] / row['count'] for row in rows
        ]))
        epoch_info['request_ready_case_macro_rmse_seconds'] = float(np.mean([
            math.sqrt(row['squared_error_sum'] / row['count'])
            for row in rows
        ]))
        epoch_info['request_ready_case_macro_bias_seconds'] = float(np.mean([
            row['signed_error_sum'] / row['count'] for row in rows
        ]))
        epoch_info['request_ready_case_macro_within_300_rate'] = float(np.mean([
            row['within_300_count'] / row['count'] for row in rows
        ]))
        epoch_info['request_ready_case_macro_late_300_rate'] = float(np.mean([
            row['late_300_count'] / row['count'] for row in rows
        ]))
        return epoch_info

    def _device_full_matching_supervision(self, logits, active_mask, labels, obs,
                                          agent_types, label_metadata, infos):
        """Audit pre-claim scores with the very same deployed matching solver."""
        edge_terms, wait_terms, totals = [], [], {}
        causes = (label_metadata or {}).get('teacher_noop_causes')
        if causes is None:
            raise RuntimeError('Full-matching audit requires typed teacher no-op metadata.')
        causes = np.asarray(causes, dtype=object)
        if causes.shape != active_mask.shape:
            raise RuntimeError('Full-matching audit no-op metadata shape mismatch.')
        for env_idx, graph in enumerate(obs):
            rows = np.flatnonzero(active_mask[env_idx])
            if not len(rows):
                continue
            local_types = np.asarray(graph.device_type_ids.detach().cpu(), dtype=np.int64).reshape(-1)
            grouped = {}
            for position, agent in enumerate(rows):
                local = int(agent) - self.policy.ac.max_plane_agents
                if local < 0 or local >= len(local_types):
                    raise RuntimeError('Active resource row is absent from device_type_ids.')
                key = ('transporter', -1) if int(agent_types[env_idx, agent]) == self.policy.ac.AGENT_TYPE_TRANSPORTER else ('ordinary', int(local_types[local]))
                grouped.setdefault(key, []).append(position)
            # Batched request padding is masked -inf. It is never a Blocking request.
            lookahead = np.ones(int(logits.shape[-1]), dtype=bool)
            actual_lookahead = np.asarray(graph.request_is_lookahead.detach().cpu(), dtype=bool).reshape(-1)
            if actual_lookahead.size > lookahead.size:
                raise RuntimeError('Graph requests exceed full-matching score columns.')
            lookahead[:actual_lookahead.size] = actual_lookahead
            selected = logits[env_idx, torch.as_tensor(rows, dtype=torch.long, device=logits.device)]
            target = np.asarray(labels)[env_idx, rows, 0]
            try:
                edge, wait, metrics = full_matching_supervision(
                    selected, target, lookahead, list(grouped.values()), causes[env_idx, rows],
                    edge_enabled=self.device_bc_full_edge_loss_coef > 0,
                    wait_enabled=self.device_bc_empty_wait_loss_coef > 0,
                    margin=self.device_bc_assignment_margin,
                )
            except ValueError as error:
                diagnostic = {
                    'error': str(error), 'env_index': int(env_idx),
                    'case_id': str(np.asarray(infos.get('case_id', ['unknown'] * len(obs)), dtype=object).reshape(-1)[env_idx]),
                    'agents': rows.tolist(), 'target': target.tolist(),
                    'resource_groups': {str(key): value for key, value in grouped.items()},
                    'noop_causes': causes[env_idx, rows].tolist(),
                    'legal_requests': [torch.nonzero(torch.isfinite(row), as_tuple=False).flatten().cpu().tolist() for row in selected],
                    'request_is_lookahead': lookahead.tolist(),
                }
                path = Path(self.run_dir) / 'logs/matching_contract_failures.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as handle:
                    handle.write(json.dumps(diagnostic) + '\n')
                raise RuntimeError(f'Full teacher/deployment contract mismatch: {diagnostic}') from error
            edge_count = metrics['device_bc_full_edge_groups']
            wait_count = metrics['device_bc_wait_eligible_groups']
            if edge_count:
                edge_terms.append((edge, edge_count))
            if wait_count and self.device_bc_empty_wait_loss_coef > 0:
                wait_terms.append((wait, wait_count))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0) + value
        def weighted(terms):
            return sum(value * count for value, count in terms) / sum(count for _, count in terms) if terms else logits.new_zeros(())
        return weighted(edge_terms), weighted(wait_terms), totals

    def _device_assignment_supervision(
        self,
        resource_action_logits,
        active_resource_mask,
        label_actions,
        obs,
        agent_types,
    ):
        """Match teacher request *sets* to exchangeable device rows.

        IGA's chromosome scores are internal search coordinates, not calibrated
        action values.  The scientifically valid target is its final one-to-one
        matching.  For every resource type we therefore rematch that selected
        request set to the current policy rows with Hungarian assignment.  The
        resulting NLL is invariant to identifiers/order of equivalent devices
        and still enforces request uniqueness and the teacher dispatch count.
        """

        zero = resource_action_logits.new_zeros(())
        assignment_terms = []
        margin_terms = []
        exact_count = 0
        f1_total = 0.0
        group_count = 0
        chosen = np.asarray(label_actions)[..., 0]
        active_np = np.asarray(active_resource_mask, dtype=bool)
        agent_types_np = np.asarray(agent_types)
        n_requests = int(resource_action_logits.shape[-1])

        for env_idx, graph in enumerate(obs):
            raw_type_ids = getattr(graph, 'device_type_ids', None)
            if raw_type_ids is None:
                raise RuntimeError(
                    'Final-matching DeviceBC requires graph.device_type_ids.'
                )
            type_ids = np.asarray(
                raw_type_ids.detach().cpu(), dtype=np.int64
            ).reshape(-1)
            groups = {}
            for agent_idx in np.flatnonzero(active_np[env_idx]):
                local_idx = int(agent_idx) - self.policy.ac.max_plane_agents
                if local_idx < 0 or local_idx >= type_ids.size:
                    continue
                role = int(agent_types_np[env_idx, agent_idx])
                key = (
                    ('transporter', -1)
                    if role == self.policy.ac.AGENT_TYPE_TRANSPORTER
                    else ('ordinary', int(type_ids[local_idx]))
                )
                groups.setdefault(key, []).append(int(agent_idx))

            for rows in groups.values():
                teacher_requests = sorted({
                    int(chosen[env_idx, row])
                    for row in rows
                    if 0 < int(chosen[env_idx, row]) < n_requests
                })
                # Empty resource-type frontiers are extremely frequent and do
                # not contain a matching decision.  Their no-op classification
                # is handled separately by the typed timing target.
                if not teacher_requests:
                    continue
                if len(teacher_requests) > len(rows):
                    raise RuntimeError(
                        'Teacher matching assigns more unique requests than '
                        f'devices: rows={rows}, requests={teacher_requests}.'
                    )
                row_tensor = torch.as_tensor(
                    rows, dtype=torch.long, device=self.device
                )
                target_tensor = torch.as_tensor(
                    teacher_requests, dtype=torch.long, device=self.device
                )
                selected_scores = resource_action_logits[
                    env_idx, row_tensor
                ][:, target_tensor]
                noop_scores = resource_action_logits[
                    env_idx, row_tensor, 0
                ].unsqueeze(1)
                noop_count = len(rows) - len(teacher_requests)
                columns = selected_scores
                if noop_count:
                    columns = torch.cat(
                        (columns, noop_scores.expand(-1, noop_count)), dim=1
                    )
                finite = torch.isfinite(columns)
                costs = torch.where(
                    finite,
                    -columns,
                    torch.full_like(columns, 1e9),
                ).detach().cpu().numpy()
                matched_rows, matched_cols = linear_sum_assignment(costs)
                matched = columns[
                    torch.as_tensor(matched_rows, device=self.device),
                    torch.as_tensor(matched_cols, device=self.device),
                ]
                if not bool(torch.isfinite(matched).all().item()):
                    raise RuntimeError(
                        'Teacher final matching cannot be represented by the '
                        f'live legal mask: rows={rows}, requests={teacher_requests}.'
                    )
                assignment_terms.append(-matched.mean())
                group_count += 1

                for row_pos, col_pos in zip(matched_rows, matched_cols):
                    if int(col_pos) >= len(teacher_requests):
                        continue
                    agent_idx = rows[int(row_pos)]
                    request_id = teacher_requests[int(col_pos)]
                    selected = resource_action_logits[
                        env_idx, agent_idx, request_id
                    ]
                    alternatives = resource_action_logits[env_idx, agent_idx].clone()
                    alternatives[target_tensor] = float('-inf')
                    best_alternative = alternatives.max()
                    if torch.isfinite(best_alternative):
                        margin_terms.append(F.relu(
                            selected.new_tensor(self.device_bc_assignment_margin)
                            - selected + best_alternative
                        ))

                # Auditable set-level inference metric, using all legal real
                # columns plus private no-op columns exactly like the decoder.
                group_logits = resource_action_logits[
                    env_idx, row_tensor
                ]
                legal_real = torch.isfinite(group_logits[:, 1:]).any(dim=0)
                real_ids = (
                    torch.nonzero(legal_real, as_tuple=False).flatten() + 1
                )
                eval_columns = group_logits[:, real_ids]
                eval_columns = torch.cat(
                    (eval_columns, noop_scores.expand(-1, len(rows))), dim=1
                )
                eval_cost = torch.where(
                    torch.isfinite(eval_columns),
                    -eval_columns,
                    torch.full_like(eval_columns, 1e9),
                ).detach().cpu().numpy()
                pred_rows, pred_cols = linear_sum_assignment(eval_cost)
                predicted = {
                    int(real_ids[int(col)].detach().cpu().item())
                    for col in pred_cols if int(col) < int(real_ids.numel())
                }
                expected = set(teacher_requests)
                overlap = len(predicted & expected)
                precision = overlap / max(1, len(predicted))
                recall = overlap / max(1, len(expected))
                f1_total += (
                    0.0 if precision + recall == 0.0
                    else 2.0 * precision * recall / (precision + recall)
                )
                exact_count += int(predicted == expected)

        assignment_loss = (
            torch.stack(assignment_terms).mean() if assignment_terms else zero
        )
        margin_loss = torch.stack(margin_terms).mean() if margin_terms else zero
        return {
            'loss': assignment_loss,
            'margin_loss': margin_loss,
            'labels': int(group_count),
            'exact': float(exact_count / max(1, group_count)),
            'f1': float(f1_total / max(1, group_count)),
        }

    @staticmethod
    def _device_bc_timing_class(
        chosen_request,
        request_is_lookahead,
        noop_cause='',
        legacy_noop_timing=False,
    ):
        """Map one teacher action to a timing stratum.

        The legacy Stage2 loss counted every no-op as an intentional defer.
        Current supervision excludes structural no-ops (for example, a
        request already claimed by an earlier resource) because they carry no
        timing choice.  Keeping the old behavior behind an explicit flag lets
        matched-budget baselines reproduce it without weakening the default.
        """

        chosen = int(chosen_request)
        if chosen <= 0:
            if legacy_noop_timing or str(noop_cause) in {
                'temporal_defer', 'deferred_noop'
            }:
                return 2
            return -1
        lookahead = np.asarray(request_is_lookahead, dtype=bool).reshape(-1)
        if chosen < lookahead.size and bool(lookahead[chosen]):
            return 1
        return 0

    def _device_bc_update(
        self,
        obs,
        rnn_states,
        active_masks,
        last_actions,
        label_actions,
        infos,
        optimizer,
        encoder_cache=None,
        return_rnn_states=False,
        label_metadata=None,
        ready_env_mask=None,
        update_parameters=True,
        cost_evidence=None,
    ):
        agent_types = infos.get(
            'agent_types',
            np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int64),
        )
        device_mask_np = (
            (active_masks.squeeze(-1) > 0.0)
            & (agent_types != self.policy.ac.AGENT_TYPE_PLANE)
        )
        label_count = int(device_mask_np.sum())
        if label_count == 0:
            return (None, None) if return_rnn_states else None

        # Fail with an actionable teacher/network contract diff before the
        # pointer actor raises a context-free replay-mask error.
        mismatch_details = []
        for env_idx, graph in enumerate(obs):
            request_masks = np.asarray(
                graph.request_mask_matrix.detach().cpu(), dtype=bool
            ).copy()
            initial_request_masks = request_masks.copy()
            raw_lookahead = getattr(graph, 'request_is_lookahead', None)
            request_is_lookahead = (
                np.asarray(raw_lookahead.detach().cpu(), dtype=bool).reshape(-1)
                if raw_lookahead is not None
                else np.zeros(request_masks.shape[1], dtype=bool)
            )
            active_row = active_masks[env_idx, :, 0] > 0.0
            for agent_idx in range(
                self.policy.ac.max_plane_agents, self.num_agents
            ):
                if not active_row[agent_idx]:
                    continue
                current_real = request_masks[agent_idx, 1:].copy()
                later_real = (
                    request_masks[agent_idx + 1:, 1:].any(axis=0)
                    if agent_idx + 1 < self.num_agents
                    else np.zeros_like(current_real)
                )
                blocking_real = current_real & ~request_is_lookahead[1:]
                last_chance = blocking_real & ~later_real
                has_last_chance = bool(last_chance.any())
                legal = np.zeros(request_masks.shape[1], dtype=bool)
                legal[0] = not has_last_chance
                legal[1:] = last_chance if has_last_chance else current_real
                chosen = int(label_actions[env_idx, agent_idx, 0])
                if chosen < 0 or chosen >= legal.size or not legal[chosen]:
                    mismatch_details.append({
                        'env': int(env_idx),
                        'agent': int(agent_idx),
                        'chosen': chosen,
                        'legal': np.flatnonzero(legal).astype(int).tolist(),
                        'prior_active_resource_labels': [
                            {
                                'agent': int(previous),
                                'chosen': int(label_actions[
                                    env_idx, previous, 0
                                ]),
                                'active': bool(active_row[previous]),
                            }
                            for previous in range(
                                self.policy.ac.max_plane_agents, agent_idx
                            )
                            if int(label_actions[env_idx, previous, 0]) > 0
                        ],
                        'real_request_audit': {
                            int(request_id): {
                                'initial_eligible_agents': [
                                    int(value) for value in np.flatnonzero(
                                        initial_request_masks[:, request_id]
                                    )
                                    if value >= self.policy.ac.max_plane_agents
                                ],
                                'active_eligible_agents': [
                                    int(value) for value in np.flatnonzero(
                                        initial_request_masks[:, request_id]
                                        & active_row
                                    )
                                    if value >= self.policy.ac.max_plane_agents
                                ],
                                'teacher_assigned_agents': [
                                    int(value) for value in np.flatnonzero(
                                        label_actions[env_idx, :, 0]
                                        == request_id
                                    )
                                    if value >= self.policy.ac.max_plane_agents
                                ],
                            }
                            for request_id in np.flatnonzero(legal[1:]) + 1
                        },
                    })
                    continue
                if chosen > 0:
                    request_masks[:, chosen] = False
        if mismatch_details:
            raise RuntimeError(
                'Stage2 BC teacher joint action diverges from the policy '
                f'autoregressive mask: {mismatch_details[:8]}'
            )

        # Only resource log-probabilities contribute to DeviceBC.  Masking
        # frozen plane agents here preserves every resource mask, recurrent
        # input and label while avoiding the otherwise duplicated plane actor
        # work in this second forward pass.
        resource_active_masks = np.asarray(active_masks).copy()
        resource_active_masks[:, :self.policy.ac.max_plane_agents, :] = 0.0
        cached_graph = (
            encoder_cache['graph']
            if encoder_cache is not None
            else Batch.from_data_list(obs)
        )
        evaluation_outputs = self.policy.evaluate_actions(
            cached_graph,
            rnn_states,
            resource_active_masks,
            last_actions[..., 0],
            last_actions[..., 1],
            label_actions,
            agent_types=agent_types,
            return_decision_mask=True,
            return_log_prob_components=True,
            encoded_graph=(
                encoder_cache['encoded_graph']
                if encoder_cache is not None else None
            ),
            return_rnn_states=return_rnn_states,
        )
        if return_rnn_states:
            (
                action_log_probs,
                _,
                decision_mask,
                log_prob_components,
                evaluated_rnn_states,
            ) = evaluation_outputs
        else:
            (
                action_log_probs,
                _,
                decision_mask,
                log_prob_components,
            ) = evaluation_outputs
            evaluated_rnn_states = None
        resource_action_logits = log_prob_components.get(
            'resource_action_logits'
        )
        if resource_action_logits is None:
            raise RuntimeError(
                'DeviceBC structured supervision requires resource logits.'
            )
        cost_loss = resource_action_logits.new_zeros(())
        cost_metrics = {}
        if cost_evidence:
            from onpolicy.utils.stage2_cost_improvement import apply_cost_evidence
            cost_loss, device_mask_np, cost_metrics = apply_cost_evidence(
                resource_action_logits, device_mask_np, cost_evidence,
                scale=self.all_args.stage2_cost_scale_seconds,
                clip=self.all_args.stage2_cost_weight_clip,
                tie_seconds=self.all_args.stage2_cost_tie_seconds)
            cost_metrics['device_bc_cost_loss'] = float(cost_loss.detach().cpu())
        full_edge_loss = resource_action_logits.new_zeros(())
        empty_wait_loss = resource_action_logits.new_zeros(())
        full_matching_metrics = {}
        if getattr(self, 'device_bc_matching_audit', False):
            full_edge_loss, empty_wait_loss, full_matching_metrics = self._device_full_matching_supervision(
                resource_action_logits, device_mask_np, label_actions, obs,
                agent_types, label_metadata, infos,
            )
        ready_loss = resource_action_logits.new_zeros(())
        ready_log_loss = resource_action_logits.new_zeros(())
        ready_seconds_loss = resource_action_logits.new_zeros(())
        ready_quantile_loss = resource_action_logits.new_zeros(())
        ready_label_count = 0
        ready_train_label_count = 0
        ready_mae_seconds = 0.0
        ready_blocking_mae_seconds = 0.0
        ready_underprediction_rate = 0.0
        ready_metric_sums = {}
        ready_quantile_metric_sums = {}
        ready_case_stats = {}
        if (self.policy.ac.request_ready_prediction
                and not getattr(self.all_args, 'device_bc_skip_ready_targets', False)):
            ready_predictions = log_prob_components.get(
                'request_ready_prediction_seconds'
            )
            ready_dag = log_prob_components.get('request_ready_dag_seconds')
            ready_valid = log_prob_components.get(
                'request_ready_prediction_valid'
            )
            ready_network_kind_ids = log_prob_components.get(
                'request_ready_kind_ids'
            )
            ready_quantiles = log_prob_components.get(
                'request_ready_quantile_seconds'
            )
            if ready_predictions is None or ready_dag is None or ready_valid is None:
                raise RuntimeError(
                    'Stage2 request-ready supervision requires prediction '
                    'outputs from the configured network head.'
                )
            ready_targets, ready_mask, ready_blocking, ready_kind_ids = (
                self._request_ready_supervision_tensors(
                label_metadata,
                ready_predictions,
                ready_dag,
                    ready_valid,
                    return_context=True,
                    return_kind=True,
                ))
            if ready_env_mask is not None:
                env_selection = torch.as_tensor(
                    ready_env_mask,
                    dtype=torch.bool,
                    device=ready_mask.device,
                ).reshape(-1, 1)
                if env_selection.shape[0] != ready_mask.shape[0]:
                    raise RuntimeError(
                        'request-ready case split mask differs from the '
                        f'environment batch: {env_selection.shape[0]} != '
                        f'{ready_mask.shape[0]}'
                    )
                ready_mask &= env_selection
            if ready_network_kind_ids is not None:
                kind_mismatch = (
                    ready_mask
                    & ready_network_kind_ids.to(
                        device=ready_kind_ids.device,
                        dtype=ready_kind_ids.dtype,
                    ).ne(ready_kind_ids)
                )
                if bool(kind_mismatch.any().item()):
                    raise RuntimeError(
                        'graph request-kind metadata differs from teacher '
                        'ready-time labels.'
                    )
            ready_label_count = int(ready_mask.sum().item())
            loss_mask = ready_mask
            if self.request_ready_exclude_blocking_loss:
                loss_mask = loss_mask & ~ready_blocking
            ready_train_label_count = int(loss_mask.sum().item())

            if bool(getattr(
                self.all_args, 'request_ready_hard_blocking', False
            )):
                blocking_prediction = ready_predictions[
                    ready_mask & ready_blocking
                ]
                if (
                    blocking_prediction.numel()
                    and not torch.equal(
                        blocking_prediction,
                        torch.zeros_like(blocking_prediction),
                    )
                ):
                    raise RuntimeError(
                        'hard-routed blocking predictions must be exactly zero.'
                    )

            def reduce_ready_terms(terms, base_weights, kinds):
                if not self.request_ready_kind_balanced_loss:
                    kind_weights = torch.ones_like(base_weights)
                    kind_weights = torch.where(
                        kinds == self.REQUEST_READY_KIND_H1,
                        kind_weights * self.request_ready_h1_weight,
                        kind_weights,
                    )
                    kind_weights = torch.where(
                        kinds == self.REQUEST_READY_KIND_H2,
                        kind_weights * self.request_ready_h2_weight,
                        kind_weights,
                    )
                    kind_weights = torch.where(
                        kinds == self.REQUEST_READY_KIND_DEPARTURE,
                        kind_weights * self.request_ready_departure_weight,
                        kind_weights,
                    )
                    weights = base_weights * kind_weights
                    return (terms * weights).sum() / weights.sum().clamp_min(1.0)

                group_terms = []
                group_weights = []
                configured = {
                    self.REQUEST_READY_KIND_OTHER: 1.0,
                    self.REQUEST_READY_KIND_H1: self.request_ready_h1_weight,
                    self.REQUEST_READY_KIND_H2: self.request_ready_h2_weight,
                    self.REQUEST_READY_KIND_DEPARTURE: (
                        self.request_ready_departure_weight
                    ),
                }
                for kind, group_weight in configured.items():
                    selected = kinds == int(kind)
                    if not bool(selected.any().item()):
                        continue
                    selected_weights = base_weights[selected]
                    group_terms.append(
                        (terms[selected] * selected_weights).sum()
                        / selected_weights.sum().clamp_min(1.0)
                    )
                    group_weights.append(float(group_weight))
                if not group_terms:
                    return terms.new_zeros(())
                weights = terms.new_tensor(group_weights)
                return (
                    torch.stack(group_terms) * weights
                ).sum() / weights.sum().clamp_min(1.0)

            if ready_train_label_count:
                scale = float(self.policy.ac.request_ready_time_scale)
                prediction_log = torch.log1p(
                    ready_predictions[loss_mask] / scale
                )
                target_log = torch.log1p(ready_targets[loss_mask] / scale)
                ready_terms = F.smooth_l1_loss(
                    prediction_log,
                    target_log,
                    reduction='none',
                )
                ready_weights = torch.ones_like(ready_terms)
                underprediction = (
                    ready_predictions[loss_mask]
                    < ready_targets[loss_mask] - 1e-6
                )
                ready_weights = torch.where(
                    underprediction,
                    ready_weights * self.request_ready_underprediction_weight,
                    ready_weights,
                )
                ready_weights = torch.where(
                    ready_blocking[loss_mask],
                    ready_weights * self.request_ready_blocking_weight,
                    ready_weights,
                )
                selected_kind_ids = ready_kind_ids[loss_mask]
                ready_log_loss = reduce_ready_terms(
                    ready_terms, ready_weights, selected_kind_ids
                )
                absolute_seconds_terms = F.smooth_l1_loss(
                    (
                        ready_predictions[loss_mask]
                        - ready_targets[loss_mask]
                    ) / self.request_ready_seconds_loss_scale,
                    torch.zeros_like(ready_predictions[loss_mask]),
                    reduction='none',
                )
                ready_seconds_loss = reduce_ready_terms(
                    absolute_seconds_terms,
                    ready_weights,
                    selected_kind_ids,
                )
                if self.request_ready_quantile_loss_coef > 0.0:
                    if ready_quantiles is None:
                        raise RuntimeError(
                            'quantile ready loss requires network quantiles.'
                        )
                    selected_quantiles = ready_quantiles[loss_mask]
                    quantile_levels = selected_quantiles.new_tensor(
                        (0.20, 0.50, 0.80)
                    )
                    quantile_error = (
                        ready_targets[loss_mask].unsqueeze(-1)
                        - selected_quantiles
                    ) / self.request_ready_seconds_loss_scale
                    quantile_terms = torch.maximum(
                        quantile_levels * quantile_error,
                        (quantile_levels - 1.0) * quantile_error,
                    ).mean(dim=-1)
                    ready_quantile_loss = reduce_ready_terms(
                        quantile_terms,
                        ready_weights,
                        selected_kind_ids,
                    )
                ready_loss = (
                    ready_log_loss
                    + self.request_ready_seconds_loss_coef
                    * ready_seconds_loss
                    + self.request_ready_quantile_loss_coef
                    * ready_quantile_loss
                )

            if ready_label_count:
                absolute_error = torch.abs(
                    ready_predictions[ready_mask] - ready_targets[ready_mask]
                )
                ready_mae_seconds = float(
                    absolute_error.mean().detach().cpu().item()
                )
                blocking_values = absolute_error[ready_blocking[ready_mask]]
                if blocking_values.numel():
                    ready_blocking_mae_seconds = float(
                        blocking_values.mean().detach().cpu().item()
                    )
                ready_underprediction_rate = float(
                    (
                        ready_predictions[ready_mask]
                        < ready_targets[ready_mask] - 1e-6
                    ).float().mean().detach().cpu().item()
                )
                ready_metric_sums = self._request_ready_metric_sums(
                    ready_predictions,
                    ready_targets,
                    ready_mask,
                    ready_kind_ids,
                )
                if ready_quantiles is not None:
                    ready_quantile_metric_sums = (
                        self._request_ready_quantile_metric_sums(
                            ready_quantiles,
                            ready_targets,
                            ready_mask,
                            ready_blocking,
                        )
                    )
                ready_case_stats = self._request_ready_case_metric_sums(
                    ready_predictions,
                    ready_targets,
                    ready_mask,
                    ready_blocking,
                    np.asarray(infos.get(
                        'case_id',
                        np.arange(ready_mask.shape[0]),
                    ), dtype=object).reshape(-1),
                    kind_ids=ready_kind_ids if self.device_bc_eval_each_epoch else None,
                )
        device_mask = torch.as_tensor(device_mask_np, dtype=torch.bool, device=self.device)
        # Forced no-op labels after another device claimed the request have a
        # single legal action and exactly zero gradient. Counting them in the
        # denominator diluted real dispatch supervision by up to 20-80x.
        device_mask &= decision_mask.bool()
        decision_trainable_count = int(device_mask.sum().item())
        filtered_ambiguous_count = 0
        if self.device_bc_min_teacher_score_margin >= 0.0:
            if label_metadata is None:
                raise RuntimeError(
                    'Teacher score-margin filtering requires label metadata.'
                )
            score_margins = np.asarray(
                label_metadata.get('teacher_score_margins'),
                dtype=np.float32,
            )
            if score_margins.shape != device_mask_np.shape:
                raise RuntimeError(
                    'Teacher score-margin shape differs from DeviceBC labels: '
                    f'{score_margins.shape} != {device_mask_np.shape}.'
                )
            chosen_request = np.asarray(label_actions)[..., 0]
            ambiguous = (
                (chosen_request > 0)
                & np.isfinite(score_margins)
                & (score_margins < self.device_bc_min_teacher_score_margin)
            )
            ambiguous_tensor = torch.as_tensor(
                ambiguous, dtype=torch.bool, device=self.device
            )
            filtered_ambiguous_count = int(
                (device_mask & ambiguous_tensor).sum().item()
            )
            device_mask &= ~ambiguous_tensor
        label_count = int(device_mask.sum().item())
        if label_count == 0:
            if ready_label_count == 0 and not getattr(self, 'device_bc_matching_audit', False):
                return (
                    (None, evaluated_rnn_states)
                    if return_rnn_states else None
                )
            loss = self.request_ready_loss_coef * ready_loss
            if cost_evidence:
                loss = loss + self.all_args.stage2_cost_loss_coef * cost_loss
            if getattr(self, 'device_bc_full_edge_loss_coef', 0.0) > 0:
                loss = loss + self.device_bc_full_edge_loss_coef * full_edge_loss
            if getattr(self, 'device_bc_empty_wait_loss_coef', 0.0) > 0:
                loss = loss + self.device_bc_empty_wait_loss_coef * empty_wait_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f'Request ready-time loss is not finite: {loss.item()}.'
                )
            grad_norm, did_step = supervised_optimizer_step(
                loss, optimizer, enabled=(update_parameters and (
                    (ready_train_label_count > 0 and self.request_ready_loss_coef > 0.0)
                    or full_edge_loss.requires_grad or empty_wait_loss.requires_grad
                    or cost_loss.requires_grad)),
                max_grad_norm=self.all_args.max_grad_norm,
            )
            ready_only_info = {
                **cost_metrics,
                **full_matching_metrics,
                'device_bc_full_edge_loss': float(full_edge_loss.detach().cpu()),
                'device_bc_empty_wait_loss': float(empty_wait_loss.detach().cpu()),
                'device_bc_optimizer_steps': int(did_step),
                'device_bc_loss': float(loss.detach().cpu().item()),
                'device_bc_categorical_loss': 0.0,
                'device_bc_timing_aux_loss': 0.0,
                'device_bc_ranking_loss': 0.0,
                'device_bc_ranking_labels': 0,
                'device_bc_ranking_accuracy': 0.0,
                'device_bc_assignment_loss': 0.0,
                'device_bc_assignment_margin_loss': 0.0,
                'device_bc_assignment_labels': 0,
                'device_bc_assignment_exact': 0.0,
                'device_bc_assignment_f1': 0.0,
                'device_bc_grad_norm': float(grad_norm.detach().cpu().item()),
                'device_bc_labels': 0,
                'device_bc_ordinary_labels': 0,
                'device_bc_transporter_labels': 0,
                'device_bc_ordinary_loss': 0.0,
                'device_bc_transporter_loss': 0.0,
                'device_bc_decision_trainable_labels': decision_trainable_count,
                'device_bc_filtered_ambiguous_labels': filtered_ambiguous_count,
                'device_bc_blocking_dispatch_labels': 0,
                'device_bc_lookahead_dispatch_labels': 0,
                'device_bc_defer_labels': 0,
                'device_bc_blocking_dispatch_loss': 0.0,
                'device_bc_lookahead_dispatch_loss': 0.0,
                'device_bc_defer_loss': 0.0,
                'request_ready_loss': float(ready_loss.detach().cpu().item()),
                'request_ready_log_loss': float(
                    ready_log_loss.detach().cpu().item()
                ),
                'request_ready_seconds_loss': float(
                    ready_seconds_loss.detach().cpu().item()
                ),
                'request_ready_quantile_loss': float(
                    ready_quantile_loss.detach().cpu().item()
                ),
                'request_ready_labels': ready_label_count,
                'request_ready_train_labels': ready_train_label_count,
                'request_ready_mae_seconds': ready_mae_seconds,
                'request_ready_blocking_mae_seconds': (
                    ready_blocking_mae_seconds
                ),
                'request_ready_underprediction_rate': (
                    ready_underprediction_rate
                ),
                '_request_ready_case_stats': ready_case_stats,
                **ready_metric_sums,
                **ready_quantile_metric_sums,
            }
            return (
                (ready_only_info, evaluated_rnn_states)
                if return_rnn_states else ready_only_info
            )
        ordinary_count = 0
        transporter_count = 0
        ordinary_loss_value = 0.0
        transporter_loss_value = 0.0
        agent_types_tensor = torch.as_tensor(
            agent_types, dtype=torch.long, device=self.device
        )
        ordinary_mask = device_mask & (
            agent_types_tensor == self.policy.ac.AGENT_TYPE_DEVICE
        )
        transporter_mask = device_mask & (
            agent_types_tensor == self.policy.ac.AGENT_TYPE_TRANSPORTER
        )
        ordinary_count = int(ordinary_mask.sum().item())
        transporter_count = int(transporter_mask.sum().item())
        if ordinary_count:
            ordinary_loss_value = float(
                (-action_log_probs[ordinary_mask].mean()).detach().cpu().item()
            )
        if transporter_count:
            transporter_loss_value = float(
                (-action_log_probs[transporter_mask].mean()).detach().cpu().item()
            )

        timing_class_np = np.full(device_mask_np.shape, -1, dtype=np.int64)
        chosen_request = np.asarray(label_actions)[..., 0]
        noop_causes = (
            np.asarray(label_metadata.get('teacher_noop_causes'), dtype=object)
            if label_metadata is not None
            and label_metadata.get('teacher_noop_causes') is not None
            else np.full(device_mask_np.shape, '', dtype=object)
        )
        if noop_causes.shape != device_mask_np.shape:
            raise RuntimeError(
                'Teacher no-op cause shape differs from DeviceBC labels: '
                f'{noop_causes.shape} != {device_mask_np.shape}.'
            )
        for env_idx, graph in enumerate(obs):
            raw_lookahead = getattr(graph, 'request_is_lookahead', None)
            request_is_lookahead = (
                np.asarray(raw_lookahead.detach().cpu(), dtype=bool).reshape(-1)
                if raw_lookahead is not None else np.zeros(1, dtype=bool)
            )
            for agent_idx in np.flatnonzero(device_mask_np[env_idx]):
                chosen = int(chosen_request[env_idx, agent_idx])
                timing_class_np[env_idx, agent_idx] = (
                    self._device_bc_timing_class(
                        chosen,
                        request_is_lookahead,
                        noop_causes[env_idx, agent_idx],
                        self.device_bc_legacy_noop_timing,
                    )
                )
        timing_class = torch.as_tensor(
            timing_class_np, dtype=torch.long, device=self.device
        )
        timing_masks = [device_mask & (timing_class == value) for value in range(3)]
        timing_counts = [int(mask.sum().item()) for mask in timing_masks]
        timing_losses = [
            float((-action_log_probs[mask].mean()).detach().cpu().item())
            if count else 0.0
            for mask, count in zip(timing_masks, timing_counts)
        ]

        strata = []
        if self.device_bc_role_balanced and self.device_bc_timing_balanced:
            for role_mask in (ordinary_mask, transporter_mask):
                strata.extend(
                    role_mask & (timing_class == value) for value in range(3)
                )
        elif self.device_bc_role_balanced:
            strata = [ordinary_mask, transporter_mask]
        elif self.device_bc_timing_balanced:
            strata = timing_masks
        else:
            strata = [device_mask]
        stratum_losses = [
            -action_log_probs[stratum].mean()
            for stratum in strata if bool(stratum.any().item())
        ]
        if not stratum_losses:
            return (
                (None, evaluated_rnn_states)
                if return_rnn_states else None
            )
        categorical_loss = torch.stack(stratum_losses).mean()

        # Factor timing from request identity.  This binary objective is
        # computed from the exact normalized pointer distribution, so it is
        # valid with both the legacy pointer and the optional timing gate.
        dispatch_log_prob = torch.logsumexp(
            resource_action_logits[..., 1:], dim=-1
        )
        noop_log_prob = resource_action_logits[..., 0]
        dispatch_target = torch.as_tensor(
            chosen_request > 0, dtype=torch.bool, device=self.device
        )
        timing_nll = torch.where(
            dispatch_target, -dispatch_log_prob, -noop_log_prob
        )
        typed_timing = timing_class >= 0
        timing_valid = device_mask & typed_timing & torch.isfinite(timing_nll)
        timing_aux_loss = (
            timing_nll[timing_valid].mean()
            if bool(timing_valid.any().item())
            else categorical_loss.new_zeros(())
        )

        # IGA exposes every legal candidate score, not merely the selected
        # action.  Distilling its conditional ranking supplies O(k) dense
        # comparisons at each visited state and removes the information loss
        # that made old DeviceBC indistinguishable from a one-hot heuristic.
        ranking_terms = []
        ranking_correct = 0
        ranking_label_count = 0
        teacher_candidate_scores = (
            label_metadata.get('teacher_candidate_scores')
            if label_metadata is not None else None
        )
        if self.device_bc_ranking_loss_coef > 0.0:
            if teacher_candidate_scores is None:
                raise RuntimeError(
                    'Ranking DeviceBC requires teacher candidate scores.'
                )
            for env_idx in range(len(teacher_candidate_scores)):
                for agent_idx in np.flatnonzero(
                    device_mask[env_idx].detach().cpu().numpy()
                ):
                    raw_scores = teacher_candidate_scores[env_idx][agent_idx]
                    if not isinstance(raw_scores, dict) or len(raw_scores) < 2:
                        continue
                    request_ids = sorted(
                        request_id for request_id in raw_scores
                        if 0 < int(request_id) < resource_action_logits.shape[-1]
                        and bool(torch.isfinite(resource_action_logits[
                            env_idx, agent_idx, int(request_id)
                        ]).item())
                    )
                    if len(request_ids) < 2:
                        continue
                    ids = torch.as_tensor(
                        request_ids, dtype=torch.long, device=self.device
                    )
                    model_scores = resource_action_logits[
                        env_idx, agent_idx, ids
                    ]
                    teacher_scores = torch.as_tensor(
                        [raw_scores[int(request_id)] for request_id in request_ids],
                        dtype=model_scores.dtype,
                        device=self.device,
                    )
                    teacher_scores = teacher_scores - teacher_scores.max()
                    teacher_prob = torch.softmax(
                        teacher_scores / self.device_bc_ranking_temperature,
                        dim=0,
                    )
                    ranking_terms.append(-torch.sum(
                        teacher_prob * torch.log_softmax(model_scores, dim=0)
                    ))
                    ranking_correct += int(
                        int(torch.argmax(model_scores).item())
                        == int(torch.argmax(teacher_scores).item())
                    )
                    ranking_label_count += 1
        ranking_loss = (
            torch.stack(ranking_terms).mean()
            if ranking_terms else categorical_loss.new_zeros(())
        )
        if (
            self.device_bc_assignment_loss_coef > 0.0
            or self.device_bc_assignment_margin_loss_coef > 0.0
            or self.device_bc_min_assignment_labels_per_epoch > 0
        ):
            assignment = self._device_assignment_supervision(
                resource_action_logits,
                device_mask_np,
                label_actions,
                obs,
                agent_types,
            )
        else:
            assignment = {
                'loss': categorical_loss.new_zeros(()),
                'margin_loss': categorical_loss.new_zeros(()),
                'labels': 0,
                'exact': 0.0,
                'f1': 0.0,
            }
        assignment_loss = assignment['loss']
        assignment_margin_loss = assignment['margin_loss']
        # Never form ``0 * inf`` for an intentionally disabled auxiliary
        # objective: globally scored rows can contain impossible categorical
        # teacher identities even though their final matching set is feasible.
        loss = resource_action_logits.new_zeros(())
        for coefficient, term in (
            (self.device_bc_categorical_loss_coef, categorical_loss),
            (self.device_bc_timing_loss_coef, timing_aux_loss),
            (self.device_bc_ranking_loss_coef, ranking_loss),
            (self.device_bc_assignment_loss_coef, assignment_loss),
            (
                self.device_bc_assignment_margin_loss_coef,
                assignment_margin_loss,
            ),
            (self.request_ready_loss_coef, ready_loss),
            (getattr(self, 'device_bc_full_edge_loss_coef', 0.0), full_edge_loss),
            (getattr(self, 'device_bc_empty_wait_loss_coef', 0.0), empty_wait_loss),
        ):
            if coefficient > 0.0:
                loss = loss + coefficient * term
        if cost_evidence:
            loss = loss + self.all_args.stage2_cost_loss_coef * cost_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Device BC loss is not finite: {loss.item()}.")

        has_train_loss = (
            ready_train_label_count > 0
            if self.device_bc_training_scope == 'ready_only'
            else label_count > 0 or cost_loss.requires_grad or (
                self.request_ready_loss_coef > 0.0 and ready_train_label_count > 0
            )
        )
        grad_norm, did_step = supervised_optimizer_step(
            loss, optimizer, enabled=update_parameters and has_train_loss,
            max_grad_norm=self.all_args.max_grad_norm,
        )

        update_info = {
            **cost_metrics,
            **full_matching_metrics,
            'device_bc_full_edge_loss': float(full_edge_loss.detach().cpu()),
            'device_bc_empty_wait_loss': float(empty_wait_loss.detach().cpu()),
            'device_bc_optimizer_steps': int(did_step),
            'device_bc_loss': float(loss.detach().cpu().item()),
            'device_bc_categorical_loss': float(
                categorical_loss.detach().cpu().item()
            ),
            'device_bc_timing_aux_loss': float(
                timing_aux_loss.detach().cpu().item()
            ),
            'device_bc_ranking_loss': float(
                ranking_loss.detach().cpu().item()
            ),
            'device_bc_ranking_labels': int(ranking_label_count),
            'device_bc_ranking_accuracy': float(
                ranking_correct / max(1, ranking_label_count)
            ),
            'device_bc_assignment_loss': float(
                assignment_loss.detach().cpu().item()
            ),
            'device_bc_assignment_margin_loss': float(
                assignment_margin_loss.detach().cpu().item()
            ),
            'device_bc_assignment_labels': int(assignment['labels']),
            'device_bc_assignment_exact': float(assignment['exact']),
            'device_bc_assignment_f1': float(assignment['f1']),
            'device_bc_grad_norm': float(grad_norm.detach().cpu().item()),
            'device_bc_labels': label_count,
            'device_bc_ordinary_labels': ordinary_count,
            'device_bc_transporter_labels': transporter_count,
            'device_bc_ordinary_loss': ordinary_loss_value,
            'device_bc_transporter_loss': transporter_loss_value,
            'device_bc_decision_trainable_labels': decision_trainable_count,
            'device_bc_filtered_ambiguous_labels': filtered_ambiguous_count,
            'device_bc_blocking_dispatch_labels': timing_counts[0],
            'device_bc_lookahead_dispatch_labels': timing_counts[1],
            'device_bc_defer_labels': timing_counts[2],
            'device_bc_blocking_dispatch_loss': timing_losses[0],
            'device_bc_lookahead_dispatch_loss': timing_losses[1],
            'device_bc_defer_loss': timing_losses[2],
            'request_ready_loss': float(ready_loss.detach().cpu().item()),
            'request_ready_log_loss': float(
                ready_log_loss.detach().cpu().item()
            ),
            'request_ready_seconds_loss': float(
                ready_seconds_loss.detach().cpu().item()
            ),
            'request_ready_quantile_loss': float(
                ready_quantile_loss.detach().cpu().item()
            ),
            'request_ready_labels': ready_label_count,
            'request_ready_train_labels': ready_train_label_count,
            'request_ready_mae_seconds': ready_mae_seconds,
            'request_ready_blocking_mae_seconds': (
                ready_blocking_mae_seconds
            ),
            'request_ready_underprediction_rate': ready_underprediction_rate,
            '_request_ready_case_stats': ready_case_stats,
            **ready_metric_sums,
            **ready_quantile_metric_sums,
        }
        return (
            (update_info, evaluated_rnn_states)
            if return_rnn_states else update_info
        )

    def device_bc_pretrain(self):
        if getattr(self.all_args, 'stage2_cost_improvement', False) and not getattr(
                self.all_args, 'stage2_policy_improvement_protocol', False):
            raise ValueError('Cost supervision requires the common recurrent N0-N3 protocol.')
        if getattr(self.all_args, 'stage2_policy_improvement_protocol', False):
            values = [self.all_args.stage2_cost_scale_seconds, self.all_args.stage2_cost_weight_clip,
                      self.all_args.stage2_cost_loss_coef, self.all_args.stage2_cost_branch_timeout_seconds]
            if (not np.isfinite(values).all() or min(values) <= 0
                    or self.all_args.stage2_cost_snapshot_horizon != 256
                    or not np.isfinite(self.all_args.stage2_cost_tie_seconds)
                    or self.all_args.stage2_cost_tie_seconds < 0):
                raise ValueError('Invalid preregistered cost protocol constants.')
        if getattr(self.all_args, 'resource_policy', 'heuristic') != 'drl':
            raise RuntimeError("Device BC pretraining requires resource_policy='drl'.")
        if self.checkpoint_dir is None:
            raise RuntimeError("Device BC pretraining requires --checkpoint_dir for the pretrained plane policy.")
        if not hasattr(self.envs, 'call'):
            raise RuntimeError("Device BC pretraining requires vector env call() support.")
        if self.training_stage in {
            CANONICAL_RESOURCE_JOINT, CANONICAL_JOINT_FINETUNE
        }:
            if self.device_bc_min_labels_per_epoch <= 0:
                raise RuntimeError(
                    "resource_joint requires a positive BC label threshold; "
                    "insufficient labels must fail closed."
                )
            if (
                self.device_bc_assignment_loss_coef > 0.0
                and self.device_bc_min_assignment_labels_per_epoch <= 0
            ):
                raise RuntimeError(
                    'final-matching DeviceBC requires a positive assignment '
                    'group threshold.'
                )
            if self.device_bc_max_rollouts_per_epoch <= 0:
                raise RuntimeError(
                    "resource_joint requires a positive BC rollout budget."
                )

        trainable_modules = self._device_bc_trainable_modules()
        if self.device_bc_eval_each_epoch and bool(getattr(self.all_args, 'train_domain_rand', False)):
            raise ValueError('Epoch-boundary BC recovery requires deterministic, non-randomized training cases.')
        trainable_params = [param for module in trainable_modules for param in module.parameters()]
        if not trainable_params:
            raise RuntimeError("Device BC pretraining found no trainable parameters.")

        stage2_expected = None
        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            stage2_expected = (
                self.protected_parameter_summary_before_bc
                or self._protected_resource_joint_summary()
            )
            if stage2_expected is None:
                raise RuntimeError(
                    "resource_joint cannot start resource BC without protected "
                    "Stage-1 M2 parameter evidence."
                )
            self.protected_parameter_summary_before_bc = stage2_expected
            if self.resource_actor_summary_before_bc is None:
                self.resource_actor_summary_before_bc = (
                    self._resource_actor_summary()
                )
            if self.request_ready_predictor_summary_before is None:
                self.request_ready_predictor_summary_before = (
                    self._request_ready_predictor_summary()
                )
            self.resource_joint_phase = 'resource_supervised'
        elif self.training_stage == CANONICAL_JOINT_FINETUNE:
            self.resource_actor_summary_before_bc = self._resource_actor_summary()
            self.resource_joint_phase = 'joint_resource_bc'

        if self.device_bc_training_scope == 'policy_frozen_ready' and self.stage2_frozen_state_before is None:
            self.stage2_frozen_state_before = self._stage2_frozen_state_summary()

        previous_requires_grad = [
            (param, param.requires_grad)
            for param in self.policy.ac.parameters()
        ]
        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            self.policy.set_resource_supervised_training_stage()
        # Apply the exact experiment scope *after* the policy helper.  The
        # historical ordering re-enabled all resource actors and made a
        # nominal head-only experiment consume actor gradients and memory.
        self._set_device_bc_requires_grad(trainable_params)
        bc_lr = self.device_bc_lr if self.device_bc_lr > 0.0 else self.all_args.lr
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=bc_lr,
            eps=self.all_args.opti_eps,
            weight_decay=self.all_args.weight_decay,
        )
        if self.exact_resume_stage2_supervised:
            try:
                optimizer.load_state_dict(
                    self.resource_supervised_optimizer_state
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    'Stage2 supervised recovery optimizer state is '
                    'incompatible with the configured trainable modules.'
                ) from error
        if self.stage2_bc_pending_rng is not None:
            restore_rng_state(self.stage2_bc_pending_rng)
            if 'device_bc_dagger' in self.stage2_bc_pending_rng:
                self.device_bc_dagger_rng.bit_generator.state = copy.deepcopy(
                    self.stage2_bc_pending_rng['device_bc_dagger']
                )
            self.stage2_bc_pending_rng = None
        bc_step = 0
        progress_interval = max(
            10.0,
            min(
                60.0,
                float(getattr(
                    self.all_args, 'status_heartbeat_seconds', 60.0
                )),
            ),
        )

        try:
            self.policy.ac.eval()
            for module in trainable_modules:
                module.train()

            first_bc_epoch = (
                int(self.device_bc_resume_epoch)
                if self.exact_resume_stage2_supervised else 0
            )
            holdout_validation_enabled = self.request_ready_holdout_folds > 1
            total_bc_passes = self.device_bc_pretrain_epochs + int(
                holdout_validation_enabled
            )
            pbar = tqdm(
                range(first_bc_epoch, total_bc_passes),
                desc="DeviceBC",
                unit="epoch",
                total=total_bc_passes,
                initial=first_bc_epoch,
                ncols=160,
            )
            for epoch in pbar:
                cost_iteration = None
                if getattr(self.all_args, 'stage2_policy_improvement_protocol', False):
                    from onpolicy.utils.stage2_cost_improvement import CostIteration
                    # Common to all N0-N3 arms: no dropout/no stale target GRU.
                    self.policy.ac.eval()
                    cost_iteration = CostIteration(self, epoch + 1)
                self.stage2_bc_at_boundary = False
                if self.device_bc_eval_each_epoch:
                    self.envs.call('reset_data_cursor')
                epoch_started_at = time.monotonic()
                holdout_validation_pass = bool(
                    holdout_validation_enabled
                    and epoch >= self.device_bc_pretrain_epochs
                )
                teacher_rate = float(
                    self.device_bc_dagger_schedule[
                        min(epoch, len(self.device_bc_dagger_schedule) - 1)
                    ]
                )
                epoch_info = {
                    'device_bc_optimizer_steps': 0,
                    'device_bc_loss': 0.0,
                    'device_bc_categorical_loss': 0.0,
                    'device_bc_timing_aux_loss': 0.0,
                    'device_bc_ranking_loss': 0.0,
                    'device_bc_ranking_labels': 0,
                    'device_bc_ranking_accuracy': 0.0,
                    'device_bc_assignment_loss': 0.0,
                    'device_bc_assignment_margin_loss': 0.0,
                    'device_bc_assignment_labels': 0,
                    'device_bc_assignment_exact': 0.0,
                    'device_bc_assignment_f1': 0.0,
                    'device_bc_grad_norm': 0.0,
                    'device_bc_labels': 0,
                    'device_bc_ordinary_labels': 0,
                    'device_bc_transporter_labels': 0,
                    'device_bc_ordinary_loss': 0.0,
                    'device_bc_transporter_loss': 0.0,
                    'device_bc_decision_trainable_labels': 0,
                    'device_bc_filtered_ambiguous_labels': 0,
                    'device_bc_blocking_dispatch_labels': 0,
                    'device_bc_lookahead_dispatch_labels': 0,
                    'device_bc_defer_labels': 0,
                    'device_bc_blocking_dispatch_loss': 0.0,
                    'device_bc_lookahead_dispatch_loss': 0.0,
                    'device_bc_defer_loss': 0.0,
                    'request_ready_loss': 0.0,
                    'request_ready_log_loss': 0.0,
                    'request_ready_seconds_loss': 0.0,
                    'request_ready_quantile_loss': 0.0,
                    'request_ready_labels': 0,
                    'request_ready_train_labels': 0,
                    'request_ready_mae_seconds': 0.0,
                    'request_ready_blocking_mae_seconds': 0.0,
                    'request_ready_underprediction_rate': 0.0,
                    'preferred_assignments': 0,
                    'greedy_legal_fills': 0,
                    'real_dispatches': 0,
                    'noop_after_claimed': 0,
                    'deferred_noop': 0,
                    'temporal_defer': 0,
                    'capacity_unmatched': 0,
                    'claimed_noop': 0,
                    'no_demand': 0,
                    'non_unique_candidate_masks': 0,
                    'ordinary_labels': 0,
                    'transporter_labels': 0,
                    'teacher_executed_envs': 0,
                    'student_executed_envs': 0,
                }
                epoch_case_stats = {}
                update_count = 0
                resuming_this_epoch = bool(
                    self.exact_resume_stage2_supervised
                    and epoch == first_bc_epoch
                )
                rollout_count = (
                    int(self.device_bc_resume_completed_rollouts)
                    if resuming_this_epoch else 0
                )
                epoch_label_total = 0
                epoch_ranking_label_total = 0
                epoch_assignment_label_total = 0
                epoch_ready_label_total = 0
                epoch_ready_train_label_total = 0
                min_rollouts, max_rollouts = self._resolve_bc_rollout_limits(
                    self.device_bc_min_rollouts_per_epoch,
                    self.device_bc_max_rollouts_per_epoch,
                    self.num_envs,
                )
                min_labels = max(0, self.device_bc_min_labels_per_epoch)
                if resuming_this_epoch and rollout_count:
                    cursors = self.envs.call(
                        'set_data_cursor', rollout_count
                    )
                    if any(int(value) != rollout_count for value in cursors):
                        raise RuntimeError(
                            'Training workers rejected the supervised Stage2 '
                            f'data cursor {rollout_count}: {cursors!r}.'
                        )
                self._report_progress(
                    'resource_bc_epoch_started',
                    device_bc_epoch=int(epoch + 1),
                    device_bc_total_epochs=int(total_bc_passes),
                    device_bc_rollout=int(rollout_count),
                    device_bc_total_rollouts=int(max_rollouts),
                    device_bc_env_step=0,
                    device_bc_updates=0,
                    device_bc_labels=0,
                )

                while rollout_count < max_rollouts:
                    rollout_started_at = time.monotonic()
                    last_progress_report = rollout_started_at
                    obs, dones, infos = self.envs.reset()
                    rnn_states = np.zeros(
                        (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                        dtype=np.float32,
                    )
                    last_actions = -np.ones((self.n_rollout_threads, self.num_agents, 2), dtype=np.int64)
                    env_done_flags = np.zeros(self.n_rollout_threads, dtype=bool)
                    if cost_iteration is not None:
                        cost_iteration.reset_rollout(rnn_states)

                    for step in range(self.episode_length):
                        active_masks = self._active_masks_from_info(infos)
                        policy_history = self._authoritative_policy_history(
                            infos,
                            last_actions,
                            self.policy.ac.max_plane_agents,
                        )
                        if cost_iteration is not None:
                            rnn_states = cost_iteration.prepare(obs, active_masks,
                                policy_history, infos, rnn_states)
                        dedicated_teacher_forcing = bool(
                            teacher_rate >= 1.0 - 1e-12 and cost_iteration is None
                        )
                        collection_active_masks = active_masks
                        if dedicated_teacher_forcing:
                            # Every live environment will execute the complete
                            # teacher resource action.  The first forward only
                            # needs frozen plane actions/RNN state; resource
                            # RNN state is obtained from the supervised replay
                            # below, before its optimizer step.  This removes a
                            # redundant resource-pointer pass while preserving
                            # the exact recurrent trajectory.
                            collection_active_masks = np.asarray(
                                active_masks
                            ).copy()
                            collection_active_masks[
                                :,
                                self.policy.ac.max_plane_agents:,
                                :,
                            ] = 0.0
                        teacher_method = {
                            'iga': 'resource_iga_teacher_actions',
                            'joint_iga': 'joint_iga_resource_teacher_actions',
                            'heuristic': 'heuristic_device_actions',
                        }[self.device_bc_teacher]
                        teacher_rpc_pending = False
                        teacher_kwargs = {'return_info': True}
                        if getattr(self.all_args, 'device_bc_teacher_deployment_projection', False):
                            teacher_kwargs['deployment_projection'] = True
                        if getattr(self.all_args, 'device_bc_skip_ready_targets', False):
                            # Factual ready timestamps cannot label student-state DAgger.
                            # Resource chromosome actions still use the current live masks.
                            teacher_kwargs['include_ready_targets'] = False
                        if self.safe_dagger_teacher_overlap:
                            self.envs.call_async(
                                teacher_method, **teacher_kwargs
                            )
                            teacher_rpc_pending = True
                        try:
                            with torch.no_grad():
                                (
                                    policy_actions,
                                    next_rnn_states,
                                    encoder_cache,
                                ) = self.policy.get_actor_actions(
                                    Batch.from_data_list(obs),
                                    rnn_states,
                                    collection_active_masks,
                                    policy_history[..., 0],
                                    policy_history[..., 1],
                                    deterministic=(
                                        self.device_bc_plane_deterministic
                                    ),
                                    agent_types=infos.get(
                                        'agent_types', None
                                    ),
                                    return_encoder_cache=True,
                                )
                        except BaseException:
                            if teacher_rpc_pending:
                                try:
                                    self.envs.call_wait()
                                except BaseException:
                                    pass
                            raise
                        policy_actions = _t2n(policy_actions)
                        next_rnn_states = _t2n(next_rnn_states)

                        label_results = (
                            self.envs.call_wait()
                            if teacher_rpc_pending
                            else self.envs.call(
                                teacher_method, **teacher_kwargs
                            )
                        )
                        teacher_execution_mask = (
                            self.device_bc_dagger_rng.random(
                                self.n_rollout_threads
                            ) < teacher_rate
                        )
                        teacher_execution_mask &= ~env_done_flags
                        (
                            actions,
                            label_actions,
                            label_stats,
                            label_metadata,
                        ) = self._merge_device_bc_actions(
                            policy_actions,
                            label_results,
                            teacher_execution_mask,
                            return_metadata=True,
                            active_env_mask=~env_done_flags,
                        )
                        # Resource teachers likewise emit no-op/sentinel rows
                        # for inactive devices.  Restoring the policy's
                        # carried history prevents unrelated events from
                        # erasing a device's recurrent selection context.
                        inactive_agents = active_masks[..., 0] <= 0.0
                        actions[inactive_agents] = policy_actions[
                            inactive_agents
                        ]

                        cost_evidence = (cost_iteration.evidence(obs, active_masks,
                            policy_history, infos, policy_actions, label_actions)
                            if cost_iteration is not None else None)

                        update_result = self._device_bc_update(
                            obs,
                            rnn_states,
                            active_masks,
                            policy_history,
                            label_actions,
                            infos,
                            optimizer,
                            encoder_cache=encoder_cache,
                            return_rnn_states=dedicated_teacher_forcing,
                            label_metadata=label_metadata,
                            ready_env_mask=(
                                np.asarray([
                                    self._request_ready_case_fold(
                                        case_id,
                                        self.request_ready_holdout_folds,
                                    ) == self.request_ready_holdout_fold
                                    for case_id in np.asarray(infos.get(
                                        'case_id',
                                        np.arange(self.n_rollout_threads),
                                    ), dtype=object).reshape(-1)
                                ], dtype=bool)
                                if holdout_validation_pass
                                else np.asarray([
                                    self._request_ready_case_fold(
                                        case_id,
                                        self.request_ready_holdout_folds,
                                    ) != self.request_ready_holdout_fold
                                    for case_id in np.asarray(infos.get(
                                        'case_id',
                                        np.arange(self.n_rollout_threads),
                                    ), dtype=object).reshape(-1)
                                ], dtype=bool)
                                if holdout_validation_enabled
                                else None
                            ),
                            update_parameters=not holdout_validation_pass,
                            cost_evidence=cost_evidence,
                        )
                        if dedicated_teacher_forcing:
                            update_info, resource_rnn_states = update_result
                            if resource_rnn_states is not None:
                                resource_rnn_states = _t2n(
                                    resource_rnn_states
                                )
                                next_rnn_states[
                                    :,
                                    self.policy.ac.max_plane_agents:,
                                ] = resource_rnn_states[
                                    :,
                                    self.policy.ac.max_plane_agents:,
                                ]
                        else:
                            update_info = update_result
                        if update_info is not None:
                            if cost_iteration is not None:
                                cost_iteration.counts['covered_joints'] += int(update_info.get(
                                    'device_bc_cost_covered_joints', 0))
                                cost_iteration.counts['preference_pairs'] += int(update_info.get(
                                    'device_bc_cost_pairs', 0))
                            ready_case_update = update_info.pop(
                                '_request_ready_case_stats', {}
                            )
                            self._merge_request_ready_case_stats(
                                epoch_case_stats, ready_case_update
                            )
                            update_count += 1
                            if (
                                not holdout_validation_pass
                                and int(update_info.get(
                                    'request_ready_train_labels', 0
                                )) > 0
                            ):
                                bc_step += 1
                            epoch_label_total += int(update_info['device_bc_labels'])
                            epoch_ranking_label_total += int(
                                update_info['device_bc_ranking_labels']
                            )
                            epoch_assignment_label_total += int(
                                update_info['device_bc_assignment_labels']
                            )
                            epoch_ready_label_total += int(
                                update_info['request_ready_labels']
                            )
                            epoch_ready_train_label_total += int(
                                update_info.get(
                                    'request_ready_train_labels', 0
                                )
                            )
                            for key, value in update_info.items():
                                epoch_info.setdefault(key, 0.0)
                                epoch_info[key] += value
                            if not self.use_wandb:
                                self.writter.add_scalar('device_bc/loss', update_info['device_bc_loss'], bc_step)
                                self.writter.add_scalar('device_bc/labels', update_info['device_bc_labels'], bc_step)

                        for key, value in label_stats.items():
                            epoch_info[key] = epoch_info.get(key, 0) + value

                        prior_obs, prior_types = obs, infos.get('agent_types')
                        obs, _, dones, infos = self.envs.step(actions)
                        if cost_iteration is not None:
                            cost_iteration.finish_step(prior_obs, active_masks,
                                policy_history, prior_types, dones)
                        next_rnn_states[dones == True] = np.zeros(
                            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                            dtype=np.float32,
                        )
                        rnn_states = next_rnn_states
                        last_actions = actions

                        env_done_flags = np.all(dones, axis=1)
                        now = time.monotonic()
                        if now - last_progress_report >= progress_interval:
                            self._report_progress(
                                'resource_bc_progress',
                                    device_bc_epoch=int(epoch + 1),
                                    device_bc_total_epochs=int(total_bc_passes),
                                device_bc_rollout=int(rollout_count + 1),
                                device_bc_total_rollouts=int(max_rollouts),
                                device_bc_env_step=int(step + 1),
                                device_bc_updates=int(update_count),
                                device_bc_labels=int(epoch_label_total),
                                device_bc_ranking_labels=int(
                                    epoch_ranking_label_total
                                ),
                                device_bc_assignment_labels=int(
                                    epoch_assignment_label_total
                                ),
                                request_ready_labels=int(
                                    epoch_ready_label_total
                                ),
                                device_bc_elapsed_seconds=float(
                                    now - epoch_started_at
                                ),
                            )
                            last_progress_report = now
                        if self.rollout_until_done and np.all(env_done_flags):
                            break

                    if self.rollout_until_done and not np.all(env_done_flags):
                        unfinished_envs = np.where(~env_done_flags)[0]
                        raise RuntimeError(
                            "Device BC rollout did not finish within "
                            f"{self.episode_length} decision steps; unfinished_envs={unfinished_envs.tolist()}."
                        )
                    rollout_count += 1
                    rollout_seconds = max(
                        time.monotonic() - rollout_started_at, 1e-9
                    )
                    self._report_progress(
                        'resource_bc_rollout_completed',
                        device_bc_epoch=int(epoch + 1),
                        device_bc_total_epochs=int(total_bc_passes),
                        device_bc_rollout=int(rollout_count),
                        device_bc_total_rollouts=int(max_rollouts),
                        device_bc_env_step=int(step + 1),
                        device_bc_updates=int(update_count),
                        device_bc_labels=int(epoch_label_total),
                        device_bc_rollout_seconds=float(rollout_seconds),
                        device_bc_elapsed_seconds=float(
                            time.monotonic() - epoch_started_at
                        ),
                    )
                    print(
                        '[DeviceBCTiming] '
                        f'epoch={epoch + 1}/'
                        f'{total_bc_passes}, '
                        f'rollout={rollout_count}/{max_rollouts}, '
                        f'steps={step + 1}, '
                        f'seconds={rollout_seconds:.3f}, '
                        f'labels={epoch_label_total}.',
                        flush=True,
                    )
                    if (
                        rollout_count >= min_rollouts
                        and epoch_label_total >= min_labels
                        and epoch_ranking_label_total
                        >= self.device_bc_min_ranking_labels_per_epoch
                        and epoch_assignment_label_total
                        >= self.device_bc_min_assignment_labels_per_epoch
                        and (
                            epoch_ready_label_total
                            if holdout_validation_pass
                            else epoch_ready_train_label_total
                        )
                        >= self.request_ready_min_labels_per_epoch
                    ):
                        break

                self.device_bc_resume_completed_rollouts = 0

                if self.training_stage in {
                    CANONICAL_RESOURCE_JOINT, CANONICAL_JOINT_FINETUNE
                }:
                    if epoch_label_total < min_labels:
                        raise RuntimeError(
                            "resource_joint resource BC collected insufficient "
                            f"labels in epoch {epoch + 1}: "
                            f"{epoch_label_total} < required {min_labels} "
                            f"after {rollout_count}/{max_rollouts} rollouts."
                        )
                    self.resource_bc_total_labels += int(epoch_label_total)
                    if (
                        epoch_ranking_label_total
                        < self.device_bc_min_ranking_labels_per_epoch
                    ):
                        raise RuntimeError(
                            'resource_joint supervised training collected '
                            'insufficient dense legal-candidate ranking labels '
                            f'in epoch {epoch + 1}: '
                            f'{epoch_ranking_label_total} < required '
                            f'{self.device_bc_min_ranking_labels_per_epoch} '
                            f'after {rollout_count}/{max_rollouts} rollouts. '
                            'Categorical action labels alone are not accepted.'
                        )
                    self.resource_dense_ranking_total_labels += int(
                        epoch_ranking_label_total
                    )
                    if (
                        epoch_assignment_label_total
                        < self.device_bc_min_assignment_labels_per_epoch
                    ):
                        raise RuntimeError(
                            'resource_joint supervised training collected '
                            'insufficient final-matching assignment groups in '
                            f'epoch {epoch + 1}: {epoch_assignment_label_total} '
                            f'< required '
                            f'{self.device_bc_min_assignment_labels_per_epoch}.'
                        )
                    self.resource_assignment_total_labels += int(
                        epoch_assignment_label_total
                    )
                    if (
                        (
                            epoch_ready_label_total
                            if holdout_validation_pass
                            else epoch_ready_train_label_total
                        )
                        < self.request_ready_min_labels_per_epoch
                    ):
                        raise RuntimeError(
                            'resource_joint supervised training collected '
                            'insufficient request ready-time labels in epoch '
                            f'{epoch + 1}: metric={epoch_ready_label_total}, '
                            f'train={epoch_ready_train_label_total} < required '
                            f'{self.request_ready_min_labels_per_epoch} after '
                            f'{rollout_count}/{max_rollouts} rollouts. Regenerate '
                            'the selected IGA/heuristic teacher with '
                            'request_ready_targets.'
                        )
                    if not holdout_validation_pass:
                        self.request_ready_total_labels += int(
                            epoch_ready_train_label_total
                        )

                if update_count > 0:
                    epoch_info['device_bc_loss'] /= update_count
                    epoch_info['device_bc_categorical_loss'] /= update_count
                    epoch_info['device_bc_timing_aux_loss'] /= update_count
                    epoch_info['device_bc_ranking_loss'] /= update_count
                    epoch_info['device_bc_ranking_accuracy'] /= update_count
                    epoch_info['device_bc_assignment_loss'] /= update_count
                    epoch_info[
                        'device_bc_assignment_margin_loss'
                    ] /= update_count
                    epoch_info['device_bc_assignment_exact'] /= update_count
                    epoch_info['device_bc_assignment_f1'] /= update_count
                    epoch_info['device_bc_grad_norm'] /= update_count
                    epoch_info['device_bc_labels'] /= update_count
                    epoch_info['device_bc_ordinary_labels'] /= update_count
                    epoch_info['device_bc_transporter_labels'] /= update_count
                    epoch_info['device_bc_ordinary_loss'] /= update_count
                    epoch_info['device_bc_transporter_loss'] /= update_count
                    epoch_info['device_bc_blocking_dispatch_loss'] /= update_count
                    epoch_info['device_bc_lookahead_dispatch_loss'] /= update_count
                    epoch_info['device_bc_defer_loss'] /= update_count
                    epoch_info['request_ready_loss'] /= update_count
                    epoch_info['request_ready_log_loss'] /= update_count
                    epoch_info['request_ready_seconds_loss'] /= update_count
                    epoch_info['request_ready_quantile_loss'] /= update_count
                    for key in ('device_bc_full_edge_loss', 'device_bc_empty_wait_loss'):
                        epoch_info[key] = epoch_info.get(key, 0.0) / update_count
                if getattr(self, 'device_bc_matching_audit', False):
                    finalize_matching_metrics(epoch_info)
                    epoch_info['device_bc_matching_audit_passed'] = True
                    if (self.device_bc_empty_wait_loss_coef > 0.0
                            and epoch_info.get('device_bc_wait_eligible_groups', 0)
                            < self.device_bc_min_wait_groups_per_epoch):
                        raise RuntimeError('Insufficient typed, legal empty-group wait decisions; this arm cannot test the waiting hypothesis.')
                epoch_info['device_bc_total_labels'] = epoch_label_total
                epoch_info['device_bc_ranking_total_labels'] = (
                    epoch_ranking_label_total
                )
                epoch_info['device_bc_assignment_total_labels'] = (
                    epoch_assignment_label_total
                )
                epoch_info['request_ready_total_labels'] = (
                    epoch_ready_label_total
                )
                epoch_info['request_ready_targets_collected'] = not bool(getattr(
                    self.all_args, 'device_bc_skip_ready_targets', False))
                epoch_info['request_ready_total_train_labels'] = (
                    epoch_ready_train_label_total
                )
                self._finalize_request_ready_epoch_metrics(epoch_info)
                self._finalize_request_ready_quantile_metrics(epoch_info)
                self._finalize_request_ready_case_metrics(
                    epoch_info, epoch_case_stats
                )
                execution_total = (
                    epoch_info['teacher_executed_envs']
                    + epoch_info['student_executed_envs']
                )
                epoch_info['device_bc_teacher_execution_target'] = teacher_rate
                epoch_info['device_bc_teacher_execution_rate'] = (
                    epoch_info['teacher_executed_envs']
                    / max(1, execution_total)
                )
                epoch_info['device_bc_training_scope'] = (
                    self.device_bc_training_scope
                )
                epoch_info['request_ready_feature_weight_norm'] = float(
                    self.policy.ac.request_ready_feature.weight.detach().norm().cpu()
                ) if self.policy.ac.request_ready_prediction else 0.0
                epoch_info['request_ready_phase'] = (
                    'holdout_validation'
                    if holdout_validation_pass else 'train'
                )
                epoch_info['request_ready_pass_index'] = int(epoch + 1)
                epoch_info['request_ready_holdout_folds'] = int(
                    self.request_ready_holdout_folds
                )
                epoch_info['request_ready_holdout_fold'] = int(
                    self.request_ready_holdout_fold
                )
                # The final pass is evaluation-only.  Do not describe it as a
                # fourth optimization epoch in downstream plots/reports.
                epoch_info['device_bc_epoch'] = int(min(
                    epoch + 1, self.device_bc_pretrain_epochs
                ))
                epoch_info['device_bc_updates'] = int(update_count)
                epoch_info['device_bc_rollouts'] = int(rollout_count)
                epoch_info['device_bc_elapsed_seconds'] = float(
                    time.monotonic() - epoch_started_at
                )
                if cost_iteration is not None:
                    cost_iteration.finish_epoch()
                metrics_path = Path(
                    self.log_dir
                ) / 'request_ready_epoch_metrics.jsonl'
                with metrics_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(
                        epoch_info,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ) + '\n')
                if self.device_bc_eval_each_epoch:
                    if epoch_info['device_bc_optimizer_steps'] <= 0:
                        raise RuntimeError('Policy BC epoch completed without any optimizer step.')
                    case_metrics_path = Path(self.log_dir) / f'request_ready_cases_epoch{epoch + 1}.json'
                    case_metrics_path.write_text(json.dumps({
                        'epoch': epoch + 1, 'phase': 'train',
                        'case_statistics': epoch_case_stats,
                    }, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
                    self._evaluate_stage2_bc_epoch(epoch + 1, optimizer)
                if self.device_bc_training_scope == 'ready_only':
                    self._atomic_torch_save(
                        {
                            'schema_version': 1,
                            'training_stage': self.training_stage,
                            'stage2_training_mode': 'ready_predictor_only',
                            'device_bc_epoch': int(epoch + 1),
                            'device_bc_training_scope': (
                                self.device_bc_training_scope
                            ),
                            'request_ready_phase': str(
                                epoch_info['request_ready_phase']
                            ),
                            'request_ready_hard_blocking': bool(
                                self.policy.ac.request_ready_hard_blocking
                            ),
                            'request_ready_context_features': bool(
                                self.policy.ac.request_ready_context_features
                            ),
                            'request_ready_head_mode': str(
                                self.policy.ac.request_ready_head_mode
                            ),
                            'request_ready_quantile_head': bool(
                                self.policy.ac.request_ready_quantile_head
                            ),
                            'request_ready_head': copy.deepcopy(
                                self.policy.ac.request_ready_head.state_dict()
                            ),
                            'epoch_metrics': dict(epoch_info),
                            'resource_actor_summary': (
                                self._resource_actor_summary()
                            ),
                            'protected_parameter_summary': (
                                self._protected_resource_joint_summary()
                            ),
                        },
                        os.path.join(
                            self.save_dir,
                            (
                                'checkpoint_RequestReadyHoldout.pt'
                                if holdout_validation_pass
                                else 'checkpoint_RequestReadyEpoch'
                                f'{epoch + 1}.pt'
                            ),
                        ),
                    )
                if not self.use_wandb:
                    for key, value in epoch_info.items():
                        if isinstance(value, (int, float, np.number)):
                            self.writter.add_scalar(
                                f'device_bc_epoch/{key}', value, epoch + 1
                            )
                pbar.set_postfix(
                    loss=epoch_info['device_bc_loss'],
                    labels=epoch_label_total,
                    ready_labels=epoch_ready_label_total,
                    non_unique=epoch_info['non_unique_candidate_masks'],
                    dispatches=epoch_info['real_dispatches'],
                    rollouts=rollout_count,
                    teacher_rate=epoch_info[
                        'device_bc_teacher_execution_rate'
                    ],
                )
        finally:
            self.resource_supervised_optimizer_state = copy.deepcopy(
                optimizer.state_dict()
            )
            self._restore_requires_grad(previous_requires_grad)
            self.policy.ac.train()

        # Only legacy non-Stage2 callers continue into PPO.  Canonical Stage2
        # terminates here and records the dedicated supervised optimizer state
        # separately instead of manufacturing PPO optimizer evidence.
        if (
            self.training_stage != CANONICAL_RESOURCE_JOINT
            and (
                self.training_stage == CANONICAL_JOINT_FINETUNE
                or self.device_bc_reset_optim
            )
        ):
            self.policy.reset_optimizers()
            self.trainer.policy = self.policy
            self.resource_bc_optimizer_reset = True
            print("[Info] Reset PPO optimizers after device BC pretraining.")
        elif self.training_stage == CANONICAL_RESOURCE_JOINT:
            self.resource_bc_optimizer_reset = False
        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            self.protected_parameter_summary_after_bc = (
                self._assert_resource_joint_protected(
                    stage2_expected,
                    'resource_supervised',
                )
            )
            self.resource_actor_summary_after_bc = self._resource_actor_summary()
            self.request_ready_predictor_summary_after = (
                self._request_ready_predictor_summary()
            )
            if (
                self.resource_actor_summary_before_bc is None
                or self.resource_actor_summary_after_bc is None
            ):
                raise RuntimeError(
                    'resource_joint resource-actor hash evidence is missing.'
                )
            resource_actor_changed = (
                self.resource_actor_summary_before_bc
                != self.resource_actor_summary_after_bc
            )
            if (
                self.device_bc_training_scope == 'ready_only'
                and resource_actor_changed
            ):
                raise RuntimeError(
                    'ready_only Stage2 changed a frozen resource actor.'
                )
            if (
                self.device_bc_training_scope in {'policy_and_ready', 'policy_frozen_ready'}
                and not resource_actor_changed
            ):
                raise RuntimeError(
                    'resource_joint resource BC completed without a bitwise '
                    'resource actor update.'
                )
            self._assert_frozen_ready_unchanged()
            if self.device_bc_training_scope != 'policy_frozen_ready' and (
                self.request_ready_predictor_summary_before is None
                or self.request_ready_predictor_summary_after is None
                or self.request_ready_predictor_summary_before
                == self.request_ready_predictor_summary_after
            ):
                raise RuntimeError(
                    'resource_joint supervised training completed without a '
                    'bitwise request-ready predictor update.'
                )
            self.resource_joint_phase = 'resource_supervised_completed'
            self._report_progress(
                'resource_supervised_completed',
                phase=self.resource_joint_phase,
                resource_bc_total_labels=int(self.resource_bc_total_labels),
                resource_dense_ranking_total_labels=int(
                    self.resource_dense_ranking_total_labels
                ),
                request_ready_total_labels=int(
                    self.request_ready_total_labels
                ),
                device_bc_training_scope=self.device_bc_training_scope,
                resource_actor_changed=bool(resource_actor_changed),
                resource_bc_optimizer_reset=False,
                protected_parameter_summary=(
                    self.protected_parameter_summary_after_bc
                ),
            )
        elif self.training_stage == CANONICAL_JOINT_FINETUNE:
            self.resource_actor_summary_after_bc = (
                self._resource_actor_summary()
            )
            if (
                self.resource_actor_summary_before_bc is None
                or self.resource_actor_summary_after_bc
                == self.resource_actor_summary_before_bc
            ):
                raise RuntimeError(
                    'joint_finetune resource BC completed without a bitwise '
                    'resource-actor update.'
                )
            self.resource_joint_phase = 'joint_bc_completed'
            self._set_joint_finetune_phase(
                'joint_bc_completed',
                event='joint_bc_completed',
                resource_bc_total_labels=int(self.resource_bc_total_labels),
                resource_bc_optimizer_reset=True,
            )
        if (
            self.training_stage != CANONICAL_RESOURCE_JOINT
            and (
            self.bc_reference_kl_coef > 0.0
            or self.bc_reference_target_kl > 0.0
            or any(
                value > 0.0
                for value in self.bc_reference_kl_coef_schedule
            )
            )
        ):
            # Joint-RL callers may regularize against the resource policy
            # produced by their supervised warm-up. Canonical Stage2 never
            # reaches this branch.
            self.policy.capture_bc_reference()
            print('[Info] Captured frozen post-resource-BC reference policy.')
        if (
            self.device_bc_save
            or self.training_stage
            in {CANONICAL_RESOURCE_JOINT, CANONICAL_JOINT_FINETUNE}
        ):
            self.save_device_bc_checkpoint()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, policy_masks \
            = self.trainer.policy.get_actions(
                            Batch.from_data_list(self.buffer.graph_obs[step]),
                            self.buffer.rnn_states[step],
                            self.buffer.active_masks[step],
                            self.buffer.actions[step - 1, ..., 0] if step > 0 else -np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32),
                            self.buffer.actions[step - 1, ..., 1] if step > 0 else -np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32),
                            return_decision_mask=True,
                            )
        # [self.envs, agents, dim]
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        rnn_states = _t2n(rnn_states)
        policy_masks = _t2n(policy_masks).astype(np.float32)[..., None]

        return values, actions, action_log_probs, rnn_states, policy_masks

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            policy_masks,
        ) = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum()), dtype=np.float32)

        active_masks = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        active_masks[infos['active_agents'] == True] = np.ones(((infos['active_agents'] == True).sum()), dtype=np.float32)
        
        self.buffer.graph_insert(
            obs,
            rnn_states,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            active_masks,
            policy_masks=policy_masks,
            decision_times=infos['env_total_time'],
            potential_values=self._graph_potential_values(obs),
        )

    @torch.no_grad()
    def eval(self, render=False, evaluation_label=None):
        if getattr(self, 'shared_eval_client', None) is not None:
            if render:
                raise ValueError('Shared HKBZ validation does not support rendering.')
            previous_tau = float(self.policy.ac.tau)
            self.policy.ac.tau = self.evaluation_tau
            try:
                return self._eval_via_shared_service(evaluation_label)
            finally:
                self.policy.ac.tau = previous_tau

        created_eval_envs = False
        train_clone_threads_suspended = False
        previous_tau = float(self.policy.ac.tau)
        self.policy.ac.tau = self.evaluation_tau
        try:
            if self.eval_envs is None:
                if self.eval_env_factory is None:
                    raise RuntimeError(
                        "Evaluation environment factory is unavailable."
                    )
                suspend_clone_threads = getattr(
                    self.envs, 'suspend_async_graph_cloning', None
                )
                if callable(suspend_clone_threads):
                    train_clone_threads_suspended = bool(
                        suspend_clone_threads()
                    )
                self.eval_envs, self.eval_case_counts = self.eval_env_factory()
                created_eval_envs = True
                self._report_progress(
                    'eval_envs_created',
                    eval_worker_count=self.n_eval_rollout_threads,
                )
            return self._eval_with_envs(
                render=render, evaluation_label=evaluation_label
            )
        finally:
            self.policy.ac.tau = previous_tau
            try:
                if created_eval_envs and self.release_eval_envs_after_eval:
                    self.eval_envs.close()
                    self.eval_envs = None
                    self.eval_case_counts = None
                    self._report_progress(
                        'eval_envs_closed',
                        eval_worker_count=self.n_eval_rollout_threads,
                    )
            finally:
                if train_clone_threads_suspended:
                    self.envs.resume_async_graph_cloning()

    @torch.no_grad()
    def _eval_via_shared_service(self, evaluation_label=None):
        # A shared per-GPU validator must be able to enter CUDA while several
        # trainers retain their long-lived model tensors.  Release only unused
        # allocator blocks at the phase boundary; live parameters, optimizer
        # state, replay data, and numerical results remain untouched.
        runner_device = getattr(self, 'device', None)
        if getattr(runner_device, 'type', None) == 'cuda':
            gib = float(1024 ** 3)
            reserved_before = torch.cuda.memory_reserved(self.device) / gib
            allocated_before = torch.cuda.memory_allocated(self.device) / gib
            gc.collect()
            torch.cuda.empty_cache()
            reserved_after = torch.cuda.memory_reserved(self.device) / gib
            print(
                '[SharedEvalMemory] '
                f'label={str(evaluation_label or "")} '
                f'allocated_gib={allocated_before:.3f} '
                f'reserved_before_gib={reserved_before:.3f} '
                f'reserved_after_gib={reserved_after:.3f}.',
                flush=True,
            )
        request_id = (
            f'{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        )
        request_dir = os.path.join(str(self.run_dir), 'shared_eval_requests')
        os.makedirs(request_dir, exist_ok=True)
        checkpoint_path = os.path.join(request_dir, f'{request_id}.pt')
        observation_metadata = stage1_observation_metadata(getattr(
            self.all_args, 'global_feature_mode', 'none'
        ))
        resource_planning = {
            'resource_release_aware_eta': bool(getattr(
                self.all_args, 'resource_release_aware_eta', False
            )),
            'device_future_intent_horizon': int(getattr(
                self.all_args, 'device_future_intent_horizon', 0
            )),
            'device_future_intent_mode': str(getattr(
                self.all_args, 'device_future_intent_mode', 'legacy_one'
            )),
            'device_frontier_max_requests': int(getattr(
                self.all_args, 'device_frontier_max_requests', 2
            )),
            'device_request_capacity_per_plane': int(getattr(
                self.all_args, 'device_request_capacity_per_plane', 0
            )),
            'device_lookahead_reservation_mode': str(getattr(
                self.all_args,
                'device_lookahead_reservation_mode',
                'none',
            )),
            'device_reservation_grace_seconds': float(getattr(
                self.all_args, 'device_reservation_grace_seconds', 300.0
            )),
        }
        cpu_model = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.policy.ac.state_dict().items()
        }
        model_sha256 = protected_parameter_summary(
            cpu_model, prefixes=('',)
        )['sha256']
        checkpoint = {
            'protocol_version': PROTOCOL_VERSION,
            'request_id': request_id,
            'model': cpu_model,
            'model_sha256': model_sha256,
            'plane_order_mode': self.policy.ac.plane_order_mode,
            'plane_pair_decoder': self.policy.ac.plane_pair_decoder,
            'stage1_baseline': str(getattr(
                self.policy.ac, 'stage1_baseline', 'proposed'
            )),
            'device_policy_head_mode': str(
                self.policy.ac.device_policy_head_mode
            ),
            'ordinary_device_type_count': int(
                self.policy.ac.ordinary_device_type_count
            ),
            'device_timing_head': bool(
                self.policy.ac.device_timing_head
            ),
            'device_global_matching': bool(
                self.policy.ac.device_global_matching
            ),
            'request_ready_prediction': bool(
                self.policy.ac.request_ready_prediction
            ),
            'request_ready_time_scale': float(
                self.policy.ac.request_ready_time_scale
            ),
            'request_ready_policy_injection': str(
                self.policy.ac.request_ready_policy_injection
            ),
            'request_ready_hard_blocking': bool(
                self.policy.ac.request_ready_hard_blocking
            ),
            'request_ready_context_features': bool(
                self.policy.ac.request_ready_context_features
            ),
            'request_ready_head_mode': str(
                self.policy.ac.request_ready_head_mode
            ),
            'request_ready_quantile_head': bool(
                self.policy.ac.request_ready_quantile_head
            ),
            'device_resource_adapter': bool(
                self.policy.ac.device_resource_adapter_enabled
            ),
            'global_feature_mode': str(getattr(
                self.all_args, 'global_feature_mode', 'none'
            )),
            'observation_schema_id': observation_metadata[
                'observation_schema_id'
            ],
            'environment_semantics_version': observation_metadata[
                'environment_semantics_version'
            ],
            'resource_planning_config': resource_planning,
        }
        self._atomic_torch_save(checkpoint, checkpoint_path)
        started = time.monotonic()
        self._report_progress(
            'shared_eval_queued',
            shared_eval_request_id=request_id,
            shared_eval_cpu_set=self.shared_eval_cpu_set,
            evaluation_label=str(evaluation_label or ''),
        )
        try:
            canary_max_cases = (
                self.canary_eval_max_cases
                if str(evaluation_label or '').startswith('canary_')
                else 0
            )
            response = self.shared_eval_client.request({
                'operation': 'evaluate',
                'request_id': request_id,
                'checkpoint_path': str(Path(checkpoint_path).resolve()),
                'evaluation_label': str(evaluation_label or ''),
                'evaluation_tau': float(self.evaluation_tau),
                'seed': int(self.all_args.seed),
                'cpu_set': self.shared_eval_cpu_set,
                'plane_order_mode': self.policy.ac.plane_order_mode,
                'plane_pair_decoder': self.policy.ac.plane_pair_decoder,
                'stage1_baseline': str(getattr(
                    self.policy.ac, 'stage1_baseline', 'proposed'
                )),
                'device_policy_head_mode': str(
                    self.policy.ac.device_policy_head_mode
                ),
                'ordinary_device_type_count': int(
                    self.policy.ac.ordinary_device_type_count
                ),
                'device_timing_head': bool(
                    self.policy.ac.device_timing_head
                ),
                'device_global_matching': bool(
                    self.policy.ac.device_global_matching
                ),
                'request_ready_prediction': bool(
                    self.policy.ac.request_ready_prediction
                ),
                'request_ready_time_scale': float(
                    self.policy.ac.request_ready_time_scale
                ),
                'request_ready_policy_injection': str(
                    self.policy.ac.request_ready_policy_injection
                ),
                'request_ready_hard_blocking': bool(
                    self.policy.ac.request_ready_hard_blocking
                ),
                'request_ready_context_features': bool(
                    self.policy.ac.request_ready_context_features
                ),
                'request_ready_head_mode': str(
                    self.policy.ac.request_ready_head_mode
                ),
                'request_ready_quantile_head': bool(
                    self.policy.ac.request_ready_quantile_head
                ),
                'device_resource_adapter': bool(
                    self.policy.ac.device_resource_adapter_enabled
                ),
                'global_feature_mode': str(getattr(
                    self.all_args, 'global_feature_mode', 'none'
                )),
                'observation_schema_id': observation_metadata[
                    'observation_schema_id'
                ],
                'environment_semantics_version': observation_metadata[
                    'environment_semantics_version'
                ],
                'resource_planning_config': resource_planning,
                'n_eval_rollout_threads': int(self.n_eval_rollout_threads),
                'model_sha256': model_sha256,
                'max_eval_cases': int(canary_max_cases),
            })
            result = self._consume_raw_evaluation(
                response['evaluation'],
                evaluation_label=evaluation_label,
            )
            self._report_progress(
                'shared_eval_completed',
                shared_eval_request_id=request_id,
                shared_eval_elapsed_seconds=float(time.monotonic() - started),
                shared_eval_service_seconds=float(
                    response.get('evaluation_seconds', 0.0)
                ),
                shared_eval_cache_hit=bool(response.get('cache_hit', False)),
                shared_eval_requested_max_cases=int(canary_max_cases),
                evaluation_label=str(evaluation_label or ''),
            )
            return result
        finally:
            try:
                os.unlink(checkpoint_path)
            except FileNotFoundError:
                pass

    @torch.no_grad()
    def _eval_with_envs(
        self,
        render=False,
        evaluation_label=None,
        finalize=True,
    ):
        case_counts = self.eval_case_counts
        if case_counts is None:
            case_counts = [1] * self.n_eval_rollout_threads
        if len(case_counts) != self.n_eval_rollout_threads:
            raise ValueError(
                "Evaluation case counts must match n_eval_rollout_threads: "
                f"counts={len(case_counts)}, threads={self.n_eval_rollout_threads}"
            )

        expected_case_count = int(sum(case_counts))
        eval_rounds = int(max(case_counts))
        all_makespans = []
        all_case_ids = []
        all_eval_records = []
        all_finish_steps = []
        all_relocations = []
        completed_count = 0
        timeout_count = 0
        cycle_count = 0
        max_no_progress = 0
        invalid_checkpoint = False

        # Early-aborted canary evaluation must not rotate the validation split.
        self.eval_envs.call('reset_data_cursor')

        for round_idx in range(eval_rounds):
            valid_ranks = self._evaluation_rank_mask(case_counts, round_idx)
            eval_obs, eval_dones, eval_infos = self.eval_envs.reset()
            round_case_ids = np.asarray(
                eval_infos.get(
                    'case_id',
                    np.array([''] * self.n_eval_rollout_threads, dtype=object),
                ),
                dtype=object,
            ).reshape(-1)

            eval_rnn_states = np.zeros(
                (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
                dtype=np.float32,
            )
            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )

            eval_actions = None
            eval_done_flags = np.zeros(self.n_eval_rollout_threads, dtype=bool)
            round_cycle_flags = np.zeros(self.n_eval_rollout_threads, dtype=bool)
            round_finish_steps = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
            round_relocations = np.zeros(self.n_eval_rollout_threads, dtype=np.float64)
            round_max_no_progress = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
            round_resource_wait = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_critical_wait = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_wait_p95 = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_slack_weighted_wait = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_slack_weighted_wait_p95 = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_avoidable_critical_lateness = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_avoidable_critical_lateness_p95 = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_rendezvous_spread = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_rendezvous_spread_p95 = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_predicted_lateness = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_predicted_lateness_p95 = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_late_dispatch_count = np.zeros(
                self.n_eval_rollout_threads, dtype=np.int32
            )
            round_resource_late_dispatch_rate = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_early_arrival = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_wait_before_dispatch = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_wait_travel = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_wait_post_arrival = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_lookahead_dispatches = np.zeros(
                self.n_eval_rollout_threads, dtype=np.int32
            )
            round_resource_visibility_to_legal = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_legal_to_idle = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_policy_defer = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_policy_defer_p95 = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_policy_defer_max = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_policy_defer_rate = np.zeros(
                self.n_eval_rollout_threads, dtype=np.float64
            )
            round_resource_policy_defer_count = np.zeros(
                self.n_eval_rollout_threads, dtype=np.int32
            )

            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()

                eval_masks = np.ones(
                    (self.n_eval_rollout_threads, self.num_agents, 1),
                    dtype=np.float32,
                )
                eval_masks[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), 1),
                    dtype=np.float32,
                )

                eval_active_masks = np.zeros_like(eval_masks)
                eval_active_masks[eval_infos['active_agents'] == True] = np.ones(
                    ((eval_infos['active_agents'] == True).sum(), 1),
                    dtype=np.float32,
                )

                previous_eval_actions = (
                    eval_actions
                    if eval_actions is not None
                    else -np.ones(
                        (
                            self.n_eval_rollout_threads,
                            self.num_agents,
                            2,
                        ),
                        dtype=np.float32,
                    )
                )
                eval_history = self._authoritative_policy_history(
                    eval_infos,
                    previous_eval_actions,
                    self.policy.ac.max_plane_agents,
                )

                eval_action, eval_rnn_states = self.trainer.policy.act(
                    Batch.from_data_list(eval_obs),
                    eval_rnn_states,
                    eval_active_masks,
                    eval_history[..., 0],
                    eval_history[..., 1],
                    deterministic=True,
                )
                eval_actions = _t2n(eval_action)
                eval_rnn_states = _t2n(eval_rnn_states)

                eval_obs, _, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
                new_done_flags = np.all(eval_dones, axis=1)
                just_finished = new_done_flags & ~eval_done_flags
                round_finish_steps[just_finished] = eval_step + 1
                eval_done_flags |= new_done_flags

                cycle_flags = np.asarray(
                    eval_infos.get(
                        'cycle_terminated',
                        np.zeros(self.n_eval_rollout_threads, dtype=bool),
                    ),
                    dtype=bool,
                ).reshape(-1)
                round_cycle_flags |= cycle_flags
                round_relocations = np.asarray(
                    eval_infos.get(
                        'total_relocations',
                        round_relocations,
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                round_no_progress = np.asarray(
                    eval_infos.get(
                        'max_no_progress',
                        np.zeros(self.n_eval_rollout_threads, dtype=np.int32),
                    ),
                    dtype=np.int32,
                ).reshape(-1)
                round_max_no_progress = np.maximum(round_max_no_progress, round_no_progress)
                for field, target in (
                    ('resource_wait_seconds', round_resource_wait),
                    (
                        'resource_critical_wait_seconds',
                        round_resource_critical_wait,
                    ),
                    ('resource_wait_p95_seconds', round_resource_wait_p95),
                    (
                        'resource_slack_weighted_wait_seconds',
                        round_resource_slack_weighted_wait,
                    ),
                    (
                        'resource_slack_weighted_wait_p95_seconds',
                        round_resource_slack_weighted_wait_p95,
                    ),
                    (
                        'resource_avoidable_critical_lateness_seconds',
                        round_resource_avoidable_critical_lateness,
                    ),
                    (
                        'resource_avoidable_critical_lateness_p95_seconds',
                        round_resource_avoidable_critical_lateness_p95,
                    ),
                    (
                        'resource_rendezvous_spread_seconds',
                        round_resource_rendezvous_spread,
                    ),
                    (
                        'resource_rendezvous_spread_p95_seconds',
                        round_resource_rendezvous_spread_p95,
                    ),
                    (
                        'resource_predicted_lateness_seconds',
                        round_resource_predicted_lateness,
                    ),
                    (
                        'resource_predicted_lateness_p95_seconds',
                        round_resource_predicted_lateness_p95,
                    ),
                    (
                        'resource_late_dispatch_rate',
                        round_resource_late_dispatch_rate,
                    ),
                    (
                        'resource_early_arrival_seconds',
                        round_resource_early_arrival,
                    ),
                    (
                        'resource_wait_before_dispatch_seconds',
                        round_resource_wait_before_dispatch,
                    ),
                    (
                        'resource_wait_travel_seconds',
                        round_resource_wait_travel,
                    ),
                    (
                        'resource_wait_post_arrival_seconds',
                        round_resource_wait_post_arrival,
                    ),
                    (
                        'resource_visibility_to_legal_seconds',
                        round_resource_visibility_to_legal,
                    ),
                    (
                        'resource_legal_to_idle_seconds',
                        round_resource_legal_to_idle,
                    ),
                    (
                        'resource_policy_defer_seconds',
                        round_resource_policy_defer,
                    ),
                    (
                        'resource_policy_defer_p95_seconds',
                        round_resource_policy_defer_p95,
                    ),
                    (
                        'resource_policy_defer_max_seconds',
                        round_resource_policy_defer_max,
                    ),
                    (
                        'resource_policy_defer_rate',
                        round_resource_policy_defer_rate,
                    ),
                ):
                    target[:] = np.asarray(
                        eval_infos.get(field, target), dtype=np.float64
                    ).reshape(-1)
                round_resource_lookahead_dispatches[:] = np.asarray(
                    eval_infos.get(
                        'resource_lookahead_dispatch_count',
                        round_resource_lookahead_dispatches,
                    ),
                    dtype=np.int32,
                ).reshape(-1)
                round_resource_late_dispatch_count[:] = np.asarray(
                    eval_infos.get(
                        'resource_late_dispatch_count',
                        round_resource_late_dispatch_count,
                    ),
                    dtype=np.int32,
                ).reshape(-1)
                round_resource_policy_defer_count[:] = np.asarray(
                    eval_infos.get(
                        'resource_policy_defer_count',
                        round_resource_policy_defer_count,
                    ),
                    dtype=np.int32,
                ).reshape(-1)
                if valid_ranks.any():
                    max_no_progress = max(
                        max_no_progress,
                        int(round_max_no_progress[valid_ranks].max(initial=0)),
                    )
                if self.rollout_until_done and np.all(eval_done_flags):
                    break

            unfinished = ~eval_done_flags
            round_finish_steps[unfinished] = self.episode_length
            valid_cycle = round_cycle_flags[valid_ranks]
            valid_timeout = unfinished[valid_ranks]
            valid_completed = ~(valid_cycle | valid_timeout)
            cycle_count += int(valid_cycle.sum())
            timeout_count += int(valid_timeout.sum())
            completed_count += int(valid_completed.sum())

            if valid_cycle.any():
                cycle_cases = round_case_ids[valid_ranks][valid_cycle].tolist()
                print(
                    "[Warning] Deterministic evaluation terminated cycle cases "
                    f"in round {round_idx + 1}/{eval_rounds}: {cycle_cases}."
                )
                invalid_checkpoint = True
            if valid_timeout.any():
                timeout_cases = round_case_ids[valid_ranks][valid_timeout].tolist()
                print(
                    "[Warning] Deterministic evaluation did not finish within "
                    f"{self.episode_length} steps in round {round_idx + 1}/{eval_rounds}; "
                    f"cases={timeout_cases}."
                )
                invalid_checkpoint = True

            round_makespans = np.asarray(
                self.eval_envs.get_episode_rewards(),
                dtype=np.float64,
            ).reshape(-1)
            all_makespans.extend(round_makespans[valid_ranks].tolist())
            all_finish_steps.extend(round_finish_steps[valid_ranks].tolist())
            all_relocations.extend(round_relocations[valid_ranks].tolist())
            valid_indices = np.flatnonzero(valid_ranks)
            for local_idx, rank_idx in enumerate(valid_indices):
                case_path = str(round_case_ids[rank_idx])
                metadata = self._case_metadata(case_path)
                all_eval_records.append({
                    **metadata,
                    'round_index': int(round_idx),
                    'rank_index': int(rank_idx),
                    'makespan': float(round_makespans[rank_idx]),
                    'finish_steps': int(round_finish_steps[rank_idx]),
                    'total_relocations': float(round_relocations[rank_idx]),
                    'max_no_progress': int(round_max_no_progress[rank_idx]),
                    'resource_wait_seconds': float(
                        round_resource_wait[rank_idx]
                    ),
                    'resource_critical_wait_seconds': float(
                        round_resource_critical_wait[rank_idx]
                    ),
                    'resource_wait_p95_seconds': float(
                        round_resource_wait_p95[rank_idx]
                    ),
                    'resource_slack_weighted_wait_seconds': float(
                        round_resource_slack_weighted_wait[rank_idx]
                    ),
                    'resource_slack_weighted_wait_p95_seconds': float(
                        round_resource_slack_weighted_wait_p95[rank_idx]
                    ),
                    'resource_avoidable_critical_lateness_seconds': float(
                        round_resource_avoidable_critical_lateness[rank_idx]
                    ),
                    'resource_avoidable_critical_lateness_p95_seconds': float(
                        round_resource_avoidable_critical_lateness_p95[
                            rank_idx
                        ]
                    ),
                    'resource_rendezvous_spread_seconds': float(
                        round_resource_rendezvous_spread[rank_idx]
                    ),
                    'resource_rendezvous_spread_p95_seconds': float(
                        round_resource_rendezvous_spread_p95[rank_idx]
                    ),
                    'resource_predicted_lateness_seconds': float(
                        round_resource_predicted_lateness[rank_idx]
                    ),
                    'resource_predicted_lateness_p95_seconds': float(
                        round_resource_predicted_lateness_p95[rank_idx]
                    ),
                    'resource_late_dispatch_count': int(
                        round_resource_late_dispatch_count[rank_idx]
                    ),
                    'resource_late_dispatch_rate': float(
                        round_resource_late_dispatch_rate[rank_idx]
                    ),
                    'resource_early_arrival_seconds': float(
                        round_resource_early_arrival[rank_idx]
                    ),
                    'resource_wait_before_dispatch_seconds': float(
                        round_resource_wait_before_dispatch[rank_idx]
                    ),
                    'resource_wait_travel_seconds': float(
                        round_resource_wait_travel[rank_idx]
                    ),
                    'resource_wait_post_arrival_seconds': float(
                        round_resource_wait_post_arrival[rank_idx]
                    ),
                    'resource_lookahead_dispatch_count': int(
                        round_resource_lookahead_dispatches[rank_idx]
                    ),
                    'resource_visibility_to_legal_seconds': float(
                        round_resource_visibility_to_legal[rank_idx]
                    ),
                    'resource_legal_to_idle_seconds': float(
                        round_resource_legal_to_idle[rank_idx]
                    ),
                    'resource_policy_defer_seconds': float(
                        round_resource_policy_defer[rank_idx]
                    ),
                    'resource_policy_defer_p95_seconds': float(
                        round_resource_policy_defer_p95[rank_idx]
                    ),
                    'resource_policy_defer_max_seconds': float(
                        round_resource_policy_defer_max[rank_idx]
                    ),
                    'resource_policy_defer_count': int(
                        round_resource_policy_defer_count[rank_idx]
                    ),
                    'resource_policy_defer_rate': float(
                        round_resource_policy_defer_rate[rank_idx]
                    ),
                    'completed': bool(valid_completed[local_idx]),
                    'cycle_terminated': bool(valid_cycle[local_idx]),
                    'timeout': bool(valid_timeout[local_idx]),
                })
            all_case_ids.extend(
                str(case_id)
                for case_id in round_case_ids[valid_ranks]
                if str(case_id)
            )

            if invalid_checkpoint and round_idx + 1 >= self.eval_canary_rounds:
                print(
                    "[Warning] Stopping full validation after the canary rounds "
                    "because the checkpoint is already invalid."
                )
                break

        evaluated_case_count = len(all_makespans)
        if not invalid_checkpoint and evaluated_case_count != expected_case_count:
            raise RuntimeError(
                "Evaluation coverage mismatch: "
                f"expected={expected_case_count}, observed={evaluated_case_count}"
            )
        if len(all_case_ids) == evaluated_case_count and len(set(all_case_ids)) != evaluated_case_count:
            raise RuntimeError("Evaluation visited duplicate cases before covering the split.")

        self.last_eval_case_count = evaluated_case_count
        self.last_eval_case_ids = all_case_ids
        self.last_eval_completed_count = completed_count
        self.last_eval_completion_rate = (
            float(completed_count) / max(1, evaluated_case_count)
        )
        self.last_eval_timeout_count = timeout_count
        self.last_eval_cycle_count = cycle_count
        self.last_eval_mean_steps = (
            float(np.mean(all_finish_steps)) if all_finish_steps else 0.0
        )
        self.last_eval_max_no_progress = max_no_progress
        self.last_eval_mean_relocations = (
            float(np.mean(all_relocations)) if all_relocations else 0.0
        )

        self.last_eval_records = all_eval_records
        if invalid_checkpoint or self.last_eval_completion_rate < 1.0:
            eval_makespan = np.inf
        else:
            eval_makespan = float(np.mean(all_makespans))

        if not finalize:
            return self._raw_evaluation_payload(eval_makespan)
        return self._finalize_evaluation(
            eval_makespan,
            evaluation_label=evaluation_label,
        )

    def _raw_evaluation_payload(self, eval_makespan):
        return {
            'raw_makespan': float(eval_makespan),
            'case_count': int(self.last_eval_case_count),
            'case_ids': list(self.last_eval_case_ids),
            'completed_count': int(self.last_eval_completed_count),
            'completion_rate': float(self.last_eval_completion_rate),
            'timeout_count': int(self.last_eval_timeout_count),
            'cycle_count': int(self.last_eval_cycle_count),
            'mean_steps': float(self.last_eval_mean_steps),
            'max_no_progress': int(self.last_eval_max_no_progress),
            'mean_relocations': float(self.last_eval_mean_relocations),
            'records': copy.deepcopy(self.last_eval_records),
        }

    def _consume_raw_evaluation(self, payload, evaluation_label=None):
        required = {
            'raw_makespan', 'case_count', 'case_ids', 'completed_count',
            'completion_rate', 'timeout_count', 'cycle_count', 'mean_steps',
            'max_no_progress', 'mean_relocations', 'records',
        }
        missing = sorted(required - set(payload))
        if missing:
            raise RuntimeError(
                f'Shared evaluator response is missing fields: {missing}'
            )
        records = copy.deepcopy(list(payload['records']))
        case_ids = [str(case_id) for case_id in payload['case_ids']]
        case_count = int(payload['case_count'])
        if case_count != len(records) or case_count != len(case_ids):
            raise RuntimeError(
                'Shared evaluator coverage mismatch: '
                f'count={case_count}, records={len(records)}, '
                f'case_ids={len(case_ids)}'
            )
        if len(set(case_ids)) != len(case_ids):
            raise RuntimeError('Shared evaluator returned duplicate case IDs.')
        if getattr(self, 'selection_metric', 'iid') in {
            'composite', 'composite_tail'
        }:
            lineage_fields = ('case_key', 'profile', 'distribution', 'case_sha256')
            missing_lineage = {
                field_name: [
                    str(record.get('case_id', record_index))
                    for record_index, record in enumerate(records)
                    if not str(record.get(field_name, '') or '').strip()
                ]
                for field_name in lineage_fields
            }
            missing_lineage = {
                field_name: case_names
                for field_name, case_names in missing_lineage.items()
                if case_names
            }
            if missing_lineage:
                summary = {
                    field_name: {
                        'count': len(case_names),
                        'examples': case_names[:3],
                    }
                    for field_name, case_names in missing_lineage.items()
                }
                raise RuntimeError(
                    'Shared evaluator response is missing immutable composite '
                    f'case metadata: {summary}'
                )
        self.last_eval_case_count = case_count
        self.last_eval_case_ids = case_ids
        self.last_eval_completed_count = int(payload['completed_count'])
        self.last_eval_completion_rate = float(payload['completion_rate'])
        self.last_eval_timeout_count = int(payload['timeout_count'])
        self.last_eval_cycle_count = int(payload['cycle_count'])
        self.last_eval_mean_steps = float(payload['mean_steps'])
        self.last_eval_max_no_progress = int(payload['max_no_progress'])
        self.last_eval_mean_relocations = float(payload['mean_relocations'])
        self.last_eval_records = records
        return self._finalize_evaluation(
            float(payload['raw_makespan']),
            evaluation_label=evaluation_label,
        )

    def _finalize_evaluation(self, eval_makespan, evaluation_label=None):

        if str(evaluation_label) in {'pre_ppo', 'pre_supervised'}:
            self.pre_ppo_case_makespan = {
                record['case_key']: float(record['makespan'])
                for record in self.last_eval_records
                if np.isfinite(record.get('makespan', np.inf))
            }
        for record in self.last_eval_records:
            baseline = self.pre_ppo_case_makespan.get(record['case_key'])
            if baseline is not None and np.isfinite(baseline):
                record['pre_ppo_makespan'] = float(baseline)
                record['delta_vs_pre_ppo'] = float(
                    record['makespan'] - baseline
                )
                record['relative_delta_vs_pre_ppo'] = float(
                    record['makespan'] / max(baseline, 1e-12) - 1.0
                )
            else:
                record['pre_ppo_makespan'] = None
                record['delta_vs_pre_ppo'] = None
                record['relative_delta_vs_pre_ppo'] = None

        selection_score = self._selection_metrics_from_records(
            eval_makespan,
            evaluation_label=evaluation_label,
        )
        self._write_evaluation_records(evaluation_label, selection_score)
        return selection_score

    # TODO: add render function for FarmEnv
    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            active_masks = np.ones_like(masks)
            available_actions = np.ones((self.n_rollout_threads, self.num_agents, self.envs.action_space[0].n),dtype=np.float32)


            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                graph_obs, vector_obs = obs['graph'], obs['vector']
                action, rnn_states = self.trainer.policy.act(
                                                Batch.from_data_list(graph_obs),
                                                np.concatenate(vector_obs),
                                                np.concatenate(rnn_states),
                                                np.concatenate(masks),
                                                np.concatenate(active_masks),
                                                np.concatenate(available_actions),
                                                deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

    @staticmethod
    def _atomic_torch_save(checkpoint, save_path):
        temporary_path = f"{save_path}.tmp.{os.getpid()}"
        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, save_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def save(self, episode=0, filename=None, extra=None):
        """Atomically save policy, optimizer, and value-normalizer state."""
        model = self.trainer.policy
        if filename is None:
            filename = 'checkpoint_Epoch' + str(episode+1) + '.pt'
        save_path = os.path.join(self.save_dir, filename)
        observation_metadata = stage1_observation_metadata(getattr(
            self.all_args, 'global_feature_mode', 'none'
        ))
        checkpoint = {
            **observation_metadata,
            'episodes': episode + 1,
            'tau': model.ac.tau,
            'training_stage': self.training_stage,
            'phase': self.resource_joint_phase,
            'stage2_training_mode': (
                (
                    'ready_predictor_only'
                    if self.device_bc_training_scope == 'ready_only'
                    else 'supervised_only'
                )
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'device_bc_training_scope': self.device_bc_training_scope,
            'stage2_supervision_contract': (
                dict(STAGE2_SUPERVISION_CONTRACT)
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'source_m2_checkpoint': dict(self.stage1_m2_source or {}),
            'source_m2_path': str((self.stage1_m2_source or {}).get('path', '')),
            'source_m2_sha256': str(
                (self.stage1_m2_source or {}).get('sha256', '')
            ),
            'stage1_m2_checkpoint_summary': self.stage1_m2_checkpoint_summary,
            'source_stage2_checkpoint': dict(self.stage2_source or {}),
            'source_stage2_path': str(
                (self.stage2_source or {}).get('path', '')
            ),
            'source_stage2_sha256': str(
                (self.stage2_source or {}).get('sha256', '')
            ),
            'stage2_checkpoint_summary': self.stage2_checkpoint_summary,
            'protected_parameter_summary': (
                self._protected_resource_joint_summary()
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'protected_parameter_summary_before_bc': (
                self.protected_parameter_summary_before_bc
            ),
            'protected_parameter_summary_after_bc': (
                self.protected_parameter_summary_after_bc
            ),
            'protected_parameter_summary_after_ppo': (
                self.protected_parameter_summary_after_ppo
            ),
            'resource_bc_optimizer_reset': bool(
                self.resource_bc_optimizer_reset
            ),
            'resource_bc_total_labels': int(self.resource_bc_total_labels),
            'resource_dense_ranking_total_labels': int(
                self.resource_dense_ranking_total_labels
            ),
            'resource_assignment_total_labels': int(
                self.resource_assignment_total_labels
            ),
            'request_ready_total_labels': int(
                self.request_ready_total_labels
            ),
            'request_ready_predictor_summary_before': (
                self.request_ready_predictor_summary_before
            ),
            'request_ready_predictor_summary_after': (
                self.request_ready_predictor_summary_after
            ),
            'resource_bc_checkpoint_source': self.resource_bc_checkpoint_source,
            'device_bc_teacher': self.device_bc_teacher,
            'device_bc_role_balanced': bool(self.device_bc_role_balanced),
            'device_bc_timing_balanced': bool(
                self.device_bc_timing_balanced
            ),
            'device_bc_legacy_noop_timing': bool(
                self.device_bc_legacy_noop_timing
            ),
            'device_bc_min_teacher_score_margin': float(
                self.device_bc_min_teacher_score_margin
            ),
            'device_bc_ranking_loss_coef': float(
                self.device_bc_ranking_loss_coef
            ),
            'device_bc_categorical_loss_coef': float(
                self.device_bc_categorical_loss_coef
            ),
            'device_bc_assignment_loss_coef': float(
                self.device_bc_assignment_loss_coef
            ),
            'device_bc_assignment_margin_loss_coef': float(
                self.device_bc_assignment_margin_loss_coef
            ),
            'device_bc_assignment_margin': float(
                self.device_bc_assignment_margin
            ),
            'device_bc_min_assignment_labels_per_epoch': int(
                self.device_bc_min_assignment_labels_per_epoch
            ),
            'device_bc_min_ranking_labels_per_epoch': int(
                self.device_bc_min_ranking_labels_per_epoch
            ),
            'device_bc_timing_loss_coef': float(
                self.device_bc_timing_loss_coef
            ),
            'device_bc_dagger_schedule': list(
                self.device_bc_dagger_schedule
            ),
            'resource_lookahead_contract': self._resource_lookahead_contract(),
            'resource_iga_teacher_dir': str(getattr(
                self.all_args, 'resource_iga_teacher_dir', ''
            )),
            'resource_iga_teacher_index': str(getattr(
                self.all_args, 'resource_iga_teacher_index', ''
            )),
            'joint_iga_teacher_dir': str(getattr(
                self.all_args, 'joint_iga_teacher_dir', ''
            )),
            'joint_iga_teacher_index': str(getattr(
                self.all_args, 'joint_iga_teacher_index', ''
            )),
            'resource_ppo_update_schedule': self.resource_ppo_update_schedule,
            'resource_ppo_warmup_epochs': int(
                self.resource_ppo_warmup_epochs
            ),
            'resource_wait_dual_value': float(
                self.resource_wait_dual_value
            ),
            'skip_pre_ppo_eval': bool(self.skip_pre_ppo_eval),
            'resource_actor_summary_before_ppo': (
                self.resource_actor_summary_before_ppo
            ),
            'resource_actor_summary_after_ppo': (
                self.resource_actor_summary_after_ppo
            ),
            'resource_actor_summary_before_bc': (
                self.resource_actor_summary_before_bc
            ),
            'resource_actor_summary_after_bc': (
                self.resource_actor_summary_after_bc
            ),
            'plane_actor_summary_before_bc': self.plane_actor_summary_before_bc,
            'plane_actor_summary_after_bc': self.plane_actor_summary_after_bc,
            'plane_actor_summary_before_ppo': (
                self.plane_actor_summary_before_ppo
            ),
            'plane_actor_summary_after_ppo': (
                self.plane_actor_summary_after_ppo
            ),
            'shared_actor_summary_before_ppo': (
                self.shared_actor_summary_before_ppo
            ),
            'shared_actor_summary_after_ppo': (
                self.shared_actor_summary_after_ppo
            ),
            'plane_order_mode': model.ac.plane_order_mode,
            'plane_pair_decoder': model.ac.plane_pair_decoder,
            'stage1_baseline': model.ac.stage1_baseline,
            'actor_lr_multiplier': float(model.actor_lr_multiplier),
            'actor_lr_decay_factor': float(model.actor_lr_decay_factor),
            'experiment_config': {
                'stage1_baseline': model.ac.stage1_baseline,
                'hindsight_reward_mode': str(
                    getattr(self.all_args, 'hindsight_reward_mode', '')
                ),
                'lr': float(self.all_args.lr),
                'critic_lr': float(self.all_args.critic_lr),
                'shared_actor_lr_scale': float(
                    self.all_args.shared_actor_lr_scale
                ),
                'shared_actor_lr_scale_schedule': list(
                    self.shared_actor_lr_scale_schedule
                ),
                'shared_actor_lr_max_multiplier': float(getattr(
                    self.all_args, 'shared_actor_lr_max_multiplier', 0.0
                )),
                'ppo_epoch': int(self.all_args.ppo_epoch),
                'actor_grad_accumulation_steps': int(
                    self.all_args.actor_grad_accumulation_steps
                ),
                'actor_grad_accumulation_target_graphs': int(getattr(
                    self.all_args,
                    'actor_grad_accumulation_target_graphs',
                    0,
                )),
                'grad_accumulation_target_graphs': int(getattr(
                    self.all_args,
                    'grad_accumulation_target_graphs',
                    0,
                )),
                'entropy_coef': float(self.all_args.entropy_coef),
                'max_grad_norm': float(self.all_args.max_grad_norm),
                'actor_grad_clip_mode': str(getattr(
                    self.all_args, 'actor_grad_clip_mode', 'global'
                )),
                'actor_group_max_grad_norm': dict(
                    self.trainer.actor_group_max_grad_norm
                ),
                'bc_reference_kl_coef': float(
                    self.bc_reference_kl_coef
                ),
                'bc_reference_checkpoint': str(
                    self.bc_reference_checkpoint
                ),
                'bc_reference_resolved_path': str(
                    self.bc_reference_resolved_path
                ),
                'bc_reference_kl_coef_schedule': list(
                    self.bc_reference_kl_coef_schedule
                ),
                'bc_reference_target_kl': float(
                    self.bc_reference_target_kl
                ),
                'bc_reference_hard_gate': bool(
                    self.bc_reference_hard_gate
                ),
                'adaptive_bc_reference_kl': bool(
                    self.adaptive_bc_reference_kl
                ),
                'actor_kl_backtrack': bool(getattr(
                    self.all_args, 'actor_kl_backtrack', False
                )),
                'actor_kl_backtrack_scales': str(getattr(
                    self.all_args,
                    'actor_kl_backtrack_scales',
                    '0.5,0.25,0.125',
                )),
                'paired_case_baseline_scope': str(
                    self.paired_case_baseline_scope
                ),
                'reset_value_normalizer_before_ppo': bool(
                    self.reset_value_normalizer_before_ppo
                ),
                'adaptive_actor_kl': bool(self.adaptive_actor_kl),
                'adaptive_actor_kl_low': float(
                    self.adaptive_actor_kl_low
                ),
                'adaptive_actor_kl_high': float(
                    self.adaptive_actor_kl_high
                ),
                'adaptive_actor_lr_max_scale': float(
                    self.adaptive_actor_lr_max_scale
                ),
                'adaptive_actor_min_step_completion': float(
                    self.adaptive_actor_min_step_completion
                ),
                'actor_lr_multiplier': float(
                    self.policy.actor_lr_multiplier
                ),
                'evaluation_tau': float(self.evaluation_tau),
                'selection_metric': self.selection_metric,
                'selection_weights': dict(self.selection_weights),
                'selection_tail_fraction': float(
                    self.selection_tail_fraction
                ),
                'selection_tail_weight': float(self.selection_tail_weight),
                'global_feature_mode': str(getattr(
                    self.all_args, 'global_feature_mode', 'none'
                )),
                'observation_schema_id': observation_metadata[
                    'observation_schema_id'
                ],
                'environment_semantics_version': observation_metadata[
                    'environment_semantics_version'
                ],
                'device_lookahead_dispatch': bool(getattr(
                    self.all_args, 'device_lookahead_dispatch', False
                )),
                'device_deadline_aware_dispatch': bool(getattr(
                    self.all_args, 'device_deadline_aware_dispatch', False
                )),
                'device_future_intent_horizon': int(getattr(
                    self.all_args, 'device_future_intent_horizon', 0
                )),
                'device_future_intent_mode': str(getattr(
                    self.all_args, 'device_future_intent_mode', 'legacy_one'
                )),
                'device_frontier_max_requests': int(getattr(
                    self.all_args, 'device_frontier_max_requests', 2
                )),
                'resource_release_aware_eta': bool(getattr(
                    self.all_args, 'resource_release_aware_eta', False
                )),
                'device_lookahead_reservation_mode': str(getattr(
                    self.all_args,
                    'device_lookahead_reservation_mode',
                    'none',
                )),
                'device_reservation_grace_seconds': float(getattr(
                    self.all_args, 'device_reservation_grace_seconds', 300.0
                )),
                'device_departure_lookahead': bool(getattr(
                    self.all_args, 'device_departure_lookahead', False
                )),
                'resource_lateness_coef': float(getattr(
                    self.all_args, 'resource_lateness_coef', 0.0
                )),
                'resource_critical_lateness_coef': float(getattr(
                    self.all_args,
                    'resource_critical_lateness_coef',
                    0.0,
                )),
                'resource_earliness_coef': float(getattr(
                    self.all_args, 'resource_earliness_coef', 0.0
                )),
                'resource_wait_constraint_target': float(
                    self.resource_wait_constraint_target
                ),
                'resource_wait_dual_lr': float(self.resource_wait_dual_lr),
                'resource_wait_dual_max': float(self.resource_wait_dual_max),
                'resource_wait_dual_value': float(
                    self.resource_wait_dual_value
                ),
                'stage2_allow_shared_unfreeze': bool(
                    self.stage2_allow_shared_unfreeze
                ),
                'resource_slack_criticality_seconds': float(getattr(
                    self.all_args,
                    'resource_slack_criticality_seconds',
                    1800.0,
                )),
                'resource_slack_forecast_seconds': float(getattr(
                    self.all_args,
                    'resource_slack_forecast_seconds',
                    0.0,
                )),
                'stage1_reward_contract': (
                    dict(self.stage1_reward_contract_metadata)
                    if self.stage1_reward_contract_metadata is not None
                    else None
                ),
                'reset_value_normalizer_on_resume': bool(
                    self.reset_value_normalizer_on_resume
                ),
                'gnn_freeze_epochs': int(self.gnn_freeze_epochs),
                'plane_order_freeze_epochs': int(
                    self.plane_order_freeze_epochs
                ),
                'joint_team_ppo': bool(
                    getattr(self.all_args, 'joint_team_ppo', False)
                ),
                'joint_team_ppo_scope': str(getattr(
                    self.all_args, 'joint_team_ppo_scope', 'plane'
                )),
                'role_atomic_ppo': bool(getattr(
                    self.all_args, 'role_atomic_ppo', False
                )),
                'role_event_returns': bool(getattr(
                    self.all_args, 'role_event_returns', False
                )),
                'role_event_credit_mode': self.role_event_credit_mode,
                'role_event_credit_uniform_mix': float(
                    self.role_event_credit_uniform_mix
                ),
                'counterfactual_q_baseline': bool(getattr(
                    self.all_args, 'counterfactual_q_baseline', False
                )),
                'counterfactual_baseline_mix': float(
                    self.current_counterfactual_baseline_mix
                ),
                'counterfactual_baseline_mix_schedule': list(
                    self.counterfactual_baseline_mix_schedule
                ),
                'role_event_gae_lambda': float(getattr(
                    self.all_args, 'role_event_gae_lambda', 1.0
                )),
                'role_event_gae_lambdas': {
                    role_name: float(value)
                    for role_name, value in zip(
                        ('plane', 'device', 'transporter'),
                        getattr(
                            self.trainer,
                            'role_event_gae_lambdas',
                            {0: 1.0, 1: 1.0, 2: 1.0},
                        ).values(),
                    )
                },
                'role_loss_weighting': str(getattr(
                    self.all_args, 'role_loss_weighting', 'fixed'
                )),
                'role_loss_min_share': float(getattr(
                    self.all_args, 'role_loss_min_share', 0.15
                )),
                'role_loss_max_share': float(getattr(
                    self.all_args, 'role_loss_max_share', 0.60
                )),
                'role_valuenorm': bool(getattr(
                    self.all_args, 'role_valuenorm', False
                )),
                'shared_gradient_diagnostics': bool(getattr(
                    self.all_args, 'shared_gradient_diagnostics', False
                )),
                'shared_encoder_pcgrad': bool(getattr(
                    self.all_args, 'shared_encoder_pcgrad', False
                )),
                'shared_gradient_method': str(getattr(
                    self.all_args, 'shared_gradient_method', 'sum'
                )),
                'shared_grad_ema_beta': float(getattr(
                    self.all_args, 'shared_grad_ema_beta', 0.97
                )),
                'shared_grad_norm_power': float(getattr(
                    self.all_args, 'shared_grad_norm_power', 0.5
                )),
                'shared_grad_min_scale': float(getattr(
                    self.all_args, 'shared_grad_min_scale', 0.5
                )),
                'shared_grad_max_scale': float(getattr(
                    self.all_args, 'shared_grad_max_scale', 2.0
                )),
                'shared_grad_conflict_threshold': float(getattr(
                    self.all_args, 'shared_grad_conflict_threshold', -0.05
                )),
                'shared_cagrad_c': float(getattr(
                    self.all_args, 'shared_cagrad_c', 0.2
                )),
                'shared_encoder_activation_checkpoint': bool(getattr(
                    self.all_args,
                    'shared_encoder_activation_checkpoint',
                    False,
                )),
                'stage3_allow_shared_frozen': bool(
                    self.stage3_allow_shared_frozen
                ),
                'role_target_kl': {
                    'plane': float(getattr(
                        self.all_args, 'plane_target_kl', 0.0025
                    )),
                    'device': float(getattr(
                        self.all_args, 'device_target_kl', 0.005
                    )),
                    'transporter': float(getattr(
                        self.all_args, 'transporter_target_kl', 0.005
                    )),
                },
                'central_team_critic': bool(
                    getattr(self.all_args, 'central_team_critic', False)
                ),
                'safe_async_graph_clone_workers': int(
                    self.safe_async_graph_clone_workers
                ),
                'safe_graph_batch_pipeline': bool(
                    self.safe_graph_batch_pipeline
                ),
                'safe_dagger_teacher_overlap': bool(
                    self.safe_dagger_teacher_overlap
                ),
                'plane_bc_pretrain_epochs': int(
                    self.plane_bc_pretrain_epochs
                ),
                'iga_potential_weights_path': str(
                    getattr(self.all_args, 'iga_potential_weights_path', '')
                ),
                'iga_potential_beta': float(
                    getattr(self.all_args, 'iga_potential_beta', 0.0)
                ),
                'iga_potential_beta_schedule': list(
                    self.iga_potential_beta_schedule
                ),
                'iga_potential_gamma': float(
                    getattr(self.all_args, 'iga_potential_gamma', 0.99)
                ),
                'tail_policy_start_fraction': float(getattr(
                    self.all_args, 'tail_policy_start_fraction', 1.0
                )),
                'tail_policy_weight': float(getattr(
                    self.all_args, 'tail_policy_weight', 1.0
                )),
                'plane_bc_shared_lr_scale': float(
                    self.plane_bc_shared_lr_scale
                ),
                'plane_bc_freeze_shared_epochs': int(
                    self.plane_bc_freeze_shared_epochs
                ),
                'plane_bc_pair_loss_coef': float(
                    self.plane_bc_pair_loss_coef
                ),
                'plane_bc_order_loss_coef': float(
                    self.plane_bc_order_loss_coef
                ),
                'plane_bc_initial_weight': float(
                    self.plane_bc_initial_weight
                ),
                'plane_bc_relocation_weight': float(
                    self.plane_bc_relocation_weight
                ),
                'plane_bc_critical_op_weight': float(
                    self.plane_bc_critical_op_weight
                ),
                'plane_bc_tail_start_fraction': float(
                    self.plane_bc_tail_start_fraction
                ),
                'plane_bc_tail_weight': float(
                    self.plane_bc_tail_weight
                ),
                'plane_bc_tail_final_start_fraction': float(
                    self.plane_bc_tail_final_start_fraction
                ),
                'plane_bc_tail_final_weight': float(
                    self.plane_bc_tail_final_weight
                ),
                'plane_bc_dagger_schedule': list(
                    self.plane_bc_dagger_schedule
                ),
                'plane_bc_dagger_tail_start_fraction': float(
                    self.plane_bc_dagger_tail_start_fraction
                ),
                'plane_bc_dagger_tail_teacher_rate': float(
                    self.plane_bc_dagger_tail_teacher_rate
                ),
                'train_sampling_mode': str(getattr(
                    self.all_args, 'train_sampling_mode', 'uniform'
                )),
                'train_sampling_weights': str(getattr(
                    self.all_args, 'train_sampling_weights', ''
                )),
                'train_sampling_size': int(getattr(
                    self.all_args, 'train_sampling_size', 0
                )),
                'train_sampling_pool_size': int(getattr(
                    self.all_args, 'train_sampling_pool_size', 0
                )),
            },
            'device_policy_head_mode': str(
                model.ac.device_policy_head_mode
            ),
            'ordinary_device_type_count': int(
                model.ac.ordinary_device_type_count
            ),
            'device_timing_head': bool(model.ac.device_timing_head),
            'device_global_matching': bool(
                model.ac.device_global_matching
            ),
            'request_ready_prediction': bool(
                model.ac.request_ready_prediction
            ),
            'request_ready_time_scale': float(
                model.ac.request_ready_time_scale
            ),
            'request_ready_policy_injection': str(
                model.ac.request_ready_policy_injection
            ),
            'request_ready_hard_blocking': bool(
                model.ac.request_ready_hard_blocking
            ),
            'request_ready_context_features': bool(
                model.ac.request_ready_context_features
            ),
            'request_ready_head_mode': str(
                model.ac.request_ready_head_mode
            ),
            'request_ready_quantile_head': bool(
                model.ac.request_ready_quantile_head
            ),
            'request_ready_exclude_blocking_loss': bool(
                self.request_ready_exclude_blocking_loss
            ),
            'request_ready_kind_balanced_loss': bool(
                self.request_ready_kind_balanced_loss
            ),
            'request_ready_quantile_loss_coef': float(
                self.request_ready_quantile_loss_coef
            ),
            'request_ready_holdout_folds': int(
                self.request_ready_holdout_folds
            ),
            'request_ready_holdout_fold': int(
                self.request_ready_holdout_fold
            ),
            'device_resource_adapter': bool(
                model.ac.device_resource_adapter_enabled
            ),
            'request_ready_loss_coef': float(
                self.request_ready_loss_coef
            ),
            'request_ready_seconds_loss_coef': float(
                self.request_ready_seconds_loss_coef
            ),
            'request_ready_seconds_loss_scale': float(
                self.request_ready_seconds_loss_scale
            ),
            'request_ready_h1_weight': float(
                self.request_ready_h1_weight
            ),
            'request_ready_h2_weight': float(
                self.request_ready_h2_weight
            ),
            'request_ready_departure_weight': float(
                self.request_ready_departure_weight
            ),
            'request_ready_underprediction_weight': float(
                self.request_ready_underprediction_weight
            ),
            'request_ready_blocking_weight': float(
                self.request_ready_blocking_weight
            ),
            'request_ready_min_labels_per_epoch': int(
                self.request_ready_min_labels_per_epoch
            ),
            'model': model.ac.state_dict(),
            'actor_optim': model.actor_optimizer.state_dict(),
            'critic_optim': model.critic_optimizer.state_dict(),
            'actor_update_health': self._actor_update_health_snapshot(),
            # 'lagrangmdvrpn_multiplier': lagrangmdvrpn_multiplier
        }
        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            checkpoint.pop('actor_optim', None)
            checkpoint.pop('critic_optim', None)
            checkpoint['supervised_optimizer'] = (
                self.resource_supervised_optimizer_state
            )
            checkpoint.update(self._stage2_bc_checkpoint_metadata())
        elif self.trainer.value_normalizer is not None:
            checkpoint['value_normalizer'] = self.trainer.value_normalizer.state_dict()
            role_states = self.trainer.role_value_normalizer_state_dict()
            if role_states is not None:
                checkpoint['role_value_normalizers'] = role_states
        if (
            self.training_stage != CANONICAL_RESOURCE_JOINT
            and hasattr(self.trainer, 'shared_gradient_state_dict')
        ):
            checkpoint['shared_gradient_state'] = (
                self.trainer.shared_gradient_state_dict()
            )
        if extra:
            checkpoint.update(extra)
        if (
            self.training_stage != CANONICAL_RESOURCE_JOINT
            and hasattr(model, 'lagrangmdvrpn_multipliers')
        ):
            checkpoint.update({
                'lagrangmdvrpn_multiplier': model.lagrangmdvrpn_multipliers,
                'lagrangmdvrpn_optimizer': model.lambda_optimizer.state_dict()
            })
        self._atomic_torch_save(checkpoint, save_path)

    def save_emergency_checkpoint(self, reason):
        episode = int(getattr(self, 'episode', -1))
        self.save(
            episode,
            filename='checkpoint_Emergency.pt',
            extra={
                'stage': 'emergency',
                'emergency_reason': str(reason),
                'total_num_steps': int(getattr(self, 'total_num_steps', 0)),
            },
        )
        print(
            f"[Warning] Saved emergency checkpoint after {reason}: "
            f"{os.path.join(self.save_dir, 'checkpoint_Emergency.pt')}"
        )

    def save_plane_bc_checkpoint(self):
        model = self.trainer.policy
        save_path = os.path.join(
            self.save_dir, 'checkpoint_PlaneBC.pt'
        )
        observation_metadata = stage1_observation_metadata(getattr(
            self.all_args, 'global_feature_mode', 'none'
        ))
        checkpoint = {
            **observation_metadata,
            'episodes': 0,
            'stage': 'plane_iga_bc_pretrain',
            'training_stage': self.training_stage,
            'tau': model.ac.tau,
            'model': model.ac.state_dict(),
            'actor_optim': model.actor_optimizer.state_dict(),
            'critic_optim': model.critic_optimizer.state_dict(),
            'plane_order_mode': model.ac.plane_order_mode,
            'plane_pair_decoder': model.ac.plane_pair_decoder,
            'stage1_baseline': model.ac.stage1_baseline,
            'plane_bc_teacher_dir': self.plane_bc_teacher_dir,
            'plane_bc_shared_lr_scale': self.plane_bc_shared_lr_scale,
            'plane_bc_freeze_shared_epochs': self.plane_bc_freeze_shared_epochs,
            'plane_bc_pair_loss_coef': self.plane_bc_pair_loss_coef,
            'plane_bc_order_loss_coef': self.plane_bc_order_loss_coef,
            'plane_bc_tail_start_fraction': (
                self.plane_bc_tail_start_fraction
            ),
            'plane_bc_tail_weight': self.plane_bc_tail_weight,
            'plane_bc_tail_final_start_fraction': (
                self.plane_bc_tail_final_start_fraction
            ),
            'plane_bc_tail_final_weight': self.plane_bc_tail_final_weight,
            'bc_reference_kl_coef': self.bc_reference_kl_coef,
            'bc_reference_checkpoint': self.bc_reference_checkpoint,
            'bc_reference_resolved_path': self.bc_reference_resolved_path,
            'bc_reference_target_kl': self.bc_reference_target_kl,
            'bc_reference_hard_gate': self.bc_reference_hard_gate,
            'adaptive_bc_reference_kl': self.adaptive_bc_reference_kl,
            'actor_kl_backtrack': bool(getattr(
                self.all_args, 'actor_kl_backtrack', False
            )),
            'actor_kl_backtrack_scales': str(getattr(
                self.all_args,
                'actor_kl_backtrack_scales',
                '0.5,0.25,0.125',
            )),
            'paired_case_baseline_scope': self.paired_case_baseline_scope,
            'plane_bc_dagger_schedule': list(
                self.plane_bc_dagger_schedule
            ),
            'plane_bc_dagger_tail_start_fraction': (
                self.plane_bc_dagger_tail_start_fraction
            ),
            'plane_bc_dagger_tail_teacher_rate': (
                self.plane_bc_dagger_tail_teacher_rate
            ),
            'safe_async_graph_clone_workers': int(
                self.safe_async_graph_clone_workers
            ),
            'safe_graph_batch_pipeline': bool(
                self.safe_graph_batch_pipeline
            ),
            'safe_dagger_teacher_overlap': bool(
                self.safe_dagger_teacher_overlap
            ),
            'global_feature_mode': str(getattr(
                self.all_args, 'global_feature_mode', 'none'
            )),
            'train_sampling_mode': str(getattr(
                self.all_args, 'train_sampling_mode', 'uniform'
            )),
            'train_sampling_weights': str(getattr(
                self.all_args, 'train_sampling_weights', ''
            )),
            'train_sampling_size': int(getattr(
                self.all_args, 'train_sampling_size', 0
            )),
            'train_sampling_pool_size': int(getattr(
                self.all_args, 'train_sampling_pool_size', 0
            )),
        }
        if self.trainer.value_normalizer is not None:
            checkpoint['value_normalizer'] = (
                self.trainer.value_normalizer.state_dict()
            )
            role_states = self.trainer.role_value_normalizer_state_dict()
            if role_states is not None:
                checkpoint['role_value_normalizers'] = role_states
        self._atomic_torch_save(checkpoint, save_path)
        print(f'[Info] Saved IGA plane BC checkpoint to {save_path}')

    def save_device_bc_checkpoint(self):
        model = self.trainer.policy
        save_path = os.path.join(self.save_dir, 'checkpoint_DeviceBC.pt')
        observation_metadata = stage1_observation_metadata(getattr(
            self.all_args, 'global_feature_mode', 'none'
        ))
        checkpoint = {
            **observation_metadata,
            'episodes': 0,
            # Keep the historical filename, but make the artifact's phase
            # unambiguous for Stage-2 recovery/auditing.
            'stage': (
                'joint_bc_warmup'
                if self.training_stage == CANONICAL_JOINT_FINETUNE
                else 'request_ready_predictor_final'
                if self.device_bc_training_scope == 'ready_only'
                else 'resource_supervised_final'
            ),
            'training_stage': self.training_stage,
            'phase': self.resource_joint_phase,
            'stage2_training_mode': (
                (
                    'ready_predictor_only'
                    if self.device_bc_training_scope == 'ready_only'
                    else 'supervised_only'
                )
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'device_bc_training_scope': self.device_bc_training_scope,
            'stage2_supervision_contract': (
                dict(STAGE2_SUPERVISION_CONTRACT)
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'source_m2_checkpoint': dict(self.stage1_m2_source or {}),
            'source_m2_path': str((self.stage1_m2_source or {}).get('path', '')),
            'source_m2_sha256': str(
                (self.stage1_m2_source or {}).get('sha256', '')
            ),
            'stage1_m2_checkpoint_summary': self.stage1_m2_checkpoint_summary,
            'source_stage2_checkpoint': dict(self.stage2_source or {}),
            'stage2_checkpoint_summary': self.stage2_checkpoint_summary,
            'protected_parameter_summary_before_bc': (
                self.protected_parameter_summary_before_bc
            ),
            'protected_parameter_summary_after_bc': (
                self.protected_parameter_summary_after_bc
            ),
            'protected_parameter_summary': (
                self._protected_resource_joint_summary()
                if self.training_stage == CANONICAL_RESOURCE_JOINT
                else None
            ),
            'resource_bc_optimizer_reset': bool(
                self.resource_bc_optimizer_reset
            ),
            'resource_bc_total_labels': int(self.resource_bc_total_labels),
            'resource_dense_ranking_total_labels': int(
                self.resource_dense_ranking_total_labels
            ),
            'resource_assignment_total_labels': int(
                self.resource_assignment_total_labels
            ),
            'request_ready_total_labels': int(
                self.request_ready_total_labels
            ),
            'request_ready_predictor_summary_before': (
                self.request_ready_predictor_summary_before
            ),
            'request_ready_predictor_summary_after': (
                self.request_ready_predictor_summary_after
            ),
            'device_bc_teacher': self.device_bc_teacher,
            'device_bc_role_balanced': bool(self.device_bc_role_balanced),
            'device_bc_timing_balanced': bool(
                self.device_bc_timing_balanced
            ),
            'device_bc_legacy_noop_timing': bool(
                self.device_bc_legacy_noop_timing
            ),
            'device_bc_min_teacher_score_margin': float(
                self.device_bc_min_teacher_score_margin
            ),
            'device_bc_ranking_loss_coef': float(
                self.device_bc_ranking_loss_coef
            ),
            'device_bc_categorical_loss_coef': float(
                self.device_bc_categorical_loss_coef
            ),
            'device_bc_assignment_loss_coef': float(
                self.device_bc_assignment_loss_coef
            ),
            'device_bc_assignment_margin_loss_coef': float(
                self.device_bc_assignment_margin_loss_coef
            ),
            'device_bc_assignment_margin': float(
                self.device_bc_assignment_margin
            ),
            'device_bc_min_assignment_labels_per_epoch': int(
                self.device_bc_min_assignment_labels_per_epoch
            ),
            'device_bc_min_ranking_labels_per_epoch': int(
                self.device_bc_min_ranking_labels_per_epoch
            ),
            'device_bc_ranking_temperature': float(
                self.device_bc_ranking_temperature
            ),
            'device_bc_timing_loss_coef': float(
                self.device_bc_timing_loss_coef
            ),
            'device_bc_dagger_schedule': list(
                self.device_bc_dagger_schedule
            ),
            'resource_lookahead_contract': self._resource_lookahead_contract(),
            'resource_iga_teacher_dir': str(getattr(
                self.all_args, 'resource_iga_teacher_dir', ''
            )),
            'resource_iga_teacher_index': str(getattr(
                self.all_args, 'resource_iga_teacher_index', ''
            )),
            'joint_iga_teacher_dir': str(getattr(
                self.all_args, 'joint_iga_teacher_dir', ''
            )),
            'joint_iga_teacher_index': str(getattr(
                self.all_args, 'joint_iga_teacher_index', ''
            )),
            'resource_actor_summary_before_bc': (
                self.resource_actor_summary_before_bc
            ),
            'resource_actor_summary_after_bc': (
                self.resource_actor_summary_after_bc
            ),
            'plane_actor_summary_before_bc': self.plane_actor_summary_before_bc,
            'plane_actor_summary_after_bc': self.plane_actor_summary_after_bc,
            'tau': model.ac.tau,
            'plane_order_mode': model.ac.plane_order_mode,
            'plane_pair_decoder': model.ac.plane_pair_decoder,
            'stage1_baseline': model.ac.stage1_baseline,
            'device_policy_head_mode': str(
                model.ac.device_policy_head_mode
            ),
            'ordinary_device_type_count': int(
                model.ac.ordinary_device_type_count
            ),
            'device_timing_head': bool(model.ac.device_timing_head),
            'device_global_matching': bool(
                model.ac.device_global_matching
            ),
            'request_ready_prediction': bool(
                model.ac.request_ready_prediction
            ),
            'request_ready_time_scale': float(
                model.ac.request_ready_time_scale
            ),
            'request_ready_policy_injection': str(
                model.ac.request_ready_policy_injection
            ),
            'request_ready_hard_blocking': bool(
                model.ac.request_ready_hard_blocking
            ),
            'request_ready_context_features': bool(
                model.ac.request_ready_context_features
            ),
            'request_ready_head_mode': str(
                model.ac.request_ready_head_mode
            ),
            'request_ready_quantile_head': bool(
                model.ac.request_ready_quantile_head
            ),
            'request_ready_exclude_blocking_loss': bool(
                self.request_ready_exclude_blocking_loss
            ),
            'request_ready_kind_balanced_loss': bool(
                self.request_ready_kind_balanced_loss
            ),
            'request_ready_quantile_loss_coef': float(
                self.request_ready_quantile_loss_coef
            ),
            'request_ready_holdout_folds': int(
                self.request_ready_holdout_folds
            ),
            'request_ready_holdout_fold': int(
                self.request_ready_holdout_fold
            ),
            'device_resource_adapter': bool(
                model.ac.device_resource_adapter_enabled
            ),
            'request_ready_loss_coef': float(
                self.request_ready_loss_coef
            ),
            'request_ready_seconds_loss_coef': float(
                self.request_ready_seconds_loss_coef
            ),
            'request_ready_seconds_loss_scale': float(
                self.request_ready_seconds_loss_scale
            ),
            'request_ready_h1_weight': float(
                self.request_ready_h1_weight
            ),
            'request_ready_h2_weight': float(
                self.request_ready_h2_weight
            ),
            'request_ready_departure_weight': float(
                self.request_ready_departure_weight
            ),
            'request_ready_underprediction_weight': float(
                self.request_ready_underprediction_weight
            ),
            'request_ready_blocking_weight': float(
                self.request_ready_blocking_weight
            ),
            'request_ready_min_labels_per_epoch': int(
                self.request_ready_min_labels_per_epoch
            ),
            'model': model.ac.state_dict(),
            'supervised_optimizer': self.resource_supervised_optimizer_state,
        }
        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            checkpoint.update(self._stage2_bc_checkpoint_metadata())
        if (
            self.training_stage != CANONICAL_RESOURCE_JOINT
            and self.trainer.value_normalizer is not None
        ):
            checkpoint['value_normalizer'] = self.trainer.value_normalizer.state_dict()
            role_states = self.trainer.role_value_normalizer_state_dict()
            if role_states is not None:
                checkpoint['role_value_normalizers'] = role_states
        self._atomic_torch_save(checkpoint, save_path)
        print(f"[Info] Saved device BC checkpoint to {save_path}")

    def _restore_shared_resource_bc(self, checkpoint_path):
        """Restore one audited DeviceBC boundary into parallel PPO branches."""
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Shared resource BC checkpoint does not exist: {checkpoint_path}'
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_head_contract = (
            str(checkpoint.get('device_policy_head_mode', 'shared')),
            int(checkpoint.get('ordinary_device_type_count', 10)),
            bool(checkpoint.get('device_timing_head', False)),
            bool(checkpoint.get('device_global_matching', False)),
        )
        runtime_head_contract = (
            str(self.policy.ac.device_policy_head_mode),
            int(self.policy.ac.ordinary_device_type_count),
            bool(self.policy.ac.device_timing_head),
            bool(self.policy.ac.device_global_matching),
        )
        if checkpoint_head_contract != runtime_head_contract:
            raise ValueError(
                'Shared resource BC device-head contract differs from this '
                f'PPO branch: checkpoint={checkpoint_head_contract}, '
                f'configured={runtime_head_contract}.'
            )
        if (
            str(checkpoint.get('training_stage', '')).strip().lower()
            != CANONICAL_RESOURCE_JOINT
            or checkpoint.get('stage') != 'resource_bc_warmup'
            or checkpoint.get('phase') != 'resource_bc_warmup_completed'
            or checkpoint.get('resource_bc_optimizer_reset') is not True
            or int(checkpoint.get('resource_bc_total_labels', 0)) <= 0
        ):
            raise ValueError(
                'Shared resource BC input is not a completed canonical '
                'resource_bc_warmup checkpoint.'
            )
        source_sha = str((self.stage1_m2_source or {}).get('sha256', ''))
        if not source_sha or checkpoint.get('source_m2_sha256') != source_sha:
            raise ValueError(
                'Shared resource BC checkpoint was not derived from the '
                'configured immutable Stage-1 M2 source.'
            )
        # Saving and restoring must use one authoritative contract.  The
        # lookahead implementation has grown beyond the original five fields;
        # duplicating the old subset here made every new DeviceBC checkpoint
        # impossible to restore even when the PPO command inherited the exact
        # same semantics.
        expected_lookahead = self._resource_lookahead_contract()
        if checkpoint.get('resource_lookahead_contract') != expected_lookahead:
            raise ValueError(
                'Shared resource BC lookahead semantics differ from this '
                f'PPO branch: checkpoint={checkpoint.get("resource_lookahead_contract")}, '
                f'configured={expected_lookahead}.'
            )
        validate_stage1_checkpoint_contract(
            checkpoint,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            stage1_baseline=self.policy.ac.stage1_baseline,
            strict_metadata=True,
        )
        model = checkpoint.get('model')
        target = self.policy.ac.state_dict()
        if not isinstance(model, dict) or set(model) != set(target):
            raise ValueError(
                'Shared resource BC model keys do not exactly match the '
                'configured Stage-2 network.'
            )
        mismatched = [
            name for name in target
            if tuple(model[name].shape) != tuple(target[name].shape)
        ]
        if mismatched:
            raise ValueError(
                'Shared resource BC tensor shapes mismatch: '
                f'{mismatched[:8]}'
            )
        protected_before = self._protected_resource_joint_summary()
        recorded_protected = checkpoint.get(
            'protected_parameter_summary_after_bc'
        )
        if protected_before != recorded_protected:
            raise ValueError(
                'Shared resource BC protected plane/encoder tensors differ '
                'from the restored Stage-1 source.'
            )
        self.policy.load_model_state(model)
        self._assert_resource_joint_protected(
            protected_before, 'shared_resource_bc_restore'
        )
        recorded_actor = checkpoint.get('resource_actor_summary_after_bc')
        if self._resource_actor_summary() != recorded_actor:
            raise RuntimeError(
                'Shared resource BC actor digest was not reproduced bitwise.'
            )
        self.protected_parameter_summary_before_bc = checkpoint.get(
            'protected_parameter_summary_before_bc'
        )
        self.protected_parameter_summary_after_bc = recorded_protected
        self.resource_actor_summary_before_bc = checkpoint.get(
            'resource_actor_summary_before_bc'
        )
        self.resource_actor_summary_after_bc = recorded_actor
        self.resource_bc_total_labels = int(
            checkpoint['resource_bc_total_labels']
        )
        self.resource_bc_optimizer_reset = True
        self.resource_joint_phase = 'resource_bc_warmup_completed'
        self.resource_bc_checkpoint_source = {
            **source_checkpoint_metadata(checkpoint_path),
            'device_bc_teacher': checkpoint.get('device_bc_teacher'),
            'device_bc_dagger_schedule': checkpoint.get(
                'device_bc_dagger_schedule'
            ),
            'resource_bc_total_labels': self.resource_bc_total_labels,
        }
        self._pending_value_normalizer_state = checkpoint.get(
            'value_normalizer', self._pending_value_normalizer_state
        )
        self._pending_role_value_normalizer_states = checkpoint.get(
            'role_value_normalizers', self._pending_role_value_normalizer_states
        )
        print(
            '[Info] Restored shared DeviceBC boundary '
            f'{checkpoint_path} labels={self.resource_bc_total_labels}.',
            flush=True,
        )

    def _restore_stage1_m2(self, checkpoint_path):
        """Strictly restore only the Stage-1 plane/shared hand-off state."""
        if not os.path.isfile(str(checkpoint_path)):
            raise FileNotFoundError(
                "resource_joint Stage-1 M2 checkpoint does not exist: "
                f"{checkpoint_path}"
            )
        source_metadata = source_checkpoint_metadata(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        observation_contract = validate_stage1_checkpoint_contract(
            checkpoint,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            stage1_baseline=self.policy.ac.stage1_baseline,
            # A resource-joint transition is never allowed to consume a
            # shape-compatible checkpoint from an older environment state
            # machine.  This is stricter than ordinary legacy resume.
            strict_metadata=True,
        )
        validation = validate_stage1_m2_checkpoint(
            checkpoint,
            self.policy.ac.state_dict(),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
        )
        # The generic loader may skip missing resource tensors by design.  The
        # strict validator above has already proved that every protected
        # tensor is present and shape-compatible, so this permissiveness cannot
        # weaken the plane/shared contract.
        self.policy.load_model_state(checkpoint['model'])
        # Even an arm that will later adapt the shared encoder must first
        # reproduce the complete Stage-1 hand-off bit-for-bit.
        loaded_protected_summary = self._stage1_handoff_summary()
        if loaded_protected_summary != validation['source_summary']:
            raise RuntimeError(
                "Strict Stage-1 M2 hand-off loader did not reproduce the "
                "validated protected plane/shared tensors bitwise."
            )
        self.policy.ac.tau = checkpoint.get('tau', self.policy.ac.tau)
        self.all_args.anneal_original = self.policy.ac.tau
        # Stage2 has no value objective.  Do not even hydrate the Stage1 PPO
        # normalization state in memory; the supervised checkpoint and its
        # eventual Stage3 consumer both start their RL state independently.
        self._pending_value_normalizer_state = None
        self._pending_role_value_normalizer_states = None
        self.stage1_m2_source = source_metadata
        self.stage1_m2_checkpoint_summary = {
            **validation,
            'observation_contract': observation_contract,
            'source_m2_checkpoint': dict(source_metadata),
        }
        self.protected_parameter_summary_before_bc = (
            self._protected_resource_joint_summary()
        )
        self.resource_joint_phase = 'stage1_m2_restored'
        print(
            "[Info] Strict Stage-1 M2 hand-off accepted for resource_joint: "
            f"{source_metadata['path']} "
            f"(sha256={source_metadata['sha256']})."
        )
        # Stage2 owns a dedicated supervised optimizer and never restores or
        # steps PPO moments from Stage1.
        self.policy.reset_optimizers()
        self.resource_bc_optimizer_reset = False
        if self.device_bc_training_scope == 'policy_frozen_ready':
            self._restore_frozen_ready_source()
            if getattr(self.all_args, 'stage2_policy_warmstart_checkpoint', ''):
                self._restore_stage2_policy_warmstart()
        return checkpoint

    def _restore_stage2_policy_warmstart(self):
        metadata = source_checkpoint_metadata(self.all_args.stage2_policy_warmstart_checkpoint)
        if metadata['sha256'] != self.all_args.stage2_policy_warmstart_sha256:
            raise ValueError('Policy warm-start checkpoint SHA256 mismatch.')
        checkpoint = torch.load(metadata['path'], map_location='cpu', weights_only=True)
        validate_stage1_checkpoint_contract(
            checkpoint, global_feature_mode=self.all_args.global_feature_mode,
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            stage1_baseline=self.policy.ac.stage1_baseline, strict_metadata=True,
        )
        expected = validate_policy_warmstart(
            checkpoint, self.policy.ac.state_dict(),
            source_sha256=self.stage1_m2_source['sha256'],
            ready_sha256=self.frozen_ready_source['sha256'],
            planning_contract=self._resource_lookahead_contract(),
            injection=self.policy.ac.request_ready_policy_injection,
            head_config=ready_head_config(self.policy.ac),
            teacher_index=self.all_args.resource_iga_teacher_index,
        )
        self.policy.ac.load_state_dict(checkpoint['model'], strict=True)
        observed = protected_parameter_summary(self.policy.ac.state_dict(), prefixes=('',))
        if observed != expected:
            raise RuntimeError('Policy warm start was not bitwise exact.')
        self.stage2_policy_warmstart = {
            **metadata, 'selected_bc_epoch': checkpoint['selected_bc_epoch'],
            'model_sha256': observed['sha256'], 'mode': 'weights_only_fork',
            'optimizer_restored': False, 'rng_restored': False,
            'source_training_seed': checkpoint.get('stage2_bc_state', {}).get('seed'),
        }
        self.stage2_policy_warmstart_replay = None
        self._assert_frozen_ready_unchanged()
        print(f'[Stage2Transfer] Exact policy fork {metadata["sha256"]}; fresh optimizer/RNG.', flush=True)

    def _check_stage2_policy_warmstart_replay(self):
        path = getattr(self.all_args, 'stage2_policy_warmstart_evaluation', '')
        if not path:
            return
        result = warmstart_replay(
            path, self.all_args.stage2_policy_warmstart_evaluation_sha256,
            self.last_eval_records,
        )
        self.stage2_policy_warmstart_replay = result
        output = Path(self.log_dir) / 'policy_warmstart_replay.json'
        output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
        self._report_progress('policy_warmstart_replay_checked', stage2_policy_warmstart_replay=result)
        if not result['passed']:
            raise RuntimeError('Policy warm-start replay differs; no gradient update permitted.')
        print(f'[Stage2Transfer] Pre-BC replay passed {result["case_count"]} cases at zero tolerance.', flush=True)

    def _restore_frozen_ready_source(self):
        metadata = source_checkpoint_metadata(self.all_args.request_ready_checkpoint)
        if metadata['sha256'] != self.all_args.request_ready_checkpoint_sha256:
            raise ValueError('Frozen ready checkpoint SHA256 mismatch.')
        checkpoint = torch.load(metadata['path'], map_location='cpu')
        validate_stage1_checkpoint_contract(
            checkpoint, global_feature_mode=self.all_args.global_feature_mode,
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            stage1_baseline=self.policy.ac.stage1_baseline, strict_metadata=True,
        )
        state = validate_frozen_ready_checkpoint(
            checkpoint, self.policy.ac, source_sha256=self.stage1_m2_source['sha256'],
            planning_contract=self._resource_lookahead_contract(),
        )
        source_index = Path(checkpoint['resource_iga_teacher_index']).resolve()
        runtime_index = Path(self.all_args.resource_iga_teacher_index).resolve()
        if source_index != runtime_index:
            raise ValueError('Ready predictor and policy BC teacher indexes differ.')
        self.policy.ac.request_ready_head.load_state_dict(state, strict=True)
        self.policy.ac.request_ready_head.requires_grad_(False)
        self.policy.ac.request_ready_head.eval()
        torch.nn.init.zeros_(self.policy.ac.request_ready_feature.weight)
        self.frozen_ready_source = {
            **metadata, 'teacher_index_sha256': hashlib.sha256(runtime_index.read_bytes()).hexdigest(),
            **ready_head_config(self.policy.ac),
        }
        self.frozen_ready_head_summary = protected_parameter_summary(
            self.policy.ac.state_dict(), prefixes=('request_ready_head.',)
        )
        self._assert_resource_joint_protected(self.protected_parameter_summary_before_bc, 'ready_source_load')
        print(f'[Stage2] Loaded frozen ready predictor {metadata["sha256"]}; projection reset to zero.', flush=True)

    def _restore_stage2_joint_finetune(self, checkpoint_path):
        """Strictly initialize Stage3 from one complete Stage2 policy."""

        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Stage3 source checkpoint does not exist: {checkpoint_path}'
            )
        provenance_resolver = None
        frozen_manifest = str(getattr(self.all_args, 'stage2_frozen_manifest', '') or '')
        if frozen_manifest:
            from onpolicy.utils.stage2_frozen import FrozenStage2Bundle
            provenance_resolver = FrozenStage2Bundle(frozen_manifest)
            if checkpoint_path != str(provenance_resolver.path('b0')):
                raise ValueError('Frozen Stage3 initialization must use the manifest B0 checkpoint.')
            provenance_resolver.validate(self.policy.ac.state_dict())
            checkpoint = provenance_resolver.checkpoint()
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        source_stage2 = source_checkpoint_metadata(checkpoint_path)
        observation_contract = validate_stage1_checkpoint_contract(
            checkpoint,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            stage1_baseline=self.policy.ac.stage1_baseline,
            strict_metadata=True,
        )
        if self.stage3_handoff_mode == 'strict':
            planning_contract = self._resource_lookahead_contract()
            contract_transition = None
        elif self.stage3_handoff_mode == 'critical_path_wave1':
            contract_transition = self._critical_path_wave1_source_contracts(
                checkpoint
            )
            planning_contract = contract_transition[
                'source_planning_contract'
            ]
        elif self.stage3_handoff_mode == 'ppo_gain_wave':
            contract_transition = self._ppo_gain_source_contracts(checkpoint)
            planning_contract = contract_transition[
                'source_planning_contract'
            ]
        else:
            raise ValueError(
                f'Unsupported Stage3 handoff mode: {self.stage3_handoff_mode!r}.'
            )
        validation = validate_stage2_joint_finetune_checkpoint(
            checkpoint,
            self.policy.ac.state_dict(),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
            planning_contract=planning_contract,
            reward_contract=None,
            request_ready_time_scale=(
                self.policy.ac.request_ready_time_scale
            ),
            resource_decoder=('hungarian' if self.policy.ac.device_global_matching
                              else 'autoregressive'),
            provenance_resolver=provenance_resolver,
        )
        source_m2 = source_checkpoint_metadata(validation['source_m2_path'])
        if source_m2['sha256'] != validation['source_m2_sha256']:
            raise ValueError(
                'Stage3 source has a stale Stage1 lineage: '
                f"recorded={validation['source_m2_sha256']} "
                f"observed={source_m2['sha256']}."
            )
        recorded_source_m2 = checkpoint.get('source_m2_checkpoint', {})
        if recorded_source_m2 and dict(recorded_source_m2) != source_m2:
            raise ValueError(
                'Stage3 source_m2_checkpoint binding is inconsistent.'
            )

        self.policy.load_model_state(checkpoint['model'])
        restored_summary = protected_parameter_summary(
            self.policy.ac.state_dict(), prefixes=('',)
        )
        if restored_summary != validation['model_summary']:
            raise RuntimeError(
                'Stage2 -> Stage3 loader did not reproduce the full policy '
                'bit-for-bit.'
            )
        self.policy.ac.tau = checkpoint.get('tau', self.policy.ac.tau)
        self.all_args.anneal_original = self.policy.ac.tau
        # Never inherit Stage2 optimizer/PPO normalization state, including
        # nested resource-RL training archives. Stage3 starts these afresh.
        self._pending_value_normalizer_state = None
        self._pending_role_value_normalizer_states = None
        self.stage1_m2_source = source_m2
        self.stage1_m2_checkpoint_summary = checkpoint.get(
            'stage1_m2_checkpoint_summary'
        )
        self.stage2_source = source_stage2
        self.stage2_checkpoint_summary = {
            **validation,
            'observation_contract': observation_contract,
            'source_stage2_checkpoint': dict(source_stage2),
            'handoff_mode': self.stage3_handoff_mode,
            'contract_transition': contract_transition,
        }
        self.plane_actor_summary_before_bc = self._plane_actor_summary()
        self.resource_actor_summary_before_bc = self._resource_actor_summary()
        self.stage3_contract_transition = contract_transition
        self.resource_wait_dual_value = float(
            self.all_args.resource_lateness_coef
        )
        self.all_args.resource_lateness_coef = self.resource_wait_dual_value
        applied_dual = self.envs.call(
            'set_resource_lateness_coef', self.resource_wait_dual_value
        )
        if any(
            not np.isclose(float(value), self.resource_wait_dual_value)
            for value in applied_dual
        ):
            raise RuntimeError(
                'Training workers rejected the fresh Stage3 resource-cost state.'
            )
        self.policy.reset_optimizers()
        self.resource_bc_optimizer_reset = False
        self.resource_joint_phase = 'stage2_handoff_restored'
        print(
            '[Info] Strict Stage2 hand-off accepted for joint_finetune: '
            f"{source_stage2['path']} "
            f"(sha256={source_stage2['sha256']}).",
            flush=True,
        )
        return checkpoint

    def _validate_stage2_method_contract(self, checkpoint):
        """Reject a recovery checkpoint from a different Stage-2 arm."""

        expected = {
            'device_bc_teacher': str(self.device_bc_teacher),
            'device_bc_role_balanced': bool(self.device_bc_role_balanced),
            'device_bc_dagger_schedule': [
                float(value) for value in self.device_bc_dagger_schedule
            ],
            'resource_ppo_update_schedule': str(
                self.resource_ppo_update_schedule
            ),
            'resource_ppo_warmup_epochs': int(
                self.resource_ppo_warmup_epochs
            ),
            'stage2_allow_shared_unfreeze': bool(
                self.stage2_allow_shared_unfreeze
            ),
            'gnn_freeze_epochs': int(self.gnn_freeze_epochs),
        }
        observed = {
            'device_bc_teacher': str(
                checkpoint.get('device_bc_teacher', '')
            ),
            'device_bc_role_balanced': bool(
                checkpoint.get('device_bc_role_balanced', False)
            ),
            'device_bc_dagger_schedule': [
                float(value)
                for value in checkpoint.get('device_bc_dagger_schedule', [])
            ],
            'resource_ppo_update_schedule': str(
                checkpoint.get('resource_ppo_update_schedule', '')
            ),
            'resource_ppo_warmup_epochs': int(
                checkpoint.get('resource_ppo_warmup_epochs', -1)
            ),
            'stage2_allow_shared_unfreeze': bool(
                checkpoint.get('stage2_allow_shared_unfreeze', False)
            ),
            'gnn_freeze_epochs': int(
                checkpoint.get('gnn_freeze_epochs', -1)
            ),
        }
        mismatches = {
            key: {'checkpoint': observed[key], 'configured': value}
            for key, value in expected.items()
            if observed[key] != value
        }

        experiment_config = checkpoint.get('experiment_config', {})
        if not isinstance(experiment_config, dict):
            experiment_config = {}
        observed_reward = str(
            experiment_config.get('hindsight_reward_mode', '')
        )
        if observed_reward != self.hindsight_reward_mode:
            mismatches['hindsight_reward_mode'] = {
                'checkpoint': observed_reward,
                'configured': self.hindsight_reward_mode,
            }

        if self.device_bc_teacher == 'iga':
            for key, configured in (
                (
                    'resource_iga_teacher_dir',
                    str(getattr(
                        self.all_args, 'resource_iga_teacher_dir', ''
                    ) or ''),
                ),
                (
                    'resource_iga_teacher_index',
                    str(getattr(
                        self.all_args, 'resource_iga_teacher_index', ''
                    ) or ''),
                ),
            ):
                checkpoint_value = str(checkpoint.get(key, '') or '')
                configured_path = str(Path(configured).expanduser().resolve())
                checkpoint_path = str(
                    Path(checkpoint_value).expanduser().resolve()
                )
                if not configured or not checkpoint_value or configured_path != checkpoint_path:
                    mismatches[key] = {
                        'checkpoint': checkpoint_value,
                        'configured': configured,
                    }
        if mismatches:
            raise ValueError(
                'Stage-2 recovery method contract mismatch: '
                f'{mismatches}.'
            )
        return expected

    def _restore_stage2_supervised_recovery(
        self,
        checkpoint_path,
        checkpoint=None,
    ):
        """Resume the closed, supervised-only Stage2 transition.

        The emergency artifact contains the already-updated resource heads and
        the dedicated supervised Adam state, but intentionally no PPO state.
        We first reconstruct and validate the immutable Stage1 hand-off, then
        replace only the full runtime model with the emergency snapshot and
        prove that every protected plane/shared tensor stayed bitwise equal.
        The explicit rollout cursor is supplied by the recovery command because
        historical emergency checkpoints predate cursor persistence.
        """

        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Stage2 supervised recovery checkpoint does not exist: '
                f'{checkpoint_path}'
            )
        if checkpoint is None:
            # Full local recovery artifacts can include legacy NumPy RNG state.
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.stage2_policy_warmstart = copy.deepcopy(checkpoint.get('stage2_policy_warmstart'))
        self.stage2_policy_warmstart_replay = copy.deepcopy(checkpoint.get('stage2_policy_warmstart_replay'))
        bc_state = checkpoint.get('stage2_bc_state') or {}
        epoch_boundary = bool(bc_state.get('at_epoch_boundary'))
        if self.device_bc_eval_each_epoch:
            if not epoch_boundary:
                raise ValueError('Epoch-selected BC recovery only accepts complete epoch boundaries.')
            if checkpoint.get('stage2_bc_run_contract') != self._stage2_bc_run_contract():
                raise ValueError('Stage2 BC recovery data/optimizer/injection contract mismatch.')
            if bool(bc_state.get('eval_each_epoch')) != self.device_bc_eval_each_epoch:
                raise ValueError('Stage2 BC recovery epoch-selection mode mismatch.')
        pre_info = bc_state.get('pre_supervised_info')
        if self.use_eval and not pre_info:
            raise ValueError('Stage2 supervised recovery lacks pre-supervised metrics; refusing before training.')
        validate_stage1_checkpoint_contract(
            checkpoint,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            stage1_baseline=self.policy.ac.stage1_baseline,
            strict_metadata=True,
        )
        if (
            str(checkpoint.get('training_stage', '')).strip().lower()
            != CANONICAL_RESOURCE_JOINT
            or checkpoint.get('stage2_training_mode') != 'supervised_only'
            or checkpoint.get('stage2_supervision_contract')
            != dict(STAGE2_SUPERVISION_CONTRACT)
            or (checkpoint.get('stage') != 'emergency' and not epoch_boundary)
            or checkpoint.get('phase') not in {
                'resource_supervised', 'resource_supervised_completed'
            }
        ):
            raise ValueError(
                'Stage2 supervised recovery requires an emergency checkpoint '
                'captured during the resource_supervised phase.'
            )

        expected_method = {
            'device_bc_teacher': str(self.device_bc_teacher),
            'device_bc_role_balanced': bool(self.device_bc_role_balanced),
            'device_bc_timing_balanced': bool(self.device_bc_timing_balanced),
            'device_bc_legacy_noop_timing': bool(
                self.device_bc_legacy_noop_timing
            ),
            'device_bc_dagger_schedule': [
                float(value) for value in self.device_bc_dagger_schedule
            ],
            'device_bc_ranking_loss_coef': float(
                self.device_bc_ranking_loss_coef
            ),
            'device_bc_timing_loss_coef': float(
                self.device_bc_timing_loss_coef
            ),
            'request_ready_loss_coef': float(self.request_ready_loss_coef),
            'request_ready_time_scale': float(
                self.policy.ac.request_ready_time_scale
            ),
        }
        observed_method = {
            key: checkpoint.get(key) for key in expected_method
        }
        method_mismatches = {}
        for key, expected_value in expected_method.items():
            observed_value = observed_method[key]
            if isinstance(expected_value, float):
                matches = (
                    observed_value is not None
                    and math.isclose(
                        float(observed_value),
                        expected_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
            elif isinstance(expected_value, list):
                matches = [
                    float(value) for value in (observed_value or [])
                ] == expected_value
            else:
                matches = observed_value == expected_value
            if not matches:
                method_mismatches[key] = {
                    'checkpoint': observed_value,
                    'configured': expected_value,
                }
        if checkpoint.get('resource_lookahead_contract') != (
            self._resource_lookahead_contract()
        ):
            method_mismatches['resource_lookahead_contract'] = {
                'checkpoint': checkpoint.get('resource_lookahead_contract'),
                'configured': self._resource_lookahead_contract(),
            }
        for key, configured in (
            (
                'resource_iga_teacher_dir',
                str(getattr(self.all_args, 'resource_iga_teacher_dir', '') or ''),
            ),
            (
                'resource_iga_teacher_index',
                str(getattr(self.all_args, 'resource_iga_teacher_index', '') or ''),
            ),
        ):
            observed = str(checkpoint.get(key, '') or '')
            if (
                not configured
                or not observed
                or Path(configured).expanduser().resolve()
                != Path(observed).expanduser().resolve()
            ):
                method_mismatches[key] = {
                    'checkpoint': observed,
                    'configured': configured,
                }
        if method_mismatches:
            raise ValueError(
                'Stage2 supervised recovery method contract mismatch: '
                f'{method_mismatches}.'
            )

        source_path = str(checkpoint.get('source_m2_path', '') or '')
        source_sha = str(checkpoint.get('source_m2_sha256', '') or '')
        if not source_path or not source_sha:
            raise ValueError(
                'Stage2 supervised recovery is missing immutable Stage1 lineage.'
            )
        source_metadata = source_checkpoint_metadata(source_path)
        if source_metadata['sha256'] != source_sha:
            raise ValueError(
                'Stage2 supervised recovery source M2 digest changed: '
                f'checkpoint={source_sha} observed={source_metadata["sha256"]}.'
            )
        recorded_source = checkpoint.get('source_m2_checkpoint', {})
        if recorded_source and dict(recorded_source) != source_metadata:
            raise ValueError(
                'Stage2 supervised recovery source_m2_checkpoint binding is '
                'inconsistent.'
            )

        recovery_model = checkpoint.get('model')
        runtime_model = self.policy.ac.state_dict()
        if not isinstance(recovery_model, dict) or set(recovery_model) != set(
            runtime_model
        ):
            raise ValueError(
                'Stage2 supervised recovery model keys do not exactly match '
                'the configured network.'
            )
        mismatched_shapes = [
            name for name in runtime_model
            if tuple(recovery_model[name].shape)
            != tuple(runtime_model[name].shape)
        ]
        if mismatched_shapes:
            raise ValueError(
                'Stage2 supervised recovery tensor shapes mismatch: '
                f'{mismatched_shapes[:8]}'
            )
        supervised_optimizer = checkpoint.get('supervised_optimizer')
        if (
            not isinstance(supervised_optimizer, dict)
            or not supervised_optimizer.get('param_groups')
        ):
            raise ValueError(
                'Stage2 supervised recovery is missing its dedicated optimizer '
                'state.'
            )

        # Rebuild the immutable source evidence through the canonical loader.
        self._restore_stage1_m2(source_metadata['path'])
        expected_protected = self.protected_parameter_summary_before_bc
        recorded_protected = checkpoint.get(
            'protected_parameter_summary_before_bc'
        )
        if recorded_protected != expected_protected:
            raise ValueError(
                'Stage2 supervised recovery protected tensors do not match '
                'the immutable Stage1 source.'
            )
        self.policy.load_model_state(recovery_model)
        self._assert_resource_joint_protected(
            expected_protected,
            'supervised_recovery_restore',
        )
        if self.device_bc_training_scope == 'policy_frozen_ready':
            if checkpoint.get('frozen_ready_source') != self.frozen_ready_source:
                raise ValueError('Stage2 recovery ready-source lineage mismatch.')
            self.stage2_frozen_state_before = checkpoint.get('stage2_frozen_state_before')
            if self.device_bc_eval_each_epoch and self.stage2_frozen_state_before is None:
                raise ValueError('Stage2 recovery lacks its frozen-scope digest.')
            self._assert_frozen_ready_unchanged()
        self.stage2_pre_supervised_info = copy.deepcopy(pre_info)
        self.stage2_bc_epoch_records = copy.deepcopy(bc_state.get('epoch_records', []))
        self.stage2_bc_completed_epochs = int(bc_state.get('completed_epochs', 0))
        self.stage2_bc_pending_rng = bc_state.get('rng')
        if epoch_boundary:
            if not self.stage2_bc_pending_rng:
                raise ValueError('Epoch-boundary recovery lacks RNG state.')
            if not 0 < self.stage2_bc_completed_epochs < self.device_bc_pretrain_epochs:
                raise ValueError('Recovery must have at least one complete and one unfinished BC epoch.')
            expected_names = self._stage2_bc_checkpoint_metadata()['stage2_bc_state']['optimizer_parameter_names']
            if bc_state.get('optimizer_parameter_names') != expected_names:
                raise ValueError('Supervised recovery optimizer parameter order mismatch.')
            # Earlier eligible epochs remain available for final selection in
            # the new run directory. The current boundary can stand alone.
            for record in self.stage2_bc_epoch_records:
                name = record['filename']
                if Path(name).name != name:
                    raise ValueError('Invalid BC epoch checkpoint filename.')
                if record['epoch'] == self.stage2_bc_completed_epochs:
                    artifact = checkpoint
                else:
                    artifact = torch.load(str(Path(checkpoint_path).parent / name), map_location='cpu', weights_only=False)
                    if artifact.get('stage2_bc_run_contract') != self._stage2_bc_run_contract():
                        raise ValueError('Prior selected epoch has a different BC contract.')
                artifact = copy.deepcopy(artifact)
                artifact['stage2_bc_state']['rng'] = portable_rng_state(artifact['stage2_bc_state']['rng'])
                self._atomic_torch_save(artifact, os.path.join(self.save_dir, name))
            copy_bc_epoch_evidence(
                Path(checkpoint_path).parent.parent, Path(self.save_dir).parent,
                self.stage2_bc_completed_epochs,
            )

        self.resource_actor_summary_before_bc = checkpoint.get(
            'resource_actor_summary_before_bc'
        )
        self.request_ready_predictor_summary_before = checkpoint.get(
            'request_ready_predictor_summary_before'
        )
        if (
            not self.resource_actor_summary_before_bc
            or not self.request_ready_predictor_summary_before
        ):
            raise ValueError(
                'Stage2 supervised recovery lacks pre-supervision digest '
                'evidence.'
            )
        self.resource_supervised_optimizer_state = copy.deepcopy(
            supervised_optimizer
        )
        self.device_bc_resume_epoch = int(getattr(
            self.all_args, 'device_bc_resume_epoch', 0
        ))
        self.device_bc_resume_completed_rollouts = int(getattr(
            self.all_args,
            'device_bc_resume_completed_rollouts',
            0,
        ))
        if epoch_boundary:
            if self.device_bc_resume_epoch not in (0, self.stage2_bc_completed_epochs):
                raise ValueError('CLI recovery epoch differs from the saved boundary.')
            if self.device_bc_resume_completed_rollouts:
                raise ValueError('An epoch-boundary recovery cannot skip within-epoch rollouts.')
            self.device_bc_resume_epoch = self.stage2_bc_completed_epochs
            for key in ('resource_bc_total_labels', 'resource_dense_ranking_total_labels',
                        'resource_assignment_total_labels', 'request_ready_total_labels'):
                setattr(self, key, int(checkpoint.get(key, 0)))
        self.resource_joint_phase = 'resource_supervised'
        self.exact_resume_stage2_supervised = True
        print(
            '[Info] Restoring supervised Stage2 from '
            f'epoch={self.device_bc_resume_epoch + 1}, completed_rollouts='
            f'{self.device_bc_resume_completed_rollouts}/'
            f'{self.device_bc_max_rollouts_per_epoch}, checkpoint='
            f'{checkpoint_path}.',
            flush=True,
        )
        return checkpoint

    def _restore_stage2_recovery(self, checkpoint_path):
        """Strictly restore a resource_joint PPO cursor and all train state."""

        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Stage-2 recovery checkpoint does not exist: {checkpoint_path}'
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        validate_stage1_checkpoint_contract(
            checkpoint,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            stage1_baseline=self.policy.ac.stage1_baseline,
            strict_metadata=True,
        )
        validation = validate_stage2_recovery_checkpoint(
            checkpoint,
            self.policy.ac.state_dict(),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
            protected_prefixes=self._resource_joint_protected_prefixes(),
        )
        self._validate_stage2_method_contract(checkpoint)
        if validation['completed_epochs'] > self.num_episodes:
            raise ValueError(
                'Stage-2 recovery cursor exceeds configured PPO epochs: '
                f"checkpoint={validation['completed_epochs']} "
                f'configured={self.num_episodes}.'
            )

        source_metadata = source_checkpoint_metadata(
            validation['source_m2_path']
        )
        if source_metadata['sha256'] != validation['source_m2_sha256']:
            raise ValueError(
                'Stage-2 recovery source M2 digest mismatch: '
                f"checkpoint={validation['source_m2_sha256']} "
                f"observed={source_metadata['sha256']}."
            )
        recorded_source = checkpoint.get('source_m2_checkpoint', {})
        if recorded_source and dict(recorded_source) != source_metadata:
            raise ValueError(
                'Stage-2 recovery source_m2_checkpoint binding is inconsistent.'
            )

        source_checkpoint = torch.load(
            source_metadata['path'], map_location='cpu'
        )
        source_validation = validate_stage1_m2_checkpoint(
            source_checkpoint,
            self.policy.ac.state_dict(),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
        )
        source_runtime_state = {
            name: value
            for name, value in source_checkpoint['model'].items()
            if name in self.policy.ac.state_dict()
        }
        source_runtime_summary = protected_parameter_summary(
            source_runtime_state,
            prefixes=self._resource_joint_protected_prefixes(),
        )
        if source_runtime_summary != validation['protected_summary']:
            raise ValueError(
                'Stage-2 recovery protected tensors no longer '
                'match the immutable Stage-1 M2 source.'
            )
        recorded_handoff = checkpoint.get('stage1_m2_checkpoint_summary')
        if (
            not isinstance(recorded_handoff, dict)
            or recorded_handoff.get('source_summary')
            != source_validation['source_summary']
        ):
            raise ValueError(
                'Stage-2 recovery is missing valid Stage-1 M2 hand-off evidence.'
            )

        self.policy.load_model_state(checkpoint['model'])
        restored_model_summary = protected_parameter_summary(
            self.policy.ac.state_dict(), prefixes=('',)
        )
        if restored_model_summary != validation['model_summary']:
            raise RuntimeError(
                'Strict Stage-2 loader did not reproduce every checkpoint '
                'model tensor bitwise.'
            )
        try:
            self.policy.actor_optimizer.load_state_dict(
                checkpoint['actor_optim']
            )
            self.policy.critic_optimizer.load_state_dict(
                checkpoint['critic_optim']
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                'Stage-2 recovery optimizer state is missing or incompatible.'
            ) from error
        if (
            bool(getattr(self.all_args, 'use_valuenorm', False))
            and not checkpoint.get('value_normalizer')
        ):
            raise ValueError(
                'Stage-2 recovery requires persisted ValueNorm statistics.'
            )

        self.policy.ac.tau = checkpoint.get('tau', self.policy.ac.tau)
        self.all_args.anneal_original = self.policy.ac.tau
        self.policy.actor_lr_multiplier = float(
            checkpoint.get('actor_lr_multiplier', 1.0)
        )
        self.policy.actor_lr_decay_factor = float(
            checkpoint.get('actor_lr_decay_factor', 1.0)
        )
        self.resource_wait_dual_value = float(checkpoint.get(
            'resource_wait_dual_value', self.resource_wait_dual_value
        ))
        self._pending_value_normalizer_state = checkpoint.get(
            'value_normalizer'
        )
        self._pending_role_value_normalizer_states = checkpoint.get(
            'role_value_normalizers'
        )
        self.stage1_m2_source = source_metadata
        self.stage1_m2_checkpoint_summary = recorded_handoff
        self.protected_parameter_summary_before_bc = checkpoint.get(
            'protected_parameter_summary_before_bc'
        )
        self.protected_parameter_summary_after_bc = checkpoint.get(
            'protected_parameter_summary_after_bc'
        )
        self.protected_parameter_summary_after_ppo = checkpoint.get(
            'protected_parameter_summary_after_ppo'
        )
        self.resource_actor_summary_before_bc = checkpoint.get(
            'resource_actor_summary_before_bc'
        )
        self.resource_actor_summary_after_bc = checkpoint.get(
            'resource_actor_summary_after_bc'
        )
        self.resource_actor_summary_before_ppo = checkpoint.get(
            'resource_actor_summary_before_ppo'
        )
        self.resource_actor_summary_after_ppo = checkpoint.get(
            'resource_actor_summary_after_ppo'
        )
        self.resource_bc_optimizer_reset = bool(
            checkpoint.get('resource_bc_optimizer_reset', False)
        )
        self.resource_bc_total_labels = int(
            checkpoint.get('resource_bc_total_labels', 0)
        )

        completed_shards = int(validation['completed_shards'])
        total_shards = int(validation['total_shards'])
        completed_epochs = int(validation['completed_epochs'])
        if completed_shards == total_shards:
            self.resume_epoch = completed_epochs
            self.resume_completed_shards = 0
        else:
            self.resume_epoch = max(0, completed_epochs - 1)
            self.resume_completed_shards = completed_shards
        self.resume_total_shards = total_shards
        self.resume_total_num_steps = int(validation['total_num_steps'])
        self.eval_epochs_without_improvement = int(
            checkpoint.get('eval_epochs_without_improvement', 0)
        )
        self.best_eval_makespan = float(
            checkpoint.get('best_eval_makespan', np.inf)
        )
        self.best_eval_iid_makespan = float(
            checkpoint.get('best_eval_iid_makespan', np.inf)
        )
        self.best_eval_composite_makespan = float(
            checkpoint.get('best_eval_composite_makespan', np.inf)
        )
        self._restore_actor_update_health(
            checkpoint.get('actor_update_health', {})
        )
        self.resource_joint_phase = 'resource_joint_ppo'
        self.exact_resume_stage2 = True
        print(
            '[Info] Restoring exact Stage-2 cursor from '
            f'epoch={self.resume_epoch + 1}, completed_shards='
            f'{self.resume_completed_shards}/{self.resume_total_shards}, '
            f'total_num_steps={self.resume_total_num_steps}, '
            f'checkpoint={checkpoint_path}.',
            flush=True,
        )
        return checkpoint

    def restore(self, checkpoint):
        """Restore policy's networks from a saved model."""
        checkpoint_path = checkpoint
        if self.training_stage == CANONICAL_RESOURCE_JOINT:
            if bool(getattr(self.all_args, 'resume_stage2', False)):
                # Explicit recovery accepts trusted full local training state,
                # including pre-portable NumPy RNG snapshots from canaries.
                # Keep Adam's non-capturable step counters on CPU; the optimizer
                # loader moves parameter moments to the parameter device itself.
                recovery = torch.load(
                    checkpoint_path, map_location='cpu', weights_only=False
                )
                if recovery.get('stage2_training_mode') == 'supervised_only':
                    self._restore_stage2_supervised_recovery(
                        checkpoint_path,
                        checkpoint=recovery,
                    )
                else:
                    self._restore_stage2_recovery(checkpoint_path)
            else:
                self._restore_stage1_m2(checkpoint_path)
            return
        if self.training_stage == CANONICAL_JOINT_FINETUNE:
            self._restore_stage2_joint_finetune(checkpoint_path)
            return
        checkpoint = torch.load(checkpoint, map_location=self.device)
        print(
            f"[Info] Loading checkpoint {checkpoint_path} "
            f"(training_stage={checkpoint.get('training_stage', 'legacy')})."
        )
        contract = validate_stage1_checkpoint_contract(
            checkpoint,
            global_feature_mode=getattr(
                self.all_args, 'global_feature_mode', 'none'
            ),
            plane_order_mode=self.policy.ac.plane_order_mode,
            plane_pair_decoder=self.policy.ac.plane_pair_decoder,
            stage1_baseline=self.policy.ac.stage1_baseline,
            strict_metadata=self.strict_checkpoint_contract,
        )
        print(
            '[Info] Stage-1 checkpoint observation contract accepted: '
            f"mode={contract['global_feature_mode']} "
            f"schema={contract['observation_schema_id']} "
            f"strict={contract['strict_metadata']}.",
            flush=True,
        )
        self.policy.load_model_state(checkpoint['model'])
        self.policy.ac.tau = checkpoint.get('tau', self.policy.ac.tau)
        self.all_args.anneal_original = self.policy.ac.tau
        if self.reset_value_normalizer_on_resume:
            self._pending_value_normalizer_state = None
            self._pending_role_value_normalizer_states = None
            print(
                '[Info] Discarded checkpoint ValueNorm statistics '
                '(--reset_value_normalizer_on_resume).',
                flush=True,
            )
        else:
            self._pending_value_normalizer_state = checkpoint.get(
                'value_normalizer'
            )
            self._pending_role_value_normalizer_states = checkpoint.get(
                'role_value_normalizers'
            )
        if self.reset_optimizers_on_resume:
            print(
                "[Info] Restored model weights with fresh actor and critic optimizers "
                "(--reset_optimizers_on_resume)."
            )
        else:
            try:
                self.policy.actor_optimizer.load_state_dict(checkpoint['actor_optim'])
                self.policy.critic_optimizer.load_state_dict(checkpoint['critic_optim'])
                experiment_config = checkpoint.get('experiment_config', {})
                self.policy.actor_lr_multiplier = float(checkpoint.get(
                    'actor_lr_multiplier',
                    experiment_config.get('actor_lr_multiplier', 1.0),
                ))
                self.policy.actor_lr_decay_factor = float(checkpoint.get(
                    'actor_lr_decay_factor', 1.0
                ))
            except (KeyError, ValueError):
                print("[Warning] Optimizer state is incompatible with the current mixed plane-device policy; using fresh optimizers.")
        exact_resume = (
            bool(getattr(self.all_args, 'resume_stage1', False))
            and not self.reset_optimizers_on_resume
            and checkpoint.get('stage') == 'post_shard_recovery'
        )
        if exact_resume:
            total_shards = int(checkpoint.get('total_shards', 0))
            completed_shards = int(checkpoint.get('completed_shard', 0))
            checkpoint_episodes = int(checkpoint.get('episodes', 0))
            if total_shards <= 0 or not 0 < completed_shards <= total_shards:
                raise ValueError(
                    "Post-shard Recovery checkpoint has an invalid resume cursor: "
                    f"completed={completed_shards}, total={total_shards}."
                )
            if completed_shards == total_shards:
                self.resume_epoch = checkpoint_episodes
                self.resume_completed_shards = 0
            else:
                self.resume_epoch = max(0, checkpoint_episodes - 1)
                self.resume_completed_shards = completed_shards
            self.resume_total_shards = total_shards
            self.resume_total_num_steps = int(checkpoint.get('total_num_steps', 0))
            self.eval_epochs_without_improvement = int(
                checkpoint.get('eval_epochs_without_improvement', 0)
            )
            restored_best = float(
                checkpoint.get('best_eval_makespan', np.inf)
            )
            best_path = os.path.join(
                os.path.dirname(str(checkpoint_path)), 'checkpoint_Best.pt'
            )
            if not np.isfinite(restored_best) and os.path.isfile(best_path):
                best_checkpoint = torch.load(best_path, map_location='cpu')
                restored_best = float(
                    best_checkpoint.get('eval_makespan', np.inf)
                )
            self.best_eval_makespan = restored_best
            self.best_eval_iid_makespan = float(
                checkpoint.get('best_eval_iid_makespan', np.inf)
            )
            self.best_eval_composite_makespan = float(
                checkpoint.get('best_eval_composite_makespan', np.inf)
            )
            self._restore_actor_update_health(
                checkpoint.get('actor_update_health', {})
            )
            self.exact_resume_stage1 = True
            print(
                "[Info] Restoring exact Stage-1 cursor from "
                f"epoch={self.resume_epoch + 1}, completed_shards="
                f"{self.resume_completed_shards}/{self.resume_total_shards}, "
                f"total_num_steps={self.resume_total_num_steps}."
            )
        # self.episode = checkpoint['episodes']
        # self.num_episodes = self.num_episodes - checkpoint['episodes']

    def _copy_resume_artifacts(self):
        """Keep BC and best-selection artifacts in the new run directory."""
        if self.resume_total_shards != self.num_envs:
            raise ValueError(
                "Recovery checkpoint shard geometry does not match the resumed "
                f"run: checkpoint={self.resume_total_shards}, "
                f"current={self.num_envs}. Keep --n_rollout_threads unchanged."
            )
        source_dir = os.path.dirname(str(self.checkpoint_dir))
        for filename in (
            'checkpoint_PlaneBC.pt',
            'checkpoint_DeviceBC.pt',
            'checkpoint_PrePPO.pt',
            'checkpoint_PrePPO_tau03.pt',
            'checkpoint_Best.pt',
            'checkpoint_Best_IID.pt',
            'checkpoint_Best_Composite.pt',
        ):
            source = os.path.join(source_dir, filename)
            target = os.path.join(self.save_dir, filename)
            if os.path.isfile(source) and not os.path.exists(target):
                shutil.copy2(source, target)
        if self.exact_resume_stage2:
            source_evaluations = os.path.join(
                os.path.dirname(source_dir), 'evaluations'
            )
            if os.path.isdir(source_evaluations):
                for source in Path(source_evaluations).glob('*.json'):
                    target = Path(self.evaluation_dir) / source.name
                    if not target.exists():
                        shutil.copy2(source, target)
