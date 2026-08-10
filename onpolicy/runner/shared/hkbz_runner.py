import time
import os
import json
import numpy as np
import torch
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
import uuid
from types import SimpleNamespace
from pathlib import Path

from onpolicy.utils.shared_eval import SharedEvalClient, parse_cpu_set

def _t2n(x):
    return x.detach().cpu().numpy()

class HKBZ_Runner(Runner):
    """Runner class to perform training, evaluation. and data collection for the IAs. See parent class for details."""
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
        self.graphs_per_forward = self._validate_graph_batch_memory_config(
            self.all_args.mini_batch_size,
            self.all_args.data_chunk_length,
            self.max_graphs_per_forward,
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
        self.training_stage = str(getattr(self.all_args, 'training_stage', 'auto'))
        self.hindsight_reward_mode = str(
            getattr(self.all_args, 'hindsight_reward_mode', '')
        )
        self.team_return_mode = self.hindsight_reward_mode in {
            'team_cmax', 'team_time'
        }
        self.team_time_return_mode = self.hindsight_reward_mode == 'team_time'
        self.recovery_checkpoint_interval_shards = max(
            0,
            int(getattr(self.all_args, 'recovery_checkpoint_interval_shards', 1)),
        )
        self.progress_callback = config.get('progress_callback')
        self.current_epoch = -1
        self.current_shard = -1
        self.last_team_cmax_mean = 0.0
        self.last_team_return_mean_raw = 0.0
        self.last_team_cycle_count = 0
        self.plane_bc_pretrain_epochs = int(
            getattr(self.all_args, 'plane_bc_pretrain_epochs', 0)
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
        self.bc_reference_kl_coef_schedule = self._parse_epoch_schedule(
            getattr(self.all_args, 'bc_reference_kl_coef_schedule', ''),
            'bc_reference_kl_coef_schedule',
        )
        self.iga_potential_beta_schedule = self._parse_epoch_schedule(
            getattr(self.all_args, 'iga_potential_beta_schedule', ''),
            'iga_potential_beta_schedule',
        )
        self.bc_reference_target_kl = float(
            getattr(self.all_args, 'bc_reference_target_kl', 0.0)
        )
        self.bc_reference_hard_gate = bool(
            getattr(self.all_args, 'bc_reference_hard_gate', False)
        )
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
        self.device_bc_lr = float(getattr(self.all_args, 'device_bc_lr', 0.0))
        self.device_bc_min_labels_per_epoch = int(getattr(self.all_args, 'device_bc_min_labels_per_epoch', 0))
        self.device_bc_min_rollouts_per_epoch = int(getattr(self.all_args, 'device_bc_min_rollouts_per_epoch', 1))
        self.device_bc_max_rollouts_per_epoch = int(getattr(self.all_args, 'device_bc_max_rollouts_per_epoch', 0))
        self.device_bc_train_gnn = bool(getattr(self.all_args, 'device_bc_train_gnn', False))
        self.device_bc_plane_deterministic = bool(getattr(self.all_args, 'device_bc_plane_deterministic', True))
        self.device_bc_reset_optim = bool(getattr(self.all_args, 'device_bc_reset_optim', True))
        self.device_bc_save = bool(getattr(self.all_args, 'device_bc_save', True))
        self.gnn_freeze_epochs = max(0, int(getattr(self.all_args, 'gnn_freeze_epochs', 0)))
        self.plane_freeze_epochs = max(0, int(getattr(self.all_args, 'plane_freeze_epochs', 0)))
        self.plane_order_freeze_epochs = max(
            0, int(getattr(self.all_args, 'plane_order_freeze_epochs', 0))
        )
        self.early_stop_patience = max(0, int(getattr(self.all_args, 'early_stop_patience', 0)))
        self.reset_optimizers_on_resume = bool(getattr(self.all_args, 'reset_optimizers_on_resume', False))
        self.canary_eval_interval_shards = int(getattr(self.all_args, 'canary_eval_interval_shards', 0))
        self.canary_max_regression = float(getattr(self.all_args, 'canary_max_regression', 0.0))
        self.canary_stop_on_regression = bool(getattr(self.all_args, 'canary_stop_on_regression', False))
        self.canary_rejected = False
        self.canary_rejection_info = {}
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
        if self.evaluation_tau <= 0.0:
            raise ValueError("--evaluation_tau must be positive.")
        if self.selection_metric not in {'iid', 'composite'}:
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
        self.best_eval_makespan = np.inf
        self.best_eval_iid_makespan = np.inf
        self.best_eval_composite_makespan = np.inf
        self.eval_epochs_without_improvement = 0
        self.exact_resume_stage1 = False
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
        self._pending_value_normalizer_state = None
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
        self.last_eval_selection_score = np.inf
        self.eval_canary_rounds = max(
            1,
            int(getattr(self.all_args, 'eval_canary_rounds', 1)),
        )

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
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

        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        if self._pending_value_normalizer_state is not None and self.trainer.value_normalizer is not None:
            if len(self._pending_value_normalizer_state) == 0:
                print(
                    "[Warning] Loaded a legacy checkpoint without persisted ValueNorm "
                    "statistics; starting ValueNorm from a fresh state."
                )
            else:
                self.trainer.value_normalizer.load_state_dict(self._pending_value_normalizer_state)
                print("[Info] Restored persisted ValueNorm statistics from checkpoint.")
        
        # buffer
        if self.evaluation_only:
            # Deterministic evaluation only needs the recurrent-state tail
            # shape.  Avoid allocating the multi-gigabyte PPO replay buffer in
            # the persistent per-GPU evaluator.
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

    def _validate_training_stage_config(self):
        resource_policy = getattr(self.all_args, 'resource_policy', 'heuristic')
        stage = self.training_stage
        if (
            self.reset_optimizers_on_resume
            and not bool(getattr(self.all_args, 'resume_stage1', False))
        ):
            raise ValueError(
                "--reset_optimizers_on_resume requires --resume_stage1."
            )
        if self.plane_bc_pretrain_epochs < 0:
            raise ValueError("--plane_bc_pretrain_epochs must be non-negative.")
        if self.plane_bc_rollouts_per_epoch < 0:
            raise ValueError("--plane_bc_rollouts_per_epoch must be non-negative.")
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
                raise ValueError("IGA plane BC is only valid for plane_pretrain.")
            if not self.plane_bc_teacher_dir:
                raise ValueError(
                    "--plane_bc_teacher_dir is required when plane BC is enabled."
                )
            if not os.path.isdir(self.plane_bc_teacher_dir):
                raise ValueError(
                    f"IGA teacher directory does not exist: {self.plane_bc_teacher_dir}"
                )
        if self.canary_eval_interval_shards < 0:
            raise ValueError("--canary_eval_interval_shards must be non-negative.")
        if self.canary_max_regression < 0.0:
            raise ValueError("--canary_max_regression must be non-negative.")
        if self.canary_eval_interval_shards > 0 and not self.use_eval:
            raise ValueError("shard canary evaluation requires --use_eval.")
        if self.canary_stop_on_regression and self.canary_eval_interval_shards <= 0:
            raise ValueError(
                "--canary_stop_on_regression requires "
                "--canary_eval_interval_shards > 0."
            )
        if stage == 'auto':
            return
        if stage == 'plane_pretrain':
            if resource_policy != 'heuristic':
                raise ValueError("plane_pretrain requires resource_policy='heuristic'.")
            resume_stage1 = bool(getattr(self.all_args, 'resume_stage1', False))
            if self.checkpoint_dir is not None and not resume_stage1:
                raise ValueError(
                    "plane_pretrain checkpoint restore requires explicit --resume_stage1."
                )
            if resume_stage1 and self.checkpoint_dir is None:
                raise ValueError("--resume_stage1 requires --checkpoint_dir.")
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
        if stage == 'device_bc':
            if resource_policy != 'drl':
                raise ValueError("device_bc requires resource_policy='drl'.")
            if self.checkpoint_dir is None:
                raise ValueError("device_bc requires the new plane checkpoint.")
            if self.device_bc_pretrain_epochs <= 0:
                raise ValueError("device_bc requires --device_bc_pretrain_epochs > 0.")
            if self.num_episodes != 0:
                raise ValueError("device_bc is an isolated stage and requires --num_episodes 0.")
            return
        if stage == 'frozen_joint':
            if resource_policy != 'drl' or self.checkpoint_dir is None:
                raise ValueError("frozen_joint requires DRL resources and the device BC checkpoint.")
            if self.device_bc_pretrain_epochs != 0:
                raise ValueError("frozen_joint cannot run device BC again.")
            if self.num_episodes <= 0:
                raise ValueError("frozen_joint requires at least one PPO epoch.")
            if (
                self.gnn_freeze_epochs < self.num_episodes
                or self.plane_freeze_epochs < self.num_episodes
            ):
                raise ValueError(
                    "frozen_joint must freeze the shared encoder and plane actor "
                    "for every epoch in this stage."
                )
            return
        if stage == 'full_joint':
            if resource_policy != 'drl' or self.checkpoint_dir is None:
                raise ValueError("full_joint requires DRL resources and the frozen-joint checkpoint.")
            if self.device_bc_pretrain_epochs != 0:
                raise ValueError("full_joint cannot run device BC again.")
            if self.gnn_freeze_epochs != 0 or self.plane_freeze_epochs != 0:
                raise ValueError("full_joint must start with all policy modules unfrozen.")
            if self.num_episodes <= 0:
                raise ValueError("full_joint requires at least one PPO epoch.")

    @staticmethod
    def _validate_graph_batch_memory_config(mini_batch_size, data_chunk_length, max_graphs):
        mini_batch_size = int(mini_batch_size)
        data_chunk_length = int(data_chunk_length)
        max_graphs = int(max_graphs)
        if mini_batch_size <= 0 or data_chunk_length <= 0:
            raise ValueError("mini_batch_size and data_chunk_length must both be positive.")
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
        self.actor_post_update_old_policy_kl_max = float(
            health.get('post_update_old_policy_kl_max', 0.0)
        )
        self.actor_post_update_bc_reference_kl_max = float(
            health.get('post_update_bc_reference_kl_max', 0.0)
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
        case_dir = os.path.basename(normalized_path)
        split_name = os.path.basename(os.path.dirname(normalized_path))
        path_key = f'{split_name}/{case_dir}'
        metadata = dict(self.dataset_case_metadata.get(path_key, {}))
        metadata.setdefault('case_id', f"{split_name}_{case_dir.rsplit('_', 1)[-1]}")
        metadata.setdefault('split', split_name)
        metadata['case_dir'] = case_dir
        metadata['case_path'] = normalized_path
        metadata['case_key'] = path_key
        return metadata

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
        """Derive IID/composite scores without allowing test cases to select."""
        raw_makespan = float(raw_makespan)
        self.last_eval_raw_makespan = raw_makespan
        if not np.isfinite(raw_makespan):
            self.last_eval_iid_makespan = np.inf
            self.last_eval_composite_makespan = np.inf
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

        if self.selection_metric == 'composite':
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
            'eval_selection_score': self.last_eval_selection_score,
            'best_eval_iid_makespan': self.best_eval_iid_makespan,
            'best_eval_composite_makespan': (
                self.best_eval_composite_makespan
            ),
        }
        info.update(self._evaluation_group_metrics())
        return info

    def _report_progress(self, event, **extra):
        callback = self.progress_callback
        if callback is None:
            return
        payload = {
            'event': str(event),
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
        best_makespan = float(self.best_eval_makespan)
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
            canary_eval_makespan=canary_makespan,
            best_eval_makespan=best_makespan,
            canary_relative_regression=float(relative_regression),
            canary_rejection_threshold=float(rejection_threshold),
            canary_rejected=bool(rejected),
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
        if not self.checkpoint_dir:
            raise RuntimeError(
                'BC-reference PPO is enabled but no plane BC snapshot is available.'
            )
        checkpoint_path = str(self.checkpoint_dir)
        candidates = []
        if os.path.basename(checkpoint_path) == 'checkpoint_PlaneBC.pt':
            candidates.append(checkpoint_path)
        candidates.append(
            os.path.join(os.path.dirname(checkpoint_path), 'checkpoint_PlaneBC.pt')
        )
        reference_path = next(
            (candidate for candidate in candidates if os.path.isfile(candidate)),
            None,
        )
        if reference_path is None:
            raise RuntimeError(
                'BC-reference PPO requires checkpoint_PlaneBC.pt next to the '
                f'restored checkpoint; checked {candidates}.'
            )
        payload = torch.load(reference_path, map_location='cpu')
        self.policy.capture_bc_reference(payload['model'])
        local_reference_path = os.path.join(
            self.save_dir, 'checkpoint_PlaneBC.pt'
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
        print(f'[Info] Restored frozen BC reference from {reference_path}.')

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
        bc_kl_coef = self._epoch_schedule_value(
            self.bc_reference_kl_coef_schedule,
            episode,
            getattr(self.all_args, 'bc_reference_kl_coef', 0.0),
        )
        observed = self.envs.call('set_iga_potential_beta', potential_beta)
        if any(not np.isclose(float(value), potential_beta) for value in observed):
            raise RuntimeError('Training workers rejected potential-beta schedule.')
        self.bc_reference_kl_coef = bc_kl_coef
        self.trainer.bc_reference_kl_coef = bc_kl_coef
        self._report_progress(
            'epoch_method_schedule_applied',
            iga_potential_beta=float(potential_beta),
            bc_reference_kl_coef=float(bc_kl_coef),
        )
        print(
            f'[MethodSchedule] epoch={episode + 1} '
            f'iga_potential_beta={potential_beta:.6g} '
            f'bc_reference_kl_coef={bc_kl_coef:.6g}.',
            flush=True,
        )

    def run(self):   
        
        start = time.time()
        episodes = self.num_episodes
        self.total_num_steps = int(self.resume_total_num_steps)
        if self.exact_resume_stage1:
            self._copy_resume_artifacts()
            self._report_progress(
                'training_resumed',
                total_epochs=int(episodes),
                resume_epoch=int(self.resume_epoch),
                resume_completed_shards=int(self.resume_completed_shards),
                total_num_steps=int(self.total_num_steps),
            )
        else:
            self._report_progress('training_started', total_epochs=int(episodes))

        if self.plane_bc_pretrain_epochs > 0:
            self.plane_bc_pretrain()

        if self.device_bc_pretrain_epochs > 0:
            self.device_bc_pretrain()

        self._ensure_bc_reference_policy()

        # Establish a pre-PPO baseline so checkpoint selection cannot silently
        # discard a stronger loaded/BC-warmed policy after the first update.
        if self.use_eval and not self.exact_resume_stage1:
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
                    eval_makespan=float(baseline_makespan),
                    best_eval_makespan=float(self.best_eval_makespan),
                )
                self._seed_best_from_selection_checkpoint()

        first_episode = int(self.resume_epoch) if self.exact_resume_stage1 else 0
        pbar = tqdm(range(first_episode, episodes),
              desc="Training",    
              unit="episode",     
              total=episodes,       
              initial=first_episode,
              ncols=160)
        for episode in pbar:
            # profiler = cProfile.Profile()
            # profiler.enable()
            self.envs.shuffer_data()
            self.episode = episode
            self.current_epoch = episode
            self.current_shard = -1
            self._apply_epoch_method_schedules(episode)
            first_shard = (
                int(self.resume_completed_shards)
                if self.exact_resume_stage1 and episode == first_episode
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
                self._reset_cuda_peak_memory()
                self.compute()

                global_shard_idx = episode * self.num_envs + shard_idx
                actor_update_enabled = global_shard_idx >= self.actor_warmup_shards
                train_infos = self.train(update_actor=actor_update_enabled)
                lr_decision = self._actor_lr_adaptation_decision(
                    train_infos.get(
                        'actor_planned_optimizer_steps', 0.0
                    ),
                    train_infos.get('actor_optimizer_steps', 0.0),
                    train_infos.get('actor_kl_stop_reason_code', 0.0),
                    self.adaptive_actor_min_step_completion,
                )
                train_infos.update(lr_decision)
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
                if actor_update_enabled:
                    self._record_actor_update_health(train_infos)
                train_infos['actor_warmup_active'] = float(not actor_update_enabled)
                train_infos['shared_encoder_frozen'] = float(freeze_shared)
                train_infos['plane_order_frozen'] = float(freeze_order)
                train_infos['global_shard_index'] = float(global_shard_idx + 1)
                train_infos['policy_tau'] = float(self.policy.ac.tau)
                self._record_and_clear_cuda_memory(train_infos, episode, shard_idx)
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
                    actor_planned_optimizer_steps=float(
                        train_infos.get(
                            'actor_planned_optimizer_steps', 0.0
                        )
                    ),
                    actor_step_completion_rate=float(
                        train_infos.get('actor_step_completion_rate', 1.0)
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
                    post_update_probe_approx_kl=float(
                        train_infos.get('post_update_probe_approx_kl', 0.0)
                    ),
                    post_update_probe_clip_fraction=float(
                        train_infos.get('post_update_probe_clip_fraction', 0.0)
                    ),
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
                )
                if canary_due:
                    canary_requested_stop = self._run_shard_canary(
                        episode,
                        completed_shards,
                    )
                    if canary_requested_stop:
                        break

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
                    eval_makespan=float(eval_makespan),
                    best_eval_makespan=float(self.best_eval_makespan),
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

            # log information
            # if episode % self.log_interval == 0:
                # env_infos = {}
                # self.log_train(train_infos, total_num_steps * self.envs.num_fields)
                # self.log_env(env_infos, total_num_steps)

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
            self.last_team_return_mean_raw = float(team_returns_raw.mean())
            self.last_team_cycle_count = int(
                sum(bool(objective['cycle_terminated']) for objective in objectives)
            )
            self.buffer.set_team_cmax_values(cmax_values)
            scaled_team_returns = team_returns_raw * self.reward_coef
            if self.team_time_return_mode:
                self.buffer.compute_team_time_returns(
                    scaled_team_returns,
                    cmax_values,
                    float(self.all_args.hindsight_terminal_cmax_coef)
                    * self.reward_coef,
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

    def train(self, update_actor=True):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, update_actor=update_actor)
        return train_infos

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

        self.buffer.rnn_states[0] = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        
        self.buffer.masks[0] = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)
        self.buffer.masks[0][dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.active_masks[0] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)
        self.buffer.active_masks[0][infos['active_agents'] == True] = np.ones(((infos['active_agents'] == True).sum(), 1), dtype=np.float32)

    def _active_masks_from_info(self, infos):
        active_masks = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[infos['active_agents'] == True] = 1.0
        return active_masks

    def _dagger_teacher_rate(self, epoch):
        if not self.plane_bc_dagger_schedule:
            return 1.0
        return float(self.plane_bc_dagger_schedule[
            min(int(epoch), len(self.plane_bc_dagger_schedule) - 1)
        ])

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
        ).reshape(policy_actions.shape[0])
        available_count = 0
        active_plane_labels = 0
        pair_correct = 0
        order_correct = 0
        order_labels = 0
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
            plane_count = self.policy.ac.max_plane_agents
            labels[env_idx, :plane_count, :action_width] = teacher_actions[
                :plane_count, :action_width
            ]
            if teacher_execution_mask[env_idx]:
                actions[env_idx, :plane_count, :action_width] = teacher_actions[
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
            'teacher_executed_envs': int(teacher_execution_mask.sum()),
            'student_executed_envs': int((~teacher_execution_mask).sum()),
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
        stratum_weights_np = np.ones_like(
            label_sites_np, dtype=np.float32
        )
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
        # Each case receives unit total mass regardless of its number of
        # teacher decisions; within a case the important strata retain their
        # requested relative multipliers.
        pair_weight = pair_weight / pair_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
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
        def category_mask(mask_np):
            result = torch.zeros_like(pair_mask, dtype=torch.bool)
            result[:, :plane_count] = torch.as_tensor(
                mask_np, dtype=torch.bool, device=self.device
            )
            return result & pair_mask

        initial_mask = category_mask(initial_np)
        relocation_mask = category_mask(relocation_np)
        same_site_mask = category_mask(same_site_np)
        critical_mask = category_mask(critical_np)
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
        if getattr(self.all_args, 'resource_policy', 'heuristic') != 'heuristic':
            raise RuntimeError('IGA plane BC requires heuristic resource policy.')
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
                        teacher_rpc_pending = False
                        if self.safe_dagger_teacher_overlap:
                            self.envs.call_async(
                                'iga_teacher_actions', return_info=True
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
                                        last_actions[..., 0],
                                        last_actions[..., 1],
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
                                'iga_teacher_actions', return_info=True
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
                        execution_rates = self._dagger_execution_rates(
                            teacher_rate, progress
                        )
                        teacher_execution_mask = (
                            self.plane_bc_dagger_rng.random(
                                self.n_rollout_threads
                            ) < execution_rates
                        )
                        teacher_execution_mask &= ~done_flags
                        actions, labels, label_stats = self._merge_plane_bc_actions(
                            _t2n(policy_actions),
                            teacher_results,
                            teacher_execution_mask=teacher_execution_mask,
                        )
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
                            last_actions,
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
                            teacher_execution_mask.sum()
                        )
                        epoch_student_executed_envs += int(
                            ((~teacher_execution_mask) & (~done_flags)).sum()
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
                    plane_bc_shared_frozen=bool(freeze_shared),
                )
        finally:
            self._restore_requires_grad(previous_requires_grad)
            self.policy.ac.train()
            self._plane_bc_case_ids = None
        self.policy.reset_optimizers()
        self.trainer.policy = self.policy
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
        modules = [
            self.policy.ac.device_sel_enc,
            self.policy.ac.transporter_sel_enc,
            self.policy.ac.device_actor,
            self.policy.ac.transporter_actor,
        ]
        if self.device_bc_train_gnn:
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

    def _merge_device_bc_actions(self, policy_actions, label_results):
        actions = policy_actions.astype(np.int64, copy=True)
        label_actions = policy_actions.astype(np.int64, copy=True)
        stats = {
            'preferred_assignments': 0,
            'greedy_legal_fills': 0,
            'real_dispatches': 0,
            'noop_after_claimed': 0,
            'deferred_noop': 0,
            'non_unique_candidate_masks': 0,
        }
        for env_idx, result in enumerate(label_results):
            env_actions = result['actions'] if isinstance(result, dict) else result
            env_info = result.get('info', {}) if isinstance(result, dict) else {}
            env_actions = np.asarray(env_actions, dtype=np.int64)
            device_slice = slice(self.policy.ac.max_plane_agents, self.num_agents)
            actions[env_idx, device_slice, :2] = env_actions[device_slice, :2]
            label_actions[env_idx, device_slice, :2] = env_actions[device_slice, :2]
            for key in stats:
                stats[key] += int(env_info.get(key, 0))
        return actions, label_actions, stats

    def _device_bc_update(self, obs, rnn_states, active_masks, last_actions, label_actions, infos, optimizer):
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
            return None

        action_log_probs, _, decision_mask = self.policy.evaluate_actions(
            Batch.from_data_list(obs),
            rnn_states,
            active_masks,
            last_actions[..., 0],
            last_actions[..., 1],
            label_actions,
            agent_types=agent_types,
            return_decision_mask=True,
        )
        device_mask = torch.as_tensor(device_mask_np, dtype=torch.bool, device=self.device)
        # Forced no-op labels after another device claimed the request have a
        # single legal action and exactly zero gradient. Counting them in the
        # denominator diluted real dispatch supervision by up to 20-80x.
        device_mask &= decision_mask.bool()
        label_count = int(device_mask.sum().item())
        if label_count == 0:
            return None
        device_mask = device_mask.float()
        loss = -(action_log_probs * device_mask).sum() / device_mask.sum().clamp_min(1.0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Device BC loss is not finite: {loss.item()}.")

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for group in optimizer.param_groups for param in group['params']],
            self.all_args.max_grad_norm,
        )
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad()
            raise RuntimeError(f"Device BC grad norm is not finite: {grad_norm.item()}.")
        optimizer.step()

        return {
            'device_bc_loss': float(loss.detach().cpu().item()),
            'device_bc_grad_norm': float(grad_norm.detach().cpu().item()),
            'device_bc_labels': label_count,
        }

    def device_bc_pretrain(self):
        if getattr(self.all_args, 'resource_policy', 'heuristic') != 'drl':
            raise RuntimeError("Device BC pretraining requires resource_policy='drl'.")
        if self.checkpoint_dir is None:
            raise RuntimeError("Device BC pretraining requires --checkpoint_dir for the pretrained plane policy.")
        if not hasattr(self.envs, 'call'):
            raise RuntimeError("Device BC pretraining requires vector env call() support.")

        trainable_modules = self._device_bc_trainable_modules()
        trainable_params = [param for module in trainable_modules for param in module.parameters()]
        if not trainable_params:
            raise RuntimeError("Device BC pretraining found no trainable parameters.")

        previous_requires_grad = self._set_device_bc_requires_grad(trainable_params)
        bc_lr = self.device_bc_lr if self.device_bc_lr > 0.0 else self.all_args.lr
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=bc_lr,
            eps=self.all_args.opti_eps,
            weight_decay=self.all_args.weight_decay,
        )
        bc_step = 0

        try:
            self.policy.ac.eval()
            for module in trainable_modules:
                module.train()

            pbar = tqdm(
                range(self.device_bc_pretrain_epochs),
                desc="DeviceBC",
                unit="epoch",
                total=self.device_bc_pretrain_epochs,
                ncols=160,
            )
            for epoch in pbar:
                epoch_info = {
                    'device_bc_loss': 0.0,
                    'device_bc_grad_norm': 0.0,
                    'device_bc_labels': 0,
                    'preferred_assignments': 0,
                    'greedy_legal_fills': 0,
                    'real_dispatches': 0,
                    'noop_after_claimed': 0,
                    'deferred_noop': 0,
                    'non_unique_candidate_masks': 0,
                }
                update_count = 0
                rollout_count = 0
                epoch_label_total = 0
                min_rollouts, max_rollouts = self._resolve_bc_rollout_limits(
                    self.device_bc_min_rollouts_per_epoch,
                    self.device_bc_max_rollouts_per_epoch,
                    self.num_envs,
                )
                min_labels = max(0, self.device_bc_min_labels_per_epoch)

                while rollout_count < max_rollouts:
                    obs, dones, infos = self.envs.reset()
                    rnn_states = np.zeros(
                        (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                        dtype=np.float32,
                    )
                    last_actions = -np.ones((self.n_rollout_threads, self.num_agents, 2), dtype=np.int64)
                    env_done_flags = np.zeros(self.n_rollout_threads, dtype=bool)

                    for step in range(self.episode_length):
                        active_masks = self._active_masks_from_info(infos)
                        with torch.no_grad():
                            _, policy_actions, _, next_rnn_states = self.policy.get_actions(
                                Batch.from_data_list(obs),
                                rnn_states,
                                active_masks,
                                last_actions[..., 0],
                                last_actions[..., 1],
                                deterministic=self.device_bc_plane_deterministic,
                                agent_types=infos.get('agent_types', None),
                            )
                        policy_actions = _t2n(policy_actions)
                        next_rnn_states = _t2n(next_rnn_states)

                        label_results = self.envs.call('heuristic_device_actions', return_info=True)
                        actions, label_actions, label_stats = self._merge_device_bc_actions(
                            policy_actions,
                            label_results,
                        )

                        update_info = self._device_bc_update(
                            obs,
                            rnn_states,
                            active_masks,
                            last_actions,
                            label_actions,
                            infos,
                            optimizer,
                        )
                        if update_info is not None:
                            update_count += 1
                            bc_step += 1
                            epoch_label_total += int(update_info['device_bc_labels'])
                            for key, value in update_info.items():
                                epoch_info[key] += value
                            if not self.use_wandb:
                                self.writter.add_scalar('device_bc/loss', update_info['device_bc_loss'], bc_step)
                                self.writter.add_scalar('device_bc/labels', update_info['device_bc_labels'], bc_step)

                        for key, value in label_stats.items():
                            epoch_info[key] += value

                        obs, _, dones, infos = self.envs.step(actions)
                        next_rnn_states[dones == True] = np.zeros(
                            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                            dtype=np.float32,
                        )
                        rnn_states = next_rnn_states
                        last_actions = actions

                        env_done_flags = np.all(dones, axis=1)
                        if self.rollout_until_done and np.all(env_done_flags):
                            break

                    if self.rollout_until_done and not np.all(env_done_flags):
                        unfinished_envs = np.where(~env_done_flags)[0]
                        raise RuntimeError(
                            "Device BC rollout did not finish within "
                            f"{self.episode_length} decision steps; unfinished_envs={unfinished_envs.tolist()}."
                        )
                    rollout_count += 1
                    if rollout_count >= min_rollouts and epoch_label_total >= min_labels:
                        break

                if update_count > 0:
                    epoch_info['device_bc_loss'] /= update_count
                    epoch_info['device_bc_grad_norm'] /= update_count
                    epoch_info['device_bc_labels'] /= update_count
                epoch_info['device_bc_total_labels'] = epoch_label_total
                if not self.use_wandb:
                    for key, value in epoch_info.items():
                        self.writter.add_scalar(f'device_bc_epoch/{key}', value, epoch + 1)
                pbar.set_postfix(
                    loss=epoch_info['device_bc_loss'],
                    labels=epoch_label_total,
                    non_unique=epoch_info['non_unique_candidate_masks'],
                    dispatches=epoch_info['real_dispatches'],
                    rollouts=rollout_count,
                )
        finally:
            self._restore_requires_grad(previous_requires_grad)
            self.policy.ac.train()

        if self.device_bc_reset_optim:
            self.policy.reset_optimizers()
            self.trainer.policy = self.policy
            print("[Info] Reset PPO optimizers after device BC pretraining.")
        if self.device_bc_save:
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
        request_id = (
            f'{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        )
        request_dir = os.path.join(str(self.run_dir), 'shared_eval_requests')
        os.makedirs(request_dir, exist_ok=True)
        checkpoint_path = os.path.join(request_dir, f'{request_id}.pt')
        checkpoint = {
            'protocol_version': 1,
            'request_id': request_id,
            'model': {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.policy.ac.state_dict().items()
            },
            'plane_order_mode': self.policy.ac.plane_order_mode,
            'plane_pair_decoder': self.policy.ac.plane_pair_decoder,
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
                'n_eval_rollout_threads': int(self.n_eval_rollout_threads),
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

                eval_action, eval_rnn_states = self.trainer.policy.act(
                    Batch.from_data_list(eval_obs),
                    eval_rnn_states,
                    eval_active_masks,
                    eval_actions[..., 0] if eval_actions is not None else -np.ones((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32),
                    eval_actions[..., 1] if eval_actions is not None else -np.ones((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32),
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

        if str(evaluation_label) == 'pre_ppo':
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
        checkpoint = {
            'episodes': episode + 1,
            'tau': model.ac.tau,
            'training_stage': self.training_stage,
            'plane_order_mode': model.ac.plane_order_mode,
            'plane_pair_decoder': model.ac.plane_pair_decoder,
            'actor_lr_multiplier': float(model.actor_lr_multiplier),
            'actor_lr_decay_factor': float(model.actor_lr_decay_factor),
            'experiment_config': {
                'hindsight_reward_mode': str(
                    getattr(self.all_args, 'hindsight_reward_mode', '')
                ),
                'lr': float(self.all_args.lr),
                'critic_lr': float(self.all_args.critic_lr),
                'shared_actor_lr_scale': float(
                    self.all_args.shared_actor_lr_scale
                ),
                'ppo_epoch': int(self.all_args.ppo_epoch),
                'actor_grad_accumulation_steps': int(
                    self.all_args.actor_grad_accumulation_steps
                ),
                'entropy_coef': float(self.all_args.entropy_coef),
                'max_grad_norm': float(self.all_args.max_grad_norm),
                'bc_reference_kl_coef': float(
                    self.bc_reference_kl_coef
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
                'global_feature_mode': str(getattr(
                    self.all_args, 'global_feature_mode', 'none'
                )),
                'gnn_freeze_epochs': int(self.gnn_freeze_epochs),
                'plane_order_freeze_epochs': int(
                    self.plane_order_freeze_epochs
                ),
                'joint_team_ppo': bool(
                    getattr(self.all_args, 'joint_team_ppo', False)
                ),
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
            },
            'model': model.ac.state_dict(),
            'actor_optim': model.actor_optimizer.state_dict(),
            'critic_optim': model.critic_optimizer.state_dict(),
            'actor_update_health': self._actor_update_health_snapshot(),
            # 'lagrangmdvrpn_multiplier': lagrangmdvrpn_multiplier
        }
        if self.trainer.value_normalizer is not None:
            checkpoint['value_normalizer'] = self.trainer.value_normalizer.state_dict()
        if extra:
            checkpoint.update(extra)
        if hasattr(model, 'lagrangmdvrpn_multipliers'):
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
        checkpoint = {
            'episodes': 0,
            'stage': 'plane_iga_bc_pretrain',
            'training_stage': self.training_stage,
            'tau': model.ac.tau,
            'model': model.ac.state_dict(),
            'actor_optim': model.actor_optimizer.state_dict(),
            'critic_optim': model.critic_optimizer.state_dict(),
            'plane_order_mode': model.ac.plane_order_mode,
            'plane_pair_decoder': model.ac.plane_pair_decoder,
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
            'bc_reference_target_kl': self.bc_reference_target_kl,
            'bc_reference_hard_gate': self.bc_reference_hard_gate,
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
        }
        if self.trainer.value_normalizer is not None:
            checkpoint['value_normalizer'] = (
                self.trainer.value_normalizer.state_dict()
            )
        self._atomic_torch_save(checkpoint, save_path)
        print(f'[Info] Saved IGA plane BC checkpoint to {save_path}')

    def save_device_bc_checkpoint(self):
        model = self.trainer.policy
        save_path = os.path.join(self.save_dir, 'checkpoint_DeviceBC.pt')
        checkpoint = {
            'episodes': 0,
            'stage': 'device_bc_pretrain',
            'training_stage': self.training_stage,
            'tau': model.ac.tau,
            'plane_order_mode': model.ac.plane_order_mode,
            'plane_pair_decoder': model.ac.plane_pair_decoder,
            'model': model.ac.state_dict(),
            'actor_optim': model.actor_optimizer.state_dict(),
            'critic_optim': model.critic_optimizer.state_dict(),
        }
        if self.trainer.value_normalizer is not None:
            checkpoint['value_normalizer'] = self.trainer.value_normalizer.state_dict()
        self._atomic_torch_save(checkpoint, save_path)
        print(f"[Info] Saved device BC checkpoint to {save_path}")

    def restore(self, checkpoint):
        """Restore policy's networks from a saved model."""
        checkpoint_path = checkpoint
        checkpoint = torch.load(checkpoint, map_location=self.device)
        print(
            f"[Info] Loading checkpoint {checkpoint_path} "
            f"(training_stage={checkpoint.get('training_stage', 'legacy')})."
        )
        checkpoint_arch = (
            checkpoint.get('plane_order_mode'), checkpoint.get('plane_pair_decoder')
        )
        configured_arch = (self.policy.ac.plane_order_mode, self.policy.ac.plane_pair_decoder)
        if None not in checkpoint_arch and checkpoint_arch != configured_arch:
            raise ValueError(f'Checkpoint decoder {checkpoint_arch} != configured {configured_arch}.')
        self.policy.load_model_state(checkpoint['model'])
        self.policy.ac.tau = checkpoint.get('tau', self.policy.ac.tau)
        self.all_args.anneal_original = self.policy.ac.tau
        self._pending_value_normalizer_state = checkpoint.get('value_normalizer')
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
            'checkpoint_PrePPO.pt',
            'checkpoint_Best.pt',
        ):
            source = os.path.join(source_dir, filename)
            target = os.path.join(self.save_dir, filename)
            if os.path.isfile(source) and not os.path.exists(target):
                shutil.copy2(source, target)
