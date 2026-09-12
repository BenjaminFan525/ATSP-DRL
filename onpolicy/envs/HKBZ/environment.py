# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import hashlib
import numpy as np
import os
from typing import List, Dict, Tuple
from onpolicy.envs.HKBZ.core import Plane, Device, Job, Resource, Site
from onpolicy.envs.HKBZ.utils import arrange_devices
import json
import matplotlib.pyplot as plt
from gymnasium.utils import seeding
import math
import torch
from torch_geometric.data import HeteroData
import copy
from scipy.optimize import linear_sum_assignment
from onpolicy.utils import hkbz_semantics
from onpolicy.envs.HKBZ.resource_teacher import (
    ResourceGenomeLayout,
    mixed_resource_actions,
)

# 调度环境类：基于Gymnasium的多智能体强化学习环境，用于模拟飞机在机场站点的调度过程
class AircraftScheduleEnv(gym.Env):
    environment_name = "Plane Schedule"
    SEMANTICS_VERSION = hkbz_semantics.ENVIRONMENT_SEMANTICS_VERSION
    INTRINSIC_READY_TIME_LABEL_SCHEMA_VERSION = 1
    INTRINSIC_READY_TIME_SEMANTICS = (
        'earliest_physical_ready_time_excluding_mobile_resource_delay'
    )
    IGA_POTENTIAL_SCHEMA_VERSION = 3
    RESOURCE_POTENTIAL_SCHEMA_VERSION = 1
    AGENT_TYPE_PLANE = 0
    AGENT_TYPE_DEVICE = 1
    AGENT_TYPE_TRANSPORTER = 2
    TRANSPORTER_RESOURCE_TYPE = "R014"
    TRANSFER_JOB_CODE = "ZY-T"
    DEPARTURE_JOB_GROUP = "出场"
    EXCLUDED_SERVICE_JOB_CODES = frozenset({"ZY01", "ZY-L"})
    TRANSPORTER_SERVICE_BONUS = 60.0
    TRANSPORTER_WAIT_PENALTY_COEF = 1.0
    TRANSPORTER_REPOSITION_PENALTY_COEF = 0.5
    TRANSPORTER_OCCUPANCY_PENALTY_COEF = 0.25
    TRANSPORTER_NOOP_PENALTY = 30.0
    TRANSPORTER_INVALID_REQUEST_PENALTY = 45.0
    IGA_POTENTIAL_BASE_FEATURES = (
        'remaining_work',
        'remaining_jobs',
        'waiting_age',
        'resource_queue',
        'relocation_debt',
    )
    IGA_POTENTIAL_TAIL_FEATURES = (
        'max_plane_remaining_work',
        'max_waiting_age',
        'future_release_tail',
    )
    IGA_POTENTIAL_DEPARTURE_FEATURES = (
        'remaining_service_work',
        'remaining_service_jobs',
        'remaining_departure_work',
        'remaining_departure_jobs',
        'departure_ready_count',
        'departure_ready_age_sum',
        'departure_ready_age_max',
        'departure_staged_count',
        'departure_r014_queue',
        'departure_r014_unavailable_count',
        'departure_r014_eta_sum',
        'departure_r014_eta_max',
        'departure_tow_work',
        'max_departure_tail_lb',
        'runway_busy_count',
        'runway_remaining_work',
        'runway_wave_lb',
        'service_site_pressure',
        'completed_plane_stand_pressure',
        'future_arrival_pressure',
    )
    IGA_POTENTIAL_FEATURES = (
        IGA_POTENTIAL_BASE_FEATURES
        + IGA_POTENTIAL_TAIL_FEATURES
        + IGA_POTENTIAL_DEPARTURE_FEATURES
    )
    RESOURCE_POTENTIAL_FEATURES = IGA_POTENTIAL_FEATURES + (
        'resource_request_cost',
        'resource_request_sum',
        'resource_request_max',
        'resource_forecast_cost',
        'resource_arrival_pressure',
        'resource_blocking_request_count',
        'resource_lookahead_request_count',
    )
    POTENTIAL_REWARD_MODES = frozenset({
        'iga_potential',
        'team_time_potential',
        'team_time_resource_fitted_potential',
    })
    GLOBAL_FEATURE_MODES = hkbz_semantics.GLOBAL_FEATURE_MODES
    GLOBAL_FEATURE_DIM = hkbz_semantics.GLOBAL_FEATURE_DIM
    GLOBAL_FEATURE_SCHEMA_VERSION = hkbz_semantics.GLOBAL_FEATURE_SCHEMA_VERSION
    GLOBAL_FEATURE_NORMALIZATION_VERSION = (
        hkbz_semantics.GLOBAL_FEATURE_NORMALIZATION_VERSION
    )
    GLOBAL_FEATURE_F1_NAMES = hkbz_semantics.GLOBAL_FEATURE_F1_NAMES
    GLOBAL_FEATURE_RESOURCE_COMMON_NAMES = (
        hkbz_semantics.GLOBAL_FEATURE_RESOURCE_COMMON_NAMES
    )
    GLOBAL_FEATURE_F2_NAMES = hkbz_semantics.GLOBAL_FEATURE_F2_NAMES
    GLOBAL_FEATURE_DEPARTURE_NAMES = (
        hkbz_semantics.GLOBAL_FEATURE_DEPARTURE_NAMES
    )

    @classmethod
    def global_feature_contract(cls, mode):
        """Return the exact, versioned meaning of the 24 global slots.

        ``f1f2`` and ``f1f2_departure`` deliberately have the same tensor
        width but different meanings in the final six slots.  A checkpoint
        therefore needs a semantic identifier in addition to shape checks.
        """
        return hkbz_semantics.global_feature_contract(mode)
    
    def __init__(self, config_list, render_mode: str = None):
        super().__init__()
        
        # --- [修改点 1]：支持传入配置文件列表 ---
        # 兼容处理：如果传入的是单个字典，转为列表
        if isinstance(config_list, dict):
            self.data_list = [config_list]
        elif isinstance(config_list, list) and len(config_list) > 0:
            self.data_list = config_list
        else:
            raise ValueError("config_list 必须是配置字典或配置字典的列表")
            
        self.data_idx = 0
        self._train_epoch_case_budget_per_worker = max(
            0,
            int(self.data_list[0].get(
                '_train_epoch_case_budget_per_worker', 0
            )),
        )
        self._training_case_cycle_initialized = False
        self.render_mode = render_mode
        
        # 初始化基础配置为列表的第一项，以防其他未剥离的逻辑需要调用 self.config
        self.config = self.data_list[0] 
        
        # MARL核心属性
        self.n_agents = 0
        self.n_actions = 0  # 动作空间维度
        self.obs_shape = 0  # 观测空间维度
        self.state_shape = 0  # 全局状态维度
        
        # 训练监控与可视化
        self.steps = 0
        self.step_time = 0
        self.total_time = 0
        self.fig = None
        self.ax = None
        
        # 基础状态容器
        self.planes = {}
        self.force_transfer_planes = []
        self.trajectory_log = []
        self.potential_transition_log = []
        self.device_trajectory_log = []
        self.device_decision_log = []
        self.pending_actions = {}
        self.pending_device_actions = {}
        self.request_list = []
        self.request_pool = {}
        self.resource_intent_ledger = {}
        self._v6_dispatch_history = {}
        # Factual, replay-only supervision ledger. Values are absolute
        # physical timestamps at which an operation could start after
        # removing only the wait for its own mobile resource. The ledger is
        # never consulted by masks, transitions, rewards, or dispatch.
        self.intrinsic_ready_time_occurrences = {}
        self._resource_lateness_metrics_cache = None
        self._deferred_lookahead_requests = set()
        self._lookahead_deferred_at_time = None
        self.device_deadlock_repeat_limit = int(self.config.get('device_deadlock_repeat_limit', 300))
        self._last_device_decision_signature = None
        self._device_decision_repeat_count = 0
        self._load_plane_safety_config(self.config)
        self.cycle_terminated = False
        self.cycle_agent_ids = set()
        self.cycle_reason = ''
        self.departure_barrier_open = False
        self.departure_ready_since = {}
        self.departure_transporter_by_plane = {}
        self.departure_plane_by_transporter = {}
        self._departure_runway_plan = {}
        self.departed_agent_ids = set()
        self.departure_log = []
        self._departed_this_step = {}
        self.departed_total_relocations = 0
        
        # 环境设置
        self.seed(self.config.get('seed', None))
        self.use_domain_rand = self.config.get('use_domain_rand', True)
        self.resource_policy = self.config.get('resource_policy', 'heuristic')
        self._load_resource_lookahead_config(self.config)
        self.global_feature_mode = str(
            self.config.get('global_feature_mode', 'none')
        )
        if self.global_feature_mode not in self.GLOBAL_FEATURE_MODES:
            raise ValueError(
                f"Unsupported global_feature_mode={self.global_feature_mode!r}."
            )
        self.hindsight_reward_mode = self.config.get('hindsight_reward_mode', 'shaping')
        self.hindsight_cmax_coef = float(self.config.get('hindsight_cmax_coef', 1.0))
        self.hindsight_shaping_coef = float(self.config.get('hindsight_shaping_coef', 1.0))
        self.hindsight_terminal_cmax_coef = float(self.config.get('hindsight_terminal_cmax_coef', 0.0))
        self.n_plane_agents = int(self.config.get('n_agents', 0))
        self.max_device_num = int(self.config.get('max_device_num', 0)) if self.resource_policy == 'drl' else 0
        
        # --- [修改点 2]：首次初始化，建立 Action/Obs 空间 ---
        # 在 __init__ 中加载一次数据，是为了让 gym.Env 能够正确初始化 action_space 等静态属性
        initial_data = self._load_data_from_disk(self.config)
        self._apply_data(initial_data, clone=False)

    def _load_resource_lookahead_config(self, config):
        """Load deadline-aware Stage-2 controls without changing graph width."""
        self.device_lookahead_dispatch = bool(
            config.get('device_lookahead_dispatch', False)
        )
        self.device_lookahead_safety_margin = float(
            config.get('device_lookahead_safety_margin', 60.0)
        )
        self.device_deadline_aware_dispatch = bool(
            config.get('device_deadline_aware_dispatch', False)
        )
        self.device_future_intent_horizon = int(
            config.get('device_future_intent_horizon', 0)
        )
        self.device_future_intent_mode = str(
            config.get('device_future_intent_mode', 'legacy_one')
        )
        self.device_frontier_max_requests = int(
            config.get('device_frontier_max_requests', 2)
        )
        self.device_request_capacity_per_plane = int(
            config.get('device_request_capacity_per_plane', 0)
        )
        self.resource_release_aware_eta = bool(
            config.get('resource_release_aware_eta', False)
        )
        self.device_lookahead_reservation_mode = str(
            config.get('device_lookahead_reservation_mode', 'none')
        )
        self.device_reservation_grace_seconds = float(
            config.get('device_reservation_grace_seconds', 300.0)
        )
        self.device_departure_lookahead = bool(
            config.get('device_departure_lookahead', False)
        )
        self.resource_lateness_coef = float(
            config.get('resource_lateness_coef', 0.0)
        )
        self.resource_critical_lateness_coef = float(
            config.get('resource_critical_lateness_coef', 0.0)
        )
        self.resource_earliness_coef = float(
            config.get('resource_earliness_coef', 0.0)
        )
        self.resource_slack_criticality_seconds = float(
            config.get('resource_slack_criticality_seconds', 1800.0)
        )
        self.resource_slack_min_weight = float(
            config.get('resource_slack_min_weight', 0.25)
        )
        self.resource_slack_forecast_seconds = float(
            config.get('resource_slack_forecast_seconds', 0.0)
        )
        if self.device_lookahead_safety_margin < 0.0:
            raise ValueError(
                'device_lookahead_safety_margin must be non-negative.'
            )
        if self.device_future_intent_horizon not in {0, 1, 2, 3}:
            raise ValueError(
                'device_future_intent_horizon must be one of 0, 1, 2 or 3.'
            )
        if self.device_future_intent_mode not in {
            'legacy_one', 'bounded_frontier'
        }:
            raise ValueError(
                'device_future_intent_mode must be legacy_one or '
                'bounded_frontier.'
            )
        if self.device_frontier_max_requests < 1:
            raise ValueError('device_frontier_max_requests must be positive.')
        if self.device_request_capacity_per_plane < 0:
            raise ValueError(
                'device_request_capacity_per_plane must be non-negative.'
            )
        if self.device_lookahead_reservation_mode not in {
            'none', 'soft', 'hard'
        }:
            raise ValueError(
                'device_lookahead_reservation_mode must be none, soft or hard.'
            )
        if (
            not math.isfinite(self.device_reservation_grace_seconds)
            or self.device_reservation_grace_seconds < 0.0
        ):
            raise ValueError(
                'device_reservation_grace_seconds must be finite and '
                'non-negative.'
            )
        reward_coefs = (
            self.resource_lateness_coef,
            self.resource_critical_lateness_coef,
            self.resource_earliness_coef,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in reward_coefs):
            raise ValueError(
                'Resource lateness coefficients must be finite and non-negative.'
            )
        if (
            not math.isfinite(self.resource_slack_criticality_seconds)
            or self.resource_slack_criticality_seconds <= 0.0
        ):
            raise ValueError(
                'resource_slack_criticality_seconds must be finite and positive.'
            )
        if (
            not math.isfinite(self.resource_slack_min_weight)
            or not 0.0 <= self.resource_slack_min_weight <= 1.0
        ):
            raise ValueError('resource_slack_min_weight must be in [0, 1].')
        if (
            not math.isfinite(self.resource_slack_forecast_seconds)
            or self.resource_slack_forecast_seconds < 0.0
        ):
            raise ValueError(
                'resource_slack_forecast_seconds must be finite and non-negative.'
            )
        if (
            self.device_future_intent_horizon > 0
            or self.device_departure_lookahead
            or self.device_deadline_aware_dispatch
        ) and not self.device_lookahead_dispatch:
            raise ValueError(
                'Deadline/future-intent controls require device_lookahead_dispatch.'
            )

    def _request_slots_per_plane(self):
        """Return a safe, optionally fixed request-node width.

        A committed action can coexist with a bounded dependency frontier.
        Departure lookahead replaces rather than adds to a service frontier in
        normal traces, but one extra slot keeps the bound fail-safe.  A fixed
        width is used by causal arms that share one persistent evaluator.
        """
        if self.device_request_capacity_per_plane > 0:
            return int(self.device_request_capacity_per_plane)
        if not self.device_lookahead_dispatch:
            return 1
        if self.device_future_intent_horizon > 0:
            return max(3, 1 + int(self.device_frontier_max_requests))
        return 2

    def _load_plane_safety_config(self, config):
        self.plane_cycle_repeat_limit = int(config.get('plane_cycle_repeat_limit', 8))
        self.plane_no_progress_limit = int(config.get('plane_no_progress_limit', 120))
        self.plane_relocation_limit = int(config.get('plane_relocation_limit', 40))
        self.plane_first_completion_bonus = float(config.get('plane_first_completion_bonus', 120.0))
        self.plane_repeat_relocation_penalty = float(
            config.get('plane_repeat_relocation_penalty', 300.0)
        )
        self.plane_reset_job_penalty = float(config.get('plane_reset_job_penalty', 120.0))
        self.plane_no_progress_penalty = float(config.get('plane_no_progress_penalty', 60.0))
        self.plane_cycle_penalty = float(config.get('plane_cycle_penalty', 20000.0))

    def set_global_feature_mode(self, mode):
        """Switch the shape-compatible global observation semantics safely.

        Shared evaluators serve checkpoints from multiple causal arms.  The
        modes used by this project retain the same tensor shape but not the
        same meaning, so the requested mode must survive the next dataset
        reset performed by evaluation.
        """
        mode = str(mode)
        if mode not in self.GLOBAL_FEATURE_MODES:
            raise ValueError(f'Unsupported global_feature_mode={mode!r}.')
        for config in self.data_list:
            config['global_feature_mode'] = mode
        self.config['global_feature_mode'] = mode
        self.global_feature_mode = mode
        return mode

    def resource_planning_config(self):
        """Return shape-compatible Stage-2 planning semantics.

        These fields may differ between causal arms served by one persistent
        validator.  They deliberately do not include reward coefficients,
        dataset paths, or any tensor-width setting.
        """
        return {
            'resource_release_aware_eta': bool(
                self.resource_release_aware_eta
            ),
            'device_future_intent_horizon': int(
                self.device_future_intent_horizon
            ),
            'device_future_intent_mode': str(
                self.device_future_intent_mode
            ),
            'device_frontier_max_requests': int(
                self.device_frontier_max_requests
            ),
            'device_request_capacity_per_plane': int(
                self.device_request_capacity_per_plane
            ),
            'device_lookahead_reservation_mode': str(
                self.device_lookahead_reservation_mode
            ),
            'device_reservation_grace_seconds': float(
                self.device_reservation_grace_seconds
            ),
        }

    def set_resource_planning_config(self, planning):
        """Switch a shared evaluator to one arm's compatible semantics."""
        if not isinstance(planning, dict):
            raise TypeError('resource planning config must be a dictionary.')
        allowed = set(self.resource_planning_config())
        unknown = set(planning).difference(allowed)
        if unknown:
            raise ValueError(
                f'Unknown resource planning fields: {sorted(unknown)!r}.'
            )
        previous_capacity = int(getattr(self, 'max_request_num', 0))
        for config in self.data_list:
            config.update(planning)
        self.config.update(planning)
        self._load_resource_lookahead_config(self.config)
        # Shared validation may switch horizon/frontier semantics only when
        # every arm reserved one identical fixed padded request width.
        requests_per_plane = self._request_slots_per_plane()
        new_capacity = requests_per_plane * self.n_plane_agents + 1
        if previous_capacity and new_capacity != previous_capacity:
            raise ValueError(
                'Shared-evaluator planning switch changed request capacity: '
                f'{previous_capacity} -> {new_capacity}.'
            )
        return self.resource_planning_config()

    def _load_iga_potential_config(self, config):
        self.iga_potential_beta = float(config.get('iga_potential_beta', 0.0))
        self.iga_potential_gamma = float(config.get('iga_potential_gamma', 0.99))
        if self.iga_potential_beta < 0.0:
            raise ValueError('iga_potential_beta must be non-negative.')
        if not 0.0 <= self.iga_potential_gamma <= 1.0:
            raise ValueError('iga_potential_gamma must be in [0, 1].')

        self.iga_potential_weights = {
            name: 0.0 for name in self.IGA_POTENTIAL_FEATURES
        }
        self.iga_potential_weights_path = str(
            config.get('iga_potential_weights_path', '') or ''
        )
        if self.hindsight_reward_mode not in self.POTENTIAL_REWARD_MODES:
            return
        if not self.iga_potential_weights_path:
            raise ValueError(
                'iga_potential reward requires iga_potential_weights_path.'
            )
        weights_path = os.path.abspath(
            os.path.expanduser(self.iga_potential_weights_path)
        )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f'IGA potential weights file does not exist: {weights_path}'
            )
        with open(weights_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        weights = payload.get('weights', payload)
        if not isinstance(weights, dict):
            raise ValueError(
                f'IGA potential weights must be a JSON object: {weights_path}'
            )
        fitted_resource = (
            self.hindsight_reward_mode
            == 'team_time_resource_fitted_potential'
        )
        strict_v2 = self.hindsight_reward_mode == 'team_time_potential'
        strict_feature_names = (
            self.RESOURCE_POTENTIAL_FEATURES
            if fitted_resource else self.IGA_POTENTIAL_FEATURES
        )
        if strict_v2:
            schema_version = int(payload.get('potential_schema_version', 0))
            semantics_version = str(
                payload.get('environment_semantics_version', '')
            )
            feature_names = tuple(payload.get('feature_names', ()))
            if schema_version != self.IGA_POTENTIAL_SCHEMA_VERSION:
                raise ValueError(
                    'team_time_potential requires potential schema '
                    f'{self.IGA_POTENTIAL_SCHEMA_VERSION}, got '
                    f'{schema_version}: {weights_path}'
                )
            if semantics_version != self.SEMANTICS_VERSION:
                raise ValueError(
                    'team_time_potential environment semantics mismatch: '
                    f'expected {self.SEMANTICS_VERSION!r}, got '
                    f'{semantics_version!r}: {weights_path}'
                )
            if feature_names != tuple(self.IGA_POTENTIAL_FEATURES):
                raise ValueError(
                    'team_time_potential feature schema does not exactly '
                    f'match the environment: {weights_path}'
                )
        elif fitted_resource:
            schema_version = int(payload.get('potential_schema_version', 0))
            semantics_version = str(
                payload.get('environment_semantics_version', '')
            )
            feature_names = tuple(payload.get('feature_names', ()))
            if schema_version != self.RESOURCE_POTENTIAL_SCHEMA_VERSION:
                raise ValueError(
                    'team_time_resource_fitted_potential requires resource '
                    f'potential schema {self.RESOURCE_POTENTIAL_SCHEMA_VERSION}, '
                    f'got {schema_version}: {weights_path}'
                )
            if semantics_version != self.SEMANTICS_VERSION:
                raise ValueError(
                    'resource fitted potential environment semantics mismatch: '
                    f'expected {self.SEMANTICS_VERSION!r}, got '
                    f'{semantics_version!r}: {weights_path}'
                )
            if feature_names != tuple(strict_feature_names):
                raise ValueError(
                    'resource fitted potential feature schema does not exactly '
                    f'match the environment: {weights_path}'
                )
        # Legacy iga_potential files remain explicitly replayable with zero
        # weights for features that did not exist when they were calibrated.
        required_features = (
            strict_feature_names
            if strict_v2 or fitted_resource
            else self.IGA_POTENTIAL_BASE_FEATURES
        )
        missing = [name for name in required_features if name not in weights]
        if missing:
            raise ValueError(
                f'IGA potential weights missing {missing}: {weights_path}'
            )
        validated = {}
        for name in strict_feature_names:
            value = float(weights.get(name, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f'IGA potential weight {name} must be finite and '
                    f'non-negative, got {value}.'
                )
            validated[name] = value
        if not any(value > 0.0 for value in validated.values()):
            raise ValueError('At least one IGA potential weight must be positive.')
        self.iga_potential_weights = validated
        self.iga_potential_weights_path = weights_path

    @staticmethod
    def _job_duration(job, fuel=30.0):
        if job.time is not None:
            return max(0.0, float(job.time))
        if job.code == 'ZY10':
            return max(0.0, math.ceil((100.0 - float(fuel)) / 5.0 * 60.0))
        return 0.0

    def get_iga_potential_features(self):
        """Return non-negative state costs used by teacher-calibrated shaping."""
        features = {name: 0.0 for name in self.IGA_POTENTIAL_FEATURES}
        current_time = float(getattr(self, 'total_time', 0.0))
        sites = getattr(self, 'sites', {})
        departure_codes = set(
            getattr(self, 'departure_job_code_list', ())
        )
        service_codes = set(getattr(self, 'service_job_code_list', ()))
        active_departure_planes = 0

        for plane in self.planes.values():
            if plane.is_completed_all_jobs():
                continue
            fuel = plane.config.get('fuel', 30.0)
            features['remaining_jobs'] += float(
                len(plane.left_jobs) + len(plane.current_jobs)
            )
            left_work = sum(
                self._job_duration(plane.jobs[job_code], fuel)
                for job_code in plane.left_jobs
            )
            if plane.is_busy:
                current_work = max(
                    0.0, float(getattr(plane.site, 'left_job_time', 0.0))
                )
            else:
                current_work = sum(
                    self._job_duration(
                        plane.jobs[job_code],
                        fuel,
                    )
                    for job_code in plane.current_jobs
                )
            plane_remaining_work = left_work + current_work
            features['remaining_work'] += plane_remaining_work
            features['max_plane_remaining_work'] = max(
                features['max_plane_remaining_work'],
                plane_remaining_work,
            )
            if plane.is_waiting:
                waiting_age = max(0.0, float(plane.waiting_time))
                features['waiting_age'] += waiting_age
                features['max_waiting_age'] = max(
                    features['max_waiting_age'],
                    waiting_age,
                )
                features['resource_queue'] += 1.0
            features['relocation_debt'] += max(
                0.0,
                float(plane.relocations_since_progress)
                + float(plane.no_progress_decisions),
            )

            service_left = [
                code for code in plane.left_jobs if code in service_codes
            ]
            departure_left = [
                code for code in plane.left_jobs if code in departure_codes
            ]
            service_current = [
                code for code in plane.current_jobs if code in service_codes
            ]
            departure_current = [
                code for code in plane.current_jobs if code in departure_codes
            ]
            features['remaining_service_jobs'] += float(
                len(service_left) + len(service_current)
            )
            features['remaining_departure_jobs'] += float(
                len(departure_left) + len(departure_current)
            )
            service_work = sum(
                self._job_duration(plane.jobs[code], fuel)
                for code in service_left
            )
            departure_work = sum(
                self._job_duration(plane.jobs[code], fuel)
                for code in departure_left
            )
            current_service_nominal = sum(
                self._job_duration(plane.jobs[code], fuel)
                for code in service_current
            )
            current_departure_nominal = sum(
                self._job_duration(plane.jobs[code], fuel)
                for code in departure_current
            )
            current_nominal = (
                current_service_nominal + current_departure_nominal
            )
            if plane.is_busy and current_nominal > 0.0:
                current_scale = current_work / current_nominal
                service_work += current_service_nominal * current_scale
                departure_work += current_departure_nominal * current_scale
            else:
                service_work += current_service_nominal
                departure_work += current_departure_nominal
            features['remaining_service_work'] += max(0.0, service_work)
            features['remaining_departure_work'] += max(
                0.0, departure_work
            )

            service_completed = bool(
                getattr(plane, 'has_completed_service_jobs', lambda: False)()
            )
            departure_started = bool(
                getattr(plane, 'has_started_departure', lambda: False)()
            )
            takeoff_sites = set(getattr(self, 'takeoff_site_code_list', ()))
            awaiting_departure = bool(
                service_completed
                and not departure_started
                and plane.site.code not in takeoff_sites
            )
            if departure_left or departure_current or service_completed:
                active_departure_planes += 1

            r014_eta = 0.0
            tow_work = (
                max(0.0, float(getattr(plane, 'left_trans_time', 0.0)))
                if getattr(plane, 'transport_purpose', None) == 'departure'
                else 0.0
            )
            if awaiting_departure:
                features['departure_ready_count'] += 1.0
                ready_since = float(
                    getattr(self, 'departure_ready_since', {}).get(
                        plane.code, current_time
                    )
                )
                ready_age = max(0.0, current_time - ready_since)
                features['departure_ready_age_sum'] += ready_age
                features['departure_ready_age_max'] = max(
                    features['departure_ready_age_max'], ready_age
                )
                if bool(getattr(plane, 'departure_staging_decided', False)):
                    features['departure_staged_count'] += 1.0
                    features['departure_r014_queue'] += 1.0

                transporters = list(
                    getattr(self, 'mobile_devices', {}).get(
                        self.TRANSPORTER_RESOURCE_TYPE, ()
                    )
                )
                candidates = [
                    device for device in transporters
                    if getattr(device, 'reserved_for_plane', None)
                    in {None, plane.code}
                ]
                if not candidates:
                    r014_eta = 36000.0
                else:
                    r014_eta = min(
                        float(max(
                            device.left_trans_time,
                            device.left_rec_time,
                        )) + (
                            abs(float(device.site.pos[0]) - float(plane.site.pos[0]))
                            + abs(float(device.site.pos[1]) - float(plane.site.pos[1]))
                        ) / max(float(getattr(device, 'velocity', 1.0)), 1.0)
                        for device in candidates
                    )
                features['departure_r014_eta_sum'] += r014_eta
                features['departure_r014_eta_max'] = max(
                    features['departure_r014_eta_max'], r014_eta
                )
                if takeoff_sites:
                    tow_work = min(
                        (
                            abs(float(plane.site.pos[0]) - float(sites[code].pos[0]))
                            + abs(float(plane.site.pos[1]) - float(sites[code].pos[1]))
                        ) / max(float(getattr(plane, 'velocity', 1.0)), 1.0)
                        + 60.0
                        for code in takeoff_sites
                        if code in sites
                    )
            features['departure_tow_work'] += tow_work
            features['max_departure_tail_lb'] = max(
                features['max_departure_tail_lb'],
                r014_eta + tow_work + departure_work,
            )

        transporters = list(
            getattr(self, 'mobile_devices', {}).get(
                self.TRANSPORTER_RESOURCE_TYPE, ()
            )
        )
        features['departure_r014_unavailable_count'] = float(sum(
            (not device.is_idle())
            or getattr(device, 'reserved_for_plane', None) is not None
            for device in transporters
        ))

        runway_sites = [
            sites[code]
            for code in getattr(self, 'takeoff_site_code_list', ())
            if code in sites
        ]
        features['runway_busy_count'] = float(sum(
            site.is_occupied or site.is_interfered for site in runway_sites
        ))
        features['runway_remaining_work'] = sum(
            max(
                0.0,
                float(getattr(site, 'left_job_time', 0.0)),
                float(getattr(site, 'left_rec_time', 0.0)),
            )
            for site in runway_sites
        )
        departure_duration = sum(
            self._job_duration(self.jobs[code])
            for code in departure_codes
            if code in self.jobs
        )
        future_planes = len(getattr(self, 'landing_list', ()))
        runway_count = max(1, len(runway_sites))
        features['runway_wave_lb'] = (
            math.ceil((active_departure_planes + future_planes) / runway_count)
            * departure_duration
        )

        service_site_codes = set(
            getattr(self, 'service_site_code_list', ())
        )
        service_sites = [
            sites[code] for code in service_site_codes
            if code in sites
        ]
        features['service_site_pressure'] = float(sum(
            site.is_occupied or site.is_interfered for site in service_sites
        ))
        features['completed_plane_stand_pressure'] = float(sum(
            bool(getattr(plane, 'has_completed_service_jobs', lambda: False)())
            and not bool(getattr(plane, 'has_started_departure', lambda: False)())
            and plane.site.code in service_site_codes
            for plane in self.planes.values()
        ))
        free_service_sites = sum(
            not site.is_occupied and not site.is_interfered
            for site in service_sites
        )
        imminent_arrivals = sum(
            max(0.0, float(item[0]) - current_time) <= 1800.0
            for item in getattr(self, 'landing_list', ())
        )
        features['future_arrival_pressure'] = float(max(
            0, imminent_arrivals - free_service_sites
        ))

        for land_time, _, _, fuel in getattr(self, 'landing_list', ()):
            future_work = sum(
                self._job_duration(self.jobs[job_code], fuel)
                for job_code in self.job_code_list
            )
            features['remaining_jobs'] += float(len(self.job_code_list))
            features['remaining_work'] += future_work
            features['max_plane_remaining_work'] = max(
                features['max_plane_remaining_work'],
                future_work,
            )
            features['future_release_tail'] = max(
                features['future_release_tail'],
                max(0.0, float(land_time) - current_time) + future_work,
            )
            future_service_codes = [
                code for code in self.job_code_list if code in service_codes
            ]
            future_departure_codes = [
                code for code in self.job_code_list if code in departure_codes
            ]
            features['remaining_service_jobs'] += float(
                len(future_service_codes)
            )
            features['remaining_departure_jobs'] += float(
                len(future_departure_codes)
            )
            features['remaining_service_work'] += sum(
                self._job_duration(self.jobs[code], fuel)
                for code in future_service_codes
            )
            features['remaining_departure_work'] += sum(
                self._job_duration(self.jobs[code], fuel)
                for code in future_departure_codes
            )
        return features

    def _iga_potential_value(self, features):
        return -sum(
            self.iga_potential_weights[name] * float(features.get(name, 0.0))
            for name in self.IGA_POTENTIAL_FEATURES
        )

    def _plane_remaining_work_seconds(self, plane):
        """Cheap per-aircraft work lower bound for critical-path weighting."""
        fuel = float(plane.config.get('fuel', 30.0))
        left_work = sum(
            self._job_duration(plane.jobs[code], fuel)
            for code in plane.left_jobs
            if code in plane.jobs
        )
        if plane.is_busy:
            current_work = max(
                0.0, float(getattr(plane.site, 'left_job_time', 0.0))
            )
        else:
            current_work = sum(
                self._job_duration(plane.jobs[code], fuel)
                for code in plane.current_jobs
                if code in plane.jobs
            )
        return max(0.0, float(left_work + current_work))

    def get_resource_slack_potential_components(self):
        """Return a non-negative, online proxy for Cmax-critical resource delay.

        Unlike the old episode-level maximum aircraft wait, this state cost is
        attached to each outstanding request.  It combines elapsed/predicted
        device delay with the owning aircraft's remaining-work slack.  A small
        optional forecast term exposes imminent stand pressure already present
        in the observation contract.  The runner uses only potential
        differences with gamma=1, so the terminal Cmax objective is unchanged.
        """
        remaining_work = {
            plane.code: self._plane_remaining_work_seconds(plane)
            for plane in self.planes.values()
            if not plane.is_completed_all_jobs()
        }
        max_remaining = max(remaining_work.values(), default=0.0)
        criticality_scale = self.resource_slack_criticality_seconds
        devices_by_type = {
            resource_type: tuple(devices)
            for resource_type, devices in self.mobile_devices.items()
        }
        eta_cache = {}
        weighted_delays = []
        blocking_count = 0
        lookahead_count = 0

        for request in self.request_list:
            if request.get('is_noop', False):
                continue
            target_site = self.sites.get(request.get('site_code'))
            if target_site is None:
                continue
            required_types = tuple(request.get('needed_res_types', ()))
            if not required_types:
                continue
            type_etas = []
            for resource_type in required_types:
                cache_key = (
                    str(resource_type),
                    str(target_site.code),
                    str(request.get('plane_id', ''))
                    if self.resource_release_aware_eta else '',
                    str(request.get('job_code', ''))
                    if self.resource_release_aware_eta else '',
                )
                eta = eta_cache.get(cache_key)
                if eta is None:
                    candidates = devices_by_type.get(resource_type, ())
                    if self.resource_release_aware_eta:
                        eta = min((
                            self._device_eta_seconds(
                                device, target_site, request=request
                            )
                            for device in candidates
                        ), default=36000.0)
                    else:
                        eta = min((
                            max(
                                0.0,
                                float(getattr(
                                    device, 'left_trans_time', 0.0
                                )),
                                float(getattr(
                                    device, 'left_rec_time', 0.0
                                )),
                            ) + self._device_travel_seconds(
                                device, target_site
                            )
                            for device in candidates
                        ), default=36000.0)
                    eta_cache[cache_key] = float(eta)
                type_etas.append(float(eta))
            # FJSP job resource lists are alternatives: Site.start_jobs picks
            # the first available type.  ``max`` is retained only for the
            # explicit legacy control arm.
            device_eta = (
                min(type_etas, default=36000.0)
                if self.resource_release_aware_eta
                else max(type_etas, default=36000.0)
            )

            is_lookahead = bool(request.get('is_lookahead', False))
            if is_lookahead:
                lookahead_count += 1
                delay = max(
                    0.0,
                    device_eta - max(0.0, float(request.get('lead_time', 0.0))),
                )
            else:
                blocking_count += 1
                delay = (
                    max(0.0, float(request.get('waiting_time', 0.0)))
                    + device_eta
                )

            plane_work = remaining_work.get(request.get('plane_id'), 0.0)
            slack = max(0.0, max_remaining - plane_work)
            min_weight = self.resource_slack_min_weight
            criticality = min_weight + (1.0 - min_weight) * math.exp(
                -slack / criticality_scale
            )
            weighted_delays.append(float(criticality * delay))

        request_sum = float(sum(weighted_delays))
        request_max = float(max(weighted_delays, default=0.0))
        # The max term protects the catastrophic aircraft while the sum term
        # still distinguishes actions when several near-critical requests tie.
        request_cost = 0.5 * request_sum + 0.5 * request_max

        service_site_codes = set(
            getattr(self, 'service_site_code_list', ())
        )
        service_sites = [
            site for code, site in self.sites.items()
            if code in service_site_codes
        ]
        free_service_sites = sum(
            not site.is_occupied and not site.is_interfered
            for site in service_sites
        )
        current_time = float(self.total_time)
        imminent_arrivals = sum(
            max(0.0, float(item[0]) - current_time) <= 1800.0
            for item in self.landing_list
        )
        arrival_pressure = float(max(
            0, imminent_arrivals - free_service_sites
        ))
        forecast_cost = (
            self.resource_slack_forecast_seconds * arrival_pressure
        )
        total_cost = max(0.0, request_cost + forecast_cost)
        if not math.isfinite(total_cost):
            raise RuntimeError('Resource-slack potential became non-finite.')
        return {
            'total_cost': float(total_cost),
            'request_cost': float(request_cost),
            'request_sum': request_sum,
            'request_max': request_max,
            'forecast_cost': float(forecast_cost),
            'arrival_pressure': arrival_pressure,
            'blocking_request_count': int(blocking_count),
            'lookahead_request_count': int(lookahead_count),
        }

    def _resource_slack_potential_value(self):
        return -self.get_resource_slack_potential_components()['total_cost']

    def get_resource_fitted_potential_features(self):
        """Return the versioned, teacher-fitted Stage2 state features."""
        features = self.get_iga_potential_features()
        components = self.get_resource_slack_potential_components()
        features.update({
            'resource_request_cost': float(components['request_cost']),
            'resource_request_sum': float(components['request_sum']),
            'resource_request_max': float(components['request_max']),
            'resource_forecast_cost': float(components['forecast_cost']),
            'resource_arrival_pressure': float(
                components['arrival_pressure']
            ),
            'resource_blocking_request_count': float(
                components['blocking_request_count']
            ),
            'resource_lookahead_request_count': float(
                components['lookahead_request_count']
            ),
        })
        return features

    def _resource_fitted_potential_value(self, features=None):
        if features is None:
            features = self.get_resource_fitted_potential_features()
        value = -sum(
            self.iga_potential_weights[name]
            * float(features.get(name, 0.0))
            for name in self.RESOURCE_POTENTIAL_FEATURES
        )
        if not math.isfinite(value):
            raise RuntimeError('Resource fitted potential became non-finite.')
        return float(value)

    def _load_data_from_disk(self, config_dict):
        """
        根据传入的配置字典，从硬盘加载所有 JSON 数据并实例化基础对象。
        返回一个包含所有环境必须数据的字典。
        """
        data_bundle = {}
        
        # 加载作业数据
        with open(config_dict['jobs_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_bundle['jobs'] = {item["作业编号"] : Job(code=item["作业编号"], 
                    time=item["作业时间"], 
                    group=item["分组"], 
                    resources=item["需要设备类型"] if isinstance(item["需要设备类型"], list) else [], 
                    predecessor=item["前置作业"] if isinstance(item["前置作业"], list) else [], 
                    exclusive=item["互斥作业"] if isinstance(item["互斥作业"], list) else [])
                for item in data}
                
        # 加载固定资源
        with open(config_dict['fixed_res_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_bundle['fixed_resources'] = {
            item["设备编号"]: Resource(
                item["设备编号"],
                item["类型"],
                [str(idx) for idx in range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1]) + 1)],
                max_service=3
            ) for item in data
        }
        
        # 加载移动资源
        with open(config_dict['mobile_res_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_bundle['mobile_resources'] = {
            item["设备编号"]: Resource(
                item["设备编号"],
                item["类型"],
                [item["初始停机位"]],
                max_service=1
            ) for item in data
        }
        
        # 加载站点
        with open(config_dict['sites_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_bundle['sites'] = {
            code: Site(code, {
                'position': pos,
                'jobs': data_bundle['jobs'],
                'fixed_resources': [res for res in data_bundle['fixed_resources'].values() if code in res.sites],
                'mobile_resources': [res for res in data_bundle['mobile_resources'].values() if code in res.sites]
            }) for code, pos in zip(data['sites_codes'], data['sites_positions'])
        }
        
        # 初始化移动设备
        mobile_devices = {}
        for res in data_bundle['mobile_resources'].values():
            device_cfg = {
                'resource': res,
                'velocity': (
                    5 if res.type == self.TRANSPORTER_RESOURCE_TYPE else 3
                ),
                'site': data_bundle['sites'][res.sites[0]]
            }
            if res.type not in mobile_devices:
                mobile_devices[res.type] = [Device(res.code, device_cfg)]
            else:
                mobile_devices[res.type].append(Device(res.code, device_cfg))
        data_bundle['mobile_devices'] = mobile_devices
        
        # 加载航班数据
        with open(config_dict['flights_path'], 'r', encoding='utf-8') as f:
            flights_data = json.load(f)
        data_bundle['flights_data'] = flights_data
        
        # 生成 Base Landing List
        base_landing_list = []
        for idx, item in enumerate(flights_data):
            fuel_percentage = int(item["初始燃油状态"].replace('%', ''))
            base_landing_list.append({
                'land_time': item["到达时间"],
                'fuel': fuel_percentage,
                'bidx': 0,
                'pidx': idx,
                'plane_id': item["飞机编号"]
            })
        base_landing_list.sort(key=lambda x: x['land_time'])
        data_bundle['base_landing_list'] = base_landing_list
        
        return data_bundle

    def _apply_data(self, new_data, clone=False):
        """
        将加载好的数据绑定到环境自身属性上，并更新依赖项。
        """
        self.jobs = new_data['jobs']
        self.fixed_resources = new_data['fixed_resources']
        self.mobile_resources = new_data['mobile_resources']
        self.sites = new_data['sites']
        self.mobile_devices = new_data['mobile_devices']
        self.flights_data = new_data['flights_data']
        self.base_landing_list = new_data['base_landing_list']
        
        self.num_planes = len(self.flights_data)
        self.resource_policy = self.config.get('resource_policy', 'heuristic')
        self._load_resource_lookahead_config(self.config)
        self.global_feature_mode = str(
            self.config.get('global_feature_mode', 'none')
        )
        if self.global_feature_mode not in self.GLOBAL_FEATURE_MODES:
            raise ValueError(
                f"Unsupported global_feature_mode={self.global_feature_mode!r}."
            )
        self.hindsight_reward_mode = self.config.get('hindsight_reward_mode', 'shaping')
        self.hindsight_cmax_coef = float(self.config.get('hindsight_cmax_coef', 1.0))
        self.hindsight_shaping_coef = float(self.config.get('hindsight_shaping_coef', 1.0))
        self.hindsight_terminal_cmax_coef = float(self.config.get('hindsight_terminal_cmax_coef', 0.0))
        self._load_iga_potential_config(self.config)
        self.n_plane_agents = int(self.config.get('n_agents', self.num_planes))
        self.max_device_num = int(self.config.get('max_device_num', 0)) if self.resource_policy == 'drl' else 0
        self.device_deadlock_repeat_limit = int(self.config.get('device_deadlock_repeat_limit', 300))
        self._load_plane_safety_config(self.config)
        self.n_agents = self.n_plane_agents + self.max_device_num
        requests_per_plane = self._request_slots_per_plane()
        self.max_request_num = requests_per_plane * self.n_plane_agents + 1
        self.device_list = [dev for devs in self.mobile_devices.values() for dev in devs]
        self.device_code_to_agent_id = {
            dev.code: self.n_plane_agents + idx for idx, dev in enumerate(self.device_list[:self.max_device_num])
        }
        
        self.waiting_sites = {job_type: [] for job_type in self.jobs.keys()}
        
        # 预计算一些常用的列表
        self.site_code_list = list(self.sites.keys())
        self.service_job_code_list = [
            job.code for job in self.jobs.values()
            if (
                job.group == '保障'
                and job.code not in self.EXCLUDED_SERVICE_JOB_CODES
            )
        ]
        self.departure_job_code_list = [
            job.code for job in self.jobs.values()
            if job.group == self.DEPARTURE_JOB_GROUP
        ]
        if self.TRANSFER_JOB_CODE not in self.jobs:
            raise ValueError(
                f"job.json is missing required transfer job {self.TRANSFER_JOB_CODE}."
            )
        if len(self.departure_job_code_list) != 2:
            raise ValueError(
                "job.json must define exactly two jobs in group '出场'; "
                f"found {self.departure_job_code_list}."
            )
        self.job_code_list = (
            self.service_job_code_list
            + [self.TRANSFER_JOB_CODE]
            + self.departure_job_code_list
        )
        self.runway_code_list = [self.site_code_list[0]] + self.site_code_list[-3:]
        self.takeoff_site_code_list = list(self.runway_code_list[1:])
        self.service_site_code_list = [
            code for code in self.site_code_list
            if code not in self.runway_code_list
        ]
        
        # 更新状态追踪数组
        self.sites_state_global = [-1] * len(self.sites)
        self.state_left_time = np.zeros(len(self.sites))
        
        # 重新定义 Action Space (⚠️ 注意：如果不同配置文件的站点/作业数量不同，这会导致 action_space 大小变化)
        self.action_space = spaces.MultiDiscrete([
            len(self.sites) + 1,
            len(self.job_code_list) + 1,
        ])

    def _build_agent_types(self):
        """Return stable MARL agent roles: plane, ordinary mobile device, transporter."""
        agent_types = np.zeros(self.n_agents, dtype=np.int64)
        if self.resource_policy != 'drl':
            return agent_types
        agent_types[self.n_plane_agents:self.n_agents] = self.AGENT_TYPE_DEVICE

        for dev_idx, dev in enumerate(self.device_list[:self.max_device_num]):
            agent_id = self.n_plane_agents + dev_idx
            if agent_id >= self.n_agents:
                break
            if dev.resource.type == self.TRANSPORTER_RESOURCE_TYPE:
                agent_types[agent_id] = self.AGENT_TYPE_TRANSPORTER
            else:
                agent_types[agent_id] = self.AGENT_TYPE_DEVICE
        return agent_types

    def _is_transporter_device(self, device):
        return device.resource.type == self.TRANSPORTER_RESOURCE_TYPE

    def _record_transporter_decision(self, device, action_idx, reason, request=None):
        if not self._is_transporter_device(device):
            return
        if reason == 'noop_without_request':
            return

        agent_id = self.device_code_to_agent_id.get(device.code)
        if agent_id is None:
            return

        self.device_decision_log.append({
            'step_idx': self.steps,
            'agent_id': agent_id,
            'action': [int(action_idx), 0],
            'start_time': self.total_time,
            'end_time': self.total_time,
            'duration': 0.0,
            'device_id': device.code,
            'device_type': device.resource.type,
            'is_transporter': True,
            'decision_type': reason,
            'job_code': request.get('job_code') if request else None,
            'plane_id': request.get('plane_id') if request else None,
            'site_id': request.get('site_code') if request else None,
            'waiting_time_at_dispatch': float(request.get('waiting_time', 0.0)) if request else 0.0,
            'is_lookahead': bool(request.get('is_lookahead', False)) if request else False,
            'request_kind': request.get('request_kind') if request else None,
            'lead_time_at_decision': float(request.get('lead_time', 0.0)) if request else 0.0,
        })

    def _job_resource_types(self, job):
        return set(getattr(job, 'resources', set()) or set())

    def _active_device_debug(self):
        devices_debug = []
        for dev_idx, device in enumerate(self.device_list[:self.max_device_num]):
            agent_id = self.n_plane_agents + dev_idx
            if agent_id >= self.n_agents or not self._device_is_dispatchable(device):
                continue

            requests = []
            for req in self.request_list[1:]:
                if not self._device_can_dispatch(device, req):
                    continue
                requests.append({
                    'id': int(req.get('id', -1)),
                    'job_code': req.get('job_code'),
                    'site_code': req.get('site_code'),
                    'plane_id': req.get('plane_id'),
                    'needed_res_types': list(req.get('needed_res_types', [])),
                    'waiting_time': float(req.get('waiting_time', 0.0)),
                })

            if requests:
                devices_debug.append({
                    'agent_id': int(agent_id),
                    'device_idx': int(dev_idx),
                    'device_id': device.code,
                    'device_type': device.resource.type,
                    'site_code': device.site.code,
                    'requests': requests[:8],
                })

        waiting_sites = {
            job_code: list(site_codes)
            for job_code, site_codes in self.waiting_sites.items()
            if site_codes
        }
        return {
            'active_devices': devices_debug[:8],
            'waiting_sites': waiting_sites,
        }

    def _device_decision_signature(self, active_agents):
        if self.resource_policy != 'drl':
            return None

        entries = []
        for dev_idx, device in enumerate(self.device_list[:self.max_device_num]):
            agent_id = self.n_plane_agents + dev_idx
            if agent_id >= self.n_agents or not active_agents[agent_id]:
                continue
            request_ids = tuple(
                int(req['id'])
                for req in self.request_list[1:]
                if self._device_can_dispatch(device, req)
            )
            if request_ids:
                entries.append((
                    int(agent_id),
                    device.code,
                    device.resource.type,
                    device.site.code,
                    request_ids,
                ))

        if not entries:
            return None

        waiting = tuple(
            (job_code, tuple(sorted(site_codes)))
            for job_code, site_codes in sorted(self.waiting_sites.items())
            if site_codes
        )
        return (
            getattr(self, 'current_case_path', ''),
            float(self.total_time),
            tuple(entries),
            waiting,
        )

    def _check_repeated_device_decision_state(self, active_agents):
        limit = int(getattr(self, 'device_deadlock_repeat_limit', 0))
        if limit <= 0:
            return

        signature = self._device_decision_signature(active_agents)
        if signature is None:
            self._last_device_decision_signature = None
            self._device_decision_repeat_count = 0
            return

        if signature == self._last_device_decision_signature:
            self._device_decision_repeat_count += 1
        else:
            self._last_device_decision_signature = signature
            self._device_decision_repeat_count = 1

        if self._device_decision_repeat_count > limit:
            active_debug = self._active_device_debug()
            raise RuntimeError(
                "Repeated device decision state detected before natural rollout "
                f"completion. repeat_count={self._device_decision_repeat_count}, "
                f"case_id={getattr(self, 'current_case_path', '')}, "
                f"env_steps={self.steps}, env_total_time={self.total_time}, "
                f"active_device_debug={active_debug}. "
                "This usually indicates a mobile-resource request that is being "
                "reissued without changing environment state."
            )

    def _register_plane_cycle(self, plane, reason):
        pid = int(plane.code.split('_')[-1])
        self.cycle_terminated = True
        self.cycle_agent_ids.add(pid)
        if not self.cycle_reason:
            self.cycle_reason = (
                f"{plane.code}: {reason}; site={plane.site.code}, "
                f"left_jobs={sorted(plane.left_jobs)}, "
                f"relocations_since_progress={plane.relocations_since_progress}, "
                f"no_progress_decisions={plane.no_progress_decisions}"
            )

    def _finalize_plane_action_record(self, plane, record):
        progress_before = int(record.get('irreversible_progress_before', 0))
        progress_after = int(plane.irreversible_progress_count)
        relocation_count = max(
            0,
            int(plane.total_relocations) - int(record.get('relocations_before', 0)),
        )
        new_first_completions = max(
            0,
            len(plane.ever_finished_jobs) - int(record.get('ever_finished_before', 0)),
        )
        irreversible_delta = max(0, progress_after - progress_before)
        repeat_relocation = bool(
            relocation_count > 0
            and irreversible_delta == 0
            and record.get('target_site_code') == record.get('previous_departed_site_code')
        )
        reset_jobs = (
            list(plane.last_transport_reset_jobs)
            if relocation_count > 0
            else []
        )

        record.update({
            'new_first_completions': new_first_completions,
            'irreversible_progress_delta': irreversible_delta,
            'relocation_count': relocation_count,
            'repeat_relocation': repeat_relocation,
            'reset_long_jobs': reset_jobs,
            'no_progress_decisions': int(plane.no_progress_decisions),
        })

        if irreversible_delta > 0:
            plane.no_progress_decisions = 0
            plane.decision_signature_counts.clear()
            return record

        plane.no_progress_decisions += 1
        signature = (
            plane.site.code,
            tuple(sorted(plane.left_jobs)),
            tuple(sorted(plane.current_jobs)),
            tuple(sorted(plane.finished_jobs)),
        )
        repeat_count = plane.decision_signature_counts.get(signature, 0) + 1
        plane.decision_signature_counts[signature] = repeat_count
        record['decision_signature_repeat_count'] = repeat_count
        record['no_progress_decisions'] = int(plane.no_progress_decisions)

        reasons = []
        if self.plane_cycle_repeat_limit > 0 and repeat_count > self.plane_cycle_repeat_limit:
            reasons.append(f"repeated post-decision state {repeat_count} times")
        if (
            self.plane_no_progress_limit > 0
            and plane.no_progress_decisions > self.plane_no_progress_limit
        ):
            reasons.append(f"no irreversible progress for {plane.no_progress_decisions} decisions")
        if (
            self.plane_relocation_limit > 0
            and plane.relocations_since_progress > self.plane_relocation_limit
        ):
            reasons.append(
                f"{plane.relocations_since_progress} relocations without irreversible progress"
            )
        if reasons:
            self._register_plane_cycle(plane, '; '.join(reasons))
        return record

    def _complete_pending_plane_action(self, plane_id, plane):
        """Finalize one plane action at the exact physical completion time."""
        record = self.pending_actions.pop(plane_id, None)
        if record is None:
            return None
        record['trans_time'] = float(plane.trans_time)
        record['job_time'] = float(plane.job_time)
        record['total_job_time'] = float(plane.total_job_time)
        record['waiting_time'] = max(
            0.0,
            float(self.total_time)
            - float(record['start_time'])
            - record['trans_time']
            - record['job_time'],
        )
        record['end_time'] = float(self.total_time)
        self._finalize_plane_action_record(plane, record)
        self.trajectory_log.append(record)
        return record

    def _build_plane_pair_mask(
        self,
        plane,
        own_op_mask,
        ptr_site_mask,
        job_site_mask,
    ):
        pair_mask = ptr_site_mask[None, :] & job_site_mask
        current_site_idx = self.site_code_list.index(plane.site.code)

        # Long-occupancy work must stay at the current stand whenever that
        # stand can execute it. Relocating it only resets completed work.
        for job_code in plane.LONG_OCCUPANCY_JOBS:
            if job_code not in self.job_code_list:
                continue
            job_idx = self.job_code_list.index(job_code)
            if pair_mask[job_idx, current_site_idx]:
                pair_mask[job_idx, :] = False
                pair_mask[job_idx, current_site_idx] = True

        # Takeoff is a two-operation sequence. ZY-S claims a free runway;
        # ZY-F must finish on that same runway and can never imply relocation.
        departure_jobs = getattr(self, 'departure_job_code_list', ())
        if len(departure_jobs) >= 2:
            first_departure = departure_jobs[0]
            first_idx = self.job_code_list.index(first_departure)
            assigned_runway = getattr(
                self, '_departure_runway_plan', {}
            ).get(plane.code)
            pair_mask[first_idx, :] = False
            if assigned_runway in self.site_code_list:
                assigned_idx = self.site_code_list.index(assigned_runway)
                pair_mask[first_idx, assigned_idx] = bool(
                    ptr_site_mask[assigned_idx]
                    and job_site_mask[first_idx, assigned_idx]
                )

            final_departure = departure_jobs[-1]
            final_idx = self.job_code_list.index(final_departure)
            if pair_mask[final_idx, current_site_idx]:
                pair_mask[final_idx, :] = False
                pair_mask[final_idx, current_site_idx] = True

        # Do not immediately return to the stand just left unless no other
        # currently ready operation-site pair exists.
        departed = plane.last_departed_site_code
        if plane.relocations_since_progress > 0 and departed in self.site_code_list:
            departed_idx = self.site_code_list.index(departed)
            candidate = pair_mask.copy()
            candidate[:, departed_idx] = False
            if (own_op_mask[:, None] & candidate).any():
                pair_mask = candidate
        return pair_mask

    def _nearest_request_for_device(self, device, requests):
        """Choose the nearest legal request, breaking ties by longer waiting time."""
        if not requests:
            return None

        def score(request):
            site = self.sites[request['site_code']]
            distance = abs(device.site.pos[0] - site.pos[0]) + abs(device.site.pos[1] - site.pos[1])
            travel_time = distance / max(float(getattr(device, 'velocity', 1.0)), 1.0)
            return (
                travel_time,
                -float(request.get('waiting_time', 0.0)),
                int(request.get('id', 0)),
            )

        return min(requests, key=score)

    def _heuristic_request_supervision_score(self, device, request):
        """Return the dense utility underlying the heuristic dispatcher.

        Higher is better.  This exposes a score for every legal request, not
        merely the categorical Hungarian winner, so Stage2 can learn relative
        dispatch quality across the complete live candidate set.
        """

        target = self.sites[request['site_code']]
        distance = (
            abs(float(device.site.pos[0]) - float(target.pos[0]))
            + abs(float(device.site.pos[1]) - float(target.pos[1]))
        )
        travel_time = distance / max(float(device.velocity), 1.0)
        if request.get('is_lookahead', False):
            lead_time = max(0.0, float(request.get('lead_time', 0.0)))
            lateness = max(0.0, travel_time - lead_time)
            earliness = max(0.0, lead_time - travel_time)
            cost = (
                10.0 * lateness
                + 0.01 * earliness
                + 1e-6 * distance
            )
        else:
            cost = (
                travel_time
                - float(request.get('waiting_time', 0.0))
            )
        return float(-cost - 1e-9 * int(request.get('id', 0)))

    def _heuristic_device_assignment_preferences(self):
        """
        Build non-mutating Hungarian preferences from the original heuristic dispatcher.

        The returned mapping is only a preference. Final labels are still emitted in
        agent order so every request action remains legal under the DRL contention mask.
        """
        preferences = {}
        assigned_devices = set()
        claimed_requests = set()

        for job_code, waiting_sites_list in self.waiting_sites.items():
            if not waiting_sites_list:
                continue
            needed_res_types = (
                ['R014']
                if job_code == 'ZY-T'
                else sorted(res for res in self.jobs.get(job_code).resources if res in self.mobile_devices)
            )
            if not needed_res_types:
                continue

            idle_devices = [
                device for device in self.get_idle_devices(needed_res_types)
                if device.code not in assigned_devices
            ]
            if not idle_devices:
                continue

            candidate_requests = [
                req for req in self.request_list[1:]
                if req['job_code'] == job_code
                and not req.get('is_lookahead', False)
                and req['site_code'] in waiting_sites_list
                and req['id'] not in claimed_requests
                and any(self._device_can_dispatch(device, req) for device in idle_devices)
            ]
            if not candidate_requests:
                continue

            waiting_planes = []
            for req in candidate_requests:
                plane = self.planes.get(req.get('plane_id'))
                if plane is None:
                    continue
                waiting_planes.append(plane)
            if not waiting_planes:
                continue

            for device, target_site in arrange_devices(idle_devices, waiting_planes):
                if device.code in assigned_devices:
                    continue
                request = next(
                    (
                        req for req in candidate_requests
                        if req['site_code'] == target_site.code
                        and req['id'] not in claimed_requests
                        and self._device_can_dispatch(device, req)
                    ),
                    None,
                )
                if request is None:
                    continue
                preferences[device.code] = int(request['id'])
                assigned_devices.add(device.code)
                claimed_requests.add(int(request['id']))

        departure_requests = sorted(
            (
                request for request in self.request_list[1:]
                if request.get('request_kind') == 'departure_pickup'
                and int(request['id']) not in claimed_requests
            ),
            key=lambda request: int(request['id']),
        )
        departure_devices = sorted(
            (
                device for device in self.device_list[:self.max_device_num]
                if device.code not in assigned_devices
                and self._is_transporter_device(device)
                and self._device_is_dispatchable(device)
            ),
            key=lambda device: device.code,
        )
        if departure_devices and departure_requests:
            infeasible = 1e12
            costs = np.full(
                (len(departure_devices), len(departure_requests)),
                infeasible,
                dtype=np.float64,
            )
            for row, device in enumerate(departure_devices):
                for column, request in enumerate(departure_requests):
                    if not self._device_can_dispatch(device, request):
                        continue
                    target = self.sites[request['site_code']]
                    distance = (
                        abs(float(device.site.pos[0]) - float(target.pos[0]))
                        + abs(float(device.site.pos[1]) - float(target.pos[1]))
                    )
                    travel_time = distance / max(
                        float(device.velocity), 1.0
                    )
                    costs[row, column] = (
                        travel_time
                        - float(request.get('waiting_time', 0.0))
                        + 1e-9 * int(request['id'])
                    )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[int(row), int(column)] >= infeasible:
                    continue
                device = departure_devices[int(row)]
                request = departure_requests[int(column)]
                preferences[device.code] = int(request['id'])
                assigned_devices.add(device.code)
                claimed_requests.add(int(request['id']))

        if self.device_lookahead_dispatch:
            lookahead_requests = sorted(
                (
                    request for request in self.request_list[1:]
                    if request.get('is_lookahead', False)
                    and int(request['id']) not in claimed_requests
                ),
                key=lambda request: int(request['id']),
            )
            idle_devices = sorted(
                (
                    device for device in self.device_list[:self.max_device_num]
                    if device.code not in assigned_devices
                    and self._device_is_dispatchable(device)
                ),
                key=lambda device: device.code,
            )
            if idle_devices and lookahead_requests:
                infeasible = 1e12
                costs = np.full(
                    (len(idle_devices), len(lookahead_requests)),
                    infeasible,
                    dtype=np.float64,
                )
                for row, device in enumerate(idle_devices):
                    for column, request in enumerate(lookahead_requests):
                        if not self._device_can_dispatch(device, request):
                            continue
                        target = self.sites[request['site_code']]
                        distance = (
                            abs(float(device.site.pos[0]) - float(target.pos[0]))
                            + abs(float(device.site.pos[1]) - float(target.pos[1]))
                        )
                        travel_time = distance / max(
                            float(device.velocity), 1.0
                        )
                        lead_time = max(0.0, float(request.get('lead_time', 0.0)))
                        # A lookahead request is an option, not an instruction
                        # to occupy the destination as soon as it is exposed.
                        # Defer dispatch until the device would arrive no more
                        # than the configured margin before the plane needs it.
                        if (
                            travel_time + self.device_lookahead_safety_margin
                            < lead_time
                        ):
                            continue
                        lateness = max(0.0, travel_time - lead_time)
                        earliness = max(0.0, lead_time - travel_time)
                        costs[row, column] = (
                            10.0 * lateness
                            + 0.01 * earliness
                            + 1e-6 * distance
                            + 1e-9 * int(request['id'])
                        )
                row_indices, column_indices = linear_sum_assignment(costs)
                for row, column in zip(row_indices, column_indices):
                    if costs[int(row), int(column)] >= infeasible:
                        continue
                    device = idle_devices[int(row)]
                    request = lookahead_requests[int(column)]
                    preferences[device.code] = int(request['id'])
                    assigned_devices.add(device.code)
                    claimed_requests.add(int(request['id']))

        return preferences

    def _load_iga_teacher(self):
        teacher_dir = str(self.config.get('iga_teacher_dir', '') or '')
        case_path = str(getattr(self, 'current_case_path', '') or '')
        cache_key = (teacher_dir, case_path)
        if getattr(self, '_iga_teacher_cache_key', None) == cache_key:
            return getattr(self, '_iga_teacher_cache', None)
        self._iga_teacher_cache_key = cache_key
        self._iga_teacher_cache = None
        if not teacher_dir or not case_path:
            return None
        case_name = os.path.basename(case_path.rstrip(os.sep))
        teacher_path = os.path.join(teacher_dir, f'{case_name}.json')
        if not os.path.isfile(teacher_path):
            return None
        with open(teacher_path, 'r', encoding='utf-8') as teacher_file:
            teacher = json.load(teacher_file)
        metadata_path = os.path.join(case_path, "metadata.json")
        expected_case_sha256 = None
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                expected_case_sha256 = json.load(metadata_file).get("case_sha256")
        try:
            teacher_makespan = float(teacher.get("makespan", math.inf))
            teacher_schema = int(teacher.get("schema_version", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"IGA teacher {teacher_path} has invalid metadata."
            ) from error
        if (
            teacher_schema < 2
            or teacher.get("teacher_scope") != "stage1_plane_policy"
            or teacher.get("resource_policy") != "heuristic"
            or not teacher.get("completion_verified", False)
            or teacher.get("environment_semantics_version")
            != self.SEMANTICS_VERSION
            or not math.isfinite(teacher_makespan)
            or teacher_makespan >= 100000.0
            or teacher.get("case") != case_name
            or not teacher.get("case_sha256")
            or (
                expected_case_sha256 is not None
                and teacher.get("case_sha256") != expected_case_sha256
            )
        ):
            raise ValueError(
                f"IGA teacher {teacher_path} is not an independently verified "
                "Stage-1 plane-policy teacher with heuristic resources for "
                f"this case and semantics {self.SEMANTICS_VERSION}."
            )
        job_priorities = np.asarray(
            teacher.get('job_priorities', []), dtype=np.float32
        )
        site_priorities = np.asarray(
            teacher.get('site_priorities', []), dtype=np.float32
        )
        actual_plane_count = len(self.flights_data)
        expected_jobs = (actual_plane_count, len(self.job_code_list))
        expected_sites = (actual_plane_count, len(self.site_code_list))
        if job_priorities.shape != expected_jobs:
            raise ValueError(
                f'IGA teacher {teacher_path} job priorities have shape '
                f'{job_priorities.shape}, expected {expected_jobs}.'
            )
        if site_priorities.shape != expected_sites:
            raise ValueError(
                f'IGA teacher {teacher_path} site priorities have shape '
                f'{site_priorities.shape}, expected {expected_sites}.'
            )
        self._iga_teacher_cache = {
            'path': teacher_path,
            'job_priorities': job_priorities,
            'site_priorities': site_priorities,
            'teacher_cmax': teacher_makespan,
        }
        return self._iga_teacher_cache

    def iga_teacher_actions(self, return_info=False):
        """Decode the stored IGA chromosome with authoritative live masks."""
        teacher = self._load_iga_teacher()
        actions = np.full((self.n_agents, 3), -1, dtype=np.int32)
        if teacher is None:
            result = {
                'actions': actions,
                'info': {'available': False, 'active_planes': 0},
            }
            return result if return_info else actions

        n_jobs = len(self.job_code_list)
        n_sites = len(self.site_code_list)
        op_mask = np.asarray(self.agent_op_mask, dtype=bool)
        site_mask = np.asarray(self.ptr_site_mask_matrix, dtype=bool)
        pair_mask = np.asarray(
            self.agent_job_site_mask_matrix, dtype=bool
        )
        preferences = {}
        best_scores = {}
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            if (
                pid >= self.n_plane_agents
                or not self._plane_can_decide(plane)
            ):
                continue
            candidates = []
            for job_idx, site_idx in np.argwhere(pair_mask[pid]):
                op_global_idx = pid * n_jobs + int(job_idx)
                site_idx = int(site_idx)
                if (
                    not op_mask[pid, op_global_idx]
                    or not site_mask[pid, site_idx]
                ):
                    continue
                score = float(
                    teacher['job_priorities'][pid, int(job_idx)]
                    + teacher['site_priorities'][pid, site_idx]
                )
                candidates.append({
                    'job_idx': int(job_idx),
                    'site_idx': site_idx,
                    'op_global_idx': op_global_idx,
                    'score': score,
                })
            if not candidates:
                raise RuntimeError(
                    f'IGA teacher has no legal action for active plane {pid}.'
                )
            candidates.sort(
                key=lambda item: (
                    -item['score'], item['job_idx'], item['site_idx']
                )
            )
            preferences[pid] = candidates
            best_scores[pid] = candidates[0]['score']

        pid_order = sorted(
            preferences,
            key=lambda pid: (-best_scores[pid], pid),
        )
        site_matches = {}

        def augment(pid, seen_sites):
            for candidate in preferences[pid]:
                site_idx = candidate['site_idx']
                if site_idx in seen_sites:
                    continue
                seen_sites.add(site_idx)
                previous = site_matches.get(site_idx)
                if previous is None or augment(previous[0], seen_sites):
                    site_matches[site_idx] = (pid, candidate)
                    return True
            return False

        for pid in pid_order:
            if not augment(pid, set()):
                raise RuntimeError(
                    f'IGA teacher cannot find a unique-site matching for plane {pid}.'
                )
        selected = {
            pid: candidate for pid, candidate in site_matches.values()
        }
        for rank, pid in enumerate(pid_order):
            candidate = selected[pid]
            actions[pid] = [
                candidate['op_global_idx'],
                candidate['site_idx'],
                rank,
            ]
        info = {
            'available': True,
            'active_planes': len(pid_order),
            'teacher_path': teacher['path'],
            'teacher_cmax': teacher['teacher_cmax'],
        }
        result = {'actions': actions, 'info': info}
        return result if return_info else actions

    def _joint_iga_planning_contract(self):
        """Return every decision-opportunity field bound by Stage3 labels."""

        return {
            'device_lookahead_dispatch': bool(self.device_lookahead_dispatch),
            'device_lookahead_safety_margin': float(
                self.device_lookahead_safety_margin
            ),
            'device_deadline_aware_dispatch': bool(
                self.device_deadline_aware_dispatch
            ),
            'device_future_intent_horizon': int(
                self.device_future_intent_horizon
            ),
            'device_future_intent_mode': str(self.device_future_intent_mode),
            'device_frontier_max_requests': int(
                self.device_frontier_max_requests
            ),
            'resource_release_aware_eta': bool(self.resource_release_aware_eta),
            'device_lookahead_reservation_mode': str(
                self.device_lookahead_reservation_mode
            ),
            'device_reservation_grace_seconds': float(
                self.device_reservation_grace_seconds
            ),
            'device_departure_lookahead': bool(self.device_departure_lookahead),
        }

    def _load_joint_iga_teacher_index(self, index_path):
        index_path = os.path.realpath(index_path)
        cache_key = (
            index_path,
            os.path.getmtime(index_path) if os.path.isfile(index_path) else None,
        )
        if getattr(self, '_joint_iga_index_cache_key', None) == cache_key:
            return getattr(self, '_joint_iga_index_cache', None)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f'Stage3 joint IGA teacher index not found: {index_path}'
            )
        with open(index_path, 'r', encoding='utf-8') as handle:
            index = json.load(handle)
        expected_planning = self._joint_iga_planning_contract()
        if (
            int(index.get('schema_version', 0)) != 1
            or index.get('teacher_scope') != 'stage3_full_joint_policy'
            or index.get('teacher_method') != 'joint_iga_all'
            or index.get('environment_semantics_version') != self.SEMANTICS_VERSION
            or index.get('planning_contract') != expected_planning
            or not isinstance(index.get('entries'), dict)
        ):
            raise ValueError(
                'Invalid or planning-incompatible Stage3 joint IGA teacher '
                f'index: {index_path}'
            )
        self._joint_iga_index_cache_key = cache_key
        self._joint_iga_index_cache = index
        return index

    def _load_joint_iga_teacher(self):
        """Load one replay-bound full-joint chromosome for live-mask BC."""

        teacher_dir = str(
            self.config.get('joint_iga_teacher_dir', '') or ''
        )
        index_path = str(
            self.config.get('joint_iga_teacher_index', '') or ''
        )
        case_path = str(getattr(self, 'current_case_path', '') or '')
        cache_key = (teacher_dir, index_path, case_path)
        if getattr(self, '_joint_iga_teacher_cache_key', None) == cache_key:
            return getattr(self, '_joint_iga_teacher_cache', None)
        if not teacher_dir or not case_path:
            self._joint_iga_teacher_cache_key = cache_key
            self._joint_iga_teacher_cache = None
            return None
        if not index_path:
            raise ValueError(
                'Stage3 joint IGA BC requires joint_iga_teacher_index; '
                'refusing an unbound chromosome.'
            )

        teacher_dir = os.path.realpath(teacher_dir)
        index = self._load_joint_iga_teacher_index(index_path)
        if os.path.realpath(str(index.get('teacher_dir', ''))) != teacher_dir:
            raise ValueError(
                'Stage3 joint teacher directory does not match its index.'
            )
        case_name = os.path.basename(case_path.rstrip(os.sep))
        entry = index['entries'].get(case_name)
        if not isinstance(entry, dict) or not entry.get('replay_verified', False):
            raise FileNotFoundError(
                f'No replay-verified Stage3 joint teacher for {case_name}.'
            )
        teacher_path = os.path.join(teacher_dir, f'{case_name}.json')
        if not os.path.isfile(teacher_path):
            raise FileNotFoundError(
                f'Indexed Stage3 joint teacher is missing: {teacher_path}'
            )
        teacher_sha = self._sha256_file(teacher_path)
        if teacher_sha != entry.get('teacher_sha256'):
            raise ValueError(
                f'Stage3 joint teacher SHA256 mismatch for {case_name}.'
            )

        metadata_path = os.path.join(case_path, 'metadata.json')
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f'Dataset metadata missing for Stage3 teacher: {metadata_path}'
            )
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            case_sha = self._case_fingerprint(json.load(handle))
        if not case_sha or case_sha != entry.get('case_sha256'):
            raise ValueError(
                f'Dataset fingerprint mismatch for Stage3 teacher {case_name}.'
            )

        with open(teacher_path, 'r', encoding='utf-8') as handle:
            teacher = json.load(handle)
        try:
            chromosome = np.asarray(
                teacher.get('search', {}).get('chromosome', []),
                dtype=np.float64,
            ).reshape(-1)
            teacher_cmax = float(entry['replay_makespan'])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(
                f'Stage3 joint teacher has invalid numeric data: {teacher_path}'
            ) from error
        if (
            int(teacher.get('schema_version', 0)) != 1
            or teacher.get('teacher_scope') != 'stage3_full_joint_policy'
            or teacher.get('teacher_method') != 'joint_iga_all'
            or teacher.get('resource_policy') != 'drl'
            or teacher.get('backends')
            not in (
                ['ordinary', 'transporter'],
                {'ordinary': 'iga', 'transporter': 'iga'},
            )
            or teacher.get('environment_semantics_version') != self.SEMANTICS_VERSION
            or teacher.get('case') != case_name
            or teacher.get('case_sha256') != case_sha
            or not teacher.get('completion_verified', False)
            or not teacher.get('completed', False)
            or not math.isfinite(teacher_cmax)
            or teacher_cmax >= 100000.0
        ):
            raise ValueError(
                f'{teacher_path} is not a verified Stage3 joint IGA teacher.'
            )

        n_jobs = len(self.job_code_list)
        n_sites = len(self.site_code_list)
        job_end = self.n_plane_agents * n_jobs
        plane_end = job_end + self.n_plane_agents * n_sites
        resource_layout = ResourceGenomeLayout.from_env(
            self, {'ordinary': 'iga', 'transporter': 'iga'}
        )
        expected_n_var = plane_end + resource_layout.n_var
        if chromosome.size != expected_n_var:
            raise ValueError(
                f'Stage3 joint layout mismatch for {case_name}: '
                f'{chromosome.size} != {expected_n_var}.'
            )
        recorded_n_var = int(teacher.get('search', {}).get('n_var', -1))
        if recorded_n_var != expected_n_var:
            raise ValueError(
                f'Stage3 teacher recorded n_var={recorded_n_var}, '
                f'expected {expected_n_var}.'
            )
        teacher_payload = {
            'path': teacher_path,
            'teacher_sha256': teacher_sha,
            'teacher_cmax': teacher_cmax,
            'original_teacher_cmax': float(teacher.get('makespan', math.inf)),
            'job_priorities': chromosome[:job_end].reshape(
                self.n_plane_agents, n_jobs
            ),
            'site_priorities': chromosome[job_end:plane_end].reshape(
                self.n_plane_agents, n_sites
            ),
            'decoded': resource_layout.decode(chromosome[plane_end:]),
            'backends': {'ordinary': 'iga', 'transporter': 'iga'},
            'intrinsic_ready_time_index': (
                self._intrinsic_ready_time_label_index(teacher)
            ),
        }
        # Publish the cache key only after every provenance, shape, and MDP
        # check succeeds.  A transient or corrupted teacher must keep failing
        # closed on retries instead of being mistaken for an unavailable
        # optional teacher after the first exception.
        self._joint_iga_teacher_cache = teacher_payload
        self._joint_iga_teacher_cache_key = cache_key
        return teacher_payload

    def joint_iga_plane_teacher_actions(self, return_info=False):
        """Decode the Stage3 plane half against authoritative live masks."""

        teacher = self._load_joint_iga_teacher()
        actions = np.full((self.n_agents, 3), -1, dtype=np.int32)
        if teacher is None:
            result = {
                'actions': actions,
                'info': {'available': False, 'active_planes': 0},
            }
            return result if return_info else actions

        n_jobs = len(self.job_code_list)
        op_mask = np.asarray(self.agent_op_mask, dtype=bool)
        site_mask = np.asarray(self.ptr_site_mask_matrix, dtype=bool)
        pair_mask = np.asarray(
            self.agent_job_site_mask_matrix, dtype=bool
        )
        preferences = {}
        best_scores = {}
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            if pid >= self.n_plane_agents or not self._plane_can_decide(plane):
                continue
            candidates = []
            for job_idx, site_idx in np.argwhere(pair_mask[pid]):
                op_global_idx = pid * n_jobs + int(job_idx)
                site_idx = int(site_idx)
                if (
                    not op_mask[pid, op_global_idx]
                    or not site_mask[pid, site_idx]
                ):
                    continue
                score = float(
                    teacher['job_priorities'][pid, int(job_idx)]
                    + teacher['site_priorities'][pid, site_idx]
                )
                candidates.append({
                    'job_idx': int(job_idx),
                    'site_idx': site_idx,
                    'op_global_idx': op_global_idx,
                    'score': score,
                })
            if not candidates:
                raise RuntimeError(
                    f'Joint IGA teacher has no legal action for active plane {pid}.'
                )
            candidates.sort(
                key=lambda item: (
                    -item['score'], item['job_idx'], item['site_idx']
                )
            )
            preferences[pid] = candidates
            best_scores[pid] = candidates[0]['score']

        pid_order = sorted(
            preferences, key=lambda pid: (-best_scores[pid], pid)
        )
        site_matches = {}

        def augment(pid, seen_sites):
            for candidate in preferences[pid]:
                site_idx = candidate['site_idx']
                if site_idx in seen_sites:
                    continue
                seen_sites.add(site_idx)
                previous = site_matches.get(site_idx)
                if previous is None or augment(previous[0], seen_sites):
                    site_matches[site_idx] = (pid, candidate)
                    return True
            return False

        for pid in pid_order:
            if not augment(pid, set()):
                raise RuntimeError(
                    f'Joint IGA teacher cannot match a unique site for plane {pid}.'
                )
        selected = {
            pid: candidate for pid, candidate in site_matches.values()
        }
        for rank, pid in enumerate(pid_order):
            candidate = selected[pid]
            actions[pid] = [
                candidate['op_global_idx'], candidate['site_idx'], rank
            ]
        result = {
            'actions': actions,
            'info': {
                'available': True,
                'active_planes': len(pid_order),
                'teacher_path': teacher['path'],
                'teacher_sha256': teacher['teacher_sha256'],
                'teacher_cmax': teacher['teacher_cmax'],
            },
        }
        return result if return_info else actions

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _case_fingerprint(metadata):
        direct = metadata.get('case_sha256')
        if direct:
            return str(direct)
        fingerprints = metadata.get('fingerprints', {})
        return str(fingerprints.get('case_sha256') or '')

    def _load_resource_iga_teacher_index(self, index_path):
        index_path = os.path.realpath(index_path)
        cache_key = (
            index_path,
            os.path.getmtime(index_path) if os.path.isfile(index_path) else None,
        )
        if getattr(self, '_resource_iga_index_cache_key', None) == cache_key:
            return getattr(self, '_resource_iga_index_cache', None)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f'Stage2 resource IGA teacher index not found: {index_path}'
            )
        with open(index_path, 'r', encoding='utf-8') as handle:
            index = json.load(handle)
        schema_version = int(index.get('schema_version', 0))
        if (
            schema_version not in {1, 2}
            or index.get('teacher_scope') != 'stage2_resource_policy'
            or index.get('teacher_method') != 'resource_iga_all'
            or index.get('environment_semantics_version') != self.SEMANTICS_VERSION
            or not isinstance(index.get('entries'), dict)
        ):
            raise ValueError(
                f'Invalid Stage2 resource IGA teacher index: {index_path}'
            )
        entries = index['entries']
        if not entries or int(index.get('case_count', len(entries))) != len(entries):
            raise ValueError(
                f'Invalid Stage2 resource IGA teacher case coverage: {index_path}'
            )
        if schema_version >= 2:
            observed = int(index.get(
                'intrinsic_ready_time_observed_request_count', 0
            ))
            labeled = int(index.get(
                'intrinsic_ready_time_labeled_request_count', 0
            ))
            coverage = float(index.get(
                'intrinsic_ready_time_label_coverage', -1.0
            ))
            if (
                int(index.get(
                    'intrinsic_ready_time_label_schema_version', 0
                )) != self.INTRINSIC_READY_TIME_LABEL_SCHEMA_VERSION
                or index.get('intrinsic_ready_time_semantics')
                != self.INTRINSIC_READY_TIME_SEMANTICS
                or observed <= 0
                or labeled <= 0
                or labeled > observed
                or not math.isclose(
                    coverage,
                    labeled / observed,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    'Invalid Stage2 intrinsic-ready label coverage in '
                    f'{index_path}'
                )
        self._resource_iga_index_cache_key = cache_key
        self._resource_iga_index_cache = index
        return index

    def _resource_teacher_planning_contract(self):
        """Return every planning field that changes Stage2 teacher labels."""

        return {
            'device_lookahead_dispatch': bool(
                self.device_lookahead_dispatch
            ),
            'device_lookahead_safety_margin': float(
                self.device_lookahead_safety_margin
            ),
            'device_deadline_aware_dispatch': bool(
                self.device_deadline_aware_dispatch
            ),
            'device_future_intent_horizon': int(
                self.device_future_intent_horizon
            ),
            'device_future_intent_mode': str(
                self.device_future_intent_mode
            ),
            'device_frontier_max_requests': int(
                self.device_frontier_max_requests
            ),
            'resource_release_aware_eta': bool(
                self.resource_release_aware_eta
            ),
            'device_lookahead_reservation_mode': str(
                self.device_lookahead_reservation_mode
            ),
            'device_reservation_grace_seconds': float(
                self.device_reservation_grace_seconds
            ),
            'device_departure_lookahead': bool(
                self.device_departure_lookahead
            ),
        }

    @staticmethod
    def _intrinsic_ready_time_label_key(record, observation_time=None):
        """Bind one ready-time label to the exact request context.

        A physical timestamp alone is not a unique event identifier.  The
        event loop can expose two consecutive request frontiers without
        advancing physical time (apart from floating-point noise).  In that
        situation the same aircraft/job/site may change from a deeper DAG
        forecast to a direct successor.  Matching only a rounded timestamp and
        request identity can therefore attach the direct-successor target to
        the earlier, deeper request.

        ``request_kind`` and the DAG lower bound are immutable parts of the
        observed request state and distinguish those frontiers.  Six decimal
        places still absorb harmless replay arithmetic noise while preserving
        the semantic context in the remaining key fields.
        """

        try:
            observed_at = float(
                record['observation_time']
                if observation_time is None else observation_time
            )
            raw_dag_lower_bound = record.get('dag_lower_bound_seconds')
            if raw_dag_lower_bound is None:
                # Blocking requests are already intrinsically ready and the
                # live request payload historically omitted ``lead_time``.
                # Their deterministic DAG lower bound is therefore zero,
                # matching the explicit value stored in schema-v1 labels.
                raw_dag_lower_bound = record.get('lead_time', 0.0)
            dag_lower_bound = float(raw_dag_lower_bound)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                'Intrinsic ready labels require numeric observation_time '
                'and DAG lower-bound context.'
            ) from error
        request_kind = str(record.get('request_kind', '')).strip()
        identity = (
            str(record.get('plane_id')),
            str(record.get('job_code')),
            str(record.get('site_code')),
        )
        if (
            not math.isfinite(observed_at)
            or not math.isfinite(dag_lower_bound)
            or dag_lower_bound < -1e-6
            or not request_kind
            or any(value in {'', 'None'} for value in identity)
        ):
            raise ValueError(
                'Invalid intrinsic ready-time request context: '
                f'{record!r}'
            )
        return (
            round(observed_at, 6),
            *identity,
            request_kind,
            round(max(0.0, dag_lower_bound), 6),
        )

    @staticmethod
    def _intrinsic_ready_time_label_index(teacher):
        """Index exact absolute ready-time labels from a baseline replay.

        New label generators may either attach ``intrinsic_ready_time`` to
        every request in ``decision_trace`` or emit the equivalent flat
        ``intrinsic_ready_time_labels`` records.  A timestamp plus immutable
        request identity prevents labels from a different visited state being
        reused after a DAgger deviation.
        """

        records = list(teacher.get('intrinsic_ready_time_labels', []) or [])
        for event in teacher.get('decision_trace', []) or []:
            observation_time = event.get('time_before')
            for request in event.get('requests', []) or []:
                if request.get('intrinsic_ready_time') is None:
                    continue
                records.append({
                    **request,
                    'observation_time': observation_time,
                })
        if records:
            expected_schema = (
                AircraftScheduleEnv.INTRINSIC_READY_TIME_LABEL_SCHEMA_VERSION
            )
            if int(teacher.get(
                'intrinsic_ready_time_label_schema_version', 0
            )) != expected_schema:
                raise ValueError(
                    'Intrinsic ready-time labels require schema version '
                    f'{expected_schema}.'
                )
            if teacher.get('intrinsic_ready_time_semantics') != (
                AircraftScheduleEnv.INTRINSIC_READY_TIME_SEMANTICS
            ):
                raise ValueError(
                    'Intrinsic ready-time labels have missing or incompatible '
                    'target semantics.'
                )
        index = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(
                    'intrinsic_ready_time_labels entries must be mappings.'
                )
            try:
                observation_time = float(record['observation_time'])
                intrinsic_ready_time = float(
                    record['intrinsic_ready_time']
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    'Intrinsic ready labels require numeric observation_time '
                    'and intrinsic_ready_time.'
                ) from error
            if (
                not math.isfinite(observation_time)
                or not math.isfinite(intrinsic_ready_time)
            ):
                raise ValueError(
                    f'Invalid intrinsic ready-time label: {record!r}'
                )
            key = AircraftScheduleEnv._intrinsic_ready_time_label_key(
                record,
                observation_time=observation_time,
            )
            previous = index.get(key)
            if previous is not None and not math.isclose(
                previous, intrinsic_ready_time, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f'Conflicting intrinsic ready-time labels for {key}: '
                    f'{previous} != {intrinsic_ready_time}.'
                )
            index[key] = intrinsic_ready_time
        return index

    def _load_resource_iga_teacher(self):
        """Load and strictly bind a Stage2 IGA chromosome to this case.

        Historical Stage2 label files have a null top-level ``case_sha256``.
        The immutable sidecar index repairs that provenance gap without
        rewriting 600 verified trajectories: it binds the current dataset
        fingerprint and the exact teacher-file digest before decoding.
        """

        teacher_dir = str(
            self.config.get('resource_iga_teacher_dir', '') or ''
        )
        index_path = str(
            self.config.get('resource_iga_teacher_index', '') or ''
        )
        case_path = str(getattr(self, 'current_case_path', '') or '')
        cache_key = (teacher_dir, index_path, case_path)
        if getattr(self, '_resource_iga_teacher_cache_key', None) == cache_key:
            return getattr(self, '_resource_iga_teacher_cache', None)
        self._resource_iga_teacher_cache_key = cache_key
        self._resource_iga_teacher_cache = None
        if not teacher_dir or not case_path:
            return None
        if not index_path:
            raise ValueError(
                'Stage2 IGA BC requires resource_iga_teacher_index; refusing '
                'unbound teacher files.'
            )

        teacher_dir = os.path.realpath(teacher_dir)
        index = self._load_resource_iga_teacher_index(index_path)
        indexed_planning_contract = index.get(
            'resource_lookahead_contract'
        )
        if (
            indexed_planning_contract is not None
            and indexed_planning_contract
            != self._resource_teacher_planning_contract()
        ):
            raise ValueError(
                'Stage2 resource teacher planning contract differs from the '
                'live environment.'
            )
        indexed_dir = os.path.realpath(str(index.get('teacher_dir', '')))
        if indexed_dir != teacher_dir:
            raise ValueError(
                'Stage2 resource teacher directory does not match its index: '
                f'{teacher_dir} != {indexed_dir}'
            )
        case_name = os.path.basename(case_path.rstrip(os.sep))
        entry = index['entries'].get(case_name)
        if not isinstance(entry, dict):
            raise FileNotFoundError(
                f'No indexed Stage2 resource teacher for {case_name}.'
            )
        teacher_path = os.path.join(teacher_dir, f'{case_name}.json')
        if not os.path.isfile(teacher_path):
            raise FileNotFoundError(
                f'Indexed Stage2 resource teacher is missing: {teacher_path}'
            )
        actual_teacher_sha = self._sha256_file(teacher_path)
        if actual_teacher_sha != entry.get('teacher_sha256'):
            raise ValueError(
                f'Stage2 resource teacher SHA256 mismatch for {case_name}.'
            )

        metadata_path = os.path.join(case_path, 'metadata.json')
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f'Dataset metadata missing for indexed teacher: {metadata_path}'
            )
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        case_sha = self._case_fingerprint(metadata)
        if not case_sha or case_sha != entry.get('case_sha256'):
            raise ValueError(
                f'Dataset fingerprint mismatch for Stage2 teacher {case_name}.'
            )

        with open(teacher_path, 'r', encoding='utf-8') as handle:
            teacher = json.load(handle)
        try:
            teacher_cmax = float(teacher.get('makespan', math.inf))
            chromosome = np.asarray(
                teacher.get('search', {}).get('chromosome', []),
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Stage2 resource teacher has invalid numeric data: {teacher_path}'
            ) from error
        if (
            int(teacher.get('schema_version', 0)) < 1
            or teacher.get('teacher_scope') != 'stage2_resource_policy'
            or teacher.get('teacher_method') != 'resource_iga_all'
            or teacher.get('resource_policy') != 'drl'
            or teacher.get('arm') != 'iga_all'
            or teacher.get('backends')
            != {'ordinary': 'iga', 'transporter': 'iga'}
            or teacher.get('environment_semantics_version')
            != self.SEMANTICS_VERSION
            or teacher.get('case') != case_name
            or not teacher.get('completion_verified', False)
            or not teacher.get('completed', False)
            or not math.isfinite(teacher_cmax)
            or teacher_cmax >= 100000.0
            or teacher.get('frozen_plane_checkpoint_sha256')
            != index.get('frozen_plane_checkpoint_sha256')
            or (
                index.get('search_contract_version') is not None
                and teacher.get('search_contract_version')
                != index.get('search_contract_version')
            )
            or (
                indexed_planning_contract is not None
                and teacher.get('resource_lookahead_contract')
                != indexed_planning_contract
            )
        ):
            raise ValueError(
                f'{teacher_path} is not a verified Stage2 resource IGA teacher '
                f'for semantics {self.SEMANTICS_VERSION}.'
            )

        backends = {'ordinary': 'iga', 'transporter': 'iga'}
        layout = ResourceGenomeLayout.from_env(self, backends)
        decoded = layout.decode(chromosome)
        if int(teacher.get('search', {}).get('n_var', -1)) != layout.n_var:
            raise ValueError(
                f'Stage2 teacher layout mismatch for {case_name}: '
                f"{teacher.get('search', {}).get('n_var')} != {layout.n_var}."
            )
        self._resource_iga_teacher_cache = {
            'path': teacher_path,
            'teacher_sha256': actual_teacher_sha,
            'teacher_cmax': teacher_cmax,
            'decoded': decoded,
            'backends': backends,
            'intrinsic_ready_time_index': (
                self._intrinsic_ready_time_label_index(teacher)
            ),
        }
        return self._resource_iga_teacher_cache

    def _request_ready_supervision_targets(self, source, teacher=None):
        """Return request-level ready-lead labels for Stage2 supervision.

        ``intrinsic_ready_time`` is an absolute physical time measured from a
        completed IGA/heuristic replay.  DAG lead time is retained only as an
        input/lower bound; it is never substituted for the target.  Missing
        labels are left missing so the canonical Stage2 gate fails closed.
        """

        labels = []
        label_index = (
            teacher.get('intrinsic_ready_time_index', {})
            if isinstance(teacher, dict) else {}
        )
        for request in self.request_list[1:]:
            intrinsic_ready_time = request.get('intrinsic_ready_time')
            if intrinsic_ready_time is None:
                key = self._intrinsic_ready_time_label_key(
                    request,
                    observation_time=self.total_time,
                )
                intrinsic_ready_time = label_index.get(key)
            if intrinsic_ready_time is None:
                continue
            intrinsic_ready_time = float(intrinsic_ready_time)
            target = max(
                0.0, intrinsic_ready_time - float(self.total_time)
            )
            dag_lower_bound = max(
                0.0, float(request.get('lead_time', 0.0))
            )
            if (
                not math.isfinite(intrinsic_ready_time)
                or target < dag_lower_bound - 1e-6
            ):
                raise ValueError(
                    'Invalid request ready-time supervision: '
                    f"request={request.get('id')} intrinsic_ready_time="
                    f'{intrinsic_ready_time} target_lead={target} '
                    f'lower_bound={dag_lower_bound}.'
                )
            labels.append({
                'request_id': int(request['id']),
                'ready_lead_seconds': max(0.0, target),
                'intrinsic_ready_time': intrinsic_ready_time,
                'observation_time': float(self.total_time),
                'dag_lower_bound_seconds': dag_lower_bound,
                'source': str(source),
                'label_schema_version': (
                    self.INTRINSIC_READY_TIME_LABEL_SCHEMA_VERSION
                ),
                'target_semantics': self.INTRINSIC_READY_TIME_SEMANTICS,
                'plane_id': request.get('plane_id'),
                'job_code': request.get('job_code'),
                'site_code': request.get('site_code'),
                'request_kind': request.get('request_kind'),
            })
        return labels

    def stage2_cost_branch_capture(self, persist_path=None):
        """Private in-worker snapshots; never mutate/reset the live environment.

        This RPC is used only by the opt-in Stage2 cost experiment. Clearing
        previous shadow state prevents recursively copying older snapshots.
        The bytes stay inside the worker; the parent receives only a digest.
        """
        import cloudpickle
        import hashlib
        self.stage2_cost_branch_clear()
        snapshot = cloudpickle.dumps(self)
        self._stage2_cost_snapshot = snapshot
        result = {'sha256': hashlib.sha256(snapshot).hexdigest(),
                  'case': str(getattr(self, 'current_case_path', '')),
                  'time': float(self.total_time)}
        if persist_path is not None:
            import gzip
            from pathlib import Path
            path = Path(persist_path)
            # The caller owns the new experiment directory; never overwrite.
            compressed = gzip.compress(snapshot, compresslevel=1, mtime=0)
            with path.open('xb') as handle:
                handle.write(compressed)
            result.update(path=str(path), file_sha256=hashlib.sha256(compressed).hexdigest(),
                          bytes=len(compressed))
        return result

    def stage2_cost_live_status(self):
        return {'completed': bool(self._is_schedule_complete()
                    and not getattr(self, 'cycle_terminated', False)),
                'makespan': float(self.total_time),
                'case': str(getattr(self, 'current_case_path', ''))}

    def stage2_cost_branch_restore(self):
        import cloudpickle
        snapshot = getattr(self, '_stage2_cost_snapshot', None)
        if snapshot is None:
            raise RuntimeError('No captured Stage2 cost state.')
        self._stage2_cost_shadow = cloudpickle.loads(snapshot)
        return True

    def stage2_cost_branch_step(self, actions):
        shadow = getattr(self, '_stage2_cost_shadow', None)
        if shadow is None:
            raise RuntimeError('No restored Stage2 cost branch.')
        obs, reward, dones, info = shadow.step(actions)
        status = {'completed': bool(np.all(dones) and shadow._is_schedule_complete()
                    and not getattr(shadow, 'cycle_terminated', False)),
                  'makespan': float(shadow.total_time)}
        return obs, reward, dones, info, status

    def stage2_cost_branch_clear(self):
        for name in ('_stage2_cost_snapshot', '_stage2_cost_shadow'):
            self.__dict__.pop(name, None)
        return True

    def stage2_v6_reference_actions(self, teacher_dir, teacher_hashes, deployment_projection=False):
        """Read-only decoding of pinned archived genes; never runs a search."""
        if not self.config.get('stage2_resource_v6_observations', False):
            raise ValueError('Archived V6 replay is opt-in only.')
        from onpolicy.envs.HKBZ.resource_teacher import ResourceGenomeLayout, deployment_compatible_teacher
        name = os.path.basename(str(self.current_case_path).rstrip(os.sep))
        path = os.path.join(teacher_dir, name + '.json')
        if name not in teacher_hashes or self._sha256_file(path) != teacher_hashes[name]:
            raise ValueError('Unpinned archived teacher or changed teacher hash.')
        with open(path, encoding='utf-8') as stream:
            saved = json.load(stream)
        with open(os.path.join(self.current_case_path, 'metadata.json'), encoding='utf-8') as stream:
            metadata = json.load(stream)
        if saved.get('case_sha256') != self._case_fingerprint(metadata):
            raise ValueError('Archived reference belongs to different case content.')
        if saved.get('resource_lookahead_contract') != self._resource_teacher_planning_contract():
            raise ValueError('Archived reference planning contract differs from live environment.')
        backends = {'ordinary': 'iga', 'transporter': 'iga'}
        decoded = ResourceGenomeLayout.from_env(self, backends).decode(saved['search']['chromosome'])
        actions, decisions = mixed_resource_actions(self, backends, decoded, record=True)
        projection = {}
        if deployment_projection:
            actions, decisions, projection = deployment_compatible_teacher(self, actions, decisions, decoded)
        from onpolicy.utils.stage2_resource_v6_observation import bind_teacher_identities
        bind_teacher_identities(self, decisions)
        return {'actions': actions, 'info': {'available': True, 'decisions': decisions,
            'teacher_path': path, 'teacher_sha256': teacher_hashes[name],
            'archived_cmax': saved.get('makespan'), 'search_calls': 0, **projection}}

    def resource_iga_teacher_actions(self, return_info=False, include_ready_targets=True,
                                     deployment_projection=False):
        """Decode the per-case IGA resource teacher against live masks."""

        if deployment_projection and include_ready_targets:
            raise ValueError('Derived deployment-compatible teachers cannot reuse factual Ready labels.')
        actions = np.zeros((self.n_agents, 2), dtype=np.int64)
        base_stats = {
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
        }
        teacher = self._load_resource_iga_teacher()
        if teacher is None:
            result = {
                'actions': actions,
                'info': {'available': False, **base_stats},
            }
            return result if return_info else actions
        actions, decisions = mixed_resource_actions(
            self,
            teacher['backends'],
            teacher['decoded'],
            record=True,
        )
        projection_info = {}
        if deployment_projection:
            from onpolicy.envs.HKBZ.resource_teacher import deployment_compatible_teacher
            from onpolicy.utils.stage2_matching import TEACHER_PROJECTION_CONTRACT
            actions, decisions, projection_info = deployment_compatible_teacher(
                self, actions, decisions, teacher['decoded'])
            projection_info.update(teacher_decoder_contract=TEACHER_PROJECTION_CONTRACT,
                                   teacher_projection_case=str(getattr(self, 'current_case_path', '')),
                                   original_teacher_cmax=teacher['teacher_cmax'],
                                   derived_teacher_cmax=None)
            base_stats['blocking_projection_noop'] = 0
        for decision in decisions:
            selected = int(decision['selected_request_id'])
            legal_count = len(decision['legal_request_ids'])
            candidate_count = len(decision['candidate_request_ids'])
            if decision['role'] == 'transporter':
                base_stats['transporter_labels'] += 1
            else:
                base_stats['ordinary_labels'] += 1
            if legal_count > 1:
                base_stats['non_unique_candidate_masks'] += 1
            if selected:
                base_stats['real_dispatches'] += 1
                if decision['reason'] == 'preferred':
                    base_stats['preferred_assignments'] += 1
                else:
                    base_stats['greedy_legal_fills'] += 1
            else:
                cause = str(decision.get('noop_cause') or 'no_demand')
                if cause not in {
                    'temporal_defer', 'capacity_unmatched',
                    'claimed_noop', 'no_demand', 'blocking_projection_noop',
                }:
                    cause = 'no_demand'
                base_stats[cause] += 1
                # Preserve the two historical aggregate counters for old
                # dashboards, without confusing capacity/matching no-ops with
                # a genuine temporal defer target.
                if cause == 'claimed_noop':
                    base_stats['noop_after_claimed'] += 1
                elif cause == 'temporal_defer':
                    base_stats['deferred_noop'] += 1
        info = {
            'available': True,
            'teacher_path': teacher['path'],
            'teacher_sha256': teacher['teacher_sha256'],
            'teacher_cmax': None if deployment_projection else teacher['teacher_cmax'],
            'decisions': decisions,
            'request_ready_targets': self._request_ready_supervision_targets(
                'iga', teacher
            ) if include_ready_targets else [],
            'request_ready_targets_collected': bool(include_ready_targets),
            **base_stats,
            **projection_info,
        }
        if self.config.get('stage2_resource_v6_observations', False):
            from onpolicy.utils.stage2_resource_v6_observation import bind_teacher_identities
            bind_teacher_identities(self, decisions)
        result = {'actions': actions, 'info': info}
        return result if return_info else actions

    def joint_iga_resource_teacher_actions(self, return_info=False):
        """Decode the resource half of the bound Stage3 chromosome.

        The implementation intentionally mirrors
        :meth:`resource_iga_teacher_actions`: both teachers are projected
        through the same production live masks.  The only difference is the
        provenance and the chromosome layout validated by the loader.
        """

        actions = np.zeros((self.n_agents, 2), dtype=np.int64)
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
        }
        teacher = self._load_joint_iga_teacher()
        if teacher is None:
            result = {
                'actions': actions,
                'info': {'available': False, **stats},
            }
            return result if return_info else actions

        actions, decisions = mixed_resource_actions(
            self,
            teacher['backends'],
            teacher['decoded'],
            record=True,
        )
        for decision in decisions:
            selected = int(decision['selected_request_id'])
            legal_count = len(decision['legal_request_ids'])
            candidate_count = len(decision['candidate_request_ids'])
            role_key = (
                'transporter_labels'
                if decision['role'] == 'transporter'
                else 'ordinary_labels'
            )
            stats[role_key] += 1
            if legal_count > 1:
                stats['non_unique_candidate_masks'] += 1
            if selected:
                stats['real_dispatches'] += 1
                preference_key = (
                    'preferred_assignments'
                    if decision['reason'] == 'preferred'
                    else 'greedy_legal_fills'
                )
                stats[preference_key] += 1
            else:
                cause = str(decision.get('noop_cause') or 'no_demand')
                if cause not in {
                    'temporal_defer', 'capacity_unmatched',
                    'claimed_noop', 'no_demand',
                }:
                    cause = 'no_demand'
                stats[cause] += 1
                if cause == 'claimed_noop':
                    stats['noop_after_claimed'] += 1
                elif cause == 'temporal_defer':
                    stats['deferred_noop'] += 1
        result = {
            'actions': actions,
            'info': {
                'available': True,
                'teacher_path': teacher['path'],
                'teacher_sha256': teacher['teacher_sha256'],
                'teacher_cmax': teacher['teacher_cmax'],
                'decisions': decisions,
                'request_ready_targets': (
                    self._request_ready_supervision_targets(
                        'joint_iga', teacher
                    )
                ),
                **stats,
            },
        }
        return result if return_info else actions

    def heuristic_device_actions(self, return_info=False):
        """
        Return supervised labels for DRL mobile-resource agents.

        Labels use the original Hungarian dispatcher as preference, then are made
        sequentially legal in MARL agent order. A device may defer to the
        Hungarian-preferred later device, while the final compatible device is
        forced to accept a still-unclaimed request to prevent all-noop deadlock.
        """
        actions = np.zeros((self.n_agents, 2), dtype=np.int64)
        info = {
            'preferred_assignments': 0,
            'greedy_legal_fills': 0,
            'real_dispatches': 0,
            'noop_after_claimed': 0,
            'deferred_noop': 0,
            'non_unique_candidate_masks': 0,
            'request_ready_targets': [],
            'ordinary_labels': 0,
            'transporter_labels': 0,
            'decisions': [],
        }
        if self.resource_policy != 'drl':
            return {'actions': actions, 'info': info} if return_info else actions

        self._refresh_request_pool()
        info['request_ready_targets'] = (
            self._request_ready_supervision_targets('heuristic')
        )
        preferences = self._heuristic_device_assignment_preferences()
        claimed_requests = set()

        for dev_idx, device in enumerate(self.device_list[:self.max_device_num]):
            agent_id = self.n_plane_agents + dev_idx
            if agent_id >= self.n_agents or not self._device_is_dispatchable(device):
                continue

            initial_requests = [
                req for req in self.request_list[1:]
                if self._device_can_dispatch(device, req)
            ]
            valid_requests, allow_noop = self._sequential_device_options(
                dev_idx,
                claimed_requests,
            )
            if len(valid_requests) > 1:
                info['non_unique_candidate_masks'] += 1
            if not valid_requests:
                if initial_requests:
                    info['noop_after_claimed'] += 1
                actions[agent_id] = [0, 0]
                if initial_requests:
                    info['decisions'].append({
                        'agent_id': int(agent_id),
                        'device_id': device.code,
                        'device_type': device.resource.type,
                        'role': (
                            'transporter'
                            if self._is_transporter_device(device)
                            else 'ordinary'
                        ),
                        'backend': 'heuristic',
                        'candidate_request_ids': [
                            int(request['id'])
                            for request in initial_requests
                        ],
                        'legal_request_ids': [],
                        'allow_noop': True,
                        'selected_request_id': 0,
                        'selected_score': None,
                        'selected_score_margin': None,
                        'candidate_scores': {},
                        'reason': 'claimed_by_earlier_device',
                    })
                continue

            preferred_req_id = preferences.get(device.code)
            chosen_request = next(
                (req for req in valid_requests if int(req['id']) == preferred_req_id),
                None,
            )
            if chosen_request is not None:
                info['preferred_assignments'] += 1
            elif allow_noop:
                # Preserve the preferred device identity instead of forcing the
                # first compatible agent in the static device list to dispatch.
                actions[agent_id] = [0, 0]
                info['deferred_noop'] += 1
                reason = 'deferred_noop'
            else:
                chosen_request = self._nearest_request_for_device(device, valid_requests)
                info['greedy_legal_fills'] += 1
                reason = 'last_chance_fallback'

            if chosen_request is None and not allow_noop:
                raise RuntimeError(
                    f"Heuristic labeler failed to choose a legal request for device agent {agent_id}."
                )

            req_id = 0
            if chosen_request is not None:
                req_id = int(chosen_request['id'])
                actions[agent_id] = [req_id, 0]
                claimed_requests.add(req_id)
                info['real_dispatches'] += 1
                reason = (
                    'preferred'
                    if req_id == preferred_req_id
                    else 'last_chance_fallback'
                )
            candidate_scores = {
                int(request['id']): self._heuristic_request_supervision_score(
                    device, request
                )
                for request in valid_requests
            }
            selected_score = candidate_scores.get(req_id)
            alternative_scores = [
                score
                for request_id, score in candidate_scores.items()
                if request_id != req_id
            ]
            selected_margin = (
                None
                if selected_score is None or not alternative_scores
                else min(
                    abs(selected_score - score)
                    for score in alternative_scores
                )
            )
            role = (
                'transporter'
                if self._is_transporter_device(device)
                else 'ordinary'
            )
            info[f'{role}_labels'] += 1
            info['decisions'].append({
                'agent_id': int(agent_id),
                'device_id': device.code,
                'device_type': device.resource.type,
                'role': role,
                'backend': 'heuristic',
                'candidate_request_ids': [
                    int(request['id']) for request in initial_requests
                ],
                'legal_request_ids': [
                    int(request['id']) for request in valid_requests
                ],
                'allow_noop': bool(allow_noop),
                'preferred_request_id': (
                    int(preferred_req_id)
                    if preferred_req_id is not None else None
                ),
                'selected_request_id': int(req_id),
                'selected_score': selected_score,
                'selected_score_margin': selected_margin,
                'candidate_scores': candidate_scores,
                'reason': reason,
            })

        return {'actions': actions, 'info': info} if return_info else actions

    def _site_resource_types(self, site):
        return {res.type for res in getattr(site, 'resources', {}).values()}

    def _site_can_eventually_support_job(self, site, job):
        """Whether a job can be executed at a site after waiting/dispatching resources."""
        required_types = self._job_resource_types(job)
        if not required_types:
            return True
        supported_types = self._site_resource_types(site).union(set(self.mobile_devices.keys()))
        # FJSP resource lists are alternatives; Site.start_jobs assigns one.
        return not required_types.isdisjoint(supported_types)

    def _deadlock_snapshot(self):
        active_planes = []
        needed_types = set()
        waiting_owners = set()
        for plane in self.planes.values():
            if plane.is_completed_all_jobs():
                continue
            pending_job = plane.pending_job
            awaiting_departure = self._plane_awaiting_departure(plane)
            if awaiting_departure and plane.departure_staging_decided:
                needed_types.add(self.TRANSPORTER_RESOURCE_TYPE)
            if plane.is_waiting:
                waiting_owners.add((pending_job, plane.site.code))
                needed_types.update(self._needed_mobile_types(pending_job))
            active_planes.append({
                'plane': plane.code,
                'site': plane.site.code,
                'idle': plane.is_idle(),
                'waiting': plane.is_waiting,
                'pending_job': pending_job,
                'chosen_job': plane.choosed_job,
                'destination': plane.destination.code if plane.destination else None,
                'busy': plane.is_busy,
                'transporting': plane.is_transporting,
                'departure_eligible': self._plane_departure_eligible(plane),
                'departure_staging_decided': bool(
                    plane.departure_staging_decided
                ),
                'reserved_departure_transporter': (
                    self.departure_transporter_by_plane.get(plane.code)
                ),
                'left_jobs': list(plane.left_jobs),
            })

        queued = {
            job_code: list(site_codes)
            for job_code, site_codes in self.waiting_sites.items()
            if site_codes
        }
        orphaned_queue_entries = [
            (job_code, site_code)
            for job_code, site_codes in queued.items()
            for site_code in site_codes
            if (job_code, site_code) not in waiting_owners
        ]
        relevant_devices = []
        for device in self.device_list:
            if needed_types and device.resource.type not in needed_types:
                continue
            relevant_devices.append({
                'device': device.code,
                'type': device.resource.type,
                'site': device.site.code,
                'idle': device.is_idle(),
                'transporting': device.is_transporting,
                'left_trans_time': device.left_trans_time,
                'resource_available': device.resource.is_available(),
                'on_service': list(device.resource.on_service),
                'reserved_for_plane': getattr(
                    device, 'reserved_for_plane', None
                ),
                'lookahead_reservation': copy.deepcopy(getattr(
                    device, 'lookahead_reservation', None
                )),
            })
        current_departure_plan = self._compute_departure_runway_plan()
        runway_state = [
            {
                'site': code,
                'occupied': bool(self.sites[code].is_occupied),
                'plane': getattr(self.sites[code].plane, 'code', None),
                'interfered': bool(self.sites[code].is_interfered),
                'left_recovery_time': float(
                    self.sites[code].left_rec_time
                ),
                'left_job_time': float(self.sites[code].left_job_time),
                'ongoing_jobs': sorted(self.sites[code].onging_jobs),
            }
            for code in self.takeoff_site_code_list
        ]
        return {
            'planes': active_planes,
            'waiting_sites': queued,
            'orphaned_queue_entries': orphaned_queue_entries,
            # Keep both values: a mismatch directly exposes a stale cached
            # observation rather than falsely suggesting that no runway is
            # currently usable.
            'cached_departure_runway_plan': dict(
                self._departure_runway_plan
            ),
            'departure_runway_plan': dict(current_departure_plan),
            'runways': runway_state,
            'departure_transporter_by_plane': dict(
                self.departure_transporter_by_plane
            ),
            'relevant_devices': relevant_devices,
        }

    def _sync_waiting_sites(self):
        """Rebuild heuristic queues from actual waiting planes.

        A queue entry is omitted while a compatible mobile device is already
        travelling to the plane's site, which prevents duplicate dispatches.
        """
        rebuilt = {job_code: [] for job_code in self.waiting_sites}
        for plane in self.planes.values():
            if not plane.is_waiting or plane.pending_job not in rebuilt:
                continue
            site_code = plane.site.code
            needed_types = set(self._needed_mobile_types(plane.pending_job))
            has_inflight_device = bool(needed_types) and any(
                device.is_transporting
                and device.site.code == site_code
                and device.resource.type in needed_types
                for device in self.device_list
            )
            if not has_inflight_device and site_code not in rebuilt[plane.pending_job]:
                rebuilt[plane.pending_job].append(site_code)

        for job_code in self.waiting_sites:
            self.waiting_sites[job_code] = rebuilt[job_code]

    def _remove_waiting_site(self, job_code, site_code):
        waiting_sites = self.waiting_sites.get(job_code)
        if waiting_sites is not None and site_code in waiting_sites:
            waiting_sites.remove(site_code)

    def _append_pending_plane_device(self, plane, device):
        if plane is None or device is None:
            return
        record = self.pending_actions.get(plane.code)
        if record is None:
            return
        if device.code not in record['device_ids']:
            record['device_ids'].append(device.code)

    def _settle_waiting_plane_if_ready(self, plane, device=None):
        if plane is None or not plane.is_waiting:
            return False

        site_code = plane.site.code
        pending_job = plane.pending_job

        if plane.destination and plane.destination != plane.site:
            transporter = plane.site.get_avail_transporter()
            if transporter is None:
                return False
            intrinsic_ready_time = float(self.total_time) - float(
                plane.waiting_time
            )
            origin_site_code = plane.site.code
            transport_purpose = plane.pending_transport_purpose
            plane.finish_waiting()
            plane.transporter = transporter
            plane.start_transport(
                plane.destination,
                transporter,
                purpose=transport_purpose,
            )
            plane.destination = None
            self._remove_waiting_site(pending_job, site_code)
            self._append_pending_plane_device(plane, transporter)
            self._record_intrinsic_ready_time(
                plane.code,
                self.TRANSFER_JOB_CODE,
                origin_site_code,
                intrinsic_ready_time,
            )
            return True

        avail_jobs = plane.get_avail_jobs(plane.site)
        chosen_job = None
        if pending_job and pending_job in avail_jobs:
            chosen_job = pending_job
        elif not pending_job and plane.choosed_job and plane.choosed_job in avail_jobs:
            chosen_job = plane.choosed_job

        if chosen_job is not None:
            intrinsic_ready_time = float(self.total_time) - float(
                plane.waiting_time
            )
            plane.finish_waiting()
            plane.choose_job(chosen_job)
            self._record_started_plane_jobs(plane, intrinsic_ready_time)
            if plane.choosed_job in plane.current_jobs:
                plane.choosed_job = None
            self._remove_waiting_site(pending_job, site_code)
            self._append_pending_plane_device(plane, device)
            return True

        return False

    def _settle_ready_waiting_planes(self, site_code=None, device=None):
        """Synchronously unblock waiting planes whose mobile resource is already available."""
        settled = 0
        for plane in list(self.planes.values()):
            if site_code is not None and plane.site.code != site_code:
                continue
            if self._settle_waiting_plane_if_ready(plane, device=device):
                settled += 1
        return settled

    def _validate_plane_action(self, pid, plane, raw_op_idx, raw_site_idx, claimed_site_indices):
        n_ops = len(self.job_code_list)
        op_global_idx = int(raw_op_idx)
        site_idx = int(raw_site_idx)

        if self.agent_op_mask is None or self.ptr_site_mask_matrix is None:
            raise RuntimeError("Action masks must be built before executing a plane action.")
        if not (pid * n_ops <= op_global_idx < (pid + 1) * n_ops):
            raise RuntimeError(
                f"Plane {pid} selected operation {op_global_idx} outside its operation block."
            )
        if not bool(self.agent_op_mask[pid, op_global_idx]):
            raise RuntimeError(
                f"Plane {pid} selected masked operation {op_global_idx} at step {self.steps}."
            )
        if not (0 <= site_idx < len(self.site_code_list)):
            raise RuntimeError(f"Plane {pid} selected out-of-range site {site_idx}.")
        if not bool(self.ptr_site_mask_matrix[pid, site_idx]):
            raise RuntimeError(
                f"Plane {pid} selected masked site {site_idx} at step {self.steps}."
            )

        job_idx = op_global_idx - pid * n_ops
        pair_mask = getattr(self, 'agent_job_site_mask_matrix', None)
        pair_is_valid = (
            bool(pair_mask[pid, job_idx, site_idx])
            if pair_mask is not None
            else bool(self.job_site_mask_matrix[job_idx, site_idx])
        )
        if not pair_is_valid:
            raise RuntimeError(
                f"Plane {pid} selected incompatible or cycle-masked operation-site pair "
                f"({op_global_idx}, {site_idx}) at step {self.steps}."
            )
        job_code = self.job_code_list[job_idx]
        target_site_code = self.site_code_list[site_idx]
        if job_code == self.TRANSFER_JOB_CODE:
            if (
                not plane.has_completed_service_jobs()
                or plane.has_started_departure()
                or plane.departure_staging_decided
                or target_site_code in self.runway_code_list
            ):
                raise RuntimeError(
                    f"Invalid post-service relocation for {plane.code}: "
                    f"target={target_site_code}."
                )
        if job_code in self.departure_job_code_list:
            if target_site_code not in self.takeoff_site_code_list:
                raise RuntimeError(
                    f"Departure job {job_code} must target a takeoff site, "
                    f"got {target_site_code}."
                )
            if job_code == self.departure_job_code_list[0]:
                assigned_runway = getattr(
                    self, '_departure_runway_plan', {}
                ).get(plane.code)
                if (
                    not self._plane_awaiting_departure(plane)
                    or not plane.departure_staging_decided
                    or self._departure_transporter_for_plane(
                        plane, ready_only=True
                    ) is None
                    or target_site_code != assigned_runway
                ):
                    raise RuntimeError(
                        f"Invalid progressive departure for {plane.code}: "
                        f"assigned_runway={assigned_runway}, "
                        f"target={target_site_code}."
                    )
            if (
                job_code == self.departure_job_code_list[-1]
                and target_site_code != plane.site.code
            ):
                raise RuntimeError(
                    f"Final departure job must remain at {plane.site.code}, "
                    f"got {target_site_code}."
                )
        if site_idx in claimed_site_indices:
            raise RuntimeError(
                f"Plane {pid} selected site {site_idx}, which was already claimed in this step."
            )
        claimed_site_indices.add(site_idx)
        return op_global_idx, site_idx
        
    def seed(self, seed=None):
        '''设置随机种子'''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def add_planes(self, new_planes_cfg):
        '''添加新飞机到环境'''
        for plane_cfg in new_planes_cfg:
            plane_id = f'Plane_{plane_cfg["batch"]}_{plane_cfg["idx"]}'
            if plane_id in self.planes:
                # print(f"Warning: Plane {plane_id} already exists!")
                continue
            plane_cfg = dict(plane_cfg)
            plane_cfg['takeoff_site_codes'] = tuple(
                self.takeoff_site_code_list
            )
            plane = Plane(plane_id, plane_cfg)
            self.planes[plane_id] = plane
            self.sites[plane.site.code].add_plane(plane)
        self.num_planes += len(new_planes_cfg)

    def remove_planes(self, plane_ids):
        '''从环境移除飞机'''
        for plane_id in plane_ids:
            if plane_id in self.planes:
                plane = self.planes[plane_id]
                ready_time = self.departure_ready_since.get(plane_id)
                self._release_departure_transporter(plane_code=plane_id)
                pid = int(plane_id.split('_')[-1])
                if self.sites[plane.site.code].plane == plane:
                    self.sites[plane.site.code].remove_plane()
                self.departed_agent_ids.add(pid)
                self._departed_this_step[pid] = plane
                self.departed_total_relocations += int(
                    plane.total_relocations
                )
                self.departure_log.append({
                    'plane_id': plane_id,
                    'agent_id': pid,
                    'time': float(self.total_time),
                    'site_code': plane.site.code,
                    'completed_jobs': list(plane.finished_jobs),
                    'departure_ready_time': ready_time,
                    'departure_flow_time': (
                        float(self.total_time) - float(ready_time)
                        if ready_time is not None else None
                    ),
                })
                del self.planes[plane_id]
                self.departure_ready_since.pop(plane_id, None)
                self.num_planes -= 1

    def _remove_departed_planes(self):
        departed = [
            plane_id for plane_id, plane in self.planes.items()
            if (
                plane.is_idle()
                and plane.has_completed_departure()
                and plane.site.code in self.takeoff_site_code_list
            )
        ]
        if departed:
            self.remove_planes(departed)
        return departed
        
    def get_idle_devices(self, res_types):
        '''获取指定资源类型的空闲设备
        
        输入:
            res_types: list - 资源类型列表
        返回:
            list - 空闲设备对象列表
        示例:
            idle_devices = env.get_idle_devices(['R001', 'R002'])
        '''
        ret = []
        for res_type in sorted(set(res_types)):
            ret += [
                device for device in self.mobile_devices[res_type]
                if self._device_is_dispatchable(device)
            ]
        return sorted(ret, key=lambda device: device.code)

    def get_avail_sites(self, plane=None):
        '''获取飞机可用的站点列表
        
        输入:
            plane: Plane对象或None - 指定飞机，默认为None
        返回:
            list - 可用站点代码列表
        逻辑:
            排除被占用、被干涉的站点，排除降落区 Z 和配置中的起飞跑道
            如果指定了飞机且不在特殊站点，包含当前站点
        示例:
            sites = env.get_avail_sites(plane_instance)
        '''
        ret = [site.code for site in self.sites.values() if not site.is_occupied and not site.is_interfered and site.code not in self.runway_code_list]
        if plane:
            if plane.site.code not in self.runway_code_list:
                ret.append(plane.site.code)  # 包括当前所在位置
            # if not plane.is_idle():
            #     # 如果飞机正在忙碌，排除当前站位
            #     ret = [site for site in ret if site != plane.site.code]
        return ret
    
    def get_avail_takeoff_sites(self):
        '''获取可用起飞跑道
        
        返回:
            list - 空闲起飞跑道代码列表
        示例:
            runways = env.get_avail_takeoff_sites()
        '''
        return [
            site.code for site in self.sites.values()
            if (
                not site.is_occupied
                and not site.is_interfered
                and site.code in self.takeoff_site_code_list
            )
        ]

    def _departure_barrier_is_open(self):
        """Compatibility signal: at least one aircraft is departure-ready.

        This flag is deliberately no longer a scheduling barrier.  Departure
        eligibility is evaluated per aircraft so a completed aircraft may
        leave while other aircraft are still being serviced or have not yet
        arrived.
        """
        self._sync_departure_pipeline_state()
        return bool(self.departure_barrier_open)

    def _plane_departure_eligible(self, plane):
        return bool(
            plane is not None
            and plane.has_completed_service_jobs()
            and (
                plane.has_started_departure()
                or plane.site.code not in self.takeoff_site_code_list
            )
        )

    def _plane_awaiting_departure(self, plane):
        return bool(
            self._plane_departure_eligible(plane)
            and not plane.has_started_departure()
            and plane.site.code not in self.takeoff_site_code_list
        )

    def _device_by_code(self, device_code):
        return next(
            (device for device in self.device_list if device.code == device_code),
            None,
        )

    def _release_departure_transporter(self, plane_code=None, device_code=None):
        if plane_code is None and device_code is not None:
            plane_code = self.departure_plane_by_transporter.get(device_code)
        if device_code is None and plane_code is not None:
            device_code = self.departure_transporter_by_plane.get(plane_code)
        if plane_code is not None:
            self.departure_transporter_by_plane.pop(plane_code, None)
        if device_code is not None:
            self.departure_plane_by_transporter.pop(device_code, None)
            device = self._device_by_code(device_code)
            if (
                device is not None
                and getattr(device, 'reserved_for_plane', None) == plane_code
            ):
                device.reserved_for_plane = None

    def _reserve_departure_transporter(self, plane, device):
        if (
            not self._plane_awaiting_departure(plane)
            or not plane.departure_staging_decided
            or not self._is_transporter_device(device)
        ):
            return False
        existing_device = self.departure_transporter_by_plane.get(plane.code)
        existing_plane = self.departure_plane_by_transporter.get(device.code)
        if existing_device == device.code and existing_plane == plane.code:
            return True
        if existing_device is not None or existing_plane is not None:
            return False
        if getattr(device, 'reserved_for_plane', None) is not None:
            return False
        self.departure_transporter_by_plane[plane.code] = device.code
        self.departure_plane_by_transporter[device.code] = plane.code
        device.reserved_for_plane = plane.code
        return True

    def _departure_transporter_for_plane(self, plane, ready_only=False):
        device_code = self.departure_transporter_by_plane.get(plane.code)
        if device_code is None:
            return None
        device = self._device_by_code(device_code)
        if (
            device is None
            or self.departure_plane_by_transporter.get(device_code) != plane.code
            or getattr(device, 'reserved_for_plane', None) != plane.code
        ):
            self._release_departure_transporter(
                plane_code=plane.code, device_code=device_code
            )
            return None
        if ready_only and not (
            device.is_idle()
            and device.resource.is_available()
            and device.site == plane.site
        ):
            return None
        return device

    def _sync_departure_pipeline_state(self):
        active_plane_codes = set(self.planes)
        for plane in self.planes.values():
            if self._plane_departure_eligible(plane):
                ready_time = self.departure_ready_since.setdefault(
                    plane.code, float(self.total_time)
                )
                self._record_intrinsic_ready_time(
                    plane.code,
                    self.TRANSFER_JOB_CODE,
                    plane.site.code,
                    ready_time,
                )
                self.departure_barrier_open = True
            if not self._plane_awaiting_departure(plane):
                self._release_departure_transporter(plane_code=plane.code)

        for plane_code in list(self.departure_transporter_by_plane):
            plane = self.planes.get(plane_code)
            if (
                plane_code not in active_plane_codes
                or not self._plane_awaiting_departure(plane)
                or not plane.departure_staging_decided
            ):
                self._release_departure_transporter(plane_code=plane_code)

    def _departure_pickup_requests(self):
        """Return blocking R014 pre-positioning requests for staged aircraft."""
        self._sync_departure_pipeline_state()
        requests = []
        for plane in sorted(
            self.planes.values(),
            key=lambda item: int(item.code.split('_')[-1]),
        ):
            if not (
                self._plane_awaiting_departure(plane)
                and plane.departure_staging_decided
                and plane.is_idle()
                and self._departure_transporter_for_plane(plane) is None
            ):
                continue
            ready_since = self.departure_ready_since.get(
                plane.code, float(self.total_time)
            )
            requests.append({
                'job_code': self.TRANSFER_JOB_CODE,
                'site_code': plane.site.code,
                'plane_id': plane.code,
                'plane_idx': int(plane.code.split('_')[-1]),
                'needed_res_types': [self.TRANSPORTER_RESOURCE_TYPE],
                'waiting_time': max(
                    0.0, float(self.total_time) - float(ready_since)
                ),
                'lead_time': 0.0,
                'is_noop': False,
                'is_lookahead': False,
                'urgent': True,
                'request_kind': 'departure_pickup',
            })
        return requests

    def _compute_departure_runway_plan(self):
        """Match ready aircraft to free runways without an ID-based wave."""
        self._sync_departure_pipeline_state()
        runways = [
            self.sites[code] for code in self.get_avail_takeoff_sites()
        ]
        first_departure = self.departure_job_code_list[0]
        planes = [
            plane for plane in self.planes.values()
            if (
                self._plane_awaiting_departure(plane)
                and plane.departure_staging_decided
                and plane.is_idle()
                and first_departure in plane.get_avail_jobs(site=None)
                and self._departure_transporter_for_plane(
                    plane, ready_only=True
                ) is not None
            )
        ]
        if not planes or not runways:
            return {}
        planes.sort(key=lambda plane: int(plane.code.split('_')[-1]))
        runways.sort(key=lambda site: site.code)
        costs = np.zeros((len(planes), len(runways)), dtype=np.float64)
        for row, plane in enumerate(planes):
            wait_age = max(
                0.0,
                float(self.total_time)
                - float(self.departure_ready_since.get(
                    plane.code, self.total_time
                )),
            )
            for column, runway in enumerate(runways):
                distance = (
                    abs(float(plane.site.pos[0]) - float(runway.pos[0]))
                    + abs(float(plane.site.pos[1]) - float(runway.pos[1]))
                )
                tow_time = distance / max(float(plane.velocity), 1.0)
                costs[row, column] = (
                    tow_time
                    - wait_age
                    + 1e-6 * int(plane.code.split('_')[-1])
                    + 1e-9 * column
                )
        row_indices, column_indices = linear_sum_assignment(costs)
        return {
            planes[int(row)].code: runways[int(column)].code
            for row, column in zip(row_indices, column_indices)
        }

    def _departure_candidates(self):
        return set(self._compute_departure_runway_plan())

    def _ready_job_codes(self, plane, departure_plan=None):
        """Return environment-authoritative ready operations for one plane."""
        ready = [
            code for code in plane.get_avail_jobs(site=None)
            if code in self.job_code_list
        ]
        departure_codes = set(self.departure_job_code_list)
        ready = [code for code in ready if code != self.TRANSFER_JOB_CODE]
        if not plane.has_completed_service_jobs():
            return [code for code in ready if code not in departure_codes]
        if not plane.has_started_departure():
            if not plane.departure_staging_decided:
                return [self.TRANSFER_JOB_CODE]
            if departure_plan is None:
                departure_plan = self._compute_departure_runway_plan()
            first_departure = self.departure_job_code_list[0]
            return [first_departure] if plane.code in departure_plan else []

        first_departure = self.departure_job_code_list[0]
        return [code for code in ready if code != first_departure]

    def _plane_can_decide(self, plane, departure_plan=None):
        return plane.is_idle() and bool(
            self._ready_job_codes(plane, departure_plan=departure_plan)
        )

    def _is_schedule_complete(self):
        if self.landing_list:
            return False
        return not self.planes

    def _build_plane_jobs(self, drop_optional=False):
        optional_jobs = {'ZY05', 'ZY06', 'ZY09'}
        selected_jobs = []
        selected_codes = set()

        for job in self.jobs.values():
            if (
                job.code not in self.job_code_list
                or job.code == self.TRANSFER_JOB_CODE
            ):
                continue
            if drop_optional and job.code in optional_jobs:
                if (self.np_random.random() if hasattr(self, 'np_random') else np.random.random()) < 0.2:
                    continue
            job_copy = copy.deepcopy(job)
            selected_jobs.append(job_copy)
            selected_codes.add(job_copy.code)

        for job in selected_jobs:
            predecessors = set(job.predecessor)
            job.predecessor = sorted(
                pred for pred in predecessors
                if pred not in optional_jobs or pred in selected_codes
            )

        # The source data models ZY-L as an implicit release operation. Make
        # the first explicit departure operation depend on every selected
        # service operation; the environment adds the ZY-L minute to R014
        # transport instead of exposing a self-referential ZY-L action.
        service_codes = {
            job.code for job in selected_jobs if job.group == '保障'
        }
        if self.departure_job_code_list:
            first_departure = self.departure_job_code_list[0]
            for job in selected_jobs:
                if job.code == first_departure:
                    job.predecessor = sorted(service_codes)
                    break

        return selected_jobs

    def _needed_mobile_types(self, job_code):
        if job_code == 'ZY-T':
            return ['R014']
        job = self.jobs.get(job_code)
        if job is None:
            return []
        return sorted(res for res in job.resources if res in self.mobile_devices)

    @staticmethod
    def _request_identity(request):
        return (
            request.get('plane_id'),
            request.get('job_code'),
            request.get('site_code'),
        )

    def _record_intrinsic_ready_time(
        self,
        plane_id,
        job_code,
        site_code,
        ready_time,
    ):
        """Record one factual ready occurrence without affecting dynamics."""

        if plane_id is None or job_code is None or site_code is None:
            return
        ready_time = float(ready_time)
        if not math.isfinite(ready_time) or ready_time < -1e-9:
            raise RuntimeError(
                'Intrinsic ready-time occurrence must be finite and '
                f'non-negative, got {ready_time!r}.'
            )
        key = (str(plane_id), str(job_code), str(site_code))
        values = self.intrinsic_ready_time_occurrences.setdefault(key, [])
        if not any(
            math.isclose(value, ready_time, rel_tol=0.0, abs_tol=1e-6)
            for value in values
        ):
            values.append(max(0.0, ready_time))
            values.sort()

    def _record_started_plane_jobs(self, plane, ready_time):
        """Record all jobs started together, including implicit parallels."""

        for job_code in tuple(getattr(plane, 'current_jobs', ())):
            self._record_intrinsic_ready_time(
                plane.code,
                job_code,
                plane.site.code,
                ready_time,
            )

    def attach_intrinsic_ready_time_labels(self, decision_trace):
        """Backfill exact factual ready labels into a completed replay trace.

        Lookahead observations are matched to the first realized occurrence
        that is not earlier than their dependency lower bound. Blocking
        requests carry their own exact wait age, so their ready timestamp is
        recovered even when the corresponding operation has not started yet.
        Speculative branches that the teacher never realizes remain
        intentionally unlabeled; Stage2 must not train a fabricated target.
        """

        trace = copy.deepcopy(list(decision_trace or []))
        occurrences = {
            tuple(key): sorted(float(value) for value in values)
            for key, values in self.intrinsic_ready_time_occurrences.items()
        }
        labels = []
        by_kind = {}
        for event in trace:
            observation_time = float(event.get('time_before', 0.0))
            for request in event.get('requests', []) or []:
                if request.get('is_noop', False):
                    continue
                kind = str(request.get('request_kind', 'unknown'))
                kind_stats = by_kind.setdefault(kind, {
                    'observed': 0,
                    'labeled': 0,
                })
                kind_stats['observed'] += 1
                lead_time = max(0.0, float(request.get('lead_time', 0.0)))
                identity = (
                    str(request.get('plane_id')),
                    str(request.get('job_code')),
                    str(request.get('site_code')),
                )
                intrinsic_ready_time = None
                if not bool(request.get('is_lookahead', False)):
                    intrinsic_ready_time = observation_time - max(
                        0.0, float(request.get('waiting_time', 0.0))
                    )
                else:
                    lower_bound = observation_time + lead_time
                    intrinsic_ready_time = next(
                        (
                            value
                            for value in occurrences.get(identity, ())
                            if value >= lower_bound - 1e-6
                        ),
                        None,
                    )
                if intrinsic_ready_time is None:
                    continue
                intrinsic_ready_time = max(0.0, float(intrinsic_ready_time))
                ready_lead = max(
                    0.0, intrinsic_ready_time - observation_time
                )
                if ready_lead < lead_time - 1e-6:
                    raise RuntimeError(
                        'Exact intrinsic ready-time label violates its DAG '
                        f'lower bound: identity={identity!r}, observation='
                        f'{observation_time}, ready={intrinsic_ready_time}, '
                        f'lead={lead_time}.'
                    )
                request['intrinsic_ready_time'] = intrinsic_ready_time
                # Keep the flat index compact: the complete candidate payload
                # already lives in decision_trace.  These immutable fields are
                # sufficient to bind one target to one visited state.
                record = {
                    'observation_time': observation_time,
                    'intrinsic_ready_time': intrinsic_ready_time,
                    'ready_lead_seconds': ready_lead,
                    'dag_lower_bound_seconds': lead_time,
                    'label_schema_version': (
                        self.INTRINSIC_READY_TIME_LABEL_SCHEMA_VERSION
                    ),
                    'target_semantics': self.INTRINSIC_READY_TIME_SEMANTICS,
                    'request_id': int(request.get('id', 0)),
                    'request_kind': kind,
                    'plane_id': request.get('plane_id'),
                    'job_code': request.get('job_code'),
                    'site_code': request.get('site_code'),
                }
                labels.append(record)
                kind_stats['labeled'] += 1

        observed = sum(item['observed'] for item in by_kind.values())
        labeled = sum(item['labeled'] for item in by_kind.values())
        return trace, labels, {
            'observed_request_count': int(observed),
            'labeled_request_count': int(labeled),
            'coverage': float(labeled / observed) if observed else 1.0,
            'by_request_kind': by_kind,
            'factual_only': True,
        }

    def _lookahead_lead_time(self, plane, target_site):
        """Return a conservative lower bound until the selected job may start."""
        if plane.is_transporting and plane.site == target_site:
            return float(max(0.0, plane.left_trans_time))
        if plane.is_busy and plane.site == target_site:
            return float(max(0.0, plane.site.left_job_time))
        if (
            plane.is_waiting
            and plane.destination is not None
            and plane.destination == target_site
        ):
            distance = (
                abs(float(plane.site.pos[0]) - float(target_site.pos[0]))
                + abs(float(plane.site.pos[1]) - float(target_site.pos[1]))
            )
            preparation = (
                60.0
                if target_site.code in self.takeoff_site_code_list
                else 120.0
            )
            return float(
                distance / max(float(plane.velocity), 1.0) + preparation
            )
        return 0.0

    @staticmethod
    def _device_travel_seconds(device, target_site):
        distance = (
            abs(float(device.site.pos[0]) - float(target_site.pos[0]))
            + abs(float(device.site.pos[1]) - float(target_site.pos[1]))
        )
        return float(distance / max(float(device.velocity), 1.0))

    def _device_service_release_seconds(self, device):
        """Return the next exact capacity release for a carried resource.

        ``Device.is_idle`` only describes transport/recovery state.  A parked
        device can still have its resource serving an operation, and the old
        ETA treated that device as immediately available.  Site job records
        contain the resource code actually selected by ``Site.start_jobs``,
        which lets us recover the next release without adding mutable clocks.
        """
        resource = device.resource
        if resource.is_available():
            return 0.0
        release_times = []
        for site_code in tuple(getattr(resource, 'on_service', ())):
            site = self.sites.get(site_code)
            if site is None:
                continue
            matching = [
                max(0.0, float(job_state[0]))
                for job_state in getattr(site, 'onging_jobs', {}).values()
                if len(job_state) > 1 and job_state[1] == resource.code
            ]
            if matching:
                # One capacity slot becomes available at the first matching
                # completion at this site.
                release_times.append(min(matching))
            else:
                release_times.append(max(
                    0.0, float(getattr(site, 'left_job_time', 0.0))
                ))
        if release_times:
            return float(min(release_times))
        # Inconsistent external state must look unavailable, never free.
        return 36000.0

    def _lookahead_reservation_blocks(self, device, request=None):
        reservation = getattr(device, 'lookahead_reservation', None)
        if not reservation:
            return False
        if request is not None:
            identity = tuple(reservation.get('identity', ()))
            if identity and identity == self._request_identity(request):
                return False
            if (
                reservation.get('mode') == 'soft'
                and not bool(request.get('is_lookahead', False))
            ):
                return False
        return True

    def _device_release_seconds(self, device, request=None):
        if (
            self.resource_release_aware_eta
            and self._lookahead_reservation_blocks(device, request=request)
            and getattr(device, 'lookahead_reservation', {}).get('mode') == 'hard'
        ):
            expires_at = float(
                device.lookahead_reservation.get('expires_at', self.total_time)
            )
            reservation_delay = max(0.0, expires_at - float(self.total_time))
        else:
            reservation_delay = 0.0
        service_delay = (
            self._device_service_release_seconds(device)
            if self.resource_release_aware_eta else 0.0
        )
        return float(max(
            0.0,
            float(getattr(device, 'left_trans_time', 0.0)),
            float(getattr(device, 'left_rec_time', 0.0)),
            service_delay,
            reservation_delay,
        ))

    def _device_eta_seconds(self, device, target_site, request=None):
        return float(
            self._device_release_seconds(device, request=request)
            + self._device_travel_seconds(device, target_site)
        )

    def _device_observation_available(self, device):
        if not self.resource_release_aware_eta:
            return bool(device.is_idle())
        return bool(
            device.is_idle()
            and device.resource.is_available()
            and getattr(device, 'reserved_for_plane', None) is None
            and getattr(device, 'lookahead_reservation', None) is None
        )

    def _resource_already_committed(self, target_site, needed_res_types):
        """Return whether a compatible resource is present or already inbound."""
        target_site.update_resources()
        if any(
            resource_type in target_site.res_avail
            for resource_type in needed_res_types
        ):
            return True
        return any(
            device.is_transporting
            and device.site == target_site
            and device.resource.type in needed_res_types
            for device in self.device_list
        )

    def _future_request(
        self,
        plane,
        job_code,
        target_site,
        lead_time,
        kind,
        *,
        dependency_depth=1,
    ):
        needed_res_types = self._needed_mobile_types(job_code)
        if not needed_res_types or self._resource_already_committed(
            target_site, needed_res_types
        ):
            return None
        return {
            'job_code': job_code,
            'site_code': target_site.code,
            'plane_id': plane.code,
            'plane_idx': int(plane.code.split('_')[-1]),
            'needed_res_types': needed_res_types,
            'waiting_time': 0.0,
            'lead_time': max(0.0, float(lead_time)),
            'dependency_depth': max(0, int(dependency_depth)),
            'predicted_need_time': float(self.total_time) + max(
                0.0, float(lead_time)
            ),
            'is_noop': False,
            'is_lookahead': True,
            'urgent': False,
            'request_kind': kind,
        }

    def _next_mobile_job_request(self, plane):
        """Expose one direct mobile-resource successor of committed work.

        The horizon is deliberately one dependency edge.  It provides enough
        notice for a vehicle to travel while avoiding speculative rollouts of
        the frozen aircraft policy or a guessed future stand.
        """
        if getattr(self, 'device_future_intent_horizon', 0) <= 0:
            return None

        target_site = plane.site
        future_finished = set(plane.finished_jobs)
        completing = set()
        lead_time = 0.0
        if plane.is_busy:
            completing.update(plane.current_jobs)
            lead_time = float(max(0.0, plane.site.left_job_time))
        else:
            record = self.pending_actions.get(plane.code)
            if record is None:
                return None
            selected_job = record.get('target_job_code')
            selected_site = record.get('target_site_code') or record.get('site_id')
            if selected_job not in plane.left_jobs or selected_site not in self.sites:
                return None
            completing.add(selected_job)
            target_site = self.sites[selected_site]
            lead_time = self._lookahead_lead_time(plane, target_site)
            lead_time += float(plane.jobs[selected_job].time or 0.0)

        if not completing:
            return None
        candidates = []
        completed_after_commitment = future_finished.union(completing)
        departure_codes = set(self.departure_job_code_list)
        for job_code in plane.left_jobs:
            if job_code in completing or job_code in departure_codes:
                continue
            needed = self._needed_mobile_types(job_code)
            if not needed:
                continue
            predecessors = set(plane.jobs[job_code].predecessor)
            if not predecessors.issubset(completed_after_commitment):
                continue
            # At least one currently unfinished predecessor must be part of
            # the committed work; otherwise this is an unrelated ready branch.
            pending_predecessors = predecessors.difference(future_finished)
            if (
                not pending_predecessors
                or not pending_predecessors.issubset(completing)
            ):
                continue
            candidates.append(job_code)
        if not candidates:
            return None
        # Prefer the longest operation at the same deadline: its delay is most
        # likely to propagate into the final synchronization tail.
        job_code = min(
            candidates,
            key=lambda code: (
                -float(plane.jobs[code].time or 0.0),
                str(code),
            ),
        )
        return self._future_request(
            plane,
            job_code,
            target_site,
            lead_time,
            'next_mobile_job',
        )

    def _bounded_mobile_frontier_requests(self, plane):
        """Expose a deterministic, bounded direct-successor frontier.

        This never guesses a future stand or rolls out the frozen aircraft
        policy.  It only expands operations made ready by work that is already
        executing or committed at a known site, then keeps the most scarce and
        longest mobile-resource successors.  The per-aircraft bound preserves
        the existing 3*N+1 request capacity.
        """
        if getattr(self, 'device_future_intent_horizon', 0) <= 0:
            return []
        if self.device_future_intent_horizon > 1:
            return self._multi_hop_mobile_frontier_requests(plane)
        target_site = plane.site
        future_finished = set(plane.finished_jobs)
        completing = set()
        lead_time = 0.0
        if plane.is_busy:
            completing.update(plane.current_jobs)
            lead_time = float(max(0.0, plane.site.left_job_time))
        else:
            record = self.pending_actions.get(plane.code)
            if record is None:
                return []
            selected_job = record.get('target_job_code')
            selected_site = (
                record.get('target_site_code') or record.get('site_id')
            )
            if (
                selected_job not in plane.left_jobs
                or selected_site not in self.sites
            ):
                return []
            completing.add(selected_job)
            target_site = self.sites[selected_site]
            lead_time = self._lookahead_lead_time(plane, target_site)
            lead_time += float(plane.jobs[selected_job].time or 0.0)
        if not completing:
            return []

        completed_after_commitment = future_finished.union(completing)
        departure_codes = set(self.departure_job_code_list)
        candidates = []
        for job_code in plane.left_jobs:
            if job_code in completing or job_code in departure_codes:
                continue
            needed = self._needed_mobile_types(job_code)
            if not needed:
                continue
            predecessors = set(plane.jobs[job_code].predecessor)
            pending_predecessors = predecessors.difference(future_finished)
            if (
                not predecessors.issubset(completed_after_commitment)
                or not pending_predecessors
                or not pending_predecessors.issubset(completing)
            ):
                continue
            capacity = sum(
                len(self.mobile_devices.get(resource_type, ()))
                for resource_type in needed
            )
            candidates.append((
                max(1, capacity),
                -float(plane.jobs[job_code].time or 0.0),
                str(job_code),
            ))

        requests = []
        for _, _, job_code in sorted(candidates):
            request = self._future_request(
                plane,
                job_code,
                target_site,
                lead_time,
                'bounded_mobile_frontier',
            )
            if request is not None:
                requests.append(request)
            if len(requests) >= self.device_frontier_max_requests:
                break
        return requests

    def _multi_hop_mobile_frontier_requests(self, plane):
        """Expose a bounded dependency frontier up to the configured horizon.

        The aircraft has committed only its currently executing/pending action.
        Deeper intents therefore keep that known stand and are deliberately
        soft: a later aircraft decision may cancel them.  We never roll out the
        aircraft actor.  Already-ready mobile branches are admitted as depth-1
        forecasts at the committed stand; deeper jobs are expanded only when
        every unfinished predecessor is already in this bounded forecast.
        """
        horizon = int(self.device_future_intent_horizon)
        if horizon <= 1:
            raise ValueError('multi-hop frontier requires horizon > 1')

        target_site = plane.site
        future_finished = set(plane.finished_jobs)
        completing = set()
        root_completion = 0.0
        if plane.is_busy:
            completing.update(plane.current_jobs)
            root_completion = float(max(0.0, plane.site.left_job_time))
        else:
            record = self.pending_actions.get(plane.code)
            if record is None:
                return []
            selected_job = record.get('target_job_code')
            selected_site = (
                record.get('target_site_code') or record.get('site_id')
            )
            if (
                selected_job not in plane.left_jobs
                or selected_site not in self.sites
            ):
                return []
            completing.add(selected_job)
            target_site = self.sites[selected_site]
            root_completion = self._lookahead_lead_time(
                plane, target_site
            ) + self._job_duration(
                plane.jobs[selected_job],
                float(plane.config.get('fuel', 30.0)),
            )
        if not completing:
            return []

        fuel = float(plane.config.get('fuel', 30.0))
        completion_time = {code: 0.0 for code in future_finished}
        dependency_depth = {code: 0 for code in future_finished}
        depends_on_commitment = {code: False for code in future_finished}
        for code in completing:
            completion_time[code] = float(root_completion)
            dependency_depth[code] = 0
            depends_on_commitment[code] = True

        departure_codes = set(self.departure_job_code_list)
        unresolved = {
            code for code in plane.left_jobs
            if code not in completing and code not in departure_codes
        }
        candidates = []
        while unresolved:
            progressed = False
            for job_code in sorted(tuple(unresolved)):
                job = plane.jobs[job_code]
                predecessors = set(job.predecessor)
                unfinished_predecessors = predecessors.difference(
                    future_finished
                )
                if not unfinished_predecessors:
                    # The current action occupies the aircraft/stand until the
                    # root finishes, so an otherwise-ready branch cannot be
                    # selected earlier than this conservative boundary.
                    depth = 1
                    ready_time = float(root_completion)
                else:
                    if not unfinished_predecessors.issubset(completion_time):
                        continue
                    if not all(
                        depends_on_commitment.get(code, False)
                        for code in unfinished_predecessors
                    ):
                        unresolved.remove(job_code)
                        progressed = True
                        continue
                    depth = 1 + max(
                        dependency_depth[code]
                        for code in unfinished_predecessors
                    )
                    ready_time = max(
                        (
                            completion_time.get(code, 0.0)
                            for code in predecessors
                        ),
                        default=0.0,
                    )
                if depth > horizon:
                    unresolved.remove(job_code)
                    progressed = True
                    continue
                completion_time[job_code] = ready_time + self._job_duration(
                    job, fuel
                )
                dependency_depth[job_code] = depth
                depends_on_commitment[job_code] = True
                unresolved.remove(job_code)
                progressed = True

                needed = self._needed_mobile_types(job_code)
                if not needed:
                    continue
                if not self._site_can_eventually_support_job(
                    target_site, job
                ):
                    continue
                capacity = sum(
                    len(self.mobile_devices.get(resource_type, ()))
                    for resource_type in needed
                )
                candidates.append((
                    depth,
                    max(1, capacity),
                    float(ready_time),
                    -self._job_duration(job, fuel),
                    str(job_code),
                ))
            if not progressed:
                # A malformed/cyclic dependency graph must not stall request
                # construction; the environment's normal action masks retain
                # responsibility for rejecting an impossible schedule.
                break

        requests = []
        for depth, _, ready_time, _, job_code in sorted(candidates):
            request = self._future_request(
                plane,
                job_code,
                target_site,
                ready_time,
                f'bounded_mobile_frontier_h{depth}',
                dependency_depth=depth,
            )
            if request is not None:
                requests.append(request)
            if len(requests) >= self.device_frontier_max_requests:
                break
        return requests

    def _departure_lookahead_request(self, plane):
        """Expose R014 while the final service commitment is still running."""
        if (
            not getattr(self, 'device_departure_lookahead', False)
            or plane.has_completed_service_jobs()
        ):
            return None
        unfinished_service = set(plane.left_jobs).union(plane.current_jobs)
        unfinished_service.intersection_update(plane.service_job_codes)
        target_site = plane.site
        lead_time = None
        if plane.is_busy and unfinished_service.issubset(set(plane.current_jobs)):
            lead_time = float(max(0.0, plane.site.left_job_time))
        else:
            record = self.pending_actions.get(plane.code)
            selected_job = record.get('target_job_code') if record else None
            selected_site = (
                record.get('target_site_code') or record.get('site_id')
                if record else None
            )
            if (
                selected_job in unfinished_service
                and unfinished_service == {selected_job}
                and selected_site in self.sites
            ):
                target_site = self.sites[selected_site]
                lead_time = self._lookahead_lead_time(plane, target_site)
                lead_time += float(plane.jobs[selected_job].time or 0.0)
        if lead_time is None:
            return None
        return self._future_request(
            plane,
            self.TRANSFER_JOB_CODE,
            target_site,
            lead_time,
            'departure_future_pickup',
        )

    def _iter_lookahead_requests(self, seen, assigned):
        """Yield requests for resources needed by an already selected future job.

        A pending plane action is the commitment boundary: the environment does
        not guess an unselected future site or operation.  The request may be
        served while the plane is travelling, waiting for R014, or executing a
        prerequisite at the committed target site.
        """
        if not self.device_lookahead_dispatch:
            return
        for plane_id, record in sorted(self.pending_actions.items()):
            plane = self.planes.get(plane_id)
            if plane is None:
                continue
            job_code = record.get('target_job_code')
            site_code = record.get('target_site_code') or record.get('site_id')
            if (
                job_code not in self.jobs
                or site_code not in self.sites
                or job_code not in plane.left_jobs
            ):
                continue
            needed_res_types = self._needed_mobile_types(job_code)
            if not needed_res_types:
                continue
            target_site = self.sites[site_code]
            target_site.update_resources()
            if any(
                resource_type in target_site.res_avail
                for resource_type in needed_res_types
            ):
                continue
            key = (plane.code, job_code, site_code)
            if (
                key in seen
                or key in assigned
                or key in self._deferred_lookahead_requests
            ):
                continue
            has_inflight_resource = any(
                device.is_transporting
                and device.site == target_site
                and device.resource.type in needed_res_types
                for device in self.device_list
            )
            if has_inflight_resource:
                continue
            yield {
                'job_code': job_code,
                'site_code': site_code,
                'plane_id': plane.code,
                'plane_idx': int(plane.code.split('_')[-1]),
                'needed_res_types': needed_res_types,
                'waiting_time': 0.0,
                'lead_time': self._lookahead_lead_time(plane, target_site),
                'is_noop': False,
                'is_lookahead': True,
                'urgent': False,
                'request_kind': 'selected_future_job',
            }

        for plane in sorted(
            self.planes.values(),
            key=lambda item: int(item.code.split('_')[-1]),
        ):
            if self.device_future_intent_mode == 'bounded_frontier':
                future_requests = self._bounded_mobile_frontier_requests(plane)
            else:
                future_requests = [self._next_mobile_job_request(plane)]
            future_requests.append(self._departure_lookahead_request(plane))
            for request in future_requests:
                if request is None:
                    continue
                key = self._request_identity(request)
                if (
                    key in seen
                    or key in assigned
                    or key in self._deferred_lookahead_requests
                ):
                    continue
                yield request

    def _sync_resource_intent_ledger(self, requests):
        """Track the soft -> firm -> dispatched -> arrived lifecycle."""
        active = set()
        for request in requests:
            if request.get('is_noop', False):
                continue
            key = self._request_identity(request)
            active.add(key)
            desired_status = (
                'soft' if request.get('is_lookahead', False) else 'firm'
            )
            entry = self.resource_intent_ledger.setdefault(key, {
                'identity': list(key),
                'created_time': float(self.total_time),
                'first_visible_time': float(self.total_time),
                'status': desired_status,
                'history': [],
            })
            previous = entry.get('status')
            if previous == 'cancelled':
                entry['status'] = desired_status
                entry['history'].append({
                    'status': desired_status,
                    'time': float(self.total_time),
                    'reason': 'intent_reactivated',
                })
            if previous == 'soft' and desired_status == 'firm':
                entry['status'] = 'firm'
                entry['history'].append({
                    'status': 'firm', 'time': float(self.total_time)
                })
            entry.update({
                'request_kind': request.get('request_kind'),
                'needed_time': float(self.total_time) + float(
                    request.get('lead_time', 0.0)
                ),
                'last_seen_time': float(self.total_time),
            })
            compatible_devices = [
                device
                for device in self.device_list[:self.max_device_num]
                if self._device_can_serve(device, request)
            ]
            legal_devices = [
                device for device in compatible_devices
                if self._lookahead_dispatch_delay(device, request) <= 1e-9
            ]
            idle_legal_devices = [
                device for device in legal_devices
                if self._device_is_dispatchable(device)
            ]
            if legal_devices:
                entry.setdefault(
                    'first_legal_time', float(self.total_time)
                )
            if idle_legal_devices:
                entry.setdefault(
                    'first_compatible_idle_time', float(self.total_time)
                )
            entry['compatible_device_count_last_seen'] = int(
                len(compatible_devices)
            )
            entry['legal_device_count_last_seen'] = int(len(legal_devices))
            entry['idle_legal_device_count_last_seen'] = int(
                len(idle_legal_devices)
            )
        for key, entry in self.resource_intent_ledger.items():
            if key in active or entry.get('status') not in {'soft', 'firm'}:
                continue
            entry['status'] = 'cancelled'
            entry['history'].append({
                'status': 'cancelled', 'time': float(self.total_time)
            })

    def _clear_lookahead_reservation(self, device, reason):
        reservation = getattr(device, 'lookahead_reservation', None)
        if reservation is None:
            return False
        identity = tuple(reservation.get('identity', ()))
        intent = self.resource_intent_ledger.get(identity)
        if intent is not None:
            intent['reservation_release_time'] = float(self.total_time)
            intent['reservation_release_reason'] = str(reason)
            intent.setdefault('history', []).append({
                'status': 'reservation_released',
                'time': float(self.total_time),
                'reason': str(reason),
                'device_id': device.code,
            })
        device.lookahead_reservation = None
        return True

    def _reconcile_lookahead_reservations(self):
        """Expire leases and hand a ready R014 to departure atomically."""
        for device in self.device_list:
            reservation = getattr(device, 'lookahead_reservation', None)
            if not reservation:
                continue
            identity = tuple(reservation.get('identity', ()))
            intent = self.resource_intent_ledger.get(identity)
            if intent is not None and intent.get('status') == 'cancelled':
                self._clear_lookahead_reservation(
                    device, 'speculative_intent_cancelled'
                )
                continue
            plane = self.planes.get(reservation.get('plane_id'))
            if plane is None:
                self._clear_lookahead_reservation(device, 'plane_absent')
                continue
            if float(self.total_time) >= float(
                reservation.get('expires_at', self.total_time)
            ) - 1e-9:
                self._clear_lookahead_reservation(device, 'lease_expired')
                continue
            job_code = reservation.get('job_code')
            site_code = reservation.get('site_code')
            if job_code == self.TRANSFER_JOB_CODE:
                if (
                    bool(getattr(plane, 'departure_staging_decided', False))
                    and plane.site.code != site_code
                ):
                    self._clear_lookahead_reservation(
                        device, 'departure_pickup_site_changed'
                    )
                    continue
                if (
                    self._plane_awaiting_departure(plane)
                    and plane.is_idle()
                    and not device.is_transporting
                    and device.site == plane.site
                    and plane.site.code == site_code
                    and device.resource.is_available()
                ):
                    if self._reserve_departure_transporter(plane, device):
                        self._clear_lookahead_reservation(
                            device, 'promoted_to_departure_reservation'
                        )
                continue
            if (
                job_code in plane.current_jobs
                or job_code in plane.finished_jobs
                or job_code not in plane.left_jobs
            ):
                self._clear_lookahead_reservation(
                    device, 'job_started_or_cancelled'
                )

    def _next_lookahead_reservation_expiry_dt(self):
        delays = [
            float(reservation.get('expires_at', self.total_time))
            - float(self.total_time)
            for device in self.device_list
            for reservation in [getattr(
                device, 'lookahead_reservation', None
            )]
            if reservation is not None
            and float(reservation.get('expires_at', self.total_time))
            > float(self.total_time) + 1e-9
        ]
        return min(delays, default=math.inf)

    def _refresh_request_pool(self):
        """Build a stable request list from planes waiting for mobile resources."""
        if (
            self._lookahead_deferred_at_time is not None
            and not math.isclose(
                float(self.total_time),
                float(self._lookahead_deferred_at_time),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            self._deferred_lookahead_requests.clear()
            self._lookahead_deferred_at_time = None
        self._reconcile_lookahead_reservations()
        self._settle_ready_waiting_planes()
        # ``waiting_sites`` is a derived index.  Rebuild it after all
        # zero-time settlements so completed/departed aircraft cannot leave
        # stale queue entries in diagnostics or heuristic teacher matching.
        self._sync_waiting_sites()
        request_list = [{
            'id': 0,
            'job_code': '__WAIT__',
            'site_code': 'Z',
            'plane_id': None,
            'plane_idx': -1,
            'needed_res_types': [],
            'waiting_time': 0.0,
            'is_noop': True,
            'is_lookahead': False,
            'urgent': False,
            'request_kind': 'noop',
        }]
        seen = set()
        assigned = {
            (
                record.get('plane_id'),
                record.get('job_code'),
                record.get('site_id'),
            )
            for record in self.pending_device_actions.values()
        }
        next_id = 1

        for plane in self.planes.values():
            if next_id >= getattr(self, 'max_request_num', self.n_plane_agents + 1):
                break
            if not plane.is_waiting:
                continue
            needed_res_types = self._needed_mobile_types(plane.pending_job)
            if not needed_res_types:
                continue
            key = (plane.code, plane.pending_job, plane.site.code)
            if key in seen or key in assigned:
                continue
            seen.add(key)
            request_list.append({
                'id': next_id,
                'job_code': plane.pending_job,
                'site_code': plane.site.code,
                'plane_id': plane.code,
                'plane_idx': int(plane.code.split('_')[-1]),
                'needed_res_types': needed_res_types,
                'waiting_time': float(plane.waiting_time),
                'is_noop': False,
                'is_lookahead': False,
                'urgent': True,
                'request_kind': 'blocking_wait',
            })
            next_id += 1

        for request in self._departure_pickup_requests():
            if next_id >= getattr(
                self, 'max_request_num', self.n_plane_agents + 1
            ):
                break
            key = self._request_identity(request)
            if key in seen or key in assigned:
                continue
            seen.add(key)
            request['id'] = next_id
            request_list.append(request)
            next_id += 1

        if self.device_lookahead_dispatch:
            for request in self._iter_lookahead_requests(seen, assigned):
                if next_id >= self.max_request_num:
                    break
                key = self._request_identity(request)
                seen.add(key)
                request['id'] = next_id
                request_list.append(request)
                next_id += 1

        self.request_list = request_list
        self.request_pool = {req['id']: req for req in request_list}
        self._sync_resource_intent_ledger(request_list)

    def _heuristic_dispatch_departure_pickups(self):
        """Pre-position idle R014s for independently ready aircraft."""
        self._refresh_request_pool()
        requests = [
            request for request in self.request_list[1:]
            if request.get('request_kind') == 'departure_pickup'
        ]
        devices = [
            device
            for device in self.mobile_devices.get(
                self.TRANSPORTER_RESOURCE_TYPE, []
            )
            if self._device_is_dispatchable(device)
        ]
        if not requests or not devices:
            return 0
        requests.sort(key=lambda request: int(request['id']))
        devices.sort(key=lambda device: device.code)
        costs = np.zeros((len(devices), len(requests)), dtype=np.float64)
        for row, device in enumerate(devices):
            for column, request in enumerate(requests):
                target = self.sites[request['site_code']]
                distance = (
                    abs(float(device.site.pos[0]) - float(target.pos[0]))
                    + abs(float(device.site.pos[1]) - float(target.pos[1]))
                )
                travel_time = distance / max(float(device.velocity), 1.0)
                costs[row, column] = (
                    travel_time
                    - float(request.get('waiting_time', 0.0))
                    + 1e-9 * int(request['id'])
                )
        rows, columns = linear_sum_assignment(costs)
        started = 0
        for row, column in zip(rows, columns):
            device = devices[int(row)]
            request = requests[int(column)]
            if self._start_device_request(
                device, request, int(request['id'])
            ):
                started += 1
        return started

    def _device_can_serve(self, device, request):
        if request.get('is_noop', False):
            return True
        if device.resource.type not in request.get('needed_res_types', []):
            return False
        return not self._lookahead_reservation_blocks(
            device, request=request
        )

    def _device_is_dispatchable(self, device):
        return bool(
            device.is_idle()
            and device.resource.is_available()
            and getattr(device, 'reserved_for_plane', None) is None
            and not (
                getattr(device, 'lookahead_reservation', None)
                and device.lookahead_reservation.get('mode') == 'hard'
            )
        )

    def _lookahead_dispatch_delay(self, device, request):
        """Seconds until a JIT request enters this device's legal window."""
        if (
            not getattr(self, 'device_deadline_aware_dispatch', False)
            or not request.get('is_lookahead', False)
        ):
            return 0.0
        target_site = self.sites.get(request.get('site_code'))
        if target_site is None:
            return math.inf
        travel_time = self._device_travel_seconds(device, target_site)
        lead_time = max(0.0, float(request.get('lead_time', 0.0)))
        return max(
            0.0,
            lead_time - travel_time - self.device_lookahead_safety_margin,
        )

    def _device_can_dispatch(self, device, request):
        return bool(
            self._device_is_dispatchable(device)
            and self._device_can_serve(device, request)
            and self._lookahead_dispatch_delay(device, request) <= 1e-9
        )

    def _next_lookahead_dispatch_dt(self):
        """Return the next event time at which one JIT action becomes legal."""
        if not getattr(self, 'device_deadline_aware_dispatch', False):
            return math.inf
        delays = []
        for request in self.request_list[1:]:
            if not request.get('is_lookahead', False):
                continue
            for device in self.device_list[:self.max_device_num]:
                if (
                    self._device_is_dispatchable(device)
                    and self._device_can_serve(device, request)
                ):
                    delay = self._lookahead_dispatch_delay(device, request)
                    if delay > 1e-9 and math.isfinite(delay):
                        delays.append(delay)
        return min(delays, default=math.inf)

    def _sequential_device_options(self, dev_idx, claimed_requests):
        """Return real requests and whether no-op is legal for one device turn.

        No-op is available while a later dispatchable device can still serve
        every request available to the current device. If one or more requests
        have no later compatible device, the current device must choose among
        those last-chance requests. This makes device identity learnable without
        permitting an all-noop joint action to stall the event loop.
        """
        if dev_idx < 0 or dev_idx >= min(len(self.device_list), self.max_device_num):
            return [], True

        device = self.device_list[dev_idx]
        valid_requests = [
            req for req in self.request_list[1:]
            if int(req['id']) not in claimed_requests
            and self._device_can_dispatch(device, req)
        ]
        if not valid_requests:
            return [], True

        later_devices = self.device_list[dev_idx + 1:self.max_device_num]
        # A lookahead request has not blocked its plane yet, so all devices may
        # legally defer it.  Only a blocking request participates in the
        # last-compatible-device rule that prevents an all-noop deadlock.
        blocking_requests = [
            req for req in valid_requests
            if not bool(req.get('is_lookahead', False))
        ]
        last_chance_requests = [
            req for req in blocking_requests
            if not any(self._device_can_dispatch(later, req) for later in later_devices)
        ]
        if last_chance_requests:
            return last_chance_requests, False
        return valid_requests, True

    def _plan_device_actions(self, action):
        """Validate a complete device turn before mutating device state.

        The actor builds its autoregressive masks from one static observation.
        Planning every device action against that same state prevents an early
        local dispatch from changing the no-op legality of a later agent while
        the joint action is still being validated.
        """
        if self.request_mask_matrix is None:
            raise RuntimeError("Request masks must be built before executing device actions.")

        claimed_requests = set()
        dispatch_plan = []
        for dev_idx, device in enumerate(self.device_list[:self.max_device_num]):
            agent_id = self.n_plane_agents + dev_idx
            if agent_id >= self.n_agents or not self._device_is_dispatchable(device):
                continue

            initial_request_ids = (
                np.flatnonzero(self.request_mask_matrix[agent_id, 1:]) + 1
            ).astype(int).tolist()
            if not initial_request_ids:
                continue

            legal_requests, allow_noop = self._sequential_device_options(
                dev_idx,
                claimed_requests,
            )
            valid_request_ids = [int(req['id']) for req in legal_requests]
            req_idx = int(action[agent_id][0])

            if req_idx == 0:
                if not allow_noop:
                    raise RuntimeError(
                        f"Device agent {agent_id} is the last compatible device for "
                        f"requests {valid_request_ids} and cannot select no-op."
                    )
                continue

            if req_idx not in valid_request_ids:
                raise RuntimeError(
                    f"Device agent {agent_id} selected masked request {req_idx}; "
                    f"legal requests are {valid_request_ids}, allow_noop={allow_noop}."
                )
            request = self.request_pool.get(req_idx)
            if request is None:
                raise RuntimeError(
                    f"Device agent {agent_id} selected missing request {req_idx}."
                )
            dispatch_plan.append((device, request, req_idx))
            claimed_requests.add(req_idx)

        return dispatch_plan

    def _execute_device_action_plan(self, dispatch_plan):
        """Execute validated actions, moving remote devices before local jobs.

        A local job may consume other idle resources at its site. Moving every
        remotely assigned device first reserves those devices for the requests
        selected by the same joint action.
        """
        def is_local_dispatch(item):
            device, request, _ = item
            return device.site == self.sites.get(request['site_code'])

        from onpolicy.utils.stage2_resource_v6_observation import capture_dispatches
        semantic_snapshots = capture_dispatches(self, dispatch_plan)
        ordered_plan = sorted(dispatch_plan, key=is_local_dispatch)
        for device, request, req_idx in ordered_plan:
            if not self._start_device_request(device, request, req_idx):
                raise RuntimeError(
                    f"Device {device.code} selected request {req_idx}, but the "
                    "validated dispatch plan could not be started."
                )
            if device.code in semantic_snapshots:
                self._v6_dispatch_history[device.code] = semantic_snapshots[device.code]

    def _dispatch_device_actions(self, action):
        dispatch_plan = self._plan_device_actions(action)
        selected_request_ids = {
            int(request_id) for _, _, request_id in dispatch_plan
        }
        deferred_keys = {
            self._request_identity(request)
            for request in self.request_list[1:]
            if bool(request.get('is_lookahead', False))
            and int(request['id']) not in selected_request_ids
            and any(
                self._device_can_dispatch(device, request)
                for device in self.device_list[:self.max_device_num]
            )
        }
        self._execute_device_action_plan(dispatch_plan)
        if deferred_keys:
            self._deferred_lookahead_requests.update(deferred_keys)
            self._lookahead_deferred_at_time = float(self.total_time)

    def _start_device_request(self, device, request, action_idx):
        if request.get('is_noop', False):
            return False
        if not self._device_can_dispatch(device, request):
            return False
        target_site = self.sites.get(request['site_code'])
        if target_site is None:
            return False

        existing_lookahead_reservation = getattr(
            device, 'lookahead_reservation', None
        )
        if (
            existing_lookahead_reservation
            and tuple(existing_lookahead_reservation.get('identity', ()))
            != self._request_identity(request)
        ):
            # Only a soft lease can reach this point; hard leases are masked.
            self._clear_lookahead_reservation(
                device, 'stolen_by_blocking_request'
            )

        if not device.resource.is_available():
            return False

        is_departure_pickup = (
            request.get('request_kind') == 'departure_pickup'
        )
        departure_plane = None
        if is_departure_pickup:
            departure_plane = self.planes.get(request.get('plane_id'))
            if departure_plane is None or not self._reserve_departure_transporter(
                departure_plane, device
            ):
                return False

        step_idx = self.steps
        agent_id = self.device_code_to_agent_id.get(device.code)
        start_site = device.site.code
        try:
            if device.site == target_site:
                if device.resource.code not in target_site.resources:
                    target_site.add_resource(device.resource)
                if device not in target_site.devices:
                    target_site.devices.append(device)
                device.resource.sites = [target_site.code]
                trans_time = 0.0
            else:
                trans_time = device.start_transport(target_site)
        except Exception:
            if is_departure_pickup:
                self._release_departure_transporter(
                    plane_code=departure_plane.code,
                    device_code=device.code,
                )
            raise
        if (
            request.get('is_lookahead', False)
            and self.device_lookahead_reservation_mode in {'soft', 'hard'}
        ):
            identity = self._request_identity(request)
            needed_time = float(self.total_time) + max(
                0.0, float(request.get('lead_time', 0.0))
            )
            device.lookahead_reservation = {
                'identity': list(identity),
                'plane_id': request.get('plane_id'),
                'job_code': request.get('job_code'),
                'site_code': request.get('site_code'),
                'mode': self.device_lookahead_reservation_mode,
                'created_time': float(self.total_time),
                'needed_time': needed_time,
                'expires_at': float(
                    needed_time + self.device_reservation_grace_seconds
                ),
            }
        self._remove_waiting_site(request['job_code'], target_site.code)

        if agent_id is not None:
            record = {
                'step_idx': step_idx,
                'agent_id': agent_id,
                'action': [int(action_idx), 0],
                'start_time': self.total_time,
                'device_id': device.code,
                'device_type': device.resource.type,
                'is_transporter': self._is_transporter_device(device),
                'from_site': start_site,
                'site_id': target_site.code,
                'job_code': request['job_code'],
                'plane_id': request.get('plane_id'),
                'waiting_time_at_dispatch': float(request.get('waiting_time', 0.0)),
                'trans_time': float(trans_time),
                'duration': float(trans_time),
                'request_kind': request.get('request_kind'),
                'dependency_depth': int(request.get('dependency_depth', 0)),
                'predicted_need_time': float(request.get(
                    'predicted_need_time', self.total_time
                )),
            }
            if request.get('is_lookahead', False):
                record.update({
                    'is_lookahead': True,
                    'request_kind': request.get('request_kind'),
                    'reservation_mode': str(
                        self.device_lookahead_reservation_mode
                    ),
                    'lead_time_at_dispatch': float(
                        request.get('lead_time', 0.0)
                    ),
                    'predicted_lateness_at_dispatch': max(
                        0.0,
                        float(trans_time)
                        - float(request.get('lead_time', 0.0)),
                    ),
                    'predicted_earliness_at_dispatch': max(
                        0.0,
                        float(request.get('lead_time', 0.0))
                        - float(trans_time),
                    ),
                })
            intent_key = self._request_identity(request)
            record['intent_identity'] = list(intent_key)
            intent = self.resource_intent_ledger.setdefault(intent_key, {
                'identity': list(intent_key),
                'created_time': float(self.total_time),
                'first_visible_time': float(self.total_time),
                'history': [],
            })
            intent.setdefault('first_legal_time', float(self.total_time))
            intent.setdefault(
                'first_compatible_idle_time', float(self.total_time)
            )
            intent.update({
                'status': 'dispatched',
                'assigned_device': device.code,
                'dispatch_time': float(self.total_time),
                'expected_arrival_time': float(self.total_time + trans_time),
                'reservation_mode': (
                    self.device_lookahead_reservation_mode
                    if request.get('is_lookahead', False) else 'none'
                ),
            })
            record.update({
                'first_visible_time': float(intent['first_visible_time']),
                'first_legal_time': float(intent['first_legal_time']),
                'first_compatible_idle_time': float(
                    intent['first_compatible_idle_time']
                ),
                'visibility_to_legal_seconds': max(
                    0.0,
                    float(intent['first_legal_time'])
                    - float(intent['first_visible_time']),
                ),
                'legal_to_idle_seconds': max(
                    0.0,
                    float(intent['first_compatible_idle_time'])
                    - float(intent['first_legal_time']),
                ),
                'policy_defer_seconds': max(
                    0.0,
                    float(self.total_time)
                    - float(intent['first_compatible_idle_time']),
                ),
            })
            intent['history'].append({
                'status': 'dispatched',
                'time': float(self.total_time),
                'device_id': device.code,
            })
            if trans_time <= 0.0:
                if device.is_transporting:
                    device.finish_transport()
                settled = self._settle_ready_waiting_planes(
                    site_code=target_site.code,
                    device=device,
                )
                record['end_time'] = self.total_time
                record['duration'] = 0.0
                record['settled_waiting_planes'] = int(settled)
                self.device_trajectory_log.append(record)
                intent['status'] = 'arrived'
                intent['arrival_time'] = float(self.total_time)
                intent['history'].append({
                    'status': 'arrived', 'time': float(self.total_time)
                })
            else:
                self.pending_device_actions[device.code] = record
        return True

    def _build_global_features(self):
        """Return a fixed-width, normalized global scheduling summary."""
        feature_dim = 24
        features = np.zeros(feature_dim, dtype=np.float32)
        mode = getattr(self, 'global_feature_mode', 'none')
        if mode == 'none':
            return features

        max_planes = max(1.0, float(self.n_plane_agents))
        n_ops = max(1.0, float(len(self.job_code_list)))
        active_planes = list(self.planes.values())
        active_count = len(active_planes)
        future_count = len(self.landing_list)
        completed_count = max(
            0,
            len(self.flights_data) - active_count - future_count,
        )
        idle_count = sum(plane.is_idle() for plane in active_planes)
        remaining_ops = [
            float(len(plane.left_jobs) + len(plane.current_jobs))
            for plane in active_planes
        ]
        remaining_work = sum(
            float(self.jobs[code].time or 0.0)
            for plane in active_planes
            for code in list(plane.left_jobs) + list(plane.current_jobs)
            if code in self.jobs
        )
        one_plane_work = sum(
            float(self.jobs[code].time or 0.0)
            for code in self.job_code_list
            if code in self.jobs
        )
        future_work = float(future_count) * one_plane_work
        future_arrivals = sorted(
            max(0.0, float(item[0]) - float(self.total_time))
            for item in self.landing_list
        )

        service_sites = [
            site
            for site in self.sites.values()
            if site.code not in self.runway_code_list and site.code != 'Z'
        ]
        site_denominator = max(1.0, float(len(service_sites)))
        free_sites = sum(
            not site.is_occupied and not site.is_interfered
            for site in service_sites
        )
        interfered_sites = sum(site.is_interfered for site in service_sites)

        features[:12] = np.asarray([
            active_count / max_planes,
            future_count / max_planes,
            completed_count / max_planes,
            idle_count / max(1.0, float(active_count)),
            sum(remaining_ops) / (max_planes * n_ops),
            (
                np.mean(remaining_ops) / n_ops
                if remaining_ops else 0.0
            ),
            min(remaining_work / (max_planes * 3600.0), 10.0),
            min(future_work / (max_planes * 3600.0), 10.0),
            min((future_arrivals[0] if future_arrivals else 36000.0) / 3600.0, 10.0),
            min((future_arrivals[min(2, len(future_arrivals) - 1)]
                 if future_arrivals else 36000.0) / 3600.0, 10.0),
            free_sites / site_denominator,
            interfered_sites / site_denominator,
        ], dtype=np.float32)
        if mode == 'f1':
            return features

        devices = list(self.device_list)
        availability = [
            (
                self._device_release_seconds(device)
                if self.resource_release_aware_eta
                else float(max(
                    device.left_trans_time, device.left_rec_time
                ))
            )
            for device in devices
        ]
        device_available = [
            self._device_observation_available(device) for device in devices
        ]
        real_requests = [
            request
            for request in self.request_list
            if not request.get('is_noop', False)
        ]
        request_waits = [
            float(request.get('waiting_time', 0.0))
            for request in real_requests
        ]
        transporters = self.mobile_devices.get(
            self.TRANSPORTER_RESOURCE_TYPE, []
        )
        if mode == 'f1f2_departure':
            ready_planes = [
                plane for plane in active_planes
                if self._plane_awaiting_departure(plane)
            ]
            ready_ages = [
                max(
                    0.0,
                    float(self.total_time)
                    - float(self.departure_ready_since.get(
                        plane.code, self.total_time
                    )),
                )
                for plane in ready_planes
            ]
            runway_sites = [
                self.sites[code]
                for code in self.takeoff_site_code_list
                if code in self.sites
            ]
            runway_busy = sum(
                site.is_occupied or site.is_interfered
                for site in runway_sites
            )
            transporter_unavailable = sum(
                not self._device_observation_available(device)
                for device in transporters
            )
            transporter_idle_unreserved = sum(
                self._device_observation_available(device)
                for device in transporters
            )
            features[12:] = np.asarray([
                sum(device_available)
                / max(1.0, float(len(devices))),
                min(
                    (np.mean(availability) if availability else 0.0)
                    / 3600.0,
                    10.0,
                ),
                min(
                    (max(availability) if availability else 0.0) / 3600.0,
                    10.0,
                ),
                len(real_requests)
                / max(1.0, float(self.max_request_num - 1)),
                min(
                    (np.mean(request_waits) if request_waits else 0.0)
                    / 3600.0,
                    10.0,
                ),
                min(
                    (max(request_waits) if request_waits else 0.0) / 3600.0,
                    10.0,
                ),
                len(ready_planes) / max_planes,
                min(
                    (np.mean(ready_ages) if ready_ages else 0.0) / 3600.0,
                    10.0,
                ),
                min(
                    (max(ready_ages) if ready_ages else 0.0) / 3600.0,
                    10.0,
                ),
                runway_busy / max(1.0, float(len(runway_sites))),
                transporter_unavailable
                / max(1.0, float(len(transporters))),
                transporter_idle_unreserved
                / max(1.0, float(len(transporters))),
            ], dtype=np.float32)
            return np.nan_to_num(
                features, nan=0.0, posinf=10.0, neginf=0.0
            )

        device_types = sorted(self.mobile_devices)
        demand_by_type = {resource_type: 0.0 for resource_type in device_types}
        for plane in active_planes:
            for code in plane.left_jobs:
                if code not in self.jobs:
                    continue
                for resource_type in self.jobs[code].resources:
                    if resource_type in demand_by_type:
                        demand_by_type[resource_type] += 1.0
        demand_values = [
            demand_by_type[resource_type] / (max_planes * n_ops)
            for resource_type in device_types
        ]
        capacity_values = [
            float(len(self.mobile_devices[resource_type]))
            / max(1.0, float(len(devices)))
            for resource_type in device_types
        ]
        capacity_demand = [
            float(len(self.mobile_devices[resource_type]))
            / max(1.0, demand_by_type[resource_type])
            for resource_type in device_types
        ]
        features[12:] = np.asarray([
            sum(device_available)
            / max(1.0, float(len(devices))),
            min((np.mean(availability) if availability else 0.0) / 3600.0, 10.0),
            min((max(availability) if availability else 0.0) / 3600.0, 10.0),
            len(real_requests) / max(1.0, float(self.max_request_num - 1)),
            min((np.mean(request_waits) if request_waits else 0.0) / 3600.0, 10.0),
            min((max(request_waits) if request_waits else 0.0) / 3600.0, 10.0),
            float(np.mean(demand_values)) if demand_values else 0.0,
            max(demand_values) if demand_values else 0.0,
            float(np.mean(capacity_values)) if capacity_values else 0.0,
            min(capacity_demand) / 5.0 if capacity_demand else 0.0,
            min(float(np.mean(capacity_demand)), 5.0) / 5.0
            if capacity_demand else 0.0,
            sum(
                self._device_observation_available(device)
                for device in transporters
            )
            / max(1.0, float(len(transporters))),
        ], dtype=np.float32)
        return np.nan_to_num(
            features, nan=0.0, posinf=10.0, neginf=0.0
        )

    def _get_obs(self):
        """
        更新环境状态并构建异构图数据对象 (HeteroData)。 
        【采用静态拓扑】：无论飞机是否在场，恒定生成 n_agents * n_ops 个工序节点，
        确保网络在不同 Step、不同 Episode 获得的张量形状绝对一致。
        """
        self._refresh_request_pool()
        data = HeteroData()
        
        # ==========================================
        # 1. 解析全局常量与映射字典
        # ==========================================
        site_list = list(self.sites.values())
        site2idx = {site.code: idx for idx, site in enumerate(site_list)}
        n_sites = len(site_list)
        
        device_list = self.device_list
        device2idx = {dev.code: idx for idx, dev in enumerate(device_list)}
        dev_types = list(self.mobile_devices.keys())

        # n_ops 直接由初始化好的 job_code_list 决定
        n_ops = len(self.job_code_list)
        n_plane_agents = self.n_plane_agents
        n_agents = self.n_agents
        
        # 提取当前场上活跃的飞机，映射为全局唯一 PID
        active_planes = {}
        for plane in self.planes.values():
            parts = plane.code.split('_')
            bidx, pidx = int(parts[1]), int(parts[2])
            global_pid = bidx * self.plane_num_per_batch + pidx
            active_planes[global_pid] = plane

        # ==========================================
        # 2. 构建停机位与设备节点 (Site & Device)
        # ==========================================
        site_features = []
        global_site_valid = []
        coordinate_scale = max(
            1.0,
            max(
                (
                    abs(float(coordinate))
                    for site in site_list
                    for coordinate in site.pos
                ),
                default=1.0,
            ),
        )
        for site in site_list:
            occ = 1.0 if site.is_occupied else 0.0
            interf = 1.0 if site.is_interfered else 0.0
            rem_time = min(
                float(max(site.left_job_time, site.left_rec_time)) / 3600.0,
                10.0,
            )
            job_onehot = site.avail_job_onehot if hasattr(site, 'avail_job_onehot') else [0] * n_ops
            
            site_features.append([
                occ,
                interf,
                rem_time,
                float(site.pos[0]) / coordinate_scale,
                float(site.pos[1]) / coordinate_scale,
            ] + job_onehot)
            
            # 物理限制
            if site.code in self.runway_code_list or site.is_interfered or site.is_occupied or site.code == "Z":
                global_site_valid.append(False)
            else:
                global_site_valid.append(True)
                
        data['site'].x = torch.tensor(site_features, dtype=torch.float32)
        
        device_features = []
        for dev in device_list:
            dtype_enc = float(dev_types.index(dev.resource.type)) / max(
                1.0, float(len(dev_types) - 1)
            )
            if self.resource_release_aware_eta:
                status_enc = (
                    0.0 if self._device_observation_available(dev) else 1.0
                )
                rem_time_seconds = self._device_release_seconds(dev)
            else:
                status_enc = 0.0 if dev.is_idle() else 1.0
                rem_time_seconds = float(max(
                    dev.left_trans_time, dev.left_rec_time
                ))
            if (
                dev.is_idle()
                and (
                    getattr(dev, 'reserved_for_plane', None) is not None
                    or getattr(dev, 'lookahead_reservation', None) is not None
                )
                and (
                    self.resource_release_aware_eta
                    or self.global_feature_mode == 'f1f2_departure'
                )
            ):
                # Reserved-idle is distinct from both genuinely available and
                # physically busy, without changing the five-dimensional node.
                status_enc = 0.5
            rem_time = min(
                float(rem_time_seconds) / 3600.0,
                10.0,
            )
            pos_x = float(dev.site.pos[0]) / coordinate_scale
            pos_y = float(dev.site.pos[1]) / coordinate_scale
            device_features.append([dtype_enc, status_enc, rem_time, pos_x, pos_y])
            
        if len(device_features) > 0:
            data['device'].x = torch.tensor(device_features, dtype=torch.float32)
        else:
            data['device'].x = torch.empty((0, 5), dtype=torch.float32)

        # ==========================================
        # 3. 构建工序节点 (Operation) & 机位掩码 (Site Mask)
        # ==========================================
        # The presence bit lets the encoder exclude ghost aircraft from global
        # pooling. Continuous magnitudes are normalized before projection.
        op_features = np.zeros(
            (n_plane_agents * n_ops, 11), dtype=np.float32
        )
        agent_op_mask = np.zeros((n_agents, n_plane_agents * n_ops), dtype=bool)
        
        # 维度缩减为 (n_agents, n_sites)
        ptr_site_mask_matrix = np.zeros((n_agents, n_sites), dtype=bool)
        job_site_mask_matrix = np.zeros((n_ops, n_sites), dtype=bool)
        agent_job_site_mask_matrix = np.zeros(
            (n_agents, n_ops, n_sites),
            dtype=bool,
        )
        departure_codes = set(self.departure_job_code_list)
        for job_idx, job_code in enumerate(self.job_code_list):
            job = self.jobs[job_code]
            for site_idx, site in enumerate(site_list):
                if job_code in departure_codes:
                    compatible = site.code in self.takeoff_site_code_list
                elif job_code == self.TRANSFER_JOB_CODE:
                    compatible = (
                        site.code != 'Z'
                        and site.code not in self.takeoff_site_code_list
                    )
                else:
                    compatible = (
                        site.code not in self.runway_code_list
                        and self._site_can_eventually_support_job(site, job)
                    )
                job_site_mask_matrix[job_idx, site_idx] = compatible

        departure_plan = self._compute_departure_runway_plan()
        self._departure_runway_plan = dict(departure_plan)
        departure_candidates = set(departure_plan)
        
        for global_pid in range(n_plane_agents):
            plane = active_planes.get(global_pid, None)
            
            if plane is not None:
                # ---------------------------------------------------
                # A. 计算该飞机对所有机位的合法性 (不再依赖具体的 job)
                # ---------------------------------------------------
                for s_idx, site in enumerate(site_list):
                    if global_site_valid[s_idx]:
                        # 1. 全局合法（无人占用且无干涉），此机位可用
                        ptr_site_mask_matrix[global_pid, s_idx] = True
                    else:
                        if site.code == 'Z':
                            ptr_site_mask_matrix[global_pid, s_idx] = False
                        # 2. 全局不合法，但如果占用它的正是当前这架飞机本身，且无干涉，则可用（允许飞机原地干活）
                        elif site == plane.site and not site.is_interfered:
                            ptr_site_mask_matrix[global_pid, s_idx] = True
                        elif (
                            site.code in self.takeoff_site_code_list
                            and plane.code in departure_candidates
                            and not site.is_occupied
                            and not site.is_interfered
                        ):
                            ptr_site_mask_matrix[global_pid, s_idx] = True
                        else:
                            ptr_site_mask_matrix[global_pid, s_idx] = False

                # ---------------------------------------------------
                # B. 计算工序特征与掩码
                # ---------------------------------------------------
                current_avail_jobs = self._ready_job_codes(
                    plane, departure_plan=departure_plan
                )
                for j_idx, job_code in enumerate(self.job_code_list):
                    u_idx = global_pid * n_ops + j_idx 
                    job_obj = plane.jobs.get(job_code, self.jobs[job_code])
                    
                    if (
                        job_code == self.TRANSFER_JOB_CODE
                        and plane.departure_staging_decided
                    ):
                        status, is_ready = 3.0, False
                    elif job_code in plane.finished_jobs:
                        status, is_ready = 3.0, False
                    elif job_code in plane.current_jobs:
                        status, is_ready = 2.0, False
                    elif job_code in current_avail_jobs:
                        status, is_ready = 1.0, True
                    else:
                        status, is_ready = 0.0, False
                        
                    can_schedule = plane.is_idle() and is_ready
                    agent_op_mask[global_pid, u_idx] = can_schedule
                    
                    proc_time = (
                        min(float(job_obj.time) / 3600.0, 10.0)
                        if job_obj.time else 0.0
                    )
                    rem_ops = float(len(plane.left_jobs)) / max(
                        1.0, float(n_ops)
                    )
                    requires_departure_tow = (
                        bool(self.departure_job_code_list)
                        and job_code == self.departure_job_code_list[0]
                    )
                    req_res = 1.0 if (
                        len(set(job_obj.resources).intersection(set(dev_types))) > 0
                        or requires_departure_tow
                    ) else 0.0
                    wait_time = (
                        min(float(plane.waiting_time) / 3600.0, 10.0)
                        if status == 1.0 else 0.0
                    )
                    relocations = min(float(plane.relocations_since_progress) / 20.0, 10.0)
                    no_progress = min(float(plane.no_progress_decisions) / 50.0, 10.0)
                    is_long_occupancy = float(job_code in plane.LONG_OCCUPANCY_JOBS)
                    irreversible_progress = min(
                        float(plane.irreversible_progress_count)
                        / max(1.0, float(n_ops)),
                        1.0,
                    )
                    
                    op_features[u_idx] = [
                        status / 3.0,
                        proc_time,
                        rem_ops,
                        req_res,
                        wait_time,
                        float(global_pid) / max(
                            1.0, float(n_plane_agents - 1)
                        ),
                        relocations,
                        no_progress,
                        is_long_occupancy,
                        irreversible_progress,
                        1.0,
                    ]
                    
            else:
                # 飞机不在场上：幽灵节点
                for j_idx, job_code in enumerate(self.job_code_list):
                    u_idx = global_pid * n_ops + j_idx
                    op_features[u_idx] = [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        float(global_pid) / max(
                            1.0, float(n_plane_agents - 1)
                        ),
                        0.0,
                        0.0,
                        float(job_code in {'ZY02', 'ZY03'}),
                        0.0,
                        0.0,
                    ]
            if plane is not None:
                own_op_mask = agent_op_mask[
                    global_pid,
                    global_pid * n_ops:(global_pid + 1) * n_ops,
                ]
                agent_job_site_mask_matrix[global_pid] = self._build_plane_pair_mask(
                    plane,
                    own_op_mask,
                    ptr_site_mask_matrix[global_pid],
                    job_site_mask_matrix,
                )

            if plane is not None and self._plane_can_decide(
                plane, departure_plan=departure_plan
            ):
                joint_mask = (
                    own_op_mask[:, None]
                    & agent_job_site_mask_matrix[global_pid]
                )
                if not joint_mask.any():
                    raise RuntimeError(
                        f"Active plane {plane.code} has no legal operation-site pair "
                        f"at environment step {self.steps}."
                    )
                
        data['operation'].x = torch.tensor(op_features, dtype=torch.float32)

        # Target-specific operation-site features are consumed directly by the
        # joint decoder. They expose moving versus staying, travel, resource
        # arrival and downstream site flexibility without forcing the GNN to
        # reconstruct those quantities from unrelated node embeddings.
        pair_feature_dim = 10
        pair_feature_storage = str(
            self.config.get('pair_feature_storage', 'sparse_legal')
        )
        if pair_feature_storage not in {'sparse_legal', 'dense_legacy'}:
            raise ValueError(
                'pair_feature_storage must be sparse_legal or dense_legacy, '
                f'got {pair_feature_storage!r}.'
            )
        # Resource actors choose request nodes and never consume plane
        # operation-site features.  The legacy tensor nevertheless allocated
        # [plane + device, job, site, feature] and therefore stored 80 rows of
        # zeros in every Stage2 transition.  In sparse mode we keep only the
        # operation-site pairs that are legal in this observation.  Masked
        # logits have no probability or gradient contribution, so scoring
        # this exact legal set is numerically equivalent to the dense path.
        dense_pair_features = (
            np.zeros(
                (n_agents, n_ops, n_sites, pair_feature_dim),
                dtype=np.float32,
            )
            if pair_feature_storage == 'dense_legacy' else None
        )
        sparse_pair_feature_values = []
        sparse_pair_feature_flat_ids = []
        current_site_indices = np.full((n_agents,), -1, dtype=np.int64)
        critical_job_codes = {'ZY10', 'ZY12'}
        critical_op_mask = np.asarray(
            [code in critical_job_codes for code in self.job_code_list],
            dtype=bool,
        )
        for global_pid, plane in active_planes.items():
            current_site_idx = site2idx.get(plane.site.code, -1)
            current_site_indices[global_pid] = current_site_idx
            own_op_mask = agent_op_mask[
                global_pid,
                global_pid * n_ops:(global_pid + 1) * n_ops,
            ]
            legal_pair_mask = (
                own_op_mask[:, None]
                & agent_job_site_mask_matrix[global_pid]
                & ptr_site_mask_matrix[global_pid][None, :]
            )
            job_indices = (
                range(n_ops)
                if dense_pair_features is not None
                else np.flatnonzero(legal_pair_mask.any(axis=1))
            )
            for raw_job_idx in job_indices:
                job_idx = int(raw_job_idx)
                job_code = self.job_code_list[job_idx]
                job_obj = plane.jobs.get(job_code, self.jobs[job_code])
                is_departure_tow = bool(
                    self.departure_job_code_list
                    and job_code == self.departure_job_code_list[0]
                )
                reserved_departure_device = (
                    self._departure_transporter_for_plane(plane)
                    if is_departure_tow else None
                )
                required_device_types = sorted(
                    set(job_obj.resources).intersection(dev_types)
                )
                if (
                    is_departure_tow
                    and self.TRANSPORTER_RESOURCE_TYPE in dev_types
                ):
                    required_device_types = [self.TRANSPORTER_RESOURCE_TYPE]
                remaining_predecessors = (
                    0
                    if job_code == self.TRANSFER_JOB_CODE
                    else sum(
                        predecessor not in plane.finished_jobs
                        for predecessor in job_obj.predecessor
                    )
                )
                site_indices = (
                    range(n_sites)
                    if dense_pair_features is not None
                    else np.flatnonzero(legal_pair_mask[job_idx])
                )
                for raw_site_idx in site_indices:
                    site_idx = int(raw_site_idx)
                    site = site_list[site_idx]
                    distance = (
                        abs(float(plane.site.pos[0]) - float(site.pos[0]))
                        + abs(float(plane.site.pos[1]) - float(site.pos[1]))
                    )
                    travel_time = distance / max(
                        float(getattr(plane, 'velocity', 1.0)), 1.0
                    )
                    type_etas = []
                    pair_request = {
                        'plane_id': plane.code,
                        'job_code': job_code,
                        'site_code': site.code,
                        'is_lookahead': True,
                    }
                    for device_type in required_device_types:
                        candidates = (
                            [reserved_departure_device]
                            if (
                                is_departure_tow
                                and device_type
                                == self.TRANSPORTER_RESOURCE_TYPE
                                and reserved_departure_device is not None
                            )
                            else self.mobile_devices.get(device_type, [])
                        )
                        candidates = [
                            device for device in candidates
                            if getattr(device, 'reserved_for_plane', None)
                            in {None, plane.code}
                        ]
                        if not candidates:
                            type_etas.append(36000.0)
                            break
                        if self.resource_release_aware_eta:
                            type_eta = min(
                                self._device_eta_seconds(
                                    device, site, request=pair_request
                                )
                                for device in candidates
                            )
                        else:
                            type_eta = min(
                                float(max(
                                    device.left_trans_time,
                                    device.left_rec_time,
                                )) + (
                                    abs(float(device.site.pos[0]) - float(site.pos[0]))
                                    + abs(float(device.site.pos[1]) - float(site.pos[1]))
                                ) / max(
                                    float(getattr(device, 'velocity', 1.0)), 1.0
                                )
                                for device in candidates
                            )
                        type_etas.append(float(type_eta))
                    device_eta = (
                        min(type_etas, default=0.0)
                        if self.resource_release_aware_eta
                        else max(type_etas, default=0.0)
                    )
                    future_compatibility = float(
                        sum(
                            self._site_can_eventually_support_job(
                                site, self.jobs[future_code]
                            )
                            for future_code in plane.left_jobs
                            if future_code in self.jobs
                        )
                    ) / max(1.0, float(len(plane.left_jobs)))
                    feature_value = [
                        float(site_idx == current_site_idx),
                        min(travel_time / 3600.0, 10.0),
                        min(float(job_obj.time or 0.0) / 3600.0, 10.0),
                        min(
                            float(max(site.left_job_time, site.left_rec_time))
                            / 3600.0,
                            10.0,
                        ),
                        min(device_eta / 3600.0, 10.0),
                        float(len(required_device_types))
                        / max(1.0, float(len(dev_types))),
                        float(job_code in plane.LONG_OCCUPANCY_JOBS),
                        float(remaining_predecessors)
                        / max(1.0, float(len(job_obj.predecessor))),
                        future_compatibility,
                        float(job_code in critical_job_codes),
                    ]
                    if dense_pair_features is not None:
                        dense_pair_features[
                            global_pid, job_idx, site_idx
                        ] = feature_value
                    else:
                        sparse_pair_feature_values.append(feature_value)
                        sparse_pair_feature_flat_ids.append(
                            (global_pid * n_ops + job_idx) * n_sites
                            + site_idx
                        )
        if dense_pair_features is not None:
            data.pair_features = torch.from_numpy(dense_pair_features)
        else:
            sparse_values = np.asarray(
                sparse_pair_feature_values, dtype=np.float32
            ).reshape(-1, pair_feature_dim)
            sparse_flat_ids = np.asarray(
                sparse_pair_feature_flat_ids, dtype=np.int32
            )
            data.pair_feature_values = torch.from_numpy(sparse_values)
            # ``flat_ids`` deliberately avoids PyG's special ``index`` name:
            # local plane-pair ids must be concatenated without graph offsets.
            data.pair_feature_flat_ids = torch.from_numpy(sparse_flat_ids)
            data.pair_feature_counts = torch.tensor(
                [len(sparse_pair_feature_flat_ids)], dtype=torch.int32
            )
        data.agent_current_site_indices = torch.tensor(
            current_site_indices, dtype=torch.long
        )
        data.critical_op_mask = torch.tensor(
            critical_op_mask, dtype=torch.bool
        )
        job_phase_codes = np.asarray([
            1 if code == self.TRANSFER_JOB_CODE
            else 2 if code in departure_codes
            else 0
            for code in self.job_code_list
        ], dtype=np.int64)
        agent_service_progress = np.zeros((n_agents,), dtype=np.float32)
        agent_departure_phase = np.full((n_agents,), -1, dtype=np.int64)
        agent_departure_ready_age = np.zeros((n_agents,), dtype=np.float32)
        service_job_codes = set(self.service_job_code_list)
        for global_pid, plane in active_planes.items():
            service_total = max(1, len(service_job_codes))
            service_finished = len(
                service_job_codes.intersection(set(plane.finished_jobs))
            )
            agent_service_progress[global_pid] = min(
                1.0, float(service_finished) / float(service_total)
            )
            if not plane.has_completed_service_jobs():
                departure_phase = 0
            elif plane.has_started_departure():
                departure_phase = 3
            elif plane.departure_staging_decided:
                departure_phase = 2
            else:
                departure_phase = 1
            agent_departure_phase[global_pid] = departure_phase
            if departure_phase in {1, 2}:
                agent_departure_ready_age[global_pid] = max(
                    0.0,
                    float(self.total_time)
                    - float(self.departure_ready_since.get(
                        plane.code, self.total_time
                    )),
                )
        data.job_phase_codes = torch.tensor(
            job_phase_codes, dtype=torch.long
        )
        data.transfer_op_mask = torch.tensor(
            job_phase_codes == 1, dtype=torch.bool
        )
        data.departure_op_mask = torch.tensor(
            job_phase_codes == 2, dtype=torch.bool
        )
        data.agent_service_progress = torch.tensor(
            agent_service_progress, dtype=torch.float32
        )
        data.agent_departure_phase = torch.tensor(
            agent_departure_phase, dtype=torch.long
        )
        data.agent_departure_ready_age = torch.tensor(
            agent_departure_ready_age, dtype=torch.float32
        )
        data.agent_legal_pair_counts = torch.tensor(
            agent_job_site_mask_matrix.reshape(n_agents, -1).sum(axis=1),
            dtype=torch.long,
        )
        potential_value = 0.0
        if (
            self.hindsight_reward_mode == 'team_time_potential'
            and self.iga_potential_beta > 0.0
        ):
            potential_value = self._iga_potential_value(
                self.get_iga_potential_features()
            )
        elif (
            self.hindsight_reward_mode == 'team_time_resource_potential'
            and self.iga_potential_beta > 0.0
        ):
            potential_value = self._resource_slack_potential_value()
        elif (
            self.hindsight_reward_mode
            == 'team_time_resource_fitted_potential'
            and self.iga_potential_beta > 0.0
        ):
            potential_value = self._resource_fitted_potential_value()
        data.iga_potential_value = torch.tensor(
            [potential_value], dtype=torch.float32
        )

        # ==========================================
        # 3B. 构建移动资源请求节点 (Request)
        # 第 0 个请求固定为 no-op，真实请求从 1 开始。
        # ==========================================
        request_features = []
        request_type_vocab = sorted(dev_types)
        for req in self.request_list:
            site = self.sites.get(req['site_code'], self.sites.get('Z'))
            req_type = req['needed_res_types'][0] if req['needed_res_types'] else ''
            req_type_idx = float(request_type_vocab.index(req_type) + 1) if req_type in request_type_vocab else 0.0
            site_idx = float(site2idx.get(site.code, 0))
            job_idx = float(self.job_code_list.index(req['job_code']) + 1) if req['job_code'] in self.job_code_list else 0.0
            plane_idx = float(req.get('plane_idx', -1))
            # Preserve the checkpoint-compatible eight request features while
            # making the decision horizon observable: blocking requests carry
            # non-negative elapsed waiting time; lookahead requests carry the
            # negative lower-bound time until the selected job may need them.
            request_time = (
                -float(req.get('lead_time', 0.0))
                if req.get('is_lookahead', False)
                else float(req.get('waiting_time', 0.0))
            )
            request_features.append([
                req_type_idx,
                request_time,
                float(site.pos[0]),
                float(site.pos[1]),
                site_idx,
                job_idx,
                plane_idx,
                1.0 if req.get('is_noop', False) else 0.0,
            ])
        while len(request_features) < self.max_request_num:
            request_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0])
        data['request'].x = torch.tensor(request_features, dtype=torch.float32)
        data.request_is_lookahead = torch.tensor(
            [
                bool(request.get('is_lookahead', False))
                for request in self.request_list
            ]
            + [False] * (self.max_request_num - len(self.request_list)),
            dtype=torch.bool,
        )
        # Request-level supervision gathers the exact originating operation
        # and target-site embeddings after the shared GNN forward.  These are
        # local (per-graph) indices on purpose: PyG concatenates the fixed-size
        # metadata without applying node offsets, and the actor reshapes it
        # back to [batch, request].
        request_operation_indices = []
        request_site_indices = []
        request_dag_lead_times = []
        request_prediction_valid = []
        request_kind_ids = []
        request_ready_context_values = []
        ready_kind_ids = {
            'bounded_mobile_frontier': 1,
            'bounded_mobile_frontier_h1': 1,
            'bounded_mobile_frontier_h2': 2,
            'blocking_wait': 3,
            'departure_pickup': 4,
        }
        for request in self.request_list:
            plane_idx = int(request.get('plane_idx', -1))
            job_code = request.get('job_code')
            site_code = request.get('site_code')
            valid = bool(
                not request.get('is_noop', False)
                and 0 <= plane_idx < n_plane_agents
                and job_code in self.job_code_list
                and site_code in site2idx
            )
            request_operation_indices.append(
                plane_idx * n_ops + self.job_code_list.index(job_code)
                if valid else -1
            )
            request_site_indices.append(
                int(site2idx[site_code]) if valid else -1
            )
            request_dag_lead_times.append(
                max(0.0, float(request.get('lead_time', 0.0)))
                if valid else 0.0
            )
            request_prediction_valid.append(valid)
            request_kind = str(request.get('request_kind', ''))
            kind_id = int(ready_kind_ids.get(request_kind, 0))
            request_kind_ids.append(kind_id if valid else 0)

            plane = active_planes.get(plane_idx) if valid else None
            job = (
                plane.jobs.get(job_code)
                if plane is not None and job_code in plane.jobs
                else self.jobs.get(job_code)
                if valid else None
            )
            predecessors = tuple(getattr(job, 'predecessor', ()) or ())
            finished = set(getattr(plane, 'finished_jobs', ()) or ())
            unfinished = [
                code for code in predecessors if code not in finished
            ]
            predecessor_durations = []
            if plane is not None:
                for code in unfinished:
                    predecessor = plane.jobs.get(code, self.jobs.get(code))
                    predecessor_durations.append(max(
                        0.0, float(getattr(predecessor, 'time', 0.0) or 0.0)
                    ))
            depth = int(request.get('dependency_depth', 0) or 0)
            if depth <= 0:
                depth = 2 if kind_id == 2 else 1 if kind_id in {1, 4} else 0
            request_ready_context_values.append([
                float(depth),
                float(len(unfinished)),
                float(sum(predecessor_durations)),
                float(max(predecessor_durations, default=0.0)),
                max(0.0, float(getattr(job, 'time', 0.0) or 0.0)),
                max(0.0, float(self.total_time)),
            ])
        pad_count = self.max_request_num - len(self.request_list)
        request_operation_indices.extend([-1] * pad_count)
        request_site_indices.extend([-1] * pad_count)
        request_dag_lead_times.extend([0.0] * pad_count)
        request_prediction_valid.extend([False] * pad_count)
        request_kind_ids.extend([0] * pad_count)
        request_ready_context_values.extend(
            [[0.0] * 6 for _ in range(pad_count)]
        )
        data.request_operation_indices = torch.tensor(
            request_operation_indices, dtype=torch.long
        )
        data.request_site_indices = torch.tensor(
            request_site_indices, dtype=torch.long
        )
        data.request_dag_lead_times = torch.tensor(
            request_dag_lead_times, dtype=torch.float32
        )
        data.request_prediction_valid = torch.tensor(
            request_prediction_valid, dtype=torch.bool
        )
        data.request_kind_ids = torch.tensor(
            request_kind_ids, dtype=torch.long
        )
        data.request_ready_context_values = torch.tensor(
            request_ready_context_values, dtype=torch.float32
        ).view(self.max_request_num, 6)
        data.global_features = torch.tensor(
            self._build_global_features(),
            dtype=torch.float32,
        ).view(1, -1)

        request_mask_matrix = np.zeros((n_agents, self.max_request_num), dtype=bool)
        for agent_idx in range(n_plane_agents):
            request_mask_matrix[agent_idx, 0] = True
        for dev_idx, dev in enumerate(device_list[:self.max_device_num]):
            agent_idx = n_plane_agents + dev_idx
            request_mask_matrix[agent_idx, 0] = True
            if self._device_is_dispatchable(dev):
                for req in self.request_list[1:]:
                    request_mask_matrix[agent_idx, req['id']] = (
                        self._device_can_dispatch(dev, req)
                    )
        for agent_idx in range(n_plane_agents + min(len(device_list), self.max_device_num), n_agents):
            request_mask_matrix[agent_idx, 0] = True

        # ==========================================
        # 4. 构建边索引与边特征 (Edges)
        # ==========================================
        edge_precedes = [[], []]   
        edge_os = [[], []]         
        attr_os = []
        edge_or = [[], []]         
        attr_or = []
        edge_dr = [[], []]
        attr_dr = []
        
        # 只为在场的活跃飞机连边，Dummy 节点作为孤岛存在即可
        for global_pid, plane in active_planes.items():
            for j_idx, j_code in enumerate(self.job_code_list):
                u_idx = global_pid * n_ops + j_idx
                job_obj = plane.jobs.get(j_code, self.jobs[j_code])
                
                # --- A. Precedes ---
                for pred_code in sorted(job_obj.predecessor):
                    if pred_code in self.job_code_list:
                        pred_j_idx = self.job_code_list.index(pred_code)
                        pred_u_idx = global_pid * n_ops + pred_j_idx
                        edge_precedes[0].append(pred_u_idx)
                        edge_precedes[1].append(u_idx)
                        
                # --- B. O-S Edge ---
                # Connect only sites that can eventually execute this job;
                # false compatibility edges previously polluted every op.
                for s_idx, is_valid in enumerate(
                    agent_job_site_mask_matrix[global_pid, j_idx]
                    & ptr_site_mask_matrix[global_pid]
                ):
                    if is_valid:
                        site = site_list[s_idx]
                        edge_os[0].append(u_idx)
                        edge_os[1].append(s_idx)
                        dist = abs(plane.site.pos[0] - site.pos[0]) + abs(plane.site.pos[1] - site.pos[1])
                        attr_os.append([min(dist / plane.velocity / 3600.0, 10.0)])
                
                # --- C. O-R Edge ---
                needed_dev_types = sorted(
                    set(job_obj.resources).intersection(dev_types)
                )
                if (
                    self.departure_job_code_list
                    and j_code == self.departure_job_code_list[0]
                    and self.TRANSPORTER_RESOURCE_TYPE in dev_types
                ):
                    needed_dev_types = [self.TRANSPORTER_RESOURCE_TYPE]
                reserved_departure_device = (
                    self._departure_transporter_for_plane(plane)
                    if (
                        self.departure_job_code_list
                        and j_code == self.departure_job_code_list[0]
                    ) else None
                )
                    
                for dev_type in needed_dev_types:
                    for dev in self.mobile_devices.get(dev_type, []):
                        if (
                            reserved_departure_device is not None
                            and dev_type == self.TRANSPORTER_RESOURCE_TYPE
                            and dev != reserved_departure_device
                        ):
                            continue
                        # Busy devices remain visible; their remaining delay is
                        # included in the ETA instead of deleting the edge.
                        v_idx = device2idx[dev.code]
                        edge_or[0].append(u_idx)
                        edge_or[1].append(v_idx)
                        valid_target_sites = [
                            site_list[site_idx]
                            for site_idx, valid in enumerate(
                                agent_job_site_mask_matrix[global_pid, j_idx]
                            )
                            if valid
                        ]
                        target_travel = min(
                            (
                                abs(dev.site.pos[0] - site.pos[0])
                                + abs(dev.site.pos[1] - site.pos[1])
                            ) / max(float(dev.velocity), 1.0)
                            for site in valid_target_sites
                        ) if valid_target_sites else 36000.0
                        availability_delay = float(max(
                            dev.left_trans_time, dev.left_rec_time
                        ))
                        attr_or.append([
                            min(
                                (availability_delay + target_travel) / 3600.0,
                                10.0,
                            )
                        ])

        for dev_idx, dev in enumerate(device_list):
            for req in self.request_list[1:]:
                if not self._device_can_serve(dev, req):
                    continue
                site = self.sites[req['site_code']]
                edge_dr[0].append(dev_idx)
                edge_dr[1].append(req['id'])
                dist = abs(dev.site.pos[0] - site.pos[0]) + abs(dev.site.pos[1] - site.pos[1])
                attr_dr.append([dist / dev.velocity])

        # ==========================================
        # 5. 赋值到 PyG 数据对象 (携带空边防崩保护)
        # ==========================================
        if len(edge_precedes[0]) > 0:
            data['operation', 'precedes', 'operation'].edge_index = torch.tensor(edge_precedes, dtype=torch.long)
        else:
            data['operation', 'precedes', 'operation'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        if len(edge_os[0]) > 0:
            data['operation', 'assignable_to', 'site'].edge_index = torch.tensor(edge_os, dtype=torch.long)
            data['operation', 'assignable_to', 'site'].edge_attr = torch.tensor(attr_os, dtype=torch.float32)
        else:
            data['operation', 'assignable_to', 'site'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['operation', 'assignable_to', 'site'].edge_attr = torch.empty((0, 1), dtype=torch.float32)
            
        if len(edge_or[0]) > 0:
            data['operation', 'needs', 'device'].edge_index = torch.tensor(edge_or, dtype=torch.long)
            data['operation', 'needs', 'device'].edge_attr = torch.tensor(attr_or, dtype=torch.float32)
        else:
            data['operation', 'needs', 'device'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['operation', 'needs', 'device'].edge_attr = torch.empty((0, 1), dtype=torch.float32)

        if len(edge_dr[0]) > 0:
            data['device', 'can_serve', 'request'].edge_index = torch.tensor(edge_dr, dtype=torch.long)
            data['device', 'can_serve', 'request'].edge_attr = torch.tensor(attr_dr, dtype=torch.float32)
        else:
            data['device', 'can_serve', 'request'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['device', 'can_serve', 'request'].edge_attr = torch.empty((0, 1), dtype=torch.float32)
            
        # 直接挂载 Numpy 生成的绝对固定形状的 Tensor
        data.op_mask = torch.tensor(agent_op_mask, dtype=torch.bool)             # Shape: [n_agents, n_ops]
        data.site_mask_matrix = torch.tensor(ptr_site_mask_matrix, dtype=torch.bool) # Shape: [n_agents, n_sites]
        data.job_site_mask_matrix = torch.tensor(job_site_mask_matrix, dtype=torch.bool)
        data.agent_job_site_mask_matrix = torch.tensor(
            agent_job_site_mask_matrix,
            dtype=torch.bool,
        )
        data.request_mask_matrix = torch.tensor(request_mask_matrix, dtype=torch.bool)
        agent_types = self._build_agent_types()
        data.agent_types = torch.tensor(agent_types, dtype=torch.long)
        ordinary_type_vocab = sorted(
            resource_type for resource_type in dev_types
            if resource_type != self.TRANSPORTER_RESOURCE_TYPE
        )
        ordinary_type_to_id = {
            resource_type: index
            for index, resource_type in enumerate(ordinary_type_vocab)
        }
        data.device_type_ids = torch.tensor([
            ordinary_type_to_id.get(dev.resource.type, -1)
            for dev in device_list
        ], dtype=torch.long)
        data.ordinary_device_type_count = torch.tensor(
            [len(ordinary_type_vocab)], dtype=torch.long
        )
        self.ptr_site_mask_matrix = ptr_site_mask_matrix
        self.job_site_mask_matrix = job_site_mask_matrix
        self.agent_job_site_mask_matrix = agent_job_site_mask_matrix
        self.agent_op_mask = agent_op_mask
        self.request_mask_matrix = request_mask_matrix
        self.agent_types = agent_types
        if self.config.get('stage2_resource_v6_observations', False):
            from onpolicy.utils.stage2_resource_v6_observation import attach_resource_view
            attach_resource_view(self, data, coordinate_scale)
            
        return data

    def _get_reward(self):
        """
        计算每个智能体的即时奖励。
        【重构核心】：
        1. 引入 Reward Shaping（阶段性奖励与离场奖励），打破稀疏惩罚，引导 Actor 走出随机探索的深渊。
        2. 修复 SMDP 奖励分配漏洞：每一步流逝的 dt 产生的奖惩，必须分发给所有在场飞机。
        """
        dt = self.step_time 
        
        # 如果没有时间流逝，直接返回 0
        if dt <= 0:
            return np.zeros((self.n_agents, 1), dtype=np.float32)

        # 1. 计算全局时间惩罚 (Global Penalty)
        # 仍在场上的飞机越多，每秒扣分越狠，倒逼模型学会并行调度以缩短 Total Flow Time
        # active_plane_count = sum(1 for p in self.planes.values() if not p.is_completed_all_jobs())
        global_time_penalty = - dt / 100.0

        rewards = np.zeros((self.n_agents, 1), dtype=np.float32)

        reward_planes = list(self.planes.items()) + [
            (plane.code, plane)
            for plane in self._departed_this_step.values()
            if plane.code not in self.planes
        ]
        for plane_id, plane in reward_planes:
            pid = int(plane_id.split('_')[-1])
            
            # --- 动态初始化追踪属性 (无需去 Plane 类里改底座代码) ---
            if not hasattr(plane, '_last_rewarded_job_count'):
                plane._last_rewarded_job_count = len(plane.ever_finished_jobs)
            if not hasattr(plane, '_has_received_completion_bonus'):
                plane._has_received_completion_bonus = False
            
            # 基础奖励：无论是否在做决策，所有在场飞机都要承担时间流逝的惩罚
            agent_reward = global_time_penalty

            # --- A. 进度激励 (Dense Reward) ---
            # 检查在刚才流逝的 dt 时间内，该飞机是否完成了新的保障工序
            current_finished_count = len(plane.ever_finished_jobs)
            newly_finished = current_finished_count - plane._last_rewarded_job_count
            
            if newly_finished > 0:
                # 每完成一个保障工序，给予正向反馈，引导模型 "多干活"
                agent_reward += newly_finished * 5.0
                plane._last_rewarded_job_count = current_finished_count

            # --- B. 终局奖励 (Terminal Reward) ---
            # 如果刚刚完成了所有任务（准备起飞或已经起飞离场）
            if (
                plane.has_completed_departure()
                and not plane._has_received_completion_bonus
            ):
                # 给予巨大的通关奖励，这是策略网络后期收敛的核心动力
                agent_reward += 50.0
                plane._has_received_completion_bonus = True
            
            # 赋值：不再用 current_active_agents 屏蔽，保障休眠期的连续奖励传递
            rewards[pid, 0] = agent_reward

        if self.resource_policy == 'drl':
            for dev_idx, dev in enumerate(self.device_list[:self.max_device_num]):
                agent_id = self.n_plane_agents + dev_idx
                if agent_id < self.n_agents and not dev.is_idle():
                    if self._is_transporter_device(dev):
                        rewards[agent_id, 0] = 1.5 * global_time_penalty
                    else:
                        rewards[agent_id, 0] = global_time_penalty
            
        return rewards
    
    def _get_done(self):
        """
        获取每个智能体在当前回合是否结束 (Episode Termination)。

        Output:
        - agent_dones: List[bool] - 当前仍在场上的每架飞机是否已完成自身所有任务
        """
        if getattr(self, 'done', False):
            return np.ones(self.n_agents, dtype=bool)

        agent_dones = [False] * self.n_agents
        for pid in self.departed_agent_ids:
            if pid < self.n_plane_agents:
                agent_dones[pid] = True
        
        # 遍历当前仍在环境中的所有飞机
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            # 单架飞机结束的标志：剩余待办作业列表为空
            if pid < self.n_plane_agents:
                agent_dones[pid] = plane.is_completed_all_jobs()
            
        return np.array(agent_dones)

    def _get_info(self):
        """
        获取环境的额外信息，包括用于 GRU 时序记忆的上一次动作索引。
        """
        self._refresh_request_pool()
        active_agents = [False] * self.n_agents
        for plane in self.planes.values():
            # 活跃条件 1：飞机处于空闲状态（不忙碌、不在运输、不在等待）
            # 活跃条件 2：飞机还有未完成的作业
            pid = int(plane.code.split('_')[-1])
            if pid < self.n_plane_agents and self._plane_can_decide(plane):
                active_agents[pid] = True

        if self.resource_policy == 'drl':
            for dev_idx, dev in enumerate(self.device_list[:self.max_device_num]):
                agent_id = self.n_plane_agents + dev_idx
                if agent_id >= self.n_agents or not self._device_is_dispatchable(dev):
                    continue
                has_real_request = any(self._device_can_dispatch(dev, req) for req in self.request_list[1:])
                if has_real_request:
                    active_agents[agent_id] = True

        self._check_repeated_device_decision_state(active_agents)

        # 准备全局固定顺序的机位列表 (与 _get_obs 中的 site_list 顺序保持绝对一致)
        site_codes = list(self.sites.keys())
        
        # 初始化记录数组，-1 表示没有历史记录 (例如 Episode 刚开始)
        last_site_indices = [-1] * self.n_agents
        last_op_indices = [-1] * self.n_agents
        
        # 遍历当前场上的飞机
        
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            
            # ---------------------------------------------------------
            # A. 计算上一次机位的全局索引
            # ---------------------------------------------------------
            if pid >= self.n_plane_agents:
                continue
            last_site_indices[pid] = plane.last_site_idx
                
            # ---------------------------------------------------------
            # B. 计算上一次工序的全局索引
            # ---------------------------------------------------------
            last_op_indices[pid] = plane.last_job_idx
                    
        resource_metrics = (
            self.get_resource_lateness_metrics()
            if self.done else {
                'total_wait_seconds': 0.0,
                'max_wait_seconds_per_aircraft': 0.0,
                'p95_wait_seconds_per_aircraft': 0.0,
                'critical_wait_seconds': 0.0,
                'critical_wait_p95_seconds_per_aircraft': 0.0,
                'critical_avoidable_lateness_seconds': 0.0,
                'critical_avoidable_lateness_p95_seconds_per_aircraft': 0.0,
                'rendezvous_spread_seconds': 0.0,
                'rendezvous_spread_p95_seconds': 0.0,
                'predicted_lateness_seconds': 0.0,
                'predicted_lateness_p95_seconds': 0.0,
                'predicted_lateness_max_seconds': 0.0,
                'late_dispatch_count': 0,
                'late_dispatch_rate': 0.0,
                'early_arrival_seconds': 0.0,
                'lookahead_dispatch_count': 0,
                'visibility_to_legal_seconds': 0.0,
                'legal_to_idle_seconds': 0.0,
                'policy_defer_seconds': 0.0,
                'policy_defer_p95_seconds': 0.0,
                'policy_defer_max_seconds': 0.0,
                'policy_defer_count': 0,
                'policy_defer_rate': 0.0,
            }
        )
        wait_decomposition = resource_metrics.get('wait_decomposition', {})
        return {
            'active_agents': np.array(active_agents), 
            'last_site_indices': np.array(last_site_indices, dtype=np.int32),
            'last_op_indices': np.array(last_op_indices, dtype=np.int32),
            'agent_types': self._build_agent_types(),
            'case_id': np.array(getattr(self, 'current_case_path', ''), dtype=object),
            'env_steps': np.array(self.steps, dtype=np.int32),
            'env_total_time': np.array(self.total_time, dtype=np.float32),
            'active_device_debug': np.array(self._active_device_debug(), dtype=object),
            'cycle_terminated': np.array(self.cycle_terminated, dtype=bool),
            'cycle_agent_mask': np.array(
                [idx in self.cycle_agent_ids for idx in range(self.n_agents)],
                dtype=bool,
            ),
            'cycle_reason': np.array(self.cycle_reason, dtype=object),
            'max_no_progress': np.array(
                max(
                    (plane.no_progress_decisions for plane in self.planes.values()),
                    default=0,
                ),
                dtype=np.int32,
            ),
            'total_relocations': np.array(
                self.departed_total_relocations + sum(
                    plane.total_relocations for plane in self.planes.values()
                ),
                dtype=np.int32,
            ),
            'departure_barrier_open': np.array(
                self._departure_barrier_is_open(), dtype=bool
            ),
            'departure_mode': np.array(
                'progressive_per_aircraft', dtype=object
            ),
            'departure_ready_plane_count': np.array(
                sum(
                    self._plane_awaiting_departure(plane)
                    for plane in self.planes.values()
                ),
                dtype=np.int32,
            ),
            'departure_reserved_r014_count': np.array(
                len(self.departure_transporter_by_plane), dtype=np.int32
            ),
            'departed_plane_count': np.array(
                len(self.departed_agent_ids), dtype=np.int32
            ),
            'resource_wait_seconds': np.array(
                resource_metrics['total_wait_seconds'], dtype=np.float32
            ),
            'resource_critical_wait_seconds': np.array(
                resource_metrics['max_wait_seconds_per_aircraft'],
                dtype=np.float32,
            ),
            'resource_wait_p95_seconds': np.array(
                resource_metrics['p95_wait_seconds_per_aircraft'],
                dtype=np.float32,
            ),
            'resource_slack_weighted_wait_seconds': np.array(
                resource_metrics['critical_wait_seconds'], dtype=np.float32
            ),
            'resource_slack_weighted_wait_p95_seconds': np.array(
                resource_metrics[
                    'critical_wait_p95_seconds_per_aircraft'
                ],
                dtype=np.float32,
            ),
            'resource_avoidable_critical_lateness_seconds': np.array(
                resource_metrics[
                    'critical_avoidable_lateness_seconds'
                ],
                dtype=np.float32,
            ),
            'resource_avoidable_critical_lateness_p95_seconds': np.array(
                resource_metrics[
                    'critical_avoidable_lateness_p95_seconds_per_aircraft'
                ],
                dtype=np.float32,
            ),
            'resource_rendezvous_spread_seconds': np.array(
                resource_metrics['rendezvous_spread_seconds'],
                dtype=np.float32,
            ),
            'resource_rendezvous_spread_p95_seconds': np.array(
                resource_metrics['rendezvous_spread_p95_seconds'],
                dtype=np.float32,
            ),
            'resource_predicted_lateness_seconds': np.array(
                resource_metrics['predicted_lateness_seconds'],
                dtype=np.float32,
            ),
            'resource_predicted_lateness_p95_seconds': np.array(
                resource_metrics['predicted_lateness_p95_seconds'],
                dtype=np.float32,
            ),
            'resource_late_dispatch_count': np.array(
                resource_metrics['late_dispatch_count'], dtype=np.int32
            ),
            'resource_late_dispatch_rate': np.array(
                resource_metrics['late_dispatch_rate'], dtype=np.float32
            ),
            'resource_early_arrival_seconds': np.array(
                resource_metrics['early_arrival_seconds'], dtype=np.float32
            ),
            'resource_wait_before_dispatch_seconds': np.array(
                wait_decomposition.get(
                    'waiting_before_dispatch_seconds', 0.0
                ),
                dtype=np.float32,
            ),
            'resource_wait_travel_seconds': np.array(
                wait_decomposition.get(
                    'travel_after_dispatch_seconds', 0.0
                ),
                dtype=np.float32,
            ),
            'resource_wait_post_arrival_seconds': np.array(
                wait_decomposition.get(
                    'post_arrival_synchronization_seconds', 0.0
                ),
                dtype=np.float32,
            ),
            'resource_lookahead_dispatch_count': np.array(
                resource_metrics['lookahead_dispatch_count'], dtype=np.int32
            ),
            'resource_visibility_to_legal_seconds': np.array(
                resource_metrics['visibility_to_legal_seconds'],
                dtype=np.float32,
            ),
            'resource_legal_to_idle_seconds': np.array(
                resource_metrics['legal_to_idle_seconds'], dtype=np.float32
            ),
            'resource_policy_defer_seconds': np.array(
                resource_metrics['policy_defer_seconds'], dtype=np.float32
            ),
            'resource_policy_defer_p95_seconds': np.array(
                resource_metrics['policy_defer_p95_seconds'],
                dtype=np.float32,
            ),
            'resource_policy_defer_max_seconds': np.array(
                resource_metrics['policy_defer_max_seconds'],
                dtype=np.float32,
            ),
            'resource_policy_defer_count': np.array(
                resource_metrics['policy_defer_count'], dtype=np.int32
            ),
            'resource_policy_defer_rate': np.array(
                resource_metrics['policy_defer_rate'], dtype=np.float32
            ),
        }
    
    def step(self, action):
        '''执行一步纯事件驱动动作，并在内部自动快进时间直到出现可决策状态'''
        self._departed_this_step = {}
        action = np.asarray(action)
        if action.ndim != 2 or action.shape[0] < self.n_agents or action.shape[1] < 2:
            raise ValueError(
                f"Expected action shape [at least {self.n_agents}, 2], got {action.shape}."
            )

        # Plane actions belong to the observation that preceded this call.
        # A zero-distance device dispatch below may make a newly supplied
        # aircraft ready immediately; that aircraft must wait for the next
        # observation instead of consuming an action that was masked before
        # the dispatch.
        plane_active_at_step_start = [False] * self.n_plane_agents
        n_ops = len(self.job_code_list)
        if self.agent_op_mask is not None:
            for pid in range(self.n_plane_agents):
                plane_active_at_step_start[pid] = bool(
                    self.agent_op_mask[
                        pid, pid * n_ops:(pid + 1) * n_ops
                    ].any()
                )

        # Device actions were selected from the pre-step request/device state.
        # Execute their validated joint plan before a plane action can start a
        # local job and consume one of those devices. Requests created by the
        # plane actions below are intentionally handled at the next decision.
        if self.resource_policy == 'drl':
            self._dispatch_device_actions(action)

        # =====================================================================
        # Phase 1A: 动作分配 (Agents' Turn) —— 每次 step 只执行一次
        # =====================================================================
        active_agents = [False] * self.n_agents
        for plane in self.planes.values():
            pid = int(plane.code.split('_')[-1])
            if pid < self.n_plane_agents:
                active_agents[pid] = plane_active_at_step_start[pid]
        self.current_active_agents = active_agents
        potential_before = (
            self.get_iga_potential_features()
            if self.hindsight_reward_mode == 'iga_potential' else None
        )
        potential_actions = []
        
        claimed_site_indices = set()
        plane_items = sorted(
            self.planes.items(),
            key=lambda item: int(item[0].split('_')[-1]),
        )
        for plane_id, plane in plane_items:
            pid = int(plane_id.split('_')[2])
            if pid < self.n_plane_agents and self.current_active_agents[pid]:
                # 解析 RL 动作
                op_global_idx, site_idx = self._validate_plane_action(
                    pid,
                    plane,
                    action[pid][0],
                    action[pid][1],
                    claimed_site_indices,
                )
                potential_actions.append({
                    'step_idx': int(self.steps),
                    'agent_id': int(pid),
                    'action': [int(op_global_idx), int(site_idx)],
                })
                job_idx = op_global_idx - pid * len(self.job_code_list)
                target_job_code = self.job_code_list[job_idx]
                target_site_code = self.site_code_list[site_idx]
                target_job = self.jobs[target_job_code]
                if target_job_code == self.TRANSFER_JOB_CODE:
                    action_phase = 'post_service_relocation'
                elif target_job_code in self.departure_job_code_list:
                    action_phase = 'departure'
                else:
                    action_phase = 'service'

                plane.last_site_idx = site_idx
                plane.last_job_idx = job_idx

                self.pending_actions[plane_id] = {
                    'step_idx': self.steps,
                    'agent_id': pid,
                    'action': [int(op_global_idx), int(site_idx)],
                    'start_time': self.total_time,
                    'plane_id': plane_id,
                    'site_id': target_site_code,
                    'origin_site_code': plane.site.code,
                    'target_site_code': target_site_code,
                    'target_job_code': target_job_code,
                    'action_phase': action_phase,
                    'departure_mode': 'progressive_per_aircraft',
                    'departure_barrier_open': bool(
                        self._departure_barrier_is_open()
                    ),
                    'departure_eligible': bool(
                        self._plane_departure_eligible(plane)
                    ),
                    'departure_ready_time': self.departure_ready_since.get(
                        plane.code
                    ),
                    'irreversible_progress_before': plane.irreversible_progress_count,
                    'ever_finished_before': len(plane.ever_finished_jobs),
                    'relocations_before': plane.total_relocations,
                    'previous_departed_site_code': plane.last_departed_site_code,
                    'device_ids': [] # 初始化为空，如果用到设备会在后续追加
                }

                # 动作执行逻辑
                is_staging = target_job_code == self.TRANSFER_JOB_CODE
                is_departure_transport = (
                    bool(self.departure_job_code_list)
                    and target_job_code == self.departure_job_code_list[0]
                )
                transport_purpose = (
                    'post_service_relocation'
                    if is_staging else
                    'departure'
                    if is_departure_transport else
                    'service_relocation'
                )
                if is_staging:
                    plane.mark_departure_staging_decided()

                if is_departure_transport:
                    transporter = self._departure_transporter_for_plane(
                        plane, ready_only=True
                    )
                    if transporter is None:
                        raise RuntimeError(
                            f"Plane {plane.code} cannot depart before its "
                            "reserved R014 reaches the pickup stand."
                        )
                    plane.choosed_job = target_job_code
                    plane.start_transport(
                        self.sites[target_site_code],
                        transporter,
                        purpose='departure',
                    )
                    self._release_departure_transporter(
                        plane_code=plane.code,
                        device_code=transporter.code,
                    )
                    self.pending_actions[plane_id]['device_ids'].append(
                        transporter.code
                    )
                    self._record_intrinsic_ready_time(
                        plane.code,
                        self.TRANSFER_JOB_CODE,
                        self.pending_actions[plane_id]['origin_site_code'],
                        self.total_time,
                    )
                elif target_site_code != plane.site.code:
                    plane.choosed_job = (
                        None if is_staging else target_job_code
                    )
                    if self.resource_policy == 'drl':
                        if plane.site.code == 'Z':
                            if is_staging or is_departure_transport:
                                raise RuntimeError(
                                    f"Plane {plane.code} cannot stage/depart from Z."
                                )
                            plane.start_transport(
                                self.sites[target_site_code],
                                None,
                                purpose=transport_purpose,
                            )
                        else:
                            self._record_intrinsic_ready_time(
                                plane.code,
                                self.TRANSFER_JOB_CODE,
                                plane.site.code,
                                self.total_time,
                            )
                            plane.start_waiting(
                                self.TRANSFER_JOB_CODE,
                                transport_purpose=transport_purpose,
                            )
                            plane.destination = self.sites[target_site_code]
                            plane.destination.add_plane(plane)
                    else:
                        if plane.site.get_avail_transporter() is None:
                            if plane.site.code == 'Z':
                                if is_staging or is_departure_transport:
                                    raise RuntimeError(
                                        f"Plane {plane.code} cannot stage/depart from Z."
                                    )
                                plane.start_transport(
                                    self.sites[target_site_code],
                                    None,
                                    purpose=transport_purpose,
                                )
                            else:
                                self._record_intrinsic_ready_time(
                                    plane.code,
                                    self.TRANSFER_JOB_CODE,
                                    plane.site.code,
                                    self.total_time,
                                )
                                plane.start_waiting(
                                    self.TRANSFER_JOB_CODE,
                                    transport_purpose=transport_purpose,
                                )
                                plane.destination = self.sites[target_site_code]
                                plane.destination.add_plane(plane)
                        else:
                            transporter = plane.site.get_avail_transporter()
                            self._record_intrinsic_ready_time(
                                plane.code,
                                self.TRANSFER_JOB_CODE,
                                plane.site.code,
                                self.total_time,
                            )
                            plane.start_transport(
                                self.sites[target_site_code],
                                transporter,
                                purpose=transport_purpose,
                            )
                            if transporter and plane_id in self.pending_actions:
                                self.pending_actions[plane_id]['device_ids'].append(transporter.code)
                else:
                    if is_staging:
                        plane.trans_time = 0
                        plane.job_time = 0
                        plane.total_job_time = 0
                        self.pending_actions[plane_id][
                            'decision_type'
                        ] = 'hold_at_current_stand'
                    elif target_job_code not in plane.get_avail_jobs(plane.site):
                        self._record_intrinsic_ready_time(
                            plane.code,
                            target_job_code,
                            plane.site.code,
                            self.total_time,
                        )
                        plane.start_waiting(target_job_code)
                        plane.choosed_job = target_job_code
                        plane.trans_time = 0
                    else:
                        plane.choose_job(target_job_code)
                        self._record_started_plane_jobs(
                            plane, self.total_time
                        )
                        plane.trans_time = 0

            if plane.is_waiting:
                if plane.site.code not in self.waiting_sites[plane.pending_job]:
                    self.waiting_sites[plane.pending_job].append(plane.site.code)

        # A ZY-T/current-site choice is an explicit parking decision with zero
        # duration. Close it before fast-forwarding to an unrelated event.
        for plane_id, plane in list(self.planes.items()):
            if plane_id in self.pending_actions and plane.is_idle():
                self._complete_pending_plane_action(plane_id, plane)

        # =====================================================================
        # 核心重构：内部事件推演循环 (Fast-Forward Loop)
        # =====================================================================
        time_prev = self.total_time
        while True:
            if self.resource_policy != 'drl':
                self._sync_waiting_sites()
                self._heuristic_dispatch_departure_pickups()

            # Request refresh is not a read-only operation: it may promote an
            # arrived R014 lookahead lease into the physical departure
            # handshake, or synchronously start a job whose resource is
            # already local.  Settle those zero-time transitions *before*
            # testing which agents can decide.  Computing the runway plan
            # first leaves a stale empty plan and can misclassify a newly
            # enabled ZY-S action as an event deadlock.
            if self.resource_policy == 'drl':
                self._refresh_request_pool()
            departure_plan = self._compute_departure_runway_plan()
            if any(
                self._plane_can_decide(
                    plane, departure_plan=departure_plan
                )
                for plane in self.planes.values()
            ):
                break

            if self.resource_policy == 'drl':
                has_new_device_decision = any(
                    self._device_can_dispatch(dev, request)
                    for dev in self.device_list[:self.max_device_num]
                    for request in self.request_list[1:]
                )
                if has_new_device_decision:
                    break

            internal_step_time = np.inf
            
            # -----------------------------------------------------------
            # Phase 1B: 环境设备自治调度 (NPCs' Turn)
            # -----------------------------------------------------------
            if self.resource_policy != 'drl':
                for job_code, waiting_sites_list in self.waiting_sites.items():
                    if not waiting_sites_list: continue
                    needed_res_types = ['R014'] if job_code == 'ZY-T' else sorted(res for res in self.jobs.get(job_code).resources if res in self.mobile_devices)
                    if not needed_res_types: continue

                    idle_devices = self.get_idle_devices(needed_res_types)
                    if idle_devices:
                        waiting_planes = [p for p in self.planes.values() if p.site.code in waiting_sites_list]
                        assignments = arrange_devices(idle_devices, waiting_planes)
                        for device, target_site in assignments:
                            device.start_transport(target_site)
                            if target_site.code in waiting_sites_list:
                                waiting_sites_list.remove(target_site.code)
                            for wp in waiting_planes:
                                if wp.site.code == target_site.code:
                                    if wp.code in self.pending_actions:
                                        self.pending_actions[wp.code]['device_ids'].append(device.code)
                                    break

            # -----------------------------------------------------------
            # Phase 1C: 统计所有忙碌实体的剩余预期时间
            # -----------------------------------------------------------
            for plane in self.planes.values():
                if plane.is_transporting:
                    internal_step_time = min(internal_step_time, plane.left_trans_time)
                elif plane.is_busy:
                    internal_step_time = min(internal_step_time, plane.site.left_job_time)
                    
            for devices in self.mobile_devices.values():
                for device in devices:
                    if not device.is_idle():
                        left_t = device.left_trans_time if device.is_transporting else device.left_rec_time
                        internal_step_time = min(internal_step_time, left_t)

            if self.resource_policy == 'drl':
                # A JIT window opening is a real scheduling event.  Without
                # this wake-up, the event loop could jump straight to job
                # completion and turn an avoidable dispatch into a late one.
                internal_step_time = min(
                    internal_step_time,
                    self._next_lookahead_dispatch_dt(),
                    self._next_lookahead_reservation_expiry_dt(),
                )
                        
            for site in self.sites.values():
                if site.is_interfered:
                    internal_step_time = min(internal_step_time, site.left_rec_time)

            # -----------------------------------------------------------
            # Phase 2: 时间跃迁 dt 计算 【已修改】
            # -----------------------------------------------------------
            # 检查 site 'Z' 是否被占用 (场上是否有飞机的当前位置是 'Z')
            z_occupied = any(p.site.code == 'Z' for p in self.planes.values())
            
            # 判断是否有已到期/超期且等待降落的飞机
            if self.landing_list and self.landing_list[0][0] <= self.total_time and not z_occupied:
                # 如果跑道空闲，且有飞机该降落了(甚至已经晚点了)，立刻将下一个跃迁时间设为0，去执行降落
                next_landing_dt = 0.0
            else:
                # 否则(跑道被占，或者当前没有急需降落的飞机)，只关注严格在未来的降落计划
                future_landings = [item[0] for item in self.landing_list if item[0] > self.total_time]
                next_landing_dt = future_landings[0] - self.total_time if future_landings else np.inf
            
            dt = min(internal_step_time, next_landing_dt)
            if dt == np.inf:
                if not self._is_schedule_complete():
                    raise RuntimeError(
                        "Environment reached a non-terminal state with no upcoming "
                        f"events. case_id={getattr(self, 'current_case_path', '')}, "
                        f"env_steps={self.steps}, env_total_time={self.total_time}, "
                        f"snapshot={self._deadlock_snapshot()}."
                    )
                break
                
            self.total_time += dt

            # -----------------------------------------------------------
            # Phase 3: 物理状态推演 (时间流逝)
            # -----------------------------------------------------------
            if dt >= 0:
                # 第一段：只结算干涉、正在运输（会到达目的地）和正在作业（会结束作业）的实体
                for site in self.sites.values():
                    if site.is_interfered: site.update(dt)
                        
                update_time_before = float(self.total_time) - float(dt)
                for plane in list(self.planes.values()):
                    if plane.is_busy or plane.is_transporting:
                        jobs_before = set(plane.current_jobs)
                        plane.update(dt)
                        if set(plane.current_jobs).difference(jobs_before):
                            self._record_started_plane_jobs(
                                plane, self.total_time
                            )

                for devices in self.mobile_devices.values():
                    for device in devices:
                        if not device.is_idle():
                            device.update(dt)

                # 第二段：结算所有“处于等待状态”的飞机
                # 此时，所有该腾空位置的飞机都已经完成了离开动作，绝不会发生旧飞机挡住新飞机的情况
                for plane in list(self.planes.values()):
                    if plane.is_waiting:
                        waiting_since = update_time_before - float(
                            plane.waiting_time
                        )
                        origin_site_code = plane.site.code
                        destination_before = plane.destination
                        jobs_before = set(plane.current_jobs)
                        plane.update(dt)
                        if (
                            destination_before is not None
                            and plane.is_transporting
                        ):
                            self._record_intrinsic_ready_time(
                                plane.code,
                                self.TRANSFER_JOB_CODE,
                                origin_site_code,
                                waiting_since,
                            )
                        if set(plane.current_jobs).difference(jobs_before):
                            self._record_started_plane_jobs(
                                plane, waiting_since
                            )

           # -----------------------------------------------------------
            # Phase 4: 触发外部事件 (处理到达预定时间的飞机降落) 【已修改】
            # -----------------------------------------------------------
            while self.landing_list and self.total_time >= self.landing_list[0][0]:
                if self.sites['Z'].is_occupied:
                    break 
                    
                # 执行降落
                land_time, bidx, pidx, _ = self.landing_list.pop(0) 
                
                plane_cfg = {
                    'velocity': 5,
                    'site': self.sites['Z'],
                    'fuel': (self.np_random.integers(0, 30) if hasattr(self, 'np_random') else np.random.randint(0, 30)) if getattr(self, 'use_domain_rand', True) else 30,
                    'jobs': self._build_plane_jobs(drop_optional=getattr(self, 'use_domain_rand', True))
                }
                self.add_planes([{'batch': bidx, 'idx': pidx, **plane_cfg}])

            for plane_id, plane in list(self.planes.items()):
                if plane_id in self.pending_actions:
                    # 动作完成的条件：飞机再次需要下发指令，或者已经彻底结束所有流程准备离场
                    if plane.is_idle() or plane.is_completed_all_jobs():
                        self._complete_pending_plane_action(plane_id, plane)

            # ZY-F completion is the physical departure event. Finalize its
            # trajectory first, then release the runway and remove the plane.
            self._remove_departed_planes()
                        
            # 【新增】：极端容错 - 拦截并结算可能被从环境中移除 (remove_planes) 的飞机日志
            for p_id in list(self.pending_actions.keys()):
                if p_id not in self.planes:
                    record = self.pending_actions.pop(p_id)
                    record.setdefault('trans_time', 0.0)
                    record.setdefault('job_time', 0.0)
                    record.setdefault('total_job_time', 0.0)
                    record['waiting_time'] = self.total_time - record['start_time'] - record['trans_time'] - record['job_time']
                    record['end_time'] = self.total_time
                    self.trajectory_log.append(record)

            for device_id, record in list(self.pending_device_actions.items()):
                device = next((d for d in self.device_list if d.code == device_id), None)
                if device is not None and device.is_idle():
                    record = self.pending_device_actions.pop(device_id)
                    record['end_time'] = self.total_time
                    record['duration'] = self.total_time - record['start_time']
                    self.device_trajectory_log.append(record)
                    intent_key = tuple(record.get('intent_identity', ()))
                    intent = self.resource_intent_ledger.get(intent_key)
                    if intent is not None:
                        intent['status'] = 'arrived'
                        intent['arrival_time'] = float(self.total_time)
                        intent['history'].append({
                            'status': 'arrived',
                            'time': float(self.total_time),
                        })
            
            # -----------------------------------------------------------
            # Phase 5: 检查是否可以退出快进循环
            # -----------------------------------------------------------
            self.done = self.cycle_terminated or self._is_schedule_complete()
            if self.done:
                break # 环境结束，跳出循环

            # As above, settle zero-time resource transitions before checking
            # plane availability.  Passing one shared plan also prevents
            # per-plane checks from observing different mutable snapshots.
            if self.resource_policy == 'drl':
                self._refresh_request_pool()
            departure_plan = self._compute_departure_runway_plan()

            # 检查场上是否出现了活跃的（可以做决策的）飞机
            has_active = False
            has_idle_site = False
            has_active_device = False
            for plane in self.planes.values():
                if self._plane_can_decide(
                    plane, departure_plan=departure_plan
                ):
                    has_active = True
                    break
            if self.resource_policy == 'drl':
                for dev in self.device_list[:self.max_device_num]:
                    if any(self._device_can_dispatch(dev, req) for req in self.request_list[1:]):
                        has_active_device = True
                        break
            for site in self.sites.values():
                if site.code not in self.runway_code_list and site.code != "Z":
                    if not site.is_interfered and not site.is_occupied:
                        has_idle_site = True
                        break
                    
            if has_active or has_active_device:
                break # 有飞机空闲了需要下发动作，跳出循环
        
        self.step_time = self.total_time - time_prev
        if potential_before is not None and potential_actions:
            self.potential_transition_log.append({
                'step_idx': int(self.steps),
                'time_before': float(time_prev),
                'time_after': float(self.total_time),
                'before': potential_before,
                'after': self.get_iga_potential_features(),
                'actions': potential_actions,
            })
        self.steps += 1
        # 最终返回观测和奖励
        return self._get_obs(), self._get_reward(), self._get_done(), self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # ==========================================================
        # 动态切换配置与加载数据
        # ==========================================================
        current_config = self.data_list[self.data_idx]
        self.config = current_config  # 更新 self.config 引用
        self.current_case_path = os.path.dirname(current_config.get('jobs_path', ''))
        
        # 实时 I/O 读取
        new_data = self._load_data_from_disk(current_config)
        
        # 应用数据 (无需 clone)
        self._apply_data(new_data, clone=False)
        
        # 循环索引，为下一次 reset 准备
        self.data_idx = (self.data_idx + 1) % len(self.data_list)
        
        # 解析域随机化开关 (优先级：options传入 > config配置 > 默认开启)
        self.steps = 0
        self.total_time = 0
        self.step_time = 0
        self.done = False
        self.trajectory_log = []
        self.potential_transition_log = []
        self.device_trajectory_log = []
        self.device_decision_log = []
        self.pending_actions = {}
        self.pending_device_actions = {}
        self.request_list = []
        self.request_pool = {}
        self.resource_intent_ledger = {}
        self._v6_dispatch_history = {}
        self._v6_request_snapshots = {}
        self.intrinsic_ready_time_occurrences = {}
        self._resource_lateness_metrics_cache = None
        self._deferred_lookahead_requests = set()
        self._lookahead_deferred_at_time = None
        self._last_device_decision_signature = None
        self._device_decision_repeat_count = 0
        self.cycle_terminated = False
        self.cycle_agent_ids = set()
        self.cycle_reason = ''
        self.departure_barrier_open = False
        self.departure_ready_since = {}
        self.departure_transporter_by_plane = {}
        self.departure_plane_by_transporter = {}
        self._departure_runway_plan = {}
        self.departed_agent_ids = set()
        self.departure_log = []
        self._departed_this_step = {}
        self.departed_total_relocations = 0
        
        self.planes.clear()
        self.num_planes = 0
        self.force_transfer_planes.clear()
        for key in self.waiting_sites.keys():
            self.waiting_sites[key] = []
            
        # ==========================================================
        # DR 1: 进场时间扰动 (Arrival Jitter)
        # ==========================================================
        self.landing_list = []
        self.plane_num_per_batch = len(self.flights_data)
        
        # 仅在开启随机化时生成预置飞机
        if self.use_domain_rand:
            num_pre_planes = self.np_random.integers(0, 3) if hasattr(self, 'np_random') else np.random.randint(0, 3)
        else:
            num_pre_planes = 0
            
        # [修改点 3]：基于 base_landing_list 恢复环境并加入随机扰动
        for idx, base_flight in enumerate(self.base_landing_list):
            if idx >= len(self.flights_data) - num_pre_planes:
                continue
                
            base_time = base_flight['land_time']
            
            # ====== 【核心修改：强制第一架飞机在 t=0 降落】 ======
            if base_time == 0:
                land_time = 0
            else:
                jitter = (self.np_random.integers(-5, 6) if hasattr(self, 'np_random') else np.random.randint(-5, 6)) if self.use_domain_rand else 0
                land_time = max(0, base_time + jitter)
            # ====================================================

            self.landing_list.append((land_time, base_flight['bidx'], base_flight['pidx'], base_flight['fuel']))
                
        # 按时间进行排序
        self.landing_list.sort(key=lambda x: x[0])
        
        # ==========================================================
        # DR 3: 移动设备初始位置打乱 
        # (这部分代码保持你上一版的原样，不需要动)
        # ==========================================================
        for resource in list(self.fixed_resources.values()) + list(self.mobile_resources.values()):
            resource.reset()
        for site in self.sites.values():
            site.reset()
        gate_codes = list(self.service_site_code_list)
        for devices in self.mobile_devices.values():
            for device in devices:
                device.reset() 
                if self.use_domain_rand and device.resource.type != 'R014': 
                    random_site_code = self.np_random.choice(gate_codes) if hasattr(self, 'np_random') else np.random.choice(gate_codes)
                    device.start_transport(self.sites[random_site_code])
                    device.finish_transport()

        # ==========================================================
        # DR 2 & 4: 初始化进场与在场飞机
        # ==========================================================
        # [修改点 4]：处理 0 时刻即到达着陆跑道的飞机
        while self.landing_list and self.total_time >= self.landing_list[0][0]:
            land_time, bidx, pidx, fuel = self.landing_list.pop(0) 
            
            # 油量也加入微小扰动以增加样本多样性
            final_fuel = fuel
            if self.use_domain_rand:
                fuel_jitter = self.np_random.integers(-5, 6) if hasattr(self, 'np_random') else np.random.randint(-5, 6)
                final_fuel = max(0, min(100, fuel + fuel_jitter))
                
            plane_cfg = {
                'velocity': 5,
                'site': self.sites['Z'],
                'fuel': final_fuel,
                'jobs': self._build_plane_jobs(drop_optional=self.use_domain_rand)
            }
            self.add_planes([{'batch': bidx, 'idx': pidx, **plane_cfg}])

        # --- 新增：随机生成开局就已经停在机位上的飞机 ---
        if num_pre_planes > 0:
            chosen_gates = self.np_random.choice(gate_codes, num_pre_planes, replace=False) if hasattr(self, 'np_random') else np.random.choice(gate_codes, num_pre_planes, replace=False)
            for i, gate_code in enumerate(chosen_gates):
                # 【修复 1】：使用合法的正数编号。接在第 0 批次的末尾
                pidx = self.plane_num_per_batch - num_pre_planes + i
                pre_cfg = {
                    'velocity': 5,
                    'site': self.sites[gate_code],
                    'fuel': 100, 
                    'jobs': self._build_plane_jobs(drop_optional=self.use_domain_rand)
                }
                # 依然当做 batch 0 注册，这样 global_pid 计算出来是完全合法的
                self.add_planes([{'batch': 0, 'idx': pidx, **pre_cfg}])
                plane_obj = self.planes[f'Plane_0_{pidx}']
                plane_obj.finished_jobs.extend(['ZY_Z', 'ZY_M', 'ZY01'])
                plane_obj.ever_finished_jobs.update(['ZY_Z', 'ZY_M', 'ZY01'])
                
        return self._get_obs(), self._get_done(), self._get_info()
    
    def shuffer_data(self):
        """Advance a deterministic case-coverage cycle between train epochs.

        Legacy callers consume every case assigned to a worker per epoch, so
        they keep the historical reshuffle-and-rewind behavior.  A bounded
        epoch budget instead walks non-overlapping windows of a larger local
        pool and reshuffles only after that pool has been exhausted.
        """
        rotating = self._train_epoch_case_budget_per_worker > 0
        if rotating:
            if (
                not self._training_case_cycle_initialized
                or self.data_idx == 0
            ):
                if len(self.data_list) > 1:
                    self.np_random.shuffle(self.data_list)
                self.data_idx = 0
                self._training_case_cycle_initialized = True
            return
        if len(self.data_list) > 1:
            self.np_random.shuffle(self.data_list)
        self.data_idx = 0

    def reset_training_case_cycle(self):
        """Start a fresh rotating pool, e.g. after critic-only calibration."""
        self.data_idx = 0
        self._training_case_cycle_initialized = False
        return True

    def reset_data_cursor(self):
        """Rewind evaluation to the first case without changing case order."""
        self.data_idx = 0

    def set_data_cursor(self, completed_cases):
        """Restore an exact per-worker case cursor for supervised recovery."""

        completed_cases = int(completed_cases)
        if completed_cases < 0 or completed_cases >= len(self.data_list):
            raise ValueError(
                'Training data cursor must identify an unfinished case window: '
                f'cursor={completed_cases}, cases={len(self.data_list)}.'
            )
        self.data_idx = completed_cases
        return int(self.data_idx)

    def set_iga_potential_beta(self, beta):
        """Update shaping strength between PPO epochs without rebuilding envs."""
        beta = float(beta)
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError('iga_potential_beta must be finite and non-negative.')
        self.iga_potential_beta = beta
        return self.iga_potential_beta

    def set_resource_lateness_coef(self, coefficient):
        """Update the bounded Stage-2 wait multiplier between PPO epochs."""
        coefficient = float(coefficient)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(
                'resource_lateness_coef must be finite and non-negative.'
            )
        for config in self.data_list:
            config['resource_lateness_coef'] = coefficient
        self.config['resource_lateness_coef'] = coefficient
        self.resource_lateness_coef = coefficient
        return self.resource_lateness_coef

    def _get_episode_rewards(self):
        return self.total_time

    def get_resource_lateness_metrics(self):
        """Return episode-level observed lateness and anti-hoarding metrics."""
        if (
            bool(getattr(self, 'done', False))
            and self._resource_lateness_metrics_cache is not None
        ):
            return dict(self._resource_lateness_metrics_cache)
        if not hasattr(self, 'job_code_list'):
            # Lightweight contract/unit-test environments created via
            # ``__new__`` predate Stage-2 resource tracing.
            return {
                'total_wait_seconds': 0.0,
                'max_wait_seconds_per_aircraft': 0.0,
                'p95_wait_seconds_per_aircraft': 0.0,
                'critical_wait_seconds': 0.0,
                'critical_wait_p95_seconds_per_aircraft': 0.0,
                'critical_avoidable_lateness_seconds': 0.0,
                'critical_avoidable_lateness_p95_seconds_per_aircraft': 0.0,
                'rendezvous_spread_seconds': 0.0,
                'rendezvous_spread_p95_seconds': 0.0,
                'predicted_lateness_seconds': 0.0,
                'predicted_lateness_p95_seconds': 0.0,
                'predicted_lateness_max_seconds': 0.0,
                'late_dispatch_count': 0,
                'late_dispatch_rate': 0.0,
                'early_arrival_seconds': 0.0,
                'lookahead_dispatch_count': 0,
                'visibility_to_legal_seconds': 0.0,
                'legal_to_idle_seconds': 0.0,
                'policy_defer_seconds': 0.0,
                'policy_defer_p95_seconds': 0.0,
                'policy_defer_max_seconds': 0.0,
                'policy_defer_count': 0,
                'policy_defer_rate': 0.0,
                'intent_status_counts': {},
            }
        from onpolicy.envs.HKBZ.experiment.resource_wait_metrics import (
            summarize_aircraft_resource_wait,
        )

        job_resources = {
            code: self._needed_mobile_types(code)
            for code in self.job_code_list
        }
        summary = summarize_aircraft_resource_wait(
            self.trajectory_log,
            self.device_trajectory_log,
            job_resources,
            transporter_type=self.TRANSPORTER_RESOURCE_TYPE,
            aircraft_count=len(self.flights_data),
            episode_cmax=float(self.total_time),
            criticality_scale_seconds=float(
                self.resource_slack_criticality_seconds
            ),
            criticality_min_weight=float(self.resource_slack_min_weight),
        )
        lookahead_records = [
            record for record in self.device_trajectory_log
            if record.get('is_lookahead', False)
        ]
        predicted_lateness_values = [
            max(0.0, float(record.get(
                'predicted_lateness_at_dispatch', 0.0
            )))
            for record in lookahead_records
        ]
        dependency_depth_counts = {
            str(depth): sum(
                int(record.get('dependency_depth', 0)) == depth
                for record in lookahead_records
            )
            for depth in range(0, self.device_future_intent_horizon + 1)
        }
        dispatched_intents = [
            entry for entry in self.resource_intent_ledger.values()
            if entry.get('dispatch_time') is not None
        ]
        visibility_to_legal = [
            max(
                0.0,
                float(entry.get('first_legal_time', entry['dispatch_time']))
                - float(entry.get('first_visible_time', entry['dispatch_time'])),
            )
            for entry in dispatched_intents
        ]
        legal_to_idle = [
            max(
                0.0,
                float(entry.get(
                    'first_compatible_idle_time', entry['dispatch_time']
                ))
                - float(entry.get('first_legal_time', entry['dispatch_time'])),
            )
            for entry in dispatched_intents
        ]
        policy_defer = [
            max(
                0.0,
                float(entry['dispatch_time'])
                - float(entry.get(
                    'first_compatible_idle_time', entry['dispatch_time']
                )),
            )
            for entry in dispatched_intents
        ]
        result = {
            **summary,
            'predicted_lateness_seconds': float(sum(
                predicted_lateness_values
            )),
            'predicted_lateness_p95_seconds': float(
                np.quantile(predicted_lateness_values, 0.95)
                if predicted_lateness_values else 0.0
            ),
            'predicted_lateness_max_seconds': float(
                max(predicted_lateness_values, default=0.0)
            ),
            'late_dispatch_count': int(sum(
                value > 1e-9 for value in predicted_lateness_values
            )),
            'late_dispatch_rate': float(
                sum(value > 1e-9 for value in predicted_lateness_values)
                / len(predicted_lateness_values)
                if predicted_lateness_values else 0.0
            ),
            'early_arrival_seconds': float(sum(
                max(0.0, float(record.get(
                    'predicted_earliness_at_dispatch', 0.0
                )))
                for record in lookahead_records
            )),
            'lookahead_dispatch_count': int(len(lookahead_records)),
            'visibility_to_legal_seconds': float(sum(visibility_to_legal)),
            'legal_to_idle_seconds': float(sum(legal_to_idle)),
            'policy_defer_seconds': float(sum(policy_defer)),
            'policy_defer_p95_seconds': float(
                np.quantile(policy_defer, 0.95) if policy_defer else 0.0
            ),
            'policy_defer_max_seconds': float(
                max(policy_defer, default=0.0)
            ),
            'policy_defer_count': int(sum(
                value > 1e-9 for value in policy_defer
            )),
            'policy_defer_rate': float(
                sum(value > 1e-9 for value in policy_defer)
                / len(policy_defer)
                if policy_defer else 0.0
            ),
            'lookahead_dependency_depth_counts': dependency_depth_counts,
            'intent_status_counts': {
                status: sum(
                    entry.get('status') == status
                    for entry in self.resource_intent_ledger.values()
                )
                for status in (
                    'soft', 'firm', 'dispatched', 'arrived', 'cancelled'
                )
            },
        }
        if bool(getattr(self, 'done', False)):
            self._resource_lateness_metrics_cache = dict(result)
        return result

    def get_training_objective(self):
        """Return the single case-level objective used by team-Cmax PPO."""
        cmax = float(self.total_time)
        team_return = -float(self.hindsight_terminal_cmax_coef) * cmax
        resource_metrics = self.get_resource_lateness_metrics()
        resource_wait = float(resource_metrics['total_wait_seconds'])
        critical_wait = float(
            resource_metrics['max_wait_seconds_per_aircraft']
        )
        early_arrival = float(resource_metrics['early_arrival_seconds'])
        lateness_penalty = float(getattr(
            self, 'resource_lateness_coef', 0.0
        )) * resource_wait
        critical_lateness_penalty = (
            float(getattr(
                self, 'resource_critical_lateness_coef', 0.0
            )) * critical_wait
        )
        earliness_penalty = float(getattr(
            self, 'resource_earliness_coef', 0.0
        )) * early_arrival
        team_return -= (
            lateness_penalty
            + critical_lateness_penalty
            + earliness_penalty
        )
        cycle_penalty = 0.0
        if self.cycle_terminated:
            cycle_penalty = float(self.plane_cycle_penalty)
            team_return -= cycle_penalty
        return {
            'cmax': cmax,
            'team_return': team_return,
            'resource_wait_seconds': resource_wait,
            'resource_critical_wait_seconds': critical_wait,
            'resource_slack_weighted_wait_seconds': float(
                resource_metrics['critical_wait_seconds']
            ),
            'resource_avoidable_critical_lateness_seconds': float(
                resource_metrics['critical_avoidable_lateness_seconds']
            ),
            'resource_rendezvous_spread_seconds': float(
                resource_metrics['rendezvous_spread_seconds']
            ),
            'resource_early_arrival_seconds': early_arrival,
            'resource_predicted_lateness_seconds': float(
                resource_metrics['predicted_lateness_seconds']
            ),
            'resource_visibility_to_legal_seconds': float(
                resource_metrics['visibility_to_legal_seconds']
            ),
            'resource_legal_to_idle_seconds': float(
                resource_metrics['legal_to_idle_seconds']
            ),
            'resource_policy_defer_seconds': float(
                resource_metrics['policy_defer_seconds']
            ),
            'resource_lateness_penalty': float(lateness_penalty),
            'resource_critical_lateness_penalty': float(
                critical_lateness_penalty
            ),
            'resource_earliness_penalty': float(earliness_penalty),
            'cycle_penalty': cycle_penalty,
            'cycle_terminated': bool(self.cycle_terminated),
            'case_id': str(getattr(self, 'current_case_path', '')),
        }

    def get_role_event_credit_weights(self, credit_mode='critical_path'):
        """Return audited post-episode scores for role-return redistribution.

        The scores are *not* an auxiliary reward and never change the case
        objective.  They only describe which completed decision events are
        plausible carriers of the already fixed team-time cost.  The replay
        buffer normalizes these non-negative scores independently on every
        physical role clock and verifies exact return-mass conservation.

        Three observable signals are used:

        * forward Cmax-frontier extension (direct terminal-path evidence),
        * request wait / predicted lateness at dispatch (late-resource cause),
        * late-horizon action duration (weak coverage for useful actions that
          finish before a plane completion advances the global frontier).

        Keeping this extraction in the environment makes step/agent identity
        explicit and lets tests audit every contribution against the recorded
        trajectory rather than relying on a second simulator.
        """
        credit_mode = str(credit_mode)
        if credit_mode not in {'critical_path', 'critical_path_v2'}:
            raise ValueError(
                'credit_mode must be critical_path or critical_path_v2.'
            )
        if credit_mode == 'critical_path_v2':
            return self._get_role_event_credit_weights_v2()

        records = []
        for record in self.trajectory_log:
            if 'end_time' in record:
                records.append(('plane', record))
        for record in self.device_trajectory_log:
            if 'end_time' in record:
                role = (
                    'transporter'
                    if record.get('is_transporter', False)
                    else 'device'
                )
                records.append((role, record))

        records.sort(key=lambda item: (
            float(item[1].get('end_time', 0.0)),
            int(item[1].get('agent_id', -1)),
        ))
        cmax = max(0.0, float(self.total_time))
        frontier = 0.0
        weights = {}
        role_frontier_seconds = {
            'plane': 0.0, 'device': 0.0, 'transporter': 0.0,
        }
        for role, record in records:
            step_idx = int(record.get('step_idx', -1))
            agent_id = int(record.get('agent_id', -1))
            if step_idx < 0 or agent_id < 0:
                continue
            end_time = max(0.0, float(record.get('end_time', 0.0)))
            frontier_delta = max(0.0, end_time - frontier)
            frontier = max(frontier, end_time)
            role_frontier_seconds[role] += frontier_delta

            waiting = max(
                0.0, float(record.get('waiting_time_at_dispatch', 0.0))
            )
            predicted_lateness = max(
                0.0,
                float(record.get('predicted_lateness_at_dispatch', 0.0)),
            )
            duration = max(
                0.0,
                float(record.get(
                    'duration',
                    end_time - float(record.get('start_time', end_time)),
                )),
            )
            # A fourth-power horizon factor makes the duration term weak in
            # the middle of a schedule and informative near the Cmax tail.
            horizon_fraction = (
                min(1.0, max(0.0, end_time / cmax)) if cmax > 0.0 else 0.0
            )
            late_duration = 0.25 * duration * horizon_fraction ** 4
            score = (
                frontier_delta + waiting + predicted_lateness + late_duration
            )
            key = (step_idx, agent_id)
            data = weights.setdefault(key, {
                'weight': 0.0,
                'cmax_frontier_seconds': 0.0,
                'request_wait_seconds': 0.0,
                'predicted_lateness_seconds': 0.0,
                'late_duration_seconds': 0.0,
                'role': role,
                'record_count': 0,
            })
            data['weight'] += score
            data['cmax_frontier_seconds'] += frontier_delta
            data['request_wait_seconds'] += waiting
            data['predicted_lateness_seconds'] += predicted_lateness
            data['late_duration_seconds'] += late_duration
            data['record_count'] += 1

        return {
            'weights': weights,
            'cmax': cmax,
            'completed_record_count': len(records),
            'global_frontier_seconds': frontier,
            'unattributed_terminal_seconds': max(0.0, cmax - frontier),
            'role_frontier_seconds': role_frontier_seconds,
        }

    def _get_role_event_credit_weights_v2(self):
        """Slack-gated causal delay attribution for role-event returns.

        Version 1 treated raw resource wait as equally important for every
        aircraft.  That can reward an action on a plane with ample terminal
        slack as strongly as a truly Cmax-blocking action.  Version 2 first
        measures each aircraft's completion slack and exponentially gates all
        delay evidence.  It keeps the exact objective unchanged: these scores
        are normalized only after the episode by the replay buffer.
        """
        from onpolicy.envs.HKBZ.experiment.resource_wait_metrics import (
            summarize_aircraft_resource_wait,
        )

        def non_negative(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            return max(0.0, value) if np.isfinite(value) else 0.0

        plane_records = [
            record for record in getattr(self, 'trajectory_log', ())
            if 'end_time' in record
        ]
        device_records = [
            record for record in getattr(self, 'device_trajectory_log', ())
            if 'end_time' in record
        ]
        cmax = non_negative(getattr(self, 'total_time', 0.0))
        scale = max(1e-9, non_negative(getattr(
            self, 'resource_slack_criticality_seconds', 600.0
        )))
        minimum = min(1.0, non_negative(getattr(
            self, 'resource_slack_min_weight', 0.05
        )))

        completion_by_plane = {}
        for record in plane_records:
            plane_id = record.get('plane_id')
            if plane_id is None:
                continue
            key = str(plane_id)
            completion_by_plane[key] = max(
                completion_by_plane.get(key, 0.0),
                non_negative(record.get('end_time')),
            )

        job_resources = {
            code: self._needed_mobile_types(code)
            for code in getattr(self, 'job_code_list', ())
        }
        wait_summary = summarize_aircraft_resource_wait(
            plane_records,
            device_records,
            job_resources,
            transporter_type=getattr(
                self, 'TRANSPORTER_RESOURCE_TYPE', 'R014'
            ),
            aircraft_count=len(getattr(self, 'flights_data', ())),
            include_events=True,
            episode_cmax=cmax,
            criticality_scale_seconds=scale,
            criticality_min_weight=minimum,
        )
        wait_events = wait_summary.get('events', ())

        def matching_wait_event(record, *, plane=False):
            plane_id = record.get('plane_id')
            job_code = record.get(
                'target_job_code' if plane else 'job_code'
            )
            if plane_id is None or job_code is None:
                return {}
            candidates = [
                event for event in wait_events
                if str(event.get('plane_id')) == str(plane_id)
                and str(event.get('job_code')) == str(job_code)
            ]
            if not candidates:
                return {}
            start = non_negative(record.get('start_time'))
            return min(
                candidates,
                key=lambda event: abs(
                    non_negative(event.get('start_time')) - start
                ),
            )

        def criticality(record):
            plane_id = record.get('plane_id')
            completion = completion_by_plane.get(
                str(plane_id), non_negative(record.get('end_time'))
            )
            slack = max(0.0, cmax - completion)
            weight = minimum + (1.0 - minimum) * math.exp(-slack / scale)
            return float(weight), float(slack)

        records = [('plane', record) for record in plane_records]
        records.extend((
            'transporter' if record.get('is_transporter', False)
            else 'device',
            record,
        ) for record in device_records)
        records.sort(key=lambda item: (
            non_negative(item[1].get('end_time')),
            int(item[1].get('agent_id', -1)),
        ))

        weights = {}
        frontier = 0.0
        role_frontier_seconds = {
            'plane': 0.0, 'device': 0.0, 'transporter': 0.0,
        }
        component_totals = {
            'cmax_frontier_seconds': 0.0,
            'critical_wait_seconds': 0.0,
            'critical_avoidable_lateness_seconds': 0.0,
            'rendezvous_spread_seconds': 0.0,
            'precedence_blocking_seconds': 0.0,
            'predicted_lateness_seconds': 0.0,
            'late_duration_seconds': 0.0,
        }
        for role, record in records:
            step_idx = int(record.get('step_idx', -1))
            agent_id = int(record.get('agent_id', -1))
            if step_idx < 0 or agent_id < 0:
                continue
            end_time = non_negative(record.get('end_time'))
            frontier_delta = max(0.0, end_time - frontier)
            frontier = max(frontier, end_time)
            role_frontier_seconds[role] += frontier_delta
            gate, completion_slack = criticality(record)
            wait_event = matching_wait_event(
                record, plane=(role == 'plane')
            )

            if role == 'plane':
                raw_wait = non_negative(record.get(
                    'waiting_time', wait_event.get('wait_seconds', 0.0)
                ))
                predicted_lateness = 0.0
            else:
                raw_wait = non_negative(record.get(
                    'waiting_time_at_dispatch', 0.0
                ))
                depth = max(0, int(record.get('dependency_depth', 0)))
                predicted_lateness = (
                    non_negative(record.get(
                        'predicted_lateness_at_dispatch', 0.0
                    )) / float(1 + depth)
                )
            precedence = non_negative(wait_event.get(
                'waiting_before_dispatch_seconds', raw_wait
            ))
            rendezvous = non_negative(wait_event.get(
                'post_arrival_synchronization_seconds', 0.0
            ))
            avoidable = precedence + rendezvous
            critical_wait = gate * raw_wait
            critical_avoidable = gate * avoidable
            critical_rendezvous = gate * rendezvous
            critical_precedence = gate * precedence
            critical_prediction = gate * predicted_lateness
            duration = non_negative(record.get(
                'duration', end_time - non_negative(record.get('start_time'))
            ))
            horizon_fraction = (
                min(1.0, end_time / cmax) if cmax > 0.0 else 0.0
            )
            late_duration = 0.25 * gate * duration * horizon_fraction ** 4

            # Critical wait already contains its avoidable sub-components;
            # they are exposed separately for audit but are not double-counted.
            score = (
                frontier_delta
                + critical_wait
                + critical_prediction
                + critical_rendezvous
                + late_duration
            )
            key = (step_idx, agent_id)
            data = weights.setdefault(key, {
                'weight': 0.0,
                'cmax_frontier_seconds': 0.0,
                'critical_wait_seconds': 0.0,
                'critical_avoidable_lateness_seconds': 0.0,
                'rendezvous_spread_seconds': 0.0,
                'precedence_blocking_seconds': 0.0,
                'predicted_lateness_seconds': 0.0,
                'late_duration_seconds': 0.0,
                'completion_slack_seconds': completion_slack,
                'criticality_weight': gate,
                'role': role,
                'record_count': 0,
            })
            contributions = {
                'cmax_frontier_seconds': frontier_delta,
                'critical_wait_seconds': critical_wait,
                'critical_avoidable_lateness_seconds': critical_avoidable,
                'rendezvous_spread_seconds': critical_rendezvous,
                'precedence_blocking_seconds': critical_precedence,
                'predicted_lateness_seconds': critical_prediction,
                'late_duration_seconds': late_duration,
            }
            data['weight'] += score
            for name, value in contributions.items():
                data[name] += value
                component_totals[name] += value
            data['record_count'] += 1

        return {
            'schema_version': 2,
            'credit_mode': 'critical_path_v2',
            'weights': weights,
            'cmax': cmax,
            'completed_record_count': len(records),
            'global_frontier_seconds': frontier,
            'unattributed_terminal_seconds': max(0.0, cmax - frontier),
            'role_frontier_seconds': role_frontier_seconds,
            'component_totals': component_totals,
            'criticality_scale_seconds': scale,
            'criticality_min_weight': minimum,
        }

    def render(self):
        """渲染环境可视化（模仿 path_test_3 画风，并自动截屏）

        功能：
        - 画停机坪 / 跑道（彩色矩形）
        - 画飞机（彩色圆点 + 编号）
        - 画移动设备（紫色小方块 + 编号）
        - 每次调用 render 时，自动按“5 个波次 × 每波 5 张”保存前 25 张图片，
          然后额外保存一张“空白底图”（只有停机坪，没有飞机和设备）。

        说明：
        - 是否保存、保存目录、波次数量、每波张数都可以在下面的默认参数里改。
        - 假设你在外部控制“隔多久调用一次 render”，环境只负责
          在调用的前 25 次里自动保存图片。
        """
        if self.render_mode is None:
            return

        import os
        import matplotlib.pyplot as plt

        # ========= 字体设置（解决中文变方框问题，只做一次） =========
        if not hasattr(self, "_font_inited"):
            # 依次尝试这些中文字体，环境里有哪个就用哪个
            plt.rcParams["font.sans-serif"] = [
                "SimHei",               # Windows 常见
                "Microsoft YaHei",      # Windows 常见
                "WenQuanYi Micro Hei",  # Ubuntu 常见
                "Noto Sans CJK SC",     # 常用 CJK 字体
                "DejaVu Sans",          # 兜底（不一定全支持中文）
            ]
            plt.rcParams["axes.unicode_minus"] = False
            self._font_inited = True

        # ========= 截图参数初始化（只做一次） =========
        if not hasattr(self, "_render_save_inited"):
            self._render_save_inited = True
            # 你可以按需修改这些默认参数（也可以在外部手动改属性）
            self.render_num_waves = getattr(self, "render_num_waves", 5)           # 波次数
            self.render_frames_per_wave = getattr(self, "render_frames_per_wave", 20)  # 每波张数
            self.render_save_dir = getattr(self, "render_save_dir", "render_output")  # 保存目录
            self.render_auto_save = getattr(self, "render_auto_save", True)        # 是否自动保存
            # ✅ 新增：截图间隔（例如 20 表示每 20 次 render 保存一张）
            self.render_save_stride = getattr(self, "render_save_stride", 30)

            self._render_saved_count = 0
            self._render_blank_saved = False
            self._render_call_count = 0
            if self.render_auto_save:
                os.makedirs(self.render_save_dir, exist_ok=True)

        # ========= 创建 / 清空 画布 =========
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(16, 11), dpi=120)
            self.ax.set_title("机场调度可视化", fontsize=16, fontweight="bold")

        ax = self.ax
        ax.clear()

        # ========= 1. 计算视野范围 =========
        xs = [site.pos[0] for site in self.sites.values()]
        ys = [site.pos[1] for site in self.sites.values()]
        if xs and ys:
            margin = 10
            xmin, xmax = min(xs) - margin, max(xs) + margin
            ymin, ymax = min(ys) - margin, max(ys) + margin
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax.set_facecolor("#f7f7f7")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

        # 小工具：根据站位 code 决定区域类型
        def _zone_type(code: str) -> str:
            if code == "Z":
                return "landing"   # 降落区
            if code in self.takeoff_site_code_list:
                return "takeoff"   # 起飞区
            return "parking"       # 普通停机坪

        # ========= 2. 画所有机位（停机坪矩形，模仿 path_test_3） =========
        for code, site in self.sites.items():
            x, y = site.pos
            zone = _zone_type(code)

            if zone == "landing":
                facecolor = "lightgreen"
                edgecolor = "darkgreen"
                lw = 2.0
                alpha = 0.7
            elif zone == "takeoff":
                facecolor = "lightyellow"
                edgecolor = "orange"
                lw = 2.0
                alpha = 0.7
            else:
                facecolor = "lightblue"
                edgecolor = "blue"
                lw = 1.5
                alpha = 0.6

            rect = plt.Rectangle(
                (x - 4, y - 4),
                8,
                8,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=lw,
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                code,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                zorder=2,
            )

        # ========= 3. 画飞机（彩色圆点 + 编号） =========
        for idx, (plane_id, plane) in enumerate(self.planes.items()):
            x, y = plane.site.pos
            bidx, pidx = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])  # 提取飞机编号
            color = plt.cm.tab10(pidx % 12)

            ax.scatter(
                x,
                y,
                s=150,
                marker="o",
                c=[color],
                edgecolors="black",
                zorder=5,
            )
            ax.text(
                x,
                y + 5,
                plane_id,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                zorder=6,
            )

        # ========= 4. 画移动设备（紫色小方块 + 编号） =========
        for device_type, devices in self.mobile_devices.items():
            if device_type == "R014":
                for d in devices:
                    x, y = d.site.pos
                    ax.scatter(
                        x,
                        y - 3,
                        s=100,
                        marker="s",
                        c=["purple"],
                        edgecolors="black",
                        zorder=4,
                    )
                    ax.text(
                        x,
                        y - 7,
                        d.code,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="purple",
                        zorder=6,
                    )

        # ========= 5. 坐标轴 & 图例（中文不再方框） =========
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor="lightgreen", edgecolor="darkgreen",
                          alpha=0.7, label="Landing Zone (Z)"),
            plt.Rectangle((0, 0), 1, 1, facecolor="lightyellow", edgecolor="orange",
                          alpha=0.7, label=(
                              "Takeoff Area ("
                              + "/".join(self.takeoff_site_code_list)
                              + ")"
                          )),
            plt.Rectangle((0, 0), 1, 1, facecolor="lightblue", edgecolor="blue",
                          alpha=0.6, label="Aircraft parking area"),
            plt.Line2D([0], [0], marker="o", color="w", label="Plane",
                       markerfacecolor="red", markeredgecolor="black", markersize=10),
            plt.Line2D([0], [0], marker="s", color="w", label="Device",
                       markerfacecolor="purple", markeredgecolor="black", markersize=8),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

        # ========= 6. 刷新画面 =========
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # ========= 7. 截图保存逻辑（按间隔保存） =========
        if not self.render_auto_save:
            return

        # 统计 render 被调用了多少次
        self._render_call_count += 1

        max_images = self.render_num_waves * self.render_frames_per_wave

        # 7.1 只在“到达间隔点”时尝试保存：
        #     比如 stride=20，就只有第 1, 21, 41, ... 次 render 会保存。
        if self._render_saved_count < max_images:
            if (self._render_call_count - 1) % self.render_save_stride != 0:
                # 还没到保存间隔，直接返回
                return

            wave_idx = self._render_saved_count // self.render_frames_per_wave
            frame_idx = self._render_saved_count % self.render_frames_per_wave
            filename = f"wave{wave_idx + 1}_frame{frame_idx + 1}_step{self.steps:05d}.png"
            filepath = os.path.join(self.render_save_dir, filename)
            # self.fig.savefig(filepath, dpi=150, bbox_inches="tight")
            self._render_saved_count += 1

        # 7.2 保存一张“空白底图”（只有停机坪）
        elif not self._render_blank_saved:
            blank_fig, blank_ax = plt.subplots(figsize=(16, 11), dpi=120)
            blank_ax.set_facecolor("#f7f7f7")
            blank_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

            if xs and ys:
                blank_ax.set_xlim(min(xs) - 10, max(xs) + 10)
                blank_ax.set_ylim(min(ys) - 10, max(ys) + 10)

            # 只画停机坪 / 跑道，不画飞机和设备
            for code, site in self.sites.items():
                x, y = site.pos
                zone = _zone_type(code)

                if zone == "landing":
                    facecolor = "lightgreen"
                    edgecolor = "darkgreen"
                    lw = 2.0
                    alpha = 0.7
                elif zone == "takeoff":
                    facecolor = "lightyellow"
                    edgecolor = "orange"
                    lw = 2.0
                    alpha = 0.7
                else:
                    facecolor = "lightblue"
                    edgecolor = "blue"
                    lw = 1.5
                    alpha = 0.6

                rect = plt.Rectangle(
                    (x - 4, y - 4),
                    8,
                    8,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=1,
                )
                blank_ax.add_patch(rect)
                blank_ax.text(
                    x,
                    y,
                    code,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    zorder=2,
                )

            blank_ax.set_xlabel("X")
            blank_ax.set_ylabel("Y")
            blank_ax.set_aspect("equal", adjustable="box")

            blank_path = os.path.join(self.render_save_dir, "blank.png")
            # blank_fig.savefig(blank_path, dpi=150, bbox_inches="tight")
            # plt.close(blank_fig)

            self._render_blank_saved = True

    def close(self):
        '''关闭环境，释放资源'''
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

    def _plane_hindsight_shaping_reward(self, record):
        new_first = float(record.get('new_first_completions', 0.0))
        irreversible_delta = float(record.get('irreversible_progress_delta', 0.0))
        relocation_count = float(record.get('relocation_count', 0.0))
        reset_count = float(len(record.get('reset_long_jobs', [])))
        parallel_bonus = 0.0
        if new_first > 0.0:
            parallel_bonus = max(
                0.0,
                float(record['total_job_time']) - float(record['job_time']),
            )

        reward = (
            parallel_bonus
            + self.plane_first_completion_bonus * new_first
            - 2.0 * float(record['waiting_time'])
            - 1.0 * float(record['trans_time'])
            - self.plane_reset_job_penalty * reset_count
        )
        if relocation_count > 0.0 and irreversible_delta <= 0.0:
            reward -= self.plane_no_progress_penalty * relocation_count
        if record.get('repeat_relocation', False):
            reward -= self.plane_repeat_relocation_penalty
        return reward

    @staticmethod
    def _device_hindsight_shaping_reward(record):
        return (
            -1.0 * record.get('trans_time', 0.0)
            - 0.5 * record.get('waiting_time_at_dispatch', 0.0)
        )

    def _transporter_hindsight_shaping_reward(self, record):
        waiting_time = float(record.get('waiting_time_at_dispatch', 0.0))
        reposition_time = float(record.get('trans_time', 0.0))
        duration = float(record.get('duration', reposition_time))
        occupied_after_reposition = max(0.0, duration - reposition_time)
        service_bonus = self.TRANSPORTER_SERVICE_BONUS
        if record.get('job_code') != 'ZY-T':
            service_bonus *= 0.5

        return (
            service_bonus
            - self.TRANSPORTER_WAIT_PENALTY_COEF * waiting_time
            - self.TRANSPORTER_REPOSITION_PENALTY_COEF * reposition_time
            - self.TRANSPORTER_OCCUPANCY_PENALTY_COEF * occupied_after_reposition
        )

    def _transporter_decision_shaping_reward(self, record):
        reason = record.get('decision_type')
        waiting_time = float(record.get('waiting_time_at_dispatch', 0.0))
        if reason == 'noop_with_request':
            return (
                -self.TRANSPORTER_NOOP_PENALTY
                - 0.5 * self.TRANSPORTER_WAIT_PENALTY_COEF * waiting_time
            )
        if reason in {'duplicate_request', 'invalid_request', 'unservable_request'}:
            return (
                -self.TRANSPORTER_INVALID_REQUEST_PENALTY
                - 0.25 * self.TRANSPORTER_WAIT_PENALTY_COEF * waiting_time
            )
        return 0.0

    @staticmethod
    def _add_hindsight_reward(step_rewards, record, reward):
        if reward == 0.0:
            return
        key = (record['step_idx'], record['agent_id'])
        if key not in step_rewards:
            step_rewards[key] = {
                'action': record['action'],
                'makespan_contribution': 0.0,
                'reward': 0.0,
            }
        step_rewards[key].update({
            'action': record['action'],
        })
        if 'decision_type' in record:
            step_rewards[key]['decision_type'] = record['decision_type']
        if 'device_type' in record:
            step_rewards[key]['device_type'] = record['device_type']
        step_rewards[key]['reward'] += reward

    def _calculate_cmax_delta_rewards(self):
        cmax_frontier = 0.0
        rewards = {}
        records = []
        for record in self.trajectory_log:
            if 'end_time' in record:
                records.append(('plane', record))
        for record in self.device_trajectory_log:
            if 'end_time' in record:
                records.append(('device', record))

        records.sort(key=lambda item: (float(item[1].get('end_time', 0.0)), item[1].get('agent_id', -1)))
        for _, record in records:
            end_time = float(record.get('end_time', 0.0))
            cmax_delta = max(0.0, end_time - cmax_frontier)
            cmax_frontier = max(cmax_frontier, end_time)
            rewards[(record['step_idx'], record['agent_id'])] = {
                'action': record['action'],
                'makespan_contribution': cmax_delta,
                'reward': -self.hindsight_cmax_coef * cmax_delta,
            }
        return rewards

    def _calculate_iga_potential_rewards(self):
        """Combine exact -delta-Cmax credit with teacher-calibrated shaping."""
        step_rewards = self._calculate_cmax_delta_rewards()
        beta = float(self.iga_potential_beta)
        gamma = float(self.iga_potential_gamma)
        for transition in self.potential_transition_log:
            actions = list(transition.get('actions', []))
            if not actions:
                continue
            phi_before = self._iga_potential_value(transition['before'])
            phi_after = self._iga_potential_value(transition['after'])
            global_shaping = beta * (gamma * phi_after - phi_before)
            reward_share = global_shaping / float(len(actions))
            for record in actions:
                key = (record['step_idx'], record['agent_id'])
                if key not in step_rewards:
                    step_rewards[key] = {
                        'action': record['action'],
                        'makespan_contribution': 0.0,
                        'reward': 0.0,
                    }
                data = step_rewards[key]
                data['action'] = record['action']
                data['iga_potential_before'] = phi_before
                data['iga_potential_after'] = phi_after
                data['iga_potential_global_shaping'] = global_shaping
                data['iga_potential_shaping'] = reward_share
                data['iga_potential_num_actions'] = len(actions)
                data['reward'] += reward_share

        for record in self.device_decision_log:
            shaping_reward = (
                self.hindsight_shaping_coef
                * self._transporter_decision_shaping_reward(record)
            )
            self._add_hindsight_reward(step_rewards, record, shaping_reward)
        return step_rewards

    def _apply_terminal_cmax_penalty(self, step_rewards):
        coef = float(getattr(self, 'hindsight_terminal_cmax_coef', 0.0))
        if coef == 0.0 or not step_rewards:
            return step_rewards

        # Give every participating agent the same global terminal objective at
        # its final recorded decision.  The previous implementation divided
        # C_max by the number of actions and added that tiny value everywhere.
        # PPO averages samples rather than summing an episode, so policies could
        # dilute the C_max signal simply by generating more decisions.
        terminal_penalty = -coef * float(self.total_time)
        last_key_by_agent = {}
        for key in step_rewards:
            step_idx, agent_id = key
            previous = last_key_by_agent.get(agent_id)
            if previous is None or step_idx > previous[0]:
                last_key_by_agent[agent_id] = key
        for key in last_key_by_agent.values():
            data = step_rewards[key]
            data['terminal_cmax_penalty'] = terminal_penalty
            data['reward'] += terminal_penalty
        return step_rewards

    def _apply_cycle_penalty(self, step_rewards):
        if not self.cycle_terminated or not self.cycle_agent_ids:
            return step_rewards
        for agent_id in self.cycle_agent_ids:
            records = [
                record for record in self.trajectory_log
                if int(record.get('agent_id', -1)) == int(agent_id)
            ]
            if not records:
                continue
            record = max(records, key=lambda item: int(item.get('step_idx', -1)))
            key = (record['step_idx'], record['agent_id'])
            if key not in step_rewards:
                step_rewards[key] = {
                    'action': record['action'],
                    'makespan_contribution': 0.0,
                    'reward': 0.0,
                }
            step_rewards[key]['cycle_penalty'] = -self.plane_cycle_penalty
            step_rewards[key]['reward'] -= self.plane_cycle_penalty
        return step_rewards

    def _finalize_hindsight_rewards(self, step_rewards):
        step_rewards = self._apply_terminal_cmax_penalty(step_rewards)
        return self._apply_cycle_penalty(step_rewards)

    def calculate_hindsight_rewards(self) -> Dict[Tuple[int, int], dict]:
        """
        根据配置回填动作级 hindsight reward。

        shaping: 保留原有等待/转运/作业时间 shaping。
        cmax_delta: 按动作结束时间前向包络惩罚 C_max 增量，使单 episode 奖励总和对齐 -C_max。
        hybrid_cmax: 原 shaping 加上 C_max 增量惩罚。
        potential_cmax: cmax_delta 的势差形式别名；前向 Cmax 包络的增量和
        精确等于终局 Cmax，用于动作级信用分配而不改变优化目标。
        iga_potential: 保留同一个 -delta-Cmax 基项，并加入由 IGA 精确回放
        标定的 gamma*Phi(s')-Phi(s) 稠密势差。每个全局决策步只生成一份
        shaping，再在同一步活跃飞机间等分，避免奖励随飞机数重复放大。
        team_cmax: runner 使用一个 case-level Monte Carlo C_max 回报；不在动作记录中复制终端奖励。
        R014 转运车使用独立 shaping，并对 no-op/错误请求回填额外惩罚。
        """
        mode = getattr(self, 'hindsight_reward_mode', 'shaping')
        if mode in {
            'team_cmax',
            'team_time',
            'team_time_potential',
            'team_time_resource_potential',
            'team_time_resource_fitted_potential',
        }:
            # The runner obtains ``get_training_objective`` once per case and
            # assigns it as a shared Monte Carlo target. Returning an empty
            # action map prevents accidental per-agent terminal duplication.
            return {}
        if mode in {'cmax_delta', 'potential_cmax'}:
            step_rewards = self._calculate_cmax_delta_rewards()
            for record in self.device_decision_log:
                shaping_reward = self.hindsight_shaping_coef * self._transporter_decision_shaping_reward(record)
                self._add_hindsight_reward(step_rewards, record, shaping_reward)
            return self._finalize_hindsight_rewards(step_rewards)
        if mode == 'iga_potential':
            return self._finalize_hindsight_rewards(self._calculate_iga_potential_rewards())

        cmax_rewards = self._calculate_cmax_delta_rewards() if mode == 'hybrid_cmax' else {}
        step_rewards = cmax_rewards if mode == 'hybrid_cmax' else {}

        for record in self.trajectory_log:
            key = (record['step_idx'], record['agent_id'])
            shaping_reward = self.hindsight_shaping_coef * self._plane_hindsight_shaping_reward(record)
            if key not in step_rewards:
                step_rewards[key] = {
                    'action': record['action'],
                    'makespan_contribution': 0.0,
                    'reward': 0.0,
                }
            step_rewards[key].update({
                'action': record['action'],
            })
            step_rewards[key]['reward'] += shaping_reward

        for record in self.device_trajectory_log:
            if record.get('is_transporter', False):
                shaping_reward = self.hindsight_shaping_coef * self._transporter_hindsight_shaping_reward(record)
            else:
                shaping_reward = self.hindsight_shaping_coef * self._device_hindsight_shaping_reward(record)
            self._add_hindsight_reward(step_rewards, record, shaping_reward)

        for record in self.device_decision_log:
            shaping_reward = self.hindsight_shaping_coef * self._transporter_decision_shaping_reward(record)
            self._add_hindsight_reward(step_rewards, record, shaping_reward)

        return self._finalize_hindsight_rewards(step_rewards)
