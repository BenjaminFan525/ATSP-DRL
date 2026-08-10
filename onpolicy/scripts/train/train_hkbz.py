#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import copy
import json
import signal
import traceback
import faulthandler
import threading
import math
from collections import Counter
from datetime import datetime
curr_path = os.path.dirname(os.path.abspath(__file__)) 
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path))) 

sys.path.append(parent_path)

from onpolicy.config.config import get_config
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.experiment.eval_common import list_case_folders
from onpolicy.envs.env_wrappers import GraphSubprocVecEnv, DummyVecEnv
from onpolicy.utils.util import shuffle_dataset
import yaml
import glob

"""Train script for MPEs."""


def apply_formal_safe_pipeline_manifest(all_args, repository_root=None):
    """Apply an explicitly declared, formal-only safe-pipeline manifest.

    The Stage-1 suite scheduler is intentionally long lived.  A scheduler that
    was started before these CLI flags existed cannot add them to later formal
    child commands without being restarted.  This opt-in manifest lets a fresh
    training child discover the declaration while keeping every screen/audit
    command on the default synchronous path.
    """
    experiment_name = str(getattr(all_args, 'experiment_name', '') or '')
    formal_marker = '_formal_'
    if formal_marker not in experiment_name:
        return None
    run_tag = experiment_name.split(formal_marker, 1)[0]
    if not run_tag or Path(run_tag).name != run_tag:
        raise ValueError(
            f'Unsafe formal experiment prefix for pipeline manifest: {run_tag!r}'
        )
    root = Path(repository_root or parent_path).resolve()
    manifest_path = (
        root / 'result/hkbz_train_logs' / run_tag
        / 'formal_safe_pipeline.json'
    )
    if not manifest_path.is_file():
        return None
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('enabled') is not True:
        return None
    if manifest.get('formal_only') is not True:
        raise ValueError(
            f'Safe-pipeline manifest must declare formal_only=true: '
            f'{manifest_path}'
        )
    if str(manifest.get('run_tag', '')) != run_tag:
        raise ValueError(
            'Safe-pipeline manifest run_tag mismatch: '
            f'expected={run_tag!r}, observed={manifest.get("run_tag")!r}'
        )
    clone_workers = int(manifest.get('safe_async_graph_clone_workers', 0))
    if clone_workers < 0:
        raise ValueError(
            'safe_async_graph_clone_workers in the activation manifest must '
            'be non-negative.'
        )
    for key in (
        'safe_graph_batch_pipeline', 'safe_dagger_teacher_overlap'
    ):
        if not isinstance(manifest.get(key), bool):
            raise ValueError(
                f'{key} in the activation manifest must be boolean.'
            )
    all_args.safe_async_graph_clone_workers = clone_workers
    all_args.safe_graph_batch_pipeline = manifest[
        'safe_graph_batch_pipeline'
    ]
    all_args.safe_dagger_teacher_overlap = manifest[
        'safe_dagger_teacher_overlap'
    ]
    all_args.safe_pipeline_activation_source = str(manifest_path.resolve())
    return {
        **manifest,
        'path': str(manifest_path.resolve()),
    }


def parse_distribution_weights(spec):
    """Parse and normalize ``name=weight`` case-sampling specifications."""
    weights = {}
    for item in str(spec or '').split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(
                'train_sampling_weights entries must use name=weight, '
                f'got {item!r}.'
            )
        name, value_text = (part.strip() for part in item.split('=', 1))
        if not name or name in weights:
            raise ValueError(
                f'Invalid or duplicate train sampling distribution {name!r}.'
            )
        value = float(value_text)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f'Train sampling weight {name!r} must be finite and positive.'
            )
        weights[name] = value
    if not weights:
        raise ValueError('train_sampling_weights must contain at least one group.')
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def _largest_remainder_quotas(weights, total):
    if total <= 0:
        raise ValueError('The requested train sample size must be positive.')
    raw = {name: total * value for name, value in weights.items()}
    quotas = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda name: (-(raw[name] - quotas[name]), name),
    )
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def _case_metadata_group(case_dir, key):
    metadata_path = Path(case_dir) / 'metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(
            'balanced sampling requires metadata.json for every '
            f'case, missing {metadata_path}.'
        )
    with metadata_path.open('r', encoding='utf-8') as handle:
        metadata = json.load(handle)
    group = str(metadata.get(key, '')).strip()
    if not group:
        raise ValueError(
            f'Case metadata has no {key} label: {metadata_path}.'
        )
    return group


def _case_distribution(case_dir):
    return _case_metadata_group(case_dir, 'distribution')


def select_train_case_dirs(
    case_dirs,
    *,
    mode,
    weights_spec,
    seed,
    max_cases=0,
    sample_size=0,
):
    """Select one deterministic case-coverage cycle and return an audit."""
    case_dirs = [str(path) for path in case_dirs]
    if not case_dirs:
        raise ValueError('Cannot sample an empty training split.')
    max_cases = max(0, int(max_cases))
    sample_size = max(0, int(sample_size))
    explicit_limits = [
        value for value in (max_cases, sample_size) if value > 0
    ]
    explicit_target = min(explicit_limits) if explicit_limits else 0
    rng = np.random.default_rng(int(seed))

    if mode == 'uniform':
        selected = list(case_dirs)
        rng.shuffle(selected)
        if explicit_target > 0:
            selected = selected[:explicit_target]
        return selected, {
            'mode': mode,
            'requested_weights': None,
            'selected_cases': len(selected),
            'unique_cases': len(set(selected)),
            'expanded_for_full_coverage': False,
        }
    group_key = {
        'distribution_balanced': 'distribution',
        'profile_balanced': 'profile',
    }.get(mode)
    if group_key is None:
        raise ValueError(f'Unsupported train_sampling_mode={mode!r}.')

    weights = parse_distribution_weights(weights_spec)
    grouped = {name: [] for name in weights}
    unexpected = Counter()
    for case_dir in sorted(case_dirs):
        group = _case_metadata_group(case_dir, group_key)
        if group not in grouped:
            unexpected[group] += 1
        else:
            grouped[group].append(case_dir)
    if unexpected:
        raise ValueError(
            f'train_sampling_weights does not cover dataset {group_key}s: '
            f'{dict(sorted(unexpected.items()))}.'
        )
    missing = [name for name, cases in grouped.items() if not cases]
    if missing:
        raise ValueError(
            f'No training cases are available for weighted groups {missing}.'
        )

    if explicit_target > 0:
        target = explicit_target
        expanded_for_full_coverage = False
    else:
        # Retain all unique cases.  For the current 480/108/12 split and
        # 50/45/5 target this gives exactly 480/432/48 = 960 samples.
        target = max(
            int(math.ceil(len(grouped[name]) / weights[name] - 1e-12))
            for name in weights
        )
        quotas = _largest_remainder_quotas(weights, target)
        while any(
            quotas[name] < len(grouped[name]) for name in weights
        ):
            target += 1
            quotas = _largest_remainder_quotas(weights, target)
        expanded_for_full_coverage = target > len(case_dirs)
    quotas = _largest_remainder_quotas(weights, target)

    selected = []
    for name in sorted(grouped):
        pool = list(grouped[name])
        quota = quotas[name]
        while quota > 0:
            permutation = rng.permutation(len(pool))
            take = min(quota, len(pool))
            selected.extend(pool[int(index)] for index in permutation[:take])
            quota -= take
    rng.shuffle(selected)
    selected_counts = Counter(
        _case_metadata_group(path, group_key) for path in selected
    )
    unique_counts = Counter(
        _case_metadata_group(path, group_key) for path in set(selected)
    )
    audit = {
        'mode': mode,
        'group_key': group_key,
        'requested_weights': weights,
        'source_cases': len(case_dirs),
        f'source_{group_key}_counts': {
            name: len(grouped[name]) for name in sorted(grouped)
        },
        'selected_cases': len(selected),
        'unique_cases': len(set(selected)),
        f'selected_{group_key}_counts': dict(sorted(selected_counts.items())),
        f'unique_{group_key}_counts': dict(sorted(unique_counts.items())),
        'expanded_for_full_coverage': expanded_for_full_coverage,
    }
    return selected, audit


def make_train_env(all_args):
    # ================= 提前读取基础配置并打乱、切分数据集 =================
    env_config_base = {}
    if os.path.exists(all_args.env_config):
        with open(all_args.env_config, 'r', encoding='utf-8') as f:
            env_config_base = yaml.safe_load(f)
    env_config_base['hindsight_reward_mode'] = all_args.hindsight_reward_mode
    env_config_base['hindsight_cmax_coef'] = all_args.hindsight_cmax_coef
    env_config_base['hindsight_shaping_coef'] = all_args.hindsight_shaping_coef
    env_config_base['hindsight_terminal_cmax_coef'] = all_args.hindsight_terminal_cmax_coef
    env_config_base['iga_potential_weights_path'] = all_args.iga_potential_weights_path
    env_config_base['iga_potential_beta'] = all_args.iga_potential_beta
    env_config_base['iga_potential_gamma'] = all_args.iga_potential_gamma
    env_config_base['iga_teacher_dir'] = all_args.plane_bc_teacher_dir
    env_config_base['device_deadlock_repeat_limit'] = all_args.device_deadlock_repeat_limit
    env_config_base['plane_cycle_repeat_limit'] = all_args.plane_cycle_repeat_limit
    env_config_base['plane_no_progress_limit'] = all_args.plane_no_progress_limit
    env_config_base['plane_relocation_limit'] = all_args.plane_relocation_limit
    env_config_base['plane_first_completion_bonus'] = all_args.plane_first_completion_bonus
    env_config_base['plane_repeat_relocation_penalty'] = all_args.plane_repeat_relocation_penalty
    env_config_base['plane_reset_job_penalty'] = all_args.plane_reset_job_penalty
    env_config_base['plane_no_progress_penalty'] = all_args.plane_no_progress_penalty
    env_config_base['plane_cycle_penalty'] = all_args.plane_cycle_penalty
    env_config_base['use_domain_rand'] = bool(all_args.train_domain_rand)
    env_config_base['global_feature_mode'] = all_args.global_feature_mode
            
    dataset_dir = env_config_base.get('dataset_dir', 'airport_dataset')
    case_dirs = sorted(glob.glob(os.path.join(dataset_dir, "case_*")))
    
    if not case_dirs:
        raise ValueError(f"🚨 错误：在目录 '{dataset_dir}' 中没有找到任何算例文件夹！请先生成数据集。")
        
    max_train_cases = max(0, int(getattr(all_args, 'max_train_cases', 0)))
    case_dirs, sampling_audit = select_train_case_dirs(
        case_dirs,
        mode=str(getattr(all_args, 'train_sampling_mode', 'uniform')),
        weights_spec=str(getattr(
            all_args,
            'train_sampling_weights',
            'iid=0.50,ood_stress=0.45,ood_scale=0.05',
        )),
        seed=all_args.seed,
        max_cases=max_train_cases,
        sample_size=max(
            0, int(getattr(all_args, 'train_sampling_size', 0))
        ),
    )
    print(
        '[TrainSampling] '
        + json.dumps(sampling_audit, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if all_args.n_rollout_threads > len(case_dirs):
        raise ValueError(
            "n_rollout_threads cannot exceed the number of selected training cases: "
            f"threads={all_args.n_rollout_threads}, cases={len(case_dirs)}"
        )
    
    # 将整个数据集等分为 n_rollout_threads 份
    split_case_dirs = [list(a) for a in np.array_split(case_dirs, all_args.n_rollout_threads)]
    # ====================================================================

    def get_env_fn(rank):
        def init_env():
            # 获取分配给当前 rank 的算例路径列表
            rank_case_dirs = split_case_dirs[rank]
            
            # 为当前环境构建配置列表
            config_list = []
            for case_dir in rank_case_dirs:
                # 使用 deepcopy 防止多个环境或多个算例配置之间发生引用污染
                config = copy.deepcopy(env_config_base)
                config['jobs_path'] = os.path.join(case_dir, "job.json")
                config['fixed_res_path'] = os.path.join(case_dir, "fixed_resources.json")
                config['mobile_res_path'] = os.path.join(case_dir, "mobile_resources.json")
                config['sites_path'] = os.path.join(case_dir, "sites.json")
                config['flights_path'] = os.path.join(case_dir, "flights.json")
                config_list.append(config)

            # 传入配置列表
            env = AircraftScheduleEnv(config_list)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    return (
        GraphSubprocVecEnv(
            [get_env_fn(rank) for rank in range(all_args.n_rollout_threads)],
            ipc_timeout_seconds=all_args.ipc_timeout_seconds,
            async_graph_clone_workers=all_args.safe_async_graph_clone_workers,
        ),
        max(len(rank_cases) for rank_cases in split_case_dirs),
    )


def make_eval_env(all_args):
    # ================= 评估环境：读取、打乱并切分数据集 =================
    env_config_base = {}
    if os.path.exists(all_args.env_config):
        with open(all_args.env_config, 'r', encoding='utf-8') as f:
            env_config_base = yaml.safe_load(f)
    env_config_base['hindsight_reward_mode'] = all_args.hindsight_reward_mode
    env_config_base['hindsight_cmax_coef'] = all_args.hindsight_cmax_coef
    env_config_base['hindsight_shaping_coef'] = all_args.hindsight_shaping_coef
    env_config_base['hindsight_terminal_cmax_coef'] = all_args.hindsight_terminal_cmax_coef
    env_config_base['iga_potential_weights_path'] = all_args.iga_potential_weights_path
    env_config_base['iga_potential_beta'] = all_args.iga_potential_beta
    env_config_base['iga_potential_gamma'] = all_args.iga_potential_gamma
    env_config_base['iga_teacher_dir'] = ''
    env_config_base['device_deadlock_repeat_limit'] = all_args.device_deadlock_repeat_limit
    env_config_base['plane_cycle_repeat_limit'] = all_args.plane_cycle_repeat_limit
    env_config_base['plane_no_progress_limit'] = all_args.plane_no_progress_limit
    env_config_base['plane_relocation_limit'] = all_args.plane_relocation_limit
    env_config_base['plane_first_completion_bonus'] = all_args.plane_first_completion_bonus
    env_config_base['plane_repeat_relocation_penalty'] = all_args.plane_repeat_relocation_penalty
    env_config_base['plane_reset_job_penalty'] = all_args.plane_reset_job_penalty
    env_config_base['plane_no_progress_penalty'] = all_args.plane_no_progress_penalty
    env_config_base['plane_cycle_penalty'] = all_args.plane_cycle_penalty
    env_config_base['use_domain_rand'] = False
    env_config_base['global_feature_mode'] = all_args.global_feature_mode
            
    dataset_dir = str(
        getattr(all_args, 'eval_dataset_dir', '')
        or env_config_base.get(
            'eval_dataset_dir',
            env_config_base.get('dataset_dir', 'airport_dataset'),
        )
    )
    all_case_dirs = sorted(glob.glob(os.path.join(dataset_dir, "case_*")))
    
    if not all_case_dirs:
        raise ValueError(f"🚨 错误：在评估目录 '{dataset_dir}' 中没有找到任何算例文件夹！")

    eval_case_offset = max(0, int(getattr(all_args, 'eval_case_offset', 0)))
    max_eval_cases = max(0, int(getattr(all_args, 'max_eval_cases', 0)))
    case_names = list_case_folders(
        dataset_dir,
        max_eval_cases,
        case_offset=eval_case_offset,
        partition_seed=int(all_args.eval_partition_seed),
        stratify_by=(
            str(getattr(all_args, 'eval_partition_stratify_by', '') or '')
            or None
        ),
    )
    case_dirs = [os.path.join(dataset_dir, name) for name in case_names]
    print(
        '[EvalPartition] '
        + json.dumps({
            'dataset_dir': str(Path(dataset_dir).resolve()),
            'partition_seed': int(all_args.eval_partition_seed),
            'stratify_by': str(
                getattr(all_args, 'eval_partition_stratify_by', '') or ''
            ),
            'offset': eval_case_offset,
            'selected_cases': len(case_dirs),
            'first_case': Path(case_dirs[0]).name if case_dirs else None,
        }, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if all_args.n_eval_rollout_threads > len(case_dirs):
        raise ValueError(
            "n_eval_rollout_threads cannot exceed the number of selected validation cases: "
            f"threads={all_args.n_eval_rollout_threads}, cases={len(case_dirs)}"
        )
    
    # 将评估数据集等分为 n_eval_rollout_threads 份
    split_case_dirs = [list(a) for a in np.array_split(case_dirs, all_args.n_eval_rollout_threads)]
    # ====================================================================

    def get_env_fn(rank):
        def init_env():
            rank_case_dirs = split_case_dirs[rank]
            
            config_list = []
            for case_dir in rank_case_dirs:
                config = copy.deepcopy(env_config_base)
                config['jobs_path'] = os.path.join(case_dir, "job.json")
                config['fixed_res_path'] = os.path.join(case_dir, "fixed_resources.json")
                config['mobile_res_path'] = os.path.join(case_dir, "mobile_resources.json")
                config['sites_path'] = os.path.join(case_dir, "sites.json")
                config['flights_path'] = os.path.join(case_dir, "flights.json")
                config_list.append(config)

            env = AircraftScheduleEnv(config_list)
            env.seed(all_args.seed * 50000 + rank * 10000)
            env.use_domain_rand = False  # 评估时严格关闭域随机化
            return env
        return init_env        

    return (
        GraphSubprocVecEnv(
            [get_env_fn(rank) for rank in range(all_args.n_eval_rollout_threads)],
            ipc_timeout_seconds=all_args.ipc_timeout_seconds,
            async_graph_clone_workers=all_args.safe_async_graph_clone_workers,
        ),
        [len(rank_cases) for rank_cases in split_case_dirs],
    )

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple', help="Which scenario to run on")
    parser.add_argument('--ac_config', type=str, default='/home/fanyx/HKBZ-environment/onpolicy/config/ac.yaml', help="Path to the ac config file")
    parser.add_argument('--env_config', type=str, default='/home/fanyx/HKBZ-environment/onpolicy/config/env.yaml', help="Path to the environment config file")

    all_args = parser.parse_known_args(args)[0]  
    if os.path.exists(all_args.env_config):
        with open(all_args.env_config, 'r', encoding='utf-8') as f:
            env_cfg = yaml.safe_load(f) or {}
        all_args.max_agent_num = int(env_cfg.get('n_agents', all_args.max_agent_num))
        all_args.max_device_num = int(env_cfg.get('max_device_num', all_args.max_device_num))
        all_args.resource_policy = env_cfg.get('resource_policy', all_args.resource_policy)
        all_args.dataset_manifest = str(env_cfg.get('dataset_manifest', ''))
        if not str(getattr(all_args, 'eval_dataset_dir', '') or '').strip():
            all_args.eval_dataset_dir = str(
                env_cfg.get(
                    'eval_dataset_dir', env_cfg.get('dataset_dir', '')
                )
            )
        # Reward and training options remain CLI-controlled.  Overwriting them
        # here made explicit experiment flags silently lose to stale YAML.

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    pipeline_manifest = apply_formal_safe_pipeline_manifest(all_args)
    if int(all_args.safe_async_graph_clone_workers) < 0:
        raise ValueError('--safe_async_graph_clone_workers must be non-negative.')
    if pipeline_manifest is not None:
        print(
            '[SafePipelineActivation] '
            + json.dumps(
                pipeline_manifest, ensure_ascii=False, sort_keys=True
            ),
            flush=True,
        )
    if all_args.iga_potential_beta < 0.0:
        raise ValueError('--iga_potential_beta must be non-negative.')
    for flag, raw in (
        ('--iga_potential_beta_schedule', all_args.iga_potential_beta_schedule),
        ('--bc_reference_kl_coef_schedule', all_args.bc_reference_kl_coef_schedule),
    ):
        try:
            values = [
                float(item.strip())
                for item in str(raw or '').split(',')
                if item.strip()
            ]
        except ValueError as error:
            raise ValueError(f'{flag} must be a comma-separated float list.') from error
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f'{flag} values must be finite and non-negative.')
    if not 0.0 <= all_args.tail_policy_start_fraction <= 1.0:
        raise ValueError('--tail_policy_start_fraction must be in [0, 1].')
    if all_args.tail_policy_weight < 1.0:
        raise ValueError('--tail_policy_weight must be at least 1.')
    if not 0.0 <= all_args.iga_potential_gamma <= 1.0:
        raise ValueError('--iga_potential_gamma must be in [0, 1].')
    if all_args.hindsight_reward_mode == 'iga_potential':
        weights_path = Path(all_args.iga_potential_weights_path).expanduser()
        if not all_args.iga_potential_weights_path or not weights_path.is_file():
            raise FileNotFoundError(
                'iga_potential reward requires an existing '
                f'--iga_potential_weights_path, got {weights_path}.'
            )
        all_args.iga_potential_weights_path = str(weights_path.resolve())
    if bool(all_args.shared_eval_socket) != bool(all_args.shared_eval_cpu_set):
        raise ValueError(
            '--shared_eval_socket and --shared_eval_cpu_set must be supplied '
            'together.'
        )
    if float(all_args.shared_eval_timeout_seconds) <= 0.0:
        raise ValueError('--shared_eval_timeout_seconds must be positive.')
    all_args.use_recurrent_policy = True
    faulthandler.enable(all_threads=True)
    torch.multiprocessing.set_sharing_strategy(all_args.torch_mp_sharing_strategy)
    print(
        f"[Info] torch multiprocessing sharing strategy: "
        f"{torch.multiprocessing.get_sharing_strategy()}"
    )
    all_args.use_naive_recurrent_policy = False

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(str(all_args.device))
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    all_args.use_wandb = False
    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs, n_envs = make_train_env(all_args)
    # Without a shared per-GPU service, evaluation workers are created lazily
    # and closed after every call.  A configured shared service owns the only
    # persistent validation pool and this trainer sends it frozen snapshots.
    eval_envs, eval_case_counts = None, None
    eval_env_factory = (
        (lambda: make_eval_env(all_args))
        if all_args.use_eval and not all_args.shared_eval_socket else None
    )

    # config
    ac_config = all_args.ac_config
    if ac_config is not None:
        with open(ac_config, 'r') as f:
            ac_config = yaml.safe_load(f)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
        "ac_config": ac_config,
        "num_agents": all_args.max_agent_num + (all_args.max_device_num if all_args.resource_policy == 'drl' else 0),
        "num_envs": n_envs,
        "eval_case_counts": eval_case_counts,
        "eval_env_factory": eval_env_factory,
        "release_eval_envs_after_eval": True,
        "shared_eval_socket": all_args.shared_eval_socket,
        "shared_eval_cpu_set": all_args.shared_eval_cpu_set,
        "shared_eval_timeout_seconds": all_args.shared_eval_timeout_seconds,
    }

    # run experiments
    from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner as Runner

    runner = Runner(config)
    status_path = run_dir / 'run_status.json'
    emergency_saved = False
    interrupted_signal = None
    status_lock = threading.Lock()
    heartbeat_stop = threading.Event()
    started_at = datetime.now().isoformat()
    progress_state = {}

    def write_status(status, **extra):
        nonlocal progress_state
        timestamp = datetime.now().isoformat()
        with status_lock:
            if extra:
                progress_state = {**progress_state, **extra}
            payload = {
                'status': status,
                'timestamp': timestamp,
                'heartbeat_timestamp': timestamp,
                'started_at': started_at,
                'experiment_name': all_args.experiment_name,
                'training_stage': all_args.training_stage,
                'checkpoint_dir': str(all_args.checkpoint_dir or ''),
                'selection_checkpoint_dir': str(all_args.selection_checkpoint_dir or ''),
                'reset_optimizers_on_resume': bool(all_args.reset_optimizers_on_resume),
                'canary_eval_interval_shards': int(all_args.canary_eval_interval_shards),
                'canary_max_regression': float(all_args.canary_max_regression),
                'canary_stop_on_regression': bool(all_args.canary_stop_on_regression),
                'canary_rejected': bool(getattr(runner, 'canary_rejected', False)),
                'canary_rejection_info': dict(
                    getattr(runner, 'canary_rejection_info', {})
                ),
                'torch_mp_sharing_strategy': all_args.torch_mp_sharing_strategy,
                'safe_async_graph_clone_workers': int(
                    all_args.safe_async_graph_clone_workers
                ),
                'safe_graph_batch_pipeline': bool(
                    all_args.safe_graph_batch_pipeline
                ),
                'safe_dagger_teacher_overlap': bool(
                    all_args.safe_dagger_teacher_overlap
                ),
                'safe_pipeline_activation_source': str(
                    getattr(
                        all_args, 'safe_pipeline_activation_source', ''
                    )
                ),
                'shared_eval_socket': str(all_args.shared_eval_socket or ''),
                'shared_eval_cpu_set': str(
                    all_args.shared_eval_cpu_set or ''
                ),
                'pid': os.getpid(),
                'ppid': os.getppid(),
                'pgid': os.getpgid(0),
                'hostname': socket.gethostname(),
                'total_num_steps': int(getattr(runner, 'total_num_steps', 0)),
                'epoch': int(getattr(runner, 'current_epoch', -1)),
                'shard': int(getattr(runner, 'current_shard', -1)),
                **progress_state,
            }
            temporary_path = status_path.with_name(
                f".{status_path.name}.tmp.{os.getpid()}"
            )
            with open(temporary_path, 'w', encoding='utf-8') as status_file:
                json.dump(payload, status_file, ensure_ascii=False, indent=2)
                status_file.flush()
                os.fsync(status_file.fileno())
            os.replace(temporary_path, status_path)

    runner.progress_callback = lambda **progress: write_status(
        'running',
        **progress,
    )

    def heartbeat_loop():
        interval = float(getattr(all_args, 'status_heartbeat_seconds', 60.0))
        if interval <= 0.0:
            return
        while not heartbeat_stop.wait(interval):
            try:
                write_status('running')
            except Exception as heartbeat_error:
                print(f"[Warning] Failed to write run heartbeat: {heartbeat_error}")

    def save_emergency(reason):
        nonlocal emergency_saved
        if emergency_saved:
            return
        emergency_saved = True
        try:
            runner.save_emergency_checkpoint(reason)
        except Exception as save_error:
            print(f"[Warning] Failed to save emergency checkpoint: {save_error}")

    def handle_signal(signum, _frame):
        nonlocal interrupted_signal
        signal_name = signal.Signals(signum).name
        interrupted_signal = signal_name
        heartbeat_stop.set()
        reason = f"received {signal_name} ({signum})"
        write_status('interrupted', event='interrupted', signal=signal_name, reason=reason)
        save_emergency(reason)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, handle_signal)

    write_status('running', event='initializing')
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        name='hkbz-status-heartbeat',
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        runner.run()
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5.0)
        write_status(
            'completed',
            event='completed',
            best_eval_makespan=float(runner.best_eval_makespan),
        )
    except BaseException as error:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5.0)
        reason = f"{type(error).__name__}: {error}"
        if interrupted_signal is None:
            write_status(
                'failed',
                event='failed',
                reason=reason,
                traceback=traceback.format_exc(),
            )
        save_emergency(reason)
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5.0)
        envs.close()
        if all_args.use_eval and runner.eval_envs is not None:
            runner.eval_envs.close()
            runner.eval_envs = None

        if all_args.use_wandb:
            run.finish()
        else:
            runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
            runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
