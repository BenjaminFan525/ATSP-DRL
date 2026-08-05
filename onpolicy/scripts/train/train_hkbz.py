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
PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.config.config import get_config
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.env_wrappers import GraphSubprocVecEnv
import yaml
import glob

"""Train the HKBZ GNN-MAPPO policy."""


def _project_path(value):
    """Resolve configuration paths relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_env_config(config_path):
    config_path = _project_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Environment config does not exist: {config_path}")

    with config_path.open('r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file) or {}

    path_keys = (
        'dataset_dir', 'eval_dataset_dir', 'jobs_path', 'fixed_res_path',
        'mobile_res_path', 'sites_path', 'flights_path',
    )
    for key in path_keys:
        if config.get(key):
            config[key] = str(_project_path(config[key]))
    return config


def _split_cases(case_dirs, thread_count, split_name):
    if thread_count > len(case_dirs):
        raise ValueError(
            f"{split_name} has {len(case_dirs)} cases but {thread_count} rollout "
            "threads were requested. Reduce the thread count."
        )
    if len(case_dirs) % thread_count:
        raise ValueError(
            f"{split_name} contains {len(case_dirs)} cases, which is not divisible "
            f"by {thread_count} rollout threads."
        )
    return [list(part) for part in np.array_split(case_dirs, thread_count)]

def make_train_env(all_args):
    # ================= 提前读取基础配置并打乱、切分数据集 =================
    env_config_base = _load_env_config(all_args.env_config)
            
    dataset_dir = env_config_base.get('dataset_dir', 'airport_dataset')
    case_dirs = sorted(glob.glob(os.path.join(dataset_dir, "case_*")))
    
    if not case_dirs:
        raise ValueError(f"🚨 错误：在目录 '{dataset_dir}' 中没有找到任何算例文件夹！请先生成数据集。")
        
    # 设置随机种子并打乱所有的 case_dirs
    rng = np.random.default_rng(all_args.seed)
    rng.shuffle(case_dirs)
    
    # 将整个数据集等分为 n_rollout_threads 份
    split_case_dirs = _split_cases(case_dirs, all_args.n_rollout_threads, "Training dataset")
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

    return GraphSubprocVecEnv([get_env_fn(rank) for rank in range(all_args.n_rollout_threads)]), len(split_case_dirs[0])


def make_eval_env(all_args):
    # ================= 评估环境：读取、打乱并切分数据集 =================
    env_config_base = _load_env_config(all_args.env_config)
            
    dataset_dir = env_config_base.get('eval_dataset_dir', env_config_base.get('dataset_dir', 'airport_dataset'))
    case_dirs = sorted(glob.glob(os.path.join(dataset_dir, "case_*")))
    
    if not case_dirs:
        raise ValueError(f"🚨 错误：在评估目录 '{dataset_dir}' 中没有找到任何算例文件夹！")
        
    # 打乱评估集 (可以保持和训练集不同的打乱方式，或者使用固定的评估顺序)
    rng = np.random.default_rng(all_args.seed * 2) 
    rng.shuffle(case_dirs)
    
    # 将评估数据集等分为 n_eval_rollout_threads 份
    split_case_dirs = _split_cases(case_dirs, all_args.n_eval_rollout_threads, "Evaluation dataset")
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

    return GraphSubprocVecEnv([get_env_fn(rank) for rank in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple', help="Which scenario to run on")
    parser.add_argument('--ac_config', type=str, default=str(PROJECT_ROOT / 'onpolicy/config/ac.yaml'), help="Path to the actor-critic config file")
    parser.add_argument('--env_config', type=str, default=str(PROJECT_ROOT / 'onpolicy/config/env.yaml'), help="Path to the environment config file")

    all_args = parser.parse_known_args(args)[0]  

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.use_recurrent_policy = True
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
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # config
    ac_config = _project_path(all_args.ac_config)
    if ac_config is not None:
        with open(ac_config, 'r', encoding='utf-8') as f:
            ac_config = yaml.safe_load(f)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
        "ac_config": ac_config,
        "num_agents": all_args.max_agent_num,
        "num_envs": n_envs
    }

    # run experiments
    from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
