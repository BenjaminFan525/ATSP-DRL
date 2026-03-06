#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

curr_path = os.path.dirname(os.path.abspath(__file__)) 
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path))) 

sys.path.append(parent_path)

from onpolicy.config.config import get_config
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.env_wrappers import GraphSubprocVecEnv, DummyVecEnv
from onpolicy.utils.util import shuffle_dataset
import yaml

"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env_config = {}
            if os.path.exists(all_args.env_config):
                with open(all_args.env_config, 'r') as f:
                    env_config = yaml.safe_load(f)
            env = AircraftScheduleEnv(env_config)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    return GraphSubprocVecEnv([get_env_fn(rank) for rank in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env_config = {}
            if os.path.exists(all_args.env_config):
                with open(all_args.env_config, 'r') as f:
                    env_config = yaml.safe_load(f)
            env = AircraftScheduleEnv(env_config)
            env.seed(all_args.seed * 50000 + rank * 10000)
            env.use_domain_rand = False
            return env
        return init_env        

    return GraphSubprocVecEnv([get_env_fn(rank) for rank in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple', help="Which scenario to run on")
    parser.add_argument('--ac_config', type=str, default='/home/fanyx/HKBZ-environment/onpolicy/config/ac.yaml', help="Path to the ac config file")
    parser.add_argument('--env_config', type=str, default='/home/fanyx/HKBZ-environment/onpolicy/config/env.yaml', help="Path to the environment config file")

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
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

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
        "num_agents": all_args.max_agent_num,
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
