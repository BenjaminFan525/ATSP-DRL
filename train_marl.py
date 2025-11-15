import numpy as np
import pickle
from env import ScheduleEnv
import sys
import os
# from os.path import dirname, abspath
# sys.path.append(dirname(dirname(abspath(__file__))))
from MARL.runner import Runner
from utils.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, \
    get_commnet_args, get_g2anet_args

np.random.seed(2)

# 强化学习决策函数，带入来自DRL的强化学习agent
def marl_agent_wrapper():


    args = get_common_args()

    if args.alg.find('coma') > -1:  # 判断模型的参数
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    # 加载调度环境
    env = ScheduleEnv()

    env.reset(args.n_agents)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    print("是否加载模型（测试必须）：", args.load_model, "是否训练：",args.learn)
    # args.replay_dir
    runner = Runner(env, args)
    
    if args.learn:
        runner.run(args.alg)  
    else:
        start_time = datetime.now()
        win_rate, reward, time , move_time = runner.evaluate()
        end_time = datetime.now()
        running_time = end_time - start_time
        print('Evaluate win_rate: {}, reward: {}, makespan: {}, move_times: {}, running_time: {}'.format(win_rate, reward, time, move_time, running_time))
    print("Exiting Main")