import numpy as np
import pickle
from environment import ScheduleEnv
import sys
import os
import datetime
# from os.path import dirname, abspath
# sys.path.append(dirname(dirname(abspath(__file__))))
from MARL.runner import Runner
from utils.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, \
    get_commnet_args, get_g2anet_args

np.random.seed(42)

def site_disable(env: ScheduleEnv, site_code: str, disable_time: int):
    """
    禁用指定站点
    :param env: 调度环境
    :param site_code: 站点代码
    :param disable_time: 禁用时间
    """
    if site_code in env.sites:
        env.sites[site_code].disable(disable_time)
        print(f"Site {site_code} disabled for {disable_time} seconds.")
    else:
        print(f"Site {site_code} does not exist in the environment.")


# 强化学习决策函数，带入来自DRL的强化学习agent
def marl_agent_wrapper():
    args = get_common_args()
    args.alg = 'qmix'
    args.n_agents = 12

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

    config = {
        'batch_num': 1,
        'plane_num_per_batch': 12,
        'jobs_path': 'utils/config/jobs.json',
        'fixed_res_path': 'utils/config/fixed_resources.json',
        'mobile_res_path': 'utils/config/mobile_resources.json',
        'sites_path': 'utils/config/sites.json',
    }
    # 添加不可用停机位

    config['interfere'] = [2100, ['10', '11', '12', '13', '14', '15'], 1800]
    # 添加
    # 加载调度环境
    env = ScheduleEnv(config)

    env.reset()
    # env_info = env.get_env_info()
    args.n_actions = env.n_actions
    args.state_shape = len(env.get_state(args.n_agents))
    args.obs_shape = env.obs_dim
    args.episode_limit = 15000
    print("是否加载模型（测试必须）：", args.load_model, "是否训练：",args.learn)
    # args.replay_dir
    args.result_name = 'Test_100_epochs'
    args.n_epoch = 200
    args.save_cycle = 40
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

if __name__ == "__main__":
    marl_agent_wrapper()