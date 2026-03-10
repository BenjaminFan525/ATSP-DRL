#!/usr/bin/env python
import sys
import os
import numpy as np
import torch
import yaml
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
# 根据实际目录结构调整：指向项目根目录
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) 
sys.path.insert(0, root_dir)

# 导入新的航空调度环境
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.config.config import get_config

def parse_args(args, parser):
    parser.add_argument('--ac_config', type=str, default='onpolicy/config/ac.yaml', help="Path to the ac config file")
    parser.add_argument('--env_config', type=str, default='onpolicy/config/env.yaml', help="Path to the environment config file")
    all_args = parser.parse_known_args(args)[0]  
    return all_args

def _t2n(x):
    return x.detach().cpu().numpy()

def check_env_randomness(env, num_episodes=5):
    print("=== 开始环境随机性诊断 ===")
    for ep in range(num_episodes):
        env.reset()
        
        # 1. 提取降落时间表 (截取前5架飞机)
        landings = env.landing_list[:5]
        
        # 2. 提取第一批降落飞机的燃油和任务数量
        first_plane_id = list(env.planes.keys())[0] if env.planes else None
        if first_plane_id:
            fuel = env.planes[first_plane_id].config['fuel']
            job_count = len(env.planes[first_plane_id].left_jobs)
        else:
            fuel, job_count = "N/A", "N/A"
            
        print(f"Episode {ep+1}:")
        print(f"  前5架飞机降落时间: {landings}")
        print(f"  首架飞机 ({first_plane_id}) -> 燃油: {fuel}, 待办任务数: {job_count}")
        print("-" * 30)

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # ======== 1. 设备与种子初始化 ========
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device('cuda:1')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # ======== 2. 网络参数配置 ========
    ac_config = {}
    if os.path.exists(all_args.ac_config):
        with open(all_args.ac_config, 'r') as f:
            ac_config = yaml.safe_load(f)
    else:
        print(f"[Warning] AC Config not found at {all_args.ac_config}, using empty init.")

    case_dir = "/home/fanyx/HKBZ-environment/onpolicy/config"
    print(f">>> 准备加载测试算例: {case_dir}")

    # 环境配置参数：替换为动态加载的算例路径
    env_config = {
        'jobs_path': os.path.join(case_dir, 'job.json'),
        'fixed_res_path': os.path.join(case_dir, 'fixed_resources.json'),
        'mobile_res_path': os.path.join(case_dir, 'mobile_resources.json'),
        'sites_path': os.path.join(case_dir, 'sites.json'),
        'flights_path': os.path.join(case_dir, 'flights.json'), # 引入新增的航班文件
        'seed': 42,
        'interfere': [-1, [], 0],     # 不开启干涉
        'force_chosen': [-1, '', 0],  # 不开启强制动作
        'use_domain_rand': False      # 测试阶段严格关闭域随机化
    }
    
    env = AircraftScheduleEnv(env_config)

    # check_env_randomness(env)

    # ======== 4. 策略网络加载 ========
    from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy as Policy

    policy = Policy(all_args, ac_config, 
                    device=device)

    # 如果有预训练模型，可以在这里 Load
    checkpoint_dir = '/home/fanyx/HKBZ-environment/onpolicy/scripts/results/IA/simple/gnn_mappo/train-ppo3/run1/models/checkpoint_Epoch8.pt'
    # checkpoint_dir = '/home/fanyx/HKBZ-environment/onpolicy/scripts/results/IA/simple/gnn_mappo/train-newdata-ppo3/run6/models/checkpoint_Epoch100.pt'
    # checkpoint_dir = None  # 替换为实际路径，如果有的话
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Loading weights from {checkpoint_dir}")
        checkpoint = torch.load(checkpoint_dir, map_location=device)
        policy.ac.load_state_dict(checkpoint['model'])  
        
    policy.ac.eval()

    # ======== 5. 仿真测试循环 ========
    print("\n" + "="*50)
    print(">>> 开始执行模型测试推理...")
    start_real_time = time.time()
    
    obs, done, info = env.reset()
    
    rnn_states = np.zeros(
        (1, all_args.max_agent_num, all_args.recurrent_N, all_args.hidden_size), 
        dtype=np.float32
    )

    step_count = 0
    total_rewards = np.zeros(all_args.max_agent_num)
    
    while not np.all(done):
        graph_obs = [obs] 
        
        active_mask = np.expand_dims(info['active_agents'], axis=0)
        last_op = np.expand_dims(info['last_op_indices'], axis=0)
        last_site = np.expand_dims(info['last_site_indices'], axis=0)

        # 1. 神经网络做决策
        action, rnn_states = policy.act(
            graph_obs=graph_obs,
            rnn_states=rnn_states,
            active_agents=active_mask,
            last_op_indices=last_op,
            last_site_indices=last_site,
            deterministic=True
        )
        
        action = _t2n(action)[0]
        rnn_states = _t2n(rnn_states)

        active_pids = np.where(info['active_agents'])[0]
        active = info['active_agents']
        if len(active_pids) > 0:
            print(f"[Step {step_count} | Env Time {env.total_time}s] 活跃飞机: {active_pids.tolist()}")

        # 2. 与环境交互（这里内部会自动快进，不需要你操心了！）
        obs, rewards, done, info = env.step(action)
        
        # 3. 统计
        total_rewards += rewards.flatten()
        step_count += 1
        
        if step_count > 2000:
            print("\n【警告】达到最大测试步数，强制终止。")
            break

    hindsight_rewards_dict = env.calculate_hindsight_rewards()
    
    # ======== 6. 结果结算 ========
    print("\n" + "="*50)
    print(">>> 仿真测试结束！")
    print(f"实际运算耗时: {time.time() - start_real_time:.4f} 秒")
    print(f"决策步数: {step_count}")
    print(f"环境推演总耗时 (C_max): {env.total_time} 秒")
    print(f"总奖励: {total_rewards.sum():.2f}")
    
    print("\n各架飞机的累积奖励:")
    for i, r in enumerate(total_rewards):
        print(f"飞机 {i}: {r:.2f}")
    print("="*50)

if __name__ == "__main__":
    main(sys.argv[1:])