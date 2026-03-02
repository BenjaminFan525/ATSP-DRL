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

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # ======== 1. 设备与种子初始化 ========
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device('cuda:0')
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

    env_config = {}
    if os.path.exists(all_args.env_config):
        with open(all_args.env_config, 'r') as f:
            env_config = yaml.safe_load(f)
    else:
        print(f"[Warning] Env Config not found at {all_args.env_config}, using empty init.")

    # ======== 3. 航空环境配置与初始化 ========
    # 这里填写 HKBZ 的具体配置
    # env_config = {
    #     'batch_num': 1,
    #     'plane_num_per_batch': 12,
    #     'n_agents': 12,
    #     'jobs_path': 'onpolicy/envs/HKBZ/utils/config/jobs.json',
    #     'fixed_res_path': 'onpolicy/envs/HKBZ/utils/config/fixed_resources.json',
    #     'mobile_res_path': 'onpolicy/envs/HKBZ/utils/config/mobile_resources.json',
    #     'sites_path': 'onpolicy/envs/HKBZ/utils/config/sites.json',
    #     'seed': all_args.seed,
    # }
    
    # 为了防止参数不匹配，强制覆盖 args 中的设置
    all_args.max_agent_num = env_config['n_agents']
    
    env = AircraftScheduleEnv(env_config)

    # ======== 4. 策略网络加载 ========
    from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy as Policy

    # 这里的 obs_space 和 act_space 传 None 或者 mock 均可，由于全图网络不依赖固定 dim
    policy = Policy(all_args, ac_config, 
                    obs_space=None, cent_obs_space=None, act_space=None, 
                    device=device)

    # 如果有预训练模型，可以在这里 Load
    # checkpoint_dir = getattr(all_args, 'model_dir', None)
    # if checkpoint_dir and os.path.exists(checkpoint_dir):
    #     print(f"Loading weights from {checkpoint_dir}")
    #     checkpoint = torch.load(checkpoint_dir, map_location=device)
    #     policy.ac.load_state_dict(checkpoint['model'])  
        
    policy.ac.eval()

    # ======== 5. 仿真测试循环 ========
    print("\n" + "="*50)
    print(">>> 开始执行模型测试推理...")
    start_real_time = time.time()
    
    obs, done, info = env.reset()
    
    # 初始化 GRU 隐藏状态: Shape [Batch, N_agents, Recurrent_N, Hidden_Size]
    # 我们用 Batch = 1 模拟单环境评估
    rnn_states = np.zeros(
        (1, all_args.max_agent_num, all_args.recurrent_N, all_args.hidden_size), 
        dtype=np.float32
    )

    step_count = 0
    total_rewards = np.zeros(all_args.max_agent_num)
    
    while not np.all(done):
        # 组装网络所需的输入 (添加 Batch=1 的维度)
        graph_obs = [obs]  # PyG Batch 需要的 List 形式
        
        # 维度扩展 [N_agents] -> [1, N_agents]
        active_mask = np.expand_dims(info['active_agents'], axis=0)
        last_op = np.expand_dims(info['last_op_indices'], axis=0)
        last_site = np.expand_dims(info['last_site_indices'], axis=0)

        # 执行推理 (Deterministic = True 用于评估)
        action, rnn_states = policy.act(
            graph_obs=graph_obs,
            rnn_states=rnn_states,
            active_agents=active_mask,
            last_op_indices=last_op,
            last_site_indices=last_site,
            deterministic=True
        )
        
        # 将张量转为 Numpy, 去掉 Batch 维度 [1, N_agents, 2] -> [N_agents, 2]
        action = _t2n(action)[0]
        rnn_states = _t2n(rnn_states)

        # 打印部分活跃飞机的动作分配
        active_pids = np.where(info['active_agents'])[0]
        active = info['active_agents']
        if len(active_pids) > 0:
            print(f"[Step {step_count} | Env Time {env.total_time}s] 活跃飞机: {active_pids.tolist()}")

        # 推演环境步
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards.flatten()*active
        
        while not np.any(info['active_agents']) and not np.all(done):
            obs, rewards, done, info = env.step(action)
         
        step_count += 1
        
        # 死锁安全逃生舱
        if step_count > 2000:
            print("\n【警告】达到最大测试步数，强制终止。")
            break

    # ======== 6. 结果结算 ========
    print("\n" + "="*50)
    print(">>> 仿真测试结束！")
    print(f"实际运算耗时: {time.time() - start_real_time:.4f} 秒")
    print(f"决策步数: {step_count}")
    print(f"环境推演总耗时 (C_max): {env.total_time} 秒")
    
    print("\n各架飞机的累积奖励:")
    for i, r in enumerate(total_rewards):
        print(f"飞机 {i}: {r:.2f}")
    print("="*50)

if __name__ == "__main__":
    main(sys.argv[1:])