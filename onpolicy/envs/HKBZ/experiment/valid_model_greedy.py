#!/usr/bin/env python
import sys
import os

# ================= 限制底层计算库的线程爆炸，防止 CPU 死锁 =================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# =======================================================================

import json
import numpy as np
import torch
import yaml
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) 
sys.path.insert(0, root_dir)

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.config.config import get_config
from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy as Policy

def parse_args(args, parser):
    parser.add_argument('--ac_config', type=str, default=os.path.join(root_dir, 'onpolicy/config/ac.yaml'), help="Path to the ac config file")
    parser.add_argument('--env_config', type=str, default=os.path.join(root_dir, 'onpolicy/config/env.yaml'), help="Path to the environment config file")
    parser.add_argument('--checkpoint', dest='checkpoint_dir',
                        help="Checkpoint produced by train_hkbz.py; alias of --checkpoint_dir.")
    parser.add_argument('--dataset_test_dir', '--dataset-dir', dest='dataset_test_dir', type=str,
                        default=os.path.join(root_dir, 'onpolicy/envs/HKBZ/dataset/fjsp_v2_t480_v60_test60/test'),
                        help="Path to the test dataset directory.")
    parser.add_argument('--output_json', type=str, default=None, help="Optional path to save evaluation results as JSON.")
    parser.add_argument('--max_cases', type=int, default=0, help="Limit number of cases for quick smoke tests; 0 means all cases.")
    all_args = parser.parse_known_args(args)[0]  
    if all_args.checkpoint_dir is None:
        parser.error('A checkpoint is required: use --checkpoint or --checkpoint_dir.')
    if os.path.exists(all_args.env_config):
        with open(all_args.env_config, 'r', encoding='utf-8') as f:
            env_cfg = yaml.safe_load(f) or {}
        all_args.max_agent_num = int(env_cfg.get('n_agents', all_args.max_agent_num))
        all_args.max_device_num = int(env_cfg.get('max_device_num', all_args.max_device_num))
        all_args.resource_policy = env_cfg.get('resource_policy', all_args.resource_policy)
    return all_args

def _t2n(x):
    return x.detach().cpu().numpy()

# ================= 1. 单次 Episode 测试函数 (Greedy) =================
def run_single_episode(env_config, policy, all_args, deterministic=True):
    try:
        env = AircraftScheduleEnv(env_config)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return None, None
        
    env.use_domain_rand = False 
    obs, done, info = env.reset()
    
    total_agent_num = all_args.max_agent_num + (all_args.max_device_num if all_args.resource_policy == 'drl' else 0)
    rnn_states = np.zeros((1, total_agent_num, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)

    step_count = 0
    start_time = time.perf_counter()
    
    while not np.all(done):
        graph_obs = [obs] 
        active_mask = np.expand_dims(info['active_agents'], axis=0)
        last_op = np.expand_dims(info['last_op_indices'], axis=0)
        last_site = np.expand_dims(info['last_site_indices'], axis=0)

        with torch.no_grad():
            action, rnn_states = policy.act(
                graph_obs=graph_obs,
                rnn_states=rnn_states,
                active_agents=active_mask,
                last_op_indices=last_op,
                last_site_indices=last_site,
                deterministic=deterministic 
            )
        
        action = _t2n(action)[0]
        rnn_states = _t2n(rnn_states)

        try:
            obs, rewards, done, info = env.step(action)
        except Exception:
            return None, None 
        
        step_count += 1
        if step_count > 2000: 
            break

    end_time = time.perf_counter()
    return env.total_time, (end_time - start_time)

# ================= 2. 主函数：评估流程 =================
def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    all_args.cuda = False # Greedy 模式下 CPU 推理通常更稳定
    device = torch.device("cpu")
        
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    ac_config = {}
    if os.path.exists(all_args.ac_config):
        with open(all_args.ac_config, 'r') as f:
            ac_config = yaml.safe_load(f)
            
    policy = Policy(all_args, ac_config, device=device)

    checkpoint_dir = os.path.abspath(os.path.expanduser(all_args.checkpoint_dir))
    if not os.path.isfile(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_dir}")
    print(f">>> 成功加载权重: {checkpoint_dir}")
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    policy.load_model_state(checkpoint['model'])
    policy.ac.tau = checkpoint['tau']
        
    policy.ac.eval()

    dataset_test_dir = os.path.abspath(os.path.expanduser(all_args.dataset_test_dir))
    if not os.path.isdir(dataset_test_dir):
        raise FileNotFoundError(f"Dataset does not exist: {dataset_test_dir}")
    case_folders = sorted([d for d in os.listdir(dataset_test_dir) 
                           if os.path.isdir(os.path.join(dataset_test_dir, d)) and d.startswith('case_')])
    if all_args.max_cases > 0:
        case_folders = case_folders[:all_args.max_cases]

    KNOWN_OPTIMAL_CMAX = {
    'case_01': 6670.0,
    'case_02': 6815.0,
    'case_03': 6858.0,
    'case_04': 7376.0,
    'case_05': 7214.0,
    'case_06': 6961.0,
    'case_07': 7126.0,
    'case_08': 7166.0,
    'case_09': 7440.0,
    'case_10': 7126.0,
    'case_11': 7161.0,
    'case_12': 7251.0,
    'case_13': 7235.0,
    'case_14': 6897.0,
    'case_15': 7824.0,
    'case_16': 7020.0,
    'case_17': 6855.0,
    'case_18': 7249.0,
    'case_19': 7053.0,
    'case_20': 6692.0,
}

    # 修改点 1：把 c_max_sum 替换为了 c_max_list
    metrics = {
        'DRL-G': {'c_max_list': [], 'time_sum': 0.0, 'gap_sum': 0.0, 'count': 0}
    }
    case_results = []

    print(f"\n🚀 开始 DRL-G (Greedy) 性能评估，共 {len(case_folders)} 个测试用例。")
    print("="*70)

    for case_name in case_folders:
        case_path = os.path.join(dataset_test_dir, case_name)
        flights_path = os.path.join(case_path, 'flights.json')
        
        # 动态获取 Agent 数量
        n_agents = 12
        if os.path.exists(flights_path):
            with open(flights_path, 'r', encoding='utf-8') as f:
                n_agents = len(json.load(f))
        configured_plane_agents = max(all_args.max_agent_num, n_agents)

        env_config = {
            'batch_num': 1, 'plane_num_per_batch': n_agents, 'n_agents': configured_plane_agents,
            'max_device_num': all_args.max_device_num,
            'resource_policy': all_args.resource_policy,
            'jobs_path': os.path.join(case_path, 'job.json'),
            'fixed_res_path': os.path.join(case_path, 'fixed_resources.json'),
            'mobile_res_path': os.path.join(case_path, 'mobile_resources.json'),
            'sites_path': os.path.join(case_path, 'sites.json'),
            'flights_path': flights_path, 'seed': 42,
            'interfere': [-1, [], 0], 'force_chosen': [-1, '', 0]
        }
        
        optimal_val = KNOWN_OPTIMAL_CMAX.get(case_name)
        if not optimal_val: continue
            
        # ---------------- 执行评估 ----------------
        c_max_g, time_g = run_single_episode(env_config, policy, all_args, deterministic=True)
        
        if c_max_g is not None:
            gap_g = ((c_max_g - optimal_val) / optimal_val) * 100.0
            
            # 修改点 2：将每次的结果追加到列表中
            metrics['DRL-G']['c_max_list'].append(c_max_g)
            metrics['DRL-G']['time_sum'] += time_g
            metrics['DRL-G']['gap_sum'] += gap_g
            metrics['DRL-G']['count'] += 1
            case_results.append({
                'case': case_name,
                'c_max': float(c_max_g),
                'gap': float(gap_g),
                'time': float(time_g),
                'optimal_cmax': float(optimal_val),
            })
            print(f"[{case_name}] C_max: {c_max_g:<8.1f} | Gap: {gap_g:>5.2f}% | Time: {time_g:.3f}s")

    # ======== 3. 打印最终统计表格 ========
    # 修改点 3：拓宽表格长度以容纳标准差，更新打印逻辑
    print("\n" + "="*65)
    print(f"{'Method':<10} {'C_max (± Std)':<20} {'Avg Time(s)':<12} {'Avg Gap(%)':<10}")
    print("-" * 65)
    
    m = metrics['DRL-G']
    if m['count'] > 0:
        avg_cmax = np.mean(m['c_max_list'])
        std_cmax = np.std(m['c_max_list'])
        cmax_display = f"{avg_cmax:.1f} ± {std_cmax:.1f}"
        
        print(f"{'DRL-G':<10} {cmax_display:<20} {m['time_sum']/m['count']:<12.3f} {m['gap_sum']/m['count']:<10.2f}")
        if all_args.output_json:
            output_dir = os.path.dirname(all_args.output_json)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(all_args.output_json, 'w', encoding='utf-8') as f:
                json.dump({
                    'method': 'DRL-G',
                    'checkpoint_dir': checkpoint_dir,
                    'dataset_test_dir': dataset_test_dir,
                    'count': int(m['count']),
                    'avg_cmax': float(avg_cmax),
                    'std_cmax': float(std_cmax),
                    'avg_time': float(m['time_sum'] / m['count']),
                    'avg_gap': float(m['gap_sum'] / m['count']),
                    'cases': case_results,
                }, f, indent=2, ensure_ascii=False)
    print("="*65)

if __name__ == "__main__":
    main(sys.argv[1:])
