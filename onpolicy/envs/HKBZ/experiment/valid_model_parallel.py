#!/usr/bin/env python
import sys
import os

# ================= 限制底层计算库的线程爆炸，防止 CPU 死锁 =================
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# =======================================================================

import json
import numpy as np
import torch
import yaml
import time
import copy
from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import Batch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) 
sys.path.insert(0, root_dir)

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.env_wrappers import GraphSubprocVecEnv, DummyVecEnv
from onpolicy.config.config import get_config
from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import GNN_MAPPOPolicy as Policy

# import cProfile
# import pstats

def parse_args(args, parser):
    parser.add_argument('--ac_config', type=str, default='onpolicy/config/ac.yaml', help="Path to the ac config file")
    parser.add_argument('--env_config', type=str, default='onpolicy/config/env.yaml', help="Path to the environment config file")
    parser.add_argument('--dataset_test_dir', type=str,
                        default='/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/dataset/fjsp_v2_t480_v60_test60/test',
                        help="Path to the test dataset directory.")
    parser.add_argument('--sample_num', type=int, default=50, help="Number of stochastic rollouts per case.")
    parser.add_argument('--output_json', type=str, default=None, help="Optional path to save evaluation results as JSON.")
    parser.add_argument('--max_cases', type=int, default=0, help="Limit number of cases for quick smoke tests; 0 means all cases.")
    parser.add_argument('--batch_cases', action='store_true', help="Evaluate all cases in one vectorized environment batch.")
    parser.add_argument('--eval_mode', type=str, default='stochastic', choices=['stochastic', 'deterministic'],
                        help="stochastic samples from the policy; deterministic uses greedy argmax actions.")
    parser.add_argument('--resource_policy_override', type=str, default=None, choices=['heuristic', 'drl'],
                        help="Override resource_policy from env_config for evaluation diagnostics.")
    all_args = parser.parse_known_args(args)[0]  
    if all_args.checkpoint_dir is None:
        all_args.checkpoint_dir = '/home/fanyx/HKBZ-environment/onpolicy/scripts/results/HKBZ/simple/gnn_mappo/e2e_drl_medium6_cuda1/run4/models/checkpoint_Epoch30.pt'
    if os.path.exists(all_args.env_config):
        with open(all_args.env_config, 'r', encoding='utf-8') as f:
            env_cfg = yaml.safe_load(f) or {}
        all_args.max_agent_num = int(env_cfg.get('n_agents', all_args.max_agent_num))
        all_args.max_device_num = int(env_cfg.get('max_device_num', all_args.max_device_num))
        all_args.resource_policy = env_cfg.get('resource_policy', all_args.resource_policy)
    if all_args.resource_policy_override is not None:
        all_args.resource_policy = all_args.resource_policy_override
    return all_args

def _t2n(x):
    return x.detach().cpu().numpy()

# ================= 核心工具：Tensor 与 Numpy 的深层转换 =================
def _to_numpy(obj):
    """递归将字典/列表中的 Tensor 转为 Numpy Array，避开 PyTorch 的 FD 序列化陷阱"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_numpy(v) for v in obj]
    return obj

def _to_tensor(obj):
    """递归将 Numpy Array 转回 Tensor"""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, dict):
        return {k: _to_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_tensor(v) for v in obj]
    return obj

# ================= 轻量化通信包装器 =================
class NumpyIPCWrapper:
    """拦截环境输出，将所有 Tensor 降级为原生 Numpy 进行极速 Pipe 传输"""
    def __init__(self, env):
        self.env = env
        
    def seed(self, seed):
        self.env.seed(seed)
        
    def reset(self):
        obs, done, info = self.env.reset()
        return _to_numpy(obs.to_dict()), done, _to_numpy(info)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _to_numpy(obs.to_dict()), reward, done, _to_numpy(info)
        
    # 【新增：补齐清理方法】
    def close(self):
        # 如果你的原始 AircraftScheduleEnv 有 close 方法就调用，没有就 pass
        if hasattr(self.env, 'close'):
            self.env.close()
        
    def __getattr__(self, name):
        # 放宽透传条件，让未显式定义的方法直接去底层环境找
        # 只要 env 有这个方法/属性，就返回它
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def make_eval_envs(env_config, num_envs, base_seed):
    def get_env_fn(rank):
        def init_env():
            env = AircraftScheduleEnv(env_config)
            env.seed(base_seed + rank * 1000)
            env.use_domain_rand = False 
            # 【应用拦截器】
            return NumpyIPCWrapper(env)
        return init_env

    # 继续保持你坚持的多进程路线
    return GraphSubprocVecEnv([get_env_fn(rank) for rank in range(num_envs)])

def make_eval_envs_from_configs(env_configs, base_seed):
    def get_env_fn(rank):
        def init_env():
            env = AircraftScheduleEnv(copy.deepcopy(env_configs[rank]))
            env.seed(base_seed + rank * 1000)
            env.use_domain_rand = False
            return NumpyIPCWrapper(env)
        return init_env

    return GraphSubprocVecEnv([get_env_fn(rank) for rank in range(len(env_configs))])

# ================= 3. 核心并行推理逻辑 =================
def run_parallel_episodes(envs, policy, all_args, num_envs, deterministic=True):
    # 动态获取模型所在的设备 (GPU/CPU)
    device = next(policy.ac.parameters()).device
    
    obs_dicts, dones, infos = envs.reset()
    
    total_agent_num = all_args.max_agent_num + (all_args.max_device_num if all_args.resource_policy == 'drl' else 0)
    rnn_states = np.zeros((num_envs, total_agent_num, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)

    active_masks = np.zeros((num_envs, total_agent_num), dtype=np.float32).reshape(num_envs, total_agent_num, 1)
    # 因为 infos 已经被 NumpyIPCWrapper 转成了 NumPy，这里可以直接安全操作
    active_masks[infos['active_agents'] == True] = np.ones(((infos['active_agents'] == True).sum(), 1), dtype=np.float32)

    dones_flag = np.zeros(num_envs, dtype=bool)
    actions = None 

    step_count = 0
    start_time = time.perf_counter()

    while not np.all(dones_flag):

        # 【核心重建】：将收到的轻量 Numpy 字典极速重组为 PyG HeteroData 并推入显存
        rebuilt_obs = [HeteroData.from_dict(_to_tensor(o)) for o in obs_dicts]
        batch_graph_obs = Batch.from_data_list(rebuilt_obs).to(device, non_blocking=True)

        with torch.inference_mode():
            action, rnn_states_out = policy.act(
                graph_obs=batch_graph_obs,
                rnn_states=rnn_states,
                active_agents=active_masks,
                last_op_indices=actions[..., 0] if step_count > 0 else -np.ones((num_envs, total_agent_num), dtype=np.float32),
                last_site_indices=actions[..., 1] if step_count > 0 else -np.ones((num_envs, total_agent_num), dtype=np.float32),
                deterministic=deterministic 
            )
        
        actions = _t2n(action)
        rnn_states_out = _t2n(rnn_states_out)

        # 向量化 Step
        obs_dicts, rewards, dones, infos = envs.step(actions)

        rnn_states_out[dones == True] = np.zeros(((dones == True).sum(), all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        rnn_states = rnn_states_out

        active_masks = np.zeros((num_envs, total_agent_num), dtype=np.float32).reshape(num_envs, total_agent_num, 1)
        active_masks[infos['active_agents'] == True] = np.ones(((infos['active_agents'] == True).sum(), 1), dtype=np.float32)

        for i in range(num_envs):
            if not dones_flag[i]:
                if np.all(dones[i]):
                    dones_flag[i] = True

        step_count += 1
        if step_count > 2000: 
            break

    inference_time = time.perf_counter() - start_time
    return envs.get_episode_rewards(), inference_time

# ================= 4. 主函数 =================
def main(args):
    # profiler = cProfile.Profile()
    # profiler.enable()
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # 强制开启 CUDA 加速
    all_args.cuda = True 
    if torch.cuda.is_available():
        print(">>> 检测到 GPU，正在启用火力全开模式...")
        device = torch.device("cuda:0")
    else:
        print(">>> 警告: 未检测到 GPU，使用 CPU...")
        device = torch.device("cpu")
        
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    ac_config = {}
    if os.path.exists(all_args.ac_config):
        with open(all_args.ac_config, 'r') as f:
            ac_config = yaml.safe_load(f)
            
    policy = Policy(all_args, ac_config, device=device)

    checkpoint_dir = all_args.checkpoint_dir

    if os.path.exists(checkpoint_dir):
        print(f">>> 成功加载权重: {checkpoint_dir}")
        checkpoint = torch.load(checkpoint_dir, map_location=device)
        policy.load_model_state(checkpoint['model'])
        policy.ac.tau = checkpoint['tau']
    else:
        print("【警告】未找到预训练模型权重！")
        
    policy.ac.eval()

    dataset_test_dir = all_args.dataset_test_dir
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
    
    SAMPLE_NUM = all_args.sample_num
    deterministic_eval = all_args.eval_mode == 'deterministic'
    method_name = 'DRL-G' if deterministic_eval else 'DRL-S'
    
    # 替换点 1：将 c_max_sum 改为存储 c_max 结果的列表
    metrics = {
        method_name: {'c_max_list': [], 'time_sum': 0.0, 'gap_sum': 0.0, 'count': 0}
    }
    case_results = []

    print(f"\n🚀 开始并行评估 ({method_name})，共 {len(case_folders)} 个测试用例。")
    print(f"ℹ️  评估模式: {all_args.eval_mode} | resource_policy: {all_args.resource_policy}")
    print(f"ℹ️  {method_name} 采样次数: {SAMPLE_NUM} 次/Case (完全多进程并发)")
    if all_args.batch_cases:
        print("ℹ️  启用 batch_cases：所有测试 case 将在同一个向量化环境批次中评估")
    print("="*70)

    eval_specs = []
    for case_name in case_folders:
        case_path = os.path.join(dataset_test_dir, case_name)
        
        flights_path = os.path.join(case_path, 'flights.json')
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

        for sample_idx in range(SAMPLE_NUM):
            eval_specs.append({
                'case_name': case_name,
                'sample_idx': sample_idx,
                'optimal_val': optimal_val,
                'env_config': copy.deepcopy(env_config),
            })

    if all_args.batch_cases and eval_specs:
        print(f"\n⚙️ 批量评估 {len(eval_specs)} 个 rollout ({len(case_folders)} cases × {SAMPLE_NUM} samples)")
        envs_s = make_eval_envs_from_configs([spec['env_config'] for spec in eval_specs], base_seed=666)
        valid_cmaxs_s, time_s = run_parallel_episodes(envs_s, policy, all_args, num_envs=len(eval_specs), deterministic=deterministic_eval)
        envs_s.close()

        grouped = {}
        for spec, cmax in zip(eval_specs, valid_cmaxs_s):
            grouped.setdefault(spec['case_name'], {'optimal_val': spec['optimal_val'], 'cmaxs': []})
            grouped[spec['case_name']]['cmaxs'].append(float(cmax))

        for case_name in case_folders:
            data = grouped.get(case_name)
            if not data:
                continue
            best_cmax_s = min(data['cmaxs'])
            optimal_val = data['optimal_val']
            gap_s = ((best_cmax_s - optimal_val) / optimal_val) * 100.0

            metrics[method_name]['c_max_list'].append(best_cmax_s)
            metrics[method_name]['time_sum'] += time_s / max(1, len(grouped))
            metrics[method_name]['gap_sum'] += gap_s
            metrics[method_name]['count'] += 1
            case_results.append({
                'case': case_name,
                'best_cmax': float(best_cmax_s),
                'gap': float(gap_s),
                'time': float(time_s / max(1, len(grouped))),
                'optimal_cmax': float(optimal_val),
                'sample_num': int(SAMPLE_NUM),
                'valid_rollouts': int(len(data['cmaxs'])),
                'rollout_cmaxs': data['cmaxs'],
                'batch_wall_time': float(time_s),
            })
            print(f"  [{method_name}] {case_name}: 最优 C_max = {best_cmax_s:.1f}s | Gap = {gap_s:.2f}%")
    elif not all_args.batch_cases:
        for case_name in case_folders:
            spec = next((item for item in eval_specs if item['case_name'] == case_name and item['sample_idx'] == 0), None)
            if spec is None:
                continue
            env_config = spec['env_config']
            optimal_val = spec['optimal_val']

            print(f"\n⚙️ 正在评估用例: {case_name} (Opt: {optimal_val})")

            envs_s = make_eval_envs(env_config, num_envs=SAMPLE_NUM, base_seed=666)
            valid_cmaxs_s, time_s = run_parallel_episodes(envs_s, policy, all_args, num_envs=SAMPLE_NUM, deterministic=deterministic_eval)
            envs_s.close()

            best_cmax_s = min(valid_cmaxs_s)
            gap_s = ((best_cmax_s - optimal_val) / optimal_val) * 100.0

            # 替换点 2：将得到的最优 c_max 添加到列表中
            metrics[method_name]['c_max_list'].append(best_cmax_s)
            metrics[method_name]['time_sum'] += time_s
            metrics[method_name]['gap_sum'] += gap_s
            metrics[method_name]['count'] += 1
            case_results.append({
                'case': case_name,
                'best_cmax': float(best_cmax_s),
                'gap': float(gap_s),
                'time': float(time_s),
                'optimal_cmax': float(optimal_val),
                'sample_num': int(SAMPLE_NUM),
                'valid_rollouts': int(len(valid_cmaxs_s)),
                'rollout_cmaxs': [float(cmax) for cmax in valid_cmaxs_s],
            })
            print(f"  [{method_name}] 采样完成 ({len(valid_cmaxs_s)}/{SAMPLE_NUM}次有效)! 最优 C_max = {best_cmax_s:.1f}s | CPU+GPU = {time_s:.3f}s | Gap = {gap_s:.2f}%")
        
        # if metrics['DRL-S']['count'] == 1:
        #     profiler.disable()
        #     stats = pstats.Stats(profiler).sort_stats('cumtime')
        #     print("\n>>> 性能分析报告:")
        #     stats.print_stats(15) 

    # 替换点 3：扩展表格并打印均值与标准差
    print("\n\n" + "="*60)
    print("Proposed Method")
    print(f"{'-'*60}")
    print(f"{'Method':<15} {'C_max (± Std)':<20} {'耗时 (s)':<10} {'Gap (%)':<10}")
    print(f"{'-'*60}")
    
    for method in [method_name]:
        count = metrics[method]['count']
        if count > 0:
            avg_cmax = np.mean(metrics[method]['c_max_list'])
            std_cmax = np.std(metrics[method]['c_max_list'])
            
            avg_time = metrics[method]['time_sum'] / count
            avg_gap = metrics[method]['gap_sum'] / count
            
            cmax_display = f"{avg_cmax:.1f} ± {std_cmax:.1f}"
            print(f"{method:<15} {cmax_display:<20} {avg_time:<10.2f} {avg_gap:<10.2f}")
            if all_args.output_json:
                output_dir = os.path.dirname(all_args.output_json)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(all_args.output_json, 'w', encoding='utf-8') as f:
                    json.dump({
                        'method': method,
                        'checkpoint_dir': checkpoint_dir,
                        'dataset_test_dir': dataset_test_dir,
                        'eval_mode': all_args.eval_mode,
                        'resource_policy': all_args.resource_policy,
                        'sample_num': int(SAMPLE_NUM),
                        'count': int(count),
                        'avg_cmax': float(avg_cmax),
                        'std_cmax': float(std_cmax),
                        'avg_time': float(avg_time),
                        'avg_gap': float(avg_gap),
                        'cases': case_results,
                    }, f, indent=2, ensure_ascii=False)
    print(f"{'-'*60}")

if __name__ == "__main__":
    main(sys.argv[1:])
