import os
import sys
import numpy as np
import time
import json
import argparse

# ================= 1. 路径与模块引入 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) 
sys.path.insert(0, root_dir)

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv 
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

# ================= 2. 染色体解码与策略执行 =================
def GA_Policy(env, info, job_priorities, site_priorities):
    """
    根据 NSGA-II 传入的基因权重，为当前活跃的智能体分配动作。
    """
    n_agents = env.n_agents
    actions = np.zeros((n_agents, 2), dtype=np.int32)
    actions[:] = [-1, -1]
    active_agents = info['active_agents']
    
    candidates = []
    pid_to_plane = {}
    avail_sites = env.get_avail_sites()
    
    # 1. 收集合法候选动作池
    for plane in env.planes.values():
        pid = int(plane.code.split('_')[2])
        pid_to_plane[pid] = plane
        
        if not active_agents[pid]: continue
            
        current_site = plane.site.code
        current_site_jobs = plane.get_avail_jobs(plane.site)
        
        if current_site != 'Z' and len(current_site_jobs) > 0:
            for j_code in current_site_jobs:
                j_idx = env.job_code_list.index(j_code)
                s_idx = env.site_code_list.index(current_site)
                score = job_priorities[pid, j_idx] + site_priorities[pid, s_idx]
                candidates.append({'pid': pid, 'site': current_site, 'job': j_code, 'score': score, 'is_move': False})
                
        else:
            for s_code in avail_sites:
                next_site_jobs = plane.get_avail_jobs(env.sites[s_code])
                for j_code in next_site_jobs:
                    j_idx = env.job_code_list.index(j_code)
                    s_idx = env.site_code_list.index(s_code)
                    score = job_priorities[pid, j_idx] + site_priorities[pid, s_idx]
                    candidates.append({'pid': pid, 'site': s_code, 'job': j_code, 'score': score, 'is_move': True})

    # 2. 根据基因解码出的 Score 进行降序排列
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # 3. 贪婪动作分配 (已同步机位抢占修复)
    assigned_pids = set()
    claimed_sites = set()
    
    for cand in candidates:
        pid, site, job = cand['pid'], cand['site'], cand['job']
        if pid in assigned_pids: continue
        if site in claimed_sites: continue # 修复漏洞：只要该机位被占用，无论是飞过去还是在原地，都锁定
            
        assigned_pids.add(pid)
        claimed_sites.add(site)
            
        job_idx = env.job_code_list.index(job) + pid * len(env.job_code_list)
        site_idx = env.site_code_list.index(site)
        actions[pid][0] = job_idx
        actions[pid][1] = site_idx

    # 4. 兜底容错
    for pid in range(n_agents):
        if active_agents[pid] and pid not in assigned_pids:
            plane = pid_to_plane.get(pid)
            if plane:
                job_idx = pid * len(env.job_code_list) 
                site_idx = env.site_code_list.index(plane.site.code)
                actions[pid][0] = job_idx
                actions[pid][1] = site_idx
                
    return actions


# ================= 3. 定义 NSGA-II 优化问题 =================
class AircraftSchedulingProblem(ElementwiseProblem):
    def __init__(self, env_config):
        self.env_config = env_config
        temp_env = AircraftScheduleEnv(env_config)
        self.n_agents = temp_env.n_agents
        self.n_jobs = len(temp_env.job_code_list)
        self.n_sites = len(temp_env.site_code_list)
        
        n_var = self.n_agents * self.n_jobs + self.n_agents * self.n_sites
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=0,
                         xl=np.zeros(n_var), xu=np.ones(n_var))

    def _evaluate(self, x, out, *args, **kwargs):
        job_genes = x[:self.n_agents * self.n_jobs]
        site_genes = x[self.n_agents * self.n_jobs:]
        
        job_priorities = job_genes.reshape((self.n_agents, self.n_jobs))
        site_priorities = site_genes.reshape((self.n_agents, self.n_sites))
        
        env = AircraftScheduleEnv(self.env_config)
        env.use_domain_rand = False 
        obs, done, info = env.reset()
        
        step_count = 0
        while not np.all(done) and step_count < 2000:
            actions = GA_Policy(env, info, job_priorities, site_priorities)
            obs, rewards, done, info = env.step(actions)
            step_count += 1
            
        f1_makespan = env.total_time
        f2_congestion = 0.0
        for log in env.trajectory_log:
            f2_congestion += log.get('waiting_time', 0)
            f2_congestion += log.get('trans_time', 0)
            
        if not np.all(done):
            f1_makespan += 100000 
            f2_congestion += 100000
            
        out["F"] = [f1_makespan, f2_congestion]


# ================= 4. 封装单次评估逻辑 =================
def test_nsga2_on_case(case_path, pop_size=20, generations=20, seed=1):
    flights_path = os.path.join(case_path, 'flights.json')
    n_agents = 12 
    if os.path.exists(flights_path):
        try:
            with open(flights_path, 'r', encoding='utf-8') as f:
                n_agents = len(json.load(f)) 
        except Exception:
            pass

    config = {
        'batch_num': 1,               
        'plane_num_per_batch': n_agents,    
        'n_agents': n_agents,               
        'jobs_path': os.path.join(case_path, 'job.json'),
        'fixed_res_path': os.path.join(case_path, 'fixed_resources.json'),
        'mobile_res_path': os.path.join(case_path, 'mobile_resources.json'),
        'sites_path': os.path.join(case_path, 'sites.json'),
        'flights_path': flights_path,
        'seed': 42,
        'interfere': [-1, [], 0],     
        'force_chosen': [-1, '', 0]   
    }
    
    try:
        problem = AircraftSchedulingProblem(config)
    except Exception as e:
        print(f"环境初始化失败 (Case: {os.path.basename(case_path)}): {e}")
        return None, None
        
    # 参数提示：正式跑论文数据时，建议将 pop_size 调至 50-100，n_gen 调至 100-200
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    start_cpu = time.process_time()
    
    # verbose 设为 False 以免批量运行时日志刷屏
    res = minimize(problem, algorithm, ('n_gen', generations), seed=seed,
                   verbose=False)
    
    end_cpu = time.process_time()
    cpu_time = end_cpu - start_cpu
    
    # 取帕累托前沿中 C_max (目标1) 最小的那个解作为最终表现
    best_cmax = np.min(res.F[:, 0])
    
    return best_cmax, cpu_time


# ================= 5. 批量执行与指标计算 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the NSGA-II baseline.")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(root_dir, "onpolicy/envs/HKBZ/dataset/test_large"),
    )
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    dataset_test_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    
    if not os.path.exists(dataset_test_dir):
        print(f"❌ 找不到测试集目录: {dataset_test_dir}")
        sys.exit(1)

    case_folders = sorted([d for d in os.listdir(dataset_test_dir) 
                           if os.path.isdir(os.path.join(dataset_test_dir, d)) and d.startswith('case_')])
    
    # 📝 【计算 GAP 所需的字典】
    # 如果此时您是拿 NSGA-II 本身作为最优基准，这里可以全填 1，跑完后直接用底部的输出替换它。
    # 填入已知的最优解格式：'case_01': 1100.0
    KNOWN_OPTIMAL_CMAX = {} 

    all_cmax_results = {} # 用于在最后生成字典代码
    
    metrics = {'c_max_sum': 0.0, 'cpu_sum': 0.0, 'gap_sum': 0.0, 'valid_count': 0, 'gap_count': 0}

    print(f"\n🚀 开始 NSGA-II 批量评估，共检测到 {len(case_folders)} 个测试用例...")
    print("⚠️ 提示：进化算法计算量较大，多样本测试可能需要较长时间。")
    print("="*60)

    for case_name in case_folders:
        case_path = os.path.join(dataset_test_dir, case_name)
        print(f"⚙️ 正在求解用例: {case_name} ...", end="", flush=True)
        
        c_max, cpu_time = test_nsga2_on_case(
            case_path,
            pop_size=args.population_size,
            generations=args.generations,
            seed=args.seed,
        )
        
        if c_max is not None:
            all_cmax_results[case_name] = c_max
            metrics['c_max_sum'] += c_max
            metrics['cpu_sum'] += cpu_time
            metrics['valid_count'] += 1
            print(f" 完成！最优 C_max = {c_max:.1f}s | CPU = {cpu_time:.2f}s | Gap = N/A")
        else:
            print(" 失败！跳过该样本。")

    # ================= 6. 打印最终结果与自动生成的代码 =================
    if metrics['valid_count'] > 0:
        avg_cmax = metrics['c_max_sum'] / metrics['valid_count']
        avg_cpu = metrics['cpu_sum'] / metrics['valid_count']
        avg_gap = (metrics['gap_sum'] / metrics['gap_count']) if metrics['gap_count'] > 0 else 0.0
        
        print("\n\n" + "="*50)
        print("🏆 NSGA-II 总体性能评估均值")
        print("="*50)
        print(f"| {'Method':<10} | {'C_max':<10} | {'CPU (s)':<10} | {'Gap (%)':<10} |")
        print(f"|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")
        
        gap_display = f"{avg_gap:.2f}" if metrics['gap_count'] > 0 else "N/A"
        print(f"| {'NSGA-II':<10} | {avg_cmax:<10.1f} | {avg_cpu:<10.2f} | {gap_display:<10} |")
        
        # ================= 生成供直接复制的 KNOWN_OPTIMAL_CMAX 字典 =================
        print("\n\n" + "-"*50)
        print("✂️ 下方是自动生成的字典，您可以直接复制去覆盖启发式脚本中的 KNOWN_OPTIMAL_CMAX：")
        print("-" * 50)
        print("KNOWN_OPTIMAL_CMAX = {")
        for case, val in all_cmax_results.items():
            print(f"    '{case}': {val:.1f},")
        print("}")
        print("-" * 50)
