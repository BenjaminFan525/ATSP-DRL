import os
import sys
import time
import json
import random
import numpy as np
import argparse

# ================= 1. 路径与模块修复 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) 
sys.path.insert(0, root_dir)

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv 

# ================= 2. 启发式综合调度策略 =================
def Heuristic_Policy(env, info, rule="Random"):
    n_agents = env.n_agents
    actions = np.zeros((n_agents, 2), dtype=np.int32)
    actions[:] = [-1, -1] 
    active_agents = info['active_agents']
    
    candidates = []
    pid_to_plane = {}
    avail_sites = env.get_avail_sites() 
    
    for plane in env.planes.values():
        pid = int(plane.code.split('_')[2])
        pid_to_plane[pid] = plane
        
        if not active_agents[pid]:
            continue
            
        current_site = plane.site.code
        current_site_jobs = plane.get_avail_jobs(plane.site)
        rem_work = sum([plane.jobs[j].time or 0 for j in plane.left_jobs])
        
        if current_site != 'Z' and len(current_site_jobs) > 0:
            for j_code in current_site_jobs:
                candidates.append({
                    'pid': pid, 'site': current_site, 'job': j_code,
                    'p_time': env.jobs[j_code].time or 0, 'rem_work': rem_work,
                    'rem_ops': len(plane.left_jobs), 'wait_time': plane.waiting_time,
                    'is_move': False, 'rand': random.random() 
                })
        else:
            for s_code in avail_sites:
                next_site_jobs = plane.get_avail_jobs(env.sites[s_code])
                for j_code in next_site_jobs:
                    candidates.append({
                        'pid': pid, 'site': s_code, 'job': j_code,
                        'p_time': env.jobs[j_code].time or 0, 'rem_work': rem_work,
                        'rem_ops': len(plane.left_jobs), 'wait_time': plane.waiting_time,
                        'is_move': True, 'rand': random.random()
                    })

    if rule == "Random":
        candidates.sort(key=lambda x: x['rand'])
    elif rule == "FIFO":
        candidates.sort(key=lambda x: (-x['wait_time'], x['rand']))
    elif rule == "SPT":
        candidates.sort(key=lambda x: (x['p_time'], x['rand']))
    elif rule == "LPT":
        candidates.sort(key=lambda x: (-x['p_time'], x['rand']))
    elif rule == "MWKR":
        candidates.sort(key=lambda x: (-x['rem_work'], x['rand']))
    else:
        raise ValueError(f"不支持的调度规则: {rule}")

    assigned_pids = set()
    claimed_sites = set()
    
    for cand in candidates:
        pid = cand['pid']
        site = cand['site']
        job = cand['job']
        
        if pid in assigned_pids or site in claimed_sites:
            continue 
            
        assigned_pids.add(pid)
        claimed_sites.add(site)
            
        job_idx = env.job_code_list.index(job) + pid * len(env.job_code_list)
        site_idx = env.site_code_list.index(site)
        
        actions[pid][0] = job_idx
        actions[pid][1] = site_idx

    for pid in range(n_agents):
        if active_agents[pid] and pid not in assigned_pids:
            plane = pid_to_plane.get(pid)
            if plane:
                job_idx = pid * len(env.job_code_list) 
                site_idx = env.site_code_list.index(plane.site.code)
                actions[pid][0] = job_idx
                actions[pid][1] = site_idx
                
    return actions

# ================= 3. 动态配置仿真主循环 =================
def test_aircraft_schedule(case_path, rule="SPT"):
    flights_path = os.path.join(case_path, 'flights.json')
    n_agents = 12 
    if os.path.exists(flights_path):
        try:
            with open(flights_path, 'r', encoding='utf-8') as f:
                flights_data = json.load(f)
                n_agents = len(flights_data) 
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
        env = AircraftScheduleEnv(config)
    except Exception as e:
        print(f"环境初始化失败 (Case: {os.path.basename(case_path)}): {e}")
        return None, None

    env.use_domain_rand = False
    obs, done, info = env.reset()
    
    step_count = 0
    start_cpu_time = time.process_time()
    
    while not np.all(done):
        actions = Heuristic_Policy(env, info, rule=rule)
        obs, rewards, done, info = env.step(actions)
        step_count += 1
        
        if step_count > 2000:
            break
            
    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time

    return env.total_time, cpu_time

# ================= 4. 批量执行与指标计算 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dispatching-rule baselines.")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(root_dir, "onpolicy/envs/HKBZ/dataset/test_large"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset_test_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    
    if not os.path.exists(dataset_test_dir):
        print(f"❌ 找不到测试集目录: {dataset_test_dir}")
        sys.exit(1)

    case_folders = sorted([d for d in os.listdir(dataset_test_dir) 
                           if os.path.isdir(os.path.join(dataset_test_dir, d)) and d.startswith('case_')])
    
    rules_to_test = ["FIFO", "SPT", "MWKR"]
    
    # 修改点 1：将 c_max 初始化为列表，以便后续计算标准差
    metrics_sum = {rule: {'c_max_list': [], 'cpu': 0.0, 'gap': 0.0, 'count': 0} for rule in rules_to_test}

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
    
    print(f"\n🚀 开始批量评估，共检测到 {len(case_folders)} 个测试用例。")
    print("="*60)

    for case_name in case_folders:
        case_path = os.path.join(dataset_test_dir, case_name)
        
        optimal_val = KNOWN_OPTIMAL_CMAX.get(case_name)
        if optimal_val is None:
            print(f"⚠️ 警告: 找不到 {case_name} 的已知最优解，跳过该算例的统计。")
            continue
            
        print(f"⚙️ 正在评估用例: {case_name} (已知最优解: {optimal_val}) ...")
        
        for rule in rules_to_test:
            c_max, cpu_time = test_aircraft_schedule(case_path, rule=rule)
            
            if c_max is not None:
                gap = ((c_max - optimal_val) / optimal_val) * 100.0
                
                # 修改点 2：将 c_max 添加到列表中
                metrics_sum[rule]['c_max_list'].append(c_max)
                metrics_sum[rule]['cpu'] += cpu_time
                metrics_sum[rule]['gap'] += gap
                metrics_sum[rule]['count'] += 1

    # ================= 5. 打印最终论文表格格式 =================
    print("\n\n" + "="*60)
    print("🏆 总体性能评估均值 (Average over all test cases)")
    print("="*60)
    
    # 修改点 3：拓宽 C_max 列以容纳标准差，更新表头分隔符
    header = f"| {'Method':<10} | {'C_max (± Std)':<20} | {'CPU (s)':<10} | {'Gap (%)':<10} |"
    divider = f"|{'-'*12}|{'-'*22}|{'-'*12}|{'-'*12}|"
    print(header)
    print(divider)
    
    for rule in rules_to_test:
        count = metrics_sum[rule]['count']
        if count > 0:
            cmax_list = metrics_sum[rule]['c_max_list']
            
            # 修改点 4：利用 numpy 计算均值和标准差
            avg_cmax = np.mean(cmax_list)
            std_cmax = np.std(cmax_list)
            
            avg_cpu = metrics_sum[rule]['cpu'] / count
            avg_gap = metrics_sum[rule]['gap'] / count
            
            cmax_display = f"{avg_cmax:.1f} ± {std_cmax:.1f}"
            print(f"| {rule:<10} | {cmax_display:<20} | {avg_cpu:<10.2f} | {avg_gap:<10.2f} |")
        else:
            print(f"| {rule:<10} | {'N/A':<20} | {'N/A':<10} | {'N/A':<10} |")
