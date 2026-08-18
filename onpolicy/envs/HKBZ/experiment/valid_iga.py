#!/usr/bin/env python
import sys
import os
import json
import numpy as np
import time

# ================= 1. 路径与模块引入 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) 
sys.path.insert(0, root_dir)

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv 
from onpolicy.envs.HKBZ.experiment.eval_common import legal_plane_candidates

# 💡 将 NSGA2 替换为单目标的 GA (Genetic Algorithm)
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback # <--- 引入 Callback 用于控制时间

# ================= 2. 染色体解码与策略执行 =================
def GA_Policy(env, info, job_priorities, site_priorities):
    """
    Decode genes through the environment's authoritative joint-action masks.

    The environment owns all temporal semantics (including post-service
    staging, the global departure barrier and the runway/R014 requirement).
    Reconstructing legality from ``Plane.get_avail_jobs`` would silently use
    the pre-departure semantics and can leave an otherwise feasible chromosome
    stuck after service completion.
    """
    active_agents = np.asarray(info['active_agents'], dtype=bool)
    preferences = {}
    priority_keys = {}

    for plane in env.planes.values():
        pid = int(plane.code.split('_')[-1])
        if pid >= active_agents.size or not active_agents[pid]:
            continue
        candidates = []
        for candidate in legal_plane_candidates(env, pid):
            candidate = dict(candidate)
            candidate['score'] = float(
                job_priorities[pid, candidate['job_idx']]
                + site_priorities[pid, candidate['site_idx']]
            )
            candidates.append(candidate)
        if not candidates:
            raise RuntimeError(
                f'Active plane {pid} has no legal joint action at step '
                f'{env.steps}.'
            )
        candidates.sort(
            key=lambda item: (
                -item['score'], item['job_idx'], item['site_idx']
            )
        )
        preferences[pid] = candidates
        priority_keys[pid] = -candidates[0]['score']

    # A greedy global candidate list can strand a later plane even when a
    # collision-free assignment exists.  Augmenting-path matching preserves
    # each plane's chromosome ordering while guaranteeing unique sites.
    site_matches = {}

    def augment(pid, seen_sites):
        for candidate in preferences[pid]:
            site_idx = candidate['site_idx']
            if site_idx in seen_sites:
                continue
            seen_sites.add(site_idx)
            previous = site_matches.get(site_idx)
            if previous is None or augment(previous[0], seen_sites):
                site_matches[site_idx] = (pid, candidate)
                return True
        return False

    pid_order = sorted(
        preferences, key=lambda pid: (priority_keys[pid], pid)
    )
    for pid in pid_order:
        if not augment(pid, set()):
            counts = {
                active_pid: len(items)
                for active_pid, items in preferences.items()
            }
            raise RuntimeError(
                'No collision-free site matching exists for active planes; '
                f'failed_pid={pid}, candidate_counts={counts}.'
            )

    selected = {
        pid: candidate for pid, candidate in site_matches.values()
    }
    if set(selected) != set(preferences):
        raise RuntimeError(
            f'Incomplete action matching: selected={sorted(selected)}, '
            f'active={sorted(preferences)}.'
        )
    actions = np.full((env.n_agents, 2), -1, dtype=np.int32)
    for pid, candidate in selected.items():
        actions[pid, 0] = candidate['op_global_idx']
        actions[pid, 1] = candidate['site_idx']
    return actions


# ================= 3. 定义 IGA 单目标优化问题 =================
class AircraftSchedulingProblem(ElementwiseProblem):
    def __init__(self, env_config, global_start_time, max_time_seconds=1800):
        self.env_config = env_config
        self.global_start_time = global_start_time
        self.max_time_seconds = max_time_seconds
        
        temp_env = AircraftScheduleEnv(env_config)
        self.n_agents = temp_env.n_agents
        self.n_jobs = len(temp_env.job_code_list)
        self.n_sites = len(temp_env.site_code_list)
        # A time limit can stop pymoo in the middle of a generation.  Keep an
        # explicit anytime incumbent so extending the budget cannot forget a
        # feasible solution already evaluated in the same run.
        self.best_feasible_makespan = np.inf
        self.best_feasible_chromosome = None
        
        n_var = self.n_agents * self.n_jobs + self.n_agents * self.n_sites
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0,
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
            # 🚨 【微观锁】：每走一步都查表，超时直接斩断，绝不多算一秒
            if time.time() - self.global_start_time > self.max_time_seconds:
                out["F"] = [env.total_time + 100000] # 超时直接给极高惩罚，作为废解
                return
                
            actions = GA_Policy(env, info, job_priorities, site_priorities)
            
            try:
                obs, rewards, done, info = env.step(actions)
            except Exception:
                out["F"] = [env.total_time + 100000]
                return
                
            step_count += 1
            
        f1_makespan = env.total_time
        
        if not np.all(done):
            f1_makespan += 100000 
            
        out["F"] = [f1_makespan]
        if (
            np.isfinite(f1_makespan)
            and f1_makespan < 100000.0
            and f1_makespan < self.best_feasible_makespan
        ):
            self.best_feasible_makespan = float(f1_makespan)
            self.best_feasible_chromosome = np.asarray(
                x, dtype=np.float64
            ).reshape(-1).copy()


# ================= 新增：超时安全拦截器 =================
class TimeLimitCallback(Callback):
    """
    监控进化过程的时间。如果超过最大限制，将强制终止算法
    保留找到的当前最优解，避免被大代数任务无限卡死。
    """
    def __init__(self, global_start_time, max_time_seconds):
        super().__init__()
        self.global_start_time = global_start_time
        self.max_time_seconds = max_time_seconds

    def notify(self, algorithm):
        elapsed_time = time.time() - self.global_start_time
        if elapsed_time > self.max_time_seconds:
            # 发送强制终止信号给 pymoo 内部控制器
            algorithm.termination.force_termination = True


# ================= 4. 封装单次评估逻辑 =================
def test_iga_on_case(
    case_path,
    max_time_seconds=1800,
    pop_size=20,
    n_gen=20,
    seed=1,
    return_solution=False,
):
    """Run IGA with an explicit anytime budget and optionally return genes."""
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
        'force_chosen': [-1, '', 0],
    }
    max_time_seconds = float(max_time_seconds)
    if max_time_seconds <= 0.0:
        raise ValueError('max_time_seconds must be positive.')
    pop_size = int(pop_size)
    n_gen = int(n_gen)
    seed = int(seed)
    if pop_size <= 1 or n_gen <= 0:
        raise ValueError('IGA pop_size must exceed one and n_gen must be positive.')

    global_start_time = time.time()
    start_cpu = time.process_time()
    try:
        problem = AircraftSchedulingProblem(
            config,
            global_start_time,
            max_time_seconds=max_time_seconds,
        )
    except Exception as error:
        print(
            f"环境初始化失败 (Case: {os.path.basename(case_path)}): {error}"
        )
        if return_solution:
            return {
                'makespan': None,
                'cpu_seconds': None,
                'wall_seconds': time.time() - global_start_time,
                'chromosome': None,
                'error': f'{type(error).__name__}: {error}',
            }
        return None, None

    algorithm = GA(
        pop_size=pop_size,
        eliminate_duplicates=True,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
    )
    time_limit_cb = TimeLimitCallback(
        global_start_time,
        max_time_seconds=max_time_seconds,
    )
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        callback=time_limit_cb,
        seed=seed,
        verbose=False,
    )
    cpu_time = time.process_time() - start_cpu
    wall_time = time.time() - global_start_time
    best_cmax = (
        float(np.asarray(res.F).reshape(-1)[0])
        if res.F is not None else None
    )
    chromosome = (
        np.asarray(res.X, dtype=np.float64).reshape(-1)
        if res.X is not None else None
    )
    incumbent_preserved = False
    if (
        problem.best_feasible_chromosome is not None
        and (
            best_cmax is None
            or not np.isfinite(best_cmax)
            or problem.best_feasible_makespan < best_cmax
        )
    ):
        best_cmax = float(problem.best_feasible_makespan)
        chromosome = problem.best_feasible_chromosome.copy()
        incumbent_preserved = True
    if not return_solution:
        return best_cmax, cpu_time
    return {
        'makespan': best_cmax,
        'cpu_seconds': float(cpu_time),
        'wall_seconds': float(wall_time),
        'chromosome': (
            chromosome.tolist() if chromosome is not None else None
        ),
        'n_agents': int(problem.n_agents),
        'n_jobs': int(problem.n_jobs),
        'n_sites': int(problem.n_sites),
        'pop_size': pop_size,
        'n_gen': n_gen,
        'seed': seed,
        'time_budget_seconds': max_time_seconds,
        'incumbent_preserved': incumbent_preserved,
    }


# ================= 5. 批量执行与指标计算 =================
if __name__ == "__main__":
    dataset_test_dir = "/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/dataset/fjsp_v2_t480_v60_test60/test"
    
    if not os.path.exists(dataset_test_dir):
        print(f"❌ 找不到测试集目录: {dataset_test_dir}")
        sys.exit(1)

    case_folders = sorted([d for d in os.listdir(dataset_test_dir) 
                           if os.path.isdir(os.path.join(dataset_test_dir, d)) and d.startswith('case_')])
    
    # 📝 填入您的 MAPPO 或者最优解基准，用于计算 IGA 的 Gap(%)
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

    all_cmax_results = {}
    
    metrics = {'c_max_sum': 0.0, 'cpu_sum': 0.0, 'gap_sum': 0.0, 'valid_count': 0, 'gap_count': 0}

    print(f"\n🚀 开始 IGA 批量评估，共检测到 {len(case_folders)} 个测试用例...")
    print("⚠️ 提示：进化算法计算量较大，已加装全局 1800 秒超时安全锁。")
    print("="*60)

    for case_name in case_folders:
        case_path = os.path.join(dataset_test_dir, case_name)
        print(f"⚙️ 正在求解用例: {case_name} ...", end="", flush=True)
        
        c_max, cpu_time = test_iga_on_case(case_path)
        
        if c_max is not None:
            all_cmax_results[case_name] = c_max
            metrics['c_max_sum'] += c_max
            metrics['cpu_sum'] += cpu_time
            metrics['valid_count'] += 1
            
            optimal_val = KNOWN_OPTIMAL_CMAX.get(case_name)
            if optimal_val:
                # 计算 IGA 相对于最优解的 Gap
                gap = ((c_max - optimal_val) / optimal_val) * 100.0
                metrics['gap_sum'] += gap
                metrics['gap_count'] += 1
                print(f" 完成！最优 C_max = {c_max:.1f}s | CPU = {cpu_time:.2f}s | Gap = {gap:.2f}%")
            else:
                print(f" 完成！最优 C_max = {c_max:.1f}s | CPU = {cpu_time:.2f}s | Gap = N/A")
        else:
            print(" 失败或由于超时等原因无合法解！跳过该样本。")

    # ================= 6. 打印最终结果与自动生成的代码 =================
    if metrics['valid_count'] > 0:
        # 新增标准差计算
        cmax_list = list(all_cmax_results.values())
        avg_cmax = np.mean(cmax_list)
        std_cmax = np.std(cmax_list)
        
        avg_cpu = metrics['cpu_sum'] / metrics['valid_count']
        avg_gap = (metrics['gap_sum'] / metrics['gap_count']) if metrics['gap_count'] > 0 else 0.0
        
        print("\n\n" + "="*60)
        print("Proposed Method")
        print(f"{'-'*60}")
        # 拓宽了列距以容纳标准差显示
        print(f"{'':<15} {'C_max (± Std)':<20} {'CPU (s)':<10} {'Gap (%)':<10}")
        print(f"{'-'*60}")
        
        gap_display = f"{avg_gap:.2f}" if metrics['gap_count'] > 0 else "N/A"
        cmax_display = f"{avg_cmax:.1f} ± {std_cmax:.1f}"
        print(f"{'IGA':<15} {cmax_display:<20} {avg_cpu:<10.2f} {gap_display:<10}")
        print(f"{'-'*60}")
        
        print("\n\n" + "-"*50)
        print("✂️ 下方是自动生成的字典，您可以直接复制去覆盖其他脚本中的 KNOWN_OPTIMAL_CMAX：")
        print("-" * 50)
        print("KNOWN_OPTIMAL_CMAX = {")
        for case, val in all_cmax_results.items():
            print(f"    '{case}': {val:.1f},")
        print("}")
        print("-" * 50)
