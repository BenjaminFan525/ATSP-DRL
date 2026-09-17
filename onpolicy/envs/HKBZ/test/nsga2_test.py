import os
import sys
import numpy as np
import time

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
from pymoo.visualization.scatter import Scatter

# ================= 2. 染色体解码与策略执行 =================
def GA_Policy(env, info, job_priorities, site_priorities):
    """
    根据 NSGA-II 传入的基因权重（优先级矩阵），为当前活跃的智能体分配动作。
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
        
        # 本地作业候选
        if current_site != 'Z' and len(current_site_jobs) > 0:
            for j_code in current_site_jobs:
                j_idx = env.job_code_list.index(j_code)
                s_idx = env.site_code_list.index(current_site)
                score = job_priorities[pid, j_idx] + site_priorities[pid, s_idx]
                candidates.append({'pid': pid, 'site': current_site, 'job': j_code, 'score': score, 'is_move': False})
                
        # 转运作业候选
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

    # 3. 贪婪动作分配防碰撞
    assigned_pids = set()
    claimed_sites = set()
    
    for cand in candidates:
        pid, site, job = cand['pid'], cand['site'], cand['job']
        if pid in assigned_pids: continue
        if cand['is_move'] and site in claimed_sites: continue
            
        assigned_pids.add(pid)
        if cand['is_move']: claimed_sites.add(site)
            
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
        # 临时初始化一个环境以获取维度信息
        temp_env = AircraftScheduleEnv(env_config)
        self.n_agents = temp_env.n_agents
        self.n_jobs = len(temp_env.job_code_list)
        self.n_sites = len(temp_env.site_code_list)
        
        # 基因长度 = 飞机数量 * (工序种类数 + 机位数量)
        n_var = self.n_agents * self.n_jobs + self.n_agents * self.n_sites
        
        super().__init__(n_var=n_var,
                         n_obj=2, # 两个优化目标：C_max 和 拥堵成本
                         n_ieq_constr=0,
                         xl=np.zeros(n_var), # 权重下界 0.0
                         xu=np.ones(n_var))  # 权重上界 1.0

    def _evaluate(self, x, out, *args, **kwargs):
        """
        核心评估函数：使用当前染色体 x 作为策略在环境中跑完一个完整的 Episode
        """
        # 1. 将 1D 染色体还原为 2D 优先级矩阵
        job_genes = x[:self.n_agents * self.n_jobs]
        site_genes = x[self.n_agents * self.n_jobs:]
        
        job_priorities = job_genes.reshape((self.n_agents, self.n_jobs))
        site_priorities = site_genes.reshape((self.n_agents, self.n_sites))
        
        # 2. 初始化环境
        env = AircraftScheduleEnv(self.env_config)
        env.use_domain_rand = False # 确保评估过程是确定性的
        obs, done, info = env.reset()
        
        step_count = 0
        while not np.all(done) and step_count < 2000:
            actions = GA_Policy(env, info, job_priorities, site_priorities)
            obs, rewards, done, info = env.step(actions)
            step_count += 1
            
        # 3. 提取目标值
        # 目标1：最小化最大完工时间 (Makespan)
        f1_makespan = env.total_time
        
        # 目标2：最小化总等待时间与运输时间之和 (分析 trajectory_log)
        f2_congestion = 0.0
        for log in env.trajectory_log:
            f2_congestion += log.get('waiting_time', 0)
            f2_congestion += log.get('trans_time', 0)
            
        # 极端情况惩罚 (如果发生死锁未完成)
        if not np.all(done):
            f1_makespan += 100000 
            f2_congestion += 100000
            
        out["F"] = [f1_makespan, f2_congestion]


# ================= 4. 运行 NSGA-II 主程序 =================
if __name__ == '__main__':
    config = {
        'batch_num': 1,               
        'plane_num_per_batch': 12,    
        'n_agents': 12,               
        'jobs_path': 'utils/config/jobs.json',
        'fixed_res_path': 'utils/config/fixed_resources.json',
        'mobile_res_path': 'utils/config/mobile_resources.json',
        'sites_path': 'utils/config/sites.json',
        'seed': 42,
    }
    
    print(">>> 正在初始化 NSGA-II 算法...")
    problem = AircraftSchedulingProblem(config)
    
    algorithm = NSGA2(
        pop_size=50, # 种群规模 (可调)
        n_offsprings=20,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15), # 模拟二进制交叉
        mutation=PM(eta=20),             # 多项式变异
        eliminate_duplicates=True
    )
    
    print(">>> 开始多目标进化求解 (预计需要一些时间，因每一代都要完整推演环境)...")
    start_time = time.time()
    
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 30), # 进化代数 (建议实际使用时设为 100-200)
                   seed=1,
                   verbose=True)  # 开启日志输出，观察每一代的收敛情况
                   
    print(f"\n>>> 求解完成！总耗时: {time.time() - start_time:.2f} 秒")
    
    # 获取帕累托前沿解
    front_f = res.F
    front_x = res.X
    
    print(f"共找到 {len(front_f)} 个非支配帕累托最优解。")
    print("前沿解目标值展示 (Makespan, Total_Congestion_Time):")
    for idx, obj in enumerate(front_f):
        print(f"解 {idx}: C_max = {obj[0]:.0f} s, 拥堵 = {obj[1]:.0f} s")
        
    # 可视化帕累托前沿 (如果环境支持 GUI 或导出)
    plot = Scatter(title="NSGA-II Pareto Front")
    plot.add(front_f, color="red")
    # plot.show() # 如果在服务器上运行，请注释掉此行并使用 plot.save('pareto.png')
    plot.save('pareto_front.png')
    print("帕累托前沿图已保存为 pareto_front.png")