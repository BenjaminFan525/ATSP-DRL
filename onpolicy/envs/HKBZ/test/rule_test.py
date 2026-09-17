import os
import sys
import time
import random
import numpy as np

# ================= 1. 路径与模块修复 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) 
sys.path.insert(0, root_dir)

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv 

# ================= 2. 启发式综合调度策略 (Heuristic Policy) =================
def Heuristic_Policy(env, info, rule="Random"):
    """
    支持五种规则的集中式启发式策略：Random, FIFO, SPT, LPT, MOR。
    通过构建全局候选池并按优先级贪婪分配，完美解决多机位抢占冲突。
    """
    n_agents = env.n_agents
    actions = np.zeros((n_agents, 2), dtype=np.int32)
    actions[:] = [-1, -1] # 默认给不活跃的智能体发送占位掩码
    active_agents = info['active_agents']
    
    # 1. 收集所有活跃飞机的合法动作候选 (Candidate Pool)
    candidates = []
    pid_to_plane = {}
    avail_sites = env.get_avail_sites() # 全局空闲机位
    
    for plane in env.planes.values():
        pid = int(plane.code.split('_')[2])
        pid_to_plane[pid] = plane
        
        if not active_agents[pid]:
            continue
            
        current_site = plane.site.code
        current_site_jobs = plane.get_avail_jobs(plane.site)
        
        # 分支 A: 如果在当前机位有活干（排除降落跑道Z）
        if current_site != 'Z' and len(current_site_jobs) > 0:
            for j_code in current_site_jobs:
                candidates.append({
                    'pid': pid,
                    'site': current_site,
                    'job': j_code,
                    'p_time': env.jobs[j_code].time or 0,
                    'rem_ops': len(plane.left_jobs),
                    'wait_time': plane.waiting_time,
                    'is_move': False,
                    'rand': random.random() # 随机平局打破器
                })
        # 分支 B: 当前机位干不了活，必须转运
        else:
            for s_code in avail_sites:
                next_site_jobs = plane.get_avail_jobs(env.sites[s_code])
                for j_code in next_site_jobs:
                    candidates.append({
                        'pid': pid,
                        'site': s_code,
                        'job': j_code,
                        'p_time': env.jobs[j_code].time or 0,
                        'rem_ops': len(plane.left_jobs),
                        'wait_time': plane.waiting_time,
                        'is_move': True,
                        'rand': random.random()
                    })

    # 2. 根据启发式规则对候选动作进行全局排序
    if rule == "Random":
        candidates.sort(key=lambda x: x['rand'])
    elif rule == "FIFO":
        # 优先调度等待时间最长的飞机
        candidates.sort(key=lambda x: (-x['wait_time'], x['rand']))
    elif rule == "SPT":
        # 优先调度加工时间最短的工序
        candidates.sort(key=lambda x: (x['p_time'], x['rand']))
    elif rule == "LPT":
        # 优先调度加工时间最长的工序
        candidates.sort(key=lambda x: (-x['p_time'], x['rand']))
    elif rule == "MOR":
        # 优先调度剩余工序数量最多的飞机
        candidates.sort(key=lambda x: (-x['rem_ops'], x['rand']))
    else:
        raise ValueError(f"不支持的调度规则: {rule}")

    # 3. 贪婪动作分配 (解决多架飞机抢同一个机位的冲突)
    assigned_pids = set()
    claimed_sites = set()
    
    for cand in candidates:
        pid = cand['pid']
        site = cand['site']
        job = cand['job']
        
        if pid in assigned_pids:
            continue # 该飞机已经分配过最高优先级的动作了
            
        if cand['is_move'] and site in claimed_sites:
            continue # 目标机位已经被优先级更高的其他飞机抢占了
            
        # 确认分配该动作
        assigned_pids.add(pid)
        if cand['is_move']:
            claimed_sites.add(site)
            
        # 转换为 RL 环境可接受的 MultiDiscrete 离散索引
        job_idx = env.job_code_list.index(job) + pid * len(env.job_code_list)
        site_idx = env.site_code_list.index(site)
        
        actions[pid][0] = job_idx
        actions[pid][1] = site_idx

    # 4. 兜底容错处理：活跃但没有合法动作可分的飞机 (例如全场机位满载)
    for pid in range(n_agents):
        if active_agents[pid] and pid not in assigned_pids:
            plane = pid_to_plane.get(pid)
            if plane:
                # 迫使其在原地发呆等待，下发当前机位与 0 号工序的虚拟动作
                job_idx = pid * len(env.job_code_list) 
                site_idx = env.site_code_list.index(plane.site.code)
                actions[pid][0] = job_idx
                actions[pid][1] = site_idx
                
    return actions


# ================= 3. 仿真主循环 =================
def test_aircraft_schedule(rule="SPT"):
    config = {
        'batch_num': 1,               
        'plane_num_per_batch': 12,    
        'n_agents': 12,               
        'jobs_path': 'utils/config/jobs.json',
        'fixed_res_path': 'utils/config/fixed_resources.json',
        'mobile_res_path': 'utils/config/mobile_resources.json',
        'sites_path': 'utils/config/sites.json',
        'seed': 42,
        'interfere': [-1, [], 0],     
        'force_chosen': [-1, '', 0]   
    }
    
    print(f"\n>>> 正在初始化航空调度环境...")
    try:
        env = AircraftScheduleEnv(config)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return

    print(f">>> 开始执行 [{rule}] 启发式规则测试...")
    start_real_time = time.time()
    
    env.use_domain_rand = False
    obs, done, info = env.reset()
    
    step_count = 0
    total_rewards = np.zeros(config['n_agents'])

    while not np.all(done):
        # 传入指定的启发式规则
        actions = Heuristic_Policy(env, info, rule=rule)

        active_pids = [pid for pid, is_active in enumerate(info['active_agents']) if is_active]
        if active_pids:
            print(f"[Step {step_count} | Env Time {env.total_time}s] 需要决策的飞机: {active_pids}")
            for pid in active_pids:
                j_code = env.job_code_list[actions[pid][0] - pid * len(env.job_code_list)]
                s_code = env.site_code_list[actions[pid][1]]
                print(f"  -> 飞机 {pid} 决策: 目标机位={s_code}, 意图作业={j_code}")

        obs, rewards, done, info = env.step(actions)
        total_rewards += rewards.flatten()
        step_count += 1
        
        if step_count > 2000:
            print("\n【警告】达到最大步数 2000，强制停止。")
            break

    end_real_time = time.time()
    
    print("\n" + "="*50)
    print(f">>> [{rule}] 规则仿真测试结束！")
    print(f"实际运算耗时: {end_real_time - start_real_time:.4f} 秒")
    print(f"总决策步数: {step_count}")
    print(f"环境推演总耗时 (完工时间 C_max): {env.total_time} 秒 ({env.total_time/3600:.2f} 小时)")
    print(f"所有任务是否均成功完成？: {done}")
    print("="*50)

if __name__ == "__main__":
    # 你可以在这里一键切换不同的策略规则进行性能对比 (C_max越小越好)
    rules_to_test = ["Random", "FIFO", "SPT", "LPT", "MOR"]
    
    # 这里演示运行 SPT，你可以使用 for 循环跑完所有的基线
    test_aircraft_schedule(rule="MOR")