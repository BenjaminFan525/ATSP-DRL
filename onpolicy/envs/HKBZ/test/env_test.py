import os
import sys
import time
import random
import glob
import numpy as np

# ================= 1. 路径与模块修复 =================
# 定位到项目的根目录 (请根据你的实际项目结构调整)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设这个测试脚本放在 onpolicy/envs/HKBZ/tests 目录下
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) 
sys.path.insert(0, root_dir)

# 导入你重构好的环境 (请确保路径匹配你的实际项目)
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv 


# ================= 2. 随机策略 (Random Policy) =================
def random_policy(env, info):
    n_agents = env.n_agents
    # 【修复 1】：将默认值从 zeros(0) 改为 full(-1)。绝对不能发 0！
    actions = np.full((n_agents, 2), -1, dtype=np.int32)
    
    active_agents = info['active_agents']
    avail_sites = env.get_avail_sites()
    
    for plane in env.planes.values():
        pid = int(plane.code.split('_')[2]) 
        
        if not active_agents[pid]:
            actions[pid] = [-1, -1]
            continue
            
        target_job_code = None
        target_site_code = plane.site.code 
        current_site_jobs = plane.get_avail_jobs(plane.site)
        
        if plane.site.code != 'Z' and len(current_site_jobs) > 0:
            target_job_code = random.choice(current_site_jobs)
        else:
            if len(avail_sites) > 0:
                target_site_code = random.choice(avail_sites)
                avail_sites.remove(target_site_code) 
                
                next_site_jobs = plane.get_avail_jobs()
                if len(next_site_jobs) > 0:
                    target_job_code = random.choice(next_site_jobs)

        # 【修复 2】：这里的初始映射也必须是 -1，找不到作业就发 -1 (等待)
        job_idx = -1
        site_idx = -1
        
        if target_job_code and target_job_code in env.job_code_list:
            job_idx = env.job_code_list.index(target_job_code) 
            
        if target_site_code and target_site_code in env.site_code_list:
            site_idx = env.site_code_list.index(target_site_code)
            
        actions[pid][0] = job_idx + pid * len(env.job_code_list)  
        actions[pid][1] = site_idx
                
    return actions


# ================= 3. 仿真主循环 =================
def test_aircraft_schedule():
    # ================= 新增：动态加载数据集算例 =================
    case_dir = "/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/dataset/test_large/case_05"
    print(f">>> 准备加载测试算例: {case_dir}")

    # 环境配置参数：替换为动态加载的算例路径
    config = {
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
    
    print(">>> 正在初始化航空调度环境...")
    try:
        env = AircraftScheduleEnv(config)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return

    print(">>> 开始执行随机探索测试...")
    start_real_time = time.time()
    
    obs, done, info = env.reset()
    
    step_count = 0
    # 从环境获取实际的飞机数量作为奖励累计维度
    total_rewards = np.zeros(env.n_agents)

    while not np.all(done):
        # 1. 使用随机策略生成动作
        actions = random_policy(env, info)

        # 2. 打印当前步的决策日志
        active_pids = [pid for pid, is_active in enumerate(info['active_agents']) if is_active]
        if active_pids:
            print(f"[Step {step_count} | Env Time {env.total_time}s] 需要决策的飞机: {active_pids}")
            for pid in active_pids:
                raw_job_idx = actions[pid][0]
                site_idx = actions[pid][1]
                
                # [修复点] 安全解析 job_code，防止无目标作业时数组负向越界
                actual_job_idx = raw_job_idx - pid * len(env.job_code_list)
                if 0 <= actual_job_idx < len(env.job_code_list) and raw_job_idx != 0:
                    j_code = env.job_code_list[actual_job_idx]
                else:
                    j_code = "等待/无动作"
                    
                s_code = env.site_code_list[site_idx] if site_idx >= 0 else "未知"
                
                print(f"  -> 飞机 {pid} 决策: 目标机位={s_code}, 意图作业={j_code}")

        # 3. 喂给环境进行推演 (环境会自动进行时间跃迁)
        obs, rewards, done, info = env.step(actions)
        
        # 累加奖励 (注意 rewards 的 shape 是 [n_agents, 1])
        total_rewards += rewards.flatten()
        step_count += 1
        
        # 防死锁保护
        if step_count > 2000:
            print("\n【警告】达到最大步数 2000，可能出现逻辑死锁，强制停止。")
            break

    # ================= 4. 结算与分析 =================
    end_real_time = time.time()
    
    print("\n" + "="*50)
    print(">>> 仿真测试结束！")
    print(f"当前测试算例: {case_dir}")
    print(f"总计飞机数量: {env.n_agents} 架")
    print(f"实际运算耗时: {end_real_time - start_real_time:.4f} 秒")
    print(f"总决策步数 (RL Steps): {step_count}")
    print(f"环境推演总耗时: {env.total_time} 秒 ({env.total_time/3600:.2f} 小时)")
    print(f"是否所有飞机都成功完成任务并清场？: {done}")
    
    print("\n[各架飞机的累积奖励/惩罚]")
    for i, rew in enumerate(total_rewards):
        print(f"飞机 {i}: {rew:.2f}")
    print("="*50)

if __name__ == "__main__":
    test_aircraft_schedule()