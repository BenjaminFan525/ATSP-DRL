import os
import sys
import time
import random
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
    """
    基于规则的随机探索策略。
    完全适配新版环境的 [n_agents, 2] Numpy 张量接口和活跃掩码逻辑。
    """
    n_agents = env.n_agents
    actions = np.zeros((n_agents, 2), dtype=np.int32)
    
    active_agents = info['active_agents']
    
    # 获取当前全局可用的机位列表和起飞跑道列表
    avail_sites = env.get_avail_sites()
    takeoff_sites = env.get_avail_takeoff_sites()
    
    for plane in env.planes.values():
        # 获取飞机的全局唯一索引
        pid = int(plane.code.split('_')[2]) 
        
        # 【分支 1】：如果不活跃（比如还没降落、正在作业或运输中），直接发占位动作
        if active_agents[pid]:
            actions[pid] = [0, 0]
            continue
            
        target_job_code = None
        target_site_code = plane.site.code # 默认原地不动
        
        # 尝试获取在当前机位能做的作业
        current_site_jobs = plane.get_avail_jobs(plane.site)
        
        # 【分支 2】：当前机位有能干的活，直接原地开干
        if len(current_site_jobs) > 0:
            target_job_code = random.choice(current_site_jobs)
            
        # 【分支 3】：当前机位干不了活（缺特定固定资源），需要转运
        else:
            if len(avail_sites) > 0:
                # 随机挑一个空闲机位
                target_site_code = random.choice(avail_sites)
                avail_sites.remove(target_site_code) # 选走就从候选池剔除，防止别的飞机也选它
                
                # 顺便预测去了新机位后能干啥，挂上意图
                next_site_jobs = plane.get_avail_jobs(env.sites[target_site_code])
                if len(next_site_jobs) > 0:
                    target_job_code = random.choice(next_site_jobs)

        # 【数字索引映射】：将字符串转换为 RL 网络输出的整数索引
        job_idx = 0
        site_idx = 0
        
        if target_job_code and target_job_code in env.job_code_list:
            job_idx = env.job_code_list.index(target_job_code)
            
        if target_site_code and target_site_code in env.site_code_list:
            site_idx = env.site_code_list.index(target_site_code)
            
        actions[pid][0] = job_idx
        actions[pid][1] = site_idx
        
    # 【清场校验】：如果飞机已经完成了所有作业，指挥它去跑道起飞
    # for plane in env.planes.values():
    #     pid = int(plane.code.split('_')[2])
    #     if plane.is_idle() and plane.is_completed_all_jobs():
    #         if len(takeoff_sites) > 0 and plane.site.code not in takeoff_sites:
    #             target_site_code = takeoff_sites.pop(0)
    #             site_idx = env.site_code_list.index(target_site_code)
    #             actions[pid][0] = 0        # 作业发占位 (没作业了)
    #             actions[pid][1] = site_idx # 机位发跑道索引
                
    return actions


# ================= 3. 仿真主循环 =================
def test_aircraft_schedule():
    # 环境配置参数
    config = {
        'batch_num': 1,               # 测试 1 个批次
        'plane_num_per_batch': 12,    # 这 1 个批次有 12 架飞机
        'n_agents': 12,               # 总智能体数量 = 1 * 12
        'jobs_path': 'utils/config/jobs.json',
        'fixed_res_path': 'utils/config/fixed_resources.json',
        'mobile_res_path': 'utils/config/mobile_resources.json',
        'sites_path': 'utils/config/sites.json',
        'seed': 42,
        'interfere': [-1, [], 0],     # 不开启干涉
        'force_chosen': [-1, '', 0]   # 不开启强制动作
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
    total_rewards = np.zeros(config['n_agents'])

    while not np.all(done):
        # 1. 使用随机策略生成动作
        actions = random_policy(env, info)

        # 2. 打印当前步的决策日志
        active_pids = [pid for pid, is_active in enumerate(info['active_agents']) if not is_active]
        if active_pids:
            print(f"[Step {step_count} | Env Time {env.total_time}s] 需要决策的飞机: {active_pids}")
            for pid in active_pids:
                j_code = env.job_code_list[actions[pid][0]]
                s_code = env.site_code_list[actions[pid][1]]
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