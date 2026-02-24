import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
import os
import time
import sys
import types

# ================= 1. 路径与模块修复 (保持不变以防止报错) =================
# 定位到 MAIA 根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) # 指向 MAIA
sys.path.insert(0, root_dir)
env_dir = os.path.abspath(os.path.join(current_dir, '../')) # 指向 IA
sys.path.insert(0, env_dir)

from onpolicy.envs.IA.environment import FarmScheduleEnv 

def test_with_plan():
    # profiler = cProfile.Profile()
    # profiler.enable()
    # ================= 配置参数 =================
    DATA_DIR = "/home/fanyx/MAIA/onpolicy/envs/IA/test/18_52_12" 
    
    if not os.path.exists(DATA_DIR):
        print(f"错误: 找不到数据目录 '{DATA_DIR}'。")
        return

    DATA_LIST = [DATA_DIR] 
    MAX_AGENT_NUM = 6   
    MAX_NODE_NUM = 200  

    # ================= 0. 定义你的规划方案 (List of Lists) =================
    full_plan = [
        [[3, 1], [2, 0], [4, 1], [1, 0], [5, 1], [0, 0], [6, 1], [7, 0]], 
        [[8, 1], [10, 0], [11, 1], [9, 0], [23, 1]], 
        [[24, 1], [22, 0], [25, 1], [21, 0], [26, 1], [27, 0], [28, 1], [29, 0]], 
        [[30, 1], [31, 0], [32, 1], [33, 0], [34, 1], [35, 0], [36, 1], [37, 0], [38, 1], [39, 0], [40, 1], [41, 0], [42, 1], [43, 0], [12, 1], [13, 0], [14, 1], [15, 0], [16, 1], [17, 0], [18, 1], [19, 0], [20, 0]],
        [],
        []]
    # full_plan = [
    #     [[6, 1], [25, 0], [9, 0], [8, 0], [35, 0], [38, 0], [7, 0], [37, 0], [36, 0], [28, 0], [43, 0], [3, 0], [11, 0]], 
    #     [[29, 1], [32, 0], [31, 0]], 
    #     [[19, 1], [0, 0], [13, 0], [39, 0], [22, 0], [27, 0], [40, 0], [33, 0], [21, 0], [14, 0]], 
    #     [[1, 1], [34, 0], [4, 0], [24, 0], [30, 0], [23, 0], [16, 0], [2, 0], [12, 0], [41, 0], [18, 0], [20, 0], [17, 0], [15, 0], [5, 0], [42, 0], [26, 0], [10, 0]], 
    #     [], 
    #     []]
    
    # 确保 full_plan 的长度匹配 MAX_AGENT_NUM，不足补空list
    while len(full_plan) < MAX_AGENT_NUM:
        full_plan.append([])

    # 初始化指针：记录每个农机当前执行到 full_plan 中的哪一步
    task_cursors = [0] * MAX_AGENT_NUM

    # ================= 初始化环境 =================
    print("正在初始化环境...")
    try:
        env = FarmScheduleEnv(DATA_LIST, MAX_AGENT_NUM, MAX_NODE_NUM)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return

    obs, done, info  = env.reset()

    # ================= 绘图设置 =================
    # fig, ax = plt.subplots(figsize=(12, 10))
    # plt.ion() 
    # ax.set_title("Farm Schedule Simulation (Planned Execution)")

    # ================= 仿真循环 =================
    print("开始执行规划方案...")
    step_count = 0
    all_done = False

    rewards_all = []
    active_all = []
    done_all = []

    while not all_done:
        # 1. 获取环境信息 (查看谁空闲)
        info = env._get_info()
        active_agents = info['active_agents'] # List[bool]: True表示空闲/需要新任务，False表示正在忙
        
        # 2. 根据规划构建动作
        actions = []
        for agent_idx in range(MAX_AGENT_NUM):
            is_active = not active_agents[agent_idx]
            
            # 如果智能体空闲(active) 且 它的任务列表里还有剩余任务
            if is_active and task_cursors[agent_idx] < len(full_plan[agent_idx]):
                # 取出当前指针指向的任务
                line, ent = full_plan[agent_idx][task_cursors[agent_idx]]
                actions.append([line, ent])
                
                # 指针后移，准备下一次取下一个任务
                task_cursors[agent_idx] += 1
                
            else:
                # 如果智能体正忙，或者任务列表已空
                # 发送 0 或 -1 (取决于环境对空动作的定义，通常发0或者重复上一个动作均可，只要不报错)
                actions.append([]) 

        # 3. 环境步进
        obs, rewards, dones, info = env.step(actions, remain_depot=False)
        rewards_all.append(rewards)
        active_all.append(1-info['active_agents'])
        done_all.append(dones)
        
        # 4. 增量渲染
        # env.render(ax=ax)
        
        # 5. 检查结束条件
        # 条件 A: 所有智能体在环境中都 Done 了
        env_all_done = all([d for d in dones])
        
        # 条件 B: 我们的规划列表也都发完了
        all_plans_dispatched = True
        for i in range(MAX_AGENT_NUM):
            if task_cursors[i] < len(full_plan[i]):
                all_plans_dispatched = False
                break
        
        # 只有当 任务全发完了 且 农机全跑完了 才算结束
        if env_all_done and all_plans_dispatched:
            all_done = True
        
        step_count += 1
        
        # 打印进度 (可选)
        # if step_count % 10 == 0:
        #     print(f"Step: {step_count} | Active: {active_agents}")

        # 防止死循环保险
        if step_count > 1000:
            print("达到最大步数，强制停止。")
            break

    rewards_all = np.stack(rewards_all)
    active_all = np.stack(active_all)
    done_all = np.stack(done_all)

    t = max([v.next_free_time for v in env.simulator.vehicles]) if env.simulator.vehicles else 0
    c = sum([v.total_cost for v in env.simulator.vehicles])
    s = sum([v.total_dist for v in env.simulator.vehicles])

    print(f"Total time: {t}, Total distance: {s}, Total cost: {c}")
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(20) # 打印耗时前20的函数
    # print("仿真及渲染结束！")
    # plt.ioff()
    # plt.show()

if __name__ == "__main__":
    test_with_plan()