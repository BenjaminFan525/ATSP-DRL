import random
import numpy as np
import pulp
from scipy.spatial.distance import cdist
import sys
import os
# 定位到 MAIA 根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../')) # 指向 MAIA
sys.path.insert(0, root_dir)
env_dir = os.path.abspath(os.path.join(current_dir, '../')) # 指向 IA
sys.path.insert(0, env_dir)

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv as ScheduleEnv
from onpolicy.envs.HKBZ.utils import seconds_to_datetime, datetime_to_seconds, site_disable, device_disable, arrange_devices
from datetime import datetime, timedelta
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def random_arrangement(env, plane_num_per_batch, batch_num, current_batch, force_chosen=None):
    '''随机安排环境和设备动作
    
    为飞机和设备生成随机动作，处理飞机转运、作业选择逻辑
    支持强制选择特定站点和转运车调度优化
    '''
    # ========= 固定随机种子 =========
    # random.seed(seed)
    # np.random.seed(seed)
    
    action = {}
    
    action["planes"] = [[[] for _ in range(plane_num_per_batch)] for _ in range(batch_num)]
    avail_sites = env.get_avail_sites()
        
    for plane_id, plane in env.planes.items():
        bidx_, pidx_ = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])
        plane_action = [None, None]
        if plane.is_idle():
            if plane.site.code == 'Z' or plane_id in env.force_transfer_planes or (random.uniform(0, 1) < 0.05 and not plane.is_completed_all_jobs()):
                plane_action[0] = random.choice(avail_sites) # 随机选取
                avail_sites.remove(plane_action[0])
                if plane_id in env.force_transfer_planes:
                    env.force_transfer_planes.remove(plane_id)  # 移除强制转运飞机
            else:
                plane_action[0] = plane.site.code  # 保持在当前位置
        jobs = plane.get_avail_jobs(env.sites[plane_action[0]]) if plane_action[0] else []
        if jobs and plane.is_idle():  # 非空
            # plane_action[1] = random.choice(jobs)
            plane_action[1] = sorted(jobs, key=lambda x: plane.jobs[x].time, reverse=True)[0]
        action["planes"][bidx_][pidx_] = plane_action
        if plane_action[0] in avail_sites:
            avail_sites.remove(plane_action[0])  # 从可用站点中移除已采样站点
    
    for batch_idx in range(current_batch+1):
        if len(env.planes) and all([plane.is_completed_all_jobs() for plane_id, plane in env.planes.items() if int(plane_id.split('_')[1]) == batch_idx]):
            takeoff_planes = []
            for site in env.get_avail_takeoff_sites():
                for plane_id, plane in env.planes.items():
                    if int(plane_id.split('_')[1]) == batch_idx:
                        if plane.is_idle() and plane_id not in takeoff_planes:
                            bidx_, pidx_ = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])
                            action["planes"][bidx_][pidx_] = [site, None]
                            takeoff_planes.append(plane_id)
                            break
    
    action["devices"] = {}
    for device_type, devices in env.mobile_devices.items():
        action["devices"][device_type] = [device.site.code for device in devices]
    
    for job_code, waiting_sites in env.waiting_sites.items():
        if len(waiting_sites) > 0:
            devices = env.get_idle_devices(env.jobs[job_code].resources)
            if len(devices) > 0:
                planes = [plane for plane in env.planes.values() if plane.site.code in waiting_sites]
                assignments = arrange_devices(devices, planes)
                for device_type, device_code, target_site in assignments:
                    for idx, device in enumerate(env.mobile_devices[device_type]):
                        if device.resource.code == device_code:
                            action["devices"][device_type][idx] = target_site
                            waiting_sites.remove(target_site)
                            break

    return action

def marl_arrangement(env, agents, last_action, epsilon, plane_num_per_batch, batch_num, current_batch, force_chosen=None):
    '''使用深度强化学习策略规划停机位和设备动作
    
    每一个batch使用单独的一个agent来决策,从而避免智能体数量大幅变化
    
    为飞机和设备生成随机动作，处理飞机转运、作业选择逻辑
    支持强制选择特定站点和转运车调度优化
    '''
    obs = env.get_obs(plane_num_per_batch*batch_num)  # [[],[],...]
    state = env.get_state(plane_num_per_batch*batch_num) # []
    avail_actions = [[env.get_avail_agent_actions() for _ in range(plane_num_per_batch)] for _ in range(batch_num)]
    
    action = {}
    action["planes"] = [[[] for _ in range(plane_num_per_batch)] for _ in range(plane_num_per_batch)]
    chosen_actions = []
    # 选择动作
    for plane_id, plane in env.planes.items():
        bidx_, pidx_ = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])
        plane_action = [None, None]

        avail_action = env.get_avail_agent_actions(plane_id)  # 获取该agent可用动作列表

        if len(env.planes) and all([plane.is_completed_all_jobs() for plane_id, plane in env.planes.items() if int(plane_id.split('_')[1]) == 0]):
            avail_action[-4:-1] = [1 if not site.is_occupied else 0 for site in env.sites.values() if site.code in ['29', '30', '31']]
        for a in chosen_actions:
            avail_action[a] = 0
        # 全0时保证等待
        if all(x == 0 for x in avail_action):    
            avail_action[-1] = 1
        agent_action = agents[batch_num].choose_action(obs[pidx_], last_action[pidx_],pidx_,                                                    avail_action, epsilon, True)
        # 除去等待的情况
        if agent_action != len(avail_action) -1:
            chosen_actions.append(agent_action)
            plane_action[0] = list(env.sites.keys())[agent_action]
        jobs = plane.get_avail_jobs(env.sites[plane_action[0]]) if plane_action[0] else []
        if jobs and plane.is_idle(): 
            plane_action[1] = sorted(jobs, key=lambda x: plane.jobs[x].time, reverse=True)[0]
        if len(env.planes) == 1 and plane.is_completed_all_jobs():
            if plane_action[0] == None:
                if plane.is_idle():
                    print(6666)
        action["planes"][bidx_][pidx_] = plane_action             
    
    action["devices"] = {}
    for device_type, devices in env.mobile_devices.items():
        action["devices"][device_type] = [device.site.code for device in devices]
    
    for job_code, waiting_sites in env.waiting_sites.items():
        if len(waiting_sites) > 0:
            devices = env.get_idle_devices(env.jobs[job_code].resources)
            if len(devices) > 0:
                planes = [plane for plane in env.planes.values() if plane.site.code in waiting_sites]
                assignments = arrange_devices(devices, planes)
                for device_type, device_code, target_site in assignments:
                    for idx, device in enumerate(env.mobile_devices[device_type]):
                        if device.resource.code == device_code:
                            action["devices"][device_type][idx] = target_site
                            waiting_sites.remove(target_site)
                            break

    return action



def run_arrangement(config, render_mode=None):
    '''运行完整的调度安排流程
    
    初始化环境，按批次添加飞机，处理干涉和强制选择事件
    生成动作历史并保存到文件
    '''
    # ========= 固定随机种子 =========
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    if 'interfere' not in config:
        config['interfere'] = [-1, [], 0]
    if 'force_chosen' not in config:
        config['force_chosen'] = [-1, '', 0]

    # 初始化环境
    env = ScheduleEnv(config, render_mode=render_mode)

    landing_list = []
    batch_num = config['batch_num']
    plane_num_per_batch = config['plane_num_per_batch']
    for bidx in range(batch_num):
        landing_list += [item + bidx*3600 for item in list(range(0, 120*plane_num_per_batch, 120))]
    config['force_chosen'][0] = min([num for num in landing_list if num > config['force_chosen'][0]]) if config['force_chosen'][0] > 0 else config['force_chosen'][0]

    total_time = 0
    step_time = 0
    current_batch = 0
    force_chosen_plane = None
    action_history = []

    figures = []
    while True:
        if total_time in landing_list or step_time == 0 or total_time == config['interfere'][0] or total_time == config['force_chosen'][0]:
            bidx = total_time // 3600
            pidx = (total_time % 3600) // 120
            if total_time in landing_list:
                plane_cfg = {
                    'velocity': 5,
                    'site': env.sites['Z'],
                    'fuel': np.random.randint(0, 30),
                    'jobs': env.jobs.values()
                }
                env.add_planes([{'batch': bidx, 'idx': pidx, **plane_cfg}])  
            if total_time == config['interfere'][0]:
                interfere_sites = config['interfere'][1]
                interfere_time = config['interfere'][2]
                env.add_interfere_sites(interfere_sites, interfere_time)
                print(f"Interference set for site {interfere_sites} from {total_time} to {total_time + interfere_time}")
                config['interfere'][0] = 0  # 只触发一次
            if total_time == config['force_chosen'][0]:
                force_site = config['force_chosen'][1]
                env.add_interfere_sites([force_site], 0)
                env.sites[force_site].is_interfered = False
                force_chosen = [bidx, pidx, force_site]
                force_chosen_plane = f"Plane_{bidx}_{pidx}"
                config['force_chosen'][0] = -1
                print(f"Forced site chosen: {force_site} at time {total_time} for Plane_{bidx}_{pidx}")
            else:
                force_chosen = None
            
            action = random_arrangement(env, plane_num_per_batch, batch_num, min(current_batch, batch_num - 1), force_chosen)
            # action = marl_arrangement(env, plane_num_per_batch, batch_num, min(current_batch, batch_num - 1), force_chosen)

            reward, done = env.step(action, env.step_time - step_time)
            step_time = env.step_time

            # 添加修理时间
            if force_chosen_plane:
                plane = env.planes[force_chosen_plane]
                if plane.is_transporting:
                    plane.left_trans_time += config['force_chosen'][2]
                print(f"Plane {force_chosen_plane} will be repaired for {config['force_chosen'][2]} seconds.")
                force_chosen_plane = None
            
            if done:
                print(f"All planes have taken off. Total time: {total_time}, Reward: {reward}")
                break 
            elif step_time == 0:
                continue
        step_time -= 1
        total_time += 1
        # print(f"Step: {env.steps}, Step Time: {step_time}, Total Time: {total_time}, Reward: {reward}")
        if total_time == landing_list[plane_num_per_batch*bidx-1]:
            print(f"All planes in batch {bidx} have landed. Total time: {total_time}")
        if len(env.planes) >= 12 and all([plane.is_completed_all_jobs() for plane_id, plane in env.planes.items() if int(plane_id.split('_')[1]) == current_batch]) and current_batch <= batch_num - 1:
            print(f"All planes in batch {current_batch} have completed their jobs.")
            current_batch += 1
            # current_batch = min(current_batch, batch_num - 1)
        
    return action_history, figures


if __name__ == "__main__":
    import json
    import os
    import cv2
    from tqdm import tqdm
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    
    config = {
        'batch_num': 1,
        'plane_num_per_batch': 12,
        'jobs_path': 'utils/config/jobs.json',
        'fixed_res_path': 'utils/config/fixed_resources.json',
        'mobile_res_path': 'utils/config/mobile_resources.json',
        'sites_path': 'utils/config/sites.json',
        'seed': 42
    }
    # config['interfere'] = [2100, ['10', '11', '12', '13', '14', '15'], 1800]  # [start_time, [site_code1, ...site_code2, ...], time_span]
    # config['interfere'] = [-1, ['10', '11', '12', '13', '14', '15'], 1800] 

    # config['force_chosen'] = [8100, '11', 1800]  # [start_time, site_code, time_span]
    # config['force_chosen'] = [-1, '4', 1800]
    action_history, figures = run_arrangement(config, render_mode="human")

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # videoWriter = cv2.VideoWriter(os.path.join('/home/fanyx/HKBZ-environment', "render_origin.mp4"), fourcc, 12, (figures[0].shape[1], figures[0].shape[0]), True)
    # # map(videoWriter.write, figs)
    # for fig in figures:
    #     videoWriter.write(fig)
    # videoWriter.release() 

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30) # 打印耗时前20的函数

    # 写入文件
    save_dir = "utils"
    file_name = "output.json"
    with open(f"{save_dir}/{file_name}", "w", encoding="utf-8") as f:
        json.dump({'Data': action_history}, f, ensure_ascii=False, indent=4)
    print(f"Action history saved to {save_dir}/{file_name}")
    print("Arrangement completed.")