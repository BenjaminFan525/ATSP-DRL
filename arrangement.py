import random
import numpy as np
import pulp
from scipy.spatial.distance import cdist
from environment import ScheduleEnv
from datetime import datetime, timedelta

def seconds_to_datetime(start: str, delta_seconds: int) -> str:
    '''start: "HH:MM:SS"
    return: "HH:MM:SS"  24h制，不跨天
    '''
    t = datetime.strptime(start, "%H:%M:%S") + timedelta(seconds=delta_seconds)
    # 折回同一天
    t = t.replace(day=1)          # 把日期固定到同一天
    return t.strftime("%H:%M:%S")

def datetime_to_seconds(now: str, start: str) -> int:
    '''计算时间差（秒数）'''
    def to_sec(t: str) -> int:
        h, m, s = map(int, t.split(":"))
        return h * 3600 + m * 60 + s
    return (to_sec(now) - to_sec(start)) % (24 * 3600)

def arrange_devices(devices, planes):
    '''使用线性规划安排转运车
    
    通过最小化总运输距离，将空闲转运车分配给等待的飞机
    支持车多任务少的情况，确保每个任务点至少被服务一次
    '''
    # 按等待时间排序，筛选前n个等待时间最长的飞机
    if len(planes) >= len(devices):
        # planes = sorted(planes, key=lambda x: x.waiting_time, reverse=True)[:len(devices)-1]
        planes = sorted(planes, key=lambda x: x.waiting_time, reverse=True)[:len(devices)]

    # 提取位置信息
    device_positions = np.array([transporter.site.pos for transporter in devices])
    plane_positions = np.array([plane.site.pos for plane in planes])
    # 计算代价矩阵
    cost_matrix = cdist(device_positions, plane_positions, metric='cityblock')
    # 定义线性规划问题
    prob = pulp.LpProblem("Transporter_Assignment", pulp.LpMinimize)
    # 定义决策变量: x[i][j] = 1 表示车辆i分配给任务点j
    x = pulp.LpVariable.dicts('分配', (range(len(devices)), range(len(planes))), cat='Binary')
    # 目标函数: 最小化总运输距离
    prob += pulp.lpSum([
        cost_matrix[i][j] * x[i][j] 
        for i in range(len(devices)) for j in range(len(planes))
    ])

    # 约束1: 每个任务点至少被一辆车服务 (车多情况)
    for j in range(len(planes)):
        prob += pulp.lpSum([x[i][j] for i in range(len(devices))]) >= 1

    # 约束2: 每辆车最多服务一个任务点
    for i in range(len(devices)):
        prob += pulp.lpSum([x[i][j] for j in range(len(planes))]) <= 1

    # 求解线性规划问题
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    ret = []
    for i in range(len(devices)):
        for j in range(len(planes)):
            if pulp.value(x[i][j]) == 1:
                device, plane = devices[i], planes[j]
                ret.append((device.resource.type, device.code, plane.site.code))

    return ret

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
            if force_chosen and plane_id == f"Plane_{force_chosen[0]}_{force_chosen[1]}":
                plane_action[0] = force_chosen[2]  # 强制选择站点
            elif plane.site.code == 'Z' or plane_id in env.force_transfer_planes or (random.uniform(0, 1) < 0.05 and not plane.is_completed_all_jobs()):
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
                # env.sites[force_site].is_occupied = False
                force_chosen = [bidx, pidx, force_site]
                force_chosen_plane = f"Plane_{bidx}_{pidx}"
                config['force_chosen'][0] = -1
                print(f"Forced site chosen: {force_site} at time {total_time} for Plane_{bidx}_{pidx}")
            else:
                force_chosen = None
            
            action = random_arrangement(env, plane_num_per_batch, batch_num, min(current_batch, batch_num - 1), force_chosen)

            for bid in range(batch_num):
                for pid in range(plane_num_per_batch):
                    plane_id = f"Plane_{bid}_{pid}"
                    plane_action = action["planes"][bid][pid]
                    if plane_id in env.planes and (plane_action[0] or plane_action[1]):
                        plane = env.planes[plane_id]
                        transporter = plane.site.get_avail_transporter()
                        transporter = transporter.code if transporter else None
                        action_history.append({'Time:': seconds_to_datetime("00:08:00", total_time), 'Plane': plane_id, 'Start_Site': plane.site.code, 'End_Site': plane_action[0], 'Job': plane_action[1],'Transporter': transporter})

            reward, done = env.step(action, env.step_time - step_time)
            step_time = env.step_time

            # 添加修理时间
            if force_chosen_plane:
                plane = env.planes[force_chosen_plane]
                if plane.is_transporting:
                    plane.left_trans_time += config['force_chosen'][2]
                print(f"Plane {force_chosen_plane} will be repaired for {config['force_chosen'][2]} seconds.")
                force_chosen_plane = None

            # ===== 调用渲染 =====
            env.render()
            
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
        
    return action_history


if __name__ == "__main__":
    import json
    config = {
        'batch_num': 5,
        'plane_num_per_batch': 12,
        'jobs_path': 'utils/config/jobs.json',
        'fixed_res_path': 'utils/config/fixed_resources.json',
        'mobile_res_path': 'utils/config/mobile_resources.json',
        'sites_path': 'utils/config/sites.json',
        'seed': 42
    }
    config['interfere'] = [2100, ['10', '11', '12', '13', '14', '15'], 1800]  # [start_time, [site_code1, ...site_code2, ...], time_span]
    # config['interfere'] = [-1, ['10', '11', '12', '13', '14', '15'], 1800] 

    # config['force_chosen'] = [8100, '11', 1800]  # [start_time, site_code, time_span]
    # config['force_chosen'] = [-1, '4', 1800]
    action_history = run_arrangement(config)

    # 写入文件
    save_dir = "utils"
    file_name = "output.txt"
    with open(f"{save_dir}/{file_name}", "w", encoding="utf-8") as f:
        json.dump({'Data': action_history}, f, ensure_ascii=False, indent=4)
    print(f"Action history saved to {save_dir}/{file_name}")
    print("Arrangement completed.")