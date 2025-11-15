# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agent import Plane, Device
from env_utils import Job, Resource, Site
import json
import matplotlib.pyplot as plt
from gym.utils import seeding

class ScheduleEnv(gym.Env):
    environment_name = "Plane Schedule"
    """
    兼容 Gymnasium 的最小网格世界环境。
    注册名: 'GridWorld-v0'
    """

    def __init__(self, config, render_mode: str = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.steps = 0
        self.step_time = 0
        self.total_time = 0
        self.num_planes = 0
        self.num_agents = 0

        # ===== 可视化相关 =====
        self.fig = None
        self.ax = None
        
        self.initialize(config)
        self.job_list = list(self.jobs.keys())
        self.seed(config.get('seed', None))

        self.obs = {}
        self.done = False


    def initialize(self, config):
        with open(config['jobs_path'], 'r') as f:
                data = json.load(f)
        self.jobs = {item["作业编号"] : Job(code=item["作业编号"], 
                    time=item["作业时间"], 
                    group=item["分组"], 
                    resources=item["需要设备类型"] if isinstance(item["需要设备类型"], list) else [], 
                    predecessor=item["前置作业"] if isinstance(item["前置作业"], list) else [], 
                    exclusive=item["互斥作业"] if isinstance(item["互斥作业"], list) else [])
                for item in data}
        
        self.waiting_sites = {}
        for job_type, devices in self.jobs.items():
            self.waiting_sites[job_type] = []

        with open(config['fixed_res_path'], 'r') as f:
            data = json.load(f)
        self.fixed_resources = {item["设备编号"]: Resource(item["设备编号"], 
                                    item["类型"], 
                                    [str(idx) for idx in range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1])+1)],
                                    max_service=5) 
                                    for item in data}
        with open(config['mobile_res_path'], 'r') as f:
            data = json.load(f)
        self.mobile_resources = {item["设备编号"]: Resource(item["设备编号"], 
                                    item["类型"], 
                                    [item["初始停机位"]], 
                                    max_service=1) 
                                    for item in data} 
        
        with open(config['sites_path'], 'r') as f:
            data = json.load(f)
        self.sites = {code: Site(code, {'position': pos, 
                            'jobs': self.jobs, 
                            'fixed_resources': [res for res in self.fixed_resources.values() if code in res.sites], 
                            'mobile_resources': [res for res in self.mobile_resources.values() if code in res.sites]}) for code, pos in zip(data['sites_codes'], data['sites_positions'])}

        self.mobile_devices = {}
        for res in self.mobile_resources.values():
            device_cfg = {
                'resource': res,
                'velocity': 5 if res.code == 'R014' else 3,# 转运车速度和飞机速度一致
                'site': self.sites[res.sites[0]]
                }
            if res.type not in self.mobile_devices:
                self.mobile_devices[res.type] = [Device(res.code, device_cfg)]
            else:
                self.mobile_devices[res.type].append(Device(res.code, device_cfg))

        self.planes = {}
        self.num_agents = len(self.mobile_devices) + 1 # 移动设备集群数量 + 飞机集群
        self.force_transfer_planes = []
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def add_planes(self, new_planes_cfg):
        for plane_cfg in new_planes_cfg:
            plane_id = f'Plane_{plane_cfg["batch"]}_{plane_cfg["idx"]}'
            if plane_id in self.planes:
                # print(f"Warning: Plane {plane_id} already exists!")
                continue
            plane = Plane(plane_id, plane_cfg)
            self.planes[plane_id] = plane
            self.sites[plane.site.code].add_plane(plane)
        self.num_planes += len(new_planes_cfg)

    def remove_planes(self, plane_ids):
        for plane_id in plane_ids:
            if plane_id in self.planes:
                self.sites[self.planes[plane_id].site.code].remove_plane()
                del self.planes[plane_id]
                self.num_planes -= 1

    def add_interfere_sites(self, site_codes, rec_time):
        for site_code in site_codes:
            if site_code in self.sites:
                plane_id = self.sites[site_code].start_interfere(rec_time)
                if plane_id:
                    self.force_transfer_planes.append(plane_id)

    def get_avail_plane_actions(self, plane):
        return [1 if job in self.job_list else 0 for job in plane.get_avail_jobs()]
        
    def get_idle_devices(self, res_types):
        ret = []
        for res_type in res_types:
            ret += [device for device in self.mobile_devices[res_type] if device.is_idle()]
        return ret
    
    def get_idle_transporters(self):
        return [device for device in self.mobile_devices["R014"] if not device.is_busy and not device.is_transporting]
    
    def get_waiting_planes(self):
       return [plane for plane in self.planes.values() if plane.is_waiting]

    def get_avail_sites(self, plane=None):
        ret = [site.code for site in self.sites.values() if not site.is_occupied and not site.is_interfered and site.code not in ['Z', '29', '30', '31']]
        if plane is not None:
            ret.append(plane.site.code)  # 包括当前所在位置
        return ret
    
    def get_avail_takeoff_sites(self):
        return [site.code for site in self.sites.values() if not site.is_occupied and site.code in ['29', '30', '31']]
    
    def step(self, action, dt):
        self.steps += 1
        self.total_time += self.step_time
        step_time = np.inf
        waiting_planes = 0
        remove_planes = []
        
        for device_type, devices in self.mobile_devices.items():
            for idx, device in enumerate(devices):
                if not device.is_idle():
                    step_time = min(step_time, device.update(dt)) 
                elif action["devices"][device_type][idx] == device.site.code:
                    continue  # 目标位置与当前位置相同，无需移动
                else:
                    step_time = min(step_time, device.start_transport(self.sites[action["devices"][device_type][idx]]))
        
        for plane_id, plane in self.planes.items():
            if not plane.is_idle():
                step_time = min(step_time, plane.update(dt))
            else:
                target_site_code, target_job = action["planes"][int(plane_id.split('_')[1])][int(plane_id.split('_')[2])]
                # 先考虑转运，再考虑作业
                if target_site_code != plane.site.code:
                    # 无可用转运车的情况
                    if plane.site.get_avail_transporter() is None:
                        if plane.site.code == 'Z':  # 起始位置可以直接滑行
                            step_time = min(step_time, plane.start_transport(self.sites[target_site_code], None))
                            self.choosed_job = target_job
                        else:
                            plane.start_waiting()
                            self.waiting_sites['ZY-T'].append(plane.site.code)
                            plane.destination = self.sites[target_site_code]
                            plane.destination.add_plane(plane)
                            self.choosed_job = target_job
                            step_time = 0
                    else:
                        step_time = min(step_time, plane.start_transport(self.sites[target_site_code], plane.site.get_avail_transporter()))
                        self.choosed_job = target_job
                
                else:
                    if target_job not in plane.get_avail_jobs() and not plane.is_completed_all_jobs(): # 资源不足
                        plane.start_waiting()
                        self.waiting_sites[target_job.code].append(plane.site.code)
                        self.choosed_job = target_job
                    elif target_job is not None:
                        step_time = min(step_time, plane.choose_job(target_job))

            if plane.is_completed_all_jobs() and plane.site.code in ['29', '30', '31']:  # 完成所有任务并且抵达起飞跑道
                remove_planes.append(plane_id)
        
        for site_id, site in self.sites.items():
            if site.is_interfered:
                step_time = min(step_time, site.update(dt))

        self.remove_planes(remove_planes)
        for plane_id in remove_planes:
            print(f"Plane {plane_id} has taken off and removed from the airport.")
        if step_time == np.inf:
            print("No planes or devices in the environment!")
        self.step_time = step_time               
        self.obs = {}
        rewards = {}
        self.done = len(self.planes) == 0
        return self.obs, rewards, self.done
    
    def reset(self, seed=None, options=None):
        pass
        

    def render(self):
        # 如果没开可视化模式，可以直接返回
        if self.render_mode is None:
            return

        # 第一次调用时创建大一点的画布
        if self.fig is None or self.ax is None:
            plt.ion()
            # 图像大一些：10x8
            self.fig, self.ax = plt.subplots(figsize=(16, 14), dpi=120)
            self.ax.set_title("Airport Schedule Environment", fontsize=14)

        self.ax.clear()

        # ==== 计算可视区域（根据所有站位自动缩放） ====
        xs = [site.pos[0] for site in self.sites.values()]
        ys = [site.pos[1] for site in self.sites.values()]
        if xs and ys:
            margin = 10
            xmin, xmax = min(xs) - margin, max(xs) + margin
            ymin, ymax = min(ys) - margin, max(ys) + margin
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)

        # 1. 画所有机位（背景）
        for code, site in self.sites.items():
            x, y = site.pos
            # 空机位：空心方块
            self.ax.scatter(x, y, marker='s', s=120, edgecolors='gray', facecolors='none')
            self.ax.text(x, y - 2, code, ha='center', va='top', fontsize=7, color='gray')

        # 2. 画有飞机的机位（高亮）  
        for plane_id, plane in self.planes.items():
            x, y = plane.site.pos
            self.ax.scatter(x, y, marker='s', s=200)  # 实心方块
            self.ax.text(x, y + 2, plane_id, ha='center', va='bottom', fontsize=8)

        # 3. 画移动设备（小车等）
        for device_type, devices in self.mobile_devices.items():
            for d in devices:
                x, y = d.site.pos
                # 稍微上移一点，避免跟机位完全重合
                self.ax.scatter(x, y + 1.5, marker='o', s=40)
                self.ax.text(x, y + 3, d.code, ha='center', va='bottom', fontsize=6)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle='--', linewidth=0.5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

if __name__ == "__main__":
    import random
    from utils.arrangement import arrange_devices

    # ========= 固定随机种子 =========
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
        
    config = {
        'jobs_path': 'utils/config/jobs.json',
        'fixed_res_path': 'utils/config/fixed_resources.json',
        'mobile_res_path': 'utils/config/mobile_resources.json',
        'sites_path': 'utils/config/sites.json',
        'seed': SEED
    }
    env = ScheduleEnv(config)
    # env = ScheduleEnv(config, render_mode="human")

    # 第一阶段：着陆
    landing_list = []
    batch_num = 5
    plane_num_per_batch = 12
    for bidx in range(batch_num):
        landing_list += [item + bidx*3600 for item in list(range(0, 120*plane_num_per_batch, 120))]
    sampled_sites = [random.sample(env.get_avail_sites(), k=plane_num_per_batch)]

    total_time = 0
    step_time = 0
    current_batch = 0

    while True:
        if total_time in landing_list or step_time == 0:
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
            
            action = {}
            
            action["planes"] = [[[] for _ in range(plane_num_per_batch)] for _ in range(batch_num)]
            avail_sites = env.get_avail_sites()
                
            for plane_id, plane in env.planes.items():
                bidx_, pidx_ = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])
                plane_action = [None, None]
                if plane.is_idle():
                    if plane.site.code == 'Z' or plane.site == None or (random.uniform(0, 1) < 0.05 and not plane.is_completed_all_jobs()):
                        plane_action[0] = random.choice(avail_sites) # 随机选取
                        avail_sites.remove(plane_action[0])
                    else:
                        plane_action[0] = plane.site.code  # 保持在当前位置
                jobs = plane.get_avail_jobs(env.sites[plane_action[0]])
                if jobs and plane.is_idle():  # 非空
                    plane_action[1] = random.choice(jobs)
                action["planes"][bidx_][pidx_] = plane_action
                if plane_action[0] in avail_sites:
                    avail_sites.remove(plane_action[0])  # 从可用站点中移除已采样站点
            
            for plane_id, plane in env.planes.items():
                bidx_, pidx_ = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])
                
            
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

            obs, rewards, done = env.step(action, env.step_time - step_time)
            step_time = env.step_time

            # ===== 调用渲染 =====
            env.render()

            if done:
                print(f"All planes have taken off. Total time: {total_time}")
                break 
            elif step_time == 0:
                continue
        else:
            step_time -= 1
        # print(f"Step: {env.steps}, Step Time: {step_time}, Total Time: {total_time}")
        total_time += 1
        if total_time == landing_list[plane_num_per_batch*bidx-1]:
            print(f"All planes in batch {bidx} have landed. Total time: {total_time}")
        if total_time % 3600 == 0 and total_time != 0:
            sampled_sites.append(random.sample(env.get_avail_sites(), k=plane_num_per_batch))
        elif len(env.planes) >= 12 and all([plane.is_completed_all_jobs() for plane_id, plane in env.planes.items() if int(plane_id.split('_')[1]) == current_batch]):
            print(f"All planes in batch {current_batch} have completed their jobs.")
            current_batch += 1
            current_batch = min(current_batch, batch_num - 1)
                                                                                                                                                                                                                                                                                                                    