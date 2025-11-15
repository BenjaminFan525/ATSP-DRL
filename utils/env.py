# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agent import Plane, Device
from env_utils import Job, Resource, Site
import json
import matplotlib.pyplot as plt
from gym.utils import seeding
import math

class ScheduleEnv(gym.Env):
    environment_name = "Plane Schedule"
    
    def __init__(self, config, render_mode: str = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        
        # MARL核心属性
        self.n_agents = 0
        self.n_actions = 0
        self.obs_shape = 0
        self.state_shape = 0
        
        # 训练监控
        self.steps = 0
        self.step_time = 0
        self.total_time = 0
        self.episode_time_slice = []  # 每个step消耗时间列表
        self.current_finishing_planes = 0  # 已完成作业的飞机数
        
        # 可视化
        self.fig = None
        self.ax = None
        
        # 初始化环境数据
        self.initialize(config)
        self.seed(config.get('seed', None))
        
        # MARL状态缓存
        self.obs4marl = None
        self.state4marl = None
        self.job_record_for_gant = []  # 调度过程记录

    def initialize(self, config):
        """加载配置文件并初始化环境"""
        # 加载作业数据
        with open(config['jobs_path'], 'r') as f:
                data = json.load(f)
        self.jobs = {item["作业编号"] : Job(code=item["作业编号"], 
                    time=item["作业时间"], 
                    group=item["分组"], 
                    resources=item["需要设备类型"] if isinstance(item["需要设备类型"], list) else [], 
                    predecessor=item["前置作业"] if isinstance(item["前置作业"], list) else [], 
                    exclusive=item["互斥作业"] if isinstance(item["互斥作业"], list) else [])
                for item in data}
        
        # 等待队列
        self.waiting_sites = {job_type: [] for job_type in self.jobs.keys()}
        
        # 加载固定资源
        with open(config['fixed_res_path'], 'r') as f:
            data = json.load(f)
        self.fixed_resources = {
            item["设备编号"]: Resource(
                item["设备编号"],
                item["类型"],
                [str(idx) for idx in range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1]) + 1)],
                max_service=5
            ) for item in data
        }
        
        # 加载移动资源
        with open(config['mobile_res_path'], 'r') as f:
            data = json.load(f)
        self.mobile_resources = {
            item["设备编号"]: Resource(
                item["设备编号"],
                item["类型"],
                [item["初始停机位"]],
                max_service=1
            ) for item in data
        }
        
        # 加载站点
        with open(config['sites_path'], 'r') as f:
            data = json.load(f)
        self.sites = {
            code: Site(code, {
                'position': pos,
                'jobs': self.jobs,
                'fixed_resources': [res for res in self.fixed_resources.values() if code in res.sites],
                'mobile_resources': [res for res in self.mobile_resources.values() if code in res.sites]
            }) for code, pos in zip(data['sites_codes'], data['sites_positions'])
        }
        
        # 初始化移动设备
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
        
        # 飞机字典
        self.planes = {}
        self.num_planes = 0
        self.n_agents = len(self.mobile_devices) + 1  # 移动设备集群 + 飞机集群
        
        # 强制转运飞机列表
        self.force_transfer_planes = []
        
        # 构建MARL动作空间：选择机位 + 等待 + 正忙 + 完成
        self.n_actions = len(self.sites) + 3
        
        # 状态相关
        self.sites_state_global = [-1] * len(self.sites)  # 站位状态
        self.state_left_time = np.zeros(len(self.sites))  # 站位剩余时间
        
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

    def get_obs_agent(self, agent_id):
        """获取单个智能体的观测（飞机作为主要智能体）"""
        return self.obs4marl[agent_id]
    
    def get_obs(self):
        """获取所有智能体的观测"""
        return [self.get_obs_agent(i) for i in range(len(self.planes))]
    
    def get_state(self):
        """获取全局状态"""
        if self.state4marl is None:
            self._update_marl_state()
        return np.array(self.state4marl)
    
    def get_avail_agent_actions(self, plane_id):
        """获取智能体可用的动作掩码"""
        # 检查飞机是否处于正忙状态
        plane = self.planes[plane_id]
        if plane.is_waiting:
            return [0] * len(self.sites) + [1, 0, 0]
        elif plane.is_transporting or plane.is_busy:
            return [0] * len(self.sites) + [0, 1, 0]
        elif plane.is_completed_all_jobs():
            return [0] * len(self.sites) + [0, 0, 1]
        else:
            # 飞机空闲，计算可用机位
            avail_sites = self.get_avail_sites()
            avail_actions = [1 if site.code in avail_sites else 0 for site in self.sites.values()]
            # 返回 [可用机位..., 1, 0, 0] 表示可以"等待"
            return avail_actions + [1, 0, 0]
    
    def get_env_info(self):
        """返回环境基本信息供Runner使用"""
        return {
            "n_actions": self.n_actions,
            "n_agents": len(self.planes),  # 实际智能体数量是飞机数量
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs()[0]) if self.obs4marl else 0,
            "episode_limit": len(self.planes) * 10
        }
    
    def update_state(self):
        """更新MARL全局状态和观测"""
        # 构建全局状态
        state = []
        avail_sites = self.get_avail_sites()
        site_occupied, site_left_time, site_resource = [], [], []
        for site in self.sites.values():
            if site.code not in ['Z', '29', '30', '31']:
                # 1. 站位占用状态
                site_occupied.append(len(self.planes) + 1 if site.code in avail_sites else int(site.plane.code.split('_')[2]))
                # 2. 站位剩余工作时间
                if site.is_interfered:
                    site_left_time.append(site.left_rec_time)
                elif site.is_occupied:
                    site_left_time.append(site.left_job_time)
                # 3. 站位资源向量（one-hot）
                site_resource += [1 if site.is_avail_job(job) else 0 for job in self.jobs.values() if job.group == '保障' and job.code not in ['ZY01', 'ZY-L']]
        state += site_occupied + site_left_time + site_resource
        # 4. 飞机状态
        for plane in self.planes.values():
            if len(plane.left_jobs) > 0:
                next_job = sorted(plane.left_jobs, key=lambda x: plane.jobs[x].time, reverse=True)[0]
                state.extend([list(plane.jobs.keys()).index(next_job), plane.jobs[next_job].time, len(plane.left_jobs)])
            else:
                state.extend([len(plane.jobs), 0, 0])  # 已完成所有作业
            
        self.state4marl = np.array(state, dtype=np.float32)
        
        obs_dim = 3 + 2 * len(self.sites) - 4 + 1    # 3 个飞机自身信息 + 每站位 2 维 + 忙碌 flag
        zero_obs = [0.0] * obs_dim

        self.obs4marl = [[] for _ in range(len(self.planes))]
        for plane_id, plane in enumerate(self.planes.values()):
            # 飞机已完成所有作业 -> 全零观测
            if plane.is_completed_all_jobs():
                self.obs4marl[plane_id] = zero_obs
                continue

            next_job = sorted(plane.left_jobs, key=lambda x: plane.jobs[x].time, reverse=True)[0]
            self_obs = [float(list(plane.jobs.keys()).index(next_job)), 
                        float(plane.jobs[next_job].time), 
                        float(len(plane.left_jobs))]

            # 对每一个站位：运输时间 + 剩余工时
            per_site_obs = []
            for site in self.sites.values():
                if site.code not in ['Z', '29', '30', '31']:
                    # 运输时间（若无资源则为 0）
                    if site.is_avail_job(self.jobs[next_job]):
                        distance = abs(plane.site.pos[0]-site.pos[0]) + abs(plane.site.pos[1]-site.pos[1]) # 曼哈顿距离
                        travel_t = distance / plane.velocity
                    else:
                        travel_t = 0.0

                    # 站位剩余工时（空闲为 0）
                    left_t = site.left_job_time
                    per_site_obs.extend([travel_t, left_t])

            # 忙碌标记
            if not plane.is_idle():
                # 正忙：除自身前 3 维外全部置 0
                per_site_obs = [0.0] * len(per_site_obs)

            self.obs4marl[plane_id] = self_obs + per_site_obs + [float(not plane.is_idle())]
    
    def reset(self, seed=None, options=None):
        """重置环境，返回初始全局状态"""
        self.steps = 0
        self.total_time = 0
        self.episode_time_slice = []
        self.current_finishing_planes = 0
        self.job_record_for_gant = []
        self.sites_state_global = [-1] * len(self.sites)
        self.state_left_time = np.zeros(len(self.sites))
        self.done = False
        
        # 清空飞机和设备状态
        self.planes.clear()
        for devices in self.mobile_devices.values():
            for device in devices:
                device.reset()
        
        # 重新初始化（保持配置不变）
        self.initialize(self.config)
        
        # 返回初始状态
        return self.get_state()
    
    def step(self, action, dt):
        """
        执行一步动作
        action: dict 包含飞机和设备的动作
        dt: int 时间差
        """
        self.steps += 1
        self.total_time += dt 
        
        # 动作处理逻辑（参考您的原始实现）
        step_time, real_did = self.process_action(action, dt)        
        
        self.step_time = step_time              
        self.obs = {}
        self.done = len(self.planes) == 0

        # 计算全局奖励
        reward = self.calculate_global_reward(real_did)

        # 更新状态
        self.update_state()
        
        return reward, self.done
    
    def process_action(self, action, dt):
        """处理动作并计算即时奖励"""
        step_time = np.inf
        real_did = 0
        remove_planes = []
        
        # 处理设备动作
        for device_type, devices in self.mobile_devices.items():
            for idx, device in enumerate(devices):
                if not device.is_idle():
                    step_time = min(step_time, device.update(dt)) 
                elif action["devices"][device_type][idx] == device.site.code:
                    continue  # 目标位置与当前位置相同，无需移动
                else:
                    step_time = min(step_time, device.start_transport(self.sites[action["devices"][device_type][idx]]))
        
        # 处理飞机动作
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
                            real_did += 1
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
                        real_did += 1
                
                else:
                    if target_job not in plane.get_avail_jobs() and not plane.is_completed_all_jobs(): # 资源不足
                        plane.start_waiting()
                        self.waiting_sites[target_job.code].append(plane.site.code)
                        self.choosed_job = target_job
                    elif target_job is not None:
                        step_time = min(step_time, plane.choose_job(target_job))
                        real_did += 1

            if plane.is_completed_all_jobs() and plane.site.code in ['29', '30', '31'] and plane.is_idle():  # 完成所有任务并且抵达起飞跑道
                remove_planes.append(plane_id)
        
        for site in self.sites.values():
            if site.is_interfered:
                step_time = min(step_time, site.update(dt))

        self.remove_planes(remove_planes)
        for plane_id in remove_planes:
            print(f"[Time: {self.total_time}] Plane {plane_id} has taken off and removed from the airport.")
        if step_time == np.inf:
            print("No planes or devices in the environment!")
        return step_time, real_did
    
    def calculate_global_reward(self, real_did):
        """计算全局奖励"""
        # 等待惩罚
        waiting_penalty = sum(1 for p in self.planes.values() if p.is_waiting) * 2
        # makespan相关奖励
        if self.done:
            makespan_reward = 6000 / self.total_time
        else:
            makespan_reward = real_did - (self.steps) / 60
        # 综合奖励
        global_reward = makespan_reward  - waiting_penalty
        
        return global_reward

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