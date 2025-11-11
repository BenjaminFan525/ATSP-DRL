# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agent import Plane, Device
from env_utils import Job, Resource, Site
import json

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
        
        self.initialize(config)
        self.job_list = list(self.jobs.keys())

        self.obs = {}
        self.action = {}
        # self.action["devices"] = {k: [self.sites[0] for _ in v] for k, v in self.mobile_devices.items()}
        # self.action["planes"] = [[self.sites[0], self.jobs[0]] for _ in range(self.num_planes)]
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
        # for idx in range(self.num_planes):
        #     plane_cfg = {
        #         'velocity': 5,
        #         'site': self.sites['Z'],
        #         'fuel': np.random.randint(0, 30),
        #         'jobs': self.jobs.values()
        #     }
        #     self.planes[f'Plane_{idx}'] = Plane(f'Plane_{idx}', plane_cfg)

        self.num_agents = len(self.mobile_devices) + 1 # 移动设备集群数量 + 飞机集群
    
    def add_planes(self, new_planes_cfg):
        for plane_cfg in new_planes_cfg:
            plane_id = f'Plane_{plane_cfg["batch"]}_{plane_cfg["idx"]}'
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

    def get_avail_plane_actions(self, plane):
        return [1 if job in self.job_list else 0 for job in plane.get_avail_jobs()]

    def get_avail_transporter(self, plane):
        current_site = plane.site
        current_site.update_resources()
        if "R014" in current_site.res_avail:
            res = current_site.resources[current_site.res_avail["R014"][-1]]
            for device in self.mobile_devices["R014"]:
                if device.resource.code == res.code and not device.is_busy and not device.is_transporting:
                    transporter = device
            return transporter
        else:
            return None
        
    def get_idle_transporters(self):
        return [device for device in self.mobile_devices["R014"] if not device.is_busy and not device.is_transporting]
    
    def get_waiting_planes(self):
       return [plane for plane in self.planes.values() if plane.is_waiting]


    def get_avail_sites(self, plane=None):
        ret = [site.code for site in self.sites.values() if not site.is_occupied and site.code not in ['Z', '29', '30', '31']]
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
                if device.is_busy or device.is_transporting:
                    step_time = min(step_time, device.update(dt)) 
                elif action["devices"][device_type][idx] == device.site.code:
                    continue  # 目标位置与当前位置相同，无需移动
                else:
                    device.start_transport(self.sites[action["devices"][device_type][idx]])
        
        for plane_id, plane in self.planes.items():
            if plane.is_busy or plane.is_transporting:
                step_time = min(step_time, plane.update(dt))
            else:
                target_site_code, target_job = action["planes"][int(plane_id.split('_')[1])][int(plane_id.split('_')[2])]
                if target_site_code != plane.site.code:
                    if self.get_avail_transporter(plane) is None:
                        waiting_planes += 1
                        if plane.site.code == 'Z':  # 起始位置可以直接滑行
                            # if plane.is_waiting:
                            #     plane.finish_waiting()
                            step_time = min(step_time, plane.start_transport(self.sites[target_site_code], None))
                        else:
                            plane.start_waiting()
                            continue  # 无可用转运车，等待
                    else:
                        step_time = min(step_time, plane.start_transport(self.sites[target_site_code], self.get_avail_transporter(plane)))
                else:
                    if target_job is None:
                        if len(plane.get_avail_jobs()) == 0 and not plane.is_completed_all_jobs():
                            waiting_planes += 1
                            plane.start_waiting()
                            continue  # 无任务，等待
                    else:
                        if plane.is_waiting:
                            plane.finish_waiting()  
                        step_time = min(step_time, plane.choose_job(target_job))

            if plane.is_completed_all_jobs() and plane.site.code in ['29', '30', '31']:  # 完成所有任务并且抵达起飞跑道
                remove_planes.append(plane_id)
        
        self.sites_avail = self.get_avail_sites()
        self.remove_planes(remove_planes)
        self.step_time = step_time               
        self.obs = {}
        rewards = {}
        self.done = len(self.planes) == 0
        return self.obs, rewards, self.done
    
    def reset(self, seed=None, options=None):
        pass
        

    def render(self):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    import random
    import pulp
    from scipy.spatial.distance import cdist

    def arrange_transporters(transporters, planes):
        '''使用线性规划安排转运车'''
        # 按等待时间排序，筛选前n个等待时间最长的飞机
        if len(planes) >= len(transporters):
            planes = sorted(planes, key=lambda x: x.waiting_time, reverse=True)[:len(transporters)-1]

        # 提取位置信息
        transporter_positions = np.array([transporter.site.pos for transporter in transporters])
        plane_positions = np.array([plane.site.pos for plane in planes])
        # 计算代价矩阵
        cost_matrix = cdist(transporter_positions, plane_positions, metric='cityblock')
        # 定义线性规划问题
        prob = pulp.LpProblem("Transporter_Assignment", pulp.LpMinimize)
        # 定义决策变量: x[i][j] = 1 表示车辆i分配给任务点j
        x = pulp.LpVariable.dicts('分配', (range(len(transporters)), range(len(planes))), cat='Binary')
        # 目标函数: 最小化总运输距离
        prob += pulp.lpSum([
            cost_matrix[i][j] * x[i][j] 
            for i in range(len(transporters)) for j in range(len(planes))
        ])

        # 约束1: 每个任务点至少被一辆车服务 (车多情况)
        for j in range(len(planes)):
            prob += pulp.lpSum([x[i][j] for i in range(len(transporters))]) >= 1

        # 约束2: 每辆车最多服务一个任务点
        for i in range(len(transporters)):
            prob += pulp.lpSum([x[i][j] for j in range(len(planes))]) <= 1

        # 求解线性规划问题
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        ret = []
        for i in range(len(transporters)):
            for j in range(len(planes)):
                if pulp.value(x[i][j]) == 1:
                    transporter, plane = transporters[i], planes[j]
                    ret.append((transporter.resource.type, transporter.code, plane.site.code))
        for plane in planes:
            plane.finish_waiting()
        return ret
        
    config = {
        'jobs_path': 'utils/config/jobs.json',
        'fixed_res_path': 'utils/config/fixed_resources.json',
        'mobile_res_path': 'utils/config/mobile_resources.json',
        'sites_path': 'utils/config/sites.json'
    }
    env = ScheduleEnv(config)

    # # 第一阶段：着陆
    # landing_per_batch = list(range(0, 1440, 120))  # 着陆时间点
    # landing_list = []
    # for bidx in range(5):
    #     landing_list += [item + bidx*3600 for item in list(range(0, 1440, 120))]
    plane_num = 6
    landing_list = list(range(0, 120*plane_num, 120))
    sampled_sites = random.sample(env.get_avail_sites(), k=plane_num)

    total_time = 0
    step_time = 0
    bidx, pidx = 0, 0

    while True:
        if total_time in landing_list or step_time == 0:
            
            if total_time in landing_list:
                pidx = landing_list.index(total_time)
                plane_cfg = {
                    'velocity': 5,
                    'site': env.sites['Z'],
                    'fuel': np.random.randint(0, 30),
                    'jobs': env.jobs.values()
                }
                env.add_planes([{'batch': 0, 'idx': pidx, **plane_cfg}])  
            
            action = {}
            action["devices"] = {}
            for device_type, devices in env.mobile_devices.items():
                action["devices"][device_type] = [device.site.code for device in devices]
            if len(env.get_waiting_planes()) > 0 and len(env.get_idle_transporters()) > 0:
                transporters = env.get_idle_transporters()
                planes = env.get_waiting_planes()
                assignments = arrange_transporters(transporters, planes)
                for device_type, device_code, target_site in assignments:
                    for idx, device in enumerate(env.mobile_devices[device_type]):
                        if device.resource.code == device_code:
                            action["devices"][device_type][idx] = target_site
                            break
            
            action["planes"] = [[[] for _ in range(pidx+1)] for _ in range(bidx+1)]
            for plane_id, plane in env.planes.items():
                bidx_, pidx_ = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])
                action["planes"][bidx_][pidx_] = [sampled_sites[pidx_], random.choice(jobs) if (jobs := plane.get_avail_jobs()) else None]
            if len(env.planes) and all([plane.is_completed_all_jobs() for plane in env.planes.values()]):
                takeoff_planes = []
                for site in env.get_avail_takeoff_sites():
                    for plane_id, plane in env.planes.items():
                        if not plane.is_transporting and not plane.is_busy:
                            bidx_, pidx_ = int(plane_id.split('_')[1]), int(plane_id.split('_')[2])
                            action["planes"][bidx_][pidx_] = [site, None]
                            takeoff_planes.append(plane_id)
                            break
            obs, rewards, done = env.step(action, env.step_time - step_time)
            step_time = env.step_time
            # takeoff_planes += remove_planes

            if done:
                print("All planes have taken off.")
                break 
            elif step_time == 0:
                continue
        else:
            step_time -= 1
        print(f"Step: {env.steps}, Step Time: {step_time}, Total Time: {total_time}")
        total_time += 1
        if total_time == landing_list[-1]:
            print("All planes have landed.")
        elif all([plane.is_completed_all_jobs() for plane in env.planes.values()]):
            print("All planes have completed their jobs.")
                                                                                                                                                                                                                                                                                                                    