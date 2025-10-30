# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agent import Plane, Device
from env_utils import Job, Resource, Site
import json

class ScheduleEnv(gym.Env):
    """
    兼容 Gymnasium 的最小网格世界环境。
    注册名: 'GridWorld-v0'
    """

    def __init__(self, config, render_mode: str = None):
        super().__init__()
        self.config = config

        self.num_planes = config['num_planes']
        self.render_mode = render_mode
        
        self.initialize(config)

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
                'velocity': 3,
                'position': self.sites[res.sites[0]]
                }
            if res.type not in self.mobile_devices:
                self.mobile_devices[res.type] = [Device(res.code, device_cfg)]
            else:
                self.mobile_devices[res.type].append(Device(res.code, device_cfg))
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_pos = np.array([0, 0], dtype=np.int32)
        return self._agent_pos.copy(), {}

    def step(self, action):
        delta = self.ACTION_TO_DELTA[action]
        new_pos = self._agent_pos + delta

        # 越界检查
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            terminated = True
            reward = -1
        else:
            self._agent_pos = new_pos
            terminated = np.array_equal(self._agent_pos, self._target_pos)
            reward = 0 if terminated else -1

        truncated = False
        return self._agent_pos.copy(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return
        grid = np.full((self.size, self.size), " ", dtype=str)
        grid[tuple(self._target_pos)] = "G"
        grid[tuple(self._agent_pos)] = "A"
        print("\n".join(["|" + "|".join(row) + "|" for row in grid]), end="\n\n")

    def close(self):
        pass

# 自动注册，方便 gym.make 调用
gym.register(
    id="GridWorld-v0",
    entry_point="grid_world_env:GridWorldEnv",
    max_episode_steps=50,
)