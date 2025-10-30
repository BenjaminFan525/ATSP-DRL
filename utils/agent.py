from env_utils import Resource, Site, Job

class Device:
    def __init__(self, code, config):
        self.code = code
        self.config = config
        self.resource : Resource = config['resource']  # 携带的资源对象
        self.velocity = config['velocity']  # 飞机速度
        self.position : Site = config['position'].pos  # 位置
        self.is_transporting = False  # 是否在移动
        self.destination = None  # 目的地
        self.left_trans_time = 0  # 当前任务剩余时间

    def start_transport(self, destination: Site):
        assert self.is_transporting is False, "Device must be idle to start transporting!"
        assert self.resource.is_available(), f"Resource must be idle to start transporting!"
        distance = abs(self.position[0]-destination[0]) + abs(self.position[1]-destination[1]) # 曼哈顿距离
        time = distance // self.velocity
        self.is_transporting = True
        self.left_trans_time = time
        self.destination = destination
        self.position.remove_resource(self.resource)
        return time

    def finish_transport(self):
        assert self.is_transporting is False and self.destination is not None, "Device must be transporting to finish transport."
        self.position = self.destination
        self.destination = None
        self.is_transporting = False
        self.resource.sites = [self.position.code]
        self.position.add_resource(self.resource)
    
    def step(self):
        assert not(self.is_busy and self.is_transporting), "Device must be either busy or transporting."
        if self.is_transporting:
            self.left_trans_time -= 1
            if self.left_trans_time == 0:
                self.finish_transport()

    def reset(self):
        self.position = self.config['position'].pos  # 位置
        self.is_transporting = False
        self.destination = None
        self.left_trans_time = 0

class Plane:
    def __init__(self, code, config):
        self.code = code
        self.config = config
        self.velocity = config['velocity']  # 飞机速度
        self.position : Site = config['position']  # 位置
        self.current_jobs = []  # 当前任务
        self.finished_jobs = []  # 已完成的作业列表
        self.is_busy = False  # 是否忙碌
        self.destination = None  # 目的地
        self.is_transporting = False  # 是否在移动
        self.left_job_time = 0  # 当前任务剩余时间
        self.left_trans_time = 0  # 当前运输剩余时间

        self.jobs = {}
        for job in config['jobs']:
            if job.group == '保障' and job.code not in ['ZY01', 'ZY-L']:
                self.jobs[job.code] = job
            if job.time is None:
                if job.code == 'ZY02':
                    job.time = (100-self.config['hydraulic_oil'])//10
                elif job.code == 'ZY03':
                    job.time = (100-self.config['electricity'])//5
                elif job.code == 'ZY10':
                    job.time = (100-self.config['fuel'])//5
            job.predecessor = set(job.predecessor)
            job.resources = set(job.resources)
        self.left_jobs = list(self.jobs.keys())  # 初始剩余作业列表

    def get_avail_jobs(self):
        assert not self.is_busy and not self.is_transporting, \
            "Current plane must be idle to get available jobs."

        avail = []                               # 存放所有可选作业编码
        for job_code in self.left_jobs:          # 遍历剩余作业
            job = self.jobs[job_code]            # 当前作业对象
            # 1. 前序作业必须全部完工
            if job.predecessor.issubset(self.finished_jobs):
                # 2. 资源需求检查
                if len(job.resources) == 0 or not job.resources.isdisjoint(self.position.res_avail.keys()):                   
                    avail.append(job_code)
        return avail

    def get_parallel_jobs(self, job_code):
        parallel_jobs = []
        for job in self.get_avail_jobs():
            if job == job_code:
                parallel_jobs.append(job)
            # 允许选择的并行作业：与所选作业不互斥，并且当前作业时间大于等于其他作业时间
            elif job_code not in self.jobs[job].exclusive and self.jobs[job_code].time >= self.jobs[job].time:
                parallel_jobs.append(job)
        return parallel_jobs
    
    def choose_job(self, job_code):
        assert job_code in self.get_avail_jobs(), f"Job {job_code} is not available."
        self.current_jobs = self.get_parallel_jobs(job_code)
        self.is_busy = True
        self.position.start_jobs([self.jobs[job] for job in self.current_jobs])

    def start_transport(self, destination: Site):
        assert self.is_busy is False and self.is_transporting is False, "Plane must be idle to start transporting."
        self.position.remove_plane()
        distance = abs(self.position[0]-destination[0]) + abs(self.position[1]-destination[1]) # 曼哈顿距离
        time = distance // self.velocity
        self.is_transporting = True
        self.left_trans_time = time
        self.destination = destination
        return time
    
    def finish_transport(self):
        assert self.is_transporting is False and self.destination is not None, "Plane must be transporting to finish transport."
        self.position = self.destination
        self.destination = None
        self.is_transporting = False
    
    def step(self):
        assert not(self.is_busy and self.is_transporting), "Plane must be either busy or transporting."
        if self.is_busy:
            self.position.step()
            if self.position.is_all_finished():
                self.finished_jobs += self.current_jobs
                self.left_jobs = [job for job in self.left_jobs if job not in self.current_jobs]
                self.current_jobs = []
                self.is_busy = False
        elif self.is_transporting:
            self.left_trans_time -= 1
            if self.left_trans_time == 0:
                self.finish_transport()

    def reset(self):
        self.velocity = self.config['velocity']  # 飞机速度
        self.position = self.config['position'].pos  # 位置
        self.current_jobs = []
        self.finished_jobs = []
        self.is_busy = False
        self.destination = None
        self.is_transporting = False
        self.left_job_time = 0  
        self.left_trans_time = 0
        self.left_jobs = list(self.jobs.keys())

if __name__ == "__main__":
    import json
    import numpy as np

    jobs_path = 'utils/config/jobs.json'
    with open(jobs_path, 'r') as f:
            data = json.load(f)
    jobs = {item["作业编号"] : Job(code=item["作业编号"], 
                time=item["作业时间"], 
                group=item["分组"], 
                resources=item["需要设备类型"] if isinstance(item["需要设备类型"], list) else [], 
                predecessor=item["前置作业"] if isinstance(item["前置作业"], list) else [], 
                exclusive=item["互斥作业"] if isinstance(item["互斥作业"], list) else [])
            for item in data}

    fixed_res_path = 'utils/config/fixed_resources.json'
    mobile_res_path = 'utils/config/mobile_resources.json'
    with open(fixed_res_path, 'r') as f:
            data = json.load(f)
    fixed_resources = {item["设备编号"]: Resource(item["设备编号"], 
                                item["类型"], 
                                [str(idx) for idx in range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1])+1)],
                                max_service=5) 
                                for item in data}
    with open(mobile_res_path, 'r') as f:
        data = json.load(f)
    mobile_resources = {item["设备编号"]: Resource(item["设备编号"], 
                                 item["类型"], 
                                 [item["初始停机位"]], 
                                 max_service=1) 
                                 for item in data} 
    
    sites_path = 'utils/config/sites.json'
    with open(sites_path, 'r') as f:
        data = json.load(f)
    sites = {code: Site(code, {'position': pos, 
                         'jobs': jobs, 
                         'fixed_resources': [res for res in fixed_resources.values() if code in res.sites], 
                         'mobile_resources': [res for res in mobile_resources.values() if code in res.sites]}) for code, pos in zip(data['sites_codes'], data['sites_positions'])}

    mobile_devices = {}
    for res in mobile_resources.values():
        device_cfg = {
            'resource': res,
            'velocity': 3,
            'position': sites[res.sites[0]]
            }
        if res.type not in mobile_devices:
            mobile_devices[res.type] = [Device(res.code, device_cfg)]
        else:
            mobile_devices[res.type].append(Device(res.code, device_cfg))


    for idx in range(5):
        plane_cfg = {
            'velocity': 5,
            'position': sites[str(idx+1)],
            'hydraulic_oil': np.random.randint(20, 60),
            'electricity': np.random.randint(0, 50),
            'fuel': np.random.randint(0, 30),
            'jobs': jobs.values()
        }
        plane = Plane(f'Plane_{idx}', plane_cfg)
        # TODO: fix bug here
        print(plane.get_avail_jobs())
        plane.choose_job('ZY02')