from env_utils import Resource, Site, Job
import math
import numpy as np

class Device:
    def __init__(self, code, config):
        self.code = code
        self.config = config
        self.resource : Resource = config['resource']  # 携带的资源对象
        self.velocity = config['velocity']  # 飞机速度
        self.site : Site = config['site']  # 位置
        self.is_transporting = False  # 是否在移动
        self.left_trans_time = 0  # 当前任务剩余时间
        self.is_busy = False  # 是否忙碌
        self.site.devices.append(self)

    def start_transport(self, destination: Site):
        assert self.is_transporting is False, "Device must be idle to start transporting!"
        assert self.resource.is_available(), f"Resource must be idle to start transporting!"
        self.site.remove_resource(self.resource)
        distance = abs(self.site.pos[0]-destination.pos[0]) + abs(self.site.pos[1]-destination.pos[1]) # 曼哈顿距离
        time = math.ceil(distance / self.velocity)
        self.is_transporting = True
        self.left_trans_time = time
        self.site = destination
        return time

    def finish_transport(self):
        # assert self.is_transporting is True and self.destination is not None, "Device must be transporting to finish transport."
        self.is_transporting = False
        self.resource.sites = [self.site.code]
        self.site.add_resource(self.resource)
        self.site.devices.append(self)
    
    def is_idle(self):
        return not self.is_busy and not self.is_transporting
    
    def update(self, time):
        assert not self.is_idle(), "Device must be either busy or transporting."
        ret = np.inf
        if self.is_transporting:
            self.left_trans_time -= time
            # if self.left_trans_time <= 0 and self.resource.type != 'R014':
            assert self.left_trans_time >= 0, "Device transport time cannot be negative."
            if self.left_trans_time == 0:
                self.finish_transport()
            return self.left_trans_time    
        self.is_busy = not self.resource.available
        return ret

    def reset(self):
        self.site = self.config['site']  # 位置
        self.is_transporting = False
        self.destination = None
        self.left_trans_time = 0
        self.is_busy = False

class Plane:
    def __init__(self, code, config):
        self.code = code
        self.config = config
        self.velocity = config['velocity']  # 飞机速度
        self.site : Site = config['site']  # 位置
        self.choosed_job = None  # 选择的作业
        self.current_jobs = []  # 当前任务
        self.finished_jobs = []  # 已完成的作业列表
        self.is_busy = False  # 是否忙碌
        self.destination = None  # 目的地
        self.transporter = None  # 转运车
        self.is_transporting = False  # 是否在移动
        self.left_job_time = 0  # 当前任务剩余时间
        self.left_trans_time = 0  # 当前运输剩余时间
        self.is_waiting = False  # 是否在等待
        self.waiting_time = 0  # 等待时间

        self.jobs = {}
        for job in config['jobs']:
            if job.group == '保障' and job.code not in ['ZY01', 'ZY-L']:
                self.jobs[job.code] = job
            if job.time is None:
                if job.code == 'ZY10':
                    job.time = math.ceil((100-self.config['fuel'])/5*60)
            job.predecessor = set(job.predecessor)
            job.resources = set(job.resources)
        self.left_jobs = list(self.jobs.keys())  # 初始剩余作业列表

    def get_avail_jobs(self):
        # assert not self.is_busy and not self.is_transporting, \
        #     "Current plane must be idle to get available jobs."
        avail = []                               # 存放所有可选作业编码
        for job_code in self.left_jobs:          # 遍历剩余作业
            job = self.jobs[job_code]            # 当前作业对象
            # 1. 前序作业必须全部完工
            if job.predecessor.issubset(self.finished_jobs):
                # 2. 资源需求检查
                self.site.update_resources()
                if len(job.resources) == 0 or not job.resources.isdisjoint(self.site.res_avail.keys()):                   
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
        self.site.start_jobs([self.jobs[job] for job in self.current_jobs])
        return self.jobs[job_code].time

    def start_transport(self, destination: Site, transporter: Device):
        # assert self.is_busy is False and self.is_transporting is False, "Plane must be idle to start transporting."
        self.site.remove_plane()
        distance = abs(self.site.pos[0]-destination.pos[0]) + abs(self.site.pos[1]-destination.pos[1]) # 曼哈顿距离
        time = math.ceil(distance / self.velocity)
        if self.site.code == 'Z':
            time += 1*60  # 加上固定的时间
        elif destination.code in ['29', '30', '31']:
            time += 5*60  # 解固+调整姿态+起飞
        else:
            time += 2*60  # 解固+固定
        self.is_transporting = True
        self.left_trans_time = time
        if transporter is not None:
            self.transporter = transporter
            transporter.start_transport(destination)
            if self.site.code == 'Z':
                transporter.left_trans_time = time - 1*60
            elif destination.code in ['29', '30', '31']:
                transporter.left_trans_time = time - 4*60
            else:
                transporter.left_trans_time = time - 2*60
        
        self.site = destination
        self.site.add_plane(self)

        if transporter is not None:
            return transporter.left_trans_time
        else:
            return time
    
    def finish_transport(self):
        assert self.is_transporting is True, "Plane must be transporting to finish transport."
        if self.transporter:
            # self.transporter.finish_transport()
            self.transporter = None
        if not self.is_completed_all_jobs():
            self.finished_jobs = [item for item in self.finished_jobs if item not in ['ZY02', 'ZY03']]  # 重置长占作业
            for job_code in ['ZY02', 'ZY03']:
                if job_code not in self.left_jobs:
                    self.left_jobs.append(job_code)
        self.is_transporting = False
    
    def start_waiting(self):
        assert self.is_busy is False and self.is_transporting is False, "Plane must be idle to start waiting."
        self.is_waiting = True
        self.waiting_time = 0

    def finish_waiting(self):
        assert self.is_waiting is True, "Plane must be waiting to finish waiting."
        self.is_waiting = False
        self.waiting_time = 0

    def is_completed_all_jobs(self):
        return len(self.left_jobs) == 0

    def is_idle(self):
        return not self.is_busy and not self.is_transporting and not self.is_waiting
    
    def update(self, time):
        assert not(self.is_busy and self.is_transporting), "Plane must be either busy or transporting."
        ret = np.inf
        if self.is_waiting:
            self.waiting_time += time
            if self.destination != self.site and self.site.get_avail_transporter() is not None:
                self.finish_waiting()
                ret = self.start_transport(self.destination, self.transporter)
            elif self.choosed_job is not None or self.choosed_job in self.get_avail_jobs():
                self.finish_waiting()
                ret = self.choose_job(self.choosed_job)
                self.choosed_job = None
                    
        elif self.is_transporting:
            self.left_trans_time -= time
            assert self.left_trans_time >= 0, "Plane transport time cannot be negative."
            if self.left_trans_time <= 0:
                self.finish_transport()
                if self.choosed_job is not None:
                    ret = self.choose_job(self.choosed_job)
                    self.choosed_job = None
            ret = self.left_trans_time

        elif self.is_busy:
            ret = self.site.update(time)
            if self.site.is_all_finished():
                self.finished_jobs += self.current_jobs
                self.left_jobs = [job for job in self.left_jobs if job not in self.current_jobs]
                self.current_jobs = []
                self.is_busy = False
        return ret

    def reset(self):
        self.velocity = self.config['velocity']  # 飞机速度
        self.site = self.config['site']  # 位置
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
    sites = {code: Site(code, {'site': pos, 
                         'jobs': jobs, 
                         'fixed_resources': [res for res in fixed_resources.values() if code in res.sites], 
                         'mobile_resources': [res for res in mobile_resources.values() if code in res.sites]}) for code, pos in zip(data['sites_codes'], data['sites_positions'])}

    mobile_devices = {}
    for res in mobile_resources.values():
        device_cfg = {
            'resource': res,
            'velocity': 3,
            'site': sites[res.sites[0]]
            }
        if res.type not in mobile_devices:
            mobile_devices[res.type] = [Device(res.code, device_cfg)]
        else:
            mobile_devices[res.type].append(Device(res.code, device_cfg))


    for idx in range(5):
        plane_cfg = {
            'velocity': 5,
            'site': sites[str(idx+1)],
            'fuel': np.random.randint(0, 30),
            'jobs': jobs.values()
        }
        plane = Plane(f'Plane_{idx}', plane_cfg)
        print(plane.get_avail_jobs())
        plane.choose_job('ZY02')