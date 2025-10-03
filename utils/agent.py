from utils.resource import Resource
from utils.site import Site


class Plane:
    def __init__(self, code, config):
        self.code = code
        self.config = config
        self.velocity = config['velocity']  # 飞机速度
        self.position : Site = config['position'].pos  # 位置
        self.current_jobs = []  # 当前任务
        self.finished_jobs = []  # 已完成的作业列表
        self.is_busy = False  # 是否忙碌
        self.destination = None  # 目的地
        self.is_transporting = False  # 是否在移动
        self.left_job_time = 0  # 当前任务剩余时间
        self.left_trans_time = 0  # 当前运输剩余时间

        self.jobs = {}
        for job in config['jobs']:
            if job.group == '保障':
                self.jobs[job.code] = job
            if job.time is None:
                if job.code == 'ZY02':
                    job.time = (100-self.config['hydraulic_oil'])//10
                elif job.code == 'ZY03':
                    job.time = (100-self.config['electricity'])//5
                elif job.code == 'ZY10':
                    job.time = (100-self.config['fuel'])//5
        self.left_jobs = list(self.jobs.keys())  # 初始剩余作业列表

    def get_avail_jobs(self):
        assert self.is_busy is False and self.is_transporting is False, "Current plane must be idle to get available jobs."
        return [job for job in self.left_jobs if self.jobs[job].predecessor.issubset(self.finished_jobs)]
    
    def get_parallel_jobs(self, job_code):
        parallel_jobs = []
        for job in self.get_avail_jobs():
            if job == job_code:
                parallel_jobs.append(job)
            elif job_code not in self.jobs[job].exclusive and self.left_job_time <= self.jobs[job].time:
                parallel_jobs.append(job)
        return parallel_jobs
    
    def choose_job(self, job_code):
        assert job_code in self.get_avail_jobs(), f"Job {job_code} is not available."
        self.is_busy = True
        self.left_job_time = self.jobs[job_code].time
        # TODO: del this line after testing
        assert self.left_job_time > 0, "Job time must be positive."
        self.current_jobs = self.get_parallel_jobs(job_code)

    def start_transport(self, destination: Site):
        assert self.is_busy is False and self.is_transporting is False, "Plane must be idle to start transporting."
        distance = abs(self.position[0]-destination[0]) + abs(self.position[1]-destination[1]) # 曼哈顿距离
        time = distance // self.velocity
        self.is_transporting = True
        self.left_trans_time = time
        self.destination = destination
        return time
    
    def step(self):
        assert not(self.is_busy and self.is_transporting), "Plane must be either busy or transporting."
        if self.is_busy:
            self.left_job_time -= 1
            if self.left_job_time == 0:
                self.finished_jobs += self.current_jobs
                self.left_jobs = [job for job in self.left_jobs if job not in self.current_jobs]
                self.current_jobs = []
                self.is_busy = False
        elif self.is_transporting:
            self.left_trans_time -= 1
            if self.left_trans_time == 0:
                self.position = self.destination
                self.destination = None
                self.is_transporting = False

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

class Device:
    def __init__(self, code, config):
        self.code = code
        self.config = config
        self.resource = config['resource']  # 携带的资源对象
        self.velocity = config['velocity']  # 飞机速度
        self.position : Site = config['position'].pos  # 位置
        self.is_transporting = False  # 是否在移动
        self.destination = None  # 目的地
        self.left_trans_time = 0  # 当前任务剩余时间

    def start_transport(self, destination: Site):
        assert self.is_transporting is False, "Device must be idle to start transporting."
        distance = abs(self.position[0]-destination[0]) + abs(self.position[1]-destination[1]) # 曼哈顿距离
        time = distance // self.velocity
        self.is_transporting = True
        self.left_trans_time = time
        self.destination = destination
        return time
    
    def step(self):
        assert not(self.is_busy and self.is_transporting), "Device must be either busy or transporting."
        if self.is_transporting:
            self.left_trans_time -= 1
            if self.left_trans_time == 0:
                self.position = self.destination
                self.destination = None
                self.is_transporting = False

    def reset(self):
        self.position = self.config['position'].pos  # 位置
        self.is_transporting = False
        self.destination = None
        self.left_trans_time = 0