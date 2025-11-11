class Job:
    def __init__(self, code: str, time, group, resources: list = [], predecessor: list = [], exclusive: list = []):
        self.code = code
        self.time = time*60 if isinstance(time, (int, float)) else None
        self.group = group
        self.resources = resources
        self.predecessor = predecessor
        self.exclusive = exclusive

class Resource():
    def __init__(self, code: str, type: str, sites: list, max_service: int = 1):
        self.code = code
        self.type = type
        self.sites = sites  # 该资源可服务的站位列表
        self.on_service = []
        self.max_service = max_service
        self.available = True  # 代表当前是否可用

    def is_available(self):
        return True if len(self.on_service) < self.max_service else False

    def add_service(self, site_code):
        assert self.available == True, f"Resource {self.code} is not available!"
        assert site_code in self.sites, f"Site {site_code} is not supported by Resource {self.code}!"
        self.on_service.append(site_code)
        self.available = self.is_available()

    def remove_service(self, site_code):
        assert site_code in self.on_service, f"Site {site_code} is not in service list of Resource {self.code}!"
        self.on_service.remove(site_code)
        self.available = self.is_available()

class Site:
    def __init__(self, code, config):
        self.code = code
        self.config = config
        self.pos = config['position']  # 位置
        self.plane = None
        self.is_occupied = False  # 是否被占用
        self.is_interfered = False  # 是否被干涉
        self.left_rec_time = 0  # 剩余干涉时间
        self.onging_jobs = {}  # 当前正在进行的作业列表, key为作业id，value为剩余时间

        self.resources = {}  # 该站位的资源对象列表，key为资源id，value为资源对象
        for res in self.config['fixed_resources'] + self.config['mobile_resources']:
            if self.code in res.sites:
                self.resources[res.code] = res
        self.update_resources()

    def add_plane(self, plane):
        assert self.is_occupied is False, f"Site {self.code} is already occupied!"
        self.plane = plane
        self.is_occupied = True

    def remove_plane(self):
        assert self.is_occupied is True, f"Site {self.code} is already empty!"
        self.plane = None
        self.is_occupied = False

    def add_resource(self, resource: Resource):
        self.resources[resource.code] = resource
        self.update_resources()

    def remove_resource(self, resource: Resource):
        if resource.code in self.resources:
            del self.resources[resource.code]
        self.update_resources()

    def start_jobs(self, jobs: list[Job]):
        # TODO: 优化资源使用逻辑
        for job in jobs:
            assert job.code not in self.onging_jobs, f"Job {job.code} is already being serviced at Site {self.code}!"
            res = None
            for res_code in job.resources:
                if res_code in self.res_avail:
                    self.resources[self.res_avail[res_code][-1]].add_service(self.code)
                    res = self.res_avail[res_code][-1]
                    break
            self.onging_jobs[job.code] = [job.time, res]
    
    def finish_jobs(self, jobs: list[Job]):
        for job in jobs:
            assert job.code in self.onging_jobs, f"Job {job.code} is not being serviced at Site {self.code}!"
            if self.onging_jobs[job.code][1] is not None:
                self.resources[self.onging_jobs[job.code][1]].remove_service(self.code)
            self.onging_jobs.pop(job.code)

    def start_interfere(self):
        assert self.is_interfered is False, f"Site {self.code} is already interfered!"
        self.is_interfered = True

    def finish_interfere(self, rec_time):
        assert self.is_interfered is True, f"Site {self.code} is not being interfered!"
        self.is_interfered = False
        self.left_rec_time = rec_time

    def is_all_finished(self):
        return len(self.onging_jobs) == 0
    
    def update(self, time):
        ret = 0
        if self.is_interfered:
            return
        if self.left_rec_time > 0:
            self.left_rec_time -= time
            if self.left_rec_time > 0:
                return
            else:
                self.is_interfered = False
        finished_jobs = []
        for job_code in self.onging_jobs:
            self.onging_jobs[job_code][0] -= time
            if self.onging_jobs[job_code][0] <= 0:
                finished_jobs.append(job_code)
            else:
                ret = max(ret, self.onging_jobs[job_code][0])
        if len(finished_jobs) > 0:
            self.finish_jobs([self.config['jobs'][job] for job in finished_jobs])
        return ret
    
    def update_resources(self):
        '''更新该站位的可用资源类型字典'''
        self.res_avail = {}
        for code, res in self.resources.items():
            if res.is_available():
                if res.type in self.res_avail:
                    self.res_avail[res.type].append(code)
                else:
                    self.res_avail[res.type] = [code]

    def reset(self):
        self.plane = None
        self.is_occupied = False
        self.is_interfered = False
        self.left_rec_time = 0
        self.onging_jobs = {}
        for res in self.resources.values():
            res.reset()
        self.update_resources()

if __name__ == "__main__":
    import json
    import numpy as np
    import math

    # test for job class
    # jobs_path = 'utils/config/jobs.json'
    # with open(jobs_path, 'r') as f:
    #         data = json.load(f)
    # jobs = [Job(code=item["作业编号"], 
    #             time=item["作业时间"], 
    #             group=item["分组"], 
    #             resources=item["需要设备类型"] if isinstance(item["需要设备类型"], list) else [], 
    #             predecessor=item["前置作业"] if isinstance(item["前置作业"], list) else [], 
    #             exclusive=item["互斥作业"] if isinstance(item["互斥作业"], list) else [])
    #         for item in data]
    # print(jobs)

    # # test for resource class
    # fixed_res_path = 'utils/config/fixed_resources.json'
    # mobile_res_path = 'utils/config/mobile_resources.json'
    # with open(fixed_res_path, 'r') as f:
    #         data = json.load(f)
    # fixed_resources = [Resource(item["设备编号"], 
    #                             item["类型"], 
    #                             [str(idx) for idx in range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1])+1)],
    #                             max_service=5) 
    #                             for item in data]
    # with open(mobile_res_path, 'r') as f:
    #     data = json.load(f)
    # mobile_resources = [Resource(item["设备编号"], 
    #                              item["类型"], 
    #                              [item["初始停机位"]], 
    #                              max_service=1) 
    #                              for item in data]    
    # print(fixed_resources)

    jobs_path = 'utils/config/jobs.json'
    with open(jobs_path, 'r') as f:
            data = json.load(f)
    jobs = [Job(code=item["作业编号"], 
                time=item["作业时间"], 
                group=item["分组"], 
                resources=item["需要设备类型"] if isinstance(item["需要设备类型"], list) else [], 
                predecessor=item["前置作业"] if isinstance(item["前置作业"], list) else [], 
                exclusive=item["互斥作业"] if isinstance(item["互斥作业"], list) else [])
            for item in data]

    fixed_res_path = 'utils/config/fixed_resources.json'
    mobile_res_path = 'utils/config/mobile_resources.json'
    with open(fixed_res_path, 'r') as f:
            data = json.load(f)
    fixed_resources = [Resource(item["设备编号"], 
                                item["类型"], 
                                [str(idx) for idx in range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1])+1)],
                                max_service=5) 
                                for item in data]
    with open(mobile_res_path, 'r') as f:
        data = json.load(f)
    mobile_resources = [Resource(item["设备编号"], 
                                 item["类型"], 
                                 [item["初始停机位"]], 
                                 max_service=1) 
                                 for item in data]  
    
    sites_path = 'utils/config/sites.json'
    with open(sites_path, 'r') as f:
        data = json.load(f)
    sites = [Site(code, {'position': pos, 
                         'jobs': jobs, 
                         'fixed_resources': [res for res in fixed_resources if code in res.sites], 
                         'mobile_resources': [res for res in mobile_resources if code in res.sites]}) for code, pos in zip(data['sites_codes'], data['sites_positions'])]