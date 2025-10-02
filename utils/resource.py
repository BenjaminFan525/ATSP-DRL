"""
资源类
包含:...

"""
import json
import numpy as np
import math

class Resources:
    def __init__(self, fixed_res_path, mobile_res_path):

        with open(fixed_res_path, 'r') as f:
            data = json.load(f)
        self.fixed_resources = [FixedResource(item["设备编号"], 
                                          item["类型"], 
                                          range(int(item["支持停机位"].split("-")[0]), int(item["支持停机位"].split("-")[1]))) 
                                          for item in data]
        with open(mobile_res_path, 'r') as f:
            data = json.load(f)
        self.mobile_resources = [MobileResource(item["设备编号"], 
                                          item["类型"], 
                                          item["初始停机位"]) 
                                          for item in data]        
        

class FixedResource():
    def __init__(self, id: str, job: str, site_range: tuple):
        self.id = id
        self.job = job
        self.sites = [str(idx) for idx in range(site_range[0], site_range[1])]  # 该资源可服务的站位列表
        self.on_service = []
        self.available = True  # 代表当前是否可用

    def is_available(self):
        self.available = True if len(self.on_service) < 5 else False
    
    def add_service(self, site_id):
        assert self.available == True, f"Resource {self.id} is not available!"
        self.on_service.append(site_id)
        self.is_available()

    def remove_service(self, site_id):
        assert site_id in self.on_service, f"Site {site_id} is not in service list of Resource {self.id}!"
        self.on_service.remove(site_id)
        self.is_available()

        
class MobileResource():
    def __init__(self, id, job, site):
        self.id = id
        self.job = job
        self.site = site  # 位置
        self.countdown = 0  # 移动倒计时
        self.available = True  # 代表当前是否可用

    def set_new_site(self, new_site: str, distance: float, speed: float):
        self.site = new_site
        self.countdown = math.ceil(distance / speed)

    def update(self):
        if self.countdown > 0:
            self.countdown -= 1
        self.available = True if self.countdown == 0 else False


if __name__ == "__main__":
    res = Resources('utils/config/fixed_resources.json', 'utils/config/mobile_resources.json')
    print(1)