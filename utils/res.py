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


if __name__ == "__main__":
    import json
    import numpy as np
    import math
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
    print(fixed_resources)