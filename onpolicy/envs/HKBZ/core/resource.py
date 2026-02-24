class Resource:
    # 资源类：定义一个可服务资源的属性和行为，用于管理各类资源（如人员、设备）的状态和服务能力
    def __init__(self, code: str, type: str, sites: list, max_service: int = 1):
        self.code = code  # 资源唯一标识码
        self.type = type  # 资源类型（如"机械师"、"检测设备"等）
        self.sites = sites  # 该资源可服务的站位列表
        self.on_service = []  # 当前正在服务的站点代码列表
        self.max_service = max_service  # 资源最大服务容量（可同时服务的站点数量）
        self.available = True  # 资源当前是否可用（True表示可接受新服务请求）

    def is_available(self):
        '''检查资源是否可用
        
        返回: bool - 如果当前服务数量小于最大容量返回True，否则返回False
        示例:
            if resource.is_available():  # 检查资源是否可分配
                resource.add_service("S01")
        '''
        return True if len(self.on_service) < self.max_service else False

    def add_service(self, site_code):
        '''为资源添加服务站点
        
        输入:
            site_code: str - 需要服务的站点代码
        作用:
            将站点添加到服务列表，并更新资源可用状态
        异常:
            AssertionError - 当资源不可用或站点不在支持列表中时触发
        示例:
            resource = Resource("R001", "机械师", ["S01", "S02"])
            resource.add_service("S01")  # 资源R001开始服务站点S01
        '''
        assert self.available == True, f"Resource {self.code} is not available!"
        assert site_code in self.sites, f"Site {site_code} is not supported by Resource {self.code}!"
        self.on_service.append(site_code)
        self.available = self.is_available()

    def remove_service(self, site_code):
        '''从资源服务列表中移除站点
        
        输入:
            site_code: str - 需要移除的站点代码
        作用:
            将站点从服务列表中移除，并更新资源可用状态
        异常:
            AssertionError - 当站点不在服务列表中时触发
        示例:
            resource.remove_service("S01")  # 资源R001停止服务站点S01
        '''
        assert site_code in self.on_service, f"Site {site_code} is not in service list of Resource {self.code}!"
        self.on_service.remove(site_code)
        self.available = self.is_available()