"""
战位的相关信息，被环境调用
A-R 20个战位
每个站位拥有:站位id,绝对位置,可以进行的作业对象列表,可以进行作业id列表
"""
import numpy as np
from utils.job import Jobs
import copy
import random


# 所有战位的类
class Sites:

    def __init__(self, planes_num):

        # 所有战位对象
        self.sites_object_list = []
        sites_codes = ['z','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', "14", '15', '16',
                       '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28','29', '30', '31']
        # 站位位置
        self.sites_positions = [
            [10, 70],
            [45,110], [35,110], [25,100], [25, 90], [35, 80], [45, 80],
            [95,120], [85,120], [75,110], [85,100], [95,100], [105,90], [95, 80], [85, 80], 
            [45, 60], [35, 60], [25, 50], [35, 40], [45, 40], [55, 30], [45, 20], [35, 20], 
            [95, 50], [85, 50], [75, 40], [75, 30], [85, 20], [95, 20],
            [130,80], [130,70], [130,60]
            ]
        
        # 前18个站位资源数量随机生成
        jobs_num = 10
        sites_resources_range = [[i for i in range(jobs_num-1)] for _ in range(len(sites_codes)-3)]  # [0,1,...,8]*18
        resource_number = [[0]*(jobs_num) for _ in range(len(sites_codes)-3)]   # [0*10]*18
        for a in range(len(resource_number)):
            while sum(resource_number[a]) == 0:
                for i in range(jobs_num-1):
                    resource_sum = 0
                    while resource_sum < planes_num:
                        for j in range(len(resource_number)):
                            resource_number[j][i] = random.randint(0, 2)
                            resource_sum += resource_number[j][i]
        for i in range(len(resource_number)):
            for j in range(jobs_num-1):
                if resource_number[i][j] == 0:
                    sites_resources_range[i].remove(j)
        # 最后几个出场站位只能执行出场作业且资源无限
        sites_resources_range.extend([[jobs_num-1]]*3)
        resource_number.extend([[0,0,0,0,0,0,0,0,0,100]]*3)
        
        
        for i in range(len(sites_codes)):
            temp_object = Site(i, self.sites_positions[i], sites_resources_range[i], resource_number[i])
            self.sites_object_list.append(temp_object)

        
    
    def update_site_resources(self, action_id, job_id):
        # 更新该站位的资源列表
        assert self.sites_object_list[action_id].resource_number[job_id] >= 1
        self.sites_object_list[action_id].resource_number[job_id] -= 1
        if self.sites_object_list[action_id].resource_number[job_id] == 0:
            temp = copy.deepcopy(self.sites_object_list[action_id].resource_ids_list)
            temp.remove(job_id)
            self.sites_object_list[action_id].update_resorces(temp)


# 每一个战位的类
class Site:
    def __init__(self, site_id: int, relative_position: tuple, resource_ids_list: list, resource_number: int):
        self.site_id = site_id
        self.absolute_position = np.array(relative_position)
        self.resource_jobs = Jobs()
        self.resource_jobs.reserved_jobs(resource_ids_list)
        self.resource_ids_list = resource_ids_list  # 代表这个site可以进行job的id列表
        self.resource_number = resource_number  # 该站位可执行作业资源数量

    # 由于资源抢占关系而更新战位的资源列表
    def update_resorces(self, new_resource_ids_list: list):
        self.resource_ids_list = new_resource_ids_list


