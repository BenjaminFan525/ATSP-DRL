"""
这个文件书写作业类，即抽象保障资源类
每个作业拥有:作业id,作业编码,作业名,作业时长
"""


class Jobs:
    def __init__(self):
        # 创建所有的job对象，并用列表index作为id的索引，相当于字典
        self.jobs_object_list = []
        # id: 0 1 2 3 4 5 6 7 8 9
        jobs_codes = ["ZCTF", "SBTF", "JY", "TYY", "TD", "YQ", "DQ", "GDDE", "GD", "CC"]
        jobs_names = ["座舱", "设备舱", "加油", "液压", "供电", "氧气", "氮气", "惯导", "挂弹", "出场"]
        jobs_times = [10, 10, 15, 4, 6, 2, 2, 10, 15, 1]   # 时间最快是74完成
        for i in range(len(jobs_names)):
            temp_object = Job(i, jobs_codes[i], jobs_names[i], jobs_times[i])
            self.jobs_object_list.append(temp_object)

    # 某个站位保留的jobs有哪些，因为不同保障位置的jobs不同
    # reserved_job_id :[0,2,3,1,...]
    def reserved_jobs(self, reserved_job_id):
        assert reserved_job_id != []
        reserved_jobs = []
        for id in reserved_job_id:
            reserved_jobs.append(self.jobs_object_list[id])
        self.jobs_object_list = reserved_jobs


class Job:
    def __init__(self, code: str, time, group, resources: list = [], predecessor: list = [], exclusive: list = []):
        self.code = code
        self.time = time if isinstance(time, (int, float)) else None
        self.group = group
        self.resources = resources
        self.predecessor = predecessor
        self.exclusive = exclusive


if __name__ == "__main__":
    import json
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
    print(jobs)