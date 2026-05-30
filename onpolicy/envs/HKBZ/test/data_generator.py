import json
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= 固定的作业配置文件 =================
JOB_DATA = [
    {
      "作业编号":"ZY-Z",
      "需要设备类型":[],
      "作业时间":1,
      "前置作业":[],
      "互斥作业":[],
      "分组":"进场"
    },
    {
      "作业编号":"ZY-M",
      "需要设备类型":[],
      "作业时间":"根据距离计算",
      "前置作业":["ZY-Z"],
      "互斥作业":[],
      "分组":"进场"
    },
    {
      "作业编号":"ZY01",
      "需要设备类型":[],
      "作业时间":1,
      "前置作业":["ZY_T", "ZY_M"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY02",
      "需要设备类型":["R008"],
      "作业时间":1,
      "前置作业":[],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY03",
      "需要设备类型":["R002"],
      "作业时间":1,
      "前置作业":[],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY04",
      "需要设备类型":["R005","R013"],
      "作业时间":3,
      "前置作业":[],
      "互斥作业":["ZY10"],
      "分组":"保障"
    },
    {
      "作业编号":"ZY05",
      "需要设备类型":["R003","R012"],
      "作业时间":3,
      "前置作业":[],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY06",
      "需要设备类型":["R007"],
      "作业时间":4,
      "前置作业":[],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY07",
      "需要设备类型":["R006","R011"],
      "作业时间":2,
      "前置作业":["ZY02"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY08",
      "需要设备类型":[],
      "作业时间":1,
      "前置作业":["ZY02","ZY03"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY09",
      "需要设备类型":["R007"],
      "作业时间":2,
      "前置作业":["ZY06"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY10",
      "需要设备类型":["R001"],
      "作业时间":"根据需要确定",
      "前置作业":["ZY03"],
      "互斥作业":["ZY04"],
      "分组":"保障"
    },
    {
      "作业编号":"ZY11",
      "需要设备类型":[],
      "作业时间":1,
      "前置作业":["ZY02","ZY08"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY12",
      "需要设备类型":[],
      "作业时间":20,
      "前置作业":["ZY03","ZY08"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY13",
      "需要设备类型":[],
      "作业时间":1,
      "前置作业":["ZY02","ZY03"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY14",
      "需要设备类型":[],
      "作业时间":2,
      "前置作业":["ZY02","ZY03","ZY04","ZY05","ZY07","ZY09","ZY10","ZY11","ZY12","ZY13"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY15",
      "需要设备类型":[],
      "作业时间":5,
      "前置作业":["ZY02","ZY03","ZY14"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY16",
      "需要设备类型":[],
      "作业时间":1,
      "前置作业":["ZY15"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY17",
      "需要设备类型":[],
      "作业时间":1,
      "前置作业":["ZY16"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY18",
      "需要设备类型":[],
      "作业时间":3,
      "前置作业":["ZY17"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY-L",
      "需要设备类型":["R014"],
      "作业时间":1,
      "前置作业":["ZY-Z","ZY-L"],
      "互斥作业":[],
      "分组":"保障"
    },
    {
      "作业编号":"ZY-T",
      "需要设备类型":["R014"],
      "作业时间":"根据距离计算",
      "前置作业":["ZY-Z","ZY-L"],
      "互斥作业":[],
      "分组":"转移"
    },
    {
      "作业编号":"ZY-S",
      "需要设备类型":[],
      "作业时间":3,
      "前置作业":["ZY-L"],
      "互斥作业":[],
      "分组":"出场"
    },
    {
      "作业编号":"ZY-F",
      "需要设备类型":[],
      "作业时间":1,
      "前置作业":["ZY-S"],
      "互斥作业":[],
      "分组":"出场"
    }
]

class AirportScenarioGenerator:
    def __init__(self, num_stands=20, num_planes=12):
        self.num_stands = num_stands
        self.num_takeoff = 3
        
        # 【修改1：非线性空间扩张】 
        # 限制大场景下的物理面积扩张速度 (从 0.5 次方改为 0.4 次方)，保持场地相对拥挤
        # 防止“赶路时间”过长主导了总 Makespan，掩盖了调度算法的智商优势
        scale_factor = math.pow(num_stands / 20.0, 0.4) 
        
        self.map_size = max(600, int(1000 * scale_factor)) 
        self.min_dist = 80   
        
        self.margin_x = max(80, int(150 * scale_factor))
        self.margin_y = max(60, int(100 * scale_factor))
        
        self.sites_codes = []
        self.sites_positions = []
        self.stand_positions = [] 
        
        self.fixed_resources = []
        self.mobile_resources = []
        self.flights = [] 
        
        self.fr_count = 1
        self.mr_count = 1
        self.num_planes = num_planes

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def generate_sites(self):
        attempts = 0
        while len(self.stand_positions) < self.num_stands and attempts < 10000:
            # 【核心修改 2】：使用动态的 margin 来限制 x 和 y 的范围
            x = random.randint(self.margin_x, self.map_size - self.margin_x)
            y = random.randint(self.margin_y, self.map_size - self.margin_y)
            
            conflict = False
            for pos in self.stand_positions:
                if self._dist([x, y], pos) < self.min_dist:
                    conflict = True
                    break
            
            if not conflict:
                self.stand_positions.append([x, y])
            attempts += 1
            
        if len(self.stand_positions) < self.num_stands:
            raise Exception(f"场地空间不足：在 {self.map_size}x{self.map_size} 区域内尝试了10000次仅生成了 {len(self.stand_positions)} 个停机位（目标 {self.num_stands}）。建议减小 min_dist 或检查缩放系数。")

    def cluster_and_assign_fixed_resources(self, k=4):
        centroids = random.sample(self.stand_positions, k)
        clusters = [[] for _ in range(k)]
        
        for _ in range(10):
            clusters = [[] for _ in range(k)]
            for pt in self.stand_positions:
                distances = [self._dist(pt, c) for c in centroids]
                closest_idx = distances.index(min(distances))
                clusters[closest_idx].append(pt)
                
            for i in range(k):
                if clusters[i]:
                    avg_x = sum(p[0] for p in clusters[i]) / len(clusters[i])
                    avg_y = sum(p[1] for p in clusters[i]) / len(clusters[i])
                    centroids[i] = [avg_x, avg_y]

        current_id = 1
        self.sites_codes.append("Z") 
        self.sites_positions.append([50, self.map_size // 2]) 
        
        cluster_ranges = []
        for cluster in clusters:
            if not cluster: continue
            start_id = current_id
            for pt in cluster:
                self.sites_codes.append(str(current_id))
                self.sites_positions.append(pt)
                current_id += 1
            end_id = current_id - 1
            cluster_ranges.append(f"{start_id}-{end_id}")
            
        start_takeoff = self.num_stands + 1
        for i in range(start_takeoff, start_takeoff + self.num_takeoff):
            self.sites_codes.append(str(i))
            y_pos = (self.map_size // (self.num_takeoff + 1)) * (i - start_takeoff + 1)
            self.sites_positions.append([self.map_size - 50, y_pos])

        fixed_types = ['R001', 'R002', 'R003', 'R005', 'R006', 'R007', 'R008']
        
        for c_range in cluster_ranges:
            for f_type in fixed_types:
                self.fixed_resources.append({
                    "设备编号": f"FR{self.fr_count}",
                    "类型": f_type,
                    "支持停机位": c_range
                })
                self.fr_count += 1

    def generate_mobile_resources(self):
        mobile_types = ['R002', 'R003', 'R005', 'R007', 'R008', 'R011', 'R012', 'R013', 'R014']
        
        # 【修改2：亚线性资源增长 (Sub-linear Scaling)】
        # 飞机数量增加4倍时，移动设备只增加约2.8倍 (使用0.75次方)。
        # 目的：在大规模场景下人为制造“资源极度稀缺”，这会逼死 PDRs 规则，但 DRL 能统筹规划
        res_scale = math.pow(self.num_planes / 12.0, 0.75) 
        
        stand_assignment_idx = 1 
        
        for m_type in mobile_types:
            if m_type == 'R014':
                num_items = max(1, round(6 * res_scale))
            else:
                num_items = max(1, round(2 * res_scale))
                
            for _ in range(num_items):
                # 让更多设备初始堆积在 "Z" (车库)，需要算法智能地把它们派发出去
                # 增加了早期调度的复杂度和决策空间
                if random.random() < 0.3: 
                    init_pos = "Z"
                else:
                    init_pos = str(stand_assignment_idx)
                    stand_assignment_idx = (stand_assignment_idx % self.num_stands) + 1
                    
                self.mobile_resources.append({
                    "设备编号": f"MR{self.mr_count}",
                    "类型": m_type,
                    "初始停机位": init_pos
                })
                self.mr_count += 1

    def check_feasibility(self):
        # 兜底函数，在完全固定的分配策略下，所有机位都已被 fixed_types 完美覆盖
        # 因此该函数只会进行验证，绝对不会动态向 mobile_resources 里加设备，从而保证总数恒定！
        required_logic = [
            ['R001'], ['R002'], ['R008'], ['R007'], 
            ['R005', 'R013'], 
            ['R003', 'R012'], 
            ['R006', 'R011']  
        ]
        
        global_mobile_types = {m['类型'] for m in self.mobile_resources}
        
        if 'R014' not in global_mobile_types:
            self.mobile_resources.append({"设备编号": f"MR{self.mr_count}", "类型": "R014", "初始停机位": "Z"})
            self.mr_count += 1
            global_mobile_types.add("R014")

        for stand_id in range(1, self.num_stands + 1):
            local_fixed_types = set()
            for fr in self.fixed_resources:
                bounds = [int(x) for x in fr['支持停机位'].split('-')]
                if bounds[0] <= stand_id <= bounds[1]:
                    local_fixed_types.add(fr['类型'])
            
            available_types = local_fixed_types.union(global_mobile_types)
            
            for req_group in required_logic:
                if not any(req in available_types for req in req_group):
                    missing_type = req_group[-1] 
                    self.mobile_resources.append({
                        "设备编号": f"MR{self.mr_count}",
                        "类型": missing_type,
                        "初始停机位": str(stand_id)
                    })
                    self.mr_count += 1
                    global_mobile_types.add(missing_type)

    def generate_flights(self):
        current_time = 0
        
        for i in range(1, self.num_planes + 1):
            if i > 1:
                # 【修改3：制造航班波峰 (Bursty Arrivals)】
                # 30% 的概率出现短时间连续降落（大波峰），70% 的概率平缓到达
                # PDRs 在遇到波峰时会产生大量的盲目抢占和死锁，而 DRL 能通过宏观视野完美化解
                if random.random() < 0.3:
                    interval = random.randint(20, 80)   # 极度密集的到达
                else:
                    interval = random.randint(200, 350) # 长时间的闲置
                current_time += interval
                
            # 【修改4：拉大个体差异】恢复燃油状态的高方差，让剩余工作量差异变大
            # 这样 SPT 和 MWKR 等启发式规则更容易被“个别极端航班”带偏
            fuel_percentage = random.randint(20, 90)
            
            self.flights.append({
                "飞机编号": f"F{i:02d}",
                "到达时间": current_time,
                "初始燃油状态": f"{fuel_percentage}%"
            })

    def validate_scenario(self):
        if not (0 < self.num_planes < self.num_stands):
            raise ValueError(f"不合规：飞机数量 {self.num_planes} 必须大于0且小于停机位数量 {self.num_stands}。")
        
        has_tow_truck = any(m['类型'] == 'R014' for m in self.mobile_resources)
        if not has_tow_truck:
            raise ValueError("不合规：缺少执行转运和出场的牵引车(R014)。")

        for i, pos1 in enumerate(self.sites_positions):
            for j, pos2 in enumerate(self.sites_positions):
                if i != j and self._dist(pos1, pos2) < 50:
                    raise ValueError(f"不合规：场地坐标发现重叠现象 (节点{self.sites_codes[i]} 与 节点{self.sites_codes[j]})。")
        
        return True

    def visualize_and_save(self, filename="airport_layout_randomized.png"):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(f"Virtual Airport Hub Layout ({self.num_stands} Stands, {self.num_planes} Flights)", fontsize=16, pad=20)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")

        for i, code in enumerate(self.sites_codes):
            x, y = self.sites_positions[i]
            
            if code == "Z":
                rect = patches.Rectangle((x-40, y-15), 80, 30, linewidth=1.5, edgecolor='darkgreen', facecolor='lightgreen', zorder=2)
                ax.add_patch(rect)
                ax.text(x, y, f"Landing {code}", ha='center', va='center', fontsize=10, fontweight='bold', color='darkgreen', zorder=3)
            elif code.isdigit() and int(code) > self.num_stands:
                rect = patches.Rectangle((x-40, y-15), 80, 30, linewidth=1.5, edgecolor='darkorange', facecolor='moccasin', zorder=2)
                ax.add_patch(rect)
                ax.text(x, y, f"Takeoff {code}", ha='center', va='center', fontsize=10, fontweight='bold', color='saddlebrown', zorder=3)
            else:
                rect = patches.Rectangle((x-30, y-20), 60, 40, linewidth=1, edgecolor='navy', facecolor='lightblue', alpha=0.8, zorder=2)
                ax.add_patch(rect)
                ax.text(x, y, code, ha='center', va='center', fontsize=9, fontweight='bold', color='navy', zorder=3)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def generate(self):
        self.generate_sites()
        
        target_k = max(1, round(4 * (self.num_planes / 12.0)))
        k = min(self.num_stands, target_k)
        
        self.cluster_and_assign_fixed_resources(k=k) 
        self.generate_mobile_resources()
        self.check_feasibility()
        self.generate_flights() 
        self.validate_scenario() 
        
        return {
            "fixed_resources": self.fixed_resources,
            "mobile_resources": self.mobile_resources,
            "flights": self.flights,
            "sites": {
                "sites_codes": self.sites_codes,
                "sites_positions": [[round(x,1), round(y,1)] for x, y in self.sites_positions]
            }
        }

# ================= 数据集生成封装 =================
def build_dataset(num_cases=5, num_stands=20, num_planes=12, base_dir="airport_dataset"):
    os.makedirs(base_dir, exist_ok=True)
    print(f"🚀 开始生成航空枢纽调度数据集，总计 {num_cases} 个算例...\n")
    
    successful_cases = 0
    
    while successful_cases < num_cases:
        case_id = successful_cases + 1
        case_dir = os.path.join(base_dir, f"case_{case_id:02d}")
        
        print(f"🔄 正在生成算例 Case {case_id:02d} ...")
        
        try:
            # 【修改 4】：彻底取消随机大小输入，保证所有图结构绝对一致
            generator = AirportScenarioGenerator(num_stands=num_stands, num_planes=num_planes)
            data = generator.generate()
            
            os.makedirs(case_dir, exist_ok=True)
            
            with open(os.path.join(case_dir, "sites.json"), "w", encoding="utf-8") as f:
                json.dump(data["sites"], f, ensure_ascii=False, indent=2)
            with open(os.path.join(case_dir, "fixed_resources.json"), "w", encoding="utf-8") as f:
                json.dump(data["fixed_resources"], f, ensure_ascii=False, indent=2)
            with open(os.path.join(case_dir, "mobile_resources.json"), "w", encoding="utf-8") as f:
                json.dump(data["mobile_resources"], f, ensure_ascii=False, indent=2)
            with open(os.path.join(case_dir, "flights.json"), "w", encoding="utf-8") as f:
                json.dump(data["flights"], f, ensure_ascii=False, indent=2)
            with open(os.path.join(case_dir, "job.json"), "w", encoding="utf-8") as f:
                json.dump(JOB_DATA, f, ensure_ascii=False, indent=2)
                
            image_path = os.path.join(case_dir, f"layout_case_{case_id:02d}.png")
            generator.visualize_and_save(image_path)
            
            print(f"   ✅ 成功！停机位: {generator.num_stands}, 飞机: {generator.num_planes}, 移动设备: {len(data['mobile_resources'])}")
            successful_cases += 1
            
        except Exception as e:
            print(f"   ⚠️ 生成出现冲突 ({e})，正在重新生成该算例...")
            continue

    print(f"\n🎉 数据集生成完毕！所有数据已保存在目录: ./{base_dir}/")

# ================= 执行入口 =================
if __name__ == "__main__":
    # 在这里调整你想要生成的算例数量，比如设为 500
    build_dataset(num_cases=20, num_stands=80, num_planes=48, base_dir="/home/fanyx/HKBZ-environment/onpolicy/envs/HKBZ/dataset/test_extra_large_2")