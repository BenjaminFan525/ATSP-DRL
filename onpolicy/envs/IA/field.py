import numpy as np
from geopandas import GeoSeries
import utm
import json
import matplotlib.pyplot as plt
import networkx as nx
from .utils import gen_node, get_euler_dis, gen_edge, get_dijkstra_path, get_dijkstra_path_length
import torch

class boundaryCoord:
    def __init__(self, coord, child = None, father = None) -> None:
        self.coord = coord
        self.add_child(child)
        self.add_father(father)
        self.root = False

    def add_child(self, child):
        self.child = child
        if self.child is not None:
            self.distance = np.linalg.norm(self.coord - self.child.coord)
            self.dir = (self.child.coord - self.coord) / self.distance

    def add_father(self, father):
        self.father = father

class Boundary:
    def __init__(self, coords):
        self.coords = coords
        self.root_coord = boundaryCoord(coords[0])
        self.num_coords = len(coords)
        self.matrix = np.stack((coords, np.roll(coords, -1, axis=0)))
        self.vector = self.matrix[1] - self.matrix[0]
        curr = self.root_coord
        # self.dir = []
        # for coord in coords[1:]:
        #     curr.add_child(boundaryCoord(coord, father=curr))
        #     self.dir.append(curr.dir)
        #     curr = curr.child
        # self.root_coord.add_father(curr)
        # curr.add_child(self.root_coord)
        # self.dir.append(curr.dir)
        # self.dir = np.array(self.dir)
        self.boundary_lenth = []
        for boundary in self.vector:
            self.boundary_lenth.append(np.linalg.norm(boundary))
        
        self.dir = np.zeros((len(self.boundary_lenth), 2)) # direction of each boundary
        for idx, boundary in enumerate(self.vector):
            self.dir[idx] = boundary / self.boundary_lenth[idx]

    # def draw_coord(self, coord):
    #     for idx in range()


# def gen_node(type, field, idx):
#     return type + "_" + str(field) + '_' + str(int(idx))

class Field():
    def __init__(self, whole_field: GeoSeries, working_width, working_direction = None, field_num = 0, 
                 n_direction = 1, dis_matrix = True, utm_base = None) -> None:
        self.field_num = field_num
        if utm_base is None:
            self.utm = utm.from_latlon(whole_field.centroid.y[0], whole_field.centroid.x[0])
        else:
            self.utm = utm_base
        if whole_field.crs.coordinate_system.name == 'ellipsoidal':
            self.whole_field = whole_field.to_crs(crs="+proj =utm +zone={} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(self.utm[2]))#"EPSG:4479"
        else:
            self.whole_field = whole_field
        # print(self.whole_field.area)
        polygon = self.whole_field.geometry.to_json()
        polygon_dict = json.loads(polygon)
        self.whole_coords = np.array(polygon_dict["features"][0]["geometry"]["coordinates"][0][:-1])
        self.num_whole_coords = len(self.whole_coords)
        # the boundary of whole area
        self.whole_boundary = np.stack((self.whole_coords, np.roll(self.whole_coords, -1, axis=0)))
        self.whole_boundary_vector = self.whole_boundary[1] - self.whole_boundary[0]
        self.whole_boundary_lenth = []
        for boundary in self.whole_boundary_vector:
            self.whole_boundary_lenth.append(np.linalg.norm(boundary))
        
        self.whole_boundary_dir = np.zeros((len(self.whole_boundary_lenth), 2)) # direction of each boundary
        for idx, boundary in enumerate(self.whole_boundary_vector):
            self.whole_boundary_dir[idx] = boundary / self.whole_boundary_lenth[idx]

        self.working_width = working_width
        self.working_field = self.whole_field.buffer(- 2 * self.working_width, resolution=1)

        # get coordinates of the working area
        if self.working_field.geometry.is_empty[0]:
            self.working_field = None
            self.working_lines = np.array([])
            self.num_working_lines = 0
            self.generate_graph()
            return
        
        polygon = self.working_field.geometry.to_json()
        polygon_dict = json.loads(polygon)        
        self.working_coords = np.array(polygon_dict["features"][0]["geometry"]["coordinates"][0][:-1])
        self.num_working_coords = len(self.working_coords)
        
        # the boundary of working area
        self.working_boundary = np.stack((self.working_coords, np.roll(self.working_coords, -1, axis=0)))
        self.working_boundary_vector = self.working_boundary[1] - self.working_boundary[0]
        self.boundary_lenth = []
        for boundary in self.working_boundary_vector:
            self.boundary_lenth.append(np.linalg.norm(boundary))
        
        self.working_boundary_dir = np.zeros((len(self.boundary_lenth), 2)) # direction of each boundary
        for idx, boundary in enumerate(self.working_boundary_vector):
            self.working_boundary_dir[idx] = boundary / self.boundary_lenth[idx]

        # determine working direction
        if working_direction is None:
            self.working_direction = self.cal_working_dir()
        else:
            self.working_direction = working_direction
        
        self.working_direction = self.working_direction.reshape((-1, 1))
        self.direction_angle = np.arctan2(self.working_direction[1], self.working_direction[0]).item()
        self.anti_direction_angle = self.direction_angle + (np.pi if self.direction_angle < 0 else - np.pi)
        
        # determine working lines
        self.n_direction = n_direction
        self.cal_lines()
        self.generate_graph()
        if dis_matrix:
            self.gen_distance_matrix()

    # Working direction is the direction on whose nomal vector the boudaries have the smallest sum of |projection|.
    # The working direction is proved to be parallel to one of the boundaries
    # 作业方向的法向量上，所有边界的投影1范数之和最小
    # 作业方向平行于某一边界
    def cal_working_dir(self):        
        # 计算投影
        proj = self.working_boundary_vector @ (np.array([[0, 1],[-1, 0]]) @ self.working_boundary_dir.T)
        proj = abs(proj)
        proj = np.sum(proj, axis=0)
        return self.working_boundary_dir[np.argmin(proj)]
    
    def cal_lines(self):
        # A line is represented as bias * v_t + coeff * t
        # where t is a unit vector represents the direction of the line,
        # v_t is the normal vector of t 
        def cal_cross_points(dir, v_dir, bias_start, bias_end, width, boundary, boundary_dir, v_boundary, coords):
            '''
            dir: 作业方向
            v_dir: 作业法向
            bias_start: 第一作业行
            bias_end: 末尾作业行
            width: 行间距
            boundary: 边界
            boundary_dir: 边界方向矢量(nx2)
            v_boundary: 边界方向法向量
            coords: 边界顶点

            Return
            cross_list: numpy.array 长度，起点，终点

            '''
            bias_boundary = []
            bias_boundary_vector = []
            coeff_boundary = []
            unit_coeff_boundary_proj = boundary_dir @ v_dir # 边在作业行法线上产生的单位投影
            cross_boundary = np.nonzero(unit_coeff_boundary_proj)[0] # 会与作业行产生交点的边
            unit_coeff_boundary_proj = unit_coeff_boundary_proj[cross_boundary]
            for idx in cross_boundary:
                bias_boundary.append(coords[idx] @ v_boundary[:, idx])
                bias_boundary_vector.append(bias_boundary[-1] * v_boundary[:, idx]) # 各边的截距矢量
                coeff_boundary.append(boundary[:, idx, :] @ boundary_dir[idx].reshape((-1, 1))) # 各边起始点的偏置量
            
            bias_boundary = np.array(bias_boundary)
            bias_boundary_vector = np.array(bias_boundary_vector).T
            coeff_boundary = np.sort(np.array(coeff_boundary).reshape((-1, 2)), axis=1)

            bias_boundary_proj = bias_boundary_vector.T @ v_dir
            start_coeff = (bias_start - bias_boundary_proj) / unit_coeff_boundary_proj # 首行与各边交点的偏置
            width_coeff = width / unit_coeff_boundary_proj

            # self.num_working_lines = int((end - start) // self.working_width) # the number of working lines
            lines = [] # list(range(self.num_working_lines))
            line_idx = 0
            while bias_start < bias_end:
                cross = []
                cross_idx = []
                cross_lower = start_coeff.squeeze(1) - coeff_boundary[:, 0] >= - 1e-3
                cross_upper = start_coeff.squeeze(1) - coeff_boundary[:, 1] <= 1e-3
                for idx_cross, idx_boundary in enumerate(cross_boundary):
                    # 如果相交，cross中增加交点，用相交的边的bias和coeff计算
                    if cross_lower[idx_cross] and cross_upper[idx_cross]:
                        cross.append(bias_boundary_vector[:, idx_cross].reshape((1, -1)) + start_coeff[idx_cross] * boundary_dir[idx_boundary])
                        cross_idx.append(idx_boundary)
                # 有些地块可能是非凸多边形的
                # 所以可能一条作业行与边界有多个交点
                # 通常来说在地块内的部分比较长
                # 此处以间距最长的相邻两点作为作业行的首尾

                # 交点按作业行方向上的coeff从小到大排列
                cross = np.array(cross).squeeze(1)
                cross_idx = np.array(cross_idx, dtype=np.int8)
                cross_working_dir_proj = (cross @ dir).squeeze(1)
                order = np.argsort(cross_working_dir_proj)
                cross = cross[order]
                cross_idx = cross_idx[order]
                cross_working_dir_proj = cross_working_dir_proj[order]

                # 找到最长线段
                line_lenth = 0
                start_point = None
                for idx in range(len(cross) - 1):
                    if line_lenth < cross_working_dir_proj[idx + 1] - cross_working_dir_proj[idx]:
                        line_lenth = cross_working_dir_proj[idx + 1] - cross_working_dir_proj[idx]
                        start_point = idx
                
                if start_point is not None:
                    # 作业行表示形式
                    # [作业行编号，作业行长度，起始点，终点，起点交线编号，终点交线编号]
                    # 由于所有交点是按作业方向上的coeff排序的，所以所有作业行的起始点都在coeff小的那一侧
                    lines.append(np.hstack((np.array([line_idx, line_lenth]), cross[start_point], cross[start_point + 1], cross_idx[start_point], cross_idx[start_point + 1])))
                    line_idx += 1
                
                # next line
                start_coeff += width_coeff    
                bias_start += self.working_width
            return np.array(lines)
        
        v_dir = np.array([[0, 1],[-1, 0]]) @ self.working_direction # normal vector of working direction
        v_boundary = np.array([[0, 1],[-1, 0]]) @ self.working_boundary_dir.T
        v_boundary_w = np.array([[0, 1],[-1, 0]]) @ self.whole_boundary_dir.T
        bias = self.working_coords @ v_dir # the projections of each coord on v_dir
        start = np.min(bias) # the working line No.0
        end = np.max(bias) 
        self.working_lines = cal_cross_points(self.working_direction, v_dir, start, end, self.working_width, self.working_boundary, self.working_boundary_dir, v_boundary, self.working_coords)
        self.working_lines = np.hstack((self.working_lines, cal_cross_points(self.working_direction, v_dir, start, end, self.working_width, self.whole_boundary, self.whole_boundary_dir, v_boundary_w, self.whole_coords)[:, -6:]))
        self.num_working_lines = len(self.working_lines)
        if self.n_direction == -1:
            self.working_lines = self.working_lines[::-1]
            self.working_lines[:, 0] = np.arange(self.num_working_lines)

    def generate_graph(self):
        self.Graph = nx.Graph()
        self.working_graph = nx.DiGraph()
        self.Graph.add_nodes_from([(gen_node('bound', self.field_num, 0), {'coord': self.whole_coords[0]})])
        for idx, coord in enumerate(self.whole_coords[1:]):
            self.Graph.add_nodes_from([(gen_node('bound', self.field_num, idx + 1), {'coord': coord})])
            self.Graph.add_weighted_edges_from([(gen_node('bound', self.field_num, idx), gen_node('bound', self.field_num, idx + 1), self.whole_boundary_lenth[idx])])
        self.Graph.add_weighted_edges_from([(gen_node('bound', self.field_num, 0), gen_node('bound', self.field_num, idx + 1), self.whole_boundary_lenth[-1])])

        if self.working_field is None:
            return 
        
        for idx, l in enumerate(self.working_lines):
            line = l.copy()
            if idx != line[0]:
                line[0] = idx
            self.Graph.add_nodes_from([
                (gen_node('line_0', self.field_num, idx), {'coord': line[2:4]}),
                (gen_node('line_1', self.field_num, idx), {'coord': line[4:6]}),
                (gen_node('line_b_0', self.field_num, idx), {'coord': line[8:10]}),
                (gen_node('line_b_1', self.field_num, idx), {'coord': line[10:12]}),
            ])
            self.Graph.add_weighted_edges_from([
                self.gen_edge(gen_node('line_0', self.field_num, idx), gen_node('line_b_0', self.field_num, idx)),
                self.gen_edge(gen_node('line_1', self.field_num, idx), gen_node('line_b_1', self.field_num, idx)),
            ])
            if idx > 0:
                self.Graph.add_weighted_edges_from([
                    self.gen_edge(gen_node('line_0', self.field_num, idx), gen_node('line_0', self.field_num, idx - 1)),
                    self.gen_edge(gen_node('line_1', self.field_num, idx), gen_node('line_1', self.field_num, idx - 1)),
                ])
                if line[-2] == self.working_lines[idx - 1][-2]:
                    self.Graph.add_weighted_edges_from([
                        self.gen_edge(gen_node('line_b_0', self.field_num, idx), gen_node('line_b_0', self.field_num, idx - 1))
                    ])
                else:
                    self.Graph.add_weighted_edges_from([
                        self.gen_edge(gen_node('line_b_0', self.field_num, idx), gen_node('bound', self.field_num, line[-2])),
                        self.gen_edge(gen_node('line_b_0', self.field_num, idx), gen_node('bound', self.field_num, (line[-2] + 1) if line[-2] < (self.num_whole_coords - 1) else 0)),
                        self.gen_edge(gen_node('line_b_0', self.field_num, idx - 1), gen_node('bound', self.field_num, self.working_lines[idx - 1][-2])),
                        self.gen_edge(gen_node('line_b_0', self.field_num, idx - 1), gen_node('bound', self.field_num, (self.working_lines[idx - 1][-2] + 1) if self.working_lines[idx - 1][-2] < (self.num_whole_coords - 1) else 0)),
                    ])
                if line[-1] == self.working_lines[idx - 1][-1]:
                    self.Graph.add_weighted_edges_from([
                        self.gen_edge(gen_node('line_b_1', self.field_num, idx), gen_node('line_b_1', self.field_num, idx - 1))
                    ])
                else:
                    self.Graph.add_weighted_edges_from([
                        self.gen_edge(gen_node('line_b_1', self.field_num, idx), gen_node('bound', self.field_num, line[-1])),
                        self.gen_edge(gen_node('line_b_1', self.field_num, idx), gen_node('bound', self.field_num, (line[-1] + 1) if line[-1] < (self.num_whole_coords - 1) else 0)),
                        self.gen_edge(gen_node('line_b_1', self.field_num, idx - 1), gen_node('bound', self.field_num, self.working_lines[idx - 1][-1])),
                        self.gen_edge(gen_node('line_b_1', self.field_num, idx - 1), gen_node('bound', self.field_num, (self.working_lines[idx - 1][-1] + 1) if self.working_lines[idx - 1][-1] < (self.num_whole_coords - 1) else 0)),
                    ])
            else:
                self.Graph.add_weighted_edges_from([
                    self.gen_edge(gen_node('line_b_0', self.field_num, idx), gen_node('bound', self.field_num, line[-2])),
                    self.gen_edge(gen_node('line_b_0', self.field_num, idx), gen_node('bound', self.field_num, (line[-2] + 1) if line[-2] < (self.num_whole_coords - 1) else 0)),
                    self.gen_edge(gen_node('line_b_1', self.field_num, idx), gen_node('bound', self.field_num, (line[-1] + 1) if line[-1] < (self.num_whole_coords - 1) else 0)),
                    self.gen_edge(gen_node('line_b_1', self.field_num, idx), gen_node('bound', self.field_num, line[-1])),
                ])
        if len(self.working_lines) > 0:
            line = self.working_lines[-1].copy()
            self.Graph.add_weighted_edges_from([
                self.gen_edge(gen_node('line_b_0', self.field_num, idx), gen_node('bound', self.field_num, line[-2])),
                self.gen_edge(gen_node('line_b_0', self.field_num, idx), gen_node('bound', self.field_num, (line[-2] + 1) if line[-2] < (self.num_whole_coords - 1) else 0)),
                self.gen_edge(gen_node('line_b_1', self.field_num, idx), gen_node('bound', self.field_num, (line[-1] + 1) if line[-1] < (self.num_whole_coords - 1) else 0)),
                self.gen_edge(gen_node('line_b_1', self.field_num, idx), gen_node('bound', self.field_num, line[-1])),
            ])

        
        self.working_graph.add_nodes_from([
            (gen_node('line', self.field_num, idx),
             {'length': line[1],
              'direction_angle': self.direction_angle,
              'anti_direction_angle': self.anti_direction_angle,
              'end0': gen_node('line_0', self.field_num, idx),
              'end1': gen_node('line_1', self.field_num, idx),
              'embed': torch.Tensor([np.sin(self.direction_angle), np.cos(self.direction_angle), line[1]])
              }) 
            for idx, line in enumerate(self.working_lines)])
             
    
    def get_euler_dis(self, node1, node2):
        return get_euler_dis(self.Graph, node1, node2)

    def gen_edge(self, node1, node2):
        return gen_edge(self.Graph, node1, node2)

    def get_dijkstra_path(self, node1: tuple, node2: tuple, cartesian=False):
        return get_dijkstra_path(self.Graph, node1, node2, cartesian)
    
    def get_dijkstra_path_length(self, node1: tuple, node2: tuple):
        return get_dijkstra_path_length(self.Graph, node1, node2)

    def gen_distance_matrix(self):
        self.D_matrix = np.zeros((self.num_working_lines, self.num_working_lines, 2, 2))
        for idx1 in range(self.num_working_lines):
            for idx2 in range(idx1, self.num_working_lines):
                for p1 in [0, 1]:
                    for p2 in [0, 1]:
                        node1 = ('line_' + str(p1), self.field_num, idx1)
                        node2 = ('line_' + str(p2), self.field_num, idx2)
                        self.D_matrix[idx1, idx2, p1, p2] = self.D_matrix[idx2, idx1, p2, p1] = self.get_dijkstra_path_length(node1, node2)
        
    def render(self, ax=None, working_lines=True, entry_point=False, show=True, boundary=False, theme='harvest'):
        """
        全能渲染函数，支持多种农业场景风格切换。
        
        :param theme: 风格主题，可选值：
            - 'tillage': 耕作/播种季 (深褐色土壤，对比度高，推荐)
            - 'growth':  生长旺季 (绿色作物，适合植保场景)
            - 'harvest': 收获季 (金黄色作物，适合收割场景)
            - 'academic': 现代学术风 (淡蓝+奶黄，清爽，适合原理讲解)
        """
        
        # --- 1. 定义配色方案库 (高亮/清爽版) ---
        style_map = {
            # 方案 A: 耕作/播种季 (核心：土) -> 调整为“沙壤土”风格
            'tillage': {
                'bg_color': '#EFE5D9',      # 极浅的米灰色 (原干燥土壤调浅)
                'bg_edge': '#D7C4BB',       # 浅藕荷色边缘
                'work_color': '#B08B69',    # 浅栗色/奶咖色 (原深褐调浅)
                'work_alpha': 0.9,          # 稍微降低一点点透明度
                'boundary': '#8D6E63',      # 浅棕色田埂
                'grid': '#A1887F',          # 浅褐参考线
                'canvas': '#FFFAF5'         # 画布底色 (接近纯白但带暖意)
            },
            
            # 方案 B: 生长旺季 (核心：绿) -> 调整为“初夏嫩绿”风格
            'growth': {
                'bg_color': '#E8E8E8',      # 浅灰白路面 (原灰褐调浅)
                'bg_edge': '#D3D3D3',
                'work_color': '#66BB6A',    # 鲜草绿/嫩绿 (原森林绿调浅，更清新)
                'work_alpha': 0.8,          # 增加通透感，像阳光穿过叶子
                'boundary': '#689F38',      # 浅橄榄绿
                'grid': '#4CAF50',          # 中绿色参考线
                'canvas': '#F5FDF5'         # 画布底色 (极淡的薄荷白)
            },
            
            # 方案 C: 收获季 (核心：金) -> 调整为“浅麦色”风格
            'harvest': {
                'bg_color': '#FAF0E6',      # 亚麻色/洗白的旧布 (原干草色调浅)
                'bg_edge': '#E0D0B8',
                'work_color': '#F4D03F',    # 明亮的向日葵黄/成熟小麦 (原暗金调亮)
                'work_alpha': 0.85,
                'boundary': '#CD853F',      # 秘鲁色 (浅红褐)
                'grid': '#DAA520',          # 金麒麟色参考线
                'canvas': '#FFFFF0'         # 画布底色 (象牙白)
            },
            
            # 方案 D: 现代学术风 (核心：蓝/黄) -> 调整为“极简水彩”风格
            'academic': {
                'bg_color': '#F2F9FF',      # 极浅的幽灵白 (几乎接近白)
                'bg_edge': '#D6EAF8',
                'work_color': '#FFF5E0',    # 极浅的奶油色
                'work_alpha': 1.0,
                'boundary': '#999999',      # 中灰 (原深灰调浅)
                'grid': '#DDDDDD',          # 极浅灰
                'canvas': '#FFFFFF'         # 纯白
            }
        }

        
        # 获取当前主题配置，默认为 'tillage'
        s = style_map.get(theme, style_map['tillage'])

        # --- 2. 初始化画布 ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            ax.axis('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # 设置画布外的背景色，使图表更融合
            fig.patch.set_facecolor(s['canvas'])
            ax.patch.set_facecolor(s['canvas'])

        # --- 3. 绘制整体田块 (背景) ---
        try:
            self.whole_field.plot(ax=ax, color=s['bg_color'], edgecolor=s['bg_edge'], linewidth=1, zorder=0)
        except AttributeError:
            from descartes import PolygonPatch
            patch = PolygonPatch(self.whole_field, fc=s['bg_color'], ec=s['bg_edge'], lw=1, zorder=0)
            ax.add_patch(patch)

        # --- 4. 绘制边界 (田埂/路) ---
        if boundary:
            for p1, p2 in zip(self.whole_boundary[0], self.whole_boundary[1]):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=2.5, color=s['boundary'], zorder=2)

        # --- 5. 绘制作业区域 (核心区) ---
        if self.working_field is not None:
            try:
                self.working_field.plot(ax=ax, color=s['work_color'], alpha=s['work_alpha'], edgecolor='none', zorder=1)
            except AttributeError:
                patch = PolygonPatch(self.working_field, fc=s['work_color'], alpha=s['work_alpha'], ec='none', zorder=1)
                ax.add_patch(patch)
            
            # --- 6. 绘制作业行参考线 ---
            if working_lines:
                for line in self.working_lines:
                    # 使用主题定义的 grid 颜色
                    ax.plot(line[[2, 4]], line[[3, 5]], '--', color=s['grid'], linewidth=0.8 if theme=='academic' else 1.0, alpha=0.6, zorder=1)
                    
                    if boundary:
                        # 封口线颜色通常跟随边界颜色
                        ax.plot(line[[8, 2]], line[[9, 3]], '-', color=s['boundary'], linewidth=1, zorder=2)
                        ax.plot(line[[10, 4]], line[[11, 5]], '-', color=s['boundary'], linewidth=1, zorder=2)
                    
                    if entry_point:
                        ax.plot(line[2], line[3], '.w', markersize=3, zorder=3)
                        ax.plot(line[4], line[5], '.k', markersize=3, zorder=3)

        if show:
            plt.show()
        return ax