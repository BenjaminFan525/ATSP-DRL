import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import argparse
from pathlib import Path

# ================= 1. 设置 Times New Roman 字体 (学术风) =================
# 方法一：系统自带 (Linux/Ubuntu)
# 必须先安装 ttf-mscorefonts-installer 并清空缓存
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['mathtext.fontset'] = 'stix' # 数学符号匹配ローマ体

# 方法二：自带字体文件 (最硬核, 无需权限)
# 假设你把 times.ttf 和 timesbd.ttf 放到了代码同目录下
# from matplotlib import font_manager as fm
# font_prop = fm.FontProperties(fname='./times.ttf')
# bold_prop = fm.FontProperties(fname='./timesbd.ttf')

# 这里为了代码兼容性，我们暂时使用 serif 通用设置。
# 如果你想百分百 Times，请取消方案二的注释并修改下方的 fontproperties 参数。
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 模拟中等规模算例的数据 =================
# 包含三架飞机：P01 (入境维护), P02 (入境维护), P03 (在港离港)
# Y轴类别设计：
# 上半区块：Sites (1~20, 29, Z, L)
# 下半区块：Bottleneck Mobile Resources (R014 拖车, R001 油车)
site_labels = ['Site Z', 'Site L', 'Site 1 (Gate)', 'Site 2 (Gate)', 'Site 3 (Gate)', 'Site 29 (Rwy)']
mobile_labels = ['R014 (Tow Truck)', 'R001 (Fuel Truck)']
y_labels = site_labels + ['---'] + mobile_labels # 加入一个分隔符

# 构建 Y 轴映射字典
y_mapping = {label: i for i, label in enumerate(reversed(y_labels))}

# 数据格式: [飞机ID, 类别(Y轴), 任务ID, 开始时间, 结束时间, 颜色主题]
data = [
    # ---- 飞机 P01 (入境维护, 蓝色系) ----
    ['P01', 'Site Z', 'ZY-Z', 10, 20, '#1976D2'],          # 落地
    ['P01', 'R014 (Tow Truck)', 'ZY-T (Towing Z to 1)', 20, 30, '#1976D2'], # 拖行Site Z -> Site 1
    ['P01', 'Site 1 (Gate)', 'ZY01-Fixing', 30, 31, '#1976D2'], # 到 Site 1 固定
    ['P01', 'Site 1 (Gate)', 'Maintenance jobs', 31, 80, '#2196F3'], # 其他维护作业
    ['P01', 'R001 (Fuel Truck)', 'ZY10-Refueling', 40, 50, '#1976D2'], # 加油 (Site 1, 等 R001 空闲)

    # ---- 飞机 P02 (入境维护, 红色系) ----
    ['P02', 'Site Z', 'ZY-Z', 15, 25, '#D32F2F'],          # 落地 (Site Z 空间互斥, 占用到25)
    # P02 需要 R014 拖行。但 P01 正在用 R014 到30。P02 需在 Taxi 区域等待。
    ['P02', 'R014 (Tow Truck)', 'ZY-T (Towing Z to 2)', 30, 40, '#D32F2F'], # 拖行Site Z -> Site 2 (Site Z 空间互斥释放 @ 30)
    ['P02', 'Site 2 (Gate)', 'ZY01-Fixing', 40, 41, '#D32F2F'], # 到 Site 2 固定
    ['P02', 'Site 2 (Gate)', 'Maintenance jobs', 41, 90, '#F44336'], # 其他维护作业
    ['P02', 'R001 (Fuel Truck)', 'ZY10-Refueling', 60, 70, '#D32F2F'], # 加油 (Site 2, 等 R001 空闲)

    # ---- 飞机 P03 (离港作业, 绿色系) ----
    # P03 已经在港夜航Site 1
    ['P03', 'Site 1 (Gate)', 'ZY-L (Unlock)', 0, 5, '#388E3C'],
    # P03 需要 R014 拖行到起飞Sites。Site 1 需等到 P03 拖行开始后释放。
    # R014 0-20 空闲。
    ['P03', 'R014 (Tow Truck)', 'ZY-T (Towing 1 to 29)', 5, 15, '#388E3C'], # 拖行Site 1 -> Site 29
    ['P03', 'Site 29 (Rwy)', 'ZY-S (Alignment)', 15, 25, '#388E3C'], # 到 Site 29
    ['P03', 'Site 29 (Rwy)', 'ZY-F (起飞)', 25, 26, '#388E3C'], # 起飞 (Site 29 释放 @ 26)
]

# 用于画运输耦合箭头的数据 [飞机ID, 源Y, 目标Y, 任务结束Y, 任务开始Y, 时间范围]
# 这用于展示 ΔT 和 Wait ΔT (Trans).
transport_delta = [
    ['P01', 'Site Z', 'R014 (Tow Truck)', 'ZY-Z', 'ZY-T (Towing Z to 1)', (20, 20)], # 转运箭头 ΔT=0 (直接拖)
    ['P01', 'R014 (Tow Truck)', 'Site 1 (Gate)', 'ZY-T (Towing Z to 1)', 'ZY01-Fixing', (30, 30)], # 拖行 ΔT=0
    
    ['P02', 'Site Z', 'R014 (Tow Truck)', 'ZY-Z', 'ZY-T (Towing Z to 2)', (25, 30)], # ΔT=0, Wait ΔT(Trans)=5 Wait for R014
    ['P02', 'R014 (Tow Truck)', 'Site 2 (Gate)', 'ZY-T (Towing Z to 2)', 'ZY01-Fixing', (40, 40)], # 拖行 ΔT=0
    
    ['P03', 'Site 1 (Gate)', 'R014 (Tow Truck)', 'ZY-L (Unlock)', 'ZY-T (Towing 1 to 29)', (5, 5)],
    ['P03', 'R014 (Tow Truck)', 'Site 29 (Rwy)', 'ZY-T (Towing 1 to 29)', 'ZY-S (Alignment)', (15, 15)],
]

# 创建 DataFrame
df = pd.DataFrame(data, columns=['AcID', 'Category', 'JobName', 'Start', 'End', 'Color'])
delta_df = pd.DataFrame(transport_delta, columns=['AcID', 'Source', 'Target', 'JobFrom', 'JobTo', 'TimePair'])

# ================= 3. 开始绘图 =================
# 创建一个跨栏的大画布
fig, ax = plt.subplots(figsize=(16, 7), dpi=300)

# 设置 Y 轴刻度和标签
ax.set_yticks(list(y_mapping.values()))
ax.set_yticklabels(list(y_mapping.keys()), fontsize=12)

# X 轴设置
ax.set_xlim(0, 100)
# 强制使用科学计数法 (学术感)
# ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
ax.set_xlabel('Time (t) [Minutes from dispatching start]', fontsize=14, labelpad=10)

# 美化边框与网格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)

# 分区横线 (Sites 和 Mobile 之间)
sep_y = y_mapping['---']
ax.axhline(sep_y, color='black', linestyle='-', linewidth=2.0, alpha=1.0)
ax.text(1, sep_y + 0.1, 'Operational Sites (Spatial)', fontsize=12, color='navy', fontweight='bold', ha='left')
ax.text(1, sep_y - 0.7, 'Mobile Resources', fontsize=12, color='saddlebrown', fontweight='bold', ha='left')

# 隐藏分隔符刻度标签
axes = plt.gca()
labels = [item.get_text() for item in axes.get_yticklabels()]
for i, label in enumerate(labels):
    if label == '---':
        labels[i] = ''
axes.set_yticklabels(labels)

# ================= 4. 绘制定制色块 (Gantt Blocks) =================
# 用于存储不同飞机ID，生成图例
legend_elements = {}

for index, row in df.iterrows():
    # 容错处理分隔符类别
    if row['Category'] == '---': continue
    
    y_pos = y_mapping[row['Category']]
    width = row['End'] - row['Start']
    
    # sites 块高一点，mobile 块矮一点，视觉更美观
    height = 0.8
    if row['Category'] in mobile_labels: height = 0.5
    
    # 绘制色块
    bar = patches.Rectangle((row['Start'], y_pos - height/2), width, height,
                            edgecolor='black', facecolor=row['Color'], alpha=0.9, linewidth=1.2, zorder=3)
    ax.add_patch(bar)
    
    # 色块内部文字 (学术感, 隐藏 AcID 只留 JobName, 确保字不溢出)
    if width > 5:
        font_size = 9
        if width > 15: font_size = 11
        ax.text(row['Start'] + width/2, y_pos, f"{row['AcID']}: {row['JobName']}", 
                ha='center', va='center', color='black', fontweight='bold', fontsize=font_size, zorder=4)

    # 记录图例元素 (按AcID)
    if row['AcID'] not in legend_elements:
        legend_elements[row['AcID']] = patches.Patch(color=row['Color'], label=f"AcID: {row['AcID']}")

# ================= 5. 绘制关键细节: 运输耦合箭头 ($\Delta T$) =================
for index, row in delta_df.iterrows():
    ac_id = row['AcID']
    y_source = y_mapping[row['Source']]
    y_target = y_mapping[row['Target']]
    
    # 查找飞机对应的颜色
    ac_color = df[df['AcID'] == ac_id]['Color'].iloc[0]
    
    # 时间点和位置
    t_start, t_end = row['TimePair']
    
    # 箭头风格 (学术感)
    # style = "-"
    if t_start == t_end: style = patches.ArrowStyle.CurveB(head_length=0.4, head_width=0.2)
    else: style = patches.ArrowStyle.Fancy(head_length=0.4, head_width=0.2, tail_width=0.01) # ΔT=0 用 CurveB, Wait ΔT 用曲线
    
    # 画箭头
    arrow = patches.FancyArrowPatch((t_start, y_source), (t_end, y_target),
                                   arrowstyle=style, linestyle='--', color=ac_color, linewidth=1.0, alpha=0.6,
                                   connectionstyle="arc3,rad=-0.1", zorder=2)
    ax.add_patch(arrow)

# ================= 6. 添加图例与标题 =================
# 按顺序排列图例
sorted_legend = [legend_elements['P01'], legend_elements['P02'], legend_elements['P03']]
# 将图例放在 Sites 块内部右上角，节省空间
ax.legend(handles=sorted_legend, loc='best', fancybox=True, shadow=False, ncol=3, fontsize=12, frameon=True)

# 加一个主标题
ax.set_title('AHMSP-HRC Exemplar Case Study: Optimized Collaborative Dispatching Plan', fontsize=16, pad=20, fontweight='bold')

# 保存为高清矢量图 (极力推荐格式)
parser = argparse.ArgumentParser(description="Plot the bundled Gantt case study.")
parser.add_argument(
    "--output",
    default=str(Path(__file__).resolve().parent / "figures/gantt_chart_casestudy.pdf"),
)
args = parser.parse_args()
output_path = os.path.abspath(os.path.expanduser(args.output))
plt.tight_layout()
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, format='pdf', bbox_inches='tight')
plt.close(fig)

print(f"🎉 AHMSP-HRC 定制甘特图样例已成功绘制: {output_path}")
