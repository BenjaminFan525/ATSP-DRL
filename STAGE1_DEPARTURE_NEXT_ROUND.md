# Stage-1 离场语义下一轮研究计划（2026-08-12）

## 1. 研究问题与边界

新环境加入了渐进式离场流程：飞机在保障作业完成后，可以先腾挪机位，再由
R014 转运至跑道执行 `ZY-S/ZY-F`。现有 IGA 回放已经说明，这不是旧环境上的小扰动：
决策步数、转运次数、尾部下界和资源瓶颈均发生了系统变化。因此下一轮不直接沿用旧
potential，也不把网络扩大作为首要变量，而是回答四个因果问题：

1. 新 IGA 轨迹能否通过“保障 / 待离场转运 / 强制离场”分阶段 BC 被可靠模仿？
2. R014、跑道和已完工飞机占位信息进入观测后，策略是否能学会提前腾位与排队？
3. 精确的团队剩余时间目标是否比动作级 `cmax_delta` 更适合新的长离场尾部？
4. departure-aware potential 能否在不改变最终 `-Cmax` 目标的前提下改善信用分配？

本轮固定以下边界：共享 GNN encoder、joint operation-site pair actor、固定飞机顺序、
R3 的 joint-team PPO、M2 的 BC-KL 退火均不改；不拆 encoder，不恢复 order head，
不使用已经在 held-out 上失败的通用绝对时间 tail policy 权重。

## 2. 数据与审计门槛

先完成新语义下 600 个训练案例的 IGA-1800 标注，并保留 IGA-180 作为教师质量对照。
正式训练前必须通过：

- 600/600 案例完成，cycle、deadlock、非法动作均为 0；
- 每个案例均可从原始输入确定性重放，重放 Cmax 与标注值误差不超过 `1e-6`；
- 轨迹记录同时覆盖保障动作、`ZY-T` 转运选择、`ZY-S/ZY-F` 强制离场动作；
- IGA-1800 相对 IGA-180 的 paired mean、bootstrap 95% CI、分层结果和逐案例差异完整；
- potential V2 只用 train600 拟合，validation 和 blind test 不参与拟合或特征选择。

## 3. Potential V2 与可观测性

在旧的 remaining-work / waiting / future-release 特征之外，加入以下非负“状态成本”：

- 剩余保障工作与作业数、剩余离场工作与作业数；
- 待离场飞机数、等待年龄总和与最大值、已腾挪等待飞机数；
- R014 离场队列、不可用比例、到达待离场飞机的 ETA 总和与最大值；
- 到最近跑道的剩余拖行下界、跑道忙碌数、跑道剩余工作和波次下界；
- 保障机位压力、已完工飞机占用保障机位压力、近期到达压力。

V2 文件带有 potential schema 和环境 semantics 版本；`team_time_potential` 模式必须
严格匹配，防止把无离场或全局 barrier 环境的权重误用于当前环境。旧
`iga_potential` 仍可显式回放旧实验，但不会作为本轮正式配置。

观测使用新的 `f1f2_departure` 模式，在保持 24 维全局输入和旧 checkpoint 形状兼容的
同时，把后 12 维改为 R014、跑道、离场队列和等待年龄摘要。图对象额外携带阶段、
保障进度与 potential 标量，只供 BC 分层和 buffer 记账，不改变 encoder 结构。

## 4. Phase-aware BC / DAgger

BC 仍为 4 epochs，共享 encoder，固定 actor 顺序；teacher execution 基础日程预注册为
`1.00, 0.70, 0.40, 0.10`。改动如下：

- DAgger 从“整个环境全 teacher 或全 student”改为逐飞机采样；冲突通过最小代价匹配
  选择 teacher/student 的可行组合，不能制造两个飞机抢同一机位的伪状态；
- 保障动作、待离场原地等待、待离场腾挪分别统计 NLL/accuracy/执行率；
- 只对存在两个及以上合法 pair 的决策计算 pair loss，`ZY-S/ZY-F` 单选强制动作不计入；
- 默认权重：保障 1、待离场原地等待 1、待离场腾挪 2；保障尾段按每架飞机自身保障
  完成度平滑加权，而不是按 episode 绝对时间把所有尾部动作统一放大；
- 旧 M2 checkpoint 只作为 warm start 对照；新语义 teacher 的 cold start 同时保留，
  用于排除旧策略先验造成的假提升。

## 5. 两阶段筛选矩阵

### 5.1 BC 因果筛选

使用两个预注册 screen seed（seed1/seed2）和相同 teacher shard。GPU0 运行 seed1 的
B0/B1/B2，GPU1 运行 seed2 的 B0/B1/B2；每个 GPU 的三个实验共用一组 validation
worker。晋级依据两个 seed 的聚合 paired validation，而不是挑选较好的单 seed：

| 编号 | 初始化 | 观测 | BC / DAgger |
|---|---|---|---|
| B0 | M2 warm start | f1f2 | 旧整环境 DAgger，控制组 |
| B1 | M2 warm start | f1f2_departure | phase-aware + per-agent DAgger |
| B2 | cold start | f1f2_departure | phase-aware + per-agent DAgger |

BC 晋级首先比较 teacher validation NLL 和 greedy validation Cmax，而不是 train
teacher accuracy。B1 必须在 OOD-stress、待离场腾挪 NLL、严重尾部失败数中至少两项
优于 B0，且 overall Cmax 不退化超过 0.5%。B2 用于判断是否仍需保留 M2 warm start。

### 5.2 PPO 因果筛选

从胜出的同一 BC checkpoint 出发，重置 optimizer；每组先跑 4 PPO epochs。两张 GPU
各静态放置 3 组实验，每张卡内共用一个 validation 服务，不同实验的 CPU 集合不重叠。

| 编号 | 观测 | 回报 / shaping | 目的 |
|---|---|---|---|
| P0 | departure | cmax_delta + V2 potential | 与旧 R3 信用形式衔接 |
| P1 | departure | team_cmax | 终局团队目标基线 |
| P2 | departure | team_time | 精确剩余时间目标 |
| P3 | departure | team_time + V2 beta ramp | 主假设 |
| P4 | legacy f1f2 | team_time + V2 beta ramp | 离场观测消融 |
| P5 | departure | team_time + fixed beta | beta 退火消融 |

P3 的 beta 日程固定为 `0.10,0.10,0.05,0.00`；P5 四轮均固定 `0.10`。四轮 screen
是有效消融的最低长度；若只跑两轮，P3 与 P5 完全等价，预检脚本会直接拒绝。所有 team-time
实验使用 `gamma=1`、terminal Cmax coefficient=1，并检查时间回报和 potential 势差的
telescoping 误差。PPO 的 BC-KL 日程保持 `0.40,0.25,0.10,0.05`，之后为 0。

## 6. Screen 晋级和 Formal 规则

PPO screen 必须同时满足：

- 相对自身 Pre-PPO validation composite Cmax 改善至少 1%；
- OOD-stress 不退化，100% 完成，cycle/deadlock 为 0；
- optimizer step completion 不低于 90%，无 zero-update shard；
- old-policy KL 不超过预注册 target KL；
- paired bootstrap 支持整体改善；
- `ZY-T` 决策覆盖、R014 等待、跑道队列和尾段 90%-Cmax 摆动没有异常恶化。

若没有任何组通过则停止 Formal。否则仅按冻结的 validation composite 选一组，运行
seeds 1/2/3、每个 8 PPO epochs；每个 seed 完成后只评估一次 test60。Formal 期间不得
根据 test60 改超参数或选择 checkpoint。

## 7. 最终报告

最终同时报告三种子 mean/std、paired bootstrap 95% CI、IID/OOD-stress/OOD-scale、
逐案例相对 IGA-180/IGA-1800 和旧 M2 的差值、gap>800 严重失败数、90% milestone 后
摆动、R014 利用率/空驶/排队、跑道利用率、完工后腾位比例、BC 分阶段 NLL、PPO KL、
参数更新幅度、step completion，以及所有 early-stop/canary/invariant 异常。

“接近 IGA”和“达到 IGA”仍采用既有三种子 test60 定义；单 seed 或 screen validation
结果不作为目标完成证据。

## 8. 执行入口与 fail-closed 预检

`onpolicy/scripts/train/prepare_stage1_departure_research.py` 是本轮唯一配置生成入口。
它在生成命令前强制检查 600 个 teacher 的新环境语义、potential V2 完整特征顺序、
600/600 重放一致和 `max_abs_cmax_drift <= 1e-6`。三种阶段分别为：

```bash
# B0--B2；输出 3 条 BC-only 命令，不执行 PPO
python3 onpolicy/scripts/train/prepare_stage1_departure_research.py \
  --phase bc_screen ... --output bc_screen_commands.json

# P0--P5；输出 6 条四 epoch screen 命令
python3 onpolicy/scripts/train/prepare_stage1_departure_research.py \
  --phase ppo_screen ... --output ppo_screen_commands.json

# validation 冻结选出 winner 后，输出 seeds 1/2/3 的 Formal 命令
python3 onpolicy/scripts/train/prepare_stage1_departure_research.py \
  --phase formal --formal-variant P3_team_time_potential_ramp ... \
  --output formal_commands.json
```

省略号位置必须显式提供 source command、train600、IGA-1800 teacher、V2 potential、
warm-start/BC checkpoint 和 run tag。生成的 manifest 同时声明每卡 3 trainer、每卡一个
共享 evaluator、trainer 间 CPU 静态隔离的硬件契约；实际调度继续复用现有 systemd
shared-evaluator launcher，不在预检脚本内隐式启动进程。

## 9. 当前准备状态（2026-08-12）

- 新语义 IGA-180 train600：600/600 完成；
- 新语义 IGA-1800 train600：600/600 完成；
- Potential V2：600/600 精确重放完成，共 210,146 transitions，最大 Cmax 漂移为 0；
  train 拟合 R²=0.9269，5-fold case-disjoint CV R²=0.9219；
- Potential 文件：`result/hkbz_train_logs/stage1_departure_research_20260812_r1/iga_potential_v2.json`；
- B0/B1/B2 已按 seed1/seed2 启动六组 BC screen。每张 GPU 三个训练服务，各训练独占
  12 个物理核及其两个 SMT 线程；每张 GPU 共用一个 60-worker validation 服务，六组
  训练错开 30/120/150/240/270 秒启动。

## 10. 2026-08-13 执行修订（覆盖 5.2 的 PPO 矩阵）

BC screen 的两种子聚合结果已经形成：B0 legacy `f1f2` 的 validation composite
Cmax 为 9504.54，优于 B1 的 10039.62 和 B2 的 10075.70。因而 departure-aware
观测和 phase-aware BC 没有达到晋级门槛。继续让 PPO 配置混用 B0 checkpoint 与
`f1f2_departure` 会把 observation schema、BC 初始化和 reward 变化混在同一个对照中，
无法解释结果。本轮 PPO 因果筛选据此修订为：所有组都从同一份 B0 seed1 PlaneBC
checkpoint 出发，统一使用 `f1f2`、固定 tau=0.30、每轮完整覆盖 960 个采样案例，并
重置 optimizer 和 ValueNorm。

修订后的六组配置为：

| 编号 | PPO 回报 | potential | terminal Cmax | 目的 |
|---|---|---:|---:|---|
| P0 | action `cmax_delta` | 0 | 0 | 动作级精确 Cmax 基线 |
| P1 | action `iga_potential` | 0.10 | 0 | 只测 V2 potential 增量 |
| P2 | `team_cmax` | 0 | 1 | 团队级终局 Cmax 基线 |
| P3 | `team_time` | 0 | 1 | 精确团队剩余时间基线 |
| P4 | `team_time` | 0.10,0.10,0.05,0 | 1 | potential 退火主假设 |
| P5 | `team_time` | 0.10,0.10,0.10,0.10 | 1 | 固定 potential 消融 |

六组均固定 `gamma=1`、`iga_potential_gamma=1`、`reward_coef=0.01` 和
`hindsight_cmax_coef=1`，因此都保持等价的缩放 `-Cmax` 最终目标。checkpoint 现在带有
显式 observation schema 和环境语义；训练与共享 evaluator 都会 fail closed。任何
schema 不匹配都会在 rollout 前退出，恢复 BC checkpoint 时必须丢弃旧 ValueNorm。

每组跑 seed1/seed2，共 12 个训练，按两个 wave 执行：Wave 1 为 P0/P1/P2，Wave 2
为 P3/P4/P5。每个 wave 每张 GPU 放置三个 trainer，每个 trainer 独占 12 个物理核及
其两个 SMT sibling（24 个逻辑 CPU）；每张 GPU 只有一个 60-worker evaluator，并在
处理某个请求时临时绑定到该 trainer 的 CPU slice。两个 wave 不同时运行，以维持每卡
三个训练进程的显存与 CPU 预算。

统一启动入口为：

```bash
WAVE=1 RUN_TAG=stage1_departure_reward_screen_20260813_r1 \
  bash onpolicy/scripts/train/launch_stage1_departure_ppo_screen.sh

# Wave 1 完成并停止其共享 evaluator 后执行
WAVE=2 RUN_TAG=stage1_departure_reward_screen_20260813_r1 \
  bash onpolicy/scripts/train/launch_stage1_departure_ppo_screen.sh
```
