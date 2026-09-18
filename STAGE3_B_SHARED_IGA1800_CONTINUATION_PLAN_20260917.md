## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-17, Asia/Shanghai
- Verification Status: UNVERIFIED（研究方案尚未执行；历史事实已从本机冻结产物核对）
- Execution Status: PLANNED
- Version Label: stage3_h3_iga1800_continuation_plan_v1
- Workspace baseline: `main`, `2feea87e4edd809d1553751d6028f935de04e7cc`
- Scope: 形成下一轮方案、输入清单和预算；本次不修改训练实现、不启动训练或 IGA 求解。

# Stage3 下一轮：延续 H3 PPO 收益，逐步超过冻结 IGA1800

## 1. 研究判断与约束

推荐顺序：**统一 canonical 评测与续训起点 → 配对比较训练 τ=0.03/0.1 → 按固定门槛延长有效分支 → 若停滞则比较状态价值基线 → 冻结配方后做多 seed 与独立验收。**

本轮有改善，但“继续训练必然超过 IGA1800”尚无证据。最终目标是单个 RL checkpoint、单次 canonical H 解码超过冻结 H3/F4/soft IGA1800；平均 makespan、尾部风险和泛化证据同时报告。收益曲线不作线性外推。

保持用户已明确的边界：

- B_SHARED，H3/F4/soft，沿用相同环境和动作约束。
- IGA 完全冻结，新增求解/教师查询为 0；不得调用求解器补缺失数据。
- 首轮仍用相同 Train240、每 epoch 384 次访问、240 环境槽、64 个 CPU 环境进程、一个 GPU 模型。
- GPU0、训练期间最多原定半机 CPU；顺序运行两个训练分支。完成训练后可按此前授权给验证器使用全部可用 CPU，保持评测批宽与数值规则。
- 不恢复已由用户豁免的 microbatch 32/64 完整数值对照。保留固定配置下的概率回放、恢复、数值与资源检查。
- Tune/Validation 案例不进入训练或 imitation loss。Tune 已用于诊断，今后不能重新包装成未见过的测试集。

## 2. 已核实的起点

证据：[本轮最终分析](result/stage3_analysis/h3_r0_final_20260917/report.md)、[机器可读分析](result/stage3_analysis/h3_r0_final_20260917/analysis.json)、[固定 H 全轨迹矩阵](result/hkbz_eval_logs/stage3_h3_canonical_s0_e6_validation120_20260916_r1_gpu0/summary.json)。

| 项目 | 本轮结果 | 对下一轮的含义 |
|---|---:|---|
| 完成预算 | 8 epochs / 3,072 次访问 / 96 次 PPO 更新 | 有完整、可恢复的父 checkpoint |
| legacy H Validation120，E8 vs S0 | 8,270.96 vs 8,368.25 秒，改善 1.163% | 有继续研究价值 |
| legacy H Tune60，E8 vs S0 | 改善 0.572%；95% 区间跨 0 | 单 seed 的迁移收益仍不确定 |
| E8 vs 冻结 IGA1800，Tune60 | 8,307.98 vs 8,182.77 秒，落后 1.530% | 仍需缩短平均 125.21 秒 |
| 同上配对 gap 95% 区间 | [+0.481%, +2.601%] | 本轮尚未追平 |
| Tune >5% 退化 / >10% 退化 | 16/60、4/60 | 均值之外必须限制大幅失败 |
| Tune 自身最差 10% 均值 / IGA 同指标 | 1.04327 | 尾部还有 4.33% 差距 |
| Tune OOD stress vs S0 / vs IGA | 改善 0.082% / 落后 2.296% | 目前压力场景几乎没有受益 |
| OOD stress 对剩余净差距的贡献 | 73.01% | 优先诊断资源竞争和耦合决策 |
| canonical S0 / E1 / E6 | 8,367.36 / 8,302.00 / 8,289.52 秒 | 同规则下 E6 vs S0 改善 0.930% |
| canonical τ 矩阵 | 1,080 条完整轨迹，三温度逐模型动作/历史/步数/makespan 一致 | 评测 τ 不再作为搜索变量 |
| E8 canonical | 尚未完成 | 下一轮的首项评测任务 |

E8 相对 E6 的 legacy Validation 改善仅 0.348%，配对区间跨 0；E8 的尾部还比 E6 差。因此不把 E8 的微小均值优势直接当作持续上升趋势。

Train240 的唯一案例分布：IID 192、OOD stress 43、OOD scale 5。访问权重为 IID 每例 1 次、OOD 每例 4 次，故每 epoch 是 384 条完整轨迹，OOD 已占一半访问量。

| Train profile | 唯一案例数 |
|---|---:|
| balanced / bursty / coupled | 48 / 39 / 19 |
| high_flex / light / resource_sparse | 29 / 19 / 38 |
| resource_ood / stress_arrival / stress_joint | 10 / 14 / 19 |
| low_load_ood | 5 |

不能因 Tune 压力场景差，就未经诊断把 OOD 重复次数从 4 倍继续提高；重复访问并不增加独立案例覆盖。

## 3. 目标、指标与证据等级

主指标：`gap = mean(C_RL) / mean(C_reference) - 1`，越小越好。先按同一案例配对，再算均值之比；不以案例百分比的平均值替代主指标。

冻结 Tune60 IGA1800 均值 8,182.771111 秒。下面是可量化的研发里程碑，不是已完成的结果；新 RL 数值统一使用 canonical H。

| 里程碑 | Tune60 gap | 对应 RL 均值上限 | 相比当前 legacy E8 需再降 | 风险目标 |
|---|---:|---:|---:|---|
| M1：缩小差距 | ≤ +1.0% | 8,264.60 秒 | 43.38 秒 | >5% 退化 ≤12/60；尾部比 ≤1.03 |
| M2：接近 | ≤ +0.5% | 8,223.68 秒 | 84.29 秒 | >5% 退化 ≤9/60；尾部比 ≤1.02 |
| M3：均值越过 | < 0 | <8,182.77 秒 | >125.21 秒 | 不单凭均值宣布成功 |
| M4：有实际幅度的超越 | ≤ −0.5% | 8,141.86 秒 | 166.12 秒 | >5% 退化 ≤6/60；尾部比 ≤1.01；OOD stress gap ≤1% |

M4 还要求完成率 100%、死锁/循环 0，并报告 >10% 退化的所有案例。0.5% 是预定实用幅度，不是根据本轮方差证明可检出的最小差异。

“确认超越”的更高证据要求：预注册主 seed 在独立、未参与选择的案例上 gap ≤−0.5%，配对 95% 区间上界 <0，风险门槛通过；另外两个预先确定的续训 seed 均值 gap <0，逐 seed 报告风险与不确定性。

**当前匹配的冻结 IGA1800 只覆盖 Tune60：Train0 / Validation0 / Confirmation0。** 因而近期能确认的是“在这组冻结 60 案例基准上的结果”。正式独立 IGA 验收只能在找到已有、同合同且未用于开发的冻结参考后进行；不允许为此重新求解。若参考继续缺失，仍可研究并报告 Tune 基准胜负，以及 Confirmation 上相对 RL 父模型的泛化，但不能宣称已独立证实普遍超过 IGA1800。

## 4. P0：冻结新起点与统一评测（训练前约 2–4 小时）

### 4.1 父模型

默认父模型为本轮 E8：

```
result/hkbz_train_logs/stage3_b_shared_h3_r0_frozen_iga_20260915_r1_gpu0/
  attempts/20260915T012204_2081156/train/models/epoch_0008.pt
SHA256: 14d63bb44ee96eea38c993bb7d1d1d1ae5f0aa59f979b14a9337e6f96215a4e6
```

预登记回退 E6，SHA256 `bde89b8a27211956005fded1ce60f034ce3d2c56e7373e8deb5f1a80919c0ff9`。两者均已在 CPU 上检查：完整逻辑批提交、保存 actor/critic Adam 状态、ValueNorm、RNG；E8 累计更新 96，E6 为 72。

先做 E8 canonical Validation120。默认继续 E8；若它不满足基础风险条件，或 canonical 均值比 E6 差 >0.2%，则采用通过条件的 E6。基础风险条件：相对 canonical S0 均值不退化、各 distribution 不退化 >1%、自身最差 10% 均值比 ≤1.01、>5% 案例退化比例 ≤10%、完成率 100%、循环 0。若 E6/E8 均不合格，完成原因分析后再修订起点，不静默改用 Tune 最佳模型。数值复现失败属于代码问题，必须先修复，不能用换父模型掩盖。

### 4.2 复现与缓存

1. 当前 `main` 刚完成集成。锁定新训练 source snapshot，并单独绑定评测 source；评测优先复用已通过全轨迹验证的冻结 canonical 执行代码。新入口在固定 12 槽布局重放 E6 的 12 条 legacy 轨迹和 12 条 canonical 轨迹，对照已有完整记录；检查动作/历史 hash、makespan、步数及权重不变。
2. 新增完整评测：E8 Validation120、E8 Tune60、S0 Tune60，均 canonical、τ=0.3，共 240 条。若回退 E6，增加 E6 Tune60，共 300 条。上一步另有 24 条 admission 轨迹。
3. S0/E1/E6 已有 canonical Validation120 全轨迹证据，只有评测计算相关代码、依赖与输入绑定相同才复用。12 条复现仅为入口检查，不能证明变更过的计算代码在所有案例上等价。若必须改动评测执行代码，则在该版本重新评测 S0/E6 的单温度 Validation120（额外 240 条，约 1.2–2.0 小时）；不重复三温度 × 三模型大矩阵。
4. Tau 不变性用已有全轨迹证据加当前实现的针对性测试维护；不以新的 Tune 分数挑评测温度。
5. 评测继续固定 12 槽、完成槽不压缩、相同 case 顺序、seed1、同精度/TF32/线程设置。保留最大覆盖→精确 raw-score 总和→物理 ID 并列规则。

所有后续改善都相对本次确定的 canonical 父模型 P 计算，legacy 数值只保留为历史参照。

## 5. P1：训练温度配对实验（本次首选）

### 5.1 因果问题

假设 H1：在相同权重和计算预算下，τ=0.1 能增加有效候选动作的探索，使固定 H 部署结果改善，尤其是 OOD stress。备择解释包括：新轨迹更差、仅增加随机性、梯度尺度变化而非探索本身、AR 采样与 H 解码收益不一致。

| 分支 | 训练 τ | 起点 | 第一阶段预算 | 有效时最大预算 | 唯一主变量 |
|---|---:|---|---:|---:|---|
| C03：继续训练对照 | 0.03 固定 | P 的完整训练状态 | 2 epochs / 768 访问 / 24 更新 | 8 epochs / 3,072 访问 / 96 更新 | 无 |
| T10：增加探索 | 0.10 固定 | 完全相同 P 状态 | 2 epochs / 768 访问 / 24 更新 | 8 epochs / 3,072 访问 / 96 更新 | 训练温度 |

两臂共用预登记新访问日程和 seed `2026091702`、优化器 shuffle seed `2026091706`；相同 case/replica 采用相同随机种子。不同策略的轨迹分化是预期现象，不要求采样动作相同。

按阶段排队：先 C03 到 E2，再 T10 到 E2，完成配对判断后才分配 E4/E8 预算；不先把一臂跑满再补另一臂的筛选结果。

这是**保留 Adam、ValueNorm 的全状态续训分支**，不是 weights-only 重启。保存旧 RNG 快照与所有谱系，再明确应用两臂共同的新采样日程；不声称与旧进程假想 E9 的随机序列逐位相同。累计更新数继续从 96（或回退时 72）记录，新 run 的 epoch 从 1 计数。

### 5.2 其余配置全部固定

| 配置 | 值 |
|---|---|
| 唯一训练案例 / 每 epoch 访问 | 240 / 384，保持原 IID:OOD 权重 |
| 环境槽 / CPU 环境进程 | 240 / 64，环境进程不加载 GPU 模型 |
| optimizer minibatch / physical microbatch | 64 / 64 条完整轨迹 |
| PPO passes / 更新数 | 2 / 每 epoch 12 次 |
| actor lr / role scales | 1e−5 / [1, 0.25, 1, 0.5] |
| critic lr | 1e−4 |
| PPO clip / 梯度裁剪 | 0.2 / 1.0 |
| soft KL / hard KL | 0.02 / 0.04 |
| TBPTT | 8 |
| 训练优势与 source_costs | 保留本轮终局 source-relative 优势及冻结 H3 B0 成本表 |
| entropy bonus | 保持当前实现，不同步引入 |
| 评测 | canonical H / H3 F4 soft / τ=0.3 / 固定 12 槽 |

提高训练温度也会改变 log-prob 梯度尺度，因此该实验首先检验“温度这个训练设置的整体效果”；仅凭胜负不能把机制全部归因于探索。

### 5.3 必要实现，不扩大算法变化

现有代码在 recipe 校验、轨迹检查、PPO replay、恢复路径中写死 τ=0.03，不能只改采样参数。必须一次性打通 `train_tau` 的配置、collection、旧 log-prob、PPO replay、checkpoint 与恢复。

- `behavior_tau`、behavior model hash、mask/history/decoder 标识随轨迹保存。
- 同一完整 collection 及其两轮 PPO 保持同一 τ；未来若退火，只能在新 collection 前变更。
- 新 protocol 显式从旧 v1 checkpoint fork：验证父 SHA、旧 manifest、规划合同、完整批边界、所有权重/优化器/normalizer；只允许预登记字段变化，不绕过旧 manifest 校验。
- 恢复后 temperature、Adam step/moments、ValueNorm、累计更新、访问游标一致；两臂独立 run 目录、不可覆写父 run。
- 初始同策略 log-prob 回放达到现有 ≤0.002 容差；本轮实测约 4.32e−5，同时报告新分支误差，不只给通过标记。
- 概率比始终使用实际行为策略的 likelihood。PPO 原论文的分母为采样策略概率，不能在采样后换 τ 重新构造分母：[Schulman et al., PPO](https://arxiv.org/abs/1707.06347)。

### 5.4 诊断记录

复用采样审计中“单候选与多候选分开”的统计思路，做不消耗额外随机数的原生埋点；旧 audit 的 monkeypatch 不直接接入正式训练。

- 按 role、distribution 记录候选数、仅多候选决策的 entropy、`entropy/log(n_legal)`、pmax、非最大分数动作比例、no-op 比例。
- 记录真实决策数、actor 梯度范数/裁剪系数、逐参数组更新量、各 role KL/ratio clip fraction。96/96 次梯度裁剪不是单独调大学习率的依据。
- 同一 epoch 报告 AR 行为回报和 canonical H 部署指标；若采样改善但 H 无改善，按部署目标判定。
- 保存预先固定 Train24 的完整轨迹，记录就绪但未服务时间、资源不可用时间、移动/调位时间、预约失效和决策等待。只有能由事件日志可靠区分的量才报告，避免把所有 idle 时间解释为调度错误。
- Train24 按已有 Train profile 和内容 hash 选取，见配套清单；不以 Tune case ID 或输给 IGA 的结果选择训练案例。

### 5.5 检查点、晋级和停止

完整保存每个 epoch；只在新增 epoch 1、2、4、6、8 进行 Validation120。两臂相同 checkpoint 的比较必须拥有相同访问量和更新预算。

1. **E2：排除明显有害分支。** 两次连续评测均比 P 差 >1%，或连续两次风险不合格，则在完整批边界结束该分支。T10 相对 C03 改善 ≥0.2% 记为探索性正信号；差异 <0.2% 为未决，可继续至 E4，不宣称无效或有效。
2. **E4：决定是否继续消耗长预算。** 至少一臂相对 P 改善 ≥0.3%、OOD stress 均值不退化，且通过下列风险门槛，才扩展到 E8。若两臂均未达到，停止当前配方延长，进入 P2 诊断。
3. 若 T10 在 E4 优于 C03 ≥0.2% 且通过风险门槛，保留两臂至相同 E8 预算以检验差异能否持续。若 T10 优势不足，优先保留通过延长门槛的 C03；不为“温度必须有效”额外扩展网格。
4. 每臂入选门槛（相对 P）：完成率 100%、循环 0；均值退化 ≤0.2%；各 distribution 退化 ≤1%；自身最差 10% 均值比 ≤1.01；>5% 案例退化比例 ≤10%。最终阶段目标更严格：Validation 平均改善 ≥0.5%，OOD stress 不退化，尾部不退化。
5. 最终只从通过门槛的预定 Validation checkpoints 和 P 中选一个；均值最低优先，差距 ≤0.1% 时选尾部更好者，再选更早 checkpoint，仍相同时选 C03。选定 hash 后才读本轮最终 Tune 成绩。额外给相同 E8 的对照做一次 Tune 可用于解释温度差异，但不据此换冠军。
6. 单臂提前停止时，后续只有存活臂的收益；不把“赢家 E8 对停掉的对照 E2”写成同预算温度优势。
7. NaN、概率/mask 不匹配、硬 KL、OOM、环境未完成均留下失败证据；当前不完整 collection 不成为可恢复提交。修复后开新 attempt，恢复最后完整提交。

上述 0.2/0.3/0.5% 是算力分配和实用幅度门槛，不是显著性阈值。所有配对区间仍完整报告。

## 6. P2：温度或延长训练停滞后，验证状态价值基线

这一步是条件研究分支，不与 P1 同时改动。

当前 actor 使用 `A = 0.01 × (B0_case_cost − trajectory_makespan)`，整条轨迹及所有角色共享这个终局标量。它是固定参考基线，并非因“全轨迹同一值”就数学错误；但不能利用状态价值减少剩余回报的不确定性。当前 critic 学习剩余时间，却没有参与 actor 的优势估计。

假设 H2：在相同采样温度和相同预算下，用采样前的状态价值作基线，能改善有效更新与压力场景质量。

从同一个已冻结 P*（按 Validation 选出的当前 checkpoint）分出两个分支：一臂保留原优势，一臂只替换为

```
G_t = -0.01 × (C_max - current_time_t)
A_t = G_t - V_old(s_t)
```

`V_old` 在 collection 开始前冻结，按当时 ValueNorm 正确反归一化；优势 detach，在同一 collection 的所有 PPO passes 中不重算。必须确认 current_time 与最终 makespan 同一时间原点。

- 第一版用完整回报减基线，相当于完整终止轨迹上 γ=1、λ=1 的形式；保持 makespan 目标，不先引入按决策步折扣。零时长决策和 padding 不能制造虚假的时间奖励。
- 先用 Train 轨迹检查 critic 校准、反归一化、explained variance；若旧 critic 比常数预测还差，先解决价值估计，不直接宣布新优势可用。
- 首批用于 actor 的 V 必须来自看到该批回报之前的 critic；不能先对本批过拟合再称其为旧基线。
- 角色间共享 team advantage 仍不等于反事实个体信用分配。这里检验的是状态基线，不承诺解决全部多智能体归因问题。
- actor lr、loss 聚合、梯度裁剪、data weighting、entropy bonus 均不同时改变。尤其不同时除以每条轨迹动作数，否则会连带改变长短轨迹权重。
- 两臂先各 2 epochs（共 1,536 访问 / 48 更新）；用与 P1 一致的评测与预算门槛决定是否扩展。若要引入 λ<1 的 GAE，再单独消融偏差/方差及不等时事件步的处理。

使用状态价值降低策略梯度估计方差的依据见 [Schulman et al., GAE](https://arxiv.org/abs/1506.02438)；在本工作区能否改善部署 makespan 是待检验假设。

## 7. P3：仍有系统性差距时的研究顺序

每次只放行一个新主变量，先做 2-epoch 配对筛选，不叠加无边界参数网格。

1. **探索后收敛**：只有固定 τ=0.1 的探索有效、后期出现随机性代价时，比较固定 0.1 与预先锁定的退火方案（例如 4 epochs 0.1、2 epochs 0.06、2 epochs 0.03），同父模型、同 8-epoch 预算。若 0.1 明显过热但熵证据支持中间温度，可改测唯一候选 0.06；两个方向不同时展开。
2. **压力分布与覆盖**：重点分析 Train 中 resource_ood 10 例、stress_joint 19 例、coupled 19 例。若反复记住这批案例却不能改善 Validation，则评估扩大“已有训练池”覆盖的独立分支，并继续隔离 Tune/Validation/Confirmation；不靠重复困难 Tune 案例改善分数。首轮 P1 的 Train240 不变。
3. **AR→H 部署差异**：若 AR 的回报/多样性改善而 H 无改善，检查相同 Train 状态下候选覆盖、raw score 排序、H 匹配和早期分歧之后的瓶颈。冻结 IGA 的 Tune 轨迹只作离线诊断，不进入损失、训练采样或冠军选择。缺少 Train H3 教师时不启动教师 BC。
4. **表示/优化受限**：若状态基线仍无效，检查共享参数梯度冲突、角色更新规模与瓶颈可观测性，再决定是否需要额外状态特征或独立的损失尺度实验；不因“KL 小且经常裁剪”直接把 lr 放大十倍。

主指标始终是单 checkpoint 的 canonical H。若以后研究多次采样取优或 RL+搜索，另列方法名、轨迹次数、延迟和预算，不把 best-of-N 分数混为单次 RL 超越。

## 8. 多 seed、独立验收和数据纪律

- Validation120 用于 checkpoint/配方选择；预定评测时点以外不反复读分数挑幸运窗口。
- Tune60 用于已暴露基准的阶段终点报告。P0 补齐历史 canonical 基准不用于改变训练参数；P1 先冻结冠军 hash，再做本阶段 Tune。
- 保留全部分支和失败结果的 exposure ledger；不得删去变差案例后重算均值。
- 配对 bootstrap 按既有 profile 分层，20,000 draws、固定 seed `2026091505`；公布 case 表、均值秒数、gap、95% 区间、W/T/L、各 distribution、最差 10% 均值、>5%/>10% 退化计数。
- 风险的“最差 10%”是各策略自身的最差 ceil(0.1n) 个案例；另列相同案例的最大退化，避免混淆。
- 单 checkpoint 的 bootstrap 只表示固定训练 seed 条件下的案例不确定性；多个 checkpoints 的区间未经多重比较修正，研发阶段不用于宣称新方法被独立确认。
- 选定配方及目标 epoch 后，以 `2026091712`、`2026091722` 两个额外 continuation seeds 重做，加首个 seed 共 3 个；对应 shuffle seeds `2026091716`、`2026091726`。不从 3 个里面只报最好的，不逐案例择优。
- 三个续训 seed 共用固定父 checkpoint，只证明在这个父模型上的续训稳健性，不等于三次独立 Stage2→Stage3 全流程训练。若最终配方包含 P2/P3，必须复现完整已锁定训练链，预算按链长重算。
- Confirmation 只在配方、epoch、seed 选择规则全部锁定后开启。只有已有、匹配环境/案例/预算的冻结 IGA 覆盖才允许做独立 IGA 验收；无覆盖则报告未具备该证据。
- Tune60 当前 gap 区间半宽约 1.06 个百分点。即使点估计达到 −0.5%，也可能仍不能排除 0；反复训练/重复这 60 例不能增加独立案例数。

## 9. 资源、吞吐与时间预算

硬件配置沿用：GPU0（UUID `GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`），CPU `0-31,64-95`；trainer `0-21,64-85`、validator `22-29,86-93`、controller `30-31,94-95`。上述分工共享半机集合，不为两臂各额外申请半机。

固定 microbatch64、激活 checkpoint、输入缓存 4096 MiB、GPU headroom 6144 MiB、MemoryHigh 102 GiB、MemoryMax 114 GiB、swap0。实际 launch 前重新检查可用资源与 cgroup，不把旧资源读数当作当前余量。

默认训练和 GPU 评测顺序调度，避免同卡互抢。增加环境数、minibatch 或同时放两个训练作业均不属于这一轮温度实验。若后续单独优化吞吐，比较相同访问/更新/完整评测工作量的完成时间、有效轨迹数、完整提交数与峰值资源；GPU utilization 不是训练效率的替代指标。

预算依据：本轮 E2–E8（排除 E1 的额外 microbatch 测试）每 epoch 4.03–5.45 小时，中位 4.59 小时。canonical 新增 720 条轨迹用时约 4.12 小时，折合 120 条约 0.69 小时；计划按每次 Validation120 0.6–1.0 小时、Tune60 0.3–0.6 小时估算。

| 阶段 | 新训练访问 / PPO 更新 | 完整评测工作量 | 预计单 GPU0 墙钟时间 |
|---|---:|---|---:|
| P0 准备/补齐 | 0 / 0 | 240（回退时 300）+24 条 admission | 2–4 小时 |
| P1 两臂各 2 epochs | 1,536 / 48 | 4 × Validation120 | 训练 16.1–21.8 小时；含 P0/验证约 21–30 小时 |
| P1 两臂都扩展至 8 epochs | 总计 6,144 / 192 | 总计 10 × Validation120；末期最多 2 × Tune60 | 含 P0/验证总计约 73–103 小时 |
| P2 状态基线 2×2 epoch 筛选（条件项） | 1,536 / 48 | 4 × Validation120，另计针对性 admission | 约 19–28 小时；正式启动前更新估计 |
| 锁定 8-epoch 配方后两个额外 seeds（条件项） | 6,144 / 192 | 每 seed 在 E4/E8 验证、最终 Tune | 约 68–94 小时；更长配方重新预算 |

P1 全预算已经包含前 2 epochs，不可再次相加。上述估计不含代码实现、失败重跑、未准备的独立 Confirmation；高温可能延长轨迹，首个可比 epoch 后用实际动作数和用时刷新 ETA。

建议首个筛选阶段墙钟上限 36 小时、P1 全阶段 120 小时。接近上限时不启动下一完整 collection；硬超时不发布未提交状态。达到时间上限或收益门槛失败后输出已有结果并决定下一分支，不能为了“直到超过”无限续跑。

监控同时检查进程/service 存活、heartbeat、当前 collection/pass/minibatch、完整 checkpoint 和评测产物；不只依赖 `state: running`。记录采样、PPO replay/backward、评测分别耗时、GPU allocated/reserved、进程 RSS 和 cgroup 峰值。

## 10. 实施工作清单与可交付物

以下为下次获准实施时的清单。本文件不声称启动命令已可用：现有 v1 runner 不能直接执行 τ=0.1 或跨 manifest 全状态 fork。

| 工作项 | 主要代码位置 | 验收 |
|---|---|---|
| 新协议/双臂清单/预算 | `onpolicy/utils/stage3_h3_frozen.py`，新增独立 continuation recipe/driver | v1 历史输入保持可读；两臂仅预登记差异；无 IGA 调用 |
| train_tau 一致性和全状态 fork | `onpolicy/runner/shared/stage3_h3_frozen_engine.py` 及实际 collection 路径 | 不再静默回退 0.03；模型/Adam/ValueNorm/游标身份可审计 |
| 决策与优化统计 | 正式采样路径；参考 `onpolicy/utils/stage3_sampling_audit.py` | 埋点不改变 RNG/动作；单候选单列；逐 role/分布输出 |
| canonical 评测与缓存 | `run_stage3_h3_canonical_tau.py` / 现有矩阵 driver | 支持 E8 和新 checkpoint；固定12合同；输入/代码/hash 绑定 |
| 晋级/停止/报告 | 新 continuation driver + 分析脚本 | 同预算比较；冻结冠军后读 Tune；失败/未通过分支保留 |
| 条件项状态价值优势 | H3 engine 独立 protocol 分支 | 时间原点/终止/padding/反归一化/旧 V 固定有针对性测试 |

必要检查聚焦本次语义：temperature 配置一致性、相同策略 likelihood 回放、epoch 边界恢复、埋点不改 RNG、canonical legacy admission、分支预算与 split 隔离。无需为低影响文档或重复实现写无意义测试。

每个正式 run 输出：`manifest.json`、`parent_checkpoint.json`、`schedule.json`、`metrics.jsonl`、`decision_stats.jsonl`、`resource_usage.jsonl`、每 epoch 完整 checkpoint、canonical 逐案例表、Train24 轨迹、`selection.json`、`exposure.json` 和 `final_result.json`。最终结果明确区分：实现可用、开发收益、冻结 Tune 基准胜负、独立确认四种状态。

配套规划产物：

- [证据与预算 JSON](result/stage3_analysis/stage3_next_plan_20260917/plan_evidence.json)
- [预算表](result/stage3_analysis/stage3_next_plan_20260917/budget.csv)
- [固定 Train240 输入清单](result/stage3_analysis/stage3_next_plan_20260917/train240.csv)
- [固定 Train24 诊断清单](result/stage3_analysis/stage3_next_plan_20260917/diagnostic_train24.json)

这些清单只复制已有案例的身份，不复制或生成训练数据，不构成新教师标签。
