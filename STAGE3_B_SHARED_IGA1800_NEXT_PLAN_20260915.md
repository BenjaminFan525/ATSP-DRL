# 下一轮 Stage3：全共享 RL 接近 H3/F4/soft IGA1800

## 计划身份与结论

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-15
- Version Label: stage3_b_shared_h3_iga1800_plan_v1
- Verification Status: **计划尚未实施；历史结果已核对，输入清单已生成。**
- 本次产物是研究计划、证据汇总和先导案例清单，没有启动训练、IGA 求解或环境评测。

**推荐主线：保留完全共享 B_SHARED，以本轮 Best6 为权重起点，统一到 H3/F4/soft，先验证联合 IGA 示范能否转化成闭环调度收益，再运行“IGA 示范引导 + PPO”，同时保留预算匹配的纯 PPO 对照。**

这轮要回答两个可区分的问题：①适当增加实际优化器更新，现有 PPO 能否继续改善；②在同样 RL 访问量和更新预算下，联合教师能否帮助策略跨过当前瓶颈。接近 IGA1800 是待验证目标，不是已有结论。保留共享结构，不同时重做网络、奖励、信用分配和解码器。

## 1. 本轮结果对下一轮的约束

主证据：[修订后的 H3 IGA 逐案例分析](result/stage3_analysis/stage3_h3_iga_vs_rl_h2_20260914/analysis.md)、[本轮训练最终审计](result/stage3_analysis/stage3_b_shared_b0_r8_final_20260914/analysis.md)。可计算输入见 [evidence.json](result/stage3_analysis/stage3_next_plan_20260915/evidence.json)。

| 已核对的观察 | 对研究设计的影响 |
| --- | --- |
| Tune60：H3 IGA180 为 8570.82 秒，H3 IGA1800 为 8182.77 秒；H2 Best6 为 8339.31 秒 | 当前 RL 快于 IGA180 2.70%，慢于 IGA1800 1.91%；必须先补同 H3 的 RL 基线 |
| Best6 对 IGA1800 为 24 胜、36 负，19/60 慢超过 5%；仅缩小 B0→IGA 差距的 8.00% | 教师不是逐例最优；保护 RL 胜例，并考察严重退步比例 |
| Best6 对 B0：Validation120 改善 0.760%，Tune60 改善 0.163%；Last8 退回 0.200% / 0.066% | 用 Best6 初始化；不因为 Last 更新而更换起点；单 seed 和单 split 不能判定成功 |
| RL 总资源等待少 11.77%，但离场接驳累计等待约为 IGA 的 5.01 倍 | 优先诊断关键请求、联合作业与离场；不直接改成最小化总等待的奖励 |
| case_0051、0007 有关键 ZY10 长等待；0008、0044 有联合作业安排差距 | 教师必须覆盖飞机作业/站点、普通资源和转运车，不能只训练资源头 |
| Train240 新增 3072 次访问、32 次实际 PPO 更新；累计为 3664 / 54，原方案是 7680 / 480 | 本轮没有完成原始学习预算；不能据此判定完全共享网络失效 |
| 新阶段最大 global KL 约 0.000185，最大 clip fraction 约 0.197% | 检查实际策略变化和有效更新；增加更新次数是实验假设，不是成功保证 |
| 16 批合计 26.83 小时，采样 5.30 小时，update 阶段 21.52 小时，占 80.2% | 主要时间在更新流水线；继续堆环境数不能直接解决瓶颈 |

上述案例编号来自 Tune60，仅用于说明机制。先导训练案例独立按当前 Train240 的 profile 和固定 seed 选取，不能把同名编号误认为同一数据集案例。

本轮完整 microbatch 数值等价对照的状态是 **用户豁免，未通过证明**。新计划保留有限值、合法 mask、行为概率回放和真实恢复检查；不重新引入已豁免的昂贵全量等价对照。

## 2. 目标与统一比较口径

主指标为同一 split 的均值之比：

`gap = mean(Cmax_policy) / mean(Cmax_IGA1800) - 1`。

以下是本计划建议预先锁定的验收阈值，并非本轮已经达到的结果：

| 层级 | 标准 |
| --- | --- |
| 阶段里程碑 | Validation120 上 gap ≤ 1% |
| “接近 IGA1800”主目标 | 独立 Confirmation120 上 gap ≤ 0.5%，逐案例配对 bootstrap 的 95% 区间上界 ≤ 1% |
| 尾部和完成性 | 100% 完成、无 cycle；比 IGA 慢超过 5% 的案例占比 ≤ 10%；各自最差 10% 案例的平均 Cmax 比值 ≤ 1.01；OOD-stress 均值 gap ≤ 1% |
| 跨 seed 稳定性 | 同一配置共 3 个训练 seed；部署 seed 在 Validation 上预选。部署 seed 满足完整主目标，另两个均值 gap 均 ≤ 1%，并报告各 seed 全部风险指标 |
| 更高目标 | 平均 gap ≤ 0%，仍满足风险要求 |

若只达到点估计目标，而区间或尾部不通过，报告“平均值接近、可靠性未达标”，不宣布完整成功。案例区间只度量给定训练 seed 的案例不确定性；不能把 3×120 个结果当作 360 个独立案例。

以已曝光 Tune60 的 IGA1800 均值举例：1% 对应 8264.60 秒，0.5% 对应 8223.68 秒，后者比当前 H2 Best6 还需少 115.63 秒。这只是目标量级示例；H3 RL 基线尚未测量，正式阈值按相应 split 的 H3 IGA 均值计算。

统一环境为 `H3/F4/soft`：future-intent horizon=3，bounded_frontier，frontier=4，request capacity=5，reservation grace=300，safety margin=60，release-aware ETA、deadline-aware dispatch、departure lookahead 开启，slack forecast=0。其余物理案例、完成条件与 IGA 合同逐项核对。

主评测继续单次 H 解码、tau=0.3、seed=1，保留固定 12 案例批次和完成槽位合同。训练先保留 AR_sample、tau=0.03。AR 评测只作为诊断；best-of-K、多次采样择优和在线搜索另列时间/查询预算，不计入单次 RL 主目标。

## 3. 初始化、数据与教师身份

**权重起点 S0：Best6，模型权重严格加载；新实验重置 Adam 和实验计数。** 这是 H3 新协议的 warm start，不声称是旧 H2 训练的数值连续恢复。R0、R1 共用同一 S0；B0 保留为不可变锚点。PPO 的 source-relative 基准成本重新计算为 B0 在 H3 下的成本，不沿用 H2 成本。

| 输入 | 固定身份或位置 |
| --- | --- |
| B0 | `artifacts/stage2_frozen/20260912/checkpoints/b0.pt`；SHA256 `b7740b2b688c83dc7fa65bed6381243cdf2f6675f4b96808ff05a41d4e44e7e8` |
| S0 / Best6 | 本轮 `attempts/20260913T164751_1357885/train/models/batch_0023.pt`；SHA256 `d8a99daa66e7cf89cc9c83a11b4ace2735e3a01c59020e80e58deb4c911fc87e` |
| 已完成 run | `result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260913_r8_train240_env240_mb240_gpu0` |
| 主教师/基准 | `result/hkbz_train_logs/stage3_matched_iga_h3f4_tune60_20260831_r1/full`；scope=`stage3_full_joint_policy`，method=`joint_iga_all` |
| Train240 | 延续现有精确名单，192 IID + 43 OOD-stress + 5 OOD-scale；[完整 CSV](result/stage3_analysis/stage3_next_plan_20260915/train240_cases.csv) |
| 先导 Fit24 / Probe24 | [已生成清单](result/stage3_analysis/stage3_next_plan_20260915/pilot_manifest.json)，seed=2026091501，两个集合不重叠 |
| 开发评测 | 现有 Validation120 用于选模；Tune60 用于历史比较和诊断，二者均已曝光 |
| 独立确认 | 现有未开启的 Confirmation120；先复核 exposure ledger，再锁配置、选模规则和 seed 后开启 |

本次按 run manifest 的 `case_sha256` 和 `content_sha256` 两种身份检查，train / validation / tune / confirmation 两两交集均为零。先导各含 10 个 profile：每类 2 例，stress_joint 另加 2 例，resource_ood / stress_arrival 各另加 1 例。**Probe24 只是本轮先导 BC 的留出集，旧 RL/Stage2 可能已见过这些案例；它不是最终泛化证据。**

主训练继续 240 个唯一案例，保留每个数据 epoch 的 `192 IID + 4×48 OOD = 384` 次访问，不根据 Tune 失败名单反复改采样分布。Train240 列表来源文件 SHA256 为 `4509e9ab09cc793292ae32bcf00baa936243af620db2cfca17b261006f012305`。

H3 新协议的 ValueNorm 只用 B0_H3 的 Train240 基线轨迹拟合一次，随后所有实验臂共用并冻结。记录这项协议变更，不把旧 Train600/H2 统计改写成 H3 统计。

现有 H3 IGA 可直接复用的已核验证据是 Tune60；Train240、Validation120、Confirmation120 的匹配 H3 全联合教师/基准需清点并补齐。旧 H1 联合教师或 H2 资源教师不能替代。每例固定一次 IGA180 + 追加 1620 秒，保留 incumbent 继承、seed、求解器版本、实际候选数和墙钟。不能额外挑选多个 seed 的最优值却仍标记为 IGA1800。

## 4. 分阶段工作与继续条件

### P0：消除比较与接口不一致

1. 建立独立 H3 协议版本：先按冻结 B0 合同严格加载权重，再显式设置 H3 环境；不修改冻结交接文件。用原固定 12 例做一次 H2 零更新 canary，复用已有完整 H2 结果。
2. 分别评测同权重 B0、S0 在 H3 的 Train240、Validation120、Tune60。H2→H3 的变化单独报告为环境/候选集效应。H3 的小固定批次复跑验证确定性，之后缓存结果。
3. 首批教师检查“可表达性”：IGA 的每一个可控飞机/资源/转运决策，是否在 RL 动作空间内且 mask 合法；用教师动作在同环境强制回放，复现成本及关键事件。环境成本复现容差先定 1e-5 仿真秒，任何更宽容差必须由数值来源解释。
4. 基线缓存键包含 checkpoint、代码/环境、案例内容、history、decoder、tau、seed、评测分批合同的哈希。版本不变时复用；身份变化时只重算受影响项。

**继续条件：**合同核对通过，教师可控动作覆盖完整、成本可复现。若 IGA 动作根本不在策略空间内，先处理明确的动作表示缺口，再谈学习效率；不将无法表达的标签静默丢弃。

初始化回退规则也在训练前固定：若 B0_H3 在 Validation120 上比 Best6_H3 平均好至少 0.3%，并且完成性、最差10%均值均不差，则两臂一起改用 B0 权重作为 S0，重新生成 S0 轨迹并记录决定。该选择只依据开发集，不能看 Confirmation 后回退。

### P1：48 例教师与闭环可学习性先导

为固定 Fit24、Probe24 生成 H3 IGA1800 完整示范；S0_H3 轨迹由 P0 复用。逐例建立示范库：IGA 至少改善 `max(5 秒, S0成本×0.1%)` 时采用 IGA；否则保留 S0 的成功轨迹。原始 IGA 基准保持原值，不被该示范库替换。

训练整个共享 encoder 及飞机、普通资源、转运车角色头；缓存原始图/标签，不能缓存会使共享 encoder 收不到梯度的固定 embedding。按案例和角色平均监督损失，防止决策数多的角色淹没其他角色。监督标签使用真实条件动作和原生 history，不把临时 request 数组位置当作稳定身份，也不把 Hungarian 边分数直接冒充动作概率。

先导最多比较两个 BC 学习率：1e-5、3e-5；每个最多 12 遍 Fit24 或 2 小时，以先到者为准。使用相同起点、案例和固定 Probe24；以 **H 闭环 Cmax** 选候选，NLL、动作命中率只做诊断。

定义集合 S 上可用教师收益 `P(S)=Σ max(C_S0−C_demo,0)`，闭环收益 `R(S)=Σ(C_S0−C_candidate)`。若 P 接近零，则没有足够的示范改进信号，应先核查教师质量，不能硬算恢复比例。

**扩大到全量的条件：**Fit24 的 R/P ≥ 50%；Probe24 平均 Cmax 相对 S0 不退步超过 0.2%，且正的教师潜力至少恢复 10%；两个集合全部完成、无 cycle。若只有拟合 loss 下降而闭环无收益，最多进行一次有明确原因的接口/history/解码对齐修复并重跑先导，之后停止此分支，不能直接追加整轮 PPO。

### P2：全量教师与两个实验臂

先导通过后，补齐剩余 192 个 Train240 教师，并生成 Validation120 的 H3 IGA1800 参考。Tune60 教师仅用于既有诊断，不进入梯度。所有 PPO 随机轨迹均保留，不能只保留优于 IGA 的采样；示范筛选仅作用于单独的监督数据流。

| 实验臂 | 方法 | 要回答的问题 |
| --- | --- | --- |
| R0 | S0_H3 → 纯 PPO，采用第 5 节的新更新日程 | 增加实际更新能否改善当前策略？ |
| R1（主线） | 同一 S0_H3 → 联合示范 BC → PPO + 衰减的示范辅助项 | 教师在相同 RL 预算上是否提供额外收益？ |
| S1（诊断 checkpoint） | R1 完成 BC、尚未 PPO 的模型 | 收益来自模仿还是后续 RL？不另开大实验 |

R1 全量 BC 最多 5 遍 Train240 或 8 小时；每遍在 Validation120 评测，按预定选模规则取 S1。BC 结束后 PPO Adam 重新初始化，与 R0 一致。BC 的梯度步数和时间单独计账，因此“相同 RL 预算”不意味着两臂总计算量相同。

R1 的 PPO 损失增加独立教师监督项。由于原 PPO 按轨迹累积决策而 BC 按案例/角色平均，不能直接拍定一个未经标定的原始系数。用固定 Train240 先导轨迹，在梯度裁剪前令教师项共享 encoder 梯度范数约为 PPO 项的 20%，得到初始系数并冻结；随后按 96 次计划 PPO 步从 1 倍线性衰减到 0.25 倍。标定比值、实际系数、教师梯度与 PPO 梯度分开记录。若 PPO 梯度近零，先修复标定，不用极端系数强行匹配。

两臂使用相同的 Train240 日程、成对 seed、环境参数、PPO 学习率和选择规则。首个研究 seed 固定为 2026091502。BC 辅助数据顺序独立记录。初始保持 actor lr=1e-5、角色倍率 shared/plane/device/transporter=1/0.25/1/0.5、clip=0.2、grad clip=1、soft/hard KL=0.02/0.04；不同时做宽范围学习率扫描。

**阶段评测：**PPO 数据 epoch 1、2、4、6、8 保存并评测 Validation120；Tune60 只在 S0、S1 和最终选定模型上作完整诊断。前 2 epoch 是 768 次访问、至多 24 次 PPO 更新。出现未完成/cycle 或明确恢复错误立即停止；相对 S0 的均值退步超过 1% 且连续两次评测出现时停止对应臂。

**中止无效方向：**到 epoch4，若 R0/R1 及 S1 均未比 S0 改善至少 0.3%，且对正的 IGA 均值差距未缩小至少 25%，本方向停止在已匹配预算处，先分析机制。若一臂有进展、另一臂安全但较弱，继续两臂到相同预算以形成可解释对照；因安全或超时提前结束的臂只比较共同预算，不拿不同预算的终点宣称因果优势。

### P3：配置锁定与独立确认

当 Validation120 上达到均值 gap≤1%、完成率100%、>5% IGA 退步比例≤20%、尾部均值比≤1.02、OOD-stress gap≤2% 的阶段准入时，才投入另外两个训练 seed：2026091503、2026091504。它们从同一 S0 权重独立运行获选配置，包括该配置规定的 BC/PPO 流程；三 seed 结论仅针对该 warm start。

锁定算法、超参数、数据、checkpoint 选择规则和部署 seed 后，开启未曝光的 Confirmation120，并一次性补齐其 H3 IGA1800 参考。三个 seed 全部报告，不根据 Confirmation 结果挑 seed、重选 checkpoint 或继续训练。达标结论按第 2 节完整标准判断；若不达标，这个集合从此记为已曝光，不能再次宣称独立确认。

## 5. 采样吞吐与优化器更新分开设计

**首选候选为 240 个并发环境 + 384 次逻辑采样 + 64 条轨迹/优化器步 + 2 遍 PPO。它是待实测的研究配置，不是已验证最快配置。**

| 参数 | 本轮 Train240 | 下一轮首选 |
| --- | --- | --- |
| 同时活跃环境槽位 | 240 | 240 |
| 承载环境的 CPU 进程 | 64，进程内多环境 | 64，进程内多环境，CPU 进程不加载网络 |
| 一次采样后更新的逻辑访问量 | 240、144 分别更新 | 同一行为策略先收集 240+144=384，再更新 |
| 优化器 minibatch | 当前完整采样批次 | 64 条完整轨迹 |
| 单次反向 microbatch | 240 | 优先 64；需要时分为 32+32，累积后仅做一次 Adam step |
| PPO 数据复用 | 每批 2 遍 | 每个 384 轨迹集合 2 遍 |
| 每数据 epoch 的 Adam 步 | 2×2=4 | (384/64)×2=12 |
| 8 epoch 新 RL 预算 | 3072 访问 / 32 步 | 3072 访问 / 至多 96 步 |

PPO 本身支持对采样数据执行多轮 minibatch 更新；参考 [PPO 原论文](https://arxiv.org/abs/1707.06347)。这里增步是否带来实际收益需要 R0 对照验证；不能由 KL 小或步数少直接推导最佳学习率。

384 条轨迹必须来自同一冻结 behavior policy，PPO 期间 old_logp 保持采样值。第二个 144 采样波次之前不做更新。RNN 按当前权重重放完整前史；TBPTT=8 限制反传长度，不能拿旧隐藏状态跳过必要历史。每个 minibatch 记录行为版本、当前版本、各角色 KL；完整一遍后对全部 384 做 KL 检查。soft KL 提前结束时记录实际步数，hard KL 触发时回到最近合法提交点。

384 是逻辑数据集合，不要求同时放入显存；原始图、history 和 old_logp 放在 CPU 或可恢复的磁盘存储中，按 minibatch 加载到受限缓存。性能先导须同时测量保留 384 条完整轨迹的主机内存峰值，不能只检查反向传播显存。

只比较固定 optimizer minibatch=64 下的物理 microbatch 32/64，先做一个完整 384 访问周期，覆盖采样、图处理、回放、两遍反向、Adam、checkpoint 和运行时验证器峰值；选择端到端耗时最低且资源达标的 microbatch。若此方案过慢，另立固定 optimizer minibatch=128、每 epoch 6 步的替代实验，重新锁预算；不能在正式两臂中途悄悄切换。

主要吞吐统计为：完成且可用于 PPO 的访问数/完整周期小时、完成的实际 Adam 步/小时、验证通过 checkpoint/天，并同时报告 RL 质量。单独决策吞吐、GPU 利用率和峰值显存只用于定位瓶颈。微调 microbatch 带来的浮点差异按用户已接受的边界处理。

## 6. 实现任务与必要验证

这是代码任务清单，尚不存在可直接运行的新方案命令；实现后应生成独立 manifest、不可变源码快照和确切入口命令。

| 工作 | 现有代码落点 / 交付物 | 必要验证 |
| --- | --- | --- |
| H3 独立合同与初始化 | `stage3_b_shared_b0_engine.py`、`onpolicy/utils/stage3_b_shared_b0.py`；独立协议版本 | B0/Best6 权重严格加载，冻结源文件不变；零更新 H2 canary 与 H3 基线身份可追踪 |
| 三种 batch 语义、计数和保存 | B0 engine、worker、controller；新逻辑 rollout/optimizer minibatch/microbatch 字段 | 384 来自同一行为版本；96 次预算算术；实际步计数；完整周期吞吐 |
| PPO minibatch 回放 | engine 的 `update` / `_backward` / `replay_metrics` | 采样概率重现、合法 mask、有限值；教师数据不能进入旧 on-policy 轨迹契约 |
| 教师转换和联合监督 | 复用 `generate_stage3_joint_iga_labels.py` 的轨迹材料，新增原生 B0 history 的监督适配器 | 完整动作空间覆盖、教师强制回放、共享层及三角色头确有梯度、H 闭环收益 |
| 可靠恢复 | 存储 optimizer、RNG、行为版本、采样集合、PPO pass/minibatch 游标、教师游标和校验值 | 真正中断后继续，不重复或漏掉 Adam 更新；若暂不支持半周期提交，只在完成整个 384 周期后提交并从该边界恢复 |
| 关键等待与作业诊断 | 扩展已有 trajectory 分析，而非改变奖励 | 关联最终实际服务的设备与取消/改派记录；区分等待、转场、并行作业、服务后间隔 |
| 评测缓存和报告 | validator 的内容寻址缓存、配对分析和 exposure ledger | 固定 checkpoint 同 split 比较，参考合同一致，历史/训练/确认结果分开 |

当前 global_batch 白名单不含 384，budget 还按“每采样组固定两次更新”计数；engine 的 freshness 检查也与整批更新绑定。因此这不是只改两个配置值就能正确实施的方案。

新改动涉及行为概率、优化器步数和恢复语义，需有针对性的集成检查；不为纯文档或无关旧分支追加测试。旧 C0 条件头代码可审查复用，不能假定它符合当前 B0 history/decoder 合同。历史小样本 NLL 大幅下降未证明闭环调度收益，故本轮 P1 是必要判别实验。

## 7. 资源与时间预算

默认只用 GPU0（RTX A6000，UUID `GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`）和 CPU `0-31,64-95`。保留一个采样/训练共享模型，240 个环境由 64 个无 CUDA 模型的 CPU worker 承载；不复制成 240 个模型。训练、验证和控制器在这半台 CPU 内分配，torch/BLAS 线程数显式限制。

沿用 4 GiB 输入缓存和 activation checkpointing 作为起始配置；GPU 总峰值保留至少 6144 MiB 余量，主机内存沿用 MemoryHigh=102 GiB、MemoryMax=114 GiB 上限并结合启动时实际可用量检查。稳定约 20 GiB 不能代表密集阶段峰值，历史探测曾到约 40069 MiB；不据此承诺一张卡可以同时跑两个完整实验。

IGA 生成只用分配到的 CPU。以 32 个单线程 solver worker 为首选，在少量相同 Train 案例上比较 16/32 worker 的候选数和完工吞吐，确定并发数后冻结求解协议；不要用超额进程让每例 1800 秒内实际搜索量明显下降。性能探测单独计账，正式基准仅采用锁定配置的一次规定求解，不能从探测重复中挑最优值。IGA 与 PPO 分阶段排队，避免争抢同一批 CPU。训练全部结束后，可按既有授权把确认空闲的 CPU 交给验证器，不干扰其他正在运行的任务。

| 阶段 | 初步机器时间预算 | 解释 |
| --- | --- | --- |
| H3 基线及 48 例先导 | 约半天至 1 天 | 含 H3 评测、约 1–2 小时先导教师生成、最多 4 小时 BC 候选；不含代码开发和未知接口修复 |
| Train240 / Validation120 教师 | 约 4–6 / 2–3 小时 CPU 阶段 | 以 32 个 solver worker 粗估；先导 48 包含在 240 内，不重复收费；须用实际完成波次修正 |
| 单个完整 PPO 臂 | 暂按 30–60 小时预留，72 小时上限 | 旧 3072 访问耗时 26.83 小时；新 H3、minibatch 和监督项耗时尚未测量。R1 全量 BC 另计至多 8 小时 |
| 确认教师 | 约 2–3 小时 CPU 阶段 | 仅在选模锁定后生成；RL 三 seed 评测时间另按实测外推 |

若两臂和额外两个 seed 全部运行，按 **4 个完整 PPO 臂、共至多 12288 次新 RL 访问、384 次 PPO 更新** 计算，约需 5–10 天 PPO 机器时间，外加 BC、教师、评测和实现时间；只有前一阶段有效才支出后续预算。当前数字是排期区间，不是实测 ETA。

PPO 更新步数增至三倍，不等于时间必然三倍，因为同一数据仍只反传两遍；但更小批次的图处理、RNN 回放和 GPU 效率可能明显变差。第一个完整周期后，分开报告剩余采样、更新、评测/教师阶段的预计时间，再重估整轮预算。若 64 轨迹方案明显超预算，在正式两臂前决定是否采用预定义的 128 替代方案。

## 8. 选模、诊断和失败后的分支

主选择规则在 Validation120 上固定：先过滤未完成/cycle，以及相对 S0 均值退步>0.2%、各自最差10%均值退步>1%、任一 distribution 均值退步>1% 的候选；其余按平均 Cmax 最低选，完全相同时选更早 checkpoint。这组保护条件与接近 IGA 的最终目标分别报告。阶段 P3 还必须满足其 IGA 准入标准。

paired bootstrap 固定 20000 次、seed=2026091505、按 profile 分层，在每次抽样中同时抽取相同案例的策略和 IGA 成本；报告均值之比的区间。profile 样本很少时不以某一类名次作结论。所有候选同时报告胜/平/负、>5%/10% 退步比例、各自 worst10%、OOD 分组和计算时间。

诊断保留：最后完工飞机、末尾数架飞机的完成时刻、关键请求的最终服务等待、改派/软预占取消、可合并作业与实际合并、离场接驳和其他服务后间隔。总等待是辅助指标；不能把它的减少直接等同于 Cmax 改善，也不能把累计接驳差当作可直接节省的完工时间。

| 实验观察 | 下一步的有限分支 |
| --- | --- |
| 教师强制回放失败或动作不可表达 | 修复合同/动作接口；先导不通过就不启动正式 PPO |
| 示范 NLL 好，但 Probe/H 调度不改善 | 检查闭环状态偏移和 AR→H 转换；至多做一次定向修复。后续可单列 learner-state 专家查询实验，禁止未验证就扩大数据 |
| R0、R1 都改善且 R1 明显更好 | 保留联合示范路线，进入锁定与三 seed 确认 |
| S1 最好，PPO 持续破坏它 | 保留 S1 的有效结果；不能把 BC 收益归因给 PPO。下一项单独研究 source-relative 标量优势的信用分配或约束方式 |
| R0 更好或 R1 与 R0 无差异 | 不追加复杂教师组件；报告主要收益来自更新日程，按获选配置确认 |
| 两臂均无实质进展 | 在匹配预算处结束；基于训练集反事实，单独比较状态相关 baseline/优势或少数关键动作的信用分配；不同时改奖励、网络和解码器 |
| 仅训练集改善、Validation 退步 | 判为泛化瓶颈；下一研究修订再考虑扩大唯一训练案例，不能在当前 Train240 对照中途混入 Train600 |

闭环先导依据是序列模仿学习中的状态分布变化问题，参见 [DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html)。保留优质自身轨迹可借鉴 [Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf)，但本计划的逐例示范筛选与辅助 CE 并非原论文算法，其收益必须在本环境中验证。

## 9. 交付物与执行顺序

执行顺序固定为：**H3 合同/零更新对齐 → 教师可表达性 → Fit24/Probe24 闭环先导 → 一周期性能/恢复验证 → 补齐教师 → R0/R1 → 配置锁定 → 三 seed 与 Confirmation120。**

每阶段输出独立的 `manifest.json`、输入/代码哈希、`gate_result.json`，训练阶段另有带实际更新时间与版本的 updates、不可变 checkpoints、行为回放和恢复收据；评测输出完整逐案例 CSV、配对统计及 exposure ledger。监控应显示当前阶段、已完成访问、已应用 Adam 步、剩余完整周期、GPU/CPU/RAM 峰值及阶段 ETA。故障不能通过仅修改 status 文件标记为恢复。

本次已交付：本计划；[证据与预算 JSON](result/stage3_analysis/stage3_next_plan_20260915/evidence.json)；[Train240 精确名单](result/stage3_analysis/stage3_next_plan_20260915/train240_cases.csv)；[Fit24/Probe24 清单](result/stage3_analysis/stage3_next_plan_20260915/pilot_manifest.json)，其 SHA256 为 `6af5f82e52242ff0ff1d34d88cf514dcbecc1ce6363230efeb4b6b5726b6bef0`。
