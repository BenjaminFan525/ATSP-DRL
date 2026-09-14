# 本机 Stage3：冻结 B0 → B_SHARED 研究计划（2026-09-12）

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Origin Mode：plan → implementation → run；用户已确认冻结 B0，并授权完成实现后启动实验。
- Origin Date：2026-09-12。
- Version Label：`stage3_b_shared_b0_native_v1`。
- Verification Status：**新入口已实现并进行 CPU 回归、GPU0 小算例技术检查；完整基线/容量/恢复准入由实验服务执行，科学结果尚未验证。**
- 本次运行 tag：`stage3_b_shared_b0_local_a6000_20260912_r2_gpu0`。r1 在 B0 基线阶段发现评估批次兼容问题后停止，未进入正式 PPO；产物保留。

本机承担一个 B_SHARED 学习臂：从同一冻结 B0 精确初始化，联合训练全层共享 GNN 与三个角色策略，检验完整训练能否改善 B0 原生部署解码的调度成本。本轮交付物是可恢复的完整训练证据、预先规定的候选选择结果和独立确认结果。

执行资源修订：用户随后指定仅使用 GPU0 和剩下的一半 CPU。已核对 GPU1 实验占用后半组 CPU，本轮使用 GPU0（UUID `GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`）及物理核 0–31、SMT 兄弟核 64–95。改为一个策略 rank，以 microbatch8 累积 global32；数据、PPO 更新数及科学门槛不变。

19:31 评测调度修订：为加速训练前准备，用两个独立的固定 12 例评测进程，各分配 15 个物理核，控制器保留 2 核；总量仍为 GPU0 和前半 32 核。该阶段调用同一 r2 冻结 evaluator、复用已完成缓存并保留全部 780+180 例检查；结束后恢复原训练 22 核/validator 8 核/控制器 2 核布局。24 例并发逐步结果对照通过，适配器另有 SHA 与测试记录，见 `docs/STAGE3_B_SHARED_B0_PREFLIGHT_ACCELERATION_20260912.md`。

## 1. 研究依据与问题

[历史回顾](docs/STAGE3_REVIEW_20260912.md)显示，已有 Stage3 曾取得约 0.5%–0.8% 的单轮改善，但未形成稳定的 ≥2% 改善证据。例如已完成的 08-31 N0，在自身合同下从 8408.0944 秒降到 epoch3 的 8339.6389 秒，epoch5 又回到 8414.3833 秒。较新文档记载 B_SHARED 收益 0.6027%，但其起点、H/F、预约与历史语义不等同本轮，原始运行产物也未在本机复核。

因此本轮研究问题是：**冻结 B0 在原生 H2/F4/soft、Hungarian 评估合同下，经过充分覆盖 Train600 的全共享联合 RL，是否能取得 ≥2% 的独立确认改善，并在训练最后两个 epoch 保持正收益？**

本轮只设一个学习臂及一个不可变 B0 对照，使用一个训练 seed。全共享沿用当前 B_SHARED 的定义：一套完整共享异构 GNN，保留飞机、普通设备、转运车各自的 GRU 与输出头；不会为三个角色复制编码器。Ready 预测头因注入为 `none` 而保留冻结，不参与本轮学习。

其他机器若运行 C_PRIVATE，只有在来源权重、环境、历史、解码、数据、损失和预算全部一致时，才能作共享/分离的架构归因。本机 B0 与其他机器历史 T2 的分数只可分别报告各自收益。

## 2. 已确定的起点与环境合同

| 项目 | 本轮固定值 |
| --- | --- |
| Stage2 入口 | [冻结 manifest](artifacts/stage2_frozen/20260912/manifest.json)，协议 `stage2_frozen_b0_v1` |
| 起点权重 | [b0.pt](artifacts/stage2_frozen/20260912/checkpoints/b0.pt)，原 `checkpoint_Best.pt`，BC epoch1 / seed11 |
| 完整 SHA256 | `b7740b2b688c83dc7fa65bed6381243cdf2f6675f4b96808ff05a41d4e44e7e8` |
| 初始化要求 | 严格加载全部 508 个模型张量；零更新时张量逐项相等；新建 Stage3 Adam 与 ValueNorm |
| 观测与环境 | `progressive-departure-r014-pipeline-v2`；最大 24 架飞机、80 台设备、104 agents；单例实际数量依数据 |
| 请求与预约 | H2/F4、`bounded_frontier`、request capacity5、soft reservation、grace300 秒、safety60 秒 |
| 其他规划语义 | release-aware ETA、deadline/departure lookahead；具体值由冻结 manifest 校验 |
| 飞机与全局特征 | fixed plane order、`joint_pair`、`f1f2` |
| Ready | `request_ready_prediction=true`，注入 `none`，辅助损失为 0，头及投影保持冻结 |
| 主评估 | 飞机 greedy + 资源 Hungarian；tau=0.3，eval seed=1；每案例一次完整调度 |
| 历史 Tune60 B0 | 平均 Cmax **8352.922222 秒**；本轮须重新回放核对 |

B0 是用户选定且可验证交接的来源；[冻结说明](STAGE2_FROZEN.md)并未宣称它超过全部历史设置或通过 IGA 科学目标。冻结包及其历史结果保持原始字节；新增 Stage3 manifest 独立记录本轮训练范围。Stage2 对 Stage1 参数的来源保护用于交接核验，不能错误延续为 Stage3 全程冻结共享 GNN/飞机策略。

在同一历史 Tune60 上，2% 改善对应 **8185.863778 秒**。历史 IGA180/IGA1800 为 8200.438889/8094.988889 秒；达到 2% 仍不等于达到 IGA1800，也不意味着等计算时间优势。validation120 和新 confirmation120 的门槛分别以各自 B0 实测成本为分母。

## 3. 第一优先级：建立 B0 专用的本机研究入口

当前通用 `train_hkbz.py` 支持冻结 B0 初始化；最新 full-data 研究入口不能直接换路径启动。此次已复现以下问题：

| 现有位置 | 已观察到的问题 | 本轮处理 |
| --- | --- | --- |
| [stage3_research_engine.py](onpolicy/runner/shared/stage3_research_engine.py) | `policy_args()` 依赖本机缺失的历史 `BASE_MANIFEST`；checkpoint loader 默认要求 B0 不含的 ValueNorm | 显式接收冻结 B0 解析后的 args、模型及新归一化状态，去除新分支对旧研究目录的依赖 |
| [stage3_encoder.py](onpolicy/algorithms/utils/stage3_encoder.py) | `install_encoder(..., 'F_SHARED')` 拒绝带 Ready 头的 B0；CPU 调用已复现 ValueError | B_SHARED 直接保留原生单个 `HeteroGraphEncoder`，全层解冻，无需表示包装器改名或复制权重 |
| [stage3_local_exploration.py](onpolicy/utils/stage3_local_exploration.py) | `RoleExploration(..., J)` 拒绝 `device_global_matching=true`；CPU 调用已复现 ValueError | 新分支明确分开训练采样和评估解码，并检查真实条件概率回放 |
| [run_stage3_full_data.py](onpolicy/scripts/train/run_stage3_full_data.py) 与其启动器 | 绑定远端 prior、`/data/fanyx`、8 卡；正式控制器未接通已有 commit 的训练恢复 | 新建本机单臂控制器、单 GPU worker、持久验证队列及完整 resume 路径 |

实现采用独立、显式 opt-in 的 B0 engine，复用经过检查的通用访问日程、梯度归约、输入缓存和验证队列组件。历史研究入口的来源限制继续有效，不通过关闭旧校验来伪装兼容。

下列文件已按职责实现；正式启动使用独立 manifest 和冻结源码，完整准入以运行产物为准：

| 计划文件 | 责任 |
| --- | --- |
| `onpolicy/config/env_stage3_b_shared_b0.yaml` | 显式训练预算及更新范围；环境/架构以冻结来源校验，禁止 YAML 隐式默认覆盖 |
| `onpolicy/utils/stage3_b_shared_b0.py` | 协议、数据清单、访问 seed、参数归属、候选规则与 commit 身份 |
| `onpolicy/runner/shared/stage3_b_shared_b0_engine.py` | B0 精确加载、原生历史、随机自回归采样、条件 logp 回放、source-relative 更新及中央 critic |
| `onpolicy/scripts/train/run_stage3_b_shared_b0.py` | prepare / verify / admit / run / resume / report；各步骤幂等且有产物 |
| `onpolicy/scripts/train/stage3_b_shared_b0_worker.py` | 每 rank 采样更新、全局提交、epoch 评估提交与恢复 |
| `onpolicy/scripts/train/launch_stage3_b_shared_b0_local.sh` | 按实测 GPU UUID/CPU 拓扑启动本机服务，持续记录资源与心跳 |
| `onpolicy/envs/HKBZ/test/test_stage3_b_shared_b0.py` | B0 语义、采样回放、全局目标、非空 Adam 恢复等有判别力的测试 |

## 4. 解码、历史及优化方法

### 4.1 主评估保留 Hungarian，训练使用合法自回归采样

[当前 actor](onpolicy/algorithms/gnn_mappo/algorithm/gnn_actor_critic.py)只在确定性推理条件下执行资源 Hungarian；随机 PPO 路径保留逐资源、带占用掩码的自回归选择。本轮采用这一区分：

- **训练行为策略**：三个角色均随机采样，温度均为 0.03；固定角色顺序，每一步依实际前缀更新合法动作掩码。`resource_rl=false`，飞机和共享 GNN 均保留梯度。
- **主评估策略 H**：B0 与所有学习 checkpoint 都使用原生飞机 greedy + 资源 Hungarian，tau=0.3、seed1。验证进程独立加载，防止训练温度 hook 残留。
- **辅助评估策略 AR**：同权重改用资源自回归 greedy，仍以 tau=0.3、seed1 评估；与同解码 B0_AR 比较，只诊断训练与部署解码的差异，不用于选最佳模型。

PPO 的概率比必须来自实际采样行为策略。[PPO 原论文](https://arxiv.org/pdf/1707.06347)定义了新旧策略对同一采样动作的概率比；据此结合本地实现，本轮不会将 Hungarian 选中边的逐设备分数当作随机联合匹配的似然。`return_log_prob_components` 在当前实现中会改变资源全局评分分支，不可为收集 PPO 日志而直接开启。

主方法是当前研究使用的**逐决策因子裁剪 MAPPO surrogate**，不是 Hungarian 联合分布上的精确 PPO。训练/部署解码存在差异是本轮明确的限制；最终是否有效由 H 评估实测决定。

### 4.2 历史语义先保持 B0 原生行为

训练、回放与评估均遵循 [HKBZRunner 的 authoritative history](onpolicy/runner/shared/hkbz_runner.py)：飞机历史来自环境报告的 last-op/last-site，资源历史保留该设备上一次实际执行动作；inactive 步与初始 hidden 的更新规则保持一致。

本轮不在初始化时引入较新研究的 `stable_request_identity_v1` 重映射。可以记录请求索引变化诊断；若后续证明必须修正历史语义，应另立协议并重新测量零更新 B0，不能把修正导致的成本变化记作本轮 RL 收益。

### 4.3 固定学习配方

设全局批次含 N=32 次访问，访问 b 的源成本为 `C_B0,H(case_b)`，本次新鲜完整随机轨迹成本为 `C_b`：

```text
A_b = 0.01 * (C_B0,H(case_b) - C_b)
r_bti = exp(log p_new(a_bti | replayed history, prefix)
            - log p_behavior(a_bti | behavior history, prefix))
L_actor = -(1/N) * sum_b sum_t,i mask_bti *
          min(r_bti * A_b, clip(r_bti, 0.8, 1.2) * A_b)
```

保留正负优势，不中心化、不按案例去重、不按轨迹长度或决策数重加权；mask 只保留实际可学习决策。每个 PPO epoch 从完整历史起点递推 hidden，仅在 TBPTT 边界 detach；不能用旧策略中间 hidden 跳过当前策略的历史回放。

| 项目 | 固定值 |
| --- | --- |
| 训练 seed | `2026091202`；每次访问独立派生环境/采样 RNG，绑定 case hash、epoch、visit ID |
| Actor | Adam 基础 LR=1e-5；shared/plane/device/transporter 倍率=1/0.25/1/0.5 |
| Critic | 保留 B0 中央 team critic 权重，Adam LR=1e-4；未使用的角色 critic 保留张量并冻结 |
| PPO | 每批 2 个完整更新 epoch，clip=0.2，gamma=1，TBPTT=8 |
| 梯度 | 全局目标归约之后统一裁剪 norm=1；共享 GNN 汇总三角色梯度，使用普通求和 |
| KL | 按有效决策加权的 soft=0.02 / hard=0.04，同时记录各角色 KL |
| 其他目标 | BC、rank、Ready 辅助损失、IGA 势函数、额外熵奖励均为 0 |
| 数值 | FP32 张量、确定性设置，无 AMP/compile；训练禁用 TF32，评估保留原生 cuDNN allow_tf32=True，CUDA matmul TF32 始终关闭 |

B0 不含优化器/ValueNorm 状态。本轮新建单个 team ValueNorm，在正式训练前用 **Train600 的 B0_H 基线轨迹**标定成本剩余量 `-0.01*(C_B0,H-t)`；按 1/4/4 访问权重累计全局事件，固化统计和 SHA，训练期间不再更新统计。此过程没有模型梯度或 optimizer step，不使用 validation/Tune/confirmation 拟合统计。

Critic 拟合同单位的实际轨迹 MC cost-to-go，每访问内按全局事件平均、再按访问平均；同一 team value 不按 104 个 agent 重复计数。输入编码 detach，critic 不更新共享 GNN/角色 GRU，两个 Adam 参数集合不相交。Actor 的优势只来自上述 B0 相对成本，不使用 critic 预测；不能引入历史 T2 的归一化或恢复状态。

若 soft KL 导致本批不足两次 actor 更新，或 hard KL/数值合同失败，停止推进并记录实际预算，不静默降低 PPO 轮数继续跑。技术失败与完整预算下未改善分别归档。仅因前 480/960 条轨迹收益低而停止，不属于本轮预注册规则。

## 5. 数据、预算与评估开销

| 数据 | 本机路径/来源 | 用途与数量 |
| --- | --- | --- |
| Train600 | `onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train` | 唯一案例 600：IID480、OOD-stress108、OOD-scale12 |
| validation120 | `onpolicy/envs/HKBZ/dataset/fjsp_v3_stage1_v2_eval_s20260803/validation` | 主选模集：60/54/6；不进入梯度 |
| Tune60 | `onpolicy/envs/HKBZ/dataset/fjsp_v3_resource_joint_eval_s20260811/joint/tune` | 历史对照及风险检查：30/27/3；不进入梯度 |
| 新 confirmation120 | 启动前独立生成并冻结，建议 namespace `stage3_b0_confirm_20260912`、seed `2026091217` | 分布 60/54/6，十个 profile 数量与 validation120 相同；成本保持未打开直至候选锁定 |

上述 train/validation/Tune 数量已逐 metadata 核对。validation 的 `stress_joint` 有 24 例，Tune 有 12 例；profile 与 IID/OOD distribution 是两个不同维度。本地 prepare 生成新 confirmation120，未打开其成本。生成器版本、参数、全套案例清单及物理内容 SHA 在启动前冻结，并与 train、validation、Tune 及已知暴露集合逐项去重。

全部正式评估使用固定 12 案例批次，保留已完成环境的前向槽位直至整批结束。Tune60 顺序与冻结 Stage2 清单相同；其余集合的顺序在 manifest 冻结。中断时只跳过整批已完成结果，否则重放该完整原批次并核对已落盘案例。恢复原生环境构造、seed/cursor 和 cuDNN 推理设置；这些均属于 B0 评估兼容合同，不能由吞吐优化改变。

旧 TrainDiag/Fit/Probe/Pilot 中属于 Train600 的案例作为训练诊断使用。历史 Stage1 test60、Stage2 Finalblind 不改名作为新的盲测。新确认集只由锁定候选触发一次评估；若本轮没有达到主验证门槛的候选，保留未打开状态。

每个数据 epoch 访问 **960 次 = 480 IID + 432 OOD-stress + 48 OOD-scale**，保证全部 600 唯一案例覆盖；每个 stress/scale 案例各访问 4 次，每次重新采样完整轨迹。每批行为策略版本固定，不跨版本拼接轨迹、不以 best-of-k 筛样本。

| 项目 | 完整预算 |
| --- | --- |
| 数据 epoch | 8 |
| 新鲜训练轨迹 | 8 × 960 = **7680** |
| 全局批次 | 32 次访问/批；30 批/epoch；共 **240 批** |
| Actor 更新 | 2 次/批；完整执行为 **480 次** |
| GPU 布局 | GPU0 上一个逻辑策略 rank；16 个环境分批采齐 32 次访问，microbatch8 累积更新 |

额外环境查询单独记账，不并入 7680：

| 查询 | 预算 |
| --- | --- |
| 原生 B0_H 基线 | Train600 + validation120 + Tune60 = 780；训练集轨迹同时提供归一化统计 |
| 新入口零更新复验 | validation120 + Tune60 = 180，另加代表性训练案例技术检查 |
| 每 epoch 主评估 H | 8 × (120 + 60) = 1440 |
| 辅助 AR 评估 | validation120，在零更新、epoch4、epoch8、最终候选执行；重复 checkpoint 去重，最多 480 |
| 新 confirmation H | B0 与唯一锁定候选各 120，共 240，仅在候选达标后执行 |
| 数值/容量/恢复 canary | 固定 Train600 案例；次数在 admission 记录，不用于选超参数或计正式轨迹 |

## 6. 开跑前必须完成的技术检查

| 检查 | 通过条件与证据 |
| --- | --- |
| 来源与零更新 | 全部 508 张量严格相等；新入口与原生评估器的 validation120/Tune60 逐例成本相等至 1e-6 秒、完成率 100%；代表案例逐步动作、mask、历史相等；Tune 均值复现 8352.922222 |
| 可训练范围 | 真实 PPO 更新后共享 GNN 各层及三个角色策略有正确梯度/参数变化；Ready 与未使用 critic 无变化；Actor/Critic Adam 无参数交集 |
| 行为概率 | 同权重强制回放随机动作，动作合法集合和历史完全一致；full-history logp 最大误差 ≤0.002，概率比接近 1，无非有限值；不使用 Hungarian 回放似然训练 |
| 全局目标 | 构造包含重复案例、不同轨迹长度与正负优势的批次，与显式全局损失/梯度对照，确认访问平均和归约缩放正确 |
| 单卡微批 | 同一 global32、访问顺序和 RNG，比对 GPU0 上 microbatch8/4 的完整累积；模型 atol=2e-6，Adam atol=2e-6/rtol=2e-5；动作/RNG 日程一致；双 GPU 检查由用户资源限制取消 |
| 真实恢复 | 完成非空 Adam 更新并发布 commit 后中断，从新进程恢复；下一次更新及最终模型、Adam、ValueNorm、RNG、游标与连续执行对齐；重启控制器不会漏评或重复发布 |
| 容量与吞吐 | 1 个训练 rank 和 1 个持久 validator 同时驻留 GPU0；普通、资源压力、长轨迹、大图均通过；记录整套服务 RAM、整卡显存和端到端批次耗时 |

技术案例从 Train600 按内容哈希固定选择，覆盖 balanced、low_load_ood、resource_ood、stress_joint 及规模/轨迹边界。用于测试的更新权重在正式训练前丢弃，正式 run 重新精确加载 B0。

本轮实现回归已取得 44 passed、2 skipped、1 deselected，覆盖新增 B0 引擎和受影响的 Stage2→Stage3 交接。跳过旧 GPU 条件测试、取消旧四 rank 检查；本轮真实单卡 global32/IPC 准入仍由新入口执行，产出实际进程日志与 admission。CPU 测试不能代替 GPU 容量或完整基线复验。

## 7. 本机资源与执行方式

已通过 `/proc` 核对本机有两张 RTX A6000，CPU 为 Threadripper PRO 5995WX、64 物理核/128 线程，RAM 约 251 GiB。当前沙箱 `nvidia-smi` 不可用不等于宿主机驱动故障；启动时必须实测 GPU UUID、总显存、现有进程和可用容量。此次内存快照约有 157 GiB available，仅作状态记录，不能视作已预留资源。

建议资源方案：

- 训练 rank 使用物理核 0–21 及兄弟核 64–85；持久 validator 使用 22–29 及 86–93，与训练同驻 GPU0；控制器使用 30–31 及 94–95。整套服务限定在前 32 个物理核。
- 单 rank 每批收集 32 条完整访问，使用 16 个并行环境；技术检查比较 microbatch8/4。无论物理并发如何，都累计到真实 global32 再裁剪和 Adam 更新。
- 整套服务建议 `MemoryHigh=96 GiB`、`MemoryMax=128 GiB`、`SwapMax=0`；只有实测可用资源能容纳上限并至少保留 32 GiB 系统/外部余量时才准入，否则先收紧可验证的物理缓存/并发。不能静默减少全局批量。
- 以整卡 NVML 总占用做准入，48 GiB 卡建议至少留 6 GiB 余量，含共卡 validator；不能只看 `torch.memory_allocated` 或继承旧 Stage2 的 0.4 显存比例。
- 只缓存不可变图输入；全层可训练 encoder 的输出不得跨 optimizer step 缓存。同一步 critic 可以复用 detached 编码。
- 每批记录 rollout、完整历史 replay、更新、checkpoint、验证队列等待与端到端耗时。单 rank 不发生跨 GPU 梯度归约；报告墙钟时间、总预留 GPU 小时和训练/验证查询数，共卡 validator 不重复计整卡预留时长。

验证队列绑定不可变 checkpoint SHA；最多积压两个 epoch 请求，超出后反压训练。资源每 10 秒采样、心跳每 30 秒，15 分钟无进度先提示并保留诊断；整轮硬超时 14 天，单评估请求 6 小时，环境 rollout 上限 4000 个事件步、IPC 超时 600 秒。OOM、非法状态、数值或概率合同失败只停止本轮拥有的进程，保留现场。

## 8. 跨天恢复与证据保存

恢复是第一轮实现的必要交付，不推迟到训练中断之后再补。

1. 每个完整 global batch 保存唯一策略 rank 的模型、Actor/Critic Adam、固定 ValueNorm、Python/NumPy/Torch/CUDA RNG、访问日程与游标、epoch、policy version；checkpoint 与更新记录完整后原子发布带 SHA 的 global commit。
2. 控制器 `resume` 显式消费已发布 commit，校验 B0 来源、代码/配置/数据 SHA 及 rank 身份。后续恢复不能再次用 `--stage2_frozen_manifest` 执行新训练初始化，也不能重标定 ValueNorm。
3. epoch 评估请求与 commit 绑定；队列恢复对已完成请求去重，对未完成请求核对后续接。保证每个完整 epoch 有且只有一份可用的 H 评估结果。
4. 部分批次没有完整 commit 时，保留为未提交 attempt；从上一完整 commit 重做该批，重做的计算量单独记账，不能跳过访问或重复累计正式预算。已提交批次不重跑。
5. 正式 run 使用独立源码快照和完整文件 SHA 清单。当前工作区含未提交/未跟踪研究代码，只记录 Git HEAD 不足以复现。外部工作区后续修改不影响冻结运行源。
6. 状态判断综合进程、心跳、游标推进、队列和 terminal/completeness 文件。技术中断标为 interrupted；完整预算未过科学目标标为 not_passed，两者不合并成算法失败。

## 9. 选模、统计及结论标准

主指标为同一案例集的 `gain = 1 - mean(C_candidate,H) / mean(C_B0,H)`。每个 epoch 保存完整 validation120/Tune60 逐案例结果，报告均值、配对差值、胜例比例、分布/profile、最差 10% 均值和退化超过 5% 的案例比例。

每个评估集均执行风险门：完成率 100%，无 cycle/deadlock/timeout；`ood_stress` distribution 与 `stress_joint` profile 的平均成本退步均 ≤0.5%；自身最差 10% 成本均值相对 B0 自身最差 10% 的退步 ≤1%；单例退步 >5% 的比例 ≤5%。要求集合包含相应分组，空子集不能默认为通过；小样本 OOD-scale 逐例另外展示。

正式选择及确认顺序固定：

1. 完成全部 8 个 epoch 和对应验证后，在 validation 与 Tune 风险门都通过的 epoch checkpoint 中，按 **validation 平均成本最低**选择；完全同分选较早 epoch。保留 Last 和全部中间 checkpoint。
2. 若没有风险合格候选，或所选模型 validation 改善不足 2%，标记本轮主目标未通过，新 confirmation 保持未打开。
3. 候选、源码、评估协议和选择 JSON 一并冻结后，只对 B0_H 与此候选打开新 confirmation120。不得用其结果改选其他 epoch、调参再测同一集合。
4. 本轮通过须同时满足：候选 validation 改善 ≥2%；confirmation 改善 ≥2% 且配对 95% CI 下界 >0；三个评估集风险门通过；最后两个完整 epoch 的 validation 收益均 >0，且 validation/Tune 风险门通过。

CI 对物理案例作配对、profile 分层 bootstrap，2000 次，固定 bootstrap seed `20260905`；每次重采样计算同一 ratio-of-means 指标。训练重访和 8 个 epoch 不作为独立样本。120 例沿用现有覆盖规模，并非已经完成的 2% 检出功效保证；报告实际 CI 宽度。结论限定为这个训练 seed 下的案例不确定性，多 seed 稳定性留待后续独立复验。

同步记录以下诊断，以解释完整训练结果：采样胜过 B0 的比例、正负优势与实际更新量、各角色/共享层裁剪前梯度、KL/clip fraction、真实 logp 回放误差，以及 H 与 AR 的成本曲线。在预先固定的训练案例上于初始/epoch4/epoch8 做少量不执行 optimizer step 的角色梯度冲突诊断；本轮不据此在线切换 PCGrad 等算法。

| 完整结果模式 | 下一步研究方向 |
| --- | --- |
| H 达到上述确认与末期稳定性门槛 | 冻结候选，下一阶段做独立训练 seeds 复验；具备同合同 C_PRIVATE 结果后再讨论架构收益 |
| 采样/AR 改善，H 不改善 | 优先研究可训练评分如何影响 Hungarian 排序，作为新的解码/目标对齐实验 |
| 采样存在赢家但训练后 greedy 不改变 | 复盘优势信号、更新幅度、clip/KL 与共享角色梯度干扰 |
| 三种口径都无充分改善 | 复盘探索分布与 credit assignment；本轮给出完整负结果，不以继续扩大网络替代诊断 |
| 数值、容量、恢复或来源检查失败 | 修复技术问题，形成新 attempt 的验证记录；不宣称已否定 B_SHARED |

## 10. 工作顺序、交付与时间估计

| 阶段 | 具体工作 | 出口产物 | 计划时间 |
| --- | --- | --- | --- |
| P0：来源与本机入口 | B0 engine、参数/历史/解码合同、Train600 日程、原生基线及确认集冻结；实现恢复控制器 | 代码及测试、manifest、source baseline、数据暴露台账、零更新对齐报告 | 约 0.5–1 个工作日开发；基线调度耗时另实测 |
| P1：真实技术准入 | 单卡微批目标一致性、非空 Adam 跨进程恢复、1 rank + validator 容量与吞吐 | JUnit、GPU/IPC 日志、资源方案、不可变 admission、完整预算 ETA | 由实测长轨迹和 global32 检查耗时确定 |
| P2：完整 B_SHARED 训练 | 从 B0 重新初始化，8 epochs/7680 轨迹/240 批；每 epoch 固定评估 | 逐批更新、全局 commits、8 份 H 验证、资源/耗时台账、Best/Last | 由 P1 的 A6000 端到端批次测量确定 |
| P3：锁定与确认 | 固定规则选模，符合门槛后打开一次 confirmation；完成配对统计与解释 | selection.json、逐案例 confirmation、科学门结果、最终研究报告 | 按评估吞吐估算；另约 0.5 个工作日整理 |

正式 ETA 使用 `剩余批次 × 最近完整批次耗时 + 未重叠验证尾部`，并给出实测波动区间。纯情景换算：若每批 20/30/45 分钟，240 批分别约 80/120/180 小时；这些是算术情景，不是本机性能预测。历史 4090 或三编码器耗时不直接外推到本机 B0。

最终研究报告至少包含：来源与协议 SHA、完整预算实际执行量、B0/Best/Last 的 H 成本与风险、8-epoch 曲线、AR 辅助诊断、confirmation 的配对 CI、资源效率、中断/重算台账及是否通过预注册标准。实现与本次启动记录见 `docs/STAGE3_B_SHARED_B0_IMPLEMENTATION_20260912.md`；是否进入正式优化以 `training_admission.json`、训练 worker 心跳及完整 commit 为准。
