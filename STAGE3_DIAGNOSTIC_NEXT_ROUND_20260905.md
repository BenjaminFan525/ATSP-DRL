# Stage3：先诊断，再比较可学习的同案例回报信号

## Material Passport

- Material ID：stage3-diagnostic-half-20260905
- 类型：代码实现与实验协议；状态：RUNNING_GATE0，实际进展以 suite/status.json 为准。
- 数据：内部 Train600、Tune60；不上传。Gate120 已有历史暴露，不能称为未接触测试集；Finalblind60 不自动评估。
- 实现：独立入口；保留旧 Stage3 runner、checkpoint 与结果。新增代码通过 manifest 文件哈希冻结。
- 范围：落实诊断、四组 pilot、共享异步验证。多种子确认及条件性网络结构改造在 pilot 审阅后决定，不自动扩张。

## 1. 不变项与主要目标

C0 是 Stage2 T2 Epoch3，文件 SHA-256 为
`41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030`，
模型摘要为 `09854d261ea2fdfb6f79aeaf971c660f7d2b38cad7b4172ca41c4e10aa648148`。
501 个模型张量严格加载；恢复三角色 ValueNorm，重置 optimizer，不继承上一轮训练结果。

保持现有 64 维、4 层共享 GNN、GRU、actor/critic 架构；H3/F4、hard reservation、request capacity 5，确定性评估 tau=0.3。
预训练 C0 的 Tune60 平均 makespan 是 8385.5889 秒；正式多种子目标仍是平均至少降低 2%（约 8217.88 秒），不能把一次 best checkpoint 当成达标。

修正一个实现可比性问题：新入口的采样、重放和验证都使用环境提供的飞机 last_op/last_site；设备保留之前实际 request 动作。固定顺序策略的两列动作统一补齐缓存第三列。collection/replay 均 eval mode，禁用 dropout 随机差异，训练阶段仍计算梯度。

## 2. 数据与资源

| 数据 | 数量 | IID / OOD-stress / OOD-scale | 用途 |
|---|---:|---|---|
| TrainDiag | 32 | 16 / 14 / 2 | hard-IGA、重放、best-of-32、分叉 |
| TrainFit | 16 | 8 / 6 / 2 | TrainDiag 子集，BC 拟合诊断 |
| TrainProbe | 64 | 32 / 28 / 4 | 与 TrainDiag、Pilot 均不重叠，拟合后的闭环泛化 |
| TrainPilot | 120 | 60 / 54 / 6 | 四组配对学习对照 |

分层抽取由内容哈希和固定 seed 决定，不按实验结果选案例。验证数据不产生训练标签。检查文件字节 SHA、canonical UTF-8 JSON 数据集指纹及跨 split 内容重复。

整个服务（含 IGA、validator、环境子进程）限制为 GPU 0–3 和 CPU 0–31、64–95，即 4/8 GPU、32/64 物理核（64/128 逻辑 CPU，完整 SMT 配对）。GPU 4–7 及另一半 CPU 不使用。

| Lane | GPU | 物理核对应逻辑 CPU | 用途 |
|---|---:|---|---|
| 0 | 0 | 0–7,64–71 | 契约/IGA/重放/BC-heads，之后训练 |
| 1 | 1 | 8–15,72–79 | best-of-32，之后训练 |
| 2 | 2 | 16–23,80–87 | 反事实/BC-full，之后训练 |
| 3 | 3 | 24–31,88–95 | 唯一共享异步 validator |

每 lane 最多 8 个环境/IGA worker，库线程数 1。GPU allocator 上限 80%，canary 峰值 reserved 必须低于 18 GiB。训练三组并行，第四组在最先空出的训练 lane 排队，不额外占第五张卡。

## 3. 分阶段执行与停止条件

1. 契约检查：重现训练/验证样例 C0；同动作重放成本一致；概率重放误差 ≤0.002；四种更新路径真实 canary。共享 validator 单独完整重现 Tune60，逐案例成本误差 ≤0.1 秒。
2. hard-IGA：TrainDiag32，每例 1800 秒、population20；从头搜索，不把 soft 结果重新贴标签。重建完整动作序列，检查 makespan 及策略支持。
3. 冻结 C0 采样：每例 32 条，记录独立轨迹 seed、完整动作、raw 成本、best-of-1/4/8/16/32。至少 25% 案例出现比 C0 好 1% 的样本，才认为有足够局部学习信号。
4. 真分叉：8 例中段真实状态，各 8 个 joint-action 干预；相同前缀，分叉后都使用确定性 C0 完成。记录改变的角色/agent、真实终局成本、value/C0/LOO 优势。它不是单设备因果归因，也不是 learned Q。
5. 小集拟合：同一 TrainFit16，heads-only 与 full 两种模式，各从 C0 重启，10 遍 BC；纯 C0 和最终模型上测 teacher-forced NLL/accuracy、闭环 Fit 和独立 TrainProbe。目标 NLL 至少减半且 Fit 闭环改善至少 1%。IGA-BC 仅为诊断，模型禁止作为 RL 初始化。
6. 准入：重放通过、采样通过、至少一种拟合模式通过。heads 能拟合则四组都冻结 encoder；只有 full 能拟合则四组共同解冻。都不通过就停止，审查状态/动作表达或训练实现，不自动添加网络宽度或堆计算量。

## 4. 四组 RL 对照

共同协议：同案例 K=8 个独立随机流；同 seed、案例顺序；120 例 × 2 遍＝1920 条新采样轨迹。每组从 C0 初始化，不使用 BC 诊断权重。采用 raw makespan、固定 reward coef 0.01、gamma=1；每案例一次 group update，轨迹等权，时间/决策项求和。不做案例内标准差归一化、不丢掉负优势。

| 组 | Actor 更新的优势/辅助项 |
|---|---|
| R0 | MC remaining-cost return 减去原三角色 value；共同新入口中的 value-baseline PPO 对照 |
| R1 | 完全替换为 `0.01*(C0(case)-C)` |
| R2 | 完全替换为 `0.01*(mean(C_other_7)-C)`，leave-one-out |
| R3 | R2，加每 4 组至多 1 条当前案例 self-winning 轨迹的独立 BC 更新 |

R0 是统一终局成本协议中的 **MC-return PPO**，不是声称与旧 runner 的 GAE/角色时钟/归一化完全等价。四组之间只比较声明的 baseline/辅助更新；与旧实验的差异必须单独报告。

PPO epoch=2，clip=0.2，KL 上限=0.02，TBPTT=8，actor lr=5e-6（plane/device/transporter 比例 .25/1/.5），critic lr=1e-4，gradient clip=1。critic 继续拟合 cost-to-go；ValueNorm 沿用 C0 固定尺度，R1–R3 的 actor 不再混入 value 优势。

同案例组完整结束后才更新；不完整/cycle/timeout 使实验失败，不能把截断低 makespan 当成胜利。R3 只保存每案例最优的一条 **自身新采样、完整且确实优于 C0** 的轨迹；重放时重新计算当前 hidden states，作为独立 off-policy BC，不伪装成 PPO。额外更新和重放量单列，不能声称严格等总算力。

## 5. 异步验证、终点与后续

每 240 新轨迹保存不可覆盖的 checkpoint；每 480 轨迹提交 Tune60。训练继续，不等验证。请求绑定 checkpoint SHA、案例集 SHA、代码 SHA、环境契约 SHA、tau、seed；结果绑定 request ID 和训练进度。缓存持久化，不依赖进程内存。单 validator 一次一模型，不与其他验证请求并行占额外 GPU。

固定终点是 1920 新轨迹；最后两次 Tune 都至少改善 1%，且 OOD-stress/stress_joint 回退 ≤0.5%、tail10% 回退 ≤1%，才能通过 pilot。输出逐案例配对差和分层案例 bootstrap 区间；**单训练 seed 的案例区间不是跨 seed 置信区间**。

pilot 结束停下来审阅；不自动开 Gate 或 Finalblind、不自动追加 seed。后续预定胜者对 R0、3 个全新配对 seed、每组 4800 条新轨迹；重新冻结正式采样权重和预算。正式目标：跨 seed 均值改善 ≥2%、每 seed 正收益、案例胜率 ≥60%、超过 5% 回退的案例 ≤5%，并满足 OOD/tail 门槛。是否扩为 5 seeds 必须事先决定。

## 6. 启动与可追溯记录

```bash
PYTHON=/data/fanyx/conda/envs/maia-hkbz-cu124-20260903/bin/python3.11
# 使用一个新的绝对路径，不覆盖已有 suite。
$PYTHON onpolicy/scripts/train/prepare_stage3_diagnostic.py --output /ABSOLUTE/NEW_SUITE
bash onpolicy/scripts/train/launch_stage3_diagnostic_half.sh /ABSOLUTE/NEW_SUITE
```

核心产物：manifest.json、status.json、commands/、logs/、contract/、iga/、replay/、sample/、counterfactual/、bc_heads/、bc_full/、diagnostic_gate.json、pilot_R*/、validator/、report.json。

遵循 academic-research-suite 实验执行流程：30 秒 heartbeat、进程状态/实际进度监控、硬超时、失败产物保留、`Restart=no`。普通进度停滞只提示；执行快照损坏、输入契约变化、错误退出及声明的硬超时会停止本 suite 的进程组；不终止其他用户任务，不自动缩小配置或静默重试。2026-09-05 恢复修订后，新建及恢复实验使用独立源码快照，共享工作区的源码变化只记录提醒，不再导致停机。

## 7. 启动记录

- 2026-09-05 14:34:11（Asia/Shanghai）启动。
- Suite：`result/hkbz_train_logs/stage3_diagnostic_half_20260905_r1/`。
- 服务：`hkbz-s3diag-stage3_diagnostic_half_20260905_r1.service`，controller PID 1910867。
- 首次复核：active/running、NRestarts=0；契约进程 PID 1910879（GPU 0），validator PID 1910878（GPU 3），心跳及 rollout 步数持续推进。已在宿主机核验 controller 与两 worker 的实际 CPU affinity；GPU 4–7 无本轮进程。
- 冻结代码：324 文件，aggregate SHA-256 `8052ece83693a87d9c0ee3a63175e72f4e48b1a425a60d157c8f391262595297`。
- 启动前验收：35 regression tests passed；heads/full 两种更新均实际反传和保存 checkpoint；开发 smoke 的 C0 case_0004 精确复现 6776.0000 秒；hard-IGA smoke 动作重放精确复现 8359.0000 秒，无不可支持动作。
- 上述 smoke 是实现验收，**不是算法改善证据**。正式 K=8 canary、完整 Tune60 C0 复现及诊断准入尚在运行；不能据此宣称达到 2% 目标。
- 验收摘要保存于 `acceptance.json`，冻结协议在 `manifest.json`，动态进展在 `status.json`。

## 8. 并行实验兼容与显式恢复（2026-09-05 修订）

原进程于 15:46:43 因共享工作区 `stage1_baselines.py` 的文件哈希变化退出；其后 `hkbz_runner.py` 也有变化。它们属于并行基线实验使用的代码，不能回滚。用户已明确授权恢复并放宽共享源码保护。

- 源码：按启动版本复制 Python、配置和脚本到本轮的 `source/`（恢复时在 `attempts/resume_*/source/`）。所有 Stage3 子进程及 validator 的工作目录、Python 导入和 AC 配置绑定到快照；数据及原始 C0 不复制、不改变。
- 共享工作区：只记录 `workspace_code` advisory。基线实验后续修改共享代码不会杀死或热更新 Stage3。快照本身、C0、数据划分、原始命令及实验参数仍校验；这不是关闭实验身份检查。
- 恢复：原 `manifest.json`、旧日志和已完成产物保留，新的执行清单与日志单独写入 `attempts/`；`active_execution.json` 指向当前执行。进程锁防止重复启动同一 suite。
- 本次复用：Canary、反事实 8/8、IGA 8/32、best-of-32 的 2/32 个完整案例。未落盘的 IGA 案例重新执行其 1800 秒预算；不声称保留未完成种群或 RL 权重。
- 恢复准入：新快照先做 6 例 C0 greedy、动作/概率重放，以及新请求 ID 的完整 Tune60 复现。评估缓存按新快照 SHA 区分，不能误用旧版本缓存。
- 采样检查点：每 K=8 条完整轨迹落盘，恢复时检查案例、seed、动作数和完成标记；只补跑缺失组。当前恢复入口仅支持已通过 contract 且尚未进入 replay/BC/RL 的诊断阶段，不能冒充通用训练 checkpoint 恢复。
- GPU 0–3、CPU 0–31/64–95 不变；GPU 4–7 的并行基线实验不在本次操作范围内。准入及研究效果目标不变。

显式恢复命令（`resume_ID` 必须是新的执行名）：

```bash
$PYTHON onpolicy/scripts/train/prepare_stage3_diagnostic_resume.py /ABSOLUTE/SUITE/manifest.json --attempt-id resume_ID
bash onpolicy/scripts/train/launch_stage3_diagnostic_half.sh /ABSOLUTE/SUITE /ABSOLUTE/SUITE/attempts/resume_ID/manifest.json
```
