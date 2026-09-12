## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan -> implementation -> run (user authorized 2026-09-12)
- Origin Date: 2026-09-12
- Verification Status: PLANNED; runtime receipts, not this document, establish admission/results
- Version Label: stage3_full_data_shared_private_v1

## 1. 研究问题与边界

检验在完整 Stage1 训练池、充分数据覆盖及同一纯 RL 配方下，全共享与全分离策略能否稳定改善原始 C0 的单次 greedy 成本。主目标为相对同环境、同历史语义的 C0 降低至少 2%；架构的额外收益、尾部风险与训练效率分别报告。单 seed，不新增监督学习、教师拟合门槛、额外算法臂或自动多 seed。C0 是已有 Stage2 checkpoint，纯 RL 指本阶段没有监督目标，并非从随机参数开始。

历史 H3/F4/hard 同预算试验中，B_SHARED/C_PRIVATE 的 Tune60 收益分别为 0.6027%/0.7271%，均未通过完整科学门槛；C_PRIVATE 训练约 21.41h，B_SHARED 约 11.47h。它们不是本轮 H2/F4/soft 的基线或耗时承诺。更近期的条件头纯 RL 480-query 结果也尚未达到 2%。本轮不能将跨轮改进全部归因于数据扩充或网络分离。

## 2. 数据与采样合同

- 完整训练池：`onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train`，600 个唯一案例：iid 480、ood_stress 108、ood_scale 12。
- 沿用 Stage1 的完整覆盖分布配额 50%/45%/5%：每个数据 epoch 为 960 次案例访问（480/432/48），确保全部 600 案例至少访问一次；每次访问采一条完整新鲜随机轨迹。重复案例使用不同 visit ID 和 RNG seed。
- 主预算：8 个数据 epoch，即每臂 7,680 条训练轨迹；不是 8 次 PPO 更新，也不是 7,680 个不同案例。单 seed 2026091201。
- 每个 global batch 固定 32 次访问，四卡各 8 条；每批 PPO 2 轮，合计每数据 epoch 30 批/60 次 actor 更新，最大 240 批/480 次 actor 更新。按访问平均，不能按 case ID 去重抵消 Stage1 的分布重采样。若容量不足，停止准入并保留结果；后续加速修订只考虑等价微批累积，不静默改变 global batch 或更新数。
- 每批固定行为策略版本；不混入旧策略轨迹、不进行 best-of-k 筛选、不把额外诊断查询算作训练轨迹。
- 原 TrainDiag32/Fit16/Probe64/Pilot120 均纳入训练池；只能作训练诊断，不能宣称留出验证。
- 主验证：Stage1 v2 validation120；历史对照/压力监测：Stage3 Tune60。逐物理案例内容核对，二者与 train600 无交集。验证不进入梯度。
- 已生成、成本未打开的 confirmation120 只在两臂 checkpoint 选择规则执行并锁定后评估一次；先核对暴露台账。历史 Stage1 test60、Stage2 Finalblind 不作为本轮新的盲测。

## 3. 两臂、初始化与 RL 配方

| 臂 | 图编码器 | GRU/策略头 | 训练 GPU |
| --- | --- | --- | --- |
| B_SHARED | 一套全层可训练 GNN | 保留原有角色专属模块 | 0、1、2、3 |
| C_PRIVATE | 三套全层独立、全层可训练 GNN | 同样的角色专属模块 | 4、5、6、7 |

这里的“全共享”指可共享的图策略主干；异构动作角色的输出头不强行合并。全分离按飞机/普通设备/转运车三角色分离，不是每台设备各建模型。三套参数不得 alias；critic 沿用两臂一致的原有角色价值模块。环境联合解码、资源冲突约束和预约仍然共享。

两臂从同一原始 C0 精确复制，重置优化器，不使用历史 B/C 最优权重或诊断更新权重。固定 H=2、F=4、soft reservation、stable_request_identity_v1；训练、回放、验证、C0 参考成本必须使用同一合同，不能引用旧 H3/hard 成本。

纯 source-relative PPO：A=0.01*(C0_case_cost-trajectory_cost)，gamma=1，时间/角色求和、访问平均；保留正负样本，不增加 BC/rank/IGA 势函数。三个角色联合训练、采样温度均为 0.03，最终 greedy 单次解码。actor 基础 LR=1e-5，沿用角色倍率 shared=1/plane=0.25/device=1/transporter=0.5；critic LR=1e-4。PPO clip=0.2，PPO epochs=2，TBPTT=8，跨卡汇总后全局 grad norm clip=1，soft KL=0.02/hard KL=0.04。全局归约和 Adam 顺序不变。没有监督拟合准入；技术正确性失败不能当作算法负结果。

## 4. 全分离的执行优化

优先实现对 B/C 都适用的有界图输入缓存、同一步 actor -> critic 的 detached 编码复用；全分离的三套 actor GNN 仍各自保留梯度。跨 optimizer step 只能复用不可变图输入，绝不缓存 trainable 编码结果。保留已经验证的统计批量传输、关闭 activation recomputation 和稳定 max-pool 数值路径。

同一全局批次记录 rollout / update / replay / checkpoint / distributed waiting 耗时。测试普通图、长轨迹、资源压力及大图案例，测全服务 RAM/单卡总显存，不以 GPU 利用率或张量微基准代替端到端吞吐。固定轨迹的旧/新执行路径必须通过 loss、梯度、参数、Adam、RNG、动作和完整历史回放检查。参考门槛沿用模型 atol=2e-6、Adam atol=2e-6/rtol=2e-5、full-history logp atol=0.002；同配置恢复必须可重复。

工程目标：在同一新合同/同一训练批次下，C_PRIVATE 完整批次时间降低至少 15%，争取 30%；未实测不宣称达成。三编码器参数批量化/按实际需要计算、设备端梯度归约可作为后续候选，不能以共享权重、跳过 GRU 历史、减少 PPO 更新或降低精度冒充等价加速。本轮不默认 AMP/TF32/compile/vmap。

## 5. 准入、资源、监控与恢复

1. 固化源码副本、配置/数据/C0 哈希，核对迁移后的 C0 行为、历史映射和采样全覆盖。
2. 两臂固定训练案例执行真实 PPO、单/多卡归约与 checkpoint 恢复检查，诊断权重禁止初始化正式训练。
3. 八个训练 rank 与两位 validator 同时驻留做容量检查；通过后发布不可变 admission，正式训练重新从 C0 开始。
4. 每臂一个逻辑策略、4 个同步 rank。每 rank 7 个物理核；两位 validator 各 3 核，与 GPU0/GPU4 共卡；控制器 2 核。SMT 兄弟核不拆分。共用一个持久化队列、两个消费者，不按臂/按 GPU 另建验证池。
5. 训练不施加旧分数显存限制；容量按整卡总量留出验证器余量。整套服务 MemoryHigh=96G、MemoryMax=108G、SwapMax=0，为约 128G 主机保留余量。启动前核对外部作业，不抢占。
6. 每个完整 global batch 保存模型/双 Adam/ValueNorm/RNG/访问游标，各 rank 成功后才发布 commit。每 epoch 提交同预算验证请求，checkpoint 不可变，积压有界且反压；启动/恢复不得漏评或重算已完成请求。
7. 每 10 秒资源采样、进程存活与进度监控；平台、低利用率和慢进度为提示，不因 480/960 轨迹未达收益阈值停止全量训练。OOM/非法状态/数值或 KL 合同失败停止推进并保留产物，不静默重试。硬超时为 14 天，控制器/子任务均有边界。
8. 工作区源码变化只提示，不阻止并行基线实验；运行中的冻结源码和数据合同严格保护。不修改历史目录。

## 6. 评估与决策

每个数据 epoch 均评估 validation120 与 Tune60 的 greedy 平均成本、配对逐案例收益、95% profile 分层 bootstrap CI、分布/profile 风险、最差 10% 均值及 bad>5% 比例。以 case 为统计单位，不把重复轨迹当独立样本。CI 仅反映单训练 seed 下的案例不确定性。

主研究目标：相同合同下 greedy 平均成本比 C0 降低 >=2%，并在后两个完整 epoch 保持正收益；最终锁定 checkpoint 的 confirmation120 也须 >=2% 且 CI 下界>0。风险门：completion=100%，无 cycle/deadlock；ood_stress/stress_joint 回归<=0.5%，尾部回归<=1%，退化>5%的案例比例<=5%。validation120 合格 checkpoint 中按最低平均成本锁定各臂候选；无合格候选则保留 Last 并标记未通过，不宣称成功。

架构结论单独计算 C_PRIVATE 相对 B_SHARED 的配对收益；把 >=0.5% 且 CI 下界>0 作为有实用价值的优势目标，同时报告相同轨迹/更新预算与相同 GPU 小时的结果。若只有单臂因故提前结束，主架构比较限于共同完成端点。全部 checkpoint 和负结果保留。

不承诺全量数据或更大网络必然达到 2%。若两臂完整训练仍接近 C0，优先复盘 credit assignment、探索收益向 greedy 概率排序的转化、角色梯度冲突与更新幅度，而非继续扩大网络。IGA 只做相同案例/环境/解码预算下的参考，不把 best-of-8 的搜索收益算作 RL 的 greedy 收益。

## 7. 交付与 ETA

交付：本计划、独立 opt-in 两臂控制器/worker/全量 sampler、缓存优化、CPU/GPU 验证记录、不可变 manifest/admission、共享验证队列、逐批训练与逐案例评测记录。先实施并检查，再启动新 run，不覆写历史产物。

每臂最多 240 个 global batch。正式 ETA 用准入实测 batch 时间和队列耗时估计：max(两臂剩余批次*各自近期耗时)+验证尾部。历史 C_PRIVATE 每批中位约 32.5 分钟，只能提示未经优化可能为多天量级，不能直接当作新合同预测。
