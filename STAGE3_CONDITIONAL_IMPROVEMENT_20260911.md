# Stage3 条件化局部改进：H2 / F4 / soft，单 seed、全机

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-11
- Verification Status: implementation CPU/GPU checks passed; scientific results UNVERIFIED
- Version Label: stage3-conditional-improvement-v1
- 用户授权：实施上一条研究计划，完成验证后使用全部 GPU/CPU 启动；所有 GPU 共享一组异步验证进程。

## 不变项与证据边界

源 checkpoint SHA256 `41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030`。
物理环境保持 H=2、F=4、soft、bounded_frontier、capacity5、release-aware ETA、grace300s、安全裕量60s。
主历史为 stable_request_identity_v1；保留同物理环境下 C0_semantic、C0_legacy 两个分母。
旧 frozen source 全量复制，只叠加本轮七个新研究文件及旧局部引擎的可选验证完成性钩子；不替换为平行基线当前可能变化的核心源码。
共享工作区代码变化只提示；实际执行副本、输入、checkpoint 和依赖版本严格校验。
一个训练 seed=2026091001。LR 配置不是多 seed。旧 Fit/Diag/Probe/Tune 已暴露，不宣称盲测。

## P0：实施与复用准入

一个 work 队列、8 个持久 GPU worker；一个 validator 队列、2 个消费者，与 GPU0/GPU4 共卡。
先 warmup 两个 validator，再在 8 张卡上运行真实环境/非零梯度/恢复 canary，覆盖三种小头。
复用上一轮同条件 dual-baseline、teacher、IGA、历史 benchmark；四个案例双基线精确回放不一致就拒绝复用。
容量检查整个 GPU（共卡训练与验证合计≤22GiB）、服务峰值≤96GiB、主机可用≥16GiB。
不抢占已有 GPU 任务。旧产物只读，失败记录不删除、不静默重试。

## P1：WAIT 因果分解和充分拟合

旧诊断残差导入后只能用于诊断。TrainDiag32 做 full、wait-only、ranking-only，96 次新终局评估；
C0 复用已核验的32案例基线。wait 与 real-ranking 两部分之和与完整 residual 分布等价（公共 logit 常数不影响分布）。
不把最早一次动作变化解释为整条轨迹的独立因果效应。

缓存拟合并行比较原 score 头和 conditional 头，各 LR=1e-4/3e-4/1e-3，初始化一致，单案例和 Fit16 各≤2000更新。
每50次记完整固定数据指标；至少200步后才检查200步平台期。独立状态 logits 另做计算路径对照，禁止部署。
缓存特征一次准备在 learner 中连续优化，无环境仿真；更新数和仿真数分别记账。
先比较同损失/同信息，仅增加一个训练缓存优胜配置的成本加权损失对照；不从 Tune 选LR。
旧指标保留，新增有意义差值 epsilon=max(5s,.001*reference cost)。偏好成本差权重按epsilon缩放、上限50，
案例等权→状态等权→对内归一化；无明显收益状态另用 .1 的 reference KL 保留项。
必须记录有收益命中率、成本遗憾、排序准确率、角色/WAIT派遣分组、未知动作、梯度范数、clip和累计KL。
未知动作不虚构成本，按覆盖不足拒绝拟合准入；后续当前参考复采样可以重新覆盖这些动作。
拟合准入：有意义偏好单案例≥95%、联合≥80%、遗憾比≤20%、无未知预测动作。
无有效标签返回 coverage/no-signal，不当作损失实现崩溃或已学会。

## 条件化头与观测边界

冻结 GNN/GRU/飞机/原actor。64宽小头分别学习 WAIT 边际修正和真实请求条件排序。
零初始化精确保留原合法动作分布；最终统一对完整动作分布 argmax，不使用二元0.5门控。
显式处理 WAIT 禁用、强制 WAIT、只有一个真实动作与 padding。
可选 conditional_pair 在原小头不足/闭环不安全时启用四个 side-channel 特征：当前释放估计、移动ETA、
预计到达减预计需求lead、当前合法竞争请求数。时间按小时归一化并限幅到[-10,10]。
不读取未来实际 ready/completion 或 request_ready_targets，不改旧 GNN 八维 request 输入。
设备身份由 actor-critic 内实际 agent_idx 显式传到资源头；实例局部 AST 钩子只在两处 actor_head 调用增加该参数，
精确检查调用点数量，原函数其余语义不变。全局类、平行服务不受影响；不是用调用序号猜设备身份。

## P2：当前策略教师与分离准入

固定标签拟合合格后做 Fit16+Probe64 完整 greedy。要求全Fit教师潜力恢复≥50%、去掉最大赢家后收益和非负、
Probe总体不退步并通过风险门。比例是闭环统计，不是学会教师动作的比例。
最多3轮新教师，各轮≤320、合计≤960终局查询（当前16例×2位置×最多8候选，理论每轮≤240）。
第一轮用新的pair头拟合当前参考教师；后续参考与起点随已选择策略变化。参考 checkpoint 身份写入每组标签，
旧参考成本不混作新参考Q值。新鲜数据保留全部16案例，包括平局/无赢家。
选点按角色、时间段、future/ready与概率边界分层，全部与未来成本结果无关。

实现失败→停止修实现；充分拟合仍失败→记录有限预算内学习能力未解决；
拟合通过但闭环不安全→不准不安全权重预热，但从同架构零残差C0运行 T_PPO/L_RL 各480查询，之后评审；
两类门均通过→正式四方法对照。不会再把一个短拟合门直接等同于所有RL失败。

## P3：正式预热和四臂

诊断权重禁止正式初始化，使用选定架构的零残差 checkpoint。
正式预热 Pilot120 每例一个参考+三个不同决策位置各一个备选，共480终局查询；
避免二动作状态重复填充同一个备选。整个教师语料参考固定C0，集齐后成本加权缓存拟合≤2000步，LR3e-4。
该权重属于正式监督预热，不叫RL收益。单独检查Tune/Probe完整greedy非退步和风险门。
只有安全的正式预热才能初始化混合RL；排序可继续作为明确标注的监督对照。

- T_PPO：零残差C0，真正当前策略完整随机资源轨迹，飞机保持原greedy；源成本优势 .01*(C0-cost)。
- L_RANK：正式预热后，继续在当前策略状态和采样候选上做成本加权排序。
- L_RL：零残差C0，当前行为在一个资源因子有放回采样4动作，固定参考greedy后缀，局部优势 .01*(reference-cost)。
- L_RANK_RL：与L_RANK共享正式预热，切换上述局部策略梯度。

局部PG是contextual surrogate，不声称原完整MDP无偏PPO。负优势保留；普通PPO禁止枚举/强制前缀教师。
正式后续LR1e-4、2epochs、clip.2、grad norm1、mean softKL.02/hard.04；rank另记录max-stateKL并设.2硬门。
每臂独立计预算，reference与全部分支均计入；共享warm的480物理查询只做一次，各预热臂均计480。
每臂最大1920，480检查风险，960 gain≥.5%+风险，最多2候选；1440和1920均≥1%+风险才算pilot通过。
混合臂晋级时保留等预算L_RANK归因对照，对照不冒充晋级。
实用目标：greedy相对semantic和legacy均≥2%；混合臂相对warm另≥1%且优于等预算rank。
风险门：完成100%，OOD-stress/stress_joint回归≤.5%，各策略最差10%成本均值回归≤1%，>5%退步占比≤5%。

## P4：冻结确认与 IGA

启动前以新独立namespace/seed生成120确认案例，按validation profile比例，不运行成本评估。
仅在有候选通过Tune实用目标后锁定一个checkpoint，再打开新确认集；旧FinalBlind不打开。
确认双C0、候选、warm；混合臂另确认等预算rank。≥2%点估计、配对案例CI下界>0和风险门同时满足。
混合RL还要求相对warm≥1%和相对rank的案例CI下界>0。失败后不在该确认集上继续调参并冒充盲测。
候选出现后才追加同8个IGA诊断案例的C0/候选 greedy+8sample；复用既有同条件IGA180/1800秒。
8案例IGA与120案例确认分开，不比较异集原始成本，不把best8当greedy或跨seed证据。

## 调度、恢复与监控

验证按6案例分片共用一个持久队列；两个消费者可协作一个checkpoint。聚合检查完整案例集合与checkpoint一致性，
只对完整结果计算统计，不因最快分片完成而提前筛选。相同checkpoint/案例/协议的评估请求去重。
候选验证轨迹不完整时记录不准入，成本均值/CI置为不可用，不把已经流逝的部分时间当成更低的终局成本；
普通训练采集、基线与canary仍要求完整轨迹，默认行为不变。缓存拟合任务优先于尚未领取的长因果轨迹任务。
每worker7物理核/最多7环境，validator各3物理核/3环境，controller2物理核；覆盖64物理核及128逻辑核。
MemoryHigh96G/MemoryMax108G/Swap0，无旧fractional显存限制。容量不是目标，占用低时不增加无关实验。
每任务记录实际终局启动/完成查询账本，缓存优化为0仿真，失败的未完成尝试不擦除。
诊断6h、正式24h、全服务30h墙钟预算；超预算保留产物、标记未完成待评审，不叫科学失败。
无进展900秒只提示，不据此杀进程；硬超时、用户已批准的阶段预算和正常完成才结束自有进程。
状态明确记录formal_training_started，不把服务运行误写成正式四臂已训练。
checkpoint包含参数、Adam、RNG、query/case游标。prepare --resume-from 支持同代码同协议的显式新attempt，
仅恢复完整commit，保留旧未完成/失败查询；禁止原目录自动重启、跨代码静默迁移。

## 启动及产物

启动：`bash onpolicy/scripts/train/launch_stage3_conditional_improvement_all.sh stage3_conditional_improvement_all_20260911_r1`。
主要产物：manifest/status、technical_admission、causal_wait_ablation、fit_convergence、closed_loop_admission、
formal_plan/prewarm、screen_admission、candidate_lock、confirmation_result、candidate_iga_comparison、result。
开发测试结果和实际服务状态以对应开发目录和正式目录为准，本文不预先声明科学成功。

## 启动前验收记录

- 数学、旧局部流程、资源协议、数值稳定性、Gloo、共享分片验证、完整预热/不安全预热隔离：99项通过。
  显式开启 `HKBZ_STAGE3_GPU_TEST=1` 补齐GPU测试，无跳过；PyG弃用提示不影响本轮路径。
- 开发r1 GPU0真实canary：9/9终局完成，零残差与原C0动作精确一致，模型恢复误差0，
  已填充Adam下一次更新误差0，RNG精确恢复，冻结C0权重不变。
- 开发r1 GPU1：score/conditional/conditional_pair三种头均完成真实分支采集与缓存拟合smoke。
- 开发r1 GPU2：真实旧缓存conditional、LR1e-3，单案例与联合各2000更新，合计约126秒且0次仿真。
  联合固定损失0.397253→0.002326，有意义收益状态31/31命中；仅证明缓存拟合，不代表闭环/泛化/RL收益。
- 开发r2 GPU0：最终验证路径双案例、conditional_pair零残差，两例均完成，成本逐例与复用C0误差<1e-6。
- 旧局部引擎仅新增默认关闭的allow_incomplete验证选项；新策略验证时未完成轨迹不作为低成本成功。
- 开发目录：`result/hkbz_train_logs/stage3_conditional_improvement_dev_20260911_r1`、`..._r2`，全部保留。

## 2026-09-11 checkpoint 导入修复与恢复

- 原 attempt `stage3_conditional_improvement_all_20260911_r1` 于11:33退出：GPU5完成conditional canary后，
  持久worker把其架构带入旧score checkpoint导入，strict加载报键名不匹配。P0通过，正式训练未开始。
- 初始化架构改为任务局部状态，成功或异常返回均清理；无checkpoint的请求不能复用错误架构。
  有checkpoint的请求仍按其元数据切换小头并复用环境池，不增加训练/验证任务之间的进程重建。
- 历史导入验证schema/source/history/H2F4soft；旧文件未标注架构时明确使用score，重建对应小头和Adam后严格加载。
  保留全部权重、已填充Adam状态和更新计数；输出仍仅用于诊断，不作为正式RL初始化。
- 回归先复现3项原故障；修复后CPU/GPU共110项通过、无跳过。真实旧checkpoint在三种先行架构之后均可导入，
  权重及Adam逐项相等，真实缓存观测概率与旧score公式一致；测试未增加终局仿真。
  第一次GPU测试命令缺少cuBLAS环境变量被数值门拦截；匹配正式服务环境后通过，两次测试报告均保留。
- 恢复attempt：`stage3_conditional_improvement_all_20260911_r1_recovery1`。旧失败目录不修改；
  无正式训练commit可续接，故使用原C0/教师/双基线，从P0重新核验后进入P1。
  环境、H/F、soft、训练seed、学习率、筛选门、全部8GPU/64物理核和同一组2消费者异步validator均不变。
- 这是用户明确授权的bugfix新attempt，不使用要求代码完全相同的`--resume-from`跨协议搬运正式权重；
  新冻结副本和代码hash独立登记，修复审计及测试报告保留在恢复目录和`stage3_conditional_import_repair_20260911`。
