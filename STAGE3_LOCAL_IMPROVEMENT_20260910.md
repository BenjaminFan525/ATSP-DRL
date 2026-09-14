# Stage3 局部策略改进：H2 / F4 / soft，单 seed、全机研究

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-10
- Verification Status: implementation under verification; live status belongs to the suite artifacts
- Version Label: stage3-local-improvement-v1
- 用户授权：按上一轮计划实现并启动，全部 GPU/CPU，共享一组异步 validator；主实验明确改成 H=2、F=4、soft reservation。

## 不变项与归因边界

原始 C0 权重 SHA256 为 `41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030`。
主物理环境固定 H2/F4/soft、bounded_frontier、request_capacity_per_plane=5、release-aware ETA、
预约 grace=300s、安全裕量=60s，其余契约继承已验证快照。旧 H3/F4/hard 的 8385.59 等均不是新分母。
同一物理契约重测 C0_legacy 与 C0_semantic，语义修正收益和训练收益分开；source.baseline 旧字段不再承载新成本。

本轮使用稳定请求身份重映射历史：从上一步观测中的请求身份寻找当前同一请求；已失效请求置 -1，
不把旧列表位置指向的新请求误作历史。飞机历史仍以环境为准，WAIT 保持 0。身份重复立即拒绝。
Padding、原始特征尺度仅审计，不同时改变旧 GNN 的全局池化或输入，以免捆绑多项表示干预。
残差新增特征只有当前可观测的等待/预计 lead、归一化坐标、lookahead/noop/presence、兼容设备数；
不使用未来实际 ready time，也不新增有序 case ID 特征。

固定一个训练 seed=2026091001，不自动多 seed、4800 扩展或 FinalBlind。TrainDiag32/Fit16/Probe64/Pilot120/Tune60
继承内容校验后的既有划分；它们有历史暴露，不宣称独立盲测或跨 seed 证据。

## 模型与学习

冻结 C0 的 GNN、GRU、原始 actor/critic。普通设备/转运设备各有独立的 64 隐宽 SiLU 残差打分器，
末层零初始化，在原始合法动作 log-prob 上加残差再归一化。相同观测版本下初始 greedy 与原生 C0 精确对齐。
飞机不训练，保持原 greedy；资源温度 .3。无额外可训练 GNN、GRU 或 Q 网络。
旧模型的循环转移和输入编码都冻结，训练只重放真实历史产生的 detached query/候选特征；这不是对可训练 GRU 做近似截断。
基础 actor 仍按真实前缀重建循环历史，强制前缀和局部分支必须精确重放。

- T_PPO：源成本优势 `0.01*(C0_semantic-cost)`；case mean → trajectory mean → decision SUM。
- L_RANK：同状态候选按真实终局成本配对排序，近似平局不产生标签；不是整条赢家轨迹 BC。
- L_RL：当前行为策略在单一自回归资源决策位置采样 4 个动作（有放回），每个动作接冻结参考 greedy 后缀，
  优势 `0.01*(reference_greedy_cost-branch_cost)`，只更新该决策。它是局部 contextual surrogate，不声称原完整 MDP 的无偏 PPO。
- L_RANK_RL：共享正式 L_RANK 预热 checkpoint，之后切换上述局部 RL。

正式 LR=1e-4，Adam eps=1e-5，epochs=2，clip=.2，global gradient norm=1，soft KL=.02/hard KL=.04。
排序 KL 比较每次更新起点，而不是永远锚定旧教师 C0。软阈值减少 epoch 时明确记账，硬阈值失败留档，不静默回滚。
负优势保留；枚举候选/强制分叉禁止进入普通 PPO；诊断权重禁止初始化正式训练。

## 阶段与门槛

P0：两个 validator 常驻后，八个 GPU 各执行真实环境 canary，检查零残差 greedy、原生动作、局部/普通 PPO 路径、
冻结参数与 checkpoint 恢复；另以明确标注的合成偏好在真实特征上检查非零排序梯度和已有 Adam 状态的下一次更新恢复。
合成标签不作为仿真结果、教师或研究指标。恢复模型 atol=2e-6；完整历史 log-prob atol=.002。
训练端无旧的分数显存上限；每 worker reserved≤21GiB、服务采样峰值≤96GiB 才准入。
随后对所有唯一 Train/Tune 案例批量重测两种 C0 历史语义，保存每案例成本及动作，不从旧 hard 成本填充。

P1：8例×4位置×最多8候选初查；R/J 的单步/四步窗口用于联合依赖诊断，不作为局部 RL 数据。
Fit16 各最多8位置×8候选建立固定教师，包括失败/退化分支和无赢家案例。
C0/B_SHARED/C_PRIVATE 在同一 Probe64、H2/F4/soft、legacy 历史、J温度.03上各做 greedy+8 sampling。
B/C 原来训练于 hard，因此该比较是固定条件下的迁移评测，不是本轮方法的训练对照。
同8个 TrainDiag 案例另测 cold IGA 180s/1800s、population20，同 H2/F4/soft；没有 warm-start，不使用其标签训练。

P2：单案例16次诊断排序更新；Fit16按4案例/批、10遍、每批2epochs做联合拟合。
要求至少8/16案例局部教师改善≥1%，全16例闭环恢复≥50%教师潜力，有改善位置 top-1命中率≥80%，
Probe64总体不退步且通过风险门。未通过则停止本路线的正式训练，保留诊断，不延长训练掩盖失败。

P3：四臂最多1920次**终局仿真评估**/臂，包括 reference 和强制分支，不等于旧版 episodes。
每逻辑批4个案例。普通 PPO 每例5条；局部 RL 每例1条参考+4条采样分支。
正式预热每例1参考+3候选，共120例×4=480次，L_RANK和L_RANK_RL共享这次物理采集；各臂独立预算仍计480次。
预热后排序-only也使用与局部RL相同的采样候选协议，仅目标不同，随当前策略访问状态重新采集。
每完整批保存不可覆盖 commit，含参数、Adam、CPU/CUDA/Python/NumPy RNG、案例游标、查询次数和验证请求。
正式新鲜组严格匹配父 checkpoint 身份；不接收半批状态或诊断权重。

960次初筛 gain≥.5%，最多2个候选；1440/1920两端点均 gain≥1%且风险门通过才称 pilot 通过。
若混合臂晋级，必要时单独保留等预算 L_RANK 归因对照；不把未过门的对照称为晋级。
最终实用目标：greedy相对新C0≥2%；混合臂还须相对预热≥1%、优于等预算排序-only。
所有评估同时提供相对同一 H2/F4/soft 下原观测 C0 的 legacy_summary；original_checkpoint_target_passed
另外要求相对该 C0 也≥2%，避免把语义修正造成的退步掩盖在重新设定的分母中。
风险门：100%完成，OOD-stress/stress_joint回归≤.5%，各策略最差10%成本均值回归≤1%，>5%退化案例比例≤5%。
报告配对案例 bootstrap，注明单训练 seed 条件下的案例不确定性。

## 资源、监控与启动

八个持久 GPU worker 共用任务队列，采集任务在空闲 GPU 上领取，小型 learner 更新在完整案例批收齐后提交；
没有八套独立策略，也没有异步 local SGD。每 worker 7物理核，最多7个环境进程。
仅一组异步验证池：一个 durable validator 队列、两个消费者，分别与 GPU0/GPU4 共卡，各3物理核，controller2核。
64物理核/128逻辑核全部纳入资源计划，SMT兄弟核不拆分。阶段依赖允许空闲，不为GPU占用率新增无关实验。
systemd MemoryHigh96G/MemoryMax108G/Swap0，服务 RuntimeMax10天；worker任务12小时、验证6小时硬超时。
10秒采样整服务内存、进程和阶段，900秒无进度只告警；所有请求失败留档，无自动重试。
不抢占外部任务，启动前等待外部GPU进程退出；用户共享源码变化仅提示，实际执行快照严格校验。

启动：`bash onpolicy/scripts/train/launch_stage3_local_improvement_all.sh stage3_local_improvement_all_20260910_r1`。
默认根目录 `result/hkbz_train_logs/stage3_local_improvement_all_20260910_r1`。
看 `status.json`、`technical_admission.json`、`fit_admission.json`、`training_admission.json`、
`screen_admission.json`、`result.json` 和 `resource_status.json`；不能把服务启动当成四臂长训练已启动。

本控制器拒绝原目录自动重启；每个完整 checkpoint 可在同协议下显式恢复 learner/collector状态。
中断后应使用保留旧失败记录的新 attempt 编排，不把 pending/running 任务自动重放成未发生过的计算。

## 开发验收

首轮 CPU/回归测试64通过、1跳过；扩展到原三臂/Gloo/编排回归后98通过、1跳过。
Gloo 在沙箱内因回环网络隔离失败，沙箱外复核通过，未修改测试或数学阈值。
GPU开发 r1 实测普通PPO非零更新，冻结权重不变；局部随机分支为平局，因此不把零梯度恢复当成有效学习证据。
GPU开发 r2 加入原生greedy对齐与明确标注的合成梯度/已填充Adam恢复测试；正式P0仍逐GPU重新验收最新快照。
独立 smoke 测试已验证真实 plane/job/site 请求身份注入、双案例批量基线、greedy验证重复成本一致。
开发产物位于 `result/hkbz_train_logs/stage3_local_improvement_dev_20260910_r*` 与 `stage3_local_improvement_smoke_20260910_r1`。
