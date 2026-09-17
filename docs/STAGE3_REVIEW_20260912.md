Stage3 历史训练与当前工作区回顾（2026-09-12）
================================================

现有证据支持：Stage3 曾出现约 0.5%–0.8% 的单轮改善，但尚不能确认已经实现稳定的 ≥2% 单次 greedy 成本下降。当前最新研究代码已转向全量 Train600、共享/角色独立 GNN 的两臂 source-relative PPO。代码合入、技术检查通过、某个 checkpoint 得分更好、完整科学门槛通过，是不同的状态。

本次核对了本机历史逐 epoch 评估 JSON、selection/run_status、研究文档、当前训练入口与公共引擎；重新计算了 C0/B0 权重哈希，检查了数据数量，并运行了 37 项 CPU 合同测试。本次未启动训练，也未修改训练实现。

证据范围与初始化
----------------

本机保留了主要在 2026-08-23 至 2026-09-01 期间运行的 Stage3 原始产物。2026-09-03 至 09-12 的较新工作主要以研究文档和导入代码存在；对应的诊断、表示、full-policy、conditional、full-data 正式运行目录不在当前 `result/hkbz_train_logs` 下。因此较新数值只能标注为文档记载，不能声称本次核验了其逐案例结果或服务器当前进度。

本次重新计算 SHA256，以下两个文件都存在且与记录匹配：

| 起点 | 身份 | SHA256 |
|---|---|---|
| 历史 C0 | Stage2 adaptive-trust T2、Epoch3，9 月 Stage3 研究使用的起点 | `41cfc782c1ff908bb527c0219b75b73a1421d48a2dafa02d449015abbe5ab030` |
| 冻结 B0 | 2026-09-12 Stage2 发布包中的 BC 起点，H2/F4 + Hungarian | `b7740b2b688c83dc7fa65bed6381243cdf2f6675f4b96808ff05a41d4e44e7e8` |

历史 C0 位于 `onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage2_adaptive_trust_20260830_r4_wave1_T2_backtrack_adaptive_soft_bc_seed1/run1/models/checkpoint_Epoch3.pt`；冻结 B0 位于 `artifacts/stage2_frozen/20260912/checkpoints/b0.pt`。早期 Stage3 还用过 B2、T0、F3 PrePPO 等其他来源，不能统一称为上述 C0。

`--stage2_frozen_manifest` 是从冻结 B0 初始化新 Stage3 的入口，不能代替历史 C0，也不能与 `--checkpoint_dir` 同用。详见 [交接说明](STAGE2_STAGE3_INTEGRATION.md) 与 [Stage2 冻结说明](../STAGE2_FROZEN.md)。

可以从本机原始结果核对的历史
--------------------------

下表成本单位为秒，越低越好。复合 selection score 单独标注；不把不同环境、起点下的绝对成本串成同一条学习曲线。

| 时间/实验 | 实际观察 | 解释 |
|---|---|---|
| 08-24 role-credit B0–B3 | B1 最好复合分数 9377.3167，共同 Pre-PPO 为 9365.2958，退步 0.1284%；B1 raw Cmax 为 8795.9222，Pre-PPO 为 8783.3944 | 原始 selection 确实选择 B1，并注明按当时要求选硬门合格者；这不代表训练优于起点。后续恢复综述将该线列为不晋级 |
| 08-24 encoder E0–E3 | E2 raw Cmax 最低到 8716.4444，Pre-PPO 为 8783.3944；但 E2 最后两轮复合均值和 tail 门未过。四组 `admitted=false` | 有短暂改善，未形成符合本轮门槛的稳定胜者；此时 E2 是渐进解冻方案，不是 09-08 的角色独立尾部 E2 |
| 08-25 critical-path C0–C3 | C3 被选中，epoch4 raw Cmax 8462.2889；它自己的 Pre-PPO 是 8448.8389。C2 最好为 8429.8722，但 empty replay 比例超过 1% 硬门 | C0/C1 使用 H1/F2/soft，C2/C3 使用 H3/F4/soft；跨臂数百秒差距不能全算为 PPO 收益，C3 相对自身起点实际退步 |
| 08-26 counterfactual HAPPO N0–N3 | N0 完成，最好 actor epoch4 为 8456.3611，高于自身 Pre-PPO 8448.8389；N1–N3 缺少完整 actor epoch 结果；无方法 admitted | N0 未改善，其他臂包含未完成问题，不能把整个套件写成四种方法充分训练后均失败 |
| 08-26 PPO-gain P0–P3 | P0/P2 已有 actor 评估退步；P3 epoch1 8446.7056，epoch3 8545.8778，共同起点 8448.8389；P1 仅有与起点相同的 epoch1 | 未看到稳定的成本下降；不能把有限/校准期记录当成完整方法比较 |
| 08-31 latest-Stage2 pure RL N0 | 已完成 5 epochs。Pre-PPO 8408.0944 → epoch3 8339.6389（改善 0.8142%）→ epoch5 8414.3833（退步 0.0748%） | 这是直接的“最好 checkpoint 有改善、训练终点未保持改善”证据，来源为 T0，不能当成后续 T2 的结果 |
| 08-31 gradient-conflict G0–G3 | H3/F4/hard，起点 8388.1333；actor epoch2/3 均未超过起点。四臂 `interrupted/SIGTERM` | 不支持选出冲突处理胜者，也不足以认定完整方法负结果 |
| 09-01 gradient-conflict、F3 PrePPO 起点 | H3/F4/soft，起点 8406.1556；四组 epoch2 约 8361.72，epoch3 分别约 8395.98/8401.00/8424.78/8409.01；均 SIGTERM | 与上一行同时改变了来源和合同，不能按 seed1/seed2 合并为配对多种子实验 |

原始选择记录：

- [role-credit selection](../result/hkbz_train_logs/stage3_role_credit_wave1_20260824_r2/analysis/wave1_method_selection.json)
- [encoder selection](../result/hkbz_train_logs/stage3_encoder_wave1_20260824_r3/analysis/encoder_wave1_selection.json)
- [critical-path selection](../result/hkbz_train_logs/stage3_critical_path_wave1_20260825_r1/analysis/critical_wave1_selection.json)
- [counterfactual HAPPO selection](../result/hkbz_train_logs/stage3_counterfactual_happo_wave1_20260826_r1/analysis/credit_happo_wave1_selection.json)
- [pure RL N0 完成记录](../onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage3_latest_stage2_pure_rl_20260831_r2_N0_seed1/run1/run_status.json)
- [pure RL N0 逐 epoch 评估目录](../onpolicy/scripts/results/HKBZ/simple/gnn_mappo/stage3_latest_stage2_pure_rl_20260831_r2_N0_seed1/run1/evaluations)

9 月研究方向的演变及证据边界
--------------------------

| 日期 | 方向/已有文档信息 | 本次可以确认的范围 |
|---|---|---|
| 09-03 | 重新从 T2 对比 sum、norm-balance、PCGrad、CAGrad | [计划](../STAGE3_NEXT_ROUND_20260903.md)记录 canary 和启动，不提供本机可复核的最终 selection |
| 09-05 | 诊断 greedy/sampling 差异、hard-IGA、反事实分叉、小集拟合，再研究 value/source/LOO 优势 | [协议](../STAGE3_DIAGNOSTIC_NEXT_ROUND_20260905.md)记录 T2 在当时合同下 Tune60 C0 为 8385.5889；这不是后续 H2/soft 分母 |
| 09-06 | 采样温度与随机角色审计；R 资源探索和 J 全角色低温探索；PPO/可选精英模仿 | [采样审计](../STAGE3_SAMPLING_AUDIT_20260906.md)、[局部探索](../STAGE3_LOCAL_EXPLORATION_20260906.md)主要是设计和执行记录，不能仅凭文档名断言各臂最终通过 |
| 09-08 | E0 冻结、E1 共享尾部、E2 三角色独立尾部、E3 参数量匹配残差，乘以 T0/T1 组批策略 | [表示研究](../STAGE3_REPRESENTATION_20260908.md)与[容量续跑](../STAGE3_CAPACITY_CONTINUATION_20260908.md)记录真实边界捕获及 B32/T16 OOM；中途 batch 和切换点变化限制因子归因 |
| 09-09 | A_E2、B_SHARED 全层共享、C_PRIVATE 三套全层独立编码器 | [09-12 文档](../STAGE3_FULL_DATA_20260912.md)回顾历史 H3/F4/hard 下 B/C Tune60 收益为 0.6027%/0.7271%，未过完整科学门；训练耗时约 11.47h/21.41h。本机缺这两条运行的逐案例和逐批证据 |
| 09-10 | 转到 H2/F4/soft、stable_request_identity_v1，冻结主干，训练资源残差头；完整 PPO 与局部 rank/PG 对照 | [局部改进协议](../STAGE3_LOCAL_IMPROVEMENT_20260910.md)；须区分环境、历史语义修正和训练效果 |
| 09-11 | WAIT 边际与真实请求条件排序小头、充分缓存拟合、闭环准入与纯 RL 分支 | [条件头记录](../STAGE3_CONDITIONAL_IMPROVEMENT_20260911.md)中的 loss 0.397253→0.002326、31/31 命中是固定缓存拟合结果；不能等同闭环或 RL 收益。09-12 文档称近期纯 RL 480-query 结果仍未达 2% |
| 09-12 | 全量 Train600 上比较 B_SHARED/C_PRIVATE | 代码和[方案](../STAGE3_FULL_DATA_20260912.md)齐备；[导入说明](STAGE2_STAGE3_INTEGRATION.md)提到服务器 recovery1 服务，但本机没有其运行目录，无法核实当前 episode、学习曲线或最终结果 |

较新 full-policy 记录仅支持全分离相对全共享多出约 0.1244 个百分点的已记载收益，且耗时约 1.87 倍；缺少原始配对结果时不能判断这个差距是否可靠。

当前训练代码如何连接
--------------------

工作区存在两条入口。通用 `train_hkbz.py → HKBZRunner → GNN_MAPPOPolicy/gnn_mappo` 保留原联合 MAPPO、credit、阶段交接和冻结 B0 初始化；最新研究使用 `run_stage3_full_data.py → stage3_full_data_worker.py → FullDataEngine → RepresentationEngine → ResearchEngine`，复用策略和环境，但有独立采样、PPO、验证、来源及恢复合同。只检查通用 MAPPO trainer 无法完整解释最新训练。

| 文件 | 责任 |
|---|---|
| [run_stage3_full_data.py](../onpolicy/scripts/train/run_stage3_full_data.py) | 来源与数据划分、源码快照、C0 基线、单/四卡一致性、容量准入、两臂控制、候选锁定与 confirmation |
| [stage3_full_data_worker.py](../onpolicy/scripts/train/stage3_full_data_worker.py) | 每 rank 采样/更新、完整 commit、epoch 验证提交、共享 validator 消费 |
| [stage3_full_data.py](../onpolicy/utils/stage3_full_data.py) | 全量访问日程、合同校验和评估请求身份 |
| [stage3_research_engine.py](../onpolicy/runner/shared/stage3_research_engine.py) | 完整环境轨迹、循环历史回放、PPO 损失、KL、Adam/ValueNorm/RNG 保存恢复 |
| [stage3_representation_engine.py](../onpolicy/runner/shared/stage3_representation_engine.py) | 安装表示变体、优化器参数归属、缓存作用域 |
| [stage3_encoder.py](../onpolicy/algorithms/utils/stage3_encoder.py) | E0–E3 尾部干预，以及 F_SHARED/F_PRIVATE 全层编码器 |
| [stage3_distributed.py](../onpolicy/utils/stage3_distributed.py) | CPU/Gloo 梯度归约，全局裁剪之前同步，以及 rank 模型/Adam/ValueNorm 一致性 |
| [stage3_performance.py](../onpolicy/utils/stage3_performance.py)、[stage3_numerics.py](../onpolicy/utils/stage3_numerics.py) | 有界图输入缓存、detached actor→critic 编码复用、批量统计传输、确定性 max-pool 路径 |

当前 `ac.yaml` 的原生模型是 64 维表示、4 层/4 头异构图编码器，包含工序、场地、设备和请求节点，再接角色 GRU/actor 与 critic。B_SHARED 训练一套完整 GNN；C_PRIVATE 从同一 C0 深复制三套完整 GNN，分别服务飞机/普通设备/转运车。异构角色保留各自输出头，共用环境约束和联合自回归解码。独立编码器参数不能 alias，actor/critic Adam 不能重复拥有同一参数。历史运行实际网络配置仍以其冻结快照为准。

最新 full-data 配方：

| 项目 | 实现值 |
|---|---|
| 环境/历史 | H2/F4/soft、`stable_request_identity_v1` |
| 起点 | 同一历史 C0；重建优化器，不用历史 B/C 最好权重或诊断权重 |
| 数据 | 本机实数 train600=480 IID+108 stress+12 scale；validation120=60/54/6；Tune60=30/27/3 |
| epoch | 960 次访问=480/432/48；每个唯一训练案例至少访问一次；stress/scale 每 epoch 各访问四次 |
| 最大预算 | 每臂 8 epochs=7680 条新鲜完整轨迹，240 个 global batches，正常通过 KL 合同则 480 次 actor 更新 |
| 并行 | 每臂一个逻辑策略、4 ranks；global batch32，每 rank8 条；两臂共用两个 validator |
| actor 优势 | `0.01 * (C0_case_cost - trajectory_cost)`；保留正负样本；按访问平均，时间/角色求和 |
| 优化 | actor LR=1e-5，shared/plane/device/transporter 倍率=1/.25/1/.5；critic LR=1e-4；PPO2 epochs；clip=.2；TBPTT8；同步后 grad norm clip1 |
| KL | soft .02、hard .04；若跳过更新或提前停止造成两轮 PPO 不完整，worker 停止并保留产物 |
| 缓存 | 全层可训练编码器只缓存图输入，不跨更新缓存训练编码结果；同一步 critic 可复用 detached 编码 |
| checkpoint | 每个完整 batch 保存所有 rank 模型、双 Adam、ValueNorm、RNG、访问游标；rank 成功后发布 commit |
| 评估 | 每 epoch greedy validation120+Tune60；风险合格者按 validation 最低均值锁定；随后 confirmation120 |
| 成功条件 | 选中模型 validation 和 confirmation 均至少改善2%，confirmation 配对案例 CI 下界>0，风险门通过，最后两个 epoch validation 正收益；仅为单训练 seed 证据 |

当前需要注意的实现缺口
----------------------

1. **历史运行依赖尚未形成独立可迁移入口。** `ResearchEngine.policy_args()` 第27行直接读取固定 `BASE_MANIFEST`。本机缺 `stage3_source_relative_rl_4090_20260904_r1/commands/wave1_g0_U0K0R0.json`，本次调用 `policy_args()` 已实际复现 `FileNotFoundError`；相关 source baseline/eval 也缺失。C0 权重本身存在且哈希正确，补一个 checkpoint 路径仍不够。
2. **启动器仍绑定服务器路径和八卡拓扑。** `launch_stage3_full_data_all.sh` 第4–6行固定 `/data/fanyx/...` 和 conditional recovery1 prior；本机该根目录和 prior 均不存在。`prepare()` 明确要求 GPU0–7。通用 `env_joint_finetune.yaml` 默认 H1/F2/soft，通用研究环境构造默认 H3/F4/hard，full-data 依靠显式 overrides 才得到 H2/F4/soft。因此不能把改路径或直接启动 YAML 当成等价复现。
3. **完整训练中断后的控制器恢复尚未接通。** worker 支持 `--resume-commit`；但 full-data `run()` 第287行总是进入 `to7680`，没有传入已有 commit。`prepare-recovery` 仅允许训练开始前的 NVML 中断，并拒绝已有 train/admission。底层状态可恢复不等于正式 full-data 套件已有通用恢复入口；应在跨天训练续接前补齐控制器、队列和已完成验证的恢复流程。
4. **最终报告的架构/效率判定尚未覆盖方案全部承诺。** `finish_study()` 输出两臂 confirmation 配对结果及按 epoch 的等轨迹预算比较；未见相同 GPU 小时的比较，也未应用 manifest 中 `.005` 的 private 增益目标生成架构达标结论。单臂故障时 `run_group()` 抛错，当前流程也不会自动生成双方共同完成端点的部分结果报告。已有逐批耗时和 checkpoint 可支持后续补做分析。

以上是本机复现边界和静态实现缺口，不证明服务器现有运行已触发这些问题，也不据此断言 PPO 数学实现错误。历史代码中的角色梯度冲突、更新幅度、探索赢家能否改变 greedy 排序，仍需匹配运行记录进一步定位。扩大网络或数据是否有效不能预先下结论。

验证记录
--------

本次运行以下四个文件的 CPU 合同检查，结果为 **37 passed，5 deselected**，耗时约7.4秒：`test_stage3_full_data.py`、`test_stage3_full_data_restart.py`、`test_stage3_full_policy.py`、`test_stage2_stage3_integration.py`。覆盖全量配额与重复访问权重、共享/独立编码器功能保持与底层梯度、缓存梯度及失效、四 rank 调度、恢复身份约束、冻结 Stage2 opt-in 接口等。

明确未运行两项参数化 Gloo 同步测试、一个四 rank Gloo 测试、两个真实 GPU canary；本次不是 GPU 准入，也不满足 `seal-tests` 的全部要求。当前沙箱中 `nvidia-smi` 无法与驱动通信，不能据此判断宿主机或远端训练服务是否运行。

原始 JUnit：[hkbz-stage3-review-20260912.xml](/tmp/hkbz-stage3-review-20260912.xml)。

此前[09-12 选择性合入报告](../result/hkbz_merge_audits/stage3-main-20260912.ucudgk1a/REPORT.md)另记录343项 Stage3 测试通过、9项历史产物依赖失败、13项条件跳过，以及118项 Stage1/Stage2 回归通过；这些是此前审计结果，不计作本次重新运行。

当前 HEAD 仍是 `9330e2f`，许多 Stage3 文件为未跟踪文件，共享文件也有未提交修改；单凭 Git HEAD 不能重建本次审阅的工作区。正式运行的来源应绑定 manifest、源码快照和文件哈希。后续最先需要补齐较新运行的 `manifest/status/training_admission/result`、逐批更新与逐案例验证，才能判断全量训练实际改善、退步或停在哪一阶段。
