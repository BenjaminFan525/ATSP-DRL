# 冻结 B0 → B_SHARED：本机实现与执行记录

用户已授权按研究计划实现并启动，资源限定为 GPU0 和剩余 CPU 半区。研究依据和科学门槛见[计划](../STAGE3_B_SHARED_B0_LOCAL_PLAN_20260912.md)。本文件记录实现及技术证据，不将诊断更新当作研究结果。

## 实现

- [配置](../onpolicy/config/env_stage3_b_shared_b0.yaml)：global32、8 数据 epochs、每批 PPO2、microbatch8；GPU0 UUID 固定；CPU `0–31,64–95`。
- [独立引擎](../onpolicy/runner/shared/stage3_b_shared_b0_engine.py)：严格加载 B0 的 508 张量，保留一个原生全层共享 GNN；三个角色联合训练；Ready/未用 critic 冻结；中央 team critic 与 actor 的 Adam 参数不重叠。
- 引擎区分随机自回归行为策略与原生 Hungarian 评估，保留 B0 authoritative history 和逐 agent 终止时的 hidden reset。评估恢复历史 requires_grad 配置、cuDNN allow_tf32=True 和固定 12 槽位批次，退出后恢复训练范围、温度和禁用 TF32。CUDA matmul TF32 始终关闭。
- Tune60 沿用冻结原始顺序；已完成案例继续参与批次前向及空步进，其记录停在真实完成步。中断评估重放原始完整批次并核对已有记录，防止补算时重新分组。通用环境 worker 新增显式 opt-in 的原生 seed/cursor reset，旧入口默认行为不变。
- 固定 source-relative 优势，按访问平均、时间/决策求和；微批只累积梯度，完整 global32 之后裁剪和更新。完整历史重算，TBPTT8；中央 critic 输入 detach；仅缓存输入及相邻 critic 可用的 detached 编码。
- [协议与恢复约束](../onpolicy/utils/stage3_b_shared_b0.py)：完整覆盖访问日程、来源及数据哈希、选模和风险门、checkpoint/更新记录共同提交、H/AR 评估请求恢复与去重。
- [控制器](../onpolicy/scripts/train/run_stage3_b_shared_b0.py)与 [worker](../onpolicy/scripts/train/stage3_b_shared_b0_worker.py)：基线、零更新、容量/微批及新进程恢复检查通过后才发布 admission；再从 B0 开始正式训练。独立持久 validator 与训练共用 GPU0。
- [启动器](../onpolicy/scripts/train/launch_stage3_b_shared_b0_local.sh)：systemd 用户服务，CPU affinity 与 cgroup AllowedCPUs 双重限制；MemoryHigh96G、MemoryMax128G、SwapMax0，14 天硬超时。

训练使用物理核 0–21（含 SMT 64–85），验证使用 22–29（86–93），控制使用 30–31（94–95）。已在宿主机确认 GPU0 UUID 为 `GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`；GPU1 实验使用后半 CPU，未对其进程和资源作修改。

## 已完成的技术验证

CPU 回归：**44 passed，2 skipped，1 deselected**。新增 [B0 测试](../onpolicy/envs/HKBZ/test/test_stage3_b_shared_b0.py)覆盖来源、参数归属、单卡资源约束、访问日程、正负优势的解析梯度对照、critic detach、原生终止历史、非空 Adam/RNG 恢复、epoch AR 请求补发、固定批次已完成槽位及中断重放。另覆盖 full-data 日程和冻结 Stage2 交接回归。

两项跳过为旧 full-data 入口的协调 GPU canary；取消的一项为旧四 rank 检查。本轮仅授权 GPU0，其实际准入由新的单卡 global32 microbatch8/4 检查承担。JUnit 源记录：[implementation tests](/tmp/hkbz-bshared-b0-implementation-20260912.xml)，prepare 会将其复制、绑定到正式运行目录。

GPU0 小算例检查使用 Train600 的 `case_0592`：原生与新入口的完整 Hungarian 动作及历史哈希相同，成本均为 6232 秒；随后随机完整轨迹通过概率回放并完成两次 PPO 更新。共享 GNN、飞机、普通设备、转运车均有非零梯度；两次更新后平均 KL 约 0.000342/0.000775。诊断模型未用于正式初始化。[原始记录](/tmp/hkbz_b0_probe_20260912.json)。这不是完整容量准入或学习收益结论。

## 执行与恢复

首次 r1 服务已在基线阶段停止：首批 16 例中 `case_0004` 新成本 7982/冻结 7802，`case_0010` 新成本 7330/冻结 7345，其余 14 例相同。原因排查确认新入口没有保留冻结记录的固定 12 案例批次合同和原生运行设置。原生参考入口重放包含这两例的两个 12 案例批次，24 例成本及步数均复现。r1 没有 admission 或正式 PPO 更新，其源码及结果保留；修复版本使用新 r2 目录和源码快照。

修正后的新入口 `get_actions` 重放同样 24 例，逐例成本、完成步数均与冻结 B0 完全一致：[固定批次实测](/tmp/hkbz_b0_fixed12_probe_20260912.json)。这覆盖 Tune 原始批次偏移 24/48，并包含上述两个差异案例；完整 Tune60 与 validation120 的检查继续由正式服务执行。

正式服务依次执行：原生 B0 的 Tune60/validation120/Train600 基线（780 案例）、新入口零更新复验（180 案例）、global32 微批及跨进程恢复准入、8-epoch 正式训练、候选锁定和条件触发的独立确认。

`prepare` 生成新 confirmation120，冻结源代码、依赖版本、数据内容、配置和访问日程；不评估确认集。正式训练的所有权重重新来自冻结 B0，Adam 新建，ValueNorm 只用 Train600 基线标定。确认集只允许 B0 和按固定规则锁定的唯一候选访问。

每个成功 global batch 的 checkpoint 和更新记录共同发布一个 commit。恢复命令使用最新完整 commit；不完整批次留在原 attempt，重新计算的开销保留在独立台账。已提交 epoch 的 H 和 AR 请求都会补齐；梯度诊断若中断，也按该 epoch 的原 checkpoint 补做。

启动命令为 `bash onpolicy/scripts/train/launch_stage3_b_shared_b0_local.sh ABSOLUTE_MANIFEST run`；显式恢复使用同一启动器的 `resume`。启动器记录真实 systemd unit 到 `last_service_unit.txt`。控制器状态、逐阶段 worker 状态、terminal.log 和资源采样记录均保存在运行目录。服务启动后的实时阶段以这些记录为准；只有生成 `training_admission.json` 后才允许正式 optimizer 更新。

2026-09-12 18:56:34（上海时间）已启动 r2，systemd unit 为 `hkbz-bshared-b0-stage3_b_shared_b0_local_a6000_20260912_r2_gpu0-1789210594.service`，控制器 PID 855658，初始基线 worker PID 855666。宿主机复核为 active/running，真实 CUDA 进程位于 GPU0；19 个进程及其线程均在指定 CPU 半区。启动时处于 Tune60 基线重放，尚未发布训练 admission。

18:59 实测首批 Tune 12 例已完成，逐例成本及完成步数全部与冻结记录一致，第二批正在推进。此时正式训练 commit 为 0；完整基线、零更新、global32 容量及跨进程恢复检查由同一后台服务继续执行，通过后自动进入训练。首批证据见运行目录 `engineering/live_tune_reference_check.json`。

- [运行 manifest](../result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260912_r2_gpu0/manifest.json)：SHA `08816a11f33b52dc09438816a74ee04d7bd4cd2f6688a4c942bc2e6d729d230e`，冻结 423 个源码文件，确认集 120 例成本保持未打开。
- [实时状态](../result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260912_r2_gpu0/run_status.json)、[启动实测回执](../result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260912_r2_gpu0/engineering/launch_receipt.json)、[技术验证与修复记录](../result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260912_r2_gpu0/engineering/technical_verification.json)。
- 运行目录中的 `engineering/` 保存了本轮检查脚本、原始 GPU 检查结果和启动前计划副本；`tests.xml` 为冻结的 44 项通过、2 项跳过回归报告。

2026-09-12 19:31 起，训练前评测切换为同 GPU0 上两个固定 12 例进程，新增调度适配器独立冻结，r2 原始源码和 manifest 不变。19 项相关 CPU 测试及 24 例并发零更新对照通过；切换时复用 Tune60、validation84、zero Tune24，剩余 66 个固定批次。适配器完成后自动恢复原 controller，原有 global32/micro8/4 和新进程恢复门继续强制执行。改动、基准与资源记录见[前置评测加速说明](STAGE3_B_SHARED_B0_PREFLIGHT_ACCELERATION_20260912.md)。
