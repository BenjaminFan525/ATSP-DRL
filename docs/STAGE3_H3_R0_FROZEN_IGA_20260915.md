# H3 R0 实施与启动说明

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: implementation -> run
- Origin Date: 2026-09-15
- Version: stage3_b_shared_h3_frozen_iga_v1
- Status: implementation complete; scientific result pending

用户已确认先运行 R0，IGA 仅作冻结评测参考。本次实施的是前一研究计划中的 H3 纯 PPO 对照；R1 模仿学习和新 IGA 求解不属于这次启动范围。

## 配置与边界

- 起点默认上轮 Best6，严格加载权重，新建 Adam。H3 Validation 上若 B0 好至少 0.3% 且尾部不差，按计划的预定规则一起改用 B0；初始化选择会保存。
- H3/F4/soft；完全共享 encoder 及原角色头；训练 AR/tau=0.03、评测 H/tau=0.3、原固定 12 例评测合同。
- 沿用精确 Train240：192 IID、43 OOD-stress、5 OOD-scale。每 epoch 为 192+4×48=384 次访问。
- GPU0 UUID `GPU-744c1334-98c8-5318-e799-7ad15eea1fbf`，CPU `0-31,64-95`。单个采样/训练模型，240 个环境由 64 个无 CUDA 模型的 CPU worker 承载。
- 逻辑采样 384（同一行为策略收集 240+144），优化器 minibatch=64，PPO 两遍；8 epoch 最多 3072 次新增训练访问、96 次实际 PPO 更新。soft KL 提前停止时报告实际更新数。
- 原学习率、角色倍率、clip、TBPTT=8、4 GiB 输入缓存和 activation checkpointing 保留。MemoryHigh=102 GiB、MemoryMax=114 GiB、MemorySwapMax=0；Torch 分配器另预留 CUDA 上下文空间以维持约 6 GiB GPU 余量。
- 物理 microbatch 64/32 在首个完整周期实测；候选都从相同权重、Adam、采样后 RNG 和冻结轨迹更新，只按完整周期时间及有效性选择。只有完成全部 12 步的候选参与速度排名。选中的状态作为 epoch1 正式结果，另一个候选的计算单独记为诊断开销。
- 跨 microbatch 完整数值等价状态沿用用户豁免，不声称通过。行为概率回放、mask、有限值和同配置恢复仍需通过。

## IGA 冻结

主参考是 `stage3_matched_iga_h3f4_tune60_20260831_r1/full` 的 IGA180、IGA1800。索引校验实际案例哈希、H3 规划参数、完成性和教师文件 SHA256；启动及结束时复核文件未变。

代码没有求解阶段或求解命令。H1 Train600 联合教师不会被改名为 H3；Tune/Validation 示范不能进入训练。当前冻结 H3 IGA 覆盖 Tune60，因此：

- Validation120 用 H3 B0 和本轮初始化 S0 作配对评测及选模。
- 最终选定模型在 Tune60 对冻结 H3 IGA1800 报告差距。
- Confirmation120 保持关闭；没有其冻结 H3 IGA 参考时，不宣称通过独立的 IGA 接近目标。

## 验证和运行顺序

1. CPU 协议/恢复单测及旧 B0 回归；测试使用已有临时 pytest 环境，Torch 和训练环境一致，没有给训练环境安装新依赖。
2. GPU H3 小规模真实更新，检查共享层及三角色头发生更新；独立新进程恢复非空 Adam、重新采样和更新，比较模型、优化器、RNG 与轨迹。
3. 12 例 H2 零更新回放，核对旧 B0 成本/动作/history；H3 基线一次生成并逐批缓存。R0 只需 B0 的 Train240 成本和 ValueNorm，因此跳过仅为 R1 准备的 Best6 Train240 轨迹，共 600 例 H3 基线。
4. 自动进入首个 384 访问周期，完成 microbatch 实测并提交 epoch1，再继续正式 PPO。
5. epoch 1/2/4/6/8 在 Validation120 评测；训练和评测串行共享 GPU 模型，评测保存/恢复训练 RNG，完成后自动继续。
6. 连续两次 Validation 均值退步超过 1% 时停止；到 epoch4 若最佳开发集收益仍不足 0.3%，按 R0 的有限预算规则停止。缺少冻结 Validation IGA 时，其差距缩小比例门槛标记为不可计算。
7. 选定 checkpoint 后评测 Tune60，输出冻结 IGA 对比和明确的单 seed/开发集限制。

只在完整逻辑周期结束后提交恢复点；半周期崩溃从最近完整 checkpoint 恢复，不把已经执行但未提交的步骤伪装成已完成预算。提交链绑定模型、更新记录、基线、累计实际 Adam 步及案例访问数。原实验源码和 checkpoint 保持不变。

## 代码和产物

- `onpolicy/utils/stage3_h3_frozen.py`：独立协议、预算、冻结参考、分层配对统计。
- `onpolicy/runner/shared/stage3_h3_frozen_engine.py`：H3、单模型采样、独立优化器 minibatch 和完整周期恢复。
- `onpolicy/scripts/train/run_stage3_h3_frozen.py`：准备不可变快照，GPU 验证、基线、训练、选模和报告。
- `onpolicy/scripts/train/launch_stage3_h3_frozen.sh`：GPU0、半台 CPU 的 systemd 服务。
- `onpolicy/envs/HKBZ/test/test_stage3_h3_frozen.py`：新行为契约和预算/冻结输入回归。

运行目录包含 `manifest.json`、`frozen_iga.json`、`run_status.json`、`resource_samples.jsonl`、`proofs/`、`baselines.json`、`physical_configuration.json`、`commits/`、`validator/` 和 `final_result.json`。`run_status` 中的当前阶段不等于 PPO 已开始；正式学习以 epoch 提交及其实际 Adam 步为准。

> 2026-09-18 更新：本文引用的 Tune60 IGA 运行目录
> `result/hkbz_train_logs/stage3_matched_iga_h3f4_tune60_20260831_r1`
> 已按用户要求删除，删除清单与冻结摘要见
> `result/stage3_analysis/iga_tune60_deletion_20260918/receipt.json`。
> 之后的所有 RL/IGA 比较只使用 Validation120 参考
> `result/hkbz_train_logs/stage3_matched_iga_h3f4_validation120_20260918_r1`，
> 规则见 `STAGE3_IGA_FROZEN_20260918.md`。本文其余内容保持原样，作为当时的记录。
