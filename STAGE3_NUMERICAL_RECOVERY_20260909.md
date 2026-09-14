## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-09
- Verification Status: 固定输入根因对照已验证；完整 PPO 准入由恢复流程重新执行
- Version Label: stage3_full_policy_numerical_recovery_v1
- Data: internal; no upload

## 故障与定位

父实验 `stage3_full_policy_all_20260909_r1` 在 13:34 因 B_SHARED 三卡恢复一致性失败退出：两次从相同起点执行 PPO 后模型最大差异 4.172325134277344e-6，超过原定 2e-6。三个 rank 同次更新后状态一致；失败发生在重复更新比较，不是 GPU 显存不足。正式 PPO 尚未开始，0 episodes。

检查使用父实验未修改的冻结源码、B_SHARED 原检查点和相同的三个真实初始观察图，固定模型与探针目标，连续六次反向传播、**不做任何优化器更新**。这些是数值诊断，不是实验效果指标。

| 诊断 | 检查点恢复 | 重复梯度最大差异 | 结论 |
| --- | --- | --- | --- |
| 原执行设置 | 模型/Adam/ValueNorm 精确一致 | 6.103515625e-5 | 即使不恢复、不更新权重也会出现差异 |
| PyTorch 确定性＋cuBLAS 配置，同一 GPU | 精确一致 | 9.1552734375e-5 | 原生确定性开关不足以覆盖扩展路径 |
| 确定性＋诊断中禁用 torch-scatter 路径 | 精确一致 | 0 | 差异位于扩展聚合路径 |
| 确定性＋仅替换图最大池化为稳定首次 argmax | 精确一致 | 0 | 六次梯度哈希完全相同，扩展库仍保留 |

诊断原始结果位于父实验 `diagnosis/{legacy_gradients,deterministic_same_gpu,deterministic_native,deterministic_stable_pool}/result.json`。未通过的中间对照完整保留。

PyG 2.7.0 的 GPU 可训练 max 聚合会调用 torch-scatter 扩展；原生确定性标志不足以约束所有第三方内核。PyTorch 也明确区分随机种子与算子确定性，并要求 CUDA/cuBLAS 的配套设置。[PyTorch 2.6 可复现性说明](https://docs.pytorch.org/docs/2.6/notes/randomness.html)

## 修复范围

1. 仅本轮 full-policy worker 启用 `torch.use_deterministic_algorithms(True, warn_only=False)`、`:4096:8` cuBLAS 工作区配置、cuDNN 确定性并关闭 TF32。
2. 图最大池化保持相同前向最大值；并列最大值的梯度固定给最早节点，与逐图 `torch.max(dim=0)` 一致。不使用“平均分配所有并列梯度”来代替原来的单节点 max 子梯度。
3. 通用 GNN 只增加一个默认关闭的 readout hook；不改基线实验的默认行为，不卸载/改写系统库，不全局禁用 torch-scatter。
4. 数值配方写入 manifest、worker 状态和 checkpoint，恢复拒绝配方不一致；恢复检查另存模型、双 Adam、ValueNorm/更新计数与四类 RNG 的具体比较及最大差异参数。
5. 原有 2e-6 模型容差、Adam 容差、KL 与科学准入门槛保持不变。三臂、单 seed、数据、source-relative 优势、学习率、batch 候选、更新次数和资源配置保持不变。

## 恢复边界

采用新目录 `stage3_full_policy_all_20260909_r1_recovery1`。父实验日志、所有检查点、成功和失败的验证结果及源码快照不覆盖、不删除。父实验没有正式训练 checkpoint，因此恢复重新从原始 C0 执行准入，绝不拿诊断更新后的权重续训。

数值执行路径已变化，单卡参考、多卡完整 PPO/恢复、三臂 C0 Tune60、正常/密图容量检查全部重新执行。通过后控制器自动开始 960 episodes 初筛，之后仍按原门槛决定 1440/1920 扩展。所有 GPU 继续共享一个持久化队列、两个异步 validator。

恢复命令：

```bash
bash onpolicy/scripts/train/launch_stage3_full_policy_all.sh stage3_full_policy_all_20260909_r1_recovery1 /data/fanyx/HKBZ-environment/result/hkbz_train_logs/stage3_full_policy_all_20260909_r1/manifest.json
```

服务不自动重试失败任务；完整恢复准入结果以新目录 `contract/`、`training_admission.json` 和 `status.json` 为准。
