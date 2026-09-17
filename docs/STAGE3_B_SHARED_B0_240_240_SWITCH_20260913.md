# Stage3 B_SHARED：第 12 批提交后切换 minibatch 240

2026-09-13 15:45 CST 已部署后台切换服务，状态为 `waiting_checkpoint`。
当前 r7 继续使用 minibatch 120 处理第 12 批；240 尚未开始容量测试或正式训练。

用户指令为“等现在的checkpoint保存后，下一个epoch，minibatch设置240”。
沿用此前明确确认的边界口径：当前 checkpoint 完整提交后，从下一批生效。

- 等待第 12 批完整提交：累计 832 次访问、24 次 PPO 更新。
- minibatch 上限 240 从第 13 批生效。该批是本数据 epoch 的尾批，只有 128 条轨迹，实际一次处理 128 条；第 14 批起用满 240。
- 保留 240 个并发环境容量、64 个 CPU 环境进程和单个共享采样/训练模型。
- 使用 GPU0，CPU 范围固定为 `0-31,64-95`。
- 保持两轮 PPO、TBPTT 8、4 GiB 输入缓存、GNN 激活重计算和已冻结的学习率配置。
- 数据访问顺序及总量保持不变：7680 次访问、41 批、82 次 PPO 更新。

切换服务先检查完整的连续提交记录及 checkpoint/update 哈希，再停止 r7 并等待 GPU0 释放。
随后执行 240 环境采样窗口和两个填满 240 条轨迹的密集 32 步反向传播窗口，
要求梯度有限、恢复一致、整卡显存保留至少 6 GiB。
容量测试不进行 Adam 更新，也不声称已经覆盖完整 PPO 批次。
完整 PPO 的有限性和策略约束继续由正式训练在提交前检查；此前用户豁免的完整数值对照不重复执行。

容量测试通过后，生成独立 r8 冻结目录，运行 CPU 回归、checkpoint 重绑定精确对照、
CPU 恢复及运行配置验证，再启动训练并检查实际 CUDA RNG 恢复及第 13 批的 128 环境尾批。
启动前的容量或准备失败会从同一个已保存 checkpoint 恢复父实验的 minibatch 120。
新训练启动后的失败保留明确错误记录，不会报告为切换成功。

本次修改包括 240 参数准入、按请求生成切换命令、尾批恢复确认及按父配置回退。
69 项 CPU 测试通过，包括完整切换流程的模拟执行、120 回退、提交边界保护和历史命令记录冲突回归。
父实验冻结代码哈希及输入验证通过；修改前的五个文件、差异和新快照均已保存。

运行记录：

- 切换服务：`hkbz-b0-switch-mb240-1789285467.service`，PID `1338895`。
- 原训练服务：`hkbz-b0-resume240-1789283675.service`，控制进程 PID `1329842`。
- 控制目录：`result/hkbz_train_logs/stage3_b_shared_b0_switch_mb240_20260913/`。
- 父实验：`result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260913_r7_env240_mb120_gpu0/`。
- 计划容量测试：`result/hkbz_train_logs/stage3_b_shared_b0_capacity240_240_recompute_20260913/`。
- 计划新实验：`result/hkbz_train_logs/stage3_b_shared_b0_local_a6000_20260913_r8_env240_mb240_gpu0/`。

控制目录中的 `request.json` 记录授权及源文件哈希；`status.json` 记录实时阶段。
只有生成 `result.json` 和新实验的 `scheduled_switch_verified.json` 后，才表示新配置启动且恢复验证完成。
截至 15:45，两个服务均为 active，切换服务没有 GPU 进程，原训练仍有新进度，已提交批数为 11。
