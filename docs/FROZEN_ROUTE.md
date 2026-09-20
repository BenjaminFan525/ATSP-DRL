# 冻结技术路线（2026-09-20）

本次整理锁定现有最终技术链，不重跑训练、IGA 或独立确认。权威文件清单是
[`artifacts/frozen_route/20260920/manifest.json`](../artifacts/frozen_route/20260920/manifest.json)。

## 路线与来源

1. Stage1：`P5_team_time_potential_fixed`，保留三个正式 seed 的 Best，默认 seed 3。
2. Stage2：原始冻结 B0，H2/F4/soft、资源评分加 Hungarian、Ready 注入 `none`；B0 SHA256 为 `b7740b2b688c83dc7fa65bed6381243cdf2f6675f4b96808ff05a41d4e44e7e8`。
3. Stage3 初始化：B_SHARED H2 的 Best6，再进入 H3 R0；R0 E8 是续训的完整模型/Adam/ValueNorm/RNG 父状态。
4. 最终阶段：C03 R9，保留全共享编码器和角色头，H3/F4/soft，训练温度 0.03，canonical H 评测温度 0.3。
5. 比较参考：冻结 Validation120 IGA-180/IGA-1800，不再使用 Tune60 IGA 作当前比较，也不新增求解。

## 最终模型

运行目录：

`result/hkbz_train_logs/stage3_h3_r0e8_fresh_20260919_r9_env384_mb192_nopost_e10_gpu0/`

最终 checkpoint：

`attempts/20260919T155025_635239/C03/epoch_0010/checkpoint.pt`

SHA256：`08b0cb111dc941749371caa11820cac3eb166daca9cebb4e1166a88e7293e34a`。

`selection.json` 按原注册规则选择 E10：先排除风险不合格的 E8；E6 与 E10 位于 0.1% 均值并列区间，再按尾部选择 E10。
E6 及全部预定评测、失败案例、epoch 提交和完整最终运行源码均保留，不能删除不利案例后重新统计。

最终续训完成 10 个局部 epochs、3840 次访问、42 次新增 PPO 更新，累计 138 次更新。
每轮 384 环境槽、64 环境进程；E1 使用 128/128，E2 起使用 optimizer minibatch/microbatch 192/192，E4 起跳过两次轮后整批回放。
已记录的原运行使用 GPU0、CPU `0-31,64-95`。这些配置不自动授权启动新实验。

## 结果口径

最终 E10 的 Validation120 平均 makespan 为 8229.5361 秒；冻结 IGA-1800 为 8310.6556 秒。
该开发集点估计约好 0.976%，但一个训练 seed、反复使用 Validation 选模、未开启独立确认及尾部风险缺口，使 `scientific_target_confirmed` 仍为 `false`。

完整逐案例复算与限制见
[`result/stage3_analysis/h3_r9_final_20260919/report.md`](../result/stage3_analysis/h3_r9_final_20260919/report.md)。
原 `final_result.json` 中残留的 Tune60 IGA 比较、旧 budget 字段和过时 limitations 保持原始字节，仅作历史记录；当前解释以最终复算报告、实际 epoch 提交和 Validation120 冻结参考为准。

## 复现所需保留项

- Stage1 三个来源 checkpoint、命令和交接清单。
- 原 Stage2 交接包全部 19 个绑定文件、历史结果封存包及退役源码包。
- 最终 Stage2 seed 1/2 固定教师复现目录和其真实 seed3 教师/R1 来源；它是条件复现，不是独立教师复现。
- H2 Best6、H3 R0 E8、R9 的完整最终结果与不可变源码；中间恢复目录只保留被清单绑定的输入、源码和必要记录。
- 原始 Train/Validation 数据与冻结教师；Validation120 IGA 摘要、逐案例结果及原搜索预算记录。
- 少量旧协议 Markdown 仍被冻结 manifest 按 SHA256 引用，保留原名和原字节，只作为来源证据。

历史来源中的 Tune60 IGA 目录早已按用户要求删除。旧 H3 R0 的严格旧输入核验及依赖它的新建实验 `prepare` 会因此拒绝；本次不伪造缺失教师，也不改写旧封存 manifest。
冻结结果的完整性使用新的只读路线校验器核对，任何新增训练协议需要另行登记。

## 验证

```bash
python scripts/verify_frozen_route.py
python scripts/export_stage2_frozen.py --verify-only
```

代码保留清单、删除原因及原始文件恢复位置见 [工作空间整理记录](WORKSPACE_RETIREMENT.md)。
