# 工作空间整理记录（2026-09-20）

保留路线：Stage1 P5 → 冻结 Stage2 B0 → B_SHARED H3 R0 E8 → C03 R9 E10，使用冻结 Validation120 IGA 参考。
当前入口见 [FROZEN_ROUTE.md](FROZEN_ROUTE.md)，机器清单见 [cleanup_receipt.json](../artifacts/frozen_route/20260920/cleanup_receipt.json)。

## 已移出的内容

- 非最终路线的历史训练/评测目录、废弃算法分支、失败启动及资源探测产物。
- 中间恢复尝试中未被最终路线引用的轨迹缓存、额外 checkpoint 和运行日志。
- 242 个旧 Python 文件、106 个 shell 启动脚本、51 份过期文档，以及旧框架缓存和资源文件；共删除 442 个原跟踪文件。
- 共从活动目录移出 53,042 个文件，约 150.761 GiB，原始字节与 inode 保持不变。

工作区（不含 `.git`）由约 192 GiB 降至 41 GiB。文件移入同一磁盘的恢复区，因此这个数字是活动工作区减少量，不是整块磁盘的空间释放量。
本次不清理 Git 对象库、迁移备份、用户工具环境或私有配置。

## 保留原则

最终运行保留全部预定评测和不利案例；不会用删除失败案例的方式改变实验结果。
最终 checkpoint、E6 比较 checkpoint、Stage1 三个父模型、B0/R1、原始教师与匹配 IGA 参考均保留。
部分带旧日期或 failed 状态的祖先目录仍含最终 manifest 按哈希绑定的输入，保留这些输入及原始源码，其他冗余载荷已移出。
原协议 Markdown 被封存 manifest 引用的，保留原名和原字节。

运行时按 Python 导入和子进程入口依赖闭包保留。少数历史命名的共享帮助模块和 Stage2 禁用门禁仍是冻结实现/测试的依赖；它们不代表重新开放其他研究路线。
完整活动代码入口见 [retained_runtime.json](../artifacts/frozen_route/20260920/retained_runtime.json)。

## 检查结果

- 323 个受保护文件清理前后 SHA256 一致；原有哈希绑定无不匹配。
- 最终路线校验通过：195 个绑定文件、1,400 个冻结源码文件。
- Stage2 原交接包校验通过：19 个文件、2,943 份封存结果。
- 保留代码的语法与静态依赖检查通过；原可解析的 Python 依赖无缺失。
- CPU 回归：390 passed，34 subtests passed；10 skipped，原因是历史本地 committed fixture 不可用。
- 没有运行新的训练、调度评测或 IGA 求解。

这些检查说明清理后保留文件与实现完整，不构成新的科学复现或目标确认。旧 Tune60 IGA 缺失及原结果元数据的不一致仍按 [冻结路线说明](FROZEN_ROUTE.md) 记录。

## 恢复位置

仓库外恢复区：`/tmp/hkbz-workspace-retirement-20260920-tv3v88kw/`。

- `retired/`：按原相对路径保存的所有移出文件。
- `git-before.git/`：清理前 Git 镜像。
- `audit/plan.json`：每个移出路径、原因和保留依赖。
- `audit/retired-file-inventory.jsonl`：逐文件大小、inode、时间和权限。
- `audit/move-journal.jsonl`：实际移动记录。
- `audit/regression.log`、`audit/regression.xml`：完整回归输出。

需要恢复时按清单还原单个路径；不要整体覆盖之后的新工作。该恢复区位于 `/tmp`，在明确不再需要回滚前应保留它。
本记录描述 2026-09-20 的清理快照；之后新增的 Stage2 实验单独保存在
[2026-09-21 checkpoint 包](../artifacts/stage2_checkpoints/20260921_h3f4_validation120/README.md)，
不属于本次移出范围。清理改动与该保存包一同纳入后续版本提交。
