# Stage1 最终交接：P5

权威入口是 [stage1_m2_handoff.json](onpolicy/config/stage1_m2_handoff.json)。文件名沿用历史命名，当前 `winner` 已是 **P5_team_time_potential_fixed**，不是早期 M2。

Stage1 已关闭，使用 `plane_pretrain`、固定飞机顺序、`joint_pair` 解码和 `f1f2` 全局特征。
环境语义为 `progressive-departure-r014-pipeline-v2`；默认来源是按正式验证分数选出的 seed 3，盲测未参与选模。

| Stage1 seed | 选中 epoch | checkpoint SHA256 前缀 |
| --- | ---: | --- |
| 1 | 8 | `3055a1110502f63bf` |
| 2 | 7 | `70498db39c336c48` |
| 3 | 8 | `8ce0244f9b0b3877` |

三份 `checkpoint_Best.pt`、原始 `formal_P5.json` 命令及其完整哈希均保留，具体路径由交接清单和 [总路线 manifest](artifacts/frozen_route/20260920/manifest.json) 给出。
Stage2 冻结 B0 使用 seed 3；后来完成的固定教师 seed 1/2 复现单独保留，未替换原冻结 B0。

交接清单记录的三 seed 正式验证均值为 9196.7833 秒，盲测均值/标准差为 8985.4111 / 70.2884 秒。
三个 seed 均通过当时的 IGA-1800 103% 接近目标，但未达到 IGA-1800 的绝对数值；这些是该 Stage1 协议的历史结论，不与 Stage3 H3 的 Validation120 混用。

后续 Stage2 必须保留所选 Stage1 的飞机策略和共享参数来源。旧 M2、闭环筛选和尾部恢复分支的启动文档已经退役，不能作为当前默认来源。
