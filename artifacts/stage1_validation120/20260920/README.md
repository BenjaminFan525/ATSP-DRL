# Stage1 validation120 实验归档（2026-09-20）

本包保存本轮 Stage1 IGA-180/1800 和固定 validation60 Best 的补充评测，
于 2026-09-21 合并到 origin 最新代码。它是 Stage1 `plane_pretrain`、heuristic
资源后端的结果，不与 Stage2/Stage3 联合资源控制得分合并排名。

## 结果与验收状态

完整面板为 120 个相同内容指纹的案例：60 IID、54 OOD-stress、6 OOD-scale。
平均 makespan 等于 0.50/0.45/0.05 加权分，单位秒，越低越好。
学习模型为 3 个训练种子的均值 ± 样本标准差；IGA 为一个搜索种子的结果。

| 方法 | validation120 makespan | 旧60例 × 3精确复现 | 状态 |
|---|---:|---:|---|
| IGA-180 | 9324.18 | 不适用 | 两档均完成120例 |
| IGA-1800 | 9016.00 | 不适用 | 111例改善、9例持平 |
| P5 | 9128.13 ± 34.30 | 180/180 | 通过复现检查 |
| DANIEL-AT | 9678.44 ± 37.19 | 177/180 | 暂定，待核查 |
| FJSP-DRL-AT | 9689.46 ± 176.47 | 151/180 | 暂定，待核查 |
| Multi-PPO-AT | 9695.10 ± 47.28 | 178/180 | 暂定，待核查 |
| L2D-AT | 10764.38 ± 81.34 | 180/180 | 通过复现检查 |

四基线12个 checkpoint 的1440次评测全部完成，无超时或循环终止；复用已审计的
P5三个 checkpoint 后共1800条结果。四基线旧720个案例-种子组合中34个结果变化，
因此 suite 保留 `needs_review`，不把文件校验成功解释为全部科学复现通过。
未跨 r2/r3 挑选更好的结果，也未用历史 makespan 替换本轮观测。

固定原 validation60 Best，不重训、不在120例上重选。validation120包含原选模60例，
不是独立测试集。IGA累计预算1800秒来自180秒搜索加继承最优解后的1620秒搜索。

## 文件入口

- [学习模型完整观测报告](learning/analysis_observed/report.md)：分布、种子、尾部、逐例胜负及限制。
- [学习模型汇总](learning/analysis_observed/summary.json)、[1800条逐例结果](learning/analysis_observed/per_case.csv)、[34条复现差异](learning/analysis_observed/overlap_differences.csv)。
- `learning/results/`、`learning/audits/`、`learning/references/`：15份原始输出、审计及原validation60对照。
- [IGA分析](iga/analysis/report.md)、[逐例结果](iga/analysis/per_case.csv)、`iga/full/iga180/summary.json` 与 `iga/full/iga1800/summary.json`。
- `prior_attempts/r1/` 与 `prior_attempts/r2/`：早期未通过或部分通过的结果及诊断记录；不作为r3替代数据。
- `learning/source_bundle.tar.gz` 与 `iga/source_bundle.tar.gz`：对应冻结源码和原分析脚本；[manifest.json](manifest.json)逐文件记录来源、大小及SHA256。

IGA的对齐validation60摘要保留原历史分组口径（29/29/2）以核对旧选模结果；
其中32例旧标签与实际数据标签不同。真实分布为30/27/3，真实标签下IGA-180/1800
对齐60例均值为9383.42/9038.50。完整validation120始终使用真实标签60/54/6。

原始JSON、CSV、报告与源码保持原字节；其中绝对路径、运行命令与时间戳是历史来源信息。
`learning/summary.json`仅含完全通过验收的方法，完整观测表应读`analysis_observed/summary.json`。
本包不包含checkpoint张量、数据集载荷、教师轨迹和日志流；其来源及摘要保留在原manifest。
原文件仍在本机忽略目录，归档时已核对学习实验1215个、IGA实验1179个冻结文件。
旧源码仅作为本次实验的来源证据，不恢复远端已退役的活动训练路线。

## 可移植只读校验

在任意新checkout的仓库根目录运行，无需GPU、训练依赖或本机绝对路径：

```bash
python scripts/verify_stage1_validation120.py
```

校验122个归档文件、920个源码成员、案例内容指纹、固定选点、历史复现差异和均值。
此命令不运行模型、搜索或训练，也不替代未归档checkpoint/数据集的实际重放。
