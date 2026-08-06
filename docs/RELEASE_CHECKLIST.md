# GitHub 发布检查清单

## 已完成

- 从 HKBZ 快照 `cbaef8ed78f15d0516dee900e054b7fac2e671ab` 建立独立整理分支；
- 移除旧 MASA-QMIX 实现、已提交的 Python bytecode、历史文本输出和无关环境示例；
- 保留 HKBZ GNN-MAPPO 主线、基线评估和该提交已有的图表数据；
- 移除本机绝对路径，给训练、数据生成和评估入口增加可移植参数；
- 扩展忽略规则，避免数据、checkpoint、日志和临时合并文件误提交；
- 增加安装、数据、训练、评估和复现边界说明。

## Push 前必须人工确认

- [ ] 选择并添加适当的开源许可证；原仓库未声明许可证，不能由整理过程代替权利人作决定。
- [ ] 确认有权公开当前 HKBZ 代码以及 MASA-QMIX/其他上游衍生部分，并补充完整 attribution。
- [ ] 确认 `experiment/results/` 和 `experiment/figures/` 可公开，且图表数据与论文版本一致。
- [ ] 决定是否发布数据集与 checkpoint；若发布，使用 GitHub Release/LFS 或独立数据仓库并给出许可证和校验和。
- [ ] 将远端仓库核对为预期的 GitHub 项目，避免推送到历史 `MASA-QMIX` remote。
- [ ] 在干净环境中按 README 完成一次安装、生成小数据集、训练 smoke test 和评估 smoke test。
- [ ] 检查提交作者、项目名称、联系信息、引用格式和敏感信息。

建议 push 前执行：

```bash
git status --short
git diff --check
python -m compileall -q onpolicy tests
pytest -q
git remote -v
```
