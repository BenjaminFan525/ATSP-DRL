# HKBZ 阶段交接说明

2026-09-12：Stage2 开发已冻结。当前唯一受支持的冻结交接说明是
[STAGE2_FROZEN.md](STAGE2_FROZEN.md)，入口为
[冻结 manifest](artifacts/stage2_frozen/20260912/manifest.json)。

- Stage1：保留 P5 的来源核验与已训练权重。
- Stage2：保留 B0 + Hungarian、H2/F4 的固定起点及历史结果；不再自动训练。
- Stage3：使用显式冻结 manifest 初始化，严格校验来源及参数，重新建立优化器；
  新研究方案、训练预算和启动需独立决定。

本冻结不代表已经超过 IGA180 或达到 IGA1800。
本文件及旧 Stage2 协议的原始版本已保存于
`artifacts/stage2_frozen/20260912/history/retired_development.tar.gz`。
历史命令、已退役的计划和旧自动启动授权均不作为当前训练入口。
