# Stage2 H3/F4 checkpoint 保存包（2026-09-21）

来源：`result/hkbz_train_logs/stage2_h3f4_stage1_seeds123_validation120_20260920_r2/`。
三个 trainer 均完成两个 BC epoch，退出码为 0。共同使用 Stage1 P5 seeds 1/2/3、
BC seed 11、seed3 H3/F4 教师数据和 Validation120；Ready 注入为 `none`。

| Stage1 seed | 原运行选中 epoch | 选中模型的 Validation120 平均 Cmax（秒） | 选中 checkpoint |
| --- | --- | --- | --- |
| 1 | 1 | 8475.1111 | [Best](checkpoints/seed1/checkpoint_Best.pt) |
| 2 | 2 | 8453.8222 | [Best](checkpoints/seed2/checkpoint_Best.pt) |
| 3 | 1 | 8337.8889 | [Best](checkpoints/seed3/checkpoint_Best.pt) |

每个 seed 保存四个原始文件：`checkpoint_Best.pt`、`checkpoint_Last.pt`、
`checkpoint_BCEpoch1.pt`、`checkpoint_BCEpoch2.pt`，共 12 个。
Best 用于保留原选模结果，Last 和 E2 保存完整第二轮训练边界；所有文件均为原字节复制，
没有重新序列化张量。原运行的其他别名文件及迁移前模型继续保留在本地来源目录。

[manifest.json](manifest.json) 列出包内相对路径、原始来源、大小、完整 SHA256、
Stage1 父模型哈希、epoch、模型摘要及保存时的 CPU 校验结果。
已核对 Best 的模型/Adam/RNG 与所选 epoch 相同，Last 的模型/Adam/RNG 与 E2 相同，
12 个文件均可加载，模型张量为有限值且具有完整 epoch 边界状态。

`evidence/` 保留原运行及祖先 manifest、完整状态、三个种子的 pre/E1/E2 逐例评测、
选模比较、教师索引、Stage1 来源命令、种子 1/2 迁移准入记录和运行源码归档。
原始记录保持字节不变；[源码归档](evidence/run/source_bundle.tar.gz) 内 211 个文件
与原 manifest 的代码哈希一致。验证命令只读取数据：

```bash
python scripts/verify_stage2_checkpoints.py
```

## 证据边界

这是实验 checkpoint 与证据的保存包，`formal_promotion=false`，不替换
`artifacts/stage2_frozen/20260912/` 的 H2/F4 B0，不修改已有 Stage3 来源链。

seed1/2 的迁移 pre/E1 评测 JSON 中 `policy_tau=1.0`，E2 为 `0.3`；
seed3 的 pre/E1/E2 均记录为 `0.3`。这与运行合同声明的 `evaluation_tau=0.3`
存在元数据差异。本包保留原运行选模，不修正历史分数，也不据此确认评测设置完全一致。
差异究竟来自实际执行还是记录转换，未在此次归档中重新评测确认。
同一 Validation120 被用于选模，三种子共享教师和 BC seed；这些结果不代表独立盲测结论。

包内文件与哈希校验可在新 checkout 使用。原始 payload 和证据里的历史绝对路径未改写；
完整训练数据、教师轨迹及运行环境仍需另行准备，本包不声称可以直接原地重启训练。
