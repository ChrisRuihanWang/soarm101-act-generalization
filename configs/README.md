# 保存的训练配置

以下文件复制自各实验的最终数字检查点，不是新运行生成的配置：

| 目录 | 原实验 | 检查点 |
|---|---|---|
| `stage1_fixed/` | `act_fixed_v2_20k` | `020000` |
| `stage2_grid/` | `act_grid5_t1_30k` | `030000` |
| `stage3_mixed/` | `act_stage23_100ep_30k` | `030000` |

每个目录包含 `train_config.json` 与策略 `config.json`。原始项目根目录已替换为 `.`，数据路径和输出路径相对于运行命令的当前工作目录。其余配置保留原值，包括历史 dataset repo_id；这些标识不代表数据已经公开发布。

这些配置记录 batch size 8、随机种子 1000、ResNet18、chunk size 100、n_action_steps 100 等训练设置。实际 rollout 可能覆盖部分设置，不能据此推断所有评估时使用的参数。

模型权重、归一化参数及训练恢复状态没有提交。仅有 JSON 文件不能加载训练后的策略。训练命令及兼容的 LeRobot 源码版本待整理验证。
