# SO-ARM101 ACT Generalization

基于 SO-101 主从机械臂、顶视/侧视摄像头和 LeRobot ACT 策略的抓取与放置实验。任务是将红色方块抓起并放入固定盒子，研究从固定位置示范到多位置、混合示范的策略表现。

本仓库处于整理阶段，第一批提供已有采集、质检、录像回放和失败诊断脚本，以及保存的训练配置。ACT 模型实现、训练循环和机械臂驱动来自外部 LeRobot 安装。

## 实验记录

| 阶段 | 本地数据集 | 示范数 | 配置中的训练步数 | 检查点 |
|---|---|---:|---:|---|
| Stage1 固定位置 | `fixed_v2_standardized_30ep` | 30 | 20,000 | `020000` |
| Stage2 Grid5 | `grid5_t1_50ep` | 50 | 30,000 | `030000` |
| Stage3 混合数据 | `act_stage23_100ep` | 100 | 30,000 | `030000` |

Stage3 的源数据映射、合并过程及各阶段的泛化测试结果待整理。不要仅根据目录名推断训练与测试位置的重叠关系。

已有 Stage1 摘要记录固定位置测试 **10/10 成功**，见 [原始评估摘要](results/stage1_evaluation_summary.txt)。这是历史人工记录，尚未在本仓库重新验证；其他阶段成功率尚未发布。

## 流程

```text
主臂遥操作 + 从臂 + 双摄像头
              ↓
示范采集 → 数据质检 → LeRobot ACT 训练
                              ↓
                  策略执行与 rollout 记录
                              ↓
                   录像检查、轨迹与失败分析
```

## 当前目录

- `scripts/`：15 个现有实验脚本。
- `configs/stage*_*/`：最终检查点中的训练和策略配置副本。
- `results/`：已整理的实验摘要；逐次结果 CSV 和图表待补充。
- `docs/`：当前运行限制、数据和模型说明。
- `requirements.txt`：离线分析依赖，取自当前本机环境。

## 脚本用途

| 文件 | 用途 |
|---|---|
| `record_fixed_v1.sh`, `record_fixed_v2.sh` | 固定位置遥操作采集，支持续录 |
| `record_grid5_t1_round.sh` | A/C/E/G/I 五个位置，每轮 5 条，共 10 轮 |
| `check_dataset.py`, `full_quality_check*.py` | 元信息、时序、数值、视频及延迟补偿跟踪检查 |
| `check_rollout_*.py` | 对比 rollout 与示范的姿态、动作和夹爪行为 |
| `diagnose_transport_*.py` | 分析抓取后搬运轨迹及限幅压力 |
| `play_*episode.py` | 使用 ffplay 播放录像，不会驱动机械臂 |
| `concat_F_rollouts.py` | 裁剪并拼接 F 位置各轮次的录像 |

## 环境与运行

离线分析依赖：

```bash
python -m pip install -r requirements.txt
```

另外需要提供 `ffmpeg`、`ffprobe` 和 `ffplay`。本机环境中观察到 LeRobot `0.6.1`（editable 安装）和 PyTorch `2.11.0+cu128`；版本号本身不能保证复现原始 LeRobot 源码或本地修改，源码来源与提交记录待确认。

现有脚本保留历史实现，默认工作目录为 `~/robotics/so101_act_fixed`。如果克隆到其他位置，需要先修改脚本顶部的 `PROJECT`、`ROOT`、`TRAIN`、`ROLL` 等路径。数据与模型未包含在本仓库，下载方式待发布。

硬件采集脚本还固定使用 `/dev/ttyACM1`（从臂）、`/dev/ttyACM0`（主臂）、`/dev/video2`（顶视）及 `/dev/video4`（侧视），摄像头为 640×480、30 FPS。运行前需按自己的硬件完成校准并核对映射。

具体限制见 [当前复现状态](docs/reproducibility_status.md)，配置说明见 [configs/README.md](configs/README.md)。当前尚未提供经过验证的独立训练和策略执行入口。

## 后续补充

- 硬件安装、校准、采集和评估协议。
- 训练、rollout 及混合数据合并脚本。
- 每次评估的成功/失败、重试和干预标注。
- 成功率图表、网格热力图和有来源记录的演示片段。
- 数据和模型下载、校验值，以及源码版本记录。
- 许可证：尚未选择，本次提交不授予额外开源许可。
