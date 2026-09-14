# 通用脚本用法

真实策略执行使用 `scripts/rollout.py`，与离线 `check.py rollout` 分开。主评估配置、必需的 `ROBOT_PORT` 和独立 25 步消融用法见 [rollout 文档](rollout.md)。

在仓库根目录、具有 `requirements.txt` 中依赖的 Python 环境下运行。检查工具适用于本项目的 SO-101 六关节 LeRobot 数据，各阶段通过路径区分；不自动判定抓取任务成功。

## 数据质量检查

```bash
python scripts/check.py dataset data/fixed_v2_standardized_30ep --expected-episodes 30
python scripts/check.py dataset data/grid5_t1_50ep --expected-episodes 50
python scripts/check.py dataset data/act_stage23_100ep --expected-episodes 100
```

`root` 是含有 `meta/info.json`、`data/` 的数据集根目录，可为绝对路径。`--expected-episodes` 可省略，始终检查实际帧数和 episode 索引与元信息是否一致。FPS 从元信息读取。

- `--skip-video`：跳过较慢的 ffprobe 视频检查；默认检查元信息中声明的所有视频摄像头。
- `--max-lag 8`：跟踪分析最大延迟帧数。短 episode 自动缩小搜索范围。
- `--min-duration 8`：短 episode 提醒阈值，单位秒。
- `--episode-limit 30`：采集时长上限附近提醒的参考值，单位秒；检查 20 秒 rollout 时可改为 20。
- `--strict`：有质检提醒时返回退出码 1。默认提醒仍返回 0；输入、读取或外部命令错误返回 2。

输出 `quality_episode_summary.csv`、`quality_tracking.csv`、`joint_ranges.csv`，以及启用视频检查时的 `video_integrity.csv`。汇总包括 NaN/Inf 计数、时间连续性、动作突变和零延迟误差；跟踪表保留旧版延迟补偿指标。含 NaN/Inf 的 episode 会被提醒并跳过其运动和跟踪计算。

## Rollout 与示范对比

```bash
python scripts/check.py rollout data/rollout_act_fixed_v2_eval10_cap10 \
  --train data/fixed_v2_standardized_30ep --skip-video

python scripts/check.py rollout data/rollout_act_grid5_t1_eval_r1 \
  --train data/grid5_t1_50ep --episode 2 --max-lag 8
```

默认逐个分析所有 episode；`--episode` 只分析指定索引。每段的初始姿态、延迟、夹爪状态转换和相对时间单独计算，重置间隔不会被当作动作。最近示范输出真实 episode 索引。

输出初始姿态与动作范围对比、逐关节跟踪、完整轨迹、夹爪状态转换以及最大夹爪跳变。即使选择单个 episode，启用的视频完整性检查仍针对整个数据集。

## 闭合后搬运诊断

```bash
python scripts/check.py transport data/rollout_act_fixed_v2_safety3 \
  --train data/fixed_v2_standardized_30ep --clamp 3

python scripts/check.py transport data/rollout_act_fixed_v2_cap10_full1 \
  --train data/fixed_v2_standardized_30ep --clamp 10 --episode 0
```

`--clamp` 必须明确提供，只设置离线分析阈值，不修改机械臂控制参数。输出整段及闭合后超过该阈值的比例、训练参考统计、1/2/3/5/8 秒后的动作和状态变化、相对运动比例及方向余弦。

`--stable-frames 5` 控制稳定闭合的连续帧数。训练和 rollout 的时间各自按元信息 FPS 换算。未检测到闭合的 episode 会记录提醒，其他 episode 仍继续分析；这不等于已经判定任务失败。

夹爪开闭参考沿用原分析假设：示范开头 15 帧夹爪张开。夹爪闭合并不能证明抓住方块。输出中的 `deg` 沿用原脚本命名，实际单位取决于记录时的归一化/校准设置。

## 报告目录

默认目录为 `logs/checks/<数据集目录名>/<模式>-<UTC时间>-<随机后缀>/`，每次运行创建独立目录。三个模式均写入 `report.json`，记录数据路径、FPS、命令参数和提醒。

可传入 `--output-dir /tmp/my-check`。显式目录必须不存在或为空，避免覆盖报告。默认生成的 CSV 不会自动进入 Git 历史；发布时再选取到 `results/`。

## 录像回放

```bash
python scripts/play_episode.py data/fixed_v2_standardized_30ep 16 top
python scripts/play_episode.py data/grid5_t1_50ep 27 side
python scripts/play_episode.py data/rollout_act_fixed_v2_eval10_cap10 3 top --dry-run
```

使用 LeRobot v3 的 `meta/episodes/` Parquet 中记录的视频 chunk/file 索引和时间区间，支持后续文件和跨文件片段。跨文件时逐段启动 ffplay。`--dry-run` 读取视频元信息并显示命令，不打开窗口、不驱动机械臂。相机默认 `top`，也支持完整 `observation.images.*` 名称。

`concat_F_rollouts.py` 使用同一定位函数，F 位置 episode 映射仍作为该特定实验的配置保留。

## 固定位置采集

```bash
# 新建 V2 数据集；目标已存在会停止。
bash scripts/record_fixed.sh data/fixed_v2_standardized_30ep chris/so101_act_fixed_v2 30 false

# 续录 5 条。
bash scripts/record_fixed.sh data/fixed_v2_standardized_30ep chris/so101_act_fixed_v2 5 true
```

顺序参数为 `DATA_ROOT REPO_ID [NUM_EPISODES=30] [RESUME=false]`。可设置 `FOLLOWER_PORT`、`LEADER_PORT`、`TOP_CAMERA`、`SIDE_CAMERA` 环境变量覆盖设备，以及 `EPISODE_TIME_S`、`RESET_TIME_S` 覆盖默认 30 秒时长。`REPO_ID` 是数据标识，采集仍关闭自动 Hub 上传。

Grid5 保留专门轮次顺序：`bash scripts/record_grid5_t1_round.sh ROUND`。采集命令会调用真实硬件，离线测试仅使用伪造的 `lerobot-record` 检查参数传递。

## 旧脚本迁移

旧文件已删除，历史实现可从 Git 历史查看。

| 旧入口 | 新入口 |
|---|---|
| `check_dataset.py`、`full_quality_check*.py` | `check.py dataset ROOT` |
| `check_rollout_safety2.py`、`check_rollout_cap10_full1.py` | `check.py rollout ROOT --train TRAIN` |
| `diagnose_transport_failure.py`、`diagnose_transport_cap10_full1.py` | `check.py transport ROOT --train TRAIN --clamp VALUE` |
| `play_episode.py EP CAMERA`、`play_grid5_episode.py`、`play_eval10_episode.py` | `play_episode.py ROOT EP [CAMERA]` |
| `record_fixed_v1.sh`、`record_fixed_v2.sh` | `record_fixed.sh ROOT REPO_ID [N] [RESUME]` |
