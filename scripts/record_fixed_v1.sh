#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/robotics/so101_act_fixed"
DATA_ROOT="$PROJECT/data/fixed_v1_30ep"

# 第一个参数：本次新增的 episode 数，默认 30。
# 第二个参数：是否续录，默认 false。
NUM_EPISODES="${1:-30}"
RESUME="${2:-false}"

# 这是你本次已经确认的设备映射。
FOLLOWER_PORT="/dev/ttyACM1"
LEADER_PORT="/dev/ttyACM0"

if ! [[ "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：episode 数必须是正整数。"
    exit 1
fi

if [[ "$RESUME" != "true" && "$RESUME" != "false" ]]; then
    echo "错误：第二个参数只能是 true 或 false。"
    exit 1
fi

mkdir -p "$PROJECT/data" "$PROJECT/logs"

# 新建数据集时，目标目录不能已经存在。
# 不会自动删除或覆盖你已有的数据。
if [[ "$RESUME" == "false" && -e "$DATA_ROOT" ]]; then
    echo "停止：数据目录已经存在：$DATA_ROOT"
    echo "请先检查已有数据；正常续录时使用第二个参数 true。"
    exit 1
fi

if [[ "$RESUME" == "true" && ! -f "$DATA_ROOT/meta/info.json" ]]; then
    echo "停止：没有找到可供续录的数据集 metadata。"
    exit 1
fi

for DEVICE in "$FOLLOWER_PORT" "$LEADER_PORT" /dev/video2 /dev/video4; do
    if [[ ! -r "$DEVICE" || ! -w "$DEVICE" ]]; then
        echo "停止：设备不存在或没有读写权限：$DEVICE"
        exit 1
    fi
done

LOG_FILE="$PROJECT/logs/record_fixed_$(date +%Y%m%d_%H%M%S).log"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=so101_follower \
  --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: 200}, side: {type: opencv, index_or_path: "/dev/video4", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: 200}}' \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=so101_leader \
  --dataset.repo_id=chris/so101_act_fixed_v1 \
  --dataset.root="$DATA_ROOT" \
  --dataset.single_task="Pick up the red cube from the fixed start position and place it inside the fixed box." \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --play_sounds=true \
  --resume="$RESUME" \
  2>&1 | tee "$LOG_FILE"
