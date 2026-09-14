#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-}"

if ! [[ "$ROUND" =~ ^([1-9]|10)$ ]]; then
    echo "Usage: $0 ROUND"
    echo "ROUND must be 1-10"
    exit 1
fi

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$PROJECT/data/grid5_t1_50ep"

FOLLOWER_PORT="/dev/ttyACM1"
LEADER_PORT="/dev/ttyACM0"

declare -A ORDER
ORDER[1]="A C E G I"
ORDER[2]="G I A C E"
ORDER[3]="C E G I A"
ORDER[4]="I A C E G"
ORDER[5]="E G I A C"
ORDER[6]="A I G E C"
ORDER[7]="E C A I G"
ORDER[8]="I G E C A"
ORDER[9]="C A I G E"
ORDER[10]="G E C A I"

START_EP=$(( (ROUND - 1) * 5 ))
END_EP=$(( START_EP + 4 ))

if [[ "$ROUND" -eq 1 ]]; then
    RESUME=false

    if [[ -e "$DATA_ROOT" ]]; then
        echo "STOP: dataset already exists:"
        echo "$DATA_ROOT"
        exit 1
    fi
else
    RESUME=true

    if [[ ! -f "$DATA_ROOT/meta/info.json" ]]; then
        echo "STOP: existing dataset metadata not found."
        echo "Cannot resume."
        exit 1
    fi
fi

echo
echo "=============================================="
echo " GRID5 T1 COLLECTION — ROUND $ROUND"
echo "=============================================="
echo
echo "Order: ${ORDER[$ROUND]}"
echo
echo "Episode mapping:"
read -ra POSITIONS <<< "${ORDER[$ROUND]}"

for i in {0..4}; do
    EP=$((START_EP + i))
    echo "  ep${EP} -> ${POSITIONS[$i]}"
done

echo
echo "Dataset:"
echo "  $DATA_ROOT"
echo
echo "Follower: $FOLLOWER_PORT"
echo "Leader:   $LEADER_PORT"
echo
echo "Check cube is at position ${POSITIONS[0]} before recording."
echo
read -rp "Press ENTER to start Round $ROUND..."

mkdir -p "$PROJECT/logs"

LOG_FILE="$PROJECT/logs/grid5_t1_round${ROUND}_$(date +%Y%m%d_%H%M%S).log"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=so101_follower \
  --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: 200}, side: {type: opencv, index_or_path: "/dev/video4", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: 200}}' \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=so101_leader \
  --dataset.repo_id=chris/so101_act_grid5_t1_50ep \
  --dataset.root="$DATA_ROOT" \
  --dataset.single_task="Pick up the red cube from its marked start position and place it inside the fixed box." \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --play_sounds=true \
  --resume="$RESUME" \
  2>&1 | tee "$LOG_FILE"

echo
echo "=============================================="
echo "ROUND $ROUND FINISHED"
echo "Expected dataset episodes after this round: $((ROUND * 5))"
echo "=============================================="
