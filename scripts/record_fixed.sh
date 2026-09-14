#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 || "${1:-}" == "--help" ]]; then
    echo "Usage: bash $0 DATA_ROOT REPO_ID [NUM_EPISODES=30] [RESUME=false]"
    echo "Device overrides: FOLLOWER_PORT LEADER_PORT TOP_CAMERA SIDE_CAMERA"
    echo "Timing overrides: EPISODE_TIME_S RESET_TIME_S (default: 30 seconds)"
    if [[ "${1:-}" == "--help" ]]; then exit 0; else exit 2; fi
fi

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$1"
REPO_ID="$2"
NUM_EPISODES="${3:-30}"
RESUME="${4:-false}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
TOP_CAMERA="${TOP_CAMERA:-/dev/video2}"
SIDE_CAMERA="${SIDE_CAMERA:-/dev/video4}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-30}"

if ! [[ "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_EPISODES must be a positive integer." >&2; exit 2
fi
if [[ "$RESUME" != "true" && "$RESUME" != "false" ]]; then
    echo "Error: RESUME must be true or false." >&2; exit 2
fi
if [[ "$RESUME" == "false" && -e "$DATA_ROOT" ]]; then
    echo "STOP: dataset already exists: $DATA_ROOT" >&2; exit 1
fi
if [[ "$RESUME" == "true" && ! -f "$DATA_ROOT/meta/info.json" ]]; then
    echo "STOP: no dataset metadata available for resuming." >&2; exit 1
fi
for DEVICE in "$FOLLOWER_PORT" "$LEADER_PORT" "$TOP_CAMERA" "$SIDE_CAMERA"; do
    if [[ ! -r "$DEVICE" || ! -w "$DEVICE" ]]; then
        echo "STOP: device missing or inaccessible: $DEVICE" >&2; exit 1
    fi
done

# Encode camera paths as JSON instead of interpolating them into configuration text.
CAMERAS="$(python - "$TOP_CAMERA" "$SIDE_CAMERA" <<'PY'
import json
import sys
print(json.dumps({name: dict(type='opencv', index_or_path=path, width=640,
                            height=480, fps=30, fourcc='MJPG', backend=200)
                  for name, path in zip(('top', 'side'), sys.argv[1:])}))
PY
)"
LOG_DIR="${LOG_DIR:-$PROJECT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$(mktemp "$LOG_DIR/record_fixed_XXXXXXXX.log")"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=so101_follower \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=so101_leader \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATA_ROOT" \
  --dataset.single_task="Pick up the red cube from the fixed start position and place it inside the fixed box." \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --play_sounds=true \
  --resume="$RESUME" \
  2>&1 | tee "$LOG_FILE"
