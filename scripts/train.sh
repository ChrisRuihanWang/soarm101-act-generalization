#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 {stage1|stage2|stage3} [--dry-run]"
    echo "Activate lerobot_so101 first. Overrides: STEPS BATCH_SIZE NUM_WORKERS OUTPUT_DIR RUN_NAME"
}
fail() { echo "Error: $*" >&2; exit 2; }

if [[ "${1:-}" == "--help" && $# -eq 1 ]]; then usage; exit 0; fi
if [[ $# -lt 1 || $# -gt 2 ]]; then usage >&2; exit 2; fi
STAGE="$1"
DRY_RUN=false
if [[ $# -eq 2 ]]; then
    [[ "$2" == "--dry-run" ]] || fail "Unknown option: $2"
    DRY_RUN=true
fi

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "$STAGE" in
    stage1) PROFILE=stage1_fixed; DEMOS=30 ;;
    stage2) PROFILE=stage2_grid; DEMOS=50 ;;
    stage3) PROFILE=stage3_mixed; DEMOS=100 ;;
    *) fail "Stage must be stage1, stage2, or stage3: $STAGE" ;;
esac
TRAIN_CONFIG="$PROJECT/configs/$PROFILE/train_config.json"

# Read verified historical values without evaluating configuration text as shell code.
# Resolve the archived relative paths from the repository root, regardless of caller cwd.
CONFIG_VALUES="$(python - "$TRAIN_CONFIG" "$PROJECT" <<'PY'
import json
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
project = Path(sys.argv[2])
values = [project / config['dataset']['root'], config['dataset']['repo_id'],
          project / config['output_dir'], config['job_name'], config['steps']]
for value in values:
    text = str(value)
    if not text or '\n' in text or '\r' in text or 'TODO' in text:
        sys.exit('Error: missing or unresolved value in training configuration')
    print(text)
PY
)"
mapfile -t VALUES <<< "$CONFIG_VALUES"
DATASET_ROOT="${VALUES[0]}"
DATASET_REPO_ID="${VALUES[1]}"
OUTPUT_DIR="${OUTPUT_DIR-${VALUES[2]}}"
RUN_NAME="${RUN_NAME-${VALUES[3]}}"
STEPS="${STEPS-${VALUES[4]}}"
BATCH_SIZE="${BATCH_SIZE-8}"
NUM_WORKERS="${NUM_WORKERS-4}"
POLICY_TYPE=act
DEVICE=cuda
PUSH_TO_HUB=false
LOG_FREQ=100
SAVE_FREQ=5000
SEED=1000
WANDB_ENABLE=false

[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail "STEPS must be a positive integer"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_SIZE must be a positive integer"
[[ "$NUM_WORKERS" =~ ^(0|[1-9][0-9]*)$ ]] || fail "NUM_WORKERS must be a nonnegative integer"
[[ -n "${OUTPUT_DIR//[[:space:]]/}" ]] || fail "OUTPUT_DIR cannot be empty"
[[ -n "${RUN_NAME//[[:space:]]/}" ]] || fail "RUN_NAME cannot be empty"
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$PROJECT/$OUTPUT_DIR"
[[ -d "$DATASET_ROOT" ]] || fail "DATASET_ROOT does not exist: $DATASET_ROOT"

# One training implementation for all stages; CLI overrides take precedence over JSON.
COMMAND=(lerobot-train
    "--config_path=$TRAIN_CONFIG"
    "--dataset.root=$DATASET_ROOT"
    "--dataset.repo_id=$DATASET_REPO_ID"
    "--output_dir=$OUTPUT_DIR"
    "--job_name=$RUN_NAME"
    "--steps=$STEPS"
    "--policy.type=$POLICY_TYPE"
    "--policy.device=$DEVICE"
    "--policy.push_to_hub=$PUSH_TO_HUB"
    "--batch_size=$BATCH_SIZE"
    "--num_workers=$NUM_WORKERS"
    "--log_freq=$LOG_FREQ"
    "--save_freq=$SAVE_FREQ"
    "--seed=$SEED"
    "--wandb.enable=$WANDB_ENABLE"
    --resume=false
)

printf 'Stage: %s (%s demonstrations)\nConfig: %s\n' "$STAGE" "$DEMOS" "$TRAIN_CONFIG"
printf 'DATASET_ROOT=%s\nDATASET_REPO_ID=%s\nOUTPUT_DIR=%s\nRUN_NAME=%s\n' \
    "$DATASET_ROOT" "$DATASET_REPO_ID" "$OUTPUT_DIR" "$RUN_NAME"
printf 'STEPS=%s BATCH_SIZE=%s NUM_WORKERS=%s\n' "$STEPS" "$BATCH_SIZE" "$NUM_WORKERS"
printf 'POLICY_TYPE=%s DEVICE=%s PUSH_TO_HUB=%s\n' "$POLICY_TYPE" "$DEVICE" "$PUSH_TO_HUB"
printf 'LOG_FREQ=%s SAVE_FREQ=%s SEED=%s WANDB_ENABLE=%s\n' "$LOG_FREQ" "$SAVE_FREQ" "$SEED" "$WANDB_ENABLE"
printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'

if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "Dry run only: OUTPUT_DIR already exists; actual training would be refused."
    else
        fail "OUTPUT_DIR already exists: $OUTPUT_DIR. Set OUTPUT_DIR to a new path."
    fi
fi
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run: training was not started; no files were created."
    exit 0
fi
command -v lerobot-train >/dev/null || fail "lerobot-train not found; activate lerobot_so101"
cd -- "$PROJECT"
exec "${COMMAND[@]}"
