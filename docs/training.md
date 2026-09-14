# ACT training

Activate the `lerobot_so101` Conda environment before running the script. It should contain the editable LeRobot 0.6.1 installation from `~/robotics/lerobot-0.6.1` and the `lerobot-train` command. Training uses CUDA. The launcher does not activate an environment or install dependencies.

```bash
./scripts/train.sh stage1
./scripts/train.sh stage2
./scripts/train.sh stage3
```

`scripts/train.sh` contains one shared command. Each stage selects its existing `configs/<profile>/train_config.json`; the saved training settings are loaded with `--config_path`. The launcher resolves dataset/output paths against the repository root, so invocation from another working directory also works.

## Verified original runs

All paths below are relative to the repository root, originally `/home/chris/robotics/so101_act_fixed`.

| Stage | Demonstrations | Dataset root | Dataset repo_id | Steps | Output directory | job_name |
|---|---:|---|---|---:|---|---|
| stage1 | 30 | `data/fixed_v2_standardized_30ep` | `chris/so101_act_fixed_v2` | 20000 | `outputs/act_fixed_v2_20k` | `act_fixed_v2_20k` |
| stage2 | 50 | `data/grid5_t1_50ep` | `chris/so101_act_grid5_t1_50ep` | 30000 | `outputs/act_grid5_t1_30k` | `act_grid5_t1_30k` |
| stage3 | 100 | `data/act_stage23_100ep` | `chris/so101_act_stage23_100ep` | 30000 | `outputs/act_stage23_100ep_30k` | `act_stage23_100ep_30k` |

These fields were verified against the local `outputs/<run>/checkpoints/last/pretrained_model/train_config.json` files and their archived copies in [stage1_fixed](../configs/stage1_fixed/train_config.json), [stage2_grid](../configs/stage2_grid/train_config.json), and [stage3_mixed](../configs/stage3_mixed/train_config.json). Local `meta/info.json` files confirm the episode counts. No requested configuration values remain unverified or require TODO placeholders. Dataset identifiers do not imply public availability.

Stage 3 uses the prepared mixture of 50 Stage 2 demonstrations and 50 continuous/random Stage 3 demonstrations. The script trains from that existing dataset; it does not merge datasets. The exact source-episode mapping remains a separate documentation task.

All stages explicitly set `policy.type=act`, `policy.device=cuda`, `policy.push_to_hub=false`, `batch_size=8`, `num_workers=4`, `log_freq=100`, `save_freq=5000`, `seed=1000`, and `wandb.enable=false`. Other training settings come from the saved JSON. Each run starts fresh with `resume=false`. The expected final model directory is `<OUTPUT_DIR>/checkpoints/last/pretrained_model`.

## Preview and overrides

```bash
./scripts/train.sh stage1 --dry-run
./scripts/train.sh stage2 --dry-run
./scripts/train.sh stage3 --dry-run

STEPS=60000 \
RUN_NAME=act_stage3_60k \
OUTPUT_DIR=/home/chris/robotics/so101_act_fixed/outputs/act_stage23_100ep_60k \
./scripts/train.sh stage3 --dry-run
```

Remove `--dry-run` to start training. The preview validates the stage, override values, and dataset directory, then prints the resolved settings and command without invoking LeRobot or creating files. An existing output is reported during preview; actual execution refuses any existing output path, including files and symbolic links. The original three output directories already exist on the experiment machine, so select a new output for another run. Nothing is deleted or resumed automatically.

Supported environment overrides are `STEPS`, `BATCH_SIZE`, `NUM_WORKERS`, `OUTPUT_DIR`, and `RUN_NAME` (mapped to LeRobot's `job_name`). Steps/batch size must be positive integers; workers may be zero. Relative output overrides are resolved from the repository root. Changing the run name alone does not change the output directory.

Validation covers Bash syntax, all three configuration previews, override handling, output protection, and parsing with the installed LeRobot configuration classes. No training was started during this implementation. Exact historical LeRobot source revisions and the complete CUDA/PyTorch environment still need documentation.
