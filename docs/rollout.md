# ACT rollout configuration

The main evaluation settings below were supplied from the actual experiments. They are stored in `configs/rollout.json` and used by the common `scripts/rollout.py` launcher.

| Setting | Main evaluation |
|---|---|
| Control FPS / recorded dataset FPS | 30 / 30 |
| Episode duration | 20 s |
| Reset duration | 30 s |
| `policy.n_action_steps` | **100** |
| `robot.type` | `so101_follower` |
| `robot.max_relative_target` | **10.0** |
| `display_data` | `false` |

| Camera | Device | Resolution | FPS | Capture format | Backend |
|---|---|---|---|---|---|
| top | `/dev/video2` | 640×480 | 30 | MJPG | 200 |
| side | `/dev/video4` | 640×480 | 30 | MJPG | 200 |

MJPG is the camera capture format, not a requirement for the saved dataset video codec. Saved video encoding follows the installed LeRobot configuration.

## Policy selection and port

| Profile | Policy directory, relative to the project root |
|---|---|
| `stage2` | `outputs/act_grid5_t1_30k/checkpoints/last/pretrained_model` |
| `stage3` | `outputs/act_stage23_100ep_30k/checkpoints/last/pretrained_model` |
| `grid-retest` | Same 100-demo policy as `stage3`, evaluated on the original grid |

On the original machine the project root is `/home/chris/robotics/so101_act_fixed`. Relative policy paths keep the launcher portable. `--policy-path` can select another local `pretrained_model` directory. The launcher resolves `last` to its current target and records that path for each run; no new model weights are included in this repository.

**`ROBOT_PORT` is required; there is no default port in the launcher or shared configuration.** Stage 2 originally used `/dev/ttyACM1`. Later Stage 3 experiments used `/dev/ttyACM0` because only the follower was connected. These are historical mappings, not automatic stage-dependent defaults. Set the device for the follower currently connected.

## Preview and run

Activate the original LeRobot environment first. The examples below deliberately start with `--dry-run`, which prints the command without importing LeRobot, accessing devices, or creating files. The output dataset directory must be new. Replace the example repository owner with your own identifier.

```bash
ROBOT_PORT=/dev/ttyACM1 python scripts/rollout.py stage2 \
  data/rollout_stage2_new_r1 OWNER/rollout_stage2_new_r1 \
  --num-episodes 9 --dry-run

ROBOT_PORT=/dev/ttyACM0 python scripts/rollout.py stage3 \
  data/rollout_stage3_new_r1 OWNER/rollout_stage3_new_r1 \
  --num-episodes 9 --dry-run

ROBOT_PORT=/dev/ttyACM0 python scripts/rollout.py grid-retest \
  data/rollout_grid_retest_new_r1 OWNER/rollout_grid_retest_new_r1 \
  --num-episodes 9 --dry-run
```

Remove `--dry-run` to execute on the robot. Use `stage3 --num-episodes 5` for a separately named stress-test session. The launcher does not choose object positions, yaw, or round order: these remain part of the manual protocol. It does not connect a leader arm or classify task success.

For execution, the launcher checks that the selected ACT model, processor files, and devices are present. It records the full command, resolved policy path, selected port, evaluation mode, and process exit code in a unique `logs/rollouts/.../run.json`, alongside `rollout.log`. Existing dataset directories are rejected; resuming and automatic Hub upload are disabled.

## Local LeRobot interface

The current editable LeRobot checkout separates teleoperation (`lerobot-record`) from policy deployment (`lerobot-rollout`). The launcher uses the equivalent Python module entry point with `--strategy.type=episodic --inference.type=sync`, so it runs in the same Python environment as the launcher.

Both `--fps=30` and `--dataset.fps=30` are set. `--dataset.no_stamp=true` keeps the requested dataset identifier unchanged. Task descriptions follow the existing rollout metadata: marked start positions for Stage 2/grid retest, current position for Stage 3.

Other behavior follows the installed LeRobot defaults. In the current checkout, episodic rollout without a leader returns the follower to its startup pose during resets, and also returns it at shutdown by default. These defaults are runtime behavior, not an independently verified record of historical reset procedures. Exact starting poses, calibration, and historical source revisions still need documentation.

## Separate 25-step inference ablation

`n_action_steps=25` was tested separately. It did not improve rollout performance and produced more abrupt robot motion. **It is not the main evaluation setting.** Use the explicit `--ablation-25` option only for that separate experiment:

```bash
ROBOT_PORT=/dev/ttyACM0 python scripts/rollout.py stage3 \
  data/rollout_stage3_ablation_n25 OWNER/rollout_stage3_ablation_n25 \
  --ablation-25 --num-episodes 9 --dry-run
```

Both the output directory name and dataset identifier must contain `ablation` in this mode. The run manifest records `ablation_n25`; normal runs record `main_n100`. An ambient `N_ACTION_STEPS` environment variable cannot change the main horizon. The checkpoint and training configuration are not modified.
