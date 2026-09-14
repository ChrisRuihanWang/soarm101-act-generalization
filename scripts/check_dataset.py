from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd

ROOT = Path.home() / "robotics/so101_act_fixed/data/fixed_v1_30ep"

print("=" * 70)
print("SO-101 ACT DATASET QUALITY CHECK")
print("=" * 70)

# --------------------------------------------------
# 1. Metadata
# --------------------------------------------------

info_path = ROOT / "meta/info.json"

if not info_path.exists():
    raise FileNotFoundError(f"Missing: {info_path}")

info = json.loads(info_path.read_text())

expected_fps = info["fps"]

print("\n[1] METADATA")
print(f"Episodes : {info.get('total_episodes')}")
print(f"Frames   : {info.get('total_frames')}")
print(f"FPS      : {expected_fps}")

print("\nFeatures:")
for k in info.get("features", {}):
    print(" ", k)


# --------------------------------------------------
# 2. Read all parquet files
# --------------------------------------------------

parquet_files = sorted((ROOT / "data").rglob("*.parquet"))

if not parquet_files:
    raise RuntimeError("No parquet files found.")

dfs = [pd.read_parquet(p) for p in parquet_files]
df = pd.concat(dfs, ignore_index=True)

print("\n[2] PARQUET")
print(f"Rows in parquet : {len(df)}")

if len(df) == info["total_frames"]:
    print("Frame count      : OK")
else:
    print(
        f"WARNING: metadata={info['total_frames']} "
        f"but parquet={len(df)}"
    )


# --------------------------------------------------
# Helper: convert array column
# --------------------------------------------------

def stack_column(series):
    return np.stack(
        series.apply(lambda x: np.asarray(x, dtype=float)).to_numpy()
    )


actions = stack_column(df["action"])
states = stack_column(df["observation.state"])


# --------------------------------------------------
# 3. Basic validity
# --------------------------------------------------

print("\n[3] BASIC VALIDITY")

print("Action NaN :", np.isnan(actions).sum())
print("State NaN  :", np.isnan(states).sum())

print("Action Inf :", np.isinf(actions).sum())
print("State Inf  :", np.isinf(states).sum())

if (
    np.isnan(actions).sum() == 0
    and np.isnan(states).sum() == 0
    and np.isinf(actions).sum() == 0
    and np.isinf(states).sum() == 0
):
    print("Numeric data: OK")


# --------------------------------------------------
# 4. Per-episode analysis
# --------------------------------------------------

print("\n[4] EPISODE QUALITY")

rows = []

for ep, ep_df in df.groupby("episode_index", sort=True):

    ep_df = ep_df.sort_values("frame_index")

    frames = len(ep_df)

    frame_idx = ep_df["frame_index"].to_numpy()
    timestamp = ep_df["timestamp"].to_numpy(dtype=float)

    act = stack_column(ep_df["action"])
    obs = stack_column(ep_df["observation.state"])

    # Duration
    if len(timestamp) > 1:
        duration_timestamp = timestamp[-1] - timestamp[0]

        dt = np.diff(timestamp)
        median_dt = np.median(dt)
        measured_fps = 1.0 / median_dt if median_dt > 0 else np.nan

        timestamp_monotonic = np.all(dt > 0)
    else:
        duration_timestamp = 0
        measured_fps = np.nan
        timestamp_monotonic = False

    nominal_duration = frames / expected_fps

    # Frame index continuity
    expected_idx = np.arange(frame_idx[0], frame_idx[0] + frames)
    frame_contiguous = np.array_equal(frame_idx, expected_idx)

    # Tracking error: leader command vs follower state
    tracking_error = np.abs(act - obs)

    mean_track_error = tracking_error.mean()
    p95_track_error = np.percentile(tracking_error, 95)
    max_track_error = tracking_error.max()

    # Frame-to-frame joint changes
    if len(act) > 1:
        action_step = np.abs(np.diff(act, axis=0))
        state_step = np.abs(np.diff(obs, axis=0))

        max_action_jump = action_step.max()
        max_state_jump = state_step.max()
    else:
        max_action_jump = np.nan
        max_state_jump = np.nan

    rows.append({
        "episode": int(ep),
        "frames": frames,
        "duration_s": round(nominal_duration, 2),
        "timestamp_duration_s": round(duration_timestamp, 2),
        "measured_fps": round(measured_fps, 2),
        "frame_index_OK": frame_contiguous,
        "timestamp_OK": timestamp_monotonic,
        "mean_track_err": round(mean_track_error, 2),
        "p95_track_err": round(p95_track_error, 2),
        "max_track_err": round(max_track_error, 2),
        "max_action_jump": round(max_action_jump, 2),
        "max_state_jump": round(max_state_jump, 2),
    })


summary = pd.DataFrame(rows)

print(summary.to_string(index=False))


# --------------------------------------------------
# 5. Flag suspicious episodes
# --------------------------------------------------

print("\n[5] POSSIBLE PROBLEMS")

problems_found = False

for _, row in summary.iterrows():

    ep = int(row["episode"])

    # Extremely short demonstrations
    if row["duration_s"] < 3:
        print(f"Episode {ep}: VERY SHORT ({row['duration_s']} s)")
        problems_found = True

    if not row["frame_index_OK"]:
        print(f"Episode {ep}: frame_index is not contiguous")
        problems_found = True

    if not row["timestamp_OK"]:
        print(f"Episode {ep}: timestamp is not strictly increasing")
        problems_found = True

    # FPS sanity range
    if not (27 <= row["measured_fps"] <= 33):
        print(
            f"Episode {ep}: unusual measured FPS "
            f"({row['measured_fps']})"
        )
        problems_found = True

if not problems_found:
    print("No obvious structural problems detected.")


# --------------------------------------------------
# 6. Joint ranges
# --------------------------------------------------

print("\n[6] ACTION / STATE RANGES")

for j in range(actions.shape[1]):

    print(
        f"Joint {j}: "
        f"action [{actions[:, j].min():.2f}, {actions[:, j].max():.2f}] | "
        f"state [{states[:, j].min():.2f}, {states[:, j].max():.2f}]"
    )


# --------------------------------------------------
# 7. Video inspection with ffprobe
# --------------------------------------------------

print("\n[7] VIDEO FILES")

for camera in ["top", "side"]:

    video_dir = ROOT / "videos" / f"observation.images.{camera}"
    videos = sorted(video_dir.rglob("*.mp4"))

    print(f"\nCamera: {camera}")
    print(f"Video files: {len(videos)}")

    for video in videos:

        cmd = [
            "ffprobe",
            "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate,duration",
            "-of", "json",
            str(video),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            data = json.loads(result.stdout)
            stream = data["streams"][0]

            print(f"  {video.name}")
            print(f"    frames   : {stream.get('nb_read_frames')}")
            print(f"    fps      : {stream.get('avg_frame_rate')}")
            print(f"    duration : {stream.get('duration')}")

        except Exception as e:
            print(f"  ffprobe failed for {video}: {e}")


print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)
