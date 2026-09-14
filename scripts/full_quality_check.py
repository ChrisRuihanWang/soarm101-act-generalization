from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd

ROOT = Path.home() / "robotics/so101_act_fixed/data/fixed_v1_30ep"

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

EXPECTED_EPISODES = 10
MAX_LAG = 8   # 8 frames @30fps ≈ 267ms

print("=" * 90)
print("SO-101 ACT DATASET — FULL QUALITY CHECK")
print("=" * 90)


# ============================================================
# Helpers
# ============================================================

def stack(series):
    return np.stack(
        series.apply(lambda x: np.asarray(x, dtype=float)).to_numpy()
    )


def circular_error_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def joint_error(a, b, joint_idx):
    if joint_idx == 4:  # wrist_roll
        return circular_error_deg(a, b)
    return a - b


def joint_diff(x, joint_idx):
    if joint_idx == 4:
        return circular_error_deg(x[1:], x[:-1])
    return np.diff(x)


# ============================================================
# 1. Metadata
# ============================================================

info_path = ROOT / "meta/info.json"

if not info_path.exists():
    raise FileNotFoundError(f"Missing {info_path}")

info = json.loads(info_path.read_text())

fps = info["fps"]
meta_episodes = info["total_episodes"]
meta_frames = info["total_frames"]

print("\n[1] METADATA")
print(f"Episodes : {meta_episodes}")
print(f"Frames   : {meta_frames}")
print(f"FPS      : {fps}")

if meta_episodes == EXPECTED_EPISODES:
    print("Episode count: OK")
else:
    print(
        f"WARNING: expected {EXPECTED_EPISODES}, "
        f"found {meta_episodes}"
    )

required_features = {
    "action",
    "observation.state",
    "observation.images.top",
    "observation.images.side",
    "timestamp",
    "frame_index",
    "episode_index",
}

features = set(info.get("features", {}).keys())

missing = required_features - features

if missing:
    print("MISSING FEATURES:", missing)
else:
    print("Required features: OK")


# ============================================================
# 2. Load Parquet
# ============================================================

files = sorted((ROOT / "data").rglob("*.parquet"))

if not files:
    raise RuntimeError("No parquet data found")

df = pd.concat(
    [pd.read_parquet(p) for p in files],
    ignore_index=True
)

print("\n[2] PARQUET")
print("Rows:", len(df))

if len(df) == meta_frames:
    print("Frame count matches metadata: OK")
else:
    print(
        f"WARNING: parquet={len(df)}, "
        f"metadata={meta_frames}"
    )

episodes = sorted(df["episode_index"].unique().tolist())

print("Episode indices:", episodes)

expected_indices = list(range(meta_episodes))

if episodes == expected_indices:
    print("Episode indexing: OK")
else:
    print("WARNING: episode indices are not contiguous")


# ============================================================
# 3. Numeric validity
# ============================================================

actions = stack(df["action"])
states = stack(df["observation.state"])

print("\n[3] NUMERIC VALIDITY")

checks = {
    "Action NaN": np.isnan(actions).sum(),
    "State NaN": np.isnan(states).sum(),
    "Action Inf": np.isinf(actions).sum(),
    "State Inf": np.isinf(states).sum(),
}

for k, v in checks.items():
    print(f"{k:12s}: {v}")

if all(v == 0 for v in checks.values()):
    print("Numeric validity: OK")


# ============================================================
# 4. Episode timing / continuity
# ============================================================

print("\n[4] EPISODE STRUCTURE")

episode_rows = []

for ep, g in df.groupby("episode_index", sort=True):

    g = g.sort_values("frame_index")

    frames = len(g)

    ts = g["timestamp"].to_numpy(dtype=float)
    fi = g["frame_index"].to_numpy()

    duration_nominal = frames / fps

    if len(ts) > 1:
        dt = np.diff(ts)

        median_dt = np.median(dt)
        measured_fps = 1 / median_dt if median_dt > 0 else np.nan
        max_gap_ms = dt.max() * 1000

        timestamp_ok = np.all(dt > 0)
    else:
        measured_fps = np.nan
        max_gap_ms = np.nan
        timestamp_ok = False

    frame_ok = np.all(np.diff(fi) == 1) if len(fi) > 1 else False

    episode_rows.append({
        "episode": int(ep),
        "frames": frames,
        "duration_s": duration_nominal,
        "measured_fps": measured_fps,
        "max_timestamp_gap_ms": max_gap_ms,
        "frame_index_OK": frame_ok,
        "timestamp_OK": timestamp_ok,
    })

episode_summary = pd.DataFrame(episode_rows)

print(
    episode_summary.round(2).to_string(index=False)
)


# ============================================================
# 5. Motion amount / abnormal jumps
# ============================================================

print("\n[5] MOTION CONSISTENCY")

motion_rows = []

for ep, g in df.groupby("episode_index", sort=True):

    g = g.sort_values("frame_index")

    act = stack(g["action"])
    obs = stack(g["observation.state"])

    total_action_motion = 0
    max_action_jump = 0
    max_state_jump = 0

    for j in range(6):

        da = np.abs(joint_diff(act[:, j], j))
        ds = np.abs(joint_diff(obs[:, j], j))

        if len(da):
            total_action_motion += da.sum()
            max_action_jump = max(max_action_jump, da.max())

        if len(ds):
            max_state_jump = max(max_state_jump, ds.max())

    motion_rows.append({
        "episode": int(ep),
        "total_action_motion_deg": total_action_motion,
        "max_action_jump_deg": max_action_jump,
        "max_state_jump_deg": max_state_jump,
    })

motion_df = pd.DataFrame(motion_rows)

median_motion = motion_df["total_action_motion_deg"].median()

motion_df["motion_vs_median"] = (
    motion_df["total_action_motion_deg"] / median_motion
)

print(motion_df.round(2).to_string(index=False))


# ============================================================
# 6. Tracking analysis with lag compensation
# ============================================================

print("\n[6] LAG-AWARE TRACKING")

tracking_rows = []

for ep, g in df.groupby("episode_index", sort=True):

    g = g.sort_values("frame_index")

    act = stack(g["action"])
    obs = stack(g["observation.state"])

    for j, name in enumerate(JOINTS):

        lag_results = []

        for lag in range(MAX_LAG + 1):

            if lag == 0:
                a = act[:, j]
                s = obs[:, j]
            else:
                a = act[:-lag, j]
                s = obs[lag:, j]

            err = joint_error(a, s, j)
            abs_err = np.abs(err)

            lag_results.append({
                "lag": lag,
                "mae": abs_err.mean(),
                "bias": np.median(err),
                "p95": np.percentile(abs_err, 95),
                "max": abs_err.max(),
            })

        best = min(lag_results, key=lambda x: x["mae"])

        tracking_rows.append({
            "episode": int(ep),
            "joint": name,
            "best_lag_frames": best["lag"],
            "best_lag_ms": best["lag"] * 1000 / fps,
            "MAE_deg": best["mae"],
            "median_bias_deg": best["bias"],
            "p95_deg": best["p95"],
            "max_deg": best["max"],
        })

tracking_df = pd.DataFrame(tracking_rows)

for ep in sorted(tracking_df["episode"].unique()):

    print("\nEpisode", ep)

    part = tracking_df[tracking_df["episode"] == ep]

    print(
        part[
            [
                "joint",
                "best_lag_frames",
                "best_lag_ms",
                "MAE_deg",
                "median_bias_deg",
                "p95_deg",
                "max_deg",
            ]
        ].round(2).to_string(index=False)
    )


# ============================================================
# 7. Video validation
# ============================================================

print("\n[7] VIDEO VALIDATION")

video_results = []

for camera in ["top", "side"]:

    video_dir = (
        ROOT
        / "videos"
        / f"observation.images.{camera}"
    )

    videos = sorted(video_dir.rglob("*.mp4"))

    print(f"\nCamera: {camera}")
    print("Video files:", len(videos))

    total_video_frames = 0

    for video in videos:

        cmd = [
            "ffprobe",
            "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate,width,height",
            "-of", "json",
            str(video),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        obj = json.loads(result.stdout)
        stream = obj["streams"][0]

        n = int(stream["nb_read_frames"])

        total_video_frames += n

        print(
            f"{video.name}: "
            f"{n} frames | "
            f"{stream['width']}x{stream['height']} | "
            f"{stream['avg_frame_rate']}"
        )

    print(
        f"Total {camera} frames: "
        f"{total_video_frames}"
    )

    if total_video_frames == meta_frames:
        print("Video ↔ dataset frame count: OK")
    else:
        print(
            "WARNING: video frame count does not "
            "match dataset"
        )

    video_results.append(
        (camera, total_video_frames)
    )


# ============================================================
# 8. Automatic warnings
# ============================================================

print("\n[8] AUTOMATIC WARNINGS")

warnings = []

for _, r in episode_summary.iterrows():

    ep = int(r["episode"])

    if r["duration_s"] < 8:
        warnings.append(
            f"Episode {ep}: suspiciously short "
            f"({r['duration_s']:.2f}s)"
        )

    if r["duration_s"] > 29.5:
        warnings.append(
            f"Episode {ep}: hit ~30s recording limit; "
            "check that task was fully completed"
        )

    if not (28.5 <= r["measured_fps"] <= 31.5):
        warnings.append(
            f"Episode {ep}: FPS unusual "
            f"({r['measured_fps']:.2f})"
        )

    if r["max_timestamp_gap_ms"] > 100:
        warnings.append(
            f"Episode {ep}: large timestamp gap "
            f"({r['max_timestamp_gap_ms']:.1f}ms)"
        )

    if not r["frame_index_OK"]:
        warnings.append(
            f"Episode {ep}: frame index discontinuity"
        )


for _, r in motion_df.iterrows():

    ep = int(r["episode"])

    if r["motion_vs_median"] < 0.60:
        warnings.append(
            f"Episode {ep}: unusually little movement "
            f"({r['motion_vs_median']:.2f}× median)"
        )

    if r["motion_vs_median"] > 1.70:
        warnings.append(
            f"Episode {ep}: unusually large movement "
            f"({r['motion_vs_median']:.2f}× median)"
        )

    if r["max_state_jump_deg"] > 15:
        warnings.append(
            f"Episode {ep}: state jump "
            f"{r['max_state_jump_deg']:.1f}° — inspect video"
        )


for _, r in tracking_df.iterrows():

    ep = int(r["episode"])
    joint = r["joint"]

    threshold = 8 if joint == "gripper" else 5

    if r["MAE_deg"] > threshold:
        warnings.append(
            f"Episode {ep}: high lag-adjusted tracking MAE "
            f"on {joint}: {r['MAE_deg']:.2f}°"
        )

    if joint == "wrist_roll" and abs(r["median_bias_deg"]) > 3:
        warnings.append(
            f"Episode {ep}: wrist_roll fixed bias "
            f"{r['median_bias_deg']:.2f}°"
        )


if warnings:
    for w in warnings:
        print("WARNING:", w)
else:
    print("No automatic quality problems detected.")


# ============================================================
# 9. Save reports
# ============================================================

REPORT_DIR = (
    Path.home()
    / "robotics/so101_act_fixed/logs"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

episode_summary.merge(
    motion_df,
    on="episode"
).to_csv(
    REPORT_DIR / "quality_episode_summary.csv",
    index=False
)

tracking_df.to_csv(
    REPORT_DIR / "quality_tracking.csv",
    index=False
)

print("\nReports saved:")
print(
    REPORT_DIR / "quality_episode_summary.csv"
)
print(
    REPORT_DIR / "quality_tracking.csv"
)

print("\n" + "=" * 90)
print("AUTOMATIC CHECK COMPLETE")
print("=" * 90)
