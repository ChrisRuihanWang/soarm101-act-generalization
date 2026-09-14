from pathlib import Path
import sys
import subprocess
import pandas as pd

ROOT = Path.home() / "robotics/so101_act_fixed/data/grid5_t1_50ep"
FPS = 30

if len(sys.argv) != 3:
    print("Usage: python play_grid5_episode.py EPISODE CAMERA")
    print("Example: python play_grid5_episode.py 27 side")
    sys.exit(1)

EP = int(sys.argv[1])
CAM = sys.argv[2]

if CAM not in ["top", "side"]:
    raise ValueError("camera must be 'top' or 'side'")

# ------------------------------------------------------------
# Read parquet and determine global episode offset
# ------------------------------------------------------------

parquet_files = sorted((ROOT / "data").rglob("*.parquet"))

if not parquet_files:
    raise RuntimeError("No parquet files found")

df = pd.concat(
    [pd.read_parquet(p) for p in parquet_files],
    ignore_index=True
)

if EP not in df["episode_index"].unique():
    raise RuntimeError(f"Episode {EP} does not exist")

counts = (
    df.groupby("episode_index")
    .size()
    .sort_index()
)

ep_frames = int(counts.loc[EP])

frames_before = int(
    counts[counts.index < EP].sum()
)

global_start = frames_before / FPS
duration = ep_frames / FPS

print(
    f"Episode {EP}, {CAM}: "
    f"global_start={global_start:.2f}s, "
    f"duration={duration:.2f}s"
)

# ------------------------------------------------------------
# Find which physical mp4 contains this episode
# ------------------------------------------------------------

video_dir = ROOT / "videos" / f"observation.images.{CAM}"

videos = sorted(video_dir.rglob("*.mp4"))

if not videos:
    raise RuntimeError(f"No videos found under {video_dir}")


def get_duration(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    return float(
        subprocess.check_output(cmd, text=True).strip()
    )


cumulative = 0.0
chosen = None
local_start = None
chosen_duration = None

for video in videos:

    vd = get_duration(video)

    if cumulative <= global_start < cumulative + vd:
        chosen = video
        local_start = global_start - cumulative
        chosen_duration = vd
        break

    cumulative += vd


if chosen is None:
    raise RuntimeError(
        f"Could not locate episode {EP} in video files. "
        f"global_start={global_start:.2f}s"
    )


print()
print("Video file:")
print(chosen)

print(
    f"\nlocal_start={local_start:.2f}s "
    f"file_duration={chosen_duration:.2f}s"
)

# Check whether this episode fits entirely in this file
available = chosen_duration - local_start

play_duration = min(duration, available)

if play_duration < duration - 0.1:
    print(
        "\nWARNING: Episode crosses a video-file boundary."
    )

print(
    f"Playing {play_duration:.2f}s..."
)

# ------------------------------------------------------------
# Play
# ------------------------------------------------------------

subprocess.run([
    "ffplay",
    "-loglevel", "warning",
    "-ss", f"{local_start:.4f}",
    "-t", f"{play_duration:.4f}",
    "-autoexit",
    str(chosen),
])
