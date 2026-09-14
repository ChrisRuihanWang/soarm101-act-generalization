from pathlib import Path
import pandas as pd
import subprocess
import sys

ROOT = Path.home() / "robotics/so101_act_fixed/data/fixed_v2_standardized_30ep"
FPS = 30

if len(sys.argv) < 2:
    print("Usage:")
    print("python play_episode.py EPISODE [top|side]")
    sys.exit(1)

episode = int(sys.argv[1])
camera = sys.argv[2] if len(sys.argv) >= 3 else "top"

files = sorted((ROOT / "data").rglob("*.parquet"))

df = pd.concat(
    [pd.read_parquet(p) for p in files],
    ignore_index=True
)

counts = (
    df.groupby("episode_index")
    .size()
    .sort_index()
)

if episode not in counts.index:
    raise ValueError(f"Episode {episode} not found")

start_frame = int(
    counts.loc[counts.index < episode].sum()
)

num_frames = int(counts.loc[episode])

start_s = start_frame / FPS
duration_s = num_frames / FPS

video = (
    ROOT
    / "videos"
    / f"observation.images.{camera}"
    / "chunk-000"
    / "file-000.mp4"
)

print(
    f"Episode {episode}, {camera}: "
    f"start={start_s:.2f}s, "
    f"duration={duration_s:.2f}s"
)

subprocess.run([
    "ffplay",
    "-ss", str(start_s),
    "-t", str(duration_s),
    "-autoexit",
    str(video),
])
