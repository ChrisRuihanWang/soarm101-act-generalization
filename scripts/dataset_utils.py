"""Shared readers and metrics for the project's six-joint SO-101 datasets."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def stack(series):
    values = np.stack([np.asarray(x, dtype=float) for x in series])
    if values.shape != (len(series), len(JOINTS)):
        raise ValueError(f"Expected SO-101 vectors with 6 joints, got {values.shape}")
    return values


def delta(a, b):
    """Signed differences; wrist_roll uses the shortest angular difference."""
    result = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    result = result.copy()
    result[..., 4] = (result[..., 4] + 180) % 360 - 180
    return result


@dataclass
class Dataset:
    root: Path
    info: dict
    frames: pd.DataFrame

    @property
    def fps(self):
        return float(self.info["fps"])

    @property
    def cameras(self):
        return [k for k, v in self.info.get("features", {}).items() if v.get("dtype") == "video"]

    def episodes(self, episode=None):
        selected = self.frames
        if episode is not None:
            selected = selected[selected.episode_index == episode]
            if selected.empty:
                raise ValueError(f"Episode {episode} not found in {self.root}")
        return selected.groupby("episode_index", sort=True)


def load_dataset(root):
    root = Path(root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid metadata FPS: {fps}")
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet data under {root / 'data'}")
    frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    required = {"episode_index", "frame_index", "timestamp", "action", "observation.state"}
    missing = required - set(frames.columns)
    if missing or frames.empty:
        raise ValueError(f"Empty dataset or missing columns: {sorted(missing)}")
    for column in ("episode_index", "frame_index"):
        values = frames[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain nonnegative integer indices")
    for column in ("action", "observation.state"):
        stack(frames[column])
        names = info.get("features", {}).get(column, {}).get("names")
        if names and [n.removesuffix('.pos') for n in names] != list(JOINTS):
            raise ValueError(f"Unsupported joint order in {column}: {names}")
    frames = frames.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    return Dataset(root, info, frames)


def require_finite(dataset):
    for column in ("action", "observation.state"):
        if not np.isfinite(stack(dataset.frames[column])).all():
            raise ValueError(f"Nonfinite {column} in {dataset.root}; run 'check.py dataset' first")


def tracking(action, state, fps, max_lag=8):
    """Call separately for each episode: lag windows never cross resets."""
    if len(action) == 0 or max_lag < 0:
        raise ValueError("Tracking needs at least one frame and a nonnegative max_lag")
    candidates = []
    for lag in range(min(max_lag, len(action) - 1) + 1):
        error = delta(action[:-lag] if lag else action, state[lag:])
        candidates.append((lag, error))
    rows = []
    for j, joint in enumerate(JOINTS):
        lag, error = min(candidates, key=lambda item: np.abs(item[1][:, j]).mean())
        signed = error[:, j]
        absolute = np.abs(signed)
        rows.append(dict(joint=joint, best_lag_frames=lag, best_lag_ms=lag * 1000 / fps,
                         MAE_deg=absolute.mean(), median_bias_deg=np.median(signed),
                         p95_deg=np.percentile(absolute, 95), max_deg=absolute.max()))
    return rows


def gripper_levels(train):
    # Historical demonstrations start with an open gripper. Document this assumption.
    starts = np.concatenate([stack(g.action)[:15, 5] for _, g in train.episodes()])
    opened = float(np.median(starts))
    low, high = np.percentile(stack(train.frames.action)[:, 5], [5, 95])
    closed = float(high if abs(high - opened) > abs(low - opened) else low)
    if abs(opened - closed) < 1e-6:
        raise ValueError("Cannot infer distinct gripper open/closed levels from training data")
    return opened, closed


def first_closed(action, opened, closed, stable_frames=5):
    threshold = (opened + closed) / 2
    mask = action[:, 5] < threshold if closed < opened else action[:, 5] > threshold
    for i in range(len(mask) - stable_frames + 1):
        if mask[i:i + stable_frames].all():
            return i
    return None


def probe_video(path, count_frames=False):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command.append("-count_frames")
    command += ["-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height:format=duration",
                "-of", "json", str(path)]
    return json.loads(subprocess.check_output(command, text=True))


def episode_video_segments(dataset, episode, camera):
    """Use LeRobot v3 episode metadata instead of assuming concatenated frame offsets."""
    key = camera if camera.startswith("observation.images.") else f"observation.images.{camera}"
    if key not in dataset.cameras:
        raise ValueError(f"Camera {camera!r} not found; available: {dataset.cameras}")
    list(dataset.episodes(episode))  # Validate the requested episode against actual frames.
    files = sorted((dataset.root / "meta/episodes").rglob("*.parquet"))
    if not files:
        raise ValueError("Playback requires LeRobot v3 meta/episodes Parquet files")
    meta = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    rows = meta[meta.episode_index == episode]
    if len(rows) != 1:
        raise ValueError(f"Expected one metadata row for episode {episode}, found {len(rows)}")
    row = rows.iloc[0]
    prefix = f"videos/{key}/"
    start, end = float(row[prefix + "from_timestamp"]), float(row[prefix + "to_timestamp"])
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError(f"Invalid video timestamps: {start}, {end}")
    template = dataset.info["video_path"]
    path = (dataset.root / template.format(video_key=key, chunk_index=int(row[prefix + "chunk_index"]),
                                          file_index=int(row[prefix + "file_index"]))).resolve()
    if not path.is_relative_to(dataset.root):
        raise ValueError("Video path must stay inside the dataset directory")
    videos = sorted((dataset.root / "videos" / key).rglob("*.mp4"))
    if path not in videos:
        raise FileNotFoundError(path)
    remaining = end - start
    segments = []
    for video in videos[videos.index(path):]:
        duration = float(probe_video(video)["format"]["duration"])
        if not np.isfinite(duration) or duration <= start:
            raise ValueError(f"Invalid video duration/start for {video}: {duration}/{start}")
        take = min(remaining, duration - start)
        segments.append((video, start, take))
        remaining -= take
        if remaining <= 1e-4:
            return segments
        start = 0.0
    raise ValueError(f"Video files end {remaining:.3f}s before the episode ends")
