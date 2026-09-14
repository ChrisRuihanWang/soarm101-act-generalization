from pathlib import Path
import subprocess
import tempfile
import pandas as pd

BASE = Path.home() / "robotics/so101_act_fixed/data"
OUT = Path.home() / "robotics/so101_act_fixed/results/F_failure_analysis"
OUT.mkdir(parents=True, exist_ok=True)

FPS = 30

# F 在五个 round 中对应的 episode
ROUNDS = [
    ("r1", 1),
    ("r2", 8),
    ("r3", 2),
    ("r4", 7),
    ("r5", 4),
]


def load_counts(root):
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet found: {root}")

    df = pd.concat(
        [pd.read_parquet(p) for p in files],
        ignore_index=True
    )

    return (
        df.groupby("episode_index")
        .size()
        .sort_index()
    )


def video_duration(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    return float(subprocess.check_output(cmd, text=True).strip())


def extract_episode(root, ep, camera, output):
    counts = load_counts(root)

    if ep not in counts.index:
        raise RuntimeError(f"Episode {ep} not found in {root}")

    frames_before = int(counts[counts.index < ep].sum())
    ep_frames = int(counts.loc[ep])

    global_start = frames_before / FPS
    duration = ep_frames / FPS
    global_end = global_start + duration

    video_dir = root / "videos" / f"observation.images.{camera}"
    videos = sorted(video_dir.rglob("*.mp4"))

    if not videos:
        raise RuntimeError(f"No {camera} videos in {root}")

    pieces = []

    cumulative = 0.0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        for idx, video in enumerate(videos):
            vd = video_duration(video)

            file_start = cumulative
            file_end = cumulative + vd

            overlap_start = max(global_start, file_start)
            overlap_end = min(global_end, file_end)

            if overlap_end > overlap_start:
                local_start = overlap_start - file_start
                seg_duration = overlap_end - overlap_start

                piece = td / f"piece_{idx:03d}.mp4"

                subprocess.run([
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error",
                    "-ss", f"{local_start:.6f}",
                    "-i", str(video),
                    "-t", f"{seg_duration:.6f}",
                    "-an",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    str(piece),
                ], check=True)

                pieces.append(piece)

            cumulative += vd

        if not pieces:
            raise RuntimeError(
                f"Could not locate ep{ep} "
                f"(start={global_start:.2f}s duration={duration:.2f}s)"
            )

        concat_file = td / "concat.txt"

        concat_file.write_text(
            "\n".join(
                f"file '{p}'"
                for p in pieces
            )
        )

        subprocess.run([
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output),
        ], check=True)

    print(
        f"  ep{ep}: start={global_start:.2f}s "
        f"duration={duration:.2f}s → {output.name}"
    )


for camera in ["top", "side"]:
    print(f"\n=== {camera.upper()} ===")

    temp_clips = []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        for order, (round_name, ep) in enumerate(ROUNDS):
            root = BASE / f"rollout_act_grid5_t1_eval_{round_name}"

            clip = td / f"{order:02d}_{round_name}_F_ep{ep}_{camera}.mp4"

            print(f"{round_name.upper()} F = ep{ep}")
            extract_episode(root, ep, camera, clip)

            temp_clips.append(clip)

        concat_file = td / "all.txt"

        concat_file.write_text(
            "\n".join(
                f"file '{p}'"
                for p in temp_clips
            )
        )

        final = OUT / f"F_{camera}_R1-R5.mp4"

        subprocess.run([
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(final),
        ], check=True)

        print(f"\nFINAL → {final}")

print("\nDone.")
