from pathlib import Path
import subprocess
import tempfile
from dataset_utils import episode_video_segments, load_dataset

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "data"
OUT = PROJECT / "results/F_failure_analysis"


# F 在五个 round 中对应的 episode
ROUNDS = [
    ("r1", 1),
    ("r2", 8),
    ("r3", 2),
    ("r4", 7),
    ("r5", 4),
]


def extract_episode(root, ep, camera, output):
    dataset = load_dataset(root)
    segments = episode_video_segments(dataset, ep, camera)
    duration = sum(segment[2] for segment in segments)
    pieces = []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        for idx, (video, local_start, seg_duration) in enumerate(segments):
            piece = td / f"piece_{idx:03d}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{local_start:.6f}", "-i", str(video),
                "-t", f"{seg_duration:.6f}", "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(piece),
            ], check=True)
            pieces.append(piece)

        if not pieces:
            raise RuntimeError(
                f"Could not locate ep{ep} "
                f"(duration={duration:.2f}s)"
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
        f"  ep{ep}: "
        f"duration={duration:.2f}s → {output.name}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
