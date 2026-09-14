"""Play any recorded episode using LeRobot v3 video metadata."""

import argparse
import shlex
import subprocess

from dataset_utils import episode_video_segments, load_dataset


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", help="Dataset root containing meta/info.json")
    cli.add_argument("episode", type=int)
    cli.add_argument("camera", nargs="?", default="top")
    cli.add_argument("--dry-run", action="store_true", help="Print segment commands without opening a player")
    args = cli.parse_args(argv)
    try:
        dataset = load_dataset(args.root)
        for path, start, duration in episode_video_segments(dataset, args.episode, args.camera):
            command = ["ffplay", "-loglevel", "warning", "-ss", f"{start:.6f}",
                       "-t", f"{duration:.6f}", "-autoexit", str(path)]
            print(shlex.join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        cli.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
