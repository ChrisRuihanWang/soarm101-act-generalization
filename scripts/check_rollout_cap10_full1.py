from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd

ROLLOUT = Path.home() / "robotics/so101_act_fixed/data/rollout_act_fixed_v2_cap10_full1"
TRAIN = Path.home() / "robotics/so101_act_fixed/data/fixed_v2_standardized_30ep"

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

FPS = 30
MAX_LAG = 8


def load_dataset(root):
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet found under {root}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def stack(series):
    return np.stack(
        series.apply(lambda x: np.asarray(x, dtype=float)).to_numpy()
    )


def circular_error_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def joint_error(a, b, j):
    if j == 4:  # wrist_roll
        return circular_error_deg(a, b)
    return a - b


print("=" * 90)
print("ACT SAFETY ROLLOUT DIAGNOSTIC")
print("=" * 90)

roll = load_dataset(ROLLOUT)
train = load_dataset(TRAIN)

roll = roll.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

action = stack(roll["action"])
state = stack(roll["observation.state"])

N = len(roll)
duration = N / FPS

print("\n[1] BASIC ROLLOUT INFO")
print("Frames   :", N)
print("Duration :", round(duration, 2), "s")
print("Episodes :", sorted(roll["episode_index"].unique().tolist()))

print("\nIMPORTANT:")
if duration < 12:
    print(
        "This rollout is much shorter than your ~15–18 s demonstrations.\n"
        "It is suitable for safety/early-behavior diagnosis, but not for judging full task success."
    )


# ============================================================
# 2. Compare rollout initial state to training starts
# ============================================================

print("\n[2] INITIAL STATE vs TRAINING STARTS")

train_starts = []

for ep, g in train.groupby("episode_index"):
    g = g.sort_values("frame_index")

    # average first 10 frames to reduce noise
    x = stack(g["observation.state"].iloc[:10]).mean(axis=0)
    train_starts.append(x)

train_starts = np.stack(train_starts)

roll_start = state[:10].mean(axis=0)

diff = np.abs(train_starts - roll_start)
mean_per_ep = diff.mean(axis=1)

nearest = int(np.argmin(mean_per_ep))

print("Rollout initial state:")
for j, name in enumerate(JOINTS):
    print(f"  {name:15s}: {roll_start[j]:8.2f}")

print(f"\nNearest training episode start: {nearest}")
print(f"Mean absolute joint difference: {mean_per_ep[nearest]:.2f} deg")

print("\nPer-joint difference to nearest training start:")
for j, name in enumerate(JOINTS):
    print(f"  {name:15s}: {diff[nearest, j]:6.2f} deg")


# ============================================================
# 3. Action / state ranges
# ============================================================

print("\n[3] ROLLOUT MOTION RANGE")

for j, name in enumerate(JOINTS):
    print(
        f"{name:15s} | "
        f"A [{action[:,j].min():7.2f}, {action[:,j].max():7.2f}] "
        f"span={np.ptp(action[:,j]):6.2f} | "
        f"S [{state[:,j].min():7.2f}, {state[:,j].max():7.2f}] "
        f"span={np.ptp(state[:,j]):6.2f}"
    )


# ============================================================
# 4. Compare motion span against training demonstrations
# ============================================================

print("\n[4] ACTION SPAN vs TRAINING")

train_spans = {name: [] for name in JOINTS}

for ep, g in train.groupby("episode_index"):
    a = stack(g["action"])

    for j, name in enumerate(JOINTS):
        if j == 4:
            # wrist roll: use cumulative shortest angular step motion,
            # but span is still useful because training range is limited here
            vals = a[:, j]
            span = np.ptp(vals)
        else:
            span = np.ptp(a[:, j])

        train_spans[name].append(span)

for j, name in enumerate(JOINTS):
    rollout_span = np.ptp(action[:, j])
    train_median = np.median(train_spans[name])

    ratio = rollout_span / train_median if train_median > 1e-6 else np.nan

    print(
        f"{name:15s}: "
        f"rollout={rollout_span:6.2f} | "
        f"train median={train_median:6.2f} | "
        f"ratio={ratio:5.2f}"
    )


# ============================================================
# 5. Lag-aware tracking
# ============================================================

print("\n[5] LAG-AWARE TRACKING")

for j, name in enumerate(JOINTS):

    results = []

    for lag in range(MAX_LAG + 1):
        if lag == 0:
            a = action[:, j]
            s = state[:, j]
        else:
            a = action[:-lag, j]
            s = state[lag:, j]

        err = joint_error(a, s, j)
        ae = np.abs(err)

        results.append({
            "lag": lag,
            "mae": ae.mean(),
            "bias": np.median(err),
            "p95": np.percentile(ae, 95),
            "max": ae.max(),
        })

    best = min(results, key=lambda x: x["mae"])

    print(
        f"{name:15s}: "
        f"lag={best['lag']} frames "
        f"({best['lag']*1000/FPS:.0f} ms) | "
        f"MAE={best['mae']:.2f}° | "
        f"bias={best['bias']:.2f}° | "
        f"p95={best['p95']:.2f}° | "
        f"max={best['max']:.2f}°"
    )


# ============================================================
# 6. Learn OPEN / CLOSED gripper levels from demonstrations
# ============================================================

print("\n[6] GRIPPER BEHAVIOR")

# Your standardized dataset starts with gripper already open.
train_open_samples = []

for ep, g in train.groupby("episode_index"):
    g = g.sort_values("frame_index")
    a = stack(g["action"])
    train_open_samples.extend(a[:15, 5])

open_ref = float(np.median(train_open_samples))

# Determine which extreme corresponds to "closed"
all_train_gripper = stack(train["action"])[:, 5]

low = float(np.percentile(all_train_gripper, 5))
high = float(np.percentile(all_train_gripper, 95))

closed_ref = high if abs(high - open_ref) > abs(low - open_ref) else low

print(f"Estimated OPEN reference   : {open_ref:.2f}")
print(f"Estimated CLOSED reference : {closed_ref:.2f}")


def classify_gripper(x):
    d_open = abs(x - open_ref)
    d_closed = abs(x - closed_ref)
    return "OPEN" if d_open <= d_closed else "CLOSED"


# Print one sample every 0.5 sec
step = max(1, FPS // 2)

print("\nTime     Action   State    Policy-class  State-class")

last_policy_class = None
transitions = []

for i in range(0, N, step):

    t = i / FPS

    ga = action[i, 5]
    gs = state[i, 5]

    pa = classify_gripper(ga)
    ps = classify_gripper(gs)

    print(
        f"{t:6.2f}s | "
        f"{ga:7.2f} | "
        f"{gs:7.2f} | "
        f"{pa:11s} | "
        f"{ps:11s}"
    )

# Find policy gripper transitions frame-by-frame
classes = [classify_gripper(x) for x in action[:, 5]]

for i in range(1, len(classes)):
    if classes[i] != classes[i-1]:
        transitions.append(
            (i / FPS, classes[i-1], classes[i], action[i, 5])
        )

print("\nPolicy gripper transitions:")
if not transitions:
    print("  No OPEN/CLOSED transition detected.")
else:
    for t, a, b, value in transitions:
        print(
            f"  {t:6.2f}s : {a} -> {b} "
            f"(action={value:.2f})"
        )


# ============================================================
# 7. Find largest gripper action changes
# ============================================================

dg = np.abs(np.diff(action[:, 5]))

idx = np.argsort(dg)[-10:][::-1]

print("\nLargest gripper action changes:")

for i in idx:
    print(
        f"  t={i/FPS:6.2f}s -> {(i+1)/FPS:6.2f}s | "
        f"{action[i,5]:7.2f} -> {action[i+1,5]:7.2f} | "
        f"delta={action[i+1,5]-action[i,5]:7.2f}"
    )


# ============================================================
# 8. Video integrity
# ============================================================

print("\n[7] VIDEO CHECK")

for camera in ["top", "side"]:

    videos = sorted(
        (
            ROLLOUT /
            "videos" /
            f"observation.images.{camera}"
        ).rglob("*.mp4")
    )

    total = 0

    for video in videos:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video),
        ]

        out = subprocess.check_output(cmd, text=True).strip()
        total += int(out)

    print(f"{camera:5s}: {total} frames")

    if total == N:
        print("       Video frame count OK")
    else:
        print("       WARNING: video/data frame mismatch")


# ============================================================
# 9. Save full trace
# ============================================================

out = pd.DataFrame({
    "time_s": np.arange(N) / FPS,
})

for j, name in enumerate(JOINTS):
    out[f"action_{name}"] = action[:, j]
    out[f"state_{name}"] = state[:, j]

csv_path = (
    Path.home() /
    "robotics/so101_act_fixed/logs/"
    "rollout_cap10_full1_trace.csv"
)

csv_path.parent.mkdir(parents=True, exist_ok=True)

out.to_csv(csv_path, index=False)

print("\nSaved trace:")
print(csv_path)

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)
