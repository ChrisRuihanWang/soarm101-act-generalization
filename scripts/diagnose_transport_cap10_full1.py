from pathlib import Path
import numpy as np
import pandas as pd

TRAIN = Path.home() / "robotics/so101_act_fixed/data/fixed_v2_standardized_30ep"
ROLL  = Path.home() / "robotics/so101_act_fixed/data/rollout_act_fixed_v2_cap10_full1"

FPS = 30
CLAMP = 3.0

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

ARM_JOINTS = JOINTS[:5]


def load(root):
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found: {root}")

    df = pd.concat(
        [pd.read_parquet(f) for f in files],
        ignore_index=True
    )

    return df.sort_values(
        ["episode_index", "frame_index"]
    ).reset_index(drop=True)


def stack(series):
    return np.stack([
        np.asarray(x, dtype=float)
        for x in series
    ])


def first_stable_closed(g, threshold, closed_is_low=True, stable_frames=5):
    a = stack(g["action"])[:, 5]

    if closed_is_low:
        mask = a < threshold
    else:
        mask = a > threshold

    for i in range(len(mask) - stable_frames + 1):
        if np.all(mask[i:i+stable_frames]):
            return i

    return None


def angle_delta(x1, x0, joint_idx):
    d = x1 - x0

    # wrist_roll shortest angular difference
    if joint_idx == 4:
        d = (d + 180) % 360 - 180

    return d


train = load(TRAIN)
roll = load(ROLL)

# ------------------------------------------------------------
# Learn gripper open / closed levels
# ------------------------------------------------------------

open_samples = []

for ep, g in train.groupby("episode_index"):
    g = g.sort_values("frame_index")
    a = stack(g["action"])
    open_samples.extend(a[:15, 5])

open_ref = float(np.median(open_samples))

all_grip = stack(train["action"])[:, 5]

low = float(np.percentile(all_grip, 5))
high = float(np.percentile(all_grip, 95))

if abs(low - open_ref) > abs(high - open_ref):
    closed_ref = low
else:
    closed_ref = high

closed_is_low = closed_ref < open_ref
threshold = (open_ref + closed_ref) / 2

print("=" * 100)
print("POST-GRASP / TRANSPORT DIAGNOSTIC")
print("=" * 100)

print(f"\nGripper OPEN   ≈ {open_ref:.2f}")
print(f"Gripper CLOSED ≈ {closed_ref:.2f}")
print(f"Threshold      ≈ {threshold:.2f}")


# ------------------------------------------------------------
# Locate grasp transition
# ------------------------------------------------------------

train_close = {}

for ep, g in train.groupby("episode_index"):
    g = g.sort_values("frame_index").reset_index(drop=True)

    idx = first_stable_closed(
        g,
        threshold,
        closed_is_low
    )

    if idx is not None:
        train_close[int(ep)] = idx


roll_ep = roll[roll["episode_index"] == roll["episode_index"].iloc[0]]
roll_ep = roll_ep.sort_values("frame_index").reset_index(drop=True)

roll_close = first_stable_closed(
    roll_ep,
    threshold,
    closed_is_low
)

print("\n[1] GRASP TIMING")

times = np.array(list(train_close.values())) / FPS

print(
    f"Training grasp time: median={np.median(times):.2f}s "
    f"range=[{times.min():.2f}, {times.max():.2f}]s"
)

if roll_close is None:
    raise RuntimeError("Could not detect stable grasp in rollout.")

print(f"Rollout grasp time : {roll_close/FPS:.2f}s")


# ------------------------------------------------------------
# Post-grasp trajectory deltas
# ------------------------------------------------------------

print("\n[2] POST-GRASP ACTION MOVEMENT")
print("Positive/negative values show joint movement relative to grasp moment.")

horizons = [1, 2, 3, 5, 8]

roll_a = stack(roll_ep["action"])

for sec in horizons:

    step = int(sec * FPS)

    if roll_close + step >= len(roll_a):
        continue

    roll_delta = np.zeros(5)

    for j in range(5):
        roll_delta[j] = angle_delta(
            roll_a[roll_close + step, j],
            roll_a[roll_close, j],
            j
        )

    train_deltas = []

    for ep, close_idx in train_close.items():

        g = (
            train[train["episode_index"] == ep]
            .sort_values("frame_index")
            .reset_index(drop=True)
        )

        a = stack(g["action"])

        if close_idx + step >= len(a):
            continue

        d = np.zeros(5)

        for j in range(5):
            d[j] = angle_delta(
                a[close_idx + step, j],
                a[close_idx, j],
                j
            )

        train_deltas.append(d)

    if not train_deltas:
        continue

    train_deltas = np.stack(train_deltas)

    median = np.median(train_deltas, axis=0)

    nr = np.linalg.norm(roll_delta)
    nt = np.linalg.norm(median)

    if nr > 1e-6 and nt > 1e-6:
        cosine = np.dot(roll_delta, median) / (nr * nt)
    else:
        cosine = np.nan

    ratio = nr / nt if nt > 1e-6 else np.nan

    print(f"\n--- {sec} s after grasp ---")

    for j, name in enumerate(ARM_JOINTS):
        print(
            f"{name:15s} "
            f"rollout={roll_delta[j]:7.2f}° | "
            f"train median={median[j]:7.2f}°"
        )

    print(f"Overall movement ratio : {ratio:.2f}")
    print(f"Direction cosine       : {cosine:.2f}")


# ------------------------------------------------------------
# Actual STATE movement after grasp
# ------------------------------------------------------------

print("\n[3] ACTUAL ROBOT STATE MOVEMENT AFTER GRASP")

roll_s = stack(roll_ep["observation.state"])

for sec in [1, 2, 3, 5, 8]:

    step = sec * FPS

    if roll_close + step >= len(roll_s):
        continue

    print(f"\n{sec}s after grasp:")

    for j, name in enumerate(ARM_JOINTS):
        d = angle_delta(
            roll_s[roll_close + step, j],
            roll_s[roll_close, j],
            j
        )

        print(f"  {name:15s}: {d:7.2f}°")


# ------------------------------------------------------------
# Is max_relative_target=3 clipping likely?
# ------------------------------------------------------------

print("\n[4] max_relative_target=3.0 PRESSURE")
print(
    "Fraction of frames where |action - current state| exceeds 3 degrees.\n"
    "A high value means the safety limiter can materially change execution."
)


def clamp_stats(df, start=0):
    a = stack(df["action"])[start:]
    s = stack(df["observation.state"])[start:]

    result = []

    for j in range(5):
        err = a[:, j] - s[:, j]

        if j == 4:
            err = (err + 180) % 360 - 180

        ae = np.abs(err)

        result.append((
            np.mean(ae > CLAMP) * 100,
            np.percentile(ae, 95),
            np.max(ae),
        ))

    return result


roll_stats_all = clamp_stats(roll_ep)
roll_stats_post = clamp_stats(roll_ep, roll_close)

print("\nRollout entire episode:")

for j, name in enumerate(ARM_JOINTS):
    frac, p95, mx = roll_stats_all[j]

    print(
        f"{name:15s}: "
        f">3° = {frac:6.1f}% | "
        f"p95={p95:6.2f}° | max={mx:6.2f}°"
    )

print("\nRollout AFTER grasp:")

for j, name in enumerate(ARM_JOINTS):
    frac, p95, mx = roll_stats_post[j]

    print(
        f"{name:15s}: "
        f">3° = {frac:6.1f}% | "
        f"p95={p95:6.2f}° | max={mx:6.2f}°"
    )


# ------------------------------------------------------------
# Training reference
# ------------------------------------------------------------

print("\n[5] TRAINING REFERENCE AFTER GRASP")

train_errors = [[] for _ in range(5)]

for ep, close_idx in train_close.items():

    g = (
        train[train["episode_index"] == ep]
        .sort_values("frame_index")
        .reset_index(drop=True)
    )

    a = stack(g["action"])[close_idx:]
    s = stack(g["observation.state"])[close_idx:]

    for j in range(5):

        err = a[:, j] - s[:, j]

        if j == 4:
            err = (err + 180) % 360 - 180

        train_errors[j].extend(np.abs(err))


for j, name in enumerate(ARM_JOINTS):

    ae = np.asarray(train_errors[j])

    print(
        f"{name:15s}: "
        f">3° = {np.mean(ae > 3)*100:6.1f}% | "
        f"p95={np.percentile(ae,95):6.2f}° | "
        f"max={np.max(ae):6.2f}°"
    )


# ------------------------------------------------------------
# Post-grasp span
# ------------------------------------------------------------

print("\n[6] ROLLOUT POST-GRASP ACTION SPAN")

post = roll_a[roll_close:]

for j, name in enumerate(JOINTS):

    span = np.ptp(post[:, j])

    print(f"{name:15s}: {span:7.2f}")


print("\n" + "=" * 100)
print("HOW TO INTERPRET")
print("=" * 100)

print("""
A) movement ratio << 1 AND direction cosine low/negative
   -> policy itself is not producing the demonstrated transport trajectory.

B) rollout ACTION resembles training transport,
   but actual STATE does not follow it
   -> execution / safety clamp / motor tracking issue.

C) many post-grasp frames have |action-state| > 3 degrees
   -> max_relative_target=3.0 may be materially clipping the trajectory.

D) policy initially follows training transport but later turns back
   -> likely action-chunk / replanning / post-grasp distribution-shift issue.
""")
