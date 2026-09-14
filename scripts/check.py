"""Generic SO-101 dataset, rollout, and post-grasp checks. See --help."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from dataset_utils import (JOINTS, PROJECT, delta, first_closed, gripper_levels,
                           load_dataset, probe_video, require_finite, stack, tracking)


def quality_tables(dataset, max_lag):
    summaries, tracks, ranges = [], [], []
    for ep, group in dataset.episodes():
        action, state = stack(group.action), stack(group['observation.state'])
        ts = group.timestamp.to_numpy(dtype=float)
        dt = np.diff(ts)
        median_dt = np.median(dt) if len(dt) else np.nan
        valid = np.isfinite(action).all() and np.isfinite(state).all()
        row = dict(episode=int(ep), frames=len(group), duration_s=len(group) / dataset.fps,
                   timestamp_duration_s=ts[-1] - ts[0],
                   measured_fps=1 / median_dt if median_dt > 0 else np.nan,
                   max_timestamp_gap_ms=dt.max() * 1000 if len(dt) else np.nan,
                   frame_index_OK=len(group) > 1 and bool((np.diff(group.frame_index) == 1).all()),
                   timestamp_OK=len(ts) > 1 and bool(np.isfinite(ts).all() and (dt > 0).all()),
                   action_nan=int(np.isnan(action).sum()), action_inf=int(np.isinf(action).sum()),
                   state_nan=int(np.isnan(state).sum()), state_inf=int(np.isinf(state).sum()))
        if valid:
            da, ds = np.abs(delta(action[1:], action[:-1])), np.abs(delta(state[1:], state[:-1]))
            errors = np.abs(delta(action, state))
            row.update(total_action_motion_deg=da.sum(), max_action_jump_deg=da.max() if da.size else 0.,
                       max_state_jump_deg=ds.max() if ds.size else 0., mean_track_err=errors.mean(),
                       p95_track_err=np.percentile(errors, 95), max_track_err=errors.max())
            tracks.extend(dict(episode=int(ep), **t) for t in tracking(action, state, dataset.fps, max_lag))
            for j, joint in enumerate(JOINTS):
                ranges.append(dict(episode=int(ep), joint=joint, action_min=action[:, j].min(),
                                   action_max=action[:, j].max(), state_min=state[:, j].min(),
                                   state_max=state[:, j].max()))
        else:
            row.update(total_action_motion_deg=np.nan, max_action_jump_deg=np.nan, max_state_jump_deg=np.nan)
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    finite_motion = summary.total_action_motion_deg.dropna()
    median = finite_motion.median() if len(finite_motion) else np.nan
    summary['motion_vs_median'] = summary.total_action_motion_deg / median if median > 0 else np.nan
    return {'quality_episode_summary': summary,
            'quality_tracking': pd.DataFrame(tracks, columns=['episode', 'joint', 'best_lag_frames', 'best_lag_ms',
                'MAE_deg', 'median_bias_deg', 'p95_deg', 'max_deg']),
            'joint_ranges': pd.DataFrame(ranges, columns=['episode', 'joint', 'action_min', 'action_max', 'state_min', 'state_max'])}


def check_quality(dataset, args, warnings):
    info, frames = dataset.info, dataset.frames
    if len(frames) != info.get('total_frames'):
        warnings.append(f"Frame count: metadata={info.get('total_frames')}, Parquet={len(frames)}")
    episodes = sorted(frames.episode_index.unique().tolist())
    if episodes != list(range(info.get('total_episodes', 0))):
        warnings.append('Episode indices do not match metadata total_episodes / contiguous indexing')
    if args.expected_episodes is not None and len(episodes) != args.expected_episodes:
        warnings.append(f'Expected {args.expected_episodes} episodes, found {len(episodes)}')
    required = {'action', 'observation.state', 'timestamp', 'frame_index', 'episode_index'}
    missing = required - set(info.get('features', {}))
    if missing:
        warnings.append(f'Metadata features missing: {sorted(missing)}')
    tables = quality_tables(dataset, args.max_lag)
    for row in tables['quality_episode_summary'].to_dict('records'):
        ep = row['episode']
        checks = [
            (row['duration_s'] < args.min_duration, 'short episode'),
            (row['duration_s'] > args.episode_limit - .5, 'near/above recording time limit'),
            (not row['frame_index_OK'], 'frame index discontinuity or single-frame episode'),
            (not row['timestamp_OK'], 'invalid/non-increasing timestamps or single-frame episode'),
            (not .95 * dataset.fps <= row['measured_fps'] <= 1.05 * dataset.fps, 'unusual measured FPS'),
            (row['max_timestamp_gap_ms'] > 100, 'timestamp gap above 100 ms'),
            (sum(row[k] for k in ('action_nan', 'action_inf', 'state_nan', 'state_inf')) > 0,
             'NaN/Inf in action/state; motion and tracking metrics skipped'),
            (row['total_action_motion_deg'] == 0, 'no action movement'),
            (row['motion_vs_median'] < .6 or row['motion_vs_median'] > 1.7, 'unusual motion amount'),
            (row['max_state_jump_deg'] > 15, 'state jump above 15; inspect video'),
        ]
        warnings.extend(f'Episode {ep}: {message}' for condition, message in checks if condition)
    for row in tables['quality_tracking'].to_dict('records'):
        if row['MAE_deg'] > (8 if row['joint'] == 'gripper' else 5):
            warnings.append(f"Episode {row['episode']}: high tracking MAE on {row['joint']}")
        if row['joint'] == 'wrist_roll' and abs(row['median_bias_deg']) > 3:
            warnings.append(f"Episode {row['episode']}: wrist_roll bias above 3")
    return tables


def check_videos(dataset, warnings):
    rows = []
    if not dataset.cameras:
        warnings.append('No video cameras declared in metadata')
    for camera in dataset.cameras:
        total = 0
        files = sorted((dataset.root / 'videos' / camera).rglob('*.mp4'))
        if not files:
            warnings.append(f'{camera}: missing video files')
        for path in files:
            stream = probe_video(path, count_frames=True)['streams'][0]
            count = int(stream['nb_read_frames'])
            fps = float(Fraction(stream['avg_frame_rate']))
            total += count
            rows.append(dict(camera=camera, file=str(path.relative_to(dataset.root)), frames=count,
                             fps=fps, width=stream['width'], height=stream['height']))
            if abs(fps - dataset.fps) > .01:
                warnings.append(f'{path.name}: video FPS {fps} != metadata {dataset.fps}')
            shape = dataset.info['features'][camera].get('shape')
            if shape and list(shape[:2]) != [stream['height'], stream['width']]:
                warnings.append(f'{camera}/{path.name}: resolution differs from metadata')
        if total != len(dataset.frames):
            warnings.append(f'{camera}: {total} video frames != {len(dataset.frames)} Parquet rows')
    return pd.DataFrame(rows)


def check_rollout(roll, train, args, warnings):
    starts, spans, ids = [], [], []
    for ep, group in train.episodes():
        ids.append(int(ep))
        starts.append(stack(group['observation.state'])[:10].mean(axis=0))
        spans.append(np.ptp(stack(group.action), axis=0))
    starts, median_span = np.stack(starts), np.median(spans, axis=0)
    median_duration = np.median([len(g) / train.fps for _, g in train.episodes()])
    opened, closed = gripper_levels(train)
    summaries, comparison, tracks, traces, transitions, jumps = [], [], [], [], [], []
    for ep, group in roll.episodes(args.episode):
        action, state = stack(group.action), stack(group['observation.state'])
        if len(group) / roll.fps < .6 * median_duration:
            warnings.append(f'Episode {ep}: rollout duration below 60% of median demonstration duration; inspect completeness')
        error = np.abs(delta(starts, state[:10].mean(axis=0)))
        nearest = int(np.argmin(error.mean(axis=1)))
        classes = np.abs(action[:, 5] - opened) > np.abs(action[:, 5] - closed)
        state_classes = np.abs(state[:, 5] - opened) > np.abs(state[:, 5] - closed)
        summaries.append(dict(episode=int(ep), frames=len(group), duration_s=len(group) / roll.fps,
                              nearest_train_episode=ids[nearest], initial_mean_difference=error[nearest].mean(),
                              gripper_open=opened, gripper_closed=closed))
        for j, joint in enumerate(JOINTS):
            span = np.ptp(action[:, j])
            comparison.append(dict(episode=int(ep), joint=joint, initial_difference=error[nearest, j],
                                   action_min=action[:, j].min(), action_max=action[:, j].max(),
                                   state_min=state[:, j].min(), state_max=state[:, j].max(),
                                   action_span=span, state_span=np.ptp(state[:, j]),
                                   train_median_span=median_span[j],
                                   span_ratio=span / median_span[j] if median_span[j] > 1e-6 else np.nan))
        tracks.extend(dict(episode=int(ep), **r) for r in tracking(action, state, roll.fps, args.max_lag))
        trace = pd.DataFrame(dict(episode=int(ep), frame_index=group.frame_index.to_numpy(),
                                 timestamp=group.timestamp.to_numpy(), time_s=np.arange(len(group)) / roll.fps,
                                 policy_gripper=np.where(classes, 'CLOSED', 'OPEN'),
                                 state_gripper=np.where(state_classes, 'CLOSED', 'OPEN')))
        for j, joint in enumerate(JOINTS):
            trace[f'action_{joint}'], trace[f'state_{joint}'] = action[:, j], state[:, j]
        traces.append(trace)
        for i in np.flatnonzero(classes[1:] != classes[:-1]) + 1:
            transitions.append(dict(episode=int(ep), time_s=i / roll.fps,
                                    before='CLOSED' if classes[i - 1] else 'OPEN',
                                    after='CLOSED' if classes[i] else 'OPEN', action=action[i, 5]))
        for i in np.argsort(np.abs(np.diff(action[:, 5])))[-10:][::-1]:
            jumps.append(dict(episode=int(ep), time_s=i / roll.fps,
                              next_time_s=(i + 1) / roll.fps, delta=action[i + 1, 5] - action[i, 5]))
    return {'rollout_summary': pd.DataFrame(summaries), 'rollout_comparison': pd.DataFrame(comparison),
            'rollout_tracking': pd.DataFrame(tracks), 'rollout_trace': pd.concat(traces, ignore_index=True),
            'gripper_transitions': pd.DataFrame(transitions, columns=['episode', 'time_s', 'before', 'after', 'action']),
            'gripper_jumps': pd.DataFrame(jumps, columns=['episode', 'time_s', 'next_time_s', 'delta'])}


def pressure_rows(action, state, clamp, **labels):
    error = np.abs(delta(action, state))
    return [dict(**labels, joint=joint, clamp=clamp, above_clamp_pct=(error[:, j] > clamp).mean() * 100,
                 p95_deg=np.percentile(error[:, j], 95), max_deg=error[:, j].max())
            for j, joint in enumerate(JOINTS[:5])]


def check_transport(roll, train, args, warnings):
    opened, closed = gripper_levels(train)
    references = []
    for ep, group in train.episodes():
        a, s = stack(group.action), stack(group['observation.state'])
        close = first_closed(a, opened, closed, args.stable_frames)
        if close is not None:
            references.append((int(ep), a, s, close))
    if not references:
        warnings.append('No stable gripper closure detected in any training episode; transport reference unavailable')
    timing, moves, pressure, spans = [], [], [], []
    for ep, _, _, close in references:
        timing.append(dict(source='train', episode=ep, closure_detected=True, close_time_s=close / train.fps))
    if references:
        a = np.concatenate([a[c:] for _, a, _, c in references])
        s = np.concatenate([s[c:] for _, _, s, c in references])
        pressure.extend(pressure_rows(a, s, args.clamp, source='train', episode=-1, period='post_close'))
    for ep, group in roll.episodes(args.episode):
        a, s = stack(group.action), stack(group['observation.state'])
        close = first_closed(a, opened, closed, args.stable_frames)
        timing.append(dict(source='rollout', episode=int(ep), closure_detected=close is not None,
                           close_time_s=close / roll.fps if close is not None else np.nan))
        pressure.extend(pressure_rows(a, s, args.clamp, source='rollout', episode=int(ep), period='entire'))
        if close is None:
            warnings.append(f'Episode {ep}: no stable gripper closure; post-close metrics skipped')
            continue
        pressure.extend(pressure_rows(a[close:], s[close:], args.clamp,
                                      source='rollout', episode=int(ep), period='post_close'))
        spans.extend(dict(episode=int(ep), joint=joint, post_close_action_span=np.ptp(a[close:, j]))
                     for j, joint in enumerate(JOINTS))
        for seconds in (1, 2, 3, 5, 8):
            step, train_step = round(seconds * roll.fps), round(seconds * train.fps)
            if close + step >= len(a):
                continue
            da, ds = delta(a[close + step], a[close])[:5], delta(s[close + step], s[close])[:5]
            reference = [delta(ta[c + train_step], ta[c])[:5] for _, ta, _, c in references
                         if c + train_step < len(ta)]
            median = np.median(reference, axis=0) if reference else np.full(5, np.nan)
            nr, nt = np.linalg.norm(da), np.linalg.norm(median)
            ratio = nr / nt if nt > 1e-6 else np.nan
            cosine = np.dot(da, median) / (nr * nt) if nr > 1e-6 and nt > 1e-6 else np.nan
            for j, joint in enumerate(JOINTS[:5]):
                moves.append(dict(episode=int(ep), seconds_after_close=seconds, joint=joint,
                                  action_delta=da[j], state_delta=ds[j], train_median_delta=median[j],
                                  train_reference_count=len(reference), movement_ratio=ratio, direction_cosine=cosine))
    return {'closure_timing': pd.DataFrame(timing),
            'transport_movement': pd.DataFrame(moves, columns=['episode', 'seconds_after_close', 'joint', 'action_delta',
                'state_delta', 'train_median_delta', 'train_reference_count', 'movement_ratio', 'direction_cosine']),
            'clamp_pressure': pd.DataFrame(pressure),
            'post_close_spans': pd.DataFrame(spans, columns=['episode', 'joint', 'post_close_action_span'])}


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='mode', required=True)
    for mode, description in [('dataset', 'Check demonstration or rollout dataset quality'),
                              ('rollout', 'Compare each rollout episode with training demonstrations'),
                              ('transport', 'Analyze post-closure movement and a supplied clamp threshold')]:
        sub = commands.add_parser(mode, help=description)
        sub.add_argument('root', type=Path, help='LeRobot dataset root (contains meta/info.json)')
        sub.add_argument('--output-dir', type=Path, help='New/empty report directory; default: logs/checks/<dataset>/<unique-run>')
        sub.add_argument('--strict', action='store_true', help='Return exit status 1 if warnings are found')
        if mode != 'transport':
            sub.add_argument('--max-lag', type=nonnegative, default=8)
            sub.add_argument('--skip-video', action='store_true', help='Skip ffprobe frame-count checks')
        if mode == 'dataset':
            sub.add_argument('--expected-episodes', type=nonnegative, help='Optional independent episode count expectation')
            sub.add_argument('--min-duration', type=positive, default=8)
            sub.add_argument('--episode-limit', type=positive, default=30)
        else:
            sub.add_argument('--train', type=Path, required=True, help='Reference demonstration dataset')
            sub.add_argument('--episode', type=nonnegative, help='Only analyze this episode; default: all')
        if mode == 'transport':
            sub.add_argument('--clamp', type=positive, required=True, help='Analysis threshold in recorded joint units; does not change robot settings')
            sub.add_argument('--stable-frames', type=int, default=5)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.mode == 'transport' and args.stable_frames < 1:
        cli.error('--stable-frames must be positive')
    try:
        if args.output_dir:
            args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir and args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
            raise ValueError('--output-dir must be new or empty; existing reports will not be overwritten')
        dataset = load_dataset(args.root)
        warnings = []
        if args.mode == 'dataset':
            tables = check_quality(dataset, args, warnings)
        else:
            train = load_dataset(args.train)
            require_finite(dataset)
            require_finite(train)
            tables = (check_rollout if args.mode == 'rollout' else check_transport)(dataset, train, args, warnings)
        if args.mode != 'transport' and not args.skip_video:
            # Check the whole dataset's videos, even when one rollout episode was selected.
            tables['video_integrity'] = check_videos(dataset, warnings)
        if args.output_dir:
            output = args.output_dir.expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
        else:
            base = PROJECT / 'logs/checks' / dataset.root.name
            base.mkdir(parents=True, exist_ok=True)
            prefix = args.mode + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-')
            output = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
        for name, table in tables.items():
            table.to_csv(output / (name + '.csv'), index=False)
            print(f'{name}: {len(table)} rows')
        report = dict(mode=args.mode, dataset=str(dataset.root), fps=dataset.fps,
                      arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      warnings=warnings, output=str(output))
        (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        for warning in warnings:
            print('WARNING:', warning)
        if args.mode != 'dataset':
            print('Gripper inference assumes training episodes start open. Closure does not establish task success.')
        print(f'Reports saved: {output}')
        return 1 if args.strict and warnings else 0
    except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError) as exc:
        cli.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
