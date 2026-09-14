"""Regression cases for episode isolation, metadata, and diagnostic edge cases."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import check
from dataset_utils import Dataset, delta, episode_video_segments, load_dataset, tracking


def dataset(episodes, fps=30):
    """Each item is (episode id, action matrix, state matrix)."""
    rows = []
    for ep, action, state in episodes:
        rows.extend(dict(episode_index=ep, frame_index=i, timestamp=i / fps,
                         action=a, **{'observation.state': s})
                    for i, (a, s) in enumerate(zip(action, state)))
    info = dict(fps=fps, total_frames=len(rows), total_episodes=len(episodes),
                features={key: {} for key in ('action', 'observation.state', 'timestamp',
                                              'frame_index', 'episode_index')})
    return Dataset(Path('/unused'), info, pd.DataFrame(rows))


def demonstration(n=60, fps=30):
    action = np.tile(np.arange(n, dtype=float)[:, None] / fps, (1, 6))
    action[:16, 5] = 10
    action[16:, 5] = 0
    return action


class AnalysisTests(unittest.TestCase):
    def test_recording_arguments_with_mock_command(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/record_fixed.sh'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / 'lerobot-record'
            executable.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            executable.chmod(0o755)
            device = root / 'mock device'
            device.touch()
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], LOG_DIR=str(root / 'logs'))
            env.update({key: str(device) for key in ('FOLLOWER_PORT', 'LEADER_PORT', 'TOP_CAMERA', 'SIDE_CAMERA')})
            target = root / 'new dataset'
            result = subprocess.run(['bash', str(script), str(target), 'test/fixed', '5', 'false'],
                                    env=env, text=True, capture_output=True, check=True)
            options = json.loads(result.stdout)
            self.assertIn('--dataset.root=' + str(target), options)
            self.assertIn('--dataset.num_episodes=5', options)
            cameras = json.loads(next(v.split('=', 1)[1] for v in options if v.startswith('--robot.cameras=')))
            self.assertEqual(cameras['top']['index_or_path'], str(device))
            target.mkdir()
            blocked = subprocess.run(['bash', str(script), str(target), 'test/fixed'], env=env, capture_output=True)
            self.assertNotEqual(blocked.returncode, 0)

    def test_recording_propagates_failure_through_tee(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/record_fixed.sh'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / 'lerobot-record'
            executable.write_text('#!/bin/sh\nexit 9\n')
            executable.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], LOG_DIR=str(root / 'logs'))
            env.update({key: '/dev/null' for key in ('FOLLOWER_PORT', 'LEADER_PORT', 'TOP_CAMERA', 'SIDE_CAMERA')})
            result = subprocess.run(['bash', str(script), str(root / 'new'), 'test/fixed'], env=env, capture_output=True)
            self.assertEqual(result.returncode, 9)

    def test_wrist_wrap_and_short_episode(self):
        a, s = np.zeros((1, 6)), np.zeros((1, 6))
        a[0, 4], s[0, 4] = -179, 179
        self.assertEqual(delta(a, s)[0, 4], 2)
        rows = tracking(a, s, 20, max_lag=8)
        self.assertEqual(rows[4]['MAE_deg'], 2)
        self.assertTrue(all(row['best_lag_frames'] == 0 for row in rows))

    def test_known_tracking_lag(self):
        a = np.tile(np.arange(10.)[:, None], (1, 6))
        s = np.vstack([np.zeros((2, 6)), a[:-2]])
        rows = tracking(a, s, fps=20)
        self.assertTrue(all(r['best_lag_frames'] == 2 and r['MAE_deg'] == 0 for r in rows))
        self.assertEqual(rows[0]['best_lag_ms'], 100)

    def test_rollout_episodes_are_separate_and_ids_preserved(self):
        a = demonstration()
        b = a.copy()
        b[:, :5] += 50
        train = dataset([(5, a, a), (9, b, b)])
        roll = dataset([(2, b, b), (7, a, a)])
        result = check.check_rollout(roll, train, SimpleNamespace(episode=None, max_lag=8), [])
        summary = result['rollout_summary']
        self.assertEqual(summary.nearest_train_episode.tolist(), [9, 5])
        self.assertEqual(len(result['rollout_tracking']), 12)
        self.assertTrue((result['rollout_tracking'].MAE_deg == 0).all())
        self.assertEqual(result['rollout_trace'].groupby('episode').time_s.min().tolist(), [0., 0.])

    def test_unknown_episode_is_rejected(self):
        a = demonstration()
        with self.assertRaisesRegex(ValueError, 'Episode 100 not found'):
            list(dataset([(5, a, a)]).episodes(100))

    def test_nonfinite_data_are_reported_and_tracking_skipped(self):
        a = demonstration()
        a[4, 0] = np.nan
        d = dataset([(0, a, a)])
        warnings = []
        args = SimpleNamespace(expected_episodes=None, max_lag=8, min_duration=1, episode_limit=30)
        with np.errstate(all='ignore'):
            result = check.check_quality(d, args, warnings)
        self.assertEqual(result['quality_episode_summary'].action_nan.iloc[0], 1)
        self.assertTrue(result['quality_tracking'].empty)
        self.assertTrue(any('NaN/Inf' in w for w in warnings))

    def test_fps_is_from_metadata(self):
        a = demonstration(60, fps=20)
        result = check.quality_tables(dataset([(0, a, a)], fps=20), 8)
        row = result['quality_episode_summary'].iloc[0]
        self.assertEqual(row.duration_s, 3)
        self.assertAlmostEqual(row.measured_fps, 20)

    def test_transport_uses_each_dataset_fps_and_supplied_clamp(self):
        ta, ra = demonstration(80, 4), demonstration(50, 2)
        train, roll = dataset([(0, ta, ta)], 4), dataset([(0, ra, ra - 5)], 2)
        for clamp, percentage in [(3, 100), (10, 0)]:
            args = SimpleNamespace(episode=None, stable_frames=5, clamp=clamp)
            result = check.check_transport(roll, train, args, [])
            pressures = result['clamp_pressure'].query("source == 'rollout'")
            self.assertTrue((pressures.above_clamp_pct == percentage).all())
            movement = result['transport_movement'].query('seconds_after_close == 1')
            np.testing.assert_allclose(movement.action_delta, 1)
            np.testing.assert_allclose(movement.train_median_delta, 1)

    def test_no_closure_is_a_reportable_outcome(self):
        train_a = demonstration()
        roll_a = np.full((10, 6), 10.)
        args = SimpleNamespace(episode=None, stable_frames=5, clamp=10)
        warnings = []
        result = check.check_transport(dataset([(0, roll_a, roll_a)]),
                                       dataset([(0, train_a, train_a)]), args, warnings)
        self.assertTrue(result['transport_movement'].empty)
        self.assertTrue(any('no stable gripper closure' in w for w in warnings))
        self.assertFalse(result['closure_timing'].query("source == 'rollout'").closure_detected.iloc[0])

    def test_no_training_closure_does_not_crash(self):
        # Isolated closed samples provide distinct levels but no stable closure.
        a = np.full((60, 6), 10.)
        a[20::2, 5] = 0
        b = demonstration()
        warnings = []
        result = check.check_transport(dataset([(0, b, b)]), dataset([(0, a, a)]),
                                       SimpleNamespace(episode=None, stable_frames=5, clamp=3), warnings)
        self.assertTrue(any('training episode' in w for w in warnings))
        self.assertEqual(result['closure_timing'].source.tolist(), ['rollout'])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / 'report.json'
            sentinel.write_text('keep me')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                check.main(['dataset', '/missing', '--output-dir', folder])
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual(sentinel.read_text(), 'keep me')

    def test_real_reader_and_strict_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'dataset'
            (root / 'meta').mkdir(parents=True)
            (root / 'data').mkdir()
            a = demonstration()
            d = dataset([(0, a, a)])
            (root / 'meta/info.json').write_text(json.dumps(d.info))
            d.frames.to_parquet(root / 'data/test.parquet')
            self.assertEqual(load_dataset(root).fps, 30)
            out = Path(folder) / 'report'
            with contextlib.redirect_stdout(io.StringIO()):
                status = check.main(['dataset', str(root), '--skip-video', '--expected-episodes', '2',
                                     '--strict', '--output-dir', str(out)])
            self.assertEqual(status, 1)
            self.assertTrue(any('Expected 2' in w for w in json.loads((out / 'report.json').read_text())['warnings']))

    def test_video_uses_metadata_and_handles_file_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            key = 'observation.images.top'
            video_dir = root / 'videos' / key / 'chunk-000'
            video_dir.mkdir(parents=True)
            paths = [video_dir / f'file-{i:03d}.mp4' for i in range(3)]
            for p in paths:
                p.touch()
            meta_dir = root / 'meta/episodes'
            meta_dir.mkdir(parents=True)
            prefix = f'videos/{key}/'
            row = {'episode_index': 5, prefix + 'chunk_index': 0, prefix + 'file_index': 1,
                   prefix + 'from_timestamp': 8., prefix + 'to_timestamp': 13.}
            pd.DataFrame([row]).to_parquet(meta_dir / 'episodes.parquet')
            a = demonstration()
            d = dataset([(5, a, a)])
            d.root = root
            d.info.update(features={key: {'dtype': 'video'}},
                          video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
            with patch('dataset_utils.probe_video', return_value={'format': {'duration': '10'}}):
                segments = episode_video_segments(d, 5, 'top')
            self.assertEqual(segments, [(paths[1], 8., 2.), (paths[2], 0., 3.)])


if __name__ == '__main__':
    unittest.main()
