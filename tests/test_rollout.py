"""Rollout command tests; no LeRobot imports or hardware connections."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rollout


class RolloutTests(unittest.TestCase):
    def args(self, directory, stage='stage2', *extra):
        return rollout.parser().parse_args([stage, str(Path(directory) / 'new_rollout'),
                                            'test/rollout_evaluation', *extra])

    def test_confirmed_main_settings_for_all_profiles(self):
        with tempfile.TemporaryDirectory() as folder:
            for stage, port in [('stage2', '/dev/test-follower-1'), ('stage3', '/dev/test-follower-0'),
                                ('grid-retest', '/dev/by-id/follower')]:
                command, settings = rollout.build_command(self.args(folder, stage), {'ROBOT_PORT': port})
                options = settings['options']
                self.assertEqual(options['robot.port'], port)
                self.assertEqual(options['policy.n_action_steps'], 100)
                self.assertEqual(options['robot.type'], 'so101_follower')
                self.assertEqual(options['robot.max_relative_target'], 10.)
                self.assertEqual(options['fps'], 30)
                self.assertEqual(options['dataset.fps'], 30)
                self.assertEqual(options['dataset.episode_time_s'], 20)
                self.assertEqual(options['dataset.reset_time_s'], 30)
                self.assertFalse(options['display_data'])
                self.assertFalse(options['dataset.push_to_hub'])
                self.assertEqual(options['strategy.type'], 'episodic')
                self.assertIn('--policy.n_action_steps=100', command)
                self.assertEqual(settings['evaluation_mode'], 'main_n100')
                cameras = json.loads(options['robot.cameras'])
                for name, device in [('top', '/dev/video2'), ('side', '/dev/video4')]:
                    self.assertEqual(cameras[name], dict(type='opencv', index_or_path=device,
                        width=640, height=480, fps=30, fourcc='MJPG', backend=200))
                policy = 'act_grid5_t1_30k' if stage == 'stage2' else 'act_stage23_100ep_30k'
                self.assertIn(policy, options['policy.path'])
                self.assertFalse(any(arg.startswith('--teleop.') for arg in command))

    def test_robot_port_has_no_default(self):
        with tempfile.TemporaryDirectory() as folder:
            for env in ({}, {'ROBOT_PORT': ''}, {'ROBOT_PORT': '  '}):
                with self.assertRaisesRegex(ValueError, 'Set ROBOT_PORT'):
                    rollout.build_command(self.args(folder), env)

    def test_environment_cannot_silently_select_ablation(self):
        with tempfile.TemporaryDirectory() as folder:
            _, settings = rollout.build_command(self.args(folder), {'ROBOT_PORT': '/mock', 'N_ACTION_STEPS': '25'})
            self.assertEqual(settings['options']['policy.n_action_steps'], 100)

    def test_ablation_is_explicit_and_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder, 'stage3', '--ablation-25')
            with self.assertRaisesRegex(ValueError, 'include "ablation"'):
                rollout.build_command(args, {'ROBOT_PORT': '/mock'})
            args.root = Path(folder) / 'ablation_n25'
            args.repo_id = 'test/rollout_ablation_n25'
            command, settings = rollout.build_command(args, {'ROBOT_PORT': '/mock'})
            self.assertIn('--policy.n_action_steps=25', command)
            self.assertEqual(settings['evaluation_mode'], 'ablation_n25')

    def test_existing_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            args.root.mkdir()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                rollout.build_command(args, {'ROBOT_PORT': '/mock'})

    def test_policy_override_and_paths_with_spaces_remain_single_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            policy = Path(folder) / 'saved policy'
            args = self.args(folder, 'stage3', '--policy-path', str(policy), '--num-episodes', '5')
            args.root = Path(folder) / 'new data'
            command, settings = rollout.build_command(args, {'ROBOT_PORT': '/dev/mock port'})
            self.assertIn('--policy.path=' + str(policy), command)
            self.assertIn('--dataset.root=' + str(args.root), command)
            self.assertEqual(settings['options']['dataset.num_episodes'], 5)

    def test_dry_run_never_launches_or_preflights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'new'
            with patch.dict(os.environ, {'ROBOT_PORT': '/missing-device'}), \
                 patch.object(rollout, 'preflight') as preflight, \
                 patch.object(rollout.subprocess, 'Popen') as popen, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                result = rollout.main(['stage2', str(root), 'test/rollout_preview', '--dry-run'])
            self.assertEqual(result, 0)
            preflight.assert_not_called()
            popen.assert_not_called()
            self.assertFalse(root.exists())
            self.assertIn('--policy.n_action_steps=100', output.getvalue())

    def test_missing_policy_fails_before_process_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {'ROBOT_PORT': '/mock'}), \
                 patch.object(rollout.subprocess, 'Popen') as popen, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                rollout.main(['stage2', str(Path(folder) / 'new'), 'test/rollout_missing',
                              '--policy-path', str(Path(folder) / 'missing')])
            self.assertEqual(raised.exception.code, 2)
            popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
