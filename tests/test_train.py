"""Exercise the shell entry point in an isolated repository with a fake trainer."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class TrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project with spaces'
        (self.root / 'scripts').mkdir(parents=True)
        shutil.copy2(PROJECT / 'scripts/train.sh', self.root / 'scripts/train.sh')
        self.configs = {}
        for stage, profile in [('stage1', 'stage1_fixed'), ('stage2', 'stage2_grid'),
                               ('stage3', 'stage3_mixed')]:
            source = PROJECT / 'configs' / profile / 'train_config.json'
            destination = self.root / 'configs' / profile / 'train_config.json'
            destination.parent.mkdir(parents=True)
            shutil.copy2(source, destination)
            config = json.loads(source.read_text())
            (self.root / config['dataset']['root']).mkdir(parents=True)
            self.configs[stage] = config
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        self.capture = self.root / 'captured.json'
        trainer = bin_dir / 'lerobot-train'
        trainer.write_text(
            '#!' + sys.executable + '\n'
            'import json, os, pathlib, sys\n'
            'pathlib.Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
            'sys.exit(int(os.environ.get("TRAIN_EXIT", "0")))\n'
        )
        trainer.chmod(0o755)
        (bin_dir / 'python').symlink_to(sys.executable)
        self.env = os.environ.copy()
        for key in ('STEPS', 'BATCH_SIZE', 'NUM_WORKERS', 'OUTPUT_DIR', 'RUN_NAME'):
            self.env.pop(key, None)
        self.env.update(PATH=str(bin_dir) + os.pathsep + self.env['PATH'], CAPTURE=str(self.capture))

    def run_script(self, *args, **overrides):
        return subprocess.run([str(self.root / 'scripts/train.sh'), *args],
                              cwd=self.temp.name, env=dict(self.env, **overrides),
                              capture_output=True, text=True)

    def test_each_stage_preserves_verified_defaults(self):
        for stage, config in self.configs.items():
            with self.subTest(stage=stage):
                result = self.run_script(stage)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(self.capture.read_text())
                self.assertIn('--dataset.repo_id=' + config['dataset']['repo_id'], args)
                self.assertIn('--dataset.root=' + str(self.root / config['dataset']['root']), args)
                self.assertIn('--output_dir=' + str(self.root / config['output_dir']), args)
                self.assertIn('--job_name=' + config['job_name'], args)
                self.assertIn('--steps=' + str(config['steps']), args)
                self.assertIn('--policy.device=cuda', args)
                self.assertIn('--resume=false', args)

    def test_overrides_and_trainer_failure_propagation(self):
        result = self.run_script('stage3', STEPS='60000', BATCH_SIZE='16', NUM_WORKERS='0',
                                 OUTPUT_DIR='outputs/new run', RUN_NAME='act_stage3_60k', TRAIN_EXIT='7')
        self.assertEqual(result.returncode, 7, result.stderr)
        args = json.loads(self.capture.read_text())
        for arg in ['--steps=60000', '--batch_size=16', '--num_workers=0',
                    '--job_name=act_stage3_60k', '--output_dir=' + str(self.root / 'outputs/new run')]:
            self.assertIn(arg, args)

    def test_existing_outputs_block_execution_but_allow_preview(self):
        directory = self.root / 'existing directory'
        directory.mkdir()
        file = self.root / 'existing file'
        file.write_text('preserve this')
        link = self.root / 'dangling link'
        link.symlink_to(self.root / 'missing target')
        for output in (directory, file, link):
            with self.subTest(output=output):
                result = self.run_script('stage2', OUTPUT_DIR=str(output))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('OUTPUT_DIR already exists', result.stderr)
                preview = self.run_script('stage2', '--dry-run', OUTPUT_DIR=str(output))
                self.assertEqual(preview.returncode, 0, preview.stderr)
                self.assertIn('actual training would be refused', preview.stdout)
                self.assertFalse(self.capture.exists())
        self.assertEqual(file.read_text(), 'preserve this')

    def test_missing_dataset_blocks_even_preview(self):
        (self.root / self.configs['stage1']['dataset']['root']).rmdir()
        for args in [('stage1',), ('stage1', '--dry-run')]:
            result = self.run_script(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('DATASET_ROOT does not exist', result.stderr)
        self.assertFalse(self.capture.exists())

    def test_invalid_inputs_never_invoke_trainer(self):
        for args in [(), ('stage4',), ('stage1', '--unknown'), ('stage1', '--dry-run', 'extra')]:
            self.assertNotEqual(self.run_script(*args).returncode, 0)
        for overrides in [dict(STEPS='0'), dict(STEPS='abc'), dict(BATCH_SIZE=''),
                          dict(NUM_WORKERS='-1'), dict(OUTPUT_DIR=''), dict(RUN_NAME=' ')]:
            self.assertNotEqual(self.run_script('stage1', **overrides).returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_dry_run_creates_no_output_and_never_invokes_trainer(self):
        result = self.run_script('stage3', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('STEPS=30000 BATCH_SIZE=8 NUM_WORKERS=4', result.stdout)
        self.assertFalse(self.capture.exists())
        self.assertFalse((self.root / 'outputs').exists())


if __name__ == '__main__':
    unittest.main()
