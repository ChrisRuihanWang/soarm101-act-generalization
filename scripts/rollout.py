"""Launch recorded ACT evaluation with an explicit ROBOT_PORT and stage profile."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / 'configs/rollout.json'


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage', choices=['stage2', 'stage3', 'grid-retest'])
    cli.add_argument('root', type=Path, help='New rollout dataset directory; existing data are never overwritten')
    cli.add_argument('repo_id', help='Dataset identifier: OWNER/rollout_NAME (automatic Hub upload is disabled)')
    cli.add_argument('--num-episodes', type=positive_int, default=9, help='Episodes in this run (default: 9)')
    cli.add_argument('--policy-path', type=Path, help='Override the selected local pretrained_model directory')
    cli.add_argument('--ablation-25', action='store_true', help='Explicitly select the separate 25-step inference ablation')
    cli.add_argument('--dry-run', action='store_true', help='Print command/settings without loading LeRobot, opening devices, or writing files')
    return cli


def build_command(args, environ=None):
    environ = os.environ if environ is None else environ
    port = environ.get('ROBOT_PORT', '').strip()
    if not port:
        raise ValueError('Set ROBOT_PORT to the connected follower device; there is no default port')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/rollout_[A-Za-z0-9_.-]+', args.repo_id):
        raise ValueError('repo_id must have the form OWNER/rollout_NAME')
    config = json.loads(CONFIG.read_text())
    policy_stage = 'stage2' if args.stage == 'stage2' else 'stage3'
    policy = (args.policy_path or PROJECT / config['policies'][policy_stage]).expanduser().resolve()
    root = args.root.expanduser().resolve()
    if root.exists():
        raise ValueError(f'Dataset already exists: {root}; choose a new directory')
    action_steps = config['ablation']['n_action_steps'] if args.ablation_25 else config['n_action_steps']
    mode = 'ablation_n25' if args.ablation_25 else 'main_n100'
    # Keep ablation recordings visibly distinct even if the caller reuses a main-style name.
    if args.ablation_25 and ('ablation' not in root.name.lower() or 'ablation' not in args.repo_id.lower()):
        raise ValueError('For --ablation-25, include "ablation" in both the dataset directory name and repo_id')
    robot = config['robot']
    options = {
        'strategy.type': 'episodic',
        'inference.type': 'sync',
        'policy.path': str(policy),
        'policy.n_action_steps': action_steps,
        'robot.type': robot['type'],
        'robot.id': robot['id'],
        'robot.port': port,
        'robot.max_relative_target': robot['max_relative_target'],
        'robot.cameras': json.dumps(robot['cameras'], separators=(',', ':')),
        'fps': config['fps'],
        'dataset.fps': config['fps'],
        'dataset.root': str(root),
        'dataset.repo_id': args.repo_id,
        'dataset.single_task': config['tasks'][args.stage],
        'dataset.num_episodes': args.num_episodes,
        'dataset.episode_time_s': config['episode_time_s'],
        'dataset.reset_time_s': config['reset_time_s'],
        'dataset.push_to_hub': False,
        'dataset.no_stamp': True,
        'display_data': config['display_data'],
        'resume': False,
    }
    command = [sys.executable, '-m', 'lerobot.scripts.lerobot_rollout']
    command += [f'--{key}={str(value).lower() if isinstance(value, bool) else value}' for key, value in options.items()]
    return command, dict(stage=args.stage, evaluation_mode=mode, options=options)


def preflight(settings):
    """Only called for execution; never connect to hardware during validation."""
    options = settings['options']
    policy = Path(options['policy.path'])
    for name in ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        if not (policy / name).is_file():
            raise ValueError(f'Missing policy artifact: {policy / name}')
    config = json.loads((policy / 'config.json').read_text())
    if config.get('type') != 'act' or options['policy.n_action_steps'] > config.get('chunk_size', 0):
        raise ValueError('Expected an ACT checkpoint with chunk_size >= n_action_steps')
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        for step in json.loads((policy / name).read_text()).get('steps', []):
            if step.get('state_file') and not (policy / step['state_file']).is_file():
                raise ValueError(f"Missing processor state: {step['state_file']}")
    devices = [options['robot.port']] + [v['index_or_path'] for v in json.loads(options['robot.cameras']).values()]
    for device in devices:
        if not os.access(device, os.R_OK | os.W_OK):
            raise ValueError(f'Device missing or inaccessible: {device}')


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        command, settings = build_command(args)
        print(f"Evaluation mode: {settings['evaluation_mode']}", flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        preflight(settings)
        base = PROJECT / 'logs/rollouts'
        base.mkdir(parents=True, exist_ok=True)
        prefix = args.stage + '-' + settings['evaluation_mode'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-')
        log_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
        record = dict(settings, command=command, log_dir=str(log_dir))
        record_path = log_dir / 'run.json'
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        print(f'Run settings and output: {log_dir}', flush=True)
        with (log_dir / 'rollout.log').open('w') as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
        record['exit_code'] = return_code
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        return return_code if return_code >= 0 else 128 - return_code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        cli.exit(2, f'Error: {exc}\n')
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
