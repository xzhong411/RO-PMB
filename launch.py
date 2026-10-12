import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['voc', 'coco'], required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--segmentation-gpus')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve() if args.output else root / 'runs' / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    pid_file = output / 'pipeline.pid'
    if pid_file.exists():
        pid = int(pid_file.read_text())
        try:
            os.kill(pid, 0)
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'pipeline.py' in cmdline:
                raise RuntimeError(f'Pipeline already running as PID {pid}')
        except ProcessLookupError:
            pass
    command = [sys.executable, '-u', str(root / 'pipeline.py'), '--dataset', args.dataset, '--gpus', args.gpus]
    for key in ['config', 'output', 'segmentation_gpus']:
        value = getattr(args, key)
        if value:
            command.extend(['--' + key.replace('_', '-'), value])
    if args.resume:
        command.append('--resume')
    with (output / 'pipeline.log').open('a') as log:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_file.write_text(str(process.pid))
    record = {'pid': process.pid, 'dataset': args.dataset, 'gpus': args.gpus, 'started': time.time(), 'log': str(output / 'pipeline.log')}
    (output / 'launch.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
