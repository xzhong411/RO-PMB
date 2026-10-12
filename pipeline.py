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
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--segmentation-gpus')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    cfg_path = Path(args.config).resolve() if args.config else root / 'configs' / (args.dataset + '.json')
    cfg = json.loads(cfg_path.read_text())
    if args.dataset == 'coco' and not (root / 'manifests/coco_ground_truth_provenance.json').is_file():
        raise RuntimeError('Run prepare_coco_gt.py before the full COCO experiment')
    output = Path(args.output).resolve() if args.output else root / 'runs' / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
    world = len(args.gpus.split(','))
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}']
    segmentation_gpus = args.segmentation_gpus or args.gpus
    segmentation_world = len(segmentation_gpus.split(','))
    segmentation_distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={segmentation_world}']
    python = [sys.executable]
    state_path = output / 'status.json'
    state = {'dataset': args.dataset, 'pid': os.getpid(), 'gpu_ids': args.gpus, 'started': time.time(), 'completed_stages': []}
    if args.resume and state_path.exists():
        old = json.loads(state_path.read_text())
        state['completed_stages'] = old.get('completed_stages', [])
    stages = [
        ('cam_training', distributed + ['train.py', '--config', str(cfg_path), '--output', str(output / 'cam')]),
        ('pseudo_labels', python + ['infer_cams.py', '--checkpoint', str(output / 'cam/last.pt'), '--manifest', cfg['train_manifest'], '--output', str(output / 'activation')]),
        ('cam_evaluation', python + ['evaluate.py', '--manifest', cfg.get('train_eval_manifest', cfg['train_manifest']), '--predictions', str(output / 'activation/raw_masks'), '--classes', str(cfg['classes']), '--output', str(output / 'cam_metrics.json')]),
        ('pseudo_label_evaluation', python + ['evaluate.py', '--manifest', cfg.get('train_eval_manifest', cfg['train_manifest']), '--predictions', str(output / 'activation/pseudo_labels'), '--classes', str(cfg['classes']), '--output', str(output / 'pseudo_label_metrics.json')]),
        ('pseudo_label_health', python + ['check_pseudo_health.py', '--predictions', str(output / 'activation/pseudo_labels'), '--output', str(output / 'pseudo_label_health.json'), '--max-foreground-fraction', '0.98']),
        ('segmentation_training', segmentation_distributed + ['train_seg.py', '--config', str(cfg_path), '--pseudo-labels', str(output / 'activation/pseudo_labels'), '--output', str(output / 'segmentation')]),
        ('segmentation_inference', python + ['infer_seg.py', '--checkpoint', str(output / 'segmentation/last.pt'), '--manifest', cfg['val_manifest'], '--output', str(output / 'val_predictions')]),
        ('segmentation_evaluation', python + ['evaluate.py', '--manifest', cfg['val_manifest'], '--predictions', str(output / 'val_predictions'), '--classes', str(cfg['classes']), '--output', str(output / 'val_metrics.json')])
    ]
    for name, command in stages:
        if name in state['completed_stages']:
            continue
        if args.resume and name in ['cam_training', 'segmentation_training']:
            checkpoint = output / ('cam' if name == 'cam_training' else 'segmentation') / 'last.pt'
            if checkpoint.exists():
                command.extend(['--resume', str(checkpoint)])
        state.update({'stage': name, 'status': 'running', 'updated': time.time(), 'command': command})
        state_path.write_text(json.dumps(state, indent=2))
        print(json.dumps({'stage': name, 'command': command}), flush=True)
        with (output / (name + '.log')).open('a') as log:
            stage_environment = dict(environment, CUDA_VISIBLE_DEVICES=segmentation_gpus) if name.startswith('segmentation_') else environment
            result = subprocess.run(command, env=stage_environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state.update({'status': 'failed', 'exit_code': result.returncode, 'updated': time.time()})
            state_path.write_text(json.dumps(state, indent=2))
            raise SystemExit(result.returncode)
        state['completed_stages'].append(name)
    state.update({'stage': 'complete', 'status': 'complete', 'updated': time.time()})
    state_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
