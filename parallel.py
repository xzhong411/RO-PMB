import os
import subprocess
import sys
import torch
import torch.distributed as dist


def inference_device(single_process=False):
    world = int(os.environ.get('WORLD_SIZE', 1))
    if world == 1 and not single_process and 'CUDA_VISIBLE_DEVICES' in os.environ and torch.cuda.device_count() > 1:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={torch.cuda.device_count()}', *sys.argv, '--single-process']
        raise SystemExit(subprocess.run(command).returncode)
    rank, local = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return device, rank, world


def finish_inference(world):
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
