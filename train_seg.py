import argparse
import json
import os
import random
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from data import ManifestDataset
from segmentation import SegmentationNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pseudo-labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    world, rank, local = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    dataset = ManifestDataset(cfg['train_manifest'], cfg, training=True, mask_root=args.pseudo_labels, limit=args.limit)
    missing = [r['id'] for r in dataset.rows if not (Path(args.pseudo_labels) / (r['id'] + '.png')).is_file()]
    if missing:
        raise RuntimeError(f'{len(missing)} missing pseudo labels')
    if cfg['batch_size'] % world:
        raise ValueError('Global batch size must be divisible by world size')
    sampler = DistributedSampler(dataset, world, rank, seed=cfg['seed'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'] // world, sampler=sampler, drop_last=True, num_workers=cfg['workers'], pin_memory=True)
    if not len(loader):
        raise RuntimeError('Empty segmentation loader')
    model = SegmentationNet(cfg['classes']).to(device)
    pretrained = torch.load(cfg['segmentation_weights'], map_location='cpu')
    own = model.state_dict()
    compatible = {k: v for k, v in pretrained.items() if k in own and v.shape == own[k].shape and not k.startswith('fc8')}
    if len(compatible) < 50:
        raise RuntimeError('Incompatible ResNet38 pretrained weights')
    model.load_state_dict(compatible, strict=False)
    model.train()
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('fc8')]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('fc8')]
    lr = cfg['segmentation_learning_rate']
    optimizer = torch.optim.SGD([{'params': backbone, 'lr': lr}, {'params': head, 'lr': lr * 10}], momentum=0.9, weight_decay=cfg['segmentation_weight_decay'])
    epoch_start, step = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if state.get('partial_epoch') or state['config'] != cfg:
            raise ValueError('Cannot resume a partial checkpoint or changed configuration')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_start, step = state['epoch'] + 1, state['step']
    bare = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    total_steps = cfg['segmentation_epochs'] * len(loader)
    for epoch in range(epoch_start, cfg['segmentation_epochs']):
        sampler.set_epoch(epoch)
        random.seed(cfg['seed'] + epoch * world + rank)
        model.train()
        for pack in loader:
            image, target = pack['image'].to(device), pack['mask'].to(device)
            if not ((target <= cfg['classes']) | (target == 255)).all():
                raise ValueError('Invalid pseudo label')
            logits = F.interpolate(model(image), target.shape[-2:], mode='bilinear', align_corners=False)
            loss = F.cross_entropy(logits, target, ignore_index=255, reduction='sum') / (target != 255).sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite segmentation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in bare.parameters() if p.requires_grad], 10.0, error_if_nonfinite=True)
            for group, multiplier in zip(optimizer.param_groups, [1, 10]):
                group['lr'] = lr * multiplier * (1 - step / total_steps) ** 0.9
            optimizer.step()
            step += 1
            metric = loss.detach().clone()
            if world > 1:
                dist.all_reduce(metric)
                metric.div_(world)
            if rank == 0 and (step == 1 or step % cfg['log_interval'] == 0):
                record = {'epoch': epoch + 1, 'step': step, 'loss': float(metric), 'gradient_norm': float(gradient_norm), 'logit_max': float(logits.detach().abs().max())}
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if rank == 0:
            torch.save({'model': bare.state_dict(), 'optimizer': optimizer.state_dict(), 'config': cfg, 'epoch': epoch, 'step': step, 'partial_epoch': bool(args.max_steps and step >= args.max_steps)}, output / 'last.tmp.pt')
            (output / 'last.tmp.pt').replace(output / 'last.pt')
        if world > 1:
            dist.barrier()
        if args.max_steps and step >= args.max_steps:
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
