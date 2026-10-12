import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from data import ManifestDataset
from model import ROPMB, cluster_features, project_conflicting


def gradients(loss, parameters, retain=True):
    values = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, values)]


@torch.no_grad()
def initialize(model, dataset, device, output):
    model.eval()
    candidates = [[] for _ in range(model.classes)]
    generator = torch.Generator().manual_seed(model.cfg['seed'])
    order = torch.randperm(len(dataset), generator=generator).tolist()
    target = model.cfg['initial_candidates']
    examined = 0
    for index in order:
        if not any(present and len(candidates[c]) < target for c, present in enumerate(dataset.rows[index]['labels'])):
            continue
        examined += 1
        pack = dataset[index]
        image = pack['image'][None].to(device)
        image = torch.nn.functional.interpolate(image, (model.cfg['crop_size'], model.cfg['crop_size']), mode='bilinear', align_corners=False)
        _, _, feature = model.encode(image)
        records = model.extract(feature, pack['seeds'][None].to(device), pack['labels'][None].to(device))
        for _, c, local in records:
            if len(candidates[c]) < target:
                candidates[c].extend(local.cpu().unbind())
        if examined % 100 == 0:
            print(json.dumps({'initialization_images': examined, 'least_candidates': min(map(len, candidates)), 'classes_below_target': sum(len(values) < target for values in candidates)}), flush=True)
        if all(len(values) >= target for values in candidates):
            break
    missing = [c for c, values in enumerate(candidates) if len(values) < model.kg]
    if missing:
        raise RuntimeError(f'Insufficient diverse initialization candidates for classes {missing}')
    for c, values in enumerate(candidates):
        centers = cluster_features(torch.stack(values).to(device), model.kg, 30)
        model.memory[c].copy_(centers)
    model.initialized.fill_(True)
    for index in order[:model.cfg['warmup_images']]:
        pack = dataset[index]
        image = torch.nn.functional.interpolate(pack['image'][None].to(device), (model.cfg['crop_size'], model.cfg['crop_size']), mode='bilinear', align_corners=False)
        results = model(image, pack['seeds'][None].to(device), pack['labels'][None].to(device))
        model.update_memory(results['updates'])
    Path(output).write_text(json.dumps({'counts': [len(v) for v in candidates], 'images_examined': examined, 'warmup_images': min(model.cfg['warmup_images'], len(order))}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.epochs:
        cfg['epochs'] = args.epochs
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'] + rank)
    random.seed(cfg['seed'] + rank)
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = ROPMB(cfg, pretrained=not args.resume).to(device)
    model.classifier.requires_grad_(bool(cfg.get('auxiliary_classification_weight', 0)))
    model.prior.copy_(torch.from_numpy(np.load(cfg['prior'])).to(device))
    dataset = ManifestDataset(cfg['train_manifest'], cfg, training=True, limit=args.limit)
    initial_dataset = ManifestDataset(cfg['train_manifest'], cfg, limit=args.limit)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    start_epoch, global_step = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get('partial_epoch'):
            raise ValueError('Partial-epoch smoke checkpoints cannot resume a full experiment')
        if checkpoint['config'] != cfg:
            raise ValueError('Resume configuration differs from checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, global_step = checkpoint['epoch'] + 1, checkpoint['step']
    elif rank == 0:
        initialize(model, initial_dataset, device, output / 'initialization.json')
    if world > 1:
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor.data, 0)
    if cfg['batch_size'] % world:
        raise ValueError('Global batch size must be divisible by world size')
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=cfg['seed'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'] // world, sampler=sampler, num_workers=cfg['workers'], pin_memory=True, drop_last=True)
    if not len(loader):
        raise RuntimeError('Empty training loader')
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(cfg, indent=2))
    started = time.time()
    for epoch in range(start_epoch, cfg['epochs']):
        sampler.set_epoch(epoch)
        random.seed(cfg['seed'] + epoch * world + rank)
        torch.manual_seed(cfg['seed'] + epoch * world + rank)
        model.train()
        if cfg.get('freeze_batch_norm', True):
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
        for pack in loader:
            optimizer.zero_grad(set_to_none=True)
            results = model(pack['image'].to(device), pack['seeds'].to(device), pack['labels'].to(device))
            health = results['health']
            minimum_norm = 0.9 if cfg.get('normalize_pixel_features', True) else 1e-8
            healthy = torch.tensor(bool(health['projected_norm_max'] > minimum_norm and (health['cam_max'] > 1e-6 or results['records'] == 0)), device=device, dtype=torch.int32)
            if world > 1:
                dist.all_reduce(healthy, op=dist.ReduceOp.MIN)
            if not healthy.item():
                diagnostic = {'epoch': epoch + 1, 'step': global_step, 'rank': rank, 'ids': pack['id'], 'health': {k: float(v) for k, v in health.items()}, 'input_max': float(pack['image'].abs().max()), 'nonzero_input_fraction': float((pack['image'].abs().sum(1) > 0).float().mean())}
                (output / f'failure_rank{rank}.json').write_text(json.dumps(diagnostic, indent=2))
                raise FloatingPointError(f'Degenerate features or CAMs at step {global_step}')
            losses = results['losses']
            base = losses['classification'] + cfg['lambda_consistency'] * losses['consistency']
            base = base + cfg.get('auxiliary_classification_weight', 0) * losses.get('auxiliary_classification', base.new_zeros(()))
            base = base + cfg.get('cam_alignment_weight', 0) * losses.get('cam_alignment', base.new_zeros(()))
            diversity = cfg['lambda_diversity'] * losses['diversity']
            smoothness = cfg['lambda_smoothness'] * losses['smoothness']
            gd = gradients(diversity, parameters)
            gs = gradients(smoothness, parameters)
            projected, conflict = project_conflicting(gd, gs)
            gb = gradients(base, parameters, False)
            for p, a, b, c in zip(parameters, gb, projected, gs):
                p.grad = a + b + c
                if world > 1:
                    dist.all_reduce(p.grad)
                    p.grad.div_(world)
            finite = torch.tensor(all(torch.isfinite(p.grad).all() for p in parameters), device=device, dtype=torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite gradients at step {global_step}')
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            updates = results['updates']
            if world > 1:
                gathered = [None] * world
                dist.all_gather_object(gathered, updates)
                updates = [item for group in gathered for item in group]
            model.update_memory(updates)
            global_step += 1
            values = torch.tensor([float(v.detach()) for v in losses.values()], device=device)
            if world > 1:
                dist.all_reduce(values)
                values.div_(world)
            if rank == 0 and (global_step % cfg['log_interval'] == 0 or global_step == 1):
                record = {'epoch': epoch + 1, 'step': global_step, 'seconds': round(time.time() - started, 2), 'conflict': conflict, 'prototype_records_rank0': results['records'], 'gradient_norm': float(gradient_norm), **{k: float(v) for k, v in health.items()}, **dict(zip(losses, values.tolist()))}
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
        if rank == 0:
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'step': global_step, 'config': cfg, 'partial_epoch': bool(args.max_steps and global_step >= args.max_steps)}
            torch.save(checkpoint, output / 'last.tmp.pt')
            (output / 'last.tmp.pt').replace(output / 'last.pt')
            print(json.dumps({'saved': str(output / 'last.pt'), 'epoch': epoch + 1}), flush=True)
        if world > 1:
            dist.barrier()
        if args.max_steps and global_step >= args.max_steps:
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
