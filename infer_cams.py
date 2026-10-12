import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from data import ManifestDataset
from model import ROPMB, normalize_cam
from parallel import inference_device, finish_inference


def dense_crf(image, probabilities, cfg):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    classes, height, width = probabilities.shape
    model = dcrf.DenseCRF2D(width, height, classes)
    model.setUnaryEnergy(unary_from_softmax(np.ascontiguousarray(probabilities)))
    model.addPairwiseGaussian(sxy=cfg['crf_gaussian_sxy'], compat=3)
    model.addPairwiseBilateral(sxy=cfg['crf_bilateral_sxy'], srgb=cfg['crf_bilateral_srgb'], rgbim=np.ascontiguousarray(image), compat=10)
    return np.asarray(model.inference(cfg['crf_iterations'])).reshape(classes, height, width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--no-crf', action='store_true')
    parser.add_argument('--single-process', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, rank, world = inference_device(args.single_process)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint['config']
    model = ROPMB(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    if not model.initialized.all():
        raise RuntimeError('Uninitialized memory bank')
    model.eval()
    dataset = ManifestDataset(args.manifest, cfg, limit=args.limit)
    total = len(dataset)
    dataset.rows = dataset.rows[rank::world]
    output = Path(args.output)
    for name in ['cams', 'raw_masks', 'pseudo_labels']:
        (output / name).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, pack in enumerate(dataset):
            images, seeds, labels = pack['image'][None].to(device), pack['seeds'][None].to(device), pack['labels'][None].to(device)
            result = model(images, seeds, labels)
            cams = normalize_cam(F.interpolate(result['cams'], images.shape[-2:], mode='bilinear', align_corners=False))[0]
            cams = cams * labels[0, :, None, None]
            present = labels[0].nonzero().flatten()
            foreground = cams[present].cpu().numpy()
            background = np.full((1, *foreground.shape[-2:]), cfg['background_score'], dtype=np.float32)
            scores = np.concatenate([background, foreground], 0)
            mapping = np.concatenate([[0], present.cpu().numpy() + 1])
            raw = mapping[scores.argmax(0)].astype(np.uint8)
            np.save(output / 'cams' / (pack['id'] + '.npy'), {'keys': present.cpu().numpy(), 'high_res': foreground})
            Image.fromarray(raw).save(output / 'raw_masks' / (pack['id'] + '.png'))
            if args.no_crf:
                pseudo = raw
            else:
                probabilities = np.clip(scores, 1e-6, None)
                probabilities /= probabilities.sum(0, keepdims=True)
                image = np.array(Image.open(pack['path']).convert('RGB'))
                refined = dense_crf(image, probabilities, cfg)
                pseudo = mapping[refined.argmax(0)].astype(np.uint8)
            Image.fromarray(pseudo).save(output / 'pseudo_labels' / (pack['id'] + '.png'))
            if i % 100 == 0:
                print(json.dumps({'rank': rank, 'processed': i + 1, 'shard_total': len(dataset)}), flush=True)
    finish_inference(world)
    if rank == 0:
        (output / 'inference.json').write_text(json.dumps({'checkpoint': args.checkpoint, 'manifest': args.manifest, 'images': total, 'workers': world, 'crf': not args.no_crf, 'config': cfg}, indent=2))


if __name__ == '__main__':
    main()
