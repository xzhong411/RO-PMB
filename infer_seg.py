import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from data import ManifestDataset
from segmentation import SegmentationNet
from parallel import inference_device, finish_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--single-process', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, rank, world = inference_device(args.single_process)
    state = torch.load(args.checkpoint, map_location='cpu')
    cfg = state['config']
    model = SegmentationNet(cfg['classes']).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    dataset = ManifestDataset(args.manifest, cfg, mask_root=None, limit=args.limit)
    dataset.rows = dataset.rows[rank::world]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, row in enumerate(dataset.rows):
            image = np.array(Image.open(row['image']).convert('RGB'))
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
            tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            logits = F.interpolate(model(tensor[None].to(device)), image.shape[:2], mode='bilinear', align_corners=False)
            prediction = logits[0].argmax(0).cpu().numpy().astype(np.uint8)
            Image.fromarray(prediction).save(output / (row['id'] + '.png'))
            if i % 100 == 0:
                print(json.dumps({'rank': rank, 'processed': i + 1, 'shard_total': len(dataset)}), flush=True)
    finish_inference(world)


if __name__ == '__main__':
    main()
