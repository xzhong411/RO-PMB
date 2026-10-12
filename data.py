import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


def load_cam(path, classes, height, width):
    value = np.load(path, allow_pickle=True).item()
    key = next((k for k in ('attn_highres', 'refined_cam', 'high_res', 'cam') if k in value), None)
    if key is None:
        raise ValueError(f'Unsupported CAM fields: {list(value)}')
    maps = torch.as_tensor(np.asarray(value[key]), dtype=torch.float32)
    keys = np.asarray(value['keys'], dtype=np.int64)
    if maps.ndim != 3 or maps.shape[0] != len(keys) or (keys < 0).any() or (keys >= classes).any():
        raise ValueError(f'Invalid CAM channels in {path}')
    if not torch.isfinite(maps).all():
        raise ValueError(f'Nonfinite CAM in {path}')
    maps = F.interpolate(maps[None], size=(height, width), mode='bilinear', align_corners=False)[0]
    output = torch.zeros(classes, height, width)
    output[keys] = maps
    output = output.clamp_min(0)
    return output / output.amax((-2, -1), keepdim=True).clamp_min(1e-6)


class ManifestDataset(Dataset):
    def __init__(self, path, cfg, training=False, mask_root=None, limit=0):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if limit:
            self.rows = self.rows[:limit]
        self.cfg = cfg
        self.training = training
        self.mask_root = Path(mask_root) if mask_root else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = torch.from_numpy(np.array(Image.open(row['image']).convert('RGB'), copy=True)).permute(2, 0, 1).float() / 255
        height, width = image.shape[-2:]
        labels = torch.tensor(row['labels'], dtype=torch.float32)
        if self.mask_root:
            mask = torch.from_numpy(np.array(Image.open(self.mask_root / (row['id'] + '.png')), copy=True)).long()
            maps = torch.zeros(1, height, width)
        else:
            mask = torch.full((height, width), 255, dtype=torch.long)
            maps = load_cam(row['cam'], self.cfg['classes'], height, width)
            maps = maps * labels[:, None, None]
        if self.training:
            size = self.cfg['crop_size']
            scale = random.uniform(*self.cfg.get('resize_range', [0.5, 2.0]))
            new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
            image = F.interpolate(image[None], (new_h, new_w), mode='bilinear', align_corners=False)[0]
            maps = F.interpolate(maps[None], (new_h, new_w), mode='bilinear', align_corners=False)[0]
            mask = F.interpolate(mask[None, None].float(), (new_h, new_w), mode='nearest')[0, 0].long()
            pad_h, pad_w = max(0, size - new_h), max(0, size - new_w)
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            image = F.pad(image - mean, (0, pad_w, 0, pad_h)) + mean
            maps = F.pad(maps, (0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=255)
            top = random.randint(0, image.shape[-2] - size)
            left = random.randint(0, image.shape[-1] - size)
            image = image[:, top:top + size, left:left + size]
            maps = maps[:, top:top + size, left:left + size]
            mask = mask[top:top + size, left:left + size]
            if random.random() < 0.5:
                image, maps, mask = image.flip(-1), maps.flip(-1), mask.flip(-1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return {'image': (image - mean) / std, 'seeds': maps, 'labels': labels, 'mask': mask, 'id': row['id'], 'path': row['image']}
