import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


ROOT = Path('/data1/public_data/zhongxiang')


def prepare(name, output):
    classes = 20 if name == 'voc' else 80
    source = ROOT / 'POT/data'
    label_file = source / ('cls_labels_voc.npy' if name == 'voc' else 'COCO_cls_labels.npy')
    labels = np.load(label_file, allow_pickle=True).item()
    dataset = ROOT / 'ToCo-main/Datasets' / ('VOC2012' if name == 'voc' else 'COCO')
    cam_root = ROOT / 'POT-main/pretrained_cams' / ('CLIP-ES' if name == 'voc' else 'CLIP-ES-COCO')
    if name == 'coco':
        annotation_file = ROOT / 'ToCo-main/Datasets/annotations/instances_train2014.json'
        categories = sorted(json.loads(annotation_file.read_text())['categories'], key=lambda category: category['id'])
        assert len(categories) == classes
        mapping = [{'category_id': category['id'], 'segmentation_id': index + 1, 'name': category['name']} for index, category in enumerate(categories)]
        (output / 'coco_categories.json').write_text(json.dumps(mapping, indent=2))
    split_files = {'train': source / ('trainaug_voc.txt' if name == 'voc' else 'train_coco.txt'), 'val': source / f'val_{name}.txt'}
    if name == 'voc':
        split_files['train_eval'] = dataset / 'ImageSets/Segmentation/train.txt'
        split_files['val'] = dataset / 'ImageSets/Segmentation/val.txt'
    matrices = {}
    audit = {}
    for split, list_file in split_files.items():
        ids = [Path(line.split()[0]).stem for line in list_file.read_text().splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate image IDs in {list_file}')
        rows, missing_cams = [], []
        for identity in ids:
            label = labels.get(identity, labels.get(identity + '.jpg'))
            if label is None:
                raise ValueError(f'Missing image-level labels for {identity}')
            label = np.asarray(label, dtype=np.float32).reshape(-1)
            if len(label) != classes or not np.isin(label, [0, 1]).all():
                raise ValueError(f'Invalid image-level label for {identity}')
            image_root = dataset / 'JPEGImages' if name == 'voc' else dataset / ('val2014' if split == 'val' else 'train2014')
            image = image_root / (identity + '.jpg')
            if not image.is_file():
                raise FileNotFoundError(image)
            cam = cam_root / (identity + '.npy')
            if not cam.is_file():
                missing_cams.append(identity)
            if name == 'voc':
                gt = dataset / ('SegmentationClass' if split == 'val' else 'SegmentationClassAug') / (identity + '.png')
            else:
                gt = dataset / 'SegmentationClass' / ('val2014' if split == 'val' else 'train2014') / (identity + '.png')
                official_gt = output / 'coco_ground_truth' / ('val2014' if split == 'val' else 'train2014') / (identity + '.png')
                if official_gt.is_file():
                    gt = official_gt
            rows.append({'id': identity, 'image': str(image), 'cam': str(cam), 'labels': label.tolist(), 'gt': str(gt)})
            if name == 'coco':
                if gt == official_gt:
                    rows[-1].update(gt_encoding='contiguous_80_classes')
                else:
                    rows[-1].update(gt_encoding='coco_category_ids_with_stuff', gt_mapping=str(output / 'coco_categories.json'))
        if split == 'train' and missing_cams:
            raise RuntimeError(f'{len(missing_cams)} missing training CAMs: {missing_cams[:10]}')
        target = output / f'{name}_{split}.jsonl'
        target.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        matrices[split] = np.stack([row['labels'] for row in rows])
        audit[split] = {'images': len(rows), 'missing_cams': len(missing_cams), 'manifest_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'missing_gt': sum(not Path(row['gt']).is_file() for row in rows)}
    y = matrices['train'].astype(np.float64)
    q = (y.T @ y) / y.sum(0).clip(1)[:, None]
    np.save(output / f'{name}_prior.npy', q.astype(np.float32))
    audit['prior'] = {'source': 'train image-level labels only', 'class_counts': y.sum(0).tolist()}
    (output / f'{name}_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({name: audit}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['voc', 'coco', 'both'], default='both')
    parser.add_argument('--output', default='manifests')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ['voc', 'coco'] if args.dataset == 'both' else [args.dataset]:
        prepare(name, output)
