import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ground_truth import read_ground_truth


def evaluate(manifest, predictions, classes, output, allow_missing=False):
    confusion = np.zeros((classes + 1, classes + 1), dtype=np.int64)
    count = 0
    missing = []
    for line in Path(manifest).read_text().splitlines():
        row = json.loads(line)
        pred_path = Path(predictions) / (row['id'] + '.png')
        if not pred_path.is_file() or not Path(row['gt']).is_file():
            missing.append(row['id'])
            continue
        pred = np.asarray(Image.open(pred_path)).astype(np.int64)
        gt = read_ground_truth(row, classes)
        if pred.shape != gt.shape or pred.max() > classes or pred.min() < 0:
            raise ValueError(f'Invalid prediction for {row["id"]}')
        valid = (gt >= 0) & (gt <= classes)
        confusion += np.bincount((classes + 1) * gt[valid] + pred[valid], minlength=(classes + 1) ** 2).reshape(classes + 1, classes + 1)
        count += 1
    if missing and not allow_missing:
        raise RuntimeError(f'{len(missing)} missing predictions or GT: {missing[:10]}')
    if not count:
        raise RuntimeError('No evaluated images')
    intersection = confusion.diagonal()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full_like(union, np.nan, dtype=np.float64), where=union > 0)
    result = {'images': count, 'missing': len(missing), 'partial_evaluation': bool(missing), 'miou_percent': float(np.nanmean(iou) * 100), 'class_iou_percent': [float(x * 100) if np.isfinite(x) else None for x in iou], 'confusion': confusion.tolist()}
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'confusion'}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--classes', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()
    evaluate(args.manifest, args.predictions, args.classes, args.output, args.allow_missing)
