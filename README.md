
## 1. Create and activate conda environment

```bash
conda create --name py38 python=3.8
conda activate py38
pip install -r requirments.txt
```

## 2. External resources

The initial CAM is a mandatory input, provided through an existing offline cache. For each training image, the cache must supply a .npy dictionary containing zero-based foreground category keys together with the CAM tensor, which may be stored under any of the following keys: attn_highres, refined_cam, high_res, or cam.


## 3. VOC dataset

```bash
python prepare.py --dataset voc
```


## 4. COCO dataset

```bash
python prepare.py --dataset coco
python prepare_coco_gt.py --workers 8
```

Semantic labels are sorted by the official category ID and mapped to 1–80, with background as 0 and ignore regions as 255. Non-crowd instances are drawn in descending order of area, so smaller instances overwrite overlapping regions; crowd regions that fall only on the background are ignored. Pixel-level labels are used for evaluation only, while segmentation training uses the generated pseudo-labels.

## 5. Run


VOC：

```bash
python launch.py --dataset voc --gpus 0,1,2,3 --segmentation-gpus 0
```

COCO：

```bash
python launch.py --dataset coco --gpus 0,1,2,3 --segmentation-gpus 0
```

The pipeline proceeds as follows: prototype model training → CAM/CRF pseudo-label generation → two quality assessments → pseudo-label sanity check → ResNet-38 segmentation training → validation-set inference and evaluation. The CAM model is trained for 45 epochs, and the segmentation model for 30 epochs.


## 6. View Progress and Results

```bash
cat runs/voc/status.json
tail -n 30 runs/voc/cam_training.log
cat runs/voc/cam_metrics.json
cat runs/voc/pseudo_label_metrics.json
cat runs/voc/val_metrics.json
```
For COCO, replace voc with coco in the path. Result files appear once the corresponding stage finishes, and the full pipeline is complete only when status.json reads complete. Each stage maintains a separate log.


