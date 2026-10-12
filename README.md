# LLM-SUB：VOC / COCO 主实验复现

本目录仅包含论文表 1（CAM / 伪标签质量）和表 2（语义分割质量）的主实验实现，不包含消融、历史版本、日志、训练权重或数据副本。目录位于服务器 `172.31.224.73:/data1/public_data/zhongxiang/LLM-SUB`。

这是无作者代码条件下的适配复现，并非官方实现，也未确认达到论文指标。VOC 采用目前的 v4 实现；COCO 使用相同方法和超参数，但尚未完成该版本的端到端验证。整理目录不会启动训练。

## 1. 环境

推荐使用服务器已安装的环境：

```bash
ssh 172.31.224.73
cd /data1/public_data/zhongxiang/LLM-SUB
export PATH=/data1/public_data/zhongxiang/conda/envs/mct/bin:$PATH
python -c "import torch, torchvision, pydensecrf, pycocotools; print(torch.__version__, torch.cuda.is_available())"
```

已使用的环境为 Python 3.9、PyTorch 2.3.1 + CUDA 11.8、torchvision 0.18.1。新建环境时先安装对应 CUDA 版本的 PyTorch，再安装 `requirements.txt`；`pydensecrf` 需要兼容的编译环境。现有 mct 环境无需重新安装。

## 2. 外部资源

以下资源保留在原位置，由本目录直接读取，不复制大文件。程序不依赖 `LLM-G` 的源代码、数据清单或实验结果。

| 资源 | 路径（均相对于 `/data1/public_data/zhongxiang/`） |
| --- | --- |
| VOC 图像和像素标签 | `ToCo-main/Datasets/VOC2012/` |
| COCO 图像 | `ToCo-main/Datasets/COCO/train2014/`、`val2014/` |
| COCO 官方实例标注 | `ToCo-main/Datasets/annotations/instances_train2014.json`、`instances_val2014.json` |
| 图像级标签 | `POT/data/cls_labels_voc.npy`、`COCO_cls_labels.npy` |
| 训练清单 | `POT/data/trainaug_voc.txt`、`train_coco.txt` |
| COCO 验证清单 | `POT/data/val_coco.txt` |
| VOC 标准划分 | `ToCo-main/Datasets/VOC2012/ImageSets/Segmentation/` |
| VOC 初始 CAM | `POT-main/pretrained_cams/CLIP-ES/` |
| COCO 初始 CAM | `POT-main/pretrained_cams/CLIP-ES-COCO/` |
| ResNet50 预训练权重 | `POT-main/resnet50-19c8e357.pth` |
| ResNet38 预训练权重 | `POT-main/segmentation/weights/res38_cls.pth` |

初始 CAM 是本实现的必要输入，使用现有离线缓存；本精简包不包含 CLIP-ES 初始 CAM 生成器。缓存需为每张训练图像提供 `.npy` 字典，含零起始前景类别 `keys` 和 CAM 张量（支持 `attn_highres`、`refined_cam`、`high_res` 或 `cam`）。缓存来源未独立核验，因此会影响与论文的可比性。

迁移服务器时，需修改 `prepare.py` 的 `ROOT`、`prepare_coco_gt.py` 的标注目录，以及两份配置中的权重和数据清单路径。所有下述命令均在本目录执行。

## 3. 准备 VOC

```bash
python prepare.py --dataset voc
```

生成 `manifests/voc_*.jsonl` 和训练图像级标签计算的类别先验 `voc_prior.npy`。训练使用 trainaug 10,582 张；表 1 对应的 CAM / 伪标签评估使用 train 1,464 张；分割评估使用 val 1,449 张。

## 4. 准备 COCO

```bash
python prepare.py --dataset coco
python prepare_coco_gt.py --workers 8
```

第二条命令必须完成后才能启动 COCO 全流程。它从官方实例标注生成本目录下的语义评估标签，并更新数据清单；不会修改原始数据集。首次生成需要时间和额外磁盘空间。

语义标签按官方 category ID 排序映射到 1–80，背景为 0，忽略区域为 255。非 crowd 实例按面积从大到小绘制，小实例覆盖重叠部分；仅落在背景上的 crowd 区域忽略。该策略属于本实现的明确选择。像素标签仅用于评估，分割训练使用生成的伪标签。

现有清单包含训练 82,081 张、验证 40,137 张，并非完整的 40,504 张 COCO val2014；因此结果不能未经核对就与完整验证集成绩等同。

## 5. 运行主实验

先用 `nvidia-smi` 查看占用，并将命令里的 GPU 编号换成空闲卡。示例使用 4 卡训练 CAM、1 卡训练分割；全局 batch size 为 4。不要直接占用正在运行原实验的 GPU。

VOC：

```bash
python launch.py --dataset voc --gpus 0,1,2,3 --segmentation-gpus 0
```

COCO（需要时手动启动）：

```bash
python launch.py --dataset coco --gpus 0,1,2,3 --segmentation-gpus 0
```

`launch.py` 在后台启动任务，退出 SSH 后继续运行。默认分别读取 `configs/voc.json`、`configs/coco.json`，结果写入 `runs/voc/`、`runs/coco/`。前台运行可将 `launch.py` 换成 `pipeline.py`。

流程依次为：原型模型训练 → CAM / CRF 伪标签生成 → 两项质量评估 → 伪标签健康检查 → ResNet38 分割训练 → 验证集推理与评估。默认 CAM 训练 45 轮、分割训练 30 轮。健康检查失败会阻止后续分割训练，需要先检查伪标签。

恢复中断任务：

```bash
python launch.py --dataset voc --gpus 0,1,2,3 --segmentation-gpus 0 --resume
python launch.py --dataset coco --gpus 0,1,2,3 --segmentation-gpus 0 --resume
```

只运行需要恢复的那条命令。恢复会跳过已完成阶段，并读取相应 `last.pt`；训练按已保存的完整轮次恢复。恢复时保持配置一致。重新开始独立实验请指定新目录，例如 `--output runs/voc_new`，避免覆盖旧结果；恢复该实验时仍需指定同一 `--output`。

## 6. 查看进度和结果

```bash
cat runs/voc/status.json
tail -n 30 runs/voc/cam_training.log
cat runs/voc/cam_metrics.json
cat runs/voc/pseudo_label_metrics.json
cat runs/voc/val_metrics.json
```

COCO 将路径中的 `voc` 替换为 `coco`。结果文件在相应阶段结束后产生；`status.json` 为 `complete` 才代表整条流程完成。每个阶段有独立日志，失败时先查看状态中的阶段名及其 `.log`。

| 实验结果 | 文件 | 单位 |
| --- | --- | --- |
| 表 1：CAM | `runs/<dataset>/cam_metrics.json` | mIoU，百分数 |
| 表 1：CRF 伪标签 | `runs/<dataset>/pseudo_label_metrics.json` | mIoU，百分数 |
| 表 2：验证集分割 | `runs/<dataset>/val_metrics.json` | mIoU，百分数 |

评估忽略标签 255，缺失预测默认报错。VOC test 没有本地公开像素真值，本包不生成或声称 VOC test 指标；论文表 2 的 test 项仍需官方评测服务。

## 7. 实现口径

保留全局 / 局部原型、类别关系图、原型记忆更新、分类 / 一致性 / 多样性 / 平滑损失及冲突梯度处理。无原作者实现的细节采用当前可运行的选择：保留投影特征幅度、欧氏聚类、图像中未出现类别响应置零、更新 BatchNorm 统计；记忆参数同时使用优化器和 EMA 更新。

为改善已观察到的收敛问题，v4 额外加入权重均为 1.0 的图像级分类辅助监督与初始 CAM 对齐损失。这两项属于适配改动。推理采用单尺度，无多尺度 / 翻转增强；CAM 使用固定背景分数 0.2 和 DenseCRF，分割输出直接 argmax。ResNet38 网络定义整理自服务器现有 POT-main 实现。上述差异、离线 CAM 来源以及评估划分均可能造成与论文指标的差距。

## 8. 文件组成

- `model.py`、`data.py`、`train.py`：原型模型、数据加载及训练。
- `segmentation.py`、`resnet38.py`、`train_seg.py`：分割网络及训练。
- `infer_cams.py`、`infer_seg.py`、`parallel.py`：推理及多卡推理支持。
- `evaluate.py`、`ground_truth.py`、`check_pseudo_health.py`：评估与流程必要检查。
- `prepare.py`、`prepare_coco_gt.py`：数据清单、先验和 COCO 语义标签准备。
- `pipeline.py`、`launch.py`：完整流程与后台启动。
- `configs/voc.json`、`configs/coco.json`、`requirements.txt`、`README.md`：配置、依赖和指引。

初始交付共 20 个文件，Python 代码不包含注释或文档字符串。`manifests/` 和 `runs/` 仅在执行准备或实验命令后生成。
