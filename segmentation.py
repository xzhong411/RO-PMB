import torch
from torch import nn
from torch.nn import functional as F
from resnet38 import Net


class SegmentationNet(Net):
    def __init__(self, classes):
        super().__init__()
        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, 3, padding=12, dilation=12)
        self.fc8_seg_conv2 = nn.Conv2d(512, classes + 1, 3, padding=12, dilation=12)
        nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)
        nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(mode)
        return self

    def forward(self, images):
        features = super().forward(images)
        return self.fc8_seg_conv2(F.relu(self.fc8_seg_conv1(features)))
