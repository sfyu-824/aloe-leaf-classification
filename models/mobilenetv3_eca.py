"""本文改进模型：MobileNetV3-Small + ECA 高效通道注意力。"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

from .attention import ECA


class MobileNetV3ECA(nn.Module):
    """在特征提取末端（全局平均池化前）嵌入 ECA 高效通道注意力。

    将 ECA 后移到高层语义特征处，仅做通道特征重标定，几乎不增加参数量，
    避免破坏 ImageNet 预训练好的底层纹理/边缘特征。
    """

    def __init__(self, num_classes=5):
        super().__init__()
        base = mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = base.features
        # 最后一层特征图通道数 = 分类头第一个 Linear 的输入维度（MobileNetV3-Small 为 576）
        channels = base.classifier[0].in_features
        self.attn = ECA(channels)
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.classifier[3] = nn.Linear(
            self.classifier[3].in_features, num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_mobilenetv3_eca(num_classes=5):
    """改进模型：MobileNetV3-Small + ECA 高效通道注意力。"""
    return MobileNetV3ECA(num_classes)
