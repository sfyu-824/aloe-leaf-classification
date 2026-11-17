"""本文改进模型：MobileNetV3-Small + CBAM 混合注意力（含 SE/SAM 消融变体）。"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

from .attention import build_attention


class MobileNetV3Attention(nn.Module):
    """在中层逆残差块（features[6]）后嵌入指定注意力模块。"""

    def __init__(self, num_classes=5, attention="cbam"):
        super().__init__()
        base = mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = base.features
        self.attn = build_attention(attention, base.features[6].out_channels)
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.classifier[3] = nn.Linear(
            self.classifier[3].in_features, num_classes
        )

    def forward(self, x):
        x = self.features[:7](x)
        x = self.attn(x)
        x = self.features[7:](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_mobilenetv3_cbam(num_classes=5):
    """改进模型：完整 CBAM（通道 + 空间注意力）。"""
    return MobileNetV3Attention(num_classes, "cbam")


def get_mobilenetv3_se(num_classes=5):
    """消融变体：仅 SE 通道注意力。"""
    return MobileNetV3Attention(num_classes, "se")


def get_mobilenetv3_sam(num_classes=5):
    """消融变体：仅 SAM 空间注意力。"""
    return MobileNetV3Attention(num_classes, "sam")
