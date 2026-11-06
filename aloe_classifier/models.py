"""模型定义：SE 通道注意力、3 条基线与本文改进模型。"""
import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, mobilenet_v3_small, resnet50,
)

NUM_CLASSES = 4


class SEAttention(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weight = self.avg_pool(x).view(b, c)
        weight = self.fc(weight).view(b, c, 1, 1)
        return x * weight


# ---------- 基线 1：ResNet-50 ----------
def build_resnet50(num_classes=NUM_CLASSES):
    model = resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------- 基线 2：EfficientNet-B0 ----------
def build_efficientnet_b0(num_classes=NUM_CLASSES):
    model = efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model


# ---------- 基线 3：原生 MobileNetV3-Small ----------
def build_mobilenetv3_base(num_classes=NUM_CLASSES):
    model = mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, num_classes
    )
    return model


# ---------- 本文改进：MobileNetV3-Small + 中层 SE ----------
class ImprovedMobileNetV3(nn.Module):
    """在中层逆残差块（features[6]）后嵌入 SE 通道注意力。"""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = base.features
        self.se = SEAttention(base.features[6].out_channels)  # 注册为子模块
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.classifier[3] = nn.Linear(
            self.classifier[3].in_features, num_classes
        )

    def forward(self, x):
        x = self.features[:7](x)
        x = self.se(x)
        x = self.features[7:](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_improved_mobilenetv3(num_classes=NUM_CLASSES):
    return ImprovedMobileNetV3(num_classes)


# 模型注册表：名称 -> 构造函数，供训练入口统一调度
MODEL_REGISTRY = {
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "mobilenetv3_base": build_mobilenetv3_base,
    "mobilenetv3_se": build_improved_mobilenetv3,
}
