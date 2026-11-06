"""本文改进模型：MobileNetV3-Small + 中层 SE 通道注意力。"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


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


class ImprovedMobileNetV3(nn.Module):
    """在中层逆残差块（features[6]）后嵌入 SE 通道注意力。"""

    def __init__(self, num_classes=4):
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


def get_mobilenetv3_se(num_classes=4):
    return ImprovedMobileNetV3(num_classes)
