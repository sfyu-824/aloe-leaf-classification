"""注意力模块：SE 通道注意力、SAM 空间注意力与 CBAM 混合注意力。

说明：SEBlock（通道注意力）与 SAMBlock（空间注意力）是 CBAM 的两个组成分支，
CBAM = SEBlock + SAMBlock，二者不可删除。
"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力（CBAM 的通道分支）。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weight = self.fc(x.mean(dim=(2, 3))).view(b, c, 1, 1)
        return x * weight


class SAMBlock(nn.Module):
    """Spatial Attention Module 空间注意力（CBAM 的空间分支）。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        weight = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * weight


class CBAM(nn.Module):
    """CBAM 混合注意力：先通道后空间。"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = SEBlock(channels, reduction)
        self.spatial_attn = SAMBlock(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
