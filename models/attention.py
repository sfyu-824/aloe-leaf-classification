"""注意力模块：ECA 高效通道注意力。"""
import math

import torch
import torch.nn as nn


class ECA(nn.Module):
    """ECA-Net 高效通道注意力（Efficient Channel Attention）。

    通过一维卷积实现局部跨通道信息交互，全程不降维，仅引入极少量参数，
    适配轻量化网络与小样本场景。
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应卷积核大小：k = |log2(C)/gamma + b/gamma| 的最近奇数
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)                    # (b, c, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)     # (b, 1, c)
        y = self.conv(y)                        # (b, 1, c)
        y = y.transpose(-1, -2).unsqueeze(-1)   # (b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y
