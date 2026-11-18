"""注意力模块：ECA 高效通道注意力。"""
import torch
import torch.nn as nn


class ECA(nn.Module):
    """ECA-Net 高效通道注意力（Efficient Channel Attention）。

    通过一维卷积实现局部跨通道信息交互，全程不降维，仅引入极少量参数，
    适配轻量化网络与小样本场景。固定卷积核 k=3 以进一步压缩参数量，
    降低小数据集上的过拟合风险。
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y
