import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        当 in_channels 小于 reduction_ratio 会导致 UserWarning: Initializing zero-element tensors is a no-op
        :param in_channels:
        :param reduction_ratio:
        """
        super(SEBlock1D, self).__init__()
        # 压缩层：全局平均池化到 (batch_size, channels, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        # 扩张层：全连接层产生每个通道的注意力权重
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 获取输入的批次大小和通道数
        batch_size, channels, _ = x.size()

        # Squeeze：全局平均池化，输出尺寸为 (batch_size, channels, 1)
        y = self.squeeze(x).view(batch_size, channels)

        # Excitation：全连接层产生通道注意力权重
        y = self.excitation(y).view(batch_size, channels, 1)

        # 对输入的每个通道进行加权
        return x * y.expand_as(x)


class SEBlock2D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        当 in_channels 小于 reduction_ratio 会导致 UserWarning: Initializing zero-element tensors is a no-op
        :param in_channels:
        :param reduction_ratio:
        """
        super(SEBlock2D, self).__init__()
        # Squeeze：全局平均池化
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation：全连接通道注意力
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Squeeze
        y = self.squeeze(x).squeeze(-1).squeeze(-1)  # (batch_size, channels)

        # Excitation
        y = self.excitation(y).view(batch_size, channels, 1, 1)

        # Scale（广播机制）
        return x * y.expand_as(x)
