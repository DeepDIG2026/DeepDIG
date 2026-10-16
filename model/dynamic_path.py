"""Dynamic path for extracting the dynamic feature ``F_dyna``.

The path applies Reliability-Aware Dynamic Convolution (RADC), comprising
RATA and CADC, before residual convolutional and CBAM feature encoding.
"""

import torch
import torch.nn as nn

from .RADC_module import ReliabilityAwareDynamicConvolution


class ChannelAttention(nn.Module):
    """Compute CBAM channel attention."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        ratio = max(1, min(ratio, channels))
        self.fc1 = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // ratio, channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        """Return channel attention weights for ``feature``."""
        average = self.fc2(self.relu(self.fc1(self.avg_pool(feature))))
        maximum = self.fc2(self.relu(self.fc1(self.max_pool(feature))))
        return self.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    """Compute CBAM spatial attention."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        """Return spatial attention weights for ``feature``."""
        average = feature.mean(dim=1, keepdim=True)
        maximum = feature.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))


class DynamicPath(nn.Module):
    """Extract the dynamic feature ``F_dyna`` from an aligned sequence."""

    def __init__(
        self,
        sequence_length,
        out_channels=64,
        cadc_kernel_size=3,
        reduction=4,
        rata_use_region=True,
        rata_use_pixel=True,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        self.radc = ReliabilityAwareDynamicConvolution(
            sequence_length=sequence_length,
            kernel_size=cadc_kernel_size,
            reduction=reduction,
            rata_use_region=rata_use_region,
            rata_use_pixel=rata_use_pixel,
        )
        self.conv1 = nn.Conv2d(sequence_length, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(sequence_length, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        ratio = 16 if out_channels >= 16 else max(1, out_channels // 4)
        self.channel_attention = ChannelAttention(out_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, aligned_sequence, raw_difference):
        """Return ``F_dyna`` with shape ``[B, C, H, W]``."""
        enhanced_sequence = self.radc(aligned_sequence, raw_difference)
        residual = self.shortcut(enhanced_sequence)
        dynamic_feature = self.bn2(
            self.conv2(self.relu(self.bn1(self.conv1(enhanced_sequence))))
        )
        dynamic_feature = self.channel_attention(dynamic_feature) * dynamic_feature
        dynamic_feature = self.spatial_attention(dynamic_feature) * dynamic_feature
        return self.relu(dynamic_feature + residual)


__all__ = ["DynamicPath"]
