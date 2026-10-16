"""Difference path for extracting the differential feature ``F_diff``."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Compute CBAM channel attention."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        ratio = max(1, min(ratio, channels))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // ratio, channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        """Return channel attention weights."""
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
        """Return spatial attention weights."""
        average = feature.mean(dim=1, keepdim=True)
        maximum = feature.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))


class FrameDifference(nn.Module):
    """Compute masked differences between the current and historical frames."""

    def forward(self, aligned_sequence):
        """Return raw differences with shape ``[B, T-1, H, W]``."""
        historical_frames = aligned_sequence[:, :-1]
        current_frame = aligned_sequence[:, -1].unsqueeze(1)
        raw_difference = current_frame - historical_frames
        valid_mask = 1 - (historical_frames == 0).to(aligned_sequence.dtype)
        return raw_difference * valid_mask


class DifferencePath(nn.Module):
    """Encode raw frame differences into ``F_diff``."""

    def __init__(self, sequence_length, out_channels=64):
        super().__init__()
        if sequence_length < 2:
            raise ValueError(f"sequence_length must be at least 2, got {sequence_length}")
        self.sequence_length = sequence_length
        self.frame_difference = FrameDifference()
        self.conv1 = nn.Conv2d(sequence_length - 1, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(sequence_length - 1, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        ratio = 16 if out_channels >= 16 else max(1, out_channels // 4)
        self.channel_attention = ChannelAttention(out_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, aligned_sequence):
        """Return ``F_diff`` and its raw difference tensor."""
        raw_difference = self.frame_difference(aligned_sequence)
        residual = self.shortcut(raw_difference)
        differential_feature = self.bn2(
            self.conv2(self.relu(self.bn1(self.conv1(raw_difference))))
        )
        differential_feature = self.channel_attention(differential_feature) * differential_feature
        differential_feature = self.spatial_attention(differential_feature) * differential_feature
        return self.relu(differential_feature + residual), raw_difference


__all__ = ["DifferencePath", "FrameDifference"]
