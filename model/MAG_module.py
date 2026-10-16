"""Motion-Guided Adaptive Gating for multi-path feature fusion."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Compute channel attention using average and maximum pooling."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        """Return channel attention weights."""
        average = self.fc2(self.relu(self.fc1(self.avg_pool(feature))))
        maximum = self.fc2(self.relu(self.fc1(self.max_pool(feature))))
        return self.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    """Compute spatial attention from channel-pooled features."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        """Return spatial attention weights."""
        average = feature.mean(dim=1, keepdim=True)
        maximum = feature.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))


class GatingGenerator(nn.Module):
    """Generate a motion-guided gate for one feature path."""

    def __init__(self, difference_channels=19, gate_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(difference_channels, gate_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, 1),
        )

    def forward(self, raw_difference):
        """Return a positive gate with shape ``[B, 1, H, W]``."""
        gate_logits = self.encoder(raw_difference)
        return (torch.sigmoid(gate_logits) + 0.01) * 10


class FusionRefinement(nn.Module):
    """Refine concatenated path features with residual CBAM processing."""

    def __init__(self, concatenated_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(concatenated_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(concatenated_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        ratio = 16 if out_channels >= 16 else max(1, out_channels // 4)
        self.channel_attention = ChannelAttention(out_channels, ratio)
        self.spatial_attention = SpatialAttention()
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature):
        """Return the refined fused feature."""
        residual = self.shortcut(feature)
        output = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(feature)))))
        output = self.channel_attention(output) * output
        output = self.spatial_attention(output) * output
        return self.refine(self.relu(output + residual))


class MAGModule(nn.Module):
    """Fuse ``F_static``, ``F_diff``, and ``F_dyna`` using motion-guided gates."""

    def __init__(self, in_channels=64, gate_channels=32, out_channels=64, difference_channels=19):
        super().__init__()
        self.static_gating_generator = GatingGenerator(difference_channels, gate_channels)
        self.dynamic_gating_generator = GatingGenerator(difference_channels, gate_channels)
        self.fusion = FusionRefinement(3 * in_channels, out_channels)
        self.difference_projection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(
        self,
        static_feature,
        differential_feature,
        dynamic_feature=None,
        raw_difference=None,
    ):
        """Return the adaptively gated and fused multi-path feature."""
        if raw_difference is None:
            raise ValueError("raw_difference is required by MAGModule")
        static_gate = self.static_gating_generator(raw_difference)
        gated_static_feature = static_feature * static_gate
        if dynamic_feature is None:
            gated_dynamic_feature = torch.zeros_like(gated_static_feature)
        else:
            dynamic_gate = self.dynamic_gating_generator(raw_difference)
            gated_dynamic_feature = dynamic_feature * dynamic_gate
        projected_difference_feature = self.difference_projection(differential_feature)
        fused_input = torch.cat(
            [gated_static_feature, projected_difference_feature, gated_dynamic_feature],
            dim=1,
        )
        return self.fusion(fused_input)


__all__ = ["MAGModule", "GatingGenerator", "FusionRefinement"]
