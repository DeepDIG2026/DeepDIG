"""
Motion-Aware Gating (MAG) Module.

Fuses static, differential, and dynamic features through motion-guided 
gating mechanism followed by attention-based refinement.

This implements the MotionGuidedGatingFusion (MGGF) from original STDMANet.

Key features:
- Uses raw_diff (T-1 channels) for gate generation (not encoded diff_feat)
- Dual-path collaborative enhancement (static + dynamic)
- Res_CTSAM-style attention fusion
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention with global pooling and MLP."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention with channel pooling and convolution."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class MotionGateEncoder(nn.Module):
    """
    Generates motion-guided gating weights from raw frame differences.
    
    Uses raw_diff (T-1 channels) instead of encoded diff_feat for sharper motion edges.
    """
    
    def __init__(self, raw_diff_channels=19, gate_channels=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(raw_diff_channels, gate_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, gate_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, 1)
        )
    
    def forward(self, raw_diff):
        """Generate gate from raw_diff [B, T-1, H, W] -> [B, 1, H, W]"""
        return self.encoder(raw_diff)


class FeatureEnhancement(nn.Module):
    """
    Motion-guided feature enhancement.
    
    Uses raw_diff to generate spatial gating weights for feature enhancement.
    """
    
    def __init__(self, raw_diff_channels=19, gate_channels=32):
        super().__init__()
        
        self.gate_encoder = nn.Sequential(
            nn.Conv2d(raw_diff_channels, gate_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, 1)
        )
    
    def forward(self, feat, raw_diff):
        """
        Args:
            feat: Input feature to enhance [B, C, H, W]
            raw_diff: Raw frame differences for gating [B, T-1, H, W]
        """
        gate = self.gate_encoder(raw_diff)
        weight = (torch.sigmoid(gate) + 0.01) * 10
        return feat * weight


class FusionRefine(nn.Module):
    """
    Attention-based feature refinement.
    
    Applies residual block with channel and spatial attention,
    followed by a refinement convolution.
    """
    
    def __init__(self, concat_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(concat_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        ratio = 16 if out_channels >= 16 else max(1, out_channels // 4)
        self.ca = ChannelAttention(out_channels, ratio)
        self.sa = SpatialAttention()
        
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out + residual)
        
        return self.refine(out)


class MAGModule(nn.Module):
    """
    Motion-Aware Gating module for multi-path feature fusion.
    
    This implements the MotionGuidedGatingFusion (MGGF) from original STDMANet.
    
    Key features (matching STDMANet):
    - Uses raw_diff (T-1 channels) for gate generation
    - Dual-path collaborative enhancement (static + dynamic)
    - CBAM-style attention fusion
    
    Args:
        in_channels: Feature channels per path (default: 64)
        gate_channels: Intermediate gating channels (default: 32)
        out_channels: Output feature channels (default: 64)
        raw_diff_channels: Raw difference channels T-1 (default: 19)
    """
    
    def __init__(self, in_channels=64, gate_channels=32, out_channels=64, raw_diff_channels=19):
        super().__init__()
        
        self.in_channels = in_channels
        self.raw_diff_channels = raw_diff_channels
        
        # Dual-path enhancement using raw_diff
        self.static_enhance = FeatureEnhancement(raw_diff_channels, gate_channels)
        self.dynamic_enhance = FeatureEnhancement(raw_diff_channels, gate_channels)
        
        # Fusion with CBAM attention
        self.fusion = FusionRefine(3 * in_channels, out_channels)
        
        # Diff feature projection
        self.diff_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, static_feat, diff_feat, dynamic_feat, raw_diff=None):
        """
        Args:
            static_feat: Static path feature [B, C, H, W]
            diff_feat: Differential path feature [B, C, H, W]
            dynamic_feat: Dynamic path feature [B, C, H, W]
            raw_diff: Raw frame differences [B, T-1, H, W] (required for gate)
        
        Returns:
            Fused feature [B, C, H, W]
        """
        if raw_diff is None:
            raise ValueError("raw_diff is required for MAGModule (use diff from DifferencePath)")
        
        # Dual-path enhancement using raw_diff as gate
        static_out = self.static_enhance(static_feat, raw_diff)
        dynamic_out = self.dynamic_enhance(dynamic_feat, raw_diff)
        
        # Concat and fuse with CBAM attention
        # Order matches STDMANet MotionGuidedGatingFusion: [enhanced_static, enhanced_dynamic, diff]
        concat = torch.cat([static_out, dynamic_out, diff_feat], dim=1)
        return self.fusion(concat)


__all__ = ['MAGModule', 'FeatureEnhancement', 'FusionRefine']
