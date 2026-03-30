"""
Difference Path.

Computes masked frame differences and encodes them into differential features.

Data flow:
    X_seq [B, T, H, W] (input sequence)
        ↓
    Differential (frame difference + mask)
        ↓
    raw_diff [B, T-1, H, W] (raw differences, for TADC and MAG)
        ↓
    ResBlock Encoder + CBAM Attention
        ↓
    F_diff [B, C, H, W] (differential features)
"""

import torch
import torch.nn as nn

from .detection_network import ResBlock


class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        ratio = max(1, min(ratio, channels))
        self.fc1 = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // ratio, channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class Differential(nn.Module):
    """
    Frame difference computation module.
    
    Computes differences between current frame and all history frames,
    applies mask to exclude zero-value regions.
    
    Args:
        in_channels: Input channels (sequence length T, for compatibility)
    """
    
    def __init__(self, in_channels=20):
        super(Differential, self).__init__()

    def forward(self, x):
        """
        Compute frame differences.
        
        Args:
            x: [B, T, H, W] input sequence
            
        Returns:
            flows: [B, T-1, H, W] raw differences (with mask)
        """
        b, t, h, w = x.size()
        x_plus0 = x[:, :-1]  # First T-1 frames
        x_plus1 = x[:, -1].unsqueeze(1)  # Current frame (last frame)
        
        # Compute difference: current - history
        flows = x_plus1 - x_plus0  # [B, T-1, H, W]
        
        # Mask: exclude zero-value regions (background/invalid)
        mask = 1 - (x_plus0 == 0).to(x.dtype)
        flows = flows * mask
        
        return flows


class DifferencePath(nn.Module):
    """
    Difference Path (matching STDMANet's Differential_block).
    
    Computes masked frame differences and encodes into differential features
    with CBAM attention enhancement.
    
    Data flow:
        X_seq [B, T, H, W]
            ↓
        Differential
            ↓
        raw_diff [B, T-1, H, W] ──────────→ (for TADC and MAG)
            ↓
        ResBlock Encoder + CBAM
            ↓
        F_diff [B, C, H, W]
    
    Args:
        seq_len: Input sequence length T
        out_channels: Output feature channels (default: 64)
    """
    
    def __init__(self, seq_len, out_channels=64):
        super().__init__()
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        
        self.seq_len = seq_len
        
        # Frame difference module
        self.diff = Differential(in_channels=seq_len)
        
        # Feature encoder: T-1 channels → out_channels (matching Differential_block)
        self.conv1 = nn.Conv2d(seq_len - 1, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut for residual connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(seq_len - 1, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # CBAM attention (matching STDMANet)
        ratio = 16 if out_channels >= 16 else max(1, out_channels // 4)
        self.ca = ChannelAttention(out_channels, ratio=ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [B, T, H, W] DBA-aligned sequence
        
        Returns:
            diff_feat: [B, C, H, W] differential features
            raw_diff: [B, T-1, H, W] raw differences (with mask, for TADC and MAG)
        """
        # Compute raw difference (with mask)
        raw_diff = self.diff(x)  # [B, T-1, H, W]
        
        # Residual connection
        residual = self.shortcut(raw_diff)
        
        # Encode with convolutions
        out = self.conv1(raw_diff)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # CBAM attention
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        # Residual add
        out = out + residual
        diff_feat = self.relu(out)
        
        return diff_feat, raw_diff


__all__ = ['DifferencePath', 'Differential']
