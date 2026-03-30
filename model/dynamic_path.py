"""
Dynamic Path (matching STDMANet's Dynamic_block + AccumulatedKernelADKE).

Contains TADC Module and Feature Encoder with CBAM for temporal motion feature enhancement.

Data flow (see Figure 2 & Figure 3):
    X_aligned [B, T, H, W] (DBA-aligned sequence)
    raw_diff [B, T-1, H, W] (raw differences, from Difference Path)
        │
        ▼
    ┌─────────────────┐
    │   TADC Module   │  (difference accumulation → kernel generation → temporal enhancement)
    └────────┬────────┘
             │
             ▼
    enhanced_seq [B, T, H, W]
             │
             ▼
    ┌─────────────────┐
    │ Feature Encoder │  (Conv + CBAM: T → C)
    └────────┬────────┘
             │
             ▼
    F_dynamic [B, C, H, W] (dynamic features)
"""

import torch
import torch.nn as nn

from .TADC_module import TADCModule


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


class DynamicPath(nn.Module):
    """
    Dynamic Path (matching STDMANet's Dynamic_block + AccumulatedKernelADKE).
    
    Contains TADC Module and Feature Encoder with CBAM attention.
    
    Components:
        1. TADC Module: Temporal Accumulated Dynamic Convolution enhancement
        2. Feature Encoder: Conv + CBAM encodes enhanced sequence to features
    
    Data flow:
        X_aligned [B, T, H, W] + raw_diff [B, T-1, H, W]
            ↓
        TADC Module (accumulate → kernel generation → enhance)
            ↓
        enhanced_seq [B, T, H, W]
            ↓
        Feature Encoder (Conv + CBAM)
            ↓
        F_dynamic [B, C, H, W]
    
    Args:
        seq_len: Input sequence length T
        out_channels: Output feature channels (default: 64)
        kernel_size: TADC module kernel size (default: 3)
        reduction: TADC module channel reduction ratio (default: 4)
    """
    
    def __init__(self, seq_len, out_channels=64, kernel_size=3, reduction=4):
        super().__init__()
        
        self.seq_len = seq_len
        self.out_channels = out_channels
        
        # TADC Module
        self.tadc = TADCModule(
            seq_len=seq_len,
            kernel_size=kernel_size,
            reduction=reduction
        )
        
        # Feature Encoder with CBAM (matching Dynamic_block)
        self.conv1 = nn.Conv2d(seq_len, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut for residual connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(seq_len, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # CBAM attention (matching STDMANet)
        ratio = 16 if out_channels >= 16 else max(1, out_channels // 4)
        self.ca = ChannelAttention(out_channels, ratio=ratio)
        self.sa = SpatialAttention()
    
    def forward(self, x_aligned, raw_diff):
        """
        Forward pass.
        
        Args:
            x_aligned: [B, T, H, W] DBA-aligned sequence
            raw_diff: [B, T-1, H, W] raw differences (from Difference Path)
        
        Returns:
            F_dynamic: [B, C, H, W] dynamic features
        """
        # TADC Module
        enhanced_seq = self.tadc(x_aligned, raw_diff)
        
        # Residual connection (from ORIGINAL x_aligned, matching STDMANet Dynamic_block)
        residual = self.shortcut(x_aligned)
        
        # Encode with convolutions
        out = self.conv1(enhanced_seq)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # CBAM attention
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        # Residual add
        out = out + residual
        dynamic_feat = self.relu(out)
        
        return dynamic_feat


__all__ = ['DynamicPath']
