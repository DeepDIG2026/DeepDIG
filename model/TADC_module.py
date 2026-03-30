"""
Temporal Accumulated Dynamic Convolution (TADC) Module.

This module performs motion-guided temporal enhancement through three stages:
(1) Motion accumulation from frame differences
(2) Dynamic kernel generation via multi-scale dilated convolutions  
(3) Pixel-wise adaptive enhancement

This is the accumulated_kernel implementation from the original STDMANet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TADCModule(nn.Module):
    """
    Temporal Accumulated Dynamic Convolution module for dynamic path enhancement.
    
    This implements the accumulated_kernel functionality from STDMANet.
    
    Args:
        seq_len: Number of input frames T
        kernel_size: Spatial kernel size for enhancement (default: 3)
        reduction: Channel reduction ratio (default: 4)
    """
    
    def __init__(self, seq_len, kernel_size=3, reduction=4):
        super().__init__()
        
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
         
        # Multi-scale dilated conv for kernel generation
        mid_ch = max(16, seq_len // reduction)
        self.kernel_generator = nn.Sequential(
            nn.Conv2d(seq_len, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, seq_len * kernel_size * kernel_size, 1, bias=False)
        )
        
        self.bias = nn.Parameter(torch.ones(1) * -1.0)
    
    def _accumulate(self, raw_diff):
        """Accumulate frame differences into motion signal."""
        B, T_minus_1, H, W = raw_diff.shape
        
        # Weighted sum of absolute differences
        accumulated = torch.zeros_like(raw_diff[:, :1])
        for i in range(T_minus_1):
            accumulated = accumulated + torch.abs(raw_diff[:, i:i+1])
        
        # Normalize
        motion = (accumulated - accumulated.mean()) / (accumulated.std() + 1e-6)
        
        # Expand to full sequence length
        return motion.expand(B, self.seq_len, H, W)

    def _generate_kernel(self, motion_seq):
        """Generate pixel-wise attention kernels from motion signal."""
        B, T, H, W = motion_seq.shape
        K = self.kernel_size
        
        raw_kernel = self.kernel_generator(motion_seq)
        raw_kernel = raw_kernel.view(B, T, K * K, H, W)
        
        # Linear mapping to [0.01, 10.01] range
        return (torch.sigmoid(raw_kernel + self.bias) + 0.01) * 10
    
    def _enhance(self, x, kernel):
        """Apply attention kernels to enhance input sequence."""
        B, T, H, W = x.shape
        K = self.kernel_size
        
        # Unfold spatial neighborhoods
        x_unfold = F.unfold(
            x.view(B * T, 1, H, W),
            kernel_size=K, padding=self.padding
        ).view(B, T, K * K, H * W)
        
        # Weighted sum
        kernel_flat = kernel.view(B, T, K * K, H * W)
        return (kernel_flat * x_unfold).sum(dim=2).view(B, T, H, W)
    
    def forward(self, x_aligned, raw_diff):
        """
        Args:
            x_aligned: Aligned frame sequence [B, T, H, W]
            raw_diff: Frame differences [B, T-1, H, W]
        
        Returns:
            Enhanced sequence [B, T, H, W]
        """
        motion_seq = self._accumulate(raw_diff)
        kernel = self._generate_kernel(motion_seq)
        return self._enhance(x_aligned, kernel)


__all__ = ['TADCModule']
