"""
DeepDIG: Deep Infrared Small Target Detection.

Multi-branch spatio-temporal framework for infrared small moving target detection.

Architecture:
    Static Path (DualTaskUNet) → F_static [B, C, H, W]
    Difference Path → F_diff [B, C, H, W] + raw_diff [B, T-1, H, W]
    Dynamic Path (TADC) → F_dynamic [B, C, H, W]
    MAG Module → F_fused [B, C, H, W]
    Detection Network → output [B, 1, H, W]
"""

import torch
import torch.nn as nn

from .static_path import StaticPath
from .difference_path import DifferencePath
from .dynamic_path import DynamicPath
from .MAG_module import MAGModule
from .detection_network import DetectionNetwork, ResBlock
from .frame_cache import FrameAlignmentManager


class DeepDIG(nn.Module):
    """
    DeepDIG base model.
    
    Assumes input sequence is already aligned. For alignment support,
    use DeepDIG_WithCache.
    
    Args:
        input_channels: Sequence length T (default: 20)
        deep_supervision: Enable deep supervision (default: False)
        detection_channels: Feature channels (default: 64)
    """
    
    def __init__(self, input_channels=20, deep_supervision=False, detection_channels=64):
        super().__init__()
        
        self.input_channels = input_channels
        self.deep_supervision = deep_supervision
        self.detection_channels = detection_channels
        
        # Static Path
        self.static_path = StaticPath(
            detection_channels=detection_channels,
            enable_xfeat=True
        )
        self.sp = self.static_path  # alias
        
        # Difference Path
        self.difference_path = DifferencePath(
            seq_len=input_channels,
            out_channels=detection_channels
        )
        
        # Dynamic Path (TADC)
        self.dynamic_path = DynamicPath(
            seq_len=input_channels,
            out_channels=detection_channels
        )
        
        # MAG Module (uses raw_diff for gate, matching STDMANet)
        self.mag_module = MAGModule(
            in_channels=detection_channels,
            gate_channels=32,
            out_channels=detection_channels,
            raw_diff_channels=input_channels - 1  # T-1 channels for raw_diff
        )
        
        # Detection Network
        self.detection_network = DetectionNetwork(
            input_channels=detection_channels,
            deep_supervision=deep_supervision,
            block=ResBlock
        )

    def forward(self, x_seq, extract_xfeat=False):
        """
        Args:
            x_seq: [B, T, H, W] aligned input sequence
            extract_xfeat: Whether to extract XFeat outputs
        
        Returns:
            Output mask [B, 1, H, W] or list if deep_supervision
        """
        # Static Path: current frame only
        x_curr = x_seq[:, -1].unsqueeze(1)
        
        if extract_xfeat:
            static_feat = self.static_path(x_curr, mode='detection')
            xfeat_out = self.static_path(x_curr, mode='xfeat')
            self.xfeat_outputs = {
                'descriptors': xfeat_out['descriptors'],
                'keypoints': xfeat_out['keypoints'],
                'heatmap': xfeat_out['heatmap'],
            }
        else:
            static_feat = self.static_path(x_curr, mode='detection')
        
        # Difference Path
        diff_feat, raw_diff = self.difference_path(x_seq)
        
        # Dynamic Path (TADC)
        dynamic_feat = self.dynamic_path(x_seq, raw_diff)
        
        # MAG Module (pass raw_diff for gate generation)
        fused_feat = self.mag_module(static_feat, diff_feat, dynamic_feat, raw_diff=raw_diff)
        
        # Detection Network
        output = self.detection_network(fused_feat)
        
        return output

    def extract_descriptors(self, image, threshold=0.1, top_k=1024):
        """Extract descriptors for DBA alignment."""
        return self.sp.extract_descriptors(image, threshold=threshold, top_k=top_k)

    def get_xfeat_outputs(self):
        """Get XFeat outputs (call forward with extract_xfeat=True first)."""
        if hasattr(self, 'xfeat_outputs'):
            return self.xfeat_outputs
        raise RuntimeError('XFeat outputs not available.')


class DeepDIG_WithCache(DeepDIG):
    """
    DeepDIG with frame alignment support.
    
    Extends DeepDIG with:
    - Frame caching for inference
    - Batch descriptor alignment for training
    
    Args:
        input_channels: Sequence length T
        deep_supervision: Enable deep supervision
        detection_channels: Feature channels
        window_size: Cache window size
        match_threshold: Descriptor matching threshold
        min_matches: Minimum matches for valid alignment
        descriptor_threshold: Keypoint detection threshold
        max_keypoints: Maximum keypoints to extract
    """
    
    def __init__(
        self,
        input_channels=20,
        deep_supervision=False,
        detection_channels=64,
        window_size=20,
        match_threshold=0.7,
        min_matches=10,
        descriptor_threshold=0.25,
        max_keypoints=2048,
    ):
        super().__init__(
            input_channels=input_channels,
            deep_supervision=deep_supervision,
            detection_channels=detection_channels,
        )
        
        # Frame Alignment Manager
        self.alignment_manager = FrameAlignmentManager(
            window_size=window_size,
            match_threshold=match_threshold,
            min_matches=min_matches,
        )
        
        self.window_size = window_size
        self.descriptor_threshold = descriptor_threshold
        self.max_keypoints = max_keypoints
    
    def enable_cache(self):
        self.alignment_manager.enable_cache()
    
    def disable_cache(self):
        self.alignment_manager.disable_cache()
    
    def reset_cache(self):
        self.alignment_manager.reset_cache()

    def forward(self, input, extract_xfeat=False, use_cache=None, 
                reset_cache=False, align_mode='none'):
        """
        Args:
            input: [B, 1, H, W] single frame (cache mode) or
                   [B, T, H, W] sequence (batch mode)
            extract_xfeat: Extract XFeat outputs
            use_cache: Use cache mode (None = use internal flag)
            reset_cache: Reset cache before processing
            align_mode: 'none' | 'descriptor' | 'cache'
        
        Returns:
            align_mode='none': output
            other modes: (output, alignment_stats)
        """
        if use_cache is None:
            use_cache = self.alignment_manager.use_cache
        
        if reset_cache:
            self.reset_cache()
        
        # Batch alignment mode (training)
        if align_mode == 'descriptor' and not use_cache:
            B, T, H, W = input.shape
            
            # Extract descriptors for all frames
            with torch.no_grad():
                all_frames = input.view(B * T, 1, H, W)
                desc_output = self.extract_descriptors(
                    all_frames,
                    threshold=self.descriptor_threshold,
                    top_k=self.max_keypoints
                )
                desc_list = [desc_output['descriptors'][i:i+1] for i in range(B * T)]
                kpts_list = desc_output['keypoints']
            
            # Align sequence
            aligned_data, align_stats = self.alignment_manager.align_sequence(
                input, desc_list, kpts_list
            )
            
            output = super().forward(aligned_data, extract_xfeat=extract_xfeat)
            return output, align_stats
        
        # Cache mode (inference)
        elif use_cache or align_mode == 'cache':
            if input.dim() != 4 or input.shape[1] != 1:
                raise ValueError(f"Cache mode requires [B, 1, H, W], got {input.shape}")
            
            B, C, H, W = input.shape
            if B != 1:
                raise ValueError(f"Cache mode requires batch_size=1, got {B}")
            
            current_frame = input[0, 0]
            
            # Extract descriptors
            desc_output = self.extract_descriptors(
                input,
                threshold=self.descriptor_threshold,
                top_k=self.max_keypoints,
            )
            
            current_desc = desc_output['descriptors']
            current_kpts = desc_output['keypoints'][0]
            
            # Build aligned sequence
            aligned_sequence, align_stats = self.alignment_manager.align_with_cache(
                current_frame, current_desc, current_kpts
            )
            
            sequence_input = aligned_sequence.unsqueeze(0)
            output = super().forward(sequence_input, extract_xfeat=extract_xfeat)
            
            return output, align_stats
        
        # No alignment (already aligned)
        else:
            return super().forward(input, extract_xfeat=extract_xfeat)


def build_deep_dig(
    input_channels=20,
    deep_supervision=False,
    detection_channels=64,
    with_cache=True,
    window_size=20,
    match_threshold=0.7,
    min_matches=10,
    descriptor_threshold=0.25,
    max_keypoints=2048,
    **kwargs,
):
    """
    Build DeepDIG model.
    
    Args:
        input_channels: Sequence length T
        deep_supervision: Enable deep supervision
        detection_channels: Feature channels
        with_cache: Include alignment support
        window_size: Cache window size
        match_threshold: Descriptor matching threshold
        min_matches: Minimum matches for alignment
        descriptor_threshold: Keypoint detection threshold
        max_keypoints: Maximum keypoints
    
    Returns:
        DeepDIG or DeepDIG_WithCache instance
    """
    if with_cache:
        return DeepDIG_WithCache(
            input_channels=input_channels,
            deep_supervision=deep_supervision,
            detection_channels=detection_channels,
            window_size=window_size,
            match_threshold=match_threshold,
            min_matches=min_matches,
            descriptor_threshold=descriptor_threshold,
            max_keypoints=max_keypoints,
        )
    else:
        return DeepDIG(
            input_channels=input_channels,
            deep_supervision=deep_supervision,
            detection_channels=detection_channels,
        )


# Backward compatibility aliases
build_stdmanet_with_cache = build_deep_dig
STDMANet_WithCache = DeepDIG_WithCache


__all__ = [
    'DeepDIG',
    'DeepDIG_WithCache',
    'build_deep_dig',
    'build_stdmanet_with_cache',
    'STDMANet_WithCache',
]
