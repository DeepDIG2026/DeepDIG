"""
DeepDIG: Deep Infrared Small Target Detection.

Multi-branch spatio-temporal framework for infrared small moving target detection.

Modules:
    - TADC: Temporal Accumulated Dynamic Convolution (accumulated_kernel)
    - MAG: Motion-Aware Gating (MotionGuidedGatingFusion)
    - DBA: Descriptor-Based Alignment
    - StaticPath, DifferencePath, DynamicPath
    - DetectionNetwork (MSHNet-based)
"""

# TADC Module (accumulated_kernel)
from .TADC_module import TADCModule

# Dynamic Path
from .dynamic_path import DynamicPath

# Difference Path
from .difference_path import DifferencePath, Differential

# MAG Module (MotionGuidedGatingFusion)
from .MAG_module import MAGModule, FeatureEnhancement, FusionRefine

# Static Path (includes DualTaskUNet)
from .static_path import StaticPath, DualTaskUNet

# DBA Module
from .DBA_module import DBAModule

# Frame Alignment Manager
from .frame_cache import FrameAlignmentManager, FrameCache

# Detection Network
from .detection_network import DetectionNetwork, ResBlock

# Main Classes
from .deep_dig import (
    DeepDIG,
    DeepDIG_WithCache,
    build_deep_dig,
    build_stdmanet_with_cache,
    STDMANet_WithCache,
)

# UNet Components
from .UNet_CBAM import Up_CBAM, DoubleConv, Down, OutConv


__all__ = [
    # TADC Module
    'TADCModule',
    
    # Dynamic Path
    'DynamicPath',
    
    # Difference Path
    'DifferencePath',
    'Differential',
    
    # MAG Module
    'MAGModule',
    'FeatureEnhancement',
    'FusionRefine',
    
    # Static Path
    'StaticPath',
    'DualTaskUNet',
    
    # DBA Module
    'DBAModule',
    
    # Frame Alignment
    'FrameAlignmentManager',
    'FrameCache',
    
    # Detection Network
    'DetectionNetwork',
    'ResBlock',
    
    # Main Classes
    'DeepDIG',
    'DeepDIG_WithCache',
    'build_deep_dig',
    'build_stdmanet_with_cache',
    'STDMANet_WithCache',
    
    # UNet Components
    'Up_CBAM',
    'DoubleConv',
    'Down',
    'OutConv',
]
