"""Public model API for the DeepDIG detection architecture."""

from .DBA_module import DBAModule
from .checkpoint import load_deepdig_checkpoint
from .deep_dig import DeepDIG, DeepDIG_WithCache, build_deep_dig
from .detection_network import DetectionNetwork, ResBlock
from .difference_path import DifferencePath, FrameDifference
from .dynamic_path import DynamicPath
from .frame_cache import FrameAlignmentManager, FrameCache
from .MAG_module import FusionRefinement, GatingGenerator, MAGModule
from .RADC_module import (
    ContentAdaptiveDynamicConvolution,
    ReliabilityAwareDynamicConvolution,
    ReliabilityAwareTemporalAggregation,
)
from .static_path import DualTaskUNet, StaticPath
from .UNet_CBAM import DoubleConv, Down, OutConv, Up, Up_CBAM

__all__ = [
    "DeepDIG",
    "DeepDIG_WithCache",
    "build_deep_dig",
    "load_deepdig_checkpoint",
    "DBAModule",
    "ReliabilityAwareDynamicConvolution",
    "ReliabilityAwareTemporalAggregation",
    "ContentAdaptiveDynamicConvolution",
    "MAGModule",
    "GatingGenerator",
    "FusionRefinement",
    "StaticPath",
    "DifferencePath",
    "FrameDifference",
    "DynamicPath",
    "FrameAlignmentManager",
    "FrameCache",
    "DetectionNetwork",
    "ResBlock",
    "DualTaskUNet",
    "Up_CBAM",
    "Up",
    "DoubleConv",
    "Down",
    "OutConv",
]
