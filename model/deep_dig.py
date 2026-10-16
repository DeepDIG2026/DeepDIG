"""Top-level DeepDIG model and optional DBA-backed frame management."""

import torch
import torch.nn as nn

from .detection_network import DetectionNetwork, ResBlock
from .difference_path import DifferencePath
from .dynamic_path import DynamicPath
from .frame_cache import FrameAlignmentManager
from .MAG_module import MAGModule
from .static_path import StaticPath


class DeepDIG(nn.Module):
    """Run static, difference, and dynamic paths on an aligned frame sequence."""

    def __init__(
        self,
        sequence_length=20,
        deep_supervision=False,
        detection_channels=64,
        rata_use_region=True,
        rata_use_pixel=True,
        use_dynamic_path=True,
        use_mag=True,
        cadc_kernel_size=3,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.deep_supervision = deep_supervision
        self.detection_channels = detection_channels
        self.use_dynamic_path = bool(use_dynamic_path)
        self.use_mag = bool(use_mag)
        self.static_path = StaticPath(
            detection_channels=detection_channels, enable_descriptors=True
        )
        self.difference_path = DifferencePath(
            sequence_length=sequence_length, out_channels=detection_channels
        )
        self.dynamic_path = DynamicPath(
            sequence_length=sequence_length,
            out_channels=detection_channels,
            cadc_kernel_size=cadc_kernel_size,
            rata_use_region=rata_use_region,
            rata_use_pixel=rata_use_pixel,
        )
        self.mag_module = MAGModule(
            in_channels=detection_channels,
            gate_channels=32,
            out_channels=detection_channels,
            difference_channels=sequence_length - 1,
        )
        self.detection_network = DetectionNetwork(
            feature_channels=detection_channels,
            deep_supervision=deep_supervision,
            block=ResBlock,
        )

    def forward(self, aligned_sequence, extract_descriptors=False):
        """Return a target mask from ``[B, T, H, W]`` aligned frames."""
        current_frame = aligned_sequence[:, -1].unsqueeze(1)
        static_feature = self.static_path(current_frame, mode="detection")
        if extract_descriptors:
            descriptor_output = self.static_path(current_frame, mode="descriptor")
            self.descriptor_outputs = {
                "descriptors": descriptor_output["descriptors"],
                "keypoints": descriptor_output["keypoints"],
                "heatmap": descriptor_output["heatmap"],
            }

        differential_feature, raw_difference = self.difference_path(aligned_sequence)
        dynamic_feature = (
            self.dynamic_path(aligned_sequence, raw_difference)
            if self.use_dynamic_path
            else None
        )
        if self.use_mag:
            fused_feature = self.mag_module(
                static_feature,
                differential_feature,
                dynamic_feature,
                raw_difference=raw_difference,
            )
        else:
            features = [static_feature, differential_feature]
            if dynamic_feature is not None:
                features.append(dynamic_feature)
            fused_feature = sum(features) / len(features)
        return self.detection_network(fused_feature)

    def extract_descriptors(self, image, threshold=0.1, top_k=1024):
        """Extract descriptors and keypoints for DBA."""
        return self.static_path.extract_descriptors(
            image, threshold=threshold, top_k=top_k
        )

    def get_descriptor_outputs(self):
        """Return descriptor outputs cached by the latest forward pass."""
        if hasattr(self, "descriptor_outputs"):
            return self.descriptor_outputs
        raise RuntimeError("Descriptor outputs are not available")


class DeepDIG_WithCache(DeepDIG):
    """Extend DeepDIG with batch DBA and streaming frame-cache alignment."""

    def __init__(
        self,
        sequence_length=20,
        deep_supervision=False,
        detection_channels=64,
        window_size=None,
        match_threshold=0.7,
        min_matches=10,
        descriptor_threshold=0.25,
        max_keypoints=2048,
        rata_use_region=True,
        rata_use_pixel=True,
        use_dynamic_path=True,
        use_mag=True,
        cadc_kernel_size=3,
    ):
        super().__init__(
            sequence_length=sequence_length,
            deep_supervision=deep_supervision,
            detection_channels=detection_channels,
            rata_use_region=rata_use_region,
            rata_use_pixel=rata_use_pixel,
            use_dynamic_path=use_dynamic_path,
            use_mag=use_mag,
            cadc_kernel_size=cadc_kernel_size,
        )
        window_size = sequence_length if window_size is None else window_size
        if window_size != sequence_length:
            raise ValueError("window_size must equal sequence_length")
        self.alignment_manager = FrameAlignmentManager(
            window_size=window_size,
            match_threshold=match_threshold,
            min_matches=min_matches,
        )
        self.window_size = window_size
        self.descriptor_threshold = descriptor_threshold
        self.max_keypoints = max_keypoints

    def enable_cache(self):
        """Enable streaming frame-cache mode."""
        self.alignment_manager.enable_cache()

    def disable_cache(self):
        """Disable streaming frame-cache mode."""
        self.alignment_manager.disable_cache()

    def reset_cache(self):
        """Clear all cached frames and descriptors."""
        self.alignment_manager.reset_cache()

    def forward(
        self,
        model_input,
        extract_descriptors=False,
        use_cache=None,
        reset_cache=False,
        align_mode="none",
    ):
        """Run direct, descriptor-aligned, or cached inference."""
        if use_cache is None:
            use_cache = self.alignment_manager.use_cache
        if reset_cache:
            self.reset_cache()

        if align_mode == "descriptor" and not use_cache:
            batch_size, sequence_length, height, width = model_input.shape
            with torch.no_grad():
                all_frames = model_input.view(
                    batch_size * sequence_length, 1, height, width
                )
                descriptor_output = self.extract_descriptors(
                    all_frames,
                    threshold=self.descriptor_threshold,
                    top_k=self.max_keypoints,
                )
                descriptor_list = [
                    descriptor_output["descriptors"][index : index + 1]
                    for index in range(batch_size * sequence_length)
                ]
            aligned_sequence, alignment_statistics = (
                self.alignment_manager.align_sequence(
                    model_input,
                    descriptor_list,
                    descriptor_output["keypoints"],
                )
            )
            output = super().forward(
                aligned_sequence, extract_descriptors=extract_descriptors
            )
            return output, alignment_statistics

        if use_cache or align_mode == "cache":
            if model_input.dim() != 4 or model_input.shape[1] != 1:
                raise ValueError(
                    f"Cache mode requires [B, 1, H, W], got {model_input.shape}"
                )
            if model_input.shape[0] != 1:
                raise ValueError("Cache mode requires batch_size=1")
            descriptor_output = self.extract_descriptors(
                model_input,
                threshold=self.descriptor_threshold,
                top_k=self.max_keypoints,
            )
            aligned_sequence, alignment_statistics = (
                self.alignment_manager.align_with_cache(
                    model_input[0, 0],
                    descriptor_output["descriptors"],
                    descriptor_output["keypoints"][0],
                )
            )
            output = super().forward(
                aligned_sequence.unsqueeze(0), extract_descriptors=extract_descriptors
            )
            return output, alignment_statistics

        return super().forward(model_input, extract_descriptors=extract_descriptors)


def build_deep_dig(
    sequence_length=20,
    deep_supervision=False,
    detection_channels=64,
    with_cache=True,
    window_size=None,
    match_threshold=0.7,
    min_matches=10,
    descriptor_threshold=0.25,
    max_keypoints=2048,
    rata_use_region=True,
    rata_use_pixel=True,
    use_dynamic_path=True,
    use_mag=True,
    cadc_kernel_size=3,
):
    """Build DeepDIG with or without frame alignment and caching support."""
    common = dict(
        sequence_length=sequence_length,
        deep_supervision=deep_supervision,
        detection_channels=detection_channels,
        rata_use_region=rata_use_region,
        rata_use_pixel=rata_use_pixel,
        use_dynamic_path=use_dynamic_path,
        use_mag=use_mag,
        cadc_kernel_size=cadc_kernel_size,
    )
    if not with_cache:
        return DeepDIG(**common)
    return DeepDIG_WithCache(
        **common,
        window_size=window_size,
        match_threshold=match_threshold,
        min_matches=min_matches,
        descriptor_threshold=descriptor_threshold,
        max_keypoints=max_keypoints,
    )


__all__ = ["DeepDIG", "DeepDIG_WithCache", "build_deep_dig"]
