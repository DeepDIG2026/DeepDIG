"""Reliability-Aware Dynamic Convolution for temporal feature enhancement.

RADC consists of Reliability-Aware Temporal Aggregation (RATA) followed by
Content-Adaptive Dynamic Convolution (CADC).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityAwareTemporalAggregation(nn.Module):
    """Aggregate frame differences with region- and pixel-level reliability.

    Region-level temporal reliability weights ``W^r`` are inferred on a grid.
    Pixel-wise spatial discriminability weights ``W^p`` are inferred from four
    complementary descriptors: raw magnitude, temporal anomaly, local contrast,
    and peak dominance. Both residual weights initialize to one.
    """

    def __init__(
        self,
        sequence_length_minus_one,
        stat_hidden=16,
        gate_hidden=8,
        region_range=0.5,
        pixel_range=0.5,
        time_encoding_dim=2,
        topk_ratio=0.01,
        grid_size=4,
        use_region=True,
        use_pixel=True,
    ):
        super().__init__()
        if sequence_length_minus_one < 1:
            raise ValueError(
                "sequence_length_minus_one must be at least 1, "
                f"got {sequence_length_minus_one}"
            )

        self.sequence_length_minus_one = sequence_length_minus_one
        self.region_range = float(region_range)
        self.pixel_range = float(pixel_range)
        self.time_encoding_dim = int(time_encoding_dim)
        self.topk_ratio = float(topk_ratio)
        self.grid_size = int(grid_size)
        self.use_region = bool(use_region)
        self.use_pixel = bool(use_pixel)

        self.cache_intermediates = False
        self.debug_cache = {}
        self._region_reliability_drive = None
        self._pixel_discriminability_drive = None
        self._last_region_reliability_weight = None
        self._last_pixel_discriminability_weight = None

        if self.use_region:
            self.register_buffer(
                "time_encoding",
                self._build_time_encoding(
                    sequence_length_minus_one, self.time_encoding_dim
                ),
                persistent=False,
            )
            input_dim = 4 + 2 * self.time_encoding_dim
            self.region_reliability_mlp = nn.Sequential(
                nn.Linear(input_dim, stat_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(stat_hidden, 1),
            )
            self.temporal_prior = nn.Parameter(
                torch.zeros(sequence_length_minus_one)
            )
            nn.init.zeros_(self.region_reliability_mlp[-1].weight)
            if self.region_reliability_mlp[-1].bias is not None:
                nn.init.zeros_(self.region_reliability_mlp[-1].bias)
        else:
            self.region_reliability_mlp = None
            self.temporal_prior = None

        if self.use_pixel:
            self.pixel_discriminability_gate = nn.Sequential(
                nn.Conv2d(4, gate_hidden, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(gate_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    gate_hidden,
                    gate_hidden,
                    3,
                    padding=4,
                    dilation=4,
                    bias=False,
                ),
                nn.BatchNorm2d(gate_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, 1, 1),
            )
            nn.init.zeros_(self.pixel_discriminability_gate[-1].weight)
            if self.pixel_discriminability_gate[-1].bias is not None:
                nn.init.zeros_(self.pixel_discriminability_gate[-1].bias)
        else:
            self.pixel_discriminability_gate = None

    @staticmethod
    def _build_time_encoding(length, number_of_pairs):
        """Build a sinusoidal encoding for difference-frame indices."""
        time = torch.arange(length, dtype=torch.float32) / max(length - 1, 1)
        columns = []
        for index in range(number_of_pairs):
            frequency = math.pi * (2**index)
            columns.append(torch.sin(frequency * time))
            columns.append(torch.cos(frequency * time))
        return torch.stack(columns, dim=-1)

    def forward(self, raw_difference):
        """Return the reliability-weighted temporal difference aggregation."""
        batch_size, difference_count, height, width = raw_difference.shape
        if difference_count != self.sequence_length_minus_one:
            raise ValueError(
                f"raw_difference has T-1={difference_count}, but the module "
                f"was built with T-1={self.sequence_length_minus_one}"
            )

        raw_magnitude = raw_difference.abs()

        if self.use_region:
            grid_size = self.grid_size
            pooled_magnitude = F.max_pool2d(
                raw_magnitude, kernel_size=4, stride=4
            )
            pooled_height, pooled_width = pooled_magnitude.shape[2:]
            pad_height = (grid_size - pooled_height % grid_size) % grid_size
            pad_width = (grid_size - pooled_width % grid_size) % grid_size
            if pad_height > 0 or pad_width > 0:
                pooled_magnitude = F.pad(
                    pooled_magnitude, (0, pad_width, 0, pad_height)
                )
                pooled_height, pooled_width = pooled_magnitude.shape[2:]

            cell_height = pooled_height // grid_size
            cell_width = pooled_width // grid_size
            # Rearrange pixels into G x G regions without changing values.
            region_values = pooled_magnitude.view(
                batch_size,
                difference_count,
                grid_size,
                cell_height,
                grid_size,
                cell_width,
            )
            region_values = region_values.permute(0, 1, 2, 4, 3, 5).reshape(
                batch_size, difference_count, grid_size * grid_size, -1
            )

            topk_count = max(1, int(region_values.size(-1) * self.topk_ratio))
            topk_values = region_values.topk(topk_count, dim=-1).values
            region_mean = topk_values.mean(dim=-1)
            # A singleton Top-K set has zero dispersion; avoid an undefined
            # sample standard deviation without changing larger-set statistics.
            region_std = (
                topk_values.std(dim=-1)
                if topk_count > 1
                else torch.zeros_like(region_mean)
            )
            region_max = topk_values[:, :, :, 0]
            region_sparsity = region_max / (region_mean + 1e-6)
            region_statistics = torch.stack(
                [region_mean, region_std, region_max, region_sparsity], dim=-1
            )
            log_region_statistics = torch.log1p(region_statistics)
            time_encoding = self.time_encoding.view(
                1, difference_count, 1, -1
            ).expand(
                batch_size,
                difference_count,
                grid_size * grid_size,
                -1,
            )
            reliability_input = torch.cat(
                [log_region_statistics, time_encoding], dim=-1
            )
            content_reliability = self.region_reliability_mlp(
                reliability_input
            ).squeeze(-1)
            region_reliability_drive = content_reliability + self.temporal_prior.view(
                1, difference_count, 1
            )
            region_grid_weight = 1.0 + self.region_range * torch.tanh(
                region_reliability_drive
            )
            region_weight_2d = region_grid_weight.view(
                batch_size * difference_count, 1, grid_size, grid_size
            )
            region_reliability_weight = F.interpolate(
                region_weight_2d,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).view(batch_size, difference_count, height, width)
            region_drive_summary = region_reliability_drive.mean(dim=-1)
            region_weight_summary = region_reliability_weight.mean(dim=(-2, -1))
        else:
            pooled_magnitude = region_values = region_statistics = None
            log_region_statistics = reliability_input = None
            region_reliability_drive = region_grid_weight = None
            region_drive_summary = region_weight_summary = None
            region_reliability_weight = torch.ones(
                batch_size,
                difference_count,
                height,
                width,
                device=raw_difference.device,
                dtype=raw_difference.dtype,
            )

        if self.use_pixel:
            temporal_peak = raw_magnitude.max(dim=1, keepdim=True).values
            temporal_anomaly = raw_magnitude - 0.5 * temporal_peak
            local_mean = F.avg_pool2d(
                raw_magnitude, kernel_size=9, stride=1, padding=4
            )
            local_contrast = raw_magnitude - local_mean
            local_peak = F.max_pool2d(
                raw_magnitude, kernel_size=5, stride=1, padding=2
            )
            peak_dominance = raw_magnitude / (local_peak + 1e-6)
            # Stack the four complementary descriptors as pixel-gate channels.
            descriptor_tensor = torch.stack(
                [
                    raw_magnitude,
                    temporal_anomaly,
                    local_contrast,
                    peak_dominance,
                ],
                dim=2,
            ).view(batch_size * difference_count, 4, height, width)
            pixel_discriminability_drive = self.pixel_discriminability_gate(
                descriptor_tensor
            ).view(batch_size, difference_count, height, width)
            pixel_discriminability_weight = 1.0 + self.pixel_range * torch.tanh(
                pixel_discriminability_drive
            )
        else:
            temporal_peak = temporal_anomaly = local_contrast = None
            peak_dominance = pixel_discriminability_drive = None
            pixel_discriminability_weight = torch.ones(
                batch_size,
                difference_count,
                height,
                width,
                device=raw_difference.device,
                dtype=raw_difference.dtype,
            )

        reliability_weight = (
            region_reliability_weight * pixel_discriminability_weight
        )
        aggregated_difference = (
            reliability_weight * raw_magnitude
        ).sum(dim=1, keepdim=True)

        self._region_reliability_drive = region_drive_summary
        self._pixel_discriminability_drive = pixel_discriminability_drive
        self._last_region_reliability_weight = region_reliability_weight
        self._last_pixel_discriminability_weight = pixel_discriminability_weight

        if self.cache_intermediates:
            self.debug_cache = {
                "raw_magnitude": raw_magnitude.detach(),
                "region_reliability_weight": (
                    region_weight_summary.detach()
                    if region_weight_summary is not None
                    else torch.ones(
                        batch_size,
                        difference_count,
                        device=raw_difference.device,
                        dtype=raw_difference.dtype,
                    )
                ),
                "pixel_discriminability_weight": pixel_discriminability_weight.detach(),
                "reliability_weight": reliability_weight.detach(),
                "aggregated_difference": aggregated_difference.detach(),
                "temporal_prior": (
                    self.temporal_prior.detach()
                    if self.temporal_prior is not None
                    else torch.zeros(
                        difference_count,
                        device=raw_difference.device,
                        dtype=raw_difference.dtype,
                    )
                ),
            }
            if self.use_region:
                self.debug_cache.update(
                    {
                        "pooled_magnitude": pooled_magnitude.detach(),
                        "region_values": region_values.detach(),
                        "region_statistics": region_statistics.detach(),
                        "region_reliability_drive": region_reliability_drive.detach(),
                        "region_grid_weight": region_grid_weight.detach(),
                        "region_reliability_map": region_reliability_weight.detach(),
                    }
                )
            if self.use_pixel:
                self.debug_cache.update(
                    {
                        "temporal_anomaly": temporal_anomaly.detach(),
                        "local_contrast": local_contrast.detach(),
                        "peak_dominance": peak_dominance.detach(),
                        "pixel_discriminability_drive": pixel_discriminability_drive.detach(),
                    }
                )

        return aggregated_difference

    def regularization_loss(self, prior_coefficient=1e-3, drive_coefficient=1e-4):
        """Return L2 penalties for the prior and latest reliability drives."""
        anchor = self.temporal_prior
        if anchor is None:
            anchor = self._pixel_discriminability_drive
        if anchor is None:
            return torch.zeros((), dtype=torch.float32)

        loss = torch.zeros((), device=anchor.device, dtype=anchor.dtype)
        if self.temporal_prior is not None:
            loss = loss + prior_coefficient * self.temporal_prior.pow(2).mean()
        if self._region_reliability_drive is not None:
            loss = loss + drive_coefficient * self._region_reliability_drive.pow(2).mean()
        if self._pixel_discriminability_drive is not None:
            loss = loss + drive_coefficient * self._pixel_discriminability_drive.pow(2).mean()
        return loss

    @torch.no_grad()
    def get_diagnostics(self, saturation_threshold=2.0):
        """Return scalar diagnostics from the latest RATA forward pass."""
        nan = float("nan")
        if self.temporal_prior is not None:
            prior_l2 = float(self.temporal_prior.pow(2).mean().sqrt().item())
            prior_max_abs = float(self.temporal_prior.abs().max().item())
        else:
            prior_l2 = prior_max_abs = nan
        diagnostics = {
            "region_saturation": nan,
            "pixel_saturation": nan,
            "region_drive_abs_mean": nan,
            "pixel_drive_abs_mean": nan,
            "region_weight_mean": nan,
            "region_weight_std": nan,
            "pixel_weight_mean": nan,
            "pixel_weight_std": nan,
            "prior_l2": prior_l2,
            "prior_max_abs": prior_max_abs,
            "region_range": self.region_range if self.use_region else 0.0,
            "pixel_range": self.pixel_range if self.use_pixel else 0.0,
            "use_region": float(self.use_region),
            "use_pixel": float(self.use_pixel),
        }
        if self._region_reliability_drive is not None:
            drive = self._region_reliability_drive.detach()
            diagnostics["region_saturation"] = float(
                (drive.abs() > saturation_threshold).float().mean().item()
            )
            diagnostics["region_drive_abs_mean"] = float(drive.abs().mean().item())
        if self._pixel_discriminability_drive is not None:
            drive = self._pixel_discriminability_drive.detach()
            diagnostics["pixel_saturation"] = float(
                (drive.abs() > saturation_threshold).float().mean().item()
            )
            diagnostics["pixel_drive_abs_mean"] = float(drive.abs().mean().item())
        if self._last_region_reliability_weight is not None:
            weight = self._last_region_reliability_weight.detach()
            diagnostics["region_weight_mean"] = float(weight.mean().item())
            diagnostics["region_weight_std"] = float(weight.std().item())
        if self._last_pixel_discriminability_weight is not None:
            weight = self._last_pixel_discriminability_weight.detach()
            diagnostics["pixel_weight_mean"] = float(weight.mean().item())
            diagnostics["pixel_weight_std"] = float(weight.std().item())
        return diagnostics


class ContentAdaptiveDynamicConvolution(nn.Module):
    """Generate and apply pixel-wise dynamic kernels from aggregated motion."""

    def __init__(self, sequence_length, kernel_size=3, reduction=4):
        super().__init__()
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        intermediate_channels = max(16, sequence_length // reduction)
        self.kernel_generator = nn.Sequential(
            nn.Conv2d(sequence_length, intermediate_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, sequence_length * kernel_size * kernel_size, 1, bias=False),
        )
        self.bias = nn.Parameter(torch.ones(1) * -1.0)

    def generate_kernel(self, motion_sequence):
        """Generate bounded pixel-wise kernels from a motion sequence."""
        batch_size, sequence_length, height, width = motion_sequence.shape
        kernel_elements = self.kernel_size * self.kernel_size
        raw_kernel = self.kernel_generator(motion_sequence).view(
            batch_size, sequence_length, kernel_elements, height, width
        )
        return (torch.sigmoid(raw_kernel + self.bias) + 0.01) * 10

    def forward(self, aligned_sequence, dynamic_kernel):
        """Apply dynamic kernels to unfolded spatial neighborhoods."""
        batch_size, sequence_length, height, width = aligned_sequence.shape
        kernel_elements = self.kernel_size * self.kernel_size
        neighborhoods = F.unfold(
            aligned_sequence.view(batch_size * sequence_length, 1, height, width),
            kernel_size=self.kernel_size,
            padding=self.padding,
        ).view(batch_size, sequence_length, kernel_elements, height * width)
        flattened_kernel = dynamic_kernel.view(
            batch_size, sequence_length, kernel_elements, height * width
        )
        return (flattened_kernel * neighborhoods).sum(dim=2).view(
            batch_size, sequence_length, height, width
        )


class ReliabilityAwareDynamicConvolution(nn.Module):
    """Enhance an aligned sequence with RATA followed by CADC."""

    def __init__(
        self,
        sequence_length,
        kernel_size=3,
        reduction=4,
        rata_use_region=True,
        rata_use_pixel=True,
    ):
        super().__init__()
        if sequence_length < 2:
            raise ValueError(f"sequence_length must be at least 2, got {sequence_length}")
        self.sequence_length = sequence_length
        self.cache_intermediates = False
        self.debug_cache = {}
        self.rata = ReliabilityAwareTemporalAggregation(
            sequence_length_minus_one=sequence_length - 1,
            use_region=rata_use_region,
            use_pixel=rata_use_pixel,
        )
        self.cadc = ContentAdaptiveDynamicConvolution(
            sequence_length=sequence_length,
            kernel_size=kernel_size,
            reduction=reduction,
        )

    def forward(self, aligned_sequence, raw_difference):
        """Return the RADC-enhanced sequence with unchanged input shape."""
        batch_size, _, height, width = raw_difference.shape
        aggregated_difference = self.rata(raw_difference)
        normalized_motion = (
            aggregated_difference - aggregated_difference.mean()
        ) / (aggregated_difference.std() + 1e-6)
        motion_sequence = normalized_motion.expand(
            batch_size, self.sequence_length, height, width
        )
        dynamic_kernel = self.cadc.generate_kernel(motion_sequence)
        enhanced_sequence = self.cadc(aligned_sequence, dynamic_kernel)

        if self.cache_intermediates:
            self.debug_cache = {
                "aligned_sequence": aligned_sequence.detach(),
                "raw_difference": raw_difference.detach(),
                "motion_sequence": motion_sequence.detach(),
                "dynamic_kernel": dynamic_kernel.detach(),
                "kernel_mean": dynamic_kernel.mean(dim=2).detach(),
                "kernel_max": dynamic_kernel.amax(dim=2).detach(),
                "enhanced_sequence": enhanced_sequence.detach(),
            }
        return enhanced_sequence


__all__ = [
    "ReliabilityAwareDynamicConvolution",
    "ReliabilityAwareTemporalAggregation",
    "ContentAdaptiveDynamicConvolution",
]
