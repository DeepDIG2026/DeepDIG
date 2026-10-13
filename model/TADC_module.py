"""
Reliability-Aware Dynamic Convolution core for DeepDIG.

AdaptiveDiffAccumulator implements RATA; TADCModule normalizes its motion
prior, generates a separate spatial kernel for each temporal position, and
applies CADC to the aligned sequence. The legacy TADC names and parameter
paths are retained so existing complete DeepDIG checkpoints load strictly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDiffAccumulator(nn.Module):
    """
    Reliability-Aware Temporal Aggregation (RATA) used by the dynamic path.

    Regional gate: max-pool absolute differences by four, partition the pooled
    map into grid_size x grid_size regions, summarize the strongest topk_ratio
    responses with mean/std/max/peak-to-mean statistics, and concatenate a
    sinusoidal time encoding. An MLP plus a learned temporal prior produces
    residual weights, which are bilinearly resized to the original resolution.

    Pixel gate: encode magnitude, temporal anomaly, 9x9 local contrast and
    5x5 local peak ratio with dilated convolutions. Multiply the two residual
    gates and sum the full-resolution absolute differences over time.

    Defaults: grid_size=4, topk_ratio=0.01, time_enc_dim=2, and both residual
    ranges=0.5. Numerical stability constants in the descriptors are 1e-6.
    Zero-initialized final gate layers recover uniform accumulation.

    Disabling either gate supports ablations. Diagnostics and the
    regularization helper are retained without changing inference.
    """

    def __init__(self, seq_len_minus_1, stat_hidden=16, gate_hidden=8,
                 global_range=0.5, spatial_range=0.5, time_enc_dim=2,
                 topk_ratio=0.01,
                 grid_size=4,       # 【新增】网格大小，默认 4x4
                 use_global=True, use_spatial=True):
        super().__init__()

        if seq_len_minus_1 < 1:
            raise ValueError(f"seq_len_minus_1 must be >= 1, got {seq_len_minus_1}")

        self.Tm1 = seq_len_minus_1
        self.global_range = float(global_range)
        self.spatial_range = float(spatial_range)
        self.time_enc_dim = int(time_enc_dim)
        self.topk_ratio = float(topk_ratio)
        self.grid_size = int(grid_size) # 新增属性
        self.use_global = bool(use_global)
        self.use_spatial = bool(use_spatial)

        self.cache_intermediates = False
        self.debug_cache = {}

        self._drive_global = None
        self._drive_spatial = None
        self._last_w_global = None
        self._last_w_spatial = None

        # ----- Regional Grid branch (原 Global branch) ----------------------
        if self.use_global:
            self.register_buffer(
                'time_enc',
                self._build_time_encoding(seq_len_minus_1, self.time_enc_dim),
                persistent=False,
            )

            in_dim = 4 + 2 * self.time_enc_dim
            self.content_mlp = nn.Sequential(
                nn.Linear(in_dim, stat_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(stat_hidden, 1),
            )
            self.temporal_prior = nn.Parameter(torch.zeros(seq_len_minus_1))

            nn.init.zeros_(self.content_mlp[-1].weight)
            if self.content_mlp[-1].bias is not None:
                nn.init.zeros_(self.content_mlp[-1].bias)
        else:
            self.content_mlp = None
            self.temporal_prior = None

        # ----- Spatial branch (C) (保持你上一次的 9x9 局部对比度修改) -------
        if self.use_spatial:
            self.spatial_gate = nn.Sequential(
                # 输入通道变为 4 (新增 local_peak)
                nn.Conv2d(4, gate_hidden, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(gate_hidden),
                nn.ReLU(inplace=True),
                # 【关键】大膨胀卷积，感受野大幅提升，用于识别建筑网格和长线条
                nn.Conv2d(gate_hidden, gate_hidden, 3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(gate_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, 1, 1),
            )
            nn.init.zeros_(self.spatial_gate[-1].weight)
            if self.spatial_gate[-1].bias is not None:
                nn.init.zeros_(self.spatial_gate[-1].bias)
        else:
            self.spatial_gate = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_time_encoding(Tm1: int, n_pairs: int) -> torch.Tensor:
        # (保持不变)
        t = torch.arange(Tm1, dtype=torch.float32) / max(Tm1 - 1, 1)
        cols = []
        for k in range(n_pairs):
            f = math.pi * (2 ** k)
            cols.append(torch.sin(f * t))
            cols.append(torch.cos(f * t))
        return torch.stack(cols, dim=-1)

    def forward(self, raw_diff):
        B, Tm1, H, W = raw_diff.shape
        if Tm1 != self.Tm1:
            raise ValueError(f"raw_diff has T-1={Tm1} but module was built with T-1={self.Tm1}")

        abs_diff = raw_diff.abs()

        # ---- (A+B) Regional Grid Weight w_regional -------------------------
        if self.use_global:
            G = self.grid_size
            # 1. 快速下采样保留极值 (降维)
            pooled = F.max_pool2d(abs_diff, kernel_size=4, stride=4)
            Hp, Wp = pooled.shape[2:]

            # 2. 补齐边缘，确保能被网格大小 G 整除
            pad_h = (G - Hp % G) % G
            pad_w = (G - Wp % G) % G
            if pad_h > 0 or pad_w > 0:
                pooled = F.pad(pooled, (0, pad_w, 0, pad_h))
                Hp, Wp = pooled.shape[2:]

            hg, wg = Hp // G, Wp // G

            # 3. 将图像重组为 G*G 个网格
            # shape 演变: [B, Tm1, G, hg, G, wg] -> [B, Tm1, G*G, hg*wg]
            grid_flat = pooled.view(B, Tm1, G, hg, G, wg)
            grid_flat = grid_flat.permute(0, 1, 2, 4, 3, 5).reshape(B, Tm1, G * G, -1)

            # 4. 在每个网格内部独立计算 Top-K 统计量
            K = max(1, int(grid_flat.size(-1) * self.topk_ratio))
            topk_vals = grid_flat.topk(K, dim=-1).values  # [B, Tm1, G*G, K]

            d_mean = topk_vals.mean(dim=-1)               # [B, Tm1, G*G]
            d_std = topk_vals.std(dim=-1)
            d_max = topk_vals[:, :, :, 0]
            d_sparsity = d_max / (d_mean + 1e-6)

            stats = torch.stack([d_mean, d_std, d_max, d_sparsity], dim=-1) # [B, Tm1, G*G, 4]
            stats_log = torch.log1p(stats)

            # 5. MLP 推理 (拼接时间编码)
            time_enc = self.time_enc.view(1, Tm1, 1, -1).expand(B, Tm1, G * G, -1)
            mlp_in = torch.cat([stats_log, time_enc], dim=-1)

            # w_content: [B, Tm1, G*G]
            w_content = self.content_mlp(mlp_in).squeeze(-1)

            # temporal_prior: [Tm1] 广播到 [B, Tm1, G*G]
            drive_regional = w_content + self.temporal_prior.view(1, Tm1, 1)
            w_grid = 1.0 + self.global_range * torch.tanh(drive_regional) # [B, Tm1, G*G]

            # 6. 平滑放大回原图尺寸
            w_grid_2d = w_grid.view(B * Tm1, 1, G, G)
            # 使用双线性插值，消除网格之间的生硬边界
            w_regional_full = F.interpolate(w_grid_2d, size=(H, W), mode='bilinear', align_corners=False)
            w_global = w_regional_full.view(B, Tm1, H, W)

            # （为了兼容代码下方的 drive 收集逻辑）
            drive_global = drive_regional.mean(dim=-1) # 简单平均作为全局诊断指标
            w_global_diag = w_global.mean(dim=(-2, -1))
        else:
            pooled = grid_flat = stats = stats_log = mlp_in = None
            w_content = drive_regional = w_grid = w_global_diag = drive_global = None
            w_global = torch.ones(B, Tm1, H, W, device=raw_diff.device, dtype=raw_diff.dtype)

        # ---- (C) Per-pixel spatial gate w_spatial --------------------------
        if self.use_spatial:
            max_abs = abs_diff.max(dim=1, keepdim=True).values
            anomaly = abs_diff - 0.5 * max_abs

            local_mean = F.avg_pool2d(abs_diff, kernel_size=9, stride=1, padding=4)
            local_contrast = abs_diff - local_mean

            # 【新增】：局部峰值/稀疏度 (Local Peak)
            # 使用 5x5 最大池化寻找局部极值点
            local_max = F.max_pool2d(abs_diff, kernel_size=5, stride=1, padding=2)
            # 避免除 0，当前像素占局部最大值的比例。如果是孤立点，这个值接近1。
            # 如果是平缓的云层或建筑面，这个值也接近1，但会被 anomaly 和 local_contrast 抑制。
            local_peak = abs_diff / (local_max + 1e-6)

            # 输入 4 个通道：绝对差分，时域异常，空间高通，局部极值特征
            gate_in = torch.stack([abs_diff, anomaly, local_contrast, local_peak], dim=2)
            gate_in = gate_in.view(B * Tm1, 4, H, W)

            spatial_logit = self.spatial_gate(gate_in).view(B, Tm1, H, W)
            w_spatial = 1.0 + self.spatial_range * torch.tanh(spatial_logit)
        else:
            max_abs = anomaly = local_contrast = local_peak = spatial_logit = None
            w_spatial = torch.ones(B, Tm1, H, W, device=raw_diff.device, dtype=raw_diff.dtype)

        # ---- Combine and accumulate ---------------------------------------
        # 注意：现在 w_global 和 w_spatial 的 shape 都是 [B, Tm1, H, W]
        # 直接相乘即可，不再需要 unsqueeze
        w = w_global * w_spatial
        accumulated = (w * abs_diff).sum(dim=1, keepdim=True)

        # ---- Save state for reg / diagnostics (保持不变) --------------------
        self._drive_global = drive_global
        self._drive_spatial = spatial_logit
        self._last_w_global = w_global
        self._last_w_spatial = w_spatial

        if self.cache_intermediates:
            cache = {
                'abs_diff': abs_diff.detach(),
                # Backward-compatible temporal summary for existing monitors.
                'w_global': (
                    w_global_diag.detach()
                    if w_global_diag is not None
                    else torch.ones(B, Tm1, device=raw_diff.device, dtype=raw_diff.dtype)
                ),
                'w_spatial': w_spatial.detach(),
                'weight': w.detach(),
                'accumulated': accumulated.detach(),
                'temporal_prior': (
                    self.temporal_prior.detach()
                    if self.temporal_prior is not None
                    else torch.zeros(Tm1, device=raw_diff.device, dtype=raw_diff.dtype)
                ),
            }
            if self.use_global:
                cache.update({
                    'pooled_abs_diff': pooled.detach(),
                    'grid_abs_diff': grid_flat.detach(),
                    'stats': stats.mean(dim=2).detach(),
                    'stats_grid': stats.detach(),
                    'drive_global': drive_global.detach(),
                    'drive_regional': drive_regional.detach(),
                    'w_grid': w_grid.detach(),
                    'w_regional_full': w_global.detach(),
                })
            if self.use_spatial:
                cache.update({
                    'max_abs': max_abs.detach(),
                    'anomaly': anomaly.detach(),
                    'local_contrast': local_contrast.detach(),
                    'local_peak': local_peak.detach(),
                    'spatial_logit': spatial_logit.detach(),
                })
            self.debug_cache = cache

        return accumulated

    # ------------------------------------------------------------------ #
    # (A) Regularization
    # ------------------------------------------------------------------ #

    def regularization_loss(self, lam_prior: float = 1e-3,
                            lam_drive: float = 1e-4) -> torch.Tensor:
        """
        L2 penalty designed to keep tanh away from saturation:

            L = lam_prior · ||temporal_prior||² / Tm1
              + lam_drive · ( mean(drive_global²) + mean(spatial_logit²) )

        Must be called *after* a forward pass in the same step (so the
        cached drives are still attached to the autograd graph). If no
        forward has run yet, the drive terms are silently zero.

        For ablation runs that disable a branch (use_global=False or
        use_spatial=False), the corresponding term is automatically dropped.
        """
        # Build a zero scalar that sits on the right device / dtype so that
        # gradient bookkeeping stays consistent even when both branches are
        # disabled (e.g. ablation A0 = naive sum).
        anchor = None
        if self.temporal_prior is not None:
            anchor = self.temporal_prior
        elif self._drive_spatial is not None:
            anchor = self._drive_spatial

        if anchor is None:
            return torch.zeros((), dtype=torch.float32)

        L = torch.zeros((), device=anchor.device, dtype=anchor.dtype)
        if self.temporal_prior is not None:
            L = L + lam_prior * self.temporal_prior.pow(2).mean()
        if self._drive_global is not None:
            L = L + lam_drive * self._drive_global.pow(2).mean()
        if self._drive_spatial is not None:
            L = L + lam_drive * self._drive_spatial.pow(2).mean()
        return L

    # ------------------------------------------------------------------ #
    # (D) Diagnostics
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_diagnostics(self, sat_thresh: float = 2.0) -> dict:
        """
        Return scalar health indicators of the latest forward pass.

        sat_thresh: |drive| above this counts as "in tanh's saturation tail"
                    (tanh(2)=0.964, tanh(3)=0.995). Default 2.0.

        Returns a flat dict of Python floats (NaN if forward has not run).
        """
        nan = float('nan')
        if self.temporal_prior is not None:
            prior_l2 = float(self.temporal_prior.pow(2).mean().sqrt().item())
            prior_max_abs = float(self.temporal_prior.abs().max().item())
        else:
            prior_l2 = nan
            prior_max_abs = nan

        out = {
            'sat_global':        nan,
            'sat_spatial':       nan,
            'drive_global_abs_mean': nan,
            'drive_spatial_abs_mean': nan,
            'w_global_mean':     nan,
            'w_global_std':      nan,
            'w_spatial_mean':    nan,
            'w_spatial_std':     nan,
            'prior_l2':          prior_l2,
            'prior_max_abs':     prior_max_abs,
            'alpha':             self.global_range if self.use_global else 0.0,
            'beta':              self.spatial_range if self.use_spatial else 0.0,
            'use_global':        float(self.use_global),
            'use_spatial':       float(self.use_spatial),
        }

        if self._drive_global is not None:
            dg = self._drive_global.detach()
            out['sat_global'] = float((dg.abs() > sat_thresh).float().mean().item())
            out['drive_global_abs_mean'] = float(dg.abs().mean().item())
        if self._drive_spatial is not None:
            ds = self._drive_spatial.detach()
            out['sat_spatial'] = float((ds.abs() > sat_thresh).float().mean().item())
            out['drive_spatial_abs_mean'] = float(ds.abs().mean().item())
        if self._last_w_global is not None:
            wg = self._last_w_global.detach()
            out['w_global_mean'] = float(wg.mean().item())
            out['w_global_std']  = float(wg.std().item())
        if self._last_w_spatial is not None:
            ws = self._last_w_spatial.detach()
            out['w_spatial_mean'] = float(ws.mean().item())
            out['w_spatial_std']  = float(ws.std().item())
        return out


class TADCModule(nn.Module):
    """
    RADC core, retaining the TADCModule name for checkpoint compatibility.

    This implements the accumulated_kernel functionality from STDMANet.

    Args:
        seq_len: Number of input frames T
        kernel_size: Spatial kernel size for enhancement (default: 3)
        reduction: Channel reduction ratio (default: 4)
        accumulator_use_global: Pass-through to AdaptiveDiffAccumulator;
            set False to disable the per-diff scalar branch (ablation).
        accumulator_use_spatial: Pass-through to AdaptiveDiffAccumulator;
            set False to disable the per-pixel spatial gate (ablation).
    """

    def __init__(self, seq_len, kernel_size=3, reduction=4,
                 accumulator_use_global=True, accumulator_use_spatial=True):
        super().__init__()

        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")

        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.cache_intermediates = False
        self.debug_cache = {}

        # Adaptive accumulator: replaces the naive uniform sum of |raw_diff|.
        # See AdaptiveDiffAccumulator for motivation and design.
        self.accumulator = AdaptiveDiffAccumulator(
            seq_len_minus_1=seq_len - 1,
            use_global=accumulator_use_global,
            use_spatial=accumulator_use_spatial,
        )

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
        """Accumulate frame differences into motion signal (adaptive)."""
        B, _, H, W = raw_diff.shape

        # Adaptive weighted accumulation (content + temporal-prior + spatial gate).
        accumulated = self.accumulator(raw_diff)

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

        # Positive bounded mapping: coefficients are in (0.1, 10.1).
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
        enhanced = self._enhance(x_aligned, kernel)

        if self.cache_intermediates:
            self.debug_cache = {
                'tadc_input': x_aligned.detach(),
                'raw_diff': raw_diff.detach(),
                'motion_seq': motion_seq.detach(),
                'kernel': kernel.detach(),
                'tadc_kernel_mean': kernel.mean(dim=2).detach(),
                'tadc_kernel_max': kernel.amax(dim=2).detach(),
                'tadc_output': enhanced.detach(),
            }

        return enhanced


__all__ = ['TADCModule', 'AdaptiveDiffAccumulator']
