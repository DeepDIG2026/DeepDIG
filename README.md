<h1 align="center">DeepDIG: Deep Background Alignment Helps See Infrared Small Target Better</h1>

DeepDIG detects infrared small targets in moving-camera video through background alignment and spatiotemporal modeling.

This repository provides the **DBA + RADC + MAG** model, evaluation code and pretrained checkpoints. Figures and results below are from the paper; please refer to it for method details.

## Method

<table align="center">
  <tr>
    <td align="center" valign="middle" width="50%"><img src="assets/alignment_example_tmm.png" alt="Figure 1: frame differences and feature responses under background motion" width="460"></td>
    <td align="center" valign="middle" width="50%"><img src="assets/architecture_tmm.png" alt="Figure 2: DeepDIG architecture with static, DBA, difference and dynamic paths" width="460"></td>
  </tr>
  <tr>
    <td align="center" valign="top"><strong>Figure 1.</strong> Frame differences and feature responses under background motion.</td>
    <td align="center" valign="top"><strong>Figure 2.</strong> DeepDIG aligns historical frames and combines static, difference and dynamic features for detection.</td>
  </tr>
</table>

### Deep Background Alignment (DBA)

DBA reuses static-encoder features to construct deep descriptors for cross-frame background alignment.

### Reliability-Aware Dynamic Convolution (RADC)

RADC combines **Reliability-Aware Temporal Aggregation (RATA)** and **Content-Adaptive Dynamic Convolution (CADC)** to refine temporal features using reliable motion cues.

<table align="center">
  <tr>
    <td align="center" valign="middle" width="50%"><img src="assets/rata_tmm.png" alt="Figure 3: regional and pixel-wise reliability gates in RATA" width="460"></td>
    <td align="center" valign="middle" width="50%"><img src="assets/cadc_tmm.png" alt="Figure 4: motion-conditioned dynamic kernel generation in CADC" width="460"></td>
  </tr>
  <tr>
    <td align="center" valign="top"><strong>Figure 3.</strong> RATA weights temporal differences using regional reliability and pixel-wise discriminability.</td>
    <td align="center" valign="top"><strong>Figure 4.</strong> CADC generates spatially adaptive dynamic kernels from accumulated motion cues.</td>
  </tr>
</table>

### Motion-guided Adaptive Gating (MAG)

<p align="center">
  <img src="assets/mag_tmm.png" alt="Figure 5: Motion-guided Adaptive Gating" width="460">
</p>

<p align="center"><strong>Figure 5.</strong> MAG uses inter-frame differences to guide adaptive fusion of spatial and temporal features.</p>

## LMIRSTD Dataset

The **Large-motion Multi-frame IRST Detection (LMIRSTD)** dataset contains **60 infrared sequences** (**45 training / 15 test**), with 200 frames per sequence at **640×512** resolution (width×height), totaling **12,000 frames**.

LMIRSTD features pronounced background motion and dim targets, with mean SCR **2.25**. Small, medium and large targets account for **32.0%, 50.6% and 17.4%**, respectively.

<p align="center">
  <img src="assets/dataset_tmm.png" alt="Figure 6: typical LMIRSTD backgrounds; green boxes indicate true targets" width="900">
</p>

<p align="center"><strong>Figure 6.</strong> Representative LMIRSTD backgrounds, with true targets marked by green boxes.</p>

<p align="center">
  <img src="assets/panorama_tmm.png" alt="Figure 7: panoramas stitched from 50 consecutive LMIRSTD frames" width="900">
</p>

<p align="center"><strong>Figure 7.</strong> Panoramas stitched from 50 consecutive LMIRSTD frames; red boxes mark individual 640×512 frames.</p>

**Table I** compares multi-frame IRST detection datasets.

| Dataset | Mean SCR | Motion Mag. |
|---|---:|---:|
| IRDST | 6.70 | 40.17 pixels |
| TSIRMT | 3.04 | 17.69 pixels |
| LMIRSTD | 2.25 | 60.44 pixels |

- **Mean SCR**: mean signal-to-clutter ratio, indicating target contrast relative to background clutter; lower values indicate less salient targets.
- **Motion Mag.**: mean pixel displacement of the background between the two frames exhibiting the largest inter-frame motion within each temporal window.

Download LMIRSTD from the [dataset folder](https://drive.google.com/drive/folders/1dp31_tDtGXmNUhTHD_0GqEGBLAVNUYb0).

## Paper Results

### Quantitative comparison

<p align="center">
  <img src="assets/results_tmm.png" alt="Table II: quantitative comparison on IRDST, TSIRMT and LMIRSTD" width="100%">
</p>

<p align="center"><strong>Table II.</strong> Quantitative comparison on IRDST, TSIRMT and LMIRSTD; Fa is in units of 10<sup>−6</sup>, and all other metrics are percentages.</p>

### Qualitative comparison

<p align="center">
  <img src="assets/comparison_tmm.png" alt="Figure 8: eight qualitative examples comparing DeepDIG with representative methods" width="100%">
</p>

<p align="center"><strong>Figure 8.</strong> Qualitative comparison: red, blue and yellow boxes mark correct detections, missed targets and false alarms, respectively.</p>

### Efficiency

Table III reports full-pipeline efficiency, including background alignment and inference, at **256×256** resolution on an **NVIDIA RTX 4090D**.

| Method | FPS ↑ | Parameters | GFLOPs |
|---|---:|---:|---:|
| STDMANet | 4.37 | 11.88M | 62.98 |
| LMAFormer | 0.65 | 407.67M | 615.96 |
| TDCNet | 2.90 | 40.46M | 3822.91 |
| DeepDIG | 9.15 | 15.21M | 33.14 |

## Installation

Environment: Python 3.10 and PyTorch 2.6.0 (CUDA 12.6).

```bash
conda create -n deepdig python=3.10
conda activate deepdig
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Pretrained checkpoints

Download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1duJ4M6fwDdcLkgSOd1911EqphCNk4BV1) and place them in `weights/`.

| Dataset | Filename |
|---|---|
| IRDST | `DeepDIG_IRDST.pth` |
| TSIRMT | `DeepDIG_TSIRMT.pth` |
| LMIRSTD | `DeepDIG_LMIRSTD.pth` |

## Test

Place IRDST in `dataset/IRDST/` and run from the repository root:

```bash
python test_deepdig.py --dataset IRDST --root ./dataset \
  --ckpt ./weights/DeepDIG_IRDST.pth --in_channels 20 --test_batch_size 2 \
  --output_dir ./results/IRDST
```

Results are saved to `results/IRDST/`. For TSIRMT or LMIRSTD, change `--dataset`, `--ckpt` and `--output_dir` accordingly.
