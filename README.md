<h1 align="center">DeepDIG: Deep Background Alignment Helps See Infrared Small Target Better</h1>

<p align="center">Wufan Wang, Haixin Li, Baiwen Zeng, Chenzhi Zhao, Zhiqiang Yu, Bo Zhang, Hui Gao, and Wendong Wang</p>
<p align="center">Manuscript submitted to <strong>IEEE Transactions on Multimedia (TMM)</strong>.</p>

DeepDIG detects small infrared targets in video sequences with camera ego-motion and complex background dynamics. It aligns the background before modeling target motion, then combines spatial appearance and temporal cues to preserve weak targets and suppress interference.

- **Deep Background Alignment (DBA):** Reuses intermediate spatial features to construct deep local descriptors for cross-frame alignment.
- **Reliability-Aware Dynamic Convolution (RADC):** Combines Reliability-Aware Temporal Aggregation (RATA) and Content-Adaptive Dynamic Convolution (CADC) to suppress alignment-induced artifacts and refine temporal representations.
- **Motion-guided Adaptive Gating (MAG):** Adaptively integrates spatial and temporal features using motion guidance.
- **LMIRSTD:** A multi-frame infrared small target detection dataset with pronounced background motion and dim targets.

This repository provides the complete **DBA + RADC + MAG** inference model, including both RATA gates, and the evaluation entry point [test_deepdig.py](test_deepdig.py). Download the [pretrained checkpoints](https://drive.google.com/drive/folders/1duJ4M6fwDdcLkgSOd1911EqphCNk4BV1) for the three datasets. The figures and experimental results presented below are from the manuscript.

<p align="center">
  <img src="assets/alignment_example_tmm.png" alt="Figure 1: frame differences and feature responses under background motion" width="760">
</p>

## Method

<p align="center">
  <img src="assets/architecture_tmm.png" alt="Figure 2: DeepDIG architecture with static, DBA, difference and dynamic paths" width="1000">
</p>

DeepDIG takes a sequence of infrared frames and predicts a target probability map for the current frame. The static path extracts spatial features. DBA aligns historical frames with the current frame using descriptors derived from the static encoder. The difference path encodes inter-frame changes, while the dynamic path refines the aligned sequence through RADC. MAG fuses the static, difference and dynamic features before the detection head produces the final prediction.

### Deep Background Alignment (DBA)

DBA builds on the intermediate features of the static-branch encoder. An XFeat-based head predicts local descriptors, keypoint locations and reliability scores. Descriptor matching, subpixel refinement and RANSAC provide a homography for warping historical frames into the current frame's coordinate system.

### Reliability-Aware Dynamic Convolution (RADC)

RADC consists of **RATA** and **CADC**. RATA estimates the reliability of inter-frame differences, and CADC uses the resulting motion prior to refine the aligned sequence.

**RATA.** Given absolute difference maps $D_i = |I_t-I_i|$, RATA accumulates motion as

$$F_{\mathrm{accum}} = \sum_i (W_i^r \odot W_i^p) \odot D_i.$$

The regional gate uses Top-K statistics (mean, standard deviation, maximum and peak-to-mean ratio), temporal position encoding and a learned temporal prior. The pixel-wise gate uses difference magnitude, temporal anomaly, local contrast and peak dominance. Both use bounded residual weighting of the form $1+\alpha\tanh(z)$, allowing attenuation and enhancement around unit weighting. Top-K selection constructs regional descriptors; all pixels remain in the weighted accumulation.

**CADC.** The normalized accumulated motion guides parallel dilated convolutions with dilation rates 1, 2 and 4. A kernel prediction head produces spatially varying dynamic coefficients using sigmoid scaling and a positive offset. These motion-conditioned kernels refine the aligned frames, and the dynamic encoder produces features for fusion.

<table align="center">
  <tr><th>RATA (Figure 3)</th><th>CADC (Figure 4)</th></tr>
  <tr>
    <td><img src="assets/rata_tmm.png" alt="Regional and pixel-wise reliability gates in RATA" width="460"></td>
    <td><img src="assets/cadc_tmm.png" alt="Motion-conditioned dynamic kernel generation in CADC" width="460"></td>
  </tr>
</table>

### Motion-guided Adaptive Gating (MAG)

MAG uses inter-frame difference information to guide the enhancement of static and dynamic features. The enhanced features are combined with the difference features and refined through a residual attention block, providing the detection head with complementary spatial and temporal information.

<p align="center">
  <img src="assets/mag_tmm.png" alt="Figure 5: Motion-guided Adaptive Gating" width="850">
</p>

### Source map

| Component | Source |
|---|---|
| Complete model | [deep_dig.py](model/deep_dig.py) |
| Static path and descriptor head | [static_path.py](model/static_path.py), [UNet_CBAM.py](model/UNet_CBAM.py) |
| DBA and frame alignment | [DBA_module.py](model/DBA_module.py), [frame_cache.py](model/frame_cache.py) |
| Difference path | [difference_path.py](model/difference_path.py) |
| RATA and CADC | [TADC_module.py](model/TADC_module.py) |
| Dynamic feature encoder | [dynamic_path.py](model/dynamic_path.py) |
| MAG | [MAG_module.py](model/MAG_module.py) |
| Detection head | [detection_network.py](model/detection_network.py) |

`AdaptiveDiffAccumulator` implements RATA. The `TADCModule` class and existing parameter paths are retained for checkpoint compatibility; together, its aggregation and dynamic-convolution components implement the RADC core.

## LMIRSTD Dataset

The **Large-motion Multi-frame IRST Detection (LMIRSTD)** dataset contains **60 real-world infrared sequences**, split into **45 training sequences and 15 test sequences**. Each sequence contains 200 frames at **640×512** resolution (width×height), for **12,000 frames** in total. Scenes include cloudy skies, urban areas, forests, mountains and lakes under different seasons and illumination conditions.

LMIRSTD combines pronounced camera-induced background motion with dim, low-contrast targets. Its mean signal-to-clutter ratio (SCR) is **2.25**. Small, medium and large targets account for **32.0%, 50.6% and 17.4%**, respectively.

<p align="center">
  <img src="assets/dataset_tmm.png" alt="Figure 6: typical LMIRSTD backgrounds; green boxes indicate true targets" width="1000">
</p>

<p align="center">
  <img src="assets/panorama_tmm.png" alt="Figure 7: panoramas stitched from 50 consecutive LMIRSTD frames" width="1000">
</p>

The panoramas are stitched from **50 consecutive frames**. Each red box shows the spatial extent of a single 640×512 frame.

The dataset comparison in Table I reports normalized inter-frame background displacement (NIBD): the mean displacement of a fixed 20×20 grid under the estimated background similarity transform, divided by the image diagonal and expressed as a percentage. P50 and P95 are the median and 95th percentile over accepted adjacent-frame pairs; screening coverage is the percentage of candidate pairs passing geometric checks.

| Dataset | Mean SCR | Background motion | NIBD P50 (% diagonal/frame) | NIBD P95 (% diagonal/frame) | Screening coverage (%) |
|---|---:|---|---:|---:|---:|
| IRDST | 6.70 | Large | 0.144 | 3.713 | 71.65 |
| TSIRMT | 3.04 | Mild | 0.460 | 0.918 | 99.50 |
| LMIRSTD | 2.25 | Large | 0.329 | 1.020 | 96.71 |

Download LMIRSTD from the [dataset folder](https://drive.google.com/drive/folders/1dp31_tDtGXmNUhTHD_0GqEGBLAVNUYb0).

## Paper Results

### Quantitative comparison

Table II compares single-frame and multi-frame methods on IRDST, TSIRMT and LMIRSTD. The complete table is reproduced directly from the manuscript. **Fa is in units of $10^{-6}$; all other metrics are percentages.**

<p align="center">
  <img src="assets/results_tmm.png" alt="Table II: quantitative comparison on IRDST, TSIRMT and LMIRSTD" width="100%">
</p>

The complete **DeepDIG (DBA + RADC + MAG)** results reported in Table II are:

| Dataset | Pd ↑ | Fa ↓ | IoU ↑ | nIoU ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|
| IRDST | 98.92 | 14.67 | 63.26 | 66.04 | 98.17 |
| TSIRMT | 92.66 | 35.55 | 73.09 | 73.11 | 94.72 |
| LMIRSTD | 91.24 | 1.87 | 80.66 | 74.80 | 89.76 |

### Qualitative comparison

<p align="center">
  <img src="assets/comparison_tmm.png" alt="Figure 8: eight qualitative examples comparing DeepDIG with representative methods" width="1000">
</p>

Columns show the input, UIUNet, DNANet, STDMANet, LMAFormer, DTUM, DeepPro-Plus, TDCNet, DeepDIG and ground truth. Red, blue and yellow boxes mark correct detections, missed targets and false alarms; corner insets enlarge challenging regions.

### Efficiency

Table III reports full-pipeline efficiency, including background alignment and inference, at **256×256** resolution on an **NVIDIA RTX 4090D**.

| Method | FPS ↑ | Parameters | GFLOPs |
|---|---:|---:|---:|
| STDMANet | 4.37 | 11.88M | 62.98 |
| LMAFormer | 0.65 | 407.67M | 615.96 |
| TDCNet | 2.90 | 40.46M | 3822.91 |
| DeepDIG | **9.15** | 15.21M | **33.14** |

### Background alignment

Figure 9 evaluates synthetic homography perturbations of LMIRSTD test images. Easy, medium and hard settings use maximum corner displacements of 10%, 30% and 50% of image size. The paper reports 100% alignment success for DeepDIG at these three tested levels and **5.54 ms per image pair**.

<p align="center">
  <img src="assets/alignment_analysis_tmm.png" alt="Figure 9: alignment success rate and pairwise alignment time" width="900">
</p>

Table IV compares alignment methods on **IRDST**, with RADC and MAG retained:

| Method | Pd ↑ | Fa ↓ | IoU ↑ | nIoU ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|
| DeepDIG (SIFT + RADC + MAG) | 96.88 | 73.58 | 54.22 | 62.35 | 95.64 |
| DeepDIG (GIM + RADC + MAG) | 98.80 | **12.55** | 61.40 | 63.82 | **98.27** |
| DeepDIG (DBA + RADC + MAG) | **98.92** | 14.67 | **63.26** | **66.04** | 98.17 |

### Sensitivity and feature analysis

The reported IRDST sensitivity experiments study temporal window length, residual-gate scaling, Top-K pooling and dynamic-kernel size. Among the tested settings, the paper selects **N=20**, **α=0.5**, **K=40** and a **3×3** kernel.

<p align="center">
  <img src="assets/sensitivity_temporal_tmm.png" alt="Figure 10: sensitivity to temporal window length and residual-gate scaling" width="900">
  <img src="assets/sensitivity_kernel_tmm.png" alt="Figure 11: sensitivity to Top-K pooling and dynamic-kernel size" width="900">
</p>

Figure 12 visualizes the input, static features, inter-frame differences, and features after RADC and MAG. Figure 13 compares uniform difference accumulation with RATA. Green boxes indicate true targets.

<p align="center">
  <img src="assets/feature_visualization_tmm.png" alt="Figure 12: intermediate features from RADC and MAG" width="950">
  <img src="assets/rata_comparison_tmm.png" alt="Figure 13: uniform accumulation versus RATA" width="950">
</p>

## Training described in the paper

DeepDIG is trained in two stages. The first jointly optimizes the static branch and an XFeat-based descriptor head. The second freezes these components and trains the difference path, dynamic path, fusion module and detection head.

| Setting | Stage 1: static branch and descriptors | Stage 2: temporal modeling and detection |
|---|---|---|
| Duration | 150,000 iterations | 50 epochs |
| Batch size | 8 | 2 |
| Learning rate | 3×10⁻⁴ | 1×10⁻³, cosine annealed to 1×10⁻⁴ |
| Input size (width×height) | 640×512 | IRDST/TSIRMT: 256×256; LMIRSTD: 640×512 |
| Objective | Soft-IoU for static detection; descriptor similarity, reliability, fine matching and keypoint losses | Soft-IoU |

The paper uses Adam and Soft-IoU smoothing of **1×10⁻⁵**. Stage 1 includes homography and thin-plate spline augmentation. RATA uses **M=2** temporal frequency pairs; the dynamic-kernel and MAG gating functions use **γ=10** and positive offset **ε=0.1**. The reported training environment is **PyTorch 2.4.1 / CUDA 12.1** on an **NVIDIA RTX 4090D**. This repository provides the model and inference/evaluation code.

## Installation and checkpoints

Download the three complete DeepDIG checkpoints from the [Google Drive weights folder](https://drive.google.com/drive/folders/1duJ4M6fwDdcLkgSOd1911EqphCNk4BV1):

| Dataset | Filename | Download |
|---|---|---|
| IRDST | `DeepDIG_IRDST.pth` | [Google Drive](https://drive.google.com/file/d/1YDaETLzq8jLxdrmsiBS2phqqDlSqkPXK/view?usp=drive_link) |
| TSIRMT | `DeepDIG_TSIRMT.pth` | [Google Drive](https://drive.google.com/file/d/1X0BZeTG_xEqHeNqKewZ425XVXJNfAKp9/view?usp=drive_link) |
| LMIRSTD | `DeepDIG_LMIRSTD.pth` | [Google Drive](https://drive.google.com/file/d/1QBT4TOWqMI5HNjlxSLj3zfP7yT3AWAkg/view?usp=drive_link) |

Place them in the ignored `weights/` directory and pass the actual path through `--ckpt`. A complete checkpoint contains the descriptor branch, so no additional first-stage XFeat checkpoint or GIM/SIFT dependency is needed. The loader accepts a pure state dictionary or a checkpoint containing `state_dict`, removes an optional `module.` prefix, and loads **strictly**. Explicitly labelled pipeline checkpoints must be `full`. Use the full configuration with 20 frames and 3×3 dynamic kernels. Checkpoints must match the published network structure.

Reference validation environment: Linux, Python 3.10, PyTorch 2.6.0+cu126. Runtime versions in `requirements.txt` match the validation environment. Install the appropriate PyTorch build first:

```bash
conda create -n deepdig python=3.10
conda activate deepdig
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

For CPU, use `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu` instead. Optional `--profile` requires `thop`, which is not needed for inference or evaluation.
