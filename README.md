# DeepDIG: Deep Background Alignment Helps See Infrared Small Target Better

Project page for the manuscript prepared for **IEEE Transactions on Multimedia (TMM)**.

DeepDIG detects small infrared targets in video sequences with significant camera-induced background motion. The current manuscript describes three components:

- **Deep Background Alignment (DBA):** Reuses spatial features to extract local descriptors and align frames.
- **Reliability-Aware Dynamic Convolution (RADC):** Combines Reliability-Aware Temporal Aggregation (RATA) with Content-Adaptive Dynamic Convolution (CADC) to reduce alignment artifacts and enhance motion cues.
- **Motion-guided Adaptive Gating (MAG):** Fuses spatial and temporal features using motion guidance.

The repository currently contains an earlier inference implementation. Its temporal module is named TADC and does not implement the full RATA and CADC design described in the TMM manuscript. The manuscript results below are reported results, not verified outputs of the currently published code and checkpoints. Updated implementation and checkpoints are needed for exact reproduction.

## Method

![DeepDIG architecture](assets/architecture_tmm.png)

The aligned sequence is processed by static, difference, and dynamic paths before MAG fuses their features for detection.

| Reliability-aware aggregation | Content-adaptive convolution |
| --- | --- |
| ![RATA module](assets/rata_tmm.png) | ![CADC module](assets/cadc_tmm.png) |

![MAG module](assets/mag_tmm.png)

## LMIRSTD Dataset

LMIRSTD contains **60 infrared video sequences**: 45 for training and 15 for testing. Each sequence has 200 frames at **640 x 512** resolution, for 12,000 frames in total. It includes dim targets and pronounced background motion across sky, urban, forest, mountain, and lake scenes. Its mean signal-to-clutter ratio (SCR) is 2.25 in the current manuscript.

![Example LMIRSTD scenes](assets/dataset_tmm.png)

| Dataset | Mean SCR | Background motion |
| --- | ---: | --- |
| IRDST | 6.70 | Large |
| TSIRMT | 3.04 | Mild |
| LMIRSTD | 2.25 | Large |

The LMIRSTD dataset is available from the [dataset folder](https://drive.google.com/drive/folders/1tv9GhDs2jT7N_RRtqT8z2lzg-IpnqTfL?usp=sharing).

## Manuscript Results

The following values are from the current TMM manuscript's quantitative comparison table. `Fa` is reported in units of `10^-6`; the other metrics are percentages.

| Dataset | Pd | Fa | IoU | nIoU | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IRDST | 99.18 | 11.30 | 65.94 | 66.55 | 98.37 |
| TSIRMT | 94.17 | 35.55 | 73.06 | 73.95 | 95.11 |
| LMIRSTD | 91.03 | 2.81 | 75.26 | 74.80 | 87.10 |

The manuscript reports **9.15 FPS**, **15.21M parameters**, and **33.14 GFLOPs** for the full DeepDIG pipeline at 256 x 256 input resolution on an NVIDIA RTX 4090D. FPS includes background alignment and inference.

![Qualitative comparison from the TMM manuscript](assets/comparison_tmm.png)

## Inference With the Currently Published Code

The [checkpoint folder](https://drive.google.com/drive/folders/1YF7dkfp9zl7Ny9WdfzAv1QAsHZgiKhSw?usp=sharing) was published with the earlier implementation. Use checkpoints that match the code version in this repository. The manuscript numbers above should not be assumed reproducible with those checkpoints.

Tested environment: Ubuntu 22.04, Python 3.10, PyTorch 2.4.1, and CUDA 12.1. Install a compatible PyTorch build first, then install the remaining dependencies:

```bash
conda create -n deepdig python=3.10
conda activate deepdig
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Place checkpoints in `weights/`. The dataset root should contain `IRDST/`, `TSIRMT/`, or `LMIRSTD/`. For example:

```bash
python test_deepdig.py \
  --ckpt ./weights/YOUR_CHECKPOINT.pth \
  --root ./dataset \
  --dataset LMIRSTD \
  --save_pred
```

Replace `YOUR_CHECKPOINT.pth` with the actual checkpoint name. The script also supports `IRDST` and `TSIRMT` through `--dataset`. By default, predictions are saved under `result/<dataset>/`.
