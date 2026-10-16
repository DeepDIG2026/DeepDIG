<h1 align="center">DeepDIG: Deep Background Alignment Helps See Infrared Small Target Better</h1>

<p align="center">Project page for the manuscript prepared for <strong>IEEE Transactions on Multimedia (TMM)</strong>.</p>

DeepDIG detects small infrared targets in video sequences with significant camera-induced background motion. The current manuscript describes three components:

- **Deep Background Alignment (DBA):** Reuses spatial features to extract local descriptors and align frames.
- **Reliability-Aware Dynamic Convolution (RADC):** Combines Reliability-Aware Temporal Aggregation (RATA) with Content-Adaptive Dynamic Convolution (CADC) to reduce alignment artifacts and enhance motion cues.
- **Motion-guided Adaptive Gating (MAG):** Fuses spatial and temporal features using motion guidance.

This repository contains the **complete DeepDIG model: DBA + RADC + MAG**, including regional and pixel-wise RATA gates. The existing `model/`, `datautils/`, `utils/`, `assets/` layout and `test_deepdig.py` entry point are retained. Complete checkpoints for IRDST, TSIRMT and LMIRSTD will be published separately. The manuscript results below are reported results; full-test reproduction with the final public checkpoints remains to be verified. [CHANGELOG.md](CHANGELOG.md) records this update and its validation scope.

## Method

<p align="center">
  <img src="assets/architecture_tmm.png" alt="DeepDIG architecture" width="900">
</p>

The aligned sequence is processed by static, difference, and dynamic paths before MAG fuses their features for detection.

| Component | Source | Implementation |
|---|---|---|
| Static path | [static_path.py](model/static_path.py), [UNet_CBAM.py](model/UNet_CBAM.py) | Dual-task U-Net with transposed-convolution upsampling; current-frame features have 64 channels. |
| DBA | [DBA_module.py](model/DBA_module.py), [frame_cache.py](model/frame_cache.py) | Descriptor sampling, mutual nearest-neighbor matching, RANSAC homography estimation and warping. It shares encoder parameters with the static path; the default descriptor and detection passes are computed separately. |
| Difference path | [difference_path.py](model/difference_path.py) | Signed current-minus-history differences, masking zero-valued history pixels; residual convolutions and CBAM encode 19 differences into 64 channels. |
| RATA / CADC | [TADC_module.py](model/TADC_module.py) | Regional Top-K descriptors and pixel-wise residual gates produce a motion prior; CADC generates a separate spatial kernel for each temporal position. |
| Dynamic encoder | [dynamic_path.py](model/dynamic_path.py) | Convolutions, CBAM and a shortcut encode the enhanced 20-frame sequence into 64 channels. The shortcut uses the enhanced sequence. |
| MAG | [MAG_module.py](model/MAG_module.py) | Gates use raw signed differences. Fusion order: enhanced static features, projected differential features, enhanced dynamic features. |
| Detection head | [detection_network.py](model/detection_network.py) | Encoder-decoder detector with deep supervision; inference uses the final prediction. |

The `TADCModule` class name and parameter paths are retained for checkpoint compatibility. `AdaptiveDiffAccumulator` implements RATA; the rest of `TADCModule` implements motion normalization and CADC.

Default tensor flow (`N=20`, feature channels `C=64`):

```text
Input / DBA-aligned sequence       [B, 20, H, W]
Raw signed differences            [B, 19, H, W]
RATA accumulated motion           [B,  1, H, W]
CADC kernels                      [B, 20, 9, H, W]
CADC enhanced sequence            [B, 20, H, W]
Static / difference / dynamic     [B, 64, H, W] each
MAG fused feature                 [B, 64, H, W]
Final probability map             [B,  1, H, W]
```

RATA first max-pools by 4, then divides the pooled map into **4×4 regions**. The strongest 1% of responses provide mean, standard deviation, maximum and peak-to-mean statistics. Two sine/cosine frequency pairs encode temporal position. Both residual gates use `1 + 0.5 * tanh(logit)`; descriptor stability constants are `1e-6`. Top-K affects only regional descriptors; full-resolution differences remain in the weighted accumulation.

DeepDIG normalizes accumulated motion over the **entire batch tensor**, so batch composition affects predictions. The reference test uses batch size 2, ordered windows, no shuffle and no online cache. Keep these settings for metric comparisons. Dynamic coefficients are `10 * sigmoid(logit + learned_bias) + 0.1`, with no sum-to-one constraint.

<table align="center">
  <tr>
    <th align="center">Reliability-aware aggregation</th>
    <th align="center">Content-adaptive convolution</th>
  </tr>
  <tr>
    <td align="center"><img src="assets/rata_tmm.png" alt="RATA module" width="430"></td>
    <td align="center"><img src="assets/cadc_tmm.png" alt="CADC module" width="430"></td>
  </tr>
</table>

<p align="center">
  <img src="assets/mag_tmm.png" alt="MAG module" width="850">
</p>

## LMIRSTD Dataset

LMIRSTD contains **60 infrared video sequences**: 45 for training and 15 for testing. Each sequence has 200 frames at **640 x 512** resolution, for 12,000 frames in total. It includes dim targets and pronounced background motion across sky, urban, forest, mountain, and lake scenes. Its mean signal-to-clutter ratio (SCR) is 2.25 in the current manuscript.

<p align="center">
  <img src="assets/dataset_tmm.png" alt="Example LMIRSTD scenes" width="1000">
</p>

<table align="center">
  <thead>
    <tr><th>Dataset</th><th>Mean SCR</th><th>Background motion</th></tr>
  </thead>
  <tbody>
    <tr><td>IRDST</td><td align="center">6.70</td><td>Large</td></tr>
    <tr><td>TSIRMT</td><td align="center">3.04</td><td>Mild</td></tr>
    <tr><td>LMIRSTD</td><td align="center">2.25</td><td>Large</td></tr>
  </tbody>
</table>

The LMIRSTD dataset is available from the [dataset folder](https://drive.google.com/drive/folders/1tv9GhDs2jT7N_RRtqT8z2lzg-IpnqTfL?usp=sharing).

## Manuscript Results

The full quantitative comparison is reproduced directly from Table II of the current TMM manuscript. `Fa` is reported in units of `10^-6`; all other metrics are percentages, and the best results are shown in **bold**.

<p align="center">
  <img src="assets/results_tmm.png" alt="Table II from the TMM manuscript: quantitative comparison on IRDST, TSIRMT, and LMIRSTD" width="100%">
</p>

<!--
<details>
<summary>View the numerical results as a table</summary>

<div align="center">
<table>
  <thead>
    <tr>
      <th rowspan="2">Type</th>
      <th rowspan="2">Method</th>
      <th colspan="5">IRDST</th>
      <th colspan="5">TSIRMT</th>
      <th colspan="5">LMIRSTD</th>
    </tr>
    <tr>
      <th>Pd ↑</th><th>Fa ↓</th><th>IoU ↑</th><th>nIoU ↑</th><th>F1 ↑</th>
      <th>Pd ↑</th><th>Fa ↓</th><th>IoU ↑</th><th>nIoU ↑</th><th>F1 ↑</th>
      <th>Pd ↑</th><th>Fa ↓</th><th>IoU ↑</th><th>nIoU ↑</th><th>F1 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="8">Single frame</td><td>U-Net</td><td>96.88</td><td>19.69</td><td>55.26</td><td>56.94</td><td>92.78</td><td>77.37</td><td>384.12</td><td>58.42</td><td>60.20</td><td>78.21</td><td>87.28</td><td>4.92</td><td>74.52</td><td>68.84</td><td>81.91</td></tr>
    <tr><td>ACM</td><td>94.60</td><td>13.90</td><td>51.92</td><td>51.62</td><td>91.55</td><td>53.31</td><td>325.34</td><td>39.86</td><td>41.78</td><td>63.09</td><td>74.02</td><td>3.42</td><td>58.05</td><td>52.60</td><td>66.71</td></tr>
    <tr><td>ALCNet</td><td>99.01</td><td>21.31</td><td>55.26</td><td>55.63</td><td>94.61</td><td>66.85</td><td>1051.19</td><td>40.60</td><td>44.10</td><td>65.75</td><td>79.79</td><td>1.86</td><td>61.84</td><td>56.37</td><td>72.00</td></tr>
    <tr><td>DNANet</td><td>98.68</td><td><strong>7.37</strong></td><td>57.81</td><td>59.45</td><td>95.74</td><td>58.59</td><td>99.40</td><td>48.44</td><td>51.81</td><td>71.31</td><td>83.72</td><td>7.45</td><td>72.04</td><td>67.13</td><td>79.16</td></tr>
    <tr><td>RDIAN</td><td>93.17</td><td>16.70</td><td>53.99</td><td>55.53</td><td>89.79</td><td>79.96</td><td>140.27</td><td>47.04</td><td>52.18</td><td>71.92</td><td>85.14</td><td>7.82</td><td>71.80</td><td>67.90</td><td>80.23</td></tr>
    <tr><td>UIUNet</td><td>94.24</td><td>27.60</td><td>55.46</td><td>54.98</td><td>91.22</td><td>66.88</td><td>130.35</td><td>53.03</td><td>56.93</td><td>78.12</td><td>84.07</td><td>3.16</td><td>72.95</td><td>68.95</td><td>82.04</td></tr>
    <tr><td>MSHNet</td><td>97.96</td><td>33.54</td><td>55.66</td><td>58.73</td><td>94.65</td><td>60.28</td><td>715.40</td><td>42.06</td><td>46.21</td><td>68.05</td><td>77.32</td><td>29.76</td><td>56.00</td><td>53.89</td><td>72.37</td></tr>
    <tr><td>L2SKNet</td><td>95.20</td><td>59.28</td><td>38.71</td><td>39.96</td><td>88.08</td><td>59.25</td><td>1136.53</td><td>32.67</td><td>34.10</td><td>56.24</td><td>71.37</td><td>13.19</td><td>46.99</td><td>41.36</td><td>63.06</td></tr>
    <tr><td rowspan="7">Multiple frame</td><td>STDMANet</td><td>95.92</td><td>11.45</td><td>54.10</td><td>54.55</td><td>93.22</td><td>83.21</td><td>153.46</td><td>59.61</td><td>58.92</td><td>85.68</td><td>87.59</td><td>2.35</td><td>73.03</td><td>69.32</td><td>85.72</td></tr>
    <tr><td>LMAFormer</td><td><strong>99.64</strong></td><td>14.95</td><td>59.17</td><td>57.51</td><td>91.56</td><td>86.10</td><td>185.78</td><td>65.89</td><td>65.63</td><td>89.78</td><td>64.51</td><td><strong>0.21</strong></td><td>33.87</td><td>26.58</td><td>50.76</td></tr>
    <tr><td>ResUNet+DTUM</td><td>97.24</td><td>37.73</td><td>55.71</td><td>58.59</td><td>93.96</td><td>76.70</td><td>822.62</td><td>49.50</td><td>52.32</td><td>72.39</td><td>82.63</td><td>6.31</td><td>66.84</td><td>59.93</td><td>73.81</td></tr>
    <tr><td>DNANet+DTUM</td><td>98.44</td><td>24.46</td><td>58.66</td><td>62.06</td><td>95.40</td><td>79.07</td><td>672.77</td><td>51.76</td><td>56.22</td><td>76.01</td><td>84.57</td><td>9.93</td><td>67.14</td><td>63.20</td><td>79.48</td></tr>
    <tr><td>DeepPro-Plus</td><td>97.48</td><td>42.74</td><td>45.13</td><td>45.80</td><td>92.45</td><td>89.71</td><td>804.90</td><td>43.93</td><td>44.48</td><td>66.26</td><td>89.16</td><td>78.34</td><td>49.84</td><td>52.43</td><td>69.26</td></tr>
    <tr><td>OSFormer</td><td>–</td><td>–</td><td>–</td><td>–</td><td>82.06</td><td>–</td><td>–</td><td>–</td><td>–</td><td>67.72</td><td>–</td><td>–</td><td>–</td><td>–</td><td>54.81</td></tr>
    <tr><td><strong>DeepDIG (ours)</strong></td><td>99.18</td><td>11.30</td><td><strong>65.94</strong></td><td><strong>66.55</strong></td><td><strong>98.37</strong></td><td><strong>94.17</strong></td><td><strong>35.55</strong></td><td><strong>73.06</strong></td><td><strong>73.95</strong></td><td><strong>95.11</strong></td><td><strong>91.03</strong></td><td>2.81</td><td><strong>75.26</strong></td><td><strong>74.80</strong></td><td><strong>87.10</strong></td></tr>
  </tbody>
</table>
</div>

</details>
-->

The manuscript reports **9.15 FPS**, **15.21M parameters**, and **33.14 GFLOPs** for the full DeepDIG pipeline at 256 x 256 input resolution on an NVIDIA RTX 4090D. FPS includes background alignment and inference.

These are manuscript-reported measurements. The default complete DeepDIG network instantiated by this release has **13,549,436 unique parameters**. The difference in parameter counts and the benchmark numbers must be reconciled with the final checkpoints and a documented measurement protocol.

<p align="center">
  <img src="assets/comparison_tmm.png" alt="Qualitative comparison from the TMM manuscript" width="1000">
</p>

## Installation and checkpoints

The three complete DeepDIG checkpoints will be uploaded separately:

| Dataset | Example filename | Download |
|---|---|---|
| IRDST | `DeepDIG_IRDST.pth` | Pending upload |
| TSIRMT | `DeepDIG_TSIRMT.pth` | Pending upload |
| LMIRSTD | `DeepDIG_LMIRSTD.pth` | Pending upload |

Place them in the ignored `weights/` directory and pass the actual path through `--ckpt`. A complete checkpoint contains the descriptor branch, so no additional first-stage XFeat checkpoint or GIM/SIFT dependency is needed. The loader accepts a pure state dictionary or a checkpoint containing `state_dict`, removes an optional `module.` prefix, and loads **strictly**. Explicitly labelled pipeline checkpoints must be `full`. Use the full configuration with 20 frames and 3×3 dynamic kernels. Checkpoints must match the published network structure.

Reference validation environment: Linux, Python 3.10, PyTorch 2.6.0+cu126. Runtime versions in `requirements.txt` match the validation environment. Install the appropriate PyTorch build first:

```bash
conda create -n deepdig python=3.10
conda activate deepdig
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

For CPU, use `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu` instead. Optional `--profile` requires `thop`, which is not needed for inference or evaluation.

This release was checked against the reference DeepDIG implementation on 2026-10-13 using CPU execution and existing complete checkpoints:

| Dataset | Real-window comparison input | Strict checkpoint loading | Maximum absolute output difference | CLI partial test |
|---|---|---|---|---|
| IRDST | `[2,20,256,256]` | All 1,023 state entries loaded | `0` for all five predictions | 2 frames; original-size masks saved |
| TSIRMT | `[2,20,256,256]` | All 1,023 state entries loaded | `0` for all five predictions | 2 frames; original-size masks saved |
| LMIRSTD | `[2,20,512,640]` | All 1,023 state entries loaded | `0` for all five predictions | 2 frames; original-size masks saved |

Valid-image preprocessing, window order and alignment statistics matched the reference implementation. A missing checkpoint key was rejected. These are compatibility and execution checks; full-test manuscript metric reproduction requires the final public checkpoints.

## Dataset layout and inference

The dataset root contains `IRDST/`, `TSIRMT/` or `LMIRSTD/`. Each dataset has `images/<sequence>/<frame>`, matching `masks/<sequence>/<frame>`, and `ImageSets/test.txt` listing one sequence ID per line. A split file at the dataset root is also supported; `--split_file alternate_test.txt` selects another split. Preserve the published test IDs and sequence order. Use the `LMIRSTD` folder name for the published dataset.

| Setting | IRDST | TSIRMT | LMIRSTD |
|---|---|---|---|
| Frames per window | 20 | 20 | 20 |
| Input size, H×W | 256×256 | 256×256 | 512×640 |
| Evaluation size, H×W | 150×200 | 150×200 | 512×640 |
| Reference batch size | 2 | 2 | 2 |
| Reference test frames | 834 | 6,000 | 3,000 |

Images are grayscale floats divided by 255. IRDST/TSIRMT use PIL LANCZOS resizing; predictions are restored by area interpolation before evaluation. LMIRSTD retains its original resolution. `resize` is the reference mode; `padding` is a separate protocol. Windows end at the current frame; the initial windows repeat the first frame until 20 frames are available. Filenames are sorted lexicographically, so frame numbers should be zero-padded consistently. Missing images or GT cause an error.

Run the existing entry point:

```bash
python test_deepdig.py --ckpt weights/DeepDIG_IRDST.pth \
  --root dataset --dataset IRDST --test_batch_size 2 --workers 0 --save_pred

python test_deepdig.py --ckpt weights/DeepDIG_TSIRMT.pth \
  --root dataset --dataset TSIRMT --test_batch_size 2 --workers 0 --save_pred

python test_deepdig.py --ckpt weights/DeepDIG_LMIRSTD.pth \
  --root dataset --dataset LMIRSTD --test_batch_size 2 --workers 0 --save_pred
```

Defaults: `--in_channels 20 --descriptor_threshold 0.25 --resize_mode resize --dthres 3 --seed 0`. Use `--device_num 0` for GPU or `--device_num ""` for CPU. Predictions retain the corresponding mask's sequence/frame path below `result/<dataset>/test_<timestamp>/predictions/`; `--output_dir` overrides the result directory.

Each evaluation writes `results.txt` and `results.json`. JSON includes unrounded metrics, frame count, arguments, checkpoint/split/source SHA-256 hashes, environment and timing scope. For a short execution check add `--max_samples 2 --warmup_batches 0`; its results are explicitly marked as a **partial test**.

Print the loaded network without reading a dataset:

```bash
python test_deepdig.py --ckpt weights/DeepDIG_LMIRSTD.pth --show_model
```

The network can also run without GT through its Python interface:

```python
import torch
from model.deep_dig import build_deep_dig

model = build_deep_dig(input_channels=20, deep_supervision=True).eval()
checkpoint = torch.load("weights/DeepDIG_LMIRSTD.pth", map_location="cpu", weights_only=False)
state = checkpoint.get("state_dict", checkpoint)
model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
# frames: preprocessed float tensor [B, 20, H, W] in [0, 1].
with torch.no_grad():
    predictions, alignment = model(frames, align_mode="descriptor")
    probability = predictions[-1]
```

## Evaluation definitions

The primary five metrics are **Pd, Fa, IoU, nIoU and F1**. The default policy is `legacy_frame_mean_v1`: it follows the historical `utils/metric.py` and frame-average test protocol, with global IoU also reported. Metrics use original-resolution predictions and threshold 0.5:

- **IoU:** accumulated intersection divided by accumulated union over all frames.
- **nIoU:** mean frame IoU from the historical `compute_miou` function. A frame with neither GT foreground nor predicted foreground scores **1**; an empty-GT frame with any predicted foreground scores **0**. Every test frame participates in the mean.
- **mIoU / mIoU_legacy:** aliases of nIoU, retained for comparison with historical training logs.
- **Precision / Recall / F1:** rounded component centroids are greedily matched one-to-one within Euclidean distance of 3 original-image pixels. Precision/recall are averaged over frames; F1 is their harmonic mean. A frame with neither targets nor predictions scores 1 for precision and recall; if exactly one of the two sets is empty, both score 0.
- **Pd:** arithmetic mean of per-frame detected-GT-component ratios. A target is detected when a predicted pixel falls within a 3-pixel square neighborhood of any GT pixel. This pixel-neighborhood rule does not impose one-to-one component matching. With no GT targets, the frame scores 1 if its prediction is also empty and 0 otherwise.
- **Fa:** arithmetic mean of per-frame false-pixel ratios. Each ratio counts predicted foreground pixels outside those GT neighborhoods, divided by that frame's full image area. With empty GT, every predicted foreground pixel is a false alarm. `FA_x1e6` is Fa multiplied by `10^6` for table display.
- **PD_frame_mean / FA_frame_mean:** aliases of the primary Pd/Fa values.

For continuity with earlier result files, JSON and text outputs also include **PD_target_weighted** (total detected GT components / total GT components), **FA_pixel_weighted** (total false pixels / total image pixels), and **nIoU_empty_zero** (frame-average IoU with empty unions scored as 0). These are diagnostic alternatives, not the primary five scores. The policy name is written to both output formats and printed by the CLI; result files created before this change used different primary Pd/Fa/nIoU definitions.

An empty-GT frame is a valid all-zero mask. A missing or unreadable GT file is an error. Frame-average scores depend on the number of empty frames, so compare methods using the same split and metric policy. The underlying matching functions, input windows and inference settings are unchanged by this aggregation update.

FPS measures the complete model call, including DBA feature extraction, matching, warping and detection, while excluding loading, metrics and output writing. Optional THOP profiling uses pre-aligned input, excludes DBA, and counts two FLOPs per MAC; it is not a full-pipeline GFLOPs measurement.
