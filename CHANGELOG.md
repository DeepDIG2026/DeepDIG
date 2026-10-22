# DeepDIG model update

This update preserves the existing repository layout and `test_deepdig.py` entry point. No training code, dataset or checkpoint is bundled or changed.

| File | Previous implementation | Updated implementation / reason |
|---|---|---|
| `model/TADC_module.py` | Unweighted difference accumulation; no RATA. | Regional Top-K descriptors, temporal encoding and pixel-wise residual gates; retains checkpoint parameter names and CADC computation. Corrects the description of bounded, non-normalized kernel coefficients. |
| `model/dynamic_path.py` | Shortcut encoded the original sequence. | Shortcut encodes the CADC-enhanced sequence; passes both RATA branch flags. |
| `model/static_path.py` | Bilinear decoder upsampling. | Transposed-convolution upsampling (`bilinear=False`), required by current checkpoint shapes. |
| `model/MAG_module.py` | Declared differential projection was unused; concatenated static/dynamic/difference. | Applies the trained projection and concatenates static/projected-difference/dynamic in the reference implementation's order; retains compatible ablation arguments. |
| `model/deep_dig.py` | Earlier model construction. | Connects the complete DeepDIG modules and configuration flags; retains the existing DBA and cache interfaces without adding alternative alignment backends. |
| `model/__init__.py` | Earlier exports and component description. | Exposes `AdaptiveDiffAccumulator` and documents RADC under the legacy checkpoint-compatible `TADCModule` name. |
| `datautils/dataloader.py` | Ignored `--split_file`; some missing inputs were skipped or replaced by blank images. | Uses the requested split and fails on missing images or LMIRSTD masks. Valid-image preprocessing and window order are preserved. |
| `test_deepdig.py` | Non-strict checkpoint loading; only historical frame-average metrics. | Strict full-model loading; reference batch size 2 and descriptor threshold; explicit IoU/nIoU and accumulated Pd/Fa alongside historical metrics; original-GT checks; structured configuration/hash records; network printing, partial-test limits and optional profiling. |
| `requirements.txt` | Unpinned runtime packages and mandatory profiling dependency. | Pins the reference runtime; PyTorch installation is documented separately and THOP is optional. |
| `README.md` | Earlier model/testing description. | Documents DeepDIG source/tensor flow, preprocessing, checkpoint requirements, three dataset commands, exact metric definitions, timing scope and pending public checkpoints. Keeps existing manuscript figures and distinguishes reported results from validation. |
| `CHANGELOG.md` | Absent. | Records every changed file and the reason for the change. |

Trailing whitespace in modified source files is normalized; unchanged model/metric files and manuscript assets are retained.

## Validation scope

Validation completed on 2026-10-13 with existing server checkpoints in an isolated directory and CPU execution, without modifying training sources or jobs.

- All 19 repository Python files parse, the existing CLI help runs, and `git diff --check` passes.
- Eight model files have the same computational AST as their checked reference counterparts; the six metric definitions/functions used by the CLI also match the reference implementation. The original DBA cache code is retained because its descriptor path already matches the reference implementation; alternative alignment backends are omitted.
- IRDST, TSIRMT and LMIRSTD full checkpoints each load all 1,023 state entries with `strict=True`; a missing entry fails. All three instantiate 13,549,436 unique parameters.
- For each dataset, real windows ending at frame indexes 19 and 39 are compared in one batch against the untouched reference implementation. Every one of the five prediction tensors is exactly equal (maximum absolute difference 0); alignment statistics, valid-image preprocessing and frame ordering agree. IRDST/TSIRMT use 256x256 and LMIRSTD uses 512x640 inputs.
- The CLI runs a separate 2-frame partial test on each dataset and writes original-size predictions, text/JSON metrics and checkpoint/split/source hashes. These partial metrics are execution checks, not manuscript accuracy results.
- Focused guards confirm custom split selection, first-frame padding, rejection of corrupt images/missing LMIRSTD masks, one-to-one precision/recall matching, the distinction between square Pd neighborhoods and Euclidean centroid matching, false-pixel normalization and historical empty-frame IoU behavior.

The three final public checkpoints and full-test reproduction of manuscript metrics remain pending. Existing manuscript figures are retained as reported results; CPU smoke-test FPS is not used as a paper benchmark.

## Public naming correction (2026-10-13)

Use DeepDIG throughout the public documentation and source annotations. Remove private experimental labels from `README.md`, this changelog, `requirements.txt`, the CLI help/result-policy text in `test_deepdig.py`, and the docstrings in `model/TADC_module.py` and `model/__init__.py`. Dataset instructions use the published LMIRSTD name.

This correction changes wording only. Network computation, configuration defaults, parameter names, checkpoint loading and metric formulas are unchanged; all Python sources still parse and the CLI help runs.

## Historical evaluation policy (2026-10-16)

The primary five test scores now use the agreed historical evaluation protocol. This changes score aggregation and reporting; model predictions, checkpoints, data loading and the matching functions in `utils/metric.py` are unchanged.

| File / output | Previous behavior | Updated behavior / reason |
|---|---|---|
| `test_deepdig.py`: nIoU | Mean frame IoU with empty unions scored as 0. | Uses historical `compute_miou` frame averages: both GT and prediction empty scores 1. Matches the selected historical metric protocol exactly. |
| `test_deepdig.py`: Pd / Fa | Primary values weighted by GT target counts / image pixel counts. | Primary values are arithmetic frame means from the historical metric functions, matching the historical test script. |
| `test_deepdig.py`: diagnostics | Historical values were separate fields. | `mIoU_legacy`, `PD_frame_mean` and `FA_frame_mean` remain aliases; previous primary definitions are preserved as `nIoU_empty_zero`, `PD_target_weighted` and `FA_pixel_weighted`. |
| `test_deepdig.py`: output metadata | No named aggregation policy; text did not explicitly record the nIoU empty-union rule. | Console, text and JSON identify `legacy_frame_mean_v1`; text records the empty-union rule and console/text include `FA_x1e6`. IoU remains accumulated intersection / union, and F1 remains the harmonic mean of frame-average precision and recall. |
| `README.md` | Described empty-union-zero nIoU and accumulated Pd/Fa as primary metrics. | Defines all five primary scores, the empty-GT cases, diagnostic alternatives and compatibility with older result files. |
| `CHANGELOG.md` | No record of this policy update. | Documents each changed output and its reason. |

Validation for this update:

- The inference and per-frame scoring portion of `test_deepdig.py` has an unchanged computational AST. Only this entry point and the two documentation files change; model, loader and metric-function files are unchanged.
- The actual summary and console/text/JSON output code is exercised using all 834 IRDST, 6,000 TSIRMT and 3,000 LMIRSTD saved per-frame measurements. Every primary score matches the independently recorded historical-protocol results exactly. This is an aggregation replay, not a new full inference run.
- Hand-computable cases cover empty GT with and without predictions, unequal target counts, and unequal image areas, distinguishing frame averages from count-weighted alternatives.
- The updated entry point also runs two real GPU inference frames per dataset. All six prediction images are byte-identical to the corresponding earlier full-test predictions, and the five output scores match the historical protocol on those frames. These partial runs are execution checks, not new dataset-level accuracy measurements.
- Python parsing, CLI help and `git diff --check` pass.

## Manuscript presentation update (2026-10-16)

Align the project presentation with the supplied TMM manuscript. Public results and figures follow the paper; the existing inference and evaluation protocol is retained.

| File / content | Previous presentation | Updated presentation / source |
|---|---|---|
| `README.md`: overview and method | Incomplete author/method overview and implementation-focused descriptions. | Adds the manuscript authors, submission status, DBA/RATA/CADC/MAG explanations, and a source map (pages 1-5). |
| `README.md`: LMIRSTD | Basic split, resolution and SCR information. | Adds target-size proportions, panorama explanation and the Table I NIBD statistics (pages 5-6). |
| `README.md`: results | Earlier comparison image, hidden obsolete numerical table and partial efficiency summary. | Uses the supplied Table II, adds the complete-model summary, Table III efficiency and Table IV alignment-method comparison (pages 7-9); removes the obsolete duplicate and implementation-validation result tables. |
| `README.md`: analysis and training | No dedicated sections for these experiments or the two-stage schedule. | Adds alignment, sensitivity, intermediate features and RATA accumulation analyses, plus the paper's two-stage training settings (pages 6-9). |
| `README.md`: inference commands | Relied on default frame count and device. | Explicitly shows 20-frame GPU inference; retains batch size 2, preprocessing, dataset layout, checkpoint loading and the agreed metric definitions. |
| `assets/results_tmm.png` | Earlier Table II. | Direct 300-dpi crop of Table II, page 7. |
| `assets/comparison_tmm.png` | Earlier six-row qualitative comparison. | Direct 300-dpi crop of the eight-row Figure 8, page 8. |
| `assets/alignment_example_tmm.png` | Absent. | Figure 1, page 1. |
| `assets/panorama_tmm.png` | Absent. | Figure 7, page 6. |
| `assets/alignment_analysis_tmm.png` | Absent. | Figure 9, page 9. |
| `assets/sensitivity_temporal_tmm.png` | Absent. | Figure 10, page 9. |
| `assets/sensitivity_kernel_tmm.png` | Absent. | Figure 11, page 9. |
| `assets/feature_visualization_tmm.png` | Absent. | Figure 12, page 9. |
| `assets/rata_comparison_tmm.png` | Absent. | Figure 13, page 9. |
| `model/DBA_module.py` | Descriptor-Based Alignment. | Deep Background Alignment, matching the paper. |
| `model/MAG_module.py` | Motion-Aware Gating; older dynamic-path names in ablation comments. | Motion-guided Adaptive Gating; RADC in those comments. |
| `model/__init__.py` | Earlier title and DBA/MAG expansions. | Paper title and module terminology; retains all exports. |
| `model/deep_dig.py` | Earlier title, TADC path label and scalar-branch description. | Paper title, RADC path label and regional temporal-gate description. |
| `model/dynamic_path.py` | TADC terminology in descriptions. | RADC/RATA/CADC terminology, with the existing class names and attribution retained. |
| `CHANGELOG.md` | No record of this presentation revision. | Records each updated file and its paper source. |

All new figure assets are direct crops of the supplied PDF. Existing architecture, RATA, CADC, MAG and dataset illustrations are retained. Source edits affect only comments and docstrings: computational ASTs, checkpoint keys, model defaults, `test_deepdig.py`, `utils/metric.py`, data loading and dependencies are unchanged. Validation checks Python parsing, executable-AST equality, README local links, paper table values, figure crop boundaries and `git diff --check`.

## Manuscript data synchronization (2026-10-22)

Use the supplied latest TMM manuscript as the source for reported results.

- Replace `assets/results_tmm.png` with a direct 300-dpi crop of the complete Table II on page 7, preserving all methods, datasets, ablations and the paper's emphasis.
- Point the README to the identical date-stamped image `assets/results_tmm_20261022.png` to avoid reuse of an older cached image at the previous URL.
- Update the complete-model IRDST results from Pd 98.92, Fa 14.67, IoU 63.26, nIoU 66.04 and F1 98.17 to Pd 99.40, Fa 8.35, IoU 65.90, nIoU 66.69 and F1 98.87. Update TSIRMT Fa from 35.55 to 35.58. The other complete-model TSIRMT results and the LMIRSTD results are unchanged. Fa is reported in units of 10^-6; the other metrics are percentages.
- Align the Figure 1 caption and alternative text with the paper's background-alignment comparison: the displayed responses are absolute frame differences, rather than intermediate feature responses.
- Check the Table I dataset comparison, LMIRSTD statistics, Table III efficiency values and existing illustrations against the latest manuscript; retain their current presentation. Preserve the two-column figure grids, full Table II image and concise installation/checkpoint/test instructions.

This is a manuscript presentation update, not a new inference or benchmark run. Model code, evaluation settings, dependencies and checkpoint download links are unchanged.
