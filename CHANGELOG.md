# V3 model update

This update preserves the existing repository layout and `test_deepdig.py` entry point. No training code, dataset or checkpoint is bundled or changed.

| File | Previous implementation | Updated implementation / reason |
|---|---|---|
| `model/TADC_module.py` | Unweighted difference accumulation; no RATA. | V3 regional Top-K descriptors, temporal encoding and pixel-wise residual gates; retains V3 parameter names and CADC computation. Corrects the description of bounded, non-normalized kernel coefficients. |
| `model/dynamic_path.py` | Shortcut encoded the original sequence. | V3 shortcut encodes the CADC-enhanced sequence; passes both RATA branch flags. |
| `model/static_path.py` | Bilinear decoder upsampling. | V3 transposed-convolution upsampling (`bilinear=False`), required by current checkpoint shapes. |
| `model/MAG_module.py` | Declared differential projection was unused; concatenated static/dynamic/difference. | Applies the trained projection and concatenates static/projected-difference/dynamic in the V3 order; retains compatible ablation arguments. |
| `model/deep_dig.py` | Earlier model construction. | Connects the V3 modules and configuration flags; retains the existing DBA and cache interfaces without adding alternative alignment backends. |
| `model/__init__.py` | Earlier exports and component description. | Exposes `AdaptiveDiffAccumulator` and documents RADC under the legacy checkpoint-compatible `TADCModule` name. |
| `datautils/dataloader.py` | Ignored `--split_file`; some missing inputs were skipped or replaced by blank images. | Uses the requested split and fails on missing images or LMIRSTD masks. Valid-image preprocessing and window order are preserved. |
| `test_deepdig.py` | Non-strict checkpoint loading; only historical frame-average metrics. | Strict full-model loading; V3 batch size 2 and descriptor threshold; explicit IoU/nIoU and accumulated Pd/Fa alongside historical metrics; original-GT checks; structured configuration/hash records; network printing, partial-test limits and optional profiling. |
| `requirements.txt` | Unpinned runtime packages and mandatory profiling dependency. | Pins the reference runtime; PyTorch installation is documented separately and THOP is optional. |
| `README.md` | Earlier model/testing description. | Documents V3 source/tensor flow, preprocessing, checkpoint requirements, three dataset commands, exact metric definitions, timing scope and pending public checkpoints. Keeps existing manuscript figures and distinguishes reported results from validation. |
| `CHANGELOG.md` | Absent. | Records every changed file and the reason for the change. |

Trailing whitespace in modified source files is normalized; unchanged model/metric files and manuscript assets are retained.

## Validation scope

Validation completed on 2026-10-13 with existing server checkpoints in an isolated directory and CPU execution, without modifying training sources or jobs.

- All 19 repository Python files parse, the existing CLI help runs, and `git diff --check` passes.
- Eight model files have the same computational AST as their checked V3 counterparts; the six metric definitions/functions used by the CLI also match V3. The original DBA cache code is retained because its descriptor path already matches V3; alternative alignment backends are omitted.
- IRDST, TSIRMT and LMIRSTD full checkpoints each load all 1,023 state entries with `strict=True`; a missing entry fails. All three instantiate 13,549,436 unique parameters.
- For each dataset, real windows ending at frame indexes 19 and 39 are compared in one batch against the untouched V3 implementation. Every one of the five prediction tensors is exactly equal (maximum absolute difference 0); alignment statistics, valid-image preprocessing and frame ordering agree. IRDST/TSIRMT use 256x256 and LMIRSTD uses 512x640 inputs.
- The CLI runs a separate 2-frame partial test on each dataset and writes original-size predictions, text/JSON metrics and checkpoint/split/source hashes. These partial metrics are execution checks, not manuscript accuracy results.
- Focused guards confirm custom split selection, first-frame padding, rejection of corrupt images/missing LMIRSTD masks, one-to-one precision/recall matching, the distinction between square Pd neighborhoods and Euclidean centroid matching, false-pixel normalization and historical empty-frame IoU behavior.

The three final public checkpoints and full-test reproduction of manuscript metrics remain pending. Existing manuscript figures are retained as reported results; CPU smoke-test FPS is not used as a paper benchmark.
