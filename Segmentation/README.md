# Stage 3 — Semantic Segmentation of EL Defects

Pixel-level defect segmentation on electroluminescence (EL) images of silicon solar
cells. This is the third stage of the SolarVortex pipeline:

For the current runnable workflow, see [the server run guide](how_to_run.md).
The shared runner now saves epoch graphs and JSON/CSV metrics, resumable checkpoints,
explicit held-out evaluation, prediction masks, and qualitative panels. Old notebook
outputs are historical; old artifacts can be archived with `python -m training.run cleanup`.

```
Stage 1  module EL image  --YOLOv8-OBB-->  rectified cell crops
Stage 2  cell crop        --ConvNeXt-T -->  OK / B-grade triage
Stage 3  cell crop        --segmentation-> per-pixel defect class   <-- this folder
```

Stage 3 is trained and evaluated on a **public, independent dataset** and does not
consume Stage 1 or Stage 2 outputs. That keeps each stage's numbers independently
reportable.

---

## 1. Dataset

[BenchmarkELimages](https://github.com/TheMakiran/BenchmarkELimages) by Pratt et al.,
version **`dataset_20221008`** (the newest, largest, and most corrected release).

> Note: `pratt2023benchmark` is already cited in the paper as the `BMRK` source for
> the Stage 1 module dataset. Stage 1 uses the *module-level* images; Stage 3 uses the
> *cell-level* images and their masks. Verify there is no cross-stage module overlap
> before publishing joint results.

### Download

The repository hosts several dataset versions. A sparse checkout pulls only the one
we use (~838 MB instead of several GB):

```bash
cd Segmentation/data
git clone --filter=blob:none --sparse --depth 1 \
    https://github.com/TheMakiran/BenchmarkELimages.git BenchmarkELimages
cd BenchmarkELimages
git sparse-checkout set dataset_20221008
```

`Segmentation/data/BenchmarkELimages/` is gitignored.

### Layout

```
dataset_20221008/
├── el_images_train/   2212 png   ← EL images
├── el_images_val/       70 png
├── el_images_test/      72 png
├── el_masks_train/    2212 png   ← indexed ground-truth masks
├── el_masks_val/        70 png
├── el_masks_test/       72 png
├── el_masks_rgb/       695 png   ← human-readable colour masks (not for training)
└── ListOfClassesAndColorCodes_20221008.csv
```

### Verified facts

All confirmed by `datasets/verify_dataset.py` against the actual files:

| Property | Value |
|---|---|
| Image mode / size | **RGBA**, 512×512, uniform |
| Mask mode / size | **L** (single channel), 512×512, uniform |
| Mask encoding | Pixel value **is** the class label, `0..28`. No palette or RGB lookup. |
| Image/mask pairing | Identical filenames within a split; pairing by name is safe |
| Classes | **29** (labels 0–28) — 13 intrinsic features, 16 extrinsic defects |

Two things that will bite you if you skip them:

1. Images are **RGBA**, not grayscale. Load with `.convert("L")`, otherwise you get a
   4-channel tensor and a silently wrong first conv layer.
2. Masks must be loaded **without** any normalisation or resizing interpolation.
   Use nearest-neighbour only; bilinear resizing of a label map invents nonexistent
   classes.

### Augmentation is already baked into `train`

Training filenames carry an augmentation prefix:

```
ARTS_00001_r4_c2.png            ← original
flip_180_ARTS_00001_r4_c2.png   ← augmented
mirror_180_ARTS_00001_r4_c2.png
rotate_180_ARTS_00001_r4_c2.png
```

Composition: `original=553, flip=553, mirror=553, rotate=553` → 2212.
`val` and `test` contain originals only.

**Implication:** the effective training set is **553 unique cells**, not 2212. Adding
your own flip/rotate augmentation on top is mostly redundant; prefer photometric and
elastic augmentation, which the shipped set does not cover.

### Split leakage (important)

`verify_dataset.py` checks two levels, since filenames encode `SOURCE_MODULE_rROW_cCOL`:

| Level | Result |
|---|---|
| **Cell** | Clean — 553 / 70 / 72 unique cells, zero overlap |
| **Module** | **Leaks** — 25 modules shared train↔val, 34 train↔test, 19 val↔test |

Different cells cut from the *same physical module* appear in different splits. Those
cells share a manufacturer, an EL capture session, illumination, and often a defect
mechanism, so the official split modestly overestimates generalisation.

The baseline trainers support two split schemes:

- **`official`** — the shipped split. Use for comparability with Pratt et al.
- **`module_grouped`** — a deterministic 80/10/10 split grouped by module ID. Use for
  leakage-free headline claims. Augmented copies stay in training; validation and test
  use original images only. The seed defaults to `42`.

Use `--split module_grouped` (default) or `--split official` with the shared runner.
Keep `--split-seed 42` fixed across comparisons. Each run saves its actual filename
manifest and class coverage. Results from different split schemes are not directly comparable.

---

## 2. Class taxonomy

Defined in [`datasets/el_classes.py`](datasets/el_classes.py), verified at runtime
against the shipped CSV so a dataset version bump fails loudly instead of silently.

### EL-29 (native)

All 29 labels. Faithful to the source, but hard to report honestly:

- **6 classes have zero pixels in the test split** — `clamp`, `crack_rbn_edge`,
  `dead_cell`, `belt_mark`, `meas_artifact`, `sp_mono_halfcut`. A 29-class macro-mIoU
  averages over undefined terms.
- 7 classes occupy under 0.01 % of training pixels.
- The test split is only **72 images**, so several classes rest on a single image.

### EL-10 (reduced, recommended for headline numbers)

Classes are merged by shared defect mechanism and appearance. The following percentages
and non-empty-group observations describe the **official split only**. Grouped seed 42
has no corrosion pixels in test, and training has no native `belt_mark` pixels. Read the
saved per-run audit before interpreting macro scores; do not change splits to optimize test results.

| id | group | merged from | train px % | test px % |
|---:|---|---|---:|---:|
| 0 | `background` | background | 65.06 | 67.67 |
| 1 | `cell_substrate` | sp_multi, sp_mono, sp_dogbone, sp_mono_halfcut | 3.87 | 4.10 |
| 2 | `interconnect` | ribbons, busbars | 4.45 | 4.72 |
| 3 | `module_furniture` | border, text, padding, clamp, frame_edge, jbox | 24.18 | 21.01 |
| 4 | `crack` | crack, crack_rbn_edge | 0.21 | 0.38 |
| 5 | `inactive` | inactive, dead_cell | 0.22 | 0.37 |
| 6 | `gridline` | gridline | 0.56 | 0.64 |
| 7 | `corrosion` | corrosion_rbn, corrosion_cell | 0.25 | 0.26 |
| 8 | `surface_anomaly` | material, scuff, belt_mark, brightening, rings, star, splice | 0.77 | 0.72 |
| 9 | `edge_dark` | edge_dark | 0.44 | 0.12 |

`meas_artifact` (label 23) is mapped to `IGNORE_INDEX = 255`. It is a camera artefact,
not a property of the cell, so it contributes to neither the loss nor the metrics.

Rationale for each merge:

- **`cell_substrate`** — wafer variants (mono/multi/dogbone/halfcut) are a *material*
  distinction, not a defect. Separating them is a texture-classification task that
  dilutes the defect objective.
- **`interconnect`** — ribbons and busbars are both metallic conductors with near-identical
  EL appearance, and both are legitimate dark structures.
- **`module_furniture`** — non-cell image regions. Useful to segment, useless to subdivide.
- **`crack`** — `crack_rbn_edge` is a crack that happens to sit at a ribbon edge; only
  12 training images and 0 test images, so it cannot be scored on its own.
- **`inactive`** — a `dead_cell` is an inactive region covering the whole cell. Same
  physics, different extent.
- **`corrosion`** — same mechanism on ribbon vs. cell.
- **`surface_anomaly`** — a deliberate catch-all for rare, visually heterogeneous marks
  that are individually unscorable. This is the weakest group; report it separately and
  do not over-claim on it.
- **`gridline`** and **`edge_dark`** stay standalone: both are common, visually distinct,
  and diagnostically meaningful.

For NumPy masks without ignored pixels, remap with `np.asarray(EL10_LUT)[fine_mask]`.
The runner instead trains EL-29 and derives EL-10 predictions by summing native softmax
probabilities per merged group before argmax. This is not a separately trained EL-10 model.

### Reporting protocol

Always report all three, because each answers a different question:

1. **EL-29 per-class IoU** — full transparency, with zero-union classes marked `null`/`n/a`.
2. **Derived EL-10 mIoU** — with its missing-class coverage disclosed.
3. **Defect-only mIoU** — mean over EL-10 groups 4–9, excluding `background`,
   `cell_substrate`, `interconnect`, `module_furniture`. This is what actually matters for
   inspection, and it is the metric a plain mIoU hides.

Background plus module furniture is ~89 % of all pixels, so pixel accuracy is a
meaningless metric here. Do not report it as a headline.

Macro means exclude undefined zero-union IoUs; a class absent from ground truth but
predicted incorrectly scores zero and remains included. The GT-present-only macro is
also saved. Absent classes are not automatically perfect or automatically undefined.

---

## 3. Environment setup

The venv lives at `Segmentation/.venv` and is gitignored. `uv` is used because the
machine has no Homebrew and the system Python is 3.9.6, too old for the FFLUNet source
(which uses `int | tuple | list` syntax requiring ≥ 3.10).

### macOS (development, MPS)

```bash
cd Segmentation
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch torchvision
uv pip install --python .venv/bin/python -r requirements.txt
```

### Ubuntu 24.04 + RTX PRO 4500 Blackwell (training)

Use a Blackwell-capable PyTorch build (2.7+ with CUDA 12.8 wheels) and a compatible driver:

```bash
cd Segmentation
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python 'torch>=2.7' torchvision \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
```

Baseline scripts select `cuda` when available and otherwise use `cpu`, which covers the
Ubuntu training device. Their dataset and output paths are resolved relative to this
Segmentation directory. SegFormer repeats grayscale input across three channels to retain
its pretrained RGB encoder and upsamples logits to the mask size before computing loss.

### Verify the dataset

```bash
cd Segmentation
.venv/bin/python datasets/verify_dataset.py --root data/BenchmarkELimages/dataset_20221008
```

Run this after any dataset change. It checks pairing, CSV/taxonomy agreement, mask value
range, augmentation composition, split leakage, and per-class pixel statistics, and
writes timestamped split manifests and detailed JSON audits under `results/runs/audits/`.

---

## 4. Folder layout

```
Segmentation/
├── data/           dataset (gitignored)
├── datasets/       el_classes.py, verify_dataset.py, loaders, split generation
├── models/
│   ├── baselines/  U-Net, DeepLabV3+, SegFormer, ...
│   └── custom/     CA-FUNet, FFLUNet-2D, and new architectures
├── training/       train loop, losses, schedulers
├── evaluation/     metrics, per-class reports, qualitative figures
├── configs/        per-experiment YAML
├── notebooks/      exploration and figure generation
└── results/        metrics, checkpoint index, plots
```

---

## 5. Models and Research Candidates

### Baselines (`models/baselines/`)

The runnable baseline trainers are U-Net, U-Net++, DeepLabV3+, and SegFormer-B0. U-Net++
and DeepLabV3+ use their distinct `segmentation-models-pytorch` architectures with a
ResNet-34 encoder; SegFormer-B0 uses the Transformers implementation. The U-Net trainer
uses its own encoder/decoder implementation. All trainers default to the same
module-grouped split and can be switched to `official` in their configuration.

PSPNet, HRNet, and Mask2Former are planned, not currently implemented as trainers.

### Custom (`models/custom/`)

**SolarUNet v1 is now runnable as `--model solar-unet`.** It combines a standard
ResNet-34 encoder/decoder with full-resolution EL edge refinement, axial context,
and per-class/per-pixel gated detail residuals. Five ablations isolate those additions.
It is an experimental candidate, not a proven best model or an nnU-Net replacement.
Architecture, limitations, and controlled experiment commands are documented in
[models/custom/README.md](models/custom/README.md).

| Model | Idea |
|---|---|
| **SolarUNet v1 (implemented)** | Native-resolution edge detail, row/column context and spatial class-specific residual fusion. Optional CE+Dice is a separate experiment, not a silent default change. |
| **FFLUNet-2D** | Faithful 2D port of the BraTS FFLUNet. Answers "does the medical architecture transfer to EL?" and is the control for everything below. |
| **CA-FUNet v2** | Rewrite of the existing CA-FUNet (see §6). FFLUNet fuses with a *static* learned scalar pair; CA-FUNet conditions the fusion weights on the input. That delta is the contribution. |
| **Feature/Defect dual-head** | Two heads — one for the 13 intrinsic features, one for the 16 defects — with the structure head conditioning the defect head. Physically motivated: you must know where a busbar *should* be to judge whether a dark line is a crack. |
| **Periodicity-aware (strip pooling)** | Row/column axial pooling to model the periodic busbar/gridline lattice, separating *expected* dark lines (gridline, busbar) from *unexpected* ones (crack). |
| **Frequency-aware branch** | Laplacian/FFT high-pass branch beside the spatial branch. Cracks are high-frequency; `inactive` and `brightening` are low-frequency. |

---

## 6. Status of the existing CA-FUNet

The prototype at [`../CAFUNet/ca_funet.py`](../CAFUNet/ca_funet.py) is integrated with
the shared runner as `--model cafunet`. Output size and class count are corrected;
pretraining is explicit; the template trainer now delegates to the shared validated
data loader, training, checkpoints, and evaluator. Real-library CPU forward/backward
checks cover all supported models, with no pretrained downloads.

Valid ablations are `--variant plain`, `fixed`, and `dynamic`. The old dynamic-only
configuration without dual paths is rejected rather than presented as an experiment.
Architecture research remains: the fine/context branches still operate at the two
deepest skip stages, and the gate is image-conditioned, not explicitly class-aware.
These prototype fixes do not establish novelty or improve accuracy by themselves.

## 7. Current Experiment Layout

Use `python -m training.run` from this directory. The original four trainer paths are
compatibility entry points. `training/run.py` owns training/evaluation; `training/models.py`
owns construction and preprocessing; `training/data.py` owns saved splits/audits;
`evaluation/metrics.py` and `evaluation/reporting.py` own numeric reports and figures.
The importable original U-Net is in `models/baselines/unet.py`.

Run artifacts are isolated under `results/runs/<model>/<run-name>/`, including the
configuration, source snapshot, environment, split, content hashes, epoch JSON/CSV,
PNG graphs, best/last checkpoints, and timestamped evaluation reports. Generated
JSON/CSV evidence, plots, prediction images, qualitative panels, and selected best/last
checkpoints can be tracked with the code; the dataset and intermediate `epoch*.pt`
checkpoints are ignored. Model checkpoints use Git LFS, so install Git LFS and fetch
its objects to obtain the actual weights. Keep separate backups of large run directories.
Full commands, metric policies, resume instructions, and the artifact tree are in
[how_to_run.md](how_to_run.md).

Default loss remains CE. Opt into `--loss ce-dice` for a separately recorded
CE-plus-foreground-Dice experiment on any model. Checkpoint selection still uses
validation CE; total/CE/Dice loss components and their numeric sources are saved.
New runs snapshot custom model sources and record model parameter counts too.
