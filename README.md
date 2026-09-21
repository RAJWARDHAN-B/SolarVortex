# SolarVortex

Automated defect inspection of photovoltaic (PV) modules from electroluminescence (EL)
images. IIT Mandi internship project.

The system is a staged pipeline: a module-level EL image is decomposed into individual
cells, each cell is triaged as OK or defective, and defective cells are segmented at the
pixel level.

```
Stage 1   module EL image  --YOLOv8-OBB-->  rectified cell crops
Stage 2   cell crop        --ConvNeXt-T -->  OK / B-grade triage
Stage 3   cell crop        --segmentation->  per-pixel defect class
```

Each stage is trained and evaluated on its own dataset so the numbers remain
independently reportable.

---

## Stages

### Stage 1 — Cell extraction ([`CellExtraction/`](CellExtraction))

Localises and rectifies individual cells from module-level EL images. Oriented bounding
boxes (OBB) are used rather than axis-aligned boxes because a tilted cell's enclosing
upright box wastes ~25 % of its area on background, which contaminates the Stage 2 crop
and causes NMS to suppress valid neighbouring detections.

- [`YOLO/YOLOv8-OBB-aug/`](CellExtraction/YOLO/YOLOv8-OBB-aug) — final approach, all five scales benchmarked
- [`YOLO/YOLOv8/`](CellExtraction/YOLO/YOLOv8), [`YOLO/YOLOv11-seg/`](CellExtraction/YOLO/YOLOv11-seg) — alternatives
- [`CellExtractOpenCV/`](CellExtraction/CellExtractOpenCV), [`FullModuleOpenCV/`](CellExtraction/FullModuleOpenCV) — classical baselines (Hough, morphology, projection profiles)

All OBB scales reach mAP@0.5 = 0.995 with recall 1.000; `yolov8n-obb` is the deployment
choice at 3.1 M parameters and 222 FPS.

### Stage 2 — Binary defect triage ([`BinaryDetection/`](BinaryDetection))

Classifies each extracted cell as OK or B-grade. Seven ImageNet-pretrained backbones
benchmarked under an identical Optuna TPE budget, using validation AUC as the objective.

- [`Optuna/`](BinaryDetection/Optuna) — the 7-architecture benchmark and its results
- [`EfficientNet/`](BinaryDetection/EfficientNet), [`ResNetfreeze/`](BinaryDetection/ResNetfreeze) — earlier single-model experiments

ConvNeXt-Tiny leads on accuracy (97.2 %) and MCC; DenseNet-121 on AUC/PR-AUC
(98.9 % / 99.3 %); MobileNet-V3-Large is the efficiency pick at 3.0 M parameters.

### Stage 3 — Defect segmentation ([`Segmentation/`](Segmentation))

Pixel-level multi-class defect segmentation on the public
[BenchmarkELimages](https://github.com/TheMakiran/BenchmarkELimages) dataset
(`dataset_20221008`, 29 classes). Standard baselines are benchmarked first, then custom
architectures.

**See [`Segmentation/README.md`](Segmentation/README.md) for dataset setup, the class
taxonomy, verified dataset facts, and the model roadmap.**

---

## Supporting folders

| Folder | Purpose |
|---|---|
| [`CAFUNet/`](CAFUNet) | Prototype class-aware fusion U-Net. Concept is sound; implementation has known bugs — see [`Segmentation/README.md`](Segmentation/README.md) §6. To be rewritten into `Segmentation/models/custom/`. |
| [`FFLUNet/`](FFLUNet) | nnU-Net v2 fork for BraTS/KiTS 3D medical segmentation. Reference implementation; its dynamic multi-view feature fusion is the inspiration for the 2D solar variants. |
| [`Raj_Paper/`](Raj_Paper) | IEEE iSPEC 2026 manuscript and figures. |
| [`frontend/`](frontend) | Demo interface. |

---

## Environments

Each stage manages its own virtual environment; there is no repository-wide env.
Stage 3 uses `uv` with Python 3.11 — see [`Segmentation/README.md`](Segmentation/README.md)
§3. The system Python (3.9.6) is too old for the FFLUNet source.

Development happens on macOS (MPS, smoke tests only). All reported runs are produced on
Ubuntu 24.04 with an NVIDIA RTX PRO 4500 Blackwell and an AMD Ryzen Threadripper 7960X.

---

## Datasets

| Stage | Dataset | Size |
|---|---|---|
| 1 | Module-level EL (ARTS, SDLE, BMRK), OBB-annotated | 4,392 images |
| 2 | Independent cell-level EL, binary OK/B-grade | 9,392 cells |
| 3 | BenchmarkELimages `dataset_20221008`, 29-class masks | 553 unique cells (2,212 augmented) |

Stage 3's dataset is public and downloaded on demand; it is gitignored. Stages 1 and 2
use datasets held outside this repository.
