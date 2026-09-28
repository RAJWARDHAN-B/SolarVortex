# How to Run the Segmentation Baselines

This guide runs the four baseline segmentation models on Ubuntu from Jupyter:

- U-Net
- U-Net++
- DeepLabV3+
- SegFormer-B0

The current project phase is baseline training. The purpose is to establish reference
models before comparing custom architectures. The baseline scripts do not currently
calculate held-out test metrics or create publication figures; see [What Gets Saved](#what-gets-saved).

## 1. Get the Project on Ubuntu

Clone the repository on the Ubuntu machine, then enter the segmentation folder:

```bash
git clone https://github.com/RAJWARDHAN-B/SolarVortex.git
cd SolarVortex/Segmentation
```

If the repository is already on the machine, `cd` to its `Segmentation` directory instead.

## 2. Create the Python Environment

Use Python 3.11. First create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the Ubuntu NVIDIA setup documented by this project (RTX PRO 4500 Blackwell), install
the CUDA 12.4 PyTorch wheels, then the segmentation requirements:

```bash
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

If your GPU or driver requires a different CUDA wheel, select the matching PyTorch
installation command for that machine. For a CPU-only run, omit the CUDA index URL:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Install Jupyter and register this environment as a kernel:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name solarvortex-seg \
    --display-name "Python (SolarVortex Segmentation)"
```

For a CUDA machine, verify that PyTorch can see the GPU:

```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU will be used')"
```

The trainers automatically select CUDA when available and otherwise run on CPU. CPU
training at 512x512 can be substantially slower.

## 3. Download the Dataset

From the `Segmentation` directory, download the `dataset_20221008` release:

```bash
git clone --filter=blob:none --sparse --depth 1 \
    https://github.com/TheMakiran/BenchmarkELimages.git data/BenchmarkELimages
git -C data/BenchmarkELimages sparse-checkout set dataset_20221008
```

The expected dataset root is:

```text
Segmentation/data/BenchmarkELimages/dataset_20221008/
```

It must contain paired `el_images_{train,val,test}` and `el_masks_{train,val,test}`
directories. The scripts use the grayscale image and indexed 29-class mask data as supplied.

Verify the dataset before training:

```bash
python datasets/verify_dataset.py \
    --root data/BenchmarkELimages/dataset_20221008
```

This checks image/mask pairing, taxonomy, label range, augmentation composition, and
split statistics. It also writes `results/class_stats.npz`.

## 4. Start Jupyter

Still in `Segmentation`, start JupyterLab:

```bash
jupyter lab
```

Open a notebook, select the **Python (SolarVortex Segmentation)** kernel, and run this
cell. Replace the example path with the actual absolute path on the Ubuntu machine:

```python
%cd /path/to/SolarVortex/Segmentation
import sys
import torch

print("Python:", sys.executable)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
```

Check that the printed Python path ends in `Segmentation/.venv/bin/python`. Keep the
notebook working directory at `Segmentation` for the commands below.

## 5. Select the Data Split

Each trainer currently defaults to:

```python
SPLIT_SCHEME = "module_grouped"
SPLIT_SEED = 42
```

This creates deterministic train/validation/test groups with no source module shared
across partitions. Augmented copies remain in training; validation and test use original
images. The current dataset produces 363/45/45 source modules and 549/74/72 original
cells for train/validation/test with seed 42.

For comparison with the published dataset split, edit `SPLIT_SCHEME` in the trainer to
`"official"`. The official split has source-module overlap, so use the grouped split for
leakage-free generalization claims. Keep the split scheme and seed consistent when
comparing models.

The trainers use the grouped training and validation sets. They report how many test
samples were reserved, but do not yet evaluate the test set.

## 6. Run the Models

Run each model in its own notebook cell. Each command starts a separate training run and
prints training and validation loss after every epoch. Wait for one run to finish before
starting another, especially if GPU memory is limited.

**U-Net**: 20 epochs, batch size 8, learning rate `1e-4`.

```python
!{sys.executable} models/baselines/U-Net/train_unet.py
```

**U-Net++**: SMP U-Net++ with a ResNet-34 encoder initialized from scratch; 20 epochs,
batch size 8, learning rate `1e-4`.

```python
!{sys.executable} models/baselines/U-Net++/train_unetpp.py
```

**DeepLabV3+**: SMP DeepLabV3+ with a ResNet-34 encoder initialized from scratch;
20 epochs, batch size 8, learning rate `1e-4`.

```python
!{sys.executable} models/baselines/DeepLabV3+/train_deeplabv3plus.py
```

**SegFormer-B0**: initialized from the Hugging Face checkpoint
`nvidia/segformer-b0-finetuned-ade-512-512`; 20 epochs, batch size 4, learning rate
`5e-5`. The first run needs internet access to download the pretrained checkpoint.

```python
!{sys.executable} models/baselines/SegFormer-B0/train_segformer.py
```

All four trainers use 29 output classes, AdamW, and cross-entropy loss with label 255
ignored. U-Net++/DeepLabV3+ are distinct models; SegFormer input channels and output
resolution are adapted in its trainer. If a run runs out of GPU memory, lower that
trainer's `BATCH_SIZE` and rerun it.

## 7. Find the Checkpoints

Each trainer saves the model weights with the lowest validation loss seen during that
run. The expected files are:

| Model | Checkpoint |
|---|---|
| U-Net | `results/U-Net/best_unet.pth` |
| U-Net++ | `results/U-Net++/best_unetpp.pth` |
| DeepLabV3+ | `results/DeepLabV3+/best_deeplabv3plus.pth` |
| SegFormer-B0 | `results/SegFormer-B0/best_segformer_b0.pth` |

These are PyTorch `state_dict` weight files, not full resumable training checkpoints.
Starting the same trainer again writes to the same path and can replace the existing
file. Copy or rename a checkpoint before a repeat run if you need to retain both.

## What Gets Saved

The current scripts print epoch-level training and validation loss and save the best
validation-loss weights listed above. They do **not** currently save loss-history files,
test-set IoU/Dice metrics, per-class metric tables, prediction masks, pixel coordinates,
plots, or image collages. The reserved test split is not evaluated yet. Therefore, these
runs provide trained baseline weights and validation-loss traces, but not complete
paper-ready baseline metric tables or figures.

## Baseline Scope

Yes: the current runs are the baseline models to establish reference performance before
running custom models. The planned comparison set in this folder is U-Net, U-Net++,
DeepLabV3+, and SegFormer-B0; the custom-model area is separate. For a fair comparison,
use the same dataset version, class taxonomy, grouped split seed, and evaluation protocol
for each baseline and custom model. Add test-set metric evaluation before treating the
results as final research scores.