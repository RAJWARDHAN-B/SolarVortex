# Segmentation: Server Run Guide

All commands below run from `SolarVortex/Segmentation`. First transfer the updated
source files to your server using your normal Git or file-sync workflow. Local edits
do not automatically update the server. Keep the dataset; do not download it again
if it is already present. The saved outputs in `Baseline_seg_runs.ipynb` are historical.

## 1. Activate the Server Environment

Use your existing working Python 3.11 environment, for example:

```bash
conda activate raj
cd /home/user/SolarVortex/Segmentation
python -m pip install -r requirements.txt
```

Alternatively create a dedicated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Keep an already working GPU PyTorch installation. For a NEW Blackwell installation,
use a Blackwell-capable PyTorch build (2.7 or newer with CUDA 12.8 wheels) and a
compatible NVIDIA driver. The previous CUDA 12.4 recommendation was not suitable:

```bash
python -m pip install 'torch>=2.7' torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For other GPUs, use the appropriate official PyTorch installation command.
Verify an actual CUDA operation, not just device detection:

```bash
nvidia-smi
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print((torch.ones(2, 2, device='cuda') @ torch.ones(2, 2, device='cuda')).cpu())"
```

## 2. Archive Old Runs Safely

```bash
python -m training.run cleanup
```

This moves legacy `results/U-Net`, `U-Net++`, `DeepLabV3+`, `SegFormer-B0`, and
`class_stats.npz` into `results/legacy/<UTC timestamp>/`, if present. It does NOT
delete weights, dataset files, notebook logs, or new runs. Nothing in an old run
needs to be destroyed to start the new protocol. Old weight-only files cannot be
resumed with the new runner. Archive them as evidence, not as new benchmark results.

## 3. Audit the Data and Run Regression Tests

Expected dataset: `data/BenchmarkELimages/dataset_20221008`, with the class CSV and
all six paired `el_images_{train,val,test}` / `el_masks_{train,val,test}` directories.
If absent, download once:

```bash
git clone --filter=blob:none --sparse --depth 1 https://github.com/TheMakiran/BenchmarkELimages.git data/BenchmarkELimages
git -C data/BenchmarkELimages sparse-checkout set dataset_20221008
```

```bash
python -m training.run audit --split module_grouped --split-seed 42
python -m unittest evaluation.test_metrics training.test_run training.test_models -v
```

Audit JSON and the exact filename manifest are saved under `results/runs/audits/`.
The compatibility command `python datasets/verify_dataset.py --root <dataset>`
uses this same audit. Training also audits the data and saves its own copy.

Seed 42 currently gives 1,863/74/72 samples, 549/74/72 original cells, and
363/45/45 source modules. Augmented copies are used only in training.
Warnings about missing rare classes are expected: training has no `belt_mark`;
grouped test has no corrosion pixels, even after EL-10 merging. Do not claim
measured corrosion recall on that test set, and do not search seeds for better
test scores. The official split (`--split official`) is available but has module
overlap and must be reported separately.

## 4. Smoke-Test Every Baseline

Run one command at a time. Each performs one training batch including backward,
then one validation batch, and saves a clearly labelled smoke report. These are
NOT benchmark runs and do not create model checkpoints.

```bash
python -m training.run smoke --model unet --batch-size 2 --device cuda
python -m training.run smoke --model unetpp --batch-size 2 --device cuda
python -m training.run smoke --model deeplabv3plus --batch-size 2 --device cuda
python -m training.run smoke --model segformer-b0 --batch-size 2 --device cuda
```

Do not continue past a failing smoke test. For CUDA out-of-memory, reduce batch size;
for worker errors, add `--workers 0`. Use `--device cpu` only for a deliberate CPU run.
Library-dependent models must pass these real server checks. Regression tests include
real-library forward/backward contracts when the model requirements are installed,
isolated stub-based tests, and a tiny model for the artifact pipeline.

## 5. Train the Four Baselines Sequentially

Start with an explicitly **all-from-scratch** comparison. SegFormer no longer silently
uses ADE20K pretraining while the CNNs start from random weights. All runs use native
29-class targets, AdamW, cross-entropy, and the same grouped split. A 50-epoch budget
is an initial experiment, not a guarantee of convergence. Inspect validation curves
before finalizing the training budget; never tune against test results.

```bash
python -m training.run train --model unet --init scratch --epochs 50 --batch-size 4 --seed 42 --run-name scratch-seed42
python -m training.run train --model unetpp --init scratch --epochs 50 --batch-size 4 --seed 42 --run-name scratch-seed42
python -m training.run train --model deeplabv3plus --init scratch --epochs 50 --batch-size 4 --seed 42 --run-name scratch-seed42
python -m training.run train --model segformer-b0 --init scratch --epochs 50 --batch-size 4 --seed 42 --run-name scratch-seed42
```

CNN learning rate defaults to `1e-4`; SegFormer defaults to `5e-5`. Override with
`--learning-rate` for a recorded validation-based experiment. No scheduler or extra
augmentation is silently added. Model and optimizer settings are in each `config.json`.
Use a new run name for a changed protocol. Existing named directories are never
overwritten. Without `--run-name`, the runner generates a unique timestamped name.
Keep batch sizes consistent where feasible; record any memory-driven differences.

Checkpoints are selected using **minimum validation cross-entropy**. Training does NOT run test
evaluation. Validation and training metrics are computed from accumulated pixel counts,
not an average of batch IoUs. Training metrics describe online training-mode predictions;
validation uses the fixed model in evaluation mode at the end of the epoch.

## 6. Evaluate the Frozen Best Checkpoints

After selecting settings on validation, evaluate each model once for the final comparison:

```bash
python -m training.run evaluate --run-dir results/runs/unet/scratch-seed42 --eval-split test --figures 72
python -m training.run evaluate --run-dir results/runs/unetpp/scratch-seed42 --eval-split test --figures 72
python -m training.run evaluate --run-dir results/runs/deeplabv3plus/scratch-seed42 --eval-split test --figures 72
python -m training.run evaluate --run-dir results/runs/segformer-b0/scratch-seed42 --eval-split test --figures 72
```

Every indexed prediction mask is saved. `--figures 72` additionally saves a qualitative
panel for each of the current 72 test images; default is 12 panels. Use
`--eval-split val` for development figures. Evaluation reloads `checkpoints/best.pt`,
uses the saved split manifest, and rejects changed dataset content. Repeated evaluations
write new timestamped directories, preserving earlier reports.

Metrics include per-class IoU, Dice, precision, recall, support pixels, TP/FP/FN,
macro scores, defect-only scores, and raw/normalized confusion matrices. Undefined
values are JSON `null`. Zero-union classes are excluded from macro IoU; absent-GT
classes with false positives score zero and remain included. The GT-present macro
is also saved. Pixel accuracy is secondary, not a defect-quality headline.

EL-10 is a **derived evaluation**: sum native softmax probabilities for each merged
group before argmax, ignore ground-truth `meas_artifact`, and report the coverage.
It is not a separately trained EL-10 model. Defect-only EL-10 summarizes groups 4-9.
Raw prediction PNGs use native EL-29 labels; a palette JSON accompanies evaluation.

## 7. Find All Saved Artifacts

```text
results/
    legacy/<archive timestamp>/           preserved previous artifacts
    runs/
        audits/<timestamp>/                split manifest and class coverage
        smoke/<model>/<timestamp>/          one-batch diagnostic results only
        <model>/<run-name>/
            config.json                      hyperparameters and preprocessing
            model_stats.json                 total/trainable parameter counts
            environment.json                 Python, library versions, device
            splits.json                      exact relative image/mask paths and checksum
            dataset_audit.json               coverage, overlap, dataset content hashes
            status.json                      completed/interrupted status
            source/                          Python source snapshot for this run
            history.json                     complete epoch train/validation metrics
            history.csv                      compact numeric training curves
            best_validation.json             selected epoch and detailed metrics
            checkpoints/
                best.pt                        best model, optimizer, RNG and history
                last.pt                        resumable latest completed epoch
            plots/
                training_curves.png             train/validation loss, IoU and Dice
                loss_components.png             CE and optional soft Dice components
                validation_el29_iou_history.png
                validation_el10_iou_history.png
            evaluation/test-<timestamp>/
                metrics.json                   complete dataset-level numeric results
                per_image.json                 image-level summary scores
                per_image.csv
                el29_per_class.csv             EL-10 counterpart also saved
                el29_per_class.png
                el29_confusion.png
                el29_confusion_counts.npy
                palette.json
                environment.json
                predictions/*.png              native label IDs, NOT normalized colors
                qualitative/*.png               input, truth, prediction, pixel errors
```

PNG graphs are refreshed after every completed epoch. Their underlying values are in
`history.json`/`history.csv`; per-class heatmaps use the per-class epoch arrays in JSON.
Evaluation plots have numeric sources in `metrics.json`, including row-normalized
confusion values. Figures alone are never the only copy of the numbers.
Generated run/archive directories are Git-ignored: back them up from the server
separately. Allow several GB across runs for optimizer checkpoints and source snapshots.

## 8. Resume an Interrupted Run

```bash
python -m training.run train --resume results/runs/unet/scratch-seed42/checkpoints/last.pt
```

This resumes at the next epoch using the saved settings, optimizer, history, and RNG.
To extend a completed validation-based experiment, also pass `--epochs 100`.
Only load checkpoints you trust. If the dataset moved unchanged, add `--data-root`.
Use the same hardware and software for reproducibility; identical seeds do not promise
bitwise equality across different devices or library versions. A failure before the
first completed epoch has no resume checkpoint; restart with a new run name.

## 9. Run CAFUNet After the Baselines

```bash
python -m training.run smoke --model cafunet --variant dynamic --init scratch --device cuda
python -m training.run train --model cafunet --variant plain --init scratch --epochs 50 --batch-size 4 --run-name plain-scratch-seed42
python -m training.run train --model cafunet --variant fixed --init scratch --epochs 50 --batch-size 4 --run-name fixed-scratch-seed42
python -m training.run train --model cafunet --variant dynamic --init scratch --epochs 50 --batch-size 4 --run-name dynamic-scratch-seed42
```

These compare the same CAFUNet encoder/decoder with plain projected skips, dual paths
with fixed 50/50 fusion, and dual paths with image-conditioned fusion. Evaluate using
the same command as step 6 with the corresponding CAFUNet run directory. The invalid
"dynamic-only, no dual paths" configuration is rejected. Output resolution and class
count are corrected, but this is still the original prototype: fusion remains at the
two deepest skip stages and the gate is not explicitly class-aware. Do not claim those
limitations were redesigned by these fixes.

A separate transfer-learning comparison can use `--init pretrained`. Original `unet`
has no pretrained encoder; use `unet-resnet34` as the matched encoder control. ResNet
models use ImageNet weights, whereas pretrained SegFormer uses the ADE20K-finetuned
checkpoint; document this extra supervision rather than presenting it as an identical
pretraining regime. Use new run names and do not mix scratch/pretrained tables.
Once this protocol is stable, add FFLUNet-2D behind the same runner and evaluator.

## SolarUNet: New Solar-Specific Candidate

The runnable candidate is `--model solar-unet`. It preserves full-resolution EL edge
detail, adds row/column context, and learns class/pixel-specific detail gates. It is
not yet trained or proven better than the baselines, and is not a self-configuring
nnU-Net implementation. Full architecture, five ablations, and the experiment matrix
are in [models/custom/README.md](models/custom/README.md).

After updating the server source and environment:

```bash
python -m training.run smoke --model solar-unet --variant dynamic --loss ce-dice --batch-size 1 --device cuda
python -m training.run train --model solar-unet --variant plain --init scratch --loss ce --epochs 50 --batch-size 4 --seed 42 --run-name plain-ce-seed42
python -m training.run train --model solar-unet --variant dynamic --init scratch --loss ce --epochs 50 --batch-size 4 --seed 42 --run-name dynamic-ce-seed42
```

These two CE runs compare the internal control and full architecture without changing
the baseline objective. Test `--variant detail`, `context`, and `fixed` to isolate
individual additions. Only `solar-unet` accepts `detail` and `context` variants.

Optional `--loss ce-dice` is supported for every model. It adds CE and batch soft
foreground Dice loss over GT-present classes; target 255 is masked in both. It is
not dataset-level Dice and not an exact nnU-Net recipe. Compare loss-matched controls,
save these runs under new names, and do not mix architecture and loss claims.
Default commands still use CE; old checkpoints without a loss setting restore as CE.

After validation-based model selection, evaluate the frozen run:

```bash
python -m training.run evaluate --run-dir results/runs/solar-unet/dynamic-ce-seed42 --eval-split test --figures 72
```

All existing plots, JSON/CSV metrics, prediction PNGs, source snapshots, and resume
support apply. Loss-component graphs and their numeric values are saved as well.
Missing corrosion/belt-mark coverage remains a data limitation; do not claim that a
new architecture resolves it.

## 10. Run From Jupyter Instead

Select the actual training environment kernel. In a notebook, set the real server path:

```python
from pathlib import Path
import subprocess
import sys

project = Path("/home/user/SolarVortex/Segmentation")
print(sys.executable)

def run(*arguments):
        subprocess.run([sys.executable, "-m", "training.run", *arguments], cwd=project, check=True)

run("audit")
run("smoke", "--model", "unetpp", "--device", "cuda")
```

Then use one cell per training command, for example:

```python
run("train", "--model", "unet", "--init", "scratch", "--epochs", "50",
        "--batch-size", "4", "--run-name", "scratch-seed42")
```

Use `run("evaluate", "--run-dir", "results/runs/unet/scratch-seed42", "--figures", "72")`
after the protocol is frozen. Old per-model Python scripts also delegate to this runner,
but the module commands above are the preferred workflow. Server tests and full GPU
training must still be run on your machine; they are not implied by local code checks.