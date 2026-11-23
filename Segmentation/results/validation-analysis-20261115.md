# Validation Analysis: 20261114 Comparison

## Protocol

The seven `comparison-20261114T182431Z` runs completed 50 epochs with scratch
initialization, cross-entropy loss, batch size 4, seed 42, and the same
module-grouped split. The best checkpoint is selected by minimum validation
cross-entropy, not defect IoU. The results below are **validation**, from one
training seed, not held-out test results. EL-10 scores are derived from EL-29
probabilities, not from separately trained EL-10 models.

| Model | Best epoch | EL-10 mIoU | EL-10 defect mIoU | EL-10 defect Dice |
|---|---:|---:|---:|---:|
| U-Net++ | 30 | 0.5652 | 0.3926 | 0.5487 |
| U-Net | 33 | 0.5651 | 0.3857 | 0.5418 |
| DeepLabV3+ | 22 | 0.5395 | 0.3566 | 0.5053 |
| U-Net-ResNet34 | 26 | 0.5341 | 0.3442 | 0.4921 |
| SolarUNet | 39 | 0.5086 | 0.3086 | 0.4392 |
| SegFormer-B0 | 37 | 0.5017 | 0.3033 | 0.4253 |
| CAFUNet | 25 | 0.4572 | 0.2313 | 0.3469 |

These are not confidence intervals; the small U-Net/U-Net++ gap needs repeated
training seeds before drawing a robust ranking. SolarUNet did not outperform
either baseline on EL-10 defect mIoU in this comparison.

## Defect-Level Findings

Validation EL-10 crack recall is 0.318 for U-Net, 0.253 for U-Net++, and
0.089 for SolarUNet. The validation set has 28,252 crack pixels. SolarUNet
also has lower inactive and gridline recall (0.169 and 0.182) than U-Net++
(0.455 and 0.364). These results motivate investigating missed rare and
thin structures, but do not establish which intervention will help.

Source reports: [U-Net++](runs/unetpp/comparison-20261114T182431Z/evaluation/val-20261115T191628087319Z/metrics.json),
[U-Net](runs/unet/comparison-20261114T182431Z/evaluation/val-20261115T191714104189Z/metrics.json),
and [SolarUNet](runs/solar-unet/comparison-20261114T182431Z/evaluation/val-20261115T191803586725Z/metrics.json).
Per-class CSVs and confusion matrices are in the same directories.

## Two Qualitative Examples

For `CSIR_00067_r3_c2.png`, the ground truth contains 6,644 EL-29 `rings`
pixels. None of the three models predicts any of those pixels as `rings`.
See the [U-Net++ panel](runs/unetpp/comparison-20261114T182431Z/evaluation/val-20261115T191628087319Z/qualitative/CSIR_00067_r3_c2.png).

For `CSIR_00094_r16_c5.png`, the ground truth contains 1,584 `crack` and
19,605 `dead_cell` pixels. None of the three models predicts any of those
pixels as their respective EL-29 classes. See the [U-Net++ panel](runs/unetpp/comparison-20261114T182431Z/evaluation/val-20261115T191628087319Z/qualitative/CSIR_00094_r16_c5.png).
The images show localized mistakes, but two cases are not representative
evidence for the whole validation set.

`per_image.csv` contains EL-29, not EL-10, per-image metrics. A zero
defect mIoU can reflect missed ground-truth defects or false-positive defect
predictions; inspect the masks before assigning a cause.

## Next Experiment

Compare the existing cross-entropy baseline with the existing CE+Dice option
using otherwise matched settings and new run names. Run it for U-Net++ and
SolarUNet so the loss effect is not confused with an architecture change.
Evaluate crack, inactive, gridline, and false-positive behavior on validation;
do not assume Dice improves recall. Keep test data untouched until the approach
and evaluation protocol are frozen. The grouped test split has no corrosion
ground-truth pixels, so corrosion recall cannot be estimated from it.
