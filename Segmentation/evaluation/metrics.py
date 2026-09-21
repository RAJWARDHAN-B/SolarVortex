from __future__ import annotations

import numpy as np
import torch


def per_class_iou(pred, target, num_classes):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        inter = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        ious.append(float(inter / (union + 1e-8)))
    return ious


def dice_score(pred, target, epsilon=1e-8):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    intersection = np.logical_and(pred == 1, target == 1).sum()
    denominator = (pred == 1).sum() + (target == 1).sum()
    return float((2 * intersection + epsilon) / (denominator + epsilon))


def compute_iou_from_logits(logits, target, num_classes):
    pred = logits.argmax(dim=1)
    return per_class_iou(pred, target, num_classes)
