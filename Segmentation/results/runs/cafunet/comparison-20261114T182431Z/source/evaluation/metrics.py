from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(pred, target, num_classes, ignore_index=255):
    pred = torch.as_tensor(pred).detach().reshape(-1).to(dtype=torch.int64)
    target = torch.as_tensor(target).detach().reshape(-1).to(pred.device, dtype=torch.int64)
    if pred.shape != target.shape:
        raise ValueError("Prediction and target must have the same number of pixels")
    valid = target != ignore_index
    pred, target = pred[valid], target[valid]
    if ((target < 0) | (target >= num_classes)).any():
        raise ValueError("Target label outside the class range")
    if ((pred < 0) | (pred >= num_classes)).any():
        raise ValueError("Prediction label outside the class range")
    counts = torch.bincount(target * num_classes + pred, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes).cpu().numpy()


def _ratio(numerator, denominator):
    return np.divide(
        numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0
    )


def _mean(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _nullable(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def summarize_confusion(matrix, class_names, defect_labels=()):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix shape does not match class names")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = matrix.diagonal().astype(float)
    union = support + predicted - true_positive
    iou = _ratio(true_positive, union)
    dice = _ratio(2 * true_positive, support + predicted)
    precision = _ratio(true_positive, predicted)
    recall = _ratio(true_positive, support)
    rows = []
    for label, name in enumerate(class_names):
        rows.append({
            "label": label,
            "name": name,
            "support_pixels": int(support[label]),
            "predicted_pixels": int(predicted[label]),
            "true_positive": int(true_positive[label]),
            "false_positive": int(predicted[label] - true_positive[label]),
            "false_negative": int(support[label] - true_positive[label]),
            "iou": _nullable(iou)[label],
            "dice": _nullable(dice)[label],
            "precision": _nullable(precision)[label],
            "recall": _nullable(recall)[label],
        })
    return {
        "macro_iou": _mean(iou),
        "macro_dice": _mean(dice),
        "macro_iou_gt_present": _mean(iou[support > 0]),
        "defect_macro_iou": _mean(iou[list(defect_labels)]),
        "defect_macro_dice": _mean(dice[list(defect_labels)]),
        "pixel_accuracy": float(true_positive.sum() / matrix.sum()) if matrix.sum() else None,
        "valid_pixels": int(matrix.sum()),
        "absent_gt_classes": [name for name, count in zip(class_names, support) if count == 0],
        "undefined_iou_classes": [name for name, count in zip(class_names, union) if count == 0],
        "per_class": rows,
        "confusion_matrix": matrix.tolist(),
        "row_normalized_confusion_matrix": np.divide(matrix, support[:, None], out=np.zeros(matrix.shape, dtype=float), where=support[:, None] > 0).tolist(),
        "policy": "Rows=target, columns=prediction. Ignore target 255. Zero-union IoU and zero-denominator Dice are null and excluded from macro means. Absent-GT classes with false positives score zero and remain in macro means.",
    }


def per_class_iou(pred, target, num_classes, ignore_index=255):
    matrix = confusion_matrix(pred, target, num_classes, ignore_index)
    true_positive = matrix.diagonal().astype(float)
    return _nullable(_ratio(true_positive, matrix.sum(0) + matrix.sum(1) - true_positive))


def dice_score(pred, target, epsilon=1e-8, num_classes=29, ignore_index=255):
    matrix = confusion_matrix(pred, target, num_classes, ignore_index)
    return _mean(_ratio(2.0 * matrix.diagonal(), matrix.sum(0) + matrix.sum(1)))


def compute_iou_from_logits(logits, target, num_classes, ignore_index=255):
    return per_class_iou(logits.argmax(dim=1), target, num_classes, ignore_index)
