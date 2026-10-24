from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.el_classes import EL_CLASSES


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(run_dir, history):
    plots = Path(run_dir) / "plots"
    plots.mkdir(exist_ok=True)
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    for phase, color in (("train", "#187864"), ("val", "#bb4638")):
        axes[0].plot(epochs, [entry[phase]["loss"] for entry in history], label=phase, color=color)
        for axis, key in zip(axes[1:], ("macro_iou", "macro_dice")):
            axis.plot(epochs, [entry[phase]["el29"][key] for entry in history], label=phase, color=color)
    for axis, title in zip(axes, ("Cross-entropy loss", "EL-29 macro IoU", "EL-29 macro Dice")):
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(plots / "training_curves.png", dpi=180)
    plt.close(figure)
    for taxonomy in ("el29", "el10"):
        names = [row["name"] for row in history[-1]["val"][taxonomy]["per_class"]]
        values = np.array([
            [row["iou"] if row["iou"] is not None else np.nan for row in entry["val"][taxonomy]["per_class"]]
            for entry in history
        ], dtype=float).T
        figure, axis = plt.subplots(figsize=(10, max(4, len(names) * 0.25)), layout="constrained")
        image = axis.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axis.set_yticks(range(len(names)), names)
        ticks = np.unique(np.linspace(0, len(epochs) - 1, min(8, len(epochs)), dtype=int))
        axis.set_xticks(ticks, [epochs[index] for index in ticks])
        axis.set(xlabel="Epoch", title=f"Validation {taxonomy.upper()} per-class IoU (blank = undefined)")
        figure.colorbar(image, ax=axis, label="IoU")
        figure.savefig(plots / f"validation_{taxonomy}_iou_history.png", dpi=180)
        plt.close(figure)


def save_evaluation(directory, report):
    directory = Path(directory)
    write_json(directory / "metrics.json", report)
    for taxonomy in ("el29", "el10"):
        metrics = report[taxonomy]
        write_csv(directory / f"{taxonomy}_per_class.csv", metrics["per_class"])
        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        np.save(directory / f"{taxonomy}_confusion_counts.npy", matrix)
        names = [row["name"] for row in metrics["per_class"]]
        figure, axis = plt.subplots(figsize=(10, max(5, len(names) * 0.25)), layout="constrained")
        positions = np.arange(len(names))
        for offset, key, color in ((-0.2, "iou", "#187864"), (0.2, "dice", "#bb4638")):
            values = [row[key] if row[key] is not None else np.nan for row in metrics["per_class"]]
            axis.barh(positions + offset, values, height=0.38, label=key, color=color)
        axis.set_yticks(positions, names)
        axis.set(xlim=(0, 1), title=f"{taxonomy.upper()} per-class scores (undefined values omitted)")
        axis.invert_yaxis()
        axis.legend()
        figure.savefig(directory / f"{taxonomy}_per_class.png", dpi=180)
        plt.close(figure)
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, totals, out=np.zeros(matrix.shape, dtype=float), where=totals > 0)
        figure, axis = plt.subplots(figsize=(11, 10), layout="constrained")
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(names)), names, rotation=90)
        axis.set_yticks(range(len(names)), names)
        axis.set(xlabel="Predicted", ylabel="Ground truth", title=f"{taxonomy.upper()} row-normalized confusion matrix")
        figure.colorbar(image, ax=axis)
        figure.savefig(directory / f"{taxonomy}_confusion.png", dpi=180)
        plt.close(figure)


def save_prediction(directory, name, image, target, prediction, save_figure):
    directory = Path(directory)
    masks = directory / "predictions"
    masks.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(prediction, dtype=np.uint8)
    Image.fromarray(prediction).save(masks / name)
    if not save_figure:
        return
    palette = np.zeros((256, 3), dtype=np.uint8)
    for entry in EL_CLASSES:
        palette[entry.label] = entry.rgb
    palette[255] = (255, 0, 255)
    figures = directory / "qualitative"
    figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(palette[target])
    axes[2].imshow(palette[prediction])
    axes[3].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(target == 255, prediction != target), cmap="Reds", vmin=0, vmax=1, alpha=0.65)
    for axis, title in zip(axes, ("EL image", "Ground truth", "Prediction", "Pixel errors")):
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(figures / name, dpi=160)
    plt.close(figure)