"""Verify dataset_20221008 integrity and report the statistics that drive design choices.

Checks performed
  1. image/mask filename pairing per split
  2. shipped class CSV matches the hardcoded taxonomy
  3. mask values are all within [0, NUM_CLASSES-1]
  4. module-level leakage across train/val/test (filenames are SOURCE_MODULE_rR_cC)
  5. per-class pixel counts and image frequency, per split

Usage:
    python datasets/verify_dataset.py --root data/BenchmarkELimages/dataset_20221008
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from el_classes import CLASS_NAMES, EL_CLASSES, NUM_CLASSES, verify_against_csv

SPLITS = ("train", "val", "test")

# Augmented copies are prefixed, e.g. flip_180_ARTS_00001_r4_c2.png
_AUG_RE = re.compile(r"^(?P<aug>flip|mirror|rotate)_\d+_")
# ARTS_00001_r4_c2.png -> module "ARTS_00001"
_MODULE_RE = re.compile(r"^(?P<module>[A-Za-z]+_\d+)")


def strip_aug(filename: str) -> tuple[str, str]:
    """Return (augmentation_tag, original_filename)."""
    m = _AUG_RE.match(filename)
    return (m.group("aug"), filename[m.end():]) if m else ("original", filename)


def module_id(filename: str) -> str:
    _, base = strip_aug(filename)
    m = _MODULE_RE.match(base)
    return m.group("module") if m else Path(base).stem


def check_pairing(root: Path) -> dict[str, list[str]]:
    paired: dict[str, list[str]] = {}
    for split in SPLITS:
        imgs = {p.name for p in (root / f"el_images_{split}").glob("*.png")}
        masks = {p.name for p in (root / f"el_masks_{split}").glob("*.png")}
        only_img, only_mask = sorted(imgs - masks), sorted(masks - imgs)
        if only_img or only_mask:
            raise ValueError(
                f"[{split}] unpaired files - images-only={only_img[:5]} masks-only={only_mask[:5]}"
            )
        paired[split] = sorted(imgs)
        print(f"  {split:5s}: {len(imgs):5d} paired image/mask files")
    return paired


def check_augmentation(paired: dict[str, list[str]]) -> None:
    for split in SPLITS:
        tally: dict[str, int] = defaultdict(int)
        for f in paired[split]:
            tally[strip_aug(f)[0]] += 1
        print(f"  {split:5s}: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))


def check_leakage(paired: dict[str, list[str]]) -> None:
    mods = {s: {module_id(f) for f in files} for s, files in paired.items()}
    cells = {s: {strip_aug(f)[1] for f in files} for s, files in paired.items()}
    for s in SPLITS:
        print(f"  {s:5s}: {len(mods[s]):4d} unique modules, {len(cells[s]):4d} unique cells")
    clean = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        cell_ov = sorted(cells[a] & cells[b])
        if cell_ov:
            clean = False
            print(f"  CELL LEAKAGE {a}<->{b}: {len(cell_ov)} shared cells e.g. {cell_ov[:3]}")
        mod_ov = sorted(mods[a] & mods[b])
        if mod_ov:
            clean = False
            print(f"  MODULE LEAKAGE {a}<->{b}: {len(mod_ov)} shared modules e.g. {mod_ov[:5]}")
    if clean:
        print("  OK: no module appears in more than one split")


def class_stats(root: Path, paired: dict[str, list[str]]) -> dict[str, np.ndarray]:
    stats = {}
    for split in SPLITS:
        mask_dir = root / f"el_masks_{split}"
        pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
        images = np.zeros(NUM_CLASSES, dtype=np.int64)
        sizes: set[tuple[int, int]] = set()
        for name in tqdm(paired[split], desc=f"  scanning {split}", leave=False):
            arr = np.array(Image.open(mask_dir / name))
            if arr.ndim != 2:
                raise ValueError(f"{name}: expected single-channel mask, got shape {arr.shape}")
            sizes.add(arr.shape)
            vals, counts = np.unique(arr, return_counts=True)
            if vals.max() >= NUM_CLASSES:
                raise ValueError(f"{name}: label {vals.max()} exceeds NUM_CLASSES-1")
            pixels[vals] += counts
            images[vals] += 1
        stats[split] = np.stack([pixels, images])
        print(f"  {split:5s}: mask sizes = {sorted(sizes)}")
    return stats


def report(stats: dict[str, np.ndarray]) -> None:
    total_px = {s: v[0].sum() for s, v in stats.items()}
    n_img = {s: v[1].max() for s, v in stats.items()}
    hdr = f"{'lbl':>3} {'name':<17} {'kind':<7}"
    for s in SPLITS:
        hdr += f" | {s:>7} px%  imgs"
    print(hdr)
    print("-" * len(hdr))
    for c in EL_CLASSES:
        line = f"{c.label:>3} {c.name:<17} {c.kind:<7}"
        for s in SPLITS:
            px, im = stats[s][0][c.label], stats[s][1][c.label]
            line += f" | {100 * px / total_px[s]:10.4f} {im:5d}"
        print(line)
    print("-" * len(hdr))
    print("images per split: " + ", ".join(f"{s}={n_img[s]}" for s in SPLITS))

    absent = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if stats["test"][1][i] == 0]
    if absent:
        print(f"\nWARNING: {len(absent)} classes absent from TEST split: {absent}")
    rare = [
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
        if 0 < stats["train"][0][i] / total_px["train"] < 1e-4
    ]
    if rare:
        print(f"\nClasses under 0.01% of train pixels: {rare}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()

    print(f"Dataset root: {root}\n")
    print("[1] image/mask pairing")
    paired = check_pairing(root)

    print("\n[2] class CSV vs taxonomy")
    csvs = list(root.glob("ListOfClassesAndColorCodes*.csv"))
    verify_against_csv(csvs[0])
    print(f"  OK: {csvs[0].name} matches el_classes.py ({NUM_CLASSES} classes)")

    print("\n[3] augmentation composition")
    check_augmentation(paired)

    print("\n[4] split leakage")
    check_leakage(paired)

    print("\n[5] mask value range + per-class statistics")
    stats = class_stats(root, paired)

    print()
    report(stats)

    np.savez(
        root.parent.parent.parent / "results" / "class_stats.npz",
        **{s: v for s, v in stats.items()},
    )


if __name__ == "__main__":
    main()
