from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

SplitSamples = dict[str, list[tuple[Path, Path]]]

_AUGMENTATION_PREFIX = re.compile(r"^(?:flip|mirror|rotate)_\d+_")
_MODULE_PREFIX = re.compile(r"^([A-Za-z]+_\d+)")


def _base_name(filename: str) -> str:
    return _AUGMENTATION_PREFIX.sub("", filename)


def _module_id(filename: str) -> str:
    base = _base_name(filename)
    match = _MODULE_PREFIX.match(base)
    return match.group(1) if match else Path(base).stem


def module_grouped_split(dataset_root: str | Path, seed: int = 42) -> SplitSamples:
    """Build deterministic 80/10/10 splits with no source module shared across splits.

    Augmented copies stay with their source module and are included only in training.
    Original images are used for validation and test.
    """
    root = Path(dataset_root)
    originals: dict[str, tuple[Path, Path]] = {}
    augmentations: dict[str, list[tuple[Path, Path]]] = defaultdict(list)

    for split in ("train", "val", "test"):
        image_dir = root / f"el_images_{split}"
        mask_dir = root / f"el_masks_{split}"
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path}")

            base = _base_name(image_path.name)
            pair = (image_path, mask_path)
            if base == image_path.name:
                if base in originals:
                    raise ValueError(f"Duplicate original cell image across splits: {base}")
                originals[base] = pair
            else:
                augmentations[base].append(pair)

    orphaned = sorted(set(augmentations) - set(originals))
    if orphaned:
        raise ValueError(f"Augmented images have no original counterpart: {orphaned[:5]}")

    modules: dict[str, list[str]] = defaultdict(list)
    for base in originals:
        modules[_module_id(base)].append(base)

    module_ids = sorted(modules)
    if len(module_ids) < 3:
        raise ValueError("Module-grouped splitting requires at least three source modules")

    random.Random(seed).shuffle(module_ids)
    test_count = max(1, round(len(module_ids) * 0.1))
    val_count = max(1, round(len(module_ids) * 0.1))
    if test_count + val_count >= len(module_ids):
        raise ValueError("Not enough source modules for train/validation/test splits")

    assigned = {
        "test": set(module_ids[:test_count]),
        "val": set(module_ids[test_count : test_count + val_count]),
        "train": set(module_ids[test_count + val_count :]),
    }

    samples: SplitSamples = {split: [] for split in ("train", "val", "test")}
    for base, original_pair in originals.items():
        module = _module_id(base)
        split = next(name for name, group in assigned.items() if module in group)
        samples[split].append(original_pair)
        if split == "train":
            samples[split].extend(augmentations.get(base, ()))

    for split in samples:
        samples[split].sort(key=lambda pair: pair[0].name)

    return samples