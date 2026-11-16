from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.data_loader import BenchmarkELDataset
from datasets.el_classes import CLASS_NAMES, EL10_LUT, EL10_NAMES, NUM_CLASSES, verify_against_csv
from datasets.module_split import _base_name, _module_id, module_grouped_split


def prepare_manifest(root, scheme, seed):
    root = Path(root).resolve()
    for split in ("train", "val", "test"):
        for prefix in ("el_images_", "el_masks_"):
            if not (root / f"{prefix}{split}").is_dir():
                raise FileNotFoundError(root / f"{prefix}{split}")
        images = {path.name for path in (root / f"el_images_{split}").glob("*.png")}
        masks = {path.name for path in (root / f"el_masks_{split}").glob("*.png")}
        if images != masks:
            raise ValueError(f"Unpaired files in {split}: images-only={sorted(images - masks)[:5]}, masks-only={sorted(masks - images)[:5]}")
    csvs = sorted(root.glob("ListOfClassesAndColorCodes*.csv"))
    if len(csvs) != 1:
        raise ValueError("Expected exactly one dataset class CSV")
    verify_against_csv(csvs[0])
    if scheme == "module_grouped":
        pairs = module_grouped_split(root, seed)
    else:
        pairs = {
            split: [(image, root / f"el_masks_{split}" / image.name) for image in sorted((root / f"el_images_{split}").glob("*.png"))]
            for split in ("train", "val", "test")
        }
    samples = {
        split: [{"image": str(image.relative_to(root)), "mask": str(mask.relative_to(root))} for image, mask in items]
        for split, items in pairs.items()
    }
    digest = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    return {"scheme": scheme, "seed": seed, "sample_list_sha256": digest, "samples": samples}


def load_datasets(root, manifest):
    root = Path(root).resolve()
    digest = hashlib.sha256(json.dumps(manifest["samples"], sort_keys=True).encode()).hexdigest()
    if digest != manifest["sample_list_sha256"]:
        raise ValueError("Saved split manifest checksum mismatch")
    datasets = {}
    for split, rows in manifest["samples"].items():
        if not rows:
            raise ValueError(f"Empty {split} split")
        pairs = []
        for row in rows:
            image, mask = [(root / row[key]).resolve() for key in ("image", "mask")]
            for path in (image, mask):
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"Missing or unsafe dataset path: {path}")
            if image.name != mask.name:
                raise ValueError(f"Unpaired image and mask: {image}, {mask}")
            if manifest["scheme"] == "module_grouped" and split != "train" and image.name != _base_name(image.name):
                raise ValueError("Grouped evaluation splits must contain originals only")
            pairs.append((image, mask))
        datasets[split] = BenchmarkELDataset(root, root, samples=pairs)
    return datasets


def audit_datasets(datasets):
    report = {"splits": {}, "module_overlap": {}, "warnings": []}
    modules = {}
    for split, dataset in datasets.items():
        pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
        images = np.zeros(NUM_CLASSES, dtype=np.int64)
        original_images = np.zeros(NUM_CLASSES, dtype=np.int64)
        modules[split] = set()
        sizes = set()
        originals = 0
        content = hashlib.sha256()
        for image_path, mask_path in dataset.samples:
            with Image.open(mask_path) as mask_file, Image.open(image_path) as image_file:
                mask = np.asarray(mask_file)
                if mask.ndim != 2 or image_file.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"Image/mask shape mismatch: {image_path}")
            if mask.dtype.kind not in "ui" or np.any(((mask < 0) | (mask >= NUM_CLASSES)) & (mask != 255)):
                raise ValueError(f"Invalid mask labels: {mask_path}")
            if min(mask.shape) < 32 or any(size % 32 for size in mask.shape):
                raise ValueError(f"Expected image dimensions divisible by 32: {image_path}")
            sizes.add(mask.shape)
            valid = mask[mask != 255]
            counts = np.bincount(valid, minlength=NUM_CLASSES)
            pixels += counts
            images += counts > 0
            original = image_path.name == _base_name(image_path.name)
            if original:
                originals += 1
                original_images += counts > 0
            modules[split].add(_module_id(image_path.name))
            content.update(image_path.name.encode())
            content.update(hashlib.sha256(image_path.read_bytes()).digest())
            content.update(hashlib.sha256(mask_path.read_bytes()).digest())
        if len(sizes) != 1 or pixels.sum() == 0:
            raise ValueError(f"Expected uniform sizes and at least one valid pixel in {split}")
        merged = np.zeros(len(EL10_NAMES), dtype=np.int64)
        for fine, coarse in enumerate(EL10_LUT):
            if coarse != 255:
                merged[coarse] += pixels[fine]
        absent = [name for name, count in zip(CLASS_NAMES, pixels) if count == 0]
        absent_coarse = [name for name, count in zip(EL10_NAMES, merged) if count == 0]
        report["splits"][split] = {
            "samples": len(dataset), "original_cells": originals, "modules": len(modules[split]),
            "mask_sizes": [list(size) for size in sorted(sizes)], "content_sha256": content.hexdigest(),
            "el29_pixel_counts": pixels.tolist(), "el29_image_counts": images.tolist(),
            "el29_original_image_counts": original_images.tolist(),
            "el10_pixel_counts": merged.tolist(), "el29_absent": absent, "el10_absent": absent_coarse,
        }
        if absent:
            report["warnings"].append(f"{split}: EL-29 classes without ground-truth pixels: {', '.join(absent)}")
        if absent_coarse:
            report["warnings"].append(f"{split}: EL-10 groups without ground-truth pixels: {', '.join(absent_coarse)}")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(modules[first] & modules[second])
        report["module_overlap"][f"{first}_{second}"] = overlap
        if overlap:
            report["warnings"].append(f"{first}/{second}: {len(overlap)} overlapping source modules")
    return report