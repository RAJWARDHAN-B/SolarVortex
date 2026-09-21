from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BenchmarkELDataset(Dataset):
    """Dataset for BenchmarkELimages cell-level segmentation masks."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.samples = sorted(
            p.name for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img_path = self.image_dir / name
        mask_path = self.mask_dir / name

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
