from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BenchmarkELDataset(Dataset):
    """Dataset for BenchmarkELimages cell-level segmentation masks."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform=None,
        samples: Sequence[tuple[str | Path, str | Path]] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        if samples is None:
            names = sorted(
                p.name for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            self.samples = [(self.image_dir / name, self.mask_dir / name) for name in names]
        else:
            self.samples = [(Path(image), Path(mask)) for image, mask in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
