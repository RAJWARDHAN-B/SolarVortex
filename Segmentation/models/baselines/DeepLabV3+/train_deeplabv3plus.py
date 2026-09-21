import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


DATA_ROOT = Path("/workspace/BenchmarkELimages/dataset_20221008")
OUT_DIR = Path("/workspace/Segmentation/results/DeepLabV3+")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 29
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ELCellSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.samples = sorted(p.name for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img = __import__("PIL").Image.open(self.image_dir / name).convert("L")
        mask = __import__("PIL").Image.open(self.mask_dir / name)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)

        image = torch.from_numpy(img).unsqueeze(0).float()
        target = torch.from_numpy(mask).long()
        return image, target


def main():
    train_ds = ELCellSegDataset(DATA_ROOT / "el_images_train", DATA_ROOT / "el_masks_train")
    val_ds = ELCellSegDataset(DATA_ROOT / "el_images_val", DATA_ROOT / "el_masks_val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_ds)
        val_loss = val_loss / len(val_ds)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUT_DIR / "best_deeplabv3plus.pth")


if __name__ == "__main__":
    main()
