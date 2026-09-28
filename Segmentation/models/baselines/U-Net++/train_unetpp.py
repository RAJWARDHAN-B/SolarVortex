import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.data_loader import BenchmarkELDataset
from datasets.module_split import module_grouped_split


DATA_ROOT = PROJECT_ROOT / "data/BenchmarkELimages/dataset_20221008"
OUT_DIR = PROJECT_ROOT / "results/U-Net++"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 29
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_SCHEME = "module_grouped"  # or "official" for comparison with the published split
SPLIT_SEED = 42


def main():
    train_img, train_mask = DATA_ROOT / "el_images_train", DATA_ROOT / "el_masks_train"
    val_img, val_mask = DATA_ROOT / "el_images_val", DATA_ROOT / "el_masks_val"
    if SPLIT_SCHEME == "module_grouped":
        split_samples = module_grouped_split(DATA_ROOT, seed=SPLIT_SEED)
        train_ds = BenchmarkELDataset(train_img, train_mask, samples=split_samples["train"])
        val_ds = BenchmarkELDataset(val_img, val_mask, samples=split_samples["val"])
        print(f"module_grouped split seed={SPLIT_SEED}; test samples={len(split_samples['test'])}")
    elif SPLIT_SCHEME == "official":
        train_ds = BenchmarkELDataset(train_img, train_mask)
        val_ds = BenchmarkELDataset(val_img, val_mask)
    else:
        raise ValueError(f"Unsupported split scheme: {SPLIT_SCHEME}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=NUM_CLASSES,
    )
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
            torch.save(model.state_dict(), OUT_DIR / "best_unetpp.pth")


if __name__ == "__main__":
    main()
