import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.data_loader import BenchmarkELDataset
from datasets.module_split import module_grouped_split


# ------------------------------
# User config
# ------------------------------
DATA_ROOT = PROJECT_ROOT / "data/BenchmarkELimages/dataset_20221008"
OUT_DIR = PROJECT_ROOT / "results/U-Net"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 29
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_SCHEME = "module_grouped"  # or "official" for comparison with the published split
SPLIT_SEED = 42

# ------------------------------
# Dataset
# ------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=29):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


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

    model = UNet(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, masks)
                total_val += loss.item() * images.size(0)
        val_loss = total_val / len(val_loader.dataset)

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUT_DIR / "best_unet.pth")

    print(f"Saved best checkpoint to: {OUT_DIR / 'best_unet.pth'}")


if __name__ == "__main__":
    main()
