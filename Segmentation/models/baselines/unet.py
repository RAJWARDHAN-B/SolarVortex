import torch
from torch import nn


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

    def forward(self, inputs):
        return self.net(inputs)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, inputs):
        return self.net(inputs)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, inputs, skip):
        inputs = self.up(inputs)
        height = skip.size(2) - inputs.size(2)
        width = skip.size(3) - inputs.size(3)
        inputs = nn.functional.pad(inputs, [width // 2, width - width // 2, height // 2, height - height // 2])
        return self.conv(torch.cat([skip, inputs], dim=1))


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

    def forward(self, inputs):
        first = self.inc(inputs)
        second = self.down1(first)
        third = self.down2(second)
        fourth = self.down3(third)
        fifth = self.down4(fourth)
        decoded = self.up1(fifth, fourth)
        decoded = self.up2(decoded, third)
        decoded = self.up3(decoded, second)
        decoded = self.up4(decoded, first)
        return self.outc(decoded)