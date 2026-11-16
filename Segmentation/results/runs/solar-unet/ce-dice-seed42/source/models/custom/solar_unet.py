from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as functional


SOLAR_VARIANTS = ("plain", "detail", "context", "fixed", "dynamic")


def normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            normalization(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            normalization(out_channels),
        )
        self.projection = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(features) + self.projection(features))


class ELEdgeFeatures(nn.Module):
    """Intensity, Sobel gradient magnitude, and signed Laplacian at native resolution."""

    def __init__(self):
        super().__init__()
        kernels = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ], dtype=torch.float32).unsqueeze(1)
        kernels[:2] /= 4
        kernels[2:] /= 4
        self.register_buffer("kernels", kernels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        edges = functional.conv2d(functional.pad(images, (1, 1, 1, 1), mode="replicate"), self.kernels)
        magnitude = (edges[:, :1].square() + edges[:, 1:2].square() + 1e-8).sqrt() - 1e-4
        return torch.cat((images, magnitude, edges[:, 2:3]), dim=1)


class AxisContext(nn.Module):
    """Local, dilated, row-pooled, and column-pooled context without quadratic attention."""

    def __init__(self, channels: int, hidden_channels: int = 128):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            normalization(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels)
        self.dilated = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2, groups=hidden_channels)
        self.row = nn.Conv2d(hidden_channels, hidden_channels, (3, 1), padding=(1, 0), groups=hidden_channels)
        self.column = nn.Conv2d(hidden_channels, hidden_channels, (1, 3), padding=(0, 1), groups=hidden_channels)
        self.expand = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, channels, 1, bias=False),
            normalization(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(features)
        row_context = self.row(reduced.mean(dim=3, keepdim=True)).expand_as(reduced)
        column_context = self.column(reduced.mean(dim=2, keepdim=True)).expand_as(reduced)
        combined = torch.cat((self.local(reduced), self.dilated(reduced), row_context, column_context), dim=1)
        return self.activation(features + self.expand(combined))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.decoder_projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.skip_projection = nn.Conv2d(skip_channels, out_channels, 1, bias=False)
        self.refine = ResidualBlock(out_channels * 2, out_channels)

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        decoded = functional.interpolate(self.decoder_projection(features), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((decoded, self.skip_projection(skip)), dim=1))


class ClassSpatialFusion(nn.Module):
    """A distinct residual-detail gate for every class and pixel."""

    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            normalization(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, classes, 1),
        )
        nn.init.normal_(self.gate[-1].weight, std=0.01)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def weights(self, semantic: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        return self.gate(torch.cat((semantic, detail), dim=1)).sigmoid()

    def forward(self, semantic: torch.Tensor, detail: torch.Tensor, semantic_logits: torch.Tensor, detail_logits: torch.Tensor) -> torch.Tensor:
        return semantic_logits + self.weights(semantic, detail) * detail_logits


class SolarUNet(nn.Module):
    """Experimental 2D EL-cell model; not a trained or self-configuring nnU-Net replacement."""

    def __init__(
        self,
        classes: int = 29,
        variant: str = "dynamic",
        pretrained: bool = False,
        normalize_encoder: bool = False,
    ):
        super().__init__()
        import timm

        if variant not in SOLAR_VARIANTS:
            raise ValueError(f"Unknown SolarUNet variant: {variant}")
        self.variant = variant
        self.normalize_encoder = normalize_encoder
        self.encoder = timm.create_model("resnet34", pretrained=pretrained, in_chans=1, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()[::-1]
        self.context = AxisContext(encoder_channels[0]) if variant in ("context", "fixed", "dynamic") else nn.Identity()
        decoder_channels = (256, 128, 64, 32)
        if len(encoder_channels) != len(decoder_channels) + 1:
            raise ValueError("SolarUNet expects five ResNet encoder feature levels")
        self.decoder = nn.ModuleList()
        in_channels = encoder_channels[0]
        for skip_channels, out_channels in zip(encoder_channels[1:], decoder_channels):
            self.decoder.append(DecoderBlock(in_channels, skip_channels, out_channels))
            in_channels = out_channels
        self.semantic_refine = ResidualBlock(in_channels, 32)
        self.semantic_head = nn.Conv2d(32, classes, 1)
        self.use_detail = variant in ("detail", "fixed", "dynamic")
        if self.use_detail:
            self.edge_features = ELEdgeFeatures()
            self.detail_stem = ResidualBlock(3, 16)
            self.detail_refine = ResidualBlock(48, 32)
            self.detail_head = nn.Conv2d(32, classes, 1)
            if variant == "dynamic":
                self.fusion = ClassSpatialFusion(32, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1 or min(images.shape[-2:]) < 32:
            raise ValueError("SolarUNet expects [batch, 1, height, width] grayscale images of size >=32")
        encoder_input = (images - 0.449) / 0.226 if self.normalize_encoder else images
        features = self.encoder(encoder_input)[::-1]
        decoded = self.context(features[0])
        for block, skip in zip(self.decoder, features[1:]):
            decoded = block(decoded, skip)
        semantic = self.semantic_refine(functional.interpolate(decoded, size=images.shape[-2:], mode="bilinear", align_corners=False))
        semantic_logits = self.semantic_head(semantic)
        if not self.use_detail:
            return semantic_logits
        detail = self.detail_stem(self.edge_features(images))
        detail = self.detail_refine(torch.cat((semantic, detail), dim=1))
        detail_logits = self.detail_head(detail)
        if self.variant == "dynamic":
            return self.fusion(semantic, detail, semantic_logits, detail_logits)
        return semantic_logits + 0.5 * detail_logits