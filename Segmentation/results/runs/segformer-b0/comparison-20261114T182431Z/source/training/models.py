from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as functional

from datasets.el_classes import CLASS_NAMES, NUM_CLASSES

MODEL_NAMES = ("unet", "unet-resnet34", "unetpp", "deeplabv3plus", "segformer-b0", "cafunet", "solar-unet")
SEGFORMER_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerAdapter(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        labels = dict(enumerate(CLASS_NAMES))
        config_values = {"num_labels": NUM_CLASSES, "id2label": labels, "label2id": {name: label for label, name in labels.items()}}
        if pretrained:
            self.network = SegformerForSemanticSegmentation.from_pretrained(
                SEGFORMER_CHECKPOINT, ignore_mismatched_sizes=True, **config_values
            )
        else:
            self.network = SegformerForSemanticSegmentation(SegformerConfig(
                hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2],
                decoder_hidden_size=256, **config_values
            ))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        normalized = (images.repeat(1, 3, 1, 1) - self.mean) / self.std
        logits = self.network(pixel_values=normalized).logits
        return functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class GrayscaleNormalizedModel(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, images):
        return self.network((images - 0.449) / 0.226)


def build_model(name, initialization="scratch", variant="dynamic", load_pretrained=True):
    pretrained = initialization == "pretrained" and load_pretrained
    if name == "solar-unet":
        from models.custom.solar_unet import SolarUNet
        return SolarUNet(
            classes=NUM_CLASSES, variant=variant, pretrained=pretrained,
            normalize_encoder=initialization == "pretrained",
        )
    if variant not in ("plain", "fixed", "dynamic"):
        raise ValueError("detail/context variants are specific to solar-unet")
    if name == "unet":
        if initialization != "scratch":
            raise ValueError("The original U-Net has no pretrained encoder; use unet-resnet34 instead")
        from models.baselines.unet import UNet
        return UNet(out_channels=NUM_CLASSES)
    if name == "segformer-b0":
        return SegFormerAdapter(pretrained)
    if name == "cafunet":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "CAFUNet"))
        from ca_funet import CAFUNet
        network = CAFUNet(
            classes=NUM_CLASSES, pretrained=pretrained,
            use_dual_path=variant != "plain", use_dynamic_weights=variant == "dynamic",
        )
    else:
        import segmentation_models_pytorch as smp
        constructors = {"unet-resnet34": smp.Unet, "unetpp": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus}
        network = constructors[name](
            encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None,
            in_channels=1, classes=NUM_CLASSES,
        )
    return GrayscaleNormalizedModel(network) if initialization == "pretrained" else network