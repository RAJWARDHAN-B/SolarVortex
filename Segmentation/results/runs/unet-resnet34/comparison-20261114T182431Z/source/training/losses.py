from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class SegmentationObjective(nn.Module):
    """Optional CE + batch soft Dice over GT-present foreground classes."""

    def __init__(self, name: str = "ce", ignore_index: int = 255):
        super().__init__()
        if name not in ("ce", "ce-dice"):
            raise ValueError(f"Unknown segmentation objective: {name}")
        self.name = name
        self.ignore_index = ignore_index

    def components(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = target != self.ignore_index
        zero = logits.sum() * 0
        if not bool(valid.any()):
            return {"total": zero, "cross_entropy": zero, "soft_dice": zero}
        cross_entropy = functional.cross_entropy(logits, target, ignore_index=self.ignore_index)
        dice_loss = zero
        if self.name == "ce-dice":
            labels = target.masked_fill(~valid, 0)
            truth = functional.one_hot(labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
            truth = truth * valid.unsqueeze(1)
            probabilities = logits.float().softmax(dim=1) * valid.unsqueeze(1)
            dimensions = (0, 2, 3)
            support = truth.sum(dim=dimensions)
            included = support > 0
            included[0] = False
            if bool(included.any()):
                intersection = (probabilities * truth).sum(dim=dimensions)
                denominator = probabilities.sum(dim=dimensions) + support
                dice_loss = 1 - ((2 * intersection[included] + 1e-5) / (denominator[included] + 1e-5)).mean()
        return {"total": cross_entropy + dice_loss, "cross_entropy": cross_entropy, "soft_dice": dice_loss}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(logits, target)["total"]