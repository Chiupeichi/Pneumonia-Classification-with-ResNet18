"""Numerically stable segmentation loss and metrics."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    return 1 - ((2 * intersection + eps) / (denominator + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        if not 0 <= bce_weight <= 1:
            raise ValueError("bce_weight must be between 0 and 1")
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice_loss(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


def overlap_scores(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> tuple[float, float]:
    predictions = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    dimensions = (1, 2, 3)
    intersection = (predictions & truth).sum(dim=dimensions).float()
    pred_area = predictions.sum(dim=dimensions).float()
    truth_area = truth.sum(dim=dimensions).float()
    union = pred_area + truth_area - intersection
    dice = ((2 * intersection + eps) / (pred_area + truth_area + eps)).mean().item()
    iou = ((intersection + eps) / (union + eps)).mean().item()
    return dice, iou
