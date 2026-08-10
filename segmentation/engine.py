"""Training and evaluation loops for lung segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from segmentation.metrics import overlap_scores


@dataclass(frozen=True)
class SegmentationMetrics:
    loss: float
    dice: float
    iou: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> SegmentationMetrics:
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "count": 0}
    grad_context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with grad_context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            dice, iou = overlap_scores(logits.detach(), masks)
            count = images.size(0)
            totals["loss"] += loss.item() * count
            totals["dice"] += dice * count
            totals["iou"] += iou * count
            totals["count"] += count
    if totals["count"] == 0:
        raise ValueError("The dataloader is empty")
    return SegmentationMetrics(
        totals["loss"] / totals["count"],
        totals["dice"] / totals["count"],
        totals["iou"] / totals["count"],
    )
