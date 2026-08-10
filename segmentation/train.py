"""Train a U-Net lung segmentation model."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from common import atomic_torch_save, load_checkpoint, resolve_device, seed_everything
from segmentation.data import LungSegmentationDataset, paired_samples
from segmentation.engine import run_epoch
from segmentation.metrics import BCEDiceLoss
from segmentation.model import UNet
from segmentation.splitting import split_filenames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", default="checkpoints/segmentation_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_samples, val_samples = split_filenames(
        paired_samples(args.images, args.masks), args.val_fraction, args.seed
    )
    train_dataset = LungSegmentationDataset(
        args.images, args.masks, train_samples, args.image_size, augment=True
    )
    val_dataset = LungSegmentationDataset(args.images, args.masks, val_samples, args.image_size)
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **options)
    val_loader = DataLoader(val_dataset, shuffle=False, **options)

    model = UNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    stale_epochs = 0
    print(f"device={device} train={len(train_dataset)} val={len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics.loss)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics.loss:.4f} "
            f"train_dice={train_metrics.dice:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_dice={val_metrics.dice:.4f} val_iou={val_metrics.iou:.4f}"
        )
        if val_metrics.loss < best_loss:
            best_loss = val_metrics.loss
            stale_epochs = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "image_size": args.image_size,
                    "threshold": 0.5,
                    "best_val_loss": best_loss,
                    "epoch": epoch,
                },
                args.output,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stopping epoch={epoch}")
                break

    checkpoint = load_checkpoint(args.output, device)
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = run_epoch(model, val_loader, criterion, device)
    print(f"best_val_dice={final_metrics.dice:.4f} best_val_iou={final_metrics.iou:.4f}")


if __name__ == "__main__":
    main()
