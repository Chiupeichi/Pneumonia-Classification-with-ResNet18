"""Train a reproducible ResNet-18 pneumonia classifier."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from common import atomic_torch_save, load_checkpoint, resolve_device, seed_everything
from classification.data import IMAGENET_MEAN, IMAGENET_STD, make_dataloaders
from classification.engine import run_epoch
from classification.model import build_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing train/val/test")
    parser.add_argument("--output", default="checkpoints/classification_best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data = make_dataloaders(args.data_dir, args.batch_size, args.image_size, args.workers)
    model = build_resnet18(len(data.class_to_idx), not args.no_pretrained).to(device)

    counts = torch.tensor(data.class_counts, dtype=torch.float32, device=device)
    class_weights = counts.sum() / counts.clamp_min(1) / len(counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_loss = float("inf")
    stale_epochs = 0
    print(f"device={device} classes={data.class_to_idx} train_counts={data.class_counts}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, data.train, criterion, device, optimizer, scaler)
        val_metrics = run_epoch(model, data.val, criterion, device)
        scheduler.step(val_metrics.loss)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f}"
        )
        if val_metrics.loss < best_loss:
            best_loss = val_metrics.loss
            stale_epochs = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": data.class_to_idx,
                    "image_size": args.image_size,
                    "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
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
    test_metrics = run_epoch(model, data.test, criterion, device)
    print(f"test_loss={test_metrics.loss:.4f} test_acc={test_metrics.accuracy:.4f}")


if __name__ == "__main__":
    main()
