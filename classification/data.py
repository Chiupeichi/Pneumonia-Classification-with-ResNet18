"""Data loading for folder-based chest X-ray classification datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClassificationData:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_to_idx: dict[str, int]
    class_counts: list[int]


def build_transform(image_size: int = 224, train: bool = False):
    operations = [transforms.Grayscale(num_output_channels=3)]
    if train:
        operations.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
            ]
        )
    else:
        operations.append(transforms.Resize((image_size, image_size)))
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(operations)


def make_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    workers: int = 0,
) -> ClassificationData:
    root = Path(data_dir).expanduser().resolve()
    split_dirs = {name: root / name for name in ("train", "val", "test")}
    missing = [str(path) for path in split_dirs.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing dataset directories: {', '.join(missing)}")

    datasets = {
        name: ImageFolder(path, transform=build_transform(image_size, name == "train"))
        for name, path in split_dirs.items()
    }
    mapping = datasets["train"].class_to_idx
    if len(mapping) != 2:
        raise ValueError(f"Expected exactly 2 classes, found {mapping}")
    for name in ("val", "test"):
        if datasets[name].class_to_idx != mapping:
            raise ValueError(f"Class mapping for {name} differs from train")

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )
        for name, dataset in datasets.items()
    }
    targets = torch.tensor(datasets["train"].targets)
    counts = torch.bincount(targets, minlength=len(mapping)).tolist()
    return ClassificationData(
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        class_to_idx=mapping,
        class_counts=counts,
    )
