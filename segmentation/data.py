"""Paired image/mask loading with mask-safe augmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


SamplePair = tuple[str, str]


def paired_samples(image_dir: str | Path, mask_dir: str | Path) -> list[SamplePair]:
    """Match masks stored as either ``name.png`` or ``name_mask.png``."""
    images = Path(image_dir).expanduser().resolve()
    masks = Path(mask_dir).expanduser().resolve()
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Image or mask directory is missing: {images}, {masks}")
    pairs: list[SamplePair] = []
    for image_path in sorted(images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        candidates = (
            masks / image_path.name,
            masks / f"{image_path.stem}_mask{image_path.suffix}",
        )
        mask_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if mask_path is not None:
            pairs.append((image_path.name, mask_path.name))
    if not pairs:
        raise ValueError("No image/mask pairs with matching filenames were found")
    return pairs


class LungSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        samples: Sequence[SamplePair],
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.mask_dir = Path(mask_dir).expanduser().resolve()
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name, mask_name = self.samples[index]
        image = Image.open(self.image_dir / image_name).convert("L")
        mask = Image.open(self.mask_dir / mask_name).convert("L")
        size = [self.image_size, self.image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, InterpolationMode.NEAREST, fill=0)

        image_tensor = TF.to_tensor(image)
        mask_array = np.asarray(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_array >= 128).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor
