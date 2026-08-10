"""Dependency-free helpers for reproducible dataset splitting."""

from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def split_filenames(
    filenames: Sequence[T], val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[T], list[T]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    shuffled = list(filenames)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    if val_count >= len(shuffled):
        raise ValueError("At least two paired samples are required")
    return shuffled[val_count:], shuffled[:val_count]
