"""U-Net architecture that returns raw logits."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.down1 = ConvBlock(in_channels, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.bottom = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = ConvBlock(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(inputs)
        down2 = self.down2(self.pool(down1))
        down3 = self.down3(self.pool(down2))
        bottom = self.bottom(self.pool(down3))
        decode3 = self.dec3(torch.cat((self.up3(bottom), down3), dim=1))
        decode2 = self.dec2(torch.cat((self.up2(decode3), down2), dim=1))
        decode1 = self.dec1(torch.cat((self.up1(decode2), down1), dim=1))
        return self.output(decode1)


def translate_legacy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map layer names from the original notebook checkpoint to this module."""
    prefix_map = {
        "outc.": "output.",
        "down1.": "down1.",
        "down2.": "down2.",
        "down3.": "down3.",
        "bottom.": "bottom.",
        "up3.": "up3.",
        "dec3.": "dec3.",
        "up2.": "up2.",
        "dec2.": "dec2.",
        "up1.": "up1.",
        "dec1.": "dec1.",
    }
    translated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for old, new in prefix_map.items():
            if key.startswith(old):
                translated[new + key[len(old) :]] = value
                break
    return translated
