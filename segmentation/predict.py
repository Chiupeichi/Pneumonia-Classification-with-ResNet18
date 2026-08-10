"""Generate a lung mask and optional overlay from one chest X-ray."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from common import load_checkpoint, resolve_device
from segmentation.model import UNet, translate_legacy_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="lung_unet_model.pth")
    parser.add_argument("--output", default="outputs/lung_mask.png")
    parser.add_argument("--overlay", help="Optional colored overlay output path")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint, device)
    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
        image_size = int(payload.get("image_size", 256))
        threshold = float(args.threshold if args.threshold is not None else payload.get("threshold", 0.5))
    else:
        state = translate_legacy_state_dict(payload)
        image_size = 256
        threshold = float(args.threshold if args.threshold is not None else 0.5)

    model = UNet().to(device)
    model.load_state_dict(state)
    model.eval()
    source = Image.open(args.image).convert("L")
    original_size = source.size
    tensor = TF.to_tensor(
        TF.resize(source, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu()
    mask = Image.fromarray((probability.numpy() >= threshold).astype(np.uint8) * 255)
    mask = mask.resize(original_size, Image.Resampling.NEAREST)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
    print(f"mask={output.resolve()}")

    if args.overlay:
        base = source.convert("RGB")
        color = Image.new("RGB", original_size, (255, 50, 50))
        tinted = Image.blend(base, color, 0.35)
        overlay = Image.composite(tinted, base, mask)
        overlay_path = Path(args.overlay)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)
        print(f"overlay={overlay_path.resolve()}")


if __name__ == "__main__":
    main()
