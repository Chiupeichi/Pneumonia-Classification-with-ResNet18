"""Predict pneumonia probability for one chest X-ray."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from common import load_checkpoint, resolve_device
from classification.data import build_transform
from classification.model import build_resnet18

DEFAULT_CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="pneumonia_resnet_model.pth")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint, device)
    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
        class_to_idx = payload.get("class_to_idx", DEFAULT_CLASSES)
        image_size = int(payload.get("image_size", 224))
    else:
        state = payload
        class_to_idx = DEFAULT_CLASSES
        image_size = 224

    model = build_resnet18(len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(state)
    model.eval()
    image = Image.open(Path(args.image)).convert("L")
    tensor = build_transform(image_size, train=False)(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    result = {idx_to_class[index]: round(probability, 6) for index, probability in enumerate(probabilities)}
    result["prediction"] = idx_to_class[int(torch.tensor(probabilities).argmax())]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
