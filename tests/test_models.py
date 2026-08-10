import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from classification.model import build_resnet18
from segmentation.metrics import BCEDiceLoss, overlap_scores
from segmentation.model import UNet


def test_classifier_output_shape():
    model = build_resnet18(num_classes=2, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 2)


def test_unet_output_shape_and_finite_loss():
    model = UNet().eval()
    image = torch.zeros(1, 1, 64, 64)
    target = torch.zeros_like(image)
    with torch.inference_mode():
        logits = model(image)
        loss = BCEDiceLoss()(logits, target)
    assert logits.shape == target.shape
    assert torch.isfinite(loss)


def test_overlap_scores_perfect_prediction():
    logits = torch.tensor([[[[20.0, -20.0], [-20.0, 20.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    dice, iou = overlap_scores(logits, target)
    assert dice == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)
