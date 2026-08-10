import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from classification.engine import run_epoch as run_classification_epoch
from segmentation.engine import run_epoch as run_segmentation_epoch
from segmentation.metrics import BCEDiceLoss


def test_classification_train_and_eval_epoch():
    images = torch.randn(4, 1, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_metrics = run_classification_epoch(
        model, loader, criterion, torch.device("cpu"), optimizer
    )
    eval_metrics = run_classification_epoch(model, loader, criterion, torch.device("cpu"))

    assert train_metrics.loss > 0
    assert 0 <= eval_metrics.accuracy <= 1


def test_segmentation_train_and_eval_epoch():
    images = torch.rand(2, 1, 8, 8)
    masks = (images > 0.5).float()
    loader = DataLoader(TensorDataset(images, masks), batch_size=1)
    model = nn.Conv2d(1, 1, kernel_size=1)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_metrics = run_segmentation_epoch(
        model, loader, criterion, torch.device("cpu"), optimizer
    )
    eval_metrics = run_segmentation_epoch(model, loader, criterion, torch.device("cpu"))

    assert train_metrics.loss > 0
    assert 0 <= eval_metrics.dice <= 1
    assert 0 <= eval_metrics.iou <= 1
