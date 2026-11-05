import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_cnn import BaselineCNN
from src.models.factory import build_model, count_parameters, get_model_summary
from src.training.losses import get_loss_fn, compute_class_weights
from src.utils.paths import SPLITS_CSV


def test_baseline_cnn_output_shape():
    """Verify BaselineCNN produces exactly 9 class outputs for [B, 3, 224, 224] input."""
    model = BaselineCNN(num_classes=9, in_channels=3, dropout_rate=0.3)
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    out = model(x)

    assert out.shape == (batch_size, 9), f"Expected shape {(batch_size, 9)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Model output contains NaN values"


def test_baseline_cnn_parameter_count():
    """Verify BaselineCNN parameter count is reasonable and trainable."""
    model = BaselineCNN(num_classes=9)
    param_info = model.get_num_parameters()
    total = param_info["total"]
    trainable = param_info["trainable"]

    assert total > 0
    assert total == trainable
    # Baseline CNN should be lightweight (< 5M parameters)
    assert total < 5_000_000, f"BaselineCNN is unexpectedly large: {total} parameters"


def test_build_model_factory():
    """Verify build_model can instantiate baseline_cnn, resnet18, mobilenet_v2 with 9 outputs."""
    # Baseline CNN
    model_cnn = build_model("baseline_cnn", num_classes=9)
    x = torch.randn(2, 3, 224, 224)
    out_cnn = model_cnn(x)
    assert out_cnn.shape == (2, 9)

    # MobileNetV2
    model_mb = build_model("mobilenet_v2", num_classes=9, pretrained=False)
    out_mb = model_mb(x)
    assert out_mb.shape == (2, 9)

    # ResNet18
    model_rn = build_model("resnet18", num_classes=9, pretrained=False)
    out_rn = model_rn(x)
    assert out_rn.shape == (2, 9)

    # Invalid model name
    with pytest.raises(ValueError):
        build_model("unknown_architecture_xyz")


def test_get_model_summary():
    """Verify model summary inspection function."""
    model = BaselineCNN(num_classes=9)
    summary = get_model_summary(model, input_size=(1, 3, 224, 224))
    assert summary["num_classes"] == 9
    assert summary["output_shape"] == [1, 9]
    assert summary["total_params"] > 0


def test_loss_with_class_weights():
    """Verify loss function computes correctly with class weights."""
    if SPLITS_CSV.exists():
        weights = compute_class_weights(SPLITS_CSV, num_classes=9)
        assert len(weights) == 9
        assert (weights > 0).all()

        criterion = get_loss_fn("cross_entropy", class_weights=weights)
        logits = torch.randn(4, 9)
        labels = torch.tensor([0, 2, 5, 8])
        loss = criterion(logits, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)


def test_single_train_step():
    """Verify end-to-end single optimization step works cleanly."""
    model = BaselineCNN(num_classes=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([1, 3, 4, 7])

    initial_param = next(model.parameters()).clone()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    updated_param = next(model.parameters())
    assert not torch.equal(initial_param, updated_param), "Parameters did not update after train step!"
