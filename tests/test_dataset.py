import pytest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import VNDCashDataset, get_dataloaders
from src.utils.paths import SPLITS_CSV


def test_vndcash_dataset_loading():
    if not SPLITS_CSV.exists():
        pytest.skip("splits.csv not available")

    train_ds = VNDCashDataset(SPLITS_CSV, split="train")
    assert len(train_ds) > 0

    img, label = train_ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 3
    assert 0 <= label <= 8


def test_get_dataloaders_batch():
    if not SPLITS_CSV.exists():
        pytest.skip("splits.csv not available")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)

    images, labels = next(iter(train_loader))
    assert images.shape[0] == 8
    assert images.shape[1] == 3
    assert labels.shape[0] == 8
