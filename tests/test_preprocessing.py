import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    get_eval_transforms,
    preprocess_image,
    denormalize_tensor
)
from src.data.augmentation import get_train_transforms


def test_eval_transforms_shape():
    img = Image.new("RGB", (192, 144), color=(120, 150, 180))
    t = preprocess_image(img, image_size=(224, 224))
    assert t.shape == (1, 3, 224, 224)
    assert isinstance(t, torch.Tensor)


def test_denormalize_tensor():
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    t = preprocess_image(img, image_size=(224, 224))
    arr = denormalize_tensor(t)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8


def test_train_transforms_shape():
    img = Image.new("RGB", (192, 144), color=(200, 100, 50))
    train_tf = get_train_transforms(image_size=(224, 224))
    t = train_tf(img)
    assert t.shape == (3, 224, 224)
