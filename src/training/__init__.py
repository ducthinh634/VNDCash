"""VNDCash Training Engine Package.

Provides Trainer loop, loss functions with class weight balancing, early stopping,
checkpoint management and metric logging.
"""

from .trainer import Trainer
from .losses import get_loss_fn, compute_class_weights, FocalLoss
from .callbacks import EarlyStopping, ModelCheckpoint, MetricLogger

__all__ = [
    "Trainer",
    "get_loss_fn",
    "compute_class_weights",
    "FocalLoss",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricLogger"
]
