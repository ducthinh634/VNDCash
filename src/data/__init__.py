"""VNDCash Data Package.

Exports dataset classes, preprocessing, augmentations, splitting, and validation tools.
"""

from .augmentation import get_train_transforms
from .dataset import VNDCashDataset, get_dataloaders
from .preprocessing import (
    denormalize_tensor,
    get_eval_transforms,
    preprocess_image,
)
from .split import stratified_group_split, verify_no_leakage
from .validation import (
    validate_dataset_pipeline,
    validate_image_files,
    validate_metadata_schema,
)

__all__ = [
    "VNDCashDataset",
    "get_dataloaders",
    "get_train_transforms",
    "get_eval_transforms",
    "preprocess_image",
    "denormalize_tensor",
    "stratified_group_split",
    "verify_no_leakage",
    "validate_metadata_schema",
    "validate_image_files",
    "validate_dataset_pipeline",
]
