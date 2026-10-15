"""PyTorch Dataset and DataLoader implementation for VNDCash.

Responsibilities (Member A):
- Implement VNDCashDataset supporting train, val, and test splits.
- Build get_dataloaders() factory for seamless integration with Member B's training loop.
- Apply train augmentations strictly to train loader.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .augmentation import get_train_transforms
from .preprocessing import get_eval_transforms
from ..utils.config import load_config
from ..utils.paths import DATA_CONFIG_PATH, PROJECT_ROOT, SPLITS_CSV


class VNDCashDataset(Dataset):
    """PyTorch Dataset for Vietnamese Banknote Recognition."""

    def __init__(
        self,
        splits_df: Union[pd.DataFrame, str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        root_dir: Optional[Union[str, Path]] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            splits_df: DataFrame or path to splits.csv.
            split: 'train', 'val', or 'test'.
            transform: Transform / augmentation callable.
            root_dir: Project root directory for relative path resolution.
            return_metadata: If True, returns dict with image, label, and metadata info.
        """
        if isinstance(splits_df, (str, Path)):
            splits_df = pd.read_csv(splits_df)

        valid_splits = {"train", "val", "test"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

        self.df = splits_df[splits_df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}' in DataFrame!")

        self.split = split
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else PROJECT_ROOT
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Dict]:
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["rel_path"]

        try:
            with Image.open(img_path) as im:
                image = im.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at {img_path}: {e}")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = get_eval_transforms()(image)

        label_idx = int(row["label_idx"])

        if self.return_metadata:
            return {
                "image": image_tensor,
                "label": label_idx,
                "denomination": str(row["denomination"]),
                "note_id": str(row["note_id"]),
                "image_id": str(row["image_id"]),
                "material": str(row["material"]),
                "path": str(img_path)
            }

        return image_tensor, label_idx


def get_dataloaders(
    config_path: Union[str, Path] = DATA_CONFIG_PATH,
    splits_csv_path: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory function creating train, val, and test DataLoaders according to configs.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    config = load_config(config_path)

    img_size = tuple(config.get("preprocessing", {}).get("image_size", (224, 224)))
    mean = config.get("preprocessing", {}).get("mean", [0.485, 0.456, 0.406])
    std = config.get("preprocessing", {}).get("std", [0.229, 0.224, 0.225])
    aug_cfg = config.get("augmentation", {}).get("train", {})

    dl_cfg = config.get("dataloader", {})
    bs = batch_size or dl_cfg.get("batch_size", 32)
    nw = num_workers if num_workers is not None else dl_cfg.get("num_workers", 0)
    pin_mem = dl_cfg.get("pin_memory", True)

    splits_file = Path(splits_csv_path) if splits_csv_path else SPLITS_CSV
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}. Please run create_split.py first!")

    df_splits = pd.read_csv(splits_file)

    train_transform = get_train_transforms(image_size=img_size, aug_cfg=aug_cfg, mean=mean, std=std)
    eval_transform = get_eval_transforms(image_size=img_size, mean=mean, std=std)

    train_ds = VNDCashDataset(df_splits, split="train", transform=train_transform)
    val_ds = VNDCashDataset(df_splits, split="val", transform=eval_transform)
    test_ds = VNDCashDataset(df_splits, split="test", transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_mem,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
