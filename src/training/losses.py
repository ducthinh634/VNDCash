from typing import Optional, Union, List, Dict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance by down-weighting easy examples."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(
    splits_csv_path: Union[str, Path],
    num_classes: int = 9,
    method: str = "balanced"
) -> torch.Tensor:
    """Computes inverse class frequency weights from training split.

    Args:
        splits_csv_path: Path to data/metadata/splits.csv.
        num_classes: Total number of banknote denominations (default 9).
        method: 'balanced' or 'smoothed'.

    Returns:
        Tensor of shape [num_classes] containing per-class loss weights.
    """
    df = pd.read_csv(splits_csv_path)
    train_df = df[df["split"] == "train"]

    counts = train_df["label_idx"].value_counts().to_dict()
    total_samples = len(train_df)

    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        if method == "balanced":
            # Heuristic standard: N / (K * n_c)
            w = total_samples / (num_classes * cnt)
        elif method == "smoothed":
            w = np.sqrt(total_samples / cnt)
        else:
            w = 1.0 / cnt
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so mean equals 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def get_loss_fn(
    loss_name: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    gamma: float = 2.0
) -> nn.Module:
    """Factory function creating loss criteria."""
    name = loss_name.strip().lower()

    if name in ["cross_entropy", "ce"]:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    elif name in ["focal_loss", "focal"]:
        return FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            reduction="mean"
        )
    else:
        raise ValueError(f"Unsupported loss function: '{loss_name}'. Use 'cross_entropy' or 'focal_loss'.")
