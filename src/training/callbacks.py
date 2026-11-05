from typing import Dict, Any, Optional
from pathlib import Path
import json
import torch
import torch.nn as nn
import pandas as pd


class EarlyStopping:
    """Early stopping callback to halt training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: str = "min",
        monitor: str = "val_loss"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False

    def step(self, current_val: float) -> bool:
        """Evaluates metric and returns True if training should stop."""
        if self.best_score is None:
            self.best_score = current_val
            return False

        if self.mode == "min":
            improvement = (self.best_score - current_val) > self.min_delta
        else:
            improvement = (current_val - self.best_score) > self.min_delta

        if improvement:
            self.best_score = current_val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpoint:
    """Saves model checkpoints during training (best model and latest checkpoint)."""

    def __init__(
        self,
        checkpoint_dir: Path,
        model_name: str = "baseline_cnn",
        mode: str = "min",
        monitor: str = "val_loss"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.mode = mode
        self.monitor = monitor
        self.best_score: Optional[float] = None
        self.best_checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pt"
        self.latest_checkpoint_path = self.checkpoint_dir / f"{model_name}_latest.pt"

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Checks if current epoch achieved a new best metric and saves checkpoints."""
        current_score = metrics.get(self.monitor)
        if current_score is None:
            return False

        is_best = False
        if self.best_score is None:
            is_best = True
            self.best_score = current_score
        else:
            if self.mode == "min":
                if current_score < self.best_score:
                    is_best = True
                    self.best_score = current_score
            else:
                if current_score > self.best_score:
                    is_best = True
                    self.best_score = current_score

        state = {
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config or {}
        }

        # Save latest checkpoint
        torch.save(state, self.latest_checkpoint_path)

        # Save best checkpoint
        if is_best:
            torch.save(state, self.best_checkpoint_path)
            # Write manifest JSON alongside
            manifest = {
                "model_name": self.model_name,
                "best_epoch": epoch,
                "best_score": current_score,
                "monitor": self.monitor,
                "metrics": metrics
            }
            manifest_path = self.checkpoint_dir / f"{self.model_name}_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

        return is_best


class MetricLogger:
    """Logs and tracks training & validation metrics across epochs; exports to CSV and JSON."""

    def __init__(self, log_dir: Path, experiment_name: str = "baseline_cnn"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

    def log(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ) -> None:
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(round(train_loss, 5))
        self.history["train_acc"].append(round(train_acc, 5))
        self.history["val_loss"].append(round(val_loss, 5))
        self.history["val_acc"].append(round(val_acc, 5))
        self.history["lr"].append(round(lr, 7))

        self.save()

    def save(self) -> None:
        # Save JSON
        json_path = self.log_dir / f"{self.experiment_name}_history.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # Save CSV
        csv_path = self.log_dir / f"{self.experiment_name}_history.csv"
        df = pd.DataFrame(self.history)
        df.to_csv(csv_path, index=False)
