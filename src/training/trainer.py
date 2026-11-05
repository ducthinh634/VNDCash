import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .callbacks import EarlyStopping, ModelCheckpoint, MetricLogger
from ..utils.logger import setup_logger


class Trainer:
    """End-to-end Trainer for Vietnamese Banknote Recognition models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.config = config or {}
        self.logger = logger or setup_logger("Trainer")

        # Discover callbacks
        self.early_stopping: Optional[EarlyStopping] = None
        self.checkpoint: Optional[ModelCheckpoint] = None
        self.metric_logger: Optional[MetricLogger] = None

        for cb in self.callbacks:
            if isinstance(cb, EarlyStopping):
                self.early_stopping = cb
            elif isinstance(cb, ModelCheckpoint):
                self.checkpoint = cb
            elif isinstance(cb, MetricLogger):
                self.metric_logger = cb

    def train_epoch(self) -> Tuple[float, float]:
        """Runs one training epoch over train_loader."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total_samples += batch_size

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self) -> Tuple[float, float]:
        """Evaluates model performance on val_loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total_samples += batch_size

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        return epoch_loss, epoch_acc

    def fit(self, epochs: int) -> Dict[str, list]:
        """Full training loop across specified number of epochs."""
        self.logger.info(f"Starting training on device: {self.device} for {epochs} epochs")
        start_time = time.time()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler if present
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start
            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr
            }

            for k, v in metrics.items():
                history[k].append(v)

            # Trigger MetricLogger
            if self.metric_logger is not None:
                self.metric_logger.log(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    lr=current_lr
                )

            # Trigger ModelCheckpoint
            is_best_str = ""
            if self.checkpoint is not None:
                is_best = self.checkpoint.step(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config=self.config
                )
                if is_best:
                    is_best_str = " [*Best*]"

            self.logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | "
                f"LR: {current_lr:.6f}{is_best_str}"
            )

            # Trigger EarlyStopping
            if self.early_stopping is not None:
                monitor_val = val_loss if self.early_stopping.monitor == "val_loss" else val_acc
                should_stop = self.early_stopping.step(monitor_val)
                if should_stop:
                    self.logger.info(f"Early stopping triggered after epoch {epoch}!")
                    break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes.")
        return history
