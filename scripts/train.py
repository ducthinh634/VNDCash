"""Training Script for Vietnamese Banknote Recognition (VNDCash).

Stack: Stack02 - Custom CNN & Training Engine
Entry point to train models with configuration, logging, checkpoints, and learning curves.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.utils.paths import CONFIGS_DIR, ARTIFACTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR
from src.data.dataset import get_dataloaders
from src.models.factory import build_model, count_parameters
from src.training.losses import get_loss_fn, compute_class_weights
from src.training.callbacks import EarlyStopping, ModelCheckpoint, MetricLogger
from src.training.trainer import Trainer


def plot_learning_curves(history: dict, output_path: Path, title: str = "Baseline CNN"):
    """Plots and saves loss and accuracy learning curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#2980b9")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#e74c3c")
    ax1.set_title(f"{title} - Loss Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy (%)", marker="o", color="#27ae60")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy (%)", marker="s", color="#f39c12")
    ax2.set_title(f"{title} - Accuracy Curve", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train VNDCash Banknote Recognition Model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIGS_DIR / "baseline.yaml"),
        help="Path to YAML training configuration"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override compute device ('cpu', 'cuda')")
    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    config = load_config(cfg_path)

    # Logging setup
    exp_name = config.get("experiment", {}).get("name", "baseline_cnn_experiment")
    log_dir = PROJECT_ROOT / config.get("artifacts", {}).get("logs_dir", "artifacts/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train", log_file=log_dir / f"{exp_name}.log")

    logger.info("=" * 70)
    logger.info(f"STARTING EXPERIMENT: {exp_name}")
    logger.info(f"Loaded configuration from: {cfg_path}")
    logger.info("=" * 70)

    # Seed for reproducibility
    seed = config.get("experiment", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Random seed initialized to: {seed}")

    # Device selection
    device_cfg = args.device or config.get("experiment", {}).get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info(f"Compute device selected: {device}")

    # Data loaders
    data_cfg = config.get("data", {})
    data_config_path = PROJECT_ROOT / data_cfg.get("config_path", "configs/data.yaml")
    splits_csv = PROJECT_ROOT / data_cfg.get("splits_csv", "data/metadata/splits.csv")
    batch_size = args.batch_size or data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 0)

    logger.info(f"Loading data: config={data_config_path}, batch_size={batch_size}")
    train_loader, val_loader, test_loader = get_dataloaders(
        config_path=data_config_path,
        splits_csv_path=splits_csv,
        batch_size=batch_size,
        num_workers=num_workers
    )
    logger.info(f"DataLoaders ready: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

    # Model creation
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "baseline_cnn")
    num_classes = model_cfg.get("num_classes", 9)
    in_channels = model_cfg.get("in_channels", 3)
    dropout = model_cfg.get("dropout", 0.3)

    logger.info(f"Building model '{model_name}' (num_classes={num_classes}, in_channels={in_channels})")
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=dropout,
        init_weights=model_cfg.get("init_weights", True)
    )
    param_info = count_parameters(model)
    logger.info(f"Model parameters: Total={param_info['total']:,} | Trainable={param_info['trainable']:,}")

    # Class weights for loss
    train_cfg = config.get("training", {})
    loss_cfg = train_cfg.get("loss", {})
    use_weights = loss_cfg.get("use_class_weights", True)
    class_weights = None
    if use_weights and splits_csv.exists():
        class_weights = compute_class_weights(splits_csv, num_classes=num_classes)
        logger.info(f"Computed inverse class weights: {class_weights.tolist()}")

    criterion = get_loss_fn(
        loss_name=loss_cfg.get("name", "cross_entropy"),
        class_weights=class_weights,
        label_smoothing=loss_cfg.get("label_smoothing", 0.05)
    )

    # Optimizer
    opt_cfg = train_cfg.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adam").lower()
    lr = args.lr or opt_cfg.get("lr", 0.001)
    weight_decay = opt_cfg.get("weight_decay", 0.0001)

    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(f"Optimizer: {opt_name.upper()} (lr={lr}, weight_decay={weight_decay})")

    # Scheduler
    sched_cfg = train_cfg.get("scheduler", {})
    sched_name = sched_cfg.get("name", "reduce_lr_on_plateau").lower()
    scheduler = None
    if sched_name == "reduce_lr_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "min"),
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 3),
            min_lr=sched_cfg.get("min_lr", 1e-5)
        )
    elif sched_name == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.5)
        )

    # Callbacks
    callbacks = []
    # Early Stopping
    es_cfg = train_cfg.get("early_stopping", {})
    if es_cfg:
        callbacks.append(EarlyStopping(
            patience=es_cfg.get("patience", 7),
            min_delta=es_cfg.get("min_delta", 0.001),
            mode=es_cfg.get("mode", "min"),
            monitor=es_cfg.get("monitor", "val_loss")
        ))

    # Checkpoint
    ckpt_dir = PROJECT_ROOT / config.get("artifacts", {}).get("checkpoints_dir", "artifacts/checkpoints")
    callbacks.append(ModelCheckpoint(
        checkpoint_dir=ckpt_dir,
        model_name=model_name,
        mode=es_cfg.get("mode", "min"),
        monitor=es_cfg.get("monitor", "val_loss")
    ))

    # Metric Logger
    metric_logger = MetricLogger(log_dir=log_dir, experiment_name=exp_name)
    callbacks.append(metric_logger)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        callbacks=callbacks,
        config=config,
        logger=logger
    )

    epochs = args.epochs or train_cfg.get("epochs", 15)
    history = trainer.fit(epochs=epochs)

    # Learning curves
    fig_dir = PROJECT_ROOT / config.get("artifacts", {}).get("figures_dir", "artifacts/figures")
    fig_path = fig_dir / f"{exp_name}_learning_curves.png"
    plot_learning_curves(metric_logger.history, fig_path, title=f"VNDCash - {model_name}")
    logger.info(f"Learning curves saved to: {fig_path}")

    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info(f"Best checkpoint saved at: {ckpt_dir / f'{model_name}_best.pt'}")
    logger.info(f"Metrics history saved at: {log_dir / f'{exp_name}_history.csv'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
