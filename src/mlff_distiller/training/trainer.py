"""
Main trainer class for MLFF model training.

This module provides the Trainer class that handles:
- Training and validation loops
- Optimizer and scheduler management
- Checkpointing and early stopping
- Logging to TensorBoard and Weights & Biases
- Mixed precision training
- Gradient clipping and accumulation
- Numerical stability monitoring

Designed with MD-specific requirements in mind:
- Force accuracy prioritization
- Energy conservation validation
- Numerical stability for long trajectories
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .losses import ForceFieldLoss, check_loss_numerical_stability


class Trainer:
    """
    Main trainer class for MLFF model training.

    Handles complete training workflow including:
    - Training and validation loops
    - Optimizer and LR scheduler management
    - Checkpointing (best model, periodic, resumption)
    - Logging (TensorBoard, W&B)
    - Early stopping
    - Mixed precision training
    - Gradient clipping and accumulation
    - Numerical stability checks
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            loss_fn: Loss function (defaults to ForceFieldLoss from config)
            device: Device to train on (defaults to config.device)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        if device is None:
            device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Model
        self.model = model.to(self.device)

        # Loss function
        if loss_fn is None:
            self.loss_fn = ForceFieldLoss(
                energy_weight=config.loss.energy_weight,
                force_weight=config.loss.force_weight,
                stress_weight=config.loss.stress_weight,
                energy_loss_type=config.loss.energy_loss_type,
                force_loss_type=config.loss.force_loss_type,
                stress_loss_type=config.loss.stress_loss_type,
                huber_delta=config.loss.huber_delta,
            )
        else:
            self.loss_fn = loss_fn
        self.loss_fn = self.loss_fn.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # LR Scheduler
        self.scheduler = self._create_scheduler()
        self.warmup_scheduler = self._create_warmup_scheduler() if config.scheduler.warmup_steps > 0 else None

        # Mixed precision
        self.use_amp = config.mixed_precision or config.precision in ["fp16", "bf16"]
        self.scaler = None
        if self.use_amp and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16")

        # Logging
        self.writer = None
        self.wandb_run = None
        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.training_history = {"train": [], "val": []}

        # Set random seed for reproducibility
        if config.seed is not None:
            self._set_seed(config.seed)

        # Deterministic mode
        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.optimizer
        params = self.model.parameters()

        if opt_config.name == "adam":
            return Adam(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.name == "adamw":
            return AdamW(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.name == "sgd":
            return SGD(
                params,
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.name == "rmsprop":
            return RMSprop(
                params,
                lr=opt_config.learning_rate,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")

    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler from config."""
        sched_config = self.config.scheduler

        if sched_config.name == "none":
            return None
        elif sched_config.name == "cosine" or sched_config.name == "warmup_cosine":
            t_max = sched_config.cosine_t_max or self.config.max_epochs
            return CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=sched_config.cosine_eta_min,
            )
        elif sched_config.name == "step":
            return StepLR(
                self.optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma,
            )
        elif sched_config.name == "exponential":
            return ExponentialLR(
                self.optimizer,
                gamma=sched_config.gamma,
            )
        elif sched_config.name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config.plateau_mode,
                factor=sched_config.plateau_factor,
                patience=sched_config.plateau_patience,
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_config.name}")

    def _create_warmup_scheduler(self) -> Optional[LambdaLR]:
        """Create warmup scheduler."""
        warmup_steps = self.config.scheduler.warmup_steps
        if warmup_steps == 0:
            return None

        start_lr = self.config.scheduler.warmup_start_lr
        target_lr = self.config.optimizer.learning_rate

        def warmup_lambda(step):
            if step < warmup_steps:
                return start_lr / target_lr + (1 - start_lr / target_lr) * step / warmup_steps
            return 1.0

        return LambdaLR(self.optimizer, lr_lambda=warmup_lambda)

    def _setup_logging(self):
        """Set up logging infrastructure."""
        if self.config.logging.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = self.config.logging.tensorboard_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
                self.writer = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                warnings.warn("TensorBoard not available. Install with: pip install tensorboard")
                self.writer = None

        if self.config.logging.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.logging.wandb_project,
                    entity=self.config.logging.wandb_entity,
                    name=self.config.logging.wandb_run_name,
                    config=self.config.model_dump(),
                )
            except ImportError:
                warnings.warn("W&B not available. Install with: pip install wandb")
                self.wandb_run = None

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _log_metrics(self, metrics: Dict[str, float], phase: str, step: Optional[int] = None):
        """
        Log metrics to TensorBoard and W&B.

        Args:
            metrics: Dictionary of metric name -> value
            phase: 'train' or 'val'
            step: Global step (uses self.global_step if None)
        """
        if step is None:
            step = self.global_step

        # TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{phase}/{name}", value, step)

        # Weights & Biases
        if self.wandb_run is not None:
            wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
            wandb_metrics["epoch"] = self.current_epoch
            import wandb
            wandb.log(wandb_metrics, step=step)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of averaged training metrics
        """
        self.model.train()
        epoch_metrics = {}
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.config.max_epochs}",
            disable=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._batch_to_device(batch)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == "cuda"):
                predictions = self.model(batch)
                loss_dict = self.loss_fn(predictions, batch)
                loss = loss_dict["total"]

                # Scale loss for gradient accumulation
                loss = loss / self.config.accumulation_steps

            # Check for numerical instability
            if not check_loss_numerical_stability(loss_dict):
                warnings.warn(f"NaN or Inf detected in losses at step {self.global_step}")
                # Skip this batch
                continue

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.grad_clip is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
                else:
                    grad_norm = None

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Warmup scheduler
                if self.warmup_scheduler is not None and self.global_step < self.config.scheduler.warmup_steps:
                    self.warmup_scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Log metrics
                if self.global_step % self.config.logging.log_interval == 0:
                    log_dict = {k: (v.mean().item() if v.numel() > 1 else v.item()) if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                    log_dict["lr"] = self.optimizer.param_groups[0]["lr"]
                    if grad_norm is not None:
                        log_dict["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    self._log_metrics(log_dict, "train", self.global_step)

            # Accumulate metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    # Handle multi-element tensors by taking mean before converting to scalar
                    if value.numel() > 1:
                        value = value.mean().item()
                    else:
                        value = value.item()
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            num_batches += 1

            # Update progress bar
            total_loss = loss_dict["total"]
            total_loss_val = (total_loss.mean().item() if total_loss.numel() > 1 else total_loss.item()) * self.config.accumulation_steps
            force_rmse_val = 0.0
            if "force_rmse" in loss_dict:
                force_rmse = loss_dict["force_rmse"]
                force_rmse_val = force_rmse.mean().item() if force_rmse.numel() > 1 else force_rmse.item()
            progress_bar.set_postfix({
                "loss": total_loss_val,
                "force_rmse": force_rmse_val,
            })

        # Average metrics
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of averaged validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        epoch_metrics = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            batch = self._batch_to_device(batch)

            # Forward pass
            predictions = self.model(batch)
            loss_dict = self.loss_fn(predictions, batch)

            # Accumulate metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    # Handle multi-element tensors by taking mean before converting to scalar
                    if value.numel() > 1:
                        value = value.mean().item()
                    else:
                        value = value.item()
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            num_batches += 1

        # Average metrics
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        return epoch_metrics

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def save_checkpoint(
        self,
        filepath: Optional[Path] = None,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint (auto-generated if None)
            is_best: Whether this is the best model
            metadata: Optional metadata to include
        """
        if filepath is None:
            filepath = self.config.checkpoint.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.model_dump(),
            "training_history": self.training_history,
        }

        if self.config.checkpoint.save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            if self.warmup_scheduler is not None:
                checkpoint["warmup_scheduler_state_dict"] = self.warmup_scheduler.state_dict()
            if self.scaler is not None:
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if metadata is not None:
            checkpoint["metadata"] = metadata

        # Save checkpoint
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)

        # Save best model
        if is_best:
            best_path = self.config.checkpoint.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        # Clean up old checkpoints
        if self.config.checkpoint.keep_last_n is not None:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_dir = self.config.checkpoint.checkpoint_dir
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        keep_n = self.config.checkpoint.keep_last_n
        if len(checkpoints) > keep_n:
            for checkpoint in checkpoints[:-keep_n]:
                checkpoint.unlink()

    def load_checkpoint(self, filepath: Path, load_optimizer: bool = True):
        """
        Load checkpoint and resume training.

        Args:
            filepath: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.training_history = checkpoint.get("training_history", {"train": [], "val": []})

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if self.warmup_scheduler is not None and "warmup_scheduler_state_dict" in checkpoint:
                self.warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    def fit(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training history
        """
        print(f"Starting training on {self.device}")
        print(f"Training config: {self.config.max_epochs} epochs, batch size {self.config.batch_size}")

        # Initial validation
        if self.config.validate_at_start and self.val_loader is not None:
            val_metrics = self.validate()
            print(f"Initial validation - Loss: {val_metrics['total']:.4f}")
            if self.config.track_force_rmse and "force_rmse" in val_metrics:
                print(f"  Force RMSE: {val_metrics['force_rmse']:.4f} eV/Å")

        # Training loop
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.training_history["train"].append(train_metrics)

            # Validate
            if self.val_loader is not None and (epoch + 1) % self.config.logging.val_interval == 0:
                val_metrics = self.validate()
                self.training_history["val"].append(val_metrics)

                # Log validation metrics
                self._log_metrics(val_metrics, "val", self.global_step)

                # Print summary
                print(f"Epoch {epoch + 1}/{self.config.max_epochs}")
                print(f"  Train Loss: {train_metrics['total']:.4f}")
                print(f"  Val Loss: {val_metrics['total']:.4f}")
                if self.config.track_force_rmse and "force_rmse" in val_metrics:
                    print(f"  Force RMSE: {val_metrics['force_rmse']:.4f} eV/Å")
                if self.config.track_energy_mae and "energy_mae" in val_metrics:
                    print(f"  Energy MAE: {val_metrics['energy_mae']:.4f} eV")

                # Check for improvement
                val_loss = val_metrics["total"]
                is_best = val_loss < self.best_val_loss

                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                # Save checkpoint
                if (epoch + 1) % self.config.checkpoint.save_interval == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)

                # Early stopping
                if self.config.early_stopping:
                    if self.epochs_without_improvement >= self.config.early_stopping_patience:
                        print(f"Early stopping after {epoch + 1} epochs")
                        break

            # Step scheduler (except plateau which needs validation loss)
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if self.val_loader is not None and (epoch + 1) % self.config.logging.val_interval == 0:
                        self.scheduler.step(val_metrics["total"])
                else:
                    # Only step after warmup
                    if self.global_step >= self.config.scheduler.warmup_steps:
                        self.scheduler.step()

        # Clean up
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()

        return self.training_history
