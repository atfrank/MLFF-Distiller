"""
Training module for MLFF model distillation.

This module provides:
- Trainer: Main training loop with checkpointing, logging, and early stopping
- Loss functions: ForceFieldLoss, EnergyLoss, ForceLoss, DistillationLoss
- Configuration: Pydantic models for training configuration
"""

from .config import (
    CheckpointConfig,
    DistillationConfig,
    LoggingConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    create_default_config,
    load_config,
    save_config,
)
from .losses import (
    DistillationLoss,
    EnergyLoss,
    ForceFieldLoss,
    ForceLoss,
    check_loss_numerical_stability,
    get_loss_function,
)
from .trainer import Trainer

__all__ = [
    # Trainer
    "Trainer",
    # Loss functions
    "ForceFieldLoss",
    "EnergyLoss",
    "ForceLoss",
    "DistillationLoss",
    "get_loss_function",
    "check_loss_numerical_stability",
    # Configuration
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "DistillationConfig",
    "create_default_config",
    "load_config",
    "save_config",
]
