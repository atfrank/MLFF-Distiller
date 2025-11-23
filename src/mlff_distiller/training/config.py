"""
Training configuration using Pydantic for validation.

This module provides configuration classes for training MLFF distilled models,
with emphasis on MD-specific requirements and force field accuracy.
"""

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OptimizerConfig(BaseModel):
    """Configuration for optimizer settings."""

    name: Literal["adam", "adamw", "sgd", "rmsprop"] = Field(
        default="adamw", description="Optimizer type"
    )
    learning_rate: float = Field(default=1e-3, gt=0, description="Initial learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay coefficient")
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999), description="Adam/AdamW beta parameters"
    )
    momentum: float = Field(default=0.9, ge=0, le=1, description="SGD momentum")
    eps: float = Field(default=1e-8, gt=0, description="Epsilon for numerical stability")

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, v):
        """Validate beta parameters are in valid range."""
        if not (0 <= v[0] < 1 and 0 <= v[1] < 1):
            raise ValueError("Beta parameters must be in [0, 1)")
        return v


class SchedulerConfig(BaseModel):
    """Configuration for learning rate scheduler."""

    name: Literal["cosine", "step", "plateau", "exponential", "warmup_cosine", "none"] = Field(
        default="warmup_cosine", description="LR scheduler type"
    )
    warmup_steps: int = Field(default=1000, ge=0, description="Number of warmup steps")
    warmup_start_lr: float = Field(default=1e-7, gt=0, description="Starting LR for warmup")
    cosine_t_max: Optional[int] = Field(
        default=None, description="T_max for cosine annealing (defaults to max_epochs)"
    )
    cosine_eta_min: float = Field(default=1e-7, gt=0, description="Minimum LR for cosine")
    step_size: int = Field(default=30, gt=0, description="Step size for StepLR")
    gamma: float = Field(default=0.1, gt=0, le=1, description="Gamma for StepLR/ExponentialLR")
    plateau_mode: Literal["min", "max"] = Field(default="min", description="Mode for ReduceLROnPlateau")
    plateau_factor: float = Field(default=0.5, gt=0, lt=1, description="Factor for ReduceLROnPlateau")
    plateau_patience: int = Field(default=10, gt=0, description="Patience for ReduceLROnPlateau")


class LossConfig(BaseModel):
    """Configuration for loss function weights.

    Following MD requirements:
    - Force errors are CRITICAL (highest weight)
    - Energy conservation is secondary
    - Stress is optional but important for NPT simulations
    """

    energy_weight: float = Field(
        default=1.0, ge=0, description="Weight for energy MSE loss"
    )
    force_weight: float = Field(
        default=100.0, ge=0, description="Weight for force MSE loss (CRITICAL for MD stability)"
    )
    stress_weight: float = Field(
        default=0.1, ge=0, description="Weight for stress MSE loss (optional)"
    )
    force_loss_type: Literal["mse", "mae", "huber"] = Field(
        default="mse", description="Loss function for forces"
    )
    energy_loss_type: Literal["mse", "mae", "huber"] = Field(
        default="mse", description="Loss function for energy"
    )
    stress_loss_type: Literal["mse", "mae", "huber"] = Field(
        default="mse", description="Loss function for stress"
    )
    huber_delta: float = Field(default=1.0, gt=0, description="Delta parameter for Huber loss")

    @field_validator("force_weight")
    @classmethod
    def validate_force_weight(cls, v, info):
        """Ensure force weight is significant (critical for MD)."""
        energy_weight = info.data.get("energy_weight", 1.0)
        if energy_weight > 0 and v < energy_weight:
            import warnings
            warnings.warn(
                f"Force weight ({v}) is less than energy weight ({energy_weight}). "
                "For MD stability, force accuracy is critical and should have higher weight.",
                UserWarning
            )
        return v


class CheckpointConfig(BaseModel):
    """Configuration for checkpointing."""

    checkpoint_dir: Path = Field(
        default=Path("checkpoints"), description="Directory for saving checkpoints"
    )
    save_interval: int = Field(default=5, gt=0, description="Save checkpoint every N epochs")
    keep_last_n: Optional[int] = Field(
        default=3, description="Keep only last N checkpoints (None = keep all)"
    )
    save_best: bool = Field(default=True, description="Save best model based on validation loss")
    save_optimizer: bool = Field(default=True, description="Save optimizer state in checkpoint")


class LoggingConfig(BaseModel):
    """Configuration for training logging."""

    log_interval: int = Field(default=10, gt=0, description="Log metrics every N steps")
    val_interval: int = Field(default=1, gt=0, description="Run validation every N epochs")
    use_tensorboard: bool = Field(default=True, description="Enable TensorBoard logging")
    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging")
    tensorboard_dir: Path = Field(default=Path("runs"), description="TensorBoard log directory")
    wandb_project: Optional[str] = Field(default=None, description="W&B project name")
    wandb_entity: Optional[str] = Field(default=None, description="W&B entity name")
    wandb_run_name: Optional[str] = Field(default=None, description="W&B run name")
    log_gradients: bool = Field(default=False, description="Log gradient statistics")
    log_weights: bool = Field(default=False, description="Log weight statistics")


class TrainingConfig(BaseModel):
    """
    Main training configuration for MLFF distillation.

    This configuration emphasizes MD-specific requirements:
    - Force accuracy is prioritized over energy
    - Support for energy conservation validation
    - Numerical stability for long MD trajectories
    """

    # Basic training settings
    max_epochs: int = Field(default=100, gt=0, description="Maximum training epochs")
    batch_size: int = Field(default=32, gt=0, description="Training batch size")
    val_batch_size: Optional[int] = Field(
        default=None, description="Validation batch size (defaults to batch_size)"
    )
    num_workers: int = Field(default=4, ge=0, description="DataLoader worker processes")
    pin_memory: bool = Field(default=True, description="Pin memory for DataLoader")

    # Gradient and numerical stability
    grad_clip: Optional[float] = Field(
        default=1.0, description="Gradient clipping value (None = no clipping)"
    )
    accumulation_steps: int = Field(
        default=1, gt=0, description="Gradient accumulation steps for effective larger batch"
    )
    mixed_precision: bool = Field(
        default=False, description="Use mixed precision training (FP16/BF16)"
    )
    precision: Literal["fp32", "fp16", "bf16"] = Field(
        default="fp32", description="Training precision"
    )

    # Early stopping
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    early_stopping_patience: int = Field(
        default=20, gt=0, description="Patience for early stopping"
    )
    early_stopping_min_delta: float = Field(
        default=1e-6, ge=0, description="Minimum change to qualify as improvement"
    )

    # Device and reproducibility
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto", description="Training device"
    )
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    deterministic: bool = Field(
        default=False, description="Enable deterministic mode (may reduce performance)"
    )

    # Sub-configurations
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Validation and monitoring
    validate_at_start: bool = Field(
        default=True, description="Run validation before training starts"
    )
    track_force_rmse: bool = Field(
        default=True, description="Track force RMSE metric (critical for MD)"
    )
    track_energy_mae: bool = Field(
        default=True, description="Track energy MAE metric"
    )
    track_stress_rmse: bool = Field(
        default=False, description="Track stress RMSE metric"
    )

    @model_validator(mode='after')
    def set_val_batch_size(self):
        """Set validation batch size to match training if not specified."""
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        return self

    def model_post_init(self, __context):
        """Post-initialization to create directories."""
        self.checkpoint.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.logging.use_tensorboard:
            self.logging.tensorboard_dir.mkdir(parents=True, exist_ok=True)


class DistillationConfig(BaseModel):
    """
    Configuration specific to knowledge distillation.

    This will be used in later milestones (M4) for distillation-specific training.
    Included here for future extensibility.
    """

    # Temperature-based distillation
    temperature: float = Field(
        default=1.0, gt=0, description="Temperature for soft target distillation"
    )
    alpha: float = Field(
        default=0.5, ge=0, le=1, description="Weight for distillation loss vs hard target loss"
    )

    # Feature matching
    use_feature_matching: bool = Field(
        default=False, description="Use intermediate layer feature matching"
    )
    feature_layers: list[str] = Field(
        default_factory=list, description="Layer names for feature matching"
    )
    feature_weight: float = Field(
        default=1.0, ge=0, description="Weight for feature matching loss"
    )

    # Progressive distillation
    use_progressive: bool = Field(
        default=False, description="Use progressive distillation strategy"
    )
    stage1_epochs: int = Field(
        default=50, gt=0, description="Epochs for stage 1 (soft targets only)"
    )
    stage2_epochs: int = Field(
        default=50, gt=0, description="Epochs for stage 2 (hard targets fine-tuning)"
    )


def create_default_config() -> TrainingConfig:
    """Create default training configuration optimized for MD force field distillation."""
    return TrainingConfig(
        max_epochs=100,
        batch_size=32,
        grad_clip=1.0,
        mixed_precision=False,
        optimizer=OptimizerConfig(
            name="adamw",
            learning_rate=1e-3,
            weight_decay=1e-5,
        ),
        scheduler=SchedulerConfig(
            name="warmup_cosine",
            warmup_steps=1000,
        ),
        loss=LossConfig(
            energy_weight=1.0,
            force_weight=100.0,  # Critical: forces matter most for MD
            stress_weight=0.1,
        ),
    )


def load_config(config_path: str | Path) -> TrainingConfig:
    """
    Load training configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated TrainingConfig instance
    """
    import json
    from pathlib import Path

    config_path = Path(config_path)

    if config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    elif config_path.suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, save_path: str | Path) -> None:
    """
    Save training configuration to file.

    Args:
        config: TrainingConfig to save
        save_path: Path to save configuration
    """
    import json
    from pathlib import Path

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2, default=str)
