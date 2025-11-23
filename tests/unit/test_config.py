"""
Unit tests for training configuration.

Tests Pydantic validation, default values, and configuration loading/saving.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.training.config import (
    CheckpointConfig,
    LossConfig,
    LoggingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    create_default_config,
    load_config,
    save_config,
)


class TestOptimizerConfig:
    """Test OptimizerConfig validation."""

    def test_default_config(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        assert config.name == "adamw"
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.0
        assert config.betas == (0.9, 0.999)

    def test_valid_config(self):
        """Test valid optimizer configuration."""
        config = OptimizerConfig(
            name="adam",
            learning_rate=1e-4,
            weight_decay=1e-5,
            betas=(0.95, 0.9999),
        )
        assert config.name == "adam"
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 1e-5
        assert config.betas == (0.95, 0.9999)

    def test_invalid_lr(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError):
            OptimizerConfig(learning_rate=-1e-3)

    def test_invalid_betas(self):
        """Test that invalid beta values raise error."""
        with pytest.raises(ValueError):
            OptimizerConfig(betas=(1.5, 0.999))


class TestSchedulerConfig:
    """Test SchedulerConfig validation."""

    def test_default_config(self):
        """Test default scheduler configuration."""
        config = SchedulerConfig()
        assert config.name == "warmup_cosine"
        assert config.warmup_steps == 1000
        assert config.warmup_start_lr == 1e-7

    def test_valid_config(self):
        """Test valid scheduler configuration."""
        config = SchedulerConfig(
            name="cosine",
            warmup_steps=500,
            cosine_t_max=100,
            cosine_eta_min=1e-6,
        )
        assert config.name == "cosine"
        assert config.warmup_steps == 500
        assert config.cosine_t_max == 100


class TestLossConfig:
    """Test LossConfig validation."""

    def test_default_config(self):
        """Test default loss configuration."""
        config = LossConfig()
        assert config.energy_weight == 1.0
        assert config.force_weight == 100.0  # Forces should dominate
        assert config.stress_weight == 0.1

    def test_force_weight_warning(self):
        """Test warning when force weight is too low."""
        with pytest.warns(UserWarning, match="Force weight.*less than energy weight"):
            LossConfig(energy_weight=100.0, force_weight=1.0)

    def test_valid_config(self):
        """Test valid loss configuration."""
        config = LossConfig(
            energy_weight=1.0,
            force_weight=200.0,
            stress_weight=0.5,
            force_loss_type="mae",
        )
        assert config.energy_weight == 1.0
        assert config.force_weight == 200.0
        assert config.force_loss_type == "mae"


class TestCheckpointConfig:
    """Test CheckpointConfig validation."""

    def test_default_config(self):
        """Test default checkpoint configuration."""
        config = CheckpointConfig()
        assert config.checkpoint_dir == Path("checkpoints")
        assert config.save_interval == 5
        assert config.keep_last_n == 3
        assert config.save_best is True

    def test_custom_path(self):
        """Test custom checkpoint directory."""
        config = CheckpointConfig(checkpoint_dir=Path("/tmp/checkpoints"))
        assert config.checkpoint_dir == Path("/tmp/checkpoints")


class TestLoggingConfig:
    """Test LoggingConfig validation."""

    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.log_interval == 10
        assert config.val_interval == 1
        assert config.use_tensorboard is True
        assert config.use_wandb is False

    def test_wandb_config(self):
        """Test W&B configuration."""
        config = LoggingConfig(
            use_wandb=True,
            wandb_project="mlff-distiller",
            wandb_entity="test-team",
        )
        assert config.use_wandb is True
        assert config.wandb_project == "mlff-distiller"


class TestTrainingConfig:
    """Test main TrainingConfig."""

    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.max_epochs == 100
        assert config.batch_size == 32
        assert config.grad_clip == 1.0
        assert config.mixed_precision is False
        assert config.early_stopping is True

    def test_nested_configs(self):
        """Test nested configuration structure."""
        config = TrainingConfig(
            optimizer=OptimizerConfig(learning_rate=1e-4),
            scheduler=SchedulerConfig(warmup_steps=500),
            loss=LossConfig(force_weight=150.0),
        )
        assert config.optimizer.learning_rate == 1e-4
        assert config.scheduler.warmup_steps == 500
        assert config.loss.force_weight == 150.0

    def test_val_batch_size_default(self):
        """Test that validation batch size defaults to training batch size."""
        config = TrainingConfig(batch_size=64)
        assert config.val_batch_size == 64

    def test_val_batch_size_custom(self):
        """Test custom validation batch size."""
        config = TrainingConfig(batch_size=32, val_batch_size=64)
        assert config.val_batch_size == 64

    def test_directory_creation(self):
        """Test that directories are created on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(
                    use_tensorboard=True,
                    tensorboard_dir=Path(tmpdir) / "runs",
                ),
            )
            assert config.checkpoint.checkpoint_dir.exists()
            assert config.logging.tensorboard_dir.exists()


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_save_and_load_json(self):
        """Test saving and loading configuration as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Create config
            original_config = TrainingConfig(
                max_epochs=50,
                batch_size=16,
                optimizer=OptimizerConfig(learning_rate=5e-4),
            )

            # Save
            save_config(original_config, config_path)
            assert config_path.exists()

            # Load
            loaded_config = load_config(config_path)
            assert loaded_config.max_epochs == 50
            assert loaded_config.batch_size == 16
            assert loaded_config.optimizer.learning_rate == 5e-4

    def test_model_dump(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(max_epochs=50, batch_size=16)
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["max_epochs"] == 50
        assert config_dict["batch_size"] == 16
        assert "optimizer" in config_dict
        assert "scheduler" in config_dict


class TestCreateDefaultConfig:
    """Test default configuration factory."""

    def test_create_default(self):
        """Test creating default configuration."""
        config = create_default_config()

        assert isinstance(config, TrainingConfig)
        assert config.max_epochs == 100
        assert config.batch_size == 32

        # Check MD-specific defaults
        assert config.loss.force_weight == 100.0  # Forces prioritized
        assert config.loss.energy_weight == 1.0
        assert config.optimizer.name == "adamw"
        assert config.scheduler.name == "warmup_cosine"


class TestConfigValidation:
    """Test configuration validation edge cases."""

    def test_negative_epochs(self):
        """Test that negative epochs raise error."""
        with pytest.raises(ValueError):
            TrainingConfig(max_epochs=-1)

    def test_zero_batch_size(self):
        """Test that zero batch size raises error."""
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)

    def test_invalid_device(self):
        """Test that invalid device raises error."""
        with pytest.raises(ValueError):
            TrainingConfig(device="invalid")

    def test_invalid_precision(self):
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError):
            TrainingConfig(precision="fp8")


class TestMDSpecificConfig:
    """Test MD-specific configuration requirements."""

    def test_force_priority_default(self):
        """Test that forces are prioritized by default."""
        config = create_default_config()
        assert config.loss.force_weight > config.loss.energy_weight

    def test_force_tracking_enabled(self):
        """Test that force tracking is enabled by default."""
        config = TrainingConfig()
        assert config.track_force_rmse is True

    def test_energy_tracking_enabled(self):
        """Test that energy tracking is enabled by default."""
        config = TrainingConfig()
        assert config.track_energy_mae is True
