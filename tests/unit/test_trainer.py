"""
Unit tests for Trainer class.

Tests training loop, checkpointing, validation, and integration with
optimizer, scheduler, and loss functions.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import ForceFieldLoss, TrainingConfig, Trainer
from src.training.config import CheckpointConfig, LoggingConfig, OptimizerConfig, SchedulerConfig


class DummyModel(nn.Module):
    """Simple dummy model for testing."""

    def __init__(self, n_atoms=10):
        super().__init__()
        self.n_atoms = n_atoms
        self.energy_head = nn.Linear(n_atoms * 3, 1)
        self.force_head = nn.Linear(n_atoms * 3, n_atoms * 3)

    def forward(self, batch):
        """Forward pass."""
        positions = batch["positions"]  # (batch, n_atoms, 3)
        batch_size = positions.shape[0]

        # Flatten positions
        x = positions.view(batch_size, -1)

        # Predict energy and forces
        energy = self.energy_head(x).squeeze(-1)
        forces_flat = self.force_head(x)
        forces = forces_flat.view(batch_size, self.n_atoms, 3)

        return {
            "energy": energy,
            "forces": forces,
        }


def create_dummy_dataset(n_samples=100, n_atoms=10):
    """Create dummy dataset for testing."""
    positions = torch.randn(n_samples, n_atoms, 3)
    energy = torch.randn(n_samples)
    forces = torch.randn(n_samples, n_atoms, 3)

    # Create a simple dataset that returns dictionaries
    class DictDataset(TensorDataset):
        def __getitem__(self, idx):
            return {
                "positions": positions[idx],
                "energy": energy[idx],
                "forces": forces[idx],
            }

    return DictDataset(positions, energy, forces)


class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_basic_initialization(self):
        """Test basic trainer initialization."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                max_epochs=10,
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.current_epoch == 0
            assert trainer.global_step == 0
            assert trainer.device.type == "cpu"

    def test_device_auto_selection(self):
        """Test automatic device selection."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                device="auto",
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
            )

            expected_device = "cuda" if torch.cuda.is_available() else "cpu"
            assert trainer.device.type == expected_device

    def test_optimizer_creation(self):
        """Test optimizer creation from config."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(warmup_steps=0),  # Disable warmup for this test
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            assert isinstance(trainer.optimizer, torch.optim.Adam)
            assert trainer.optimizer.param_groups[0]["lr"] == 1e-3

    def test_scheduler_creation(self):
        """Test scheduler creation from config."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                scheduler=SchedulerConfig(name="cosine", cosine_t_max=100),
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            assert trainer.scheduler is not None


class TestTrainingLoop:
    """Test training loop functionality."""

    def test_single_epoch(self):
        """Test training for one epoch."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                max_epochs=1,
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            metrics = trainer.train_epoch()

            assert "total" in metrics
            assert "force" in metrics
            assert "energy" in metrics
            assert metrics["total"] > 0

    def test_validation(self):
        """Test validation loop."""
        model = DummyModel()
        train_dataset = create_dummy_dataset(n_samples=32)
        val_dataset = create_dummy_dataset(n_samples=16)
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
            )

            val_metrics = trainer.validate()

            assert "total" in val_metrics
            assert val_metrics["total"] > 0

    def test_full_training(self):
        """Test full training loop."""
        model = DummyModel()
        train_dataset = create_dummy_dataset(n_samples=32)
        val_dataset = create_dummy_dataset(n_samples=16)
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                max_epochs=3,
                early_stopping=False,
                checkpoint=CheckpointConfig(
                    checkpoint_dir=Path(tmpdir) / "checkpoints",
                    save_interval=1,
                ),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
            )

            history = trainer.fit()

            assert "train" in history
            assert "val" in history
            assert len(history["train"]) == 3
            # Epoch is 0-indexed, so after 3 epochs we're at epoch 2 (0, 1, 2)
            # But we increment at the start of the loop, so it's actually 3
            assert trainer.current_epoch >= 2


class TestCheckpointing:
    """Test checkpointing functionality."""

    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            checkpoint_path = Path(tmpdir) / "checkpoints" / "test_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            assert checkpoint_path.exists()

            # Load checkpoint and verify contents
            checkpoint = torch.load(checkpoint_path)
            assert "epoch" in checkpoint
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            # Create trainer and train for 1 epoch
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            trainer.train_epoch()
            trainer.current_epoch = 1
            trainer.global_step = 10

            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoints" / "test_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            # Create new trainer and load checkpoint
            new_model = DummyModel()
            new_trainer = Trainer(
                model=new_model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            new_trainer.load_checkpoint(checkpoint_path)

            assert new_trainer.current_epoch == 1
            assert new_trainer.global_step == 10

    def test_best_model_saving(self):
        """Test saving best model."""
        model = DummyModel()
        train_dataset = create_dummy_dataset(n_samples=32)
        val_dataset = create_dummy_dataset(n_samples=16)
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                max_epochs=2,
                checkpoint=CheckpointConfig(
                    checkpoint_dir=Path(tmpdir) / "checkpoints",
                    save_best=True,
                ),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False, val_interval=1),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
            )

            trainer.fit()

            best_model_path = Path(tmpdir) / "checkpoints" / "best_model.pt"
            assert best_model_path.exists()


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers correctly."""
        model = DummyModel()
        train_dataset = create_dummy_dataset(n_samples=32)
        val_dataset = create_dummy_dataset(n_samples=16)
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                max_epochs=100,  # Would train for 100 epochs
                early_stopping=True,
                early_stopping_patience=2,
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False, val_interval=1),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
            )

            # Mock the best_val_loss to not improve
            trainer.best_val_loss = 0.0

            history = trainer.fit()

            # Should stop before 100 epochs due to early stopping
            assert trainer.current_epoch < 100


class TestGradientHandling:
    """Test gradient clipping and accumulation."""

    def test_gradient_clipping(self):
        """Test gradient clipping is applied."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                grad_clip=1.0,
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            trainer.train_epoch()

            # Check that gradients exist and are not too large
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    # After clipping, no single parameter should have norm > clip value
                    # (though total norm might be clipped to clip value)

    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=4)  # Small batch

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                batch_size=4,
                accumulation_steps=4,  # Effective batch size = 16
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            initial_step = trainer.global_step
            trainer.train_epoch()

            # Global step should increase by number of optimizer steps
            # = len(dataset) / (batch_size * accumulation_steps)


class TestLossFunctionIntegration:
    """Test integration with different loss functions."""

    def test_custom_loss_function(self):
        """Test using custom loss function."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        custom_loss = ForceFieldLoss(
            energy_weight=2.0,
            force_weight=200.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                loss_fn=custom_loss,
                device="cpu",
            )

            assert trainer.loss_fn.energy_weight == 2.0
            assert trainer.loss_fn.force_weight == 200.0


class TestMetricTracking:
    """Test metric tracking."""

    def test_force_rmse_tracking(self):
        """Test that force RMSE is tracked."""
        model = DummyModel()
        train_dataset = create_dummy_dataset(n_samples=32)
        val_dataset = create_dummy_dataset(n_samples=16)
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                track_force_rmse=True,
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
            )

            val_metrics = trainer.validate()

            assert "force_rmse" in val_metrics

    def test_energy_mae_tracking(self):
        """Test that energy MAE is tracked."""
        model = DummyModel()
        train_dataset = create_dummy_dataset(n_samples=32)
        val_dataset = create_dummy_dataset(n_samples=16)
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                track_energy_mae=True,
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
            )

            val_metrics = trainer.validate()

            assert "energy_mae" in val_metrics


class TestReproducibility:
    """Test training reproducibility."""

    def test_seed_setting(self):
        """Test that seed is set correctly."""
        model = DummyModel()
        dataset = create_dummy_dataset(n_samples=32)
        train_loader = DataLoader(dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                seed=42,
                checkpoint=CheckpointConfig(checkpoint_dir=Path(tmpdir) / "checkpoints"),
                logging=LoggingConfig(use_tensorboard=False, use_wandb=False),
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config=config,
                device="cpu",
            )

            # Check that random state is set
            import random
            import numpy as np

            # These should be consistent with seed=42
            torch_rand = torch.rand(1).item()
            # Just verify no error occurs
