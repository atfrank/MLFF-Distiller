"""
Unit tests for loss functions.

Tests ForceFieldLoss, EnergyLoss, ForceLoss, and numerical stability.
"""

import pytest
import torch

from mlff_distiller.training.losses import (
    DistillationLoss,
    EnergyLoss,
    ForceFieldLoss,
    ForceLoss,
    check_loss_numerical_stability,
    get_loss_function,
)


class TestForceFieldLoss:
    """Test ForceFieldLoss functionality."""

    def test_initialization(self):
        """Test loss function initialization."""
        loss_fn = ForceFieldLoss(
            energy_weight=1.0,
            force_weight=100.0,
            stress_weight=0.1,
        )
        assert loss_fn.energy_weight == 1.0
        assert loss_fn.force_weight == 100.0
        assert loss_fn.stress_weight == 0.1

    def test_force_priority_warning(self):
        """Test warning when force weight is too low."""
        with pytest.warns(UserWarning, match="Force weight.*energy weight"):
            ForceFieldLoss(energy_weight=100.0, force_weight=1.0)

    def test_energy_only_loss(self):
        """Test energy-only loss computation."""
        loss_fn = ForceFieldLoss(
            energy_weight=1.0,
            force_weight=0.0,
            stress_weight=0.0,
        )

        batch_size = 4
        predictions = {
            "energy": torch.randn(batch_size),
        }
        targets = {
            "energy": torch.randn(batch_size),
        }

        loss_dict = loss_fn(predictions, targets)

        assert "total" in loss_dict
        assert "energy" in loss_dict
        assert "energy_rmse" in loss_dict
        assert "energy_mae" in loss_dict
        assert "force" not in loss_dict

    def test_force_only_loss(self):
        """Test force-only loss computation."""
        loss_fn = ForceFieldLoss(
            energy_weight=0.0,
            force_weight=1.0,
            stress_weight=0.0,
        )

        batch_size = 4
        n_atoms = 10
        predictions = {
            "forces": torch.randn(batch_size, n_atoms, 3),
        }
        targets = {
            "forces": torch.randn(batch_size, n_atoms, 3),
        }

        loss_dict = loss_fn(predictions, targets)

        assert "total" in loss_dict
        assert "force" in loss_dict
        assert "force_rmse" in loss_dict
        assert "force_mae" in loss_dict
        assert "force_rmse_x" in loss_dict
        assert "force_rmse_y" in loss_dict
        assert "force_rmse_z" in loss_dict
        assert "energy" not in loss_dict

    def test_combined_loss(self):
        """Test combined energy + force loss."""
        loss_fn = ForceFieldLoss(
            energy_weight=1.0,
            force_weight=100.0,
            stress_weight=0.0,
            angular_weight=0.0,  # Disable angular loss for simpler expected total calculation
        )

        batch_size = 4
        n_atoms = 10
        predictions = {
            "energy": torch.randn(batch_size),
            "forces": torch.randn(batch_size, n_atoms, 3),
        }
        targets = {
            "energy": torch.randn(batch_size),
            "forces": torch.randn(batch_size, n_atoms, 3),
        }

        loss_dict = loss_fn(predictions, targets)

        assert "total" in loss_dict
        assert "energy" in loss_dict
        assert "force" in loss_dict
        assert "energy_rmse" in loss_dict
        assert "force_rmse" in loss_dict

        # Total should be weighted sum (energy + force only, angular disabled)
        expected_total = (
            1.0 * loss_dict["energy"] + 100.0 * loss_dict["force"]
        )
        assert torch.isclose(loss_dict["total"], expected_total, atol=1e-6)

    def test_stress_loss(self):
        """Test stress loss computation."""
        loss_fn = ForceFieldLoss(
            energy_weight=0.0,
            force_weight=0.0,
            stress_weight=1.0,
        )

        batch_size = 4
        predictions = {
            "stress": torch.randn(batch_size, 3, 3),
        }
        targets = {
            "stress": torch.randn(batch_size, 3, 3),
        }

        loss_dict = loss_fn(predictions, targets)

        assert "total" in loss_dict
        assert "stress" in loss_dict
        assert "stress_rmse" in loss_dict
        assert "stress_mae" in loss_dict

    def test_mse_loss_type(self):
        """Test MSE loss type."""
        loss_fn = ForceFieldLoss(
            energy_weight=1.0,
            force_weight=0.0,
            energy_loss_type="mse",
        )

        predictions = {"energy": torch.tensor([1.0, 2.0, 3.0])}
        targets = {"energy": torch.tensor([1.5, 2.5, 3.5])}

        loss_dict = loss_fn(predictions, targets)

        # MSE = mean((pred - target)^2) = mean([0.25, 0.25, 0.25]) = 0.25
        assert torch.isclose(loss_dict["energy"], torch.tensor(0.25), atol=1e-6)

    def test_mae_loss_type(self):
        """Test MAE loss type."""
        loss_fn = ForceFieldLoss(
            energy_weight=1.0,
            force_weight=0.0,
            energy_loss_type="mae",
        )

        predictions = {"energy": torch.tensor([1.0, 2.0, 3.0])}
        targets = {"energy": torch.tensor([1.5, 2.5, 3.5])}

        loss_dict = loss_fn(predictions, targets)

        # MAE = mean(|pred - target|) = mean([0.5, 0.5, 0.5]) = 0.5
        assert torch.isclose(loss_dict["energy"], torch.tensor(0.5), atol=1e-6)

    def test_huber_loss_type(self):
        """Test Huber loss type."""
        loss_fn = ForceFieldLoss(
            energy_weight=1.0,
            force_weight=0.0,
            energy_loss_type="huber",
            huber_delta=1.0,
        )

        predictions = {"energy": torch.tensor([1.0, 2.0, 3.0])}
        targets = {"energy": torch.tensor([1.5, 2.5, 3.5])}

        loss_dict = loss_fn(predictions, targets)

        assert "energy" in loss_dict
        assert loss_dict["energy"] > 0

    def test_loss_with_mask(self):
        """Test loss computation with mask."""
        loss_fn = ForceFieldLoss(energy_weight=1.0, force_weight=0.0)

        batch_size = 4
        predictions = {"energy": torch.randn(batch_size)}
        targets = {"energy": torch.randn(batch_size)}
        mask = {"energy": torch.tensor([True, True, False, False])}

        loss_dict = loss_fn(predictions, targets, mask)

        assert "energy" in loss_dict
        # Loss should only use first 2 elements

    def test_zero_weights(self):
        """Test that zero weights skip computation."""
        loss_fn = ForceFieldLoss(
            energy_weight=0.0,
            force_weight=0.0,
            stress_weight=0.0,
        )

        predictions = {
            "energy": torch.randn(4),
            "forces": torch.randn(4, 10, 3),
        }
        targets = {
            "energy": torch.randn(4),
            "forces": torch.randn(4, 10, 3),
        }

        loss_dict = loss_fn(predictions, targets)

        assert loss_dict["total"] == 0.0
        assert "energy" not in loss_dict
        assert "force" not in loss_dict


class TestEnergyLoss:
    """Test EnergyLoss functionality."""

    def test_energy_loss(self):
        """Test energy-only loss."""
        loss_fn = EnergyLoss(loss_type="mse")

        predictions = {"energy": torch.tensor([1.0, 2.0, 3.0])}
        targets = {"energy": torch.tensor([1.5, 2.5, 3.5])}

        loss_dict = loss_fn(predictions, targets)

        assert "total" in loss_dict
        assert "energy" in loss_dict
        assert "energy_rmse" in loss_dict
        assert "energy_mae" in loss_dict
        assert torch.isclose(loss_dict["energy"], torch.tensor(0.25), atol=1e-6)


class TestForceLoss:
    """Test ForceLoss functionality."""

    def test_force_loss(self):
        """Test force-only loss."""
        loss_fn = ForceLoss(loss_type="mse")

        batch_size = 2
        n_atoms = 3
        predictions = {"forces": torch.zeros(batch_size, n_atoms, 3)}
        targets = {"forces": torch.ones(batch_size, n_atoms, 3)}

        loss_dict = loss_fn(predictions, targets)

        assert "total" in loss_dict
        assert "force" in loss_dict
        assert "force_rmse" in loss_dict
        assert "force_mae" in loss_dict
        assert "force_rmse_x" in loss_dict

        # MSE = mean((0 - 1)^2) = 1.0
        assert torch.isclose(loss_dict["force"], torch.tensor(1.0), atol=1e-6)


class TestDistillationLoss:
    """Test DistillationLoss functionality."""

    def test_hard_targets_only(self):
        """Test distillation loss with hard targets only."""
        base_loss = ForceFieldLoss(energy_weight=1.0, force_weight=0.0)
        loss_fn = DistillationLoss(base_loss=base_loss, alpha=0.0)

        predictions = {"energy": torch.randn(4)}
        hard_targets = {"energy": torch.randn(4)}

        loss_dict = loss_fn(predictions, hard_targets, soft_targets=None)

        assert "total" in loss_dict
        assert "energy" in loss_dict

    def test_soft_targets(self):
        """Test distillation loss with soft targets."""
        base_loss = ForceFieldLoss(energy_weight=1.0, force_weight=0.0)
        loss_fn = DistillationLoss(base_loss=base_loss, alpha=0.5, temperature=2.0)

        predictions = {"energy": torch.randn(4)}
        hard_targets = {"energy": torch.randn(4)}
        soft_targets = {"energy": torch.randn(4)}

        loss_dict = loss_fn(predictions, hard_targets, soft_targets)

        assert "total" in loss_dict
        assert "distill_loss" in loss_dict
        assert "hard_loss" in loss_dict


class TestNumericalStability:
    """Test numerical stability checks."""

    def test_stable_losses(self):
        """Test that normal losses are stable."""
        loss_dict = {
            "total": torch.tensor(1.0),
            "energy": torch.tensor(0.5),
            "force": torch.tensor(0.5),
        }
        assert check_loss_numerical_stability(loss_dict) is True

    def test_nan_detection(self):
        """Test NaN detection."""
        loss_dict = {
            "total": torch.tensor(float("nan")),
            "energy": torch.tensor(0.5),
        }
        assert check_loss_numerical_stability(loss_dict) is False

    def test_inf_detection(self):
        """Test Inf detection."""
        loss_dict = {
            "total": torch.tensor(1.0),
            "energy": torch.tensor(float("inf")),
        }
        assert check_loss_numerical_stability(loss_dict) is False


class TestLossFactory:
    """Test loss function factory."""

    def test_force_field_factory(self):
        """Test creating ForceFieldLoss via factory."""
        loss_fn = get_loss_function("force_field", energy_weight=1.0, force_weight=100.0)
        assert isinstance(loss_fn, ForceFieldLoss)
        assert loss_fn.energy_weight == 1.0
        assert loss_fn.force_weight == 100.0

    def test_energy_factory(self):
        """Test creating EnergyLoss via factory."""
        loss_fn = get_loss_function("energy")
        assert isinstance(loss_fn, EnergyLoss)

    def test_force_factory(self):
        """Test creating ForceLoss via factory."""
        loss_fn = get_loss_function("force")
        assert isinstance(loss_fn, ForceLoss)

    def test_distillation_factory(self):
        """Test creating DistillationLoss via factory."""
        loss_fn = get_loss_function("distillation", temperature=2.0, alpha=0.5)
        assert isinstance(loss_fn, DistillationLoss)

    def test_invalid_type(self):
        """Test invalid loss type raises error."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_function("invalid")


class TestLossGradients:
    """Test that losses produce valid gradients."""

    def test_energy_gradients(self):
        """Test energy loss produces gradients."""
        loss_fn = ForceFieldLoss(energy_weight=1.0, force_weight=0.0)

        energy_pred = torch.randn(4, requires_grad=True)
        energy_target = torch.randn(4)

        predictions = {"energy": energy_pred}
        targets = {"energy": energy_target}

        loss_dict = loss_fn(predictions, targets)
        loss_dict["total"].backward()

        assert energy_pred.grad is not None
        assert not torch.isnan(energy_pred.grad).any()

    def test_force_gradients(self):
        """Test force loss produces gradients."""
        loss_fn = ForceFieldLoss(energy_weight=0.0, force_weight=1.0)

        forces_pred = torch.randn(2, 5, 3, requires_grad=True)
        forces_target = torch.randn(2, 5, 3)

        predictions = {"forces": forces_pred}
        targets = {"forces": forces_target}

        loss_dict = loss_fn(predictions, targets)
        loss_dict["total"].backward()

        assert forces_pred.grad is not None
        assert not torch.isnan(forces_pred.grad).any()


class TestLossDimensions:
    """Test loss functions handle various tensor dimensions correctly."""

    def test_batch_size_one(self):
        """Test loss with batch size of 1."""
        loss_fn = ForceFieldLoss(energy_weight=1.0, force_weight=1.0)

        predictions = {
            "energy": torch.randn(1),
            "forces": torch.randn(1, 10, 3),
        }
        targets = {
            "energy": torch.randn(1),
            "forces": torch.randn(1, 10, 3),
        }

        loss_dict = loss_fn(predictions, targets)
        assert loss_dict["total"].dim() == 0  # Scalar loss

    def test_large_batch(self):
        """Test loss with large batch size."""
        loss_fn = ForceFieldLoss(energy_weight=1.0, force_weight=1.0)

        batch_size = 128
        predictions = {
            "energy": torch.randn(batch_size),
            "forces": torch.randn(batch_size, 20, 3),
        }
        targets = {
            "energy": torch.randn(batch_size),
            "forces": torch.randn(batch_size, 20, 3),
        }

        loss_dict = loss_fn(predictions, targets)
        assert loss_dict["total"].dim() == 0  # Scalar loss

    def test_varying_atoms(self):
        """Test loss with different number of atoms."""
        loss_fn = ForceLoss()

        # Different numbers of atoms should work
        for n_atoms in [5, 10, 50, 100]:
            predictions = {"forces": torch.randn(4, n_atoms, 3)}
            targets = {"forces": torch.randn(4, n_atoms, 3)}

            loss_dict = loss_fn(predictions, targets)
            assert loss_dict["total"] > 0
