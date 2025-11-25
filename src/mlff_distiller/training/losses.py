"""
Loss functions for training MLFF models.

This module implements loss functions specifically designed for molecular force fields,
with emphasis on force accuracy (critical for MD stability) and energy conservation.
"""

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForceFieldLoss(nn.Module):
    """
    Combined loss function for force field training.

    Implements weighted combination of:
    - Energy loss (MSE/MAE/Huber)
    - Force loss (MSE/MAE/Huber) - CRITICAL for MD stability
    - Stress loss (MSE/MAE/Huber) - Optional, for NPT simulations

    Following MD requirements:
    - Force accuracy is prioritized (default weight: 100x energy)
    - Proper handling of units (eV for energy, eV/Angstrom for forces)
    - Numerical stability for long MD trajectories
    """

    def __init__(
        self,
        energy_weight: float = 1.0,
        force_weight: float = 100.0,
        stress_weight: float = 0.1,
        angular_weight: float = 10.0,
        energy_loss_type: Literal["mse", "mae", "huber"] = "mse",
        force_loss_type: Literal["mse", "mae", "huber"] = "mse",
        stress_loss_type: Literal["mse", "mae", "huber"] = "mse",
        huber_delta: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """
        Initialize ForceFieldLoss.

        Args:
            energy_weight: Weight for energy loss component
            force_weight: Weight for force loss component (should be >> energy_weight for MD)
            stress_weight: Weight for stress loss component
            angular_weight: Weight for angular force loss (directional accuracy)
            energy_loss_type: Loss function type for energy predictions
            force_loss_type: Loss function type for force predictions
            stress_loss_type: Loss function type for stress predictions
            huber_delta: Delta parameter for Huber loss
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.angular_weight = angular_weight
        self.huber_delta = huber_delta
        self.reduction = reduction

        # Store loss types
        self.energy_loss_type = energy_loss_type
        self.force_loss_type = force_loss_type
        self.stress_loss_type = stress_loss_type

        # Warn if force weight is not dominant
        if force_weight > 0 and energy_weight > 0 and force_weight < energy_weight:
            import warnings
            warnings.warn(
                f"Force weight ({force_weight}) < energy weight ({energy_weight}). "
                "For MD stability, forces should be weighted higher than energy.",
                UserWarning
            )

    def _compute_component_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """
        Compute a single loss component.

        Args:
            pred: Predicted values
            target: Target values
            loss_type: Type of loss ('mse', 'mae', 'huber')

        Returns:
            Loss value
        """
        if loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        elif loss_type == "mae":
            loss = F.l1_loss(pred, target, reduction=self.reduction)
        elif loss_type == "huber":
            loss = F.huber_loss(pred, target, delta=self.huber_delta, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return loss

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined force field loss.

        Args:
            predictions: Dictionary with keys 'energy', 'forces', optionally 'stress'
                - energy: (batch_size,) tensor
                - forces: (batch_size, n_atoms, 3) tensor
                - stress: (batch_size, 3, 3) or (batch_size, 6) tensor
            targets: Dictionary with same structure as predictions
            mask: Optional dictionary of boolean masks for selective loss computation

        Returns:
            Dictionary containing:
                - 'total': Total weighted loss
                - 'energy': Energy loss component
                - 'force': Force loss component
                - 'stress': Stress loss component (if applicable)
                - 'energy_rmse': Energy RMSE metric
                - 'force_rmse': Force RMSE metric (critical for MD)
                - 'stress_rmse': Stress RMSE metric (if applicable)
        """
        losses = {}
        total_loss = 0.0

        # Energy loss
        if "energy" in predictions and "energy" in targets and self.energy_weight > 0:
            pred_energy = predictions["energy"]
            target_energy = targets["energy"]

            if mask is not None and "energy" in mask:
                pred_energy = pred_energy[mask["energy"]]
                target_energy = target_energy[mask["energy"]]

            energy_loss = self._compute_component_loss(
                pred_energy, target_energy, self.energy_loss_type
            )
            losses["energy"] = energy_loss
            total_loss = total_loss + self.energy_weight * energy_loss

            # Compute RMSE for monitoring (always use MSE reduction for RMSE)
            with torch.no_grad():
                energy_mse = F.mse_loss(pred_energy, target_energy, reduction="mean")
                losses["energy_rmse"] = torch.sqrt(energy_mse)
                losses["energy_mae"] = F.l1_loss(pred_energy, target_energy, reduction="mean")

        # Force loss (CRITICAL for MD stability)
        if "forces" in predictions and "forces" in targets and self.force_weight > 0:
            pred_forces = predictions["forces"]
            target_forces = targets["forces"]

            if mask is not None and "forces" in mask:
                pred_forces = pred_forces[mask["forces"]]
                target_forces = target_forces[mask["forces"]]

            force_loss = self._compute_component_loss(
                pred_forces, target_forces, self.force_loss_type
            )
            losses["force"] = force_loss
            total_loss = total_loss + self.force_weight * force_loss

            # Angular loss (directional accuracy)
            if self.angular_weight > 0:
                # Cosine similarity loss: 1 - cos(θ) between force vectors
                # Normalize force vectors (avoid division by zero)
                pred_norms = torch.norm(pred_forces, dim=-1, keepdim=True).clamp(min=1e-8)
                target_norms = torch.norm(target_forces, dim=-1, keepdim=True).clamp(min=1e-8)

                pred_normalized = pred_forces / pred_norms
                target_normalized = target_forces / target_norms

                # Cosine similarity: dot product of normalized vectors
                cos_sim = (pred_normalized * target_normalized).sum(dim=-1)

                # Angular loss: 1 - cos(θ), ranges from 0 (parallel) to 2 (antiparallel)
                if self.reduction == "mean":
                    angular_loss = (1.0 - cos_sim).mean()
                elif self.reduction == "sum":
                    angular_loss = (1.0 - cos_sim).sum()
                else:
                    angular_loss = 1.0 - cos_sim

                losses["angular"] = angular_loss
                total_loss = total_loss + self.angular_weight * angular_loss

                # Angular error in degrees (for monitoring)
                with torch.no_grad():
                    # Clamp cos_sim to [-1, 1] to avoid numerical issues with acos
                    cos_sim_clamped = torch.clamp(cos_sim, -1.0, 1.0)
                    angular_error_rad = torch.acos(cos_sim_clamped)
                    angular_error_deg = angular_error_rad * (180.0 / 3.14159265)
                    losses["angular_error_mean_deg"] = angular_error_deg.mean()
                    losses["angular_error_max_deg"] = angular_error_deg.max()

            # Compute RMSE for monitoring (critical MD metric)
            with torch.no_grad():
                force_mse = F.mse_loss(pred_forces, target_forces, reduction="mean")
                losses["force_rmse"] = torch.sqrt(force_mse)
                losses["force_mae"] = F.l1_loss(pred_forces, target_forces, reduction="mean")
                # Component-wise RMSE for detailed analysis
                if pred_forces.dim() >= 2 and pred_forces.shape[-1] == 3:
                    # Compute MSE per component (x, y, z) by flattening batch and atom dims
                    # pred_forces: (batch, n_atoms, 3) -> compute MSE along axis 0 and 1
                    diff_sq = (pred_forces - target_forces) ** 2
                    # Mean over all but last dimension to get per-component MSE
                    force_mse_x = diff_sq[..., 0].mean()
                    force_mse_y = diff_sq[..., 1].mean()
                    force_mse_z = diff_sq[..., 2].mean()
                    losses["force_rmse_x"] = torch.sqrt(force_mse_x)
                    losses["force_rmse_y"] = torch.sqrt(force_mse_y)
                    losses["force_rmse_z"] = torch.sqrt(force_mse_z)

        # Stress loss (for NPT simulations)
        if "stress" in predictions and "stress" in targets and self.stress_weight > 0:
            pred_stress = predictions["stress"]
            target_stress = targets["stress"]

            if mask is not None and "stress" in mask:
                pred_stress = pred_stress[mask["stress"]]
                target_stress = target_stress[mask["stress"]]

            stress_loss = self._compute_component_loss(
                pred_stress, target_stress, self.stress_loss_type
            )
            losses["stress"] = stress_loss
            total_loss = total_loss + self.stress_weight * stress_loss

            # Compute RMSE for monitoring
            with torch.no_grad():
                stress_mse = F.mse_loss(pred_stress, target_stress, reduction="mean")
                losses["stress_rmse"] = torch.sqrt(stress_mse)
                losses["stress_mae"] = F.l1_loss(pred_stress, target_stress, reduction="mean")

        losses["total"] = total_loss
        return losses


class EnergyLoss(nn.Module):
    """
    Energy-only loss function.

    Simple wrapper for consistency with ForceFieldLoss API.
    """

    def __init__(
        self,
        loss_type: Literal["mse", "mae", "huber"] = "mse",
        huber_delta: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.reduction = reduction

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute energy loss."""
        pred_energy = predictions["energy"]
        target_energy = targets["energy"]

        if mask is not None and "energy" in mask:
            pred_energy = pred_energy[mask["energy"]]
            target_energy = target_energy[mask["energy"]]

        if self.loss_type == "mse":
            loss = F.mse_loss(pred_energy, target_energy, reduction=self.reduction)
        elif self.loss_type == "mae":
            loss = F.l1_loss(pred_energy, target_energy, reduction=self.reduction)
        elif self.loss_type == "huber":
            loss = F.huber_loss(
                pred_energy, target_energy, delta=self.huber_delta, reduction=self.reduction
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Metrics
        with torch.no_grad():
            rmse = torch.sqrt(F.mse_loss(pred_energy, target_energy, reduction="mean"))
            mae = F.l1_loss(pred_energy, target_energy, reduction="mean")

        return {
            "total": loss,
            "energy": loss,
            "energy_rmse": rmse,
            "energy_mae": mae,
        }


class ForceLoss(nn.Module):
    """
    Force-only loss function.

    Critical for MD stability. Should be the primary training objective
    for models used in molecular dynamics simulations.
    """

    def __init__(
        self,
        loss_type: Literal["mse", "mae", "huber"] = "mse",
        huber_delta: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.reduction = reduction

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute force loss."""
        pred_forces = predictions["forces"]
        target_forces = targets["forces"]

        if mask is not None and "forces" in mask:
            pred_forces = pred_forces[mask["forces"]]
            target_forces = target_forces[mask["forces"]]

        if self.loss_type == "mse":
            loss = F.mse_loss(pred_forces, target_forces, reduction=self.reduction)
        elif self.loss_type == "mae":
            loss = F.l1_loss(pred_forces, target_forces, reduction=self.reduction)
        elif self.loss_type == "huber":
            loss = F.huber_loss(
                pred_forces, target_forces, delta=self.huber_delta, reduction=self.reduction
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Metrics (critical for MD performance assessment)
        with torch.no_grad():
            rmse = torch.sqrt(F.mse_loss(pred_forces, target_forces, reduction="mean"))
            mae = F.l1_loss(pred_forces, target_forces, reduction="mean")

        result = {
            "total": loss,
            "force": loss,
            "force_rmse": rmse,
            "force_mae": mae,
        }

        # Component-wise metrics (only if forces have 3 components in last dim)
        with torch.no_grad():
            if pred_forces.dim() >= 2 and pred_forces.shape[-1] == 3:
                # Compute MSE per component (x, y, z) by flattening batch and atom dims
                diff_sq = (pred_forces - target_forces) ** 2
                # Mean over all but last dimension to get per-component MSE
                force_mse_x = diff_sq[..., 0].mean()
                force_mse_y = diff_sq[..., 1].mean()
                force_mse_z = diff_sq[..., 2].mean()
                result["force_rmse_x"] = torch.sqrt(force_mse_x)
                result["force_rmse_y"] = torch.sqrt(force_mse_y)
                result["force_rmse_z"] = torch.sqrt(force_mse_z)

        return result


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining soft targets from teacher and hard targets.

    This will be used in later milestones (M4) for actual distillation training.
    Currently implements basic temperature-scaled KL divergence framework.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        temperature: float = 1.0,
        alpha: float = 0.5,
    ):
        """
        Initialize distillation loss.

        Args:
            base_loss: Base loss function (e.g., ForceFieldLoss)
            temperature: Temperature for soft target scaling
            alpha: Weight for distillation loss (1-alpha for hard target loss)
        """
        super().__init__()
        self.base_loss = base_loss
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        hard_targets: Dict[str, torch.Tensor],
        soft_targets: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.

        Args:
            predictions: Model predictions
            hard_targets: Ground truth targets
            soft_targets: Teacher model predictions (optional)
            mask: Optional masks

        Returns:
            Dictionary of losses
        """
        # Hard target loss
        hard_loss_dict = self.base_loss(predictions, hard_targets, mask)
        hard_loss = hard_loss_dict["total"]

        if soft_targets is None or self.alpha == 0:
            # No distillation, just return hard loss
            return hard_loss_dict

        # Soft target loss (simple MSE for now, can be extended)
        soft_loss = 0.0

        if "energy" in predictions and "energy" in soft_targets:
            soft_loss = soft_loss + F.mse_loss(
                predictions["energy"] / self.temperature,
                soft_targets["energy"] / self.temperature,
            )

        if "forces" in predictions and "forces" in soft_targets:
            soft_loss = soft_loss + F.mse_loss(
                predictions["forces"] / self.temperature,
                soft_targets["forces"] / self.temperature,
            )

        # Combine losses
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        result = hard_loss_dict.copy()
        result["total"] = total_loss
        result["distill_loss"] = soft_loss
        result["hard_loss"] = hard_loss

        return result


def check_loss_numerical_stability(loss_dict: Dict[str, torch.Tensor]) -> bool:
    """
    Check if loss values are numerically stable (no NaN or Inf).

    Args:
        loss_dict: Dictionary of loss tensors

    Returns:
        True if all losses are stable, False otherwise
    """
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any() or torch.isinf(value).any():
                return False
    return True


def get_loss_function(
    loss_type: str,
    energy_weight: float = 1.0,
    force_weight: float = 100.0,
    stress_weight: float = 0.1,
    angular_weight: float = 10.0,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss ('force_field', 'energy', 'force', 'distillation')
        energy_weight: Weight for energy component
        force_weight: Weight for force component
        stress_weight: Weight for stress component
        angular_weight: Weight for angular (directional) component
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Loss function module
    """
    if loss_type == "force_field":
        return ForceFieldLoss(
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            angular_weight=angular_weight,
            **kwargs,
        )
    elif loss_type == "energy":
        return EnergyLoss(**kwargs)
    elif loss_type == "force":
        return ForceLoss(**kwargs)
    elif loss_type == "distillation":
        base_loss = ForceFieldLoss(
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            angular_weight=angular_weight,
        )
        return DistillationLoss(base_loss=base_loss, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
