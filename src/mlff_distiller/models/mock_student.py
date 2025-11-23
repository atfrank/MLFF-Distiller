"""
Mock Student Model for Testing

This module provides simple mock student models for testing the StudentCalculator
without requiring trained models. These models produce deterministic outputs based
on simple formulas, allowing unit tests to validate the calculator interface.

NOT FOR PRODUCTION USE - Only for testing and development.

Author: ML Architecture Designer
Date: 2025-11-23
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class MockStudentModel(nn.Module):
    """
    Mock student model that produces deterministic outputs for testing.

    This model doesn't perform real ML predictions. Instead, it computes simple
    deterministic outputs based on atomic positions and numbers, allowing tests
    to validate the StudentCalculator interface without requiring trained models.

    The outputs are physically meaningless but numerically consistent, making
    them suitable for interface testing, unit tests, and development.

    Args:
        hidden_dim: Hidden dimension (unused, for interface compatibility)
        predict_stress: If True, include stress in outputs
        energy_scale: Scaling factor for energy output
        force_scale: Scaling factor for force output

    Example:
        >>> model = MockStudentModel()
        >>> input_data = {
        ...     "positions": torch.randn(10, 3),
        ...     "atomic_numbers": torch.randint(1, 10, (10,)),
        ...     "batch": torch.zeros(10, dtype=torch.long)
        ... }
        >>> output = model(input_data)
        >>> print(output.keys())
        dict_keys(['energy', 'forces', 'stress'])
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        predict_stress: bool = True,
        energy_scale: float = 1.0,
        force_scale: float = 0.1,
    ):
        """Initialize mock student model."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.predict_stress = predict_stress
        self.energy_scale = energy_scale
        self.force_scale = force_scale

        # Dummy parameters (to make it a proper nn.Module)
        self.dummy_linear = nn.Linear(1, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing mock predictions.

        Args:
            batch: Dictionary with keys:
                - positions: (n_atoms, 3) positions in Angstroms
                - atomic_numbers: (n_atoms,) atomic numbers
                - batch: (n_atoms,) batch indices
                - cell: (3, 3) optional cell matrix
                - pbc: (3,) optional periodic boundary conditions

        Returns:
            Dictionary with keys:
                - energy: (1,) or scalar energy in eV
                - forces: (n_atoms, 3) forces in eV/Angstrom
                - stress: (6,) stress in eV/Angstrom^3 (if predict_stress=True)
        """
        positions = batch["positions"]
        atomic_numbers = batch["atomic_numbers"]
        batch_idx = batch["batch"]

        n_atoms = positions.shape[0]
        device = positions.device

        # Mock energy: sum of distances from origin, weighted by atomic number
        distances = torch.norm(positions, dim=1)
        atomic_weights = atomic_numbers.float()
        energy_per_atom = distances * atomic_weights * self.energy_scale

        # Sum over atoms in each structure
        n_structures = batch_idx.max().item() + 1
        energies = torch.zeros(n_structures, device=device)
        for i in range(n_structures):
            mask = batch_idx == i
            energies[i] = energy_per_atom[mask].sum()

        # Mock forces: negative gradient-like (point toward origin)
        # F = -k * r (simple harmonic oscillator-like)
        forces = -positions * self.force_scale
        # Weight by atomic number
        forces = forces * atomic_weights.unsqueeze(-1)

        # Prepare output
        output = {
            "energy": energies,
            "forces": forces,
        }

        # Mock stress: simple function of cell volume if available
        if self.predict_stress:
            if "cell" in batch and batch["cell"] is not None:
                cell = batch["cell"]
                # Compute cell volume
                volume = torch.abs(torch.det(cell))
                # Mock stress: isotropic pressure based on volume
                stress_value = 0.01 / (volume + 1.0)  # Avoid division by zero
                stress = torch.ones(6, device=device) * stress_value
            else:
                # No cell provided, return small stress
                stress = torch.ones(6, device=device) * 0.001

            output["stress"] = stress

        return output


class SimpleMLP(nn.Module):
    """
    Simple MLP-based student model for testing.

    A minimal but functional neural network that can be used to test
    the full training and inference pipeline. Uses a simple MLP to
    predict per-atom energies from positions and atomic numbers.

    Args:
        input_dim: Input feature dimension per atom
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        predict_stress: If True, predict stress tensor

    Example:
        >>> model = SimpleMLP(hidden_dim=64, num_layers=2)
        >>> model.eval()
        >>> input_data = {
        ...     "positions": torch.randn(10, 3),
        ...     "atomic_numbers": torch.randint(1, 10, (10,)),
        ...     "batch": torch.zeros(10, dtype=torch.long)
        ... }
        >>> output = model(input_data)
    """

    def __init__(
        self,
        input_dim: int = 4,  # 3 (positions) + 1 (atomic number)
        hidden_dim: int = 128,
        num_layers: int = 3,
        predict_stress: bool = True,
    ):
        """Initialize SimpleMLP model."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_stress = predict_stress

        # Build MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        self.mlp = nn.Sequential(*layers)

        # Output heads
        self.energy_head = nn.Linear(hidden_dim, 1)

        if predict_stress:
            self.stress_head = nn.Linear(hidden_dim, 6)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MLP.

        Args:
            batch: Dictionary with positions, atomic_numbers, batch, etc.

        Returns:
            Dictionary with energy, forces, and optionally stress.
        """
        positions = batch["positions"]
        atomic_numbers = batch["atomic_numbers"]
        batch_idx = batch["batch"]

        # Prepare input features: [x, y, z, atomic_number]
        features = torch.cat(
            [positions, atomic_numbers.float().unsqueeze(-1)], dim=-1
        )

        # Forward through MLP
        hidden = self.mlp(features)

        # Predict per-atom energies
        atom_energies = self.energy_head(hidden).squeeze(-1)

        # Sum to get total energy per structure
        n_structures = batch_idx.max().item() + 1
        device = positions.device
        energies = torch.zeros(n_structures, device=device)
        for i in range(n_structures):
            mask = batch_idx == i
            energies[i] = atom_energies[mask].sum()

        # Enable gradient computation for forces
        if positions.requires_grad:
            # Forces are negative gradient of energy w.r.t. positions
            forces = -torch.autograd.grad(
                energies.sum(),
                positions,
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            # If positions don't require grad, return zeros
            forces = torch.zeros_like(positions)

        # Prepare output
        output = {
            "energy": energies,
            "forces": forces,
        }

        # Predict stress if requested
        if self.predict_stress:
            # Use mean of hidden features for stress prediction
            stress_features = hidden.mean(dim=0)
            stress = self.stress_head(stress_features)
            output["stress"] = stress

        return output


# Convenience exports
__all__ = ["MockStudentModel", "SimpleMLP"]
