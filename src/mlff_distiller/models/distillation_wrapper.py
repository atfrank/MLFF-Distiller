#!/usr/bin/env python3
"""
Model Wrapper for Distillation Training

Wraps the StudentForceField model to provide the interface expected by
the Trainer class, handling forward pass and force computation.

Author: ML Distillation Project
Date: 2025-11-24
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class DistillationWrapper(nn.Module):
    """
    Wrapper for StudentForceField to handle training interface.

    The Trainer expects models to:
    1. Accept a batch dictionary
    2. Return predictions dictionary with 'energy' and 'forces'
    3. Handle force computation via autograd

    This wrapper adapts StudentForceField's forward signature:
        forward(atomic_numbers, positions, cell, pbc, batch) → energy

    To the Trainer's expected interface:
        forward(batch_dict) → {'energy': ..., 'forces': ...}

    Args:
        model: StudentForceField instance

    Example:
        >>> from mlff_distiller.models.student_model import StudentForceField
        >>> student = StudentForceField(num_interactions=3, hidden_dim=128)
        >>> wrapped_model = DistillationWrapper(student)
        >>>
        >>> # Use with Trainer
        >>> trainer = Trainer(model=wrapped_model, ...)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for distillation training.

        Args:
            batch: Dictionary from distillation_collate_fn containing:
                - 'atomic_numbers': [N_atoms_batch] int64
                - 'positions': [N_atoms_batch, 3] float, requires_grad
                - 'cell': [batch_size, 3, 3] float
                - 'pbc': [batch_size, 3] bool
                - 'batch': [N_atoms_batch] int64 - structure indices
                - 'n_atoms': [batch_size] int64
                - 'atom_splits': [batch_size+1] int64

        Returns:
            Dictionary with:
                - 'energy': [batch_size] - predicted energies
                - 'forces': [N_atoms_batch, 3] - predicted forces
        """
        # Clone positions and enable gradients for force computation
        # Clone is needed to create a new leaf tensor that can require gradients
        positions = batch['positions'].clone().detach().requires_grad_(True)

        # Enable gradient tracking even during eval (needed for force computation)
        # We'll use torch.set_grad_enabled to ensure gradients flow through forward pass
        with torch.set_grad_enabled(True):
            # Forward pass through student model
            energy = self.model(
                atomic_numbers=batch['atomic_numbers'],
                positions=positions,
                cell=batch['cell'],
                pbc=batch['pbc'],
                batch=batch['batch']
            )

            # Compute forces via automatic differentiation
            # Forces = -∇E with respect to positions
            # create_graph=True allows gradients to flow back through force computation
            forces = -torch.autograd.grad(
                outputs=energy.sum(),  # Scalar for autograd
                inputs=positions,
                create_graph=self.training,  # Only need graph during training
                retain_graph=self.training,   # Only retain during training
            )[0]

        return {
            'energy': energy,
            'forces': forces,
        }

    def num_parameters(self) -> int:
        """Get number of trainable parameters."""
        if hasattr(self.model, 'num_parameters'):
            return self.model.num_parameters()
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def create_wrapped_student(
    num_interactions: int = 3,
    hidden_dim: int = 128,
    num_rbf: int = 20,
    cutoff: float = 5.0,
    **kwargs
) -> DistillationWrapper:
    """
    Factory function to create wrapped student model.

    Args:
        num_interactions: Number of PaiNN interaction blocks
        hidden_dim: Hidden dimension size
        num_rbf: Number of radial basis functions
        cutoff: Cutoff radius in Angstroms
        **kwargs: Additional arguments for StudentForceField

    Returns:
        Wrapped student model ready for training

    Example:
        >>> model = create_wrapped_student(num_interactions=3, hidden_dim=128)
        >>> trainer = Trainer(model=model, ...)
    """
    from .student_model import StudentForceField

    student = StudentForceField(
        num_interactions=num_interactions,
        hidden_dim=hidden_dim,
        num_rbf=num_rbf,
        cutoff=cutoff,
        **kwargs
    )

    wrapped = DistillationWrapper(student)
    return wrapped
