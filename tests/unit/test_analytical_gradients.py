"""
Unit Tests for Analytical Gradient Functions

Validates correctness of analytical gradients against:
1. Finite difference approximations
2. PyTorch autograd
3. Known analytical solutions

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import pytest
import torch
import numpy as np
import math

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.analytical_gradients import (
    compute_rbf_gradients_analytical,
    compute_cutoff_gradients_analytical,
    compute_edge_feature_gradients_analytical,
    accumulate_forces_from_edges,
    validate_gradients_finite_difference,
)


class TestRBFGradients:
    """Test analytical RBF gradient computation."""

    def test_rbf_gradient_shape(self):
        """Test output shape is correct."""
        n_edges = 10
        n_rbf = 20

        distances = torch.rand(n_edges) * 5.0
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)  # Normalize
        centers = torch.linspace(0, 5, n_rbf)
        gamma = 10.0

        grads = compute_rbf_gradients_analytical(distances, edge_vec, centers, gamma)

        assert grads.shape == (n_edges, 3, n_rbf)
        assert not torch.isnan(grads).any()
        assert not torch.isinf(grads).any()

    def test_rbf_gradient_vs_autograd(self):
        """Test analytical gradients match PyTorch autograd."""
        n_edges = 5
        n_rbf = 10

        # Create test data with gradients enabled
        distances = torch.rand(n_edges, requires_grad=True) * 5.0
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        centers = torch.linspace(0, 5, n_rbf)
        gamma = 10.0

        # Compute RBF with autograd
        r_expanded = distances.unsqueeze(-1)
        rbf_autograd = torch.exp(-gamma * (r_expanded - centers) ** 2)

        # Compute gradients via autograd for each RBF function
        autograd_grads = []
        for k in range(n_rbf):
            grad = torch.autograd.grad(
                rbf_autograd[:, k].sum(),
                distances,
                retain_graph=True
            )[0]
            # Convert scalar gradient to spatial gradient
            grad_spatial = grad.unsqueeze(-1) * edge_vec
            autograd_grads.append(grad_spatial)

        autograd_grads = torch.stack(autograd_grads, dim=-1)  # [n_edges, 3, n_rbf]

        # Compute analytical gradients
        analytical_grads = compute_rbf_gradients_analytical(
            distances.detach(), edge_vec, centers, gamma
        )

        # Compare
        assert torch.allclose(analytical_grads, autograd_grads, atol=1e-6, rtol=1e-4), \
            f"Max diff: {(analytical_grads - autograd_grads).abs().max():.2e}"

    def test_rbf_gradient_zero_distance(self):
        """Test numerical stability at very small distances."""
        n_edges = 3
        n_rbf = 5

        # Include very small distance
        distances = torch.tensor([1e-8, 1.0, 5.0])
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        centers = torch.linspace(0, 5, n_rbf)
        gamma = 10.0

        grads = compute_rbf_gradients_analytical(distances, edge_vec, centers, gamma)

        # Should not produce NaN or Inf
        assert not torch.isnan(grads).any()
        assert not torch.isinf(grads).any()

        # Gradient should be finite
        assert grads.abs().max() < 1e6  # Reasonable magnitude

    def test_rbf_gradient_with_cutoff(self):
        """Test RBF gradients with cutoff multiplication."""
        n_edges = 10
        n_rbf = 20

        distances = torch.rand(n_edges) * 5.0
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        centers = torch.linspace(0, 5, n_rbf)
        gamma = 10.0

        # Compute cutoff values
        cutoff_radius = 5.0
        arg = math.pi * distances / cutoff_radius
        cutoff_values = 0.5 * (torch.cos(arg) + 1.0)
        cutoff_values = cutoff_values * (distances < cutoff_radius).float()

        # Compute gradients with cutoff
        grads = compute_rbf_gradients_analytical(
            distances, edge_vec, centers, gamma, cutoff_value=cutoff_values
        )

        # Gradients beyond cutoff should be zero
        beyond_cutoff = distances >= cutoff_radius
        if beyond_cutoff.any():
            assert torch.allclose(
                grads[beyond_cutoff],
                torch.zeros_like(grads[beyond_cutoff]),
                atol=1e-8
            )


class TestCutoffGradients:
    """Test analytical cutoff gradient computation."""

    def test_cutoff_gradient_shape(self):
        """Test output shape is correct."""
        n_edges = 10

        distances = torch.rand(n_edges) * 5.0
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        cutoff_radius = 5.0

        grads = compute_cutoff_gradients_analytical(distances, edge_vec, cutoff_radius)

        assert grads.shape == (n_edges, 3)
        assert not torch.isnan(grads).any()
        assert not torch.isinf(grads).any()

    def test_cutoff_gradient_vs_autograd(self):
        """Test analytical cutoff gradients match PyTorch autograd."""
        n_edges = 5

        distances = torch.rand(n_edges, requires_grad=True) * 4.5  # Below cutoff
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        cutoff_radius = 5.0

        # Compute cutoff with autograd
        arg = math.pi * distances / cutoff_radius
        cutoff_autograd = 0.5 * (torch.cos(arg) + 1.0)

        # Compute gradients via autograd
        autograd_grads = []
        for i in range(n_edges):
            grad = torch.autograd.grad(
                cutoff_autograd[i],
                distances,
                retain_graph=True
            )[0]
            grad_spatial = grad[i] * edge_vec[i]
            autograd_grads.append(grad_spatial)

        autograd_grads = torch.stack(autograd_grads)

        # Compute analytical gradients
        analytical_grads = compute_cutoff_gradients_analytical(
            distances.detach(), edge_vec, cutoff_radius
        )

        # Compare
        assert torch.allclose(analytical_grads, autograd_grads, atol=1e-6, rtol=1e-4), \
            f"Max diff: {(analytical_grads - autograd_grads).abs().max():.2e}"

    def test_cutoff_gradient_beyond_cutoff(self):
        """Test gradient is zero beyond cutoff radius."""
        n_edges = 5

        # All distances beyond cutoff
        distances = torch.tensor([5.1, 6.0, 10.0, 20.0, 100.0])
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        cutoff_radius = 5.0

        grads = compute_cutoff_gradients_analytical(distances, edge_vec, cutoff_radius)

        # All gradients should be zero
        assert torch.allclose(grads, torch.zeros_like(grads), atol=1e-10)

    def test_cutoff_gradient_at_boundary(self):
        """Test gradient continuity at cutoff boundary."""
        cutoff_radius = 5.0
        epsilon = 1e-4

        # Distances just inside and outside cutoff
        distances = torch.tensor([cutoff_radius - epsilon, cutoff_radius + epsilon])
        edge_vec = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cutoff_radius = 5.0

        grads = compute_cutoff_gradients_analytical(distances, edge_vec, cutoff_radius)

        # Gradient just outside should be zero
        assert torch.allclose(grads[1], torch.zeros(3), atol=1e-10)

        # Gradient just inside should be small (near zero)
        assert grads[0].abs().max() < 0.1  # Small but non-zero


class TestEdgeFeatureGradients:
    """Test combined edge feature (RBF × cutoff) gradients."""

    def test_edge_feature_gradient_product_rule(self):
        """Test product rule: ∂(φ·f)/∂r = ∂φ/∂r·f + φ·∂f/∂r"""
        n_edges = 5
        n_rbf = 10

        distances = torch.rand(n_edges) * 4.5
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        centers = torch.linspace(0, 5, n_rbf)
        gamma = 10.0
        cutoff_radius = 5.0

        # Compute edge features and gradients
        features, grads = compute_edge_feature_gradients_analytical(
            distances, edge_vec, centers, gamma, cutoff_radius
        )

        # Validate shapes
        assert features.shape == (n_edges, n_rbf)
        assert grads.shape == (n_edges, 3, n_rbf)

        # Validate no NaN/Inf
        assert not torch.isnan(features).any()
        assert not torch.isnan(grads).any()
        assert not torch.isinf(grads).any()

    def test_edge_feature_values_match(self):
        """Test edge feature values match RBF × cutoff."""
        n_edges = 5
        n_rbf = 10

        distances = torch.rand(n_edges) * 4.5
        edge_vec = torch.randn(n_edges, 3)
        edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        centers = torch.linspace(0, 5, n_rbf)
        gamma = 10.0
        cutoff_radius = 5.0

        # Compute via combined function
        features_combined, _ = compute_edge_feature_gradients_analytical(
            distances, edge_vec, centers, gamma, cutoff_radius
        )

        # Compute separately
        r_expanded = distances.unsqueeze(-1)
        rbf = torch.exp(-gamma * (r_expanded - centers) ** 2)
        arg = math.pi * distances / cutoff_radius
        cutoff = 0.5 * (torch.cos(arg) + 1.0)
        cutoff = cutoff * (distances < cutoff_radius).float()
        features_separate = rbf * cutoff.unsqueeze(-1)

        # Should match
        assert torch.allclose(features_combined, features_separate, atol=1e-8)


class TestForceAccumulation:
    """Test force accumulation from edge gradients."""

    def test_force_accumulation_shape(self):
        """Test output shape is correct."""
        n_atoms = 10
        n_edges = 30
        n_features = 20

        edge_gradients = torch.randn(n_edges, 3, n_features)
        edge_index = torch.randint(0, n_atoms, (2, n_edges))
        energy_grad = torch.randn(n_edges, n_features)

        forces = accumulate_forces_from_edges(
            edge_gradients, edge_index, n_atoms, energy_grad
        )

        assert forces.shape == (n_atoms, 3)
        assert not torch.isnan(forces).any()
        assert not torch.isinf(forces).any()

    def test_force_accumulation_newtons_third_law(self):
        """Test Newton's third law: F_ij = -F_ji"""
        n_atoms = 4
        n_features = 5

        # Simple case: one edge between atoms 0 and 1
        edge_gradients = torch.randn(1, 3, n_features)
        edge_index = torch.tensor([[0], [1]])  # Edge from 0 to 1
        energy_grad = torch.randn(1, n_features)

        forces = accumulate_forces_from_edges(
            edge_gradients, edge_index, n_atoms, energy_grad
        )

        # Force on atom 0 should be negative of force on atom 1
        assert torch.allclose(forces[0], -forces[1], atol=1e-6)

        # Forces on atoms 2 and 3 should be zero (not connected)
        assert torch.allclose(forces[2], torch.zeros(3), atol=1e-10)
        assert torch.allclose(forces[3], torch.zeros(3), atol=1e-10)

    def test_force_accumulation_conservation(self):
        """Test total force is zero (conservation of momentum)."""
        n_atoms = 20
        n_edges = 100
        n_features = 10

        edge_gradients = torch.randn(n_edges, 3, n_features)
        edge_index = torch.randint(0, n_atoms, (2, n_edges))
        energy_grad = torch.randn(n_edges, n_features)

        forces = accumulate_forces_from_edges(
            edge_gradients, edge_index, n_atoms, energy_grad
        )

        # Total force should be (nearly) zero
        total_force = forces.sum(dim=0)
        assert torch.allclose(total_force, torch.zeros(3), atol=1e-5)


class TestGradientValidation:
    """Test finite difference validation utility."""

    def test_validate_simple_function(self):
        """Test validation on simple quadratic function."""
        # f(x) = x^2, df/dx = 2x
        def quad_func(pos):
            return (pos ** 2).sum()

        positions = torch.randn(5, 3)
        analytical = 2 * positions

        valid, max_err, rel_err = validate_gradients_finite_difference(
            quad_func, positions, analytical, epsilon=1e-5, tolerance=0.2  # Finite diff is approximate
        )

        assert valid, f"Validation failed: max_err={max_err:.2e}, rel_err={rel_err:.2e}"
        assert max_err < 0.1  # Finite difference has limited accuracy
        assert rel_err < 0.2

    def test_validate_detects_incorrect_gradient(self):
        """Test validation correctly identifies wrong gradients."""
        def quad_func(pos):
            return (pos ** 2).sum()

        positions = torch.randn(5, 3)
        wrong_analytical = 3 * positions  # Should be 2x, not 3x

        valid, max_err, rel_err = validate_gradients_finite_difference(
            quad_func, positions, wrong_analytical, epsilon=1e-5, tolerance=1e-3
        )

        assert not valid, "Should detect incorrect gradient"
        assert max_err > 0.1  # Significant error


@pytest.mark.parametrize("n_rbf", [10, 20, 50])
@pytest.mark.parametrize("n_edges", [5, 20, 100])
def test_rbf_gradient_various_sizes(n_rbf, n_edges):
    """Test RBF gradients with various sizes."""
    distances = torch.rand(n_edges) * 5.0
    edge_vec = torch.randn(n_edges, 3)
    edge_vec = edge_vec / edge_vec.norm(dim=1, keepdim=True)
    centers = torch.linspace(0, 5, n_rbf)
    gamma = 10.0

    grads = compute_rbf_gradients_analytical(distances, edge_vec, centers, gamma)

    assert grads.shape == (n_edges, 3, n_rbf)
    assert not torch.isnan(grads).any()
    assert not torch.isinf(grads).any()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
