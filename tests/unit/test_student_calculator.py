"""
Unit tests for StudentCalculator.

Tests the StudentCalculator class to ensure:
1. Proper ASE Calculator interface implementation
2. Correct initialization with various model types
3. Property calculations (energy, forces, stress)
4. Device handling (CPU/CUDA)
5. Drop-in replacement compatibility with teacher calculators
6. Buffer reuse and memory efficiency

Author: ML Architecture Designer
Date: 2025-11-23
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator

from mlff_distiller.models.student_calculator import StudentCalculator
from mlff_distiller.models.mock_student import MockStudentModel, SimpleMLP


class TestStudentCalculatorInterface(unittest.TestCase):
    """Test StudentCalculator ASE interface compliance."""

    def setUp(self):
        """Set up mock student model for testing."""
        self.model = MockStudentModel()
        self.calc = StudentCalculator(model=self.model, device="cpu")

    def test_inherits_from_calculator(self):
        """Test that StudentCalculator inherits from ASE Calculator."""
        self.assertIsInstance(self.calc, Calculator)

    def test_implemented_properties(self):
        """Test that required properties are declared as implemented."""
        self.assertIn("energy", self.calc.implemented_properties)
        self.assertIn("forces", self.calc.implemented_properties)
        self.assertIn("stress", self.calc.implemented_properties)

    def test_has_calculate_method(self):
        """Test that calculate method exists and is callable."""
        self.assertTrue(hasattr(self.calc, "calculate"))
        self.assertTrue(callable(self.calc.calculate))

    def test_has_standard_ase_methods(self):
        """Test that standard ASE Calculator methods exist."""
        self.assertTrue(hasattr(self.calc, "get_potential_energy"))
        self.assertTrue(hasattr(self.calc, "get_forces"))
        self.assertTrue(hasattr(self.calc, "get_stress"))
        self.assertTrue(hasattr(self.calc, "reset"))


class TestStudentCalculatorInitialization(unittest.TestCase):
    """Test StudentCalculator initialization with different configurations."""

    def test_initialization_with_model_instance(self):
        """Test initialization with pre-created model instance."""
        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cpu")

        self.assertIsNotNone(calc.model)
        self.assertEqual(calc.device, torch.device("cpu"))

    def test_initialization_with_model_factory(self):
        """Test initialization with model factory function."""
        def model_factory(hidden_dim=128):
            return MockStudentModel(hidden_dim=hidden_dim)

        calc = StudentCalculator(
            model=model_factory,
            model_config={"hidden_dim": 64},
            device="cpu"
        )

        self.assertIsNotNone(calc.model)

    def test_initialization_requires_model_or_path(self):
        """Test that initialization requires either model or model_path."""
        with self.assertRaises(ValueError) as context:
            StudentCalculator(device="cpu")

        self.assertIn("model' or 'model_path'", str(context.exception))

    def test_device_placement_cpu(self):
        """Test that model is placed on CPU when requested."""
        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cpu")

        # Check that model parameters are on CPU
        for param in calc.model.parameters():
            self.assertEqual(param.device.type, "cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement_cuda(self):
        """Test that model is placed on CUDA when requested."""
        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cuda")

        # Check that model parameters are on CUDA
        for param in calc.model.parameters():
            self.assertEqual(param.device.type, "cuda")

    def test_dtype_setting(self):
        """Test that dtype is set correctly."""
        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cpu", dtype=torch.float64)

        self.assertEqual(calc.dtype, torch.float64)


class TestStudentCalculatorCalculations(unittest.TestCase):
    """Test StudentCalculator property calculations."""

    def setUp(self):
        """Set up calculator with mock model and test atoms."""
        self.model = MockStudentModel()
        self.calc = StudentCalculator(model=self.model, device="cpu")
        self.atoms = molecule("H2O")

    def test_calculate_populates_results(self):
        """Test that calculate method populates results."""
        self.atoms.calc = self.calc
        self.calc.calculate(self.atoms, properties=["energy", "forces", "stress"])

        self.assertIn("energy", self.calc.results)
        self.assertIn("forces", self.calc.results)
        self.assertIn("stress", self.calc.results)

    def test_energy_calculation(self):
        """Test energy calculation returns correct type and shape."""
        self.atoms.calc = self.calc
        energy = self.atoms.get_potential_energy()

        self.assertIsInstance(energy, (float, np.floating))

    def test_forces_calculation(self):
        """Test forces calculation returns correct type and shape."""
        self.atoms.calc = self.calc
        forces = self.atoms.get_forces()

        self.assertIsInstance(forces, np.ndarray)
        self.assertEqual(forces.shape, (len(self.atoms), 3))

    def test_stress_calculation(self):
        """Test stress calculation returns correct type and shape."""
        # Use periodic system for stress
        atoms_periodic = bulk("Si", "diamond", a=5.43)
        atoms_periodic.calc = self.calc

        stress = atoms_periodic.get_stress()

        self.assertIsInstance(stress, np.ndarray)
        self.assertEqual(len(stress), 6)  # Voigt notation

    def test_calculate_multiple_times(self):
        """Test that calculator can be called multiple times."""
        self.atoms.calc = self.calc

        # First call
        energy1 = self.atoms.get_potential_energy()
        forces1 = self.atoms.get_forces()

        # Second call (should use caching)
        energy2 = self.atoms.get_potential_energy()
        forces2 = self.atoms.get_forces()

        # Results should be identical (caching)
        self.assertEqual(energy1, energy2)
        np.testing.assert_array_equal(forces1, forces2)

        # Modify atoms to invalidate cache
        self.atoms.positions[0] += 0.1

        # Third call (should recalculate)
        energy3 = self.atoms.get_potential_energy()

        # Result should be different now
        self.assertNotEqual(energy1, energy3)

    def test_buffer_reuse(self):
        """Test that buffers are reused for efficiency."""
        self.atoms.calc = self.calc

        # First calculation
        _ = self.atoms.get_potential_energy()
        buffer1_id = id(self.calc._position_buffer)

        # Modify positions (same number of atoms)
        self.atoms.positions[0] += 0.1

        # Second calculation (should reuse buffer)
        _ = self.atoms.get_potential_energy()
        buffer2_id = id(self.calc._position_buffer)

        # Buffer should be reused (same id)
        self.assertEqual(buffer1_id, buffer2_id)

    def test_different_system_sizes(self):
        """Test calculator handles different system sizes."""
        self.calc = StudentCalculator(model=MockStudentModel(), device="cpu")

        # Small molecule
        atoms_small = molecule("H2")
        atoms_small.calc = self.calc
        energy_small = atoms_small.get_potential_energy()
        self.assertIsInstance(energy_small, (float, np.floating))

        # Larger molecule
        atoms_large = molecule("CH3CH2OH")
        atoms_large.calc = self.calc
        energy_large = atoms_large.get_potential_energy()
        self.assertIsInstance(energy_large, (float, np.floating))

    def test_periodic_vs_nonperiodic(self):
        """Test calculator handles both periodic and non-periodic systems."""
        self.calc = StudentCalculator(model=MockStudentModel(), device="cpu")

        # Non-periodic (molecule)
        mol = molecule("H2O")
        mol.calc = self.calc
        energy_mol = mol.get_potential_energy()
        self.assertIsInstance(energy_mol, (float, np.floating))

        # Periodic (crystal)
        crystal = bulk("Si", "diamond", a=5.43)
        crystal.calc = self.calc
        energy_crystal = crystal.get_potential_energy()
        self.assertIsInstance(energy_crystal, (float, np.floating))


class TestStudentCalculatorCheckpointLoading(unittest.TestCase):
    """Test StudentCalculator checkpoint loading functionality."""

    def test_load_nonexistent_checkpoint_fails(self):
        """Test that loading from non-existent checkpoint fails gracefully."""
        fake_path = Path("/nonexistent/path/model.pth")

        with self.assertRaises(FileNotFoundError):
            StudentCalculator(model_path=fake_path, device="cpu")

    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoint."""
        # Create and save a model
        model = MockStudentModel(hidden_dim=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_model.pth"

            # Save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_class": MockStudentModel,
                "model_config": {"hidden_dim": 64},
            }
            torch.save(checkpoint, checkpoint_path)

            # Load checkpoint
            calc = StudentCalculator(
                model_path=checkpoint_path,
                device="cpu"
            )

            # Verify model loaded correctly
            self.assertIsNotNone(calc.model)

            # Test that loaded model works
            atoms = molecule("H2O")
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            self.assertIsInstance(energy, (float, np.floating))


class TestStudentCalculatorDropInCompatibility(unittest.TestCase):
    """Test drop-in replacement compatibility with teacher calculators."""

    def test_same_interface_as_calculator(self):
        """Test StudentCalculator has same public interface as ASE Calculator."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")

        # Check for essential ASE Calculator methods
        essential_methods = [
            "calculate",
            "get_potential_energy",
            "get_forces",
            "get_stress",
            "reset",
        ]

        for method_name in essential_methods:
            self.assertTrue(
                hasattr(calc, method_name),
                f"Missing essential method: {method_name}"
            )

    def test_can_attach_to_atoms(self):
        """Test that StudentCalculator can be attached to Atoms object."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = molecule("H2O")

        # Attach calculator
        atoms.calc = calc

        # Verify attachment
        self.assertIs(atoms.calc, calc)

    def test_works_with_ase_atoms_methods(self):
        """Test that calculator works with standard ASE Atoms methods."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = molecule("H2O")
        atoms.calc = calc

        # These should all work without errors
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        self.assertIsInstance(energy, (float, np.floating))
        self.assertIsInstance(forces, np.ndarray)

    def test_initialization_signature_compatible(self):
        """Test initialization signature is compatible with teacher calculators."""
        # Should accept similar parameters to teacher calculators
        calc = StudentCalculator(
            model=MockStudentModel(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32,
        )

        self.assertIsNotNone(calc)


class TestStudentCalculatorPerformance(unittest.TestCase):
    """Test StudentCalculator performance characteristics."""

    def test_n_calls_tracking(self):
        """Test that number of calculate calls is tracked."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = molecule("H2O")
        atoms.calc = calc

        # Initially zero calls
        self.assertEqual(calc.n_calls, 0)

        # Call once
        _ = atoms.get_potential_energy()
        self.assertEqual(calc.n_calls, 1)

        # Modify and call again
        atoms.positions[0] += 0.1
        _ = atoms.get_potential_energy()
        self.assertEqual(calc.n_calls, 2)

    def test_no_memory_leak_over_many_calls(self):
        """Test that repeated calls don't leak memory."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = molecule("H2O")
        atoms.calc = calc

        # Make many calls
        for i in range(100):
            atoms.positions[0, 0] += 0.001  # Small perturbation
            _ = atoms.get_potential_energy()

        # If we got here without OOM, test passes
        self.assertEqual(calc.n_calls, 100)


class TestStudentCalculatorEdgeCases(unittest.TestCase):
    """Test StudentCalculator edge cases and error handling."""

    def test_single_atom_system(self):
        """Test calculator handles single atom."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = Atoms("H", positions=[[0, 0, 0]])
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        self.assertIsInstance(energy, (float, np.floating))
        self.assertEqual(forces.shape, (1, 3))

    def test_large_system(self):
        """Test calculator handles large system."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")

        # Create larger system (64 atoms)
        atoms = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        self.assertIsInstance(energy, (float, np.floating))
        self.assertEqual(forces.shape, (len(atoms), 3))

    def test_missing_output_key_raises_error(self):
        """Test that missing output key raises appropriate error."""
        # Create model that doesn't output forces
        class BadModel(nn.Module):
            def forward(self, batch):
                return {"energy": torch.tensor([0.0])}

        calc = StudentCalculator(model=BadModel(), device="cpu")
        atoms = molecule("H2O")
        atoms.calc = calc

        # Energy should work
        energy = atoms.get_potential_energy()
        self.assertIsInstance(energy, (float, np.floating))

        # Forces should raise error
        with self.assertRaises(KeyError):
            _ = atoms.get_forces()


class TestStudentCalculatorWithSimpleMLP(unittest.TestCase):
    """Test StudentCalculator with SimpleMLP model."""

    def test_simple_mlp_forward(self):
        """Test that SimpleMLP works with StudentCalculator."""
        model = SimpleMLP(hidden_dim=32, num_layers=2)
        calc = StudentCalculator(model=model, device="cpu")

        atoms = molecule("H2O")
        # Enable gradient tracking for force calculation
        atoms.calc = calc

        # This should work
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        self.assertIsInstance(energy, (float, np.floating))
        self.assertIsInstance(forces, np.ndarray)
        self.assertEqual(forces.shape, (len(atoms), 3))


class TestStudentCalculatorRepr(unittest.TestCase):
    """Test StudentCalculator string representation."""

    def test_repr_with_model_path(self):
        """Test __repr__ includes model path when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"

            # Save a checkpoint
            checkpoint = {
                "model_state_dict": MockStudentModel().state_dict(),
                "model_class": MockStudentModel,
            }
            torch.save(checkpoint, model_path)

            calc = StudentCalculator(model_path=model_path, device="cpu")
            repr_str = repr(calc)

            self.assertIn("StudentCalculator", repr_str)
            self.assertIn("model_path", repr_str)

    def test_repr_without_model_path(self):
        """Test __repr__ without model path."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        repr_str = repr(calc)

        self.assertIn("StudentCalculator", repr_str)
        self.assertIn("device", repr_str)


if __name__ == "__main__":
    unittest.main()
