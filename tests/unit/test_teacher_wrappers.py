"""
Unit tests for teacher model wrapper calculators.

Tests the OrbCalculator and FeNNolCalculator wrapper classes to ensure:
1. Proper ASE Calculator interface implementation
2. Correct initialization and model loading
3. Property calculations (energy, forces, stress)
4. Device handling (CPU/CUDA)
5. Drop-in replacement compatibility

Author: ML Architecture Designer
Date: 2025-11-23
"""

import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

import pytest


class TestOrbCalculatorInterface(unittest.TestCase):
    """Test OrbCalculator ASE interface compliance."""

    def setUp(self):
        """Set up mock Orb calculator for testing."""
        # Mock the orb_models modules
        self.mock_pretrained = MagicMock()
        self.mock_orb_calc_class = MagicMock()
        self.mock_orb_calc_instance = MagicMock(spec=Calculator)
        self.mock_orb_calc_instance.results = {}
        self.mock_orb_calc_class.return_value = self.mock_orb_calc_instance

        sys.modules["orb_models"] = MagicMock()
        sys.modules["orb_models.forcefield"] = MagicMock()
        sys.modules["orb_models.forcefield.pretrained"] = self.mock_pretrained
        sys.modules["orb_models.forcefield.calculator"] = MagicMock(
            ORBCalculator=self.mock_orb_calc_class
        )

        # Mock model loader
        self.mock_pretrained.orb_v2.return_value = MagicMock()

        from mlff_distiller.models.teacher_wrappers import OrbCalculator

        self.calc = OrbCalculator(model_name="orb-v2", device="cpu")

    def tearDown(self):
        """Clean up mocked modules."""
        for module in [
            "orb_models",
            "orb_models.forcefield",
            "orb_models.forcefield.pretrained",
            "orb_models.forcefield.calculator",
        ]:
            if module in sys.modules:
                del sys.modules[module]

    def test_inherits_from_calculator(self):
        """Test that OrbCalculator inherits from ASE Calculator."""
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


class TestOrbCalculatorInitialization(unittest.TestCase):
    """Test OrbCalculator initialization with different models."""

    def setUp(self):
        """Set up mocked modules."""
        self.mock_pretrained = MagicMock()
        self.mock_orb_calc_class = MagicMock()

        sys.modules["orb_models"] = MagicMock()
        sys.modules["orb_models.forcefield"] = MagicMock()
        sys.modules["orb_models.forcefield.pretrained"] = self.mock_pretrained
        sys.modules["orb_models.forcefield.calculator"] = MagicMock(
            ORBCalculator=self.mock_orb_calc_class
        )

        # Set up model loaders
        self.mock_pretrained.orb_v1.return_value = MagicMock()
        self.mock_pretrained.orb_v2.return_value = MagicMock()
        self.mock_pretrained.orb_v3_conservative_inf_omat.return_value = MagicMock()

        mock_calc = MagicMock()
        mock_calc.results = {}
        self.mock_orb_calc_class.return_value = mock_calc

    def tearDown(self):
        """Clean up mocked modules."""
        for module in [
            "orb_models",
            "orb_models.forcefield",
            "orb_models.forcefield.pretrained",
            "orb_models.forcefield.calculator",
        ]:
            if module in sys.modules:
                del sys.modules[module]

    def test_initialization_orb_v1(self):
        """Test initialization with orb-v1."""
        from mlff_distiller.models.teacher_wrappers import OrbCalculator

        calc = OrbCalculator(model_name="orb-v1", device="cpu")
        # Check that calculator was created successfully
        self.assertEqual(calc.model_name, "orb-v1")
        self.assertEqual(calc.device, "cpu")

    def test_initialization_orb_v2(self):
        """Test initialization with orb-v2."""
        from mlff_distiller.models.teacher_wrappers import OrbCalculator

        calc = OrbCalculator(model_name="orb-v2", device="cpu")
        self.assertEqual(calc.model_name, "orb-v2")
        self.assertEqual(calc.device, "cpu")

    def test_initialization_orb_v3(self):
        """Test initialization with orb-v3."""
        from mlff_distiller.models.teacher_wrappers import OrbCalculator

        calc = OrbCalculator(model_name="orb-v3", device="cpu")
        self.assertEqual(calc.model_name, "orb-v3")
        self.assertEqual(calc.device, "cpu")

    def test_invalid_model_name(self):
        """Test that invalid model name raises ValueError."""
        from mlff_distiller.models.teacher_wrappers import OrbCalculator

        with self.assertRaises(ValueError) as context:
            OrbCalculator(model_name="invalid-model", device="cpu")

        self.assertIn("Unknown model name", str(context.exception))


class TestOrbCalculatorCalculations(unittest.TestCase):
    """Test OrbCalculator property calculations."""

    def setUp(self):
        """Set up mock calculator with fake results."""
        # Set up mock results
        self.mock_energy = -10.5
        self.mock_forces = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
        self.mock_stress = np.array([0.01, 0.02, 0.03, 0.0, 0.0, 0.0])

        mock_calc_instance = MagicMock(spec=Calculator)
        mock_calc_instance.results = {
            "energy": self.mock_energy,
            "forces": self.mock_forces,
            "stress": self.mock_stress,
        }

        # Mock modules
        mock_pretrained = MagicMock()
        mock_pretrained.orb_v2.return_value = MagicMock()

        sys.modules["orb_models"] = MagicMock()
        sys.modules["orb_models.forcefield"] = MagicMock()
        sys.modules["orb_models.forcefield.pretrained"] = mock_pretrained
        sys.modules["orb_models.forcefield.calculator"] = MagicMock(
            ORBCalculator=MagicMock(return_value=mock_calc_instance)
        )

        from mlff_distiller.models.teacher_wrappers import OrbCalculator

        self.calc = OrbCalculator(model_name="orb-v2", device="cpu")
        self.atoms = Atoms(
            "H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True
        )

    def tearDown(self):
        """Clean up mocked modules."""
        for module in [
            "orb_models",
            "orb_models.forcefield",
            "orb_models.forcefield.pretrained",
            "orb_models.forcefield.calculator",
        ]:
            if module in sys.modules:
                del sys.modules[module]

    def test_calculate_populates_results(self):
        """Test that calculate method populates results."""
        self.atoms.calc = self.calc
        self.calc.calculate(self.atoms, properties=["energy", "forces", "stress"])

        self.assertIn("energy", self.calc.results)
        self.assertIn("forces", self.calc.results)
        self.assertIn("stress", self.calc.results)


class TestFeNNolCalculatorInterface(unittest.TestCase):
    """Test FeNNolCalculator ASE interface compliance."""

    def setUp(self):
        """Set up mock FeNNol calculator for testing."""
        # Mock JAX and FeNNol
        mock_fennix_instance = MagicMock(spec=Calculator)
        mock_fennix_instance.results = {}

        mock_fennix_class = MagicMock()
        mock_fennix_class.from_pretrained.return_value = mock_fennix_instance

        sys.modules["jax"] = MagicMock()
        sys.modules["fennol"] = MagicMock()
        sys.modules["fennol.calculators"] = MagicMock(FENNIXCalculator=mock_fennix_class)

        from mlff_distiller.models.teacher_wrappers import FeNNolCalculator

        self.calc = FeNNolCalculator(model_name="ani-2x", device="cpu")

    def tearDown(self):
        """Clean up mocked modules."""
        for module in ["jax", "fennol", "fennol.calculators"]:
            if module in sys.modules:
                del sys.modules[module]

    def test_inherits_from_calculator(self):
        """Test that FeNNolCalculator inherits from ASE Calculator."""
        self.assertIsInstance(self.calc, Calculator)

    def test_implemented_properties(self):
        """Test that required properties are declared as implemented."""
        self.assertIn("energy", self.calc.implemented_properties)
        self.assertIn("forces", self.calc.implemented_properties)

    def test_has_calculate_method(self):
        """Test that calculate method exists and is callable."""
        self.assertTrue(hasattr(self.calc, "calculate"))
        self.assertTrue(callable(self.calc.calculate))


class TestFeNNolCalculatorInitialization(unittest.TestCase):
    """Test FeNNolCalculator initialization."""

    def setUp(self):
        """Set up mocked modules."""
        sys.modules["jax"] = MagicMock()
        sys.modules["fennol"] = MagicMock()
        sys.modules["fennol.calculators"] = MagicMock(
            FENNIXCalculator=MagicMock(from_pretrained=MagicMock(), from_checkpoint=MagicMock())
        )

    def tearDown(self):
        """Clean up mocked modules."""
        for module in ["jax", "fennol", "fennol.calculators"]:
            if module in sys.modules:
                del sys.modules[module]

    def test_requires_model_path_or_name(self):
        """Test that initialization requires either model_path or model_name."""
        from mlff_distiller.models.teacher_wrappers import FeNNolCalculator

        with self.assertRaises(ValueError) as context:
            FeNNolCalculator(device="cpu")

        self.assertIn("model_path or model_name must be provided", str(context.exception))


class TestDropInCompatibility(unittest.TestCase):
    """Test drop-in replacement compatibility for both calculators."""

    def setUp(self):
        """Set up mocked modules."""
        # Mock Orb
        mock_pretrained = MagicMock()
        mock_pretrained.orb_v2.return_value = MagicMock()
        mock_orb_calc = MagicMock()
        mock_orb_calc.return_value = MagicMock(results={})

        sys.modules["orb_models"] = MagicMock()
        sys.modules["orb_models.forcefield"] = MagicMock()
        sys.modules["orb_models.forcefield.pretrained"] = mock_pretrained
        sys.modules["orb_models.forcefield.calculator"] = MagicMock(ORBCalculator=mock_orb_calc)

        # Mock FeNNol
        sys.modules["jax"] = MagicMock()
        sys.modules["fennol"] = MagicMock()
        sys.modules["fennol.calculators"] = MagicMock(
            FENNIXCalculator=MagicMock(from_pretrained=MagicMock(return_value=MagicMock(results={})))
        )

    def tearDown(self):
        """Clean up mocked modules."""
        for module in [
            "orb_models",
            "orb_models.forcefield",
            "orb_models.forcefield.pretrained",
            "orb_models.forcefield.calculator",
            "jax",
            "fennol",
            "fennol.calculators",
        ]:
            if module in sys.modules:
                del sys.modules[module]

    def test_orb_calculator_can_attach_to_atoms(self):
        """Test that OrbCalculator can be attached to Atoms object."""
        from mlff_distiller.models.teacher_wrappers import OrbCalculator

        calc = OrbCalculator(model_name="orb-v2", device="cpu")
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calc

        self.assertIs(atoms.calc, calc)

    def test_fennol_calculator_can_attach_to_atoms(self):
        """Test that FeNNolCalculator can be attached to Atoms object."""
        from mlff_distiller.models.teacher_wrappers import FeNNolCalculator

        calc = FeNNolCalculator(model_name="ani-2x", device="cpu")
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = calc

        self.assertIs(atoms.calc, calc)


if __name__ == "__main__":
    unittest.main()
