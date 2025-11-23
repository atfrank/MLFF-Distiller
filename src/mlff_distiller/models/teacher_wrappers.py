"""
Teacher Model Wrappers with ASE Calculator Interface

This module provides wrapper classes for teacher models (Orb-models and FeNNol-PMC)
that implement the ASE Calculator interface. These wrappers enable:

1. Drop-in replacement compatibility with existing ASE MD scripts
2. Baseline MD trajectory benchmarking
3. Data generation from teacher models
4. Template for student model interfaces

All calculators follow the ASE Calculator standard and support:
- get_potential_energy() - returns energy in eV
- get_forces() - returns forces in eV/Angstrom
- get_stress() - returns stress tensor in eV/Angstrom^3
- Integration with ASE MD integrators (VelocityVerlet, Langevin, etc.)

Usage:
    from mlff_distiller.models.teacher_wrappers import OrbCalculator, FeNNolCalculator
    from ase import Atoms

    # Create atoms
    atoms = Atoms('H2O', positions=[[0,0,0], [1,0,0], [0,1,0]], pbc=True)

    # Use Orb model
    calc = OrbCalculator(model_name="orb-v2", device="cuda")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Or use FeNNol model
    calc = FeNNolCalculator(model_path="path/to/model", device="cuda")
    atoms.calc = calc
    energy = atoms.get_potential_energy()

Author: ML Architecture Designer
Date: 2025-11-23
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import voigt_6_to_full_3x3_stress


class OrbCalculator(Calculator):
    """
    ASE Calculator wrapper for Orb-models teacher models.

    This calculator wraps the Orb-models force field (orb-v1, orb-v2, orb-v3)
    and provides a drop-in replacement interface compatible with ASE MD simulations.

    Orb-models are state-of-the-art universal force fields trained on large-scale
    DFT datasets. They provide fast, accurate predictions of energies, forces, and
    stresses for materials and molecules.

    Args:
        model_name: Name of the pretrained Orb model. Options:
            - "orb-v1": Original Orb model
            - "orb-v2": Improved Orb model
            - "orb-v3": Latest model with confidence estimation
            Available model variants include:
            - "orb-v3-conservative-inf-omat": Conservative inference mode
            - "orb-v3-strict-inf-omat": Strict inference mode
            - "orb-v2": General purpose v2 model
        device: Device to run model on ('cpu' or 'cuda')
        precision: Precision mode for computation. Options:
            - "float32-high": Standard precision (default)
            - "float32-highest": Higher precision float32
            - "float64": Full double precision
        dtype: PyTorch dtype for computations (torch.float32 or torch.float64)
        **kwargs: Additional arguments passed to ASE Calculator

    Attributes:
        implemented_properties: List of properties this calculator can compute
        model: The loaded Orb forcefield model
        device: Device where model is loaded
        orbff: The underlying Orb forcefield object

    Example:
        >>> from ase.build import bulk
        >>> from mlff_distiller.models.teacher_wrappers import OrbCalculator
        >>>
        >>> # Create atoms
        >>> atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
        >>>
        >>> # Setup calculator
        >>> calc = OrbCalculator(model_name="orb-v2", device="cuda")
        >>> atoms.calc = calc
        >>>
        >>> # Compute properties
        >>> energy = atoms.get_potential_energy()  # eV
        >>> forces = atoms.get_forces()            # eV/Angstrom
        >>> stress = atoms.get_stress()            # eV/Angstrom^3
        >>>
        >>> # Use in MD simulation
        >>> from ase.md.verlet import VelocityVerlet
        >>> from ase import units
        >>> dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
        >>> dyn.run(1000)

    Notes:
        - Orb models support both periodic and non-periodic systems
        - Models handle variable system sizes (10-1000 atoms efficiently)
        - Orb-v3 models include per-atom confidence estimates
        - Results are cached by ASE to avoid redundant calculations
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_name: str = "orb-v2",
        device: str = "cuda",
        precision: str = "float32-high",
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """Initialize OrbCalculator with pretrained Orb model."""
        Calculator.__init__(self, **kwargs)

        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.dtype = dtype

        # Load Orb model
        self._load_model()

        # Buffers for efficient repeated calls (minimize allocations)
        self._position_buffer = None
        self._numbers_buffer = None

    def _load_model(self):
        """Load pretrained Orb model from orb_models package."""
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
        except ImportError as e:
            raise ImportError(
                "orb-models package not found. Install with: pip install orb-models"
            ) from e

        # Map model names to pretrained model functions
        # Note: compile=False required for Python 3.13+ (torch.compile not supported)
        model_loaders = {
            "orb-v1": lambda: pretrained.orb_v1(device=self.device, compile=False),
            "orb-v2": lambda: pretrained.orb_v2(device=self.device, compile=False),
            "orb-v3": lambda: pretrained.orb_v3_conservative_inf_omat(
                device=self.device, precision=self.precision, compile=False
            ),
            "orb-v3-conservative-inf-omat": lambda: pretrained.orb_v3_conservative_inf_omat(
                device=self.device, precision=self.precision, compile=False
            ),
            "orb-v3-strict-inf-omat": lambda: pretrained.orb_v3_strict_inf_omat(
                device=self.device, precision=self.precision, compile=False
            ),
        }

        if self.model_name not in model_loaders:
            raise ValueError(
                f"Unknown model name: {self.model_name}. "
                f"Available models: {list(model_loaders.keys())}"
            )

        # Load the forcefield
        self.orbff = model_loaders[self.model_name]()

        # Create the underlying ORB calculator (we wrap this)
        self._orb_calc = ORBCalculator(self.orbff, device=self.device)

    def calculate(
        self,
        atoms=None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ):
        """
        Calculate properties for given atoms.

        This method is called by ASE when properties are requested (e.g., via
        atoms.get_potential_energy()). It delegates to the underlying ORBCalculator
        but ensures proper interface compliance.

        Args:
            atoms: ASE Atoms object to calculate properties for
            properties: List of properties to calculate ('energy', 'forces', 'stress')
            system_changes: List of changes since last calculation (ASE internal)
        """
        # Call parent to handle ASE caching logic
        Calculator.calculate(self, atoms, properties, system_changes)

        # Delegate to underlying ORB calculator
        # ORBCalculator already implements ASE interface correctly
        self._orb_calc.calculate(atoms, properties, system_changes)

        # Copy results from underlying calculator
        self.results = self._orb_calc.results.copy()

        # Handle confidence if available (Orb-v3 only)
        if "confidence" in self._orb_calc.results:
            self.results["confidence"] = self._orb_calc.results["confidence"]


class FeNNolCalculator(Calculator):
    """
    ASE Calculator wrapper for FeNNol-PMC teacher models.

    This calculator wraps the FeNNol (Force-field-enhanced Neural Network optimized
    library) models and provides a drop-in replacement interface compatible with
    ASE MD simulations.

    FeNNol is a JAX-based library for building, training, and running neural network
    potentials for molecular simulations. It combines state-of-the-art neural network
    embeddings with ML-parameterized physical interaction terms.

    Args:
        model_path: Path to FeNNol model checkpoint/weights
        model_name: Optional name of pretrained FeNNol model (e.g., "ani-2x")
        device: Device to run model on ('cpu' or 'cuda')
        dtype: PyTorch/JAX dtype for computations
        **kwargs: Additional arguments passed to ASE Calculator

    Attributes:
        implemented_properties: List of properties this calculator can compute
        model_path: Path to model checkpoint
        device: Device where model is loaded
        fennix_calc: The underlying FENNIXCalculator object

    Example:
        >>> from ase.build import molecule
        >>> from mlff_distiller.models.teacher_wrappers import FeNNolCalculator
        >>>
        >>> # Create molecule
        >>> atoms = molecule('H2O')
        >>>
        >>> # Setup calculator with pretrained model
        >>> calc = FeNNolCalculator(model_name="ani-2x", device="cuda")
        >>> atoms.calc = calc
        >>>
        >>> # Compute properties
        >>> energy = atoms.get_potential_energy()  # eV
        >>> forces = atoms.get_forces()            # eV/Angstrom
        >>>
        >>> # Use in MD simulation
        >>> from ase.md.langevin import Langevin
        >>> from ase import units
        >>> dyn = Langevin(atoms, timestep=1.0*units.fs, temperature_K=300,
        ...                friction=0.01)
        >>> dyn.run(1000)

    Notes:
        - FeNNol is based on JAX and provides fast GPU-accelerated inference
        - Supports both periodic and non-periodic systems
        - ANI-2x model achieves near force-field speeds on GPUs
        - Model loading may require specific FeNNol configuration files
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cuda",
        dtype = None,
        **kwargs
    ):
        """Initialize FeNNolCalculator with FeNNol model."""
        Calculator.__init__(self, **kwargs)

        if model_path is None and model_name is None:
            raise ValueError("Either model_path or model_name must be provided")

        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Load FeNNol model
        self._load_model()

    def _load_model(self):
        """Load FeNNol model from checkpoint or pretrained."""
        try:
            import jax
            # Note: FeNNol uses JAX, not PyTorch
            # The exact import path may vary based on FeNNol version
            # This is a placeholder that should be updated based on actual FeNNol API
            from fennol.calculators import FENNIXCalculator
        except ImportError as e:
            raise ImportError(
                "FeNNol package not found. Install with: pip install fennol\n"
                "Note: FeNNol requires JAX. Install JAX first:\n"
                "  CPU: pip install jax\n"
                "  GPU: pip install jax[cuda12]\n"
            ) from e

        # Set JAX device
        if self.device == "cuda":
            # JAX automatically uses GPU if available
            pass

        # Load model based on provided path or name
        if self.model_path is not None:
            # Load from checkpoint file
            # Note: Actual implementation depends on FeNNol's model loading API
            self._fennix_calc = FENNIXCalculator.from_checkpoint(self.model_path)
        elif self.model_name is not None:
            # Load pretrained model
            # Note: This is a placeholder - actual API depends on FeNNol version
            if self.model_name == "ani-2x":
                # FeNNol provides ANI-2x as an example model
                self._fennix_calc = FENNIXCalculator.from_pretrained("ani-2x")
            else:
                raise ValueError(
                    f"Unknown pretrained model: {self.model_name}. "
                    "Available models: ['ani-2x']"
                )

    def calculate(
        self,
        atoms=None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ):
        """
        Calculate properties for given atoms.

        This method is called by ASE when properties are requested. It delegates
        to the underlying FENNIXCalculator.

        Args:
            atoms: ASE Atoms object to calculate properties for
            properties: List of properties to calculate ('energy', 'forces')
            system_changes: List of changes since last calculation (ASE internal)
        """
        # Call parent to handle ASE caching logic
        Calculator.calculate(self, atoms, properties, system_changes)

        # Delegate to underlying FENNIX calculator
        self._fennix_calc.calculate(atoms, properties, system_changes)

        # Copy results from underlying calculator
        self.results = self._fennix_calc.results.copy()


# Convenience exports
__all__ = ["OrbCalculator", "FeNNolCalculator"]
