"""
MLFF Distiller: Fast, CUDA-optimized distilled force fields.

This package provides tools for knowledge distillation of machine learning
force fields (MLFFs) from large teacher models (Orb-models, FeNNol-PMC) to
smaller, faster student models with CUDA optimization.

Quick Start:
    # Load a trained student model
    from mlff_distiller.models import StudentForceField
    model = StudentForceField.load('checkpoints/best_model.pt')

    # Use as ASE calculator for MD simulations
    from mlff_distiller.inference import StudentForceFieldCalculator
    calc = StudentForceFieldCalculator('checkpoints/best_model.pt')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Run NVE MD validation
    from mlff_distiller.testing import NVEMDHarness
    harness = NVEMDHarness(atoms, calc, temperature=300.0)
    results = harness.run_simulation(steps=1000)
"""

__version__ = "0.1.0"

# Lazy imports for main components to avoid circular dependencies
# and reduce import time
from . import data
from . import models
from . import training
from . import inference
from . import testing
from . import cuda

# Convenience re-exports for common use cases
from .models import StudentForceField
from .inference import StudentForceFieldCalculator
from .testing import NVEMDHarness

__all__ = [
    # Version
    "__version__",
    # Submodules
    "data",
    "models",
    "training",
    "inference",
    "testing",
    "cuda",
    # Convenience exports
    "StudentForceField",
    "StudentForceFieldCalculator",
    "NVEMDHarness",
]
