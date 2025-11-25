"""
Models package for MLFF Distiller.

This package contains:
- StudentForceField: Main PaiNN-based student model for force field predictions
- Teacher model wrappers (OrbCalculator, FeNNolCalculator)
- Student model calculator (StudentCalculator)
- Mock models for testing (MockStudentModel, SimpleMLP)
- Base model interfaces and utilities
"""

from .student_model import StudentForceField
from .teacher_wrappers import FeNNolCalculator, OrbCalculator
from .student_calculator import StudentCalculator
from .mock_student import MockStudentModel, SimpleMLP

__all__ = [
    # Main student model
    "StudentForceField",
    # Teacher model wrappers
    "OrbCalculator",
    "FeNNolCalculator",
    # Utilities
    "StudentCalculator",
    "MockStudentModel",
    "SimpleMLP",
]
