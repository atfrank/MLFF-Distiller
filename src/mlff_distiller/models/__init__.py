"""
Models package for MLFF Distiller.

This package contains:
- Teacher model wrappers (OrbCalculator, FeNNolCalculator)
- Student model calculator (StudentCalculator)
- Mock models for testing (MockStudentModel, SimpleMLP)
- Base model interfaces and utilities
"""

from .teacher_wrappers import FeNNolCalculator, OrbCalculator
from .student_calculator import StudentCalculator
from .mock_student import MockStudentModel, SimpleMLP

__all__ = [
    "OrbCalculator",
    "FeNNolCalculator",
    "StudentCalculator",
    "MockStudentModel",
    "SimpleMLP",
]
