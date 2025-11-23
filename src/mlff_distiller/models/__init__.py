"""
Models package for MLFF Distiller.

This package contains:
- Teacher model wrappers (OrbCalculator, FeNNolCalculator)
- Student model architectures (to be implemented)
- Base model interfaces and utilities
"""

from .teacher_wrappers import FeNNolCalculator, OrbCalculator

__all__ = ["OrbCalculator", "FeNNolCalculator"]
