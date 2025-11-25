"""
Command-line interface for MLFF Distiller.

This module provides CLI commands for training, validation, and benchmarking
of distilled force field models.

Commands:
    mlff-train      Train a student model via knowledge distillation
    mlff-validate   Validate a trained model with MD simulations
    mlff-benchmark  Benchmark model performance
"""

from .main import train, validate, benchmark

__all__ = ['train', 'validate', 'benchmark']
