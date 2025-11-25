"""
Custom CUDA/Triton kernels for optimized force field inference.

This module contains fused kernels that optimize bottlenecks in the
force computation pipeline.
"""

from .fused_rbf_cutoff import fused_rbf_cutoff_triton, FusedRBFCutoff

__all__ = [
    'fused_rbf_cutoff_triton',
    'FusedRBFCutoff',
]
