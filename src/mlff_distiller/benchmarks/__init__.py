"""
Benchmarking Utilities for MD Simulations

This module provides comprehensive benchmarking tools for measuring MD trajectory
performance, tracking memory stability, and comparing teacher vs student models.

Key Components:
- MDTrajectoryBenchmark: Core class for running MD trajectory benchmarks
- MDProtocol: Enum for MD protocol types (NVE, NVT, NPT)
- BenchmarkResults: Data class for storing and analyzing results
- Visualization utilities for performance comparison
"""

from .md_trajectory import (
    MDProtocol,
    MDTrajectoryBenchmark,
    BenchmarkResults,
    create_benchmark_system,
    compare_calculators,
)
from .visualization import (
    plot_latency_distribution,
    plot_energy_conservation,
    plot_performance_comparison,
    create_benchmark_report,
)

__all__ = [
    "MDProtocol",
    "MDTrajectoryBenchmark",
    "BenchmarkResults",
    "create_benchmark_system",
    "compare_calculators",
    "plot_latency_distribution",
    "plot_energy_conservation",
    "plot_performance_comparison",
    "create_benchmark_report",
]
