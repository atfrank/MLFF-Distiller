"""
Visualization Utilities for MD Benchmark Results

This module provides plotting and reporting functions for MD trajectory benchmarks:
- Latency distribution histograms
- Energy conservation plots
- Performance comparison tables
- Comprehensive HTML reports

Author: Testing & Benchmark Engineer
Date: 2025-11-23
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .md_trajectory import BenchmarkResults


def plot_latency_distribution(
    results: Union[BenchmarkResults, Dict[str, BenchmarkResults]],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Plot per-step latency distribution.

    Args:
        results: Single BenchmarkResults or dict of results
        output_path: Optional path to save figure
        show: Whether to display plot

    Example:
        >>> plot_latency_distribution(results, output_path="latency_dist.png")
    """
    # Handle single result or multiple
    if isinstance(results, BenchmarkResults):
        results_dict = {results.name: results}
    else:
        results_dict = results

    fig, axes = plt.subplots(
        len(results_dict), 1,
        figsize=(10, 4 * len(results_dict)),
        squeeze=False
    )

    for idx, (name, result) in enumerate(results_dict.items()):
        ax = axes[idx, 0]

        # Histogram of step times
        ax.hist(result.step_times_ms, bins=50, alpha=0.7, edgecolor='black')

        # Add vertical lines for statistics
        ax.axvline(result.mean_step_time_ms, color='r', linestyle='--',
                   label=f'Mean: {result.mean_step_time_ms:.2f} ms')
        ax.axvline(result.median_step_time_ms, color='g', linestyle='--',
                   label=f'Median: {result.median_step_time_ms:.2f} ms')
        ax.axvline(result.p95_step_time_ms, color='orange', linestyle='--',
                   label=f'P95: {result.p95_step_time_ms:.2f} ms')
        ax.axvline(result.p99_step_time_ms, color='purple', linestyle='--',
                   label=f'P99: {result.p99_step_time_ms:.2f} ms')

        ax.set_xlabel('Step Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Per-Step Latency Distribution: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_energy_conservation(
    results: Union[BenchmarkResults, Dict[str, BenchmarkResults]],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Plot energy over trajectory (for NVE energy conservation check).

    Args:
        results: Single BenchmarkResults or dict of results
        output_path: Optional path to save figure
        show: Whether to display plot

    Example:
        >>> plot_energy_conservation(nve_results, output_path="energy_conservation.png")
    """
    # Handle single result or multiple
    if isinstance(results, BenchmarkResults):
        results_dict = {results.name: results}
    else:
        results_dict = results

    fig, axes = plt.subplots(
        len(results_dict), 1,
        figsize=(12, 4 * len(results_dict)),
        squeeze=False
    )

    for idx, (name, result) in enumerate(results_dict.items()):
        ax = axes[idx, 0]

        if not result.energies:
            ax.text(0.5, 0.5, 'No energy data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Energy Conservation: {name}')
            continue

        energies = np.array(result.energies)

        # Plot absolute energy
        ax.plot(energies, alpha=0.7, label='Energy')

        # Add mean line
        mean_energy = np.mean(energies)
        ax.axhline(mean_energy, color='r', linestyle='--',
                  label=f'Mean: {mean_energy:.4f} eV')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Energy (eV)')
        ax.set_title(f'Energy Conservation: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add second axis for energy drift
        ax2 = ax.twinx()
        energy_drift = (energies - energies[0]) / abs(energies[0])
        ax2.plot(energy_drift, color='orange', alpha=0.5, label='Relative Drift')
        ax2.set_ylabel('Relative Energy Drift', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        if result.protocol == "NVE":
            # Highlight if drift is too large
            if result.energy_drift and abs(result.energy_drift) > 1e-4:
                ax2.text(0.5, 0.95, f'WARNING: Large drift ({result.energy_drift:.2e})',
                        ha='center', va='top', transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_comparison(
    results: Dict[str, BenchmarkResults],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Create comparison bar plots for multiple calculators.

    Args:
        results: Dictionary mapping names to BenchmarkResults
        output_path: Optional path to save figure
        show: Whether to display plot

    Example:
        >>> results = compare_calculators(calculators)
        >>> plot_performance_comparison(results, output_path="comparison.png")
    """
    names = list(results.keys())
    n_calcs = len(names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean latency comparison
    ax = axes[0, 0]
    means = [results[name].mean_step_time_ms for name in names]
    stds = [results[name].std_step_time_ms for name in names]
    x_pos = np.arange(n_calcs)

    ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Mean Step Time (ms)')
    ax.set_title('Mean Latency Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations (relative to first)
    baseline = means[0]
    for i, mean in enumerate(means):
        speedup = baseline / mean if mean > 0 else 0
        if i > 0:  # Don't show speedup for baseline
            ax.text(i, mean, f'{speedup:.2f}x', ha='center', va='bottom')

    # 2. Throughput comparison
    ax = axes[0, 1]
    throughputs = [results[name].steps_per_second for name in names]

    ax.bar(x_pos, throughputs, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Steps/Second')
    ax.set_title('Throughput Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Latency percentiles
    ax = axes[1, 0]
    width = 0.2
    metrics = ['P50', 'P95', 'P99']
    colors = ['blue', 'orange', 'red']

    for i, metric in enumerate(metrics):
        if metric == 'P50':
            values = [results[name].median_step_time_ms for name in names]
        elif metric == 'P95':
            values = [results[name].p95_step_time_ms for name in names]
        else:  # P99
            values = [results[name].p99_step_time_ms for name in names]

        offset = (i - 1) * width
        ax.bar(x_pos + offset, values, width, label=metric, alpha=0.7, color=colors[i])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Step Time (ms)')
    ax.set_title('Latency Percentiles')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Memory usage
    ax = axes[1, 1]
    peak_mems = [results[name].peak_memory_gb for name in names]
    deltas = [results[name].memory_delta_gb * 1024 for name in names]  # Convert to MB

    x_pos2 = np.arange(n_calcs)
    width = 0.35

    ax.bar(x_pos2 - width/2, peak_mems, width, label='Peak Memory (GB)', alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x_pos2 + width/2, deltas, width, label='Memory Delta (MB)',
            alpha=0.7, color='orange')

    ax.set_xticks(x_pos2)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Peak Memory (GB)', color='blue')
    ax2.set_ylabel('Memory Delta (MB)', color='orange')
    ax.set_title('Memory Usage')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def create_comparison_table(results: Dict[str, BenchmarkResults]) -> str:
    """
    Create formatted markdown table comparing benchmark results.

    Args:
        results: Dictionary mapping names to BenchmarkResults

    Returns:
        Markdown-formatted comparison table

    Example:
        >>> table = create_comparison_table(results)
        >>> print(table)
    """
    lines = [
        "# MD Trajectory Benchmark Comparison\n",
        "| Calculator | Atoms | Steps/s | Mean (ms) | P95 (ms) | P99 (ms) | Memory (GB) | Speedup |",
        "|------------|-------|---------|-----------|----------|----------|-------------|---------|",
    ]

    # Calculate baseline (first entry) for speedup
    names = list(results.keys())
    baseline_mean = results[names[0]].mean_step_time_ms

    for name in names:
        result = results[name]
        speedup = baseline_mean / result.mean_step_time_ms if result.mean_step_time_ms > 0 else 0

        line = (
            f"| {name:<20} | {result.n_atoms:5d} | {result.steps_per_second:7.1f} | "
            f"{result.mean_step_time_ms:9.4f} | {result.p95_step_time_ms:8.4f} | "
            f"{result.p99_step_time_ms:8.4f} | {result.peak_memory_gb:11.4f} | "
            f"{speedup:7.2f}x |"
        )
        lines.append(line)

    return '\n'.join(lines)


def create_benchmark_report(
    results: Dict[str, BenchmarkResults],
    output_dir: Union[str, Path],
    title: str = "MD Trajectory Benchmark Report",
):
    """
    Create comprehensive benchmark report with plots and tables.

    Args:
        results: Dictionary mapping names to BenchmarkResults
        output_dir: Directory to save report files
        title: Report title

    Creates:
        - report.md: Markdown report with tables
        - latency_dist.png: Latency distribution plots
        - energy_conservation.png: Energy conservation plots
        - comparison.png: Performance comparison plots

    Example:
        >>> create_benchmark_report(results, "reports/baseline_benchmark")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_latency_distribution(
        results,
        output_path=output_dir / "latency_dist.png",
        show=False
    )

    plot_energy_conservation(
        results,
        output_path=output_dir / "energy_conservation.png",
        show=False
    )

    plot_performance_comparison(
        results,
        output_path=output_dir / "comparison.png",
        show=False
    )

    # Generate markdown report
    report_lines = [
        f"# {title}\n",
        f"**Generated**: {np.datetime64('now')}\n",
        f"**Number of calculators**: {len(results)}\n",
        "",
        "## Summary\n",
        create_comparison_table(results),
        "",
        "## Visualizations\n",
        "### Performance Comparison",
        "![Performance Comparison](comparison.png)\n",
        "### Latency Distributions",
        "![Latency Distributions](latency_dist.png)\n",
        "### Energy Conservation",
        "![Energy Conservation](energy_conservation.png)\n",
        "",
        "## Detailed Results\n",
    ]

    # Add detailed results for each calculator
    for name, result in results.items():
        report_lines.extend([
            f"### {name}\n",
            "```",
            result.summary(),
            "```\n",
        ])

    # Write report
    report_path = output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nBenchmark report created: {report_path}")
    print(f"  - Comparison table: report.md")
    print(f"  - Latency plot: latency_dist.png")
    print(f"  - Energy plot: energy_conservation.png")
    print(f"  - Comparison plot: comparison.png")
