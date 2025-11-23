#!/usr/bin/env python3
"""
Example Profiling Script for MLFF Distiller

Demonstrates comprehensive profiling workflows for MD simulations:
1. Basic performance benchmarking
2. Memory leak detection
3. PyTorch profiler integration
4. Nsight Systems profiling preparation
5. MD trajectory simulation

This script serves as a template for profiling both teacher and student models
in realistic MD simulation scenarios.

Usage:
    python benchmarks/profile_example.py
    python benchmarks/profile_example.py --n-calls 10000
    python benchmarks/profile_example.py --profile-memory
    python benchmarks/profile_example.py --export-traces
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn

from cuda.benchmark_utils import (
    BenchmarkResult,
    ProfilerContext,
    benchmark_function,
    benchmark_md_trajectory,
    compare_implementations,
    print_comparison_table,
    profile_with_nsys,
)
from cuda.device_utils import (
    get_device,
    get_gpu_memory_info,
    print_device_summary,
    print_gpu_memory_summary,
    warmup_cuda,
)


class DummyForceFieldModel(nn.Module):
    """
    Dummy force field model for profiling demonstration.

    Simulates a typical ML force field architecture:
    - Input: atomic coordinates and types
    - Output: energies and forces

    In practice, replace with actual teacher/student models.
    """

    def __init__(self, n_atoms: int = 100, hidden_dim: int = 128):
        super().__init__()
        self.n_atoms = n_atoms

        # Simplified architecture
        self.embedding = nn.Embedding(10, hidden_dim)  # 10 atom types
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for coordinates
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.energy_head = nn.Linear(hidden_dim, 1)
        self.force_head = nn.Linear(hidden_dim, 3)

    def forward(self, atom_types, positions):
        """
        Forward pass.

        Args:
            atom_types: Tensor of shape [n_atoms] with atom type indices
            positions: Tensor of shape [n_atoms, 3] with coordinates

        Returns:
            energy: Scalar energy
            forces: Tensor of shape [n_atoms, 3] with forces
        """
        # Embed atom types
        x = self.embedding(atom_types)  # [n_atoms, hidden_dim]

        # Concatenate with positions
        x = torch.cat([x, positions], dim=-1)  # [n_atoms, hidden_dim + 3]

        # Encode
        x = self.encoder(x)  # [n_atoms, hidden_dim]

        # Compute energy (sum over atoms)
        energy = self.energy_head(x).sum()  # scalar

        # Compute forces
        forces = self.force_head(x)  # [n_atoms, 3]

        return energy, forces


def generate_dummy_input(n_atoms: int = 100, device: torch.device = None):
    """Generate dummy input for profiling."""
    if device is None:
        device = torch.device('cpu')

    atom_types = torch.randint(0, 10, (n_atoms,), device=device)
    positions = torch.randn(n_atoms, 3, device=device)

    return atom_types, positions


def profile_basic_benchmark(device: torch.device, n_calls: int = 100):
    """Demonstrate basic benchmarking."""
    print("\n" + "=" * 80)
    print("1. Basic Performance Benchmark")
    print("=" * 80)

    # Create model
    model = DummyForceFieldModel(n_atoms=100, hidden_dim=128).to(device)
    model.eval()

    # Create input
    atom_types, positions = generate_dummy_input(n_atoms=100, device=device)

    # Benchmark
    with torch.no_grad():
        result = benchmark_function(
            lambda: model(atom_types, positions),
            n_warmup=10,
            n_calls=n_calls,
            device=device,
            name="Dummy Force Field (100 atoms)",
        )

    print(result.summary())

    return result


def profile_md_trajectory_simulation(device: torch.device, n_steps: int = 1000):
    """Demonstrate MD trajectory profiling."""
    print("\n" + "=" * 80)
    print("2. MD Trajectory Simulation")
    print("=" * 80)
    print(f"Simulating {n_steps}-step MD trajectory")
    print("(In practice, this would be real MD with changing atomic positions)")

    # Create model
    model = DummyForceFieldModel(n_atoms=100, hidden_dim=128).to(device)
    model.eval()

    # Generate trajectory inputs (simulating MD timesteps)
    trajectory_inputs = []
    for _ in range(n_steps):
        atom_types, positions = generate_dummy_input(n_atoms=100, device=device)
        trajectory_inputs.append((atom_types, positions))

    print(f"Generated {len(trajectory_inputs)} trajectory frames")

    # Benchmark trajectory
    with torch.no_grad():
        result = benchmark_md_trajectory(
            model_func=lambda inp: model(inp[0], inp[1]),
            inputs=trajectory_inputs,
            n_warmup=10,
            device=device,
            name=f"MD Trajectory ({n_steps} steps)",
            check_memory_leak=True,
            leak_tolerance_mb=10.0,
        )

    print(result.summary())

    # Check memory stability
    if abs(result.memory_delta_gb * 1024) > 10:
        print("WARNING: Memory increased by more than 10 MB during trajectory!")
        print("This indicates a potential memory leak that will cause issues in long MD runs.")
    else:
        print("PASS: Memory is stable over trajectory (no leak detected)")

    return result


def profile_model_comparison(device: torch.device, n_calls: int = 100):
    """Demonstrate comparing multiple implementations."""
    print("\n" + "=" * 80)
    print("3. Implementation Comparison")
    print("=" * 80)

    # Create models of different sizes (simulating teacher vs student)
    model_large = DummyForceFieldModel(n_atoms=100, hidden_dim=256).to(device)
    model_medium = DummyForceFieldModel(n_atoms=100, hidden_dim=128).to(device)
    model_small = DummyForceFieldModel(n_atoms=100, hidden_dim=64).to(device)

    model_large.eval()
    model_medium.eval()
    model_small.eval()

    # Create input
    atom_types, positions = generate_dummy_input(n_atoms=100, device=device)

    # Prepare implementations
    implementations = {
        "Large Model (256 hidden)": lambda: model_large(atom_types, positions),
        "Medium Model (128 hidden)": lambda: model_medium(atom_types, positions),
        "Small Model (64 hidden)": lambda: model_small(atom_types, positions),
    }

    # Benchmark all
    with torch.no_grad():
        results = compare_implementations(
            implementations,
            n_warmup=10,
            n_calls=n_calls,
            device=device,
        )

    # Print comparison
    print_comparison_table(results)

    return results


def profile_with_pytorch_profiler(device: torch.device, output_dir: Path):
    """Demonstrate PyTorch profiler integration."""
    print("\n" + "=" * 80)
    print("4. PyTorch Profiler Integration")
    print("=" * 80)

    # Create model
    model = DummyForceFieldModel(n_atoms=100, hidden_dim=128).to(device)
    model.eval()

    # Create input
    atom_types, positions = generate_dummy_input(n_atoms=100, device=device)

    # Profile
    with ProfilerContext(
        output_dir=output_dir,
        profile_memory=True,
        record_shapes=True,
        name="dummy_force_field",
    ) as prof:
        with torch.no_grad():
            # Run a few iterations
            for _ in range(20):
                _ = model(atom_types, positions)
                prof.step()

    print(f"\nProfiler results saved to: {output_dir}")
    print(f"View Chrome trace: chrome://tracing (load {output_dir}/dummy_force_field_trace.json)")


def profile_memory_scaling(device: torch.device):
    """Demonstrate memory scaling with system size."""
    print("\n" + "=" * 80)
    print("5. Memory Scaling Analysis")
    print("=" * 80)

    system_sizes = [50, 100, 200, 500, 1000]
    results = []

    for n_atoms in system_sizes:
        print(f"\nTesting {n_atoms} atoms...")

        model = DummyForceFieldModel(n_atoms=n_atoms, hidden_dim=128).to(device)
        model.eval()

        atom_types, positions = generate_dummy_input(n_atoms=n_atoms, device=device)

        # Benchmark
        with torch.no_grad():
            result = benchmark_function(
                lambda: model(atom_types, positions),
                n_warmup=5,
                n_calls=50,
                device=device,
                name=f"{n_atoms} atoms",
            )

        results.append((n_atoms, result.mean_ms, result.peak_memory_gb))

        print(f"  {n_atoms} atoms: {result.mean_ms:.4f} ms, {result.peak_memory_gb:.4f} GB")

    # Summary table
    print("\n" + "-" * 60)
    print(f"{'Atoms':>10} {'Latency (ms)':>15} {'Memory (GB)':>15} {'µs/atom':>12}")
    print("-" * 60)
    for n_atoms, latency, memory in results:
        us_per_atom = (latency * 1000) / n_atoms
        print(f"{n_atoms:>10} {latency:>15.4f} {memory:>15.4f} {us_per_atom:>12.2f}")
    print("-" * 60)


def demonstrate_nsys_profiling():
    """Show how to use Nsight Systems profiling."""
    print("\n" + "=" * 80)
    print("6. Nsight Systems (nsys) Profiling")
    print("=" * 80)

    # Generate command
    cmd = profile_with_nsys(
        command="python benchmarks/profile_example.py --n-calls 100",
        output_file="profile_example.nsys-rep",
        cuda_events=True,
    )

    print("This will generate detailed GPU profiling including:")
    print("  - Kernel execution timeline")
    print("  - Memory transfers")
    print("  - CUDA API calls")
    print("  - Bottleneck analysis")
    print("\nFor production profiling of teacher/student models:")
    print("  nsys profile --trace=cuda,nvtx python train.py")


def main():
    parser = argparse.ArgumentParser(description="MLFF Distiller Profiling Example")
    parser.add_argument(
        "--n-calls",
        type=int,
        default=100,
        help="Number of benchmark calls (default: 100)",
    )
    parser.add_argument(
        "--n-trajectory-steps",
        type=int,
        default=1000,
        help="Number of MD trajectory steps (default: 1000)",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Run memory scaling analysis",
    )
    parser.add_argument(
        "--export-traces",
        action="store_true",
        help="Export PyTorch profiler traces",
    )
    parser.add_argument(
        "--show-nsys",
        action="store_true",
        help="Show nsys profiling command",
    )

    args = parser.parse_args()

    # Print device summary
    print_device_summary()

    # Get device
    device = get_device(prefer_cuda=True)
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        warmup_cuda(device=device)
        print("CUDA warmup complete")

    # Run profiling examples
    try:
        # 1. Basic benchmark
        profile_basic_benchmark(device, n_calls=args.n_calls)

        # 2. MD trajectory
        profile_md_trajectory_simulation(device, n_steps=args.n_trajectory_steps)

        # 3. Model comparison
        profile_model_comparison(device, n_calls=args.n_calls)

        # 4. PyTorch profiler (if requested)
        if args.export_traces:
            output_dir = Path(__file__).parent.parent / "profiling_results"
            profile_with_pytorch_profiler(device, output_dir)

        # 5. Memory scaling (if requested)
        if args.profile_memory:
            profile_memory_scaling(device)

        # 6. Nsys command (if requested)
        if args.show_nsys:
            demonstrate_nsys_profiling()

        # Final memory summary
        if device.type == 'cuda':
            print_gpu_memory_summary(device)

        print("\n" + "=" * 80)
        print("Profiling Complete!")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Replace DummyForceFieldModel with actual teacher/student models")
        print("2. Use real atomic configurations from MD trajectories")
        print("3. Profile with realistic system sizes (100-500 atoms)")
        print("4. Run long trajectories (10,000+ steps) to check memory stability")
        print("5. Use nsys for detailed GPU kernel analysis")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
