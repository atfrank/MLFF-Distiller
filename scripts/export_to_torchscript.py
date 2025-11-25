"""
Export StudentForceField Model to TorchScript for Optimized Inference

This script exports the trained StudentForceField to TorchScript format using
torch.jit.script or torch.jit.trace. TorchScript provides:
- Faster inference than eager PyTorch
- Better kernel fusion
- Can be further optimized with CUDA optimizations
- No dependency on Python interpreter at runtime

This is a more reliable alternative to ONNX export for models with dynamic
operations like neighbor search.

Usage:
    python scripts/export_to_torchscript.py \\
        --checkpoint checkpoints/best_model.pt \\
        --output models/student_model_jit.pt \\
        --method trace \\
        --validate

Author: CUDA Optimization Engineer
Date: 2025-11-24
Issue: M3 #24 - TensorRT Optimization (TorchScript alternative)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import sys
import time

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlff_distiller.models.student_model import StudentForceField

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_to_torchscript(
    checkpoint_path: Path,
    output_path: Path,
    method: str = 'trace',
    num_atoms: int = 50,
    device: str = 'cuda',
    optimize: bool = True
) -> torch.jit.ScriptModule:
    """
    Export StudentForceField model to TorchScript format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt file)
        output_path: Path to save TorchScript model (.pt file)
        method: Export method ('trace' or 'script')
        num_atoms: Number of atoms for dummy input (for tracing)
        device: Device to load model on
        optimize: Apply optimization passes

    Returns:
        TorchScript model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Load trained model
    model = StudentForceField.load(checkpoint_path, device=device)
    model.eval()

    logger.info(f"Exporting to TorchScript using method: {method}")

    if method == 'trace':
        # Create a wrapper that has a simpler signature for tracing
        class SimpleWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model

            def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                return self.model(atomic_numbers, positions, None, None, None)

        wrapped_model = SimpleWrapper(model)
        wrapped_model.eval()

        # Create dummy inputs for tracing
        dummy_atomic_numbers = torch.randint(1, 10, (num_atoms,), dtype=torch.long, device=device)
        dummy_positions = torch.randn(num_atoms, 3, dtype=torch.float32, device=device)

        logger.info(f"Tracing with {num_atoms} atoms")

        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapped_model,
                (dummy_atomic_numbers, dummy_positions)
            )

        # Optimize if requested
        if optimize:
            logger.info("Applying optimization passes...")
            traced_model = torch.jit.optimize_for_inference(traced_model)

        jit_model = traced_model

    elif method == 'script':
        # Script the model (preserves control flow)
        logger.info("Scripting model...")
        try:
            jit_model = torch.jit.script(model)

            # Optimize if requested
            if optimize:
                logger.info("Applying optimization passes...")
                jit_model = torch.jit.optimize_for_inference(jit_model)

        except Exception as e:
            logger.error(f"Scripting failed: {e}")
            logger.info("Falling back to trace method...")

            # Fallback to trace with wrapper
            class SimpleWrapper(torch.nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.model = base_model

                def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                    return self.model(atomic_numbers, positions, None, None, None)

            wrapped_model = SimpleWrapper(model)
            wrapped_model.eval()

            dummy_atomic_numbers = torch.randint(1, 10, (num_atoms,), dtype=torch.long, device=device)
            dummy_positions = torch.randn(num_atoms, 3, dtype=torch.float32, device=device)

            with torch.no_grad():
                jit_model = torch.jit.trace(
                    wrapped_model,
                    (dummy_atomic_numbers, dummy_positions)
                )

            if optimize:
                jit_model = torch.jit.optimize_for_inference(jit_model)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving TorchScript model to {output_path}")

    torch.jit.save(jit_model, output_path)

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"TorchScript model size: {size_mb:.2f} MB")

    return jit_model


def validate_torchscript_export(
    checkpoint_path: Path,
    jit_path: Path,
    num_test_cases: int = 10,
    tolerance: float = 1e-5,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Validate TorchScript model against PyTorch model.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        jit_path: Path to TorchScript model
        num_test_cases: Number of test cases to validate
        tolerance: Absolute tolerance for numerical differences (eV)
        device: Device for inference

    Returns:
        Dictionary with validation metrics
    """
    logger.info("Validating TorchScript export...")

    # Load PyTorch model
    pytorch_model = StudentForceField.load(checkpoint_path, device=device)
    pytorch_model.eval()

    # Load TorchScript model
    jit_model = torch.jit.load(jit_path, map_location=device)
    jit_model.eval()

    # Test on random molecules of various sizes
    test_sizes = [10, 20, 30, 50, 75, 100]
    errors = []
    relative_errors = []

    for i, num_atoms in enumerate(test_sizes[:num_test_cases]):
        # Generate random test input
        atomic_numbers = torch.randint(1, 10, (num_atoms,), dtype=torch.long, device=device)
        positions = torch.randn(num_atoms, 3, dtype=torch.float32, device=device)

        # PyTorch inference
        with torch.no_grad():
            pytorch_energy = pytorch_model(atomic_numbers, positions, None, None, None)
            pytorch_energy_val = pytorch_energy.cpu().item()

        # TorchScript inference (simpler interface)
        with torch.no_grad():
            jit_energy = jit_model(atomic_numbers, positions)
            jit_energy_val = jit_energy.cpu().item()

        # Compute error
        abs_error = abs(pytorch_energy_val - jit_energy_val)
        rel_error = abs_error / (abs(pytorch_energy_val) + 1e-8)

        errors.append(abs_error)
        relative_errors.append(rel_error)

        logger.info(
            f"Test {i+1}/{num_test_cases} ({num_atoms} atoms): "
            f"PyTorch={pytorch_energy_val:.6f} eV, "
            f"TorchScript={jit_energy_val:.6f} eV, "
            f"Error={abs_error:.6e} eV ({rel_error*100:.6f}%)"
        )

    # Summary statistics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    mean_rel_error = np.mean(relative_errors)

    logger.info("\nValidation Summary:")
    logger.info(f"  Mean absolute error: {mean_error:.6e} eV")
    logger.info(f"  Max absolute error:  {max_error:.6e} eV")
    logger.info(f"  Mean relative error: {mean_rel_error*100:.6f}%")

    # Check if within tolerance
    if max_error > tolerance:
        logger.warning(
            f"Max error ({max_error:.6e} eV) exceeds tolerance ({tolerance} eV)!"
        )
    else:
        logger.info(f"All errors within tolerance ({tolerance} eV)")

    return {
        'mean_abs_error': mean_error,
        'max_abs_error': max_error,
        'mean_rel_error': mean_rel_error,
        'passed': max_error <= tolerance
    }


def benchmark_torchscript_vs_pytorch(
    checkpoint_path: Path,
    jit_path: Path,
    num_atoms: int = 50,
    num_iterations: int = 100,
    device: str = 'cuda',
    use_fp16: bool = False
) -> Dict[str, float]:
    """
    Benchmark TorchScript model vs PyTorch model.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        jit_path: Path to TorchScript model
        num_atoms: Number of atoms for benchmarking
        num_iterations: Number of benchmark iterations
        device: Device for inference
        use_fp16: Use FP16 mixed precision

    Returns:
        Dictionary with timing results
    """
    logger.info("\nBenchmarking TorchScript vs PyTorch...")

    # Load PyTorch model
    pytorch_model = StudentForceField.load(checkpoint_path, device=device)
    pytorch_model.eval()

    # Load TorchScript model
    jit_model = torch.jit.load(jit_path, map_location=device)
    jit_model.eval()

    # Create test input
    atomic_numbers = torch.randint(1, 10, (num_atoms,), dtype=torch.long, device=device)
    positions = torch.randn(num_atoms, 3, dtype=torch.float32, device=device)

    # Warm-up
    logger.info("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model(atomic_numbers, positions, None, None, None)
            _ = jit_model(atomic_numbers, positions)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark PyTorch
    logger.info(f"Benchmarking PyTorch ({num_iterations} iterations)...")
    pytorch_times = []

    for _ in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            if use_fp16 and device == 'cuda':
                with torch.amp.autocast('cuda'):
                    _ = pytorch_model(atomic_numbers, positions, None, None, None)
            else:
                _ = pytorch_model(atomic_numbers, positions, None, None, None)

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        pytorch_times.append((end - start) * 1000)  # Convert to ms

    # Benchmark TorchScript
    logger.info(f"Benchmarking TorchScript ({num_iterations} iterations)...")
    jit_times = []

    for _ in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            if use_fp16 and device == 'cuda':
                with torch.amp.autocast('cuda'):
                    _ = jit_model(atomic_numbers, positions)
            else:
                _ = jit_model(atomic_numbers, positions)

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        jit_times.append((end - start) * 1000)  # Convert to ms

    # Compute statistics
    pytorch_mean = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    pytorch_median = np.median(pytorch_times)

    jit_mean = np.mean(jit_times)
    jit_std = np.std(jit_times)
    jit_median = np.median(jit_times)

    speedup = pytorch_mean / jit_mean

    logger.info("\nBenchmark Results:")
    logger.info(f"PyTorch:     {pytorch_mean:.3f} ± {pytorch_std:.3f} ms (median: {pytorch_median:.3f} ms)")
    logger.info(f"TorchScript: {jit_mean:.3f} ± {jit_std:.3f} ms (median: {jit_median:.3f} ms)")
    logger.info(f"Speedup:     {speedup:.2f}x")

    return {
        'pytorch_mean_ms': pytorch_mean,
        'pytorch_std_ms': pytorch_std,
        'jit_mean_ms': jit_mean,
        'jit_std_ms': jit_std,
        'speedup': speedup
    }


def main():
    parser = argparse.ArgumentParser(
        description='Export StudentForceField model to TorchScript format'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/best_model.pt'),
        help='Path to PyTorch checkpoint'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/student_model_jit.pt'),
        help='Path to save TorchScript model'
    )
    parser.add_argument(
        '--method',
        choices=['trace', 'script'],
        default='trace',
        help='Export method (trace or script)'
    )
    parser.add_argument(
        '--num-atoms',
        type=int,
        default=50,
        help='Number of atoms for dummy input (for tracing)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate TorchScript export against PyTorch'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark TorchScript vs PyTorch'
    )
    parser.add_argument(
        '--use-fp16',
        action='store_true',
        help='Use FP16 mixed precision in benchmarking'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device for inference (cuda or cpu)'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Skip optimization passes'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Export to TorchScript
    jit_model = export_to_torchscript(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        method=args.method,
        num_atoms=args.num_atoms,
        device=args.device,
        optimize=not args.no_optimize
    )

    # Validate export
    if args.validate:
        validation_results = validate_torchscript_export(
            checkpoint_path=args.checkpoint,
            jit_path=args.output,
            device=args.device
        )

        if not validation_results['passed']:
            logger.warning("Validation failed! TorchScript model may not be accurate.")

    # Benchmark
    if args.benchmark:
        benchmark_results = benchmark_torchscript_vs_pytorch(
            checkpoint_path=args.checkpoint,
            jit_path=args.output,
            num_atoms=args.num_atoms,
            device=args.device,
            use_fp16=args.use_fp16
        )

        logger.info(f"\nFinal speedup: {benchmark_results['speedup']:.2f}x")


if __name__ == '__main__':
    main()
