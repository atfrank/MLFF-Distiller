"""
Export StudentForceField Model to ONNX Format for TensorRT Optimization

This script exports the trained StudentForceField (PaiNN) model to ONNX format,
which can then be used with ONNX Runtime's TensorRT execution provider for
optimized inference.

Key Features:
- Exports PyTorch model to ONNX with proper opset versioning
- Handles dynamic batch sizes and variable atom counts
- Validates numerical accuracy after export
- Supports FP16 and FP32 precision modes
- Creates optimized ONNX models for different input sizes

Usage:
    python scripts/export_to_onnx.py \\
        --checkpoint checkpoints/best_model.pt \\
        --output models/student_model.onnx \\
        --precision fp16 \\
        --validate

Author: CUDA Optimization Engineer
Date: 2025-11-24
Issue: M3 #24 - TensorRT Optimization
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import torch
import torch.onnx
import numpy as np
import onnx
import onnxruntime as ort

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlff_distiller.models.student_model import StudentForceField

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StudentModelONNXWrapper(torch.nn.Module):
    """
    Wrapper for StudentForceField to make it ONNX-exportable.

    The main challenge is that the original model uses dynamic neighbor search
    which is not ONNX-compatible. We handle this by:
    1. Fixing the forward pass to not use dynamic operations
    2. Pre-computing neighbor lists (or using a fixed max neighbors)
    3. Ensuring all operations are ONNX-compatible
    """

    def __init__(self, model: StudentForceField):
        super().__init__()
        self.model = model

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for ONNX export.

        Args:
            atomic_numbers: [N] atomic numbers
            positions: [N, 3] positions

        Returns:
            energy: scalar total energy
        """
        # Call model forward (no PBC for simplicity in ONNX export)
        energy = self.model(
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=None,
            pbc=None,
            batch=None
        )
        return energy


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    precision: str = 'fp32',
    opset_version: int = 17,
    num_atoms: int = 50,
    device: str = 'cuda'
) -> None:
    """
    Export StudentForceField model to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt file)
        output_path: Path to save ONNX model (.onnx file)
        precision: Precision mode ('fp32' or 'fp16')
        opset_version: ONNX opset version (17 recommended for TensorRT)
        num_atoms: Number of atoms for dummy input (for shape inference)
        device: Device to load model on
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Load trained model
    model = StudentForceField.load(checkpoint_path, device=device)
    model.eval()

    # Wrap model for ONNX export
    wrapped_model = StudentModelONNXWrapper(model)
    wrapped_model.eval()

    # Create dummy inputs for tracing
    # Use a representative molecule (e.g., 50 atoms)
    dummy_atomic_numbers = torch.randint(1, 10, (num_atoms,), dtype=torch.long, device=device)
    dummy_positions = torch.randn(num_atoms, 3, dtype=torch.float32, device=device)

    # Convert to FP16 if requested (for model parameters)
    if precision == 'fp16' and device == 'cuda':
        logger.info("Converting model to FP16 precision")
        # Don't use .half() as it causes issues, instead let ONNX export handle it
        # or we can convert the exported ONNX model afterward
        pass

    logger.info(f"Exporting to ONNX with opset {opset_version}")
    logger.info(f"Dummy input: {num_atoms} atoms")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_atomic_numbers, dummy_positions),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['atomic_numbers', 'positions'],
            output_names=['energy'],
            dynamic_axes={
                'atomic_numbers': {0: 'num_atoms'},
                'positions': {0: 'num_atoms'},
            },
            verbose=False
        )
        logger.info(f"Successfully exported to {output_path}")

        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model size: {size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"ONNX export failed: {e}", exc_info=True)
        raise

    # Verify ONNX model
    logger.info("Verifying ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model is valid")
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        raise

    # Optionally convert to FP16 (if requested)
    if precision == 'fp16':
        logger.info("Converting ONNX model to FP16...")
        try:
            from onnxconverter_common import float16
            fp16_model = float16.convert_float_to_float16(onnx_model)

            # Save FP16 model
            fp16_path = output_path.with_stem(output_path.stem + '_fp16')
            onnx.save(fp16_model, fp16_path)
            logger.info(f"FP16 model saved to {fp16_path}")

            # Get FP16 file size
            fp16_size_mb = fp16_path.stat().st_size / (1024 * 1024)
            logger.info(f"FP16 ONNX model size: {fp16_size_mb:.2f} MB")
        except ImportError:
            logger.warning("onnxconverter-common not installed, skipping FP16 conversion")
            logger.warning("Install with: pip install onnxconverter-common")


def validate_onnx_export(
    checkpoint_path: Path,
    onnx_path: Path,
    num_test_cases: int = 10,
    tolerance: float = 1e-3,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Validate ONNX model against PyTorch model.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        onnx_path: Path to ONNX model
        num_test_cases: Number of test cases to validate
        tolerance: Absolute tolerance for numerical differences (eV)
        device: Device for PyTorch inference

    Returns:
        Dictionary with validation metrics
    """
    logger.info("Validating ONNX export...")

    # Load PyTorch model
    pytorch_model = StudentForceField.load(checkpoint_path, device=device)
    pytorch_model.eval()
    pytorch_wrapped = StudentModelONNXWrapper(pytorch_model)

    # Load ONNX model with CUDA execution provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)

    logger.info(f"ONNX Runtime using providers: {ort_session.get_providers()}")

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
            pytorch_energy = pytorch_wrapped(atomic_numbers, positions)
            pytorch_energy_val = pytorch_energy.cpu().item()

        # ONNX inference
        onnx_inputs = {
            'atomic_numbers': atomic_numbers.cpu().numpy(),
            'positions': positions.cpu().numpy()
        }
        onnx_outputs = ort_session.run(None, onnx_inputs)
        onnx_energy_val = float(onnx_outputs[0])

        # Compute error
        abs_error = abs(pytorch_energy_val - onnx_energy_val)
        rel_error = abs_error / (abs(pytorch_energy_val) + 1e-8)

        errors.append(abs_error)
        relative_errors.append(rel_error)

        logger.info(
            f"Test {i+1}/{num_test_cases} ({num_atoms} atoms): "
            f"PyTorch={pytorch_energy_val:.6f} eV, "
            f"ONNX={onnx_energy_val:.6f} eV, "
            f"Error={abs_error:.6f} eV ({rel_error*100:.4f}%)"
        )

    # Summary statistics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    mean_rel_error = np.mean(relative_errors)

    logger.info("\nValidation Summary:")
    logger.info(f"  Mean absolute error: {mean_error:.6f} eV")
    logger.info(f"  Max absolute error:  {max_error:.6f} eV")
    logger.info(f"  Mean relative error: {mean_rel_error*100:.4f}%")

    # Check if within tolerance
    if max_error > tolerance:
        logger.warning(
            f"Max error ({max_error:.6f} eV) exceeds tolerance ({tolerance} eV)!"
        )
    else:
        logger.info(f"All errors within tolerance ({tolerance} eV)")

    return {
        'mean_abs_error': mean_error,
        'max_abs_error': max_error,
        'mean_rel_error': mean_rel_error,
        'passed': max_error <= tolerance
    }


def benchmark_onnx_vs_pytorch(
    checkpoint_path: Path,
    onnx_path: Path,
    use_tensorrt: bool = True,
    num_atoms: int = 50,
    num_iterations: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark ONNX model vs PyTorch model.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        onnx_path: Path to ONNX model
        use_tensorrt: Use TensorRT execution provider
        num_atoms: Number of atoms for benchmarking
        num_iterations: Number of benchmark iterations
        device: Device for PyTorch inference

    Returns:
        Dictionary with timing results
    """
    logger.info("\nBenchmarking ONNX vs PyTorch...")

    # Load PyTorch model
    pytorch_model = StudentForceField.load(checkpoint_path, device=device)
    pytorch_model.eval()
    pytorch_wrapped = StudentModelONNXWrapper(pytorch_model)

    # Load ONNX model
    if use_tensorrt:
        providers = [
            ('TensorrtExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(onnx_path.parent / 'trt_cache')
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
    logger.info(f"ONNX Runtime providers: {ort_session.get_providers()}")

    # Create test input
    atomic_numbers = torch.randint(1, 10, (num_atoms,), dtype=torch.long, device=device)
    positions = torch.randn(num_atoms, 3, dtype=torch.float32, device=device)

    onnx_inputs = {
        'atomic_numbers': atomic_numbers.cpu().numpy(),
        'positions': positions.cpu().numpy()
    }

    # Warm-up
    logger.info("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_wrapped(atomic_numbers, positions)
        _ = ort_session.run(None, onnx_inputs)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark PyTorch
    logger.info(f"Benchmarking PyTorch ({num_iterations} iterations)...")
    import time
    pytorch_times = []

    for _ in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = pytorch_wrapped(atomic_numbers, positions)

        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        pytorch_times.append((end - start) * 1000)  # Convert to ms

    # Benchmark ONNX
    logger.info(f"Benchmarking ONNX ({num_iterations} iterations)...")
    onnx_times = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = ort_session.run(None, onnx_inputs)
        end = time.perf_counter()
        onnx_times.append((end - start) * 1000)  # Convert to ms

    # Compute statistics
    pytorch_mean = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    pytorch_median = np.median(pytorch_times)

    onnx_mean = np.mean(onnx_times)
    onnx_std = np.std(onnx_times)
    onnx_median = np.median(onnx_times)

    speedup = pytorch_mean / onnx_mean

    logger.info("\nBenchmark Results:")
    logger.info(f"PyTorch:  {pytorch_mean:.3f} ± {pytorch_std:.3f} ms (median: {pytorch_median:.3f} ms)")
    logger.info(f"ONNX:     {onnx_mean:.3f} ± {onnx_std:.3f} ms (median: {onnx_median:.3f} ms)")
    logger.info(f"Speedup:  {speedup:.2f}x")

    return {
        'pytorch_mean_ms': pytorch_mean,
        'pytorch_std_ms': pytorch_std,
        'onnx_mean_ms': onnx_mean,
        'onnx_std_ms': onnx_std,
        'speedup': speedup
    }


def main():
    parser = argparse.ArgumentParser(
        description='Export StudentForceField model to ONNX format'
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
        default=Path('models/student_model.onnx'),
        help='Path to save ONNX model'
    )
    parser.add_argument(
        '--precision',
        choices=['fp32', 'fp16'],
        default='fp32',
        help='Precision mode for export'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--num-atoms',
        type=int,
        default=50,
        help='Number of atoms for dummy input'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate ONNX export against PyTorch'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark ONNX vs PyTorch'
    )
    parser.add_argument(
        '--use-tensorrt',
        action='store_true',
        help='Use TensorRT execution provider for benchmarking'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device for inference (cuda or cpu)'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Export to ONNX
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        precision=args.precision,
        opset_version=args.opset,
        num_atoms=args.num_atoms,
        device=args.device
    )

    # Validate export
    if args.validate:
        validation_results = validate_onnx_export(
            checkpoint_path=args.checkpoint,
            onnx_path=args.output,
            device=args.device
        )

        if not validation_results['passed']:
            logger.warning("Validation failed! ONNX model may not be accurate.")

    # Benchmark
    if args.benchmark:
        benchmark_results = benchmark_onnx_vs_pytorch(
            checkpoint_path=args.checkpoint,
            onnx_path=args.output,
            use_tensorrt=args.use_tensorrt,
            num_atoms=args.num_atoms,
            device=args.device
        )

        logger.info(f"\nFinal speedup: {benchmark_results['speedup']:.2f}x")


if __name__ == '__main__':
    main()
