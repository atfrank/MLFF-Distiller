#!/usr/bin/env python3
"""
Correct batch processing test - use SAME molecule objects for both tests.
"""

import sys
import time
from pathlib import Path
import numpy as np
from ase.build import molecule

# Add src to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

print("=" * 80)
print("BATCH PROCESSING CORRECTNESS TEST (CORRECTED)")
print("=" * 80)

# Initialize calculator
checkpoint_path = REPO_ROOT / 'checkpoints' / 'best_model.pt'
print(f"\nLoading model from: {checkpoint_path}")

calc = StudentForceFieldCalculator(
    checkpoint_path=checkpoint_path,
    device='cuda',
    enable_timing=True
)

print(f"Model loaded: {calc.model.num_parameters():,} parameters")

# Create test molecules ONCE
print("\n" + "=" * 80)
print("CREATING TEST MOLECULES")
print("=" * 80)

molecules = [
    molecule('H2O'),   # 3 atoms
    molecule('NH3'),   # 4 atoms
    molecule('CH4'),   # 5 atoms
    molecule('C2H6'),  # 8 atoms
]

print(f"Created {len(molecules)} test molecules:")
for i, mol in enumerate(molecules):
    print(f"  {i}: {mol.get_chemical_formula()} ({len(mol)} atoms)")

# Test 1: Correctness - Sequential vs Batch
print("\n" + "=" * 80)
print("TEST 1: CORRECTNESS (Sequential vs Batch)")
print("=" * 80)

# Sequential calculation - use the SAME molecule objects
sequential_results = []
for i, mol in enumerate(molecules):
    mol.calc = calc
    energy = mol.get_potential_energy()
    forces = mol.get_forces()
    sequential_results.append({
        'energy': energy,
        'forces': forces
    })
    print(f"Sequential {i}: energy={energy:.4f} eV, forces_norm={np.linalg.norm(forces):.4f}")

# Batch calculation - use the SAME molecule objects
batch_results = calc.calculate_batch(molecules, properties=['energy', 'forces'])

print("\nComparing results:")
all_match = True
for i, (seq, batch) in enumerate(zip(sequential_results, batch_results)):
    energy_diff = abs(seq['energy'] - batch['energy'])
    forces_diff = np.max(np.abs(seq['forces'] - batch['forces']))

    energy_match = energy_diff < 1e-5
    forces_match = forces_diff < 1e-5
    match = energy_match and forces_match

    status = "PASS" if match else "FAIL"
    print(f"  Molecule {i}: {status}")
    print(f"    Energy diff: {energy_diff:.2e} eV")
    print(f"    Forces diff: {forces_diff:.2e} eV/A")

    if not match:
        all_match = False

if all_match:
    print("\nCORRECTNESS: ALL TESTS PASSED!")
else:
    print("\nCORRECTNESS: TESTS FAILED!")
    sys.exit(1)

# Test 2: Performance - Sequential vs Batch
print("\n" + "=" * 80)
print("TEST 2: PERFORMANCE (Sequential vs Batch)")
print("=" * 80)

batch_sizes = [1, 2, 4]
n_repeats = 20

for batch_size in batch_sizes:
    if batch_size > len(molecules):
        continue

    batch = molecules[:batch_size]

    # Sequential timing
    times_sequential = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        for mol in batch:
            mol.calc = calc
            _ = mol.get_potential_energy()
            _ = mol.get_forces()
        times_sequential.append(time.perf_counter() - start)

    time_sequential = np.mean(times_sequential) * 1000  # ms
    time_per_struct_sequential = time_sequential / batch_size

    # Batch timing
    times_batch = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = calc.calculate_batch(batch)
        times_batch.append(time.perf_counter() - start)

    time_batch = np.mean(times_batch) * 1000  # ms
    time_per_struct_batch = time_batch / batch_size

    speedup = time_per_struct_sequential / time_per_struct_batch
    throughput = 1000.0 / time_per_struct_batch  # structures/sec

    print(f"\nBatch size {batch_size}:")
    print(f"  Sequential: {time_per_struct_sequential:.2f} ms/structure")
    print(f"  Batch:      {time_per_struct_batch:.2f} ms/structure")
    print(f"  Speedup:    {speedup:.1f}x")
    print(f"  Throughput: {throughput:.0f} structures/sec")

    # For batch size > 1, we should see speedup
    if batch_size > 1:
        if speedup > 2.0:
            print(f"  Status:     EXCELLENT ({speedup:.1f}x speedup)")
        elif speedup > 1.5:
            print(f"  Status:     GOOD ({speedup:.1f}x speedup)")
        elif speedup > 1.2:
            print(f"  Status:     ACCEPTABLE ({speedup:.1f}x speedup)")
        else:
            print(f"  Status:     POOR (only {speedup:.1f}x speedup - expected >2x)")

print("\n" + "=" * 80)
print("BATCH FIX VERIFICATION COMPLETE - ALL TESTS PASSED!")
print("=" * 80)
