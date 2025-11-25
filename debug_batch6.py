#!/usr/bin/env python3
"""
Debug - check neighbor list generation.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from ase.build import molecule

# Add src to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import radius_graph_native

# Create 3 identical CH4 molecules
molecules = [molecule('CH4'), molecule('CH4'), molecule('CH4')]

# Prepare batch data
atomic_numbers_list = []
positions_list = []
batch_idx_list = []

for i, mol in enumerate(molecules):
    n_atoms = len(mol)
    atomic_numbers_list.append(torch.tensor(mol.get_atomic_numbers(), dtype=torch.long))
    positions_list.append(torch.tensor(mol.get_positions(), dtype=torch.float32))
    batch_idx_list.append(torch.full((n_atoms,), i, dtype=torch.long))

# Concatenate
atomic_numbers = torch.cat(atomic_numbers_list)
positions = torch.cat(positions_list)
batch_idx = torch.cat(batch_idx_list)

print(f"Batch data:")
print(f"  Total atoms: {len(atomic_numbers)}")
print(f"  Batch indices: {batch_idx}")
print(f"  Positions shape: {positions.shape}")

# Generate neighbor list
edge_index = radius_graph_native(
    positions,
    r=5.0,  # cutoff
    batch=batch_idx,
    loop=False
)

print(f"\nEdge index:")
print(f"  Shape: {edge_index.shape}")
print(f"  Num edges: {edge_index.shape[1]}")

# Check if there are cross-batch edges
src, dst = edge_index
batch_src = batch_idx[src]
batch_dst = batch_idx[dst]
cross_batch_edges = (batch_src != batch_dst).sum().item()

print(f"\nCross-batch edges: {cross_batch_edges}")

if cross_batch_edges > 0:
    print("ERROR: Found edges connecting atoms from different structures!")
    # Show some examples
    mask = batch_src != batch_dst
    print(f"  Examples:")
    for i in range(min(10, cross_batch_edges)):
        idx = torch.where(mask)[0][i]
        print(f"    Edge {i}: atom {src[idx].item()} (batch {batch_src[idx].item()}) -> "
              f"atom {dst[idx].item()} (batch {batch_dst[idx].item()})")
else:
    print("SUCCESS: No cross-batch edges found!")

# Count edges per structure
for i in range(len(molecules)):
    mask = batch_src == i
    n_edges = mask.sum().item()
    print(f"  Structure {i}: {n_edges} edges")
