#!/usr/bin/env python3
"""
Deep-Dive Validation Analysis with PyMOL Visualization

This script performs detailed per-atom error analysis on a trained student model
and generates PyMOL visualization with force vectors rendered as arrows.

Usage:
    python scripts/validate_model_detailed.py --checkpoint checkpoints/best_model.pt

Author: ML Force Field Distiller
Date: 2025-11-24
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from ase import Atoms
from ase.io import write

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.models.distillation_wrapper import DistillationWrapper


def load_model(checkpoint_path: Path, device: str = "cuda") -> DistillationWrapper:
    """Load trained student model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model architecture
    student = StudentForceField(
        num_interactions=3,
        hidden_dim=128,
        num_rbf=20,
        cutoff=5.0,
        max_z=100,
    )

    model = DistillationWrapper(student)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")

    return model


def load_validation_structure(hdf5_path: Path, structure_idx: int = 0) -> Dict[str, np.ndarray]:
    """Load a single structure from the validation dataset."""
    print(f"\nLoading structure {structure_idx} from: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Get structure boundaries
        structures_group = f['structures']
        labels_group = f['labels']

        atom_splits = structures_group['atomic_numbers_splits'][:]
        forces_splits = labels_group['forces_splits'][:]

        # Get data for specific structure
        atom_start_idx = atom_splits[structure_idx]
        atom_end_idx = atom_splits[structure_idx + 1]
        n_atoms = atom_end_idx - atom_start_idx

        forces_start_idx = forces_splits[structure_idx]
        forces_end_idx = forces_splits[structure_idx + 1]

        # Extract structure data
        structure = {
            'atomic_numbers': structures_group['atomic_numbers'][atom_start_idx:atom_end_idx],
            'positions': structures_group['positions'][atom_start_idx:atom_end_idx],  # Already (N, 3)
            'energy': labels_group['energy'][structure_idx],
            'forces': labels_group['forces'][forces_start_idx:forces_end_idx].reshape(-1, 3),
            'cell': structures_group['cells'][structure_idx],
            'pbc': structures_group['pbc'][structure_idx],
        }

        print(f"  Number of atoms: {n_atoms}")
        print(f"  Atomic numbers: {np.unique(structure['atomic_numbers'])}")
        print(f"  Energy: {structure['energy']:.4f} eV")
        print(f"  Force magnitude range: {np.linalg.norm(structure['forces'], axis=1).min():.4f} - {np.linalg.norm(structure['forces'], axis=1).max():.4f} eV/Å")

    return structure


def predict_structure(
    model: DistillationWrapper,
    structure: Dict[str, np.ndarray],
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Run model prediction on a single structure."""
    print("\nRunning model prediction...")

    # Prepare batch (disable PBC to avoid radius_graph issues)
    batch = {
        'atomic_numbers': torch.from_numpy(structure['atomic_numbers']).long().to(device),
        'positions': torch.from_numpy(structure['positions']).float().to(device),
        'cell': torch.from_numpy(structure['cell']).float().unsqueeze(0).to(device),
        'pbc': torch.zeros(3, dtype=torch.bool, device=device).unsqueeze(0),  # Disable PBC
        'batch': torch.zeros(len(structure['atomic_numbers']), dtype=torch.long, device=device),
    }

    # Predict
    with torch.no_grad():
        predictions = model(batch)

    print(f"  Predicted energy: {predictions['energy'].item():.4f} eV")
    print(f"  Predicted force magnitude range: {torch.norm(predictions['forces'], dim=1).min():.4f} - {torch.norm(predictions['forces'], dim=1).max():.4f} eV/Å")

    return predictions


def compute_per_atom_errors(
    structure: Dict[str, np.ndarray],
    predictions: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    """Compute detailed per-atom error metrics."""
    print("\nComputing per-atom error metrics...")

    # Convert to numpy
    pred_forces = predictions['forces'].detach().cpu().numpy()
    true_forces = structure['forces']

    # Overall errors
    energy_error = predictions['energy'].detach().cpu().numpy() - structure['energy']

    # Per-atom force errors
    force_errors = pred_forces - true_forces  # [n_atoms, 3]
    force_magnitudes_true = np.linalg.norm(true_forces, axis=1)
    force_magnitudes_pred = np.linalg.norm(pred_forces, axis=1)
    force_magnitude_errors = force_magnitudes_pred - force_magnitudes_true

    # Component-wise errors
    force_errors_x = force_errors[:, 0]
    force_errors_y = force_errors[:, 1]
    force_errors_z = force_errors[:, 2]

    # RMSE per atom
    force_rmse_per_atom = np.linalg.norm(force_errors, axis=1)

    # Angular errors (cosine similarity)
    # Avoid division by zero
    cos_sim = np.sum(pred_forces * true_forces, axis=1) / (
        force_magnitudes_pred * force_magnitudes_true + 1e-8
    )
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angular_errors = np.arccos(cos_sim) * 180 / np.pi  # Convert to degrees

    errors = {
        'energy_error': energy_error,
        'force_errors': force_errors,
        'force_errors_x': force_errors_x,
        'force_errors_y': force_errors_y,
        'force_errors_z': force_errors_z,
        'force_magnitude_errors': force_magnitude_errors,
        'force_rmse_per_atom': force_rmse_per_atom,
        'force_magnitudes_true': force_magnitudes_true,
        'force_magnitudes_pred': force_magnitudes_pred,
        'angular_errors': angular_errors,
    }

    # Print summary statistics
    print(f"\n  Energy Error: {energy_error:.4f} eV ({abs(energy_error)/len(structure['atomic_numbers'])*1000:.2f} meV/atom)")
    print(f"\n  Force Errors:")
    print(f"    RMSE (per atom): {np.mean(force_rmse_per_atom):.4f} ± {np.std(force_rmse_per_atom):.4f} eV/Å")
    print(f"    Max RMSE: {np.max(force_rmse_per_atom):.4f} eV/Å (atom {np.argmax(force_rmse_per_atom)})")
    print(f"    Min RMSE: {np.min(force_rmse_per_atom):.4f} eV/Å (atom {np.argmin(force_rmse_per_atom)})")
    print(f"\n  Component Errors:")
    print(f"    X: MAE={np.mean(np.abs(force_errors_x)):.4f} eV/Å, RMSE={np.sqrt(np.mean(force_errors_x**2)):.4f} eV/Å")
    print(f"    Y: MAE={np.mean(np.abs(force_errors_y)):.4f} eV/Å, RMSE={np.sqrt(np.mean(force_errors_y**2)):.4f} eV/Å")
    print(f"    Z: MAE={np.mean(np.abs(force_errors_z)):.4f} eV/Å, RMSE={np.sqrt(np.mean(force_errors_z**2)):.4f} eV/Å")
    print(f"\n  Magnitude Errors:")
    print(f"    MAE: {np.mean(np.abs(force_magnitude_errors)):.4f} eV/Å")
    print(f"    RMSE: {np.sqrt(np.mean(force_magnitude_errors**2)):.4f} eV/Å")
    print(f"\n  Angular Errors:")
    print(f"    Mean: {np.mean(angular_errors):.2f}°")
    print(f"    Median: {np.median(angular_errors):.2f}°")
    print(f"    Max: {np.max(angular_errors):.2f}° (atom {np.argmax(angular_errors)})")

    return errors


def create_error_plots(
    structure: Dict[str, np.ndarray],
    predictions: Dict[str, torch.Tensor],
    errors: Dict[str, np.ndarray],
    output_dir: Path
):
    """Create comprehensive error analysis plots."""
    print("\nGenerating error analysis plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert predictions to numpy
    pred_forces = predictions['forces'].cpu().numpy()
    true_forces = structure['forces']
    atomic_numbers = structure['atomic_numbers']

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # 1. Per-atom RMSE distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1a. RMSE histogram
    axes[0, 0].hist(errors['force_rmse_per_atom'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(errors['force_rmse_per_atom']), color='red', linestyle='--', label=f"Mean: {np.mean(errors['force_rmse_per_atom']):.4f}")
    axes[0, 0].axvline(np.median(errors['force_rmse_per_atom']), color='orange', linestyle='--', label=f"Median: {np.median(errors['force_rmse_per_atom']):.4f}")
    axes[0, 0].set_xlabel('Force RMSE (eV/Å)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Per-Atom Force RMSE Distribution')
    axes[0, 0].legend()

    # 1b. RMSE by atom type
    unique_elements = np.unique(atomic_numbers)
    element_names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
    rmse_by_element = []
    element_labels = []
    for z in unique_elements:
        mask = atomic_numbers == z
        rmse_by_element.append(errors['force_rmse_per_atom'][mask])
        element_labels.append(element_names.get(z, f'Z={z}'))

    bp = axes[0, 1].boxplot(rmse_by_element, labels=element_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('skyblue')
    axes[0, 1].set_xlabel('Element')
    axes[0, 1].set_ylabel('Force RMSE (eV/Å)')
    axes[0, 1].set_title('Force RMSE by Element Type')
    axes[0, 1].grid(True, alpha=0.3)

    # 1c. Component-wise errors
    component_errors = [
        errors['force_errors_x'],
        errors['force_errors_y'],
        errors['force_errors_z']
    ]
    bp = axes[1, 0].boxplot(component_errors, labels=['X', 'Y', 'Z'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['red', 'green', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Force Component')
    axes[1, 0].set_ylabel('Error (eV/Å)')
    axes[1, 0].set_title('Component-wise Force Errors')
    axes[1, 0].grid(True, alpha=0.3)

    # 1d. Angular errors
    axes[1, 1].hist(errors['angular_errors'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].axvline(np.mean(errors['angular_errors']), color='red', linestyle='--', label=f"Mean: {np.mean(errors['angular_errors']):.2f}°")
    axes[1, 1].set_xlabel('Angular Error (degrees)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Force Direction Errors')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'error_distribution.png'}")
    plt.close()

    # 2. Force magnitude correlation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 2a. Predicted vs True (all components)
    axes[0].scatter(true_forces.flatten(), pred_forces.flatten(), alpha=0.3, s=10)
    min_val = min(true_forces.min(), pred_forces.min())
    max_val = max(true_forces.max(), pred_forces.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('True Forces (eV/Å)')
    axes[0].set_ylabel('Predicted Forces (eV/Å)')
    axes[0].set_title('Force Component Correlation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2b. Magnitude correlation
    axes[1].scatter(errors['force_magnitudes_true'], errors['force_magnitudes_pred'], alpha=0.5, s=20)
    min_mag = min(errors['force_magnitudes_true'].min(), errors['force_magnitudes_pred'].min())
    max_mag = max(errors['force_magnitudes_true'].max(), errors['force_magnitudes_pred'].max())
    axes[1].plot([min_mag, max_mag], [min_mag, max_mag], 'r--', linewidth=2, label='Perfect')
    axes[1].set_xlabel('True Force Magnitude (eV/Å)')
    axes[1].set_ylabel('Predicted Force Magnitude (eV/Å)')
    axes[1].set_title('Force Magnitude Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 2c. Error vs magnitude
    axes[2].scatter(errors['force_magnitudes_true'], errors['force_rmse_per_atom'], alpha=0.5, s=20, c=atomic_numbers, cmap='viridis')
    axes[2].set_xlabel('True Force Magnitude (eV/Å)')
    axes[2].set_ylabel('Force RMSE (eV/Å)')
    axes[2].set_title('Error vs Force Magnitude')
    axes[2].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
    cbar.set_label('Atomic Number')

    plt.tight_layout()
    plt.savefig(output_dir / 'force_correlations.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'force_correlations.png'}")
    plt.close()

    # 3. Spatial distribution of errors
    fig = plt.figure(figsize=(15, 5))

    positions = structure['positions']

    # 3a. X-Y plane
    ax1 = fig.add_subplot(131)
    sc = ax1.scatter(positions[:, 0], positions[:, 1], c=errors['force_rmse_per_atom'],
                     s=100, cmap='hot', vmin=0)
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_title('Force RMSE (X-Y Plane)')
    ax1.set_aspect('equal')
    plt.colorbar(sc, ax=ax1, label='RMSE (eV/Å)')

    # 3b. X-Z plane
    ax2 = fig.add_subplot(132)
    sc = ax2.scatter(positions[:, 0], positions[:, 2], c=errors['force_rmse_per_atom'],
                     s=100, cmap='hot', vmin=0)
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Z (Å)')
    ax2.set_title('Force RMSE (X-Z Plane)')
    ax2.set_aspect('equal')
    plt.colorbar(sc, ax=ax2, label='RMSE (eV/Å)')

    # 3c. Y-Z plane
    ax3 = fig.add_subplot(133)
    sc = ax3.scatter(positions[:, 1], positions[:, 2], c=errors['force_rmse_per_atom'],
                     s=100, cmap='hot', vmin=0)
    ax3.set_xlabel('Y (Å)')
    ax3.set_ylabel('Z (Å)')
    ax3.set_title('Force RMSE (Y-Z Plane)')
    ax3.set_aspect('equal')
    plt.colorbar(sc, ax=ax3, label='RMSE (eV/Å)')

    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_errors.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'spatial_errors.png'}")
    plt.close()


def create_pymol_visualization(
    structure: Dict[str, np.ndarray],
    predictions: Dict[str, torch.Tensor],
    errors: Dict[str, np.ndarray],
    output_dir: Path
):
    """Create PyMOL visualization with force vectors as arrows."""
    print("\nGenerating PyMOL visualization...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    pred_forces = predictions['forces'].cpu().numpy()
    true_forces = structure['forces']
    atomic_numbers = structure['atomic_numbers']
    positions = structure['positions']

    # Create ASE Atoms object
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=structure['cell'],
        pbc=structure['pbc']
    )

    # Save structure as PDB
    pdb_path = output_dir / 'structure.pdb'
    write(str(pdb_path), atoms)
    print(f"  Saved structure: {pdb_path}")

    # Create PyMOL script
    pymol_script = output_dir / 'visualize_forces.pml'

    # Calculate arrow scaling factor (make arrows visible)
    max_force_magnitude = max(
        np.linalg.norm(true_forces, axis=1).max(),
        np.linalg.norm(pred_forces, axis=1).max()
    )
    scale_factor = 2.0 / max_force_magnitude  # Scale to reasonable size

    with open(pymol_script, 'w') as f:
        f.write("# PyMOL Visualization Script for Force Field Validation\n")
        f.write("# Generated by ML Force Field Distiller\n\n")

        # Load structure
        f.write(f"load {pdb_path.name}\n")
        f.write("hide everything\n")
        f.write("show sticks, structure\n")
        f.write("show spheres, structure\n")
        f.write("set sphere_scale, 0.3\n")
        f.write("set stick_radius, 0.15\n\n")

        # Color by element
        f.write("color gray, structure\n")
        f.write("util.cbaw structure\n\n")

        # Create CGO objects for force vectors
        f.write("# True forces (green arrows)\n")
        f.write("from pymol.cgo import *\n")
        f.write("from pymol import cmd\n\n")

        # Generate true force arrows (green)
        f.write("true_forces = [\n")
        for i, (pos, force) in enumerate(zip(positions, true_forces)):
            end_pos = pos + force * scale_factor
            # Arrow as cylinder + cone
            f.write(f"    CYLINDER, {pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}, "
                   f"{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}, 0.05, "
                   f"0.0, 1.0, 0.0, 0.0, 1.0, 0.0,\n")
            # Cone for arrow head
            cone_base = pos + force * scale_factor * 0.85
            f.write(f"    CONE, {cone_base[0]:.4f}, {cone_base[1]:.4f}, {cone_base[2]:.4f}, "
                   f"{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}, 0.15, 0.0, "
                   f"0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,\n")
        f.write("]\n")
        f.write("cmd.load_cgo(true_forces, 'true_forces')\n\n")

        # Generate predicted force arrows (red)
        f.write("# Predicted forces (red arrows)\n")
        f.write("pred_forces = [\n")
        for i, (pos, force) in enumerate(zip(positions, pred_forces)):
            end_pos = pos + force * scale_factor
            # Arrow as cylinder + cone
            f.write(f"    CYLINDER, {pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}, "
                   f"{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}, 0.05, "
                   f"1.0, 0.0, 0.0, 1.0, 0.0, 0.0,\n")
            # Cone for arrow head
            cone_base = pos + force * scale_factor * 0.85
            f.write(f"    CONE, {cone_base[0]:.4f}, {cone_base[1]:.4f}, {cone_base[2]:.4f}, "
                   f"{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}, 0.15, 0.0, "
                   f"1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,\n")
        f.write("]\n")
        f.write("cmd.load_cgo(pred_forces, 'pred_forces')\n\n")

        # Generate error vectors (yellow, scaled differently)
        f.write("# Error vectors (yellow arrows)\n")
        f.write("error_forces = [\n")
        for i, (pos, error) in enumerate(zip(positions, errors['force_errors'])):
            end_pos = pos + error * scale_factor * 5.0  # Scale up errors for visibility
            # Arrow as cylinder + cone
            f.write(f"    CYLINDER, {pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}, "
                   f"{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}, 0.03, "
                   f"1.0, 1.0, 0.0, 1.0, 1.0, 0.0,\n")
            # Cone for arrow head
            if np.linalg.norm(error) > 0.01:  # Only draw cone for visible errors
                cone_base = pos + error * scale_factor * 5.0 * 0.85
                f.write(f"    CONE, {cone_base[0]:.4f}, {cone_base[1]:.4f}, {cone_base[2]:.4f}, "
                       f"{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}, 0.1, 0.0, "
                       f"1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,\n")
        f.write("]\n")
        f.write("cmd.load_cgo(error_forces, 'error_forces')\n\n")

        # View settings
        f.write("# View settings\n")
        f.write("bg_color white\n")
        f.write("set ray_shadows, 0\n")
        f.write("set depth_cue, 0\n")
        f.write("set antialias, 2\n")
        f.write("set cartoon_fancy_helices, 1\n\n")

        # Labels for legend
        f.write("# Legend\n")
        f.write("cmd.pseudoatom('legend_true', pos=[0, 0, 0], color='green', label='True Forces')\n")
        f.write("cmd.pseudoatom('legend_pred', pos=[0, 0, 0], color='red', label='Predicted Forces')\n")
        f.write("cmd.pseudoatom('legend_error', pos=[0, 0, 0], color='yellow', label='Errors (5x scaled)')\n")
        f.write("hide everything, legend_*\n\n")

        # Orient and zoom
        f.write("orient structure\n")
        f.write("zoom structure, 5\n\n")

        # Save session
        session_path = output_dir / 'force_visualization.pse'
        f.write(f"# Save session\n")
        f.write(f"save {session_path.name}\n\n")

        f.write("# To toggle force arrows:\n")
        f.write("# disable true_forces\n")
        f.write("# disable pred_forces\n")
        f.write("# disable error_forces\n")

    print(f"  Saved PyMOL script: {pymol_script}")
    print(f"\n  To visualize:")
    print(f"    cd {output_dir}")
    print(f"    pymol visualize_forces.pml")

    # Try to run PyMOL if available
    try:
        import subprocess
        result = subprocess.run(
            ['pymol', '-c', str(pymol_script)],
            capture_output=True,
            timeout=30,
            cwd=output_dir
        )
        if result.returncode == 0:
            print(f"  ✓ PyMOL session saved: {session_path}")
        else:
            print(f"  ⚠ PyMOL execution failed (install PyMOL to auto-generate session)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  ℹ PyMOL not found - run script manually to generate session")


def create_detailed_report(
    structure: Dict[str, np.ndarray],
    predictions: Dict[str, torch.Tensor],
    errors: Dict[str, np.ndarray],
    output_dir: Path
):
    """Create detailed markdown report."""
    print("\nGenerating detailed analysis report...")

    report_path = output_dir / 'validation_analysis.md'

    # Element names
    element_names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}

    # Per-element statistics
    unique_elements = np.unique(structure['atomic_numbers'])
    element_stats = []
    for z in unique_elements:
        mask = structure['atomic_numbers'] == z
        element_stats.append({
            'element': element_names.get(z, f'Z={z}'),
            'count': mask.sum(),
            'rmse_mean': errors['force_rmse_per_atom'][mask].mean(),
            'rmse_std': errors['force_rmse_per_atom'][mask].std(),
            'angular_mean': errors['angular_errors'][mask].mean(),
            'angular_max': errors['angular_errors'][mask].max(),
        })

    with open(report_path, 'w') as f:
        f.write("# Deep-Dive Validation Analysis\n\n")
        f.write("**ML Force Field Distiller - Student Model Validation**\n\n")
        f.write(f"Date: 2025-11-24\n\n")
        f.write("---\n\n")

        f.write("## Structure Overview\n\n")
        f.write(f"- **Number of atoms**: {len(structure['atomic_numbers'])}\n")
        f.write(f"- **Elements**: {', '.join([element_names.get(z, f'Z={z}') for z in unique_elements])}\n")
        f.write(f"- **Cell**: {structure['cell'].tolist()}\n")
        f.write(f"- **PBC**: {structure['pbc'].tolist()}\n\n")

        f.write("## Energy Predictions\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| True Energy | {structure['energy']:.4f} eV |\n")
        f.write(f"| Predicted Energy | {predictions['energy'].item():.4f} eV |\n")
        f.write(f"| Absolute Error | {abs(errors['energy_error']):.4f} eV |\n")
        f.write(f"| Error per Atom | {abs(errors['energy_error'])/len(structure['atomic_numbers'])*1000:.2f} meV/atom |\n")
        f.write(f"| Relative Error | {abs(errors['energy_error'])/abs(structure['energy'])*100:.2f}% |\n\n")

        f.write("## Force Predictions - Overall Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Mean RMSE | {np.mean(errors['force_rmse_per_atom']):.4f} eV/Å |\n")
        f.write(f"| Median RMSE | {np.median(errors['force_rmse_per_atom']):.4f} eV/Å |\n")
        f.write(f"| Std Dev RMSE | {np.std(errors['force_rmse_per_atom']):.4f} eV/Å |\n")
        f.write(f"| Max RMSE | {np.max(errors['force_rmse_per_atom']):.4f} eV/Å |\n")
        f.write(f"| Min RMSE | {np.min(errors['force_rmse_per_atom']):.4f} eV/Å |\n\n")

        f.write("## Force Predictions - Component Analysis\n\n")
        f.write(f"| Component | MAE (eV/Å) | RMSE (eV/Å) | Max Error (eV/Å) |\n")
        f.write(f"|-----------|------------|-------------|------------------|\n")
        for comp, name in zip(['force_errors_x', 'force_errors_y', 'force_errors_z'], ['X', 'Y', 'Z']):
            mae = np.mean(np.abs(errors[comp]))
            rmse = np.sqrt(np.mean(errors[comp]**2))
            max_err = np.max(np.abs(errors[comp]))
            f.write(f"| {name} | {mae:.4f} | {rmse:.4f} | {max_err:.4f} |\n")
        f.write("\n")

        f.write("## Force Predictions - By Element Type\n\n")
        f.write(f"| Element | Count | Mean RMSE (eV/Å) | Std Dev (eV/Å) | Mean Angular Error (°) | Max Angular Error (°) |\n")
        f.write(f"|---------|-------|------------------|----------------|----------------------|---------------------|\n")
        for stat in element_stats:
            f.write(f"| {stat['element']} | {stat['count']} | {stat['rmse_mean']:.4f} | "
                   f"{stat['rmse_std']:.4f} | {stat['angular_mean']:.2f} | {stat['angular_max']:.2f} |\n")
        f.write("\n")

        f.write("## Directional Accuracy\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Mean Angular Error | {np.mean(errors['angular_errors']):.2f}° |\n")
        f.write(f"| Median Angular Error | {np.median(errors['angular_errors']):.2f}° |\n")
        f.write(f"| Max Angular Error | {np.max(errors['angular_errors']):.2f}° |\n")
        f.write(f"| Atoms with <5° error | {(errors['angular_errors'] < 5).sum()} / {len(errors['angular_errors'])} ({(errors['angular_errors'] < 5).sum()/len(errors['angular_errors'])*100:.1f}%) |\n")
        f.write(f"| Atoms with <10° error | {(errors['angular_errors'] < 10).sum()} / {len(errors['angular_errors'])} ({(errors['angular_errors'] < 10).sum()/len(errors['angular_errors'])*100:.1f}%) |\n")
        f.write(f"| Atoms with <30° error | {(errors['angular_errors'] < 30).sum()} / {len(errors['angular_errors'])} ({(errors['angular_errors'] < 30).sum()/len(errors['angular_errors'])*100:.1f}%) |\n\n")

        f.write("## Top 10 Worst Force Predictions\n\n")
        worst_indices = np.argsort(errors['force_rmse_per_atom'])[-10:][::-1]
        f.write(f"| Rank | Atom Index | Element | Position (Å) | RMSE (eV/Å) | Angular Error (°) |\n")
        f.write(f"|------|------------|---------|--------------|-------------|------------------|\n")
        for rank, idx in enumerate(worst_indices, 1):
            element = element_names.get(structure['atomic_numbers'][idx], f"Z={structure['atomic_numbers'][idx]}")
            pos = structure['positions'][idx]
            rmse = errors['force_rmse_per_atom'][idx]
            angular = errors['angular_errors'][idx]
            f.write(f"| {rank} | {idx} | {element} | ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | {rmse:.4f} | {angular:.2f} |\n")
        f.write("\n")

        f.write("## Visualization Files\n\n")
        f.write("Generated files:\n")
        f.write("- `error_distribution.png` - Distribution of per-atom errors\n")
        f.write("- `force_correlations.png` - Correlation between predicted and true forces\n")
        f.write("- `spatial_errors.png` - Spatial distribution of force errors\n")
        f.write("- `structure.pdb` - Structure file for PyMOL\n")
        f.write("- `visualize_forces.pml` - PyMOL script to visualize force vectors\n")
        f.write("- `force_visualization.pse` - PyMOL session file (if PyMOL is installed)\n\n")

        f.write("### PyMOL Visualization\n\n")
        f.write("To view the force vectors:\n")
        f.write("```bash\n")
        f.write(f"cd {output_dir}\n")
        f.write("pymol visualize_forces.pml\n")
        f.write("```\n\n")
        f.write("**Color coding:**\n")
        f.write("- 🟢 **Green arrows**: True forces from teacher model\n")
        f.write("- 🔴 **Red arrows**: Predicted forces from student model\n")
        f.write("- 🟡 **Yellow arrows**: Error vectors (5x scaled for visibility)\n\n")

        f.write("---\n\n")
        f.write("**Generated by ML Force Field Distiller validation pipeline**\n")

    print(f"  Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Deep-dive validation analysis with PyMOL visualization"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(REPO_ROOT / 'checkpoints/best_model.pt'),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(REPO_ROOT / 'data/merged_dataset_4883/merged_dataset.h5'),
        help='Path to validation dataset'
    )
    parser.add_argument(
        '--structure-idx',
        type=int,
        default=0,
        help='Index of structure to analyze (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(REPO_ROOT / 'validation_analysis'),
        help='Output directory for analysis'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )

    args = parser.parse_args()

    print("="*70)
    print("DEEP-DIVE VALIDATION ANALYSIS")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(Path(args.checkpoint), device=args.device)

    # Load structure
    structure = load_validation_structure(Path(args.data), args.structure_idx)

    # Predict
    predictions = predict_structure(model, structure, device=args.device)

    # Compute errors
    errors = compute_per_atom_errors(structure, predictions)

    # Create visualizations
    create_error_plots(structure, predictions, errors, output_dir)
    create_pymol_visualization(structure, predictions, errors, output_dir)
    create_detailed_report(structure, predictions, errors, output_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nTo view PyMOL visualization:")
    print(f"  cd {output_dir}")
    print(f"  pymol visualize_forces.pml")
    print()


if __name__ == "__main__":
    main()
