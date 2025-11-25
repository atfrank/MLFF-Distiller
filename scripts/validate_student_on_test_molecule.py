#!/usr/bin/env python3
"""
Validate Student Model on Test MolDiff Molecule

This script validates the student model by comparing its predictions
against the teacher (Orb-v2) on an unseen MolDiff test molecule.

Usage:
    python scripts/validate_student_on_test_molecule.py \
        --checkpoint checkpoints/best_model.pt \
        --test-molecule data/generative_test/moldiff/test_10mols_20251123_181225_SDF/0.sdf

Author: ML Distillation Project
Date: 2025-11-24
"""

import sys
import argparse
from pathlib import Path
import logging

import numpy as np
import torch
from ase.io import read

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.data.sdf_utils import read_structure_with_hydrogen_support, check_hydrogen_content
from mlff_distiller.models.teacher_wrappers import OrbCalculator
from mlff_distiller.models.student_model import StudentForceField


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_student_model(checkpoint_path: Path, device: str):
    """Load student model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same architecture as training (128D hidden, 3 interactions, 5.0Å cutoff, max_z=100)
    model = StudentForceField(
        hidden_dim=128,
        num_interactions=3,
        num_rbf=20,
        cutoff=5.0,
        max_z=100
    ).to(device)

    # Load weights - handle "model." prefix from DistillationWrapper
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('model.') for k in state_dict.keys()):
        # Remove "model." prefix
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_with_student(model, atoms, device: str):
    """Get predictions from student model."""
    # Get atomic numbers and positions
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    positions.requires_grad_(True)  # Needed for force computation

    # Create batch index (all atoms in same system)
    batch = torch.zeros(len(atomic_numbers), dtype=torch.long, device=device)

    # Forward pass for energy (no grad on model params, but keep grad on positions)
    energy = model(atomic_numbers, positions, cell=None, pbc=None, batch=batch)

    # Compute forces via autograd
    forces = -torch.autograd.grad(
        energy.sum(),
        positions,
        create_graph=False,
        retain_graph=False
    )[0]

    return energy.item(), forces.cpu().numpy()


def compute_metrics(teacher_energy, teacher_forces, student_energy, student_forces):
    """Compute detailed comparison metrics."""
    metrics = {}

    # Energy metrics
    metrics['energy_error'] = abs(student_energy - teacher_energy)
    metrics['energy_rel_error'] = 100 * metrics['energy_error'] / max(abs(teacher_energy), 1e-8)

    # Force metrics
    force_diff = student_forces - teacher_forces
    metrics['force_mae'] = np.mean(np.abs(force_diff))
    metrics['force_rmse'] = np.sqrt(np.mean(force_diff**2))
    metrics['force_max_error'] = np.max(np.abs(force_diff))

    # Force vector RMSE
    force_mag_teacher = np.linalg.norm(teacher_forces, axis=1)
    force_mag_student = np.linalg.norm(student_forces, axis=1)
    metrics['force_magnitude_rmse'] = np.sqrt(np.mean((force_mag_teacher - force_mag_student)**2))

    # Angular error (cosine similarity)
    dot_products = np.sum(teacher_forces * student_forces, axis=1)
    norms_teacher = np.linalg.norm(teacher_forces, axis=1)
    norms_student = np.linalg.norm(student_forces, axis=1)

    # Avoid division by zero
    valid_mask = (norms_teacher > 1e-6) & (norms_student > 1e-6)
    if np.any(valid_mask):
        cos_sim = dot_products[valid_mask] / (norms_teacher[valid_mask] * norms_student[valid_mask])
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angular_errors_rad = np.arccos(cos_sim)
        angular_errors_deg = np.degrees(angular_errors_rad)
        metrics['angular_error_mean'] = np.mean(angular_errors_deg)
        metrics['angular_error_median'] = np.median(angular_errors_deg)
        metrics['angular_error_max'] = np.max(angular_errors_deg)
        metrics['cosine_similarity_mean'] = np.mean(cos_sim)
    else:
        metrics['angular_error_mean'] = 0.0
        metrics['angular_error_median'] = 0.0
        metrics['angular_error_max'] = 0.0
        metrics['cosine_similarity_mean'] = 1.0

    # Per-atom force errors
    per_atom_force_errors = np.linalg.norm(force_diff, axis=1)
    metrics['force_error_per_atom_mean'] = np.mean(per_atom_force_errors)
    metrics['force_error_per_atom_median'] = np.median(per_atom_force_errors)
    metrics['force_error_per_atom_max'] = np.max(per_atom_force_errors)

    return metrics


def print_results(atoms, teacher_energy, teacher_forces, student_energy, student_forces, metrics, logger):
    """Print detailed validation results."""
    h_stats = check_hydrogen_content(atoms)

    logger.info("=" * 80)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nMolecule Information:")
    logger.info(f"  Total atoms: {len(atoms)}")
    logger.info(f"  Hydrogen atoms: {h_stats['n_hydrogen']} ({h_stats['h_percentage']:.1f}%)")
    logger.info(f"  Heavy atoms: {h_stats['n_atoms'] - h_stats['n_hydrogen']}")
    logger.info(f"  Composition: {atoms.get_chemical_formula()}")

    logger.info(f"\nEnergy Predictions:")
    logger.info(f"  Teacher (Orb-v2): {teacher_energy:.6f} eV")
    logger.info(f"  Student (PaiNN):  {student_energy:.6f} eV")
    logger.info(f"  Absolute error:   {metrics['energy_error']:.6f} eV")
    logger.info(f"  Relative error:   {metrics['energy_rel_error']:.2f}%")

    logger.info(f"\nForce Predictions:")
    logger.info(f"  Force MAE:        {metrics['force_mae']:.6f} eV/Å")
    logger.info(f"  Force RMSE:       {metrics['force_rmse']:.6f} eV/Å")
    logger.info(f"  Max force error:  {metrics['force_max_error']:.6f} eV/Å")
    logger.info(f"  Force mag RMSE:   {metrics['force_magnitude_rmse']:.6f} eV/Å")

    logger.info(f"\nAngular Accuracy (Force Direction):")
    logger.info(f"  Mean angle error:   {metrics['angular_error_mean']:.2f}°")
    logger.info(f"  Median angle error: {metrics['angular_error_median']:.2f}°")
    logger.info(f"  Max angle error:    {metrics['angular_error_max']:.2f}°")
    logger.info(f"  Mean cosine sim:    {metrics['cosine_similarity_mean']:.4f}")

    logger.info(f"\nPer-Atom Force Errors:")
    logger.info(f"  Mean:   {metrics['force_error_per_atom_mean']:.6f} eV/Å")
    logger.info(f"  Median: {metrics['force_error_per_atom_median']:.6f} eV/Å")
    logger.info(f"  Max:    {metrics['force_error_per_atom_max']:.6f} eV/Å")

    # Teacher force statistics
    teacher_force_norms = np.linalg.norm(teacher_forces, axis=1)
    logger.info(f"\nTeacher Force Statistics:")
    logger.info(f"  Mean magnitude: {np.mean(teacher_force_norms):.4f} eV/Å")
    logger.info(f"  Max magnitude:  {np.max(teacher_force_norms):.4f} eV/Å")
    logger.info(f"  Min magnitude:  {np.min(teacher_force_norms):.4f} eV/Å")

    # Student force statistics
    student_force_norms = np.linalg.norm(student_forces, axis=1)
    logger.info(f"\nStudent Force Statistics:")
    logger.info(f"  Mean magnitude: {np.mean(student_force_norms):.4f} eV/Å")
    logger.info(f"  Max magnitude:  {np.max(student_force_norms):.4f} eV/Å")
    logger.info(f"  Min magnitude:  {np.min(student_force_norms):.4f} eV/Å")

    logger.info("=" * 80)

    # Quality assessment
    logger.info("\nQUALITY ASSESSMENT")
    logger.info("=" * 80)

    quality_score = 0
    max_score = 0

    # Energy accuracy (20 points)
    max_score += 20
    if metrics['energy_rel_error'] < 1.0:
        quality_score += 20
        logger.info("✓ Energy accuracy: EXCELLENT (<1% error)")
    elif metrics['energy_rel_error'] < 5.0:
        quality_score += 15
        logger.info("✓ Energy accuracy: GOOD (<5% error)")
    elif metrics['energy_rel_error'] < 10.0:
        quality_score += 10
        logger.info("⚠ Energy accuracy: ACCEPTABLE (<10% error)")
    else:
        logger.info("✗ Energy accuracy: POOR (>10% error)")

    # Force RMSE (30 points)
    max_score += 30
    if metrics['force_rmse'] < 0.1:
        quality_score += 30
        logger.info("✓ Force RMSE: EXCELLENT (<0.1 eV/Å)")
    elif metrics['force_rmse'] < 0.2:
        quality_score += 25
        logger.info("✓ Force RMSE: GOOD (<0.2 eV/Å)")
    elif metrics['force_rmse'] < 0.5:
        quality_score += 15
        logger.info("⚠ Force RMSE: ACCEPTABLE (<0.5 eV/Å)")
    else:
        logger.info("✗ Force RMSE: POOR (>0.5 eV/Å)")

    # Angular accuracy (30 points)
    max_score += 30
    if metrics['angular_error_mean'] < 5.0:
        quality_score += 30
        logger.info("✓ Angular accuracy: EXCELLENT (<5° mean error)")
    elif metrics['angular_error_mean'] < 10.0:
        quality_score += 25
        logger.info("✓ Angular accuracy: GOOD (<10° mean error)")
    elif metrics['angular_error_mean'] < 20.0:
        quality_score += 15
        logger.info("⚠ Angular accuracy: ACCEPTABLE (<20° mean error)")
    else:
        logger.info("✗ Angular accuracy: POOR (>20° mean error)")

    # Max force error (20 points)
    max_score += 20
    if metrics['force_max_error'] < 0.5:
        quality_score += 20
        logger.info("✓ Max force error: EXCELLENT (<0.5 eV/Å)")
    elif metrics['force_max_error'] < 1.0:
        quality_score += 15
        logger.info("✓ Max force error: GOOD (<1.0 eV/Å)")
    elif metrics['force_max_error'] < 2.0:
        quality_score += 10
        logger.info("⚠ Max force error: ACCEPTABLE (<2.0 eV/Å)")
    else:
        logger.info("✗ Max force error: POOR (>2.0 eV/Å)")

    overall_score = 100 * quality_score / max_score
    logger.info(f"\n{'=' * 80}")
    logger.info(f"OVERALL QUALITY SCORE: {quality_score}/{max_score} ({overall_score:.1f}%)")
    logger.info(f"{'=' * 80}\n")

    if overall_score >= 90:
        logger.info("✓ EXCELLENT: Student model performs very well on unseen molecule")
    elif overall_score >= 75:
        logger.info("✓ GOOD: Student model shows good generalization")
    elif overall_score >= 60:
        logger.info("⚠ ACCEPTABLE: Student model has moderate accuracy")
    else:
        logger.info("✗ POOR: Student model needs improvement")


def main():
    parser = argparse.ArgumentParser(
        description="Validate student model on test MolDiff molecule"
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=REPO_ROOT / 'checkpoints/best_model.pt',
        help='Path to student model checkpoint'
    )
    parser.add_argument(
        '--test-molecule',
        type=Path,
        default=REPO_ROOT / 'data/generative_test/moldiff/test_10mols_20251123_181225_SDF/0.sdf',
        help='Path to test molecule SDF file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference'
    )

    args = parser.parse_args()
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("Student Model Validation on Test Molecule")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test molecule: {args.test_molecule}")
    logger.info(f"Device: {args.device}")

    # Load test molecule with explicit hydrogens
    logger.info("\nLoading test molecule with explicit hydrogens...")
    atoms = read_structure_with_hydrogen_support(args.test_molecule)
    h_stats = check_hydrogen_content(atoms)
    logger.info(f"✓ Loaded {h_stats['n_atoms']} atoms ({h_stats['h_percentage']:.1f}% hydrogen)")

    # Load teacher model
    logger.info("\nInitializing teacher model (Orb-v2)...")
    teacher = OrbCalculator(device=args.device)
    logger.info("✓ Teacher model loaded")

    # Load student model
    logger.info("\nLoading student model...")
    student = load_student_model(args.checkpoint, args.device)
    logger.info("✓ Student model loaded")

    # Get teacher predictions
    logger.info("\nGetting teacher predictions...")
    atoms.calc = teacher
    teacher_energy = atoms.get_potential_energy()
    teacher_forces = atoms.get_forces()
    logger.info("✓ Teacher predictions computed")

    # Get student predictions
    logger.info("\nGetting student predictions...")
    student_energy, student_forces = predict_with_student(student, atoms, args.device)
    logger.info("✓ Student predictions computed")

    # Compute metrics
    logger.info("\nComputing comparison metrics...")
    metrics = compute_metrics(teacher_energy, teacher_forces, student_energy, student_forces)

    # Print results
    print_results(atoms, teacher_energy, teacher_forces, student_energy, student_forces, metrics, logger)

    return 0


if __name__ == '__main__':
    sys.exit(main())
