#!/usr/bin/env python3
"""
Detailed Force Analysis and Visualization

Creates comprehensive plots analyzing force predictions:
- Per-atom force magnitude comparison
- Force component analysis (X, Y, Z)
- Angular error distribution
- Error vs force magnitude correlation
- Per-element force statistics
- Directional error analysis

Usage:
    python scripts/analyze_forces_detailed.py \
        --checkpoint checkpoints/best_model.pt \
        --test-molecule data/generative_test/moldiff/test_10mols_20251123_181225_SDF/0.sdf \
        --output-dir visualizations/force_analysis

Author: ML Distillation Project
Date: 2025-11-24
"""

import sys
import argparse
from pathlib import Path
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns

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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_with_student(model, atoms, device: str):
    """Get predictions from student model."""
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    positions.requires_grad_(True)

    batch = torch.zeros(len(atomic_numbers), dtype=torch.long, device=device)

    energy = model(atomic_numbers, positions, cell=None, pbc=None, batch=batch)

    forces = -torch.autograd.grad(
        energy.sum(),
        positions,
        create_graph=False,
        retain_graph=False
    )[0]

    return energy.item(), forces.cpu().numpy()


def plot_force_analysis(
    atoms,
    teacher_forces,
    student_forces,
    output_dir: Path,
    logger=None
):
    """
    Create comprehensive force analysis plots.

    Args:
        atoms: ASE Atoms object
        teacher_forces: Teacher force predictions (N, 3)
        student_forces: Student force predictions (N, 3)
        output_dir: Output directory for plots
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10

    # Compute metrics
    force_errors = student_forces - teacher_forces
    force_error_mags = np.linalg.norm(force_errors, axis=1)
    teacher_force_mags = np.linalg.norm(teacher_forces, axis=1)
    student_force_mags = np.linalg.norm(student_forces, axis=1)

    # Angular errors
    dot_products = np.sum(teacher_forces * student_forces, axis=1)
    valid_mask = (teacher_force_mags > 1e-6) & (student_force_mags > 1e-6)
    cos_sim = np.zeros_like(teacher_force_mags)
    cos_sim[valid_mask] = dot_products[valid_mask] / (teacher_force_mags[valid_mask] * student_force_mags[valid_mask])
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angular_errors = np.degrees(np.arccos(cos_sim))

    # Get element symbols
    symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    element_colors = {
        'H': '#FFFFFF',
        'C': '#909090',
        'N': '#3050F8',
        'O': '#FF0D0D',
        'S': '#FFFF30',
        'P': '#FF8000',
    }

    # Create comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Force Magnitude Comparison (Scatter)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(teacher_force_mags, student_force_mags, alpha=0.6, s=30, c='steelblue')
    max_force = max(teacher_force_mags.max(), student_force_mags.max())
    ax1.plot([0, max_force], [0, max_force], 'r--', linewidth=2, label='Perfect Agreement')
    ax1.set_xlabel('Teacher Force Magnitude (eV/Å)', fontweight='bold')
    ax1.set_ylabel('Student Force Magnitude (eV/Å)', fontweight='bold')
    ax1.set_title('Force Magnitude: Teacher vs Student', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add R² and RMSE
    # Compute R² manually
    ss_res = np.sum((teacher_force_mags - student_force_mags)**2)
    ss_tot = np.sum((teacher_force_mags - teacher_force_mags.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((teacher_force_mags - student_force_mags)**2))
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f} eV/Å',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Per-Atom Force Error
    ax2 = fig.add_subplot(gs[0, 1])
    atom_indices = np.arange(len(atoms))
    colors_by_element = [element_colors.get(s, '#CCCCCC') for s in symbols]
    bars = ax2.bar(atom_indices, force_error_mags, color=colors_by_element, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Atom Index', fontweight='bold')
    ax2.set_ylabel('Force Error Magnitude (eV/Å)', fontweight='bold')
    ax2.set_title('Per-Atom Force Error Magnitude', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add legend for elements
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=element_colors.get(elem, '#CCCCCC'),
                            edgecolor='black', label=elem)
                      for elem in unique_elements]
    ax2.legend(handles=legend_elements, loc='upper right', ncol=len(unique_elements))

    # 3. Angular Error Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(angular_errors[valid_mask], bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(angular_errors[valid_mask]), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(angular_errors[valid_mask]):.2f}°')
    ax3.axvline(np.median(angular_errors[valid_mask]), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(angular_errors[valid_mask]):.2f}°')
    ax3.set_xlabel('Angular Error (degrees)', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Force Direction Error Distribution', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Force Components Comparison (X, Y, Z)
    for idx, (component, label) in enumerate([(0, 'X'), (1, 'Y'), (2, 'Z')]):
        ax = fig.add_subplot(gs[1, idx])

        teacher_comp = teacher_forces[:, component]
        student_comp = student_forces[:, component]

        ax.scatter(teacher_comp, student_comp, alpha=0.6, s=30, c='steelblue')
        max_val = max(abs(teacher_comp).max(), abs(student_comp).max())
        ax.plot([-max_val, max_val], [-max_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel(f'Teacher F{label} (eV/Å)', fontweight='bold')
        ax.set_ylabel(f'Student F{label} (eV/Å)', fontweight='bold')
        ax.set_title(f'Force Component {label}', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add R²
        ss_res_comp = np.sum((teacher_comp - student_comp)**2)
        ss_tot_comp = np.sum((teacher_comp - teacher_comp.mean())**2)
        r2_comp = 1 - (ss_res_comp / ss_tot_comp) if ss_tot_comp > 0 else 0.0
        ax.text(0.05, 0.95, f'R² = {r2_comp:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 5. Error vs Force Magnitude
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(teacher_force_mags, force_error_mags, alpha=0.6, s=30, c='coral')
    ax5.set_xlabel('Teacher Force Magnitude (eV/Å)', fontweight='bold')
    ax5.set_ylabel('Force Error Magnitude (eV/Å)', fontweight='bold')
    ax5.set_title('Error vs Force Magnitude', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(teacher_force_mags, force_error_mags, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(teacher_force_mags.min(), teacher_force_mags.max(), 100)
    ax5.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
    ax5.legend()

    # 6. Per-Element Force Statistics
    ax6 = fig.add_subplot(gs[2, 1])

    element_data = {}
    for elem in unique_elements:
        mask = np.array([s == elem for s in symbols])
        element_data[elem] = {
            'teacher_mean': teacher_force_mags[mask].mean(),
            'student_mean': student_force_mags[mask].mean(),
            'error_mean': force_error_mags[mask].mean(),
            'count': mask.sum()
        }

    x_pos = np.arange(len(unique_elements))
    width = 0.35

    teacher_means = [element_data[e]['teacher_mean'] for e in unique_elements]
    student_means = [element_data[e]['student_mean'] for e in unique_elements]

    ax6.bar(x_pos - width/2, teacher_means, width, label='Teacher', color='steelblue', alpha=0.7, edgecolor='black')
    ax6.bar(x_pos + width/2, student_means, width, label='Student', color='coral', alpha=0.7, edgecolor='black')

    ax6.set_xlabel('Element', fontweight='bold')
    ax6.set_ylabel('Mean Force Magnitude (eV/Å)', fontweight='bold')
    ax6.set_title('Mean Force Magnitude by Element', fontweight='bold', fontsize=12)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(unique_elements)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Add counts on top
    for i, elem in enumerate(unique_elements):
        count = element_data[elem]['count']
        ax6.text(i, max(teacher_means[i], student_means[i]) + 0.1, f'n={count}',
                ha='center', va='bottom', fontsize=8)

    # 7. Per-Element Error Statistics
    ax7 = fig.add_subplot(gs[2, 2])

    error_means = [element_data[e]['error_mean'] for e in unique_elements]
    colors_elem = [element_colors.get(e, '#CCCCCC') for e in unique_elements]

    bars = ax7.bar(unique_elements, error_means, color=colors_elem, alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Element', fontweight='bold')
    ax7.set_ylabel('Mean Force Error (eV/Å)', fontweight='bold')
    ax7.set_title('Mean Force Error by Element', fontweight='bold', fontsize=12)
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Cosine Similarity Distribution
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.hist(cos_sim[valid_mask], bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    ax8.axvline(np.mean(cos_sim[valid_mask]), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(cos_sim[valid_mask]):.4f}')
    ax8.set_xlabel('Cosine Similarity', fontweight='bold')
    ax8.set_ylabel('Count', fontweight='bold')
    ax8.set_title('Force Direction Cosine Similarity', fontweight='bold', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Error Heatmap (Top 20 errors)
    ax9 = fig.add_subplot(gs[3, 1])

    top_error_indices = np.argsort(force_error_mags)[-20:][::-1]
    error_matrix = np.abs(force_errors[top_error_indices])

    im = ax9.imshow(error_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax9.set_xlabel('Atom Index (Top 20 Errors)', fontweight='bold')
    ax9.set_ylabel('Force Component', fontweight='bold')
    ax9.set_yticks([0, 1, 2])
    ax9.set_yticklabels(['X', 'Y', 'Z'])
    ax9.set_xticks(range(len(top_error_indices)))
    ax9.set_xticklabels([f'{i}\n{symbols[i]}' for i in top_error_indices], fontsize=8)
    ax9.set_title('Force Error Components (Top 20 Atoms)', fontweight='bold', fontsize=12)

    cbar = plt.colorbar(im, ax=ax9)
    cbar.set_label('|Error| (eV/Å)', fontweight='bold')

    # 10. Angular Error vs Force Magnitude
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.scatter(teacher_force_mags[valid_mask], angular_errors[valid_mask], alpha=0.6, s=30, c='purple')
    ax10.set_xlabel('Teacher Force Magnitude (eV/Å)', fontweight='bold')
    ax10.set_ylabel('Angular Error (degrees)', fontweight='bold')
    ax10.set_title('Angular Error vs Force Magnitude', fontweight='bold', fontsize=12)
    ax10.grid(True, alpha=0.3)

    # 11. Force Magnitude Error Distribution
    ax11 = fig.add_subplot(gs[4, 0])
    magnitude_errors = student_force_mags - teacher_force_mags
    ax11.hist(magnitude_errors, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax11.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax11.axvline(np.mean(magnitude_errors), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(magnitude_errors):.4f} eV/Å')
    ax11.set_xlabel('Force Magnitude Error (Student - Teacher) (eV/Å)', fontweight='bold')
    ax11.set_ylabel('Count', fontweight='bold')
    ax11.set_title('Force Magnitude Error Distribution', fontweight='bold', fontsize=12)
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # 12. Cumulative Error Distribution
    ax12 = fig.add_subplot(gs[4, 1])
    sorted_errors = np.sort(force_error_mags)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax12.plot(sorted_errors, cumulative, linewidth=2, color='darkblue')
    ax12.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50th percentile')
    ax12.axhline(90, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='90th percentile')
    ax12.axhline(95, color='darkred', linestyle='--', linewidth=1, alpha=0.5, label='95th percentile')
    ax12.set_xlabel('Force Error Magnitude (eV/Å)', fontweight='bold')
    ax12.set_ylabel('Cumulative Percentage (%)', fontweight='bold')
    ax12.set_title('Cumulative Error Distribution', fontweight='bold', fontsize=12)
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    # Add percentile values
    p50 = np.percentile(force_error_mags, 50)
    p90 = np.percentile(force_error_mags, 90)
    p95 = np.percentile(force_error_mags, 95)
    ax12.text(0.6, 0.3, f'50th: {p50:.4f} eV/Å\n90th: {p90:.4f} eV/Å\n95th: {p95:.4f} eV/Å',
             transform=ax12.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 13. Relative Error vs Teacher Force
    ax13 = fig.add_subplot(gs[4, 2])
    relative_errors = force_error_mags / (teacher_force_mags + 1e-8) * 100
    ax13.scatter(teacher_force_mags, relative_errors, alpha=0.6, s=30, c='teal')
    ax13.set_xlabel('Teacher Force Magnitude (eV/Å)', fontweight='bold')
    ax13.set_ylabel('Relative Error (%)', fontweight='bold')
    ax13.set_title('Relative Force Error', fontweight='bold', fontsize=12)
    ax13.grid(True, alpha=0.3)
    ax13.set_ylim([0, min(200, relative_errors.max())])

    # 14. Summary Statistics Table
    ax14 = fig.add_subplot(gs[5, :])
    ax14.axis('off')

    stats_data = [
        ['Metric', 'Teacher', 'Student', 'Error'],
        ['Mean Force Mag (eV/Å)', f'{teacher_force_mags.mean():.4f}', f'{student_force_mags.mean():.4f}', f'{force_error_mags.mean():.4f}'],
        ['Max Force Mag (eV/Å)', f'{teacher_force_mags.max():.4f}', f'{student_force_mags.max():.4f}', f'{force_error_mags.max():.4f}'],
        ['Min Force Mag (eV/Å)', f'{teacher_force_mags.min():.4f}', f'{student_force_mags.min():.4f}', f'{force_error_mags.min():.4f}'],
        ['Std Force Mag (eV/Å)', f'{teacher_force_mags.std():.4f}', f'{student_force_mags.std():.4f}', f'{force_error_mags.std():.4f}'],
        ['Force RMSE (eV/Å)', '-', '-', f'{np.sqrt(np.mean(force_errors**2)):.4f}'],
        ['Force MAE (eV/Å)', '-', '-', f'{np.mean(np.abs(force_errors)):.4f}'],
        ['Mean Angular Error (°)', '-', '-', f'{np.mean(angular_errors[valid_mask]):.2f}'],
        ['Median Angular Error (°)', '-', '-', f'{np.median(angular_errors[valid_mask]):.2f}'],
        ['Mean Cosine Sim', '-', '-', f'{np.mean(cos_sim[valid_mask]):.4f}'],
    ]

    table = ax14.table(cellText=stats_data, cellLoc='center', loc='center',
                      bbox=[0.1, 0.0, 0.8, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax14.set_title('Summary Statistics', fontweight='bold', fontsize=14, pad=20)

    # Save figure
    output_file = output_dir / 'force_analysis_comprehensive.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Saved comprehensive force analysis to {output_file}")

    # Create individual focused plots
    create_focused_plots(atoms, teacher_forces, student_forces, output_dir, logger)


def create_focused_plots(atoms, teacher_forces, student_forces, output_dir, logger):
    """Create individual focused plots for detailed analysis."""

    symbols = atoms.get_chemical_symbols()
    force_errors = student_forces - teacher_forces
    force_error_mags = np.linalg.norm(force_errors, axis=1)
    teacher_force_mags = np.linalg.norm(teacher_forces, axis=1)
    student_force_mags = np.linalg.norm(student_forces, axis=1)

    # 1. High-resolution force comparison
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.scatter(teacher_force_mags, student_force_mags, alpha=0.7, s=50, c=force_error_mags,
              cmap='YlOrRd', edgecolors='black', linewidth=0.5)
    max_force = max(teacher_force_mags.max(), student_force_mags.max())
    ax.plot([0, max_force], [0, max_force], 'k--', linewidth=2, label='Perfect Agreement')
    ax.set_xlabel('Teacher Force Magnitude (eV/Å)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Student Force Magnitude (eV/Å)', fontweight='bold', fontsize=14)
    ax.set_title('Force Magnitude Comparison\n(Color = Error Magnitude)', fontweight='bold', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Error Magnitude (eV/Å)', fontweight='bold', fontsize=12)

    plt.savefig(output_dir / 'force_comparison_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Per-atom detailed breakdown
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), dpi=150)

    atom_indices = np.arange(len(atoms))

    # Force magnitudes
    ax1.plot(atom_indices, teacher_force_mags, 'o-', label='Teacher', linewidth=2, markersize=6, color='steelblue')
    ax1.plot(atom_indices, student_force_mags, 's-', label='Student', linewidth=2, markersize=6, color='coral')
    ax1.set_ylabel('Force Magnitude (eV/Å)', fontweight='bold', fontsize=12)
    ax1.set_title('Per-Atom Force Magnitude Comparison', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add element labels
    for i, sym in enumerate(symbols):
        if i % 3 == 0:  # Label every 3rd atom to avoid crowding
            ax1.text(i, max(teacher_force_mags[i], student_force_mags[i]) + 0.2, sym,
                    ha='center', va='bottom', fontsize=7, alpha=0.7)

    # Force errors
    ax2.bar(atom_indices, force_error_mags, color='crimson', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Force Error Magnitude (eV/Å)', fontweight='bold', fontsize=12)
    ax2.set_title('Per-Atom Force Error', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Relative errors
    relative_errors = force_error_mags / (teacher_force_mags + 1e-8) * 100
    ax3.bar(atom_indices, relative_errors, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Atom Index', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Relative Error (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Per-Atom Relative Force Error', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, min(100, relative_errors.max())])

    plt.tight_layout()
    plt.savefig(output_dir / 'per_atom_detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info("Saved focused analysis plots")


def main():
    parser = argparse.ArgumentParser(
        description="Detailed force analysis and visualization"
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
        '--output-dir',
        type=Path,
        default=REPO_ROOT / 'visualizations/force_analysis',
        help='Output directory for plots'
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
    logger.info("Detailed Force Analysis")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test molecule: {args.test_molecule}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    # Load test molecule
    logger.info("\nLoading test molecule...")
    atoms = read_structure_with_hydrogen_support(args.test_molecule)
    h_stats = check_hydrogen_content(atoms)
    logger.info(f"✓ Loaded {h_stats['n_atoms']} atoms ({h_stats['h_percentage']:.1f}% hydrogen)")

    # Load models
    logger.info("\nInitializing teacher model (Orb-v2)...")
    teacher = OrbCalculator(device=args.device)
    logger.info("✓ Teacher model loaded")

    logger.info("\nLoading student model...")
    student = load_student_model(args.checkpoint, args.device)
    logger.info("✓ Student model loaded")

    # Get predictions
    logger.info("\nGetting predictions...")
    atoms.calc = teacher
    teacher_energy = atoms.get_potential_energy()
    teacher_forces = atoms.get_forces()

    student_energy, student_forces = predict_with_student(student, atoms, args.device)
    logger.info("✓ Predictions computed")

    # Create plots
    logger.info("\nCreating comprehensive force analysis plots...")
    plot_force_analysis(atoms, teacher_forces, student_forces, args.output_dir, logger)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nPlots saved to: {args.output_dir}")
    logger.info("  - force_analysis_comprehensive.png (14-panel overview)")
    logger.info("  - force_comparison_detailed.png (high-res comparison)")
    logger.info("  - per_atom_detailed_analysis.png (per-atom breakdown)")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
