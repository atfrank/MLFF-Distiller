#!/usr/bin/env python3
"""
Simple Progressive Distillation Training for Compact Models

No batching complications - just train on one molecule at a time.
"""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json
import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField


def train_compact_model(
    teacher_ckpt: str,
    output_dir: Path,
    hidden_dim: int,
    num_interactions: int,
    num_rbf: int,
    epochs: int = 100,
    lr: float = 5e-4,
    n_samples: int = 1000
):
    """Train a compact model with simple synthetic data."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load teacher
    print(f"Loading teacher from {teacher_ckpt}")
    teacher_state = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    teacher_config = teacher_state.get('config', {
        'hidden_dim': 128, 'num_interactions': 3, 'num_rbf': 20,
        'cutoff': 5.0, 'max_z': 100
    })
    teacher = StudentForceField(**teacher_config).to(device)
    teacher.load_state_dict(teacher_state['model_state_dict'])
    teacher.eval()

    # Create student
    student_config = {
        'hidden_dim': hidden_dim,
        'num_interactions': num_interactions,
        'num_rbf': num_rbf,
        'cutoff': 5.0,
        'max_z': 100
    }
    student = StudentForceField(**student_config).to(device)

    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {n_params:,} params ({n_params*4/1024/1024:.2f} MB)")

    # Optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)

    # Training history
    history = {'energy_loss': [], 'force_loss': [], 'total_loss': []}
    best_loss = float('inf')

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        student.train()
        epoch_e_loss = 0.0
        epoch_f_loss = 0.0
        epoch_total = 0.0

        pbar = tqdm(range(n_samples), desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for _ in pbar:
            # Generate random molecule
            n_atoms = torch.randint(5, 20, (1,)).item()
            z = torch.randint(1, 10, (n_atoms,), device=device)
            pos = torch.randn(n_atoms, 3, device=device) * 2.5

            # Teacher forward (compute then detach)
            pos_t = pos.clone().requires_grad_(True)
            e_teacher = teacher(z, pos_t)
            f_teacher = -torch.autograd.grad(e_teacher, pos_t, create_graph=False)[0]
            e_teacher = e_teacher.detach()
            f_teacher = f_teacher.detach()

            #Student forward
            pos_s = pos.clone().requires_grad_(True)
            e_student = student(z, pos_s)
            f_student = -torch.autograd.grad(e_student, pos_s, create_graph=True)[0]

            # Loss
            e_loss = nn.functional.mse_loss(e_student, e_teacher)
            f_loss = nn.functional.mse_loss(f_student, f_teacher)
            total_loss = e_loss + 15.0 * f_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            # Track
            epoch_e_loss += e_loss.item()
            epoch_f_loss += f_loss.item()
            epoch_total += total_loss.item()

            pbar.set_postfix({'E': f"{e_loss.item():.4f}", 'F': f"{f_loss.item():.4f}"})

        # Epoch stats
        avg_e = epoch_e_loss / n_samples
        avg_f = epoch_f_loss / n_samples
        avg_total = epoch_total / n_samples

        history['energy_loss'].append(avg_e)
        history['force_loss'].append(avg_f)
        history['total_loss'].append(avg_total)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: E={avg_e:.4f}, F={avg_f:.4f}, Total={avg_total:.4f}")

        # Save best
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'config': student_config,
                'best_loss': best_loss
            }, output_dir / 'best_model.pt')

    # Save final
    torch.save({
        'epoch': epochs,
        'model_state_dict': student.state_dict(),
        'config': student_config,
        'history': history
    }, output_dir / 'final_model.pt')

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default='checkpoints/best_model.pt')
    parser.add_argument('--output-tiny', default='checkpoints/tiny')
    parser.add_argument('--output-ultra', default='checkpoints/ultra_tiny')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # Train Tiny
    print("="*70)
    print("TRAINING TINY MODEL")
    print("="*70)
    train_compact_model(
        teacher_ckpt=args.teacher,
        output_dir=Path(args.output_tiny),
        hidden_dim=64,
        num_interactions=2,
        num_rbf=12,
        epochs=args.epochs
    )

    # Train Ultra-tiny
    print("\n" + "="*70)
    print("TRAINING ULTRA-TINY MODEL")
    print("="*70)
    train_compact_model(
        teacher_ckpt=args.teacher,
        output_dir=Path(args.output_ultra),
        hidden_dim=32,
        num_interactions=2,
        num_rbf=10,
        epochs=args.epochs
    )

    print("\n" + "="*70)
    print("✅ ALL MODELS TRAINED!")
    print("="*70)
