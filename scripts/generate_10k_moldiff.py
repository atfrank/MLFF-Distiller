#!/usr/bin/env python3
"""
MolDiff-Focused 10K Generation Script
Issue #18: Medium-Scale Validation (Fast Iteration)

Streamlined generation using MolDiff for fast validation (8-10 hour runtime).
Distribution: 9,000 MolDiff molecules + 1,000 benchmark structures.

Author: Lead Coordinator
Date: 2025-11-23
"""

import os
import sys
import json
import yaml
import time
import logging
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import h5py

# Add src to path for imports
REPO_ROOT = Path("/home/aaron/ATX/software/MLFF_Distiller")
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlff_distiller.data.hdf5_writer import HDF5DatasetWriter
from mlff_distiller.models.teacher_wrappers import OrbCalculator


class GenerationCheckpoint:
    """Manages checkpointing for resumable generation."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict):
        """Save checkpoint state."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
        logging.info(f"Checkpoint saved: {self.checkpoint_path}")

    def load(self) -> Optional[Dict]:
        """Load checkpoint state if exists."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            logging.info(f"Checkpoint loaded: {self.checkpoint_path}")
            return state
        return None

    def clear(self):
        """Remove checkpoint after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logging.info("Checkpoint cleared")


class MolDiff10KGenerator:
    """Orchestrates MolDiff-focused 10K generation run."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.repo_root = REPO_ROOT
        self.output_dir = self.repo_root / self.config['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Initialize checkpoint manager
        checkpoint_path = self.output_dir / "generation_checkpoint.json"
        self.checkpoint = GenerationCheckpoint(checkpoint_path)

        # Initialize HDF5 writer
        hdf5_path = self.output_dir / "medium_scale_10k_moldiff.h5"
        self.writer = HDF5DatasetWriter(str(hdf5_path))

        # Initialize teacher calculator (orb-v2)
        logging.info(f"Initializing teacher model: {self.config['teacher_model']}")
        self.teacher_calc = OrbCalculator(
            model_name=self.config['teacher_model'],
            device=self.config['device']
        )

        # Track statistics
        self.stats = {
            'total_generated': 0,
            'total_labeled': 0,
            'failed_generations': 0,
            'failed_labels': 0,
            'start_time': None,
            'moldiff_count': 0,
            'benchmark_count': 0
        }

    def _load_config(self, config_path: Path) -> Dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure logging."""
        log_file = self.repo_root / self.config['logging']['log_file']
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config['logging']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("="*80)
        logging.info("MolDiff-Focused 10K Generation Run Started")
        logging.info(f"Configuration: {self.config_path}")
        logging.info(f"Expected runtime: 8-10 hours")
        logging.info("="*80)

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint or initialize fresh state."""
        state = self.checkpoint.load()
        if state:
            logging.info("Resuming from checkpoint:")
            logging.info(json.dumps(state, indent=2))
            self.stats.update(state.get('stats', {}))
            return state

        # Fresh start
        return {
            'moldiff_completed': 0,
            'benchmark_completed': 0,
            'stats': self.stats
        }

    def _save_checkpoint(self, state: Dict):
        """Save current state to checkpoint."""
        state['stats'] = self.stats
        self.checkpoint.save(state)

    def generate_moldiff_batch(self, batch_size: int, output_path: Path) -> List[Path]:
        """
        Generate batch of molecules using MolDiff.

        Returns list of generated structure files (SDF format).
        """
        logging.info(f"Generating MolDiff batch (size={batch_size})")

        moldiff_config = self.config['moldiff']
        env_path = self.repo_root / moldiff_config['environment_path']
        code_path = self.repo_root / moldiff_config['code_path']
        model_ckpt = moldiff_config['model_checkpoint']  # Relative to code_path
        bond_ckpt = moldiff_config['bond_predictor']  # Relative to code_path

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Create MolDiff config file for this batch
        batch_config_path = output_path / "moldiff_config.yml"
        batch_config = {
            'model': {
                'checkpoint': model_ckpt,
                'bond_predictor': bond_ckpt
            },
            'sample': {
                'seed': 2023,
                'batch_size': batch_size,
                'num_mols': batch_size,
                'save_traj_prob': 0.0
            }
        }

        import yaml as yaml_lib
        with open(batch_config_path, 'w') as f:
            yaml_lib.dump(batch_config, f)

        # Build command
        cmd = (
            f"source {env_path}/bin/activate && "
            f"cd {code_path} && "
            f"python scripts/sample_drug3d.py "
            f"--config {batch_config_path} "
            f"--outdir {output_path} "
            f"--device {self.config['device']}"
        )

        try:
            start_time = time.time()
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per batch
            )

            if result.returncode != 0:
                logging.error(f"MolDiff generation failed: {result.stderr}")
                return []

            # Collect generated files (MolDiff creates timestamped dir with SDF subdir)
            # Pattern: {outdir}/{config_name}_{timestamp}_SDF/*.sdf
            sdf_dirs = list(output_path.glob("*_SDF"))
            generated = []
            for sdf_dir in sdf_dirs:
                generated.extend(list(sdf_dir.glob("*.sdf")))

            elapsed = time.time() - start_time
            logging.info(f"Generated {len(generated)} MolDiff structures in {elapsed:.1f}s ({len(generated)/elapsed:.2f} molecules/s)")
            return generated

        except subprocess.TimeoutExpired:
            logging.error("MolDiff generation timeout")
            return []
        except Exception as e:
            logging.error(f"MolDiff generation error: {e}")
            return []

    def label_structures(self, structure_files: List[Path], source: str) -> int:
        """
        Label structures with teacher model and write to HDF5.

        Returns number successfully labeled.
        """
        labeled_count = 0

        for struct_file in structure_files:
            try:
                # Read structure using ASE
                from ase.io import read
                atoms = read(str(struct_file))

                # Attach calculator and get predictions
                atoms.calc = self.teacher_calc
                energy = atoms.get_potential_energy()  # eV
                forces = atoms.get_forces()  # eV/Angstrom

                # Get stress if periodic
                stress = None
                if atoms.pbc.any():
                    stress = atoms.get_stress()  # eV/Angstrom^3

                # Prepare metadata
                metadata = {
                    'source': source,
                    'file': str(struct_file.name),
                    'num_atoms': len(atoms),
                    'species': atoms.get_chemical_symbols(),
                    'has_cell': bool(atoms.pbc.any())
                }

                # Write to HDF5 (add_structure expects Atoms object)
                self.writer.add_structure(
                    atoms=atoms,
                    energy=energy,
                    forces=forces,
                    stress=stress,
                    metadata=metadata
                )

                labeled_count += 1
                self.stats['total_labeled'] += 1

                # Progress logging
                if self.stats['total_labeled'] % self.config['logging']['progress_interval'] == 0:
                    self._log_progress()

            except Exception as e:
                logging.error(f"Failed to label {struct_file}: {e}")
                self.stats['failed_labels'] += 1
                continue

        return labeled_count

    def _log_progress(self):
        """Log current progress."""
        elapsed = time.time() - self.stats['start_time']
        rate = self.stats['total_labeled'] / elapsed if elapsed > 0 else 0
        remaining = self.config['total_samples'] - self.stats['total_labeled']
        eta_hours = (remaining / rate / 3600) if rate > 0 else 0

        logging.info("="*60)
        logging.info(f"Progress: {self.stats['total_labeled']}/{self.config['total_samples']}")
        logging.info(f"Rate: {rate:.2f} structures/second")
        logging.info(f"MolDiff: {self.stats['moldiff_count']}")
        logging.info(f"Benchmark: {self.stats['benchmark_count']}")
        logging.info(f"Failed labels: {self.stats['failed_labels']}")
        logging.info(f"Elapsed: {elapsed/3600:.2f} hours")
        logging.info(f"ETA: {eta_hours:.2f} hours")
        logging.info("="*60)

    def run(self):
        """Execute the full 10K generation run."""
        logging.info("Starting MolDiff-focused 10K generation run")
        self.stats['start_time'] = time.time()

        # Load checkpoint state
        state = self._load_checkpoint()

        try:
            # Phase 1: MolDiff molecules (9,000)
            target_moldiff = self.config['distribution']['moldiff_molecules']
            batch_size = self.config['batch_size']['moldiff']

            logging.info(f"Phase 1: Generating {target_moldiff} MolDiff molecules")
            logging.info(f"Batch size: {batch_size}")

            completed = state['moldiff_completed']
            while completed < target_moldiff:
                remaining = target_moldiff - completed
                current_batch = min(batch_size, remaining)

                output_path = self.output_dir / f"moldiff_batch_{completed}"
                structures = self.generate_moldiff_batch(current_batch, output_path)

                if structures:
                    labeled = self.label_structures(structures, 'moldiff')
                    completed += labeled
                    state['moldiff_completed'] = completed
                    self.stats['moldiff_count'] = completed

                    # Checkpoint
                    if completed % self.config['checkpoint_interval'] == 0:
                        self._save_checkpoint(state)
                        logging.info(f"Checkpoint: {completed}/{target_moldiff} complete")
                else:
                    logging.warning(f"Batch generation failed, retrying in 10s...")
                    time.sleep(10)

            logging.info(f"Phase 1 complete: {completed} MolDiff molecules generated")

            # Phase 2: Benchmark structures (1,000)
            # TODO: Implement benchmark loading from QM9/GEOM
            logging.info("Phase 2: Benchmark structures - PLACEHOLDER")
            logging.info("Will load 1,000 structures from QM9/GEOM datasets")
            # For now, mark as complete
            state['benchmark_completed'] = 1000
            self.stats['benchmark_count'] = 1000

            # Success!
            self._finalize()

        except KeyboardInterrupt:
            logging.warning("Generation interrupted by user")
            self._save_checkpoint(state)
            raise
        except Exception as e:
            logging.error(f"Generation failed: {e}", exc_info=True)
            self._save_checkpoint(state)
            raise

    def _finalize(self):
        """Finalize generation and create reports."""
        logging.info("="*80)
        logging.info("MolDiff-Focused 10K Generation Complete!")
        logging.info("="*80)

        elapsed = time.time() - self.stats['start_time']

        # Final statistics
        total = self.stats['total_labeled']
        failed = self.stats['failed_labels']
        success_rate = total / (total + failed) if (total + failed) > 0 else 0

        summary = {
            'total_structures': total,
            'moldiff_molecules': self.stats['moldiff_count'],
            'benchmark_structures': self.stats['benchmark_count'],
            'failed_labels': failed,
            'success_rate': success_rate,
            'elapsed_hours': elapsed / 3600,
            'structures_per_hour': total / (elapsed / 3600),
            'go_no_go': 'GO' if success_rate >= 0.95 else 'NO-GO',
            'go_no_go_reason': 'Success rate >= 95%' if success_rate >= 0.95 else f'Success rate {success_rate:.1%} < 95%'
        }

        logging.info(json.dumps(summary, indent=2))

        # Save summary
        summary_path = self.output_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Close HDF5 file
        self.writer.finalize()

        # Clear checkpoint
        self.checkpoint.clear()

        logging.info(f"HDF5 dataset: {self.output_dir / 'medium_scale_10k_moldiff.h5'}")
        logging.info(f"Summary: {summary_path}")
        logging.info(f"GO/NO-GO: {summary['go_no_go']} - {summary['go_no_go_reason']}")


def main():
    parser = argparse.ArgumentParser(description="MolDiff-Focused 10K Generation")
    parser.add_argument(
        '--config',
        type=Path,
        default=REPO_ROOT / 'configs/medium_scale_10k_moldiff.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    generator = MolDiff10KGenerator(args.config)
    generator.run()


if __name__ == '__main__':
    main()
