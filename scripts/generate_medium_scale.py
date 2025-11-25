#!/usr/bin/env python3
"""
Medium-Scale 10K Generation Orchestrator
Issue #18: Medium-Scale Validation Run

Orchestrates generation of 10,000 structures across:
- MatterGen (5K crystals)
- MolDiff (4K molecules)
- Benchmark (1K traditional)

Includes checkpointing, progress monitoring, and teacher labeling.
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
from mlff_distiller.models.teacher_wrappers import TeacherWrapper


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


class MediumScaleGenerator:
    """Orchestrates medium-scale 10K generation run."""

    def __init__(self, config_path: Path):
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
        hdf5_path = self.output_dir / "medium_scale_10k.h5"
        self.writer = HDF5DatasetWriter(str(hdf5_path))

        # Initialize teacher wrapper
        logging.info(f"Initializing teacher model: {self.config['teacher_model']}")
        self.teacher = TeacherWrapper(
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
            'mattergen_count': 0,
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
        logging.info("Medium-Scale 10K Generation Run Started")
        logging.info(f"Configuration: {config_path}")
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
            'mattergen_completed': 0,
            'moldiff_completed': 0,
            'benchmark_completed': 0,
            'stats': self.stats
        }

    def _save_checkpoint(self, state: Dict):
        """Save current state to checkpoint."""
        state['stats'] = self.stats
        self.checkpoint.save(state)

    def generate_mattergen_batch(self, batch_size: int, output_path: Path) -> List[Path]:
        """
        Generate batch of crystals using MatterGen.

        Returns list of generated structure files.
        """
        logging.info(f"Generating MatterGen batch (size={batch_size})")

        mattergen_config = self.config['mattergen']
        env_path = self.repo_root / mattergen_config['environment_path']
        code_path = self.repo_root / mattergen_config['code_path']

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = (
            f"source {env_path}/bin/activate && "
            f"cd {code_path} && "
            f"python -c \""
            f"from mattergen.generation import generate_crystals; "
            f"generate_crystals("
            f"  output_dir='{output_path}', "
            f"  num_samples={batch_size}, "
            f"  pretrained_name='{mattergen_config['pretrained_name']}', "
            f"  device='{self.config['device']}'"
            f")\""
        )

        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per batch
            )

            if result.returncode != 0:
                logging.error(f"MatterGen generation failed: {result.stderr}")
                return []

            # Collect generated files
            generated = list(output_path.glob("*.cif"))
            logging.info(f"Generated {len(generated)} MatterGen structures")
            return generated

        except subprocess.TimeoutExpired:
            logging.error("MatterGen generation timeout")
            return []
        except Exception as e:
            logging.error(f"MatterGen generation error: {e}")
            return []

    def generate_moldiff_batch(self, batch_size: int, output_path: Path) -> List[Path]:
        """
        Generate batch of molecules using MolDiff.

        Returns list of generated structure files.
        """
        logging.info(f"Generating MolDiff batch (size={batch_size})")

        moldiff_config = self.config['moldiff']
        env_path = self.repo_root / moldiff_config['environment_path']
        code_path = self.repo_root / moldiff_config['code_path']

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = (
            f"source {env_path}/bin/activate && "
            f"cd {code_path} && "
            f"python scripts/sample_drug3d.py "
            f"--outdir {output_path} "
            f"--ckpt {moldiff_config['model_checkpoint']} "
            f"--bond_ckpt {moldiff_config['bond_predictor']} "
            f"--num_samples {batch_size} "
            f"--device {self.config['device']}"
        )

        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per batch
            )

            if result.returncode != 0:
                logging.error(f"MolDiff generation failed: {result.stderr}")
                return []

            # Collect generated files (MolDiff outputs .sdf or .xyz)
            generated = list(output_path.glob("*.sdf")) + list(output_path.glob("*.xyz"))
            logging.info(f"Generated {len(generated)} MolDiff structures")
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
                # Read structure (simplified - actual implementation needs ASE/pymatgen)
                # This is a placeholder - real implementation would use proper parsers
                from ase.io import read
                atoms = read(str(struct_file))

                # Get teacher predictions
                result = self.teacher.predict(
                    positions=atoms.get_positions(),
                    species=atoms.get_atomic_numbers(),
                    cell=atoms.get_cell() if atoms.pbc.any() else None
                )

                # Write to HDF5
                metadata = {
                    'source': source,
                    'file': str(struct_file.name),
                    'num_atoms': len(atoms),
                    'species': atoms.get_chemical_symbols(),
                    'has_cell': atoms.pbc.any()
                }

                self.writer.write_sample(
                    positions=atoms.get_positions(),
                    species=atoms.get_atomic_numbers(),
                    energy=result['energy'],
                    forces=result['forces'],
                    stress=result.get('stress'),
                    cell=atoms.get_cell() if atoms.pbc.any() else None,
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

        logging.info("="*60)
        logging.info(f"Progress: {self.stats['total_labeled']}/{self.config['total_samples']}")
        logging.info(f"Rate: {rate:.2f} structures/second")
        logging.info(f"MatterGen: {self.stats['mattergen_count']}")
        logging.info(f"MolDiff: {self.stats['moldiff_count']}")
        logging.info(f"Benchmark: {self.stats['benchmark_count']}")
        logging.info(f"Failed labels: {self.stats['failed_labels']}")
        logging.info(f"Elapsed: {elapsed/3600:.2f} hours")
        logging.info("="*60)

    def run(self):
        """Execute the full 10K generation run."""
        logging.info("Starting 10K generation run")
        self.stats['start_time'] = time.time()

        # Load checkpoint state
        state = self._load_checkpoint()

        try:
            # Phase 1: MatterGen crystals
            target_mattergen = self.config['distribution']['mattergen_crystals']
            batch_size = self.config['batch_size']['mattergen']

            logging.info(f"Phase 1: Generating {target_mattergen} MatterGen crystals")

            completed = state['mattergen_completed']
            while completed < target_mattergen:
                remaining = target_mattergen - completed
                current_batch = min(batch_size, remaining)

                output_path = self.output_dir / f"mattergen_batch_{completed}"
                structures = self.generate_mattergen_batch(current_batch, output_path)

                if structures:
                    labeled = self.label_structures(structures, 'mattergen')
                    completed += labeled
                    state['mattergen_completed'] = completed
                    self.stats['mattergen_count'] = completed

                    # Checkpoint
                    if completed % self.config['checkpoint_interval'] == 0:
                        self._save_checkpoint(state)
                else:
                    logging.warning(f"Batch generation failed, retrying...")
                    time.sleep(10)

            # Phase 2: MolDiff molecules
            target_moldiff = self.config['distribution']['moldiff_molecules']
            batch_size = self.config['batch_size']['moldiff']

            logging.info(f"Phase 2: Generating {target_moldiff} MolDiff molecules")

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
                else:
                    logging.warning(f"Batch generation failed, retrying...")
                    time.sleep(10)

            # Phase 3: Benchmark structures (simplified - just log for now)
            logging.info("Phase 3: Benchmark structures - PLACEHOLDER")
            # TODO: Implement benchmark structure loading

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
        logging.info("10K Generation Complete!")
        logging.info("="*80)

        elapsed = time.time() - self.stats['start_time']

        # Final statistics
        summary = {
            'total_structures': self.stats['total_labeled'],
            'mattergen_crystals': self.stats['mattergen_count'],
            'moldiff_molecules': self.stats['moldiff_count'],
            'benchmark_structures': self.stats['benchmark_count'],
            'failed_labels': self.stats['failed_labels'],
            'success_rate': self.stats['total_labeled'] / (self.stats['total_labeled'] + self.stats['failed_labels']),
            'elapsed_hours': elapsed / 3600,
            'structures_per_hour': self.stats['total_labeled'] / (elapsed / 3600)
        }

        logging.info(json.dumps(summary, indent=2))

        # Save summary
        summary_path = self.output_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Close HDF5 file
        self.writer.close()

        # Clear checkpoint
        self.checkpoint.clear()

        logging.info(f"HDF5 dataset: {self.output_dir / 'medium_scale_10k.h5'}")
        logging.info(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="10K Medium-Scale Generation")
    parser.add_argument(
        '--config',
        type=Path,
        default=REPO_ROOT / 'configs/medium_scale_10k.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    generator = MediumScaleGenerator(args.config)
    generator.run()


if __name__ == '__main__':
    main()
