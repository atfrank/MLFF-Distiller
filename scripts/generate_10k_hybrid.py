#!/usr/bin/env python3
"""
Hybrid 10K Generation Script with RNA Integration
Issue #18: Medium-Scale Validation with RNA Biomolecules

Distribution: 8,000 MolDiff + 1,000 RNA + 1,000 benchmark structures.

Author: Lead Coordinator
Date: 2025-11-23
"""

import os
import sys
import json
import yaml
import time
import gzip
import random
import logging
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import h5py
from ase.io import read

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


class Hybrid10KGenerator:
    """Orchestrates hybrid 10K generation with MolDiff + RNA."""

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
        hdf5_path = self.output_dir / f"{self.config['project_name']}.h5"
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
            'rna_count': 0,
            'benchmark_count': 0
        }

        # Prepare RNA file list
        self.rna_files = self._prepare_rna_file_list()

    def _load_config(self, config_path: Path) -> Dict:
        """Load YAML configuration."""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure logging."""
        log_file = self.repo_root / self.config['logging']['log_file']
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _prepare_rna_file_list(self) -> List[Path]:
        """Prepare list of RNA PDB files to sample from."""
        rna_config = self.config['rna']
        rna_source = Path(rna_config['source_path'])

        logging.info(f"Scanning RNA structures from: {rna_source}")

        # Get all RNA systems (directories)
        rna_systems = [d for d in rna_source.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logging.info(f"Found {len(rna_systems)} RNA systems")

        # Collect all PDB files from all systems
        all_pdb_files = []
        for system_dir in rna_systems:
            coord_dir = system_dir / "coordinates"
            if coord_dir.exists():
                pdb_files = list(coord_dir.glob("*.pdb.gz"))
                all_pdb_files.extend(pdb_files)

        logging.info(f"Found {len(all_pdb_files)} total RNA structures")

        # Sample desired number uniformly
        num_to_sample = rna_config['num_structures']
        if len(all_pdb_files) > num_to_sample:
            sampled_files = random.sample(all_pdb_files, num_to_sample)
        else:
            sampled_files = all_pdb_files

        logging.info(f"Sampled {len(sampled_files)} RNA structures for generation")
        return sampled_files

    def generate_moldiff_batch(self, batch_size: int, batch_id: int) -> List[Path]:
        """Generate a batch of molecules using MolDiff."""
        moldiff_config = self.config['moldiff']
        moldiff_dir = self.repo_root / moldiff_config['code_path']
        env_path = self.repo_root / moldiff_config['environment_path']

        # Create output directory for this batch
        batch_output = self.output_dir / f"moldiff_batch_{batch_id}"
        batch_output.mkdir(exist_ok=True)

        # Create temporary config for this batch
        batch_config = {
            'model': {
                'checkpoint': moldiff_config['model_checkpoint'],
                'bond_predictor': moldiff_config['bond_predictor']
            },
            'sample': {
                'seed': 2023 + batch_id,
                'batch_size': batch_size,
                'num_mols': batch_size,
                'save_traj_prob': 0.0
            }
        }

        config_path = batch_output / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(batch_config, f)

        # Run MolDiff generation
        cmd = (
            f"bash -c 'source {env_path}/bin/activate && "
            f"cd {moldiff_dir} && "
            f"python scripts/sample_drug3d.py "
            f"--outdir {batch_output} "
            f"--config {config_path} "
            f"--device {self.config['device']}'"
        )

        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        duration = time.time() - start_time

        if result.returncode != 0:
            logging.error(f"MolDiff generation failed: {result.stderr}")
            return []

        # Find generated SDF files
        sdf_dirs = list(batch_output.glob("*_SDF"))
        if not sdf_dirs:
            logging.error(f"No SDF output found in {batch_output}")
            return []

        sdf_files = list(sdf_dirs[0].glob("*.sdf"))
        logging.info(f"Generated {len(sdf_files)} MolDiff structures in {duration:.1f}s "
                    f"({len(sdf_files)/duration:.2f} molecules/s)")

        return sdf_files

    def load_rna_batch(self, batch_size: int, start_idx: int) -> List[Path]:
        """Load a batch of RNA structures (already exist, just return paths)."""
        end_idx = min(start_idx + batch_size, len(self.rna_files))
        batch_files = self.rna_files[start_idx:end_idx]
        logging.info(f"RNA batch: returning {len(batch_files)} structures")
        return batch_files

    def label_with_teacher(self, structure_files: List[Path], source_type: str) -> int:
        """
        Label structures with orb-v2 teacher and write to HDF5.

        Args:
            structure_files: List of structure file paths
            source_type: 'moldiff', 'rna', or 'benchmark'

        Returns:
            Number of successfully labeled structures
        """
        success_count = 0

        for file_path in structure_files:
            try:
                # Load structure
                if file_path.suffix == '.gz':
                    # RNA PDB files are gzipped
                    with gzip.open(file_path, 'rt') as f:
                        atoms = read(f, format='pdb')
                else:
                    atoms = read(str(file_path))

                # Label with teacher
                atoms.calc = self.teacher_calc
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                # Validate (no NaN/Inf)
                if np.isnan(energy) or np.isinf(energy):
                    logging.warning(f"Invalid energy for {file_path.name}, skipping")
                    self.stats['failed_labels'] += 1
                    continue

                if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
                    logging.warning(f"Invalid forces for {file_path.name}, skipping")
                    self.stats['failed_labels'] += 1
                    continue

                # Write to HDF5
                self.writer.add_structure(
                    atoms=atoms,
                    energy=energy,
                    forces=forces,
                    metadata={
                        'source': source_type,
                        'source_file': str(file_path.name),
                        'num_atoms': len(atoms)
                    }
                )

                success_count += 1
                self.stats['total_labeled'] += 1

            except Exception as e:
                logging.error(f"Failed to label {file_path.name}: {e}")
                self.stats['failed_labels'] += 1

        return success_count

    def run_generation(self):
        """Main generation loop."""
        logging.info("="*60)
        logging.info(f"Starting Hybrid 10K Generation - {self.config['project_name']}")
        logging.info("="*60)
        logging.info(f"Distribution:")
        logging.info(f"  MolDiff:   {self.config['distribution']['moldiff_molecules']}")
        logging.info(f"  RNA:       {self.config['distribution']['rna_structures']}")
        logging.info(f"  Benchmark: {self.config['distribution']['benchmark_traditional']}")
        logging.info("="*60)

        self.stats['start_time'] = time.time()

        # Get targets
        target_moldiff = self.config['distribution']['moldiff_molecules']
        target_rna = self.config['distribution']['rna_structures']
        batch_size_moldiff = self.config['batch_size']['moldiff']
        batch_size_rna = self.config['batch_size']['rna']

        # Generate MolDiff structures
        logging.info("\n" + "="*60)
        logging.info("PHASE 1: MolDiff Generation")
        logging.info("="*60)

        batch_id = 0
        while self.stats['moldiff_count'] < target_moldiff:
            remaining = target_moldiff - self.stats['moldiff_count']
            current_batch_size = min(batch_size_moldiff, remaining)

            logging.info(f"Generating MolDiff batch (size={current_batch_size})")
            sdf_files = self.generate_moldiff_batch(current_batch_size, batch_id)

            if sdf_files:
                labeled = self.label_with_teacher(sdf_files, 'moldiff')
                self.stats['moldiff_count'] += labeled
                self._log_progress()

            batch_id += 1

        # Load and label RNA structures
        logging.info("\n" + "="*60)
        logging.info("PHASE 2: RNA Structure Loading")
        logging.info("="*60)

        rna_idx = 0
        while self.stats['rna_count'] < target_rna:
            remaining = target_rna - self.stats['rna_count']
            current_batch_size = min(batch_size_rna, remaining)

            logging.info(f"Loading RNA batch (size={current_batch_size})")
            rna_files = self.load_rna_batch(current_batch_size, rna_idx)

            if not rna_files:
                logging.warning(f"No more RNA files available (loaded {self.stats['rna_count']}/{target_rna})")
                break

            labeled = self.label_with_teacher(rna_files, 'rna')
            self.stats['rna_count'] += labeled
            self._log_progress()

            rna_idx += current_batch_size

        # Finalize
        self.writer.finalize()
        self._log_final_summary()

    def _log_progress(self):
        """Log current progress."""
        total_target = self.config['total_samples']
        total_current = self.stats['moldiff_count'] + self.stats['rna_count'] + self.stats['benchmark_count']

        elapsed = time.time() - self.stats['start_time']
        rate = total_current / elapsed if elapsed > 0 else 0
        remaining = total_target - total_current
        eta = remaining / rate if rate > 0 else 0

        logging.info("="*60)
        logging.info(f"Progress: {total_current}/{total_target}")
        logging.info(f"Rate: {rate:.2f} structures/second")
        logging.info(f"MolDiff: {self.stats['moldiff_count']}")
        logging.info(f"RNA: {self.stats['rna_count']}")
        logging.info(f"Benchmark: {self.stats['benchmark_count']}")
        logging.info(f"Failed labels: {self.stats['failed_labels']}")
        logging.info(f"Elapsed: {elapsed/3600:.2f} hours")
        logging.info(f"ETA: {eta/3600:.2f} hours")
        logging.info("="*60)

    def _log_final_summary(self):
        """Log final generation summary."""
        elapsed = time.time() - self.stats['start_time']
        total = self.stats['moldiff_count'] + self.stats['rna_count'] + self.stats['benchmark_count']

        logging.info("\n" + "="*60)
        logging.info("GENERATION COMPLETE!")
        logging.info("="*60)
        logging.info(f"Total structures: {total}")
        logging.info(f"  MolDiff:   {self.stats['moldiff_count']}")
        logging.info(f"  RNA:       {self.stats['rna_count']}")
        logging.info(f"  Benchmark: {self.stats['benchmark_count']}")
        logging.info(f"Failed labels: {self.stats['failed_labels']}")
        logging.info(f"Success rate: {100*total/(total+self.stats['failed_labels']):.1f}%")
        logging.info(f"Total time: {elapsed/3600:.2f} hours")
        logging.info(f"Output: {self.output_dir}")
        logging.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hybrid 10K Generation with RNA")
    parser.add_argument(
        '--config',
        type=Path,
        default=REPO_ROOT / "configs/medium_scale_10k_hybrid.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    generator = Hybrid10KGenerator(args.config)
    generator.run_generation()


if __name__ == '__main__':
    main()
