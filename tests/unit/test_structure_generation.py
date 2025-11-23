"""
Unit tests for molecular structure generation.

Tests:
- MoleculeGenerator functionality
- CrystalGenerator functionality
- ClusterSurfaceGenerator functionality
- StructureGenerator integration

Author: Data Pipeline Engineer
Date: 2025-11-23
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mlff_distiller.data.sampling import SamplingConfig, SystemType
from mlff_distiller.data.structure_generation import (
    ClusterSurfaceGenerator,
    CrystalGenerator,
    MoleculeGenerator,
    StructureGenerator,
)


class TestMoleculeGenerator:
    """Test MoleculeGenerator class."""

    def test_initialization(self):
        """Test molecule generator initialization."""
        gen = MoleculeGenerator(seed=42)

        assert gen.rng is not None
        assert len(gen.element_set) > 0
        assert len(gen.ase_templates) > 0

    def test_generate_random_molecule(self):
        """Test random molecule generation."""
        gen = MoleculeGenerator(seed=42)
        mol = gen.generate_random_molecule(target_size=20)

        assert mol is not None
        assert len(mol) == 20
        assert not mol.pbc.any()  # Should be non-periodic

    def test_generate_from_template(self):
        """Test template-based generation."""
        gen = MoleculeGenerator(seed=42)
        mol = gen.generate_from_template(target_size=30)

        # May be None if generation fails, but usually succeeds
        if mol is not None:
            assert isinstance(mol, Atoms)
            assert not mol.pbc.any()

    def test_generate(self):
        """Test high-level generate method."""
        gen = MoleculeGenerator(seed=42)

        # Generate molecules of various sizes
        for size in [10, 20, 50, 100]:
            mol = gen.generate(target_size=size)

            assert isinstance(mol, Atoms)
            assert len(mol) >= 5  # At least some atoms
            assert len(mol) <= size + 10  # Within reasonable range
            assert not mol.pbc.any()

    def test_validate_molecule(self):
        """Test molecule validation."""
        gen = MoleculeGenerator(seed=42)

        # Valid molecule
        valid_mol = Atoms(
            "H2O",
            positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]],
            pbc=False,
        )
        assert gen._validate_molecule(valid_mol)

        # Invalid: atoms too close
        invalid_mol = Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]],
            pbc=False,
        )
        assert not gen._validate_molecule(invalid_mol)

        # Invalid: atoms too far apart
        far_mol = Atoms(
            "H2O",
            positions=[[0, 0, 0], [25, 0, 0], [0, 25, 0]],
            pbc=False,
        )
        assert not gen._validate_molecule(far_mol)

    def test_element_diversity(self):
        """Test that generated molecules use diverse elements."""
        gen = MoleculeGenerator(seed=42, element_set={"H", "C", "N", "O"})

        elements_seen = set()
        for _ in range(20):
            mol = gen.generate(target_size=30)
            elements_seen.update(mol.get_chemical_symbols())

        # Should see multiple elements
        assert len(elements_seen) >= 3


class TestCrystalGenerator:
    """Test CrystalGenerator class."""

    def test_initialization(self):
        """Test crystal generator initialization."""
        gen = CrystalGenerator(seed=42)

        assert gen.rng is not None
        assert len(gen.element_set) > 0
        assert len(gen.prototypes) > 0

    def test_generate_from_prototype(self):
        """Test prototype-based crystal generation."""
        gen = CrystalGenerator(seed=42)

        for prototype in ["fcc", "bcc", "diamond"]:
            crystal = gen.generate_from_prototype(target_size=100, prototype=prototype)

            if crystal is not None:
                assert isinstance(crystal, Atoms)
                assert crystal.pbc.all()  # Should be periodic
                assert len(crystal) > 0

    def test_generate_random_crystal(self):
        """Test random crystal generation."""
        gen = CrystalGenerator(seed=42)
        crystal = gen.generate_random_crystal(target_size=100)

        assert crystal is not None
        assert len(crystal) == 100
        assert crystal.pbc.all()  # Should be periodic
        assert crystal.cell is not None

    def test_generate(self):
        """Test high-level generate method."""
        gen = CrystalGenerator(seed=42)

        # Generate crystals of various sizes
        for size in [50, 100, 200, 500]:
            crystal = gen.generate(target_size=size)

            assert isinstance(crystal, Atoms)
            assert len(crystal) >= 10
            assert len(crystal) <= size + 50  # Within reasonable range
            assert crystal.pbc.all()

    def test_validate_crystal(self):
        """Test crystal validation."""
        gen = CrystalGenerator(seed=42)

        # Valid crystal
        from ase.build import bulk

        valid_crystal = bulk("Si", "diamond", a=5.43)
        assert gen._validate_crystal(valid_crystal)

        # Invalid: atoms too close
        invalid_crystal = valid_crystal.copy()
        invalid_crystal.positions[0] = invalid_crystal.positions[1]
        assert not gen._validate_crystal(invalid_crystal)


class TestClusterSurfaceGenerator:
    """Test ClusterSurfaceGenerator class."""

    def test_initialization(self):
        """Test cluster/surface generator initialization."""
        gen = ClusterSurfaceGenerator(seed=42)

        assert gen.rng is not None
        assert len(gen.element_set) > 0

    def test_generate_cluster(self):
        """Test cluster generation."""
        gen = ClusterSurfaceGenerator(seed=42)

        for size in [20, 50, 100]:
            cluster = gen.generate_cluster(target_size=size)

            assert cluster is not None
            assert len(cluster) == size
            assert not cluster.pbc.any()  # Non-periodic

    def test_generate_surface(self):
        """Test surface slab generation."""
        gen = ClusterSurfaceGenerator(seed=42)

        for size in [50, 100, 200]:
            surface = gen.generate_surface(target_size=size)

            assert surface is not None
            assert len(surface) >= 10
            assert surface.pbc.any()  # At least partially periodic

    def test_generate_both_types(self):
        """Test generating both clusters and surfaces."""
        gen = ClusterSurfaceGenerator(seed=42)

        cluster = gen.generate(SystemType.CLUSTER, target_size=50)
        surface = gen.generate(SystemType.SURFACE, target_size=50)

        assert isinstance(cluster, Atoms)
        assert isinstance(surface, Atoms)
        assert not cluster.pbc.any()
        assert surface.pbc.any()

    def test_relax_overlaps(self):
        """Test overlap relaxation."""
        gen = ClusterSurfaceGenerator(seed=42)

        # Create atoms with overlaps
        positions = np.array([
            [0, 0, 0],
            [0.1, 0, 0],  # Too close
            [5, 0, 0],
        ])
        atoms = Atoms("HHH", positions=positions, pbc=False)

        relaxed = gen._relax_overlaps(atoms)

        # Check distances increased
        pos = relaxed.get_positions()
        dist = np.linalg.norm(pos[1] - pos[0])
        assert dist > 1.0  # Should be separated


class TestStructureGenerator:
    """Test StructureGenerator integration."""

    def test_initialization(self):
        """Test structure generator initialization."""
        config = SamplingConfig(total_samples=100, seed=42)
        gen = StructureGenerator(config)

        assert gen.config == config
        assert gen.molecule_gen is not None
        assert gen.crystal_gen is not None
        assert gen.cluster_surface_gen is not None
        assert gen.sampler is not None

    def test_generate_structure_all_types(self):
        """Test generating all structure types."""
        config = SamplingConfig(seed=42)
        gen = StructureGenerator(config)

        # Test each system type
        mol = gen.generate_structure(SystemType.MOLECULE, target_size=30)
        assert isinstance(mol, Atoms)
        assert not mol.pbc.any()

        crystal = gen.generate_structure(SystemType.CRYSTAL, target_size=100)
        assert isinstance(crystal, Atoms)
        assert crystal.pbc.all()

        cluster = gen.generate_structure(SystemType.CLUSTER, target_size=50)
        assert isinstance(cluster, Atoms)
        assert not cluster.pbc.any()

        surface = gen.generate_structure(SystemType.SURFACE, target_size=80)
        assert isinstance(surface, Atoms)
        assert surface.pbc.any()

    def test_generate_dataset_small(self):
        """Test generating small dataset."""
        config = SamplingConfig(total_samples=50, seed=42)
        gen = StructureGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            structures = gen.generate_dataset(Path(tmpdir), max_samples=50)

            # Check structure counts
            assert len(structures) > 0
            total = sum(len(structs) for structs in structures.values())
            assert total > 0
            assert total <= 50

            # Check files saved
            assert any(Path(tmpdir).glob("*.pkl"))

    def test_save_and_load_structures(self):
        """Test saving and loading structures."""
        config = SamplingConfig(seed=42)
        gen = StructureGenerator(config)

        # Generate a few structures
        structures = [
            gen.generate_structure(SystemType.MOLECULE, 20) for _ in range(5)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            gen._save_structures(
                structures, Path(tmpdir), SystemType.MOLECULE
            )

            # Load
            filename = Path(tmpdir) / "molecule_structures.pkl"
            with open(filename, "rb") as f:
                loaded = pickle.load(f)

            assert len(loaded) == 5
            for orig, load in zip(structures, loaded):
                assert len(orig) == len(load)
                assert np.allclose(orig.positions, load.positions)

    def test_reproducibility(self):
        """Test that same seed gives same structures."""
        config1 = SamplingConfig(seed=42)
        config2 = SamplingConfig(seed=42)

        gen1 = StructureGenerator(config1)
        gen2 = StructureGenerator(config2)

        # Generate same structure
        mol1 = gen1.generate_structure(SystemType.MOLECULE, target_size=30)
        mol2 = gen2.generate_structure(SystemType.MOLECULE, target_size=30)

        # Should be identical
        assert len(mol1) == len(mol2)
        assert list(mol1.symbols) == list(mol2.symbols)
        assert np.allclose(mol1.positions, mol2.positions)


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_end_to_end_generation(self):
        """Test end-to-end structure generation pipeline."""
        # Configure for small test
        config = SamplingConfig(
            total_samples=20,
            seed=42,
            system_distribution={
                SystemType.MOLECULE: 0.5,
                SystemType.CRYSTAL: 0.5,
            },
            size_ranges={
                SystemType.MOLECULE: (10, 50),
                SystemType.CRYSTAL: (20, 100),
            },
        )

        gen = StructureGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            structures = gen.generate_dataset(Path(tmpdir), max_samples=20)

            # Check we got structures
            assert SystemType.MOLECULE in structures
            assert SystemType.CRYSTAL in structures

            # Check structure properties
            for sys_type, structs in structures.items():
                for atoms in structs:
                    assert isinstance(atoms, Atoms)
                    assert len(atoms) > 0

                    # Check periodicity
                    if sys_type == SystemType.MOLECULE:
                        assert not atoms.pbc.any()
                    elif sys_type == SystemType.CRYSTAL:
                        assert atoms.pbc.all()

    def test_diversity_check(self):
        """Test that generated structures are diverse."""
        config = SamplingConfig(
            total_samples=50,
            seed=42,
            element_set={"H", "C", "N", "O"},
        )

        gen = StructureGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            structures = gen.generate_dataset(Path(tmpdir), max_samples=50)

            # Flatten structures
            all_atoms = []
            for structs in structures.values():
                all_atoms.extend(structs)

            # Check element diversity
            all_elements = set()
            for atoms in all_atoms:
                all_elements.update(atoms.get_chemical_symbols())

            assert len(all_elements) >= 3  # At least 3 elements

            # Check size diversity
            sizes = [len(atoms) for atoms in all_atoms]
            assert len(set(sizes)) >= 10  # At least 10 different sizes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
