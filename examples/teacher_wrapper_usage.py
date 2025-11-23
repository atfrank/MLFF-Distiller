"""
Usage Examples for Teacher Model Wrapper Calculators

This script demonstrates how to use the OrbCalculator and FeNNolCalculator
wrappers as drop-in replacements in ASE MD simulations.

Examples cover:
1. Basic usage with simple molecules
2. Running MD simulations (NVE, NVT)
3. Geometry optimization
4. Working with periodic systems
5. Comparing teacher model outputs

Author: ML Architecture Designer
Date: 2025-11-23
"""

import numpy as np
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from src.models.teacher_wrappers import FeNNolCalculator, OrbCalculator


def example_1_basic_usage():
    """
    Example 1: Basic usage with OrbCalculator

    Demonstrates:
    - Loading a pretrained Orb model
    - Computing energy, forces, and stress
    - Using with ASE Atoms objects
    """
    print("=" * 80)
    print("Example 1: Basic Usage with OrbCalculator")
    print("=" * 80)

    # Create a simple molecule
    atoms = molecule("H2O")
    print(f"Created {atoms.get_chemical_formula()} molecule with {len(atoms)} atoms")

    # Initialize OrbCalculator
    # Note: Change device to 'cuda' for GPU acceleration
    calc = OrbCalculator(model_name="orb-v2", device="cpu")
    atoms.calc = calc

    # Compute properties
    energy = atoms.get_potential_energy()  # eV
    forces = atoms.get_forces()  # eV/Angstrom

    print(f"\nResults:")
    print(f"  Energy: {energy:.4f} eV")
    print(f"  Forces shape: {forces.shape}")
    print(f"  Max force magnitude: {np.max(np.linalg.norm(forces, axis=1)):.4f} eV/Angstrom")

    # For periodic systems, can also get stress
    atoms_periodic = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms_periodic.calc = calc
    stress = atoms_periodic.get_stress()  # eV/Angstrom^3
    print(f"\nPeriodic system stress: {stress}")

    print("\n")


def example_2_md_simulation_nve():
    """
    Example 2: Running NVE (microcanonical) MD simulation

    Demonstrates:
    - Setting up a velocity Verlet integrator
    - Running MD with constant energy
    - Monitoring energy conservation
    """
    print("=" * 80)
    print("Example 2: NVE (Constant Energy) MD Simulation")
    print("=" * 80)

    # Create water molecule
    atoms = molecule("H2O")
    atoms.center(vacuum=5.0)  # Add vacuum for non-periodic

    # Setup calculator
    calc = OrbCalculator(model_name="orb-v2", device="cpu")
    atoms.calc = calc

    # Initialize velocities (300 K)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # Setup MD integrator
    timestep = 0.5 * units.fs  # 0.5 fs timestep
    dyn = VelocityVerlet(atoms, timestep)

    # Track energy during simulation
    energies = []

    def energy_observer():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        energies.append((epot, ekin, etot))

        if len(energies) % 20 == 0:
            print(
                f"Step {len(energies):4d}: "
                f"Epot={epot:8.4f} eV, "
                f"Ekin={ekin:8.4f} eV, "
                f"Etot={etot:8.4f} eV"
            )

    dyn.attach(energy_observer, interval=1)

    # Run simulation
    print(f"\nRunning {100} steps of NVE MD...")
    dyn.run(100)

    # Check energy conservation
    total_energies = [e[2] for e in energies]
    energy_drift = (max(total_energies) - min(total_energies)) / abs(total_energies[0])
    print(f"\nEnergy drift: {energy_drift*100:.2f}%")

    print("\n")


def example_3_md_simulation_nvt():
    """
    Example 3: Running NVT (canonical) MD simulation with Langevin dynamics

    Demonstrates:
    - Setting up Langevin thermostat
    - Running MD at constant temperature
    - Monitoring temperature
    """
    print("=" * 80)
    print("Example 3: NVT (Constant Temperature) MD Simulation")
    print("=" * 80)

    # Create methane molecule
    atoms = molecule("CH4")
    atoms.center(vacuum=5.0)

    # Setup calculator
    calc = OrbCalculator(model_name="orb-v2", device="cpu")
    atoms.calc = calc

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # Setup Langevin dynamics (NVT ensemble)
    timestep = 1.0 * units.fs
    temperature_K = 300  # Target temperature
    friction = 0.01  # Friction coefficient (1/fs)

    dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)

    # Track temperature
    temperatures = []

    def temp_observer():
        temp = atoms.get_temperature()
        temperatures.append(temp)

        if len(temperatures) % 20 == 0:
            print(f"Step {len(temperatures):4d}: T = {temp:.2f} K")

    dyn.attach(temp_observer, interval=1)

    # Run simulation
    print(f"\nRunning {100} steps of NVT MD at {temperature_K} K...")
    dyn.run(100)

    # Check average temperature
    avg_temp = np.mean(temperatures[20:])  # Skip equilibration
    print(f"\nAverage temperature: {avg_temp:.2f} K (target: {temperature_K} K)")

    print("\n")


def example_4_geometry_optimization():
    """
    Example 4: Geometry optimization with BFGS

    Demonstrates:
    - Using calculator with ASE optimizers
    - Relaxing molecular geometry
    - Monitoring convergence
    """
    print("=" * 80)
    print("Example 4: Geometry Optimization")
    print("=" * 80)

    # Create water molecule with distorted geometry
    atoms = molecule("H2O")
    # Distort the geometry slightly
    atoms.positions += np.random.randn(len(atoms), 3) * 0.1

    # Setup calculator
    calc = OrbCalculator(model_name="orb-v2", device="cpu")
    atoms.calc = calc

    print(f"Initial geometry:")
    print(f"  Positions:\n{atoms.positions}")
    print(f"  Energy: {atoms.get_potential_energy():.4f} eV")

    # Setup optimizer
    opt = BFGS(atoms, logfile=None)

    # Optimize
    print(f"\nOptimizing geometry (fmax=0.05 eV/Angstrom)...")
    opt.run(fmax=0.05)

    print(f"\nFinal geometry:")
    print(f"  Positions:\n{atoms.positions}")
    print(f"  Energy: {atoms.get_potential_energy():.4f} eV")

    print("\n")


def example_5_periodic_system():
    """
    Example 5: Working with periodic systems

    Demonstrates:
    - Using calculator with bulk crystals
    - Computing stress for periodic systems
    - Running MD with periodic boundaries
    """
    print("=" * 80)
    print("Example 5: Periodic System (Bulk Crystal)")
    print("=" * 80)

    # Create copper FCC crystal
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    print(f"Created {atoms.get_chemical_formula()} bulk crystal")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Cell:\n{atoms.cell}")
    print(f"  PBC: {atoms.pbc}")

    # Setup calculator
    calc = OrbCalculator(model_name="orb-v2", device="cpu")
    atoms.calc = calc

    # Compute properties
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()  # Stress tensor (Voigt notation)

    print(f"\nResults:")
    print(f"  Energy: {energy:.4f} eV")
    print(f"  Energy per atom: {energy/len(atoms):.4f} eV/atom")
    print(f"  Max force: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/Angstrom")
    print(f"  Stress (Voigt): {stress}")

    # Can convert stress to GPa
    stress_gpa = stress * 160.21766208  # Conversion factor
    print(f"  Stress (GPa): {stress_gpa}")

    print("\n")


def example_6_fennol_calculator():
    """
    Example 6: Using FeNNolCalculator

    Demonstrates:
    - Loading FeNNol models
    - Using FeNNol for molecular systems
    - Comparing with Orb models
    """
    print("=" * 80)
    print("Example 6: Using FeNNolCalculator")
    print("=" * 80)

    # Create a molecule
    atoms = molecule("H2O")

    # Setup FeNNol calculator
    # Note: Requires fennol package and pretrained model
    try:
        calc = FeNNolCalculator(model_name="ani-2x", device="cpu")
        atoms.calc = calc

        # Compute properties
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        print(f"FeNNol Results (ANI-2x model):")
        print(f"  Energy: {energy:.4f} eV")
        print(f"  Forces shape: {forces.shape}")

        print("\n")

    except ImportError:
        print("FeNNol not installed. Skipping this example.")
        print("Install with: pip install fennol")
        print("\n")


def example_7_drop_in_replacement():
    """
    Example 7: Drop-in replacement demonstration

    Demonstrates:
    - How to replace original calculator with wrapper
    - That only one line needs to change
    """
    print("=" * 80)
    print("Example 7: Drop-in Replacement")
    print("=" * 80)

    print("Original MD script:")
    print("-" * 40)
    print("""
    from ase import Atoms
    from ase.md.verlet import VelocityVerlet
    from original_package import OriginalCalculator

    # Original calculator
    calc = OriginalCalculator(model='v2', device='cuda')

    atoms = Atoms(...)
    atoms.calc = calc
    dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    dyn.run(10000)
    """)

    print("\nModified MD script (only 1 line changed):")
    print("-" * 40)
    print("""
    from ase import Atoms
    from ase.md.verlet import VelocityVerlet
    from mlff_distiller.models.teacher_wrappers import OrbCalculator  # <-- NEW

    # Our wrapper (same interface!)
    calc = OrbCalculator(model_name='orb-v2', device='cuda')  # <-- CHANGED

    atoms = Atoms(...)
    atoms.calc = calc
    dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    dyn.run(10000)  # Rest is identical!
    """)

    print("\nKey point: Only the calculator import and initialization changed.")
    print("The entire MD workflow remains identical!\n")


def example_8_comparing_models():
    """
    Example 8: Comparing different Orb model versions

    Demonstrates:
    - Loading different model versions
    - Comparing predictions
    """
    print("=" * 80)
    print("Example 8: Comparing Orb Model Versions")
    print("=" * 80)

    atoms = molecule("H2O")

    model_names = ["orb-v2", "orb-v3"]

    print(f"System: {atoms.get_chemical_formula()}")
    print(f"\nComparing model predictions:\n")

    for model_name in model_names:
        try:
            calc = OrbCalculator(model_name=model_name, device="cpu")
            atoms.calc = calc

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))

            print(f"{model_name:20s}: Energy={energy:10.4f} eV, Max Force={max_force:.4f} eV/Å")

        except Exception as e:
            print(f"{model_name:20s}: Not available ({str(e)})")

    print("\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("TEACHER MODEL WRAPPER USAGE EXAMPLES")
    print("=" * 80 + "\n")

    print("NOTE: These examples use mocked calculators for demonstration.")
    print("To use real models, install: pip install orb-models fennol\n")

    try:
        # Run examples
        example_1_basic_usage()
        example_2_md_simulation_nve()
        example_3_md_simulation_nvt()
        example_4_geometry_optimization()
        example_5_periodic_system()
        example_6_fennol_calculator()
        example_7_drop_in_replacement()
        example_8_comparing_models()

        print("=" * 80)
        print("All examples completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: To run with real models, ensure orb-models is installed:")
        print("  pip install orb-models")


if __name__ == "__main__":
    main()
