# [Testing] [M1] Create MD simulation benchmark framework

## Assigned Agent
testing-benchmark-engineer

## Milestone
M1: Setup & Baseline

## Priority
CRITICAL - Required to measure MD performance targets

## Task Description
Create a comprehensive benchmarking framework that measures performance on molecular dynamics workloads, not just single inference. This framework will track the 5-10x speedup target on realistic MD trajectories and ensure models are optimized for repeated inference.

**CRITICAL**: Benchmarks must measure full MD trajectories (1000+ steps), not just single inference calls.

## Context & Background
Single inference benchmarks don't capture MD simulation performance because:
1. MD calls models millions of times - overhead per call matters
2. Memory management over long runs is critical
3. Device transfer overhead accumulates
4. JIT compilation effects differ
5. Cache behavior differs with repeated calls

We need benchmarks that measure:
- Total MD trajectory wall time
- Per-step latency (average, min, max, std)
- Memory usage over time
- Energy conservation (correctness check)
- Throughput for batch/parallel MD

## Acceptance Criteria
- [ ] Benchmark framework in `benchmarks/md_benchmarks.py`
- [ ] Support for multiple MD protocols:
  - [ ] NVE (microcanonical) - energy conservation check
  - [ ] NVT (canonical) - Langevin dynamics
  - [ ] NPT (isothermal-isobaric) - pressure coupling
- [ ] Measure key metrics:
  - [ ] Total trajectory time (wall clock)
  - [ ] Per-step inference time (mean, std, min, max)
  - [ ] Memory usage (initial, peak, final)
  - [ ] Energy conservation (for NVE)
  - [ ] Throughput (steps/second)
- [ ] Support for different system sizes:
  - [ ] Small (32-64 atoms)
  - [ ] Medium (128-256 atoms)
  - [ ] Large (512-1024 atoms)
- [ ] Benchmark both single and batched inference (parallel MD)
- [ ] Compare teacher vs student models on same trajectories
- [ ] JSON output for tracking over time
- [ ] Plotting utilities for visualization
- [ ] CI integration for regression detection
- [ ] Documentation and usage examples

## Technical Notes

### Benchmark Framework Design
```python
# benchmarks/md_benchmarks.py
import time
import torch
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
import json

class MDBenchmark:
    def __init__(self, calculator, system_size, protocol='NVE'):
        self.calculator = calculator
        self.system_size = system_size
        self.protocol = protocol
        self.results = {}

    def setup_system(self):
        """Create benchmark system"""
        # Create atoms object with specified size
        pass

    def run_trajectory(self, n_steps=1000):
        """Run MD trajectory and collect metrics"""
        atoms = self.setup_system()
        atoms.calc = self.calculator

        if self.protocol == 'NVE':
            dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
        elif self.protocol == 'NVT':
            dyn = Langevin(atoms, timestep=1.0*units.fs,
                          temperature_K=300, friction=0.01)

        # Track metrics
        energies = []
        times = []
        memory_usage = []

        start_time = time.time()
        for i in range(n_steps):
            step_start = time.time()
            dyn.run(1)
            step_time = time.time() - step_start

            times.append(step_time)
            energies.append(atoms.get_potential_energy())

            if i % 100 == 0:
                memory_usage.append(torch.cuda.memory_allocated() / 1e9)

        total_time = time.time() - start_time

        # Compute metrics
        self.results = {
            'total_time': total_time,
            'steps_per_second': n_steps / total_time,
            'mean_step_time': np.mean(times),
            'std_step_time': np.std(times),
            'min_step_time': np.min(times),
            'max_step_time': np.max(times),
            'peak_memory_gb': max(memory_usage),
            'energy_drift': self._compute_energy_drift(energies),
        }

        return self.results

    def _compute_energy_drift(self, energies):
        """Compute energy drift for NVE (should be near zero)"""
        if self.protocol != 'NVE':
            return None
        return (energies[-1] - energies[0]) / abs(energies[0])

    def save_results(self, filename):
        """Save benchmark results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

    def compare_with_baseline(self, baseline_results):
        """Compare current run with baseline"""
        speedup = baseline_results['total_time'] / self.results['total_time']
        print(f"Speedup: {speedup:.2f}x")
        return speedup

# Usage example
def benchmark_teacher_vs_student():
    from mlff_distiller.calculators import OrbTeacherCalculator, DistilledOrbCalculator

    teacher_calc = OrbTeacherCalculator(model="orb-v2", device="cuda")
    student_calc = DistilledOrbCalculator(model="orb-v2-distilled", device="cuda")

    # Benchmark teacher
    teacher_bench = MDBenchmark(teacher_calc, system_size=128, protocol='NVE')
    teacher_results = teacher_bench.run_trajectory(n_steps=1000)

    # Benchmark student
    student_bench = MDBenchmark(student_calc, system_size=128, protocol='NVE')
    student_results = student_bench.run_trajectory(n_steps=1000)

    # Compare
    speedup = teacher_results['total_time'] / student_results['total_time']
    print(f"Student speedup: {speedup:.2f}x (target: 5-10x)")
```

### Key Metrics to Track
1. **Performance**:
   - Total trajectory time (primary metric)
   - Per-step latency (mean, std)
   - Throughput (steps/second)
   - Speedup vs teacher

2. **Memory**:
   - Initial memory
   - Peak memory
   - Final memory (check for leaks)
   - Memory growth rate

3. **Correctness**:
   - Energy conservation (NVE)
   - Energy distribution (NVT)
   - Pressure stability (NPT)
   - Force correctness

4. **Scalability**:
   - Performance vs system size
   - Batched inference speedup
   - Multi-GPU scaling

## Related Issues
- Depends on: #6 (teacher calculator)
- Related to: #26 (student calculator)
- Enables: #23 (baseline benchmarks)
- Related to: #18 (profiling MD trajectories)
- Related to: #25 (performance regression tests)

## Dependencies
- ase
- torch
- numpy
- matplotlib (for plotting)
- pytest (for regression tests)

## Estimated Complexity
Medium (3-4 days)

## Definition of Done
- [ ] Benchmark framework implemented
- [ ] All acceptance criteria met
- [ ] Supports NVE, NVT, NPT protocols
- [ ] Measures all key metrics
- [ ] Works with multiple system sizes
- [ ] JSON output format
- [ ] Plotting utilities
- [ ] CI integration
- [ ] Documentation with examples
- [ ] Tested on teacher and student models
- [ ] PR created and reviewed
- [ ] PR merged to main

## Success Metrics
- Can measure 5-10x speedup target on MD trajectories
- Benchmarks run in reasonable time (< 10 min for full suite)
- Clear, actionable output
- Reproducible results (< 5% variance)
- Catches performance regressions
