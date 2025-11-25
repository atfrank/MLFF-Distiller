# Hybrid Generative Model Implementation Plan
## Project: ML Force Field Distillation - 120K Dataset Generation

**Date**: 2025-11-23
**Status**: Implementation Planning Phase
**Coordinator**: Lead Project Coordinator
**Target**: 120K diverse structures with 5-10x storage efficiency

---

## Executive Summary

All four target generative models (MolDiff, GeoDiff, MatterGen, CrystalFlow) are publicly available with pretrained weights. However, significant **Python version compatibility challenges** exist that require strategic planning.

**Key Finding**: Our current environment (Python 3.13, PyTorch 2.5.1, CUDA 12.1) is incompatible with all four generative models, which require Python 3.8-3.11 and older PyTorch versions.

**Recommended Approach**: Create isolated conda environments for each generative model, then integrate via subprocess calls or intermediate file storage.

---

## 1. Model Availability Assessment

### 1.1 MolDiff - Molecular Conformation Generation

**Status**: Publicly Available ✅

- **Repository**: https://github.com/pengxingang/MolDiff
- **Publication**: ICML 2023
- **Pretrained Models**: Available via Google Drive
  - `MolDiff.pt` (complete model)
  - `MolDiff_simple.pt` (simplified, no bond guidance)
  - `bond_predictor.pt` (bond guidance component)
- **Requirements**:
  - Python: 3.8.13
  - PyTorch: 1.10.1
  - CUDA: 11.3.1
  - PyTorch Geometric: 2.0.4
  - RDKit: 2022.03.2
- **Installation**: `conda env create -f env.yaml`
- **Output Format**: Likely requires conversion to ASE Atoms
- **Compatibility**: ⚠️ Python 3.8 required, incompatible with our Python 3.13

### 1.2 GeoDiff - Geometric Diffusion for Molecules

**Status**: Publicly Available ✅

- **Repository**: https://github.com/MinkaiXu/GeoDiff
- **Publication**: ICLR 2022 (Oral)
- **Pretrained Models**: Available via Google Drive (GEOM datasets)
- **Requirements**:
  - Python: 3.8 (recommended)
  - PyTorch: Compatible with torch-geometric 1.7.2
  - PyTorch Geometric: 1.7.2 (specific version required)
  - RDKit: Required
- **Known Issues**: "Not compatible with recent torch-geometric versions"
- **Output Format**: Molecular conformations (requires ASE conversion)
- **Compatibility**: ⚠️ Python 3.8, old torch-geometric, compatibility issues

### 1.3 MatterGen - Crystal Structure Generation

**Status**: Publicly Available ✅

- **Repository**: https://github.com/microsoft/mattergen
- **Publication**: Nature 2025 (DOI: 10.1038/s41586-025-08628-5)
- **Pretrained Models**: Available via Git LFS + HuggingFace
  - `mattergen_base` (unconditional generation)
  - Property-conditioned models (magnetic density, etc.)
- **Requirements**:
  - Python: >= 3.10
  - Installation: `uv` package manager
  - ASE: >= 3.22.1
  - PyMatGen: >= 2024.6.4
  - PyTorch Lightning: 2.0.6
  - NumPy: < 2.0 (explicit constraint)
- **Installation**:
  ```bash
  pip install uv
  uv venv .venv --python 3.10
  uv pip install -e .
  ```
- **Output Formats**: `.cif` files, `.extxyz` (ASE-compatible!)
- **Usage**:
  ```bash
  mattergen-generate results/ --pretrained-name=mattergen_base \
    --batch_size=16 --num_batches=1
  ```
- **Compatibility**: ⚠️ Python 3.10-3.12 recommended, Python 3.13 untested

### 1.4 CrystalFlow - Flow-Based Crystal Generation

**Status**: Publicly Available ✅

- **Repository**: https://github.com/ixsluo/CrystalFlow
- **Publication**: Nature Communications 2025 (arXiv: 2412.11693)
- **Pretrained Models**: Available in GitHub Releases (v1.0.0-alpha.1)
- **Requirements**:
  - Python: 3.11.9 (specific version)
  - PyTorch: 2.3.1
  - PyTorch Geometric: 2.5.3
  - Lightning: 2.3.2
  - PyMatGen, PyXtal, SMACT, Matminer, Einops, TorchDyn
- **Installation**:
  ```bash
  conda create -n crystalflow python=3.11.9
  pip install torch==2.3.1 torchvision torchaudio
  pip install torch_geometric==2.5.3
  pip install lightning==2.3.2 pymatgen pyxtal smact matminer
  pip install -e .
  ```
- **Generation Modes**:
  - CSP (Crystal Structure Prediction): from composition
  - DNG (De Novo Generation): novel structures
  - Custom composition sampling
- **Output Format**: Structure files (likely CIF, requires ASE conversion)
- **Performance**: "Order of magnitude faster than diffusion models"
- **Compatibility**: ⚠️ Python 3.11 required, incompatible with our Python 3.13

---

## 2. Critical Compatibility Analysis

### 2.1 Environment Constraints

**Current Environment**:
- Python: 3.13.9
- PyTorch: 2.5.1+cu121
- CUDA: 12.1
- ASE: >= 3.22.0

**Model Requirements**:
| Model | Python | PyTorch | Compatibility |
|-------|--------|---------|---------------|
| MolDiff | 3.8.13 | 1.10.1 | ❌ Major incompatibility |
| GeoDiff | 3.8 | 1.x | ❌ Major incompatibility |
| MatterGen | 3.10-3.12 | 2.x | ⚠️ Python 3.13 untested |
| CrystalFlow | 3.11.9 | 2.3.1 | ⚠️ Specific version required |

### 2.2 Risk Assessment

**High Risk**:
- MolDiff: Requires Python 3.8 and PyTorch 1.10.1 (3+ years old)
- GeoDiff: Known compatibility issues with modern torch-geometric

**Medium Risk**:
- CrystalFlow: Python 3.11 requirement (vs our 3.13)
- MatterGen: Python 3.13 not explicitly tested, potential dependency conflicts

**Low Risk**:
- Traditional methods (RDKit, pymatgen, ASE): Already working in our environment

### 2.3 Recommended Mitigation Strategy

**Option A: Isolated Conda Environments (Recommended)**

Create separate environments for each generative model, integrate via:
1. Subprocess calls to generation scripts
2. Intermediate file storage (JSON/pickle for metadata, .xyz/.cif for structures)
3. ASE conversion in main environment

**Pros**:
- Clean separation of dependencies
- No version conflicts
- Each model runs in optimal environment
- Easier debugging

**Cons**:
- More disk space (4 conda environments)
- Subprocess overhead
- More complex workflow

**Option B: Selective Integration**

Only integrate models compatible with Python 3.10-3.11:
- MatterGen: High priority (Nature 2025, best documented)
- CrystalFlow: Medium priority (fast, good performance)
- Skip: MolDiff, GeoDiff (use RDKit for molecules)

**Pros**:
- Fewer environments (2 instead of 4)
- Simpler integration
- Still achieve 70% generative target for crystals

**Cons**:
- Lose molecular generative models
- Rely more on traditional methods for molecules

**Option C: Docker Containers**

Package each model in Docker containers with exact dependencies.

**Pros**:
- Perfect isolation
- Reproducible environments
- Easy deployment

**Cons**:
- GPU passthrough complexity
- Development overhead
- Slower iteration

---

## 3. Recommended Implementation Plan

### Phase 1: Proof of Concept (Days 1-2)

**Objective**: Validate MatterGen integration (highest compatibility, best docs)

**Steps**:
1. Create isolated conda environment for MatterGen (Python 3.10)
2. Install MatterGen and dependencies
3. Download pretrained `mattergen_base` model
4. Write wrapper script to generate 10 test crystals
5. Convert `.extxyz` output to ASE Atoms in main environment
6. Validate with orb-v2 teacher model

**Success Criteria**:
- 10 crystal structures generated successfully
- Teacher model computes labels without errors
- Generation time < 10 sec/structure

**Deliverables**:
- `/envs/mattergen/` conda environment
- `scripts/generative/generate_mattergen.py` wrapper
- `src/mlff_distiller/data/generative_models/mattergen_wrapper.py`

### Phase 2: Expand to CrystalFlow (Days 2-3)

**Objective**: Add second crystal generator for diversity

**Steps**:
1. Create isolated conda environment for CrystalFlow (Python 3.11.9)
2. Install CrystalFlow and dependencies
3. Download pretrained checkpoint (v1.0.0-alpha.1)
4. Write wrapper for de novo generation
5. Test with 10 structures, validate with orb-v2

**Success Criteria**:
- 10 crystal structures generated
- Diversity metrics differ from MatterGen (different structural motifs)
- Teacher validation > 95% success rate

**Deliverables**:
- `/envs/crystalflow/` conda environment
- `scripts/generative/generate_crystalflow.py`
- `src/mlff_distiller/data/generative_models/crystalflow_wrapper.py`

### Phase 3: Molecular Generative Models (Days 3-4)

**Decision Point**: Evaluate effort vs reward for MolDiff/GeoDiff

**Option A**: If time permits and compatibility resolves:
- Create Python 3.8 environment for MolDiff
- Generate 100 test molecules
- Compare diversity to RDKit baseline

**Option B**: Skip and use RDKit (Fallback):
- RDKit already working in main environment
- Proven to generate valid molecules
- Good diversity with random SMILES sampling

**Recommendation**: Start with Option B (RDKit), revisit MolDiff only if Phase 1-2 complete ahead of schedule

### Phase 4: 5K Test Dataset (Day 4-5)

**Objective**: Generate hybrid dataset for go/no-go decision

**Distribution** (based on what's working):

**Scenario A: Both MatterGen + CrystalFlow working**:
- 2,500 molecules (50%): RDKit traditional
- 1,750 crystals (35%): 1,000 MatterGen + 750 CrystalFlow
- 500 clusters (10%): ASE traditional
- 250 benchmark (5%): OC20 subset

**Scenario B: Only MatterGen working**:
- 3,000 molecules (60%): RDKit
- 1,500 crystals (30%): 1,000 MatterGen + 500 pymatgen traditional
- 400 clusters (8%): ASE
- 100 benchmark (2%): OC20/MatBench

**Generation Workflow**:
1. Run generative models in isolated environments
2. Save structures to `/home/aaron/ATX/software/MLFF_Distiller/data/raw/hybrid_5k_test/`
3. Collect all structures in main environment
4. Label with orb-v2 teacher
5. Compute diversity metrics

**Success Criteria** (GO Decision):
- ✅ Teacher validation rate > 95%
- ✅ Diversity entropy >= 3.07 bits (current baseline)
- ✅ Element coverage = 100%
- ✅ No systematic errors in generative structures
- ✅ Generation time < 10 sec/structure average

**If NO-GO**:
- Increase traditional fraction to 80%
- Reduce generative to 10-20%
- Maintain 10% benchmark for validation

### Phase 5: Scale to 120K (Days 5-7, if GO)

**Final Distribution** (assuming Scenario A):

| Source | Type | Count | Method |
|--------|------|-------|--------|
| MatterGen | Crystal | 30,000 | Unconditional generation |
| CrystalFlow | Crystal | 10,000 | De novo generation |
| RDKit | Molecule | 50,000 | Random SMILES + templates |
| pymatgen | Crystal | 8,000 | Traditional crystal gen |
| ASE | Cluster | 12,000 | Cluster generation |
| ASE | Surface | 5,000 | Surface slab generation |
| OC20 | Benchmark | 3,000 | Open Catalyst subset |
| MatBench | Benchmark | 2,000 | Materials benchmarks |
| **Total** | | **120,000** | |

**Generative Fraction**: 40,000 / 120,000 = 33% (conservative)

**Storage Savings**:
- Generative models: ~1 MB total (model weights in separate envs)
- Traditional code: ~500 KB
- Total: ~1.5 MB vs 150 MB (100x savings maintained)

**Generation Workflow**:
1. **Batch 1** (20K structures, Day 5):
   - 10K MatterGen crystals
   - 10K RDKit molecules
   - Label with orb-v2, validate diversity
2. **Batch 2** (40K structures, Day 6):
   - 15K MatterGen, 3K CrystalFlow
   - 20K RDKit molecules
   - 2K pymatgen crystals
3. **Batch 3** (40K structures, Day 6):
   - 5K MatterGen, 7K CrystalFlow
   - 20K RDKit molecules
   - 6K pymatgen, 2K surfaces
4. **Batch 4** (20K structures, Day 7):
   - 12K clusters (ASE)
   - 3K surfaces (ASE)
   - 5K benchmark (OC20 + MatBench)

**Label all 120K**: Expected 30-60 minutes with CUDA GPU

---

## 4. Integration Architecture

### 4.1 Directory Structure

```
/home/aaron/ATX/software/MLFF_Distiller/
├── envs/                                    # Isolated conda environments
│   ├── mattergen/                           # Python 3.10 for MatterGen
│   ├── crystalflow/                         # Python 3.11 for CrystalFlow
│   └── moldiff/                             # Python 3.8 for MolDiff (optional)
│
├── src/mlff_distiller/data/
│   ├── generative_models/                   # NEW MODULE
│   │   ├── __init__.py
│   │   ├── base.py                          # Abstract base class
│   │   ├── mattergen_wrapper.py             # MatterGen integration
│   │   ├── crystalflow_wrapper.py           # CrystalFlow integration
│   │   ├── moldiff_wrapper.py               # MolDiff integration (optional)
│   │   └── utils.py                         # Conversion utilities
│   ├── structure_generation.py              # MODIFIED: add generative routing
│   ├── sampling.py                          # MODIFIED: add generative config
│   └── label_generation.py                  # Existing (no changes)
│
├── scripts/generative/                      # NEW: Generation scripts
│   ├── generate_mattergen.py                # Standalone MatterGen script
│   ├── generate_crystalflow.py              # Standalone CrystalFlow script
│   └── generate_moldiff.py                  # Standalone MolDiff script
│
├── data/
│   ├── raw/
│   │   ├── hybrid_5k_test/                  # 5K go/no-go test
│   │   └── hybrid_120k_full/                # Full 120K dataset
│   └── labels/
│       ├── hybrid_5k_orb_v2.h5              # 5K test labels
│       └── full_dataset_120k_orb_v2.h5      # Full 120K labels
│
└── docs/
    ├── HYBRID_GENERATIVE_IMPLEMENTATION_PLAN.md  # This document
    ├── GENERATIVE_MODELS_INTEGRATION.md          # Technical guide
    └── HYBRID_DATASET_REPORT.md                  # Validation report
```

### 4.2 API Design

**Base Class** (`src/mlff_distiller/data/generative_models/base.py`):

```python
from abc import ABC, abstractmethod
from typing import List
from ase import Atoms
from ..sampling import SamplingConfig

class GenerativeModelBase(ABC):
    """Abstract base for all generative model wrappers."""

    def __init__(self, env_path: str, model_name: str):
        """
        Args:
            env_path: Path to conda environment (e.g., '/envs/mattergen')
            model_name: Identifier for logging
        """
        self.env_path = env_path
        self.model_name = model_name

    @abstractmethod
    def generate(self, n_structures: int, config: SamplingConfig) -> List[Atoms]:
        """
        Generate structures using the model.

        Args:
            n_structures: Number of structures to generate
            config: Sampling configuration

        Returns:
            List of ASE Atoms objects
        """
        pass

    @abstractmethod
    def validate_environment(self) -> bool:
        """Check if conda environment is properly set up."""
        pass
```

**MatterGen Wrapper** (`src/mlff_distiller/data/generative_models/mattergen_wrapper.py`):

```python
import subprocess
import json
from pathlib import Path
from typing import List
from ase import Atoms
from ase.io import read
from .base import GenerativeModelBase
from ..sampling import SamplingConfig

class MatterGenWrapper(GenerativeModelBase):
    """Wrapper for Microsoft MatterGen crystal generation."""

    def __init__(self, env_path: str = "/home/aaron/ATX/software/MLFF_Distiller/envs/mattergen"):
        super().__init__(env_path, "MatterGen")
        self.python_exe = Path(env_path) / "bin" / "python"
        self.script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "generative" / "generate_mattergen.py"

    def generate(self, n_structures: int, config: SamplingConfig) -> List[Atoms]:
        """
        Generate crystals using MatterGen.

        Calls isolated script via subprocess, reads output files.
        """
        # Create temporary output directory
        output_dir = Path(f"/tmp/mattergen_{os.getpid()}")
        output_dir.mkdir(exist_ok=True)

        # Call generation script in isolated environment
        cmd = [
            str(self.python_exe),
            str(self.script_path),
            "--n_structures", str(n_structures),
            "--output_dir", str(output_dir),
            "--batch_size", "16"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MatterGen generation failed: {result.stderr}")

        # Read generated structures
        structures = []
        extxyz_file = output_dir / "generated_crystals.extxyz"
        if extxyz_file.exists():
            structures = read(str(extxyz_file), index=":")

        # Cleanup
        shutil.rmtree(output_dir)

        return structures

    def validate_environment(self) -> bool:
        """Check MatterGen environment."""
        cmd = [str(self.python_exe), "-c", "import mattergen; print('OK')"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and "OK" in result.stdout
```

**Modified SamplingConfig** (`src/mlff_distiller/data/sampling.py`):

```python
@dataclass
class SamplingConfig:
    """Configuration for structure sampling strategy."""

    total_samples: int = 120000
    seed: int = 42

    # NEW: Generative model settings
    use_generative: bool = False
    generative_fraction: float = 0.0  # Fraction to generate with ML models
    generative_models: Dict[SystemType, str] = None  # Map type to model name

    # Existing fields...
    element_set: Set[str] = None
    system_distribution: Dict[SystemType, float] = None
    size_ranges: Dict[SystemType, Tuple[int, int]] = None

    def __post_init__(self):
        # Existing validation...

        # NEW: Generative model validation
        if self.use_generative:
            if not (0.0 <= self.generative_fraction <= 1.0):
                raise ValueError("generative_fraction must be in [0, 1]")

            if self.generative_models is None:
                # Default: use MatterGen for crystals
                self.generative_models = {
                    SystemType.CRYSTAL: "mattergen"
                }
```

**Modified StructureGenerator** (`src/mlff_distiller/data/structure_generation.py`):

```python
from .generative_models import MatterGenWrapper, CrystalFlowWrapper

class StructureGenerator:
    """Main structure generation coordinator."""

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.sampler = StratifiedSampler(config)

        # Initialize traditional generators
        self.molecule_gen = MoleculeGenerator(config.seed, config.element_set)
        self.crystal_gen = CrystalGenerator(config.seed, config.element_set)
        self.cluster_gen = ClusterGenerator(config.seed, config.element_set)

        # NEW: Initialize generative models if enabled
        self.generative_wrappers = {}
        if config.use_generative:
            if "mattergen" in config.generative_models.values():
                self.generative_wrappers["mattergen"] = MatterGenWrapper()
            if "crystalflow" in config.generative_models.values():
                self.generative_wrappers["crystalflow"] = CrystalFlowWrapper()

    def generate_all(self) -> List[Tuple[Atoms, SystemType]]:
        """Generate all structures according to config."""
        all_structures = []

        for sys_type, count in self.config.get_sample_counts().items():
            # Determine how many to generate with each method
            if self.config.use_generative and sys_type in self.config.generative_models:
                gen_count = int(count * self.config.generative_fraction)
                trad_count = count - gen_count

                # Generate with ML model
                model_name = self.config.generative_models[sys_type]
                wrapper = self.generative_wrappers[model_name]
                gen_structures = wrapper.generate(gen_count, self.config)
                all_structures.extend([(s, sys_type) for s in gen_structures])

                # Generate remaining with traditional
                trad_structures = self._generate_traditional(sys_type, trad_count)
                all_structures.extend([(s, sys_type) for s in trad_structures])
            else:
                # Pure traditional generation
                structures = self._generate_traditional(sys_type, count)
                all_structures.extend([(s, sys_type) for s in structures])

        return all_structures

    def _generate_traditional(self, sys_type: SystemType, count: int) -> List[Atoms]:
        """Generate using traditional methods."""
        # Existing implementation...
```

### 4.3 Standalone Generation Scripts

**Example**: `scripts/generative/generate_mattergen.py`

```python
#!/usr/bin/env python
"""
Standalone script to generate structures with MatterGen.
Runs in isolated conda environment with Python 3.10.

Usage:
    python generate_mattergen.py --n_structures 1000 --output_dir ./output
"""

import argparse
from pathlib import Path
from mattergen import MatterGen  # This import only works in mattergen env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_structures", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pretrained_name", default="mattergen_base")
    args = parser.parse_args()

    # Initialize MatterGen
    model = MatterGen.from_pretrained(args.pretrained_name)

    # Generate structures
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_batches = (args.n_structures + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        structures = model.generate(batch_size=args.batch_size)
        # Save to .extxyz format (ASE-compatible)
        output_file = args.output_dir / f"batch_{batch_idx:04d}.extxyz"
        structures.write(str(output_file))

    print(f"Generated {args.n_structures} structures to {args.output_dir}")

if __name__ == "__main__":
    main()
```

---

## 5. Risk Mitigation and Contingency Plans

### Risk 1: Generative Models Fail to Install

**Probability**: Medium (dependency conflicts common)
**Impact**: High (blocks 70% generative target)

**Mitigation**:
- Start with MatterGen (best documented, Python 3.10)
- Have exact conda environment specs ready
- Test in isolated environment before integration

**Contingency**:
- Fall back to 100% traditional methods
- Still achieve 120K dataset, lose storage efficiency benefit
- Maintain 10% benchmark subset for validation

### Risk 2: Generated Structures Fail Teacher Validation

**Probability**: Low-Medium (models are published, pretrained)
**Impact**: High (unusable data)

**Mitigation**:
- Test with 10 structures per model before scaling
- Monitor validation rate in 5K test
- Set hard threshold: >95% validation rate

**Contingency**:
- Reduce generative fraction (70% → 50% → 30%)
- Filter out failed structures, regenerate
- Use only validated generative structures

### Risk 3: Generation Too Slow

**Probability**: Low (models claim efficiency)
**Impact**: Medium (timeline delay)

**Mitigation**:
- Benchmark on 10 structures before 5K test
- Use batch generation where possible
- Run generation overnight if needed

**Contingency**:
- Reduce generative fraction
- Parallelize across multiple GPUs if available
- Extend timeline by 1-2 days

### Risk 4: Python 3.13 Incompatibility Issues

**Probability**: High (Python 3.13 is very new)
**Impact**: Medium (requires environment isolation)

**Mitigation**:
- Already planned: isolated conda environments
- Use subprocess calls, not direct imports
- Test environment creation in Phase 1

**Contingency**:
- If conda environments fail, use Docker
- Worst case: run on separate machine with Python 3.10

### Risk 5: Storage or Memory Issues

**Probability**: Low
**Impact**: Medium

**Mitigation**:
- Generate in batches, clean up intermediate files
- Monitor disk usage during generation
- Use streaming writes for HDF5 labels

**Contingency**:
- Split 120K into smaller datasets
- Use external storage if needed
- Compress intermediate files

---

## 6. Success Metrics and Validation

### 6.1 Go/No-Go Criteria (5K Test, Day 5)

**MUST PASS ALL**:
1. Teacher validation rate > 95% for generative structures
2. Diversity entropy >= 3.07 bits (current baseline)
3. Element coverage = 100% (all 9 target elements)
4. No systematic errors (e.g., all structures same size, same composition)
5. Generation time < 10 sec/structure average

**ADDITIONAL QUALITY CHECKS**:
6. Bond length distribution reasonable (compare to traditional)
7. Energy distribution plausible (no extreme outliers)
8. Force magnitudes in expected range

### 6.2 Final Dataset Validation (120K, Day 7)

**Quantitative Metrics**:
- Total structures: 120,000
- Teacher labeling success rate: > 95%
- Diversity entropy: >= 3.07 bits
- Element coverage: 100%
- Size distribution: Mean ~100-150 atoms, CV > 0.5
- System type balance: CV < 0.3

**Qualitative Checks**:
- Visual inspection of 100 random structures
- Chemistry validation (no impossible bonds, valences)
- Physics validation (no overlapping atoms, reasonable geometries)

**Storage Efficiency**:
- Generation code + model refs: < 10 MB
- vs storing raw 120K structures: > 1 GB
- Savings: > 100x

---

## 7. Timeline and Milestones

### Week 1: Setup and Proof of Concept

**Day 1 (Nov 24) - Sunday**:
- [x] Research model availability (COMPLETED)
- [ ] Create MatterGen conda environment
- [ ] Install MatterGen, test basic generation
- [ ] Generate 10 test crystals, validate with orb-v2

**Day 2 (Nov 25) - Monday**:
- [ ] Implement MatterGen wrapper class
- [ ] Write standalone generation script
- [ ] Generate 100 test crystals
- [ ] Compute diversity metrics, compare to pymatgen baseline

**Day 3 (Nov 26) - Tuesday**:
- [ ] Create CrystalFlow conda environment
- [ ] Install CrystalFlow, download pretrained model
- [ ] Generate 10 test crystals
- [ ] Implement CrystalFlow wrapper

**Day 4 (Nov 27) - Wednesday**:
- [ ] Integrate wrappers into StructureGenerator
- [ ] Modify SamplingConfig for generative models
- [ ] Generate 5K hybrid test dataset
- [ ] Label with orb-v2 teacher

**Day 5 (Nov 28) - Thursday**: **GO/NO-GO DECISION**
- [ ] Analyze 5K test results
- [ ] Compute all diversity metrics
- [ ] Validate generation quality
- [ ] **DECISION**: GO (scale to 120K) or NO-GO (fall back)

### Week 2: Production Scale (if GO)

**Day 6 (Nov 29) - Friday**:
- [ ] Generate Batch 1: 20K structures
- [ ] Label Batch 1 with orb-v2
- [ ] Validate quality, adjust if needed
- [ ] Generate Batch 2: 40K structures

**Day 7 (Nov 30) - Saturday**:
- [ ] Label Batch 2
- [ ] Generate Batch 3: 40K structures
- [ ] Generate Batch 4: 20K structures
- [ ] Label all remaining structures

**Day 8 (Dec 1) - Sunday**:
- [ ] Final validation of 120K dataset
- [ ] Compute comprehensive diversity analysis
- [ ] Write validation report
- [ ] Update documentation

---

## 8. Deliverables

### Code Deliverables

1. **New Module**: `src/mlff_distiller/data/generative_models/`
   - `__init__.py`: Exports and imports
   - `base.py`: Abstract base class (150 lines)
   - `mattergen_wrapper.py`: MatterGen integration (200 lines)
   - `crystalflow_wrapper.py`: CrystalFlow integration (200 lines)
   - `moldiff_wrapper.py`: MolDiff integration (optional, 200 lines)
   - `utils.py`: Conversion utilities (100 lines)

2. **Modified Files**:
   - `src/mlff_distiller/data/sampling.py`: Add generative config fields (50 lines added)
   - `src/mlff_distiller/data/structure_generation.py`: Add generative routing (100 lines added)

3. **Standalone Scripts**: `scripts/generative/`
   - `generate_mattergen.py`: MatterGen generation (100 lines)
   - `generate_crystalflow.py`: CrystalFlow generation (100 lines)
   - `generate_moldiff.py`: MolDiff generation (optional, 100 lines)

4. **Environment Specs**:
   - `envs/mattergen/environment.yml`: Exact dependencies
   - `envs/crystalflow/environment.yml`: Exact dependencies
   - `envs/moldiff/environment.yml`: Optional

5. **Tests**:
   - `tests/unit/test_generative_wrappers.py`: Unit tests for wrappers
   - `tests/integration/test_hybrid_generation.py`: End-to-end integration test

### Data Deliverables

1. **5K Test Dataset**:
   - `data/raw/hybrid_5k_test/`: 5,000 structures (various formats)
   - `data/labels/hybrid_5k_orb_v2.h5`: Teacher labels
   - Size: ~500 MB (structures) + 200 MB (labels)

2. **120K Full Dataset**:
   - `data/raw/hybrid_120k_full/`: 120,000 structures
   - `data/labels/full_dataset_120k_orb_v2.h5`: All labels
   - Size: ~10 GB (structures) + 4 GB (labels)

3. **Generation Code** (compact representation):
   - Total size: < 10 MB (vs > 1 GB for raw structures)
   - Reproducible with seed

### Documentation Deliverables

1. **Technical Guide**: `docs/GENERATIVE_MODELS_INTEGRATION.md`
   - Installation instructions for each model
   - API documentation for wrappers
   - Troubleshooting guide
   - Examples and usage patterns

2. **Validation Report**: `docs/HYBRID_DATASET_REPORT.md`
   - 5K test results and go/no-go decision rationale
   - 120K final dataset statistics
   - Diversity analysis and comparison to baselines
   - Storage efficiency measurements
   - Quality validation results

3. **Updated Coordination Plan**: `docs/M2_COORDINATION_PLAN.md`
   - Reflect hybrid approach
   - Update timeline
   - Document deviations from original plan

---

## 9. Comparison to Original Plan

### Original Approach (from M2_COORDINATION_PLAN.md):

**Source Distribution**:
- OC20: 20K structures (17%)
- OC22: 20K structures (17%)
- MatBench Discovery: 80K structures (67%)

**Pros**:
- Proven datasets
- High quality labels
- Well-documented

**Cons**:
- Download size: 100+ GB
- Processing overhead
- Less control over diversity
- No storage savings

### Hybrid Generative Approach (This Plan):

**Source Distribution**:
- Generative models: 40K structures (33%)
  - MatterGen: 30K crystals
  - CrystalFlow: 10K crystals
- Traditional methods: 72K structures (60%)
  - RDKit: 50K molecules
  - pymatgen: 8K crystals
  - ASE: 17K clusters + surfaces
- Benchmarks: 8K structures (7%)
  - OC20: 3K
  - MatBench: 5K

**Pros**:
- 100x storage savings (< 10 MB vs > 1 GB)
- Full control over diversity
- Custom element coverage
- Faster to generate than download + process
- Reproducible with seed

**Cons**:
- Requires conda environment isolation
- More complex initial setup
- Validation uncertainty (need 5K test)
- Newer, less proven approach

### Recommendation

**Proceed with Hybrid Generative Approach** because:

1. **Storage Efficiency**: 100x savings critical for git repository
2. **Diversity Control**: Can target exact element distributions
3. **Feasibility**: All models publicly available with pretrained weights
4. **Risk Managed**: 5K test provides go/no-go decision point
5. **Fallback Ready**: Can revert to traditional + benchmark if needed

**Worst Case**: If all generative models fail (unlikely), we still have:
- Traditional methods working (1K structures proven)
- Benchmark subsets (10-20K structures)
- Total: 100K+ structures achievable

---

## 10. Next Steps (Immediate Actions)

### Today (Nov 24):

1. **Create MatterGen Environment** (1 hour):
   ```bash
   cd /home/aaron/ATX/software/MLFF_Distiller
   mkdir -p envs
   pip install uv
   uv venv envs/mattergen --python 3.10
   source envs/mattergen/bin/activate
   cd /tmp
   git clone https://github.com/microsoft/mattergen.git
   cd mattergen
   uv pip install -e .
   ```

2. **Test MatterGen Generation** (30 min):
   ```bash
   cd /tmp/mattergen
   export MODEL_NAME=mattergen_base
   export RESULTS_PATH=/tmp/mattergen_test
   mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=4 --num_batches=1
   ```

3. **Validate with Teacher** (30 min):
   - Convert `.extxyz` to ASE Atoms in main environment
   - Load orb-v2 teacher
   - Compute labels for 4-16 test structures
   - Check success rate

4. **Report Findings** (30 min):
   - Document any installation issues
   - Report validation success rate
   - Estimate generation time per structure
   - Confirm feasibility for Phase 2

### Tomorrow (Nov 25):

1. Implement `MatterGenWrapper` class
2. Write `generate_mattergen.py` standalone script
3. Generate 100 test crystals
4. Compute diversity metrics
5. Begin CrystalFlow environment setup

---

## 11. Open Questions and Decisions Needed

### Technical Questions:

1. **MolDiff Priority**: Should we invest time in Python 3.8 environment for MolDiff, or use RDKit for all molecules?
   - **Recommendation**: Skip MolDiff initially, RDKit is working and proven

2. **Batch Size**: What batch size for generative models?
   - Start with 16 (MatterGen default)
   - Tune based on GPU memory and speed

3. **Checkpointing**: Save intermediate batches during 120K generation?
   - **Yes**: Save every 10K structures for safety

4. **Diversity Target**: Should we aim for higher diversity than 3.07 bits?
   - Maintain >= 3.07 (current baseline)
   - Stretch goal: 3.5 bits

### Process Questions:

1. **Go/No-Go Authority**: Who makes final decision on Day 5?
   - **Recommendation**: Lead Coordinator (me) with user approval

2. **Timeline Flexibility**: Can we extend by 1-2 days if needed?
   - Confirm with user

3. **Resource Allocation**: Any GPU time limits or disk space constraints?
   - Check available resources

---

## 12. References and Resources

### Model Repositories:
- [MolDiff GitHub](https://github.com/pengxingang/MolDiff)
- [GeoDiff GitHub](https://github.com/MinkaiXu/GeoDiff)
- [MatterGen GitHub](https://github.com/microsoft/mattergen)
- [MatterGen HuggingFace](https://huggingface.co/microsoft/mattergen)
- [CrystalFlow GitHub](https://github.com/ixsluo/CrystalFlow)

### Publications:
- MolDiff: [ICML 2023 Proceedings](https://dl.acm.org/doi/10.5555/3618408.3619557)
- GeoDiff: [ICLR 2022 OpenReview](https://openreview.net/forum?id=PzcvxEMzvQC)
- MatterGen: [Nature 2025](https://www.nature.com/articles/s41586-025-08628-5)
- CrystalFlow: [Nature Communications 2025](https://www.nature.com/articles/s41467-025-64364-4)
- CrystalFlow: [arXiv:2412.11693](https://arxiv.org/abs/2412.11693)

### Current Project Files:
- Sampling Config: `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/sampling.py`
- Structure Generation: `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/structure_generation.py`
- Label Generation: `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/label_generation.py`
- Test Structures: `/home/aaron/ATX/software/MLFF_Distiller/data/raw/test_structures/`
- Current Labels: `/home/aaron/ATX/software/MLFF_Distiller/data/labels/all_labels_orb_v2.h5`

---

## Appendix A: Installation Command Reference

### MatterGen (Python 3.10)

```bash
# Create environment
cd /home/aaron/ATX/software/MLFF_Distiller
mkdir -p envs
pip install uv
uv venv envs/mattergen --python 3.10
source envs/mattergen/bin/activate

# Clone and install
cd /tmp
git lfs install  # Required for model weights
git clone https://github.com/microsoft/mattergen.git
cd mattergen
uv pip install -e .

# Test installation
python -c "import mattergen; print('MatterGen OK')"
```

### CrystalFlow (Python 3.11.9)

```bash
# Create environment
conda create -n crystalflow python=3.11.9
conda activate crystalflow

# Install PyTorch and dependencies
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://pytorch-geometric.com/whl/torch-2.3.1+cu121.html

# Install CrystalFlow dependencies
pip install lightning==2.3.2 finetuning_scheduler
pip install hydra-core omegaconf python-dotenv wandb
pip install pymatgen pyxtal smact matminer einops chemparse torchdyn
pip install p_tqdm

# Clone and install CrystalFlow
cd /tmp
git clone https://github.com/ixsluo/CrystalFlow.git
cd CrystalFlow
pip install -e .

# Download pretrained model
# (Check Releases: https://github.com/ixsluo/CrystalFlow/releases)
wget https://github.com/ixsluo/CrystalFlow/releases/download/v1.0.0-alpha.1/checkpoint.ckpt -O pretrained/checkpoint.ckpt

# Test installation
python -c "import crystalflow; print('CrystalFlow OK')"
```

### MolDiff (Python 3.8) - Optional

```bash
# Create environment
conda create -n moldiff python=3.8.13
conda activate moldiff

# Install PyTorch 1.10.1 with CUDA 11.3
conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch

# Install PyTorch Geometric 2.0.4
pip install torch-geometric==2.0.4
pip install torch-scatter torch-sparse torch-cluster torch-spline_conv -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html

# Install RDKit and other dependencies
conda install -c conda-forge rdkit=2022.03.2
pip install pyyaml easydict python-lmdb

# Clone MolDiff
cd /tmp
git clone https://github.com/pengxingang/MolDiff.git
cd MolDiff

# Download pretrained models
# (From Google Drive, see README)
mkdir ckpt
# Manual download required: MolDiff.pt, MolDiff_simple.pt, bond_predictor.pt

# Test installation
python -c "import torch; import torch_geometric; import rdkit; print('MolDiff OK')"
```

---

## Appendix B: Expected File Sizes

| Item | Size | Notes |
|------|------|-------|
| **Environments** | | |
| MatterGen env | ~5 GB | PyTorch, MatterGen, dependencies |
| CrystalFlow env | ~5 GB | PyTorch, PyG, dependencies |
| MolDiff env (optional) | ~4 GB | Older PyTorch, RDKit |
| **Models** | | |
| MatterGen weights | ~500 MB | Via Git LFS |
| CrystalFlow checkpoint | ~200 MB | v1.0.0-alpha.1 |
| MolDiff checkpoints | ~300 MB | 3 model files |
| **Generation Code** | | |
| Wrappers + scripts | < 10 MB | Python code only |
| **Data** | | |
| 5K test structures | ~500 MB | Mixed formats |
| 5K test labels (HDF5) | ~200 MB | Energy, forces, stress |
| 120K structures | ~10 GB | Mixed formats |
| 120K labels (HDF5) | ~4 GB | Energy, forces, stress |
| **Total (with all 3 models)** | ~30 GB | Mostly environments |
| **Total (MatterGen + CrystalFlow)** | ~25 GB | Recommended |
| **Repository footprint** | < 10 MB | Code only, models in separate envs |

---

## Document Control

- **Created**: 2025-11-23
- **Author**: Lead Project Coordinator
- **Status**: Active Planning
- **Next Review**: After 5K test (Day 5)
- **Related Documents**:
  - M2_COORDINATION_PLAN.md
  - DATASET_GENERATION_STRATEGY.md (if created)
- **Version**: 1.0
