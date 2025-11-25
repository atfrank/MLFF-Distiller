# Quick Start: Generative Model Integration

**Status**: Ready to begin implementation
**Next Action**: Create MatterGen environment and test generation
**Timeline**: 7 days to 120K dataset (5 days to go/no-go)

---

## Immediate Next Steps (Today - Nov 24)

### Step 1: Create MatterGen Environment (30 min)

```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Install uv if not already installed
pip install uv

# Create Python 3.10 environment for MatterGen
mkdir -p envs
uv venv envs/mattergen --python 3.10

# Activate environment
source envs/mattergen/bin/activate

# Clone MatterGen (requires Git LFS for model weights)
cd /tmp
git lfs install
git clone https://github.com/microsoft/mattergen.git
cd mattergen

# Install MatterGen
uv pip install -e .

# Verify installation
python -c "import mattergen; print('MatterGen installed successfully!')"
```

### Step 2: Test Basic Generation (30 min)

```bash
# Still in MatterGen environment
export MODEL_NAME=mattergen_base
export RESULTS_PATH=/tmp/mattergen_test_001

# Generate 4 test crystals (1 batch)
mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME \
  --batch_size=4 --num_batches=1

# Check output
ls -lh $RESULTS_PATH/
# Expected files:
#   - generated_crystals_cif.zip
#   - generated_crystals.extxyz
#   - (possibly trajectories)
```

### Step 3: Convert to ASE and Validate (30 min)

```bash
# Deactivate MatterGen environment
deactivate

# Back to main environment (Python 3.13)
cd /home/aaron/ATX/software/MLFF_Distiller

# Python script to load and validate
python << 'EOF'
from ase.io import read
from pathlib import Path

# Load generated structures
extxyz_file = Path("/tmp/mattergen_test_001/generated_crystals.extxyz")

if extxyz_file.exists():
    structures = read(str(extxyz_file), index=":")
    print(f"\nLoaded {len(structures)} structures from MatterGen")

    for i, atoms in enumerate(structures):
        print(f"\nStructure {i+1}:")
        print(f"  Formula: {atoms.get_chemical_formula()}")
        print(f"  Number of atoms: {len(atoms)}")
        print(f"  Periodic: {atoms.pbc.any()}")
        print(f"  Cell volume: {atoms.get_volume():.2f} Angstrom^3")
else:
    print(f"ERROR: {extxyz_file} not found")
    print("Check MatterGen generation output")
EOF
```

### Step 4: Test with Teacher Model (30 min)

```python
# test_mattergen_with_teacher.py
import sys
from pathlib import Path
from ase.io import read

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlff_distiller.data.label_generation import OrbV2Teacher

def main():
    # Load MatterGen structures
    extxyz_file = Path("/tmp/mattergen_test_001/generated_crystals.extxyz")
    structures = read(str(extxyz_file), index=":")
    print(f"Loaded {len(structures)} structures from MatterGen")

    # Initialize teacher
    teacher = OrbV2Teacher(device="cuda")
    print("Initialized orb-v2 teacher model")

    # Test labeling
    success_count = 0
    for i, atoms in enumerate(structures):
        try:
            result = teacher.label_structure(atoms)
            print(f"\nStructure {i+1}: SUCCESS")
            print(f"  Energy: {result['energy']:.4f} eV")
            print(f"  Max force: {result['forces'].max():.4f} eV/A")
            success_count += 1
        except Exception as e:
            print(f"\nStructure {i+1}: FAILED - {e}")

    success_rate = success_count / len(structures) * 100
    print(f"\n\nValidation Success Rate: {success_rate:.1f}%")
    print(f"Target: >95%")

    if success_rate >= 95:
        print("STATUS: PASS - Proceed to wrapper implementation")
    else:
        print("STATUS: FAIL - Investigate errors")

if __name__ == "__main__":
    main()
```

Save as `scripts/test_mattergen_with_teacher.py` and run:

```bash
python scripts/test_mattergen_with_teacher.py
```

---

## Expected Outcomes

### Success Criteria:
- MatterGen environment installs without errors
- Generates 4 crystal structures successfully
- Structures load as ASE Atoms objects
- Teacher model computes labels with >95% success rate
- Generation time < 10 seconds per structure

### If Successful:
Proceed to Day 2:
- Implement `MatterGenWrapper` class
- Create standalone generation script
- Generate 100 test structures
- Compute diversity metrics

### If Issues Occur:

**Installation fails**:
- Check Python version: `python --version` (should be 3.10 in env)
- Try conda instead of uv: `conda create -n mattergen python=3.10`
- Check Git LFS: `git lfs version`

**Generation fails**:
- Check CUDA availability in MatterGen env
- Try smaller batch size: `--batch_size=1`
- Check error messages for missing dependencies

**Teacher validation fails**:
- Check if structures have valid cell parameters
- Verify periodic boundary conditions are set
- Check for NaN values in positions

**Fallback Plan**:
If MatterGen completely fails, skip to CrystalFlow (Day 3) or revert to 100% traditional methods.

---

## Key Files and Paths

### Current Environment:
- Main Python: 3.13.9
- Main repo: `/home/aaron/ATX/software/MLFF_Distiller`
- Working structures: `/home/aaron/ATX/software/MLFF_Distiller/data/raw/test_structures/` (1K samples)
- Working labels: `/home/aaron/ATX/software/MLFF_Distiller/data/labels/all_labels_orb_v2.h5` (930 labeled)

### New Environments:
- MatterGen env: `/home/aaron/ATX/software/MLFF_Distiller/envs/mattergen/`
- CrystalFlow env: `/home/aaron/ATX/software/MLFF_Distiller/envs/crystalflow/` (Day 3)

### Code to Create:
- `src/mlff_distiller/data/generative_models/__init__.py`
- `src/mlff_distiller/data/generative_models/base.py`
- `src/mlff_distiller/data/generative_models/mattergen_wrapper.py`
- `scripts/generative/generate_mattergen.py`
- `scripts/test_mattergen_with_teacher.py`

---

## Timeline at a Glance

| Day | Date | Task | Deliverable |
|-----|------|------|-------------|
| 1 | Nov 24 | MatterGen setup + test | 4-16 validated structures |
| 2 | Nov 25 | MatterGen wrapper | 100 test structures, wrapper class |
| 3 | Nov 26 | CrystalFlow setup + wrapper | 100 test structures |
| 4 | Nov 27 | Integration + 5K generation | 5K hybrid dataset |
| 5 | Nov 28 | **GO/NO-GO DECISION** | Validation report |
| 6 | Nov 29 | Scale to 120K (Batch 1-2) | 60K structures |
| 7 | Nov 30 | Scale to 120K (Batch 3-4) | 120K structures, all labeled |

---

## Contact and Questions

**Project Coordinator**: Lead Coordinator (me)
**Reference Document**: `docs/HYBRID_GENERATIVE_IMPLEMENTATION_PLAN.md`
**Status Updates**: Will update user after each major milestone

**Critical Decision Points**:
- Day 1 end: Is MatterGen working? (determines Day 2 plan)
- Day 3 end: Is CrystalFlow working? (determines 5K test distribution)
- Day 5: GO/NO-GO for 120K scaling

---

## Quick Checks

Before starting:
- [ ] Main environment has orb-v2 working
- [ ] CUDA GPU is available
- [ ] ~30 GB disk space available for environments
- [ ] Git LFS installed: `git lfs version`
- [ ] Internet connection for downloading models

During testing:
- [ ] MatterGen generates structures
- [ ] Structures are valid ASE Atoms
- [ ] Teacher validation >95%
- [ ] Generation time <10 sec/structure

After Day 1:
- [ ] Document any installation issues
- [ ] Report validation success rate
- [ ] Estimate timeline for 120K generation
- [ ] Decide on Day 2 priorities
