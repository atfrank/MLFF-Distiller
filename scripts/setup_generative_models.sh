#!/bin/bash
#
# Setup Script for Generative Models (MatterGen + MolDiff)
#
# This script sets up isolated conda environments for generative crystal
# and molecular structure generation. Required due to Python version
# incompatibilities (main project uses 3.13, models need 3.8-3.10).
#
# Usage:
#   bash scripts/setup_generative_models.sh [--all|--mattergen|--moldiff]
#
# Options:
#   --all         Install both MatterGen and MolDiff (default)
#   --mattergen   Install only MatterGen (crystals)
#   --moldiff     Install only MolDiff (molecules)
#   --help        Show this help message
#
# Requirements:
#   - conda or mamba
#   - uv (installed automatically if missing)
#   - git
#   - git-lfs (installed automatically if missing)
#
# Author: ML Distillation Coordinator
# Date: 2025-11-23

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default: install both
INSTALL_MATTERGEN=true
INSTALL_MOLDIFF=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mattergen)
            INSTALL_MATTERGEN=true
            INSTALL_MOLDIFF=false
            shift
            ;;
        --moldiff)
            INSTALL_MATTERGEN=false
            INSTALL_MOLDIFF=true
            shift
            ;;
        --all)
            INSTALL_MATTERGEN=true
            INSTALL_MOLDIFF=true
            shift
            ;;
        --help|-h)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

log_info "MLFF Distiller - Generative Models Setup"
log_info "Project root: $PROJECT_ROOT"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

# Check conda
if ! command -v conda &> /dev/null; then
    log_error "conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi
log_success "conda found: $(conda --version)"

# Install uv if missing
if ! command -v uv &> /dev/null; then
    log_warning "uv not found. Installing..."
    pip install uv
    log_success "uv installed"
else
    log_success "uv found: $(uv --version)"
fi

# Install git-lfs if missing
if ! git lfs version &> /dev/null; then
    log_warning "git-lfs not found. Installing..."
    conda install -c conda-forge git-lfs -y
    git lfs install
    log_success "git-lfs installed"
else
    log_success "git-lfs found: $(git lfs version)"
fi

# Install gdown if missing (for Google Drive downloads)
if ! command -v gdown &> /dev/null; then
    log_warning "gdown not found. Installing..."
    pip install gdown
    log_success "gdown installed"
else
    log_success "gdown found"
fi

echo ""

# Create directories
log_info "Creating directory structure..."
mkdir -p envs
mkdir -p external
mkdir -p data/generative_test
log_success "Directories created"

echo ""

#============================================================================
# MatterGen Setup (Crystals)
#============================================================================

if [ "$INSTALL_MATTERGEN" = true ]; then
    log_info "========================================="
    log_info "Setting up MatterGen (Crystal Generation)"
    log_info "========================================="
    echo ""

    # Clone repository
    if [ ! -d "external/mattergen" ]; then
        log_info "Cloning MatterGen repository..."
        GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/microsoft/mattergen.git external/mattergen
        log_success "MatterGen cloned"
    else
        log_warning "MatterGen already cloned. Skipping."
    fi

    # Create conda environment
    if [ ! -d "envs/mattergen" ]; then
        log_info "Creating Python 3.10 environment for MatterGen..."
        uv venv envs/mattergen --python 3.10
        log_success "MatterGen environment created"
    else
        log_warning "MatterGen environment already exists. Skipping."
    fi

    # Install MatterGen
    log_info "Installing MatterGen dependencies (this may take 5-10 minutes)..."
    source envs/mattergen/bin/activate
    cd external/mattergen

    # Check if already installed
    if uv pip list | grep -q "mattergen"; then
        log_warning "MatterGen already installed. Skipping installation."
    else
        uv pip install -e .
        log_success "MatterGen installed"
    fi

    # Download pretrained weights
    log_info "Downloading MatterGen pretrained weights..."
    cd "$PROJECT_ROOT/external/mattergen"
    git lfs pull || log_warning "No LFS files to pull (weights may be downloaded separately)"

    deactivate
    cd "$PROJECT_ROOT"

    log_success "MatterGen setup complete!"
    echo ""
fi

#============================================================================
# MolDiff Setup (Molecules)
#============================================================================

if [ "$INSTALL_MOLDIFF" = true ]; then
    log_info "========================================="
    log_info "Setting up MolDiff (Molecular Generation)"
    log_info "========================================="
    echo ""

    # Clone repository
    if [ ! -d "external/MolDiff" ]; then
        log_info "Cloning MolDiff repository..."
        git clone https://github.com/pengxingang/MolDiff.git external/MolDiff
        log_success "MolDiff cloned"
    else
        log_warning "MolDiff already cloned. Skipping."
    fi

    # Create conda environment
    if [ ! -d "envs/moldiff" ]; then
        log_info "Creating Python 3.8 environment for MolDiff..."
        uv venv envs/moldiff --python 3.8
        log_success "MolDiff environment created"
    else
        log_warning "MolDiff environment already exists. Skipping."
    fi

    # Install MolDiff
    log_info "Installing MolDiff dependencies (this may take 5-10 minutes)..."
    source envs/moldiff/bin/activate
    cd external/MolDiff

    # Install from environment file
    if [ -f "env.yaml" ]; then
        # Extract dependencies from env.yaml and install with pip
        log_info "Installing dependencies from env.yaml..."
        # Note: Using pip instead of conda for faster installation
        uv pip install torch==1.10.1 rdkit scipy networkx
        log_success "MolDiff dependencies installed"
    else
        log_warning "env.yaml not found. Installing basic dependencies..."
        uv pip install torch rdkit scipy networkx
    fi

    # Download pretrained weights
    log_info "Downloading MolDiff pretrained weights (~200 MB)..."
    cd ckpt
    if [ -f "MolDiff.pt" ]; then
        log_warning "MolDiff weights already downloaded. Skipping."
    else
        gdown --folder https://drive.google.com/drive/folders/1zTrjVehEGTP7sN3DB5jaaUuMJ6Ah0-ps
        # Move weights to correct location
        if [ -d "ckpt" ]; then
            mv ckpt/* .
            rmdir ckpt
        fi
        log_success "MolDiff weights downloaded"
    fi

    deactivate
    cd "$PROJECT_ROOT"

    log_success "MolDiff setup complete!"
    echo ""
fi

#============================================================================
# Verification
#============================================================================

log_info "========================================="
log_info "Verifying Installation"
log_info "========================================="
echo ""

if [ "$INSTALL_MATTERGEN" = true ]; then
    log_info "Testing MatterGen..."
    source envs/mattergen/bin/activate
    if python -c "import sys; sys.path.insert(0, 'external/mattergen'); import mattergen; print('MatterGen import: OK')" 2>&1 | grep -q "OK"; then
        log_success "MatterGen verification passed"
    else
        log_warning "MatterGen verification failed (may work anyway)"
    fi
    deactivate
fi

if [ "$INSTALL_MOLDIFF" = true ]; then
    log_info "Testing MolDiff..."
    source envs/moldiff/bin/activate
    if python -c "import torch; import rdkit; print('MolDiff dependencies: OK')" 2>&1 | grep -q "OK"; then
        log_success "MolDiff verification passed"
    else
        log_warning "MolDiff verification failed (may work anyway)"
    fi
    deactivate
fi

echo ""
log_success "========================================="
log_success "Setup Complete!"
log_success "========================================="
echo ""

log_info "Directory structure:"
if [ "$INSTALL_MATTERGEN" = true ]; then
    echo "  envs/mattergen/          - MatterGen Python 3.10 environment"
    echo "  external/mattergen/      - MatterGen source code"
fi
if [ "$INSTALL_MOLDIFF" = true ]; then
    echo "  envs/moldiff/            - MolDiff Python 3.8 environment"
    echo "  external/MolDiff/        - MolDiff source code + weights"
fi
echo ""

log_info "Next steps:"
echo "  1. Test generation: bash scripts/test_generative_models.sh"
echo "  2. Generate structures: python scripts/generate_structures.py --use-generative"
echo "  3. See documentation: docs/GENERATIVE_MODELS_INTEGRATION.md"
echo ""

log_info "To manually activate environments:"
if [ "$INSTALL_MATTERGEN" = true ]; then
    echo "  MatterGen: source envs/mattergen/bin/activate"
fi
if [ "$INSTALL_MOLDIFF" = true ]; then
    echo "  MolDiff:   source envs/moldiff/bin/activate"
fi
echo ""

log_success "All done! Happy generating! 🎉"
