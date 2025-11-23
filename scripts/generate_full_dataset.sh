#!/bin/bash
#
# Generate Full 120K Training Dataset
#
# This script generates the complete 120,000 structure dataset for
# ML force field training. It includes progress monitoring and validation.
#
# Usage:
#   bash scripts/generate_full_dataset.sh
#
# Output:
#   data/raw/full_dataset/
#
# Author: Data Pipeline Engineer
# Date: 2025-11-23

set -e  # Exit on error

# Configuration
OUTPUT_DIR="data/raw/full_dataset"
NUM_SAMPLES=120000
SEED=42

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Full Dataset Generation (120K)    ${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Output: $OUTPUT_DIR"
echo "  Samples: $NUM_SAMPLES"
echo "  Seed: $SEED"
echo ""
echo -e "${YELLOW}Distribution:${NC}"
echo "  Molecules: 60,000 (50%)"
echo "  Crystals:  39,600 (33%)"
echo "  Clusters:  12,000 (10%)"
echo "  Surfaces:   8,400 (7%)"
echo ""

# Check if output directory exists
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Warning: Output directory exists${NC}"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

echo -e "${GREEN}Starting generation...${NC}"
echo ""

# Run generation
python scripts/generate_structures.py \
    --output "$OUTPUT_DIR" \
    --num-samples $NUM_SAMPLES \
    --seed $SEED

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Generation Complete!              ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Validate output
echo -e "${YELLOW}Validating output...${NC}"

# Check files exist
EXPECTED_FILES=(
    "molecule_structures.pkl"
    "crystal_structures.pkl"
    "cluster_structures.pkl"
    "surface_structures.pkl"
    "sampling_config.json"
    "diversity_metrics.json"
)

ALL_FOUND=true
for file in "${EXPECTED_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/$file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $file ($SIZE)"
    else
        echo -e "  ${YELLOW}✗${NC} $file (missing)"
        ALL_FOUND=false
    fi
done

echo ""

if [ "$ALL_FOUND" = true ]; then
    echo -e "${GREEN}All files generated successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Inspect structures: python scripts/inspect_structures.py $OUTPUT_DIR"
    echo "  2. Run teacher inference: python scripts/run_teacher_inference.py"
    echo ""
else
    echo -e "${YELLOW}Warning: Some files are missing${NC}"
    exit 1
fi

# Display disk usage
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
echo "Total dataset size: $TOTAL_SIZE"
echo ""
echo -e "${GREEN}Dataset ready for teacher model inference!${NC}"
