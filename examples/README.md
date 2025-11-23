# Examples

This directory contains example scripts and notebooks demonstrating how to use MLFF Distiller.

## Available Examples

### Data Pipeline
- `data_loading_example.py` - Load and batch molecular structures (Coming in M1)
- `data_generation_example.py` - Generate training data from teacher models (Coming in M2)

### Model Usage
- `teacher_inference_example.py` - Use teacher model wrappers (Coming in M1)
- `student_inference_example.py` - Use distilled student models (Coming in M3)

### Training
- `train_example.py` - Train a student model (Coming in M4)
- `evaluate_model.py` - Evaluate model accuracy (Coming in M4)

### Optimization
- `benchmark_models.py` - Benchmark inference performance (Coming in M1)
- `cuda_optimized_inference.py` - Use CUDA-optimized models (Coming in M5)

## Running Examples

```bash
# Install the package first
pip install -e .

# Run an example
python examples/data_loading_example.py
```

## Contributing Examples

When adding new examples:
1. Keep examples simple and focused
2. Add docstrings explaining what the example demonstrates
3. Include comments for clarity
4. Ensure examples run with minimal setup
5. Update this README with description
