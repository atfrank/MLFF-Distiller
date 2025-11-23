# Contributing to MLFF Distiller

Thank you for your interest in contributing to MLFF Distiller! This guide will help you get started.

## Multi-Agent Development Model

This project uses a specialized multi-agent development approach where different agents handle specific domains:

### Agent Roles

1. **Data Pipeline Engineer** (`data-pipeline-engineer`)
   - Data generation from teacher models
   - Preprocessing and augmentation pipelines
   - Dataset management and validation
   - Data format conversions

2. **ML Architecture Designer** (`ml-architecture-designer`)
   - Student model architecture design
   - Teacher model integration
   - Model component interfaces
   - Architecture research and prototyping

3. **Distillation Training Engineer** (`distillation-training-engineer`)
   - Training loop implementation
   - Loss function design
   - Hyperparameter tuning
   - Training monitoring and logging

4. **CUDA Optimization Engineer** (`cuda-optimization-engineer`)
   - CUDA kernel development
   - Memory optimization
   - Performance profiling
   - Inference optimization

5. **Testing & Benchmark Engineer** (`testing-benchmark-engineer`)
   - Test framework setup
   - Unit and integration tests
   - Benchmark suite development
   - Performance regression detection

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- CUDA 12.x (for GPU acceleration)
- Basic knowledge of PyTorch and molecular dynamics

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/atfrank/MLFF_Distiller.git
cd MLFF_Distiller

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Finding Work

1. Visit the [GitHub Projects board](https://github.com/atfrank/MLFF_Distiller/projects)
2. Look for issues in the "Backlog" or "To Do" columns
3. Filter by your agent role using labels (e.g., `agent:data-pipeline`)
4. Pick an issue that matches your expertise and availability
5. Comment on the issue to claim it
6. Move it to "In Progress" when you start working

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates
- `test/` - Test additions or fixes

### 2. Make Your Changes

Follow these coding standards:

#### Code Style
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Follow PEP 8 guidelines
- Add type hints to function signatures
- Write docstrings for public APIs

```python
def process_atomic_structure(
    positions: np.ndarray,
    species: np.ndarray,
    cell: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Process atomic structure data for model input.

    Args:
        positions: Atomic positions of shape (n_atoms, 3)
        species: Atomic numbers of shape (n_atoms,)
        cell: Unit cell vectors of shape (3, 3), optional

    Returns:
        Dictionary containing processed structure data
    """
    # Implementation
    pass
```

#### Testing Requirements
- Write tests for all new functionality
- Maintain >80% code coverage
- Include both unit and integration tests when appropriate
- Use descriptive test names

```python
def test_data_loader_handles_periodic_boundaries():
    """Test that DataLoader correctly handles periodic boundary conditions."""
    # Test implementation
    pass
```

### 3. Run Tests and Checks

Before committing, ensure all checks pass:

```bash
# Format code
black src tests
isort src tests

# Run linter
ruff check src tests

# Type checking
mypy src

# Run tests
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/unit/test_data_loader.py -v
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "[Data Pipeline] Add support for periodic boundary conditions

- Implement periodic wrapping in DataLoader
- Add tests for PBC handling
- Update documentation with PBC examples

Closes #42"
```

Commit message format:
- First line: `[Agent Role] Brief description (50 chars max)`
- Body: Detailed explanation of changes
- Footer: Reference related issues

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub with:
- Clear title: `[Agent] [Milestone] Description`
- Description of changes
- Link to related issues using `Closes #issue-number`
- Screenshots or benchmarks if applicable

#### PR Template

```markdown
## Description
Brief description of changes

## Related Issues
Closes #XX

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Benchmarks run (if applicable)
```

## Code Review Process

1. **Automated Checks**: CI/CD must pass (tests, linting, formatting)
2. **Peer Review**: At least one approval from a team member
3. **Coordinator Review**: Lead coordinator reviews architectural decisions
4. **Merge**: Squash and merge into main branch

### Review Guidelines

When reviewing PRs:
- Check for code quality and style compliance
- Verify tests are comprehensive
- Ensure documentation is updated
- Test functionality locally when appropriate
- Provide constructive feedback
- Approve only when acceptance criteria are met

## Working with Issues

### Issue Labels

- `milestone:M1` through `milestone:M6` - Project milestones
- `agent:data-pipeline`, `agent:architecture`, etc. - Agent assignments
- `priority:high`, `priority:medium`, `priority:low` - Priority levels
- `status:blocked` - Blocked on dependencies or decisions
- `type:bug`, `type:feature`, `type:refactor` - Issue types

### Creating Issues

When creating new issues:

1. Use the appropriate template (Bug Report, Feature Request, Agent Task)
2. Provide clear, actionable description
3. Define acceptance criteria
4. Add relevant labels
5. Link related issues
6. Estimate complexity if possible

### Handling Blockers

If you're blocked:

1. Add the `status:blocked` label
2. Comment explaining the blocker
3. Tag the lead coordinator: `@coordinator`
4. Suggest potential solutions if you have ideas
5. Look for parallel work while waiting

## Architecture Decisions

For major architectural changes:

1. Create an RFC (Request for Comments) issue
2. Tag it with `needs-decision`
3. Outline the problem and proposed solutions
4. List pros/cons for each approach
5. Wait for coordinator approval before implementing

## Performance Considerations

When working on performance-critical code:

1. **Benchmark Before**: Establish baseline performance
2. **Profile**: Use profiling tools to identify bottlenecks
3. **Optimize**: Make targeted improvements
4. **Benchmark After**: Measure improvement
5. **Document**: Record performance characteristics

Example benchmark:

```python
# benchmarks/test_inference_speed.py
def benchmark_model_inference(model, batch_size=32, n_iterations=100):
    """Benchmark model inference speed."""
    # Implementation
    pass
```

## Documentation

Update documentation when:
- Adding new features
- Changing APIs
- Modifying behavior
- Fixing bugs with user-facing impact

Documentation locations:
- `docs/` - Detailed documentation
- Docstrings - Inline code documentation
- `README.md` - Quick start and overview
- `examples/` - Usage examples

## Communication

### Issue Comments
- Be clear and concise
- Provide context and rationale
- Tag relevant people with @mentions
- Update status when you start/finish work

### PR Comments
- Respond to all review feedback
- Explain your reasoning for decisions
- Be open to suggestions
- Mark conversations as resolved when addressed

### Project Board
- Keep your issues up to date
- Move cards to appropriate columns
- Update progress in issue comments

## Release Process

(To be defined by coordinator - typically):

1. Version bump in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish packages
5. Update documentation

## Questions?

- Check existing issues and PRs
- Review documentation in `docs/`
- Ask in issue comments
- Tag the coordinator for architectural questions

## Code of Conduct

- Be respectful and professional
- Provide constructive feedback
- Collaborate openly
- Focus on the work, not the person
- Celebrate successes together

---

Thank you for contributing to MLFF Distiller! Your work helps advance molecular dynamics simulation capabilities.
