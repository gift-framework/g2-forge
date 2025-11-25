# Contributing to g2-forge

Thank you for your interest in contributing to g2-forge! This document provides guidelines and instructions for contributing.

## 🎯 Project Goals

g2-forge aims to be the **universal framework** for neural construction of G₂ holonomy metrics. Key principles:

1. **Universality**: Work for any (b₂, b₃) topology, not just specific parameters
2. **Auto-sizing**: Networks automatically adapt to manifold topology
3. **Parameterized losses**: Loss functions scale with Betti numbers
4. **Production-ready**: Complete training infrastructure with checkpointing, validation, metrics

## 🚀 Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/gift-framework/g2-forge.git
cd g2-forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,notebook]"

# Run tests to verify setup
pytest tests/unit/ -v
```

### Project Structure

```
g2forge/
├── core/           # Differential geometry operators, losses
├── manifolds/      # G₂ manifold implementations
├── networks/       # Neural networks (auto-sizing!)
├── training/       # Training infrastructure
├── validation/     # Geometric validators
├── physics/        # Volume normalizer, RG flow
├── analysis/       # Spectral analysis
└── utils/          # Configuration system
```

## 📝 How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python and PyTorch versions
   - Configuration used (topology, training params)
   - Full error traceback
   - Steps to reproduce

### Suggesting Features

1. Check the [ROADMAP.md](ROADMAP.md) for planned features
2. Open a feature request issue
3. Describe the use case and expected behavior

### Pull Requests

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/my-feature`
3. Make your changes following our coding standards
4. **Write tests** for new functionality
5. Run the test suite: `pytest tests/ -v`
6. **Commit** with clear messages
7. **Push** and open a PR

## 🎨 Coding Standards

### Style Guide

- **Formatter**: Black (line length 100)
- **Imports**: isort (black profile)
- **Linter**: flake8
- **Type hints**: Required for all public functions

```bash
# Format code
black g2forge/ tests/ --line-length 100
isort g2forge/ tests/

# Lint
flake8 g2forge/ --max-line-length=100
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `PhiNetwork`, `K7Manifold`)
- **Functions/Methods**: `snake_case` (e.g., `compute_exterior_derivative`)
- **Constants**: `UPPER_CASE`
- **Private**: `_leading_underscore`

### Documentation

Every public function needs:

```python
def compute_exterior_derivative(phi: Tensor, coords: Tensor) -> Tensor:
    """
    Compute exterior derivative dφ : Λ³ → Λ⁴.

    Mathematical description of what the function does.

    Args:
        phi: Tensor[batch, 7, 7, 7] - Antisymmetric 3-form
        coords: Tensor[batch, 7] - Manifold coordinates

    Returns:
        dphi: Tensor[batch, 7, 7, 7, 7] - Exterior derivative

    Raises:
        ValueError: If coords does not have gradients enabled

    Example:
        >>> coords = torch.randn(100, 7, requires_grad=True)
        >>> phi = phi_network.get_phi_tensor(coords)
        >>> dphi = compute_exterior_derivative(phi, coords)
    """
```

## ✅ Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# By category
pytest tests/unit/ -v
pytest tests/integration/ -v

# Skip slow tests
pytest tests/ -m "not slow"

# With coverage
pytest tests/ --cov=g2forge --cov-report=html
```

### Writing Tests

1. **Unit tests** in `tests/unit/test_<module>.py`
2. **Integration tests** in `tests/integration/`
3. Use **parametrized tests** for multiple topologies:

```python
import pytest

@pytest.mark.parametrize("b2,b3", [(5, 20), (10, 40), (21, 77)])
def test_harmonic_network_output_dimensions(b2, b3):
    """Test that HarmonicNetwork output matches topology."""
    topology = TopologyConfig(b2=b2, b3=b3)
    h2_net = HarmonicNetwork(topology, p=2, hidden_dims=[128])
    assert h2_net.n_forms == b2
```

## 🔬 Mathematical Conventions

### G₂ Geometry

- **φ** (phi): G₂ 3-form defining the G₂ structure
- **★φ** (star phi): Hodge dual of φ, a 4-form
- **dφ = 0, d★φ = 0**: Torsion-free conditions
- **b₂, b₃**: Betti numbers (dimensions of H², H³)

### Tensor Conventions

- Indices: `i, j, k, l, m, n, p` for 7D coordinates (0-6)
- Forms: Fully antisymmetric
- Metric: Positive definite, symmetric

## 🚨 Critical Guidelines

### DO NOT hardcode topology values!

❌ **Bad**:
```python
identity_h2 = torch.eye(21)  # Hardcoded!
```

✅ **Good**:
```python
identity_h2 = torch.eye(topology.b2)  # Parameterized!
```

### Gradient-friendly operations

- No in-place operations (`+=`, `*=`) on gradient tensors
- Use `torch.autograd.grad()` for derivatives
- Return new tensors, don't modify inputs

## 📚 Resources

- **CLAUDE.md**: Comprehensive development guide
- **ROADMAP.md**: Development roadmap
- **GIFT Repository**: https://github.com/gift-framework/GIFT (original implementation)

## 📫 Contact

- **Issues**: https://github.com/gift-framework/g2-forge/issues
- **Email**: brieuc@bdelaf.com

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to g2-forge! 🎉
