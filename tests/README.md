# G2-Forge Test Suite

Comprehensive test suite for g2-forge, covering differential geometry operators, loss functions, neural networks, and configuration system.

## 📊 Test Coverage

**Phase 1 Implementation** (Critical Tests):
- ✅ **Operators** (27 tests): Levi-Civita, Hodge star, exterior derivative, metric reconstruction
- ✅ **Losses** (23 tests): Torsion losses, Gram matrix universality, adaptive scheduling
- ✅ **Networks** (25 tests): PhiNetwork, HarmonicNetwork antisymmetry, universality
- ✅ **Configuration** (25 tests): TCS topology validation, serialization, universality

**Total**: ~100 tests implemented

## 🚀 Quick Start

### Installation

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or install manually
pip install pytest pytest-cov torch numpy scipy
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_operators.py

# Run with coverage report
pytest tests/ --cov=g2forge --cov-report=html

# Run tests in parallel (faster)
pytest tests/ -n auto
```

### Coverage Report

```bash
# Generate HTML coverage report
pytest tests/ --cov=g2forge --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 📁 Test Structure

```
tests/
├── unit/
│   ├── test_operators.py      # 27 tests - Core differential operators
│   ├── test_losses.py          # 23 tests - Loss functions and universality
│   ├── test_networks.py        # 25 tests - Neural network antisymmetry
│   └── test_config.py          # 25 tests - Configuration and TCS validation
│
├── integration/               # (Future) Integration tests
├── regression/                # (Future) Regression tests
│
└── fixtures/
    └── conftest.py            # Shared pytest fixtures
```

## 🔬 Test Categories

### Unit Tests (100 tests)

#### Operators (`test_operators.py` - 27 tests)
- **Levi-Civita Tensor**: Shape, signs, permutations
- **Hodge Star**: Correctness, metric dependence, batch processing
- **Exterior Derivative**: d(constant)=0, gradient accuracy
- **Metric Reconstruction**: Symmetry, positive definiteness
- **Volume Form**: Identity metric, scaling
- **Antisymmetry Validation**: 2-forms and 3-forms

#### Losses (`test_losses.py` - 23 tests)
- **Torsion Losses**: Closure, coclosure
- **Volume Loss**: Target matching, custom targets
- **Gram Matrix Loss** (CRITICAL): Universality across b₂∈{3,5,10,21,50}, b₃∈{10,20,40,77,150}
- **Adaptive Scheduler**: Plateau detection, safety caps
- **Composite Loss**: Multi-topology support

#### Networks (`test_networks.py` - 25 tests)
- **Fourier Features**: Initialization, determinism, batch invariance
- **PhiNetwork**: Antisymmetry (CRITICAL!), parameter count
- **HarmonicNetwork**: Auto-sizing universality, antisymmetry for H² and H³
- **Universality**: Same code works for b₂∈{3,5,10,21,50}, b₃∈{10,20,40,77,150}

#### Configuration (`test_config.py` - 25 tests)
- **TopologyConfig**: Poincaré duality, Euler characteristic
- **TCSParameters**: Total topology = M₁ + M₂ (CRITICAL!)
- **ManifoldConfig**: TCS consistency validation
- **G2ForgeConfig**: GIFT v1.0 reproduction, serialization
- **Universality**: Config system works for ANY topology

## ✅ Key Tests

### Critical Antisymmetry Tests
```python
# PhiNetwork must produce antisymmetric 3-forms
test_phi_network_antisymmetry()  # φ_{ijk} = -φ_{jik}

# HarmonicNetwork must produce antisymmetric forms
test_harmonic_network_h2_antisymmetry()  # ω_{ij} = -ω_{ji}
test_harmonic_network_h3_antisymmetry()  # α_{ijk} = -α_{jik}
```

### Critical Universality Tests
```python
# Gram matrix loss works for ANY b₂, b₃
@pytest.mark.parametrize("b2", [3, 5, 10, 21, 50])
test_gram_matrix_loss_small_topologies(b2)

# Networks auto-size for ANY topology
@pytest.mark.parametrize("b2", [3, 5, 10, 21, 50])
test_harmonic_network_h2_universality(b2)

# TCS topology consistency: b₂ = b₂_m1 + b₂_m2
test_tcs_parameters_total_topology()
test_manifold_config_tcs_consistency()
```

## 📈 Coverage Goals

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1 (Critical) | 40% | ✅ 100 tests implemented |
| Phase 2 (High Priority) | 60% | 🔄 Planned |
| Phase 3 (Integration) | 75% | 🔄 Planned |
| Phase 4 (Comprehensive) | 85%+ | 🔄 Planned |

## 🐛 Debugging Tests

### Run specific test with verbose output
```bash
pytest tests/unit/test_operators.py::test_hodge_star_shape -vv
```

### Run tests with debugger
```bash
pytest tests/unit/test_losses.py -k "gram_matrix" --pdb
```

### Run tests matching pattern
```bash
pytest tests/ -k "antisymmetry"
pytest tests/ -k "universality"
```

## 📝 Test Fixtures

Common fixtures available (from `conftest.py`):
- `small_topology_config`: b₂=5, b₃=20
- `gift_config`: GIFT v1.0 (b₂=21, b₃=77)
- `large_topology_config`: b₂=30, b₃=100
- `levi_civita`: Cached Levi-Civita tensor
- `sample_phi_antisymmetric`: Properly antisymmetrized 3-form
- `k7_manifold_small`, `k7_manifold_gift`: K7 manifolds

## 🎯 Next Steps

**Phase 2** (High Priority):
- [ ] K7 manifold region weight tests (6 tests)
- [ ] Cycle sampling validation (5 tests)
- [ ] Trainer component tests (8 tests)
- [ ] Curriculum learning integration (5 tests)

**Phase 3** (Integration):
- [ ] Full training pipeline (8 tests)
- [ ] Checkpointing integrity (5 tests)
- [ ] GIFT v1.0 reproduction validation (3 tests)

**Phase 4** (Regression):
- [ ] Numerical precision tests (6 tests)
- [ ] Deterministic training tests (3 tests)

## 📚 References

See [TEST_COVERAGE_ANALYSIS.md](../TEST_COVERAGE_ANALYSIS.md) for full analysis and roadmap.

---

**Status**: Phase 1 Complete ✅ (100 critical tests)
