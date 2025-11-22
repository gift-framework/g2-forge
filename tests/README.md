# G2-Forge Test Suite

Comprehensive test suite for g2-forge, covering differential geometry operators, loss functions, neural networks, and configuration system.

## 📊 Test Coverage

**Phase 1** (Critical Tests) - ✅ **COMPLETE**:
- ✅ **Operators** (27 tests): Levi-Civita, Hodge star, exterior derivative, metric reconstruction
- ✅ **Losses** (23 tests): Torsion losses, Gram matrix universality, adaptive scheduling
- ✅ **Networks** (25 tests): PhiNetwork, HarmonicNetwork antisymmetry, universality
- ✅ **Configuration** (25 tests): TCS topology validation, serialization, universality

**Phase 2** (High Priority) - ✅ **COMPLETE**:
- ✅ **K7 Manifold** (27 tests): TCS construction, coordinate sampling, region weights
- ✅ **Trainer** (18 tests): Initialization, optimizer, scheduler, curriculum phases
- ✅ **Integration** (25 tests): Training pipeline, gradient flow, universality end-to-end

**Phase 3** (Integration & Regression) - ✅ **COMPLETE**:
- ✅ **Checkpointing** (18 tests): Save/load, state restoration, resume training
- ✅ **GIFT v1.0** (22 tests): Configuration, reproduction, expected behavior
- ✅ **Numerical Precision** (13 tests): Operator precision, gradient stability
- ✅ **Deterministic** (18 tests): Reproducibility with fixed seeds

**Total**: ~210 tests implemented | **Coverage**: ~75%

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
│   ├── test_operators.py       # 27 tests - Core differential operators
│   ├── test_losses.py          # 23 tests - Loss functions and universality
│   ├── test_networks.py        # 25 tests - Neural network antisymmetry
│   ├── test_config.py          # 25 tests - Configuration and TCS validation
│   ├── test_k7_manifold.py     # 27 tests - K7 TCS, sampling, regions (NEW!)
│   └── test_trainer.py         # 18 tests - Trainer components (NEW!)
│
├── integration/
│   ├── test_training_pipeline.py   # 25 tests - Training workflows
│   ├── test_checkpointing.py       # 18 tests - Checkpoint save/load (NEW!)
│   └── test_gift_reproduction.py   # 22 tests - GIFT v1.0 (NEW!)
│
├── regression/
│   ├── test_numerical_precision.py # 13 tests - Numerical stability (NEW!)
│   └── test_deterministic.py       # 18 tests - Reproducibility (NEW!)
│
└── fixtures/
    └── conftest.py                 # Shared pytest fixtures
```

## 🔬 Test Categories

### Unit Tests (145 tests)

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

#### K7 Manifold (`test_k7_manifold.py` - 27 tests) - **NEW!**
- **Manifold Creation**: GIFT, custom, dimension, topology consistency
- **Coordinate Sampling**: Shape, device, ranges, distribution
- **Region Weights** (CRITICAL): Sum to 1, M1/Neck/M2 behavior, smoothness
- **Cycles**: Associative, coassociative, index specifications
- **Integration**: Full workflow, universality

#### Trainer (`test_trainer.py` - 18 tests) - **NEW!**
- **Initialization**: Networks, manifold, optimizer, scheduler, loss function
- **Optimizer**: AdamW type, learning rate, weight decay, parameters
- **Scheduler**: Warmup, cosine annealing
- **Curriculum**: Phase retrieval, transitions, loss weights
- **Universality**: Different topologies work

### Integration Tests (65 tests)

#### Training Pipeline (`test_training_pipeline.py` - 25 tests)
- **Basic Training**: Single epoch, multiple epochs, metrics, finite loss
- **Curriculum Learning**: Phase transitions, loss weight changes
- **Gradient Flow** (CRITICAL): Through all networks, no NaN/Inf
- **Universality Integration** (CRITICAL): Training works for ANY topology
- **Convergence**: 100-epoch tests, torsion reduction
- **Reproducibility**: Deterministic with seed
- **Metrics History**: Logging, epochs, components
- **Error Handling**: Zero epochs

#### Checkpointing (`test_checkpointing.py` - 18 tests) - **NEW!**
- **Save/Load**: File creation, state restoration, all networks included
- **Optimizer & Scheduler**: State preservation across save/load
- **Resume Training**: Continue from checkpoint, state integrity
- **Best Checkpoint**: Track lowest loss
- **Compatibility**: Topology mismatch handling
- **Roundtrip**: Complete save/load/resume cycle

#### GIFT v1.0 Reproduction (`test_gift_reproduction.py` - 22 tests) - **NEW!**
- **Configuration**: Correct topology (21,77), TCS parameters, 5 phases
- **Training**: Initialization, short runs, metrics structure
- **Expected Behavior**: Network sizes, loss function, region weights
- **Parameters**: Optimizer (AdamW, lr=1e-4), training (batch=2048)
- **Outcomes**: Torsion range, convergence behavior

### Regression Tests (31 tests) - **NEW!**

#### Numerical Precision (`test_numerical_precision.py` - 13 tests)
- **Operator Precision**: Hodge star, exterior derivative, metric reconstruction
- **Loss Precision**: Gram matrix, torsion losses
- **Network Precision**: PhiNetwork, HarmonicNetworks
- **Gradient Precision**: Through networks and operators
- **Stability**: CPU precision, loss accumulation
- **Boundary Values**: Near zero, large values

#### Deterministic Behavior (`test_deterministic.py` - 18 tests)
- **Network Determinism**: PhiNetwork, HarmonicNetworks with fixed seed
- **Manifold Determinism**: Sampling, region weights
- **Training Determinism**: Single step, multiple epochs, config seed
- **Operator Determinism**: Hodge star, exterior derivative
- **Initialization**: Network and optimizer parameters
- **Sanity Check**: Different seeds → different results

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

| Phase | Target | Tests | Status |
|-------|--------|-------|--------|
| Phase 1 (Critical) | 40% | 100 | ✅ **COMPLETE** |
| Phase 2 (High Priority) | 60% | 60 | ✅ **COMPLETE** |
| Phase 3 (Integration) | 75% | ~30 | 🔄 Planned |
| Phase 4 (Comprehensive) | 85%+ | ~20 | 🔄 Planned |

**Current Total**: ~160 tests | **Coverage**: ~60%

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

**Phase 3** (Integration & Regression):
- [ ] Full training pipeline (8 tests)
- [ ] Checkpointing integrity (5 tests)
- [ ] GIFT v1.0 reproduction validation (3 tests)

**Phase 4** (Polish & Performance):
- [ ] Numerical precision tests (6 tests)
- [ ] Performance benchmarks (3 tests)
- [ ] Edge case handling (5 tests)

## 📚 References

- [TEST_COVERAGE_ANALYSIS.md](../TEST_COVERAGE_ANALYSIS.md) - Full analysis and roadmap
- [PHASE2_SUMMARY.md](./PHASE2_SUMMARY.md) - Phase 2 detailed summary

---

**Status**: Phase 1-2 Complete ✅ (~160 tests, 60% coverage)
