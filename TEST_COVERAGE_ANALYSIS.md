# Test Coverage Analysis for g2-forge

**Date**: 2025-11-22
**Analyzed Codebase**: ~3,716 lines of production code
**Current Test Coverage**: ~350 lines (3 test files)
**Estimated Coverage**: ~15-20% of functionality

---

## Executive Summary

The g2-forge codebase has **minimal test coverage** with only 3 basic test files:
1. `crash_test.py` - Smoke tests (7 basic checks)
2. `test_networks.py` - Network auto-sizing validation (4 tests)
3. `test_phase2.py` - Phase 2 validation (5 integration tests)

**Critical Gap**: Only ~15-20% of the codebase has any test coverage. Key missing areas include:
- Differential geometry operators (no unit tests)
- Loss functions (no validation of correctness)
- Configuration validation (minimal edge case testing)
- Training infrastructure (no isolated component tests)
- Manifold coordinate sampling (no distribution validation)

---

## Current Test Coverage

### ✅ What IS Tested

| Component | Test File | Coverage Level |
|-----------|-----------|----------------|
| Network auto-sizing | `test_networks.py` | **Good** (3 topologies tested) |
| Basic imports | `crash_test.py`, `test_phase2.py` | **Good** |
| GIFT config creation | `crash_test.py` | **Basic** |
| Custom config creation | `crash_test.py` | **Basic** |
| Manifold creation | `test_phase2.py` | **Basic** |
| Coordinate sampling | `test_phase2.py` | **Minimal** (only shape check) |
| Levi-Civita construction | `test_phase2.py` | **Minimal** (only size check) |
| Mini training (5 epochs) | `crash_test.py` | **Smoke test only** |

### ❌ What is NOT Tested

| Category | Component | Risk Level |
|----------|-----------|------------|
| **Core Operators** | `hodge_star_3()` correctness | **CRITICAL** |
| **Core Operators** | `compute_exterior_derivative()` accuracy | **CRITICAL** |
| **Core Operators** | `compute_coclosure()` accuracy | **CRITICAL** |
| **Core Operators** | `reconstruct_metric_from_phi()` validity | **HIGH** |
| **Core Operators** | `validate_antisymmetry()` edge cases | **MEDIUM** |
| **Loss Functions** | `torsion_closure_loss()` correctness | **CRITICAL** |
| **Loss Functions** | `gram_matrix_loss()` universality | **CRITICAL** |
| **Loss Functions** | `AdaptiveLossScheduler` behavior | **HIGH** |
| **Loss Functions** | Calibration losses | **MEDIUM** |
| **Networks** | `PhiNetwork` antisymmetry guarantee | **HIGH** |
| **Networks** | `HarmonicNetwork` orthogonality | **HIGH** |
| **Networks** | Fourier features stability | **MEDIUM** |
| **Manifolds** | K7 region weight consistency | **HIGH** |
| **Manifolds** | Cycle sampling validity | **MEDIUM** |
| **Manifolds** | TCS topology validation | **HIGH** |
| **Config** | Validation edge cases | **MEDIUM** |
| **Config** | Serialization round-trip | **MEDIUM** |
| **Training** | Curriculum phase transitions | **HIGH** |
| **Training** | Gradient flow | **CRITICAL** |
| **Training** | Checkpointing integrity | **HIGH** |

---

## Detailed Coverage Analysis by Module

### 1. Core Operators (`g2forge/core/operators.py` - 480 lines)

**Current Coverage**: ~5% (only Levi-Civita size check)

#### Missing Tests:

##### 1.1 Levi-Civita Tensor (`build_levi_civita_sparse_7d`)
- ✅ **Tested**: Output shape (5040 permutations)
- ❌ **Missing**:
  - Sign correctness for even/odd permutations
  - Completeness (all permutations present)
  - Antisymmetry property validation
  - Known permutation spot checks (e.g., [0,1,2,3,4,5,6] → +1)

##### 1.2 Hodge Star (`hodge_star_3`)
- ❌ **Missing** (NO TESTS):
  - Mathematical correctness (★★ω = ±ω test)
  - Volume form preservation
  - Metric dependence (different metrics → different results)
  - Numerical stability with ill-conditioned metrics
  - Batch processing consistency
  - Known analytical test cases

##### 1.3 Exterior Derivative (`compute_exterior_derivative`)
- ❌ **Missing** (NO TESTS):
  - d(dω) = 0 (exact sequence property)
  - Gradient computation accuracy
  - Antisymmetry preservation
  - Subsampling consistency
  - Known analytical derivatives

##### 1.4 Coclosure (`compute_coclosure`)
- ❌ **Missing** (NO TESTS):
  - Mathematical correctness (δδω = 0)
  - Relationship to exterior derivative (δ = ★d★)
  - Subsampling effects
  - Gradient flow validation

##### 1.5 Metric Reconstruction (`reconstruct_metric_from_phi`)
- ❌ **Missing** (NO TESTS):
  - Positive definiteness guarantee
  - Symmetry validation
  - Known G₂ 3-form → metric examples
  - Regularization effects

##### 1.6 Region Weighted Torsion
- ❌ **Missing** (NO TESTS):
  - Region weight normalization
  - M₁/Neck/M₂ decomposition correctness
  - Edge cases (single region, no regions)

**Proposed Tests**: 15-20 unit tests

---

### 2. Loss Functions (`g2forge/core/losses.py` - 563 lines)

**Current Coverage**: ~0% (NO DIRECT TESTS)

#### Missing Tests:

##### 2.1 Torsion Losses
- ❌ **Missing**:
  - Zero loss for closed forms
  - Non-zero loss for non-closed forms
  - Gradient behavior near zero
  - Batch consistency

##### 2.2 Gram Matrix Loss
- ❌ **Missing** (CRITICAL FOR UNIVERSALITY):
  - Orthonormality enforcement for various b₂, b₃
  - Determinant constraint
  - Rank computation accuracy
  - Eigenvalue tolerance edge cases
  - Small vs large topology (b₂=3 vs b₂=100)
  - Identity target convergence

##### 2.3 Adaptive Loss Scheduler
- ❌ **Missing**:
  - Plateau detection accuracy
  - Weight boost behavior
  - Safety cap enforcement (prevent runaway)
  - History tracking
  - Reset functionality

##### 2.4 Composite Loss
- ❌ **Missing**:
  - Component weight application
  - Curriculum phase integration
  - Adaptive weight interaction
  - Missing region weights handling
  - Missing cycles handling

**Proposed Tests**: 20-25 unit tests

---

### 3. Networks (`g2forge/networks/` - 703 lines)

**Current Coverage**: ~30% (auto-sizing tested, but not correctness)

#### Missing Tests:

##### 3.1 PhiNetwork
- ✅ **Tested**: Output shape (batch, 35)
- ✅ **Tested**: Tensor expansion (batch, 7, 7, 7)
- ❌ **Missing**:
  - Antisymmetry of output tensor (critical property!)
  - Fourier feature randomness stability
  - Weight initialization distribution
  - Parameter counting accuracy
  - Different activation functions
  - Forward pass gradient flow
  - Known input → expected output structure

##### 3.2 HarmonicNetwork
- ✅ **Tested**: Auto-sizing for different topologies
- ✅ **Tested**: Output shape (batch, n_forms, n_components)
- ❌ **Missing**:
  - Form antisymmetry (2-forms and 3-forms)
  - Gram-Schmidt orthogonalization correctness
  - Linear independence of forms
  - Gradient flow through multiple heads
  - Edge cases (n_forms=1, n_forms=100)
  - Parameter efficiency vs n_forms

##### 3.3 FourierFeatures
- ❌ **Missing** (NO ISOLATED TESTS):
  - Frequency matrix stability (non-trainable)
  - Output distribution properties
  - Coordinate mapping coverage
  - Batch invariance

**Proposed Tests**: 15-20 unit tests

---

### 4. Manifolds (`g2forge/manifolds/` - 778 lines)

**Current Coverage**: ~10% (basic creation and sampling)

#### Missing Tests:

##### 4.1 K7Manifold
- ✅ **Tested**: Creation with different topologies
- ✅ **Tested**: Basic coordinate sampling (shape only)
- ❌ **Missing**:
  - Coordinate distribution validation (uniform in t, periodic in θ)
  - Grid sampling completeness (n^7 points)
  - Random sampling uniformity
  - Hybrid sampling properties
  - Coordinate bounds validation (t ∈ [0,1], θ ∈ [0,2π])
  - Region weights sum to ~1
  - Region weights smoothness (sigmoid properties)
  - Neck region centering
  - M₁/M₂ asymptotic behavior

##### 4.2 Cycles
- ❌ **Missing** (NO TESTS):
  - Associative cycle validity
  - Coassociative cycle validity
  - Cycle sampling (sample_on_cycle)
  - Cycle volume computation
  - Parametrization correctness

##### 4.3 TCSManifold Base
- ❌ **Missing**:
  - Region indicator smoothness
  - Transition sharpness parameter effects
  - Topology consistency validation

**Proposed Tests**: 15-18 unit tests

---

### 5. Configuration (`g2forge/utils/config.py` - 613 lines)

**Current Coverage**: ~15% (basic creation tested)

#### Missing Tests:

##### 5.1 TopologyConfig
- ✅ **Tested**: Basic creation (b₂, b₃)
- ❌ **Missing**:
  - Poincaré duality properties (b₄=b₃, b₅=b₂)
  - Euler characteristic computation
  - Validation edge cases (negative Betti numbers)
  - b₁ ≠ 0 rejection

##### 5.2 TCSParameters
- ❌ **Missing**:
  - total_b2/total_b3 computation
  - Neck width validation (must be in (0,1))
  - Topology consistency checks
  - Edge cases (zero Betti numbers)

##### 5.3 ManifoldConfig
- ✅ **Tested**: GIFT and custom creation
- ❌ **Missing**:
  - TCS topology mismatch detection
  - Missing tcs_params for TCS construction
  - Dimension validation (must be 7)
  - Invalid construction types

##### 5.4 G2ForgeConfig
- ❌ **Missing**:
  - GIFT v1.0 reproduction validation (5 phases)
  - JSON serialization round-trip
  - YAML serialization round-trip
  - Config validation cascading
  - Curriculum phase ordering
  - Phase epoch range overlap detection

##### 5.5 NetworkArchitectureConfig
- ❌ **Missing**:
  - Output dimension calculation for various topologies
  - Architecture parameter validation

**Proposed Tests**: 20-25 unit tests

---

### 6. Training (`g2forge/training/trainer.py` - 388 lines)

**Current Coverage**: ~5% (mini 5-epoch smoke test only)

#### Missing Tests:

##### 6.1 Trainer Initialization
- ✅ **Tested**: Basic creation
- ❌ **Missing**:
  - Network creation from config
  - Optimizer initialization
  - Scheduler setup (warmup + cosine)
  - Loss function creation with correct topology
  - Device placement

##### 6.2 Training Loop
- ✅ **Tested**: 5-epoch smoke test (passes without error)
- ❌ **Missing** (CRITICAL):
  - Single train_step isolated test
  - Gradient computation validation
  - Gradient clipping behavior
  - Loss decrease over epochs
  - Metrics tracking accuracy
  - Curriculum phase transitions (5 phases)
  - Learning rate schedule (warmup → cosine)
  - Batch size effects

##### 6.3 Checkpointing
- ❌ **Missing** (NO TESTS):
  - Checkpoint save/load integrity
  - State restoration (networks, optimizer, scheduler)
  - Resume training from checkpoint
  - Metrics history preservation
  - Best checkpoint tracking

**Proposed Tests**: 12-15 integration tests

---

### 7. Integration & End-to-End

**Current Coverage**: ~10% (mini training only)

#### Missing Tests:

##### 7.1 Full Pipeline
- ✅ **Tested**: 5-epoch mini training (smoke test)
- ❌ **Missing**:
  - 100-epoch convergence test
  - Torsion reduction over time
  - Harmonic rank convergence
  - Multiple topology training (b₂∈{5,21,50}, b₃∈{20,77,150})
  - GPU vs CPU consistency

##### 7.2 Universality Validation
- ❌ **Missing** (KEY FEATURE NOT TESTED):
  - Same code, different topologies (3+ configs)
  - Network parameter count scaling with topology
  - Loss function adaptation to topology
  - Convergence for small (b₂=3) vs large (b₂=100)

##### 7.3 Reproduction Tests
- ❌ **Missing**:
  - GIFT v1.0 config → expected metrics
  - Deterministic training (fixed seed)
  - Numerical precision regression tests

**Proposed Tests**: 8-10 integration tests

---

## Proposed Test Structure

```
tests/
├── unit/
│   ├── test_operators.py           # 15-20 tests
│   ├── test_losses.py              # 20-25 tests
│   ├── test_phi_network.py         # 8-10 tests
│   ├── test_harmonic_network.py    # 8-10 tests
│   ├── test_fourier_features.py    # 5-6 tests
│   ├── test_k7_manifold.py         # 12-15 tests
│   ├── test_cycles.py              # 5-6 tests
│   ├── test_topology_config.py     # 8-10 tests
│   ├── test_tcs_config.py          # 6-8 tests
│   ├── test_g2forge_config.py      # 10-12 tests
│   └── test_trainer_components.py  # 8-10 tests
│
├── integration/
│   ├── test_training_pipeline.py   # 8-10 tests
│   ├── test_curriculum_learning.py # 5-6 tests
│   ├── test_checkpointing.py       # 5-6 tests
│   ├── test_universality.py        # 6-8 tests (KEY!)
│   └── test_gift_reproduction.py   # 3-5 tests
│
├── regression/
│   ├── test_numerical_precision.py # 5-8 tests
│   └── test_deterministic.py       # 3-5 tests
│
└── fixtures/
    ├── conftest.py                 # Pytest fixtures
    ├── sample_configs.py           # Test configurations
    ├── analytical_solutions.py     # Known mathematical solutions
    └── reference_data.py           # Numerical reference values
```

**Total Proposed Tests**: ~180-220 tests

---

## Priority Test Areas

### 🔴 Critical Priority (Must Implement First)

1. **Hodge Star Correctness** (`test_operators.py`)
   - Test mathematical property: ★★ω = (-1)^{p(7-p)} ω
   - Known analytical cases
   - **Risk**: Core G₂ geometry depends on this

2. **Gram Matrix Loss Universality** (`test_losses.py`)
   - Test with b₂ ∈ {5, 21, 50, 100}
   - Verify orthonormality enforcement
   - **Risk**: Key universal feature not validated

3. **Network Antisymmetry** (`test_phi_network.py`, `test_harmonic_network.py`)
   - PhiNetwork: φ_{ijk} = -φ_{jik}
   - HarmonicNetwork: ω_{ij} = -ω_{ji}, α_{ijk} = -α_{jik}
   - **Risk**: Violates differential geometry requirements

4. **Training Gradient Flow** (`test_training_pipeline.py`)
   - Verify gradients propagate through all operators
   - Check for gradient explosion/vanishing
   - **Risk**: Training may fail silently

5. **TCS Topology Consistency** (`test_tcs_config.py`)
   - Verify b₂ = b₂_m1 + b₂_m2
   - Verify b₃ = b₃_m1 + b₃_m2
   - **Risk**: Incorrect manifold construction

### 🟡 High Priority (Implement Soon)

6. **Exterior Derivative** (`test_operators.py`)
   - Test d(dω) = 0 property
   - Known analytical examples

7. **Adaptive Loss Scheduler** (`test_losses.py`)
   - Plateau detection
   - Safety caps (prevent runaway)

8. **Region Weight Consistency** (`test_k7_manifold.py`)
   - Verify M₁ + Neck + M₂ ≈ 1
   - Smoothness properties

9. **Curriculum Phase Transitions** (`test_curriculum_learning.py`)
   - Verify loss weight changes at boundaries
   - Phase ordering correctness

10. **Config Serialization** (`test_g2forge_config.py`)
    - JSON round-trip (save → load → identical)
    - YAML round-trip

### 🟢 Medium Priority (Nice to Have)

11. **Calibration Losses** (`test_losses.py`)
12. **Cycle Sampling** (`test_cycles.py`)
13. **Fourier Features** (`test_fourier_features.py`)
14. **Checkpointing** (`test_checkpointing.py`)
15. **Numerical Regression** (`test_numerical_precision.py`)

---

## Test Implementation Guidelines

### 1. Unit Test Structure

```python
import pytest
import torch
from g2forge.core import hodge_star_3, build_levi_civita_sparse_7d

def test_hodge_star_double_application():
    """Test ★★ω = (-1)^{p(7-p)} ω for p=3."""
    # Setup
    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
    eps_indices, eps_signs = build_levi_civita_sparse_7d()

    # Apply Hodge star twice
    star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)
    star_star_phi = hodge_star_4to3(star_phi, metric, eps_indices, eps_signs)

    # For p=3 in 7D: ★★ω = (-1)^{3(7-3)} ω = (-1)^12 ω = ω
    expected_sign = (-1) ** (3 * (7 - 3))  # = +1

    # Check
    torch.testing.assert_close(
        star_star_phi,
        expected_sign * phi,
        rtol=1e-5,
        atol=1e-6
    )
```

### 2. Integration Test Structure

```python
def test_training_convergence_small_topology():
    """Test that training converges for small topology (b₂=5, b₃=20)."""
    # Setup
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = g2.training.Trainer(config, device='cpu', verbose=False)

    # Train for 100 epochs
    results = trainer.train(num_epochs=100)

    # Check convergence
    assert results['final_metrics']['loss'] < results['initial_metrics']['loss']
    assert results['final_metrics']['torsion_closure'] < 1e-2
    assert results['final_metrics']['rank_h2'] >= 3  # At least partial rank
    assert results['final_metrics']['rank_h3'] >= 10
```

### 3. Fixtures (conftest.py)

```python
import pytest
import torch
import g2forge as g2

@pytest.fixture
def small_topology_config():
    """Small topology for fast testing."""
    return g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

@pytest.fixture
def gift_config():
    """GIFT v1.0 configuration."""
    return g2.G2ForgeConfig.from_gift_v1_0()

@pytest.fixture
def levi_civita():
    """Cached Levi-Civita tensor."""
    return g2.build_levi_civita_sparse_7d()

@pytest.fixture
def sample_phi():
    """Sample 3-form for testing."""
    phi = torch.randn(10, 7, 7, 7)
    # Antisymmetrize
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if i < j < k:
                    phi[:, j, i, k] = -phi[:, i, j, k]
                    phi[:, k, j, i] = -phi[:, i, j, k]
                    # ... (all permutations)
    return phi
```

---

## Code Coverage Tools

### Recommended Setup

```bash
pip install pytest pytest-cov pytest-xdist

# Run tests with coverage
pytest tests/ --cov=g2forge --cov-report=html --cov-report=term

# Parallel execution (faster)
pytest tests/ -n auto --cov=g2forge

# Coverage target
# - Phase 1: Aim for 60% coverage (core operators, losses, networks)
# - Phase 2: Aim for 75% coverage (add manifolds, config)
# - Phase 3: Aim for 85% coverage (training, integration)
```

---

## Specific Test Examples

### Example 1: Levi-Civita Correctness

```python
def test_levi_civita_sign_correctness():
    """Test that Levi-Civita signs are correct for known permutations."""
    indices, signs = build_levi_civita_sparse_7d()

    # Identity permutation [0,1,2,3,4,5,6] → +1
    identity_idx = ((indices == torch.arange(7)).all(dim=1)).nonzero()[0]
    assert signs[identity_idx] == 1.0

    # Single swap [1,0,2,3,4,5,6] → -1
    swap_perm = torch.tensor([1, 0, 2, 3, 4, 5, 6])
    swap_idx = ((indices == swap_perm).all(dim=1)).nonzero()[0]
    assert signs[swap_idx] == -1.0

    # Double swap [1,0,3,2,4,5,6] → +1
    double_swap = torch.tensor([1, 0, 3, 2, 4, 5, 6])
    double_idx = ((indices == double_swap).all(dim=1)).nonzero()[0]
    assert signs[double_idx] == 1.0
```

### Example 2: Gram Matrix Universality

```python
@pytest.mark.parametrize("b2,b3", [
    (5, 20),
    (21, 77),  # GIFT
    (50, 150),
])
def test_gram_matrix_loss_universality(b2, b3):
    """Test gram matrix loss works for different topologies."""
    topology = TopologyConfig(b2=b2, b3=b3)

    # Generate orthonormal forms
    batch_size = 100
    n_components = 21  # For 2-forms in 7D
    harmonic_forms = torch.randn(batch_size, b2, n_components)

    # Orthonormalize (Gram-Schmidt)
    # ... (implementation)

    # Compute loss
    loss, det, rank = gram_matrix_loss(harmonic_forms, target_rank=b2)

    # Should be low loss for orthonormal forms
    assert loss < 1e-2
    assert abs(det - 1.0) < 0.1
    assert rank == b2
```

### Example 3: Network Antisymmetry

```python
def test_phi_network_antisymmetry():
    """Test that PhiNetwork output is antisymmetric."""
    config = g2.G2ForgeConfig.from_gift_v1_0()
    phi_net = g2.networks.create_phi_network_from_config(config)

    coords = torch.randn(10, 7)
    phi_tensor = phi_net.get_phi_tensor(coords)  # [10, 7, 7, 7]

    # Check antisymmetry: φ_{ijk} = -φ_{jik}
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if i != j and i != k and j != k:
                    torch.testing.assert_close(
                        phi_tensor[:, i, j, k],
                        -phi_tensor[:, j, i, k],
                        rtol=1e-5, atol=1e-6
                    )
```

---

## Summary & Recommendations

### Current State
- **Test Coverage**: ~15-20%
- **Test Count**: ~16 tests (mostly smoke/integration)
- **Missing**: Core operator correctness, loss function validation

### Target State
- **Test Coverage**: 75-85%
- **Test Count**: 180-220 tests (unit + integration + regression)
- **Structure**: Organized test suite with fixtures

### Implementation Roadmap

**Phase 1 (Week 1-2)**: Critical Tests
- [ ] Hodge star correctness (5 tests)
- [ ] Exterior derivative (5 tests)
- [ ] Gram matrix universality (6 tests)
- [ ] Network antisymmetry (4 tests)
- [ ] TCS topology validation (4 tests)
- **Target**: 25 tests, 40% coverage

**Phase 2 (Week 3-4)**: High Priority
- [ ] Adaptive scheduler (5 tests)
- [ ] Region weights (6 tests)
- [ ] Config validation (10 tests)
- [ ] Curriculum phases (5 tests)
- [ ] Training gradient flow (4 tests)
- **Target**: 55 tests total, 60% coverage

**Phase 3 (Week 5-6)**: Integration & Regression
- [ ] Full pipeline tests (8 tests)
- [ ] Universality validation (6 tests)
- [ ] Checkpointing (5 tests)
- [ ] Numerical regression (6 tests)
- [ ] GIFT reproduction (3 tests)
- **Target**: 83 tests total, 75% coverage

**Phase 4 (Ongoing)**: Maintain & Extend
- [ ] Additional edge cases
- [ ] Performance benchmarks
- [ ] Coverage: 85%+

### Key Benefits of Improved Testing

1. **Confidence in Universality**: Verify the framework works for ANY topology
2. **Correctness Validation**: Ensure differential geometry operators are mathematically correct
3. **Regression Prevention**: Catch bugs before they reach production
4. **Documentation**: Tests serve as executable documentation
5. **Refactoring Safety**: Enable confident code improvements
6. **Reproducibility**: Validate GIFT v1.0 reproduction claims

---

**Bottom Line**: The codebase is well-structured and implements sophisticated mathematics, but has minimal test coverage. Implementing comprehensive tests is **critical** for validating the universal topology support and ensuring correctness of core geometric operators.
