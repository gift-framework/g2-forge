# CLAUDE.md - AI Assistant Guide for g2-forge

**Version**: 0.1.0-dev
**Last Updated**: 2025-11-24
**For**: Claude Code and other AI coding assistants

This document provides comprehensive guidance for AI assistants working on the g2-forge codebase - a universal framework for neural construction of G₂ holonomy metrics on compact 7-manifolds using Physics-Informed Neural Networks (PINNs).

---

## 🎯 Project Overview

### What is g2-forge?

**g2-forge** is the **first universal framework** for constructing G₂ holonomy metrics on **ANY** compact 7-manifold using neural networks. Unlike previous implementations hardcoded to specific manifolds (e.g., GIFT's b₂=21, b₃=77), g2-forge works for arbitrary topologies.

**Key Innovation**: Same code, any G₂ manifold topology.

### Scientific Context

- **G₂ manifolds**: 7-dimensional Riemannian manifolds with exceptional holonomy group G₂
- **Torsion-free condition**: dφ = 0 and d★φ = 0 (where φ is the G₂ 3-form)
- **Topology**: Characterized by Betti numbers b₂ (2-cohomology rank) and b₃ (3-cohomology rank)
- **TCS Construction**: Twisted Connected Sum of two building blocks M₁ and M₂

### Project Goals

1. **Universality**: Work for any (b₂, b₃) topology, not just specific parameter sets
2. **Auto-sizing**: Neural networks automatically adapt to manifold topology
3. **Parameterized losses**: Loss functions scale with Betti numbers
4. **Production-ready**: Complete training infrastructure with checkpointing, validation, metrics

### Technical Stack

- **Language**: Python 3.10+
- **ML Framework**: PyTorch 2.0+
- **Scientific Computing**: NumPy, SciPy
- **Testing**: pytest with comprehensive test suite (34 test files, 8000+ lines)
- **Code Volume**: ~20,000 lines total (5,300 core, 8,000+ tests)

---

## 📁 Repository Structure

```
g2-forge/
├── g2forge/                      # Main package (5,302 lines)
│   ├── __init__.py               # Top-level API exports
│   ├── core/                     # Differential geometry (1,128 lines)
│   │   ├── operators.py          # Hodge star, exterior derivative, metric reconstruction
│   │   └── losses.py             # All loss functions (parameterized by topology)
│   ├── manifolds/                # G₂ manifold implementations
│   │   ├── base.py               # Abstract Manifold, TCSManifold base classes
│   │   └── k7.py                 # K7 TCS construction (428 lines)
│   ├── networks/                 # Auto-sizing neural networks
│   │   ├── phi_network.py        # PhiNetwork for G₂ 3-form
│   │   └── harmonic_network.py   # HarmonicNetwork (auto-sizes from topology!)
│   ├── training/                 # Training infrastructure
│   │   └── trainer.py            # Trainer with curriculum learning
│   ├── validation/               # Geometric validation
│   │   └── geometric.py          # Ricci, holonomy, metric validators
│   ├── physics/                  # Physics modules (GIFT v1.2+)
│   │   ├── volume_normalizer.py  # Volume normalization
│   │   └── rg_flow.py            # RG flow computation
│   ├── analysis/                 # Analysis tools
│   │   └── spectral.py           # Spectral analysis, harmonic extraction
│   └── utils/                    # Configuration system
│       └── config.py             # Dataclass configs (764 lines!)
│
├── tests/                        # Comprehensive test suite (8,000+ lines)
│   ├── conftest.py               # Pytest fixtures and utilities
│   ├── unit/                     # 15 unit test files
│   ├── integration/              # 6 integration test files
│   ├── regression/               # 3 regression test files
│   ├── edge_cases/               # 2 edge case test files
│   └── performance/              # Performance benchmarks
│
├── examples/                     # Usage examples
│   ├── complete_example.py       # Full pipeline demonstration
│   └── configs/                  # Config examples
│
├── requirements*.txt             # Separated dependency files
├── setup.py                      # Package installation
├── pytest.ini                    # Pytest configuration
├── setup_tests.sh                # Test setup script
├── run_new_tests.sh              # Run physics tests
│
└── [Documentation files]
    ├── README.md                 # Main documentation
    ├── ANALYSIS.md               # Code analysis
    ├── ROADMAP.md                # Development plan
    ├── REQUIREMENTS_GUIDE.md     # Requirements breakdown
    └── [Other status/phase docs]
```

---

## 🏗️ Architecture Deep Dive

### 1. Configuration System (`g2forge/utils/config.py`)

**Design Pattern**: Hierarchical dataclass-based configuration with validation

**Key Classes** (764 lines total):

```python
@dataclass
class TopologyConfig:
    """Betti numbers defining manifold topology"""
    b2: int  # Rank of H²(M, ℝ)
    b3: int  # Rank of H³(M, ℝ)
    b1: int = 0  # Usually 0 (simply connected)
    b0: int = 1  # Always 1 (connected)

    @property
    def euler_characteristic(self) -> int:
        return 2 * (self.b0 - self.b1 + self.b2 - self.b3)

@dataclass
class ManifoldConfig:
    """Complete manifold specification"""
    topology: TopologyConfig
    tcs_params: TCSParameters
    moduli: ModuliParameters

@dataclass
class NetworkArchitectureConfig:
    """Neural network architecture (AUTO-SIZES from topology!)"""
    # Networks determine output dims from topology.b2, topology.b3
    phi_hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 512, 256])
    harmonic_hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 256])

@dataclass
class G2ForgeConfig:
    """Top-level configuration combining all subsystems"""
    manifold: ManifoldConfig
    architecture: NetworkArchitectureConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    validation: ValidationConfig
```

**Factory Functions**:
- `create_k7_config(b2_m1, b3_m1, b2_m2, b3_m2)` - Create custom K7 config
- `G2ForgeConfig.from_gift_v1_0()` - GIFT reproduction (b₂=21, b₃=77)
- `G2ForgeConfig.from_gift_v1_2b()` - GIFT v1.2b with volume normalizer

**Serialization**: All configs support `.to_dict()`, `.from_dict()`, `.to_json()`, `.from_json()`, `.to_yaml()`, `.from_yaml()`

**Validation**: Every config has `.validate()` method that checks:
- Topological consistency (Poincaré duality)
- Dimension must be 7 for G₂
- TCS parameter compatibility
- Raises descriptive `ValueError` on failure

### 2. Differential Geometry Operators (`g2forge/core/operators.py`)

**Key Insight**: These operators are **pure mathematics** - universal to ALL G₂ manifolds.

**Core Functions** (480 lines):

```python
def build_levi_civita_sparse_7d() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sparse Levi-Civita tensor (5040 entries = 7!)"""
    # Returns (indices, signs) for efficient computation

def hodge_star_3(phi: Tensor, metric: Tensor, eps_indices, eps_signs) -> Tensor:
    """
    Hodge star operator: ★φ : Λ³ → Λ⁴
    (★φ)_{ijkl} = (1/3!) Σ ε_{ijklmnp} g^{mm'} g^{nn'} g^{pp'} φ_{m'n'p'} / √det(g)
    """

def compute_exterior_derivative(phi: Tensor, coords: Tensor) -> Tensor:
    """
    Exterior derivative: dφ : Λ³ → Λ⁴
    (dφ)_{ijkl} = ∂_i φ_{jkl} - ∂_j φ_{ikl} + ∂_k φ_{ijl} - ∂_l φ_{ijk}
    Uses automatic differentiation for exactness.
    """

def compute_coclosure(omega: Tensor, coords: Tensor, metric: Tensor, ...) -> Tensor:
    """Codifferential δ = d★ : Λᵖ → Λᵖ⁻¹"""

def reconstruct_metric_from_phi(phi: Tensor) -> Tensor:
    """
    Reconstruct metric from G₂ 3-form:
    g_ij = (1/6) Σ_{pqr} φ_ipq φ_jpqr
    """

def validate_antisymmetry(form: Tensor, p: int) -> bool:
    """Validate p-form antisymmetry"""
```

**Design Principles**:
- All use automatic differentiation (torch.autograd)
- No in-place operations (gradient-friendly)
- Subsampling for efficiency on large batches
- Sparse representations where possible (Levi-Civita)

### 3. Loss Functions (`g2forge/core/losses.py`)

**CRITICAL**: All losses are **parameterized by `TopologyConfig`** - no hardcoded values!

**Core Loss Functions** (648 lines):

```python
def torsion_closure_loss(dphi: Tensor) -> Tensor:
    """dφ = 0 constraint (mean squared)"""

def torsion_coclosure_loss(d_star_phi: Tensor) -> Tensor:
    """d★φ = 0 constraint"""

def volume_loss(metric: Tensor) -> Tensor:
    """det(g) ≈ 1 constraint"""

def gram_matrix_loss(
    h2_forms: Tensor,
    h3_forms: Tensor,
    topology: TopologyConfig,  # ← PARAMETERIZED!
    metric: Tensor
) -> Tensor:
    """
    Harmonic form orthonormality: Gram matrix → Identity

    CRITICAL: Uses topology.b2 and topology.b3 for matrix dimensions!
    NOT hardcoded to (21, 77) like original GIFT.
    """

def boundary_smoothness_loss(phi: Tensor, coords: Tensor, manifold) -> Tensor:
    """Smoothness in TCS neck region"""

def calibration_associative_loss(phi: Tensor, coords: Tensor) -> Tensor:
    """G₂ calibration for associative 3-cycles"""

def calibration_coassociative_loss(phi: Tensor, coords: Tensor, ...) -> Tensor:
    """G₂ calibration for coassociative 4-cycles"""
```

**Composite Loss**:

```python
class CompositeLoss(nn.Module):
    def __init__(self, topology: TopologyConfig, manifold: Manifold, ...):
        """
        Combines all losses with adaptive weighting.
        Parameterized by topology for universality!
        """

    def forward(self, phi, h2_forms, h3_forms, coords, epoch=0) -> Dict[str, Tensor]:
        # Returns dict with individual losses and total
```

**Adaptive Loss Scheduler**:

```python
class AdaptiveLossScheduler:
    """
    Dynamically adjusts loss weights during training.
    Used in curriculum learning (5 phases).
    """
```

### 4. Manifolds (`g2forge/manifolds/`)

**Abstract Base** (`base.py`):

```python
@dataclass
class Cycle:
    """Geometric cycle (3- or 4-dimensional submanifold)"""
    dimension: int
    basis_generators: List[Callable]

class Manifold(ABC):
    """Abstract base for all G₂ manifolds"""

    @abstractmethod
    def sample_coordinates(self, n_samples, **kwargs) -> Tensor:
        """Sample points on manifold"""

    @abstractmethod
    def compute_metric(self, coords: Tensor) -> Tensor:
        """Compute background metric"""

class TCSManifold(Manifold):
    """Base for Twisted Connected Sum construction"""

    def __init__(self, topology: TopologyConfig, tcs_params: TCSParameters):
        self.topology = topology  # Stores (b₂, b₃)
        self.tcs_params = tcs_params
```

**K7 Implementation** (`k7.py` - 428 lines):

```python
class K7Manifold(TCSManifold):
    """
    K7 manifold via TCS construction.

    Structure: M₁ #_T³ M₂
    - M₁: Building block 1 (Fano scheme)
    - M₂: Building block 2 (Fano scheme)
    - T³: 3-torus "neck" region

    Topology: b₂ = b₂(M₁) + b₂(M₂), b₃ = b₃(M₁) + b₃(M₂) + 1
    """

    def sample_coordinates(self, n_samples, grid_n=None, device='cpu') -> Tensor:
        """Sample on grid or randomly"""

    def compute_associated_calibrated_metrics(self, coords: Tensor) -> Dict:
        """Compute ACyl metrics on building blocks"""

    def neck_transition_function(self, t: Tensor) -> Tensor:
        """Smooth interpolation in neck region"""
```

### 5. Neural Networks (`g2forge/networks/`)

**KEY FEATURE**: Networks **auto-size** from topology configuration!

**PhiNetwork** (`phi_network.py`):

```python
class FourierFeatures(nn.Module):
    """Random Fourier feature encoding for coordinate input"""

class PhiNetwork(nn.Module):
    """
    Neural network generating G₂ 3-form φ.

    Architecture:
    - Input: 7D coordinates → Fourier features
    - Hidden: MLP with configurable layers
    - Output: 35 components (unique entries of antisymmetric 3-form)
    - Post-processing: Antisymmetrize to get full φ[7,7,7]
    """

    def get_phi_tensor(self, coords: Tensor) -> Tensor:
        """Returns fully antisymmetrized φ[batch, 7, 7, 7]"""
```

**HarmonicNetwork** (`harmonic_network.py`):

```python
class HarmonicNetwork(nn.Module):
    """
    Extracts harmonic p-forms from coordinates.

    CRITICAL: n_forms parameter set from topology!
    - For H²: n_forms = topology.b2
    - For H³: n_forms = topology.b3

    Networks automatically adapt to manifold topology!
    """

    def __init__(self, topology: TopologyConfig, p: int, hidden_dims: List[int]):
        self.topology = topology
        self.p = p  # Form degree

        if p == 2:
            self.n_forms = topology.b2  # ← AUTO-SIZED!
            self.output_dim = 21  # Unique entries in antisym 2-form
        elif p == 3:
            self.n_forms = topology.b3  # ← AUTO-SIZED!
            self.output_dim = 35  # Unique entries in antisym 3-form

    def forward(self, coords: Tensor) -> Tensor:
        """Returns [batch, n_forms, 7^p] fully antisymmetrized forms"""
```

**Factory Functions**:

```python
def create_phi_network_from_config(config: G2ForgeConfig) -> PhiNetwork:
    """Create PhiNetwork from config"""

def create_harmonic_networks_from_config(config: G2ForgeConfig) -> Tuple[HarmonicNetwork, HarmonicNetwork]:
    """Create (H² network, H³ network) auto-sized from topology"""
```

### 6. Training Infrastructure (`g2forge/training/trainer.py`)

**Trainer Class** - Complete training loop with curriculum learning:

```python
class Trainer:
    """
    Main training class for G₂ metric construction.

    Features:
    - Curriculum learning (5 phases)
    - AdamW optimizer with cosine annealing
    - Adaptive loss scheduling
    - Checkpointing and resumption
    - Metrics tracking and validation
    - Works for ANY topology configuration
    """

    def __init__(self, config: G2ForgeConfig, device='cpu', verbose=True):
        self.config = config
        self.manifold = create_manifold(config.manifold)
        self.phi_network = create_phi_network_from_config(config)
        self.h2_net, self.h3_net = create_harmonic_networks_from_config(config)
        self.loss_fn = CompositeLoss(config.manifold.topology, self.manifold)
        # ... optimizer, scheduler setup

    def train(self, num_epochs: int) -> Dict:
        """
        Main training loop.

        Returns dict with:
        - final_metrics: torsion, rank_h2, rank_h3, etc.
        - training_history: loss curves
        - checkpoints: saved states
        """

    def save_checkpoint(self, path: str):
        """Save complete training state"""

    def load_checkpoint(self, path: str):
        """Resume from checkpoint"""
```

**Curriculum Learning** (5 phases in GIFT v1.0):

1. **Phase 1 (0-2k epochs)**: Neck stability - Focus on torsion-free in neck region
2. **Phase 2 (2k-5k)**: ACyl matching - Match to asymptotic cylindrical metrics
3. **Phase 3 (5k-8k)**: Cohomology refinement - Enforce harmonic orthogonality
4. **Phase 4 (8k-10k)**: Harmonic extraction - Improve H² and H³ extraction
5. **Phase 5 (10k-15k)**: Calibration finetune - Final polishing with calibration

Each phase has different:
- `grid_n`: Sampling density
- `loss_weights`: Relative importance of each loss term
- `region_weights`: TCS region importance (neck vs bulk)

### 7. Validation (`g2forge/validation/geometric.py`)

**Geometric Validators** (495 lines):

```python
class RicciValidator:
    """Verify Ricci-flatness (G₂ → Ricci = 0)"""

    def validate(self, metric: Tensor, coords: Tensor) -> Dict:
        """Returns Ricci tensor norm and max component"""

class HolonomyTester:
    """Test G₂ holonomy preservation under parallel transport"""

class MetricValidator:
    """Validate metric properties (symmetry, positive-definiteness)"""

class GeometricValidator:
    """Combined validator for full geometric consistency"""

    def validate_ricci_flatness(self, metric, coords) -> ValidationResult
    def validate_holonomy(self, phi, metric, coords) -> ValidationResult
    def validate_metric_properties(self, metric) -> ValidationResult
```

### 8. Physics Modules (`g2forge/physics/`)

**Volume Normalizer** (`volume_normalizer.py` - GIFT v1.2a/b):

```python
class VolumeNormalizer:
    """
    Adaptive volume normalization during training.
    Ensures vol(M) ≈ target by rescaling metric.
    """

class PhiNetwork:
    """
    Enhanced PhiNetwork with integrated volume normalization.
    Used in GIFT v1.2a/b.
    """
```

**RG Flow** (`rg_flow.py` - GIFT 2.1):

```python
class RGFlowModule:
    """
    Renormalization Group flow computation.
    Tracks scale-dependent behavior of G₂ structure.
    """

    def compute_beta_functions(self, phi, metric, coords) -> Dict:
        """RG β-functions for φ and g"""
```

### 9. Analysis Tools (`g2forge/analysis/spectral.py`)

**Spectral Analysis** (493 lines):

```python
def compute_laplacian_spectrum(metric: Tensor, coords: Tensor, p: int) -> Tensor:
    """Compute eigenvalues of Laplacian Δ = dδ + δd on Λᵖ"""

def extract_harmonic_forms(h_network: HarmonicNetwork, manifold: Manifold, ...) -> Tensor:
    """Extract harmonic p-forms from trained network"""

def verify_cohomology_ranks(h2_forms, h3_forms, topology: TopologyConfig) -> Dict:
    """Verify extracted forms match expected ranks b₂, b₃"""

def compute_harmonic_penalty(forms: Tensor, metric: Tensor, coords: Tensor) -> Tensor:
    """Harmonic equation penalty: Δω = 0"""
```

---

## 🔧 Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/gift-framework/g2-forge.git
cd g2-forge

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-core.txt  # Core: torch, numpy, scipy
pip install -r requirements-dev.txt   # Dev: pytest, jupyter, matplotlib
pip install -r requirements-test.txt  # Testing: pytest-cov, pytest-xdist

# Install package in editable mode
pip install -e .

# Verify installation
python -c "import g2forge as g2; print(g2.__version__)"

# Run test setup script
./setup_tests.sh  # Checks dependencies, CUDA availability
```

### Running Tests

**Test Organization** (pytest.ini configuration):

```bash
# All tests
pytest tests/

# By category (using markers)
pytest tests/ -m unit           # Unit tests only
pytest tests/ -m integration    # Integration tests
pytest tests/ -m "not slow"     # Skip slow tests

# By directory
pytest tests/unit/              # All unit tests
pytest tests/integration/       # All integration tests

# Specific test file
pytest tests/unit/test_operators.py -v

# Specific test function
pytest tests/unit/test_losses.py::test_gram_matrix_loss_parameterized -v

# Parallel execution (faster!)
pytest tests/unit/ -n auto      # Uses all CPU cores

# With coverage
pytest tests/ --cov=g2forge --cov-report=html

# Use provided scripts
./run_new_tests.sh              # Run physics module tests
```

**Test Markers** (defined in pytest.ini):
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.regression` - Regression tests
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.edge_case` - Edge case tests

### Common Test Fixtures

Available in `tests/conftest.py` (265 lines):

**Configuration Fixtures**:
- `small_topology_config` - Fast tests (b₂=5, b₃=20)
- `medium_topology_config` - (b₂=10, b₃=40)
- `gift_config` - GIFT v1.0 (b₂=21, b₃=77)
- `large_topology_config` - Stress tests (b₂=30, b₃=100)

**Geometric Fixtures**:
- `levi_civita` - Cached Levi-Civita tensor (session scope)
- `sample_coordinates` - 10 random 7D points with gradients
- `sample_metric` - Identity metric for testing
- `sample_phi_antisymmetric` - Properly antisymmetrized 3-form
- `sample_2form_antisymmetric` - Properly antisymmetrized 2-form

**Network Fixtures**:
- `phi_network_small`, `phi_network_gift` - PhiNetwork instances
- `harmonic_networks_small`, `harmonic_networks_gift` - (H², H³) network pairs

**Manifold Fixtures**:
- `k7_manifold_small`, `k7_manifold_gift` - K7Manifold instances

**Utility Functions** (also in conftest.py):
- `assert_antisymmetric_3form(phi, rtol, atol)` - Validate 3-form antisymmetry
- `assert_antisymmetric_2form(omega, rtol, atol)` - Validate 2-form antisymmetry
- `assert_symmetric(tensor, rtol, atol)` - Validate tensor symmetry

### Git Workflow

**Branch Strategy**:
- `main` - Stable releases
- `claude/*` - AI assistant feature branches (auto-created by Claude Code)
- Feature branches should start with descriptive name

**Current Branch**:
```bash
git branch
# * claude/claude-md-mid5m46wlb94j7ty-016aq8rQ1x414hhg6cnXprAN
```

**Making Changes**:

1. **Before starting**: Check current status
   ```bash
   git status
   git log --oneline -5
   ```

2. **Making commits**: Follow commit message conventions
   ```bash
   # Stage changes
   git add g2forge/module/file.py tests/unit/test_file.py

   # Descriptive commit message (imperative mood)
   git commit -m "Add universal topology support to LossFunction

   - Parameterize gram_matrix_loss by TopologyConfig
   - Remove hardcoded (21, 77) dimensions
   - Add topology validation in CompositeLoss.__init__
   - Update tests for multiple topologies"

   # Push to remote
   git push -u origin claude/claude-md-mid5m46wlb94j7ty-016aq8rQ1x414hhg6cnXprAN
   ```

3. **Commit Message Style**:
   - First line: Imperative mood, <72 chars ("Add feature" not "Added feature")
   - Blank line
   - Bullet points with details (what changed and why)
   - Reference issue numbers if applicable

### Code Style and Conventions

**Type Hints** - Always use comprehensive type hints:

```python
from typing import Dict, List, Tuple, Optional, Callable
import torch
from torch import Tensor

def hodge_star_3(
    phi: Tensor,
    metric: Tensor,
    eps_indices: Tensor,
    eps_signs: Tensor
) -> Tensor:
    """Docstring..."""
    ...

def create_k7_config(
    b2_m1: int,
    b3_m1: int,
    b2_m2: int,
    b3_m2: int
) -> G2ForgeConfig:
    """Docstring..."""
    ...
```

**Docstring Style** - Google format with mathematical notation:

```python
def compute_exterior_derivative(phi: Tensor, coords: Tensor) -> Tensor:
    """
    Compute exterior derivative dφ : Λ³ → Λ⁴.

    For a 3-form φ, the exterior derivative is:
        (dφ)_{ijkl} = ∂_i φ_{jkl} - ∂_j φ_{ikl} + ∂_k φ_{ijl} - ∂_l φ_{ijk}

    Uses automatic differentiation for exactness.

    Args:
        phi: Tensor[batch, 7, 7, 7] - Antisymmetric 3-form
        coords: Tensor[batch, 7] - Manifold coordinates (requires_grad=True)

    Returns:
        dphi: Tensor[batch, 7, 7, 7, 7] - Exterior derivative (4-form)

    Raises:
        ValueError: If coords does not have gradients enabled

    References:
        - Joyce (2000), Section 10.2
        - Spivak (1999), "Calculus on Manifolds"
    """
    ...
```

**Naming Conventions**:
- **Classes**: `PascalCase` (`PhiNetwork`, `K7Manifold`, `TopologyConfig`)
- **Functions/Methods**: `snake_case` (`compute_exterior_derivative`, `sample_coordinates`)
- **Constants**: `UPPER_CASE` (rare in this codebase)
- **Private/Protected**: `_leading_underscore` (`_create_scheduler`, `_compute_loss_component`)
- **Module imports**: `import g2forge as g2` (recommended alias)

**Mathematical Notation** - Use Unicode when helpful:
- φ (phi) for the G₂ 3-form
- ★ (star) for Hodge star operator
- Λ³, Λ⁴ for form spaces
- H², H³ for cohomology
- ℝ, ℤ for number systems
- Subscripts: b₂, b₃ (in docstrings/comments)

**Import Organization**:

```python
# Standard library
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Third-party
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eigh

# Local imports (relative)
from ..utils.config import TopologyConfig, G2ForgeConfig
from ..manifolds import Manifold
from .operators import hodge_star_3, compute_exterior_derivative
```

**Error Handling** - Descriptive errors with context:

```python
def validate(self):
    if self.dimension != 7:
        raise ValueError(
            f"G₂ manifolds must be 7-dimensional, got dimension={self.dimension}"
        )

    if self.topology.b1 != 0:
        raise ValueError(
            f"G₂ manifolds are typically simply connected (b₁=0), got b₁={self.topology.b1}"
        )

    if self.topology.b2 <= 0 or self.topology.b3 <= 0:
        raise ValueError(
            f"Invalid Betti numbers: b₂={self.topology.b2}, b₃={self.topology.b3}. "
            f"Both must be positive for non-trivial G₂ manifolds."
        )
```

**Testing Conventions**:

```python
import pytest
import torch
from torch.testing import assert_close

def test_hodge_star_3_returns_correct_shape(sample_phi_antisymmetric, sample_metric, levi_civita):
    """Test Hodge star returns 4-form with correct shape."""
    eps_indices, eps_signs = levi_civita
    phi = sample_phi_antisymmetric
    metric = sample_metric

    star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)

    assert star_phi.shape == (10, 7, 7, 7, 7), f"Expected shape (10,7,7,7,7), got {star_phi.shape}"

def test_exterior_derivative_torsion_free_constraint(phi_network_small, sample_coordinates):
    """Test that learned φ satisfies dφ ≈ 0 after training."""
    phi = phi_network_small.get_phi_tensor(sample_coordinates)
    dphi = compute_exterior_derivative(phi, sample_coordinates)

    torsion = torch.mean(dphi ** 2)

    assert torsion < 1e-6, f"Torsion too large: {torsion:.2e}"

@pytest.mark.parametrize("b2,b3", [(5, 20), (10, 40), (21, 77)])
def test_harmonic_network_output_dimensions_match_topology(b2, b3):
    """Test that HarmonicNetwork output dimension matches topology."""
    topology = TopologyConfig(b2=b2, b3=b3)
    h2_net = HarmonicNetwork(topology, p=2, hidden_dims=[128, 256, 128])

    assert h2_net.n_forms == b2, f"Expected {b2} H² forms, got {h2_net.n_forms}"
```

---

## 🎯 Common Tasks for AI Assistants

### Task 1: Adding a New Loss Function

**Scenario**: Implement a new physics-informed loss term.

**Steps**:

1. **Add function to `g2forge/core/losses.py`**:
   ```python
   def new_physics_loss(
       phi: Tensor,
       coords: Tensor,
       topology: TopologyConfig,  # ← Parameterize if needed!
       manifold: Manifold
   ) -> Tensor:
       """
       New physics constraint for G₂ structure.

       Mathematical background: ...

       Args:
           phi: G₂ 3-form
           coords: Manifold coordinates
           topology: Topology configuration (for parameterization)
           manifold: Manifold instance

       Returns:
           loss: Scalar loss value
       """
       # Implementation...
       return loss
   ```

2. **Integrate into `CompositeLoss`**:
   ```python
   class CompositeLoss(nn.Module):
       def forward(self, phi, h2_forms, h3_forms, coords, epoch=0):
           losses = {}

           # ... existing losses ...

           # Add new loss
           losses['new_physics'] = new_physics_loss(
               phi, coords, self.topology, self.manifold
           )

           # Weight it
           weighted_losses['new_physics'] = weights['new_physics'] * losses['new_physics']

           return {'losses': losses, 'weighted_losses': weighted_losses, 'total': total_loss}
   ```

3. **Add to configuration** (`g2forge/utils/config.py`):
   ```python
   @dataclass
   class CurriculumPhaseConfig:
       loss_weights: Dict[str, float] = field(default_factory=lambda: {
           'torsion_closure': 1.0,
           # ... existing weights ...
           'new_physics': 0.1,  # ← Add here
       })
   ```

4. **Write tests** (`tests/unit/test_losses.py`):
   ```python
   def test_new_physics_loss_shape(sample_phi_antisymmetric, sample_coordinates, small_topology):
       """Test new physics loss returns scalar."""
       manifold = create_manifold(...)
       loss = new_physics_loss(sample_phi_antisymmetric, sample_coordinates, small_topology, manifold)
       assert loss.shape == ()
       assert loss >= 0

   def test_new_physics_loss_gradient_flow():
       """Test gradients flow through new loss."""
       # ... implementation
   ```

5. **Document** in docstring and update ROADMAP.md if significant

### Task 2: Implementing a New Manifold Type

**Scenario**: Add Joyce construction (non-TCS) manifold.

**Steps**:

1. **Create new file** `g2forge/manifolds/joyce.py`:
   ```python
   from .base import Manifold
   from ..utils.config import TopologyConfig

   class JoyceManifold(Manifold):
       """
       G₂ manifold via Joyce construction.

       References:
           - Joyce (1996): "Compact Riemannian 7-manifolds with holonomy G₂"
       """

       def __init__(self, topology: TopologyConfig, joyce_params: JoyceParameters):
           self.topology = topology
           self.joyce_params = joyce_params
           self.dimension = 7

       def sample_coordinates(self, n_samples, **kwargs) -> Tensor:
           """Sample coordinates on Joyce manifold."""
           # Implementation...

       def compute_metric(self, coords: Tensor) -> Tensor:
           """Compute background metric."""
           # Implementation...
   ```

2. **Add parameters** to `g2forge/utils/config.py`:
   ```python
   @dataclass
   class JoyceParameters:
       """Parameters for Joyce construction."""
       # ... specific parameters ...

   @dataclass
   class ManifoldConfig:
       topology: TopologyConfig
       construction_type: str  # 'tcs' or 'joyce'
       tcs_params: Optional[TCSParameters] = None
       joyce_params: Optional[JoyceParameters] = None
   ```

3. **Update factory** in `g2forge/manifolds/__init__.py`:
   ```python
   def create_manifold(config: ManifoldConfig) -> Manifold:
       if config.construction_type == 'tcs':
           return K7Manifold(config.topology, config.tcs_params)
       elif config.construction_type == 'joyce':
           return JoyceManifold(config.topology, config.joyce_params)
       else:
           raise ValueError(f"Unknown construction type: {config.construction_type}")
   ```

4. **Write comprehensive tests**:
   - Unit tests: `tests/unit/test_joyce_manifold.py`
   - Integration tests: `tests/integration/test_joyce_training.py`

5. **Document** in README.md and create example in `examples/joyce_example.py`

### Task 3: Adding Support for New Topology Configuration

**Scenario**: User wants to train on topology (b₂=15, b₃=60).

**This should already work!** Verify:

```python
import g2forge as g2

# Method 1: Use create_k7_config with building blocks summing to target
config = g2.create_k7_config(b2_m1=8, b3_m1=30, b2_m2=7, b3_m2=30)
# Results in b₂=15, b₃=61 (close, but b₃ = b₃_m1 + b₃_m2 + 1)

# Method 2: Manually construct
from g2forge.utils import TopologyConfig, TCSParameters, ModuliParameters, ManifoldConfig

topology = TopologyConfig(b2=15, b3=60)
tcs_params = TCSParameters(...)
moduli = ModuliParameters(...)
manifold_config = ManifoldConfig(topology, tcs_params, moduli)
config = G2ForgeConfig(manifold=manifold_config, ...)

# Verify networks auto-size
trainer = g2.training.Trainer(config, device='cuda')
print(f"H² network outputs: {trainer.h2_net.n_forms}")  # Should be 15
print(f"H³ network outputs: {trainer.h3_net.n_forms}")  # Should be 60

# Train
results = trainer.train(num_epochs=15000)
```

**If not working**, check:
1. `HarmonicNetwork.__init__` sets `self.n_forms` from `topology.b2` / `topology.b3`
2. `CompositeLoss` uses `self.topology.b2` / `self.topology.b3` in gram matrix loss
3. No hardcoded `21` or `77` in loss functions
4. Validation functions use `config.manifold.topology` not hardcoded values

### Task 4: Debugging Training Issues

**Common Issues**:

**Issue 1: Loss exploding / NaN**

Check:
1. Learning rate too high (reduce in config)
2. Gradient clipping (add to Trainer)
3. Batch size too small (increase sampling)
4. Numerical instability in Hodge star (check det(g) > 0)

Debug:
```python
# Add gradient monitoring
for name, param in trainer.phi_network.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: grad_norm={grad_norm:.2e}")

# Check intermediate values
print(f"det(g) range: [{metric_det.min():.2e}, {metric_det.max():.2e}]")
print(f"phi norm: {phi.norm():.2e}")
```

**Issue 2: Torsion not decreasing**

Check:
1. Curriculum phase weights (torsion_closure should be high in phase 1)
2. Coordinate sampling (need enough points, requires_grad=True)
3. Exterior derivative implementation (check autodiff works)

Debug:
```python
# Test exterior derivative in isolation
coords = torch.randn(100, 7, requires_grad=True)
phi = phi_network.get_phi_tensor(coords)
dphi = compute_exterior_derivative(phi, coords)
print(f"dφ norm: {dphi.norm():.2e}")
```

**Issue 3: Harmonic forms not orthogonal**

Check:
1. Gram matrix loss weight (should increase in phase 3+)
2. Network capacity (may need larger hidden dims for higher b₂, b₃)
3. Metric quality (need good metric for inner product)

Debug:
```python
# Compute Gram matrix directly
h2_forms = trainer.h2_net(coords)
gram = compute_gram_matrix(h2_forms, metric, coords)
print(f"Gram matrix:\n{gram[:5, :5]}")  # Should be near identity
print(f"Off-diagonal max: {(gram - torch.eye(gram.size(0))).abs().max():.2e}")
```

### Task 5: Adding Validation Checks

**Scenario**: Add new geometric validation (e.g., check scalar curvature).

**Steps**:

1. **Implement in `g2forge/validation/geometric.py`**:
   ```python
   class ScalarCurvatureValidator:
       """Validate scalar curvature R = 0 (consequence of Ricci-flatness)."""

       def validate(self, metric: Tensor, coords: Tensor) -> ValidationResult:
           """
           Compute scalar curvature and check R ≈ 0.

           For Ricci-flat manifolds (like G₂), R = tr(Ric) = 0.
           """
           # Compute Christoffel symbols
           christoffel = compute_christoffel_symbols(metric, coords)

           # Compute Riemann tensor
           riemann = compute_riemann_tensor(christoffel, coords)

           # Contract to scalar curvature
           R = compute_scalar_curvature(riemann, metric)

           scalar_curvature_norm = R.abs().mean().item()

           return ValidationResult(
               passed=scalar_curvature_norm < self.tolerance,
               metric_value=scalar_curvature_norm,
               details={'max_R': R.abs().max().item()}
           )
   ```

2. **Add to `GeometricValidator`**:
   ```python
   class GeometricValidator:
       def __init__(self):
           self.ricci_validator = RicciValidator()
           self.holonomy_tester = HolonomyTester()
           self.metric_validator = MetricValidator()
           self.scalar_curvature_validator = ScalarCurvatureValidator()  # ← New

       def validate_all(self, phi, metric, coords) -> Dict[str, ValidationResult]:
           results = {}
           results['ricci'] = self.ricci_validator.validate(metric, coords)
           results['holonomy'] = self.holonomy_tester.validate(phi, metric, coords)
           results['metric'] = self.metric_validator.validate(metric)
           results['scalar_curvature'] = self.scalar_curvature_validator.validate(metric, coords)
           return results
   ```

3. **Write tests** in `tests/unit/test_geometric_validation.py`

4. **Integrate into training** (optional):
   ```python
   # In Trainer.train():
   if epoch % 1000 == 0:
       with torch.no_grad():
           validation_results = self.validator.validate_all(phi, metric, coords)
           for name, result in validation_results.items():
               print(f"{name}: {'✓' if result.passed else '✗'} (value={result.metric_value:.2e})")
   ```

### Task 6: Optimizing Performance

**Common Optimizations**:

**1. Batch Size Tuning**:
```python
# Increase batch size for better GPU utilization
config.training.batch_size = 2048  # From default 1024
```

**2. Mixed Precision Training**:
```python
# In Trainer.__init__:
from torch.cuda.amp import GradScaler, autocast

self.scaler = GradScaler()

# In training loop:
with autocast():
    phi = self.phi_network.get_phi_tensor(coords)
    loss_dict = self.loss_fn(phi, h2_forms, h3_forms, coords, epoch)
    loss = loss_dict['total']

self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

**3. Gradient Checkpointing** (for large networks):
```python
import torch.utils.checkpoint as checkpoint

# In PhiNetwork.forward:
def forward(self, x):
    x = self.fourier_features(x)
    x = checkpoint.checkpoint(self.mlp, x)  # Trade compute for memory
    return x
```

**4. Operator Subsampling** (already implemented):
```python
# In compute_exterior_derivative:
if batch_size > 500:
    # Subsample for efficiency
    indices = torch.randperm(batch_size)[:500]
    dphi_sampled = compute_dphi(phi[indices], coords[indices])
    # ...
```

**5. Profile Code**:
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    results = trainer.train(num_epochs=100)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 🚨 Critical Design Principles

### 1. **UNIVERSALITY IS KEY**

**DO NOT hardcode topology-specific values!**

❌ **Bad**:
```python
def gram_matrix_loss(h2_forms, h3_forms, metric, coords):
    # WRONG: Hardcoded to GIFT's topology
    gram_h2 = compute_gram(h2_forms, metric, coords)  # shape [21, 21]
    identity_h2 = torch.eye(21)  # ← HARDCODED!
    loss_h2 = torch.mean((gram_h2 - identity_h2) ** 2)
    # ...
```

✅ **Good**:
```python
def gram_matrix_loss(h2_forms, h3_forms, topology: TopologyConfig, metric, coords):
    # CORRECT: Parameterized by topology
    gram_h2 = compute_gram(h2_forms, metric, coords)  # shape [b2, b2]
    identity_h2 = torch.eye(topology.b2)  # ← PARAMETERIZED!
    loss_h2 = torch.mean((gram_h2 - identity_h2) ** 2)
    # ...
```

**Check all new code**: Search for literal `21` or `77` - these should almost never appear except in comments or tests for GIFT reproduction.

### 2. **Type Safety with Dataclasses**

Always use configuration dataclasses, never raw dictionaries:

❌ **Bad**:
```python
config = {
    'b2': 21,
    'b3': 77,
    'hidden_dims': [256, 512, 256]
}
```

✅ **Good**:
```python
topology = TopologyConfig(b2=21, b3=77)
architecture = NetworkArchitectureConfig(
    phi_hidden_dims=[256, 512, 512, 256],
    harmonic_hidden_dims=[256, 512, 256]
)
config = G2ForgeConfig(
    manifold=ManifoldConfig(topology, ...),
    architecture=architecture,
    ...
)
```

Benefits:
- Type checking
- Validation on construction
- Serialization built-in
- Self-documenting

### 3. **Gradient-Friendly Operations**

All geometric operators must support automatic differentiation:

✅ **Good practices**:
- No in-place operations (`+=`, `*=`) on tensors requiring grad
- Use `torch.autograd.grad()` for derivatives
- Return new tensors, don't modify inputs
- Keep operations differentiable (avoid `.detach()` unless necessary)

Example:
```python
def compute_exterior_derivative(phi: Tensor, coords: Tensor) -> Tensor:
    """Uses autograd - no manual derivative computation!"""
    batch_size = phi.shape[0]
    dphi = torch.zeros(batch_size, 7, 7, 7, 7, device=phi.device)

    for i in range(7):
        # Gradient w.r.t. coordinate i
        grad_phi = torch.autograd.grad(
            phi, coords,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]
        # No in-place operations!
        dphi = dphi + (some_function(grad_phi))  # Creates new tensor

    return dphi
```

### 4. **Test Everything**

For every new function/class:

1. **Unit test** - Test in isolation with simple inputs
2. **Integration test** - Test in context of full pipeline
3. **Parameterized test** - Test with multiple topologies
4. **Edge case test** - Test boundary conditions

Example:
```python
# Unit test
def test_my_function_basic():
    """Test basic functionality with known input."""
    result = my_function(simple_input)
    assert result == expected_output

# Parameterized test (CRITICAL for universality!)
@pytest.mark.parametrize("b2,b3", [(5, 20), (10, 40), (21, 77), (30, 100)])
def test_my_function_multiple_topologies(b2, b3):
    """Test with various topologies."""
    topology = TopologyConfig(b2=b2, b3=b3)
    result = my_function(topology)
    assert result.shape[0] == b2  # Or whatever topology-dependent property

# Edge case test
def test_my_function_edge_cases():
    """Test boundary conditions."""
    # Zero input
    # Negative values (should raise)
    # Very large values
    # NaN/inf handling
```

### 5. **Documentation Standards**

Every public function needs:

1. **Docstring** with:
   - One-line summary
   - Mathematical description (if applicable)
   - Args with types and descriptions
   - Returns with type and description
   - Raises for exceptions
   - References to papers/theory

2. **Type hints** on function signature

3. **Example usage** (for complex functions)

Example:
```python
def compute_exterior_derivative(phi: Tensor, coords: Tensor) -> Tensor:
    """
    Compute exterior derivative dφ : Λ³ → Λ⁴.

    For a 3-form φ on a 7-manifold, the exterior derivative is the 4-form:
        (dφ)_{ijkl} = ∂_i φ_{jkl} - ∂_j φ_{ikl} + ∂_k φ_{ijl} - ∂_l φ_{ijk}

    This implementation uses automatic differentiation for exactness and
    naturally handles the alternating signs from antisymmetrization.

    Args:
        phi: Tensor[batch, 7, 7, 7]
            Antisymmetric 3-form on the manifold
        coords: Tensor[batch, 7]
            Manifold coordinates with requires_grad=True

    Returns:
        dphi: Tensor[batch, 7, 7, 7, 7]
            Exterior derivative (antisymmetric 4-form)

    Raises:
        ValueError: If coords does not have gradients enabled
        RuntimeError: If phi was not computed from coords with autodiff

    Example:
        >>> coords = torch.randn(100, 7, requires_grad=True)
        >>> phi = phi_network.get_phi_tensor(coords)
        >>> dphi = compute_exterior_derivative(phi, coords)
        >>> torsion = torch.mean(dphi ** 2)  # Should be ~0 for trained network

    References:
        - Joyce (2000), "Compact Manifolds with Special Holonomy", Section 10.2
        - Spivak (1999), "Calculus on Manifolds", Chapter 7

    Note:
        This operation is computationally expensive (O(n²) in batch size due to
        autodiff). For large batches, consider subsampling.
    """
    if not coords.requires_grad:
        raise ValueError("coords must have requires_grad=True for gradient computation")

    # Implementation...
```

---

## 📚 Key References

### Mathematical Background

1. **G₂ Geometry**:
   - Joyce, D. (2000). *Compact Manifolds with Special Holonomy*. Oxford University Press.
   - Bryant, R., & Salamon, S. (1989). "On the construction of some complete metrics with exceptional holonomy"

2. **TCS Construction**:
   - Kovalev, A. (2003). "Twisted connected sums and special Riemannian holonomy"
   - Corti, A., Haskins, M., Nordström, J., & Pacini, T. (2015). "G₂-manifolds and associative submanifolds via semi-Fano 3-folds"

3. **Differential Geometry**:
   - Spivak, M. (1999). *A Comprehensive Introduction to Differential Geometry*
   - Lee, J. M. (2018). *Introduction to Riemannian Manifolds*

### Computational Methods

1. **Physics-Informed Neural Networks**:
   - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks"
   - Lu, L., et al. (2021). "DeepXDE: A deep learning library for solving differential equations"

2. **GIFT Framework**:
   - Original validated implementation at https://github.com/gift-framework/GIFT
   - GIFT v1.0: Torsion-free G₂ metrics (10⁻⁷ to 10⁻¹¹ precision)
   - GIFT v1.1b: Multi-grid analysis
   - GIFT v1.2a/b: Volume normalization
   - GIFT 2.1: RG flow computation

---

## 🐛 Known Issues and Limitations

### Current Limitations

1. **TCS Construction Only**: Currently supports only K7 manifolds via TCS. Joyce construction not yet implemented.

2. **GPU Required for Training**: Full 15k epoch training requires GPU. CPU training is possible but slow (~100x slower).

3. **High Memory Usage**: Full batch training with b₂=21, b₃=77 needs ~8GB GPU memory. Reduce batch size if OOM.

4. **Long Training Times**: Full convergence takes 15k epochs (~2-6 hours on modern GPU depending on topology).

5. **No Distributed Training**: Multi-GPU training not yet implemented.

### Known Issues

1. **Gram Matrix Scaling**: For very large b₂, b₃ (>100), Gram matrix loss may need rescaling.

2. **Neck Region Sampling**: TCS neck region needs careful sampling. Too coarse → poor boundary matching.

3. **Initial Conditions**: Network initialization affects convergence. May need multiple runs for difficult topologies.

### Workarounds

**Issue**: Out of memory during training
**Workaround**:
```python
config.training.batch_size = 512  # Reduce from default 1024
config.training.checkpoint_interval = 500  # Save more frequently
```

**Issue**: Torsion plateau (not decreasing below 10⁻⁴)
**Workaround**:
```python
# Increase torsion loss weight in later phases
config.training.curriculum_phases['phase4'].loss_weights['torsion_closure'] = 5.0
config.training.curriculum_phases['phase4'].loss_weights['torsion_coclosure'] = 5.0
```

**Issue**: Training instability with new topology
**Workaround**:
```python
# Start with smaller learning rate
config.training.learning_rate = 5e-4  # From default 1e-3

# Add warmup
config.training.warmup_epochs = 500
```

---

## 🔍 Debugging Tips

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

trainer = Trainer(config, device='cuda', verbose=True)
```

### Check Intermediate Values

```python
# In training loop, add:
if epoch % 100 == 0:
    print(f"Epoch {epoch}:")
    print(f"  φ norm: {phi.norm():.2e}")
    print(f"  det(g) range: [{metric_det.min():.2e}, {metric_det.max():.2e}]")
    print(f"  dφ norm: {dphi.norm():.2e}")
    print(f"  H² forms norm: {h2_forms.norm():.2e}")
    print(f"  Gram H² diagonal: {gram_h2.diag().mean():.3f}")
```

### Visualize Loss Components

```python
import matplotlib.pyplot as plt

history = results['training_history']
epochs = range(len(history['total_loss']))

plt.figure(figsize=(12, 8))
for name in ['torsion_closure', 'torsion_coclosure', 'gram_matrix', 'volume']:
    plt.plot(epochs, history[name], label=name)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Components Over Training')
plt.savefig('loss_breakdown.png')
```

### Test Components Independently

```python
# Test operator in isolation
coords = torch.randn(10, 7, requires_grad=True)
phi = torch.randn(10, 7, 7, 7)
phi = 0.5 * (phi - phi.transpose(1, 2))  # Antisymmetrize i,j
dphi = compute_exterior_derivative(phi, coords)
print(f"dphi shape: {dphi.shape}")
print(f"dphi antisymmetric: {validate_antisymmetry(dphi, 4)}")
```

### Use PyTorch Anomaly Detection

```python
# Detect NaN/inf and trace back to source
torch.autograd.set_detect_anomaly(True)

try:
    results = trainer.train(num_epochs=1000)
except RuntimeError as e:
    print(f"Anomaly detected: {e}")
    # Will show exact operation causing NaN
```

---

## 📞 Getting Help

### For AI Assistants

If you encounter issues while working on g2-forge:

1. **Check this document** for relevant sections
2. **Read existing code** in similar modules (patterns are consistent)
3. **Review tests** to understand expected behavior
4. **Check docstrings** for mathematical context
5. **Consult reference papers** listed above

### For Human Developers

- **Issues**: https://github.com/gift-framework/g2-forge/issues
- **Email**: brieuc@bdelaf.com
- **GIFT Framework**: https://github.com/gift-framework/GIFT

---

## 🎓 Learning Path for New AI Assistants

### Level 1: Understanding the Basics

1. Read README.md for high-level overview
2. Study `g2forge/utils/config.py` to understand configuration system
3. Explore `g2forge/core/operators.py` for mathematical foundations
4. Review `examples/complete_example.py` for typical usage

### Level 2: Understanding the Architecture

1. Study `g2forge/networks/` for neural network design (especially auto-sizing!)
2. Read `g2forge/core/losses.py` for training objectives
3. Understand `g2forge/training/trainer.py` for curriculum learning
4. Review test fixtures in `tests/conftest.py`

### Level 3: Contributing

1. Pick a simple task (e.g., add a validation check)
2. Write tests first (TDD approach)
3. Implement feature following existing patterns
4. Ensure all tests pass: `pytest tests/ -v`
5. Commit with descriptive message
6. Push and create PR (or provide to user)

---

## ✅ Pre-Commit Checklist

Before committing code, verify:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] No hardcoded topology values (search for `21`, `77`)
- [ ] Type hints on all public functions
- [ ] Docstrings in Google format with mathematical notation
- [ ] New functions have corresponding tests
- [ ] Tests include multiple topologies (parameterized)
- [ ] No in-place operations on gradient tensors
- [ ] Error messages are descriptive
- [ ] Code follows naming conventions (PascalCase/snake_case)
- [ ] Imports organized (stdlib, third-party, local)
- [ ] No commented-out code
- [ ] No debug print statements (use logging)

Quick check:
```bash
# Run all tests
pytest tests/ -v --tb=short

# Check for hardcoded values
git diff | grep -E '(^[\+].*\b21\b|^[\+].*\b77\b)' | grep -v 'test_' | grep -v '#'

# Check for type hints
git diff g2forge/ | grep -E '^[\+]def ' | grep -v ' -> '

# Run specific test categories
pytest tests/unit/ -v -m "not slow"
pytest tests/integration/ -v
```

---

## 🎉 Success Criteria

You'll know you're working correctly with g2-forge when:

1. ✅ Your code works for **arbitrary (b₂, b₃)**, not just GIFT's (21, 77)
2. ✅ Networks automatically adapt to topology configuration
3. ✅ All tests pass across multiple topology configurations
4. ✅ Loss functions use `TopologyConfig` parameters, not hardcoded values
5. ✅ Code is type-safe with comprehensive docstrings
6. ✅ Changes follow existing architecture patterns
7. ✅ Git commits are descriptive and well-organized

**The Universal Test**: Can your code train a G₂ metric on a manifold with (b₂=19, b₃=73) without ANY code changes, just a config change? If yes, you've achieved universality! 🚀

---

**End of CLAUDE.md**

*This document is maintained as part of the g2-forge project. For updates or corrections, please submit a PR or contact brieuc@bdelaf.com.*
