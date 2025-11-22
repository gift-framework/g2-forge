# g2-forge Development Roadmap

**Vision:** Transform GIFT-specific G₂ metric construction into a universal framework for any G₂ manifold

---

## ✅ Phase 1: Analysis & Foundation (COMPLETED)

### What We Did

1. **Copied GIFT examples** → `gift_examples/`
   - v1.0: Production TCS implementation
   - v1.1: Explicit metric construction
   - v1.1a: Refined version with training history

2. **Analyzed code structure** → `ANALYSIS.md`
   - Identified 92% of code is universally applicable
   - Found only 8% is GIFT-specific (config hardcoding)
   - Documented clear path to generalization

3. **Created package structure** → `g2forge/`
   ```
   g2forge/
   ├── core/          # Operators, losses, metrics
   ├── manifolds/     # Manifold definitions (K7, Joyce, etc.)
   ├── networks/      # Neural network architectures
   ├── training/      # Training loops, curriculum
   ├── validation/    # Geometric & topological tests
   ├── analysis/      # Yukawa, spectral analysis
   └── utils/         # Config, checkpoints, helpers
   ```

### Key Insights

- **90%+ code reuse possible** - Most is pure differential geometry
- **Minimal refactoring needed** - Just parameterize topology
- **Proven algorithms** - v1.0 achieved world-record torsion (10⁻¹¹)

---

## 🔄 Phase 2: Core Implementation (NEXT)

### Step 2.1: Configuration System

**Goal:** Universal config that works for any G₂ manifold

**Files to create:**
- `g2forge/utils/config.py` - Config dataclasses
- `examples/configs/k7_gift.yaml` - GIFT K₇ reproduction
- `examples/configs/k7_custom.yaml` - Custom topology example

**Code:**
```python
@dataclass
class TopologyConfig:
    b1: int = 0
    b2: int  # Required
    b3: int  # Required
    b4: int = 0

@dataclass
class ManifoldConfig:
    type: str  # "K7", "Joyce", "Custom"
    construction: str  # "TCS", "ConnectedSum"
    topology: TopologyConfig
    # ... moduli, parameters
```

**Validation:** Load GIFT config, verify equivalent to original

---

### Step 2.2: Port Differential Operators

**Goal:** 100% reusable geometric operators

**Files to create:**
- `g2forge/core/operators.py` ← Copy from `gift_examples/1.0/tcs_operators.py`

**Changes needed:** NONE (already universal!)

**Functions:**
```python
✅ build_levi_civita_sparse_7d()
✅ hodge_star_3(phi, metric, eps_indices, eps_signs)
✅ compute_coclosure(star_phi, coords)
✅ compute_exterior_derivative(phi, coords)
✅ region_weighted_torsion(dphi, region_weights)
```

**Validation:** Unit tests against known geometric identities

---

### Step 2.3: Port Loss Functions

**Goal:** Parameterized loss functions

**Files to create:**
- `g2forge/core/losses.py` ← Adapt from `gift_examples/1.0/losses.py`

**Changes needed:**
```python
# BEFORE (hardcoded):
target_rank=21

# AFTER (parameterized):
target_rank=config.topology.b2
```

**Validation:** Reproduce GIFT loss values exactly

---

### Step 2.4: Manifold Abstraction

**Goal:** Abstract base class + K₇ implementation

**Files to create:**
- `g2forge/manifolds/base.py` - Abstract Manifold
- `g2forge/manifolds/k7.py` - K₇ from GIFT

**Abstract interface:**
```python
class Manifold(ABC):
    @abstractmethod
    def sample_coordinates(self, n_samples, grid_n) -> Tensor

    @abstractmethod
    def get_region_weights(self, coords) -> Dict[str, Tensor]

    @abstractmethod
    def get_associative_cycles(self) -> List[Cycle]

    @property
    @abstractmethod
    def topology(self) -> TopologyConfig
```

**K₇ implementation:** Extract from GIFT notebooks

---

## 🎯 Phase 3: Training System (WEEK 2)

### Step 3.1: Training Infrastructure

**Files:**
- `g2forge/training/trainer.py`
- `g2forge/training/curriculum.py`

**Key features:**
- Configurable curriculum (phase durations, loss weights)
- Automatic batch size / GPU memory management
- Checkpoint resume with state preservation

---

### Step 3.2: Network Architectures

**Files:**
- `g2forge/networks/phi_network.py` - 3-form network
- `g2forge/networks/harmonic_network.py` - H² and H³ extraction

**Auto-sizing:**
```python
# Network output dimension determined by topology
harmonic_h2 = HarmonicNetwork(
    n_forms=config.topology.b2  # Automatically sized!
)
```

---

## 🧪 Phase 4: Validation & Testing (WEEK 3)

### Step 4.1: Reproduce GIFT Results

**Goal:** Exact reproduction of GIFT v1.0 performance

**Test:**
```python
def test_gift_v1_0_reproduction():
    config = load_gift_v1_0_config()
    trainer = Trainer(config)
    results = trainer.train(epochs=15000)

    assert results['torsion_closure'] < 1e-3
    assert results['rank_h2'] == 21
    assert results['rank_h3'] == 77
```

---

### Step 4.2: Geometric Validation Suite

**Files:**
- `g2forge/validation/geometric.py` ← From `gift_examples/1.0/validation.py`

**Tests:**
- Ricci-flatness: ||Ric|| < 10⁻⁴
- Holonomy G₂: φ preserved under parallel transport
- Volume: det(g) ≈ 1

---

## 🚀 Phase 5: High-Level API (WEEK 4)

### User-Facing API

**Goal:** Simple, Pythonic interface

```python
import g2forge as g2

# Define manifold
manifold = g2.manifolds.K7(
    b2_m1=11, b3_m1=40,
    b2_m2=10, b3_m2=37
)

# Create model
model = g2.networks.RegionalG2Network(
    manifold=manifold,
    hidden_dims=[384, 384, 256]
)

# Train
trainer = g2.training.CurriculumTrainer(
    model=model,
    phases=5,
    epochs_per_phase=[2000, 3000, 3000, 2000, 5000]
)

results = trainer.train()
print(f"Final torsion: {results['torsion']:.2e}")
```

---

## 📚 Phase 6: Documentation & Examples

### Examples

1. **Basic K₇ construction** (`examples/k7_basic.py`)
2. **Reproduce GIFT** (`examples/reproduce_gift_v1_0.py`)
3. **Custom topology** (`examples/custom_b2_b3.py`)
4. **Joyce manifold** (`examples/joyce_construction.py`)

### Documentation

- Theory background (G₂ geometry primer)
- Architecture guide (network design)
- Training tutorial (curriculum learning)
- API reference (auto-generated)

---

## 🎯 Success Criteria

### MVP (Minimum Viable Product)

- [ ] Reproduce GIFT v1.0 results within 10%
- [ ] Support arbitrary (b₂, b₃) for K₇ manifold
- [ ] Clean API matching README examples
- [ ] 3 working example notebooks

### v1.0 Release

- [ ] K₇ and Joyce manifolds supported
- [ ] Full test coverage (>80%)
- [ ] Documentation complete
- [ ] Published on PyPI
- [ ] Preprint on arXiv

---

## 📊 Progress Tracking

| Phase | Status | Completion | Estimated Time |
|-------|--------|------------|----------------|
| 1. Analysis | ✅ Done | 100% | ~2 hours |
| 2. Core | 🔄 In Progress | 10% | ~6 hours |
| 3. Training | ⏳ Pending | 0% | ~8 hours |
| 4. Validation | ⏳ Pending | 0% | ~4 hours |
| 5. API | ⏳ Pending | 0% | ~3 hours |
| 6. Docs | ⏳ Pending | 0% | ~4 hours |

**Total to MVP:** ~27 hours (~1 week full-time)

---

## 🔥 Immediate Next Steps

1. **Create config system** (`g2forge/utils/config.py`)
2. **Port operators** (`g2forge/core/operators.py`)
3. **Define Manifold ABC** (`g2forge/manifolds/base.py`)
4. **Implement K₇** (`g2forge/manifolds/k7.py`)

**Should we continue? Pick any task and let's build! 🚀**
