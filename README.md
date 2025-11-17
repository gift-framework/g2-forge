# g2-forge

**Neural Construction of Exceptional Holonomy Metrics**

Physics-Informed Neural Networks (PINNs) for constructing explicit G₂ holonomy metrics on compact 7-manifolds via curriculum learning and regional network architectures.

---

## What is this?

G₂ manifolds are 7-dimensional spaces with exceptional geometric properties (Ricci-flat, torsion-free) that appear in M-theory compactifications and string phenomenology. Constructing explicit metrics on these manifolds is a notoriously difficult open problem in differential geometry.

**g2-forge** solves this computationally using deep learning, achieving:

- **Torsion-free precision**: 10⁻⁷ to 10⁻¹¹ (world-record for numerical constructions)
- **Topological consistency**: Explicit harmonic basis with b₂ = 21, b₃ = 77
- **Computational efficiency**: ~2 hours training on single GPU
- **Full differentiability**: All geometric quantities computed via automatic differentiation

---

## Quick Start

```python
import g2forge as g2

# Initialize K₇ manifold with twisted connected sum structure
manifold = g2.manifolds.K7(
    b2_m1=11, b3_m1=40,  # M₁ topology
    b2_m2=10, b3_m2=37,  # M₂ topology
)

# Create regional network architecture
model = g2.networks.RegionalG2Network(
    manifold=manifold,
    hidden_dims=[384, 384, 256],
    fourier_features=32
)

# Train with 4-phase curriculum
trainer = g2.training.CurriculumTrainer(
    model=model,
    phases=4,
    epochs_per_phase=[2000, 3000, 3000, 2000]
)

results = trainer.train()
print(f"Final torsion: {results['torsion']:.2e}")
# Final torsion: 1.08e-07
```

---

## Benchmark Results

| Version | Torsion | det(Gram) | Training Time | Method |
|---------|---------|-----------|---------------|---------|
| v0.4 | 1.33×10⁻¹¹ | 1.12 | 6.4 h | Dual network |
| v0.7 | 1.08×10⁻⁷ | 1.002 | 4.0 h | Simplified |
| **v0.9a** | **1.08×10⁻⁷** | **1.0021** | **1.76 h** | Regional TCS |

*K₇ manifold with b₂ = 21, b₃ = 77. Single A100 GPU.*

---

## Installation

```bash
# Clone repository
git clone https://github.com/gift-framework/g2-forge.git
cd g2-forge

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy, SciPy
- CUDA-capable GPU (recommended)

---

## Key Features

### Regional Network Architecture
Explicitly models the twisted connected sum (TCS) structure K₇ = M₁ᵀ ∪_φ M₂ᵀ with separate networks for:
- Asymptotically cylindrical end M₁
- Gluing neck region
- Asymptotically cylindrical end M₂

### Curriculum Learning
Four-phase training strategy progressively emphasizing:
1. Topological foundation (neck stability)
2. Asymptotic matching (acyl structures)
3. Torsion reduction (geometric refinement)
4. Harmonic extraction (cohomology stabilization)

### Geometric Rigor
All constraints enforced through physics-informed losses:
- `dφ = 0` and `d*φ = 0` (torsion-free G₂ structure)
- `det(g) = 1` (volume normalization)
- `Gram(H²) ≈ I₂₁` (orthonormal harmonic basis)

---

## Documentation

- [Theory Background](docs/theory.md) - G₂ geometry and TCS construction
- [Architecture Guide](docs/architecture.md) - Neural network design
- [Training Tutorial](examples/k7_full.ipynb) - Step-by-step walkthrough
- [API Reference](docs/api.md) - Complete function documentation

---

## Examples

**Basic K₇ construction:**
```python
# See examples/k7_basic.ipynb
```

**Custom manifold:**
```python
# See examples/custom_manifold.ipynb
```

**Reproduce published results:**
```python
# See examples/reproduce_v09a.ipynb
```

---

## Scientific Context

This work emerged from the [GIFT framework](https://github.com/gift-framework/GIFT) (Geometric Information Field Theory), which explores connections between G₂ geometry and fundamental physics. However, **g2-forge is a standalone computational tool** applicable to any G₂ manifold construction problem in differential geometry, string theory, or M-theory phenomenology.

### Publications

- **v0.9a preprint**: [Zenodo DOI] - "Numerical G₂ Metric Construction via Regional PINNs"
- **GIFT main**: [Zenodo DOI] - Broader theoretical framework

---

## Contributing

Contributions welcome! Areas of particular interest:

- Support for Joyce construction (non-TCS G₂ manifolds)
- Spectral geometry analysis (Laplacian eigendecomposition)
- Geodesic computation and minimal submanifolds
- Performance optimization (distributed training, mixed precision)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) file.

**Citation:**
```bibtex
@software{g2forge2024,
  title={g2-forge: Neural Construction of Exceptional Holonomy Metrics},
  author={[Your Name]},
  year={2024},
  url={https://github.com/gift-framework/g2-forge}
}
```

---

## Acknowledgments

- Dominic Joyce - G₂ manifold theory
- Alexei Kovalev - Twisted connected sum construction
- Corti, Haskins, Nordström, Pacini - TCS refinements
- Raissi, Perdikaris, Karniadakis - Physics-informed neural networks

---

**Status**: 🚧 Active development toward v1.0

**Contact**: [your-email] | [Twitter/X handle]
