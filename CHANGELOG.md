# Changelog

All notable changes to g2-forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test coverage improvements (60+ new tests)
- Physics modules: VolumeNormalizer and RG flow computation
- Spectral analysis tools for harmonic form extraction
- GitHub Pages documentation site

### Changed
- Improved error messages with mathematical context
- Enhanced configuration validation

### Fixed
- Various edge case handling in geometric operators

## [0.1.0] - 2025-01-15

### Added
- **Universal Framework**: First release supporting ANY G2 manifold topology
- **Configuration System**: Type-safe dataclass configuration with validation
  - `TopologyConfig` for Betti numbers (b2, b3)
  - `ManifoldConfig` for complete manifold specification
  - `G2ForgeConfig` for full training configuration
- **Differential Geometry Operators** (`g2forge/core/operators.py`)
  - `hodge_star_3`: Hodge star operator for 3-forms
  - `compute_exterior_derivative`: Exterior derivative via autodiff
  - `compute_coclosure`: Codifferential operator
  - `build_levi_civita_sparse_7d`: Sparse Levi-Civita tensor
  - `reconstruct_metric_from_phi`: Metric reconstruction from G2 3-form
- **Parameterized Loss Functions** (`g2forge/core/losses.py`)
  - `torsion_closure_loss`: dφ = 0 constraint
  - `torsion_coclosure_loss`: d★φ = 0 constraint
  - `gram_matrix_loss`: Harmonic form orthonormality (parameterized by topology!)
  - `volume_loss`: Volume normalization
  - `CompositeLoss`: Combined loss with adaptive weighting
- **Manifold Abstractions** (`g2forge/manifolds/`)
  - `Manifold`: Abstract base class
  - `TCSManifold`: Twisted Connected Sum base
  - `K7Manifold`: K7 TCS implementation
- **Auto-sizing Neural Networks** (`g2forge/networks/`)
  - `PhiNetwork`: G2 3-form generator
  - `HarmonicNetwork`: Auto-sizes from topology (b2 or b3 forms)
- **Training Infrastructure** (`g2forge/training/`)
  - `Trainer`: Complete training loop with curriculum learning
  - 5-phase curriculum from GIFT v1.0
  - Checkpointing and resumption
  - Metrics tracking
- **Geometric Validation** (`g2forge/validation/`)
  - `RicciValidator`: Verify Ricci-flatness
  - `HolonomyTester`: Test G2 holonomy
  - `MetricValidator`: Validate metric properties
- **Examples**
  - `complete_example.py`: Full pipeline demonstration
  - GIFT v1.0 reproduction configuration
- **Testing**
  - Comprehensive test suite (34 files, 8000+ lines)
  - Unit, integration, regression, and edge case tests
  - Parameterized tests for multiple topologies

### Technical Details
- ~5,300 lines of production code
- ~8,000 lines of tests
- Based on validated GIFT v1.0-1.1b algorithms
- 87% code reuse from GIFT, 13% new universalization logic

## [0.0.1] - 2024-12-01

### Added
- Initial project structure
- Basic operator implementations
- Proof of concept for universal topology support

---

[Unreleased]: https://github.com/gift-framework/g2-forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gift-framework/g2-forge/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/gift-framework/g2-forge/releases/tag/v0.0.1
