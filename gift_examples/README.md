# GIFT Examples - Reference Implementations

This directory contains reference implementations from the GIFT framework's G2_ML module (versions 1.0, 1.1, 1.1a). These serve as the basis for extracting and generalizing the methodology into the standalone g2-forge package.

## Purpose

These examples are **GIFT-specific** implementations that work for the K₇ manifold with:
- b₂ = 21 harmonic 2-forms
- b₃ = 77 harmonic 3-forms
- Specific GIFT framework parameters

Our goal is to **analyze, extract, and generalize** these implementations to work for **any G₂ manifold** with arbitrary topological parameters.

## Directory Structure

### 1.0/ - Production TCS Implementation
- **Focus**: Torsion Cohomology Solver with Twisted Connected Sum (TCS) structure
- **Key files**:
  - `losses.py` - Loss functions for G₂ constraints
  - `training.py` - Curriculum learning implementation
  - `validation.py` - Geometric validation (Ricci, holonomy)
  - `yukawa.py` - Yukawa tensor computation
  - `tcs_operators.py` - TCS-specific differential operators
- **Performance**: Torsion < 10⁻³, Full b₃=77 extraction
- **Architecture**: Regional networks (M₁, neck, M₂)

### 1_1/ - Explicit Metric v1.1
- **Focus**: Simplified explicit metric construction
- **Key file**: `K7_G2_TCS_ExplicitMetric_v1_1.ipynb`

### 1_1a/ - Refined v1.1a
- **Focus**: Latest refinements with improved training
- **Key file**: `K7_G2_TCS_ExplicitMetric_v1_1a.ipynb`
- **Outputs**:
  - Checkpoints, harmonic forms, Yukawa tensor
  - Training history and metadata

## Analysis Strategy

To generalize these implementations, we need to identify:

1. **Generic components** (applicable to any G₂ manifold):
   - Core differential geometry operators (exterior derivative, Hodge star)
   - Loss function formulations (torsion closure/coclosure)
   - Network architectures (Phi networks, harmonic networks)
   - Training curriculum strategies

2. **GIFT-specific components** (need abstraction):
   - Fixed topological parameters (b₂=21, b₃=77)
   - K₇-specific TCS structure (M₁/M₂ topology)
   - GIFT framework dependencies
   - Hardcoded calibration cycles

3. **Abstraction targets**:
   - Configurable topology (arbitrary b₂, b₃)
   - Flexible manifold construction (not just K₇)
   - Pluggable architecture for different G₂ types (Joyce, TCS, etc.)
   - Framework-agnostic implementations

## Next Steps

1. ✅ Copy reference implementations
2. 🔄 Analyze code structure and dependencies
3. ⏳ Identify generic vs. specific components
4. ⏳ Design universal API for g2-forge
5. ⏳ Refactor into modular package structure

---

**Note**: These files are for **reference only**. The goal is to extract the mathematical and algorithmic insights, not to directly port GIFT-specific code.
