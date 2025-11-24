# Requirements Guide for g2-forge

This document explains the dependency structure and installation options for the g2-forge framework.

## 📦 Requirements Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `requirements.txt` | Main entry point | General installation |
| `requirements-core.txt` | Core dependencies only | Minimal installation |
| `requirements-dev.txt` | Development tools | Contributors, researchers |
| `requirements-test.txt` | Testing suite | Running tests, CI/CD |
| `requirements-cuda.txt` | GPU acceleration | Training on NVIDIA GPUs |
| `requirements-optional.txt` | Advanced features | Specialized use cases |

---

## 🚀 Quick Start

### Basic Installation (CPU)
```bash
pip install -r requirements.txt
```

This installs core dependencies: PyTorch, NumPy, SciPy, and tqdm.

### Development Installation
```bash
# For contributors and active development
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
```

Includes visualization, profiling, code quality tools, and full test suite.

### GPU Installation (CUDA)
```bash
# For training on NVIDIA GPUs
pip install -r requirements-cuda.txt
```

**Note:** Requires NVIDIA GPU with CUDA drivers installed. Verify with:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📋 Detailed Requirements

### Core (`requirements-core.txt`)
**Essential dependencies for running g2-forge:**
- `torch>=2.0.0,<3.0.0` - PyTorch for neural networks
- `numpy>=1.24.0,<2.0.0` - Numerical computing
- `scipy>=1.10.0,<2.0.0` - Scientific computing
- `tqdm>=4.65.0,<5.0.0` - Progress bars

**Size:** ~2-3 GB (CPU version)

### Development (`requirements-dev.txt`)
**Tools for development and research:**

**Visualization:**
- matplotlib, seaborn, plotly - Various plotting tools
- tensorboard - Training visualization

**Interactive:**
- jupyter, jupyterlab - Notebook interface
- ipython, ipywidgets - Enhanced Python shell

**Code Quality:**
- black, isort - Code formatting
- flake8, pylint, mypy - Linting and type checking

**Documentation:**
- sphinx, sphinx-rtd-theme - Documentation generation

**Data Science:**
- pandas - Data analysis
- h5py - Large dataset storage

**Size:** Additional ~500 MB

### Testing (`requirements-test.txt`)
**Comprehensive testing framework:**

**Core Testing:**
- pytest - Test framework
- pytest-cov - Coverage reporting
- pytest-xdist - Parallel execution
- pytest-timeout - Timeout control
- pytest-mock - Mocking utilities

**Coverage & Reports:**
- coverage[toml] - Coverage with TOML support
- pytest-html - HTML test reports

**Performance:**
- pytest-benchmark - Benchmarking
- memory-profiler - Memory profiling

**Advanced:**
- hypothesis - Property-based testing

**Size:** Additional ~200 MB

### CUDA (`requirements-cuda.txt`)
**GPU-accelerated PyTorch:**

**CUDA 11.8 (Recommended):**
```bash
pip install -r requirements-cuda.txt
```

**CUDA 12.1 (Latest):**
Edit `requirements-cuda.txt` and uncomment CUDA 12.1 section.

**Verification:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.device_count())"
```

**Size:** ~4-5 GB (includes CUDA libraries)

### Optional (`requirements-optional.txt`)
**Advanced features for specialized use cases:**

**3D Visualization:**
- mayavi, pyvista, vtk - Complex 3D manifold visualization

**Symbolic Math:**
- sympy - Verify differential geometry identities

**Optimization:**
- optuna - Hyperparameter tuning
- wandb, mlflow - Experiment tracking

**HPC:**
- dask, ray, mpi4py - Distributed/parallel computing

**Advanced Data:**
- zarr, xarray, netCDF4 - Large-scale data handling

**ML Extras:**
- scikit-learn, tensorly, einops - Additional ML tools

**Development:**
- line-profiler, pdbpp, ipdb - Advanced debugging

**Documentation:**
- nbconvert, jupytext, rise - Notebook tools

**Size:** Varies significantly depending on what you install

---

## 🔧 Installation Scenarios

### 1. Researcher (Exploration & Analysis)
```bash
# Full development + testing environment
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
```

**What you get:**
- ✅ All core dependencies
- ✅ Jupyter notebooks
- ✅ Visualization tools
- ✅ Full test suite
- ✅ Profiling tools

### 2. Developer (Contributing Code)
```bash
# Development + testing + optional tools
pip install -r requirements-dev.txt
pip install -r requirements-test.txt

# Optional: Install pre-commit
pip install pre-commit
pre-commit install
```

**What you get:**
- ✅ Everything from Researcher
- ✅ Code quality tools (black, flake8, mypy)
- ✅ Documentation tools (Sphinx)
- ✅ Pre-commit hooks (optional)

### 3. GPU User (Training Large Models)
```bash
# CUDA-enabled PyTorch + development
pip install -r requirements-cuda.txt
pip install -r requirements-dev.txt
```

**What you get:**
- ✅ GPU-accelerated PyTorch
- ✅ Visualization & profiling
- ✅ TensorBoard for monitoring

### 4. Production/Deployment (Minimal)
```bash
# Core dependencies only
pip install -r requirements-core.txt
```

**What you get:**
- ✅ Essential libraries only
- ✅ Smallest installation (~2-3 GB)
- ✅ Suitable for Docker containers

### 5. HPC Cluster (Large-Scale)
```bash
# Core + CUDA + HPC tools
pip install -r requirements-cuda.txt
pip install -r requirements-optional.txt  # Select HPC tools
```

**What you get:**
- ✅ GPU support
- ✅ Distributed computing (dask, ray, MPI)
- ✅ Large-scale data handling

---

## 🐳 Docker Installation

For containerized deployment:

```dockerfile
# Minimal production image
FROM python:3.11-slim
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Development image
FROM python:3.11
COPY requirements-dev.txt requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt -r requirements-test.txt

# GPU image (requires NVIDIA Container Toolkit)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip
COPY requirements-cuda.txt .
RUN pip install --no-cache-dir -r requirements-cuda.txt
```

---

## 🔍 Verification

After installation, verify your setup:

```bash
# Check core imports
python -c "import torch; import numpy; import scipy; print('Core: OK')"

# Check CUDA (if installed)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check g2forge
python -c "import g2forge as g2; print(f'g2-forge: {g2.__version__}')"

# Run tests
pytest tests/unit/test_operators.py -v
```

---

## 📊 Dependency Version Philosophy

### Version Ranges
We use **flexible version ranges** (`>=X.Y.Z,<X+1.0.0`) to:
- Allow patch and minor updates
- Maintain compatibility
- Avoid breaking changes

### Update Policy
- **Patch updates** (X.Y.Z → X.Y.Z+1): Automatic
- **Minor updates** (X.Y → X.Y+1): Test before adopting
- **Major updates** (X → X+1): Careful evaluation required

### Why Wide Ranges?
Given the research nature of g2-forge:
- Allows integration with other research codebases
- Flexibility for HPC environments
- Enables experimentation with newer versions

---

## 🆘 Troubleshooting

### Issue: PyTorch Installation Fails
```bash
# Try installing from official PyTorch index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: CUDA Not Detected
```bash
# Check CUDA drivers
nvidia-smi

# Reinstall with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Test Dependencies Not Found
```bash
# Install test requirements
pip install -r requirements-test.txt

# Verify pytest-cov
python -c "import pytest_cov; print('OK')"
```

### Issue: Import Errors in Tests
```bash
# Ensure g2forge is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/g2-forge"

# Or install in development mode
pip install -e .
```

---

## 📝 Contributing

When adding new dependencies:
1. Add to appropriate requirements file
2. Use version ranges, not exact versions
3. Add comments explaining why needed
4. Update this guide
5. Test installation in clean environment

---

## 🔗 Related Files

- `setup.py` - Package configuration
- `pytest.ini` - Test configuration
- `pyproject.toml` - Project metadata (if exists)
- `.python-version` - Recommended Python version

---

## 📚 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Compatibility Matrix](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [pip Requirements Format](https://pip.pypa.io/en/stable/reference/requirements-file-format/)

---

**Last Updated:** 2025-11-24
**Maintainer:** g2-forge development team
