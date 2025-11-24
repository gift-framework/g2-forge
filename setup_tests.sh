#!/bin/bash
# Setup script for running the g2-forge test suite
# Ensures all test dependencies are installed

set -e  # Exit on error

echo "========================================="
echo "g2-forge Test Suite Setup"
echo "========================================="
echo

# Check if we're in the right directory
if [ ! -f "pytest.ini" ]; then
    echo "❌ ERROR: Not in g2-forge root directory!"
    echo "Please run from the project root."
    exit 1
fi

echo "✓ Found g2-forge project root"
echo

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

echo "Checking Python version..."
echo "  Current: Python $PYTHON_VERSION"
echo "  Required: Python >=$REQUIRED_VERSION"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "❌ ERROR: Python 3.10 or higher required"
    echo "Please upgrade Python or use a virtual environment"
    exit 1
fi

echo "✓ Python version OK"
echo

# Install test dependencies
echo "Installing test dependencies..."
echo "  This may take a few minutes..."
echo

if pip install -q -r requirements-test.txt; then
    echo "✓ Test dependencies installed"
else
    echo "❌ ERROR: Failed to install test dependencies"
    echo "Trying alternative installation method..."

    # Try installing core first, then test
    if pip install -q -r requirements-core.txt && \
       pip install -q pytest pytest-cov pytest-xdist pytest-timeout pytest-mock; then
        echo "✓ Core + test framework installed"
    else
        echo "❌ ERROR: Installation failed"
        echo "Please install manually:"
        echo "  pip install -r requirements-test.txt"
        exit 1
    fi
fi

echo

# Verify critical imports
echo "Verifying imports..."
VERIFICATION_FAILED=0

check_import() {
    local module=$1
    local name=$2
    if python -c "import $module" 2>/dev/null; then
        echo "  ✓ $name"
        return 0
    else
        echo "  ❌ $name"
        return 1
    fi
}

check_import "torch" "PyTorch" || VERIFICATION_FAILED=1
check_import "numpy" "NumPy" || VERIFICATION_FAILED=1
check_import "scipy" "SciPy" || VERIFICATION_FAILED=1
check_import "pytest" "pytest" || VERIFICATION_FAILED=1
check_import "pytest_cov" "pytest-cov" || VERIFICATION_FAILED=1
check_import "g2forge" "g2forge" || VERIFICATION_FAILED=1

echo

if [ $VERIFICATION_FAILED -eq 1 ]; then
    echo "⚠️  WARNING: Some imports failed"
    echo "Tests may not run correctly. Check your installation:"
    echo "  pip install -r requirements-test.txt"
    echo
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ All critical imports OK"
fi

echo

# Check CUDA availability (informational only)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo "ℹ️  CUDA Status:"
    echo "  ✓ CUDA available: $CUDA_VERSION"
    echo "  ✓ GPU count: $GPU_COUNT"
else
    echo "ℹ️  CUDA Status:"
    echo "  ○ CUDA not available (CPU-only mode)"
    echo "  For GPU support, see: requirements-cuda.txt"
fi

echo
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo
echo "You can now run tests:"
echo "  • All new tests:         ./run_new_tests.sh"
echo "  • Specific test:         pytest tests/unit/test_volume_normalizer.py -v"
echo "  • With coverage:         pytest tests/unit/ --cov=g2forge.physics"
echo "  • Parallel execution:    pytest tests/unit/ -n auto"
echo
echo "For more options, see: NEW_TESTS_SUMMARY.md"
echo
