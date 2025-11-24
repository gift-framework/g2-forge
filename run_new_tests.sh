#!/bin/bash
# Script to run all new physics and multi-grid analysis tests

echo "========================================="
echo "Running New Test Suite for GIFT v1.2b"
echo "========================================="
echo

# Install dependencies if needed
echo "Checking dependencies..."
python -c "import torch" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install torch numpy scipy tqdm pytest pytest-cov
}

echo
echo "========================================="
echo "Unit Tests - VolumeNormalizer"
echo "========================================="
pytest tests/unit/test_volume_normalizer.py -v --tb=short

echo
echo "========================================="
echo "Unit Tests - RGFlowModule"
echo "========================================="
pytest tests/unit/test_rg_flow.py -v --tb=short

echo
echo "========================================="
echo "Unit Tests - Fractality Analysis"
echo "========================================="
pytest tests/unit/test_fractality_analysis.py -v --tb=short

echo
echo "========================================="
echo "Unit Tests - Multi-Grid Analysis"
echo "========================================="
pytest tests/unit/test_multi_grid_analysis.py -v --tb=short

echo
echo "========================================="
echo "Integration Tests - Physics Modules"
echo "========================================="
pytest tests/integration/test_physics_integration.py -v --tb=short

echo
echo "========================================="
echo "Test Summary"
echo "========================================="
pytest tests/unit/test_volume_normalizer.py \
       tests/unit/test_rg_flow.py \
       tests/unit/test_fractality_analysis.py \
       tests/unit/test_multi_grid_analysis.py \
       tests/integration/test_physics_integration.py \
       --tb=line --co

echo
echo "========================================="
echo "Coverage Report"
echo "========================================="
pytest tests/unit/test_volume_normalizer.py \
       tests/unit/test_rg_flow.py \
       tests/unit/test_fractality_analysis.py \
       tests/unit/test_multi_grid_analysis.py \
       tests/integration/test_physics_integration.py \
       --cov=g2forge.physics \
       --cov=g2forge.analysis.spectral \
       --cov-report=term-missing \
       --cov-report=html

echo
echo "Done! Coverage report available at htmlcov/index.html"
