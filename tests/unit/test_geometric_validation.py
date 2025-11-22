"""
Unit tests for geometric validation module.

Tests validators for Ricci-flatness, holonomy preservation,
and metric properties.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.validation.geometric import (
    RicciValidator,
    HolonomyTester,
    MetricValidator,
    GeometricValidator,
    ValidationResult,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# RICCI VALIDATOR TESTS
# ============================================================

def test_ricci_validator_initialization():
    """Test RicciValidator can be initialized."""
    validator = RicciValidator(n_test_points=100, tolerance=1e-4)

    assert validator.n_test_points == 100
    assert validator.tolerance == 1e-4


def test_ricci_validator_on_flat_metric():
    """Test Ricci validator on flat (identity) metric."""
    validator = RicciValidator(n_test_points=50, tolerance=1e-4)
    manifold = g2.create_gift_k7()

    # Identity metric function
    def identity_metric_fn(coords):
        batch_size = coords.shape[0]
        return torch.eye(7, device=coords.device).unsqueeze(0).repeat(batch_size, 1, 1)

    result = validator.validate(identity_metric_fn, manifold, device='cpu')

    # Should be a ValidationResult
    assert isinstance(result, ValidationResult)
    assert result.metric_value >= 0
    assert result.threshold == 1e-4


def test_ricci_validator_result_structure():
    """Test that RicciValidator returns properly structured results."""
    validator = RicciValidator(n_test_points=50)
    manifold = g2.create_gift_k7()

    def simple_metric_fn(coords):
        batch_size = coords.shape[0]
        return torch.eye(7, device=coords.device).unsqueeze(0).repeat(batch_size, 1, 1)

    result = validator.validate(simple_metric_fn, manifold)

    assert hasattr(result, 'passed')
    assert hasattr(result, 'metric_value')
    assert hasattr(result, 'threshold')
    assert hasattr(result, 'message')
    assert hasattr(result, 'details')
    assert isinstance(result.passed, bool)


# ============================================================
# HOLONOMY TESTER TESTS
# ============================================================

def test_holonomy_tester_initialization():
    """Test HolonomyTester can be initialized."""
    tester = HolonomyTester(n_loops=5, n_steps_per_loop=25, tolerance=1e-4)

    assert tester.n_loops == 5
    assert tester.n_steps_per_loop == 25
    assert tester.tolerance == 1e-4


def test_holonomy_tester_generates_closed_loops():
    """Test that HolonomyTester generates valid closed loops."""
    tester = HolonomyTester(n_loops=3, n_steps_per_loop=50)
    manifold = g2.create_gift_k7()

    loop = tester._generate_closed_loop(manifold, n_steps=50, device='cpu')

    # Check shape
    assert loop.shape == (50, 7)

    # Check that it's approximately closed (start ≈ end)
    start = loop[0]
    end = loop[-1]
    # For a circular loop, start and end should be close
    # (allowing some tolerance due to discretization)
    distance = torch.norm(end - start)
    assert distance < 0.5, f"Loop not closed: distance={distance}"


def test_holonomy_tester_with_constant_phi():
    """Test holonomy tester with constant φ network."""
    from g2forge.networks.phi_network import PhiNetwork

    tester = HolonomyTester(n_loops=3, n_steps_per_loop=20, tolerance=1e-4)
    manifold = g2.create_gift_k7()

    # Create a simple phi network
    phi_network = PhiNetwork(hidden_dims=[128, 128]).to('cpu')
    phi_network.eval()

    result = tester.test_holonomy_preservation(phi_network, manifold, device='cpu')

    # Should return a result
    assert isinstance(result, ValidationResult)
    assert result.metric_value >= 0


# ============================================================
# METRIC VALIDATOR TESTS
# ============================================================

def test_metric_validator_initialization():
    """Test MetricValidator can be initialized."""
    validator = MetricValidator(n_test_points=100)

    assert validator.n_test_points == 100


def test_metric_validator_positive_definiteness():
    """Test positive-definiteness validation."""
    validator = MetricValidator(n_test_points=50)
    manifold = g2.create_gift_k7()

    # Identity metric (always positive-definite)
    def identity_metric_fn(coords):
        batch_size = coords.shape[0]
        return torch.eye(7, device=coords.device).unsqueeze(0).repeat(batch_size, 1, 1)

    result = validator.validate_positive_definiteness(identity_metric_fn, manifold)

    assert isinstance(result, ValidationResult)
    assert result.passed, "Identity metric should be positive-definite"
    assert result.metric_value > 0, "Min eigenvalue should be positive"


def test_metric_validator_detects_negative_eigenvalues():
    """Test that validator detects non-positive-definite metrics."""
    validator = MetricValidator(n_test_points=50)
    manifold = g2.create_gift_k7()

    # Metric with negative eigenvalue
    def bad_metric_fn(coords):
        batch_size = coords.shape[0]
        metric = torch.eye(7, device=coords.device).unsqueeze(0).repeat(batch_size, 1, 1)
        metric[:, 0, 0] = -1.0  # Make first eigenvalue negative
        return metric

    result = validator.validate_positive_definiteness(bad_metric_fn, manifold)

    assert isinstance(result, ValidationResult)
    assert not result.passed, "Should fail for negative eigenvalue"
    assert result.metric_value < 0, "Min eigenvalue should be negative"


def test_metric_validator_symmetry():
    """Test symmetry validation."""
    validator = MetricValidator(n_test_points=50)
    manifold = g2.create_gift_k7()

    # Symmetric metric
    def symmetric_metric_fn(coords):
        batch_size = coords.shape[0]
        return torch.eye(7, device=coords.device).unsqueeze(0).repeat(batch_size, 1, 1)

    result = validator.validate_symmetry(symmetric_metric_fn, manifold, tolerance=1e-6)

    assert isinstance(result, ValidationResult)
    assert result.passed, "Identity metric should be symmetric"


def test_metric_validator_detects_asymmetry():
    """Test that validator detects asymmetric metrics."""
    validator = MetricValidator(n_test_points=50)
    manifold = g2.create_gift_k7()

    # Asymmetric metric
    def asymmetric_metric_fn(coords):
        batch_size = coords.shape[0]
        metric = torch.eye(7, device=coords.device).unsqueeze(0).repeat(batch_size, 1, 1)
        metric[:, 0, 1] = 1.0
        metric[:, 1, 0] = 0.0  # Breaks symmetry
        return metric

    result = validator.validate_symmetry(asymmetric_metric_fn, manifold, tolerance=1e-6)

    assert isinstance(result, ValidationResult)
    assert not result.passed, "Should fail for asymmetric metric"


def test_metric_validator_smoothness():
    """Test smoothness validation."""
    validator = MetricValidator(n_test_points=50)
    manifold = g2.create_gift_k7()

    # Constant metric (perfectly smooth)
    def constant_metric_fn(coords):
        batch_size = coords.shape[0]
        return torch.eye(7, device=coords.device).unsqueeze(0).repeat(batch_size, 1, 1)

    result = validator.validate_smoothness(constant_metric_fn, manifold)

    assert isinstance(result, ValidationResult)
    # Constant metric should be smooth (zero variation)
    assert result.metric_value >= 0


# ============================================================
# GEOMETRIC VALIDATOR (INTEGRATION) TESTS
# ============================================================

def test_geometric_validator_initialization():
    """Test GeometricValidator can be initialized."""
    validator = GeometricValidator()

    assert validator.ricci_validator is not None
    assert validator.holonomy_tester is not None
    assert validator.metric_validator is not None


def test_geometric_validator_final_validation():
    """Test complete validation suite."""
    from g2forge.networks.phi_network import PhiNetwork

    validator = GeometricValidator()
    manifold = g2.create_gift_k7()

    # Create simple models
    phi_network = PhiNetwork(hidden_dims=[128, 128]).to('cpu')
    phi_network.eval()

    def metric_fn(coords):
        batch_size = coords.shape[0]
        return torch.eye(7, device=coords.device).unsqueeze(0).repeat(batch_size, 1, 1)

    models = {
        'phi_network': phi_network,
        'metric_fn': metric_fn
    }

    results = validator.final_validation(models, manifold, device='cpu')

    # Should have multiple validation results
    assert isinstance(results, dict)
    assert 'holonomy' in results
    assert 'ricci' in results
    assert 'positive_definite' in results
    assert 'symmetry' in results
    assert 'smoothness' in results


def test_geometric_validator_save_report(tmp_path):
    """Test saving validation report to JSON."""
    validator = GeometricValidator()

    # Create dummy results
    results = {
        'test1': ValidationResult(
            passed=True,
            metric_value=1e-5,
            threshold=1e-4,
            message="Test passed",
            details={'n_points': 100}
        ),
        'test2': ValidationResult(
            passed=False,
            metric_value=2e-3,
            threshold=1e-4,
            message="Test failed",
            details={'n_points': 50}
        )
    }

    # Save report
    report_path = tmp_path / "validation_report.json"
    validator.save_validation_report(results, str(report_path))

    # Check file exists
    assert report_path.exists()

    # Load and verify
    import json
    with open(report_path, 'r') as f:
        loaded = json.load(f)

    assert 'test1' in loaded
    assert 'test2' in loaded
    assert loaded['test1']['passed'] == True
    assert loaded['test2']['passed'] == False
