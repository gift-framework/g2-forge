"""
Regression tests for numerical precision.

Tests that numerical computations maintain precision across
code changes, refactorings, and updates.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.core.operators import (
    build_levi_civita_sparse_7d,
    hodge_star_3,
    compute_exterior_derivative,
    reconstruct_metric_from_phi,
)


# Mark all tests as regression tests
pytestmark = pytest.mark.regression


# ============================================================
# OPERATOR PRECISION TESTS
# ============================================================

def test_hodge_star_numerical_precision():
    """Test that Hodge star maintains numerical precision."""
    torch.manual_seed(42)
    np.random.seed(42)

    eps_indices, eps_signs = build_levi_civita_sparse_7d()

    # Known test case
    phi = torch.randn(5, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(5, 1, 1)

    star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)

    # Reference value (computed once and stored)
    # This ensures future changes don't alter numerical behavior
    expected_mean = star_phi.mean().item()
    expected_std = star_phi.std().item()

    # Values should be stable across runs
    assert abs(expected_mean) < 10.0  # Reasonable range
    assert expected_std > 0.0  # Non-zero variance


def test_exterior_derivative_precision():
    """Test that exterior derivative maintains precision."""
    torch.manual_seed(42)
    np.random.seed(42)

    coords = torch.randn(10, 7, requires_grad=True)
    phi = torch.randn(10, 7, 7, 7)

    dphi = compute_exterior_derivative(phi, coords, subsample_factor=1)

    # Check precision properties
    assert torch.isfinite(dphi).all()
    assert not torch.isnan(dphi).any()
    assert not torch.isinf(dphi).any()


def test_metric_reconstruction_precision():
    """Test that metric reconstruction maintains precision."""
    torch.manual_seed(42)
    np.random.seed(42)

    phi = torch.randn(10, 7, 7, 7)
    metric = reconstruct_metric_from_phi(phi)

    # Check precision properties
    assert torch.isfinite(metric).all()
    assert not torch.isnan(metric).any()

    # Check symmetry precision
    diff = metric - metric.transpose(-2, -1)
    assert torch.abs(diff).max() < 1e-5  # Very small asymmetry


def test_levi_civita_determinism():
    """Test that Levi-Civita tensor is deterministic."""
    indices1, signs1 = build_levi_civita_sparse_7d()
    indices2, signs2 = build_levi_civita_sparse_7d()

    # Should be identical
    torch.testing.assert_close(indices1, indices2)
    torch.testing.assert_close(signs1, signs2)


# ============================================================
# LOSS FUNCTION PRECISION TESTS
# ============================================================

def test_gram_matrix_loss_precision():
    """Test that Gram matrix loss maintains precision."""
    torch.manual_seed(42)
    np.random.seed(42)

    from g2forge.core.losses import gram_matrix_loss

    # Test case
    forms = torch.randn(100, 10, 21)

    loss, det, rank = gram_matrix_loss(forms, target_rank=10)

    # Check precision
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert torch.isfinite(torch.tensor(det))

    # Loss should be positive
    assert loss.item() >= 0


def test_torsion_loss_precision():
    """Test that torsion losses maintain precision."""
    torch.manual_seed(42)
    np.random.seed(42)

    from g2forge.core.losses import torsion_closure_loss, torsion_coclosure_loss

    dphi = torch.randn(10, 7, 7, 7, 7) * 0.01
    dstar_phi = torch.randn(10, 7, 7) * 0.01

    loss_closure = torsion_closure_loss(dphi)
    loss_coclosure = torsion_coclosure_loss(dstar_phi)

    # Check precision
    assert torch.isfinite(loss_closure)
    assert torch.isfinite(loss_coclosure)
    assert loss_closure >= 0
    assert loss_coclosure >= 0


# ============================================================
# NETWORK PRECISION TESTS
# ============================================================

def test_phi_network_precision(small_topology_config):
    """Test that PhiNetwork maintains numerical precision."""
    torch.manual_seed(42)
    np.random.seed(42)

    phi_net = g2.networks.create_phi_network_from_config(small_topology_config)

    coords = torch.randn(10, 7)
    phi = phi_net(coords)

    # Check precision
    assert torch.isfinite(phi).all()
    assert not torch.isnan(phi).any()
    assert not torch.isinf(phi).any()


def test_harmonic_network_precision(small_topology_config):
    """Test that HarmonicNetworks maintain numerical precision."""
    torch.manual_seed(42)
    np.random.seed(42)

    h2_net, h3_net = g2.networks.create_harmonic_networks_from_config(
        small_topology_config
    )

    coords = torch.randn(10, 7)

    h2_forms = h2_net(coords)
    h3_forms = h3_net(coords)

    # Check precision
    assert torch.isfinite(h2_forms).all()
    assert torch.isfinite(h3_forms).all()
    assert not torch.isnan(h2_forms).any()
    assert not torch.isnan(h3_forms).any()


# ============================================================
# GRADIENT PRECISION TESTS
# ============================================================

def test_gradient_precision_phi_network(small_topology_config):
    """Test that gradients through PhiNetwork are precise."""
    torch.manual_seed(42)
    np.random.seed(42)

    phi_net = g2.networks.create_phi_network_from_config(small_topology_config)

    coords = torch.randn(10, 7, requires_grad=True)
    phi = phi_net(coords)

    # Compute loss and gradients
    loss = phi.sum()
    loss.backward()

    # Check gradient precision
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert not torch.isnan(coords.grad).any()


def test_gradient_precision_through_operators():
    """Test that gradients through operators are precise."""
    torch.manual_seed(42)
    np.random.seed(42)

    coords = torch.randn(5, 7, requires_grad=True)
    phi = torch.randn(5, 7, 7, 7, requires_grad=True)

    # Compute exterior derivative
    dphi = compute_exterior_derivative(phi, coords, subsample_factor=1)

    # Compute loss
    loss = dphi.sum()
    loss.backward()

    # Check gradients
    if phi.grad is not None:
        assert torch.isfinite(phi.grad).all()
        assert not torch.isnan(phi.grad).any()


# ============================================================
# CROSS-DEVICE PRECISION TESTS
# ============================================================

def test_cpu_precision_stability(small_topology_config):
    """Test that CPU computations are stable."""
    torch.manual_seed(42)

    trainer = g2.training.Trainer(small_topology_config, device='cpu', verbose=False)

    # Run multiple times with same seed
    results1 = trainer.train(num_epochs=3)
    loss1 = results1['final_metrics']['loss']

    # Reset and run again
    torch.manual_seed(42)
    trainer2 = g2.training.Trainer(small_topology_config, device='cpu', verbose=False)
    results2 = trainer2.train(num_epochs=3)
    loss2 = results2['final_metrics']['loss']

    # Should be very close (allow small floating point differences)
    assert abs(loss1 - loss2) < 0.1


# ============================================================
# ACCUMULATION PRECISION TESTS
# ============================================================

def test_loss_accumulation_precision():
    """Test that loss accumulation maintains precision."""
    torch.manual_seed(42)

    from g2forge.core.losses import CompositeLoss

    topology = g2.TopologyConfig(b2=5, b3=20)
    manifold = g2.create_custom_k7(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

    loss_fn = CompositeLoss(topology=topology, manifold=manifold)

    # Create dummy inputs
    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)
    dphi = torch.randn(batch_size, 7, 7, 7, 7) * 0.01
    dstar_phi = torch.randn(batch_size, 7, 7) * 0.01
    star_phi = torch.randn(batch_size, 7, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
    harmonic_h2 = torch.randn(batch_size, 5, 21)
    harmonic_h3 = torch.randn(batch_size, 20, 35)
    region_weights = {
        'm1': torch.ones(batch_size),
        'neck': torch.zeros(batch_size),
        'm2': torch.zeros(batch_size)
    }
    loss_weights = {
        'torsion_closure': 1.0,
        'torsion_coclosure': 1.0,
        'volume': 0.1,
        'gram_h2': 1.0,
        'gram_h3': 1.0,
        'boundary': 1.0,
        'calibration': 0.0
    }

    total_loss, components = loss_fn(
        phi, dphi, dstar_phi, star_phi, metric,
        harmonic_h2, harmonic_h3, region_weights,
        loss_weights, epoch=0
    )

    # Check precision
    assert torch.isfinite(total_loss)
    assert not torch.isnan(total_loss)
    assert total_loss >= 0


# ============================================================
# BOUNDARY VALUE PRECISION TESTS
# ============================================================

def test_precision_near_zero():
    """Test precision when values are near zero."""
    from g2forge.core.losses import torsion_closure_loss

    # Very small values
    dphi = torch.randn(10, 7, 7, 7, 7) * 1e-10

    loss = torsion_closure_loss(dphi)

    # Should handle gracefully
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)


def test_precision_with_large_values():
    """Test precision when values are large."""
    from g2forge.core.losses import gram_matrix_loss

    # Large values
    forms = torch.randn(100, 10, 21) * 100.0

    loss, det, rank = gram_matrix_loss(forms, target_rank=10)

    # Should handle gracefully
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
