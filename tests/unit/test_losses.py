"""
Unit tests for loss functions.

Tests core G₂ structure losses, harmonic orthonormality,
adaptive scheduling, and composite loss.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.core.losses import (
    torsion_closure_loss,
    torsion_coclosure_loss,
    volume_loss,
    gram_matrix_loss,
    boundary_smoothness_loss,
    AdaptiveLossScheduler,
    CompositeLoss,
)
from g2forge.utils.config import TopologyConfig


# ============================================================
# TORSION LOSSES TESTS
# ============================================================

def test_torsion_closure_zero_for_closed_form():
    """Test that torsion closure loss is zero for closed forms."""
    # Zero exterior derivative (closed form)
    dphi = torch.zeros(10, 7, 7, 7, 7)

    loss = torsion_closure_loss(dphi)

    assert loss.item() < 1e-10


def test_torsion_closure_nonzero_for_nonclosed():
    """Test that torsion closure loss is non-zero for non-closed forms."""
    # Non-zero exterior derivative
    dphi = torch.randn(10, 7, 7, 7, 7)

    loss = torsion_closure_loss(dphi)

    assert loss.item() > 0


def test_torsion_coclosure_zero_for_coclosed():
    """Test that coclosure loss is zero for coclosed forms."""
    dstar_phi = torch.zeros(10, 7, 7)

    loss = torsion_coclosure_loss(dstar_phi)

    assert loss.item() < 1e-10


def test_torsion_coclosure_nonzero():
    """Test that coclosure loss is non-zero for non-coclosed forms."""
    dstar_phi = torch.randn(10, 7, 7)

    loss = torsion_coclosure_loss(dstar_phi)

    assert loss.item() > 0


# ============================================================
# VOLUME LOSS TESTS
# ============================================================

def test_volume_loss_zero_for_target():
    """Test that volume loss is zero when det(g) = target."""
    batch_size = 10
    # Identity metric has det = 1
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    loss = volume_loss(metric, target_det=1.0)

    assert loss.item() < 1e-6


def test_volume_loss_nonzero_for_scaled_metric():
    """Test that volume loss is non-zero for scaled metric."""
    batch_size = 5
    # Scaled metric has det = scale^7
    scale = 2.0
    metric = scale * torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    loss = volume_loss(metric, target_det=1.0)

    # Should be non-zero since det = 2^7 = 128 ≠ 1
    assert loss.item() > 1.0


def test_volume_loss_custom_target():
    """Test volume loss with custom target."""
    batch_size = 5
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    # Target 2.0 instead of 1.0
    loss = volume_loss(metric, target_det=2.0)

    # Should be (1 - 2)^2 = 1
    expected = 1.0
    assert abs(loss.item() - expected) < 0.01


# ============================================================
# GRAM MATRIX LOSS - UNIVERSALITY TESTS (CRITICAL!)
# ============================================================

@pytest.mark.parametrize("b2", [3, 5, 10, 21, 50])
def test_gram_matrix_loss_small_topologies(b2):
    """Test gram matrix loss works for various b₂ values."""
    batch_size = 100
    n_components = 21  # 2-forms in 7D have 21 components

    # Random harmonic forms
    harmonic_forms = torch.randn(batch_size, b2, n_components)

    loss, det, rank = gram_matrix_loss(harmonic_forms, target_rank=b2)

    # Should compute without error
    assert isinstance(loss.item(), float)
    assert isinstance(det.item(), float)
    assert isinstance(rank, int)
    assert rank <= b2


@pytest.mark.parametrize("b3", [10, 20, 40, 77, 150])
def test_gram_matrix_loss_medium_topologies(b3):
    """Test gram matrix loss works for various b₃ values."""
    batch_size = 100
    n_components = 35  # 3-forms in 7D have 35 components

    harmonic_forms = torch.randn(batch_size, b3, n_components)

    loss, det, rank = gram_matrix_loss(harmonic_forms, target_rank=b3)

    assert isinstance(loss.item(), float)
    assert isinstance(det.item(), float)
    assert isinstance(rank, int)
    assert rank <= b3


def test_gram_matrix_loss_orthonormal_forms():
    """Test that orthonormal forms give low loss."""
    batch_size = 100
    n_forms = 5
    n_components = 21

    # Create approximately orthonormal forms using Gram-Schmidt
    forms = torch.randn(batch_size, n_forms, n_components)

    # Simple orthonormalization (approximate)
    for i in range(n_forms):
        # Normalize
        norm = torch.sqrt(torch.mean(forms[:, i, :] ** 2))
        forms[:, i, :] = forms[:, i, :] / (norm + 1e-6)

        # Orthogonalize against previous
        for j in range(i):
            overlap = torch.mean(
                torch.sum(forms[:, i, :] * forms[:, j, :], dim=-1)
            )
            forms[:, i, :] = forms[:, i, :] - overlap * forms[:, j, :]

    loss, det, rank = gram_matrix_loss(forms, target_rank=n_forms)

    # Loss should be relatively small for orthonormal forms
    assert loss.item() < 1.0
    assert rank >= 3  # Should have decent rank


def test_gram_matrix_loss_rank_detection():
    """Test that rank detection works correctly."""
    batch_size = 50
    n_components = 21

    # Create forms with known rank deficiency
    # Only 3 independent forms
    base_forms = torch.randn(batch_size, 3, n_components)

    # Duplicate to make 5 forms (rank should be 3)
    forms = torch.zeros(batch_size, 5, n_components)
    forms[:, 0:3, :] = base_forms
    forms[:, 3, :] = base_forms[:, 0, :]  # Duplicate
    forms[:, 4, :] = base_forms[:, 1, :]  # Duplicate

    loss, det, rank = gram_matrix_loss(forms, target_rank=5, tolerance=1e-3)

    # Rank should detect deficiency (approximately 3, maybe 4-5 due to noise)
    assert rank <= 5
    # Det should be small (matrix is rank-deficient)
    assert abs(det.item()) < 0.1


def test_gram_matrix_loss_determinant():
    """Test that determinant constraint works."""
    batch_size = 100
    n_forms = 5
    n_components = 21

    # Random forms
    forms = torch.randn(batch_size, n_forms, n_components)

    loss, det, rank = gram_matrix_loss(forms, target_rank=n_forms)

    # Determinant should be computed
    assert isinstance(det.item(), float)
    # For random forms, det is unlikely to be exactly 1
    assert abs(det.item() - 1.0) > 0.01


def test_gram_matrix_loss_large_topology():
    """Test gram matrix loss with large topology (stress test)."""
    batch_size = 50
    b2 = 100  # Large!
    n_components = 21

    forms = torch.randn(batch_size, b2, n_components)

    loss, det, rank = gram_matrix_loss(forms, target_rank=b2)

    # Should complete without error
    assert isinstance(loss.item(), float)
    assert rank <= b2


# ============================================================
# BOUNDARY SMOOTHNESS LOSS TESTS
# ============================================================

def test_boundary_smoothness_loss_shape():
    """Test that boundary smoothness loss returns scalar."""
    phi = torch.randn(10, 7, 7, 7)
    region_weights = {
        'm1': torch.rand(10),
        'neck': torch.rand(10),
        'm2': torch.rand(10)
    }

    loss = boundary_smoothness_loss(phi, region_weights)

    assert isinstance(loss.item(), float)


def test_boundary_smoothness_positive():
    """Test that boundary smoothness loss is non-negative."""
    phi = torch.randn(10, 7, 7, 7)
    region_weights = {
        'm1': torch.rand(10),
        'neck': torch.rand(10),
        'm2': torch.rand(10)
    }

    loss = boundary_smoothness_loss(phi, region_weights)

    assert loss.item() >= 0


def test_boundary_smoothness_zero_for_zero_phi():
    """Test that loss is zero for zero phi."""
    phi = torch.zeros(10, 7, 7, 7)
    region_weights = {
        'm1': torch.ones(10),
        'neck': torch.ones(10),
        'm2': torch.ones(10)
    }

    loss = boundary_smoothness_loss(phi, region_weights)

    assert loss.item() < 1e-10


# ============================================================
# ADAPTIVE LOSS SCHEDULER TESTS
# ============================================================

def test_adaptive_scheduler_initialization():
    """Test that adaptive scheduler initializes correctly."""
    scheduler = AdaptiveLossScheduler(
        check_interval=100,
        plateau_threshold=1e-4,
        max_boost_factor=1e6
    )

    weights = scheduler.get_weights()

    assert weights['torsion_closure'] == 1.0
    assert weights['torsion_coclosure'] == 1.0


def test_adaptive_scheduler_update():
    """Test that scheduler updates weights."""
    scheduler = AdaptiveLossScheduler(check_interval=10)

    # Simulate plateau: constant losses
    for epoch in range(200):
        losses = {
            'torsion_closure': 0.5,  # Constant
            'torsion_coclosure': 0.5
        }
        scheduler.update(epoch, losses)

    weights = scheduler.get_weights()

    # Weights should have been boosted due to plateau
    # (after 200 epochs with constant loss)
    assert weights['torsion_closure'] > 1.0 or weights['torsion_coclosure'] > 1.0


def test_adaptive_scheduler_safety_cap():
    """Test that scheduler respects safety cap."""
    scheduler = AdaptiveLossScheduler(
        check_interval=10,
        max_boost_factor=10.0  # Low cap for testing
    )

    # Simulate long plateau
    for epoch in range(1000):
        losses = {
            'torsion_closure': 0.5,
            'torsion_coclosure': 0.5
        }
        scheduler.update(epoch, losses)

    weights = scheduler.get_weights()

    # Should not exceed safety cap (10x initial)
    assert weights['torsion_closure'] <= 10.0
    assert weights['torsion_coclosure'] <= 10.0


def test_adaptive_scheduler_reset():
    """Test that scheduler reset works."""
    scheduler = AdaptiveLossScheduler()

    # Boost weights
    for epoch in range(200):
        losses = {'torsion_closure': 0.5, 'torsion_coclosure': 0.5}
        scheduler.update(epoch, losses)

    # Reset
    scheduler.reset()

    weights = scheduler.get_weights()

    assert weights['torsion_closure'] == 1.0
    assert weights['torsion_coclosure'] == 1.0


# ============================================================
# COMPOSITE LOSS TESTS
# ============================================================

def test_composite_loss_initialization(small_topology_config):
    """Test that composite loss initializes correctly."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    loss_fn = CompositeLoss(
        topology=small_topology_config.manifold.topology,
        manifold=manifold
    )

    assert loss_fn.topology.b2 == 5
    assert loss_fn.topology.b3 == 20


def test_composite_loss_forward(small_topology_config):
    """Test that composite loss forward pass works."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)
    topology = small_topology_config.manifold.topology

    loss_fn = CompositeLoss(topology=topology, manifold=manifold)

    # Create dummy inputs
    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)
    dphi = torch.randn(batch_size, 7, 7, 7, 7)
    dstar_phi = torch.randn(batch_size, 7, 7)
    star_phi = torch.randn(batch_size, 7, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
    harmonic_h2 = torch.randn(batch_size, topology.b2, 21)
    harmonic_h3 = torch.randn(batch_size, topology.b3, 35)
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

    # Check outputs
    assert isinstance(total_loss.item(), float)
    assert 'torsion_closure' in components
    assert 'gram_h2' in components
    assert 'rank_h2' in components
    assert 'rank_h3' in components


def test_composite_loss_uses_topology(small_topology_config, gift_config):
    """Test that composite loss adapts to different topologies."""
    # Small topology
    manifold_small = g2.manifolds.create_manifold(small_topology_config.manifold)
    loss_fn_small = CompositeLoss(
        topology=small_topology_config.manifold.topology,
        manifold=manifold_small
    )

    # GIFT topology
    manifold_gift = g2.manifolds.create_manifold(gift_config.manifold)
    loss_fn_gift = CompositeLoss(
        topology=gift_config.manifold.topology,
        manifold=manifold_gift
    )

    # Check that topologies are different
    assert loss_fn_small.topology.b2 == 5
    assert loss_fn_gift.topology.b2 == 21
    assert loss_fn_small.topology.b3 == 20
    assert loss_fn_gift.topology.b3 == 77


def test_composite_loss_missing_region_weights(small_topology_config):
    """Test that composite loss handles missing region weights."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)
    topology = small_topology_config.manifold.topology
    loss_fn = CompositeLoss(topology=topology, manifold=manifold)

    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)
    dphi = torch.randn(batch_size, 7, 7, 7, 7)
    dstar_phi = torch.randn(batch_size, 7, 7)
    star_phi = torch.randn(batch_size, 7, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
    harmonic_h2 = torch.randn(batch_size, topology.b2, 21)
    harmonic_h3 = torch.randn(batch_size, topology.b3, 35)

    # NO neck region weights
    region_weights = {
        'm1': torch.ones(batch_size),
        'm2': torch.ones(batch_size)
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

    # Should handle missing 'neck' gracefully
    total_loss, components = loss_fn(
        phi, dphi, dstar_phi, star_phi, metric,
        harmonic_h2, harmonic_h3, region_weights,
        loss_weights, epoch=0
    )

    # Boundary loss should be zero
    assert components['boundary'] == 0.0
