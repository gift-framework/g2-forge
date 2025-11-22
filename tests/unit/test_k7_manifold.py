"""
Unit tests for K7 manifold implementation.

Tests coordinate sampling, region weights, TCS construction,
and cycle generation for the K7 manifold.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.manifolds.k7 import K7Manifold


# ============================================================
# K7 MANIFOLD CREATION TESTS
# ============================================================

def test_k7_manifold_creation_gift():
    """Test that GIFT K7 manifold creates correctly."""
    k7 = g2.create_gift_k7()

    assert isinstance(k7, K7Manifold)
    assert k7.b2 == 21
    assert k7.b3 == 77
    assert k7.b2_m1 == 11
    assert k7.b3_m1 == 40
    assert k7.b2_m2 == 10
    assert k7.b3_m2 == 37


def test_k7_manifold_creation_custom():
    """Test that custom K7 manifold creates correctly."""
    k7 = g2.create_custom_k7(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    assert isinstance(k7, K7Manifold)
    assert k7.b2 == 10
    assert k7.b3 == 40
    assert k7.b2_m1 == 5
    assert k7.b3_m1 == 20


def test_k7_manifold_dimension():
    """Test that K7 manifold is 7-dimensional."""
    k7 = g2.create_gift_k7()

    assert k7.dimension == 7


def test_k7_manifold_topology_consistency():
    """Test that K7 topology is consistent with TCS construction."""
    k7 = g2.create_custom_k7(b2_m1=8, b3_m1=30, b2_m2=7, b3_m2=25)

    # b2 = b2_m1 + b2_m2
    assert k7.b2 == k7.b2_m1 + k7.b2_m2
    assert k7.b2 == 15

    # b3 = b3_m1 + b3_m2
    assert k7.b3 == k7.b3_m1 + k7.b3_m2
    assert k7.b3 == 55


# ============================================================
# COORDINATE SAMPLING TESTS
# ============================================================

def test_k7_sample_coordinates_shape():
    """Test that coordinate sampling produces correct shape."""
    k7 = g2.create_gift_k7()

    coords = k7.sample_coordinates(n_samples=100, device='cpu')

    assert coords.shape == (100, 7)


def test_k7_sample_coordinates_device():
    """Test that coordinates are on correct device."""
    k7 = g2.create_gift_k7()

    coords = k7.sample_coordinates(n_samples=50, device='cpu')

    assert coords.device.type == 'cpu'


def test_k7_sample_coordinates_ranges():
    """
    Test that coordinates are in correct ranges.

    For K7 TCS: x[0] = t ∈ [0,1], x[1:7] = θ ∈ [0,2π]
    """
    k7 = g2.create_gift_k7()

    coords = k7.sample_coordinates(n_samples=1000, device='cpu')

    # t coordinate (first) should be in [0, 1]
    t = coords[:, 0]
    assert torch.all(t >= 0.0)
    assert torch.all(t <= 1.0)

    # θ coordinates should be in [0, 2π]
    theta = coords[:, 1:]
    assert torch.all(theta >= 0.0)
    assert torch.all(theta <= 2 * np.pi)


def test_k7_sample_coordinates_uniform_distribution():
    """Test that t coordinate is approximately uniformly distributed."""
    k7 = g2.create_gift_k7()

    coords = k7.sample_coordinates(n_samples=10000, device='cpu')
    t = coords[:, 0]

    # Mean should be ~0.5 for uniform [0,1]
    mean_t = t.mean().item()
    assert abs(mean_t - 0.5) < 0.05

    # Std should be ~1/sqrt(12) ≈ 0.289 for uniform [0,1]
    std_t = t.std().item()
    expected_std = 1.0 / np.sqrt(12)
    assert abs(std_t - expected_std) < 0.05


def test_k7_sample_different_sizes():
    """Test that sampling works for different batch sizes."""
    k7 = g2.create_gift_k7()

    for n_samples in [1, 10, 100, 1000]:
        coords = k7.sample_coordinates(n_samples=n_samples, device='cpu')
        assert coords.shape == (n_samples, 7)


# ============================================================
# REGION WEIGHTS TESTS (CRITICAL FOR TCS!)
# ============================================================

def test_k7_region_weights_keys():
    """Test that region weights contain M1, Neck, M2."""
    k7 = g2.create_gift_k7()

    coords = k7.sample_coordinates(n_samples=100, device='cpu')
    weights = k7.get_region_weights(coords)

    assert 'm1' in weights
    assert 'neck' in weights
    assert 'm2' in weights


def test_k7_region_weights_shapes():
    """Test that region weights have correct shapes."""
    k7 = g2.create_gift_k7()

    coords = k7.sample_coordinates(n_samples=100, device='cpu')
    weights = k7.get_region_weights(coords)

    assert weights['m1'].shape == (100,)
    assert weights['neck'].shape == (100,)
    assert weights['m2'].shape == (100,)


def test_k7_region_weights_sum_to_one():
    """
    CRITICAL TEST: Region weights should approximately sum to 1.

    This ensures proper partitioning of the manifold.
    """
    k7 = g2.create_gift_k7()

    coords = k7.sample_coordinates(n_samples=1000, device='cpu')
    weights = k7.get_region_weights(coords)

    total = weights['m1'] + weights['neck'] + weights['m2']

    # Should sum to approximately 1 for all points
    torch.testing.assert_close(
        total,
        torch.ones(1000),
        rtol=1e-4,
        atol=1e-5
    )


def test_k7_region_weights_range():
    """Test that region weights are in [0, 1]."""
    k7 = g2.create_gift_k7()

    coords = k7.sample_coordinates(n_samples=1000, device='cpu')
    weights = k7.get_region_weights(coords)

    for region in ['m1', 'neck', 'm2']:
        assert torch.all(weights[region] >= 0.0)
        assert torch.all(weights[region] <= 1.0)


def test_k7_region_weights_m1_near_zero():
    """Test that M1 region weight is high near t=0."""
    k7 = g2.create_gift_k7()

    # Create coordinates near t=0
    coords = torch.zeros(100, 7)
    coords[:, 0] = 0.01  # Very small t
    coords[:, 1:] = torch.rand(100, 6) * 2 * np.pi

    weights = k7.get_region_weights(coords)

    # M1 weight should be dominant
    assert weights['m1'].mean() > 0.8


def test_k7_region_weights_m2_near_one():
    """Test that M2 region weight is high near t=1."""
    k7 = g2.create_gift_k7()

    # Create coordinates near t=1
    coords = torch.zeros(100, 7)
    coords[:, 0] = 0.99  # Very large t
    coords[:, 1:] = torch.rand(100, 6) * 2 * np.pi

    weights = k7.get_region_weights(coords)

    # M2 weight should be dominant
    assert weights['m2'].mean() > 0.8


def test_k7_region_weights_neck_at_center():
    """Test that neck weight is high near t=0.5 (center)."""
    k7 = g2.create_gift_k7()

    # Create coordinates at center
    coords = torch.zeros(100, 7)
    coords[:, 0] = 0.5  # Center
    coords[:, 1:] = torch.rand(100, 6) * 2 * np.pi

    weights = k7.get_region_weights(coords)

    # Neck weight should be significant at center
    # (actual value depends on neck_width parameter)
    assert weights['neck'].mean() > 0.2


def test_k7_region_weights_smoothness():
    """Test that region weights vary smoothly along t."""
    k7 = g2.create_gift_k7()

    # Sample along t axis
    n_points = 100
    coords = torch.zeros(n_points, 7)
    coords[:, 0] = torch.linspace(0, 1, n_points)
    coords[:, 1:] = torch.rand(n_points, 6) * 2 * np.pi

    weights = k7.get_region_weights(coords)

    # M1 should decrease monotonically
    m1_values = weights['m1']
    m1_diffs = m1_values[1:] - m1_values[:-1]
    assert torch.all(m1_diffs <= 0.01)  # Mostly decreasing

    # M2 should increase monotonically
    m2_values = weights['m2']
    m2_diffs = m2_values[1:] - m2_values[:-1]
    assert torch.all(m2_diffs >= -0.01)  # Mostly increasing


# ============================================================
# CYCLE TESTS
# ============================================================

def test_k7_associative_cycles():
    """Test that K7 provides associative cycles."""
    k7 = g2.create_gift_k7()

    cycles = k7.get_associative_cycles()

    # Should have some cycles
    assert len(cycles) > 0

    # Each cycle should be dimension 3 (associative)
    for cycle in cycles:
        assert cycle.dimension == 3
        assert cycle.type == 'associative'


def test_k7_coassociative_cycles():
    """Test that K7 provides coassociative cycles."""
    k7 = g2.create_gift_k7()

    cycles = k7.get_coassociative_cycles()

    # Should have some cycles
    assert len(cycles) > 0

    # Each cycle should be dimension 4 (coassociative)
    for cycle in cycles:
        assert cycle.dimension == 4
        assert cycle.type == 'coassociative'


def test_k7_cycles_have_indices():
    """Test that cycles have valid index specifications."""
    k7 = g2.create_gift_k7()

    assoc_cycles = k7.get_associative_cycles()
    coassoc_cycles = k7.get_coassociative_cycles()

    for cycle in assoc_cycles:
        assert hasattr(cycle, 'indices')
        assert len(cycle.indices) >= 3  # At least 3 for 3-cycle

    for cycle in coassoc_cycles:
        assert hasattr(cycle, 'indices')
        assert len(cycle.indices) >= 4  # At least 4 for 4-cycle


# ============================================================
# STRING REPRESENTATION TESTS
# ============================================================

def test_k7_repr():
    """Test that K7 has informative string representation."""
    k7 = g2.create_custom_k7(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    repr_str = repr(k7)

    # Should contain key information
    assert 'K7' in repr_str
    assert '10' in repr_str  # b2
    assert '40' in repr_str  # b3


def test_k7_str():
    """Test that K7 has readable string conversion."""
    k7 = g2.create_gift_k7()

    str_repr = str(k7)

    # Should be informative
    assert len(str_repr) > 20
    assert 'K7' in str_repr or 'manifold' in str_repr.lower()


# ============================================================
# INTEGRATION TESTS
# ============================================================

def test_k7_full_workflow():
    """Test complete K7 workflow: creation → sampling → regions."""
    # Create manifold
    k7 = g2.create_custom_k7(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    # Sample coordinates
    coords = k7.sample_coordinates(n_samples=500, device='cpu')

    # Get region weights
    weights = k7.get_region_weights(coords)

    # Verify consistency
    assert coords.shape == (500, 7)
    assert weights['m1'].shape == (500,)
    total = weights['m1'] + weights['neck'] + weights['m2']
    torch.testing.assert_close(total, torch.ones(500), rtol=1e-4, atol=1e-5)


def test_k7_different_topologies_same_interface():
    """
    Test that different K7 topologies use same interface.

    This validates universality: same code, different manifolds.
    """
    topologies = [
        (5, 20, 5, 20),      # Small
        (11, 40, 10, 37),    # GIFT
        (15, 50, 15, 50),    # Large
    ]

    for b2_m1, b3_m1, b2_m2, b3_m2 in topologies:
        k7 = g2.create_custom_k7(
            b2_m1=b2_m1, b3_m1=b3_m1,
            b2_m2=b2_m2, b3_m2=b3_m2
        )

        # All should work with same interface
        coords = k7.sample_coordinates(n_samples=100, device='cpu')
        weights = k7.get_region_weights(coords)
        assoc = k7.get_associative_cycles()
        coassoc = k7.get_coassociative_cycles()

        # Basic checks
        assert coords.shape == (100, 7)
        assert 'm1' in weights
        assert len(assoc) > 0
        assert len(coassoc) > 0
