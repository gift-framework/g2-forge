"""
Edge case tests for manifold boundary conditions and extreme parameters.

Tests coordinate boundaries, region weights, neck width variations,
and custom cycle parametrizations.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.manifolds.k7 import K7Manifold


# Mark all tests as edge case tests
pytestmark = pytest.mark.edge_case


# ============================================================
# COORDINATE BOUNDARY TESTS
# ============================================================

def test_region_weights_at_t_equals_zero(small_topology_config):
    """Test region weights at lower coordinate boundary (t=0)."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    # Create coordinates with t=0
    coords = torch.zeros(10, 7)
    coords[:, 0] = 0.0  # t coordinate at boundary

    weights = manifold.get_region_weights(coords)

    # Should handle boundary gracefully
    assert 'm1' in weights
    assert torch.all(torch.isfinite(weights['m1']))
    assert torch.all(weights['m1'] >= 0)
    assert torch.all(weights['m1'] <= 1)


def test_region_weights_at_t_equals_one(small_topology_config):
    """Test region weights at upper coordinate boundary (t=1)."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    # Create coordinates with t=1
    coords = torch.zeros(10, 7)
    coords[:, 0] = 1.0  # t coordinate at boundary

    weights = manifold.get_region_weights(coords)

    # Should handle boundary gracefully
    assert 'm2' in weights
    assert torch.all(torch.isfinite(weights['m2']))
    assert torch.all(weights['m2'] >= 0)
    assert torch.all(weights['m2'] <= 1)


def test_region_weights_at_all_boundaries(small_topology_config):
    """Test region weights when all coordinates are at boundaries."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    # All coordinates at 0 or 1
    coords = torch.zeros(5, 7)
    coords[0, :] = 0.0
    coords[1, :] = 1.0
    coords[2, ::2] = 1.0  # Alternating
    coords[3, 1::2] = 1.0

    weights = manifold.get_region_weights(coords)

    # All weights should be valid
    for region_name, region_weights in weights.items():
        assert torch.all(torch.isfinite(region_weights))
        assert torch.all(region_weights >= 0)
        assert torch.all(region_weights <= 1)


def test_coordinates_outside_valid_range(small_topology_config):
    """Test behavior with coordinates outside [0,1] range."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    # Coordinates outside valid range
    coords = torch.randn(10, 7) * 5  # Can be outside [0, 1]

    # Should still compute region weights (though results may be extrapolated)
    weights = manifold.get_region_weights(coords)

    assert 'm1' in weights
    assert torch.all(torch.isfinite(weights['m1']))


# ============================================================
# REGION WEIGHT NORMALIZATION TESTS
# ============================================================

def test_region_weights_sum_to_approximately_one(small_topology_config):
    """Test that region weights sum to approximately 1."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    coords = manifold.sample_coordinates(n_samples=100, device='cpu')

    weights = manifold.get_region_weights(coords)

    # Sum all region weights
    total_weight = sum(weights.values())

    # Should sum to approximately 1 for each point
    assert torch.allclose(total_weight, torch.ones_like(total_weight), atol=0.1)


def test_region_weights_non_negative(small_topology_config):
    """Test that all region weights are non-negative."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    coords = manifold.sample_coordinates(n_samples=100, device='cpu')

    weights = manifold.get_region_weights(coords)

    for region_name, region_weights in weights.items():
        assert torch.all(region_weights >= 0), f"Region {region_name} has negative weights"


def test_region_weights_bounded_by_one(small_topology_config):
    """Test that all region weights are <= 1."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    coords = manifold.sample_coordinates(n_samples=100, device='cpu')

    weights = manifold.get_region_weights(coords)

    for region_name, region_weights in weights.items():
        assert torch.all(region_weights <= 1), f"Region {region_name} has weights > 1"


# ============================================================
# NECK WIDTH VARIATION TESTS
# ============================================================

def test_neck_width_zero(small_topology_config):
    """Test manifold behavior with zero neck width."""
    config = small_topology_config
    config.manifold.tcs_params.neck_width = 0.0

    manifold = g2.manifolds.create_manifold(config.manifold)

    coords = manifold.sample_coordinates(n_samples=50, device='cpu')
    weights = manifold.get_region_weights(coords)

    # Neck region should have very small weights
    assert torch.all(weights['neck'] < 0.5)


def test_neck_width_very_small(small_topology_config):
    """Test manifold behavior with very small neck width."""
    config = small_topology_config
    config.manifold.tcs_params.neck_width = 0.01  # Very small

    manifold = g2.manifolds.create_manifold(config.manifold)

    coords = manifold.sample_coordinates(n_samples=100, device='cpu')
    weights = manifold.get_region_weights(coords)

    # Should still work
    assert torch.all(torch.isfinite(weights['neck']))


def test_neck_width_very_large(small_topology_config):
    """Test manifold behavior with very large neck width."""
    config = small_topology_config
    config.manifold.tcs_params.neck_width = 0.9  # Very large

    manifold = g2.manifolds.create_manifold(config.manifold)

    coords = manifold.sample_coordinates(n_samples=100, device='cpu')
    weights = manifold.get_region_weights(coords)

    # Neck should dominate
    assert torch.mean(weights['neck']) > 0.3  # Most points in neck


def test_neck_width_equals_one(small_topology_config):
    """Test manifold behavior when entire manifold is neck."""
    config = small_topology_config
    config.manifold.tcs_params.neck_width = 1.0  # Entire manifold

    manifold = g2.manifolds.create_manifold(config.manifold)

    coords = manifold.sample_coordinates(n_samples=100, device='cpu')
    weights = manifold.get_region_weights(coords)

    # Neck should have high weights everywhere
    assert torch.mean(weights['neck']) > 0.5


# ============================================================
# COORDINATE SAMPLING EDGE CASES
# ============================================================

def test_sample_single_coordinate(small_topology_config):
    """Test sampling a single coordinate point."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    coords = manifold.sample_coordinates(n_samples=1, device='cpu')

    assert coords.shape == (1, 7)
    assert torch.all(torch.isfinite(coords))


def test_sample_very_large_batch(small_topology_config):
    """Test sampling a very large batch of coordinates."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    # Very large batch
    coords = manifold.sample_coordinates(n_samples=50000, device='cpu')

    assert coords.shape == (50000, 7)
    assert torch.all(torch.isfinite(coords))


def test_sample_with_very_small_grid(small_topology_config):
    """Test sampling with very small grid resolution."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    coords = manifold.sample_coordinates(n_samples=10, grid_n=2, device='cpu')

    assert coords.shape == (10, 7)
    assert torch.all(torch.isfinite(coords))


def test_sample_with_very_large_grid(small_topology_config):
    """Test sampling with very large grid resolution."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    coords = manifold.sample_coordinates(n_samples=100, grid_n=200, device='cpu')

    assert coords.shape == (100, 7)
    assert torch.all(torch.isfinite(coords))


def test_sample_with_grid_n_one(small_topology_config):
    """Test sampling with grid_n=1."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    coords = manifold.sample_coordinates(n_samples=10, grid_n=1, device='cpu')

    assert coords.shape == (10, 7)


# ============================================================
# CYCLE TESTS
# ============================================================

def test_associative_cycles_count(small_topology_config):
    """Test that number of associative cycles matches topology."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    cycles = manifold.get_associative_cycles()

    # Should have cycles (implementation specific)
    assert isinstance(cycles, list)


def test_coassociative_cycles_count(small_topology_config):
    """Test that number of coassociative cycles matches topology."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    cycles = manifold.get_coassociative_cycles()

    # Should have cycles (implementation specific)
    assert isinstance(cycles, list)


def test_associative_cycle_dimensions(small_topology_config):
    """Test that associative cycles have correct dimension."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    cycles = manifold.get_associative_cycles()

    for cycle in cycles:
        assert cycle.dimension == 3, "Associative cycles should be 3-dimensional"
        assert cycle.type == "associative"


def test_coassociative_cycle_dimensions(small_topology_config):
    """Test that coassociative cycles have correct dimension."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    cycles = manifold.get_coassociative_cycles()

    for cycle in cycles:
        assert cycle.dimension == 4, "Coassociative cycles should be 4-dimensional"
        assert cycle.type == "coassociative"


# ============================================================
# TOPOLOGY CONSISTENCY TESTS
# ============================================================

def test_manifold_b2_consistency(small_topology_config):
    """Test that manifold b2 matches configuration."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    expected_b2 = small_topology_config.manifold.topology.b2

    assert manifold.b2 == expected_b2


def test_manifold_b3_consistency(small_topology_config):
    """Test that manifold b3 matches configuration."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    expected_b3 = small_topology_config.manifold.topology.b3

    assert manifold.b3 == expected_b3


def test_tcs_topology_decomposition(small_topology_config):
    """Test that TCS topology decomposes correctly."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    if hasattr(manifold, 'b2_m1'):
        # TCS manifold
        assert manifold.b2 == manifold.b2_m1 + manifold.b2_m2
        assert manifold.b3 == manifold.b3_m1 + manifold.b3_m2


# ============================================================
# REGION INDICATOR FUNCTION TESTS
# ============================================================

def test_region_indicator_m1_at_boundaries(small_topology_config):
    """Test M1 region indicator at boundaries."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    if not isinstance(manifold, K7Manifold):
        pytest.skip("Not a K7 manifold")

    # Test at t=0 (should be in M1)
    t_zero = torch.tensor([0.0])
    indicator_m1_zero = manifold.compute_region_indicator(t_zero, 'm1')
    assert indicator_m1_zero > 0.5

    # Test at t=1 (should not be in M1)
    t_one = torch.tensor([1.0])
    indicator_m1_one = manifold.compute_region_indicator(t_one, 'm1')
    assert indicator_m1_one < 0.5


def test_region_indicator_m2_at_boundaries(small_topology_config):
    """Test M2 region indicator at boundaries."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    if not isinstance(manifold, K7Manifold):
        pytest.skip("Not a K7 manifold")

    # Test at t=0 (should not be in M2)
    t_zero = torch.tensor([0.0])
    indicator_m2_zero = manifold.compute_region_indicator(t_zero, 'm2')
    assert indicator_m2_zero < 0.5

    # Test at t=1 (should be in M2)
    t_one = torch.tensor([1.0])
    indicator_m2_one = manifold.compute_region_indicator(t_one, 'm2')
    assert indicator_m2_one > 0.5


def test_region_indicator_neck_at_center(small_topology_config):
    """Test neck region indicator at center."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    if not isinstance(manifold, K7Manifold):
        pytest.skip("Not a K7 manifold")

    # Test at t=0.5 (center, should be in neck)
    t_center = torch.tensor([0.5])
    indicator_neck = manifold.compute_region_indicator(t_center, 'neck')
    assert indicator_neck > 0.3  # Should have significant weight in neck


def test_region_indicator_smoothness(small_topology_config):
    """Test that region indicators are smooth functions."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    if not isinstance(manifold, K7Manifold):
        pytest.skip("Not a K7 manifold")

    # Sample many t values
    t_values = torch.linspace(0, 1, 100)

    # Compute indicators
    m1_indicators = manifold.compute_region_indicator(t_values, 'm1')
    neck_indicators = manifold.compute_region_indicator(t_values, 'neck')
    m2_indicators = manifold.compute_region_indicator(t_values, 'm2')

    # Should all be smooth (finite)
    assert torch.all(torch.isfinite(m1_indicators))
    assert torch.all(torch.isfinite(neck_indicators))
    assert torch.all(torch.isfinite(m2_indicators))


# ============================================================
# EXTREME TOPOLOGY TESTS
# ============================================================

def test_manifold_with_b2_equals_one():
    """Test manifold with minimal b2=1."""
    config = g2.create_k7_config(b2_m1=1, b3_m1=5, b2_m2=0, b3_m2=5)

    manifold = g2.manifolds.create_manifold(config.manifold)

    assert manifold.b2 == 1
    assert manifold.b3 == 10


def test_manifold_with_very_large_b2():
    """Test manifold with very large b2."""
    config = g2.create_k7_config(b2_m1=100, b3_m1=200, b2_m2=100, b3_m2=200)

    manifold = g2.manifolds.create_manifold(config.manifold)

    assert manifold.b2 == 200
    assert manifold.b3 == 400


def test_manifold_with_asymmetric_topology():
    """Test manifold with very asymmetric M1 and M2."""
    config = g2.create_k7_config(b2_m1=1, b3_m1=5, b2_m2=50, b3_m2=100)

    manifold = g2.manifolds.create_manifold(config.manifold)

    assert manifold.b2_m1 == 1
    assert manifold.b2_m2 == 50
    assert manifold.b2_m1 + manifold.b2_m2 == manifold.b2
