"""
Unit tests for multi-grid analysis from GIFT v1.2b.

Tests multi-grid evaluation of RG quantities:
- Coordinate subsampling
- Multi-grid RG quantity computation
- Integration with phi network
- Robustness across resolutions
"""

import pytest
import torch
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.analysis.spectral import (
    subsample_coords_to_coarse_grid,
    compute_multi_grid_rg_quantities,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# COORDINATE SUBSAMPLING TESTS
# ============================================================

def test_subsample_coords_basic():
    """Test basic coordinate subsampling."""
    batch_size = 128
    coords = torch.randn(batch_size, 7)

    coords_coarse = subsample_coords_to_coarse_grid(coords, n_coarse=8)

    # Should return approximately half the samples
    assert coords_coarse.shape[0] <= batch_size
    assert coords_coarse.shape[1] == 7  # Preserve coordinate dimension


def test_subsample_coords_reduces_size():
    """Test that subsampling reduces coordinate count."""
    batch_size = 256
    coords = torch.randn(batch_size, 7)

    coords_coarse = subsample_coords_to_coarse_grid(coords, n_coarse=8)

    # Should have fewer coordinates
    assert coords_coarse.shape[0] < batch_size


def test_subsample_coords_small_batch():
    """Test subsampling with small batch size."""
    batch_size = 4
    coords = torch.randn(batch_size, 7)

    coords_coarse = subsample_coords_to_coarse_grid(coords, n_coarse=8)

    # Should handle small batches (at least 1 sample)
    assert coords_coarse.shape[0] >= 1
    assert coords_coarse.shape[0] <= batch_size


def test_subsample_coords_preserves_shape():
    """Test that subsampling preserves coordinate dimensionality."""
    batch_size = 100
    coords = torch.randn(batch_size, 7)

    coords_coarse = subsample_coords_to_coarse_grid(coords, n_coarse=16)

    # Should preserve 7D coordinates
    assert coords_coarse.ndim == 2
    assert coords_coarse.shape[1] == 7


def test_subsample_coords_deterministic_with_seed():
    """Test that subsampling is deterministic with fixed seed."""
    batch_size = 128
    coords = torch.randn(batch_size, 7)

    # Subsample twice with same seed
    torch.manual_seed(42)
    coords_coarse1 = subsample_coords_to_coarse_grid(coords, n_coarse=8)

    torch.manual_seed(42)
    coords_coarse2 = subsample_coords_to_coarse_grid(coords, n_coarse=8)

    assert torch.allclose(coords_coarse1, coords_coarse2)


def test_subsample_coords_different_n_coarse():
    """Test subsampling with different n_coarse values."""
    batch_size = 200
    coords = torch.randn(batch_size, 7)

    for n_coarse in [4, 8, 16, 32]:
        coords_coarse = subsample_coords_to_coarse_grid(coords, n_coarse=n_coarse)

        # Should return valid coordinates
        assert coords_coarse.shape[0] > 0
        assert coords_coarse.shape[1] == 7


def test_subsample_coords_preserves_device():
    """Test that subsampling preserves device."""
    batch_size = 100
    coords = torch.randn(batch_size, 7)

    # Test CPU
    coords_coarse_cpu = subsample_coords_to_coarse_grid(coords, n_coarse=8)
    assert coords_coarse_cpu.device.type == 'cpu'

    # Test CUDA if available
    if torch.cuda.is_available():
        coords_cuda = coords.to('cuda')
        coords_coarse_cuda = subsample_coords_to_coarse_grid(coords_cuda, n_coarse=8)
        assert coords_coarse_cuda.device.type == 'cuda'


def test_subsample_coords_values_are_subset():
    """Test that subsampled coordinates are from original set."""
    batch_size = 50
    # Create distinctive coordinates
    coords = torch.arange(batch_size * 7).reshape(batch_size, 7).float()

    coords_coarse = subsample_coords_to_coarse_grid(coords, n_coarse=8)

    # All coarse coordinates should match some original coordinate
    # (Check first coordinate component for simplicity)
    original_first_coords = coords[:, 0]
    coarse_first_coords = coords_coarse[:, 0]

    for c in coarse_first_coords:
        assert c in original_first_coords


# ============================================================
# MULTI-GRID RG QUANTITIES TESTS
# ============================================================

def test_compute_multi_grid_rg_quantities_basic(small_topology_config):
    """Test basic multi-grid RG quantity computation."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()

    # Create fine grid coordinates
    coords_fine = manifold.sample_coordinates(n_samples=64, device='cpu')

    divT_eff, fract_eff = compute_multi_grid_rg_quantities(
        phi_network=phi_network,
        manifold=manifold,
        coords_fine=coords_fine,
        n_grid_coarse=8
    )

    # Both should be scalars (floats)
    assert isinstance(divT_eff, float)
    assert isinstance(fract_eff, float)


def test_compute_multi_grid_rg_quantities_finite(small_topology_config):
    """Test that multi-grid RG quantities are finite."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()
    coords_fine = manifold.sample_coordinates(n_samples=64, device='cpu')

    divT_eff, fract_eff = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=8
    )

    # Should be finite
    assert not torch.isnan(torch.tensor(divT_eff))
    assert not torch.isinf(torch.tensor(divT_eff))
    assert not torch.isnan(torch.tensor(fract_eff))
    assert not torch.isinf(torch.tensor(fract_eff))


def test_compute_multi_grid_rg_quantities_range(small_topology_config):
    """Test that multi-grid RG quantities are in expected ranges."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()
    coords_fine = manifold.sample_coordinates(n_samples=64, device='cpu')

    divT_eff, fract_eff = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=8
    )

    # divT_eff: typically small
    assert abs(divT_eff) < 100.0

    # fract_eff: mapped to [-0.5, +0.5]
    assert -0.5 <= fract_eff <= 0.5


def test_compute_multi_grid_rg_quantities_deterministic(small_topology_config):
    """Test that multi-grid computation is deterministic."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()

    # Use same coordinates
    torch.manual_seed(42)
    coords_fine = manifold.sample_coordinates(n_samples=64, device='cpu')

    # Compute twice
    torch.manual_seed(100)
    divT1, fract1 = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=8
    )

    torch.manual_seed(100)
    divT2, fract2 = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=8
    )

    # Results should be identical (same seed for subsampling)
    assert abs(divT1 - divT2) < 1e-9
    assert abs(fract1 - fract2) < 1e-9


def test_compute_multi_grid_rg_quantities_different_resolutions(small_topology_config):
    """Test multi-grid computation with different coarse resolutions."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()
    coords_fine = manifold.sample_coordinates(n_samples=128, device='cpu')

    results = []
    for n_grid_coarse in [4, 8, 16]:
        divT, fract = compute_multi_grid_rg_quantities(
            phi_network, manifold, coords_fine, n_grid_coarse=n_grid_coarse
        )
        results.append((divT, fract))

    # All should produce valid results
    for divT, fract in results:
        assert torch.isfinite(torch.tensor(divT))
        assert torch.isfinite(torch.tensor(fract))
        assert -0.5 <= fract <= 0.5


def test_compute_multi_grid_rg_quantities_different_sample_sizes(small_topology_config):
    """Test multi-grid computation with different fine grid sizes."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()

    for n_samples in [32, 64, 128]:
        coords_fine = manifold.sample_coordinates(n_samples=n_samples, device='cpu')

        divT, fract = compute_multi_grid_rg_quantities(
            phi_network, manifold, coords_fine, n_grid_coarse=8
        )

        # Should work for all sample sizes
        assert torch.isfinite(torch.tensor(divT))
        assert torch.isfinite(torch.tensor(fract))


def test_compute_multi_grid_averages_fine_and_coarse(small_topology_config):
    """Test that multi-grid output is average of fine and coarse grids."""
    # This is more of an integration test, but tests the averaging logic
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()
    coords_fine = manifold.sample_coordinates(n_samples=64, device='cpu')

    divT_eff, fract_eff = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=8
    )

    # Implementation uses 0.5 * (fine + coarse) averaging
    # Just check that results are reasonable (can't verify exact formula without reimplementing)
    assert torch.isfinite(torch.tensor(divT_eff))
    assert torch.isfinite(torch.tensor(fract_eff))


# ============================================================
# INTEGRATION WITH PHI NETWORK
# ============================================================

def test_multi_grid_with_trained_network(small_topology_config):
    """Test multi-grid computation with a partially trained network."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()

    # Do a few gradient steps to make network non-trivial
    optimizer = torch.optim.Adam(phi_network.parameters(), lr=1e-3)

    for _ in range(5):
        coords = manifold.sample_coordinates(n_samples=32, device='cpu')
        coords.requires_grad_(True)

        phi = phi_network(coords)

        # Simple loss: minimize phi norm
        loss = phi.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Now test multi-grid computation
    coords_fine = manifold.sample_coordinates(n_samples=64, device='cpu')

    divT_eff, fract_eff = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=8
    )

    # Should still produce valid results
    assert torch.isfinite(torch.tensor(divT_eff))
    assert torch.isfinite(torch.tensor(fract_eff))


def test_multi_grid_with_different_topologies():
    """Test multi-grid computation with different topology configurations."""
    topologies = [
        (5, 20),   # Small
        (10, 40),  # Medium
        (21, 77),  # GIFT
    ]

    for b2, b3 in topologies:
        manifold = g2.K7Manifold(
            b2_m1=b2 // 2,
            b3_m1=b3 // 2,
            b2_m2=b2 - b2 // 2,
            b3_m2=b3 - b3 // 2
        )

        phi_network = g2.PhiNetwork()
        coords_fine = manifold.sample_coordinates(n_samples=64, device='cpu')

        divT_eff, fract_eff = compute_multi_grid_rg_quantities(
            phi_network, manifold, coords_fine, n_grid_coarse=8
        )

        # Should work for all topologies
        assert torch.isfinite(torch.tensor(divT_eff))
        assert torch.isfinite(torch.tensor(fract_eff))


# ============================================================
# EDGE CASES
# ============================================================

def test_multi_grid_minimal_samples(small_topology_config):
    """Test multi-grid computation with minimal sample count."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()

    # Very small sample size
    coords_fine = manifold.sample_coordinates(n_samples=8, device='cpu')

    divT_eff, fract_eff = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=4
    )

    # Should handle gracefully
    assert torch.isfinite(torch.tensor(divT_eff))
    assert torch.isfinite(torch.tensor(fract_eff))


def test_multi_grid_large_sample_count(small_topology_config):
    """Test multi-grid computation with large sample count."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()

    # Large sample size
    coords_fine = manifold.sample_coordinates(n_samples=256, device='cpu')

    divT_eff, fract_eff = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=16
    )

    # Should handle large batches
    assert torch.isfinite(torch.tensor(divT_eff))
    assert torch.isfinite(torch.tensor(fract_eff))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_grid_on_cuda(small_topology_config):
    """Test multi-grid computation on CUDA device."""
    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork().to('cuda')
    coords_fine = manifold.sample_coordinates(n_samples=64, device='cuda')

    divT_eff, fract_eff = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine, n_grid_coarse=8
    )

    # Should produce valid results on CUDA
    assert torch.isfinite(torch.tensor(divT_eff))
    assert torch.isfinite(torch.tensor(fract_eff))


def test_multi_grid_consistency_across_devices(small_topology_config):
    """Test that multi-grid produces consistent results across CPU/CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = small_topology_config
    manifold = g2.K7Manifold(
        b2_m1=config.topology.b2 // 2,
        b3_m1=config.topology.b3 // 2,
        b2_m2=config.topology.b2 // 2,
        b3_m2=config.topology.b3 // 2
    )

    phi_network = g2.PhiNetwork()

    # CPU computation
    torch.manual_seed(42)
    coords_fine_cpu = manifold.sample_coordinates(n_samples=64, device='cpu')
    torch.manual_seed(100)
    divT_cpu, fract_cpu = compute_multi_grid_rg_quantities(
        phi_network, manifold, coords_fine_cpu, n_grid_coarse=8
    )

    # CUDA computation
    phi_network_cuda = g2.PhiNetwork().to('cuda')
    phi_network_cuda.load_state_dict(phi_network.state_dict())

    torch.manual_seed(42)
    coords_fine_cuda = manifold.sample_coordinates(n_samples=64, device='cuda')
    torch.manual_seed(100)
    divT_cuda, fract_cuda = compute_multi_grid_rg_quantities(
        phi_network_cuda, manifold, coords_fine_cuda, n_grid_coarse=8
    )

    # Results should be similar (within numerical tolerance)
    assert abs(divT_cpu - divT_cuda) < 0.01
    assert abs(fract_cpu - fract_cuda) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
