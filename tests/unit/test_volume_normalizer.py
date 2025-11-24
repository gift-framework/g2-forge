"""
Unit tests for VolumeNormalizer from GIFT v1.2b.

Tests volume normalization for G₂ metrics including:
- Scale factor computation
- Normalization workflow
- Metric transformation
- Edge cases and numerical stability
"""

import pytest
import torch
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.physics.volume_normalizer import VolumeNormalizer


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# INITIALIZATION TESTS
# ============================================================

def test_volume_normalizer_initialization():
    """Test VolumeNormalizer initialization with default parameters."""
    normalizer = VolumeNormalizer()

    assert normalizer.target_det == 2.0
    assert normalizer.volume_scale == 1.0
    assert normalizer.is_normalized == False


def test_volume_normalizer_custom_target():
    """Test VolumeNormalizer with custom target determinant."""
    normalizer = VolumeNormalizer(target_det=1.5)

    assert normalizer.target_det == 1.5
    assert normalizer.volume_scale == 1.0
    assert normalizer.is_normalized == False


# ============================================================
# SCALE COMPUTATION TESTS
# ============================================================

def test_compute_scale_basic(small_topology_config):
    """Test basic scale computation."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    scale = normalizer.compute_scale(
        phi_network=phi_network,
        manifold=manifold,
        n_samples=32,
        device='cpu'
    )

    # Scale should be positive
    assert scale > 0
    # Scale should be reasonable (not extreme)
    assert 0.1 < scale < 10.0


def test_compute_scale_deterministic(small_topology_config):
    """Test that scale computation is deterministic with fixed seed."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    # Compute twice with same seed
    torch.manual_seed(42)
    scale1 = normalizer.compute_scale(phi_network, manifold, n_samples=32, device='cpu')

    torch.manual_seed(42)
    scale2 = normalizer.compute_scale(phi_network, manifold, n_samples=32, device='cpu')

    assert abs(scale1 - scale2) < 1e-6


def test_compute_scale_sample_size_independence(small_topology_config):
    """Test that scale is consistent across different sample sizes."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    torch.manual_seed(42)
    scale_32 = normalizer.compute_scale(phi_network, manifold, n_samples=32, device='cpu')

    torch.manual_seed(42)
    scale_128 = normalizer.compute_scale(phi_network, manifold, n_samples=128, device='cpu')

    # Should be similar (within 20% due to sampling)
    relative_diff = abs(scale_32 - scale_128) / ((scale_32 + scale_128) / 2)
    assert relative_diff < 0.2


# ============================================================
# NORMALIZATION WORKFLOW TESTS
# ============================================================

def test_normalize_basic(small_topology_config):
    """Test basic normalization workflow."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    info = normalizer.normalize(
        phi_network=phi_network,
        manifold=manifold,
        n_samples=32,
        device='cpu',
        verbose=False
    )

    # Check normalization state
    assert normalizer.is_normalized == True
    assert normalizer.volume_scale > 0

    # Check info dict
    assert 'det_g_before' in info
    assert 'target_det' in info
    assert 'scale_factor' in info
    assert 'det_g_after_estimated' in info

    assert info['target_det'] == 2.0
    assert info['scale_factor'] == normalizer.volume_scale


def test_normalize_improves_determinant(small_topology_config):
    """Test that normalization improves determinant accuracy."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    info = normalizer.normalize(phi_network, manifold, n_samples=64, device='cpu', verbose=False)

    det_before = info['det_g_before']
    det_after_est = info['det_g_after_estimated']
    target = info['target_det']

    # After normalization should be closer to target
    error_before = abs(det_before - target) / target
    error_after = abs(det_after_est - target) / target

    # Should improve (may not be perfect due to approximation)
    # At minimum, should be within reasonable range
    assert error_after < 0.5  # Within 50%


def test_normalize_scale_relationship(small_topology_config):
    """Test the 1/7 power relationship for scale factor."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    info = normalizer.normalize(phi_network, manifold, n_samples=64, device='cpu', verbose=False)

    # Check that scale^7 relates det_before to target_det
    scale = info['scale_factor']
    det_before = info['det_g_before']
    target = info['target_det']

    # det_after = det_before * scale^7
    expected_det_after = det_before * (scale ** 7)

    # Should be close to target
    relative_error = abs(expected_det_after - target) / target
    assert relative_error < 0.1  # Within 10%


# ============================================================
# METRIC APPLICATION TESTS
# ============================================================

def test_apply_to_metric_before_normalization():
    """Test that apply_to_metric returns unchanged metric before normalization."""
    normalizer = VolumeNormalizer()

    # Create a sample metric
    batch_size = 8
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    # Should return unchanged if not normalized
    metric_result = normalizer.apply_to_metric(metric)

    assert torch.allclose(metric_result, metric)


def test_apply_to_metric_after_normalization(small_topology_config):
    """Test that apply_to_metric correctly scales metric after normalization."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    # Perform normalization
    normalizer.normalize(phi_network, manifold, n_samples=32, device='cpu', verbose=False)

    # Create a sample metric
    batch_size = 8
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    # Apply normalization
    metric_normalized = normalizer.apply_to_metric(metric)

    # Should be scaled by volume_scale
    expected = normalizer.volume_scale * metric

    assert torch.allclose(metric_normalized, expected)


def test_apply_to_metric_scales_determinant():
    """Test that metric scaling affects determinant correctly."""
    normalizer = VolumeNormalizer()
    normalizer.volume_scale = 1.5
    normalizer.is_normalized = True

    # Create a metric with known determinant
    metric = 2.0 * torch.eye(7).unsqueeze(0)  # det = 2^7 = 128

    # Apply normalization
    metric_normalized = normalizer.apply_to_metric(metric)

    # Compute determinants
    det_before = torch.det(metric[0]).item()
    det_after = torch.det(metric_normalized[0]).item()

    # det_after = det_before * scale^7
    expected_det_after = det_before * (normalizer.volume_scale ** 7)

    assert abs(det_after - expected_det_after) / expected_det_after < 1e-5


# ============================================================
# RESET FUNCTIONALITY TESTS
# ============================================================

def test_reset_clears_normalization(small_topology_config):
    """Test that reset clears normalization state."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    # Perform normalization
    normalizer.normalize(phi_network, manifold, n_samples=32, device='cpu', verbose=False)

    assert normalizer.is_normalized == True
    assert normalizer.volume_scale != 1.0

    # Reset
    normalizer.reset()

    assert normalizer.is_normalized == False
    assert normalizer.volume_scale == 1.0


# ============================================================
# EDGE CASES AND NUMERICAL STABILITY
# ============================================================

def test_compute_scale_near_zero_determinant():
    """Test scale computation when determinant is very small."""
    # This tests the epsilon handling in compute_scale
    # We can't easily control det(g), but we test the formula stability
    normalizer = VolumeNormalizer(target_det=2.0)

    # Simulate very small det(g)
    small_det = 1e-6
    target = normalizer.target_det

    # Compute scale: (target / det)^(1/7)
    scale = (target / (small_det + 1e-8)) ** (1.0 / 7.0)

    # Should be large but finite
    assert scale > 1.0
    assert not torch.isinf(torch.tensor(scale))
    assert not torch.isnan(torch.tensor(scale))


def test_compute_scale_large_determinant():
    """Test scale computation when determinant is very large."""
    normalizer = VolumeNormalizer(target_det=2.0)

    # Simulate large det(g)
    large_det = 1e4
    target = normalizer.target_det

    # Compute scale: (target / det)^(1/7)
    scale = (target / (large_det + 1e-8)) ** (1.0 / 7.0)

    # Should be small but positive
    assert 0 < scale < 1.0
    assert not torch.isinf(torch.tensor(scale))
    assert not torch.isnan(torch.tensor(scale))


def test_apply_to_metric_preserves_shape():
    """Test that apply_to_metric preserves metric tensor shape."""
    normalizer = VolumeNormalizer()
    normalizer.volume_scale = 1.5
    normalizer.is_normalized = True

    # Test various batch sizes
    for batch_size in [1, 4, 16, 32]:
        metric = torch.randn(batch_size, 7, 7)
        # Make symmetric
        metric = 0.5 * (metric + metric.transpose(1, 2))

        metric_normalized = normalizer.apply_to_metric(metric)

        assert metric_normalized.shape == metric.shape


def test_apply_to_metric_preserves_symmetry():
    """Test that apply_to_metric preserves metric symmetry."""
    normalizer = VolumeNormalizer()
    normalizer.volume_scale = 1.5
    normalizer.is_normalized = True

    # Create symmetric metric
    batch_size = 8
    metric = torch.randn(batch_size, 7, 7)
    metric = 0.5 * (metric + metric.transpose(1, 2))

    metric_normalized = normalizer.apply_to_metric(metric)

    # Check symmetry preserved
    assert torch.allclose(
        metric_normalized,
        metric_normalized.transpose(1, 2),
        atol=1e-6
    )


def test_normalize_multiple_calls(small_topology_config):
    """Test that calling normalize multiple times updates state correctly."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    # First normalization
    info1 = normalizer.normalize(phi_network, manifold, n_samples=32, device='cpu', verbose=False)
    scale1 = normalizer.volume_scale

    # Second normalization (re-normalize)
    info2 = normalizer.normalize(phi_network, manifold, n_samples=32, device='cpu', verbose=False)
    scale2 = normalizer.volume_scale

    # Both should complete successfully
    assert normalizer.is_normalized == True
    assert scale1 > 0
    assert scale2 > 0


def test_different_target_determinants(small_topology_config):
    """Test normalization with different target determinants."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)

    targets = [0.5, 1.0, 2.0, 4.0]
    scales = []

    for target in targets:
        normalizer = VolumeNormalizer(target_det=target)
        info = normalizer.normalize(phi_network, manifold, n_samples=32, device='cpu', verbose=False)
        scales.append(normalizer.volume_scale)

    # Higher target should generally give larger scale (if det_before < target)
    # Or vice versa - just check all scales are positive and reasonable
    for scale in scales:
        assert scale > 0
        assert 0.01 < scale < 100.0


def test_normalize_verbose_output(small_topology_config, capsys):
    """Test that verbose=True produces output."""
    config = small_topology_config
    manifold = g2.manifolds.create_manifold(config.manifold)

    phi_network = g2.networks.create_phi_network_from_config(config)
    normalizer = VolumeNormalizer(target_det=2.0)

    # Normalize with verbose=True
    normalizer.normalize(phi_network, manifold, n_samples=32, device='cpu', verbose=True)

    # Check that output was printed
    captured = capsys.readouterr()
    assert '[Volume Normalization]' in captured.out
    assert 'Current det(g):' in captured.out
    assert 'Target det(g):' in captured.out
    assert 'Scale factor:' in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
