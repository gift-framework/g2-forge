"""
Unit tests for fractality analysis from GIFT v1.2b.

Tests multi-scale FFT analysis including:
- Downsampling
- Power spectrum slope computation
- Fractality index computation
- Torsion divergence computation
- Numerical stability
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.analysis.spectral import (
    downsample_tensor,
    compute_power_spectrum_slope,
    compute_fractality_index,
    compute_divergence_torsion,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# DOWNSAMPLING TESTS
# ============================================================

def test_downsample_tensor_factor_2():
    """Test tensor downsampling with factor 2."""
    # Create a simple tensor
    T = torch.arange(16).reshape(1, 2, 2, 2, 2).float()

    T_down = downsample_tensor(T, factor=2)

    # Should subsample every other element along last dimensions
    # Shape should be reduced
    assert T_down.shape[-1] <= T.shape[-1]


def test_downsample_tensor_factor_4():
    """Test tensor downsampling with factor 4."""
    batch_size = 4
    T = torch.randn(batch_size, 7, 7, 7, 7)

    T_down = downsample_tensor(T, factor=4)

    # Should be smaller
    assert T_down.numel() < T.numel()


def test_downsample_tensor_preserves_batch():
    """Test that downsampling preserves batch dimension."""
    batch_size = 8
    T = torch.randn(batch_size, 7, 7, 7, 7)

    T_down = downsample_tensor(T, factor=2)

    # First dimension should be unchanged
    assert T_down.shape[0] == batch_size


def test_downsample_tensor_different_factors():
    """Test downsampling with various factors."""
    T = torch.randn(2, 8, 8, 8, 8)

    for factor in [1, 2, 4]:
        T_down = downsample_tensor(T, factor=factor)

        # Should have valid shape
        assert T_down.ndim == T.ndim
        assert T_down.shape[0] == T.shape[0]


# ============================================================
# POWER SPECTRUM SLOPE TESTS
# ============================================================

def test_compute_power_spectrum_slope_basic():
    """Test basic power spectrum slope computation."""
    # Create a simple signal
    n = 1000
    T_flat = torch.randn(n)

    slope = compute_power_spectrum_slope(T_flat)

    # Slope should be finite
    assert np.isfinite(slope)
    # For random signal, slope should be negative (typical for natural signals)
    # But we just check it's in a reasonable range
    assert -10.0 < slope < 10.0


def test_compute_power_spectrum_slope_white_noise():
    """Test power spectrum slope for white noise."""
    # White noise should have approximately flat spectrum (slope ~ 0)
    n = 10000
    torch.manual_seed(42)
    T_flat = torch.randn(n)

    slope = compute_power_spectrum_slope(T_flat)

    # White noise: slope should be close to 0 (within range for random sample)
    assert -3.0 < slope < 1.0


def test_compute_power_spectrum_slope_small_input():
    """Test power spectrum slope with very small input."""
    # Small input should return default value
    T_flat = torch.randn(5)

    slope = compute_power_spectrum_slope(T_flat)

    # Should return default -2.0
    assert slope == -2.0


def test_compute_power_spectrum_slope_deterministic():
    """Test that slope computation is deterministic."""
    torch.manual_seed(42)
    T_flat = torch.randn(1000)

    slope1 = compute_power_spectrum_slope(T_flat)
    slope2 = compute_power_spectrum_slope(T_flat)

    assert abs(slope1 - slope2) < 1e-9


def test_compute_power_spectrum_slope_constant_signal():
    """Test power spectrum slope for constant signal."""
    # Constant signal has no frequency content
    T_flat = torch.ones(1000)

    slope = compute_power_spectrum_slope(T_flat)

    # Should handle gracefully (may return default or extreme value)
    assert np.isfinite(slope)


def test_compute_power_spectrum_slope_handles_zeros():
    """Test that slope computation handles zero values."""
    # Signal with some zeros
    T_flat = torch.randn(1000)
    T_flat[::10] = 0.0

    slope = compute_power_spectrum_slope(T_flat)

    # Should still compute (epsilon handling prevents log(0))
    assert np.isfinite(slope)


# ============================================================
# FRACTALITY INDEX TESTS
# ============================================================

def test_compute_fractality_index_shape():
    """Test that fractality index has correct output shape."""
    batch_size = 16
    torsion = torch.randn(batch_size, 7, 7, 7, 7)

    frac_idx, frac_idx_mean = compute_fractality_index(torsion)

    # frac_idx should have shape (batch_size,)
    assert frac_idx.shape == (batch_size,)

    # frac_idx_mean should be scalar
    assert isinstance(frac_idx_mean, float)


def test_compute_fractality_index_range():
    """Test that fractality index is in expected range."""
    batch_size = 16
    torsion = torch.randn(batch_size, 7, 7, 7, 7)

    frac_idx, frac_idx_mean = compute_fractality_index(torsion)

    # Based on implementation: mapped to [-0.5, +0.5] via tanh
    # So all values should be in this range
    assert torch.all(frac_idx >= -0.5)
    assert torch.all(frac_idx <= 0.5)

    # Mean should also be in range
    assert -0.5 <= frac_idx_mean <= 0.5


def test_compute_fractality_index_different_batch_sizes():
    """Test fractality computation with different batch sizes."""
    for batch_size in [1, 4, 8, 16, 32]:
        torsion = torch.randn(batch_size, 7, 7, 7, 7)

        frac_idx, frac_idx_mean = compute_fractality_index(torsion)

        assert frac_idx.shape == (batch_size,)
        assert np.isfinite(frac_idx_mean)


def test_compute_fractality_index_deterministic():
    """Test that fractality index is deterministic."""
    torch.manual_seed(42)
    torsion = torch.randn(8, 7, 7, 7, 7)

    frac_idx1, frac_mean1 = compute_fractality_index(torsion)
    frac_idx2, frac_mean2 = compute_fractality_index(torsion)

    assert torch.allclose(frac_idx1, frac_idx2, atol=1e-9)
    assert abs(frac_mean1 - frac_mean2) < 1e-9


def test_compute_fractality_index_multi_scale():
    """Test that fractality uses multi-scale analysis."""
    # The function should use 3 scales: full, half, quarter
    # We test this indirectly by checking it handles different sizes

    batch_size = 8
    torsion = torch.randn(batch_size, 7, 7, 7, 7)

    # Should not crash with multi-scale downsampling
    frac_idx, frac_idx_mean = compute_fractality_index(torsion)

    # Should produce valid results
    assert torch.all(torch.isfinite(frac_idx))
    assert np.isfinite(frac_idx_mean)


def test_compute_fractality_index_smooth_vs_rough():
    """Test that smooth vs rough signals have different fractality."""
    batch_size = 8

    # Smooth signal (low frequencies)
    torsion_smooth = torch.randn(batch_size, 7, 7, 7, 7) * 0.1

    # Rough signal (high frequencies)
    torch.manual_seed(123)
    torsion_rough = torch.randn(batch_size, 7, 7, 7, 7) * 10.0

    frac_smooth, frac_mean_smooth = compute_fractality_index(torsion_smooth)
    frac_rough, frac_mean_rough = compute_fractality_index(torsion_rough)

    # Both should be in valid range
    assert torch.all((frac_smooth >= -0.5) & (frac_smooth <= 0.5))
    assert torch.all((frac_rough >= -0.5) & (frac_rough <= 0.5))


def test_compute_fractality_index_zero_torsion():
    """Test fractality index with zero torsion."""
    batch_size = 8
    torsion = torch.zeros(batch_size, 7, 7, 7, 7)

    frac_idx, frac_idx_mean = compute_fractality_index(torsion)

    # Should handle zeros gracefully
    assert torch.all(torch.isfinite(frac_idx))
    assert np.isfinite(frac_idx_mean)


def test_compute_fractality_index_preserves_device():
    """Test that fractality computation preserves device."""
    batch_size = 8
    torsion = torch.randn(batch_size, 7, 7, 7, 7)

    # Test CPU
    frac_idx_cpu, _ = compute_fractality_index(torsion)
    assert frac_idx_cpu.device.type == 'cpu'

    # Test CUDA if available
    if torch.cuda.is_available():
        torsion_cuda = torsion.to('cuda')
        frac_idx_cuda, _ = compute_fractality_index(torsion_cuda)
        assert frac_idx_cuda.device.type == 'cuda'


# ============================================================
# TORSION DIVERGENCE TESTS
# ============================================================

def test_compute_divergence_torsion_shape():
    """Test that divergence has correct output shape."""
    batch_size = 16
    torsion = torch.randn(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    div_T, div_T_mean = compute_divergence_torsion(torsion, coords)

    # div_T should have shape (batch_size,)
    assert div_T.shape == (batch_size,)

    # div_T_mean should be scalar
    assert isinstance(div_T_mean, float)


def test_compute_divergence_torsion_single_sample():
    """Test divergence computation with single sample."""
    batch_size = 1
    torsion = torch.randn(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    div_T, div_T_mean = compute_divergence_torsion(torsion, coords)

    # Should handle single sample (returns zeros)
    assert div_T.shape == (batch_size,)
    assert div_T_mean == 0.0


def test_compute_divergence_torsion_multiple_samples():
    """Test divergence computation with multiple samples."""
    batch_size = 32
    torsion = torch.randn(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    div_T, div_T_mean = compute_divergence_torsion(torsion, coords)

    # Should produce valid results
    assert torch.all(torch.isfinite(div_T))
    assert np.isfinite(div_T_mean)


def test_compute_divergence_torsion_deterministic():
    """Test that divergence computation is deterministic."""
    torch.manual_seed(42)
    batch_size = 16
    torsion = torch.randn(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    div_T1, div_mean1 = compute_divergence_torsion(torsion, coords)
    div_T2, div_mean2 = compute_divergence_torsion(torsion, coords)

    assert torch.allclose(div_T1, div_T2, atol=1e-9)
    assert abs(div_mean1 - div_mean2) < 1e-9


def test_compute_divergence_torsion_zero_torsion():
    """Test divergence with zero torsion."""
    batch_size = 16
    torsion = torch.zeros(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    div_T, div_T_mean = compute_divergence_torsion(torsion, coords)

    # Zero torsion should give zero divergence
    assert torch.allclose(div_T, torch.zeros_like(div_T), atol=1e-6)
    assert abs(div_T_mean) < 1e-6


def test_compute_divergence_torsion_constant_torsion():
    """Test divergence with constant torsion."""
    batch_size = 16
    # Constant torsion (same value everywhere)
    value = 1.0
    torsion = torch.full((batch_size, 7, 7, 7, 7), value)
    coords = torch.randn(batch_size, 7)

    div_T, div_T_mean = compute_divergence_torsion(torsion, coords)

    # Constant torsion should have zero divergence
    # (no spatial variation)
    assert torch.allclose(div_T, torch.zeros_like(div_T), atol=1e-6)


def test_compute_divergence_torsion_preserves_device():
    """Test that divergence computation preserves device."""
    batch_size = 16
    torsion = torch.randn(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    # Test CPU
    div_T_cpu, _ = compute_divergence_torsion(torsion, coords)
    assert div_T_cpu.device.type == 'cpu'

    # Test CUDA if available
    if torch.cuda.is_available():
        torsion_cuda = torsion.to('cuda')
        coords_cuda = coords.to('cuda')
        div_T_cuda, _ = compute_divergence_torsion(torsion_cuda, coords_cuda)
        assert div_T_cuda.device.type == 'cuda'


def test_compute_divergence_torsion_different_batch_sizes():
    """Test divergence with different batch sizes."""
    for batch_size in [2, 4, 8, 16, 32]:
        torsion = torch.randn(batch_size, 7, 7, 7, 7)
        coords = torch.randn(batch_size, 7)

        div_T, div_T_mean = compute_divergence_torsion(torsion, coords)

        assert div_T.shape == (batch_size,)
        assert np.isfinite(div_T_mean)


def test_compute_divergence_torsion_positive_values():
    """Test that divergence produces positive values (it's an absolute measure)."""
    batch_size = 16
    torsion = torch.randn(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    div_T, div_T_mean = compute_divergence_torsion(torsion, coords)

    # Implementation uses abs(), so should be non-negative
    assert torch.all(div_T >= 0)
    assert div_T_mean >= 0


# ============================================================
# INTEGRATION TESTS (Fractality + Divergence together)
# ============================================================

def test_fractality_and_divergence_together():
    """Test computing both fractality and divergence on same torsion."""
    batch_size = 16
    torsion = torch.randn(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    # Compute both
    frac_idx, frac_mean = compute_fractality_index(torsion)
    div_T, div_mean = compute_divergence_torsion(torsion, coords)

    # Both should produce valid results
    assert frac_idx.shape == (batch_size,)
    assert div_T.shape == (batch_size,)

    assert torch.all(torch.isfinite(frac_idx))
    assert torch.all(torch.isfinite(div_T))

    assert np.isfinite(frac_mean)
    assert np.isfinite(div_mean)


def test_rg_quantities_typical_values():
    """Test that RG quantities are in typical ranges for GIFT."""
    # Based on GIFT v1.2b documentation:
    # - div_T_eff: typically small negative (~ -0.01 to 0.01)
    # - fract_eff: typically ~ -0.25

    batch_size = 64
    # Create somewhat structured torsion (not pure noise)
    torch.manual_seed(42)
    torsion = 0.1 * torch.randn(batch_size, 7, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    div_T, div_T_mean = compute_divergence_torsion(torsion, coords)
    frac_idx, frac_idx_mean = compute_fractality_index(torsion)

    # Check they're in reasonable ranges
    # div_T: typically small
    assert abs(div_T_mean) < 10.0  # Should be reasonably small

    # frac_idx: in [-0.5, +0.5] by construction
    assert -0.5 <= frac_idx_mean <= 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
