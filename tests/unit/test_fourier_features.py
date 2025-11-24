"""
Unit tests for FourierFeatures network component.

Tests the random Fourier feature encoding used in PhiNetwork for
multi-scale geometric representation.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

from g2forge.networks.phi_network import FourierFeatures


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# INITIALIZATION AND BASIC PROPERTIES
# ============================================================

def test_fourier_features_initialization():
    """Test FourierFeatures initializes correctly."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32, scale=1.0)

    assert ff.input_dim == 7
    assert ff.n_frequencies == 32
    assert ff.B.shape == (32, 7)


def test_fourier_features_default_parameters():
    """Test FourierFeatures with default parameters."""
    ff = FourierFeatures()

    assert ff.input_dim == 7
    assert ff.n_frequencies == 32
    assert ff.B.shape == (32, 7)


def test_fourier_features_custom_input_dim():
    """Test FourierFeatures with custom input dimension."""
    ff = FourierFeatures(input_dim=5, n_frequencies=16)

    assert ff.input_dim == 5
    assert ff.B.shape == (16, 5)


def test_fourier_features_custom_frequencies():
    """Test FourierFeatures with custom number of frequencies."""
    ff = FourierFeatures(input_dim=7, n_frequencies=64)

    assert ff.n_frequencies == 64
    assert ff.B.shape == (64, 7)


def test_fourier_features_b_matrix_is_buffer():
    """Test that B matrix is registered as a buffer (not trainable)."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    # Should be in buffers, not parameters
    assert 'B' in dict(ff.named_buffers())
    assert 'B' not in dict(ff.named_parameters())


# ============================================================
# OUTPUT DIMENSION PROPERTY
# ============================================================

def test_fourier_features_output_dim_property():
    """Test output_dim property returns 2 * n_frequencies."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    assert ff.output_dim == 64  # 2 * 32


def test_fourier_features_output_dim_various_frequencies():
    """Test output_dim for various frequency counts."""
    test_cases = [
        (16, 32),   # n_freq=16 -> output=32
        (32, 64),   # n_freq=32 -> output=64
        (64, 128),  # n_freq=64 -> output=128
        (128, 256), # n_freq=128 -> output=256
    ]

    for n_freq, expected_output in test_cases:
        ff = FourierFeatures(input_dim=7, n_frequencies=n_freq)
        assert ff.output_dim == expected_output


# ============================================================
# FORWARD PASS
# ============================================================

def test_fourier_features_forward_output_shape():
    """Test FourierFeatures forward pass produces correct output shape."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.randn(10, 7)
    features = ff(x)

    assert features.shape == (10, 64)  # (batch, 2*n_frequencies)


def test_fourier_features_forward_single_sample():
    """Test FourierFeatures with single sample."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.randn(1, 7)
    features = ff(x)

    assert features.shape == (1, 64)


def test_fourier_features_forward_large_batch():
    """Test FourierFeatures with large batch."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.randn(1000, 7)
    features = ff(x)

    assert features.shape == (1000, 64)


def test_fourier_features_forward_output_finite():
    """Test that forward pass produces finite values."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.randn(100, 7)
    features = ff(x)

    assert torch.isfinite(features).all()


def test_fourier_features_forward_no_gradients():
    """Test that FourierFeatures doesn't require gradients (fixed encoding)."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    # B matrix should not require gradients
    assert not ff.B.requires_grad


# ============================================================
# FREQUENCY SCALE PARAMETER
# ============================================================

def test_fourier_features_scale_affects_b_matrix():
    """Test that scale parameter affects B matrix magnitude."""
    torch.manual_seed(42)
    ff_small = FourierFeatures(input_dim=7, n_frequencies=32, scale=1.0)

    torch.manual_seed(42)
    ff_large = FourierFeatures(input_dim=7, n_frequencies=32, scale=10.0)

    # Larger scale should give larger frequency components
    assert ff_large.B.abs().mean() > ff_small.B.abs().mean()
    assert ff_large.B.abs().mean() == pytest.approx(10.0 * ff_small.B.abs().mean(), rel=0.2)


def test_fourier_features_scale_affects_encoding():
    """Test that scale affects the encoded features."""
    x = torch.randn(10, 7)

    torch.manual_seed(42)
    ff_small = FourierFeatures(input_dim=7, n_frequencies=32, scale=0.1)

    torch.manual_seed(42)
    ff_large = FourierFeatures(input_dim=7, n_frequencies=32, scale=10.0)

    features_small = ff_small(x)
    features_large = ff_large(x)

    # Different scales should produce different features
    assert not torch.allclose(features_small, features_large)


def test_fourier_features_very_small_scale():
    """Test FourierFeatures with very small scale."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32, scale=0.01)

    x = torch.randn(10, 7)
    features = ff(x)

    # Should still work
    assert features.shape == (10, 64)
    assert torch.isfinite(features).all()


def test_fourier_features_very_large_scale():
    """Test FourierFeatures with very large scale."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32, scale=100.0)

    x = torch.randn(10, 7)
    features = ff(x)

    # Should still work
    assert features.shape == (10, 64)
    assert torch.isfinite(features).all()


# ============================================================
# DETERMINISM AND REPRODUCIBILITY
# ============================================================

def test_fourier_features_deterministic_with_seed():
    """Test that setting random seed makes FourierFeatures deterministic."""
    torch.manual_seed(42)
    ff1 = FourierFeatures(input_dim=7, n_frequencies=32)

    torch.manual_seed(42)
    ff2 = FourierFeatures(input_dim=7, n_frequencies=32)

    # B matrices should be identical
    assert torch.allclose(ff1.B, ff2.B)


def test_fourier_features_forward_deterministic():
    """Test that forward pass is deterministic for same input."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.randn(10, 7)

    features1 = ff(x)
    features2 = ff(x)

    # Should be identical
    assert torch.allclose(features1, features2)


def test_fourier_features_different_without_seed():
    """Test that FourierFeatures instances differ without seed."""
    ff1 = FourierFeatures(input_dim=7, n_frequencies=32)
    ff2 = FourierFeatures(input_dim=7, n_frequencies=32)

    # B matrices should be different (extremely unlikely to be same)
    assert not torch.allclose(ff1.B, ff2.B)


# ============================================================
# MATHEMATICAL PROPERTIES
# ============================================================

def test_fourier_features_bounded_output():
    """Test that output is bounded (cos and sin are in [-1, 1])."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.randn(100, 7) * 10  # Large input values

    features = ff(x)

    # Each feature should be in [-1, 1]
    assert features.min() >= -1.0
    assert features.max() <= 1.0


def test_fourier_features_zero_input():
    """Test FourierFeatures with zero input."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.zeros(10, 7)
    features = ff(x)

    # cos(0) = 1, sin(0) = 0
    # First half should be all 1s, second half all 0s
    assert torch.allclose(features[:, :32], torch.ones(10, 32))
    assert torch.allclose(features[:, 32:], torch.zeros(10, 32))


def test_fourier_features_linearity_in_input():
    """Test that Fourier features capture non-linearity."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x1 = torch.randn(10, 7)
    x2 = 2 * x1  # Double input

    features1 = ff(x1)
    features2 = ff(x2)

    # Output should NOT be linear (f(2x) ≠ 2f(x))
    assert not torch.allclose(features2, 2 * features1, atol=0.1)


def test_fourier_features_translation_invariant():
    """Test that features change with input translation."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.randn(10, 7)
    shift = torch.ones(7) * 0.5

    features1 = ff(x)
    features2 = ff(x + shift)

    # Features should differ (not translation invariant)
    assert not torch.allclose(features1, features2)


# ============================================================
# INTEGRATION WITH PHI NETWORK
# ============================================================

def test_fourier_features_compatible_with_phi_network():
    """Test that FourierFeatures integrates with PhiNetwork."""
    from g2forge.networks.phi_network import PhiNetwork

    # PhiNetwork should use FourierFeatures internally
    phi_net = PhiNetwork(hidden_dims=[128, 128], n_fourier=32)

    assert hasattr(phi_net, 'fourier')
    assert isinstance(phi_net.fourier, FourierFeatures)


def test_fourier_features_output_feeds_mlp():
    """Test that FourierFeatures output is suitable for MLP input."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    x = torch.randn(10, 7)
    features = ff(x)

    # Should be able to feed to Linear layer
    linear = torch.nn.Linear(ff.output_dim, 128)
    output = linear(features)

    assert output.shape == (10, 128)


# ============================================================
# EDGE CASES
# ============================================================

def test_fourier_features_single_frequency():
    """Test FourierFeatures with single frequency."""
    ff = FourierFeatures(input_dim=7, n_frequencies=1)

    x = torch.randn(10, 7)
    features = ff(x)

    assert features.shape == (10, 2)  # cos + sin


def test_fourier_features_many_frequencies():
    """Test FourierFeatures with many frequencies."""
    ff = FourierFeatures(input_dim=7, n_frequencies=256)

    x = torch.randn(10, 7)
    features = ff(x)

    assert features.shape == (10, 512)


def test_fourier_features_different_devices():
    """Test FourierFeatures on different devices."""
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        ff = FourierFeatures(input_dim=7, n_frequencies=32).to(device)
        x = torch.randn(10, 7, device=device)

        features = ff(x)

        assert features.device.type == device
        assert features.shape == (10, 64)


# ============================================================
# FREQUENCY DIVERSITY
# ============================================================

def test_fourier_features_frequency_diversity():
    """Test that random frequencies provide diverse encodings."""
    ff = FourierFeatures(input_dim=7, n_frequencies=256)

    # B matrix should have good spread
    assert ff.B.shape == (256, 7)

    # Check that frequencies are reasonably diverse
    b_std = ff.B.std(dim=0)
    assert torch.all(b_std > 0.5)  # Should have decent variation


def test_fourier_features_captures_multiple_scales():
    """Test that features capture multiple spatial scales."""
    ff = FourierFeatures(input_dim=7, n_frequencies=64, scale=1.0)

    # Test with inputs at different scales
    x_small = torch.randn(50, 7) * 0.1   # Small scale
    x_large = torch.randn(50, 7) * 10.0  # Large scale

    features_small = ff(x_small)
    features_large = ff(x_large)

    # Both should produce valid features
    assert torch.isfinite(features_small).all()
    assert torch.isfinite(features_large).all()

    # Features should be different
    assert not torch.allclose(features_small.mean(dim=0), features_large.mean(dim=0))


# ============================================================
# PERFORMANCE
# ============================================================

def test_fourier_features_forward_is_fast():
    """Test that FourierFeatures forward pass is reasonably fast."""
    import time

    ff = FourierFeatures(input_dim=7, n_frequencies=256)
    x = torch.randn(10000, 7)

    start = time.time()
    features = ff(x)
    elapsed = time.time() - start

    # Should be fast (< 0.1s for 10k samples on CPU)
    assert elapsed < 0.5, f"FourierFeatures too slow: {elapsed:.3f}s"
    assert features.shape == (10000, 512)


# ============================================================
# COMPARISON TESTS
# ============================================================

def test_fourier_features_different_scales_produce_different_features():
    """Test that different scales lead to different feature distributions."""
    x = torch.randn(1000, 7)

    ff_s1 = FourierFeatures(input_dim=7, n_frequencies=32, scale=0.5)
    ff_s2 = FourierFeatures(input_dim=7, n_frequencies=32, scale=2.0)

    features_s1 = ff_s1(x)
    features_s2 = ff_s2(x)

    # Mean and std should differ
    assert not torch.allclose(features_s1.mean(), features_s2.mean(), atol=0.1)


def test_fourier_features_more_frequencies_capture_more_detail():
    """Test that more frequencies provide richer representation."""
    x = torch.randn(100, 7)

    ff_small = FourierFeatures(input_dim=7, n_frequencies=8)
    ff_large = FourierFeatures(input_dim=7, n_frequencies=128)

    features_small = ff_small(x)  # (100, 16)
    features_large = ff_large(x)  # (100, 256)

    # Larger should have more dimensions
    assert features_large.shape[1] > features_small.shape[1]
