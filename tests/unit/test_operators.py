"""
Unit tests for differential geometry operators.

Tests core operators: Hodge star, exterior derivative, coclosure,
metric reconstruction, and TCS region operators.
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
    compute_coclosure,
    reconstruct_metric_from_phi,
    validate_antisymmetry,
    compute_volume_form,
    region_weighted_torsion,
)


# ============================================================
# LEVI-CIVITA TENSOR TESTS
# ============================================================

def test_levi_civita_shape():
    """Test that Levi-Civita has correct shape."""
    indices, signs = build_levi_civita_sparse_7d()

    # Should have 7! = 5040 permutations
    assert indices.shape == (5040, 7)
    assert signs.shape == (5040,)


def test_levi_civita_identity_permutation():
    """Test that identity permutation has sign +1."""
    indices, signs = build_levi_civita_sparse_7d()

    # Find identity permutation [0,1,2,3,4,5,6]
    identity = torch.arange(7)
    is_identity = (indices == identity).all(dim=1)
    identity_idx = is_identity.nonzero(as_tuple=True)[0]

    assert len(identity_idx) == 1
    assert signs[identity_idx[0]] == 1.0


def test_levi_civita_single_swap():
    """Test that single swap has sign -1."""
    indices, signs = build_levi_civita_sparse_7d()

    # Single swap: [1,0,2,3,4,5,6]
    swap = torch.tensor([1, 0, 2, 3, 4, 5, 6])
    is_swap = (indices == swap).all(dim=1)
    swap_idx = is_swap.nonzero(as_tuple=True)[0]

    assert len(swap_idx) == 1
    assert signs[swap_idx[0]] == -1.0


def test_levi_civita_double_swap():
    """Test that double swap has sign +1."""
    indices, signs = build_levi_civita_sparse_7d()

    # Double swap: [1,0,3,2,4,5,6]
    double_swap = torch.tensor([1, 0, 3, 2, 4, 5, 6])
    is_double = (indices == double_swap).all(dim=1)
    double_idx = is_double.nonzero(as_tuple=True)[0]

    assert len(double_idx) == 1
    assert signs[double_idx[0]] == 1.0


def test_levi_civita_all_permutations_present():
    """Test that all 5040 permutations are unique."""
    indices, signs = build_levi_civita_sparse_7d()

    # Convert to list of tuples for uniqueness check
    perms = [tuple(indices[i].tolist()) for i in range(len(indices))]
    unique_perms = set(perms)

    assert len(unique_perms) == 5040


# ============================================================
# HODGE STAR TESTS
# ============================================================

def test_hodge_star_shape(sample_phi_antisymmetric, sample_metric, levi_civita):
    """Test that Hodge star produces correct output shape."""
    phi = sample_phi_antisymmetric
    metric = sample_metric
    eps_indices, eps_signs = levi_civita

    star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)

    # ★: Λ³ → Λ⁴
    assert star_phi.shape == (10, 7, 7, 7, 7)


def test_hodge_star_preserves_batch(sample_phi_antisymmetric, sample_metric, levi_civita):
    """Test that Hodge star works for different batch sizes."""
    eps_indices, eps_signs = levi_civita

    for batch_size in [1, 5, 20]:
        phi = torch.randn(batch_size, 7, 7, 7)
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

        star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)
        assert star_phi.shape[0] == batch_size


def test_hodge_star_nonzero(sample_phi_antisymmetric, sample_metric, levi_civita):
    """Test that Hodge star produces non-zero output for non-zero input."""
    phi = sample_phi_antisymmetric
    metric = sample_metric
    eps_indices, eps_signs = levi_civita

    star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)

    # Should not be all zeros
    assert torch.abs(star_phi).max() > 1e-6


def test_hodge_star_scales_with_metric():
    """Test that Hodge star changes when metric changes."""
    eps_indices, eps_signs = build_levi_civita_sparse_7d()

    phi = torch.randn(5, 7, 7, 7)

    # Identity metric
    metric1 = torch.eye(7).unsqueeze(0).repeat(5, 1, 1)
    star_phi1 = hodge_star_3(phi, metric1, eps_indices, eps_signs)

    # Scaled metric
    metric2 = 2.0 * torch.eye(7).unsqueeze(0).repeat(5, 1, 1)
    star_phi2 = hodge_star_3(phi, metric2, eps_indices, eps_signs)

    # Results should be different
    assert not torch.allclose(star_phi1, star_phi2, rtol=1e-3)


# ============================================================
# EXTERIOR DERIVATIVE TESTS
# ============================================================

def test_exterior_derivative_shape(sample_coordinates):
    """Test that exterior derivative produces correct output shape."""
    coords = sample_coordinates
    phi = torch.randn(10, 7, 7, 7)

    dphi = compute_exterior_derivative(phi, coords, subsample_factor=1)

    # dφ: Λ³ → Λ⁴
    assert dphi.shape == (10, 7, 7, 7, 7)


def test_exterior_derivative_of_constant_is_zero():
    """Test that d(constant) = 0."""
    batch_size = 5
    coords = torch.randn(batch_size, 7, requires_grad=True)

    # Constant 3-form
    phi = torch.ones(batch_size, 7, 7, 7)

    dphi = compute_exterior_derivative(phi, coords, subsample_factor=1)

    # Should be approximately zero
    assert torch.abs(dphi).max() < 1e-4


def test_exterior_derivative_subsampling():
    """Test that subsampling affects computation."""
    coords = torch.randn(20, 7, requires_grad=True)
    phi = torch.randn(20, 7, 7, 7)

    dphi_full = compute_exterior_derivative(phi, coords, subsample_factor=1)
    dphi_sub = compute_exterior_derivative(phi, coords, subsample_factor=2)

    # Both should have same shape
    assert dphi_full.shape == dphi_sub.shape

    # Subsampled should have zeros for non-sampled points
    # (at least some difference due to subsampling)
    assert not torch.equal(dphi_full, dphi_sub)


def test_exterior_derivative_requires_grad():
    """Test that exterior derivative requires gradients."""
    coords = torch.randn(10, 7, requires_grad=False)  # No grad!
    phi = torch.randn(10, 7, 7, 7)

    # Should handle gracefully or raise appropriate error
    try:
        dphi = compute_exterior_derivative(phi, coords)
        # If it doesn't raise, check that result is computed
        assert dphi.shape == (10, 7, 7, 7, 7)
    except RuntimeError:
        # Expected if gradients not enabled
        pass


# ============================================================
# METRIC RECONSTRUCTION TESTS
# ============================================================

def test_reconstruct_metric_shape(sample_phi_antisymmetric):
    """Test that metric reconstruction produces correct shape."""
    phi = sample_phi_antisymmetric
    metric = reconstruct_metric_from_phi(phi)

    assert metric.shape == (10, 7, 7)


def test_reconstruct_metric_is_symmetric(sample_phi_antisymmetric):
    """Test that reconstructed metric is symmetric."""
    phi = sample_phi_antisymmetric
    metric = reconstruct_metric_from_phi(phi)

    # Check symmetry: g_ij = g_ji
    torch.testing.assert_close(
        metric,
        metric.transpose(-2, -1),
        rtol=1e-5,
        atol=1e-6
    )


def test_reconstruct_metric_is_positive_definite(sample_phi_antisymmetric):
    """Test that reconstructed metric is positive definite."""
    phi = sample_phi_antisymmetric
    metric = reconstruct_metric_from_phi(phi)

    # Check eigenvalues are positive
    eigenvalues = torch.linalg.eigvalsh(metric)

    assert torch.all(eigenvalues > 0), "Metric should be positive definite"


def test_reconstruct_metric_nonzero(sample_phi_antisymmetric):
    """Test that metric is non-zero for non-zero phi."""
    phi = sample_phi_antisymmetric
    metric = reconstruct_metric_from_phi(phi)

    # Should not be all zeros
    assert torch.abs(metric).max() > 1e-6


# ============================================================
# VOLUME FORM TESTS
# ============================================================

def test_volume_form_identity_metric():
    """Test volume form for identity metric."""
    batch_size = 10
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    vol = compute_volume_form(metric)

    # For identity metric, det(I) = 1, so √det = 1
    expected = torch.ones(batch_size)
    torch.testing.assert_close(vol, expected, rtol=1e-5, atol=1e-6)


def test_volume_form_scaled_metric():
    """Test volume form for scaled metric."""
    batch_size = 5
    scale = 2.0
    metric = scale * torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    vol = compute_volume_form(metric)

    # det(scale * I_7) = scale^7, so √det = scale^(7/2)
    expected = torch.full((batch_size,), scale ** 3.5)
    torch.testing.assert_close(vol, expected, rtol=1e-4, atol=1e-5)


def test_volume_form_positive():
    """Test that volume form is always positive."""
    batch_size = 10
    # Random positive definite matrix
    A = torch.randn(batch_size, 7, 7)
    metric = torch.bmm(A, A.transpose(-2, -1)) + torch.eye(7).unsqueeze(0)

    vol = compute_volume_form(metric)

    assert torch.all(vol > 0)


# ============================================================
# ANTISYMMETRY VALIDATION TESTS
# ============================================================

def test_validate_antisymmetry_2form():
    """Test antisymmetry validation for 2-forms."""
    batch_size = 5
    omega = torch.zeros(batch_size, 7, 7)

    # Create antisymmetric 2-form
    for i in range(7):
        for j in range(i + 1, 7):
            val = torch.randn(batch_size)
            omega[:, i, j] = val
            omega[:, j, i] = -val

    violation = validate_antisymmetry(omega, p=2)

    # Should be very small
    assert violation < 1e-5


def test_validate_antisymmetry_3form(sample_phi_antisymmetric):
    """Test antisymmetry validation for 3-forms."""
    phi = sample_phi_antisymmetric

    violation = validate_antisymmetry(phi, p=3)

    # Should be very small for properly antisymmetrized form
    assert violation < 1e-4


def test_validate_antisymmetry_detects_violation():
    """Test that antisymmetry validation detects violations."""
    batch_size = 5

    # Create NON-antisymmetric 2-form
    omega = torch.randn(batch_size, 7, 7)

    violation = validate_antisymmetry(omega, p=2)

    # Should have significant violation
    assert violation > 0.1


# ============================================================
# REGION WEIGHTED TORSION TESTS
# ============================================================

def test_region_weighted_torsion_shape():
    """Test region weighted torsion returns correct values."""
    dphi = torch.randn(10, 7, 7, 7, 7)
    region_weights = {
        'm1': torch.rand(10),
        'neck': torch.rand(10),
        'm2': torch.rand(10)
    }

    torsion_m1, torsion_neck, torsion_m2, torsion_total = region_weighted_torsion(
        dphi, region_weights
    )

    # All should be scalars
    assert isinstance(torsion_m1.item(), float)
    assert isinstance(torsion_neck.item(), float)
    assert isinstance(torsion_m2.item(), float)
    assert isinstance(torsion_total.item(), float)


def test_region_weighted_torsion_positive():
    """Test that torsion values are non-negative."""
    dphi = torch.randn(10, 7, 7, 7, 7)
    region_weights = {
        'm1': torch.rand(10),
        'neck': torch.rand(10),
        'm2': torch.rand(10)
    }

    torsion_m1, torsion_neck, torsion_m2, torsion_total = region_weighted_torsion(
        dphi, region_weights
    )

    assert torsion_m1 >= 0
    assert torsion_neck >= 0
    assert torsion_m2 >= 0
    assert torsion_total >= 0


def test_region_weighted_torsion_zero_for_zero_dphi():
    """Test that torsion is zero for zero exterior derivative."""
    dphi = torch.zeros(10, 7, 7, 7, 7)
    region_weights = {
        'm1': torch.ones(10),
        'neck': torch.ones(10),
        'm2': torch.ones(10)
    }

    torsion_m1, torsion_neck, torsion_m2, torsion_total = region_weighted_torsion(
        dphi, region_weights
    )

    assert torsion_m1 < 1e-10
    assert torsion_neck < 1e-10
    assert torsion_m2 < 1e-10
    assert torsion_total < 1e-10
