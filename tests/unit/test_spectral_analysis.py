"""
Unit tests for spectral analysis tools.

Tests Laplacian spectrum, harmonic form extraction,
and cohomology verification.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.analysis.spectral import (
    compute_laplacian_spectrum,
    extract_harmonic_forms,
    compute_gram_matrix,
    compute_rank,
    analyze_spectral_gap,
    verify_cohomology_ranks,
    compute_harmonic_penalty,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# GRAM MATRIX TESTS
# ============================================================

def test_compute_gram_matrix_orthonormal():
    """Test Gram matrix for orthonormal set."""
    n_samples = 100
    n_forms = 3
    n_components = 10

    # Create orthonormal forms (simplified)
    forms = torch.randn(n_samples, n_forms, n_components)

    # Normalize each form
    for i in range(n_forms):
        norm = torch.norm(forms[:, i, :], dim=1, keepdim=True)
        forms[:, i, :] = forms[:, i, :] / (norm + 1e-8)

    gram = compute_gram_matrix(forms)

    # Shape should be [n_forms, n_forms]
    assert gram.shape == (n_forms, n_forms)

    # Should be symmetric
    assert torch.allclose(gram, gram.T, atol=1e-3)


def test_compute_gram_matrix_identity_forms():
    """Test Gram matrix for identical forms."""
    n_samples = 50
    n_forms = 2
    n_components = 5

    # Create two identical forms
    form = torch.randn(n_samples, n_components)
    forms = torch.stack([form, form], dim=1)  # [n_samples, 2, n_components]

    gram = compute_gram_matrix(forms)

    # Diagonal should be equal
    assert torch.allclose(gram[0, 0], gram[1, 1], atol=1e-4)

    # Off-diagonal should equal diagonal (identical forms)
    assert torch.allclose(gram[0, 1], gram[0, 0], atol=1e-4)


# ============================================================
# RANK COMPUTATION TESTS
# ============================================================

def test_compute_rank_full_rank():
    """Test rank computation for full-rank matrix."""
    matrix = torch.eye(5)

    rank = compute_rank(matrix, tolerance=1e-6)

    assert rank == 5


def test_compute_rank_deficient():
    """Test rank computation for rank-deficient matrix."""
    # Create rank-2 matrix
    matrix = torch.zeros(5, 5)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    # Rest are zero

    rank = compute_rank(matrix, tolerance=1e-6)

    assert rank == 2


def test_compute_rank_zero_matrix():
    """Test rank computation for zero matrix."""
    matrix = torch.zeros(4, 4)

    rank = compute_rank(matrix, tolerance=1e-6)

    assert rank == 0


# ============================================================
# SPECTRAL GAP ANALYSIS TESTS
# ============================================================

def test_analyze_spectral_gap_with_zeros():
    """Test spectral gap analysis with zero modes."""
    # Create eigenvalues: 3 near-zero, then a gap, then larger values
    eigenvalues = torch.tensor([1e-8, 2e-8, 3e-8, 0.5, 0.7, 1.0])

    result = analyze_spectral_gap(eigenvalues, expected_zero_modes=3)

    assert 'spectral_gap' in result
    assert 'gap_ratio' in result
    assert result['spectral_gap'] > 0


def test_analyze_spectral_gap_no_gap():
    """Test spectral gap when there's no clear gap."""
    eigenvalues = torch.linspace(0.1, 1.0, 10)

    result = analyze_spectral_gap(eigenvalues, expected_zero_modes=0)

    assert 'spectral_gap' in result
    assert result['min_eigenvalue'] >= 0
    assert result['max_eigenvalue'] <= 1.0


# ============================================================
# COHOMOLOGY VERIFICATION TESTS
# ============================================================

def test_verify_cohomology_ranks_correct():
    """Test cohomology verification with correct ranks."""
    # Create full-rank Gram matrices
    b2 = 5
    b3 = 10

    gram_h2 = torch.eye(b2) + 0.01 * torch.randn(b2, b2)
    gram_h2 = (gram_h2 + gram_h2.T) / 2  # Make symmetric

    gram_h3 = torch.eye(b3) + 0.01 * torch.randn(b3, b3)
    gram_h3 = (gram_h3 + gram_h3.T) / 2

    result = verify_cohomology_ranks(
        gram_h2, gram_h3,
        expected_b2=b2,
        expected_b3=b3
    )

    assert result['b2_rank'] == b2
    assert result['b3_rank'] == b3
    assert result['b2_correct']
    assert result['b3_correct']


def test_verify_cohomology_ranks_incorrect():
    """Test cohomology verification with incorrect ranks."""
    # Create deficient Gram matrices
    b2_expected = 5
    b3_expected = 10

    # But make them rank-deficient
    gram_h2 = torch.zeros(b2_expected, b2_expected)
    gram_h2[0, 0] = 1.0
    gram_h2[1, 1] = 1.0  # Only rank 2

    gram_h3 = torch.zeros(b3_expected, b3_expected)
    for i in range(5):
        gram_h3[i, i] = 1.0  # Only rank 5

    result = verify_cohomology_ranks(
        gram_h2, gram_h3,
        expected_b2=b2_expected,
        expected_b3=b3_expected
    )

    assert not result['b2_correct']
    assert not result['b3_correct']
    assert result['b2_rank'] < b2_expected
    assert result['b3_rank'] < b3_expected


# ============================================================
# LAPLACIAN SPECTRUM TESTS
# ============================================================

def test_compute_laplacian_spectrum():
    """Test Laplacian spectrum computation."""
    batch_size = 20

    # Create simple metric and phi
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
    phi = torch.randn(batch_size, 7, 7, 7)
    coords = torch.randn(batch_size, 7)

    result = compute_laplacian_spectrum(metric, phi, coords, n_eigenvalues=7)

    assert 'metric_eigenvalues' in result
    assert result['metric_eigenvalues'].shape[0] == 7
    assert result['batch_size'] == batch_size


# ============================================================
# HARMONIC FORM EXTRACTION TESTS
# ============================================================

def test_extract_harmonic_forms():
    """Test harmonic form extraction."""
    from g2forge.networks.harmonic_network import HarmonicNetwork

    manifold = g2.create_gift_k7()

    b2 = 5
    b3 = 10

    # Create networks
    h2_network = HarmonicNetwork(n_forms=b2, form_type='H2').to('cpu')
    h3_network = HarmonicNetwork(n_forms=b3, form_type='H3').to('cpu')

    h2_network.eval()
    h3_network.eval()

    result = extract_harmonic_forms(
        h2_network, h3_network, manifold,
        n_samples=100, device='cpu'
    )

    # Check result structure
    assert 'h2_forms' in result
    assert 'h3_forms' in result
    assert 'gram_h2' in result
    assert 'gram_h3' in result
    assert 'b2_rank' in result
    assert 'b3_rank' in result

    # Check shapes
    assert result['h2_forms'].shape[1] == b2
    assert result['h3_forms'].shape[1] == b3
    assert result['gram_h2'].shape == (b2, b2)
    assert result['gram_h3'].shape == (b3, b3)


def test_compute_harmonic_penalty():
    """Test harmonic penalty computation."""
    n_samples = 50
    n_forms = 3
    n_components = 10

    harmonic_forms = torch.randn(n_samples, n_forms, n_components)
    dphi = torch.randn(10, 7, 7, 7, 7)  # Dummy dphi

    penalty = compute_harmonic_penalty(harmonic_forms, dphi)

    assert isinstance(penalty, float)
    assert penalty >= 0
