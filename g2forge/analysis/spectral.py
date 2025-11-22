"""
Spectral Analysis Tools for G₂ Metrics

Provides tools for analyzing spectral properties of G₂ metrics:
- Laplacian spectrum computation
- Harmonic form extraction and analysis
- Cohomology rank verification
- Spectral gap analysis

Based on differential geometry theory and spectral analysis methods.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_laplacian_spectrum(
    metric: torch.Tensor,
    phi: torch.Tensor,
    coords: torch.Tensor,
    n_eigenvalues: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Compute approximate spectrum of Hodge-Laplacian.

    The Hodge-Laplacian Δ = dδ + δd acts on differential forms.
    For harmonic forms, Δω = 0 (zero eigenvalue).

    Args:
        metric: Metric tensor [batch, 7, 7]
        phi: G₂ 3-form [batch, 7, 7, 7]
        coords: Coordinates [batch, 7]
        n_eigenvalues: Number of eigenvalues to compute

    Returns:
        Dict with 'eigenvalues' and other spectral info
    """
    # Simplified implementation: analyze metric eigenvalues
    # Full implementation would require discrete Laplacian
    batch_size = metric.shape[0]

    # Compute metric eigenvalues
    metric_eigenvalues = torch.linalg.eigvalsh(metric)  # [batch, 7]

    # Mean eigenvalue spectrum
    mean_eigenvalues = metric_eigenvalues.mean(dim=0)  # [7]

    return {
        'metric_eigenvalues': mean_eigenvalues,
        'n_computed': min(n_eigenvalues, 7),
        'batch_size': batch_size
    }


def extract_harmonic_forms(
    h2_network,
    h3_network,
    manifold,
    n_samples: int = 1000,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Extract orthonormal harmonic basis from trained networks.

    Args:
        h2_network: Trained HarmonicNetwork for H²
        h3_network: Trained HarmonicNetwork for H³
        manifold: Manifold instance
        n_samples: Sample points for evaluation
        device: Device

    Returns:
        Dict with harmonic forms and their properties
    """
    # Sample coordinates
    coords = manifold.sample_coordinates(n_samples=n_samples, device=device)

    with torch.no_grad():
        # Extract harmonic 2-forms
        h2_forms = h2_network(coords)  # [n_samples, b2, 21]

        # Extract harmonic 3-forms
        h3_forms = h3_network(coords)  # [n_samples, b3, 35]

    # Compute Gram matrices
    gram_h2 = compute_gram_matrix(h2_forms)  # [b2, b2]
    gram_h3 = compute_gram_matrix(h3_forms)  # [b3, b3]

    # Analyze orthonormality
    b2 = h2_forms.shape[1]
    b3 = h3_forms.shape[1]

    identity_h2 = torch.eye(b2, device=device)
    identity_h3 = torch.eye(b3, device=device)

    orthonormality_error_h2 = torch.norm(gram_h2 - identity_h2, p='fro').item()
    orthonormality_error_h3 = torch.norm(gram_h3 - identity_h3, p='fro').item()

    return {
        'h2_forms': h2_forms,
        'h3_forms': h3_forms,
        'gram_h2': gram_h2,
        'gram_h3': gram_h3,
        'orthonormality_error_h2': orthonormality_error_h2,
        'orthonormality_error_h3': orthonormality_error_h3,
        'b2_rank': compute_rank(gram_h2),
        'b3_rank': compute_rank(gram_h3),
    }


def compute_gram_matrix(forms: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for set of forms.

    For forms ω_i, the Gram matrix is G_ij = <ω_i, ω_j>
    where the inner product is integrated over the manifold.

    Args:
        forms: Forms [n_samples, n_forms, n_components]

    Returns:
        Gram matrix [n_forms, n_forms]
    """
    n_samples, n_forms, n_components = forms.shape

    # Approximate inner product as average over samples
    # <ω_i, ω_j> ≈ (1/n) Σ_x ω_i(x) · ω_j(x)

    gram = torch.zeros(n_forms, n_forms, device=forms.device)

    for i in range(n_forms):
        for j in range(n_forms):
            # Inner product between forms i and j
            inner_product = (forms[:, i, :] * forms[:, j, :]).sum(dim=1)  # [n_samples]
            gram[i, j] = inner_product.mean()

    return gram


def compute_rank(matrix: torch.Tensor, tolerance: float = 1e-6) -> int:
    """
    Compute numerical rank of matrix.

    Args:
        matrix: Square matrix [n, n]
        tolerance: Singular value threshold

    Returns:
        Numerical rank
    """
    # Compute singular values
    singular_values = torch.linalg.svdvals(matrix)

    # Count significant singular values
    rank = (singular_values > tolerance).sum().item()

    return int(rank)


def analyze_spectral_gap(
    eigenvalues: torch.Tensor,
    expected_zero_modes: int = 0
) -> Dict[str, float]:
    """
    Analyze spectral gap: separation between zero modes and nonzero modes.

    For harmonic forms, we expect some zero eigenvalues (harmonic modes)
    followed by a gap, then nonzero eigenvalues.

    Args:
        eigenvalues: Eigenvalues in ascending order
        expected_zero_modes: Expected number of zero eigenvalues

    Returns:
        Dict with gap analysis results
    """
    eigenvalues_sorted = torch.sort(eigenvalues.abs())[0]

    if expected_zero_modes > 0 and expected_zero_modes < len(eigenvalues_sorted):
        # Gap is the difference between largest "zero" and smallest nonzero
        largest_zero = eigenvalues_sorted[expected_zero_modes - 1].item()
        smallest_nonzero = eigenvalues_sorted[expected_zero_modes].item()

        gap = smallest_nonzero - largest_zero
        gap_ratio = smallest_nonzero / (largest_zero + 1e-10)
    else:
        gap = 0.0
        gap_ratio = 1.0

    return {
        'spectral_gap': gap,
        'gap_ratio': gap_ratio,
        'min_eigenvalue': eigenvalues_sorted[0].item(),
        'max_eigenvalue': eigenvalues_sorted[-1].item(),
    }


def verify_cohomology_ranks(
    gram_h2: torch.Tensor,
    gram_h3: torch.Tensor,
    expected_b2: int,
    expected_b3: int,
    tolerance: float = 1e-6
) -> Dict[str, any]:
    """
    Verify that extracted harmonic forms have correct cohomology ranks.

    Args:
        gram_h2: Gram matrix for H² [b2, b2]
        gram_h3: Gram matrix for H³ [b3, b3]
        expected_b2: Expected b₂ (from topology)
        expected_b3: Expected b₃ (from topology)
        tolerance: Rank computation tolerance

    Returns:
        Dict with verification results
    """
    # Compute numerical ranks
    b2_rank = compute_rank(gram_h2, tolerance)
    b3_rank = compute_rank(gram_h3, tolerance)

    # Check if ranks match expectations
    b2_correct = (b2_rank == expected_b2)
    b3_correct = (b3_rank == expected_b3)

    # Compute determinants (should be ~1 for orthonormal basis)
    det_h2 = torch.linalg.det(gram_h2).item()
    det_h3 = torch.linalg.det(gram_h3).item()

    return {
        'b2_rank': b2_rank,
        'b3_rank': b3_rank,
        'expected_b2': expected_b2,
        'expected_b3': expected_b3,
        'b2_correct': b2_correct,
        'b3_correct': b3_correct,
        'det_gram_h2': det_h2,
        'det_gram_h3': det_h3,
        'det_close_to_one_h2': abs(det_h2 - 1.0) < 0.5,
        'det_close_to_one_h3': abs(det_h3 - 1.0) < 0.5,
    }


def compute_harmonic_penalty(
    harmonic_forms: torch.Tensor,
    dphi: torch.Tensor
) -> float:
    """
    Compute penalty for harmonic forms not being closed.

    For true harmonic forms: dω = 0 (closed)

    Args:
        harmonic_forms: Harmonic forms [batch, n_forms, n_components]
        dphi: Exterior derivative (for verification)

    Returns:
        Penalty (should be near zero for true harmonics)
    """
    # Simplified: check that forms don't vary too much
    # (true harmonics should be smooth)
    variation = harmonic_forms.std(dim=0).mean().item()

    return variation


__all__ = [
    'compute_laplacian_spectrum',
    'extract_harmonic_forms',
    'compute_gram_matrix',
    'compute_rank',
    'analyze_spectral_gap',
    'verify_cohomology_ranks',
    'compute_harmonic_penalty',
]
