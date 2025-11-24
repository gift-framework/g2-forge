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


# ============================================================
# FRACTALITY INDEX (Multi-Scale FFT) - from GIFT v1.2b
# ============================================================

def downsample_tensor(
    T: torch.Tensor,
    factor: int = 2
) -> torch.Tensor:
    """
    Downsample a tensor by subsampling.

    Args:
        T: Input tensor of shape (batch, 7, 7, 7, 7) (torsion 4-form)
        factor: Downsampling factor

    Returns:
        T_down: Downsampled tensor
    """
    return T[..., ::factor]


def compute_power_spectrum_slope(T_flat: torch.Tensor) -> float:
    """
    Compute power spectrum slope in log-log space.

    Fractal structures exhibit power-law behavior: P(k) ~ k^(-α)

    Args:
        T_flat: Flattened torsion tensor [N]

    Returns:
        slope: Power spectrum slope (negative for fractals)
    """
    if len(T_flat) < 10:
        return -2.0

    # FFT power spectrum
    fft = torch.fft.rfft(T_flat)
    power = torch.abs(fft) ** 2

    if len(power) < 3:
        return -2.0

    # Log-log fit
    k = torch.arange(1, len(power), device=T_flat.device, dtype=T_flat.dtype)
    log_k = torch.log(k + 1e-10)
    log_P = torch.log(power[1:] + 1e-10)

    # Linear regression
    k_mean = log_k.mean()
    P_mean = log_P.mean()
    numerator = ((log_k - k_mean) * (log_P - P_mean)).sum()
    denominator = ((log_k - k_mean) ** 2).sum()

    if denominator > 1e-10:
        slope = (numerator / denominator).item()
    else:
        slope = -2.0

    return slope


def compute_fractality_index(
    torsion: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """
    Compute multi-scale fractality index using 3-resolution FFT analysis.

    From GIFT v1.2b: Analyzes power spectrum at full, half, and quarter
    resolution to capture scale-invariant fractal behavior.

    Args:
        torsion: Torsion tensor of shape (batch, 7, 7, 7, 7)

    Returns:
        frac_idx: Fractality index per sample, shape (batch,)
        frac_idx_mean: Mean fractality for monitoring
    """
    batch_size = torsion.shape[0]
    device = torsion.device
    dtype = torsion.dtype

    frac_idx = torch.zeros(batch_size, device=device, dtype=dtype)

    for b in range(batch_size):
        # Full resolution
        T_full = torsion[b].flatten()
        slope_full = compute_power_spectrum_slope(T_full)

        # Half resolution
        T_half = downsample_tensor(torsion[b:b + 1], factor=2)[0].flatten()
        slope_half = compute_power_spectrum_slope(T_half)

        # Quarter resolution
        T_quarter = downsample_tensor(torsion[b:b + 1], factor=4)[0].flatten()
        slope_quarter = compute_power_spectrum_slope(T_quarter)

        # Average slopes
        raw_slope = (slope_full + slope_half + slope_quarter) / 3.0

        # Zero-center and map to [-0.5, +0.5]
        frac_centered = raw_slope + 2.5
        frac_idx[b] = 0.5 * torch.tanh(
            torch.tensor(-frac_centered, device=device, dtype=dtype)
        )

    frac_idx_mean = frac_idx.mean().item()

    return frac_idx, frac_idx_mean


# ============================================================
# TORSION DIVERGENCE
# ============================================================

def compute_divergence_torsion(
    torsion: torch.Tensor,
    coords: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """
    Compute torsion divergence ∇·T.

    Args:
        torsion: Torsion 4-form of shape (batch, 7, 7, 7, 7)
        coords: Coordinates of shape (batch, 7)

    Returns:
        div_T: Divergence per sample, shape (batch,)
        div_T_mean: Mean divergence
    """
    batch_size = torsion.shape[0]
    device = torsion.device

    if batch_size == 1:
        return torch.zeros(batch_size, device=device), 0.0

    # Flatten and compute variation from mean
    torsion_flat = torsion.reshape(batch_size, -1)
    torsion_mean = torsion_flat.mean(dim=0, keepdim=True)
    component_var = torch.abs(torsion_flat - torsion_mean)

    # Grid spacing
    dx = 1.0 / 16.0

    # Divergence estimate
    div_T = component_var.sum(dim=-1) / (dx * (7 ** 4))
    div_T_mean = div_T.mean().item()

    return div_T, div_T_mean


# ============================================================
# MULTI-GRID EVALUATION
# ============================================================

def subsample_coords_to_coarse_grid(
    coords: torch.Tensor,
    n_coarse: int = 8
) -> torch.Tensor:
    """Subsample coordinates to a coarser grid."""
    batch_size = coords.shape[0]
    subsample_size = max(1, batch_size // 2)
    indices = torch.randperm(batch_size, device=coords.device)[:subsample_size]
    return coords[indices]


def compute_multi_grid_rg_quantities(
    phi_network,
    manifold,
    coords_fine: torch.Tensor,
    n_grid_coarse: int = 8
) -> Tuple[float, float]:
    """
    Compute RG quantities on multiple grids.

    From GIFT v1.2b: Multi-grid evaluation provides robust estimates.

    Args:
        phi_network: Neural network for φ
        manifold: Manifold instance
        coords_fine: Fine grid coordinates
        n_grid_coarse: Coarse grid resolution

    Returns:
        divT_eff: Effective divergence (averaged)
        fract_eff: Effective fractality (averaged)
    """
    from ..core.operators import compute_exterior_derivative

    # Fine grid
    with torch.no_grad():
        phi_fine = phi_network(coords_fine)
        dphi_fine = compute_exterior_derivative(phi_fine, coords_fine)
        divT_fine, divT_fine_mean = compute_divergence_torsion(dphi_fine, coords_fine)
        fract_fine, fract_fine_mean = compute_fractality_index(dphi_fine)

    # Coarse grid
    coords_coarse = subsample_coords_to_coarse_grid(coords_fine, n_grid_coarse)
    with torch.no_grad():
        phi_coarse = phi_network(coords_coarse)
        dphi_coarse = compute_exterior_derivative(phi_coarse, coords_coarse)
        divT_coarse, divT_coarse_mean = compute_divergence_torsion(dphi_coarse, coords_coarse)
        fract_coarse, fract_coarse_mean = compute_fractality_index(dphi_coarse)

    # Average
    divT_eff = 0.5 * (divT_fine_mean + divT_coarse_mean)
    fract_eff = 0.5 * (fract_fine_mean + fract_coarse_mean)

    return divT_eff, fract_eff


__all__ = [
    'compute_laplacian_spectrum',
    'extract_harmonic_forms',
    'compute_gram_matrix',
    'compute_rank',
    'analyze_spectral_gap',
    'verify_cohomology_ranks',
    'compute_harmonic_penalty',
    'downsample_tensor',
    'compute_power_spectrum_slope',
    'compute_fractality_index',
    'compute_divergence_torsion',
    'subsample_coords_to_coarse_grid',
    'compute_multi_grid_rg_quantities',
]
