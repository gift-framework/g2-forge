"""
Spectral analysis tools for G₂ metrics.

Provides tools for:
- Laplacian spectrum computation
- Harmonic form extraction
- Cohomology rank verification
- Spectral gap analysis
"""

from .spectral import (
    compute_laplacian_spectrum,
    extract_harmonic_forms,
    compute_gram_matrix,
    compute_rank,
    analyze_spectral_gap,
    verify_cohomology_ranks,
    compute_harmonic_penalty,
)

__all__ = [
    'compute_laplacian_spectrum',
    'extract_harmonic_forms',
    'compute_gram_matrix',
    'compute_rank',
    'analyze_spectral_gap',
    'verify_cohomology_ranks',
    'compute_harmonic_penalty',
]
