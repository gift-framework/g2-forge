"""
Core differential geometry operations for G‚ manifolds.

This module contains pure mathematical operators that work
for any G‚ manifold, not just GIFT-specific parameters.
"""

from .operators import (
    build_levi_civita_sparse_7d,
    hodge_star_3,
    compute_exterior_derivative,
    compute_coclosure,
    region_weighted_torsion,
    neck_smoothness_loss,
    reconstruct_metric_from_phi,
    validate_antisymmetry,
    compute_volume_form,
)

__all__ = [
    'build_levi_civita_sparse_7d',
    'hodge_star_3',
    'compute_exterior_derivative',
    'compute_coclosure',
    'region_weighted_torsion',
    'neck_smoothness_loss',
    'reconstruct_metric_from_phi',
    'validate_antisymmetry',
    'compute_volume_form',
]
