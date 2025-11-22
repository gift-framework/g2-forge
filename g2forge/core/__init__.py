# -*- coding: utf-8 -*-
"""
Core differential geometry operations for G2 manifolds.

This module contains pure mathematical operators and loss functions
that work for any G2 manifold, not just GIFT-specific parameters.
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

from .losses import (
    torsion_closure_loss,
    torsion_coclosure_loss,
    volume_loss,
    gram_matrix_loss,
    boundary_smoothness_loss,
    calibration_associative_loss,
    calibration_coassociative_loss,
    AdaptiveLossScheduler,
    CompositeLoss,
)

__all__ = [
    # Operators
    'build_levi_civita_sparse_7d',
    'hodge_star_3',
    'compute_exterior_derivative',
    'compute_coclosure',
    'region_weighted_torsion',
    'neck_smoothness_loss',
    'reconstruct_metric_from_phi',
    'validate_antisymmetry',
    'compute_volume_form',
    # Losses
    'torsion_closure_loss',
    'torsion_coclosure_loss',
    'volume_loss',
    'gram_matrix_loss',
    'boundary_smoothness_loss',
    'calibration_associative_loss',
    'calibration_coassociative_loss',
    'AdaptiveLossScheduler',
    'CompositeLoss',
]
