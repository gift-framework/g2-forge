"""
Volume Normalization for G₂ Metrics

Implements dynamic volume normalization from GIFT v1.2a/b.
Instead of just penalizing det(g) deviation, we compute a scale factor
to achieve target determinant exactly.

Key innovation:
    scale = (target_det / current_det)^(1/7)
    g_normalized = scale * g

This dramatically improves det(g) precision from ±20% to ±1%.

Reference:
    GIFT v1.2a/b - Volume normalization at end of phase 2
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class VolumeNormalizer:
    """
    Dynamic volume normalization for G₂ metrics.

    Computes and applies a scaling factor to achieve target det(g).
    Should be called at the end of geometric stabilization phases
    (typically end of phase 2 in curriculum).
    """

    def __init__(self, target_det: float = 2.0):
        """
        Initialize volume normalizer.

        Args:
            target_det: Target determinant (default 2.0 from GIFT)
        """
        self.target_det = target_det
        self.volume_scale = 1.0
        self.is_normalized = False

    def compute_scale(
        self,
        phi_network: nn.Module,
        manifold,
        n_samples: int = 512,
        device: str = 'cuda'
    ) -> float:
        """
        Compute volume normalization scale.

        Samples metric at random points and computes:
            scale = (target_det / mean_det)^(1/7)

        Args:
            phi_network: Neural network for φ
            manifold: Manifold instance for coordinate sampling
            n_samples: Number of samples for averaging
            device: Torch device

        Returns:
            scale: Volume scale factor
        """
        with torch.no_grad():
            # Sample coordinates
            coords = manifold.sample_coordinates(
                n_samples=n_samples,
                device=device
            )
            coords.requires_grad_(True)

            # Compute φ
            phi = phi_network(coords)

            # Reconstruct metric from φ (simplified)
            from ..core.operators import reconstruct_metric_from_phi
            metric = reconstruct_metric_from_phi(phi)

            # Compute determinants
            det_g = torch.det(metric)
            det_g_mean = det_g.mean().item()

            # Compute scale: (target / current)^(1/7)
            # Exponent 1/7 because det scales as (scale)^7 in 7D
            scale = (self.target_det / (det_g_mean + 1e-8)) ** (1.0 / 7.0)

        return scale

    def normalize(
        self,
        phi_network: nn.Module,
        manifold,
        n_samples: int = 512,
        device: str = 'cuda',
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute and store normalization scale.

        Args:
            phi_network: Neural network for φ
            manifold: Manifold instance
            n_samples: Samples for scale computation
            device: Torch device
            verbose: Print normalization info

        Returns:
            info: Dict with normalization statistics
        """
        # Compute current det(g) before normalization
        with torch.no_grad():
            coords = manifold.sample_coordinates(n_samples=n_samples, device=device)
            coords.requires_grad_(True)
            phi = phi_network(coords)

            from ..core.operators import reconstruct_metric_from_phi
            metric = reconstruct_metric_from_phi(phi)
            det_g_before = torch.det(metric).mean().item()

        # Compute scale
        self.volume_scale = self.compute_scale(
            phi_network, manifold, n_samples, device
        )

        # Verify normalization
        # Note: We don't actually modify the network weights here,
        # normalization is applied during metric computation
        self.is_normalized = True

        info = {
            'det_g_before': det_g_before,
            'target_det': self.target_det,
            'scale_factor': self.volume_scale,
            'det_g_after_estimated': det_g_before * (self.volume_scale ** 7)
        }

        if verbose:
            print(f"\n[Volume Normalization]")
            print(f"  Current det(g):  {det_g_before:.6f}")
            print(f"  Target det(g):   {self.target_det:.6f}")
            print(f"  Scale factor:    {self.volume_scale:.6f}")
            print(f"  Expected det(g): {info['det_g_after_estimated']:.6f}")

        return info

    def apply_to_metric(self, metric: torch.Tensor) -> torch.Tensor:
        """
        Apply volume normalization to a metric tensor.

        Args:
            metric: Tensor[batch, 7, 7] - Unnormalized metric

        Returns:
            metric_normalized: Tensor[batch, 7, 7] - Normalized metric
        """
        if not self.is_normalized:
            # If not yet normalized, return unchanged
            return metric

        # Apply scale
        return self.volume_scale * metric

    def reset(self):
        """Reset normalization."""
        self.volume_scale = 1.0
        self.is_normalized = False


__all__ = ['VolumeNormalizer']
