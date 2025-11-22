"""
K₇ Manifold Implementation

The K₇ manifold is a compact 7-dimensional G₂ manifold constructed
via Twisted Connected Sum (TCS) by Kovalev (2003).

K₇ = M₁ᵀ ∪_φ M₂ᵀ

where M₁ᵀ and M₂ᵀ are asymptotically cylindrical (ACyl) Calabi-Yau 3-folds
with matching asymptotic structures, glued along a neck region.

This implementation supports ARBITRARY topology parameters (b₂, b₃),
not just GIFT's specific values (21, 77).

References:
    - Kovalev (2003): "Twisted connected sums and special Riemannian holonomy"
    - Corti, Haskins, Nordström, Pacini (2013-2015): TCS refinements
"""

from typing import Dict, List, Optional
import torch
import numpy as np

from .base import TCSManifold, Cycle
from ..utils.config import ManifoldConfig


class K7Manifold(TCSManifold):
    """
    K₇ manifold with configurable topology.

    Parametrized by t-coordinate and 6 periodic coordinates on S¹×S¹×S¹×...

    Coordinates:
        x[0] = t ∈ [0, 1] - TCS gluing parameter
        x[1:7] = θ₁,...,θ₆ ∈ [0, 2π] - Periodic coordinates
    """

    def __init__(self, config: ManifoldConfig):
        """
        Initialize K₇ manifold.

        Args:
            config: ManifoldConfig with type="K7", construction="TCS"
        """
        if config.type != "K7":
            raise ValueError(f"K7Manifold requires type='K7', got '{config.type}'")

        super().__init__(config)

        # Cache Levi-Civita for efficiency (used in multiple places)
        self._levi_civita_cache = None

    # ================================================================
    # COORDINATE SAMPLING
    # ================================================================

    def sample_coordinates(
        self,
        n_samples: int,
        grid_n: Optional[int] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Sample coordinates on K₇.

        Supports two modes:
        1. Grid sampling: Structured grid on [0,1] × [0,2π]⁶
        2. Random sampling: Uniform random in parameter space

        Args:
            n_samples: Number of samples (if grid_n is None)
            grid_n: Grid resolution (if provided, creates n^7 points)
            device: Torch device

        Returns:
            coords: Tensor[n_actual, 7]
                coords[:, 0] = t ∈ [0, 1]
                coords[:, 1:] = θ ∈ [0, 2π]

        Note:
            Hybrid mode (50% grid + 50% random) recommended for training.
        """
        if grid_n is not None:
            # Grid sampling
            coords = self._sample_grid(grid_n, device)
        else:
            # Random sampling
            coords = self._sample_random(n_samples, device)

        return coords

    def _sample_grid(self, n: int, device: str) -> torch.Tensor:
        """
        Create structured grid on K₇.

        Args:
            n: Grid resolution per dimension
            device: Torch device

        Returns:
            coords: Tensor[n^7, 7]

        Note:
            n^7 grows very fast! Use n ≤ 12 typically.
            n=8 → 2M points, n=10 → 10M points, n=12 → 35M points
        """
        # Create 1D grids
        t_grid = torch.linspace(0, 1, n, device=device)
        theta_grid = torch.linspace(0, 2*np.pi, n, device=device)

        # Meshgrid for all 7 dimensions
        grids = torch.meshgrid(
            [t_grid] + [theta_grid] * 6,
            indexing='ij'
        )

        # Flatten and stack
        coords = torch.stack([g.flatten() for g in grids], dim=1)

        return coords  # [n^7, 7]

    def _sample_random(self, n: int, device: str) -> torch.Tensor:
        """
        Random uniform sampling on K₇.

        Args:
            n: Number of samples
            device: Torch device

        Returns:
            coords: Tensor[n, 7]
        """
        coords = torch.zeros(n, 7, device=device)

        # t ∈ [0, 1]
        coords[:, 0] = torch.rand(n, device=device)

        # θ ∈ [0, 2π]
        coords[:, 1:] = 2 * np.pi * torch.rand(n, 6, device=device)

        return coords

    def sample_hybrid(
        self,
        n_samples: int,
        grid_n: int,
        grid_fraction: float = 0.5,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Hybrid sampling: part grid, part random.

        Recommended for training: combines coverage (grid) with
        diversity (random).

        Args:
            n_samples: Total number of samples
            grid_n: Grid resolution
            grid_fraction: Fraction from grid (0.5 = 50% grid, 50% random)
            device: Torch device

        Returns:
            coords: Tensor[n_samples, 7]
        """
        n_grid = int(n_samples * grid_fraction)
        n_random = n_samples - n_grid

        # Grid component
        grid_coords = self._sample_grid(grid_n, device)
        if grid_coords.shape[0] > n_grid:
            # Subsample if grid is too large
            indices = torch.randperm(grid_coords.shape[0])[:n_grid]
            grid_coords = grid_coords[indices]
        elif grid_coords.shape[0] < n_grid:
            # If grid is smaller, just use what we have
            n_grid = grid_coords.shape[0]
            n_random = n_samples - n_grid

        # Random component
        random_coords = self._sample_random(n_random, device)

        # Combine
        coords = torch.cat([grid_coords, random_coords], dim=0)

        # Shuffle
        indices = torch.randperm(coords.shape[0])
        coords = coords[indices]

        return coords

    # ================================================================
    # REGION WEIGHTS (TCS)
    # ================================================================

    def get_region_weights(
        self,
        coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute soft region weights for TCS structure.

        Regions:
        - M₁: t < 0.5 - ε₀/2
        - Neck: |t - 0.5| < ε₀/2
        - M₂: t > 0.5 + ε₀/2

        Uses smooth sigmoid transitions.

        Args:
            coords: Tensor[batch, 7]

        Returns:
            weights: Dict['m1'|'neck'|'m2', Tensor[batch]]

        Note:
            Weights sum to approximately 1 for each point.
        """
        t = coords[:, 0]  # Extract t-coordinate

        sharpness = self.tcs_params.transition_sharpness

        weights = {
            'm1': self.compute_region_indicator(t, 'm1', sharpness),
            'neck': self.compute_region_indicator(t, 'neck', sharpness),
            'm2': self.compute_region_indicator(t, 'm2', sharpness)
        }

        # Normalize to sum to 1 (approximately)
        total = weights['m1'] + weights['neck'] + weights['m2']
        for key in weights:
            weights[key] = weights[key] / (total + 1e-8)

        return weights

    # ================================================================
    # CALIBRATION CYCLES
    # ================================================================

    def get_associative_cycles(self) -> List[Cycle]:
        """
        Get associative 3-cycles for K₇.

        For K₇, associative cycles come from the Calabi-Yau factors.
        This is a simplified implementation.

        Returns:
            cycles: List of Cycle objects (dimension=3)

        Note:
            Number of cycles depends on topology.
            For GIFT (b₂=21, b₃=77), there are 21 associative cycles.
            Here we return a representative subset for calibration.
        """
        cycles = []

        # Create a few representative associative cycles
        # These are simplified - full implementation would use
        # detailed geometry of the Calabi-Yau factors

        for i in range(min(5, self.b2)):  # Use first 5 for efficiency
            cycle = Cycle(
                type="associative",
                dimension=3,
                indices=(0, i+1, (i+2) % 6 + 1),  # Simple parametrization
                volume=1.0
            )
            cycles.append(cycle)

        return cycles

    def get_coassociative_cycles(self) -> List[Cycle]:
        """
        Get coassociative 4-cycles for K₇.

        Returns:
            cycles: List of Cycle objects (dimension=4)

        Note:
            Simplified implementation. Full version would use
            actual geometric cycles from the construction.
        """
        cycles = []

        # Create representative coassociative cycles
        for i in range(min(3, self.b3 // 20)):  # Use small subset
            cycle = Cycle(
                type="coassociative",
                dimension=4,
                indices=(0, i+1, (i+2) % 6 + 1, (i+3) % 6 + 1),
                volume=1.0
            )
            cycles.append(cycle)

        return cycles

    def sample_on_cycle(
        self,
        cycle: Cycle,
        n_samples: int = 256
    ) -> torch.Tensor:
        """
        Sample points on a given cycle.

        Args:
            cycle: Cycle to sample on
            n_samples: Number of points

        Returns:
            coords: Tensor[n_samples, 7] - Points on cycle

        Note:
            Simplified implementation. Real version would use
            proper parametrization of the cycle.
        """
        device = 'cpu'  # Can be made configurable

        if cycle.dimension == 3:
            # Sample on associative 3-cycle
            coords = torch.zeros(n_samples, 7, device=device)

            # Use cycle indices to determine coordinates
            i, j, k = cycle.indices[:3]

            # Sample uniformly on the cycle's coordinates
            coords[:, i] = torch.rand(n_samples)
            coords[:, j] = 2 * np.pi * torch.rand(n_samples)
            coords[:, k] = 2 * np.pi * torch.rand(n_samples)

            # Fix other coordinates to center
            for l in range(7):
                if l not in [i, j, k]:
                    coords[:, l] = np.pi  # Fix at center

        elif cycle.dimension == 4:
            # Sample on coassociative 4-cycle
            coords = torch.zeros(n_samples, 7, device=device)

            i, j, k, l = cycle.indices[:4]

            coords[:, i] = torch.rand(n_samples)
            coords[:, j] = 2 * np.pi * torch.rand(n_samples)
            coords[:, k] = 2 * np.pi * torch.rand(n_samples)
            coords[:, l] = 2 * np.pi * torch.rand(n_samples)

            # Fix remaining coordinates
            for m in range(7):
                if m not in [i, j, k, l]:
                    coords[:, m] = np.pi

        return coords

    # ================================================================
    # UTILITIES
    # ================================================================

    def __repr__(self) -> str:
        """Enhanced string representation."""
        return (
            f"K7Manifold(TCS)\n"
            f"  M₁: b₂={self.b2_m1}, b₃={self.b3_m1}\n"
            f"  M₂: b₂={self.b2_m2}, b₃={self.b3_m2}\n"
            f"  Total: b₂={self.b2}, b₃={self.b3}\n"
            f"  Neck width: {self.neck_width:.4f}\n"
            f"  χ={self.euler_characteristic}"
        )


# ================================================================
# CONVENIENCE FUNCTIONS
# ================================================================

def create_gift_k7() -> K7Manifold:
    """
    Create K₇ manifold with GIFT parameters.

    Returns GIFT v1.0 configuration for validation.

    Returns:
        K7Manifold with b₂=21, b₃=77
    """
    from ..utils.config import G2ForgeConfig

    config = G2ForgeConfig.from_gift_v1_0()
    return K7Manifold(config.manifold)


def create_custom_k7(
    b2_m1: int,
    b3_m1: int,
    b2_m2: int,
    b3_m2: int,
    **kwargs
) -> K7Manifold:
    """
    Create K₇ manifold with custom topology.

    Args:
        b2_m1, b3_m1: M₁ topology
        b2_m2, b3_m2: M₂ topology
        **kwargs: Additional parameters

    Returns:
        K7Manifold with specified topology

    Example:
        >>> # Create K₇ with different topology than GIFT
        >>> k7 = create_custom_k7(b2_m1=10, b3_m1=38, b2_m2=9, b3_m2=35)
        >>> print(k7.b2, k7.b3)  # 19, 73
    """
    from ..utils.config import create_k7_config

    config = create_k7_config(
        b2_m1=b2_m1,
        b3_m1=b3_m1,
        b2_m2=b2_m2,
        b3_m2=b3_m2,
        **kwargs
    )

    return K7Manifold(config.manifold)


__all__ = [
    'K7Manifold',
    'create_gift_k7',
    'create_custom_k7',
]
