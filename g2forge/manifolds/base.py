"""
Abstract base class for G₂ manifolds.

Defines the interface that all G₂ manifold implementations must satisfy.
This allows g2-forge to work with any G₂ manifold (K₇, Joyce, custom, etc.)
in a unified way.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch

from ..utils.config import ManifoldConfig, TopologyConfig


@dataclass
class Cycle:
    """
    Geometric cycle (submanifold) in G₂ manifold.

    Used for calibration conditions and topology.
    """
    # Cycle type
    type: str  # "associative" (3-cycle) or "coassociative" (4-cycle)

    # Dimension
    dimension: int  # 3 or 4

    # Indices defining the cycle (simplified representation)
    indices: Tuple[int, ...]

    # Expected volume (for calibration)
    volume: float = 1.0

    # Parametrization (optional)
    parametrization: Optional[callable] = None


class Manifold(ABC):
    """
    Abstract base class for G₂ manifolds.

    All manifold implementations (K₇, Joyce, Custom) must inherit
    from this class and implement the required methods.

    This abstraction allows g2-forge to be manifold-agnostic.
    """

    def __init__(self, config: ManifoldConfig):
        """
        Initialize manifold from configuration.

        Args:
            config: ManifoldConfig specifying topology and parameters
        """
        self.config = config
        self.config.validate()

    # ================================================================
    # REQUIRED METHODS (must be implemented by subclasses)
    # ================================================================

    @abstractmethod
    def sample_coordinates(
        self,
        n_samples: int,
        grid_n: Optional[int] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Sample coordinates on the manifold.

        Args:
            n_samples: Number of points to sample
            grid_n: Optional grid resolution (for structured sampling)
            device: Torch device ('cpu' or 'cuda')

        Returns:
            coords: Tensor[n_samples, 7] - Sampled coordinates

        Note:
            Coordinates should cover the manifold representatively.
            Can use grid sampling, random sampling, or hybrid.
        """
        pass

    @abstractmethod
    def get_region_weights(
        self,
        coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute soft region weights for coordinates.

        For TCS manifolds: regions = {m1, neck, m2}
        For other manifolds: can return empty dict or single region

        Args:
            coords: Tensor[batch, 7] - Coordinates

        Returns:
            weights: Dict[region_name, Tensor[batch]]
                Each weight ∈ [0, 1], should sum to ~1

        Note:
            Used for regional loss weighting in TCS construction.
            Non-TCS manifolds can return uniform weights.
        """
        pass

    @abstractmethod
    def get_associative_cycles(self) -> List[Cycle]:
        """
        Get list of associative 3-cycles for calibration.

        Associative cycles satisfy: φ|_Σ = vol_Σ

        Returns:
            cycles: List of Cycle objects (dimension=3)

        Note:
            Used for calibration loss in training.
            Can return empty list if not using calibration.
        """
        pass

    @abstractmethod
    def get_coassociative_cycles(self) -> List[Cycle]:
        """
        Get list of coassociative 4-cycles for calibration.

        Coassociative cycles satisfy: ★φ|_Ω = vol_Ω

        Returns:
            cycles: List of Cycle objects (dimension=4)

        Note:
            Used for calibration loss in training.
            Can return empty list if not using calibration.
        """
        pass

    # ================================================================
    # PROVIDED METHODS (common to all manifolds)
    # ================================================================

    @property
    def topology(self) -> TopologyConfig:
        """Get topology configuration."""
        return self.config.topology

    @property
    def b2(self) -> int:
        """Number of harmonic 2-forms."""
        return self.config.topology.b2

    @property
    def b3(self) -> int:
        """Number of harmonic 3-forms."""
        return self.config.topology.b3

    @property
    def dimension(self) -> int:
        """Manifold dimension (always 7 for G₂)."""
        return self.config.dimension

    @property
    def euler_characteristic(self) -> int:
        """Euler characteristic χ."""
        return self.config.topology.euler_characteristic

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.config.type}, "
            f"b₂={self.b2}, "
            f"b₃={self.b3}, "
            f"χ={self.euler_characteristic})"
        )


class TCSManifold(Manifold):
    """
    Abstract base for Twisted Connected Sum manifolds.

    TCS manifolds have additional structure: K₇ = M₁ ∪_neck M₂

    Subclasses (like K7) should implement TCS-specific logic.
    """

    def __init__(self, config: ManifoldConfig):
        """Initialize TCS manifold."""
        super().__init__(config)

        if config.construction != "TCS":
            raise ValueError(
                f"TCSManifold requires construction='TCS', got '{config.construction}'"
            )

        if config.tcs_params is None:
            raise ValueError("TCSManifold requires tcs_params in config")

        self.tcs_params = config.tcs_params

    @property
    def b2_m1(self) -> int:
        """b₂ of M₁ component."""
        return self.tcs_params.b2_m1

    @property
    def b3_m1(self) -> int:
        """b₃ of M₁ component."""
        return self.tcs_params.b3_m1

    @property
    def b2_m2(self) -> int:
        """b₂ of M₂ component."""
        return self.tcs_params.b2_m2

    @property
    def b3_m2(self) -> int:
        """b₃ of M₂ component."""
        return self.tcs_params.b3_m2

    @property
    def neck_width(self) -> float:
        """Width of neck region."""
        return self.tcs_params.neck_width

    def compute_region_indicator(
        self,
        t: torch.Tensor,
        region: str,
        sharpness: float = 10.0
    ) -> torch.Tensor:
        """
        Compute smooth indicator function for TCS regions.

        Uses sigmoid functions for smooth transitions.

        Args:
            t: Tensor[batch] - t-coordinate ∈ [0, 1]
            region: 'm1', 'neck', or 'm2'
            sharpness: Transition sharpness (higher = sharper)

        Returns:
            indicator: Tensor[batch] ∈ [0, 1]

        Note:
            M₁: t < 0.5 - neck_width/2
            Neck: 0.5 - neck_width/2 < t < 0.5 + neck_width/2
            M₂: t > 0.5 + neck_width/2
        """
        center = self.tcs_params.neck_center
        half_width = self.neck_width / 2.0

        if region == 'm1':
            # Indicator for t < center - half_width
            return torch.sigmoid(-sharpness * (t - (center - half_width)))

        elif region == 'neck':
            # Indicator for |t - center| < half_width
            left = torch.sigmoid(sharpness * (t - (center - half_width)))
            right = torch.sigmoid(-sharpness * (t - (center + half_width)))
            return left * right

        elif region == 'm2':
            # Indicator for t > center + half_width
            return torch.sigmoid(sharpness * (t - (center + half_width)))

        else:
            raise ValueError(f"Unknown region: {region}")

    def __repr__(self) -> str:
        """String representation for TCS manifolds."""
        return (
            f"{self.__class__.__name__}("
            f"M₁: b₂={self.b2_m1},b₃={self.b3_m1}, "
            f"M₂: b₂={self.b2_m2},b₃={self.b3_m2}, "
            f"Total: b₂={self.b2},b₃={self.b3})"
        )


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def create_manifold(config: ManifoldConfig) -> Manifold:
    """
    Factory function to create manifold from configuration.

    Args:
        config: ManifoldConfig

    Returns:
        Manifold instance

    Raises:
        ValueError: If manifold type is unknown

    Supported manifold types:
        - "K7": TCS K7 manifold (Kovalev construction)
        - "Joyce": Joyce construction (resolved orbifold) [STUB]
    """
    # Import here to avoid circular imports
    from .k7 import K7Manifold
    from .joyce import JoyceManifold

    if config.type == "K7":
        return K7Manifold(config)
    elif config.type == "Joyce":
        return JoyceManifold(config)
    else:
        raise ValueError(
            f"Unknown manifold type: {config.type}. "
            f"Supported types: 'K7', 'Joyce'"
        )


__all__ = [
    'Cycle',
    'Manifold',
    'TCSManifold',
    'create_manifold',
]
