"""
PhiNetwork: Neural network for generating G₂ 3-form φ

Learns a function φ: ℝ⁷ → Λ³(ℝ⁷) that defines the G₂ structure.

The 3-form φ is the fundamental object in G₂ geometry:
- It determines the metric via g_ij = (1/6) Σ φ_ipq φ_jpq
- Must satisfy dφ = 0 and d★φ = 0 for torsion-free G₂

Architecture:
- Fourier feature encoding for multi-scale geometry
- MLP with SiLU activation
- Antisymmetric output (valid 3-form)

Universal - works for any G₂ manifold!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class FourierFeatures(nn.Module):
    """
    Random Fourier features for coordinate encoding.

    Maps input x ∈ ℝ⁷ to high-dimensional features that capture
    multi-scale geometric information.

    φ(x) = [cos(2π B x), sin(2π B x)]

    where B is a random matrix of frequencies.
    """

    def __init__(
        self,
        input_dim: int = 7,
        n_frequencies: int = 32,
        scale: float = 1.0
    ):
        """
        Initialize Fourier features.

        Args:
            input_dim: Input dimension (7 for G₂)
            n_frequencies: Number of random frequencies
            scale: Frequency scale (controls multi-scale capture)
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_frequencies = n_frequencies

        # Random frequency matrix B
        # Fixed (not trainable) for stability
        B = torch.randn(n_frequencies, input_dim) * scale
        self.register_buffer('B', B)

    @property
    def output_dim(self) -> int:
        """Output dimension: 2 * n_frequencies (cos + sin)."""
        return 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features.

        Args:
            x: Tensor[batch, input_dim] - Input coordinates

        Returns:
            features: Tensor[batch, 2*n_frequencies]
        """
        # x B^T: [batch, input_dim] × [input_dim, n_freq] → [batch, n_freq]
        xB = torch.matmul(x, self.B.t())

        # Concatenate [cos(2π x B), sin(2π x B)]
        features = torch.cat([
            torch.cos(2 * np.pi * xB),
            torch.sin(2 * np.pi * xB)
        ], dim=-1)

        return features


class PhiNetwork(nn.Module):
    """
    Neural network that generates the G₂ 3-form φ.

    Architecture:
        x ∈ ℝ⁷ → Fourier → MLP → 35 components → Antisymmetrize → φ ∈ Λ³

    The output is a valid antisymmetric 3-form.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [384, 384, 256],
        n_fourier: int = 32,
        activation: str = 'silu',
        fourier_scale: float = 1.0
    ):
        """
        Initialize Phi network.

        Args:
            hidden_dims: List of hidden layer dimensions
            n_fourier: Number of Fourier frequencies
            activation: Activation function ('silu', 'relu', 'gelu')
            fourier_scale: Fourier feature scale

        Note:
            Output dimension is fixed at 35 (independent components of 3-form)
            35 = C(7,3) = 7!/(3!4!) = number of ways to choose 3 from 7
        """
        super().__init__()

        self.hidden_dims = hidden_dims
        self.n_fourier = n_fourier

        # Fourier feature encoding
        self.fourier = FourierFeatures(
            input_dim=7,
            n_frequencies=n_fourier,
            scale=fourier_scale
        )

        # Select activation
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP
        layers = []
        input_dim = self.fourier.output_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim

        # Output layer: 35 independent components
        # (i,j,k) with 0 ≤ i < j < k < 7
        layers.append(nn.Linear(input_dim, 35))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Glorot initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate φ from coordinates.

        Args:
            x: Tensor[batch, 7] - Coordinates on manifold

        Returns:
            phi_components: Tensor[batch, 35] - Independent components
        """
        # Fourier encoding
        features = self.fourier(x)  # [batch, 2*n_fourier]

        # MLP
        phi_components = self.mlp(features)  # [batch, 35]

        return phi_components

    def get_phi_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get full antisymmetric 3-form tensor.

        Args:
            x: Tensor[batch, 7] - Coordinates

        Returns:
            phi: Tensor[batch, 7, 7, 7] - Antisymmetric 3-form

        Note:
            This expands the 35 independent components into
            the full 7×7×7 tensor with antisymmetry.
        """
        batch_size = x.shape[0]
        device = x.device

        # Get components
        components = self.forward(x)  # [batch, 35]

        # Initialize tensor
        phi = torch.zeros(batch_size, 7, 7, 7, device=device)

        # Fill in with antisymmetrization
        idx = 0
        for i in range(7):
            for j in range(i + 1, 7):
                for k in range(j + 1, 7):
                    val = components[:, idx]

                    # All permutations with correct signs
                    # Even permutations: +val
                    phi[:, i, j, k] = val
                    phi[:, j, k, i] = val
                    phi[:, k, i, j] = val

                    # Odd permutations: -val
                    phi[:, i, k, j] = -val
                    phi[:, j, i, k] = -val
                    phi[:, k, j, i] = -val

                    idx += 1

        return phi

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation."""
        n_params = self.count_parameters()
        return (
            f"PhiNetwork(\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  n_fourier={self.n_fourier},\n"
            f"  n_parameters={n_params:,}\n"
            f")"
        )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_phi_network_from_config(config) -> PhiNetwork:
    """
    Create PhiNetwork from configuration.

    Args:
        config: G2ForgeConfig or NetworkArchitectureConfig

    Returns:
        PhiNetwork instance

    Example:
        >>> from g2forge.utils import G2ForgeConfig
        >>> config = G2ForgeConfig.from_gift_v1_0()
        >>> phi_net = create_phi_network_from_config(config)
    """
    # Handle both full config and architecture config
    if hasattr(config, 'architecture'):
        arch = config.architecture
    else:
        arch = config

    return PhiNetwork(
        hidden_dims=arch.phi_hidden_dims,
        n_fourier=arch.phi_n_fourier,
        activation=arch.phi_activation
    )


__all__ = [
    'FourierFeatures',
    'PhiNetwork',
    'create_phi_network_from_config',
]
