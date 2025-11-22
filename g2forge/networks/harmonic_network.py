"""
HarmonicNetwork: Extract harmonic forms from G₂ metric

Learns harmonic p-forms ω ∈ H^p(M) that satisfy:
- Closedness: dω = 0
- Coclosedness: d★ω = 0
- Orthonormality: ⟨ωᵢ, ωⱼ⟩ = δᵢⱼ

For G₂ manifolds:
- H² has dimension b₂ (varies by manifold!)
- H³ has dimension b₃ (varies by manifold!)

**KEY FEATURE: Auto-sized from topology config!**
Not hardcoded to GIFT's b₂=21, b₃=77.

Architecture:
- Fourier features for coordinate encoding
- MLP with shared backbone
- Multiple output heads (one per form)
- Gram-Schmidt orthonormalization (optional)
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .phi_network import FourierFeatures


class HarmonicNetwork(nn.Module):
    """
    Neural network for extracting harmonic p-forms.

    Generates n_forms orthonormal harmonic forms of degree p.

    The number of forms (n_forms) is determined by topology:
    - For H²: n_forms = b₂ (from TopologyConfig)
    - For H³: n_forms = b₃ (from TopologyConfig)

    **UNIVERSAL** - not hardcoded to specific values!
    """

    def __init__(
        self,
        p: int,  # Form degree (2 or 3)
        n_forms: int,  # Number of forms (b₂ or b₃ from topology!)
        hidden_dim: int = 128,
        n_fourier: int = 24,
        activation: str = 'silu',
        use_gram_schmidt: bool = False
    ):
        """
        Initialize harmonic form network.

        Args:
            p: Form degree (2 for 2-forms, 3 for 3-forms)
            n_forms: Number of forms to generate
                For H²: n_forms = topology.b2
                For H³: n_forms = topology.b3
            hidden_dim: Hidden layer dimension
            n_fourier: Fourier feature frequencies
            activation: Activation function
            use_gram_schmidt: Apply Gram-Schmidt orthonormalization

        Note:
            **n_forms comes from config, not hardcoded!**
            This makes the network universal.
        """
        super().__init__()

        self.p = p
        self.n_forms = n_forms
        self.hidden_dim = hidden_dim
        self.n_fourier = n_fourier
        self.use_gram_schmidt = use_gram_schmidt

        # Number of independent components for p-form
        # 2-form: C(7,2) = 21 components
        # 3-form: C(7,3) = 35 components
        if p == 2:
            self.n_components = 21  # C(7,2)
        elif p == 3:
            self.n_components = 35  # C(7,3)
        else:
            raise ValueError(f"Only p=2 or p=3 supported, got p={p}")

        # Fourier encoding
        self.fourier = FourierFeatures(
            input_dim=7,
            n_frequencies=n_fourier
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

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(self.fourier.output_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )

        # Multiple output heads (one per form)
        # Each head outputs n_components
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.n_components)
            for _ in range(n_forms)
        ])

        # Initialize
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate harmonic forms.

        Args:
            x: Tensor[batch, 7] - Coordinates

        Returns:
            forms: Tensor[batch, n_forms, n_components]
                Harmonic form coefficients

        Note:
            n_forms is determined by topology at __init__!
        """
        batch_size = x.shape[0]

        # Fourier encoding
        features = self.fourier(x)  # [batch, 2*n_fourier]

        # Shared backbone
        h = self.backbone(features)  # [batch, hidden_dim]

        # Multiple heads
        forms = []
        for head in self.heads:
            form = head(h)  # [batch, n_components]
            forms.append(form)

        forms = torch.stack(forms, dim=1)  # [batch, n_forms, n_components]

        # Optional: Gram-Schmidt orthonormalization
        if self.use_gram_schmidt:
            forms = self._gram_schmidt(forms)

        return forms

    def _gram_schmidt(self, forms: torch.Tensor) -> torch.Tensor:
        """
        Gram-Schmidt orthonormalization (optional).

        Args:
            forms: Tensor[batch, n_forms, n_components]

        Returns:
            orth_forms: Tensor[batch, n_forms, n_components]
                Orthonormalized forms

        Note:
            This is optional - can also enforce via loss.
            Gram-Schmidt can be numerically unstable for large n_forms.
        """
        batch_size = forms.shape[0]
        orth_forms = torch.zeros_like(forms)

        for i in range(self.n_forms):
            # Start with original form
            v = forms[:, i, :].clone()

            # Subtract projections onto previous forms
            for j in range(i):
                u = orth_forms[:, j, :]
                # Projection: ⟨v, u⟩ u / ⟨u, u⟩
                proj = (torch.sum(v * u, dim=-1, keepdim=True) /
                       (torch.sum(u * u, dim=-1, keepdim=True) + 1e-8)) * u
                v = v - proj

            # Normalize
            norm = torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True) + 1e-8)
            orth_forms[:, i, :] = v / norm

        return orth_forms

    def get_forms_as_tensors(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Get harmonic forms as antisymmetric tensors.

        Args:
            x: Tensor[batch, 7] - Coordinates

        Returns:
            tensors: For p=2: Tensor[batch, n_forms, 7, 7]
                    For p=3: Tensor[batch, n_forms, 7, 7, 7]

        Note:
            This expands independent components into full tensor.
        """
        forms = self.forward(x)  # [batch, n_forms, n_components]
        batch_size = forms.shape[0]
        device = forms.device

        if self.p == 2:
            # 2-form tensors
            tensors = torch.zeros(
                batch_size, self.n_forms, 7, 7,
                device=device
            )

            for form_idx in range(self.n_forms):
                idx = 0
                for i in range(7):
                    for j in range(i + 1, 7):
                        val = forms[:, form_idx, idx]
                        tensors[:, form_idx, i, j] = val
                        tensors[:, form_idx, j, i] = -val
                        idx += 1

        elif self.p == 3:
            # 3-form tensors
            tensors = torch.zeros(
                batch_size, self.n_forms, 7, 7, 7,
                device=device
            )

            for form_idx in range(self.n_forms):
                idx = 0
                for i in range(7):
                    for j in range(i + 1, 7):
                        for k in range(j + 1, 7):
                            val = forms[:, form_idx, idx]

                            # Antisymmetrize
                            tensors[:, form_idx, i, j, k] = val
                            tensors[:, form_idx, j, k, i] = val
                            tensors[:, form_idx, k, i, j] = val
                            tensors[:, form_idx, i, k, j] = -val
                            tensors[:, form_idx, j, i, k] = -val
                            tensors[:, form_idx, k, j, i] = -val

                            idx += 1

        return tensors

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation."""
        n_params = self.count_parameters()
        return (
            f"HarmonicNetwork(\n"
            f"  degree={self.p},\n"
            f"  n_forms={self.n_forms},  # ← From topology!\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  n_parameters={n_params:,}\n"
            f")"
        )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_harmonic_h2_network(
    topology,
    hidden_dim: int = 128,
    n_fourier: int = 24,
    **kwargs
) -> HarmonicNetwork:
    """
    Create H² network from topology.

    Args:
        topology: TopologyConfig (contains b₂)
        hidden_dim: Hidden dimension
        n_fourier: Fourier frequencies
        **kwargs: Additional arguments

    Returns:
        HarmonicNetwork for H² with n_forms = b₂

    Example:
        >>> from g2forge.utils import TopologyConfig
        >>> topology = TopologyConfig(b2=21, b3=77)
        >>> h2_net = create_harmonic_h2_network(topology)
        >>> print(h2_net.n_forms)  # 21
    """
    return HarmonicNetwork(
        p=2,
        n_forms=topology.b2,  # ✨ Auto-sized!
        hidden_dim=hidden_dim,
        n_fourier=n_fourier,
        **kwargs
    )


def create_harmonic_h3_network(
    topology,
    hidden_dim: int = 128,
    n_fourier: int = 24,
    **kwargs
) -> HarmonicNetwork:
    """
    Create H³ network from topology.

    Args:
        topology: TopologyConfig (contains b₃)
        hidden_dim: Hidden dimension
        n_fourier: Fourier frequencies
        **kwargs: Additional arguments

    Returns:
        HarmonicNetwork for H³ with n_forms = b₃

    Example:
        >>> topology = TopologyConfig(b2=19, b3=73)  # Custom!
        >>> h3_net = create_harmonic_h3_network(topology)
        >>> print(h3_net.n_forms)  # 73 (not 77!)
    """
    return HarmonicNetwork(
        p=3,
        n_forms=topology.b3,  # ✨ Auto-sized!
        hidden_dim=hidden_dim,
        n_fourier=n_fourier,
        **kwargs
    )


def create_harmonic_networks_from_config(config):
    """
    Create both H² and H³ networks from config.

    Args:
        config: G2ForgeConfig

    Returns:
        h2_network: HarmonicNetwork for H²
        h3_network: HarmonicNetwork for H³

    Example:
        >>> from g2forge.utils import create_k7_config
        >>> config = create_k7_config(b2_m1=10, b3_m1=38, b2_m2=9, b3_m2=35)
        >>> h2_net, h3_net = create_harmonic_networks_from_config(config)
        >>> print(h2_net.n_forms, h3_net.n_forms)  # 19, 73
    """
    topology = config.manifold.topology
    arch = config.architecture

    h2_network = HarmonicNetwork(
        p=2,
        n_forms=topology.b2,
        hidden_dim=arch.h2_hidden_dim,
        n_fourier=arch.h2_n_fourier
    )

    h3_network = HarmonicNetwork(
        p=3,
        n_forms=topology.b3,
        hidden_dim=arch.h3_hidden_dim,
        n_fourier=arch.h3_n_fourier
    )

    return h2_network, h3_network


__all__ = [
    'HarmonicNetwork',
    'create_harmonic_h2_network',
    'create_harmonic_h3_network',
    'create_harmonic_networks_from_config',
]
