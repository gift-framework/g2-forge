"""
Neural network architectures for G‚ metric construction.

Provides networks that are auto-sized from topology configuration,
not hardcoded to specific manifold parameters.
"""

from .phi_network import (
    FourierFeatures,
    PhiNetwork,
    create_phi_network_from_config,
)

from .harmonic_network import (
    HarmonicNetwork,
    create_harmonic_h2_network,
    create_harmonic_h3_network,
    create_harmonic_networks_from_config,
)

__all__ = [
    # Phi network
    'FourierFeatures',
    'PhiNetwork',
    'create_phi_network_from_config',
    # Harmonic networks
    'HarmonicNetwork',
    'create_harmonic_h2_network',
    'create_harmonic_h3_network',
    'create_harmonic_networks_from_config',
]
