# -*- coding: utf-8 -*-
"""
Neural network architectures for G2 metric construction.

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
    # Phi network (G2 3-form)
    'FourierFeatures',
    'PhiNetwork',
    'create_phi_network_from_config',
    # Harmonic networks (auto-sized)
    'HarmonicNetwork',
    'create_harmonic_h2_network',
    'create_harmonic_h3_network',
    'create_harmonic_networks_from_config',
]
