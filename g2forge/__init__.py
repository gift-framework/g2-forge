# -*- coding: utf-8 -*-
"""
g2-forge: Neural Construction of Exceptional Holonomy Metrics

A universal framework for constructing G2 metrics on compact 7-manifolds
using Physics-Informed Neural Networks (PINNs).

Not limited to GIFT parameters - works for ANY G2 manifold!

Quick Start:
    >>> import g2forge as g2
    >>>
    >>> # Create K7 manifold with custom topology
    >>> config = g2.create_k7_config(
    ...     b2_m1=10, b3_m1=38,
    ...     b2_m2=9, b3_m2=35
    ... )
    >>> # Results in b2=19, b3=73

Example:
    >>> from g2forge.utils import G2ForgeConfig
    >>>
    >>> # GIFT reproduction
    >>> config = G2ForgeConfig.from_gift_v1_0()
    >>> print(f"Topology: b2={config.manifold.topology.b2}")
    Topology: b2=21

References:
    - Joyce (2000): Compact Manifolds with Special Holonomy
    - Kovalev (2003): Twisted Connected Sum Construction
    - GIFT Framework: https://github.com/gift-framework/GIFT
"""

__version__ = '0.1.0-dev'
__author__ = 'Brieuc de La Fourniere'

# Core modules
from . import core
from . import manifolds
from . import networks
from . import training
from . import utils

# Convenience imports
from .utils import (
    G2ForgeConfig,
    TopologyConfig,
    ManifoldConfig,
    create_k7_config,
)

from .manifolds import (
    K7Manifold,
    create_gift_k7,
    create_custom_k7,
)

from .core import (
    build_levi_civita_sparse_7d,
    hodge_star_3,
    compute_exterior_derivative,
    compute_coclosure,
)

__all__ = [
    # Version
    '__version__',

    # Modules
    'core',
    'manifolds',
    'networks',
    'training',
    'utils',

    # Config
    'G2ForgeConfig',
    'TopologyConfig',
    'ManifoldConfig',
    'create_k7_config',

    # Manifolds
    'K7Manifold',
    'create_gift_k7',
    'create_custom_k7',

    # Core operators
    'build_levi_civita_sparse_7d',
    'hodge_star_3',
    'compute_exterior_derivative',
    'compute_coclosure',
]
