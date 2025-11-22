"""
g2-forge: Neural Construction of Exceptional Holonomy Metrics

A universal framework for constructing G‚ metrics on compact 7-manifolds
using Physics-Informed Neural Networks (PINNs).

Not limited to GIFT parameters - works for ANY G‚ manifold!

Quick Start:
    >>> import g2forge as g2
    >>>
    >>> # Create K‡ manifold with custom topology
    >>> manifold = g2.manifolds.create_custom_k7(
    ...     b2_m1=10, b3_m1=38,
    ...     b2_m2=9, b3_m2=35
    ... )
    >>> print(manifold)  # b‚=19, bƒ=73

Example:
    >>> from g2forge.utils import create_k7_config
    >>>
    >>> # Custom configuration
    >>> config = create_k7_config(
    ...     b2_m1=11, b3_m1=40,
    ...     b2_m2=10, b3_m2=37
    ... )
    >>>
    >>> print(f"Total: b‚={config.manifold.topology.b2}")
    Total: b‚=21

References:
    - Joyce (2000): Compact Manifolds with Special Holonomy
    - Kovalev (2003): Twisted Connected Sum Construction
    - GIFT Framework: https://github.com/gift-framework/GIFT
"""

__version__ = '0.1.0-dev'
__author__ = 'Brieuc de La Fournière'

# Core modules
from . import core
from . import manifolds
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
