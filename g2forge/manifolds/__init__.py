"""
G‚ manifold implementations.

Provides abstract base classes and concrete implementations
for various G‚ manifolds (K‡, Joyce, custom).
"""

from .base import (
    Cycle,
    Manifold,
    TCSManifold,
    create_manifold,
)

from .k7 import (
    K7Manifold,
    create_gift_k7,
    create_custom_k7,
)

__all__ = [
    # Base classes
    'Cycle',
    'Manifold',
    'TCSManifold',
    'create_manifold',
    # K7 implementation
    'K7Manifold',
    'create_gift_k7',
    'create_custom_k7',
]
