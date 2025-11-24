"""
Physics modules for G₂ metric construction

This module contains physics-informed components:
- Volume normalization (from GIFT v1.2a/b)
- RG flow computation (GIFT 2.1)
- Spectral analysis and fractality detection
"""

from .volume_normalizer import VolumeNormalizer
from .rg_flow import RGFlowModule

__all__ = [
    'VolumeNormalizer',
    'RGFlowModule',
]
