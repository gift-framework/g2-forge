"""
Unit tests for abstract manifold base classes.

Tests the Manifold, TCSManifold abstract base classes and Cycle dataclass
to ensure proper abstraction enforcement and extensibility.
"""

import pytest
import torch
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.manifolds.base import Manifold, TCSManifold, Cycle, create_manifold
from g2forge.utils.config import ManifoldConfig, TopologyConfig, TCSParameters


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# CYCLE DATACLASS TESTS
# ============================================================

def test_cycle_creation_associative():
    """Test Cycle creation for associative 3-cycle."""
    cycle = Cycle(
        type="associative",
        dimension=3,
        indices=(0, 1, 2),
        volume=1.0
    )

    assert cycle.type == "associative"
    assert cycle.dimension == 3
    assert cycle.indices == (0, 1, 2)
    assert cycle.volume == 1.0


def test_cycle_creation_coassociative():
    """Test Cycle creation for coassociative 4-cycle."""
    cycle = Cycle(
        type="coassociative",
        dimension=4,
        indices=(0, 1, 2, 3),
        volume=2.0
    )

    assert cycle.type == "coassociative"
    assert cycle.dimension == 4
    assert cycle.indices == (0, 1, 2, 3)
    assert cycle.volume == 2.0


def test_cycle_default_volume():
    """Test Cycle has default volume of 1.0."""
    cycle = Cycle(
        type="associative",
        dimension=3,
        indices=(0, 1, 2)
    )

    assert cycle.volume == 1.0


def test_cycle_with_parametrization():
    """Test Cycle with optional parametrization function."""
    def param_func(t):
        return torch.cos(t)

    cycle = Cycle(
        type="associative",
        dimension=3,
        indices=(0, 1, 2),
        parametrization=param_func
    )

    assert cycle.parametrization is not None
    assert callable(cycle.parametrization)
    result = cycle.parametrization(0.0)
    assert result == torch.cos(torch.tensor(0.0))


# ============================================================
# MANIFOLD ABSTRACT CLASS TESTS
# ============================================================

def test_manifold_cannot_be_instantiated_directly():
    """Test that Manifold abstract class cannot be instantiated."""
    topology = TopologyConfig(b2=5, b3=20)
    config = ManifoldConfig(
        type="TestManifold",
        construction="Custom",
        topology=topology
    )

    # Should raise TypeError because Manifold has abstract methods
    with pytest.raises(TypeError, match="abstract"):
        Manifold(config)


def test_manifold_subclass_must_implement_abstract_methods():
    """Test that Manifold subclasses must implement all abstract methods."""

    # Create incomplete subclass
    class IncompleteManifold(Manifold):
        # Missing all abstract method implementations
        pass

    topology = TopologyConfig(b2=5, b3=20)
    config = ManifoldConfig(
        type="TestManifold",
        construction="Custom",
        topology=topology
    )

    # Should raise TypeError for missing abstract methods
    with pytest.raises(TypeError):
        IncompleteManifold(config)


def test_manifold_subclass_with_all_methods_can_be_instantiated():
    """Test that Manifold subclass with all methods can be instantiated."""

    class CompleteManifold(Manifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            batch_size = coords.shape[0]
            return {'default': torch.ones(batch_size)}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=5, b3=20)
    config = ManifoldConfig(
        type="TestManifold",
        construction="Custom",
        topology=topology
    )

    # Should succeed
    manifold = CompleteManifold(config)
    assert manifold is not None
    assert manifold.b2 == 5
    assert manifold.b3 == 20


def test_manifold_properties_accessible():
    """Test that Manifold properties are accessible from subclass."""

    class TestManifold(Manifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'default': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    config = ManifoldConfig(
        type="TestManifold",
        construction="Custom",
        topology=topology
    )

    manifold = TestManifold(config)

    # Test all properties
    assert manifold.topology == topology
    assert manifold.b2 == 21
    assert manifold.b3 == 77
    assert manifold.dimension == 7
    assert manifold.euler_characteristic == 2 * (1 - 0 + 21 - 77)


def test_manifold_repr():
    """Test Manifold string representation."""

    class TestManifold(Manifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'default': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    config = ManifoldConfig(
        type="TestManifold",
        construction="Custom",
        topology=topology
    )

    manifold = TestManifold(config)
    repr_str = repr(manifold)

    assert "TestManifold" in repr_str
    assert "b₂=21" in repr_str
    assert "b₃=77" in repr_str


# ============================================================
# TCS MANIFOLD BASE CLASS TESTS
# ============================================================

def test_tcs_manifold_requires_tcs_construction():
    """Test that TCSManifold requires construction='TCS'."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'m1': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    # Wrong construction type
    config = ManifoldConfig(
        type="K7",
        construction="Custom",  # Should be "TCS"
        topology=topology,
        tcs_params=tcs_params
    )

    with pytest.raises(ValueError, match="TCS"):
        TestTCSManifold(config)


def test_tcs_manifold_requires_tcs_params():
    """Test that TCSManifold requires tcs_params in config."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'m1': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)

    # Missing tcs_params
    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=None  # Missing!
    )

    with pytest.raises(ValueError, match="tcs_params"):
        TestTCSManifold(config)


def test_tcs_manifold_properties():
    """Test TCSManifold exposes TCS-specific properties."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'m1': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.125
    )

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    manifold = TestTCSManifold(config)

    # Test TCS-specific properties
    assert manifold.b2_m1 == 11
    assert manifold.b3_m1 == 40
    assert manifold.b2_m2 == 10
    assert manifold.b3_m2 == 37
    assert manifold.neck_width == 0.125


def test_tcs_manifold_region_indicator_m1():
    """Test TCSManifold.compute_region_indicator() for M₁ region."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'m1': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.2,
        neck_center=0.5
    )

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    manifold = TestTCSManifold(config)

    # Test points in M₁ region (t < 0.4)
    t = torch.tensor([0.0, 0.1, 0.2, 0.3])
    indicator = manifold.compute_region_indicator(t, region='m1')

    # Should be close to 1 for small t
    assert indicator.shape == (4,)
    assert torch.all(indicator > 0.5), "M₁ indicator should be high for t < neck"


def test_tcs_manifold_region_indicator_neck():
    """Test TCSManifold.compute_region_indicator() for neck region."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'neck': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.2,
        neck_center=0.5
    )

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    manifold = TestTCSManifold(config)

    # Test point at neck center
    t = torch.tensor([0.5])
    indicator = manifold.compute_region_indicator(t, region='neck')

    # Should be high at center of neck
    assert indicator.shape == (1,)
    assert indicator[0] > 0.5, "Neck indicator should be high at center"


def test_tcs_manifold_region_indicator_m2():
    """Test TCSManifold.compute_region_indicator() for M₂ region."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'m2': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.2,
        neck_center=0.5
    )

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    manifold = TestTCSManifold(config)

    # Test points in M₂ region (t > 0.6)
    t = torch.tensor([0.7, 0.8, 0.9, 1.0])
    indicator = manifold.compute_region_indicator(t, region='m2')

    # Should be close to 1 for large t
    assert indicator.shape == (4,)
    assert torch.all(indicator > 0.5), "M₂ indicator should be high for t > neck"


def test_tcs_manifold_region_indicator_invalid_region():
    """Test TCSManifold.compute_region_indicator() raises for invalid region."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'m1': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    manifold = TestTCSManifold(config)

    t = torch.tensor([0.5])

    with pytest.raises(ValueError, match="Unknown region"):
        manifold.compute_region_indicator(t, region='invalid_region')


def test_tcs_manifold_region_indicator_sharpness():
    """Test TCSManifold.compute_region_indicator() sharpness parameter."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'m1': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.2
    )

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    manifold = TestTCSManifold(config)

    # Test transition point
    t = torch.tensor([0.4])  # At boundary

    # Lower sharpness = smoother transition
    indicator_smooth = manifold.compute_region_indicator(t, region='m1', sharpness=1.0)

    # Higher sharpness = sharper transition
    indicator_sharp = manifold.compute_region_indicator(t, region='m1', sharpness=100.0)

    # Both should be in [0, 1]
    assert 0 <= indicator_smooth <= 1
    assert 0 <= indicator_sharp <= 1


def test_tcs_manifold_repr():
    """Test TCSManifold string representation."""

    class TestTCSManifold(TCSManifold):
        def sample_coordinates(self, n_samples, grid_n=None, device='cpu'):
            return torch.randn(n_samples, 7, device=device)

        def get_region_weights(self, coords):
            return {'m1': torch.ones(coords.shape[0])}

        def get_associative_cycles(self):
            return []

        def get_coassociative_cycles(self):
            return []

    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    manifold = TestTCSManifold(config)
    repr_str = repr(manifold)

    # Should include TCS-specific info
    assert "M₁" in repr_str or "M1" in repr_str
    assert "M₂" in repr_str or "M2" in repr_str
    assert "11" in repr_str  # b2_m1
    assert "40" in repr_str  # b3_m1
