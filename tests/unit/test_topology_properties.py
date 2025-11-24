"""
Unit tests for topology configuration property methods.

Tests all @property methods in TopologyConfig and related classes,
including Poincaré duality, Euler characteristic, and derived properties.
"""

import pytest
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.utils.config import TopologyConfig, TCSParameters, ManifoldConfig


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# POINCARÉ DUALITY PROPERTIES
# ============================================================

def test_poincare_duality_b4_equals_b3():
    """Test Poincaré duality: b₄ = b₃."""
    topology = TopologyConfig(b2=21, b3=77)

    assert topology.b4 == topology.b3
    assert topology.b4 == 77


def test_poincare_duality_b5_equals_b2():
    """Test Poincaré duality: b₅ = b₂."""
    topology = TopologyConfig(b2=21, b3=77)

    assert topology.b5 == topology.b2
    assert topology.b5 == 21


def test_poincare_duality_b6_equals_b1():
    """Test Poincaré duality: b₆ = b₁."""
    topology = TopologyConfig(b2=21, b3=77)

    assert topology.b6 == topology.b1
    assert topology.b6 == 0


def test_poincare_duality_b7_equals_b0():
    """Test Poincaré duality: b₇ = b₀."""
    topology = TopologyConfig(b2=21, b3=77)

    assert topology.b7 == topology.b0
    assert topology.b7 == 1


def test_poincare_duality_all_betti_numbers():
    """Test all Poincaré duality relations together."""
    topology = TopologyConfig(b2=50, b3=150)

    # Direct Betti numbers
    assert topology.b0 == 1
    assert topology.b1 == 0
    assert topology.b2 == 50
    assert topology.b3 == 150

    # Poincaré dual Betti numbers
    assert topology.b4 == 150  # = b₃
    assert topology.b5 == 50   # = b₂
    assert topology.b6 == 0    # = b₁
    assert topology.b7 == 1    # = b₀


@pytest.mark.parametrize("b2,b3", [
    (1, 5),
    (5, 20),
    (10, 40),
    (21, 77),
    (50, 150),
    (100, 300),
])
def test_poincare_duality_multiple_topologies(b2, b3):
    """Test Poincaré duality holds for various topologies."""
    topology = TopologyConfig(b2=b2, b3=b3)

    assert topology.b4 == b3
    assert topology.b5 == b2
    assert topology.b6 == 0
    assert topology.b7 == 1


# ============================================================
# EULER CHARACTERISTIC
# ============================================================

def test_euler_characteristic_formula():
    """Test Euler characteristic formula: χ = 2(b₀ - b₁ + b₂ - b₃)."""
    topology = TopologyConfig(b2=21, b3=77)

    expected_chi = 2 * (1 - 0 + 21 - 77)
    assert topology.euler_characteristic == expected_chi
    assert topology.euler_characteristic == -110


def test_euler_characteristic_gift_topology():
    """Test Euler characteristic for GIFT topology."""
    topology = TopologyConfig(b2=21, b3=77)

    assert topology.euler_characteristic == 2 * (1 - 0 + 21 - 77)
    assert topology.euler_characteristic == -110


@pytest.mark.parametrize("b2,b3,expected_chi", [
    (1, 5, 2 * (1 - 0 + 1 - 5)),      # -6
    (5, 20, 2 * (1 - 0 + 5 - 20)),    # -28
    (10, 40, 2 * (1 - 0 + 10 - 40)),  # -58
    (21, 77, 2 * (1 - 0 + 21 - 77)),  # -110
    (50, 150, 2 * (1 - 0 + 50 - 150)), # -198
])
def test_euler_characteristic_various_topologies(b2, b3, expected_chi):
    """Test Euler characteristic computation for multiple topologies."""
    topology = TopologyConfig(b2=b2, b3=b3)

    assert topology.euler_characteristic == expected_chi


def test_euler_characteristic_negative_for_g2_manifolds():
    """Test that Euler characteristic is typically negative for G₂ manifolds."""
    # For compact G₂ manifolds, typically b₃ > b₂, so χ < 0
    topologies = [
        TopologyConfig(b2=1, b3=5),
        TopologyConfig(b2=5, b3=20),
        TopologyConfig(b2=21, b3=77),
        TopologyConfig(b2=50, b3=150),
    ]

    for topology in topologies:
        assert topology.euler_characteristic < 0, \
            f"Expected negative χ for b₂={topology.b2}, b₃={topology.b3}"


# ============================================================
# TCS PARAMETERS PROPERTIES
# ============================================================

def test_tcs_total_b2_sum():
    """Test that total_b2 equals sum of M₁ and M₂ components."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    assert tcs.total_b2 == 11 + 10
    assert tcs.total_b2 == 21


def test_tcs_total_b3_sum():
    """Test that total_b3 equals sum of M₁ and M₂ components."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    assert tcs.total_b3 == 40 + 37
    assert tcs.total_b3 == 77


@pytest.mark.parametrize("b2_m1,b3_m1,b2_m2,b3_m2", [
    (1, 5, 0, 5),        # Asymmetric
    (3, 10, 2, 10),      # Small
    (5, 20, 5, 20),      # Symmetric medium
    (11, 40, 10, 37),    # GIFT
    (25, 75, 25, 75),    # Large symmetric
])
def test_tcs_total_topology_various_cases(b2_m1, b3_m1, b2_m2, b3_m2):
    """Test TCS total topology for various component choices."""
    tcs = TCSParameters(
        b2_m1=b2_m1, b3_m1=b3_m1,
        b2_m2=b2_m2, b3_m2=b3_m2
    )

    assert tcs.total_b2 == b2_m1 + b2_m2
    assert tcs.total_b3 == b3_m1 + b3_m2


def test_tcs_total_topology_zero_component():
    """Test TCS total topology when one component has b₂=0."""
    tcs = TCSParameters(
        b2_m1=5, b3_m1=20,
        b2_m2=0, b3_m2=15
    )

    assert tcs.total_b2 == 5
    assert tcs.total_b3 == 35


# ============================================================
# MANIFOLD DIMENSION PROPERTY
# ============================================================

def test_manifold_config_dimension_always_seven():
    """Test that ManifoldConfig.dimension is always 7 for G₂."""
    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    assert config.dimension == 7


def test_manifold_config_dimension_default():
    """Test that dimension defaults to 7."""
    topology = TopologyConfig(b2=5, b3=20)
    tcs_params = TCSParameters(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

    # Don't specify dimension
    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    assert config.dimension == 7


def test_manifold_from_k7_config_has_dimension_7():
    """Test that configs created via factory have dimension=7."""
    config = g2.create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    assert config.manifold.dimension == 7


# ============================================================
# PROPERTY CONSISTENCY TESTS
# ============================================================

def test_topology_properties_consistent_with_manual_calculation():
    """Test that properties match manual calculations."""
    topology = TopologyConfig(b2=10, b3=40)

    # Manually calculate
    manual_b4 = topology.b3  # Poincaré duality
    manual_b5 = topology.b2
    manual_chi = 2 * (topology.b0 - topology.b1 + topology.b2 - topology.b3)

    # Check against properties
    assert topology.b4 == manual_b4
    assert topology.b5 == manual_b5
    assert topology.euler_characteristic == manual_chi


def test_tcs_properties_consistent_with_manual_calculation():
    """Test that TCS properties match manual sums."""
    tcs = TCSParameters(
        b2_m1=7, b3_m1=25,
        b2_m2=8, b3_m2=30
    )

    # Manually calculate
    manual_total_b2 = tcs.b2_m1 + tcs.b2_m2
    manual_total_b3 = tcs.b3_m1 + tcs.b3_m2

    # Check against properties
    assert tcs.total_b2 == manual_total_b2
    assert tcs.total_b3 == manual_total_b3


def test_topology_properties_are_read_only():
    """Test that computed properties cannot be assigned to."""
    topology = TopologyConfig(b2=21, b3=77)

    # Properties should be read-only
    with pytest.raises(AttributeError):
        topology.b4 = 100

    with pytest.raises(AttributeError):
        topology.euler_characteristic = 0


def test_tcs_properties_are_read_only():
    """Test that TCS computed properties cannot be assigned to."""
    tcs = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    # Properties should be read-only
    with pytest.raises(AttributeError):
        tcs.total_b2 = 100

    with pytest.raises(AttributeError):
        tcs.total_b3 = 100


# ============================================================
# EDGE CASES
# ============================================================

def test_topology_properties_with_minimal_betti_numbers():
    """Test properties with minimal Betti numbers."""
    topology = TopologyConfig(b2=1, b3=1)

    assert topology.b4 == 1
    assert topology.b5 == 1
    assert topology.euler_characteristic == 2 * (1 - 0 + 1 - 1)
    assert topology.euler_characteristic == 2


def test_topology_properties_with_large_betti_numbers():
    """Test properties with very large Betti numbers."""
    topology = TopologyConfig(b2=1000, b3=3000)

    assert topology.b4 == 3000
    assert topology.b5 == 1000
    assert topology.euler_characteristic == 2 * (1 - 0 + 1000 - 3000)
    assert topology.euler_characteristic == -3998


def test_tcs_properties_symmetric_components():
    """Test TCS properties when M₁ and M₂ are symmetric."""
    tcs = TCSParameters(
        b2_m1=10, b3_m1=40,
        b2_m2=10, b3_m2=40
    )

    assert tcs.total_b2 == 20
    assert tcs.total_b3 == 80
    assert tcs.b2_m1 == tcs.b2_m2
    assert tcs.b3_m1 == tcs.b3_m2


def test_tcs_properties_highly_asymmetric_components():
    """Test TCS properties with highly asymmetric components."""
    tcs = TCSParameters(
        b2_m1=50, b3_m1=150,
        b2_m2=1, b3_m2=5
    )

    assert tcs.total_b2 == 51
    assert tcs.total_b3 == 155
    assert tcs.b2_m1 // tcs.b2_m2 == 50  # 50x larger


# ============================================================
# INTEGRATION WITH FULL CONFIGS
# ============================================================

def test_g2forge_config_topology_properties_accessible():
    """Test that topology properties are accessible from full config."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    # Should be able to access topology properties
    assert config.manifold.topology.b2 == 21
    assert config.manifold.topology.b3 == 77
    assert config.manifold.topology.b4 == 77
    assert config.manifold.topology.b5 == 21
    assert config.manifold.topology.euler_characteristic == -110


def test_g2forge_config_tcs_properties_accessible():
    """Test that TCS properties are accessible from full config."""
    config = g2.create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    # Should be able to access TCS properties
    assert config.manifold.tcs_params.total_b2 == 21
    assert config.manifold.tcs_params.total_b3 == 77
    assert config.manifold.tcs_params.b2_m1 == 11
    assert config.manifold.tcs_params.b3_m1 == 40


def test_properties_preserved_after_serialization():
    """Test that properties work correctly after serialization."""
    original = TopologyConfig(b2=21, b3=77)

    # Serialize and deserialize
    json_str = original.to_json()
    restored = TopologyConfig.from_json(json_str)

    # Properties should still work
    assert restored.b4 == 77
    assert restored.b5 == 21
    assert restored.euler_characteristic == -110
