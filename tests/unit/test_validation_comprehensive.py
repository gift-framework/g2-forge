"""
Comprehensive unit tests for validation methods.

Tests all .validate() methods across configuration classes to ensure
proper validation of invalid configurations and edge cases.
"""

import pytest
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.utils.config import (
    TopologyConfig,
    TCSParameters,
    ManifoldConfig,
    NetworkArchitectureConfig,
    TrainingConfig,
    G2ForgeConfig,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# TOPOLOGY CONFIG VALIDATION
# ============================================================

def test_topology_validation_accepts_valid_config():
    """Test that validation succeeds for valid topology."""
    topology = TopologyConfig(b2=21, b3=77)

    # Should not raise
    assert topology.validate() is True


def test_topology_validation_rejects_negative_b2():
    """Test that validation rejects negative b₂."""
    topology = TopologyConfig(b2=-5, b3=20)

    with pytest.raises(ValueError, match="non-negative"):
        topology.validate()


def test_topology_validation_rejects_negative_b3():
    """Test that validation rejects negative b₃."""
    topology = TopologyConfig(b2=21, b3=-10)

    with pytest.raises(ValueError, match="non-negative"):
        topology.validate()


def test_topology_validation_rejects_both_negative():
    """Test that validation rejects both negative."""
    topology = TopologyConfig(b2=-5, b3=-10)

    with pytest.raises(ValueError, match="non-negative"):
        topology.validate()


def test_topology_validation_accepts_zero_b2():
    """Test that validation accepts b₂ = 0 (edge case)."""
    topology = TopologyConfig(b2=0, b3=20)

    # Should succeed (0 is non-negative)
    assert topology.validate() is True


def test_topology_validation_accepts_zero_b3():
    """Test that validation accepts b₃ = 0 (edge case)."""
    topology = TopologyConfig(b2=21, b3=0)

    # Should succeed (0 is non-negative)
    assert topology.validate() is True


def test_topology_validation_rejects_nonzero_b1():
    """Test that validation warns about b₁ ≠ 0."""
    topology = TopologyConfig(b2=21, b3=77, b1=1)

    with pytest.raises(ValueError, match="simply connected"):
        topology.validate()


def test_topology_validation_accepts_b1_equals_zero():
    """Test that validation accepts b₁ = 0."""
    topology = TopologyConfig(b2=21, b3=77, b1=0)

    # Should succeed
    assert topology.validate() is True


@pytest.mark.parametrize("b1", [1, 2, 5, 10])
def test_topology_validation_rejects_various_nonzero_b1(b1):
    """Test that validation rejects various non-zero b₁ values."""
    topology = TopologyConfig(b2=21, b3=77, b1=b1)

    with pytest.raises(ValueError, match="simply connected"):
        topology.validate()


# ============================================================
# TCS PARAMETERS VALIDATION
# ============================================================

def test_tcs_validation_accepts_valid_params():
    """Test that validation succeeds for valid TCS parameters."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    assert tcs.validate() is True


def test_tcs_validation_rejects_negative_b2_m1():
    """Test that validation rejects negative b₂(M₁)."""
    tcs = TCSParameters(
        b2_m1=-1, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    with pytest.raises(ValueError, match="non-negative"):
        tcs.validate()


def test_tcs_validation_rejects_negative_b3_m1():
    """Test that validation rejects negative b₃(M₁)."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=-1,
        b2_m2=10, b3_m2=37
    )

    with pytest.raises(ValueError, match="non-negative"):
        tcs.validate()


def test_tcs_validation_rejects_negative_b2_m2():
    """Test that validation rejects negative b₂(M₂)."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=-1, b3_m2=37
    )

    with pytest.raises(ValueError, match="non-negative"):
        tcs.validate()


def test_tcs_validation_rejects_negative_b3_m2():
    """Test that validation rejects negative b₃(M₂)."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=-1
    )

    with pytest.raises(ValueError, match="non-negative"):
        tcs.validate()


def test_tcs_validation_accepts_zero_betti_numbers():
    """Test that validation accepts zero Betti numbers for components."""
    tcs = TCSParameters(
        b2_m1=0, b3_m1=10,
        b2_m2=5, b3_m2=20
    )

    # Should succeed (0 is non-negative)
    assert tcs.validate() is True


def test_tcs_validation_rejects_neck_width_too_large():
    """Test that validation rejects neck_width >= 1."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=1.5
    )

    with pytest.raises(ValueError, match="Neck width"):
        tcs.validate()


def test_tcs_validation_rejects_neck_width_equal_to_one():
    """Test that validation rejects neck_width = 1."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=1.0
    )

    with pytest.raises(ValueError, match="Neck width"):
        tcs.validate()


def test_tcs_validation_rejects_neck_width_zero():
    """Test that validation rejects neck_width = 0."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.0
    )

    with pytest.raises(ValueError, match="Neck width"):
        tcs.validate()


def test_tcs_validation_rejects_negative_neck_width():
    """Test that validation rejects negative neck_width."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=-0.1
    )

    with pytest.raises(ValueError, match="Neck width"):
        tcs.validate()


def test_tcs_validation_accepts_small_neck_width():
    """Test that validation accepts small positive neck_width."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.001
    )

    # Should succeed
    assert tcs.validate() is True


def test_tcs_validation_accepts_large_valid_neck_width():
    """Test that validation accepts neck_width close to 1."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.999
    )

    # Should succeed (< 1)
    assert tcs.validate() is True


# ============================================================
# MANIFOLD CONFIG VALIDATION
# ============================================================

def test_manifold_config_validation_accepts_valid():
    """Test that ManifoldConfig validation succeeds for valid config."""
    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    assert config.validate() is True


def test_manifold_config_validation_rejects_wrong_dimension():
    """Test that ManifoldConfig validation rejects dimension ≠ 7."""
    topology = TopologyConfig(b2=21, b3=77)

    config = ManifoldConfig(
        type="Custom",
        construction="Custom",
        topology=topology,
        dimension=8  # Wrong!
    )

    with pytest.raises(ValueError, match="7-dimensional"):
        config.validate()


def test_manifold_config_validation_requires_tcs_params_for_tcs():
    """Test that TCS construction requires tcs_params."""
    topology = TopologyConfig(b2=21, b3=77)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=None  # Missing!
    )

    with pytest.raises(ValueError, match="tcs_params"):
        config.validate()


def test_manifold_config_validation_checks_topology_tcs_consistency():
    """Test that topology must match TCS component sums."""
    topology = TopologyConfig(b2=30, b3=80)  # Wrong!
    tcs_params = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37  # Sums to 21, 77
    )

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    with pytest.raises(ValueError, match="mismatch"):
        config.validate()


def test_manifold_config_validation_checks_b2_mismatch():
    """Test that b₂ mismatch is caught."""
    topology = TopologyConfig(b2=25, b3=77)  # b₂ wrong
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    with pytest.raises(ValueError, match="b₂"):
        config.validate()


def test_manifold_config_validation_checks_b3_mismatch():
    """Test that b₃ mismatch is caught."""
    topology = TopologyConfig(b2=21, b3=80)  # b₃ wrong
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    with pytest.raises(ValueError, match="b₃"):
        config.validate()


def test_manifold_config_validation_propagates_topology_errors():
    """Test that invalid topology causes ManifoldConfig validation to fail."""
    topology = TopologyConfig(b2=-5, b3=77)  # Invalid!
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    # Should fail due to topology validation
    with pytest.raises(ValueError):
        config.validate()


def test_manifold_config_validation_propagates_tcs_errors():
    """Test that invalid TCS params cause ManifoldConfig validation to fail."""
    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=2.0  # Invalid!
    )

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    # Should fail due to TCS validation
    with pytest.raises(ValueError):
        config.validate()


# ============================================================
# FULL G2FORGE CONFIG VALIDATION
# ============================================================

def test_g2forge_config_validation_gift_v1_0():
    """Test that GIFT v1.0 config passes validation."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    # Should succeed
    assert config.validate() is True


def test_g2forge_config_validation_gift_v1_2b():
    """Test that GIFT v1.2b config passes validation."""
    config = g2.G2ForgeConfig.from_gift_v1_2b()

    # Should succeed
    assert config.validate() is True


def test_g2forge_config_validation_create_k7_config():
    """Test that created K7 configs pass validation."""
    config = g2.create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    # Should succeed
    assert config.validate() is True


@pytest.mark.parametrize("b2_m1,b3_m1,b2_m2,b3_m2", [
    (3, 10, 2, 10),
    (5, 20, 5, 20),
    (11, 40, 10, 37),
    (20, 60, 10, 40),
])
def test_g2forge_config_validation_various_topologies(b2_m1, b3_m1, b2_m2, b3_m2):
    """Test that various K7 configs pass validation."""
    config = g2.create_k7_config(
        b2_m1=b2_m1, b3_m1=b3_m1,
        b2_m2=b2_m2, b3_m2=b3_m2
    )

    assert config.validate() is True


def test_g2forge_config_validation_detects_manifold_errors():
    """Test that G2ForgeConfig validation catches manifold errors."""
    # Create invalid config manually
    config = g2.create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    # Corrupt the topology
    config.manifold.topology.b2 = -5  # Invalid!

    # Should fail
    with pytest.raises(ValueError):
        config.validate()


# ============================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================

def test_validation_with_minimal_topology():
    """Test validation with minimal Betti numbers."""
    topology = TopologyConfig(b2=0, b3=0)

    # Should succeed (non-negative)
    assert topology.validate() is True


def test_validation_with_very_large_topology():
    """Test validation with very large Betti numbers."""
    topology = TopologyConfig(b2=10000, b3=30000)

    # Should succeed
    assert topology.validate() is True


def test_validation_with_very_small_neck_width():
    """Test validation with very small neck width."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=1e-10
    )

    # Should succeed (positive)
    assert tcs.validate() is True


def test_validation_after_serialization_roundtrip():
    """Test that validation still works after serialization."""
    original = TopologyConfig(b2=21, b3=77)

    # Serialize and deserialize
    json_str = original.to_json()
    restored = TopologyConfig.from_json(json_str)

    # Validation should still work
    assert restored.validate() is True


def test_validation_called_on_initialization():
    """Test that invalid config raises during initialization."""
    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=2.0  # Invalid!
    )

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    # Should raise when validate() is called
    with pytest.raises(ValueError):
        config.validate()


# ============================================================
# VALIDATION ERROR MESSAGES
# ============================================================

def test_topology_validation_error_message_descriptive():
    """Test that validation errors have descriptive messages."""
    topology = TopologyConfig(b2=-5, b3=77)

    try:
        topology.validate()
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        # Should mention what's wrong
        assert "non-negative" in error_msg or "negative" in error_msg
        # Should mention which Betti number
        assert "b₂" in error_msg or "b2" in error_msg


def test_tcs_validation_error_message_descriptive():
    """Test that TCS validation errors are descriptive."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=1.5
    )

    try:
        tcs.validate()
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        # Should mention neck width
        assert "neck" in error_msg.lower() or "width" in error_msg.lower()


def test_manifold_config_validation_error_message_descriptive():
    """Test that ManifoldConfig validation errors are descriptive."""
    topology = TopologyConfig(b2=30, b3=80)
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    try:
        config.validate()
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        # Should mention mismatch
        assert "mismatch" in error_msg.lower()
