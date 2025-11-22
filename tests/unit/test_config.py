"""
Unit tests for configuration system.

Tests TopologyConfig, TCSParameters, ManifoldConfig, and G2ForgeConfig
for correctness, validation, and serialization.
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.utils.config import (
    TopologyConfig,
    TCSParameters,
    ManifoldConfig,
    G2ForgeConfig,
    create_k7_config,
)


# ============================================================
# TOPOLOGY CONFIG TESTS
# ============================================================

def test_topology_config_initialization():
    """Test that TopologyConfig initializes correctly."""
    topology = TopologyConfig(b2=21, b3=77)

    assert topology.b2 == 21
    assert topology.b3 == 77
    assert topology.b1 == 0
    assert topology.b0 == 1


def test_topology_config_poincare_duality():
    """Test that Poincaré duality properties hold."""
    topology = TopologyConfig(b2=21, b3=77)

    # b₄ = b₃ by Poincaré duality
    assert topology.b4 == topology.b3
    assert topology.b4 == 77

    # b₅ = b₂ by Poincaré duality
    assert topology.b5 == topology.b2
    assert topology.b5 == 21

    # b₆ = b₁
    assert topology.b6 == topology.b1
    assert topology.b6 == 0

    # b₇ = b₀
    assert topology.b7 == topology.b0
    assert topology.b7 == 1


def test_topology_config_euler_characteristic():
    """Test Euler characteristic computation."""
    topology = TopologyConfig(b2=21, b3=77)

    # χ = 2(b₀ - b₁ + b₂ - b₃)
    expected_chi = 2 * (1 - 0 + 21 - 77)
    assert topology.euler_characteristic == expected_chi
    assert topology.euler_characteristic == -110


def test_topology_config_validation_negative_betti():
    """Test that validation rejects negative Betti numbers."""
    topology = TopologyConfig(b2=-5, b3=20)

    with pytest.raises(ValueError, match="non-negative"):
        topology.validate()


def test_topology_config_validation_nonzero_b1():
    """Test that validation warns about non-zero b₁."""
    topology = TopologyConfig(b2=21, b3=77, b1=1)

    with pytest.raises(ValueError, match="simply connected"):
        topology.validate()


def test_topology_config_validation_success():
    """Test that validation succeeds for valid topology."""
    topology = TopologyConfig(b2=21, b3=77)

    assert topology.validate() is True


# ============================================================
# TCS PARAMETERS TESTS
# ============================================================

def test_tcs_parameters_initialization():
    """Test that TCSParameters initializes correctly."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    assert tcs.b2_m1 == 11
    assert tcs.b3_m1 == 40
    assert tcs.b2_m2 == 10
    assert tcs.b3_m2 == 37


def test_tcs_parameters_total_topology():
    """
    CRITICAL TEST: Verify that total topology is sum of M₁ and M₂.

    This is the fundamental TCS construction property.
    """
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    assert tcs.total_b2 == 11 + 10
    assert tcs.total_b2 == 21
    assert tcs.total_b3 == 40 + 37
    assert tcs.total_b3 == 77


def test_tcs_parameters_validation_negative():
    """Test that validation rejects negative Betti numbers."""
    tcs = TCSParameters(
        b2_m1=-1, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    with pytest.raises(ValueError, match="non-negative"):
        tcs.validate()


def test_tcs_parameters_validation_invalid_neck_width():
    """Test that validation rejects invalid neck widths."""
    # Neck width too large
    tcs1 = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=1.5  # > 1
    )

    with pytest.raises(ValueError, match="Neck width"):
        tcs1.validate()

    # Neck width negative
    tcs2 = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=-0.1
    )

    with pytest.raises(ValueError, match="Neck width"):
        tcs2.validate()


def test_tcs_parameters_default_values():
    """Test that TCS parameters have sensible defaults."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    assert tcs.neck_width == 0.125  # GIFT default
    assert tcs.neck_center == 0.5
    assert tcs.transition_sharpness == 10.0


# ============================================================
# MANIFOLD CONFIG TESTS
# ============================================================

def test_manifold_config_initialization():
    """Test that ManifoldConfig initializes correctly."""
    topology = TopologyConfig(b2=21, b3=77)
    tcs = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs
    )

    assert manifold.type == "K7"
    assert manifold.construction == "TCS"
    assert manifold.dimension == 7


def test_manifold_config_tcs_consistency():
    """
    CRITICAL TEST: Verify TCS topology consistency validation.

    The total topology must match M₁ + M₂.
    """
    topology = TopologyConfig(b2=21, b3=77)
    tcs = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs
    )

    # Should validate successfully
    assert manifold.validate() is True


def test_manifold_config_tcs_mismatch_b2():
    """Test that TCS topology mismatch is detected for b₂."""
    # Inconsistent: b₂ = 21 but M₁ + M₂ = 20
    topology = TopologyConfig(b2=21, b3=77)
    tcs = TCSParameters(b2_m1=10, b3_m1=40, b2_m2=10, b3_m2=37)

    manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs
    )

    with pytest.raises(ValueError, match="b₂.*mismatch"):
        manifold.validate()


def test_manifold_config_tcs_mismatch_b3():
    """Test that TCS topology mismatch is detected for b₃."""
    # Inconsistent: b₃ = 77 but M₁ + M₂ = 76
    topology = TopologyConfig(b2=21, b3=77)
    tcs = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=36)

    manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs
    )

    with pytest.raises(ValueError, match="b₃.*mismatch"):
        manifold.validate()


def test_manifold_config_missing_tcs_params():
    """Test that TCS construction requires tcs_params."""
    topology = TopologyConfig(b2=21, b3=77)

    manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=None  # Missing!
    )

    with pytest.raises(ValueError, match="requires tcs_params"):
        manifold.validate()


def test_manifold_config_wrong_dimension():
    """Test that non-7D manifolds are rejected."""
    topology = TopologyConfig(b2=21, b3=77)

    manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        dimension=5  # Wrong!
    )

    with pytest.raises(ValueError, match="7-dimensional"):
        manifold.validate()


# ============================================================
# G2FORGE CONFIG TESTS
# ============================================================

def test_g2forge_config_from_gift_v1_0():
    """Test that GIFT v1.0 config is created correctly."""
    config = G2ForgeConfig.from_gift_v1_0()

    # Check topology
    assert config.manifold.topology.b2 == 21
    assert config.manifold.topology.b3 == 77

    # Check TCS parameters
    assert config.manifold.tcs_params.b2_m1 == 11
    assert config.manifold.tcs_params.b3_m1 == 40
    assert config.manifold.tcs_params.b2_m2 == 10
    assert config.manifold.tcs_params.b3_m2 == 37

    # Check construction type
    assert config.manifold.type == "K7"
    assert config.manifold.construction == "TCS"

    # Check version
    assert config.version == "gift-v1.0-reproduction"


def test_g2forge_config_gift_has_curriculum():
    """Test that GIFT config has all 5 curriculum phases."""
    config = G2ForgeConfig.from_gift_v1_0()

    # Should have 5 phases
    assert len(config.training.curriculum) == 5

    phase_names = list(config.training.curriculum.keys())
    assert "phase1_neck_stability" in phase_names
    assert "phase2_acyl_matching" in phase_names
    assert "phase3_cohomology_refinement" in phase_names
    assert "phase4_harmonic_extraction" in phase_names
    assert "phase5_calibration_finetune" in phase_names


def test_g2forge_config_validation():
    """Test that config validation cascades."""
    config = G2ForgeConfig.from_gift_v1_0()

    # Should validate successfully
    assert config.validate() is True


def test_create_k7_config_factory():
    """Test that create_k7_config factory works."""
    config = create_k7_config(
        b2_m1=10, b3_m1=38,
        b2_m2=9, b3_m2=35
    )

    assert config.manifold.topology.b2 == 19
    assert config.manifold.topology.b3 == 73
    assert config.manifold.tcs_params.b2_m1 == 10
    assert config.manifold.tcs_params.b3_m1 == 38


def test_create_k7_config_defaults():
    """Test that create_k7_config uses GIFT defaults when not specified."""
    config = create_k7_config()

    # Should use GIFT defaults
    assert config.manifold.tcs_params.b2_m1 == 11
    assert config.manifold.tcs_params.b3_m1 == 40
    assert config.manifold.tcs_params.b2_m2 == 10
    assert config.manifold.tcs_params.b3_m2 == 37


# ============================================================
# SERIALIZATION TESTS
# ============================================================

def test_config_to_dict():
    """Test that config converts to dictionary."""
    config = G2ForgeConfig.from_gift_v1_0()

    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert 'manifold' in config_dict
    assert 'architecture' in config_dict
    assert 'training' in config_dict


def test_config_json_serialization():
    """Test that config can be saved and loaded from JSON."""
    config = create_k7_config(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Save to JSON
        config.to_json(temp_path)

        # Load from JSON
        loaded_config = G2ForgeConfig.from_json(temp_path)

        # Check that key values match
        assert loaded_config.manifold.topology.b2 == config.manifold.topology.b2
        assert loaded_config.manifold.topology.b3 == config.manifold.topology.b3
        assert loaded_config.manifold.tcs_params.b2_m1 == config.manifold.tcs_params.b2_m1
    finally:
        temp_path.unlink()


def test_config_dict_roundtrip():
    """Test that config can be converted to dict and back."""
    config = G2ForgeConfig.from_gift_v1_0()

    # Convert to dict
    config_dict = config.to_dict()

    # Convert back
    loaded_config = G2ForgeConfig.from_dict(config_dict)

    # Check topology
    assert loaded_config.manifold.topology.b2 == 21
    assert loaded_config.manifold.topology.b3 == 77
    assert loaded_config.manifold.tcs_params.total_b2 == 21
    assert loaded_config.manifold.tcs_params.total_b3 == 77


# ============================================================
# UNIVERSALITY TESTS
# ============================================================

@pytest.mark.parametrize("b2_m1,b3_m1,b2_m2,b3_m2", [
    (5, 20, 5, 20),      # Small
    (11, 40, 10, 37),    # GIFT
    (15, 50, 15, 50),    # Large
])
def test_config_universality(b2_m1, b3_m1, b2_m2, b3_m2):
    """
    CRITICAL TEST: Verify that config system works for ANY topology.

    This tests the key universality feature.
    """
    config = create_k7_config(
        b2_m1=b2_m1, b3_m1=b3_m1,
        b2_m2=b2_m2, b3_m2=b3_m2
    )

    # Validate
    assert config.validate() is True

    # Check topology consistency
    assert config.manifold.topology.b2 == b2_m1 + b2_m2
    assert config.manifold.topology.b3 == b3_m1 + b3_m2
    assert config.manifold.tcs_params.total_b2 == b2_m1 + b2_m2
    assert config.manifold.tcs_params.total_b3 == b3_m1 + b3_m2


def test_config_different_topologies_create_different_configs():
    """Test that different topologies produce different configs."""
    config1 = create_k7_config(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)
    config2 = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    assert config1.manifold.topology.b2 != config2.manifold.topology.b2
    assert config1.manifold.topology.b3 != config2.manifold.topology.b3


def test_network_architecture_auto_sizing():
    """Test that network architecture auto-sizes from topology."""
    config = create_k7_config(b2_m1=10, b3_m1=35, b2_m2=8, b3_m2=30)

    topology = config.manifold.topology
    arch = config.architecture

    # Check that output dimensions adapt to topology
    assert arch.get_h2_output_dim(topology) == topology.b2
    assert arch.get_h3_output_dim(topology) == topology.b3
    assert arch.get_h2_output_dim(topology) == 18  # 10 + 8
    assert arch.get_h3_output_dim(topology) == 65  # 35 + 30
