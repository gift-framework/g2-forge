"""
Unit tests for configuration serialization and deserialization.

Tests JSON, YAML, and dict conversion for all config dataclasses.
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
    ModuliParameters,
    ManifoldConfig,
    NetworkArchitectureConfig,
    TrainingConfig,
    G2ForgeConfig,
    create_k7_config,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# TOPOLOGY CONFIG SERIALIZATION
# ============================================================

def test_topology_config_to_dict():
    """Test TopologyConfig.to_dict() includes all fields."""
    topology = TopologyConfig(b2=21, b3=77, b1=0, b0=1)

    config_dict = topology.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict['b2'] == 21
    assert config_dict['b3'] == 77
    assert config_dict['b1'] == 0
    assert config_dict['b0'] == 1


def test_topology_config_from_dict():
    """Test TopologyConfig.from_dict() reconstruction."""
    original = TopologyConfig(b2=21, b3=77)
    config_dict = original.to_dict()

    restored = TopologyConfig.from_dict(config_dict)

    assert restored.b2 == original.b2
    assert restored.b3 == original.b3
    assert restored.b1 == original.b1
    assert restored.b0 == original.b0


def test_topology_config_dict_roundtrip():
    """Test TopologyConfig dict roundtrip preserves all values."""
    original = TopologyConfig(b2=50, b3=150)

    # Roundtrip
    config_dict = original.to_dict()
    restored = TopologyConfig.from_dict(config_dict)

    # Verify all properties match
    assert restored.b2 == original.b2
    assert restored.b3 == original.b3
    assert restored.euler_characteristic == original.euler_characteristic


def test_topology_config_to_json():
    """Test TopologyConfig.to_json() produces valid JSON."""
    topology = TopologyConfig(b2=21, b3=77)

    json_str = topology.to_json()

    # Should be valid JSON
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert parsed['b2'] == 21
    assert parsed['b3'] == 77


def test_topology_config_from_json():
    """Test TopologyConfig.from_json() reconstruction."""
    original = TopologyConfig(b2=21, b3=77)
    json_str = original.to_json()

    restored = TopologyConfig.from_json(json_str)

    assert restored.b2 == original.b2
    assert restored.b3 == original.b3


def test_topology_config_json_roundtrip():
    """Test TopologyConfig JSON roundtrip preserves values."""
    original = TopologyConfig(b2=10, b3=40)

    json_str = original.to_json()
    restored = TopologyConfig.from_json(json_str)

    assert restored.b2 == original.b2
    assert restored.b3 == original.b3
    assert restored.b1 == original.b1
    assert restored.b0 == original.b0


def test_topology_config_to_yaml():
    """Test TopologyConfig.to_yaml() produces valid YAML."""
    topology = TopologyConfig(b2=21, b3=77)

    yaml_str = topology.to_yaml()

    # Should be string containing YAML
    assert isinstance(yaml_str, str)
    assert 'b2' in yaml_str
    assert 'b3' in yaml_str


def test_topology_config_from_yaml():
    """Test TopologyConfig.from_yaml() reconstruction."""
    original = TopologyConfig(b2=21, b3=77)
    yaml_str = original.to_yaml()

    restored = TopologyConfig.from_yaml(yaml_str)

    assert restored.b2 == original.b2
    assert restored.b3 == original.b3


def test_topology_config_yaml_roundtrip():
    """Test TopologyConfig YAML roundtrip preserves values."""
    original = TopologyConfig(b2=5, b3=20)

    yaml_str = original.to_yaml()
    restored = TopologyConfig.from_yaml(yaml_str)

    assert restored.b2 == original.b2
    assert restored.b3 == original.b3


# ============================================================
# TCS PARAMETERS SERIALIZATION
# ============================================================

def test_tcs_parameters_to_dict():
    """Test TCSParameters.to_dict() includes all fields."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.125
    )

    config_dict = tcs.to_dict()

    assert config_dict['b2_m1'] == 11
    assert config_dict['b3_m1'] == 40
    assert config_dict['b2_m2'] == 10
    assert config_dict['b3_m2'] == 37
    assert config_dict['neck_width'] == 0.125


def test_tcs_parameters_from_dict():
    """Test TCSParameters.from_dict() reconstruction."""
    original = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)
    config_dict = original.to_dict()

    restored = TCSParameters.from_dict(config_dict)

    assert restored.b2_m1 == original.b2_m1
    assert restored.b3_m1 == original.b3_m1
    assert restored.b2_m2 == original.b2_m2
    assert restored.b3_m2 == original.b3_m2


def test_tcs_parameters_json_roundtrip():
    """Test TCSParameters JSON roundtrip."""
    original = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.15
    )

    json_str = original.to_json()
    restored = TCSParameters.from_json(json_str)

    assert restored.b2_m1 == original.b2_m1
    assert restored.b3_m1 == original.b3_m1
    assert restored.neck_width == pytest.approx(original.neck_width)


def test_tcs_parameters_yaml_roundtrip():
    """Test TCSParameters YAML roundtrip."""
    original = TCSParameters(
        b2_m1=5, b3_m1=20,
        b2_m2=5, b3_m2=20,
        neck_width=0.1,
        neck_center=0.5
    )

    yaml_str = original.to_yaml()
    restored = TCSParameters.from_yaml(yaml_str)

    assert restored.b2_m1 == original.b2_m1
    assert restored.neck_center == pytest.approx(original.neck_center)


# ============================================================
# MODULI PARAMETERS SERIALIZATION
# ============================================================

def test_moduli_parameters_to_dict():
    """Test ModuliParameters.to_dict() includes params dict."""
    moduli = ModuliParameters(params={'alpha': 1.5, 'beta': 2.0})

    config_dict = moduli.to_dict()

    assert 'params' in config_dict
    assert config_dict['params']['alpha'] == 1.5
    assert config_dict['params']['beta'] == 2.0


def test_moduli_parameters_from_dict():
    """Test ModuliParameters.from_dict() reconstruction."""
    original = ModuliParameters(params={'gamma': 3.0, 'delta': 4.0})
    config_dict = original.to_dict()

    restored = ModuliParameters.from_dict(config_dict)

    assert restored['gamma'] == original['gamma']
    assert restored['delta'] == original['delta']


def test_moduli_parameters_json_roundtrip():
    """Test ModuliParameters JSON roundtrip."""
    original = ModuliParameters(params={'param1': 1.0, 'param2': 2.0})

    json_str = original.to_json()
    restored = ModuliParameters.from_json(json_str)

    assert restored['param1'] == pytest.approx(original['param1'])
    assert restored['param2'] == pytest.approx(original['param2'])


def test_moduli_parameters_empty():
    """Test ModuliParameters with no params."""
    moduli = ModuliParameters()

    json_str = moduli.to_json()
    restored = ModuliParameters.from_json(json_str)

    assert len(restored.params) == 0


# ============================================================
# MANIFOLD CONFIG SERIALIZATION
# ============================================================

def test_manifold_config_to_dict():
    """Test ManifoldConfig.to_dict() includes nested structures."""
    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    config_dict = config.to_dict()

    assert config_dict['type'] == "K7"
    assert config_dict['construction'] == "TCS"
    assert config_dict['topology']['b2'] == 21
    assert config_dict['tcs_params']['b2_m1'] == 11


def test_manifold_config_from_dict():
    """Test ManifoldConfig.from_dict() with nested configs."""
    original_topology = TopologyConfig(b2=21, b3=77)
    original_tcs = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    original = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=original_topology,
        tcs_params=original_tcs
    )

    config_dict = original.to_dict()
    restored = ManifoldConfig.from_dict(config_dict)

    assert restored.type == original.type
    assert restored.topology.b2 == original.topology.b2
    assert restored.tcs_params.b2_m1 == original.tcs_params.b2_m1


def test_manifold_config_json_roundtrip():
    """Test ManifoldConfig JSON roundtrip preserves nested structures."""
    topology = TopologyConfig(b2=5, b3=20)
    tcs_params = TCSParameters(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

    original = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    json_str = original.to_json()
    restored = ManifoldConfig.from_json(json_str)

    assert restored.type == original.type
    assert restored.topology.b2 == original.topology.b2
    assert restored.tcs_params.b2_m1 == original.tcs_params.b2_m1


def test_manifold_config_yaml_roundtrip():
    """Test ManifoldConfig YAML roundtrip."""
    topology = TopologyConfig(b2=10, b3=40)
    tcs_params = TCSParameters(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    original = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    yaml_str = original.to_yaml()
    restored = ManifoldConfig.from_yaml(yaml_str)

    assert restored.type == original.type
    assert restored.topology.b3 == original.topology.b3


# ============================================================
# FULL G2FORGE CONFIG SERIALIZATION
# ============================================================

def test_g2forge_config_to_dict():
    """Test G2ForgeConfig.to_dict() includes all subsystems."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config_dict = config.to_dict()

    assert 'manifold' in config_dict
    assert 'architecture' in config_dict
    assert 'training' in config_dict
    assert config_dict['manifold']['topology']['b2'] == 21


def test_g2forge_config_from_dict():
    """Test G2ForgeConfig.from_dict() reconstruction."""
    original = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)
    config_dict = original.to_dict()

    restored = G2ForgeConfig.from_dict(config_dict)

    assert restored.manifold.topology.b2 == original.manifold.topology.b2
    assert restored.manifold.topology.b3 == original.manifold.topology.b3


def test_g2forge_config_json_roundtrip():
    """Test G2ForgeConfig JSON roundtrip."""
    original = create_k7_config(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    json_str = original.to_json()
    restored = G2ForgeConfig.from_json(json_str)

    assert restored.manifold.topology.b2 == original.manifold.topology.b2
    assert restored.manifold.topology.b3 == original.manifold.topology.b3
    assert restored.manifold.type == original.manifold.type


def test_g2forge_config_yaml_roundtrip():
    """Test G2ForgeConfig YAML roundtrip."""
    original = G2ForgeConfig.from_gift_v1_0()

    yaml_str = original.to_yaml()
    restored = G2ForgeConfig.from_yaml(yaml_str)

    assert restored.manifold.topology.b2 == 21
    assert restored.manifold.topology.b3 == 77


def test_g2forge_config_save_load_json_file():
    """Test saving and loading G2ForgeConfig from JSON file."""
    original = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "config.json"

        # Save to file
        original.save_json(str(filepath))

        # Load from file
        restored = G2ForgeConfig.load_json(str(filepath))

        assert restored.manifold.topology.b2 == original.manifold.topology.b2
        assert restored.manifold.topology.b3 == original.manifold.topology.b3


def test_g2forge_config_save_load_yaml_file():
    """Test saving and loading G2ForgeConfig from YAML file."""
    original = G2ForgeConfig.from_gift_v1_0()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "config.yaml"

        # Save to file
        original.save_yaml(str(filepath))

        # Load from file
        restored = G2ForgeConfig.load_yaml(str(filepath))

        assert restored.manifold.topology.b2 == 21
        assert restored.manifold.topology.b3 == 77


# ============================================================
# ERROR HANDLING
# ============================================================

def test_topology_config_from_dict_missing_required_field():
    """Test TopologyConfig.from_dict() handles missing required fields."""
    incomplete_dict = {"b2": 21}  # Missing b3

    with pytest.raises((KeyError, TypeError)):
        TopologyConfig.from_dict(incomplete_dict)


def test_topology_config_from_json_invalid_json():
    """Test TopologyConfig.from_json() handles invalid JSON."""
    invalid_json = "{ invalid json }"

    with pytest.raises(json.JSONDecodeError):
        TopologyConfig.from_json(invalid_json)


def test_manifold_config_from_dict_invalid_nested_structure():
    """Test ManifoldConfig.from_dict() validates nested structures."""
    invalid_dict = {
        'type': 'K7',
        'construction': 'TCS',
        'topology': {'b2': 21},  # Missing b3
        'tcs_params': None
    }

    # Should fail when trying to create TopologyConfig
    with pytest.raises((KeyError, TypeError)):
        ManifoldConfig.from_dict(invalid_dict)


# ============================================================
# SPECIAL VALUES
# ============================================================

def test_serialization_preserves_float_precision():
    """Test that float values maintain precision through serialization."""
    tcs = TCSParameters(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37,
        neck_width=0.123456789
    )

    json_str = tcs.to_json()
    restored = TCSParameters.from_json(json_str)

    # Should preserve reasonable precision
    assert abs(restored.neck_width - 0.123456789) < 1e-9


def test_serialization_handles_default_values():
    """Test that default values are correctly serialized."""
    topology = TopologyConfig(b2=21, b3=77)
    # b1 and b0 have defaults

    json_str = topology.to_json()
    restored = TopologyConfig.from_json(json_str)

    assert restored.b1 == 0  # Default
    assert restored.b0 == 1  # Default


def test_config_serialization_independence():
    """Test that serializing multiple configs doesn't cause interference."""
    config1 = TopologyConfig(b2=21, b3=77)
    config2 = TopologyConfig(b2=50, b3=150)

    json1 = config1.to_json()
    json2 = config2.to_json()

    restored1 = TopologyConfig.from_json(json1)
    restored2 = TopologyConfig.from_json(json2)

    assert restored1.b2 == 21
    assert restored2.b2 == 50
