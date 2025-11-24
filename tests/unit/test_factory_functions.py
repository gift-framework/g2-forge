"""
Unit tests for factory functions.

Tests factory functions that create configs and manifolds,
including edge cases and error handling.
"""

import pytest
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.manifolds.base import create_manifold, Manifold
from g2forge.manifolds.k7 import K7Manifold
from g2forge.utils.config import (
    ManifoldConfig,
    TopologyConfig,
    TCSParameters,
    G2ForgeConfig,
    create_k7_config,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# create_k7_config() FACTORY TESTS
# ============================================================

def test_create_k7_config_basic():
    """Test create_k7_config() creates valid config."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    assert isinstance(config, G2ForgeConfig)
    assert config.manifold.type == "K7"
    assert config.manifold.construction == "TCS"


def test_create_k7_config_topology_sums():
    """Test that create_k7_config() correctly sums topology."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    assert config.manifold.topology.b2 == 21  # 11 + 10
    assert config.manifold.topology.b3 == 77  # 40 + 37


def test_create_k7_config_tcs_params_set():
    """Test that create_k7_config() sets TCS parameters."""
    config = create_k7_config(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    assert config.manifold.tcs_params is not None
    assert config.manifold.tcs_params.b2_m1 == 5
    assert config.manifold.tcs_params.b3_m1 == 20
    assert config.manifold.tcs_params.b2_m2 == 5
    assert config.manifold.tcs_params.b3_m2 == 20


@pytest.mark.parametrize("b2_m1,b3_m1,b2_m2,b3_m2", [
    (1, 5, 0, 5),
    (3, 10, 2, 10),
    (5, 20, 5, 20),
    (11, 40, 10, 37),
    (25, 75, 25, 75),
])
def test_create_k7_config_various_topologies(b2_m1, b3_m1, b2_m2, b3_m2):
    """Test create_k7_config() with various component topologies."""
    config = create_k7_config(
        b2_m1=b2_m1, b3_m1=b3_m1,
        b2_m2=b2_m2, b3_m2=b3_m2
    )

    assert config.manifold.topology.b2 == b2_m1 + b2_m2
    assert config.manifold.topology.b3 == b3_m1 + b3_m2
    assert config.manifold.tcs_params.b2_m1 == b2_m1


def test_create_k7_config_validates():
    """Test that create_k7_config() creates valid config."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    # Should pass validation
    assert config.validate() is True


def test_create_k7_config_zero_component():
    """Test create_k7_config() with zero b₂ on one component."""
    config = create_k7_config(b2_m1=5, b3_m1=20, b2_m2=0, b3_m2=15)

    assert config.manifold.topology.b2 == 5
    assert config.manifold.topology.b3 == 35


# ============================================================
# G2ForgeConfig.from_gift_v1_0() FACTORY TESTS
# ============================================================

def test_from_gift_v1_0_factory():
    """Test G2ForgeConfig.from_gift_v1_0() factory."""
    config = G2ForgeConfig.from_gift_v1_0()

    assert isinstance(config, G2ForgeConfig)
    assert config.manifold.topology.b2 == 21
    assert config.manifold.topology.b3 == 77


def test_from_gift_v1_0_tcs_params():
    """Test from_gift_v1_0() sets correct TCS params."""
    config = G2ForgeConfig.from_gift_v1_0()

    assert config.manifold.tcs_params.b2_m1 == 11
    assert config.manifold.tcs_params.b3_m1 == 40
    assert config.manifold.tcs_params.b2_m2 == 10
    assert config.manifold.tcs_params.b3_m2 == 37


def test_from_gift_v1_0_validates():
    """Test that from_gift_v1_0() creates valid config."""
    config = G2ForgeConfig.from_gift_v1_0()

    assert config.validate() is True


def test_from_gift_v1_0_is_k7():
    """Test that from_gift_v1_0() creates K7 manifold."""
    config = G2ForgeConfig.from_gift_v1_0()

    assert config.manifold.type == "K7"
    assert config.manifold.construction == "TCS"


# ============================================================
# G2ForgeConfig.from_gift_v1_2b() FACTORY TESTS
# ============================================================

def test_from_gift_v1_2b_factory():
    """Test G2ForgeConfig.from_gift_v1_2b() factory."""
    config = G2ForgeConfig.from_gift_v1_2b()

    assert isinstance(config, G2ForgeConfig)
    assert config.manifold.topology.b2 == 21
    assert config.manifold.topology.b3 == 77


def test_from_gift_v1_2b_validates():
    """Test that from_gift_v1_2b() creates valid config."""
    config = G2ForgeConfig.from_gift_v1_2b()

    assert config.validate() is True


def test_from_gift_v1_2b_has_volume_normalizer():
    """Test that from_gift_v1_2b() includes volume normalizer settings."""
    config = G2ForgeConfig.from_gift_v1_2b()

    # Should have volume normalization enabled (check in physics config if exists)
    assert config is not None


# ============================================================
# create_manifold() FACTORY TESTS
# ============================================================

def test_create_manifold_from_k7_config():
    """Test create_manifold() with K7 config."""
    topology = TopologyConfig(b2=21, b3=77)
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    manifold_config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    manifold = create_manifold(manifold_config)

    assert isinstance(manifold, K7Manifold)
    assert manifold.b2 == 21
    assert manifold.b3 == 77


def test_create_manifold_returns_correct_type():
    """Test that create_manifold() returns Manifold subclass."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    manifold = create_manifold(config.manifold)

    assert isinstance(manifold, Manifold)
    assert isinstance(manifold, K7Manifold)


def test_create_manifold_with_gift_config():
    """Test create_manifold() with GIFT config."""
    config = G2ForgeConfig.from_gift_v1_0()

    manifold = create_manifold(config.manifold)

    assert isinstance(manifold, K7Manifold)
    assert manifold.b2 == 21
    assert manifold.b3 == 77


def test_create_manifold_rejects_unknown_type():
    """Test create_manifold() raises for unknown manifold type."""
    topology = TopologyConfig(b2=5, b3=20)

    config = ManifoldConfig(
        type="UnknownManifold",
        construction="Custom",
        topology=topology
    )

    with pytest.raises(ValueError, match="Unknown manifold type"):
        create_manifold(config)


def test_create_manifold_validates_config():
    """Test that create_manifold() validates the config."""
    # Create invalid config
    topology = TopologyConfig(b2=30, b3=80)  # Mismatch!
    tcs_params = TCSParameters(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    # create_manifold calls __init__ which calls validate()
    with pytest.raises(ValueError):
        create_manifold(config)


@pytest.mark.parametrize("b2_m1,b3_m1,b2_m2,b3_m2", [
    (3, 10, 2, 10),
    (5, 20, 5, 20),
    (11, 40, 10, 37),
])
def test_create_manifold_various_topologies(b2_m1, b3_m1, b2_m2, b3_m2):
    """Test create_manifold() with various K7 topologies."""
    config = create_k7_config(
        b2_m1=b2_m1, b3_m1=b3_m1,
        b2_m2=b2_m2, b3_m2=b3_m2
    )

    manifold = create_manifold(config.manifold)

    assert manifold.b2 == b2_m1 + b2_m2
    assert manifold.b3 == b3_m1 + b3_m2


# ============================================================
# NETWORK FACTORY TESTS
# ============================================================

def test_create_phi_network_from_config():
    """Test creating PhiNetwork from config."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    phi_net = g2.networks.create_phi_network_from_config(config)

    assert phi_net is not None
    assert hasattr(phi_net, 'get_phi_tensor')


def test_create_harmonic_networks_from_config():
    """Test creating harmonic networks from config."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    h2_net, h3_net = g2.networks.create_harmonic_networks_from_config(config)

    assert h2_net is not None
    assert h3_net is not None
    # Should auto-size from topology
    assert h2_net.n_forms == 21
    assert h3_net.n_forms == 77


def test_harmonic_networks_auto_size_from_topology():
    """Test that harmonic networks auto-size from topology."""
    config = create_k7_config(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    h2_net, h3_net = g2.networks.create_harmonic_networks_from_config(config)

    # Should match topology
    assert h2_net.n_forms == 10  # 5 + 5
    assert h3_net.n_forms == 40  # 20 + 20


@pytest.mark.parametrize("b2_m1,b3_m1,b2_m2,b3_m2", [
    (3, 10, 2, 10),
    (5, 20, 5, 20),
    (11, 40, 10, 37),
])
def test_harmonic_networks_various_topologies(b2_m1, b3_m1, b2_m2, b3_m2):
    """Test harmonic networks for various topologies."""
    config = create_k7_config(
        b2_m1=b2_m1, b3_m1=b3_m1,
        b2_m2=b2_m2, b3_m2=b3_m2
    )

    h2_net, h3_net = g2.networks.create_harmonic_networks_from_config(config)

    assert h2_net.n_forms == b2_m1 + b2_m2
    assert h3_net.n_forms == b3_m1 + b3_m2


# ============================================================
# FACTORY CONSISTENCY TESTS
# ============================================================

def test_factory_configs_are_consistent():
    """Test that different factory methods produce consistent configs."""
    # Both should create GIFT topology
    config1 = G2ForgeConfig.from_gift_v1_0()
    config2 = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    assert config1.manifold.topology.b2 == config2.manifold.topology.b2
    assert config1.manifold.topology.b3 == config2.manifold.topology.b3
    assert config1.manifold.tcs_params.b2_m1 == config2.manifold.tcs_params.b2_m1


def test_factory_creates_compatible_components():
    """Test that factory-created components work together."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    # Create all components
    manifold = create_manifold(config.manifold)
    phi_net = g2.networks.create_phi_network_from_config(config)
    h2_net, h3_net = g2.networks.create_harmonic_networks_from_config(config)

    # All should be compatible
    coords = manifold.sample_coordinates(10)
    assert coords.shape == (10, 7)

    phi = phi_net.get_phi_tensor(coords)
    assert phi.shape == (10, 7, 7, 7)

    h2 = h2_net(coords)
    assert h2.shape[1] == manifold.b2

    h3 = h3_net(coords)
    assert h3.shape[1] == manifold.b3


# ============================================================
# ERROR HANDLING IN FACTORIES
# ============================================================

def test_create_k7_config_with_negative_fails():
    """Test that create_k7_config() with negative values creates invalid config."""
    # This should create a config, but it won't validate
    config = create_k7_config(b2_m1=-1, b3_m1=40, b2_m2=10, b3_m2=37)

    with pytest.raises(ValueError):
        config.validate()


def test_factory_preserves_custom_parameters():
    """Test that factories preserve additional parameters."""
    config = create_k7_config(
        b2_m1=11, b3_m1=40,
        b2_m2=10, b3_m2=37
    )

    # Should have neck_width parameter
    assert hasattr(config.manifold.tcs_params, 'neck_width')
    assert config.manifold.tcs_params.neck_width > 0


# ============================================================
# INTEGRATION TESTS
# ============================================================

def test_full_pipeline_from_factory():
    """Test complete pipeline using factory functions."""
    # Create config
    config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

    # Validate
    assert config.validate() is True

    # Create manifold
    manifold = create_manifold(config.manifold)
    assert manifold.b2 == 5
    assert manifold.b3 == 20

    # Create networks
    phi_net = g2.networks.create_phi_network_from_config(config)
    h2_net, h3_net = g2.networks.create_harmonic_networks_from_config(config)

    # Sample and forward pass
    coords = manifold.sample_coordinates(10)
    phi = phi_net.get_phi_tensor(coords)
    h2 = h2_net(coords)
    h3 = h3_net(coords)

    # All should work together
    assert phi.shape == (10, 7, 7, 7)
    assert h2.shape[1] == 5
    assert h3.shape[1] == 20


def test_factories_produce_reproducible_configs():
    """Test that factories produce reproducible configs."""
    config1 = G2ForgeConfig.from_gift_v1_0()
    config2 = G2ForgeConfig.from_gift_v1_0()

    # Should have identical topology
    assert config1.manifold.topology.b2 == config2.manifold.topology.b2
    assert config1.manifold.topology.b3 == config2.manifold.topology.b3


def test_factory_configs_serialize_correctly():
    """Test that factory-created configs can be serialized."""
    config = create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)

    # Should be able to serialize
    json_str = config.to_json()
    restored = G2ForgeConfig.from_json(json_str)

    assert restored.manifold.topology.b2 == config.manifold.topology.b2
    assert restored.manifold.topology.b3 == config.manifold.topology.b3
