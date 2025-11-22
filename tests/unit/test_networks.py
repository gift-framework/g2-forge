"""
Unit tests for neural network architectures.

Tests PhiNetwork, HarmonicNetwork, and FourierFeatures for correctness,
antisymmetry properties, and universality across topologies.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.networks.phi_network import PhiNetwork, FourierFeatures
from g2forge.networks.harmonic_network import HarmonicNetwork


# ============================================================
# FOURIER FEATURES TESTS
# ============================================================

def test_fourier_features_initialization():
    """Test that Fourier features initialize correctly."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32, scale=1.0)

    # Should have correct dimensions
    assert ff.B.shape == (7, 32)
    # B should not be trainable
    assert not ff.B.requires_grad


def test_fourier_features_output_shape():
    """Test that Fourier features produce correct output shape."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    coords = torch.randn(10, 7)
    features = ff(coords)

    # Output: [batch, 2 * n_frequencies] (cos + sin)
    assert features.shape == (10, 2 * 32)


def test_fourier_features_deterministic():
    """Test that Fourier features are deterministic (B is fixed)."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    coords = torch.randn(10, 7)
    features1 = ff(coords)
    features2 = ff(coords)

    # Should be identical
    torch.testing.assert_close(features1, features2)


def test_fourier_features_batch_invariance():
    """Test that Fourier features work for different batch sizes."""
    ff = FourierFeatures(input_dim=7, n_frequencies=32)

    for batch_size in [1, 5, 20, 100]:
        coords = torch.randn(batch_size, 7)
        features = ff(coords)
        assert features.shape == (batch_size, 64)


# ============================================================
# PHI NETWORK TESTS
# ============================================================

def test_phi_network_initialization(gift_config):
    """Test that PhiNetwork initializes correctly."""
    phi_net = g2.networks.create_phi_network_from_config(gift_config)

    assert isinstance(phi_net, PhiNetwork)
    # Should have fourier features
    assert hasattr(phi_net, 'fourier')


def test_phi_network_output_shape(gift_config):
    """Test that PhiNetwork produces correct output shape."""
    phi_net = g2.networks.create_phi_network_from_config(gift_config)

    coords = torch.randn(10, 7)
    phi_components = phi_net(coords)

    # Should output 35 components (C(7,3) = 35)
    assert phi_components.shape == (10, 35)


def test_phi_network_tensor_shape(gift_config):
    """Test that PhiNetwork tensor expansion produces correct shape."""
    phi_net = g2.networks.create_phi_network_from_config(gift_config)

    coords = torch.randn(10, 7)
    phi_tensor = phi_net.get_phi_tensor(coords)

    # Should be [batch, 7, 7, 7]
    assert phi_tensor.shape == (10, 7, 7, 7)


def test_phi_network_antisymmetry(gift_config):
    """
    CRITICAL TEST: Verify that PhiNetwork output is antisymmetric.

    This is a fundamental requirement for a valid 3-form.
    """
    phi_net = g2.networks.create_phi_network_from_config(gift_config)

    coords = torch.randn(10, 7)
    phi_tensor = phi_net.get_phi_tensor(coords)

    # Check antisymmetry: φ_{ijk} = -φ_{jik}
    violations = []
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if i != j and i != k and j != k:
                    # φ_{ijk} = -φ_{jik}
                    diff = torch.abs(
                        phi_tensor[:, i, j, k] + phi_tensor[:, j, i, k]
                    ).max()
                    violations.append(diff.item())

    max_violation = max(violations)

    # Should be very small (numerical precision)
    assert max_violation < 1e-5, f"Antisymmetry violation: {max_violation}"


def test_phi_network_batch_sizes(small_topology_config):
    """Test that PhiNetwork works for different batch sizes."""
    phi_net = g2.networks.create_phi_network_from_config(small_topology_config)

    for batch_size in [1, 5, 20, 100]:
        coords = torch.randn(batch_size, 7)
        phi_components = phi_net(coords)
        assert phi_components.shape == (batch_size, 35)


def test_phi_network_parameter_count(gift_config):
    """Test that parameter counting works."""
    phi_net = g2.networks.create_phi_network_from_config(gift_config)

    param_count = phi_net.count_parameters()

    # Should be positive and reasonable
    assert param_count > 0
    # GIFT config uses [384, 384, 256] hidden dims
    # Should be > 100k parameters
    assert param_count > 100_000


def test_phi_network_nonzero_output(gift_config):
    """Test that PhiNetwork produces non-zero output."""
    phi_net = g2.networks.create_phi_network_from_config(gift_config)

    coords = torch.randn(10, 7)
    phi_components = phi_net(coords)

    # Should not be all zeros
    assert torch.abs(phi_components).max() > 1e-6


# ============================================================
# HARMONIC NETWORK TESTS
# ============================================================

def test_harmonic_network_h2_initialization(small_topology):
    """Test that H² network initializes correctly."""
    h2_net = g2.networks.create_harmonic_h2_network(small_topology)

    assert isinstance(h2_net, HarmonicNetwork)
    assert h2_net.p == 2  # 2-forms
    assert h2_net.n_forms == small_topology.b2  # Auto-sized!


def test_harmonic_network_h3_initialization(small_topology):
    """Test that H³ network initializes correctly."""
    h3_net = g2.networks.create_harmonic_h3_network(small_topology)

    assert isinstance(h3_net, HarmonicNetwork)
    assert h3_net.p == 3  # 3-forms
    assert h3_net.n_forms == small_topology.b3  # Auto-sized!


@pytest.mark.parametrize("b2", [3, 5, 10, 21, 50])
def test_harmonic_network_h2_universality(b2):
    """
    CRITICAL TEST: Verify H² network auto-sizes for different topologies.

    This tests the key universality feature.
    """
    topology = g2.TopologyConfig(b2=b2, b3=20)
    h2_net = g2.networks.create_harmonic_h2_network(topology)

    # Should auto-size to b2
    assert h2_net.n_forms == b2


@pytest.mark.parametrize("b3", [10, 20, 40, 77, 150])
def test_harmonic_network_h3_universality(b3):
    """
    CRITICAL TEST: Verify H³ network auto-sizes for different topologies.
    """
    topology = g2.TopologyConfig(b2=10, b3=b3)
    h3_net = g2.networks.create_harmonic_h3_network(topology)

    # Should auto-size to b3
    assert h3_net.n_forms == b3


def test_harmonic_network_h2_output_shape(small_topology):
    """Test that H² network produces correct output shape."""
    h2_net = g2.networks.create_harmonic_h2_network(small_topology)

    coords = torch.randn(10, 7)
    h2_forms = h2_net(coords)

    # Output: [batch, n_forms, n_components]
    # 2-forms have 21 components in 7D
    assert h2_forms.shape == (10, small_topology.b2, 21)


def test_harmonic_network_h3_output_shape(small_topology):
    """Test that H³ network produces correct output shape."""
    h3_net = g2.networks.create_harmonic_h3_network(small_topology)

    coords = torch.randn(10, 7)
    h3_forms = h3_net(coords)

    # Output: [batch, n_forms, n_components]
    # 3-forms have 35 components in 7D
    assert h3_forms.shape == (10, small_topology.b3, 35)


def test_harmonic_network_h2_antisymmetry(small_topology):
    """
    CRITICAL TEST: Verify that H² network outputs are antisymmetric.

    2-forms must satisfy ω_{ij} = -ω_{ji}.
    """
    h2_net = g2.networks.create_harmonic_h2_network(small_topology)

    coords = torch.randn(10, 7)
    h2_forms = h2_net(coords)

    # Get full tensors
    h2_tensors = h2_net.get_forms_as_tensors(coords)  # [batch, n_forms, 7, 7]

    # Check antisymmetry for each form
    for form_idx in range(small_topology.b2):
        omega = h2_tensors[:, form_idx, :, :]  # [batch, 7, 7]

        # ω_{ij} = -ω_{ji}
        torch.testing.assert_close(
            omega,
            -omega.transpose(-2, -1),
            rtol=1e-4,
            atol=1e-5,
            msg=f"H² form {form_idx} is not antisymmetric"
        )


def test_harmonic_network_h3_antisymmetry(small_topology):
    """
    CRITICAL TEST: Verify that H³ network outputs are antisymmetric.

    3-forms must satisfy α_{ijk} = -α_{jik}.
    """
    h3_net = g2.networks.create_harmonic_h3_network(small_topology)

    coords = torch.randn(10, 7)
    h3_forms = h3_net(coords)

    # Get full tensors
    h3_tensors = h3_net.get_forms_as_tensors(coords)  # [batch, n_forms, 7, 7, 7]

    # Check antisymmetry for each form (sample a few for speed)
    for form_idx in range(min(3, small_topology.b3)):
        alpha = h3_tensors[:, form_idx, :, :, :]  # [batch, 7, 7, 7]

        # Check α_{ijk} = -α_{jik} for sample indices
        violations = []
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    if i != j and i != k and j != k:
                        diff = torch.abs(
                            alpha[:, i, j, k] + alpha[:, j, i, k]
                        ).max()
                        violations.append(diff.item())

        max_violation = max(violations) if violations else 0.0
        assert max_violation < 1e-4, f"H³ form {form_idx} antisymmetry violation: {max_violation}"


def test_harmonic_network_batch_consistency(small_topology):
    """Test that HarmonicNetwork produces consistent results."""
    h2_net = g2.networks.create_harmonic_h2_network(small_topology)

    coords = torch.randn(10, 7)
    h2_forms1 = h2_net(coords)
    h2_forms2 = h2_net(coords)

    # Should be deterministic
    torch.testing.assert_close(h2_forms1, h2_forms2)


def test_harmonic_network_parameter_count(gift_topology):
    """Test that parameter counting works for harmonic networks."""
    h2_net = g2.networks.create_harmonic_h2_network(gift_topology)
    h3_net = g2.networks.create_harmonic_h3_network(gift_topology)

    h2_params = h2_net.count_parameters()
    h3_params = h3_net.count_parameters()

    # Both should have positive parameter counts
    assert h2_params > 0
    assert h3_params > 0

    # H³ should have more parameters (more forms: 77 vs 21)
    assert h3_params > h2_params


def test_harmonic_network_nonzero_output(small_topology):
    """Test that harmonic networks produce non-zero output."""
    h2_net = g2.networks.create_harmonic_h2_network(small_topology)
    h3_net = g2.networks.create_harmonic_h3_network(small_topology)

    coords = torch.randn(10, 7)

    h2_forms = h2_net(coords)
    h3_forms = h3_net(coords)

    assert torch.abs(h2_forms).max() > 1e-6
    assert torch.abs(h3_forms).max() > 1e-6


def test_harmonic_network_factory_creates_both(small_topology_config):
    """Test that factory function creates both H² and H³ networks."""
    h2_net, h3_net = g2.networks.create_harmonic_networks_from_config(
        small_topology_config
    )

    assert isinstance(h2_net, HarmonicNetwork)
    assert isinstance(h3_net, HarmonicNetwork)
    assert h2_net.p == 2
    assert h3_net.p == 3
    assert h2_net.n_forms == small_topology_config.manifold.topology.b2
    assert h3_net.n_forms == small_topology_config.manifold.topology.b3


# ============================================================
# INTEGRATION TESTS
# ============================================================

def test_all_networks_together(small_topology_config):
    """Test that all networks can be created and used together."""
    # Create all networks
    phi_net = g2.networks.create_phi_network_from_config(small_topology_config)
    h2_net, h3_net = g2.networks.create_harmonic_networks_from_config(
        small_topology_config
    )

    # Forward pass
    coords = torch.randn(10, 7)

    phi = phi_net(coords)
    h2 = h2_net(coords)
    h3 = h3_net(coords)

    # Check shapes
    assert phi.shape == (10, 35)
    assert h2.shape == (10, 5, 21)
    assert h3.shape == (10, 20, 35)


def test_networks_scale_with_topology():
    """
    Test that network sizes scale appropriately with topology.

    This verifies the universality: same code, different topologies.
    """
    topologies = [
        g2.TopologyConfig(b2=5, b3=20),
        g2.TopologyConfig(b2=21, b3=77),
        g2.TopologyConfig(b2=50, b3=150),
    ]

    for topology in topologies:
        h2_net = g2.networks.create_harmonic_h2_network(topology)
        h3_net = g2.networks.create_harmonic_h3_network(topology)

        # Verify auto-sizing
        assert h2_net.n_forms == topology.b2
        assert h3_net.n_forms == topology.b3

        # Verify output shapes
        coords = torch.randn(5, 7)
        h2_forms = h2_net(coords)
        h3_forms = h3_net(coords)

        assert h2_forms.shape == (5, topology.b2, 21)
        assert h3_forms.shape == (5, topology.b3, 35)
