"""
Pytest fixtures for g2-forge tests.

Provides reusable test fixtures for configurations, manifolds,
networks, and geometric objects.
"""

import pytest
import torch
import numpy as np
import g2forge as g2


# ============================================================
# Configuration Fixtures
# ============================================================

@pytest.fixture
def small_topology_config():
    """Small topology for fast testing (b₂=5, b₃=20)."""
    return g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)


@pytest.fixture
def medium_topology_config():
    """Medium topology (b₂=10, b₃=40)."""
    return g2.create_k7_config(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)


@pytest.fixture
def gift_config():
    """GIFT v1.0 configuration (b₂=21, b₃=77)."""
    return g2.G2ForgeConfig.from_gift_v1_0()


@pytest.fixture
def large_topology_config():
    """Large topology for stress testing (b₂=30, b₃=100)."""
    return g2.create_k7_config(b2_m1=15, b3_m1=50, b2_m2=15, b3_m2=50)


# ============================================================
# Topology Fixtures
# ============================================================

@pytest.fixture
def small_topology():
    """Small TopologyConfig (b₂=5, b₃=20)."""
    return g2.TopologyConfig(b2=5, b3=20)


@pytest.fixture
def gift_topology():
    """GIFT topology (b₂=21, b₃=77)."""
    return g2.TopologyConfig(b2=21, b3=77)


@pytest.fixture
def large_topology():
    """Large topology (b₂=50, b₃=150)."""
    return g2.TopologyConfig(b2=50, b3=150)


# ============================================================
# Geometric Object Fixtures
# ============================================================

@pytest.fixture(scope="session")
def levi_civita():
    """Cached Levi-Civita tensor for efficiency."""
    return g2.build_levi_civita_sparse_7d()


@pytest.fixture
def sample_coordinates():
    """Sample 7D coordinates for testing."""
    batch_size = 10
    coords = torch.randn(batch_size, 7)
    coords.requires_grad_(True)
    return coords


@pytest.fixture
def sample_metric():
    """Sample metric tensor (identity for simplicity)."""
    batch_size = 10
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
    return metric


@pytest.fixture
def sample_phi_antisymmetric():
    """
    Sample antisymmetric 3-form φ.

    Properly antisymmetrized to satisfy φ_{ijk} = -φ_{jik}.
    """
    batch_size = 10
    phi = torch.zeros(batch_size, 7, 7, 7)

    # Fill only the canonical components (i < j < k)
    # and antisymmetrize
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                # Random value for canonical component
                val = torch.randn(batch_size)

                # All even permutations get +val
                phi[:, i, j, k] = val
                phi[:, j, k, i] = val
                phi[:, k, i, j] = val

                # All odd permutations get -val
                phi[:, i, k, j] = -val
                phi[:, k, j, i] = -val
                phi[:, j, i, k] = -val

    return phi


@pytest.fixture
def sample_2form_antisymmetric():
    """
    Sample antisymmetric 2-form ω.

    Properly antisymmetrized to satisfy ω_{ij} = -ω_{ji}.
    """
    batch_size = 10
    omega = torch.zeros(batch_size, 7, 7)

    # Fill only upper triangle and antisymmetrize
    for i in range(7):
        for j in range(i + 1, 7):
            val = torch.randn(batch_size)
            omega[:, i, j] = val
            omega[:, j, i] = -val

    return omega


# ============================================================
# Manifold Fixtures
# ============================================================

@pytest.fixture
def k7_manifold_small(small_topology_config):
    """K7 manifold with small topology."""
    return g2.manifolds.create_manifold(small_topology_config.manifold)


@pytest.fixture
def k7_manifold_gift(gift_config):
    """K7 manifold with GIFT topology."""
    return g2.manifolds.create_manifold(gift_config.manifold)


# ============================================================
# Network Fixtures
# ============================================================

@pytest.fixture
def phi_network_small(small_topology_config):
    """PhiNetwork for small topology."""
    return g2.networks.create_phi_network_from_config(small_topology_config)


@pytest.fixture
def harmonic_networks_small(small_topology_config):
    """H² and H³ networks for small topology."""
    return g2.networks.create_harmonic_networks_from_config(small_topology_config)


@pytest.fixture
def phi_network_gift(gift_config):
    """PhiNetwork for GIFT topology."""
    return g2.networks.create_phi_network_from_config(gift_config)


@pytest.fixture
def harmonic_networks_gift(gift_config):
    """H² and H³ networks for GIFT topology."""
    return g2.networks.create_harmonic_networks_from_config(gift_config)


# ============================================================
# Utility Functions
# ============================================================

def assert_antisymmetric_3form(phi: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-6):
    """
    Assert that a 3-form is properly antisymmetric.

    Args:
        phi: Tensor[batch, 7, 7, 7]
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    batch_size = phi.shape[0]

    for i in range(7):
        for j in range(7):
            for k in range(7):
                if i != j and i != k and j != k:
                    # φ_{ijk} = -φ_{jik}
                    torch.testing.assert_close(
                        phi[:, i, j, k],
                        -phi[:, j, i, k],
                        rtol=rtol,
                        atol=atol,
                        msg=f"Antisymmetry failed at ({i},{j},{k})"
                    )


def assert_antisymmetric_2form(omega: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-6):
    """
    Assert that a 2-form is properly antisymmetric.

    Args:
        omega: Tensor[batch, 7, 7]
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    torch.testing.assert_close(
        omega,
        -omega.transpose(-2, -1),
        rtol=rtol,
        atol=atol,
        msg="2-form antisymmetry failed"
    )


def assert_symmetric(tensor: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-6):
    """
    Assert that a tensor is symmetric.

    Args:
        tensor: Tensor[..., n, n]
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    torch.testing.assert_close(
        tensor,
        tensor.transpose(-2, -1),
        rtol=rtol,
        atol=atol,
        msg="Tensor is not symmetric"
    )


# Make utility functions available to all tests
@pytest.fixture
def assert_antisymmetric_3form_fixture():
    return assert_antisymmetric_3form


@pytest.fixture
def assert_antisymmetric_2form_fixture():
    return assert_antisymmetric_2form


@pytest.fixture
def assert_symmetric_fixture():
    return assert_symmetric
