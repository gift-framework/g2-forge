"""
Edge Case Tests for g2-forge

Tests extreme topologies, numerical edge cases, and boundary conditions.
"""

import pytest
import torch
import numpy as np

# Import g2-forge components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import g2_forge as g2
from g2_forge.trainer import Trainer
from g2_forge.config import G2ForgeConfig, TCSParameters
from g2_forge.operators import (
    levi_civita_tensor,
    exterior_derivative_3,
    hodge_star_3,
)
from g2_forge.networks import PhiNetwork, HarmonicNetwork


# ============================================================================
# EXTREME TOPOLOGY TESTS
# ============================================================================

@pytest.mark.edge_case
def test_minimal_topology_b2_equals_1():
    """Test with minimal topology b₂ = 1."""
    config = g2.create_k7_config(b2_m1=1, b3_m1=5, b2_m2=0, b3_m2=5)
    trainer = Trainer(config, device='cpu', verbose=False)

    # Should create networks without error
    assert trainer.h2_network.n_forms == 1
    assert trainer.h3_network.n_forms == 10

    # Should train without error
    results = trainer.train(num_epochs=1)
    assert results is not None


@pytest.mark.edge_case
def test_very_large_topology_b2_equals_100():
    """Test with very large topology b₂ = 100."""
    config = g2.create_k7_config(b2_m1=50, b3_m1=150, b2_m2=50, b3_m2=150)
    trainer = Trainer(config, device='cpu', verbose=False)

    assert trainer.h2_network.n_forms == 100
    assert trainer.h3_network.n_forms == 300

    # Should at least initialize without error
    coords = torch.randn(10, 7)
    h2 = trainer.h2_network(coords)
    assert h2.shape == (10, 100, 21)  # (batch, b₂, components)


@pytest.mark.edge_case
def test_asymmetric_topology():
    """Test highly asymmetric TCS: b₂_m1 >> b₂_m2."""
    config = g2.create_k7_config(b2_m1=20, b3_m1=60, b2_m2=1, b3_m2=10)

    assert config.manifold.topology.b2 == 21
    assert config.manifold.topology.b3 == 70

    trainer = Trainer(config, device='cpu', verbose=False)
    results = trainer.train(num_epochs=1)
    assert results is not None


@pytest.mark.edge_case
def test_zero_b2_on_one_component():
    """Test TCS where one component has b₂ = 0."""
    config = g2.create_k7_config(b2_m1=5, b3_m1=20, b2_m2=0, b3_m2=15)

    assert config.manifold.topology.b2 == 5
    trainer = Trainer(config, device='cpu', verbose=False)

    # Should still create valid networks
    assert trainer.h2_network.n_forms == 5
    results = trainer.train(num_epochs=1)
    assert results is not None


# ============================================================================
# NUMERICAL EDGE CASES
# ============================================================================

@pytest.mark.edge_case
def test_very_small_coordinates():
    """Test network behavior with very small coordinate values."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    # Coordinates near zero
    coords = torch.randn(50, 7) * 1e-6

    phi = trainer.phi_network(coords)
    h2 = trainer.h2_network(coords)

    assert torch.isfinite(phi).all()
    assert torch.isfinite(h2).all()


@pytest.mark.edge_case
def test_very_large_coordinates():
    """Test network behavior with very large coordinate values."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    # Large coordinates
    coords = torch.randn(50, 7) * 1e3

    phi = trainer.phi_network(coords)
    h2 = trainer.h2_network(coords)

    assert torch.isfinite(phi).all()
    assert torch.isfinite(h2).all()


@pytest.mark.edge_case
def test_coordinates_at_origin():
    """Test network behavior exactly at origin."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    # All zeros
    coords = torch.zeros(10, 7)

    phi = trainer.phi_network(coords)
    h2 = trainer.h2_network(coords)

    assert torch.isfinite(phi).all()
    assert torch.isfinite(h2).all()
    assert phi.shape == (10, 7, 7, 7)


@pytest.mark.edge_case
def test_mixed_scale_coordinates():
    """Test coordinates with mixed scales across dimensions."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    # Different scales per dimension
    coords = torch.randn(50, 7)
    coords[:, 0] *= 1e-3
    coords[:, 1] *= 1e3
    coords[:, 2] *= 1.0

    phi = trainer.phi_network(coords)
    assert torch.isfinite(phi).all()


# ============================================================================
# DEGENERATE METRIC CASES
# ============================================================================

@pytest.mark.edge_case
def test_nearly_degenerate_metric():
    """Test Hodge star with nearly degenerate metric."""
    from g2_forge.operators import hodge_star_3, levi_civita_tensor

    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)

    # Nearly degenerate metric (very small eigenvalues)
    metric = torch.eye(7).unsqueeze(0).expand(batch_size, 7, 7).clone()
    metric[:, 0, 0] = 1e-6  # Nearly degenerate in one direction

    eps_indices, eps_signs = levi_civita_tensor(7)

    # Should still compute without NaN (may have numerical issues)
    star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)

    # Check for finiteness
    assert torch.isfinite(star_phi).all()


@pytest.mark.edge_case
def test_identity_metric_hodge_star():
    """Test Hodge star with exactly identity metric."""
    from g2_forge.operators import hodge_star_3, levi_civita_tensor

    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).expand(batch_size, 7, 7)

    eps_indices, eps_signs = levi_civita_tensor(7)
    star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)

    assert star_phi.shape == (batch_size, 7, 7, 7, 7)
    assert torch.isfinite(star_phi).all()


@pytest.mark.edge_case
def test_scaled_metric():
    """Test operators with uniformly scaled metric."""
    from g2_forge.operators import hodge_star_3, levi_civita_tensor

    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)

    # Metric scaled by constant factor
    scale = 5.0
    metric = scale * torch.eye(7).unsqueeze(0).expand(batch_size, 7, 7)

    eps_indices, eps_signs = levi_civita_tensor(7)
    star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)

    # Should still be finite
    assert torch.isfinite(star_phi).all()


# ============================================================================
# BATCH SIZE EDGE CASES
# ============================================================================

@pytest.mark.edge_case
def test_batch_size_one():
    """Test with batch size of 1."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    coords = torch.randn(1, 7)
    phi = trainer.phi_network(coords)

    assert phi.shape == (1, 7, 7, 7)
    assert torch.isfinite(phi).all()


@pytest.mark.edge_case
@pytest.mark.slow
def test_very_large_batch():
    """Test with very large batch size."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    # Large batch
    coords = torch.randn(1000, 7)
    phi = trainer.phi_network(coords)

    assert phi.shape == (1000, 7, 7, 7)
    assert torch.isfinite(phi).all()


# ============================================================================
# GRADIENT EDGE CASES
# ============================================================================

@pytest.mark.edge_case
def test_gradients_with_zero_loss():
    """Test gradient computation when loss is exactly zero."""
    config = g2.create_k7_config(b2_m1=2, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    coords = torch.randn(10, 7)
    phi = trainer.phi_network(coords)

    # Zero loss (artificial)
    loss = torch.tensor(0.0, requires_grad=True)

    trainer.optimizer.zero_grad()
    loss.backward()

    # Gradients should exist but be zero
    for param in trainer.phi_network.parameters():
        if param.grad is not None:
            assert torch.allclose(param.grad, torch.zeros_like(param.grad))


@pytest.mark.edge_case
def test_gradients_with_very_large_loss():
    """Test gradient computation with very large loss values."""
    config = g2.create_k7_config(b2_m1=2, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    coords = torch.randn(10, 7)
    phi = trainer.phi_network(coords)

    # Artificially large loss
    loss = phi.pow(2).sum() * 1e6

    trainer.optimizer.zero_grad()
    loss.backward()

    # Gradients should be finite (not NaN/Inf)
    for param in trainer.phi_network.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


# ============================================================================
# ANTISYMMETRY EDGE CASES
# ============================================================================

@pytest.mark.edge_case
def test_antisymmetry_with_identical_indices():
    """Test that φ_iii = 0 (antisymmetry property)."""
    config = g2.create_k7_config(b2_m1=2, b3_m1=10, b2_m2=2, b3_m2=10)
    phi_net = PhiNetwork(config, device='cpu')

    coords = torch.randn(50, 7)
    phi = phi_net.get_phi_tensor(coords)

    # φ_iii should be exactly zero for all i
    for i in range(7):
        assert torch.allclose(phi[:, i, i, i], torch.zeros(50), atol=1e-6)


@pytest.mark.edge_case
def test_antisymmetry_with_two_identical_indices():
    """Test that φ_iij = 0 for all i,j."""
    config = g2.create_k7_config(b2_m1=2, b3_m1=10, b2_m2=2, b3_m2=10)
    phi_net = PhiNetwork(config, device='cpu')

    coords = torch.randn(50, 7)
    phi = phi_net.get_phi_tensor(coords)

    # φ_iij = 0 for all i,j
    for i in range(7):
        for j in range(7):
            assert torch.allclose(phi[:, i, i, j], torch.zeros(50), atol=1e-6)
            assert torch.allclose(phi[:, i, j, i], torch.zeros(50), atol=1e-6)
            assert torch.allclose(phi[:, j, i, i], torch.zeros(50), atol=1e-6)


# ============================================================================
# HARMONIC FORM EDGE CASES
# ============================================================================

@pytest.mark.edge_case
def test_harmonic_forms_with_b2_equals_1():
    """Test harmonic 2-forms with minimal b₂ = 1."""
    config = g2.create_k7_config(b2_m1=1, b3_m1=10, b2_m2=0, b3_m2=10)
    h2_net = HarmonicNetwork(n_forms=1, hidden_dim=64, device='cpu')

    coords = torch.randn(50, 7)
    h2 = h2_net(coords)

    assert h2.shape == (50, 1, 21)  # (batch, 1 form, 21 components)
    assert torch.isfinite(h2).all()


@pytest.mark.edge_case
def test_harmonic_forms_orthogonality_minimal():
    """Test Gram matrix with minimal b₂."""
    from g2_forge.losses import gram_matrix_loss

    # Just 2 harmonic forms
    h2 = torch.randn(50, 2, 21)

    loss, det, rank = gram_matrix_loss(h2, target_rank=2)

    assert torch.isfinite(loss)
    assert rank <= 2


# ============================================================================
# TRAINING EDGE CASES
# ============================================================================

@pytest.mark.edge_case
def test_training_with_zero_learning_rate():
    """Test that training with lr=0 doesn't change parameters."""
    config = g2.create_k7_config(b2_m1=2, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer = Trainer(config, device='cpu', verbose=False)

    # Manually set learning rate to zero
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = 0.0

    # Store initial parameters
    initial_params = [p.clone() for p in trainer.phi_network.parameters()]

    # Train
    trainer.train(num_epochs=5)

    # Parameters should not change
    for p_initial, p_current in zip(initial_params, trainer.phi_network.parameters()):
        assert torch.allclose(p_initial, p_current)


@pytest.mark.edge_case
def test_training_with_single_sample():
    """Test training with batch size of 1."""
    config = g2.create_k7_config(b2_m1=2, b3_m1=10, b2_m2=2, b3_m2=10)

    # Override batch size
    config.training.batch_size = 1

    trainer = Trainer(config, device='cpu', verbose=False)
    results = trainer.train(num_epochs=2)

    assert results is not None
    assert torch.isfinite(torch.tensor(results['loss']))


# ============================================================================
# K7 MANIFOLD EDGE CASES
# ============================================================================

@pytest.mark.edge_case
def test_k7_regions_at_extremes():
    """Test region weights at extreme coordinates."""
    from g2_forge.manifolds import K7Manifold

    tcs = TCSParameters(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    k7 = K7Manifold(tcs, device='cpu')

    # Extreme coordinates
    coords_extreme = torch.tensor([
        [1e6, 0, 0, 0, 0, 0, 0],   # Very far in x
        [0, 0, 0, 0, 0, 0, -1e6],  # Very far in -z
    ])

    weights = k7.get_region_weights(coords_extreme)

    # Should still sum to 1
    total = weights['m1'] + weights['neck'] + weights['m2']
    assert torch.allclose(total, torch.ones(2), rtol=1e-4)


@pytest.mark.edge_case
def test_k7_transition_exactly_at_boundary():
    """Test region weights exactly at transition boundaries."""
    from g2_forge.manifolds import K7Manifold

    tcs = TCSParameters(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    k7 = K7Manifold(tcs, device='cpu')

    # Coordinates exactly at boundaries (if we can determine them)
    coords = torch.zeros(10, 7)

    weights = k7.get_region_weights(coords)

    # Should be finite
    assert torch.isfinite(weights['m1']).all()
    assert torch.isfinite(weights['neck']).all()
    assert torch.isfinite(weights['m2']).all()


# ============================================================================
# SUMMARY
# ============================================================================

@pytest.mark.edge_case
def test_edge_cases_summary():
    """Print summary of all edge case categories tested."""
    print("\n" + "="*80)
    print("EDGE CASE TEST SUMMARY")
    print("="*80)
    print("\nCategories tested:")
    print("  ✓ Extreme topologies (minimal b₂=1, large b₂=100, asymmetric)")
    print("  ✓ Numerical edge cases (tiny/large coords, origin, mixed scales)")
    print("  ✓ Degenerate metrics (nearly singular, identity, scaled)")
    print("  ✓ Batch size extremes (size 1, very large batches)")
    print("  ✓ Gradient edge cases (zero loss, very large loss)")
    print("  ✓ Antisymmetry edge cases (identical indices)")
    print("  ✓ Training edge cases (lr=0, batch size 1)")
    print("  ✓ K7 manifold edge cases (extreme coordinates, boundaries)")
    print("\n" + "="*80)

    assert True
