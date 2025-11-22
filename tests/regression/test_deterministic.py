"""
Regression tests for deterministic behavior.

Tests that g2-forge produces deterministic results with fixed seeds,
ensuring reproducibility across runs.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer


# Mark all tests as regression tests
pytestmark = pytest.mark.regression


# ============================================================
# DETERMINISM SETUP HELPERS
# ============================================================

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# NETWORK DETERMINISM TESTS
# ============================================================

def test_phi_network_deterministic(small_topology_config):
    """Test that PhiNetwork produces deterministic outputs."""
    coords = torch.randn(10, 7)

    # First run
    set_all_seeds(42)
    phi_net1 = g2.networks.create_phi_network_from_config(small_topology_config)
    output1 = phi_net1(coords)

    # Second run with same seed
    set_all_seeds(42)
    phi_net2 = g2.networks.create_phi_network_from_config(small_topology_config)
    output2 = phi_net2(coords)

    # Should be identical
    torch.testing.assert_close(output1, output2)


def test_harmonic_networks_deterministic(small_topology_config):
    """Test that HarmonicNetworks produce deterministic outputs."""
    coords = torch.randn(10, 7)

    # First run
    set_all_seeds(42)
    h2_net1, h3_net1 = g2.networks.create_harmonic_networks_from_config(
        small_topology_config
    )
    h2_output1 = h2_net1(coords)
    h3_output1 = h3_net1(coords)

    # Second run with same seed
    set_all_seeds(42)
    h2_net2, h3_net2 = g2.networks.create_harmonic_networks_from_config(
        small_topology_config
    )
    h2_output2 = h2_net2(coords)
    h3_output2 = h3_net2(coords)

    # Should be identical
    torch.testing.assert_close(h2_output1, h2_output2)
    torch.testing.assert_close(h3_output1, h3_output2)


# ============================================================
# MANIFOLD SAMPLING DETERMINISM TESTS
# ============================================================

def test_manifold_sampling_deterministic():
    """Test that manifold sampling is deterministic with seed."""
    k7 = g2.create_gift_k7()

    # First run
    set_all_seeds(42)
    coords1 = k7.sample_coordinates(100, device='cpu')

    # Second run with same seed
    set_all_seeds(42)
    coords2 = k7.sample_coordinates(100, device='cpu')

    # Should be identical
    torch.testing.assert_close(coords1, coords2)


def test_region_weights_deterministic():
    """Test that region weights are deterministic."""
    k7 = g2.create_gift_k7()

    set_all_seeds(42)
    coords = k7.sample_coordinates(100, device='cpu')

    # Compute twice
    weights1 = k7.get_region_weights(coords)
    weights2 = k7.get_region_weights(coords)

    # Should be identical
    for key in ['m1', 'neck', 'm2']:
        torch.testing.assert_close(weights1[key], weights2[key])


# ============================================================
# TRAINING DETERMINISM TESTS
# ============================================================

def test_single_training_step_deterministic(small_topology_config):
    """Test that single training step is deterministic."""
    # First run
    set_all_seeds(42)
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    results1 = trainer1.train(num_epochs=1)

    # Second run with same seed
    set_all_seeds(42)
    trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
    results2 = trainer2.train(num_epochs=1)

    # Losses should be very close (allow tiny floating point differences)
    loss1 = results1['final_metrics']['loss']
    loss2 = results2['final_metrics']['loss']

    assert abs(loss1 - loss2) < 1e-6


def test_multiple_epochs_deterministic(small_topology_config):
    """Test that multiple epochs produce deterministic results."""
    num_epochs = 5

    # First run
    set_all_seeds(42)
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    results1 = trainer1.train(num_epochs=num_epochs)

    # Second run with same seed
    set_all_seeds(42)
    trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
    results2 = trainer2.train(num_epochs=num_epochs)

    # Compare final metrics
    loss1 = results1['final_metrics']['loss']
    loss2 = results2['final_metrics']['loss']

    # Should be very close
    assert abs(loss1 - loss2) < 1e-5


def test_training_with_config_seed(small_topology_config):
    """Test that config seed ensures determinism."""
    config = small_topology_config
    config.seed = 123

    # First run
    set_all_seeds(config.seed)
    trainer1 = Trainer(config, device='cpu', verbose=False)
    results1 = trainer1.train(num_epochs=3)

    # Second run
    set_all_seeds(config.seed)
    trainer2 = Trainer(config, device='cpu', verbose=False)
    results2 = trainer2.train(num_epochs=3)

    # Compare
    loss1 = results1['final_metrics']['loss']
    loss2 = results2['final_metrics']['loss']

    assert abs(loss1 - loss2) < 1e-5


# ============================================================
# OPERATOR DETERMINISM TESTS
# ============================================================

def test_hodge_star_deterministic():
    """Test that Hodge star is deterministic."""
    from g2forge.core.operators import hodge_star_3, build_levi_civita_sparse_7d

    eps_indices, eps_signs = build_levi_civita_sparse_7d()

    set_all_seeds(42)
    phi = torch.randn(5, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(5, 1, 1)

    # Compute twice
    star_phi1 = hodge_star_3(phi, metric, eps_indices, eps_signs)
    star_phi2 = hodge_star_3(phi, metric, eps_indices, eps_signs)

    # Should be identical
    torch.testing.assert_close(star_phi1, star_phi2)


def test_exterior_derivative_deterministic():
    """Test that exterior derivative is deterministic."""
    from g2forge.core.operators import compute_exterior_derivative

    set_all_seeds(42)
    coords = torch.randn(10, 7, requires_grad=True)
    phi = torch.randn(10, 7, 7, 7)

    # Compute twice
    dphi1 = compute_exterior_derivative(phi, coords, subsample_factor=1)
    dphi2 = compute_exterior_derivative(phi, coords, subsample_factor=1)

    # Should be identical
    torch.testing.assert_close(dphi1, dphi2)


# ============================================================
# LOSS FUNCTION DETERMINISM TESTS
# ============================================================

def test_composite_loss_deterministic(small_topology_config):
    """Test that composite loss is deterministic."""
    from g2forge.core.losses import CompositeLoss

    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)
    topology = small_topology_config.manifold.topology

    loss_fn = CompositeLoss(topology=topology, manifold=manifold)

    # Create dummy inputs
    set_all_seeds(42)
    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)
    dphi = torch.randn(batch_size, 7, 7, 7, 7)
    dstar_phi = torch.randn(batch_size, 7, 7)
    star_phi = torch.randn(batch_size, 7, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
    harmonic_h2 = torch.randn(batch_size, topology.b2, 21)
    harmonic_h3 = torch.randn(batch_size, topology.b3, 35)
    region_weights = {
        'm1': torch.ones(batch_size),
        'neck': torch.zeros(batch_size),
        'm2': torch.zeros(batch_size)
    }
    loss_weights = {
        'torsion_closure': 1.0,
        'torsion_coclosure': 1.0,
        'volume': 0.1,
        'gram_h2': 1.0,
        'gram_h3': 1.0,
        'boundary': 1.0,
        'calibration': 0.0
    }

    # Compute twice
    total_loss1, _ = loss_fn(
        phi, dphi, dstar_phi, star_phi, metric,
        harmonic_h2, harmonic_h3, region_weights,
        loss_weights, epoch=0
    )

    total_loss2, _ = loss_fn(
        phi, dphi, dstar_phi, star_phi, metric,
        harmonic_h2, harmonic_h3, region_weights,
        loss_weights, epoch=0
    )

    # Should be identical
    torch.testing.assert_close(total_loss1, total_loss2)


# ============================================================
# PARAMETER INITIALIZATION DETERMINISM TESTS
# ============================================================

def test_network_initialization_deterministic(small_topology_config):
    """Test that network initialization is deterministic."""
    # Create two networks with same seed
    set_all_seeds(42)
    phi_net1 = g2.networks.create_phi_network_from_config(small_topology_config)

    set_all_seeds(42)
    phi_net2 = g2.networks.create_phi_network_from_config(small_topology_config)

    # Parameters should be identical
    for p1, p2 in zip(phi_net1.parameters(), phi_net2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_optimizer_initialization_deterministic(small_topology_config):
    """Test that optimizer initialization is deterministic."""
    set_all_seeds(42)
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    opt_state1 = trainer1.optimizer.state_dict()

    set_all_seeds(42)
    trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
    opt_state2 = trainer2.optimizer.state_dict()

    # Optimizer states should match
    assert opt_state1['param_groups'][0]['lr'] == opt_state2['param_groups'][0]['lr']


# ============================================================
# DIFFERENT SEEDS PRODUCE DIFFERENT RESULTS
# ============================================================

def test_different_seeds_produce_different_results(small_topology_config):
    """Test that different seeds produce different results."""
    # Run with seed 42
    set_all_seeds(42)
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    results1 = trainer1.train(num_epochs=3)

    # Run with seed 123
    set_all_seeds(123)
    trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
    results2 = trainer2.train(num_epochs=3)

    # Losses should be different
    loss1 = results1['final_metrics']['loss']
    loss2 = results2['final_metrics']['loss']

    assert abs(loss1 - loss2) > 0.01  # Should be noticeably different


def test_network_outputs_different_with_different_seeds(small_topology_config):
    """Test that networks produce different outputs with different seeds."""
    coords = torch.randn(10, 7)

    # Seed 42
    set_all_seeds(42)
    phi_net1 = g2.networks.create_phi_network_from_config(small_topology_config)
    output1 = phi_net1(coords)

    # Seed 123
    set_all_seeds(123)
    phi_net2 = g2.networks.create_phi_network_from_config(small_topology_config)
    output2 = phi_net2(coords)

    # Should be different
    assert not torch.allclose(output1, output2, rtol=1e-3)
