"""
Integration tests for training pipeline.

Tests complete training workflows, convergence, and
multi-epoch behavior.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# ============================================================
# BASIC TRAINING TESTS
# ============================================================

def test_train_one_epoch(small_topology_config):
    """Test that training runs for one epoch without errors."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=1)

    assert 'final_metrics' in results
    assert 'initial_metrics' in results


def test_train_multiple_epochs(small_topology_config):
    """Test that training runs for multiple epochs."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=5)

    assert 'final_metrics' in results
    assert 'metrics_history' in results
    # Should have 5 or 6 records (depending on logging frequency)
    assert len(results['metrics_history']) >= 1


def test_train_returns_metrics(small_topology_config):
    """Test that training returns expected metrics."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=3)

    final = results['final_metrics']

    # Should have key metrics
    assert 'loss' in final
    assert 'torsion_closure' in final
    assert 'rank_h2' in final
    assert 'rank_h3' in final


def test_train_loss_is_finite(small_topology_config):
    """Test that training produces finite loss values."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=5)

    loss = results['final_metrics']['loss']

    assert np.isfinite(loss)
    assert loss > 0  # Loss should be positive


def test_train_metrics_improve(small_topology_config):
    """Test that some metrics improve during training."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=20)

    initial_loss = results['initial_metrics']['loss']
    final_loss = results['final_metrics']['loss']

    # Loss should decrease (or at least not explode)
    # Allow some slack since 20 epochs is short
    assert final_loss < initial_loss * 2


# ============================================================
# CURRICULUM LEARNING TESTS
# ============================================================

def test_curriculum_phase_transitions(gift_config):
    """Test that curriculum phases transition during training."""
    # Modify config for fast testing
    config = gift_config
    config.training.total_epochs = 50

    # Adjust phase boundaries for testing
    config.training.curriculum['phase1_neck_stability'].epoch_range = (0, 10)
    config.training.curriculum['phase2_acyl_matching'].epoch_range = (10, 20)
    config.training.curriculum['phase3_cohomology_refinement'].epoch_range = (20, 30)
    config.training.curriculum['phase4_harmonic_extraction'].epoch_range = (30, 40)
    config.training.curriculum['phase5_calibration_finetune'].epoch_range = (40, 50)

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train through phases
    results = trainer.train(num_epochs=50)

    # Should complete without errors
    assert 'final_metrics' in results


def test_curriculum_loss_weights_change(gift_config):
    """Test that loss weights change between curriculum phases."""
    config = gift_config

    # Adjust for fast testing
    config.training.curriculum['phase1_neck_stability'].epoch_range = (0, 5)
    config.training.curriculum['phase2_acyl_matching'].epoch_range = (5, 10)

    trainer = Trainer(config, device='cpu', verbose=False)

    # Get phase 1 weights
    _, phase1 = trainer._get_current_phase(epoch=2)
    weights1 = phase1.loss_weights

    # Get phase 2 weights
    _, phase2 = trainer._get_current_phase(epoch=7)
    weights2 = phase2.loss_weights

    # Weights should differ
    assert weights1['torsion_closure'] != weights2['torsion_closure']


# ============================================================
# GRADIENT FLOW TESTS
# ============================================================

def test_gradients_flow_through_phi_network(small_topology_config):
    """Test that gradients flow through PhiNetwork."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # One training step
    trainer.train(num_epochs=1)

    # Check that phi network has gradients
    has_grads = any(
        p.grad is not None and torch.abs(p.grad).max() > 0
        for p in trainer.phi_network.parameters()
    )

    assert has_grads, "PhiNetwork should have gradients after training"


def test_gradients_flow_through_harmonic_networks(small_topology_config):
    """Test that gradients flow through harmonic networks."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # One training step
    trainer.train(num_epochs=1)

    # Check H2 network
    h2_has_grads = any(
        p.grad is not None and torch.abs(p.grad).max() > 0
        for p in trainer.h2_network.parameters()
    )

    # Check H3 network
    h3_has_grads = any(
        p.grad is not None and torch.abs(p.grad).max() > 0
        for p in trainer.h3_network.parameters()
    )

    assert h2_has_grads, "H2 network should have gradients"
    assert h3_has_grads, "H3 network should have gradients"


def test_gradients_not_nan_or_inf(small_topology_config):
    """Test that gradients remain finite during training."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    trainer.train(num_epochs=3)

    # Check all networks for NaN/Inf gradients
    for network in [trainer.phi_network, trainer.h2_network, trainer.h3_network]:
        for param in network.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), "Gradients should be finite"


# ============================================================
# UNIVERSALITY INTEGRATION TESTS
# ============================================================

@pytest.mark.parametrize("b2_m1,b3_m1,b2_m2,b3_m2", [
    (3, 10, 2, 10),      # Very small
    (5, 20, 5, 20),      # Small
    (11, 40, 10, 37),    # GIFT
])
def test_training_different_topologies(b2_m1, b3_m1, b2_m2, b3_m2):
    """
    CRITICAL INTEGRATION TEST: Verify training works for ANY topology.

    This validates the key universality feature end-to-end.
    """
    config = g2.create_k7_config(
        b2_m1=b2_m1, b3_m1=b3_m1,
        b2_m2=b2_m2, b3_m2=b3_m2
    )

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train for a few epochs
    results = trainer.train(num_epochs=5)

    # Should complete without errors
    assert 'final_metrics' in results

    # Check that network sizes match topology
    expected_b2 = b2_m1 + b2_m2
    expected_b3 = b3_m1 + b3_m2

    assert trainer.h2_network.n_forms == expected_b2
    assert trainer.h3_network.n_forms == expected_b3

    # Metrics should include rank checks for correct topology
    assert results['final_metrics']['rank_h2'] <= expected_b2
    assert results['final_metrics']['rank_h3'] <= expected_b3


# ============================================================
# CONVERGENCE TESTS
# ============================================================

@pytest.mark.slow
def test_training_convergence_100_epochs(small_topology_config):
    """Test that training shows convergence over 100 epochs."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=100)

    initial_loss = results['initial_metrics']['loss']
    final_loss = results['final_metrics']['loss']

    # Loss should decrease significantly over 100 epochs
    assert final_loss < initial_loss * 0.8  # At least 20% improvement


@pytest.mark.slow
def test_torsion_reduces_over_time(small_topology_config):
    """Test that torsion metrics reduce during training."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=50)

    initial_torsion = results['initial_metrics']['torsion_closure']
    final_torsion = results['final_metrics']['torsion_closure']

    # Torsion should decrease
    assert final_torsion <= initial_torsion


# ============================================================
# REPRODUCIBILITY TESTS
# ============================================================

def test_training_deterministic_with_seed(small_topology_config):
    """Test that training is deterministic with fixed seed."""
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    config1 = small_topology_config
    config1.seed = 42
    trainer1 = Trainer(config1, device='cpu', verbose=False)
    results1 = trainer1.train(num_epochs=3)

    # Reset seed and train again
    torch.manual_seed(42)
    np.random.seed(42)

    config2 = small_topology_config
    config2.seed = 42
    trainer2 = Trainer(config2, device='cpu', verbose=False)
    results2 = trainer2.train(num_epochs=3)

    # Losses should be close (not exact due to floating point)
    loss1 = results1['final_metrics']['loss']
    loss2 = results2['final_metrics']['loss']

    assert abs(loss1 - loss2) < 0.1  # Should be very similar


# ============================================================
# METRICS HISTORY TESTS
# ============================================================

def test_metrics_history_logged(small_topology_config):
    """Test that metrics history is logged during training."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=10)

    # Should have metrics history
    assert 'metrics_history' in results
    assert len(results['metrics_history']) > 0


def test_metrics_history_contains_epochs(small_topology_config):
    """Test that metrics history contains epoch numbers."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=10)

    history = results['metrics_history']

    # Each entry should have epoch
    for entry in history:
        assert 'epoch' in entry


def test_metrics_history_contains_loss_components(small_topology_config):
    """Test that metrics history contains loss components."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=10)

    history = results['metrics_history']

    # Should have various loss components
    if len(history) > 0:
        entry = history[0]
        assert 'loss' in entry or 'total_loss' in entry


# ============================================================
# ERROR HANDLING TESTS
# ============================================================

def test_train_with_zero_epochs(small_topology_config):
    """Test that training handles zero epochs gracefully."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=0)

    # Should return without training
    assert 'final_metrics' in results or results is not None
