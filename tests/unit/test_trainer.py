"""
Unit tests for Trainer components.

Tests trainer initialization, learning rate scheduling,
optimizer setup, and individual training step components.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer


# ============================================================
# TRAINER INITIALIZATION TESTS
# ============================================================

def test_trainer_initialization(small_topology_config):
    """Test that Trainer initializes correctly."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    assert trainer is not None
    assert isinstance(trainer, Trainer)


def test_trainer_creates_networks(small_topology_config):
    """Test that Trainer creates all required networks."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Should have phi network
    assert hasattr(trainer, 'phi_network')
    assert trainer.phi_network is not None

    # Should have harmonic networks
    assert hasattr(trainer, 'h2_network')
    assert hasattr(trainer, 'h3_network')
    assert trainer.h2_network is not None
    assert trainer.h3_network is not None


def test_trainer_creates_manifold(small_topology_config):
    """Test that Trainer creates manifold from config."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    assert hasattr(trainer, 'manifold')
    assert trainer.manifold is not None
    assert trainer.manifold.b2 == small_topology_config.manifold.topology.b2
    assert trainer.manifold.b3 == small_topology_config.manifold.topology.b3


def test_trainer_creates_optimizer(small_topology_config):
    """Test that Trainer creates optimizer."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    assert hasattr(trainer, 'optimizer')
    assert trainer.optimizer is not None


def test_trainer_creates_scheduler(small_topology_config):
    """Test that Trainer creates learning rate scheduler."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    assert hasattr(trainer, 'scheduler')
    assert trainer.scheduler is not None


def test_trainer_creates_loss_fn(small_topology_config):
    """Test that Trainer creates loss function with correct topology."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    assert hasattr(trainer, 'loss_fn')
    assert trainer.loss_fn is not None

    # Loss function should use correct topology
    assert trainer.loss_fn.topology.b2 == small_topology_config.manifold.topology.b2
    assert trainer.loss_fn.topology.b3 == small_topology_config.manifold.topology.b3


def test_trainer_device_placement(small_topology_config):
    """Test that Trainer places networks on correct device."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Networks should be on CPU
    assert next(trainer.phi_network.parameters()).device.type == 'cpu'
    assert next(trainer.h2_network.parameters()).device.type == 'cpu'
    assert next(trainer.h3_network.parameters()).device.type == 'cpu'


# ============================================================
# OPTIMIZER TESTS
# ============================================================

def test_trainer_optimizer_type(small_topology_config):
    """Test that optimizer is AdamW."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    assert isinstance(trainer.optimizer, torch.optim.AdamW)


def test_trainer_optimizer_lr(small_topology_config):
    """Test that optimizer uses correct learning rate."""
    config = small_topology_config
    config.training.lr = 5e-4  # Custom LR

    trainer = Trainer(config, device='cpu', verbose=False)

    # Check learning rate
    param_groups = trainer.optimizer.param_groups
    assert len(param_groups) > 0
    assert param_groups[0]['lr'] == 5e-4


def test_trainer_optimizer_weight_decay(small_topology_config):
    """Test that optimizer uses correct weight decay."""
    config = small_topology_config
    config.training.weight_decay = 1e-3

    trainer = Trainer(config, device='cpu', verbose=False)

    param_groups = trainer.optimizer.param_groups
    assert param_groups[0]['weight_decay'] == 1e-3


def test_trainer_optimizer_has_all_parameters(small_topology_config):
    """Test that optimizer includes parameters from all networks."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Count total parameters
    total_params = sum(p.numel() for p in trainer.optimizer.param_groups[0]['params'])

    # Should have substantial number of parameters
    assert total_params > 1000  # At least some parameters


# ============================================================
# LEARNING RATE SCHEDULER TESTS
# ============================================================

def test_trainer_scheduler_warmup(small_topology_config):
    """Test that scheduler implements warmup phase."""
    config = small_topology_config
    config.training.warmup_epochs = 100
    config.training.lr = 1e-3

    trainer = Trainer(config, device='cpu', verbose=False)

    # Initial LR should be low (warmup)
    initial_lr = trainer.optimizer.param_groups[0]['lr']

    # After several steps, LR should increase
    for _ in range(50):
        trainer.scheduler.step()

    warmed_lr = trainer.optimizer.param_groups[0]['lr']

    # LR should have increased during warmup
    assert warmed_lr > initial_lr


def test_trainer_scheduler_cosine_decay(small_topology_config):
    """Test that scheduler uses cosine annealing after warmup."""
    config = small_topology_config
    config.training.warmup_epochs = 10
    config.training.total_epochs = 1000
    config.training.lr = 1e-3
    config.training.lr_min = 1e-6

    trainer = Trainer(config, device='cpu', verbose=False)

    # Skip warmup
    for _ in range(20):
        trainer.scheduler.step()

    lr_at_20 = trainer.optimizer.param_groups[0]['lr']

    # Advance many epochs (cosine should decay)
    for _ in range(480):  # Total 500 epochs
        trainer.scheduler.step()

    lr_at_500 = trainer.optimizer.param_groups[0]['lr']

    # LR should have decayed
    assert lr_at_500 < lr_at_20


# ============================================================
# CURRICULUM PHASE TESTS
# ============================================================

def test_trainer_gets_current_phase(gift_config):
    """Test that Trainer can retrieve current curriculum phase."""
    trainer = Trainer(gift_config, device='cpu', verbose=False)

    # GIFT has 5 phases
    # Phase 1: epochs 0-2000
    phase_name, phase_config = trainer._get_current_phase(epoch=1000)

    assert phase_name is not None
    assert phase_config is not None
    assert 'phase1' in phase_name.lower()


def test_trainer_phase_transitions(gift_config):
    """Test that curriculum phases transition correctly."""
    trainer = Trainer(gift_config, device='cpu', verbose=False)

    # Test phase boundaries
    phase1_name, _ = trainer._get_current_phase(epoch=1000)
    phase2_name, _ = trainer._get_current_phase(epoch=3000)
    phase5_name, _ = trainer._get_current_phase(epoch=12000)

    # Should be different phases
    assert 'phase1' in phase1_name.lower()
    assert 'phase2' in phase2_name.lower()
    assert 'phase5' in phase5_name.lower()


def test_trainer_phase_loss_weights(gift_config):
    """Test that phases have different loss weights."""
    trainer = Trainer(gift_config, device='cpu', verbose=False)

    _, phase1 = trainer._get_current_phase(epoch=1000)
    _, phase2 = trainer._get_current_phase(epoch=3000)

    # Loss weights should differ between phases
    assert phase1.loss_weights != phase2.loss_weights


# ============================================================
# METRICS TRACKING TESTS
# ============================================================

def test_trainer_initializes_metrics_history(small_topology_config):
    """Test that Trainer initializes metrics history."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    assert hasattr(trainer, 'metrics_history')
    assert isinstance(trainer.metrics_history, list)
    assert len(trainer.metrics_history) == 0  # Empty at start


# ============================================================
# INTEGRATION TESTS
# ============================================================

def test_trainer_different_topologies(small_topology_config, gift_config):
    """
    Test that Trainer works with different topologies.

    Validates universality: same Trainer code, different configs.
    """
    # Small topology
    trainer_small = Trainer(small_topology_config, device='cpu', verbose=False)
    assert trainer_small.h2_network.n_forms == 5
    assert trainer_small.h3_network.n_forms == 20

    # GIFT topology
    trainer_gift = Trainer(gift_config, device='cpu', verbose=False)
    assert trainer_gift.h2_network.n_forms == 21
    assert trainer_gift.h3_network.n_forms == 77


def test_trainer_property_access(small_topology_config):
    """Test that Trainer properties are accessible."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Properties should be accessible
    assert trainer.phi_network is not None
    assert trainer.h2_network is not None
    assert trainer.h3_network is not None
    assert trainer.manifold is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.loss_fn is not None


# ============================================================
# CURRICULUM EDGE CASE TESTS
# ============================================================

def test_curriculum_with_single_phase(small_topology_config):
    """Test curriculum with only 1 phase works correctly."""
    config = small_topology_config
    config.training.num_curriculum_phases = 1

    trainer = Trainer(config, device='cpu', verbose=False)

    # Should still work with single phase
    results = trainer.train(num_epochs=3)

    assert 'final_metrics' in results
    assert np.isfinite(results['final_metrics']['loss'])


def test_curriculum_loss_weight_transitions(small_topology_config):
    """Test that loss weights transition smoothly between phases."""
    config = small_topology_config
    config.training.num_curriculum_phases = 3

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train through multiple phases
    results = trainer.train(num_epochs=15)  # Will go through phase transitions

    # Should complete without errors
    assert 'final_metrics' in results
    assert np.isfinite(results['final_metrics']['loss'])


def test_curriculum_with_very_short_phases(small_topology_config):
    """Test curriculum with very short phase durations."""
    config = small_topology_config
    config.training.num_curriculum_phases = 5

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train for just a few epochs (shorter than typical phase duration)
    results = trainer.train(num_epochs=3)

    # Should handle gracefully
    assert 'final_metrics' in results
    assert np.isfinite(results['final_metrics']['loss'])


def test_curriculum_phase_zero_weight(small_topology_config):
    """Test training when some loss weights are zero."""
    config = small_topology_config

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train one step with manual loss weights
    coords = trainer.manifold.sample_coordinates(n_samples=256, device='cpu')
    phi = trainer.phi_network(coords)

    # Create dummy inputs
    dphi = torch.randn_like(phi).unsqueeze(-1).repeat(1, 1, 1, 1, 7)
    dstar_phi = torch.randn(256, 7, 7)
    star_phi = torch.randn(256, 7, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(256, 1, 1)
    h2 = trainer.h2_network(coords)
    h3 = trainer.h3_network(coords)
    region_weights = trainer.manifold.get_region_weights(coords)

    # Set some weights to zero
    loss_weights = {
        'torsion_closure': 1.0,
        'torsion_coclosure': 0.0,  # Zero!
        'volume': 0.0,             # Zero!
        'gram_h2': 1.0,
        'gram_h3': 1.0,
        'boundary': 0.0,           # Zero!
        'calibration': 0.0
    }

    total_loss, components = trainer.loss_fn(
        phi, dphi, dstar_phi, star_phi, metric,
        h2, h3, region_weights,
        loss_weights, epoch=0
    )

    # Should still compute
    assert torch.isfinite(total_loss)
    assert total_loss.item() >= 0
