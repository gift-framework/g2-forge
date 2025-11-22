"""
Integration tests for GIFT v1.0 reproduction.

Tests that g2-forge can reproduce GIFT v1.0 results with
the same configuration and parameters.
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
# GIFT V1.0 CONFIGURATION TESTS
# ============================================================

def test_gift_config_has_correct_topology():
    """Test that GIFT v1.0 config has correct topology."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    assert config.manifold.topology.b2 == 21
    assert config.manifold.topology.b3 == 77


def test_gift_config_has_correct_tcs_parameters():
    """Test that GIFT v1.0 config has correct TCS parameters."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    assert config.manifold.tcs_params.b2_m1 == 11
    assert config.manifold.tcs_params.b3_m1 == 40
    assert config.manifold.tcs_params.b2_m2 == 10
    assert config.manifold.tcs_params.b3_m2 == 37

    # Total should match topology
    assert config.manifold.tcs_params.total_b2 == 21
    assert config.manifold.tcs_params.total_b3 == 77


def test_gift_config_has_5_curriculum_phases():
    """Test that GIFT v1.0 config has 5 curriculum phases."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    curriculum = config.training.curriculum

    assert len(curriculum) == 5

    # Check phase names
    expected_phases = [
        'phase1_neck_stability',
        'phase2_acyl_matching',
        'phase3_cohomology_refinement',
        'phase4_harmonic_extraction',
        'phase5_calibration_finetune'
    ]

    for phase_name in expected_phases:
        assert phase_name in curriculum


def test_gift_config_phase_epoch_ranges():
    """Test that GIFT curriculum phases have correct epoch ranges."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    curriculum = config.training.curriculum

    # Phase 1: 0-2000
    assert curriculum['phase1_neck_stability'].epoch_range == (0, 2000)

    # Phase 2: 2000-5000
    assert curriculum['phase2_acyl_matching'].epoch_range == (2000, 5000)

    # Phase 3: 5000-8000
    assert curriculum['phase3_cohomology_refinement'].epoch_range == (5000, 8000)

    # Phase 4: 8000-10000
    assert curriculum['phase4_harmonic_extraction'].epoch_range == (8000, 10000)

    # Phase 5: 10000-15000
    assert curriculum['phase5_calibration_finetune'].epoch_range == (10000, 15000)


def test_gift_config_version_tag():
    """Test that GIFT config has correct version tag."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    assert 'gift' in config.version.lower()
    assert 'v1.0' in config.version or '1.0' in config.version


# ============================================================
# GIFT V1.0 TRAINING TESTS
# ============================================================

def test_gift_trainer_initialization():
    """Test that Trainer initializes with GIFT v1.0 config."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    trainer = Trainer(config, device='cpu', verbose=False)

    # Check network sizes
    assert trainer.h2_network.n_forms == 21
    assert trainer.h3_network.n_forms == 77

    # Check manifold
    assert trainer.manifold.b2 == 21
    assert trainer.manifold.b3 == 77


@pytest.mark.slow
def test_gift_short_training_completes():
    """Test that GIFT configuration completes short training run."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    # Reduce epochs for testing
    config.training.total_epochs = 50

    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=50)

    # Should complete without errors
    assert 'final_metrics' in results
    assert 'loss' in results['final_metrics']


@pytest.mark.slow
def test_gift_training_metrics_structure():
    """Test that GIFT training produces expected metrics."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=10)

    final = results['final_metrics']

    # Should have GIFT-specific metrics
    assert 'torsion_closure' in final
    assert 'torsion_coclosure' in final
    assert 'rank_h2' in final
    assert 'rank_h3' in final

    # Ranks should be <= Betti numbers
    assert final['rank_h2'] <= 21
    assert final['rank_h3'] <= 77


# ============================================================
# GIFT V1.0 EXPECTED BEHAVIOR TESTS
# ============================================================

def test_gift_networks_have_correct_output_sizes():
    """Test that GIFT networks have correct output dimensions."""
    config = g2.G2ForgeConfig.from_gift_v1_0()
    trainer = Trainer(config, device='cpu', verbose=False)

    # Sample coordinates
    coords = trainer.manifold.sample_coordinates(10, device='cpu')

    # PhiNetwork: should output 35 components (C(7,3))
    phi = trainer.phi_network(coords)
    assert phi.shape == (10, 35)

    # H² network: should output 21 forms
    h2 = trainer.h2_network(coords)
    assert h2.shape == (10, 21, 21)  # [batch, n_forms=21, n_components=21]

    # H³ network: should output 77 forms
    h3 = trainer.h3_network(coords)
    assert h3.shape == (10, 77, 35)  # [batch, n_forms=77, n_components=35]


def test_gift_loss_function_uses_correct_topology():
    """Test that GIFT loss function uses correct topology."""
    config = g2.G2ForgeConfig.from_gift_v1_0()
    trainer = Trainer(config, device='cpu', verbose=False)

    # Loss function should be parameterized by GIFT topology
    assert trainer.loss_fn.topology.b2 == 21
    assert trainer.loss_fn.topology.b3 == 77


def test_gift_manifold_region_weights():
    """Test that GIFT manifold provides region weights."""
    config = g2.G2ForgeConfig.from_gift_v1_0()
    trainer = Trainer(config, device='cpu', verbose=False)

    coords = trainer.manifold.sample_coordinates(100, device='cpu')
    weights = trainer.manifold.get_region_weights(coords)

    # Should have TCS regions
    assert 'm1' in weights
    assert 'neck' in weights
    assert 'm2' in weights

    # Weights should sum to 1
    total = weights['m1'] + weights['neck'] + weights['m2']
    torch.testing.assert_close(total, torch.ones(100), rtol=1e-4, atol=1e-5)


# ============================================================
# GIFT V1.0 CONVERGENCE EXPECTATIONS
# ============================================================

@pytest.mark.slow
def test_gift_torsion_reduces_during_training():
    """Test that torsion reduces during GIFT training."""
    config = g2.G2ForgeConfig.from_gift_v1_0()
    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=50)

    initial_torsion = results['initial_metrics']['torsion_closure']
    final_torsion = results['final_metrics']['torsion_closure']

    # Torsion should decrease
    # (may not be strict due to short training)
    assert final_torsion <= initial_torsion * 1.1  # Allow 10% slack


@pytest.mark.slow
def test_gift_rank_improves_during_training():
    """Test that harmonic rank improves during GIFT training."""
    config = g2.G2ForgeConfig.from_gift_v1_0()
    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=50)

    initial_rank_h2 = results['initial_metrics'].get('rank_h2', 0)
    final_rank_h2 = results['final_metrics']['rank_h2']

    initial_rank_h3 = results['initial_metrics'].get('rank_h3', 0)
    final_rank_h3 = results['final_metrics']['rank_h3']

    # Ranks should improve or stay constant
    assert final_rank_h2 >= initial_rank_h2
    assert final_rank_h3 >= initial_rank_h3


# ============================================================
# GIFT V1.0 PARAMETER VALIDATION
# ============================================================

def test_gift_optimizer_parameters():
    """Test that GIFT uses correct optimizer parameters."""
    config = g2.G2ForgeConfig.from_gift_v1_0()
    trainer = Trainer(config, device='cpu', verbose=False)

    # Check optimizer type
    assert isinstance(trainer.optimizer, torch.optim.AdamW)

    # Check learning rate
    lr = trainer.optimizer.param_groups[0]['lr']
    assert lr == config.training.lr

    # Check weight decay
    wd = trainer.optimizer.param_groups[0]['weight_decay']
    assert wd == config.training.weight_decay


def test_gift_training_parameters():
    """Test that GIFT config has expected training parameters."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    # Total epochs
    assert config.training.total_epochs == 15000

    # Batch size
    assert config.training.batch_size == 2048

    # Learning rate
    assert config.training.lr == 1e-4

    # Warmup
    assert config.training.warmup_epochs == 500


def test_gift_network_architecture():
    """Test that GIFT uses expected network architecture."""
    config = g2.G2ForgeConfig.from_gift_v1_0()

    arch = config.architecture

    # PhiNetwork architecture
    assert arch.phi_hidden_dims == [384, 384, 256]
    assert arch.phi_n_fourier == 32

    # Harmonic network architecture
    assert arch.h2_hidden_dim == 128
    assert arch.h2_n_fourier == 24
    assert arch.h3_hidden_dim == 128
    assert arch.h3_n_fourier == 24


# ============================================================
# GIFT V1.0 EXPECTED OUTCOMES (REFERENCE VALUES)
# ============================================================

@pytest.mark.slow
def test_gift_expected_torsion_range():
    """
    Test that GIFT torsion is in expected range after training.

    Note: This is a weak test since we only train briefly.
    Full GIFT v1.0 achieves torsion ~1e-7 to 1e-11 after 15k epochs.
    """
    config = g2.G2ForgeConfig.from_gift_v1_0()
    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=100)

    final_torsion = results['final_metrics']['torsion_closure']

    # After 100 epochs, torsion should be finite and decreasing
    # (not the final 1e-11 target, but reasonable)
    assert final_torsion < 100.0  # Should have decreased significantly
    assert np.isfinite(final_torsion)
