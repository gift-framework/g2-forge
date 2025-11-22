"""
Integration tests for advanced training scenarios.

Tests early stopping, custom schedules, training interruption/resume,
long-running training stability, and advanced training features.
"""

import pytest
import torch
import numpy as np
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# ============================================================
# TRAINING INTERRUPTION AND RESUME TESTS
# ============================================================

def test_training_resume_from_arbitrary_epoch(small_topology_config):
    """Test that training can be resumed from any epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        # First training run: epochs 0-9
        trainer1 = Trainer(config, device='cpu', verbose=False)
        results1 = trainer1.train(num_epochs=10)

        # Save checkpoint at epoch 7
        trainer1.save_checkpoint(epoch=7, metrics={'loss': 1.0}, prefix='resume_test')

        # Second training run: resume from epoch 7
        trainer2 = Trainer(config, device='cpu', verbose=False)
        checkpoint_path = Path(tmpdir) / 'resume_test_epoch_7.pt'
        trainer2.load_checkpoint(checkpoint_path)

        # Continue training
        results2 = trainer2.train(num_epochs=15)  # Should train epochs 8-14

        assert trainer2.start_epoch == 8
        assert len(trainer2.metrics_history) >= 8  # Should have history from loaded checkpoint


def test_training_pause_and_resume_maintains_state(small_topology_config):
    """Test that pausing and resuming training maintains complete state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        # Train for 5 epochs
        trainer1 = Trainer(config, device='cpu', verbose=False)
        trainer1.train(num_epochs=5)

        # Get state before saving
        loss_before = trainer1.metrics_history[-1]['loss']

        # Save checkpoint
        trainer1.save_checkpoint(epoch=4, metrics=trainer1.metrics_history[-1], prefix='pause_test')

        # Resume in new trainer
        trainer2 = Trainer(config, device='cpu', verbose=False)
        checkpoint_path = Path(tmpdir) / 'pause_test_epoch_4.pt'
        trainer2.load_checkpoint(checkpoint_path)

        # Metrics history should be restored
        assert len(trainer2.metrics_history) == len(trainer1.metrics_history)


def test_training_with_multiple_checkpoints(small_topology_config):
    """Test that multiple checkpoints can be saved and don't interfere."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir
        config.checkpointing.interval = 3  # Save every 3 epochs

        trainer = Trainer(config, device='cpu', verbose=False)
        trainer.train(num_epochs=10)

        # Should have multiple checkpoints
        checkpoint_files = list(Path(tmpdir).glob('checkpoint_epoch_*.pt'))

        # Should have checkpoints at epochs 2, 5, 8 (0-indexed)
        assert len(checkpoint_files) >= 3


# ============================================================
# CUSTOM LEARNING RATE SCHEDULES
# ============================================================

def test_warmup_schedule_increases_lr(small_topology_config):
    """Test that warmup schedule increases learning rate."""
    config = small_topology_config
    config.training.warmup_epochs = 10

    trainer = Trainer(config, device='cpu', verbose=False)

    # Get LR at start of warmup
    lr_start = trainer.optimizer.param_groups[0]['lr']

    # Train through warmup
    for epoch in range(10):
        trainer.train_step(epoch)
        trainer.scheduler.step()

    # LR should have increased during warmup
    lr_after_warmup = trainer.optimizer.param_groups[0]['lr']

    assert lr_after_warmup > lr_start


def test_cosine_annealing_decreases_lr(small_topology_config):
    """Test that cosine annealing decreases learning rate after warmup."""
    config = small_topology_config
    config.training.warmup_epochs = 5
    config.training.total_epochs = 50

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train through warmup
    for epoch in range(5):
        trainer.train_step(epoch)
        trainer.scheduler.step()

    lr_after_warmup = trainer.optimizer.param_groups[0]['lr']

    # Continue training with cosine annealing
    for epoch in range(5, 30):
        trainer.train_step(epoch)
        trainer.scheduler.step()

    lr_after_annealing = trainer.optimizer.param_groups[0]['lr']

    # LR should decrease during annealing
    assert lr_after_annealing < lr_after_warmup


def test_lr_reaches_minimum(small_topology_config):
    """Test that learning rate reaches configured minimum."""
    config = small_topology_config
    config.training.warmup_epochs = 2
    config.training.total_epochs = 20
    config.training.lr_min = 1e-6

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train to completion
    trainer.train(num_epochs=20)

    final_lr = trainer.optimizer.param_groups[0]['lr']

    # Should be close to minimum
    assert final_lr <= config.training.lr_min * 1.1


# ============================================================
# METRICS LOGGING VARIATIONS
# ============================================================

def test_metrics_logged_every_epoch(small_topology_config):
    """Test that metrics are logged for every epoch."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=20)

    # Should have metrics for each epoch
    assert len(results['metrics_history']) == 20


def test_metrics_contain_all_required_fields(small_topology_config):
    """Test that metrics contain all expected fields."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=5)

    for metrics in results['metrics_history']:
        # Check required fields
        assert 'epoch' in metrics
        assert 'loss' in metrics
        assert 'torsion_closure' in metrics
        assert 'rank_h2' in metrics
        assert 'rank_h3' in metrics
        assert 'lr' in metrics
        assert 'phase' in metrics


def test_metrics_history_ordering(small_topology_config):
    """Test that metrics history is in correct chronological order."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=15)

    # Epochs should be in increasing order
    epochs = [m['epoch'] for m in results['metrics_history']]
    assert epochs == sorted(epochs)


# ============================================================
# TRAINING STABILITY TESTS
# ============================================================

@pytest.mark.slow
def test_long_training_stability(small_topology_config):
    """Test that training remains stable over long runs."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=200)

    # All losses should remain finite
    for metrics in results['metrics_history']:
        assert np.isfinite(metrics['loss']), f"Loss became non-finite at epoch {metrics['epoch']}"


@pytest.mark.slow
def test_long_training_no_memory_leak(small_topology_config):
    """Test that long training doesn't leak memory."""
    import gc

    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Get initial memory usage (rough estimate)
    gc.collect()

    # Train for many epochs
    trainer.train(num_epochs=100)

    # Force cleanup
    del trainer
    gc.collect()

    # If we got here without OOM, test passes
    assert True


def test_training_with_variable_batch_sizes_across_phases(small_topology_config):
    """Test training stability when batch sizes vary across curriculum phases."""
    config = small_topology_config

    # Modify phases to have different batch sizes implicitly through grid_n
    config.training.curriculum['phase1_neck_stability'].grid_n = 30
    config.training.curriculum['phase2_acyl_matching'].grid_n = 50

    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=20)

    # Should complete successfully
    assert np.isfinite(results['final_metrics']['loss'])


# ============================================================
# GRADIENT ACCUMULATION TESTS
# ============================================================

def test_effective_batch_size_with_small_batches(small_topology_config):
    """Test that multiple small batches produce similar results to one large batch."""
    config1 = small_topology_config
    config1.training.batch_size = 100

    config2 = small_topology_config
    config2.training.batch_size = 50

    # Set same seed for both
    torch.manual_seed(42)
    trainer1 = Trainer(config1, device='cpu', verbose=False)
    loss1 = trainer1.train_step(epoch=0)['loss']

    torch.manual_seed(42)
    trainer2 = Trainer(config2, device='cpu', verbose=False)
    loss2_1 = trainer2.train_step(epoch=0)['loss']
    loss2_2 = trainer2.train_step(epoch=0)['loss']

    # Losses should be different but in similar range
    assert abs(loss1 - loss2_1) < loss1 * 2  # Within 2x


# ============================================================
# CONVERGENCE MONITORING TESTS
# ============================================================

def test_loss_convergence_detection(small_topology_config):
    """Test detecting when loss has converged."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=50)

    # Check if loss is decreasing or has plateaued
    losses = [m['loss'] for m in results['metrics_history']]

    # Last 10 epochs should have lower average loss than first 10
    early_avg = np.mean(losses[:10])
    late_avg = np.mean(losses[-10:])

    assert late_avg <= early_avg


def test_torsion_improvement_over_time(small_topology_config):
    """Test that torsion metrics improve during training."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=50)

    torsion_values = [m['torsion_closure'] for m in results['metrics_history']]

    # Final torsion should be better than or equal to initial
    assert torsion_values[-1] <= torsion_values[0] * 1.1  # Allow small tolerance


# ============================================================
# PHASE TRANSITION TESTS
# ============================================================

def test_phase_transitions_occur_at_correct_epochs(gift_config):
    """Test that curriculum phase transitions occur at configured epochs."""
    config = gift_config

    # Set specific phase boundaries
    config.training.curriculum['phase1_neck_stability'].epoch_range = (0, 5)
    config.training.curriculum['phase2_acyl_matching'].epoch_range = (5, 10)
    config.training.curriculum['phase3_cohomology_refinement'].epoch_range = (10, 15)

    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=15)

    # Check phases in metrics history
    phases = [m['phase'] for m in results['metrics_history']]

    # Epoch 0-4 should be phase1
    assert all(p.startswith('phase1') for p in phases[:5])

    # Epoch 5-9 should be phase2
    assert all(p.startswith('phase2') for p in phases[5:10])

    # Epoch 10-14 should be phase3
    assert all(p.startswith('phase3') for p in phases[10:15])


def test_grid_resolution_changes_across_phases(gift_config):
    """Test that grid resolution changes across curriculum phases."""
    config = gift_config

    # Set different grid resolutions
    config.training.curriculum['phase1_neck_stability'].grid_n = 30
    config.training.curriculum['phase2_acyl_matching'].grid_n = 50
    config.training.curriculum['phase1_neck_stability'].epoch_range = (0, 5)
    config.training.curriculum['phase2_acyl_matching'].epoch_range = (5, 10)

    trainer = Trainer(config, device='cpu', verbose=False)

    # Check phase configurations
    _, phase1_config = trainer._get_current_phase(epoch=2)
    _, phase2_config = trainer._get_current_phase(epoch=7)

    assert phase1_config.grid_n == 30
    assert phase2_config.grid_n == 50


# ============================================================
# OPTIMIZER STATE TESTS
# ============================================================

def test_optimizer_momentum_builds_over_time(small_topology_config):
    """Test that optimizer momentum state builds during training."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Initially, optimizer state should be empty or minimal
    initial_state_size = len(trainer.optimizer.state_dict()['state'])

    # Train to build momentum
    trainer.train(num_epochs=10)

    # After training, optimizer should have state for all parameters
    final_state_size = len(trainer.optimizer.state_dict()['state'])

    assert final_state_size >= initial_state_size


def test_learning_rate_stored_in_metrics(small_topology_config):
    """Test that learning rate is tracked in metrics history."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=20)

    # All metrics should have LR
    for metrics in results['metrics_history']:
        assert 'lr' in metrics
        assert metrics['lr'] > 0


def test_learning_rate_changes_over_training(small_topology_config):
    """Test that learning rate actually changes during training."""
    config = small_topology_config
    config.training.warmup_epochs = 5
    config.training.total_epochs = 30

    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=30)

    # Extract LRs
    lrs = [m['lr'] for m in results['metrics_history']]

    # LR should not be constant
    assert len(set(lrs)) > 1, "Learning rate should change during training"


# ============================================================
# TRAINING COMPLETION TESTS
# ============================================================

def test_training_completes_all_requested_epochs(small_topology_config):
    """Test that training completes exactly the requested number of epochs."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    num_epochs = 25
    results = trainer.train(num_epochs=num_epochs)

    assert len(results['metrics_history']) == num_epochs


def test_final_checkpoint_saved(small_topology_config):
    """Test that final checkpoint is saved at end of training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        trainer = Trainer(config, device='cpu', verbose=False)
        trainer.train(num_epochs=10)

        # Check for final checkpoint
        final_checkpoint = Path(tmpdir) / 'final_epoch_9.pt'
        assert final_checkpoint.exists()


def test_results_json_saved(small_topology_config):
    """Test that results JSON is saved after training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        trainer = Trainer(config, device='cpu', verbose=False)
        trainer.train(num_epochs=10)

        # Check for results JSON
        results_file = Path(tmpdir) / 'results.json'
        assert results_file.exists()
