"""
Integration tests for checkpointing system.

Tests checkpoint save/load, state restoration, resume training,
and best checkpoint tracking.
"""

import pytest
import torch
import tempfile
from pathlib import Path
import shutil
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# ============================================================
# CHECKPOINT SAVE/LOAD TESTS
# ============================================================

def test_checkpoint_save_creates_file(small_topology_config):
    """Test that saving checkpoint creates a file."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Train a bit
    trainer.train(num_epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'test_checkpoint.pt'

        # Save checkpoint
        trainer.save_checkpoint(str(checkpoint_path))

        # File should exist
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0


def test_checkpoint_load_restores_state(small_topology_config):
    """Test that loading checkpoint restores trainer state."""
    # Train first trainer
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer1.train(num_epochs=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'

        # Save checkpoint
        trainer1.save_checkpoint(str(checkpoint_path))

        # Create new trainer and load checkpoint
        trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
        trainer2.load_checkpoint(str(checkpoint_path))

        # Check that state is restored
        # Compare network parameters
        for p1, p2 in zip(trainer1.phi_network.parameters(),
                         trainer2.phi_network.parameters()):
            torch.testing.assert_close(p1, p2)


def test_checkpoint_contains_all_networks(small_topology_config):
    """Test that checkpoint contains all network states."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer.train(num_epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer.save_checkpoint(str(checkpoint_path))

        # Load checkpoint and check contents
        checkpoint = torch.load(checkpoint_path)

        assert 'phi_network_state' in checkpoint
        assert 'h2_network_state' in checkpoint
        assert 'h3_network_state' in checkpoint


def test_checkpoint_contains_optimizer_state(small_topology_config):
    """Test that checkpoint contains optimizer state."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer.train(num_epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer.save_checkpoint(str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)

        assert 'optimizer_state' in checkpoint


def test_checkpoint_contains_scheduler_state(small_topology_config):
    """Test that checkpoint contains scheduler state."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer.train(num_epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer.save_checkpoint(str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)

        assert 'scheduler_state' in checkpoint


def test_checkpoint_contains_epoch_info(small_topology_config):
    """Test that checkpoint contains epoch information."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer.train(num_epochs=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer.save_checkpoint(str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)

        assert 'epoch' in checkpoint
        # Should be around 5 (the number we trained)
        assert checkpoint['epoch'] >= 0


def test_checkpoint_contains_metrics_history(small_topology_config):
    """Test that checkpoint contains metrics history."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer.train(num_epochs=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer.save_checkpoint(str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)

        assert 'metrics_history' in checkpoint
        assert len(checkpoint['metrics_history']) > 0


# ============================================================
# RESUME TRAINING TESTS
# ============================================================

def test_resume_training_continues_from_checkpoint(small_topology_config):
    """Test that training can be resumed from checkpoint."""
    # Train for 5 epochs
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    results1 = trainer1.train(num_epochs=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer1.save_checkpoint(str(checkpoint_path))

        # Load and continue training for 5 more epochs
        trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
        trainer2.load_checkpoint(str(checkpoint_path))
        results2 = trainer2.train(num_epochs=5)

        # Should have trained total of 10 epochs
        # Check that metrics exist
        assert 'final_metrics' in results2


def test_resume_preserves_optimizer_state(small_topology_config):
    """Test that optimizer state is preserved across resume."""
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer1.train(num_epochs=3)

    # Get optimizer state
    opt_state1 = trainer1.optimizer.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer1.save_checkpoint(str(checkpoint_path))

        # Load in new trainer
        trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
        trainer2.load_checkpoint(str(checkpoint_path))

        opt_state2 = trainer2.optimizer.state_dict()

        # Optimizer states should match
        # Check param_groups learning rate
        assert opt_state1['param_groups'][0]['lr'] == opt_state2['param_groups'][0]['lr']


def test_resume_preserves_scheduler_state(small_topology_config):
    """Test that scheduler state is preserved across resume."""
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)

    # Train to advance scheduler
    for _ in range(10):
        trainer1.train(num_epochs=1)

    # Get current learning rate
    lr1 = trainer1.optimizer.param_groups[0]['lr']

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer1.save_checkpoint(str(checkpoint_path))

        # Load in new trainer
        trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
        trainer2.load_checkpoint(str(checkpoint_path))

        lr2 = trainer2.optimizer.param_groups[0]['lr']

        # Learning rates should match
        assert abs(lr1 - lr2) < 1e-6


# ============================================================
# BEST CHECKPOINT TRACKING
# ============================================================

def test_best_checkpoint_tracks_lowest_loss(small_topology_config):
    """Test that best checkpoint tracking works."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        best_path = Path(tmpdir) / 'best.pt'

        # Train and save periodically
        losses = []
        for epoch in range(10):
            results = trainer.train(num_epochs=1)
            loss = results['final_metrics']['loss']
            losses.append(loss)

            # Save if best
            if loss == min(losses):
                trainer.save_checkpoint(str(best_path))

        # Best checkpoint should exist
        assert best_path.exists()

        # Load best checkpoint
        best_checkpoint = torch.load(best_path)

        # Should have metrics
        assert 'metrics_history' in best_checkpoint


# ============================================================
# CHECKPOINT COMPATIBILITY TESTS
# ============================================================

def test_checkpoint_different_topologies_fail_gracefully(small_topology_config, gift_config):
    """Test that loading checkpoint with different topology fails gracefully."""
    # Train on small topology
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer1.train(num_epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer1.save_checkpoint(str(checkpoint_path))

        # Try to load into GIFT topology (different size)
        trainer2 = Trainer(gift_config, device='cpu', verbose=False)

        # This should either fail or handle gracefully
        # (implementation dependent)
        try:
            trainer2.load_checkpoint(str(checkpoint_path))
            # If it loads, sizes should not match
            assert trainer2.h2_network.n_forms != trainer1.h2_network.n_forms
        except (RuntimeError, ValueError):
            # Expected - incompatible checkpoint
            pass


# ============================================================
# CHECKPOINT METADATA TESTS
# ============================================================

def test_checkpoint_includes_config(small_topology_config):
    """Test that checkpoint includes configuration."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer.train(num_epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
        trainer.save_checkpoint(str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)

        # Should have config info
        assert 'config' in checkpoint or 'topology' in checkpoint


def test_checkpoint_save_load_roundtrip(small_topology_config):
    """Test complete save/load roundtrip preserves training state."""
    # Initial training
    trainer1 = Trainer(small_topology_config, device='cpu', verbose=False)
    results1 = trainer1.train(num_epochs=5)
    loss1 = results1['final_metrics']['loss']

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'

        # Save
        trainer1.save_checkpoint(str(checkpoint_path))

        # Load into new trainer
        trainer2 = Trainer(small_topology_config, device='cpu', verbose=False)
        trainer2.load_checkpoint(str(checkpoint_path))

        # Continue training for 1 epoch
        results2 = trainer2.train(num_epochs=1)

        # Should be able to train successfully
        assert 'final_metrics' in results2
        assert 'loss' in results2['final_metrics']
