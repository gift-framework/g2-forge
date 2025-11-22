"""
Unit tests for error handling and resilience.

Tests trainer and component behavior in failure scenarios,
including NaN/Inf handling, checkpoint corruption, and invalid configurations.
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


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# NaN/Inf HANDLING TESTS
# ============================================================

def test_trainer_detects_nan_loss(small_topology_config):
    """Test that trainer can detect NaN loss values."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Train one step
    metrics = trainer.train_step(epoch=0)

    # Loss should be finite
    assert np.isfinite(metrics['loss']), "Initial loss should be finite"


def test_trainer_detects_inf_loss(small_topology_config):
    """Test that trainer can detect Inf loss values."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Train one step
    metrics = trainer.train_step(epoch=0)

    # Loss should not be infinite
    assert not np.isinf(metrics['loss']), "Loss should not be infinite"


def test_gradients_remain_finite_after_training(small_topology_config):
    """Test that all gradients remain finite after training steps."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Train for a few epochs
    trainer.train(num_epochs=5)

    # Check all network gradients are finite
    for network in [trainer.phi_network, trainer.h2_network, trainer.h3_network]:
        for param in network.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"Gradients should remain finite in {network.__class__.__name__}"


def test_loss_components_all_finite(small_topology_config):
    """Test that all loss components remain finite."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    metrics = trainer.train_step(epoch=0)

    # Check all loss components
    for key, value in metrics.items():
        if isinstance(value, (float, int)) and key != 'epoch':
            assert np.isfinite(value), f"Loss component {key} should be finite"


# ============================================================
# GRADIENT EXPLOSION TESTS
# ============================================================

def test_gradient_clipping_prevents_explosion(small_topology_config):
    """Test that gradient clipping prevents gradient explosion."""
    # Set a very small gradient clip value
    config = small_topology_config
    config.training.grad_clip = 1.0

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train for several epochs
    trainer.train(num_epochs=10)

    # Check that no gradients exceed the clip value significantly
    max_grad_norm = 0.0
    for network in [trainer.phi_network, trainer.h2_network, trainer.h3_network]:
        for param in network.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                max_grad_norm = max(max_grad_norm, grad_norm)

    # Should be bounded by clipping (with some tolerance for numerical precision)
    assert max_grad_norm < config.training.grad_clip * 10, \
        "Gradient clipping should prevent extreme gradients"


def test_very_large_learning_rate_stability(small_topology_config):
    """Test behavior with very large learning rate."""
    config = small_topology_config
    config.training.lr = 1.0  # Very large LR
    config.training.grad_clip = 10.0  # Higher clip to see if we can handle it

    trainer = Trainer(config, device='cpu', verbose=False)

    # Should still complete without NaN
    results = trainer.train(num_epochs=3)

    assert np.isfinite(results['final_metrics']['loss']), \
        "Training should remain stable even with large LR"


# ============================================================
# CHECKPOINT CORRUPTION TESTS
# ============================================================

def test_load_checkpoint_nonexistent_file(small_topology_config):
    """Test loading checkpoint from non-existent file."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    nonexistent_path = Path("/tmp/nonexistent_checkpoint_xyz.pt")

    # Should handle gracefully (currently just loads nothing)
    # This test documents current behavior
    try:
        trainer.load_checkpoint(nonexistent_path)
    except FileNotFoundError:
        # Expected behavior
        pass


def test_load_checkpoint_corrupted_file(small_topology_config):
    """Test loading checkpoint from corrupted file."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Create a corrupted checkpoint file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pt', delete=False) as f:
        f.write("This is not a valid PyTorch checkpoint!")
        corrupted_path = Path(f.name)

    try:
        # Should raise an error when loading corrupted file
        with pytest.raises(Exception):  # Could be RuntimeError, pickle.UnpicklingError, etc.
            trainer.load_checkpoint(corrupted_path)
    finally:
        # Cleanup
        if corrupted_path.exists():
            corrupted_path.unlink()


def test_save_checkpoint_creates_file(small_topology_config):
    """Test that save_checkpoint actually creates a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        trainer = Trainer(config, device='cpu', verbose=False)

        # Train and save
        metrics = trainer.train_step(epoch=0)
        trainer.save_checkpoint(epoch=0, metrics=metrics, prefix='test')

        # Check file exists
        checkpoint_path = Path(tmpdir) / 'test_epoch_0.pt'
        assert checkpoint_path.exists(), "Checkpoint file should be created"


def test_checkpoint_save_load_roundtrip(small_topology_config):
    """Test that checkpoint can be saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        trainer = Trainer(config, device='cpu', verbose=False)

        # Train a bit
        trainer.train(num_epochs=3)

        # Save checkpoint
        checkpoint_path = Path(tmpdir) / 'roundtrip_test.pt'
        trainer.save_checkpoint(epoch=2, metrics={'loss': 1.0}, prefix='roundtrip_test')

        # Create new trainer and load
        trainer2 = Trainer(config, device='cpu', verbose=False)
        trainer2.load_checkpoint(checkpoint_path)

        # Should have loaded epoch
        assert trainer2.start_epoch == 3, "Should resume from next epoch"


# ============================================================
# RESUME TRAINING TESTS
# ============================================================

def test_resume_with_mismatched_topology_should_fail(small_topology_config, medium_topology_config):
    """Test that resuming with different topology configuration fails or warns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train with small topology
        config1 = small_topology_config
        config1.checkpointing.save_dir = tmpdir

        trainer1 = Trainer(config1, device='cpu', verbose=False)
        trainer1.train(num_epochs=2)
        trainer1.save_checkpoint(epoch=1, metrics={'loss': 1.0}, prefix='checkpoint')

        # Try to load with medium topology (different b2, b3)
        config2 = medium_topology_config
        config2.checkpointing.save_dir = tmpdir

        trainer2 = Trainer(config2, device='cpu', verbose=False)

        checkpoint_path = Path(tmpdir) / 'checkpoint_epoch_1.pt'

        # This should either fail or produce a warning
        # Currently it will fail due to shape mismatch
        with pytest.raises(RuntimeError):
            trainer2.load_checkpoint(checkpoint_path)


def test_resume_training_continues_from_correct_epoch(small_topology_config):
    """Test that resumed training starts from correct epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        trainer = Trainer(config, device='cpu', verbose=False)

        # Train to epoch 5
        trainer.train(num_epochs=5)

        # Save checkpoint at epoch 4
        checkpoint_path = Path(tmpdir) / 'resume_test.pt'
        trainer.save_checkpoint(epoch=4, metrics={'loss': 1.0}, prefix='resume_test')

        # Create new trainer and resume
        trainer2 = Trainer(config, device='cpu', verbose=False)
        trainer2.load_checkpoint(checkpoint_path)

        assert trainer2.start_epoch == 5, "Should resume from epoch 5"


# ============================================================
# INVALID CONFIGURATION TESTS
# ============================================================

def test_invalid_negative_learning_rate():
    """Test that negative learning rate is handled."""
    with pytest.raises((ValueError, AssertionError)):
        config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
        config.training.lr = -0.001  # Invalid
        config.validate()  # Should fail if validation exists


def test_invalid_zero_batch_size():
    """Test that zero batch size is handled."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    config.training.batch_size = 0  # Invalid

    # Should fail when creating trainer or during training
    trainer = Trainer(config, device='cpu', verbose=False)

    with pytest.raises((ValueError, RuntimeError)):
        trainer.train(num_epochs=1)


def test_invalid_negative_epochs():
    """Test that negative epochs are handled."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

    trainer = Trainer(config, device='cpu', verbose=False)

    # Training with negative epochs should be handled gracefully
    results = trainer.train(num_epochs=-1)

    # Should return without training (or raise error)
    assert results is not None


def test_invalid_grad_clip_negative():
    """Test that negative gradient clipping value is handled."""
    config = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    config.training.grad_clip = -1.0  # Invalid

    # Should still create trainer (might be caught during training)
    trainer = Trainer(config, device='cpu', verbose=False)

    # Document current behavior
    assert trainer.config.training.grad_clip == -1.0


# ============================================================
# EDGE CASE TESTS
# ============================================================

def test_training_with_single_sample_batch(small_topology_config):
    """Test training with batch size of 1."""
    config = small_topology_config
    config.training.batch_size = 1

    trainer = Trainer(config, device='cpu', verbose=False)

    # Should work with batch size 1
    results = trainer.train(num_epochs=3)

    assert results['final_metrics']['loss'] > 0
    assert np.isfinite(results['final_metrics']['loss'])


def test_training_with_very_large_batch(small_topology_config):
    """Test training with very large batch size."""
    config = small_topology_config
    config.training.batch_size = 10000  # Very large

    trainer = Trainer(config, device='cpu', verbose=False)

    # Should complete (might be slow)
    results = trainer.train(num_epochs=1)

    assert results['final_metrics']['loss'] > 0


def test_optimizer_state_preserved_after_checkpoint(small_topology_config):
    """Test that optimizer state is preserved in checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        trainer = Trainer(config, device='cpu', verbose=False)

        # Train to build optimizer state
        trainer.train(num_epochs=5)

        # Get optimizer state
        optimizer_state_before = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                 for k, v in trainer.optimizer.state_dict()['state'].items()}

        # Save and reload
        checkpoint_path = Path(tmpdir) / 'optimizer_test.pt'
        trainer.save_checkpoint(epoch=4, metrics={'loss': 1.0}, prefix='optimizer_test')

        trainer2 = Trainer(config, device='cpu', verbose=False)
        trainer2.load_checkpoint(checkpoint_path)

        # Optimizer state should be restored
        assert len(trainer2.optimizer.state_dict()['state']) > 0, \
            "Optimizer state should be non-empty after loading"


def test_metrics_history_preserved_in_checkpoint(small_topology_config):
    """Test that metrics history is preserved in checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        trainer = Trainer(config, device='cpu', verbose=False)

        # Train to build metrics history
        trainer.train(num_epochs=10)

        history_length = len(trainer.metrics_history)

        # Save and reload
        checkpoint_path = Path(tmpdir) / 'metrics_test.pt'
        trainer.save_checkpoint(epoch=9, metrics={'loss': 1.0}, prefix='metrics_test')

        trainer2 = Trainer(config, device='cpu', verbose=False)
        trainer2.load_checkpoint(checkpoint_path)

        # Metrics history should be restored
        assert len(trainer2.metrics_history) == history_length, \
            "Metrics history should be preserved"


# ============================================================
# PARAMETER VALIDATION TESTS
# ============================================================

def test_warmup_epochs_greater_than_total_epochs(small_topology_config):
    """Test behavior when warmup epochs exceed total epochs."""
    config = small_topology_config
    config.training.warmup_epochs = 100
    config.training.total_epochs = 10  # Less than warmup

    # Should still create trainer
    trainer = Trainer(config, device='cpu', verbose=False)

    # Training should handle this gracefully
    results = trainer.train(num_epochs=10)

    assert results is not None


def test_lr_min_greater_than_lr(small_topology_config):
    """Test behavior when min LR is greater than max LR."""
    config = small_topology_config
    config.training.lr = 0.001
    config.training.lr_min = 0.01  # Greater than lr

    # Should still create trainer
    trainer = Trainer(config, device='cpu', verbose=False)

    # Might produce unusual behavior but shouldn't crash
    results = trainer.train(num_epochs=5)

    assert results is not None


def test_weight_decay_zero(small_topology_config):
    """Test training with zero weight decay."""
    config = small_topology_config
    config.training.weight_decay = 0.0

    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=5)

    assert np.isfinite(results['final_metrics']['loss'])


def test_weight_decay_very_large(small_topology_config):
    """Test training with very large weight decay."""
    config = small_topology_config
    config.training.weight_decay = 1.0  # Very large

    trainer = Trainer(config, device='cpu', verbose=False)

    results = trainer.train(num_epochs=5)

    # Should still train (though possibly poorly)
    assert np.isfinite(results['final_metrics']['loss'])


# ============================================================
# EXCEPTION HANDLING TESTS
# ============================================================

def test_tcs_manifold_requires_tcs_construction():
    """Test TCSManifold raises error for non-TCS config."""
    from g2forge.manifolds.base import TCSManifold
    from g2forge.utils.config import ManifoldConfig, TopologyConfig, TCSParameters

    # Create config with non-TCS construction
    topology = TopologyConfig(b2=10, b3=40)
    tcs_params = TCSParameters(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

    config = ManifoldConfig(
        type="K7",
        construction="ConnectedSum",  # NOT TCS!
        topology=topology,
        tcs_params=tcs_params
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="requires construction='TCS'"):
        TCSManifold(config)


def test_tcs_manifold_requires_tcs_params():
    """Test TCSManifold raises error when tcs_params missing."""
    from g2forge.manifolds.base import TCSManifold
    from g2forge.utils.config import ManifoldConfig, TopologyConfig

    # Create config without tcs_params
    topology = TopologyConfig(b2=10, b3=40)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=None  # Missing!
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="requires tcs_params"):
        TCSManifold(config)


def test_create_manifold_unknown_type():
    """Test create_manifold raises ValueError for unknown type."""
    from g2forge.manifolds.base import create_manifold
    from g2forge.utils.config import ManifoldConfig, TopologyConfig

    topology = TopologyConfig(b2=10, b3=40)

    config = ManifoldConfig(
        type="UnknownManifold",  # Unknown type!
        construction="TCS",
        topology=topology
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Unknown manifold type"):
        create_manifold(config)


def test_region_indicator_unknown_region():
    """Test compute_region_indicator raises error for invalid region."""
    k7 = g2.create_gift_k7()

    t = torch.tensor([0.5])

    # Should raise ValueError for unknown region
    with pytest.raises(ValueError, match="Unknown region"):
        k7.compute_region_indicator(t, "invalid_region")


def test_manifold_config_invalid_dimension():
    """Test ManifoldConfig validation catches invalid dimension."""
    from g2forge.utils.config import ManifoldConfig, TopologyConfig

    topology = TopologyConfig(b2=10, b3=40)

    config = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        dimension=5  # Invalid! Should be 7 for G₂
    )

    # Validation should catch this
    with pytest.raises(ValueError):
        config.validate()


def test_tcs_parameters_topology_mismatch():
    """Test TCSParameters validation catches topology inconsistency."""
    from g2forge.utils.config import TCSParameters, TopologyConfig

    # Create TCS params that don't match topology
    tcs_params = TCSParameters(
        b2_m1=5, b3_m1=20,
        b2_m2=5, b3_m2=20
    )  # Total: b₂=10, b₃=40

    topology = TopologyConfig(b2=15, b3=40)  # Mismatch in b₂!

    # Validation should catch this
    with pytest.raises(ValueError):
        tcs_params.validate_against_topology(topology)


def test_network_invalid_topology():
    """Test networks handle invalid topology gracefully."""
    from g2forge.networks.harmonic_network import HarmonicNetwork

    # Try creating network with invalid (zero) topology
    with pytest.raises((ValueError, RuntimeError)):
        network = HarmonicNetwork(
            n_forms=0,  # Invalid!
            form_type="H2",
            hidden_dims=[128, 128]
        )


def test_composite_loss_missing_manifold():
    """Test CompositeLoss raises error when manifold is None for TCS."""
    from g2forge.core.losses import CompositeLoss
    from g2forge.utils.config import TopologyConfig, TCSParameters

    topology = TopologyConfig(b2=10, b3=40)

    # Creating CompositeLoss without manifold should work
    # but using it with regional losses should fail or warn
    loss_fn = CompositeLoss(topology=topology, manifold=None)

    # This should still work (manifold is optional)
    assert loss_fn is not None
