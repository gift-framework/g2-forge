"""
Unit tests for device management and GPU operations.

Tests device placement, CPU/GPU transfers, CUDA operations,
and mixed precision training.
"""

import pytest
import torch
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# DEVICE PLACEMENT TESTS
# ============================================================

def test_trainer_cpu_device_placement(small_topology_config):
    """Test that trainer correctly places all components on CPU."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Check all networks are on CPU
    assert next(trainer.phi_network.parameters()).device.type == 'cpu'
    assert next(trainer.h2_network.parameters()).device.type == 'cpu'
    assert next(trainer.h3_network.parameters()).device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trainer_cuda_device_placement(small_topology_config):
    """Test that trainer correctly places all components on CUDA."""
    trainer = Trainer(small_topology_config, device='cuda', verbose=False)

    # Check all networks are on CUDA
    assert next(trainer.phi_network.parameters()).device.type == 'cuda'
    assert next(trainer.h2_network.parameters()).device.type == 'cuda'
    assert next(trainer.h3_network.parameters()).device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trainer_cuda_training_runs(small_topology_config):
    """Test that training runs successfully on CUDA."""
    trainer = Trainer(small_topology_config, device='cuda', verbose=False)

    # Should train without errors
    results = trainer.train(num_epochs=3)

    assert results['final_metrics']['loss'] > 0
    assert torch.isfinite(torch.tensor(results['final_metrics']['loss']))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_levi_civita_on_cuda(small_topology_config):
    """Test that Levi-Civita tensor is properly placed on CUDA."""
    trainer = Trainer(small_topology_config, device='cuda', verbose=False)

    # Levi-Civita should be on CUDA
    assert trainer.eps_indices.device.type == 'cuda'
    assert trainer.eps_signs.device.type == 'cuda'


# ============================================================
# DEVICE TRANSFER TESTS
# ============================================================

def test_network_move_from_cpu_to_cpu(small_topology_config):
    """Test moving network from CPU to CPU (no-op)."""
    network = g2.networks.create_phi_network_from_config(small_topology_config)

    # Initially on CPU
    assert next(network.parameters()).device.type == 'cpu'

    # Move to CPU (no-op)
    network.to('cpu')

    assert next(network.parameters()).device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_network_move_cpu_to_cuda(small_topology_config):
    """Test moving network from CPU to CUDA."""
    network = g2.networks.create_phi_network_from_config(small_topology_config)

    # Initially on CPU
    assert next(network.parameters()).device.type == 'cpu'

    # Move to CUDA
    network.to('cuda')

    assert next(network.parameters()).device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_network_move_cuda_to_cpu(small_topology_config):
    """Test moving network from CUDA to CPU."""
    network = g2.networks.create_phi_network_from_config(small_topology_config)

    # Move to CUDA
    network.to('cuda')
    assert next(network.parameters()).device.type == 'cuda'

    # Move back to CPU
    network.to('cpu')
    assert next(network.parameters()).device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_manifold_coordinates_device_consistency(small_topology_config):
    """Test that manifold generates coordinates on requested device."""
    manifold = g2.manifolds.create_manifold(small_topology_config.manifold)

    # Sample on CPU
    coords_cpu = manifold.sample_coordinates(n_samples=10, device='cpu')
    assert coords_cpu.device.type == 'cpu'

    # Sample on CUDA
    coords_cuda = manifold.sample_coordinates(n_samples=10, device='cuda')
    assert coords_cuda.device.type == 'cuda'


# ============================================================
# CHECKPOINT DEVICE COMPATIBILITY TESTS
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_save_on_cuda_load_on_cpu(small_topology_config):
    """Test that checkpoint saved on CUDA can be loaded on CPU."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        # Train on CUDA
        trainer_cuda = Trainer(config, device='cuda', verbose=False)
        trainer_cuda.train(num_epochs=2)

        checkpoint_path = Path(tmpdir) / 'cuda_checkpoint.pt'
        trainer_cuda.save_checkpoint(epoch=1, metrics={'loss': 1.0}, prefix='cuda_checkpoint')

        # Load on CPU
        trainer_cpu = Trainer(config, device='cpu', verbose=False)
        trainer_cpu.load_checkpoint(checkpoint_path)

        # Should work and be on CPU
        assert next(trainer_cpu.phi_network.parameters()).device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_save_on_cpu_load_on_cuda(small_topology_config):
    """Test that checkpoint saved on CPU can be loaded on CUDA."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        config = small_topology_config
        config.checkpointing.save_dir = tmpdir

        # Train on CPU
        trainer_cpu = Trainer(config, device='cpu', verbose=False)
        trainer_cpu.train(num_epochs=2)

        checkpoint_path = Path(tmpdir) / 'cpu_checkpoint.pt'
        trainer_cpu.save_checkpoint(epoch=1, metrics={'loss': 1.0}, prefix='cpu_checkpoint')

        # Load on CUDA
        trainer_cuda = Trainer(config, device='cuda', verbose=False)
        trainer_cuda.load_checkpoint(checkpoint_path)

        # Should work and be on CUDA
        assert next(trainer_cuda.phi_network.parameters()).device.type == 'cuda'


# ============================================================
# GPU MEMORY MANAGEMENT TESTS
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_memory_cleanup_after_training(small_topology_config):
    """Test that CUDA memory is properly cleaned up after training."""
    # Clear CUDA cache
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    trainer = Trainer(small_topology_config, device='cuda', verbose=False)
    trainer.train(num_epochs=5)

    # Delete trainer and clear cache
    del trainer
    torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()

    # Memory should not grow unboundedly
    # Allow some tolerance for PyTorch caching
    assert final_memory <= initial_memory * 1.5, \
        "CUDA memory should be mostly cleaned up after training"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multiple_trainers_cuda_memory(small_topology_config):
    """Test that multiple trainer instances don't leak CUDA memory."""
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    # Create and destroy multiple trainers
    for i in range(3):
        trainer = Trainer(small_topology_config, device='cuda', verbose=False)
        trainer.train(num_epochs=2)
        del trainer
        torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()

    # Should not accumulate memory across instances
    assert final_memory <= initial_memory * 2, \
        "Multiple trainer instances should not leak memory"


# ============================================================
# MIXED PRECISION TESTS (if supported)
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_autocast_compatibility(small_topology_config):
    """Test that training works with automatic mixed precision."""
    trainer = Trainer(small_topology_config, device='cuda', verbose=False)

    # Manually perform a training step with autocast
    from torch.cuda.amp import autocast

    coords = trainer.manifold.sample_coordinates(
        n_samples=trainer.config.training.batch_size,
        grid_n=50,
        device='cuda'
    )
    coords.requires_grad_(True)

    with autocast():
        phi_tensor = trainer.phi_network.get_phi_tensor(coords)

        # Should work without errors
        assert phi_tensor.dtype in [torch.float16, torch.float32]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradient_scaler_compatibility(small_topology_config):
    """Test that training is compatible with gradient scaling."""
    from torch.cuda.amp import autocast, GradScaler

    trainer = Trainer(small_topology_config, device='cuda', verbose=False)
    scaler = GradScaler()

    # Perform one training step with gradient scaling
    coords = trainer.manifold.sample_coordinates(
        n_samples=trainer.config.training.batch_size,
        grid_n=50,
        device='cuda'
    )
    coords.requires_grad_(True)

    with autocast():
        phi_tensor = trainer.phi_network.get_phi_tensor(coords)

        # Simple loss for testing
        loss = phi_tensor.pow(2).mean()

    # Scale and backward
    scaler.scale(loss).backward()
    scaler.step(trainer.optimizer)
    scaler.update()

    # Should complete without errors
    assert True


# ============================================================
# MULTI-GPU TESTS (basic)
# ============================================================

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs not available")
def test_network_on_specific_gpu(small_topology_config):
    """Test placing network on specific GPU."""
    network = g2.networks.create_phi_network_from_config(small_topology_config)

    # Move to GPU 1
    network.to('cuda:1')

    assert next(network.parameters()).device.index == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs not available")
def test_trainer_on_specific_gpu(small_topology_config):
    """Test creating trainer on specific GPU."""
    trainer = Trainer(small_topology_config, device='cuda:1', verbose=False)

    # Should be on GPU 1
    assert next(trainer.phi_network.parameters()).device.index == 1


# ============================================================
# DEVICE CONSISTENCY TESTS
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_pass_device_consistency(small_topology_config):
    """Test that forward pass maintains device consistency."""
    trainer = Trainer(small_topology_config, device='cuda', verbose=False)

    # Sample coordinates on CUDA
    coords = trainer.manifold.sample_coordinates(n_samples=10, device='cuda')

    # Forward pass
    phi_tensor = trainer.phi_network.get_phi_tensor(coords)

    # Output should be on same device
    assert phi_tensor.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_loss_computation_device_consistency(small_topology_config):
    """Test that loss computation maintains device consistency."""
    trainer = Trainer(small_topology_config, device='cuda', verbose=False)

    # Perform one training step
    metrics = trainer.train_step(epoch=0)

    # Loss should be computable (on CPU after .item())
    assert isinstance(metrics['loss'], float)


def test_cpu_forward_pass_device_consistency(small_topology_config):
    """Test that CPU forward pass maintains device consistency."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Sample coordinates on CPU
    coords = trainer.manifold.sample_coordinates(n_samples=10, device='cpu')

    # Forward pass
    phi_tensor = trainer.phi_network.get_phi_tensor(coords)

    # Output should be on CPU
    assert phi_tensor.device.type == 'cpu'


# ============================================================
# ERROR HANDLING FOR DEVICE MISMATCHES
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_mismatch_error(small_topology_config):
    """Test that device mismatch produces clear error."""
    trainer = Trainer(small_topology_config, device='cuda', verbose=False)

    # Sample coordinates on CPU (mismatch!)
    coords = trainer.manifold.sample_coordinates(n_samples=10, device='cpu')

    # Forward pass should fail with device mismatch
    with pytest.raises(RuntimeError):
        phi_tensor = trainer.phi_network.get_phi_tensor(coords)


def test_all_components_same_device(small_topology_config):
    """Test that all trainer components are on the same device."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Get devices
    phi_device = next(trainer.phi_network.parameters()).device
    h2_device = next(trainer.h2_network.parameters()).device
    h3_device = next(trainer.h3_network.parameters()).device
    loss_device = next(trainer.loss_fn.parameters()).device if hasattr(trainer.loss_fn, 'parameters') else phi_device

    # All should be the same
    assert phi_device == h2_device == h3_device
