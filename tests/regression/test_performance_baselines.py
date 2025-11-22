"""
Regression tests for performance baselines.

Tracks performance metrics over time to detect regressions in:
- Training speed
- Memory usage
- Parameter counts
- Forward/backward pass timing
"""

import pytest
import torch
import time
import sys
import psutil
import os
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer


# Mark all tests as regression tests
pytestmark = pytest.mark.regression


# ============================================================
# BASELINE CONSTANTS (update these after establishing baselines)
# ============================================================

# Training speed baselines (iterations/second on CPU)
BASELINE_TRAIN_STEP_TIME = 2.0  # seconds per epoch (small topology)
BASELINE_TRAIN_STEP_TOLERANCE = 0.5  # 50% tolerance

# Parameter count baselines
BASELINE_PHI_PARAMS_SMALL = 50000  # Approximate
BASELINE_H2_PARAMS_SMALL = 10000
BASELINE_H3_PARAMS_SMALL = 20000

# Memory baselines (MB)
BASELINE_MEMORY_SMALL_TOPOLOGY = 500  # MB


# ============================================================
# TRAINING SPEED TESTS
# ============================================================

def test_training_step_performance_baseline(small_topology_config):
    """Test that single training step completes within baseline time."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Warmup
    trainer.train_step(epoch=0)

    # Measure
    start_time = time.time()
    trainer.train_step(epoch=1)
    elapsed = time.time() - start_time

    # Should complete within reasonable time
    assert elapsed < BASELINE_TRAIN_STEP_TIME, \
        f"Training step took {elapsed:.2f}s, expected < {BASELINE_TRAIN_STEP_TIME}s"


def test_multi_epoch_training_speed(small_topology_config):
    """Test speed of multi-epoch training."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    num_epochs = 10

    start_time = time.time()
    trainer.train(num_epochs=num_epochs)
    elapsed = time.time() - start_time

    avg_time_per_epoch = elapsed / num_epochs

    # Should average less than baseline
    assert avg_time_per_epoch < BASELINE_TRAIN_STEP_TIME * 1.5, \
        f"Average {avg_time_per_epoch:.2f}s/epoch, expected < {BASELINE_TRAIN_STEP_TIME * 1.5}s"


def test_forward_pass_timing(small_topology_config):
    """Test that forward pass completes quickly."""
    phi_network = g2.networks.create_phi_network_from_config(small_topology_config)

    coords = torch.randn(100, 7, requires_grad=True)

    # Warmup
    _ = phi_network.get_phi_tensor(coords)

    # Measure
    start_time = time.time()
    for _ in range(10):
        phi_tensor = phi_network.get_phi_tensor(coords)
    elapsed = time.time() - start_time

    avg_time = elapsed / 10

    # Should be fast (< 0.1s per forward pass on CPU)
    assert avg_time < 0.1, f"Forward pass took {avg_time:.4f}s, expected < 0.1s"


def test_backward_pass_timing(small_topology_config):
    """Test that backward pass completes within reasonable time."""
    phi_network = g2.networks.create_phi_network_from_config(small_topology_config)

    coords = torch.randn(100, 7, requires_grad=True)

    # Warmup
    phi_tensor = phi_network.get_phi_tensor(coords)
    loss = phi_tensor.pow(2).mean()
    loss.backward()

    # Clear gradients
    phi_network.zero_grad()

    # Measure
    start_time = time.time()
    phi_tensor = phi_network.get_phi_tensor(coords)
    loss = phi_tensor.pow(2).mean()
    loss.backward()
    elapsed = time.time() - start_time

    # Should be reasonably fast (< 0.2s on CPU)
    assert elapsed < 0.2, f"Backward pass took {elapsed:.4f}s, expected < 0.2s"


# ============================================================
# PARAMETER COUNT REGRESSION TESTS
# ============================================================

def test_phi_network_parameter_count_regression(small_topology_config):
    """Test that PhiNetwork parameter count hasn't grown unexpectedly."""
    network = g2.networks.create_phi_network_from_config(small_topology_config)

    param_count = network.count_parameters()

    # Parameter count should be in expected range
    # (Allow 20% variance for reasonable changes)
    assert param_count < BASELINE_PHI_PARAMS_SMALL * 1.2, \
        f"PhiNetwork has {param_count} params, expected < {BASELINE_PHI_PARAMS_SMALL * 1.2}"


def test_h2_network_parameter_count_regression(small_topology_config):
    """Test that H2Network parameter count scales correctly."""
    h2_network, _ = g2.networks.create_harmonic_networks_from_config(small_topology_config)

    param_count = h2_network.count_parameters()

    # Should be in expected range
    assert param_count < BASELINE_H2_PARAMS_SMALL * 1.2, \
        f"H2Network has {param_count} params, expected < {BASELINE_H2_PARAMS_SMALL * 1.2}"


def test_h3_network_parameter_count_regression(small_topology_config):
    """Test that H3Network parameter count scales correctly."""
    _, h3_network = g2.networks.create_harmonic_networks_from_config(small_topology_config)

    param_count = h3_network.count_parameters()

    # Should be in expected range
    assert param_count < BASELINE_H3_PARAMS_SMALL * 1.2, \
        f"H3Network has {param_count} params, expected < {BASELINE_H3_PARAMS_SMALL * 1.2}"


def test_parameter_count_scales_with_topology():
    """Test that parameter count scales reasonably with topology size."""
    # Small topology
    config_small = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    h2_small, h3_small = g2.networks.create_harmonic_networks_from_config(config_small)

    # Large topology
    config_large = g2.create_k7_config(b2_m1=15, b3_m1=50, b2_m2=15, b3_m2=50)
    h2_large, h3_large = g2.networks.create_harmonic_networks_from_config(config_large)

    # Large should have more parameters
    assert h2_large.count_parameters() > h2_small.count_parameters()
    assert h3_large.count_parameters() > h3_small.count_parameters()

    # But scaling should be reasonable (linear in b2/b3, not exponential)
    b2_ratio = config_large.manifold.topology.b2 / config_small.manifold.topology.b2
    b3_ratio = config_large.manifold.topology.b3 / config_small.manifold.topology.b3

    param_ratio_h2 = h2_large.count_parameters() / h2_small.count_parameters()
    param_ratio_h3 = h3_large.count_parameters() / h3_small.count_parameters()

    # Should scale roughly linearly (allow 2x factor for overhead)
    assert param_ratio_h2 < b2_ratio * 2
    assert param_ratio_h3 < b3_ratio * 2


# ============================================================
# MEMORY USAGE REGRESSION TESTS
# ============================================================

def test_memory_usage_baseline(small_topology_config):
    """Test that memory usage stays within baseline."""
    process = psutil.Process(os.getpid())

    # Measure baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Create trainer and train
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)
    trainer.train(num_epochs=5)

    # Measure after training
    after_memory = process.memory_info().rss / 1024 / 1024  # MB

    memory_increase = after_memory - baseline_memory

    # Memory increase should be reasonable
    assert memory_increase < BASELINE_MEMORY_SMALL_TOPOLOGY, \
        f"Memory increased by {memory_increase:.1f}MB, expected < {BASELINE_MEMORY_SMALL_TOPOLOGY}MB"


def test_memory_does_not_grow_during_training(small_topology_config):
    """Test that memory doesn't grow unboundedly during training."""
    import gc

    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    process = psutil.Process(os.getpid())

    # Train a bit to establish steady state
    trainer.train(num_epochs=5)

    gc.collect()
    memory_after_5 = process.memory_info().rss / 1024 / 1024

    # Continue training
    trainer.train(num_epochs=15)

    gc.collect()
    memory_after_15 = process.memory_info().rss / 1024 / 1024

    memory_growth = memory_after_15 - memory_after_5

    # Memory should not grow significantly during continued training
    # Allow 100MB growth for metrics history, etc.
    assert memory_growth < 100, \
        f"Memory grew by {memory_growth:.1f}MB during training (potential leak)"


# ============================================================
# THROUGHPUT TESTS
# ============================================================

def test_samples_per_second_throughput(small_topology_config):
    """Test throughput in samples processed per second."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    batch_size = trainer.config.training.batch_size

    # Warmup
    trainer.train_step(epoch=0)

    # Measure throughput
    num_steps = 10
    start_time = time.time()

    for epoch in range(num_steps):
        trainer.train_step(epoch)

    elapsed = time.time() - start_time

    samples_per_second = (num_steps * batch_size) / elapsed

    # Should process at least 100 samples/second on CPU
    assert samples_per_second > 100, \
        f"Throughput: {samples_per_second:.1f} samples/s, expected > 100"


# ============================================================
# OPERATOR PERFORMANCE TESTS
# ============================================================

def test_levi_civita_construction_speed():
    """Test that Levi-Civita tensor construction is fast."""
    start_time = time.time()

    for _ in range(10):
        eps_indices, eps_signs = g2.build_levi_civita_sparse_7d()

    elapsed = time.time() - start_time
    avg_time = elapsed / 10

    # Should be very fast (< 0.01s per construction)
    assert avg_time < 0.01, f"Levi-Civita construction took {avg_time:.4f}s"


def test_hodge_star_performance(levi_civita):
    """Test Hodge star operator performance."""
    batch_size = 100
    phi = torch.randn(batch_size, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    eps_indices, eps_signs = levi_civita

    # Warmup
    _ = g2.hodge_star_3(phi, metric, eps_indices, eps_signs)

    # Measure
    start_time = time.time()
    for _ in range(10):
        star_phi = g2.hodge_star_3(phi, metric, eps_indices, eps_signs)
    elapsed = time.time() - start_time

    avg_time = elapsed / 10

    # Should be fast (< 0.1s on CPU for batch of 100)
    assert avg_time < 0.1, f"Hodge star took {avg_time:.4f}s"


def test_exterior_derivative_performance():
    """Test exterior derivative computation performance."""
    batch_size = 50
    phi = torch.randn(batch_size, 7, 7, 7, requires_grad=True)

    # Warmup
    _ = g2.compute_exterior_derivative(phi, phi)

    # Measure
    start_time = time.time()
    dphi = g2.compute_exterior_derivative(phi, phi)
    elapsed = time.time() - start_time

    # Should complete quickly
    assert elapsed < 0.5, f"Exterior derivative took {elapsed:.4f}s"


# ============================================================
# COMPARATIVE PERFORMANCE TESTS
# ============================================================

def test_small_vs_medium_topology_performance():
    """Compare performance between small and medium topologies."""
    # Small topology
    config_small = g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)
    trainer_small = Trainer(config_small, device='cpu', verbose=False)

    start_small = time.time()
    trainer_small.train(num_epochs=3)
    time_small = time.time() - start_small

    # Medium topology
    config_medium = g2.create_k7_config(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)
    trainer_medium = Trainer(config_medium, device='cpu', verbose=False)

    start_medium = time.time()
    trainer_medium.train(num_epochs=3)
    time_medium = time.time() - start_medium

    # Medium should be slower, but not dramatically so
    # (within 5x given roughly 4x more harmonics)
    assert time_medium < time_small * 5, \
        f"Medium topology took {time_medium/time_small:.1f}x longer than small"


# ============================================================
# PERFORMANCE TRACKING UTILITIES
# ============================================================

def test_record_performance_metrics(small_topology_config):
    """Record current performance metrics for future comparison."""
    trainer = Trainer(small_topology_config, device='cpu', verbose=False)

    # Measure various metrics
    start_time = time.time()
    trainer.train(num_epochs=5)
    training_time = time.time() - start_time

    param_counts = {
        'phi': trainer.phi_network.count_parameters(),
        'h2': trainer.h2_network.count_parameters(),
        'h3': trainer.h3_network.count_parameters(),
    }

    metrics = {
        'training_time_5_epochs': training_time,
        'avg_time_per_epoch': training_time / 5,
        'parameter_counts': param_counts,
        'total_parameters': sum(param_counts.values()),
    }

    # Print for recording (can be captured in CI)
    print("\n=== Performance Metrics ===")
    print(f"Training time (5 epochs): {metrics['training_time_5_epochs']:.2f}s")
    print(f"Avg time per epoch: {metrics['avg_time_per_epoch']:.2f}s")
    print(f"Total parameters: {metrics['total_parameters']:,}")
    print(f"  - PhiNetwork: {param_counts['phi']:,}")
    print(f"  - H2Network: {param_counts['h2']:,}")
    print(f"  - H3Network: {param_counts['h3']:,}")
    print("==========================\n")

    # This test always passes - it's just for recording
    assert True
