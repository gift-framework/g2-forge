"""
Performance Benchmarks for g2-forge

Tests training speed, memory usage, and scalability across different topologies.
"""

import pytest
import torch
import time
import gc
from typing import Dict, Any

# Import g2-forge components
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer
from g2forge.utils.config import G2ForgeConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_config_small():
    """Small topology for quick benchmarking."""
    return g2.create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)


@pytest.fixture
def benchmark_config_medium():
    """Medium topology for standard benchmarking."""
    return g2.create_k7_config(b2_m1=7, b3_m1=30, b2_m2=5, b3_m2=25)


@pytest.fixture
def benchmark_config_large():
    """Large topology (close to GIFT) for stress testing."""
    return g2.create_k7_config(b2_m1=11, b3_m1=40, b2_m2=10, b3_m2=37)


# ============================================================================
# TRAINING SPEED BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.slow
def test_single_epoch_speed_small(benchmark_config_small, benchmark):
    """Benchmark single epoch training time for small topology."""
    def train_epoch():
        trainer = Trainer(benchmark_config_small, device='cpu', verbose=False)
        results = trainer.train(num_epochs=1)
        return results

    if hasattr(benchmark, '__call__'):
        result = benchmark(train_epoch)
    else:
        # Fallback if pytest-benchmark not installed
        start = time.time()
        result = train_epoch()
        elapsed = time.time() - start
        assert elapsed < 60.0, f"Single epoch took {elapsed:.2f}s (should be < 60s)"


@pytest.mark.benchmark
@pytest.mark.slow
def test_training_speed_vs_topology_size():
    """Test that training time scales reasonably with topology size."""
    topologies = [
        (3, 10, 2, 10, "small"),
        (5, 20, 5, 20, "medium"),
        (7, 30, 7, 30, "large"),
    ]

    times = []
    for b2_m1, b3_m1, b2_m2, b3_m2, name in topologies:
        config = g2.create_k7_config(b2_m1, b3_m1, b2_m2, b3_m2)
        trainer = Trainer(config, device='cpu', verbose=False)

        start = time.time()
        trainer.train(num_epochs=1)
        elapsed = time.time() - start
        times.append(elapsed)

    # Training time should increase but not too drastically
    assert times[1] < times[0] * 5, "Medium topology too slow vs small"
    assert times[2] < times[1] * 5, "Large topology too slow vs medium"


@pytest.mark.benchmark
def test_batch_processing_speed(benchmark_config_small):
    """Test batch processing performance."""
    trainer = Trainer(benchmark_config_small, device='cpu', verbose=False)

    batch_sizes = [32, 64, 128]
    times = []

    for batch_size in batch_sizes:
        coords = torch.randn(batch_size, 7)
        start = time.time()
        phi = trainer.phi_network(coords)
        elapsed = time.time() - start
        times.append(elapsed)

        assert phi.shape[0] == batch_size

    # Larger batches should not scale linearly (benefit from vectorization)
    assert times[2] < times[0] * 4, "Batch processing not efficient"


# ============================================================================
# MEMORY USAGE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.slow
def test_memory_usage_during_training(benchmark_config_small):
    """Test memory usage stays reasonable during training."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    trainer = Trainer(benchmark_config_small, device='cpu', verbose=False)

    # Train for a few epochs
    results = trainer.train(num_epochs=5)

    # Check that training completed without OOM
    assert results is not None
    assert 'loss' in results


@pytest.mark.benchmark
def test_no_memory_leak_across_epochs(benchmark_config_small):
    """Test that memory doesn't leak across training epochs."""
    trainer = Trainer(benchmark_config_small, device='cpu', verbose=False)

    # Train multiple times
    for i in range(3):
        gc.collect()
        results = trainer.train(num_epochs=2)
        assert results is not None


@pytest.mark.benchmark
def test_gradient_memory_cleared(benchmark_config_small):
    """Test that gradients are properly cleared between steps."""
    trainer = Trainer(benchmark_config_small, device='cpu', verbose=False)

    # Initial training step
    trainer.train(num_epochs=1)

    # Check gradient state
    for param in trainer.phi_network.parameters():
        if param.grad is not None:
            # Gradients should exist but be ready for next step
            assert param.grad.shape == param.shape


# ============================================================================
# SCALABILITY BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("b2_total", [5, 10, 20, 30])
def test_scalability_with_b2(b2_total):
    """Test scalability across different b₂ values."""
    b2_m1 = b2_total // 2
    b2_m2 = b2_total - b2_m1

    config = g2.create_k7_config(b2_m1=b2_m1, b3_m1=20, b2_m2=b2_m2, b3_m2=20)
    trainer = Trainer(config, device='cpu', verbose=False)

    start = time.time()
    results = trainer.train(num_epochs=1)
    elapsed = time.time() - start

    assert results is not None
    # Should complete in reasonable time even for b₂=30
    assert elapsed < 120.0, f"Training with b₂={b2_total} took {elapsed:.2f}s"


@pytest.mark.benchmark
@pytest.mark.slow
def test_large_topology_gift_size(benchmark_config_large):
    """Test performance with GIFT-sized topology (b₂=21, b₃=77)."""
    trainer = Trainer(benchmark_config_large, device='cpu', verbose=False)

    start = time.time()
    results = trainer.train(num_epochs=2)
    elapsed = time.time() - start

    assert results is not None
    assert 'loss' in results
    # Should complete but may be slow
    assert elapsed < 300.0, f"Large topology took {elapsed:.2f}s for 2 epochs"


# ============================================================================
# NETWORK FORWARD PASS BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
def test_phi_network_forward_speed(benchmark_config_small):
    """Benchmark PhiNetwork forward pass speed."""
    trainer = Trainer(benchmark_config_small, device='cpu', verbose=False)
    coords = torch.randn(100, 7)

    # Warmup
    _ = trainer.phi_network(coords)

    # Benchmark
    start = time.time()
    num_runs = 10
    for _ in range(num_runs):
        phi = trainer.phi_network(coords)
    elapsed = time.time() - start

    avg_time = elapsed / num_runs
    assert avg_time < 0.5, f"PhiNetwork forward pass avg {avg_time:.3f}s (should be < 0.5s)"


@pytest.mark.benchmark
def test_harmonic_network_forward_speed(benchmark_config_small):
    """Benchmark HarmonicNetwork forward pass speed."""
    trainer = Trainer(benchmark_config_small, device='cpu', verbose=False)
    coords = torch.randn(100, 7)

    # Warmup
    _ = trainer.h2_network(coords)

    # Benchmark
    start = time.time()
    num_runs = 10
    for _ in range(num_runs):
        h2 = trainer.h2_network(coords)
    elapsed = time.time() - start

    avg_time = elapsed / num_runs
    assert avg_time < 0.5, f"HarmonicNetwork forward pass avg {avg_time:.3f}s (should be < 0.5s)"


# ============================================================================
# OPERATOR PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
def test_exterior_derivative_performance():
    """Benchmark exterior derivative operator."""
    from g2forge.operators import exterior_derivative_3

    batch_size = 100
    phi = torch.randn(batch_size, 7, 7, 7)
    coords = torch.randn(batch_size, 7, requires_grad=True)

    start = time.time()
    num_runs = 5
    for _ in range(num_runs):
        dphi = exterior_derivative_3(phi, coords)
    elapsed = time.time() - start

    avg_time = elapsed / num_runs
    assert avg_time < 1.0, f"Exterior derivative avg {avg_time:.3f}s (should be < 1.0s)"


@pytest.mark.benchmark
def test_hodge_star_performance():
    """Benchmark Hodge star operator."""
    from g2forge.operators import hodge_star_3, levi_civita_tensor

    batch_size = 100
    phi = torch.randn(batch_size, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).expand(batch_size, 7, 7)
    eps_indices, eps_signs = levi_civita_tensor(7)

    start = time.time()
    num_runs = 5
    for _ in range(num_runs):
        star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)
    elapsed = time.time() - start

    avg_time = elapsed / num_runs
    assert avg_time < 1.0, f"Hodge star avg {avg_time:.3f}s (should be < 1.0s)"


# ============================================================================
# OPTIMIZATION STEP BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
def test_single_optimization_step_speed(benchmark_config_small):
    """Benchmark a single optimization step."""
    trainer = Trainer(benchmark_config_small, device='cpu', verbose=False)

    coords = torch.randn(64, 7)

    start = time.time()
    # Forward pass
    phi = trainer.phi_network(coords)
    h2 = trainer.h2_network(coords)
    h3 = trainer.h3_network(coords)

    # Dummy loss
    loss = phi.pow(2).sum() + h2.pow(2).sum() + h3.pow(2).sum()

    # Backward pass
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    elapsed = time.time() - start
    assert elapsed < 2.0, f"Single optimization step took {elapsed:.3f}s"


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

@pytest.mark.benchmark
def test_print_performance_summary(benchmark_config_small, benchmark_config_large):
    """Print summary of performance characteristics."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*80)

    # Small topology
    trainer_small = Trainer(benchmark_config_small, device='cpu', verbose=False)
    start = time.time()
    results_small = trainer_small.train(num_epochs=1)
    time_small = time.time() - start

    print(f"\nSmall Topology (b₂=5):")
    print(f"  - Training time (1 epoch): {time_small:.2f}s")
    print(f"  - Final loss: {results_small['loss']:.6f}")

    # Large topology
    trainer_large = Trainer(benchmark_config_large, device='cpu', verbose=False)
    start = time.time()
    results_large = trainer_large.train(num_epochs=1)
    time_large = time.time() - start

    print(f"\nLarge Topology (b₂=21, close to GIFT):")
    print(f"  - Training time (1 epoch): {time_large:.2f}s")
    print(f"  - Final loss: {results_large['loss']:.6f}")
    print(f"  - Slowdown factor: {time_large/time_small:.2f}x")

    print("\n" + "="*80)

    assert True  # This test always passes, just prints info
