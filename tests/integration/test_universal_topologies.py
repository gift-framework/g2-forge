"""
Integration tests for universal topology support.

Tests that the framework works correctly with various topologies
in end-to-end training scenarios, validating true universality.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.training.trainer import Trainer
from g2forge.utils.config import G2ForgeConfig


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# ============================================================
# MINIMAL TOPOLOGY TESTS
# ============================================================

def test_training_with_minimal_topology():
    """
    Test full training with minimal topology b₂=1, b₃=1.

    This tests the lower bound of the universal framework.
    """
    # Create config with minimal topology
    config = G2ForgeConfig.create_gift_preset()

    # Override with minimal topology
    from g2forge.utils.config import (
        TopologyConfig,
        TCSParameters,
        ManifoldConfig
    )

    topology = TopologyConfig(b2=1, b3=1)
    tcs_params = TCSParameters(b2_m1=1, b3_m1=1, b2_m2=0, b3_m2=0)

    config.manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params,
        dimension=7
    )

    # Train for a few epochs
    trainer = Trainer(config, device='cpu', verbose=False)
    results = trainer.train(num_epochs=3)

    # Should complete without errors
    assert 'final_metrics' in results
    assert np.isfinite(results['final_metrics']['loss'])

    # Networks should be correctly sized
    assert trainer.h2_network.n_forms == 1
    assert trainer.h3_network.n_forms == 1


def test_training_with_large_topology():
    """
    Test full training with large topology b₂=50, b₃=150.

    This tests scalability of the universal framework.
    """
    config = G2ForgeConfig.create_gift_preset()

    from g2forge.utils.config import (
        TopologyConfig,
        TCSParameters,
        ManifoldConfig
    )

    topology = TopologyConfig(b2=50, b3=150)
    tcs_params = TCSParameters(b2_m1=25, b3_m1=75, b2_m2=25, b3_m2=75)

    config.manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params,
        dimension=7
    )

    # Reduce training scale for speed
    config.training.batch_size = 512
    config.training.num_curriculum_phases = 1

    trainer = Trainer(config, device='cpu', verbose=False)
    results = trainer.train(num_epochs=2)

    # Should complete without errors
    assert 'final_metrics' in results
    assert np.isfinite(results['final_metrics']['loss'])

    # Networks should be correctly sized
    assert trainer.h2_network.n_forms == 50
    assert trainer.h3_network.n_forms == 150


def test_training_with_asymmetric_tcs():
    """
    Test TCS with highly asymmetric components.

    M₁ component much larger than M₂.
    """
    config = G2ForgeConfig.create_gift_preset()

    from g2forge.utils.config import (
        TopologyConfig,
        TCSParameters,
        ManifoldConfig
    )

    # Asymmetric: M₁ has most of the topology
    topology = TopologyConfig(b2=30, b3=90)
    tcs_params = TCSParameters(
        b2_m1=25, b3_m1=80,  # Large M₁
        b2_m2=5, b3_m2=10    # Small M₂
    )

    config.manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params,
        dimension=7
    )

    config.training.batch_size = 512

    trainer = Trainer(config, device='cpu', verbose=False)
    results = trainer.train(num_epochs=3)

    # Should handle asymmetry correctly
    assert 'final_metrics' in results
    assert np.isfinite(results['final_metrics']['loss'])


# ============================================================
# TOPOLOGY VARIATION TESTS
# ============================================================

def test_multiple_topologies_consistent_behavior():
    """
    Test that different topologies show consistent training behavior.

    All should train without errors and produce finite losses.
    """
    topologies = [
        (2, 10),    # Very small
        (5, 20),    # Small
        (10, 40),   # Medium
        (21, 77),   # GIFT
    ]

    for b2, b3 in topologies:
        # Create config
        config = G2ForgeConfig.create_gift_preset()

        from g2forge.utils.config import (
            TopologyConfig,
            TCSParameters,
            ManifoldConfig
        )

        topology = TopologyConfig(b2=b2, b3=b3)
        tcs_params = TCSParameters(
            b2_m1=b2//2, b3_m1=b3//2,
            b2_m2=b2 - b2//2, b3_m2=b3 - b3//2
        )

        config.manifold = ManifoldConfig(
            type="K7",
            construction="TCS",
            topology=topology,
            tcs_params=tcs_params,
            dimension=7
        )

        config.training.batch_size = 256

        # Train
        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=2)

        # All should work
        assert 'final_metrics' in results, f"Failed for topology ({b2}, {b3})"
        assert np.isfinite(results['final_metrics']['loss']), \
            f"Non-finite loss for topology ({b2}, {b3})"


def test_zero_b2_component_topology():
    """
    Test topology with zero b₂ in one TCS component.

    Tests edge case: M₂ has no harmonic 2-forms.
    """
    config = G2ForgeConfig.create_gift_preset()

    from g2forge.utils.config import (
        TopologyConfig,
        TCSParameters,
        ManifoldConfig
    )

    topology = TopologyConfig(b2=10, b3=40)
    tcs_params = TCSParameters(
        b2_m1=10, b3_m1=20,  # All b₂ in M₁
        b2_m2=0, b3_m2=20    # No b₂ in M₂
    )

    config.manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params,
        dimension=7
    )

    trainer = Trainer(config, device='cpu', verbose=False)
    results = trainer.train(num_epochs=3)

    assert 'final_metrics' in results
    assert np.isfinite(results['final_metrics']['loss'])


# ============================================================
# GRADIENT FLOW ACROSS TOPOLOGIES
# ============================================================

def test_gradient_flow_small_topology():
    """Test gradient flow for small topology."""
    config = G2ForgeConfig.create_gift_preset()

    from g2forge.utils.config import (
        TopologyConfig,
        TCSParameters,
        ManifoldConfig
    )

    topology = TopologyConfig(b2=3, b3=10)
    tcs_params = TCSParameters(b2_m1=2, b3_m1=5, b2_m2=1, b3_m2=5)

    config.manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params,
        dimension=7
    )

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train one step
    metrics = trainer.train_step(epoch=0)

    # Check gradients exist and are finite
    for network in [trainer.phi_network, trainer.h2_network, trainer.h3_network]:
        for param in network.parameters():
            assert param.grad is not None, "Gradient should exist"
            assert torch.isfinite(param.grad).all(), "Gradients should be finite"


def test_gradient_flow_large_topology():
    """Test gradient flow for large topology."""
    config = G2ForgeConfig.create_gift_preset()

    from g2forge.utils.config import (
        TopologyConfig,
        TCSParameters,
        ManifoldConfig
    )

    topology = TopologyConfig(b2=30, b3=100)
    tcs_params = TCSParameters(b2_m1=15, b3_m1=50, b2_m2=15, b3_m2=50)

    config.manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params,
        dimension=7
    )

    config.training.batch_size = 512

    trainer = Trainer(config, device='cpu', verbose=False)

    # Train one step
    metrics = trainer.train_step(epoch=0)

    # Check gradients
    for network in [trainer.phi_network, trainer.h2_network, trainer.h3_network]:
        for param in network.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
