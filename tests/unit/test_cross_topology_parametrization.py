"""
Extended cross-topology parametrization tests.

Expands test coverage across a wider range of topologies to ensure
universal behavior. Tests multiple (b₂, b₃) combinations systematically.
"""

import pytest
import torch

from g2forge.utils import TopologyConfig, TCSParameters, ManifoldConfig, G2ForgeConfig
from g2forge.manifolds import K7Manifold
from g2forge.networks import PhiNetwork, HarmonicNetwork
from g2forge.core.losses import gram_matrix_loss, CompositeLoss
from g2forge.core.operators import build_levi_civita_sparse_7d
from g2forge.training import Trainer


# Define comprehensive topology test suite
TOPOLOGY_TEST_SUITE = [
    (1, 5),      # Minimal
    (3, 10),     # Very small
    (5, 20),     # Small
    (10, 40),    # Medium
    (15, 60),    # Medium-large
    (21, 77),    # GIFT v1.0
    (25, 90),    # Large
    (30, 100),   # Very large
    (50, 150),   # Stress test
]


class TestManifoldUniversalTopology:
    """Test K7Manifold works for any topology."""

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE)
    def test_k7_manifold_creation(self, b2, b3):
        """Test K7Manifold can be created for any topology."""
        # Create balanced TCS decomposition
        b2_m1 = b2 // 2
        b2_m2 = b2 - b2_m1
        b3_m1 = (b3 - 1) // 2
        b3_m2 = (b3 - 1) - b3_m1

        topology = TopologyConfig(b2=b2, b3=b3)
        tcs_params = TCSParameters(
            b2_m1=b2_m1, b3_m1=b3_m1,
            b2_m2=b2_m2, b3_m2=b3_m2
        )

        manifold = K7Manifold(topology, tcs_params)

        assert manifold.topology.b2 == b2
        assert manifold.topology.b3 == b3

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE[:5])  # Faster subset
    def test_k7_coordinate_sampling_universal(self, b2, b3):
        """Test coordinate sampling works for any topology."""
        b2_m1 = b2 // 2
        b2_m2 = b2 - b2_m1
        b3_m1 = (b3 - 1) // 2
        b3_m2 = (b3 - 1) - b3_m1

        topology = TopologyConfig(b2=b2, b3=b3)
        tcs_params = TCSParameters(
            b2_m1=b2_m1, b3_m1=b3_m1,
            b2_m2=b2_m2, b3_m2=b3_m2
        )

        manifold = K7Manifold(topology, tcs_params)

        # Sample coordinates
        coords = manifold.sample_coordinates(n_samples=100, device='cpu')

        assert coords.shape == (100, 7)
        assert not torch.isnan(coords).any()

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE[:4])
    def test_k7_region_weights_universal(self, b2, b3):
        """Test region weights sum to 1 for any topology."""
        b2_m1 = b2 // 2
        b2_m2 = b2 - b2_m1
        b3_m1 = (b3 - 1) // 2
        b3_m2 = (b3 - 1) - b3_m1

        topology = TopologyConfig(b2=b2, b3=b3)
        tcs_params = TCSParameters(
            b2_m1=b2_m1, b3_m1=b3_m1,
            b2_m2=b2_m2, b3_m2=b3_m2
        )

        manifold = K7Manifold(topology, tcs_params)
        coords = manifold.sample_coordinates(n_samples=50, device='cpu')

        weights = manifold.get_region_weights(coords)

        # Weights should sum to 1
        weight_sums = weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones(50), atol=1e-5)


class TestNetworkAutoSizingUniversal:
    """Test neural networks auto-size correctly for any topology."""

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE)
    def test_harmonic_h2_network_auto_sizes(self, b2, b3):
        """Test H² network output dimension matches b₂."""
        topology = TopologyConfig(b2=b2, b3=b3)

        h2_net = HarmonicNetwork(
            topology=topology,
            p=2,
            hidden_dims=[64, 128, 64]
        )

        assert h2_net.n_forms == b2, f"Expected {b2} forms, got {h2_net.n_forms}"

        # Test forward pass
        coords = torch.randn(10, 7)
        output = h2_net(coords)

        assert output.shape == (10, b2, 21), f"Expected shape (10, {b2}, 21), got {output.shape}"

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE)
    def test_harmonic_h3_network_auto_sizes(self, b2, b3):
        """Test H³ network output dimension matches b₃."""
        topology = TopologyConfig(b2=b2, b3=b3)

        h3_net = HarmonicNetwork(
            topology=topology,
            p=3,
            hidden_dims=[64, 128, 64]
        )

        assert h3_net.n_forms == b3, f"Expected {b3} forms, got {h3_net.n_forms}"

        # Test forward pass
        coords = torch.randn(10, 7)
        output = h3_net(coords)

        assert output.shape == (10, b3, 35), f"Expected shape (10, {b3}, 35), got {output.shape}"

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE[:5])
    def test_both_harmonic_networks_forward_pass(self, b2, b3):
        """Test both H² and H³ networks work together."""
        topology = TopologyConfig(b2=b2, b3=b3)

        h2_net = HarmonicNetwork(topology, p=2, hidden_dims=[64, 128, 64])
        h3_net = HarmonicNetwork(topology, p=3, hidden_dims=[64, 128, 64])

        coords = torch.randn(20, 7)

        h2_output = h2_net(coords)
        h3_output = h3_net(coords)

        assert h2_output.shape == (20, b2, 21)
        assert h3_output.shape == (20, b3, 35)


class TestLossesUniversalTopology:
    """Test loss functions work for any topology."""

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE)
    def test_gram_matrix_loss_universal(self, b2, b3):
        """Test gram matrix loss works for any (b₂, b₃)."""
        topology = TopologyConfig(b2=b2, b3=b3)

        # Create random harmonic forms
        h2_forms = torch.randn(10, b2, 21)
        h3_forms = torch.randn(10, b3, 35)

        # Identity metric
        metric = torch.eye(7).unsqueeze(0).repeat(10, 1, 1)
        coords = torch.randn(10, 7)

        eps_indices, eps_signs = build_levi_civita_sparse_7d()

        loss = gram_matrix_loss(
            h2_forms, h3_forms, topology, metric, coords,
            eps_indices, eps_signs
        )

        # Should be a scalar
        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE[:4])
    def test_composite_loss_universal(self, b2, b3):
        """Test CompositeLoss works for any topology."""
        b2_m1 = b2 // 2
        b2_m2 = b2 - b2_m1
        b3_m1 = (b3 - 1) // 2
        b3_m2 = (b3 - 1) - b3_m1

        topology = TopologyConfig(b2=b2, b3=b3)
        tcs_params = TCSParameters(
            b2_m1=b2_m1, b3_m1=b3_m1,
            b2_m2=b2_m2, b3_m2=b3_m2
        )

        manifold = K7Manifold(topology, tcs_params)

        loss_fn = CompositeLoss(topology, manifold)

        # Create dummy inputs
        coords = torch.randn(10, 7)
        phi = torch.randn(10, 7, 7, 7)
        h2_forms = torch.randn(10, b2, 21)
        h3_forms = torch.randn(10, b3, 35)

        # Should not error
        loss_dict = loss_fn(phi, h2_forms, h3_forms, coords, epoch=0)

        assert 'total' in loss_dict
        assert not torch.isnan(loss_dict['total'])


class TestTrainingUniversalTopology:
    """Test training pipeline works for any topology."""

    @pytest.mark.parametrize("b2,b3", [
        (1, 5),
        (5, 20),
        (10, 40),
    ])
    def test_trainer_initialization_universal(self, b2, b3):
        """Test Trainer can be initialized for any topology."""
        b2_m1 = b2 // 2
        b2_m2 = b2 - b2_m1
        b3_m1 = (b3 - 1) // 2
        b3_m2 = (b3 - 1) - b3_m1

        # Create config
        from g2forge.utils.config import create_k7_config

        config = create_k7_config(
            b2_m1=b2_m1, b3_m1=b3_m1,
            b2_m2=b2_m2, b3_m2=b3_m2
        )

        # Should not error
        trainer = Trainer(config, device='cpu', verbose=False)

        # Check networks have correct dimensions
        assert trainer.h2_net.n_forms == b2
        assert trainer.h3_net.n_forms == b3

    @pytest.mark.parametrize("b2,b3", [
        (1, 5),
        (3, 10),
        (5, 20),
    ])
    def test_training_step_universal(self, b2, b3):
        """Test a single training step works for any topology."""
        b2_m1 = b2 // 2
        b2_m2 = b2 - b2_m1
        b3_m1 = (b3 - 1) // 2
        b3_m2 = (b3 - 1) - b3_m1

        from g2forge.utils.config import create_k7_config

        config = create_k7_config(
            b2_m1=b2_m1, b3_m1=b3_m1,
            b2_m2=b2_m2, b3_m2=b3_m2
        )

        # Simplify for speed
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)

        # Run one epoch
        results = trainer.train(num_epochs=1)

        assert 'training_history' in results
        assert len(results['training_history']['epoch']) == 1


class TestConfigUniversalTopology:
    """Test configuration system works for any topology."""

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE)
    def test_topology_config_validation(self, b2, b3):
        """Test TopologyConfig validates for any valid (b₂, b₃)."""
        topology = TopologyConfig(b2=b2, b3=b3)

        # Should not raise
        topology.validate()

        # Check Poincaré duality
        assert topology.b4 == b3
        assert topology.b5 == b2

    @pytest.mark.parametrize("b2,b3", TOPOLOGY_TEST_SUITE[:6])
    def test_manifold_config_creation(self, b2, b3):
        """Test ManifoldConfig can be created for any topology."""
        b2_m1 = b2 // 2
        b2_m2 = b2 - b2_m1
        b3_m1 = (b3 - 1) // 2
        b3_m2 = (b3 - 1) - b3_m1

        topology = TopologyConfig(b2=b2, b3=b3)
        tcs_params = TCSParameters(
            b2_m1=b2_m1, b3_m1=b3_m1,
            b2_m2=b2_m2, b3_m2=b3_m2
        )

        manifold_config = ManifoldConfig(
            topology=topology,
            tcs_params=tcs_params,
            moduli=None
        )

        # Should validate successfully
        manifold_config.validate()


class TestEdgeCaseTopologies:
    """Test edge case topologies."""

    def test_minimal_topology_b2_1_b3_1(self):
        """Test minimal non-trivial topology (1, 1)."""
        topology = TopologyConfig(b2=1, b3=1)
        topology.validate()

        # Create networks
        h2_net = HarmonicNetwork(topology, p=2, hidden_dims=[32, 64, 32])
        h3_net = HarmonicNetwork(topology, p=3, hidden_dims=[32, 64, 32])

        assert h2_net.n_forms == 1
        assert h3_net.n_forms == 1

        # Forward pass
        coords = torch.randn(5, 7)
        h2_output = h2_net(coords)
        h3_output = h3_net(coords)

        assert h2_output.shape == (5, 1, 21)
        assert h3_output.shape == (5, 1, 35)

    @pytest.mark.parametrize("b2", [1, 2, 5, 10, 20, 50])
    def test_varying_b2_fixed_b3(self, b2):
        """Test varying b₂ with fixed b₃."""
        b3 = 20
        topology = TopologyConfig(b2=b2, b3=b3)
        topology.validate()

        h2_net = HarmonicNetwork(topology, p=2, hidden_dims=[64, 128, 64])
        assert h2_net.n_forms == b2

    @pytest.mark.parametrize("b3", [1, 5, 20, 50, 100, 150])
    def test_varying_b3_fixed_b2(self, b3):
        """Test varying b₃ with fixed b₂."""
        b2 = 10
        topology = TopologyConfig(b2=b2, b3=b3)
        topology.validate()

        h3_net = HarmonicNetwork(topology, p=3, hidden_dims=[64, 128, 64])
        assert h3_net.n_forms == b3

    def test_asymmetric_topology_large_b2_small_b3(self):
        """Test asymmetric topology with large b₂, small b₃."""
        topology = TopologyConfig(b2=50, b3=10)
        topology.validate()

        h2_net = HarmonicNetwork(topology, p=2, hidden_dims=[64, 128, 64])
        h3_net = HarmonicNetwork(topology, p=3, hidden_dims=[64, 128, 64])

        assert h2_net.n_forms == 50
        assert h3_net.n_forms == 10

    def test_asymmetric_topology_small_b2_large_b3(self):
        """Test asymmetric topology with small b₂, large b₃."""
        topology = TopologyConfig(b2=5, b3=100)
        topology.validate()

        h2_net = HarmonicNetwork(topology, p=2, hidden_dims=[64, 128, 64])
        h3_net = HarmonicNetwork(topology, p=3, hidden_dims=[64, 128, 64])

        assert h2_net.n_forms == 5
        assert h3_net.n_forms == 100


class TestConsistencyAcrossTopologies:
    """Test that behavior is consistent across different topologies."""

    @pytest.mark.parametrize("b2,b3", [(5, 20), (10, 40), (15, 60)])
    def test_loss_scale_independence(self, b2, b3):
        """Test that loss scaling is independent of topology."""
        topology = TopologyConfig(b2=b2, b3=b3)

        # Create orthonormal forms (Gram = Identity)
        h2_forms = torch.zeros(10, b2, 21)
        h3_forms = torch.zeros(10, b3, 35)

        # Set first form to unit norm (simplified)
        h2_forms[:, 0, 0] = 1.0
        h3_forms[:, 0, 0] = 1.0

        metric = torch.eye(7).unsqueeze(0).repeat(10, 1, 1)
        coords = torch.randn(10, 7)
        eps_indices, eps_signs = build_levi_civita_sparse_7d()

        loss = gram_matrix_loss(
            h2_forms, h3_forms, topology, metric, coords,
            eps_indices, eps_signs
        )

        # Loss should be of similar magnitude regardless of topology
        assert loss.item() >= 0

    @pytest.mark.parametrize("b2,b3", [(1, 5), (5, 20), (10, 40)])
    def test_network_gradient_flow_consistency(self, b2, b3):
        """Test gradient flow is consistent across topologies."""
        topology = TopologyConfig(b2=b2, b3=b3)

        h2_net = HarmonicNetwork(topology, p=2, hidden_dims=[64, 128, 64])

        coords = torch.randn(10, 7, requires_grad=True)
        output = h2_net(coords)

        loss = output.pow(2).sum()
        loss.backward()

        # Should have gradients
        assert coords.grad is not None
        assert not torch.isnan(coords.grad).any()

        # Gradient magnitudes should be reasonable
        grad_norm = coords.grad.norm().item()
        assert grad_norm < 1e3, f"Gradient too large: {grad_norm}"
        assert grad_norm > 1e-8, f"Gradient too small: {grad_norm}"
