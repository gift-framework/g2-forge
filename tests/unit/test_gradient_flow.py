"""
Unit tests for gradient flow and backpropagation.

Tests that gradients propagate correctly through:
- Differential geometry operators
- Loss functions
- Neural networks
- Training pipeline

Ensures that autodiff works correctly for all PDE-constrained optimization.
"""

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from g2forge.core.operators import (
    hodge_star_3,
    compute_exterior_derivative,
    compute_coclosure,
    reconstruct_metric_from_phi,
    build_levi_civita_sparse_7d
)
from g2forge.core.losses import (
    torsion_closure_loss,
    torsion_coclosure_loss,
    volume_loss,
    gram_matrix_loss,
)
from g2forge.networks import PhiNetwork, HarmonicNetwork
from g2forge.utils import TopologyConfig


class TestGradientFlowThroughOperators:
    """Test gradient flow through differential geometry operators."""

    def test_gradients_flow_through_exterior_derivative(self, sample_coordinates):
        """Test that gradients propagate through exterior derivative."""
        coords = sample_coordinates.clone().requires_grad_(True)

        # Create simple 3-form depending on coords
        phi = torch.zeros(coords.shape[0], 7, 7, 7, requires_grad=True)
        # Make it depend on coords through a simple function
        phi = phi + coords[:, 0].view(-1, 1, 1, 1) * 0.1

        # Compute exterior derivative
        dphi = compute_exterior_derivative(phi, coords)

        # Create scalar loss
        loss = dphi.sum()

        # Backpropagate
        loss.backward()

        # Check gradients exist and are non-zero
        assert coords.grad is not None, "Gradient w.r.t. coords should exist"
        assert phi.grad is not None, "Gradient w.r.t. phi should exist"
        assert coords.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_gradients_flow_through_hodge_star(self, sample_phi_antisymmetric, sample_metric, levi_civita):
        """Test gradient flow through Hodge star operator."""
        eps_indices, eps_signs = levi_civita

        phi = sample_phi_antisymmetric.clone().requires_grad_(True)
        metric = sample_metric.clone().requires_grad_(True)

        # Compute Hodge star
        star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)

        # Scalar loss
        loss = star_phi.sum()
        loss.backward()

        # Check gradients
        assert phi.grad is not None
        assert metric.grad is not None
        assert phi.grad.abs().sum() > 0
        assert metric.grad.abs().sum() > 0

    def test_gradients_flow_through_coclosure(self, sample_phi_antisymmetric, sample_coordinates, sample_metric, levi_civita):
        """Test gradient flow through coclosure operator."""
        eps_indices, eps_signs = levi_civita

        coords = sample_coordinates.clone().requires_grad_(True)
        phi = sample_phi_antisymmetric.clone().requires_grad_(True)
        metric = sample_metric.clone().requires_grad_(True)

        # Compute coclosure
        delta_phi = compute_coclosure(phi, coords, metric, eps_indices, eps_signs)

        # Scalar loss
        loss = delta_phi.sum()
        loss.backward()

        # Check gradients
        assert coords.grad is not None
        assert phi.grad is not None
        assert metric.grad is not None

    def test_gradients_flow_through_metric_reconstruction(self, sample_phi_antisymmetric):
        """Test gradient flow through metric reconstruction."""
        phi = sample_phi_antisymmetric.clone().requires_grad_(True)

        # Reconstruct metric from phi
        metric = reconstruct_metric_from_phi(phi)

        # Scalar loss
        loss = metric.sum()
        loss.backward()

        # Check gradients
        assert phi.grad is not None
        assert phi.grad.abs().sum() > 0


class TestGradientFlowThroughLosses:
    """Test gradient flow through loss functions."""

    def test_torsion_closure_loss_gradients(self, sample_coordinates):
        """Test that torsion closure loss produces gradients."""
        coords = sample_coordinates.clone().requires_grad_(True)

        # Create phi depending on coords
        phi = torch.randn(coords.shape[0], 7, 7, 7, requires_grad=True)

        # Compute exterior derivative and loss
        dphi = compute_exterior_derivative(phi, coords)
        loss = torsion_closure_loss(dphi)

        # Backpropagate
        loss.backward()

        # Check gradients exist
        assert coords.grad is not None
        assert phi.grad is not None

    def test_torsion_coclosure_loss_gradients(self, sample_phi_antisymmetric, sample_coordinates, sample_metric, levi_civita):
        """Test that torsion coclosure loss produces gradients."""
        eps_indices, eps_signs = levi_civita

        coords = sample_coordinates.clone().requires_grad_(True)
        phi = sample_phi_antisymmetric.clone().requires_grad_(True)
        metric = sample_metric.clone().requires_grad_(True)

        # Compute d*phi
        star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)
        d_star_phi = compute_exterior_derivative(star_phi, coords)

        loss = torsion_coclosure_loss(d_star_phi)
        loss.backward()

        # Check gradients
        assert coords.grad is not None
        assert phi.grad is not None
        assert metric.grad is not None

    def test_volume_loss_gradients(self, sample_metric):
        """Test that volume loss produces gradients."""
        metric = sample_metric.clone().requires_grad_(True)

        loss = volume_loss(metric)
        loss.backward()

        assert metric.grad is not None
        assert metric.grad.abs().sum() > 0

    def test_gram_matrix_loss_gradients(self, sample_2form_antisymmetric, sample_metric, sample_coordinates, levi_civita):
        """Test that gram matrix loss produces gradients."""
        eps_indices, eps_signs = levi_civita

        h2_forms = sample_2form_antisymmetric.clone().requires_grad_(True)
        h3_forms = torch.randn(5, 35, requires_grad=True)
        metric = sample_metric.clone().requires_grad_(True)
        coords = sample_coordinates.clone().requires_grad_(True)

        topology = TopologyConfig(b2=5, b3=5)

        loss = gram_matrix_loss(
            h2_forms, h3_forms, topology, metric, coords,
            eps_indices, eps_signs
        )
        loss.backward()

        # Check gradients for all inputs
        assert h2_forms.grad is not None
        assert h3_forms.grad is not None
        assert metric.grad is not None
        assert coords.grad is not None


class TestGradientFlowThroughNetworks:
    """Test gradient flow through neural networks."""

    def test_phi_network_gradient_flow(self, phi_network_small):
        """Test that gradients flow through PhiNetwork."""
        network = phi_network_small
        coords = torch.randn(10, 7, requires_grad=True)

        # Forward pass
        phi = network.get_phi_tensor(coords)

        # Scalar loss
        loss = phi.sum()
        loss.backward()

        # Check gradients exist for network parameters
        for name, param in network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

        # Check gradient for input
        assert coords.grad is not None

    def test_harmonic_network_gradient_flow(self, small_topology_config):
        """Test that gradients flow through HarmonicNetwork."""
        topology = small_topology_config.manifold.topology
        network = HarmonicNetwork(topology, p=2, hidden_dims=[64, 128, 64])

        coords = torch.randn(10, 7, requires_grad=True)

        # Forward pass
        h2_forms = network(coords)

        # Scalar loss
        loss = h2_forms.sum()
        loss.backward()

        # Check gradients for network parameters
        for name, param in network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

        # Check input gradient
        assert coords.grad is not None

    def test_gradient_flow_through_full_forward_pass(self, phi_network_small, small_topology_config):
        """Test gradient flow through complete forward pass (phi -> dphi -> loss)."""
        network = phi_network_small
        coords = torch.randn(10, 7, requires_grad=True)

        # Forward pass: coords -> phi -> dphi -> loss
        phi = network.get_phi_tensor(coords)
        dphi = compute_exterior_derivative(phi, coords)
        loss = torsion_closure_loss(dphi)

        # Backpropagate
        loss.backward()

        # Check gradients throughout the chain
        assert coords.grad is not None
        for param in network.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0


class TestGradientMagnitudes:
    """Test that gradient magnitudes are reasonable (no explosion/vanishing)."""

    def test_gradient_magnitude_reasonable_for_phi_network(self, phi_network_small):
        """Test that PhiNetwork gradients are in reasonable range."""
        network = phi_network_small
        coords = torch.randn(100, 7, requires_grad=True)

        phi = network.get_phi_tensor(coords)
        loss = phi.pow(2).mean()
        loss.backward()

        # Check gradient magnitudes
        for name, param in network.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1e3, f"Gradient too large for {name}: {grad_norm:.2e}"
                assert grad_norm > 1e-6, f"Gradient too small for {name}: {grad_norm:.2e}"

    def test_gradient_magnitude_through_exterior_derivative(self, sample_coordinates):
        """Test that exterior derivative doesn't cause gradient explosion."""
        coords = sample_coordinates.clone().requires_grad_(True)

        # Simple phi depending on coords
        phi = torch.randn(coords.shape[0], 7, 7, 7, requires_grad=True)
        phi = phi + coords[:, 0].view(-1, 1, 1, 1) * 0.1

        dphi = compute_exterior_derivative(phi, coords)
        loss = dphi.pow(2).mean()
        loss.backward()

        # Check gradient magnitudes
        coord_grad_norm = coords.grad.norm().item()
        phi_grad_norm = phi.grad.norm().item()

        assert coord_grad_norm < 1e2, f"Coord gradient too large: {coord_grad_norm:.2e}"
        assert phi_grad_norm < 1e2, f"Phi gradient too large: {phi_grad_norm:.2e}"


class TestSecondOrderGradients:
    """Test second-order gradients for Hessian-based methods."""

    def test_second_order_gradients_through_loss(self, sample_phi_antisymmetric):
        """Test that second-order gradients can be computed."""
        phi = sample_phi_antisymmetric.clone().requires_grad_(True)

        # Simple loss
        loss = phi.pow(2).sum()

        # First-order gradient
        grad_phi = torch.autograd.grad(loss, phi, create_graph=True)[0]

        # Second-order gradient (Hessian diagonal approximation)
        loss2 = grad_phi.pow(2).sum()
        grad2_phi = torch.autograd.grad(loss2, phi)[0]

        assert grad2_phi is not None
        assert not torch.isnan(grad2_phi).any()

    def test_second_order_gradients_through_phi_network(self, phi_network_small):
        """Test second-order gradients through neural network."""
        network = phi_network_small
        coords = torch.randn(5, 7, requires_grad=True)

        # Forward pass
        phi = network.get_phi_tensor(coords)
        loss = phi.pow(2).sum()

        # First-order gradient
        grad_coords = torch.autograd.grad(loss, coords, create_graph=True)[0]

        # Second-order gradient
        loss2 = grad_coords.pow(2).sum()
        grad2_coords = torch.autograd.grad(loss2, coords)[0]

        assert grad2_coords is not None
        assert not torch.isnan(grad2_coords).any()


class TestGradientNaNDetection:
    """Test detection and handling of NaN gradients."""

    def test_nan_detection_in_loss(self):
        """Test that NaN in loss is detected."""
        x = torch.tensor([1.0], requires_grad=True)
        loss = torch.tensor(float('nan'))

        # Attempting to backward through NaN should be detectable
        try:
            loss.backward()
            # If no error, check gradient
            if x.grad is not None:
                assert torch.isnan(x.grad).any(), "Should have NaN gradient"
        except RuntimeError:
            # Some PyTorch versions raise error for NaN backward
            pass

    def test_gradient_with_inf_values(self):
        """Test behavior with inf values."""
        x = torch.tensor([1.0], requires_grad=True)

        # Loss that produces inf
        loss = torch.exp(x * 1000)  # Will overflow to inf

        try:
            loss.backward()
            if x.grad is not None:
                assert torch.isinf(x.grad).any() or x.grad.abs() > 1e10
        except RuntimeError:
            pass


class TestLossGradientIndependence:
    """Test that individual loss components produce independent gradients."""

    @pytest.mark.parametrize("loss_fn", [
        torsion_closure_loss,
        torsion_coclosure_loss,
        volume_loss,
    ])
    def test_individual_loss_produces_valid_gradients(self, loss_fn, sample_phi_antisymmetric, sample_coordinates, sample_metric, levi_civita):
        """Test that each loss function independently produces valid gradients."""
        eps_indices, eps_signs = levi_civita

        # Prepare inputs with gradients
        phi = sample_phi_antisymmetric.clone().requires_grad_(True)
        coords = sample_coordinates.clone().requires_grad_(True)
        metric = sample_metric.clone().requires_grad_(True)

        # Compute loss based on function
        if loss_fn == torsion_closure_loss:
            dphi = compute_exterior_derivative(phi, coords)
            loss = loss_fn(dphi)
        elif loss_fn == torsion_coclosure_loss:
            star_phi = hodge_star_3(phi, metric, eps_indices, eps_signs)
            d_star_phi = compute_exterior_derivative(star_phi, coords)
            loss = loss_fn(d_star_phi)
        elif loss_fn == volume_loss:
            loss = loss_fn(metric)
        else:
            pytest.skip(f"Loss function {loss_fn} not handled")

        # Backpropagate
        loss.backward()

        # Check that at least one input has gradient
        has_gradient = (
            (phi.grad is not None and phi.grad.abs().sum() > 0) or
            (coords.grad is not None and coords.grad.abs().sum() > 0) or
            (metric.grad is not None and metric.grad.abs().sum() > 0)
        )

        assert has_gradient, f"No gradient for {loss_fn.__name__}"


class TestGradientFlowConsistency:
    """Test consistency of gradient flow across different batch sizes."""

    def test_gradient_consistency_across_batch_sizes(self, phi_network_small):
        """Test that gradient direction is consistent for different batch sizes."""
        network = phi_network_small

        # Same coordinate repeated
        coord_single = torch.randn(1, 7)
        coords_batch = coord_single.repeat(10, 1)

        coords_single = coord_single.clone().requires_grad_(True)
        coords_batch = coords_batch.clone().requires_grad_(True)

        # Forward pass
        phi_single = network.get_phi_tensor(coords_single)
        phi_batch = network.get_phi_tensor(coords_batch)

        # Loss
        loss_single = phi_single.pow(2).mean()
        loss_batch = phi_batch.pow(2).mean()

        # Backward
        loss_single.backward()
        loss_batch.backward()

        # Gradients should point in same direction (may differ in magnitude)
        grad_single = coords_single.grad[0]
        grad_batch = coords_batch.grad[0]  # First sample

        # Normalize and compare direction
        grad_single_norm = grad_single / (grad_single.norm() + 1e-8)
        grad_batch_norm = grad_batch / (grad_batch.norm() + 1e-8)

        # Cosine similarity should be close to 1
        cosine_sim = (grad_single_norm * grad_batch_norm).sum()
        assert cosine_sim > 0.9, f"Gradient directions differ: cosine sim = {cosine_sim:.3f}"
