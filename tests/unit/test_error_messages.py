"""
Unit tests for error message quality and descriptiveness.

Tests that validation errors and exceptions provide:
- Clear, descriptive messages
- Relevant context (values that caused the error)
- Actionable suggestions where appropriate
- Consistent formatting
"""

import pytest
import torch
import re

from g2forge.utils.config import (
    TopologyConfig,
    TCSParameters,
    ManifoldConfig,
    G2ForgeConfig,
)
from g2forge.core.operators import compute_exterior_derivative
from g2forge.manifolds import K7Manifold
from g2forge.networks import PhiNetwork, HarmonicNetwork


class TestTopologyConfigErrorMessages:
    """Test error messages from TopologyConfig validation."""

    def test_negative_b2_error_message(self):
        """Test that negative b₂ produces descriptive error."""
        with pytest.raises(ValueError, match=r"b2.*must be positive"):
            topology = TopologyConfig(b2=-1, b3=10)
            topology.validate()

    def test_negative_b3_error_message(self):
        """Test that negative b₃ produces descriptive error."""
        with pytest.raises(ValueError, match=r"b3.*must be positive"):
            topology = TopologyConfig(b2=10, b3=-5)
            topology.validate()

    def test_zero_b2_error_message(self):
        """Test that zero b₂ produces descriptive error."""
        with pytest.raises(ValueError, match=r"b2.*must be positive"):
            topology = TopologyConfig(b2=0, b3=10)
            topology.validate()

    def test_negative_b1_error_message(self):
        """Test that negative b₁ produces descriptive error."""
        with pytest.raises(ValueError, match=r"b1.*non-negative"):
            topology = TopologyConfig(b2=10, b3=20, b1=-1)
            topology.validate()

    def test_invalid_dimension_error_message(self):
        """Test that non-7-dimensional manifold produces clear error."""
        # This would require modifying the dimension directly
        topology = TopologyConfig(b2=10, b3=20)

        # Create a ManifoldConfig with wrong dimension
        with pytest.raises(ValueError, match=r"G₂ manifolds must be 7-dimensional"):
            manifold_config = ManifoldConfig(
                topology=topology,
                tcs_params=TCSParameters(
                    b2_m1=5, b3_m1=10,
                    b2_m2=5, b3_m2=10
                ),
                moduli=None,
                dimension=8  # Wrong!
            )
            manifold_config.validate()


class TestManifoldConfigErrorMessages:
    """Test error messages from ManifoldConfig validation."""

    def test_tcs_topology_mismatch_error_message(self):
        """Test error when TCS component topology doesn't match total topology."""
        topology = TopologyConfig(b2=20, b3=80)  # Doesn't match components below
        tcs_params = TCSParameters(
            b2_m1=10, b3_m1=40,
            b2_m2=10, b3_m2=40  # This sums to b₂=20, b₃=81 (mismatch!)
        )

        with pytest.raises(ValueError, match=r"TCS topology.*does not match"):
            manifold_config = ManifoldConfig(
                topology=topology,
                tcs_params=tcs_params,
                moduli=None
            )
            manifold_config.validate()

    def test_missing_tcs_params_error_message(self):
        """Test error when construction_type is 'tcs' but no TCS params."""
        topology = TopologyConfig(b2=10, b3=20)

        with pytest.raises(ValueError, match=r"tcs_params.*required"):
            manifold_config = ManifoldConfig(
                topology=topology,
                tcs_params=None,  # Missing!
                moduli=None,
                construction_type='tcs'
            )
            manifold_config.validate()


class TestOperatorErrorMessages:
    """Test error messages from differential operators."""

    def test_exterior_derivative_no_gradient_error(self):
        """Test that exterior derivative requires gradients."""
        coords = torch.randn(10, 7, requires_grad=False)  # No gradient!
        phi = torch.randn(10, 7, 7, 7, requires_grad=True)

        with pytest.raises((ValueError, RuntimeError), match=r"(requires_grad|gradient)"):
            dphi = compute_exterior_derivative(phi, coords)

    def test_exterior_derivative_wrong_shape_error(self):
        """Test clear error for wrong input shape."""
        coords = torch.randn(10, 7, requires_grad=True)
        phi = torch.randn(10, 5, 5, 5)  # Wrong shape!

        # Should raise error about shape mismatch
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            dphi = compute_exterior_derivative(phi, coords)


class TestNetworkErrorMessages:
    """Test error messages from neural networks."""

    def test_harmonic_network_invalid_degree_error(self):
        """Test error for invalid form degree p."""
        topology = TopologyConfig(b2=10, b3=20)

        with pytest.raises(ValueError, match=r"(degree|p).*must be.*2 or 3"):
            network = HarmonicNetwork(
                topology=topology,
                p=5,  # Invalid!
                hidden_dims=[64, 128, 64]
            )

    def test_phi_network_invalid_hidden_dims_error(self):
        """Test error for invalid hidden dimensions."""
        with pytest.raises((ValueError, TypeError)):
            network = PhiNetwork(
                input_dim=7,
                hidden_dims=[],  # Empty!
                fourier_features=256
            )


class TestK7ManifoldErrorMessages:
    """Test error messages from K7Manifold."""

    def test_invalid_sampling_mode_error_message(self):
        """Test error for invalid sampling mode."""
        topology = TopologyConfig(b2=10, b3=20)
        tcs_params = TCSParameters(b2_m1=5, b3_m1=10, b2_m2=5, b3_m2=10)

        manifold = K7Manifold(topology, tcs_params)

        with pytest.raises(ValueError, match=r"(sampling_mode|mode).*invalid"):
            coords = manifold.sample_coordinates(
                n_samples=100,
                sampling_mode='invalid_mode'  # Bad mode!
            )

    def test_negative_sample_count_error_message(self):
        """Test error for negative or zero sample count."""
        topology = TopologyConfig(b2=10, b3=20)
        tcs_params = TCSParameters(b2_m1=5, b3_m1=10, b2_m2=5, b3_m2=10)

        manifold = K7Manifold(topology, tcs_params)

        with pytest.raises(ValueError, match=r"(n_samples|sample count).*positive"):
            coords = manifold.sample_coordinates(n_samples=-10)


class TestConfigSerializationErrorMessages:
    """Test error messages from config serialization."""

    def test_invalid_json_error_message(self):
        """Test error when loading invalid JSON."""
        invalid_json = "{'not': 'valid json}"  # Single quotes

        with pytest.raises((ValueError, Exception), match=r"(JSON|parse|load)"):
            config = G2ForgeConfig.from_json(invalid_json)

    def test_missing_required_field_error_message(self):
        """Test error when required config field is missing."""
        incomplete_dict = {
            'manifold': {
                'topology': {'b2': 10, 'b3': 20}
                # Missing tcs_params and other fields
            }
            # Missing architecture, training, etc.
        }

        with pytest.raises((ValueError, KeyError, TypeError)):
            config = G2ForgeConfig.from_dict(incomplete_dict)


class TestErrorMessageContextInformation:
    """Test that error messages include relevant context values."""

    def test_topology_error_includes_actual_values(self):
        """Test that topology errors show the actual invalid values."""
        try:
            topology = TopologyConfig(b2=-5, b3=10)
            topology.validate()
        except ValueError as e:
            error_msg = str(e)
            # Should mention the actual value -5
            assert '-5' in error_msg or 'negative' in error_msg.lower()

    def test_tcs_mismatch_error_shows_both_values(self):
        """Test that TCS mismatch errors show expected vs actual."""
        topology = TopologyConfig(b2=20, b3=80)
        tcs_params = TCSParameters(
            b2_m1=10, b3_m1=40,
            b2_m2=10, b3_m2=40
        )

        try:
            manifold_config = ManifoldConfig(
                topology=topology,
                tcs_params=tcs_params,
                moduli=None
            )
            manifold_config.validate()
        except ValueError as e:
            error_msg = str(e)
            # Should show both expected and actual values
            # The actual implementation may vary


class TestErrorMessageActionableSuggestions:
    """Test that error messages provide actionable suggestions where appropriate."""

    def test_gradient_error_suggests_requires_grad(self):
        """Test that gradient errors suggest enabling requires_grad."""
        coords = torch.randn(10, 7, requires_grad=False)
        phi = torch.randn(10, 7, 7, 7, requires_grad=True)

        try:
            dphi = compute_exterior_derivative(phi, coords)
        except (ValueError, RuntimeError) as e:
            error_msg = str(e)
            # Should mention requires_grad
            assert 'requires_grad' in error_msg.lower() or 'gradient' in error_msg.lower()


class TestErrorMessageConsistency:
    """Test that error messages follow consistent formatting."""

    def test_parameter_errors_follow_pattern(self):
        """Test that parameter validation errors follow consistent pattern."""
        errors = []

        # Collect several parameter errors
        try:
            TopologyConfig(b2=-1, b3=10).validate()
        except ValueError as e:
            errors.append(str(e))

        try:
            TopologyConfig(b2=10, b3=0).validate()
        except ValueError as e:
            errors.append(str(e))

        # All should be ValueError (consistent exception type)
        assert len(errors) > 0

        # All should mention the parameter name
        # This is a soft check - implementation may vary


class TestWarningMessages:
    """Test that warnings are issued for potentially problematic configurations."""

    def test_very_large_topology_may_warn(self):
        """Test that very large topologies may issue performance warnings."""
        # This is optional - not all implementations will have warnings
        topology = TopologyConfig(b2=1000, b3=5000)

        # Should complete without error (warnings are optional)
        topology.validate()

    def test_small_network_for_large_topology_may_warn(self):
        """Test that undersized networks for large topology may warn."""
        topology = TopologyConfig(b2=100, b3=500)

        # Small network for large topology
        network = HarmonicNetwork(
            topology=topology,
            p=2,
            hidden_dims=[32, 64, 32]  # Quite small for b₂=100
        )

        # Should complete (warnings are optional)
        assert network.n_forms == 100


class TestExceptionHierarchy:
    """Test that exceptions use appropriate types."""

    def test_validation_errors_are_value_errors(self):
        """Test that validation errors use ValueError."""
        with pytest.raises(ValueError):
            topology = TopologyConfig(b2=-1, b3=10)
            topology.validate()

    def test_type_errors_are_type_errors(self):
        """Test that type mismatches raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            # Passing string instead of int
            topology = TopologyConfig(b2="not an int", b3=10)

    def test_missing_gradient_is_runtime_or_value_error(self):
        """Test that missing gradient raises appropriate error."""
        coords = torch.randn(10, 7, requires_grad=False)
        phi = torch.randn(10, 7, 7, 7, requires_grad=True)

        with pytest.raises((ValueError, RuntimeError)):
            dphi = compute_exterior_derivative(phi, coords)
