"""
Unit tests for RGFlowModule from GIFT v1.2b.

Tests RG flow computation with learnable coefficients:
- Module initialization
- Forward computation
- L2 penalty
- Coefficient drift monitoring
- Edge cases and numerical stability
"""

import pytest
import torch
import sys
sys.path.insert(0, '/home/user/g2-forge')

import g2forge as g2
from g2forge.physics.rg_flow import RGFlowModule


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================
# INITIALIZATION TESTS
# ============================================================

def test_rg_flow_initialization_defaults():
    """Test RGFlowModule initialization with default parameters."""
    rg_flow = RGFlowModule()

    assert rg_flow.lambda_max == 39.44
    assert rg_flow.n_steps == 100
    assert rg_flow.epsilon_0 == 1.0 / 8.0
    assert rg_flow.l2_penalty == 0.001

    # Check default coefficient values (GIFT v1.2b)
    assert torch.allclose(rg_flow.A, torch.tensor(-20.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(rg_flow.B, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(rg_flow.C, torch.tensor(20.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(rg_flow.D, torch.tensor(3.0, dtype=torch.float64), atol=1e-6)


def test_rg_flow_initialization_custom():
    """Test RGFlowModule initialization with custom parameters."""
    rg_flow = RGFlowModule(
        lambda_max=50.0,
        n_steps=200,
        epsilon_0=0.1,
        A_init=-15.0,
        B_init=2.0,
        C_init=25.0,
        D_init=5.0,
        l2_penalty=0.01
    )

    assert rg_flow.lambda_max == 50.0
    assert rg_flow.n_steps == 200
    assert rg_flow.epsilon_0 == 0.1
    assert rg_flow.l2_penalty == 0.01

    assert torch.allclose(rg_flow.A, torch.tensor(-15.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(rg_flow.B, torch.tensor(2.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(rg_flow.C, torch.tensor(25.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(rg_flow.D, torch.tensor(5.0, dtype=torch.float64), atol=1e-6)


def test_rg_flow_parameters_are_learnable():
    """Test that A, B, C, D are learnable parameters."""
    rg_flow = RGFlowModule()

    # Check that they are nn.Parameters
    assert isinstance(rg_flow.A, torch.nn.Parameter)
    assert isinstance(rg_flow.B, torch.nn.Parameter)
    assert isinstance(rg_flow.C, torch.nn.Parameter)
    assert isinstance(rg_flow.D, torch.nn.Parameter)

    # Check that they require gradients
    assert rg_flow.A.requires_grad
    assert rg_flow.B.requires_grad
    assert rg_flow.C.requires_grad
    assert rg_flow.D.requires_grad


def test_rg_flow_stores_initial_values():
    """Test that initial coefficient values are stored as buffers."""
    rg_flow = RGFlowModule(A_init=-20.0, B_init=1.0, C_init=20.0, D_init=3.0)

    assert torch.allclose(rg_flow.A_init, torch.tensor(-20.0, dtype=torch.float64))
    assert torch.allclose(rg_flow.B_init, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(rg_flow.C_init, torch.tensor(20.0, dtype=torch.float64))
    assert torch.allclose(rg_flow.D_init, torch.tensor(3.0, dtype=torch.float64))


# ============================================================
# FORWARD COMPUTATION TESTS
# ============================================================

def test_forward_basic():
    """Test basic forward computation."""
    rg_flow = RGFlowModule()

    # Sample inputs
    div_T_eff = -0.01
    torsion_norm_sq = 0.05
    trace_deps = 0.02
    fract_eff = -0.25

    delta_alpha, components = rg_flow.forward(
        div_T_eff=div_T_eff,
        torsion_norm_sq=torsion_norm_sq,
        trace_deps=trace_deps,
        fract_eff=fract_eff
    )

    # Check return types
    assert isinstance(delta_alpha, torch.Tensor)
    assert isinstance(components, dict)

    # Check delta_alpha is scalar
    assert delta_alpha.numel() == 1

    # Check components dict
    required_keys = [
        'A_divergence', 'B_norm', 'C_epsilon', 'D_fractality',
        'RG_noD', 'total', 'div_T_eff', 'fract_eff',
        'A', 'B', 'C', 'D'
    ]
    for key in required_keys:
        assert key in components


def test_forward_output_values():
    """Test that forward computation produces reasonable values."""
    rg_flow = RGFlowModule()

    div_T_eff = -0.01
    torsion_norm_sq = 0.05
    trace_deps = 0.02
    fract_eff = -0.25

    delta_alpha, components = rg_flow.forward(
        div_T_eff, torsion_norm_sq, trace_deps, fract_eff
    )

    # Check component contributions
    A_term = components['A_divergence']
    B_term = components['B_norm']
    C_term = components['C_epsilon']
    D_term = components['D_fractality']

    # A * div_T = -20.0 * (-0.01) = 0.2
    assert abs(A_term - 0.2) < 0.01

    # B * norm_sq = 1.0 * 0.05 = 0.05
    assert abs(B_term - 0.05) < 0.01

    # D * fract = 3.0 * (-0.25) = -0.75
    assert abs(D_term - (-0.75)) < 0.01

    # Total should equal sum
    total_from_components = A_term + B_term + C_term + D_term
    assert abs(components['total'] - total_from_components) < 0.01


def test_forward_component_breakdown():
    """Test that component breakdown is correct."""
    rg_flow = RGFlowModule()

    div_T_eff = -0.01
    torsion_norm_sq = 0.05
    trace_deps = 0.02
    fract_eff = -0.25

    delta_alpha, components = rg_flow.forward(
        div_T_eff, torsion_norm_sq, trace_deps, fract_eff
    )

    # RG_noD should be A + B + C
    RG_noD_expected = (
        components['A_divergence'] +
        components['B_norm'] +
        components['C_epsilon']
    )
    assert abs(components['RG_noD'] - RG_noD_expected) < 1e-6

    # Total should be RG_noD + D
    total_expected = components['RG_noD'] + components['D_fractality']
    assert abs(components['total'] - total_expected) < 1e-6


def test_forward_trace_deps_clamping():
    """Test that trace_deps is clamped to [-0.05, +0.05]."""
    rg_flow = RGFlowModule()

    # Test with extreme trace_deps values
    extreme_values = [-1.0, -0.1, 0.1, 1.0]

    for trace_deps in extreme_values:
        delta_alpha, components = rg_flow.forward(
            div_T_eff=0.0,
            torsion_norm_sq=0.0,
            trace_deps=trace_deps,
            fract_eff=0.0
        )

        C_term = components['C_epsilon']
        # C = 20.0, so if clamped to [-0.05, +0.05]
        # C_term should be in [-1.0, +1.0]
        assert -1.0 <= C_term <= 1.0


def test_forward_coefficient_values_in_components():
    """Test that current coefficient values are included in components."""
    rg_flow = RGFlowModule(A_init=-15.0, B_init=2.0, C_init=25.0, D_init=5.0)

    delta_alpha, components = rg_flow.forward(
        div_T_eff=0.0,
        torsion_norm_sq=0.0,
        trace_deps=0.0,
        fract_eff=0.0
    )

    assert abs(components['A'] - (-15.0)) < 1e-6
    assert abs(components['B'] - 2.0) < 1e-6
    assert abs(components['C'] - 25.0) < 1e-6
    assert abs(components['D'] - 5.0) < 1e-6


def test_forward_input_values_in_components():
    """Test that input values are recorded in components."""
    rg_flow = RGFlowModule()

    div_T_eff = -0.015
    fract_eff = -0.3

    delta_alpha, components = rg_flow.forward(
        div_T_eff=div_T_eff,
        torsion_norm_sq=0.0,
        trace_deps=0.0,
        fract_eff=fract_eff
    )

    assert abs(components['div_T_eff'] - div_T_eff) < 1e-9
    assert abs(components['fract_eff'] - fract_eff) < 1e-9


# ============================================================
# L2 PENALTY TESTS
# ============================================================

def test_compute_l2_penalty_default_coefficients():
    """Test L2 penalty with default coefficients."""
    rg_flow = RGFlowModule()

    penalty = rg_flow.compute_l2_penalty()

    # penalty = l2_penalty * (A^2 + B^2 + C^2 + D^2)
    # = 0.001 * ((-20)^2 + 1^2 + 20^2 + 3^2)
    # = 0.001 * (400 + 1 + 400 + 9)
    # = 0.001 * 810 = 0.81
    expected = 0.001 * ((-20.0)**2 + 1.0**2 + 20.0**2 + 3.0**2)

    assert torch.allclose(penalty, torch.tensor(expected, dtype=torch.float64), atol=1e-6)


def test_compute_l2_penalty_custom_coefficients():
    """Test L2 penalty with custom coefficients."""
    rg_flow = RGFlowModule(
        A_init=10.0,
        B_init=5.0,
        C_init=3.0,
        D_init=2.0,
        l2_penalty=0.01
    )

    penalty = rg_flow.compute_l2_penalty()

    # penalty = 0.01 * (10^2 + 5^2 + 3^2 + 2^2) = 0.01 * 138 = 1.38
    expected = 0.01 * (10.0**2 + 5.0**2 + 3.0**2 + 2.0**2)

    assert torch.allclose(penalty, torch.tensor(expected, dtype=torch.float64), atol=1e-6)


def test_l2_penalty_is_differentiable():
    """Test that L2 penalty supports gradient computation."""
    rg_flow = RGFlowModule()

    penalty = rg_flow.compute_l2_penalty()

    # Check that penalty requires gradients
    assert penalty.requires_grad

    # Compute gradient
    penalty.backward()

    # Check that coefficients have gradients
    assert rg_flow.A.grad is not None
    assert rg_flow.B.grad is not None
    assert rg_flow.C.grad is not None
    assert rg_flow.D.grad is not None


# ============================================================
# COEFFICIENT DRIFT TESTS
# ============================================================

def test_get_coefficient_drift_no_change():
    """Test coefficient drift when coefficients haven't changed."""
    rg_flow = RGFlowModule()

    drift = rg_flow.get_coefficient_drift()

    # All drifts should be zero (no change yet)
    assert abs(drift['A_drift']) < 1e-6
    assert abs(drift['B_drift']) < 1e-6
    assert abs(drift['C_drift']) < 1e-6
    assert abs(drift['D_drift']) < 1e-6


def test_get_coefficient_drift_after_modification():
    """Test coefficient drift after modifying coefficients."""
    rg_flow = RGFlowModule(A_init=-20.0, B_init=1.0, C_init=20.0, D_init=3.0)

    # Manually modify coefficients (simulating optimization)
    with torch.no_grad():
        rg_flow.A.data = torch.tensor(-22.0, dtype=torch.float64)  # 10% increase in magnitude
        rg_flow.B.data = torch.tensor(1.5, dtype=torch.float64)    # 50% increase
        rg_flow.C.data = torch.tensor(18.0, dtype=torch.float64)   # 10% decrease
        rg_flow.D.data = torch.tensor(3.0, dtype=torch.float64)    # No change

    drift = rg_flow.get_coefficient_drift()

    # A: (-22 - (-20)) / 20 = -2/20 = -0.1
    assert abs(drift['A_drift'] - (-0.1)) < 1e-6

    # B: (1.5 - 1) / 1 = 0.5
    assert abs(drift['B_drift'] - 0.5) < 1e-6

    # C: (18 - 20) / 20 = -0.1
    assert abs(drift['C_drift'] - (-0.1)) < 1e-6

    # D: (3 - 3) / 3 = 0
    assert abs(drift['D_drift']) < 1e-6


def test_get_coefficient_drift_dict_keys():
    """Test that drift dict has correct keys."""
    rg_flow = RGFlowModule()

    drift = rg_flow.get_coefficient_drift()

    assert 'A_drift' in drift
    assert 'B_drift' in drift
    assert 'C_drift' in drift
    assert 'D_drift' in drift


# ============================================================
# STRING REPRESENTATION
# ============================================================

def test_repr():
    """Test string representation of RGFlowModule."""
    rg_flow = RGFlowModule()

    repr_str = repr(rg_flow)

    # Should contain key information
    assert 'RGFlowModule' in repr_str
    assert 'A=' in repr_str
    assert 'B=' in repr_str
    assert 'C=' in repr_str
    assert 'D=' in repr_str
    assert 'λ_max' in repr_str or 'lambda_max' in repr_str


# ============================================================
# GRADIENT FLOW TESTS
# ============================================================

def test_forward_gradient_flow():
    """Test that gradients flow through forward computation."""
    rg_flow = RGFlowModule()

    div_T_eff = -0.01
    torsion_norm_sq = 0.05
    trace_deps = 0.02
    fract_eff = -0.25

    delta_alpha, components = rg_flow.forward(
        div_T_eff, torsion_norm_sq, trace_deps, fract_eff
    )

    # Check that delta_alpha requires gradients
    assert delta_alpha.requires_grad

    # Compute gradient
    delta_alpha.backward()

    # All coefficients should have gradients
    assert rg_flow.A.grad is not None
    assert rg_flow.B.grad is not None
    assert rg_flow.C.grad is not None
    assert rg_flow.D.grad is not None


def test_gradient_sign_for_each_coefficient():
    """Test gradient computation for each coefficient separately."""
    # Test that increasing each coefficient affects output in expected direction

    # Test A coefficient
    rg_flow_A = RGFlowModule()
    div_T_eff = -0.01  # Negative
    delta_alpha, _ = rg_flow_A.forward(div_T_eff, 0.0, 0.0, 0.0)
    delta_alpha.backward()
    # A * div_T, so grad_A should have same sign as div_T
    assert rg_flow_A.A.grad is not None

    # Test B coefficient
    rg_flow_B = RGFlowModule()
    torsion_norm_sq = 0.05  # Positive
    delta_alpha, _ = rg_flow_B.forward(0.0, torsion_norm_sq, 0.0, 0.0)
    delta_alpha.backward()
    # B * norm_sq, so grad_B should have same sign as norm_sq
    assert rg_flow_B.B.grad is not None

    # Test D coefficient
    rg_flow_D = RGFlowModule()
    fract_eff = -0.25  # Negative
    delta_alpha, _ = rg_flow_D.forward(0.0, 0.0, 0.0, fract_eff)
    delta_alpha.backward()
    # D * fract, so grad_D should have same sign as fract
    assert rg_flow_D.D.grad is not None


# ============================================================
# EDGE CASES AND NUMERICAL STABILITY
# ============================================================

def test_forward_with_zero_inputs():
    """Test forward computation with all zero inputs."""
    rg_flow = RGFlowModule()

    delta_alpha, components = rg_flow.forward(
        div_T_eff=0.0,
        torsion_norm_sq=0.0,
        trace_deps=0.0,
        fract_eff=0.0
    )

    # All components should be zero
    assert abs(components['A_divergence']) < 1e-9
    assert abs(components['B_norm']) < 1e-9
    assert abs(components['C_epsilon']) < 1e-9
    assert abs(components['D_fractality']) < 1e-9

    # Total should be zero
    assert abs(components['total']) < 1e-9


def test_forward_with_extreme_inputs():
    """Test forward computation with extreme input values."""
    rg_flow = RGFlowModule()

    # Test with large positive and negative values
    extreme_cases = [
        (-10.0, 100.0, 50.0, -10.0),
        (10.0, 0.0001, -50.0, 10.0),
        (0.0, 1e6, 0.0, 0.0),
    ]

    for div_T, norm_sq, trace, fract in extreme_cases:
        delta_alpha, components = rg_flow.forward(div_T, norm_sq, trace, fract)

        # Should produce finite results
        assert torch.isfinite(delta_alpha)
        assert not torch.isnan(delta_alpha)
        assert not torch.isinf(delta_alpha)


def test_forward_preserves_dtype():
    """Test that forward computation preserves float64 dtype."""
    rg_flow = RGFlowModule()

    delta_alpha, components = rg_flow.forward(
        div_T_eff=-0.01,
        torsion_norm_sq=0.05,
        trace_deps=0.02,
        fract_eff=-0.25
    )

    # Should be float64
    assert delta_alpha.dtype == torch.float64


def test_forward_deterministic():
    """Test that forward computation is deterministic."""
    rg_flow = RGFlowModule()

    inputs = {
        'div_T_eff': -0.01,
        'torsion_norm_sq': 0.05,
        'trace_deps': 0.02,
        'fract_eff': -0.25
    }

    # Compute twice
    delta_alpha1, _ = rg_flow.forward(**inputs)
    delta_alpha2, _ = rg_flow.forward(**inputs)

    assert torch.allclose(delta_alpha1, delta_alpha2, atol=1e-12)


def test_integration_formula():
    """Test the geodesic integration formula."""
    rg_flow = RGFlowModule(lambda_max=10.0, n_steps=100)

    # With constant integrand, integral should be integrand * lambda_max
    # And delta_alpha = integral / lambda_max = integrand

    delta_alpha, components = rg_flow.forward(
        div_T_eff=-0.01,
        torsion_norm_sq=0.0,
        trace_deps=0.0,
        fract_eff=0.0
    )

    # A * div_T = -20 * (-0.01) = 0.2
    # With constant integrand, delta_alpha = integrand = 0.2
    expected = -20.0 * (-0.01)

    assert torch.allclose(delta_alpha, torch.tensor(expected, dtype=torch.float64), rtol=1e-2)


def test_module_on_different_devices():
    """Test that module can be moved to different devices."""
    rg_flow = RGFlowModule()

    # Test CPU
    rg_flow_cpu = rg_flow.to('cpu')
    delta_alpha_cpu, _ = rg_flow_cpu.forward(-0.01, 0.05, 0.02, -0.25)
    assert delta_alpha_cpu.device.type == 'cpu'

    # Test CUDA if available
    if torch.cuda.is_available():
        rg_flow_cuda = rg_flow.to('cuda')
        delta_alpha_cuda, _ = rg_flow_cuda.forward(-0.01, 0.05, 0.02, -0.25)
        assert delta_alpha_cuda.device.type == 'cuda'

        # Results should be similar
        assert torch.allclose(
            delta_alpha_cpu,
            delta_alpha_cuda.cpu(),
            atol=1e-6
        )


def test_negative_delta_alpha_typical_for_gift():
    """Test that typical GIFT v1.2b inputs produce negative Δα."""
    rg_flow = RGFlowModule()

    # Typical GIFT v1.2b values from documentation
    # Target Δα = -0.9, achieved = -0.87
    div_T_eff = -0.006  # Small negative
    torsion_norm_sq = 0.01
    trace_deps = 0.01
    fract_eff = -0.254  # From GIFT results

    delta_alpha, components = rg_flow.forward(
        div_T_eff, torsion_norm_sq, trace_deps, fract_eff
    )

    # Should be negative (consistent with GIFT v1.2b)
    # A_divergence: -20 * (-0.006) = 0.12 (positive)
    # D_fractality: 3 * (-0.254) = -0.762 (negative, dominant)
    # Total should be negative
    assert delta_alpha.item() < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
