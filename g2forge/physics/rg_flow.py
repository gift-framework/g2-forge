"""
RG Flow Module for GIFT 2.1

Implements renormalization group (RG) flow computation with learnable coefficients.
This extends beyond pure G₂ geometry to include physical RG running.

From GIFT 2.1:
    Δα = (1/λ_max) ∫₀^λ_max ℱ_RG(λ) dλ

where the integrand is:
    ℱ_RG = A·(∇·T) + B·‖T‖² + C·(∂_ε g) + D·fract(T)

Key innovations from GIFT v1.2b:
- Learnable coefficients A, B, C, D (not hardcoded!)
- L2 penalty to prevent divergence
- Multi-grid evaluation of divergence and fractality
- Clamped epsilon derivative contribution

Reference:
    GIFT/G2_ML/1_2b/ - RG flow with fractal corrections
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class RGFlowModule(nn.Module):
    """
    RG Flow module with learnable coefficients.

    Computes RG running Δα from geometric and topological contributions:
    - A: Torsion divergence coefficient
    - B: Torsion norm coefficient
    - C: Epsilon variation coefficient
    - D: Fractality coefficient

    All coefficients are learnable parameters, initialized to GIFT v1.2b values.
    """

    def __init__(
        self,
        lambda_max: float = 39.44,
        n_steps: int = 100,
        epsilon_0: float = 1.0 / 8.0,
        A_init: float = -20.0,
        B_init: float = 1.0,
        C_init: float = 20.0,
        D_init: float = 3.0,
        l2_penalty: float = 0.001
    ):
        """
        Initialize RG Flow module.

        Args:
            lambda_max: Maximum geodesic parameter (GIFT: 39.44)
            n_steps: Integration steps for ∫ dλ
            epsilon_0: GIFT symmetry breaking scale (1/8)
            A_init: Initial divergence coefficient (v1.2b: -20.0, negative = convergence)
            B_init: Initial norm coefficient (v1.2b: 1.0)
            C_init: Initial epsilon coefficient (v1.2b: 20.0)
            D_init: Initial fractality coefficient (v1.2b: 3.0, reduced from 15)
            l2_penalty: L2 regularization on coefficients

        Note:
            GIFT v1.2b key changes:
            - D: 15.0 → 3.0 (reduced fractal dominance)
            - A: -20.0 (restored correct sign, divergence dominant)
            - L2 penalty added to prevent runaway
        """
        super().__init__()

        self.lambda_max = lambda_max
        self.n_steps = n_steps
        self.epsilon_0 = epsilon_0
        self.l2_penalty = l2_penalty

        # Learnable coefficients
        self.A = nn.Parameter(torch.tensor(A_init, dtype=torch.float64))
        self.B = nn.Parameter(torch.tensor(B_init, dtype=torch.float64))
        self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float64))
        self.D = nn.Parameter(torch.tensor(D_init, dtype=torch.float64))

        # Store initial values for monitoring
        self.register_buffer('A_init', torch.tensor(A_init, dtype=torch.float64))
        self.register_buffer('B_init', torch.tensor(B_init, dtype=torch.float64))
        self.register_buffer('C_init', torch.tensor(C_init, dtype=torch.float64))
        self.register_buffer('D_init', torch.tensor(D_init, dtype=torch.float64))

    def compute_l2_penalty(self) -> torch.Tensor:
        """
        Compute L2 penalty on coefficients.

        Prevents runaway growth during optimization.

        Returns:
            penalty: Scalar L2 penalty
        """
        return self.l2_penalty * (self.A ** 2 + self.B ** 2 + self.C ** 2 + self.D ** 2)

    def forward(
        self,
        div_T_eff: float,
        torsion_norm_sq: float,
        trace_deps: float,
        fract_eff: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute RG flow: Δα = (1/λ_max) ∫ ℱ_RG dλ

        Args:
            div_T_eff: Effective torsion divergence (multi-grid averaged)
            torsion_norm_sq: Squared torsion norm ‖T‖²
            trace_deps: Trace of epsilon derivative ∂_ε g
            fract_eff: Effective fractality index (multi-scale FFT)

        Returns:
            delta_alpha: RG running value
            components: Dict with breakdown:
                - A_divergence: A·(∇·T)
                - B_norm: B·‖T‖²
                - C_epsilon: C·(∂_ε g)
                - D_fractality: D·fract(T)
                - RG_noD: RG without fractal term (A+B+C)
                - total: Full Δα
                - div_T_eff, fract_eff: Input values
                - A, B, C, D: Current coefficient values

        Note:
            From GIFT v1.2b results:
            - Target Δα = -0.9
            - Achieved Δα = -0.87 (3.5% error)
            - A_divergence typically dominates (~-0.13)
            - D_fractality provides correction (~-0.76)
        """
        device = self.A.device
        dtype = self.A.dtype

        # Convert inputs to tensors
        div_T_t = torch.tensor(div_T_eff, device=device, dtype=dtype)
        torsion_norm_sq_t = torch.tensor(torsion_norm_sq, device=device, dtype=dtype)
        trace_deps_t = torch.tensor(trace_deps, device=device, dtype=dtype)
        fract_eff_t = torch.tensor(fract_eff, device=device, dtype=dtype)

        # Component terms
        A_term = self.A * div_T_t
        B_term = self.B * torsion_norm_sq_t

        # Clamp trace_deps to prevent growth (GIFT v1.2b enhancement)
        trace_deps_clamped = torch.clamp(trace_deps_t, -0.05, +0.05)
        C_term = self.C * trace_deps_clamped

        D_term = self.D * fract_eff_t

        # Total integrand ℱ_RG
        integrand = A_term + B_term + C_term + D_term

        # Geodesic integration over λ ∈ [0, λ_max]
        # For simplicity, assume constant integrand (valid for slowly varying ℱ_RG)
        lambdas = torch.linspace(0, self.lambda_max, self.n_steps, device=device, dtype=dtype)

        # Trapezoidal integration
        integral = torch.trapz(integrand * torch.ones_like(lambdas), lambdas)

        # Normalize by λ_max
        delta_alpha = integral / self.lambda_max

        # Component breakdown
        components = {
            'A_divergence': A_term.item(),
            'B_norm': B_term.item(),
            'C_epsilon': C_term.item(),
            'D_fractality': D_term.item(),
            'RG_noD': (A_term + B_term + C_term).item(),  # RG without fractal
            'total': delta_alpha.item(),
            'div_T_eff': div_T_eff,
            'fract_eff': fract_eff,
            'A': self.A.item(),
            'B': self.B.item(),
            'C': self.C.item(),
            'D': self.D.item(),
        }

        return delta_alpha, components

    def get_coefficient_drift(self) -> Dict[str, float]:
        """
        Compute drift of coefficients from initial values.

        Useful for monitoring training dynamics.

        Returns:
            drift: Dict with relative changes
        """
        return {
            'A_drift': (self.A.item() - self.A_init.item()) / abs(self.A_init.item() + 1e-8),
            'B_drift': (self.B.item() - self.B_init.item()) / abs(self.B_init.item() + 1e-8),
            'C_drift': (self.C.item() - self.C_init.item()) / abs(self.C_init.item() + 1e-8),
            'D_drift': (self.D.item() - self.D_init.item()) / abs(self.D_init.item() + 1e-8),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RGFlowModule(\n"
            f"  A={self.A.item():+.4f} (init: {self.A_init.item():+.4f})\n"
            f"  B={self.B.item():+.4f} (init: {self.B_init.item():+.4f})\n"
            f"  C={self.C.item():+.4f} (init: {self.C_init.item():+.4f})\n"
            f"  D={self.D.item():+.4f} (init: {self.D_init.item():+.4f})\n"
            f"  λ_max={self.lambda_max:.2f}, ε₀={self.epsilon_0:.4f}\n"
            f")"
        )


__all__ = ['RGFlowModule']
