"""
Configuration system for g2-forge.

Provides universal configuration schema for G₂ metric construction
that works for any manifold, not just GIFT-specific parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import yaml


@dataclass
class TopologyConfig:
    """
    Topological invariants of a G₂ manifold.

    For a compact G₂ manifold M, the Betti numbers satisfy:
    - b₀ = 1 (connected)
    - b₁ = 0 (simply connected, typically)
    - b₂ = dim H²(M) (harmonic 2-forms)
    - b₃ = dim H³(M) (harmonic 3-forms)
    - b₄ = b₃ (Poincaré duality)
    - b₅ = b₂ (Poincaré duality)
    - b₆ = b₁ = 0
    - b₇ = b₀ = 1

    Euler characteristic: χ = 2(b₀ - b₁ + b₂ - b₃)
    """
    b2: int
    b3: int
    b1: int = 0
    b0: int = 1

    @property
    def b4(self) -> int:
        """b₄ = b₃ by Poincaré duality."""
        return self.b3

    @property
    def b5(self) -> int:
        """b₅ = b₂ by Poincaré duality."""
        return self.b2

    @property
    def b6(self) -> int:
        """b₆ = b₁ by Poincaré duality."""
        return self.b1

    @property
    def b7(self) -> int:
        """b₇ = b₀ by Poincaré duality."""
        return self.b0

    @property
    def euler_characteristic(self) -> int:
        """Euler characteristic χ = Σ(-1)ⁱ bᵢ = 2(b₀ - b₁ + b₂ - b₃)."""
        return 2 * (self.b0 - self.b1 + self.b2 - self.b3)

    def validate(self) -> bool:
        """
        Validate topological consistency.

        Returns:
            True if topology is valid

        Raises:
            ValueError if topology is inconsistent
        """
        if self.b2 < 0 or self.b3 < 0:
            raise ValueError(f"Betti numbers must be non-negative: b₂={self.b2}, b₃={self.b3}")

        if self.b1 != 0:
            raise ValueError(
                f"G₂ manifolds are typically simply connected (b₁=0), got b₁={self.b1}"
            )

        return True


@dataclass
class TCSParameters:
    """
    Parameters for Twisted Connected Sum (TCS) construction.

    K₇ = M₁ᵀ ∪_φ M₂ᵀ

    where M₁ᵀ and M₂ᵀ are asymptotically cylindrical (ACyl) manifolds
    glued along a neck region via matching map φ.
    """
    # Topology of M₁
    b2_m1: int
    b3_m1: int

    # Topology of M₂
    b2_m2: int
    b3_m2: int

    # Neck geometry
    neck_width: float = 0.125  # ε₀ in GIFT
    neck_center: float = 0.5   # t-coordinate of neck center

    # Transition sharpness
    transition_sharpness: float = 10.0

    def validate(self) -> bool:
        """
        Validate TCS parameters.

        For TCS, total topology satisfies:
        - b₂(K₇) = b₂(M₁) + b₂(M₂)
        - b₃(K₇) = b₃(M₁) + b₃(M₂)
        """
        if self.b2_m1 < 0 or self.b2_m2 < 0:
            raise ValueError(f"M₁/M₂ Betti numbers must be non-negative")

        if self.neck_width <= 0 or self.neck_width >= 1:
            raise ValueError(f"Neck width must be in (0, 1), got {self.neck_width}")

        return True

    @property
    def total_b2(self) -> int:
        """Total b₂ = b₂(M₁) + b₂(M₂)."""
        return self.b2_m1 + self.b2_m2

    @property
    def total_b3(self) -> int:
        """Total b₃ = b₃(M₁) + b₃(M₂)."""
        return self.b3_m1 + self.b3_m2


@dataclass
class ModuliParameters:
    """
    Moduli space parameters for specific G₂ manifolds.

    These are manifold-dependent geometric parameters that
    specify a particular point in the moduli space.
    """
    params: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.params[key]

    def __setitem__(self, key: str, value: float):
        self.params[key] = value

    def get(self, key: str, default: float = 0.0) -> float:
        return self.params.get(key, default)


@dataclass
class ManifoldConfig:
    """
    Complete manifold specification.

    Supports different construction methods:
    - "TCS": Twisted Connected Sum (Kovalev construction)
    - "Joyce": Joyce's original construction
    - "ConnectedSum": Standard connected sum
    - "Quotient": Quotient by discrete group
    - "Custom": User-defined
    """
    # Manifold type
    type: str  # e.g., "K7", "Joyce", "Custom"

    # Construction method
    construction: str  # e.g., "TCS", "Joyce", "ConnectedSum"

    # Topology
    topology: TopologyConfig

    # Optional: TCS-specific parameters
    tcs_params: Optional[TCSParameters] = None

    # Optional: Moduli space parameters
    moduli: Optional[ModuliParameters] = None

    # Dimension (always 7 for G₂)
    dimension: int = 7

    def validate(self) -> bool:
        """Validate manifold configuration consistency."""
        if self.dimension != 7:
            raise ValueError(f"G₂ manifolds must be 7-dimensional, got {self.dimension}")

        self.topology.validate()

        if self.construction == "TCS":
            if self.tcs_params is None:
                raise ValueError("TCS construction requires tcs_params")
            self.tcs_params.validate()

            # Check topology consistency
            if self.topology.b2 != self.tcs_params.total_b2:
                raise ValueError(
                    f"TCS topology mismatch: b₂={self.topology.b2} != "
                    f"b₂(M₁)+b₂(M₂)={self.tcs_params.total_b2}"
                )
            if self.topology.b3 != self.tcs_params.total_b3:
                raise ValueError(
                    f"TCS topology mismatch: b₃={self.topology.b3} != "
                    f"b₃(M₁)+b₃(M₂)={self.tcs_params.total_b3}"
                )

        return True


@dataclass
class NetworkArchitectureConfig:
    """Neural network architecture configuration."""

    # Phi network (3-form generator)
    phi_hidden_dims: List[int] = field(default_factory=lambda: [384, 384, 256])
    phi_n_fourier: int = 32
    phi_activation: str = "silu"

    # Harmonic 2-forms network
    h2_hidden_dim: int = 128
    h2_n_fourier: int = 24

    # Harmonic 3-forms network
    h3_hidden_dim: int = 128
    h3_n_fourier: int = 24

    # Auto-compute output dimensions from topology
    def get_h2_output_dim(self, topology: TopologyConfig) -> int:
        """Output dimension for H² network = b₂."""
        return topology.b2

    def get_h3_output_dim(self, topology: TopologyConfig) -> int:
        """Output dimension for H³network = b₃."""
        return topology.b3


@dataclass
class CurriculumPhaseConfig:
    """Configuration for a single curriculum phase."""
    epoch_range: Tuple[int, int]
    grid_n: int
    loss_weights: Dict[str, float]
    region_weights: Optional[Dict[str, float]] = None


@dataclass
class TrainingConfig:
    """Training hyperparameters and schedule."""

    # Optimization
    total_epochs: int = 15000
    batch_size: int = 2048
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    grad_accumulation: int = 4

    # Learning rate schedule
    warmup_epochs: int = 500
    lr_min: float = 1e-7

    # Mixed-precision training (AMP)
    use_amp: bool = True  # Enable automatic mixed precision (2x speedup on GPU)

    # Subsampling for efficiency
    subsample_coclosure: int = 8
    subsample_harmonic: int = 16

    # Calibration
    calibration_interval: int = 50

    # Curriculum learning
    curriculum: Dict[str, CurriculumPhaseConfig] = field(default_factory=dict)

    def add_phase(self, name: str, phase: CurriculumPhaseConfig):
        """Add a curriculum phase."""
        self.curriculum[name] = phase

    def get_phase_at_epoch(self, epoch: int) -> Optional[Tuple[str, CurriculumPhaseConfig]]:
        """Get curriculum phase for given epoch."""
        for name, phase in self.curriculum.items():
            if phase.epoch_range[0] <= epoch < phase.epoch_range[1]:
                return name, phase
        return None


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    interval: int = 500
    keep_best: int = 5
    auto_resume: bool = True
    save_dir: str = "checkpoints"


@dataclass
class ValidationConfig:
    """Validation and testing configuration."""
    interval: int = 100
    ricci_interval: int = 500
    ricci_n_points: int = 1000

    # Holonomy test
    holonomy_n_loops: int = 10
    holonomy_n_steps: int = 50
    holonomy_tolerance: float = 1e-4


@dataclass
class YukawaConfig:
    """Yukawa coupling computation configuration."""
    n_mc_samples: int = 20000
    grid_n: int = 10
    tucker_rank: Tuple[int, int, int] = (3, 3, 3)
    antisymmetry_tolerance: float = 1e-6


@dataclass
class G2ForgeConfig:
    """
    Complete configuration for g2-forge.

    This is the top-level configuration object that contains
    all parameters needed for G₂ metric construction.
    """
    # Manifold specification
    manifold: ManifoldConfig

    # Neural network architecture
    architecture: NetworkArchitectureConfig = field(default_factory=NetworkArchitectureConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Checkpointing
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Validation
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Yukawa computation
    yukawa: YukawaConfig = field(default_factory=YukawaConfig)

    # Random seed
    seed: int = 42

    # Version tracking
    version: str = "g2forge-v0.1"

    def validate(self) -> bool:
        """Validate complete configuration."""
        self.manifold.validate()
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path: Path) -> None:
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'G2ForgeConfig':
        """Load from dictionary."""
        # Reconstruct nested dataclasses
        topology = TopologyConfig(**data['manifold']['topology'])

        tcs_params = None
        if data['manifold'].get('tcs_params'):
            tcs_params = TCSParameters(**data['manifold']['tcs_params'])

        moduli = None
        if data['manifold'].get('moduli'):
            moduli = ModuliParameters(params=data['manifold']['moduli'].get('params', {}))

        manifold = ManifoldConfig(
            type=data['manifold']['type'],
            construction=data['manifold']['construction'],
            topology=topology,
            tcs_params=tcs_params,
            moduli=moduli,
            dimension=data['manifold'].get('dimension', 7)
        )

        architecture = NetworkArchitectureConfig(**data.get('architecture', {}))

        training_data = data.get('training', {})
        curriculum_data = training_data.pop('curriculum', {})
        training = TrainingConfig(**training_data)

        # Reconstruct curriculum phases
        for phase_name, phase_data in curriculum_data.items():
            phase = CurriculumPhaseConfig(**phase_data)
            training.add_phase(phase_name, phase)

        checkpointing = CheckpointConfig(**data.get('checkpointing', {}))
        validation = ValidationConfig(**data.get('validation', {}))
        yukawa = YukawaConfig(**data.get('yukawa', {}))

        return cls(
            manifold=manifold,
            architecture=architecture,
            training=training,
            checkpointing=checkpointing,
            validation=validation,
            yukawa=yukawa,
            seed=data.get('seed', 42),
            version=data.get('version', 'g2forge-v0.1')
        )

    @classmethod
    def from_json(cls, path: Path) -> 'G2ForgeConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: Path) -> 'G2ForgeConfig':
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_gift_v1_0(cls) -> 'G2ForgeConfig':
        """
        Create config that reproduces GIFT v1.0 exactly.

        This is for validation - we should get identical results
        to GIFT when using this configuration.
        """
        # GIFT K₇ topology
        topology = TopologyConfig(b2=21, b3=77)

        # GIFT TCS parameters
        tcs_params = TCSParameters(
            b2_m1=11, b3_m1=40,
            b2_m2=10, b3_m2=37,
            neck_width=0.125
        )

        # GIFT moduli
        moduli = ModuliParameters(params={
            'tau': 3.8967452300785634,
            'xi': 0.9817477042468103,
            'epsilon0': 0.125
        })

        manifold = ManifoldConfig(
            type="K7",
            construction="TCS",
            topology=topology,
            tcs_params=tcs_params,
            moduli=moduli
        )

        architecture = NetworkArchitectureConfig(
            phi_hidden_dims=[384, 384, 256],
            phi_n_fourier=32,
            h2_hidden_dim=128,
            h2_n_fourier=24,
            h3_hidden_dim=128,
            h3_n_fourier=24
        )

        training = TrainingConfig(
            total_epochs=15000,
            batch_size=2048,
            lr=1e-4,
            weight_decay=1e-4,
            grad_clip=1.0,
            grad_accumulation=4,
            warmup_epochs=500
        )

        # Add GIFT curriculum phases
        training.add_phase("phase1_neck_stability", CurriculumPhaseConfig(
            epoch_range=(0, 2000),
            grid_n=8,
            loss_weights={
                'torsion_closure': 0.5,
                'torsion_coclosure': 0.5,
                'volume': 2.0,
                'gram_h2': 1.0,
                'gram_h3': 0.5,
                'boundary': 0.5,
                'neck_smoothness': 0.1,
                'harmonic_penalty': 0.01,
                'calibration': 0.0
            }
        ))

        training.add_phase("phase2_acyl_matching", CurriculumPhaseConfig(
            epoch_range=(2000, 5000),
            grid_n=8,
            loss_weights={
                'torsion_closure': 1.0,
                'torsion_coclosure': 1.0,
                'volume': 0.5,
                'gram_h2': 1.5,
                'gram_h3': 1.0,
                'boundary': 1.5,
                'neck_smoothness': 0.2,
                'harmonic_penalty': 0.02,
                'calibration': 0.001
            }
        ))

        training.add_phase("phase3_cohomology_refinement", CurriculumPhaseConfig(
            epoch_range=(5000, 8000),
            grid_n=10,
            loss_weights={
                'torsion_closure': 2.0,
                'torsion_coclosure': 2.0,
                'volume': 0.2,
                'gram_h2': 3.0,
                'gram_h3': 2.0,
                'boundary': 2.0,
                'neck_smoothness': 0.3,
                'harmonic_penalty': 0.05,
                'calibration': 0.002
            }
        ))

        training.add_phase("phase4_harmonic_extraction", CurriculumPhaseConfig(
            epoch_range=(8000, 10000),
            grid_n=10,
            loss_weights={
                'torsion_closure': 3.0,
                'torsion_coclosure': 3.0,
                'volume': 0.1,
                'gram_h2': 5.0,
                'gram_h3': 3.0,
                'boundary': 1.5,
                'neck_smoothness': 0.2,
                'harmonic_penalty': 0.1,
                'calibration': 0.005
            }
        ))

        training.add_phase("phase5_calibration_finetune", CurriculumPhaseConfig(
            epoch_range=(10000, 15000),
            grid_n=12,
            loss_weights={
                'torsion_closure': 5.0,
                'torsion_coclosure': 5.0,
                'volume': 0.05,
                'gram_h2': 5.0,
                'gram_h3': 4.0,
                'boundary': 1.0,
                'neck_smoothness': 0.1,
                'harmonic_penalty': 0.2,
                'calibration': 0.01
            }
        ))

        return cls(
            manifold=manifold,
            architecture=architecture,
            training=training,
            seed=42,
            version="gift-v1.0-reproduction"
        )

    @classmethod
    def from_gift_v1_2b(cls) -> 'G2ForgeConfig':
        """
        Create configuration matching GIFT v1.2b with enhanced curriculum.

        GIFT v1.2b improvements over v1.0:
        - More aggressive torsion targeting (0.5 → 2.5 progressive)
        - ACyl strict loss for radial derivative penalization
        - Volume normalization at end of phase 2
        - RG flow calibration in phases 3-5
        - Stronger multi-phase targeting

        Returns:
            G2ForgeConfig matching GIFT v1.2b setup

        Reference:
            GIFT/G2_ML/1_2b/K7_G2_TCS_GIFT_Full_v1_2b.ipynb
        """
        # Same manifold as v1.0
        topology = TopologyConfig(b2=21, b3=77)

        tcs_params = TCSParameters(
            b2_m1=11, b3_m1=40,
            b2_m2=10, b3_m2=37,
            neck_width=0.125,
            neck_center=0.5,
            transition_sharpness=10.0
        )

        moduli = ModuliParameters(params={
            'K7_r_acyl_cutoff': 10.0,
            'twist_angle': 1.047  # π/3
        })

        manifold = ManifoldConfig(
            type="K7",
            construction="TCS",
            topology=topology,
            tcs_params=tcs_params,
            moduli=moduli
        )

        # Same architecture as v1.0
        architecture = NetworkArchitectureConfig(
            phi_hidden_dims=[384, 384, 256],
            phi_n_fourier=32,
            h2_hidden_dim=128,
            h2_n_fourier=24,
            h3_hidden_dim=128,
            h3_n_fourier=24
        )

        training = TrainingConfig(
            total_epochs=10000,  # v1.2b: 5 phases × 2000 epochs
            batch_size=1024,     # Smaller for stability
            lr=1e-4,
            weight_decay=1e-4,
            grad_clip=1.0,
            grad_accumulation=4,
            warmup_epochs=200
        )

        # GIFT v1.2b curriculum with enhanced weights
        # Phase 1: TCS Neck - Initial stabilization
        training.add_phase("phase1_tcs_neck", CurriculumPhaseConfig(
            epoch_range=(0, 2000),
            grid_n=16,
            loss_weights={
                'torsion_closure': 0.5,      # Reduced initial
                'torsion_coclosure': 0.5,
                'volume': 0.5,
                'gram_h2': 1.0,
                'gram_h3': 1.0,
                'boundary': 2.0,             # Focus on neck
                'acyl_strict': 0.0,          # Not yet
                'calibration': 0.0
            }
        ))

        # Phase 2: ACyl Matching - ACyl strict behavior
        training.add_phase("phase2_acyl_matching", CurriculumPhaseConfig(
            epoch_range=(2000, 4000),
            grid_n=16,
            loss_weights={
                'torsion_closure': 0.5,
                'torsion_coclosure': 0.5,
                'volume': 0.8,
                'gram_h2': 1.5,
                'gram_h3': 1.0,
                'boundary': 0.5,
                'acyl_strict': 0.5,          # Enable ACyl strict
                'calibration': 0.0
            }
        ))

        # Phase 3: Cohomology Refinement - Increase torsion weight
        training.add_phase("phase3_cohomology", CurriculumPhaseConfig(
            epoch_range=(4000, 6000),
            grid_n=8,  # Coarse for harmonics
            loss_weights={
                'torsion_closure': 1.5,      # Increased
                'torsion_coclosure': 1.5,
                'volume': 0.5,
                'gram_h2': 1.0,
                'gram_h3': 1.0,
                'boundary': 0.5,
                'acyl_strict': 1.0,          # Strengthened
                'calibration': 0.2           # Begin RG flow
            }
        ))

        # Phase 4: Harmonic Extraction - High harmonicity
        training.add_phase("phase4_harmonic_extraction", CurriculumPhaseConfig(
            epoch_range=(6000, 8000),
            grid_n=8,
            loss_weights={
                'torsion_closure': 2.5,      # Much stronger
                'torsion_coclosure': 2.5,
                'volume': 1.0,
                'gram_h2': 3.0,              # High harmonicity
                'gram_h3': 3.0,
                'boundary': 0.2,
                'acyl_strict': 1.0,
                'calibration': 0.5           # Increase RG flow
            }
        ))

        # Phase 5: RG Calibration - Final polish
        training.add_phase("phase5_rg_calibration", CurriculumPhaseConfig(
            epoch_range=(8000, 10000),
            grid_n=8,
            loss_weights={
                'torsion_closure': 2.5,      # Maintain high
                'torsion_coclosure': 2.5,
                'volume': 2.0,               # Strong volume control
                'gram_h2': 1.0,              # Reduce to avoid overfitting
                'gram_h3': 1.0,
                'boundary': 0.1,
                'acyl_strict': 1.0,
                'calibration': 3.0           # Maximize RG flow
            }
        ))

        return cls(
            manifold=manifold,
            architecture=architecture,
            training=training,
            seed=42,
            version="gift-v1.2b-enhanced"
        )


# Convenience function for quick config creation
def create_k7_config(
    b2_m1: int = 11,
    b3_m1: int = 40,
    b2_m2: int = 10,
    b3_m2: int = 37,
    **kwargs
) -> G2ForgeConfig:
    """
    Quick K₇ configuration with sensible defaults.

    Args:
        b2_m1, b3_m1: M₁ topology
        b2_m2, b3_m2: M₂ topology
        **kwargs: Override any default parameters

    Returns:
        G2ForgeConfig ready for training
    """
    topology = TopologyConfig(b2=b2_m1 + b2_m2, b3=b3_m1 + b3_m2)

    tcs_params = TCSParameters(
        b2_m1=b2_m1, b3_m1=b3_m1,
        b2_m2=b2_m2, b3_m2=b3_m2
    )

    manifold = ManifoldConfig(
        type="K7",
        construction="TCS",
        topology=topology,
        tcs_params=tcs_params
    )

    config = G2ForgeConfig(manifold=manifold)

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
