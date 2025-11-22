"""
GIFT v1.0 Reproduction Configuration

This configuration exactly reproduces GIFT v1.0 results.
Use this to validate that g2-forge produces identical
performance to the original GIFT implementation.

Expected results:
- Torsion closure: < 1e-3
- Rank H²: 21/21
- Rank H³: 77/77
- Training time: ~6-8 hours on A100
"""

from g2forge.utils import G2ForgeConfig

# Load GIFT v1.0 configuration
config = G2ForgeConfig.from_gift_v1_0()

# Display configuration
print("="*60)
print("GIFT v1.0 Reproduction Configuration")
print("="*60)
print(f"\nManifold: {config.manifold.type}")
print(f"Construction: {config.manifold.construction}")
print(f"Topology: b₂={config.manifold.topology.b2}, b₃={config.manifold.topology.b3}")

if config.manifold.tcs_params:
    print(f"\nTCS Structure:")
    print(f"  M₁: b₂={config.manifold.tcs_params.b2_m1}, b₃={config.manifold.tcs_params.b3_m1}")
    print(f"  M₂: b₂={config.manifold.tcs_params.b2_m2}, b₃={config.manifold.tcs_params.b3_m2}")
    print(f"  Neck width: {config.manifold.tcs_params.neck_width}")

print(f"\nArchitecture:")
print(f"  Phi network: {config.architecture.phi_hidden_dims}")
print(f"  H² output dim: {config.architecture.get_h2_output_dim(config.manifold.topology)}")
print(f"  H³ output dim: {config.architecture.get_h3_output_dim(config.manifold.topology)}")

print(f"\nTraining:")
print(f"  Total epochs: {config.training.total_epochs}")
print(f"  Batch size: {config.training.batch_size}")
print(f"  Learning rate: {config.training.lr}")
print(f"  Phases: {len(config.training.curriculum)}")

print(f"\nCurriculum Phases:")
for phase_name, phase in config.training.curriculum.items():
    print(f"  {phase_name}:")
    print(f"    Epochs: {phase.epoch_range[0]}-{phase.epoch_range[1]}")
    print(f"    Grid: {phase.grid_n}³")
    print(f"    Torsion weight: {phase.loss_weights.get('torsion_closure', 0)}")

print("\n" + "="*60)
print("Ready for training!")
print("="*60)

# Save to JSON for inspection
config.to_json("k7_gift_v1_0.json")
print("\nConfiguration saved to: k7_gift_v1_0.json")
