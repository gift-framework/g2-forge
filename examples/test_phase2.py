"""
Phase 2 Validation Test

Quick test to verify that Phase 2 implementation works:
1. Configuration system
2. Differential operators
3. Manifold abstraction
4. K₇ implementation
"""

import sys
sys.path.insert(0, '/home/user/g2-forge')

print("="*60)
print("Phase 2 Validation Test")
print("="*60)

# Test 1: Import g2forge
print("\n[1/5] Testing imports...")
try:
    import g2forge as g2
    print("✓ g2forge imported successfully")
    print(f"   Version: {g2.__version__}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Configuration system
print("\n[2/5] Testing configuration system...")
try:
    # GIFT reproduction
    config_gift = g2.G2ForgeConfig.from_gift_v1_0()
    print(f"✓ GIFT config created: b₂={config_gift.manifold.topology.b2}, b₃={config_gift.manifold.topology.b3}")

    # Custom config
    config_custom = g2.create_k7_config(b2_m1=10, b3_m1=38, b2_m2=9, b3_m2=35)
    print(f"✓ Custom config created: b₂={config_custom.manifold.topology.b2}, b₃={config_custom.manifold.topology.b3}")

    # Validation
    config_gift.validate()
    config_custom.validate()
    print("✓ Configurations validated")

except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Manifold creation
print("\n[3/5] Testing manifold creation...")
try:
    # GIFT K₇
    k7_gift = g2.create_gift_k7()
    print(f"✓ GIFT K₇ created:")
    print(f"   {k7_gift}")

    # Custom K₇
    k7_custom = g2.create_custom_k7(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)
    print(f"✓ Custom K₇ created:")
    print(f"   {k7_custom}")

except Exception as e:
    print(f"✗ Manifold creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Coordinate sampling
print("\n[4/5] Testing coordinate sampling...")
try:
    import torch

    # Sample coordinates
    coords = k7_custom.sample_coordinates(n_samples=100, device='cpu')
    print(f"✓ Sampled {coords.shape[0]} coordinates, shape={coords.shape}")

    # Check ranges
    t_min, t_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    theta_min, theta_max = coords[:, 1:].min().item(), coords[:, 1:].max().item()
    print(f"   t ∈ [{t_min:.3f}, {t_max:.3f}]")
    print(f"   θ ∈ [{theta_min:.3f}, {theta_max:.3f}]")

    # Region weights
    weights = k7_custom.get_region_weights(coords)
    print(f"✓ Region weights computed:")
    print(f"   M₁: mean={weights['m1'].mean():.3f}")
    print(f"   Neck: mean={weights['neck'].mean():.3f}")
    print(f"   M₂: mean={weights['m2'].mean():.3f}")

except Exception as e:
    print(f"✗ Coordinate sampling failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Differential operators
print("\n[5/5] Testing differential operators...")
try:
    # Build Levi-Civita
    eps_indices, eps_signs = g2.build_levi_civita_sparse_7d()
    print(f"✓ Levi-Civita built: {eps_indices.shape[0]} permutations")

    # Test with small tensor
    batch_size = 10
    phi = torch.randn(batch_size, 7, 7, 7)
    metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

    # Exterior derivative
    coords.requires_grad_(True)
    phi_test = phi[:batch_size]
    dphi = g2.compute_exterior_derivative(phi_test, coords[:batch_size])
    print(f"✓ Exterior derivative computed: shape={dphi.shape}")

    # Hodge star (small batch for speed)
    star_phi = g2.hodge_star_3(phi[:5], metric[:5], eps_indices, eps_signs)
    print(f"✓ Hodge star computed: shape={star_phi.shape}")

except Exception as e:
    print(f"✗ Operator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✨ Phase 2 Validation: ALL TESTS PASSED! ✨")
print("="*60)
print("\nReady for Phase 3:")
print("  - Loss functions (parameterized)")
print("  - Neural network architectures")
print("  - Training infrastructure")
print("\n" + "="*60)
