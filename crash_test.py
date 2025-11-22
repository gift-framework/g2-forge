#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick crash-test for g2-forge
Tests core functionality without full training
"""

print("=" * 60)
print("G2-FORGE CRASH-TEST 🔨💥")
print("=" * 60)

# Test 1: Import
print("\n[1/7] Testing imports...")
try:
    import g2forge as g2
    print(f"  ✓ g2forge {g2.__version__}")
    print(f"  ✓ Author: {g2.__author__}")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    exit(1)

# Test 2: PyTorch availability
print("\n[2/7] Checking PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  ✓ Device: {device}")
except Exception as e:
    print(f"  ✗ PyTorch error: {e}")
    exit(1)

# Test 3: GIFT config
print("\n[3/7] Testing GIFT v1.0 config...")
try:
    config_gift = g2.G2ForgeConfig.from_gift_v1_0()
    print(f"  ✓ Config created")
    print(f"  ✓ Topology: b2={config_gift.manifold.topology.b2}, b3={config_gift.manifold.topology.b3}")
    assert config_gift.manifold.topology.b2 == 21, "b2 should be 21!"
    assert config_gift.manifold.topology.b3 == 77, "b3 should be 77!"
    print(f"  ✓ GIFT topology validated (21, 77)")
except Exception as e:
    print(f"  ✗ Config failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Custom config
print("\n[4/7] Testing custom config...")
try:
    config_custom = g2.create_k7_config(
        b2_m1=10, b3_m1=38,
        b2_m2=9, b3_m2=35
    )
    print(f"  ✓ Custom config created")
    print(f"  ✓ Topology: b2={config_custom.manifold.topology.b2}, b3={config_custom.manifold.topology.b3}")
    assert config_custom.manifold.topology.b2 == 19, "b2 should be 19!"
    assert config_custom.manifold.topology.b3 == 73, "b3 should be 73!"
    print(f"  ✓ Custom topology validated (19, 73)")
except Exception as e:
    print(f"  ✗ Custom config failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Network auto-sizing
print("\n[5/7] Testing network auto-sizing...")
try:
    # GIFT networks
    h2_gift = g2.networks.create_harmonic_h2_network(config_gift.manifold.topology)
    h3_gift = g2.networks.create_harmonic_h3_network(config_gift.manifold.topology)
    print(f"  ✓ GIFT networks: H2 outputs {h2_gift.n_forms} forms, H3 outputs {h3_gift.n_forms} forms")
    assert h2_gift.n_forms == 21, "H2 should output 21 forms!"
    assert h3_gift.n_forms == 77, "H3 should output 77 forms!"

    # Custom networks
    h2_custom = g2.networks.create_harmonic_h2_network(config_custom.manifold.topology)
    h3_custom = g2.networks.create_harmonic_h3_network(config_custom.manifold.topology)
    print(f"  ✓ Custom networks: H2 outputs {h2_custom.n_forms} forms, H3 outputs {h3_custom.n_forms} forms")
    assert h2_custom.n_forms == 19, "H2 should output 19 forms!"
    assert h3_custom.n_forms == 73, "H3 should output 73 forms!"

    print(f"  ✓ Auto-sizing works! Same code, different topologies! 🎉")
except Exception as e:
    print(f"  ✗ Network auto-sizing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Manifold creation
print("\n[6/7] Testing manifold creation...")
try:
    manifold = g2.manifolds.create_manifold(config_gift.manifold)
    print(f"  ✓ Manifold created: {type(manifold).__name__}")

    # Sample coordinates
    coords = manifold.sample_coordinates(10, device='cpu')
    print(f"  ✓ Sampled coords: shape {coords.shape}")
    assert coords.shape == (10, 7), "Coords should be (10, 7)!"

    # Get regions
    weights = manifold.get_region_weights(coords)
    print(f"  ✓ Region weights: {list(weights.keys())}")
except Exception as e:
    print(f"  ✗ Manifold failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Mini training (5 epochs only!)
print("\n[7/7] Testing mini training (5 epochs)...")
try:
    from g2forge.training import Trainer

    # Use small config for speed
    trainer = Trainer(config_custom, device='cpu', verbose=False)
    print(f"  ✓ Trainer created")

    # Train just 5 epochs
    print(f"  ⏳ Running 5 epochs (be patient)...")
    results = trainer.train(num_epochs=5)

    print(f"  ✓ Training completed!")
    print(f"  ✓ Final loss: {results['final_metrics']['loss']:.4f}")
    print(f"  ✓ Rank H2: {results['final_metrics']['rank_h2']}/19")
    print(f"  ✓ Rank H3: {results['final_metrics']['rank_h3']}/73")

except Exception as e:
    print(f"  ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ CRASH-TEST PASSED! 🎉")
print("=" * 60)
print("\ng2-forge is ready to forge some G2 metrics! 🔨✨")
