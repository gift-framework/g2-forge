"""
Test Networks: Verify auto-sizing from topology

Tests that neural networks automatically size themselves
based on topology configuration (b₂, b₃).

This is THE key feature - no hardcoded 21, 77!
"""

import sys
sys.path.insert(0, '/home/user/g2-forge')

import torch
import g2forge as g2

print("="*60)
print("Network Auto-Sizing Test")
print("="*60)

# Test 1: GIFT topology (validation)
print("\n[1/3] GIFT topology (b₂=21, b₃=77)...")
config_gift = g2.G2ForgeConfig.from_gift_v1_0()

# Phi network
phi_net_gift = g2.networks.create_phi_network_from_config(config_gift)
print(f"✓ Phi network: {phi_net_gift.count_parameters():,} parameters")

# Harmonic networks - AUTO-SIZED!
h2_net_gift, h3_net_gift = g2.networks.create_harmonic_networks_from_config(config_gift)
print(f"✓ H² network: n_forms={h2_net_gift.n_forms} (should be 21)")
print(f"✓ H³ network: n_forms={h3_net_gift.n_forms} (should be 77)")

assert h2_net_gift.n_forms == 21, "H² should have 21 forms for GIFT"
assert h3_net_gift.n_forms == 77, "H³ should have 77 forms for GIFT"

# Test 2: Custom small topology
print("\n[2/3] Custom small topology (b₂=10, b₃=40)...")
config_small = g2.create_k7_config(b2_m1=5, b3_m1=20, b2_m2=5, b3_m2=20)

h2_net_small, h3_net_small = g2.networks.create_harmonic_networks_from_config(config_small)
print(f"✓ H² network: n_forms={h2_net_small.n_forms} (should be 10)")
print(f"✓ H³ network: n_forms={h3_net_small.n_forms} (should be 40)")

assert h2_net_small.n_forms == 10, "H² should have 10 forms"
assert h3_net_small.n_forms == 40, "H³ should have 40 forms"

# Test 3: Custom large topology
print("\n[3/3] Custom large topology (b₂=30, b₃=100)...")
config_large = g2.create_k7_config(b2_m1=15, b3_m1=50, b2_m2=15, b3_m2=50)

h2_net_large, h3_net_large = g2.networks.create_harmonic_networks_from_config(config_large)
print(f"✓ H² network: n_forms={h2_net_large.n_forms} (should be 30)")
print(f"✓ H³ network: n_forms={h3_net_large.n_forms} (should be 100)")

assert h2_net_large.n_forms == 30, "H² should have 30 forms"
assert h3_net_large.n_forms == 100, "H³ should have 100 forms"

# Test 4: Forward pass
print("\n[4/4] Forward pass test...")
coords = torch.randn(10, 7)  # Batch of 10 coordinates

# Phi network
phi_components = phi_net_gift(coords)
print(f"✓ Phi output: {phi_components.shape} (should be [10, 35])")
assert phi_components.shape == (10, 35)

phi_tensor = phi_net_gift.get_phi_tensor(coords)
print(f"✓ Phi tensor: {phi_tensor.shape} (should be [10, 7, 7, 7])")
assert phi_tensor.shape == (10, 7, 7, 7)

# H² network (GIFT)
h2_forms = h2_net_gift(coords)
print(f"✓ H² forms: {h2_forms.shape} (should be [10, 21, 21])")
assert h2_forms.shape == (10, 21, 21)  # [batch, n_forms=21, n_components=21]

# H² network (small - different size!)
h2_forms_small = h2_net_small(coords)
print(f"✓ H² forms (small): {h2_forms_small.shape} (should be [10, 10, 21])")
assert h2_forms_small.shape == (10, 10, 21)  # [batch, n_forms=10, n_components=21]

# H³ network (GIFT)
h3_forms = h3_net_gift(coords)
print(f"✓ H³ forms: {h3_forms.shape} (should be [10, 77, 35])")
assert h3_forms.shape == (10, 77, 35)  # [batch, n_forms=77, n_components=35]

# H³ network (large - different size!)
h3_forms_large = h3_net_large(coords)
print(f"✓ H³ forms (large): {h3_forms_large.shape} (should be [10, 100, 35])")
assert h3_forms_large.shape == (10, 100, 35)  # [batch, n_forms=100, n_components=35]

# Summary
print("\n" + "="*60)
print("✨ All Networks Auto-Size Correctly! ✨")
print("="*60)
print("\nKey Achievement:")
print("  Networks adapt to ANY topology (b₂, b₃)")
print("  Not hardcoded to GIFT's (21, 77)")
print("\nTested:")
print(f"  ✓ GIFT: b₂=21, b₃=77")
print(f"  ✓ Small: b₂=10, b₃=40")
print(f"  ✓ Large: b₂=30, b₃=100")
print("\nAll forward passes work! 🎉")
print("="*60)
