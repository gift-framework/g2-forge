"""
Custom K₇ Topology Example

This demonstrates how to create a K₇ manifold with
arbitrary topology parameters, not limited to GIFT's
specific b₂=21, b₃=77.

This is the whole point of g2-forge: universal framework!
"""

from g2forge.utils import create_k7_config

# Example 1: Different Betti numbers
print("="*60)
print("Example 1: K₇ with b₂=19, b₃=73")
print("="*60)

config1 = create_k7_config(
    b2_m1=10,  # Different from GIFT!
    b3_m1=38,
    b2_m2=9,
    b3_m2=35
)

print(f"Topology: b₂={config1.manifold.topology.b2}, b₃={config1.manifold.topology.b3}")
print(f"H² network output: {config1.architecture.get_h2_output_dim(config1.manifold.topology)}")
print(f"H³ network output: {config1.architecture.get_h3_output_dim(config1.manifold.topology)}")
print(f"Euler characteristic: χ={config1.manifold.topology.euler_characteristic}")

# Example 2: Larger topology
print("\n" + "="*60)
print("Example 2: K₇ with b₂=30, b₃=100")
print("="*60)

config2 = create_k7_config(
    b2_m1=15,
    b3_m1=50,
    b2_m2=15,
    b3_m2=50
)

print(f"Topology: b₂={config2.manifold.topology.b2}, b₃={config2.manifold.topology.b3}")
print(f"H² network output: {config2.architecture.get_h2_output_dim(config2.manifold.topology)}")
print(f"H³ network output: {config2.architecture.get_h3_output_dim(config2.manifold.topology)}")
print(f"Training epochs: {config2.training.total_epochs}")

# Example 3: Minimal topology
print("\n" + "="*60)
print("Example 3: K₇ with b₂=5, b₃=20 (minimal)")
print("="*60)

config3 = create_k7_config(
    b2_m1=3,
    b3_m1=10,
    b2_m2=2,
    b3_m2=10
)

print(f"Topology: b₂={config3.manifold.topology.b2}, b₃={config3.manifold.topology.b3}")
print(f"TCS structure validated: {config3.manifold.validate()}")

# Save examples
config1.to_json("k7_b2_19_b3_73.json")
config2.to_json("k7_b2_30_b3_100.json")
config3.to_json("k7_b2_5_b3_20.json")

print("\n" + "="*60)
print("✨ See? g2-forge works for ANY topology!")
print("="*60)
print("\nNot limited to GIFT's b₂=21, b₃=77 anymore! 🚀")
