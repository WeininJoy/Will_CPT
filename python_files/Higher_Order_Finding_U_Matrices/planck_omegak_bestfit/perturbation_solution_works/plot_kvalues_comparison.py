#!/usr/bin/env python3
"""
Plot comparison of L70_kvalues between allowedK and integerK datasets
"""
import numpy as np
import matplotlib.pyplot as plt

# Load the k-values from both directories
kvalues_allowedK = np.load('./data/data_allowedK/L70_kvalues.npy')
kvalues_integerK = np.load('./data/data_integerK/L70_kvalues.npy')

# DeltaK = np.diff(kvalues_allowedK)
# DeltaK_avg = np.average(DeltaK[-8:])  # average spacing of the last 10 k-values in allowedK
# plt.hlines(np.zeros_like(DeltaK), 0, len(DeltaK)-1, colors='gray', linestyles='dashed', label='Zero Line')
# plt.plot(DeltaK-DeltaK_avg, label='allowedK')
# # plt.yscale('symlog', linthresh=1e-5)
# plt.show()

print(f"allowedK shape: {kvalues_allowedK.shape}")
print(f"integerK shape: {kvalues_integerK.shape}")
print(f"\nallowedK values:\n{kvalues_allowedK}")
print(f"\nintegerK values:\n{kvalues_integerK}")

# Create figure with multiple subplots for comprehensive comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Direct comparison (integerK vs allowedK)
ax1 = axes[0, 0]
# Find the common length for plotting
min_len = min(len(kvalues_allowedK), len(kvalues_integerK))
ax1.scatter(kvalues_allowedK[:min_len], kvalues_integerK[:min_len],
            alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax1.plot([kvalues_allowedK.min(), kvalues_allowedK.max()],
         [kvalues_allowedK.min(), kvalues_allowedK.max()],
         'r--', alpha=0.5, label='y=x')
ax1.set_xlabel('k-values (allowedK)', fontsize=12)
ax1.set_ylabel('k-values (integerK)', fontsize=12)
ax1.set_title('integerK vs allowedK', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Both datasets on the same plot vs index
ax2 = axes[0, 1]
ax2.plot(kvalues_allowedK, 'o-', label='allowedK', markersize=5, alpha=0.7)
ax2.plot(kvalues_integerK, 's-', label='integerK', markersize=5, alpha=0.7)
ax2.set_xlabel('Index', fontsize=12)
ax2.set_ylabel('k-values', fontsize=12)
ax2.set_title('k-values vs Index', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Difference between the two datasets
ax3 = axes[1, 0]
if len(kvalues_allowedK) == len(kvalues_integerK):
    diff = kvalues_integerK - kvalues_allowedK
    ax3.plot(diff, 'o-', markersize=5, color='purple', alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Index', fontsize=12)
    ax3.set_ylabel('Difference (integerK - allowedK)', fontsize=12)
    ax3.set_title('Difference between Datasets', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    print(f"\nDifference statistics:")
    print(f"  Mean: {np.mean(diff):.6e}")
    print(f"  Std: {np.std(diff):.6e}")
    print(f"  Max: {np.max(np.abs(diff)):.6e}")
else:
    ax3.text(0.5, 0.5, f'Different lengths:\nallowedK: {len(kvalues_allowedK)}\nintegerK: {len(kvalues_integerK)}',
             ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.set_title('Difference (N/A - different lengths)', fontsize=13, fontweight='bold')

# Plot 4: Log-log plot if values are positive
ax4 = axes[1, 1]
if np.all(kvalues_allowedK > 0) and np.all(kvalues_integerK > 0):
    ax4.loglog(kvalues_allowedK[:min_len], kvalues_integerK[:min_len],
               'o', alpha=0.6, markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    ax4.plot([kvalues_allowedK.min(), kvalues_allowedK.max()],
             [kvalues_allowedK.min(), kvalues_allowedK.max()],
             'r--', alpha=0.5, label='y=x')
    ax4.set_xlabel('k-values (allowedK)', fontsize=12)
    ax4.set_ylabel('k-values (integerK)', fontsize=12)
    ax4.set_title('Log-Log: integerK vs allowedK', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
else:
    ax4.plot(kvalues_allowedK, 'o-', label='allowedK', markersize=5, alpha=0.7)
    ax4.plot(kvalues_integerK, 's-', label='integerK', markersize=5, alpha=0.7)
    ax4.set_xlabel('Index', fontsize=12)
    ax4.set_ylabel('k-values', fontsize=12)
    ax4.set_title('k-values Comparison', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kvalues_comparison.pdf', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'kvalues_comparison.pdf'")
plt.show()
