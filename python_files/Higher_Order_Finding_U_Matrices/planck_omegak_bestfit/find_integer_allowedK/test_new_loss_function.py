#!/usr/bin/env python3
"""
Test the new loss function to visualize its behavior.
This script loads existing allowedK data and compares old vs new loss calculation.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, '../perturbation_solution_works/')

def calculate_integer_loss_old(allowedK_integer, n_ignore=8):
    """
    OLD loss function (for comparison).
    """

    high_k = np.array(allowedK_integer[n_ignore:])
    indices = np.arange(len(high_k))

    slope, intercept = np.polyfit(indices, high_k, 1)
    ideal_spacing = np.round(slope)

    if ideal_spacing == 0:
        return float('inf')

    ideal_intercept = np.mean(high_k - ideal_spacing * indices)
    ideal_k_sequence = ideal_spacing * indices + np.round(ideal_intercept)

    squared_errors = (high_k - ideal_k_sequence)**2

    # With weighting
    weights = np.linspace(0.1, 1.0, len(high_k))
    loss = np.mean(weights * squared_errors)

    return loss

def calculate_integer_loss_new(allowedK_integer, n_tail=7, slope_weight=2.0):
    """
    NEW loss function (focusing on tail).
    """

    all_k = np.array(allowedK_integer)

    # Step 1: Determine ideal spacing from high-k region (after n_ignore)
    tail_k = all_k[-n_tail:]
    tail_indices = np.arange(len(tail_k))

    slope, _ = np.polyfit(tail_indices, tail_k, 1)
    ideal_spacing = np.round(slope)

    if ideal_spacing == 0:
        return float('inf')

    # Compute ideal sequence for the tail
    ideal_intercept_tail = np.mean(tail_k - ideal_spacing * tail_indices)
    ideal_tail_sequence = ideal_spacing * tail_indices + np.round(ideal_intercept_tail)

    # Step 3: Compute deviations from ideal
    deviations = tail_k - ideal_tail_sequence

    # Step 4: Loss components
    mse_loss = np.mean(deviations**2)
    deviation_slope, _ = np.polyfit(tail_indices, deviations, 1)
    slope_loss = deviation_slope**2

    # Step 5: Combined loss
    total_loss = mse_loss + slope_weight * slope_loss

    return total_loss, deviations, deviation_slope, mse_loss, slope_loss

# Load data
data_path = '../perturbation_solution_works/data_test/data_all_k/'
allowedK_file = data_path + 'allowedK_integer.npy'

if not os.path.exists(allowedK_file):
    print(f"Error: {allowedK_file} not found!")
    print("Please run generate_data.py first.")
    sys.exit(1)

allowedK_integer = np.load(allowedK_file)
print(f"Loaded {len(allowedK_integer)} allowedK values")
print(f"allowedK_integer = {allowedK_integer}")

# Test different n_tail values
n_tail_values = [8, 10, 12, 15]
n_ignore = 8

print("\n" + "="*80)
print("COMPARING OLD vs NEW LOSS FUNCTIONS")
print("="*80)

# Calculate old loss
loss_old = calculate_integer_loss_old(allowedK_integer, n_ignore=n_ignore)
print(f"\nOLD LOSS (weighted MSE on all high-k): {loss_old:.6e}")

# Calculate new loss for different n_tail
print(f"\nNEW LOSS (focusing on tail):")
print(f"{'n_tail':<8} {'Total Loss':<15} {'MSE Loss':<15} {'Slope Loss':<15} {'Deviation Slope':<15}")
print("-"*80)

best_n_tail = None
best_loss = float('inf')
results = {}

for n_tail in n_tail_values:
    total_loss, deviations, dev_slope, mse_loss, slope_loss = calculate_integer_loss_new(
        allowedK_integer, n_tail=n_tail, slope_weight=10.0
    )
    print(f"{n_tail:<8} {total_loss:<15.6e} {mse_loss:<15.6e} {slope_loss:<15.6e} {dev_slope:<15.6f}")

    results[n_tail] = {
        'total_loss': total_loss,
        'deviations': deviations,
        'dev_slope': dev_slope,
        'mse_loss': mse_loss,
        'slope_loss': slope_loss
    }

    if total_loss < best_loss:
        best_loss = total_loss
        best_n_tail = n_tail

print(f"\nBest n_tail: {best_n_tail} (loss = {best_loss:.6e})")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Compute ideal sequence for visualization
all_k = np.array(allowedK_integer)
high_k_for_fit = all_k[n_ignore:]
indices_for_fit = np.arange(len(high_k_for_fit))
slope, _ = np.polyfit(indices_for_fit, high_k_for_fit, 1)
ideal_spacing = np.round(slope)

# Full ideal sequence
indices_all = np.arange(len(all_k))
ideal_intercept = np.mean(all_k[n_ignore:] - ideal_spacing * np.arange(len(all_k[n_ignore:])))
ideal_k_full = ideal_spacing * indices_all + np.round(ideal_intercept)

# Full deviations
deviations_full = all_k - ideal_k_full

# Plot 1: allowedK vs ideal
ax1 = axes[0, 0]
ax1.plot(indices_all, all_k, 'o-', label='allowedK', markersize=4, alpha=0.7)
ax1.plot(indices_all, ideal_k_full, 's--', label=f'Ideal (spacing={ideal_spacing})', markersize=3, alpha=0.7)
ax1.axvline(x=n_ignore, color='gray', linestyle=':', alpha=0.5, label=f'n_ignore={n_ignore}')
for n_tail in n_tail_values:
    ax1.axvline(x=len(all_k)-n_tail, color='red', linestyle=':', alpha=0.3)
ax1.set_xlabel('Index')
ax1.set_ylabel('k-value')
ax1.set_title('allowedK vs Ideal Sequence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Full deviations with tail regions highlighted
ax2 = axes[0, 1]
ax2.plot(indices_all, deviations_full, 'o-', markersize=4, alpha=0.7, color='purple')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.axvline(x=n_ignore, color='gray', linestyle=':', alpha=0.5, label=f'n_ignore={n_ignore}')

# Highlight different tail regions
colors = ['blue', 'green', 'orange', 'red']
for i, n_tail in enumerate(n_tail_values):
    tail_start = len(all_k) - n_tail
    ax2.axvspan(tail_start, len(all_k), alpha=0.1, color=colors[i], label=f'n_tail={n_tail}')

ax2.set_xlabel('Index')
ax2.set_ylabel('Deviation (allowedK - ideal)')
ax2.set_title('Deviations from Ideal Sequence')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Tail deviations for different n_tail
ax3 = axes[1, 0]
for i, n_tail in enumerate(n_tail_values):
    tail_k = all_k[-n_tail:]
    tail_indices = np.arange(len(tail_k))
    ideal_intercept_tail = np.mean(tail_k - ideal_spacing * tail_indices)
    ideal_tail = ideal_spacing * tail_indices + np.round(ideal_intercept_tail)
    tail_dev = tail_k - ideal_tail

    # Fit line to deviations
    dev_slope, dev_intercept = np.polyfit(tail_indices, tail_dev, 1)
    fit_line = dev_slope * tail_indices + dev_intercept

    ax3.plot(tail_indices, tail_dev, 'o-', label=f'n_tail={n_tail} (slope={dev_slope:.4f})',
             color=colors[i], markersize=6, alpha=0.7)
    ax3.plot(tail_indices, fit_line, '--', color=colors[i], alpha=0.5)

ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('Tail Index')
ax3.set_ylabel('Deviation')
ax3.set_title('Tail Deviations (Goal: flat line at y=0)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Loss comparison
ax4 = axes[1, 1]
n_tail_arr = np.array(n_tail_values)
total_losses = [results[n]['total_loss'] for n in n_tail_values]
mse_losses = [results[n]['mse_loss'] for n in n_tail_values]
slope_losses = [results[n]['slope_loss'] for n in n_tail_values]

x_pos = np.arange(len(n_tail_values))
width = 0.25

ax4.bar(x_pos - width, mse_losses, width, label='MSE Loss', alpha=0.7)
ax4.bar(x_pos, [s*10 for s in slope_losses], width, label='Slope Loss (×10)', alpha=0.7)
ax4.bar(x_pos + width, total_losses, width, label='Total Loss', alpha=0.7)

ax4.set_xlabel('n_tail')
ax4.set_ylabel('Loss')
ax4.set_title('Loss Components for Different n_tail')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(n_tail_values)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('loss_function_comparison.pdf', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'loss_function_comparison.pdf'")
plt.show()

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"Based on the analysis, n_tail={best_n_tail} gives the best balance.")
print(f"This focuses on the last {best_n_tail} points where oscillations have stabilized.")
print(f"\nThe new loss function will:")
print(f"  1. Use indices {n_ignore}+ to determine ideal spacing = {ideal_spacing}")
print(f"  2. Focus on last {best_n_tail} points for optimization")
print(f"  3. Minimize both deviations AND their slope (trending)")
