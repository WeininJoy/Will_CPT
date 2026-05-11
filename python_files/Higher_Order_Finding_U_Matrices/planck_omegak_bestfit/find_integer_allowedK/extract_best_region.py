"""
Extract parameter region around best-fit points from Bayesian optimization results.
This script identifies the "island" with low loss_integer values and outputs new bounds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
STATE_FILE = "./bayes_opt_state_planck_bounds.json"
OUTPUT_BOUNDS_FILE = "./refined_planck_bounds.txt"
OUTPUT_FIG_DIR = "./figures/"

# Analysis parameters
TOP_N_PERCENT = 10  # Consider top 10% of points
EXPANSION_FACTOR = 1.2  # Expand bounds by 20% for safety margin

def load_bayes_opt_state(filepath):
    """Load the Bayesian optimization state from JSON file."""
    with open(filepath, 'r') as f:
        state = json.load(f)
    return state

def analyze_best_region(state, top_n_percent=10):
    """
    Analyze the parameter space and identify the best region.

    Parameters:
    -----------
    state : dict
        Bayesian optimization state
    top_n_percent : float
        Percentage of top points to consider

    Returns:
    --------
    dict : Parameter bounds for the best region
    dict : Statistics about the best region
    """
    # Extract data
    params = np.array(state['params'])
    targets = np.array(state['target'])
    param_names = state['keys']
    original_bounds = state['pbounds']

    # Convert targets back to loss values (targets = -0.5 * loss)
    loss_values = -2.0 * targets

    # Find top N% of points
    n_top = max(int(len(targets) * top_n_percent / 100), 5)  # At least 5 points
    top_indices = np.argsort(targets)[-n_top:]  # Highest targets = lowest loss

    print(f"\n{'='*80}")
    print(f"ANALYSIS OF BEST PARAMETER REGION")
    print(f"{'='*80}")
    print(f"\nTotal evaluations: {len(targets)}")
    print(f"Analyzing top {n_top} points ({top_n_percent}%)")

    # Statistics on the best points
    best_loss = loss_values[top_indices]
    print(f"\nLoss statistics for top {n_top} points:")
    print(f"  Min loss:  {np.min(best_loss):.6e}")
    print(f"  Max loss:  {np.max(best_loss):.6e}")
    print(f"  Mean loss: {np.mean(best_loss):.6e}")
    print(f"  Std loss:  {np.std(best_loss):.6e}")

    # Extract best parameters
    best_params = params[top_indices]

    # Calculate new bounds for each parameter
    new_bounds = {}
    print(f"\n{'='*80}")
    print("PARAMETER BOUNDS ANALYSIS")
    print(f"{'='*80}")

    for i, param_name in enumerate(param_names):
        param_values = best_params[:, i]

        # Calculate statistics
        param_min = np.min(param_values)
        param_max = np.max(param_values)
        param_mean = np.mean(param_values)
        param_std = np.std(param_values)
        param_range = param_max - param_min

        # Calculate new bounds with expansion factor
        center = param_mean
        half_range = (param_max - param_min) / 2 * EXPANSION_FACTOR
        new_min = center - half_range
        new_max = center + half_range

        # Ensure new bounds are within original bounds
        orig_min, orig_max = original_bounds[param_name]
        new_min = max(new_min, orig_min)
        new_max = min(new_max, orig_max)

        new_bounds[param_name] = (new_min, new_max)

        print(f"\n{param_name}:")
        print(f"  Original bounds:  [{orig_min:.6f}, {orig_max:.6f}]")
        print(f"  Best points range: [{param_min:.6f}, {param_max:.6f}]")
        print(f"  Mean ± std:        {param_mean:.6f} ± {param_std:.6f}")
        print(f"  New bounds:        [{new_min:.6f}, {new_max:.6f}]")
        print(f"  Reduction:         {(1 - (new_max - new_min)/(orig_max - orig_min))*100:.1f}%")

    # Print best point details
    best_idx = top_indices[-1]
    print(f"\n{'='*80}")
    print("BEST POINT DETAILS")
    print(f"{'='*80}")
    print(f"Loss: {loss_values[best_idx]:.6e}")
    print(f"Target: {targets[best_idx]:.6e}")
    for i, param_name in enumerate(param_names):
        print(f"  {param_name:<15} = {params[best_idx, i]:.6f}")

    # Calculate statistics
    stats = {
        'n_total': len(targets),
        'n_best': n_top,
        'best_loss': np.min(best_loss),
        'mean_loss': np.mean(best_loss),
        'std_loss': np.std(best_loss),
        'best_params': params[best_idx],
        'top_indices': top_indices
    }

    return new_bounds, stats

def save_bounds_file(bounds, filepath, stats):
    """Save bounds to a text file in the format expected by load_planck_bounds()."""
    with open(filepath, 'w') as f:
        f.write("# Refined parameter bounds from Bayesian optimization\n")
        f.write(f"# Generated from top {stats['n_best']} points out of {stats['n_total']} evaluations\n")
        f.write(f"# Best loss: {stats['best_loss']:.6e}\n")
        f.write(f"# Mean loss: {stats['mean_loss']:.6e}\n")
        f.write("#\n")
        f.write("# Format: parameter_name  lower_bound  upper_bound\n")
        f.write("#\n")

        for param_name, (lower, upper) in bounds.items():
            f.write(f"{param_name:<15} {lower:.10f}  {upper:.10f}\n")

    print(f"\n{'='*80}")
    print(f"Bounds saved to: {filepath}")
    print(f"{'='*80}")

def plot_best_region(state, stats, output_dir):
    """Create visualization of the best region."""
    params = np.array(state['params'])
    targets = np.array(state['target'])
    param_names = state['keys']
    top_indices = stats['top_indices']

    # Convert targets to loss
    loss_values = -2.0 * targets

    # Create figure directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot 1: 2D projections showing best region
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    param_pairs = [
        (0, 1),  # mt vs kt
        (0, 2),  # mt vs Omegab_ratio
        (0, 3),  # mt vs h
        (1, 2),  # kt vs Omegab_ratio
        (1, 3),  # kt vs h
        (2, 3),  # Omegab_ratio vs h
    ]

    for idx, (i, j) in enumerate(param_pairs):
        ax = axes[idx]

        # Plot all points
        scatter = ax.scatter(params[:, i], params[:, j],
                           c=loss_values, s=30, alpha=0.5,
                           cmap='viridis_r', vmin=0, vmax=np.percentile(loss_values, 95))

        # Highlight best points
        ax.scatter(params[top_indices, i], params[top_indices, j],
                  c='red', s=100, marker='*', edgecolors='black',
                  label=f'Top {len(top_indices)} points', zorder=5)

        # Mark the very best point
        best_idx = top_indices[-1]
        ax.scatter(params[best_idx, i], params[best_idx, j],
                  c='yellow', s=200, marker='*', edgecolors='black',
                  linewidths=2, label='Best point', zorder=6)

        ax.set_xlabel(param_names[i], fontsize=11)
        ax.set_ylabel(param_names[j], fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            plt.colorbar(scatter, ax=ax, label='Loss')

    plt.tight_layout()
    output_path = Path(output_dir) / 'best_region_analysis.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()

    # Plot 2: Loss distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of all losses
    ax1.hist(loss_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(stats['best_loss'], color='red', linestyle='--', linewidth=2,
                label=f"Best: {stats['best_loss']:.2e}")
    ax1.axvline(stats['mean_loss'], color='green', linestyle='--', linewidth=2,
                label=f"Mean of top: {stats['mean_loss']:.2e}")
    ax1.set_xlabel('Loss', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Loss Values', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Scatter: iteration vs loss
    ax2.scatter(range(len(loss_values)), loss_values, c='blue', s=20, alpha=0.5)
    ax2.scatter(top_indices, loss_values[top_indices], c='red', s=60,
                marker='*', edgecolors='black', label=f'Top {len(top_indices)}')
    best_idx = top_indices[-1]
    ax2.scatter(best_idx, loss_values[best_idx], c='yellow', s=120,
                marker='*', edgecolors='black', linewidths=2, label='Best')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Evolution', fontsize=13)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'loss_distribution_analysis.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()

def main():
    # Load optimization state
    print(f"Loading optimization results from: {STATE_FILE}")
    state = load_bayes_opt_state(STATE_FILE)

    # Analyze best region
    new_bounds, stats = analyze_best_region(state, top_n_percent=TOP_N_PERCENT)

    # Save bounds to file
    save_bounds_file(new_bounds, OUTPUT_BOUNDS_FILE, stats)

    # Create visualizations
    plot_best_region(state, stats, OUTPUT_FIG_DIR)

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Review the refined bounds in: {OUTPUT_BOUNDS_FILE}")
    print(f"2. Check the visualizations in: {OUTPUT_FIG_DIR}")
    print(f"3. Run the optimization again with refined bounds:")
    print(f"   Modify load_planck_bounds() in find_intK_baye_opt_planck_bounds.py")
    print(f"   to use: '{OUTPUT_BOUNDS_FILE}'")
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
