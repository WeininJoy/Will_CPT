"""
Extract a tighter parameter region around the best "island" from Bayesian optimization.
Focus on the narrow region where the very best points cluster.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
STATE_FILE = "./bayes_opt_state_planck_bounds.json"
OUTPUT_BOUNDS_FILE = "./refined_planck_bounds_tight.txt"
OUTPUT_FIG_DIR = "./figures/"

# Tighter analysis parameters
LOSS_PERCENTILE = 1  # Use only top 1% (or specify absolute threshold)
USE_ABSOLUTE_THRESHOLD = True  # Use absolute loss threshold instead
LOSS_THRESHOLD = 0.02  # Only consider points with loss < this value

# Manual bounds specification (set to None to auto-detect)
MANUAL_BOUNDS = {
    'mt': [416.831616, 418.831616],
    'kt': [1.40, 1.49],
    'Omegab_ratio': None,  # Will auto-detect
    'h': None,  # Will auto-detect
}

EXPANSION_FACTOR = 1.1  # Smaller expansion (10% instead of 20%)

def load_bayes_opt_state(filepath):
    """Load the Bayesian optimization state from JSON file."""
    with open(filepath, 'r') as f:
        state = json.load(f)
    return state

def analyze_best_region_tight(state, loss_threshold=None, percentile=None, manual_bounds=None):
    """
    Analyze the parameter space focusing on the tightest cluster of best points.
    """
    # Extract data
    params = np.array(state['params'])
    targets = np.array(state['target'])
    param_names = state['keys']
    original_bounds = state['pbounds']

    # Convert targets back to loss values
    loss_values = -2.0 * targets

    # Select best points based on threshold or percentile
    if loss_threshold is not None:
        best_mask = loss_values < loss_threshold
        top_indices = np.where(best_mask)[0]
        selection_method = f"loss < {loss_threshold}"
    else:
        n_top = max(int(len(targets) * percentile / 100), 3)
        top_indices = np.argsort(targets)[-n_top:]
        selection_method = f"top {percentile}%"

    print(f"\n{'='*80}")
    print(f"TIGHT REGION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal evaluations: {len(targets)}")
    print(f"Selection method: {selection_method}")
    print(f"Selected points: {len(top_indices)}")

    if len(top_indices) < 3:
        print("\nWARNING: Too few points selected. Consider relaxing the threshold.")
        return None, None

    # Statistics on selected points
    best_loss = loss_values[top_indices]
    print(f"\nLoss statistics for selected {len(top_indices)} points:")
    print(f"  Min loss:  {np.min(best_loss):.6e}")
    print(f"  Max loss:  {np.max(best_loss):.6e}")
    print(f"  Mean loss: {np.mean(best_loss):.6e}")
    print(f"  Std loss:  {np.std(best_loss):.6e}")

    # Extract best parameters
    best_params = params[top_indices]

    # Calculate new bounds
    new_bounds = {}
    print(f"\n{'='*80}")
    print("TIGHT PARAMETER BOUNDS")
    print(f"{'='*80}")

    for i, param_name in enumerate(param_names):
        # Check if manual bounds specified
        if manual_bounds and manual_bounds.get(param_name) is not None:
            new_min, new_max = manual_bounds[param_name]
            print(f"\n{param_name}:")
            print(f"  Using MANUAL bounds: [{new_min:.6f}, {new_max:.6f}]")
        else:
            param_values = best_params[:, i]

            # Calculate statistics
            param_min = np.min(param_values)
            param_max = np.max(param_values)
            param_mean = np.mean(param_values)
            param_std = np.std(param_values)

            # Calculate new bounds with tight expansion
            center = param_mean
            half_range = (param_max - param_min) / 2 * EXPANSION_FACTOR
            new_min = center - half_range
            new_max = center + half_range

            # Ensure within original bounds
            orig_min, orig_max = original_bounds[param_name]
            new_min = max(new_min, orig_min)
            new_max = min(new_max, orig_max)

            print(f"\n{param_name}:")
            print(f"  Original bounds:   [{orig_min:.6f}, {orig_max:.6f}]")
            print(f"  Best points range: [{param_min:.6f}, {param_max:.6f}]")
            print(f"  Mean ± std:        {param_mean:.6f} ± {param_std:.6f}")
            print(f"  New bounds:        [{new_min:.6f}, {new_max:.6f}]")
            orig_range = orig_max - orig_min
            new_range = new_max - new_min
            print(f"  Reduction:         {(1 - new_range/orig_range)*100:.1f}%")

        new_bounds[param_name] = (new_min, new_max)

    # Print best point
    best_idx = top_indices[np.argmin(best_loss)]
    print(f"\n{'='*80}")
    print("ABSOLUTE BEST POINT")
    print(f"{'='*80}")
    print(f"Loss: {loss_values[best_idx]:.6e}")
    for i, param_name in enumerate(param_names):
        print(f"  {param_name:<15} = {params[best_idx, i]:.10f}")

    # Print all selected points
    print(f"\n{'='*80}")
    print(f"ALL {len(top_indices)} SELECTED POINTS")
    print(f"{'='*80}")
    print(f"{'Loss':<12} {'mt':<12} {'kt':<12} {'Omegab_ratio':<12} {'h':<12}")
    print("-" * 60)
    for idx in sorted(top_indices, key=lambda x: loss_values[x]):
        print(f"{loss_values[idx]:<12.6e} ", end="")
        for i in range(len(param_names)):
            print(f"{params[idx, i]:<12.6f} ", end="")
        print()

    stats = {
        'n_total': len(targets),
        'n_best': len(top_indices),
        'best_loss': np.min(best_loss),
        'mean_loss': np.mean(best_loss),
        'std_loss': np.std(best_loss),
        'best_params': params[best_idx],
        'top_indices': top_indices
    }

    return new_bounds, stats

def save_bounds_file(bounds, filepath, stats):
    """Save bounds to file."""
    with open(filepath, 'w') as f:
        f.write("# Tight refined parameter bounds from Bayesian optimization\n")
        f.write(f"# Generated from top {stats['n_best']} points out of {stats['n_total']} evaluations\n")
        f.write(f"# Best loss: {stats['best_loss']:.6e}\n")
        f.write(f"# Mean loss: {stats['mean_loss']:.6e}\n")
        f.write(f"# Std loss:  {stats['std_loss']:.6e}\n")
        f.write("#\n")
        f.write("# Format: parameter_name  lower_bound  upper_bound\n")
        f.write("#\n")

        for param_name, (lower, upper) in bounds.items():
            f.write(f"{param_name:<15} {lower:.10f}  {upper:.10f}\n")

    print(f"\n{'='*80}")
    print(f"Tight bounds saved to: {filepath}")
    print(f"{'='*80}")

def plot_tight_region(state, stats, output_dir):
    """Visualize the tight region."""
    params = np.array(state['params'])
    targets = np.array(state['target'])
    param_names = state['keys']
    top_indices = stats['top_indices']
    loss_values = -2.0 * targets

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 2D projections
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    param_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    for idx, (i, j) in enumerate(param_pairs):
        ax = axes[idx]

        # All points (faded)
        ax.scatter(params[:, i], params[:, j],
                  c=loss_values, s=20, alpha=0.3,
                  cmap='viridis_r', vmin=0, vmax=np.percentile(loss_values, 95))

        # Selected tight region points
        scatter = ax.scatter(params[top_indices, i], params[top_indices, j],
                           c=loss_values[top_indices], s=80, alpha=0.8,
                           cmap='Reds_r', edgecolors='black', linewidths=1,
                           label=f'Selected {len(top_indices)} points', zorder=5)

        # Best point
        best_idx = top_indices[np.argmin(loss_values[top_indices])]
        ax.scatter(params[best_idx, i], params[best_idx, j],
                  c='yellow', s=250, marker='*', edgecolors='black',
                  linewidths=2, label='Best', zorder=6)

        ax.set_xlabel(param_names[i], fontsize=11)
        ax.set_ylabel(param_names[j], fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax, label='Loss (selected)')

    plt.suptitle('Tight Region Analysis - Best Points Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path(output_dir) / 'tight_region_analysis.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()

def main():
    print(f"Loading optimization results from: {STATE_FILE}")
    state = load_bayes_opt_state(STATE_FILE)

    # Analyze with tight constraints
    if USE_ABSOLUTE_THRESHOLD:
        new_bounds, stats = analyze_best_region_tight(
            state,
            loss_threshold=LOSS_THRESHOLD,
            manual_bounds=MANUAL_BOUNDS
        )
    else:
        new_bounds, stats = analyze_best_region_tight(
            state,
            percentile=LOSS_PERCENTILE,
            manual_bounds=MANUAL_BOUNDS
        )

    if new_bounds is None:
        print("\nAnalysis failed. Adjust thresholds and try again.")
        return

    # Save and visualize
    save_bounds_file(new_bounds, OUTPUT_BOUNDS_FILE, stats)
    plot_tight_region(state, stats, OUTPUT_FIG_DIR)

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Review: {OUTPUT_BOUNDS_FILE}")
    print(f"2. Check: {OUTPUT_FIG_DIR}/tight_region_analysis.pdf")
    print(f"3. Use in load_planck_bounds(): bounds_file='{OUTPUT_BOUNDS_FILE}'")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
