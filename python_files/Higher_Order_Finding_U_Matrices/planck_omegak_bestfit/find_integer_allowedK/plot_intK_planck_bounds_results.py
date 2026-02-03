"""
Plot Bayesian optimization results for integer K search with Planck bounds.
Reads data from ./data/try_intK_planck_bounds/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# Data directory
data_folder = './data/try_intK_planck_bounds/'

def load_all_data(data_dir=data_folder):
    """
    Load all loss_params files from the data directory.
    Returns arrays of parameters and target values.
    """
    all_data = []

    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found!")
        return None

    # Find all loss_params files
    pattern = os.path.join(data_dir, 'try_*', 'loss_params_*.txt')
    files = glob.glob(pattern)

    print(f"Found {len(files)} loss_params files")

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                line = f.readline().strip()
                values = [float(x) for x in line.split()]
                if len(values) == 5:
                    loss_integer, mt, kt, Omegab_ratio, h = values
                    all_data.append({
                        'loss': loss_integer,
                        'mt': mt,
                        'kt': kt,
                        'Omegab_ratio': Omegab_ratio,
                        'h': h,
                        'target': -0.5 * loss_integer  # Same as in the optimization
                    })
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    if not all_data:
        print("No valid data loaded!")
        return None

    # Convert to structured format
    n_points = len(all_data)
    params = {
        'mt': np.array([d['mt'] for d in all_data]),
        'kt': np.array([d['kt'] for d in all_data]),
        'Omegab_ratio': np.array([d['Omegab_ratio'] for d in all_data]),
        'h': np.array([d['h'] for d in all_data])
    }
    targets = np.array([d['target'] for d in all_data])
    losses = np.array([d['loss'] for d in all_data])

    print(f"\nLoaded {n_points} data points")
    print(f"Parameter ranges:")
    for param_name, param_values in params.items():
        print(f"  {param_name:<15}: [{param_values.min():.6f}, {param_values.max():.6f}]")
    print(f"  loss_integer    : [{losses.min():.6f}, {losses.max():.6f}]")
    print(f"  -0.5*loss       : [{targets.min():.6f}, {targets.max():.6f}]")

    return params, targets, losses

def plot_2d_scatter(params, targets, losses, param1='mt', param2='kt',
                    output_file='./figures/bayes_opt_intK_planck_bounds_2d_scatter.pdf'):
    """Create a 2D scatter plot of observed points with log scale colorbar"""

    x_obs = params[param1]
    y_obs = params[param2]
    z_obs = losses  # Use loss_integer values directly (positive, for log scale)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use LogNorm for logarithmic color scale
    scatter = ax.scatter(x_obs, y_obs, c=z_obs, cmap='viridis_r',
                        s=50, edgecolors='black', linewidths=0.5,
                        norm=LogNorm(vmin=z_obs.min(), vmax=z_obs.max()))

    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)
    ax.set_title('Bayesian Optimization: Observed Points (Planck Bounds, Integer Loss Only)', fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('loss_integer (log scale)', fontsize=12)

    # Mark the best point (minimum loss)
    best_idx = np.argmin(z_obs)
    ax.plot(x_obs[best_idx], y_obs[best_idx], 'r*', markersize=20,
            label=f'Best: {param1}={x_obs[best_idx]:.2f}, {param2}={y_obs[best_idx]:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\nSaved 2D scatter plot to: {output_file}")
    plt.show()

def plot_all_param_pairs(params, targets, losses,
                         output_dir='./figures/'):
    """Create 2D scatter plots for all parameter pairs"""

    param_names = list(params.keys())

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i, param1 in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            if i < j:  # Only plot unique pairs
                print(f"\nCreating plot for {param1} vs {param2}...")
                output_file = os.path.join(output_dir,
                                          f'bayes_opt_planck_bounds_{param1}_{param2}_scatter.pdf')
                plot_2d_scatter(params, targets, losses, param1, param2, output_file)

def print_best_points(params, targets, losses, n_best=10):
    """Print the top N best points"""

    # Sort by target (higher is better)
    sorted_indices = np.argsort(targets)[::-1]

    print(f"\n{'='*80}")
    print(f"TOP {n_best} BEST PARAMETER SETS")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'loss':<12} {'-0.5*loss':<12} {'mt':<12} {'kt':<10} {'Ωb_ratio':<10} {'h':<10}")
    print(f"{'-'*80}")

    for i, idx in enumerate(sorted_indices[:n_best]):
        print(f"{i+1:<6} {losses[idx]:<12.6f} {targets[idx]:<12.6f} "
              f"{params['mt'][idx]:<12.4f} {params['kt'][idx]:<10.4f} "
              f"{params['Omegab_ratio'][idx]:<10.6f} {params['h'][idx]:<10.6f}")

    print(f"{'='*80}")

if __name__ == "__main__":
    print("="*80)
    print("PLOTTING BAYESIAN OPTIMIZATION RESULTS (PLANCK BOUNDS)")
    print("="*80)

    # Load all data
    result = load_all_data()

    if result is None:
        print("Failed to load data. Exiting.")
        exit(1)

    params, targets, losses = result

    # Print best points
    print_best_points(params, targets, losses, n_best=10)

    # Create the main plot (mt vs kt)
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)

    plot_2d_scatter(params, targets, losses, 'mt', 'kt',
                   output_file='./figures/bayes_opt_intK_planck_bounds_2d_scatter.pdf')

    # Optionally create plots for all parameter pairs
    print("\nDo you want to create plots for all parameter pairs? (yes/no)")
    # For automated execution, uncomment the next line to generate all plots
    # plot_all_param_pairs(params, targets, losses)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
