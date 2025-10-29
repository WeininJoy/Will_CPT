import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt import BayesianOptimization
import json

# The state file saved by the optimizer
STATE_FILE = "./bayes_opt_state.json"

# Define the bounds (must match find_intK_baye_opt.py)
pbounds = {
    'mt': (350, 500),
    'kt': (1.2, 1.8),
    'Omegab_ratio': (0.15, 0.17),
    'h': (0.5, 0.62)
}

def dummy_function(**kwargs):
    """Dummy function - we won't actually call it"""
    return 0

def load_optimizer_state(state_file=STATE_FILE):
    """Load the optimizer state from JSON file"""
    optimizer = BayesianOptimization(
        f=dummy_function,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.load_state(state_file)
    print(f"Loaded optimizer with {len(optimizer.space)} data points")
    return optimizer

def extract_data_2d(optimizer, param1='mt', param2='kt'):
    """
    Extract 2D data for specific parameters.
    Returns the observed points and their target values.
    """
    # Get all observed data
    data = optimizer.space.params
    targets = optimizer.space.target

    # Find indices for the two parameters
    param_names = list(pbounds.keys())
    idx1 = param_names.index(param1)
    idx2 = param_names.index(param2)

    # Extract the values
    x_values = data[:, idx1]
    y_values = data[:, idx2]
    z_values = targets

    return x_values, y_values, z_values

def predict_on_grid(optimizer, param1='mt', param2='kt', grid_size=100):
    """
    Use the Gaussian Process to predict values on a regular grid.
    Fixes other parameters to their best values.
    """
    # Get the best point
    best_params = optimizer.max['params']

    # Create grid for the two parameters of interest
    param_names = list(pbounds.keys())
    idx1 = param_names.index(param1)
    idx2 = param_names.index(param2)

    x_range = pbounds[param1]
    y_range = pbounds[param2]

    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Prepare points for prediction
    n_points = grid_size * grid_size
    test_points = np.zeros((n_points, len(param_names)))

    # Set all parameters to their best values
    for i, pname in enumerate(param_names):
        test_points[:, i] = best_params[pname]

    # Override with grid values for the two parameters we're plotting
    test_points[:, idx1] = X.ravel()
    test_points[:, idx2] = Y.ravel()

    # Predict using GP
    Z_pred, sigma = optimizer._gp.predict(test_points, return_std=True)
    Z_pred = Z_pred.reshape(grid_size, grid_size)
    sigma = sigma.reshape(grid_size, grid_size)

    return X, Y, Z_pred, sigma

def plot_2d_scatter(x_obs, y_obs, z_obs, param1='mt', param2='kt'):
    """Create a simple 2D scatter plot of observed points"""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(x_obs, y_obs, c=z_obs, cmap='viridis',
                        s=50, edgecolors='black', linewidths=0.5)

    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)
    ax.set_title('Bayesian Optimization: Observed Points (Integer Loss Only)', fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('-0.5 × loss_integer', fontsize=12)

    # Mark the best point
    best_idx = np.argmax(z_obs)
    ax.plot(x_obs[best_idx], y_obs[best_idx], 'r*', markersize=20,
            label=f'Best: {param1}={x_obs[best_idx]:.2f}, {param2}={y_obs[best_idx]:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig('./bayes_opt_intK_2d_scatter.pdf')
    print("Saved 2D scatter plot to: bayes_opt_intK_2d_scatter.pdf")
    plt.show()

def plot_2d_gp_contour(X, Y, Z_pred, x_obs, y_obs, z_obs, param1='mt', param2='kt'):
    """Create a 2D contour plot using GP predictions"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create filled contour plot from GP predictions
    contour_filled = ax.contourf(X, Y, Z_pred, levels=20, cmap='viridis', alpha=0.8)
    contour_lines = ax.contour(X, Y, Z_pred, levels=10, colors='black', alpha=0.3, linewidths=0.5)

    # Overlay the actual observed points
    scatter = ax.scatter(x_obs, y_obs, c=z_obs, cmap='viridis',
                        s=50, edgecolors='red', linewidths=1.5, label='Observed points')

    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)
    ax.set_title('Bayesian Optimization: GP Interpolation (Integer Loss Only)', fontsize=14)

    cbar = plt.colorbar(contour_filled, ax=ax)
    cbar.set_label('-0.5 × loss_integer', fontsize=12)

    # Mark the best observed point
    best_idx = np.argmax(z_obs)
    ax.plot(x_obs[best_idx], y_obs[best_idx], 'r*', markersize=20,
            label=f'Best: {param1}={x_obs[best_idx]:.2f}, {param2}={y_obs[best_idx]:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig('./bayes_opt_intK_2d_gp_contour.pdf')
    print("Saved 2D GP contour plot to: bayes_opt_intK_2d_gp_contour.pdf")
    plt.show()

def plot_3d_surface(X, Y, Z_pred, x_obs, y_obs, z_obs, param1='mt', param2='kt'):
    """Create a 3D surface plot using GP predictions"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the GP surface
    surf = ax.plot_surface(X, Y, Z_pred, cmap='viridis', alpha=0.7,
                          edgecolor='none', antialiased=True)

    # Overlay the observed points
    ax.scatter(x_obs, y_obs, z_obs, c='red', s=50, edgecolors='black',
              linewidths=1, label='Observed points')

    ax.set_xlabel(param1, fontsize=11)
    ax.set_ylabel(param2, fontsize=11)
    ax.set_zlabel('-0.5 × loss_integer', fontsize=11)
    ax.set_title('Bayesian Optimization: GP Surface (Integer Loss Only)', fontsize=14)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('-0.5 × loss_integer', fontsize=11)

    plt.tight_layout()
    plt.savefig('./bayes_opt_intK_3d_gp_surface.pdf')
    print("Saved 3D GP surface plot to: bayes_opt_intK_3d_gp_surface.pdf")
    plt.show()

def plot_uncertainty(X, Y, sigma, x_obs, y_obs, param1='mt', param2='kt'):
    """Plot the GP uncertainty (standard deviation)"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create filled contour plot of uncertainty
    contour_filled = ax.contourf(X, Y, sigma, levels=20, cmap='Reds', alpha=0.8)
    contour_lines = ax.contour(X, Y, sigma, levels=10, colors='black', alpha=0.3, linewidths=0.5)

    # Overlay the observed points
    ax.scatter(x_obs, y_obs, c='blue', s=30, edgecolors='black',
              linewidths=0.5, label='Observed points')

    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)
    ax.set_title('GP Uncertainty (Standard Deviation)', fontsize=14)

    cbar = plt.colorbar(contour_filled, ax=ax)
    cbar.set_label('Uncertainty (σ)', fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig('./bayes_opt_intK_gp_uncertainty.pdf')
    print("Saved GP uncertainty plot to: bayes_opt_intK_gp_uncertainty.pdf")
    plt.show()

if __name__ == "__main__":
    # Load the optimizer state
    print(f"Loading optimizer state from {STATE_FILE}...")
    optimizer = load_optimizer_state()

    # Extract observed data for mt vs kt
    print("\nExtracting data for mt vs kt...")
    x_obs, y_obs, z_obs = extract_data_2d(optimizer, 'mt', 'kt')

    print(f"\nData statistics:")
    print(f"  Number of points: {len(x_obs)}")
    print(f"  mt range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
    print(f"  kt range: [{y_obs.min():.3f}, {y_obs.max():.3f}]")
    print(f"  -0.5*loss_integer range: [{z_obs.min():.2f}, {z_obs.max():.2f}]")
    print(f"  Best -0.5*loss_integer: {z_obs.max():.2f} at mt={x_obs[np.argmax(z_obs)]:.2f}, kt={y_obs[np.argmax(z_obs)]:.3f}")
    print(f"  Best from optimizer: {optimizer.max}")

    # Predict on a grid using the GP
    print("\nGenerating GP predictions on grid...")
    X, Y, Z_pred, sigma = predict_on_grid(optimizer, 'mt', 'kt', grid_size=100)

    # Create plots
    print("\nGenerating plots...")
    plot_2d_scatter(x_obs, y_obs, z_obs, 'mt', 'kt')
    plot_2d_gp_contour(X, Y, Z_pred, x_obs, y_obs, z_obs, 'mt', 'kt')
    plot_3d_surface(X, Y, Z_pred, x_obs, y_obs, z_obs, 'mt', 'kt')
    plot_uncertainty(X, Y, sigma, x_obs, y_obs, 'mt', 'kt')

    print("\nDone! Generated 4 plots:")
    print("  1. bayes_opt_intK_2d_scatter.pdf - Observed points only")
    print("  2. bayes_opt_intK_2d_gp_contour.pdf - GP interpolation as contour")
    print("  3. bayes_opt_intK_3d_gp_surface.pdf - GP interpolation as 3D surface")
    print("  4. bayes_opt_intK_gp_uncertainty.pdf - GP uncertainty map")
