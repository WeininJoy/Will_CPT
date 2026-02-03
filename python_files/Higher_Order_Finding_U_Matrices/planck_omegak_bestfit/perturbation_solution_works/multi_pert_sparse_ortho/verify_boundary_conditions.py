#!/usr/bin/env python3
"""
Verify boundary conditions for original and sparse eigenfunctions.

This script checks if the reconstruction method is correct by examining
boundary conditions for both original and sparse rotated solutions.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

def check_boundary_conditions(eta_grid, solutions, pert_type, label=""):
    """
    Check boundary conditions at the end of the universe.

    Parameters:
    -----------
    eta_grid : array
        Time grid
    solutions : list of arrays
        List of solution arrays for each eigenfunction
    pert_type : str
        Type of perturbation ('dr', 'dm', 'vr', 'vm')
    label : str
        Label for printing
    """
    print(f"\n{label} - {pert_type} boundary conditions:")
    print("-" * 60)

    for i, sol in enumerate(solutions):
        # Value at the end
        end_value = sol[-1]

        # Derivative at the end (using last few points)
        if len(eta_grid) > 10:
            d_eta = eta_grid[-1] - eta_grid[-2]
            derivative = (sol[-1] - sol[-2]) / d_eta
        else:
            derivative = 0.0

        print(f"  Mode {i}: end_value = {end_value:+.6e}, derivative = {derivative:+.6e}")

        # Check expected behavior
        if pert_type in ['vr', 'vm']:
            # Anti-symmetric: should have zero at end
            if abs(end_value) > 1e-3:
                print(f"    WARNING: Anti-symmetric mode should have ~0 at end, got {end_value:.6e}")
        elif pert_type in ['dr', 'dm']:
            # Symmetric: should have zero derivative at end
            if abs(derivative) > 1e-2:
                print(f"    WARNING: Symmetric mode should have ~0 derivative at end, got {derivative:.6e}")


def reconstruct_original_eigenfunctions(results, N_plot=5):
    """
    Reconstruct original eigenfunctions (before any rotation) and check boundary conditions.
    """
    print("="*80)
    print("CHECKING ORIGINAL EIGENFUNCTIONS (Before Sparse Rotation)")
    print("="*80)

    eta_grid = results['eta_grid']
    eigenvals_1 = results['eigenvals_1']
    eigenvecs_1 = results['eigenvecs_1']
    coefficients_1 = results['coefficients_1']
    basis_1 = results['basis_1']

    # Filter to eigenvalues > 0.99
    eigenvals_array = np.array(eigenvals_1)
    valid_mask = eigenvals_array.real > 0.99
    valid_indices = np.where(valid_mask)[0]

    print(f"\nFound {len(valid_indices)} eigenvectors with λ > 0.99")
    print(f"Eigenvalues: {eigenvals_array[valid_indices].real}")

    N_plot = min(N_plot, len(valid_indices))

    perturbation_types = ['dr', 'dm', 'vr', 'vm']

    for pert_type in perturbation_types:
        if pert_type not in coefficients_1:
            continue

        solutions = []

        for i in range(N_plot):
            idx = valid_indices[i]

            # Reconstruct using coefficients and basis
            solution = np.zeros_like(eta_grid)
            coeffs = coefficients_1[pert_type][idx, :]
            N_basis = len(coeffs)

            for k in range(N_basis):
                solution += coeffs[k] * basis_1[pert_type][:, k]

            solutions.append(solution)

        check_boundary_conditions(eta_grid, solutions, pert_type,
                                 label="ORIGINAL")


def reconstruct_sparse_eigenfunctions(results, sparse_results, N_plot=5):
    """
    Reconstruct sparse rotated eigenfunctions and check boundary conditions.
    """
    print("\n" + "="*80)
    print("CHECKING SPARSE EIGENFUNCTIONS (After Varimax Rotation)")
    print("="*80)

    eta_grid = results['eta_grid']
    eigenvals_valid = sparse_results['eigenvals_valid']
    coefficients_sparse = sparse_results['coefficients_sparse']
    basis_1 = results['basis_1']

    N_plot = min(N_plot, len(eigenvals_valid))

    print(f"\nChecking {N_plot} sparse eigenfunctions")
    print(f"Eigenvalues: {eigenvals_valid.real}")

    perturbation_types = ['dr', 'dm', 'vr', 'vm']

    for pert_type in perturbation_types:
        if pert_type not in coefficients_sparse:
            continue

        solutions = []

        # Sort by dominant k
        max_indices = np.argmax(np.abs(coefficients_sparse[pert_type].real), axis=1)
        sorted_order = np.argsort(max_indices)

        for i in range(N_plot):
            plot_idx = sorted_order[i]

            # Reconstruct using sparse coefficients and basis
            solution = np.zeros_like(eta_grid)
            coeffs = coefficients_sparse[pert_type][plot_idx, :]
            N_basis = len(coeffs)

            for k in range(N_basis):
                solution += coeffs[k] * basis_1[pert_type][:, k]

            solutions.append(solution)

        check_boundary_conditions(eta_grid, solutions, pert_type,
                                 label="SPARSE")


def plot_comparison_with_boundary_check(results, sparse_results):
    """
    Plot original vs sparse with boundary condition visualization.
    """
    eta_grid = results['eta_grid']
    eigenvals_valid = sparse_results['eigenvals_valid']
    coefficients_original = sparse_results['coefficients_original']
    coefficients_sparse = sparse_results['coefficients_sparse']
    basis_1 = results['basis_1']

    N_plot = min(3, len(eigenvals_valid))

    # Plot for vr (anti-symmetric, should → 0)
    pert_type = 'vr'

    fig, axes = plt.subplots(N_plot, 2, figsize=(14, 2.5*N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Boundary Condition Check: {pert_type} (should → 0 at end)", fontsize=14)

    # Sort by dominant k
    max_indices = np.argmax(np.abs(coefficients_sparse[pert_type].real), axis=1)
    sorted_order = np.argsort(max_indices)

    for i in range(N_plot):
        plot_idx = sorted_order[i]

        # Original
        ax_orig = axes[i, 0]
        solution_orig = np.zeros_like(eta_grid)
        coeffs_orig = coefficients_original[pert_type][plot_idx, :]
        for k in range(len(coeffs_orig)):
            solution_orig += coeffs_orig[k] * basis_1[pert_type][:, k]

        ax_orig.plot(eta_grid, solution_orig, 'r-', linewidth=2.0, alpha=0.8)
        end_val = solution_orig[-1]
        ax_orig.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax_orig.set_title(f"Original (λ={eigenvals_valid[plot_idx]:.3f}, end={end_val:.2e})",
                         fontsize=10)
        ax_orig.grid(True, alpha=0.3)
        ax_orig.set_ylabel("Amplitude", fontsize=10)
        if i == N_plot - 1:
            ax_orig.set_xlabel("Conformal Time η", fontsize=10)

        # Sparse
        ax_sparse = axes[i, 1]
        solution_sparse = np.zeros_like(eta_grid)
        coeffs_sparse = coefficients_sparse[pert_type][plot_idx, :]
        for k in range(len(coeffs_sparse)):
            solution_sparse += coeffs_sparse[k] * basis_1[pert_type][:, k]

        dominant_k = max_indices[plot_idx]
        end_val_sparse = solution_sparse[-1]
        ax_sparse.plot(eta_grid, solution_sparse, 'g-', linewidth=2.0, alpha=0.8)
        ax_sparse.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax_sparse.set_title(f"Sparse (k={dominant_k+1}, end={end_val_sparse:.2e})",
                           fontsize=10)
        ax_sparse.grid(True, alpha=0.3)
        if i == N_plot - 1:
            ax_sparse.set_xlabel("Conformal Time η", fontsize=10)

    plt.tight_layout()
    plt.savefig("./figures/boundary_condition_check_vr.pdf", dpi=300, bbox_inches='tight')
    print("\nSaved boundary condition check plot to ./figures/boundary_condition_check_vr.pdf")
    plt.show()

    # Plot for dr (symmetric, should have zero derivative)
    pert_type = 'dr'

    fig, axes = plt.subplots(N_plot, 2, figsize=(14, 2.5*N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Boundary Condition Check: {pert_type} (should have flat end)", fontsize=14)

    for i in range(N_plot):
        plot_idx = sorted_order[i]

        # Original
        ax_orig = axes[i, 0]
        solution_orig = np.zeros_like(eta_grid)
        coeffs_orig = coefficients_original[pert_type][plot_idx, :]
        for k in range(len(coeffs_orig)):
            solution_orig += coeffs_orig[k] * basis_1[pert_type][:, k]

        d_eta = eta_grid[-1] - eta_grid[-2]
        deriv = (solution_orig[-1] - solution_orig[-2]) / d_eta

        ax_orig.plot(eta_grid, solution_orig, 'r-', linewidth=2.0, alpha=0.8)
        ax_orig.set_title(f"Original (λ={eigenvals_valid[plot_idx]:.3f}, deriv={deriv:.2e})",
                         fontsize=10)
        ax_orig.grid(True, alpha=0.3)
        ax_orig.set_ylabel("Amplitude", fontsize=10)
        if i == N_plot - 1:
            ax_orig.set_xlabel("Conformal Time η", fontsize=10)

        # Sparse
        ax_sparse = axes[i, 1]
        solution_sparse = np.zeros_like(eta_grid)
        coeffs_sparse = coefficients_sparse[pert_type][plot_idx, :]
        for k in range(len(coeffs_sparse)):
            solution_sparse += coeffs_sparse[k] * basis_1[pert_type][:, k]

        dominant_k = max_indices[plot_idx]
        deriv_sparse = (solution_sparse[-1] - solution_sparse[-2]) / d_eta

        ax_sparse.plot(eta_grid, solution_sparse, 'g-', linewidth=2.0, alpha=0.8)
        ax_sparse.set_title(f"Sparse (k={dominant_k+1}, deriv={deriv_sparse:.2e})",
                           fontsize=10)
        ax_sparse.grid(True, alpha=0.3)
        if i == N_plot - 1:
            ax_sparse.set_xlabel("Conformal Time η", fontsize=10)

    plt.tight_layout()
    plt.savefig("./figures/boundary_condition_check_dr.pdf", dpi=300, bbox_inches='tight')
    print("Saved boundary condition check plot to ./figures/boundary_condition_check_dr.pdf")
    plt.show()


if __name__ == "__main__":
    # Load original results
    print("Loading multi_perturbation_analysis results...")
    with open("../multi_perturbation_results.pickle", 'rb') as f:
        results = pickle.load(f)

    # Check original eigenfunctions
    reconstruct_original_eigenfunctions(results, N_plot=5)

    # Load and check sparse results if available
    try:
        print("\nLoading Varimax sparse results...")
        with open("varimax_sparse_results.pickle", 'rb') as f:
            sparse_results = pickle.load(f)

        reconstruct_sparse_eigenfunctions(results, sparse_results, N_plot=5)

        # Create comparison plots with boundary condition info
        print("\n" + "="*80)
        print("Creating comparison plots...")
        print("="*80)
        plot_comparison_with_boundary_check(results, sparse_results)

    except FileNotFoundError:
        print("\nNo sparse results found. Run varimax_sparse_rotation.py first.")

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
