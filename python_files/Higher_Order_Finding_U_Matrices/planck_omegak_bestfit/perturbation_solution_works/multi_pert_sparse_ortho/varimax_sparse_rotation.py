# -*- coding: utf-8 -*-
"""
Varimax Rotation for Sparse Orthonormal Solutions

This script applies Varimax rotation to the eigenvectors found by multi_perturbation_analysis.py
to find sparse and orthonormal valid solutions while maintaining strict orthogonality.

Varimax rotation maximizes the "kurtosis" (spikiness) of the vectors, corresponding to
localized wave numbers in the physics context.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import sys
sys.path.append('..')
from multi_perturbation_analysis import multi_perturbation_analysis

# =============================================================================
# Varimax Rotation Implementation
# =============================================================================

def varimax_rotation(Phi, gamma=1.0, q=100, tol=1e-8):
    """
    Rotates a matrix Phi (N x K) by a rotation matrix R (K x K)
    to maximize the sparsity of Phi * R using Varimax criterion.

    This is the standard method in physics and chemistry (often called
    "localized orbitals" in quantum mechanics). It finds a rotation that
    maximizes the "kurtosis" (spikiness) of the vectors.

    Args:
        Phi: The N x K matrix of valid solutions (eigenvectors).
        gamma: Simplicity parameter (default=1.0 for standard Varimax)
        q: Maximum number of iterations (default=100)
        tol: Convergence tolerance (default=1e-8)

    Returns:
        Phi_rotated: The sparse, orthonormal solutions (N x K)
        R: The rotation matrix (K x K)
        converged: Boolean indicating convergence
        iterations: Number of iterations performed
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0

    for i in range(q):
        d_old = d
        # Calculate the variance of the squared elements (simplicity criterion)
        Lambda = np.dot(Phi, R)

        # Compute the gradient for Varimax criterion
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))
            )
        )
        R = np.dot(u, vh)  # Enforces strict orthogonality
        d = np.sum(s)

        # Check convergence
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            print(f"  Varimax converged after {i+1} iterations")
            return np.dot(Phi, R), R, True, i+1

    print(f"  Varimax reached max iterations ({q})")
    return np.dot(Phi, R), R, False, q


def apply_varimax_to_eigenvectors(eigenvecs, eigenvals, eigenvalue_threshold=0.95):
    """
    Apply Varimax rotation to eigenvectors with eigenvalues above threshold.

    Args:
        eigenvecs: List of eigenvectors from eigenvalue decomposition
        eigenvals: Array of corresponding eigenvalues
        eigenvalue_threshold: Minimum eigenvalue to include (default: 0.95)

    Returns:
        eigenvecs_sparse: Sparse rotated eigenvectors
        R: Rotation matrix applied
        valid_indices: Indices of eigenvectors that were rotated
    """
    eigenvals = np.array(eigenvals)

    # Filter eigenvectors with eigenvalues > threshold
    valid_mask = eigenvals.real > eigenvalue_threshold
    valid_indices = np.where(valid_mask)[0]

    print(f"\nFiltering eigenvectors with eigenvalue > {eigenvalue_threshold}")
    print(f"Found {len(valid_indices)} eigenvectors above threshold out of {len(eigenvals)} total")

    if len(valid_indices) == 0:
        print("No eigenvectors found above threshold. Try lowering the threshold.")
        return None, None, None

    # Convert list of eigenvectors to matrix (N x K)
    # Each eigenvector is a column
    eigenvecs_array = np.array([eigenvecs[i] for i in valid_indices]).T

    print(f"Eigenvector matrix shape for Varimax: {eigenvecs_array.shape}")
    print(f"Eigenvalues of selected modes: {eigenvals[valid_indices].real}")

    # Apply Varimax rotation
    print("\nApplying Varimax rotation...")
    eigenvecs_sparse, R, converged, iterations = varimax_rotation(eigenvecs_array)

    # Check orthogonality
    orth_error = np.linalg.norm(eigenvecs_sparse.T @ eigenvecs_sparse - np.eye(eigenvecs_sparse.shape[1]))
    print(f"Orthogonality error after Varimax: {orth_error:.2e}")

    # Calculate sparsity metrics
    sparsity_before = np.mean(np.abs(eigenvecs_array) < 1e-3)
    sparsity_after = np.mean(np.abs(eigenvecs_sparse) < 1e-3)
    print(f"Sparsity (fraction of near-zero elements):")
    print(f"  Before: {sparsity_before:.3f}")
    print(f"  After:  {sparsity_after:.3f}")

    return eigenvecs_sparse, R, valid_indices


def compute_sparse_coefficients(eigenvecs_sparse, transformation_matrix):
    """
    Compute linear combination coefficients for sparse eigenvectors.

    Args:
        eigenvecs_sparse: Sparse rotated eigenvectors (N x K)
        transformation_matrix: Transformation matrix from QR decomposition

    Returns:
        coefficients: Coefficients for reconstructing solutions (K x N_basis)
    """
    # eigenvecs_sparse is N x K, each column is a sparse eigenvector
    # transformation_matrix is N x N_basis
    # We want: coefficients[i, :] = eigenvecs_sparse[:, i].T @ transformation_matrix

    K = eigenvecs_sparse.shape[1]
    N_basis = transformation_matrix.shape[1]
    coefficients = np.zeros((K, N_basis), dtype=float)

    for i in range(K):
        coefficients[i, :] = np.dot(eigenvecs_sparse[:, i], transformation_matrix)

    return coefficients


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_sparse_coefficients_comparison(coeffs_original, coeffs_sparse, eigenvals_valid,
                                       N_plot=10, output_filename="./figures/varimax_sparse_coefficients_comparison.pdf"):
    """
    Compare original and sparse coefficients side by side.
    """
    N_plot = min(N_plot, len(eigenvals_valid))
    N_basis = coeffs_original.shape[1]

    fig, axes = plt.subplots(N_plot, 2, figsize=(7.5, 0.7*N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Original vs Varimax Sparse Coefficients", fontsize=12)

    for i in range(N_plot):
        # Original coefficients
        ax_orig = axes[i, 0]
        k_values = np.arange(1, N_basis + 1)
        ax_orig.bar(k_values, coeffs_original[i, :].real, width=0.4)
        ax_orig.set_xlim(0, min(20, N_basis + 1))
        ax_orig.set_ylim(-1, 1)
        ax_orig.set_title(f"Original (λ={eigenvals_valid[i]:.3f})", fontsize=8, loc='left')
        ax_orig.grid(True, alpha=0.3, axis='y')

        # Sparse coefficients
        ax_sparse = axes[i, 1]
        ax_sparse.bar(k_values, coeffs_sparse[i, :].real, width=0.4, color='green')
        ax_sparse.set_xlim(0, min(20, N_basis + 1))
        ax_sparse.set_ylim(-1, 1)

        # Find dominant k for sparse version
        dominant_k = np.argmax(np.abs(coeffs_sparse[i, :].real))
        ax_sparse.axvline(dominant_k + 1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_sparse.set_title(f"Varimax (dom k={dominant_k+1})", fontsize=8, loc='left')
        ax_sparse.grid(True, alpha=0.3, axis='y')

        # Format
        for ax in [ax_orig, ax_sparse]:
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.label_outer()

    axes[-1, 0].set_xlabel(r"$k$ mode index", fontsize=8)
    axes[-1, 1].set_xlabel(r"$k$ mode index", fontsize=8)
    axes[-1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_filename}")
    plt.show()


def plot_sparse_coefficients_sorted(coeffs_sparse, eigenvals_valid, N_plot=10,
                                   output_filename="./figures/varimax_sparse_coefficients_sorted.pdf"):
    """
    Plot sparse coefficients sorted by dominant k mode.
    """
    N_plot = min(N_plot, len(eigenvals_valid))
    N_basis = coeffs_sparse.shape[1]

    # Find dominant k for each eigenfunction
    max_indices = np.argmax(np.abs(coeffs_sparse.real), axis=1)

    # Sort by dominant k
    sorted_order = np.argsort(max_indices)

    print(f"\nDominant k indices after Varimax: {max_indices[sorted_order]}")
    print(f"Corresponding eigenvalues: {eigenvals_valid[sorted_order].real}")

    # Create figure
    fig, axs = plt.subplots(N_plot, figsize=(3.8, 0.7*N_plot))
    if N_plot == 1:
        axs = [axs]

    fig.suptitle(f"Varimax Sparse Coefficients (sorted by dominant k)", fontsize=10)

    for i in range(N_plot):
        plot_idx = sorted_order[i]
        coefficients = coeffs_sparse[plot_idx, :].real
        dominant_k = max_indices[plot_idx]
        eigenval = eigenvals_valid[plot_idx].real

        # Bar plot
        k_values = np.arange(1, N_basis + 1)
        axs[i].bar(k_values, coefficients, width=0.4, color='green')
        axs[i].set_xlim(0, min(20, N_basis + 1))
        axs[i].set_ylim(-1, 1)

        # Title and highlight
        axs[i].set_title(f"λ={eigenval:.3f}, dominant k={dominant_k+1}",
                        fontsize=8, loc='right')
        axs[i].axvline(dominant_k + 1, color='red', linestyle='--',
                      alpha=0.5, linewidth=1)

        # Format
        axs[i].xaxis.set_tick_params(labelsize=8)
        axs[i].yaxis.set_tick_params(labelsize=8)
        axs[i].label_outer()
        axs[i].grid(True, alpha=0.3, axis='y')

    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[-1].set_xlabel(r"$k$ mode index", fontsize=8)
    fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved sorted plot to {output_filename}")
    plt.show()

    return sorted_order, max_indices


def plot_sparse_eigenfunctions(results, sparse_results, N_plot=5,
                                output_filename="./figures/varimax_sparse_eigenfunctions.pdf"):
    """
    Plot the reconstructed sparse eigenfunctions (time evolution of perturbations).
    Similar to plot_multi_perturbation_results() from multi_perturbation_analysis.py

    Parameters:
    -----------
    results : dict
        Original results from multi_perturbation_analysis
    sparse_results : dict
        Sparse results from varimax_sparse_analysis
    N_plot : int
        Number of eigenfunctions to plot
    output_filename : str
        Output PDF filename
    """
    eta_grid = results['eta_grid']
    eigenvals_valid = sparse_results['eigenvals_valid']
    coefficients_sparse = sparse_results['coefficients_sparse']
    basis_1 = results['basis_1']

    N_plot = min(N_plot, len(eigenvals_valid))
    if N_plot == 0:
        print("No valid sparse eigenfunctions to plot")
        return

    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    n_pert = len(perturbation_types)

    fig, axes = plt.subplots(N_plot, n_pert, figsize=(16, 2*N_plot),
                           constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Varimax Sparse Eigenfunctions (Time Evolution)", fontsize=16)

    # Sort by dominant k for better visualization
    reference_pert = 'vr' if 'vr' in coefficients_sparse else list(coefficients_sparse.keys())[0]
    max_indices = np.argmax(np.abs(coefficients_sparse[reference_pert].real), axis=1)
    sorted_order = np.argsort(max_indices)

    for i in range(N_plot):
        plot_idx = sorted_order[i]

        for j, pert_type in enumerate(perturbation_types):
            ax = axes[i, j]

            if pert_type not in coefficients_sparse:
                ax.text(0.5, 0.5, f"No data\nfor {pert_type}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{pert_type} (eigenval {plot_idx+1})")
                continue

            # Reconstruct solution from sparse coefficients
            solution_sparse = np.zeros_like(eta_grid)
            coeffs = coefficients_sparse[pert_type][plot_idx, :]
            N_basis = len(coeffs)

            for k in range(N_basis):
                solution_sparse += coeffs[k] * basis_1[pert_type][:, k]

            # Plot
            ax.plot(eta_grid, solution_sparse, 'g-', linewidth=2.5,
                   label='Varimax Sparse', alpha=0.8)

            dominant_k = max_indices[plot_idx]
            ax.set_title(f"{pert_type} (λ={eigenvals_valid[plot_idx]:.3f}, k={dominant_k+1})",
                        fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend()
            if i == N_plot - 1:
                ax.set_xlabel("Conformal Time η", fontsize=10)
            if j == 0:
                ax.set_ylabel("Amplitude", fontsize=10)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved sparse eigenfunction plot to {output_filename}")
    plt.show()


def plot_sparse_vs_original_eigenfunctions(results, sparse_results, N_plot=3,
                                           output_filename="./figures/varimax_sparse_vs_original.pdf"):
    """
    Compare original and sparse eigenfunctions side by side.

    Parameters:
    -----------
    results : dict
        Original results from multi_perturbation_analysis
    sparse_results : dict
        Sparse results from varimax_sparse_analysis
    N_plot : int
        Number of eigenfunctions to plot
    output_filename : str
        Output PDF filename
    """
    eta_grid = results['eta_grid']
    eigenvals_valid = sparse_results['eigenvals_valid']
    valid_indices = sparse_results['valid_indices']
    coefficients_original = sparse_results['coefficients_original']
    coefficients_sparse = sparse_results['coefficients_sparse']
    basis_1 = results['basis_1']

    N_plot = min(N_plot, len(eigenvals_valid))
    if N_plot == 0:
        print("No valid eigenfunctions to compare")
        return

    # Use vr as reference
    reference_pert = 'vr' if 'vr' in coefficients_sparse else list(coefficients_sparse.keys())[0]

    fig, axes = plt.subplots(N_plot, 2, figsize=(12, 2.5*N_plot),
                           constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Original vs Varimax Sparse Eigenfunctions ({reference_pert})", fontsize=14)

    # Sort by dominant k
    max_indices = np.argmax(np.abs(coefficients_sparse[reference_pert].real), axis=1)
    sorted_order = np.argsort(max_indices)

    for i in range(N_plot):
        plot_idx = sorted_order[i]

        # Original eigenfunction
        ax_orig = axes[i, 0]
        solution_orig = np.zeros_like(eta_grid)
        coeffs_orig = coefficients_original[reference_pert][plot_idx, :]
        for k in range(len(coeffs_orig)):
            solution_orig += coeffs_orig[k] * basis_1[reference_pert][:, k]

        ax_orig.plot(eta_grid, solution_orig, 'r-', linewidth=2.0, alpha=0.8)
        ax_orig.set_title(f"Original (λ={eigenvals_valid[plot_idx]:.3f})", fontsize=10)
        ax_orig.grid(True, alpha=0.3)
        ax_orig.set_ylabel("Amplitude", fontsize=10)
        if i == N_plot - 1:
            ax_orig.set_xlabel("Conformal Time η", fontsize=10)

        # Sparse eigenfunction
        ax_sparse = axes[i, 1]
        solution_sparse = np.zeros_like(eta_grid)
        coeffs_sparse = coefficients_sparse[reference_pert][plot_idx, :]
        for k in range(len(coeffs_sparse)):
            solution_sparse += coeffs_sparse[k] * basis_1[reference_pert][:, k]

        dominant_k = max_indices[plot_idx]
        ax_sparse.plot(eta_grid, solution_sparse, 'g-', linewidth=2.0, alpha=0.8)
        ax_sparse.set_title(f"Varimax (dominant k={dominant_k+1})", fontsize=10)
        ax_sparse.grid(True, alpha=0.3)
        if i == N_plot - 1:
            ax_sparse.set_xlabel("Conformal Time η", fontsize=10)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_filename}")
    plt.show()


# =============================================================================
# Main Workflow
# =============================================================================

def varimax_sparse_analysis(results, eigenvalue_threshold=0.95, N_plot=10):
    """
    Complete workflow for Varimax sparse rotation analysis.

    Args:
        results: Results dictionary from multi_perturbation_analysis
        eigenvalue_threshold: Minimum eigenvalue to include
        N_plot: Number of modes to plot

    Returns:
        Dictionary with sparse analysis results
    """
    print("="*80)
    print("VARIMAX SPARSE ROTATION ANALYSIS")
    print("="*80)

    # Extract results
    eigenvecs_1 = results['eigenvecs_1']
    eigenvals_1 = results['eigenvals_1']
    coefficients_1 = results['coefficients_1']

    # Apply Varimax rotation to eigenvectors
    eigenvecs_sparse, R, valid_indices = apply_varimax_to_eigenvectors(
        eigenvecs_1, eigenvals_1, eigenvalue_threshold)

    if eigenvecs_sparse is None:
        print("Failed to apply Varimax rotation")
        return None

    # Get filtered eigenvalues
    eigenvals_valid = np.array(eigenvals_1)[valid_indices]

    # Compute sparse coefficients for each perturbation type
    coefficients_sparse_dict = {}
    coefficients_original_dict = {}

    # Use 'vr' as reference perturbation (or any available one)
    reference_pert = 'vr' if 'vr' in coefficients_1 else list(coefficients_1.keys())[0]

    print(f"\nComputing sparse coefficients using {reference_pert} transformation...")

    for pert_type in coefficients_1.keys():
        print(f"  Processing {pert_type}...")

        # Recompute QR decomposition to get transformation matrix
        # basis_1[pert_type] has shape (N_t, N_basis)
        basis_matrix = results['basis_1'][pert_type]
        Q, R_qr = np.linalg.qr(basis_matrix)
        # transformation_matrix is the inverse of R_qr^T which maps orthonormal basis back to original basis
        transform_matrix = np.linalg.inv(R_qr.T)  # Shape: (N_basis, N_basis)

        # Original coefficients (filtered to valid indices)
        coeffs_orig = coefficients_1[pert_type][valid_indices]
        coefficients_original_dict[pert_type] = coeffs_orig

        # Compute sparse coefficients
        # eigenvecs_sparse is N x K where N is number of orthonormal modes (23)
        # transform_matrix is N_basis x N_basis (23 x 23)
        # We want to express the sparse eigenvectors in terms of the original k-basis
        coeffs_sparse = compute_sparse_coefficients(eigenvecs_sparse, transform_matrix)
        coefficients_sparse_dict[pert_type] = coeffs_sparse

    # Plot comparison for reference perturbation
    print(f"\n--- Plotting results for {reference_pert} ---")

    plot_sparse_coefficients_comparison(
        coefficients_original_dict[reference_pert],
        coefficients_sparse_dict[reference_pert],
        eigenvals_valid,
        N_plot=N_plot,
        output_filename="./figures/varimax_sparse_coefficients_comparison.pdf"
    )

    sorted_order, dominant_k = plot_sparse_coefficients_sorted(
        coefficients_sparse_dict[reference_pert],
        eigenvals_valid,
        N_plot=N_plot,
        output_filename="./figures/varimax_sparse_coefficients_sorted.pdf"
    )

    # Save results
    sparse_results = {
        'eigenvecs_sparse': eigenvecs_sparse,
        'rotation_matrix': R,
        'valid_indices': valid_indices,
        'eigenvals_valid': eigenvals_valid,
        'coefficients_sparse': coefficients_sparse_dict,
        'coefficients_original': coefficients_original_dict,
        'sorted_order': sorted_order,
        'dominant_k': dominant_k,
        'method': 'varimax'
    }

    # Save to pickle
    with open("varimax_sparse_results.pickle", 'wb') as f:
        pickle.dump(sparse_results, f)
    print("\nSaved results to varimax_sparse_results.pickle")

    return sparse_results


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    import os

    # Check if we should recompute the analysis or load from existing results
    results_file = "../multi_perturbation_results.pickle"

    if os.path.exists(results_file):
        print("Loading existing multi_perturbation_analysis results...")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print("Loaded results successfully")
    else:
        print("No existing results found, running multi_perturbation_analysis...")
        # Add the parent directory to path to import the analysis function
        folder_path = f'../data/'
        allowedK = np.load(folder_path + 'data_allowedK/L70_kvalues.npy')
        results = multi_perturbation_analysis(N=len(allowedK), N_t=1000, folder_path=folder_path)

        # Save for future use
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results to {results_file}")

    # Run Varimax sparse rotation analysis
    print("\n" + "="*80)
    print("Running Varimax Sparse Rotation Analysis")
    print("="*80 + "\n")

    sparse_results = varimax_sparse_analysis(
        results,
        eigenvalue_threshold=0.99,  # Use stricter threshold to preserve boundary conditions
        N_plot=10
    )

    if sparse_results is not None:
        print("\n" + "="*80)
        print("VARIMAX ANALYSIS COMPLETE")
        print("="*80)
        print(f"Number of sparse eigenvectors: {sparse_results['eigenvecs_sparse'].shape[1]}")
        print(f"Orthogonality preserved: Yes (by construction)")
        print(f"Output files:")
        print("  - varimax_sparse_results.pickle")
        print("  - varimax_sparse_coefficients_comparison.pdf")
        print("  - varimax_sparse_coefficients_sorted.pdf")

        # Plot the time evolution of sparse eigenfunctions
        print("\n" + "="*80)
        print("Plotting sparse eigenfunction time evolution")
        print("="*80)
        plot_sparse_eigenfunctions(results, sparse_results, N_plot=5)
        plot_sparse_vs_original_eigenfunctions(results, sparse_results, N_plot=3)

        print("\nAdditional output files:")
        print("  - varimax_sparse_eigenfunctions.pdf")
        print("  - varimax_sparse_vs_original.pdf")
    else:
        print("\nVarimax analysis failed")
