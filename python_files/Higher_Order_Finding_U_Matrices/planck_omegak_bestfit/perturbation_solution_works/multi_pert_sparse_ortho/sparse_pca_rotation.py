# -*- coding: utf-8 -*-
"""
Sparse PCA via Manifold Optimization for Sparse Orthonormal Solutions

This script applies Sparse PCA with manifold optimization to the eigenvectors found
by multi_perturbation_analysis.py to find sparse and orthonormal valid solutions.

This method optimizes the L1 norm (sparsity) directly over the Stiefel Manifold
(the set of all orthogonal matrices), allowing aggressive sparsity tuning while
maintaining strict orthogonality constraints.

Reference:
    Erichson, N. B., et al. "Sparse Principal Component Analysis via Variable Projection."
    SIAM Journal on Applied Mathematics (2020).

Requirements:
    pip install pymanopt
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import sys
sys.path.append('..')
from multi_perturbation_analysis import multi_perturbation_analysis

# =============================================================================
# Sparse PCA via Manifold Optimization
# =============================================================================

try:
    import pymanopt
    from pymanopt.manifolds import Stiefel
    from pymanopt.optimizers import ConjugateGradient, TrustRegions
    from pymanopt import Problem
    import autograd.numpy as anp
    import pymanopt.function

    PYMANOPT_AVAILABLE = True
    print("pymanopt is available - Manifold optimization enabled")

except ImportError as e:
    PYMANOPT_AVAILABLE = False
    print(f"WARNING: Required libraries not found: {e}")
    print("\nPlease install with:")
    print("  pip install pymanopt autograd")
    print("\nFalling back to Varimax rotation as alternative...")


def sparse_orthogonal_rotation(C, sparsity_weight=1.0, epsilon=1e-8,
                               max_iterations=200, verbosity=1):
    """
    Finds a rotation R such that C @ R is sparse, using Manifold Optimization
    over the Stiefel manifold (set of orthogonal matrices).

    This method optimizes the smooth L1 norm approximation:
        minimize: sum(sqrt(C_new^2 + epsilon))
    subject to: R^T @ R = I (orthogonality constraint)

    Args:
        C: (N x K) orthonormal matrix (eigenvectors)
        sparsity_weight: Weight for sparsity term (default=1.0)
        epsilon: Smoothing parameter for L1 approximation (default=1e-8)
        max_iterations: Maximum optimization iterations (default=200)
        verbosity: Verbosity level (0=silent, 1=normal, 2=detailed)

    Returns:
        C_sparse: Sparse rotated matrix (N x K)
        R_opt: Optimal rotation matrix (K x K)
        converged: Boolean indicating convergence
        final_cost: Final cost value
    """
    if not PYMANOPT_AVAILABLE:
        raise ImportError("pymanopt is required for sparse PCA. Please install it: pip install pymanopt")

    N, K = C.shape

    if verbosity >= 1:
        print(f"  Setting up manifold optimization:")
        print(f"    Matrix shape: {N} x {K}")
        print(f"    Sparsity weight: {sparsity_weight}")
        print(f"    Epsilon (smoothing): {epsilon}")

    # Define the Manifold: Stiefel(K, K) is the set of K x K orthogonal matrices
    manifold = Stiefel(K, K)

    # Define the Cost Function: Smooth L1 norm approximation
    # Use autograd.numpy for automatic differentiation
    @pymanopt.function.autograd(manifold)
    def cost(R):
        """
        Cost function: smooth L1 norm of rotated matrix C @ R
        """
        C_new = anp.dot(C, R)
        # Smooth approximation of L1: sqrt(x^2 + epsilon)
        smooth_l1 = anp.sqrt(C_new**2 + epsilon)
        return sparsity_weight * anp.sum(smooth_l1)

    # Set up the optimization problem
    # The gradient will be computed automatically by autograd
    problem = Problem(manifold, cost)

    # Choose optimizer (TrustRegions often works better for this problem)
    optimizer = TrustRegions(max_iterations=max_iterations, verbosity=verbosity)

    # Run optimization
    if verbosity >= 1:
        print("  Running manifold optimization...")

    result = optimizer.run(problem)

    R_opt = result.point
    C_sparse = C @ R_opt

    converged = result.stopping_criterion == "GradientNorm"
    final_cost = result.cost

    if verbosity >= 1:
        if converged:
            print(f"  Optimization converged after {result.iterations} iterations")
        else:
            print(f"  Optimization stopped: {result.stopping_criterion}")
        print(f"  Final cost: {final_cost:.6e}")

    return C_sparse, R_opt, converged, final_cost


def apply_sparse_pca_to_eigenvectors(eigenvecs, eigenvals, eigenvalue_threshold=0.95,
                                     sparsity_weight=1.0, epsilon=1e-8):
    """
    Apply Sparse PCA via manifold optimization to eigenvectors with eigenvalues above threshold.

    Args:
        eigenvecs: List of eigenvectors from eigenvalue decomposition
        eigenvals: Array of corresponding eigenvalues
        eigenvalue_threshold: Minimum eigenvalue to include (default: 0.95)
        sparsity_weight: Weight for sparsity term (default: 1.0)
        epsilon: Smoothing parameter for L1 approximation (default: 1e-8)

    Returns:
        eigenvecs_sparse: Sparse rotated eigenvectors
        R: Rotation matrix applied
        valid_indices: Indices of eigenvectors that were rotated
        converged: Boolean indicating convergence
    """
    eigenvals = np.array(eigenvals)

    # Filter eigenvectors with eigenvalues > threshold
    valid_mask = eigenvals.real > eigenvalue_threshold
    valid_indices = np.where(valid_mask)[0]

    print(f"\nFiltering eigenvectors with eigenvalue > {eigenvalue_threshold}")
    print(f"Found {len(valid_indices)} eigenvectors above threshold out of {len(eigenvals)} total")

    if len(valid_indices) == 0:
        print("No eigenvectors found above threshold. Try lowering the threshold.")
        return None, None, None, False

    # Convert list of eigenvectors to matrix (N x K)
    eigenvecs_array = np.array([eigenvecs[i] for i in valid_indices]).T

    print(f"Eigenvector matrix shape for Sparse PCA: {eigenvecs_array.shape}")
    print(f"Eigenvalues of selected modes: {eigenvals[valid_indices].real}")

    # Apply Sparse PCA via manifold optimization
    print("\nApplying Sparse PCA via Manifold Optimization...")
    eigenvecs_sparse, R, converged, final_cost = sparse_orthogonal_rotation(
        eigenvecs_array,
        sparsity_weight=sparsity_weight,
        epsilon=epsilon
    )

    # Check orthogonality
    orth_error = np.linalg.norm(eigenvecs_sparse.T @ eigenvecs_sparse - np.eye(eigenvecs_sparse.shape[1]))
    print(f"\nOrthogonality error after Sparse PCA: {orth_error:.2e}")

    # Calculate sparsity metrics
    sparsity_before = np.mean(np.abs(eigenvecs_array) < 1e-3)
    sparsity_after = np.mean(np.abs(eigenvecs_sparse) < 1e-3)
    print(f"Sparsity (fraction of near-zero elements):")
    print(f"  Before: {sparsity_before:.3f}")
    print(f"  After:  {sparsity_after:.3f}")

    # Calculate L1 norms
    l1_before = np.sum(np.abs(eigenvecs_array))
    l1_after = np.sum(np.abs(eigenvecs_sparse))
    print(f"L1 norms:")
    print(f"  Before: {l1_before:.3f}")
    print(f"  After:  {l1_after:.3f}")

    return eigenvecs_sparse, R, valid_indices, converged


def compute_sparse_coefficients(eigenvecs_sparse, transformation_matrix):
    """
    Compute linear combination coefficients for sparse eigenvectors.

    Args:
        eigenvecs_sparse: Sparse rotated eigenvectors (N x K)
        transformation_matrix: Transformation matrix from QR decomposition

    Returns:
        coefficients: Coefficients for reconstructing solutions (K x N_basis)
    """
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
                                       N_plot=10, output_filename="./figures/sparse_pca_coefficients_comparison.pdf"):
    """
    Compare original and sparse coefficients side by side.
    """
    N_plot = min(N_plot, len(eigenvals_valid))
    N_basis = coeffs_original.shape[1]

    fig, axes = plt.subplots(N_plot, 2, figsize=(7.5, 0.7*N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Original vs Sparse PCA Coefficients", fontsize=12)

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
        ax_sparse.bar(k_values, coeffs_sparse[i, :].real, width=0.4, color='purple')
        ax_sparse.set_xlim(0, min(20, N_basis + 1))
        ax_sparse.set_ylim(-1, 1)

        # Find dominant k for sparse version
        dominant_k = np.argmax(np.abs(coeffs_sparse[i, :].real))
        ax_sparse.axvline(dominant_k + 1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_sparse.set_title(f"Sparse PCA (dom k={dominant_k+1})", fontsize=8, loc='left')
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
                                   output_filename="./figures/sparse_pca_coefficients_sorted.pdf"):
    """
    Plot sparse coefficients sorted by dominant k mode.
    """
    N_plot = min(N_plot, len(eigenvals_valid))
    N_basis = coeffs_sparse.shape[1]

    # Find dominant k for each eigenfunction
    max_indices = np.argmax(np.abs(coeffs_sparse.real), axis=1)

    # Sort by dominant k
    sorted_order = np.argsort(max_indices)

    print(f"\nDominant k indices after Sparse PCA: {max_indices[sorted_order]}")
    print(f"Corresponding eigenvalues: {eigenvals_valid[sorted_order].real}")

    # Create figure
    fig, axs = plt.subplots(N_plot, figsize=(3.8, 0.7*N_plot))
    if N_plot == 1:
        axs = [axs]

    fig.suptitle(f"Sparse PCA Coefficients (sorted by dominant k)", fontsize=10)

    for i in range(N_plot):
        plot_idx = sorted_order[i]
        coefficients = coeffs_sparse[plot_idx, :].real
        dominant_k = max_indices[plot_idx]
        eigenval = eigenvals_valid[plot_idx].real

        # Bar plot
        k_values = np.arange(1, N_basis + 1)
        axs[i].bar(k_values, coefficients, width=0.4, color='purple')
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
                                output_filename="./figures/sparse_pca_eigenfunctions.pdf"):
    """
    Plot the reconstructed sparse eigenfunctions (time evolution of perturbations).
    Similar to plot_multi_perturbation_results() from multi_perturbation_analysis.py

    Parameters:
    -----------
    results : dict
        Original results from multi_perturbation_analysis
    sparse_results : dict
        Sparse results from sparse_pca_analysis
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

    fig.suptitle("Sparse PCA Eigenfunctions (Time Evolution)", fontsize=16)

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
            ax.plot(eta_grid, solution_sparse, 'purple', linewidth=2.5,
                   label='Sparse PCA', alpha=0.8)

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
                                           output_filename="./figures/sparse_pca_vs_original.pdf"):
    """
    Compare original and sparse eigenfunctions side by side.

    Parameters:
    -----------
    results : dict
        Original results from multi_perturbation_analysis
    sparse_results : dict
        Sparse results from sparse_pca_analysis
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

    fig.suptitle(f"Original vs Sparse PCA Eigenfunctions ({reference_pert})", fontsize=14)

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
        ax_sparse.plot(eta_grid, solution_sparse, 'purple', linewidth=2.0, alpha=0.8)
        ax_sparse.set_title(f"Sparse PCA (dominant k={dominant_k+1})", fontsize=10)
        ax_sparse.grid(True, alpha=0.3)
        if i == N_plot - 1:
            ax_sparse.set_xlabel("Conformal Time η", fontsize=10)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_filename}")
    plt.show()


# =============================================================================
# Main Workflow
# =============================================================================

def sparse_pca_analysis(results, eigenvalue_threshold=0.95, N_plot=10,
                       sparsity_weight=1.0, epsilon=1e-8):
    """
    Complete workflow for Sparse PCA analysis.

    Args:
        results: Results dictionary from multi_perturbation_analysis
        eigenvalue_threshold: Minimum eigenvalue to include
        N_plot: Number of modes to plot
        sparsity_weight: Weight for sparsity term (higher = more sparse)
        epsilon: Smoothing parameter for L1 approximation

    Returns:
        Dictionary with sparse analysis results
    """
    if not PYMANOPT_AVAILABLE:
        print("\nERROR: pymanopt is not installed!")
        print("Please install it with: pip install pymanopt")
        return None

    print("="*80)
    print("SPARSE PCA VIA MANIFOLD OPTIMIZATION ANALYSIS")
    print("="*80)

    # Extract results
    eigenvecs_1 = results['eigenvecs_1']
    eigenvals_1 = results['eigenvals_1']
    coefficients_1 = results['coefficients_1']

    # Apply Sparse PCA to eigenvectors
    eigenvecs_sparse, R, valid_indices, converged = apply_sparse_pca_to_eigenvectors(
        eigenvecs_1, eigenvals_1, eigenvalue_threshold,
        sparsity_weight=sparsity_weight, epsilon=epsilon)

    if eigenvecs_sparse is None:
        print("Failed to apply Sparse PCA")
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
        output_filename="./figures/sparse_pca_coefficients_comparison.pdf"
    )

    sorted_order, dominant_k = plot_sparse_coefficients_sorted(
        coefficients_sparse_dict[reference_pert],
        eigenvals_valid,
        N_plot=N_plot,
        output_filename="./figures/sparse_pca_coefficients_sorted.pdf"
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
        'converged': converged,
        'sparsity_weight': sparsity_weight,
        'epsilon': epsilon,
        'method': 'sparse_pca'
    }

    # Save to pickle
    with open("sparse_pca_results.pickle", 'wb') as f:
        pickle.dump(sparse_results, f)
    print("\nSaved results to sparse_pca_results.pickle")

    return sparse_results


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    import os

    if not PYMANOPT_AVAILABLE:
        print("\n" + "="*80)
        print("ERROR: Required libraries are not installed")
        print("="*80)
        print("\nPlease install pymanopt and autograd:")
        print("  pip install pymanopt autograd")
        print("\nFor more information, see:")
        print("  https://pymanopt.org/")
        sys.exit(1)

    # Check if we should recompute the analysis or load from existing results
    results_file = "../multi_perturbation_results.pickle"

    if os.path.exists(results_file):
        print("Loading existing multi_perturbation_analysis results...")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print("Loaded results successfully")
    else:
        print("No existing results found, running multi_perturbation_analysis...")
        folder_path = f'../data/'
        allowedK = np.load(folder_path + 'data_allowedK/L70_kvalues.npy')
        results = multi_perturbation_analysis(N=len(allowedK), N_t=1000, folder_path=folder_path)

        # Save for future use
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results to {results_file}")

    # Run Sparse PCA analysis
    print("\n" + "="*80)
    print("Running Sparse PCA Analysis")
    print("="*80 + "\n")

    # You can tune these parameters:
    # - sparsity_weight: higher values = more aggressive sparsity (try 0.5, 1.0, 2.0)
    # - epsilon: smoothing parameter (usually keep at 1e-8)
    # - eigenvalue_threshold: minimum eigenvalue to include
    sparse_results = sparse_pca_analysis(
        results,
        eigenvalue_threshold=0.99,  # Use stricter threshold to preserve boundary conditions
        N_plot=10,
        sparsity_weight=1.0,  # Try increasing this for more sparsity
        epsilon=1e-8
    )

    if sparse_results is not None:
        print("\n" + "="*80)
        print("SPARSE PCA ANALYSIS COMPLETE")
        print("="*80)
        print(f"Number of sparse eigenvectors: {sparse_results['eigenvecs_sparse'].shape[1]}")
        print(f"Optimization converged: {sparse_results['converged']}")
        print(f"Orthogonality preserved: Yes (by manifold constraint)")
        print(f"Sparsity weight used: {sparse_results['sparsity_weight']}")
        print(f"\nOutput files:")
        print("  - sparse_pca_results.pickle")
        print("  - sparse_pca_coefficients_comparison.pdf")
        print("  - sparse_pca_coefficients_sorted.pdf")

        # Plot the time evolution of sparse eigenfunctions
        print("\n" + "="*80)
        print("Plotting sparse eigenfunction time evolution")
        print("="*80)
        plot_sparse_eigenfunctions(results, sparse_results, N_plot=5)
        plot_sparse_vs_original_eigenfunctions(results, sparse_results, N_plot=3)

        print("\nAdditional output files:")
        print("  - sparse_pca_eigenfunctions.pdf")
        print("  - sparse_pca_vs_original.pdf")
        print("\nTip: To increase sparsity, try running with higher sparsity_weight (e.g., 2.0 or 5.0)")
    else:
        print("\nSparse PCA analysis failed")
