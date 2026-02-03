# -*- coding: utf-8 -*-
"""
Stacked Varimax Rotation for Joint Sparse Orthonormal Solutions

This script applies Stacked Varimax rotation to find solutions that are sparse
in BOTH Basis 1 and Basis 2 simultaneously. This creates a much tighter constraint
and should lead to unique physical modes.

The key idea: Instead of maximizing sparsity in just one basis, we stack the
coefficients from both bases and apply Varimax to the combined matrix. This finds
a single rotation R that simplifies BOTH representations simultaneously.

Based on varimax_sparse_rotation.py but extended to joint sparsification.
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

def varimax_rotation(Phi, gamma=1.0, q=500, tol=1e-6):
    """
    Robust Varimax with Random Initialization to avoid saddle points.

    Rotates a matrix Phi (N x K) by a rotation matrix R (K x K)
    to maximize the sparsity of Phi * R using Varimax criterion.

    This is the standard method in physics and chemistry (often called
    "localized orbitals" in quantum mechanics). It finds a rotation that
    maximizes the "kurtosis" (spikiness) of the vectors.

    Args:
        Phi: The N x K matrix of valid solutions (eigenvectors).
        gamma: Simplicity parameter (default=1.0 for standard Varimax)
        q: Maximum number of iterations (default=500)
        tol: Convergence tolerance (default=1e-6)

    Returns:
        Phi_rotated: The sparse, orthonormal solutions (N x K)
        R: The rotation matrix (K x K)
        converged: Boolean indicating convergence
        iterations: Number of iterations performed
    """
    p, k = Phi.shape

    # 1. Random Initialization (Crucial for pre-orthogonalized data)
    # Using a fixed seed ensures reproducibility while breaking symmetry
    rs = np.random.RandomState(42)
    M_rand = rs.randn(k, k)
    if np.iscomplexobj(Phi):
        M_rand = M_rand + 1j * rs.randn(k, k)
    Q, _ = np.linalg.qr(M_rand)
    R = Q

    d = 0
    for i in range(q):
        d_old = d
        B = np.dot(Phi, R)

        # 2. Gradient Calculation (Complex-safe)
        # We maximize sum(|B|^4) (Kurtosis/Sparsity)
        Msquared = np.abs(B)**2

        # Gradient of the objective function
        # For real: B^3; For complex: B * |B|^2
        Temp = B * Msquared

        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.conj().T,
                Temp - (gamma/p) * np.dot(B, np.diag(np.sum(Msquared, axis=0)))
            )
        )

        # 3. Enforce Orthogonality
        R = np.dot(u, vh)
        d = np.sum(s)

        if d_old != 0 and abs(d - d_old) / d_old < tol:
            print(f"  Stacked Varimax converged after {i+1} iterations")
            return np.dot(Phi, R), R, True, i+1

    print(f"  Stacked Varimax reached max iterations ({q})")
    return np.dot(Phi, R), R, False, q


def apply_stacked_varimax_to_eigenvectors(eigenvecs_1, eigenvecs_2, eigenvals_1,
                                          eigenvalue_threshold=0.95, weight_ratio=1.0):
    """
    Apply Stacked Varimax rotation to eigenvectors to achieve sparsity in BOTH bases.

    The key idea: We construct a stacked matrix containing coefficients from both
    Basis 1 and Basis 2, then apply a single Varimax rotation to the combined matrix.
    This finds the rotation that makes the solution sparse in both representations
    simultaneously.

    Args:
        eigenvecs_1: List of eigenvectors from Basis 1
        eigenvecs_2: List of eigenvectors from Basis 2 (must correspond to same solutions)
        eigenvals_1: Array of corresponding eigenvalues
        eigenvalue_threshold: Minimum eigenvalue to include (default: 0.95)
        weight_ratio: Relative weight of Basis 1 vs Basis 2 (default: 1.0 = equal weight)

    Returns:
        eigenvecs_sparse_1: Sparse rotated eigenvectors in Basis 1
        eigenvecs_sparse_2: Sparse rotated eigenvectors in Basis 2
        R: Rotation matrix applied
        valid_indices: Indices of eigenvectors that were rotated
    """
    eigenvals = np.array(eigenvals_1)

    # Filter eigenvectors with eigenvalues > threshold
    valid_mask = eigenvals.real > eigenvalue_threshold
    valid_indices = np.where(valid_mask)[0]

    print(f"\nFiltering eigenvectors with eigenvalue > {eigenvalue_threshold}")
    print(f"Found {len(valid_indices)} eigenvectors above threshold out of {len(eigenvals)} total")

    if len(valid_indices) == 0:
        print("No eigenvectors found above threshold. Try lowering the threshold.")
        return None, None, None, None

    # Convert list of eigenvectors to matrices (N x K)
    # Each eigenvector is a column
    eigenvecs_array_1 = np.array([eigenvecs_1[i] for i in valid_indices]).T
    eigenvecs_array_2 = np.array([eigenvecs_2[i] for i in valid_indices]).T

    print(f"Eigenvector matrix shapes for Stacked Varimax:")
    print(f"  Basis 1: {eigenvecs_array_1.shape}")
    print(f"  Basis 2: {eigenvecs_array_2.shape}")
    print(f"Eigenvalues of selected modes: {eigenvals[valid_indices].real}")

    # Normalize Basis 2 eigenvectors to have equal weight in optimization
    # (Optional: they might already be normalized, but this ensures consistency)
    norms_2 = np.linalg.norm(eigenvecs_array_2, axis=0)
    eigenvecs_array_2_normalized = eigenvecs_array_2 / norms_2

    # Construct the stacked matrix S = [C; C_tilde]
    # We can weight them differently if desired
    # weight_ratio > 1 emphasizes Basis 1, < 1 emphasizes Basis 2
    S = np.vstack([
        np.sqrt(weight_ratio) * eigenvecs_array_1,
        eigenvecs_array_2_normalized / np.sqrt(weight_ratio)
    ])

    print(f"Stacked matrix shape: {S.shape}")

    # Apply Varimax rotation to the stacked matrix
    print("\nApplying Stacked Varimax rotation...")
    S_rotated, R, converged, iterations = varimax_rotation(S)

    # Extract the rotated components
    N1 = eigenvecs_array_1.shape[0]
    eigenvecs_sparse_1 = S_rotated[:N1, :] / np.sqrt(weight_ratio)
    eigenvecs_sparse_2 = S_rotated[N1:, :] * np.sqrt(weight_ratio)

    # Renormalize Basis 2 back
    eigenvecs_sparse_2 = eigenvecs_sparse_2 * norms_2

    # Check orthogonality (should be preserved in Basis 1)
    orth_error_1 = np.linalg.norm(eigenvecs_sparse_1.T @ eigenvecs_sparse_1 - np.eye(eigenvecs_sparse_1.shape[1]))
    print(f"\nOrthogonality error in Basis 1: {orth_error_1:.2e}")

    # Calculate sparsity metrics for both bases
    sparsity_before_1 = np.mean(np.abs(eigenvecs_array_1) < 1e-3)
    sparsity_after_1 = np.mean(np.abs(eigenvecs_sparse_1) < 1e-3)
    sparsity_before_2 = np.mean(np.abs(eigenvecs_array_2_normalized) < 1e-3)
    sparsity_after_2 = np.mean(np.abs(eigenvecs_sparse_2 / norms_2) < 1e-3)

    print(f"\nSparsity (fraction of near-zero elements):")
    print(f"  Basis 1 - Before: {sparsity_before_1:.3f}, After: {sparsity_after_1:.3f}")
    print(f"  Basis 2 - Before: {sparsity_before_2:.3f}, After: {sparsity_after_2:.3f}")

    return eigenvecs_sparse_1, eigenvecs_sparse_2, R, valid_indices


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

def plot_joint_sparse_coefficients(coeffs_1, coeffs_2, eigenvals_valid,
                                   N_plot=10, output_filename="./figures/stack_varimax_coefficients.pdf"):
    """
    Plot sparse coefficients for both bases side by side.

    This visualization shows that the solutions are sparse in BOTH bases simultaneously.
    """
    N_plot = min(N_plot, len(eigenvals_valid))
    N_basis_1 = coeffs_1.shape[1]
    N_basis_2 = coeffs_2.shape[1]

    fig, axes = plt.subplots(N_plot, 2, figsize=(10, 0.7*N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Stacked Varimax: Joint Sparse Coefficients (Basis 1 vs Basis 2)", fontsize=12)

    # Find dominant modes in both bases
    max_indices_1 = np.argmax(np.abs(coeffs_1.real), axis=1)
    max_indices_2 = np.argmax(np.abs(coeffs_2.real), axis=1)

    # Sort by dominant k in Basis 1
    sorted_order = np.argsort(max_indices_1)

    for i in range(N_plot):
        plot_idx = sorted_order[i]

        # Basis 1 coefficients
        ax_1 = axes[i, 0]
        k_values_1 = np.arange(1, N_basis_1 + 1)
        ax_1.bar(k_values_1, coeffs_1[plot_idx, :].real, width=0.4, color='blue', alpha=0.7)
        ax_1.set_xlim(0, min(20, N_basis_1 + 1))
        ax_1.set_ylim(-1, 1)

        dominant_k_1 = max_indices_1[plot_idx]
        ax_1.axvline(dominant_k_1 + 1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_1.set_title(f"Basis 1: k={dominant_k_1+1} (λ={eigenvals_valid[plot_idx]:.3f})",
                      fontsize=8, loc='left')
        ax_1.grid(True, alpha=0.3, axis='y')

        # Basis 2 coefficients
        ax_2 = axes[i, 1]
        k_values_2 = np.arange(1, N_basis_2 + 1)
        ax_2.bar(k_values_2, coeffs_2[plot_idx, :].real, width=0.4, color='green', alpha=0.7)
        ax_2.set_xlim(0, min(20, N_basis_2 + 1))
        ax_2.set_ylim(-1, 1)

        dominant_k_2 = max_indices_2[plot_idx]
        ax_2.axvline(dominant_k_2 + 1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_2.set_title(f"Basis 2: k={dominant_k_2+1}", fontsize=8, loc='left')
        ax_2.grid(True, alpha=0.3, axis='y')

        # Format
        for ax in [ax_1, ax_2]:
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.label_outer()

    axes[-1, 0].set_xlabel(r"$k$ mode index (Basis 1)", fontsize=8)
    axes[-1, 1].set_xlabel(r"$k$ mode index (Basis 2)", fontsize=8)
    axes[-1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved joint sparse coefficients plot to {output_filename}")
    plt.show()


def plot_sparsity_comparison(coeffs_original_1, coeffs_original_2,
                            coeffs_sparse_1, coeffs_sparse_2,
                            eigenvals_valid, N_plot=5,
                            output_filename="./figures/stack_varimax_sparsity_comparison.pdf"):
    """
    Compare sparsity before and after Stacked Varimax for both bases.
    """
    N_plot = min(N_plot, len(eigenvals_valid))

    fig, axes = plt.subplots(N_plot, 4, figsize=(14, 2*N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Sparsity Comparison: Original vs Stacked Varimax (Both Bases)", fontsize=12)

    for i in range(N_plot):
        N_basis_1 = coeffs_original_1.shape[1]
        N_basis_2 = coeffs_original_2.shape[1]
        k_values_1 = np.arange(1, N_basis_1 + 1)
        k_values_2 = np.arange(1, N_basis_2 + 1)

        # Original Basis 1
        ax = axes[i, 0]
        ax.bar(k_values_1, coeffs_original_1[i, :].real, width=0.4, color='blue', alpha=0.5)
        ax.set_xlim(0, min(20, N_basis_1 + 1))
        ax.set_ylim(-1, 1)
        ax.set_title(f"Original B1 (λ={eigenvals_valid[i]:.3f})", fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Original Basis 2
        ax = axes[i, 1]
        ax.bar(k_values_2, coeffs_original_2[i, :].real, width=0.4, color='green', alpha=0.5)
        ax.set_xlim(0, min(20, N_basis_2 + 1))
        ax.set_ylim(-1, 1)
        ax.set_title(f"Original B2", fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Sparse Basis 1
        ax = axes[i, 2]
        dominant_k_1 = np.argmax(np.abs(coeffs_sparse_1[i, :].real))
        ax.bar(k_values_1, coeffs_sparse_1[i, :].real, width=0.4, color='blue', alpha=0.8)
        ax.axvline(dominant_k_1 + 1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlim(0, min(20, N_basis_1 + 1))
        ax.set_ylim(-1, 1)
        ax.set_title(f"Sparse B1 (k={dominant_k_1+1})", fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Sparse Basis 2
        ax = axes[i, 3]
        dominant_k_2 = np.argmax(np.abs(coeffs_sparse_2[i, :].real))
        ax.bar(k_values_2, coeffs_sparse_2[i, :].real, width=0.4, color='green', alpha=0.8)
        ax.axvline(dominant_k_2 + 1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlim(0, min(20, N_basis_2 + 1))
        ax.set_ylim(-1, 1)
        ax.set_title(f"Sparse B2 (k={dominant_k_2+1})", fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Format all axes
        for ax in axes[i, :]:
            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.label_outer()

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$k$ mode", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)
    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved sparsity comparison plot to {output_filename}")
    plt.show()


def plot_joint_sparsity_heatmap(coeffs_sparse_1, coeffs_sparse_2, eigenvals_valid,
                                output_filename="./figures/stack_varimax_heatmap.pdf"):
    """
    Create a heatmap showing which modes are dominant in both bases.
    This reveals the joint sparsity structure.
    """
    N_modes = len(eigenvals_valid)

    # Find dominant modes
    dominant_1 = np.argmax(np.abs(coeffs_sparse_1.real), axis=1)
    dominant_2 = np.argmax(np.abs(coeffs_sparse_2.real), axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap 1: Coefficients in Basis 1
    im1 = ax1.imshow(np.abs(coeffs_sparse_1.real[:min(N_modes, 15), :min(20, coeffs_sparse_1.shape[1])]),
                     aspect='auto', cmap='Blues', interpolation='nearest')
    ax1.set_xlabel('k mode index (Basis 1)', fontsize=10)
    ax1.set_ylabel('Eigenfunction index', fontsize=10)
    ax1.set_title('Sparse Coefficients in Basis 1', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='|Coefficient|')

    # Heatmap 2: Coefficients in Basis 2
    im2 = ax2.imshow(np.abs(coeffs_sparse_2.real[:min(N_modes, 15), :min(20, coeffs_sparse_2.shape[1])]),
                     aspect='auto', cmap='Greens', interpolation='nearest')
    ax2.set_xlabel('k mode index (Basis 2)', fontsize=10)
    ax2.set_ylabel('Eigenfunction index', fontsize=10)
    ax2.set_title('Sparse Coefficients in Basis 2', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='|Coefficient|')

    fig.suptitle('Joint Sparsity Structure from Stacked Varimax', fontsize=13, y=1.02)
    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved joint sparsity heatmap to {output_filename}")
    plt.show()


def plot_dominant_mode_correlation(coeffs_sparse_1, coeffs_sparse_2, eigenvals_valid,
                                   output_filename="./figures/stack_varimax_correlation.pdf"):
    """
    Plot the correlation between dominant modes in Basis 1 and Basis 2.
    If the stacked Varimax is working well, we should see a clear pattern.
    """
    dominant_1 = np.argmax(np.abs(coeffs_sparse_1.real), axis=1)
    dominant_2 = np.argmax(np.abs(coeffs_sparse_2.real), axis=1)

    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(dominant_1 + 1, dominant_2 + 1,
                        c=eigenvals_valid.real, s=100,
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add diagonal line for reference
    max_k = max(np.max(dominant_1), np.max(dominant_2)) + 1
    ax.plot([0, max_k], [0, max_k], 'r--', alpha=0.5, linewidth=1, label='Equal modes')

    ax.set_xlabel('Dominant k in Basis 1', fontsize=12)
    ax.set_ylabel('Dominant k in Basis 2', fontsize=12)
    ax.set_title('Correlation of Dominant Modes Between Bases', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    cbar = plt.colorbar(scatter, ax=ax, label='Eigenvalue')
    cbar.set_label('Eigenvalue', fontsize=11)

    # Add text annotation
    ax.text(0.05, 0.95, f'N={len(eigenvals_valid)} modes',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved dominant mode correlation plot to {output_filename}")
    plt.show()


def plot_sparse_eigenfunctions(results, sparse_results, N_plot=5,
                                output_filename="./figures/stack_varimax_sparse_eigenfunctions.pdf"):
    """
    Plot the reconstructed sparse eigenfunctions (time evolution of perturbations).
    This shows solutions sparse in BOTH bases.

    Parameters:
    -----------
    results : dict
        Original results from multi_perturbation_analysis
    sparse_results : dict
        Sparse results from stacked_varimax_sparse_analysis
    N_plot : int
        Number of eigenfunctions to plot
    output_filename : str
        Output PDF filename
    """
    eta_grid = results['eta_grid']
    eigenvals_valid = sparse_results['eigenvals_valid']
    coefficients_sparse_1 = sparse_results['coefficients_sparse_1']
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

    fig.suptitle("Stacked Varimax Sparse Eigenfunctions (Time Evolution - Basis 1)", fontsize=16)

    # Sort by dominant k for better visualization
    reference_pert = 'vr' if 'vr' in coefficients_sparse_1 else list(coefficients_sparse_1.keys())[0]
    max_indices = np.argmax(np.abs(coefficients_sparse_1[reference_pert].real), axis=1)
    sorted_order = np.argsort(max_indices)

    for i in range(N_plot):
        plot_idx = sorted_order[i]

        for j, pert_type in enumerate(perturbation_types):
            ax = axes[i, j]

            if pert_type not in coefficients_sparse_1:
                ax.text(0.5, 0.5, f"No data\nfor {pert_type}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{pert_type} (eigenval {plot_idx+1})")
                continue

            # Reconstruct solution from sparse coefficients
            solution_sparse = np.zeros_like(eta_grid)
            coeffs = coefficients_sparse_1[pert_type][plot_idx, :]
            N_basis = len(coeffs)

            for k in range(N_basis):
                solution_sparse += coeffs[k] * basis_1[pert_type][:, k]

            # Plot
            ax.plot(eta_grid, solution_sparse, 'g-', linewidth=2.5,
                   label='Stacked Varimax Sparse', alpha=0.8)

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
                                           output_filename="./figures/stack_varimax_sparse_vs_original.pdf"):
    """
    Compare original and sparse eigenfunctions side by side for both bases.

    Parameters:
    -----------
    results : dict
        Original results from multi_perturbation_analysis
    sparse_results : dict
        Sparse results from stacked_varimax_sparse_analysis
    N_plot : int
        Number of eigenfunctions to plot
    output_filename : str
        Output PDF filename
    """
    eta_grid = results['eta_grid']
    eigenvals_valid = sparse_results['eigenvals_valid']
    coefficients_original_1 = sparse_results['coefficients_original_1']
    coefficients_sparse_1 = sparse_results['coefficients_sparse_1']
    basis_1 = results['basis_1']

    N_plot = min(N_plot, len(eigenvals_valid))
    if N_plot == 0:
        print("No valid eigenfunctions to compare")
        return

    # Use vr as reference
    reference_pert = 'vr' if 'vr' in coefficients_sparse_1 else list(coefficients_sparse_1.keys())[0]

    fig, axes = plt.subplots(N_plot, 2, figsize=(12, 2.5*N_plot),
                           constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Original vs Stacked Varimax Sparse Eigenfunctions ({reference_pert})", fontsize=14)

    # Sort by dominant k
    max_indices = np.argmax(np.abs(coefficients_sparse_1[reference_pert].real), axis=1)
    sorted_order = np.argsort(max_indices)

    for i in range(N_plot):
        plot_idx = sorted_order[i]

        # Original eigenfunction
        ax_orig = axes[i, 0]
        solution_orig = np.zeros_like(eta_grid)
        coeffs_orig = coefficients_original_1[reference_pert][plot_idx, :]
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
        coeffs_sparse = coefficients_sparse_1[reference_pert][plot_idx, :]
        for k in range(len(coeffs_sparse)):
            solution_sparse += coeffs_sparse[k] * basis_1[reference_pert][:, k]

        dominant_k = max_indices[plot_idx]
        ax_sparse.plot(eta_grid, solution_sparse, 'g-', linewidth=2.0, alpha=0.8)
        ax_sparse.set_title(f"Stacked Varimax (dominant k={dominant_k+1})", fontsize=10)
        ax_sparse.grid(True, alpha=0.3)
        if i == N_plot - 1:
            ax_sparse.set_xlabel("Conformal Time η", fontsize=10)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_filename}")
    plt.show()


# =============================================================================
# Main Workflow
# =============================================================================

def stacked_varimax_sparse_analysis(results, eigenvalue_threshold=0.95,
                                   weight_ratio=1.0, N_plot=10):
    """
    Complete workflow for Stacked Varimax sparse rotation analysis (CORRECTED).

    This finds solutions that are sparse in BOTH bases simultaneously by
    applying Varimax to a stacked matrix containing coefficients from both bases.

    Args:
        results: Results dictionary from multi_perturbation_analysis
        eigenvalue_threshold: Minimum eigenvalue to include
        weight_ratio: Relative weight of Basis 1 vs Basis 2 (1.0 = equal)
        N_plot: Number of modes to plot

    Returns:
        Dictionary with stacked sparse analysis results
    """
    print("\n" + "="*80)
    print("STACKED VARIMAX SPARSE ROTATION ANALYSIS (CORRECTED)")
    print("="*80)

    # 1. Get Basis 1 Eigenvectors
    eigenvecs_1 = results['eigenvecs_1']
    eigenvals_1 = np.array(results['eigenvals_1'])

    # 2. Get the Mixing Matrix M
    # We need M to correctly project C -> C_tilde
    if 'M_combined' in results:
        M = results['M_combined']
    else:
        # Fallback: Recompute M if not saved (assuming M_combined logic from your code)
        print("M_combined not found in results, attempting to reconstruct...")
        # (You might need to pass M explicitly if it's not in the pickle)
        raise ValueError("Results dictionary must contain 'M_combined' matrix!")

    # 3. Filter Valid Modes
    valid_mask = eigenvals_1.real > eigenvalue_threshold
    valid_indices = np.where(valid_mask)[0]
    K = len(valid_indices)

    print(f"Filtering eigenvalues > {eigenvalue_threshold}")
    print(f"Found {K} valid modes.")

    if K == 0:
        return None

    # --- CRITICAL FIX START ---

    # Get Coefficients for Basis 1 (C)
    # Shape (N, K)
    C = np.array([eigenvecs_1[i] for i in valid_indices]).T

    # Calculate Coefficients for Basis 2 (C_tilde) via PROJECTION
    # Do NOT use eigenvecs_2 from the file, as they are not phase-aligned.
    # Formula: C_tilde = M^dagger * C  (divided by singular value sigma approx 1)

    # Note: Since C are eigenvectors of MM^dagger with val ~ 1,
    # M^dagger C gives us the corresponding vectors in Basis 2 space.
    C_tilde_raw = M.conj().T @ C

    # Normalize C_tilde columns to ensure unitary behavior in the stack
    norms_tilde = np.linalg.norm(C_tilde_raw, axis=0)
    C_tilde = C_tilde_raw / norms_tilde

    print("Calculated C_tilde via projection (M^H @ C) to ensure physical alignment.")

    # --- CRITICAL FIX END ---

    # 4. Construct Stack
    # S = [ sqrt(w)*C ;  1/sqrt(w)*C_tilde ]
    w_sqrt = np.sqrt(weight_ratio)
    S = np.vstack([w_sqrt * C, C_tilde / w_sqrt])

    print(f"Stacked matrix shape: {S.shape}")

    # 5. Apply Varimax
    print("\nApplying Stacked Varimax rotation...")
    S_rotated, R, converged, iterations = varimax_rotation(S)

    # 6. Unstack
    N1 = C.shape[0]
    eigenvecs_sparse_1 = S_rotated[:N1, :] / w_sqrt
    eigenvecs_sparse_2 = S_rotated[N1:, :] * w_sqrt

    # Re-normalize Basis 2 for output consistency
    eigenvecs_sparse_2 = eigenvecs_sparse_2 * norms_tilde

    # 7. Metrics
    orth_error = np.linalg.norm(eigenvecs_sparse_1.T @ eigenvecs_sparse_1 - np.eye(K))
    print(f"Orthogonality error: {orth_error:.2e}")

    # Check alignment of dominant modes
    dom_1 = np.argmax(np.abs(eigenvecs_sparse_1), axis=0)
    dom_2 = np.argmax(np.abs(eigenvecs_sparse_2), axis=0)

    # Note: In physics, k_index does not always equal omega_index.
    # But for high k, they should be correlated.
    print(f"Mode alignment correlation: {np.corrcoef(dom_1, dom_2)[0,1]:.4f}")

    # --- Recompute Reconstruction Coefficients ---
    print("\nComputing sparse coefficients for both bases...")

    coefficients_sparse_1_dict = {}
    coefficients_original_1_dict = {}
    coefficients_sparse_2_dict = {}
    coefficients_original_2_dict = {}

    basis_1 = results['basis_1']
    basis_2 = results['basis_2']
    coefficients_1 = results['coefficients_1']
    coefficients_2 = results['coefficients_2']

    for pert in basis_1.keys():
        Q, R_qr = np.linalg.qr(basis_1[pert])
        T_mat = np.linalg.inv(R_qr.T)
        coefficients_sparse_1_dict[pert] = (eigenvecs_sparse_1.T @ T_mat)

        # Original coefficients (filtered to valid indices)
        coefficients_original_1_dict[pert] = coefficients_1[pert][valid_indices]

    for pert in basis_2.keys():
        Q, R_qr = np.linalg.qr(basis_2[pert])
        T_mat = np.linalg.inv(R_qr.T)
        coefficients_sparse_2_dict[pert] = (eigenvecs_sparse_2.T @ T_mat)

        # Original coefficients (filtered to valid indices)
        coefficients_original_2_dict[pert] = coefficients_2[pert][valid_indices]

    # Get filtered eigenvalues
    eigenvals_valid = eigenvals_1[valid_indices]

    # Use 'vr' as reference perturbation for plotting
    reference_pert = 'vr' if 'vr' in coefficients_1 else list(coefficients_1.keys())[0]
    print(f"\n--- Plotting results for {reference_pert} ---")

    # Create visualizations
    plot_joint_sparse_coefficients(
        coefficients_sparse_1_dict[reference_pert],
        coefficients_sparse_2_dict[reference_pert],
        eigenvals_valid,
        N_plot=N_plot,
        output_filename="./figures/stack_varimax_coefficients.pdf"
    )

    plot_sparsity_comparison(
        coefficients_original_1_dict[reference_pert],
        coefficients_original_2_dict[reference_pert],
        coefficients_sparse_1_dict[reference_pert],
        coefficients_sparse_2_dict[reference_pert],
        eigenvals_valid,
        N_plot=min(N_plot, 5),
        output_filename="./figures/stack_varimax_sparsity_comparison.pdf"
    )

    plot_joint_sparsity_heatmap(
        coefficients_sparse_1_dict[reference_pert],
        coefficients_sparse_2_dict[reference_pert],
        eigenvals_valid,
        output_filename="./figures/stack_varimax_heatmap.pdf"
    )

    plot_dominant_mode_correlation(
        coefficients_sparse_1_dict[reference_pert],
        coefficients_sparse_2_dict[reference_pert],
        eigenvals_valid,
        output_filename="./figures/stack_varimax_correlation.pdf"
    )

    # Save results
    sparse_results = {
        'eigenvecs_sparse_1': eigenvecs_sparse_1,
        'eigenvecs_sparse_2': eigenvecs_sparse_2,
        'rotation_matrix': R,
        'valid_indices': valid_indices,
        'eigenvals_valid': eigenvals_valid,
        'coefficients_sparse_1': coefficients_sparse_1_dict,
        'coefficients_sparse_2': coefficients_sparse_2_dict,
        'coefficients_original_1': coefficients_original_1_dict,
        'coefficients_original_2': coefficients_original_2_dict,
        'weight_ratio': weight_ratio,
        'method': 'stacked_varimax'
    }

    # Save to pickle
    with open("stack_varimax_sparse_results.pickle", 'wb') as f:
        pickle.dump(sparse_results, f)
    print("\nSaved results to stack_varimax_sparse_results.pickle")

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

        # Verify that both bases are present
        if 'basis_2' not in results or 'eigenvecs_2' not in results:
            print("ERROR: Results file doesn't contain Basis 2 data!")
            print("Please re-run multi_perturbation_analysis with both bases enabled.")
            sys.exit(1)
    else:
        print("No existing results found, running multi_perturbation_analysis...")
        folder_path = f'../data/'
        allowedK = np.load(folder_path + 'data_allowedK/L70_kvalues.npy')
        results = multi_perturbation_analysis(N=len(allowedK), N_t=1000, folder_path=folder_path)

        # Save for future use
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results to {results_file}")

    # Run Stacked Varimax sparse rotation analysis
    print("\n" + "="*80)
    print("Running Stacked Varimax Sparse Rotation Analysis")
    print("="*80 + "\n")

    sparse_results = stacked_varimax_sparse_analysis(
        results,
        eigenvalue_threshold=0.99,  # Use stricter threshold
        weight_ratio=1.0,  # Equal weight for both bases
        N_plot=10
    )

    if sparse_results is not None:
        print("\n" + "="*80)
        print("STACKED VARIMAX ANALYSIS COMPLETE")
        print("="*80)
        print(f"Number of sparse eigenvectors: {sparse_results['eigenvecs_sparse_1'].shape[1]}")
        print(f"Orthogonality preserved: Yes (by construction)")
        print(f"Sparsity achieved in: BOTH Basis 1 AND Basis 2")
        print(f"\nOutput files:")
        print("  - stack_varimax_sparse_results.pickle")
        print("  - stack_varimax_coefficients.pdf")
        print("  - stack_varimax_sparsity_comparison.pdf")
        print("  - stack_varimax_heatmap.pdf")
        print("  - stack_varimax_correlation.pdf")

        # Print summary statistics
        ref_pert = 'vr' if 'vr' in sparse_results['coefficients_sparse_1'] else list(sparse_results['coefficients_sparse_1'].keys())[0]
        dominant_1 = np.argmax(np.abs(sparse_results['coefficients_sparse_1'][ref_pert].real), axis=1)
        dominant_2 = np.argmax(np.abs(sparse_results['coefficients_sparse_2'][ref_pert].real), axis=1)

        print(f"\nDominant modes in Basis 1: {dominant_1 + 1}")
        print(f"Dominant modes in Basis 2: {dominant_2 + 1}")
        print(f"\nMode alignment (how often dominant k matches): {np.mean(dominant_1 == dominant_2):.1%}")

        # Plot the time evolution of sparse eigenfunctions
        print("\n" + "="*80)
        print("Plotting sparse eigenfunction time evolution")
        print("="*80)
        plot_sparse_eigenfunctions(results, sparse_results, N_plot=5)
        plot_sparse_vs_original_eigenfunctions(results, sparse_results, N_plot=3)

        print("\nAdditional output files:")
        print("  - stack_varimax_sparse_eigenfunctions.pdf")
        print("  - stack_varimax_sparse_vs_original.pdf")
    else:
        print("\nStacked Varimax analysis failed")
