# -*- coding: utf-8 -*-
"""
Comparison Script: Varimax vs Sparse PCA

This script runs both Varimax rotation and Sparse PCA methods and compares their results.
It helps you understand the differences between the two approaches for finding sparse
orthonormal solutions.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import sys
import os

sys.path.append('..')
from multi_perturbation_analysis import multi_perturbation_analysis

# Import both methods
from varimax_sparse_rotation import apply_varimax_to_eigenvectors, compute_sparse_coefficients
try:
    from sparse_pca_rotation import apply_sparse_pca_to_eigenvectors, PYMANOPT_AVAILABLE
except ImportError:
    PYMANOPT_AVAILABLE = False
    print("Warning: Could not import sparse_pca_rotation")


def compare_sparsity_metrics(coeffs_varimax, coeffs_sparse_pca, method_names=['Varimax', 'Sparse PCA']):
    """
    Compare sparsity metrics between two methods.
    """
    print("\n" + "="*80)
    print("SPARSITY METRICS COMPARISON")
    print("="*80)

    # Threshold for "near-zero" elements
    thresholds = [1e-1, 1e-2, 1e-3]

    for threshold in thresholds:
        sparsity_varimax = np.mean(np.abs(coeffs_varimax) < threshold)
        sparsity_sparse_pca = np.mean(np.abs(coeffs_sparse_pca) < threshold)

        print(f"\nFraction of elements < {threshold:.0e}:")
        print(f"  {method_names[0]:15s}: {sparsity_varimax:.3f}")
        print(f"  {method_names[1]:15s}: {sparsity_sparse_pca:.3f}")

    # L1 norms
    l1_varimax = np.sum(np.abs(coeffs_varimax))
    l1_sparse_pca = np.sum(np.abs(coeffs_sparse_pca))

    print(f"\nTotal L1 norm:")
    print(f"  {method_names[0]:15s}: {l1_varimax:.3f}")
    print(f"  {method_names[1]:15s}: {l1_sparse_pca:.3f}")

    # L0 norm (number of non-zero elements, with threshold)
    threshold_l0 = 1e-3
    l0_varimax = np.sum(np.abs(coeffs_varimax) > threshold_l0)
    l0_sparse_pca = np.sum(np.abs(coeffs_sparse_pca) > threshold_l0)

    print(f"\nL0 norm (non-zeros, threshold={threshold_l0:.0e}):")
    print(f"  {method_names[0]:15s}: {l0_varimax}")
    print(f"  {method_names[1]:15s}: {l0_sparse_pca}")

    # Maximum coefficient magnitude
    max_varimax = np.max(np.abs(coeffs_varimax))
    max_sparse_pca = np.max(np.abs(coeffs_sparse_pca))

    print(f"\nMaximum coefficient magnitude:")
    print(f"  {method_names[0]:15s}: {max_varimax:.4f}")
    print(f"  {method_names[1]:15s}: {max_sparse_pca:.4f}")

    return {
        'sparsity': [sparsity_varimax, sparsity_sparse_pca],
        'l1_norm': [l1_varimax, l1_sparse_pca],
        'l0_norm': [l0_varimax, l0_sparse_pca],
        'max_coeff': [max_varimax, max_sparse_pca]
    }


def plot_method_comparison(coeffs_varimax, coeffs_sparse_pca, eigenvals_valid,
                          N_plot=6, output_filename="./figures/method_comparison.pdf"):
    """
    Plot side-by-side comparison of both methods.
    """
    N_plot = min(N_plot, coeffs_varimax.shape[0])
    N_basis = coeffs_varimax.shape[1]

    # Find dominant k for each method
    max_indices_varimax = np.argmax(np.abs(coeffs_varimax.real), axis=1)
    max_indices_sparse_pca = np.argmax(np.abs(coeffs_sparse_pca.real), axis=1)

    # Sort by dominant k (using varimax as reference)
    sorted_order = np.argsort(max_indices_varimax)

    fig, axes = plt.subplots(N_plot, 2, figsize=(7.5, 0.7*N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Method Comparison: Varimax vs Sparse PCA", fontsize=12)

    for i in range(N_plot):
        plot_idx = sorted_order[i]

        # Varimax
        ax_varimax = axes[i, 0]
        k_values = np.arange(1, N_basis + 1)
        ax_varimax.bar(k_values, coeffs_varimax[plot_idx, :].real, width=0.4, color='green')
        ax_varimax.set_xlim(0, min(20, N_basis + 1))
        ax_varimax.set_ylim(-1, 1)

        dominant_k_varimax = max_indices_varimax[plot_idx]
        ax_varimax.axvline(dominant_k_varimax + 1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_varimax.set_title(f"Varimax (λ={eigenvals_valid[plot_idx]:.3f}, k={dominant_k_varimax+1})",
                            fontsize=8, loc='left')
        ax_varimax.grid(True, alpha=0.3, axis='y')

        # Sparse PCA
        ax_sparse_pca = axes[i, 1]
        ax_sparse_pca.bar(k_values, coeffs_sparse_pca[plot_idx, :].real, width=0.4, color='purple')
        ax_sparse_pca.set_xlim(0, min(20, N_basis + 1))
        ax_sparse_pca.set_ylim(-1, 1)

        dominant_k_sparse_pca = max_indices_sparse_pca[plot_idx]
        ax_sparse_pca.axvline(dominant_k_sparse_pca + 1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_sparse_pca.set_title(f"Sparse PCA (λ={eigenvals_valid[plot_idx]:.3f}, k={dominant_k_sparse_pca+1})",
                               fontsize=8, loc='left')
        ax_sparse_pca.grid(True, alpha=0.3, axis='y')

        # Format
        for ax in [ax_varimax, ax_sparse_pca]:
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


def plot_sparsity_comparison_bar(metrics, output_filename="./figures/sparsity_comparison_bar.pdf"):
    """
    Create bar chart comparing sparsity metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    methods = ['Varimax', 'Sparse PCA']
    colors = ['green', 'purple']

    # Sparsity (fraction near-zero)
    ax = axes[0]
    x = np.arange(len(methods))
    ax.bar(x, metrics['sparsity'], color=colors, alpha=0.7)
    ax.set_ylabel('Fraction of near-zero elements')
    ax.set_title('Sparsity (|coeff| < 0.001)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(True, alpha=0.3, axis='y')

    # L1 norm
    ax = axes[1]
    ax.bar(x, metrics['l1_norm'], color=colors, alpha=0.7)
    ax.set_ylabel('Total L1 norm')
    ax.set_title('L1 Norm (sum of |coeffs|)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(True, alpha=0.3, axis='y')

    # L0 norm
    ax = axes[2]
    ax.bar(x, metrics['l0_norm'], color=colors, alpha=0.7)
    ax.set_ylabel('Number of non-zero elements')
    ax.set_title('L0 Norm (count of |coeff| > 0.001)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved bar chart comparison to {output_filename}")
    plt.show()


def run_comparison(results, eigenvalue_threshold=0.9, N_plot=6,
                  sparsity_weight=1.0, epsilon=1e-8):
    """
    Run both methods and compare results.
    """
    print("\n" + "="*80)
    print("COMPARING VARIMAX AND SPARSE PCA METHODS")
    print("="*80)

    eigenvecs_1 = results['eigenvecs_1']
    eigenvals_1 = results['eigenvals_1']
    coefficients_1 = results['coefficients_1']

    # Use 'vr' as reference perturbation
    reference_pert = 'vr' if 'vr' in coefficients_1 else list(coefficients_1.keys())[0]
    print(f"Using {reference_pert} as reference perturbation")

    # Run Varimax
    print("\n" + "-"*80)
    print("RUNNING VARIMAX ROTATION")
    print("-"*80)
    eigenvecs_varimax, R_varimax, valid_indices_varimax = apply_varimax_to_eigenvectors(
        eigenvecs_1, eigenvals_1, eigenvalue_threshold)

    if eigenvecs_varimax is None:
        print("Varimax failed!")
        return None

    eigenvals_valid = np.array(eigenvals_1)[valid_indices_varimax]

    # Recompute QR decomposition to get transformation matrix
    basis_matrix = results['basis_1'][reference_pert]
    Q, R_qr = np.linalg.qr(basis_matrix)
    transform_matrix = np.linalg.inv(R_qr.T)  # Shape: (N_basis, N_basis)

    coeffs_varimax = compute_sparse_coefficients(eigenvecs_varimax, transform_matrix)

    # Run Sparse PCA
    if PYMANOPT_AVAILABLE:
        print("\n" + "-"*80)
        print("RUNNING SPARSE PCA")
        print("-"*80)
        eigenvecs_sparse_pca, R_sparse_pca, valid_indices_sparse_pca, converged = \
            apply_sparse_pca_to_eigenvectors(
                eigenvecs_1, eigenvals_1, eigenvalue_threshold,
                sparsity_weight=sparsity_weight, epsilon=epsilon)

        if eigenvecs_sparse_pca is None:
            print("Sparse PCA failed!")
            return None

        coeffs_sparse_pca = compute_sparse_coefficients(eigenvecs_sparse_pca, transform_matrix)

        # Compare metrics
        metrics = compare_sparsity_metrics(coeffs_varimax, coeffs_sparse_pca)

        # Plot comparisons
        plot_method_comparison(coeffs_varimax, coeffs_sparse_pca, eigenvals_valid,
                             N_plot=N_plot, output_filename="./figures/method_comparison.pdf")

        plot_sparsity_comparison_bar(metrics, output_filename="./figures/sparsity_comparison_bar.pdf")

        # Save comparison results
        comparison_results = {
            'varimax': {
                'eigenvecs': eigenvecs_varimax,
                'rotation_matrix': R_varimax,
                'coefficients': coeffs_varimax
            },
            'sparse_pca': {
                'eigenvecs': eigenvecs_sparse_pca,
                'rotation_matrix': R_sparse_pca,
                'coefficients': coeffs_sparse_pca,
                'converged': converged
            },
            'metrics': metrics,
            'eigenvals_valid': eigenvals_valid,
            'reference_perturbation': reference_pert
        }

        with open("comparison_results.pickle", 'wb') as f:
            pickle.dump(comparison_results, f)
        print("\nSaved comparison results to comparison_results.pickle")

        return comparison_results

    else:
        print("\nSkipping Sparse PCA comparison (pymanopt not available)")
        return {
            'varimax': {
                'eigenvecs': eigenvecs_varimax,
                'rotation_matrix': R_varimax,
                'coefficients': coeffs_varimax
            },
            'eigenvals_valid': eigenvals_valid
        }


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    # Load or compute results
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

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results to {results_file}")

    # Run comparison
    comparison = run_comparison(
        results,
        eigenvalue_threshold=0.9,
        N_plot=6,
        sparsity_weight=1.0,  # Adjust this to tune Sparse PCA sparsity
        epsilon=1e-8
    )

    if comparison is not None:
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print("\nOutput files:")
        print("  - comparison_results.pickle")
        print("  - method_comparison.pdf")
        if PYMANOPT_AVAILABLE:
            print("  - sparsity_comparison_bar.pdf")
        print("\nBoth methods preserve orthonormality by construction.")
        if PYMANOPT_AVAILABLE:
            print("Sparse PCA typically produces sparser results than Varimax,")
            print("but Varimax is faster and requires no hyperparameters.")
