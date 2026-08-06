# -*- coding: utf-8 -*-
"""
CCA-Based Sparse Joint Eigenfunction Method  —  Derivative Regularisation Variant
===================================================================================

Identical to cca_sparse_analysis.py except that all three Gram matrices are
augmented with a Sobolev-type k-derivative term:

    G1_reg   = sum_p  w_p  [(X̄1^p)^T X̄1^p  +  lambda * (dX̄1^p/dk)^T (dX̄1^p/dk)]
    G2_reg   = sum_p  w_p  [(X̄2^p)^T X̄2^p  +  lambda * (dX̄2^p/dk)^T (dX̄2^p/dk)]
    G12_reg  = sum_p  w_p  [(X̄1^p)^T X̄2^p  +  lambda * (dX̄1^p/dk)^T (dX̄2^p/dk)]

where dX̄^p/dk is computed column-wise via np.gradient with the actual k-spacing.

Motivation (see mixing_coef_problem.md):
    At high k, X(k_i) ≈ X(k_{i+1}), making G1 and G2 ill-conditioned.
    The derivative term penalises directions where adjacent k-modes look similar,
    effectively pushing apart the eigenvalues of G1/G2 in the high-k block and
    making the whitening step stable without suppressing canonical correlations
    (unlike Tikhonov lambda*I which biases all correlations downward).

Key advantage over Tikhonov:
    The derivative term is *data-adaptive*: it is large where modes are nearly
    degenerate (high k) and small where modes are well-separated (low k), so it
    regularises only where needed and preserves rho_i ≈ 1 for well-separated modes.

Outputs go to ./figures_reg/  (separate from ./figures_cca/ and ./figures_nomix/).

Entry points:
    1.  cca_sparse_reg_analysis(lambda_reg=...)   — single run
    2.  lambda_sweep(...)                          — scan several lambda values
                                                     and compare heatmaps
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append('../multi_pert_sparse_ortho')

from multi_perturbation_analysis import generate_multi_perturbation_bases, fcb_time
from cca_sparse_analysis import (
    matrix_sqrt_inv,
    solve_cca,
    apply_sparse_rotation,
    gini,
    normalize_columns,
    plot_gram_spectrum,
    plot_canonical_correlations,
    plot_coefficients,
    plot_reconstructed_timeseries,
    PERTURBATION_TYPES,
    FOLDER_PATH,
)

# ---------------------------------------------------------------------------
OUTPUT_DIR    = './figures_reg/'
RESULTS_CACHE = 'cca_reg_results.pickle'
# ---------------------------------------------------------------------------


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ============================================================================
# STEP 1 — Load bases  (same as cca_sparse_analysis)
# ============================================================================

def load_bases(folder_path=FOLDER_PATH, N_t=1000):
    """Load integerK (basis 1) and allowedK (basis 2) onto a uniform eta grid."""
    eta_grid = np.linspace(0, fcb_time, N_t)
    print("Loading integerK (basis 1)...")
    basis_1 = generate_multi_perturbation_bases("integerK", eta_grid,
                                                folder_path=folder_path)
    print("\nLoading allowedK (basis 2)...")
    basis_2 = generate_multi_perturbation_bases("allowedK", eta_grid,
                                                folder_path=folder_path)
    return basis_1, basis_2, eta_grid


# ============================================================================
# STEP 1b — Load k-values for derivative computation
# ============================================================================

def load_kvalues(dataset_name, folder_path=FOLDER_PATH):
    """
    Load the k-value array for a given dataset.

    k-values live in data_{dataset}_timeseries/L70_kvalues.npy, which is
    the same file that generate_multi_perturbation_bases reads internally.

    Parameters
    ----------
    dataset_name : 'integerK' or 'allowedK'
    folder_path  : path to the data/ directory

    Returns
    -------
    k_values : (N_k,) float array of wavenumbers in ascending order
    """
    path = os.path.join(folder_path,
                        f'data_{dataset_name}_timeseries',
                        'L70_kvalues.npy')
    k_values = np.load(path)
    print(f"  Loaded {len(k_values)} k-values for {dataset_name}  "
          f"(k_min={k_values.min():.4f}, k_max={k_values.max():.4f})")
    return k_values


# ============================================================================
# STEP 2 — Frobenius normalise  (same as cca_sparse_analysis)
# ============================================================================

def frobenius_normalize(basis_dict):
    """
    Divide each perturbation-type matrix by its Frobenius norm.

    Returns
    -------
    Xn    : dict  {p: normalised (N_t, N_k) array}
    norms : dict  {p: Frobenius norm}
    """
    Xn, norms = {}, {}
    for p in PERTURBATION_TYPES:
        X = basis_dict[p]
        n = np.linalg.norm(X, 'fro')
        norms[p] = n
        Xn[p] = X / n
    return Xn, norms


# ============================================================================
# STEP 2b — k-derivatives of Frobenius-normalised matrices
# ============================================================================

def compute_k_derivatives(Xn_dict, k_values):
    """
    Compute dX̄^p/dk for each perturbation type using np.gradient.

    np.gradient uses second-order central differences for interior points and
    first-order one-sided differences at boundaries.  It handles non-uniform
    k-spacing correctly (important for allowedK).

    Parameters
    ----------
    Xn_dict  : dict {p: (N_t, N_k) Frobenius-normalised array}
    k_values : (N_k,) array of k-values with the correct spacing for this basis

    Returns
    -------
    dXn : dict {p: (N_t, N_k) array of dX̄^p/dk}
    """
    dXn = {}
    for p in PERTURBATION_TYPES:
        # np.gradient(f, x, axis) differentiates f along `axis` using spacing x
        dXn[p] = np.gradient(Xn_dict[p], k_values, axis=1)
    return dXn


# ============================================================================
# STEP 3 — Regularised Gram matrices
# ============================================================================

def compute_regularised_gram_matrices(Xn1, Xn2, dXn1, dXn2,
                                      lambda_reg=1e-2, weights=None):
    """
    Compute the three regularised k-space Gram matrices:

        G1_reg  = sum_p  w_p  [(X̄1^p)^T X̄1^p  +  lambda * (dX̄1^p/dk)^T (dX̄1^p/dk)]
        G2_reg  = sum_p  w_p  [(X̄2^p)^T X̄2^p  +  lambda * (dX̄2^p/dk)^T (dX̄2^p/dk)]
        G12_reg = sum_p  w_p  [(X̄1^p)^T X̄2^p  +  lambda * (dX̄1^p/dk)^T (dX̄2^p/dk)]

    Each standard term has shape (N_k, N_k); the derivative term has the same
    shape.  The derivative terms are computed from the already-normalised
    matrices, so the scale of dX̄/dk is consistent with that of X̄.

    Parameters
    ----------
    Xn1, Xn2   : dicts {p: (N_t, N_k)} of Frobenius-normalised arrays
    dXn1, dXn2 : dicts {p: (N_t, N_k)} of k-derivatives (from compute_k_derivatives)
    lambda_reg  : regularisation strength (dimensionless, same units as the
                  ratio ||dX̄/dk||² / ||X̄||²)
    weights     : dict {p: float} or None (unit weights)

    Returns
    -------
    G1, G2, G12  each (N_k, N_k)
    """
    if weights is None:
        weights = {p: 1.0 for p in PERTURBATION_TYPES}

    N_k = Xn1[PERTURBATION_TYPES[0]].shape[1]
    G1  = np.zeros((N_k, N_k))
    G2  = np.zeros((N_k, N_k))
    G12 = np.zeros((N_k, N_k))

    for p in PERTURBATION_TYPES:
        w    = weights[p]
        X1p  = Xn1[p]    # (N_t, N_k)
        X2p  = Xn2[p]
        dX1p = dXn1[p]   # (N_t, N_k)  dX̄1^p/dk
        dX2p = dXn2[p]   # (N_t, N_k)  dX̄2^p/dk

        # Standard cross-time inner products
        G1  += w * (X1p.T @ X1p  +  lambda_reg * dX1p.T @ dX1p)
        G2  += w * (X2p.T @ X2p  +  lambda_reg * dX2p.T @ dX2p)
        G12 += w * (X1p.T @ X2p  +  lambda_reg * dX1p.T @ dX2p)

    return G1, G2, G12


# ============================================================================
# Diagnostic: derivative power spectrum
# ============================================================================

def plot_derivative_power(Xn1, Xn2, dXn1, dXn2, k_values_1, k_values_2,
                          output_dir=OUTPUT_DIR):
    """
    Plot ||X̄^p(:,i)||² and ||dX̄^p(:,i)/dk||² vs k for each perturbation type.

    This shows the relative scale of the standard and derivative terms and
    helps choose lambda_reg.  A good lambda_reg makes the derivative term
    comparable to the standard term in the high-k region.
    """
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey='row')
    fig.suptitle('Column norms:  ||X̄(:,i)||  and  ||dX̄/dk(:,i)||  vs k',
                 fontsize=12)

    for col, p in enumerate(PERTURBATION_TYPES):
        # Basis 1
        norm1  = np.linalg.norm(Xn1[p],  axis=0)   # (N_k,)
        dnorm1 = np.linalg.norm(dXn1[p], axis=0)

        # Basis 2
        norm2  = np.linalg.norm(Xn2[p],  axis=0)
        dnorm2 = np.linalg.norm(dXn2[p], axis=0)

        # Row 0: column norms of X̄
        ax = axes[0, col]
        ax.semilogy(k_values_1, norm1,  'b-',  lw=1.2, label='integerK')
        ax.semilogy(k_values_2, norm2,  'g--', lw=1.2, label='allowedK')
        ax.set_title(f'{p}  — ||X̄||', fontsize=9)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Column norm', fontsize=9)
            ax.legend(fontsize=8)

        # Row 1: column norms of dX̄/dk
        ax = axes[1, col]
        ax.semilogy(k_values_1, dnorm1, 'b-',  lw=1.2, label='integerK')
        ax.semilogy(k_values_2, dnorm2, 'g--', lw=1.2, label='allowedK')
        ax.set_title(f'{p}  — ||dX̄/dk||', fontsize=9)
        ax.set_xlabel('k', fontsize=9)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Column norm of derivative', fontsize=9)

    fig.tight_layout()
    out = output_dir + 'reg_derivative_power.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Heatmap  (same logic as cca_sparse_analysis)
# ============================================================================

def plot_heatmap(A_sparse, B_sparse, lambda_reg, output_dir=OUTPUT_DIR,
                 suffix=''):
    """Heatmap of column-normalised |coefficients|, sorted by dominant k in basis 1."""
    _ensure_dir(output_dir)
    A_norm = normalize_columns(A_sparse)
    B_norm = normalize_columns(B_sparse)
    dom1   = np.argmax(np.abs(A_norm), axis=0)
    order  = np.argsort(dom1)
    A_s    = A_norm[:, order]
    B_s    = B_norm[:, order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(np.abs(A_s.T), aspect='auto', cmap='Blues',
                     interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xlabel('k index (integerK)')
    ax1.set_ylabel('Mode (sorted by dominant k)')
    ax1.set_title('|Coefficients| — Basis 1 (integerK)')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.abs(B_s.T), aspect='auto', cmap='Greens',
                     interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('k index (allowedK)')
    ax2.set_ylabel('Mode (sorted by dominant k)')
    ax2.set_title('|Coefficients| — Basis 2 (allowedK)')
    plt.colorbar(im2, ax=ax2)

    fig.suptitle(f'Derivative-reg CCA heatmap  (lambda = {lambda_reg:.1e})',
                 fontsize=12)
    fig.tight_layout()
    out = output_dir + f'reg_heatmap{suffix}.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Lambda sweep
# ============================================================================

def lambda_sweep(Xn1, Xn2, dXn1, dXn2,
                 basis_1, basis_2, eta_grid, norms_1, norms_2,
                 lambdas=None,
                 weights=None,
                 ev_threshold=1e-10,
                 rho_min=0.99,
                 rotation_method='promax',
                 output_dir=OUTPUT_DIR):
    """
    Run CCA + sparse rotation for several lambda_reg values.

    The standard Gram terms (lambda=0) are computed once; each lambda value
    just rescales the pre-computed derivative Gram terms and adds them — so
    the sweep is cheap after the first computation.

    Parameters
    ----------
    lambdas : list of lambda_reg values to try; None → default grid

    Returns
    -------
    dict  {lambda: {'A_sparse', 'B_sparse', 'rho', 'n_modes'}}
    """
    if lambdas is None:
        lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    if weights is None:
        weights = {p: 1.0 for p in PERTURBATION_TYPES}

    _ensure_dir(output_dir)

    # Pre-compute standard and derivative Gram blocks separately
    N_k = Xn1[PERTURBATION_TYPES[0]].shape[1]
    G1_std  = np.zeros((N_k, N_k))
    G2_std  = np.zeros((N_k, N_k))
    G12_std = np.zeros((N_k, N_k))
    G1_drv  = np.zeros((N_k, N_k))
    G2_drv  = np.zeros((N_k, N_k))
    G12_drv = np.zeros((N_k, N_k))

    for p in PERTURBATION_TYPES:
        w = weights[p]
        G1_std  += w * Xn1[p].T  @ Xn1[p]
        G2_std  += w * Xn2[p].T  @ Xn2[p]
        G12_std += w * Xn1[p].T  @ Xn2[p]
        G1_drv  += w * dXn1[p].T @ dXn1[p]
        G2_drv  += w * dXn2[p].T @ dXn2[p]
        G12_drv += w * dXn1[p].T @ dXn2[p]

    results = {}
    for lam in lambdas:
        label = f'lam{lam:.0e}' if lam > 0 else 'lam0'
        print(f"\n{'='*55}")
        print(f"  lambda_reg = {lam}")
        print(f"{'='*55}")

        G1  = G1_std  + lam * G1_drv
        G2  = G2_std  + lam * G2_drv
        G12 = G12_std + lam * G12_drv

        A, B, rho, rho_all = solve_cca(G1, G2, G12, ev_threshold, rho_min)
        n_modes = len(rho)
        print(f"  Valid modes: {n_modes}")

        if n_modes == 0:
            print("  No valid modes — skipping rotation and plot.")
            results[lam] = dict(A_sparse=None, B_sparse=None,
                                rho=rho, n_modes=0)
            continue

        A_sparse, B_sparse = apply_sparse_rotation(A, B, method=rotation_method)
        plot_heatmap(A_sparse, B_sparse, lam,
                     output_dir=output_dir, suffix=f'_{label}')
        results[lam] = dict(A_sparse=A_sparse, B_sparse=B_sparse,
                            rho=rho, n_modes=n_modes)

    # ── Summary: all heatmaps (Basis 1) side by side ────────────────────────
    valid_lams = [l for l in lambdas if results[l]['n_modes'] > 0]
    if valid_lams:
        n = len(valid_lams)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        for col, lam in enumerate(valid_lams):
            A_s   = results[lam]['A_sparse']
            A_n   = normalize_columns(A_s)
            dom1  = np.argmax(np.abs(A_n), axis=0)
            order = np.argsort(dom1)
            im    = axes[0, col].imshow(
                np.abs(A_n[:, order].T),
                aspect='auto', cmap='Blues',
                interpolation='nearest', vmin=0, vmax=1,
            )
            n_m = results[lam]['n_modes']
            axes[0, col].set_title(f'λ={lam:.0e}  (m={n_m})', fontsize=10)
            axes[0, col].set_xlabel('k index (integerK)')
            if col == 0:
                axes[0, col].set_ylabel('Mode')
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        fig.suptitle('Derivative-reg sweep — Basis 1 heatmaps', fontsize=12)
        fig.tight_layout()
        out = output_dir + 'reg_lambda_sweep.pdf'
        plt.savefig(out, bbox_inches='tight')
        print(f"\nSaved lambda-sweep summary: {out}")
        plt.close()

    return results


# ============================================================================
# Main analysis function
# ============================================================================

def cca_sparse_reg_analysis(
        N_t              = 1000,
        weights          = None,
        lambda_reg       = 1e-2,       # derivative regularisation strength
        ev_threshold     = 1e-10,
        rho_min          = 0.99,
        rotation_method  = 'promax',
        folder_path      = FOLDER_PATH,
        output_dir       = OUTPUT_DIR,
        results_cache    = RESULTS_CACHE,
        force_recompute  = False,
        run_lambda_sweep = True,
        sweep_lambdas    = None,
):
    """
    Full CCA sparse analysis with Sobolev-type derivative regularisation.

    Parameters
    ----------
    lambda_reg       : weight of the derivative term in the Gram matrices.
                       lambda=0 recovers the original cca_sparse_analysis result.
                       Increase lambda to push apart high-k degenerate modes.
    run_lambda_sweep : also run a sweep over several lambda values
    sweep_lambdas    : list of lambda values for the sweep;
                       None → [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    Returns
    -------
    dict with all results including the regularised Gram matrices.
    """

    if not force_recompute and os.path.exists(results_cache):
        print(f"Loading cached results from {results_cache}")
        with open(results_cache, 'rb') as f:
            return pickle.load(f)

    # ── Step 1: Load bases ────────────────────────────────────────────────────
    basis_1, basis_2, eta_grid = load_bases(folder_path, N_t)

    # ── Step 1b: Load k-values ────────────────────────────────────────────────
    print("\n--- Step 1b: Loading k-values ---")
    k_values_1 = load_kvalues('integerK', folder_path)
    k_values_2 = load_kvalues('allowedK', folder_path)

    # ── Step 2: Frobenius normalise ───────────────────────────────────────────
    print("\n--- Step 2: Frobenius normalisation ---")
    Xn1, norms_1 = frobenius_normalize(basis_1)
    Xn2, norms_2 = frobenius_normalize(basis_2)

    # ── Step 2b: k-derivatives ────────────────────────────────────────────────
    print("\n--- Step 2b: Computing k-derivatives ---")
    dXn1 = compute_k_derivatives(Xn1, k_values_1)
    dXn2 = compute_k_derivatives(Xn2, k_values_2)
    print(f"  dX/dk shape (integerK, dr): {dXn1['dr'].shape}")

    # ── Diagnostic: derivative power ──────────────────────────────────────────
    print("\n--- Diagnostic: derivative power spectrum ---")
    plot_derivative_power(Xn1, Xn2, dXn1, dXn2,
                          k_values_1, k_values_2, output_dir)

    # ── Step 3: Regularised Gram matrices ─────────────────────────────────────
    print(f"\n--- Step 3: Regularised Gram matrices (lambda={lambda_reg}) ---")
    G1, G2, G12 = compute_regularised_gram_matrices(
        Xn1, Xn2, dXn1, dXn2, lambda_reg=lambda_reg, weights=weights)
    print(f"  Gram matrix shape: {G1.shape}")

    # ── Step 4: CCA (whitened SVD) ────────────────────────────────────────────
    print("\n--- Step 4: CCA (whitened SVD) ---")
    A, B, rho, rho_all = solve_cca(G1, G2, G12, ev_threshold, rho_min)
    print(f"  A shape: {A.shape},  B shape: {B.shape}")

    # ── Step 5: Sparse rotation ───────────────────────────────────────────────
    print("\n--- Step 5: Sparse rotation ---")
    A_sparse, B_sparse = apply_sparse_rotation(A, B, method=rotation_method)

    # ── Sparsity metrics ──────────────────────────────────────────────────────
    g_before_1 = np.mean([gini(A[:, i]) for i in range(A.shape[1])])
    g_after_1  = np.mean([gini(A_sparse[:, i]) for i in range(A_sparse.shape[1])])
    g_before_2 = np.mean([gini(B[:, i]) for i in range(B.shape[1])])
    g_after_2  = np.mean([gini(B_sparse[:, i]) for i in range(B_sparse.shape[1])])
    print(f"\nMean Gini sparsity (higher = sparser):")
    print(f"  Basis 1 — before: {g_before_1:.3f},  after: {g_after_1:.3f}")
    print(f"  Basis 2 — before: {g_before_2:.3f},  after: {g_after_2:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Plotting ---")
    plot_gram_spectrum(G1, G2, output_dir)
    plot_canonical_correlations(rho_all, rho_min, output_dir)
    n_show = min(20, len(rho))
    plot_coefficients(A, B, A_sparse, B_sparse, rho,
                      N_plot=n_show, output_dir=output_dir)
    plot_heatmap(A_sparse, B_sparse, lambda_reg, output_dir=output_dir)
    plot_reconstructed_timeseries(
        basis_1, basis_2,
        A_sparse, B_sparse, rho,
        eta_grid, norms_1, norms_2,
        N_plot=5, output_dir=output_dir,
    )

    # ── Lambda sweep ──────────────────────────────────────────────────────────
    if run_lambda_sweep:
        print("\n--- Lambda sweep ---")
        lambda_sweep(
            Xn1, Xn2, dXn1, dXn2,
            basis_1, basis_2, eta_grid, norms_1, norms_2,
            lambdas=sweep_lambdas,
            weights=weights,
            ev_threshold=ev_threshold,
            rho_min=rho_min,
            rotation_method=rotation_method,
            output_dir=output_dir,
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    results = dict(
        eta_grid        = eta_grid,
        G1=G1, G2=G2, G12=G12,
        A=A, B=B,
        A_sparse        = A_sparse,
        B_sparse        = B_sparse,
        rho             = rho,
        rho_all         = rho_all,
        basis_1         = basis_1,
        basis_2         = basis_2,
        norms_1         = norms_1,
        norms_2         = norms_2,
        k_values_1      = k_values_1,
        k_values_2      = k_values_2,
        weights         = weights,
        rotation_method = rotation_method,
        lambda_reg      = lambda_reg,
        ev_threshold    = ev_threshold,
        rho_min         = rho_min,
    )
    with open(results_cache, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_cache}")
    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    weights = {
        'dr': 1.0,
        'dm': 100.0,
        'vr': 1.0,
        'vm': 1.0,
    }

    results = cca_sparse_reg_analysis(
        N_t              = 1000,
        weights          = weights,
        lambda_reg       = 1e-2,
        ev_threshold     = 1e-10,
        rho_min          = 0.99,
        rotation_method  = 'promax',
        force_recompute  = True,
        run_lambda_sweep = True,
        sweep_lambdas    = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Valid modes found : {len(results['rho'])}")
    print(f"  rho range         : {np.round(results['rho'], 6)}")
    print(f"  N_k               : {results['G1'].shape[0]}")
    print(f"  lambda_reg        : {results['lambda_reg']}")
    print("=" * 60)
