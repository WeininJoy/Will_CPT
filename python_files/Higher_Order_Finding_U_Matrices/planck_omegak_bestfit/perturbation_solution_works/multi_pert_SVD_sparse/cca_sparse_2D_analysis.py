# -*- coding: utf-8 -*-
"""
CCA-Based Sparse Joint Eigenfunction Method  —  2D Space-Time Variant
======================================================================

Extension of cca_sparse_analysis.py that includes the spatial part of the
perturbation eigenfunctions in the inner product, making the Gram matrices
sensitive to *both* time-evolution similarity and spatial compatibility.

Physical motivation
-------------------
In a closed universe (S^3), the radial scalar spatial eigenfunction is:

    Pi_beta(chi) = sin(beta * chi) / (beta * sin(chi)),   chi in [0, pi]

where beta is the dimensionless wavenumber.  For the closed universe
quantisation condition, beta takes integer values (beta = 3, 4, 5, ...),
which is exactly the integerK basis.  The allowedK basis uses non-integer
beta values constrained by other boundary conditions.

The spatial overlap between mode i (beta1_i) and mode j (beta2_j) is:

    S(beta1, beta2) = integral_0^pi  Pi_beta1(chi) Pi_beta2(chi) sin^2(chi) dchi
                    = (1 / beta1 beta2) integral_0^pi sin(beta1 chi) sin(beta2 chi) dchi

Analytic result:
    diff = beta1 - beta2,  summ = beta1 + beta2
    integral = sin(diff*pi)/(2*diff) - sin(summ*pi)/(2*summ)
    (limit diff->0:  integral -> pi/2 - sin(summ*pi)/(2*summ))

Key consequences:
  * If both beta are integers and beta1 != beta2: S = 0  (exact orthogonality)
  * If beta1 = beta2 (integer):  S = pi/(2*beta^2)
  * For allowedK (non-integer beta2) near an integerK beta1: S is small but
    non-zero, with magnitude decaying rapidly as |beta1 - beta2| increases.

2D Gram matrices
----------------
Space and time are separable in the background cosmology, so the 2D
space-time Gram matrix is simply the Hadamard (element-wise) product of
the time-Gram matrix and the spatial overlap matrix:

    G1_2D  = G1_time  o S11   (G1 from cca_sparse_analysis * spatial self-overlap)
    G2_2D  = G2_time  o S22
    G12_2D = G12_time o S12

Because S11 is diagonal (all zeros off-diagonal for integer beta), G1_2D is
also diagonal: the integerK modes are perfectly orthogonal in space-time.
This makes the CCA problem well-conditioned and decouple mode-by-mode, with
no need for regularisation or post-hoc rotation.

For allowedK, S22 is approximately diagonal (small off-diagonal entries),
so G2_2D is approximately diagonal too.

Implementation
--------------
1. Load integerK (basis 1) and allowedK (basis 2) time series.
2. Convert stored k-values to beta wavenumbers:  beta = sqrt(k^2 + 1).
3. Frobenius-normalise each perturbation type.
4. Compute time-only Gram matrices G1_time, G2_time, G12_time.
5. Compute spatial overlap matrices S11, S22, S12 analytically.
6. Form 2D Gram matrices via Hadamard products.
7. Run whitened SVD (CCA) on 2D Gram matrices.
8. Apply Promax sparse rotation.

Outputs go to ./figures_2D/
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
OUTPUT_DIR    = './figures_2D/'
RESULTS_CACHE = 'cca_2D_results.pickle'
# ---------------------------------------------------------------------------


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ============================================================================
# STEP 1 — Load bases
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
# STEP 1b — Load k-values and convert to beta
# ============================================================================

def load_kvalues(dataset_name, folder_path=FOLDER_PATH):
    """Load k-value array for a given dataset from the timeseries data directory."""
    path = os.path.join(folder_path,
                        f'data_{dataset_name}_timeseries',
                        'L70_kvalues.npy')
    k_values = np.load(path)
    print(f"  Loaded {len(k_values)} k-values for {dataset_name}  "
          f"(k_min={k_values.min():.4f}, k_max={k_values.max():.4f})")
    return k_values


def k_to_beta(k_values):
    """
    Convert stored k-values to spatial wavenumber beta.

    In a closed universe K=+1, the relation between the dimensionless
    wavenumber beta and the comoving wavenumber k is:

        k^2 = beta^2 - 1   =>   beta = sqrt(k^2 + 1)

    For integerK: beta should be close to integers (3, 4, 5, ...).
    For allowedK: beta takes non-integer values near the same integers.

    Parameters
    ----------
    k_values : (N_k,) array of stored k-values

    Returns
    -------
    beta : (N_k,) array of spatial wavenumbers
    """
    beta = np.sqrt(k_values**2 + 1.0)
    return beta


# ============================================================================
# STEP 2 — Frobenius normalise
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
# STEP 3a — Time-only Gram matrices (standard CCA Gram matrices)
# ============================================================================

def compute_time_gram_matrices(Xn1, Xn2, weights=None):
    """
    Compute the three time-only k-space Gram matrices:

        G1_time  = sum_p  w_p  (Xn1^p)^T  Xn1^p    shape (N_k1, N_k1)
        G2_time  = sum_p  w_p  (Xn2^p)^T  Xn2^p    shape (N_k2, N_k2)
        G12_time = sum_p  w_p  (Xn1^p)^T  Xn2^p    shape (N_k1, N_k2)

    Parameters
    ----------
    Xn1, Xn2 : dicts of Frobenius-normalised (N_t, N_k) arrays
    weights  : dict {p: float} or None (unit weights)

    Returns
    -------
    G1_t, G2_t, G12_t
    """
    if weights is None:
        weights = {p: 1.0 for p in PERTURBATION_TYPES}

    N_k1 = Xn1[PERTURBATION_TYPES[0]].shape[1]
    N_k2 = Xn2[PERTURBATION_TYPES[0]].shape[1]
    G1_t  = np.zeros((N_k1, N_k1))
    G2_t  = np.zeros((N_k2, N_k2))
    G12_t = np.zeros((N_k1, N_k2))

    for p in PERTURBATION_TYPES:
        w   = weights[p]
        X1p = Xn1[p]   # (N_t, N_k1)
        X2p = Xn2[p]   # (N_t, N_k2)
        G1_t  += w * X1p.T @ X1p
        G2_t  += w * X2p.T @ X2p
        G12_t += w * X1p.T @ X2p

    return G1_t, G2_t, G12_t


# ============================================================================
# STEP 3b — Spatial overlap matrix
# ============================================================================

def compute_spatial_overlap(beta1, beta2):
    """
    Compute the spatial overlap matrix S[i,j] between modes with wavenumbers
    beta1[i] and beta2[j] in a closed universe (K=+1).

    The integral is:
        S[i,j] = integral_0^pi Pi_beta1[i](chi) Pi_beta2[j](chi) sin^2(chi) dchi

    where Pi_beta(chi) = sin(beta*chi) / (beta * sin(chi)).

    After substitution:
        S[i,j] = (1 / beta1[i] * beta2[j])
                 * (sin(diff*pi)/(2*diff) - sin(summ*pi)/(2*summ))
    with diff = beta1[i] - beta2[j], summ = beta1[i] + beta2[j].

    The limit diff->0 is handled analytically:
        lim_{diff->0} sin(diff*pi)/(2*diff) = pi/2

    Parameters
    ----------
    beta1 : (N1,) array of wavenumbers for basis 1
    beta2 : (N2,) array of wavenumbers for basis 2

    Returns
    -------
    S : (N1, N2) spatial overlap matrix
    """
    beta1 = np.asarray(beta1, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    b1 = beta1[:, None]   # (N1, 1)
    b2 = beta2[None, :]   # (1,  N2)

    diff = b1 - b2         # (N1, N2)
    summ = b1 + b2         # (N1, N2)  always > 0

    # term1 = sin(diff*pi) / (2*diff),  limit pi/2 when diff -> 0
    near_zero = np.abs(diff) < 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(near_zero,
                         np.pi / 2.0,
                         np.sin(diff * np.pi) / (2.0 * diff))

    # term2 = sin(summ*pi) / (2*summ)  — summ is always > 0, no singularity
    term2 = np.sin(summ * np.pi) / (2.0 * summ)

    integral = term1 - term2       # (N1, N2)
    S = integral / (b1 * b2)       # (N1, N2)

    return S


# ============================================================================
# STEP 3c — 2D (space-time) Gram matrices via Hadamard product
# ============================================================================

def compute_2d_gram_matrices(G1_t, G2_t, G12_t, S11, S22, S12,
                             normalise_spatial=True):
    """
    Form 2D space-time Gram matrices as Hadamard products:

        G1_2D  = G1_time  * S11
        G2_2D  = G2_time  * S22
        G12_2D = G12_time * S12

    Parameters
    ----------
    G1_t, G2_t, G12_t : time-only Gram matrices
    S11, S22, S12      : spatial overlap matrices
    normalise_spatial  : if True, normalise S so that diag(S11) = diag(S22) = 1.
                         This makes the diagonal entries of G1_2D equal to
                         those of G1_t, preserving scale.  The off-diagonal
                         suppression by the spatial part is unchanged.

    Returns
    -------
    G1_2D, G2_2D, G12_2D
    """
    if normalise_spatial:
        # Normalise S so that S[i,i] = 1 for each diagonal
        d11 = np.sqrt(np.diag(S11))                    # (N_k1,)
        d22 = np.sqrt(np.diag(S22))                    # (N_k2,)
        # Avoid division by zero
        d11 = np.where(d11 < 1e-30, 1.0, d11)
        d22 = np.where(d22 < 1e-30, 1.0, d22)
        S11_n = S11 / np.outer(d11, d11)
        S22_n = S22 / np.outer(d22, d22)
        S12_n = S12 / np.outer(d11, d22)
    else:
        S11_n, S22_n, S12_n = S11, S22, S12

    G1_2D  = G1_t  * S11_n
    G2_2D  = G2_t  * S22_n
    G12_2D = G12_t * S12_n

    return G1_2D, G2_2D, G12_2D


# ============================================================================
# Diagnostics
# ============================================================================

def print_beta_info(beta1, beta2):
    """Print beta values and how close they are to integers."""
    print(f"\n  integerK: {len(beta1)} modes,  "
          f"beta min={beta1.min():.4f}, max={beta1.max():.4f}")
    nearest_int = np.round(beta1).astype(int)
    max_err = np.max(np.abs(beta1 - nearest_int))
    print(f"  integerK max deviation from integer: {max_err:.2e}")

    print(f"\n  allowedK: {len(beta2)} modes,  "
          f"beta min={beta2.min():.4f}, max={beta2.max():.4f}")
    nearest_int2 = np.round(beta2).astype(int)
    max_err2 = np.max(np.abs(beta2 - nearest_int2))
    print(f"  allowedK max deviation from integer: {max_err2:.2e}")


def plot_spatial_overlap(S12, beta1, beta2, output_dir=OUTPUT_DIR):
    """Heatmap of the spatial overlap matrix S12 between the two bases."""
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Raw S12
    im0 = axes[0].imshow(np.abs(S12), aspect='auto', cmap='viridis',
                         interpolation='nearest')
    axes[0].set_xlabel('allowedK index (beta2)')
    axes[0].set_ylabel('integerK index (beta1)')
    axes[0].set_title('|S12| — raw spatial overlap')
    plt.colorbar(im0, ax=axes[0])

    # Column-normalised S12
    col_max = np.max(np.abs(S12), axis=0)
    col_max = np.where(col_max < 1e-30, 1.0, col_max)
    S12_n = np.abs(S12) / col_max[None, :]
    im1 = axes[1].imshow(S12_n, aspect='auto', cmap='viridis',
                         interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_xlabel('allowedK index (beta2)')
    axes[1].set_ylabel('integerK index (beta1)')
    axes[1].set_title('|S12| column-normalised')
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(f'Spatial overlap S12  (N1={len(beta1)}, N2={len(beta2)})',
                 fontsize=12)
    fig.tight_layout()
    out = output_dir + '2D_spatial_overlap.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_gram_comparison(G1_t, G2_t, G12_t, G1_2D, G2_2D, G12_2D,
                         output_dir=OUTPUT_DIR):
    """Compare eigenvalue spectra before and after applying spatial overlap."""
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [
        (G1_t,  G1_2D,  'G1  (integerK self)'),
        (G2_t,  G2_2D,  'G2  (allowedK self)'),
        (G12_t, G12_2D, 'G12 (cross)'),
    ]
    for ax, (G_t, G_2D, title) in zip(axes, pairs):
        ev_t  = np.sort(np.abs(np.linalg.eigvalsh(G_t)))[::-1]
        ev_2D = np.sort(np.abs(np.linalg.eigvalsh(G_2D)))[::-1]
        ev_t  = ev_t  / ev_t[0]
        ev_2D = ev_2D / ev_2D[0]
        ax.semilogy(ev_t,  'b-o', ms=3, label='time-only')
        ax.semilogy(ev_2D, 'r-s', ms=3, label='2D space-time')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Index')
        ax.set_ylabel('λ_i / λ_max')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Gram eigenvalue spectra: time-only vs 2D space-time', fontsize=11)
    fig.tight_layout()
    out = output_dir + '2D_gram_comparison.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_heatmap(A_sparse, B_sparse, output_dir=OUTPUT_DIR, suffix=''):
    """Heatmap of column-normalised |coefficients| sorted by dominant k."""
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

    fig.suptitle('2D Space-Time CCA heatmap (column-normalised)', fontsize=12)
    fig.tight_layout()
    out = output_dir + f'2D_heatmap{suffix}.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Main analysis function
# ============================================================================

def cca_sparse_2D_analysis(
        N_t               = 1000,
        weights           = None,
        normalise_spatial = True,    # normalise S so diag(S11)=diag(S22)=1
        ev_threshold      = 1e-10,
        rho_min           = 0.99,
        rotation_method   = 'promax',
        folder_path       = FOLDER_PATH,
        output_dir        = OUTPUT_DIR,
        results_cache     = RESULTS_CACHE,
        force_recompute   = False,
):
    """
    Full 2D space-time CCA sparse joint eigenfunction analysis.

    Adds the hyperspherical harmonic spatial part to the CCA inner product,
    making Gram matrices sensitive to both time-evolution and spatial scale.
    For integerK (integer beta), G1_2D is diagonal => CCA decouples mode-by-mode.

    Parameters
    ----------
    N_t               : number of time-grid points
    weights           : per-perturbation-type weights {p: float}; None = unit
    normalise_spatial : if True, normalise S matrices so diagonal = 1
    ev_threshold      : relative eigenvalue cutoff for G^{-1/2} computation
    rho_min           : keep modes with canonical correlation > rho_min
    rotation_method   : 'varimax' | 'promax' | 'ica'
    folder_path       : path to data/ directory
    output_dir        : directory for output figures
    results_cache     : pickle file for caching results
    force_recompute   : ignore cache if True

    Returns
    -------
    dict with all results including 2D Gram matrices and spatial overlaps
    """

    if not force_recompute and os.path.exists(results_cache):
        print(f"Loading cached results from {results_cache}")
        with open(results_cache, 'rb') as f:
            return pickle.load(f)

    # ── Step 1: Load bases ────────────────────────────────────────────────────
    basis_1, basis_2, eta_grid = load_bases(folder_path, N_t)

    # ── Step 1b: Load k-values and convert to beta ────────────────────────────
    print("\n--- Step 1b: Loading k-values and converting to beta ---")
    k_values_1 = load_kvalues('integerK', folder_path)
    k_values_2 = load_kvalues('allowedK', folder_path)
    beta1 = k_to_beta(k_values_1)
    beta2 = k_to_beta(k_values_2)
    print_beta_info(beta1, beta2)

    # ── Step 2: Frobenius normalise ───────────────────────────────────────────
    print("\n--- Step 2: Frobenius normalisation ---")
    Xn1, norms_1 = frobenius_normalize(basis_1)
    Xn2, norms_2 = frobenius_normalize(basis_2)

    # ── Step 3a: Time-only Gram matrices ──────────────────────────────────────
    print("\n--- Step 3a: Time-only Gram matrices ---")
    G1_t, G2_t, G12_t = compute_time_gram_matrices(Xn1, Xn2, weights)
    print(f"  G1_t shape: {G1_t.shape},  G12_t shape: {G12_t.shape}")

    # ── Step 3b: Spatial overlap matrices ────────────────────────────────────
    print("\n--- Step 3b: Spatial overlap matrices ---")
    print("  Computing S11 (integerK self-overlap)...")
    S11 = compute_spatial_overlap(beta1, beta1)
    print("  Computing S22 (allowedK self-overlap)...")
    S22 = compute_spatial_overlap(beta2, beta2)
    print("  Computing S12 (cross-overlap)...")
    S12 = compute_spatial_overlap(beta1, beta2)
    print(f"  S11 diagonal range: [{np.diag(S11).min():.3e}, {np.diag(S11).max():.3e}]")
    print(f"  S11 off-diag max:   {(np.abs(S11) - np.diag(np.abs(np.diag(S11)))).max():.3e}")
    print(f"  S12 diagonal approx range: "
          f"[{np.diag(S12[:min(len(beta1),len(beta2)), :min(len(beta1),len(beta2))]).min():.3e}, "
          f"{np.diag(S12[:min(len(beta1),len(beta2)), :min(len(beta1),len(beta2))]).max():.3e}]")

    # ── Step 3c: 2D Gram matrices ─────────────────────────────────────────────
    print(f"\n--- Step 3c: 2D space-time Gram matrices "
          f"(normalise_spatial={normalise_spatial}) ---")
    G1_2D, G2_2D, G12_2D = compute_2d_gram_matrices(
        G1_t, G2_t, G12_t, S11, S22, S12,
        normalise_spatial=normalise_spatial)
    print(f"  G1_2D shape: {G1_2D.shape}")

    # ── Step 4: CCA (whitened SVD) ────────────────────────────────────────────
    print("\n--- Step 4: CCA on 2D Gram matrices ---")
    A, B, rho, rho_all = solve_cca(G1_2D, G2_2D, G12_2D, ev_threshold, rho_min)
    print(f"  A shape: {A.shape},  B shape: {B.shape}")

    # ── Step 5: Sparse rotation ───────────────────────────────────────────────
    print("\n--- Step 5: Sparse rotation ---")
    if A.shape[1] > 0:
        A_sparse, B_sparse = apply_sparse_rotation(A, B, method=rotation_method)
    else:
        print("  No valid modes — skipping rotation.")
        A_sparse = A.copy()
        B_sparse = B.copy()

    # ── Sparsity metrics ──────────────────────────────────────────────────────
    if A_sparse.shape[1] > 0:
        g_before_1 = np.mean([gini(A[:, i]) for i in range(A.shape[1])])
        g_after_1  = np.mean([gini(A_sparse[:, i]) for i in range(A_sparse.shape[1])])
        g_before_2 = np.mean([gini(B[:, i]) for i in range(B.shape[1])])
        g_after_2  = np.mean([gini(B_sparse[:, i]) for i in range(B_sparse.shape[1])])
        print(f"\nMean Gini sparsity (higher = sparser):")
        print(f"  Basis 1 — before: {g_before_1:.3f},  after: {g_after_1:.3f}")
        print(f"  Basis 2 — before: {g_before_2:.3f},  after: {g_after_2:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Plotting ---")
    plot_spatial_overlap(S12, beta1, beta2, output_dir)
    plot_gram_comparison(G1_t, G2_t, G12_t, G1_2D, G2_2D, G12_2D, output_dir)
    plot_gram_spectrum(G1_2D, G2_2D, output_dir)
    plot_canonical_correlations(rho_all, rho_min, output_dir)

    if A_sparse.shape[1] > 0:
        n_show = min(20, len(rho))
        plot_coefficients(A, B, A_sparse, B_sparse, rho,
                          N_plot=n_show, output_dir=output_dir)
        plot_heatmap(A_sparse, B_sparse, output_dir=output_dir)
        plot_reconstructed_timeseries(
            basis_1, basis_2,
            A_sparse, B_sparse, rho,
            eta_grid, norms_1, norms_2,
            N_plot=5, output_dir=output_dir,
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    results = dict(
        eta_grid          = eta_grid,
        G1_t=G1_t, G2_t=G2_t, G12_t=G12_t,
        G1_2D=G1_2D, G2_2D=G2_2D, G12_2D=G12_2D,
        S11=S11, S22=S22, S12=S12,
        A=A, B=B,
        A_sparse          = A_sparse,
        B_sparse          = B_sparse,
        rho               = rho,
        rho_all           = rho_all,
        basis_1           = basis_1,
        basis_2           = basis_2,
        norms_1           = norms_1,
        norms_2           = norms_2,
        k_values_1        = k_values_1,
        k_values_2        = k_values_2,
        beta1             = beta1,
        beta2             = beta2,
        weights           = weights,
        rotation_method   = rotation_method,
        normalise_spatial = normalise_spatial,
        ev_threshold      = ev_threshold,
        rho_min           = rho_min,
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

    results = cca_sparse_2D_analysis(
        N_t               = 1000,
        weights           = weights,
        normalise_spatial = True,
        ev_threshold      = 1e-10,
        rho_min           = 0.99,
        rotation_method   = 'promax',
        force_recompute   = True,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Valid modes found : {len(results['rho'])}")
    if len(results['rho']) > 0:
        print(f"  rho range         : {np.round(results['rho'], 6)}")
    print(f"  N_k1 (integerK)   : {results['G1_2D'].shape[0]}")
    print(f"  N_k2 (allowedK)   : {results['G2_2D'].shape[0]}")
    print(f"  normalise_spatial : {results['normalise_spatial']}")
    print("=" * 60)
