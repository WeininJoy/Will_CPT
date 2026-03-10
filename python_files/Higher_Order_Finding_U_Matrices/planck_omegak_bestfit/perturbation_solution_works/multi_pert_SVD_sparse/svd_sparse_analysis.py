# -*- coding: utf-8 -*-
"""
SVD-Based Sparse Joint Eigenfunction Method (Low-k Modes Only)

Pipeline:
  1. Load integerK (basis 1) and allowedK (basis 2) perturbation solutions.
  2. Frobenius-normalise each perturbation type and stack into a single
     matrix  X_stacked  of shape (4*N_t, N_k).
  3. Compute truncated SVD:  X = U Σ V^T  (handles rank deficiency cleanly).
  4. Mixing matrix  M = U1^T @ U2;  SVD of M gives principal angles.
     Modes with cos(angle) > threshold are "valid joint eigenfunctions".
  5. Back-transform to original k-space:  c_i = V @ Σ^{-1} @ e_i
     This gives UNIVERSAL coefficients valid for ALL perturbation types.
  6. Apply Promax oblique rotation (or Varimax / FastICA) to C = [c_1,...,c_m]
     to find maximally sparse linear combinations  c̃_j = Σ_i M_{ji} c_i.

See SVD_SPARSE_METHOD.md for full mathematical derivation.
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../multi_pert_sparse_ortho')

from multi_perturbation_analysis import generate_multi_perturbation_bases, fcb_time

# ---------------------------------------------------------------------------
PERTURBATION_TYPES = ['dr', 'dm', 'vr', 'vm']
FOLDER_PATH        = '../data/'
OUTPUT_DIR         = './figures/'
RESULTS_CACHE      = 'svd_sparse_results.pickle'
# ---------------------------------------------------------------------------


# ============================================================================
# STEP 1 — Load perturbation bases
# ============================================================================

def load_bases(folder_path=FOLDER_PATH, N_t=1000):
    """Load integerK and allowedK bases onto a uniform eta grid."""
    eta_grid = np.linspace(0, fcb_time, N_t)
    print("Loading integerK (basis 1)...")
    basis_1 = generate_multi_perturbation_bases("integerK", eta_grid,
                                                folder_path=folder_path)
    print("\nLoading allowedK (basis 2)...")
    basis_2 = generate_multi_perturbation_bases("allowedK", eta_grid,
                                                folder_path=folder_path)
    return basis_1, basis_2, eta_grid


# ============================================================================
# STEP 2 — Frobenius normalise and stack all perturbation types
# ============================================================================

def stack_and_normalize(basis_dict):
    """
    Stack the four perturbation types into one matrix of shape (4*N_t, N_k).
    Each type is divided by its Frobenius norm so all types contribute equally.

    Returns
    -------
    X_stacked : (4*N_t, N_k)
    norms     : dict  {p: Frobenius norm of basis_dict[p]}
    """
    blocks, norms = [], {}
    for p in PERTURBATION_TYPES:
        X = basis_dict[p]            # (N_t, N_k)
        n = np.linalg.norm(X, 'fro')
        norms[p] = n
        blocks.append(X / n)
    return np.vstack(blocks), norms  # (4*N_t, N_k)


# ============================================================================
# STEP 3 — Truncated SVD
# ============================================================================

def truncated_svd(X, sv_rel_threshold=1e-10):
    """
    Economy SVD of X (m x n) truncated to effective rank r.

        X ≈ U (m x r)  @  diag(S) (r x r)  @  V^T (r x n)

    Parameters
    ----------
    sv_rel_threshold : keep singular values  σ_i > threshold * σ_max

    Returns
    -------
    U (m, r),  S (r,),  V (n, r),  r
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(np.sum(S > sv_rel_threshold * S[0]))
    print(f"  Effective rank: {r} / {len(S)}  "
          f"(threshold = {sv_rel_threshold:.0e} × σ_max = "
          f"{sv_rel_threshold * S[0]:.3e},  σ_min_kept = {S[r-1]:.3e})")
    return U[:, :r], S[:r], Vt[:r, :].T, r


# ============================================================================
# STEP 4 — Mixing matrix and valid-mode extraction
# ============================================================================

def compute_mixing_and_valid_modes(U1, S1, V1,
                                   U2, S2, V2,
                                   eigenvalue_threshold=0.99):
    """
    Compute M = U1^T @ U2, take SVD of M.

    The singular values  Sm[i] = cos(θ_i)  are the cosines of the principal
    angles between the column spaces of X1_stacked and X2_stacked.

    Back-transform valid eigenvectors to original k-space:
        c_i = V @ Σ^{-1} @ e_i

    Parameters
    ----------
    eigenvalue_threshold : keep modes with  Sm[i] > threshold

    Returns
    -------
    C1      (N_k, n_valid)  k-space coefficients for basis 1
    C2      (N_k, n_valid)  k-space coefficients for basis 2
    Sm_valid (n_valid,)     principal-angle cosines of kept modes
    M_matrix (r, r)         mixing matrix
    Sm_all   (r,)           all singular values of M (for plotting)
    """
    r = min(U1.shape[1], U2.shape[1])
    M = U1[:, :r].T @ U2[:, :r]           # (r, r)

    Um, Sm, Vmh = np.linalg.svd(M)        # M = Um @ diag(Sm) @ Vmh
    # Note: np.linalg.svd returns Vmh already transposed, i.e. V^H

    valid     = Sm > eigenvalue_threshold
    n_valid   = int(np.sum(valid))
    print(f"  Valid modes (Sm > {eigenvalue_threshold}): {n_valid} / {r}")
    if n_valid > 0:
        print(f"  Sm range of valid modes: "
              f"{Sm[valid].min():.6f} – {Sm[valid].max():.6f}")

    # Eigenvectors in compressed U-spaces
    e1 = Um[:, valid]          # (r, n_valid) — in U1 space
    e2 = Vmh[valid, :].T       # (r, n_valid) — in U2 space

    # Back-transform: c_i = V[:, :r] @ diag(1/S[:r]) @ e_i
    inv_S1 = 1.0 / S1[:r]
    inv_S2 = 1.0 / S2[:r]
    C1 = V1[:, :r] @ (inv_S1[:, None] * e1)   # (N_k1, n_valid)
    C2 = V2[:, :r] @ (inv_S2[:, None] * e2)   # (N_k2, n_valid)

    return C1, C2, Sm[valid], M, Sm


# ============================================================================
# STEP 5a — Varimax (orthogonal rotation, numpy only)
# ============================================================================

def varimax_rotation(Phi, gamma=1.0, q=500, tol=1e-6):
    """
    Varimax rotation: find orthogonal R (K×K) to maximise sparsity of Phi @ R.
    Phi has shape (N, K).
    Returns (Phi_rotated, R).
    """
    p, k = Phi.shape
    rs  = np.random.RandomState(42)
    R   = np.linalg.qr(rs.randn(k, k))[0]
    d   = 0.0
    for i in range(q):
        d_old = d
        B     = Phi @ R
        M2    = B ** 2
        u, s, vh = np.linalg.svd(
            Phi.T @ (B * M2
                     - (gamma / p) * B @ np.diag(M2.sum(axis=0)))
        )
        R = u @ vh
        d = s.sum()
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            print(f"  Varimax converged in {i+1} iterations")
            return Phi @ R, R
    print(f"  Varimax reached max iterations ({q})")
    return Phi @ R, R


# ============================================================================
# STEP 5b — Promax (oblique rotation, numpy only)
# ============================================================================

def promax_rotation(Phi, power=3, gamma=1.0, q=500, tol=1e-6):
    """
    Promax oblique rotation on Phi (N × K).

    Algorithm:
      1. Apply Varimax  →  F_v = Phi @ R_v
      2. Target  T = sign(F_v) × |F_v|^power   (element-wise)
      3. Oblique fit  L = lstsq(F_v, T)
      4. Normalise columns of L
      5. Pattern = F_v @ L_norm

    Returns
    -------
    pattern  (N, K)  : oblique rotated loadings (sparse columns)
    R_v              : underlying Varimax rotation matrix (K×K)
    L_norm           : oblique transformation (K×K)
                       so  pattern = Phi @ R_v @ L_norm
    """
    # Step 1 — Varimax starting point
    F_v, R_v = varimax_rotation(Phi, gamma=gamma, q=q, tol=tol)

    # Step 2 — Power target (sign-preserving)
    T = np.sign(F_v) * (np.abs(F_v) ** power)

    # Step 3 — Least-squares oblique fit  F_v @ L ≈ T
    L, _, _, _ = np.linalg.lstsq(F_v, T, rcond=None)   # (K, K)

    # Step 4 — Normalise columns of L to unit length
    col_norms = np.linalg.norm(L, axis=0)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    L_norm    = L / col_norms

    pattern = F_v @ L_norm                              # (N, K)
    return pattern, R_v, L_norm


# ============================================================================
# STEP 5c — FastICA (sklearn)
# ============================================================================

def ica_rotation(C1, C2):
    """
    Apply FastICA to C1 to find maximally independent (≈ sparse) columns.
    The same linear transformation is then applied to C2.

    Returns C1_sparse, C2_sparse.
    """
    from sklearn.decomposition import FastICA
    m   = C1.shape[1]
    ica = FastICA(n_components=m, random_state=42, max_iter=2000, tol=1e-5)
    C1_sparse = ica.fit_transform(C1)                          # (N_k, m)
    # Recover rotation matrix: C1_sparse ≈ C1 @ W
    W, _, _, _ = np.linalg.lstsq(C1, C1_sparse, rcond=None)   # (m, m)
    C2_sparse  = C2 @ W                                        # (N_k, m)
    print("  FastICA done; rotation recovered from C1.")
    return C1_sparse, C2_sparse


# ============================================================================
# STEP 5 — Dispatch sparse rotation; apply same transform to both bases
# ============================================================================

def apply_sparse_rotation(C1, C2, method='promax'):
    """
    Apply a sparsifying rotation to C1 and apply the same transformation
    to C2 so both share the same k-space linear combinations.

    method : 'varimax' | 'promax' | 'ica'
    Returns C1_sparse, C2_sparse  each of shape (N_k, m).
    """
    print(f"\nApplying {method} rotation...")

    if method == 'ica':
        return ica_rotation(C1, C2)

    if method == 'promax':
        C1_sparse, R_v, L_norm = promax_rotation(C1)
        # Apply same oblique transform to basis-2 coefficients
        C2_sparse = (C2 @ R_v) @ L_norm
        return C1_sparse, C2_sparse

    # Default: Varimax
    C1_sparse, R = varimax_rotation(C1)
    C2_sparse    = C2 @ R
    return C1_sparse, C2_sparse


# ============================================================================
# Utility — Gini coefficient (sparsity measure)
# ============================================================================

def gini(v):
    """Gini coefficient of |v|. Range [0,1]; 1 = perfectly sparse."""
    v = np.sort(np.abs(v.ravel()))
    n = len(v)
    s = v.sum()
    if s < 1e-30:
        return 0.0
    return 1.0 - 2.0 * np.dot(v, n - np.arange(n)) / (n * s)


# ============================================================================
# Plotting helpers
# ============================================================================

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_singular_values(S1, S2, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, S, label in zip(axes,
                             [S1, S2],
                             ['Basis 1 (integerK)', 'Basis 2 (allowedK)']):
        ax.semilogy(np.arange(1, len(S)+1), S / S[0], 'o-', ms=4)
        ax.axhline(1e-10, color='r', ls='--', alpha=0.6, label='threshold 1e-10')
        ax.set_xlabel('Index')
        ax.set_ylabel('σ_i / σ_max')
        ax.set_title(f'Singular values: {label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir + 'svd_singular_values.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_principal_angles(Sm_all, threshold, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, len(Sm_all)+1), Sm_all, 'o-', ms=4)
    ax.axhline(threshold, color='r', ls='--',
               label=f'threshold = {threshold}')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('cos(principal angle)')
    ax.set_title('Principal angles between Basis 1 and Basis 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir + 'svd_principal_angles.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_coefficients(C1, C2, C1_sparse, C2_sparse, Sm_valid,
                      N_plot=10, output_dir=OUTPUT_DIR):
    """Four-column plot: orig B1 | orig B2 | sparse B1 | sparse B2."""
    _ensure_dir(output_dir)
    N_k1   = C1.shape[0]
    N_k2   = C2.shape[0]
    N_plot = min(N_plot, C1.shape[1])

    dom1  = np.argmax(np.abs(C1_sparse), axis=0)
    order = np.argsort(dom1)

    fig, axes = plt.subplots(N_plot, 4, figsize=(14, 1.8 * N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("k-space coefficients — original vs sparse (both bases)", fontsize=11)

    for row, idx in enumerate(order[:N_plot]):
        d_list = [C1[:, idx].real, C2[:, idx].real,
                  C1_sparse[:, idx].real, C2_sparse[:, idx].real]
        t_list = [f"B1 orig  Sm={Sm_valid[idx]:.4f}",
                  "B2 orig",
                  f"B1 sparse  dom k={dom1[idx]+1}",
                  "B2 sparse"]
        c_list = ['steelblue', 'seagreen', 'steelblue', 'seagreen']
        nk_list = [N_k1, N_k2, N_k1, N_k2]

        for col, (d, t, c, nk) in enumerate(zip(d_list, t_list, c_list, nk_list)):
            ax = axes[row, col]
            ax.bar(np.arange(1, nk + 1), d, color=c, alpha=0.7, width=0.6)
            ax.set_title(t, fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(labelsize=7)
            ax.label_outer()

    for col, lbl in enumerate(["k index (B1)", "k index (B2)",
                                "k index (B1)", "k index (B2)"]):
        axes[-1, col].set_xlabel(lbl, fontsize=8)

    fig.tight_layout()
    out = output_dir + 'svd_coefficients.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def normalize_columns(C):
    """Normalize each column of C to unit L2 norm. Returns normalized array."""
    col_norms = np.linalg.norm(C, axis=0)
    col_norms = np.where(col_norms < 1e-30, 1.0, col_norms)
    return C / col_norms


def plot_heatmap(C1_sparse, C2_sparse, output_dir=OUTPUT_DIR):
    """
    Heatmap of column-normalized |coefficients| sorted by dominant k in basis 1.
    Each column is divided by its L2 norm before plotting so the color scale
    reflects relative weight across k-modes within each sparse solution.
    """
    _ensure_dir(output_dir)

    # Normalize each column (sparse solution) to unit L2 norm
    C1_norm = normalize_columns(C1_sparse)
    C2_norm = normalize_columns(C2_sparse)

    dom1  = np.argmax(np.abs(C1_norm), axis=0)
    order = np.argsort(dom1)
    C1_s  = C1_norm[:, order]
    C2_s  = C2_norm[:, order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(np.abs(C1_s.T), aspect='auto', cmap='Blues',
                     interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xlabel('k index (integerK)')
    ax1.set_ylabel('Mode (sorted by dominant k)')
    ax1.set_title('|Normalized sparse coefficients| — Basis 1')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.abs(C2_s.T), aspect='auto', cmap='Greens',
                     interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('k index (allowedK)')
    ax2.set_ylabel('Mode (sorted by dominant k)')
    ax2.set_title('|Normalized sparse coefficients| — Basis 2')
    plt.colorbar(im2, ax=ax2)

    fig.suptitle('Joint Sparsity Heatmap (SVD method, column-normalized)', fontsize=12)
    fig.tight_layout()
    out = output_dir + 'svd_heatmap.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_reconstructed_timeseries(basis_1, basis_2,
                                  C1_sparse, C2_sparse, Sm_valid,
                                  eta_grid, N_plot=5,
                                  output_dir=OUTPUT_DIR):
    """
    Reconstruct time series from both bases using sparse coefficients
    and overlay them to verify agreement.
    """
    _ensure_dir(output_dir)
    dom1   = np.argmax(np.abs(C1_sparse), axis=0)
    order  = np.argsort(dom1)
    N_plot = min(N_plot, C1_sparse.shape[1])

    fig, axes = plt.subplots(N_plot, 4, figsize=(14, 2.2 * N_plot),
                             constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("SVD sparse modes: time-series (Basis 1 red vs Basis 2 green)",
                 fontsize=11)

    for row, idx in enumerate(order[:N_plot]):
        dom_k = dom1[idx]
        for col, p in enumerate(PERTURBATION_TYPES):
            ax  = axes[row, col]
            s1  = basis_1[p] @ C1_sparse[:, idx]
            s2  = basis_2[p] @ C2_sparse[:, idx]
            if np.dot(s1, s2) < 0:
                s2 = -s2
            ax.plot(eta_grid, s1, 'r-',  lw=2,   alpha=0.85, label='B1')
            ax.plot(eta_grid, s2, 'g--', lw=1.5, alpha=0.85, label='B2')
            ax.set_title(f"{p}  k={dom_k+1}  Sm={Sm_valid[idx]:.4f}",
                         fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            if row == N_plot - 1:
                ax.set_xlabel("η", fontsize=9)

    out = output_dir + 'svd_timeseries.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Main analysis function
# ============================================================================

def svd_sparse_analysis(
        N_t                 = 1000,
        sv_threshold        = 1e-10,
        eigenvalue_threshold= 0.99,
        rotation_method     = 'promax',    # 'varimax' | 'promax' | 'ica'
        folder_path         = FOLDER_PATH,
        output_dir          = OUTPUT_DIR,
        results_cache       = RESULTS_CACHE,
        force_recompute     = False,
):
    """
    Full SVD-based sparse joint eigenfunction analysis for low-k modes.

    Parameters
    ----------
    N_t                 : number of time-grid points
    sv_threshold        : relative singular-value cutoff for rank truncation
    eigenvalue_threshold: minimum cos(principal angle) to count as valid mode
    rotation_method     : sparsifying rotation — 'varimax', 'promax', or 'ica'
    folder_path         : path to data/ directory
    output_dir          : directory for output figures
    results_cache       : pickle file to cache/reload results
    force_recompute     : ignore cache if True

    Returns
    -------
    dict with keys: eta_grid, S1, S2, V1, V2, Sm_valid, Sm_all,
                    C1, C2, C1_sparse, C2_sparse, basis_1, basis_2, ...
    """

    if not force_recompute and os.path.exists(results_cache):
        print(f"Loading cached results from {results_cache}")
        with open(results_cache, 'rb') as f:
            return pickle.load(f)

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    basis_1, basis_2, eta_grid = load_bases(folder_path, N_t)

    # ── Step 2: Stack & normalise ─────────────────────────────────────────────
    print("\n--- Step 2: Stacking and normalising ---")
    X1_stacked, norms_1 = stack_and_normalize(basis_1)
    X2_stacked, norms_2 = stack_and_normalize(basis_2)
    print(f"Stacked shapes:  X1 = {X1_stacked.shape},  X2 = {X2_stacked.shape}")

    # ── Step 3: SVD ───────────────────────────────────────────────────────────
    print("\n--- Step 3: Truncated SVD ---")
    print("Basis 1:")
    U1, S1, V1, r1 = truncated_svd(X1_stacked, sv_threshold)
    print("Basis 2:")
    U2, S2, V2, r2 = truncated_svd(X2_stacked, sv_threshold)

    # ── Step 4: Valid modes ───────────────────────────────────────────────────
    print("\n--- Step 4: Mixing matrix and valid modes ---")
    C1, C2, Sm_valid, M_matrix, Sm_all = compute_mixing_and_valid_modes(
        U1, S1, V1, U2, S2, V2, eigenvalue_threshold)
    print(f"C1 shape: {C1.shape},  C2 shape: {C2.shape}")

    # ── Step 5: Sparse rotation ───────────────────────────────────────────────
    print("\n--- Step 5: Sparse rotation ---")
    C1_sparse, C2_sparse = apply_sparse_rotation(C1, C2, method=rotation_method)

    # ── Sparsity metrics ──────────────────────────────────────────────────────
    g_before_1 = np.mean([gini(C1[:, i]) for i in range(C1.shape[1])])
    g_after_1  = np.mean([gini(C1_sparse[:, i]) for i in range(C1_sparse.shape[1])])
    g_before_2 = np.mean([gini(C2[:, i]) for i in range(C2.shape[1])])
    g_after_2  = np.mean([gini(C2_sparse[:, i]) for i in range(C2_sparse.shape[1])])
    print(f"\nMean Gini sparsity (higher = sparser):")
    print(f"  Basis 1 — before: {g_before_1:.3f},  after: {g_after_1:.3f}")
    print(f"  Basis 2 — before: {g_before_2:.3f},  after: {g_after_2:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Plotting ---")
    plot_singular_values(S1, S2, output_dir)
    plot_principal_angles(Sm_all, eigenvalue_threshold, output_dir)
    n_show = min(20, len(Sm_valid))
    plot_coefficients(C1, C2, C1_sparse, C2_sparse, Sm_valid,
                      N_plot=n_show, output_dir=output_dir)
    plot_heatmap(C1_sparse, C2_sparse, output_dir)
    plot_reconstructed_timeseries(basis_1, basis_2,
                                  C1_sparse, C2_sparse, Sm_valid,
                                  eta_grid, N_plot=5, output_dir=output_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    results = dict(
        eta_grid            = eta_grid,
        S1=S1, S2=S2, V1=V1, V2=V2,
        r1=r1, r2=r2,
        Sm_valid            = Sm_valid,
        Sm_all              = Sm_all,
        C1=C1, C2=C2,
        C1_sparse           = C1_sparse,
        C2_sparse           = C2_sparse,
        basis_1             = basis_1,
        basis_2             = basis_2,
        norms_1             = norms_1,
        norms_2             = norms_2,
        rotation_method     = rotation_method,
        sv_threshold        = sv_threshold,
        eigenvalue_threshold= eigenvalue_threshold,
    )
    with open(results_cache, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_cache}")
    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    results = svd_sparse_analysis(
        N_t                  = 1000,
        sv_threshold         = 1e-10,
        eigenvalue_threshold = 0.99,
        rotation_method      = 'promax',    # 'varimax' | 'promax' | 'ica'
        force_recompute      = True,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Valid modes found : {len(results['Sm_valid'])}")
    print(f"  Sm_valid          : {np.round(results['Sm_valid'], 6)}")
    print(f"  Effective rank r1 : {results['r1']}")
    print(f"  Effective rank r2 : {results['r2']}")
    print("=" * 60)
