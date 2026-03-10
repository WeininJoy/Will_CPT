# -*- coding: utf-8 -*-
"""
Matching-Solution Sparse Joint Eigenfunction Method

Finds coefficient vectors a, b in R^{N_k} such that
    X_1^p @ a = X_2^p @ b   for ALL perturbation types p simultaneously,
rather than merely maximising the correlation between them.

Pipeline (see cca_formulation/cca_formulation_match_sol.pdf):
  1. Load integerK (basis 1) and allowedK (basis 2) perturbation solutions.
  2. Frobenius-normalise each perturbation type.
  3. Compute per-field Gram matrices G1^p, G2^p, G12^p  (each N_k x N_k).
  4. Construct the (2*N_k, 2*N_k) block matrix:
         M = sum_p w_p [[G1^p, -G12^p], [-(G12^p)^T, G2^p]]
  5. Solve M c_i = lambda_i c_i for the smallest eigenvalues.
     Each eigenvector c_i = [a_i; b_i] splits into basis-1 and basis-2
     k-space coefficients.  lambda_i = sum_p w_p ||X_1^p a_i - X_2^p b_i||^2,
     so lambda_i = 0 means exact per-field agreement.
  6. Keep modes with lambda_i / lambda_max < lambda_threshold.
  7. Apply Promax oblique rotation (or Varimax / FastICA) for sparsity.

Key difference from cca_sparse_analysis.py:
  - CCA maximises a weighted-average CORRELATION -> three fields can outvote dm.
  - This method minimises the sum of SQUARED DIFFERENCES -> every field is
    penalised individually; no field can be hidden by others.
  - lambda_i = 0 guarantees exact equality, not just proportionality.

See cca_formulation/cca_formulation_match_sol.pdf for full derivation.
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
OUTPUT_DIR         = './figures_match/'
RESULTS_CACHE      = 'cca_match_results.pickle'
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
# STEP 2 — Frobenius normalise
# ============================================================================

def frobenius_normalize(basis_dict):
    """
    Divide each perturbation-type matrix by its Frobenius norm.

    Returns
    -------
    Xn    : dict {p: normalised (N_t, N_k) array}
    norms : dict {p: Frobenius norm}
    """
    Xn, norms = {}, {}
    for p in PERTURBATION_TYPES:
        X = basis_dict[p]
        n = np.linalg.norm(X, 'fro')
        norms[p] = n
        Xn[p] = X / n
    return Xn, norms


# ============================================================================
# STEP 3 — Per-field Gram matrices
# ============================================================================

def compute_perfield_gram(Xn1, Xn2):
    """
    Compute per-field Gram matrices for each perturbation type p:

        G1p[p]  = (Xn1^p)^T Xn1^p    shape (N_k, N_k)
        G2p[p]  = (Xn2^p)^T Xn2^p
        G12p[p] = (Xn1^p)^T Xn2^p

    Returns dicts keyed by perturbation type.
    """
    G1p, G2p, G12p = {}, {}, {}
    for p in PERTURBATION_TYPES:
        X1 = Xn1[p]   # (N_t, N_k)
        X2 = Xn2[p]
        G1p[p]  = X1.T @ X1
        G2p[p]  = X2.T @ X2
        G12p[p] = X1.T @ X2
    return G1p, G2p, G12p


# ============================================================================
# STEP 4 — Block matrix M
# ============================================================================

def build_block_matrices(G1p, G2p, G12p, weights=None):
    """
    Construct two (2*N_k, 2*N_k) matrices for the generalized eigenvalue problem
    M c = lambda N c.

    M = sum_p w_p [[ G1^p,      -G12^p        ],   (residual matrix)
                   [ -(G12^p)^T,  G2^p        ]]

    N = [[ G1,  0  ],                               (normalization matrix)
         [  0,  G2 ]]
    where G1 = sum_p w_p G1^p, G2 = sum_p w_p G2^p.

    The generalized Rayleigh quotient lambda = c^T M c / c^T N c equals:
        sum_p w_p ||Xn1^p a - Xn2^p b||^2 / (a^T G1 a + b^T G2 b)

    This prevents the trivial zero solution: the constraint c^T N c = 1 ensures
    the coefficients produce non-trivial time evolutions in both bases.
    lambda = 0 iff Xn1^p a = Xn2^p b for all p (exact per-field match).

    Parameters
    ----------
    weights : dict {p: float} or None (unit weights)

    Returns
    -------
    M : (2*N_k, 2*N_k) symmetric PSD  — residual matrix
    N : (2*N_k, 2*N_k) symmetric PSD  — normalization matrix
    """
    if weights is None:
        weights = {p: 1.0 for p in PERTURBATION_TYPES}

    N_k = G1p[PERTURBATION_TYPES[0]].shape[0]
    M   = np.zeros((2 * N_k, 2 * N_k))
    Nm  = np.zeros((2 * N_k, 2 * N_k))

    for p in PERTURBATION_TYPES:
        w   = weights[p]
        G1  = G1p[p]
        G2  = G2p[p]
        G12 = G12p[p]

        M[:N_k, :N_k] += w * G1
        M[N_k:, N_k:] += w * G2
        M[:N_k, N_k:] -= w * G12
        M[N_k:, :N_k] -= w * G12.T

        Nm[:N_k, :N_k] += w * G1
        Nm[N_k:, N_k:] += w * G2

    # Symmetrise to remove floating-point asymmetry
    M  = 0.5 * (M  + M.T)
    Nm = 0.5 * (Nm + Nm.T)
    return M, Nm


# ============================================================================
# STEP 5 — Eigenvalue problem: find smallest eigenvectors of M
# ============================================================================

def solve_matching(M, Nm, lambda_threshold=1e-3, reg=1e-10):
    """
    Solve the generalized eigenvalue problem  M c = lambda N c.

    The Rayleigh quotient
        lambda = c^T M c / c^T N c
              = sum_p w_p ||Xn1^p a - Xn2^p b||^2 / (a^T G1 a + b^T G2 b)
    is minimised.  lambda = 0 means exact per-field match for all p.
    The normalization c^T N c = 1 prevents trivial zero solutions by requiring
    non-trivial time evolution in both bases.

    Parameters
    ----------
    M                : (2*N_k, 2*N_k) residual matrix
    Nm               : (2*N_k, 2*N_k) normalization matrix (block-diagonal G1, G2)
    lambda_threshold : keep modes with lambda_i / lambda_max < lambda_threshold
    reg              : relative regularization added to Nm to handle near-singularity

    Returns
    -------
    A         (N_k, m)  basis-1 k-space coefficients a_i = c_i[:N_k]
    B         (N_k, m)  basis-2 k-space coefficients b_i = c_i[N_k:]
    lam_valid (m,)      generalized eigenvalues of kept modes
    lam_all   (r,)      all generalized eigenvalues (ascending)
    """
    from scipy.linalg import eigh as scipy_eigh

    # Regularize Nm to handle near-singular Gram matrices
    Nm_reg = Nm + reg * np.trace(Nm) / Nm.shape[0] * np.eye(Nm.shape[0])

    # Solve generalized eigenvalue problem; eigh returns ascending eigenvalues
    lam_all, evecs = scipy_eigh(M, Nm_reg)

    # Clip small negative values from numerical noise
    lam_all = np.clip(lam_all, 0.0, None)

    lam_max = lam_all[-1]
    valid   = (lam_all / lam_max) < lambda_threshold
    m       = int(np.sum(valid))

    print(f"  lambda_max = {lam_max:.4e}")
    print(f"  Valid modes (lambda/lambda_max < {lambda_threshold}): {m} / {len(lam_all)}")
    if m > 0:
        print(f"  lambda range of valid modes: "
              f"{lam_all[valid].min():.4e} – {lam_all[valid].max():.4e}")
        print(f"  lambda/lambda_max range: "
              f"{lam_all[valid].min()/lam_max:.2e} – "
              f"{lam_all[valid].max()/lam_max:.2e}")

    N_k = M.shape[0] // 2
    C   = evecs[:, valid]           # (2*N_k, m)
    A   = C[:N_k, :]                # (N_k, m) — basis-1 coefficients
    B   = C[N_k:, :]                # (N_k, m) — basis-2 coefficients

    return A, B, lam_all[valid], lam_all


# ============================================================================
# STEP 6 — Sparse rotation (same logic as cca_sparse_analysis.py)
# ============================================================================

def varimax_rotation(Phi, gamma=1.0, q=500, tol=1e-6):
    """Varimax orthogonal rotation on Phi (N x K). Returns (Phi_rotated, R)."""
    p, k = Phi.shape
    rs   = np.random.RandomState(42)
    R    = np.linalg.qr(rs.randn(k, k))[0]
    d    = 0.0
    for i in range(q):
        d_old = d
        Bm    = Phi @ R
        M2    = Bm ** 2
        u, s, vh = np.linalg.svd(
            Phi.T @ (Bm * M2
                     - (gamma / p) * Bm @ np.diag(M2.sum(axis=0)))
        )
        R = u @ vh
        d = s.sum()
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            print(f"  Varimax converged in {i+1} iterations")
            return Phi @ R, R
    print(f"  Varimax reached max iterations ({q})")
    return Phi @ R, R


def promax_rotation(Phi, power=3, gamma=1.0, q=500, tol=1e-6):
    """Promax oblique rotation on Phi (N x K). Returns (pattern, R_v, L_norm)."""
    F_v, R_v   = varimax_rotation(Phi, gamma=gamma, q=q, tol=tol)
    T          = np.sign(F_v) * (np.abs(F_v) ** power)
    L, _, _, _ = np.linalg.lstsq(F_v, T, rcond=None)
    col_norms  = np.linalg.norm(L, axis=0)
    col_norms  = np.where(col_norms < 1e-12, 1.0, col_norms)
    L_norm     = L / col_norms
    return F_v @ L_norm, R_v, L_norm


def ica_rotation(A, B):
    """FastICA on A; apply same rotation to B. Returns A_sparse, B_sparse."""
    from sklearn.decomposition import FastICA
    m    = A.shape[1]
    ica  = FastICA(n_components=m, random_state=42, max_iter=2000, tol=1e-5)
    A_sp = ica.fit_transform(A)
    W, _, _, _ = np.linalg.lstsq(A, A_sp, rcond=None)
    B_sp = B @ W
    print("  FastICA done; rotation recovered from A.")
    return A_sp, B_sp


def apply_sparse_rotation(A, B, method='promax'):
    """
    Sparsify A via oblique rotation; apply same transform to B.
    method : 'varimax' | 'promax' | 'ica'
    Returns A_sparse, B_sparse  each (N_k, m).
    """
    print(f"\nApplying {method} rotation...")
    if method == 'ica':
        return ica_rotation(A, B)
    if method == 'promax':
        A_sparse, R_v, L_norm = promax_rotation(A)
        B_sparse = (B @ R_v) @ L_norm
        return A_sparse, B_sparse
    # Varimax
    A_sparse, R = varimax_rotation(A)
    B_sparse    = B @ R
    return A_sparse, B_sparse


# ============================================================================
# Utilities
# ============================================================================

def gini(v):
    """Gini coefficient of |v|. Range [0,1]; 1 = perfectly sparse."""
    v = np.sort(np.abs(v.ravel()))
    n = len(v)
    s = v.sum()
    if s < 1e-30:
        return 0.0
    return 1.0 - 2.0 * np.dot(v, n - np.arange(n)) / (n * s)


def normalize_columns(C):
    norms = np.linalg.norm(C, axis=0)
    norms = np.where(norms < 1e-30, 1.0, norms)
    return C / norms


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def compute_per_field_residuals(Xn1, Xn2, A, B):
    """
    Compute per-field residual norms ||Xn1^p a_i - Xn2^p b_i|| for each
    mode i and field p.

    Returns residuals of shape (m, N_p).
    """
    m    = A.shape[1]
    N_p  = len(PERTURBATION_TYPES)
    res  = np.zeros((m, N_p))
    for j, p in enumerate(PERTURBATION_TYPES):
        for i in range(m):
            r       = Xn1[p] @ A[:, i] - Xn2[p] @ B[:, i]
            res[i, j] = np.linalg.norm(r)
    return res


# ============================================================================
# Plotting
# ============================================================================

def plot_eigenvalue_spectrum(lam_all, lambda_threshold, output_dir=OUTPUT_DIR):
    """Plot all eigenvalues normalised by lambda_max."""
    _ensure_dir(output_dir)
    lam_max = lam_all[-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(np.arange(1, len(lam_all)+1), lam_all / lam_max, 'o-', ms=4)
    ax.axhline(lambda_threshold, color='r', ls='--',
               label=f'threshold = {lambda_threshold}')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('λ_i / λ_max')
    ax.set_title('Matching-solution eigenvalue spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir + 'match_eigenvalue_spectrum.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_per_field_residuals(residuals, lam_valid, output_dir=OUTPUT_DIR):
    """
    Bar chart of per-field residual norms for each valid mode.
    Helps confirm that all fields are well-matched, not just an average.
    """
    _ensure_dir(output_dir)
    m   = residuals.shape[0]
    x   = np.arange(m)
    w   = 0.2
    fig, ax = plt.subplots(figsize=(max(8, m * 0.6), 4))
    for j, p in enumerate(PERTURBATION_TYPES):
        ax.bar(x + j * w, residuals[:, j], width=w, label=p, alpha=0.8)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([f"mode {i+1}\nλ={lam_valid[i]:.2e}" for i in range(m)],
                       fontsize=7)
    ax.set_ylabel('||X1 a - X2 b||  (residual norm)')
    ax.set_title('Per-field residual norms — all should be small')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = output_dir + 'match_per_field_residuals.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_coefficients(A, B, A_sparse, B_sparse, lam_valid,
                      N_plot=10, output_dir=OUTPUT_DIR):
    """Four-column bar plot: orig B1 | orig B2 | sparse B1 | sparse B2."""
    _ensure_dir(output_dir)
    N_k    = A.shape[0]
    N_plot = min(N_plot, A.shape[1])

    dom1  = np.argmax(np.abs(A_sparse), axis=0)
    order = np.argsort(dom1)

    fig, axes = plt.subplots(N_plot, 4, figsize=(14, 1.8 * N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("k-space coefficients — original vs sparse (both bases)", fontsize=11)

    for row, idx in enumerate(order[:N_plot]):
        d_list = [A[:, idx].real, B[:, idx].real,
                  A_sparse[:, idx].real, B_sparse[:, idx].real]
        t_list = [f"B1 orig  λ={lam_valid[idx]:.2e}", "B2 orig",
                  f"B1 sparse  dom k={dom1[idx]+1}", "B2 sparse"]
        c_list = ['steelblue', 'seagreen', 'steelblue', 'seagreen']

        for col, (d, t, c) in enumerate(zip(d_list, t_list, c_list)):
            ax = axes[row, col]
            ax.bar(np.arange(1, N_k + 1), d, color=c, alpha=0.7, width=0.6)
            ax.set_title(t, fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(labelsize=7)
            ax.label_outer()

    for col, lbl in enumerate(["k index (B1)", "k index (B2)"] * 2):
        axes[-1, col].set_xlabel(lbl, fontsize=8)

    fig.tight_layout()
    out = output_dir + 'match_coefficients.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_heatmap(A_sparse, B_sparse, output_dir=OUTPUT_DIR):
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
    ax1.set_title('|Normalised sparse coefficients| — Basis 1 (match)')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.abs(B_s.T), aspect='auto', cmap='Greens',
                     interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('k index (allowedK)')
    ax2.set_ylabel('Mode (sorted by dominant k)')
    ax2.set_title('|Normalised sparse coefficients| — Basis 2 (match)')
    plt.colorbar(im2, ax=ax2)

    fig.suptitle('Joint Sparsity Heatmap (matching-solution method)', fontsize=12)
    fig.tight_layout()
    out = output_dir + 'match_heatmap.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_reconstructed_timeseries(basis_1, basis_2,
                                  A_sparse, B_sparse, lam_valid,
                                  eta_grid, N_plot=5,
                                  output_dir=OUTPUT_DIR):
    """
    Overlay time series from both bases using sparse coefficients.
    All four perturbation types should now align for valid modes.
    """
    _ensure_dir(output_dir)
    dom1   = np.argmax(np.abs(A_sparse), axis=0)
    order  = np.argsort(dom1)
    N_plot = min(N_plot, A_sparse.shape[1])

    fig, axes = plt.subplots(N_plot, 4, figsize=(14, 2.2 * N_plot),
                             constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Match-solution sparse modes: time-series "
                 "(Basis 1 red vs Basis 2 green)", fontsize=11)

    for row, idx in enumerate(order[:N_plot]):
        dom_k = dom1[idx]
        for col, p in enumerate(PERTURBATION_TYPES):
            ax = axes[row, col]
            s1 = basis_1[p] @ A_sparse[:, idx]
            s2 = basis_2[p] @ B_sparse[:, idx]
            if np.dot(s1, s2) < 0:
                s2 = -s2
            ax.plot(eta_grid, s1, 'r-',  lw=2,   alpha=0.85, label='B1')
            ax.plot(eta_grid, s2, 'g--', lw=1.5, alpha=0.85, label='B2')
            ax.set_title(f"{p}  k={dom_k+1}  λ={lam_valid[idx]:.2e}",
                         fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            if row == N_plot - 1:
                ax.set_xlabel("η", fontsize=9)

    out = output_dir + 'match_timeseries.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Main analysis function
# ============================================================================

def cca_sparse_match_analysis(
        N_t              = 1000,
        weights          = None,       # dict {p: float} or None for unit weights
        lambda_threshold = 1e-3,       # keep modes with lambda_i/lambda_max < threshold
        rotation_method  = 'promax',   # 'varimax' | 'promax' | 'ica'
        folder_path      = FOLDER_PATH,
        output_dir       = OUTPUT_DIR,
        results_cache    = RESULTS_CACHE,
        force_recompute  = False,
):
    """
    Full matching-solution sparse joint eigenfunction analysis.

    Finds modes where X_1^p a = X_2^p b for ALL perturbation types p, not
    just in a weighted-average correlation sense.

    Parameters
    ----------
    N_t              : number of time-grid points
    weights          : per-perturbation-type weights {p: float}; None = unit weights
    lambda_threshold : relative eigenvalue cutoff lambda_i/lambda_max < threshold
    rotation_method  : sparsifying rotation — 'varimax', 'promax', or 'ica'
    folder_path      : path to data/ directory
    output_dir       : directory for output figures
    results_cache    : pickle file to cache/reload results
    force_recompute  : ignore cache if True

    Returns
    -------
    dict with keys: eta_grid, M, A, B, A_sparse, B_sparse,
                    lam_valid, lam_all, residuals,
                    basis_1, basis_2, norms_1, norms_2, ...
    """

    if not force_recompute and os.path.exists(results_cache):
        print(f"Loading cached results from {results_cache}")
        with open(results_cache, 'rb') as f:
            return pickle.load(f)

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    basis_1, basis_2, eta_grid = load_bases(folder_path, N_t)

    # ── Step 2: Frobenius normalise ───────────────────────────────────────────
    print("\n--- Step 2: Frobenius normalisation ---")
    Xn1, norms_1 = frobenius_normalize(basis_1)
    Xn2, norms_2 = frobenius_normalize(basis_2)

    # ── Step 3: Per-field Gram matrices ───────────────────────────────────────
    print("\n--- Step 3: Per-field Gram matrices ---")
    G1p, G2p, G12p = compute_perfield_gram(Xn1, Xn2)
    N_k = G1p[PERTURBATION_TYPES[0]].shape[0]
    print(f"  N_k = {N_k},  block matrix will be ({2*N_k} x {2*N_k})")

    # ── Step 4: Block matrices M and N ───────────────────────────────────────
    print("\n--- Step 4: Building block matrices M and N ---")
    M, Nm = build_block_matrices(G1p, G2p, G12p, weights)

    # ── Step 5: Generalized eigenvalue problem ────────────────────────────────
    print("\n--- Step 5: Generalized eigenvalue problem M c = lambda N c ---")
    A, B, lam_valid, lam_all = solve_matching(M, Nm, lambda_threshold)
    print(f"  A shape: {A.shape},  B shape: {B.shape}")

    # ── Per-field residuals (diagnostic) ──────────────────────────────────────
    print("\n--- Diagnostic: per-field residual norms ---")
    residuals = compute_per_field_residuals(Xn1, Xn2, A, B)
    for i in range(residuals.shape[0]):
        res_str = "  ".join(
            f"{p}={residuals[i, j]:.4f}"
            for j, p in enumerate(PERTURBATION_TYPES)
        )
        print(f"  Mode {i+1} (λ={lam_valid[i]:.2e}):  {res_str}")

    # ── Step 6: Sparse rotation ───────────────────────────────────────────────
    print("\n--- Step 6: Sparse rotation ---")
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
    plot_eigenvalue_spectrum(lam_all, lambda_threshold, output_dir)
    plot_per_field_residuals(residuals, lam_valid, output_dir)
    n_show = min(20, len(lam_valid))
    plot_coefficients(A, B, A_sparse, B_sparse, lam_valid,
                      N_plot=n_show, output_dir=output_dir)
    plot_heatmap(A_sparse, B_sparse, output_dir)
    plot_reconstructed_timeseries(basis_1, basis_2,
                                  A_sparse, B_sparse, lam_valid,
                                  eta_grid, N_plot=5, output_dir=output_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    results = dict(
        eta_grid         = eta_grid,
        M=M, Nm=Nm,
        A=A, B=B,
        A_sparse         = A_sparse,
        B_sparse         = B_sparse,
        lam_valid        = lam_valid,
        lam_all          = lam_all,
        residuals        = residuals,
        basis_1          = basis_1,
        basis_2          = basis_2,
        Xn1              = Xn1,
        Xn2              = Xn2,
        norms_1          = norms_1,
        norms_2          = norms_2,
        weights          = weights,
        rotation_method  = rotation_method,
        lambda_threshold = lambda_threshold,
    )
    with open(results_cache, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_cache}")
    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    results = cca_sparse_match_analysis(
        N_t              = 1000,
        weights          = None,      # unit weights; try e.g. {'dr':1,'dm':2,'vr':1,'vm':1}
        lambda_threshold = 1e-3,      # relative eigenvalue threshold
        rotation_method  = 'promax',  # 'varimax' | 'promax' | 'ica'
        force_recompute  = True,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Valid modes found : {len(results['lam_valid'])}")
    print(f"  lambda_valid      : {np.round(results['lam_valid'], 6)}")
    print(f"  lambda_max        : {results['lam_all'][-1]:.4e}")
    print(f"  N_k               : {results['M'].shape[0] // 2}")
    print(f"  lambda_threshold  : {results['lambda_threshold']}")
    print("=" * 60)
