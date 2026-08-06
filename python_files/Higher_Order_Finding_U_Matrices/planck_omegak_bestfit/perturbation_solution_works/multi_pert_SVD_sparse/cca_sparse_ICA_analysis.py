# -*- coding: utf-8 -*-
"""
CCA-Based Sparse Joint Eigenfunction Method  —  ICA Rotation Variant
======================================================================

Identical to cca_sparse_analysis.py up to Step 4 (CCA solve).  The sparse
rotation step (Step 5) is replaced by Independent Component Analysis (FastICA).

Two ICA strategies are implemented:

    1. Global ICA  — apply FastICA to all m canonical vectors A at once.
       Same as setting rotation_method='ica' in cca_sparse_analysis.py, but
       with multiple random restarts to avoid local optima.

    2. Block-wise ICA  (recommended, see mixing_coef_problem.md §Alt C)
       Identify groups of canonical vectors whose singular values ρ_i are
       nearly equal (degenerate subspaces).  Apply FastICA independently
       within each block.  For size-1 blocks (non-degenerate modes) no
       rotation is applied.

       Rationale: the mixing at high k arises because SVD returns an
       arbitrary basis for a degenerate subspace.  ICA can recover the sparse
       sources within each degenerate block by maximising non-Gaussianity
       (kurtosis), which is equivalent to maximising sparsity for peaked
       distributions.  Applying ICA block-by-block prevents the rotation from
       coupling well-separated low-k modes with degenerate high-k modes.

Diagnostics produced:
    - rho spectrum with block boundaries marked
    - mixing matrix heatmap (ICA unmixing weights)
    - Gini sparsity per mode before / after ICA
    - standard heatmap, coefficients, and time-series plots

Outputs go to ./figures_ica/.
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append('../multi_pert_sparse_ortho')

from cca_sparse_analysis import (
    load_bases,
    frobenius_normalize,
    compute_gram_matrices,
    matrix_sqrt_inv,
    solve_cca,
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
OUTPUT_DIR    = './figures_ica/'
RESULTS_CACHE = 'cca_ica_results.pickle'
# ---------------------------------------------------------------------------


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ============================================================================
# ICA utilities
# ============================================================================

def _run_ica_once(X, seed, max_iter=5000, tol=1e-6):
    """
    Run FastICA on (N_samples, N_components) matrix X with a given random seed.

    Returns
    -------
    S       : (N_samples, N_components)  recovered sources
    W_unmix : (N_components, N_components)  unmixing matrix  (S = X @ W_unmix)
    converged : bool
    """
    m = X.shape[1]
    ica = FastICA(n_components=m, random_state=seed,
                  max_iter=max_iter, tol=tol, whiten='unit-variance')
    try:
        S = ica.fit_transform(X)
        converged = ica.n_iter_ < max_iter
    except Exception:
        S = X.copy()
        converged = False

    # Recover unmixing matrix: S ≈ X @ W_unmix
    W_unmix, _, _, _ = np.linalg.lstsq(X, S, rcond=None)
    return S, W_unmix, converged


def ica_rotation_global(A, B, n_init=10, max_iter=5000, tol=1e-6):
    """
    Apply FastICA to all m canonical vectors simultaneously.

    Multiple random restarts are used; the run with the highest mean Gini
    coefficient (sparsest solution) is kept.

    Parameters
    ----------
    A, B    : (N_k, m)  canonical vectors for basis 1 and 2
    n_init  : number of random restarts

    Returns
    -------
    A_sparse, B_sparse : (N_k, m)  sparsified vectors
    W_unmix            : (m, m)    unmixing matrix applied to A
    best_gini          : float     mean Gini of best solution
    """
    best_gini  = -1.0
    best_A     = A.copy()
    best_W     = np.eye(A.shape[1])

    for seed in range(n_init):
        S, W, converged = _run_ica_once(A, seed=seed,
                                         max_iter=max_iter, tol=tol)
        g = np.mean([gini(S[:, i]) for i in range(S.shape[1])])
        status = "converged" if converged else "max_iter"
        print(f"    seed={seed:2d}  Gini={g:.4f}  [{status}]")
        if g > best_gini:
            best_gini = g
            best_A    = S
            best_W    = W

    B_sparse = B @ best_W
    print(f"  Global ICA best mean Gini = {best_gini:.4f}")
    return best_A, B_sparse, best_W, best_gini


# ============================================================================
# Block identification
# ============================================================================

def identify_degenerate_blocks(rho, rho_tol=1e-3):
    """
    Group canonical vectors into blocks of nearly equal singular values.

    Two consecutive modes i and i+1 belong to the same block if
        |rho[i] - rho[i+1]| < rho_tol.

    Parameters
    ----------
    rho     : (m,) array of canonical correlations in descending order
    rho_tol : tolerance for declaring two rho values degenerate

    Returns
    -------
    blocks : list of lists of integer indices, e.g. [[0],[1,2,3],[4],[5,6]]
    """
    if len(rho) == 0:
        return []

    blocks  = []
    current = [0]
    for i in range(1, len(rho)):
        if abs(rho[i] - rho[i - 1]) < rho_tol:
            current.append(i)
        else:
            blocks.append(current)
            current = [i]
    blocks.append(current)
    return blocks


# ============================================================================
# Block-wise ICA
# ============================================================================

def ica_rotation_blockwise(A, B, rho,
                            rho_tol=1e-3,
                            n_init=10,
                            max_iter=5000,
                            tol=1e-6):
    """
    Apply FastICA independently within each degenerate block of canonical vectors.

    For blocks of size 1 the mode is returned unchanged.
    For larger blocks FastICA is run with n_init random restarts; the
    sparsest solution (highest mean Gini) is kept.

    The same block structure and unmixing matrices are applied to B.

    Parameters
    ----------
    A, B    : (N_k, m) canonical vectors
    rho     : (m,) canonical correlations (used to identify degenerate blocks)
    rho_tol : tolerance for block identification

    Returns
    -------
    A_sparse, B_sparse : (N_k, m)
    W_full             : (m, m)  block-diagonal unmixing matrix
    blocks             : list of index lists
    block_ginis        : list of mean Gini per block (after ICA)
    """
    m      = A.shape[1]
    blocks = identify_degenerate_blocks(rho, rho_tol)

    print(f"\n  Identified {len(blocks)} blocks from {m} modes "
          f"(rho_tol={rho_tol}):")
    for b in blocks:
        rho_vals = rho[b]
        print(f"    block {b}  "
              f"rho=[{rho_vals.min():.5f}, {rho_vals.max():.5f}]  "
              f"size={len(b)}")

    A_sparse    = np.zeros_like(A)
    B_sparse    = np.zeros_like(B)
    W_full      = np.zeros((m, m))
    block_ginis = []

    for b in blocks:
        idx = np.array(b)
        Ab  = A[:, idx]   # (N_k, block_size)
        Bb  = B[:, idx]

        if len(b) == 1:
            # Non-degenerate: keep as is
            A_sparse[:, idx] = Ab
            B_sparse[:, idx] = Bb
            W_full[np.ix_(idx, idx)] = np.eye(len(b))
            block_ginis.append(gini(Ab[:, 0]))
            continue

        # Degenerate block: run ICA with multiple restarts
        print(f"\n  Block {b} (size {len(b)}) — running ICA "
              f"({n_init} restarts) ...")
        best_gini_b = -1.0
        best_Ab     = Ab.copy()
        best_W_b    = np.eye(len(b))

        for seed in range(n_init):
            S, W_b, converged = _run_ica_once(Ab, seed=seed,
                                               max_iter=max_iter, tol=tol)
            g = np.mean([gini(S[:, i]) for i in range(S.shape[1])])
            status = "ok" if converged else "max"
            print(f"    seed={seed:2d}  Gini={g:.4f}  [{status}]")
            if g > best_gini_b:
                best_gini_b = g
                best_Ab     = S
                best_W_b    = W_b

        A_sparse[:, idx]             = best_Ab
        B_sparse[:, idx]             = Bb @ best_W_b
        W_full[np.ix_(idx, idx)]     = best_W_b
        block_ginis.append(best_gini_b)
        print(f"  Block {b} best mean Gini = {best_gini_b:.4f}")

    return A_sparse, B_sparse, W_full, blocks, block_ginis


# ============================================================================
# Diagnostics
# ============================================================================

def plot_rho_blocks(rho, blocks, output_dir=OUTPUT_DIR):
    """Bar plot of canonical correlations with block boundaries marked."""
    _ensure_dir(output_dir)
    m   = len(rho)
    fig, ax = plt.subplots(figsize=(max(8, m * 0.3), 4))
    ax.bar(np.arange(1, m + 1), rho, color='steelblue', alpha=0.8)

    # Mark block boundaries
    pos = 0
    for b in blocks:
        pos += len(b)
        if pos < m:
            ax.axvline(pos + 0.5, color='red', lw=1.2, ls='--', alpha=0.7)

    # Shade degenerate blocks (size > 1)
    pos = 0
    for b in blocks:
        if len(b) > 1:
            ax.axvspan(pos + 0.5, pos + len(b) + 0.5,
                       color='orange', alpha=0.15,
                       label='degenerate block' if pos == 0 else '')
        pos += len(b)

    ax.set_xlabel('Mode index')
    ax.set_ylabel('Canonical correlation ρ')
    ax.set_ylim(0, 1.05)
    ax.set_title('CCA canonical correlations with degenerate blocks (orange)')
    handles = [plt.Line2D([0], [0], color='red', ls='--', label='block boundary'),
               plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.3,
                              label='degenerate block (size > 1)')]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = output_dir + 'ica_rho_blocks.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_mixing_matrix(W_full, blocks, output_dir=OUTPUT_DIR):
    """Heatmap of the ICA unmixing matrix W with block structure."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(W_full), aspect='auto', cmap='viridis',
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='|W_ij|')

    # Draw block boundaries
    pos = 0
    for b in blocks:
        pos += len(b)
        if pos < W_full.shape[0]:
            ax.axhline(pos - 0.5, color='red', lw=1, ls='--')
            ax.axvline(pos - 0.5, color='red', lw=1, ls='--')

    ax.set_xlabel('Input canonical vector index')
    ax.set_ylabel('Output ICA component index')
    ax.set_title('ICA unmixing matrix |W|  (red = block boundaries)')
    fig.tight_layout()
    out = output_dir + 'ica_mixing_matrix.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_gini_comparison(A, A_global, A_block, output_dir=OUTPUT_DIR):
    """
    Per-mode Gini coefficient: original CCA vs global ICA vs block-wise ICA.
    """
    _ensure_dir(output_dir)
    m = A.shape[1]
    g_orig   = [gini(A[:, i])        for i in range(m)]
    g_global = [gini(A_global[:, i]) for i in range(m)]
    g_block  = [gini(A_block[:, i])  for i in range(m)]

    x = np.arange(1, m + 1)
    fig, ax = plt.subplots(figsize=(max(8, m * 0.35), 4))
    ax.plot(x, g_orig,   'ko-', ms=4, lw=1.2, label='CCA (no rotation)')
    ax.plot(x, g_global, 'bs-', ms=4, lw=1.2, label='Global ICA')
    ax.plot(x, g_block,  'r^-', ms=4, lw=1.2, label='Block-wise ICA')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Gini coefficient (higher = sparser)')
    ax.set_ylim(0, 1.05)
    ax.set_title('Sparsity per mode: original CCA vs ICA variants')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir + 'ica_gini_comparison.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_heatmap(A_sparse, B_sparse, title_suffix='', output_dir=OUTPUT_DIR,
                 suffix=''):
    """Heatmap of column-normalised |coefficients| sorted by dominant k in basis 1."""
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

    fig.suptitle(f'ICA CCA heatmap{title_suffix}', fontsize=12)
    fig.tight_layout()
    out = output_dir + f'ica_heatmap{suffix}.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Main analysis function
# ============================================================================

def cca_sparse_ICA_analysis(
        N_t              = 1000,
        weights          = None,
        ev_threshold     = 1e-10,
        rho_min          = 0.99,
        rho_tol          = 1e-3,     # tolerance for degenerate-block detection
        n_init           = 10,       # ICA random restarts per block
        max_iter         = 5000,
        ica_tol          = 1e-6,
        folder_path      = FOLDER_PATH,
        output_dir       = OUTPUT_DIR,
        results_cache    = RESULTS_CACHE,
        force_recompute  = False,
):
    """
    Full CCA sparse analysis with ICA rotation (global and block-wise).

    Parameters
    ----------
    rho_tol   : canonical correlations within rho_tol of each other are
                treated as a degenerate block; ICA is applied within each block
    n_init    : number of FastICA random restarts per block (keeps sparsest)
    max_iter  : max FastICA iterations per restart
    ica_tol   : FastICA convergence tolerance

    Returns
    -------
    dict with keys: eta_grid, G1, G2, G12, A, B,
                    A_global, B_global  (global ICA result),
                    A_block,  B_block   (block-wise ICA result),
                    rho, rho_all, blocks, ...
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

    # ── Step 3: Gram matrices ─────────────────────────────────────────────────
    print("\n--- Step 3: Gram matrices ---")
    G1, G2, G12 = compute_gram_matrices(Xn1, Xn2, weights)
    print(f"  Gram matrix shape: {G1.shape}")

    # ── Step 4: CCA ───────────────────────────────────────────────────────────
    print("\n--- Step 4: CCA (whitened SVD) ---")
    A, B, rho, rho_all = solve_cca(G1, G2, G12, ev_threshold, rho_min)
    print(f"  A shape: {A.shape},  B shape: {B.shape}")

    if A.shape[1] == 0:
        print("  No valid modes found. Exiting.")
        return {}

    # ── Step 5a: Global ICA ───────────────────────────────────────────────────
    print("\n--- Step 5a: Global ICA ---")
    A_global, B_global, W_global, gini_global = ica_rotation_global(
        A, B, n_init=n_init, max_iter=max_iter, tol=ica_tol)

    # ── Step 5b: Block-wise ICA ───────────────────────────────────────────────
    print("\n--- Step 5b: Block-wise ICA ---")
    A_block, B_block, W_block, blocks, block_ginis = ica_rotation_blockwise(
        A, B, rho,
        rho_tol=rho_tol, n_init=n_init,
        max_iter=max_iter, tol=ica_tol)

    # ── Sparsity summary ──────────────────────────────────────────────────────
    m = A.shape[1]
    g_orig   = np.mean([gini(A[:, i])        for i in range(m)])
    g_global = np.mean([gini(A_global[:, i]) for i in range(m)])
    g_block  = np.mean([gini(A_block[:, i])  for i in range(m)])
    print(f"\nMean Gini (Basis 1):")
    print(f"  No rotation  : {g_orig:.4f}")
    print(f"  Global ICA   : {g_global:.4f}")
    print(f"  Block-wise ICA: {g_block:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Plotting ---")
    plot_gram_spectrum(G1, G2, output_dir)
    plot_canonical_correlations(rho_all, rho_min, output_dir)
    plot_rho_blocks(rho, blocks, output_dir)
    plot_mixing_matrix(W_block, blocks, output_dir)
    plot_gini_comparison(A, A_global, A_block, output_dir)

    n_show = min(20, m)
    # Coefficients for block-wise ICA (primary result)
    plot_coefficients(A, B, A_block, B_block, rho,
                      N_plot=n_show, output_dir=output_dir)

    plot_heatmap(A_global, B_global,
                 title_suffix=' — Global ICA',
                 output_dir=output_dir, suffix='_global')
    plot_heatmap(A_block, B_block,
                 title_suffix=' — Block-wise ICA',
                 output_dir=output_dir, suffix='_block')

    plot_reconstructed_timeseries(
        basis_1, basis_2,
        A_block, B_block, rho,
        eta_grid, norms_1, norms_2,
        N_plot=5, output_dir=output_dir,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    results = dict(
        eta_grid  = eta_grid,
        G1=G1, G2=G2, G12=G12,
        A=A, B=B,
        A_global  = A_global,
        B_global  = B_global,
        W_global  = W_global,
        A_block   = A_block,
        B_block   = B_block,
        W_block   = W_block,
        blocks    = blocks,
        block_ginis = block_ginis,
        rho       = rho,
        rho_all   = rho_all,
        basis_1   = basis_1,
        basis_2   = basis_2,
        norms_1   = norms_1,
        norms_2   = norms_2,
        weights   = weights,
        rho_tol   = rho_tol,
        n_init    = n_init,
        ev_threshold = ev_threshold,
        rho_min   = rho_min,
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

    results = cca_sparse_ICA_analysis(
        N_t             = 1000,
        weights         = weights,
        ev_threshold    = 1e-10,
        rho_min         = 0.99,
        rho_tol         = 1e-3,   # modes within 1e-3 in rho treated as degenerate
        n_init          = 10,     # restarts per block
        max_iter        = 5000,
        ica_tol         = 1e-6,
        force_recompute = True,
    )

    if results:
        print("\n" + "=" * 60)
        print("DONE")
        print(f"  Valid modes          : {len(results['rho'])}")
        print(f"  rho range            : "
              f"{results['rho'].min():.6f} – {results['rho'].max():.6f}")
        print(f"  Degenerate blocks    : {len(results['blocks'])}")
        print(f"  Block sizes          : "
              f"{[len(b) for b in results['blocks']]}")
        print(f"  Mean Gini (block ICA): "
              f"{np.mean([gini(results['A_block'][:, i]) for i in range(results['A_block'].shape[1])]):.4f}")
        print("=" * 60)
