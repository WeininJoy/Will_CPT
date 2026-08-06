# -*- coding: utf-8 -*-
"""
Diagnostic: Does stacked Varimax show missing high-k modes for cosine wave eq?

Mirrors the pipeline in stack_varimax_sparse_extendK.py but for the simple
cosine basis from solve_pde_LA_real_2D.py:
  Basis 1: cos(sqrt(k^2 + kappa^2) * t) * exp(ikx),  k = 1..N
  Basis 2: cos(n*pi/T * t) * exp(i*sqrt((n*pi/T)^2 - kappa^2)*x), n = start_n..

Pipeline: QR orthonormalize -> M matrix -> eigenvalue filter -> Stacked Varimax
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Parameters (match existing data: N30_Nt1000_T3.14_m1.50) ─────────────────
N      = 30
N_t    = 500   # reduced grid for speed (existing data uses 1000)
N_x    = 100   # reduced grid for speed (existing data uses 500)
kappa  = 1.5
T      = np.pi
L      = 2. * np.pi
EIGENVALUE_THRESHOLD = 0.99

DATA_DIR = "./data/linear_combination/"
FIG_DIR  = "./figures/"
os.makedirs(FIG_DIR, exist_ok=True)

print(f"Parameters: N={N}, N_t={N_t}, N_x={N_x}, kappa={kappa}, T=pi")

# ── 1. Build bases ────────────────────────────────────────────────────────────
t_arr = np.linspace(0, T, N_t)
x_arr = np.linspace(0, L, N_x)
t, x = np.meshgrid(t_arr, x_arr)   # shape (N_x, N_t)

basis_1_raw = [np.cos(np.sqrt(k**2 + kappa**2) * t) * np.exp(1j * k * x)
               for k in range(1, N + 1)]

if (np.pi / T)**2 - kappa**2 < 0:
    start_n = round(np.sqrt(abs((np.pi / T)**2 - kappa**2))) + 1
else:
    start_n = 1
basis_2_raw = [np.cos(n * np.pi / T * t)
               * np.exp(1j * np.sqrt((n * np.pi / T)**2 - kappa**2) * x)
               for n in range(start_n, N + start_n)]

k_labels    = list(range(1, N + 1))          # original k indices for basis 1
n_labels    = list(range(start_n, N+start_n))  # original n indices for basis 2
print(f"Basis 1: k = {k_labels[0]}..{k_labels[-1]}")
print(f"Basis 2: n = {n_labels[0]}..{n_labels[-1]}")

# ── 2. Gram-Schmidt QR orthonormalization ─────────────────────────────────────
def ip(f, g):
    """Discrete inner product over the full (x, t) grid."""
    return np.sum(np.conj(f) * g)

def gram_schmidt(functions):
    """Modified Gram-Schmidt; returns list of orthonormal functions."""
    ortho = []
    norms_list = []
    for f in functions:
        q = f.copy().astype(complex)
        for v in ortho:
            q -= ip(v, q) * v          # modified GS: re-project onto current q
        n = np.sqrt(ip(q, q).real)
        norms_list.append(n)
        if n < 1e-12:
            print(f"  WARNING: near-zero norm {n:.2e} — linear dependence detected!")
        ortho.append(q / n)
    return ortho, norms_list

print("\nGram-Schmidt orthonormalizing basis 1...")
ortho_1, norms_1 = gram_schmidt(basis_1_raw)
print(f"  Min norm during GS: {min(norms_1):.3e}")

print("Gram-Schmidt orthonormalizing basis 2...")
ortho_2, norms_2 = gram_schmidt(basis_2_raw)
print(f"  Min norm during GS: {min(norms_2):.3e}")

# ── 3. M matrix: M[i,j] = <ortho_1[i], ortho_2[j]> ──────────────────────────
print("\nComputing M matrix...")
M_mat = np.array([[ip(ortho_1[i], ortho_2[j]) for j in range(N)]
                  for i in range(N)])

# ── 4. Eigenvalues of MM^H ───────────────────────────────────────────────────
MMH = M_mat @ M_mat.conj().T
eigenvalues, eigenvectors = np.linalg.eigh(MMH)   # ascending order
eigenvalues  = eigenvalues[::-1].real              # descending
eigenvectors = eigenvectors[:, ::-1]               # matching columns

print(f"\nAll {N} eigenvalues of MM^H (descending):")
for i, ev in enumerate(eigenvalues):
    flag = " ✓" if ev > EIGENVALUE_THRESHOLD else f" ✗ ({ev:.4f} < {EIGENVALUE_THRESHOLD})"
    print(f"  [{i+1:2d}] {ev:.6f}{flag}")

# Compare with saved all_eigenvalues file
all_ev_file = DATA_DIR + f"all_eigenvalues1_2d_N{N}_Nt500_T3.14_m1.50.txt"
if os.path.exists(all_ev_file):
    ev_saved = np.loadtxt(all_ev_file, dtype=complex).real
    ev_saved_sorted = sorted(ev_saved, reverse=True)
    print(f"\nSaved all_eigenvalues (N=30, Nt=500): {[f'{e:.3f}' for e in ev_saved_sorted]}")
    print(f"Max diff from saved: {max(abs(eigenvalues[i] - ev_saved_sorted[i]) for i in range(N)):.4f}")

# ── 5. Filter valid modes ────────────────────────────────────────────────────
valid_mask    = eigenvalues > EIGENVALUE_THRESHOLD
valid_indices = np.where(valid_mask)[0]
K = len(valid_indices)
print(f"\nValid modes (eigenvalue > {EIGENVALUE_THRESHOLD}): {K}/{N}")

# ── 6. For each eigenvector, find dominant k in original basis ────────────────
# Project eigenvectors back to original k-space via the Gram-Schmidt
# transformation: ortho_i = sum_j T[i,j] * basis_raw[j]
# We need T such that C_orig = eigenvectors.T @ T (coefficients in original k basis)
# Build T by computing <basis_raw[j], ortho_i> for each i, j
print("\nComputing transformation to original k-basis...")
T_orig_1 = np.array([[ip(ortho_1[i], basis_1_raw[j]).real for j in range(N)]
                      for i in range(N)])
# coefficients of each eigenvector in original k-space:
# eigenvector[:,v] is in QR basis -> coefficients in k-space = T_orig_1.T @ eigenvector[:,v]
coeffs_in_k = (T_orig_1.T @ eigenvectors).T    # shape (N, N): row = eigenvec, col = k-mode
dominant_k_all = np.argmax(np.abs(coeffs_in_k), axis=1)  # dominant original k-index (0-based)

print(f"Dominant original k-mode for each eigenvalue (1-indexed):")
for i in range(N):
    ev_i = eigenvalues[i]
    dom_k = dominant_k_all[i] + 1
    flag = " ✓" if ev_i > EIGENVALUE_THRESHOLD else " ✗"
    print(f"  [{i+1:2d}] eigenvalue={ev_i:.4f}{flag}  dominant k={dom_k}")

# ── 7. Stacked Varimax ────────────────────────────────────────────────────────
def varimax_rotation(Phi, gamma=1.0, q=500, tol=1e-6):
    p, k = Phi.shape
    rs = np.random.RandomState(42)
    Q, _ = np.linalg.qr(rs.randn(k, k) + 1j * rs.randn(k, k))
    R, d = Q, 0.0
    for i in range(q):
        d_old = d
        B = Phi @ R
        Msq = np.abs(B)**2
        u, s, vh = np.linalg.svd(
            Phi.conj().T @ (B * Msq - (gamma / p) * B @ np.diag(Msq.sum(axis=0)))
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            print(f"  Varimax converged in {i+1} iterations")
            return Phi @ R, R
    print(f"  Varimax reached max iterations ({q})")
    return Phi @ R, R

if K == 0:
    print("No valid modes! Try lowering the eigenvalue threshold.")
else:
    C = eigenvectors[:, valid_indices]              # N x K  (QR basis)
    C_tilde_raw = M_mat.conj().T @ C               # N x K
    norms_tilde = np.linalg.norm(C_tilde_raw, axis=0)
    C_tilde = C_tilde_raw / norms_tilde

    S = np.vstack([C, C_tilde])                    # 2N x K
    print(f"\nApplying Stacked Varimax (stack shape {S.shape})...")
    S_rot, R_var = varimax_rotation(S)

    sparse_1 = S_rot[:N, :]                        # N x K  (QR basis)
    sparse_2 = S_rot[N:, :] * norms_tilde          # N x K  (QR basis)

    # Map sparse_1 back to original k-space
    coeffs_sparse_k = (T_orig_1.T @ sparse_1).T    # K x N
    dominant_k_sparse = np.argmax(np.abs(coeffs_sparse_k), axis=1) + 1  # 1-indexed

    print(f"\nAfter Varimax — dominant original k-modes (1-indexed): {sorted(dominant_k_sparse)}")
    all_k = set(range(1, N+1))
    found_k = set(dominant_k_sparse.tolist())
    missing_k = all_k - found_k
    print(f"Missing k modes (not dominant in any Varimax mode): {sorted(missing_k)}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) Eigenvalue spectrum colored by dominant k
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, N))
sc = ax.scatter(range(1, N+1), eigenvalues,
                c=dominant_k_all + 1, cmap='viridis', s=60, zorder=3)
ax.axhline(EIGENVALUE_THRESHOLD, color='r', linestyle='--',
           label=f'threshold={EIGENVALUE_THRESHOLD}')
ax.set_xlabel('Eigenvalue rank', fontsize=11)
ax.set_ylabel('Eigenvalue of MM^H', fontsize=11)
ax.set_title(f'Eigenvalue spectrum\n(N={N}, κ={kappa}, T=π)', fontsize=11)
plt.colorbar(sc, ax=ax, label='Dominant original k')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Dominant k vs eigenvalue rank
ax = axes[1]
ax.scatter(range(1, N+1), dominant_k_all + 1, s=60, color='steelblue')
ax.axvline(K + 0.5, color='r', linestyle='--', label=f'threshold cut (K={K})')
ax.set_xlabel('Eigenvalue rank', fontsize=11)
ax.set_ylabel('Dominant original k-mode', fontsize=11)
ax.set_title('Which k-mode dominates\neach eigenvalue?', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Sparse coefficient heatmap in original k-space (if valid modes exist)
ax = axes[2]
if K > 0:
    im = ax.imshow(np.abs(coeffs_sparse_k), aspect='auto', cmap='Blues',
                   extent=[0.5, N+0.5, K+0.5, 0.5])
    ax.set_xlabel('Original k-mode index', fontsize=11)
    ax.set_ylabel('Varimax mode', fontsize=11)
    ax.set_title(f'Sparse coefficients in k-space\n(K={K} valid modes, threshold={EIGENVALUE_THRESHOLD})', fontsize=11)
    plt.colorbar(im, ax=ax)
else:
    ax.text(0.5, 0.5, 'No valid modes', ha='center', va='center', transform=ax.transAxes)

fig.tight_layout()
out_file = FIG_DIR + f"cosine_varimax_test_N{N}_threshold{EIGENVALUE_THRESHOLD}.pdf"
plt.savefig(out_file, dpi=200, bbox_inches='tight')
print(f"\nSaved figure: {out_file}")

# ── 9. Re-run with lower threshold for comparison ─────────────────────────────
for thresh in [0.95, 0.90]:
    vmask = eigenvalues > thresh
    vidx  = np.where(vmask)[0]
    k_val = len(vidx)
    if k_val == 0:
        print(f"Threshold {thresh}: 0 valid modes")
        continue
    C2 = eigenvectors[:, vidx]
    C2t_raw = M_mat.conj().T @ C2
    nt2 = np.linalg.norm(C2t_raw, axis=0)
    S2 = np.vstack([C2, C2t_raw / nt2])
    print(f"\nThreshold {thresh}: {k_val} valid modes -> applying Varimax...")
    S2_rot, _ = varimax_rotation(S2)
    sp2_k = (T_orig_1.T @ S2_rot[:N, :]).T
    dom2 = np.argmax(np.abs(sp2_k), axis=1) + 1
    found2 = set(dom2.tolist())
    missing2 = set(range(1, N+1)) - found2
    print(f"  Found k-modes: {sorted(found2)}")
    print(f"  Missing k-modes: {sorted(missing2)}")

# ── 10. Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"N modes per basis : {N}")
print(f"Threshold 0.99    : {K} valid modes, "
      f"missing k = {sorted(set(range(1,N+1)) - set(dominant_k_sparse.tolist())) if K>0 else 'N/A'}")
print(f"Dominant k of low-eigenvalue modes (below threshold): "
      f"{sorted((dominant_k_all[~valid_mask] + 1).tolist())}")
print(f"\nConclusion: Missing modes correspond to k = "
      f"{sorted((dominant_k_all[~valid_mask] + 1).tolist())} "
      f"({'low-k' if K>0 and max(dominant_k_all[~valid_mask]+1) < N//2 else 'high-k or mixed'})")

# ══════════════════════════════════════════════════════════════════════════════
# ── 11. HIGH-K EXTENSION ──────────────────────────────────────────────────────
# Generate additional high-K modes using the basis_1 formula:
#   cos(sqrt(k^2 + kappa^2) * t) * exp(ikx),  k = N+1 .. N+N_highK
# and append them identically to BOTH bases before re-running the full
# QR -> M -> eigenvalue -> Stacked Varimax pipeline.
# This mirrors multi_perturbation_analysis_with_full_extension(), where
# basis_highk_dict is concatenated to both basis_1_dict and basis_2_dict
# (same data, totally aligned) before eigenvalue analysis.
# ══════════════════════════════════════════════════════════════════════════════
N_highK       = 20                           # extra modes to add
N_total       = N + N_highK
k_highK_range = range(N + 1, N_total + 1)

print(f"\n{'='*60}")
print(f"HIGH-K EXTENSION  (adding {N_highK} modes, k = {N+1}..{N+N_highK})")
print(f"{'='*60}")

# High-K modes use the same basis_1 definition
highK_modes = [np.cos(np.sqrt(k**2 + kappa**2) * t) * np.exp(1j * k * x)
               for k in k_highK_range]

# Append identically to both bases — "totally aligned"
basis_1_ext_raw = basis_1_raw + highK_modes
basis_2_ext_raw = basis_2_raw + highK_modes

k_labels_ext = k_labels + list(k_highK_range)
print(f"Extended basis 1: k = 1..{N_total}")
print(f"Extended basis 2: basis_2 core (n={n_labels[0]}..{n_labels[-1]}) "
      f"+ high-K modes (k={N+1}..{N+N_highK}) [identical to basis_1 high-K]")

# ── 12. Gram-Schmidt on extended bases ───────────────────────────────────────
print("\nGram-Schmidt orthonormalizing extended basis 1...")
ortho_1_ext, norms_1_ext = gram_schmidt(basis_1_ext_raw)
print(f"  Min norm: {min(norms_1_ext):.3e}")

print("Gram-Schmidt orthonormalizing extended basis 2...")
ortho_2_ext, norms_2_ext = gram_schmidt(basis_2_ext_raw)
print(f"  Min norm: {min(norms_2_ext):.3e}")

# ── 13. Extended M matrix and eigenvalues ────────────────────────────────────
print(f"\nComputing extended M matrix ({N_total}×{N_total})...")
M_ext = np.array([[ip(ortho_1_ext[i], ortho_2_ext[j]) for j in range(N_total)]
                  for i in range(N_total)])

MMH_ext = M_ext @ M_ext.conj().T
eigenvalues_ext, eigenvectors_ext = np.linalg.eigh(MMH_ext)
eigenvalues_ext  = eigenvalues_ext[::-1].real    # descending
eigenvectors_ext = eigenvectors_ext[:, ::-1]

print(f"\nAll {N_total} eigenvalues (extended, descending):")
for i, ev in enumerate(eigenvalues_ext):
    flag = " ✓" if ev > EIGENVALUE_THRESHOLD else f" ✗ ({ev:.4f})"
    print(f"  [{i+1:2d}] {ev:.6f}{flag}")

valid_mask_ext    = eigenvalues_ext > EIGENVALUE_THRESHOLD
valid_indices_ext = np.where(valid_mask_ext)[0]
K_ext = len(valid_indices_ext)
print(f"\nValid modes (eigenvalue > {EIGENVALUE_THRESHOLD}): {K_ext}/{N_total}")

# Transformation to original k-basis (extended)
print("\nComputing transformation to original k-basis (extended)...")
T_orig_1_ext = np.array([[ip(ortho_1_ext[i], basis_1_ext_raw[j]).real
                           for j in range(N_total)]
                          for i in range(N_total)])
coeffs_in_k_ext    = (T_orig_1_ext.T @ eigenvectors_ext).T   # N_total × N_total
dominant_k_ext_all = np.argmax(np.abs(coeffs_in_k_ext), axis=1)

print(f"\nDominant k-mode per eigenvalue (1-indexed, extended):")
for i in range(N_total):
    ev_i  = eigenvalues_ext[i]
    dom_k = dominant_k_ext_all[i] + 1
    flag  = " ✓" if ev_i > EIGENVALUE_THRESHOLD else " ✗"
    print(f"  [{i+1:2d}] eigenvalue={ev_i:.4f}{flag}  dominant k={dom_k}")

# ── 14. Stacked Varimax on extended system ────────────────────────────────────
if K_ext == 0:
    print("\nNo valid modes in extended system — cannot apply Varimax!")
    dominant_k_sparse_ext = np.array([], dtype=int)
    coeffs_sparse_k_ext   = None
else:
    C_ext           = eigenvectors_ext[:, valid_indices_ext]    # N_total × K_ext
    C_tilde_ext_raw = M_ext.conj().T @ C_ext                    # N_total × K_ext
    norms_tilde_ext = np.linalg.norm(C_tilde_ext_raw, axis=0)
    C_tilde_ext     = C_tilde_ext_raw / norms_tilde_ext

    S_ext = np.vstack([C_ext, C_tilde_ext])                     # 2*N_total × K_ext
    print(f"\nApplying Stacked Varimax on extended system "
          f"(stack shape {S_ext.shape})...")
    S_ext_rot, R_var_ext = varimax_rotation(S_ext)

    sparse_1_ext = S_ext_rot[:N_total, :]                       # N_total × K_ext
    sparse_2_ext = S_ext_rot[N_total:, :] * norms_tilde_ext     # N_total × K_ext

    # Map sparse_1_ext back to original k-space
    coeffs_sparse_k_ext   = (T_orig_1_ext.T @ sparse_1_ext).T  # K_ext × N_total
    dominant_k_sparse_ext = (np.argmax(np.abs(coeffs_sparse_k_ext), axis=1) + 1)

    print(f"\nAfter Varimax (extended) — dominant k-modes: "
          f"{sorted(dominant_k_sparse_ext.tolist())}")

# ── 15. Missing-mode check ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"MISSING-MODE CHECK")
print(f"{'='*60}")

all_k_ext     = set(range(1, N_total + 1))
found_k_ext   = set(dominant_k_sparse_ext.tolist()) if K_ext > 0 else set()
missing_k_ext = all_k_ext - found_k_ext

print(f"Extended basis: N_total = {N_total}  "
      f"({N} core + {N_highK} high-K)")
print(f"Varimax valid modes: K_ext = {K_ext}")
print(f"Dominant k found : {sorted(found_k_ext)}")
print(f"Missing k modes  : {sorted(missing_k_ext)}")

if K > 0:
    orig_missing  = set(range(1, N + 1)) - set(dominant_k_sparse.tolist())
    recovered     = orig_missing - missing_k_ext
    still_missing = orig_missing & missing_k_ext
    print(f"\nComparing with original N={N} analysis:")
    print(f"  Originally missing k (N={N}):  {sorted(orig_missing)}")
    print(f"  Recovered by high-K extension: {sorted(recovered)}")
    print(f"  Still missing after extension: {sorted(still_missing)}")

# ── 16. Plots for extended system ─────────────────────────────────────────────
fig_ext, axes_ext = plt.subplots(1, 3, figsize=(15, 4))

# (a) Eigenvalue spectrum — extended
ax = axes_ext[0]
sc_ext = ax.scatter(range(1, N_total + 1), eigenvalues_ext,
                    c=dominant_k_ext_all + 1, cmap='viridis', s=40, zorder=3)
ax.axhline(EIGENVALUE_THRESHOLD, color='r', linestyle='--',
           label=f'threshold={EIGENVALUE_THRESHOLD}')
ax.axvline(N + 0.5, color='orange', linestyle=':', linewidth=1.5,
           label=f'high-K boundary (k>{N})')
ax.set_xlabel('Eigenvalue rank', fontsize=11)
ax.set_ylabel('Eigenvalue of MM^H', fontsize=11)
ax.set_title(f'Extended eigenvalue spectrum\n'
             f'(N={N}+{N_highK} high-K, κ={kappa}, T=π)', fontsize=11)
plt.colorbar(sc_ext, ax=ax, label='Dominant original k')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Dominant k vs eigenvalue rank — extended
ax = axes_ext[1]
ax.scatter(range(1, N_total + 1), dominant_k_ext_all + 1, s=40, color='steelblue')
ax.axvline(K_ext + 0.5, color='r', linestyle='--',
           label=f'threshold cut (K_ext={K_ext})')
ax.axvline(N + 0.5, color='orange', linestyle=':', linewidth=1.5,
           label=f'high-K boundary')
ax.set_xlabel('Eigenvalue rank', fontsize=11)
ax.set_ylabel('Dominant original k-mode', fontsize=11)
ax.set_title('Dominant k-mode (extended)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Sparse coefficient heatmap — extended
ax = axes_ext[2]
if K_ext > 0 and coeffs_sparse_k_ext is not None:
    im_ext = ax.imshow(np.abs(coeffs_sparse_k_ext), aspect='auto', cmap='Blues',
                       extent=[0.5, N_total + 0.5, K_ext + 0.5, 0.5])
    ax.axvline(N + 0.5, color='orange', linestyle=':', linewidth=1.5,
               label=f'high-K boundary')
    ax.set_xlabel('Original k-mode index', fontsize=11)
    ax.set_ylabel('Varimax mode', fontsize=11)
    ax.set_title(f'Sparse coefficients (extended)\n(K_ext={K_ext})', fontsize=11)
    plt.colorbar(im_ext, ax=ax)
    ax.legend(fontsize=9)
else:
    ax.text(0.5, 0.5, 'No valid extended modes', ha='center', va='center',
            transform=ax.transAxes)

fig_ext.tight_layout()
out_ext = FIG_DIR + (f"cosine_varimax_extended_N{N}_highK{N_highK}"
                     f"_threshold{EIGENVALUE_THRESHOLD}.pdf")
plt.savefig(out_ext, dpi=200, bbox_inches='tight')
print(f"\nSaved extended figure: {out_ext}")

# ── 17. Extended summary ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXTENDED SUMMARY")
print("="*60)
print(f"Original N modes per basis : {N}")
print(f"High-K extension           : +{N_highK} modes (k={N+1}..{N+N_highK})")
print(f"Total N_total              : {N_total}")
print(f"Threshold                  : {EIGENVALUE_THRESHOLD}")
print(f"Original valid modes K     : {K}   "
      f"missing k = {sorted(set(range(1,N+1)) - set(dominant_k_sparse.tolist())) if K>0 else 'N/A'}")
print(f"Extended valid modes K_ext : {K_ext}   "
      f"missing k = {sorted(missing_k_ext)}")

# ══════════════════════════════════════════════════════════════════════════════
# ── 18. REVERSED-ORDERING EXTENSION (high-k first) ───────────────────────────
# QR/GS processes columns left-to-right: the FIRST column is orthogonalized
# cleanly (no prior projections), and each later column is projected against
# all preceding QR vectors.
#
# Forward ordering [basis_core | highK]:
#   highK QR vectors = highK_raw - (projections onto basis_core QR vectors)
#   → different contamination in basis_1 vs basis_2 because their cores differ
#   → M[high-k block] ≠ I
#
# Reversed ordering [highK | basis_core]:
#   highK input is identical in both bases AND processed first with no priors
#   → ortho_1_rev[i] = ortho_2_rev[i]  exactly  for i < N_highK
#   → M_rev[:N_highK, :N_highK] = I  (guaranteed, not approximate)
#   → cross-terms M_rev[:N_highK, N_highK:] = 0  (by GS construction)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"REVERSED-ORDERING EXTENSION  (high-k first, low-k appended)")
print(f"{'='*60}")

# High-k modes placed FIRST in both bases
basis_1_rev_raw = highK_modes + basis_1_raw   # [k_{N+1}..k_{N+highK}, k_1..k_N]
basis_2_rev_raw = highK_modes + basis_2_raw   # [k_{N+1}..k_{N+highK}, n_1..n_N]

# Column j of basis_1_rev_raw/basis_2_rev_raw corresponds to k = k_labels_rev[j]
k_labels_rev = list(k_highK_range) + k_labels  # [N+1,..,N+N_highK, 1,..,N]

# ── 19. Gram-Schmidt on reversed bases ───────────────────────────────────────
print("\nGram-Schmidt orthonormalizing reversed basis 1 (high-k first)...")
ortho_1_rev, norms_1_rev = gram_schmidt(basis_1_rev_raw)
print(f"  Min norm: {min(norms_1_rev):.3e}")

print("Gram-Schmidt orthonormalizing reversed basis 2 (high-k first)...")
ortho_2_rev, norms_2_rev = gram_schmidt(basis_2_rev_raw)
print(f"  Min norm: {min(norms_2_rev):.3e}")

# ── 20. Full M matrix and eigenvalues (reversed) ─────────────────────────────
print(f"\nComputing reversed M matrix ({N_total}×{N_total})...")
M_rev = np.array([[ip(ortho_1_rev[i], ortho_2_rev[j]) for j in range(N_total)]
                  for i in range(N_total)])

# Verify block structure: top-left (high-k) block should be ~I, off-diagonal ~0
M_rev_hk = M_rev[:N_highK, :N_highK]
print(f"\n  Block structure check:")
print(f"  Top-left  ({N_highK}×{N_highK}, high-k): ||M - I||_F = "
      f"{np.linalg.norm(M_rev_hk - np.eye(N_highK)):.3e}")
print(f"  Top-right ({N_highK}×{N}, cross):       ||M||_F     = "
      f"{np.linalg.norm(M_rev[:N_highK, N_highK:]):.3e}")
print(f"  Bot-left  ({N}×{N_highK}, cross):       ||M||_F     = "
      f"{np.linalg.norm(M_rev[N_highK:, :N_highK]):.3e}")

# Compare: high-k block in forward ordering (bottom-right) vs reversed (top-left)
M_fwd_hk = M_ext[N:, N:]   # bottom-right N_highK×N_highK block in forward ordering
print(f"\n  Forward ordering high-k block: ||M - I||_F = "
      f"{np.linalg.norm(M_fwd_hk - np.eye(N_highK)):.3e}  (should be >> 0)")
print(f"  Reversed ordering high-k block: ||M - I||_F = "
      f"{np.linalg.norm(M_rev_hk - np.eye(N_highK)):.3e}  (should be ~0)")

MMH_rev = M_rev @ M_rev.conj().T
eigenvalues_rev, eigenvectors_rev = np.linalg.eigh(MMH_rev)
eigenvalues_rev  = eigenvalues_rev[::-1].real    # descending
eigenvectors_rev = eigenvectors_rev[:, ::-1]

print(f"\nAll {N_total} eigenvalues (reversed ordering, descending):")
for i, ev in enumerate(eigenvalues_rev):
    flag = " ✓" if ev > EIGENVALUE_THRESHOLD else f" ✗ ({ev:.4f})"
    print(f"  [{i+1:2d}] {ev:.6f}{flag}")

valid_mask_rev    = eigenvalues_rev > EIGENVALUE_THRESHOLD
valid_indices_rev = np.where(valid_mask_rev)[0]
K_rev = len(valid_indices_rev)
print(f"\nValid modes (eigenvalue > {EIGENVALUE_THRESHOLD}): {K_rev}/{N_total}")

# Transformation to original k-basis (reversed ordering)
# T_orig_1_rev[i,j] = <ortho_1_rev[i], basis_1_rev_raw[j]>
# Column j of basis_1_rev_raw has k = k_labels_rev[j]
print("\nComputing transformation to original k-basis (reversed)...")
T_orig_1_rev = np.array([[ip(ortho_1_rev[i], basis_1_rev_raw[j]).real
                           for j in range(N_total)]
                          for i in range(N_total)])
coeffs_in_k_rev    = (T_orig_1_rev.T @ eigenvectors_rev).T  # N_total × N_total
dominant_idx_rev_all = np.argmax(np.abs(coeffs_in_k_rev), axis=1)
dominant_k_rev_all   = np.array([k_labels_rev[idx] for idx in dominant_idx_rev_all])

print(f"\nDominant k-mode per eigenvalue (reversed ordering):")
for i in range(N_total):
    ev_i  = eigenvalues_rev[i]
    dom_k = dominant_k_rev_all[i]
    flag  = " ✓" if ev_i > EIGENVALUE_THRESHOLD else " ✗"
    print(f"  [{i+1:2d}] eigenvalue={ev_i:.4f}{flag}  dominant k={dom_k}")

# ── 21. Stacked Varimax (reversed) ────────────────────────────────────────────
if K_rev == 0:
    print("\nNo valid modes in reversed system — cannot apply Varimax!")
    dominant_k_sparse_rev = np.array([], dtype=int)
    coeffs_sparse_k_rev   = None
else:
    C_rev           = eigenvectors_rev[:, valid_indices_rev]    # N_total × K_rev
    C_tilde_rev_raw = M_rev.conj().T @ C_rev                    # N_total × K_rev
    norms_tilde_rev = np.linalg.norm(C_tilde_rev_raw, axis=0)
    C_tilde_rev     = C_tilde_rev_raw / norms_tilde_rev

    S_rev = np.vstack([C_rev, C_tilde_rev])                     # 2*N_total × K_rev
    print(f"\nApplying Stacked Varimax (reversed, stack shape {S_rev.shape})...")
    S_rev_rot, R_var_rev = varimax_rotation(S_rev)

    sparse_1_rev = S_rev_rot[:N_total, :]                       # N_total × K_rev
    sparse_2_rev = S_rev_rot[N_total:, :] * norms_tilde_rev     # N_total × K_rev

    # Map back: column index j → k_labels_rev[j] → original k value
    coeffs_sparse_k_rev    = (T_orig_1_rev.T @ sparse_1_rev).T  # K_rev × N_total
    dominant_idx_sparse_rev = np.argmax(np.abs(coeffs_sparse_k_rev), axis=1)
    dominant_k_sparse_rev   = np.array([k_labels_rev[idx]
                                        for idx in dominant_idx_sparse_rev])

    print(f"\nAfter Varimax (reversed) — dominant k-modes: "
          f"{sorted(dominant_k_sparse_rev.tolist())}")

# ── 22. Missing-mode check and three-way comparison ──────────────────────────
all_k_rev     = set(range(1, N_total + 1))
found_k_rev   = set(dominant_k_sparse_rev.tolist()) if K_rev > 0 else set()
missing_k_rev = all_k_rev - found_k_rev

miss_orig = sorted(set(range(1,N+1)) - set(dominant_k_sparse.tolist())) if K>0 else []
miss_ext  = sorted(missing_k_ext)
miss_rev  = sorted(missing_k_rev)

print(f"\n{'='*60}")
print(f"THREE-WAY MISSING-MODE COMPARISON")
print(f"{'='*60}")
print(f"Original N={N}           : K={K:2d}     missing = {miss_orig}")
print(f"Fwd ext  N={N_total} : K_ext={K_ext:2d}  missing = {miss_ext}")
print(f"Rev ext  N={N_total} : K_rev={K_rev:2d}  missing = {miss_rev}")
if K_rev > 0 and K > 0:
    print(f"\nModes recovered by reversed ordering (vs forward): "
          f"{sorted(set(miss_ext) - set(miss_rev))}")
    print(f"Modes still missing in reversed ordering:           "
          f"{sorted(set(miss_rev) & set(range(1, N+1)))}")

# ── 23. Comparison plots ───────────────────────────────────────────────────────
fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(15, 4))

# (a) Eigenvalue spectra — all three overlaid
ax = axes_cmp[0]
ax.plot(range(1, N+1),       eigenvalues,     'o-', ms=5,
        label=f'Original N={N}', color='steelblue')
ax.plot(range(1, N_total+1), eigenvalues_ext, 's--', ms=4,
        label=f'Fwd ext (+{N_highK})', color='green', alpha=0.8)
ax.plot(range(1, N_total+1), eigenvalues_rev, '^-', ms=4,
        label=f'Rev ext (+{N_highK})', color='red', alpha=0.8)
ax.axhline(EIGENVALUE_THRESHOLD, color='k', linestyle='--', linewidth=1,
           label=f'threshold')
ax.axvline(N + 0.5, color='orange', linestyle=':', linewidth=1.5,
           label=f'high-K boundary')
ax.set_xlabel('Eigenvalue rank', fontsize=11)
ax.set_ylabel('Eigenvalue of MM^H', fontsize=11)
ax.set_title('Eigenvalue spectra\n(all three cases)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) |M[high-k block]| — forward ordering (contaminated, should deviate from I)
ax = axes_cmp[1]
im1 = ax.imshow(np.abs(M_fwd_hk), aspect='auto', cmap='Blues', vmin=0, vmax=1)
ax.set_title(f'|M[high-k block]| — fwd ordering\n'
             f'(should be ≈I, ||·-I||_F={np.linalg.norm(M_fwd_hk - np.eye(N_highK)):.3f})',
             fontsize=10)
ax.set_xlabel(f'Col index (high-k, {N_highK} modes)', fontsize=10)
ax.set_ylabel(f'Row index (high-k, {N_highK} modes)', fontsize=10)
plt.colorbar(im1, ax=ax)

# (c) |M[high-k block]| — reversed ordering (clean, should be ~I)
ax = axes_cmp[2]
im2 = ax.imshow(np.abs(M_rev_hk), aspect='auto', cmap='Blues', vmin=0, vmax=1)
ax.set_title(f'|M[high-k block]| — rev ordering\n'
             f'(should be ≈I, ||·-I||_F={np.linalg.norm(M_rev_hk - np.eye(N_highK)):.3f})',
             fontsize=10)
ax.set_xlabel(f'Col index (high-k, {N_highK} modes)', fontsize=10)
ax.set_ylabel(f'Row index (high-k, {N_highK} modes)', fontsize=10)
plt.colorbar(im2, ax=ax)

fig_cmp.tight_layout()
out_cmp = FIG_DIR + (f"cosine_varimax_comparison_N{N}_highK{N_highK}"
                     f"_threshold{EIGENVALUE_THRESHOLD}.pdf")
plt.savefig(out_cmp, dpi=200, bbox_inches='tight')
print(f"\nSaved comparison figure: {out_cmp}")
