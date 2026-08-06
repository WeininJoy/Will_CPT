"""
compare_perturbations_soln.py

Two tests motivated by cca_rho_less_than_1_explanation.md:

(1) CONVERGENCE TEST
    Show that |k_allowedK - k_integerK| → 0 as k increases (high-k modes
    approach each other in k-space), while their FCB boundary values vr^∞
    remain different: allowedK has vr^∞ ≈ 0 by construction, integerK has
    vr^∞ ≠ 0 in general.

(2) SOLUTION COMPARISON
    Plot the last N_PLOT high-k modes' perturbation solutions (dr, dm, vr, vm)
    for both bases on the same axes.  Even though Δk/k < 1%, the small k
    deviation is enough to make the oscillation behaviour different:
    - allowedK solutions satisfy the palindrome (vr^∞ ≈ 0 at FCB)
    - integerK solutions are NOT symmetric/anti-symmetric at the end of the
      universe (vr^∞ ≠ 0 at FCB)

Reconstruction formula (from generate_multi_perturbation_bases):
    x_inf  = lstsq( (A@X1 + D@X2 + GX3)[2:6, :],  recs[2:6] )
    Y(η)   = einsum('ijt,j->it', U_ABC(η), X1@x_inf)
           + einsum('ijt,j->it', U_DEF(η), X2@x_inf)

Variable indices in Y (75 stored variables):
    0 = Φ (gravitational potential)
    1 = Φ̇  (or auxiliary field)
    2 = δ_r   ← dr
    3 = δ_m   ← dm
    4 = v_r   ← vr
    5 = v_m   ← vm
    6…74 = higher photon/neutrino multipoles

Outputs:
    ./figures/compare_k_convergence.pdf
    ./figures/compare_perturbations_soln_highk.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import os

# ─────────────────────────────────────────────────────────────────────────────
# 0. Cosmological parameters (identical to generate_data.py)
# ─────────────────────────────────────────────────────────────────────────────
nu_spacing4_bestfit = [426.32052836334015, 1.5217932160492045,
                       0.1555991124300318, 0.5567037724095327,
                       2.030476, 0.967708, 0.046112]
mt, kt, omega_b_ratio, h = nu_spacing4_bestfit[:4]
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
deltaeta = 6.6e-4   # gap between end_time and FCB

def cosmological_parameters(mt, kt, h):
    Omega_r = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
    def f(a0):
        return a0**4 - 3*kt*a0**2 + mt*a0 + (1 - 1./Omega_r)
    a0 = root_scalar(f, bracket=[1, 1e3]).root
    Omega_lambda = Omega_r * a0**4
    Omega_m  = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K  = -3 * kt * np.sqrt(Omega_lambda * Omega_r)
    return Omega_lambda, Omega_m, Omega_K

OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
H0 = 1 / np.sqrt(3 * OmegaLambda)

# Unit conversion: internal k → Mpc⁻¹
c_kms   = 299792.458
H0_phys = (h * 100) / c_kms
conv    = H0_phys * np.sqrt(3 * OmegaLambda)   # multiply internal k by this

os.makedirs('./figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load static matrices and timeseries for both bases
# ─────────────────────────────────────────────────────────────────────────────

def load_static(folder):
    return dict(
        kvalues = np.load(folder + 'L70_kvalues.npy'),
        ABC     = np.load(folder + 'L70_ABCmatrices.npy'),   # (N, 75, 6)
        DEF     = np.load(folder + 'L70_DEFmatrices.npy'),   # (N, 75, 2)
        GHI     = np.load(folder + 'L70_GHIvectors.npy'),    # (N, 75)
        X1      = np.load(folder + 'L70_X1matrices.npy'),    # (N, 6, 4)
        X2      = np.load(folder + 'L70_X2matrices.npy'),    # (N, 2, 4)
        rec     = np.load(folder + 'L70_recValues.npy'),     # (N, 75)
    )

def load_timeseries(folder):
    return dict(
        t_grid  = np.load(folder + 't_grid.npy'),            # (n_t,)
        kvalues = np.load(folder + 'L70_kvalues.npy'),       # (N,)
        ABC     = np.load(folder + 'L70_ABC_solutions.npy'), # (N, 75, 6, n_t)
        DEF     = np.load(folder + 'L70_DEF_solutions.npy'), # (N, 75, 2, n_t)
    )

base = './data/'
print("Loading static matrices …")
st_ak = load_static(base + 'data_allowedK/')
st_ik = load_static(base + 'data_integerK/')
print("Loading timeseries …")
ts_ak = load_timeseries(base + 'data_allowedK_timeseries/')
ts_ik = load_timeseries(base + 'data_integerK_timeseries/')

k_ak = st_ak['kvalues']
k_ik = st_ik['kvalues']
n_modes = min(len(k_ak), len(k_ik))   # both should be 105

print(f"  allowedK: {len(k_ak)} modes,  k = {k_ak[0]:.4f} … {k_ak[-1]:.4f}")
print(f"  integerK: {len(k_ik)} modes,  k = {k_ik[0]:.4f} … {k_ik[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute x_inf = [dr^∞, dm^∞, vr^∞, vm^∞] for every mode in both bases
#    (same lstsq procedure as plot_vrfcb_integerK.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_xinf_all(static):
    """Return x_inf for every mode; shape (N, 4)."""
    N = len(static['kvalues'])
    xinf = np.zeros((N, 4))
    for j in range(N):
        A  = static['ABC'][j, 0:6, :]   # (6, 6)
        D  = static['DEF'][j, 0:6, :]   # (6, 2)
        G  = static['GHI'][j, 0:6]      # (6,)
        X1 = static['X1'][j]            # (6, 4)
        X2 = static['X2'][j]            # (2, 4)

        GX3 = np.zeros((6, 4))
        GX3[:, 2] = G

        mat   = (A @ X1 + D @ X2 + GX3)[[2, 3, 4, 5], :]  # (4, 4): dr,dm,vr,vm
        xrecs = static['rec'][j, [2, 3, 4, 5]]

        xinf[j] = np.linalg.lstsq(mat, xrecs, rcond=None)[0]
    return xinf

print("Computing x_inf for allowedK …")
xinf_ak = compute_xinf_all(st_ak)   # (N, 4)
print("Computing x_inf for integerK …")
xinf_ik = compute_xinf_all(st_ik)

vr_ak = xinf_ak[:, 2]   # vr^∞ for allowedK  (should be ≈ 0)
vr_ik = xinf_ik[:, 2]   # vr^∞ for integerK  (generally ≠ 0)

print(f"\n  allowedK |vr^∞|: max={np.max(np.abs(vr_ak)):.2e}  "
      f"mean={np.mean(np.abs(vr_ak)):.2e}")
print(f"  integerK |vr^∞|: max={np.max(np.abs(vr_ik)):.2e}  "
      f"mean={np.mean(np.abs(vr_ik)):.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PART 1 — k-convergence test
# ─────────────────────────────────────────────────────────────────────────────
# Modes are matched by index (both bases were generated with the same ordered
# integer-n sequence, so k_ak[i] and k_ik[i] correspond to the same n).

idx = np.arange(n_modes)
dk_abs = np.abs(k_ak[:n_modes] - k_ik[:n_modes])
dk_rel = dk_abs / (0.5 * (k_ak[:n_modes] + k_ik[:n_modes]))
k_mid_phys = 0.5 * (k_ak[:n_modes] + k_ik[:n_modes]) * conv   # Mpc⁻¹

print("\n── Last 10 modes: k comparison ──")
print(f"{'idx':>4}  {'k_aK':>9}  {'k_iK':>9}  {'|Δk|':>9}  {'|Δk|/k':>9}  {'vr∞_aK':>11}  {'vr∞_iK':>11}")
for i in range(n_modes - 10, n_modes):
    print(f"{i:4d}  {k_ak[i]:9.5f}  {k_ik[i]:9.5f}  "
          f"{dk_abs[i]:9.5f}  {dk_rel[i]:9.4%}  "
          f"{vr_ak[i]:11.4e}  {vr_ik[i]:11.4e}")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# ── (a) k-values for both bases ──
ax = axes[0, 0]
ax.plot(idx, k_ak[:n_modes] * conv * 1e3, 'b.', ms=4, lw=0, label='allowedK')
ax.plot(idx, k_ik[:n_modes] * conv * 1e3, 'r.', ms=4, lw=0, label='integerK')
ax.set_xlabel('Mode index $i$')
ax.set_ylabel(r'$k_{\rm phys}\ [10^{-3}\ {\rm Mpc}^{-1}]$')
ax.set_title('(a) k-values: allowedK vs integerK')
ax.legend(fontsize=9, markerscale=2)
ax.grid(True, alpha=0.3)

# ── (b) |Δk| vs mode index (log scale) ──
ax = axes[0, 1]
ax.semilogy(idx, dk_abs * conv, 'g.-', ms=3, lw=0.8,
            label=r'$|k_{\rm aK} - k_{\rm iK}|$')
ax.set_xlabel('Mode index $i$')
ax.set_ylabel(r'$|\Delta k|\ [{\rm Mpc}^{-1}]$')
ax.set_title(r'(b) Absolute $\Delta k$ (converges at high $k$)')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# ── (c) relative |Δk|/k vs k_phys ──
ax = axes[1, 0]
ax.plot(k_mid_phys * 1e3, dk_rel * 100, '.-', color='purple', ms=3, lw=0.8)
ax.set_xlabel(r'$\bar{k}\ [10^{-3}\ {\rm Mpc}^{-1}]$')
ax.set_ylabel(r'$|\Delta k| / \bar{k}\ [\%]$')
ax.set_title(r'(c) Relative $\Delta k / k$ — converges at high $k$')
ax.grid(True, alpha=0.3)

# ── (d) |vr^∞| for both bases vs k_phys ──
ax = axes[1, 1]
ax.plot(k_ik[:n_modes] * conv * 1e3, np.abs(vr_ik[:n_modes]),
        'r.-', ms=3, lw=0.8, label=r'integerK $|v_r^\infty|$')
ax.plot(k_ak[:n_modes] * conv * 1e3, np.abs(vr_ak[:n_modes]),
        'b.-', ms=3, lw=0.8, alpha=0.7, label=r'allowedK $|v_r^\infty| \approx 0$')
ax.set_xlabel(r'$k_{\rm phys}\ [10^{-3}\ {\rm Mpc}^{-1}]$')
ax.set_ylabel(r'$|v_r^\infty|$ at FCB')
ax.set_title(r'(d) FCB value $|v_r^\infty|$: allowedK$\approx 0$, integerK $\neq 0$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle(
    'Part 1 — convergence test: allowedK and integerK k-values approach each other at high $k$\n'
    r'but carry different FCB boundary conditions ($v_r^\infty$ differs)',
    fontsize=11)
fig.tight_layout()
out1 = './figures/compare_k_convergence.pdf'
plt.savefig(out1, bbox_inches='tight')
print(f"\nSaved: {out1}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. PART 2 — reconstruct and compare perturbation solutions for last N_PLOT modes
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_Y(ts, static, idx_mode):
    """
    Reconstruct Y(η) for mode idx_mode.

    Returns
    -------
    t_grid : (n_t,)
    Y      : (75, n_t)  — vars 2,3,4,5 = dr, dm, vr, vm
    x_inf  : (4,)       — [dr^∞, dm^∞, vr^∞, vm^∞] at FCB
    k_int  : float      — internal k value
    """
    A  = static['ABC'][idx_mode, 0:6, :]
    D  = static['DEF'][idx_mode, 0:6, :]
    G  = static['GHI'][idx_mode, 0:6]
    X1 = static['X1'][idx_mode]
    X2 = static['X2'][idx_mode]

    GX3 = np.zeros((6, 4))
    GX3[:, 2] = G
    mat   = (A @ X1 + D @ X2 + GX3)[[2, 3, 4, 5], :]
    xrecs = static['rec'][idx_mode, [2, 3, 4, 5]]
    x_inf = np.linalg.lstsq(mat, xrecs, rcond=None)[0]

    x_prime = X1 @ x_inf   # (6,)
    y_prime = X2 @ x_inf   # (2,)

    ABC_sols = ts['ABC'][idx_mode]   # (75, 6, n_t)
    DEF_sols = ts['DEF'][idx_mode]   # (75, 2, n_t)

    Y = (np.einsum('ijt,j->it', ABC_sols, x_prime) +
         np.einsum('ijt,j->it', DEF_sols, y_prime))   # (75, n_t)

    return ts['t_grid'], Y, x_inf, ts['kvalues'][idx_mode]


# ── Field layout ──
# Y[2] = dr,  Y[3] = dm,  Y[4] = vr,  Y[5] = vm
FIELDS = [
    (4, r'$v_r$',       r'$v_r^\infty$'),
    (5, r'$v_m$',       r'$v_m^\infty$'),
    (2, r'$\delta_r$',  r'$\delta_r^\infty$'),
    (3, r'$\delta_m$',  r'$\delta_m^\infty$'),
]
XINF_FIELD_IDX = [2, 3, 0, 1]   # x_inf = [dr^∞, dm^∞, vr^∞, vm^∞]
                                  # so vr^∞ = x_inf[2], vm^∞ = x_inf[3],
                                  # dr^∞ = x_inf[0], dm^∞ = x_inf[1]

N_PLOT = 5   # show last N_PLOT high-k modes

# Color maps: blues for allowedK, reds for integerK
cmap_ak = plt.cm.Blues(np.linspace(0.45, 0.90, N_PLOT))
cmap_ik = plt.cm.Reds (np.linspace(0.45, 0.90, N_PLOT))

print(f"\n── Reconstructing solutions for the last {N_PLOT} modes ──")

# Pre-compute all reconstructions
recs = []   # list of dicts
for plot_rank, mode_idx in enumerate(range(n_modes - N_PLOT, n_modes)):
    t_ak_g, Y_ak, xi_ak, ki_ak = reconstruct_Y(ts_ak, st_ak, mode_idx)
    t_ik_g, Y_ik, xi_ik, ki_ik = reconstruct_Y(ts_ik, st_ik, mode_idx)
    recs.append(dict(
        mode_idx  = mode_idx,
        plot_rank = plot_rank,
        t_ak=t_ak_g, Y_ak=Y_ak, xi_ak=xi_ak, ki_ak=ki_ak,
        t_ik=t_ik_g, Y_ik=Y_ik, xi_ik=xi_ik, ki_ik=ki_ik,
    ))
    dk_rel_i = abs(ki_ak - ki_ik) / (0.5*(ki_ak + ki_ik)) * 100
    print(f"  mode {mode_idx}: k_aK={ki_ak:.4f}  k_iK={ki_ik:.4f}  "
          f"Δk/k={dk_rel_i:.3f}%  "
          f"vr∞_aK={xi_ak[2]:.3e}  vr∞_iK={xi_ik[2]:.3e}")

# ──────────────────────────────────────────────
# Figure 2a: all 4 fields in one N_PLOT×4 grid
# ──────────────────────────────────────────────
nrows = N_PLOT
ncols = len(FIELDS)
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 2.8*nrows), sharex='col')

for row, rec in enumerate(recs):
    mode_idx = rec['mode_idx']
    t_ak_g   = rec['t_ak']
    t_ik_g   = rec['t_ik']
    xi_ak    = rec['xi_ak']
    xi_ik    = rec['xi_ik']
    ki_ak    = rec['ki_ak']
    ki_ik    = rec['ki_ik']

    # Append FCB point (t_end + deltaeta, value = x_inf component)
    t_fcb_ak = t_ak_g[-1] + deltaeta
    t_fcb_ik = t_ik_g[-1] + deltaeta

    dk_rel_i = abs(ki_ak - ki_ik) / (0.5*(ki_ak + ki_ik)) * 100

    for col, (var_idx, field_label, fcb_label) in enumerate(FIELDS):
        ax = axes[row, col]
        xi_field_idx = XINF_FIELD_IDX[col]

        y_ak = rec['Y_ak'][var_idx]
        y_ik = rec['Y_ik'][var_idx]

        # Full curves including FCB endpoint
        t_full_ak = np.append(t_ak_g, t_fcb_ak)
        t_full_ik = np.append(t_ik_g, t_fcb_ik)
        y_full_ak = np.append(y_ak, xi_ak[xi_field_idx])
        y_full_ik = np.append(y_ik, xi_ik[xi_field_idx])

        # Plot solutions
        ax.plot(t_full_ak, y_full_ak, color=cmap_ak[rec['plot_rank']],
                lw=1.4, label='allowedK' if row == 0 and col == 0 else '')
        ax.plot(t_full_ik, y_full_ik, color=cmap_ik[rec['plot_rank']],
                lw=1.4, ls='--',
                label='integerK' if row == 0 and col == 0 else '')

        # Mark FCB endpoints as filled dots
        ax.scatter([t_fcb_ak], [xi_ak[xi_field_idx]],
                   color=cmap_ak[rec['plot_rank']], zorder=6, s=40)
        ax.scatter([t_fcb_ik], [xi_ik[xi_field_idx]],
                   color=cmap_ik[rec['plot_rank']], zorder=6, s=40, marker='D')

        # Vertical line at FCB
        ax.axvline(t_fcb_ak, color='gray', lw=0.5, ls=':', alpha=0.6)

        # Horizontal reference at zero
        ax.axhline(0, color='k', lw=0.5, ls='-', alpha=0.3)

        # Column header (top row only)
        if row == 0:
            ax.set_title(field_label, fontsize=12)

        # Row annotation: k info on left column only
        if col == 0:
            ax.set_ylabel(
                f'mode {mode_idx}\n'
                f'$k_{{\\rm aK}}={ki_ak*conv*1e3:.3f}$\n'
                f'$k_{{\\rm iK}}={ki_ik*conv*1e3:.3f}$\n'
                f'$\\Delta k/k={dk_rel_i:.2f}\\%$\n'
                r'[$\times10^{-3}$ Mpc$^{-1}$]',
                fontsize=7.5, labelpad=2)

        # Bottom row: x-axis label
        if row == nrows - 1:
            ax.set_xlabel(r'Conformal time $\eta$', fontsize=9)

        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

        # Annotate FCB values for vr (col=0) to highlight asymmetry
        if col == 0:   # vr column
            ax.annotate(
                rf'$v_r^\infty$ aK$={xi_ak[2]:.2e}$''\n'
                rf'$v_r^\infty$ iK$={xi_ik[2]:.2e}$',
                xy=(0.02, 0.03), xycoords='axes fraction',
                fontsize=6.5, color='dimgray',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

# Global legend
handles = [
    plt.Line2D([0],[0], color='steelblue', lw=2, label='allowedK ($v_r^\\infty \\approx 0$, palindrome ✓)'),
    plt.Line2D([0],[0], color='firebrick', lw=2, ls='--', label='integerK ($v_r^\\infty \\neq 0$, palindrome ✗)'),
    plt.scatter([], [], color='steelblue', s=40, label='FCB endpoint (allowedK)'),
    plt.scatter([], [], color='firebrick', s=40, marker='D', label='FCB endpoint (integerK)'),
]
fig.legend(handles=handles, loc='upper right', fontsize=8.5,
           bbox_to_anchor=(1.0, 1.0), ncol=2,
           bbox_transform=fig.transFigure)

fig.suptitle(
    f'Part 2 — last {N_PLOT} high-$k$ modes: perturbation solutions (rec→FCB)\n'
    'Even at similar $k$, the small $\\Delta k$ shifts the FCB phase: '
    r'integerK $v_r^\infty \neq 0$ (not palindromic)',
    fontsize=11, y=1.01)

fig.tight_layout()
out2 = './figures/compare_perturbations_soln_highk.pdf'
plt.savefig(out2, bbox_inches='tight')
print(f"\nSaved: {out2}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2b: Zoom on vr near FCB — highlights the palindrome violation clearly
# ──────────────────────────────────────────────────────────────────────────────
# Show just the last 10% of conformal time (near FCB) for vr

fig, axes = plt.subplots(1, N_PLOT, figsize=(4*N_PLOT, 4), sharey=False)

for col, rec in enumerate(recs):
    ax = axes[col]
    t_ak_g = rec['t_ak']
    t_ik_g = rec['t_ik']
    xi_ak  = rec['xi_ak']
    xi_ik  = rec['xi_ik']
    ki_ak  = rec['ki_ak']
    ki_ik  = rec['ki_ik']

    t_fcb_ak = t_ak_g[-1] + deltaeta
    t_fcb_ik = t_ik_g[-1] + deltaeta

    # Zoom window: last 10% of t_grid
    t_min_zoom = t_ak_g[-1] - 0.10 * (t_ak_g[-1] - t_ak_g[0])
    mask_ak = t_ak_g >= t_min_zoom
    mask_ik = t_ik_g >= t_min_zoom

    t_zoom_ak = np.append(t_ak_g[mask_ak], t_fcb_ak)
    t_zoom_ik = np.append(t_ik_g[mask_ik], t_fcb_ik)
    vr_zoom_ak = np.append(rec['Y_ak'][4, mask_ak], xi_ak[2])
    vr_zoom_ik = np.append(rec['Y_ik'][4, mask_ik], xi_ik[2])

    ax.plot(t_zoom_ak, vr_zoom_ak, '-', color='steelblue', lw=1.8,
            label=f'allowedK  $v_r^\\infty={xi_ak[2]:.2e}$')
    ax.plot(t_zoom_ik, vr_zoom_ik, '--', color='firebrick', lw=1.8,
            label=f'integerK  $v_r^\\infty={xi_ik[2]:.2e}$')

    # FCB endpoints
    ax.scatter([t_fcb_ak], [xi_ak[2]], color='steelblue', zorder=6, s=60)
    ax.scatter([t_fcb_ik], [xi_ik[2]], color='firebrick',  zorder=6, s=60, marker='D')

    # Zero reference
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5, label='$v_r = 0$')
    ax.axvline(t_fcb_ak, color='gray', lw=0.6, ls=':', alpha=0.7)

    dk_rel_i = abs(ki_ak - ki_ik) / (0.5*(ki_ak + ki_ik)) * 100
    ax.set_title(
        f'mode {rec["mode_idx"]}\n'
        f'$k_{{\\rm aK}}={ki_ak*conv*1e3:.3f}$, '
        f'$k_{{\\rm iK}}={ki_ik*conv*1e3:.3f}$\n'
        f'$[10^{{-3}}$ Mpc$^{{-1}}]$, $\\Delta k/k={dk_rel_i:.2f}\\%$',
        fontsize=8)
    ax.set_xlabel(r'$\eta$ (near FCB)', fontsize=9)
    if col == 0:
        ax.set_ylabel(r'$v_r(\eta)$', fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

fig.suptitle(
    r'Near-FCB zoom of $v_r(\eta)$ for the last ' + f'{N_PLOT}' + r' high-$k$ modes''\n'
    r'allowedK: $v_r \to 0$ at FCB (palindrome) — '
    r'integerK: $v_r \not\to 0$ (NOT symmetric/anti-symmetric)',
    fontsize=11)
fig.tight_layout()
out3 = './figures/compare_vr_near_FCB.pdf'
plt.savefig(out3, bbox_inches='tight')
print(f"Saved: {out3}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2c: Overlay all N_PLOT modes for each field on one panel
#            (one-per-field figure to see the shape envelope differences)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax_flat_idx, (var_idx, field_label, fcb_label) in enumerate(FIELDS):
    ax = axes.flat[ax_flat_idx]
    xi_field_idx = XINF_FIELD_IDX[ax_flat_idx]

    for rec in recs:
        r = rec['plot_rank']
        t_fcb = rec['t_ak'][-1] + deltaeta

        t_full_ak = np.append(rec['t_ak'], t_fcb)
        t_full_ik = np.append(rec['t_ik'], t_fcb)
        y_full_ak = np.append(rec['Y_ak'][var_idx], rec['xi_ak'][xi_field_idx])
        y_full_ik = np.append(rec['Y_ik'][var_idx], rec['xi_ik'][xi_field_idx])

        lbl_ak = f'aK mode {rec["mode_idx"]}  ($v_r^\\infty={rec["xi_ak"][2]:.1e}$)'
        lbl_ik = f'iK mode {rec["mode_idx"]}  ($v_r^\\infty={rec["xi_ik"][2]:.1e}$)'

        ax.plot(t_full_ak, y_full_ak, '-',  color=cmap_ak[r], lw=1.3, label=lbl_ak)
        ax.plot(t_full_ik, y_full_ik, '--', color=cmap_ik[r], lw=1.3, label=lbl_ik)

        # FCB endpoints
        ax.scatter([t_fcb], [rec['xi_ak'][xi_field_idx]],
                   color=cmap_ak[r], zorder=6, s=30)
        ax.scatter([t_fcb], [rec['xi_ik'][xi_field_idx]],
                   color=cmap_ik[r], zorder=6, s=30, marker='D')

    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(rec['t_ak'][-1] + deltaeta, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlabel(r'Conformal time $\eta$', fontsize=9)
    ax.set_ylabel(field_label, fontsize=11)
    ax.set_title(f'{field_label}  (solid=allowedK, dashed=integerK)', fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.25)

fig.suptitle(
    f'Part 2 — last {N_PLOT} high-$k$ modes: all fields overlaid\n'
    r'allowedK (solid, blue shades) vs integerK (dashed, red shades). '
    r'Filled circle = FCB endpoint.  integerK $v_r^\infty \neq 0$.',
    fontsize=11)
fig.tight_layout()
out4 = './figures/compare_perturbations_overlay.pdf'
plt.savefig(out4, bbox_inches='tight')
print(f"Saved: {out4}")
plt.close()

print("\nDone.  Output figures:")
print(f"  {out1}   (convergence of k-values + vr^∞ comparison)")
print(f"  {out2}   (N_PLOT × 4 grid: all fields, mode-by-mode)")
print(f"  {out3}   (zoom on vr near FCB: palindrome violation)")
print(f"  {out4}   (all fields overlaid, all modes)")
